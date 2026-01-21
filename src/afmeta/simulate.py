# src/afmeta/simulate.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import sys

# OpenMM + PLUMED
from openmmplumed import PlumedForce
from openmm import Platform, MonteCarloBarostat
from openmm.unit import (
    kelvin,
    picoseconds,
    nanometer,
    bar,
    kilojoule_per_mole,
)
from openmm.app import (
    Simulation,
    PDBFile,
    ForceField,
    PME,
    HBonds,
    StateDataReporter,
    CheckpointReporter,
)

# Reporters
from mdtraj.reporters import XTCReporter


@dataclass(frozen=True)
class SimulationOutputs:
    output_dir: Path
    xtc_path: Path
    state_csv_path: Path
    checkpoint_path: Optional[Path]
    final_pdb_path: Path


@dataclass(frozen=True)
class OpenMMSimConfig:
    # integrator
    dt_ps: float = 0.002
    temperature_K: float = 300.0
    friction_per_ps: float = 1.0

    # system
    nonbonded_cutoff_nm: float = 1.2
    switch_distance_nm: float = 1.0
    constraints: Any = HBonds

    # ensemble control
    use_barostat: bool = True
    pressure_bar: float = 1.0

    # run control
    nsteps: int = 5_000_000
    outfreq_steps: int = 500
    minimize: bool = True
    minimize_tolerance_kj_mol_nm: float = 10.0
    minimize_max_iters: int = 1000

    # checkpointing
    save_checkpoint: bool = True
    checkpoint_freq_steps: int = 500_000
    restart: bool = False  # load checkpoint if present

    # forcefield
    forcefield_files: tuple[str, ...] = ("charmm36.xml", "charmm36/water.xml")


def _select_platform(
    *,
    on_gpu: bool,
    gpu_id: int = 0,
    cuda_precision: str = "mixed",
    use_cpu_pme: bool = False,
) -> tuple[Any, Optional[Dict[str, str]]]:
    """
    Returns (platform, properties) for Simulation(...).
    """
    if on_gpu:
        platform = Platform.getPlatformByName("CUDA")
        props: Dict[str, str] = {
            "Precision": cuda_precision,            # "single" | "mixed" | "double"
            "UseCpuPme": "true" if use_cpu_pme else "false",
            "DeviceIndex": str(gpu_id),
        }
        return platform, props

    platform = Platform.getPlatformByName("CPU")
    return platform, None


def run_openmm(
    *,
    pdb_path: Path,
    outdir: Path,
    run_name: Optional[str] = None,
    plumed_path: Optional[Path] = None,
    biased: bool = True,
    on_gpu: bool = True,
    gpu_id: int = 0,
    config: OpenMMSimConfig = OpenMMSimConfig(),
) -> SimulationOutputs:
    """
    Run OpenMM simulation optionally biased with PLUMED.

    - If plumed_path is provided and biased=True, adds a PlumedForce from that file.
    - Writes:
        <run_name>.xtc
        <run_name>_state.csv
        <run_name>.chk (optional)
        <run_name>_final.pdb
    - Uses explicit paths (no chdir, no globals).

    Notes:
    - Assumes pdb_path already contains the system you want to simulate (e.g. fixed/solvated).
    - Feature remapping + PLUMED writing should be done before calling this.
    """
    pdb_path = Path(pdb_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = run_name or pdb_path.stem
    base = outdir / stem

    xtc_path = base.with_suffix(".xtc")
    state_csv_path = outdir / f"{stem}_state.csv"
    chk_path = base.with_suffix(".chk")  # always define a canonical checkpoint path
    final_pdb_path = outdir / f"{stem}_final.pdb"

    # Load system
    pdb = PDBFile(str(pdb_path))
    ff = ForceField(*config.forcefield_files)

    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=config.nonbonded_cutoff_nm * nanometer,
        switchDistance=config.switch_distance_nm * nanometer,
        constraints=config.constraints,
    )

    # Optional PLUMED bias
    if plumed_path is not None:
        plumed_text = Path(plumed_path).read_text()

        if biased and ("METAD" not in plumed_text):
            raise ValueError("biased=True but PLUMED file does not contain METAD.")
        if (not biased) and ("METAD" in plumed_text):
            raise ValueError("biased=False but PLUMED file contains METAD (would still bias).")

        system.addForce(PlumedForce(plumed_text))

    # Optional barostat (NPT)
    if config.use_barostat:
        system.addForce(MonteCarloBarostat(config.pressure_bar * bar, config.temperature_K * kelvin))

    # Integrator
    from openmm import LangevinIntegrator  # local import keeps module import clean

    integrator = LangevinIntegrator(
        config.temperature_K * kelvin,
        config.friction_per_ps / picoseconds,
        config.dt_ps * picoseconds,
    )

    # Platform
    platform, properties = _select_platform(on_gpu=on_gpu, gpu_id=gpu_id)
    if properties is None:
        simulation = Simulation(pdb.topology, system, integrator, platform)
    else:
        simulation = Simulation(pdb.topology, system, integrator, platform, properties)

    # Positions / restart
    simulation.context.setPositions(pdb.positions)

    # Reporters
    if config.save_checkpoint:
        simulation.reporters.append(CheckpointReporter(str(chk_path), config.checkpoint_freq_steps))

    if config.restart:
        if not chk_path.exists():
            raise FileNotFoundError(f"restart=True but checkpoint not found: {chk_path}")
        simulation.loadCheckpoint(str(chk_path))


    simulation.reporters.append(
        StateDataReporter(
            str(state_csv_path),
            config.outfreq_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            progress=True,
            remainingTime=True,
            speed=True,
            elapsedTime=True,
            totalSteps=config.nsteps,
        )
    )

    # Mirror to stdout (handy on clusters)
    simulation.reporters.append(
        StateDataReporter(
            sys.stdout,
            config.outfreq_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            progress=True,
            remainingTime=True,
            speed=True,
            elapsedTime=True,
            totalSteps=config.nsteps,
        )
    )

    simulation.reporters.append(XTCReporter(str(xtc_path), config.outfreq_steps))

    # Minimize then run
    if config.minimize and (not config.restart):
        simulation.minimizeEnergy(
            tolerance=config.minimize_tolerance_kj_mol_nm * kilojoule_per_mole / nanometer,
            maxIterations=config.minimize_max_iters,
        )

    simulation.step(config.nsteps)

    # Final snapshot
    pos = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    with open(final_pdb_path, "w") as fh:
        PDBFile.writeFile(simulation.topology, pos, fh, keepIds=True)

    return SimulationOutputs(
        output_dir=outdir,
        xtc_path=xtc_path,
        state_csv_path=state_csv_path,
        checkpoint_path=chk_path,
        final_pdb_path=final_pdb_path,
    )
