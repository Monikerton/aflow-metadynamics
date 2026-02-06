from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Dict, Any
import mdtraj as md

from .inputs import CompareJobInputs
from .io import load_reference_md, load_run_dir, load_alphaflow_ensemble
from .preprocess import standardize_to_reference
from .align import slice_to_common_ca
from .registry import METRICS, PLOTTERS


def run_compare_job(
    *,
    out_dir: Path,
    ref_crystal_path: Path,
    ref_md_top: Path,
    ref_md_traj: Path,
    biased_run_dir: Path,
    unbiased_run_dir: Optional[Path] = None,
    alphaflow_pdb: Optional[Path] = None,
    metrics: Optional[Iterable[str]] = None,
    plotters: Optional[Iterable[str]] = None,
    atom_sel: str = "protein and name CA",
    superpose: bool = True,
    temperature_K: float = 300.0,
    energy_unit: str = "kJ",
) -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    ref_crystal = md.load(str(ref_crystal_path)) # a trajectory with 1 frame
    ref_md = load_reference_md(ref_top=ref_md_top, ref_traj=ref_md_traj)
    biased = load_run_dir(run_dir=biased_run_dir, kind="biased", label="biased")
    unbiased = load_run_dir(run_dir=unbiased_run_dir, kind="unbiased", label="unbiased") if unbiased_run_dir else None
    alphaflow = load_alphaflow_ensemble(pdb_path=alphaflow_pdb) if alphaflow_pdb else None

    # Standardize everything into reference space
    print("Slicing all datasets to common CA set...")
    ref_crystal_ca, sliced, keys = slice_to_common_ca(
        ref=ref_crystal,
        others=[
            ("ref_md", ref_md.traj),
            ("biased", biased.traj),
            ("unbiased", unbiased.traj if unbiased else None),
            ("alphaflow", alphaflow.traj if alphaflow else None),
        ],
    )

    def _n_ca(t): return sum(a.name == "CA" for a in t.topology.atoms)

    kept = len(keys)
    print(f"[align] kept={kept}  ref={_n_ca(ref_crystal)}"  
        f"  ref_md={_n_ca(ref_md.traj)}"
        f"  biased={_n_ca(biased.traj)}"
        f"{'' if not unbiased else f'  unb={_n_ca(unbiased.traj)}'}"
        f"{'' if not alphaflow else f'  af={_n_ca(alphaflow.traj)}'}")
    print(f"[align] keys: {keys[0]} ... {keys[-1]}")

    ref_md_ca = sliced["ref_md"]
    biased_ca = sliced["biased"]
    unbiased_ca = sliced["unbiased"]
    alphaflow_ca = sliced["alphaflow"]

    # Now standardize in the sliced CA space
    print("Standardizing to reference space...")
    ref_md_std = standardize_to_reference(ref_md_ca, ref_crystal_ca, atom_sel="name CA", superpose=superpose)
    biased_std = standardize_to_reference(biased_ca, ref_crystal_ca, atom_sel="name CA", superpose=superpose)
    unbiased_std = standardize_to_reference(unbiased_ca, ref_crystal_ca, atom_sel="name CA", superpose=superpose) if unbiased_ca else None
    alphaflow_std = standardize_to_reference(alphaflow_ca, ref_crystal_ca, atom_sel="name CA", superpose=superpose) if alphaflow_ca else None

    job = CompareJobInputs(
        out_dir=out_dir,
        ref_crystal_path=ref_crystal_path,
        ref_crystal=ref_crystal,
        reference_md=ref_md,
        biased=biased,
        unbiased=unbiased,
        alphaflow=alphaflow,
        ref_md_std=ref_md_std,
        biased_std=biased_std,
        unbiased_std=unbiased_std,
        alphaflow_std=alphaflow_std,
        temperature_K=float(temperature_K),
        energy_unit=energy_unit,
    )

    metric_names = list(metrics) if metrics is not None else ["pca_compare"]
    print(f"Running metrics: {metric_names}")

    metrics_root = out_dir / "metrics"
    metrics_root.mkdir(exist_ok=True)

    results: Dict[str, Any] = {}
    for name in metric_names:
        print(f"Computing metric '{name}'...")
        if name not in METRICS:
            raise ValueError(f"Unknown metric '{name}'. Available: {sorted(METRICS)}")
        
        m_out = metrics_root / name
        m_out.mkdir(parents=True, exist_ok=True)
        res = METRICS[name](job, m_out)
        results[name] = res

    plotter_names = list(plotters) if plotters is not None else ["pca_compare"]
    print(f"Generating plots: {plotter_names}")

    plots_root = out_dir / "plots"
    plots_root.mkdir(exist_ok=True)

    for name in plotter_names:
        print(f"Generating plot '{name}'..." )
        if name not in PLOTTERS:
            raise ValueError(f"Unknown metric '{name}'. Available: {sorted(PLOTTERS)}")
        
        p_out = plots_root / name
        p_out.mkdir(parents=True, exist_ok=True)
        plot_path = PLOTTERS[name](results[name], p_out)
        results[name]["plot_path"] = str(plot_path)


    meta = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ref_crystal_path": str(Path(ref_crystal_path).resolve()),
        "ref_top": str(Path(ref_md_top).resolve()),
        "ref_traj": str(Path(ref_md_traj).resolve()),
        "biased_run_dir": str(Path(biased_run_dir).resolve()),
        "unbiased_run_dir": str(Path(unbiased_run_dir).resolve()) if unbiased_run_dir else None,
        "alphaflow_pdb": str(Path(alphaflow_pdb).resolve()) if alphaflow_pdb else None,
        "atom_sel": atom_sel,
        "superpose": superpose,
        "metrics": metric_names,
        "results": results,
        "temperature_K": temperature_K,
        "energy_unit": energy_unit,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir
