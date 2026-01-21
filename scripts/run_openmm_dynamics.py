#!/usr/bin/env python3
"""
scripts/run_openmm_dynamics.py

End-to-end driver to:
1) Load an ensemble trajectory (multi-model PDB) with MDTraj
2) Build distance features, select top-M by variance, fit PCA(2)
3) Remap those feature atom indices from reference pdb (first model in the ensemble) -> fixed
4) Write a PLUMED input that computes PC1/PC2 from those distances and runs METAD
5) Write params.json and create empty HILLS / COLVAR_biased.dat placeholders
6) (optional) Initialize W&B
7) (optional) Stream COLVAR to W&B
8) Run OpenMM (+ PLUMED)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import mdtraj as md

from src.afmeta.protonate import fix_pdb
from src.afmeta.remap import remap_feature_defs
from src.afmeta.features import prepare_pca_features, suggest_metad_params_from_pca
from src.afmeta.plumed_writer import write_plumed_pca

from src.afmeta.simulate import OpenMMSimConfig, run_openmm
from src.afmeta.wandb_utils import (
    wandb_init,
    wandb_log,
    wandb_finish,
    start_wandb_colvar_stream,
)
from src.afmeta.config import RunConfig, load_yaml_config, apply_overrides, dump_yaml, auto_descriptor, derive_paths_from_prot 
from argparse import BooleanOptionalAction
 



def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_tuple2_float(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated floats, e.g. '0.1,0.1'")
    return float(parts[0]), float(parts[1])


def _parse_tuple2_int(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated ints, e.g. '200,200'")
    return int(parts[0]), int(parts[1])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare PCA CVs, write PLUMED, and run OpenMM metadynamics.")
    p.add_argument("--config", type=Path, default="configs/run_openmm_dynamics.yaml", help="YAML config file")
    
    p.add_argument("--ensemble_pdb", type=Path, help="Multi-model PDB used for PCA training.")
    p.add_argument("--outdir", type=Path, help="Output directory.")
    p.add_argument("--top_m", type=int, help="Top-M distance features by variance.")
    p.add_argument("--min_seq_sep", type=int, help="Minimum residue index separation for CA-CA pairs.")
    p.add_argument("--periodic", action=BooleanOptionalAction, default=None)
    p.add_argument("--biased", action=BooleanOptionalAction, default=None)

    # Metadynamics knobs
    p.add_argument("--well_tempered", action=BooleanOptionalAction, default=None)
    p.add_argument("--pace", type=int)
    p.add_argument("--height", type=float)
    p.add_argument("--biasfactor", type=float)
    p.add_argument("--sigma", type=_parse_tuple2_float, help="e.g. '0.1,0.1'")
    p.add_argument("--grid_min", type=_parse_tuple2_float, help="e.g. '-5,-5'")
    p.add_argument("--grid_max", type=_parse_tuple2_float, help="e.g. '5,5'")
    p.add_argument("--grid_bin", type=_parse_tuple2_int, help="e.g. '200,200'")
    p.add_argument("--temperature", type=float)
    p.add_argument("--stride", type=int)

    # PCA knobs
    p.add_argument("--n_components", type=int)
    p.add_argument("--random_state", type=int)
    p.add_argument("--label_prefix", type=str)

    # Canonical topology (pre-fix)
    p.add_argument("--reference_pdb", type=Path)

    # OpenMM run knobs (minimal)
    p.add_argument("--run_name", type=str, help="Optional output stem for simulation outputs.")
    p.add_argument("--nsteps", type=int)
    p.add_argument("--dt_ps", type=float)
    p.add_argument("--friction_per_ps", type=float)
    p.add_argument("--outfreq_steps", type=int)
    p.add_argument("--minimize", action=BooleanOptionalAction, default=None)
    p.add_argument("--barostat", action=BooleanOptionalAction, default=None)
    p.add_argument("--pressure_bar", type=float)

    p.add_argument("--gpu", action=BooleanOptionalAction, default=None)
    p.add_argument("--gpu_id", type=int)

    # W&B
    p.add_argument("--wandb", action=BooleanOptionalAction, default=None)
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_name", type=str)
    p.add_argument("--wandb_group", type=str)

    # Stream COLVAR -> W&B (only relevant if --wandb)
    p.add_argument("--wandb_colvar", action=BooleanOptionalAction, default=None)
    p.add_argument("--wandb_colvar_poll_s", type=float)

    return p

def build_config(args) -> RunConfig:
    cfg = RunConfig()

    # 1) YAML overrides
    if args.config is not None:
        y = load_yaml_config(args.config)
        cfg = apply_overrides(cfg, y)


    # 2) CLI overrides (only those user explicitly provided)
    #    argparse gives None for unset values if you set default=None
    cli = vars(args)

    # booleans from store_true need careful handling:
    # - if user passes --metad_from_pca, set True (override YAML)
    # - if they don't pass it, leave YAML value unchanged
    # We'll detect "passed" booleans by checking the raw True, and otherwise not overriding.
    bool_flags = {"suggest_metad_params", "periodic", "biased"}

    overrides = {}
    for k, v in cli.items():
        if k == "config":
            continue

        if k in bool_flags:
            if v is not None:
                overrides[k] = v  # True or False
            continue

        if v is not None:
            overrides[k] = v

    cfg = apply_overrides(cfg, overrides)

    cfg = derive_paths_from_prot(cfg)

    # validate required
    if not cfg.ensemble_pdb or not cfg.reference_pdb:
        raise ValueError("ensemble_pdb and reference_pdb must be set via YAML or CLI.")

    return cfg




def main() -> None:
    args = build_argparser().parse_args()
    cfg = build_config(args)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mode = "biased" if bool(cfg.biased) else "unbiased"
    plumed_path = outdir / f"plumed_{mode}.dat"
    params_path = outdir / "params.json"
    hills_path = (outdir / "HILLS") if cfg.biased else None
    colvar_path = outdir / f"COLVAR_{mode}.dat"

    print(f"[info] Using ensemble PDB: {cfg.prot_name}")
    print(f"RUNNING FOR {cfg.nsteps * cfg.dt_ps / 1000} nanoseconds.")
    print(f"Mode is set to {cfg.biased}.")
    time.sleep(3)  # give user a moment to cancel if nstep is too high

    # 1) Load ensemble
    traj = md.load(str(cfg.ensemble_pdb))

    # 2) Fit PCA on distance features (FeatureDef indices initially live in ENSEMBLE topology)
    res = prepare_pca_features(
        traj,
        top_m=cfg.top_m,
        min_seq_sep=cfg.min_seq_sep,
        periodic=bool(cfg.periodic),
        n_components=cfg.n_components,
        random_state=cfg.random_state,
        label_prefix=cfg.label_prefix,
    )

    metad_suggestion = None
    if getattr(cfg, "suggest_metad_params", False):
        metad_suggestion = suggest_metad_params_from_pca(res.features, res.model)
        print("[suggest] METAD params from PCA fluctuations:")
        print(f"          sigma    = {metad_suggestion['sigma']}")
        print(f"          grid_min = {metad_suggestion['grid_min']}")
        print(f"          grid_max = {metad_suggestion['grid_max']}")
        print(f"          grid_bin = {metad_suggestion['grid_bin']}")


    if getattr(cfg, "suggest_metad_params", False):
        cfg.sigma = metad_suggestion["sigma"]
        cfg.grid_min = metad_suggestion["grid_min"]
        cfg.grid_max = metad_suggestion["grid_max"]
        cfg.grid_bin = metad_suggestion["grid_bin"]

    # Fix/protonate the reference PDB (for simulation)
    fixed_pdb = outdir / "fixed.pdb"
    fix_pdb(cfg.reference_pdb, fixed_pdb, Path(cfg.solvated_dir))

    # Remap feature defs from REFERENCE -> FIXED
    feature_defs_fixed = remap_feature_defs(
        res.feature_defs,
        old_pdb=cfg.reference_pdb,
        new_pdb=fixed_pdb,
    )

    print(f"[info] Prepared {len(feature_defs_fixed)} features for PLUMED from {res.full_features_count} total.")
    print("original feature defs", res.feature_defs[:5])
    print("remapped feature defs", feature_defs_fixed[:5])


    # Write PLUMED
    write_plumed_pca(
        plumed_path,
        res.model,
        feature_defs_fixed,
        periodic=bool(cfg.periodic),
        energy_unit=cfg.energy_unit,
        include_metad=bool(cfg.biased),
        well_tempered=bool(cfg.well_tempered),
        pace=cfg.pace,
        height=cfg.height,
        biasfactor=cfg.biasfactor,
        sigma=cfg.sigma,
        grid_min=cfg.grid_min,
        grid_max=cfg.grid_max,
        grid_bin=cfg.grid_bin,
        temperature=cfg.temperature,
        stride=cfg.stride,
        hills_path=hills_path,
        colvar_path=colvar_path,
    )

    # # 8) Record params (before run)
    payload: Dict[str, Any] = {
        "inputs": {
            "ensemble_pdb": str(cfg.ensemble_pdb),
            "reference_pdb": str(cfg.reference_pdb),
            "fixed_pdb": str(fixed_pdb),
        },
        "features": {
            "kind": res.kind,
            "top_m": cfg.top_m,
            "min_seq_sep": cfg.min_seq_sep,
            "periodic": bool(cfg.periodic),
            "n_features_total": res.full_features_count,
            "n_features_selected": len(feature_defs_fixed),
            "labels_preview": res.labels[:5],
            "remap_chain": "ensemble -> reference -> fixed",
        },
        "pca": {
            "n_components": cfg.n_components,
            "random_state": cfg.random_state,
            "explained_variance_ratio": getattr(res.model, "explained_variance_ratio_", None).tolist()
            if hasattr(res.model, "explained_variance_ratio_")
            else None,
        },
        "plumed": {
            "plumed_input": str(plumed_path),
            "hills": str(hills_path) if hills_path is not None else None,
            "colvar": str(colvar_path),
            "well_tempered": bool(cfg.well_tempered),
            "pace": cfg.pace,
            "height": cfg.height,
            "biasfactor": cfg.biasfactor if cfg.well_tempered else None,
            "suggested_from_pca": metad_suggestion,
            "sigma": list(cfg.sigma),
            "grid_min": list(cfg.grid_min),
            "grid_max": list(cfg.grid_max),
            "grid_bin": list(cfg.grid_bin),
            "temperature": cfg.temperature if cfg.well_tempered else None,
            "stride": cfg.stride,
        },
    }
    _write_json(params_path, payload)

    # 9) W&B init
    run = wandb_init(
        project=cfg.wandb_project,
        outdir=outdir,
        config=payload,
        name=cfg.wandb_name,
        group=cfg.wandb_group,
        enabled=bool(cfg.wandb),
    )

    # 10) Optional COLVAR -> W&B stream
    colvar_stop = None
    if bool(cfg.wandb) and bool(cfg.wandb_colvar):
        colvar_stop = start_wandb_colvar_stream(
            run=run,
            colvar_path=colvar_path,
            stride_steps=cfg.stride,
            poll_seconds=cfg.wandb_colvar_poll_s,
            prefix="colvar",
        )

    # 11) Run OpenMM (+ PLUMED)
    sim_cfg = OpenMMSimConfig(
        dt_ps=cfg.dt_ps,
        temperature_K=cfg.temperature,
        friction_per_ps=cfg.friction_per_ps,
        nsteps=cfg.nsteps,
        outfreq_steps=cfg.outfreq_steps,
        minimize=bool(cfg.minimize),
        use_barostat=cfg.barostat,
        pressure_bar=cfg.pressure_bar,
    )

    try:
        outs = run_openmm(
            pdb_path=fixed_pdb,
            outdir=outdir,
            run_name=cfg.run_name,
            plumed_path=plumed_path,
            biased=cfg.biased,
            on_gpu=bool(cfg.gpu),
            gpu_id=cfg.gpu_id,
            config=sim_cfg,
        )
        wandb_log(
            run,
            {
                "outputs/xtc": str(outs.xtc_path),
                "outputs/state_csv": str(outs.state_csv_path),
                "outputs/final_pdb": str(outs.final_pdb_path),
            },
        )
    finally:
        if colvar_stop is not None:
            colvar_stop.set()
        wandb_finish(run)

    print(f"[ok] Wrote: {plumed_path}")
    print(f"[ok] Wrote: {params_path}")
    print(f"[ok] Touched: {hills_path}")
    print(f"[ok] Touched: {colvar_path}")
    print(f"[ok] Ran OpenMM; outputs in: {outdir}")


if __name__ == "__main__":
    main()
