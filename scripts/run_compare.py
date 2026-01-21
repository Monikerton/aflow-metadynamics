from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any, Dict, Tuple
import warnings

import yaml

from src.afmeta.config import load_yaml_config
from src.afmeta.compare.runner import run_compare_job


def _must_exist(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{label} does not exist: {p}")


def build_compare_config(args: argparse.Namespace) -> Dict[str, Any]:
    # Start with YAML if provided
    cfg: Dict[str, Any] = {}
    if args.config is not None:
        cfg = load_yaml_config(args.config)

    # CLI overrides (only if user provided them)
    # NOTE: set CLI defaults to None so we can detect "not provided"
    if args.out is not None:
        cfg["out_dir"] = str(args.out)

    if args.ref_top is not None or args.ref_traj is not None:
        cfg.setdefault("reference_md", {})
        if args.ref_top is not None:
            cfg["reference_md"]["top"] = str(args.ref_top)
        if args.ref_traj is not None:
            cfg["reference_md"]["traj"] = str(args.ref_traj)

    if args.biased_run is not None:
        cfg["biased_run_dir"] = str(args.biased_run)
    if args.unbiased_run is not None:
        cfg["unbiased_run_dir"] = str(args.unbiased_run)
    if args.alphaflow_pdb is not None:
        cfg["alphaflow_pdb"] = str(args.alphaflow_pdb)

    # Important edge case: nargs="*" can produce [] if user passes "--metrics" with no values.
    # In that case, DO NOT override YAML / defaults.
    if args.metrics is not None:
        if len(args.metrics) > 0:
            cfg["metrics"] = list(args.metrics)
        # else: leave cfg["metrics"] untouched

    if args.atom_sel is not None:
        cfg["atom_sel"] = args.atom_sel
    if args.superpose is not None:
        cfg["superpose"] = args.superpose  # already bool from BooleanOptionalAction

    # Defaults
    cfg.setdefault("atom_sel", "protein and name CA")
    cfg.setdefault("superpose", True)

    # Validate required keys
    if "out_dir" not in cfg:
        raise ValueError("Missing out_dir (set --out or out_dir in YAML).")
    if (
        "reference_md" not in cfg
        or "top" not in cfg["reference_md"]
        or "traj" not in cfg["reference_md"]
    ):
        raise ValueError(
            "Missing reference_md.top/traj (set --ref-top/--ref-traj or in YAML)."
        )
    if "biased_run_dir" not in cfg:
        raise ValueError("Missing biased_run_dir (set --biased-run or in YAML).")

    # Friendly warning.
    if not cfg.get("unbiased_run_dir") and not cfg.get("alphaflow_pdb"):
        warnings.warn(
            "Neither unbiased_run_dir nor alphaflow_pdb provided. "
            "Will compare biased vs reference-only metrics (where applicable)."
        )

    return cfg


def build_compare_kwargs(cfg: Dict[str, Any]) -> dict:
    # Convert config to kwargs for run_compare_job
    return dict(
        out_dir=Path(cfg["out_dir"]),
        ref_top=Path(cfg["reference_md"]["top"]),
        ref_traj=Path(cfg["reference_md"]["traj"]),
        biased_run_dir=Path(cfg["biased_run_dir"]),
        unbiased_run_dir=Path(cfg["unbiased_run_dir"]) if cfg.get("unbiased_run_dir") else None,
        alphaflow_pdb=Path(cfg["alphaflow_pdb"]) if cfg.get("alphaflow_pdb") else None,
        metrics=cfg.get("metrics", None),
        atom_sel=cfg.get("atom_sel", "protein and name CA"),
        superpose=bool(cfg.get("superpose", True)),
    )


def write_effective_config(out_dir: Path, cfg: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "config.effective.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_compare_pca.yaml", type=Path)

    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ref-top", dest="ref_top", type=Path, default=None)
    ap.add_argument("--ref-traj", dest="ref_traj", type=Path, default=None)
    ap.add_argument("--biased-run", dest="biased_run", type=Path, default=None)
    ap.add_argument("--unbiased-run", dest="unbiased_run", type=Path, default=None)
    ap.add_argument("--alphaflow-pdb", dest="alphaflow_pdb", type=Path, default=None)

    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--atom-sel", dest="atom_sel", default=None)

    # default None means "don't override YAML"
    ap.add_argument(
        "--superpose",
        dest="superpose",
        action=BooleanOptionalAction,
        default=None,
        help="Superpose trajectories to reference frame 0 (use --no-superpose to disable).",
    )

    args = ap.parse_args()

    cfg = build_compare_config(args)
    kwargs = build_compare_kwargs(cfg)

    # Check for errors
    _must_exist(kwargs["ref_top"], "reference_md.top")
    _must_exist(kwargs["ref_traj"], "reference_md.traj")
    _must_exist(kwargs["biased_run_dir"], "biased_run_dir")
    if kwargs["unbiased_run_dir"] is not None:
        _must_exist(kwargs["unbiased_run_dir"], "unbiased_run_dir")
    if kwargs["alphaflow_pdb"] is not None:
        _must_exist(kwargs["alphaflow_pdb"], "alphaflow_pdb")

    eff = write_effective_config(kwargs["out_dir"], cfg)

    print("Running the compare job with config", args.config)
    out = run_compare_job(**kwargs)
    print(f"Wrote compare outputs to: {out}")
    print(f"Wrote effective config to: {eff}")


if __name__ == "__main__":
    main()
