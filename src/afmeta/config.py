# src/afmeta/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal
import yaml
import json
import hashlib


@dataclass
class RunConfig:
    # Derivation inputs
    prot_name: Optional[str] = None
    ensemble_dir: Optional[Path] = None
    reference_dir: Optional[Path] = None
    out_base_dir: Optional[Path] = None
    descriptor: Optional[str] = None

    # Explicit paths 
    ensemble_pdb: Optional[Path] = None
    reference_pdb: Optional[Path] = None
    outdir: Optional[Path] = None
    solvated_dir: Optional[Path] = None  # keeps existing semantics


    # Feature/PCA
    top_m: int = 200
    min_seq_sep: int = 3
    periodic: bool = False
    n_components: int = 2
    random_state: int = 0
    label_prefix: str = "cc_r"

    #makes the model biased or not
    biased: bool = True

    # Metadynamics
    well_tempered: bool = True
    pace: int = 500
    height: float = 1.2
    energy_unit: str = "kJ/mol"  # or "kcal/mol", or "Ha" sometimes. must be one of these three. 
    biasfactor: float = 10.0
    suggest_metad_params: bool = False
    sigma: Tuple[float, float] = (0.1, 0.1)
    grid_min: Tuple[float, float] = (-5.0, -5.0)
    grid_max: Tuple[float, float] = (5.0, 5.0)
    grid_bin: Tuple[int, int] = (200, 200)
    temperature: float = 300.0
    stride: int = 500

    # OpenMM
    run_name: Optional[str] = None
    nsteps: int = 5_000_000
    dt_ps: float = 0.002
    friction_per_ps: float = 1.0
    outfreq_steps: int = 500
    minimize: bool = False
    barostat: bool = True
    pressure_bar: float = 1.0

    # GPU
    gpu: bool = True
    gpu_id: int = 0

    # W&B
    wandb: bool = False
    wandb_project: str = "aflow-metadynamics"
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_colvar: bool = False
    wandb_colvar_poll_s: float = 10.0


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict, got: {type(data)}")
    return data


def apply_overrides(cfg: RunConfig, overrides: Dict[str, Any]) -> RunConfig:
    valid = set(asdict(cfg).keys())
    unknown = [k for k in overrides.keys() if k not in valid]
    if unknown:
        raise ValueError(f"Unknown config keys in YAML: {unknown}")

    # fields that should become Path objects if provided as strings
    path_fields = {
        "ensemble_dir", "reference_dir", "out_base_dir",
        "ensemble_pdb", "reference_pdb", "outdir",
    }
    for k, v in overrides.items():
        if v is None:
            continue
        if k in path_fields:
            # allow already-Path or string
            setattr(cfg, k, Path(v))
        else:
            setattr(cfg, k, v)
    return cfg


def dump_yaml(cfg: RunConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))


def _hash_sweep(cfg) -> str:
    d = {
        "biased": bool(cfg.biased),
        "pace": cfg.pace,
        "height": cfg.height,
        "sigma": tuple(cfg.sigma) if cfg.sigma is not None else None,
        "biasfactor": cfg.biasfactor,
        "nsteps": cfg.nsteps,
    }
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]

def _fmt_nsteps(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"n{n//1_000_000}e6"
    if n % 1_000 == 0:
        return f"n{n//1_000}e3"
    return f"n{n}"

def auto_descriptor(cfg) -> str:
    h = _hash_sweep(cfg)
    ntag = _fmt_nsteps(cfg.nsteps)

    if bool(cfg.biased):
        s0, s1 = cfg.sigma
        core = f"p{cfg.pace}_h{cfg.height}_s{s0:.3g}-{s1:.3g}_bf{cfg.biasfactor}_{ntag}"
    else:
        core = f"{ntag}"

    return f"{core}__{h}"

def derive_paths_from_prot(cfg: RunConfig) -> RunConfig:
    if cfg.prot_name:
        if cfg.ensemble_pdb is None and cfg.ensemble_dir is not None:
            cfg.ensemble_pdb = Path(cfg.ensemble_dir) / f"{cfg.prot_name}.pdb"
        if cfg.reference_pdb is None and cfg.reference_dir is not None:
            cfg.reference_pdb = Path(cfg.reference_dir) / f"{cfg.prot_name}.pdb"

        desc = cfg.descriptor or auto_descriptor(cfg)

        if cfg.outdir is None and cfg.out_base_dir is not None:
            mode = "biased" if bool(cfg.biased) else "unbiased"
            date = datetime.now().strftime("%Y-%m-%d")
            cfg.outdir = Path(cfg.out_base_dir) / mode / cfg.prot_name / date / desc


    return cfg
