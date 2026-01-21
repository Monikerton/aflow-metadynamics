from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any

import mdtraj as md
import pandas as pd

from .inputs import Dataset


def read_plumed_colvar(path: Path) -> pd.DataFrame:
    with path.open("r") as f:
        for line in f:
            if line.startswith("#! FIELDS"):
                fields = line.strip().split()[2:]
                break
        else:
            raise ValueError(f"Missing '#! FIELDS' header in COLVAR: {path}")

    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None).dropna(how="all")
    df.columns = fields

    required = {"time", "metad.rbias"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"COLVAR missing required columns {sorted(missing)}. Have: {list(df.columns)}")

    return df

def load_reference_md(*, ref_top: Path, ref_traj: Path, label: str = "reference_md") -> Dataset:
    ref_top = ref_top.resolve()
    ref_traj = ref_traj.resolve()

    traj = md.load(str(ref_traj), top=str(ref_top))
    meta: Dict[str, Any] = {
        "ref_top": str(ref_top),
        "ref_traj": str(ref_traj),
    }
    return Dataset(role="reference_md", label=label, traj=traj, colvar=None, meta=meta)


def load_run_dir(*, run_dir: Path, kind: str, label: Optional[str] = None) -> Dataset:
    """
    kind: "biased" or "unbiased"
    Canonical filenames (NO GUESSING):
      - topology: fixed_final.pdb
      - traj: fixed.xtc
      - colvar: COLVAR_biased.dat or COLVAR_unbiased.dat
    """
    if kind not in {"biased", "unbiased"}:
        raise ValueError(f"kind must be 'biased' or 'unbiased', got {kind!r}")

    run_dir = run_dir.resolve()
    top = run_dir / "fixed_final.pdb"
    xtc = run_dir / "fixed.xtc"
    colvar = run_dir / f"COLVAR_{kind}.dat"

    if not top.exists():
        raise FileNotFoundError(f"Missing topology: {top}")
    if not xtc.exists():
        raise FileNotFoundError(f"Missing trajectory: {xtc}")
    if not colvar.exists():
        # allow missing colvar for some workflows; but for biased this is typically required
        colvar_df = None
    else:
        colvar_df = read_plumed_colvar(colvar)

    traj = md.load(str(xtc), top=str(top))
    meta: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "topology": str(top),
        "trajectory": str(xtc),
        "colvar": str(colvar) if colvar.exists() else None,
        "kind": kind,
    }
    return Dataset(role=kind, label=label or run_dir.name, traj=traj, colvar=colvar_df, meta=meta)


def load_alphaflow_ensemble(*, pdb_path: Path, label: str = "alphaflow") -> Dataset:
    """
    Alphaflow ensemble PDB (multi-model). No COLVAR.
    """
    pdb_path = pdb_path.resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"Missing alphaflow pdb: {pdb_path}")

    traj = md.load(str(pdb_path))  # topology included in PDB
    meta: Dict[str, Any] = {"alphaflow_pdb": str(pdb_path)}
    return Dataset(role="alphaflow", label=label, traj=traj, colvar=None, meta=meta)
