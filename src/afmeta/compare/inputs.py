# compare/inputs.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Literal

import mdtraj as md
import pandas as pd

Role = Literal["reference_md", "biased", "unbiased", "alphaflow"]
EnergyUnit = Literal["kJ", "kcal", "kBT"]


@dataclass(frozen=True)
class Dataset:
    role: Role
    label: str
    traj: md.Trajectory
    colvar: Optional[pd.DataFrame]
    meta: Dict[str, Any]


@dataclass
class CompareJobInputs:
    """
    reference_md, biased required.
    """
    out_dir: Path
    reference_md: Dataset
    biased: Dataset
    unbiased: Optional[Dataset] = None
    alphaflow: Optional[Dataset] = None

    # standardized versions produced by preprocess (optional)
    ref_std: Optional[md.Trajectory] = None
    biased_std: Optional[md.Trajectory] = None
    unbiased_std: Optional[md.Trajectory] = None
    alphaflow_std: Optional[md.Trajectory] = None

    # thermodynamic interpretation for rbias reweighting
    temperature_K: float = 300.0
    energy_unit: EnergyUnit = "kJ"
