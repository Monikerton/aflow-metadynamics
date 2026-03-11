from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import mdtraj as md


def _rmsf_A_from_traj(traj: md.Trajectory, indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute per-atom RMSF (Å) for selected atom indices.

    If indices is None, uses all atoms in `traj`.
    Returns 1D array of length n_selected_atoms.
    """
    if traj.n_frames == 0:
        return np.array([], dtype=np.float32)
    if indices is None:
        # use all atoms
        xyz = traj.xyz  # (T, N, 3) in nm
        n_sel = traj.n_atoms
    else:
        xyz = traj.xyz[:, indices, :]
        n_sel = indices.size

    # convert to Å
    xyz_A = (xyz.astype(np.float32, copy=False) * 10.0)
    # compute mean position per atom
    mean_xyz = xyz_A.mean(axis=0)  # (n_sel,3)
    # fluctuations
    diffs = xyz_A - mean_xyz[None, ...]  # (T,n_sel,3)
    var = np.mean(np.sum(diffs * diffs, axis=2), axis=0)  # (n_sel,)
    rmsf = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    return rmsf


def compute(job, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # prefer standardized trajectories (fast, consistent ordering)
    ref_std = getattr(job, "ref_md_std", None)
    af_std = getattr(job, "alphaflow_std", None)
    bia_std = getattr(job, "biased_std", None)
    unb_std = getattr(job, "unbiased_std", None)

    # raw trajectories fallback
    ref_raw = getattr(job.reference_md, "traj", None)
    af_raw = getattr(job.alphaflow, "traj", None) if getattr(job, "alphaflow", None) is not None else None
    bia_raw = getattr(job.biased, "traj", None) if getattr(job, "biased", None) is not None else None
    unb_raw = getattr(job.unbiased, "traj", None) if getattr(job, "unbiased", None) is not None else None

    # choose preferred source
    if ref_std is not None:
        mode = "std"
        ref = ref_std
        af = af_std
        bia = bia_std
        unb = unb_std
        atom_selection = "std_atoms"
    elif ref_raw is not None:
        mode = "raw"
        ref = ref_raw
        af = af_raw
        bia = bia_raw
        unb = unb_raw
        # try to prefer CA atoms if present
        ca_sel = np.array(ref.topology.select("name CA"), dtype=int)
        atom_selection = "CA" if ca_sel.size > 0 else "all"
    else:
        raise ValueError("No reference trajectory found for RMSF (need ref_std or reference_md.traj)")

    results: Dict[str, Any] = {
        "metric": "rmsf",
        "params": {"mode": mode, "atom_selection": atom_selection},
        "scores": {},
        "artifacts": {},
        "warnings": [],
    }

    # ensure ref has frames
    if ref.n_frames == 0:
        raise ValueError("Reference trajectory has zero frames; cannot compute RMSF")

    # for raw mode, precompute CA indices if requested
    ca_indices = None
    if mode == "raw" and atom_selection == "CA":
        ca_indices = np.array(ref.topology.select("name CA"), dtype=int)

    # helper to compute for a single traj
    def _process(name: str, traj: Optional[md.Trajectory]):
        if traj is None:
            return None
        if traj.n_frames == 0:
            results["warnings"].append(f"Skipping {name}: zero frames")
            return None
        # if using std mode, require same atom counts
        if mode == "std" and traj.n_atoms != ref.n_atoms:
            results["warnings"].append(f"Skipping {name}: atom count mismatch with ref std ({traj.n_atoms} vs {ref.n_atoms})")
            return None

        # choose indices
        if mode == "std":
            indices = None  # use all atoms in std
            L = traj.n_atoms
        else:
            if atom_selection == "CA":
                if ca_indices is None or ca_indices.size == 0:
                    results["warnings"].append(f"Skipping {name}: no CA atoms found for raw mode")
                    return None
                indices = ca_indices
                L = indices.size
            else:
                indices = None
                L = traj.n_atoms

        arr = _rmsf_A_from_traj(traj, indices=indices)  # per-atom (or per-CA) RMSF in Å
        # basic stats
        if arr.size == 0:
            mean_A = float("nan")
            median_A = float("nan")
            p95_A = float("nan")
        else:
            mean_A = float(arr.mean())
            median_A = float(np.median(arr))
            p95_A = float(np.quantile(arr, 0.95))

        # save artifact
        fname = out_dir / f"{name}_rmsf_A.npy"
        np.save(fname, arr.astype(np.float32))
        results["artifacts"][f"{name}_rmsf_A"] = str(fname)

        return {
            "mean_rmsf_A": mean_A,
            "median_rmsf_A": median_A,
            "p95_rmsf_A": p95_A,
            "n_frames": int(traj.n_frames),
            "L": int(L),
        }

    for nm, tr in (("af", af), ("bia", bia), ("unb", unb), ("ref", ref)):
        try:
            sc = _process(nm, tr)
            if sc is not None:
                results["scores"][nm] = sc
        except Exception as e:
            results["warnings"].append(f"Failed {nm}: {e}")

    return results
