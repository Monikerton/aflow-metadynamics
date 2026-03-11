from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from scipy.linalg import sqrtm


"""
EMD - Earth Mover's Distance-like metrics for atomic distributions.
Compares per-atom positional distributions between reference and other trajectories.
Basically like the 'work' needed to morph one distribution into the other, per-atom.
"""

def _remove_hydrogens(traj):
    keep = [a.index for a in traj.top.atoms if (a.element is None or a.element.symbol != 'H')]
    return traj.atom_slice(keep, inplace=False)


def get_mean_covar(xyz: np.ndarray):
    mean = xyz.mean(0)
    xyz0 = xyz - mean
    covar = (xyz0[..., None] * xyz0[..., None, :]).mean(0)
    return mean, covar


def _sqrtm_dot(A, B):
    """Compute sqrtm(A @ B)."""
    return sqrtm(A @ B)


def compute(job, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer standardized CA trajectories — they are guaranteed to share the same
    # atom ordering and count across all ensembles. Fall back to full-atom only
    # when std trajectories are not available (they rarely share atom counts).
    ref_std = getattr(job, "ref_md_std", None)
    if ref_std is not None:
        ref_f = ref_std
        af_f  = getattr(job, "alphaflow_std", None)
        bia_f = getattr(job, "biased_std", None)
        unb_f = getattr(job, "unbiased_std", None)
    else:
        ref = job.reference_md.traj
        af = job.alphaflow.traj if job.alphaflow is not None else None
        bia = job.biased.traj if job.biased is not None else None
        unb = job.unbiased.traj if job.unbiased is not None else None
        ref_f = _remove_hydrogens(ref)
        af_f  = _remove_hydrogens(af)  if af  is not None else None
        bia_f = _remove_hydrogens(bia) if bia is not None else None
        unb_f = _remove_hydrogens(unb) if unb is not None else None

    results: Dict[str, Any] = {"metric": "atomic_emd"}

    def _per_atom_mean_dist(a_xyz, b_xyz):
        # a_xyz: (n_frames, n_atoms, 3) ; b_xyz: (m_frames, n_atoms, 3)
        if a_xyz.shape[1] != b_xyz.shape[1]:
            raise ValueError("Atom counts differ between inputs for per-atom mean distance")
        ma = a_xyz.mean(0)
        mb = b_xyz.mean(0)
        d = np.sqrt(((ma - mb) ** 2).sum(-1)) * 10.0
        return d.tolist()

    # mean distance per atom (ref vs af / biased / unbiased)
    if af_f is not None:
        results["emd_mean_ref_af_per_atom"] = _per_atom_mean_dist(ref_f.xyz, af_f.xyz)
    if bia_f is not None:
        results["emd_mean_ref_bia_per_atom"] = _per_atom_mean_dist(ref_f.xyz, bia_f.xyz)
    if unb_f is not None:
        results["emd_mean_ref_unb_per_atom"] = _per_atom_mean_dist(ref_f.xyz, unb_f.xyz)

    # variance EMD-like scalar using covariance matrices
    def _emd_var(a_xyz, b_xyz):
        if a_xyz.shape[1] != b_xyz.shape[1]:
            raise ValueError("Atom counts differ between inputs for covariance EMD computation")
        ma, Ca = get_mean_covar(a_xyz)
        mb, Cb = get_mean_covar(b_xyz)
        # compute trace(Ca + Cb - 2 * sqrt(Ca @ Cb)) per-atom block
        try:
            S = _sqrtm_dot(Ca, Cb)
            tr = np.trace(Ca + Cb - 2 * S, axis1=1, axis2=2)
            out = np.sqrt(np.maximum(tr, 0.0)) * 10.0
            return out.tolist()
        except Exception:
            # fallback: use sqrt(trace(Ca)) as rough estimate
            tr_ca = np.trace(Ca, axis1=1, axis2=2)
            return (np.sqrt(np.maximum(tr_ca, 0.0)) * 10.0).tolist()

    if af_f is not None:
        results["emd_var_ref_af_per_atom"] = _emd_var(ref_f.xyz, af_f.xyz)
    if bia_f is not None:
        results["emd_var_ref_bia_per_atom"] = _emd_var(ref_f.xyz, bia_f.xyz)
    if unb_f is not None:
        results["emd_var_ref_unb_per_atom"] = _emd_var(ref_f.xyz, unb_f.xyz)

    return results
