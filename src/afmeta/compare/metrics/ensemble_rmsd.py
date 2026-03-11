from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.decomposition import PCA


"""
Ensemble RMSD and PCA cosine similarity metrics.
Compares structural ensembles via pairwise RMSD statistics and PCA of conformational space.
Basically captures the distances between two specific structures.

Cosine similarity measures directional distance, so checks if conformational changes are aligned in direction.
"""

def _flat(traj) -> np.ndarray:
    return traj.xyz.reshape(traj.n_frames, -1)


def get_rmsds(traj1, traj2, broadcast: bool = True):
    """
    Compute fixed-frame RMSD between two sets of structures.

    Notes
    -----
    - This computes RMSD in the current coordinate frame (no optimal per-pair
      superposition). If you need per-pair superposition, use a Kabsch/mdtraj
      routine explicitly.

    Parameters
    ----------
    traj1, traj2 : array-like
        Trajectories with shape (n_frames, n_atoms, 3). The first dimension is
        the number of frames. THIS FUNCTION ASSUMES THE ATOMS ARE ORDERED THE
        SAME WAY IN BOTH TRAJECTORIES.
    broadcast : bool
        If True, compute an all-vs-all matrix of shape (n_frames1, n_frames2).
        If False, compute per-frame RMSD between corresponding frames; in that
        case `traj1` and `traj2` must have the same number of frames.
    """

    if not broadcast and traj1.shape[0] != traj2.shape[0]:
        raise ValueError("traj1 and traj2 must have the same number of frames when broadcast=False")

    if traj1.shape[1] != traj2.shape[1]:
        raise ValueError("traj1 and traj2 must have the same number of atoms (shape[1])")

    n_atoms = traj1.shape[1]
    t1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    t2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if broadcast:
        t1, t2 = t1[:, None, :], t2[None, :, :]

    # root-mean-square per atom, converted to Angstroms (mdtraj uses nm)
    distmat = np.sqrt(np.square(t1 - t2).sum(-1)) / (n_atoms ** 0.5) * 10.0
    return distmat


def compute(job, out_dir: Path) -> Dict[str, Any]:
    """Compute ensemble RMSD and PCA similarity metrics (CA-space).

    Notes
    -----
    - RMSDs are computed in a fixed (already-standardized) coordinate frame
      and are returned in Angstroms (suffix `_A`). If you need per-pair
      optimal superposition RMSDs, compute them separately.

    Returns
    -------
    Dict[str, Any]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = job.ref_md_std
    bia = job.biased_std
    unb = job.unbiased_std
    af = job.alphaflow_std

    if ref is None:
        raise ValueError("Reference standardized trajectory (ref_std) required for ensemble_rmsd")

    results: Dict[str, Any] = {"metric": "ensemble_rmsd"}

    rng = np.random.default_rng(137)

    # sample reference frames (avoid O(N^2) blowups)
    n_ref = ref.n_frames
    n_sample = min(2000, n_ref)
    if n_ref > n_sample:
        ref_idx = rng.choice(n_ref, size=n_sample, replace=False)
    else:
        ref_idx = np.arange(n_ref)

    # within-reference pairwise (upper triangle excluding diagonal)
    M_ref = get_rmsds(ref.xyz[ref_idx], ref.xyz[ref_idx], broadcast=True)
    i, j = np.triu_indices(M_ref.shape[0], k=1)
    if i.size > 0:
        vals = M_ref[i, j]
        results["ref_within_mean_pairwise_rmsd_A"] = float(vals.mean())
        results["ref_within_rms_pairwise_rmsd_A"] = float(np.sqrt(np.square(vals).mean()))
    else:
        results["ref_within_mean_pairwise_rmsd_A"] = 0.0
        results["ref_within_rms_pairwise_rmsd_A"] = 0.0

    # helper to sample other ensembles up to n_sample frames
    def _sample_traj(traj, n):
        if traj is None:
            return None, None
        m = traj.n_frames
        k = min(n, m)
        idx = rng.choice(m, size=k, replace=False) if m >= k else rng.choice(m, size=k, replace=True)
        return traj, idx

    # compute pairwise stats for each ensemble (sampled)
    def _ensemble_stats(name, traj):
        t, idx = _sample_traj(traj, n_sample)
        if t is None:
            return
        M = get_rmsds(t.xyz[idx], t.xyz[idx], broadcast=True)
        i, j = np.triu_indices(M.shape[0], k=1)
        if i.size > 0:
            vals = M[i, j]
            results[f"{name}_within_mean_pairwise_rmsd_A"] = float(vals.mean())
            results[f"{name}_within_rms_pairwise_rmsd_A"] = float(np.sqrt(np.square(vals).mean()))
        else:
            results[f"{name}_within_mean_pairwise_rmsd_A"] = 0.0
            results[f"{name}_within_rms_pairwise_rmsd_A"] = 0.0

    _ensemble_stats("af", af)
    _ensemble_stats("bia", bia)
    _ensemble_stats("unb", unb)

    # cross-ensemble pairwise (ref vs X) using sampled indices
    def _cross_stats(name, traj):
        if traj is None:
            return
        t, idx = _sample_traj(traj, n_sample)
        M = get_rmsds(ref.xyz[ref_idx], t.xyz[idx], broadcast=True)
        results[f"ref_vs_{name}_mean_pairwise_rmsd_A"] = float(M.mean())
        results[f"ref_vs_{name}_rms_pairwise_rmsd_A"] = float(np.sqrt(np.square(M).mean()))

    _cross_stats("af", af)
    _cross_stats("bia", bia)
    _cross_stats("unb", unb)

    # PCA comparisons: fit PCA on reference sample and project others into it
    def _fit_pca_on_ref(ref_traj, ref_idx, n_comp=5):
        data = ref_traj.xyz[ref_idx].reshape(len(ref_idx), -1)
        k = min(n_comp, data.shape[0], data.shape[1])
        if k <= 0:
            return None
        pca = PCA(n_components=k)
        pca.fit(data)
        return pca

    ref_pca = _fit_pca_on_ref(ref, ref_idx, n_comp=5)

    def _proj_var_frac(pca, traj, sample_k=500):
        if pca is None or traj is None:
            return None
        m = traj.n_frames
        k = min(sample_k, m)
        idx = rng.choice(m, size=k, replace=False) if m >= k else rng.choice(m, size=k, replace=True)
        coords = pca.transform(traj.xyz[idx].reshape(len(idx), -1))
        total = np.var(coords, axis=0).sum()
        if total <= 0:
            return 0.0
        return float(np.var(coords[:, 0]) / total)

    # normalized cosine between ref PC1 and ensemble PC1 (and abs version)
    def _cosine_and_proj(name, traj):
        if ref_pca is None or traj is None:
            return
        ref_pc = ref_pca.components_[0]
        # compute ensemble's own PC1 (for signed cosine) if possible
        data_t = traj.xyz.reshape(traj.n_frames, -1)
        k = min(5, data_t.shape[0], data_t.shape[1])
        if k <= 0:
            return
        pca_t = PCA(n_components=k).fit(data_t)
        t_pc = pca_t.components_[0]
        # normalized cosine
        denom = (np.linalg.norm(ref_pc) * np.linalg.norm(t_pc))
        cosine = float(np.dot(ref_pc, t_pc) / denom) if denom > 0 else 0.0
        results[f"cosine_ref_{name}_signed"] = float(cosine)
        results[f"cosine_ref_{name}_abs"] = float(abs(cosine))
        # fraction of variance in this ensemble captured by ref PC1 basis
        results[f"{name}_var_frac_along_ref_pc"] = _proj_var_frac(ref_pca, traj)

    _cosine_and_proj("af", af)
    _cosine_and_proj("bia", bia)
    _cosine_and_proj("unb", unb)

    return results
