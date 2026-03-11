from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

import numpy as np
import mdtraj as md


"""
SASA + contact probability metric.

- Sidechain SASA per residue (Shrake–Rupley), reported in Å^2
- Exposure probability per residue above a threshold
- SASA mutual information (MI) matrix (L x L) from binary exposure states
- CA–CA contact probability matrix (L x L) from standardized CA trajectories

Notes:
- mdtraj.shrake_rupley returns SASA in nm^2. We convert to Å^2 by *100.
- Contact cutoff uses nm (0.8 nm = 8 Å).
"""


_BACKBONE_NAMES = {"CA", "C", "N", "O", "OXT"}


def condense_sidechain_sasas(
    sasas_nm2: np.ndarray,
    top: md.Topology,
    *,
    exclude_hydrogen: bool = True,
    protein_only: bool = True,
) -> np.ndarray:
    """
    Condense atom-level SASA (nm^2) into per-residue *sidechain* SASA (nm^2).

    sasas_nm2: shape (frames, atoms), produced by md.shrake_rupley(mode="atom")
    returns: shape (frames, residues)
    """
    if top.n_residues <= 1:
        raise ValueError("Topology appears to have <=1 residues; unexpected for SASA condensation")
    if sasas_nm2.ndim != 2:
        raise ValueError(f"Expected 2D SASA array (frames, atoms), got {sasas_nm2.shape}")
    if top.n_atoms != sasas_nm2.shape[1]:
        raise ValueError(
            f"Atom count mismatch: topology has {top.n_atoms} atoms, SASA has {sasas_nm2.shape[1]} columns. "
            "Make sure you computed atom-level SASA with mode='atom' using the same topology."
        )

    atoms = list(top.atoms)
    res_id = np.fromiter((a.residue.index for a in atoms), dtype=int, count=top.n_atoms)

    sc_mask = np.fromiter((a.name not in _BACKBONE_NAMES for a in atoms), dtype=bool, count=top.n_atoms)

    if exclude_hydrogen:
        h_mask = np.fromiter(
            ((a.element is not None and a.element.symbol == "H") for a in atoms),
            dtype=bool,
            count=top.n_atoms,
        )
        sc_mask &= ~h_mask

    if protein_only:
        prot_mask = np.fromiter((a.residue.is_protein for a in atoms), dtype=bool, count=top.n_atoms)
        sc_mask &= prot_mask

    n_frames = sasas_nm2.shape[0]
    rsd_sasa = np.zeros((n_frames, top.n_residues), dtype=np.float32)

    idx = res_id[sc_mask]
    if idx.size == 0:
        # e.g., CA-only trajectory/topology
        return rsd_sasa

    # Fast accumulation per frame using bincount
    for t in range(n_frames):
        rsd_sasa[t] = np.bincount(idx, weights=sasas_nm2[t, sc_mask], minlength=top.n_residues).astype(np.float32)

    return rsd_sasa


def sasa_mi(sasa_bool: np.ndarray, *, eps: float = 1e-12, base: str = "e") -> np.ndarray:
    """
    Mutual information matrix between binary exposure states.

    sasa_bool: boolean array shape (frames, residues), True=exposed
    returns: float matrix shape (residues, residues), diagonal set to 0

    base:
      - "e": nats
      - "2": bits
    """
    X = np.asarray(sasa_bool)
    if X.ndim != 2:
        raise ValueError(f"sasa_mi expects shape (frames, residues), got {X.shape}")
    if X.dtype != np.bool_:
        X = X.astype(bool, copy=False)

    N, L = X.shape
    if N == 0:
        raise ValueError("sasa_mi received zero frames")
    if L == 0:
        raise ValueError("sasa_mi received zero residues")

    # Joint probabilities P(a,b) for a,b in {0,1}
    joint = np.empty((L, L, 2, 2), dtype=np.float64)

    A = X
    joint[:, :, 1, 1] = (A[:, :, None] & A[:, None, :]).mean(axis=0)
    joint[:, :, 1, 0] = (A[:, :, None] & ~A[:, None, :]).mean(axis=0)
    joint[:, :, 0, 1] = (~A[:, :, None] & A[:, None, :]).mean(axis=0)
    joint[:, :, 0, 0] = (~A[:, :, None] & ~A[:, None, :]).mean(axis=0)

    p1 = A.mean(axis=0)  # P(exposed=1) per residue
    marg = np.stack([1.0 - p1, p1], axis=-1)  # (L,2)
    indep = marg[None, :, None, :] * marg[:, None, :, None]  # (L,L,2,2)

    joint = np.clip(joint, eps, 1.0)
    indep = np.clip(indep, eps, 1.0)

    mi = np.sum(joint * np.log(joint / indep), axis=(2, 3))

    if base == "2":
        mi /= np.log(2.0)
    elif base != "e":
        raise ValueError("base must be 'e' (nats) or '2' (bits)")

    np.fill_diagonal(mi, 0.0)
    return mi


def contact_prob_matrix_from_ca_xyz(
    xyz_nm: np.ndarray,
    *,
    cutoff_nm: float = 0.8,
    chunk_frames: int = 200,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Full CA–CA contact probability matrix.

    xyz_nm: shape (T, L, 3), nm
    returns: shape (L, L), float32 with probabilities in [0,1]

    This is chunked to avoid the huge (T,L,L,3) allocation.
    """
    xyz_nm = np.asarray(xyz_nm, dtype=np.float32)
    if xyz_nm.ndim != 3 or xyz_nm.shape[-1] != 3:
        raise ValueError(f"Expected xyz shape (T, L, 3), got {xyz_nm.shape}")

    T, L, _ = xyz_nm.shape
    if T == 0:
        raise ValueError("No frames provided for contact matrix")
    if L == 0:
        raise ValueError("No residues/atoms provided for contact matrix")

    cutoff2 = np.float32(cutoff_nm * cutoff_nm)
    counts = np.zeros((L, L), dtype=np.uint32)

    for start in range(0, T, chunk_frames):
        end = min(T, start + chunk_frames)
        x = xyz_nm[start:end]  # (t, L, 3)

        # d2(t,i,j) = ||x(t,i)||^2 + ||x(t,j)||^2 - 2 * x(t,i)·x(t,j)
        s = np.sum(x * x, axis=-1)                           # (t, L)
        g = np.einsum("tik,tjk->tij", x, x, optimize=True)    # (t, L, L)
        d2 = s[:, :, None] + s[:, None, :] - 2.0 * g         # (t, L, L)
        d2 = np.maximum(d2, 0.0)

        contacts = d2 < cutoff2
        counts += contacts.sum(axis=0).astype(np.uint32)

    prob = counts.astype(np.float32) / np.float32(T)
    if zero_diagonal:
        np.fill_diagonal(prob, 0.0)
    return prob


def _save_npy(path: Path, arr: np.ndarray) -> str:
    np.save(path, arr)
    return str(path)


def compute(job, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = job.reference_md.traj
    af = job.alphaflow.traj if getattr(job, "alphaflow", None) is not None else None
    biased = job.biased.traj if getattr(job, "biased", None) is not None else None
    unb = job.unbiased.traj if getattr(job, "unbiased", None) is not None else None

    results: Dict[str, Any] = {"metric": "sasa_contact"}
    artifacts: Dict[str, str] = {}
    warns = []

    # Parameters matching the AlphaFlow analyze_ensembles script
    probe_radius_nm = 0.28
    sasa_thresh_nm2 = 0.02
    sasa_thresh_A2 = sasa_thresh_nm2 * 100.0  # 2 Å^2
    mi_log_base = "e"  # change to "2" if you want bits
    exclude_h = True
    protein_only = True

    # Guards (still compute for ~300 residues, but warn/skip if huge)
    max_res_for_mi = 350
    max_frames_for_mi = 1500
    max_frames_for_sasa = 2000   # shrake_rupley segfaults on very large trajectories
    approx_bool_ops_limit = 200_000_000  # rough guard on N*L^2

    rng = np.random.default_rng(137)

    def _compute_sasa(tag: str, traj: md.Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        # Subsample if too many frames to avoid shrake_rupley segfault on large arrays
        if traj.n_frames > max_frames_for_sasa:
            idx = rng.choice(traj.n_frames, size=max_frames_for_sasa, replace=False)
            traj = traj[sorted(idx)]
        # Atom SASA (nm^2)
        atom_sasa_nm2 = md.shrake_rupley(traj, probe_radius=probe_radius_nm, mode="atom")
        # Sidechain per residue (nm^2)
        sc_nm2 = condense_sidechain_sasas(
            atom_sasa_nm2,
            traj.topology,
            exclude_hydrogen=exclude_h,
            protein_only=protein_only,
        )
        sc_A2 = sc_nm2 * 100.0
        exposed = sc_A2 > sasa_thresh_A2
        results[f"{tag}_sa_prob"] = exposed.mean(axis=0).astype(np.float32).tolist()
        results[f"{tag}_mean_sc_sasa_A2"] = float(sc_A2.mean())
        results[f"{tag}_mean_sa_prob"] = float(exposed.mean())
        return sc_A2, exposed

    # SASA per ensemble
    ref_sc_A2, ref_exp = _compute_sasa("ref", ref)

    if af is not None:
        af_sc_A2, af_exp = _compute_sasa("af", af)
    else:
        af_sc_A2, af_exp = None, None

    if unb is not None:
        unb_sc_A2, unb_exp = _compute_sasa("unb", unb)
    else:
        unb_sc_A2, unb_exp = None, None

    if biased is not None:
        bia_sc_A2, bia_exp = _compute_sasa("bia", biased)
    else:
        bia_sc_A2, bia_exp = None, None

    # MI matrices (save to .npy; do NOT dump to json)
    def _maybe_mi(tag: str, exposed_bool: np.ndarray, top: md.Topology) -> None:
        L = top.n_residues
        if L > max_res_for_mi:
            msg = f"Skipping {tag} SASA MI: L={L} > max_res_for_mi={max_res_for_mi}"
            warns.append(msg)
            warnings.warn(msg)
            return

        X = exposed_bool
        N = X.shape[0]
        if N > max_frames_for_mi:
            idx = rng.choice(N, size=max_frames_for_mi, replace=False)
            X = X[idx]
            msg = f"Downsampling {tag} SASA MI frames: {N} -> {max_frames_for_mi}"
            warns.append(msg)
            warnings.warn(msg)

        approx_ops = X.shape[0] * (L * L)
        if approx_ops > approx_bool_ops_limit:
            msg = f"Skipping {tag} SASA MI: approx N*L^2={approx_ops} too large"
            warns.append(msg)
            warnings.warn(msg)
            return

        mi = sasa_mi(X, base=mi_log_base).astype(np.float32)
        artifacts[f"{tag}_mi_mat_npy"] = _save_npy(out_dir / f"{tag}_sasa_mi.npy", mi)
        results[f"{tag}_mi_mean"] = float(mi.mean())
        results[f"{tag}_mi_p95"] = float(np.quantile(mi, 0.95))

    _maybe_mi("ref", ref_exp, ref.topology)
    if af_exp is not None:
        _maybe_mi("af", af_exp, af.topology)
    if unb_exp is not None:
        _maybe_mi("unb", unb_exp, unb.topology)
    if bia_exp is not None:
        _maybe_mi("bia", bia_exp, biased.topology)

    # Contact probability matrices (CA-only standardized trajectories preferred)
    # Uses job.*_std if present: expected shape (T, L, 3) where L ~ residues (CA atoms).
    # Saves full LxL contact probability matrices to .npy
    if getattr(job, "ref_md_std", None) is not None:
        ca_ref = job.ref_md_std
        try:
            ref_cp = contact_prob_matrix_from_ca_xyz(ca_ref.xyz, cutoff_nm=0.8, chunk_frames=200, zero_diagonal=True)
            artifacts["ref_contact_prob_npy"] = _save_npy(out_dir / "ref_contact_prob.npy", ref_cp)
            results["ref_contact_mean"] = float(ref_cp.mean())
        except Exception as e:
            msg = f"Failed to compute ref contact_prob matrix: {e}"
            warns.append(msg)
            warnings.warn(msg)

        if getattr(job, "alphaflow_std", None) is not None:
            ca_af = job.alphaflow_std
            if ca_af.n_atoms != ca_ref.n_atoms:
                raise ValueError("Standardized CA trajectories have different atom counts for ref and AF")
            af_cp = contact_prob_matrix_from_ca_xyz(ca_af.xyz, cutoff_nm=0.8, chunk_frames=200, zero_diagonal=True)
            artifacts["af_contact_prob_npy"] = _save_npy(out_dir / "af_contact_prob.npy", af_cp)
            results["af_contact_mean"] = float(af_cp.mean())

        if getattr(job, "unbiased_std", None) is not None:
            ca_unb = job.unbiased_std
            if ca_unb.n_atoms != ca_ref.n_atoms:
                raise ValueError("Standardized CA trajectories have different atom counts for ref and unbiased")
            unb_cp = contact_prob_matrix_from_ca_xyz(ca_unb.xyz, cutoff_nm=0.8, chunk_frames=200, zero_diagonal=True)
            artifacts["unb_contact_prob_npy"] = _save_npy(out_dir / "unb_contact_prob.npy", unb_cp)
            results["unb_contact_mean"] = float(unb_cp.mean())

        if getattr(job, "biased_std", None) is not None:
            ca_bia = job.biased_std
            if ca_bia.n_atoms != ca_ref.n_atoms:
                raise ValueError("Standardized CA trajectories have different atom counts for ref and biased")
            bia_cp = contact_prob_matrix_from_ca_xyz(ca_bia.xyz, cutoff_nm=0.8, chunk_frames=200, zero_diagonal=True)
            artifacts["bia_contact_prob_npy"] = _save_npy(out_dir / "bia_contact_prob.npy", bia_cp)
            results["bia_contact_mean"] = float(bia_cp.mean())

    results["params"] = {
        "probe_radius_nm": probe_radius_nm,
        "sasa_thresh_nm2": sasa_thresh_nm2,
        "sasa_thresh_A2": sasa_thresh_A2,
        "exclude_hydrogen": exclude_h,
        "protein_only": protein_only,
        "mi_log_base": mi_log_base,
        "contact_cutoff_nm": 0.8,
    }
    results["warnings"] = warns
    results["artifacts"] = artifacts
    return results
