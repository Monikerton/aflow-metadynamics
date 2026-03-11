from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def _flat_xyz_nm(traj) -> np.ndarray:
    """traj.xyz: (T, L, 3) nm -> (T, 3L) nm (float32)

    Ensures float32 and returns (T, 3L) array in nm.
    """
    arr = traj.xyz.astype(np.float32, copy=False)
    return arr.reshape(traj.n_frames, -1)


def _sample_indices(n_frames: int, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0,), dtype=int)
    if n_frames <= n_sample:
        return np.arange(n_frames, dtype=int)
    return rng.choice(n_frames, size=n_sample, replace=False)


def _min_dist_to_set_A(
    query_nm: np.ndarray,
    ref_nm: np.ndarray,
    *,
    chunk_q: int = 256,
    chunk_r: int = 2048,
) -> np.ndarray:
    """
    For each query frame, compute min fixed-frame RMSD(Å) to any ref frame.
    Memory-safe: does not allocate full (Q,R) matrix.
    """
    Q = int(query_nm.shape[0])
    R = int(ref_nm.shape[0])
    if Q == 0:
        return np.zeros((0,), dtype=np.float32)
    if R == 0:
        return np.full((Q,), np.inf, dtype=np.float32)

    D = query_nm.shape[1]
    if ref_nm.shape[1] != D:
        raise ValueError(f"Dim mismatch: query D={D}, ref D={ref_nm.shape[1]}")
    if D % 3 != 0:
        raise ValueError(f"Expected D multiple of 3 (flattened xyz), got D={D}")

    n_atoms = D // 3
    out = np.empty((Q,), dtype=np.float32)

    for qs in range(0, Q, chunk_q):
        qe = min(Q, qs + chunk_q)
        q = query_nm[qs:qe]  # (qB, D)
        best = np.full((q.shape[0],), np.inf, dtype=np.float32)

        q2 = np.sum(q * q, axis=1, keepdims=True)  # (qB,1)

        for rs in range(0, R, chunk_r):
            re = min(R, rs + chunk_r)
            r = ref_nm[rs:re]                        # (rB, D)
            r2 = np.sum(r * r, axis=1, keepdims=True).T  # (1,rB)
            qr = q @ r.T                                 # (qB,rB)

            d2 = np.maximum(q2 + r2 - 2.0 * qr, 0.0)      # (qB,rB)
            min_d2 = d2.min(axis=1)                       # (qB,)

            rmsd_A = (np.sqrt(min_d2 / float(n_atoms)) * 10.0).astype(np.float32)
            best = np.minimum(best, rmsd_A)

        out[qs:qe] = best

    return out


def _precision_recall_at_tau(min_g2r_A: np.ndarray, min_r2g_A: np.ndarray, tau_A: float) -> Dict[str, float]:
    if min_g2r_A.size == 0:
        precision = 0.0
    else:
        precision = float(np.mean(min_g2r_A <= tau_A))

    if min_r2g_A.size == 0:
        recall = 0.0
    else:
        recall = float(np.mean(min_r2g_A <= tau_A))

    return {"precision": precision, "recall": recall}


def compute(job, out_dir: Path) -> Dict[str, Any]:
    """
    Precision/Recall (coverage) between reference ensemble and each available ensemble.

    Requires:
      job.ref_md_std (standardized CA trajectory; same atoms/order for all *_std)

    Computes:
      min distances (Å) from gen->ref and ref->gen, then precision/recall at tau_A.

    Saves:
      <name>_min_to_ref_A.npy
      ref_min_to_<name>_A.npy
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = getattr(job, "ref_md_std", None)
    if ref is None:
        raise ValueError("precision_recall requires job.ref_md_std (standardized CA trajectory)")
    if ref.n_frames == 0:
        raise ValueError("precision_recall requires job.ref_md_std to contain at least one frame")

    # ensembles to compare
    ensembles = {
        "af": getattr(job, "alphaflow_std", None),
        "bia": getattr(job, "biased_std", None),
        "unb": getattr(job, "unbiased_std", None),
    }

    # defaults (keep simple; can revise later)
    tau_A = 3.0
    sample_size = 1000  # sample frames per ensemble for nearest-neighbor coverage
    chunk_q = 256
    chunk_r = 2048

    rng = np.random.default_rng(137)

    ref_idx = _sample_indices(ref.n_frames, sample_size, rng)
    ref_nm = _flat_xyz_nm(ref[ref_idx])

    results: Dict[str, Any] = {
        "metric": "precision_recall",
        "params": {
            "tau_A": tau_A,
            "sample_size": sample_size,
            "chunk_q": chunk_q,
            "chunk_r": chunk_r,
            "method": "fixed_frame",
        },
        "ref_n_sample": int(ref_nm.shape[0]),
        "ref_n_total": int(ref.n_frames),
        "L_ca": int(ref.n_atoms),
        "scores": {},
        "artifacts": {},
        "warnings": [],
    }

    # sanity: fixed-frame assumes already aligned
    results["warnings"].append(
        "precision_recall uses fixed-frame RMSD in standardized CA space; ensure *_std trajectories are superposed consistently."
    )

    for name, traj in ensembles.items():
        if traj is None:
            continue

        if traj.n_frames == 0:
            results["warnings"].append(f"Skipping {name}: trajectory has zero frames")
            continue

        if traj.n_atoms != ref.n_atoms:
            results["warnings"].append(
                f"Skipping {name}: atom count mismatch (ref {ref.n_atoms} vs {name} {traj.n_atoms})"
            )
            continue

        gen_idx = _sample_indices(traj.n_frames, sample_size, rng)
        gen_nm = _flat_xyz_nm(traj[gen_idx])

        # min distances in Å
        min_g2r_A = _min_dist_to_set_A(gen_nm, ref_nm, chunk_q=chunk_q, chunk_r=chunk_r)
        min_r2g_A = _min_dist_to_set_A(ref_nm, gen_nm, chunk_q=chunk_q, chunk_r=chunk_r)

        # save for later plotting/curves (only if non-empty)
        if min_g2r_A.size > 0:
            p1 = out_dir / f"{name}_min_to_ref_A.npy"
            np.save(p1, min_g2r_A.astype(np.float32))
            results["artifacts"][f"{name}_min_to_ref_A"] = str(p1)
        if min_r2g_A.size > 0:
            p2 = out_dir / f"ref_min_to_{name}_A.npy"
            np.save(p2, min_r2g_A.astype(np.float32))
            results["artifacts"][f"ref_min_to_{name}_A"] = str(p2)

        sc = _precision_recall_at_tau(min_g2r_A, min_r2g_A, tau_A)
        sc.update(
            {
                "n_gen_sample": int(gen_nm.shape[0]),
                "n_gen_total": int(traj.n_frames),
                "min_to_ref_mean_A": float(min_g2r_A.mean()) if min_g2r_A.size else float("nan"),
                "min_to_ref_p95_A": float(np.quantile(min_g2r_A, 0.95)) if min_g2r_A.size else float("nan"),
                "ref_min_to_gen_mean_A": float(min_r2g_A.mean()) if min_r2g_A.size else float("nan"),
                "ref_min_to_gen_p95_A": float(np.quantile(min_r2g_A, 0.95)) if min_r2g_A.size else float("nan"),
            }
        )
        results["scores"][name] = sc

    return results
