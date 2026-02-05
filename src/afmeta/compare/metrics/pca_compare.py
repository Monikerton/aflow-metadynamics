# compare/metrics/pca_compare.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from ..preprocess import rbias_per_frame_from_colvar
from ...features import prepare_pca_features, transform_with_feature_result



def _flat(traj) -> np.ndarray:
    return traj.xyz.reshape(traj.n_frames, -1)

def _basic_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "max": float(x.max()),
    }

def _xy_range(*xys: Optional[np.ndarray], pad: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = np.concatenate([xy[:, 0] for xy in xys if xy is not None and len(xy)])
    ys = np.concatenate([xy[:, 1] for xy in xys if xy is not None and len(xy)])
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    dx = pad * (x1 - x0 if x1 > x0 else 1.0)
    dy = pad * (y1 - y0 if y1 > y0 else 1.0)
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def _fes_kbt(
    xy: np.ndarray,
    *,
    bins: int,
    xy_range: Tuple[Tuple[float, float], Tuple[float, float]],
    weights: Optional[np.ndarray],
    seed_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a free emergy surface (FES) in kBT units from 2D data.
    Uses weights if provided (e.g. from reweighting).
    """
    (xmin, xmax), (ymin, ymax) = xy_range
    H, x_edges, y_edges = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=weights,
    )
    P = H / H.sum() if H.sum() > 0 else H
    
    F = -np.log(P + 1e-12)  # epsilon to avoid log(0)
    F[H == 0] = np.nan # the 0 regions are regarded as snan

    # shift: try seed bin, else global min
    sx, sy = float(seed_xy[0, 0]), float(seed_xy[0, 1])
    ix = int(np.searchsorted(x_edges, sx, side="right") - 1)
    iy = int(np.searchsorted(y_edges, sy, side="right") - 1)
    if 0 <= ix < F.shape[0] and 0 <= iy < F.shape[1] and np.isfinite(F[ix, iy]):
        F = F - F[ix, iy]
    else:
        F = F - np.nanmin(F)

    return F, x_edges, y_edges


def _beta(energy_unit: str, T: float) -> float:
    # beta = 1/(kB*T) in matching energy units
    if energy_unit == "kbt":
        return 1.0
    if energy_unit == "kj":
        kB = 0.0083144621
        return 1.0 / (kB * T)
    if energy_unit == "kcal":
        kB = 0.0019872041
        return 1.0 / (kB * T)
    raise ValueError(f"Unsupported energy_unit={energy_unit!r} (use 'kj','kcal','kbt').")


def compute(job, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = job.ref_std or job.reference_md.traj
    biased = job.biased_std or job.biased.traj
    unbiased = (job.unbiased_std or (job.unbiased.traj if job.unbiased else None))
    alphaflow = (job.alphaflow_std or (job.alphaflow.traj if job.alphaflow else None))

    # if no unbiased, plot ref subset with same frame count as biased
    if unbiased is None:
        # at the present moment, the reference trajectory was calculated to have 50000 frames for 5 microseconds, 
        # but the biased trajectory has 50000 frames for 200 nanoseconds. 
        # So we take a subset of reference trajectory with same frame count as biased trajectory for better comparison.
        # n = min(int(ref.n_frames), int(biased.n_frames)) * 2 // 50 #  TODO: make this generalizable 
        n = 2000
        unbiased_plot = ref[:n]
        unb_title = f"Reference (subset, n={n})"
    else:
        unbiased_plot = unbiased
        unb_title = "Unbiased (OpenMM)"
    
    ft = prepare_pca_features( # TODO: Wrap this to automatically output from the specific job.
        ref,
        top_m=80,
        min_seq_sep=job.reference_md.min_seq_sep if hasattr(job.reference_md, "min_seq_sep") else 3,
        periodic=False,
        n_components=2,
        random_state=0,
        label_prefix="cc_r",
    )



    ref_xy = transform_with_feature_result(ref, ft)
    bia_xy = transform_with_feature_result(biased, ft)
    unb_xy = transform_with_feature_result(unbiased_plot, ft)
    af_xy = transform_with_feature_result(alphaflow, ft) if alphaflow is not None else None
    seed_xy = ref_xy[:1]

    # weights from rbias if present
    weights = None
    rbias_stats = None

    if job.biased.colvar is not None:
        rb = rbias_per_frame_from_colvar(job.biased.colvar, biased.time)
        if rb.shape[0] != biased.n_frames:
            raise ValueError(f"rbias length mismatch: rb={rb.shape[0]} vs biased.n_frames={biased.n_frames}")

        beta = _beta(job.energy_unit, float(job.temperature_K))
        w = np.exp(beta * (rb - rb.max()))
        weights = w / w.mean()

        rbias_stats = {
            "rbias": _basic_stats(rb)}

    rng = _xy_range(ref_xy, bia_xy, unb_xy, af_xy)
    bins = 120

    #reference 
    F_ref, x_edges, y_edges = _fes_kbt(ref_xy, bins=bins, xy_range=rng, weights=None, seed_xy=seed_xy)
    F_bia, _, _ = _fes_kbt(bia_xy, bins=bins, xy_range=rng, weights=weights, seed_xy=seed_xy)
    F_unb, _, _ = _fes_kbt(unb_xy, bins=bins, xy_range=rng, weights=None, seed_xy=seed_xy)

    vals = np.concatenate([F_ref[np.isfinite(F_ref)], F_bia[np.isfinite(F_bia)], F_unb[np.isfinite(F_unb)]])
    vmin, vmax = (float(np.percentile(vals, 5)), float(np.percentile(vals, 95))) if vals.size else (0.0, 1.0)

    #save artifacts
    artifacts: Dict[str, str] = {}

    def _save(name: str, data: np.ndarray) -> None:
        p = out_dir / f"{name}.npy"
        np.save(p, data)
        artifacts[name] = p.name
    
    _save("ref_xy", ref_xy)
    _save("bia_xy", bia_xy)
    _save("unb_xy", unb_xy)
    if af_xy is not None:
        _save("af_xy", af_xy)
    _save("seed_xy", seed_xy)
    _save("F_ref", F_ref)
    _save("F_bia", F_bia)
    _save("F_unb", F_unb)
    _save("x_edges", x_edges)
    _save("y_edges", y_edges)   


    return {
        "metric": "pca_compare",
        "artifacts": artifacts,
        "unb_title": unb_title,
        "pca_explained_variance_ratio": [float(x) for x in ft.model.explained_variance_ratio_],
        "temperature_K": float(job.temperature_K),
        "energy_unit": job.energy_unit,
        "rbias": rbias_stats,
        "fes_color_limits": {"vmin": vmin, "vmax": vmax},
    }
