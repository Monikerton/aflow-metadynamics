from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


def _xy_range(*xys: Optional[np.ndarray], pad: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = np.concatenate([xy[:, 0] for xy in xys if xy is not None and len(xy)])
    ys = np.concatenate([xy[:, 1] for xy in xys if xy is not None and len(xy)])
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    dx = pad * (x1 - x0 if x1 > x0 else 1.0)
    dy = pad * (y1 - y0 if y1 > y0 else 1.0)
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def plot(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    art = results.get("artifacts", {})
    if not art:
        raise ValueError("pca_compare plotter expected results['artifacts'] with .npy paths")
    
    # Assume metrics/<metric_name> is a sibling of plots/<metric_name>
    metric_name = out_dir.name
    run_root = out_dir.parents[1]              # .../<run_root>
    metrics_dir = run_root / "metrics" / metric_name

    def _resolve(p: str) -> Path:
        pth = Path(p)
        return pth if pth.is_absolute() else (metrics_dir / pth)

    def _load(name: str) -> np.ndarray:
        p = art.get(name)
        if not p:
            raise ValueError(f"Missing artifact {name!r}")
        return np.load(_resolve(p))


    ref_xy = _load("ref_xy")
    bia_xy = _load("bia_xy")
    unb_xy = _load("unb_xy")
    af_xy = _load("af_xy") if "af_xy" in art else None
    seed_xy = _load("seed_xy")

    F_ref = _load("F_ref")
    F_bia = _load("F_bia")
    F_unb = _load("F_unb")
    x_edges = _load("x_edges")
    y_edges = _load("y_edges")

    # pull color limits + title from results
    vmin = float(results.get("fes_color_limits", {}).get("vmin", np.nanmin(F_ref)))
    vmax = float(results.get("fes_color_limits", {}).get("vmax", np.nanmax(F_ref)))
    unb_title = results.get("unb_title", "Unbiased (OpenMM)")

    rng = _xy_range(ref_xy, bia_xy, unb_xy, af_xy)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)

    def fes(ax, F, title: str):
        im = ax.imshow(F.T, origin="lower", extent=extent, aspect="auto",
                       cmap="Greens_r", vmin=vmin, vmax=vmax)
        ax.set(title=title, xlabel="PC1", ylabel="PC2")
        ax.plot(seed_xy[0, 0], seed_xy[0, 1], "o", ms=4, mew=0.5, color="red", alpha=0.9)
        return im

    def scatter(ax, xy, title: str):
        ax.scatter(xy[:, 0], xy[:, 1], s=10, alpha=0.35, linewidths=0)
        ax.set(title=title, xlabel="PC1", ylabel="PC2")
        ax.plot(seed_xy[0, 0], seed_xy[0, 1], "o", ms=4, mew=0.5, color="red", alpha=0.9)
        (xmin, xmax), (ymin, ymax) = rng
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    im0 = fes(axes[0, 0], F_ref, "Reference MD")
    scatter(
        axes[0, 1],
        af_xy if af_xy is not None else ref_xy,
        "AlphaFlow ensemble" if af_xy is not None else "(No AlphaFlow) Reference scatter",
    )
    fes(axes[1, 0], F_bia, "Biased (reweighted)" if "weights" in art else "Biased")
    fes(axes[1, 1], F_unb, unb_title)

    cbar = fig.colorbar(im0, ax=[axes[0, 0], axes[1, 0], axes[1, 1]], pad=0.02)
    cbar.set_label("ΔG (kBT)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plot_path = out_dir / "pca_compare.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path
