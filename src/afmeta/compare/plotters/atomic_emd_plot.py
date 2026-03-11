from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


_CROSS_STYLES = {
    "af":  dict(label="ref vs AlphaFlow", color="#9C27B0", lw=1.5, zorder=3),
    "bia": dict(label="ref vs MetaD",     color="#FF5722", lw=1.5, zorder=2),
    "unb": dict(label="ref vs Unbiased",  color="#4CAF50", lw=1.5, zorder=2),
}


def plot(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    # --- Mean per-atom distance ---
    ax = axes[0]
    has_mean = False
    for tag in ("af", "bia", "unb"):
        key = f"emd_mean_ref_{tag}_per_atom"
        if key not in results:
            continue
        arr = np.asarray(results[key], dtype=np.float32)
        st = _CROSS_STYLES[tag]
        ax.plot(np.arange(1, arr.size + 1), arr,
                label=st["label"], color=st["color"], lw=st["lw"], zorder=st["zorder"])
        has_mean = True

    ax.set_xlabel("Atom index")
    ax.set_ylabel("Mean distance (Å)")
    ax.set_title("Per-atom mean positional distance (ref vs ensemble)")
    if has_mean:
        ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- EMD variance per atom ---
    ax = axes[1]
    has_var = False
    for tag in ("af", "bia", "unb"):
        key = f"emd_var_ref_{tag}_per_atom"
        if key not in results:
            continue
        arr = np.asarray(results[key], dtype=np.float32)
        st = _CROSS_STYLES[tag]
        ax.plot(np.arange(1, arr.size + 1), arr,
                label=st["label"], color=st["color"], lw=st["lw"], zorder=st["zorder"])
        has_var = True

    ax.set_xlabel("Atom index")
    ax.set_ylabel("EMD (Å)")
    ax.set_title("Per-atom covariance EMD (ref vs ensemble)")
    if has_var:
        ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle("Atomic EMD Summary", fontsize=11, fontweight="bold")

    plot_path = out_dir / "atomic_emd.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path
