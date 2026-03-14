from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# bia is primary; unb is secondary; af is de-emphasized
_CROSS_STYLES = {
    "bia": dict(label="ref vs MetaD",    color="#FF5722", lw=2.0, zorder=3, ls="-",  alpha=1.0),
    "unb": dict(label="ref vs Unbiased", color="#4CAF50", lw=1.5, zorder=2, ls="-",  alpha=0.8),
    "af":  dict(label="ref vs AlphaFlow",color="#9C27B0", lw=1.0, zorder=1, ls="--", alpha=0.55),
}


def plot(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    for ax_i, metric in enumerate(("mean", "var")):
        ax = axes[ax_i]
        has_data = False
        for tag in ("bia", "unb", "af"):
            key = f"emd_{metric}_ref_{tag}_per_atom"
            if key not in results:
                continue
            arr = np.asarray(results[key], dtype=np.float32)
            st = _CROSS_STYLES[tag]
            ax.plot(np.arange(1, arr.size + 1), arr,
                    label=st["label"], color=st["color"], lw=st["lw"],
                    zorder=st["zorder"], ls=st["ls"], alpha=st["alpha"])
            has_data = True

        ax.set_xlabel("Residue (CA)")
        if metric == "mean":
            ax.set_ylabel("Mean distance (Å)")
            ax.set_title("Per-residue mean positional distance (ref vs ensemble)")
        else:
            ax.set_ylabel("EMD (Å)")
            ax.set_title("Per-residue covariance EMD (ref vs ensemble)")
        if has_data:
            ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Atomic EMD Summary", fontsize=11, fontweight="bold")

    plot_path = out_dir / "atomic_emd.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path
