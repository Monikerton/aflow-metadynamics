from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


_STYLES = {
    "ref": dict(label="Reference MD", color="#2196F3"),
    "unb": dict(label="Unbiased MD",  color="#4CAF50"),
    "bia": dict(label="MetaD",        color="#FF5722"),
    "af":  dict(label="AlphaFlow",    color="#9C27B0"),
}

_CROSS_STYLES = {
    "af":  dict(label="ref vs AlphaFlow", color="#9C27B0"),
    "bia": dict(label="ref vs MetaD",     color="#FF5722"),
    "unb": dict(label="ref vs Unbiased",  color="#4CAF50"),
}


def _bar(ax, labels, values, colors, title, ylabel):
    xs = np.arange(len(labels))
    bars = ax.bar(xs, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)


def plot(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)

    # --- Within-ensemble mean pairwise RMSD ---
    within_tags = [t for t in ("ref", "af", "bia", "unb")
                   if f"{t}_within_mean_pairwise_rmsd_A" in results]
    if within_tags:
        _bar(
            axes[0, 0],
            labels=[_STYLES[t]["label"] for t in within_tags],
            values=[results[f"{t}_within_mean_pairwise_rmsd_A"] for t in within_tags],
            colors=[_STYLES[t]["color"] for t in within_tags],
            title="Within-ensemble mean pairwise RMSD",
            ylabel="RMSD (Å)",
        )

    # --- Cross-ensemble (ref vs X) mean pairwise RMSD ---
    cross_tags = [t for t in ("af", "bia", "unb")
                  if f"ref_vs_{t}_mean_pairwise_rmsd_A" in results]
    if cross_tags:
        _bar(
            axes[0, 1],
            labels=[_CROSS_STYLES[t]["label"] for t in cross_tags],
            values=[results[f"ref_vs_{t}_mean_pairwise_rmsd_A"] for t in cross_tags],
            colors=[_CROSS_STYLES[t]["color"] for t in cross_tags],
            title="Cross-ensemble mean pairwise RMSD (ref vs X)",
            ylabel="RMSD (Å)",
        )

    # --- PC1 cosine similarity (abs) ---
    cosine_tags = [t for t in ("af", "bia", "unb") if f"cosine_ref_{t}_abs" in results]
    if cosine_tags:
        _bar(
            axes[1, 0],
            labels=[_CROSS_STYLES[t]["label"] for t in cosine_tags],
            values=[results[f"cosine_ref_{t}_abs"] for t in cosine_tags],
            colors=[_CROSS_STYLES[t]["color"] for t in cosine_tags],
            title="|Cosine| of PC1: ref vs each ensemble",
            ylabel="|cosine|",
        )
        axes[1, 0].set_ylim(0, 1.05)

    # --- Variance fraction along ref PC1 ---
    var_tags = [t for t in ("af", "bia", "unb") if f"{t}_var_frac_along_ref_pc" in results]
    if var_tags:
        _bar(
            axes[1, 1],
            labels=[_STYLES[t]["label"] for t in var_tags],
            values=[results[f"{t}_var_frac_along_ref_pc"] for t in var_tags],
            colors=[_STYLES[t]["color"] for t in var_tags],
            title="Variance fraction along ref PC1",
            ylabel="Fraction",
        )
        axes[1, 1].set_ylim(0, 1.05)

    fig.suptitle("Ensemble RMSD Summary", fontsize=11, fontweight="bold")

    plot_path = out_dir / "ensemble_rmsd.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path
