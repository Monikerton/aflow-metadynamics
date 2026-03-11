from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


_STYLES = {
    "af":  dict(label="AlphaFlow", color="#9C27B0"),
    "bia": dict(label="MetaD",     color="#FF5722"),
    "unb": dict(label="Unbiased",  color="#4CAF50"),
}


def plot(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = results.get("scores", {})
    artifacts = results.get("artifacts", {})
    tau_A = results.get("params", {}).get("tau_A", 3.0)

    present_tags = [t for t in ("af", "bia", "unb") if t in scores]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    # --- Precision / Recall bar chart ---
    ax = axes[0]
    x = np.arange(len(present_tags))
    width = 0.35
    precs = [scores[t].get("precision", float("nan")) for t in present_tags]
    recs  = [scores[t].get("recall",    float("nan")) for t in present_tags]
    colors = [_STYLES[t]["color"] for t in present_tags]
    labels = [_STYLES[t]["label"] for t in present_tags]

    bars_p = ax.bar(x - width / 2, precs, width, label="Precision", alpha=0.85,
                    color=colors, edgecolor="white")
    bars_r = ax.bar(x + width / 2, recs,  width, label="Recall",    alpha=0.5,
                    color=colors, edgecolor="white", hatch="//")

    for bar, v in zip(list(bars_p) + list(bars_r), precs + recs):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(f"Precision & Recall (τ = {tau_A:.1f} Å)")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # --- CDF of min-to-ref distances ---
    ax = axes[1]
    has_cdf = False
    for tag in present_tags:
        key = f"{tag}_min_to_ref_A"
        p = artifacts.get(key)
        if p and Path(p).exists():
            dists = np.load(p).astype(np.float32)
            dists_sorted = np.sort(dists)
            cdf = np.arange(1, len(dists_sorted) + 1) / len(dists_sorted)
            st = _STYLES[tag]
            ax.plot(dists_sorted, cdf, label=st["label"], color=st["color"], lw=1.5)
            has_cdf = True

    ax.axvline(tau_A, color="gray", lw=1.0, ls="--", label=f"τ = {tau_A:.1f} Å")
    ax.set_xlabel("Min distance to reference (Å)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of nearest-reference distances (gen → ref)")
    if has_cdf:
        ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    fig.suptitle("Precision / Recall Coverage", fontsize=11, fontweight="bold")

    plot_path = out_dir / "precision_recall.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path
