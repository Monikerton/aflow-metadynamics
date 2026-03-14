from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# bia is primary; unb is secondary; af is de-emphasized
_STYLES = {
    "bia": dict(label="MetaD",     color="#FF5722", alpha=1.0,  lw=2.0, ls="-"),
    "unb": dict(label="Unbiased",  color="#4CAF50", alpha=0.8,  lw=1.5, ls="-"),
    "af":  dict(label="AlphaFlow", color="#9C27B0", alpha=0.55, lw=1.0, ls="--"),
}

# Order: bia first, then unb, then af
_ORDER = ("bia", "unb", "af")


def plot(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = results.get("scores", {})
    artifacts = results.get("artifacts", {})
    tau_A = results.get("params", {}).get("tau_A", 3.0)

    present_tags = [t for t in _ORDER if t in scores]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    # --- Precision / Recall bar chart ---
    ax = axes[0]
    x = np.arange(len(present_tags))
    width = 0.35
    precs = [scores[t].get("precision", float("nan")) for t in present_tags]
    recs  = [scores[t].get("recall",    float("nan")) for t in present_tags]

    for i, (tag, p, r) in enumerate(zip(present_tags, precs, recs)):
        st = _STYLES[tag]
        bar_p = ax.bar(i - width / 2, p, width, label="Precision" if i == 0 else "_",
                       color=st["color"], alpha=st["alpha"], edgecolor="white")
        bar_r = ax.bar(i + width / 2, r, width, label="Recall" if i == 0 else "_",
                       color=st["color"], alpha=st["alpha"] * 0.5,
                       edgecolor="white", hatch="//")
        for bar, v in [(bar_p, p), (bar_r, r)]:
            if not np.isnan(v):
                ax.text(bar[0].get_x() + bar[0].get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([_STYLES[t]["label"] for t in present_tags], fontsize=9)
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
            st = _STYLES[tag]
            dists = np.sort(np.load(p).astype(np.float32))
            cdf = np.arange(1, len(dists) + 1) / len(dists)
            ax.plot(dists, cdf, label=st["label"], color=st["color"],
                    lw=st["lw"], ls=st["ls"], alpha=st["alpha"])
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
