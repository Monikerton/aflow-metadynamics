from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# ref and bia are the primary comparison; unb is secondary; af is de-emphasized
_ENSEMBLE_STYLES = {
    "ref": dict(label="Reference MD", color="#2196F3", lw=2.0, zorder=4, ls="-",  alpha=1.0),
    "bia": dict(label="MetaD",        color="#FF5722", lw=2.0, zorder=3, ls="-",  alpha=1.0),
    "unb": dict(label="Unbiased MD",  color="#4CAF50", lw=1.5, zorder=2, ls="-",  alpha=0.8),
    "af":  dict(label="AlphaFlow",    color="#9C27B0", lw=1.0, zorder=1, ls="--", alpha=0.55),
}


def plot(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = results.get("artifacts", {})
    scores = results.get("scores", {})

    arrays: Dict[str, np.ndarray] = {}
    for tag in ("ref", "bia", "unb", "af"):
        key = f"{tag}_rmsf_A"
        p = artifacts.get(key)
        if p and Path(p).exists():
            arrays[tag] = np.load(p)

    if not arrays:
        raise ValueError("No RMSF artifact arrays found in results['artifacts']")

    n_res = max(a.size for a in arrays.values())
    residue_idx = np.arange(1, n_res + 1)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    for tag in ("ref", "bia", "unb", "af"):
        if tag not in arrays:
            continue
        arr = arrays[tag]
        st = _ENSEMBLE_STYLES[tag]
        sc = scores.get(tag, {})
        n_frames = sc.get("n_frames", "?")
        label = f"{st['label']} (n={n_frames})"
        ax.plot(residue_idx[: arr.size], arr, label=label,
                color=st["color"], lw=st["lw"], zorder=st["zorder"],
                ls=st["ls"], alpha=st["alpha"])

    ax.set_xlabel("Residue index")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title("Per-residue RMSF")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_res)
    ax.set_ylim(bottom=0)

    plot_path = out_dir / "rmsf.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path
