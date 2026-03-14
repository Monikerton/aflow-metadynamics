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

_CONTACT_TITLES = {
    "bia": "MetaD",
    "unb": "Unbiased MD",
    "af":  "AlphaFlow",
}


def plot(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = results.get("artifacts", {})

    # ------------------------------------------------------------------
    # Figure 1: Per-residue solvent-accessibility probability
    # ------------------------------------------------------------------
    sa_tags = [t for t in ("ref", "bia", "unb", "af") if f"{t}_sa_prob" in results]
    if sa_tags:
        fig1, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
        n_res = max(len(results[f"{t}_sa_prob"]) for t in sa_tags)
        for tag in ("ref", "bia", "unb", "af"):
            if tag not in sa_tags:
                continue
            arr = np.asarray(results[f"{tag}_sa_prob"], dtype=np.float32)
            st = _ENSEMBLE_STYLES[tag]
            ax1.plot(np.arange(1, arr.size + 1), arr,
                     label=st["label"], color=st["color"],
                     lw=st["lw"], zorder=st["zorder"], ls=st["ls"], alpha=st["alpha"])
        ax1.set_xlabel("Residue index")
        ax1.set_ylabel("Solvent-accessibility probability")
        ax1.set_title("Per-residue solvent-accessibility probability")
        ax1.legend(framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, n_res)
        ax1.set_ylim(0, 1)
        sa_path = out_dir / "sasa_sa_prob.png"
        fig1.savefig(sa_path, dpi=200, bbox_inches="tight")
        plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Contact probability matrices
    # Show ref absolute, then bia diff (primary), then unb diff, then af diff
    # ------------------------------------------------------------------
    contact_mats = {}
    for tag in ("ref", "bia", "unb", "af"):
        key = f"{tag}_contact_prob_npy"
        p = artifacts.get(key)
        if p and Path(p).exists():
            contact_mats[tag] = np.load(p)

    if contact_mats:
        # diff order: bia first (primary comparison), then unb, then af
        diff_tags = [t for t in ("bia", "unb", "af") if t in contact_mats]
        n_cols = 1 + len(diff_tags)
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4),
                                   constrained_layout=True)
        if n_cols == 1:
            axes2 = [axes2]

        if "ref" in contact_mats:
            ref_cp = contact_mats["ref"]
            im = axes2[0].imshow(ref_cp, vmin=0, vmax=1, cmap="Blues", origin="lower")
            axes2[0].set_title("Reference MD\ncontact prob", fontsize=9)
            axes2[0].set_xlabel("Residue")
            axes2[0].set_ylabel("Residue")
            plt.colorbar(im, ax=axes2[0], fraction=0.046, pad=0.04, label="P(contact)")

        diff_cmap = plt.get_cmap("RdBu_r")
        for col_i, tag in enumerate(diff_tags, start=1):
            st = _ENSEMBLE_STYLES[tag]
            if "ref" in contact_mats:
                diff = contact_mats[tag] - contact_mats["ref"]
                vlim = max(abs(diff.min()), abs(diff.max()), 0.01)
                im = axes2[col_i].imshow(diff, vmin=-vlim, vmax=vlim,
                                         cmap=diff_cmap, origin="lower",
                                         alpha=st["alpha"])
                axes2[col_i].set_title(
                    f"{_CONTACT_TITLES.get(tag, tag)}\n− Reference MD", fontsize=9)
            else:
                cp = contact_mats[tag]
                im = axes2[col_i].imshow(cp, vmin=0, vmax=1, cmap="Blues", origin="lower")
                axes2[col_i].set_title(
                    f"{_CONTACT_TITLES.get(tag, tag)}\ncontact prob", fontsize=9)
            axes2[col_i].set_xlabel("Residue")
            axes2[col_i].set_ylabel("Residue")
            plt.colorbar(im, ax=axes2[col_i], fraction=0.046, pad=0.04)

        fig2.suptitle("CA–CA Contact Probability", fontsize=11, fontweight="bold")
        contact_path = out_dir / "sasa_contact_prob.png"
        fig2.savefig(contact_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)

    primary = out_dir / "sasa_sa_prob.png"
    if not primary.exists():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No SASA data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(primary, dpi=100)
        plt.close(fig)

    return primary
