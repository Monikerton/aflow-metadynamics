from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


_ENSEMBLE_STYLES = {
    "ref": dict(label="Reference MD", color="#2196F3", lw=1.5, zorder=3),
    "unb": dict(label="Unbiased MD",  color="#4CAF50", lw=1.5, zorder=2),
    "bia": dict(label="MetaD",        color="#FF5722", lw=1.5, zorder=2),
    "af":  dict(label="AlphaFlow",    color="#9C27B0", lw=1.5, zorder=2),
}

_CONTACT_TITLES = {
    "ref": "Reference MD",
    "af":  "AlphaFlow",
    "bia": "MetaD",
    "unb": "Unbiased MD",
}


def plot(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = results.get("artifacts", {})

    # ------------------------------------------------------------------
    # Figure 1: Per-residue solvent-accessibility probability
    # ------------------------------------------------------------------
    sa_tags = [t for t in ("ref", "af", "bia", "unb") if f"{t}_sa_prob" in results]
    if sa_tags:
        fig1, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
        n_res = max(len(results[f"{t}_sa_prob"]) for t in sa_tags)
        for tag in sa_tags:
            arr = np.asarray(results[f"{tag}_sa_prob"], dtype=np.float32)
            st = _ENSEMBLE_STYLES[tag]
            ax1.plot(np.arange(1, arr.size + 1), arr,
                     label=st["label"], color=st["color"],
                     lw=st["lw"], zorder=st["zorder"])
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
    # Figure 2: Contact probability matrices (ref + diffs)
    # ------------------------------------------------------------------
    contact_tags = [t for t in ("ref", "af", "bia", "unb")
                    if f"{t}_contact_prob_npy" in artifacts]
    if contact_tags:
        contact_mats = {}
        for tag in contact_tags:
            p = artifacts[f"{tag}_contact_prob_npy"]
            if Path(p).exists():
                contact_mats[tag] = np.load(p)

        if contact_mats:
            diff_tags = [t for t in ("af", "bia", "unb") if t in contact_mats]
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
                if "ref" in contact_mats:
                    diff = contact_mats[tag] - contact_mats["ref"]
                    vlim = max(abs(diff.min()), abs(diff.max()), 0.01)
                    im = axes2[col_i].imshow(diff, vmin=-vlim, vmax=vlim,
                                             cmap=diff_cmap, origin="lower")
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

    # Return the SA-prob plot as primary path (contact plot is also saved)
    primary = out_dir / "sasa_sa_prob.png"
    if not primary.exists():
        # fallback: write a blank placeholder so runner doesn't crash
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No SASA data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(primary, dpi=100)
        plt.close(fig)

    return primary
