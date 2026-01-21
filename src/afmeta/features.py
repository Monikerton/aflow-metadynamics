from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Literal

import numpy as np
import mdtraj as md
from sklearn.decomposition import PCA

FeatureKind = Literal["distance", "angle", "dihedral"]


@dataclass
class FeatureDef:
    kind: FeatureKind
    atom_indices: Tuple[int, ...]
    label: str


@dataclass
class FeatureTransformResult:
    model: object
    feature_defs: List[FeatureDef]
    labels: List[str]
    features: np.ndarray
    keep_idx: np.ndarray
    full_features_count: int
    periodic: bool
    min_seq_sep: int
    kind: FeatureKind = "distance"


def build_ca_pairs(topology, *, min_seq_sep: int = 3) -> List[Tuple[int, int]]:
    ca_atoms = [a for a in topology.atoms if a.name == "CA" and a.residue.is_protein]

    pairs: List[Tuple[int, int]] = []
    for i, ai in enumerate(ca_atoms):
        for aj in ca_atoms[i + 1 :]:
            if abs(ai.residue.index - aj.residue.index) > min_seq_sep: #skip nearby residues
                pairs.append((ai.index, aj.index))
    return pairs


def build_distance_feature_defs(
    pairs: Sequence[Tuple[int, int]], *, prefix: str = "cc_r"
) -> List[FeatureDef]:
    return [
        FeatureDef(kind="distance", atom_indices=(i, j), label=f"{prefix}{k}")
        for k, (i, j) in enumerate(pairs, start=1)
    ]


def select_top_m_by_variance(
    features: np.ndarray,
    feature_defs: Sequence[FeatureDef],
    *,
    top_m: int,
) -> Tuple[np.ndarray, List[FeatureDef], np.ndarray]:
    var = np.var(features, axis=0)
    keep_idx = np.argsort(var)[::-1][:top_m]
    return features[:, keep_idx], [feature_defs[i] for i in keep_idx], keep_idx


def fit_pca(
    features: np.ndarray,
    *,
    n_components: int = 2,
    random_state: Optional[int] = 0,
) -> PCA:
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(features)
    return pca


def prepare_pca_features(
    traj,
    *,
    top_m: int,
    min_seq_sep: int = 3,
    periodic: bool = False,
    n_components: int = 2,
    random_state: Optional[int] = 0,
    label_prefix: str = "cc_r",
) -> FeatureTransformResult:
    pairs = build_ca_pairs(traj.topology, min_seq_sep=min_seq_sep)
    feature_defs_all = build_distance_feature_defs(pairs, prefix=label_prefix)
    features_all = compute_features(traj, feature_defs_all, periodic=periodic)

    features, feature_defs, keep_idx = select_top_m_by_variance( #since we call this here, we keep the original label from the original full list. 
        features_all, feature_defs_all, top_m=top_m
    )

    pca = fit_pca(features, n_components=n_components, random_state=random_state)

    return FeatureTransformResult(
        model=pca,
        feature_defs=feature_defs,
        labels=[fd.label for fd in feature_defs],
        features=features,
        keep_idx=keep_idx,
        full_features_count=len(feature_defs_all),
        periodic=periodic,
        min_seq_sep=min_seq_sep,
        kind="distance",
    )


def compute_features(traj, feature_defs: Sequence[FeatureDef], *, periodic: bool = False) -> np.ndarray:
    kinds = {fd.kind for fd in feature_defs}

    if kinds == {"distance"}:
        pairs = np.asarray([fd.atom_indices for fd in feature_defs], dtype=int)
        return md.compute_distances(traj, pairs, periodic=periodic)

    # if kinds == {"dihedral"}:
    #     quads = np.asarray([fd.atom_indices for fd in feature_defs], dtype=int)
    #     # md.compute_dihedrals returns radians
    #     return md.compute_dihedrals(traj, quads)

    raise NotImplementedError(f"Mixed/unsupported kinds: {sorted(kinds)}")

def suggest_metad_params_from_pca(dists, pca):
    """
    A method of calculating sigma, and potentially viable numbers for grid_min, grid_max, and grid_bin
    dists: (n_frames, n_pairs) raw distances used to fit PCA (same order)
    pca:   fitted sklearn PCA (with components_, mean_, explained_variance_)
    """
    pcs = (dists - pca.mean_) @ pca.components_[:2].T   # (n_frames, 2). Transforms raw distances to PC1/PC2 space
    m = pcs.mean(axis=0)                                 # mean of PC1/PC2
    s = pcs.std(axis=0, ddof=1)                          # std  of PC1/PC2

    # SIGMA = 0.5 * std  (simple, good starting point).
    # Plumed says 1/2 to 1/3  of estimated fluctuations is good: https://www.plumed-tutorials.org/lessons/21/004/data/INSTRUCTIONS.html
    sigma = 0.5 * s

    # GRID bounds = mean ± 10 * std . Farther is better, just so that we sample the entire grid. 
    grid_min = (m - 10*s)
    grid_max = (m + 10*s)

    # Bins from spacing ≈ SIGMA/5 (guard against zero). Plumed suggests this as a viable starting point for a grid: https://www.plumed.org/doc-v2.9/user-doc/html/_m_e_t_a_d.html 
    spacing = np.maximum(sigma / 5.0, 1e-6)
    bins = np.ceil((grid_max - grid_min) / spacing).astype(int)
    bins = np.maximum(bins, 10)  # keep at least 10 bins per axis

    # Return simple dict for PLUMED
    return {
        "sigma": [float(sigma[0]), float(sigma[1])],
        "grid_min": [float(grid_min[0]), float(grid_min[1])],
        "grid_max": [float(grid_max[0]), float(grid_max[1])],
        "grid_bin": [int(bins[0]), int(bins[1])],
    }

def transform_with_feature_result(
    traj,
    ft: FeatureTransformResult,
) -> np.ndarray:
    """
    Project `traj` into the PCA space defined by `ft` (same feature_defs + fitted PCA).
    Returns (n_frames, n_components) coordinates.
    """
    X = compute_features(traj, ft.feature_defs, periodic=ft.periodic)  # (n_frames, top_m)
    return ft.model.transform(X)
