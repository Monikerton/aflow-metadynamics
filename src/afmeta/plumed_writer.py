from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from typing import TYPE_CHECKING
from .features import FeatureDef



def define_distance_actions_from_feature_defs(
    feature_defs: List["FeatureDef"],
    *,
    periodic: bool = True,
) -> List[str]:
    """Emit PLUMED DISTANCE lines from ordered FeatureDefs (distance-only).

    PLUMED variable labels come directly from
    `FeatureDef.label`, to ensure that
      - the base DISTANCE actions
      - the PCA COMBINE ARG list
      match. 

    Assumptions
    -----------
    - Each FeatureDef is a distance feature with exactly two 0-based atom indices.
    - PLUMED atom indices are 1-based, so indices are converted internally.

    Parameters
    ----------
    feature_defs:
        Ordered feature definitions (must match the order used to fit PCA).

    periodic:
        If True (default), distances are PBC-aware (PLUMED default).
        If False, writes distances with NOPBC.

    Returns
    -------
    lines:
        PLUMED input lines defining each distance CV.
    """
    lines: List[str] = []
    nopbc = "" if periodic else " NOPBC"

    for fd in feature_defs:
        if getattr(fd, "kind", None) != "distance":
            raise NotImplementedError(
                f"Only distance features are supported; got kind={fd.kind!r}"
            )
        atom_idx = getattr(fd, "atom_indices", None)
        if atom_idx is None or len(atom_idx) != 2:
            raise ValueError(f"Distance feature must have 2 atom indices; got {atom_idx}")

        i0, j0 = atom_idx
        i, j = int(i0) + 1, int(j0) + 1  # PLUMED is 1-based
        label = str(getattr(fd, "label", None))
        if not label:
            raise ValueError("FeatureDef.label must be a non-empty string.")

        lines.append(f"{label}: DISTANCE ATOMS={i},{j}{nopbc}\n")

    return lines


def write_plumed_pca(
    plumed_outfile: Path,
    pca,  # fitted sklearn PCA (components_.shape = (>=2, n_features))
    feature_defs: List["FeatureDef"],  # ordered feature defs used to fit PCA
    periodic: bool,
    energy_unit: str | None = None,
    *, #everything after must be defined
    include_metad: bool = True,
    well_tempered: bool = False,
    pace: int = 500,
    height: float = 1.2,
    biasfactor: float = 10.0,
    sigma: Tuple[float, float] = (0.1, 0.1),
    grid_min: Tuple[float, float] = (-5.0, -5.0),
    grid_max: Tuple[float, float] = (5.0, 5.0),
    grid_bin: Tuple[int, int] = (200, 200),
    temperature: float = 300.0,
    stride: int = 500,
    hills_path: Path | None = None,
    colvar_path: Path | None = None,
    
) -> None:
    """Write a PLUMED input that computes PC1/PC2 from distance features and runs METAD.

    Requirements
    ------------
    - `feature_defs` must be the exact ordered feature list used to construct the
      feature matrix passed to `pca.fit(...)`.
    - `pca` must expose sklearn-like `components_` and `mean_`.

    Periodicity
    ----------
    - If `periodic=True`, DISTANCE actions are PBC-aware (PLUMED default).
    - If `periodic=False`, DISTANCE actions include `NOPBC`.

    PCA embedding in PLUMED
    -----------------------
    PCA is defined as:
        PC_i(x) = w_i · (x - mean) = (w_i · x) + c_i
    where:
        c_i = - w_i · mean

    We implement this in PLUMED by:
    - creating a constant `one: CONSTANT VALUE=1.0`
    - using COMBINE on [features..., one] with coefficients [w_i..., c_i]

    Notes
    -----
    - COMBINE is written with PERIODIC=NO, appropriate for distance-derived PCs.
    """
    if len(feature_defs) == 0:
        raise ValueError("feature_defs is empty; cannot write PLUMED input.")

    labels = [fd.label for fd in feature_defs]

    # Validate PCA dimensions vs selected features
    w = np.asarray(pca.components_[:2], dtype=float)  # (2, n_features)
    mean = np.asarray(pca.mean_, dtype=float)         # (n_features,)
    n_feat = len(feature_defs)
    if w.shape[1] != n_feat or mean.shape[0] != n_feat:
        raise ValueError(
            "PCA model dimensions do not match feature_defs ordering/length: "
            f"components_.shape={getattr(pca, 'components_', None).shape}, "
            f"mean_.shape={getattr(pca, 'mean_', None).shape}, "
            f"n_features={n_feat}"
        )

    intercept = -(w @ mean)  # (2,)

    # Default output locations (alongside plumed file)
    if colvar_path is None:
        colvar_path = plumed_outfile.parent / ("COLVAR_biased.dat" if include_metad else "COLVAR_unbiased.dat")
    if include_metad and hills_path is None:
        hills_path = plumed_outfile.parent / "HILLS"

    with plumed_outfile.open("w") as f:
        if energy_unit is not None:
            assert energy_unit in {"kj/mol", "kcal/mol", "Ha", "j/mol", "eV"}, (f"Invalid PLUMED energy unit: {energy_unit!r}")
            f.write(f"UNITS ENERGY={energy_unit}\n\n")


        # Constant used to add intercept inside COMBINE
        f.write("one: CONSTANT VALUE=1.0\n")

        # 1) Base CVs (distance features), labels taken directly from FeatureDefs
        f.writelines(
            define_distance_actions_from_feature_defs(feature_defs, periodic=periodic)
        )

        # 2) PCs as linear combinations (COMBINE)
        for i in range(2):
            coeffs = ",".join(f"{c:.6f}" for c in w[i])
            f.write(
                f"pc{i+1}: COMBINE ARG={','.join(labels)},one "
                f"COEFFICIENTS={coeffs},{intercept[i]:.6f} PERIODIC=NO\n"
            )

        # 3) Metadynamics block
        if include_metad:
            f.write("\nMETAD ...\n")
            f.write("  LABEL=metad\n")
            f.write("  ARG=pc1,pc2\n")
            f.write(f"  PACE={pace}\n")
            f.write(f"  HEIGHT={height}\n")
            if well_tempered:
                f.write(f"  BIASFACTOR={biasfactor}\n")
            f.write(f"  SIGMA={sigma[0]},{sigma[1]}\n")
            f.write(f"  FILE={hills_path}\n")
            f.write(f"  GRID_MIN={grid_min[0]},{grid_min[1]}\n")
            f.write(f"  GRID_MAX={grid_max[0]},{grid_max[1]}\n")
            f.write(f"  GRID_BIN={grid_bin[0]},{grid_bin[1]}\n")
            if well_tempered:
                f.write(f"  TEMP={temperature}\n")
            f.write(f"  CALC_RCT RCT_USTRIDE={stride}\n")
            f.write("... METAD\n\n")

        # 4) Output
        if include_metad:
            f.write(
                f"PRINT ARG=pc1,pc2,metad.bias,metad.rbias,metad.rct "
                f"STRIDE={stride} FILE={colvar_path}\n"
            )
        else:
            f.write(
                f"PRINT ARG=pc1,pc2 STRIDE={stride} FILE={colvar_path}\n"
            )
    print(f"Wrote PLUMED metadynamics input to {plumed_outfile}")
