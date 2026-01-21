from __future__ import annotations

from typing import Optional

import numpy as np
import mdtraj as md
import pandas as pd


def standardize_to_reference(
    traj: md.Trajectory,
    ref: md.Trajectory,
    *,
    atom_sel: str = "protein and name CA",
    superpose: bool = True,
) -> md.Trajectory:
    """
    Select atoms and superpose traj onto ref (both sliced by same selection).
    Requires compatible selections (same atom count).
    """
    ref_idx = ref.topology.select(atom_sel)
    trj_idx = traj.topology.select(atom_sel)

    ref_s = ref.atom_slice(ref_idx)
    trj_s = traj.atom_slice(trj_idx)

    if ref_s.n_atoms != trj_s.n_atoms:
        raise ValueError(
            f"Selection '{atom_sel}' produced different atom counts: "
            f"ref={ref_s.n_atoms}, traj={trj_s.n_atoms}. Topologies likely not aligned."
        )

    if superpose:
        # Superpose to reference first frame (in sliced space)
        trj_s = trj_s[:]  # copy frames/xyz view
        trj_s.superpose(ref_s, frame=0)

    return trj_s


def rbias_per_frame_from_colvar(colvar: pd.DataFrame, frame_times_ps: np.ndarray) -> np.ndarray:
    t_cv = colvar["time"].to_numpy(float)              # PLUMED: fs
    rb   = colvar["metad.rbias"].to_numpy(float)
    t_fr = np.asarray(frame_times_ps, float) 
    o = np.argsort(t_cv)
    idx = np.searchsorted(t_cv[o], t_fr, side="left") 
    return rb[o][np.clip(idx, 0, rb.size - 1)]
