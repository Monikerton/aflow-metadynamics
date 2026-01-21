from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import mdtraj as md


ResKey = Tuple[str, int, str]  # (chain_id, resSeq, resName)


def _ca_keymap(traj: md.Trajectory) -> Dict[ResKey, int]:
    """
    Map residue identity -> CA atom index.
    Uses: (chain_id, residue.resSeq, residue.name)
    """
    out: Dict[ResKey, int] = {}
    for atom in traj.topology.atoms:
        if atom.name != "CA":
            continue
        res = atom.residue
        chain_id = str(res.chain.index)  # stable within topology
        key: ResKey = (chain_id, int(res.resSeq), str(res.name))
        # If duplicates occur (rare), keep the first deterministically
        if key not in out:
            out[key] = atom.index
    return out


def slice_to_common_ca(
    *,
    ref: md.Trajectory,
    others: Iterable[Tuple[str, Optional[md.Trajectory]]],
) -> Tuple[md.Trajectory, Dict[str, Optional[md.Trajectory]], List[ResKey]]:
    """
    Slice ref + all provided trajectories to the common set of CA residues,
    ordered deterministically by the reference topology's keys.

    Parameters
    ----------
    ref : Trajectory
        Reference trajectory (defines ordering).
    others : iterable of (name, traj_or_None)
        Other datasets to include in the intersection.

    Returns
    -------
    ref_s : Trajectory
    sliced : dict[name -> Trajectory or None]
    keys : list[ResKey]
        The residue keys kept (in ref order).
    """
    ref_map = _ca_keymap(ref)

    # start with all keys in ref, then intersect
    common = set(ref_map.keys())
    other_maps: Dict[str, Dict[ResKey, int]] = {}

    for name, tr in others:
        if tr is None:
            continue
        m = _ca_keymap(tr)
        other_maps[name] = m
        common &= set(m.keys())

    # Order by reference (deterministic)
    keys = [k for k in ref_map.keys() if k in common]

    if len(keys) == 0:
        raise ValueError("No common CA residues found across datasets.")

    ref_idx = [ref_map[k] for k in keys]
    ref_s = ref.atom_slice(ref_idx, inplace=False)

    sliced: Dict[str, Optional[md.Trajectory]] = {}
    for name, tr in others:
        if tr is None:
            sliced[name] = None
            continue
        m = other_maps[name]
        idx = [m[k] for k in keys]
        sliced[name] = tr.atom_slice(idx, inplace=False)

    return ref_s, sliced, keys

