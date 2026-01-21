from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import mdtraj as md

from .features import FeatureDef


AtomKey = Tuple[str, int, str]  # (chain_id, resSeq, atom_name)


def _atom_key(atom) -> AtomKey:
    res = atom.residue
    chain = res.chain

    chain_id = getattr(chain, "chain_id", None)
    if chain_id in (None, ""):
        chain_id = str(chain.index)

    resseq = getattr(res, "resSeq", None)
    if resseq is None:
        # fallback if resSeq is missing (less ideal but works)
        resseq = res.index

    return (str(chain_id), int(resseq), atom.name)


def _build_key_to_index(pdb: Path) -> Dict[AtomKey, int]:
    traj = md.load(str(pdb))
    mapping: Dict[AtomKey, int] = {}
    for a in traj.topology.atoms:
        k = _atom_key(a)
        # if duplicates exist, it's ambiguous (altlocs/etc.)
        if k in mapping:
            raise ValueError(f"Duplicate atom key {k} in {pdb}. Cannot remap uniquely.")
        mapping[k] = a.index
    return mapping


def remap_feature_defs(
    feature_defs: List[FeatureDef],
    *,
    old_pdb: Path,
    new_pdb: Path,
) -> List[FeatureDef]:
    """Remap FeatureDef.atom_indices from old_pdb topology onto new_pdb topology.

    Assumes FeatureDefs reference atoms present in both structures (true for protein CA).
    """
    old_traj = md.load(str(old_pdb))
    new_map = _build_key_to_index(new_pdb)

    remapped: List[FeatureDef] = []
    for fd in feature_defs:
        if fd.kind != "distance":
            raise NotImplementedError("Only distance features are supported for remapping right now.")
        if len(fd.atom_indices) != 2:
            raise ValueError(f"Distance feature must have 2 atoms; got {fd.atom_indices}")

        i_old, j_old = map(int, fd.atom_indices)

        ai_old = old_traj.topology.atom(i_old)
        aj_old = old_traj.topology.atom(j_old)

        ki = _atom_key(ai_old)
        kj = _atom_key(aj_old)

        if ki not in new_map:
            raise KeyError(f"Could not find atom {ki} from old_pdb in new_pdb={new_pdb}")
        if kj not in new_map:
            raise KeyError(f"Could not find atom {kj} from old_pdb in new_pdb={new_pdb}")

        remapped.append(replace(fd, atom_indices=(new_map[ki], new_map[kj])))

    return remapped
