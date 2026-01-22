#!/usr/bin/env python3
"""
Extract the first model from AlphaFlow multi-model PDBs using MDTraj.

AlphaFlow outputs PDB ensembles (multiple MODEL/ENDMDL blocks).
This script loads them as trajectories and writes only frame 0
as a single-model PDB for downstream simulation.

** This does get rid of the masses, but doesn't matter for the PDB.
"""

from pathlib import Path
import mdtraj as md
import sys

indir = Path("") #path to alphaflow multi-model pdbs
outdir = Path("")

print(indir.is_dir())

def extract_first_model(inp: Path, out: Path) -> None:
    traj = md.load(str(inp))
    traj[0].save_pdb(str(out))
    print(f"Extracted first model: {inp} -> {out}")


if __name__ == "__main__":
    outdir.mkdir(exist_ok=True, parents=True)

    if indir.is_dir():
        for pdb in indir.glob("*.pdb"):
            extract_first_model(pdb, outdir / pdb.name)
    else:
        extract_first_model(indir, outdir / indir.name)
