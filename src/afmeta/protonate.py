from __future__ import annotations

from pathlib import Path
import shutil


import pdbfixer
from openmm import app
from openmm.app import PDBFile
from openmm.unit import angstroms, molar

def fix_pdb(inpdb: Path, outpdb: Path, solvated_dir: Path) -> None:
    """
    Fixes (protonates / solvates) the first model in alphaflow. 
    This needs to be performed to cleanup the pdb and to start simulation.
    Fixes performed: missing residues, missing atoms (including hydrogens) and missing terminals
    Additionally, solvates the pdb with the specified box, in line with ATLAS dataset. 
    """

    # If already solvated, reuse it
    cached = solvated_dir / inpdb.name #look for cached solvated pdb using the pdb protien name
    if cached.exists():
        outpdb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, outpdb)
        print(f"Reused solvated PDB: {cached} -> {outpdb}")
        return

    fixer = pdbfixer.PDBFixer(str(inpdb))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    # from af2rave.simulation.utils, in the SimulationBox class. I included this because I thought it was important
    '''
    PDB fixer automatically add disulfide bonds when two cys-S are close.
    First, this may not always be the case. Second, this behavior is not
    verbose in AMBER ffs as they actually have the CYM residue for disulfide-
    bonded CYS. Meanwhile, CHARMM ffs do not have CYM and modeller will
    complain. We will remove the disulfide bond here.
    '''
    modeller = app.Modeller(fixer.topology, fixer.positions)
    ds_bonds = []
    for bond in modeller.topology.bonds():
        if bond.atom1.name == 'SG' and bond.atom2.name == 'SG':
            ds_bonds.append(bond)
    modeller.delete(ds_bonds)
    
    #change the topology here. 
    fixer.topology, fixer.positions = modeller.topology, modeller.positions


    # Solvate (using parameters the same as ATLAS)
    fixer.addSolvent(
        padding=10*angstroms,
        positiveIon='Na+',
        negativeIon='Cl-',
        ionicStrength=0.15 * molar
    )

    #write the new pdb file
    with outpdb.open("w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    
    print(f"Fixed and solvated PDB: {inpdb} -> {outpdb}")

if __name__ == "__main__":
    #as a default, sovlate all the single model pdbs generated from extract_first_model.py
    indir = Path("data/alphaflow/single_pdbs")
    outdir = Path("data/alphaflow/solvated_pdbs")
    outdir.mkdir(exist_ok=True, parents=True)

    if indir.is_dir():
        for pdb in indir.glob("*.pdb"):
            fix_pdb(pdb, outdir / pdb.name)
    else:
        fix_pdb(indir, outdir / indir.name)