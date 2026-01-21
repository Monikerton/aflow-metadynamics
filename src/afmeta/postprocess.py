# src/afmeta/postprocess.py
from __future__ import annotations

from pathlib import Path
import shutil
import mdtraj as md


def clean_xtc(
    *,
    xtc_in: Path,
    top_pdb: Path,
    xtc_out: Path,
    make_whole: bool = True,
    image_molecules: bool = True,
    center: bool = False,
) -> Path:
    """
    Minimal MDTraj-based cleanup:
      - make molecules whole (fix broken molecules across PBC)
      - image molecules back into the primary unit cell
      - optional centering

    Writes cleaned trajectory to xtc_out.
    """
    xtc_in = Path(xtc_in)
    top_pdb = Path(top_pdb)
    xtc_out = Path(xtc_out)
    xtc_out.parent.mkdir(parents=True, exist_ok=True)

    t = md.load(str(xtc_in), top=str(top_pdb))

    # Order matters: make whole first, then image.
    if make_whole:
        t.make_molecules_whole(inplace=True)
    if image_molecules:
        t.image_molecules(inplace=True)

    if center:
        # optional: often not necessary if you image + whole
        t.center_coordinates(inplace=True)

    t.save_xtc(str(xtc_out))
    return xtc_out


def write_canonical_outputs(
    *,
    run_xtc: Path,
    run_final_pdb: Path,
    top_pdb_for_xtc: Path,
    outdir: Path,
) -> tuple[Path, Path]:
    """
    Produces canonical filenames expected by compare/io.py:
      - fixed.xtc
      - fixed_final.pdb
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fixed_xtc = outdir / "fixed.xtc"
    fixed_final = outdir / "fixed_final.pdb"

    clean_xtc(xtc_in=run_xtc, top_pdb=top_pdb_for_xtc, xtc_out=fixed_xtc)
    shutil.copyfile(run_final_pdb, fixed_final)

    return fixed_xtc, fixed_final


if __name__ == "__main__":
    from pathlib import Path

    FORCE = True

    RUN_DIRS = [
        Path("/data/cb/scratch/dkwabiad/aflow-metadynamics/outputs/runs/biased/7jfl_C/2026-01-20/golden_p500_h0.1_s1.19-0.918_bf10.0_n100e6__87e07bb7"),
    ]

    for run_dir in RUN_DIRS:
        xtc = run_dir / "fixed.xtc"
        top = run_dir / "fixed_final.pdb"  # use this as topology
        out = run_dir / "fixed.xtc"        # overwrite in place

        if not xtc.exists():
            raise FileNotFoundError(xtc)
        if not top.exists():
            raise FileNotFoundError(top)

        if (not FORCE) :
            print(f"[skip] {run_dir}")
            continue

        tmp = run_dir / "fixed.xtc.tmp"
        print(f"[cleaning] {run_dir} ... ", end="", flush=True)
        clean_xtc(xtc_in=xtc, top_pdb=top, xtc_out=tmp, make_whole=True, image_molecules=True)
        tmp.replace(out)

        print(f"[done] {run_dir}")
