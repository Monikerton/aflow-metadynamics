from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Dict, Any

from .inputs import CompareJobInputs
from .io import load_reference_md, load_run_dir, load_alphaflow_ensemble
from .preprocess import standardize_to_reference
from .align import slice_to_common_ca
from .registry import METRICS


def run_compare_job(
    *,
    out_dir: Path,
    ref_top: Path,
    ref_traj: Path,
    biased_run_dir: Path,
    unbiased_run_dir: Optional[Path] = None,
    alphaflow_pdb: Optional[Path] = None,
    metrics: Optional[Iterable[str]] = None,
    atom_sel: str = "protein and name CA",
    superpose: bool = True,
    temperature_K: float = 300.0,
    energy_unit: str = "kj",
) -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    ref = load_reference_md(ref_top=ref_top, ref_traj=ref_traj)
    biased = load_run_dir(run_dir=biased_run_dir, kind="biased", label="biased")
    unbiased = load_run_dir(run_dir=unbiased_run_dir, kind="unbiased", label="unbiased") if unbiased_run_dir else None
    alphaflow = load_alphaflow_ensemble(pdb_path=alphaflow_pdb) if alphaflow_pdb else None

    # Standardize everything into reference space
    print("Slicing all datasets to common CA set...")
    ref_ca, sliced, keys = slice_to_common_ca(
        ref=ref.traj,
        others=[
            ("biased", biased.traj),
            ("unbiased", unbiased.traj if unbiased else None),
            ("alphaflow", alphaflow.traj if alphaflow else None),
        ],
    )

    def _n_ca(t): return sum(a.name == "CA" for a in t.topology.atoms)

    kept = len(keys)
    print(f"[align] kept={kept}  ref={_n_ca(ref.traj)}  biased={_n_ca(biased.traj)}"
        f"{'' if not unbiased else f'  unb={_n_ca(unbiased.traj)}'}"
        f"{'' if not alphaflow else f'  af={_n_ca(alphaflow.traj)}'}")
    print(f"[align] keys: {keys[0]} ... {keys[-1]}")


    biased_ca = sliced["biased"]
    unbiased_ca = sliced["unbiased"]
    alphaflow_ca = sliced["alphaflow"]

    # Now standardize in the sliced CA space
    print("Standardizing to reference space...")
    ref_std = standardize_to_reference(ref_ca, ref_ca, atom_sel="name CA", superpose=False)
    biased_std = standardize_to_reference(biased_ca, ref_ca, atom_sel="name CA", superpose=superpose)
    unbiased_std = standardize_to_reference(unbiased_ca, ref_ca, atom_sel="name CA", superpose=superpose) if unbiased_ca else None
    alphaflow_std = standardize_to_reference(alphaflow_ca, ref_ca, atom_sel="name CA", superpose=superpose) if alphaflow_ca else None

    job = CompareJobInputs(
        out_dir=out_dir,
        reference_md=ref,
        biased=biased,
        unbiased=unbiased,
        alphaflow=alphaflow,
        ref_std=ref_std,
        biased_std=biased_std,
        unbiased_std=unbiased_std,
        alphaflow_std=alphaflow_std,
        temperature_K=float(temperature_K),
        energy_unit=energy_unit,
    )

    metric_names = list(metrics) if metrics is not None else ["pca_compare"]
    print(f"Running metrics: {metric_names}")

    results: Dict[str, Any] = {}
    for name in metric_names:
        if name not in METRICS:
            raise ValueError(f"Unknown metric '{name}'. Available: {sorted(METRICS)}")
        m_out = out_dir / name
        m_out.mkdir(exist_ok=True)
        results[name] = METRICS[name](job, m_out)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ref_top": str(Path(ref_top).resolve()),
        "ref_traj": str(Path(ref_traj).resolve()),
        "biased_run_dir": str(Path(biased_run_dir).resolve()),
        "unbiased_run_dir": str(Path(unbiased_run_dir).resolve()) if unbiased_run_dir else None,
        "alphaflow_pdb": str(Path(alphaflow_pdb).resolve()) if alphaflow_pdb else None,
        "atom_sel": atom_sel,
        "superpose": superpose,
        "metrics": metric_names,
        "results": results,
        "temperature_K": temperature_K,
        "energy_unit": energy_unit,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir
