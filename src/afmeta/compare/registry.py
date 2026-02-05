from __future__ import annotations

from typing import Callable, Dict

from .metrics import pca_compare #, ensemble_rmsd, atomic_emd, sasa_contact, precision_recall_diversity, rmsf
from .plotters import pca_compare_plot #, ensemble_rmsd, atomic_emd, sasa_contact, precision_recall_diversity, rmsf

# Each metric: compute(job_inputs, out_dir) -> dict
METRICS: Dict[str, Callable] = {
    "pca_compare": pca_compare.compute,
    # "ensemble_rmsd": ensemble_rmsd.compute,
    # "atomic_emd": atomic_emd.compute,
    # "sasa_contact": sasa_contact.compute,
    # "precision_recall_diversity": precision_recall_diversity.compute,
    # "rmsf": rmsf.compute,
}

PLOTTERS: Dict[str, Callable] = {
    "pca_compare": pca_compare_plot.plot,
}
