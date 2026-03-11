from __future__ import annotations

from typing import Callable, Dict

from .metrics import pca_compare, rmsf, ensemble_rmsd, atomic_emd, sasa_contact, precision_recall
from .plotters import (pca_compare_plot, rmsf_plot,
                       ensemble_rmsd_plot, atomic_emd_plot,
                       sasa_contact_plot, precision_recall_plot)

# Each metric: compute(job_inputs, out_dir) -> dict
METRICS: Dict[str, Callable] = {
    "pca_compare": pca_compare.compute,
    "rmsf": rmsf.compute,
    "ensemble_rmsd": ensemble_rmsd.compute,
    "atomic_emd": atomic_emd.compute,
    "sasa_contact": sasa_contact.compute,
    "precision_recall": precision_recall.compute,
}

PLOTTERS: Dict[str, Callable] = {
    "pca_compare": pca_compare_plot.plot,
    "rmsf": rmsf_plot.plot,
    "ensemble_rmsd": ensemble_rmsd_plot.plot,
    "atomic_emd": atomic_emd_plot.plot,
    "sasa_contact": sasa_contact_plot.plot,
    "precision_recall": precision_recall_plot.plot,
}
