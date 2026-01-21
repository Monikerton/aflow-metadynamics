from __future__ import annotations

from typing import Callable, Dict

from .metrics import pca_compare

# Each metric: compute(job_inputs, out_dir) -> dict
METRICS: Dict[str, Callable] = {
    "pca_compare": pca_compare.compute,
}
