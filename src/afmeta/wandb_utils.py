# src/afmeta/wandb_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import threading
import time

import pandas as pd


def wandb_init(
    *,
    project: str,
    outdir: Path,
    config: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
    enabled: bool = True,
):
    """
    Minimal W&B init.
    Returns a run-like object with .log() and .finish().
    If disabled or wandb isn't installed, returns None.
    """
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        return None

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    return wandb.init(
        project=project,
        name=name,
        group=group,
        config=config or {},
        dir=str(outdir),
    )


def wandb_log(run, data: Dict[str, Any], *, step: Optional[int] = None) -> None:
    """Minimal metric logging."""
    if run is None:
        return
    if step is None:
        run.log(data)
    else:
        run.log(data, step=step)


def wandb_finish(run) -> None:
    """Finish run."""
    if run is None:
        return
    run.finish()


def start_wandb_colvar_stream(
    *,
    run,
    colvar_path: Path,
    stride_steps: int,
    poll_seconds: float = 10.0,
    prefix: str = "colvar",
) -> Optional[threading.Event]:
    """
    Stream COLVAR rows to W&B as they appear.

    Assumes your PLUMED PRINT order is:
      pc1 pc2 metad.bias metad.rbias metad.rct

    Logs at step = row_index * stride_steps (MD steps).

    Returns a stop_event (call stop_event.set()) or None if run is disabled.
    """
    if run is None:
        return None

    stop_event = threading.Event()
    last_rows = 0

    names = ["pc1", "pc2", "metad.bias", "metad.rbias", "metad.rct"]

    def loop() -> None:
        nonlocal last_rows
        while not stop_event.is_set():
            try:
                if not colvar_path.exists() or colvar_path.stat().st_size == 0:
                    time.sleep(poll_seconds)
                    continue

                df = pd.read_csv(
                    colvar_path,
                    sep=r"\s+",
                    comment="#",
                    header=None,
                    names=names,
                    engine="python",
                )
                n = len(df)
                if n <= last_rows:
                    time.sleep(poll_seconds)
                    continue

                new = df.iloc[last_rows:n]

                for i, row in new.iterrows():
                    step = int(i) * int(stride_steps)
                    wandb_log(
                        run,
                        {
                            f"{prefix}/pc1": float(row["pc1"]),
                            f"{prefix}/pc2": float(row["pc2"]),
                            f"{prefix}/metad_bias": float(row["metad.bias"]),
                            f"{prefix}/metad_rbias": float(row["metad.rbias"]),
                            f"{prefix}/metad_rct": float(row["metad.rct"]),
                        },
                        step=step,
                    )

                last_rows = n

            except Exception:
                # COLVAR can be mid-append; transient parse errors are normal
                pass

            time.sleep(poll_seconds)

    threading.Thread(target=loop, daemon=True).start()
    return stop_event
