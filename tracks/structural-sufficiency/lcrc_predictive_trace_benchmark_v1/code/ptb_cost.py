"""Hardware cost measurement: short timed CUDA runs for every unit family.

Writes ``analysis/cost_measurement.json`` with the measured seconds per epoch
for each ``(family, dataset, model)``.  Only these measurements may be used to
choose the TU cross-validation scheme (protocol §8.2); no score enters the
choice.

Each measurement is an ordinary short smoke run of the *real* training script
with a 3-epoch cap, a tiny data subset and ``smoke=True``; nothing measured here
is reported as a result.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import ptb_common as C
import ptb_data as D
import ptb_train_mol as MOL
import ptb_train_tu as TU


def _mol_cost(dataset: str, model: str, epochs: int, device: torch.device) -> dict[str, Any]:
    out_dir = C.RUNS / "cost" / C.mol_run_name(dataset, model, 0)
    t0 = time.perf_counter()
    rec = MOL.train_unit(
        dataset, model, 0, out_dir, device, epochs=epochs, smoke=True, force=True, subset=False
    )
    return {
        "family": "molecule",
        "dataset": dataset,
        "model": model,
        "epochs_measured": int(epochs),
        "mean_epoch_seconds": float(rec["timing"]["mean_epoch_seconds"]),
        "train_wall_seconds": float(rec["timing"]["train_wall_seconds"]),
        "process_wall_seconds": float(rec["timing"]["process_wall_seconds"]),
        "peak_allocated_bytes": int(rec["timing"]["gpu_peak_allocated_bytes"]),
        "n_train_full": int(rec["n_train"]),
        "wall_seconds": float(time.perf_counter() - t0),
    }


def _tu_cost(
    dataset: str, model: str, epochs: int, device: torch.device, n_splits: int, fold: int
) -> dict[str, Any]:
    out_dir = C.RUNS / "cost" / C.tu_run_name(dataset, model, fold, 0)
    t0 = time.perf_counter()
    prep_t = time.perf_counter()
    D.prepare_fold(dataset, int(n_splits), int(fold), 20261102)
    prep_seconds = time.perf_counter() - prep_t
    rec = TU.train_unit(
        dataset,
        model,
        fold,
        0,
        out_dir,
        device,
        n_splits=int(n_splits),
        fold_seed=20261102,
        epochs=epochs,
        smoke=True,
        force=True,
    )
    return {
        "family": "tu",
        "dataset": dataset,
        "model": model,
        "fold": int(fold),
        "n_splits": int(n_splits),
        "epochs_measured": int(epochs),
        "mean_epoch_seconds": float(rec["timing"]["mean_epoch_seconds"]),
        "train_wall_seconds": float(rec["timing"]["train_wall_seconds"]),
        "process_wall_seconds": float(rec["timing"]["process_wall_seconds"]),
        "peak_allocated_bytes": int(rec["timing"]["gpu_peak_allocated_bytes"]),
        "n_train": int(rec["n_train"]),
        "fold_cache_build_seconds": float(prep_seconds),
        "wall_seconds": float(time.perf_counter() - t0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--tu-splits", type=int, default=5)
    ap.add_argument("--out", default=str(C.ANALYSIS / "cost_measurement.json"))
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for dataset in C.MOLECULE_ORDER:
        for model in C.MODELS:
            C.log(f"cost measure molecule {dataset}/{model}")
            rows.append(_mol_cost(dataset, model, int(args.epochs), device))
    for dataset in C.TU_DATASETS:
        for model in C.MODELS:
            C.log(f"cost measure TU {dataset}/{model}")
            rows.append(_tu_cost(dataset, model, int(args.epochs), device, int(args.tu_splits), 0))
    payload = {
        "schema": "lcrc-ptb-cost-measurement-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "env": C.env_info(),
        "epochs_measured": int(args.epochs),
        "tu_splits_for_measurement": int(args.tu_splits),
        "measurement_wall_seconds": float(time.perf_counter() - t_start),
        "rows": rows,
    }
    C.write_json(Path(args.out), payload)
    C.log(f"wrote {args.out} in {payload['measurement_wall_seconds']/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
