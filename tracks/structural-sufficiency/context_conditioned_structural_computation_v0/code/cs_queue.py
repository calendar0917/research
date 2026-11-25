"""Sequential per-GPU queue for CSC v0 (one training process per GPU).

The queue order is fixed by the protocol (``cs_common.QUEUE``) and is passed in
explicitly, so nothing can be re-ordered because of an intermediate result.  The
queue never shortens a started unit: either the unit gets its full horizon or it
is not started.  If the wall deadline is reached, the remaining units are
recorded as ``NOT_STARTED`` with the reason.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import cs_common as C

C.setup_paths()

import torch  # noqa: E402

import cs_data as D  # noqa: E402
import cs_train as T  # noqa: E402


def unit_dir(dataset: str, arm: str) -> Path:
    return C.RUNS_DIR / f"{dataset.replace('ogbg-', '')}_{arm}_seed{C.SEED}"


def parse_units(text: str) -> list[tuple[str, str]]:
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        dataset, arm = chunk.split(":")
        if dataset not in C.DATASETS or arm not in C.ARMS:
            raise KeyError(chunk)
        out.append((dataset, arm))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", required=True, help="dataset:arm,dataset:arm,...")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-seconds", type=float, default=5400.0,
                        help="wall clock available to this queue (hard limit of the round)")
    parser.add_argument("--report-reserve-seconds", type=float, default=1200.0,
                        help="time kept free for analysis and reporting, never used by training")
    args = parser.parse_args(argv)
    units = parse_units(args.units)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.log("CUDA unavailable; refusing to start a queue")
        return 2
    t0 = time.perf_counter()
    train_budget = float(args.wall_seconds) - float(args.report_reserve_seconds)
    records: list[dict[str, Any]] = []
    for dataset, arm in units:
        out_dir = unit_dir(dataset, arm)
        existing = out_dir / "run.json"
        if existing.is_file():
            payload = C.read_json(existing)
            if payload.get("status") == "COMPLETED":
                C.log(f"skip {dataset} {arm}: already COMPLETED at {out_dir}")
                records.append({"dataset": dataset, "arm": arm, "status": "REUSED", "out_dir": C.track_rel(out_dir)})
                continue
        elapsed = time.perf_counter() - t0
        remaining = train_budget - elapsed
        # one lipo unit needs ~15 min at worst on this GPU; require at least that much slack
        need = 900.0 if dataset == "ogbg-mollipo" else 300.0
        if remaining < need:
            C.log(f"stop queue before {dataset} {arm}: {remaining:.0f}s of training budget left")
            records.append({"dataset": dataset, "arm": arm, "status": "NOT_STARTED", "reason": "WALL_BUDGET"})
            continue
        bundle = D.load_bundle(dataset)
        run = T.train_unit(
            bundle,
            arm,
            out_dir=out_dir,
            device=device,
            epochs=int(C.RECIPE["epochs"]),
            soft_seconds=remaining,
            hard_seconds=remaining,
        )
        records.append(
            {
                "dataset": dataset,
                "arm": arm,
                "status": run["status"],
                "best_epoch": run["best_epoch"],
                "best_valid_rmse_raw": run["best_valid_rmse_raw"],
                "gpu_hours": run["gpu_hours"],
                "out_dir": C.track_rel(out_dir),
            }
        )
        stamp = C.utcnow().replace(":", "").replace("-", "")[:15]
        C.write_json(C.ANALYSIS_DIR / f"queue_{device.type}{device.index or 0}_{stamp}.json", {"units": records, "created_utc": C.utcnow()})
    C.log("queue finished: " + ", ".join(f"{r['dataset']}:{r['arm']}={r['status']}" for r in records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
