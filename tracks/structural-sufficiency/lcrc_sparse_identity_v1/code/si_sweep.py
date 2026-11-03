"""Sequential sweep runner: one process per GPU runs several formal units.

Only the loop lives here; every unit is executed by ``si_train.train_unit`` with the
frozen recipe and writes its own ``run.json`` / ``selection.json`` / ``checkpoint.pt``.
A unit that already has a record is never overwritten (``train_unit`` refuses), so an
interrupted sweep can be inspected instead of silently restarted.

The frozen branch-input matrices are built once per arm inside this process.  They are
a deterministic function of the frozen vocabulary and the production caches, so reuse
across seeds cannot change them.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Sequence

import torch

import si_common as C
import si_train as TRAIN


def parse_units(spec: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        arm, seed = chunk.split(":")
        if arm not in C.NEURAL:
            raise ValueError(f"unknown arm {arm!r}")
        out.append((arm, int(seed)))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run several formal units sequentially")
    ap.add_argument("--units", required=True, help='e.g. "A0:0,A0:1,WIDE_TRUE:0"')
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    units = parse_units(args.units)
    device = torch.device(args.device)
    C.log(f"sweep start: {units} on {device}")
    cache: dict[str, Any] = {}
    results = []
    for arm, seed in units:
        if arm not in cache:
            t0 = time.perf_counter()
            cache[arm] = TRAIN.load_arm_features(arm)
            C.log(f"branch matrices for {arm} built in {time.perf_counter() - t0:.1f}s")
        feats_train, feats_valid, meta = cache[arm]
        record = TRAIN.train_unit(
            arm,
            int(seed),
            C.run_dir(arm, int(seed)),
            device,
            feats_train=feats_train,
            feats_valid=feats_valid,
            feature_meta=meta,
            epochs=None,
            smoke=False,
            force=False,
        )
        results.append(
            {
                "unit": C.run_name(arm, int(seed)),
                "best_epoch": int(record["checkpoint"]["best_epoch"]),
                "valid_roc_auc": float(record["validation"]["metrics"]["roc_auc"]),
                "gpu_hours": float(record["timing"]["gpu_hours_estimate"]),
            }
        )
    C.log(f"sweep done: {C.stable_json({'units': results})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
