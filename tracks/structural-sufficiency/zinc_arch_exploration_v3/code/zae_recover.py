"""Recover the curve of a run that was stopped before its declared horizon.

A cancelled run never reaches the code path that writes ``curve.csv`` and
``train.json``, but every epoch line is still in the job's stdout log and the
best checkpoint is on disk.  This script rebuilds exactly those two artifacts
from the log plus ``best.pt`` and marks the run as truncated, so nothing in the
report can mistake a stopped run for a completed one.

Only used for budget decisions (a run whose best-valid has clearly stopped
improving and cannot reach the reference level); completion is still the
default for every declared trajectory.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
for _p in (str(_HERE), str(_EXP.parent / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

LINE = re.compile(
    r"^\[(?P<arm>[A-Za-z_]+) s(?P<seed>\d+)\] ep=(?P<epoch>\d+) valid=(?P<valid>[0-9.]+) "
    r"online=(?P<online>[0-9.]+) lr=(?P<lr>[0-9.eE+-]+) best=(?P<best>[0-9.]+)@(?P<best_ep>\d+) "
    r"(?P<sec>[0-9.]+)s"
)


def recover(run_dir: Path, reason: str) -> dict:
    log = run_dir / "stdout.log"
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    curve = []
    for line in lines:
        m = LINE.match(line.strip())
        if not m:
            continue
        curve.append({
            "epoch": int(m.group("epoch")),
            "valid_mae": float(m.group("valid")),
            "online_train_mae": float(m.group("online")),
            "lr": float(m.group("lr")),
            "seconds": float(m.group("sec")),
            "steps": None,
        })
    if not curve:
        raise SystemExit(f"{log}: no epoch lines found")
    epochs_run = curve[-1]["epoch"]
    wall = float(sum(c["seconds"] for c in curve))
    steps = 157  # 10,000 train molecules at batch 64; verified against the completed A run
    for c in curve:
        c["steps"] = steps
    best = min(curve, key=lambda c: (c["valid_mae"], c["epoch"]))
    blob = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if int(blob["epoch"]) != int(best["epoch"]) or abs(float(blob["valid_mae"]) - best["valid_mae"]) > 1e-6:
        raise SystemExit("best.pt does not match the best epoch line in the log")
    C.write_curve_csv(run_dir / "curve.csv", curve)
    payload = {
        "round": "zinc-arch-exploration-v3",
        "protocol_id": "zinc-arch-exploration-v3",
        "candidate": blob["candidate"],
        "seed": int(blob["seed"]),
        "status": "CANCELLED_BY_BUDGET",
        "cancel_reason": reason,
        "epochs_declared": 1200,
        "epochs_run": epochs_run,
        "declared_steps_per_epoch": steps,
        "batch_size": 64,
        "updates": epochs_run * steps,
        "wall_seconds_logged": wall,
        "gpu_hours_logged": wall / 3600.0,
        "best_valid": {"valid_mae": float(best["valid_mae"]), "epoch": int(best["epoch"])},
        "last_epoch_line": {k: curve[-1][k] for k in ("epoch", "valid_mae", "lr", "online_train_mae")},
        "best_checkpoint_sha256": C.sha256_file(run_dir / "best.pt"),
        "recovered_from": str(log.name),
        "recovered_at": C.utcnow(),
        "note": (
            "the run was stopped before its horizon by an explicit budget decision; "
            "curve.csv and this record are reconstructed from the epoch log and the "
            "best checkpoint, and the run is reported as truncated, not as completed"
        ),
    }
    C.write_json(run_dir / "train_partial.json", payload)
    C.log(json.dumps(payload, indent=1)[:1200])
    return payload


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--reason", default="best-valid stopped improving and cannot reach the reference level")
    args = p.parse_args(argv)
    recover(Path(args.run_dir).resolve(), args.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
