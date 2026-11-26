"""One-shot formal driver: gates first, training only if every gate passed.

``ss_checks`` writes ``analysis/checks.json`` + ``analysis/preflight_timing.json``
and returns non-zero if any gate fails; ``ss_train`` then runs the single formal
training unit.  Running both in one execution unit keeps the gate and the run in
the same provenance record; the gate result is never bypassed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import ss_checks
import ss_common as R
import ss_train


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(R.arm_run_dir()))
    parser.add_argument("--analysis-dir", default=str(R.ANALYSIS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--remaining-seconds", type=float, required=True)
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    gate = ss_checks.main(
        [
            "--out-dir", str(Path(args.analysis_dir)),
            "--device", str(args.device),
            "--remaining-seconds", str(float(args.remaining_seconds)),
        ]
    )
    gate_seconds = float(time.perf_counter() - t0)
    R.C.log(f"formal driver: gate rc={gate} after {gate_seconds:.1f}s")
    if gate != 0:
        R.C.log("formal driver: gate failed, formal training NOT started")
        return gate

    t1 = time.perf_counter()
    rc = ss_train.main(
        [
            "--out-dir", str(Path(args.out_dir)),
            "--device", str(args.device),
            "--soft-seconds", str(float(args.soft_seconds)),
            "--hard-seconds", str(float(args.hard_seconds)),
        ]
    )
    training_seconds = float(time.perf_counter() - t1)
    R.C.log(
        f"formal driver: training rc={rc} after {training_seconds:.1f}s "
        f"(unit total {float(time.perf_counter() - t0):.1f}s)"
    )
    run_json = Path(args.out_dir) / "run.json"
    if run_json.is_file():
        payload = R.read_json(run_json)
        payload["formal_driver"] = {
            "gate_seconds": gate_seconds,
            "gate_rc": int(gate),
            "training_seconds": training_seconds,
            "unit_total_seconds": float(time.perf_counter() - t0),
            "remaining_seconds_declared": float(args.remaining_seconds),
            "soft_seconds": float(args.soft_seconds),
            "hard_seconds": float(args.hard_seconds),
            "gate_and_training_in_one_unit": True,
        }
        R.write_json(run_json, payload)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
