"""Run the whole diagnosis end to end on CPU (no training, no GPU).

Order: Validation replay -> inventory/alignment -> P1 -> P2 -> P3 -> P4 -> tables.

All steps are idempotent and read-only with respect to historical rounds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import lc_common as C

CODE = Path(__file__).resolve().parent
LOG_PATH = C.ANALYSIS / "run_log.json"


def _run(module: str, *, timeout: int, args: tuple[str, ...] = ()) -> dict[str, object]:
    t0 = time.perf_counter()
    print(f"\n=== {module} {' '.join(args)} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, str(CODE / f"{module}.py"), *args],
        cwd=str(CODE.parent),
        timeout=timeout,
        check=False,
    )
    seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{module} failed with exit code {proc.returncode}")
    return {"module": module, "args": list(args), "seconds": seconds, "exit_code": proc.returncode}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-valid", action="store_true", help="reuse existing Validation replay")
    parser.add_argument("--skip-structures", action="store_true", help="reuse the WL2 support build")
    args = parser.parse_args(argv)

    steps: list[dict[str, object]] = []
    if not args.skip_valid:
        steps.append(_run("lc_valid", timeout=3600, args=("run",)))
    steps.append(_run("lc_inventory", timeout=1800))
    steps.append(_run("lc_rank", timeout=3600))
    if not args.skip_structures:
        steps.append(_run("lc_structures", timeout=3600))
    steps.append(_run("lc_gain", timeout=3600))
    steps.append(_run("lc_macro", timeout=1800))
    steps.append(_run("lc_tables", timeout=600))

    payload = {
        "schema": "lcrc-pcba-bottleneck-diagnosis-run-log-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "created_utc": C.utcnow(),
        "device": "cpu",
        "gpu_hours_used": 0.0,
        "train_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "steps": steps,
        "total_seconds": float(sum(float(s["seconds"]) for s in steps)),
    }
    C.write_json(LOG_PATH, payload)
    print(f"\n[run_all] total {payload['total_seconds']:.1f}s -> {C.rel(LOG_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
