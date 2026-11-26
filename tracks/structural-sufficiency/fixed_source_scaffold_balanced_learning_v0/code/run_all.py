"""One-command reproduction of the round: freeze -> checks -> train -> analyse.

Order and stop rules (PROTOCOL sections 4 and 6):

1. ``sb_freeze``  scaffolds + weights + the weight-contrast gate.
   Exit code 4 means ``INSUFFICIENT_WEIGHT_CONTRAST``: stop, no training.
2. ``sb_checks``  the nine pre-training checks + throughput preflight.
   Exit code 5 means a check failed: stop, no training.
3. ``sb_train``   the single formal run (360 epochs, CPU, hard wall clock).
4. ``sb_analyse`` main table, curve, per-molecule comparison, verdict.

Every stage is timed; ``analysis/run_all_log.json`` records the stage outcomes.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import sb_common as C


STAGES = (
    ("freeze", "sb_freeze.py"),
    ("checks", "sb_checks.py"),
    ("train", "sb_train.py"),
    ("analyze", "sb_analyze.py"),
)


def main() -> int:
    log: dict[str, object] = {
        "round_id": C.ROUND_ID,
        "started_utc": C.utcnow(),
        "stages": [],
    }
    t_all = time.perf_counter()
    code = 0
    for name, script in STAGES:
        cmd = [sys.executable, str(C.CODE_DIR / script)]
        t0 = time.perf_counter()
        C.log(f"stage {name}: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=False)
        entry = {
            "stage": name,
            "script": script,
            "returncode": int(proc.returncode),
            "seconds": float(time.perf_counter() - t0),
            "finished_utc": C.utcnow(),
        }
        log["stages"].append(entry)
        if proc.returncode != 0:
            code = int(proc.returncode)
            log["stopped_at"] = name
            C.log(f"stage {name} returned {proc.returncode}; stopping")
            break
    log["total_seconds"] = float(time.perf_counter() - t_all)
    log["finished_utc"] = C.utcnow()
    log["gpu_hours"] = 0.0
    Path(C.ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    C.write_json(Path(C.ANALYSIS_DIR) / "run_all_log.json", log)
    C.log(f"run_all finished in {log['total_seconds']:.1f}s (code {code})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
