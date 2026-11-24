"""Run the complete read-only audit in the pre-specified order."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
ANALYSIS = ROUND_DIR / "analysis"

STEPS = (
    "ib_replay.py",
    "ib_identity.py",
    "ib_gap.py",
    "ib_topology.py",
    "ib_sensitivity.py",
    "ib_summary.py",
)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = []
    for name in STEPS:
        path = CODE_DIR / name
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run([sys.executable, str(path)], cwd=str(ROUND_DIR))
        records.append({
            "script": name,
            "returncode": int(result.returncode),
            "seconds": float(time.time() - t0),
            "ok": bool(result.returncode == 0),
        })
        if result.returncode != 0:
            break
    payload = {
        "schema": "r2-internal-binding-gap-audit-run-all-v1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": float(time.time() - started),
        "steps": records,
        "all_ok": all(rec["ok"] for rec in records),
    }
    (ANALYSIS / "run_all.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("\nrun_all:", json.dumps(payload, indent=1, sort_keys=True))
    if not payload["all_ok"]:
        raise SystemExit("audit pipeline failed")


if __name__ == "__main__":
    main()