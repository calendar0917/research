"""Build `analysis/gpu_ledger.json` from `rr jobs --json` (local-only helper).

Usage (from the repo root, after `rr jobs --json > /tmp/jobs.json`):

    uv run python -m tracks.structural-sufficiency.zinc_rooted_moment_v1.code.rmc_ledger /tmp/jobs.json

GPU hours are wall-clock seconds between `started_at` and `completed_at`
multiplied by the number of visible GPUs recorded by `rr` (0 for CPU jobs,
1 for the formal single-GPU runs).  Queue wait is not charged, matching the
`zinc_native_relation_v1` ledger.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))

import rmc_common as C  # noqa: E402

PREFIXES = ("rmc-",)


def _dt(s: str | None):
    return datetime.fromisoformat(s) if s else None


def main(path: str) -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    total = 0.0
    for job in payload["jobs"]:
        exp = str(job.get("experiment") or "")
        if not exp.startswith(PREFIXES):
            continue
        started, done = _dt(job.get("started_at")), _dt(job.get("completed_at"))
        seconds = (done - started).total_seconds() if (started and done) else 0.0
        visible = str(job.get("cuda_visible_devices") or "").strip()
        gpus = 0 if visible in ("", "-1", "None") else len(visible.split(","))
        gpu_h = seconds / 3600.0 * gpus
        total += gpu_h
        rows.append({
            "experiment": exp,
            "run_id": job.get("run_id"),
            "state": job.get("state"),
            "exit_code": job.get("exit_code"),
            "pool": job.get("pool"),
            "commit": job.get("commit"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "wall_minutes": round(seconds / 60.0, 2),
            "gpus": gpus,
            "gpu_hours": round(gpu_h, 4),
        })
    rows.sort(key=lambda r: (r["started_at"] or "", r["experiment"]))
    out = {
        "protocol_id": C.PROTOCOL_ID,
        "kind": "gpu_ledger",
        "source": "rr jobs --json (process backend on res); gpu_h = wall x visible GPUs",
        "include_failures": True,
        "total_gpu_hours": round(total, 4),
        "budget_gpu_hours": 5.0,
        "rows": rows,
    }
    C.write_json(C.ANALYSIS_DIR / "gpu_ledger.json", out)
    print(json.dumps({"total_gpu_hours": out["total_gpu_hours"], "n_jobs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
