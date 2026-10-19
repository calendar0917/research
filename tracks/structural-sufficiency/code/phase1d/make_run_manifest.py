"""Build the Phase 1d run manifest (TSV) from rr status + local run JSONs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = REPO / "tracks/structural-sufficiency/runs/phase1d"
SMOKE = REPO / "tracks/structural-sufficiency/runs/phase1d_smoke_gpu"
OUT = REPO / "tracks/structural-sufficiency/results/phase1d/run_manifest.tsv"

# (phase, experiment, [(model, seed, label)])
JOBS: list[tuple[str, str, list[tuple[str, int, str | None]]]] = [
    ("smoke", "ssrc-p1d-smoke-zinc", [("R_incidence", 0, "smoke")]),
    ("B", "ssrc-p1d-zinc-incidence-s0", [("R_incidence", 0, None)]),
    ("C", "ssrc-p1d-zinc-R1cap-s0", [("R1_size_cap", 0, None)]),
    ("C", "ssrc-p1d-zinc-Inccap-s0", [("R_incidence_cap", 0, None)]),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(exp: str) -> dict | None:
    p = subprocess.run(["rr", "status", exp, "--json"], cwd=REPO, capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
    except Exception:
        return None
    return d if d.get("ok") else None


def run_path(model: str, seed: int, label: str | None) -> Path:
    if label == "smoke":
        return SMOKE / "zinc" / f"{model}_seed{seed}_smoke.json"
    suffix = f"_{label}" if label else ""
    return RUNS / "zinc" / f"{model}_seed{seed}{suffix}.json"


def main() -> int:
    rows = []
    job_seconds_by_exp: dict[str, float | None] = {}
    for phase, exp, runs in JOBS:
        st = status(exp)
        if st is None:
            print(f"skip (no rr status): {exp}")
            continue
        started = datetime.fromisoformat(st["started_at"])
        completed = datetime.fromisoformat(st["completed_at"]) if st.get("completed_at") else None
        job_s = (completed - started).total_seconds() if completed else None
        job_seconds_by_exp[exp] = job_s
        for model, seed, label in runs:
            path = run_path(model, seed, label)
            if not path.exists():
                print(f"missing local output: {path}")
                continue
            d = json.loads(path.read_text())
            rows.append(
                {
                    "phase": phase,
                    "experiment": exp,
                    "run_id": st["run_id"],
                    "dataset": "zinc",
                    "model": model,
                    "seed": seed,
                    "label": label or "",
                    "gpu_seconds": round(float(d.get("gpu_seconds") or 0), 3),
                    "job_seconds": job_s,
                    "best_epoch": d.get("best_epoch"),
                    "epochs_run": len(d.get("curves", [])),
                    "monitor_mae": round(d["best_metrics"]["monitor"]["mae"], 6),
                    "monitor_rmse": round(d["best_metrics"]["monitor"]["rmse"], 6),
                    "params": d.get("parameters"),
                    "step_ms": round(float(d.get("step_time_ms_median") or 0), 3),
                    "commit": d.get("environment", {}).get("git_commit", "")[:12],
                    "output_sha256": sha256(path),
                    "output": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                }
            )
    cols = [
        "phase", "experiment", "run_id", "dataset", "model", "seed", "label",
        "gpu_seconds", "job_seconds", "best_epoch", "epochs_run", "monitor_mae",
        "monitor_rmse", "params", "step_ms", "commit", "output_sha256", "output",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    run_level = sum(float(r["gpu_seconds"]) for r in rows)
    occupancy = sum(v for v in job_seconds_by_exp.values() if v is not None)
    print(f"wrote {OUT} with {len(rows)} rows")
    print(f"gpu_seconds run-level = {run_level:.1f} = {run_level/3600:.4f} h")
    print(f"job occupancy = {occupancy:.1f} s = {occupancy/3600:.4f} h over {len(job_seconds_by_exp)} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
