"""Build the Phase 1b run manifest (TSV) from rr status + local run JSONs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = REPO / "tracks/structural-sufficiency/runs/phase1b"
SMOKE = REPO / "tracks/structural-sufficiency/runs/phase1b_smoke_gpu"
OUT = REPO / "tracks/structural-sufficiency/results/phase1b/run_manifest.tsv"

# experiment -> list of (dataset, model, seed, label)
JOBS: dict[str, list[tuple[str, str, int, str | None]]] = {
    "ssrc-p1b-prep": [],
    "ssrc-p1b-smoke-lipo": [("lipo", m, 0, "smoke") for m in ["S0", "R1", "M1", "M2", "G_ref"]],
    "ssrc-p1b-smoke-zinc": [("zinc", m, 0, "smoke") for m in ["S0", "R1", "M1", "M2", "G_ref"]],
    "ssrc-p1b-zinc-S0-s0": [("zinc", "S0", 0, None)],
    "ssrc-p1b-lipo-S0-s0": [("lipo", "S0", 0, None)],
    "ssrc-p1b-zinc-a2": [("zinc", m, 0, None) for m in ["R1", "M1", "M2", "G_ref"]],
    "ssrc-p1b-lipo-a2": [("lipo", m, 0, None) for m in ["R1", "M1", "M2", "G_ref"]],
    "ssrc-p1b-zinc-B1-rewire": [("zinc", "R1", 0, "rewire")],
    "ssrc-p1b-zinc-B2-M1long": [("zinc", "M1", 0, "long")],
    "ssrc-p1b-zinc-S0-s1": [("zinc", "S0", 1, None)],
    "ssrc-p1b-zinc-R1-s1": [("zinc", "R1", 1, None)],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(exp: str) -> dict:
    p = subprocess.run(["rr", "status", exp, "--json"], cwd=REPO, capture_output=True, text=True)
    d = json.loads(p.stdout)
    if not d.get("ok"):
        raise RuntimeError(f"rr status failed for {exp}: {d}")
    return d


def run_path(dataset: str, model: str, seed: int, label: str | None) -> Path:
    suffix = ("_smoke" if label == "smoke" else "") + (f"_{label}" if label and label != "smoke" else "")
    base = SMOKE if label == "smoke" else RUNS
    return base / dataset / f"{model}_seed{seed}{suffix}.json"


def main() -> int:
    rows = []
    for exp, runs in JOBS.items():
        st = status(exp)
        started = datetime.fromisoformat(st["started_at"])
        completed = datetime.fromisoformat(st["completed_at"]) if st.get("completed_at") else None
        job_s = (completed - started).total_seconds() if completed else None
        for dataset, model, seed, label in runs:
            path = run_path(dataset, model, seed, label)
            if not path.exists():
                rows.append({
                    "phase": "smoke" if label == "smoke" else "",
                    "experiment": exp, "run_id": st["run_id"], "dataset": dataset,
                    "model": model, "seed": seed, "label": label or "",
                    "gpu_seconds": "", "job_seconds": job_s, "best_epoch": "",
                    "monitor_mae": "", "params": "", "step_ms": "",
                    "commit": st.get("commit", ""), "output_sha256": "",
                    "output": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                })
                continue
            d = json.loads(path.read_text())
            phase = "smoke" if label == "smoke" else ("B" if label in ("rewire", "long") else ("D" if seed == 1 else "A"))
            rows.append({
                "phase": phase, "experiment": exp, "run_id": st["run_id"],
                "dataset": dataset, "model": model, "seed": seed, "label": label or "",
                "gpu_seconds": round(float(d.get("gpu_seconds") or 0), 3),
                "job_seconds": job_s,
                "best_epoch": d.get("best_epoch"),
                "monitor_mae": round(d["best_metrics"]["monitor"]["mae"], 6),
                "params": d.get("parameters"),
                "step_ms": round(float(d.get("step_time_ms_median") or 0), 3),
                "commit": d.get("environment", {}).get("git_commit", "")[:12],
                "output_sha256": sha256(path),
                "output": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            })
    cols = ["phase", "experiment", "run_id", "dataset", "model", "seed", "label",
            "gpu_seconds", "job_seconds", "best_epoch", "monitor_mae", "params",
            "step_ms", "commit", "output_sha256", "output"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {OUT} with {len(rows)} rows")
    total = sum(float(r["gpu_seconds"]) for r in rows if r["gpu_seconds"] != "")
    print(f"gpu_seconds total (run-level, includes smoke) = {total:.1f} = {total/3600:.3f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
