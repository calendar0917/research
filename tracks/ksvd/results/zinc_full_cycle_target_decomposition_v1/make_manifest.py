"""Build manifest.json / run_meta/summary.json / budget.json for zftd-v1.

Reads the delivered artifacts in this directory and records SHA-256, sizes and
the frozen provenance.  No model, no fitting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent

RUNS = {
    "zftd-Y-s0": {"arm": "Y", "seed": 0},
    "zftd-O-s0": {"arm": "O", "seed": 0},
    "zftd-Y-s1": {"arm": "Y", "seed": 1},
    "zftd-O-s1": {"arm": "O", "seed": 1},
}

# wall-clock seconds from each run's JSON (engine truth).
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    files = sorted(
        p for p in RESULTS.rglob("*")
        if p.is_file() and p.name not in {"manifest.json"}
        and "__pycache__" not in p.parts and p.suffix in {".py", ".md", ".json", ".csv", ".npz", ".pt"}
    )
    manifest = {
        "protocol_version": "zinc-full-cycle-target-decomposition-v1",
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "delivered_commit_context": "deployed formal commit 2cab640be142; results committed on the task branch",
        "files": {
            str(p.relative_to(RESULTS)): {
                "sha256": sha256(p),
                "bytes": int(p.stat().st_size),
            }
            for p in files
        },
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_meta = {}
    total_wall = 0.0
    for exp, info in RUNS.items():
        meta = json.loads((RESULTS / "run_meta" / f"{exp}.meta.json").read_text())
        result = json.loads((RESULTS / f"{info['arm']}_seed{info['seed']}.json").read_text())
        rt = meta.get("runtime", {})
        entry = {
            "experiment": exp,
            "arm": info["arm"],
            "seed": info["seed"],
            "run_id": meta.get("run_id"),
            "host": meta.get("host"),
            "backend": meta.get("backend"),
            "pool": meta.get("pool"),
            "node": rt.get("node"),
            "driver_version": rt.get("driver_version"),
            "gpus": rt.get("gpus"),
            "torch_version": rt.get("torch_version"),
            "torch_cuda": rt.get("torch_cuda"),
            "python_version": rt.get("python_version"),
            "git_commit": meta.get("git_commit"),
            "git_branch": meta.get("git_branch"),
            "git_dirty": meta.get("git_dirty"),
            "exit_code": rt.get("exit_code"),
            "state": meta.get("state"),
            "command": meta.get("command"),
            "wall_clock_s": result.get("wall_clock_s"),
            "seconds_per_epoch": result.get("seconds_per_epoch"),
            "peak_gpu_memory_mb": result.get("peak_gpu_memory_mb"),
            "init_state_sha256": result.get("init_state_sha256"),
            "soup_state_sha256": result.get("soup_state_sha256"),
            "h1_lambda": result.get("h1_lambda"),
            "scale_seed": result.get("scale_seed"),
        }
        run_meta[exp] = entry
        total_wall += float(result.get("wall_clock_s") or 0.0)
    (RESULTS / "run_meta" / "summary.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    budget = {
        "wall_clock_start_utc": "2026-10-03T05:46:55Z",
        "compute_cutoff_utc": "2026-10-03T08:16:55Z",
        "hard_stop_utc": "2026-10-03T08:46:55Z",
        "wall_cap_minutes": 180,
        "gpu_hours_cap": 3.5,
        "concurrency_gpus": 2,
        "local_cpu_threads_cap": 8,
        "formal_runs": {exp: {"wall_clock_s": e["wall_clock_s"], "gpu_hours": round(float(e["wall_clock_s"]) / 3600.0, 4)} for exp, e in run_meta.items()},
        "formal_gpu_hours_total": round(total_wall / 3600.0, 4),
        "remote_smokes": [
            {"experiment": "zftd-smoke", "outcome": "failed (CUDA device bug, fixed)"},
            {"experiment": "zftd-smoke2", "outcome": "failed (float32 equality check, fixed)"},
            {"experiment": "zftd-smoke3", "outcome": "completed, mechanism_ok=True"},
        ],
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    (RESULTS / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    print(f"manifest: {len(files)} files; formal gpu-hours={total_wall/3600:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())