"""Record the execution regime and artifact hashes of a formal run.

Reads ``rr jobs --json`` so the provenance is the scheduler's own record
(commit, pool, node, driver, torch, exit code) instead of a hand-written note.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import lfa_common as C

STAGES = [
    ("fields-lcrc", "lfa-fields-lcrc", "bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh fields-lcrc",
     "results/fields/lcrc_*.jsonl + fields_manifest_lcrc.json"),
    ("fields-gine", "lfa-fields-gine", "bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh fields-gine",
     "results/fields/gine_*.jsonl + fields_manifest_gine.json"),
    ("pool", "lfa-pool", "bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh pool",
     "results/pool/pool_{main,aux}.jsonl + pool_manifest.json"),
    ("select-lcrc-main", "lfa-select-lcrc-main", "bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh select-lcrc-main",
     "results/select/select_lcrc_main.jsonl + select_manifest_lcrc_main.json"),
    ("select-lcrc-aux", "lfa-select-lcrc-aux", "bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh select-lcrc-aux",
     "results/select/select_lcrc_aux.jsonl + select_manifest_lcrc_aux.json"),
    ("select-gine-main", "lfa-select-gine-main", "bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh select-gine-main",
     "results/select/select_gine_main.jsonl + select_manifest_gine_main.json"),
    ("select-gine-aux", "lfa-select-gine-aux", "bash tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/run_remote.sh select-gine-aux",
     "results/select/select_gine_aux.jsonl + select_manifest_gine_aux.json"),
]

ANALYSIS_FILES = [
    "protocol_freeze.json", "identity.json", "gates.json", "lipo_replay.json",
    "main_summary.json", "main_stratified_by_k.csv", "main_paired_differences.csv",
    "main_concentration.csv", "main_molecule_results.csv", "decision.json",
    "aux_summary.json", "aux_endpoint_results.csv", "aux_paired_differences.csv",
    "aux_pair_predictions.csv",
]


def dirty_files() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [line[3:].strip() for line in out.stdout.splitlines() if line.strip()]


def load_jobs(path: Path | None) -> list[dict]:
    if path is None:
        out = subprocess.run(["rr", "jobs", "--json"], capture_output=True, text=True, check=True)
        return json.loads(out.stdout)["jobs"]
    return json.loads(Path(path).read_text(encoding="utf-8"))["jobs"]


def pick(jobs: Sequence[dict], experiment: str) -> dict | None:
    hits = [j for j in jobs if j.get("experiment") == experiment]
    if not hits:
        return None
    return sorted(hits, key=lambda j: j.get("created_at") or "")[-1]


def run(*, jobs_path: Path | None = None, out_dir: Path | str = C.ANALYSIS_DIR) -> dict[str, Any]:
    out_dir = Path(out_dir)
    jobs = load_jobs(jobs_path)
    stages, runs = [], {}
    gpu_seconds = 0.0
    for name, experiment, command, outputs in STAGES:
        job = pick(jobs, experiment)
        runs[experiment] = job
        record: dict[str, Any] = {"name": name, "experiment": experiment, "command": command,
                                  "outputs": outputs}
        if job is None:
            record["job"] = None
            record["status"] = "NOT_RUN"
        else:
            runtime = None
            if job.get("started_at") and job.get("completed_at"):
                from datetime import datetime
                a = datetime.fromisoformat(job["started_at"])
                b = datetime.fromisoformat(job["completed_at"])
                runtime = (b - a).total_seconds()
                if int(job.get("gpu_count") or 0) > 0:
                    gpu_seconds += runtime * int(job["gpu_count"])
            record["job"] = {
                "run_id": job.get("run_id"), "pool": job.get("pool"), "node": job.get("node"),
                "gpu_count": job.get("gpu_count"), "gpu_names": job.get("gpu_names"),
                "driver_version": job.get("driver_version"),
                "cuda_visible_devices": job.get("cuda_visible_devices"),
                "torch_version": job.get("torch_version"), "torch_cuda": job.get("torch_cuda"),
                "commit": job.get("commit"), "branch": job.get("branch"), "dirty": job.get("dirty"),
                "state": job.get("state"), "exit_code": job.get("exit_code"),
                "created_at": job.get("created_at"), "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"), "runtime_s": runtime,
                "driver_label": job.get("pool_or_host"),
            }
            record["status"] = "OK" if job.get("exit_code") == 0 else "FAILED"
        stages.append(record)
    artifacts = {}
    for name in ANALYSIS_FILES:  # run_manifest.json is excluded on purpose (self-referential)
        path = out_dir / name
        if path.is_file():
            artifacts[C.relpath(path)] = C.sha256_file(path)
    for path in sorted(C.FIELDS_DIR.glob("*.jsonl")):
        artifacts[C.relpath(path)] = C.sha256_file(path)
    for path in sorted(C.POOL_DIR.glob("*.jsonl")):
        artifacts[C.relpath(path)] = C.sha256_file(path)
    for path in sorted(C.SELECT_DIR.glob("*.jsonl")):
        artifacts[C.relpath(path)] = C.sha256_file(path)
    manifest = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utc_now(),
        "git_commit": C.git_commit(),
        "git_branch": C.git_branch(),
        "git_dirty_tracked": C.git_dirty_tracked(),
        "git_dirty_files": dirty_files(),
        "freeze_hash": C.freeze_hash(),
        "code_hashes": C.code_hashes(),
        "analysis_code": {
            "note": "hashes at report time; the execution-time hashes are frozen in protocol_freeze.json",
            "git_head": C.git_commit(),
            "git_head_now": C.git_head_now(),
            "hashes": C.code_hashes(),
            "frozen_execution_hashes": (C.read_json(C.ANALYSIS_DIR / "protocol_freeze.json") or {}).get("code_hashes", {}),
        },
        "local_environment": C.environment(),
        "stages": stages,
        "gpu_hours": float(gpu_seconds / 3600.0),
        "artifacts": artifacts,
        "notes": [
            "two execution regimes are not mixed: every stage ran on host res",
            "field/pool/select artifacts are pulled copies; the analysis stages re-hash them",
            "failed launch lfa-pool-20261117-165758-48810cb9: dead-end matched-control proposal raised inside grow_matched; "
            "fixed by returning None (rejected proposal, matching rule unchanged) and the stage was rerun as "
            "lfa-pool-20261117-170211-c72a5a9c; the resumable writer discarded the torn tail, no partial row was used",
            "rr deploy required moving four stale untracked sprint-round result files aside on the remote checkout "
            "(deployment housekeeping only; no experiment was rerun because of it)",
        ],
    }
    C.write_json(out_dir / "run_manifest.json", manifest)
    C.write_json(out_dir / "rr_runs.json", runs)
    print(f"[manifest] stages={len(stages)} gpu_hours={manifest['gpu_hours']:.4f} "
          f"artifacts={len(artifacts)}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default=None)
    parser.add_argument("--out-dir", default=str(C.ANALYSIS_DIR))
    args = parser.parse_args()
    run(jobs_path=Path(args.jobs) if args.jobs else None, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
