"""Round bookkeeping: artifact manifest, checkpoint manifest view, ledger row.

Writes (all force-added to git at the end of the round):

* ``results/manifest.json`` — every artifact hash + checkpoint hashes + Test policy;
* ``results/ledger.csv``  — one row per execution unit (gate unit, formal run);
* ``results/runs/STRUCT_DROP_SCRATCH_seed0/`` — lightweight run records
  (the ``.pt`` checkpoints stay in the git-ignored ``runs/`` tree).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import ss_common as R

LIGHT_FILES = (
    "run.json",
    "train_curve.csv",
    "per_task_ap.csv",
    "initial_eval.json",
    "best_checkpoint_eval.json",
    "training_order.json",
)
LEDGER_FIELDS = [
    "unit",
    "purpose",
    "command",
    "gpu",
    "status",
    "gpu_hours",
    "wall_seconds",
    "commit",
    "note",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-gpu-hours", type=float, default=0.0)
    parser.add_argument("--gate-seconds", type=float, default=0.0)
    parser.add_argument("--probe-gpu0-seconds", type=float, default=0.0)
    parser.add_argument("--probe-gpu1-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)

    run_dir = R.arm_run_dir()
    run = R.read_json(run_dir / "run.json")
    summary = R.read_json(R.ANALYSIS / "summary.json")
    checks = R.read_json(R.ANALYSIS / "checks.json")
    preflight = R.read_json(R.ANALYSIS / "preflight_timing.json")

    light_dir = R.RESULTS / f"runs/{R.ARM}_seed{R.SEED}"
    light_dir.mkdir(parents=True, exist_ok=True)
    for name in LIGHT_FILES:
        src = run_dir / name
        if src.is_file():
            shutil.copyfile(src, light_dir / name)

    analysis_files = sorted(R.ANALYSIS.glob("*"))
    artifacts = {
        f"analysis/{p.name}": {"sha256": R.sha256_file(p), "bytes": int(p.stat().st_size)}
        for p in analysis_files
        if p.is_file()
    }
    for p in sorted(light_dir.glob("*")):
        artifacts[R.C.track_rel(p)] = {"sha256": R.sha256_file(p), "bytes": int(p.stat().st_size)}
    for name in ("PROTOCOL.md", "REPORT.md", "configs/round.json"):
        p = R.ROUND / name
        if p.is_file():
            artifacts[name] = {"sha256": R.sha256_file(p), "bytes": int(p.stat().st_size)}

    checkpoints = [
        {
            "role": row["role"],
            "epoch": int(row["epoch"]),
            "global_step": int(row["global_step"]),
            "valid_ap": row["valid_ap"],
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in R.read_json(R.ANALYSIS / "checkpoint_manifest.json")["checkpoints"]
    ]

    manifest = {
        "schema": R.SCHEMA_MANIFEST,
        "round_id": R.ROUND_ID,
        "protocol_id": R.PROTOCOL_ID,
        "arm": R.ARM,
        "seed": R.SEED,
        "epochs": R.EPOCHS,
        "parameters": int(run["parameters"]),
        "initialisation": run["initialisation"],
        "initial_state_sha256": run["initial_state_sha256"],
        "expected_initial_state_sha256": R.EXPECTED_INITIAL_STATE_SHA256,
        "initial_state_matches_historical_fixed1": run["initial_state_sha256"] == R.EXPECTED_INITIAL_STATE_SHA256,
        "structural_dropout": {
            "q": R.Q,
            "mask_seed": R.MASK_SEED,
            "active_from_step": 0,
            "mask_draws_total": int(run["structural_dropout"]["mask_draws_total"]),
            "mean_dropped_fraction": run["structural_dropout"]["mean_epoch_dropped_fraction"],
        },
        "checkpoints": checkpoints,
        "verdict": summary["verdict"],
        "headline": summary["headline"],
        "artifacts": artifacts,
        "artifacts_sha256_of_shas": R.C.sha256_bytes(
            json.dumps(artifacts, sort_keys=True).encode()
        ),
        "checks_all_passed": bool(checks["all_passed"]),
        "timing_preflight": {
            "projected_total_seconds": preflight.get("projected_total_seconds"),
            "projected_gpu_hours": preflight.get("projected_gpu_hours"),
            "step_seconds_mean": preflight.get("step_seconds_mean"),
            "measured_full_valid_seconds": preflight.get("measured_full_valid_seconds"),
            "gpu_peak_allocated_bytes": preflight.get("gpu_peak_allocated_bytes"),
            "gpu_free_bytes_at_end": preflight.get("gpu_free_bytes_at_end"),
        },
        "gpu_hours_formal_run": float(run["gpu_hours"]),
        "gpu_hours_total": float(
            float(run["gpu_hours"])
            + args.gate_gpu_hours / 3600.0
            + (args.probe_gpu0_seconds + args.probe_gpu1_seconds) / 3600.0
        ),
        "test_policy": {
            "test_scored": False,
            "test_used_for_selection": False,
            "test_labels_loaded": False,
        },
        "environment": run["environment"],
        "git_commit": run["git_commit"],
        "created_utc": R.utcnow(),
    }
    R.write_json(R.RESULTS / "manifest.json", manifest)

    rows = [
        {
            "unit": "fss-probe-g0",
            "purpose": "checks + timing probe (rejected: contended GPU)",
            "command": "ss_checks.py --device cuda (res-gpu0)",
            "gpu": "res-gpu0 / A100-40GB",
            "status": "checks passed except timing gate",
            "gpu_hours": round(args.probe_gpu0_seconds / 3600.0, 4),
            "wall_seconds": round(args.probe_gpu0_seconds, 1),
            "commit": "e066e862a7a2",
            "note": "0.0729 s/step and 19.7 GB free -> projected 90 min / 1.50 GPU-h; not used",
        },
        {
            "unit": "fss-probe-g1",
            "purpose": "checks + timing probe (selected GPU)",
            "command": "ss_checks.py --device cuda (res-gpu1)",
            "gpu": "res-gpu1 / A100-40GB",
            "status": "gates passed after code fixes (probe probe)",
            "gpu_hours": round(args.probe_gpu1_seconds / 3600.0, 4),
            "wall_seconds": round(args.probe_gpu1_seconds, 1),
            "commit": "d08e15b02f90",
            "note": "0.0421 s/step, 41.2 GB free -> projected 54 min / 0.90 GPU-h",
        },
        {
            "unit": "fss-scratch-s0",
            "purpose": "formal unit: gates (24.8 s) + 25-epoch STRUCT_DROP_SCRATCH training",
            "command": "ss_formal.py --remaining-seconds 3300 --soft-seconds 3900 --hard-seconds 4200",
            "gpu": "res-gpu1 / A100-40GB",
            "status": run["status"],
            "gpu_hours": round(float(run["gpu_hours"]), 4),
            "wall_seconds": round(float(run["gpu_seconds"]), 1),
            "commit": run["git_commit"],
            "note": f"checkpoint gate {run['status'].lower()}; best epoch {run['best_epoch']}",
        },
    ]
    R.write_csv(R.RESULTS / "ledger.csv", LEDGER_FIELDS, rows)
    R.C.log(f"manifest written: {len(artifacts)} artifacts, {len(checkpoints)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
