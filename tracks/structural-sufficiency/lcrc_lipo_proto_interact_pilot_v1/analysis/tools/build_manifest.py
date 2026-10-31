"""Assemble analysis/run_manifest.json and .tsv.

Inputs (all committed / pulled artefacts):

* ``analysis/rr_status/*.json``  -- ``rr status <experiment> --json`` output
* ``analysis/runs/*/train.json`` -- per-run training payload
* ``analysis/gpu_ledger.json``   -- GPU budget ledger

This tool only merges and prints; it never re-trains or re-scores anything and
it is outside the frozen ``code/`` hash set (which stayed unchanged after the
protocol freeze).

    uv run python analysis/tools/build_manifest.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
ANALYSIS = EXP / "analysis"

# experiment name in rr -> (arm, run directory key)
RR_MAP = {
    "lipo-pilot-base0": ("BASE_G0", "BASE_G0_s0_fold2"),
    "lipo-pilot-base1": ("BASE_G1", "BASE_G1_s0_fold2"),
    "lipo-pilot-proto": ("PROTO_K16_TOP4", "PROTO_K16_TOP4_s0_fold2"),
    "lipo-pilot-pairtrue": ("PAIR_TRUE", "PAIR_TRUE_s0_fold2"),
    "lipo-pilot-pairnull": ("PAIR_NULL", "PAIR_NULL_s0_fold2"),
    "lipo-pilot-smoke": ("SMOKE_ALL_ARMS", "smoke"),
}


def main() -> None:
    ledger = json.loads((ANALYSIS / "gpu_ledger.json").read_text())
    gpu_by_exp = {row["experiment"]: row for row in ledger["runs"]}
    rows = []
    for experiment, (arm, directory) in RR_MAP.items():
        status_path = ANALYSIS / "rr_status" / f"{experiment}.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        train_path = ANALYSIS / "runs" / directory / "train.json"
        if not train_path.exists():
            train_path = ANALYSIS / "smoke" / directory / "train.json"
        train = json.loads(train_path.read_text()) if train_path.exists() else {}
        gpu = gpu_by_exp.get(experiment, {})
        rows.append(
            {
                "experiment": experiment,
                "arm": arm,
                "rr_run_id": status.get("run_id"),
                "state": status.get("state"),
                "exit_code": status.get("exit_code"),
                "pool": status.get("pool"),
                "node": status.get("node"),
                "driver_version": status.get("driver_version"),
                "torch": status.get("torch_version"),
                "gpu_seconds_rr": gpu.get("gpu_seconds"),
                "started_at": status.get("started_at"),
                "completed_at": status.get("completed_at"),
                "commit": status.get("commit"),
                "dirty": status.get("dirty"),
                "parameters": train.get("model", {}).get("parameters"),
                "train_rmse_raw": train.get("metrics", {}).get("train_rmse_raw"),
                "train_mae_raw": train.get("metrics", {}).get("train_mae_raw"),
                "dev_rmse_raw": train.get("metrics", {}).get("dev_rmse_raw"),
                "dev_mae_raw": train.get("metrics", {}).get("dev_mae_raw"),
                "dev_r2": train.get("metrics", {}).get("dev_r2"),
                "epochs": train.get("recipe", {}).get("epochs"),
                "optimizer_steps": train.get("recipe", {}).get("total_updates"),
                "soup_members": train.get("recipe", {}).get("soup_members"),
                "grad_clip_events": train.get("anomalies", {}).get("grad_clip_events"),
                "wall_seconds": train.get("timing", {}).get("process_wall_seconds"),
                "soup_sha256": train.get("artifacts", {}).get("soup", {}).get("sha256"),
                "soup_state_hash": train.get("model", {}).get("soup_state_hash"),
                "predictions_dev_sha256": train.get("artifacts", {}).get("predictions_dev", {}).get("sha256"),
                "labels_scope": train.get("labels_scope"),
                "diagnostics_written": (ANALYSIS / "runs" / directory / "diagnostics.json").exists(),
            }
        )
    payload = {
        "protocol_id": "lcrc-lipo-proto-interact-pilot-v1",
        "note": "merged view of analysis/rr_status/*.json, analysis/runs/*/train.json and analysis/gpu_ledger.json",
        "runs": rows,
        "total_gpu_seconds": ledger["total_gpu_seconds"],
        "total_gpu_hours": ledger["total_gpu_hours"],
        "gpu_hard_cap_hours": ledger["hard_cap_gpu_hours"],
        "within_cap": ledger["within_cap"],
    }
    (ANALYSIS / "run_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "experiment",
        "arm",
        "rr_run_id",
        "state",
        "pool",
        "gpu_seconds_rr",
        "parameters",
        "train_rmse_raw",
        "dev_rmse_raw",
        "dev_mae_raw",
        "dev_r2",
        "epochs",
        "optimizer_steps",
        "wall_seconds",
        "grad_clip_events",
        "soup_members",
        "soup_sha256",
        "commit",
        "dirty",
    ]
    with open(ANALYSIS / "run_manifest.tsv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(fields)
        for row in rows:
            out = []
            for field in fields:
                value = row[field]
                if field == "soup_members" and value:
                    value = "/".join(str(x) for x in value)
                if isinstance(value, float):
                    value = f"{value:.6f}"
                out.append("" if value is None else value)
            writer.writerow(out)
    print(json.dumps({k: payload[k] for k in ("runs",)}, indent=1)[:400])
    print(f"wrote {ANALYSIS/'run_manifest.json'} and run_manifest.tsv")


if __name__ == "__main__":
    main()
