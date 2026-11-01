"""Command-line driver for ``lcrc_scaffold_error_pilot_v1``.

Commands
--------
split           generate the fixed new scaffold partition and audit it
audit-split     re-audit the existing partition
freeze          freeze split + config + code before P1 training
p0              P0 per-molecule error attribution on old scaffold checkpoints
smoke           short CPU smoke of BASE and G2L-ALL on the new partition
train           one formal P1 arm (CUDA, one-shot)
manifest        verify six checkpoints and freeze their manifest before outer read
eval-outer      one-shot outer evaluation of the frozen manifest
ledger          sum GPU hours from the six run records (and rr status files if present)
capture-rr      save `rr status --json` for the six experiment names
status          print round status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pilot_common as PC
import pilot_data as PD
import pilot_train as PT


def cmd_split(args: argparse.Namespace) -> dict[str, Any]:
    del args
    payload = PD.write_split()
    audit = PD.audit_split()
    PC.log(f"[split] sizes={payload['sizes']} sha={payload['split_sha256'][:12]} audit_passed={audit['passed']}")
    return payload


def cmd_audit_split(args: argparse.Namespace) -> dict[str, Any]:
    del args
    audit = PD.audit_split()
    PC.log(f"[audit-split] passed={audit['passed']} sizes={audit['sizes']}")
    return audit


def cmd_freeze(args: argparse.Namespace) -> dict[str, Any]:
    del args
    payload = PC.freeze_protocol()
    PC.log(f"[freeze] sha={payload['freeze_sha256'][:12]} split={payload['split_manifest_sha256'][:12]}")
    return payload


def cmd_p0(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import pilot_p0

    payload = pilot_p0.run()
    PC.log(f"[p0] all_reproduced={payload['all_reproduced']}")
    return payload


def cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    del args
    payload = PT.smoke()
    PC.log(f"[smoke] {json.dumps(payload['results'], sort_keys=True)}")
    return payload


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    payload = PT.train(str(args.arm), str(args.device), str(args.pool))
    PC.log(
        f"[train] {args.arm} best_epoch={payload['checkpoint']['best_epoch']} "
        f"inner_valid={payload['metrics']['valid']['rmse']:.6f}"
    )
    return payload


def cmd_manifest(args: argparse.Namespace) -> dict[str, Any]:
    del args
    payload = PT.freeze_manifest()
    PC.log(f"[manifest] frozen {payload['n_arms']} checkpoints sha={payload['manifest_sha256'][:12]}")
    return payload


def cmd_eval_outer(args: argparse.Namespace) -> dict[str, Any]:
    payload = PT.evaluate_outer(force=bool(args.force))
    summary = payload["summary"]["models"]
    for model, rec in summary.items():
        PC.log(
            f"[eval-outer] {model} outer_rmse={rec['outer_rmse_mean']:.6f} "
            f"outer_mae={rec['outer_mae_mean']:.6f}"
        )
    return payload


def cmd_capture_rr(args: argparse.Namespace) -> dict[str, Any]:
    del args
    PC.RR_STATUS.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for arm in PC.PILOT_ARMS:
        try:
            raw = subprocess.check_output(["rr", "status", arm, "--json"], text=True, stderr=subprocess.STDOUT)
            payload = json.loads(raw)
            PC.write_json(PC.RR_STATUS / f"{arm}.json", payload)
            out[arm] = "saved"
        except Exception as exc:  # noqa: BLE001 - provenance capture must not hide training state
            out[arm] = f"FAILED: {type(exc).__name__}: {exc}"
    PC.write_json(PC.RR_STATUS / "capture.json", out)
    return out


def _find_elapsed_seconds(payload: Any) -> float | None:
    if isinstance(payload, dict):
        for key in ("elapsed_seconds", "wall_seconds", "duration_seconds"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                return float(value)
        start = payload.get("started_epoch_seconds")
        finish = payload.get("finished_epoch_seconds")
        if isinstance(start, (int, float)) and isinstance(finish, (int, float)) and finish >= start:
            return float(finish - start)
        for value in payload.values():
            found = _find_elapsed_seconds(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_elapsed_seconds(value)
            if found is not None:
                return found
    return None


def cmd_ledger(args: argparse.Namespace) -> dict[str, Any]:
    del args
    runs: dict[str, Any] = {}
    total_process_seconds = 0.0
    for arm in PC.PILOT_ARMS:
        train_json = PT.run_dir(arm) / "train.json"
        if not train_json.exists():
            runs[arm] = {"status": "MISSING"}
            continue
        payload = PC.read_json(train_json)
        process_seconds = float(payload["timing"]["process_wall_seconds"])
        train_seconds = float(payload["timing"]["train_wall_seconds"])
        total_process_seconds += process_seconds
        rr_path = PC.RR_STATUS / f"{arm}.json"
        rr_elapsed = None
        if rr_path.exists():
            rr_elapsed = _find_elapsed_seconds(PC.read_json(rr_path))
        runs[arm] = {
            "status": "COMPLETED",
            "device": payload.get("device"),
            "cuda_device_name": payload.get("cuda_device_name"),
            "started_utc": payload.get("recipe", {}).get("started_utc"),
            "process_wall_seconds": process_seconds,
            "train_wall_seconds": train_seconds,
            "rr_elapsed_seconds": rr_elapsed,
            "gpu_count": 1 if str(payload.get("device", "")).startswith("cuda") else 0,
        }
    gpu_hours = total_process_seconds / 3600.0
    ledger = {
        "protocol_id": PC.PROTOCOL_ID,
        "created_utc": PC.utcnow(),
        "gpu_hard_cap_hours": 2.0,
        "formal_runs_completed": sum(1 for rec in runs.values() if rec.get("status") == "COMPLETED"),
        "total_process_seconds": total_process_seconds,
        "total_gpu_hours_estimate": gpu_hours,
        "charging_rule": "one A100 per run; training-process wall clock from train.json",
        "runs": runs,
        "within_cap": bool(gpu_hours <= 2.0),
    }
    PC.write_json(PC.ANALYSIS / "gpu_ledger.json", ledger)
    PC.log(f"[ledger] gpu_hours={gpu_hours:.4f} / 2.0 within_cap={ledger['within_cap']}")
    return ledger


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    del args
    status = {
        "protocol_id": PC.PROTOCOL_ID,
        "git_commit": PC.git_commit(),
        "git_dirty_tracked": PC.git_dirty(),
        "artifacts": {
            "split_manifest": PC.SPLIT_MANIFEST.exists(),
            "split_audit": PC.SPLIT_AUDIT.exists(),
            "protocol_freeze": PC.PROTOCOL_FREEZE.exists(),
            "p0_summary": (PC.P0_DIR / "p0_summary.json").exists(),
            "manifest": PC.MANIFEST.exists(),
            "manifest_frozen": PC.MANIFEST_FROZEN.exists(),
            "outer_metrics": (PC.EVAL / "metrics.json").exists(),
            "ledger": (PC.ANALYSIS / "gpu_ledger.json").exists(),
        },
        "runs": {arm: (PT.run_dir(arm) / "train.json").exists() for arm in PC.PILOT_ARMS},
        "n_runs_completed": sum((PT.run_dir(arm) / "train.json").exists() for arm in PC.PILOT_ARMS),
        "n_runs_expected": len(PC.PILOT_ARMS),
    }
    PC.log(json.dumps(status, indent=2, sort_keys=True))
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("split")
    sub.add_parser("audit-split")
    sub.add_parser("freeze")
    sub.add_parser("p0")
    sub.add_parser("smoke")
    sub.add_parser("manifest")
    sub.add_parser("ledger")
    sub.add_parser("capture-rr")
    sub.add_parser("status")

    p_train = sub.add_parser("train")
    p_train.add_argument("--arm", required=True, choices=list(PC.PILOT_ARMS))
    p_train.add_argument("--device", default="cuda:0", choices=["cuda", "cuda:0"])
    p_train.add_argument("--pool", required=True)

    p_eval = sub.add_parser("eval-outer")
    p_eval.add_argument("--force", action="store_true")
    return parser


HANDLERS = {
    "split": cmd_split,
    "audit-split": cmd_audit_split,
    "freeze": cmd_freeze,
    "p0": cmd_p0,
    "smoke": cmd_smoke,
    "train": cmd_train,
    "manifest": cmd_manifest,
    "eval-outer": cmd_eval_outer,
    "ledger": cmd_ledger,
    "capture-rr": cmd_capture_rr,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    HANDLERS[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
