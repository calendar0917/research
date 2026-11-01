"""Command-line driver for ``lcrc_esol_quickstart_v1``.

Commands
--------
prepare       CSV -> label-free raw graphs, physchem descriptors, scaffolds, source manifest
split         generate the fixed Bemis-Murcko partition and audit it
audit-split   re-audit the existing partition (needs RDKit)
p0            ESOL data/feature compatibility + correctness gates (needs no GPU)
freeze        freeze split + config + code before any training
smoke         short CPU smoke of every model kind on tiny slices
train         one formal arm (CUDA, one-shot)
manifest      verify the trained checkpoints and freeze their manifest (before Test)
eval-test     one-shot Test evaluation of the frozen manifest
ledger        sum GPU hours from the run records
capture-rr    save `rr status --json` for every trained arm
status        print round status
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

import esol_common as C  # noqa: E402
import esol_data as D  # noqa: E402
import esol_eval as E  # noqa: E402
import esol_raw as R  # noqa: E402
import esol_split as S  # noqa: E402


def cmd_prepare(args: argparse.Namespace) -> dict[str, Any]:
    del args
    csv_payload = R.read_csv()
    raw = R.build_raw_graphs(csv_payload)
    physchem = R.build_physchem(csv_payload)
    scaffolds = R.build_scaffolds(csv_payload)
    manifest = R.source_manifest()
    C.log(
        f"[prepare] rows={csv_payload['n']} raw_rows={len(raw['rows'])} "
        f"empty_scaffolds={scaffolds['n_empty']} csv_sha={C.CSV_SHA256[:12]}"
    )
    return manifest


def cmd_split(args: argparse.Namespace) -> dict[str, Any]:
    del args
    payload = S.write_split()
    audit = S.audit_split()
    C.log(f"[split] sizes={payload['sizes']} sha={payload['split_sha256'][:12]} audit_passed={audit['passed']}")
    return payload


def cmd_audit_split(args: argparse.Namespace) -> dict[str, Any]:
    del args
    audit = S.audit_split()
    C.log(f"[audit-split] passed={audit['passed']} sizes={audit['sizes']}")
    return audit


def cmd_p0(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import esol_p0

    payload = esol_p0.run()
    failed = [key for key, value in payload["checks"].items() if not value]
    C.log(f"[p0] status={payload['status']} failed_checks={failed}")
    if payload["status"] != "PASS":
        raise SystemExit(1)
    return payload


def cmd_freeze(args: argparse.Namespace) -> dict[str, Any]:
    del args
    payload = C.freeze_protocol()
    C.log(
        f"[freeze] sha={payload['freeze_sha256'][:12]} split={payload['split_manifest_sha256'][:12]} "
        f"config={payload['config_sha256'][:12]}"
    )
    return payload


def cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import torch

    import esol_train as T

    views = D.build_views(include_test=False)
    cfg = C.load_config()
    C.SMOKE.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for model in ("base", "drop20", "physchem6"):
        arm = f"{model}_s0"
        run_path = C.SMOKE / f"{arm}_smoke"
        payload = T.train_arm(
            arm=arm,
            cfg=cfg,
            views=views,
            device=torch.device("cpu"),
            run_dir=run_path,
            pool="local-smoke",
            smoke=True,
            limit_train=64,
            limit_valid=32,
            max_epochs=2,
        )
        out[arm] = {
            "run_dir": C.track_rel(run_path),
            "best_epoch": payload["checkpoint"]["best_epoch"],
            "valid_rmse": payload["metrics"]["valid"]["rmse"],
            "parameters": payload["model_audit"]["parameters"],
            "recheck_abs_diff": payload["checkpoint"]["valid_rmse_recheck_abs_diff"],
            "gradient_probe_passed": payload["diagnostics"]["gradient_probe"]["passed"],
        }
    result = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "COMPLETED",
        "smoke": True,
        "results": out,
        "environment": C.env_info(),
    }
    C.write_json(C.SMOKE / "smoke.json", result)
    C.log(f"[smoke] {json.dumps(out, sort_keys=True)}")
    return result


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    import esol_train as T

    arm = str(args.arm)
    C.require_freeze()
    S.load_split()
    device = torch.device(str(args.device))
    if device.type != "cuda":
        raise PermissionError("formal ESOL training must run on CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    run_path = C.run_dir(arm)
    if (run_path / "train.json").exists():
        raise PermissionError(f"{arm} already trained (one-shot)")
    cfg = C.load_config()
    views = D.build_views(include_test=False)
    payload = T.train_arm(
        arm=arm,
        cfg=cfg,
        views=views,
        device=device,
        run_dir=run_path,
        pool=str(args.pool),
        smoke=False,
    )
    C.log(
        f"[train] {arm} best_epoch={payload['checkpoint']['best_epoch']} "
        f"train={payload['metrics']['train']['rmse']:.6f} valid={payload['metrics']['valid']['rmse']:.6f}"
    )
    return payload


def cmd_manifest(args: argparse.Namespace) -> dict[str, Any]:
    arms = [a for a in str(args.arms).split(",") if a] if args.arms else C.arms_with_runs()
    if not arms:
        raise SystemExit("no trained arms found")
    payload = E.freeze_manifest(arms)
    C.log(f"[manifest] frozen {payload['n_arms']} checkpoints sha={payload['manifest_sha256'][:12]} arms={arms}")
    return payload


def cmd_eval_test(args: argparse.Namespace) -> dict[str, Any]:
    payload = E.evaluate_test(force=bool(args.force))
    for model, rec in payload["summary"]["models"].items():
        C.log(
            f"[eval-test] {model} test_rmse={rec['test_rmse_mean']:.6f} "
            f"test_mae={rec['test_mae_mean']:.6f} valid_rmse={rec['valid_rmse_mean']:.6f}"
        )
    return payload


def cmd_ledger(args: argparse.Namespace) -> dict[str, Any]:
    del args
    arms = C.arms_with_runs()
    runs: dict[str, Any] = {}
    total_seconds = 0.0
    for arm in arms:
        payload = C.read_json(C.run_dir(arm) / "train.json")
        seconds = float(payload["timing"]["process_wall_seconds"])
        total_seconds += seconds
        runs[arm] = {
            "status": "COMPLETED",
            "device": payload.get("device"),
            "cuda_device_name": payload.get("cuda_device_name"),
            "process_wall_seconds": seconds,
            "train_wall_seconds": float(payload["timing"]["train_wall_seconds"]),
            "gpu_count": 1 if str(payload.get("device", "")).startswith("cuda") else 0,
            "best_epoch": int(payload["checkpoint"]["best_epoch"]),
            "valid_rmse": float(payload["metrics"]["valid"]["rmse"]),
        }
    gpu_hours = total_seconds / 3600.0
    ledger = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "formal_runs_completed": len(runs),
        "total_process_seconds": total_seconds,
        "total_gpu_hours_estimate": gpu_hours,
        "charging_rule": "one A100 per run; training-process wall clock from train.json",
        "runs": runs,
        "within_cap": bool(gpu_hours <= C.GPU_HARD_CAP_HOURS),
    }
    C.write_json(C.GPU_LEDGER, ledger)
    C.log(f"[ledger] runs={len(runs)} gpu_hours={gpu_hours:.4f} / {C.GPU_HARD_CAP_HOURS} within_cap={ledger['within_cap']}")
    return ledger


def cmd_capture_rr(args: argparse.Namespace) -> dict[str, Any]:
    del args
    out_dir = C.ANALYSIS / "rr_status"
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for arm in C.arms_with_runs():
        try:
            raw = subprocess.check_output(["rr", "status", arm, "--json"], text=True, stderr=subprocess.STDOUT)
            C.write_json(out_dir / f"{arm}.json", json.loads(raw))
            out[arm] = "saved"
        except Exception as exc:  # noqa: BLE001 - provenance capture must not hide training state
            out[arm] = f"FAILED: {type(exc).__name__}: {exc}"
    C.write_json(out_dir / "capture.json", out)
    return out


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    del args
    runs = C.arms_with_runs()
    status = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "artifacts": {
            "source_manifest": C.SOURCE_MANIFEST.exists(),
            "raw_graph": C.RAW_GRAPH.exists(),
            "physchem": C.PHYSCHEM_FILE.exists(),
            "scaffolds": C.SCAFFOLD_FILE.exists(),
            "split_manifest": C.SPLIT_MANIFEST.exists(),
            "split_audit": C.SPLIT_AUDIT.exists(),
            "p0_summary": C.P0_SUMMARY.exists(),
            "protocol_freeze": C.PROTOCOL_FREEZE.exists(),
            "manifest": C.MANIFEST.exists(),
            "manifest_frozen": C.MANIFEST_FROZEN.exists(),
            "test_eval": (C.EVAL / "metrics.json").exists(),
            "ledger": C.GPU_LEDGER.exists(),
            "p2_hypothesis": C.P2_HYPOTHESIS.exists(),
        },
        "runs": runs,
        "n_runs_completed": len(runs),
    }
    C.log(json.dumps(status, indent=2, sort_keys=True))
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "split", "audit-split", "p0", "freeze", "smoke", "ledger", "capture-rr", "status"):
        sub.add_parser(name)
    p_train = sub.add_parser("train")
    p_train.add_argument("--arm", required=True)
    p_train.add_argument("--device", default="cuda:0", choices=["cuda", "cuda:0"])
    p_train.add_argument("--pool", required=True)
    p_manifest = sub.add_parser("manifest")
    p_manifest.add_argument("--arms", default="", help="comma separated; default = every arm with train.json")
    p_eval = sub.add_parser("eval-test")
    p_eval.add_argument("--force", action="store_true")
    return parser


HANDLERS = {
    "prepare": cmd_prepare,
    "split": cmd_split,
    "audit-split": cmd_audit_split,
    "p0": cmd_p0,
    "freeze": cmd_freeze,
    "smoke": cmd_smoke,
    "train": cmd_train,
    "manifest": cmd_manifest,
    "eval-test": cmd_eval_test,
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
