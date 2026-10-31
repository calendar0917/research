"""Command-line driver for ``lcrc-lipo-contextual-contrib-v1``.

Sub-commands
------------
audit      data / split / role-coverage audit (CPU, no training)
gates      the 12 required CPU correctness gates (must pass before CUDA)
freeze     freeze config + code + data + gates hashes before formal training
smoke      short two-arm smoke run (CPU locally, CUDA on the GPU host)
train      one formal arm (base on res-gpu0, context on res-gpu1)
analysis   Validation-only comparison, tail audit, subgroups and decision
ledger     rebuild analysis/gpu_ledger.json from rr status JSON
status     print the current round status

The formal ``train`` command refuses to run unless the CPU gates passed, the
protocol freeze matches the current code/config, the requested arm matches its
fixed pool, and the GPU budget is not exhausted.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import x1_common as C
import x1_data as XD
import x1_models as M
import x1_train as T


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def require_gates() -> dict[str, Any]:
    if not C.CPU_GATES.exists():
        raise PermissionError("CPU_GATES_MISSING: run `run_x1.py gates` first")
    payload = C.read_json(C.CPU_GATES)
    if payload.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("cpu_gates.json has the wrong protocol id")
    if payload.get("ok") is not True or int(payload.get("n_passed", 0)) != len(payload.get("checks", [])):
        raise PermissionError("cpu_gates.json records a failing gate")
    if payload.get("config_hash") != C.config_hash(C.load_config()):
        raise PermissionError("CPU gates were produced with a different config")
    if payload.get("code_hashes") != C.code_file_hashes():
        raise PermissionError("CPU gates were produced with different code")
    return payload


def require_freeze() -> dict[str, Any]:
    if not C.PROTOCOL_FREEZE.exists():
        raise PermissionError("PROTOCOL_FREEZE_MISSING: run `run_x1.py freeze` first")
    frozen = C.read_json(C.PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("protocol freeze has the wrong protocol id")
    if frozen.get("config_hash") != C.config_hash(C.load_config()):
        raise PermissionError("config changed after the protocol freeze")
    if frozen.get("code_hashes") != C.code_file_hashes():
        changed = sorted(
            key
            for key in set(frozen.get("code_hashes", {})) | set(C.code_file_hashes())
            if frozen.get("code_hashes", {}).get(key) != C.code_file_hashes().get(key)
        )
        raise PermissionError(f"code changed after the protocol freeze: {changed}")
    if frozen.get("cpu_gates_sha256") != C.sha256_file(C.CPU_GATES):
        raise PermissionError("CPU gates file changed after the protocol freeze")
    if frozen.get("protocol_sha256") != C.sha256_file(C.PROTOCOL_PATH):
        raise PermissionError("PROTOCOL.md changed after the protocol freeze")
    return frozen


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return device


def _start_envelope(run_key: str, arm: str, device: str, run_dir: Path, argv: list[str]) -> dict[str, Any]:
    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "protocol_id": C.PROTOCOL_ID,
        "run_key": run_key,
        "arm": arm,
        "started_utc": C.utcnow(),
        "started_epoch_seconds": time.time(),
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu",
        "hostname": socket.gethostname(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "argv": argv,
        "status": "RUNNING",
    }
    C.write_json(run_dir / "envelope.json", env)
    return env


def _finish_envelope(env: dict[str, Any], run_dir: Path, *, status: str, error: str | None = None) -> None:
    env.update(
        {
            "finished_utc": C.utcnow(),
            "finished_epoch_seconds": time.time(),
            "status": status,
        }
    )
    env["elapsed_seconds"] = env["finished_epoch_seconds"] - env["started_epoch_seconds"]
    if error is not None:
        env["error"] = error
    C.write_json(run_dir / "envelope.json", env)


def _budget_guard() -> dict[str, Any]:
    import x1_ledger

    ledger = x1_ledger.build_ledger()
    if ledger["total_gpu_hours"] >= C.GPU_HARD_CAP_HOURS:
        raise SystemExit(
            f"GPU budget exhausted: {ledger['total_gpu_hours']:.4f} h >= {C.GPU_HARD_CAP_HOURS} h"
        )
    return ledger


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> dict[str, Any]:
    split, labels = XD.load_frozen_split_and_labels()
    with XD.read_audit() as opened:
        views = XD.build_lcrc_views(split, labels)
        audit = x1_gates_build_data_audit(split, labels, views)
        forbidden = XD.forbidden_reads(opened)
    audit["runtime_open_audit"] = {
        "n_files_opened": len(opened),
        "forbidden_opens": forbidden,
        "scope": "data audit only; the training/analysis audit is repeated in the CPU gates",
    }
    if forbidden:
        raise PermissionError(f"forbidden artifact opened during data audit: {forbidden}")
    C.write_json(C.DATA_AUDIT, audit)
    C.log(f"[audit] split+labels+role coverage OK ({C.track_rel(C.DATA_AUDIT)})")
    return audit


def x1_gates_build_data_audit(split, labels, views):
    import x1_gates

    return x1_gates.build_data_audit(split, labels, views)


def cmd_gates(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import x1_gates

    payload = x1_gates.run_gates()
    if not payload["ok"]:
        failed = [c["id"] for c in payload["checks"] if not c["passed"]]
        raise SystemExit(f"CPU gates failed: {failed}")
    return payload


def cmd_freeze(args: argparse.Namespace) -> dict[str, Any]:
    del args
    cfg = C.load_config()
    gates = require_gates()
    if not C.DATA_AUDIT.exists():
        raise PermissionError("data audit missing; run `run_x1.py audit` first")
    if not C.PROTOCOL_PATH.exists():
        raise PermissionError("PROTOCOL.md missing")
    frozen = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "status": "FROZEN_BEFORE_FORMAL_TRAINING",
        "scope": "two arms, one seed (0), random42 Train3360/Validation420, Validation-only reporting",
        "test_access": "forbidden",
        "config_hash": C.config_hash(cfg),
        "config_file_sha256": C.sha256_file(C.CONFIG_PATH),
        "code_hashes": C.code_file_hashes(),
        "data_audit_sha256": C.sha256_file(C.DATA_AUDIT),
        "cpu_gates_sha256": C.sha256_file(C.CPU_GATES),
        "cpu_gates_n_passed": int(gates["n_passed"]),
        "protocol_sha256": C.sha256_file(C.PROTOCOL_PATH),
        "split_file_sha256": C.V1_SPLIT_SHA256,
        "labels_file_sha256": C.V1_LABELS_SHA256,
        "target_norm": {
            "mean": C.TARGET_MEAN,
            "scale": C.TARGET_SCALE,
            "fit_on": "Train3360 only",
        },
        "frozen_recipe": C.TRAIN_RECIPE,
        "frozen_pools": C.ARM_POOL,
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "decision_rules": {
            "gain_threshold": C.GAIN_THRESHOLD,
            "mae_worsening_tolerance": C.MAE_WORSENING_TOLERANCE,
            "tail_rows": list(C.TAIL_ROWS),
            "tail_share_threshold": C.TAIL_SHARE_THRESHOLD,
            "tail_rule": cfg["decision"]["tail_rule"],
        },
        "test_labels_read": False,
        "test_predictions_read": False,
    }
    C.write_json(C.PROTOCOL_FREEZE, frozen)
    C.write_json(
        C.PROTOCOL_SHA256,
        {
            "protocol_id": C.PROTOCOL_ID,
            "created_utc": C.utcnow(),
            "protocol_md": C.sha256_file(C.PROTOCOL_PATH),
            "config_yaml": C.sha256_file(C.CONFIG_PATH),
            "protocol_freeze_json": C.sha256_file(C.PROTOCOL_FREEZE),
            "code_hashes": C.code_file_hashes(),
        },
    )
    C.log(f"[freeze] config={frozen['config_hash']} gates={gates['n_passed']}/{gates['n_gates']}")
    return frozen


def cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    cfg = C.load_config()
    device = _device(args.device)
    split, labels = XD.load_frozen_split_and_labels()
    root = C.ANALYSIS / ("smoke_cuda" if device.type == "cuda" else "smoke_cpu")
    root.mkdir(parents=True, exist_ok=True)
    env = _start_envelope("smoke", "both", str(device), root, sys.argv[1:])
    results: dict[str, Any] = {}
    try:
        for arm in C.ARMS:
            run_dir = root / f"{arm}_smoke"
            payload = T.train_arm(
                arm=arm,
                cfg=cfg,
                split=split,
                labels=labels,
                device=device,
                run_dir=run_dir,
                limit_train=int(args.limit_train),
                limit_valid=int(args.limit_valid),
                max_epochs=int(args.epochs),
                soup_last=int(args.soup_last),
                pool=str(args.pool or "local"),
                smoke=True,
            )
            results[arm] = {
                "run_dir": C.track_rel(run_dir),
                "metrics": payload["metrics"],
                "parameters": payload["model_audit"]["parameters"],
                "parameters_new": payload["model_audit"]["parameters_new"],
                "timing": payload["timing"],
                "cuda_device_name": payload["cuda_device_name"],
                "device": payload["device"],
                "dropout_hook_calls": payload["model_audit"]["dropout_hook_calls"],
                "context_path_trained_and_used": (
                    payload.get("diagnostics", {}).get("context_usage", {}).get("gradient_probe", {}).get(
                        "all_new_tensors_received_nonzero_gradient"
                    )
                ),
            }
    except Exception as exc:
        _finish_envelope(env, root, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    _finish_envelope(env, root, status="COMPLETED")
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "COMPLETED",
        "device": str(device),
        "epochs": int(args.epochs),
        "limit_train": int(args.limit_train),
        "limit_valid": int(args.limit_valid),
        "soup_last": int(args.soup_last),
        "results": results,
        "environment": C.env_info(),
        "test_labels_read": False,
    }
    C.write_json(root / "smoke.json", payload)
    C.log(f"[smoke] device={device} base_valid={results['base']['metrics']['valid']['rmse']:.5f} "
          f"context_valid={results['context']['metrics']['valid']['rmse']:.5f}")
    return payload


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    cfg = C.load_config()
    gates = require_gates()
    freeze = require_freeze()
    arm = str(args.arm)
    if arm not in C.ARMS:
        raise ValueError(arm)
    if str(args.pool) != C.ARM_POOL[arm]:
        raise PermissionError(f"arm {arm} must run on pool {C.ARM_POOL[arm]}, got {args.pool}")
    ledger = _budget_guard()
    device = _device(args.device)
    if device.type != "cuda":
        raise PermissionError("formal training must run on CUDA; CPU checks live in `gates`/`smoke`")
    run_key = C.ARM_RUN_KEY[arm]
    run_dir = C.RUNS / run_key
    env = _start_envelope(
        run_key,
        arm,
        str(device),
        run_dir,
        sys.argv[1:],
    )
    extra = {
        "gates_n_passed": int(gates["n_passed"]),
        "protocol_freeze_sha256": C.sha256_file(C.PROTOCOL_FREEZE),
        "protocol_sha256": freeze["protocol_sha256"],
        "budget_before_gpu_hours": float(ledger["total_gpu_hours"]),
    }
    try:
        split, labels = XD.load_frozen_split_and_labels()
        payload = T.train_arm(
            arm=arm,
            cfg=cfg,
            split=split,
            labels=labels,
            device=device,
            run_dir=run_dir,
            pool=str(args.pool),
            smoke=False,
        )
        payload.update(extra)
        C.write_json(run_dir / "train.json", payload)
    except Exception as exc:
        _finish_envelope(env, run_dir, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    _finish_envelope(env, run_dir, status="COMPLETED")
    C.log(
        f"[train] arm={arm} pool={args.pool} valid={payload['metrics']['valid']['rmse']:.6f} "
        f"params={payload['model_audit']['parameters']} wall={payload['timing']['train_wall_seconds']:.1f}s"
    )
    return payload


def cmd_analysis(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import x1_analysis

    payload = x1_analysis.analyze()
    C.log(
        f"[analysis] gain={payload['decision']['gain']:+.6f} "
        f"decision={payload['decision']['exploratory_decision']} "
        f"tail={payload['decision']['tail_status']}"
    )
    return payload


def cmd_ledger(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import x1_ledger

    payload = x1_ledger.build_ledger()
    C.log(
        f"[ledger] gpu_hours={payload['total_gpu_hours']:.4f} / {payload['hard_cap_gpu_hours']:.2f} "
        f"within_cap={payload['within_cap']}"
    )
    return payload


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    del args
    status = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "artifacts": {
            "data_audit": C.DATA_AUDIT.exists(),
            "cpu_gates": C.CPU_GATES.exists(),
            "protocol_freeze": C.PROTOCOL_FREEZE.exists(),
            "base_run": (C.RUNS / C.ARM_RUN_KEY["base"] / "train.json").exists(),
            "context_run": (C.RUNS / C.ARM_RUN_KEY["context"] / "train.json").exists(),
            "comparison": C.COMPARISON.exists(),
            "decision": C.DECISION.exists(),
        },
    }
    if C.CPU_GATES.exists():
        gates = C.read_json(C.CPU_GATES)
        status["cpu_gates"] = {"ok": gates.get("ok"), "n_passed": gates.get("n_passed"), "n_gates": gates.get("n_gates")}
    if C.DECISION.exists():
        status["decision"] = C.read_json(C.DECISION)["outcome"]
    C.log(json.dumps(status, indent=2, sort_keys=True))
    return status


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("audit", help="data / split / role-coverage audit")
    sub.add_parser("gates", help="run the 12 required CPU correctness gates")
    sub.add_parser("freeze", help="freeze config + code + data + gates hashes")
    sub.add_parser("analysis", help="Validation-only comparison, tail audit, subgroups, decision")
    sub.add_parser("ledger", help="rebuild the GPU ledger from rr status JSON")
    sub.add_parser("status", help="print the current round status")

    p_smoke = sub.add_parser("smoke", help="short two-arm smoke run")
    p_smoke.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0", "auto"])
    p_smoke.add_argument("--epochs", type=int, default=2)
    p_smoke.add_argument("--limit-train", type=int, default=256)
    p_smoke.add_argument("--limit-valid", type=int, default=64)
    p_smoke.add_argument("--soup-last", type=int, default=2)
    p_smoke.add_argument("--pool", default=None)

    p_train = sub.add_parser("train", help="train one formal arm")
    p_train.add_argument("--arm", required=True, choices=list(C.ARMS))
    p_train.add_argument("--device", default="cuda", choices=["cuda", "cuda:0"])
    p_train.add_argument("--pool", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and args.pool is None:
        args.pool = C.ARM_POOL[args.arm]
    handlers = {
        "audit": cmd_audit,
        "gates": cmd_gates,
        "freeze": cmd_freeze,
        "smoke": cmd_smoke,
        "train": cmd_train,
        "analysis": cmd_analysis,
        "ledger": cmd_ledger,
        "status": cmd_status,
    }
    handlers[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
