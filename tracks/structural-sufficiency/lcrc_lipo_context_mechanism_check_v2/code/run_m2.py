"""Command-line driver for ``lcrc-lipo-context-mechanism-check-v2``.

Sub-commands
------------
audit             frozen data / split / Test-label audit (CPU)
gates             required CPU correctness gates (must pass before CUDA)
freeze            freeze config + code + data + gates + protocol hashes
eval-frozen-s0    part A: evaluate the historical seed-0 BASE/CONTEXT checkpoints
smoke             short three-arm smoke run (CPU locally, CUDA on the GPU host)
train             one formal arm (base_s1/context_s1 on gpu0/gpu1, roleblind_s0 on gpu1)
eval-all          evaluate all five frozen checkpoints on Validation420 and Test420
analysis          comparison tables, tail audit and the two-dimensional decision
ledger            rebuild analysis/gpu_ledger.json from rr status JSON
status            print round status

Formal ``train`` refuses to run unless the CPU gates passed, the protocol freeze
matches the current code/config, the requested arm matches its fixed pool and
seed, the arm has not been trained yet, and the GPU budget is not exhausted.
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

import m2_common as C
import m2_data as MD
import m2_models as M
import m2_train as T


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def require_gates() -> dict[str, Any]:
    if not C.CPU_GATES.exists():
        raise PermissionError("CPU_GATES_MISSING: run `run_m2.py gates` first")
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
        raise PermissionError("PROTOCOL_FREEZE_MISSING: run `run_m2.py freeze` first")
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
    if frozen.get("data_audit_sha256") != C.sha256_file(C.DATA_AUDIT):
        raise PermissionError("data audit file changed after the protocol freeze")
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
    import m2_ledger

    if C.GPU_LEDGER.exists():
        ledger = C.read_json(C.GPU_LEDGER)
        total = float(ledger.get("total_gpu_hours", 0.0))
        if total >= C.GPU_HARD_CAP_HOURS:
            raise SystemExit(f"GPU budget exhausted: {total:.4f} h >= {C.GPU_HARD_CAP_HOURS} h")
        return ledger
    return {"total_gpu_hours": 0.0, "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> dict[str, Any]:
    del args
    views = MD.build_views(include_test=True)
    split = views["split"]
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "PASS",
        "git_commit": C.git_commit(),
        "environment": C.env_info(),
        "split": {
            "file": C.track_rel(C.V1_SPLIT_FILE),
            "sha256": C.V1_SPLIT_SHA256,
            "row_hashes": {name: C.rows_sha256_int64(split["rows"][name]) for name in ("train", "valid", "test")},
            "sizes": {name: len(split["rows"][name]) for name in ("train", "valid", "test")},
            "first_five": {name: [int(x) for x in split["rows"][name][:5]] for name in ("train", "valid", "test")},
            "test_access": "EXPOSED_TEST_EXPLORATORY",
        },
        "labels": views["labels"]["audit"],
        "test_labels": views["test"]["audit"],
        "target_norm": views["target_norm"],
        "train_norm_fingerprint": views["train_norm_fingerprint"],
        "lcrc_cache": {
            "path": C.track_rel(views["cache_built"]["path"]),
            "file_sha256": views["cache_built"]["file_sha256"],
            "content_fingerprint": views["cache_built"]["content_fingerprint"],
            "labels_present": False,
        },
        "feature_spec": {
            "name": views["spec"].name,
            "hash": views["spec"].hash(),
            "atom_feat_dim": int(views["spec"].atom_feat_dim),
            "n_atom_types": int(views["spec"].n_atom_types),
        },
        "n_molecules": {
            "train": int(views["train_cache"]["n_molecules"]),
            "valid": int(views["valid_cache"]["n_molecules"]),
            "test": int(views["test_cache"]["n_molecules"]),
        },
        "test_labels_read": True,
        "test_predictions_read": False,
        "test_metrics_read": False,
    }
    C.write_json(C.DATA_AUDIT, payload)
    C.log(f"[audit] split+labels+cache+Test labels OK ({C.track_rel(C.DATA_AUDIT)})")
    return payload


def cmd_gates(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import m2_gates

    payload = m2_gates.run_gates()
    if not payload["ok"]:
        failed = [c["id"] for c in payload["checks"] if not c["passed"]]
        raise SystemExit(f"CPU gates failed: {failed}")
    return payload


def cmd_freeze(args: argparse.Namespace) -> dict[str, Any]:
    del args
    cfg = C.load_config()
    gates = require_gates()
    if not C.DATA_AUDIT.exists():
        raise PermissionError("data audit missing; run `run_m2.py audit` first")
    if not C.PROTOCOL_PATH.exists():
        raise PermissionError("PROTOCOL.md missing")
    frozen = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "status": "FROZEN_BEFORE_FORMAL_TRAINING",
        "scope": (
            "part A exposed-Test evaluation of frozen seed-0 checkpoints; part B seed-1 "
            "BASE/CONTEXT; part C ROLEBLIND_s0; Validation420 and Test420 reporting"
        ),
        "test_access": "EXPOSED_TEST_EXPLORATORY",
        "config_hash": C.config_hash(cfg),
        "config_file_sha256": C.config_file_sha256(),
        "code_hashes": C.code_file_hashes(),
        "data_audit_sha256": C.sha256_file(C.DATA_AUDIT),
        "cpu_gates_sha256": C.sha256_file(C.CPU_GATES),
        "cpu_gates_n_passed": int(gates["n_passed"]),
        "protocol_sha256": C.sha256_file(C.PROTOCOL_PATH),
        "split_file_sha256": C.V1_SPLIT_SHA256,
        "labels_file_sha256": C.V1_LABELS_SHA256,
        "test_rows_sha256": C.SPLIT_SHA256["test"],
        "test_label_vector_sha256": C.TEST_VECTOR_SHA256,
        "target_norm": {"mean": C.TARGET_MEAN, "scale": C.TARGET_SCALE, "fit_on": "Train3360 only"},
        "frozen_arms": C.ARM_SPECS,
        "frozen_recipe": C.TRAIN_RECIPE,
        "frozen_historical_checkpoints": {
            key: {"sha256": entry["sha256"], "family": entry["family"], "seed": entry["seed"]}
            for key, entry in C.HISTORICAL_CHECKPOINTS.items()
        },
        "evaluation": {
            "bootstrap": C.BOOTSTRAP,
            "tail_rows": list(C.TAIL_ROWS),
            "non_tail_count": C.NON_TAIL_COUNT,
            "performance_rule": C.PERFORMANCE_RULE,
            "role_specificity_rule": C.ROLE_SPECIFICITY_RULE,
            "historical_reference": {
                "pna_native_test_rmse": C.PNA_NATIVE_TEST_RMSE,
                "historical_lcrc_test_rmse": C.HISTORICAL_LCRC_TEST_RMSE,
            },
        },
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "test_labels_read": True,
        "test_predictions_read": False,
    }
    C.write_json(C.PROTOCOL_FREEZE, frozen)
    C.write_json(
        C.PROTOCOL_SHA256,
        {
            "protocol_id": C.PROTOCOL_ID,
            "created_utc": C.utcnow(),
            "protocol_md": C.sha256_file(C.PROTOCOL_PATH),
            "config_yaml": C.config_file_sha256(),
            "protocol_freeze_json": C.sha256_file(C.PROTOCOL_FREEZE),
            "code_hashes": C.code_file_hashes(),
        },
    )
    C.log(f"[freeze] config={frozen['config_hash']} gates={gates['n_passed']}/{gates['n_gates']}")
    return frozen


def cmd_eval_frozen_s0(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import m2_evaluate

    return m2_evaluate.evaluate_frozen_s0()


def cmd_eval_all(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import m2_evaluate

    payload = m2_evaluate.evaluate_all()
    C.log(
        "[eval-all] base_s0={:.6f} context_s0={:.6f} base_s1={:.6f} context_s1={:.6f} roleblind_s0={:.6f} (Test RMSE)".format(
            payload["models"]["base_s0"]["test"]["rmse"],
            payload["models"]["context_s0"]["test"]["rmse"],
            payload["models"]["base_s1"]["test"]["rmse"],
            payload["models"]["context_s1"]["test"]["rmse"],
            payload["models"]["roleblind_s0"]["test"]["rmse"],
        )
    )
    return payload


def cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    cfg = C.load_config()
    device = _device(args.device)
    root = C.SMOKE_CUDA if device.type == "cuda" else C.SMOKE_CPU
    root.mkdir(parents=True, exist_ok=True)
    env = _start_envelope("smoke", "all", str(device), root, sys.argv[1:])
    views = MD.build_views(include_test=False)
    results: dict[str, Any] = {}
    try:
        for arm_key in C.ARM_SPECS:
            run_dir = root / f"{arm_key}_smoke"
            payload = T.train_arm(
                arm_key=arm_key,
                cfg=cfg,
                views=views,
                device=device,
                run_dir=run_dir,
                limit_train=int(args.limit_train),
                limit_valid=int(args.limit_valid),
                max_epochs=int(args.epochs),
                soup_last=int(args.soup_last),
                pool=str(args.pool or "local"),
                smoke=True,
            )
            results[arm_key] = {
                "run_dir": C.track_rel(run_dir),
                "metrics": payload["metrics"],
                "parameters": payload["model_audit"]["parameters"],
                "parameters_new": payload["model_audit"]["parameters_new"],
                "timing": payload["timing"],
                "cuda_device_name": payload["cuda_device_name"],
                "device": payload["device"],
                "dropout_hook_calls": payload["model_audit"]["dropout_hook_calls"],
            }
        # Device-side ROLEBLIND invariance on the smoke checkpoint.
        rb_dir = root / "roleblind_s0_smoke"
        rb_model, _meta = M.build_family("roleblind", views["spec"], views["train_norm"], 0, device)
        M.load_checkpoint(rb_model, rb_dir / "checkpoint.pt", expected_family="roleblind", expected_seed=0)
        rows = list(range(min(16, int(views["valid_cache"]["n_molecules"]))))
        results["roleblind_device_audits"] = {
            "aggregation": M.roleblind_aggregation_audit(rb_model, views["valid_cache"], rows, device),
            "batch_invariance": M.roleblind_batch_invariance_audit(
                rb_model, views["valid_cache"], rows, device, subset_size=len(rows), tolerance=1.0e-5
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
        "device_audit_passed": bool(
            results["roleblind_device_audits"]["aggregation"]["passed"]
            and results["roleblind_device_audits"]["batch_invariance"]["passed"]
        ),
        "environment": C.env_info(),
        "test_labels_read": False,
    }
    C.write_json(root / "smoke.json", payload)
    for arm_key in C.ARM_SPECS:
        C.log(
            f"[smoke] {arm_key} valid={results[arm_key]['metrics']['valid']['rmse']:.5f} "
            f"params={results[arm_key]['parameters']}"
        )
    C.log(f"[smoke] roleblind device audits passed={payload['device_audit_passed']}")
    if not payload["device_audit_passed"]:
        raise SystemExit("ROLEBLIND device audit failed")
    return payload


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    cfg = C.load_config()
    gates = require_gates()
    freeze = require_freeze()
    arm_key = str(args.arm)
    if arm_key not in C.ARM_SPECS:
        raise ValueError(arm_key)
    arm = C.ARM_SPECS[arm_key]
    if str(args.pool) != arm["pool"]:
        raise PermissionError(f"arm {arm_key} must run on pool {arm['pool']}, got {args.pool}")
    run_dir = C.RUNS / arm["run_key"]
    if (run_dir / "train.json").exists():
        raise PermissionError(f"{arm_key} already has train.json; formal training is one-shot and will not be repeated")
    ledger = _budget_guard()
    device = _device(args.device)
    if device.type != "cuda":
        raise PermissionError("formal training must run on CUDA; CPU checks live in `gates`/`smoke`")
    env = _start_envelope(arm_key, arm["family"], str(device), run_dir, sys.argv[1:])
    extra = {
        "gates_n_passed": int(gates["n_passed"]),
        "protocol_freeze_sha256": C.sha256_file(C.PROTOCOL_FREEZE),
        "protocol_sha256": freeze["protocol_sha256"],
        "budget_before_gpu_hours": float(ledger.get("total_gpu_hours", 0.0)),
        "train_seed": int(arm["seed"]),
        "split_seed": C.SPLIT_SEED,
    }
    try:
        views = MD.build_views(include_test=False)
        payload = T.train_arm(
            arm_key=arm_key,
            cfg=cfg,
            views=views,
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
        f"[train] arm={arm_key} pool={args.pool} valid={payload['metrics']['valid']['rmse']:.6f} "
        f"params={payload['model_audit']['parameters']} wall={payload['timing']['train_wall_seconds']:.1f}s"
    )
    return payload


def cmd_analysis(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import m2_analysis

    return m2_analysis.analyze()


def cmd_ledger(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import m2_ledger

    payload = m2_ledger.build_ledger()
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
            "eval_frozen_s0": (C.EVAL_FROZEN_S0 / "metrics.json").exists(),
            "base_s1": (C.RUNS / "base_s1/train.json").exists(),
            "context_s1": (C.RUNS / "context_s1/train.json").exists(),
            "roleblind_s0": (C.RUNS / "roleblind_s0/train.json").exists(),
            "eval_all": (C.EVAL_DIR / "metrics.json").exists(),
            "comparison": C.COMPARISON.exists(),
            "decision": C.DECISION.exists(),
        },
    }
    if C.CPU_GATES.exists():
        gates = C.read_json(C.CPU_GATES)
        status["cpu_gates"] = {"ok": gates.get("ok"), "n_passed": gates.get("n_passed"), "n_gates": gates.get("n_gates")}
    if C.DECISION.exists():
        decision = C.read_json(C.DECISION)
        status["decision"] = {
            "PERFORMANCE_SIGNAL": decision["PERFORMANCE_SIGNAL"],
            "ROLE_SPECIFICITY_SIGNAL": decision["ROLE_SPECIFICITY_SIGNAL"],
        }
    C.log(json.dumps(status, indent=2, sort_keys=True))
    return status


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="frozen data / split / Test-label audit")
    sub.add_parser("gates", help="required CPU correctness gates")
    sub.add_parser("freeze", help="freeze config + code + data + gates + protocol hashes")
    sub.add_parser("eval-frozen-s0", help="part A: evaluate the historical seed-0 checkpoints")
    sub.add_parser("eval-all", help="evaluate all five frozen checkpoints")
    sub.add_parser("analysis", help="comparison tables, tail audit, decision")
    sub.add_parser("ledger", help="rebuild the GPU ledger from rr status JSON")
    sub.add_parser("status", help="print round status")

    p_smoke = sub.add_parser("smoke", help="short three-arm smoke run")
    p_smoke.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0", "auto"])
    p_smoke.add_argument("--epochs", type=int, default=1)
    p_smoke.add_argument("--limit-train", type=int, default=64)
    p_smoke.add_argument("--limit-valid", type=int, default=32)
    p_smoke.add_argument("--soup-last", type=int, default=1)
    p_smoke.add_argument("--pool", default=None)

    p_train = sub.add_parser("train", help="train one formal arm")
    p_train.add_argument("--arm", required=True, choices=list(C.ARM_SPECS))
    p_train.add_argument("--device", default="cuda", choices=["cuda", "cuda:0"])
    p_train.add_argument("--pool", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and args.pool is None:
        args.pool = C.ARM_SPECS[args.arm]["pool"]
    handlers = {
        "audit": cmd_audit,
        "gates": cmd_gates,
        "freeze": cmd_freeze,
        "eval-frozen-s0": cmd_eval_frozen_s0,
        "eval-all": cmd_eval_all,
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
