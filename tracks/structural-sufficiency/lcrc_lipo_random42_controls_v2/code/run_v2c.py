"""Command-line driver for lcrc-lipo-random42-controls-v2.

Sub-commands
------------
freeze        verify reuse, freeze config/code/gates/width rule (+ sha lock)
width-search  CPU-only PNA width selection (no labels, no Validation scores)
gates         CPU gates (data reuse, capacity, S2 invariance, gradients, width)
preflight     remote CPU re-run of the gates without touching frozen artifacts
smoke         small Train/Validation smoke run for one new model
train         formal single-seed training for s2off or pna_small
benchmark     same-device Validation420 inference timing for all four models
evaluate      Validation-only table + paired bootstrap (no Test)
ledger        rebuild analysis/gpu_ledger.json
status        print the current round status
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import c2_common as C


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _start_envelope(run_key: str, model: str, device: str, run_dir: Path, argv: list[str]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    import torch

    env = {
        "protocol_id": C.PROTOCOL_ID,
        "run_key": run_key,
        "model": model,
        "started_utc": C.utcnow(),
        "started_epoch_seconds": time.time(),
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu",
        "hostname": __import__("socket").gethostname(),
        "git_commit": C.git_commit(),
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
    import c2_ledger

    ledger = c2_ledger.build_ledger()
    if ledger["gpu_hours_charged"] >= C.GPU_HARD_CAP_HOURS:
        raise SystemExit(
            f"GPU budget exhausted: {ledger['gpu_hours_charged']:.4f} h >= {C.GPU_HARD_CAP_HOURS} h"
        )
    return ledger


def cmd_freeze(args: argparse.Namespace) -> int:
    cfg = C.load_config()
    if cfg["models"]["s2off"] != C.LCRC_RECIPE:
        raise PermissionError("config s2off recipe differs from the frozen constant")
    if cfg["models"]["pna"] != C.PNA_BASE_CONFIG:
        raise PermissionError("config PNA base recipe differs from the frozen constant")
    if float(cfg["budget"]["gpu_hard_cap_hours"]) != C.GPU_HARD_CAP_HOURS:
        raise PermissionError("config GPU cap differs from the frozen constant")
    if not C.WIDTH_SEARCH.exists() or not C.CPU_GATES.exists():
        raise PermissionError("width-search and CPU gates must exist before freeze")
    width = C.read_json(C.WIDTH_SEARCH)
    gates = C.read_json(C.CPU_GATES)
    if width.get("protocol_id") != C.PROTOCOL_ID or width.get("labels_read") is not False:
        raise PermissionError("width-search artifact is not valid")
    if gates.get("status") != "PASS":
        raise PermissionError("CPU gates did not pass")
    reuse = C.verify_history()
    C.write_json(C.REUSE_AUDIT, reuse)
    C.write_json(
        C.ANALYSIS / "data_audit.json",
        {
            "protocol_id": C.PROTOCOL_ID,
            "created_utc": C.utcnow(),
            "status": "PASS",
            "split": reuse["split"],
            "structures": reuse["structures"],
            "csv": reuse["csv"],
            "historical_models": reuse["models"],
            "test_labels_read": False,
            "test_predictions_read": False,
        },
    )
    frozen = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "FROZEN_BEFORE_TRAINING",
        "scope": "two_new_models_seed0_validation_only",
        "test_access": "forbidden",
        "test_labels_read": False,
        "test_predictions_read": False,
        "config_path": C.track_rel(C.CONFIG_PATH),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "config_canonical_sha256": C.config_hash(cfg),
        "protocol_sha256": C.sha256_file(C.PROTOCOL_PATH),
        "code_hashes": C.code_file_hashes(),
        "reuse_audit_sha256": C.sha256_file(C.REUSE_AUDIT),
        "width_search_path": C.track_rel(C.WIDTH_SEARCH),
        "width_search_sha256": C.sha256_file(C.WIDTH_SEARCH),
        "width_selected": int(width["selected_width"]),
        "width_selected_parameters": int(width["selected_parameters_total"]),
        "width_relative_difference": float(width["relative_difference"]),
        "width_tightly_size_matched": bool(width["tightly_size_matched"]),
        "cpu_gates_path": C.track_rel(C.CPU_GATES),
        "cpu_gates_sha256": C.sha256_file(C.CPU_GATES),
        "cpu_gates_status": gates["status"],
        "source_hashes": C.verify_source_hashes(),
        "split_file_sha256": C.V1_SPLIT_SHA256,
        "split_index_sha256": dict(C.SPLIT_SHA256),
        "label_file_sha256": C.V1_LABELS_SHA256,
        "structures_sha256": C.V1_STRUCTURES_SHA256,
        "target_mean": C.TARGET_MEAN,
        "target_scale": C.TARGET_SCALE,
        "training_seed": C.TRAIN_SEED,
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "bootstrap_resamples": C.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": C.BOOTSTRAP_SEED,
        "env": C.env_info(),
    }
    C.write_json(C.PROTOCOL_FREEZE, frozen)
    C.write_json(
        C.PROTOCOL_SHA256,
        {
            "protocol_id": C.PROTOCOL_ID,
            "created_utc": C.utcnow(),
            "protocol_freeze_sha256": C.sha256_file(C.PROTOCOL_FREEZE),
            "config_sha256": frozen["config_sha256"],
            "width_search_sha256": frozen["width_search_sha256"],
            "cpu_gates_sha256": frozen["cpu_gates_sha256"],
        },
    )
    C.log(f"[freeze] protocol frozen sha256={C.sha256_file(C.PROTOCOL_FREEZE)}")
    return 0


def cmd_width_search(args: argparse.Namespace) -> int:
    import c2_models as CM
    import r42_data as D

    split = D.load_split()
    payload = CM.run_pna_width_search(split["rows"]["train"])
    C.write_json(C.WIDTH_SEARCH, payload)
    C.log(
        f"[width-search] selected width={payload['selected_width']} "
        f"params={payload['selected_parameters_total']} rel={payload['relative_difference']:.6f} "
        f"tight={payload['tightly_size_matched']}"
    )
    return 0


def cmd_gates(args: argparse.Namespace) -> int:
    import c2_gates

    payload = c2_gates.run_cpu_gates(write=True)
    C.log(f"[gates] status={payload['status']} failed={payload['failed']}")
    return 0 if payload["status"] == "PASS" else 1


def cmd_preflight(args: argparse.Namespace) -> int:
    import c2_gates

    C.verify_frozen(require_clean=False)
    width = C.read_json(C.WIDTH_SEARCH)
    gates = c2_gates.run_cpu_gates(write=False)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": gates["status"],
        "frozen_sha256": C.sha256_file(C.PROTOCOL_FREEZE),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "width_search_sha256": C.sha256_file(C.WIDTH_SEARCH),
        "cpu_gates_sha256": C.sha256_file(C.CPU_GATES),
        "width_selected": int(width["selected_width"]),
        "width_selected_parameters": int(width["selected_parameters_total"]),
        "n_gates": gates["n_gates"],
        "n_failed": gates["n_failed"],
        "failed": gates["failed"],
        "gate_results": [{"name": row["name"], "ok": row["ok"]} for row in gates["gates"]],
        "gpu_used": False,
        "test_labels_read": False,
        "env": C.env_info(),
        "seconds": gates["seconds"],
    }
    C.write_json(C.PREFLIGHT_REMOTE, payload)
    C.log(f"[preflight] status={payload['status']} failed={payload['failed']}")
    return 0 if payload["status"] == "PASS" else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    import c2_data as CD
    import c2_train as CT
    import r42_labels as L

    _budget_guard()
    device = _device(args.device)
    run_key = f"smoke_{args.model}"
    run_dir = C.RUNS / run_key
    env = _start_envelope(run_key, f"{args.model}-smoke", str(device), run_dir, sys.argv)
    try:
        C.verify_frozen(require_clean=False)
        split, labels = CD.load_frozen_split_and_labels()
        if args.model == "s2off":
            payload = CT.train_s2off(
                cfg=C.LCRC_RECIPE,
                split=split,
                labels=labels,
                device=device,
                run_dir=run_dir,
                limit_train=args.limit_train,
                limit_valid=args.limit_valid,
                max_epochs=args.epochs,
                soup_last=args.epochs,
                label="LCRC-S2off-smoke",
            )
        elif args.model == "pna_small":
            import c2_models as CM

            payload = CT.train_pna_small(
                cfg=CM.pna_small_config(),
                split=split,
                labels=labels,
                device=device,
                run_dir=run_dir,
                limit_train=args.limit_train,
                limit_valid=args.limit_valid,
                max_epochs=args.epochs,
                label="PNA-small-smoke",
            )
        else:
            raise SystemExit(f"unknown smoke model {args.model!r}")
    except BaseException as exc:
        _finish_envelope(env, run_dir, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    _finish_envelope(env, run_dir, status="COMPLETED")
    env["extra"] = {"valid_rmse": payload["metrics"]["valid"]["rmse"], "parameters_total": payload["model_audit"].get("parameters_total") or payload["model_audit"].get("parameters_total_nominal")}
    C.write_json(run_dir / "envelope.json", env)
    C.log(f"[smoke] {args.model} ok valid={payload['metrics']['valid']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import c2_data as CD
    import c2_models as CM
    import c2_train as CT
    import r42_labels as L

    _budget_guard()
    C.verify_frozen(require_clean=True)
    device = _device(args.device)
    split, labels = CD.load_frozen_split_and_labels()
    if args.model == "s2off":
        run_key = "s2off_s0"
        model_name = "LCRC-S2off-capacity-matched"
        cfg = dict(C.LCRC_RECIPE)
    elif args.model == "pna_small":
        run_key = "pna_small_s45_s0"
        model_name = "PNA-small"
        cfg = CM.pna_small_config()
    else:
        raise SystemExit(f"unknown model {args.model!r}")
    run_dir = C.RUNS / run_key
    env = _start_envelope(run_key, model_name, str(device), run_dir, sys.argv)
    try:
        if args.model == "s2off":
            payload = CT.train_s2off(
                cfg=cfg, split=split, labels=labels, device=device, run_dir=run_dir
            )
        else:
            payload = CT.train_pna_small(
                cfg=cfg, split=split, labels=labels, device=device, run_dir=run_dir
            )
    except BaseException as exc:
        _finish_envelope(env, run_dir, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    _finish_envelope(env, run_dir, status="COMPLETED")
    env["extra"] = {
        "valid_rmse": payload["metrics"]["valid"]["rmse"],
        "parameters_total": payload["model_audit"].get("parameters_total") or payload["model_audit"].get("parameters_total_nominal"),
        "best_epoch": payload["training"].get("best_epoch"),
    }
    C.write_json(run_dir / "envelope.json", env)
    C.log(f"[train] {args.model} done valid={payload['metrics']['valid']} elapsed={env['elapsed_seconds']:.1f}s")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    import c2_eval as CE

    _budget_guard()
    C.verify_frozen(require_clean=True)
    run_dir = C.RUNS / "benchmark_inference"
    env = _start_envelope("benchmark_inference", "four-model-inference-benchmark", "cuda", run_dir, sys.argv)
    try:
        payload = CE.run_inference_benchmark(batch_size=args.batch_size, warmup=args.warmup, repeats=args.repeats)
    except BaseException as exc:
        _finish_envelope(env, run_dir, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    _finish_envelope(env, run_dir, status="COMPLETED")
    env["extra"] = {
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "models": [
            {
                "name": row["name"],
                "parameters_nominal": row["parameters_nominal"],
                "preprocessing_seconds_per_valid420": row["timing"]["preprocessing_seconds_per_valid420"],
                "forward_seconds_per_valid420": row["timing"]["forward_seconds_per_valid420"],
                "peak_memory_allocated_bytes": row["peak_memory_allocated_bytes"],
            }
            for row in payload["models"]
        ],
    }
    C.write_json(run_dir / "envelope.json", env)
    C.log(f"[benchmark] completed in {env['elapsed_seconds']:.1f}s")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    import c2_eval as CE
    import c2_ledger

    C.verify_frozen(require_clean=False)
    ledger = c2_ledger.build_ledger()
    table = CE.build_validation_table()
    bootstrap = CE.run_paired_bootstrap()
    C.log(
        f"[evaluate] 4 models, validation-only; ledger {ledger['gpu_hours_charged']:.4f} GPU-h; "
        f"delta_rmse(S2off-LCRC)={bootstrap['observed']['delta_rmse']:+.6f}"
    )
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    import c2_ledger

    payload = c2_ledger.build_ledger()
    C.log(
        f"[ledger] {payload['gpu_hours_charged']:.4f} GPU-h over {payload['n_gpu_entries']} GPU entries; "
        f"within_cap={payload['within_cap']}"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    out: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "git_commit": C.git_commit()}
    for name, path in (
        ("protocol_freeze", C.PROTOCOL_FREEZE),
        ("cpu_gates", C.CPU_GATES),
        ("width_search", C.WIDTH_SEARCH),
        ("gpu_ledger", C.GPU_LEDGER),
        ("validation_table", C.VALIDATION_TABLE),
        ("paired_bootstrap", C.PAIRED_BOOTSTRAP),
        ("inference_benchmark", C.INFERENCE_BENCHMARK),
    ):
        if not path.exists():
            out[name] = None
            continue
        payload = C.read_json(path)
        out[name] = {
            "sha256": C.sha256_file(path),
            "status": payload.get("status"),
            "selected_width": payload.get("selected_width"),
            "gpu_hours_charged": payload.get("gpu_hours_charged"),
            "n_failed": payload.get("n_failed"),
        }
    import json

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    sub.add_parser("width-search").set_defaults(func=cmd_width_search)
    sub.add_parser("gates").set_defaults(func=cmd_gates)
    sub.add_parser("preflight").set_defaults(func=cmd_preflight)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("model", choices=["s2off", "pna_small"])
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--limit-train", type=int, default=64)
    smoke.add_argument("--limit-valid", type=int, default=32)
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.set_defaults(func=cmd_smoke)
    train = sub.add_parser("train")
    train.add_argument("model", choices=["s2off", "pna_small"])
    train.add_argument("--device", default="auto")
    train.set_defaults(func=cmd_train)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--batch-size", type=int, default=128)
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--repeats", type=int, default=10)
    bench.set_defaults(func=cmd_benchmark)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    sub.add_parser("ledger").set_defaults(func=cmd_ledger)
    sub.add_parser("status").set_defaults(func=cmd_status)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
