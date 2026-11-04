"""CLI entry point for the bond-object pairing pilot."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import common as C
import data as D
import evaluate as E
import fit_labels
import gates as G
import model as M
import predict as P
import resource_audit as RA
import train as T


def _read_or_none(path: Path):
    return C.read_json(path) if path.exists() else None


def cmd_verify_data(_args) -> int:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    split = C.load_frozen_split()
    caches = D.load_caches(spec)
    fit = fit_labels.load_fit_labels(arm_guard=True)
    target = D.load_target_norm(verify=False)
    stats = fit_labels.fit_label_stats(fit["y"])
    hist = C.verify_historical_a_predictions()
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "env": C.env_info(),
        "feature_spec": {
            "sha256": C.sha256_file(C.PREP_ANALYSIS / "feature_spec_lipo.json"),
            "hash": spec.hash(),
            "dimensions": {
                "atom": spec.atom_feat_dim,
                "bond": spec.bond_feat_dim,
                "global": spec.global_dim,
                "scalars": spec.n_scalars,
                "n_bond_types": spec.n_bond_types,
            },
        },
        "norm": {
            "sha256": C.sha256_file(C.PREP_ANALYSIS / "norm_lipo.json"),
            "fingerprint": C.prep_data.norm_fingerprint(norm),
            "valid_or_test_used": norm["valid_or_test_used"],
        },
        "split": {
            "sha256": C.sha256_file(C.LIPO_SPLIT),
            "protocol_id": split["payload"]["protocol_id"],
            "status": split["payload"]["status"],
            "row_hashes": split["hashes"],
            "sizes": {k: len(v) for k, v in split["split"].items()},
            "scope": split["payload"]["scope"],
            "overlap": {
                "fit_monitor": len(set(split["split"]["fit"]) & set(split["split"]["monitor"])),
                "fit_reserved": len(set(split["split"]["fit"]) & set(split["split"]["reserved"])),
                "monitor_reserved": len(set(split["split"]["monitor"]) & set(split["split"]["reserved"])),
            },
        },
        "caches": {
            k: {
                "path": C.track_rel(v["path"]),
                "file_sha256": v["file_sha256"],
                "content_fingerprint": v["content_fingerprint"],
                "rebuilt_from_raw": v["rebuilt"],
                "file_sha256_matches_historical_serialization": v["file_sha256_matches_historical_serialization"],
                "content_fingerprint_matches_frozen": v["content_fingerprint_matches_frozen"],
                "consistency_basis": v["consistency_basis"],
                "n_molecules": v["n_molecules"],
                "labels_present": v["cache"]["meta"]["labels_present"],
                "row_order_matches_split": True,
                "n_atoms": int(v["cache"]["atom_ptr"][-1]),
                "n_bonds": int(v["cache"]["bond_ptr"][-1]),
                "n_s1_codes": int(v["cache"]["nbr_ptr"][-1]),
                "n_s2_paths": int(v["cache"]["path_ptr"][-1]),
            }
            for k, v in caches.items()
        },
        "fit_labels": {
            "audit": fit["audit"],
            "stats": stats,
            "target_norm_match": all(target[k] == stats[k] for k in ("mean", "scale", "fit_label_vector_sha256")),
        },
        "historical_A": {
            "manifest": hist["manifest"],
            "manifest_sha256": hist["manifest_sha256"],
            "predictions": {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in hist["predictions"].items()},
        },
        "forbidden_sources_present_but_unread": {
            "csv": C.LIPO_CSV.exists(),
            "official_split": C.LIPO_OFFICIAL_SPLIT.exists(),
            "reserved_cache": (C.PREP_CACHE / "lipo_reserved.pt").exists(),
        },
        "reserved_rows_used": False,
        "official_valid_test_used": False,
        "monitor_labels_read": False,
        "gpu_used": False,
    }
    C.write_json(C.ANALYSIS / "data_audit.json", payload)
    C.log(f"[verify-data] fit={payload['split']['sizes']['fit']} monitor={payload['split']['sizes']['monitor']} caches={list(caches)}")
    return 0


def cmd_preflight(_args) -> int:
    C.ensure_dir(C.ANALYSIS)
    manifest = C.build_preflight_manifest()
    hist = C.verify_historical_a_predictions()
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    batch = D.collate(D.load_caches(spec)["fit"]["cache"], list(range(8)))
    params = {}
    for seed in C.SEEDS:
        R, ri = M.build_model("R", spec, norm, seed, "cpu")
        N, ni = M.build_model("N", spec, norm, seed, "cpu")
        R.eval(); N.eval()
        with torch.no_grad():
            yR = R(batch)
            yN = N(batch)
        params[f"seed_{seed}"] = {
            "R": {k: ri[k] for k in ("parameters", "parameters_trunk", "parameters_new", "state_hash", "trunk_state_hash", "trunk_expected_hash", "trunk_init_matches_historical")},
            "N": {k: ni[k] for k in ("parameters", "parameters_trunk", "parameters_new", "state_hash", "trunk_state_hash", "trunk_expected_hash", "trunk_init_matches_historical")},
            "R_N_init_equal": ri["state_hash"] == ni["state_hash"],
            "R_N_prediction_max_abs": float((yR - yN).abs().max()),
            "parameters_total_match": ri["parameters"] == ni["parameters"] == C.TOTAL_PARAMS_REFERENCE,
            "new_parameters_match": ri["parameters_new"] == ni["parameters_new"] == C.NEW_PARAMS,
        }
    param_payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "reference_trunk": C.LCRC_PARAMS_REFERENCE,
        "new_parameters": C.NEW_PARAMS,
        "new_breakdown": C.NEW_PARAM_BREAKDOWN,
        "total_expected": C.TOTAL_PARAMS_REFERENCE,
        "by_seed": params,
        "historical_A_init_hashes": {str(k): v for k, v in C.EXPECTED_TRUNK_INIT_HASH.items()},
        "all_checks_pass": all(v["parameters_total_match"] and v["new_parameters_match"] and v["R_N_init_equal"] for v in params.values()),
    }
    C.write_json(C.ANALYSIS / "parameter_audit.json", param_payload)
    manifest["historical_A"] = {
        "manifest": hist["manifest"],
        "manifest_sha256": hist["manifest_sha256"],
        "rows_sha256": hist["rows_sha256"],
        "predictions": {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in hist["predictions"].items()},
    }
    manifest["parameter_audit"] = param_payload
    C.write_json(C.ANALYSIS / "preflight.json", manifest)
    C.log(f"[preflight] code_hashes={len(manifest['code_hashes'])} params_ok={param_payload['all_checks_pass']}")
    return 0


def _g7_record() -> dict[str, Any]:
    res = _read_or_none(C.ANALYSIS / "resource_isolation.json")
    if not res:
        return {"gate": "G7_resource_and_conflict", "status": "PENDING", "passed": False, "reason": "resource_isolation.json missing"}
    return {
        "gate": "G7_resource_and_conflict",
        "status": "PASS" if res.get("status") == "PASS" else "FAIL",
        "passed": res.get("status") == "PASS",
        "resource_isolation": C.track_rel(C.ANALYSIS / "resource_isolation.json"),
        "blocking_reasons": res.get("blocking_reasons", []),
        "selected_physical_gpu": res.get("planned_execution", {}).get("physical_gpu_index"),
        "CUDA_VISIBLE_DEVICES": res.get("planned_execution", {}).get("CUDA_VISIBLE_DEVICES"),
    }


def _write_gates_with_g7(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["gates"] = list(payload.get("gates", [])) + [_g7_record()]
    payload["all_passed"] = all(g.get("passed") for g in payload["gates"])
    C.write_json(C.ANALYSIS / "gates.json", payload)
    return payload


def cmd_gates(_args) -> int:
    payload = G.run_gates(write=False)
    payload = _write_gates_with_g7(payload)
    for gate in payload["gates"]:
        C.log(f"[gate] {gate['gate']}: {gate['status']}")
    C.log(f"[gates] all_passed={payload['all_passed']}")
    return 0 if payload["all_passed"] else 1


def cmd_resource_audit(_args) -> int:
    payload = RA.run_resource_audit()
    C.write_json(C.ANALYSIS / "resource_isolation.json", payload)
    C.log(f"[resource] status={payload['status']} blocking={payload['blocking_reasons']}")
    if (C.ANALYSIS / "gates.json").exists():
        gates_payload = C.read_json(C.ANALYSIS / "gates.json")
        gates_payload["gates"] = [g for g in gates_payload.get("gates", []) if g.get("gate") != "G7_resource_and_conflict"] + [_g7_record()]
        gates_payload["all_passed"] = all(g.get("passed") for g in gates_payload["gates"])
        C.write_json(C.ANALYSIS / "gates.json", gates_payload)
    return 0 if payload["status"] == "PASS" else 1


def cmd_benchmark(args) -> int:
    torch.set_num_threads(int(args.threads))
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    caches = D.load_caches(spec)
    fit = fit_labels.load_fit_labels(arm_guard=True)
    y_fit = fit["y"]
    target = D.load_target_norm(verify=False)
    fit_rows = [int(r) for r in caches["fit"]["rows"]]
    n = C.SPLIT_SIZES["fit"]
    train_positions = D.fit_training_positions(n)
    fit_cache = caches["fit"]["cache"]
    out = {}
    for arm in C.ARMS:
        model, info = M.build_model(arm, spec, norm, 0, "cpu")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.TOTAL_UPDATES, eta_min=C.LR_MIN)
        gen = torch.Generator().manual_seed(0 + C.TRAIN_SHUFFLE_OFFSET)
        started = time.perf_counter()
        seen = 0
        sq = 0.0
        for rows in D.shuffle_batches(n, C.BATCH_SIZE, gen):
            plan = D.plan_fit_batch(rows, train_positions=train_positions, cache=fit_cache, fit_global_rows=fit_rows)
            info_batch = D.assemble_fit_batch(fit_cache, y_fit, plan)
            batch = D.to_device(info_batch["batch"], "cpu")
            y = torch.as_tensor((info_batch["y"] - target["mean"]) / target["scale"], dtype=torch.float32)
            pred = model(batch)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            sched.step()
            with torch.no_grad():
                err = pred - y
                sq += float((err * err).sum())
                seen += int(err.numel())
        elapsed = time.perf_counter() - started
        out[arm] = {
            "seconds": elapsed,
            "updates": C.STEPS_PER_EPOCH,
            "train_mse_normalised_last_epoch": sq / max(seen, 1),
            "parameters": info["parameters"],
        }
        C.log(f"[benchmark] {arm}: {elapsed:.2f}s/epoch ({elapsed/C.STEPS_PER_EPOCH*1000:.1f} ms/update)")
    slow = max(out["R"]["seconds"], out["N"]["seconds"])
    estimate = slow * C.EPOCHS * 6
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "device": "cpu",
        "threads": int(args.threads),
        "per_arm": out,
        "slowest_epoch_seconds": slow,
        "estimated_six_runs_seconds": estimate,
        "cpu_allowed_threshold_seconds": 2700,
        "cpu_eligible": estimate <= 2700,
        "note": "measured one complete Fit2352 epoch per arm with forward/backward/AdamW/scheduler",
        "monitor_labels_read": False,
    }
    C.write_json(C.ANALYSIS / "cpu_timing.json", payload)
    C.log(f"[benchmark] slowest={slow:.2f}s estimate_6runs={estimate:.0f}s cpu_eligible={payload['cpu_eligible']}")
    return 0


def cmd_smoke(args) -> int:
    torch.set_num_threads(int(args.threads))
    out = {}
    for arm in C.ARMS:
        cfg = T.TrainConfig(arm=arm, seed=0, device="cpu", smoke=True, fit_limit=int(args.fit_limit), epochs=int(args.epochs), threads=int(args.threads))
        out[arm] = T.train_one(cfg)
        C.log(f"[smoke] {arm}: fit_rmse={out[arm]['fit_diagnostics']['soup_fit_rmse_raw']:.5f}")
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "device": "cpu",
        "threads": int(args.threads),
        "fit_limit": int(args.fit_limit),
        "epochs": int(args.epochs),
        "runs": out,
        "monitor_labels_read": False,
    }
    C.write_json(C.ANALYSIS / "smoke.json", payload)
    return 0


def cmd_select_backend(args) -> int:
    timing = _read_or_none(C.ANALYSIS / "cpu_timing.json")
    if args.device == "cpu":
        if not timing or not timing.get("cpu_eligible"):
            raise PermissionError("CPU backend requires a measured eligible cpu_timing.json")
        estimate = timing["estimated_six_runs_seconds"]
        reason = f"measured CPU estimate {estimate:.0f}s <= 2700s; Peptides runs on a separate remote host"
    else:
        reason = "CPU estimate exceeded the allowed budget or CPU was intentionally ruled out; use the isolated A100"
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "device": args.device,
        "threads": int(args.threads),
        "CUDA_VISIBLE_DEVICES": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_info": C.gpu_runtime_info() if args.device == "cuda" else None,
        "same_backend_for_all_six": True,
        "cpu_timing": timing,
        "selection_reason": reason,
        "gpu1_was_observed_free": (C.read_json(C.ANALYSIS / "resource_isolation.json").get("checks", {}).get("selected_gpu_free") if (C.ANALYSIS / "resource_isolation.json").exists() else None),
    }
    C.write_json(C.ANALYSIS / "backend.json", payload)
    C.log(f"[backend] {args.device} threads={args.threads} reason={reason}")
    return 0


def _backend_record() -> dict[str, Any]:
    return _read_or_none(C.ANALYSIS / "backend.json") or {}


def cmd_train(args) -> int:
    C.verify_protocol_freeze(require_clean=True)
    gates_payload = _read_or_none(C.ANALYSIS / "gates.json")
    if not gates_payload or not gates_payload.get("all_passed"):
        raise PermissionError("correctness gates are not all PASS")
    res = _read_or_none(C.ANALYSIS / "resource_isolation.json")
    if not res or res.get("status") != "PASS":
        raise PermissionError("resource isolation gate is not PASS")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        vis = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
        if vis != "1":
            raise PermissionError(f"formal GPU run requires CUDA_VISIBLE_DEVICES=1, got {vis!r}")
    backend = _backend_record()
    if backend and backend.get("device") != args.device:
        raise PermissionError(f"all six formal runs must use the same backend; existing={backend.get('device')} requested={args.device}")
    if not backend:
        C.write_json(C.ANALYSIS / "backend.json", {
            "protocol_id": C.PROTOCOL_ID,
            "created_utc": C.utcnow(),
            "device": args.device,
            "threads": int(args.threads),
            "CUDA_VISIBLE_DEVICES": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_info": C.gpu_runtime_info() if args.device == "cuda" else None,
            "same_backend_for_all_six": True,
        })
    # Cumulative GPU-hour cap.
    used = 0.0
    for arm in C.ARMS:
        for seed in C.SEEDS:
            p = C.RUNS / f"{arm}_s{seed}" / "train.json"
            if p.exists():
                used += float(C.read_json(p).get("gpu_hours", 0.0) or 0.0)
    if args.device == "cuda" and used >= C.GPU_HOUR_CAP:
        raise PermissionError(f"GPU-hour budget exhausted: used={used:.4f} cap={C.GPU_HOUR_CAP}")
    cfg = T.TrainConfig(arm=args.arm, seed=args.seed, device=args.device, smoke=False, threads=int(args.threads))
    payload = T.train_one(cfg)
    total = used + float(payload.get("gpu_hours", 0.0) or 0.0)
    if args.device == "cuda" and total > C.GPU_HOUR_CAP:
        raise RuntimeError(f"GPU-hour cap exceeded after run: total={total:.4f} cap={C.GPU_HOUR_CAP}")
    C.log(f"[train] {args.arm}_s{args.seed} complete gpu_hours_total={total:.4f}")
    return 0


def cmd_predict(args) -> int:
    C.verify_protocol_freeze(require_clean=False)
    keys = [(args.arm, args.seed)] if args.arm is not None else [(a, s) for a in C.ARMS for s in C.SEEDS]
    for arm, seed in keys:
        P.predict_nn(arm, seed, device=args.device)
    return 0


def cmd_freeze(_args) -> int:
    manifest = P.freeze_predictions()
    C.log(f"[freeze] status={manifest['status']} manifest_sha256={manifest.get('manifest_sha256')}")
    return 0


def cmd_evaluate(_args) -> int:
    result = E.evaluate()
    C.log(f"[evaluate] case={result['decision']['case']} mean_R-A={result['decision']['mean_rmse_R_minus_A']:.4f} mean_R-N={result['decision']['mean_rmse_R_minus_N']:.4f}")
    return 0


def cmd_status(_args) -> int:
    files = [
        "preflight.json", "resource_isolation.json", "data_audit.json", "gates.json",
        "parameter_audit.json", "cpu_timing.json", "backend.json",
        "prediction_manifest.json", "prediction_manifest.sha256",
        "paired_metrics.json", "mechanism_diagnostics.json",
    ]
    for name in files:
        p = C.ANALYSIS / name
        C.log(f"{name}: {'present' if p.exists() else 'missing'}")
    for arm in C.ARMS:
        for seed in C.SEEDS:
            p = C.RUNS / f"{arm}_s{seed}" / "train.json"
            C.log(f"{arm}_s{seed}: {'complete' if p.exists() else 'missing'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-data")
    sub.add_parser("preflight")
    sub.add_parser("gates")
    sub.add_parser("resource-audit")
    pbench = sub.add_parser("benchmark")
    pbench.add_argument("--threads", type=int, default=4)
    psmoke = sub.add_parser("smoke")
    psmoke.add_argument("--threads", type=int, default=4)
    psmoke.add_argument("--fit-limit", type=int, default=128)
    psmoke.add_argument("--epochs", type=int, default=2)
    psel = sub.add_parser("select-backend")
    psel.add_argument("--device", choices=("cpu", "cuda"), required=True)
    psel.add_argument("--threads", type=int, default=4)
    ptra = sub.add_parser("train")
    ptra.add_argument("--arm", choices=C.ARMS, required=True)
    ptra.add_argument("--seed", type=int, choices=C.SEEDS, required=True)
    ptra.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ptra.add_argument("--threads", type=int, default=2)
    ppre = sub.add_parser("predict")
    ppre.add_argument("--arm", choices=C.ARMS, default=None)
    ppre.add_argument("--seed", type=int, choices=C.SEEDS, default=None)
    ppre.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    sub.add_parser("freeze")
    sub.add_parser("evaluate")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "predict" and (args.arm is None) != (args.seed is None):
        parser.error("--arm and --seed must be supplied together")
    fn = {
        "verify-data": cmd_verify_data,
        "preflight": cmd_preflight,
        "gates": cmd_gates,
        "resource-audit": cmd_resource_audit,
        "benchmark": cmd_benchmark,
        "smoke": cmd_smoke,
        "select-backend": cmd_select_backend,
        "train": cmd_train,
        "predict": cmd_predict,
        "freeze": cmd_freeze,
        "evaluate": cmd_evaluate,
        "status": cmd_status,
    }[args.command]
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
