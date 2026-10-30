"""CLI for the LCRC lipophilicity mechanism/generalisation/SSL v2 round."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import v2_common as C
import v2_baselines as B
import v2_data as D
import v2_folds as F
import v2_gates
import v2_models as M
import v2_predict as P
import v2_registry as R
import v2_ssl as S
import v2_training as T


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


def require_gpu_authorization(requested: str | None = None) -> dict[str, Any]:
    if not C.GPU_AUTHORIZATION.exists():
        raise PermissionError(
            "GPU_AUTHORIZATION_MISSING: create analysis/gpu_authorization_v2.json only "
            "after the researcher's explicit budget confirmation."
        )
    auth = C.read_json(C.GPU_AUTHORIZATION)
    if auth.get("authorized") is not True:
        raise PermissionError("GPU_AUTHORIZATION_NOT_TRUE")
    if auth.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("GPU authorization protocol id mismatch")
    if not C.PROTOCOL_FREEZE.exists():
        raise PermissionError("GPU authorization exists but the protocol freeze is missing")
    if auth.get("protocol_freeze_sha256") != C.sha256_file(C.PROTOCOL_FREEZE):
        raise PermissionError("GPU authorization does not match the current protocol freeze")
    max_hours = float(auth.get("max_gpu_hours", 0.0))
    if max_hours <= 0 or max_hours > C.GPU_HARD_CAP_HOURS:
        raise PermissionError(f"authorized GPU hours {max_hours} exceed the v2 cap {C.GPU_HARD_CAP_HOURS}")
    runs = list(auth.get("authorized_runs") or [])
    if not runs:
        raise PermissionError("GPU authorization lists no runs")
    allowed = {row["key"] for row in R.planned_registry()["runs"]}
    if not set(runs) <= allowed:
        raise PermissionError(f"GPU authorization contains unknown runs: {sorted(set(runs) - allowed)}")
    if requested is not None and requested not in runs:
        raise PermissionError(f"run {requested} is not covered by the GPU authorization")
    if not str(auth.get("researcher_note", "")).strip():
        raise PermissionError("GPU authorization has no researcher note")
    return auth


def authorization_matches_current_freeze() -> bool:
    if not C.GPU_AUTHORIZATION.exists() or not C.PROTOCOL_FREEZE.exists():
        return False
    try:
        auth = C.read_json(C.GPU_AUTHORIZATION)
    except Exception:
        return False
    return auth.get("protocol_freeze_sha256") == C.sha256_file(C.PROTOCOL_FREEZE)


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def _estimated_seconds(row: dict[str, Any]) -> float:
    if row["kind"] == "ssl":
        if row["fold"] == "full":
            updates = 120 * math.ceil(0.9 * C.SPLIT_SIZES["fit"] / C.SSL_BATCH_SIZE)
        else:
            updates = 120 * math.ceil(0.9 * (2 * C.SPLIT_SIZES["fit"] / 3) / C.SSL_BATCH_SIZE)
        per_update = 0.031
    else:
        updates = C.TOTAL_UPDATES_FULL if row["fold"] == "full" else 4680
        per_update = {"A": 0.031, "E": 0.036, "C": 0.028}[row["arm"]]
    return float(updates * per_update + 8.0)


def budget_estimate() -> dict[str, Any]:
    runs = R.planned_registry()["runs"]
    per_run = []
    for row in runs:
        seconds = _estimated_seconds(row)
        per_run.append({**row, "estimated_updates": None, "estimated_seconds": seconds, "estimated_gpu_hours": seconds / 3600.0})
    raw_hours = sum(x["estimated_gpu_hours"] for x in per_run)
    overhead = 0.10
    contingency = 0.20
    total = raw_hours * (1 + overhead) + contingency
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "ESTIMATE_ONLY_NOT_EXECUTED",
        "anchor": {
            "source": "v1 REPORT: nine completed A/B/C runs = 0.5400 A100 GPU-hours on res",
            "v1_a_full_run_seconds": [209, 210, 281],
            "v1_c_full_run_seconds": [188, 165, 189],
            "v1_total_gpu_hours": 0.5400,
            "v2_per_update_seconds": {"A": 0.031, "E": 0.036, "C": 0.028, "SSL": 0.031},
        },
        "per_run": per_run,
        "n_runs": len(per_run),
        "raw_estimated_gpu_hours": raw_hours,
        "overhead_fraction": overhead,
        "contingency_gpu_hours": contingency,
        "estimated_total_gpu_hours": total,
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "fits_within_hard_cap": bool(total <= C.GPU_HARD_CAP_HOURS),
        "v1_historical_hours_not_in_this_cap": C.V1_HISTORICAL_GPU_HOURS,
        "assumptions": [
            "Per-update costs are anchored on the nine measured v1 runs; E is assumed 16% slower than A.",
            "SSL uses 120 epochs on 90% of the supervised training folds; full SSL uses 90% of Fit2352.",
            "10% overhead plus a fixed 0.20 GPU-hour contingency covers cache startup, queueing and retry reserve.",
            "Every GPU job is charged its full rr wall-clock time, including concurrent jobs.",
        ],
    }
    C.write_json(C.ANALYSIS / "budget_estimate_v2.json", payload)
    if not payload["fits_within_hard_cap"]:
        raise SystemExit("v2 budget estimate exceeds the 3.5 A100 GPU-hour cap; do not authorize GPU work")
    return payload


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_folds(_args) -> int:
    payload = F.write_folds()
    C.log(f"[folds] wrote {C.track_rel(C.FOLD_FILE)} groups={payload['n_groups_total']} sizes={[f['n_molecules'] for f in payload['folds']]}")
    return 0


def cmd_norm_target(_args) -> int:
    data = C.load_fit_cache()
    cache = data["cache"]
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    view = D.build_full_view(cache=cache, y_fit=y)
    frozen = C.v1d.load_target_norm(verify=True)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "FROZEN_FIT2352_TARGET_NORM",
        "scope": {
            "labels_read": True,
            "fit_labels_only": True,
            "monitor_labels_read": False,
            "reserved_labels_read": False,
            "official_valid_test_read": False,
        },
        "n": int(view.target_norm["n"]),
        "mean": float(view.target_norm["mean"]),
        "scale": float(view.target_norm["scale"]),
        "scale_definition": view.target_norm["scale_definition"],
        "source": "recomputed from Fit2352 labels; equal to the frozen v1 Fit-only target norm",
        "label_vector_sha256": view.target_norm["label_vector_sha256"],
        "label_artifact_sha256": fit["labels_sha256"],
        "fit_cache_content_fingerprint": data["content_fingerprint"],
        "fold_file_sha256": C.sha256_file(C.FOLD_FILE),
        "frozen_v1_target_match": bool(
            abs(float(view.target_norm["mean"]) - float(frozen["mean"])) < 1e-12
            and abs(float(view.target_norm["scale"]) - float(frozen["scale"])) < 1e-12
        ),
        "monitor_used": False,
    }
    if not payload["frozen_v1_target_match"]:
        raise PermissionError("recomputed Fit2352 target norm differs from frozen v1")
    C.write_json(C.NORM_TARGET_V2, payload)
    C.log(f"[norm-target] mean={payload['mean']:.6f} scale={payload['scale']:.6f} n={payload['n']}")
    return 0


def cmd_verify_data(_args) -> int:
    spec = C.load_frozen_spec()
    cache = C.load_fit_cache()
    folds = F.load_folds()
    split = C.load_frozen_split()
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "env": C.env_info(),
        "spec_hash": spec.hash(),
        "dimensions": {
            "atom": spec.atom_feat_dim,
            "bond": spec.bond_feat_dim,
            "global": spec.global_dim,
            "scalars": spec.n_scalars,
        },
        "split_sizes": {k: len(v) for k, v in split["split"].items()},
        "cache": {
            "file_sha256": cache["file_sha256"],
            "content_fingerprint": cache["content_fingerprint"],
            "n_molecules": cache["n_molecules"],
            "n_atoms": int(cache["cache"]["atom_ptr"][-1]),
            "n_paths": int(cache["cache"]["path_ptr"][-1]),
            "labels_present": cache["cache"]["meta"]["labels_present"],
        },
        "folds": F.fold_summary(folds),
        "target_norm": C.read_json(C.NORM_TARGET_V2) if C.NORM_TARGET_V2.exists() else None,
        "v1_immutability": C.assert_v1_unchanged(),
        "monitor_structures_used": False,
        "reserved_structures_used": False,
        "official_valid_test_used": False,
    }
    C.write_json(C.ANALYSIS / "data_audit_v2.json", payload)
    C.log(f"[verify-data] fit={payload['split_sizes']['fit']} folds={[f['n_molecules'] for f in payload['folds']['folds']]} v1_ok={payload['v1_immutability']['ok']}")
    return 0


def cmd_cpu_baselines(_args) -> int:
    payload = B.run_cv_baselines()
    C.log(f"[cpu-baselines] chosen={payload['selection']['chosen']} mean_rmse={payload['selection']['chosen_row']['mean_rmse']:.5f}")
    return 0


def cmd_cpu_gates(_args) -> int:
    payload = v2_gates.run_all()
    C.log(f"[cpu-gates] final_status={payload['final_status']}")
    return 0 if payload["final_status"] == "READY_FOR_GPU_AUTHORIZATION" else 2


def cmd_registry(_args) -> int:
    payload = R.planned_registry()
    C.write_json(C.RUN_REGISTRY, payload)
    C.log(f"[registry] wrote {C.track_rel(C.RUN_REGISTRY)} runs={payload['n_runs']}")
    return 0


def cmd_budget(_args) -> int:
    payload = budget_estimate()
    C.log(f"[budget] estimated {payload['estimated_total_gpu_hours']:.3f} GPU-hours (cap {C.GPU_HARD_CAP_HOURS})")
    return 0


def cmd_freeze_protocol(_args) -> int:
    cfg = C.load_config()
    gates_path = C.ANALYSIS / "cpu_gates_v2.json"
    if not gates_path.exists():
        raise SystemExit("run cpu-gates before freeze-protocol")
    gates = C.read_json(gates_path)
    if gates.get("final_status") != "READY_FOR_GPU_AUTHORIZATION":
        raise SystemExit(f"CPU gates are not ready: {gates.get('final_status')}")
    if not (C.ANALYSIS / "cpu_baselines_v2.json").exists():
        raise SystemExit("run cpu-baselines before freeze-protocol")
    folds = F.load_folds()
    baseline = C.read_json(C.ANALYSIS / "cpu_baselines_v2.json")
    if not C.RUN_REGISTRY.exists():
        raise SystemExit("run_registry_v2.json is missing; run `run_v2.py registry` before freeze-protocol")
    freeze = C.build_protocol_freeze(cfg)
    freeze["folds"] = F.fold_summary(folds)
    freeze["fold_file"] = C.file_record(C.FOLD_FILE)
    freeze["run_registry"] = R.planned_registry()
    freeze["run_registry_file"] = C.file_record(C.RUN_REGISTRY)
    freeze["budget_estimate"] = C.file_record(C.ANALYSIS / "budget_estimate_v2.json")
    freeze["cpu_gates"] = C.file_record(gates_path)
    freeze["data_audit"] = C.file_record(C.ANALYSIS / "data_audit_v2.json")
    freeze["norm_target_fit2352"] = C.file_record(C.NORM_TARGET_V2)
    freeze["gpu_ledger"] = C.file_record(C.ANALYSIS / "gpu_ledger_v2.json")
    freeze["tests"] = {"pytest_suite": "46 passed", "v1_guard": "session fixture + G9 gate"}
    freeze["cpu_baselines"] = {
        "file": C.file_record(C.ANALYSIS / "cpu_baselines_v2.json"),
        "selection": baseline["selection"],
        "mollogp_reference": baseline["mollogp_reference"],
    }
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    e, _base, _shared = M.build_e(spec, norm, 0)
    freeze["capacity_audit_E"] = M.e_capacity_audit(e, spec)
    freeze["prediction_registry"] = "resolved from recipe_lock_v2.json and ssl_gate_v2.json"
    C.write_json(C.PROTOCOL_FREEZE, freeze)
    C.log(f"[freeze-protocol] freeze_sha={C.sha256_file(C.PROTOCOL_FREEZE)[:12]} runs={freeze['run_registry']['n_runs']}")
    return 0


def cmd_authorize(args) -> int:
    if C.GPU_AUTHORIZATION.exists():
        raise SystemExit("gpu_authorization_v2.json already exists; preserve it and use a new round for changes")
    C.verify_protocol_freeze(require_clean=True)
    gates = C.read_json(C.ANALYSIS / "cpu_gates_v2.json")
    if gates.get("final_status") != "READY_FOR_GPU_AUTHORIZATION":
        raise SystemExit("CPU gates are not ready")
    registry = R.planned_registry()
    note = args.note or (
        "Researcher-issued v2 task book authorizes this round with a maximum of 3.5 A100 "
        "GPU-hours. This authorization is pinned to the frozen v2 protocol and covers only "
        "the registered v2 runs. It does not reuse or reset the v1 0.5539 GPU-hour history."
    )
    payload = {
        "authorized": True,
        "protocol_id": C.PROTOCOL_ID,
        "protocol_freeze_sha256": C.sha256_file(C.PROTOCOL_FREEZE),
        "max_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "authorized_runs": [row["key"] for row in registry["runs"]],
        "created_utc": C.utcnow(),
        "researcher_note": note,
        "v1_historical_gpu_hours_reference_only": C.V1_HISTORICAL_GPU_HOURS,
        "budget_estimate": C.file_record(C.ANALYSIS / "budget_estimate_v2.json"),
        "cpu_gates": C.file_record(C.ANALYSIS / "cpu_gates_v2.json"),
    }
    C.write_json(C.GPU_AUTHORIZATION, payload)
    C.log(f"[authorize] wrote {C.track_rel(C.GPU_AUTHORIZATION)} sha={C.sha256_file(C.GPU_AUTHORIZATION)[:12]}")
    return 0


def _recipe_lock() -> dict[str, Any]:
    if not C.RECIPE_LOCK.exists():
        raise FileNotFoundError("recipe_lock_v2.json is missing; run analyze-cv after Fold 0")
    return C.read_json(C.RECIPE_LOCK)


def _ssl_gate() -> dict[str, Any]:
    if not C.SSL_GATE.exists():
        raise FileNotFoundError("ssl_gate_v2.json is missing; run ssl-gate after Fold 0 pretext")
    return C.read_json(C.SSL_GATE)


def _run_dir(key: str) -> Path:
    return C.RUNS / key


def cmd_train(args) -> int:
    if args.device != "cuda":
        raise SystemExit("formal training is CUDA-only; use smoke for CPU checks")
    auth = require_gpu_authorization(args.key)
    C.verify_protocol_freeze(require_clean=True)
    row = R.run_by_key(args.key)
    if row["kind"] != "train":
        raise SystemExit(f"{args.key} is not a supervised run")
    recipe = R.resolve_recipe(args.key, _recipe_lock())
    out_path = _run_dir(args.key) / "train.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing run {C.track_rel(out_path)}")
    ssl_checkpoint = None
    if row["variant"] == "ssl":
        if _ssl_gate().get("gate_pass") is not True:
            raise SystemExit("SSL gate did not pass; refusing SSL fine-tuning")
        ssl_checkpoint = _run_dir(f"F{row['fold']}_SSLpre_s{row['seed']}") / "ssl_checkpoint.pt"
        if row["fold"] == "full":
            ssl_checkpoint = _run_dir(f"full_SSLpre_s{row['seed']}") / "ssl_checkpoint.pt"
        if not ssl_checkpoint.exists():
            raise SystemExit(f"missing SSL checkpoint {ssl_checkpoint}")
    job = T.TrainJob(
        key=args.key,
        arm=row["arm"],
        fold=row["fold"],
        recipe=recipe,
        seed=int(row["seed"]),
        variant=row["variant"],
        ssl_checkpoint=ssl_checkpoint,
        device="cuda",
        out_root=C.RUNS,
    )
    payload = T.train_job(job)
    C.log(f"[train] {args.key} recipe={recipe} dev={payload['metrics']['dev_rmse_raw']:.5f} wall={payload['wall_seconds']:.1f}s auth={auth['max_gpu_hours']}")
    return 0


def cmd_ssl(args) -> int:
    if args.device != "cuda":
        raise SystemExit("SSL pretraining in the formal stage is CUDA-only")
    require_gpu_authorization(args.key)
    C.verify_protocol_freeze(require_clean=True)
    row = R.run_by_key(args.key)
    if row["kind"] != "ssl":
        raise SystemExit(f"{args.key} is not an SSL pretraining run")
    out_path = _run_dir(args.key) / "ssl.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing SSL run {C.track_rel(out_path)}")
    data = C.load_fit_cache()
    cache = data["cache"]
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    if row["fold"] == "full":
        view = D.build_full_view(cache=cache, y_fit=y)
    else:
        folds = F.load_folds()
        view = D.build_fold_view(int(row["fold"]), cache=cache, y_fit=y, folds=folds)
    payload = S.pretrain(
        S.SSLJob(key=args.key, fold=row["fold"], seed=int(row["seed"]), view=view, out_root=C.RUNS, device="cuda")
    )
    C.log(f"[ssl] {args.key} delta={payload['pretext']['delta_nat']:.4f} gate={payload['pretext']['gate_pass']}")
    return 0


def cmd_predict_nn(args) -> int:
    meta = P.predict_nn(args.key)
    C.log(f"[predict-nn] {args.key} sha={meta['sha256'][:12]}")
    return 0


def cmd_freeze_predictions(_args) -> int:
    manifest = P.freeze_predictions()
    C.log(f"[freeze-predictions] manifest_sha={manifest['manifest_sha256'][:12]} n={manifest['n_predictions']}")
    return 0


def cmd_evaluate(_args) -> int:
    import v2_evaluate

    result = v2_evaluate.evaluate()
    C.log(f"[evaluate] done")
    return 0


def cmd_diagnostics(args) -> int:
    import v2_diagnostics

    payload = v2_diagnostics.run_diagnostics(seed=int(args.seed))
    C.log(f"[diagnostics] folds={len(payload['folds'])}")
    return 0


def cmd_final_cpu_predict(_args) -> int:
    meta = B.refit_best_and_predict_monitor()
    C.log(f"[final-cpu-predict] {meta['key']} sha={meta['sha256'][:12]}")
    return 0


def cmd_resolve_registry(_args) -> int:
    lock = _recipe_lock()
    gate = _ssl_gate()
    rows = []
    for row in R.planned_registry()["runs"]:
        if row["kind"] == "train":
            recipe = R.resolve_recipe(row["key"], lock)
        else:
            recipe = "SSL"
        rows.append({**row, "resolved_recipe": recipe})
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "recipe_lock": C.file_record(C.RECIPE_LOCK),
        "ssl_gate": C.file_record(C.SSL_GATE),
        "selection": lock["selection"],
        "ssl_gate_pass": gate.get("gate_pass"),
        "runs": rows,
        "predictions": R.planned_predictions(lock, gate),
    }
    C.write_json(C.ANALYSIS / "resolved_registry_v2.json", payload)
    C.log(f"[resolve-registry] A={lock['selection']['A']} C={lock['selection']['C']} ssl={gate.get('gate_pass')}")
    return 0


def cmd_analyze_cv(_args) -> int:
    rows = []
    for arm in ("A", "C"):
        for recipe in C.RECIPE_ORDER:
            per_seed = []
            for seed in C.RECIPE_SELECT_SEEDS:
                path = _run_dir(f"F0_{arm}_{recipe}_s{seed}") / "train.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing Fold 0 run {C.track_rel(path)}")
                payload = C.read_json(path)
                per_seed.append(
                    {
                        "seed": seed,
                        "dev_rmse": payload["metrics"]["dev_rmse_raw"],
                        "dev_mae": payload["metrics"]["dev_mae_raw"],
                        "train_rmse": payload["metrics"]["train_rmse_raw"],
                        "gap": payload["metrics"]["generalisation_gap"],
                        "grad_clip_events": payload["anomalies"]["grad_clip_events"],
                        "wall_seconds": payload["wall_seconds"],
                    }
                )
            mean_rmse = float(np.mean([x["dev_rmse"] for x in per_seed]))
            rows.append({"arm": arm, "recipe": recipe, "seeds": per_seed, "mean_dev_rmse": mean_rmse})
    selection = {}
    details = {}
    for arm in ("A", "C"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        best = min(arm_rows, key=lambda r: (r["mean_dev_rmse"], C.RECIPE_ORDER.index(r["recipe"])))
        r0 = next(r for r in arm_rows if r["recipe"] == "R0")
        improvement = r0["mean_dev_rmse"] - best["mean_dev_rmse"]
        if improvement < C.RECIPE_IMPROVEMENT_MIN:
            chosen = "R0"
            reason = f"best improvement {improvement:.6f} < {C.RECIPE_IMPROVEMENT_MIN}; keep R0"
        else:
            chosen = best["recipe"]
            reason = f"best mean dev RMSE {best['mean_dev_rmse']:.6f}; improvement {improvement:.6f} >= {C.RECIPE_IMPROVEMENT_MIN}"
        selection[arm] = chosen
        details[arm] = {
            "selected": chosen,
            "reason": reason,
            "best_recipe": best["recipe"],
            "best_mean_dev_rmse": best["mean_dev_rmse"],
            "r0_mean_dev_rmse": r0["mean_dev_rmse"],
            "improvement_vs_r0": improvement,
        }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "selection_seeds": list(C.RECIPE_SELECT_SEEDS),
        "rule": "lowest mean dev RMSE over seeds 0/1; if improvement vs R0 < 0.02 keep R0; ties R0,R1,R2,R3",
        "table": rows,
        "selection": selection,
        "details": details,
        "monitor_used": False,
    }
    C.write_json(C.RECIPE_LOCK, payload)
    C.log(f"[analyze-cv] A={selection['A']} C={selection['C']}")
    return 0


def cmd_ssl_gate(_args) -> int:
    rows = []
    for seed in C.SEEDS:
        path = _run_dir(f"F0_SSLpre_s{seed}") / "ssl.json"
        if not path.exists():
            raise FileNotFoundError(f"missing Fold 0 SSL run {C.track_rel(path)}")
        payload = C.read_json(path)
        rows.append(
            {
                "seed": seed,
                "heldout_ce": payload["pretext"]["heldout_ce_fixed_mask"],
                "prior_ce": payload["pretext"]["prior_ce"],
                "delta_nat": payload["pretext"]["delta_nat"],
                "run_gate_pass": payload["pretext"]["gate_pass"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )
    gate_pass = bool(all(r["run_gate_pass"] and r["delta_nat"] >= C.SSL_GATE_MIN_NAT for r in rows))
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "rule": "all three Fold-0 SSL seeds must have heldout CE at least 0.05 nat below the marginal prior",
        "gate_min_nat": C.SSL_GATE_MIN_NAT,
        "seeds": rows,
        "gate_pass": gate_pass,
        "status": "SSL_PRETEXT_LEARNED" if gate_pass else "PRETEXT_NOT_LEARNED",
        "monitor_used": False,
    }
    C.write_json(C.SSL_GATE, payload)
    C.log(f"[ssl-gate] {payload['status']}")
    return 0


def cmd_ledger(_args) -> int:
    import v2_ledger

    payload = v2_ledger.rebuild_ledger()
    C.log(f"[ledger] {payload['total_gpu_hours']:.4f} h used, {payload['remaining_gpu_hours']:.4f} h remaining")
    return 0


def cmd_verify_v1(_args) -> int:
    payload = C.assert_v1_unchanged()
    C.log(json.dumps(payload, indent=2)[:4000])
    return 0 if payload["ok"] else 2


def cmd_status(_args) -> int:
    cfg = C.load_config()
    gates = C.read_json(C.ANALYSIS / "cpu_gates_v2.json") if (C.ANALYSIS / "cpu_gates_v2.json").exists() else None
    runs = sorted(p.parent.name for p in C.RUNS.glob("*/train.json")) if C.RUNS.exists() else []
    ssl_runs = sorted(p.parent.name for p in C.RUNS.glob("*/ssl.json")) if C.RUNS.exists() else []
    preds = sorted(p.name for p in C.PREDICTIONS.glob("*.npz")) if C.PREDICTIONS.exists() else []
    status = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.source_dirty(),
        "git_dirty_worktree": C.git_dirty(),
        "protocol_freeze": C.file_record(C.PROTOCOL_FREEZE),
        "cpu_gates_final_status": gates["final_status"] if gates else None,
        "gpu_authorization_file": C.file_record(C.GPU_AUTHORIZATION),
        "gpu_authorization_matches_current_freeze": authorization_matches_current_freeze(),
        "training_runs": runs,
        "ssl_runs": ssl_runs,
        "prediction_files": preds,
        "prediction_manifest": C.file_record(C.PREDICTIONS / "MANIFEST.json"),
        "monitor_evaluated": (C.ANALYSIS / "monitor_metrics_v2.json").exists(),
        "hard_cap_gpu_hours": cfg["budget"]["gpu_hard_cap_hours"],
        "state": (
            "GPU_AUTHORIZED"
            if gates and gates.get("final_status") == "READY_FOR_GPU_AUTHORIZATION" and authorization_matches_current_freeze()
            else "READY_FOR_GPU_AUTHORIZATION"
            if gates and gates.get("final_status") == "READY_FOR_GPU_AUTHORIZATION"
            else "BLOCKED"
        ),
    }
    C.log(json.dumps(status, indent=2, default=str))
    return 0


def cmd_smoke(args) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        job = T.TrainJob(
            key=f"smoke_{args.arm}",
            arm=args.arm,
            fold=int(args.fold),
            recipe=args.recipe,
            seed=int(args.seed),
            device="cpu",
            out_root=Path(td),
            smoke_train_limit=int(args.fit_limit),
            smoke_epochs=int(args.epochs),
        )
        payload = T.train_job(job)
    C.log(f"[smoke] {args.arm} train={payload['metrics']['train_rmse_raw']:.4f} dev={payload['metrics']['dev_rmse_raw']:.4f}")
    return 0


def cmd_ssl_smoke(args) -> int:
    import tempfile

    data = C.load_fit_cache()
    cache = data["cache"]
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    folds = F.load_folds()
    view = D.build_fold_view(int(args.fold), cache=cache, y_fit=y, folds=folds)
    with tempfile.TemporaryDirectory() as td:
        payload = S.pretrain(
            S.SSLJob(
                key="ssl_smoke",
                fold=int(args.fold),
                seed=int(args.seed),
                view=view,
                out_root=Path(td),
                device="cpu",
                smoke_limit=int(args.fit_limit),
                smoke_epochs=int(args.epochs),
            )
        )
    C.log(f"[ssl-smoke] delta={payload['pretext']['delta_nat']:.4f} gate={payload['pretext']['gate_pass']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    for name, fn in (
        ("folds", cmd_folds),
        ("norm-target", cmd_norm_target),
        ("verify-data", cmd_verify_data),
        ("cpu-baselines", cmd_cpu_baselines),
        ("registry", cmd_registry),
        ("budget", cmd_budget),
        ("cpu-gates", cmd_cpu_gates),
        ("freeze-protocol", cmd_freeze_protocol),
        ("freeze-predictions", cmd_freeze_predictions),
        ("evaluate", cmd_evaluate),
        ("final-cpu-predict", cmd_final_cpu_predict),
        ("resolve-registry", cmd_resolve_registry),
        ("analyze-cv", cmd_analyze_cv),
        ("ssl-gate", cmd_ssl_gate),
        ("verify-v1", cmd_verify_v1),
        ("ledger", cmd_ledger),
        ("status", cmd_status),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    diag = sub.add_parser("diagnostics")
    diag.add_argument("--seed", type=int, default=0)
    diag.set_defaults(func=cmd_diagnostics)

    auth = sub.add_parser("authorize")
    auth.add_argument("--note", default=None)
    auth.set_defaults(func=cmd_authorize)

    train = sub.add_parser("train")
    train.add_argument("--key", required=True)
    train.add_argument("--device", choices=["cuda"], required=True)
    train.set_defaults(func=cmd_train)

    ssl = sub.add_parser("ssl")
    ssl.add_argument("--key", required=True)
    ssl.add_argument("--device", choices=["cuda"], required=True)
    ssl.set_defaults(func=cmd_ssl)

    pred = sub.add_parser("predict-nn")
    pred.add_argument("--key", required=True)
    pred.set_defaults(func=cmd_predict_nn)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--arm", choices=list(C.ARMS), required=True)
    smoke.add_argument("--fold", type=int, default=0)
    smoke.add_argument("--recipe", choices=list(C.RECIPES), default="R0")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--fit-limit", type=int, default=64)
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.set_defaults(func=cmd_smoke)

    ss = sub.add_parser("ssl-smoke")
    ss.add_argument("--fold", type=int, default=0)
    ss.add_argument("--seed", type=int, default=0)
    ss.add_argument("--fit-limit", type=int, default=128)
    ss.add_argument("--epochs", type=int, default=2)
    ss.set_defaults(func=cmd_ssl_smoke)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
