"""CLI entry point for the LCRC Lipophilicity screen.

Stages
------
``verify-data``          verify frozen artifacts and cache row alignment (no labels)
``freeze-protocol``      write Fit-only target normalisation, parameter table,
                         budget estimate and the protocol freeze record
``parameters``           write the parameter/budget tables only
``cpu-gates``            run the CPU correctness gates -> analysis/cpu_gates.json
``smoke``                short CPU-only engineering smoke on a Fit subset
``train``                formal recipe (CUDA only; requires an authorization file)
``predict-nn``           label-blind prediction for one neural arm/seed
``baseline-ecfp``        ECFP4 + XGBoost, Fit2352 -> Monitor504 predictions
``freeze-predictions``   hash and lock every registered prediction
``evaluate``             read Monitor labels and compute the frozen metrics
``status``               print the current round status

GPU training is mechanically blocked: ``train --device cuda`` and any CUDA
prediction require ``analysis/gpu_authorization.json`` with an explicit
researcher authorization matching the frozen protocol hash.  That file does not
exist in this round.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

import access_guard
import baseline_ecfp_xgb as BASE
import screen_gates
import screen_common as C
import screen_data as D
import screen_models as M
import screen_predict as P
import screen_training as T

GPU_AUTHORIZATION = C.ANALYSIS / "gpu_authorization.json"


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


def require_gpu_authorization(requested: str | None = None) -> dict[str, Any]:
    """Refuse any CUDA work unless the researcher authorization file is valid."""
    if not GPU_AUTHORIZATION.exists():
        raise PermissionError(
            "GPU_AUTHORIZATION_MISSING: no analysis/gpu_authorization.json. "
            "This round is READY_FOR_GPU_AUTHORIZATION only; GPU training must not start."
        )
    auth = C.read_json(GPU_AUTHORIZATION)
    if auth.get("authorized") is not True:
        raise PermissionError("GPU_AUTHORIZATION_NOT_TRUE")
    if auth.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("GPU authorization protocol id mismatch")
    if not C.PROTOCOL_FREEZE.exists():
        raise PermissionError("GPU authorization exists but the protocol freeze is missing")
    if auth.get("protocol_freeze_sha256") != C.sha256_file(C.PROTOCOL_FREEZE):
        raise PermissionError("GPU authorization does not match the current protocol freeze")
    max_hours = float(auth.get("max_gpu_hours", 0.0))
    if max_hours <= 0 or max_hours > float(C.load_config()["budget"]["hard_cap_gpu_hours"]):
        raise PermissionError(
            f"authorized GPU hours {max_hours} exceed the frozen hard cap "
            f"{C.load_config()['budget']['hard_cap_gpu_hours']}"
        )
    runs = list(auth.get("authorized_runs") or [])
    if not runs:
        raise PermissionError("GPU authorization lists no runs")
    allowed = {f"{arm}_s{seed}" for arm in C.NN_ARMS for seed in C.SEEDS}
    if not set(runs) <= allowed:
        raise PermissionError(f"GPU authorization contains unknown runs: {sorted(set(runs) - allowed)}")
    if requested is not None and requested not in runs:
        raise PermissionError(f"run {requested} is not covered by the GPU authorization")
    if not str(auth.get("researcher_note", "")).strip():
        raise PermissionError("GPU authorization has no researcher note")
    return auth


# ---------------------------------------------------------------------------
# data / parameter / budget artifacts
# ---------------------------------------------------------------------------


def cmd_verify_data(_args) -> int:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    split = C.load_frozen_split()
    caches = D.load_caches(spec)
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
                "n_nbr_codes": spec.n_nbr_codes,
                "n_path_codes": spec.n_path_codes,
            },
        },
        "norm": {
            "sha256": C.sha256_file(C.PREP_ANALYSIS / "norm_lipo.json"),
            "fingerprint": C.prep_data.norm_fingerprint(norm),
            "valid_or_test_used": norm["valid_or_test_used"],
            "global_constant_indices": norm["global_constant_indices"],
            "scalar_constant_indices": norm["scalar_constant_indices"],
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
        "labels": {
            "sha256": C.sha256_file(C.LIPO_LABELS),
            "read_in_this_stage": False,
            "note": "structure/data audit only; labels are read by freeze-protocol (Fit) and evaluate (Monitor)",
        },
        "forbidden_sources_present_but_unread": {
            "csv": C.LIPO_CSV.exists(),
            "official_split": C.LIPO_OFFICIAL_SPLIT.exists(),
            "reserved_cache": (C.PREP_CACHE / "lipo_reserved.pt").exists(),
        },
        "reserved_rows_used": False,
        "official_valid_test_used": False,
        "gpu_used": False,
    }
    C.write_json(C.ANALYSIS / "data_audit.json", payload)
    C.log(
        f"[verify-data] fit={payload['split']['sizes']['fit']} monitor={payload['split']['sizes']['monitor']} "
        f"reserved={payload['split']['sizes']['reserved']} caches={list(caches)}"
    )
    return 0


def _lcrc_parameter_payload() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    by_seed: dict[str, Any] = {}
    for seed in C.SEEDS:
        a, info_a = M.build_model("A", spec, norm, seed, "cpu")
        b, info_b = M.build_model("B", spec, norm, seed, "cpu")
        by_seed[f"seed_{seed}"] = {
            "state_hash_A": info_a["state_hash"],
            "state_hash_B": info_b["state_hash"],
            "init_identical": info_a["state_hash"] == info_b["state_hash"],
            "parameters_A": info_a["parameters"],
            "parameters_B": info_b["parameters"],
            "effective_parameters_B": info_b["effective_parameters"],
        }
    b, info_b = M.build_model("B", spec, norm, 0, "cpu")
    controller = b._s2_controller
    dead = sorted(M.dead_parameter_names_lcrc(b, controller))
    dead_counts = {name: int(b.state_dict()[name].numel()) for name in dead}
    dead_counts["local.mlp.lins.0.weight[:, 320:520]"] = int(b.local.mlp.lins[0].weight[:, controller.start : controller.end].numel())
    return {
        "reference": "lcrc_cross_task_preparation_v1 analysis/cpu_audit.json (247,106)",
        "parameters": info_b["parameters"],
        "breakdown": info_b["parameters_breakdown"],
        "config": info_b["config"],
        "input_widths": info_b["input_widths"],
        "ssl_loaded": False,
        "by_seed": by_seed,
        "arm_B": {
            "s2_localmlp_slice": [controller.start, controller.end],
            "dead_parameters": dead_counts,
            "dead_parameter_total": int(sum(dead_counts.values())),
            "effective_parameters": info_b["effective_parameters"],
            "effective_fraction": info_b["effective_parameters"] / info_b["parameters"],
            "note": "B has the same nominal count but a smaller effective count; not information-matched",
        },
    }


def _gine_parameter_payload() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    table = []
    for hidden in range(128, 193, 2):
        model = M.GINERegressor(spec, norm, hidden=hidden, layers=C.GINE_LAYERS)
        table.append({"hidden": hidden, "parameters": M.parameter_count(model), "abs_delta_vs_lcrc": abs(M.parameter_count(model) - C.LCRC_PARAMS_REFERENCE)})
    chosen = min(table, key=lambda row: (row["abs_delta_vs_lcrc"], row["hidden"]))
    model, info = M.build_model("C", spec, norm, 0, "cpu")
    return {
        "hidden_search": table,
        "chosen_hidden": chosen["hidden"],
        "chosen_parameters": chosen["parameters"],
        "rule": "layers=3 fixed; even hidden minimising |N_gine - 247106|; no performance consulted",
        "info": info,
        "soup_safety": M.soup_safety_report(model),
    }


def cmd_parameters(_args) -> int:
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "lcrc": _lcrc_parameter_payload(),
        "gine": _gine_parameter_payload(),
        "baseline_ecfp4_xgb": {
            "fingerprint_bits": 2048,
            "fingerprint_radius": 2,
            "xgb_estimators": 500,
            "xgb_max_depth": 6,
            "trainable_parameters": None,
            "note": "tree ensemble; no gradient parameter count comparable to the neural arms",
        },
    }
    C.write_json(C.ANALYSIS / "parameters.json", payload)
    C.log(
        f"[parameters] A/B={payload['lcrc']['parameters']} B_eff={payload['lcrc']['arm_B']['effective_parameters']} "
        f"C={payload['gine']['chosen_parameters']} (hidden {payload['gine']['chosen_hidden']})"
    )
    return 0


def cmd_budget(_args) -> int:
    cfg = C.load_config()
    anchor_path = (
        C.REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/train_P_s0.json"
    )
    anchor = C.read_json(anchor_path)
    anchor_seconds = float(anchor["wall_seconds"])
    anchor_updates = int(anchor["updates"])
    sec_per_update = anchor_seconds / anchor_updates

    # CPU step timing as an independent sanity check (not used for the GPU math).
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    cache = D.load_caches(spec)["fit"]["cache"]
    batch = D.collate(cache, list(range(C.BATCH_SIZE)))
    cpu_steps = {}
    for arm in C.NN_ARMS:
        model, _info = M.build_model(arm, spec, norm, 0, "cpu")
        opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        t0 = time.perf_counter()
        for _ in range(3):
            y = model(batch)
            loss = (y ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        cpu_steps[arm] = (time.perf_counter() - t0) / 3.0

    per_run = {
        "A": {"updates": C.TOTAL_UPDATES, "factor_vs_anchor": 1.0},
        "B": {"updates": C.TOTAL_UPDATES, "factor_vs_anchor": 1.0},
        "C": {"updates": C.TOTAL_UPDATES, "factor_vs_anchor": 2.0},
    }
    for arm, row in per_run.items():
        row["estimate_seconds"] = row["updates"] * sec_per_update * row["factor_vs_anchor"]
        row["estimate_gpu_hours"] = row["estimate_seconds"] / 3600.0
        row["runs"] = len(C.SEEDS)
    nn_hours = sum(row["estimate_seconds"] * row["runs"] for row in per_run.values()) / 3600.0
    baseline_hours_estimate = 0.01  # local CPU; recorded for completeness, no GPU
    overhead = 0.20
    smoke_hours = 0.05
    prediction_hours = 0.02
    total = nn_hours * (1 + overhead) + smoke_hours + prediction_hours
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "ESTIMATE_ONLY_NOT_EXECUTED",
        "anchor": {
            "path": C.track_rel(anchor_path),
            "model": "BJOC-Lipo-SSL (P) seed 0",
            "device": anchor.get("device"),
            "gpu": anchor.get("cuda_device_name"),
            "wall_seconds": anchor_seconds,
            "updates": anchor_updates,
            "seconds_per_update": sec_per_update,
            "n_train_rows": anchor.get("n_train_rows"),
            "parameters": anchor.get("parameter_audit", {}).get("actual_parameters"),
            "why_comparable": "same task, same 128-batch/360-epoch/per-step-cosine recipe family, A100",
        },
        "cpu_step_seconds_128": cpu_steps,
        "per_run_estimate": per_run,
        "runs": {
            "nn": f"{len(C.NN_ARMS)} arms x {len(C.SEEDS)} seeds = {len(C.NN_ARMS) * len(C.SEEDS)}",
            "baseline": f"{len(C.BASELINE_ARMS)} x {len(C.BASELINE_REPLICATES)} XGBoost fits (local CPU, no GPU)",
            "nn_gpu_hours": nn_hours,
            "baseline_gpu_hours": 0.0,
            "baseline_cpu_hours_estimate": baseline_hours_estimate,
        },
        "overhead_fraction": overhead,
        "smoke_gpu_hours_reserved": smoke_hours,
        "prediction_gpu_hours_reserved": prediction_hours,
        "estimated_total_gpu_hours": total,
        "hard_cap_gpu_hours": float(cfg["budget"]["hard_cap_gpu_hours"]),
        "suggested_external_cap_gpu_hours": float(cfg["budget"]["suggested_external_cap_gpu_hours"]),
        "per_run_timeout_seconds": int(cfg["budget"]["per_run_timeout_seconds"]),
        "fits_within_hard_cap": bool(total <= float(cfg["budget"]["hard_cap_gpu_hours"])),
        "gpu_hours_used_so_far": 0.0,
        "gpu_authorized": False,
        "assumptions": [
            "Per-update cost is anchored on one measured A100 BJOC run; no GPU timing was collected in this round.",
            "Arm C is assumed to cost 2x arm A per update (3 GINE layers at hidden 170).",
            "20% overhead covers restart/retry and evaluation I/O.",
        ],
    }
    _require_budget(payload)
    C.write_json(C.ANALYSIS / "budget_estimate.json", payload)
    C.log(f"[budget] estimated total {total:.3f} GPU-hours; hard cap {payload['hard_cap_gpu_hours']}")
    return 0


def _require_budget(payload: dict[str, Any]) -> None:
    if not payload["fits_within_hard_cap"]:
        raise SystemExit("budget estimate exceeds the frozen hard cap; do not request GPU authorization")


def cmd_freeze_protocol(_args) -> int:
    cfg = C.load_config()
    stats = D.save_target_norm()
    access_guard.disarm()
    cmd_parameters(None)
    cmd_budget(None)
    freeze = C.build_protocol_freeze(cfg)
    freeze["fit_target_statistics"] = {
        "source": stats["source"],
        "n": stats["n"],
        "mean": stats["mean"],
        "scale": stats["scale"],
        "fit_label_vector_sha256": stats["fit_label_vector_sha256"],
        "monitor_or_reserved_used": False,
    }
    freeze["prediction_registry"] = P._expected_keys()
    C.write_json(C.PROTOCOL_FREEZE, freeze)
    C.log(
        f"[freeze-protocol] config_hash={freeze['config_hash'][:12]} "
        f"freeze_sha={C.sha256_file(C.PROTOCOL_FREEZE)[:12]} fit_mean={stats['mean']:.6f} fit_std={stats['scale']:.6f}"
    )
    return 0


# ---------------------------------------------------------------------------
# training / prediction / evaluation
# ---------------------------------------------------------------------------


def cmd_smoke(args) -> int:
    cfg = T.TrainConfig(
        arm=args.arm,
        seed=int(args.seed),
        device="cpu",
        smoke=True,
        fit_limit=int(args.fit_limit),
        epochs=int(args.epochs),
    )
    payload = T.train_one(cfg)
    C.log(
        f"[smoke] {payload['run_id']} mse {payload['curve'][0]['train_mse_normalised']:.5f} -> "
        f"{payload['curve'][-1]['train_mse_normalised']:.5f} (monitor={payload['monitor_labels_read']})"
    )
    return 0


def cmd_train(args) -> int:
    if args.device != "cuda":
        raise SystemExit("formal training is CUDA-only; use `smoke` for CPU engineering checks")
    request = f"{args.arm}_s{args.seed}"
    auth = require_gpu_authorization(request)
    frozen = C.verify_protocol_freeze(require_clean=True)
    if frozen["gpu_authorized"] is True:
        raise PermissionError("the protocol freeze already records an authorized GPU phase; use a new round")
    out = C.RUNS / request / "train.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite an existing run: {C.track_rel(out)}")
    cfg = T.TrainConfig(arm=args.arm, seed=int(args.seed), device="cuda")
    payload = T.train_one(cfg)
    C.log(
        f"[train] {request} soup fit_rmse={payload['fit_diagnostics']['soup_fit_rmse_raw']:.5f} "
        f"wall={payload['wall_seconds']:.1f}s authorized_hours={auth['max_gpu_hours']}"
    )
    return 0


def cmd_predict_nn(args) -> int:
    if args.device == "cuda":
        require_gpu_authorization(f"{args.arm}_s{args.seed}")
    meta = P.predict_nn(args.arm, int(args.seed), args.device)
    C.log(f"[predict-nn] {meta['key']} sha={meta['sha256'][:12]}")
    return 0


def cmd_baseline(_args) -> int:
    out = BASE.fit_and_predict()
    C.log(f"[baseline-ecfp] wrote {len(out['fits'])} prediction files")
    return 0


def cmd_freeze_predictions(_args) -> int:
    manifest = P.freeze_predictions()
    C.log(f"[freeze-predictions] manifest_sha={manifest['manifest_sha256'][:12]} n={len(manifest['predictions'])}")
    return 0


def cmd_evaluate(_args) -> int:
    evaluation = evaluate_module().evaluate()
    C.log(f"[evaluate] decision={evaluation['decision']['case']}")
    return 0


def evaluate_module():
    import evaluate as module

    return module


def cmd_cpu_gates(_args) -> int:
    payload = screen_gates.run_all()
    C.log(f"[cpu-gates] final_status={payload['final_status']}")
    return 0 if payload["final_status"] == "READY_FOR_GPU_AUTHORIZATION" else 2


def cmd_status(_args) -> int:
    cfg = C.load_config()
    gates = C.read_json(C.ANALYSIS / "cpu_gates.json") if (C.ANALYSIS / "cpu_gates.json").exists() else None
    runs = sorted(p.parent.name for p in C.RUNS.glob("*/train.json")) if C.RUNS.exists() else []
    preds = sorted(p.name for p in C.PREDICTIONS.glob("*.npz")) if C.PREDICTIONS.exists() else []
    status = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.source_dirty(),
        "git_dirty_worktree": C.git_dirty(),
        "protocol_freeze": C.file_record(C.PROTOCOL_FREEZE),
        "cpu_gates_final_status": gates["final_status"] if gates else None,
        "gpu_authorization_file": C.file_record(GPU_AUTHORIZATION),
        "gpu_authorized": False,
        "training_runs": runs,
        "prediction_files": preds,
        "prediction_manifest": C.file_record(C.PREDICTION_MANIFEST),
        "monitor_evaluated": (C.ANALYSIS / "monitor_metrics.json").exists(),
        "gpu_hours_used": 0.0,
        "hard_cap_gpu_hours": cfg["budget"]["hard_cap_gpu_hours"],
        "state": (
            "READY_FOR_GPU_AUTHORIZATION"
            if gates and gates["final_status"] == "READY_FOR_GPU_AUTHORIZATION" and not GPU_AUTHORIZATION.exists()
            else "NOT_READY"
        ),
    }
    C.log(json.dumps(status, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-data", help="verify frozen artifacts and caches").set_defaults(func=cmd_verify_data)
    sub.add_parser("parameters", help="write analysis/parameters.json").set_defaults(func=cmd_parameters)
    sub.add_parser("budget", help="write analysis/budget_estimate.json").set_defaults(func=cmd_budget)
    sub.add_parser("freeze-protocol", help="write the frozen protocol record").set_defaults(func=cmd_freeze_protocol)
    sub.add_parser("cpu-gates", help="run the CPU gates").set_defaults(func=cmd_cpu_gates)
    sub.add_parser("baseline-ecfp", help="fit ECFP4+XGBoost on Fit and predict Monitor").set_defaults(func=cmd_baseline)
    sub.add_parser("freeze-predictions", help="hash and lock all predictions").set_defaults(func=cmd_freeze_predictions)
    sub.add_parser("evaluate", help="evaluate frozen predictions on Monitor").set_defaults(func=cmd_evaluate)
    sub.add_parser("status", help="print the round status").set_defaults(func=cmd_status)

    smoke = sub.add_parser("smoke", help="CPU-only short training smoke")
    smoke.add_argument("--arm", choices=list(C.NN_ARMS), required=True)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--fit-limit", type=int, default=256)
    smoke.add_argument("--epochs", type=int, default=12)
    smoke.set_defaults(func=cmd_smoke)

    train = sub.add_parser("train", help="formal training run (CUDA only, authorization required)")
    train.add_argument("--arm", choices=list(C.NN_ARMS), required=True)
    train.add_argument("--seed", type=int, choices=list(C.SEEDS), required=True)
    train.add_argument("--device", choices=["cuda"], required=True)
    train.set_defaults(func=cmd_train)

    pred = sub.add_parser("predict-nn", help="label-blind prediction for one neural arm/seed")
    pred.add_argument("--arm", choices=list(C.NN_ARMS), required=True)
    pred.add_argument("--seed", type=int, choices=list(C.SEEDS), required=True)
    pred.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    pred.set_defaults(func=cmd_predict_nn)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
