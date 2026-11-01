"""Command-line driver for ``lcrc_esol_physchem_s2_v1``.

Commands
--------
preflight    frozen-input hashes + BASE checkpoint replay (no training)
p0           CPU gates: alignment, single standardisation, init, E capacity/paths
freeze       freeze code/config/inputs/P0/BASE-replay before any new training
smoke        short local CPU smoke of the two new arms (never a formal run)
train        one formal arm (CUDA, one-shot)
analyse      Validation-only paired comparison + group bootstrap + usage probe
manifest     verify the six new checkpoints and freeze their hashes
ledger       GPU-hour accounting for the new runs only
status       print round status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402

import esol2_common as C  # noqa: E402
import esol2_data as D  # noqa: E402
import esol2_models as M  # noqa: E402
import esol2_train as T  # noqa: E402


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def run_base_replay(out_path: Path | str | None = None) -> dict[str, Any]:
    views = D.build_views()
    spec = views["spec"]
    pins = C.v1_base_checkpoint_pins()
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    y_train = np.asarray(views["labels"]["y_train"], dtype=np.float64)
    norm = views["target_norm"]
    per_seed: dict[str, Any] = {}
    for seed in C.SEEDS:
        base, _meta = M.build_base(spec, views["train_norm"], seed, "cpu", 0.10)
        rec = pins[seed]
        M.load_checkpoint(
            base,
            rec["checkpoint"],
            expected_sha256=rec["checkpoint_sha256"],
            expected_family="base",
            expected_seed=seed,
        )
        pred_valid = T.denormalise(M.predict(base, views["valid_cache"], "cpu"), norm)
        pred_train = T.denormalise(M.predict(base, views["train_cache"], "cpu"), norm)
        valid_metrics = C.all_metrics(pred_valid, y_valid)
        train_metrics = C.all_metrics(pred_train, y_train)
        npz_path = C.v1_run_dir(f"base_s{seed}") / "valid_predictions.npz"
        if npz_path.exists():
            saved = np.asarray(np.load(npz_path)["pred"], dtype=np.float64)
            npz_diff = float(np.max(np.abs(pred_valid - saved)))
        else:
            # the *.npz prediction files are git-ignored and therefore absent on
            # the GPU host; the checkpoint replay is the authoritative check
            npz_diff = None
        per_seed[str(seed)] = {
            "arm": f"base_s{seed}",
            "checkpoint_sha256": rec["checkpoint_sha256"],
            "replayed_valid_rmse": float(valid_metrics["rmse"]),
            "recorded_valid_rmse": float(C.V1_BASE_VALID_RMSE[seed]),
            "manifest_valid_rmse": float(rec["valid_rmse"]),
            "abs_diff": float(abs(valid_metrics["rmse"] - C.V1_BASE_VALID_RMSE[seed])),
            "replayed_valid_mae": float(valid_metrics["mae"]),
            "replayed_train_rmse": float(train_metrics["rmse"]),
            "recorded_train_rmse": float(C.V1_BASE_TRAIN_RMSE[seed]),
            "train_abs_diff": float(abs(train_metrics["rmse"] - C.V1_BASE_TRAIN_RMSE[seed])),
            "max_abs_pred_diff_vs_v1_saved_predictions": npz_diff,
            "v1_saved_predictions_available": npz_diff is not None,
            "best_epoch_recorded": int(C.V1_BASE_BEST_EPOCH[seed]),        }
    checks = {
        "readonly_inputs": C.verify_readonly_inputs()["all_matched"],
        "valid_rmse_reproduced": all(rec["abs_diff"] <= C.BASE_REPLAY_TOL for rec in per_seed.values()),
        "train_rmse_reproduced": all(rec["train_abs_diff"] <= C.BASE_REPLAY_TOL for rec in per_seed.values()),
        # v1 stored its predictions from a CUDA forward; CPU float32 accumulation
        # order differs at ~1e-6, so this bound is loose by construction and is
        # reported rather than asserted at 1e-9.
        "predictions_reproduced_within_cpu_float32": all(
            rec["max_abs_pred_diff_vs_v1_saved_predictions"] is None
            or rec["max_abs_pred_diff_vs_v1_saved_predictions"] <= 1.0e-4
            for rec in per_seed.values()
        ),
    }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "purpose": "reload the three frozen BASE checkpoints and reproduce their Validation RMSE",
        "tolerance": C.BASE_REPLAY_TOL,
        "base_retrained": False,
        "per_seed": per_seed,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "environment": C.env_info(),
    }
    target = Path(out_path) if out_path is not None else C.BASE_REPLAY
    C.write_json(target, payload)
    payload["written_to"] = C.track_rel(target)
    return payload


def cmd_preflight(args: argparse.Namespace) -> dict[str, Any]:
    payload = run_base_replay(getattr(args, "out", None))
    C.log(
        f"[preflight] passed={payload['passed']} written_to={payload['written_to']} "
        + " ".join(f"s{k}={v['replayed_valid_rmse']:.6f}" for k, v in payload["per_seed"].items())
    )
    if not payload["passed"]:
        raise SystemExit(f"BASE replay failed; see {payload['written_to']}")
    return payload


# ---------------------------------------------------------------------------
# gates / freeze / smoke / train
# ---------------------------------------------------------------------------


def cmd_p0(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import esol2_p0

    replay = C.read_json(C.BASE_REPLAY) if C.BASE_REPLAY.exists() else None
    payload = esol2_p0.run(base_replay=replay)
    C.log(f"[p0] status={payload['status']} checks={payload['n_checks']} failed={payload['failed_checks']}")
    if payload["status"] != "PASS":
        raise SystemExit("P0 gates failed; see analysis/p0_gates.json")
    return payload


def cmd_freeze(args: argparse.Namespace) -> dict[str, Any]:
    del args
    p0 = C.read_json(C.P0_GATES)
    replay = C.read_json(C.BASE_REPLAY)
    payload = C.freeze_protocol(p0=p0, replay=replay)
    C.log(
        f"[freeze] sha={payload['freeze_sha256'][:12]} p0={payload['p0_status']} "
        f"config={payload['config_sha256'][:12]} base_replay={payload['base_replay']['passed']}"
    )
    return payload


def cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import torch

    views = D.build_views()
    cfg = C.load_config()
    C.SMOKE.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for model in C.CANDIDATES:
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
            "parameters_effective": payload["model_audit"]["parameters_effective"],
            "recheck_abs_diff": payload["checkpoint"]["valid_rmse_recheck_abs_diff"],
            "gradient_probe_passed": payload["diagnostics"]["gradient_probe"]["passed"],
            "information_path": payload["diagnostics"]["information_path"],
        }
    result = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "COMPLETED",
        "smoke": True,
        "note": "local CPU smoke only; never counted as a formal run",
        "results": out,
        "environment": C.env_info(),
    }
    C.write_json(C.SMOKE / "smoke.json", result)
    C.log(f"[smoke] {json.dumps({k: v['valid_rmse'] for k, v in out.items()}, sort_keys=True)}")
    return result


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    arm = str(args.arm)
    model_kind, seed = C.arm_parts(arm)
    if model_kind == "base":
        raise SystemExit("BASE is reused read-only; it is not retrained in this round")
    C.require_freeze()
    device = torch.device(str(args.device))
    if device.type != "cuda":
        raise PermissionError("formal training must run on CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    run_path = C.run_dir(arm)
    if (run_path / "train.json").exists():
        raise PermissionError(f"{arm} already trained (one-shot)")
    cfg = C.load_config()
    views = D.build_views()
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


# ---------------------------------------------------------------------------
# analysis / manifest / ledger
# ---------------------------------------------------------------------------


def cmd_analyse(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import esol2_analysis

    summary = esol2_analysis.analyse()
    for model, rec in summary["candidates"].items():
        C.log(
            f"[analyse] {rec['model_display']}: delta_valid_rmse mean={rec['delta_valid_rmse_mean']:+.4f} "
            f"sd={rec['delta_valid_rmse_sd']:.4f} wins={rec['n_seeds_candidate_better']}/3 "
            f"base={rec['base_valid_rmse_mean']:.4f} cand={rec['candidate_valid_rmse_mean']:.4f}"
        )
    return summary


def verify_new_arm(arm: str, views) -> dict[str, Any]:
    model_kind, seed = C.arm_parts(arm)
    run_path = C.run_dir(arm)
    train_json = C.read_json(run_path / "train.json")
    if train_json.get("smoke") is not False:
        raise PermissionError(f"{arm}: manifest refuses smoke runs")
    if train_json.get("test_labels_read") is not False:
        raise PermissionError(f"{arm}: training record claims Test-label access")
    ckpt_path = run_path / "checkpoint.pt"
    ckpt_sha = C.sha256_file(ckpt_path)
    if ckpt_sha != train_json["checkpoint"]["sha256"]:
        raise PermissionError(f"{arm}: checkpoint hash differs from train.json")
    physchem_train, physchem_valid = D.physchem_for(model_kind, views)
    model, meta = M.build(
        model_kind,
        views["spec"],
        views["train_norm"],
        seed,
        "cpu",
        dropout=float(C.MODEL_DROPOUT[model_kind]),
        physchem_norm=views.get("physchem_norm"),
    )
    checkpoint = M.load_checkpoint(
        model, ckpt_path, expected_sha256=ckpt_sha, expected_family=model_kind, expected_seed=seed
    )
    valid_pred = T.denormalise(M.predict(model, views["valid_cache"], "cpu", physchem=physchem_valid), views["target_norm"])
    metrics = C.all_metrics(valid_pred, np.asarray(views["labels"]["y_valid"], dtype=np.float64))
    recheck = abs(float(metrics["rmse"]) - float(checkpoint["best_valid_rmse"]))
    if recheck > 1.0e-6:
        raise RuntimeError(f"{arm}: reloaded Validation RMSE does not reproduce ({recheck})")
    import csv as _csv

    with open(run_path / "curve.csv", newline="", encoding="utf-8") as handle:
        curve = list(_csv.DictReader(handle))
    curve_valid = np.asarray([float(row["valid_rmse_raw"]) for row in curve], dtype=np.float64)
    argmin_epoch = int(np.argmin(curve_valid)) + 1
    if argmin_epoch != int(checkpoint["best_epoch"]):
        raise RuntimeError(f"{arm}: curve argmin {argmin_epoch} != best epoch {checkpoint['best_epoch']}")
    state_hash = C.state_dict_fingerprint(checkpoint["state_dict"])
    if state_hash != checkpoint["best_state_hash"]:
        raise RuntimeError(f"{arm}: checkpoint state hash mismatch")
    return {
        "arm": arm,
        "model": C.MODEL_DISPLAY[model_kind],
        "family": model_kind,
        "seed": int(seed),
        "dropout": float(C.MODEL_DROPOUT[model_kind]),
        "checkpoint": C.track_rel(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "curve_sha256": C.sha256_file(run_path / "curve.csv"),
        "train_json_sha256": C.sha256_file(run_path / "train.json"),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_inner_valid_rmse": float(checkpoint["best_valid_rmse"]),
        "reloaded_valid_rmse": float(metrics["rmse"]),
        "recheck_abs_diff": float(recheck),
        "curve_argmin_epoch": argmin_epoch,
        "parameters": int(train_json["model_audit"]["parameters"]),
        "parameters_effective": int(train_json["model_audit"]["parameters_effective"]),
        "state_hash": state_hash,
        "test_labels_read": False,
    }


#: Audit / reporting / driver modules: hashed, but they cannot change a trained
#: checkpoint.  The pre-training freeze pinned the full dictionary; a later edit
#: to one of these files is reported as analysis-only drift, never as a change
#: of the trained artefacts.
ANALYSIS_ONLY_MODULES = ("esol2_analysis.py", "esol2_p0.py", "run_esol2.py")


def training_code_hashes() -> dict[str, str]:
    drop = {f"esol2/{name}" for name in ANALYSIS_ONLY_MODULES}
    return {key: value for key, value in C.code_hashes().items() if key not in drop}


def verify_freeze_for_post_run() -> dict[str, Any]:
    """Post-run check of the pre-training freeze (training subset must match)."""
    if not C.PROTOCOL_FREEZE.exists():
        raise PermissionError("protocol freeze missing")
    frozen = C.read_json(C.PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("protocol freeze has the wrong protocol id")
    if frozen.get("config_sha256") != C.config_sha256():
        raise PermissionError("config changed after the freeze")
    if frozen.get("p0_status") != "PASS" or frozen.get("base_replay", {}).get("passed") is not True:
        raise PermissionError("freeze does not record P0 PASS and a passed BASE replay")
    if frozen.get("split_manifest_sha256") != C.sha256_file(C.SPLIT_MANIFEST):
        raise PermissionError("split manifest changed after the freeze")
    frozen_training = {key: value for key, value in frozen["code_hashes"].items() if key in training_code_hashes()}
    if frozen_training != training_code_hashes():
        raise PermissionError("training code changed after the freeze")
    return frozen


def cmd_manifest(args: argparse.Namespace) -> dict[str, Any]:
    del args
    views = D.build_views()
    arms = C.arms_with_runs()
    lanes = {arm: verify_new_arm(arm, views) for arm in arms}
    # Post-run provenance: the freeze was written before training; an
    # analysis-only module may have been edited afterwards.  Record the exact
    # difference and verify that every run's *training* code equals the freeze.
    frozen = verify_freeze_for_post_run()
    current_full = C.code_hashes()
    drift = {
        key: {"frozen": frozen["code_hashes"].get(key), "current": value}
        for key, value in current_full.items()
        if frozen["code_hashes"].get(key) != value
    }
    current_training = training_code_hashes()
    training_matches_freeze = {
        arm: {
            key: value
            for key, value in C.read_json(C.run_dir(arm) / "train.json")["code_hashes"].items()
            if key in current_training
        }
        == current_training
        for arm in arms
    }
    if not all(training_matches_freeze.values()):
        raise PermissionError(f"a run was produced by different training code: {training_matches_freeze}")
    post_run = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "freeze_sha256": frozen["freeze_sha256"],
        "full_code_hash_drift_since_freeze": drift,
        "drift_is_analysis_only": bool(
            all(key.split("/")[-1] in ANALYSIS_ONLY_MODULES for key in drift)
        ),
        "training_code_hashes_match_freeze": True,
        "per_run_training_code_matches_current": training_matches_freeze,
        "note": (
            "the drift entries are audit/reporting modules edited after the six runs; "
            "the training modules, the config and every reused frozen module are "
            "byte-identical to the pre-training freeze and to each run record"
        ),
    }
    C.write_json(C.ANALYSIS / "post_run_provenance.json", post_run)
    coverage = {
        "new_formal_runs": len(arms),
        "hard_max": 6,
        "within_cap": len(arms) <= 6,
        "base_retrained": any(arm.startswith("base_") for arm in arms),
        "test_evaluated": any((C.ANALYSIS / "eval").glob("*")) if (C.ANALYSIS / "eval").exists() else False,
        "test_rows_used": False,
    }
    if coverage["base_retrained"] or not coverage["within_cap"] or coverage["test_evaluated"]:
        raise PermissionError(f"coverage violation: {coverage}")
    body = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "status": "NEW_CHECKPOINTS_FROZEN_NO_TEST_EVALUATION",
        "scope": "the six new ESOL runs only; BASE checkpoints stay with the v1 manifest",
        "config_sha256": C.config_sha256(),
        "code_hashes": C.code_hashes(),
        "split_manifest_sha256": C.sha256_file(C.SPLIT_MANIFEST),
        "n_arms": len(lanes),
        "arms": lanes,
        "coverage": coverage,
        "test_policy": "no Test evaluation in this round (old Test split already read and analysed)",
        "test_labels_read_at_freeze_time": False,
    }
    body["manifest_sha256"] = C.sha256_json(body)
    C.write_json(C.MANIFEST, body)
    C.write_json(
        C.COVERAGE,
        {
            "manifest_sha256": body["manifest_sha256"],
            "created_utc": body["created_utc"],
            "coverage": coverage,
        },
    )
    C.log(f"[manifest] frozen {len(lanes)} new checkpoints sha={body['manifest_sha256'][:12]}")
    return body


def cmd_ledger(args: argparse.Namespace) -> dict[str, Any]:
    del args
    runs: dict[str, Any] = {}
    total = 0.0
    for arm in C.arms_with_runs():
        payload = C.read_json(C.run_dir(arm) / "train.json")
        seconds = float(payload["timing"]["process_wall_seconds"])
        total += seconds
        runs[arm] = {
            "status": "COMPLETED",
            "device": payload.get("device"),
            "cuda_device_name": payload.get("cuda_device_name"),
            "process_wall_seconds": seconds,
            "train_wall_seconds": float(payload["timing"]["train_wall_seconds"]),
            "gpu_count": 1 if str(payload.get("device", "")).startswith("cuda") else 0,
            "best_epoch": int(payload["checkpoint"]["best_epoch"]),
            "valid_rmse": float(payload["metrics"]["valid"]["rmse"]),
            "hostname": payload.get("hostname"),
            "git_commit": payload.get("git_commit"),
        }
    hours = total / 3600.0
    ledger = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "charging_rule": "one A100 per run; training-process wall clock from train.json",
        "base_runs_charged": 0,
        "formal_runs_completed": len(runs),
        "total_process_seconds": total,
        "total_gpu_hours": hours,
        "within_cap": bool(hours <= C.GPU_HARD_CAP_HOURS),
        "runs": runs,
    }
    C.write_json(C.GPU_LEDGER, ledger)
    C.log(f"[ledger] runs={len(runs)} gpu_hours={hours:.4f} / {C.GPU_HARD_CAP_HOURS}")
    return ledger


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    del args
    status = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "artifacts": {
            "p0_gates": C.P0_GATES.exists(),
            "base_replay": C.BASE_REPLAY.exists(),
            "protocol_freeze": C.PROTOCOL_FREEZE.exists(),
            "valid_comparison": C.VALID_COMPARISON.exists(),
            "paired_valid": C.PAIRED_VALID.exists(),
            "paired_summary": C.PAIRED_SUMMARY.exists(),
            "manifest": C.MANIFEST.exists(),
            "gpu_ledger": C.GPU_LEDGER.exists(),
        },
        "new_runs": C.arms_with_runs(),
        "n_new_runs": len(C.arms_with_runs()),
    }
    if C.P0_GATES.exists():
        status["p0_status"] = C.read_json(C.P0_GATES)["status"]
    if C.BASE_REPLAY.exists():
        status["base_replay_passed"] = C.read_json(C.BASE_REPLAY)["passed"]
    C.log(json.dumps(status, indent=2, sort_keys=True))
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("p0", "freeze", "smoke", "analyse", "manifest", "ledger", "status"):
        sub.add_parser(name)
    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--out", default=None, help="output path (default analysis/base_replay.json)")
    p_train = sub.add_parser("train")
    p_train.add_argument("--arm", required=True)
    p_train.add_argument("--device", default="cuda:0", choices=["cuda", "cuda:0"])
    p_train.add_argument("--pool", required=True)
    return parser


HANDLERS = {
    "preflight": cmd_preflight,
    "p0": cmd_p0,
    "freeze": cmd_freeze,
    "smoke": cmd_smoke,
    "train": cmd_train,
    "analyse": cmd_analyse,
    "manifest": cmd_manifest,
    "ledger": cmd_ledger,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    HANDLERS[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
