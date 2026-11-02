"""Freeze one stage's trained checkpoints and re-verify Validation on CPU.

Run this *after* every pre-registered run of a stage has finished and *before*
any Test target is decoded.  It records the checkpoint bytes, best epochs,
Validation metrics and row indices, and refuses to continue if the stored curve,
the argmin epoch, the checkpoint hash or the CPU re-inference disagree with the
recorded values.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import dissect_common as C
import dissect_data as D
import dissect_model as M


def _valid_rmse(model, cache: Mapping[str, Any], y: np.ndarray, target_norm: Mapping[str, Any]) -> dict[str, float]:
    pred_std = D.predict_scores(model, cache, torch.device("cpu"))
    pred = pred_std * float(target_norm["scale"]) + float(target_norm["mean"])
    err = pred - y
    ss_res = float((err * err).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "rmse": float(math.sqrt(ss_res / y.size)),
        "mae": float(np.abs(err).mean()),
        "r2": float(1.0 - ss_res / ss_tot),
        "bias": float(err.mean()),
    }


def freeze(
    *,
    repeat: int,
    variants: Sequence[str],
    seeds: Sequence[int],
    tag: str,
    stage: str,
) -> dict[str, Any]:
    view = D.build_view(int(repeat))
    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    for variant in variants:
        for seed in seeds:
            run_dir = C.run_dir(repeat, variant, seed)
            name = C.run_name(repeat, variant, seed)
            train_json = run_dir / "train.json"
            ckpt_path = run_dir / "checkpoint.pt"
            curve_path = run_dir / "curve.csv"
            if not (train_json.exists() and ckpt_path.exists() and curve_path.exists()):
                problems.append(f"{name}: missing train.json/checkpoint.pt/curve.csv")
                continue
            rec = C.read_json(train_json)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            local_problems: list[str] = []

            if rec["variant"] != variant or int(rec["seed"]) != int(seed) or int(rec["repeat"]) != int(repeat):
                local_problems.append("train.json identity mismatch")
            if rec["smoke"]:
                local_problems.append("smoke run cannot be frozen as a result")
            if int(rec["recipe"]["epochs_run"]) != int(C.RECIPE["epochs"]):
                local_problems.append(f"epochs_run {rec['recipe']['epochs_run']} != {C.RECIPE['epochs']}")
            if int(rec["recipe"]["optimizer_steps"]) != 27 * int(C.RECIPE["epochs"]):
                local_problems.append("optimizer_steps != 9720")
            if rec["model"]["parameters"] != C.EXPECTED_PARAMS[variant]:
                local_problems.append("trainable parameters != frozen expectation")
            if rec["data"]["test_labels_read"] or rec["data"]["test_cache_built"]:
                local_problems.append("training record shows Test access")
            if rec["recipe"]["test_used_for_any_selection"]:
                local_problems.append("training record shows Test-based selection")
            if not rec["model"]["shared_init_bitwise_equal_to_full"]:
                local_problems.append("shared init not bitwise equal to FULL")
            if rec["checkpoint"]["sha256"] != C.sha256_file(ckpt_path):
                local_problems.append("checkpoint sha256 mismatch")
            if rec["checkpoint"]["curve_sha256"] != C.sha256_file(curve_path):
                local_problems.append("curve sha256 mismatch")
            if ckpt["best_epoch"] != rec["checkpoint"]["best_epoch"]:
                local_problems.append("checkpoint best_epoch != train.json")
            if ckpt["best_state_hash"] != rec["checkpoint"]["best_state_hash"]:
                local_problems.append("checkpoint best_state_hash != train.json")
            if int(ckpt["parameters"]) != C.EXPECTED_PARAMS[variant]:
                local_problems.append("checkpoint parameters != frozen expectation")
            if [int(x) for x in rec["variant_local_cols"]] != list(C.variant_cols(variant)[0]):
                local_problems.append("local column mapping drifted from the frozen definition")
            if [int(x) for x in rec["variant_rho_cols"]] != list(C.variant_cols(variant)[1]):
                local_problems.append("rho column mapping drifted from the frozen definition")

            # curve argmin must equal the tracked best epoch (ties keep earlier)
            import csv as _csv

            with open(curve_path, newline="", encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))
            values = np.asarray([float(r["valid_rmse_raw"]) for r in rows], dtype=np.float64)
            argmin = int(np.argmin(values) + 1)
            if argmin != int(ckpt["best_epoch"]):
                local_problems.append(f"curve argmin {argmin} != best epoch {ckpt['best_epoch']}")
            if len(rows) != int(C.RECIPE["epochs"]):
                local_problems.append("curve length != 360")

            # reload + CPU re-inference
            model = M.load_checkpoint_model(
                view.spec, view.train_norm, variant, ckpt["state_dict"], torch.device("cpu")
            )
            metrics = _valid_rmse(model, view.valid_cache, view.y_valid, ckpt["target_norm"])
            reload_diff = abs(metrics["rmse"] - float(ckpt["best_metric_value"]))
            if reload_diff > C.REPRO_TOL:
                local_problems.append(
                    f"CPU reload Valid RMSE {metrics['rmse']:.12f} != recorded {ckpt['best_metric_value']:.12f} "
                    f"(diff {reload_diff:.3e})"
                )
            # The stored target normaliser must be the Train-only one from this repeat.
            if abs(float(ckpt["target_norm"]["mean"]) - float(view.target_norm["mean"])) > 1e-12:
                local_problems.append("checkpoint target mean != repeat Train mean")
            if abs(float(ckpt["target_norm"]["scale"]) - float(view.target_norm["scale"])) > 1e-12:
                local_problems.append("checkpoint target scale != repeat Train std")

            entries.append(
                {
                    "run": name,
                    "repeat": int(repeat),
                    "variant": variant,
                    "variant_name": C.VARIANT_NAMES[variant],
                    "seed": int(seed),
                    "stage": stage,
                    "parameters": int(rec["model"]["parameters"]),
                    "expected_parameters": C.EXPECTED_PARAMS[variant],
                    "optimizer_steps": int(rec["recipe"]["optimizer_steps"]),
                    "epochs_run": int(rec["recipe"]["epochs_run"]),
                    "best_epoch": int(ckpt["best_epoch"]),
                    "curve_argmin_epoch": argmin,
                    "valid_rmse": float(metrics["rmse"]),
                    "valid_mae": float(metrics["mae"]),
                    "valid_r2": float(metrics["r2"]),
                    "valid_bias": float(metrics["bias"]),
                    "valid_rmse_recorded": float(ckpt["best_metric_value"]),
                    "valid_reload_abs_diff": float(reload_diff),
                    "train_rmse": float(rec["metrics"]["train"]["rmse"]),
                    "train_mae": float(rec["metrics"]["train"]["mae"]),
                    "train_r2": float(rec["metrics"]["train"]["r2"]),
                    "train_wall_seconds": float(rec["timing"]["train_wall_seconds"]),
                    "inference_seconds_per_valid": float(rec["timing"]["inference_seconds_per_valid_split"]),
                    "gpu_hours": float(rec["gpu"]["gpu_hours_charged"]),
                    "target_norm": dict(ckpt["target_norm"]),
                    "checkpoint_path": C.track_rel(ckpt_path),
                    "checkpoint_sha256": C.sha256_file(ckpt_path),
                    "best_state_hash": ckpt["best_state_hash"],
                    "init_state_hash": ckpt["init_state_hash"],
                    "full_init_state_hash": ckpt["full_init_state_hash"],
                    "curve_path": C.track_rel(curve_path),
                    "curve_sha256": C.sha256_file(curve_path),
                    "train_json_path": C.track_rel(train_json),
                    "train_json_sha256": C.sha256_file(train_json),
                    "valid_rows_sha256": C.row_sha256(view.rows["valid"]),
                    "train_rows_sha256": C.row_sha256(view.rows["train"]),
                    "split_file_sha256": view.source_meta["split_file_sha256"],
                    "train_norm_fingerprint": rec["data"]["train_norm_fingerprint"],
                    "batch_order_probe_epoch1_first10": rec["recipe"]["batch_order_probe_epoch1_first10"],
                    "problems": local_problems,
                }
            )
            problems.extend(f"{name}: {p}" for p in local_problems)
            C.log(f"[freeze {stage} r{repeat}] {name} best={ckpt['best_epoch']} valid={metrics['rmse']:.6f}")

    # cross-variant reproducibility of the batch order for this seed/repeat
    order_checks: dict[str, Any] = {}
    for seed in seeds:
        probes = {
            e["variant"]: e["batch_order_probe_epoch1_first10"] for e in entries if e["seed"] == int(seed)
        }
        if probes:
            values = list(probes.values())
            order_checks[str(seed)] = {
                "n_variants": len(values),
                "all_identical": all(v == values[0] for v in values),
                "first10": values[0],
            }

    summary = {
        "schema": "lcrc-lipo-dissection-freeze-v1",
        "tag": tag,
        "stage": stage,
        "repeat": int(repeat),
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "variants": list(variants),
        "seeds": [int(s) for s in seeds],
        "n_runs": len(entries),
        "entries": entries,
        "batch_order_checks": order_checks,
        "problems": problems,
        "ok": not problems,
        "test_scored": False,
        "note": "Frozen before any Test target of this stage is decoded.",
    }
    out = C.FREEZE / f"{tag}.json"
    C.write_json(out, summary)
    if problems:
        raise SystemExit(f"freeze failed with {len(problems)} problem(s); see {C.track_rel(out)}")
    C.log(f"wrote {C.track_rel(out)} ({len(entries)} runs)")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a stage's checkpoints (Validation only)")
    parser.add_argument("--repeat", type=int, default=C.MAIN_REPEAT, choices=list(C.REPEATS))
    parser.add_argument("--stage", required=True, choices=["A", "B", "C", "cross"])
    parser.add_argument("--variants", default=None, help="comma-separated; default from the stage")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    elif args.stage == "cross":
        raise SystemExit("--variants is required for the cross-split freeze")
    else:
        variants = list(C.STAGE_VARIANTS[args.stage])
    tag = args.tag or f"{args.stage}_r{args.repeat}"
    freeze(repeat=args.repeat, variants=variants, seeds=C.SEEDS, tag=tag, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
