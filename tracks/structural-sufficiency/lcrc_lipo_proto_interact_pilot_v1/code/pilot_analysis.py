"""Pilot analysis: five-model table, paired comparisons, slices and decision.

Only the five planned runs are read.  Every comparison is paired on
``global_row_id`` (Fit2352 Dev fold), uses raw logD units, and reports the fixed
slices requested by the task book: improved-molecule counts, removal of the
five worst BASE molecules, removal of the five largest improvements, and
scaffold-group descriptive error statistics.  The decision rule is frozen in
``configs/pilot.yaml`` and is applied mechanically.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pilot_common as C
import pilot_pairs as P


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def run_dir(arm: str) -> Path:
    return C.RUNS / f"{arm}_s{C.SEED}_fold{C.FOLD}"


def load_run(arm: str) -> dict[str, Any]:
    directory = run_dir(arm)
    payload = C.read_json(directory / "train.json")
    pred = np.load(directory / "predictions_dev.npz", allow_pickle=False)
    pred_train = np.load(directory / "predictions_train.npz", allow_pickle=False)
    diagnostics_path = directory / "diagnostics.json"
    diagnostics = C.read_json(diagnostics_path) if diagnostics_path.exists() else None
    return {
        "arm": arm,
        "dir": directory,
        "train": payload,
        "dev": {k: pred[k] for k in pred.files},
        "train_pred": {k: pred_train[k] for k in pred_train.files},
        "diagnostics": diagnostics,
    }


def require_runs(arms: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    missing = []
    for arm in arms:
        if not (run_dir(arm) / "train.json").exists():
            missing.append(arm)
            continue
        out[arm] = load_run(arm)
    if missing:
        raise FileNotFoundError(f"missing completed runs: {missing}")
    return out


# ---------------------------------------------------------------------------
# metrics / comparisons
# ---------------------------------------------------------------------------


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2)))


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))))


def _drop_and_compare(
    mask: np.ndarray, base_pred: np.ndarray, cand_pred: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    keep = ~mask
    b = rmse(base_pred[keep], y[keep])
    c = rmse(cand_pred[keep], y[keep])
    return {
        "n_dropped": int(mask.sum()),
        "n_remaining": int(keep.sum()),
        "base_rmse_remaining": b,
        "cand_rmse_remaining": c,
        "delta_rmse_remaining": c - b,
    }


def scaffold_group_stats(
    base_pred: np.ndarray, cand_pred: np.ndarray, y: np.ndarray, group_keys: np.ndarray, scaffolds: np.ndarray
) -> dict[str, Any]:
    index: dict[str, list[int]] = {}
    for i, key in enumerate(group_keys.tolist()):
        index.setdefault(str(key), []).append(i)
    rows = []
    for key, idx in index.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        rows.append(
            {
                "group_key_sha256": key,
                "scaffold": str(scaffolds[idx_arr[0]]),
                "n_molecules": int(idx_arr.size),
                "base_rmse": rmse(base_pred[idx_arr], y[idx_arr]),
                "cand_rmse": rmse(cand_pred[idx_arr], y[idx_arr]),
                "delta_rmse": rmse(cand_pred[idx_arr], y[idx_arr]) - rmse(base_pred[idx_arr], y[idx_arr]),
                "base_mae": mae(base_pred[idx_arr], y[idx_arr]),
                "cand_mae": mae(cand_pred[idx_arr], y[idx_arr]),
            }
        )
    rows.sort(key=lambda r: (-r["n_molecules"], r["group_key_sha256"]))
    buckets = [(1, 1), (2, 3), (4, 7), (8, 10 ** 9)]
    bucket_rows = []
    for lo, hi in buckets:
        idx = np.asarray([i for i, key in enumerate(group_keys.tolist()) if lo <= len(index[str(key)]) <= hi], dtype=np.int64)
        if idx.size == 0:
            continue
        bucket_rows.append(
            {
                "group_size_bucket": f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+",
                "n_groups": int(sum(1 for v in index.values() if lo <= len(v) <= hi)),
                "n_molecules": int(idx.size),
                "base_rmse": rmse(base_pred[idx], y[idx]),
                "cand_rmse": rmse(cand_pred[idx], y[idx]),
                "delta_rmse": rmse(cand_pred[idx], y[idx]) - rmse(base_pred[idx], y[idx]),
            }
        )
    return {
        "n_groups": len(rows),
        "groups": rows,
        "group_size_buckets": bucket_rows,
    }


def compare(base: Mapping[str, Any], cand: Mapping[str, Any]) -> dict[str, Any]:
    bd, cd = base["dev"], cand["dev"]
    if not np.array_equal(bd["global_row_id"], cd["global_row_id"]):
        raise ValueError("Dev row ids differ between the two runs")
    y = np.asarray(bd["y_true"], dtype=np.float64)
    if not np.array_equal(y, np.asarray(cd["y_true"], dtype=np.float64)):
        raise ValueError("Dev targets differ between the two runs")
    b_pred = np.asarray(bd["pred"], dtype=np.float64)
    c_pred = np.asarray(cd["pred"], dtype=np.float64)
    b_abs = np.abs(b_pred - y)
    c_abs = np.abs(c_pred - y)
    improvement = b_abs - c_abs
    worst_base = np.zeros(y.size, dtype=bool)
    worst_base[np.argsort(-b_abs)[:5]] = True
    top_improvers = np.zeros(y.size, dtype=bool)
    top_improvers[np.argsort(-improvement)[:5]] = True
    return {
        "base_arm": base["arm"],
        "cand_arm": cand["arm"],
        "n_dev": int(y.size),
        "base_rmse": rmse(b_pred, y),
        "cand_rmse": rmse(c_pred, y),
        "delta_rmse": rmse(c_pred, y) - rmse(b_pred, y),
        "base_mae": mae(b_pred, y),
        "cand_mae": mae(c_pred, y),
        "delta_mae": mae(c_pred, y) - mae(b_pred, y),
        "n_abs_error_improved": int((improvement > 0).sum()),
        "n_abs_error_worse": int((improvement < 0).sum()),
        "n_unchanged": int((improvement == 0).sum()),
        "mean_abs_error_improvement": float(improvement.mean()),
        "drop_five_worst_base": _drop_and_compare(worst_base, b_pred, c_pred, y),
        "drop_five_largest_improvements": _drop_and_compare(top_improvers, b_pred, c_pred, y),
        "worst_base_global_row_ids": [int(x) for x in bd["global_row_id"][worst_base].tolist()],
        "top_improver_global_row_ids": [int(x) for x in bd["global_row_id"][top_improvers].tolist()],
        "scaffold_groups": scaffold_group_stats(
            b_pred, c_pred, y, bd["group_key_sha256"], bd["scaffold"]
        ),
        "direction": "negative delta means the candidate has the lower (better) RMSE",
    }


# ---------------------------------------------------------------------------
# five-model table
# ---------------------------------------------------------------------------


def model_row(run: Mapping[str, Any]) -> dict[str, Any]:
    t = run["train"]
    m = t["metrics"]
    return {
        "arm": t["arm"],
        "pool": t["pool"],
        "parameters": t["model"]["parameters"],
        "train_rmse": m["train_rmse_raw"],
        "train_mae": m["train_mae_raw"],
        "dev_rmse": m["dev_rmse_raw"],
        "dev_mae": m["dev_mae_raw"],
        "dev_r2": m["dev_r2"],
        "generalisation_gap": m["generalisation_gap"],
        "epochs": t["recipe"]["epochs"],
        "optimizer_steps": t["recipe"]["total_updates"],
        "soup_members": t["recipe"]["soup_members"],
        "grad_clip_events": t["anomalies"]["grad_clip_events"],
        "grad_norm_max": t["anomalies"]["grad_norm_max"],
        "wall_seconds": t["timing"]["process_wall_seconds"],
        "train_loop_seconds": t["timing"]["train_loop_seconds"],
        "cuda_step_seconds": t["timing"]["cuda_step_seconds"],
        "cuda_peak_memory_gib": t["cuda_peak_memory_bytes"] / (1024 ** 3),
        "cuda_device_name": t["cuda_device_name"],
        "init_state_hash": t["model"]["init_state_hash"],
        "soup_state_hash": t["model"]["soup_state_hash"],
        "soup_sha256": t.get("artifacts", {}).get("soup", {}).get("sha256"),
        "git_commit": t["git_commit"],
        "git_dirty_tracked": t["git_dirty_tracked"],
        "torch": t["torch"],
    }


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(runs: Mapping[str, Mapping[str, Any]], comparisons: Mapping[str, Any]) -> dict[str, Any]:
    proto = comparisons[f"BASE_G0__vs__{C.PROTO_NAME}"]
    pair_vs_base = comparisons["BASE_G1__vs__PAIR_TRUE"]
    pair_vs_null = comparisons["PAIR_TRUE__vs__PAIR_NULL"]
    proto_gain = float(proto["base_rmse"] - proto["cand_rmse"])
    pair_gain = float(pair_vs_base["base_rmse"] - pair_vs_base["cand_rmse"])
    null_gain = float(pair_vs_null["base_rmse"] - pair_vs_null["cand_rmse"])
    proto_signal = proto_gain >= C.PROTO_STRONG_GAIN
    pair_signal = pair_gain >= C.PAIR_STRONG_GAIN and null_gain >= C.PAIR_VS_NULL_GAIN
    labels = []
    if proto_signal:
        labels.append("PROTO_STRONG_EXPLORATORY_SIGNAL")
    if pair_signal:
        labels.append("PAIR_STRONG_EXPLORATORY_SIGNAL")
    if not labels:
        labels.append("NO_STRONG_SIGNAL_THIS_CONFIGURATION")
    return {
        "proto": {
            "comparison": proto["base_arm"] + " -> " + proto["cand_arm"],
            "base_dev_rmse": proto["base_rmse"],
            "cand_dev_rmse": proto["cand_rmse"],
            "gain": proto_gain,
            "threshold": C.PROTO_STRONG_GAIN,
            "passed": bool(proto_gain >= C.PROTO_STRONG_GAIN),
            "label": "PROTO_STRONG_EXPLORATORY_SIGNAL" if proto_gain >= C.PROTO_STRONG_GAIN
            else "no_strong_signal_for_PROTO_this_configuration",
        },
        "pair": {
            "comparison_vs_base": pair_vs_base["base_arm"] + " -> " + pair_vs_base["cand_arm"],
            "gain_vs_base": pair_gain,
            "threshold_vs_base": C.PAIR_STRONG_GAIN,
            "comparison_vs_null": pair_vs_null["base_arm"] + " -> " + pair_vs_null["cand_arm"],
            "gain_vs_null": null_gain,
            "threshold_vs_null": C.PAIR_VS_NULL_GAIN,
            "passed": bool(pair_signal),
            "label": "PAIR_STRONG_EXPLORATORY_SIGNAL" if pair_signal
            else "no_strong_signal_for_PAIR_this_configuration",
        },
        "device_control": {
            "comparison": "BASE_G0 vs BASE_G1",
            "delta_rmse": comparisons["BASE_G0__vs__BASE_G1"]["delta_rmse"],
            "note": "same seed/code/data/recipe on two GPUs; the gap is run-to-run drift, not a model effect",
        },
        "labels": labels,
        "caveats": [
            "single seed (0)",
            "single scaffold fold (v2 Fold 2, Fit2352 only)",
            "exploratory only; not a confirmed mechanism or a paper-level performance claim",
            "the fixed thresholds are exploratory gates, not significance tests",
        ],
    }


# ---------------------------------------------------------------------------
# summary / ledger / verification
# ---------------------------------------------------------------------------


def summarize(*, write: bool = True) -> dict[str, Any]:
    runs = require_runs(C.ARMS)
    table = {arm: model_row(runs[arm]) for arm in C.ARMS}
    comparisons = {}
    for base_arm, cand_arm in C.COMPARISONS:
        comparisons[f"{base_arm}__vs__{cand_arm}"] = compare(runs[base_arm], runs[cand_arm])
    decision = decide(runs, comparisons)

    proto_diag = runs[C.PROTO_NAME]["diagnostics"]
    pair_true_diag = runs["PAIR_TRUE"]["diagnostics"]
    pair_null_diag = runs["PAIR_NULL"]["diagnostics"]

    summary = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "config_hash": C.config_hash(C.load_config()),
        "primary_metric": "Dev RMSE in original logD units (lower is better)",
        "protocol": {
            "dataset": "Lipophilicity Fit2352, v2 frozen scaffold Fold 2",
            "n_train": C.N_TRAIN,
            "n_dev": C.N_DEV,
            "seed": C.SEED,
            "fold": C.FOLD,
            "recipe": {
                "optimizer": "AdamW",
                "peak_lr": C.PEAK_LR,
                "min_lr": C.MIN_LR,
                "scheduler": "per_step_cosine",
                "weight_decay": C.WEIGHT_DECAY,
                "batch_size": C.BATCH_SIZE,
                "epochs": C.EPOCHS,
                "grad_clip": C.GRAD_CLIP,
                "dropout_at_h": C.DROPOUT,
                "soup_last": C.SOUP_LAST,
                "early_stopping": False,
                "dev_checkpoint_selection": False,
            },
            "held_out_labels_untouched": ["Monitor504", "Reserved504", "random42 Validation420/Test420"],
        },
        "models": table,
        "comparisons": comparisons,
        "diagnostics": {
            "proto_train": proto_diag["proto"]["train"],
            "proto_dev": proto_diag["proto"]["dev"],
            "proto_key_shuffle": proto_diag["proto"]["key_shuffle"],
            "proto_gradients": proto_diag["proto"]["gradients"],
            "pair_true_delta_dev": pair_true_diag["pair"]["delta_dev"],
            "pair_null_delta_dev": pair_null_diag["pair"]["delta_dev"],
            "pair_true_coverage_train": pair_true_diag["pair"]["coverage_train"],
            "pair_true_coverage_dev": pair_true_diag["pair"]["coverage_dev"],
            "pair_true_zero_residual": pair_true_diag["pair"]["zero_residual"],
            "pair_true_gradients": pair_true_diag["pair"]["gradients"],
            "pair_null_gradients": pair_null_diag["pair"]["gradients"],
        },
        "decision": decision,
        "artifacts": {
            "runs": {arm: C.track_rel(run_dir(arm)) for arm in C.ARMS},
        },
    }
    if write:
        C.write_json(C.SUMMARY, summary)
        write_tables(summary)
        C.write_json(
            C.ANALYSIS / "comparisons.json",
            {
                "protocol_id": C.PROTOCOL_ID,
                "created_utc": C.utcnow(),
                "comparisons": comparisons,
                "decision": decision,
            },
        )
    return summary


def write_tables(summary: Mapping[str, Any]) -> None:
    table_csv = C.ANALYSIS / "table_models.csv"
    with open(table_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = [
            "arm",
            "pool",
            "parameters",
            "train_rmse",
            "train_mae",
            "dev_rmse",
            "dev_mae",
            "dev_r2",
            "gen_gap",
            "epochs",
            "optimizer_steps",
            "grad_clip_events",
            "wall_seconds",
            "soup_members",
            "soup_sha256",
        ]
        writer.writerow(header)
        for arm in C.ARMS:
            row = summary["models"][arm]
            writer.writerow(
                [
                    row["arm"],
                    row["pool"],
                    row["parameters"],
                    f"{row['train_rmse']:.6f}",
                    f"{row['train_mae']:.6f}",
                    f"{row['dev_rmse']:.6f}",
                    f"{row['dev_mae']:.6f}",
                    f"{row['dev_r2']:.6f}",
                    f"{row['generalisation_gap']:.6f}",
                    row["epochs"],
                    row["optimizer_steps"],
                    row["grad_clip_events"],
                    f"{row['wall_seconds']:.1f}",
                    "/".join(str(x) for x in row["soup_members"]),
                    row["soup_sha256"],
                ]
            )

    comp_csv = C.ANALYSIS / "table_comparisons.csv"
    with open(comp_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "comparison",
                "base_dev_rmse",
                "cand_dev_rmse",
                "delta_rmse",
                "delta_mae",
                "n_abs_error_improved",
                "delta_rmse_drop5_worst_base",
                "delta_rmse_drop5_top_improvers",
            ]
        )
        for base_arm, cand_arm in C.COMPARISONS:
            row = summary["comparisons"][f"{base_arm}__vs__{cand_arm}"]
            writer.writerow(
                [
                    f"{base_arm} -> {cand_arm}",
                    f"{row['base_rmse']:.6f}",
                    f"{row['cand_rmse']:.6f}",
                    f"{row['delta_rmse']:+.6f}",
                    f"{row['delta_mae']:+.6f}",
                    row["n_abs_error_improved"],
                    f"{row['drop_five_worst_base']['delta_rmse_remaining']:+.6f}",
                    f"{row['drop_five_largest_improvements']['delta_rmse_remaining']:+.6f}",
                ]
            )

    groups_csv = C.ANALYSIS / "table_scaffold_groups.csv"
    with open(groups_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["comparison", "group_key_sha256", "scaffold", "n_molecules", "base_rmse", "cand_rmse", "delta_rmse"]
        )
        for base_arm, cand_arm in C.COMPARISONS:
            row = summary["comparisons"][f"{base_arm}__vs__{cand_arm}"]
            for group in row["scaffold_groups"]["groups"]:
                writer.writerow(
                    [
                        f"{base_arm} -> {cand_arm}",
                        group["group_key_sha256"],
                        group["scaffold"],
                        group["n_molecules"],
                        f"{group['base_rmse']:.6f}",
                        f"{group['cand_rmse']:.6f}",
                        f"{group['delta_rmse']:+.6f}",
                    ]
                )


def ledger(rr_status_dir: Path | str | None = None, *, write: bool = True) -> dict[str, Any]:
    """Assemble the GPU budget ledger from rr status JSON plus in-process timing."""
    directory = Path(rr_status_dir) if rr_status_dir is not None else (C.ANALYSIS / "rr_status")
    records = []
    for path in sorted(directory.glob("*.json")):
        status = C.read_json(path)
        key = str(status.get("experiment", path.stem))
        from datetime import datetime

        started = status.get("started_at")
        completed = status.get("completed_at")
        seconds = None
        if started and completed:
            seconds = (datetime.fromisoformat(completed) - datetime.fromisoformat(started)).total_seconds()
        records.append(
            {
                "experiment": key,
                "run_id": status.get("run_id"),
                "state": status.get("state"),
                "pool": status.get("pool"),
                "gpu_count": status.get("gpu_count"),
                "gpu_names": status.get("gpu_names"),
                "driver_version": status.get("driver_version"),
                "commit": status.get("commit"),
                "dirty": status.get("dirty"),
                "started_at": started,
                "completed_at": completed,
                "gpu_seconds": seconds,
                "exit_code": status.get("exit_code"),
                "source_json": C.track_rel(path),
            }
        )
    total_seconds = float(sum(r["gpu_seconds"] or 0.0 for r in records))
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "method": (
            "GPU-seconds are the rr-measured process lifetime (started_at -> completed_at) on an "
            "exclusive pool GPU; parallel pools are summed, not wall-clocked. In-process "
            "train-loop/pair-precompute/eval seconds are recorded separately for cross-checking."
        ),
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "runs": records,
        "n_runs": len(records),
        "total_gpu_seconds": total_seconds,
        "total_gpu_hours": total_seconds / 3600.0,
        "within_cap": bool(total_seconds / 3600.0 <= C.GPU_HARD_CAP_HOURS),
        "remaining_gpu_hours": max(0.0, C.GPU_HARD_CAP_HOURS - total_seconds / 3600.0),
    }
    if write:
        C.write_json(C.GPU_LEDGER, payload)
    return payload


def verify() -> dict[str, Any]:
    """Re-verify every run artefact and re-derive the reported metrics."""
    checks: list[dict[str, Any]] = []
    ok = True

    def check(name: str, condition: bool, detail: Any = None) -> None:
        nonlocal ok
        ok = ok and bool(condition)
        checks.append({"check": name, "ok": bool(condition), "detail": detail})

    view = C.load_fold_view()
    check("fold_dev_rows", list(view.dev_positions) == list(C.load_fold_view().dev_positions))
    for arm in C.ARMS:
        directory = run_dir(arm)
        if not (directory / "train.json").exists():
            check(f"{arm}:train_json", False, "missing")
            continue
        payload = C.read_json(directory / "train.json")
        soup = directory / "soup.pt"
        check(f"{arm}:soup_exists", soup.exists())
        if soup.exists():
            actual = C.sha256_file(soup)
            expected = payload.get("artifacts", {}).get("soup", {}).get("sha256")
            check(f"{arm}:soup_sha256", actual == expected, {"actual": actual, "expected": expected})
        pred_path = directory / "predictions_dev.npz"
        check(f"{arm}:predictions_exist", pred_path.exists())
        if pred_path.exists():
            pred = np.load(pred_path, allow_pickle=False)
            r = rmse(pred["pred"], pred["y_true"])
            reported = float(payload["metrics"]["dev_rmse_raw"])
            check(f"{arm}:dev_rmse_recomputes", abs(r - reported) < 1e-9, {"recomputed": r, "reported": reported})
            check(
                f"{arm}:dev_row_count",
                int(pred["global_row_id"].size) == C.N_DEV,
                int(pred["global_row_id"].size),
            )
            check(
                f"{arm}:residual_matches",
                bool(np.allclose(pred["residual"], pred["pred"] - pred["y_true"], atol=1e-12)),
            )
            check(
                f"{arm}:fit_cache_position_matches_fold",
                np.array_equal(
                    np.asarray(pred["fit_cache_position"], dtype=np.int64),
                    np.asarray(view.dev_positions, dtype=np.int64)[: int(pred["global_row_id"].size)],
                ),
            )
            check(
                f"{arm}:labels_scope",
                payload["labels_scope"]["monitor_labels_read"] is False
                and payload["labels_scope"]["reserved_labels_read"] is False
                and payload["labels_scope"]["official_valid_test_read"] is False
                and payload["labels_scope"]["random42_splits_read"] is False,
            )
            check(f"{arm}:clean_commit", payload["git_dirty_tracked"] is False, payload["git_commit"])
            check(f"{arm}:soup_members", payload["recipe"]["soup_members"] == list(range(356, 361)),
                  payload["recipe"]["soup_members"])
            check(f"{arm}:epochs", payload["recipe"]["epochs"] == C.EPOCHS)
            # State-dict reload reproduces the stored predictions.
            spec = C.V2C.load_frozen_spec()
            state = torch.load(soup, map_location="cpu", weights_only=False)
            from pilot_models import build_arm

            model, _ = build_arm(arm, spec, view.train_norm, C.SEED, "cpu")
            model.load_state_dict({k: v for k, v in state.items()}, strict=True)
            semantics = {"PAIR_TRUE": "true", "PAIR_NULL": "null"}.get(arm)
            pairs = P.build_pair_cache(view.dev_cache, semantics=semantics) if semantics else None
            from pilot_train import predict

            reloaded, _ = predict(model, view.dev_cache, device=torch.device("cpu"), pairs=pairs)
            stored = C.V2D.denormalise(reloaded, view.target_norm)
            # The stored predictions were produced on CUDA; this reload runs on
            # CPU, so only float32 reduction-order noise (~2e-6 logD) is allowed.
            reload_diff = float(np.abs(stored - pred["pred"]).max())
            check(f"{arm}:reload_predictions_match", reload_diff < 1e-5, reload_diff)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "ok": ok,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
        "checks": checks,
    }
    C.write_json(C.ANALYSIS / "verify_pilot.json", payload)
    return payload


__all__ = [
    "run_dir",
    "load_run",
    "require_runs",
    "compare",
    "model_row",
    "decide",
    "summarize",
    "write_tables",
    "ledger",
    "verify",
]
