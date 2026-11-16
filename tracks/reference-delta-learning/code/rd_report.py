"""Aggregate stage: tables, checks, decision and REPORT for rd-pilot-small-v1.

Everything here runs on CPU from the pulled per-dataset artefacts plus the
committed SQ caches.  It never needs a GPU, never retrains G/L, and never reads
the large pair feature matrices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import rd_common as C
import rd_data as D
import rd_eval as E

ARMS = ("D", "N", "R", "C")
REFERENCE_ARMS = ("G", "L")
COVERAGE_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SQ_RESULTS = C.SQ_TRACK_ROOT / "results" / "sq-pilot-small-v1"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _relative(path: Path) -> str:
    try:
        return str(Path(path).relative_to(C.REPO_ROOT))
    except ValueError:  # a rehearsal writing outside the repo
        return str(path)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def load_dataset(dataset: str) -> dict[str, Any]:
    base = C.results_dir(dataset)
    audit = C.read_json(base / "data_audit.json")
    references = C.read_json(base / "references.json")
    pairs = C.read_json(base / "pairs.json")
    selection = C.read_json(base / "selection.json")
    summary = C.read_json(base / "worker_summary.json")
    checks = C.read_json(base / "checks.json")
    predictions = {
        arm: _read_npz(base / f"pred_{arm}.npz") for arm in ARMS
    }
    delta = _read_npz(base / "pred_C_delta.npz")
    curves: dict[str, list[dict[str, Any]]] = {}
    for preset in C.PRESET_ORDER:
        curves[f"D_{preset}"] = _read_curves(base / f"curves_D_{preset}.csv")
        curves[f"C_{preset}"] = _read_curves(base / f"curves_C_{preset}.csv")
    pair_curves: dict[str, list[dict[str, Any]]] = {}
    for preset in C.PRESET_ORDER:
        pair_curves[preset] = _read_pair_curves(base / f"pair_curves_C_{preset}.csv")
    return {
        "dataset": dataset,
        "audit": audit,
        "references_json": references,
        "pairs_json": pairs,
        "selection": selection,
        "summary": summary,
        "checks": checks,
        "predictions": predictions,
        "delta": delta,
        "curves": curves,
        "pair_curves": pair_curves,
        "base": base,
    }


def _read_curves(path: Path) -> list[dict[str, Any]]:
    import csv

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "preset": row["preset"],
                    "round": int(row["round"]),
                    "split": row["split"],
                    "config": row["config"],
                    "rmse_raw": float(row["rmse_raw"]),
                    "mae_raw": float(row["mae_raw"]),
                }
            )
    return rows


def _read_pair_curves(path: Path) -> list[dict[str, Any]]:
    import csv

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "preset": row["preset"],
                    "round": int(row["round"]),
                    "pair_rmse_std": float(row["pair_rmse_std"]),
                    "pair_rmse_std_antisym": float(row["pair_rmse_std_antisym"]),
                }
            )
    return rows


def load_reference_arm(dataset: str, arm: str) -> dict[str, Any]:
    """Reused G/L point reference with full verification against this split."""
    npz_path = SQ_RESULTS / dataset / f"method_{arm}_pred.npz"
    json_path = SQ_RESULTS / dataset / f"method_{arm}.json"
    payload: dict[str, Any] = {
        "arm": arm,
        "source": _relative(npz_path),
        "source_commit": "4430042683bfad26c02391c10d17a26932716812",
        "available": False,
        "point_only": True,
        "reason": None,
    }
    task = D.load_task(dataset)
    if not npz_path.is_file():
        payload["reason"] = "per-sample predictions not present on this machine"
        return payload
    data = _read_npz(npz_path)
    payload["sha256"] = C.sha256_file(npz_path)
    payload["available"] = True
    for split in C.SPLITS:
        row_id = data[f"{split}_row_id"].astype(np.int64)
        y = data[f"{split}_y_raw"].astype(np.float64)
        pred = data[f"{split}_pred_raw"].astype(np.float64)
        if not np.array_equal(row_id, task.row_id[split]):
            payload["reason"] = f"{split} row_id mismatch"
            return payload
        if float(np.max(np.abs(y - task.y[split]))) > 1e-6:
            payload["reason"] = f"{split} label mismatch"
            return payload
        payload[f"{split}_rmse_raw"] = C.rmse(pred, y)
        payload[f"{split}_mae_raw"] = C.mae(pred, y)
        payload[f"{split}_pred"] = pred
        payload[f"{split}_y"] = y
        payload[f"{split}_row_id"] = row_id
    if json_path.is_file():
        meta = C.read_json(json_path)
        selected = meta.get("selected", {})
        payload["report"] = {
            key: selected.get(key)
            for key in ("valid_rmse_raw", "test_rmse_raw", "valid_mae_raw", "test_mae_raw", "train_rmse_raw")
            if key in selected
        }
        payload["report_json_sha256"] = C.sha256_file(json_path)
        for split in ("valid", "test", "train"):
            key = f"{split}_rmse_raw"
            if key in payload["report"] and payload["report"][key] is not None:
                payload[f"{split}_recompute_abs_diff"] = abs(
                    float(payload[key]) - float(payload["report"][key])
                )
        # the SQ json stores float32-accumulated metrics; 1e-6 is the metric tolerance
        payload["verified"] = all(
            payload.get(f"{split}_recompute_abs_diff", 0.0) <= 1e-6 for split in ("valid", "test")
        )
    return payload


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def main_table(data: Mapping[str, Any], references: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    dataset = data["dataset"]
    selection = data["selection"]
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        state = selection[arm]
        entry = {
            "dataset": dataset,
            "arm": arm,
            "preset": state.get("preset"),
            "round": state.get("round"),
            "used_rounds": state.get("used_rounds"),
            "trained_rounds": state.get("trained_rounds"),
            "config": state.get("config", "none"),
            "K": None,
            "weights": None,
        }
        if state.get("config"):
            k, mode = _split_config(state["config"])
            entry["K"] = k
            entry["weights"] = mode
        for split in C.SPLITS:
            entry[f"{split}_rmse"] = float(state[f"{split}_rmse_raw"])
            entry[f"{split}_mae"] = float(state[f"{split}_mae_raw"])
        entry["selected_by"] = "valid_rmse"
        entry["source"] = "this round"
        rows.append(entry)
    for arm in REFERENCE_ARMS:
        ref = references[arm]
        if not ref["available"]:
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "preset": "reused",
                    "round": None,
                    "used_rounds": None,
                    "trained_rounds": None,
                    "config": "none",
                    "K": None,
                    "weights": None,
                    "train_rmse": float("nan"),
                    "valid_rmse": float("nan"),
                    "test_rmse": float("nan"),
                    "train_mae": float("nan"),
                    "valid_mae": float("nan"),
                    "test_mae": float("nan"),
                    "selected_by": "reference (not retrained)",
                    "source": f"reused {arm} (missing)",
                }
            )
            continue
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "preset": "reused",
                "round": None,
                "used_rounds": None,
                "trained_rounds": None,
                "config": "none",
                "K": None,
                "weights": None,
                "train_rmse": ref["train_rmse_raw"],
                "valid_rmse": ref["valid_rmse_raw"],
                "test_rmse": ref["test_rmse_raw"],
                "train_mae": ref["train_mae_raw"],
                "valid_mae": ref["valid_mae_raw"],
                "test_mae": ref["test_mae_raw"],
                "selected_by": "reused point reference",
                "source": ref["source"],
            }
        )
    return rows


def _split_config(name: str) -> tuple[int, str]:
    for config in C.CONFIGS:
        if config["name"] == name:
            return int(config["K"]), str(config["weights"])
    raise KeyError(name)



def _curve_rows(
    data: Mapping[str, Any], arm: str, preset: str, config: str | None
) -> list[dict[str, Any]]:
    """Curve rows of one arm/preset/config from the persisted per-preset CSVs."""
    if arm == "N":
        return []
    table = data["curves"]["D" if arm in ("D", "R") else "C"] if preset in C.PRESET_ORDER else {}
    if not isinstance(table, dict):
        return []
    rows = table.get(preset, [])
    return [
        row
        for row in rows
        if row["method"] == arm and (config is None or row["config"] == config)
    ]


def curve_summary(
    data: Mapping[str, Any], arm: str, state: Mapping[str, Any]
) -> dict[str, Any]:
    """Selected / final-round / mean-of-last-20 values of the *same* frozen recipe."""
    out: dict[str, Any] = {"dataset": data["dataset"], "arm": arm}
    preset = str(state.get("preset"))
    config = state.get("config")
    rows = _curve_rows(data, arm, preset, config)
    trained = int(state.get("trained_rounds") or 0)
    for split in ("valid", "test"):
        selected_value = float(state[f"{split}_rmse_raw"])
        out[f"{split}_selected"] = selected_value
        if not rows:
            # N is a constant predictor: its single state *is* its curve
            out[f"{split}_final"] = selected_value
            out[f"{split}_mean_last20"] = selected_value
            continue
        at_final = [float(row["rmse_raw"]) for row in rows if row["split"] == split and int(row["round"]) == trained]
        out[f"{split}_final"] = at_final[0] if at_final else None
        tail = sorted(
            (row for row in rows if row["split"] == split), key=lambda row: int(row["round"])
        )[-20:]
        out[f"{split}_mean_last20"] = float(np.mean([float(row["rmse_raw"]) for row in tail])) if tail else None
    out["rounds"] = trained
    return out

def reference_coverage(
    data: Mapping[str, Any], task: D.TaskData
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = data["dataset"]
    refs = _read_npz(data["base"] / "references.npz")
    predictions = {arm: data["predictions"][arm] for arm in ARMS}
    rows: list[dict[str, Any]] = []
    quantiles: dict[str, Any] = {}
    for split in C.SPLITS:
        max_sim = refs[f"{split}__max_sim"].astype(np.float64)
        y = predictions["D"][f"{split}__y_raw"].astype(np.float64)
        if not np.array_equal(
            predictions["D"][f"{split}__row_id"].astype(np.int64), task.row_id[split]
        ):
            raise AssertionError(f"{dataset}: prediction row_id does not match the split")
        quantiles[split] = {
            "n": int(max_sim.size),
            "min": float(max_sim.min()),
            "q25": float(np.quantile(max_sim, 0.25)),
            "median": float(np.quantile(max_sim, 0.5)),
            "q75": float(np.quantile(max_sim, 0.75)),
            "max": float(max_sim.max()),
            "mean": float(max_sim.mean()),
        }
        for lower, upper in zip(COVERAGE_BINS[:-1], COVERAGE_BINS[1:]):
            last = upper == COVERAGE_BINS[-1]
            mask = (max_sim >= lower) & ((max_sim <= upper) if last else (max_sim < upper))
            entry = {
                "dataset": dataset,
                "split": split,
                "bin_low": lower,
                "bin_high": upper,
                "n_queries": int(mask.sum()),
            }
            for arm in ARMS:
                pred = predictions[arm][f"{split}__pred_raw"].astype(np.float64)
                entry[f"{arm}_rmse"] = float(C.rmse(pred[mask], y[mask])) if mask.any() else None
            rows.append(entry)
    return rows, quantiles


def comparisons(
    data: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
    *,
    n_resamples: int = 2000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = data["dataset"]
    selection = data["selection"]
    preds = {arm: data["predictions"][arm] for arm in ARMS}
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    indices: dict[str, np.ndarray] = {}
    for split in ("valid", "test"):
        n = int(preds["C"][f"{split}__y_raw"].size)
        indices[split] = rng.integers(0, n, size=(int(n_resamples), n))
    for split in ("valid", "test"):
        y = preds["C"][f"{split}__y_raw"].astype(np.float64)
        pred_c = preds["C"][f"{split}__pred_raw"].astype(np.float64)
        base = {"D": preds["D"], "N": preds["N"], "R": preds["R"]}
        for other in ("D", "N", "R"):
            pred_b = base[other][f"{split}__pred_raw"].astype(np.float64)
            boot = _bootstrap_with_indices(pred_c, pred_b, y, indices[split])
            drop = E.leave_one_out_delta(pred_c, pred_b, y)
            row_id = preds["C"][f"{split}__row_id"].astype(np.int64)
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"C-{other}",
                    "split": split,
                    "rmse_c": boot["rmse_c"],
                    "rmse_b": boot["rmse_b"],
                    "delta_rmse": boot["delta"],
                    "relative_delta": boot["delta"] / boot["rmse_b"],
                    "gain": 1.0 - boot["rmse_c"] / boot["rmse_b"],
                    "ci95_low": boot["ci95_low"],
                    "ci95_high": boot["ci95_high"],
                    "n_resamples": int(n_resamples),
                    "drop_one_row_id": int(row_id[drop["dropped_index"]]),
                    "drop_one_delta": drop["delta_drop1"],
                    "drop_one_sign_flip": bool(drop["sign_flip"]),
                }
            )
    # B = best verifiable non-joint reference by Val RMSE, Test travels with it
    candidates: dict[str, dict[str, float]] = {}
    for arm in ("D", "N", "R"):
        state = selection[arm]
        candidates[arm] = {
            "valid": float(state["valid_rmse_raw"]),
            "test": float(state["test_rmse_raw"]),
        }
    for arm in REFERENCE_ARMS:
        ref = references[arm]
        if ref["available"]:
            candidates[arm] = {
                "valid": float(ref["valid_rmse_raw"]),
                "test": float(ref["test_rmse_raw"]),
            }
        else:
            candidates[arm] = {"valid": float("nan"), "test": float("nan")}
    usable = {k: v for k, v in candidates.items() if np.isfinite(v["valid"])}
    best_arm = min(usable, key=lambda name: (usable[name]["valid"], name))
    best = {"arm": best_arm, **usable[best_arm]}
    c_state = selection["C"]
    best["competitive_c"] = bool(
        float(c_state["valid_rmse_raw"]) <= 1.05 * best["valid"]
        and float(c_state["test_rmse_raw"]) <= 1.05 * best["test"]
    )
    best["ratio_c_over_b"] = {
        "valid": float(c_state["valid_rmse_raw"]) / best["valid"],
        "test": float(c_state["test_rmse_raw"]) / best["test"],
    }
    return rows, best


def _bootstrap_with_indices(
    pred_c: np.ndarray, pred_b: np.ndarray, y: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    base_c = float(C.rmse(pred_c, y))
    base_b = float(C.rmse(pred_b, y))
    deltas = np.empty(int(indices.shape[0]), dtype=np.float64)
    for row in range(int(indices.shape[0])):
        chosen = indices[row]
        deltas[row] = C.rmse(pred_c[chosen], y[chosen]) - C.rmse(pred_b[chosen], y[chosen])
    return {
        "rmse_c": base_c,
        "rmse_b": base_b,
        "delta": base_c - base_b,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def case_ledger(data: Mapping[str, Any], task: D.TaskData) -> list[dict[str, Any]]:
    """Three Val queries per task: median C error, best and worst C-vs-D change."""
    dataset = data["dataset"]
    selection = data["selection"]
    preds = data["predictions"]
    y = preds["C"]["valid__y_raw"].astype(np.float64)
    pred_c = preds["C"]["valid__pred_raw"].astype(np.float64)
    pred_d = preds["D"]["valid__pred_raw"].astype(np.float64)
    row_id = preds["C"]["valid__row_id"].astype(np.int64)
    err_c = np.abs(pred_c - y)
    err_d = np.abs(pred_d - y)
    order = np.lexsort((row_id, err_c))
    n = int(y.size)
    picked: list[tuple[str, int]] = [("median_abs_error_C", int(order[n // 2]))]
    for label, key in (
        ("largest_improvement_vs_D", -(err_c - err_d)),
        ("largest_regression_vs_D", (err_c - err_d)),
    ):
        chosen = None
        for candidate in np.lexsort((row_id, key)):
            if int(candidate) not in [index for _, index in picked]:
                chosen = int(candidate)
                break
        if chosen is not None:
            picked.append((label, chosen))

    delta = data["delta"]
    weights = delta["valid__weights"].astype(np.float64)
    y_ref = delta["valid__y_ref"].astype(np.float64)
    ref_row_id = delta["valid__ref_row_id"].astype(np.int64)
    sim = delta["valid__sim"].astype(np.float64)
    delta_raw = delta["valid__delta_raw"].astype(np.float64)
    config = selection["C"]["config"]
    k_eff = min(_split_config(config)[0], int(np.sum(weights > 0, axis=1)[0]))
    rows: list[dict[str, Any]] = []
    smiles = task.smiles["valid"]
    # ref_row_id is the *global* row_id; the SMILES list is in Train-local order
    train_smiles = {
        int(row_id): text for row_id, text in zip(task.row_id["train"], task.smiles["train"])
    }
    train_ids = set(train_smiles)
    for label, index in picked:
        if not set(ref_row_id[index, :k_eff].tolist()) <= train_ids:
            raise AssertionError(f"{dataset}: a reference is not a Train row")
        contribution = weights[index, :k_eff] * (y_ref[index, :k_eff] + delta_raw[index, :k_eff])
        total = float(contribution.sum())
        top = np.argsort(-weights[index, :k_eff])[:5]
        top_sum = float(contribution[top].sum())
        remainder = total - top_sum
        rows.append(
            {
                "dataset": dataset,
                "ledger_case": label,
                "query_index": int(index),
                "query_row_id": int(row_id[index]),
                "query_smiles": smiles[index],
                "y_raw": float(y[index]),
                "pred_D_raw": float(pred_d[index]),
                "pred_C_raw": float(pred_c[index]),
                "abs_error_C": float(err_c[index]),
                "abs_error_D": float(err_d[index]),
                "change_vs_D": float(err_c[index] - err_d[index]),
                "config": config,
                "k_eff": int(k_eff),
                "top5_weighted_sum": top_sum,
                "remaining_weighted_sum": remainder,
                "ledger_total": total,
                "prediction_abs_diff": abs(total - float(pred_c[index])),
                "top5": [
                    {
                        "slot": int(slot),
                        "ref_row_id": int(ref_row_id[index, slot]),
                        "ref_smiles": train_smiles[int(ref_row_id[index, slot])],
                        "similarity": float(sim[index, slot]),
                        "weight": float(weights[index, slot]),
                        "y_ref_raw": float(y_ref[index, slot]),
                        "delta_raw": float(delta_raw[index, slot]),
                        "y_ref_plus_delta": float(y_ref[index, slot] + delta_raw[index, slot]),
                    }
                    for slot in top
                ],
            }
        )
    return rows


def resource_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = data["dataset"]
    summary = data["summary"]
    rows: list[dict[str, Any]] = []
    for preset, info in summary["trajectories"]["D"].items():
        rows.append(
            {
                "dataset": dataset,
                "arm": "D",
                "preset": preset,
                "rounds": int(info["trained_rounds"]),
                "train_seconds": float(info["seconds"]),
                "eval_seconds": float(info["eval_seconds"]),
                "matrix_build_seconds": 0.0,
                "seconds_per_round": float(info["seconds"]) / max(int(info["trained_rounds"]), 1),
                "gpu_peak_mb": info.get("gpu_peak_mb"),
                "model_sha256": info.get("model_sha256"),
            }
        )
    for preset, info in summary["trajectories"]["C"].items():
        rows.append(
            {
                "dataset": dataset,
                "arm": "C",
                "preset": preset,
                "rounds": int(info["trained_rounds"]),
                "train_seconds": float(info["seconds"]),
                "eval_seconds": float(info["eval_seconds"]),
                "matrix_build_seconds": float(info["matrix_build_seconds"]),
                "seconds_per_round": float(info["seconds"]) / max(int(info["trained_rounds"]), 1),
                "gpu_peak_mb": info.get("gpu_peak_mb"),
                "model_sha256": info.get("model_sha256"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decision_payload(
    all_main: Sequence[Mapping[str, Any]],
    all_comparisons: Sequence[Mapping[str, Any]],
    best: Mapping[str, Mapping[str, Any]],
    datasets: Sequence[str],
    checks_ok: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    per_dataset: dict[str, Any] = {}
    for dataset in datasets:
        rows = {row["arm"]: row for row in all_main if row["dataset"] == dataset}
        comp = {
            (row["comparison"], row["split"]): row
            for row in all_comparisons
            if row["dataset"] == dataset
        }
        gains = {
            f"{other}_{split}": float(comp[(f"C-{other}", split)]["gain"])
            for other in ("D", "N", "R")
            for split in ("valid", "test")
        }
        strong_conditions = {key: bool(value >= 0.03) for key, value in gains.items()}
        regress_conditions = {key: bool(value >= -0.05) for key, value in gains.items()}
        strong = all(strong_conditions.values())
        not_regressing = all(regress_conditions.values())
        rmse = {
            arm: {"valid": float(rows[arm]["valid_rmse"]), "test": float(rows[arm]["test_rmse"])}
            for arm in ARMS
        }
        per_dataset[dataset] = {
            "gains": gains,
            "strong_conditions": strong_conditions,
            "not_worse_than_5pct_conditions": regress_conditions,
            "strong_joint_task": strong,
            "not_regressing": not_regressing,
            "rmse": rmse,
            "best_reference": dict(best[dataset]),
            "competitive_with_frozen_references": bool(best[dataset]["competitive_c"]),
            "progress": _progress_axis(rmse, best[dataset]),
        }

    strong_tasks = [dataset for dataset in datasets if per_dataset[dataset]["strong_joint_task"]]
    signal: str
    if strong_tasks and all(
        per_dataset[dataset]["not_regressing"] for dataset in datasets if dataset not in strong_tasks
    ):
        signal = "JOINT_DELTA_SIGNAL"
    elif strong_tasks:
        signal = "CROSS_TASK_TRADEOFF"
    else:
        better_val = [
            per_dataset[dataset]["rmse"]["C"]["valid"]
            < min(per_dataset[dataset]["rmse"][arm]["valid"] for arm in ("D", "N", "R"))
            for dataset in datasets
        ]
        better_test = [
            per_dataset[dataset]["rmse"]["C"]["test"]
            < min(per_dataset[dataset]["rmse"][arm]["test"] for arm in ("D", "N", "R"))
            for dataset in datasets
        ]
        beats_n = any(
            per_dataset[dataset]["gains"]["N_valid"] > 0 and per_dataset[dataset]["gains"]["N_test"] > 0
            for dataset in datasets
        )
        beats_r = any(
            per_dataset[dataset]["gains"]["R_valid"] > 0 and per_dataset[dataset]["gains"]["R_test"] > 0
            for dataset in datasets
        )
        if any(a != b for a, b in zip(better_val, better_test)):
            signal = "SPLIT_INCONSISTENT"
        elif not beats_n and not beats_r:
            signal = "NO_JOINT_INCREMENT_THIS_RECIPE"
        else:
            signal = "SMALL_OR_UNCERTAIN_GAIN"
    substantive = any(
        per_dataset[dataset]["strong_joint_task"]
        and per_dataset[dataset]["progress"]["substantive_progress"]
        for dataset in datasets
    )
    reference_complete = all(
        references[dataset][arm]["available"] and references[dataset][arm].get("verified", False)
        for dataset in datasets
        for arm in REFERENCE_ARMS
    )
    execution_pass = bool(
        all(checks_ok[dataset]["pass"] for dataset in datasets)
        and all(
            per_dataset[dataset]["rmse"][arm]["valid"] > 0.0 for dataset in datasets for arm in ARMS
        )
    )
    return {
        "protocol_id": C.ROUND_ID,
        "execution": {
            "EXECUTION_PASS": execution_pass,
            "checks_ok": {dataset: dict(checks_ok[dataset]) for dataset in datasets},
            "formal_trajectories": 4 * len(datasets),
            "reference_complete": reference_complete,
        },
        "joint_signal": {
            "JOINT_DELTA_SIGNAL": signal == "JOINT_DELTA_SIGNAL",
            "classification": signal,
            "strong_joint_tasks": strong_tasks,
        },
        "substantive_progress": {
            "SUBSTANTIVE_PROGRESS": bool(substantive),
            "per_dataset": {
                dataset: per_dataset[dataset]["progress"] for dataset in datasets
            },
        },
        "competitive": {
            dataset: {
                "COMPETITIVE_WITH_FROZEN_REFERENCES": per_dataset[dataset][
                    "competitive_with_frozen_references"
                ],
                "best_reference": per_dataset[dataset]["best_reference"],
            }
            for dataset in datasets
        },
        "reference_axis": "REFERENCE_COMPLETE" if reference_complete else "REFERENCE_INCOMPLETE",
        "worth_structural_comparator_followup": bool(
            execution_pass and signal == "JOINT_DELTA_SIGNAL" and substantive
        ),
        "per_dataset": per_dataset,
    }


def _progress_axis(
    rmse: Mapping[str, Mapping[str, float]], best: Mapping[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {"per_split": {}}
    for split in ("valid", "test"):
        j = min(rmse[arm][split] for arm in ("D", "N", "R"))
        b = float(best[split])
        c = float(rmse["C"][split])
        if j > b + 1e-12:
            half_gap_closed = bool(c <= b + 0.5 * (j - b))
            half_gap_status: Any = half_gap_closed
        else:
            half_gap_closed = None
            half_gap_status = "not_applicable"
        competitive = bool(c <= 1.05 * b)
        out["per_split"][split] = {
            "J_min_DNR": j,
            "b_best_reference": b,
            "c_joint": c,
            "half_gap_closed": half_gap_status,
            "competitive_5pct": competitive,
            "progress_pass": bool(competitive or (half_gap_closed is True)),
        }
    out["substantive_progress"] = all(
        out["per_split"][split]["progress_pass"] for split in ("valid", "test")
    )
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and (value != value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def write_tables(
    datasets: Sequence[str],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, str]:
    out_dir = C.ensure_dir(C.RESULTS_DIR / "tables")
    files: dict[str, str] = {}
    for name, rows in tables.items():
        path = out_dir / f"{name}.csv"
        if rows:
            columns = list(rows[0].keys())
            for row in rows[1:]:
                for key in row:
                    if key not in columns:
                        columns.append(key)
            C.write_csv(path, rows, columns=columns)
        else:
            path.write_text("", encoding="utf-8")
        files[name] = _relative(path)
    return files


def aggregate(*, ledger: C.Ledger) -> dict[str, Any]:
    datasets = list(C.DATASETS)
    data = {}
    tasks = {}
    references: dict[str, dict[str, Mapping[str, Any]]] = {}
    with C.stage_timer(ledger, "aggregate_load"):
        for dataset in datasets:
            data[dataset] = load_dataset(dataset)
            tasks[dataset] = D.load_task(dataset)
            references[dataset] = {
                arm: load_reference_arm(dataset, arm) for arm in REFERENCE_ARMS
            }

    main_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    resource_rows_all: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    curve_rows_summary: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    quantiles: dict[str, Any] = {}
    with C.stage_timer(ledger, "aggregate_tables"):
        for dataset in datasets:
            main_rows.extend(main_table(data[dataset], references[dataset]))
            comp, best_ref = comparisons(data[dataset], references[dataset])
            comparison_rows.extend(comp)
            best[dataset] = best_ref
            cov, quant = reference_coverage(data[dataset], tasks[dataset])
            coverage_rows.extend(cov)
            quantiles[dataset] = quant
            case_rows.extend(case_ledger(data[dataset], tasks[dataset]))
            resource_rows_all.extend(resource_rows(data[dataset]))
            for arm in ARMS:
                curve_rows_summary.append(
                    curve_summary(data[dataset], arm, data[dataset]["selection"][arm])
                )
            state = data[dataset]["selection"]["C"]
            fit_rows.append(
                {
                    "dataset": dataset,
                    "arm": "C",
                    "config": state["config"],
                    "pair_rmse_std_selected": state.get("pair_rmse_std_at_selected"),
                    "pair_rmse_std_antisym_selected": state.get(
                        "pair_rmse_std_antisym_at_selected"
                    ),
                    "mean_last20_pair_rmse_std": state.get("mean_last20_pair_rmse_std"),
                    "train_rmse_molecule": state["train_rmse_raw"],
                    "valid_rmse_molecule": state["valid_rmse_raw"],
                    "test_rmse_molecule": state["test_rmse_raw"],
                }
            )
        fit_rows.append(
            {
                "dataset": datasets[0],
                "arm": "C-fusion-check",
                "config": "C == R algebra test in tests/test_rd_algebra.py",
                "pair_rmse_std_selected": None,
                "pair_rmse_std_antisym_selected": None,
                "mean_last20_pair_rmse_std": None,
                "train_rmse_molecule": None,
                "valid_rmse_molecule": None,
                "test_rmse_molecule": None,
            }
        )

    expected_trajectories = 2 * len(C.PRESET_ORDER) * len(datasets)
    checks_ok = {dataset: data[dataset]["summary"]["checks_ok"] for dataset in datasets}
    decision = decision_payload(
        main_rows, comparison_rows, best, datasets, checks_ok, references
    )
    decision["execution"]["formal_trajectories"] = expected_trajectories
    decision["execution"]["EXECUTION_PASS"] = bool(
        decision["execution"]["EXECUTION_PASS"]
        and all(
            len(data[dataset]["selection"]["trajectories"]["D"]) == len(C.PRESET_ORDER)
            and len(data[dataset]["selection"]["trajectories"]["C"]) == len(C.PRESET_ORDER)
            for dataset in datasets
        )
    )

    with C.stage_timer(ledger, "aggregate_write"):
        files = write_tables(
            datasets,
            {
                "main": main_rows,
                "comparisons": comparison_rows,
                "coverage": coverage_rows,
                "cases": case_rows,
                "resources": resource_rows_all,
                "c_fit_levels": fit_rows,
                "curve_summary": curve_rows_summary,
                "configs": [
                    config
                    for dataset in datasets
                    for config in data[dataset]["references_json"]["summary"]
                ],
            },
        )
        C.write_json(C.RESULTS_DIR / "decision.json", decision)
        C.write_json(
            C.RESULTS_DIR / "coverage_quantiles.json",
            quantiles,
        )
        C.write_json(
            C.RESULTS_DIR / "reference_check.json",
            {
                dataset: {
                    arm: {
                        key: value
                        for key, value in references[dataset][arm].items()
                        if not key.endswith(("_pred", "_y", "_row_id"))
                    }
                    for arm in REFERENCE_ARMS
                }
                for dataset in datasets
            },
        )
        report = render_report(
            datasets, data, main_rows, comparison_rows, coverage_rows, case_rows,
            resource_rows_all, fit_rows, curve_rows_summary, quantiles, decision, references,
        )
        (C.RESULTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    metrics = {
        "datasets": datasets,
        "execution_pass": decision["execution"]["EXECUTION_PASS"],
        "joint_signal": decision["joint_signal"]["classification"],
        "substantive_progress": decision["substantive_progress"]["SUBSTANTIVE_PROGRESS"],
        "worth_followup": decision["worth_structural_comparator_followup"],
        "reference_axis": decision["reference_axis"],
        "tables": files,
        "main_table": main_rows,
        "decision": decision,
    }
    return json.loads(json.dumps(metrics, default=_json_default))


def render_report(
    datasets: Sequence[str],
    data: Mapping[str, Any],
    main_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    resource_rows_all: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    curve_rows_summary: Sequence[Mapping[str, Any]],
    quantiles: Mapping[str, Any],
    decision: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> str:
    lines: list[str] = []
    lines.append("# Reference-delta learning pilot — rd-pilot-small-v1")
    lines.append("")
    lines.append(
        "官方 OGB `ogbg-molesol` / `ogbg-mollipo`，官方 Train/Validation/Test 划分，"
        "单训练 seed=0，`test_policy: always`（Test 作开发评估集，不参与任何选择）。"
        "本文件由 `stage=aggregate` 依据两个 GPU worker 的产物生成。"
    )
    lines.append("")
    lines.append("## 主表（每任务每臂：Val 选中状态，原始单位）")
    lines.append("")
    lines.append(
        _table(
            main_rows,
            [
                "dataset",
                "arm",
                "preset",
                "round",
                "used_rounds",
                "trained_rounds",
                "config",
                "train_rmse",
                "valid_rmse",
                "test_rmse",
                "train_mae",
                "valid_mae",
                "test_mae",
            ],
        )
    )
    lines.append("")
    lines.append(
        "D/N/R/C 均为本轮产物：D、C 每个 preset 一条正式轨迹，N 为常量（无训练），"
        "R 复用 D 的树前缀并重新选择检索配置。G/L 为 44300426 的复用点参照，"
        "行级 row_id/y 已核验并复算 RMSE；未重训。"
    )
    lines.append("")
    lines.append("## 曲线摘要（同一配方/配置：选中轮 / 最后一轮 / 最后 20 轮均值）")
    lines.append("")
    lines.append(
        _table(
            curve_rows_summary,
            [
                "dataset",
                "arm",
                "rounds",
                "valid_selected",
                "valid_final",
                "valid_mean_last20",
                "test_selected",
                "test_final",
                "test_mean_last20",
            ],
        )
    )
    lines.append("")
    lines.append(
        "最后一轮与最后 20 轮均值只作曲线摘要报告，从不替代 Val 选中的状态；"
        "N 为常量预测，三者相同。选中轮早于最后一轮时，二者的差就是该配方在本任务上的"
        "过拟合/漂移幅度。"
    )
    lines.append("")
    lines.append("## C 相对 D / N / R 的差距与配对不确定性")
    lines.append("")
    lines.append(
        _table(
            comparison_rows,
            [
                "dataset",
                "comparison",
                "split",
                "rmse_c",
                "rmse_b",
                "delta_rmse",
                "relative_delta",
                "ci95_low",
                "ci95_high",
                "drop_one_delta",
                "drop_one_sign_flip",
            ],
        )
    )
    lines.append("")
    lines.append(
        "bootstrap 按 query 分子重采样 2000 次（RNG 1703，固定模型与训练参照，"
        "pair 不作为独立样本）；`drop_one_*` 为移除该比较中参照模型误差最大的同一个 "
        "query 后重算的 ΔRMSE。区间只描述固定模型下的重采样不确定性。"
    )
    lines.append("")
    lines.append("## 冻结判定")
    lines.append("")
    lines.append(
        _table(
            [
                {
                    "dataset": dataset,
                    "arm_C_val": decision["per_dataset"][dataset]["rmse"]["C"]["valid"],
                    "arm_C_test": decision["per_dataset"][dataset]["rmse"]["C"]["test"],
                    "best_reference": decision["per_dataset"][dataset]["best_reference"]["arm"],
                    "best_valid": decision["per_dataset"][dataset]["best_reference"]["valid"],
                    "best_test": decision["per_dataset"][dataset]["best_reference"]["test"],
                    "competitive": decision["per_dataset"][dataset][
                        "competitive_with_frozen_references"
                    ],
                    "strong_joint_task": decision["per_dataset"][dataset]["strong_joint_task"],
                    "not_regressing": decision["per_dataset"][dataset]["not_regressing"],
                    "val_progress": decision["per_dataset"][dataset]["progress"]["per_split"][
                        "valid"
                    ]["progress_pass"],
                    "test_progress": decision["per_dataset"][dataset]["progress"]["per_split"][
                        "test"
                    ]["progress_pass"],
                    "substantive": decision["per_dataset"][dataset]["progress"][
                        "substantive_progress"
                    ],
                }
                for dataset in datasets
            ],
            [
                "dataset",
                "arm_C_val",
                "arm_C_test",
                "best_reference",
                "best_valid",
                "best_test",
                "competitive",
                "strong_joint_task",
                "not_regressing",
                "val_progress",
                "test_progress",
                "substantive",
            ],
        )
    )
    lines.append("")
    for dataset in datasets:
        entry = decision["per_dataset"][dataset]
        lines.append(f"**{dataset}**")
        lines.append("")
        lines.append(
            "- 相对增益 gain_b = 1 - RMSE_C/RMSE_b：" + ", ".join(
                f"{key}={_fmt(value)}" for key, value in sorted(entry["gains"].items())
            )
        )
        lines.append(
            "- strong_joint_task="
            + _fmt(entry["strong_joint_task"])
            + "（六条件："
            + ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(entry["strong_conditions"].items()))
            + "）"
        )
        lines.append(
            "- 另一任务 5% 条件："
            + ", ".join(
                f"{k}={_fmt(v)}" for k, v in sorted(entry["not_worse_than_5pct_conditions"].items())
            )
        )
        for split in ("valid", "test"):
            axis = entry["progress"]["per_split"][split]
            lines.append(
                f"- {split}: J={_fmt(axis['J_min_DNR'])} b={_fmt(axis['b_best_reference'])} "
                f"c={_fmt(axis['c_joint'])} half_gap_closed={_fmt(axis['half_gap_closed'])} "
                f"competitive_5pct={_fmt(axis['competitive_5pct'])} "
                f"progress_pass={_fmt(axis['progress_pass'])}"
            )
        lines.append("")
    lines.append(
        "- JOINT_DELTA_SIGNAL: **" + decision["joint_signal"]["classification"] + "**"
    )
    lines.append(
        "- SUBSTANTIVE_PROGRESS: **"
        + _fmt(decision["substantive_progress"]["SUBSTANTIVE_PROGRESS"])
        + "**"
    )
    lines.append(
        "- EXECUTION_PASS: **" + _fmt(decision["execution"]["EXECUTION_PASS"]) + "**"
    )
    lines.append(
        "- WORTH_STRUCTURAL_COMPARATOR_FOLLOWUP: **"
        + _fmt(decision["worth_structural_comparator_followup"])
        + "**"
    )
    lines.append("")
    lines.append("## C 的拟合层级（pair / 分子 Train / Val / Test）")
    lines.append("")
    lines.append(
        _table(
            [row for row in fit_rows if row["arm"] == "C"],
            [
                "dataset",
                "config",
                "pair_rmse_std_selected",
                "pair_rmse_std_antisym_selected",
                "mean_last20_pair_rmse_std",
                "train_rmse_molecule",
                "valid_rmse_molecule",
                "test_rmse_molecule",
            ],
        )
    )
    lines.append("")
    lines.append("## 参照覆盖（最高合法参照 Tanimoto 的固定分箱）")
    lines.append("")
    lines.append(
        _table(
            coverage_rows,
            [
                "dataset",
                "split",
                "bin_low",
                "bin_high",
                "n_queries",
                "D_rmse",
                "N_rmse",
                "R_rmse",
                "C_rmse",
            ],
        )
    )
    lines.append("")
    lines.append(
        "最高相似度分位数：" + json.dumps(quantiles, ensure_ascii=False, sort_keys=True)
    )
    lines.append(
        "空箱保留；阈值未搜索。覆盖子群上的局部优势不能替代完整任务结论。"
    )
    lines.append("")
    lines.append("## 参照计算账本（每任务 3 个 Val query，仅计算分解）")
    lines.append("")
    for row in case_rows:
        lines.append(
            f"### {row['dataset']} · {row['ledger_case']} · row_id={row['query_row_id']} "
            f"· SMILES={row['query_smiles']}"
        )
        lines.append("")
        lines.append(
            f"y={row['y_raw']:.6f}  pred_D={row['pred_D_raw']:.6f}  "
            f"pred_C={row['pred_C_raw']:.6f}  |err_C|={row['abs_error_C']:.6f}  "
            f"config={row['config']}  K_eff={row['k_eff']}"
        )
        lines.append("")
        lines.append(
            _table(
                row["top5"],
                [
                    "slot",
                    "ref_row_id",
                    "ref_smiles",
                    "similarity",
                    "weight",
                    "y_ref_raw",
                    "delta_raw",
                    "y_ref_plus_delta",
                ],
            )
        )
        lines.append("")
        lines.append(
            f"其余参照加权和 = {row['remaining_weighted_sum']:.9f}；"
            f"Top5 加权和 = {row['top5_weighted_sum']:.9f}；"
            f"合计 = {row['ledger_total']:.9f}（与 pred_C 差 {row['prediction_abs_diff']:.2e}）"
        )
        lines.append("")
    lines.append("## 资源账本")
    lines.append("")
    lines.append(
        _table(
            resource_rows_all,
            [
                "dataset",
                "arm",
                "preset",
                "rounds",
                "train_seconds",
                "eval_seconds",
                "matrix_build_seconds",
                "seconds_per_round",
                "gpu_peak_mb",
                "model_sha256",
            ],
        )
    )
    lines.append("")
    total_gpu_seconds = sum(
        float(data[dataset]["summary"].get("wall_seconds_total", 0.0)) for dataset in datasets
    )
    lines.append(
        f"- 两 worker 墙钟合计 = {total_gpu_seconds:.1f} s = {total_gpu_seconds / 3600.0:.3f} GPU·h"
        f"（每个 worker 独占一张 GPU；两 worker 并行执行）。"
    )
    lines.append(
        "- peak RSS / GPU 峰值来自 worker 进程采样；每轮评价成本已与一次性 pair 矩阵构建分开。"
    )
    lines.append("")
    lines.append("## 数据与模型身份")
    lines.append("")
    identity_rows = []
    for dataset in datasets:
        audit = data[dataset]["audit"]
        identity_rows.append(
            {
                "dataset": dataset,
                "graph_cache_sha256": audit["cache_sha256"],
                "ecfp_sha256": audit["ecfp_sha256"],
                "split_sizes": str(audit["split_sizes"]),
                "split_hash_valid": audit["split_hash"]["valid"][:16],
                "reference_hash": data[dataset]["summary"]["reference_hash"],
                "pair_hash": data[dataset]["summary"]["pair_hash"],
                "pair_records": data[dataset]["pairs_json"]["stats"]["n_records"],
                "mu": audit["train_target_scaler"]["mean"],
                "sigma": audit["train_target_scaler"]["std"],
            }
        )
    lines.append(
        _table(
            identity_rows,
            [
                "dataset",
                "graph_cache_sha256",
                "ecfp_sha256",
                "split_sizes",
                "split_hash_valid",
                "reference_hash",
                "pair_hash",
                "pair_records",
                "mu",
                "sigma",
            ],
        )
    )
    lines.append("")
    lines.append("## 正确性检查（EXECUTION_PASS 的依据）")
    lines.append("")
    check_rows = []
    for dataset in datasets:
        entry = data[dataset]["summary"]["checks_ok"]
        check_rows.append(
            {
                "dataset": dataset,
                "max_abs_diff_D_raw": entry["max_abs_diff_D_raw"],
                "max_abs_margin_diff_C": entry["max_abs_margin_diff_C"],
                "max_abs_metric_diff": entry["max_abs_metric_diff"],
                "pass": entry["pass"],
            }
        )
    lines.append(
        _table(
            check_rows,
            [
                "dataset",
                "max_abs_diff_D_raw",
                "max_abs_margin_diff_C",
                "max_abs_metric_diff",
                "pass",
            ],
        )
    )
    lines.append("")
    lines.append("## 复用参照 G / L 的核验")
    lines.append("")
    ref_rows = []
    for dataset in datasets:
        for arm in REFERENCE_ARMS:
            ref = references[dataset][arm]
            ref_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "available": ref["available"],
                    "point_only": True,
                    "train_rmse": ref.get("train_rmse_raw"),
                    "valid_rmse": ref.get("valid_rmse_raw"),
                    "test_rmse": ref.get("test_rmse_raw"),
                    "recompute_abs_diff_valid": ref.get("valid_recompute_abs_diff"),
                    "recompute_abs_diff_test": ref.get("test_recompute_abs_diff"),
                    "source_commit": ref["source_commit"],
                    "sha256": ref.get("sha256"),
                }
            )
    lines.append(
        _table(
            ref_rows,
            [
                "dataset",
                "arm",
                "available",
                "point_only",
                "train_rmse",
                "valid_rmse",
                "test_rmse",
                "recompute_abs_diff_valid",
                "recompute_abs_diff_test",
                "source_commit",
                "sha256",
            ],
        )
    )
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(
        f"- 执行有效性：EXECUTION_PASS = {_fmt(decision['execution']['EXECUTION_PASS'])}；"
        f"参照轴：{decision['reference_axis']}。"
    )
    lines.append(
        f"- 联合差值信号：{decision['joint_signal']['classification']}"
        f"（strong_joint_tasks={decision['joint_signal']['strong_joint_tasks']}）。"
    )
    lines.append(
        f"- 实质进展：SUBSTANTIVE_PROGRESS = "
        f"{_fmt(decision['substantive_progress']['SUBSTANTIVE_PROGRESS'])}。"
    )
    lines.append(
        f"- 是否值得下一步设计结构比较器："
        f"{_fmt(decision['worth_structural_comparator_followup'])}。"
        "这只表示“值得进一步设计结构比较器”，不表示已获得结构解释、化学机制、"
        "组合泛化或第一梯队性能。"
    )
    lines.append("")
    lines.append(
        "本轮的 3%/5%/50% 是投入门槛，不是统计显著性界限；负结果只针对本配方。"
    )
    lines.append("")
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")
