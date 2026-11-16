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
    cost_path = base / "inference_cost.json"
    cost = C.read_json(cost_path) if cost_path.is_file() else None
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
        "cost": cost,
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
        return []  # a constant predictor has a single state, not a curve
    key = f"{'D' if arm in ('D', 'R') else 'C'}_{preset}"
    rows = data["curves"].get(key, [])
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
        # lexsort ascending on key: most negative change first = largest improvement
        ("largest_improvement_vs_D", err_c - err_d),
        ("largest_regression_vs_D", -(err_c - err_d)),
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
                "trained_at": info.get("trained_at"),
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
                "trained_at": info.get("trained_at"),
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
        "本文件由 `stage=aggregate` 依据两个 worker 的产物生成；"
        f"评价修订为 `{C.EVALUATION_REVISION}`。"
    )
    lines.append("")
    lines.append(
        "> **评价纠错（rd-evalfix-v1）。** 原 C 评价在 Val/Test 上把当前 split 的"
        " query 局部行号误当成 Train 行号读取特征；参照与标签却是真正的 Val/Test。"
        "因此旧 C 的 Val/Test 曲线、选中状态、预测、比较、案例与路线判定无效，"
        "已在 **不重训任何 booster** 的前提下重算。旧报告中的“C 只能选第 1 轮”"
        "“越训练越不泛化”“NO_JOINT_INCREMENT”等表述已撤回。"
        "根因、撤回范围与模型 SHA 见 `CORRIGENDUM_evalfix_v1.md`。"
    )
    lines.append("")
    lines.append("## 问题与四条臂")
    lines.append("")
    lines.append(
        "在**静态分子输入完全相同**（同一 2059 维 ECFP4+描述符矩阵、同一官方划分、"
        "同一 Train 目标标准化）的前提下："
    )
    lines.append("")
    lines.append(
        "**联合学习两个分子的性质差**（推理时用 Train 参照还原成绝对值），"
        "是否优于直接绝对回归、Train 邻居标签直接预测、以及“直接模型 + 邻居残差修正”？"
    )
    lines.append("")
    lines.append(
        "| 臂 | 定义 | 训练 |\n| --- | --- | --- |\n"
        "| **D** | `D_t(G) = mu + sigma * f_t(x_G)` | 800 轮 |\n"
        "| **N** | `N(G) = sum_j w_j y_j` | 无 |\n"
        "| **C** | `delta_t(G,R) = sigma/2 [h_t(z(G,R)) - h_t(z(R,G))]`，`C_t(G) = sum_j w_j (y_j + delta_t(G,R_j))` | 1600 轮 |\n"
        "| **R** | `R_t(G) = D_t(G) + sum_j w_j (y_j - D_t(R_j))` | 复用 D |"
    )
    lines.append("")
    lines.append(
        "`z(i,j) = [x_i, x_j, x_i - x_j]`（6177 维）；参照一律来自 Train，"
        "按 Tanimoto 取 Top-64，排除自身与完全相同 SMILES 的行；"
        "`w_j` 由冻结的 6 个 (K, 权重) 配置之一给出（`K ∈ {4,16,64}`，uniform 或 softmax τ=0.1）。"
        "C 的反对称化是**推理定义**的一部分（`delta(G,R) = -delta(R,G)`），不是事后修补；"
        "`delta` 是原始单位差值，乘 `sigma` 之后**不再加 `mu`**。"
    )
    lines.append("")
    lines.append(
        "判定（投入门槛，不是统计显著性界限）：C 需在 **Val 与 Test 同时** 比 D、N、R 各好 ≥3% "
        "才算 `strong_joint_task`；另一任务不得差于任何一条臂超过 5%；"
        "`SUBSTANTIVE_PROGRESS` 要求两个 split 都满足 `c <= 1.05 b` 或关闭一半差距；"
        "`COMPETITIVE` 要求 C 的 Val 与 Test 都不超过最优可核验参照 B 的 105%。"
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
    lines.append("## C 的曲线与拟合层级")
    lines.append("")
    lines.extend(mechanism_lines(datasets, data, main_rows, decision))
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
    lines.append("## 单分子推理成本（描述性）")
    lines.append("")
    cost_rows = []
    for dataset in datasets:
        cost = data[dataset]["cost"]
        if cost is None:
            continue
        for arm, entry in cost["arms"].items():
            cost_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "device": entry["device"],
                    "references_per_query": entry["references_per_query"],
                    "us_per_molecule": round(1000.0 * float(entry["ms_per_molecule"]), 3),
                    "molecules_timed": entry["molecules"],
                    "config": entry.get("config"),
                }
            )
    if cost_rows:
        lines.append(
            _table(
                cost_rows,
                [
                    "dataset",
                    "arm",
                    "device",
                    "references_per_query",
                    "us_per_molecule",
                    "molecules_timed",
                    "config",
                ],
            )
        )
        lines.append("")
        one_off = []
        for dataset in datasets:
            cost = data[dataset]["cost"]
            if cost is None:
                continue
            one_off.append(
                {
                    "dataset": dataset,
                    "design_matrix_seconds": cost["one_off"]["design_matrix_seconds"],
                    "retrieval_table_seconds": cost["one_off"]["retrieval_table_seconds"],
                    "pair_list_seconds": cost["one_off"]["pair_list_seconds"],
                    "pair_matrix_seconds": cost["one_off"]["pair_matrix_seconds_for_selected_C"],
                    "pair_matrix_rows": cost["one_off"]["pair_matrix_rows"],
                }
            )
        lines.append(
            _table(
                one_off,
                [
                    "dataset",
                    "design_matrix_seconds",
                    "retrieval_table_seconds",
                    "pair_list_seconds",
                    "pair_matrix_seconds",
                    "pair_matrix_rows",
                ],
            )
        )
        lines.append("")
        lines.append(
            "定义：`us_per_molecule` = 用选中模型做一次预测的中位耗时 / 分子数"
            "（CPU，nthread=4，重载后的 booster，无 batching/调优）；`references_per_query` 是"
            "该臂选中配置的 `K_eff`。D 不需要参照；N 只做加权求和；R 行只计“取参照 + 加权残差”"
            "这一步（锚点模型 D 的前向已单列在 D 行，实际 R 推理 ≈ D + R 两行之和）；"
            "C 需要正反两向的 6177 维 pair 特征与两次树遍历，因此成本最高，"
            "且只在 Test split 上计时（pair 矩阵是最贵的部分），其余臂覆盖三个 split。"
            "一次性成本里，`retrieval_table_seconds` 覆盖三个 split 的 Top-64 全表，"
            "`pair_list_seconds` 是一次性冻结配对，`pair_matrix_seconds` 是 C 在 Test 上"
            "构建选中配置 pair 特征的时间（这些都不进入每轮评测）。"
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
                "trained_at",
                "model_sha256",
            ],
        )
    )
    lines.append("")
    compute_seconds = sum(
        float(row["train_seconds"]) + float(row["eval_seconds"]) + float(row["matrix_build_seconds"])
        for row in resource_rows_all
    )
    wall_seconds = sum(
        float(data[dataset]["summary"].get("wall_seconds_total", 0.0)) for dataset in datasets
    )
    eval_fix_wall = sum(
        float(data[dataset]["summary"].get("evalfix", {}).get("wall_seconds", 0.0))
        for dataset in datasets
    )
    peak_rss = {
        dataset: data[dataset]["summary"].get("peak_rss_mb") for dataset in datasets
    }
    lines.append(
        f"- 轨迹算力合计（训练 + 每轮三集评测 + 一次性 pair 矩阵构建）= {compute_seconds:.1f} s "
        f"= {compute_seconds / 3600.0:.3f} GPU·h；两个 worker 各独占一张 A100，并行执行。"
    )
    lines.append(
        f"- 两个 worker 训练时最后一次执行的墙钟合计 = {wall_seconds:.1f} s"
        f"（每 GPU 一个训练 worker；本轮的启动失败与 resume 重跑另计，均记录在 `rr jobs`）。"
    )
    lines.append(
        f"- 本次 `rd-evalfix-v1` 新增评价墙钟（两任务合计）= {eval_fix_wall:.1f} s；"
        "只加载既有 booster，CPU 评价，新增训练 0 条、新增 GPU 计算 = 0；"
        f"按独占分配计约 {eval_fix_wall / 3600.0:.3f} GPU·h（GPU 空闲等待）。"
    )
    lines.append(
        "- worker 进程 peak RSS："
        + "，".join(f"{dataset} {peak_rss[dataset]:.0f} MB" for dataset in datasets)
        + "。GPU 峰值是 nvidia-smi 采样的整卡占用，只作上界参考。"
    )
    lines.append(
        "- 预算：soft 3 GPU·h / hard 6 GPU·h（含 smoke 与重跑）；本轮轨迹算力远低于 soft 预算。"
    )
    ratios = []
    for row in resource_rows_all:
        if row["arm"] == "C" and float(row["train_seconds"]) > 0:
            ratios.append(
                f"{row['dataset']}/{row['preset']} {float(row['eval_seconds']) / float(row['train_seconds']):.2f}x"
            )
    lines.append(
        "- 每轮评价成本已与一次性 pair 矩阵构建分开记录；本次修正后 C 的评测/训练时间比 = "
        + "，".join(ratios)
        + "。这是“每轮完整三集评测”的直接代价（K 越大、查询越多，代价越高）。"
    )
    lines.append("")
    lines.append("## 评价纠错与重跑范围（rd-evalfix-v1）")
    lines.append("")
    lines.append(
        "- 根因：`pair_matrix` / `pair_feature_blocks` 原实现固定读取 `task.x[\"train\"]`。"
        "Train–Train 训练对因此正确，但 C 的 Val/Test 评价 query 也被映射到 Train 矩阵；"
        "参照表、权重和评价标签仍对应真正的 Val/Test query，取数身份不一致。"
    )
    lines.append(
        "- 修复：两个构造函数改为 keyword-only 的显式 `query_x` / `reference_x` 两个输入域；"
        "训练对两边都传 Train，评价 query 传当前 split、参照传 Train。反向对仍由"
        "`BlockLayout(reverse=True)` 在正确构造的正向块上交换 X1/X2 并对 X3 取负。"
    )
    lines.append(
        "- 撤回：旧 C 的 Val/Test 曲线、Val-selected preset/round/config、`pred_C.npz`、"
        "`pred_C_delta.npz`、C-D/C-N/C-R 比较、paired bootstrap、drop-one、参照覆盖中的 C 误差、"
        "三个 Val 案例账本，以及由旧 C 状态支撑的“C 退化/第 1 轮/NO_JOINT_INCREMENT”结论。"
    )
    lines.append(
        "- 保留：八个 D/C booster、D/N/R 的选中状态与预测、G/L 点参照、参照表、"
        "训练 pair、样本权重和六种推理配置均未改变；本次新增训练 0 条。"
    )
    lines.append(
        "- 重算：从两个既有 C booster 各提取全部 1600 棵树，逐前缀重算三 split × 六配置的"
        "分子级 RMSE/MAE；Train 分子曲线与训练 pair 曲线在相同状态下的修复前后差均为 0。"
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


def _curve_direction(first: float, last: float) -> str:
    if last < first - 1e-12:
        return "下降"
    if last > first + 1e-12:
        return "上升"
    return "基本持平"


def mechanism_lines(
    datasets: Sequence[str],
    data: Mapping[str, Any],
    main_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> list[str]:
    """Descriptive curve reading of the frozen trajectories (never a selection)."""
    lines: list[str] = []
    for dataset in datasets:
        state = data[dataset]["selection"]["C"]
        preset = str(state["preset"])
        config = str(state["config"])
        curves = data[dataset]["curves"].get(f"C_{preset}", [])
        rows = [row for row in curves if row["method"] == "C" and row["config"] == config]
        if not rows:
            continue
        last_round = max(int(row["round"]) for row in rows)

        def value(split: str, round_index: int) -> float | None:
            for row in rows:
                if row["split"] == split and int(row["round"]) == round_index:
                    return float(row["rmse_raw"])
            return None

        pair_curves = data[dataset]["pair_curves"].get(preset, [])
        pair_at = {int(row["round"]): float(row["pair_rmse_std"]) for row in pair_curves}
        n_state = {row["arm"]: row for row in main_rows if row["dataset"] == dataset}["N"]
        d_state = {row["arm"]: row for row in main_rows if row["dataset"] == dataset}["D"]
        c_selected_round = int(state["round"])
        lines.append(
            f"**{dataset}**（C-{preset}，配置 `{config}`，选中轮 = {c_selected_round} / 共 {last_round} 轮）"
        )
        lines.append("")
        lines.append(
            "- 训练 pair 加权 RMSE（标准化目标）："
            f"第 1 轮 {pair_at.get(1, float('nan')):.4f} → 第 {last_round} 轮 "
            f"{pair_at.get(last_round, float('nan')):.4f}"
        )
        directions: dict[str, str] = {}
        best_valid_round = None
        best_valid_value = None
        for split, label in (("train", "分子 Train"), ("valid", "Val"), ("test", "Test")):
            first = value(split, 1)
            last = value(split, last_round)
            best_row = min(
                (row for row in rows if row["split"] == split),
                key=lambda row: (float(row["rmse_raw"]), int(row["round"])),
            )
            if first is None or last is None:
                continue
            directions[split] = _curve_direction(first, last)
            if split == "valid":
                best_valid_round = int(best_row["round"])
                best_valid_value = float(best_row["rmse_raw"])
            lines.append(
                f"- {label} RMSE：第 1 轮 {first:.4f} → 第 {last_round} 轮 {last:.4f}"
                f"（{directions[split]}；该 split 全曲线最好 {float(best_row['rmse_raw']):.4f}，"
                f"第 {int(best_row['round'])} 轮）"
            )
        lines.append(
            f"- 对照：N 的 Val/Test = {n_state['valid_rmse']:.4f}/{n_state['test_rmse']:.4f}，"
            f"D 的 Val/Test = {d_state['valid_rmse']:.4f}/{d_state['test_rmse']:.4f}"
        )
        beats = {}
        for split in ("valid", "test"):
            per_round = sorted(
                (int(row["round"]), float(row["rmse_raw"]))
                for row in rows
                if row["split"] == split
            )
            baseline = float(n_state[f"{split}_rmse"])
            wins = [round_index for round_index, value in per_round if value < baseline]
            beats[split] = (len(wins), len(per_round), wins[:3])
        lines.append(
            f"- 与自身基线 N 相比：Val 上 C 在 {beats['valid'][1]} 轮中有 {beats['valid'][0]} 轮优于 N"
            f"（N = {n_state['valid_rmse']:.4f}），Test 上 {beats['test'][0]}/{beats['test'][1]} 轮"
            f"（N = {n_state['test_rmse']:.4f}）"
        )
        selected_valid = value("valid", c_selected_round)
        selected_test = value("test", c_selected_round)
        lines.append(
            f"- 读法：分子 Train 曲线{directions.get('train', '未知')}，"
            f"Val {directions.get('valid', '未知')}，Test {directions.get('test', '未知')}；"
            f"Val 全曲线最小 RMSE 在第 {best_valid_round} 轮（{best_valid_value:.4f}），"
            f"Val-selected 状态是第 {c_selected_round} 轮"
            f"（Val/Test {selected_valid:.4f}/{selected_test:.4f}）。"
            "这是固定配方下的曲线描述；选中仍只由完整 Val RMSE 决定。"
        )
        lines.append("")
    rows = []
    for dataset in datasets:
        entry = {row["arm"]: row for row in main_rows if row["dataset"] == dataset}
        rows.append(
            {
                "dataset": dataset,
                "D_valid": entry["D"]["valid_rmse"],
                "R_valid": entry["R"]["valid_rmse"],
                "gain_valid": 1.0 - entry["R"]["valid_rmse"] / entry["D"]["valid_rmse"],
                "D_test": entry["D"]["test_rmse"],
                "R_test": entry["R"]["test_rmse"],
                "gain_test": 1.0 - entry["R"]["test_rmse"] / entry["D"]["test_rmse"],
            }
        )
    lines.append("### 同一参照还原机制的对照：R vs D")
    lines.append("")
    lines.append(_table(rows, ["dataset", "D_valid", "R_valid", "gain_valid", "D_test", "R_test", "gain_test"]))
    lines.append("")
    improved = [
        f"{row['dataset']} Val {100 * row['gain_valid']:+.1f}% / Test {100 * row['gain_test']:+.1f}%"
        for row in rows
    ]
    worse_test = [row["dataset"] for row in rows if row["gain_test"] < 0.0]
    lines.append(
        "R 与 C 共用同一套 Tanimoto 参照、同一批权重配置和同一还原公式；"
        "差别只在 R 的锚点是已训练的绝对模型 D，而 C 的差值 h 是从 Train-Train 对里从零学的。"
        "R 相对 D 的增益：" + "；".join(improved) + "。"
        + (f"Test 上 R 低于 D 的任务：{', '.join(worse_test)}。" if worse_test else "Test 上 R 均不差于 D。")
        + "该对照是描述性的，不参与冻结判定。"
    )
    return lines
