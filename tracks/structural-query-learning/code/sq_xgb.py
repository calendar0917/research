"""Final gradient-boosted read-out (F / R / A / E) with per-round three-split curves.

Frozen final configuration (identical for F, R, A and E except the two depth/eta
presets that must both be run):

    objective=reg:squarederror  tree_method=hist  device=cuda:0
    num_boost_round=800  min_child_weight=5  reg_lambda=10  reg_alpha=0
    subsample=1  colsample_bytree=1  max_bin=64  base_score=0.0  seed=0
    nthread=4  early_stopping=None
    H1: max_depth=3, eta=0.05      H2: max_depth=5, eta=0.03

One boosting round is one epoch of this method (never compared to a neural
epoch as if the compute were equivalent).  Selection is *only* minimum Val RMSE
over the union of both presets and all rounds, ties broken by H1 first and then
by the earlier round.  The Test column of the same frozen state is reported
afterwards and never used to choose anything.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import sq_common as C
from sq_common import mae, rmse

NUM_BOOST_ROUND = 800
MIN_CHILD_WEIGHT = 5
REG_LAMBDA = 10.0
REG_ALPHA = 0.0
SUBSAMPLE = 1.0
COLSAMPLE = 1.0
MAX_BIN = 64
NTHREAD = 4

PRESETS: dict[str, dict[str, float]] = {
    "H1": {"max_depth": 3, "eta": 0.05},
    "H2": {"max_depth": 5, "eta": 0.03},
}
PRESET_ORDER: tuple[str, ...] = ("H1", "H2")


# ---------------------------------------------------------------------------
# device resolution (honest, probed)
# ---------------------------------------------------------------------------


def resolve_backend(requested: str = "cuda:0") -> tuple[str, dict[str, Any]]:
    """Return the device XGBoost *actually* uses, with the reason recorded."""
    import xgboost as xgb

    info = xgb.build_info()
    status: dict[str, Any] = {
        "requested": requested,
        "build_info_use_cuda": str(info.get("USE_CUDA")),
        "build_info_cuda_version": str(info.get("CUDA_VERSION")),
        "fallback_reason": None,
        "probed": True,
    }
    if not requested.startswith("cuda"):
        return requested, status
    if str(info.get("USE_CUDA")).lower() != "true":
        status["fallback_reason"] = "xgboost built without CUDA"
        return "cpu", status
    rng = np.random.default_rng(0)
    probe = xgb.DMatrix(
        rng.normal(size=(16, 4)).astype(np.float32), label=rng.normal(size=16)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            xgb.train(
                {
                    "objective": "reg:squarederror",
                    "tree_method": "hist",
                    "device": requested,
                    "nthread": 1,
                },
                probe,
                num_boost_round=1,
                verbose_eval=False,
            )
        except Exception as exc:  # noqa: BLE001 - a real failure is a CPU fallback
            status["fallback_reason"] = f"probe raised {type(exc).__name__}: {exc}"
            return "cpu", status
        messages = [str(item.message) for item in caught]
    for message in messages:
        lowered = message.lower()
        if "setting device to cpu" in lowered or "not compiled with cuda" in lowered or "no visible gpu" in lowered:
            status["fallback_reason"] = message.splitlines()[-1][:200]
            status["probe_warnings"] = [m[:200] for m in messages]
            return "cpu", status
    status["probe_warnings"] = [m[:200] for m in messages]
    return requested, status


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def final_params(preset: str, device: str) -> dict[str, Any]:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}")
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": device,
        "max_depth": int(PRESETS[preset]["max_depth"]),
        "eta": float(PRESETS[preset]["eta"]),
        "min_child_weight": MIN_CHILD_WEIGHT,
        "reg_lambda": REG_LAMBDA,
        "reg_alpha": REG_ALPHA,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
        "max_bin": MAX_BIN,
        "base_score": 0.0,
        "seed": C.TRAIN_SEED,
        "nthread": NTHREAD,
        "eval_metric": ["rmse", "mae"],
    }


@dataclass
class HeadResult:
    method: str
    preset: str
    booster: Any
    feature_names: list[str]
    curves: list[dict[str, Any]]
    best: dict[str, Any]
    final: dict[str, Any]
    last20: dict[str, Any]
    seconds: float
    predictions: dict[str, dict[str, np.ndarray]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "preset": self.preset,
            "best": self.best,
            "final": self.final,
            "last20": self.last20,
            "seconds": self.seconds,
            "n_features": len(self.feature_names),
            "n_trees": int(self.booster.num_boosted_rounds()),
        }


def _raw_predictions(
    booster, dmats: Mapping[str, Any], scaler: Mapping[str, float], *, rounds: int | None
) -> dict[str, np.ndarray]:
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    out: dict[str, np.ndarray] = {}
    for split, dmat in dmats.items():
        if rounds is None:
            pred_std = booster.predict(dmat)
        else:
            pred_std = booster.predict(dmat, iteration_range=(0, int(rounds)))
        out[split] = np.asarray(pred_std, dtype=np.float64) * std + mean
    return out


def run_head(
    method: str,
    preset: str,
    matrices: Mapping[str, np.ndarray],
    targets_raw: Mapping[str, np.ndarray],
    scaler: Mapping[str, float],
    feature_names: Sequence[str],
    *,
    device: str,
    num_boost_round: int = NUM_BOOST_ROUND,
) -> HeadResult:
    """Fit one final head; record every round on all three splits."""
    import xgboost as xgb

    mean = float(scaler["mean"])
    std = float(scaler["std"])
    names = list(feature_names)
    dmats = {}
    targets_std = {}
    for split in ("train", "valid", "test"):
        y_raw = np.asarray(targets_raw[split], dtype=np.float64)
        y_std = (y_raw - mean) / std
        targets_std[split] = y_std
        dmats[split] = xgb.DMatrix(matrices[split], label=y_std, feature_names=names)
    evals = [(dmats[split], split) for split in ("train", "valid", "test")]
    evals_result: dict[str, dict[str, list[float]]] = {}
    t0 = time.perf_counter()
    booster = xgb.train(
        final_params(preset, device),
        dmats["train"],
        num_boost_round=int(num_boost_round),
        evals=evals,
        evals_result=evals_result,
        verbose_eval=False,
        early_stopping_rounds=None,
    )
    seconds = time.perf_counter() - t0

    curves: list[dict[str, Any]] = []
    n_rounds = int(booster.num_boosted_rounds())
    for index in range(n_rounds):
        for split in ("train", "valid", "test"):
            curves.append(
                {
                    "method": method,
                    "preset": preset,
                    "iteration": index + 1,
                    "split": split,
                    "rmse_raw": float(evals_result[split]["rmse"][index]) * std,
                    "mae_raw": float(evals_result[split]["mae"][index]) * std,
                }
            )

    def _best_from(curves_rows: list[dict[str, Any]], use_final_round: bool) -> dict[str, Any]:
        if use_final_round:
            rows = [row for row in curves_rows if row["iteration"] == n_rounds]
        else:
            rows = curves_rows
        split_rows = {
            split: [row for row in rows if row["split"] == split] for split in ("train", "valid", "test")
        }
        if use_final_round:
            chosen = split_rows["valid"][0]
        else:
            # Strictly minimum Val RMSE; ties resolved by the earlier round.
            chosen = min(
                split_rows["valid"], key=lambda row: (row["rmse_raw"], row["iteration"])
            )
        iteration = int(chosen["iteration"])
        predictions = _raw_predictions(booster, dmats, scaler, rounds=iteration)
        state = {
            "iteration": iteration,
            "iteration_index0": iteration - 1,
            "preset": preset,
            "train_rmse_raw": split_rows["train"][iteration - 1 if not use_final_round else 0]["rmse_raw"],
            "valid_rmse_raw": chosen["rmse_raw"],
            "test_rmse_raw": split_rows["test"][iteration - 1 if not use_final_round else 0]["rmse_raw"],
            "train_mae_raw": split_rows["train"][iteration - 1 if not use_final_round else 0]["mae_raw"],
            "valid_mae_raw": split_rows["valid"][iteration - 1 if not use_final_round else 0]["mae_raw"],
            "test_mae_raw": split_rows["test"][iteration - 1 if not use_final_round else 0]["mae_raw"],
        }
        # Recompute raw metrics from the frozen prediction arrays (unit cross-check).
        for split in ("train", "valid", "test"):
            state[f"{split}_rmse_raw_recomputed"] = rmse(predictions[split], targets_raw[split])
            state[f"{split}_mae_raw_recomputed"] = mae(predictions[split], targets_raw[split])
        return state, predictions

    best, best_predictions = _best_from(curves, use_final_round=False)
    final, final_predictions = _best_from(curves, use_final_round=True)
    last20 = {
        f"mean_last20_{split}_rmse_raw": float(
            np.mean([row["rmse_raw"] for row in curves if row["split"] == split][-20:])
        )
        for split in ("train", "valid", "test")
    }
    last20.update(
        {
            f"mean_last20_{split}_mae_raw": float(
                np.mean([row["mae_raw"] for row in curves if row["split"] == split][-20:])
            )
            for split in ("train", "valid", "test")
        }
    )
    return HeadResult(
        method=method,
        preset=preset,
        booster=booster,
        feature_names=names,
        curves=curves,
        best=best,
        final=final,
        last20=last20,
        seconds=seconds,
        predictions=best_predictions,
    )


def selection_key(results: Sequence[HeadResult]) -> HeadResult:
    """Minimum Val RMSE over both presets; ties -> H1 first, then earlier round."""
    def key(result: HeadResult):
        preset_rank = PRESET_ORDER.index(result.preset)
        return (result.best["valid_rmse_raw"], preset_rank, result.best["iteration"])

    return min(results, key=key)


def run_method(
    method: str,
    matrices: Mapping[str, np.ndarray],
    targets_raw: Mapping[str, np.ndarray],
    scaler: Mapping[str, float],
    feature_names: Sequence[str],
    *,
    device: str,
    presets: Sequence[str] = PRESET_ORDER,
    num_boost_round: int = NUM_BOOST_ROUND,
) -> dict[str, Any]:
    """Run every preset for one method and return the Val-selected state."""
    results = [
        run_head(
            method,
            preset,
            matrices,
            targets_raw,
            scaler,
            feature_names,
            device=device,
            num_boost_round=num_boost_round,
        )
        for preset in presets
    ]
    chosen = selection_key(results)
    return {
        "method": method,
        "selected_preset": chosen.preset,
        "selected": chosen.best,
        "selected_predictions": chosen.predictions,
        "selected_booster": chosen.booster,
        "feature_names": chosen.feature_names,
        "curves": [row for result in results for row in result.curves],
        "per_preset": {result.preset: result.to_dict() for result in results},
        "head_seconds_total": float(sum(result.seconds for result in results)),
        "last20_selected": chosen.last20,
        "final_round_selected": chosen.final,
        "selected_rows": chosen.predictions,
    }


# ---------------------------------------------------------------------------
# native export / replay
# ---------------------------------------------------------------------------


def export_trees(booster, feature_names: Sequence[str], *, rounds: int) -> dict[str, Any]:
    """Exact tree structure of the first ``rounds`` boosting rounds.

    Split thresholds and leaf values are cast back to float32 (the C++ storage
    width) so that a replay compares exactly the same numbers XGBoost does.

    Values are taken from the model JSON (float32-exact), not from the
    human-readable text dump, so a native replay can match to ~1e-7.  Leaf
    values already contain the shrinkage factor: never multiply by eta again.
    """
    raw = json.loads(bytes(booster.save_raw(raw_format="json")))
    model = raw["learner"]["gradient_booster"]["model"]
    trees = model["trees"][: int(rounds)]
    if len(trees) != int(rounds):
        raise ValueError(f"model exposes {len(trees)} trees, requested {rounds}")
    exported: list[dict[str, Any]] = []
    for tree in trees:
        exported.append(
            {
                "left": [int(v) for v in tree["left_children"]],
                "right": [int(v) for v in tree["right_children"]],
                "split_index": [int(v) for v in tree["split_indices"]],
                "split_condition": [float(v) for v in np.asarray(tree["split_conditions"], dtype=np.float32)],
                "default_left": [int(v) for v in tree["default_left"]],
                "split_type": [int(v) for v in tree["split_type"]],
            }
        )
    payload = {
        "format": "sq-native-gbtree-v1",
        "n_trees": len(exported),
        "feature_names": list(feature_names),
        "base_score": 0.0,
        "trees": exported,
        "note": (
            "leaf values already include shrinkage; base_score is 0.0; "
            "numeric splits use value < split_condition, missing goes to default_left"
        ),
    }
    return payload


def replay_native(
    trees_payload: Mapping[str, Any],
    design: np.ndarray,
    feature_names: Sequence[str],
    scaler: Mapping[str, float],
) -> np.ndarray:
    """Execute the exported trees exactly (standardized margin -> raw units)."""
    names = list(trees_payload["feature_names"])
    if list(feature_names) != names:
        raise ValueError("feature name mismatch between export and matrix")
    n_rows = int(design.shape[0])
    index_of = {name: i for i, name in enumerate(names)}
    margin = np.zeros(n_rows, dtype=np.float64)
    for tree in trees_payload["trees"]:
        left = np.asarray(tree["left"], dtype=np.int64)
        right = np.asarray(tree["right"], dtype=np.int64)
        split_index = np.asarray(tree["split_index"], dtype=np.int64)
        split_condition = np.asarray(tree["split_condition"], dtype=np.float64)
        default_left = np.asarray(tree["default_left"], dtype=np.int64)
        node = np.zeros(n_rows, dtype=np.int64)
        active = np.ones(n_rows, dtype=bool)
        while active.any():
            rows = np.nonzero(active)[0]
            idx = node[rows]
            is_leaf = left[idx] == -1
            # leaves stop here; their value is added after the loop
            active[rows[is_leaf]] = False
            move = ~is_leaf
            if not move.any():
                continue
            rows_m = rows[move]
            idx_m = idx[move]
            columns = split_index[idx_m]
            values = np.zeros(rows_m.size, dtype=np.float64)
            for column in np.unique(columns):
                selected = columns == column
                values[selected] = design[rows_m[selected], index_of[names[int(column)]]]
            # same width and operator as XGBoost's C++ traversal
            go_left = values.astype(np.float32) < split_condition.astype(np.float32)[idx_m]
            missing = ~np.isfinite(values)
            if missing.any():
                go_left = np.where(missing, default_left[idx_m] == 1, go_left)
            node[rows_m] = np.where(go_left, left[idx_m], right[idx_m])
        margin += split_condition[node]
    return margin * float(scaler["std"]) + float(scaler["mean"])


def tree_paths(
    trees_payload: Mapping[str, Any],
    design_row: Mapping[str, float],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Exact decision path of every tree for one molecule (for the case studies)."""
    paths: list[dict[str, Any]] = []
    total = 0.0
    for tree_id, tree in enumerate(trees_payload["trees"]):
        node = 0
        steps: list[dict[str, Any]] = []
        while tree["left"][node] != -1:
            col = int(tree["split_index"][node])
            name = feature_names[col]
            value = float(design_row[name])
            go_left = float(np.float32(value)) < float(np.float32(tree["split_condition"][node]))
            steps.append(
                {
                    "node": node,
                    "feature": name,
                    "value": value,
                    "threshold": float(tree["split_condition"][node]),
                    "direction": "left" if go_left else "right",
                }
            )
            node = tree["left"][node] if go_left else tree["right"][node]
        leaf_value = float(tree["split_condition"][node])
        total += leaf_value
        paths.append(
            {
                "tree": tree_id,
                "steps": steps,
                "path_length": len(steps),
                "leaf_node": node,
                "leaf_value_std": leaf_value,
            }
        )
    return {"trees": paths, "margin_std": total, "total_path_length": sum(p["path_length"] for p in paths)}
