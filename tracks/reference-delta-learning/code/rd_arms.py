"""The four arms D / N / C / R under the single frozen recipe.

D: direct absolute regression ``x -> (y - mu) / sigma`` (sample weight 1).
N: no training; ``N(G) = sum_j w_j y_j`` on the frozen Train references.
C: one XGBoost on the weighted, bidirectional pair records
   ``z(i,j) = [x_i, x_j, x_i - x_j]`` with target ``(y_i - y_j)/sigma``;
   inference is *antisymmetrised by construction*:
   ``delta(G,R) = sigma/2 [h(z(G,R)) - h(z(R,G))]`` and
   ``C(G) = sum_j w_j (y_j + delta(G,R_j))``.
R: the D model plus a Train-reference residual correction, reusing the same six
   retrieval configurations.

Every arm records all three splits at every boosting round in raw units.  Only
Val RMSE ever selects a state; Test is evaluated and stored but never used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

import rd_common as C
import rd_data as D
import rd_eval as E
import rd_pairs as P
import rd_refs as R
import rd_tree as T


# ---------------------------------------------------------------------------
# shared preparation
# ---------------------------------------------------------------------------


@dataclass
class ArmContext:
    """Everything the arms need that is label-free except the Train labels."""

    task: D.TaskData
    ref_table: dict[str, Any]
    weights: dict[str, dict[str, np.ndarray]]
    y_ref: dict[str, np.ndarray]
    mu: float
    sigma: float

    @classmethod
    def build(cls, task: D.TaskData, ref_table: dict[str, Any]) -> "ArmContext":
        weights = {split: R.config_weights_by_name(ref_table, split) for split in C.SPLITS}
        y_ref = {
            split: E.reference_targets(task, ref_table["splits"][split]["ref_local"])
            for split in C.SPLITS
        }
        return cls(
            task=task,
            ref_table=ref_table,
            weights=weights,
            y_ref=y_ref,
            mu=float(task.scaler["mean"]),
            sigma=float(task.scaler["std"]),
        )


def _ref_local(ctx: ArmContext, split: str) -> np.ndarray:
    return np.asarray(ctx.ref_table["splits"][split]["ref_local"], dtype=np.int64)


def _curve_row(
    dataset: str,
    method: str,
    preset: str,
    round_index: int,
    split: str,
    config: str,
    rmse_raw: float,
    mae_raw: float,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "preset": preset,
        "round": int(round_index),
        "split": split,
        "config": config,
        "rmse_raw": float(rmse_raw),
        "mae_raw": float(mae_raw),
    }


# ---------------------------------------------------------------------------
# D + R
# ---------------------------------------------------------------------------


@dataclass
class DRun:
    preset: str
    booster: Any
    trees: list[dict[str, np.ndarray]]
    curves: list[dict[str, Any]]
    final_accumulator: np.ndarray
    final_predictions: dict[str, np.ndarray]
    seconds: float
    trained_rounds: int
    eval_seconds: float = 0.0
    checkpoints: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def best(self) -> dict[str, Any]:
        rows = [row for row in self.curves if row["split"] == "valid" and row["config"] == "none"]
        return min(rows, key=lambda row: (row["rmse_raw"], row["round"]))

    def curve_index(self) -> dict[tuple[int, str, str], dict[str, Any]]:
        return {(row["round"], row["split"], row["config"]): row for row in self.curves}


def run_d(
    ctx: ArmContext,
    *,
    preset: str,
    device: str,
    rounds: int,
    verbose: bool = False,
) -> DRun:
    import xgboost as xgb

    task = ctx.task
    names = list(task.feature_names)
    y_std = task.standardise(task.y["train"]).astype(np.float32)
    dtrain = xgb.DMatrix(
        task.x["train"], label=y_std, weight=np.ones(task.n("train"), dtype=np.float32),
        feature_names=names,
    )
    params = C.SQX.final_params(preset, device)
    t0 = time.perf_counter()
    booster = xgb.train(params, dtrain, num_boost_round=int(rounds), verbose_eval=verbose)
    seconds = time.perf_counter() - t0
    trees = T.extract_trees(booster, names, rounds=int(rounds))
    t_eval = time.perf_counter()
    result = _evaluate_d(ctx, preset, booster, trees, seconds)
    result.eval_seconds = time.perf_counter() - t_eval
    return result


def _evaluate_d(
    ctx: ArmContext,
    preset: str,
    booster: Any,
    trees: Sequence[dict[str, np.ndarray]],
    seconds: float,
) -> DRun:
    task = ctx.task
    splits = C.SPLITS
    x_all = task.x_concat(splits)
    n_all = int(x_all.shape[0])
    offsets = {}
    cursor = 0
    for split in splits:
        offsets[split] = (cursor, cursor + task.n(split))
        cursor += task.n(split)
    layout = T.DenseLayout(x_all)

    ctx_split = {}
    for split in splits:
        ctx_split[split] = {
            "ref": _ref_local(ctx, split),
            "y_ref": ctx.y_ref[split],
            "weights": ctx.weights[split],
            "slice": offsets[split],
        }

    acc = np.zeros(n_all, dtype=np.float32)
    curves: list[dict[str, Any]] = []
    checkpoints: dict[int, np.ndarray] = {}
    mu, sigma = ctx.mu, ctx.sigma
    for index, tree in enumerate(trees, start=1):
        np.add(acc, T.tree_leaf_values(tree, layout, n_all), out=acc)
        if index in (1, 2, 3):
            checkpoints[index] = acc.copy()
        d_all = mu + sigma * acc.astype(np.float64)
        for split in splits:
            start, stop = ctx_split[split]["slice"]
            d_q = d_all[start:stop]
            rmse_raw, mae_raw = E.rmse_mae(d_q, task.y[split])
            curves.append(_curve_row(task.dataset, "D", preset, index, split, "none", rmse_raw, mae_raw))
            ref = ctx_split[split]["ref"]
            d_ref = d_all[ref]
            y_ref = ctx_split[split]["y_ref"]
            residual = y_ref - d_ref
            for config in C.CONFIGS:
                weights = ctx_split[split]["weights"][config["name"]]
                pred = d_q + np.sum(np.asarray(weights, dtype=np.float64) * residual, axis=1)
                rmse_r, mae_r = E.rmse_mae(pred, task.y[split])
                curves.append(
                    _curve_row(task.dataset, "R", preset, index, split, config["name"], rmse_r, mae_r)
                )
    final_predictions = {}
    for split in splits:
        start, stop = ctx_split[split]["slice"]
        final_predictions[split] = mu + sigma * acc[start:stop].astype(np.float64)
    return DRun(
        preset=preset,
        booster=booster,
        trees=list(trees),
        curves=curves,
        final_accumulator=acc,
        final_predictions=final_predictions,
        seconds=float(seconds),
        trained_rounds=len(trees),
        checkpoints=checkpoints,
    )


def predict_d_full(
    ctx: ArmContext, booster: Any, trees: Sequence[dict[str, np.ndarray]], rounds: int
) -> dict[str, np.ndarray]:
    """Full-prefix D prediction from scratch (independent of the per-round loop)."""
    task = ctx.task
    x_all = task.x_concat(C.SPLITS)
    acc = np.zeros(int(x_all.shape[0]), dtype=np.float32)
    layout = T.DenseLayout(x_all)
    T.accumulate(acc, trees, layout, stop=int(rounds))
    d_all = ctx.mu + ctx.sigma * acc.astype(np.float64)
    out: dict[str, np.ndarray] = {}
    cursor = 0
    for split in C.SPLITS:
        out[split] = d_all[cursor:cursor + task.n(split)]
        cursor += task.n(split)
    return out


def d_r_predictions_at(
    ctx: ArmContext,
    d_predictions: dict[str, np.ndarray],
    config_name: str,
    split: str,
) -> np.ndarray:
    """Apply the R correction to a given D state."""
    d_q = d_predictions[split]
    train_all = np.concatenate([d_predictions[s] for s in C.SPLITS])
    ref = _ref_local(ctx, split)
    d_ref = train_all[ref]
    weights = ctx.weights[split][config_name]
    return E.aggregate_r(d_q, d_ref, weights, ctx.y_ref[split])


# ---------------------------------------------------------------------------
# N
# ---------------------------------------------------------------------------


def run_n(ctx: ArmContext) -> dict[str, Any]:
    task = ctx.task
    predictions: dict[str, dict[str, np.ndarray]] = {}
    curves: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for config in C.CONFIGS:
        predictions[config["name"]] = {}
        for split in C.SPLITS:
            weights = ctx.weights[split][config["name"]]
            pred = E.aggregate_n(weights, ctx.y_ref[split])
            predictions[config["name"]][split] = pred
            rmse_raw, mae_raw = E.rmse_mae(pred, task.y[split])
            curves.append(_curve_row(task.dataset, "N", "constant", 0, split, config["name"], rmse_raw, mae_raw))
            if split == "valid":
                states.append(
                    {
                        "config": config["name"],
                        "K": int(config["K"]),
                        "weights": config["weights"],
                        "valid_rmse_raw": rmse_raw,
                        "valid_mae_raw": mae_raw,
                    }
                )
    selected = select_config(states)
    selected.update(
        {
            "train_rmse_raw": _lookup(curves, "train", selected["config"]),
            "test_rmse_raw": _lookup(curves, "test", selected["config"]),
            "train_mae_raw": _lookup(curves, "train", selected["config"], "mae_raw"),
            "test_mae_raw": _lookup(curves, "test", selected["config"], "mae_raw"),
        }
    )
    return {
        "curves": curves,
        "predictions": predictions,
        "selected": selected,
        "selected_predictions": predictions[selected["config"]],
        "states": states,
    }


def _lookup(curves: Iterable[dict[str, Any]], split: str, config: str, key: str = "rmse_raw") -> float:
    for row in curves:
        if row["split"] == split and row["config"] == config:
            return float(row[key])
    raise KeyError((split, config, key))


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def preset_rank(preset: str) -> int:
    return C.PRESET_ORDER.index(preset)


def config_rank(config_name: str) -> tuple[int, int]:
    for config in C.CONFIGS:
        if config["name"] == config_name:
            return int(config["K"]), (0 if config["weights"] == "uniform" else 1)
    raise KeyError(config_name)


def select_config(states: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Minimum Val RMSE over the six retrieval configs (smaller K, uniform first)."""
    return dict(min(states, key=lambda row: (row["valid_rmse_raw"], *config_rank(row["config"]))))


def select_d(results: Sequence[DRun]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for result in results:
        best = result.best
        candidates.append(
            {
                "preset": result.preset,
                "round": int(best["round"]),
                "valid_rmse_raw": float(best["rmse_raw"]),
            }
        )
    return min(candidates, key=lambda row: (row["valid_rmse_raw"], preset_rank(row["preset"]), row["round"]))


def select_r(results: Sequence[DRun]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for result in results:
        rows = [row for row in result.curves if row["method"] == "R" and row["split"] == "valid"]
        for row in rows:
            candidates.append(
                {
                    "preset": result.preset,
                    "round": int(row["round"]),
                    "config": row["config"],
                    "valid_rmse_raw": float(row["rmse_raw"]),
                }
            )
    return min(
        candidates,
        key=lambda row: (
            row["valid_rmse_raw"],
            preset_rank(row["preset"]),
            row["round"],
            *config_rank(row["config"]),
        ),
    )


def select_c(results: Sequence["CRun"]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for result in results:
        for row in result.curves:
            if row["split"] == "valid":
                candidates.append(
                    {
                        "preset": result.preset,
                        "round": int(row["round"]),
                        "config": row["config"],
                        "valid_rmse_raw": float(row["rmse_raw"]),
                    }
                )
    return min(
        candidates,
        key=lambda row: (
            row["valid_rmse_raw"],
            preset_rank(row["preset"]),
            row["round"],
            *config_rank(row["config"]),
        ),
    )


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------


@dataclass
class CRun:
    preset: str
    booster: Any
    trees: list[dict[str, np.ndarray]]
    curves: list[dict[str, Any]]
    pair_curves: list[dict[str, Any]]
    seconds: float
    trained_rounds: int
    matrix_build_seconds: float = 0.0
    eval_seconds: float = 0.0
    final_margins: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    checkpoints: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)


def _eval_pairs(ctx: ArmContext) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per split: (query, ref, accumulators' shape, y_ref matrix)."""
    out = {}
    for split in C.SPLITS:
        entry = ctx.ref_table["splits"][split]
        query, ref = P.evaluation_pairs(entry)
        out[split] = (query, ref, entry, ctx.y_ref[split])
    return out


def build_eval_blocks(ctx: ArmContext) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Correct feature-major blocks for every evaluated split.

    Query rows come from the evaluated split; reference rows always come from
    Train.  The two domains are passed explicitly, so a Val/Test query local
    row can never be looked up in the Train matrix.
    """
    task = ctx.task
    blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in C.SPLITS:
        query, ref = P.evaluation_pairs(ctx.ref_table["splits"][split])
        blocks[split] = P.pair_feature_blocks(
            query_x=task.x[split],
            reference_x=task.x["train"],
            query_local=query,
            reference_local=ref,
        )
    return blocks


def run_c(
    ctx: ArmContext,
    pair_spec: dict[str, Any],
    *,
    preset: str,
    device: str,
    rounds: int,
    verbose: bool = False,
) -> CRun:
    import xgboost as xgb

    task = ctx.task
    names = [f"pair_{name}" for name in task.feature_names] + [
        f"pair_b_{name}" for name in task.feature_names
    ] + [f"pair_d_{name}" for name in task.feature_names]
    src = np.asarray(pair_spec["src_local"], dtype=np.int64)
    ref = np.asarray(pair_spec["ref_local"], dtype=np.int64)
    t_matrix = time.perf_counter()
    matrix = P.pair_matrix(
        query_x=task.x["train"],
        reference_x=task.x["train"],
        query_local=src,
        reference_local=ref,
    )
    dtrain = xgb.DMatrix(
        matrix,
        label=np.asarray(pair_spec["target_std"], dtype=np.float32),
        weight=np.asarray(pair_spec["weight"], dtype=np.float32),
        feature_names=names,
    )
    matrix_build_seconds = time.perf_counter() - t_matrix
    del matrix
    params = C.SQX.final_params(preset, device)
    t0 = time.perf_counter()
    booster = xgb.train(params, dtrain, num_boost_round=int(rounds), verbose_eval=verbose)
    seconds = time.perf_counter() - t0
    del dtrain
    trees = T.extract_trees(booster, names, rounds=int(rounds))
    t_eval = time.perf_counter()
    result = _evaluate_c(ctx, pair_spec, preset, booster, trees, seconds)
    result.matrix_build_seconds = matrix_build_seconds
    result.eval_seconds = time.perf_counter() - t_eval
    return result


def _evaluate_c(
    ctx: ArmContext,
    pair_spec: dict[str, Any],
    preset: str,
    booster: Any,
    trees: Sequence[dict[str, np.ndarray]],
    seconds: float,
    *,
    eval_blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    include_pairs: bool = True,
) -> CRun:
    task = ctx.task
    n_features = task.n_features
    sigma = ctx.sigma
    blocks = eval_blocks if eval_blocks is not None else build_eval_blocks(ctx)
    shapes: dict[str, tuple[int, int]] = {}
    for split in C.SPLITS:
        shapes[split] = (task.n(split), C.MAX_REFERENCES)
    pair_blocks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    pair_target = np.asarray(pair_spec["target_std"], dtype=np.float64)
    pair_weight = np.asarray(pair_spec["weight"], dtype=np.float64)
    if include_pairs:
        pair_blocks = P.pair_feature_blocks(
            query_x=task.x["train"],
            reference_x=task.x["train"],
            query_local=np.asarray(pair_spec["src_local"], dtype=np.int64),
            reference_local=np.asarray(pair_spec["ref_local"], dtype=np.int64),
        )

    keys = list(C.SPLITS) + (["pairs"] if include_pairs else [])
    acc: dict[str, dict[str, np.ndarray]] = {}
    layout: dict[str, dict[str, T.BlockLayout]] = {}
    for split in C.SPLITS:
        n_rows = int(shapes[split][0] * shapes[split][1])
        acc[split] = {
            "fwd": np.zeros(n_rows, dtype=np.float32),
            "rev": np.zeros(n_rows, dtype=np.float32),
        }
        layout[split] = {
            "fwd": T.BlockLayout(blocks[split], n_features, reverse=False),
            "rev": T.BlockLayout(blocks[split], n_features, reverse=True),
        }
    if include_pairs:
        n_pair = int(pair_target.size)
        acc["pairs"] = {
            "fwd": np.zeros(n_pair, dtype=np.float32),
            "rev": np.zeros(n_pair, dtype=np.float32),
        }
        layout["pairs"] = {
            "fwd": T.BlockLayout(pair_blocks, n_features, reverse=False),
            "rev": T.BlockLayout(pair_blocks, n_features, reverse=True),
        }

    curves: list[dict[str, Any]] = []
    pair_curves: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[int, np.ndarray]] = {split: {} for split in C.SPLITS}
    for index, tree in enumerate(trees, start=1):
        for key in keys:
            np.add(
                acc[key]["fwd"],
                T.tree_leaf_values(tree, layout[key]["fwd"], int(acc[key]["fwd"].size)),
                out=acc[key]["fwd"],
            )
            np.add(
                acc[key]["rev"],
                T.tree_leaf_values(tree, layout[key]["rev"], int(acc[key]["rev"].size)),
                out=acc[key]["rev"],
            )
        if index in (1, 2, 3):
            for split in C.SPLITS:
                checkpoints[split][index] = np.concatenate(
                    [acc[split]["fwd"], acc[split]["rev"]]
                ).copy()
        for split in C.SPLITS:
            n_query, width = shapes[split]
            delta = E.delta_from_margins(
                acc[split]["fwd"].reshape(n_query, width),
                acc[split]["rev"].reshape(n_query, width),
                sigma,
            )
            y_ref = ctx.y_ref[split]
            for config in C.CONFIGS:
                weights = ctx.weights[split][config["name"]]
                pred = E.aggregate_c(weights, y_ref, delta)
                rmse_raw, mae_raw = E.rmse_mae(pred, task.y[split])
                curves.append(
                    _curve_row(task.dataset, "C", preset, index, split, config["name"], rmse_raw, mae_raw)
                )
        if include_pairs:
            pair_pred = acc["pairs"]["fwd"].astype(np.float64)
            antisym = 0.5 * (acc["pairs"]["fwd"].astype(np.float64) - acc["pairs"]["rev"].astype(np.float64))
            pair_curves.append(
                {
                    "dataset": task.dataset,
                    "method": "C",
                    "preset": preset,
                    "round": int(index),
                    "pair_rmse_std": E.weighted_rmse(pair_pred, pair_target, pair_weight),
                    "pair_rmse_std_antisym": E.weighted_rmse(antisym, pair_target, pair_weight),
                }
            )
    final_margins = {
        key: {"fwd": acc[key]["fwd"], "rev": acc[key]["rev"]} for key in keys
    }
    return CRun(
        preset=preset,
        booster=booster,
        trees=list(trees),
        curves=curves,
        pair_curves=pair_curves,
        seconds=float(seconds),
        trained_rounds=len(trees),
        final_margins=final_margins,
        checkpoints=checkpoints,
    )


# ---------------------------------------------------------------------------
# full-prefix recomputation of a selected C state (independent of the loop)
# ---------------------------------------------------------------------------


def c_state(
    ctx: ArmContext,
    *,
    booster: Any,
    trees: Sequence[dict[str, np.ndarray]],
    rounds: int,
    config_name: str,
    splits: Sequence[str] = C.SPLITS,
    sample_rows: dict[str, np.ndarray] | None = None,
    eval_blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Recompute C for one tree prefix from scratch, per split.

    Returns raw C predictions, the raw-unit delta matrix, both standardised
    margin accumulators (for the XGBoost cross-check) and the pair sample
    comparison when ``sample_rows`` is given.
    """
    task = ctx.task
    n_features = task.n_features
    out: dict[str, Any] = {
        "delta": {},
        "pred": {},
        "margins": {},
        "sample": {},
        "shapes": {},
    }
    for split in splits:
        entry = ctx.ref_table["splits"][split]
        query, ref = P.evaluation_pairs(entry)
        if eval_blocks is not None and split in eval_blocks:
            blocks = eval_blocks[split]
        else:
            blocks = P.pair_feature_blocks(
                query_x=task.x[split],
                reference_x=task.x["train"],
                query_local=query,
                reference_local=ref,
            )
        n_query, width = int(query.size // C.MAX_REFERENCES), C.MAX_REFERENCES
        n_rows = int(query.size)
        acc_f = np.zeros(n_rows, dtype=np.float32)
        acc_r = np.zeros(n_rows, dtype=np.float32)
        T.accumulate(acc_f, trees, T.BlockLayout(blocks, n_features, reverse=False), stop=int(rounds))
        T.accumulate(acc_r, trees, T.BlockLayout(blocks, n_features, reverse=True), stop=int(rounds))
        delta = E.delta_from_margins(acc_f.reshape(n_query, width), acc_r.reshape(n_query, width), ctx.sigma)
        weights = ctx.weights[split][config_name]
        pred = E.aggregate_c(weights, ctx.y_ref[split], delta)
        out["delta"][split] = delta.astype(np.float32)
        out["pred"][split] = pred
        out["margins"][split] = {"fwd": acc_f, "rev": acc_r}
        out["shapes"][split] = (n_query, width)
        if sample_rows is not None and split in sample_rows:
            rows = np.asarray(sample_rows[split], dtype=np.int64)
            fwd_matrix, rev_matrix = _sample_pair_matrices(task, split, entry, rows)
            import xgboost as xgb

            names = _pair_feature_names(task)
            fwd_pred = booster.predict(
                xgb.DMatrix(fwd_matrix, feature_names=names), iteration_range=(0, int(rounds))
            )
            rev_pred = booster.predict(
                xgb.DMatrix(rev_matrix, feature_names=names), iteration_range=(0, int(rounds))
            )
            out["sample"][split] = {
                "rows": rows,
                "xgboost_fwd": np.asarray(fwd_pred, dtype=np.float64),
                "xgboost_rev": np.asarray(rev_pred, dtype=np.float64),
                "ours_fwd": acc_f[rows].astype(np.float64),
                "ours_rev": acc_r[rows].astype(np.float64),
            }
        del blocks
    return out


def _pair_feature_names(task: D.TaskData) -> list[str]:
    return [f"pair_{name}" for name in task.feature_names] + [
        f"pair_b_{name}" for name in task.feature_names
    ] + [f"pair_d_{name}" for name in task.feature_names]


def _sample_pair_matrices(
    task: D.TaskData,
    split: str,
    entry: dict[str, Any],
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward/reverse pair matrices for sampled flat slots of one split.

    Forward is ``z(query_split, reference_train)``.  Reverse is
    ``z(reference_train, query_split)``, built from the two *correct* domains,
    not by swapping two local indices inside the Train matrix.
    """
    query, ref = P.evaluation_pairs(entry)
    src = query[rows].astype(np.int64)
    dst = ref[rows].astype(np.int64)
    forward = P.pair_matrix(
        query_x=task.x[split],
        reference_x=task.x["train"],
        query_local=src,
        reference_local=dst,
    )
    reverse = P.pair_matrix(
        query_x=task.x["train"],
        reference_x=task.x[split],
        query_local=dst,
        reference_local=src,
    )
    return forward, reverse


def sample_flat_rows(n_rows: int, count: int, *, seed: int = C.BOOTSTRAP_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=min(count, n_rows), replace=False)).astype(np.int64)


def verify_d(
    ctx: ArmContext,
    *,
    booster: Any,
    trees: Sequence[dict[str, np.ndarray]],
    round_index: int,
    recorded_predictions: dict[str, np.ndarray],
    curves_index: dict[tuple[int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Full-prefix XGBoost recomputation of D (and the R correction it implies)."""
    import xgboost as xgb

    task = ctx.task
    names = list(task.feature_names)
    x_all = task.x_concat(C.SPLITS)
    pred_std = booster.predict(xgb.DMatrix(x_all, feature_names=names), iteration_range=(0, int(round_index)))
    d_all = ctx.mu + ctx.sigma * np.asarray(pred_std, dtype=np.float64)
    report: dict[str, Any] = {"round": int(round_index), "splits": {}}
    cursor = 0
    fresh: dict[str, np.ndarray] = {}
    for split in C.SPLITS:
        fresh[split] = d_all[cursor:cursor + task.n(split)]
        cursor += task.n(split)
    for split in C.SPLITS:
        diff = float(np.max(np.abs(fresh[split] - recorded_predictions[split])))
        recorded = curves_index[(int(round_index), split, "none")]
        rmse_raw, mae_raw = E.rmse_mae(fresh[split], task.y[split])
        report["splits"][split] = {
            "max_abs_sample_diff_raw": diff,
            "rmse_xgboost": rmse_raw,
            "rmse_recorded": recorded["rmse_raw"],
            "mae_xgboost": mae_raw,
            "mae_recorded": recorded["mae_raw"],
            "rmse_abs_diff": abs(rmse_raw - recorded["rmse_raw"]),
        }
    report["r_axis"] = verify_r(ctx, fresh, round_index, curves_index)
    return report


def verify_r(
    ctx: ArmContext,
    d_predictions: dict[str, np.ndarray],
    round_index: int,
    curves_index: dict[tuple[int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Recompute every R configuration from a given D state."""
    task = ctx.task
    report: dict[str, Any] = {}
    for split in C.SPLITS:
        for config in C.CONFIGS:
            pred = d_r_predictions_at(ctx, d_predictions, config["name"], split)
            rmse_raw, mae_raw = E.rmse_mae(pred, task.y[split])
            recorded = curves_index[(int(round_index), split, config["name"])]
            report[f"{split}/{config['name']}"] = {
                "rmse_fresh": rmse_raw,
                "rmse_recorded": recorded["rmse_raw"],
                "rmse_abs_diff": abs(rmse_raw - recorded["rmse_raw"]),
                "mae_abs_diff": abs(mae_raw - recorded["mae_raw"]),
            }
    return report


def verify_c(
    ctx: ArmContext,
    *,
    booster: Any,
    pair_spec: dict[str, Any],
    final_margins: dict[str, dict[str, np.ndarray]],
    state: dict[str, Any],
    round_index: int,
    final_round: int,
    curves_index: dict[tuple[int, str, str], dict[str, Any]],
    config_name: str,
    sample_count: int = 512,
) -> dict[str, Any]:
    """Cross-check C: fresh-prefix metrics plus XGBoost per-row margins."""
    task = ctx.task
    report: dict[str, Any] = {"round": int(round_index), "splits": {}, "pair_sample": None}
    for split in C.SPLITS:
        n_query, width = state["shapes"][split]
        pred = state["pred"][split]
        recorded = curves_index[(int(round_index), split, config_name)]
        rmse_raw, mae_raw = E.rmse_mae(pred, task.y[split])
        sample = state["sample"][split]
        report["splits"][split] = {
            "rmse_fresh": rmse_raw,
            "rmse_recorded": recorded["rmse_raw"],
            "rmse_abs_diff": abs(rmse_raw - recorded["rmse_raw"]),
            "mae_fresh": mae_raw,
            "mae_recorded": recorded["mae_raw"],
            "max_abs_margin_diff_sel_fwd": float(
                np.max(np.abs(sample["xgboost_fwd"] - sample["ours_fwd"]))
            ),
            "max_abs_margin_diff_sel_rev": float(
                np.max(np.abs(sample["xgboost_rev"] - sample["ours_rev"]))
            ),
            "n_sample_rows": int(sample["rows"].size),
        }
    pair_rows = sample_flat_rows(int(pair_spec["src_local"].size), sample_count)
    src = np.asarray(pair_spec["src_local"], dtype=np.int64)[pair_rows]
    ref = np.asarray(pair_spec["ref_local"], dtype=np.int64)[pair_rows]
    fwd_matrix = P.pair_matrix(
        query_x=task.x["train"],
        reference_x=task.x["train"],
        query_local=src,
        reference_local=ref,
    )
    rev_matrix = P.pair_matrix(
        query_x=task.x["train"],
        reference_x=task.x["train"],
        query_local=ref,
        reference_local=src,
    )
    names = _pair_feature_names(task)
    import xgboost as xgb

    fwd_pred = np.asarray(
        booster.predict(xgb.DMatrix(fwd_matrix, feature_names=names), iteration_range=(0, int(final_round))),
        dtype=np.float64,
    )
    rev_pred = np.asarray(
        booster.predict(xgb.DMatrix(rev_matrix, feature_names=names), iteration_range=(0, int(final_round))),
        dtype=np.float64,
    )
    report["pair_sample"] = {
        "n_rows": int(pair_rows.size),
        "max_abs_margin_diff_final_fwd": float(
            np.max(np.abs(fwd_pred - final_margins["pairs"]["fwd"][pair_rows].astype(np.float64)))
        ),
        "max_abs_margin_diff_final_rev": float(
            np.max(np.abs(rev_pred - final_margins["pairs"]["rev"][pair_rows].astype(np.float64)))
        ),
    }
    report["final_state"] = compare_final_margins(
        ctx,
        booster=booster,
        final_margins=final_margins,
        final_round=final_round,
        sample_count=sample_count,
    )
    return report


def compare_final_margins(
    ctx: ArmContext,
    *,
    booster: Any,
    final_margins: dict[str, dict[str, np.ndarray]],
    final_round: int,
    sample_count: int = 512,
) -> dict[str, Any]:
    """XGBoost full-prefix margins vs the loop's final accumulator (sampled rows)."""
    task = ctx.task
    names = _pair_feature_names(task)
    import xgboost as xgb

    report: dict[str, Any] = {"round": int(final_round), "splits": {}}
    for split in C.SPLITS:
        entry = ctx.ref_table["splits"][split]
        n_rows = int(task.n(split) * C.MAX_REFERENCES)
        rows = sample_flat_rows(n_rows, sample_count, seed=C.PAIR_RNG_SEED)
        fwd, rev = _sample_pair_matrices(task, split, entry, rows)
        fwd_pred = np.asarray(
            booster.predict(xgb.DMatrix(fwd, feature_names=names), iteration_range=(0, int(final_round))),
            dtype=np.float64,
        )
        rev_pred = np.asarray(
            booster.predict(xgb.DMatrix(rev, feature_names=names), iteration_range=(0, int(final_round))),
            dtype=np.float64,
        )
        report["splits"][split] = {
            "n_rows": int(rows.size),
            "max_abs_margin_diff_fwd": float(
                np.max(np.abs(fwd_pred - final_margins[split]["fwd"][rows].astype(np.float64)))
            ),
            "max_abs_margin_diff_rev": float(
                np.max(np.abs(rev_pred - final_margins[split]["rev"][rows].astype(np.float64)))
            ),
        }
    return report


def verify_c_prefix(
    ctx: ArmContext,
    *,
    state: dict[str, Any],
    round_index: int,
    config_name: str,
    curves_index: dict[tuple[int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Round-level C check: fresh metrics and XGBoost per-row margins on a sample."""
    task = ctx.task
    report: dict[str, Any] = {"round": int(round_index), "splits": {}}
    for split in C.SPLITS:
        pred = state["pred"][split]
        recorded = curves_index[(int(round_index), split, config_name)]
        rmse_raw, mae_raw = E.rmse_mae(pred, task.y[split])
        sample = state["sample"][split]
        report["splits"][split] = {
            "rmse_fresh": rmse_raw,
            "rmse_recorded": recorded["rmse_raw"],
            "rmse_abs_diff": abs(rmse_raw - recorded["rmse_raw"]),
            "mae_abs_diff": abs(mae_raw - recorded["mae_raw"]),
            "max_abs_margin_diff_fwd": float(
                np.max(np.abs(sample["xgboost_fwd"] - sample["ours_fwd"]))
            ),
            "max_abs_margin_diff_rev": float(
                np.max(np.abs(sample["xgboost_rev"] - sample["ours_rev"]))
            ),
            "n_sample_rows": int(sample["rows"].size),
        }
    return report
