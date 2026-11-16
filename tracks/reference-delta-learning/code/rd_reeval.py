"""Evaluation-only repair helpers for ``rd-evalfix-v1``.

The original C evaluator reused a pair constructor that always read
``task.x["train"]``.  That was correct for the Train--Train training records and
wrong for every Val/Test evaluation query.  This module contains only
evaluation, identity and oracle checks; it never trains or overwrites a model.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import rd_arms as A
import rd_common as C
import rd_data as D
import rd_eval as E
import rd_pairs as P


# ---------------------------------------------------------------------------
# independent reference implementation (deliberately not P.pair_matrix)
# ---------------------------------------------------------------------------


def independent_pair_matrix(
    *,
    query_x: np.ndarray,
    reference_x: np.ndarray,
    query_local: np.ndarray,
    reference_local: np.ndarray,
    chunk: int = 2048,
) -> np.ndarray:
    """A second, deliberately separate implementation of the pair layout.

    The real evaluation must not be verified by calling the same function that
    produced the bug.  This helper is written directly against the raw split
    matrices and is used only by the independent checks.
    """
    q = np.asarray(query_local, dtype=np.int64).reshape(-1)
    r = np.asarray(reference_local, dtype=np.int64).reshape(-1)
    qx = np.asarray(query_x)
    rx = np.asarray(reference_x)
    if qx.ndim != 2 or rx.ndim != 2 or qx.shape[1] != rx.shape[1]:
        raise ValueError("independent pair domains must be two 2-D matrices of equal width")
    if q.size != r.size:
        raise ValueError("independent pair domains must have one reference per query")
    n_features = int(qx.shape[1])
    out = np.empty((int(q.size), 3 * n_features), dtype=np.float32)
    for start in range(0, int(q.size), int(chunk)):
        stop = min(start + int(chunk), int(q.size))
        head = qx[q[start:stop]]
        tail = rx[r[start:stop]]
        out[start:stop, :n_features] = head
        out[start:stop, n_features:2 * n_features] = tail
        out[start:stop, 2 * n_features:] = head - tail
    return out


# ---------------------------------------------------------------------------
# training prohibition
# ---------------------------------------------------------------------------


@contextmanager
def forbid_training():
    """Replace every boosting fit entry point with a hard failure.

    The counter is asserted to stay zero by the reevaluation stage.  Loading a
    booster from ``.ubj`` and calling ``predict`` are unaffected.
    """
    import xgboost as xgb

    calls = {"train": 0, "booster_update": 0, "sklearn_fit": 0}
    saved: list[tuple[Any, str, Any]] = []

    def _trip(kind: str):
        def _forbidden(*args: Any, **kwargs: Any) -> Any:
            calls[kind] += 1
            raise AssertionError(
                f"evaluation-only stage attempted xgboost {kind}; models are frozen"
            )

        return _forbidden

    for name in ("train",):
        if hasattr(xgb, name):
            saved.append((xgb, name, getattr(xgb, name)))
            setattr(xgb, name, _trip("train"))
    if hasattr(xgb, "Booster") and hasattr(xgb.Booster, "update"):
        saved.append((xgb.Booster, "update", xgb.Booster.update))
        setattr(xgb.Booster, "update", _trip("booster_update"))
    for name in ("XGBRegressor", "XGBClassifier", "XGBRanker"):
        cls = getattr(xgb, name, None)
        if cls is not None and hasattr(cls, "fit"):
            saved.append((cls, "fit", cls.fit))
            setattr(cls, "fit", _trip("sklearn_fit"))
    try:
        yield calls
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


# ---------------------------------------------------------------------------
# identity / fingerprint
# ---------------------------------------------------------------------------


def _code_hashes() -> dict[str, str]:
    names = ("rd_common.py", "rd_pairs.py", "rd_arms.py", "rd_eval.py", "rd_reeval.py")
    return {name: C.sha256_file(C.CODE_DIR / name) for name in names}


def build_evaluation_identity(
    task: D.TaskData,
    ref_table: dict[str, Any],
    *,
    reference_hash: str,
    pair_hash: str,
    model_sha256: Mapping[str, str],
    config_weights: Mapping[str, Mapping[str, np.ndarray]],
    revision: str = C.EVALUATION_REVISION,
) -> dict[str, Any]:
    """Identity of every C evaluation input, not just the dataset name.

    The cache key includes the revision, code hashes, each split's row_id and
    feature hashes, the Train reference domain, the frozen reference/pair
    hashes, the actual weight matrices, the scaler and both model files.
    """
    payload: dict[str, Any] = {
        "evaluation_revision": revision,
        "protocol_id": C.PROTOCOL_ID,
        "dataset": task.dataset,
        "code_sha256": _code_hashes(),
        "split_sizes": {split: task.n(split) for split in C.SPLITS},
        "query_split": {
            split: {
                "row_id_sha256": C.sha256_array(task.row_id[split]),
                "feature_sha256": C.sha256_array(task.x[split]),
                "n_rows": task.n(split),
            }
            for split in C.SPLITS
        },
        "reference_split": {
            "train": {
                "row_id_sha256": C.sha256_array(task.row_id["train"]),
                "feature_sha256": C.sha256_array(task.x["train"]),
                "n_rows": task.n("train"),
            }
        },
        "reference_hash": str(reference_hash),
        "pair_hash": str(pair_hash),
        "weights_sha256": {
            split: {
                name: C.sha256_array(np.asarray(matrix, dtype=np.float64))
                for name, matrix in by_name.items()
            }
            for split, by_name in config_weights.items()
        },
        "configs": [dict(config) for config in C.CONFIGS],
        "scaler": dict(task.scaler),
        "model_sha256": {preset: str(sha) for preset, sha in model_sha256.items()},
    }
    payload["cache_key"] = C.sha256_json(payload)
    return payload


def identity_short(identity: Mapping[str, Any]) -> str:
    return str(identity["cache_key"])[:20]


# ---------------------------------------------------------------------------
# fixed legal slot sample (seed 1703, first/last query covered)
# ---------------------------------------------------------------------------


def sample_legal_slots(
    ref_table: dict[str, Any],
    split: str,
    *,
    count: int = 8,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    entry = ref_table["splits"][split]
    ref_local = np.asarray(entry["ref_local"], dtype=np.int32)
    n_kept = np.asarray(entry["n_kept"], dtype=np.int32)
    n_query, width = ref_local.shape
    legal_q = np.repeat(np.arange(n_query, dtype=np.int64), n_kept.astype(np.int64))
    legal_slot = np.concatenate(
        [np.arange(int(n_kept[q]), dtype=np.int64) for q in range(n_query)]
    ) if n_query else np.zeros(0, dtype=np.int64)
    total = int(legal_q.size)
    if total == 0:
        return {
            "rows": np.zeros(0, dtype=np.int64),
            "query_local": np.zeros(0, dtype=np.int64),
            "reference_local": np.zeros(0, dtype=np.int64),
            "slot": np.zeros(0, dtype=np.int64),
        }
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    for query in (0, n_query - 1):
        if int(n_kept[query]) > 0:
            selected.add(int(query * width))
    take = min(int(count), total)
    if len(selected) < take:
        for flat in rng.choice(total, size=total, replace=False).tolist():
            selected.add(int(legal_q[flat] * width + legal_slot[flat]))
            if len(selected) >= take:
                break
    rows = np.asarray(sorted(selected), dtype=np.int64)
    query_local = rows // width
    slot = rows % width
    reference_local = ref_local[query_local, slot].astype(np.int64)
    return {
        "rows": rows,
        "query_local": query_local,
        "reference_local": reference_local,
        "slot": slot,
    }


# ---------------------------------------------------------------------------
# input identity checks
# ---------------------------------------------------------------------------


def check_input_identity(
    task: D.TaskData,
    ref_table: dict[str, Any],
    eval_blocks: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sample: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}, "pass": True}
    for split in C.SPLITS:
        rows = np.asarray(sample[split]["rows"], dtype=np.int64)
        q = np.asarray(sample[split]["query_local"], dtype=np.int64)
        r = np.asarray(sample[split]["reference_local"], dtype=np.int64)
        entry = ref_table["splits"][split]
        q_true = task.x[split][q]
        r_true = task.x["train"][r]
        z_true_fwd = np.concatenate([q_true, r_true, q_true - r_true], axis=1)
        z_true_rev = np.concatenate([r_true, q_true, r_true - q_true], axis=1)
        x1, x2, x3 = eval_blocks[split]
        z_actual_fwd = np.concatenate(
            [x1[:, rows].T, x2[:, rows].T, x3[:, rows].T], axis=1
        )
        z_actual_rev = np.concatenate(
            [x2[:, rows].T, x1[:, rows].T, (-x3[:, rows]).T], axis=1
        )
        row_id_query = np.asarray(entry["query_row_id"], dtype=np.int64)
        row_id_ref = np.asarray(entry["ref_row_id"], dtype=np.int64)
        query_id_ok = bool(
            np.array_equal(row_id_query[q], task.row_id[split][q])
        )
        reference_id_ok = bool(
            np.array_equal(row_id_ref[q, np.asarray(sample[split]["slot"])], task.row_id["train"][r])
        )
        query_max = float(np.max(np.abs(z_actual_fwd[:, : q_true.shape[1]] - q_true))) if rows.size else 0.0
        ref_max = float(
            np.max(
                np.abs(
                    z_actual_fwd[:, q_true.shape[1]:2 * q_true.shape[1]] - r_true
                )
            )
        ) if rows.size else 0.0
        diff_max = float(
            np.max(
                np.abs(z_actual_fwd[:, 2 * q_true.shape[1]:] - (q_true - r_true))
            )
        ) if rows.size else 0.0
        fwd_max = float(np.max(np.abs(z_actual_fwd - z_true_fwd))) if rows.size else 0.0
        rev_max = float(np.max(np.abs(z_actual_rev - z_true_rev))) if rows.size else 0.0
        split_pass = bool(
            query_id_ok and reference_id_ok and fwd_max == 0.0 and rev_max == 0.0
        )
        report["splits"][split] = {
            "n_sample_slots": int(rows.size),
            "first_query": int(q[0]) if rows.size else None,
            "last_query": int(q[-1]) if rows.size else None,
            "query_row_id_matches": query_id_ok,
            "reference_row_id_matches": reference_id_ok,
            "max_abs_query_block_diff": query_max,
            "max_abs_reference_block_diff": ref_max,
            "max_abs_difference_block_diff": diff_max,
            "max_abs_forward_pair_diff": fwd_max,
            "max_abs_reverse_pair_diff": rev_max,
            "pass": split_pass,
        }
        report["pass"] = bool(report["pass"] and split_pass)
    return report


# ---------------------------------------------------------------------------
# independent XGBoost prefix / full-prediction checks
# ---------------------------------------------------------------------------


def check_prefix_predictions(
    task: D.TaskData,
    ref_table: dict[str, Any],
    *,
    booster: Any,
    sample: Mapping[str, Mapping[str, np.ndarray]],
    snapshots: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    sigma: float,
) -> dict[str, Any]:
    """Direct XGBoost vs the incremental replay on the same independent input."""
    import xgboost as xgb

    report: dict[str, Any] = {"rounds": {}, "pass": True}
    names = A._pair_feature_names(task)
    for split in C.SPLITS:
        rows = np.asarray(sample[split]["rows"], dtype=np.int64)
        q = np.asarray(sample[split]["query_local"], dtype=np.int64)
        r = np.asarray(sample[split]["reference_local"], dtype=np.int64)
        fwd_matrix = independent_pair_matrix(
            query_x=task.x[split],
            reference_x=task.x["train"],
            query_local=q,
            reference_local=r,
        )
        rev_matrix = independent_pair_matrix(
            query_x=task.x["train"],
            reference_x=task.x[split],
            query_local=r,
            reference_local=q,
        )
        fwd_dm = xgb.DMatrix(fwd_matrix, feature_names=names)
        rev_dm = xgb.DMatrix(rev_matrix, feature_names=names)
        for round_index in sorted(int(key) for key in snapshots):
            snapshot = snapshots[int(round_index)][split]
            xgb_fwd = np.asarray(
                booster.predict(fwd_dm, iteration_range=(0, int(round_index))),
                dtype=np.float64,
            )
            xgb_rev = np.asarray(
                booster.predict(rev_dm, iteration_range=(0, int(round_index))),
                dtype=np.float64,
            )
            ours_fwd = np.asarray(snapshot["fwd"][rows], dtype=np.float64)
            ours_rev = np.asarray(snapshot["rev"][rows], dtype=np.float64)
            delta_xgb = E.delta_from_margins(xgb_fwd, xgb_rev, sigma)
            delta_ours = E.delta_from_margins(ours_fwd, ours_rev, sigma)
            entry = {
                "n_rows": int(rows.size),
                "max_abs_margin_diff_fwd": float(
                    np.max(np.abs(xgb_fwd - ours_fwd))
                ) if rows.size else 0.0,
                "max_abs_margin_diff_rev": float(
                    np.max(np.abs(xgb_rev - ours_rev))
                ) if rows.size else 0.0,
                "max_abs_raw_delta_diff": float(
                    np.max(np.abs(delta_xgb - delta_ours))
                ) if rows.size else 0.0,
            }
            entry["pass"] = bool(
                entry["max_abs_margin_diff_fwd"] <= 1e-5
                and entry["max_abs_margin_diff_rev"] <= 1e-5
                and entry["max_abs_raw_delta_diff"] <= 1e-5
            )
            report["rounds"].setdefault(str(int(round_index)), {})[split] = entry
            report["pass"] = bool(report["pass"] and entry["pass"])
    return report


def independent_full_predictions(
    task: D.TaskData,
    ref_table: dict[str, Any],
    *,
    booster: Any,
    split: str,
    round_index: int,
    config_name: str,
    weights: np.ndarray,
    y_ref: np.ndarray,
    block_queries: int = 64,
) -> np.ndarray:
    """Chunked native XGBoost prediction from raw split matrices."""
    import xgboost as xgb

    entry = ref_table["splits"][split]
    ref_local = np.asarray(entry["ref_local"], dtype=np.int32)
    n_kept = np.asarray(entry["n_kept"], dtype=np.int32)
    n_query, width = ref_local.shape
    names = A._pair_feature_names(task)
    out = np.empty(n_query, dtype=np.float64)
    q_local = np.arange(n_query, dtype=np.int64)
    for start in range(0, n_query, int(block_queries)):
        stop = min(start + int(block_queries), n_query)
        block = q_local[start:stop]
        query = np.repeat(block, width)
        slot = np.tile(np.arange(width, dtype=np.int64), block.size)
        reference = ref_local[start:stop].reshape(-1).astype(np.int64)
        # legal slots only; padded slots carry zero weight by construction
        legal = slot < np.repeat(n_kept[start:stop], width)
        reference = np.where(legal, reference, 0).astype(np.int64)
        fwd = independent_pair_matrix(
            query_x=task.x[split],
            reference_x=task.x["train"],
            query_local=query,
            reference_local=reference,
        )
        rev = independent_pair_matrix(
            query_x=task.x["train"],
            reference_x=task.x[split],
            query_local=reference,
            reference_local=query,
        )
        m_fwd = np.asarray(
            booster.predict(
                xgb.DMatrix(fwd, feature_names=names),
                iteration_range=(0, int(round_index)),
            ),
            dtype=np.float64,
        )
        m_rev = np.asarray(
            booster.predict(
                xgb.DMatrix(rev, feature_names=names),
                iteration_range=(0, int(round_index)),
            ),
            dtype=np.float64,
        )
        delta = E.delta_from_margins(
            m_fwd.reshape(stop - start, width),
            m_rev.reshape(stop - start, width),
            float(task.scaler["std"]),
        )
        out[start:stop] = E.aggregate_c(
            np.asarray(weights[start:stop], dtype=np.float64),
            np.asarray(y_ref[start:stop], dtype=np.float64),
            delta,
        )
    return out


def check_full_native_state(
    task: D.TaskData,
    ref_table: dict[str, Any],
    *,
    booster: Any,
    state: Mapping[str, Any],
    round_index: int,
    config_name: str,
    ctx: A.ArmContext,
    splits: Sequence[str] = ("valid", "test"),
) -> dict[str, Any]:
    report: dict[str, Any] = {"round": int(round_index), "config": config_name, "splits": {}, "pass": True}
    for split in splits:
        native = independent_full_predictions(
            task,
            ref_table,
            booster=booster,
            split=split,
            round_index=int(round_index),
            config_name=config_name,
            weights=ctx.weights[split][config_name],
            y_ref=ctx.y_ref[split],
        )
        ours = np.asarray(state["pred"][split], dtype=np.float64)
        rmse_native, mae_native = E.rmse_mae(native, task.y[split])
        rmse_ours, mae_ours = E.rmse_mae(ours, task.y[split])
        entry = {
            "n_queries": int(native.size),
            "max_abs_prediction_diff": float(np.max(np.abs(native - ours))),
            "rmse_native": rmse_native,
            "mae_native": mae_native,
            "rmse_replayed": rmse_ours,
            "mae_replayed": mae_ours,
            "rmse_abs_diff": abs(rmse_native - rmse_ours),
            "mae_abs_diff": abs(mae_native - mae_ours),
        }
        entry["pass"] = bool(
            entry["max_abs_prediction_diff"] <= 1e-5
            and entry["rmse_abs_diff"] <= 1e-9
            and entry["mae_abs_diff"] <= 1e-9
        )
        report["splits"][split] = entry
        report["pass"] = bool(report["pass"] and entry["pass"])
    return report


# ---------------------------------------------------------------------------
# training-input invariance
# ---------------------------------------------------------------------------


def legacy_train_pair_matrix(
    x_train: np.ndarray,
    query_local: np.ndarray,
    reference_local: np.ndarray,
    *,
    chunk: int = 8192,
) -> np.ndarray:
    """The pre-repair constructor restricted to its *correct* Train use.

    Used only to prove that the repaired explicit-domain call is bit-identical
    on Train--Train records.
    """
    q = np.asarray(query_local, dtype=np.int64)
    r = np.asarray(reference_local, dtype=np.int64)
    n_features = int(x_train.shape[1])
    out = np.empty((int(q.size), 3 * n_features), dtype=np.float32)
    for start in range(0, int(q.size), int(chunk)):
        stop = min(start + int(chunk), int(q.size))
        head = x_train[q[start:stop]]
        tail = x_train[r[start:stop]]
        out[start:stop, :n_features] = head
        out[start:stop, n_features:2 * n_features] = tail
        out[start:stop, 2 * n_features:] = head - tail
    return out


def legacy_train_pair_blocks(
    x_train: np.ndarray,
    query_local: np.ndarray,
    reference_local: np.ndarray,
    *,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(query_local, dtype=np.int64)
    r = np.asarray(reference_local, dtype=np.int64)
    n_features = int(x_train.shape[1])
    x1 = np.empty((n_features, int(q.size)), dtype=np.float32)
    x2 = np.empty((n_features, int(q.size)), dtype=np.float32)
    x3 = np.empty((n_features, int(q.size)), dtype=np.float32)
    for start in range(0, int(q.size), int(chunk)):
        stop = min(start + int(chunk), int(q.size))
        head = x_train[q[start:stop]]
        tail = x_train[r[start:stop]]
        x1[:, start:stop] = head.T
        x2[:, start:stop] = tail.T
        x3[:, start:stop] = (head - tail).T
    return x1, x2, x3


def check_training_input_unchanged(
    task: D.TaskData,
    pair_spec: Mapping[str, Any],
    *,
    count: int = 512,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    src = np.asarray(pair_spec["src_local"], dtype=np.int64)
    ref = np.asarray(pair_spec["ref_local"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rows = np.sort(
        rng.choice(src.size, size=min(int(count), int(src.size)), replace=False)
    ) if src.size else np.zeros(0, dtype=np.int64)
    new = P.pair_matrix(
        query_x=task.x["train"],
        reference_x=task.x["train"],
        query_local=src[rows],
        reference_local=ref[rows],
    )
    legacy = legacy_train_pair_matrix(task.x["train"], src[rows], ref[rows])
    new_blocks = P.pair_feature_blocks(
        query_x=task.x["train"],
        reference_x=task.x["train"],
        query_local=src[rows],
        reference_local=ref[rows],
    )
    legacy_blocks = legacy_train_pair_blocks(task.x["train"], src[rows], ref[rows])
    max_diff = float(np.max(np.abs(new - legacy))) if rows.size else 0.0
    blocks_diff = max(
        float(np.max(np.abs(a - b))) for a, b in zip(new_blocks, legacy_blocks, strict=True)
    ) if rows.size else 0.0
    return {
        "n_sample_records": int(rows.size),
        "max_abs_dense_diff": max_diff,
        "max_abs_block_diff": blocks_diff,
        "bitwise_identical": bool(max_diff == 0.0 and blocks_diff == 0.0),
        "sample_sha256": C.sha256_array(new),
        "pass": bool(max_diff == 0.0 and blocks_diff == 0.0),
    }


# ---------------------------------------------------------------------------
# curve comparison and cache
# ---------------------------------------------------------------------------


def compare_train_curves(
    old_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
    *,
    preset: str,
) -> dict[str, Any]:
    old = {
        (int(row["round"]), row["split"], row["config"]): row
        for row in old_rows
        if row["preset"] == preset
    }
    new = {
        (int(row["round"]), row["split"], row["config"]): row
        for row in new_rows
        if row["preset"] == preset
    }
    missing = sorted(set(old) - set(new))
    extra = sorted(set(new) - set(old))
    worst_rmse = 0.0
    worst_mae = 0.0
    count = 0
    for key in sorted(set(old) & set(new)):
        if key[1] != "train":
            continue
        worst_rmse = max(worst_rmse, abs(float(old[key]["rmse_raw"]) - float(new[key]["rmse_raw"])))
        worst_mae = max(worst_mae, abs(float(old[key]["mae_raw"]) - float(new[key]["mae_raw"])))
        count += 1
    return {
        "n_train_rows_compared": int(count),
        "max_abs_train_rmse_diff": float(worst_rmse),
        "max_abs_train_mae_diff": float(worst_mae),
        "missing_keys": [list(key) for key in missing[:5]],
        "extra_keys": [list(key) for key in extra[:5]],
        "pass": bool(
            count > 0
            and not missing
            and not extra
            and worst_rmse <= 1e-5
            and worst_mae <= 1e-5
        ),
    }


def cache_path(dataset: str, preset: str, identity: Mapping[str, Any]) -> Path:
    return (
        C.CACHE_DIR
        / C.EVALUATION_REVISION
        / dataset
        / f"C_{preset}_{identity_short(identity)}.npz"
    )


def save_curve_cache(
    path: Path, rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "identity_hash": np.asarray(C.sha256_json(dict(identity))),
        "evaluation_revision": np.asarray(str(identity["evaluation_revision"])),
        "round": np.asarray([int(row["round"]) for row in rows], dtype=np.int32),
        "split": np.asarray([str(row["split"]) for row in rows]),
        "config": np.asarray([str(row["config"]) for row in rows]),
        "rmse_raw": np.asarray([float(row["rmse_raw"]) for row in rows], dtype=np.float64),
        "mae_raw": np.asarray([float(row["mae_raw"]) for row in rows], dtype=np.float64),
    }
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "sha256": C.sha256_file(path), "bytes": int(path.stat().st_size)}


def load_curve_cache(
    path: Path, *, preset: str, dataset: str, identity: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    expected = C.sha256_json(dict(identity))
    with np.load(path, allow_pickle=False) as handle:
        if str(handle["identity_hash"]) != expected:
            return None
        if str(handle["evaluation_revision"]) != str(identity["evaluation_revision"]):
            return None
        rows = [
            {
                "dataset": dataset,
                "method": "C",
                "preset": preset,
                "round": int(handle["round"][index]),
                "split": str(handle["split"][index]),
                "config": str(handle["config"][index]),
                "rmse_raw": float(handle["rmse_raw"][index]),
                "mae_raw": float(handle["mae_raw"][index]),
            }
            for index in range(int(handle["round"].size))
        ]
    if len(rows) != C.ROUNDS_C * len(C.SPLITS) * len(C.CONFIGS):
        return None
    return rows
