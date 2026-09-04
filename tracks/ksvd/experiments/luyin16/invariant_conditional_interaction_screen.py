"""Train-only screen for coordinate-stable structure--attribute interactions.

The previous centre-interaction route projects a 96 x 53 topology--attribute
covariance and a 2 x (64 x 40 + 32 x 13) role-binding object with PCA.  PCA is
useful for the XGBoost probe, but its coordinates can rotate when the fitting
scope changes.  This runner keeps the same cached, permutation-invariant
objects and adds two small, label-free readouts:

* ``invariant_cross``: singular-spectrum, energy, entropy and norm summaries
  of the covariance blocks.  These summaries do not depend on a choice of PCA
  axes (the singular values are invariant to orthogonal changes of basis).
* ``conditional_binding``: role-conditioned concentration summaries of the
  binding residual.  The row norms and row/column entropy describe how
  attributes are distributed *given* a rooted-WL role, without concatenating
  the full dense role x attribute table.

Only official-train scaffold folds are used for tuning and model selection.
The official validation/test splits are not touched by this entry point.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import yaml
from sklearn.metrics import roc_auc_score

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.cross_center_interaction_screen import (
    _frozen_s_rows,
    _resolve,
    _sha256,
)
from tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal import (
    _official_train_scaffold_splits,
)
from tracks.ksvd.experiments.luyin16.task_aligned_interaction_screen import (
    XGB_SEARCH_KEYS,
    _fit_xgb_predict,
    _suggest_xgb_params,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/invariant_conditional_interaction_screen.yaml"
VIEW_NAMES = ("s_marginal", "s_invariant_cross", "s_conditional_binding", "s_invariant_both")

# The cached interaction encoder uses these fixed semantic widths.
NODE_ROLE_WIDTH = 64
EDGE_ROLE_WIDTH = 32
NODE_ATTRIBUTE_WIDTH = 40
EDGE_ATTRIBUTE_WIDTH = 13
CROSS_TOPOLOGY_WIDTH = NODE_ROLE_WIDTH + EDGE_ROLE_WIDTH
CROSS_ATTRIBUTE_WIDTH = NODE_ATTRIBUTE_WIDTH + EDGE_ATTRIBUTE_WIDTH
BINDING_WIDTH = NODE_ROLE_WIDTH * NODE_ATTRIBUTE_WIDTH + EDGE_ROLE_WIDTH * EDGE_ATTRIBUTE_WIDTH


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _strip_search_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: params[key] for key in XGB_SEARCH_KEYS if key in params}


def _in_range(params: Mapping[str, Any], ranges: Mapping[str, Sequence[float]]) -> bool:
    return all(
        key in params
        and float(ranges[key][0]) <= float(params[key]) <= float(ranges[key][1])
        for key in XGB_SEARCH_KEYS
    )


def _spectral_summary(matrix: np.ndarray, top_k: int) -> np.ndarray:
    """Compact summary whose main coordinates are singular-value invariants."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"matrix summary expects 2-D input, got {values.shape}")
    if values.size == 0:
        return np.zeros(21 + int(top_k), dtype=np.float32)
    values = np.nan_to_num(values, copy=False)
    singular = np.linalg.svd(values, compute_uv=False).astype(np.float64, copy=False)
    energy = float(np.dot(singular, singular))
    root_energy = float(np.sqrt(max(energy, 0.0)))
    probabilities = (singular * singular) / max(energy, 1.0e-12)
    probabilities = probabilities[probabilities > 1.0e-12]
    spectral_entropy = float(-np.sum(probabilities * np.log(probabilities))) if probabilities.size else 0.0
    effective_rank = float(np.exp(spectral_entropy))
    nonzero = singular[singular > 1.0e-10]
    normalized = singular / max(root_energy, 1.0e-12)
    top3_energy = float(np.sum(probabilities[:3])) if probabilities.size else 0.0
    top5_energy = float(np.sum(probabilities[:5])) if probabilities.size else 0.0
    top10_energy = float(np.sum(probabilities[:10])) if probabilities.size else 0.0
    # Every quantity below is a function of singular values only.  Therefore
    # the complete block is invariant to orthogonal changes of basis on either
    # the topology or attribute coordinates; unlike PCA coordinates, its
    # meaning cannot rotate when the fitting scope changes.
    output = [
        root_energy,
        float(energy / max(values.size, 1)),
        float(singular.sum()),
        float(singular.mean() if singular.size else 0.0),
        float(singular.std() if singular.size else 0.0),
        float(singular[0] if singular.size else 0.0),
        float((singular[0] / max(root_energy, 1.0e-12)) if singular.size else 0.0),
        float(spectral_entropy),
        effective_rank,
        float(1.0 / np.sum(probabilities * probabilities)) if probabilities.size else 0.0,
        float(nonzero.size),
        float(nonzero[-1] if nonzero.size else 0.0),
        float(nonzero[-1] / max(root_energy, 1.0e-12)) if nonzero.size else 0.0,
        top3_energy,
        top5_energy,
        top10_energy,
        float((singular[0] / max(singular.sum(), 1.0e-12)) if singular.size else 0.0),
        float((singular[:3].sum() / max(singular.sum(), 1.0e-12)) if singular.size else 0.0),
        float((singular[:5].sum() / max(singular.sum(), 1.0e-12)) if singular.size else 0.0),
        float((singular[:10].sum() / max(singular.sum(), 1.0e-12)) if singular.size else 0.0),
        float((singular.sum() / max(root_energy, 1.0e-12)) if singular.size else 0.0),
    ]
    output.extend(float(value) for value in normalized[: int(top_k)])
    if normalized.size < int(top_k):
        output.extend([0.0] * (int(top_k) - normalized.size))
    return np.asarray(output, dtype=np.float32)


def _binding_conditional_summary(matrix: np.ndarray, top_k: int) -> np.ndarray:
    """Summarize role-conditioned residuals without retaining dense coordinates.

    For each role x attribute matrix we retain role/attribute concentration,
    signed and absolute mass, and a few sorted role strengths.  Sorting the
    strengths makes the summary insensitive to a permutation of hash bins,
    while the entropy terms retain a direct interpretation as conditional
    concentration.
    """
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"binding summary expects 2-D input, got {values.shape}")
    if values.size == 0:
        return np.zeros(21 + int(top_k), dtype=np.float32)
    values = np.nan_to_num(values, copy=False).astype(np.float64, copy=False)
    absolute = np.abs(values)
    row_strength = np.linalg.norm(values, axis=1)
    col_strength = np.linalg.norm(values, axis=0)
    row_mass = absolute.sum(axis=1)
    col_mass = absolute.sum(axis=0)

    def _entropy(vector: np.ndarray) -> float:
        total = float(vector.sum())
        if total <= 1.0e-12:
            return 0.0
        p = vector / total
        return float(-np.sum(p[p > 1.0e-12] * np.log(p[p > 1.0e-12])))

    def _stats(vector: np.ndarray) -> list[float]:
        if vector.size == 0:
            return [0.0] * 4
        total = float(vector.sum())
        sorted_values = np.sort(vector)
        return [
            float(vector.mean()),
            float(vector.std()),
            float(vector.max(initial=0.0)),
            float(sorted_values[-min(3, vector.size) :].sum() / max(total, 1.0e-12)),
        ]

    row_mass_total = float(row_mass.sum())
    col_mass_total = float(col_mass.sum())
    row_attr_entropy = np.asarray(
        [_entropy(absolute[index]) for index in range(absolute.shape[0])], dtype=np.float64
    )
    col_role_entropy = np.asarray(
        [_entropy(absolute[:, index]) for index in range(absolute.shape[1])], dtype=np.float64
    )
    role_top = np.sort(row_strength)[::-1][: int(top_k)]
    role_top = role_top / max(float(row_strength.sum()), 1.0e-12)
    output = [
        float(values.sum()),
        float(absolute.sum()),
        float(np.linalg.norm(values)),
        float(values.mean()),
        float(values.std()),
        _entropy(row_mass),
        _entropy(col_mass),
        float(row_mass.max(initial=0.0) / max(row_mass_total, 1.0e-12)),
        float(col_mass.max(initial=0.0) / max(col_mass_total, 1.0e-12)),
        float(row_attr_entropy.mean()),
        float(row_attr_entropy.std()),
        float(col_role_entropy.mean()),
        float(col_role_entropy.std()),
    ]
    output.extend(_stats(row_strength))
    output.extend(_stats(col_strength))
    output.extend(float(value) for value in role_top)
    if role_top.size < int(top_k):
        output.extend([0.0] * (int(top_k) - role_top.size))
    return np.asarray(output, dtype=np.float32)


def _cross_blocks(row: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(row, dtype=np.float32)
    expected = CROSS_TOPOLOGY_WIDTH * CROSS_ATTRIBUTE_WIDTH
    if values.size != expected:
        raise ValueError(f"cross row width {values.size} != {expected}")
    full = values.reshape(CROSS_TOPOLOGY_WIDTH, CROSS_ATTRIBUTE_WIDTH)
    return [
        full[:NODE_ROLE_WIDTH, :NODE_ATTRIBUTE_WIDTH],
        full[:NODE_ROLE_WIDTH, NODE_ATTRIBUTE_WIDTH:],
        full[NODE_ROLE_WIDTH:, :NODE_ATTRIBUTE_WIDTH],
        full[NODE_ROLE_WIDTH:, NODE_ATTRIBUTE_WIDTH:],
    ]


def _binding_blocks(row: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(row, dtype=np.float32)
    if values.size != 2 * BINDING_WIDTH:
        raise ValueError(f"binding row width {values.size} != {2 * BINDING_WIDTH}")
    mean = values[:BINDING_WIDTH]
    std = values[BINDING_WIDTH:]
    return [
        mean[: NODE_ROLE_WIDTH * NODE_ATTRIBUTE_WIDTH].reshape(NODE_ROLE_WIDTH, NODE_ATTRIBUTE_WIDTH),
        mean[NODE_ROLE_WIDTH * NODE_ATTRIBUTE_WIDTH :].reshape(EDGE_ROLE_WIDTH, EDGE_ATTRIBUTE_WIDTH),
        std[: NODE_ROLE_WIDTH * NODE_ATTRIBUTE_WIDTH].reshape(NODE_ROLE_WIDTH, NODE_ATTRIBUTE_WIDTH),
        std[NODE_ROLE_WIDTH * NODE_ATTRIBUTE_WIDTH :].reshape(EDGE_ROLE_WIDTH, EDGE_ATTRIBUTE_WIDTH),
    ]


def _build_summary_features(
    cross: np.ndarray,
    binding: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cross_values = np.asarray(cross, dtype=np.float32)
    binding_values = np.asarray(binding, dtype=np.float32)
    if cross_values.ndim != 2 or binding_values.ndim != 2 or cross_values.shape[0] != binding_values.shape[0]:
        raise ValueError(f"unaligned interaction arrays: {cross_values.shape}, {binding_values.shape}")

    cross_rows: list[np.ndarray] = []
    binding_rows: list[np.ndarray] = []
    for cross_row, binding_row in zip(cross_values, binding_values, strict=True):
        cross_rows.append(np.concatenate([_spectral_summary(block, top_k) for block in _cross_blocks(cross_row)]))
        binding_rows.append(
            np.concatenate([_binding_conditional_summary(block, top_k) for block in _binding_blocks(binding_row)])
        )
    cross_out = np.stack(cross_rows, axis=0).astype(np.float32, copy=False)
    binding_out = np.stack(binding_rows, axis=0).astype(np.float32, copy=False)
    return cross_out, binding_out, {
        "cross_blocks": ["node_role_x_node_attribute", "node_role_x_edge_attribute", "edge_role_x_node_attribute", "edge_role_x_edge_attribute"],
        "binding_blocks": ["node_mean", "edge_mean", "node_std", "edge_std"],
        "top_k_singular_or_role_strength": int(top_k),
        "cross_dimension": int(cross_out.shape[1]),
        "binding_dimension": int(binding_out.shape[1]),
    }


def _load_split_indices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        original = np.asarray(archive["original_indices"], dtype=np.int64)
        train = original[np.asarray(archive["official_train_indices"], dtype=np.int64)]
        valid = original[np.asarray(archive["official_valid_indices"], dtype=np.int64)]
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    return train, valid, fold_archive


def _fit_params(
    fold_matrices: Sequence[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    tuning: Mapping[str, Any],
    classifier: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Equal-budget Optuna search on official-train scaffold folds."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
    ranges = tuning["ranges"]
    fixed = _strip_search_params(classifier)
    if _in_range(fixed, ranges):
        study.enqueue_trial(fixed)
    for candidate in tuning.get("warm_start_params", []):
        candidate = _strip_search_params(candidate)
        if _in_range(candidate, ranges):
            study.enqueue_trial(candidate)

    def score(params: Mapping[str, Any]) -> list[float]:
        values: list[float] = []
        y = np.asarray(labels, dtype=np.int64)
        for (x_train, x_valid), (train, valid) in zip(fold_matrices, splits, strict=True):
            pred = _fit_xgb_predict(x_train, y[train], x_valid, params)
            values.append(float(roc_auc_score(y[valid], pred)))
        return values

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial, tuning, seed=seed)
        try:
            return float(np.mean(score(params)))
        finally:
            gc.collect()

    study.optimize(objective, n_trials=int(tuning["n_trials"]), show_progress_bar=False)
    best = dict(study.best_trial.params)
    best.update({"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "random_state": 0, "n_jobs": int(tuning.get("n_jobs", -1))})
    fold_auc = score(best)
    return {
        "n_trials": int(tuning["n_trials"]),
        "best_trial": int(study.best_trial.number),
        "best_scaffold_auc": float(np.mean(fold_auc)),
        "best_scaffold_fold_auc": fold_auc,
        "best_params": best,
        "trials": [
            {"trial": int(trial.number), "value": None if trial.value is None else float(trial.value), "params": dict(trial.params), "state": str(trial.state)}
            for trial in study.trials
        ],
    }


def _fixed_params(classifier: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(classifier)
    params.update({"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "random_state": 0})
    return params


def _score_view(
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    params: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold, (train, valid) in enumerate(splits):
        fold_rows: list[float] = []
        for seed in seeds:
            current = dict(params)
            current["random_state"] = int(seed)
            prediction = _fit_xgb_predict(matrix[train], labels[train], matrix[valid], current)
            fold_rows.append(float(roc_auc_score(labels[valid], prediction)))
        rows.append({"fold": int(fold), "seed_auc": fold_rows, "mean_auc": float(np.mean(fold_rows)), "std_auc": float(np.std(fold_rows))})
    means = [row["mean_auc"] for row in rows]
    return {"folds": rows, "mean_auc": float(np.mean(means)), "fold_std": float(np.std(means))}


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    train_indices, valid_indices, fold_archive = _load_split_indices(_resolve(data["scaffold_folds"]))
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    labels = np.asarray(bundle.y, dtype=np.int64)
    if not np.array_equal(np.sort(bundle.split["train"]), np.sort(train_indices)) or not np.array_equal(np.sort(bundle.split["valid"]), np.sort(valid_indices)):
        raise RuntimeError("scaffold archive does not match OGB official train/valid")
    dev_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64, copy=False)
    with np.load(_resolve(data["frozen_features"]), allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(_resolve(data["interaction_cache"]), allow_pickle=False) as archive:
        cache = {name: np.asarray(archive[name]) for name in archive.files if name != "signature"}
    cache_indices = np.asarray(cache["dataset_indices"], dtype=np.int64)
    row_by_index = {int(index): position for position, index in enumerate(cache_indices)}
    dev_rows = np.asarray([row_by_index[int(index)] for index in dev_indices], dtype=np.int64)
    if not np.array_equal(cache_indices[dev_rows], dev_indices):
        raise RuntimeError("interaction cache row mapping mismatch")
    s = _frozen_s_rows(frozen, dev_indices)
    base = np.concatenate([s, cache["marginal"][dev_rows], cache["context"][dev_rows]], axis=1).astype(np.float32, copy=False)
    cross_summary, binding_summary, summary_meta = _build_summary_features(
        cache["cross_cov"][dev_rows], cache["binding"][dev_rows], top_k=int(config["representation"].get("summary_top_k", 8))
    )
    matrices = {
        "s_marginal": base,
        "s_invariant_cross": np.concatenate([base, cross_summary], axis=1),
        "s_conditional_binding": np.concatenate([base, binding_summary], axis=1),
        "s_invariant_both": np.concatenate([base, cross_summary, binding_summary], axis=1),
    }
    splits, split_meta = _official_train_scaffold_splits(fold_archive, train_indices)
    local_splits = splits
    train_labels = labels[train_indices]
    # Convert matrices to official-train row space; valid is intentionally not
    # included in any search matrix.
    train_matrix = {name: matrix[: train_indices.size] for name, matrix in matrices.items()}
    classifier = config["classifier"]
    tuning = config["tuning"]
    seeds = [int(value) for value in config.get("model_seeds", [0, 1, 2])]
    results: dict[str, Any] = {}
    start = time.perf_counter()
    for view_id, view in enumerate(VIEW_NAMES):
        search = _fit_params(
            [(train_matrix[view][tr], train_matrix[view][va]) for tr, va in local_splits],
            train_labels,
            local_splits,
            tuning,
            classifier,
            seed=int(tuning["seed"]) + 997 * view_id,
        )
        fixed = _score_view(train_matrix[view], train_labels, local_splits, _fixed_params(classifier), seeds)
        tuned = _score_view(train_matrix[view], train_labels, local_splits, search["best_params"], seeds)
        results[view] = {
            "dimension": int(matrices[view].shape[1]),
            "search": search,
            "fixed_scaffold": fixed,
            "tuned_scaffold": tuned,
        }
        print(f"{view}: tuned={tuned['mean_auc']:.6f}; fixed={fixed['mean_auc']:.6f}; dim={matrices[view].shape[1]}", flush=True)
    selected = max(VIEW_NAMES, key=lambda name: (results[name]["search"]["best_scaffold_auc"], int(name == "s_invariant_both")))
    output = {
        "protocol_id": config["protocol_id"],
        "stage": "official_train_scaffold_search_only",
        "official_validation_or_test_evaluated": False,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(Path(__file__)),
        "data": {"dataset": data["dataset"], "n_train": int(train_indices.size), "n_valid": int(valid_indices.size), "interaction_cache": str(_resolve(data["interaction_cache"])), "scaffold_folds": str(_resolve(data["scaffold_folds"])), "scaffold_folds_sha256": _sha256(_resolve(data["scaffold_folds"])), "split_meta": split_meta},
        "representation": {"base": "frozen S + marginal + context", "summary_meta": summary_meta, "invariant_definition": "singular values/energy/entropy/norm concentration per topology-attribute block", "conditional_definition": "role-conditioned binding residual concentration per mean/std block"},
        "model_seeds": seeds,
        "views": results,
        "selection": {"rule": "highest mean official-train scaffold CV; s_invariant_both wins ties", "selected_view": selected, "candidate_views": list(VIEW_NAMES)},
        "runtime": {"seconds": float(time.perf_counter() - start), "python": platform.python_version(), "platform": platform.platform()},
    }
    output_json = _resolve(config["output_json"])
    output_md = _resolve(config["output_markdown"])
    _write_json_atomic(output_json, output)
    lines = ["# Invariant/conditional centre-interaction screen", "", f"Protocol: `{config['protocol_id']}`", "", "Official validation/test were not encoded or evaluated.", "", "| view | dimension | tuned scaffold mean | fixed scaffold mean |", "|---|---:|---:|---:|"]
    for view in VIEW_NAMES:
        row = results[view]
        lines.append(f"| `{view}` | {row['dimension']} | {row['search']['best_scaffold_auc']:.6f} | {row['fixed_scaffold']['mean_auc']:.6f} |")
    lines.extend(["", f"Selected by train-only scaffold CV: `{selected}`.", "", "The invariant block uses no PCA coordinates; the conditional block keeps low-dimensional role-conditioned concentration statistics."])
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    print(json.dumps({"protocol_id": result["protocol_id"], "selected_view": result["selection"]["selected_view"], "views": {name: {"tuned_scaffold_auc": result["views"][name]["search"]["best_scaffold_auc"], "dimension": result["views"][name]["dimension"]} for name in VIEW_NAMES}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
