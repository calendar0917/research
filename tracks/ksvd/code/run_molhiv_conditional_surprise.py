"""Cross-view conditional-surprise gate for MolHIV patch representations.

Unlike ordinary reconstruction dictionaries, this route first removes the
chemistry that is predictable from a patch's topology (and vice versa).  A
dictionary then represents only the remaining, paired local exception.  The
classifier always retains the identical raw patch readout, so this is a true
incremental-value test rather than a compression replacement test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .conditional_surprise import (
    CrossViewPredictors,
    cross_view_residuals,
    fit_cross_view_predictors,
    normalize_residual_columns,
    residual_error,
)
from .coupled_support_dictionary import coupled_sparse_encode, fit_coupled_support_dictionary
from .data_molhiv import load_molhiv
from .dual_dictionary import relation_readout
from .graph_level import GraphLevelConfig
from .ksvd import _omp, ksvd
from .run_molhiv_coupled_support_gate import (
    _collect,
    _columns,
    _derangement,
    _fit_scaler,
    _resolve,
    _scale,
)


METHODS = (
    "raw_two_view",
    "raw_plus_uncompressed_surprise",
    "raw_plus_pca_surprise",
    "raw_plus_random_surprise",
    "raw_plus_independent_ksvd_surprise",
    "raw_plus_coupled_ksvd_surprise",
    "raw_plus_coupled_ksvd_inference_shuffled",
    "raw_plus_coupled_ksvd_training_shuffled",
)


def _codes(dictionary: np.ndarray, values: np.ndarray, sparsity: int) -> np.ndarray:
    return np.stack([_omp(dictionary, values[:, i], sparsity) for i in range(values.shape[1])], axis=1)


def _random_dictionary(values: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = rng.choice(values.shape[1], size=min(n_atoms, values.shape[1]), replace=False)
    out = values[:, selected].copy()
    return out / np.maximum(np.linalg.norm(out, axis=0, keepdims=True), 1e-12)


def _scaled_pairs(records, scaler):
    return [
        (
            _scale(record.structure, scaler["structure_mean"], float(scaler["structure_rms"])),
            _scale(record.chemistry, scaler["chemistry_mean"], float(scaler["chemistry_rms"])),
        )
        for record in records
    ]


def _raw_readout(structure: np.ndarray, chemistry: np.ndarray) -> np.ndarray:
    return np.concatenate([relation_readout(structure), relation_readout(chemistry)])


def _norm_summary(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=0)
    if not len(norms):
        return np.zeros(4, dtype=np.float64)
    return np.asarray([norms.mean(), norms.std(), norms.max(), norms.min()], dtype=np.float64)


def _residual_readout(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([
        relation_readout(left), relation_readout(right), _norm_summary(left), _norm_summary(right)
    ])


def _residual_pairs(pairs, predictors: CrossViewPredictors):
    return [cross_view_residuals(S, C, predictors) for S, C in pairs]


def _normalized_columns(residual_pairs) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.concatenate([x[0] for x in residual_pairs], axis=1), np.concatenate([x[1] for x in residual_pairs], axis=1)
    nleft, _ = normalize_residual_columns(left)
    nright, _ = normalize_residual_columns(right)
    return nleft, nright


def _residual_feature_pairs(residual_pairs, encoder_left, encoder_right):
    rows = []
    for left, right in residual_pairs:
        nleft, _ = normalize_residual_columns(left)
        nright, _ = normalize_residual_columns(right)
        rows.append(_residual_readout(encoder_left(nleft), encoder_right(nright)))
    return np.stack(rows)


def _labels(records) -> np.ndarray:
    return np.asarray([record.label for record in records], dtype=np.float64)


def _score(train_x, train_y, valid_x, valid_y, c: float) -> dict[str, float | int]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=0, solver="liblinear"),
    )
    model.fit(train_x, train_y)
    probability = model.predict_proba(valid_x)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(valid_y, probability)),
        "average_precision": float(average_precision_score(valid_y, probability)),
        "n_features": int(train_x.shape[1]),
    }


def _fit_models(true_residuals, shuffled_residuals, args):
    left, right = _normalized_columns(true_residuals)
    shuffle_left, shuffle_right = _normalized_columns(shuffled_residuals)
    pca_left = PCA(n_components=min(args.n_atoms, left.shape[0], left.shape[1]), random_state=args.seed).fit(left.T)
    pca_right = PCA(n_components=min(args.n_atoms, right.shape[0], right.shape[1]), random_state=args.seed).fit(right.T)
    ind_left, _, ind_left_info = ksvd(left, n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=args.seed)
    ind_right, _, ind_right_info = ksvd(right, n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=args.seed + 1)
    concat, _, concat_info = ksvd(np.vstack([left, right]), n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=args.seed + 2)
    split = left.shape[0]
    initial_left, initial_right = concat[:split], concat[split:]
    initial_left /= np.maximum(np.linalg.norm(initial_left, axis=0, keepdims=True), 1e-12)
    initial_right /= np.maximum(np.linalg.norm(initial_right, axis=0, keepdims=True), 1e-12)
    coupled_left, coupled_right, coupled_info = fit_coupled_support_dictionary(
        left, right, n_atoms=args.n_atoms, sparsity=args.sparsity, n_iter=args.ksvd_iter,
        seed=args.seed + 3, initial_structure=initial_left, initial_chemistry=initial_right,
    )
    shuffled_left_dict, shuffled_right_dict, shuffled_info = fit_coupled_support_dictionary(
        shuffle_left, shuffle_right, n_atoms=args.n_atoms, sparsity=args.sparsity, n_iter=args.ksvd_iter,
        seed=args.seed + 3, initial_structure=initial_left, initial_chemistry=initial_right,
    )
    return {
        "pca_left": pca_left, "pca_right": pca_right,
        "random_left": _random_dictionary(left, args.n_atoms, args.seed + 10),
        "random_right": _random_dictionary(right, args.n_atoms, args.seed + 11),
        "ind_left": ind_left, "ind_right": ind_right,
        "coupled_left": coupled_left, "coupled_right": coupled_right,
        "shuffle_left": shuffled_left_dict, "shuffle_right": shuffled_right_dict,
        "fit": {
            "independent_left": ind_left_info, "independent_right": ind_right_info,
            "concatenated_initialization": concat_info, "coupled": coupled_info,
            "coupled_shuffled_training": shuffled_info,
        },
    }


def _feature_matrix(raw_pairs, residual_pairs, method: str, models, args) -> np.ndarray:
    rows = []
    for (S, C), (Rs, Rc) in zip(raw_pairs, residual_pairs):
        raw = _raw_readout(S, C)
        if method == "raw_two_view":
            row = raw
        elif method == "raw_plus_uncompressed_surprise":
            row = np.concatenate([raw, _residual_readout(Rs, Rc)])
        else:
            nleft, _ = normalize_residual_columns(Rs)
            nright, _ = normalize_residual_columns(Rc)
            if method == "raw_plus_pca_surprise":
                extra = _residual_readout(
                    models["pca_left"].transform(nleft.T).T,
                    models["pca_right"].transform(nright.T).T,
                )
            elif method == "raw_plus_random_surprise":
                extra = _residual_readout(
                    _codes(models["random_left"], nleft, args.sparsity),
                    _codes(models["random_right"], nright, args.sparsity),
                )
            elif method == "raw_plus_independent_ksvd_surprise":
                extra = _residual_readout(
                    _codes(models["ind_left"], nleft, args.sparsity),
                    _codes(models["ind_right"], nright, args.sparsity),
                )
            elif method in {"raw_plus_coupled_ksvd_surprise", "raw_plus_coupled_ksvd_inference_shuffled", "raw_plus_coupled_ksvd_training_shuffled"}:
                left_dict, right_dict = (
                    (models["shuffle_left"], models["shuffle_right"])
                    if method == "raw_plus_coupled_ksvd_training_shuffled"
                    else (models["coupled_left"], models["coupled_right"])
                )
                zs, zc = coupled_sparse_encode(nleft, nright, left_dict, right_dict, args.sparsity)
                extra = _residual_readout(zs, zc)
            else:  # pragma: no cover
                raise ValueError(method)
            row = np.concatenate([raw, extra])
        rows.append(row)
    return np.stack(rows)


def _one_fold(bundle, folds, args, fold: int) -> dict[str, object]:
    train_idx, valid_idx = folds[f"fold_{fold}_train_indices"], folds[f"fold_{fold}_valid_indices"]
    cfg = GraphLevelConfig(seed=args.seed, max_patches_per_graph=8, normalize_patches=False)
    train = _collect(bundle, train_idx, cfg, args.fit_graphs, args.seed + 101)
    valid = _collect(bundle, valid_idx, cfg, args.eval_graphs, args.seed + 202)
    S_raw, C_raw = _columns(train)
    scaler = _fit_scaler(S_raw, C_raw)
    train_pairs, valid_pairs = _scaled_pairs(train, scaler), _scaled_pairs(valid, scaler)
    S_train = np.concatenate([x[0] for x in train_pairs], axis=1)
    C_train = np.concatenate([x[1] for x in train_pairs], axis=1)
    true_predictors = fit_cross_view_predictors(S_train, C_train, alpha=args.ridge_alpha)
    order = _derangement(C_train.shape[1], args.seed + 404)
    # The shuffled counterpart is formed globally, so every structure patch is
    # paired with a different chemistry patch while the two training bags stay
    # exactly unchanged.
    shuffled_predictors = fit_cross_view_predictors(S_train, C_train[:, order], alpha=args.ridge_alpha)
    true_train_residuals = _residual_pairs(train_pairs, true_predictors)
    shuffled_train_residuals = [
        cross_view_residuals(S_train, C_train[:, order], shuffled_predictors)
    ]
    # Split global shuffled residuals back to molecules; this maintains the
    # same patch counts needed by the graph-level readout.
    shuffled_train_residuals = []
    cursor = 0
    global_rs, global_rc = cross_view_residuals(S_train, C_train[:, order], shuffled_predictors)
    for S, C in train_pairs:
        width = S.shape[1]
        shuffled_train_residuals.append((global_rs[:, cursor:cursor + width], global_rc[:, cursor:cursor + width]))
        cursor += width
    models = _fit_models(true_train_residuals, shuffled_train_residuals, args)
    true_valid_residuals = _residual_pairs(valid_pairs, true_predictors)
    # Strict inference control: raw patch readout stays exactly unchanged; only
    # the paired inputs of the surprise branch are deranged within each graph.
    inference_shuffled_residuals = []
    for graph_index, (S, C) in enumerate(valid_pairs):
        wrong = C[:, _derangement(C.shape[1], args.seed + 505 + graph_index * 1009)]
        inference_shuffled_residuals.append(cross_view_residuals(S, wrong, true_predictors))
    training_shuffled_valid_residuals = _residual_pairs(valid_pairs, shuffled_predictors)

    y_train, y_valid = _labels(train), _labels(valid)
    scores = {}
    for method in METHODS:
        train_residuals = true_train_residuals
        valid_residuals = true_valid_residuals
        if method == "raw_plus_coupled_ksvd_inference_shuffled":
            valid_residuals = inference_shuffled_residuals
        elif method == "raw_plus_coupled_ksvd_training_shuffled":
            train_residuals = shuffled_train_residuals
            valid_residuals = training_shuffled_valid_residuals
        train_x = _feature_matrix(train_pairs, train_residuals, method, models, args)
        valid_x = _feature_matrix(valid_pairs, valid_residuals, method, models, args)
        scores[method] = _score(train_x, y_train, valid_x, y_valid, args.classifier_c)

    true_error = residual_error(*[np.concatenate([r[i] for r in true_valid_residuals], axis=1) for i in (0, 1)])
    wrong_error = residual_error(*[np.concatenate([r[i] for r in inference_shuffled_residuals], axis=1) for i in (0, 1)])
    shuffled_train_error = residual_error(*[np.concatenate([r[i] for r in training_shuffled_valid_residuals], axis=1) for i in (0, 1)])
    return {
        "fold": fold,
        "n_train_graphs": len(train), "n_valid_graphs": len(valid),
        "n_train_patches": int(S_train.shape[1]),
        "fit": models["fit"],
        "mechanism": {
            "true_pair_residual": true_error,
            "inference_shuffled_pair_residual": wrong_error,
            "training_shuffled_predictor_residual": shuffled_train_error,
            "inference_shuffled_minus_true_mean_mse": wrong_error["mean_mse"] - true_error["mean_mse"],
            "training_shuffled_minus_true_mean_mse": shuffled_train_error["mean_mse"] - true_error["mean_mse"],
        },
        "scores": scores,
        "paired": {
            "uncompressed_surprise_minus_raw_roc_auc": scores["raw_plus_uncompressed_surprise"]["roc_auc"] - scores["raw_two_view"]["roc_auc"],
            "coupled_surprise_minus_raw_roc_auc": scores["raw_plus_coupled_ksvd_surprise"]["roc_auc"] - scores["raw_two_view"]["roc_auc"],
            "coupled_surprise_minus_pca_roc_auc": scores["raw_plus_coupled_ksvd_surprise"]["roc_auc"] - scores["raw_plus_pca_surprise"]["roc_auc"],
            "coupled_surprise_minus_random_roc_auc": scores["raw_plus_coupled_ksvd_surprise"]["roc_auc"] - scores["raw_plus_random_surprise"]["roc_auc"],
            "coupled_surprise_minus_independent_roc_auc": scores["raw_plus_coupled_ksvd_surprise"]["roc_auc"] - scores["raw_plus_independent_ksvd_surprise"]["roc_auc"],
            "coupled_surprise_minus_inference_shuffle_roc_auc": scores["raw_plus_coupled_ksvd_surprise"]["roc_auc"] - scores["raw_plus_coupled_ksvd_inference_shuffled"]["roc_auc"],
            "coupled_surprise_minus_training_shuffle_roc_auc": scores["raw_plus_coupled_ksvd_surprise"]["roc_auc"] - scores["raw_plus_coupled_ksvd_training_shuffled"]["roc_auc"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--fit-graphs", type=int, default=999999)
    parser.add_argument("--eval-graphs", type=int, default=999999)
    parser.add_argument("--n-atoms", type=int, default=12)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--ksvd-iter", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()
    bundle = load_molhiv(with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV atom and bond features are required")
    folds = np.load(_resolve(args.fold_cache))
    fold_ids = [args.fold] if args.fold is not None else [0, 1, 2]
    output_folds = [_one_fold(bundle, folds, args, fold) for fold in fold_ids]
    summary = {
        method: {
            "roc_auc_mean": float(np.mean([item["scores"][method]["roc_auc"] for item in output_folds])),
            "roc_auc_std": float(np.std([item["scores"][method]["roc_auc"] for item in output_folds])),
            "average_precision_mean": float(np.mean([item["scores"][method]["average_precision"] for item in output_folds])),
        }
        for method in METHODS
    }
    paired = {name: float(np.mean([item["paired"][name] for item in output_folds])) for name in output_folds[0]["paired"]}
    mechanism = {name: float(np.mean([item["mechanism"][name] for item in output_folds])) for name in output_folds[0]["mechanism"] if isinstance(output_folds[0]["mechanism"][name], float)}
    result = {
        "protocol_id": "molhiv-cross-view-conditional-surprise-v1",
        "config": vars(args),
        "design": {
            "idea": "learn label-free residual patches unexplained across the true structure-chemistry pairing, then test whether a coupled KSVD of residual directions adds value beyond raw patches",
            "classifier": "identical standardized logistic regression for every method",
            "inference_shuffle": "only pairs used by the surprise branch are deranged within each graph; raw patch features remain unchanged",
            "training_shuffle": "cross-view predictors and residual dictionaries train on a no-fixed-point globally deranged training pairing",
            "leakage_guard": "all scalers, predictors, PCA, dictionaries, and classifiers fit on train-fold molecules only",
        },
        "folds": output_folds, "summary": summary, "paired_summary": paired,
        "mechanism_summary": mechanism, "elapsed_sec": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_summary": paired, "mechanism_summary": mechanism}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
