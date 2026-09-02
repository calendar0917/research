"""Downstream test after the shared-support mechanism gate has passed.

The only learned graph classifier is a standardized logistic regression.  The
experiment therefore tests the patch representation rather than whether a
larger GNN/Transformer can hide a weak representation.
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

from .coupled_support_dictionary import coupled_sparse_encode, fit_coupled_support_dictionary
from .ksvd import _omp, ksvd
from .dual_dictionary import relation_readout
from .run_molhiv_coupled_support_gate import (
    _collect,
    _columns,
    _derangement,
    _fit_scaler,
    _resolve,
    _scale,
)
from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig


METHODS = (
    "raw_two_view",
    "pca_two_view",
    "independent_ksvd",
    "concatenated_shared_coefficient",
    "coupled_shared_support",
    "coupled_inference_binding_shuffled",
    "coupled_training_binding_shuffled",
    "raw_plus_coupled_shared_support",
    "raw_plus_coupled_inference_binding_shuffled",
    "raw_plus_coupled_training_binding_shuffled",
)


def _codes(dictionary: np.ndarray, values: np.ndarray, sparsity: int) -> np.ndarray:
    return np.stack([_omp(dictionary, values[:, i], sparsity) for i in range(values.shape[1])], axis=1)


def _scaled_pairs(records, scaler):
    pairs = []
    for record in records:
        pairs.append((
            _scale(record.structure, scaler["structure_mean"], float(scaler["structure_rms"])),
            _scale(record.chemistry, scaler["chemistry_mean"], float(scaler["chemistry_rms"])),
        ))
    return pairs


def _two_readouts(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([relation_readout(left), relation_readout(right)])


def _features(pairs, method: str, models: dict[str, object], sparsity: int, seed: int) -> np.ndarray:
    rows = []
    for index, (S, C) in enumerate(pairs):
        raw = _two_readouts(S, C)
        if method == "raw_two_view":
            row = raw
        elif method == "pca_two_view":
            ps = models["pca_structure"].transform(S.T).T
            pc = models["pca_chemistry"].transform(C.T).T
            row = _two_readouts(ps, pc)
        elif method == "independent_ksvd":
            row = _two_readouts(
                _codes(models["ind_structure"], S, sparsity),
                _codes(models["ind_chemistry"], C, sparsity),
            )
        elif method == "concatenated_shared_coefficient":
            z = _codes(models["concat"], np.vstack([S, C]), sparsity)
            # Duplicate only for equal reader width; it adds no information.
            row = _two_readouts(z, z)
        elif method in {
            "coupled_shared_support", "coupled_inference_binding_shuffled", "coupled_training_binding_shuffled",
            "raw_plus_coupled_shared_support", "raw_plus_coupled_inference_binding_shuffled", "raw_plus_coupled_training_binding_shuffled",
        }:
            suffix = method.removeprefix("raw_plus_")
            ds, dc = (
                (models["coupled_structure"], models["coupled_chemistry"])
                if suffix != "coupled_training_binding_shuffled"
                else (models["shuffle_structure"], models["shuffle_chemistry"])
            )
            c_input = C
            if suffix == "coupled_inference_binding_shuffled":
                c_input = C[:, _derangement(C.shape[1], seed + index * 1009)]
            zs, zc = coupled_sparse_encode(S, c_input, ds, dc, sparsity)
            coupled = _two_readouts(zs, zc)
            row = np.concatenate([raw, coupled]) if method.startswith("raw_plus_") else coupled
        else:  # pragma: no cover - parser and METHODS protect this
            raise ValueError(method)
        rows.append(row)
    return np.stack(rows)


def _labels(records) -> np.ndarray:
    return np.asarray([record.label for record in records], dtype=np.float64)


def _score(train_x, train_y, valid_x, valid_y, c: float) -> dict[str, float | int]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=0, solver="liblinear"),
    )
    model.fit(train_x, train_y)
    prob = model.predict_proba(valid_x)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(valid_y, prob)),
        "average_precision": float(average_precision_score(valid_y, prob)),
        "n_features": int(train_x.shape[1]),
    }


def _one_fold(bundle, folds, args, fold: int) -> dict[str, object]:
    train_idx = folds[f"fold_{fold}_train_indices"]
    valid_idx = folds[f"fold_{fold}_valid_indices"]
    cfg = GraphLevelConfig(seed=args.seed, max_patches_per_graph=8, normalize_patches=False)
    train = _collect(bundle, train_idx, cfg, args.fit_graphs, args.seed + 101)
    valid = _collect(bundle, valid_idx, cfg, args.eval_graphs, args.seed + 202)
    S_raw, C_raw = _columns(train)
    scaler = _fit_scaler(S_raw, C_raw)
    train_pairs, valid_pairs = _scaled_pairs(train, scaler), _scaled_pairs(valid, scaler)
    S_train = np.concatenate([pair[0] for pair in train_pairs], axis=1)
    C_train = np.concatenate([pair[1] for pair in train_pairs], axis=1)

    ds_ind, _, ind_s_info = ksvd(S_train, n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=args.seed)
    dc_ind, _, ind_c_info = ksvd(C_train, n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=args.seed + 1)
    d_concat, _, concat_info = ksvd(np.vstack([S_train, C_train]), n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=args.seed + 2)
    split = S_train.shape[0]
    ds_init, dc_init = d_concat[:split], d_concat[split:]
    ds_init = ds_init / np.maximum(np.linalg.norm(ds_init, axis=0, keepdims=True), 1e-12)
    dc_init = dc_init / np.maximum(np.linalg.norm(dc_init, axis=0, keepdims=True), 1e-12)
    ds_coupled, dc_coupled, coupled_info = fit_coupled_support_dictionary(
        S_train, C_train, n_atoms=args.n_atoms, sparsity=args.sparsity, n_iter=args.ksvd_iter,
        seed=args.seed + 3, initial_structure=ds_init, initial_chemistry=dc_init,
    )
    ds_shuffle, dc_shuffle, shuffled_info = fit_coupled_support_dictionary(
        S_train, C_train[:, _derangement(C_train.shape[1], args.seed + 404)],
        n_atoms=args.n_atoms, sparsity=args.sparsity, n_iter=args.ksvd_iter,
        seed=args.seed + 3, initial_structure=ds_init, initial_chemistry=dc_init,
    )
    n_components_s = min(args.n_atoms, S_train.shape[0], S_train.shape[1])
    n_components_c = min(args.n_atoms, C_train.shape[0], C_train.shape[1])
    models: dict[str, object] = {
        "ind_structure": ds_ind, "ind_chemistry": dc_ind, "concat": d_concat,
        "coupled_structure": ds_coupled, "coupled_chemistry": dc_coupled,
        "shuffle_structure": ds_shuffle, "shuffle_chemistry": dc_shuffle,
        "pca_structure": PCA(n_components=n_components_s, random_state=args.seed).fit(S_train.T),
        "pca_chemistry": PCA(n_components=n_components_c, random_state=args.seed).fit(C_train.T),
    }
    y_train, y_valid = _labels(train), _labels(valid)
    scores = {}
    for method in METHODS:
        train_x = _features(train_pairs, method, models, args.sparsity, args.seed + 505)
        valid_x = _features(valid_pairs, method, models, args.sparsity, args.seed + 606)
        scores[method] = _score(train_x, y_train, valid_x, y_valid, args.classifier_c)
    return {
        "fold": fold,
        "n_train_graphs": len(train), "n_valid_graphs": len(valid),
        "n_train_patches": int(S_train.shape[1]),
        "fit": {"independent_structure": ind_s_info, "independent_chemistry": ind_c_info, "concatenated": concat_info, "coupled": coupled_info, "coupled_shuffled_training": shuffled_info},
        "scores": scores,
        "paired": {
            "coupled_minus_concat_roc_auc": scores["coupled_shared_support"]["roc_auc"] - scores["concatenated_shared_coefficient"]["roc_auc"],
            "coupled_minus_independent_roc_auc": scores["coupled_shared_support"]["roc_auc"] - scores["independent_ksvd"]["roc_auc"],
            "coupled_minus_pca_roc_auc": scores["coupled_shared_support"]["roc_auc"] - scores["pca_two_view"]["roc_auc"],
            "coupled_minus_inference_shuffle_roc_auc": scores["coupled_shared_support"]["roc_auc"] - scores["coupled_inference_binding_shuffled"]["roc_auc"],
            "coupled_minus_training_shuffle_roc_auc": scores["coupled_shared_support"]["roc_auc"] - scores["coupled_training_binding_shuffled"]["roc_auc"],
            "raw_plus_coupled_minus_raw_roc_auc": scores["raw_plus_coupled_shared_support"]["roc_auc"] - scores["raw_two_view"]["roc_auc"],
            "raw_plus_coupled_minus_inference_shuffle_roc_auc": scores["raw_plus_coupled_shared_support"]["roc_auc"] - scores["raw_plus_coupled_inference_binding_shuffled"]["roc_auc"],
            "raw_plus_coupled_minus_training_shuffle_roc_auc": scores["raw_plus_coupled_shared_support"]["roc_auc"] - scores["raw_plus_coupled_training_binding_shuffled"]["roc_auc"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz")
    parser.add_argument("--fold", type=int, default=None, help="one fold only; default runs all three")
    parser.add_argument("--fit-graphs", type=int, default=999999)
    parser.add_argument("--eval-graphs", type=int, default=999999)
    parser.add_argument("--n-atoms", type=int, default=12)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--ksvd-iter", type=int, default=4)
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
    paired = {
        name: float(np.mean([item["paired"][name] for item in output_folds]))
        for name in output_folds[0]["paired"]
    }
    result = {
        "protocol_id": "molhiv-coupled-support-classification-v1",
        "config": vars(args),
        "design": {
            "classifier": "the same standardized logistic regression for every method",
            "proposal": "paired KSVD atoms with shared support and view-specific coefficients",
            "test_time_binding_control": "all patch bags unchanged; only which chemical patch is paired to each structural patch is deranged before joint encoding",
            "train_time_binding_control": "the same no-fixed-point derangement is used only while fitting the paired dictionary",
            "additive_test": "raw-plus variants append only the paired-dictionary readout to the identical raw two-view readout; they test incremental value without replacing raw chemistry",
            "leakage_guard": "scalers, PCA, dictionaries, and classifier fit on the training part of each scaffold fold only",
        },
        "folds": output_folds,
        "summary": summary,
        "paired_summary": paired,
        "elapsed_sec": time.time() - started,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_summary": paired}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
