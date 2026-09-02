"""Compression gate for signed local chemical changes on MolPCBA.

This follows ``run_molpcba_local_change_gate``.  It asks a narrower question:
once real single-substituent changes have a training-only signal, does a
K-SVD code preserve that signal more effectively than equally small PCA and
random-real-change dictionaries?

It is still a paired mechanism experiment, not a claim about ordinary
single-molecule classification.  All extraction, scaling, dictionary fitting
and classifiers are confined to the training part of each internal scaffold
fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ksvd import _omp, ksvd
from .run_molpcba_local_change_gate import _fold_for_scaffold, _replace_for_task


def _scale_fit(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(values * values, axis=0) + 1e-8)


def _codes(dictionary: np.ndarray, values: np.ndarray, sparsity: int) -> np.ndarray:
    return np.stack([_omp(dictionary, row, sparsity) for row in values], axis=0)


def _fit_score(
    train_positive: np.ndarray,
    train_negative: np.ndarray,
    test_positive: np.ndarray,
    test_negative: np.ndarray,
) -> dict[str, float]:
    X_train = np.concatenate([train_positive, train_negative], axis=0)
    y_train = np.concatenate([np.ones(len(train_positive)), np.zeros(len(train_negative))])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=0))
    model.fit(X_train, y_train)
    pos_probability = model.predict_proba(test_positive)[:, 1]
    neg_probability = model.predict_proba(test_negative)[:, 1]
    y_test = np.concatenate([np.ones(len(test_positive)), np.zeros(len(test_negative))])
    probability = np.concatenate([pos_probability, neg_probability])
    return {
        "direction_accuracy": float(np.mean(pos_probability > neg_probability)),
        "paired_roc_auc": float(roc_auc_score(y_test, probability)),
    }


def _random_dictionary(values: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(values), size=min(n_atoms, len(values)), replace=False)
    dictionary = values[chosen].T.copy()
    dictionary /= np.maximum(np.linalg.norm(dictionary, axis=0, keepdims=True), 1e-12)
    return dictionary


def _relative_reconstruction(values: np.ndarray, dictionary: np.ndarray, sparsity: int) -> float:
    codes = _codes(dictionary, values, sparsity)
    reconstructed = codes @ dictionary.T
    return float(np.linalg.norm(values - reconstructed) / max(np.linalg.norm(values), 1e-12))


def _one_fold(values: np.ndarray, folds: np.ndarray, fold: int, args: argparse.Namespace) -> dict[str, object]:
    train, test = folds != fold, folds == fold
    raw_train, raw_test = values[train], values[test]
    scale = _scale_fit(raw_train)
    train_scaled, test_scaled = raw_train / scale, raw_test / scale
    pca = PCA(n_components=min(args.n_atoms, train_scaled.shape[0], train_scaled.shape[1]), random_state=args.seed).fit(train_scaled)
    random_dictionary = _random_dictionary(train_scaled, args.n_atoms, args.seed + fold)
    ksvd_dictionary, _, ksvd_info = ksvd(
        train_scaled.T,
        n_atoms=args.n_atoms,
        T=args.sparsity,
        n_iter=args.ksvd_iter,
        seed=args.seed + fold,
    )
    representations = {
        "raw_change": (train_scaled, -train_scaled, test_scaled, -test_scaled),
        "pca": (
            pca.transform(train_scaled), pca.transform(-train_scaled),
            pca.transform(test_scaled), pca.transform(-test_scaled),
        ),
        "random_real_change_prototypes": (
            _codes(random_dictionary, train_scaled, args.sparsity),
            _codes(random_dictionary, -train_scaled, args.sparsity),
            _codes(random_dictionary, test_scaled, args.sparsity),
            _codes(random_dictionary, -test_scaled, args.sparsity),
        ),
        "ksvd_change_dictionary": (
            _codes(ksvd_dictionary, train_scaled, args.sparsity),
            _codes(ksvd_dictionary, -train_scaled, args.sparsity),
            _codes(ksvd_dictionary, test_scaled, args.sparsity),
            _codes(ksvd_dictionary, -test_scaled, args.sparsity),
        ),
    }
    scores = {name: _fit_score(*arrays) for name, arrays in representations.items()}
    return {
        "fold": fold,
        "n_train_replacements": int(np.sum(train)),
        "n_test_replacements": int(np.sum(test)),
        "scores": scores,
        "reconstruction_relative": {
            "random_real_change_prototypes": _relative_reconstruction(test_scaled, random_dictionary, args.sparsity),
            "ksvd_change_dictionary": _relative_reconstruction(test_scaled, ksvd_dictionary, args.sparsity),
        },
        "ksvd_fit": ksvd_info,
    }


def _summarize(folds: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    names = list(folds[0]["scores"].keys())  # type: ignore[index]
    return {
        name: {
            metric: float(np.mean([row["scores"][name][metric] for row in folds]))  # type: ignore[index]
            for metric in ("direction_accuracy", "paired_roc_auc")
        }
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ogb/ogbg_molpcba")
    parser.add_argument("--task", type=int, default=93)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bits", type=int, default=512)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-atoms", type=int, default=32)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--ksvd-iter", type=int, default=4)
    parser.add_argument("--pair-retries", type=int, default=4)
    parser.add_argument("--max-per-scaffold", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root)
    header = pd.read_csv(root / "mapping/mol.csv.gz", nrows=0)
    label_columns = list(header.columns[:128])
    if not 0 <= args.task < len(label_columns):
        raise ValueError("task outside MolPCBA's 128 labels")
    assay = label_columns[args.task]
    table = pd.read_csv(root / "mapping/mol.csv.gz", usecols=["smiles", assay])
    train_indices = pd.read_csv(root / "split/scaffold/train.csv.gz", header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    replacements, counts = _replace_for_task(
        table, train_indices, assay,
        n_bits=args.n_bits, radius=args.radius, seed=args.seed + args.task,
        retries=args.pair_retries, max_per_scaffold=args.max_per_scaffold,
    )
    if len(replacements) < 300:
        raise RuntimeError("too few clean replacements for compression audit")
    values = np.stack([replacement.positive.fingerprint.astype(np.float64) - replacement.negative.fingerprint.astype(np.float64) for replacement in replacements])
    folds = np.asarray([_fold_for_scaffold(item.scaffold, 3) for item in replacements], dtype=np.int64)
    results = [_one_fold(values, folds, fold, args) for fold in range(3)]
    summary = _summarize(results)
    ksvd = summary["ksvd_change_dictionary"]
    pca = summary["pca"]
    random = summary["random_real_change_prototypes"]
    raw = summary["raw_change"]
    # This is intentionally demanding: K-SVD may trade a little raw accuracy
    # for compression, but it must beat both equal-sized controls and retain
    # almost all of the uncompressed directional signal.
    pass_folds = sum(
        row["scores"]["ksvd_change_dictionary"]["paired_roc_auc"]
        > row["scores"]["pca"]["paired_roc_auc"]
        and row["scores"]["ksvd_change_dictionary"]["paired_roc_auc"]
        > row["scores"]["random_real_change_prototypes"]["paired_roc_auc"]
        for row in results
    )
    passes = bool(
        pass_folds >= 2
        and ksvd["paired_roc_auc"] >= raw["paired_roc_auc"] - 0.01
        and ksvd["paired_roc_auc"] > pca["paired_roc_auc"]
        and ksvd["paired_roc_auc"] > random["paired_roc_auc"]
    )
    payload = {
        "protocol_id": "molpcba-local-chemical-change-compression-v1",
        "scope": "OGB official training split only; three internal Murcko-scaffold folds; no official valid/test read",
        "purpose": "Compare equal-size compression of raw signed one-site changes.",
        "config": vars(args),
        "assay": assay,
        "counts": counts,
        "folds": results,
        "summary": summary,
        "decision": {
            "ksvd_beats_both_controls_in_folds": int(pass_folds),
            "passes": passes,
            "next_step": (
                "Attempt a frozen, single-molecule multi-task classification transfer."
                if passes
                else "Stop the K-SVD compression claim for this task; raw changes alone do not justify a dictionary method."
            ),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
