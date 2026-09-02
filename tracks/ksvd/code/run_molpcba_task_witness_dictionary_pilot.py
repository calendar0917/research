"""Quick pilot for a task-directed local-change dictionary.

This is deliberately smaller than a differentiable supervised K-SVD model.
For each internal scaffold fold, a logistic direction model is fit only on
the training signed changes.  Atoms are then selected from real training
changes with high task-direction score and diverse cosine geometry.  The
selected atoms are used as a frozen OMP dictionary and evaluated with the
same paired readout as the existing raw/PCA/K-SVD compression audit.

The pilot asks one narrow question: does task-directed *atom selection* beat
the reconstruction-oriented K-SVD dictionary without leaking held-out cores?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ksvd import _omp, ksvd
from .run_molpcba_local_change_compression import (
    _fit_score,
    _random_dictionary,
    _relative_reconstruction,
)
from .run_molpcba_local_change_gate import _fold_for_scaffold, _replace_for_task


def _scale_fit(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(values * values, axis=0) + 1e-8)


def _codes(dictionary: np.ndarray, values: np.ndarray, sparsity: int) -> np.ndarray:
    return np.stack([_omp(dictionary, row, sparsity) for row in values], axis=0)


def _task_witness_dictionary(
    values: np.ndarray,
    n_atoms: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select diverse real changes from the task-positive end of the margin."""
    X = np.concatenate([values, -values], axis=0)
    y = np.concatenate([np.ones(len(values)), np.zeros(len(values))])
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    )
    model.fit(X, y)
    scores = model.decision_function(values)
    # Candidate pool is label-directed, but the final selection is greedy
    # maximin to avoid selecting near-duplicate witness changes.
    order = np.argsort(-scores, kind="stable")
    pool_size = min(len(order), max(4 * n_atoms, n_atoms))
    candidates = values[order[:pool_size]]
    norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    normalized = candidates / np.maximum(norms, 1e-12)
    rng = np.random.default_rng(seed)
    selected: list[int] = [0]
    while len(selected) < min(n_atoms, len(candidates)):
        chosen = normalized[np.asarray(selected)]
        similarity = np.max(normalized @ chosen.T, axis=1)
        similarity[np.asarray(selected)] = np.inf
        # Tiny random tie break prevents dependence on platform sort details.
        similarity = similarity + rng.normal(0.0, 1e-9, size=len(similarity))
        selected.append(int(np.argmin(similarity)))
    dictionary = candidates[np.asarray(selected)].T.copy()
    dictionary /= np.maximum(np.linalg.norm(dictionary, axis=0, keepdims=True), 1e-12)
    return dictionary, {
        "candidate_pool": int(pool_size),
        "selected_atoms": int(dictionary.shape[1]),
        "selected_score_min": float(np.min(scores[order[np.asarray(selected)]])),
        "selected_score_max": float(np.max(scores[order[np.asarray(selected)]])),
    }


def _one_fold(values: np.ndarray, folds: np.ndarray, fold: int, args: argparse.Namespace) -> dict[str, object]:
    train, test = folds != fold, folds == fold
    raw_train, raw_test = values[train], values[test]
    scale = _scale_fit(raw_train)
    train_scaled, test_scaled = raw_train / scale, raw_test / scale
    pca = PCA(
        n_components=min(args.n_atoms, train_scaled.shape[0], train_scaled.shape[1]),
        random_state=args.seed,
    ).fit(train_scaled)
    random_dictionary = _random_dictionary(train_scaled, args.n_atoms, args.seed + fold)
    ksvd_dictionary, _, ksvd_info = ksvd(
        train_scaled.T,
        n_atoms=args.n_atoms,
        T=args.sparsity,
        n_iter=args.ksvd_iter,
        seed=args.seed + fold,
    )
    witness_dictionary, witness_info = _task_witness_dictionary(
        train_scaled, args.n_atoms, args.seed + 1009 * (fold + 1)
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
        "task_witness_dictionary": (
            _codes(witness_dictionary, train_scaled, args.sparsity),
            _codes(witness_dictionary, -train_scaled, args.sparsity),
            _codes(witness_dictionary, test_scaled, args.sparsity),
            _codes(witness_dictionary, -test_scaled, args.sparsity),
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
            "task_witness_dictionary": _relative_reconstruction(test_scaled, witness_dictionary, args.sparsity),
        },
        "ksvd_fit": ksvd_info,
        "task_witness_fit": witness_info,
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
    assay = label_columns[args.task]
    table = pd.read_csv(root / "mapping/mol.csv.gz", usecols=["smiles", assay])
    train_indices = pd.read_csv(root / "split/scaffold/train.csv.gz", header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    replacements, counts = _replace_for_task(
        table,
        train_indices,
        assay,
        n_bits=args.n_bits,
        radius=args.radius,
        seed=args.seed + args.task,
        retries=args.pair_retries,
        max_per_scaffold=args.max_per_scaffold,
    )
    if len(replacements) < 300:
        raise RuntimeError("too few clean replacements")
    values = np.stack([
        item.positive.fingerprint.astype(np.float64) - item.negative.fingerprint.astype(np.float64)
        for item in replacements
    ])
    folds = np.asarray([_fold_for_scaffold(item.scaffold, 3) for item in replacements], dtype=np.int64)
    results = [_one_fold(values, folds, fold, args) for fold in range(3)]
    summary = _summarize(results)
    witness = summary["task_witness_dictionary"]
    ksvd_summary = summary["ksvd_change_dictionary"]
    decision = {
        "task_witness_beats_ksvd_mean_auc": bool(witness["paired_roc_auc"] > ksvd_summary["paired_roc_auc"]),
        "task_witness_beats_ksvd_folds": int(sum(
            row["scores"]["task_witness_dictionary"]["paired_roc_auc"] > row["scores"]["ksvd_change_dictionary"]["paired_roc_auc"]  # type: ignore[index]
            for row in results
        )),
        "interpretation": "promising_task_directed_atom_selection" if witness["paired_roc_auc"] > ksvd_summary["paired_roc_auc"] else "no_pilot_evidence_over_reconstruction_ksvd",
    }
    payload = {
        "protocol_id": "molpcba-task-witness-dictionary-pilot-v1",
        "scope": "official training split only; three internal Murcko-scaffold folds",
        "purpose": "Quick test of task-directed real witness atom selection before differentiable dictionary learning.",
        "config": vars(args),
        "assay": assay,
        "counts": counts,
        "folds": results,
        "summary": summary,
        "decision": decision,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
