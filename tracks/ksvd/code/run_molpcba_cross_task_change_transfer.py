"""Can a local activity-change direction transfer across MolPCBA assays?

This is a gate for the only still-plausible *new* KSVD idea: a dictionary
shared across tasks but with task-specific effects.  It does not train such a
dictionary.  First it tests the necessary raw-data premise: a direction
learned from one assay's same-core changes should rank the direction of a
second assay on held-out cores.

If this premise fails, a task-shared dictionary would merely pool unrelated
supervision, so the branch stops before another K-SVD variant is introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .run_molpcba_local_change_gate import _fold_for_scaffold, _replace_for_task


def _values_and_folds(replacements) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack(
        [item.positive.fingerprint.astype(np.float64) - item.negative.fingerprint.astype(np.float64) for item in replacements],
        axis=0,
    )
    folds = np.asarray([_fold_for_scaffold(item.scaffold, 3) for item in replacements], dtype=np.int64)
    return values, folds


def _score(source_train: np.ndarray, target_test: np.ndarray) -> dict[str, float]:
    X_train = np.concatenate([source_train, -source_train], axis=0)
    y_train = np.concatenate([np.ones(len(source_train)), np.zeros(len(source_train))])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=0))
    model.fit(X_train, y_train)
    pos = model.predict_proba(target_test)[:, 1]
    neg = model.predict_proba(-target_test)[:, 1]
    y_test = np.concatenate([np.ones(len(target_test)), np.zeros(len(target_test))])
    return {
        "direction_accuracy": float(np.mean(pos > neg)),
        "paired_roc_auc": float(roc_auc_score(y_test, np.concatenate([pos, neg]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ogb/ogbg_molpcba")
    parser.add_argument("--source-task", type=int, default=93)
    parser.add_argument("--target-task", type=int, default=94)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bits", type=int, default=512)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--pair-retries", type=int, default=4)
    parser.add_argument("--max-per-scaffold", type=int, default=8)
    parser.add_argument("--max-label-agreement", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root)
    header = pd.read_csv(root / "mapping/mol.csv.gz", nrows=0)
    columns = list(header.columns[:128])
    if not (0 <= args.source_task < 128 and 0 <= args.target_task < 128):
        raise ValueError("task outside MolPCBA's 128 labels")
    source_assay, target_assay = columns[args.source_task], columns[args.target_task]
    table = pd.read_csv(root / "mapping/mol.csv.gz", usecols=["smiles", source_assay, target_assay])
    train_indices = pd.read_csv(root / "split/scaffold/train.csv.gz", header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    source_labels = pd.to_numeric(table.loc[train_indices, source_assay], errors="coerce").to_numpy(dtype=np.float64)
    target_labels = pd.to_numeric(table.loc[train_indices, target_assay], errors="coerce").to_numpy(dtype=np.float64)
    both_known = ~np.isnan(source_labels) & ~np.isnan(target_labels)
    label_agreement = float(np.mean(
        (source_labels[both_known] > 0.5) == (target_labels[both_known] > 0.5)
    ))
    common = dict(n_bits=args.n_bits, radius=args.radius, retries=args.pair_retries, max_per_scaffold=args.max_per_scaffold)
    source, source_counts = _replace_for_task(table, train_indices, source_assay, seed=args.seed + args.source_task, **common)
    target, target_counts = _replace_for_task(table, train_indices, target_assay, seed=args.seed + args.target_task, **common)
    source_values, source_folds = _values_and_folds(source)
    target_values, target_folds = _values_and_folds(target)
    folds = []
    for fold in range(3):
        forward = _score(source_values[source_folds != fold], target_values[target_folds == fold])
        reverse = _score(target_values[target_folds != fold], source_values[source_folds == fold])
        folds.append({"fold": fold, "source_to_target": forward, "target_to_source": reverse})
    forward_auc = [row["source_to_target"]["paired_roc_auc"] for row in folds]
    reverse_auc = [row["target_to_source"]["paired_roc_auc"] for row in folds]
    # A shared task dictionary is not worth trying unless transfer is clearly
    # above chance in both directions on most held-out scaffold folds.
    forward_pass = sum(value > 0.55 for value in forward_auc)
    reverse_pass = sum(value > 0.55 for value in reverse_auc)
    payload = {
        "protocol_id": "molpcba-cross-task-local-change-transfer-v1",
        "scope": "official train only; source and target use matched internal scaffold folds; no official valid/test read",
        "assays": {"source": {"task": args.source_task, "name": source_assay}, "target": {"task": args.target_task, "name": target_assay}},
        "counts": {"source": source_counts, "target": target_counts},
        "folds": folds,
        "label_overlap_audit": {
            "both_known_molecules": int(np.sum(both_known)),
            "binary_agreement": label_agreement,
            "maximum_allowed_for_independent_transfer_claim": args.max_label_agreement,
            "passes_independence_guard": bool(label_agreement <= args.max_label_agreement),
        },
        "summary": {
            "source_to_target_paired_roc_auc": float(np.mean(forward_auc)),
            "target_to_source_paired_roc_auc": float(np.mean(reverse_auc)),
            "source_to_target_folds_above_0_55": int(forward_pass),
            "target_to_source_folds_above_0_55": int(reverse_pass),
        },
        "decision": {
            "passes": bool(
                forward_pass >= 2
                and reverse_pass >= 2
                and label_agreement <= args.max_label_agreement
            ),
            "next_step": (
                "A task-shared, supervised change dictionary is worth a separately preregistered comparison."
                if forward_pass >= 2 and reverse_pass >= 2 and label_agreement <= args.max_label_agreement
                else "Stop the task-shared dictionary idea: transfer is absent or explained by near-duplicate task labels."
            ),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
