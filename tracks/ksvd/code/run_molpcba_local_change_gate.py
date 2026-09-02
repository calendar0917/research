"""Training-only gate for a local chemical-change dictionary.

This is intentionally *not* a MolPCBA leaderboard model.  Before asking K-SVD
to compress changes, we must establish that a raw, chemically anchored,
single-substituent replacement carries activity-direction signal beyond
chance on held-out Murcko cores.

The script reads only OGB's official training split.  It creates internal
scaffold folds, so molecules from one core family never occur on both sides
of a mechanism evaluation.  The official validation and test splits are not
opened or consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .local_chemical_change import (
    MoleculeBranches,
    Replacement,
    decompose_molecule,
    replacement_vector,
    replacements_for_labelled_group,
    scaffold_from_smiles,
)


def _fold_for_scaffold(scaffold: str, n_folds: int) -> int:
    digest = hashlib.blake2b(scaffold.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_folds


def _replace_for_task(
    table: pd.DataFrame,
    train_indices: np.ndarray,
    label_column: str,
    *,
    n_bits: int,
    radius: int,
    seed: int,
    retries: int,
    max_per_scaffold: int,
) -> tuple[list[Replacement], dict[str, int]]:
    """Make label-directed replacements without exposing any held-out data."""
    # Keep just SMILES and labels in the first pass.  Retaining every branch
    # fingerprint for a 350k-molecule dataset is unnecessary and can turn a
    # scientifically small audit into an avoidable memory failure.  Full
    # branch decompositions are made and released one mixed-core family at a
    # time below.
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    stats = defaultdict(int)
    labels = pd.to_numeric(table.loc[train_indices, label_column], errors="coerce").to_numpy(dtype=np.float64)
    smiles = table.loc[train_indices, "smiles"].fillna("").astype(str).to_numpy()
    for value, smile in zip(labels, smiles):
        if np.isnan(value):
            continue
        stats["known_labels"] += 1
        scaffold = scaffold_from_smiles(str(smile))
        if scaffold is None:
            stats["unusable_molecule"] += 1
            continue
        stats["scaffolded_molecule"] += 1
        grouped[scaffold].append((str(smile), int(value > 0.5)))

    all_replacements: list[Replacement] = []
    for scaffold, entries in grouped.items():
        labels_here = [label for _, label in entries]
        if not (0 in labels_here and 1 in labels_here):
            continue
        stats["mixed_scaffold"] += 1
        members: list[tuple[MoleculeBranches, int]] = []
        for smile, label in entries:
            decomposition = decompose_molecule(smile, n_bits=n_bits, radius=radius)
            if decomposition is None:
                stats["unusable_mixed_molecule"] += 1
                continue
            stats["decomposed_mixed_molecule"] += 1
            members.append((decomposition, label))
        unique: dict[tuple[str, str, str], Replacement] = {}
        # A single random maximal pairing can miss a clean replacement in a
        # series with several alternatives.  A fixed small number of retries
        # increases coverage without turning this into a hyperparameter scan.
        for attempt in range(retries):
            local_seed = seed + 7919 * attempt + int.from_bytes(
                hashlib.blake2b(scaffold.encode("utf-8"), digest_size=4).digest(), "little"
            )
            for replacement in replacements_for_labelled_group(members, seed=local_seed):
                key = (
                    replacement.scaffold,
                    replacement.positive.exact_key,
                    replacement.negative.exact_key,
                )
                unique[key] = replacement
        choices = list(unique.values())
        # Stable, label-free selection.  It prevents a few giant series from
        # becoming most of the samples and makes the audit reproducible.
        choices.sort(key=lambda item: (item.positive.exact_key, item.negative.exact_key))
        all_replacements.extend(choices[:max_per_scaffold])
        stats["eligible_scaffold"] += int(bool(choices))
    stats["replacements"] = len(all_replacements)
    return all_replacements, {str(key): int(value) for key, value in stats.items()}


def _cross_core_control(
    replacements: list[Replacement], folds: np.ndarray, *, seed: int) -> np.ndarray:
    """Break same-core binding while retaining exactly the branch marginals.

    Positive branches and negative branches are each real branches from the
    same fold.  Only their shared-core correspondence is replaced by a random
    cross-core pairing.  This makes a useful check against an apparent result
    driven solely by globally common active/inactive fragments.
    """
    output = np.empty((len(replacements), len(replacement_vector(replacements[0]))), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for fold in np.unique(folds):
        indices = np.flatnonzero(folds == fold)
        if len(indices) < 2:
            raise ValueError("each fold needs at least two replacements")
        permutation = rng.permutation(indices)
        # There are many cores in each fold.  Re-draw a bounded number of
        # times; the final deterministic rotation handles the improbable
        # collision left in a tiny fold.
        for _ in range(20):
            if all(replacements[int(a)].scaffold != replacements[int(b)].scaffold for a, b in zip(indices, permutation)):
                break
            permutation = rng.permutation(indices)
        for offset, (left, right) in enumerate(zip(indices, permutation)):
            if replacements[int(left)].scaffold == replacements[int(right)].scaffold:
                for candidate in np.roll(indices, offset + 1):
                    if replacements[int(left)].scaffold != replacements[int(candidate)].scaffold:
                        right = candidate
                        break
            output[int(left)] = (
                replacements[int(left)].positive.fingerprint.astype(np.float64)
                - replacements[int(right)].negative.fingerprint.astype(np.float64)
            )
    return output


def _direction_metrics(values: np.ndarray, folds: np.ndarray, n_folds: int) -> dict[str, object]:
    """Predict whether a signed change points from inactive to active."""
    result = []
    for fold in range(n_folds):
        train = folds != fold
        test = folds == fold
        if not np.any(train) or not np.any(test):
            raise ValueError("empty internal scaffold fold")
        # Sign augmentation creates the two possible directions of each
        # chemical replacement.  It is symmetric by construction and never
        # places a core family on both sides of a split.
        X_train = np.concatenate([values[train], -values[train]], axis=0)
        y_train = np.concatenate([np.ones(np.sum(train)), np.zeros(np.sum(train))])
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, random_state=0),
        )
        model.fit(X_train, y_train)
        probability = model.predict_proba(values[test])[:, 1]
        # The paired AUC includes the reverse direction and is exactly the
        # right quantity for this balanced directional decision.
        paired_y = np.concatenate([np.ones(np.sum(test)), np.zeros(np.sum(test))])
        paired_probability = np.concatenate([probability, 1.0 - probability])
        accuracy = float(np.mean(probability >= 0.5))
        result.append(
            {
                "fold": fold,
                "n_train_replacements": int(np.sum(train)),
                "n_test_replacements": int(np.sum(test)),
                "direction_accuracy": accuracy,
                "paired_roc_auc": float(roc_auc_score(paired_y, paired_probability)),
            }
        )
    return {
        "folds": result,
        "mean_direction_accuracy": float(np.mean([row["direction_accuracy"] for row in result])),
        "mean_paired_roc_auc": float(np.mean([row["paired_roc_auc"] for row in result])),
    }


def _task_result(
    table: pd.DataFrame,
    train_indices: np.ndarray,
    task: int,
    label_column: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    replacements, counts = _replace_for_task(
        table,
        train_indices,
        label_column,
        n_bits=args.n_bits,
        radius=args.radius,
        seed=args.seed + task,
        retries=args.pair_retries,
        max_per_scaffold=args.max_per_scaffold,
    )
    if len(replacements) < args.min_replacements:
        return {
            "task": task,
            "assay": label_column,
            "counts": counts,
            "status": "insufficient_clean_replacements",
            "required": args.min_replacements,
        }
    folds = np.asarray([_fold_for_scaffold(item.scaffold, args.n_folds) for item in replacements], dtype=np.int64)
    if len(np.unique(folds)) != args.n_folds:
        return {
            "task": task,
            "assay": label_column,
            "counts": counts,
            "status": "insufficient_fold_coverage",
        }
    local = np.stack([replacement_vector(item) for item in replacements], axis=0)
    cross = _cross_core_control(replacements, folds, seed=args.seed + 100003 + task)
    local_metric = _direction_metrics(local, folds, args.n_folds)
    cross_metric = _direction_metrics(cross, folds, args.n_folds)
    passed_folds = sum(
        float(row["direction_accuracy"]) > args.direction_threshold
        for row in local_metric["folds"]  # type: ignore[index]
    )
    binding_margins = [
        float(local_row["paired_roc_auc"] - control_row["paired_roc_auc"])
        for local_row, control_row in zip(local_metric["folds"], cross_metric["folds"])  # type: ignore[index]
    ]
    binding_passed = sum(value >= args.binding_auc_margin for value in binding_margins)
    return {
        "task": task,
        "assay": label_column,
        "counts": counts,
        "representation": {
            "type": "Morgan fingerprint of full shared core plus one attached branch",
            "n_bits": args.n_bits,
            "radius": args.radius,
            "signed_difference": "negative branch -> positive branch",
        },
        "same_core_local_change": local_metric,
        "cross_core_binding_control": cross_metric,
        "status": "completed",
        "raw_change_gate": {
            "threshold_per_fold": args.direction_threshold,
            "folds_above_threshold": int(passed_folds),
            "binding_auc_margin_per_fold": args.binding_auc_margin,
            "binding_auc_margins": binding_margins,
            "folds_above_binding_margin": int(binding_passed),
            "passes": bool(passed_folds >= 2 and binding_passed >= 2),
            "interpretation": "A pass requires both directional signal and a same-core advantage over cross-core binding shuffle. It is not yet single-molecule classification.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ogb/ogbg_molpcba")
    parser.add_argument("--tasks", default="93,94,47,60")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bits", type=int, default=512)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--pair-retries", type=int, default=4)
    parser.add_argument("--max-per-scaffold", type=int, default=8)
    parser.add_argument("--min-replacements", type=int, default=300)
    parser.add_argument("--direction-threshold", type=float, default=0.55)
    parser.add_argument("--binding-auc-margin", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.n_folds != 3:
        raise ValueError("this preregistered first audit uses exactly three scaffold folds")
    root = Path(args.root)
    # mol.csv has 128 label columns.  Loading all of them costs substantial
    # memory but this fixed-task gate needs only its requested assays and the
    # SMILES column.  Read the header first, then only those columns.
    header = pd.read_csv(root / "mapping/mol.csv.gz", nrows=0)
    label_columns = list(header.columns[:128])
    tasks = [int(value) for value in args.tasks.split(",") if value.strip()]
    if any(task < 0 or task >= len(label_columns) for task in tasks):
        raise ValueError("requested task outside MolPCBA's 128 labels")
    selected_columns = ["smiles"] + [label_columns[task] for task in tasks]
    table = pd.read_csv(root / "mapping/mol.csv.gz", usecols=selected_columns)
    train_indices = pd.read_csv(root / "split/scaffold/train.csv.gz", header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    results = [_task_result(table, train_indices, task, label_columns[task], args) for task in tasks]
    complete = [row for row in results if row["status"] == "completed"]
    passed = [row for row in complete if row["raw_change_gate"]["passes"]]  # type: ignore[index]
    payload = {
        "protocol_id": "molpcba-local-chemical-change-raw-gate-v1",
        "scope": "OGB official training split only; internal Murcko-scaffold folds; no official valid/test read",
        "purpose": "Test raw one-site chemical changes before any K-SVD or Transformer.",
        "config": vars(args),
        "tasks": results,
        "decision": {
            "n_completed": len(complete),
            "n_raw_change_passed": len(passed),
            "next_step": (
                "Fit K-SVD, PCA, and random-prototype compression only for passed tasks."
                if passed
                else "Stop the local-change K-SVD branch: raw changes did not clear the training-only gate."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
