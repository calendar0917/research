"""Check whether MolPCBA contains enough within-scaffold label changes.

This is a data suitability audit for a later *local chemical change*
dictionary.  It never trains on labels: labels are counted only to determine
whether a task has enough same-core positive/negative pairs to make that
future supervised evaluation meaningful.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    # Acyclic molecules have an empty Murcko scaffold and do not provide a
    # stable shared core for the proposed edit representation.
    return value or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ogb/ogbg_molpcba")
    parser.add_argument("--tasks", default="93,94,47,60")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    table = pd.read_csv(root / "mapping/mol.csv.gz")
    train_indices = pd.read_csv(root / "split/scaffold/train.csv.gz", header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    tasks = [int(value) for value in args.tasks.split(",") if value.strip()]
    task_columns = list(table.columns[:128])
    if any(task < 0 or task >= len(task_columns) for task in tasks):
        raise ValueError("requested task outside MolPCBA's 128 labels")
    smiles = table.loc[train_indices, "smiles"].fillna("").astype(str).tolist()
    scaffold_rows: dict[str, list[int]] = defaultdict(list)
    invalid = 0
    for local_index, value in enumerate(smiles):
        core = _scaffold(value)
        if core is None:
            invalid += 1
            continue
        scaffold_rows[core].append(local_index)
    groups = list(scaffold_rows.values())
    group_sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
    task_result = []
    for task in tasks:
        values = pd.to_numeric(table.loc[train_indices, task_columns[task]], errors="coerce").to_numpy(dtype=np.float64)
        positive = negative = eligible_groups = mixed_groups = pair_capacity = 0
        mixed_molecules = 0
        for group in groups:
            known = values[np.asarray(group, dtype=np.int64)]
            known = known[~np.isnan(known)]
            if not len(known):
                continue
            pos, neg = int(np.sum(known > 0.5)), int(np.sum(known <= 0.5))
            positive += pos
            negative += neg
            if pos + neg >= 2:
                eligible_groups += 1
            if pos and neg:
                mixed_groups += 1
                pair_capacity += min(pos, neg)
                mixed_molecules += pos + neg
        task_result.append({
            "task": task, "assay": str(task_columns[task]),
            "positive": positive, "negative": negative,
            "positive_rate": float(positive / max(positive + negative, 1)),
            "scaffolds_with_two_known_labels": eligible_groups,
            "mixed_label_scaffolds": mixed_groups,
            "mixed_label_scaffold_fraction": float(mixed_groups / max(eligible_groups, 1)),
            "balanced_pair_capacity": pair_capacity,
            "molecules_in_mixed_scaffolds": mixed_molecules,
        })
    result = {
        "dataset": "ogbg-molpcba", "split": "official_train",
        "n_train_molecules": int(len(train_indices)),
        "n_nonempty_scaffold_molecules": int(len(train_indices) - invalid),
        "n_scaffolds": int(len(groups)),
        "scaffold_size": {
            "mean": float(group_sizes.mean()), "median": float(np.median(group_sizes)),
            "p95": float(np.quantile(group_sizes, 0.95)), "max": int(group_sizes.max()),
            "groups_size_at_least_2": int(np.sum(group_sizes >= 2)),
        },
        "tasks": task_result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
