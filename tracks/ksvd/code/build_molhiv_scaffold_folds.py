"""Build official-train-only Bemis--Murcko development folds for MolHIV.

The output indices are aligned to an existing localized node-token cache.  No
OGB official-valid/test label is used to construct the folds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()


def _read_column(path: Path, column: str) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return [row[column] for row in csv.DictReader(f)]


def _read_scalar_rows(path: Path) -> list[str]:
    """Read OGB scalar CSV files, which do not necessarily have a header."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return [row[0] for row in csv.reader(f) if row]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--data-root", default="../../data/ogb")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")

    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise RuntimeError("RDKit is required to build scaffold folds") from exc

    RDLogger.DisableLog("rdApp.warning")
    cache_path = Path(args.token_cache)
    with np.load(cache_path, allow_pickle=False) as cache:
        original_indices = np.asarray(cache["original_indices"], dtype=np.int64)
        official_train = np.asarray(cache["train_indices"], dtype=np.int64)
        official_valid = np.asarray(cache["valid_indices"], dtype=np.int64)
        official_test = np.asarray(cache["test_indices"], dtype=np.int64)

    dataset_dir = Path(args.data_root) / "ogbg_molhiv"
    smiles = _read_column(dataset_dir / "mapping" / "mol.csv.gz", "smiles")
    labels = np.asarray(
        [float(x) for x in _read_scalar_rows(dataset_dir / "raw" / "graph-label.csv.gz")],
        dtype=np.float64,
    )
    if len(smiles) != len(labels):
        raise ValueError(f"SMILES/label length mismatch: {len(smiles)} vs {len(labels)}")

    train_original = original_indices[official_train]
    y = labels[train_original].astype(np.int64)
    groups: list[str] = []
    invalid: list[int] = []
    for local_i, original_i in zip(official_train.tolist(), train_original.tolist()):
        text = smiles[int(original_i)]
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            # Keep invalid records isolated rather than letting them share an
            # accidental empty scaffold group.
            invalid.append(int(local_i))
            groups.append(f"__invalid_original_{int(original_i)}")
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        groups.append(scaffold)

    groups_np = np.asarray(groups, dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    archive: dict[str, np.ndarray] = {
        "original_indices": original_indices,
        "official_train_indices": official_train,
        "official_valid_indices": official_valid,
        "official_test_indices": official_test,
        "train_scaffold_groups": groups_np.astype(str),
    }
    fold_meta: list[dict[str, object]] = []
    for fold, (train_pos, valid_pos) in enumerate(
        splitter.split(np.zeros(len(official_train)), y, groups_np)
    ):
        inner_train = official_train[np.asarray(train_pos, dtype=np.int64)]
        inner_valid = official_train[np.asarray(valid_pos, dtype=np.int64)]
        train_groups = set(groups_np[train_pos].tolist())
        valid_groups = set(groups_np[valid_pos].tolist())
        overlap = train_groups & valid_groups
        if overlap:
            raise AssertionError(f"fold {fold} scaffold leakage: {len(overlap)} groups")
        if set(inner_train.tolist()) & set(inner_valid.tolist()):
            raise AssertionError(f"fold {fold} index overlap")
        if set(np.concatenate([inner_train, inner_valid]).tolist()) != set(official_train.tolist()):
            raise AssertionError(f"fold {fold} does not partition official train")
        archive[f"fold_{fold}_train_indices"] = inner_train
        archive[f"fold_{fold}_valid_indices"] = inner_valid
        fold_meta.append(
            {
                "fold": fold,
                "n_train": int(len(inner_train)),
                "n_valid": int(len(inner_valid)),
                "n_train_pos": int(y[train_pos].sum()),
                "n_valid_pos": int(y[valid_pos].sum()),
                "n_train_scaffolds": int(len(train_groups)),
                "n_valid_scaffolds": int(len(valid_groups)),
                "largest_valid_scaffold": int(
                    max(Counter(groups_np[valid_pos].tolist()).values())
                ),
                "train_sha256": _sha256(inner_train),
                "valid_sha256": _sha256(inner_valid),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **archive)
    counts = Counter(groups_np.tolist())
    metadata = {
        "protocol_id": "molhiv-official-train-bemis-murcko-development-folds-v1",
        "token_cache": str(cache_path),
        "data_root": str(Path(args.data_root)),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "official_valid_labels_used": False,
        "official_test_labels_used": False,
        "n_official_train": int(len(official_train)),
        "n_train_pos": int(y.sum()),
        "n_unique_scaffolds": int(len(counts)),
        "n_empty_scaffold": int(counts.get("", 0)),
        "n_invalid_smiles": int(len(invalid)),
        "invalid_local_indices": invalid,
        "largest_scaffolds": counts.most_common(20),
        "folds": fold_meta,
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
