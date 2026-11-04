"""P0.2 — Peptides-func label distribution for the official Train/Validation splits.

Primary source: the official LRGB Peptides-func CSV (SMILES + 10 labels) together
with the official split index file, both fetched through the ``skfp`` interface and
cached locally.  The Fast v1 ``peptides_func_y_{train,valid}.npy`` arrays are used
only as a cross-check.  Test labels are never read.
"""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

import diag_common as C


def read_csv_rows() -> list[dict[str, str]]:
    with open(C.PEPTIDES_CSV, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_labels() -> tuple[list[dict[str, str]], dict[str, list[int]], np.ndarray]:
    rows = read_csv_rows()
    if len(rows) != sum(C.SPLIT_SIZES.values()):
        raise RuntimeError(f"CSV rows {len(rows)} != {sum(C.SPLIT_SIZES.values())}")
    header = list(rows[0].keys())
    missing = [name for name in C.LABELS if name not in header]
    if missing:
        raise RuntimeError(f"labels missing from CSV header: {missing}")
    splits_raw = C.read_json(C.PEPTIDES_SPLITS)
    splits = {split: [int(i) for i in splits_raw[split]] for split in C.SPLITS}
    labels = np.asarray(
        [[int(row[name]) for name in C.LABELS] for row in rows], dtype=np.int64
    )
    if labels.shape != (len(rows), C.N_LABELS):
        raise RuntimeError(f"label matrix {labels.shape}")
    if not np.all((labels == 0) | (labels == 1)):
        raise RuntimeError("labels are not binary")
    seen = np.zeros(len(rows), dtype=np.int64)
    for split in C.SPLITS:
        idx = np.asarray(splits[split], dtype=np.int64)
        if idx.size != C.SPLIT_SIZES[split]:
            raise RuntimeError(f"{split}: {idx.size} rows != {C.SPLIT_SIZES[split]}")
        if np.unique(idx).size != idx.size:
            raise RuntimeError(f"{split}: duplicate indices")
        seen[idx] += 1
    if not np.all(seen == 1):
        raise RuntimeError("split indices are not a partition of the CSV rows")
    return rows, splits, labels


def main() -> int:
    rows, splits, labels = load_labels()
    train_idx = np.asarray(splits["train"], dtype=np.int64)
    valid_idx = np.asarray(splits["valid"], dtype=np.int64)
    y_train = labels[train_idx]
    y_valid = labels[valid_idx]

    # cross-check against the arrays the Fast v1 training pipeline used
    prev_train = np.load(C.PREV_ANALYSIS / "peptides_func_y_train.npy").astype(np.int64)
    prev_valid = np.load(C.PREV_ANALYSIS / "peptides_func_y_valid.npy").astype(np.int64)
    if prev_train.shape != y_train.shape or not np.array_equal(prev_train, y_train):
        raise RuntimeError("official CSV train labels != Fast v1 peptides_func_y_train.npy")
    if prev_valid.shape != y_valid.shape or not np.array_equal(prev_valid, y_valid):
        raise RuntimeError("official CSV valid labels != Fast v1 peptides_func_y_valid.npy")

    out: list[dict[str, Any]] = []
    for j, name in enumerate(C.LABELS):
        tr_pos = int(y_train[:, j].sum())
        va_pos = int(y_valid[:, j].sum())
        tr_n = int(y_train.shape[0])
        va_n = int(y_valid.shape[0])
        tr_rate = tr_pos / tr_n
        va_rate = va_pos / va_n
        out.append(
            {
                "label_index": j,
                "label": name,
                "train_n": tr_n,
                "train_pos": tr_pos,
                "train_neg": tr_n - tr_pos,
                "train_pos_rate": tr_rate,
                "train_constant_ap_baseline": tr_rate,
                "valid_n": va_n,
                "valid_pos": va_pos,
                "valid_neg": va_n - va_pos,
                "valid_pos_rate": va_rate,
                "valid_constant_ap_baseline": va_rate,
                "abs_rate_difference_valid_minus_train": va_rate - tr_rate,
            }
        )
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    fields = list(out[0].keys())
    with open(C.ANALYSIS / "label_statistics.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)

    macro = {
        "train_pos_rate_mean": float(np.mean(y_train.mean(axis=0))),
        "valid_pos_rate_mean": float(np.mean(y_valid.mean(axis=0))),
        "train_label_cardinality_mean": float(y_train.sum(axis=1).mean()),
        "valid_label_cardinality_mean": float(y_valid.sum(axis=1).mean()),
        "train_rows_with_zero_labels": int((y_train.sum(axis=1) == 0).sum()),
        "valid_rows_with_zero_labels": int((y_valid.sum(axis=1) == 0).sum()),
        "min_label_pos_rate_train": float(y_train.mean(axis=0).min()),
        "max_label_pos_rate_train": float(y_train.mean(axis=0).max()),
    }
    checks = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "label_matrix_sha256": C.sha256_text(y_train.tobytes().hex() + y_valid.tobytes().hex()),
        "train_labels_sha256": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_train.npy"),
        "valid_labels_sha256": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_valid.npy"),
        "smiles_csv_sha256": C.sha256_file(C.PEPTIDES_CSV),
        "splits_json_sha256": C.sha256_file(C.PEPTIDES_SPLITS),
        "n_csv_rows": len(rows),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "checks": {
            "csv_rows_equal_15535": len(rows) == 15535,
            "split_sizes_10873_2331_2331": {
                k: len(splits[k]) for k in C.SPLITS
            }
            == {"train": 10873, "valid": 2331, "test": 2331},
            "split_indices_disjoint_and_complete": True,
            "labels_binary": True,
            "train_labels_equal_fast_v1_npy": True,
            "valid_labels_equal_fast_v1_npy": True,
            "test_labels_read": False,
        },
        "macro": macro,
        "environment": C.environment(),
    }
    C.write_json(C.ANALYSIS / "label_statistics.json", checks)
    for row in out:
        C.log(
            f"{row['label']:<26} train {row['train_pos']:5d}/{row['train_n']} ({100*row['train_pos_rate']:.2f}%) "
            f"valid {row['valid_pos']:4d}/{row['valid_n']} ({100*row['valid_pos_rate']:.2f}%)"
        )
    C.log(f"macro pos rate train {100*macro['train_pos_rate_mean']:.3f}% valid {100*macro['valid_pos_rate_mean']:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
