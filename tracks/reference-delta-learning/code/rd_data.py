"""Frozen data view: official OGB splits + the reused SQ ECFP4 design matrix.

Everything here is read-only.  The two committed label-free caches of the SQ
round are verified by SHA-256 before use, the official split order is taken
verbatim from them, and no row is dropped.

Design matrix ``x(G)`` is exactly what ``sq_data.ecfp_design_matrix`` returns:

* 2048 unpacked Morgan radius-2 bits (dense float32, zero is a real value),
* 10 cached RDKit descriptors (missing values filled with the Train median of
  successfully parsed rows only),
* 1 RDKit parse-failure flag,

i.e. 2059 columns, no extra scaling, no extra filtering, no chirality recompute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

import rd_common as C
import sq_data as SQD


@dataclass
class TaskData:
    """One task in the frozen split order (labels are float64 raw units)."""

    dataset: str
    x: dict[str, np.ndarray]
    bits: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    row_id: dict[str, np.ndarray]
    smiles: dict[str, list[str]]
    scaler: dict[str, float]
    feature_names: list[str]
    design_stats: dict[str, Any]
    cache_sha256: str
    ecfp_sha256: str
    graph_cache: Mapping[str, Any]
    ecfp_cache: Mapping[str, Any]

    def n(self, split: str) -> int:
        return int(self.x[split].shape[0])

    def x_concat(self, splits: tuple[str, ...]) -> np.ndarray:
        return np.concatenate([self.x[s] for s in splits], axis=0)

    def y_concat(self, splits: tuple[str, ...]) -> np.ndarray:
        return np.concatenate([self.y[s] for s in splits], axis=0)

    def row_id_concat(self, splits: tuple[str, ...]) -> np.ndarray:
        return np.concatenate([self.row_id[s] for s in splits], axis=0)

    @property
    def n_features(self) -> int:
        return int(self.x["train"].shape[1])

    @property
    def train_y_std(self) -> np.ndarray:
        return (self.y["train"] - float(self.scaler["mean"])) / float(self.scaler["std"])

    def standardise(self, y_raw: np.ndarray) -> np.ndarray:
        return (np.asarray(y_raw, dtype=np.float64) - float(self.scaler["mean"])) / float(
            self.scaler["std"]
        )

    def raw_from_std(self, y_std: np.ndarray) -> np.ndarray:
        return np.asarray(y_std, dtype=np.float64) * float(self.scaler["std"]) + float(
            self.scaler["mean"]
        )


def verify_cache_hashes(dataset: str) -> dict[str, str]:
    """Refuse to run on a cache that is not the frozen one."""
    graph = SQD.cache_path(dataset, base=C.DATA_DIR)
    ecfp = SQD.ecfp_path(dataset, base=C.DATA_DIR)
    out: dict[str, str] = {}
    for path in (graph, ecfp):
        expected = C.EXPECTED_CACHE_SHA256.get(path.name)
        if expected is None:
            raise ValueError(f"no frozen hash registered for {path.name}")
        actual = C.sha256_file(path)
        if actual != expected:
            raise ValueError(f"cache sha mismatch for {path.name}: {actual} != {expected}")
        out[path.name] = actual
    return out


def load_task(dataset: str, *, verify: bool = True) -> TaskData:
    if dataset not in C.DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}")
    if verify:
        verify_cache_hashes(dataset)
    graph = SQD.load_cache(dataset, base=C.DATA_DIR)
    ecfp = SQD.load_ecfp(dataset, base=C.DATA_DIR)
    expected_ecfp = C.EXPECTED_CACHE_SHA256[f"{dataset.replace('-', '_')}_ecfp4_sq-pilot-cache-v1.npz"]
    if ecfp["cache_sha256"] != expected_ecfp:
        raise ValueError(f"{dataset}: ECFP cache sha mismatch")

    x: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    design_stats: dict[str, Any] = {}
    for split in C.SPLITS:
        matrix, stats = SQD.ecfp_design_matrix(ecfp, graph, split)
        if feature_names is None:
            feature_names = list(stats["names"])
        elif list(stats["names"]) != feature_names:
            raise ValueError(f"{dataset}: feature names differ between splits")
        design_stats[split] = stats
        x[split] = np.ascontiguousarray(matrix, dtype=np.float32)
    assert feature_names is not None
    if len(feature_names) != 2059:  # 2048 bits + 10 descriptors + 1 flag
        raise ValueError(f"{dataset}: design matrix has {len(feature_names)} columns, expected 2059")

    bits_raw = np.asarray(ecfp["ecfp_bits"], dtype=np.uint8)
    bits: dict[str, np.ndarray] = {}
    for split in C.SPLITS:
        idx = SQD.split_indices(graph, split)
        bits[split] = np.ascontiguousarray(
            np.unpackbits(bits_raw[idx], axis=1).astype(np.float32)
        )
    row_id = {split: C.row_id_array(graph, split).astype(np.int64) for split in C.SPLITS}
    smiles = {
        split: [str(graph["smiles"][int(i)]).strip() for i in SQD.split_indices(graph, split).tolist()]
        for split in C.SPLITS
    }
    if not np.array_equal(row_id["train"], SQD.split_indices(graph, "train")):
        raise ValueError(f"{dataset}: Train row_id order does not match the frozen split order")
    labels = {split: C.label_array(graph, split) for split in C.SPLITS}
    scaler = C.train_target_scaler(labels["train"])

    return TaskData(
        dataset=dataset,
        x=x,
        bits=bits,
        y=labels,
        row_id=row_id,
        smiles=smiles,
        scaler=scaler,
        feature_names=feature_names,
        design_stats=design_stats,
        cache_sha256=str(graph["cache_sha256"]),
        ecfp_sha256=str(ecfp["cache_sha256"]),
        graph_cache=graph,
        ecfp_cache=ecfp,
    )


def data_audit(task: TaskData) -> dict[str, Any]:
    """Identity block written before any training starts."""
    split_sizes = {split: task.n(split) for split in C.SPLITS}
    expected = C.EXPECTED_SPLIT_SIZES[task.dataset]
    if split_sizes != expected:
        raise ValueError(f"{task.dataset}: split sizes {split_sizes} != frozen {expected}")
    return {
        "dataset": task.dataset,
        "cache_sha256": task.cache_sha256,
        "ecfp_sha256": task.ecfp_sha256,
        "split_sizes": split_sizes,
        "split_hash": {split: C.sha256_array(task.row_id[split]) for split in C.SPLITS},
        "row_id_first_last": {
            split: [int(task.row_id[split][0]), int(task.row_id[split][-1])] for split in C.SPLITS
        },
        "row_id_matches_frozen_split": True,
        "label_range": {
            split: [float(task.y[split].min()), float(task.y[split].max())] for split in C.SPLITS
        },
        "train_target_scaler": dict(task.scaler),
        "n_features": task.n_features,
        "feature_names_hash": C.sha256_json(task.feature_names),
        # the full list is required deliverable evidence; it is only 2059 short names
        "feature_names": list(task.feature_names),
        "design_stats": task.design_stats,
        "n_parse_failed_total": int(np.asarray(task.ecfp_cache["parse_failed"]).sum()),
        "n_parse_failed_by_split": {
            split: int(
                np.asarray(task.ecfp_cache["parse_failed"])[
                    SQD.split_indices(task.graph_cache, split)
                ].sum()
            )
            for split in C.SPLITS
        },
        "smiles_duplicates_within_train": int(_duplicate_smiles(task.smiles["train"])),
    }


def _duplicate_smiles(smiles: list[str]) -> int:
    seen: dict[str, int] = {}
    duplicates = 0
    for text in smiles:
        if text in seen:
            duplicates += 1
        seen[text] = seen.get(text, 0) + 1
    return duplicates
