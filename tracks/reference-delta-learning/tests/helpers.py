"""Small synthetic task fixtures (never touch the real caches)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import rd_data as D  # noqa: E402


def make_task(
    *,
    n_train: int = 40,
    n_valid: int = 8,
    n_test: int = 8,
    n_bits: int = 8,
    n_descriptors: int = 2,
    seed: int = 0,
    dataset: str = "synthetic",
    duplicate_smiles: bool = True,
) -> D.TaskData:
    rng = np.random.default_rng(seed)
    splits = ("train", "valid", "test")
    sizes = {"train": n_train, "valid": n_valid, "test": n_test}
    bits: dict[str, np.ndarray] = {}
    x: dict[str, np.ndarray] = {}
    y: dict[str, np.ndarray] = {}
    row_id: dict[str, np.ndarray] = {}
    smiles: dict[str, list[str]] = {}
    cursor = 0
    descriptors = rng.normal(size=(n_train + n_valid + n_test, n_descriptors)).astype(np.float32)
    flags = np.zeros((n_train + n_valid + n_test, 1), dtype=np.float32)
    all_bits = rng.integers(0, 2, size=(n_train + n_valid + n_test, n_bits)).astype(np.float32)
    all_y = rng.normal(size=n_train + n_valid + n_test) * 2.0 - 1.0
    all_smiles = [f"C{i}" for i in range(n_train + n_valid + n_test)]
    if duplicate_smiles and n_train > 2:
        all_smiles[n_train - 1] = all_smiles[0]
    for split in splits:
        n = sizes[split]
        idx = np.arange(cursor, cursor + n)
        cursor += n
        bits[split] = all_bits[idx]
        x[split] = np.concatenate([all_bits[idx], descriptors[idx], flags[idx]], axis=1)
        y[split] = all_y[idx]
        row_id[split] = idx.astype(np.int64)
        smiles[split] = [all_smiles[int(i)] for i in idx]
    scaler = {"mean": float(y["train"].mean()), "std": float(y["train"].std(ddof=0))}
    feature_names = (
        [f"morgan4_{i}" for i in range(n_bits)]
        + [f"desc_{i}" for i in range(n_descriptors)]
        + ["rdkit_parse_failed"]
    )
    return D.TaskData(
        dataset=dataset,
        x=x,
        bits=bits,
        y=y,
        row_id=row_id,
        smiles=smiles,
        scaler=scaler,
        feature_names=feature_names,
        design_stats={},
        cache_sha256="test",
        ecfp_sha256="test",
        graph_cache={},
        ecfp_cache={},
    )
