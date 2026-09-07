"""Dataset and split fingerprints for comparability guardrails.

A fingerprint is a content hash of everything that defines what data and
which split a run used.  Two runs are only directly comparable when both the
dataset fingerprint and the split fingerprint match.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .serialization import dumps_json, jsonable, sha256_file, sha256_text


def raw_dir_records(raw_dir: Path) -> list[dict[str, Any]]:
    """One record per file in a raw directory: name, bytes, sha256."""
    records: list[dict[str, Any]] = []
    if not Path(raw_dir).is_dir():
        return records
    for path in sorted(Path(raw_dir).iterdir()):
        if not path.is_file():
            continue
        records.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def raw_dir_fingerprint(raw_dir: Path) -> str:
    records = raw_dir_records(raw_dir)
    return sha256_text(dumps_json(jsonable(records)))


def zinc_fingerprints(data_root: Path, expected_sizes: dict[str, int] | None = None):
    """ZINC (PyG ``subset=True``) dataset + split fingerprints.

    ``split_fingerprint`` hashes the official split identities (train/val/test
    names with their index-file SHA-256) plus the split sizes; the dataset
    fingerprint additionally covers the molecule pickle files and metadata so
    a raw-file change is detected.
    """
    root = Path(data_root)
    splits = ("train", "val", "test")
    raw = raw_dir_records(root / "raw")
    split_sizes: dict[str, int | None] = {}
    for split in splits:
        index_path = root / "raw" / f"{split}.index"
        if index_path.is_file():
            split_sizes[split] = line_count(index_path)
        else:
            split_sizes[split] = None
    split_payload = {
        "policy": "official_pyg_zinc_subset_split",
        "splits": ["train", "val", "test"],
        "sizes": split_sizes,
        "expected_sizes": expected_sizes,
        "index_sha256": {
            record["name"]: record["sha256"]
            for record in raw
            if record["name"].endswith(".index")
        },
    }
    return {
        "policy": "official_pyg_zinc_data",
        "raw_files": raw,
        "split": split_payload,
        "dataset_fingerprint": sha256_text(dumps_json(jsonable({"raw": raw}))),
        "split_fingerprint": sha256_text(dumps_json(jsonable(split_payload))),
    }


def line_count(path: Path) -> int:
    """Count entries in a PyG-style index file (newline or comma separated)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return 0
    if not text:
        return 0
    return len([entry for entry in re.split(r"[\s,]+", text) if entry])


def split_sizes_from_dataset(dataset, splits: tuple[str, ...]) -> dict[str, int]:
    return {split: len(dataset) for split in splits}
