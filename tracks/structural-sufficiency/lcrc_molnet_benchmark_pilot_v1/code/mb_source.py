"""Vendored public MoleculeNet data + published split indices (ChemBench).

Source
------
``https://github.com/shenwanxiang/ChemBench`` at the commit recorded in
``analysis/source/chembench_source_manifest.json``.  Each dataset directory contains the
published CSV and three published split-index files
(``*.random122/123/124.ind.pkl`` or ``*.scaffold122/123/124.ind.pkl``).

Why the files are vendored
--------------------------
The public repository has no tagged release and no checksums.  The raw bytes of
the four CSV files and all twelve split-index files are therefore copied into
this track at a fixed commit and hash-pinned, so that

* a fresh clone can rebuild the exact benchmark without network access;
* the remote (air-gapped) cluster executes the same bytes as the local audit;
* the choice of split repeat 0 is auditable against the other two repeats.

Discipline
----------
* No split is generated, reshuffled, filtered or repaired here.
* Row ids are the 0-based positional index of the source CSV.
* No label-derived quantity is computed at load time beyond the raw target
  column itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import mb_common as C

#: Upstream commit.  All blobs below were read at exactly this revision.
CHEMBENCH_COMMIT = "0b10d4c1ac72b6929503694a5f60446824736391"
CHEMBENCH_COMMIT_DATE = "2022-03-29T14:17:38+08:00"
CHEMBENCH_URL = "https://github.com/shenwanxiang/ChemBench"
CHEMBENCH_RAW_BASE = (
    "https://raw.githubusercontent.com/shenwanxiang/ChemBench/" + CHEMBENCH_COMMIT
)

#: git blob ids of the source tree at ``CHEMBENCH_COMMIT`` (independently
#: recomputable with ``git ls-tree -r <commit>``).
SOURCE_GIT_BLOBS = {
    "ESOL/ESOL.csv.gz": "399643ddd1f4a1a98890923a735317a5960019c9",
    "ESOL/ESOL.random122.ind.pkl": "bcf75e7b633fce906eb9715e3d903bc2bde72390",
    "ESOL/ESOL.random123.ind.pkl": "939160aebe3942ad322cbd2cfda5ebf0261bb301",
    "ESOL/ESOL.random124.ind.pkl": "db72657665fdd02b190bd6da9192c75385810aca",
    "Lipop/Lipop.csv.gz": "e1a548e7d6e2530fe015b58dadb9131164283c81",
    "Lipop/Lipop.random122.ind.pkl": "dbc3fc7ed78e78d5d2f4109e4d9c52025f1a9ba2",
    "Lipop/Lipop.random123.ind.pkl": "6535f8d5db1bb6cfacb06841339f2fa23d28ef8c",
    "Lipop/Lipop.random124.ind.pkl": "1e4acb991559a5e31ff0acfbae021240d743ba78",
    "FreeSolv/FreeSolv.csv.gz": "b72c7f96fc6027d0f46a2fd06751d6a5300b93a5",
    "FreeSolv/FreeSolv.random122.ind.pkl": "8029f538bb8da1229d7072b9c5440a2345fbc023",
    "FreeSolv/FreeSolv.random123.ind.pkl": "0376bb56a68ee20ca4b36357d02b40bd20d3f9f0",
    "FreeSolv/FreeSolv.random124.ind.pkl": "d418f6ea796e24e7673f9480efae6252b50a7369",
    "BACE/BACE.csv.gz": "223d916618053a747618603b42ee4f47fd75ac8f",
    "BACE/BACE.scaffold122.ind.pkl": "db67f92c5af6816e627717929e63597156f91871",
    "BACE/BACE.scaffold123.ind.pkl": "db67f92c5af6816e627717929e63597156f91871",
    "BACE/BACE.scaffold124.ind.pkl": "db67f92c5af6816e627717929e63597156f91871",
}

#: Split file stems per dataset, in filename order.
SPLIT_STEMS = {
    "ESOL": "ESOL.random",
    "Lipophilicity": "Lipop.random",
    "FreeSolv": "FreeSolv.random",
    "BACE": "BACE.scaffold",
}


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def build_source_manifest() -> dict[str, Any]:
    """Recompute the vendored-source manifest from the local bytes."""
    files: dict[str, Any] = {}
    for rel, blob in sorted(SOURCE_GIT_BLOBS.items()):
        path = C.CHEMBENCH_DIR / rel
        if not path.exists():
            raise FileNotFoundError(path)
        files[rel] = {
            "path": C.track_rel(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": C.sha256_file(path),
            "git_blob_sha1": blob,
            "url": f"{CHEMBENCH_RAW_BASE}/src/chembench/data_and_index/{rel}",
        }
    per_dataset: dict[str, dict[str, Any]] = {}
    for key, meta in C.DATASETS.items():
        task = meta["chembench_task"]
        stem = SPLIT_STEMS[key]
        repeats = []
        for suffix in (122, 123, 124):
            rel = f"{task}/{stem}{suffix}.ind.pkl"
            repeats.append(
                {
                    "repeat_file_index": suffix,
                    "file": rel,
                    "sha256": files[rel]["sha256"],
                }
            )
        per_dataset[key] = {
            "chembench_task": task,
            "csv": files[f"{task}/{task}.csv.gz"],
            "split_repeats": repeats,
            "selected_repeat_index": C.SPLIT_REPEAT_INDEX,
            "selected_repeat_file_index": C.SPLIT_REPEAT_FILE_INDEX,
            "selection_rule": (
                "repeat 0 is defined as the lexicographically first published split file "
                f"(suffix {C.SPLIT_REPEAT_FILE_INDEX}); ChemBench load_data() globs *.pkl with no "
                "guaranteed order, so the file name -- not glob order -- is the frozen identifier"
            ),
        }
    return {
        "schema": "lcrc-molnet-chembench-source-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "upstream": {
            "repository": CHEMBENCH_URL,
            "commit": CHEMBENCH_COMMIT,
            "commit_date": CHEMBENCH_COMMIT_DATE,
            "raw_base": CHEMBENCH_RAW_BASE,
            "license_note": "see upstream LICENSE; data redistributed for reproducible benchmarking",
        },
        "load_data_reference": (
            "from chembench import load_data; df, split_repeats = load_data(<task>); "
            "train_idx, valid_idx, test_idx = split_repeats[0]"
        ),
        "files": files,
        "datasets": per_dataset,
        "vendored_copy": {
            "root": C.track_rel(C.CHEMBENCH_DIR),
            "reason": "no tagged release / no upstream checksums; fixed commit + byte hashes required",
        },
    }


def load_source_manifest(*, verify: bool = True) -> dict[str, Any]:
    if not C.SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            f"{C.SOURCE_MANIFEST} is missing; run code/mb_source_manifest.py to create it"
        )
    manifest = C.read_json(C.SOURCE_MANIFEST)
    if manifest.get("upstream", {}).get("commit") != CHEMBENCH_COMMIT:
        raise ValueError("vendored source manifest commit != mb_source.CHEMBENCH_COMMIT")
    if verify:
        for rel, info in manifest["files"].items():
            actual = C.sha256_file(C.CHEMBENCH_DIR / rel)
            if actual != info["sha256"]:
                raise ValueError(f"sha256 mismatch for {rel}: {actual} != {info['sha256']}")
    return manifest


# ---------------------------------------------------------------------------
# dataset loading
# ---------------------------------------------------------------------------


def _read_split_file(path: Path) -> list[list[int]]:
    series = pd.read_pickle(path)
    parts = [list(part) for part in series.to_list()]
    if len(parts) != 3:
        raise ValueError(f"{path.name}: expected 3 split parts, got {len(parts)}")
    return [[int(x) for x in part] for part in parts]


def load_dataset(key: str, *, verify: bool = True, labels: str = "train_valid") -> dict[str, Any]:
    """Load one dataset: SMILES, published split indices and (optionally) targets.

    ``labels``
        ``"none"``          the CSV is opened with ``usecols=["smiles"]``; no
                            label byte is ever decoded (used by the P0
                            structure/alignment audit).
        ``"train_valid"``   train and validation targets are decoded; the test
                            target values are dropped from the returned object
                            (used by P0 target audit, training and model
                            selection).
        ``"all"``          every target is decoded.  Only the P2 one-shot
                            evaluation may request this.
    """
    if labels not in ("none", "train_valid", "all"):
        raise ValueError(f"labels must be none/train_valid/all, got {labels!r}")
    if key not in C.DATASETS:
        raise KeyError(f"unknown dataset {key!r}")
    meta = C.DATASETS[key]
    manifest = load_source_manifest(verify=verify)
    ds_manifest = manifest["datasets"][key]
    task = meta["chembench_task"]

    csv_path = C.CHEMBENCH_DIR / f"{task}/{task}.csv.gz"
    if labels == "none":
        if meta["label_column"] in ("smiles",):
            raise ValueError("label column collides with the smiles column")
        df = pd.read_csv(csv_path, usecols=["smiles"])
    else:
        df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError(f"{key}: no 'smiles' column")
    if len(df) != int(meta["expected_rows"]):
        raise ValueError(f"{key}: {len(df)} rows != expected {meta['expected_rows']}")

    stem = SPLIT_STEMS[key]
    split_paths = sorted((C.CHEMBENCH_DIR / task).glob(f"{stem}*.ind.pkl"))
    if len(split_paths) != 3:
        raise ValueError(f"{key}: expected 3 split repeat files, found {[p.name for p in split_paths]}")
    if f"{C.SPLIT_REPEAT_FILE_INDEX}" not in split_paths[0].name:
        raise ValueError(
            f"{key}: lexicographically first split file {split_paths[0].name} is not the frozen "
            f"suffix {C.SPLIT_REPEAT_FILE_INDEX}"
        )
    repeats = []
    for path in split_paths:
        parts = _read_split_file(path)
        repeats.append(
            {
                "file": path.name,
                "sha256": C.sha256_file(path),
                "sizes": [len(p) for p in parts],
                "rows_sha256": [C.row_sha256(p) for p in parts],
            }
        )
    chosen = repeats[C.SPLIT_REPEAT_INDEX]
    train, valid, test = _read_split_file(split_paths[C.SPLIT_REPEAT_INDEX])

    n = len(df)
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        if len(set(part)) != len(part):
            raise ValueError(f"{key}: duplicate row ids inside {name}")
        if min(part) < 0 or max(part) >= n:
            raise ValueError(f"{key}: {name} row id outside [0, {n})")
    union = set(train) | set(valid) | set(test)
    if len(union) != n:
        raise ValueError(f"{key}: split union {len(union)} != dataset rows {n}")
    if set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test):
        raise ValueError(f"{key}: split partitions overlap")
    if [len(train), len(valid), len(test)] != list(meta["expected_split_sizes"]):
        raise ValueError(
            f"{key}: split sizes {[len(train), len(valid), len(test)]} != {meta['expected_split_sizes']}"
        )

    smiles = [str(s) for s in df["smiles"].tolist()]
    y: dict[str, np.ndarray] | None
    if labels == "none":
        y = None
    else:
        if meta["label_column"] not in df.columns:
            raise ValueError(f"{key}: label column {meta['label_column']!r} not in {list(df.columns)}")
        raw = np.asarray(df[meta["label_column"]].to_numpy(), dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError(f"{key}: non-finite target values")
        y = {
            "train": raw[np.asarray(train, dtype=np.int64)],
            "valid": raw[np.asarray(valid, dtype=np.int64)],
        }
        if labels == "all":
            y["test"] = raw[np.asarray(test, dtype=np.int64)]
    return {
        "key": key,
        "meta": meta,
        "dataframe": df,
        "smiles": smiles,
        "y": y,
        "rows": {"train": train, "valid": valid, "test": test},
        "repeats": repeats,
        "selected_repeat": chosen,
        "csv_path": csv_path,
        "csv_sha256": C.sha256_file(csv_path),
        "source_commit": manifest["upstream"]["commit"],
        "dataset_manifest": ds_manifest,
        "labels_scope": labels,
    }


__all__ = [
    "CHEMBENCH_COMMIT",
    "CHEMBENCH_COMMIT_DATE",
    "CHEMBENCH_URL",
    "CHEMBENCH_RAW_BASE",
    "SOURCE_GIT_BLOBS",
    "SPLIT_STEMS",
    "build_source_manifest",
    "load_source_manifest",
    "load_dataset",
]
