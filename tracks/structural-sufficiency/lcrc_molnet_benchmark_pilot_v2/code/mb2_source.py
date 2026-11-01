"""Frozen data sources for the seven-task benchmark (v2).

Two independent upstreams, both hash-pinned and *not* re-split here:

1. **ChemBench** (``shenwanxiang/ChemBench``, commit
   ``0b10d4c1ac72b6929503694a5f60446824736391``) supplies six of the seven
   datasets with the *published* split index files.  The four Sprint-01
   datasets are read from the vendored bytes of
   ``lcrc_molnet_benchmark_pilot_v1`` (read-only); BBBP and Tox21 are vendored
   byte-for-byte into this round.  Nothing is regenerated, filtered or repaired.

2. **OGB** supplies ``ogbg-molhiv`` with the *official* scaffold split.  The
   SMILES come from the official molecular mapping file
   (``mapping/mol.csv.gz``), the labels from ``raw/graph-label.csv.gz`` and the
   split row ids from ``split/scaffold/{train,valid,test}.csv.gz``.  OGB's
   pre-computed node/edge features are **not** used as model input: the LCRC
   input graph is rebuilt from SMILES with the frozen ``lipo_v1`` adapter.  That
   deliberate difference is recorded in ``ogb_input_semantics()``.

Discipline
----------
* Row ids are 0-based positional indices into the frozen source table.
* Split row ids are kept in the exact order published (no sorting).
* ``labels="none"`` never decodes a label byte: the CSV is opened with
  ``usecols=["smiles"]`` (ChemBench) or only the SMILES column is read (OGB).
* Tox21 missing labels stay ``NaN``.  They are never filled with 0.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import mb2_common as C

# ---------------------------------------------------------------------------
# ChemBench
# ---------------------------------------------------------------------------

CHEMBENCH_COMMIT = "0b10d4c1ac72b6929503694a5f60446824736391"
CHEMBENCH_COMMIT_DATE = "2022-03-29T14:17:38+08:00"
CHEMBENCH_URL = "https://github.com/shenwanxiang/ChemBench"
CHEMBENCH_RAW_BASE = "https://raw.githubusercontent.com/shenwanxiang/ChemBench/" + CHEMBENCH_COMMIT

#: git blob ids at ``CHEMBENCH_COMMIT`` (recomputable with ``git ls-tree -r``).
#: The v1 rows are re-verified, not trusted: ``build_source_manifest`` hashes them.
SOURCE_GIT_BLOBS: dict[str, tuple[str, str]] = {
    # rel path -> (git blob sha1, "v1" | "v2" vendor)
    "ESOL/ESOL.csv.gz": ("399643ddd1f4a1a98890923a735317a5960019c9", "v1"),
    "ESOL/ESOL.random122.ind.pkl": ("bcf75e7b633fce906eb9715e3d903bc2bde72390", "v1"),
    "ESOL/ESOL.random123.ind.pkl": ("939160aebe3942ad322cbd2cfda5ebf0261bb301", "v1"),
    "ESOL/ESOL.random124.ind.pkl": ("db72657665fdd02b190bd6da9192c75385810aca", "v1"),
    "Lipop/Lipop.csv.gz": ("e1a548e7d6e2530fe015b58dadb9131164283c81", "v1"),
    "Lipop/Lipop.random122.ind.pkl": ("dbc3fc7ed78e78d5d2f4109e4d9c52025f1a9ba2", "v1"),
    "Lipop/Lipop.random123.ind.pkl": ("6535f8d5db1bb6cfacb06841339f2fa23d28ef8c", "v1"),
    "Lipop/Lipop.random124.ind.pkl": ("1e4acb991559a5e31ff0acfbae021240d743ba78", "v1"),
    "FreeSolv/FreeSolv.csv.gz": ("b72c7f96fc6027d0f46a2fd06751d6a5300b93a5", "v1"),
    "FreeSolv/FreeSolv.random122.ind.pkl": ("8029f538bb8da1229d7072b9c5440a2345fbc023", "v1"),
    "FreeSolv/FreeSolv.random123.ind.pkl": ("0376bb56a68ee20ca4b36357d02b40bd20d3f9f0", "v1"),
    "FreeSolv/FreeSolv.random124.ind.pkl": ("d418f6ea796e24e7673f9480efae6252b50a7369", "v1"),
    "BACE/BACE.csv.gz": ("223d916618053a747618603b42ee4f47fd75ac8f", "v1"),
    "BACE/BACE.scaffold122.ind.pkl": ("db67f92c5af6816e627717929e63597156f91871", "v1"),
    "BACE/BACE.scaffold123.ind.pkl": ("db67f92c5af6816e627717929e63597156f91871", "v1"),
    "BACE/BACE.scaffold124.ind.pkl": ("db67f92c5af6816e627717929e63597156f91871", "v1"),
    "BBBP/BBBP.csv.gz": ("7348cd24b8b4298c984aed31111c3ffa41976c1c", "v2"),
    "BBBP/BBBP.scaffold122.ind.pkl": ("14f2f6e5b58bf65a07882ae4c06c8d3fa599001d", "v2"),
    "BBBP/BBBP.scaffold123.ind.pkl": ("14f2f6e5b58bf65a07882ae4c06c8d3fa599001d", "v2"),
    "BBBP/BBBP.scaffold124.ind.pkl": ("14f2f6e5b58bf65a07882ae4c06c8d3fa599001d", "v2"),
    "Tox21/Tox21.csv.gz": ("1ac7ca7937abc240295291af6d47affce812fa44", "v2"),
    "Tox21/Tox21.random122.ind.pkl": ("52186544a768ff043ad6ac8d7996c862db8334d5", "v2"),
    "Tox21/Tox21.random123.ind.pkl": ("e16c6273edbcb11a8ec2aa0690c0488b95a8bbd5", "v2"),
    "Tox21/Tox21.random124.ind.pkl": ("8e19260220000f28e8462ac418e8e9f480686f8a", "v2"),
}


def chembench_path(rel: str) -> Path:
    """Resolve a vendored ChemBench file to whichever round's copy holds it."""
    if rel not in SOURCE_GIT_BLOBS:
        raise KeyError(f"unexpected ChemBench path {rel!r}")
    vendor = SOURCE_GIT_BLOBS[rel][1]
    root = C.CHEMBENCH_V1 if vendor == "v1" else C.CHEMBENCH_V2
    return root / rel


def _read_split_file(path: Path) -> list[list[int]]:
    series = pd.read_pickle(path)
    parts = [list(part) for part in series.to_list()]
    if len(parts) != 3:
        raise ValueError(f"{path.name}: expected 3 split parts, got {len(parts)}")
    return [[int(x) for x in part] for part in parts]


def _target_matrix(df: pd.DataFrame, columns: Sequence[str], task: str) -> np.ndarray:
    """Target matrix [N, n_tasks]; NaN kept as missing (Tox21), else finite-checked."""
    if task == "multitask":
        raw = df[list(columns)].to_numpy(dtype=np.float64)
        bad = np.isfinite(raw) & ~np.isin(raw, (0.0, 1.0))
        if bad.any():
            raise ValueError(f"non-binary observed label in a multitask target: {raw[bad][:5]}")
        return raw  # NaN preserved: never coerced to 0
    col = columns[0]
    raw = np.asarray(df[col].to_numpy(), dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(raw).all():
        raise ValueError(f"non-finite target values in {col!r}")
    return raw


def load_chembench(key: str, *, verify: bool = True, labels: str = "train_valid") -> dict[str, Any]:
    """Load one ChemBench dataset: SMILES, published split indices, targets."""
    if labels not in ("none", "train_valid", "all"):
        raise ValueError(f"labels must be none/train_valid/all, got {labels!r}")
    meta = C.dataset_entry(key)
    if meta["source"] != "chembench":
        raise ValueError(f"{key} is not a ChemBench dataset")
    manifest = load_source_manifest(verify=verify) if verify else C.read_json(C.SOURCE_MANIFEST)
    task = meta["chembench_task"]
    csv_path = chembench_path(f"{task}/{task}.csv.gz")
    if labels == "none":
        df = pd.read_csv(csv_path, usecols=["smiles"])
    else:
        df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError(f"{key}: no 'smiles' column")
    if len(df) != int(meta["expected_rows"]):
        raise ValueError(f"{key}: {len(df)} rows != expected {meta['expected_rows']}")

    stem = meta["split_stem"]
    split_paths = sorted(chembench_path(f"{task}/{stem}{s}.ind.pkl") for s in (122, 123, 124))
    for p in split_paths:
        if not p.exists():
            raise FileNotFoundError(p)
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
    sizes = (len(train), len(valid), len(test))
    if tuple(sizes) != tuple(meta["expected_split_sizes"]):
        raise ValueError(f"{key}: split sizes {sizes} != frozen {meta['expected_split_sizes']}")

    smiles = [str(s) for s in df["smiles"].tolist()]
    y: dict[str, np.ndarray] | None = None
    if labels != "none":
        missing = [c for c in meta["label_columns"] if c not in df.columns]
        if missing:
            raise ValueError(f"{key}: label column(s) {missing} not in {list(df.columns)}")
        raw = _target_matrix(df, meta["label_columns"], meta["task"])
        y = {"train": raw[np.asarray(train, dtype=np.int64)], "valid": raw[np.asarray(valid, dtype=np.int64)]}
        if labels == "all":
            y["test"] = raw[np.asarray(test, dtype=np.int64)]
    return {
        "key": key,
        "meta": meta,
        "dataframe": df if labels != "none" else None,
        "columns": list(df.columns),
        "smiles": smiles,
        "y": y,
        "rows": {"train": train, "valid": valid, "test": test},
        "repeats": repeats,
        "selected_repeat": repeats[C.SPLIT_REPEAT_INDEX],
        "csv_path": csv_path,
        "csv_sha256": C.sha256_file(csv_path),
        "source_kind": "chembench",
        "source_commit": CHEMBENCH_COMMIT,
        "labels_scope": labels,
    }


# ---------------------------------------------------------------------------
# OGB ogbg-molhiv
# ---------------------------------------------------------------------------

OGB_DATASET = "ogbg-molhiv"
OGB_DOC_URL = "https://ogb.stanford.edu/docs/graphprop/"
OGB_DATA_URL = "https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/ogbg-molhiv.zip"

OGB_FILES: dict[str, str] = {
    "RELEASE_v1.txt": "release note (dataset version v1, 2020-05-04)",
    "mapping/mol.csv.gz": "official molecular mapping: HIV_active, smiles, mol_id",
    "mapping/README.md": "official note on the mapping file",
    "raw/graph-label.csv.gz": "official per-graph label, row order = OGB graph index",
    "split/scaffold/train.csv.gz": "official scaffold train row ids (single column, published order)",
    "split/scaffold/valid.csv.gz": "official scaffold validation row ids",
    "split/scaffold/test.csv.gz": "official scaffold test row ids",
}


def ogb_path(rel: str) -> Path:
    return C.OGB_SOURCE / rel


def ogb_file_manifest() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel, role in OGB_FILES.items():
        p = ogb_path(rel)
        if not p.exists():
            raise FileNotFoundError(p)
        out[rel] = {"role": role, "path": C.track_rel(p), "size_bytes": int(p.stat().st_size), "sha256": C.sha256_file(p)}
    return out


def ogb_input_semantics() -> dict[str, Any]:
    """Record exactly how the LCRC input differs from OGB's reference input."""
    return {
        "ogb_reference_input": (
            "PyG graphs with 9-dimensional node features (atomic number, chirality, degree, formal "
            "charge, num H, radical electrons, hybridisation, aromaticity, ring membership) and "
            "3-dimensional edge features (bond type, stereo, conjugation), plus pre-computed "
            "num-node/num-edge lists; the official examples feed these tensors straight into a GNN."
        ),
        "lcrc_input": (
            "Graphs rebuilt from the official SMILES with the frozen lipo_v1 adapter: atom_class "
            "(element x aromatic, 15 classes), formal-charge bucket, total-H bucket, directed bond "
            "class/stereo, plus the frozen lipo_v1 per-atom scalars and global descriptors that the "
            "FeatureSpec derives from the RDKit molecule."
        ),
        "not_used": (
            "ogbg-molhiv node-feat.csv.gz / edge-feat.csv.gz / num-*-list.csv.gz are NOT read as model "
            "input, and the 324 MB PyG processed artifact is not used at all; using them would change "
            "the method definition (the model would consume OGB's feature engineering instead of the "
            "frozen LCRC structural encoding)."
        ),
        "sample_set_identical": True,
        "split_identical": True,
        "consequence": (
            "A difference against an OGB leaderboard number cannot be attributed to the split; it can "
            "be attributed to (a) the input featurisation and (b) the model family. This round reports "
            "it that way instead of claiming an exact-match win."
        ),
    }


def load_ogb_hiv(*, verify: bool = True, labels: str = "train_valid") -> dict[str, Any]:
    """Official ogbg-molhiv: SMILES from the mapping file, official scaffold split."""
    if labels not in ("none", "train_valid", "all"):
        raise ValueError(f"labels must be none/train_valid/all, got {labels!r}")
    meta = C.dataset_entry("ogbg-molhiv")
    files = ogb_file_manifest() if verify else {}

    split: dict[str, list[int]] = {}
    for name in ("train", "valid", "test"):
        arr = pd.read_csv(ogb_path(f"split/scaffold/{name}.csv.gz"), header=None).to_numpy().reshape(-1)
        split[name] = [int(x) for x in arr]  # published order preserved, never sorted

    mapping = pd.read_csv(ogb_path("mapping/mol.csv.gz"))
    if list(mapping.columns)[:2] != ["HIV_active", "smiles"]:
        raise ValueError(f"unexpected mapping columns {list(mapping.columns)}")
    n = len(mapping)
    if n != int(meta["expected_rows"]):
        raise ValueError(f"ogbg-molhiv: {n} mapping rows != expected {meta['expected_rows']}")

    for name, part in split.items():
        if len(set(part)) != len(part):
            raise ValueError(f"{name}: duplicate row ids")
        if min(part) < 0 or max(part) >= n:
            raise ValueError(f"{name}: row id outside [0, {n})")
    union = set(split["train"]) | set(split["valid"]) | set(split["test"])
    if len(union) != n:
        raise ValueError(f"split union {len(union)} != mapping rows {n}")
    sizes = (len(split["train"]), len(split["valid"]), len(split["test"]))
    if sizes != tuple(meta["expected_split_sizes"]):
        raise ValueError(f"split sizes {sizes} != frozen {meta['expected_split_sizes']}")

    smiles = [str(s) for s in mapping["smiles"].tolist()]
    y: dict[str, np.ndarray] | None = None
    label_agreement = None
    if labels != "none":
        graph_label = pd.read_csv(ogb_path("raw/graph-label.csv.gz"), header=None).to_numpy().reshape(-1, 1)
        if len(graph_label) != n:
            raise ValueError("graph-label rows != mapping rows")
        # the mapping file carries the same label; require agreement so that a
        # reordered mapping file would be detected instead of silently used
        same = np.asarray(mapping["HIV_active"].to_numpy(dtype=np.float64).reshape(-1, 1) == graph_label)
        label_agreement = {
            "rows_compared": int(n),
            "mismatches": int((~same).sum()),
            "identical": bool(same.all()),
        }
        if not same.all():
            raise ValueError("mapping HIV_active disagrees with raw/graph-label.csv.gz")
        y = {
            "train": graph_label[np.asarray(split["train"], dtype=np.int64)],
            "valid": graph_label[np.asarray(split["valid"], dtype=np.int64)],
        }
        if labels == "all":
            y["test"] = graph_label[np.asarray(split["test"], dtype=np.int64)]
    return {
        "key": "ogbg-molhiv",
        "meta": meta,
        "dataframe": mapping if labels != "none" else None,
        "columns": list(mapping.columns),
        "smiles": smiles,
        "y": y,
        "rows": split,
        "repeats": [{"file": "split/scaffold/{train,valid,test}.csv.gz", "sha256": files["split/scaffold/train.csv.gz"]["sha256"]}],
        "selected_repeat": {"file": "split/scaffold", "sha256": files["split/scaffold/train.csv.gz"]["sha256"]},
        "csv_path": ogb_path("mapping/mol.csv.gz"),
        "csv_sha256": C.sha256_file(ogb_path("mapping/mol.csv.gz")),
        "source_kind": "ogb",
        "source_commit": "ogbg-molhiv release v1 (2020-05-04)",
        "labels_scope": labels,
        "files": files,
        "label_agreement": label_agreement,
        "n_positive": int(y["train"].sum() + y["valid"].sum()) if y is not None else None,
        "mol_id_column_all_empty": bool(mapping["mol_id"].isna().all()),
    }


# ---------------------------------------------------------------------------
# unified entry point
# ---------------------------------------------------------------------------


def load_dataset(key: str, *, verify: bool = True, labels: str = "train_valid") -> dict[str, Any]:
    if C.dataset_entry(key)["source"] == "ogb":
        return load_ogb_hiv(verify=verify, labels=labels)
    return load_chembench(key, verify=verify, labels=labels)


def all_smiles(key: str) -> list[str]:
    """Label-free full SMILES column (used by the raw-attribute artifact)."""
    return load_dataset(key, labels="none")["smiles"]


# ---------------------------------------------------------------------------
# source manifest
# ---------------------------------------------------------------------------


def build_source_manifest() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for rel, (blob, vendor) in sorted(SOURCE_GIT_BLOBS.items()):
        path = chembench_path(rel)
        if not path.exists():
            raise FileNotFoundError(path)
        files[rel] = {
            "vendor": vendor,
            "path": C.track_rel(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": C.sha256_file(path),
            "git_blob_sha1": blob,
            "url": f"{CHEMBENCH_RAW_BASE}/src/chembench/data_and_index/{rel}",
        }
    per_dataset: dict[str, Any] = {}
    for key in C.DATASET_ORDER:
        meta = C.DATASETS[key]
        if meta["source"] != "chembench":
            continue
        task = meta["chembench_task"]
        stem = meta["split_stem"]
        repeats = [
            {
                "repeat_file_index": suffix,
                "file": f"{task}/{stem}{suffix}.ind.pkl",
                "sha256": files[f"{task}/{stem}{suffix}.ind.pkl"]["sha256"],
            }
            for suffix in (122, 123, 124)
        ]
        per_dataset[key] = {
            "chembench_task": task,
            "csv": files[f"{task}/{task}.csv.gz"],
            "split_repeats": repeats,
            "selected_repeat_index": C.SPLIT_REPEAT_INDEX,
            "selected_repeat_file_index": C.SPLIT_REPEAT_FILE_INDEX,
            "selection_rule": (
                f"repeat 0 = the lexicographically first published split file (suffix "
                f"{C.SPLIT_REPEAT_FILE_INDEX}); ChemBench load_data() globs *.pkl with no guaranteed "
                "order, so the file name -- not glob order -- is the frozen identifier"
            ),
        }
    return {
        "schema": "lcrc-molnet-v2-source-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "chembench": {
            "repository": CHEMBENCH_URL,
            "commit": CHEMBENCH_COMMIT,
            "commit_date": CHEMBENCH_COMMIT_DATE,
            "raw_base": CHEMBENCH_RAW_BASE,
            "load_data_reference": (
                "from chembench import load_data; df, split_repeats = load_data(<task>); "
                "train_idx, valid_idx, test_idx = split_repeats[0]"
            ),
            "vendored_roots": {
                "v1 (read-only reuse)": C.track_rel(C.CHEMBENCH_V1),
                "v2 (new in this round)": C.track_rel(C.CHEMBENCH_V2),
            },
            "files": files,
            "datasets": per_dataset,
        },
        "ogb": {
            "dataset": OGB_DATASET,
            "documentation": OGB_DOC_URL,
            "download_url": OGB_DATA_URL,
            "release": "v1 (RELEASE_v1.txt: 2020-05-04)",
            "official_split_api": "ogb.graphproppred.dataset.GraphPropDataset('ogbg-molhiv').get_idx_split()",
            "official_evaluator": "ogb.graphproppred.Evaluator(name='ogbg-molhiv')",
            "vendored_from": "the same CSV files that OGB's csv_mol_download zip ships",
            "files": ogb_file_manifest(),
            "input_semantics": ogb_input_semantics(),
            "note": (
                "the 324 MB PyG processed artifact under data/ogb/ is not committed and is not used; "
                "graphs are rebuilt from the official SMILES with the frozen lipo_v1 adapter"
            ),
        },
        "label_columns": {k: list(v["label_columns"]) for k, v in C.DATASETS.items()},
    }


def load_source_manifest(*, verify: bool = True) -> dict[str, Any]:
    if not C.SOURCE_MANIFEST.exists():
        raise FileNotFoundError(f"{C.SOURCE_MANIFEST} is missing; run code/mb2_source.py --write")
    manifest = C.read_json(C.SOURCE_MANIFEST)
    if manifest.get("chembench", {}).get("commit") != CHEMBENCH_COMMIT:
        raise ValueError("source manifest ChemBench commit != mb2_source.CHEMBENCH_COMMIT")
    if verify:
        for rel, info in manifest["chembench"]["files"].items():
            actual = C.sha256_file(chembench_path(rel))
            if actual != info["sha256"]:
                raise ValueError(f"sha256 mismatch for {rel}: {actual} != {info['sha256']}")
        for rel, info in manifest["ogb"]["files"].items():
            actual = C.sha256_file(ogb_path(rel))
            if actual != info["sha256"]:
                raise ValueError(f"sha256 mismatch for OGB file {rel}: {actual} != {info['sha256']}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="write / verify the v2 source manifest")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        manifest = build_source_manifest()
        C.write_json(C.SOURCE_MANIFEST, manifest)
        C.log(
            f"wrote {C.track_rel(C.SOURCE_MANIFEST)}: "
            f"{len(manifest['chembench']['files'])} ChemBench files + "
            f"{len(manifest['ogb']['files'])} OGB files"
        )
        return 0
    manifest = load_source_manifest(verify=True)
    C.log(f"source manifest verified: {C.track_rel(C.SOURCE_MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHEMBENCH_COMMIT",
    "CHEMBENCH_COMMIT_DATE",
    "CHEMBENCH_URL",
    "CHEMBENCH_RAW_BASE",
    "SOURCE_GIT_BLOBS",
    "chembench_path",
    "load_chembench",
    "OGB_DATASET",
    "OGB_DOC_URL",
    "OGB_FILES",
    "ogb_path",
    "ogb_file_manifest",
    "ogb_input_semantics",
    "load_ogb_hiv",
    "load_dataset",
    "all_smiles",
    "build_source_manifest",
    "load_source_manifest",
]
