"""Official OGB ``ogbg-mollipo`` acquisition, identity checks and local artifacts.

This module is **local-only** (it needs RDKit, which the GPU hosts do not have;
see ``pyproject.toml``: rdkit lives in the ``lipo`` dependency group).  It does
three things and nothing else:

1. fetch/verify the official OGB archive for ``ogbg-mollipo``
   (``lipophilicity.zip`` from the URL recorded in the OGB
   ``ogb/graphproppred/master.csv`` metadata at commit ``OGB_MASTER_COMMIT``),
2. prove the *identity* of the data this round uses - SMILES, target, task rows,
   official scaffold split - against several independent official bytes,
3. write the committed, label-free raw-attribute artifact, the committed label
   array and the committed chemprop-native D-MPNN feature archive.

No split is generated here: the official scaffold indices are read verbatim and
compared against ``GraphPropPredDataset('ogbg-mollipo').get_idx_split()``.

Usage
-----
    uv run python .../code/lp_source.py            # build + verify + manifest
    uv run python .../code/lp_source.py --check    # verify only
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import lp_common as C

RAW_FIELDS = (
    "mol_index",
    "mol_id",
    "smiles",
    "canonical_smiles",
    "atom_class",
    "atom_charge",
    "atom_h",
    "edge_index",
    "edge_class",
    "edge_stereo",
    "n_components",
)
RAW_SCHEMA = "lcrc-ogb-mollipo-rawgraph-v1"

DOWNLOAD_DIR = C.DATA / "download"


# ---------------------------------------------------------------------------
# official bytes
# ---------------------------------------------------------------------------


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    tmp.replace(path)
    return path


def ensure_archive(*, rebuild: bool = False) -> Path:
    """Download the official archive once and verify its SHA-256."""
    archive = DOWNLOAD_DIR / "lipophilicity.zip"
    if rebuild or not archive.is_file():
        C.log(f"downloading {C.OGB_ZIP_URL}")
        _download(C.OGB_ZIP_URL, archive)
    digest = C.sha256_file(archive)
    if digest != C.OGB_ZIP_SHA256:
        raise SystemExit(
            f"official archive sha256 {digest} != frozen {C.OGB_ZIP_SHA256}; "
            "the upstream bytes changed, so this round's data identity is broken"
        )
    extract = DOWNLOAD_DIR / "extracted"
    marker = extract / C.OGB_DOWNLOAD_NAME / "mapping/mol.csv.gz"
    if rebuild or not marker.is_file():
        import zipfile

        if extract.exists():
            shutil.rmtree(extract)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extract)
    return extract / C.OGB_DOWNLOAD_NAME


def master_csv_entry() -> dict[str, Any]:
    """The metadata row that defines ``ogbg-mollipo`` (task type metric split).

    The metadata file is fetched once from the pinned OGB commit and cached next
    to the archive, so the round does not depend on network availability when it
    is re-verified on a compute host.
    """
    import csv
    import io
    import subprocess

    cached = DOWNLOAD_DIR / "ogb_graphproppred_master.csv"
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["curl", "-sSfL", "--max-time", "60", C.OGB_MASTER_CSV_URL, "-o", str(cached)], check=True
        )
    text = cached.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    col = header.index(C.OGB_NAME)
    out: dict[str, Any] = {
        "url_of_metadata": C.OGB_MASTER_CSV_URL,
        "column": C.OGB_NAME,
        "cached_path": C.track_rel(cached),
        "metadata_sha256": C.sha256_bytes(text.encode("utf-8")),
    }
    for row in rows[1:]:
        out[row[0]] = row[col]
    return out


# ---------------------------------------------------------------------------
# official contents
# ---------------------------------------------------------------------------


def read_mol_csv(root: Path) -> dict[str, Any]:
    path = root / "mapping/mol.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        rows = [line.rstrip("\n").split(",") for line in handle if line.strip()]
    if header != [C.TARGET_COLUMN, "smiles", "mol_id"]:
        raise SystemExit(f"unexpected mol.csv header {header}")
    smiles = [r[1] for r in rows]
    y = np.asarray([float(r[0]) for r in rows], dtype=np.float64)
    mol_ids = [r[2] for r in rows]
    if len(smiles) != C.N_MOLECULES:
        raise SystemExit(f"mol.csv rows {len(smiles)} != {C.N_MOLECULES}")
    if len(set(smiles)) != len(smiles):
        raise SystemExit("mol.csv contains duplicated SMILES; row identity is ambiguous")
    if len(set(mol_ids)) != len(mol_ids):
        raise SystemExit("mol.csv contains duplicated mol_id")
    return {
        "path": path,
        "sha256": C.sha256_file(path),
        "smiles": smiles,
        "y": y,
        "mol_ids": mol_ids,
    }


def read_source_labels(root: Path) -> np.ndarray:
    """The OGB raw label column (independent bytes from ``mol.csv``)."""
    path = root / "raw/graph-label.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        values = [float(line.strip()) for line in handle if line.strip()]
    return np.asarray(values, dtype=np.float64)


def read_source_sizes(root: Path) -> tuple[np.ndarray, np.ndarray]:
    def _read(name: str) -> np.ndarray:
        with gzip.open(root / f"raw/{name}", "rt", encoding="utf-8") as handle:
            return np.asarray([int(float(line.strip())) for line in handle if line.strip()], dtype=np.int64)

    return _read("num-node-list.csv.gz"), _read("num-edge-list.csv.gz")


def read_split_files(root: Path) -> dict[str, Any]:
    """Official scaffold split indices, read verbatim from the archive."""
    out: dict[str, Any] = {}
    for split in C.SPLITS:
        path = root / f"split/{C.OGB_SPLIT_METHOD}/{split}.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            idx = [int(float(line.strip())) for line in handle if line.strip()]
        out[split] = np.asarray(idx, dtype=np.int64)
        out[f"{split}_sha256"] = C.sha256_file(path)
    return out


def _graph_stat(graph: Any, name: str) -> int:
    """Node/edge count of one OGB graph object, tolerant of the container type."""
    if name == "num_nodes":
        if isinstance(graph, Mapping) and "num_nodes" in graph:
            return int(np.asarray(graph["num_nodes"]).reshape(-1)[0])
        return int(graph.number_of_nodes())
    if isinstance(graph, Mapping) and "edge_index" in graph:
        return int(np.asarray(graph["edge_index"]).shape[1] // 2)
    return int(graph.number_of_edges())


def official_reader_split() -> dict[str, Any] | None:
    """``GraphPropPredDataset('ogbg-mollipo').get_idx_split()``, if available."""
    try:
        import warnings

        from ogb.graphproppred import GraphPropPredDataset

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = GraphPropPredDataset(name=C.OGB_NAME, root=str(DOWNLOAD_DIR / "ogb_reader"))
            split = dataset.get_idx_split()
            labels = np.asarray(dataset.labels, dtype=np.float64)
            graphs = [item[0] if isinstance(item, tuple) else item for item in dataset]
            num_nodes = np.asarray([int(_graph_stat(g, "num_nodes")) for g in graphs], dtype=np.int64)
            num_edges = np.asarray([int(_graph_stat(g, "num_edges")) for g in graphs], dtype=np.int64)
        return {
            "available": True,
            "n": int(len(dataset)),
            "num_tasks": int(dataset.num_tasks),
            "task_type": str(dataset.task_type),
            "eval_metric": str(dataset.eval_metric),
            "splits": {k: np.asarray(v, dtype=np.int64) for k, v in split.items()},
            "labels": labels.reshape(-1),
            "num_nodes": np.asarray(num_nodes, dtype=np.int64).reshape(-1),
            "num_edges": np.asarray(num_edges, dtype=np.int64).reshape(-1),
        }
    except Exception as exc:  # pragma: no cover - optional cross-check
        C.log(f"official reader unavailable: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# raw attributes (RDKit; local only)
# ---------------------------------------------------------------------------


def build_raw_rows(smiles: Sequence[str], mol_ids: Sequence[str]) -> list[dict[str, Any]]:
    import mb_rdkit as R

    rows: list[dict[str, Any]] = []
    for index, smi in enumerate(smiles):
        raw = R.smiles_to_raw_row(str(smi), int(index))
        raw["mol_id"] = str(mol_ids[index])
        rows.append(raw)
    return rows


def _jsonable(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in RAW_FIELDS:
        value = row[key]
        if isinstance(value, np.ndarray):
            out[key] = np.asarray(value).tolist()
        elif isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def write_raw_artifact(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema": RAW_SCHEMA,
        "dataset": C.DATASET,
        "label_free": True,
        "n_molecules": len(rows),
        "row_order": "OGB mol.csv.gz row order (0-based); split indices index this order",
        "fields": list(RAW_FIELDS),
        "rows": [_jsonable(row) for row in rows],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    C.RAW_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(C.RAW_ARTIFACT, "wb", compresslevel=9) as handle:
        handle.write(text)
    if b'"exp"' in text:
        raise SystemExit("raw artifact would carry a label field")
    return {
        "path": C.track_rel(C.RAW_ARTIFACT),
        "sha256": C.sha256_file(C.RAW_ARTIFACT),
        "bytes": C.RAW_ARTIFACT.stat().st_size,
        "schema": RAW_SCHEMA,
        "label_free": True,
    }


def load_raw_artifact(*, verify: bool = True) -> list[dict[str, Any]]:
    with gzip.open(C.RAW_ARTIFACT, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != RAW_SCHEMA:
        raise SystemExit(f"raw artifact schema {payload.get('schema')} != {RAW_SCHEMA}")
    if payload.get("label_free") is not True:
        raise SystemExit("raw artifact is not declared label-free")
    rows = payload["rows"]
    if len(rows) != C.N_MOLECULES:
        raise SystemExit(f"raw artifact rows {len(rows)} != {C.N_MOLECULES}")
    if verify:
        for i, row in enumerate(rows):
            if int(row["mol_index"]) != i:
                raise SystemExit(f"raw artifact row {i} has mol_index {row['mol_index']}")
            if "exp" in row:
                raise SystemExit(f"raw artifact row {i} carries a label")
    return rows


# ---------------------------------------------------------------------------
# labels / split / dmpnn features
# ---------------------------------------------------------------------------


def write_labels(y: np.ndarray, mol_ids: Sequence[str]) -> dict[str, Any]:
    np.savez_compressed(
        C.LABELS,
        y=np.asarray(y, dtype=np.float64),
        mol_id=np.asarray(list(mol_ids), dtype="U32"),
    )
    return {"path": C.track_rel(C.LABELS), "sha256": C.sha256_file(C.LABELS), "n": int(len(y))}


def load_labels() -> dict[str, Any]:
    with np.load(C.LABELS, allow_pickle=False) as handle:
        return {"y": np.asarray(handle["y"], dtype=np.float64), "mol_id": [str(x) for x in handle["mol_id"]]}


def write_split(split: Mapping[str, Any], checks: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "dataset": C.DATASET,
        "split_method": C.OGB_SPLIT_METHOD,
        "sizes": {k: int(len(split[k])) for k in C.SPLITS},
        "rows": {k: [int(x) for x in np.asarray(split[k]).tolist()] for k in C.SPLITS},
        "rows_sha256": {k: C.row_sha256(np.asarray(split[k]).tolist()) for k in C.SPLITS},
        "source_file_sha256": {k: split[f"{k}_sha256"] for k in C.SPLITS},
        "verification": dict(checks),
    }
    if payload["sizes"] != dict(C.SPLIT_SIZES):
        raise SystemExit(f"official split sizes {payload['sizes']} != frozen {C.SPLIT_SIZES}")
    C.write_json(C.SPLIT_JSON, payload)
    return {"path": C.track_rel(C.SPLIT_JSON), "sha256": C.sha256_file(C.SPLIT_JSON), "sizes": payload["sizes"]}


def load_split() -> dict[str, Any]:
    payload = C.read_json(C.SPLIT_JSON)
    return {
        "rows": {k: np.asarray(payload["rows"][k], dtype=np.int64) for k in C.SPLITS},
        "rows_sha256": dict(payload["rows_sha256"]),
        "source_file_sha256": dict(payload["source_file_sha256"]),
        "sizes": dict(payload["sizes"]),
        "verification": dict(payload["verification"]),
    }


def build_dmpnn_features(smiles: Sequence[str]) -> dict[str, Any]:
    """chemprop-native (72, 14) featurisation in ``mol.csv`` row order."""
    import lcpr_dmpnn as PDN

    V_parts, E_parts, ei_parts, rev_parts = [], [], [], []
    atom_off = 0
    edge_off = 0
    mol_atom_ptr = [0]
    mol_edge_ptr = [0]
    for smi in smiles:
        V, E, ei, rev = PDN.molgraph(str(smi))
        V_parts.append(V)
        E_parts.append(E)
        ei_parts.append(ei + atom_off if ei.size else ei)
        rev_parts.append(rev + edge_off)
        atom_off += int(V.shape[0])
        edge_off += int(E.shape[0])
        mol_atom_ptr.append(atom_off)
        mol_edge_ptr.append(edge_off)
    V = np.concatenate(V_parts, axis=0)
    E = np.concatenate(E_parts, axis=0) if E_parts else np.zeros((0, PDN.BOND_FDIM))
    edge_index = np.concatenate(ei_parts, axis=1) if ei_parts else np.zeros((2, 0), int)
    rev_edge_index = np.concatenate(rev_parts) if rev_parts else np.zeros(0, int)
    if (V.shape[1], E.shape[1]) != (PDN.ATOM_FDIM, PDN.BOND_FDIM):
        raise SystemExit("chemprop feature widths drifted")
    np.savez_compressed(
        C.DMPNN_FEATURES,
        V=V.astype(np.float32),
        E=E.astype(np.float32),
        edge_index=edge_index.astype(np.int32),
        rev_edge_index=rev_edge_index.astype(np.int32),
        mol_atom_ptr=np.asarray(mol_atom_ptr, np.int64),
        mol_edge_ptr=np.asarray(mol_edge_ptr, np.int64),
        n_molecules=np.int64(len(smiles)),
        n_atoms=np.int64(V.shape[0]),
        n_edges=np.int64(E.shape[0]),
        atom_fdim=np.int64(PDN.ATOM_FDIM),
        bond_fdim=np.int64(PDN.BOND_FDIM),
    )
    return {
        "path": C.track_rel(C.DMPNN_FEATURES),
        "sha256": C.sha256_file(C.DMPNN_FEATURES),
        "bytes": C.DMPNN_FEATURES.stat().st_size,
        "n_atoms": int(V.shape[0]),
        "n_directed_edges": int(E.shape[0]),
        "feature_version": PDN.FEATURE_VERSION,
        "atom_fdim": int(PDN.ATOM_FDIM),
        "bond_fdim": int(PDN.BOND_FDIM),
    }


# ---------------------------------------------------------------------------
# verification + manifest
# ---------------------------------------------------------------------------


def verify_identity(root: Path, mol: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    """All the cross-checks that make the row/SMILES/label/split identity explicit."""
    y = np.asarray(mol["y"])
    smiles = list(mol["smiles"])
    checks: dict[str, Any] = {}

    source_labels = read_source_labels(root)
    checks["labels_equal_raw_graph_label_csv"] = bool(
        source_labels.shape == y.shape and np.array_equal(source_labels, y)
    )

    node_counts, edge_counts = read_source_sizes(root)
    checks["n_node_list_matches"] = bool(node_counts.shape == (C.N_MOLECULES,))
    checks["n_edge_list_matches"] = bool(edge_counts.shape == (C.N_MOLECULES,))

    raw = load_raw_artifact()
    atom_ok = all(int(len(r["atom_class"])) == int(node_counts[i]) for i, r in enumerate(raw))
    bond_ok = all(int(len(r["edge_class"])) == 2 * int(edge_counts[i]) for i, r in enumerate(raw))
    checks["raw_atom_count_matches_ogb_num_node"] = bool(atom_ok)
    checks["raw_directed_edge_count_matches_ogb_2x_num_edge"] = bool(bond_ok)
    checks["raw_smiles_equal_mol_csv"] = all(str(r["smiles"]) == smiles[i] for i, r in enumerate(raw))
    checks["raw_mol_id_equal_mol_csv"] = all(str(r["mol_id"]) == mol["mol_ids"][i] for i, r in enumerate(raw))

    canon = [str(r["canonical_smiles"]) for r in raw]
    checks["canonical_smiles_unique"] = len(set(canon)) == C.N_MOLECULES

    tr, va, te = (np.asarray(split[k], dtype=np.int64) for k in C.SPLITS)
    checks["split_sizes"] = {"train": int(tr.size), "valid": int(va.size), "test": int(te.size)}
    checks["split_sizes_match_frozen"] = checks["split_sizes"] == dict(C.SPLIT_SIZES)
    checks["split_disjoint"] = bool(
        not (set(tr.tolist()) & set(va.tolist()))
        and not (set(tr.tolist()) & set(te.tolist()))
        and not (set(va.tolist()) & set(te.tolist()))
    )
    union = sorted(set(tr.tolist()) | set(va.tolist()) | set(te.tolist()))
    checks["split_covers_all_rows"] = union == list(range(C.N_MOLECULES))
    checks["split_no_random_split_used"] = True
    checks["split_source"] = f"official OGB {C.OGB_SPLIT_METHOD} index files (read verbatim)"

    reader = official_reader_split()
    if reader is None:
        checks["official_reader_matches_split_files"] = None
        checks["official_reader_note"] = "GraphPropPredDataset cross-check unavailable on this host"
    else:
        ok = all(
            np.array_equal(np.asarray(reader["splits"][k]), np.asarray(split[k])) for k in C.SPLITS
        )
        checks["official_reader_matches_split_files"] = bool(ok)
        checks["official_reader"] = {
            "n": reader["n"],
            "num_tasks": reader["num_tasks"],
            "task_type": reader["task_type"],
            "eval_metric": reader["eval_metric"],
        }
        checks["official_reader_labels_equal_mol_csv"] = bool(
            np.array_equal(np.asarray(reader["labels"]).reshape(-1), y)
        )
        checks["official_reader_num_nodes_match"] = bool(
            np.array_equal(np.asarray(reader["num_nodes"]).reshape(-1), np.asarray(node_counts))
        )
        checks["official_reader_num_edges_match"] = bool(
            np.array_equal(np.asarray(reader["num_edges"]).reshape(-1), np.asarray(edge_counts))
        )
        if not ok:
            raise SystemExit("the archive split files disagree with the official OGB reader")

    hard = [
        "labels_equal_raw_graph_label_csv",
        "n_node_list_matches",
        "n_edge_list_matches",
        "raw_atom_count_matches_ogb_num_node",
        "raw_directed_edge_count_matches_ogb_2x_num_edge",
        "raw_smiles_equal_mol_csv",
        "raw_mol_id_equal_mol_csv",
        "canonical_smiles_unique",
        "split_sizes_match_frozen",
        "split_disjoint",
        "split_covers_all_rows",
    ]
    failed = [name for name in hard if not checks.get(name)]
    if failed:
        raise SystemExit(f"data identity checks failed: {failed}")
    return checks


def build(*, rebuild: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    root = ensure_archive(rebuild=rebuild)
    mol = read_mol_csv(root)
    split = read_split_files(root)
    reader = official_reader_split()
    if reader is not None:
        for k in C.SPLITS:
            if not np.array_equal(np.asarray(reader["splits"][k]), np.asarray(split[k])):
                raise SystemExit(f"official reader split {k} != archive split file")
    raw_rows = build_raw_rows(mol["smiles"], mol["mol_ids"])
    raw_meta = write_raw_artifact(raw_rows)
    labels_meta = write_labels(mol["y"], mol["mol_ids"])
    feat_meta = build_dmpnn_features(mol["smiles"])
    checks = verify_identity(root, mol, split)
    split_meta = write_split(split, checks)
    master = master_csv_entry()
    if master.get("download_name") != C.OGB_DOWNLOAD_NAME:
        raise SystemExit(f"OGB metadata download_name {master.get('download_name')} != {C.OGB_DOWNLOAD_NAME}")
    if master.get("url") != C.OGB_ZIP_URL:
        raise SystemExit(f"OGB metadata url {master.get('url')} != {C.OGB_ZIP_URL}")

    manifest = {
        "schema": "lcrc-lipo-context-small-dataset-v1",
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "dataset": C.DATASET,
        "task": C.TASK,
        "target_column": C.TARGET_COLUMN,
        "units": C.UNITS,
        "split_method": C.OGB_SPLIT_METHOD,
        "ogb_metadata": master,
        "ogb_archive": {
            "url": C.OGB_ZIP_URL,
            "sha256": C.OGB_ZIP_SHA256,
            "download_name": C.OGB_DOWNLOAD_NAME,
        },
        "files": {
            "mol_csv": {"sha256": mol["sha256"], "rows": C.N_MOLECULES},
            "raw_graph_label_csv": {"sha256": C.sha256_file(root / "raw/graph-label.csv.gz")},
            "num_node_list": {"sha256": C.sha256_file(root / "raw/num-node-list.csv.gz")},
            "num_edge_list": {"sha256": C.sha256_file(root / "raw/num-edge-list.csv.gz")},
            **{f"split_{k}": {"sha256": split[f"{k}_sha256"]} for k in C.SPLITS},
        },
        "artifacts": {
            "raw_attributes": raw_meta,
            "labels": labels_meta,
            "split": split_meta,
            "dmpnn_features": feat_meta,
        },
        "identity_checks": checks,
        "label_statistics": {
            "mean": float(np.mean(mol["y"])),
            "std": float(np.std(mol["y"])),
            "min": float(np.min(mol["y"])),
            "max": float(np.max(mol["y"])),
            "train_mean": float(np.mean(mol["y"][split["train"]])),
            "train_std": float(np.std(mol["y"][split["train"]])),
        },
        "split_policy": (
            "official OGB scaffold indices read verbatim; no random_split, no re-derived "
            "scaffold split, no ChemBench random122 substitution"
        ),
        "test_access": (
            "development-evaluation access declared for this round: Test is evaluated every "
            "epoch and reported; selection stays on Validation"
        ),
        "seconds": float(time.perf_counter() - started),
        "hostname": __import__("platform").node(),
        "git_commit": C.git_commit(),
    }
    C.write_json(C.DATA_MANIFEST, manifest)
    C.log(f"data manifest -> {C.track_rel(C.DATA_MANIFEST)}")
    return manifest


def check() -> dict[str, Any]:
    manifest = C.read_json(C.DATA_MANIFEST)
    for name, entry in manifest["artifacts"].items():
        path = C.REPO / entry["path"]
        if not path.is_file():
            raise SystemExit(f"artifact {name} missing at {entry['path']}")
        if C.sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"artifact {name} sha256 drifted")
    if C.sha256_file(DOWNLOAD_DIR / "lipophilicity.zip") != C.OGB_ZIP_SHA256:
        raise SystemExit("official archive sha256 drifted")
    return {"ok": True, "artifacts": list(manifest["artifacts"])}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OGB ogbg-mollipo: build/verify local artifacts")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(check(), indent=2))
        return 0
    manifest = build(rebuild=bool(args.rebuild))
    print(json.dumps(manifest["artifacts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
