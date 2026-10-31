"""Data views for the random42 protocol.

Three independent structure representations are produced from the same 4,200
raw rows, in original CSV order:

* ``LCRC``  the frozen ``lipo_v1`` S1/S2 cache (rebuilt from the committed
  label-free raw-graph artifact with the read-only preparation code);
* ``ESA/PNA`` the upstream Chemprop one-hot graph (RDKit ``AddHs``) produced by
  the vendored upstream transform chain;
* ``XGB``   ECFP4 + the frozen 217 RDKit descriptors (CPU, local).

The split is the exact ``torch.utils.data.random_split`` call with seed 42.
No label is read here.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import r42_common as C
import r42_labels as L
import r42_upstream as U

if str(C.PREP_CODE) not in sys.path:
    sys.path.insert(0, str(C.PREP_CODE))

import adapters as prep_adapters  # noqa: E402
import feature_spec as prep_feature_spec  # noqa: E402
import lcrc_common as prep_common  # noqa: E402
import lcrc_data as prep_data  # noqa: E402


# ---------------------------------------------------------------------------
# raw structures (no exp column)
# ---------------------------------------------------------------------------


def read_smiles_only(path: Path | str = C.CSV_PATH) -> tuple[list[str], list[str]]:
    """Read compound ids and SMILES only; the ``exp`` column is never stored."""
    path = Path(path)
    C.verify_sha256(path, C.CSV_SHA256, "Lipophilicity.csv")
    ids: list[str] = []
    smiles: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["CMPD_CHEMBLID", "exp", "smiles"]:
            raise ValueError(f"unexpected CSV header {header!r}")
        for i, row in enumerate(reader):
            if len(row) != 3:
                raise ValueError(f"row {i}: expected 3 fields")
            ids.append(row[0])
            smiles.append(row[2])
    if len(ids) != C.N_MOLECULES:
        raise ValueError(f"expected {C.N_MOLECULES} rows, parsed {len(ids)}")
    return ids, smiles


def compute_split() -> dict[str, list[int]]:
    train, valid, test = torch.utils.data.random_split(
        list(range(C.N_MOLECULES)),
        lengths=[C.SPLIT_SIZES["train"], C.SPLIT_SIZES["valid"], C.SPLIT_SIZES["test"]],
        generator=torch.Generator().manual_seed(C.SPLIT_SEED),
    )
    out = {
        "train": [int(x) for x in train.indices],
        "valid": [int(x) for x in valid.indices],
        "test": [int(x) for x in test.indices],
    }
    verify_split(out)
    return out


def verify_split(split: Mapping[str, Sequence[int]]) -> None:
    for name, size in C.SPLIT_SIZES.items():
        rows = [int(x) for x in split[name]]
        if len(rows) != size:
            raise ValueError(f"{name} split has {len(rows)} rows, expected {size}")
        if len(set(rows)) != size:
            raise ValueError(f"{name} split contains duplicates")
        digest = C.rows_sha256_int64(rows)
        if digest != C.SPLIT_SHA256[name]:
            raise ValueError(f"{name} split sha256 {digest} != frozen {C.SPLIT_SHA256[name]}")
        if rows[:5] != C.SPLIT_FIRST_ROWS[name]:
            raise ValueError(f"{name} split first five rows drifted")
    union = set(split["train"]) | set(split["valid"]) | set(split["test"])
    if union != set(range(C.N_MOLECULES)):
        raise ValueError("the union of the three splits is not exactly 0..4199")
    if set(split["train"]) & set(split["valid"]) or set(split["train"]) & set(split["test"]) or set(split["valid"]) & set(split["test"]):
        raise ValueError("split overlap")


def freeze_split(path: Path | str = C.SPLIT_FILE) -> dict[str, Any]:
    ids, smiles = read_smiles_only()
    split = compute_split()
    if C.MULTI_COMPONENT_ROW not in split["train"] + split["valid"] + split["test"]:
        raise ValueError("the multi-component molecule row is missing")
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "source_csv_sha256": C.CSV_SHA256,
        "n_molecules": C.N_MOLECULES,
        "method": "torch.utils.data.random_split(list(range(4200)), [3360, 420, 420], generator=torch.Generator().manual_seed(42))",
        "seed": C.SPLIT_SEED,
        "sizes": dict(C.SPLIT_SIZES),
        "sha256": dict(C.SPLIT_SHA256),
        "hash_definition": "sha256(numpy.asarray(rows, dtype='<i8').tobytes())",
        "first_five_rows": dict(C.SPLIT_FIRST_ROWS),
        "rows": split,
        "compound_ids": ids,
        "smiles": smiles,
        "compound_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "smiles_sha256": hashlib.sha256("\n".join(smiles).encode("utf-8")).hexdigest(),
        "multi_component_row": C.MULTI_COMPONENT_ROW,
        "multi_component_smiles": smiles[C.MULTI_COMPONENT_ROW],
        "multi_component_split": next(name for name in ("train", "valid", "test") if C.MULTI_COMPONENT_ROW in split[name]),
    }
    C.write_json(path, payload)
    return payload


def load_split(path: Path | str = C.SPLIT_FILE) -> dict[str, Any]:
    payload = C.read_json(path)
    if payload.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("split protocol mismatch")
    if payload.get("source_csv_sha256") != C.CSV_SHA256:
        raise PermissionError("split was built from a different CSV")
    verify_split(payload["rows"])
    return payload


# ---------------------------------------------------------------------------
# upstream ESA/PNA structures (Chemprop one-hot, AddHs)
# ---------------------------------------------------------------------------


def build_upstream_structures(smiles: Sequence[str]) -> list[Any]:
    from torch_geometric.data import Data  # noqa: F401
    from torch_geometric.utils import from_smiles

    mods = U.install()
    transforms = mods["transforms"]
    chain = [
        transforms.ChempropFeatures(one_hot=True, max_atomic_number=53),
        transforms.AddMaxEdge(),
        transforms.AddMaxNode(),
    ]
    out: list[Any] = []
    dropped: list[int] = []
    for i, smi in enumerate(smiles):
        data = from_smiles(str(smi))
        for transform in chain:
            data = transform(data)
            if data is None:
                break
        if data is None:
            dropped.append(i)
            continue
        if not hasattr(data, "smiles") or str(data.smiles) != str(smi):
            raise ValueError(f"row {i}: SMILES identity changed during featurisation")
        data.mol_index = int(i)
        out.append(data)
    if dropped:
        raise RuntimeError(f"upstream preprocessing filtered {len(dropped)} molecules: {dropped[:10]}")
    if len(out) != len(smiles):
        raise RuntimeError(f"upstream preprocessing returned {len(out)} / {len(smiles)} molecules")
    return out


def save_structure_artifact(data_list: Sequence[Any], path: Path | str = C.STRUCTURES_FILE, meta_path: Path | str = C.STRUCTURES_META) -> dict[str, Any]:
    atom_ptr = [0]
    bond_ptr = [0]
    max_edges = []
    max_nodes = []
    atom_blocks = []
    edge_blocks = []
    edge_attr_blocks = []
    for i, data in enumerate(data_list):
        if int(data.mol_index) != i:
            raise ValueError(f"structure {i} has mol_index {data.mol_index}")
        x = data.x.detach().cpu().numpy().astype(np.uint8)
        ei = data.edge_index.detach().cpu().numpy()
        ea = data.edge_attr.detach().cpu().numpy().astype(np.uint8)
        if ei.ndim != 2 or ei.shape[0] != 2 or ea.shape[0] != ei.shape[1]:
            raise ValueError(f"structure {i} has inconsistent edge tensors")
        atom_blocks.append(x)
        # Store edge_index as [E, 2] to match the upstream layout after transpose.
        edge_blocks.append(np.ascontiguousarray(ei.T.astype(np.int32)))
        edge_attr_blocks.append(ea)
        atom_ptr.append(atom_ptr[-1] + x.shape[0])
        bond_ptr.append(bond_ptr[-1] + ei.shape[1])
        max_edges.append(int(data.max_edge.item()))
        max_nodes.append(int(data.max_node.item()))
    atom_feat = np.concatenate(atom_blocks, axis=0)
    edge_index = np.concatenate(edge_blocks, axis=0)
    edge_attr = np.concatenate(edge_attr_blocks, axis=0)
    arrays = {
        "atom_ptr": np.asarray(atom_ptr, dtype=np.int64),
        "atom_feat": atom_feat,
        "bond_ptr": np.asarray(bond_ptr, dtype=np.int64),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "max_edge": np.asarray(max_edges, dtype=np.int64),
        "max_node": np.asarray(max_nodes, dtype=np.int64),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta = {
        "protocol_id": C.PROTOCOL_ID,
        "source": "upstream ChempropFeatures(one_hot=True, max_atomic_number=53) + AddMaxEdge + AddMaxNode",
        "upstream_commit": C.UPSTREAM_COMMIT,
        "n_molecules": len(data_list),
        "n_atoms": int(atom_ptr[-1]),
        "n_directed_edges": int(bond_ptr[-1]),
        "atom_feat_dim": int(atom_feat.shape[1]),
        "edge_feat_dim": int(edge_attr.shape[1]),
        "max_edge_global": int(max(max_edges)),
        "max_node_global": int(max(max_nodes)),
        "atom_feat_dtype": str(atom_feat.dtype),
        "edge_index_dtype": str(edge_index.dtype),
        "file_sha256": C.sha256_file(path),
        "arrays": {
            key: {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "sha256": C.sha256_bytes(np.ascontiguousarray(arr).tobytes()),
            }
            for key, arr in arrays.items()
        },
    }
    C.write_json(meta_path, meta)
    return meta


_STRUCT_CACHE: dict[str, Any] = {}


def load_structure_arrays() -> dict[str, np.ndarray]:
    if "arrays" in _STRUCT_CACHE:
        return _STRUCT_CACHE["arrays"]
    if not C.STRUCTURES_FILE.exists():
        raise FileNotFoundError(f"missing {C.track_rel(C.STRUCTURES_FILE)}")
    meta = C.read_json(C.STRUCTURES_META)
    actual = C.sha256_file(C.STRUCTURES_FILE)
    if actual != meta["file_sha256"]:
        raise PermissionError("structure artifact hash mismatch")
    with np.load(C.STRUCTURES_FILE) as npz:
        arrays = {key: npz[key] for key in npz.files}
    _STRUCT_CACHE["arrays"] = arrays
    _STRUCT_CACHE["meta"] = meta
    return arrays


def load_structure_data(rows: Sequence[int] | None = None) -> list[Any]:
    """Reconstruct the per-molecule PyG Data objects in original row order."""
    from torch_geometric.data import Data

    arrays = load_structure_arrays()
    atom_ptr = arrays["atom_ptr"]
    bond_ptr = arrays["bond_ptr"]
    n = int(atom_ptr.size - 1)
    wanted = list(range(n)) if rows is None else [int(r) for r in rows]
    out = []
    for r in wanted:
        if r < 0 or r >= n:
            raise IndexError(r)
        a0, a1 = int(atom_ptr[r]), int(atom_ptr[r + 1])
        b0, b1 = int(bond_ptr[r]), int(bond_ptr[r + 1])
        ei = torch.from_numpy(np.ascontiguousarray(arrays["edge_index"][b0:b1].T)).long()
        data = Data(
            x=torch.from_numpy(np.ascontiguousarray(arrays["atom_feat"][a0:a1])).long(),
            edge_index=ei,
            edge_attr=torch.from_numpy(np.ascontiguousarray(arrays["edge_attr"][b0:b1])).long(),
        )
        data.mol_index = r
        data.max_edge = torch.tensor(int(arrays["max_edge"][r]))
        data.max_node = torch.tensor(int(arrays["max_node"][r]))
        out.append(data)
    return out


def structure_summary() -> dict[str, Any]:
    meta = C.read_json(C.STRUCTURES_META)
    return {
        "path": C.track_rel(C.STRUCTURES_FILE),
        "sha256": meta["file_sha256"],
        "n_molecules": meta["n_molecules"],
        "n_atoms": meta["n_atoms"],
        "n_directed_edges": meta["n_directed_edges"],
        "atom_feat_dim": meta["atom_feat_dim"],
        "edge_feat_dim": meta["edge_feat_dim"],
        "max_edge_global": meta["max_edge_global"],
        "max_node_global": meta["max_node_global"],
    }


# ---------------------------------------------------------------------------
# LCRC structures (lipo_v1 cache)
# ---------------------------------------------------------------------------


def load_raw_graph_artifact() -> dict[str, Any]:
    C.verify_sha256(C.RAW_GRAPH, "8416837ea3e56e3e5218865391c88913bb16948b48b7a61be1ee890ea4dda6dd", "lipo_raw")
    C.verify_sha256(C.ROW_MANIFEST, "2c638e4814cb6d21ccec9d8c8af357d28ce210fb2c20c3d09b9c19afd476073d", "row_manifest")
    raw = prep_adapters.load_lipo_raw_artifact(C.RAW_GRAPH)
    if len(raw["rows"]) != C.N_MOLECULES:
        raise ValueError("raw graph artifact does not contain 4200 rows")
    ids, smiles = read_smiles_only()
    for i, row in enumerate(raw["rows"]):
        if "exp" in row:
            raise PermissionError(f"raw row {i} contains a label field")
        if int(row["mol_index"]) != i:
            raise ValueError(f"raw row {i} has mol_index {row['mol_index']}")
        if str(row["smiles"]) != smiles[i]:
            raise ValueError(f"raw row {i} SMILES differs from the CSV")
        if str(row["compound_id"]) != ids[i]:
            raise ValueError(f"raw row {i} compound id differs from the CSV")
    return raw


def build_lcrc_cache(path: Path | str = C.CACHE_DIR / "lcrc_all4200.pt", *, force: bool = False) -> dict[str, Any]:
    """Build (or load) the full 4,200-molecule lipo_v1 structural cache."""
    path = Path(path)
    spec = prep_feature_spec.lipo_v1_spec()
    if path.exists() and not force:
        cache = prep_data.load_cache(path)
        if int(cache["n_molecules"]) != C.N_MOLECULES:
            raise ValueError("cached LCRC cache has the wrong molecule count")
        return {
            "cache": cache,
            "path": path,
            "file_sha256": C.sha256_file(path),
            "content_fingerprint": prep_common.content_fingerprint(cache),
            "rebuilt": False,
        }
    raw = load_raw_graph_artifact()
    graphs = [prep_adapters.lipo_row_to_graph(raw["rows"][i], i, spec) for i in range(C.N_MOLECULES)]
    cache = prep_data.build_cache(graphs, spec, split="random42_all4200")
    if cache["meta"].get("labels_present") is not False:
        raise PermissionError("LCRC cache is not certified label-free")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = prep_data.save_cache(cache, path)
    return {
        "cache": cache,
        "path": path,
        "file_sha256": digest,
        "content_fingerprint": prep_common.content_fingerprint(cache),
        "rebuilt": True,
    }


# ---------------------------------------------------------------------------
# cache subsetting (adapted from v2_data.select_cache; structure-only)
# ---------------------------------------------------------------------------


def select_cache(cache: Mapping[str, Any], positions: Sequence[int], *, split_label: str) -> dict[str, Any]:
    pos = [int(p) for p in positions]
    if len(pos) != len(set(pos)):
        raise ValueError("duplicate cache position")
    n = int(cache["n_molecules"])
    if any(p < 0 or p >= n for p in pos):
        raise IndexError("cache position outside the frozen cache")

    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)

    atom_ranges = [(int(atom_ptr[p]), int(atom_ptr[p + 1])) for p in pos]
    bond_ranges = [(int(bond_ptr[p]), int(bond_ptr[p + 1])) for p in pos]
    nbr_ranges = [(int(nbr_ptr[a0]), int(nbr_ptr[a1])) for a0, a1 in atom_ranges]
    path_ranges = [(int(path_ptr[a0]), int(path_ptr[a1])) for a0, a1 in atom_ranges]

    out: dict[str, Any] = {}
    out["atom_feat"] = np.concatenate([np.asarray(cache["atom_feat"])[a0:a1] for a0, a1 in atom_ranges], axis=0)
    out["atom_type"] = np.concatenate([np.asarray(cache["atom_type"])[a0:a1] for a0, a1 in atom_ranges], axis=0)
    out["atom_scalar"] = np.concatenate([np.asarray(cache["atom_scalar"])[a0:a1] for a0, a1 in atom_ranges], axis=0)
    out["bond_feat"] = np.concatenate([np.asarray(cache["bond_feat"])[b0:b1] for b0, b1 in bond_ranges], axis=0)
    out["bond_type"] = np.concatenate([np.asarray(cache["bond_type"])[b0:b1] for b0, b1 in bond_ranges], axis=0)
    out["bond_index"] = np.concatenate([np.asarray(cache["bond_index"])[b0:b1] for b0, b1 in bond_ranges], axis=0)
    out["nbr_code"] = np.concatenate([np.asarray(cache["nbr_code"])[n0:n1] for n0, n1 in nbr_ranges], axis=0)
    out["path_cat"] = np.concatenate([np.asarray(cache["path_cat"])[p0:p1] for p0, p1 in path_ranges], axis=0)
    out["path_end63"] = np.concatenate([np.asarray(cache["path_end63"])[p0:p1] for p0, p1 in path_ranges], axis=0)
    out["path_end"] = np.concatenate([np.asarray(cache["path_end"])[p0:p1] for p0, p1 in path_ranges], axis=0)
    out["path_closure"] = np.concatenate([np.asarray(cache["path_closure"])[p0:p1] for p0, p1 in path_ranges], axis=0)
    out["global"] = np.asarray(cache["global"])[np.asarray(pos, dtype=np.int64)].copy()
    out["n_components"] = np.asarray(cache["n_components"])[np.asarray(pos, dtype=np.int64)].copy()
    out["n_junctions"] = np.asarray(cache["n_junctions"])[np.asarray(pos, dtype=np.int64)].copy()
    out["n_ring_systems"] = np.asarray(cache["n_ring_systems"])[np.asarray(pos, dtype=np.int64)].copy()
    out["molecule_ids"] = [str(cache["molecule_ids"][p]) for p in pos]
    out["smiles"] = [str(cache["smiles"][p]) for p in pos]

    atom_counts = np.asarray([a1 - a0 for a0, a1 in atom_ranges], dtype=np.int64)
    bond_counts = np.asarray([b1 - b0 for b0, b1 in bond_ranges], dtype=np.int64)
    out["atom_ptr"] = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int64)
    out["bond_ptr"] = np.concatenate([[0], np.cumsum(bond_counts)]).astype(np.int64)
    nbr_counts = np.concatenate(
        [np.asarray(cache["nbr_ptr"])[a0 + 1 : a1 + 1] - np.asarray(cache["nbr_ptr"])[a0:a1] for a0, a1 in atom_ranges]
    ) if atom_ranges else np.zeros(0, dtype=np.int64)
    path_counts = np.concatenate(
        [np.asarray(cache["path_ptr"])[a0 + 1 : a1 + 1] - np.asarray(cache["path_ptr"])[a0:a1] for a0, a1 in atom_ranges]
    ) if atom_ranges else np.zeros(0, dtype=np.int64)
    out["nbr_ptr"] = np.concatenate([[0], np.cumsum(nbr_counts)]).astype(np.int64)
    out["path_ptr"] = np.concatenate([[0], np.cumsum(path_counts)]).astype(np.int64)

    out["n_molecules"] = len(pos)
    meta = dict(cache.get("meta", {}))
    meta["n_molecules"] = len(pos)
    meta["subset_of"] = str(cache.get("split", ""))
    meta["subset_positions_sha256"] = C.rows_sha256_int64(pos)
    out["meta"] = meta
    out["split"] = split_label
    return out


def fit_target_scaler(y_train: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("invalid Train3360 labels")
    mean = float(np.mean(y))
    scale = float(np.std(y, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid target scaler")
    return {
        "source": "train3360_labels_only",
        "valid_or_test_used": False,
        "n": int(y.size),
        "mean": mean,
        "scale": scale,
        "scale_definition": "population_std_float64 (sklearn StandardScaler convention)",
        "label_vector_sha256": C.sha256_bytes(np.ascontiguousarray(y, dtype="<f8").tobytes()),
    }


def build_lcrc_views(split: Mapping[str, Any], labels: L.TrainValidLabels) -> dict[str, Any]:
    built = build_lcrc_cache()
    cache_all = built["cache"]
    rows_train = [int(r) for r in split["rows"]["train"]]
    rows_valid = [int(r) for r in split["rows"]["valid"]]
    # The full cache is in original row order, so cache position == CSV row.
    train_cache = select_cache(cache_all, rows_train, split_label="train3360")
    valid_cache = select_cache(cache_all, rows_valid, split_label="valid420")
    # lipo_v1 stores the original CSV row index as ``molecule_id``.
    if [str(x) for x in train_cache["molecule_ids"]] != [str(r) for r in rows_train]:
        raise ValueError("Train3360 cache row identity drifted")
    if [str(x) for x in valid_cache["molecule_ids"]] != [str(r) for r in rows_valid]:
        raise ValueError("Validation420 cache row identity drifted")
    if [str(x) for x in train_cache["smiles"]] != [str(split["smiles"][r]) for r in rows_train]:
        raise ValueError("Train3360 cache SMILES identity drifted")
    if [str(x) for x in valid_cache["smiles"]] != [str(split["smiles"][r]) for r in rows_valid]:
        raise ValueError("Validation420 cache SMILES identity drifted")
    spec = prep_feature_spec.lipo_v1_spec()
    train_norm = prep_data.fit_norm(train_cache, spec, source="train3360_structures")
    target_norm = fit_target_scaler(labels.y_train)
    return {
        "built": built,
        "spec": spec,
        "train_cache": train_cache,
        "valid_cache": valid_cache,
        "train_norm": train_norm,
        "target_norm": target_norm,
        "train_norm_fingerprint": prep_data.norm_fingerprint(train_norm),
        "rows_train": rows_train,
        "rows_valid": rows_valid,
    }


__all__ = [
    "read_smiles_only",
    "compute_split",
    "verify_split",
    "freeze_split",
    "load_split",
    "build_upstream_structures",
    "save_structure_artifact",
    "load_structure_arrays",
    "load_structure_data",
    "structure_summary",
    "load_raw_graph_artifact",
    "build_lcrc_cache",
    "select_cache",
    "fit_target_scaler",
    "build_lcrc_views",
]
