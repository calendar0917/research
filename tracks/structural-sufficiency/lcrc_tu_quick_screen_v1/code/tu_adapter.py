"""TU dataset adapter: official TUDataset loading, cleaning, node types, spec.

Only the official PyG ``TUDataset`` interface is used
(``cleaned=False``, ``use_node_attr=False``, ``use_edge_attr=False``).  The
adapter

* drops self-loops and merges duplicated directed edge records into one
  physical undirected edge per pair, recording every change count;
* keeps isolated nodes and disconnected graphs;
* builds the discrete node-type vocabulary from the **Train split only**
  (label datasets) or the fixed degree buckets (IMDB);
* constructs a TU :class:`feature_spec.FeatureSpec` (``K + 29`` node features,
  2 edge features, 9 scalars, 28 global fields, ``B = 2`` edge-type slots);
* reuses only the label-free graph algorithms of ``molecular_graph.py``.

No RDKit, no pretrained weights, no external descriptors, no chemical labels.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import tu_common as C
import molecular_graph as MG
from feature_spec import (
    FEATURE_DTYPE,
    SCHEMA_VERSION,
    STRUCTURAL_CODE_DTYPE,
    STRUCTURAL_INDEX_DTYPE,
    FeatureField,
    FeatureSpec,
    TensorSpec,
)

# ---------------------------------------------------------------------------
# dataset location / raw manifest
# ---------------------------------------------------------------------------

_RAW_CORE = ("{name}_A.txt", "{name}_graph_indicator.txt", "{name}_graph_labels.txt")
_RAW_OPTIONAL = (
    "{name}_node_labels.txt",
    "{name}_node_attributes.txt",
    "{name}_edge_labels.txt",
    "{name}_edge_attributes.txt",
    "{name}_graph_attributes.txt",
    "README.txt",
)


def _raw_dir_candidates(name: str) -> list[tuple[Any, Any]]:
    return [
        (C.TU_ROOT / name, C.TU_ROOT / name / name / "raw"),  # PyG >= 2.4 layout
        (C.TU_ROOT, C.TU_ROOT / name / "raw"),               # PyG legacy layout
    ]


def resolve_dataset_root(name: str):
    """Return the ``root`` argument for ``TUDataset`` and the raw directory."""
    for root, raw_dir in _raw_dir_candidates(name):
        if all((raw_dir / f.format(name=name)).is_file() for f in _RAW_CORE):
            return root, raw_dir
    raise FileNotFoundError(
        f"no complete official raw file set for {name} under {C.TU_ROOT}; "
        f"expected {[(str(r), str(d)) for r, d in _raw_dir_candidates(name)]}"
    )


def raw_manifest(name: str) -> dict[str, Any]:
    root, raw_dir = resolve_dataset_root(name)
    files = []
    for pattern in _RAW_CORE + _RAW_OPTIONAL:
        p = raw_dir / pattern.format(name=name)
        if p.is_file():
            files.append(
                {
                    "file": p.name,
                    "bytes": int(p.stat().st_size),
                    "sha256": C.sha256_file(p),
                }
            )
    return {
        "dataset": name,
        "root": C.track_rel(root),
        "raw_dir": C.track_rel(raw_dir),
        "source": f"{C.TU_ARCHIVE_ROOT}/{name}.zip",
        "source_docs": C.TU_DATASETS_DOCS,
        "loader_docs": C.TU_PYG_DOCS,
        "loader": "torch_geometric.datasets.TUDataset(cleaned=False, use_node_attr=False, use_edge_attr=False)",
        "files": files,
    }


# ---------------------------------------------------------------------------
# loading + edge cleaning
# ---------------------------------------------------------------------------


def clean_edge_index(edge_index: np.ndarray, n_atoms: int) -> dict[str, Any]:
    """Drop self-loops and merge duplicated directed records.

    Returns the canonical physical edge list (``u < v``, sorted) plus explicit
    change counts.  No graph is dropped.
    """
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index shape {ei.shape}; expected [2, M]")
    u_all = ei[0]
    v_all = ei[1]
    if u_all.size and (u_all.min() < 0 or v_all.min() < 0 or u_all.max() >= n_atoms or v_all.max() >= n_atoms):
        raise ValueError("edge endpoint outside node range")
    n_directed = int(u_all.size)
    self_loop_mask = u_all == v_all
    n_self_loops = int(self_loop_mask.sum())
    u = u_all[~self_loop_mask]
    v = v_all[~self_loop_mask]
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(u.tolist(), v.tolist()):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        counts[key] = counts.get(key, 0) + 1
    keys = sorted(counts)
    pairs = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
    n_physical = int(pairs.shape[0])
    n_merged = n_directed - n_self_loops - n_physical
    return {
        "pairs": pairs,
        "n_directed_records": n_directed,
        "n_self_loops_dropped": n_self_loops,
        "n_physical_edges": n_physical,
        "n_duplicate_directed_records_merged": int(n_merged),
        "n_pairs_with_multiplicity_gt2": int(sum(1 for c in counts.values() if c > 2)),
        "n_pairs_with_odd_multiplicity": int(sum(1 for c in counts.values() if c % 2 == 1)),
    }


def load_dataset(name: str) -> dict[str, Any]:
    """Load one official TUDataset split-agnostically and canonicalise edges."""
    from torch_geometric.datasets import TUDataset

    meta = C.DATASETS[name]
    root, raw_dir = resolve_dataset_root(name)
    ds = TUDataset(
        root=str(root),
        name=name,
        cleaned=False,
        use_node_attr=False,
        use_edge_attr=False,
    )
    records: list[dict[str, Any]] = []
    clean_totals = {
        "n_directed_records": 0,
        "n_self_loops_dropped": 0,
        "n_physical_edges": 0,
        "n_duplicate_directed_records_merged": 0,
        "n_pairs_with_multiplicity_gt2": 0,
        "n_pairs_with_odd_multiplicity": 0,
    }
    for gid in range(len(ds)):
        data = ds[gid]
        n_atoms = int(data.num_nodes)
        if n_atoms < 1:
            raise ValueError(f"{name}[{gid}] has no nodes; this round must not drop graphs")
        cleaned = clean_edge_index(data.edge_index.numpy(), n_atoms)
        for k in clean_totals:
            clean_totals[k] += int(cleaned[k])
        y = int(data.y.reshape(-1)[0].item())
        if not (0 <= y < meta["n_classes"]):
            raise ValueError(f"{name}[{gid}] label {y} outside 0..{meta['n_classes'] - 1}")
        if meta["node_type"] == "label":
            if data.x is None or data.x.numel() == 0:
                raise ValueError(f"{name}[{gid}] has no discrete node labels")
            x = data.x.detach().cpu().numpy()
            labels = np.argmax(x, axis=1).astype(np.int64)
            if not np.allclose(x.sum(axis=1), 1.0):
                raise ValueError(f"{name}[{gid}] node labels are not one-hot")
        else:
            if data.x is not None and data.x.numel() > 0:
                raise ValueError(f"{name}[{gid}] unexpectedly carries node attributes")
            labels = None
        records.append(
            {
                "gid": gid,
                "y": y,
                "n_atoms": n_atoms,
                "pairs": cleaned["pairs"],
                "node_labels": labels,
                "n_physical_edges": cleaned["n_physical_edges"],
                "clean": {k: cleaned[k] for k in clean_totals},
            }
        )
    labels_all = np.asarray([r["y"] for r in records], dtype=np.int64)
    counts = np.bincount(labels_all, minlength=int(meta["n_classes"]))
    stats = {
        "dataset": name,
        "n_graphs_loaded": len(records),
        "n_graphs_expected": int(meta["n_graphs"]),
        "n_classes_observed": int(len(np.unique(labels_all))),
        "n_classes_expected": int(meta["n_classes"]),
        "class_counts": [int(x) for x in counts],
        "node_type_mode": meta["node_type"],
        "n_nodes_total": int(sum(r["n_atoms"] for r in records)),
        "min_nodes": int(min(r["n_atoms"] for r in records)),
        "max_nodes": int(max(r["n_atoms"] for r in records)),
        "n_graphs_with_isolated_nodes": int(sum(1 for r in records if r["n_physical_edges"] == 0 or r["n_atoms"] > np.unique(r["pairs"]).size)),
        "clean_totals": clean_totals,
        "labels_dropped": False,
    }
    if stats["n_graphs_loaded"] != stats["n_graphs_expected"]:
        raise ValueError(f"{name}: loaded {stats['n_graphs_loaded']} graphs != expected {stats['n_graphs_expected']}")
    if stats["n_classes_observed"] != stats["n_classes_expected"]:
        raise ValueError(f"{name}: observed {stats['n_classes_observed']} classes != expected {stats['n_classes_expected']}")
    return {"name": name, "records": records, "labels": labels_all, "stats": stats, "raw_dir": raw_dir}


# ---------------------------------------------------------------------------
# fixed stratified 8:1:1 split
# ---------------------------------------------------------------------------


def build_split(labels: np.ndarray, n_train: int, n_valid: int, n_test: int, seed: int = C.SPLIT_SEED) -> dict[str, np.ndarray]:
    """The single frozen split: two stratified ``train_test_split`` calls."""
    from sklearn.model_selection import train_test_split

    y = np.asarray(labels, dtype=np.int64)
    if n_train + n_valid + n_test != y.size:
        raise ValueError(f"split sizes {n_train}+{n_valid}+{n_test} != {y.size}")
    idx_all = np.arange(y.size, dtype=np.int64)
    train_idx, rest_idx = train_test_split(
        idx_all, train_size=int(n_train), stratify=y, random_state=int(seed)
    )
    valid_idx, test_idx = train_test_split(
        rest_idx,
        train_size=int(n_valid),
        stratify=y[rest_idx],
        random_state=int(seed),
    )
    out = {
        "train": np.sort(np.asarray(train_idx, dtype=np.int64)),
        "valid": np.sort(np.asarray(valid_idx, dtype=np.int64)),
        "test": np.sort(np.asarray(test_idx, dtype=np.int64)),
    }
    union = np.concatenate([out["train"], out["valid"], out["test"]])
    if np.unique(union).size != y.size or not np.array_equal(np.sort(union), idx_all):
        raise ValueError("split is not a partition of the dataset")
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if np.intersect1d(out[a], out[b]).size:
            raise ValueError(f"split overlap between {a} and {b}")
    for split in out:
        if out[split].size != {"train": n_train, "valid": n_valid, "test": n_test}[split]:
            raise ValueError(f"{split} has {out[split].size} rows")
    return out


def split_record(dataset: str, split: Mapping[str, np.ndarray], labels: np.ndarray) -> dict[str, Any]:
    meta = C.DATASETS[dataset]
    payload: dict[str, Any] = {
        "dataset": dataset,
        "split_seed": C.SPLIT_SEED,
        "sizes_expected": [int(meta["n_train"]), int(meta["n_valid"]), int(meta["n_test"])],
        "procedure": (
            "sklearn.model_selection.train_test_split(train_size=n_train, stratify=y, random_state=0) "
            "then train_test_split(rest, train_size=n_valid, stratify=y_rest, random_state=0)"
        ),
        "n_classes": int(meta["n_classes"]),
    }
    for name in ("train", "valid", "test"):
        rows = np.asarray(split[name], dtype=np.int64)
        cls = np.bincount(labels[rows], minlength=int(meta["n_classes"]))
        payload[name] = {
            "n": int(rows.size),
            "class_counts": [int(x) for x in cls],
            "index_sha256": C.index_hash(rows),
            "rows_sha256": C.sha256_array(np.ascontiguousarray(rows.astype(np.int32))),
            "first_rows": [int(x) for x in rows[:8]],
            "last_rows": [int(x) for x in rows[-8:]],
        }
    payload["partition_sha256"] = C.sha256_bytes(
        b"".join(np.asarray(split[n], dtype=np.int64).tobytes() for n in ("train", "valid", "test"))
    )
    return payload


# ---------------------------------------------------------------------------
# node types + feature spec
# ---------------------------------------------------------------------------


def node_type_table(mode: str, train_labels: Sequence[int] | None) -> dict[str, Any]:
    """Build the discrete node-type vocabulary.

    ``label`` mode: sorted raw node labels observed in **Train**, then ``UNK``.
    ``degree`` mode: buckets 0..15, ``>= 16`` and a reserved ``UNK``.
    """
    if mode == "label":
        if train_labels is None:
            raise ValueError("label mode needs Train node labels")
        vocab = sorted({int(x) for x in train_labels})
        names = [f"NODE_LABEL_{v}" for v in vocab] + ["UNK"]
        mapping = {v: i for i, v in enumerate(vocab)}
        return {
            "mode": "label",
            "names": tuple(names),
            "train_vocab": vocab,
            "map": mapping,
            "unk_index": len(vocab),
            "encode": "raw discrete node label -> Train-vocabulary index; unseen -> UNK",
        }
    if mode == "degree":
        names = tuple([f"DEG_{i}" for i in range(C.DEGREE_BUCKET_MAX)] + ["DEG_16PLUS", "UNK"])
        return {
            "mode": "degree",
            "names": names,
            "train_vocab": list(range(C.DEGREE_BUCKET_MAX + 1)),
            "map": {i: i for i in range(C.DEGREE_BUCKET_MAX + 1)},
            "unk_index": C.DEGREE_BUCKET_MAX + 1,
            "encode": "atom_type = min(degree, 16); UNK index reserved and never produced",
        }
    raise KeyError(mode)


def _output_tensors(atom_dim: int, bond_dim: int, scalar_dim: int, global_dim: int) -> tuple[TensorSpec, ...]:
    code = STRUCTURAL_CODE_DTYPE
    idx = STRUCTURAL_INDEX_DTYPE
    f32 = FEATURE_DTYPE
    return (
        TensorSpec("atom_feat", f"[N, {atom_dim}]", f32, "per-node TU feature matrix"),
        TensorSpec("atom_type", "[N]", idx, "discrete node type id (0-based)"),
        TensorSpec("atom_scalar", f"[N, {scalar_dim}]", f32, "per-node topology scalars before Fit-only standardisation"),
        TensorSpec("bond_feat", f"[M, {bond_dim}]", f32, "per-physical-edge TU feature matrix"),
        TensorSpec("bond_type", "[M]", idx, "edge type id (1-based; always EDGE=1)"),
        TensorSpec("bond_index", "[M, 2]", idx, "physical edge endpoints, u < v, sorted"),
        TensorSpec("atom_ptr", "[B + 1]", idx, "molecule node offsets"),
        TensorSpec("bond_ptr", "[B + 1]", idx, "molecule edge offsets"),
        TensorSpec("nbr_ptr", "[N_total + 1]", idx, "per-node S1 code offsets"),
        TensorSpec("nbr_code", "[nnz_S1]", code, "joint (neighbour node type, edge type) code"),
        TensorSpec("path_ptr", "[N_total + 1]", idx, "per-root S2 record offsets"),
        TensorSpec("path_cat", "[nnz_S2]", code, "joint (middle type, edge1, edge2) code"),
        TensorSpec("path_end63", "[nnz_S2]", code, "joint (end node type, edge2) code"),
        TensorSpec("path_end", "[nnz_S2]", idx, "absolute end-node index in the batch"),
        TensorSpec("path_closure", "[nnz_S2]", "int8", "1 iff the end node is adjacent to the root"),
        TensorSpec("global", f"[B, {global_dim}]", f32, "graph-level TU topology statistics before Fit-only standardisation"),
    )


def build_tu_spec(dataset: str, node_types: Mapping[str, Any]) -> FeatureSpec:
    """Construct the TU ``FeatureSpec`` for one dataset (width ``K + 29``)."""
    meta = C.DATASETS[dataset]
    names = tuple(str(x) for x in node_types["names"])
    k = len(names)
    atom_fields = (
        FeatureField("node_type", k, FEATURE_DTYPE, "one-hot discrete node type (Train vocab or degree bucket + UNK)", "TU node label / degree bucket"),
        FeatureField("degree_bucket", C.N_DEGREE_BUCKETS, FEATURE_DTYPE, "degree one-hot 0..15,>=16 from the cleaned physical graph", "TU graph"),
        FeatureField("in_cycle", 1, FEATURE_DTYPE, "node lies on at least one cycle", "biconnected-block analysis"),
        FeatureField("min_cycle_size", 7, FEATURE_DTYPE, "one-hot acyclic,3,4,5,6,7,>=8 minimum cycle length", "smallest cycle through the node"),
        FeatureField("log1p_cycle_degree", 1, FEATURE_DTYPE, "log1p(number of incident cycle edges)", "biconnected-block analysis"),
        FeatureField("cycle_degree_fraction", 1, FEATURE_DTYPE, "cycle_degree / max(degree, 1)", "biconnected-block analysis"),
        FeatureField("is_terminal", 1, FEATURE_DTYPE, "degree == 1", "TU graph"),
        FeatureField("cycle_degree_ge3", 1, FEATURE_DTYPE, "cycle_degree >= 3", "biconnected-block analysis"),
    )
    bond_fields = (
        FeatureField("constant_one", 1, FEATURE_DTYPE, "always 1 (the single physical edge-type slot)", "TU adapter"),
        FeatureField("is_cycle_edge", 1, FEATURE_DTYPE, "edge lies on at least one cycle", "biconnected-block analysis"),
    )
    scalar_fields = tuple(
        FeatureField(name, 1, FEATURE_DTYPE, desc, "TU topology")
        for name, desc in C.SCALAR_FIELDS
    )
    global_fields = tuple(
        FeatureField(name, 1, FEATURE_DTYPE, desc, "TU graph statistics")
        for name, desc in C.GLOBAL_FIELDS
    )
    atom_type_metadata = tuple(
        {
            "index": i,
            "name": name,
            "is_unknown": i == int(node_types["unk_index"]),
            "source": "TU discrete node label" if node_types["mode"] == "label" else "node degree bucket",
        }
        for i, name in enumerate(names)
    )
    spec = FeatureSpec(
        schema_version=SCHEMA_VERSION,
        name=f"tu_v1_{dataset.lower().replace('-', '_')}",
        description=(
            f"TU quick-screen schema for {dataset}: label-free topology only.  Node features are the "
            "discrete node type plus generic graph-cycle/degree descriptors; no chemical fields."
        ),
        atom_type_names=names,
        atom_type_metadata=atom_type_metadata,
        atom_unk_index=int(node_types["unk_index"]),
        atom_unknown_policy=(
            "map_to_UNK: " + str(node_types["encode"])
        ),
        atom_type_encoding=str(node_types["encode"]),
        bond_type_names=C.BOND_TYPE_NAMES,
        bond_type_metadata=(
            {"id": 1, "name": "EDGE", "is_unknown": False, "source": "TU raw A.txt (no native edge labels)"},
            {"id": 2, "name": "UNK", "is_unknown": True, "source": "reserved; never produced by the TU adapter"},
        ),
        bond_unk_id=C.BOND_UNK_ID,
        bond_unknown_policy="all physical TU edges use EDGE=1; UNK=2 is reserved and never produced",
        bond_type_encoding="bond_type = 1 (EDGE) for every physical edge; one-based",
        atom_feature_fields=atom_fields,
        bond_feature_fields=bond_fields,
        atom_scalar_fields=scalar_fields,
        global_feature_fields=global_fields,
        nbr_code_formula="node_type[neighbour] * 2 + (edge_type - 1)",
        path_code_formula="node_type[middle] * 4 + (bt1 - 1) * 2 + (bt2 - 1)",
        path_end_code_formula="node_type[end] * 2 + (bt2 - 1)",
        structural_code_dtype=STRUCTURAL_CODE_DTYPE,
        structural_index_dtype=STRUCTURAL_INDEX_DTYPE,
        feature_dtype=FEATURE_DTYPE,
        supports_disconnected=True,
        disconnected_policy=(
            "supported with finite statistics: unreachable pairs (shortest-path sentinel -1) are excluded from "
            "diameter/mean distance, reported via frac_unreachable_pairs, and never counted as a finite distance >= 3"
        ),
        atom_feature_source="TU adapter: discrete node type + degree/cycle descriptors of the cleaned physical graph",
        bond_feature_source="TU adapter: constant 1 + is_cycle_edge",
        global_feature_source="TU adapter: label-free graph statistics with explicit disconnected handling",
        source_provenance={
            "dataset": dataset,
            "loader": "torch_geometric.datasets.TUDataset",
            "cleaned": False,
            "use_node_attr": False,
            "use_edge_attr": False,
            "edge_cleaning": "self-loops dropped; mirrored/duplicated directed records merged into one physical edge",
            "node_type_mode": node_types["mode"],
        },
        output_tensors=_output_tensors(k + 29, 2, C.SCALAR_DIM, C.GLOBAL_DIM),
    )
    spec.validate()
    if spec.atom_feat_dim != k + 29:
        raise RuntimeError(f"atom feature width {spec.atom_feat_dim} != K+29 = {k + 29}")
    if spec.global_dim != 28 or spec.n_scalars != 9:
        raise RuntimeError("TU spec must have 28 global fields and 9 scalars")
    return spec


# ---------------------------------------------------------------------------
# node/edge features + graph construction
# ---------------------------------------------------------------------------


def node_type_ids(record: Mapping[str, Any], node_types: Mapping[str, Any], degree: np.ndarray) -> np.ndarray:
    if node_types["mode"] == "label":
        raw = np.asarray(record["node_labels"], dtype=np.int64)
        mapping = node_types["map"]
        unk = int(node_types["unk_index"])
        return np.asarray([mapping.get(int(v), unk) for v in raw.tolist()], dtype=np.int64)
    bucket = np.minimum(np.asarray(degree, dtype=np.int64), C.DEGREE_BUCKET_MAX)
    return bucket.astype(np.int64)


def _cycle_size_slot(min_cycle_size: np.ndarray) -> np.ndarray:
    m = np.asarray(min_cycle_size, dtype=np.int64)
    slot = np.zeros(m.shape, dtype=np.int64)
    non_zero = m > 0
    slot[non_zero] = np.where(m[non_zero] >= 8, 6, m[non_zero] - 2)
    return slot


def build_atom_features(
    spec: FeatureSpec,
    node_types: np.ndarray,
    analysis: MG.GraphAnalysis,
) -> np.ndarray:
    n = int(analysis.n_atoms)
    k = spec.n_atom_types
    out = np.zeros((n, k + 29), dtype=np.float32)
    out[np.arange(n), np.asarray(node_types, dtype=np.int64)] = 1.0
    bucket = np.minimum(np.asarray(analysis.degree, dtype=np.int64), C.DEGREE_BUCKET_MAX)
    out[np.arange(n), k + bucket] = 1.0
    out[:, k + C.N_DEGREE_BUCKETS] = analysis.ring_atom.astype(np.float32)
    slot = _cycle_size_slot(analysis.min_ring_size)
    out[np.arange(n), k + C.N_DEGREE_BUCKETS + 1 + slot] = 1.0
    cycle_degree = np.asarray(analysis.ring_bond_count, dtype=np.float64)
    degree = np.asarray(analysis.degree, dtype=np.float64)
    base = k + C.N_DEGREE_BUCKETS + 1 + 7
    out[:, base + 0] = np.log1p(cycle_degree).astype(np.float32)
    out[:, base + 1] = (cycle_degree / np.maximum(degree, 1.0)).astype(np.float32)
    out[:, base + 2] = (degree == 1).astype(np.float32)
    out[:, base + 3] = (cycle_degree >= 3).astype(np.float32)
    if out.shape[1] != spec.atom_feat_dim:
        raise RuntimeError(f"atom feature width {out.shape[1]} != {spec.atom_feat_dim}")
    return out


def build_bond_features(analysis: MG.GraphAnalysis) -> np.ndarray:
    m = int(analysis.n_bonds)
    out = np.zeros((m, 2), dtype=np.float32)
    if m:
        out[:, 0] = 1.0
        out[:, 1] = analysis.is_ring_bond.astype(np.float32)
    return out


def build_atom_scalars(
    analysis: MG.GraphAnalysis,
    s2_records: Sequence[Sequence[Mapping[str, int]]],
) -> np.ndarray:
    n = int(analysis.n_atoms)
    path_counts = np.asarray([len(r) for r in s2_records], dtype=np.float64)
    closure_counts = np.asarray([sum(int(x["closure"]) for x in r) for r in s2_records], dtype=np.float64)
    closure_ratio = closure_counts / np.maximum(path_counts, 1.0)
    n_d2 = np.sum(np.asarray(analysis.dist) == 2, axis=1).astype(np.float64)
    degree = np.asarray(analysis.degree, dtype=np.float64)
    cycle_degree = np.asarray(analysis.ring_bond_count, dtype=np.float64)
    min_cycle = np.asarray(analysis.min_ring_size, dtype=np.float64)
    cols = [
        np.log1p(n_d2),
        np.log1p(path_counts),
        closure_ratio,
        np.log1p(degree),
        analysis.ring_atom.astype(np.float64),
        np.log1p(min_cycle),
        np.log1p(cycle_degree),
        (degree == 1).astype(np.float64),
        (cycle_degree >= 3).astype(np.float64),
    ]
    out = np.stack(cols, axis=1).astype(np.float32)
    if out.shape != (n, C.SCALAR_DIM):
        raise RuntimeError(f"scalar shape {out.shape} != {(n, C.SCALAR_DIM)}")
    return out


def build_global_features(graph: MG.MolecularGraph, analysis: MG.GraphAnalysis) -> np.ndarray:
    """The 28 fixed TU global topology fields (chemistry fields removed)."""
    n = int(analysis.n_atoms)
    m = int(analysis.n_bonds)
    deg = np.asarray(analysis.degree, dtype=np.float64)
    dist = np.asarray(analysis.dist)
    if n > 1:
        pair_dist = dist[np.triu_indices(n, 1)]
    else:
        pair_dist = np.zeros(0, dtype=np.int64)
    finite = pair_dist >= 0
    finite_dist = pair_dist[finite]
    n_pairs = int(pair_dist.size)
    diameter = float(finite_dist.max()) if finite_dist.size else 0.0
    mean_finite = float(finite_dist.mean()) if finite_dist.size else 0.0
    frac_unreachable = float((n_pairs - finite_dist.size) / n_pairs) if n_pairs else 0.0
    frac_finite_ge3 = float((finite_dist >= 3).sum() / n_pairs) if n_pairs else 0.0

    ring_atom = analysis.ring_atom
    ring_bond = analysis.is_ring_bond
    n_ring_atoms = int(ring_atom.sum())
    n_junctions = int(analysis.n_junctions)
    ring_size_counts = np.zeros(6, dtype=np.float64)
    for atom in np.flatnonzero(ring_atom):
        size = int(analysis.min_ring_size[atom])
        if size <= 0:
            continue
        slot = min(max(size - 3, 0), 5)
        ring_size_counts[slot] += 1.0
    if n_ring_atoms:
        ring_size_counts /= float(n_ring_atoms)

    junction_deg2 = 0
    junction_deg3plus = 0
    junction_ring = 0
    for v in range(n):
        d = int(deg[v])
        if d < 2:
            continue
        count = d * (d - 1) // 2
        if d == 2:
            junction_deg2 += count
        else:
            junction_deg3plus += count
        if bool(ring_atom[v]):
            junction_ring += count

    values = {
        "log1p_n_atoms": np.log1p(n),
        "log1p_n_bonds": np.log1p(m),
        "log1p_n_junctions": np.log1p(n_junctions),
        "cyclomatic_number": float(m - n + analysis.n_components),
        "n_components": float(analysis.n_components),
        "ring_atom_fraction": float(ring_atom.mean()) if n else 0.0,
        "ring_bond_fraction": float(ring_bond.mean()) if m else 0.0,
        "n_ring_systems": float(analysis.n_ring_systems),
        "diameter_max_finite": diameter,
        "mean_finite_pair_distance": mean_finite,
        "frac_unreachable_pairs": frac_unreachable,
        "frac_finite_pairs_ge3": frac_finite_ge3,
        "bonds_per_atom": float(m) / max(n, 1),
        "log1p_n_ring_atoms": np.log1p(n_ring_atoms),
        "max_degree": float(deg.max()) if n else 0.0,
        "terminal_atom_fraction": float((deg == 1).mean()) if n else 0.0,
        "junction_per_bond": float(n_junctions) / max(m, 1),
        "frac_junction_centers_ring": (float(junction_ring) / n_junctions) if n_junctions else 0.0,
        "frac_deg2_junctions": (float(junction_deg2) / n_junctions) if n_junctions else 0.0,
        "frac_deg3plus_junctions": (float(junction_deg3plus) / n_junctions) if n_junctions else 0.0,
        "ring_size3_fraction": float(ring_size_counts[0]),
        "ring_size4_fraction": float(ring_size_counts[1]),
        "ring_size5_fraction": float(ring_size_counts[2]),
        "ring_size6_fraction": float(ring_size_counts[3]),
        "ring_size7_fraction": float(ring_size_counts[4]),
        "ring_size8plus_fraction": float(ring_size_counts[5]),
        "fused_ring_atom_fraction": float((analysis.ring_bond_count >= 3).mean()) if n else 0.0,
        "distinct_atom_types": float(len(set(int(x) for x in graph.atom_type.tolist()))),
    }
    out = np.asarray([values[name] for name, _desc in C.GLOBAL_FIELDS], dtype=np.float32)
    if out.shape != (C.GLOBAL_DIM,):
        raise RuntimeError(f"global shape {out.shape} != {(C.GLOBAL_DIM,)}")
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite TU global feature")
    return out


def build_tu_graph(
    record: Mapping[str, Any],
    spec: FeatureSpec,
    node_types: Mapping[str, Any],
    *,
    molecule_id: str,
) -> tuple[MG.MolecularGraph, MG.GraphAnalysis]:
    """Build the canonical TU graph plus the generic graph analysis."""
    n = int(record["n_atoms"])
    pairs = np.asarray(record["pairs"], dtype=np.int64).reshape(-1, 2)
    u = pairs[:, 0]
    v = pairs[:, 1]
    m = int(pairs.shape[0])
    degree = MG.degrees(n, u, v)
    tids = node_type_ids(record, node_types, degree)
    if np.any(tids < 0) or np.any(tids >= spec.n_atom_types):
        raise ValueError(f"{molecule_id}: node type outside vocabulary")
    bond_type = np.full(m, C.EDGE_TYPE, dtype=np.int64)
    placeholder = MG.build_molecular_graph(
        atom_type=tids,
        bond_index=pairs,
        bond_type=bond_type,
        atom_feat=np.zeros((n, spec.atom_feat_dim), dtype=np.float32),
        bond_feat=np.zeros((m, spec.bond_feat_dim), dtype=np.float32),
        molecule_id=molecule_id,
    )
    analysis = MG.analyse_graph(placeholder)
    atom_feat = build_atom_features(spec, tids, analysis)
    bond_feat = build_bond_features(analysis)
    graph = replace(placeholder, atom_feat=atom_feat, bond_feat=bond_feat)
    return graph, analysis


__all__ = [
    "resolve_dataset_root",
    "raw_manifest",
    "clean_edge_index",
    "load_dataset",
    "build_split",
    "split_record",
    "node_type_table",
    "build_tu_spec",
    "build_tu_graph",
    "node_type_ids",
    "build_atom_features",
    "build_bond_features",
    "build_atom_scalars",
    "build_global_features",
]
