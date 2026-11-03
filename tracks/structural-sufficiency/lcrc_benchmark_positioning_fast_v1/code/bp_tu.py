"""Official TU loading, label-free cache construction and 10-fold splits."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import bp_common as C
import bp_data as D
import bp_graph as BG
import bp_spec as S
import lcrc_data as PREP_DATA
import molecular_graph as MG
from feature_spec import save_spec


class InputAdapterBlocked(RuntimeError):
    """Raised when an original TU attribute cannot be represented exactly."""


class LargeGraphBlocked(RuntimeError):
    """The exact S2 cache exceeds the pre-declared eager-cache budget."""


RAW_CORE = ("{name}_A.txt", "{name}_graph_indicator.txt", "{name}_graph_labels.txt")
RAW_OPTIONAL = (
    "{name}_node_labels.txt",
    "{name}_node_attributes.txt",
    "{name}_edge_labels.txt",
    "{name}_edge_attributes.txt",
    "README.txt",
)


def raw_dir(name: str) -> Path:
    return C.TU_ROOT / name / name / "raw"


def raw_manifest(name: str) -> dict[str, Any]:
    d = raw_dir(name)
    files = []
    for pattern in RAW_CORE + RAW_OPTIONAL:
        p = d / pattern.format(name=name)
        if p.is_file():
            files.append({"file": p.name, "bytes": int(p.stat().st_size), "sha256": C.sha256_file(p)})
    return {
        "dataset": name,
        "raw_dir": C.track_rel(d),
        "source": f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip",
        "source_docs": "https://chrsmrrs.github.io/datasets/docs/datasets/",
        "loader": "torch_geometric.datasets.TUDataset(cleaned=False, use_node_attr=True, use_edge_attr=True)",
        "files": files,
    }


def _clean_edges(
    edge_index: np.ndarray,
    edge_label: np.ndarray | None,
    edge_attr: np.ndarray | None,
    name: str,
) -> dict[str, Any]:
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"{name}: edge_index shape {ei.shape}")
    m_dir = int(ei.shape[1])
    if edge_label is not None and len(edge_label) != m_dir:
        raise ValueError(f"{name}: edge label length")
    if edge_attr is not None and len(edge_attr) != m_dir:
        raise ValueError(f"{name}: edge attribute length")
    seen: dict[tuple[int, int], tuple[int | None, tuple[float, ...] | None]] = {}
    self_loops = 0
    duplicates = 0
    conflicts = 0
    for k in range(m_dir):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u == v:
            self_loops += 1
            continue
        key = (min(u, v), max(u, v))
        lab = None if edge_label is None else int(edge_label[k])
        att = None if edge_attr is None else tuple(float(x) for x in np.asarray(edge_attr[k]).reshape(-1).tolist())
        if key in seen:
            duplicates += 1
            if seen[key] != (lab, att):
                conflicts += 1
            continue
        seen[key] = (lab, att)
    if conflicts:
        raise InputAdapterBlocked(
            f"{name}: mirrored/duplicate directed records carry different edge labels/attributes; refusing to merge silently"
        )
    keys = sorted(seen)
    pairs = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
    labels = None if edge_label is None else np.asarray([seen[k][0] for k in keys], dtype=np.int64)
    attrs = None if edge_attr is None else np.asarray([seen[k][1] for k in keys], dtype=np.float32)
    return {
        "pairs": pairs,
        "edge_labels": labels,
        "edge_attrs": attrs,
        "n_directed_records": m_dir,
        "n_self_loops_dropped": int(self_loops),
        "n_duplicate_directed_records_merged": int(duplicates),
        "n_physical_edges": int(pairs.shape[0]),
    }


def scan_dataset(name: str) -> dict[str, Any]:
    """Load one official TUDataset with all raw attributes retained."""
    from torch_geometric.datasets import TUDataset

    meta = C.TU_DATASETS[name]
    root = C.TU_ROOT / name
    if not raw_dir(name).is_dir():
        raise FileNotFoundError(f"{name}: raw directory {raw_dir(name)} is missing")
    ds = TUDataset(root=str(root), name=name, cleaned=False, use_node_attr=True, use_edge_attr=True)
    sizes = ds.sizes
    n_attr = int(sizes["num_node_attributes"])
    n_lab = int(sizes["num_node_labels"])
    e_attr = int(sizes["num_edge_attributes"])
    e_lab = int(sizes["num_edge_labels"])
    if e_attr:
        raise InputAdapterBlocked(
            f"{name}: raw continuous edge attributes (dim {e_attr}) are present; no exact adapter was frozen for this round"
        )
    records: list[dict[str, Any]] = []
    clean_totals = {
        "n_directed_records": 0,
        "n_self_loops_dropped": 0,
        "n_duplicate_directed_records_merged": 0,
        "n_physical_edges": 0,
    }
    for gid in range(len(ds)):
        data = ds[gid]
        n = int(data.num_nodes)
        if n < 1:
            raise InputAdapterBlocked(f"{name}[{gid}]: empty graph")
        x = None if data.x is None else np.asarray(data.x.detach().cpu().numpy(), dtype=np.float32)
        node_labels = None
        node_attrs = None
        if x is not None:
            if x.shape != (n, n_attr + n_lab):
                raise InputAdapterBlocked(f"{name}[{gid}]: unexpected x shape {x.shape}")
            if n_attr:
                node_attrs = np.asarray(x[:, :n_attr], dtype=np.float32)
                if not np.isfinite(node_attrs).all():
                    raise InputAdapterBlocked(f"{name}[{gid}]: non-finite continuous node attributes")
            if n_lab:
                lab_block = np.asarray(x[:, n_attr : n_attr + n_lab], dtype=np.float32)
                if not np.allclose(lab_block.sum(axis=1), 1.0, atol=1e-5):
                    raise InputAdapterBlocked(f"{name}[{gid}]: node labels are not one-hot")
                node_labels = np.argmax(lab_block, axis=1).astype(np.int64)
        edge_label = None
        edge_attr = None
        ea = None if data.edge_attr is None else np.asarray(data.edge_attr.detach().cpu().numpy(), dtype=np.float32)
        if ea is not None:
            if ea.shape[1] != e_attr + e_lab:
                raise InputAdapterBlocked(f"{name}[{gid}]: unexpected edge_attr shape {ea.shape}")
            if e_attr:
                edge_attr = np.asarray(ea[:, :e_attr], dtype=np.float32)
            if e_lab:
                lab_block = np.asarray(ea[:, e_attr : e_attr + e_lab], dtype=np.float32)
                if not np.allclose(lab_block.sum(axis=1), 1.0, atol=1e-5):
                    raise InputAdapterBlocked(f"{name}[{gid}]: edge labels are not one-hot")
                edge_label = np.argmax(lab_block, axis=1).astype(np.int64)
        cleaned = _clean_edges(
            data.edge_index.detach().cpu().numpy(), edge_label, edge_attr, f"{name}[{gid}]"
        )
        for k in clean_totals:
            clean_totals[k] += int(cleaned[k])
        y = int(data.y.reshape(-1)[0].item())
        if not (0 <= y < int(meta["n_classes"])):
            raise InputAdapterBlocked(f"{name}[{gid}]: label {y} outside expected classes")
        records.append(
            {
                "gid": gid,
                "y": y,
                "n_atoms": n,
                "pairs": cleaned["pairs"],
                "node_labels": node_labels,
                "node_attrs": node_attrs,
                "edge_labels": cleaned["edge_labels"],
                "edge_attrs": cleaned["edge_attrs"],
                "clean": {k: int(cleaned[k]) for k in clean_totals},
            }
        )
    y_all = np.asarray([r["y"] for r in records], dtype=np.int64)
    if len(records) != int(meta["n_graphs"]):
        raise InputAdapterBlocked(f"{name}: loaded {len(records)} graphs != expected {meta['n_graphs']}")
    if len(np.unique(y_all)) != int(meta["n_classes"]):
        raise InputAdapterBlocked(f"{name}: observed {len(np.unique(y_all))} classes != {meta['n_classes']}")
    has_node_labels = all(r["node_labels"] is not None for r in records)
    has_node_attrs = all(r["node_attrs"] is not None for r in records)
    if any((r["node_labels"] is not None) != has_node_labels for r in records):
        raise InputAdapterBlocked(f"{name}: mixed presence of raw node labels")
    if any((r["node_attrs"] is not None) != has_node_attrs for r in records):
        raise InputAdapterBlocked(f"{name}: mixed presence of raw node attributes")
    has_edge_labels = all(r["edge_labels"] is not None for r in records)
    if any((r["edge_labels"] is not None) != has_edge_labels for r in records):
        raise InputAdapterBlocked(f"{name}: mixed presence of raw edge labels")
    node_attr_dim = int(records[0]["node_attrs"].shape[1]) if has_node_attrs else 0
    stats = {
        "dataset": name,
        "n_graphs": len(records),
        "n_classes": int(meta["n_classes"]),
        "class_counts": [int(x) for x in np.bincount(y_all, minlength=int(meta["n_classes"])).tolist()],
        "node_label_mode": "raw_node_labels" if has_node_labels else "degree_bucket",
        "has_continuous_node_attributes": bool(has_node_attrs),
        "continuous_node_attribute_dim": node_attr_dim,
        "has_edge_labels": bool(has_edge_labels),
        "has_continuous_edge_attributes": bool(e_attr),
        "n_nodes_total": int(sum(r["n_atoms"] for r in records)),
        "min_nodes": int(min(r["n_atoms"] for r in records)),
        "max_nodes": int(max(r["n_atoms"] for r in records)),
        "clean_totals": clean_totals,
        "graph_labels_used_for_schema": False,
    }
    return {
        "name": name,
        "records": records,
        "y": y_all,
        "stats": stats,
        "sizes": sizes,
        "n_node_attributes": n_attr,
        "n_node_labels": n_lab,
        "n_edge_attributes": e_attr,
        "n_edge_labels": e_lab,
    }


def _node_type_table(ds: Mapping[str, Any]) -> dict[str, Any]:
    if ds["stats"]["node_label_mode"] == "raw_node_labels":
        raw = sorted({int(v) for r in ds["records"] for v in np.asarray(r["node_labels"]).tolist()})
        mapping = {v: i for i, v in enumerate(raw)}
        names = [f"NODE_LABEL_{v}" for v in raw] + ["UNK"]
        return {
            "mode": "label",
            "raw_vocab": raw,
            "map": mapping,
            "names": names,
            "unk_index": len(raw),
            "encoding": "raw node label -> rank in the sorted dataset-wide vocabulary; unseen -> UNK",
        }
    names = [f"DEG_{i}" for i in range(S.DEGREE_BUCKET_MAX)] + ["DEG_16PLUS", "UNK"]
    return {
        "mode": "degree",
        "raw_vocab": list(range(S.DEGREE_BUCKET_MAX + 1)),
        "map": {i: i for i in range(S.DEGREE_BUCKET_MAX + 1)},
        "names": names,
        "unk_index": S.DEGREE_BUCKET_MAX + 1,
        "encoding": "atom_type = min(degree,16); UNK index reserved and never produced",
    }


def _bond_type_table(ds: Mapping[str, Any]) -> dict[str, Any]:
    if not ds["stats"]["has_edge_labels"]:
        return {
            "names": ["EDGE", "UNK"],
            "map": None,
            "unk_id": 2,
            "encoding": "all physical edges use EDGE=1; UNK=2 reserved and never produced",
        }
    raw = sorted({int(v) for r in ds["records"] for v in np.asarray(r["edge_labels"]).tolist()})
    mapping = {v: i + 1 for i, v in enumerate(raw)}
    names = [f"EDGE_LABEL_{v}" for v in raw] + ["UNK"]
    return {
        "names": names,
        "map": mapping,
        "unk_id": len(names),
        "encoding": "raw edge label -> rank+1 in the sorted dataset-wide vocabulary; one-based; UNK reserved",
    }


def _atom_type_ids(record: Mapping[str, Any], node_types: Mapping[str, Any], degree: np.ndarray) -> np.ndarray:
    if node_types["mode"] == "label":
        raw = np.asarray(record["node_labels"], dtype=np.int64)
        return np.asarray([node_types["map"].get(int(v), int(node_types["unk_index"])) for v in raw.tolist()], dtype=np.int64)
    return np.minimum(np.asarray(degree, dtype=np.int64), S.DEGREE_BUCKET_MAX).astype(np.int64)


def _build_molecule(
    record: Mapping[str, Any],
    spec,
    node_types: Mapping[str, Any],
    bond_types: Mapping[str, Any],
    dataset: str,
) -> dict[str, Any]:
    n = int(record["n_atoms"])
    pairs = np.asarray(record["pairs"], dtype=np.int64).reshape(-1, 2)
    u, v = pairs[:, 0], pairs[:, 1]
    m = int(pairs.shape[0])
    degree = MG.degrees(n, u, v)
    atom_type = _atom_type_ids(record, node_types, degree)
    if np.any(atom_type < 0) or np.any(atom_type >= spec.n_atom_types):
        raise InputAdapterBlocked(f"{dataset}#{record['gid']}: atom type outside vocabulary")
    if bond_types["map"] is None:
        bond_type = np.ones(m, dtype=np.int64)
    else:
        raw = np.asarray(record["edge_labels"], dtype=np.int64)
        bond_type = np.asarray([bond_types["map"][int(x)] for x in raw.tolist()], dtype=np.int64)
    if m and (np.any(bond_type < 1) or np.any(bond_type > spec.n_bond_types)):
        raise InputAdapterBlocked(f"{dataset}#{record['gid']}: bond type outside vocabulary")
    placeholder = MG.build_molecular_graph(
        atom_type=atom_type,
        bond_index=pairs,
        bond_type=bond_type if m else np.zeros(0, dtype=np.int64),
        atom_feat=np.zeros((n, spec.atom_feat_dim), dtype=np.float32),
        bond_feat=np.zeros((m, spec.bond_feat_dim), dtype=np.float32),
        molecule_id=f"{dataset}#{record['gid']}",
    )
    analysis = BG.analyse_graph_fast(placeholder)
    k = spec.n_atom_types
    onehot = np.zeros((n, k), dtype=np.float32)
    onehot[np.arange(n), atom_type] = 1.0
    topo = S.topology_atom_features(analysis)
    parts = [onehot, topo]
    cont_dim = int(spec.atom_feat_dim) - (k + S.TOPOLOGY_ATOM_DIM)
    if cont_dim:
        attrs = np.asarray(record["node_attrs"], dtype=np.float32)
        if attrs.shape != (n, cont_dim):
            raise InputAdapterBlocked(f"{dataset}#{record['gid']}: continuous attribute shape")
        parts.append(attrs)
    atom_feat = np.concatenate(parts, axis=1).astype(np.float32)
    bond_feat = np.concatenate(
        [np.ones((m, 1), dtype=np.float32), S.bond_topology_features(analysis)], axis=1
    ).astype(np.float32)
    s1_ptr, s1_code = BG.s1_arrays(atom_type, analysis.adj, spec.n_bond_types)
    s2 = BG.s2_arrays(atom_type, analysis.adj, spec.n_bond_types)
    per_root = np.diff(s2["path_ptr"]).astype(np.int64)
    closure_counts = np.zeros(n, dtype=np.int64)
    if int(s2["path_ptr"][-1]):
        closure_counts = np.bincount(
            np.repeat(np.arange(n, dtype=np.int64), per_root),
            weights=np.asarray(s2["path_closure"], dtype=np.float64),
            minlength=n,
        ).astype(np.int64)
    mol = {
        "n_atoms": np.int64(n),
        "n_bonds": np.int64(m),
        "atom_feat": atom_feat,
        "atom_type": atom_type,
        "atom_scalar": S.scalar_features(analysis, per_root, closure_counts),
        "bond_feat": bond_feat,
        "bond_type": bond_type,
        "bond_index": np.stack([u, v], axis=1).astype(np.int64) if m else np.zeros((0, 2), dtype=np.int64),
        "nbr_ptr": s1_ptr,
        "nbr_code": s1_code,
        "path_ptr": s2["path_ptr"],
        "path_cat": s2["path_cat"],
        "path_end63": s2["path_end63"],
        "path_end": s2["path_end"],
        "path_closure": s2["path_closure"],
        "global": S.global_features(analysis, atom_type),
        "n_components": np.int64(analysis.n_components),
        "n_junctions": np.int64(analysis.n_junctions),
        "n_ring_systems": np.int64(analysis.n_ring_systems),
        "molecule_id": f"{dataset}#{record['gid']}",
        "smiles": "",
    }
    PREP_DATA.validate_molecule(mol, spec)
    return {
        "mol": mol,
        "per_graph": {
            "gid": int(record["gid"]),
            "n_atoms": int(n),
            "n_bonds": int(m),
            "s1_codes": int(s1_ptr[-1]),
            "s2_paths": int(s2["path_ptr"][-1]),
        },
    }


def build_folds(y: np.ndarray, dataset: str) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold, train_test_split

    y = np.asarray(y, dtype=np.int64)
    skf = StratifiedKFold(n_splits=C.TU_N_SPLITS, shuffle=True, random_state=C.TU_OUTER_SPLIT_SEED)
    folds = []
    for fold, (train_pool, test_rows) in enumerate(skf.split(np.zeros(y.size, dtype=np.float64), y)):
        train_pool = np.asarray(train_pool, dtype=np.int64)
        test_rows = np.asarray(test_rows, dtype=np.int64)
        train_rows, valid_rows = train_test_split(
            train_pool,
            test_size=1.0 / 9.0,
            stratify=y[train_pool],
            random_state=C.TU_INNER_VALID_SEED_BASE + int(fold),
        )
        train_rows = np.sort(np.asarray(train_rows, dtype=np.int64))
        valid_rows = np.sort(np.asarray(valid_rows, dtype=np.int64))
        test_rows = np.sort(test_rows)
        union = np.concatenate([train_rows, valid_rows, test_rows])
        if np.unique(union).size != y.size or not np.array_equal(np.sort(union), np.arange(y.size)):
            raise RuntimeError(f"{dataset} fold {fold}: split is not a partition")
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            rows_a = {"train": train_rows, "valid": valid_rows, "test": test_rows}[a]
            rows_b = {"train": train_rows, "valid": valid_rows, "test": test_rows}[b]
            if np.intersect1d(rows_a, rows_b).size:
                raise RuntimeError(f"{dataset} fold {fold}: {a}/{b} overlap")
        payload = {
            "fold": int(fold),
            "outer_test_fold": int(fold),
            "inner_valid_seed": C.TU_INNER_VALID_SEED_BASE + int(fold),
            "model_seed": C.TU_MODEL_SEED_BASE + int(fold),
            "sizes": {"train": int(train_rows.size), "valid": int(valid_rows.size), "test": int(test_rows.size)},
            "class_counts": {
                split: [int(x) for x in np.bincount(y[rows], minlength=int(y.max()) + 1).tolist()]
                for split, rows in (("train", train_rows), ("valid", valid_rows), ("test", test_rows))
            },
            "index_sha256": {
                split: C.index_hash(rows)
                for split, rows in (("train", train_rows), ("valid", valid_rows), ("test", test_rows))
            },
            "rows": {
                "train": train_rows.tolist(),
                "valid": valid_rows.tolist(),
                "test": test_rows.tolist(),
            },
        }
        payload["partition_sha256"] = C.sha256_bytes(
            b"".join(rows.tobytes() for rows in (train_rows, valid_rows, test_rows))
        )
        folds.append(payload)
    return {
        "dataset": dataset,
        "n_splits": C.TU_N_SPLITS,
        "shuffle": True,
        "random_state": C.TU_OUTER_SPLIT_SEED,
        "inner_valid_random_state": f"{C.TU_INNER_VALID_SEED_BASE} + fold_id",
        "folds": folds,
    }


def precheck_counts(name: str) -> dict[str, Any]:
    """Exact per-graph S2 counts without materialising any S2 record."""
    t0 = time.perf_counter()
    ds = scan_dataset(name)
    rows = []
    for rec in ds["records"]:
        n = int(rec["n_atoms"])
        pairs = np.asarray(rec["pairs"], dtype=np.int64).reshape(-1, 2)
        m = int(pairs.shape[0])
        degree = MG.degrees(n, pairs[:, 0], pairs[:, 1]) if m else np.zeros(n, dtype=np.int64)
        bond_type = np.ones(m, dtype=np.int64)
        g = MG.build_molecular_graph(
            atom_type=np.zeros(n, dtype=np.int64),
            bond_index=pairs,
            bond_type=bond_type if m else np.zeros(0, dtype=np.int64),
            atom_feat=np.zeros((n, 1), dtype=np.float32),
            bond_feat=np.zeros((m, 1), dtype=np.float32),
        )
        adj = MG.adjacency(n, g.bond_u, g.bond_v, g.bond_type)
        counts = BG.s2_counts_from_formula(adj)
        rows.append({"gid": int(rec["gid"]), "n_atoms": n, "n_bonds": m, "s2_paths": int(counts.sum())})
    arr_n = np.asarray([r["n_atoms"] for r in rows], dtype=np.int64)
    arr_m = np.asarray([r["n_bonds"] for r in rows], dtype=np.int64)
    arr_s = np.asarray([r["s2_paths"] for r in rows], dtype=np.int64)
    out = {
        "dataset": name,
        "scanned_graphs": len(rows),
        "scan_seconds": float(time.perf_counter() - t0),
        "n_atoms": _dist(arr_n),
        "n_bonds": _dist(arr_m),
        "s2_paths": _dist(arr_s),
        "s2_paths_total": int(arr_s.sum()),
        "per_graph": rows,
        "cache_built": False,
        "method": "exact formula per root: sum_{w in N(v)} (deg(w)-1); no S2 record stored",
    }
    return out


def _dist(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "min": int(a.min()) if a.size else 0,
        "max": int(a.max()) if a.size else 0,
        "mean": float(a.mean()) if a.size else 0.0,
        "p50": float(np.percentile(a, 50)) if a.size else 0.0,
        "p95": float(np.percentile(a, 95)) if a.size else 0.0,
        "p99": float(np.percentile(a, 99)) if a.size else 0.0,
        "total": int(a.sum()) if a.size else 0,
    }


def build_dataset(name: str, *, force: bool = False) -> dict[str, Any]:
    """Build one label-free cache and the frozen 10-fold split record.

    For the four pre-declared large-graph datasets, the exact S2 count is
    computed first.  If it exceeds ``EAGER_S2_LIMIT`` the dataset is blocked
    here rather than materialising a multi-gigabyte path cache or silently
    sampling/truncating S2.
    """
    if name in C.LARGE_GRAPH_DATASETS:
        counts = precheck_counts(name)
        C.write_json(C.ANALYSIS / f"{name}_precheck_counts.json", counts)
        if counts["s2_paths_total"] > C.EAGER_S2_LIMIT:
            raise LargeGraphBlocked(
                f"{name}: exact S2 total {counts['s2_paths_total']} exceeds "
                f"EAGER_S2_LIMIT={C.EAGER_S2_LIMIT}; eager per-path cache not materialised"
            )
    ds = scan_dataset(name)
    node_types = _node_type_table(ds)
    bond_types = _bond_type_table(ds)
    cont_dim = int(ds["stats"]["continuous_node_attribute_dim"])
    spec = S.tu_spec(
        name,
        node_types["names"],
        int(node_types["unk_index"]),
        bond_types["names"],
        int(bond_types["unk_id"]),
        continuous_atom_dim=cont_dim,
        source_provenance={
            "dataset": name,
            "loader": "torch_geometric.datasets.TUDataset(cleaned=False, use_node_attr=True, use_edge_attr=True)",
            "node_label_mode": node_types["mode"],
            "node_type_encoding": node_types["encoding"],
            "bond_type_encoding": bond_types["encoding"],
            "source_manifest": raw_manifest(name),
        },
    )
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    spec_path = C.ANALYSIS / f"{name}_spec.json"
    save_spec(spec, spec_path)
    cache_path = C.CACHE / f"{name}.pt"
    t0 = time.perf_counter()
    mols = []
    per_graph = []
    for i, rec in enumerate(ds["records"]):
        built = _build_molecule(rec, spec, node_types, bond_types, name)
        mols.append(built["mol"])
        per_graph.append(built["per_graph"])
        if (i + 1) % 1000 == 0:
            C.log(f"[cache {name}] {i + 1}/{len(ds['records'])} graphs {time.perf_counter() - t0:.1f}s")
    cache = PREP_DATA.concat_molecules(mols, spec, split="all")
    cache["gids"] = np.arange(len(mols), dtype=np.int64)
    cache["meta"]["build_seconds"] = float(time.perf_counter() - t0)
    cache["meta"]["dataset"] = name
    PREP_DATA.validate_cache(cache, spec)
    D.save_cache(cache, cache_path)
    y = np.asarray(ds["y"], dtype=np.int64)
    np.save(C.ANALYSIS / f"{name}_labels.npy", y)
    folds = build_folds(y, name)
    C.write_json(C.ANALYSIS / f"{name}_folds.json", folds)
    arr_n = np.asarray([r["n_atoms"] for r in per_graph], dtype=np.int64)
    arr_m = np.asarray([r["n_bonds"] for r in per_graph], dtype=np.int64)
    arr_s = np.asarray([r["s2_paths"] for r in per_graph], dtype=np.int64)
    np.savez(
        C.ANALYSIS / f"{name}_graph_distribution.npz",
        gids=np.asarray([r["gid"] for r in per_graph], dtype=np.int64),
        n_atoms=arr_n,
        n_bonds=arr_m,
        s2_paths=arr_s,
    )
    stats = {
        "dataset": name,
        "cache": C.track_rel(cache_path),
        "cache_sha256": C.sha256_file(cache_path),
        "cache_bytes": int(cache_path.stat().st_size),
        "cache_build_seconds": float(cache["meta"]["build_seconds"]),
        "feature_spec_hash": spec.hash(),
        "feature_spec": spec.to_dict(include_hash=False),
        "data": ds["stats"],
        "node_types": node_types,
        "bond_types": bond_types,
        "n_atoms": _dist(arr_n),
        "n_bonds": _dist(arr_m),
        "s2_paths": _dist(arr_s),
        "s2_paths_total": int(arr_s.sum()),
        "s1_codes_total": int(cache["meta"]["n_s1_codes"]),
        "cache_validation": PREP_DATA.validate_cache(cache, spec),
    }
    C.write_json(C.ANALYSIS / f"{name}_cache_stats.json", stats)
    C.write_json(C.ANALYSIS / f"{name}_data_manifest.json", raw_manifest(name))
    return stats


__all__ = [
    "InputAdapterBlocked",
    "LargeGraphBlocked",
    "raw_dir",
    "raw_manifest",
    "scan_dataset",
    "build_folds",
    "build_dataset",
    "precheck_counts",
]
