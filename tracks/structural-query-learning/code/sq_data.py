"""Official OGB data, canonical physical-bond graphs and label-free primitives.

Design
======
The round uses the two official OGB graph-property tasks ``ogbg-molesol`` and
``ogbg-mollipo`` with ``PygGraphPropPredDataset`` + ``get_idx_split()`` and the
OGB ``Evaluator`` for the official metric.  All of that happens in
``prepare_dataset`` on a machine with the official raw files; the resulting
*label-free* graph cache (raw atom/bond columns, computed unlabeled structural
primitives, official row ids, official split indices, targets) is frozen into a
small committed ``.npz`` and verified by SHA-256 at every run on either machine.
That is the engineering adaptation the round brief allows ("one local data
preparation, then hand the label-free cache to res"): ``res`` has no RDKit and
must not depend on a live download for a formal result.

Graph conventions (pinned by tests)
==================================
* Atom features are the raw OGB nine categorical columns, dtype kept integral.
* A physical undirected bond is stored exactly once (``u < v``); the OGB
  directed duplicate records are merged only after proving row equality of the
  three bond columns.  Degrees and neighbour counts use physical bonds.
* ``min_ring[v]`` is the shortest simple cycle through ``v`` computed as
  ``2 + dist_G-v(u, w)`` over all neighbour pairs ``(u, w)`` of ``v`` — the
  definition is labelling independent and is *not* derived from an arbitrary
  ``cycle_basis``.  Buckets: ``0`` acyclic, ``3..8`` exact, ``9`` means ``>=9``.
* Connected components keep every molecule (including isolated atoms and
  multi-component salts); no distance crosses components.
* Targets are standardised with Train mean and population std (``ddof=0``);
  all reported metrics are back in raw units with no second un-standardisation.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sq_common as C
from sq_ast import PackedGraph

CACHE_VERSION = "sq-pilot-cache-v1"

ATOM_COLUMNS = 9
BOND_COLUMNS = 3

ATOM_COLUMN_NAMES: tuple[str, ...] = (
    "atomic_num",
    "chirality",
    "degree",
    "formal_charge",
    "num_hs",
    "num_radical_electrons",
    "hybridization",
    "is_aromatic",
    "is_in_ring",
)
BOND_COLUMN_NAMES: tuple[str, ...] = ("bond_type", "bond_stereo", "bond_is_in_ring")


# ---------------------------------------------------------------------------
# low-level graph algorithms (no labels, no learned quantity)
# ---------------------------------------------------------------------------


def canonical_physical_bonds(
    n_atoms: int,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge OGB directed bond records into sorted physical bonds.

    Every directed duplicate must agree on *all* bond feature columns; a
    disagreement is a data error (never silently averaged).  Returns
    ``(u, v, feat)`` with ``u < v`` sorted by ``(u, v)``.
    """
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.size == 0:
        ei = ei.reshape(2, 0)
    if ei.shape[0] != 2:
        raise ValueError(f"edge_index must be [2, E], got {ei.shape}")
    ea = np.asarray(edge_attr)
    if ea.size == 0:
        ea = ea.reshape(0, BOND_COLUMNS)
    if ea.shape[0] != ei.shape[1]:
        raise ValueError("edge_attr rows do not match edge_index columns")
    if np.any(ei < 0) or np.any(ei >= n_atoms):
        raise ValueError("bond endpoint outside atom range")
    if np.any(ei[0] == ei[1]):
        raise ValueError("self-loop in molecular graph")
    seen: dict[tuple[int, int], tuple[int, ...]] = {}
    for k in range(ei.shape[1]):
        a = int(ei[0, k])
        b = int(ei[1, k])
        key = (a, b) if a < b else (b, a)
        feat = tuple(int(x) for x in ea[k])
        if key in seen and seen[key] != feat:
            raise ValueError(f"directed bond records disagree for {key}: {seen[key]} vs {feat}")
        seen[key] = feat
    keys = sorted(seen)
    u = np.asarray([k[0] for k in keys], dtype=np.int32)
    v = np.asarray([k[1] for k in keys], dtype=np.int32)
    feat = np.asarray([seen[k] for k in keys], dtype=np.int16) if keys else np.zeros((0, BOND_COLUMNS), np.int16)
    return u, v, feat


def min_ring_buckets(n_atoms: int, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact shortest-ring-size bucket per atom, from the physical bond graph.

    ``0`` = acyclic (no simple cycle through the atom), ``3..8`` = exact size,
    ``9`` = ``>= 9``.  Computed as ``2 + dist_{G-v}(x, y)`` over neighbour pairs
    with ``v`` deleted, i.e. a genuine shortest cycle through ``v``.
    """
    adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
    for a, b in zip(u.tolist(), v.tolist()):
        adjacency[a].append(b)
        adjacency[b].append(a)
    buckets = np.zeros(n_atoms, dtype=np.int8)
    for node in range(n_atoms):
        neighbors = adjacency[node]
        if len(neighbors) < 2:
            continue
        best = None
        for i in range(len(neighbors)):
            start = neighbors[i]
            # BFS in G - node from one neighbour to every other neighbour.
            dist = {start: 0}
            frontier = [start]
            while frontier and (best is None or 2 + dist[frontier[0]] < best):
                nxt: list[int] = []
                for cur in frontier:
                    for other in adjacency[cur]:
                        if other == node or other in dist:
                            continue
                        dist[other] = dist[cur] + 1
                        nxt.append(other)
                frontier = nxt
            for j in range(len(neighbors)):
                if j == i:
                    continue
                target = neighbors[j]
                d = dist.get(target)
                if d is None:
                    continue
                length = 2 + d
                if best is None or length < best:
                    best = length
        if best is None:
            continue
        buckets[node] = 9 if best >= 9 else int(best)
    return buckets, (buckets > 0)


def connected_components(n_atoms: int, u: np.ndarray, v: np.ndarray) -> int:
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(u.tolist(), v.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n_atoms)})


# ---------------------------------------------------------------------------
# OGB loading (prepare stage, requires the official dataset on disk)
# ---------------------------------------------------------------------------


def read_ogb_smiles(dataset_root: Path) -> list[str]:
    path = Path(dataset_root) / "mapping" / "mol.csv.gz"
    with gzip.open(path, "rt") as handle:
        reader = csv.DictReader(handle)
        return [row["smiles"].strip() for row in reader]


def prepare_dataset(
    dataset: str,
    *,
    root: str | Path = "data/ogb",
    verify: bool = True,
) -> dict[str, Any]:
    """Read the official OGB dataset and build the label-free cache payload."""
    from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = C.REPO_ROOT / root_path
    ogb_dataset = PygGraphPropPredDataset(name=dataset, root=str(root_path))
    split = ogb_dataset.get_idx_split()
    n_graphs = len(ogb_dataset)

    node_ptr = [0]
    bond_ptr = [0]
    atom_chunks: list[np.ndarray] = []
    degree_chunks: list[np.ndarray] = []
    ring_chunks: list[np.ndarray] = []
    in_cycle_chunks: list[np.ndarray] = []
    bond_u: list[int] = []
    bond_v: list[int] = []
    bond_feat: list[np.ndarray] = []
    labels = np.zeros(n_graphs, dtype=np.float32)
    components = np.zeros(n_graphs, dtype=np.int32)
    cycle_rank = np.zeros(n_graphs, dtype=np.int32)
    bond_count = np.zeros(n_graphs, dtype=np.int32)
    node_count = np.zeros(n_graphs, dtype=np.int32)

    for graph_id in range(n_graphs):
        data = ogb_dataset[graph_id]
        x = np.asarray(data.x, dtype=np.int16)
        if x.ndim != 2 or x.shape[1] != ATOM_COLUMNS:
            raise ValueError(f"graph {graph_id}: atom feature shape {x.shape}")
        n_atoms = int(x.shape[0])
        y = np.asarray(data.y, dtype=np.float64).reshape(-1)
        if y.size != 1:
            raise ValueError(f"graph {graph_id}: expected a scalar target, got {y.size}")
        labels[graph_id] = float(y[0])

        u, v, feat = canonical_physical_bonds(
            n_atoms, np.asarray(data.edge_index), np.asarray(data.edge_attr)
        )
        degree = np.zeros(n_atoms, dtype=np.int8)
        for a, b in zip(u.tolist(), v.tolist()):
            degree[a] += 1
            degree[b] += 1
        if n_atoms and int(degree.max()) > 127:  # pragma: no cover - molecular graphs
            raise ValueError("degree does not fit int8")
        rings, in_cycle = min_ring_buckets(n_atoms, u, v)
        n_components = connected_components(n_atoms, u, v)

        atom_chunks.append(x)
        degree_chunks.append(degree.astype(np.int16))
        ring_chunks.append(rings.astype(np.int16))
        in_cycle_chunks.append(in_cycle.astype(np.uint8))
        bond_u.extend(u.tolist())
        bond_v.extend(v.tolist())
        if len(feat):
            bond_feat.append(feat)
        node_ptr.append(node_ptr[-1] + n_atoms)
        bond_ptr.append(bond_ptr[-1] + int(len(u)))
        components[graph_id] = n_components
        node_count[graph_id] = n_atoms
        bond_count[graph_id] = int(len(u))
        cycle_rank[graph_id] = int(len(u) - n_atoms + n_components)

    smiles = read_ogb_smiles(Path(ogb_dataset.root))
    if len(smiles) != n_graphs:
        raise ValueError(f"{dataset}: mapping/mol.csv.gz has {len(smiles)} rows, expected {n_graphs}")

    payload = {
        "dataset": dataset,
        "cache_version": CACHE_VERSION,
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "bond_ptr": np.asarray(bond_ptr, dtype=np.int64),
        "atom_feat": np.concatenate(atom_chunks, axis=0).astype(np.uint8) if atom_chunks else np.zeros((0, ATOM_COLUMNS), np.uint8),
        "degree": np.concatenate(degree_chunks).astype(np.uint8) if degree_chunks else np.zeros(0, np.uint8),
        "min_ring": np.concatenate(ring_chunks).astype(np.uint8) if ring_chunks else np.zeros(0, np.uint8),
        "in_cycle": np.concatenate(in_cycle_chunks).astype(np.uint8) if in_cycle_chunks else np.zeros(0, np.uint8),
        "bond_u": np.asarray(bond_u, dtype=np.int32),
        "bond_v": np.asarray(bond_v, dtype=np.int32),
        "bond_feat": np.concatenate(bond_feat, axis=0).astype(np.uint8) if bond_feat else np.zeros((0, BOND_COLUMNS), np.uint8),
        "labels": labels,
        "graph_n_components": components,
        "graph_cycle_rank": cycle_rank,
        "graph_n_bonds": bond_count,
        "graph_n_atoms": node_count,
        "smiles": np.asarray(smiles, dtype=str),
        "row_ids": np.arange(n_graphs, dtype=np.int64),
        "train_idx": np.asarray(sorted(int(i) for i in split["train"]), dtype=np.int64),
        "valid_idx": np.asarray(sorted(int(i) for i in split["valid"]), dtype=np.int64),
        "test_idx": np.asarray(sorted(int(i) for i in split["test"]), dtype=np.int64),
    }

    audit = {
        "dataset": dataset,
        "n_graphs": n_graphs,
        "split_sizes": {k: int(len(split[k])) for k in ("train", "valid", "test")},
        "split_hash": {k: C.sha256_array(payload[f"{k}_idx"]) for k in ("train", "valid", "test")},
        "label_range": {
            "min": float(labels.min()),
            "max": float(labels.max()),
            "mean": float(labels.mean()),
        },
        "atom_columns": ATOM_COLUMNS,
        "bond_columns": BOND_COLUMNS,
        "n_atoms_total": int(payload["node_ptr"][-1]),
        "n_physical_bonds_total": int(payload["bond_ptr"][-1]),
        "n_directed_bond_records": None,
        "n_components_total": int(components.sum()),
        "n_multi_component_molecules": int((components > 1).sum()),
        "n_isolated_atom_molecules": int((node_count == 1).sum()),
        "cycle_rank_total": int(cycle_rank.sum()),
        "ring_bucket_counts": {
            str(b): int((np.concatenate(ring_chunks) == b).sum()) for b in range(0, 10)
        },
        "evaluator": _evaluator_metric_name(dataset) if verify else None,
    }
    if verify:
        audit.update(verify_against_ogb(payload, ogb_dataset, dataset))
    return {"payload": payload, "audit": audit}


def verify_against_ogb(payload: Mapping[str, Any], ogb_dataset, dataset: str) -> dict[str, Any]:
    """Cross-checks that the cache really is the official dataset."""
    from ogb.graphproppred import Evaluator

    expected = C.EXPECTED_SPLIT_SIZES[dataset]
    sizes = {k: int(len(payload[f"{k}_idx"])) for k in ("train", "valid", "test")}
    if sizes != expected:
        raise ValueError(f"{dataset}: official split sizes {sizes} != expected {expected}")
    n_graphs = int(payload["row_ids"].size)
    labels = np.asarray(payload["labels"], dtype=np.float64)
    max_label_diff = 0.0
    max_node_diff = 0
    max_bond_diff = 0
    for graph_id in range(n_graphs):
        data = ogb_dataset[graph_id]
        max_label_diff = max(
            max_label_diff, abs(float(np.asarray(data.y).reshape(-1)[0]) - labels[graph_id])
        )
        max_node_diff = max(max_node_diff, abs(int(data.x.shape[0]) - int(payload["graph_n_atoms"][graph_id])))
        directed = int(np.asarray(data.edge_index).shape[1])
        physical = int(payload["graph_n_bonds"][graph_id])
        if directed != 2 * physical:
            raise ValueError(
                f"graph {graph_id}: {directed} directed bond records != 2 * {physical} physical bonds"
            )
        max_bond_diff = max(max_bond_diff, abs(directed - 2 * physical))
    if max_label_diff > 1e-9 or max_node_diff or max_bond_diff:
        raise ValueError(
            f"{dataset}: cache diverges from the official dataset "
            f"(label {max_label_diff}, nodes {max_node_diff}, bonds {max_bond_diff})"
        )
    evaluator = Evaluator(dataset)
    probe = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
    official_rmse = float(
        evaluator.eval({"y_true": probe, "y_pred": probe, "name": evaluator.eval_metric})[
            evaluator.eval_metric
        ]
    )
    return {
        "verified_against_ogb": True,
        "max_label_abs_diff": float(max_label_diff),
        "max_node_abs_diff": int(max_node_diff),
        "self_prediction_rmse": official_rmse,
        "evaluator_metric": evaluator.eval_metric,
    }


# ---------------------------------------------------------------------------
# cache IO
# ---------------------------------------------------------------------------


def cache_path(dataset: str, *, base: Path | None = None) -> Path:
    root = C.DATA_DIR if base is None else Path(base)
    return root / f"{dataset.replace('-', '_')}_{CACHE_VERSION}.npz"


def save_cache(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in payload.items()})
    return {"path": str(path), "sha256": C.sha256_file(path), "bytes": int(path.stat().st_size)}


def load_cache(dataset: str, *, base: Path | None = None, verify_sha: str | None = None) -> dict[str, Any]:
    path = cache_path(dataset, base=base)
    if not path.is_file():
        raise FileNotFoundError(f"graph cache missing for {dataset}: {path}")
    digest = C.sha256_file(path)
    if verify_sha is not None and digest != verify_sha:
        raise ValueError(f"graph cache sha mismatch for {dataset}: {digest} != {verify_sha}")
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["cache_sha256"] = digest
    payload["cache_path"] = str(path)
    payload["smiles"] = [str(s) for s in payload["smiles"].tolist()]
    for key in ("node_ptr", "bond_ptr", "atom_feat", "degree", "min_ring", "in_cycle",
                "bond_u", "bond_v", "bond_feat", "labels", "row_ids",
                "train_idx", "valid_idx", "test_idx", "graph_n_bonds", "graph_n_atoms",
                "graph_n_components", "graph_cycle_rank"):
        payload[key] = np.asarray(payload[key])
    return payload


# ---------------------------------------------------------------------------
# packed split views
# ---------------------------------------------------------------------------


SPLIT_KEYS = {"train": "train_idx", "valid": "valid_idx", "test": "test_idx"}


def split_indices(payload: Mapping[str, Any], split: str) -> np.ndarray:
    return np.asarray(payload[SPLIT_KEYS[split]], dtype=np.int64)


def packed_split(payload: Mapping[str, Any], split: str) -> PackedGraph:
    """Concatenate one split into the packed executor representation."""
    indices = split_indices(payload, split)
    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    atom_feat = np.asarray(payload["atom_feat"])
    degree = np.asarray(payload["degree"])
    min_ring = np.asarray(payload["min_ring"])
    in_cycle = np.asarray(payload["in_cycle"])
    bond_u = np.asarray(payload["bond_u"], dtype=np.int32)
    bond_v = np.asarray(payload["bond_v"], dtype=np.int32)
    bond_feat = np.asarray(payload["bond_feat"])
    labels = np.asarray(payload["labels"], dtype=np.float32)

    node_parts = []
    node_graph_parts = []
    degree_parts = []
    ring_parts = []
    cycle_parts = []
    bu_parts = []
    bv_parts = []
    bf_parts = []
    node_offset = 0
    for local, graph_id in enumerate(indices.tolist()):
        a0, a1 = int(node_ptr[graph_id]), int(node_ptr[graph_id + 1])
        b0, b1 = int(bond_ptr[graph_id]), int(bond_ptr[graph_id + 1])
        n = a1 - a0
        node_parts.append(atom_feat[a0:a1])
        node_graph_parts.append(np.full(n, local, dtype=np.int32))
        degree_parts.append(degree[a0:a1])
        ring_parts.append(min_ring[a0:a1])
        cycle_parts.append(in_cycle[a0:a1].astype(bool))
        bu_parts.append(bond_u[b0:b1] + node_offset)
        bv_parts.append(bond_v[b0:b1] + node_offset)
        bf_parts.append(bond_feat[b0:b1])
        node_offset += n

    def _cat(parts, empty_shape, dtype):
        if parts:
            return np.concatenate(parts, axis=0).astype(dtype)
        return np.zeros(empty_shape, dtype=dtype)

    packed = PackedGraph(
        dataset=str(payload["dataset"]),
        split=split,
        n_graphs=int(len(indices)),
        n_nodes=int(node_offset),
        node_graph=_cat(node_graph_parts, (0,), np.int32),
        atom_feat=_cat(node_parts, (0, ATOM_COLUMNS), np.int16).astype(np.int16),
        degree=_cat(degree_parts, (0,), np.int16).astype(np.int16),
        min_ring=_cat(ring_parts, (0,), np.int16).astype(np.int16),
        in_cycle=_cat(cycle_parts, (0,), bool),
        n_bonds=int(sum(len(p) for p in bu_parts)),
        bond_u=_cat(bu_parts, (0,), np.int32),
        bond_v=_cat(bv_parts, (0,), np.int32),
        bond_feat=_cat(bf_parts, (0, BOND_COLUMNS), np.int16).astype(np.int16),
        graph_n_atoms=np.asarray(payload["graph_n_atoms"])[indices].astype(np.int32),
        graph_n_bonds=np.asarray(payload["graph_n_bonds"])[indices].astype(np.int32),
        graph_n_components=np.asarray(payload["graph_n_components"])[indices].astype(np.int32),
        graph_cycle_rank=np.asarray(payload["graph_cycle_rank"])[indices].astype(np.int32),
        row_ids=indices.astype(np.int64),
        labels=labels[indices].astype(np.float32),
        smiles=[payload["smiles"][int(i)] for i in indices.tolist()],
    )
    return packed


def train_vocabulary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Train-only value sets for atom / bond columns (never Val/Test labels)."""
    train = split_indices(payload, "train")
    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    atom_feat = np.asarray(payload["atom_feat"])
    bond_feat = np.asarray(payload["bond_feat"])
    atom_values = {j: set() for j in range(ATOM_COLUMNS)}
    bond_values = {j: set() for j in range(BOND_COLUMNS)}
    degree_values: set[int] = set()
    ring_values: set[int] = set()
    for graph_id in train.tolist():
        a0, a1 = int(node_ptr[graph_id]), int(node_ptr[graph_id + 1])
        b0, b1 = int(bond_ptr[graph_id]), int(bond_ptr[graph_id + 1])
        block = atom_feat[a0:a1]
        for j in range(ATOM_COLUMNS):
            atom_values[j].update(int(v) for v in np.unique(block[:, j]).tolist())
        bonds = bond_feat[b0:b1]
        for j in range(BOND_COLUMNS):
            bond_values[j].update(int(v) for v in np.unique(bonds[:, j]).tolist())
    packed = packed_split(payload, "train")
    degree_values.update(int(v) for v in np.unique(packed.degree).tolist())
    ring_values.update(int(v) for v in np.unique(packed.min_ring).tolist())
    return {
        "atom_values": {j: sorted(v) for j, v in atom_values.items()},
        "bond_values": {j: sorted(v) for j, v in bond_values.items()},
        "degree_values": sorted(degree_values),
        "ring_values": sorted(ring_values),
        "element_values": sorted(atom_values[0]),
    }


def official_evaluator(dataset: str):
    from ogb.graphproppred import Evaluator

    return Evaluator(dataset)


def _evaluator_metric_name(dataset: str) -> str:
    from ogb.graphproppred import Evaluator

    return str(Evaluator(dataset).eval_metric)


def official_metric(dataset: str, pred: np.ndarray, target: np.ndarray, *, unit: str = "raw") -> dict[str, float]:
    """Official OGB metric (RMSE for both tasks) plus MAE, in the given units."""
    evaluator = official_evaluator(dataset)
    metric = str(evaluator.eval_metric)
    y_true = np.asarray(target, dtype=np.float64).reshape(-1, 1)
    y_pred = np.asarray(pred, dtype=np.float64).reshape(-1, 1)
    values = evaluator.eval({"y_true": y_true, "y_pred": y_pred, "name": metric})
    return {
        f"{unit}_{metric}": float(values[metric]),
        f"{unit}_mae": C.mae(y_pred, y_true),
    }


# ---------------------------------------------------------------------------
# classic ECFP4 + descriptors (local RDKit only; frozen with the cache)
# ---------------------------------------------------------------------------

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "TotalFormalCharge",
)


def build_ecfp_payload(dataset: str) -> dict[str, Any]:
    """Morgan r=2 / 2048 bits / chirality + the ten frozen descriptors.

    Parsing failures are kept (never dropped): all-zero fingerprint, the Train
    median descriptor vector filled by the caller, and an explicit flag.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    payload = load_cache(dataset)
    smiles = payload["smiles"]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    n_graphs = len(smiles)
    bits = np.zeros((n_graphs, 256), dtype=np.uint8)
    desc = np.full((n_graphs, len(DESCRIPTOR_NAMES)), np.nan, dtype=np.float32)
    failed = np.zeros(n_graphs, dtype=np.uint8)
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(text).strip())
        if mol is None:
            failed[i] = 1
            continue
        fp = generator.GetFingerprint(mol)
        bit_chars = np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")
        if bit_chars.size != 2048:  # pragma: no cover - rdkit contract
            raise ValueError(f"unexpected fingerprint length {bit_chars.size}")
        bits[i] = np.packbits(bit_chars.astype(np.uint8))
        raw = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            float(mol.GetNumHeavyAtoms()),
            float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        ]
        desc[i] = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(desc[failed == 0]).all():
        raise ValueError(f"{dataset}: descriptor computation produced non-finite values")
    return {
        "dataset": dataset,
        "ecfp_bits": bits,
        "descriptors": desc,
        "parse_failed": failed,
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "n_failed": int(failed.sum()),
    }


def ecfp_path(dataset: str, *, base: Path | None = None) -> Path:
    root = C.DATA_DIR if base is None else Path(base)
    return root / f"{dataset.replace('-', '_')}_ecfp4_{CACHE_VERSION}.npz"


def save_ecfp(ecfp: Mapping[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ecfp_bits=np.asarray(ecfp["ecfp_bits"]),
        descriptors=np.asarray(ecfp["descriptors"], dtype=np.float32),
        parse_failed=np.asarray(ecfp["parse_failed"], dtype=np.uint8),
        descriptor_names=np.asarray(ecfp["descriptor_names"], dtype=str),
    )
    return {"path": str(path), "sha256": C.sha256_file(path), "bytes": int(path.stat().st_size)}


def load_ecfp(dataset: str, *, base: Path | None = None) -> dict[str, Any]:
    path = ecfp_path(dataset, base=base)
    if not path.is_file():
        raise FileNotFoundError(f"ECFP cache missing for {dataset}: {path}")
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["descriptor_names"] = [str(s) for s in payload["descriptor_names"].tolist()]
    payload["cache_sha256"] = C.sha256_file(path)
    return payload


def ecfp_design_matrix(
    ecfp: Mapping[str, Any], payload: Mapping[str, Any], split: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dense float32 design matrix: 2048 bits + 10 descriptors + failure flag."""
    indices = split_indices(payload, split)
    bits = np.asarray(ecfp["ecfp_bits"], dtype=np.uint8)[indices]
    desc = np.asarray(ecfp["descriptors"], dtype=np.float32)[indices]
    failed = np.asarray(ecfp["parse_failed"], dtype=np.uint8)[indices]
    train_idx = split_indices(payload, "train")
    train_failed = np.asarray(ecfp["parse_failed"], dtype=np.uint8)[train_idx]
    train_desc = np.asarray(ecfp["descriptors"], dtype=np.float32)[train_idx]
    ok = train_failed == 0
    if not ok.any():
        raise ValueError("no successfully parsed Train molecule; descriptors have no median")
    median = np.nanmedian(train_desc[ok], axis=0)
    desc = np.where(np.isfinite(desc), desc, median[None, :])
    bits_unpacked = np.unpackbits(bits, axis=1).astype(np.float32)
    design = np.concatenate(
        [bits_unpacked, desc.astype(np.float32), failed.reshape(-1, 1).astype(np.float32)], axis=1
    )
    names = [f"morgan4_{i}" for i in range(2048)] + list(ecfp["descriptor_names"]) + ["rdkit_parse_failed"]
    stats = {
        "n_columns": int(design.shape[1]),
        "n_parse_failed_total": int(np.asarray(ecfp["parse_failed"]).sum()),
        "n_parse_failed_train": int(train_failed.sum()),
        "n_parse_failed_in_split": int(failed.sum()),
        "train_median_used_for_failures": median.tolist(),
    }
    return design.astype(np.float32), {"names": names, **stats}


# ---------------------------------------------------------------------------
# frozen ``lipo_v1`` raw attributes (local RDKit only; for the C1 reference L)
# ---------------------------------------------------------------------------

LIPO_RAW_VERSION = "sq-pilot-lipo-raw-v1"
LIPO_RAW_ATOM_CLASSES: tuple[tuple[str, bool], ...] = (
    ("B", False), ("Br", False), ("C", False), ("C", True), ("Cl", False),
    ("F", False), ("I", False), ("N", False), ("N", True), ("O", False),
    ("O", True), ("P", False), ("S", False), ("S", True), ("Si", False),
)
LIPO_RAW_ATOM_UNK = len(LIPO_RAW_ATOM_CLASSES)
LIPO_RAW_BOND_CLASSES: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
LIPO_RAW_BOND_UNK = 4
LIPO_RAW_CHARGE_BUCKETS: tuple[int, ...] = (-2, -1, 0, 1, 2)
LIPO_RAW_H_BUCKETS: tuple[int, ...] = (0, 1, 2, 3, 4)


def _lipo_charge_code(value: int) -> int:
    return LIPO_RAW_CHARGE_BUCKETS.index(int(value)) if int(value) in LIPO_RAW_CHARGE_BUCKETS else 5


def _lipo_h_code(value: int) -> int:
    value = int(value)
    if value in LIPO_RAW_H_BUCKETS:
        return LIPO_RAW_H_BUCKETS.index(value)
    if value > LIPO_RAW_H_BUCKETS[-1]:
        return len(LIPO_RAW_H_BUCKETS) - 1
    return 5


def _lipo_stereo_code(stereo) -> int:
    from rdkit import Chem

    return {
        Chem.BondStereo.STEREONONE: 0,
        Chem.BondStereo.STEREOE: 1,
        Chem.BondStereo.STEREOZ: 2,
    }.get(stereo, 3)


def build_lipo_raw_payload(dataset: str) -> dict[str, Any]:
    """Frozen ``lipo_v1`` raw attributes for every official row (no labels read).

    This mirrors ``lcrc_molnet_benchmark_pilot_v2/code/mb2_rdkit.py`` (the frozen
    extraction): atom class = (element, is_aromatic) index in a 15-entry table
    else UNK 15; charge bucket; H bucket with an explicit ``4+`` bucket; bond
    class and stereo codes.  A test cross-checks a sample against that module.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    payload = load_cache(dataset)
    smiles = [str(s).strip() for s in payload["smiles"]]
    atom_ptr = [0]
    edge_ptr = [0]
    classes: list[int] = []
    charges: list[int] = []
    hydrogens: list[int] = []
    edge_u: list[int] = []
    edge_v: list[int] = []
    edge_class: list[int] = []
    edge_stereo: list[int] = []
    n_components = np.zeros(len(smiles), dtype=np.int16)
    failures: list[int] = []
    index_of = {pair: k for k, pair in enumerate(LIPO_RAW_ATOM_CLASSES)}
    for row, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            failures.append(row)
            atom_ptr.append(atom_ptr[-1])
            edge_ptr.append(edge_ptr[-1])
            n_components[row] = 0
            continue
        n_atoms = int(mol.GetNumAtoms())
        for atom in mol.GetAtoms():
            classes.append(index_of.get((atom.GetSymbol(), bool(atom.GetIsAromatic())), LIPO_RAW_ATOM_UNK))
            charges.append(_lipo_charge_code(int(atom.GetFormalCharge())))
            hydrogens.append(_lipo_h_code(int(atom.GetTotalNumHs())))
        for bond in mol.GetBonds():
            i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
            bond_type = str(bond.GetBondType())
            code = LIPO_RAW_BOND_CLASSES.index(bond_type) if bond_type in LIPO_RAW_BOND_CLASSES else LIPO_RAW_BOND_UNK
            stereo = _lipo_stereo_code(bond.GetStereo())
            for a, b in ((i, j), (j, i)):
                edge_u.append(a)
                edge_v.append(b)
                edge_class.append(code)
                edge_stereo.append(stereo)
        n_components[row] = int(len(Chem.GetMolFrags(mol)))
        atom_ptr.append(atom_ptr[-1] + n_atoms)
        edge_ptr.append(edge_ptr[-1] + 2 * int(mol.GetNumBonds()))
    return {
        "dataset": dataset,
        "version": LIPO_RAW_VERSION,
        "atom_ptr": np.asarray(atom_ptr, dtype=np.int64),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "atom_class": np.asarray(classes, dtype=np.uint8),
        "atom_charge": np.asarray(charges, dtype=np.uint8),
        "atom_h": np.asarray(hydrogens, dtype=np.uint8),
        "edge_u": np.asarray(edge_u, dtype=np.int32),
        "edge_v": np.asarray(edge_v, dtype=np.int32),
        "edge_class": np.asarray(edge_class, dtype=np.uint8),
        "edge_stereo": np.asarray(edge_stereo, dtype=np.uint8),
        "n_components": n_components,
        "smiles": np.asarray(smiles, dtype=str),
        "parse_failed": np.asarray(failures, dtype=np.int64),
    }


def lipo_raw_path(dataset: str, *, base: Path | None = None) -> Path:
    root = C.DATA_DIR if base is None else Path(base)
    return root / f"{dataset.replace('-', '_')}_{LIPO_RAW_VERSION}.npz"


def save_lipo_raw(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in payload.items()})
    return {"path": str(path), "sha256": C.sha256_file(path), "bytes": int(path.stat().st_size)}


def load_lipo_raw(dataset: str, *, base: Path | None = None) -> dict[str, Any]:
    path = lipo_raw_path(dataset, base=base)
    if not path.is_file():
        raise FileNotFoundError(f"lipo raw attribute cache missing for {dataset}: {path}")
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["smiles"] = [str(s) for s in payload["smiles"].tolist()]
    payload["cache_sha256"] = C.sha256_file(path)
    return payload


def lipo_raw_row(payload: Mapping[str, Any], row: int) -> dict[str, Any]:
    """One row in the schema expected by ``adapters.lipo_row_to_graph``."""
    ap = np.asarray(payload["atom_ptr"], dtype=np.int64)
    ep = np.asarray(payload["edge_ptr"], dtype=np.int64)
    a0, a1 = int(ap[row]), int(ap[row + 1])
    e0, e1 = int(ep[row]), int(ep[row + 1])
    return {
        "smiles": payload["smiles"][row],
        "atom_class": np.asarray(payload["atom_class"])[a0:a1].astype(np.int64),
        "atom_charge": np.asarray(payload["atom_charge"])[a0:a1].astype(np.int64),
        "atom_h": np.asarray(payload["atom_h"])[a0:a1].astype(np.int64),
        "edge_index": np.stack(
            [
                np.asarray(payload["edge_u"])[e0:e1].astype(np.int64),
                np.asarray(payload["edge_v"])[e0:e1].astype(np.int64),
            ]
        ),
        "edge_class": np.asarray(payload["edge_class"])[e0:e1].astype(np.int64),
        "edge_stereo": np.asarray(payload["edge_stereo"])[e0:e1].astype(np.int64),
        "n_components": int(np.asarray(payload["n_components"])[row]),
    }
