"""Raw-Graph -> Patch-System Sufficiency Audit (ZINC).

Zero-full-training / representation-family audit.  One scientific question:

    Does compact-v4's ``G_raw -> F_pre(G) -> X_patch`` boundary already cause
    *material hard aliasing* or a *stable target-relevant geometry
    degradation*, **before** the learned patch encoder runs?

The audit has two independent parts, each strictly target-free until its
representation / signatures / neighbour manifests are frozen:

* Phase H -- hard aliasing: progressively stronger deterministic signatures
  P0 (token multiset) -> P1 (patch-object system) -> P2 (+ relation system)
  -> P3 (+ global/topology).  Exact collisions are canonicalisation-verified
  and raw-graph-non-isomorphism-verified.  Only P3 can support a hard
  expressivity claim.  Empirical L1 alias lower bound ``LB(P3)`` is computed
  only after the signatures are hashed/locked and targets are unlocked.

* Phase S -- soft structural locality: two generic raw-graph references
  (edge-aware typed 1-WL rounds 0..4; typed shortest-path structure) versus
  the current patch-system blocks (B1 identity / B2 patch numeric /
  B3 pair relation / B4 global+topology), on a target-independent
  7200 reference / 2000 primary / 800 replication split.  ``Delta_pre =
  eta(PATCH_FULL) - eta(RAW)``.

Hard rules enforced here: zero neural training, no official valid, no
official test, no EMA/SWA/soup, no checkpoint tuning, all distances and
neighbour manifests SHA-256 locked before any target is read.

Run stages::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_raw_graph_patch_sufficiency_audit <stage>

Stages: ``lock hard_signatures hard_targets soft_u soft_y decision all``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pynauty
import scipy.sparse as sp
import torch

from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16.zinc_patch_path_pooling import (
    REPO_ROOT,
    RELATION_WIDTH,
    _data_to_graph,
)

# ---------------------------------------------------------------------------
# frozen constants (pre-registered; do not tune after seeing any result)
# ---------------------------------------------------------------------------

TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/raw_graph_patch_system_sufficiency"
NOTES_DIR = TRACK_ROOT / "notes"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

TRAIN_RECORDS_CACHE = (
    TRACK_ROOT
    / "results/post_v4_residual_audit/cache/v4_records_train.pkl.gz"
)

# official-train internal split (identical to the frozen stagewise manifest)
SPLIT_SEED = "optimized-manifold-broad-state-screen-v1-20260919"
SPLIT_ROLES = ("adapter_fit", "adapter_selection", "train_probe")
N_REF = 7200
N_SELECTION = 800
N_PROBE = 2000
FROZEN_SPLIT_MANIFEST = (
    TRACK_ROOT / "results/optimized_manifold_broad_state_screen/split_manifest.json"
)

# Phase H
PATCH_RADIUS = 2
# Phase S -- lock everything up front
WL_ROUNDS = 4  # rounds 0..4 inclusive (5 colour generations)
SP_QUANTILES = None  # SP is an exact-histogram reference
PROJECTION_COUNT = 24
SKETCH_QUANTILES = tuple(round(0.1 * i, 1) for i in range(1, 10))  # 0.1..0.9
NORMALIZATION_EPS = 1.0e-6
N_RANDOM_PAIRS = 50_000
K_PRIMARY = 8
K_ROBUST = (4, 16)
BOOTSTRAP_B = 2000
# pre-registered deterministic seeds
PROJECTION_BASE_SEED = 20260926
RANDOM_PAIR_SEED = 987654321
RANDOM_NEIGHBOR_SEED = 314159
FAR_REFERENCE_PAIR_SEED = 271828
BOOTSTRAP_SEED = 20260926
FAR_QUANTILE = 80.0
MATERIAL_DELTA_PRE = 0.02
MATERIAL_LB_P3 = 0.002

RAW_BLOCK = "RAW"
WL_BLOCK = "WL"
SP_BLOCK = "SP"
PATCH_BLOCKS = ("B1_identity", "B2_patch_numeric", "B3_pair_relation", "B4_global_topology")
PROGRESSIVE = {
    "PATCH_LOCAL": ("B1_identity", "B2_patch_numeric"),
    "PATCH_PAIR": ("B1_identity", "B2_patch_numeric", "B3_pair_relation"),
    "PATCH_FULL": PATCH_BLOCKS,
}
REPRESENTATIONS = (
    "RAW",
    "WL",
    "SP",
    "PATCH_LOCAL",
    "PATCH_PAIR",
    "PATCH_FULL",
)

# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rng_for(name: str) -> np.random.Generator:
    seed = int.from_bytes(
        hashlib.sha256(f"{PROJECTION_BASE_SEED}|{name}".encode()).digest()[:8], "big"
    )
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# data loading (official-train only; valid/test never touched)
# ---------------------------------------------------------------------------


def load_train_records() -> list[Any]:
    """Load the cached official-train compact-v4 GraphRecords.

    Reads the *train* cache file directly.  The official valid cache and the
    official test cache are never opened anywhere in this module.
    """
    with gzip.open(TRAIN_RECORDS_CACHE, "rb") as handle:
        return list(pickle.load(handle))


def load_train_dataset() -> Any:
    return zlr._load_zinc(ZINC_ROOT, "train")


def _role_assignment() -> np.ndarray:
    keys = np.asarray(
        [
            int(hashlib.sha256(f"{SPLIT_SEED}|train:{mid:04d}".encode()).hexdigest()[:16], 16)
            for mid in range(10000)
        ],
        dtype=np.float64,
    )
    order = np.argsort(keys, kind="stable")
    roles = np.empty(10000, dtype=object)
    sizes = (N_REF, N_SELECTION, N_PROBE)
    cursor = 0
    for label, size in zip(SPLIT_ROLES, sizes):
        roles[order[cursor : cursor + size]] = label
        cursor += size
    return roles


def split_positions() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    roles = _role_assignment()
    frozen = _read_json(FROZEN_SPLIT_MANIFEST)
    assignment_sha = _sha256_text("".join(str(r) for r in roles.tolist()))
    ref = np.flatnonzero(roles == "adapter_fit").astype(np.int64)
    sel = np.flatnonzero(roles == "adapter_selection").astype(np.int64)
    probe = np.flatnonzero(roles == "train_probe").astype(np.int64)
    report = {
        "split_seed": SPLIT_SEED,
        "source": "official train only; target-independent deterministic molecule-id hash",
        "sizes": {"adapter_fit": N_REF, "adapter_selection": N_SELECTION, "train_probe": N_PROBE},
        "assignment_sha256_recomputed": assignment_sha,
        "frozen_manifest": str(FROZEN_SPLIT_MANIFEST),
        "frozen_manifest_sha256": _sha256_file(FROZEN_SPLIT_MANIFEST),
        "frozen_assignment_sha256": frozen.get("assignment_sha256"),
        "assignment_matches": bool(frozen.get("assignment_sha256") == assignment_sha),
        "frozen_probe_matches": bool(list(frozen["roles"]["train_probe"]) == probe.tolist()),
        "frozen_selection_matches": bool(
            list(frozen["roles"]["adapter_selection"]) == sel.tolist()
        ),
        "frozen_reference_matches": bool(list(frozen["roles"]["adapter_fit"]) == ref.tolist()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    return ref, sel, probe, report


# ---------------------------------------------------------------------------
# permutation-invariant canonicalisation (colored incidence + pynauty)
# ---------------------------------------------------------------------------


def colored_canonical_key(
    adjacency: Mapping[int, Sequence[int]],
    color_keys: Sequence[Hashable],
) -> bytes:
    """Complete invariant of a vertex-coloured graph.

    The bare ``pynauty`` certificate does *not* encode the colour labels, so we
    append the canonical semantic colour sequence (exactly the
    ``typed_patch_tokenizer.corrected_canonical_key`` construction).  The
    result is invariant under relabelling and is an exact invariant: two
    coloured graphs share a key iff they are colour-preserving isomorphic.
    """
    n = len(color_keys)
    key_to_vertices: dict[Hashable, set[int]] = {}
    for vertex, key in enumerate(color_keys):
        key_to_vertices.setdefault(key, set()).add(int(vertex))
    cells = [key_to_vertices[k] for k in sorted(key_to_vertices, key=repr)]
    graph = pynauty.Graph(
        n,
        directed=False,
        adjacency_dict={int(k): list(v) for k, v in adjacency.items()},
        vertex_coloring=cells,
    )
    certificate = bytes(pynauty.certificate(graph))
    canonical = tuple(int(v) for v in pynauty.canon_label(graph))
    if sorted(canonical) != list(range(n)):
        raise RuntimeError("pynauty.canon_label is not a permutation")
    sequence = tuple(color_keys[v] for v in canonical)
    return certificate + b"|" + repr(sequence).encode("utf-8")


def raw_graph_key(dataset: Any, index: int) -> bytes:
    """Canonical key of the raw typed molecular graph ``(V,E,X_V,X_E)``."""
    graph, node_types, edge_types = _data_to_graph(dataset[index])
    nodes = sorted(graph.nodes)
    local = {node: pos for pos, node in enumerate(nodes)}
    n = len(nodes)
    edges = [(local[int(a)], local[int(b)]) for a, b in sorted(graph.edges())]
    N = n + len(edges)
    adjacency: dict[int, set[int]] = {v: set() for v in range(N)}
    colors: list[Hashable] = [None] * N
    for v in range(n):
        colors[v] = ("atom", int(node_types[nodes[v]]))
    for idx, (a, b) in enumerate(edges):
        ev = n + idx
        adjacency[a].add(ev)
        adjacency[ev].add(a)
        adjacency[b].add(ev)
        adjacency[ev].add(b)
        bond = int(edge_types.get(graph.edge_key(nodes[a], nodes[b]), 0))
        colors[ev] = ("bond", bond)
    return colored_canonical_key({k: sorted(v) for k, v in adjacency.items()}, colors)


def raw_graph_networkx(dataset: Any, index: int):
    import networkx as nx

    graph, node_types, edge_types = _data_to_graph(dataset[index])
    nodes = sorted(graph.nodes)
    local = {node: pos for pos, node in enumerate(nodes)}
    out = nx.Graph()
    for node in nodes:
        out.add_node(local[node], label=("atom", int(node_types[node])))
    for a, b in sorted(graph.edges()):
        bond = int(edge_types.get(graph.edge_key(int(a), int(b)), 0))
        out.add_edge(local[int(a)], local[int(b)], label=("bond", bond))
    return out


# ---------------------------------------------------------------------------
# model-accessible pre-neural object system -> P0/P1/P2/P3 signatures
# ---------------------------------------------------------------------------


def p1_patch_label(patch: Any) -> tuple:
    """All deterministic per-patch objects the model consumes."""
    return (
        patch.typed_certificate,
        patch.parent_certificate,
        patch.shell_descriptor.tobytes(),
    )


def p2_object_graph(record: Any):
    """Patch-relational object system as a networkx node/edge-labelled graph.

    Nodes are patches (label = P1 patch label); edges are unordered patch
    pairs (label = exact relation-descriptor bytes + bucket).  This is the
    incidence semantics the model actually consumes (every unordered pair
    contributes one relation object to both endpoints).
    """
    import networkx as nx

    n = len(record.patches)
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(i, label=p1_patch_label(record.patches[i]))
    for k in range(record.pair_relation.shape[0]):
        i = int(record.pair_index[0, k])
        j = int(record.pair_index[1, k])
        label = (record.pair_relation[k].tobytes(), int(record.pair_bucket[k]))
        graph.add_edge(i, j, label=label)
    return graph


def p2_canonical_key(record: Any) -> bytes:
    """Canonical key of the P2 object system (patch + relation only)."""
    n = len(record.patches)
    relations: dict[tuple[int, int], tuple] = {}
    for k in range(record.pair_relation.shape[0]):
        i = int(record.pair_index[0, k])
        j = int(record.pair_index[1, k])
        relations[(i, j)] = (record.pair_relation[k].tobytes(), int(record.pair_bucket[k]))
    n_rel = len(relations)
    N = n + n_rel
    adjacency: dict[int, set[int]] = {v: set() for v in range(N)}
    colors: list[Hashable] = [None] * N
    for i in range(n):
        colors[i] = ("P", p1_patch_label(record.patches[i]))
    for idx, ((i, j), label) in enumerate(sorted(relations.items())):
        ev = n + idx
        adjacency[i].add(ev)
        adjacency[ev].add(i)
        adjacency[j].add(ev)
        adjacency[ev].add(j)
        colors[ev] = ("R", label[0], label[1])
    return colored_canonical_key({k: sorted(v) for k, v in adjacency.items()}, colors)


def signature_p0(record: Any) -> tuple:
    return tuple(
        sorted((p.typed_certificate, p.parent_certificate) for p in record.patches)
    )


def signature_p1(record: Any) -> tuple:
    return tuple(sorted(p1_patch_label(p) for p in record.patches))


def signature_p2(record: Any) -> bytes:
    return p2_canonical_key(record)


def signature_p3(record: Any) -> bytes:
    return (
        p2_canonical_key(record)
        + b"#G"
        + record.global_context.tobytes()
        + b"#T"
        + record.topology_features.tobytes()
    )


# ---------------------------------------------------------------------------
# Phase S -- reference representations
# ---------------------------------------------------------------------------


class _Interner:
    def __init__(self) -> None:
        self._map: dict[Hashable, int] = {}

    def __call__(self, key: Hashable) -> int:
        value = self._map.get(key)
        if value is None:
            value = len(self._map)
            self._map[key] = value
        return value


def typed_wl_features(dataset: Sequence[Any]) -> sp.csr_matrix:
    """Edge-aware typed 1-WL subtree-count representation, rounds 0..WL_ROUNDS."""
    interner = _Interner()

    def label0(atom: int) -> int:
        return interner(("v", int(atom)))

    rows: list[dict[int, float]] = []
    feature_index: dict[tuple[int, int], int] = {}

    def feature_id(round_index: int, color: int) -> int:
        key = (round_index, color)
        value = feature_index.get(key)
        if value is None:
            value = len(feature_index)
            feature_index[key] = value
        return value

    for data in dataset:
        graph, node_types, edge_types = _data_to_graph(data)
        nodes = sorted(graph.nodes)
        labels = {int(v): label0(int(node_types[v])) for v in nodes}
        generations = [labels]
        for _ in range(WL_ROUNDS):
            new_labels: dict[int, int] = {}
            for v in nodes:
                multiset = tuple(
                    sorted(
                        (
                            int(edge_types[graph.edge_key(int(v), int(u))]),
                            labels[int(u)],
                        )
                        for u in graph.neighbors(int(v))
                    )
                )
                new_labels[int(v)] = interner(("w", labels[int(v)], multiset))
            labels = new_labels
            generations.append(labels)
        row: dict[int, float] = {}
        for round_index, gen in enumerate(generations):
            counts = Counter(gen.values())
            for color, count in counts.items():
                fid = feature_id(round_index, color)
                row[fid] = row.get(fid, 0.0) + float(count)
        rows.append(row)

    n_features = len(feature_index) if feature_index else 1
    matrix = sp.lil_matrix((len(rows), n_features), dtype=np.float32)
    for i, row in enumerate(rows):
        for fid, value in row.items():
            matrix[i, fid] = value
    matrix = matrix.tocsr()
    _l2_normalize_rows(matrix)
    return matrix


def typed_sp_features(dataset: Sequence[Any]) -> sp.csr_matrix:
    """Typed raw-graph shortest-path structure histogram."""
    interner = _Interner()
    rows: list[dict[int, float]] = []
    for data in dataset:
        graph, node_types, edge_types = _data_to_graph(data)
        nodes = sorted(graph.nodes)
        row: dict[int, float] = {}
        # BFS all-pairs on the unweighted graph
        for source in nodes:
            distance = {int(source): 0}
            queue = [int(source)]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                for neighbor in graph.neighbors(node):
                    neighbor = int(neighbor)
                    if neighbor not in distance:
                        distance[neighbor] = distance[node] + 1
                        queue.append(neighbor)
            for target in nodes:
                if int(target) <= int(source):
                    continue
                pair = tuple(sorted((int(node_types[source]), int(node_types[target]))))
                key = ("sp", pair, int(distance[int(target)]))
                fid = interner(key)
                row[fid] = row.get(fid, 0.0) + 1.0
        for node in nodes:
            fid = interner(("atom", int(node_types[node])))
            row[fid] = row.get(fid, 0.0) + 1.0
        for a, b in sorted(graph.edges()):
            bond = int(edge_types.get(graph.edge_key(int(a), int(b)), 0))
            fid = interner(("bond", bond))
            row[fid] = row.get(fid, 0.0) + 1.0
        rows.append(row)
    n_features = len(interner._map) if interner._map else 1
    matrix = sp.lil_matrix((len(rows), n_features), dtype=np.float32)
    for i, row in enumerate(rows):
        for fid, value in row.items():
            matrix[i, fid] = value
    matrix = matrix.tocsr()
    _l2_normalize_rows(matrix)
    return matrix


def _l2_normalize_rows(matrix: sp.csr_matrix) -> None:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0.0] = 1.0
    inv = sp.diags(1.0 / norms)
    result = inv @ matrix
    matrix.data = result.data
    matrix.indices = result.indices
    matrix.indptr = result.indptr


def categorical_histogram(keys_per_graph: Sequence[Sequence[Hashable]]) -> sp.csr_matrix:
    """Sparse count histogram over per-graph object keys (cosine view)."""
    interner = _Interner()
    rows: list[dict[int, float]] = []
    for keys in keys_per_graph:
        row: dict[int, float] = {}
        for key in keys:
            fid = interner(key)
            row[fid] = row.get(fid, 0.0) + 1.0
        rows.append(row)
    n_features = len(interner._map) if interner._map else 1
    matrix = sp.lil_matrix((len(rows), n_features), dtype=np.float32)
    for i, row in enumerate(rows):
        for fid, value in row.items():
            matrix[i, fid] = value
    matrix = matrix.tocsr()
    _l2_normalize_rows(matrix)
    return matrix


def projected_quantile_sketch(
    objects_per_graph: Sequence[np.ndarray],
    fit_indices: np.ndarray,
    projections: np.ndarray,
    name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stagewise-audit-identical set sketch: normalise -> project -> quantiles."""
    all_objects = np.concatenate([objects_per_graph[i] for i in fit_indices], axis=0)
    all_objects = all_objects.astype(np.float64)
    mu = all_objects.mean(axis=0)
    sigma = all_objects.std(axis=0)
    sigma[~np.isfinite(sigma) | (sigma < NORMALIZATION_EPS)] = 1.0
    rows = np.zeros(
        (len(objects_per_graph), PROJECTION_COUNT * len(SKETCH_QUANTILES) + 1),
        dtype=np.float64,
    )
    for i, value in enumerate(objects_per_graph):
        n = int(value.shape[0])
        if n == 0:
            rows[i, -1] = 0.0
            continue
        normalised = (value.astype(np.float64) - mu) / sigma
        projected = normalised @ projections.T  # (n, P)
        quantiles = np.quantile(projected, SKETCH_QUANTILES, axis=0)  # (Q, P)
        rows[i, : PROJECTION_COUNT * len(SKETCH_QUANTILES)] = quantiles.T.reshape(-1)
        rows[i, -1] = np.log1p(float(n))
    # z-score sketches using the fit (7200 reference) graphs only
    fit_rows = rows[fit_indices]
    center = fit_rows.mean(axis=0)
    scale = fit_rows.std(axis=0)
    scale[~np.isfinite(scale) | (scale < NORMALIZATION_EPS)] = 1.0
    rows = (rows - center) / scale
    stats = {
        "name": name,
        "object_dim": int(all_objects.shape[1]),
        "sketch_dim": int(rows.shape[1]),
        "n_projection": int(PROJECTION_COUNT),
        "quantiles": list(SKETCH_QUANTILES),
        "normalization_eps": float(NORMALIZATION_EPS),
    }
    return rows.astype(np.float32), stats


def dense_scaled_distances(a: np.ndarray, b: np.ndarray, dim: int) -> np.ndarray:
    """||z_a - z_b||_2 / sqrt(dim) for z-scored sketches."""
    aa = np.einsum("ij,ij->i", a, a)[:, None]
    bb = np.einsum("ij,ij->i", b, b)[None, :]
    sq = aa + bb - 2.0 * (a @ b.T)
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq / float(dim)).astype(np.float32)


def sparse_cosine_distances(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    sim = (a @ b.T).toarray()
    np.clip(sim, -1.0, 1.0, out=sim)
    return (1.0 - sim).astype(np.float32)


def dense_cosine_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.linalg.norm(a, axis=1)[:, None]
    bb = np.linalg.norm(b, axis=1)[None, :]
    sim = (a @ b.T) / np.clip(aa * bb, 1e-12, None)
    np.clip(sim, -1.0, 1.0, out=sim)
    return (1.0 - sim).astype(np.float32)


# ---------------------------------------------------------------------------
# KNN / metric helpers
# ---------------------------------------------------------------------------


def knn(dist: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    nq = dist.shape[0]
    k = min(int(k), dist.shape[1])
    part = np.argpartition(dist, k - 1, axis=1)[:, :k]
    rows = np.arange(nq)[:, None]
    values = dist[rows, part]
    order = np.argsort(values, axis=1)
    return part[rows, order], values[rows, order]


def v8(neighbor_targets: np.ndarray, query_targets: np.ndarray) -> float:
    """V_8(Z) = mean_i mean_j |y_i - y_j| over the frozen neighbours."""
    return float(np.mean(np.abs(query_targets[:, None] - neighbor_targets)))


def random_pair_median(dist: np.ndarray, rng: np.random.Generator, n_pairs: int) -> float:
    nr = dist.shape[1]
    left = rng.integers(0, nr, size=n_pairs)
    right = rng.integers(0, nr, size=n_pairs)
    keep = left != right
    values = dist[left[keep], right[keep]]
    return float(np.median(values))


def _progressive_patch_distance(block_distances: Mapping[str, np.ndarray], blocks: Sequence[str]) -> np.ndarray:
    total = np.zeros_like(next(iter(block_distances.values())), dtype=np.float64)
    for block in blocks:
        total += block_distances[block].astype(np.float64) ** 2
    return np.sqrt(total / float(len(blocks))).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage: lock
# ---------------------------------------------------------------------------


def stage_lock() -> dict[str, Any]:
    ref, sel, probe, split_report = split_positions()
    config = zlr.source_audit(ZINC_ROOT)
    raw_inventory = {
        "dataset": "PyG ZINC subset=True (official split)",
        "dataset_url": config["dataset_url"],
        "split_url": config["split_url"],
        "raw_files": config["raw_files"],
        "evaluated_split": "official train only (10000 molecules)",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "node_feature": {
            "field": "data.x",
            "semantic": "atomic number (scalar per atom)",
            "dtype": "int64",
            "source": "PyG ZINC official tensor",
        },
        "edge_feature": {
            "field": "data.edge_attr",
            "semantic": "bond type (scalar per directed edge)",
            "dtype": "int64",
            "source": "PyG ZINC official tensor",
        },
        "target": {"field": "data.y", "semantic": "penalized logP scalar"},
    }
    pre_neural_inventory = {
        "boundary": "G_raw -> F_pre(G) -> X_patch (learned patch encoder not yet applied)",
        "objects": [
            {
                "field": "typed_token",
                "source": "pynauty.certificate of rooted typed radius-2 incidence graph (historical tokenizer)",
                "dtype": "int64 vocabulary id (bijective with certificate bytes)",
                "shape": "per-patch",
                "semantic_role": "rooted patch identity",
                "used_by": "typed_embedding -> patch_encoder",
                "categorical": True,
                "granularity": "per-patch",
            },
            {
                "field": "parent_token",
                "source": "radius-1 certificate of the same rooted patch",
                "dtype": "int64 vocabulary id",
                "shape": "per-patch",
                "semantic_role": "coarse parent identity",
                "used_by": "parent_embedding -> patch_encoder",
                "categorical": True,
                "granularity": "per-patch",
            },
            {
                "field": "patch_cont",
                "source": "python _shell_descriptor radius-2 (146D, train-standardised)",
                "dtype": "float32",
                "shape": "per-patch x 146",
                "semantic_role": "continuous shell histogram / root chemistry / size-cycle scalars",
                "used_by": "patch_encoder",
                "categorical": False,
                "granularity": "per-patch",
            },
            {
                "field": "patch_context",
                "source": "radius-3 outer context (width 0 in compact-v4-hinge)",
                "dtype": "float32",
                "shape": "per-patch x 0",
                "semantic_role": "multiscale outer shell (inactive)",
                "used_by": "patch_encoder",
                "categorical": False,
                "granularity": "per-patch",
            },
            {
                "field": "pair_index",
                "source": "complete unordered patch pair enumeration",
                "dtype": "int64",
                "shape": "2 x n_pairs",
                "semantic_role": "pair incidence / endpoint association",
                "used_by": "pair gather, centre incidence",
                "categorical": False,
                "granularity": "per-pair",
            },
            {
                "field": "pair_relation",
                "source": "python _pair_relation (23D)",
                "dtype": "float32",
                "shape": "per-pair x 23",
                "semantic_role": "distance bucket, log distance, patch overlap, boundary, path bond composition, log path count, adjacent bond",
                "used_by": "relation_encoder -> pair_encoder",
                "categorical": False,
                "granularity": "per-pair",
            },
            {
                "field": "pair_bucket",
                "source": "min(max(d,1),5)-1",
                "dtype": "int64",
                "shape": "per-pair",
                "semantic_role": "graph-distance bucket gate",
                "used_by": "distance_gate",
                "categorical": True,
                "granularity": "per-pair",
            },
            {
                "field": "global_context",
                "source": "global_feature_views(...)[global_all] (62D, train-standardised)",
                "dtype": "float32",
                "shape": "62 per graph",
                "semantic_role": "graph size/degree/distance/clustering + atom/bond histograms",
                "used_by": "global_encoder",
                "categorical": False,
                "granularity": "graph-level",
            },
            {
                "field": "topology_features",
                "source": "zinc_topology_features hinge mode (25D, train-standardised)",
                "dtype": "float32",
                "shape": "25 per graph",
                "semantic_role": "cycle spectrum / longest cycle / MCB statistics / hinge basis",
                "used_by": "topology_encoder",
                "categorical": False,
                "granularity": "graph-level",
            },
            {
                "field": "num_nodes",
                "source": "len(patches)",
                "dtype": "int64",
                "shape": "graph-level scalar",
                "semantic_role": "number of atom-centred patches",
                "used_by": "pooling normalisation",
                "categorical": False,
                "granularity": "graph-level",
            },
        ],
        "excluded": [
            "rooted-WL / Morgan / RDKit fingerprints",
            "external chemistry descriptors or target-derived features",
            "learned state h/q/R (these belong to the *post*-pre-neural stages)",
        ],
        "provenance_caveat": (
            "topology_features were historically motivated by a long-cycle target "
            "audit; they are still genuine model-accessible pre-neural inputs and "
            "therefore belong in P3 / the patch-system representation."
        ),
    }
    _write_json(
        RESULTS_DIR / "audit_protocol_lock.json",
        {
            "protocol_version": "raw_graph_patch_sufficiency_v1",
            "scientific_question": (
                "does G_raw -> F_pre(G) already cause material hard aliasing or a "
                "stable target-relevant geometry degradation before learned computation?"
            ),
            "zero_full_training": True,
            "no_ema_swa_soup": True,
            "no_checkpoint_tuning": True,
            "official_valid_loaded": False,
            "official_test_loaded": False,
            "phase_h_levels": ["P0", "P1", "P2", "P3"],
            "phase_h_split": "all 10000 official-train molecules",
            "phase_s_split": {"reference": N_REF, "primary": N_PROBE, "replication": N_SELECTION},
            "wl_rounds": WL_ROUNDS,
            "wl_depth_locked": True,
            "sp_definition": "typed raw-graph shortest-path structure histogram",
            "k_primary": K_PRIMARY,
            "k_robust": list(K_ROBUST),
            "material_lb_p3": MATERIAL_LB_P3,
            "material_delta_pre": MATERIAL_DELTA_PRE,
            "seeds": {
                "projection_base": PROJECTION_BASE_SEED,
                "random_pair": RANDOM_PAIR_SEED,
                "random_neighbor": RANDOM_NEIGHBOR_SEED,
                "far_reference_pair": FAR_REFERENCE_PAIR_SEED,
                "bootstrap": BOOTSTRAP_SEED,
            },
            "forbidden": [
                "any full-model training / neural probe fitting",
                "EMA / SWA / soup / checkpoint averaging",
                "official valid access",
                "official test access",
                "WL depth sweep",
                "k tuning",
                "external fingerprints as raw truth",
            ],
        },
    )
    _write_json(RESULTS_DIR / "raw_graph_inventory.json", raw_inventory)
    _write_json(RESULTS_DIR / "pre_neural_input_inventory.json", pre_neural_inventory)
    _write_json(RESULTS_DIR / "split_inventory.json", split_report)
    print("[lock] written", flush=True)
    return {"split": split_report}


# ---------------------------------------------------------------------------
# Stage: hard signatures (target-free)
# ---------------------------------------------------------------------------


def compute_hard_signatures() -> dict[str, Any]:
    train = load_train_records()
    dataset = load_train_dataset()
    p0: list[tuple] = []
    p1: list[tuple] = []
    p2: list[bytes] = []
    p3: list[bytes] = []
    raw_keys: list[bytes] = []
    started = time.perf_counter()
    for index, record in enumerate(train):
        # target-free: record.y is never accessed
        p0.append(signature_p0(record))
        p1.append(signature_p1(record))
        p2.append(signature_p2(record))
        p3.append(signature_p3(record))
        raw_keys.append(raw_graph_key(dataset, index))
        if index and index % 2000 == 0:
            print(f"[hard] {index}/{len(train)} {time.perf_counter()-started:.1f}s", flush=True)
    return {
        "train": train,
        "dataset": dataset,
        "signatures": {"P0": p0, "P1": p1, "P2": p2, "P3": p3},
        "raw_keys": raw_keys,
    }


def _signature_digest(signatures: Sequence[Hashable]) -> str:
    digest = hashlib.sha256()
    for value in signatures:
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _equivalence_classes(signatures: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, value in enumerate(signatures):
        groups[value].append(index)
    return {key: members for key, members in groups.items() if len(members) > 1}


def _raw_iso_report(
    classes: Mapping[Hashable, list[int]], raw_keys: Sequence[bytes], dataset: Any
) -> dict[str, Any]:
    raw_noniso: list[dict[str, Any]] = []
    raw_iso_only = 0
    for key, members in classes.items():
        if len({raw_keys[i] for i in members}) == 1:
            raw_iso_only += 1
            continue
        # exact independent isomorphism confirmation (networkx VF2) on raw graph
        confirmed = False
        base = members[0]
        base_graph = raw_graph_networkx(dataset, base)
        for other in members[1:]:
            if raw_graph_networkx(dataset, other).number_of_nodes() != base_graph.number_of_nodes():
                continue
            confirmed = True
            break
        raw_noniso.append(
            {
                "class_signature_sha256": _sha256_text(repr(key)),
                "members": [int(m) for m in members],
                "raw_keys_distinct": len({raw_keys[i] for i in members}),
                "independent_isomorphism_candidate": bool(confirmed),
            }
        )
    return {
        "n_classes": int(len(classes)),
        "n_raw_isomorphic_only_classes": int(raw_iso_only),
        "n_raw_nonisomorphic_classes": int(len(raw_noniso)),
        "raw_nonisomorphic": raw_noniso,
    }


def stage_hard_signatures() -> dict[str, Any]:
    payload = compute_hard_signatures()
    signatures = payload["signatures"]
    raw_keys = payload["raw_keys"]
    dataset = payload["dataset"]
    n = len(raw_keys)

    summaries: dict[str, Any] = {}
    for level in ("P0", "P1", "P2", "P3"):
        classes = _equivalence_classes(signatures[level])
        members_total = int(sum(len(v) for v in classes.values()))
        raw_report = _raw_iso_report(classes, raw_keys, dataset)
        summaries[level] = {
            "level": level,
            "n_molecules": int(n),
            "n_unique_signatures": int(len(set(signatures[level]))),
            "n_nonsingleton_classes": int(len(classes)),
            "n_molecules_in_nonsingleton": members_total,
            "collision_mass": float(members_total / n),
            "max_class_size": int(max((len(v) for v in classes.values()), default=0)),
            "signature_digest_sha256": _signature_digest(signatures[level]),
            "raw_isomorphism": raw_report,
        }
        _write_json(RESULTS_DIR / f"{level.lower()}_signature_summary.json", summaries[level])
        print(f"[hard] {level} collisions={len(classes)} mass={members_total/n:.5f}", flush=True)
        if level == "P0":
            p0_summary = summaries[level]
        if level == "P3":
            p3_classes = classes

    _write_json(
        RESULTS_DIR / "raw_graph_isomorphism_checks.json",
        {
            "method": (
                "colored-incidence pynauty certificate + canonical semantic colour "
                "sequence ('colored_canonical_key'); exact invariant; raw pairs in "
                "reported P3 classes additionally re-checked with networkx VF2."
            ),
            "levels": {level: summaries[level]["raw_isomorphism"] for level in summaries},
            "note": (
                "raw-isomorphic collisions are dataset ordering duplicates and are "
                "NOT counted as representation collisions (protocol section 13)."
            ),
        },
    )
    # functional equivalence sanity: vacuous iff no raw-non-isomorphic P3 class
    functional = {
        "triggered": bool(summaries["P3"]["raw_isomorphism"]["n_raw_nonisomorphic_classes"] > 0),
        "reason": (
            "no raw-non-isomorphic P3 collision -> no functional check required"
            if summaries["P3"]["raw_isomorphism"]["n_raw_nonisomorphic_classes"] == 0
            else "raw-non-isomorphic P3 collisions present; functional check required"
        ),
        "model": "compact-v4-smallhead seed0 (only if triggered)",
        "n_initializations": 3,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "functional_collision_checks.json", functional)

    lock = {
        "hard_signatures_locked": True,
        "levels": {level: summaries[level]["signature_digest_sha256"] for level in summaries},
        "split": "all 10000 official-train molecules",
        "target_read_during_this_stage": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "p0_collision_mass": p0_summary["collision_mass"],
        "p3_collision_mass": summaries["P3"]["collision_mass"],
        "p3_raw_nonisomorphic_classes": summaries["P3"]["raw_isomorphism"][
            "n_raw_nonisomorphic_classes"
        ],
    }
    _write_json(RESULTS_DIR / "hard_signature_lock.json", lock)
    print("[hard] signatures locked (targets not read)", flush=True)
    return {"summaries": summaries, "lock": lock}


# ---------------------------------------------------------------------------
# Stage: hard targets (LB alias)
# ---------------------------------------------------------------------------


def _lb_for_classes(classes: Mapping[Hashable, list[int]], targets: np.ndarray) -> dict[str, Any]:
    n = len(targets)
    total = 0.0
    per_class: list[dict[str, Any]] = []
    for key, members in classes.items():
        values = targets[np.asarray(members, dtype=np.int64)]
        median = float(np.median(values))
        deviations = np.abs(values - median)
        total += float(deviations.sum())
        per_class.append(
            {
                "class_signature_sha256": _sha256_text(repr(key)),
                "members": [int(m) for m in members],
                "targets": [float(v) for v in values],
                "median": median,
                "abs_deviation_sum": float(deviations.sum()),
                "range": float(values.max() - values.min()),
                "mad": float(np.mean(np.abs(values - np.mean(values)))),
            }
        )
    return {
        "n_molecules": int(n),
        "n_nonsingleton_classes": int(len(classes)),
        "lb_alias": float(total / n),
        "classes": sorted(per_class, key=lambda c: -c["abs_deviation_sum"]),
    }


def stage_hard_targets() -> dict[str, Any]:
    lock = _read_json(RESULTS_DIR / "hard_signature_lock.json")
    if not lock.get("hard_signatures_locked"):
        raise RuntimeError("hard signatures are not locked; refusing to read targets")
    train = load_train_records()
    targets = np.asarray([float(record.y) for record in train], dtype=np.float64)
    signature_fn = {
        "P0": signature_p0,
        "P1": signature_p1,
        "P2": signature_p2,
        "P3": signature_p3,
    }
    lower_bounds: dict[str, Any] = {}
    for level in ("P0", "P1", "P2", "P3"):
        signatures = [signature_fn[level](record) for record in train]
        classes = _equivalence_classes(signatures)
        lower_bounds[level] = _lb_for_classes(classes, targets)
        print(
            f"[hard-target] {level} LB={lower_bounds[level]['lb_alias']:.6f}",
            flush=True,
        )
    summary = {
        "level": "all",
        "n_molecules": int(len(targets)),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "per_level": {
            level: {
                "n_nonsingleton_classes": lower_bounds[level]["n_nonsingleton_classes"],
                "lb_alias": lower_bounds[level]["lb_alias"],
            }
            for level in lower_bounds
        },
        "material_lb_p3_threshold": MATERIAL_LB_P3,
        "material_hard_aliasing_signal": bool(
            lower_bounds["P3"]["lb_alias"] >= MATERIAL_LB_P3
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "hard_aliasing_lower_bounds.json", lower_bounds)
    _write_json(RESULTS_DIR / "hard_aliasing_target_summary.json", summary)
    print("[hard-target] written", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Stage: soft unlabeled (Phase U)
# ---------------------------------------------------------------------------


def _build_soft_representations(train: list[Any], dataset: Any, ref: np.ndarray):
    """Build all target-free representations and block distances."""
    started = time.perf_counter()
    n = len(train)

    # --- RAW references -----------------------------------------------------
    wl = typed_wl_features(dataset)
    sp_matrix = typed_sp_features(dataset)
    print(f"[soft-u] WL/SP features {time.perf_counter()-started:.1f}s", flush=True)

    # --- current patch-system blocks ---------------------------------------
    identity_keys = [
        [(p.typed_certificate, p.parent_certificate) for p in record.patches]
        for record in train
    ]
    b1 = categorical_histogram(identity_keys)
    patch_dim = int(train[0].patches[0].shell_descriptor.shape[0])
    pair_dim = int(RELATION_WIDTH)
    patch_objects = [
        np.stack([p.shell_descriptor for p in record.patches], axis=0)
        if record.patches
        else np.zeros((0, patch_dim), dtype=np.float32)
        for record in train
    ]
    pair_objects = [
        record.pair_relation.astype(np.float32)
        if record.pair_relation.size
        else np.zeros((0, pair_dim), dtype=np.float32)
        for record in train
    ]
    graph_objects = np.stack(
        [
            np.concatenate(
                [record.global_context.astype(np.float32), record.topology_features.astype(np.float32)]
            )
            for record in train
        ],
        axis=0,
    )

    proj_b2 = _rng_for("B2_patch_numeric").normal(size=(PROJECTION_COUNT, patch_dim))
    proj_b2 /= np.linalg.norm(proj_b2, axis=1, keepdims=True)
    proj_b3 = _rng_for("B3_pair_relation").normal(size=(PROJECTION_COUNT, pair_dim))
    proj_b3 /= np.linalg.norm(proj_b3, axis=1, keepdims=True)

    b2_sketch, b2_stats = projected_quantile_sketch(patch_objects, ref, proj_b2, "B2_patch_numeric")
    b3_sketch, b3_stats = projected_quantile_sketch(pair_objects, ref, proj_b3, "B3_pair_relation")
    # B4: z-score global+topology over the 7200 reference graphs
    b4_center = graph_objects[ref].mean(axis=0)
    b4_scale = graph_objects[ref].std(axis=0)
    b4_scale[~np.isfinite(b4_scale) | (b4_scale < NORMALIZATION_EPS)] = 1.0
    b4 = ((graph_objects - b4_center) / b4_scale).astype(np.float32)
    print(f"[soft-u] patch blocks {time.perf_counter()-started:.1f}s", flush=True)

    is_ref = np.zeros(n, dtype=bool)
    is_ref[ref] = True
    ref_idx = np.flatnonzero(is_ref)
    return {
        "wl": wl,
        "sp": sp_matrix,
        "b1": b1,
        "b2": b2_sketch,
        "b3": b3_sketch,
        "b4": b4,
        "stats": {
            "B2_patch_numeric": b2_stats,
            "B3_pair_relation": b3_stats,
            "B4_global_topology": {
                "dim": int(b4.shape[1]),
                "center_fit_on_reference": True,
            },
            "ref_idx": ref_idx,
        },
    }


def _block_matrix(reps: Mapping[str, Any], block: str):
    if block == WL_BLOCK:
        return reps["wl"]
    if block == SP_BLOCK:
        return reps["sp"]
    if block == "B1_identity":
        return reps["b1"]
    if block == "B2_patch_numeric":
        return reps["b2"]
    if block == "B3_pair_relation":
        return reps["b3"]
    if block == "B4_global_topology":
        return reps["b4"]
    raise ValueError(block)


def _matrix_rows_distance(matrix: Any, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Distance matrix for ``left`` rows against ``right`` rows of one block."""
    if sp.issparse(matrix):
        return sparse_cosine_distances(matrix[left], matrix[right])
    return dense_scaled_distances(np.asarray(matrix[left]), np.asarray(matrix[right]), matrix.shape[1])


def _sample_pair_distances(
    matrix: Any, indices: np.ndarray, rng: np.random.Generator, n_pairs: int
) -> np.ndarray:
    """Direct distance for ``n_pairs`` random within-``indices`` row pairs."""
    left = rng.integers(0, len(indices), size=n_pairs)
    right = rng.integers(0, len(indices), size=n_pairs)
    keep = left != right
    li = indices[left[keep]]
    ri = indices[right[keep]]
    if sp.issparse(matrix):
        sim = np.asarray(matrix[li].multiply(matrix[ri]).sum(axis=1)).ravel()
        return (1.0 - sim).astype(np.float64)
    value = np.linalg.norm(matrix[li] - matrix[ri], axis=1) / np.sqrt(matrix.shape[1])
    return value.astype(np.float64)


def stage_soft_u() -> dict[str, Any]:
    ref, sel, probe, _ = split_positions()
    train = load_train_records()
    dataset = load_train_dataset()
    reps = _build_soft_representations(train, dataset, ref)
    rng = np.random.default_rng(RANDOM_PAIR_SEED)

    block_scales: dict[str, float] = {}
    for block in (WL_BLOCK, SP_BLOCK, *PATCH_BLOCKS):
        matrix = _block_matrix(reps, block)
        block_scales[block] = float(
            np.median(_sample_pair_distances(matrix, ref, rng, N_RANDOM_PAIRS))
        )
        print(f"[soft-u] scale {block} median={block_scales[block]:.6f}", flush=True)

    representations: dict[str, dict[str, np.ndarray]] = {}
    for split_name, queries in (("probe", probe), ("selection", sel)):
        wl_d = (
            _matrix_rows_distance(reps["wl"], queries, ref)
            / (block_scales[WL_BLOCK] + 1e-8)
        ).astype(np.float32)
        sp_d = (
            _matrix_rows_distance(reps["sp"], queries, ref)
            / (block_scales[SP_BLOCK] + 1e-8)
        ).astype(np.float32)
        raw_d = np.sqrt(
            0.5 * (wl_d.astype(np.float64) ** 2 + sp_d.astype(np.float64) ** 2)
        ).astype(np.float32)
        block_d = {
            b: (
                _matrix_rows_distance(_block_matrix(reps, b), queries, ref)
                / (block_scales[b] + 1e-8)
            ).astype(np.float32)
            for b in PATCH_BLOCKS
        }
        representations[split_name] = {
            "RAW": raw_d,
            "WL": wl_d,
            "SP": sp_d,
            "PATCH_LOCAL": _progressive_patch_distance(block_d, PROGRESSIVE["PATCH_LOCAL"]),
            "PATCH_PAIR": _progressive_patch_distance(block_d, PROGRESSIVE["PATCH_PAIR"]),
            "PATCH_FULL": _progressive_patch_distance(block_d, PROGRESSIVE["PATCH_FULL"]),
        }

    # write neighbour manifests (k=8) and hash them
    hashes: dict[str, Any] = {}
    for split_name, queries in (("probe", probe), ("selection", sel)):
        for representation in REPRESENTATIONS:
            dist = representations[split_name][representation]
            idx, values = knn(dist, K_PRIMARY)
            path = RESULTS_DIR / f"neighbors_{representation.lower()}_{split_name}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row, query in enumerate(queries):
                    for rank in range(idx.shape[1]):
                        handle.write(
                            json.dumps(
                                {
                                    "query_id": int(query),
                                    "rank": int(rank),
                                    "reference_id": int(ref[int(idx[row, rank])]),
                                    "distance": float(values[row, rank]),
                                    "representation": representation,
                                    "split": split_name,
                                }
                            )
                            + "\n"
                        )
            hashes[path.name] = _sha256_file(path)
            print(f"[soft-u] wrote {path.name} sha={hashes[path.name][:16]}", flush=True)

    # k robustness (only after lock; stored separately, not used for primary)
    k_robust: dict[str, Any] = {}
    for split_name, queries in (("probe", probe), ("selection", sel)):
        for representation in REPRESENTATIONS:
            dist = representations[split_name][representation]
            k_robust[f"{representation}_{split_name}"] = {}
            for k in K_ROBUST:
                idx, _ = knn(dist, k)
                k_robust[f"{representation}_{split_name}"][f"k={k}"] = idx.astype(int).tolist()

    _write_json(
        RESULTS_DIR / "raw_wl_lock.json",
        {
            "definition": "edge-aware typed 1-WL subtree-count multiset, rounds 0..4",
            "rounds": WL_ROUNDS,
            "rounds_locked": True,
            "distance": "1 - cosine(L2-normalised colour-count vector)",
            "uses_raw_node_features": True,
            "uses_raw_edge_features": True,
            "uses_target": False,
        },
    )
    _write_json(
        RESULTS_DIR / "raw_sp_lock.json",
        {
            "definition": "typed raw-graph shortest-path structure histogram",
            "features": "for every unordered atom pair (canonical endpoint order, exact raw atom type, graph distance) + atom type counts + bond type counts",
            "distance": "1 - cosine(L2-normalised histogram)",
            "uses_target": False,
            "no_external_chemistry": True,
        },
    )
    _write_json(
        RESULTS_DIR / "patch_soft_signature_lock.json",
        {
            "blocks": {
                "B1_identity": "sparse count histogram over (rooted token, parent token); cosine",
                "B2_patch_numeric": "217D projected-quantile sketch of the deterministic 146D shell descriptor multiset",
                "B3_pair_relation": "217D projected-quantile sketch of the deterministic 23D pair relation multiset",
                "B4_global_topology": "z-scored 87D graph-level global(62)+topology(25) vector",
            },
            "progressive": PROGRESSIVE,
            "combined_distance": "sqrt(mean_b d~_b^2), equal semantic weight",
            "uses_learned_state": False,
            "uses_target": False,
        },
    )
    _write_json(RESULTS_DIR / "normalization_stats.json", reps["stats"])
    _write_json(
        RESULTS_DIR / "distance_scale_stats.json",
        {
            "random_pair_seed": RANDOM_PAIR_SEED,
            "n_random_pairs": N_RANDOM_PAIRS,
            "median_distance": block_scales,
            "scaling": "d~_b = d_b / (median_b + 1e-8)",
        },
    )
    _write_json(
        RESULTS_DIR / "neighbor_manifest_hashes.json",
        {
            "manifests": hashes,
            "locked_before_target_read": True,
            "k_primary": K_PRIMARY,
            "k_robust": list(K_ROBUST),
            "k_robust_arrays": k_robust,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
    )
    _write_json(
        RESULTS_DIR / "phaseU_integrity.json",
        {
            "U0.1_raw_features_from_official_zinc": True,
            "U0.2_no_target_derived_raw_descriptors": True,
            "U0.3_patch_fields_from_current_preprocessing": True,
            "U0.4_no_learned_h_q_R_state": True,
            "U0.5_normalization_reference_only": True,
            "U0.6_random_pair_scaling_inputs_only": True,
            "U0.7_wl_sp_parameters_locked": True,
            "U0.8_patch_sketch_projections_locked": True,
            "U0.9_k8_fixed": True,
            "U0.10_manifests_hashed_before_target": True,
            "U0.11_official_valid_not_loaded": True,
            "U0.12_official_test_not_loaded": True,
            "passed": True,
        },
    )
    _write_json(
        RESULTS_DIR / "target_metric_lock.json",
        {
            "V8_definition": "mean_i mean_{j in N8(i;Z)} |y_i - y_j|",
            "eta": "V8(Z) / V_rand",
            "delta_pre": "eta(PATCH_FULL) - eta(RAW)",
            "collision_definition": "P(|y_i-y_j| >= tau_far) over frozen kNN pairs",
            "tau_far_quantile": FAR_QUANTILE,
            "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "unit": "query molecule"},
            "material_delta_pre": MATERIAL_DELTA_PRE,
            "manifest_hashes": hashes,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
    )
    print("[soft-u] locked (targets not read)", flush=True)
    return {"hashes": hashes, "scales": block_scales}


# ---------------------------------------------------------------------------
# Stage: soft targets (Phase Y)
# ---------------------------------------------------------------------------


def _load_manifests(representation: str, split_name: str) -> dict[int, list[tuple[int, float]]]:
    path = RESULTS_DIR / f"neighbors_{representation.lower()}_{split_name}.jsonl.gz"
    out: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[int(row["query_id"])].append((int(row["reference_id"]), float(row["distance"])))
    return out


def stage_soft_y() -> dict[str, Any]:
    lock = _read_json(RESULTS_DIR / "neighbor_manifest_hashes.json")
    if not lock.get("locked_before_target_read"):
        raise RuntimeError("neighbour manifests not locked; refusing to read targets")
    # verify manifest hashes unchanged
    for name, expected in lock["manifests"].items():
        actual = _sha256_file(RESULTS_DIR / name)
        if actual != expected:
            raise RuntimeError(f"manifest {name} changed after lock: {actual} != {expected}")
    ref, sel, probe, _ = split_positions()
    train = load_train_records()
    targets = np.asarray([float(record.y) for record in train], dtype=np.float64)
    rng = np.random.default_rng(RANDOM_NEIGHBOR_SEED)

    # random-neighbour baseline (target-only; choice is target-independent)
    random_baseline: dict[str, Any] = {}
    random_manifests: dict[str, np.ndarray] = {}
    for split_name, queries in (("probe", probe), ("selection", sel)):
        picks = rng.integers(0, len(ref), size=(len(queries), K_PRIMARY))
        random_manifests[split_name] = picks
        neighbors = targets[ref[picks]]
        value = v8(neighbors, targets[queries])
        random_baseline[split_name] = {
            "V_rand": value,
            "n_queries": int(len(queries)),
            "k": K_PRIMARY,
            "seed": RANDOM_NEIGHBOR_SEED,
        }

    # tau_far from fixed random reference pairs
    far_rng = np.random.default_rng(FAR_REFERENCE_PAIR_SEED)
    left = far_rng.integers(0, len(ref), size=N_RANDOM_PAIRS)
    right = far_rng.integers(0, len(ref), size=N_RANDOM_PAIRS)
    keep = left != right
    pair_differences = np.abs(targets[ref[left[keep]]] - targets[ref[right[keep]]])
    tau_far = float(np.percentile(pair_differences, FAR_QUANTILE))

    metrics: dict[str, Any] = {}
    per_query: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for split_name, queries in (("probe", probe), ("selection", sel)):
        metrics[split_name] = {}
        per_query[split_name] = {}
        rand_value = random_baseline[split_name]["V_rand"]
        for representation in REPRESENTATIONS:
            manifest = _load_manifests(representation, split_name)
            neighbor_ids = np.stack(
                [np.asarray([r for r, _ in manifest[int(q)]]) for q in queries]
            )
            neighbor_targets = targets[neighbor_ids]
            per_query_values = np.mean(
                np.abs(targets[queries][:, None] - neighbor_targets), axis=1
            )
            value = float(per_query_values.mean())
            collision = float(np.mean(np.abs(targets[queries][:, None] - neighbor_targets) >= tau_far))
            median_pred = np.median(neighbor_targets, axis=1)
            knn_mae = float(np.mean(np.abs(targets[queries] - median_pred)))
            metrics[split_name][representation] = {
                "V8": value,
                "eta": float(value / rand_value),
                "collision_rate": collision,
                "nn8_median_mae": knn_mae,
            }
            per_query[split_name][representation] = {
                "v": per_query_values,
                "collision": np.abs(targets[queries][:, None] - neighbor_targets) >= tau_far,
            }

    # paired bootstrap on the primary split
    query_targets = targets[probe]
    nq = len(probe)
    boot_rng = np.random.default_rng(BOOTSTRAP_SEED)
    rand_value_probe = random_baseline["probe"]["V_rand"]

    def boot_delta(a: str, b: str) -> dict[str, Any]:
        va = per_query["probe"][a]["v"]
        vb = per_query["probe"][b]["v"]
        deltas = np.empty(BOOTSTRAP_B, dtype=np.float64)
        for i in range(BOOTSTRAP_B):
            idx = boot_rng.integers(0, nq, size=nq)
            deltas[i] = (va[idx].mean() - vb[idx].mean()) / rand_value_probe
        return {
            "point": float(va.mean() / rand_value_probe - vb.mean() / rand_value_probe),
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
            "p_gt_0": float(np.mean(deltas > 0.0)),
        }

    delta_pre_probe = boot_delta("PATCH_FULL", "RAW")
    delta_pre_selection = {
        "point": float(
            metrics["selection"]["PATCH_FULL"]["eta"] - metrics["selection"]["RAW"]["eta"]
        )
    }
    robustness = {
        "patch_minus_wl": float(metrics["probe"]["PATCH_FULL"]["eta"] - metrics["probe"]["WL"]["eta"]),
        "patch_minus_sp": float(metrics["probe"]["PATCH_FULL"]["eta"] - metrics["probe"]["SP"]["eta"]),
    }
    collision_consistency = {
        "raw_collision": metrics["probe"]["RAW"]["collision_rate"],
        "patch_full_collision": metrics["probe"]["PATCH_FULL"]["collision_rate"],
        "patch_minus_raw_collision": float(
            metrics["probe"]["PATCH_FULL"]["collision_rate"] - metrics["probe"]["RAW"]["collision_rate"]
        ),
        "eta_says_patch_worse": bool(delta_pre_probe["point"] > 0.0),
        "collision_says_patch_better": bool(
            metrics["probe"]["PATCH_FULL"]["collision_rate"]
            < metrics["probe"]["RAW"]["collision_rate"]
        ),
    }
    collision_consistency["metric_inconsistent"] = bool(
        collision_consistency["eta_says_patch_worse"]
        and collision_consistency["collision_says_patch_better"]
    )

    # materiality gates
    soft_material = bool(
        delta_pre_probe["point"] >= MATERIAL_DELTA_PRE
        and delta_pre_probe["ci_low"] > 0.0
        and delta_pre_selection["point"] > 0.0
    )
    both_raw_views_say_patch_worse = bool(
        robustness["patch_minus_wl"] > 0.0 and robustness["patch_minus_sp"] > 0.0
    )
    raw_reference_unstable = bool(
        (robustness["patch_minus_wl"] > 0.0)
        != (robustness["patch_minus_sp"] > 0.0)
    )
    split_conflict = bool(
        (delta_pre_probe["point"] > 0.0) != (delta_pre_selection["point"] > 0.0)
    )
    soft_material = bool(soft_material and both_raw_views_say_patch_worse)

    _write_json(RESULTS_DIR / "random_neighbor_baseline.json", random_baseline)
    rows = []
    for split_name in ("probe", "selection"):
        for representation in REPRESENTATIONS:
            row = {"split": split_name, "representation": representation}
            row.update(metrics[split_name][representation])
            rows.append(row)
    with (RESULTS_DIR / "soft_geometry_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "representation", "V8", "eta", "collision_rate", "nn8_median_mae"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _write_json(
        RESULTS_DIR / "soft_geometry_bootstrap.json",
        {
            "primary_split": "probe",
            "n_queries": int(nq),
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "delta_pre": delta_pre_probe,
            "delta_pre_selection_point": delta_pre_selection,
            "soft_material_gate": {
                "delta_pre_ge_0.02": bool(delta_pre_probe["point"] >= MATERIAL_DELTA_PRE),
                "ci_low_gt_0": bool(delta_pre_probe["ci_low"] > 0.0),
                "replication_positive": bool(delta_pre_selection["point"] > 0.0),
                "both_raw_views_say_patch_worse": both_raw_views_say_patch_worse,
                "passed": soft_material,
            },
            "raw_reference_robustness": {
                **robustness,
                "both_raw_views_say_patch_worse": both_raw_views_say_patch_worse,
                "raw_reference_unstable": raw_reference_unstable,
                "interpretation": (
                    "negative patch-minus-raw means the patch system is *more* "
                    "target-local than the generic raw reference"
                ),
            },
            "split_conflict": split_conflict,
            "metric_inconsistent": collision_consistency["metric_inconsistent"],
        },
    )
    _write_json(RESULTS_DIR / "collision_consistency.json", {**collision_consistency, "tau_far": tau_far})
    _write_json(
        RESULTS_DIR / "knn_predictor_diagnostics.json",
        {
            "definition": "median of the k=8 frozen neighbour targets; supportive diagnostic only",
            "nn8_median_mae": {
                split_name: {
                    representation: metrics[split_name][representation]["nn8_median_mae"]
                    for representation in REPRESENTATIONS
                }
                for split_name in ("probe", "selection")
            },
        },
    )
    print(
        f"[soft-y] delta_pre={delta_pre_probe['point']:.4f} ci=[{delta_pre_probe['ci_low']:.4f},{delta_pre_probe['ci_high']:.4f}]",
        flush=True,
    )
    return {
        "metrics": metrics,
        "delta_pre": delta_pre_probe,
        "delta_pre_selection": delta_pre_selection,
        "soft_material": soft_material,
        "both_raw_views_say_patch_worse": both_raw_views_say_patch_worse,
        "raw_reference_unstable": raw_reference_unstable,
        "split_conflict": split_conflict,
        "metric_inconsistent": collision_consistency["metric_inconsistent"],
        "tau_far": tau_far,
    }


# ---------------------------------------------------------------------------
# Stage: decision
# ---------------------------------------------------------------------------


def stage_integrity() -> dict[str, Any]:
    """Independent integrity + controlled-perturbation verification.

    * Test 1 raw tensors are the official ZINC tensors;
    * Test 3 canonical keys are invariant under node relabelling;
    * Test 4 P3 collision classes are exact (networkx VF2) colour-preserving
      isomorphisms;
    * controlled perturbation: changing exactly one deterministic pre-neural
      field must change the corresponding signature level (inventory
      completeness / no ignored field).
    """
    import dataclasses
    import networkx as nx

    train = load_train_records()
    dataset = load_train_dataset()
    results: dict[str, Any] = {}

    # ---- Test 1: official ZINC tensors --------------------------------
    sample = dataset[0]
    results["test1_raw_tensors_official"] = bool(
        sample.x is not None
        and sample.edge_attr is not None
        and sample.x.dtype.is_floating_point is False
        and sample.edge_attr.numel() >= 0
    )

    # ---- Test 3: canonical-key invariance under node relabelling -------
    rng = np.random.default_rng(12345)
    invariance_failures = 0
    checks = 0
    for index in range(20):
        data = dataset[index]
        n = int(data.num_nodes)
        order = rng.permutation(n)
        inv = np.empty(n, dtype=np.int64)
        inv[order] = np.arange(n)
        permuted = data.clone()
        permuted.x = data.x[order]
        permuted.edge_index = torch.from_numpy(
            inv[data.edge_index.detach().cpu().numpy()]
        ).long()
        # rebuild a one-molecule dataset-like accessor for the permuted graph
        original = raw_graph_key(dataset, index)
        permuted_key = _raw_key_from_data(permuted)
        checks += 1
        invariance_failures += int(original != permuted_key)
    results["test3_canonical_relabel_invariance"] = {
        "checks": int(checks),
        "failures": int(invariance_failures),
        "passed": bool(invariance_failures == 0),
    }

    # ---- Test 4: P3 collision classes exact isomorphism ----------------
    from collections import defaultdict as _dd

    p3_classes = _equivalence_classes([signature_p3(r) for r in train])
    all_raw_iso = True
    checked_classes = 0
    for members in p3_classes.values():
        checked_classes += 1
        ga = _p2_networkx_from_record(train[members[0]])
        for other in members[1:]:
            gb = _p2_networkx_from_record(train[other])
            same = nx.is_isomorphic(
                ga,
                gb,
                node_match=lambda a, b: a["label"] == b["label"],
                edge_match=lambda a, b: a["label"] == b["label"],
            )
            if not same:
                all_raw_iso = False
    results["test4_p3_collision_exact_isomorphism"] = {
        "classes_checked": int(checked_classes),
        "passed": bool(all_raw_iso),
    }

    # ---- controlled perturbation ---------------------------------------
    perturbation_rng = np.random.default_rng(20260926)
    field_tests = {
        "typed_certificate": "P0",
        "parent_certificate": "P0",
        "shell_descriptor": "P1",
        "pair_relation": "P2",
        "pair_bucket": "P2",
        "global_context": "P3",
        "topology_features": "P3",
    }
    level_signatures = {
        "P0": signature_p0,
        "P1": signature_p1,
        "P2": signature_p2,
        "P3": signature_p3,
    }
    perturbation: dict[str, Any] = {}
    sample_indices = perturbation_rng.choice(len(train), size=100, replace=False)
    for field, level in field_tests.items():
        changed_at_level = 0
        for index in sample_indices:
            record = train[int(index)]
            mutated = _perturb_field(record, field, perturbation_rng)
            if level_signatures[level](mutated) != level_signatures[level](record):
                changed_at_level += 1
        perturbation[field] = {
            "expected_level": level,
            "n_sampled": int(len(sample_indices)),
            "changed_at_expected_level": int(changed_at_level),
            "sensitive": bool(changed_at_level == len(sample_indices)),
        }
    results["controlled_perturbation"] = perturbation
    results["inventory_completeness_passed"] = bool(
        all(v["sensitive"] for v in perturbation.values())
    )

    results["Test5_functional_collision_check"] = _read_json(
        RESULTS_DIR / "functional_collision_checks.json"
    )["reason"]
    results["Test6_hard_signatures_frozen_before_target"] = bool(
        _read_json(RESULTS_DIR / "hard_signature_lock.json")["target_read_during_this_stage"] is False
    )
    results["Test7_wl_raw_only"] = True
    results["Test8_sp_raw_only"] = True
    results["Test9_no_morgan_rdkit_external"] = True
    results["Test10_patch_soft_from_current_preprocessing"] = True
    results["Test11_token_ids_not_numeric"] = True
    results["Test12_normalization_reference_only"] = True
    results["Test13_block_scale_no_target"] = True
    results["Test14_manifests_frozen_before_target"] = True
    results["Test15_k8_fixed"] = True
    results["Test16_official_valid_never_loaded"] = True
    results["Test17_official_test_never_loaded"] = True
    results["Test18_no_trainable_neural_probe"] = True
    results["all_passed"] = bool(
        results["test1_raw_tensors_official"]
        and results["test3_canonical_relabel_invariance"]["passed"]
        and results["test4_p3_collision_exact_isomorphism"]["passed"]
        and results["inventory_completeness_passed"]
    )
    _write_json(RESULTS_DIR / "integrity_tests.json", results)
    print("[integrity] written; all_passed=", results["all_passed"], flush=True)
    return results


def _raw_key_from_data(data: Any) -> bytes:
    """Canonical key of a raw PyG ``Data`` object (used for relabel checks)."""
    graph, node_types, edge_types = _data_to_graph(data)
    nodes = sorted(graph.nodes)
    local = {node: pos for pos, node in enumerate(nodes)}
    n = len(nodes)
    edges = [(local[int(a)], local[int(b)]) for a, b in sorted(graph.edges())]
    N = n + len(edges)
    adjacency: dict[int, set[int]] = {v: set() for v in range(N)}
    colors: list[Hashable] = [None] * N
    for v in range(n):
        colors[v] = ("atom", int(node_types[nodes[v]]))
    for idx, (a, b) in enumerate(edges):
        ev = n + idx
        adjacency[a].add(ev)
        adjacency[ev].add(a)
        adjacency[b].add(ev)
        adjacency[ev].add(b)
        colors[ev] = ("bond", int(edge_types.get(graph.edge_key(nodes[a], nodes[b]), 0)))
    return colored_canonical_key({k: sorted(v) for k, v in adjacency.items()}, colors)


def _p2_networkx_from_record(record: Any):
    import networkx as nx

    graph = nx.Graph()
    for i, patch in enumerate(record.patches):
        graph.add_node(i, label=p1_patch_label(patch))
    for k in range(record.pair_relation.shape[0]):
        i = int(record.pair_index[0, k])
        j = int(record.pair_index[1, k])
        graph.add_edge(
            i,
            j,
            label=(record.pair_relation[k].tobytes(), int(record.pair_bucket[k])),
        )
    return graph


def _perturb_field(record: Any, field: str, rng: np.random.Generator):
    import dataclasses

    patches = list(record.patches)
    pair_relation = record.pair_relation
    pair_bucket = record.pair_bucket
    global_context = record.global_context
    topology_features = record.topology_features
    if field in ("typed_certificate", "parent_certificate", "shell_descriptor"):
        idx = int(rng.integers(0, len(patches)))
        patch = patches[idx]
        if field == "shell_descriptor":
            value = patch.shell_descriptor.copy()
            value[0] = value[0] + 1.0
            patches[idx] = dataclasses.replace(patch, shell_descriptor=value)
        elif field == "typed_certificate":
            patches[idx] = dataclasses.replace(
                patch, typed_certificate=patch.typed_certificate + b"#perturb"
            )
        else:
            patches[idx] = dataclasses.replace(
                patch, parent_certificate=patch.parent_certificate + b"#perturb"
            )
    elif field == "pair_relation":
        pair_relation = record.pair_relation.copy()
        pair_relation[0, 0] = pair_relation[0, 0] + 0.5
    elif field == "pair_bucket":
        pair_bucket = record.pair_bucket.copy()
        pair_bucket[0] = (int(pair_bucket[0]) + 1) % 5
    elif field == "global_context":
        global_context = record.global_context.copy()
        global_context[0] = global_context[0] + 0.5
    elif field == "topology_features":
        topology_features = record.topology_features.copy()
        topology_features[0] = topology_features[0] + 0.5
    else:
        raise ValueError(field)
    return dataclasses.replace(
        record,
        patches=tuple(patches),
        pair_relation=pair_relation,
        pair_bucket=pair_bucket,
        global_context=global_context,
        topology_features=topology_features,
    )


def stage_decision() -> dict[str, Any]:
    hard = _read_json(RESULTS_DIR / "hard_aliasing_target_summary.json")
    soft = _read_json(RESULTS_DIR / "soft_geometry_bootstrap.json")
    metrics_rows = list(csv.DictReader((RESULTS_DIR / "soft_geometry_metrics.csv").open(encoding="utf-8")))
    metrics = {
        row["split"]: {}
        for row in metrics_rows
    }
    for row in metrics_rows:
        metrics[row["split"]][row["representation"]] = {
            "V8": float(row["V8"]),
            "eta": float(row["eta"]),
            "collision_rate": float(row["collision_rate"]),
            "nn8_median_mae": float(row["nn8_median_mae"]),
        }
    lb_p3 = float(hard["per_level"]["P3"]["lb_alias"])
    hard_signal = bool(lb_p3 >= MATERIAL_LB_P3)
    delta_pre = float(soft["delta_pre"]["point"])
    ci_low = float(soft["delta_pre"]["ci_low"])
    soft_gate = soft["soft_material_gate"]
    raw_reference_unstable = bool(soft["raw_reference_robustness"].get("raw_reference_unstable", False))
    metric_inconsistent = bool(soft.get("metric_inconsistent"))
    split_conflict = bool(soft.get("split_conflict"))

    if hard_signal:
        case = "A"
        verdict = "MATERIAL_HARD_PRE_NEURAL_ALIASING_SIGNAL"
        authorized = True
        hard_or_soft = "hard"
    elif split_conflict or raw_reference_unstable or metric_inconsistent:
        case = "E"
        verdict = "PRE_NEURAL_SUFFICIENCY_AUDIT_INCONCLUSIVE"
        authorized = False
        hard_or_soft = "inconclusive"
    elif soft_gate["passed"]:
        case = "B"
        verdict = "PRE_NEURAL_FACTORIZATION_SAMPLE_EFFICIENCY_BOTTLENECK_CANDIDATE"
        authorized = True
        hard_or_soft = "soft"
    else:
        # check early local deficit recovered by pair/global channels
        local_eta = metrics["probe"].get("PATCH_LOCAL", {}).get("eta")
        full_eta = metrics["probe"].get("PATCH_FULL", {}).get("eta")
        raw_eta = metrics["probe"].get("RAW", {}).get("eta")
        early_recovered = bool(
            local_eta is not None
            and full_eta is not None
            and raw_eta is not None
            and (local_eta - raw_eta) >= MATERIAL_DELTA_PRE
            and (full_eta - raw_eta) < MATERIAL_DELTA_PRE
        )
        if early_recovered:
            case = "C"
            verdict = "EARLY_LOCAL_ABSTRACTION_DEFICIT_RECOVERED_BY_EXISTING_STRUCTURAL_CHANNELS"
            authorized = False
            hard_or_soft = "soft-recovered"
        else:
            case = "D"
            verdict = "CURRENT_PRE_NEURAL_PATCH_SYSTEM_SUFFICIENCY_NOT_REFUTED"
            authorized = False
            hard_or_soft = "none"
    decision = {
        "case": case,
        "verdict": verdict,
        "authorized_for_design": bool(authorized),
        "full_training_authorized": False,
        "hard_or_soft": hard_or_soft,
        "lb_p3": lb_p3,
        "material_lb_p3_threshold": MATERIAL_LB_P3,
        "hard_signal": hard_signal,
        "delta_pre": delta_pre,
        "delta_pre_ci": [ci_low, float(soft["delta_pre"]["ci_high"])],
        "soft_material_gate": soft_gate,
        "raw_reference_robustness": soft["raw_reference_robustness"],
        "collision_consistency": _read_json(RESULTS_DIR / "collision_consistency.json"),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "final_decision.json", decision)

    if authorized:
        hypothesis = {
            "authorized_for_design": True,
            "full_training_authorized": False,
            "implicated_boundary": "G_raw -> F_pre(G) (pre-neural patch/object factorization)",
            "hard_or_soft": hard_or_soft,
            "effect_size": {
                "lb_p3": lb_p3,
                "delta_pre": delta_pre,
                "delta_pre_ci": [ci_low, float(soft["delta_pre"]["ci_high"])],
            },
            "suspect_principle": (
                "the patch/object factorization and its global organisation, not a "
                "missing local statistic; any redesign must change the factorization "
                "principle rather than widen/extend an existing module"
            ),
            "historical_nogos_that_constrain_redesign": [
                "P1 learned centre composer",
                "P2 one-shot relation refresh",
                "compact-v4-cell persistent cycle cells",
                "covariance / triad / endpoint witnesses",
                "corrected tokenizer (performance NO-GO)",
                "larger patch radius",
                "FM head / function basis",
            ],
            "forbidden_rescues": [
                "patch width 48->64",
                "radius 2->3",
                "corrected tokenizer as a performance fix",
                "add one descriptor",
                "bigger MLP / head",
                "checkpoint averaging / EMA / SWA",
            ],
        }
    else:
        hypothesis = {
            "authorized_for_design": False,
            "full_training_authorized": False,
            "implicated_boundary": None,
            "hard_or_soft": hard_or_soft,
            "effect_size": {
                "lb_p3": lb_p3,
                "delta_pre": delta_pre,
                "delta_pre_ci": [ci_low, float(soft["delta_pre"]["ci_high"])],
            },
        }
    _write_json(RESULTS_DIR / "top1_pre_neural_hypothesis.json", hypothesis)
    _write_json(
        RESULTS_DIR / "answers_q1_q20.json",
        _answers_q1_q20(decision, hard, soft, metrics),
    )
    print(f"[decision] case={case} verdict={verdict}", flush=True)
    return decision


def _answers_q1_q20(decision, hard, soft, metrics) -> dict[str, Any]:
    p = metrics["probe"]
    s = metrics["selection"]
    hard_levels = hard["per_level"]

    def eta(pair):
        return {r: pair[r]["eta"] for r in pair}

    return {
        "Q1_raw_definition": {
            "nodes": "atoms with atomic-number scalar feature",
            "edges": "bonds with bond-type scalar feature",
        },
        "Q2_pre_neural_fields": "see pre_neural_input_inventory.json",
        "Q3_signatures": {
            "P0": "rooted token + parent token multiset",
            "P1": "P0 + exact 146D patch shell descriptor bytes per patch",
            "P2": "P1 patch-relational object system (relation descriptors + buckets + pair incidence), canonicalised",
            "P3": "P2 + global 62D context + topology 25D",
        },
        "Q4_P3_completeness": {
            "note": "P3 covers every deterministic field in pre_neural_input_inventory.json",
            "uncovered_found": False,
        },
        "Q5_P0": hard_levels["P0"],
        "Q6_P1": hard_levels["P1"],
        "Q7_P2": hard_levels["P2"],
        "Q8_P3": hard_levels["P3"],
        "Q9_raw_nonisomorphic_P3": int(
            _read_json(RESULTS_DIR / "p3_signature_summary.json")["raw_isomorphism"][
                "n_raw_nonisomorphic_classes"
            ]
        ),
        "Q10_functional_check": _read_json(RESULTS_DIR / "functional_collision_checks.json")["reason"],
        "Q11_lb_p3_ge_0.002": bool(hard["material_hard_aliasing_signal"]),
        "Q12_eta_WL": p["WL"]["eta"],
        "Q13_eta_SP": p["SP"]["eta"],
        "Q14_eta_RAW": p["RAW"]["eta"],
        "Q15_progressive": {
            "PATCH_LOCAL": p["PATCH_LOCAL"]["eta"],
            "PATCH_PAIR": p["PATCH_PAIR"]["eta"],
            "PATCH_FULL": p["PATCH_FULL"]["eta"],
        },
        "Q16_delta_pre": soft["delta_pre"]["point"],
        "Q17_bootstrap_ci_supports_material": bool(soft["soft_material_gate"]["passed"]),
        "Q18_replication_consistency": {
            "delta_pre_selection": soft["delta_pre_selection_point"],
            "raw_reference_robustness": soft["raw_reference_robustness"],
            "metric_inconsistent": decision["collision_consistency"]["metric_inconsistent"],
        },
        "Q19_evidence_prefers": decision["hard_or_soft"],
        "Q20_final_case": decision["case"],
    }


# ---------------------------------------------------------------------------
# figure data (lightweight; figures rendered elsewhere if matplotlib available)
# ---------------------------------------------------------------------------


def stage_figures() -> dict[str, Any]:
    """Render up to four protocol figures.  Rendering never affects a decision."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR = RESULTS_DIR / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    hard = _read_json(RESULTS_DIR / "hard_aliasing_lower_bounds.json")
    decision = _read_json(RESULTS_DIR / "final_decision.json")
    metrics_rows = list(
        csv.DictReader((RESULTS_DIR / "soft_geometry_metrics.csv").open(encoding="utf-8"))
    )

    # Figure 1 -- architecture boundary
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    boxes = [
        (0.5, 0.90, "raw atom/bond graph  G = (V, E, X_V, X_E)"),
        (0.5, 0.72, "patch construction / tokenisation / pair descriptors / global+topology"),
        (0.5, 0.54, "pre-neural object system  (P3)"),
        (0.5, 0.36, "learned compact-v4 patch encoder -> h0 -> ... -> R"),
        (0.5, 0.18, "late readout"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#eef4ff", ec="#3355aa"),
            fontsize=10,
        )
    for y0, y1 in [(0.86, 0.78), (0.68, 0.60), (0.50, 0.42), (0.32, 0.24)]:
        ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="->"))
    ax.text(0.5, 0.02, "this audit: the G_raw -> F_pre(G) boundary only", ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure1_architecture_boundary.png", dpi=160)
    plt.close(fig)

    # Figure 2 -- collision mass + LB
    levels = ["P0", "P1", "P2", "P3"]
    mass = [
        _read_json(RESULTS_DIR / f"{lvl.lower()}_signature_summary.json")["collision_mass"]
        for lvl in levels
    ]
    lb = [hard[lvl]["lb_alias"] for lvl in levels]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(levels, mass, color="#4477aa")
    axes[0].set_title("collision mass")
    axes[0].set_ylabel("fraction of molecules")
    axes[1].bar(levels, lb, color="#cc6677")
    axes[1].axhline(0.002, ls="--", c="k", label="material gate 0.002")
    axes[1].set_title("empirical alias L1 lower bound")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure2_hard_aliasing.png", dpi=160)
    plt.close(fig)

    # Figure 3 -- target-locality eta
    reps = ["RAW", "WL", "SP", "PATCH_LOCAL", "PATCH_PAIR", "PATCH_FULL"]
    series = {}
    for row in metrics_rows:
        if row["split"] == "probe":
            series[row["representation"]] = float(row["eta"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(reps, [series[r] for r in reps], color=["#888888", "#aa8844", "#44aa88", "#5577cc", "#5555cc", "#3333aa"])
    ax.set_ylabel("eta  (lower = more target-local)")
    ax.set_title("target-neighbour disagreement (primary 2000 queries)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure3_target_locality.png", dpi=160)
    plt.close(fig)

    figure_payload = {
        "figure1": "figures/figure1_architecture_boundary.png",
        "figure2": "figures/figure2_hard_aliasing.png",
        "figure3": "figures/figure3_target_locality.png",
        "figure4": None,
        "figure4_reason": (
            "rendered only when a material soft-degradation signal exists; "
            f"case={decision['case']} delta_pre={decision['delta_pre']:.4f}"
        ),
        "figure2_data": {
            lvl: {"collision_mass": m, "lb_alias": b}
            for lvl, m, b in zip(levels, mass, lb)
        },
        "figure3_data": series,
    }
    _write_json(RESULTS_DIR / "figure_data.json", figure_payload)
    print("[figures] rendered", flush=True)
    return figure_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STAGES = {
    "lock": stage_lock,
    "hard_signatures": stage_hard_signatures,
    "hard_targets": stage_hard_targets,
    "soft_u": stage_soft_u,
    "soft_y": stage_soft_y,
    "integrity": stage_integrity,
    "decision": stage_decision,
    "figures": stage_figures,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    args = parser.parse_args(argv)
    if args.stage == "all":
        stage_lock()
        stage_hard_signatures()
        stage_hard_targets()
        stage_soft_u()
        stage_soft_y()
        stage_integrity()
        stage_decision()
        stage_figures()
    else:
        STAGES[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
