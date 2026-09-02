"""Mentor-facing compact cache for the audited subgraph bank.

Reads the audited NetworkX bank ``data/subgraphs_50_20_10000_batch_0.pkl`` — a
list of 10 000 connected, simple, undirected 50-node graphs whose node ids are
source-graph node id strings (shared across the whole bank) — and converts it
once into a compact, pickle-free NPZ cache::

    adjacency        uint8   (G, 50, 50)   symmetric 0/1; row ``i`` of graph
                                           ``g`` corresponds to
                                           ``global_node_ids[g, i]``
    global_node_ids  int32   (G, 50)       index into ``vocab``
    vocab            unicode (V,)          global source node ids (sorted)
    roots            int32   (G,)          vocab index of the root candidate:
                                           the first inserted node of the
                                           original subgraph, i.e. the first
                                           node in the NetworkX insertion order
    edge_counts      int32   (G,)
    avg_degrees      float64 (G,)          2 * edge_counts / 50
    density_percent  float64 (G,)          100 * 2E / (50 * 49)
    metadata         unicode scalar        JSON string: source path + sha256,
                                           validation results, stratum counts

Build-time audit (raises ValueError/TypeError on any violation):

* every element is an undirected, connected, simple (no self-loop, no parallel
  edge) NetworkX Graph with exactly 50 nodes;
* node ids are normalized plain ``str`` (np.str_ accepted): non-empty, no
  surrounding whitespace, unique within a graph;
* shared node pairs never conflict: a pair that is an edge in one graph is an
  edge in every graph containing both endpoints;
* shared edges keep one identity: the same edge ``id`` always has the same
  endpoints, and the same endpoint pair always has the same edge ``id``.

Load-time audit (``allow_pickle=False``): shapes/dtypes, symmetry, binary
values, zero diagonal, ids within the vocabulary, ``roots == row[:, 0]``,
``edge_counts == row-sum / 2`` and the degree/density formulas, plus a source
checksum match when the source pkl is still present.  The original pkl is
never written to.

Deterministic density strata and root-aware stratified pilot selection:

``density_stratum`` maps average degree to bins ``<5 / [5,10) / [10,15) /
[15,25) / >=25``.  ``select_pilot_indices`` allocates the pilot size across
strata with equal quotas across non-empty strata (redistributing shortages), then
inside each stratum greedily prefers graphs whose root id has not been picked
yet — rare global roots first, ties broken by a seeded permutation — and fills
any remaining quota with duplicates in the same order.  The result is fully
deterministic for a fixed seed and provably maximizes the number of distinct
roots per stratum.

CLI: ``python -m code.data_mentor_subgraphs [--limit N] [--force]`` builds the
cache next to the source pkl and prints a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

EXPECTED_N_NODES = 50
DEFAULT_SOURCE_NAME = "subgraphs_50_20_10000_batch_0.pkl"
SCHEMA_VERSION = 1

# Average-degree strata frozen by the real-data pilot protocol.
AVERAGE_DEGREE_BIN_EDGES = (5.0, 10.0, 15.0, 25.0)
STRATUM_NAMES = ("lt5", "d5_10", "d10_15", "d15_25", "ge25")

DEFAULT_PILOT_SIZE = 500
DEFAULT_PILOT_SEED = 20260806

_REQUIRED_CACHE_KEYS = {
    "adjacency",
    "global_node_ids",
    "vocab",
    "roots",
    "edge_counts",
    "avg_degrees",
    "density_percent",
    "metadata",
}


class MentorSubgraphBundle:
    """Immutable view over the compact subgraph-bank cache."""

    __slots__ = (
        "adjacency",
        "global_node_ids",
        "vocab",
        "roots",
        "edge_counts",
        "avg_degrees",
        "density_percent",
        "metadata",
    )

    def __init__(
        self,
        adjacency: np.ndarray,
        global_node_ids: np.ndarray,
        vocab: np.ndarray,
        roots: np.ndarray,
        edge_counts: np.ndarray,
        avg_degrees: np.ndarray,
        density_percent: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        self.adjacency = adjacency
        self.global_node_ids = global_node_ids
        self.vocab = vocab
        self.roots = roots
        self.edge_counts = edge_counts
        self.avg_degrees = avg_degrees
        self.density_percent = density_percent
        self.metadata = metadata

    @property
    def n_graphs(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[1])

    def node_ids(self, graph_index: int) -> list[str]:
        """Global source node ids of one graph, in row order (round-trip)."""
        row = self.global_node_ids[graph_index]
        return [str(self.vocab[int(i)]) for i in row]


class PilotSelection:
    """Deterministic, root-aware stratified pilot index selection."""

    __slots__ = (
        "indices",
        "strata",
        "quotas",
        "selected_counts",
        "n_pilot",
        "seed",
        "n_distinct_roots",
    )

    def __init__(
        self,
        indices: np.ndarray,
        strata: np.ndarray,
        quotas: tuple[int, ...],
        selected_counts: tuple[int, ...],
        n_pilot: int,
        seed: int,
        n_distinct_roots: int,
    ) -> None:
        self.indices = indices
        self.strata = strata
        self.quotas = quotas
        self.selected_counts = selected_counts
        self.n_pilot = n_pilot
        self.seed = seed
        self.n_distinct_roots = n_distinct_roots

    @property
    def size(self) -> int:
        return int(self.indices.size)

    def summary(self) -> dict[str, Any]:
        stratum_counts = np.bincount(
            self.strata, minlength=len(STRATUM_NAMES)
        ).tolist()
        return {
            "n_pilot": self.n_pilot,
            "seed": self.seed,
            "size": self.size,
            "stratum_counts": dict(zip(STRATUM_NAMES, stratum_counts)),
            "quotas": dict(zip(STRATUM_NAMES, self.quotas)),
            "selected_counts": dict(zip(STRATUM_NAMES, self.selected_counts)),
            "n_distinct_roots": self.n_distinct_roots,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_source_path() -> Path:
    return _repo_root() / "data" / DEFAULT_SOURCE_NAME


def default_cache_path(source: str | Path | None = None) -> Path:
    return Path(source or default_source_path()).with_suffix(".npz")


def sha256_of(path: str | Path) -> str:
    """Hex sha256 of a file, streamed in chunks."""
    digest = sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def density_stratum(average_degrees: np.ndarray) -> np.ndarray:
    """Map average degree to the five frozen real-data density strata."""
    values = np.asarray(average_degrees, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("average_degrees must be a 1-D array")
    edges = np.asarray(AVERAGE_DEGREE_BIN_EDGES, dtype=np.float64)
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def edge_density_percent(
    edge_counts: np.ndarray, n_nodes: int = EXPECTED_N_NODES
) -> np.ndarray:
    """Density in percent: 100 * 2E / (n_nodes * (n_nodes - 1))."""
    counts = np.asarray(edge_counts)
    if counts.ndim != 1:
        raise ValueError("edge_counts must be a 1-D array")
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    return 100.0 * 2.0 * counts.astype(np.float64) / (n_nodes * (n_nodes - 1))


def edge_avg_degree(
    edge_counts: np.ndarray, n_nodes: int = EXPECTED_N_NODES
) -> np.ndarray:
    """Average degree: 2E / n_nodes."""
    counts = np.asarray(edge_counts)
    if counts.ndim != 1:
        raise ValueError("edge_counts must be a 1-D array")
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    return 2.0 * counts.astype(np.float64) / n_nodes


def _normalize_node_id(raw: object) -> str:
    """Normalize a node id to a plain str; np.str_ is accepted."""
    if not isinstance(raw, str):
        raise TypeError(f"node id must be str, got {type(raw).__name__}: {raw!r}")
    normalized = str(raw)
    if not normalized:
        raise ValueError("node id must be non-empty")
    if normalized != normalized.strip():
        raise ValueError(
            f"node id must not contain surrounding whitespace: {normalized!r}"
        )
    return normalized


def _normalize_edge_id(data: Mapping[str, Any]) -> str:
    if not isinstance(data, Mapping):
        raise TypeError(f"edge data must be a mapping, got {type(data).__name__}")
    if "id" not in data:
        raise ValueError("edge is missing the 'id' attribute")
    return _normalize_node_id(data["id"])


def _pair_key(a: int, b: int, vocab_size: int) -> int:
    """Canonical int key for the unordered pair of vocab indices (a, b)."""
    lo, hi = (a, b) if a <= b else (b, a)
    return lo * vocab_size + hi


def _validate_graph_structure(graph: Any, index: int) -> None:
    import networkx as nx

    if not isinstance(graph, nx.Graph):
        raise TypeError(
            f"graph[{index}] is not a networkx.Graph (got {type(graph).__name__})"
        )
    if nx.is_directed(graph):
        raise ValueError(f"graph[{index}] is directed; expected undirected")
    if graph.is_multigraph():
        raise ValueError(f"graph[{index}] contains a parallel edge; expected simple graph")
    if graph.number_of_nodes() != EXPECTED_N_NODES:
        raise ValueError(
            f"graph[{index}] has {graph.number_of_nodes()} nodes; "
            f"expected {EXPECTED_N_NODES}"
        )
    if not nx.is_connected(graph):
        raise ValueError(f"graph[{index}] is disconnected")


def bundle_from_graphs(
    graphs: Sequence[Any],
    *,
    source_path: str | Path | None = None,
    limit: int | None = None,
) -> MentorSubgraphBundle:
    """Audit a graph list and build the compact in-memory bundle.

    Runs the full build-time audit and raises on any violation.  The root
    candidate of each graph is the first node in the NetworkX insertion order
    (the original first-inserted node of the subgraph).
    """
    import networkx as nx  # noqa: F401  (required to read the nx.Graph objects)

    graphs = list(graphs)
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be >= 0")
        graphs = graphs[:limit]
    if not graphs:
        raise ValueError("graph list is empty")
    n_graphs = len(graphs)

    # --- Pass 1: structural validation + id/edge collection -----------------
    node_id_rows: list[list[str]] = []
    edge_count_list: list[int] = []
    root_list: list[str] = []
    all_ids: set[str] = set()
    edge_count_min = edge_count_max = 0
    for gi, graph in enumerate(graphs):
        _validate_graph_structure(graph, gi)
        node_ids = [_normalize_node_id(n) for n in graph.nodes()]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(f"graph[{gi}] contains duplicate node ids")
        node_id_rows.append(node_ids)
        all_ids.update(node_ids)
        root_list.append(node_ids[0])
        pair_seen: set[tuple[str, str]] = set()
        for u, v, data in graph.edges(data=True):
            uu = _normalize_node_id(u)
            vv = _normalize_node_id(v)
            if uu == vv:
                raise ValueError(f"graph[{gi}] contains a self-loop on {uu!r}")
            _normalize_edge_id(data)
            key = (uu, vv) if uu <= vv else (vv, uu)
            if key in pair_seen:
                raise ValueError(f"graph[{gi}] contains a parallel edge {key!r}")
            pair_seen.add(key)
        edge_count = len(pair_seen)
        edge_count_list.append(edge_count)
        if gi == 0 or edge_count < edge_count_min:
            edge_count_min = edge_count
        if edge_count > edge_count_max:
            edge_count_max = edge_count

    # --- Vocabulary (sorted global source node ids) -------------------------
    vocab_size = len(all_ids)
    if vocab_size > np.iinfo(np.int32).max:
        raise ValueError(f"vocabulary too large for int32 ids: {vocab_size}")
    vocab_list = sorted(all_ids)
    vocab = np.asarray(vocab_list, dtype=f"U{max(len(s) for s in vocab_list)}")
    id_to_index = {node_id: i for i, node_id in enumerate(vocab_list)}

    # --- Pass 2: arrays + global consistency checks -------------------------
    adjacency = np.zeros(
        (n_graphs, EXPECTED_N_NODES, EXPECTED_N_NODES), dtype=np.uint8
    )
    global_node_ids = np.zeros((n_graphs, EXPECTED_N_NODES), dtype=np.int32)
    roots = np.zeros(n_graphs, dtype=np.int32)
    edge_keys_chunks: list[np.ndarray] = []
    nonedge_keys_chunks: list[np.ndarray] = []
    eid_chunks: list[np.ndarray] = []
    degree_min = degree_max = 0
    for gi, graph in enumerate(graphs):
        node_ids = node_id_rows[gi]
        row = np.asarray(
            [id_to_index[node_id] for node_id in node_ids], dtype=np.int32
        )
        global_node_ids[gi] = row
        roots[gi] = id_to_index[root_list[gi]]
        position = {node_id: p for p, node_id in enumerate(node_ids)}
        edge_keys: list[int] = []
        eids: list[str] = []
        for u, v, data in graph.edges(data=True):
            uu = _normalize_node_id(u)
            vv = _normalize_node_id(v)
            try:
                iu, iv = position[uu], position[vv]
            except KeyError:
                raise ValueError(
                    f"graph[{gi}] edge endpoint {uu!r}/{vv!r} not in its node set"
                )
            adjacency[gi, iu, iv] = 1
            adjacency[gi, iv, iu] = 1
            a, b = int(row[iu]), int(row[iv])
            edge_keys.append(_pair_key(a, b, vocab_size))
            eids.append(_normalize_edge_id(data))
        edge_keys_chunks.append(np.asarray(edge_keys, dtype=np.int64))
        eid_chunks.append(np.asarray(eids, dtype="U"))
        # Non-edge pairs of this graph (both endpoints present).
        left_pos, right_pos = np.triu_indices(EXPECTED_N_NODES, k=1)
        left = row[left_pos].astype(np.int64)
        right = row[right_pos].astype(np.int64)
        all_pair_keys = np.minimum(left, right) * vocab_size + np.maximum(
            left, right
        )
        nonedge_keys_chunks.append(
            np.setdiff1d(all_pair_keys, np.unique(edge_keys_chunks[-1]))
        )
        degrees = adjacency[gi].sum(axis=0)
        if gi == 0:
            degree_min = int(degrees.min())
            degree_max = int(degrees.max())
        else:
            degree_min = min(degree_min, int(degrees.min()))
            degree_max = max(degree_max, int(degrees.max()))

    # Shared-pair adjacency conflict: edge in one graph, non-edge in another.
    edge_keys_all = np.unique(np.concatenate(edge_keys_chunks))
    nonedge_keys_all = np.unique(np.concatenate(nonedge_keys_chunks))
    conflicting_pairs = np.intersect1d(edge_keys_all, nonedge_keys_all)
    if conflicting_pairs.size:
        raise ValueError(
            f"{conflicting_pairs.size} node pairs appear as an edge in one "
            "graph and a non-edge in another"
        )

    # Edge id consistency: one id -> one pair, and one pair -> one id.
    eid_codes = np.concatenate(eid_chunks)
    all_pair_keys = np.concatenate(edge_keys_chunks)
    by_eid = np.lexsort((all_pair_keys, eid_codes))
    n_eid_endpoint_conflicts = int(
        np.count_nonzero(
            (eid_codes[by_eid][1:] == eid_codes[by_eid][:-1])
            & (all_pair_keys[by_eid][1:] != all_pair_keys[by_eid][:-1])
        )
    )
    if n_eid_endpoint_conflicts:
        raise ValueError(
            f"{n_eid_endpoint_conflicts} edge ids map to conflicting endpoints"
        )
    by_pair = np.lexsort((eid_codes, all_pair_keys))
    n_pair_eid_conflicts = int(
        np.count_nonzero(
            (all_pair_keys[by_pair][1:] == all_pair_keys[by_pair][:-1])
            & (eid_codes[by_pair][1:] != eid_codes[by_pair][:-1])
        )
    )
    if n_pair_eid_conflicts:
        raise ValueError(
            f"{n_pair_eid_conflicts} node pairs map to conflicting edge ids"
        )

    edge_counts = np.asarray(edge_count_list, dtype=np.int32)
    density = edge_density_percent(edge_counts)
    average_degrees = edge_avg_degree(edge_counts)
    strata = density_stratum(average_degrees)
    stratum_counts = np.bincount(strata, minlength=len(STRATUM_NAMES)).tolist()

    source = Path(source_path).resolve() if source_path is not None else None
    if source is not None and not source.exists():
        raise FileNotFoundError(f"source pkl not found: {source}")
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "builder": Path(__file__).name,
        "source_path": str(source) if source is not None else None,
        "source_size_bytes": source.stat().st_size if source is not None else None,
        "source_sha256": sha256_of(source) if source is not None else None,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "limit": None if limit is None else int(limit),
        "average_degree_bins": list(AVERAGE_DEGREE_BIN_EDGES),
        "n_graphs": n_graphs,
        "n_nodes_per_graph": EXPECTED_N_NODES,
        "n_edges_total": int(edge_counts.sum()),
        "n_unique_pairs": int(edge_keys_all.size),
        "n_unique_edge_ids": int(np.unique(eid_codes).size),
        "n_unique_node_ids": vocab_size,
        "n_unique_roots": len(set(root_list)),
        "edge_count_min": edge_count_min,
        "edge_count_max": edge_count_max,
        "degree_min": degree_min,
        "degree_max": degree_max,
        "density_percent_min": float(density.min()),
        "density_percent_max": float(density.max()),
        "stratum_counts": dict(zip(STRATUM_NAMES, stratum_counts)),
        "validation": {
            "n_nodes_per_graph": EXPECTED_N_NODES,
            "undirected": True,
            "connected": True,
            "simple": True,
            "binary": True,
            "node_ids_normalized": True,
            "n_pair_adjacency_conflicts": 0,
            "n_edge_id_endpoint_conflicts": 0,
            "n_pair_edge_id_conflicts": 0,
        },
    }
    return MentorSubgraphBundle(
        adjacency=adjacency,
        global_node_ids=global_node_ids,
        vocab=vocab,
        roots=roots,
        edge_counts=edge_counts,
        avg_degrees=average_degrees,
        density_percent=density,
        metadata=metadata,
    )


def audit_graph_list(graphs: Sequence[Any]) -> dict[str, Any]:
    """Run the full build-time audit and return the metadata (raises on any
    violation)."""
    return bundle_from_graphs(graphs).metadata


def save_bundle(bundle: MentorSubgraphBundle, cache_path: str | Path) -> None:
    """Write the bundle as a compact NPZ (atomic replace, no pickle)."""
    cache_path = Path(cache_path)
    if cache_path.suffix.lower() == ".pkl":
        raise ValueError(
            f"refusing to write a cache over a .pkl path: {cache_path}"
        )
    payload = {
        "adjacency": bundle.adjacency,
        "global_node_ids": bundle.global_node_ids,
        "vocab": bundle.vocab,
        "roots": bundle.roots,
        "edge_counts": bundle.edge_counts,
        "avg_degrees": bundle.avg_degrees,
        "density_percent": bundle.density_percent,
        "metadata": np.array(json.dumps(bundle.metadata, indent=2)),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, cache_path)


def load_graph_list(source_pkl: str | Path) -> list[Any]:
    """Load the raw pkl; must contain a list (of networkx.Graph)."""
    import networkx as nx  # noqa: F401  (required to unpickle nx.Graph objects)

    source_pkl = Path(source_pkl)
    if not source_pkl.exists():
        raise FileNotFoundError(f"source pkl not found: {source_pkl}")
    with open(source_pkl, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, list):
        raise ValueError(
            f"source pkl must contain a list, got {type(payload).__name__}"
        )
    return payload


def load_cache(
    cache_path: str | Path,
    *,
    limit: int | None = None,
    verify_checksum: bool = True,
) -> MentorSubgraphBundle:
    """Load the NPZ cache with ``allow_pickle=False`` and re-audit it.

    Structural checks run on every load; the source-pkl checksum is verified
    when the source file recorded in the metadata is still present.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"cache not found: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as loaded:
        missing = _REQUIRED_CACHE_KEYS.difference(loaded.files)
        if missing:
            raise ValueError(f"cache is missing {sorted(missing)}")
        adjacency = np.array(loaded["adjacency"], copy=True)
        global_node_ids = np.array(loaded["global_node_ids"], copy=True)
        vocab = np.array(loaded["vocab"], copy=True)
        roots = np.array(loaded["roots"], copy=True)
        edge_counts = np.array(loaded["edge_counts"], copy=True)
        avg_degrees = np.array(loaded["avg_degrees"], copy=True)
        density_percent = np.array(loaded["density_percent"], copy=True)
        meta_raw = loaded["metadata"]
        meta_text = str(meta_raw) if meta_raw.ndim == 0 else str(meta_raw[0])
    try:
        metadata: dict[str, Any] = json.loads(meta_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"cache metadata is not valid JSON: {exc}")

    # --- Load-time audit -----------------------------------------------------
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"cache schema_version {metadata.get('schema_version')!r} "
            f"!= expected {SCHEMA_VERSION}; rebuild with force=True"
        )
    if adjacency.ndim != 3 or adjacency.shape[1:] != (EXPECTED_N_NODES, EXPECTED_N_NODES):
        raise ValueError(f"invalid adjacency shape {adjacency.shape}")
    n_graphs = adjacency.shape[0]
    if adjacency.dtype != np.uint8:
        raise ValueError(f"adjacency must be uint8, got {adjacency.dtype}")
    if global_node_ids.shape != (n_graphs, EXPECTED_N_NODES):
        raise ValueError(
            f"global_node_ids shape {global_node_ids.shape} inconsistent "
            f"with adjacency {adjacency.shape}"
        )
    if global_node_ids.dtype != np.int32:
        raise ValueError(f"global_node_ids must be int32, got {global_node_ids.dtype}")
    if vocab.ndim != 1 or vocab.dtype.kind != "U":
        raise ValueError(
            f"vocab must be a 1-D unicode array, got shape {vocab.shape} "
            f"dtype {vocab.dtype}"
        )
    if roots.shape != (n_graphs,):
        raise ValueError(f"invalid roots shape {roots.shape}")
    if edge_counts.shape != (n_graphs,):
        raise ValueError(f"invalid edge_counts shape {edge_counts.shape}")
    if avg_degrees.shape != (n_graphs,):
        raise ValueError(f"invalid avg_degrees shape {avg_degrees.shape}")
    if density_percent.shape != (n_graphs,):
        raise ValueError(f"invalid density_percent shape {density_percent.shape}")
    if not np.array_equal(adjacency, adjacency.transpose(0, 2, 1)):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.diagonal(adjacency, axis1=1, axis2=2) != 0):
        raise ValueError("adjacency diagonal must be zero")
    if not np.all((adjacency == 0) | (adjacency == 1)):
        raise ValueError("adjacency must be binary")
    vocab_size = len(vocab)
    if not np.all((global_node_ids >= 0) & (global_node_ids < vocab_size)):
        raise ValueError("global_node_ids out of vocabulary range")
    if not np.all((roots >= 0) & (roots < vocab_size)):
        raise ValueError("roots out of vocabulary range")
    if not np.array_equal(global_node_ids[:, 0], roots):
        raise ValueError("roots must equal global_node_ids[:, 0]")
    expected_edges = adjacency.sum(axis=(1, 2), dtype=np.int64) // 2
    if not np.array_equal(expected_edges, edge_counts):
        raise ValueError("edge_counts inconsistent with adjacency row sums")
    if not np.array_equal(avg_degrees, edge_avg_degree(edge_counts)):
        raise ValueError("avg_degrees inconsistent with edge_counts")
    if not np.array_equal(density_percent, edge_density_percent(edge_counts)):
        raise ValueError("density_percent inconsistent with edge_counts")

    # --- Checksum vs the source pkl (when still present) --------------------
    metadata = dict(metadata)
    source = metadata.get("source_path")
    if verify_checksum and source and metadata.get("source_sha256"):
        if Path(source).exists():
            actual = sha256_of(source)
            if actual != metadata["source_sha256"]:
                raise ValueError(
                    "source pkl checksum mismatch: the cache is stale, "
                    "rebuild with force=True"
                )
            metadata["checksum_verified"] = True
        else:
            metadata["checksum_verified"] = "source_missing"
    else:
        metadata["checksum_verified"] = "source_missing"

    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be >= 0")
        limit = min(limit, n_graphs)
        if limit < n_graphs:
            metadata["cache_n_graphs"] = n_graphs
            metadata["limit_applied"] = limit
            metadata["n_graphs"] = limit
            adjacency = adjacency[:limit]
            global_node_ids = global_node_ids[:limit]
            roots = roots[:limit]
            edge_counts = edge_counts[:limit]
            avg_degrees = avg_degrees[:limit]
            density_percent = density_percent[:limit]

    return MentorSubgraphBundle(
        adjacency=adjacency,
        global_node_ids=global_node_ids,
        vocab=vocab,
        roots=roots,
        edge_counts=edge_counts,
        avg_degrees=avg_degrees,
        density_percent=density_percent,
        metadata=metadata,
    )


def load_bundle(
    *,
    source_pkl: str | Path | None = None,
    cache_path: str | Path | None = None,
    limit: int | None = None,
    force: bool = False,
) -> MentorSubgraphBundle:
    """Load the cache, building it from the source pkl when missing/stale."""
    source_pkl = Path(source_pkl) if source_pkl is not None else default_source_path()
    cache_path = Path(cache_path) if cache_path is not None else default_cache_path(source_pkl)
    if cache_path.resolve() == source_pkl.resolve():
        raise ValueError(
            "cache path must differ from the source pkl (the pkl is never overwritten)"
        )
    if cache_path.exists() and not force:
        return load_cache(cache_path, limit=limit)
    graphs = load_graph_list(source_pkl)
    bundle = bundle_from_graphs(graphs, source_path=source_pkl, limit=limit)
    save_bundle(bundle, cache_path)
    return bundle


def _balanced_quotas(counts: np.ndarray, n_pilot: int) -> np.ndarray:
    """Allocate equal stratum quotas, redistributing shortages deterministically."""
    counts = np.asarray(counts, dtype=np.int64)
    remaining = min(int(n_pilot), int(counts.sum()))
    quotas = np.zeros_like(counts)
    while remaining > 0:
        active = np.flatnonzero(quotas < counts)
        if active.size == 0:
            break
        share, extra = divmod(remaining, int(active.size))
        proposed = np.full(active.size, share, dtype=np.int64)
        proposed[:extra] += 1
        capacity = counts[active] - quotas[active]
        added = np.minimum(proposed, capacity)
        quotas[active] += added
        used = int(added.sum())
        if used == 0:
            break
        remaining -= used
    return quotas


def select_pilot_indices(
    bundle: MentorSubgraphBundle,
    n_pilot: int = DEFAULT_PILOT_SIZE,
    seed: int = DEFAULT_PILOT_SEED,
) -> PilotSelection:
    """Root-aware stratified pilot selection.

    Quotas are equal across non-empty strata, with deterministic redistribution
    if one stratum is too small. Inside each stratum the greedy picks graphs with
    not-yet-seen roots first, rarest
    global roots first, ties broken by a seeded permutation; remaining quota
    is filled with duplicates in the same order.  Deterministic for a fixed
    seed; maximizes the number of distinct roots per stratum.
    """
    n_pilot = int(n_pilot)
    if n_pilot < 0:
        raise ValueError("n_pilot must be >= 0")
    seed = int(seed)
    roots = bundle.roots
    strata = density_stratum(bundle.avg_degrees)
    n_graphs = bundle.n_graphs
    n_pilot = min(n_pilot, n_graphs)
    counts = np.bincount(strata, minlength=len(STRATUM_NAMES))
    quotas = _balanced_quotas(counts, n_pilot)
    root_freq = np.bincount(roots.astype(np.int64), minlength=len(bundle.vocab))
    rng = np.random.default_rng(seed)

    selected: list[int] = []
    selected_counts: list[int] = []
    for s in range(len(STRATUM_NAMES)):
        quota = int(quotas[s])
        indices = np.flatnonzero(strata == s)
        if quota == 0 or indices.size == 0:
            selected_counts.append(0)
            continue
        randomized_tie = rng.permutation(indices.size)
        order = np.lexsort(
            (
                indices,
                randomized_tie,
                root_freq[roots[indices].astype(np.int64)],
            )
        )
        ordered_indices = indices[order]
        chosen: list[int] = []
        seen_roots: set[int] = set()
        for raw_index in ordered_indices:
            if len(chosen) >= quota:
                break
            graph_index = int(raw_index)
            root = int(roots[graph_index])
            if root not in seen_roots:
                seen_roots.add(root)
                chosen.append(graph_index)
        chosen_set = set(chosen)
        for raw_index in ordered_indices:  # fill the remaining quota with duplicates
            if len(chosen) >= quota:
                break
            graph_index = int(raw_index)
            if graph_index not in chosen_set:
                chosen.append(graph_index)
                chosen_set.add(graph_index)
        selected.extend(chosen)
        selected_counts.append(len(chosen))

    indices = np.sort(np.asarray(selected, dtype=np.int64))
    n_distinct_roots = int(np.unique(roots[indices]).size)
    return PilotSelection(
        indices=indices,
        strata=strata,
        quotas=tuple(int(q) for q in quotas),
        selected_counts=tuple(selected_counts),
        n_pilot=n_pilot,
        seed=seed,
        n_distinct_roots=n_distinct_roots,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build/load the mentor subgraph NPZ cache and print a "
        "summary with the deterministic pilot selection."
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"source pkl (default: {DEFAULT_SOURCE_NAME} under repo data/)",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="output npz cache (default: source path with .npz suffix)",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only convert/load the first N graphs"
    )
    ap.add_argument(
        "--force", action="store_true", help="rebuild the cache even if it exists"
    )
    ap.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE)
    ap.add_argument("--pilot-seed", type=int, default=DEFAULT_PILOT_SEED)
    args = ap.parse_args()

    bundle = load_bundle(
        source_pkl=args.source,
        cache_path=args.cache,
        limit=args.limit,
        force=args.force,
    )
    pilot = select_pilot_indices(bundle, n_pilot=args.pilot_size, seed=args.pilot_seed)
    meta = bundle.metadata
    summary = {
        "cache": str(args.cache or default_cache_path(args.source)),
        "source_path": meta.get("source_path"),
        "source_sha256": meta.get("source_sha256"),
        "checksum_verified": meta.get("checksum_verified"),
        "n_graphs": bundle.n_graphs,
        "n_nodes_per_graph": bundle.n_nodes,
        "n_edges_total": meta.get("n_edges_total"),
        "n_unique_pairs": meta.get("n_unique_pairs"),
        "n_unique_edge_ids": meta.get("n_unique_edge_ids"),
        "n_unique_node_ids": meta.get("n_unique_node_ids"),
        "n_unique_roots": meta.get("n_unique_roots"),
        "stratum_counts": meta.get("stratum_counts"),
        "validation": meta.get("validation"),
        "pilot": pilot.summary(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
