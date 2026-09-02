"""Self-tests for tracks/ksvd/code/data_mentor_subgraphs.py."""
from __future__ import annotations

import itertools
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from .data_mentor_subgraphs import (
    DEFAULT_PILOT_SEED,
    DEFAULT_PILOT_SIZE,
    EXPECTED_N_NODES,
    STRATUM_NAMES,
    audit_graph_list,
    bundle_from_graphs,
    default_cache_path,
    default_source_path,
    density_stratum,
    edge_avg_degree,
    edge_density_percent,
    load_bundle,
    load_cache,
    save_bundle,
    select_pilot_indices,
    sha256_of,
)


# ---------------------------------------------------------------------------
# Synthetic bank: connected 50-node simple graphs over a shared source-node
# pool, mirroring the audited pkl (str node ids, edge 'id'/'weight' attrs,
# roots drawn from a small pool, pairs/edges shared across graphs).
# ---------------------------------------------------------------------------


def _synthetic_graphs(n_graphs: int = 700, n_roots: int = 24) -> list:
    """Deterministic synthetic bank; every density stratum is populated and
    pair/edge-id consistency holds by construction."""
    import networkx as nx

    rng = np.random.default_rng(20260806)
    pool = [f"{10**14 + i}" for i in range(900)]  # 15-digit source ids
    root_pool = pool[:n_roots]
    body_pool = [f"{2 * 10**14 + i}" for i in range(n_graphs * 49)]

    # Phase A: node sets (root cycles through a small pool -> shared roots).
    # Body nodes are graph-local so every requested density remains feasible.
    node_sets: list[list[str]] = []
    for g in range(n_graphs):
        root = root_pool[g % n_roots]
        others = body_pool[g * 49 : (g + 1) * 49]
        node_sets.append([root] + others)
    pair_count: dict[tuple[str, str], int] = {}
    for node_ids in node_sets:
        for i in range(50):
            for j in range(i + 1, 50):
                a, b = node_ids[i], node_ids[j]
                key = (a, b) if a <= b else (b, a)
                pair_count[key] = pair_count.get(key, 0) + 1

    # Phase B: edge sets.  Pairs co-occurring in >= 2 graphs are edges in
    # every graph containing them (forced); all other edges are graph-local,
    # so adjacency can never conflict across graphs.
    bands = [(49, 124), (125, 249), (250, 374), (375, 624), (625, 890)]
    pair_eid: dict[tuple[str, str], str] = {}
    next_eid = itertools.count(1)
    graphs: list = []
    for g, node_ids in enumerate(node_sets):
        edge_set: set[tuple[str, str]] = set()
        for i in range(50):
            for j in range(i + 1, 50):
                a, b = node_ids[i], node_ids[j]
                key = (a, b) if a <= b else (b, a)
                if pair_count[key] >= 2:
                    edge_set.add(key)
        for k in range(1, 50):  # random spanning tree -> connected
            j = int(rng.integers(0, k))
            a, b = node_ids[k], node_ids[j]
            edge_set.add((a, b) if a <= b else (b, a))
        lo, hi = bands[g % len(bands)]
        target = int(rng.integers(lo, hi + 1))
        candidates = [
            (i, j)
            for i in range(50)
            for j in range(i + 1, 50)
            if (node_ids[i] <= node_ids[j] and (node_ids[i], node_ids[j]) not in edge_set)
            or (node_ids[j] < node_ids[i] and (node_ids[j], node_ids[i]) not in edge_set)
        ]
        for p in rng.permutation(len(candidates)):
            if len(edge_set) >= target:
                break
            i, j = candidates[p]
            a, b = node_ids[i], node_ids[j]
            edge_set.add((a, b) if a <= b else (b, a))
        assert len(edge_set) == target, f"graph {g}: {len(edge_set)} != {target}"
        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        for a, b in edge_set:
            key = (a, b)
            eid = pair_eid.get(key)
            if eid is None:
                eid = str(next(next_eid))
                pair_eid[key] = eid
            graph.add_edge(a, b, id=eid, weight=1.0)
        graphs.append(graph)
    return graphs


def _write_source_pkl(graphs: list, directory: Path) -> Path:
    source = directory / "subgraphs_synthetic.pkl"
    with open(source, "wb") as fh:
        pickle.dump(list(graphs), fh, protocol=pickle.HIGHEST_PROTOCOL)
    return source


def _cycle_graph(node_ids: list, edge_ids: list[str] | None = None) -> object:
    """Connected simple cycle over the given node ids (as-is keys)."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    n = len(node_ids)
    for k in range(n):
        eid = edge_ids[k] if edge_ids is not None else str(1000 + k)
        graph.add_edge(node_ids[k], node_ids[(k + 1) % n], id=eid, weight=1.0)
    return graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _test_density_stratum() -> None:
    edge_counts = np.asarray([49, 124, 125, 249, 250, 374, 375, 624, 625, 890])
    density = edge_density_percent(edge_counts)
    average_degree = edge_avg_degree(edge_counts)
    expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    strata = density_stratum(average_degree)
    assert strata.dtype == np.int64
    assert strata.shape == (len(edge_counts),)
    assert strata.tolist() == expected
    # determinism
    assert np.array_equal(strata, density_stratum(average_degree))
    assert np.array_equal(strata, density_stratum(edge_avg_degree(edge_counts)))
    # formulas
    assert np.array_equal(density, 100.0 * 2.0 * edge_counts / (50 * 49))
    assert np.array_equal(
        edge_avg_degree(edge_counts), 2.0 * edge_counts.astype(np.float64) / 50
    )
    # input validation
    try:
        density_stratum(np.zeros((2, 2)))
        raise AssertionError("expected ValueError for 2-D input")
    except ValueError:
        pass


def _test_round_trip_and_cache() -> None:
    graphs = _synthetic_graphs()
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        source = _write_source_pkl(graphs, directory)
        cache = directory / "subgraphs_synthetic.npz"
        bundle = bundle_from_graphs(graphs, source_path=source)
        save_bundle(bundle, cache)
        assert cache.exists()
        loaded = load_cache(cache)

        # arrays round-trip exactly
        assert np.array_equal(loaded.adjacency, bundle.adjacency)
        assert np.array_equal(loaded.global_node_ids, bundle.global_node_ids)
        assert np.array_equal(loaded.roots, bundle.roots)
        assert np.array_equal(loaded.edge_counts, bundle.edge_counts)
        assert np.array_equal(loaded.avg_degrees, bundle.avg_degrees)
        assert np.array_equal(loaded.density_percent, bundle.density_percent)

        # vocab: unicode, not object; exact round-trip of source ids
        assert loaded.vocab.dtype.kind == "U"
        assert loaded.vocab.dtype != object
        all_ids = {str(n) for g in graphs for n in g.nodes()}
        assert {str(s) for s in loaded.vocab} == all_ids
        assert loaded.vocab.tolist() == sorted(all_ids)

        # per-row round-trip: vocab[global_node_ids] == original node ids,
        # row 0 == root candidate == first inserted node
        for gi, graph in enumerate(graphs[:20]):
            expected_ids = [str(n) for n in graph.nodes()]
            assert loaded.node_ids(gi) == expected_ids
            assert str(loaded.vocab[int(loaded.roots[gi])]) == expected_ids[0]
            assert int(loaded.global_node_ids[gi, 0]) == int(loaded.roots[gi])

        # metadata
        assert loaded.metadata["source_sha256"] == sha256_of(source)
        assert loaded.metadata["source_path"] == str(source.resolve())
        assert loaded.metadata["n_graphs"] == len(graphs)
        assert loaded.metadata["n_nodes_per_graph"] == EXPECTED_N_NODES
        assert loaded.metadata["validation"]["n_pair_adjacency_conflicts"] == 0
        assert loaded.metadata["validation"]["n_edge_id_endpoint_conflicts"] == 0
        assert loaded.metadata["validation"]["n_pair_edge_id_conflicts"] == 0
        assert loaded.metadata["checksum_verified"] is True

        # adjacency invariants
        adjacency = loaded.adjacency
        assert adjacency.dtype == np.uint8
        assert adjacency.shape == (len(graphs), 50, 50)
        assert np.array_equal(adjacency, adjacency.transpose(0, 2, 1))
        assert np.all((adjacency == 0) | (adjacency == 1))
        assert np.all(np.diagonal(adjacency, axis1=1, axis2=2) == 0)

        # per-graph ground truth vs networkx
        for gi, graph in enumerate(graphs):
            assert int(loaded.edge_counts[gi]) == graph.number_of_edges()
            position = {str(n): p for p, n in enumerate(graph.nodes())}
            for u, v in graph.edges():
                assert adjacency[gi, position[str(u)], position[str(v)]] == 1

        # derived arrays consistent with edge counts
        edge_counts = loaded.edge_counts.astype(np.float64)
        assert np.array_equal(loaded.avg_degrees, 2.0 * edge_counts / 50.0)
        assert np.array_equal(loaded.density_percent, 100.0 * 2.0 * edge_counts / 2450.0)

        # metadata stratum counts match the deterministic strata
        strata = density_stratum(loaded.avg_degrees)
        counts = np.bincount(strata, minlength=len(STRATUM_NAMES))
        assert [
            loaded.metadata["stratum_counts"][STRATUM_NAMES[s]] for s in range(5)
        ] == counts.tolist()
        assert all(counts > 0)

        # load-time limit
        limited = load_cache(cache, limit=60)
        assert limited.adjacency.shape == (60, 50, 50)
        assert np.array_equal(limited.adjacency, loaded.adjacency[:60])
        assert np.array_equal(limited.global_node_ids, loaded.global_node_ids[:60])
        assert limited.metadata["limit_applied"] == 60
        assert limited.metadata["cache_n_graphs"] == len(graphs)
        assert limited.metadata["n_graphs"] == 60

        # build-time limit
        partial = bundle_from_graphs(graphs, source_path=source, limit=100)
        assert partial.n_graphs == 100
        assert partial.metadata["limit"] == 100
        assert partial.metadata["n_graphs"] == 100

        # audit_graph_list returns the same metadata (no source)
        audit = audit_graph_list(graphs)
        assert audit["validation"]["n_pair_adjacency_conflicts"] == 0
        assert audit["n_graphs"] == len(graphs)


def _expect_raise(callable_fn, message: str, *args, **kwargs) -> None:
    try:
        callable_fn(*args, **kwargs)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        if message in str(exc):
            return
        raise AssertionError(f"unexpected error for {message}: {exc}")
    raise AssertionError(f"expected failure for: {message}")


def _test_validation_violations() -> None:
    ids = [f"n{i}" for i in range(50)]
    empty: list = []

    # structural violations
    _expect_raise(bundle_from_graphs, "empty", empty)
    _expect_raise(bundle_from_graphs, "not a networkx.Graph", [42])
    _expect_raise(bundle_from_graphs, "49 nodes", [_cycle_graph(ids[:49])])
    import networkx as nx

    disconnected = nx.disjoint_union(
        _cycle_graph(ids[:25]), _cycle_graph(ids[25:])
    )
    disconnected = nx.relabel_nodes(
        disconnected, {index: ids[index] for index in range(50)}
    )
    _expect_raise(bundle_from_graphs, "disconnected", [disconnected])
    _expect_raise(
        bundle_from_graphs,
        "directed",
        [_directed_cycle(ids)],
    )
    # self-loop / parallel edge
    self_loop = nx.Graph()
    self_loop.add_nodes_from(ids)
    for k in range(50):
        self_loop.add_edge(ids[k], ids[(k + 1) % 50], id=str(k), weight=1.0)
    self_loop.add_edge(ids[0], ids[0], id="loop", weight=1.0)
    _expect_raise(bundle_from_graphs, "self-loop", [self_loop])

    parallel = nx.MultiGraph()
    parallel.add_nodes_from(ids)
    for k in range(50):
        parallel.add_edge(ids[k], ids[(k + 1) % 50], id=str(k), weight=1.0)
    parallel.add_edge(ids[0], ids[1], id="dup", weight=1.0)
    _expect_raise(bundle_from_graphs, "parallel edge", [parallel])

    # node id normalization
    _expect_raise(bundle_from_graphs, "must be str", [_cycle_graph(list(range(50)))])
    _expect_raise(
        bundle_from_graphs, "whitespace", [_cycle_graph([f" n{i}" for i in range(50)])]
    )

    # edge missing 'id'
    missing_id = nx.Graph()
    missing_id.add_nodes_from(ids)
    for k in range(49):
        missing_id.add_edge(ids[k], ids[(k + 1) % 50], id=str(k), weight=1.0)
    missing_id.add_edge(ids[49], ids[0], weight=1.0)
    _expect_raise(bundle_from_graphs, "missing the 'id'", [missing_id])

    # shared-pair adjacency conflict: pair (n0, n1) edge in g1, non-edge in g2
    g1 = _cycle_graph(ids)
    g2 = _cycle_graph(ids)
    g2.remove_edge("n0", "n1")
    assert g2.number_of_edges() == 49
    _expect_raise(bundle_from_graphs, "non-edge in another", [g1, g2])

    # edge id endpoint conflict: same id on two existing endpoint pairs
    g4 = _cycle_graph(ids, edge_ids=[str(1000 + k) for k in range(50)])
    g4["n1"]["n2"]["id"] = "1000"  # also used by (n0, n1)
    _expect_raise(bundle_from_graphs, "conflicting endpoints", [g4])

    # pair -> edge id conflict: same pair, two ids
    g5 = _cycle_graph(ids)
    g6 = _cycle_graph(ids)
    g6.remove_edge("n0", "n1")
    g6.add_edge("n0", "n1", id="999999", weight=1.0)
    _expect_raise(bundle_from_graphs, "conflicting edge ids", [g5, g6])

    # positive control: identical graphs share edge ids without conflict
    g7 = _cycle_graph(ids)
    g8 = _cycle_graph(ids)
    bundle = bundle_from_graphs([g7, g8])
    assert bundle.n_graphs == 2

    # np.str_ node ids are accepted and normalized to plain str
    np_ids = [np.str_(f"s{i}") for i in range(50)]
    bundle = bundle_from_graphs([_cycle_graph(np_ids)])
    assert bundle.vocab.dtype.kind == "U"
    assert bundle.node_ids(0) == [f"s{i}" for i in range(50)]

    # save_bundle must never write over a .pkl path
    with tempfile.TemporaryDirectory() as td:
        _expect_raise(
            save_bundle, "over a .pkl", bundle, Path(td) / "bank.pkl"
        )


def _directed_cycle(node_ids: list) -> object:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    n = len(node_ids)
    for k in range(n):
        graph.add_edge(node_ids[k], node_ids[(k + 1) % n], id=str(k), weight=1.0)
    return graph


def _test_pilot_selection() -> None:
    graphs = _synthetic_graphs()
    bundle = bundle_from_graphs(graphs)
    n_graphs = bundle.n_graphs
    assert n_graphs == 700

    selection = select_pilot_indices(bundle)
    assert selection.n_pilot == DEFAULT_PILOT_SIZE
    assert selection.seed == DEFAULT_PILOT_SEED
    assert selection.size == 500

    # determinism (direct, repeated, and via cache reload)
    again = select_pilot_indices(bundle)
    assert np.array_equal(selection.indices, again.indices)
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        cache = directory / "bank.npz"
        save_bundle(bundle, cache)
        reloaded = select_pilot_indices(load_cache(cache))
        assert np.array_equal(selection.indices, reloaded.indices)
    # seed sensitivity
    other = select_pilot_indices(bundle, seed=DEFAULT_PILOT_SEED + 1)
    assert not np.array_equal(selection.indices, other.indices)

    # shape / bounds / uniqueness / sortedness
    indices = selection.indices
    assert indices.dtype == np.int64
    assert indices.ndim == 1
    assert indices.min() >= 0 and indices.max() < n_graphs
    assert np.unique(indices).size == indices.size
    assert np.array_equal(indices, np.sort(indices))

    # coverage: every non-empty stratum contributes
    counts = np.bincount(selection.strata, minlength=len(STRATUM_NAMES))
    assert all(counts > 0)
    assert all(c > 0 for c in selection.selected_counts)

    # quotas: exact total, capped by stratum size, balanced across strata
    assert sum(selection.quotas) == 500
    assert sum(selection.selected_counts) == selection.size
    for s in range(len(STRATUM_NAMES)):
        assert selection.quotas[s] <= int(counts[s])
        assert selection.selected_counts[s] == selection.quotas[s]
    assert max(selection.quotas) - min(selection.quotas) <= 1

    # root-aware diversity: per stratum the greedy picks every distinct root
    # when the quota allows (synthetic bank: all 24 roots in every stratum)
    roots = bundle.roots
    for s in range(len(STRATUM_NAMES)):
        in_stratum = np.flatnonzero(selection.strata == s)
        picked = indices[np.isin(indices, in_stratum)]
        distinct_in_stratum = len(set(int(roots[i]) for i in in_stratum))
        distinct_picked = len(set(int(roots[i]) for i in picked))
        assert distinct_picked == min(int(selection.quotas[s]), distinct_in_stratum)
    assert selection.n_distinct_roots == len(set(int(roots[i]) for i in indices))
    assert selection.n_distinct_roots == 24  # all roots of the bank present

    # summary()
    summary = selection.summary()
    assert summary["size"] == 500
    assert summary["n_distinct_roots"] == 24
    assert set(summary["quotas"]) == set(STRATUM_NAMES)

    # n_pilot >= G selects everything
    all_selected = select_pilot_indices(bundle, n_pilot=10**6)
    assert all_selected.size == n_graphs
    assert sum(all_selected.quotas) == n_graphs
    assert np.array_equal(all_selected.indices, np.arange(n_graphs, dtype=np.int64))


def _test_load_bundle_orchestration() -> None:
    graphs = _synthetic_graphs(n_graphs=120)
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        source = _write_source_pkl(graphs, directory)
        cache = directory / "bank.npz"
        built = load_bundle(source_pkl=source, cache_path=cache)
        assert cache.exists()
        assert "checksum_verified" not in built.metadata  # fresh build
        cached = load_bundle(source_pkl=source, cache_path=cache)
        assert cached.metadata["checksum_verified"] is True  # load path used
        assert np.array_equal(built.adjacency, cached.adjacency)
        forced = load_bundle(source_pkl=source, cache_path=cache, force=True)
        assert "checksum_verified" not in forced.metadata
        assert np.array_equal(built.adjacency, forced.adjacency)
        # limit propagates through the orchestrator
        limited = load_bundle(source_pkl=source, cache_path=cache, limit=40)
        assert limited.adjacency.shape == (40, 50, 50)
        # default cache path derivation
        assert default_cache_path(source) == source.with_suffix(".npz")
        assert default_source_path().name == "subgraphs_50_20_10000_batch_0.pkl"
        # never overwrite the source pkl
        _expect_raise(
            load_bundle, "never overwritten", source_pkl=source, cache_path=source
        )


def _test_checksum_verification() -> None:
    graphs = _synthetic_graphs(n_graphs=60)
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        source = _write_source_pkl(graphs, directory)
        cache = directory / "bank.npz"
        bundle = bundle_from_graphs(graphs, source_path=source)
        save_bundle(bundle, cache)
        assert load_cache(cache).metadata["checksum_verified"] is True

        # tampered cache (asymmetric adjacency) -> load-time audit fails
        with np.load(cache, allow_pickle=False) as loaded:
            payload = {key: np.array(loaded[key], copy=True) for key in loaded.files}
        payload["adjacency"][0, 0, 1] ^= 1
        np.savez_compressed(cache, **payload)
        _expect_raise(load_cache, "symmetric", cache)

        # stale cache (source pkl changed) -> checksum mismatch
        save_bundle(bundle, cache)
        with open(source, "ab") as fh:
            fh.write(b"tamper")
        _expect_raise(load_cache, "checksum mismatch", cache)

        # sha256_of matches hashlib
        assert sha256_of(source) == __import__("hashlib").sha256(
            source.read_bytes()
        ).hexdigest()

        # missing source -> verified state is 'source_missing'
        source.unlink()
        assert load_cache(cache).metadata["checksum_verified"] == "source_missing"


def _test_real_bank_smoke() -> None:
    try:
        import networkx  # noqa: F401
    except ImportError:
        print("SKIP real-bank smoke: networkx not installed")
        return
    source = default_source_path()
    if not source.exists():
        print(f"SKIP real-bank smoke: {source} not present")
        return
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "subgraphs_50_20_10000_batch_0.npz"
        bundle = load_bundle(source_pkl=source, cache_path=cache, limit=300)
        assert bundle.n_graphs == 300
        assert bundle.n_nodes == EXPECTED_N_NODES
        assert cache.exists()
        meta = bundle.metadata
        assert meta["validation"]["n_pair_adjacency_conflicts"] == 0
        assert meta["validation"]["n_edge_id_endpoint_conflicts"] == 0
        assert meta["validation"]["n_pair_edge_id_conflicts"] == 0
        assert meta["validation"]["connected"] is True
        assert meta["source_sha256"] == sha256_of(source)
        assert meta["source_path"] == str(source.resolve())
        assert meta["n_unique_node_ids"] >= 50
        assert meta["n_unique_roots"] >= 1
        strata = density_stratum(bundle.avg_degrees)
        counts = np.bincount(strata, minlength=len(STRATUM_NAMES))
        assert all(counts > 0), f"first 300 graphs miss strata: {counts.tolist()}"
        assert [meta["stratum_counts"][STRATUM_NAMES[s]] for s in range(5)] == (
            counts.tolist()
        )
        # pilot on the limited bank (300 < default 500 -> all selected)
        pilot = select_pilot_indices(bundle)
        assert pilot.size == 300
        assert pilot.n_distinct_roots >= 1
        # reload with verification
        reloaded = load_cache(cache)
        assert reloaded.metadata["checksum_verified"] is True
        assert np.array_equal(reloaded.adjacency, bundle.adjacency)
        # JSON metadata string is intact
        assert json.loads(json.dumps(reloaded.metadata)) == reloaded.metadata


def main() -> int:
    _test_density_stratum()
    _test_round_trip_and_cache()
    _test_validation_violations()
    _test_pilot_selection()
    _test_load_bundle_orchestration()
    _test_checksum_verification()
    _test_real_bank_smoke()
    print("data_mentor_subgraphs self-tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
