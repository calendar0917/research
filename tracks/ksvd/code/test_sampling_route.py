"""Fast invariance and control checks for :mod:`sampling_route`.

Run from tracks/ksvd with:
  python -m code.test_sampling_route
"""
from __future__ import annotations

import numpy as np

from .graph import Graph, from_edges
from .sampling_route import (
    MatchedSamplingConfig,
    PatchCollection,
    content_readout,
    relation_channels,
    relation_graph_readout,
    relation_readout,
    sample_matched,
    sparse_code_matrix,
    vectorize_collection,
)


def _toy() -> Graph:
    return from_edges(9, [(i, (i + 1) % 9) for i in range(9)] + [(0, 4), (2, 7)])


def _relabel(g: Graph, old_to_new: np.ndarray) -> Graph:
    return from_edges(g.n, [(int(old_to_new[u]), int(old_to_new[v])) for u, v in g.edges()])


def test_sampler_contracts() -> None:
    g = _toy()
    cfg = MatchedSamplingConfig(n_patches=8, max_nodes=5, walk_length=10, seed=7)
    before = set(g.edges())
    for method in ["B0", "R2", "UniformRW", "CoverageRW", "Path", "PPR"]:
        pc = sample_matched(g, method, cfg)
        assert len(pc.node_sets) == cfg.n_patches, method
        assert len(pc.centers) == cfg.n_patches, method
        assert len(pc.trajectories) == cfg.n_patches, method
        for c, S in zip(pc.centers, pc.node_sets):
            assert c in S, (method, c, S)
            assert 1 <= len(S) <= cfg.max_nodes, (method, len(S))
    assert set(g.edges()) == before, "CoverageRW must never mutate/delete graph edges"


def test_relation_shuffle_contract() -> None:
    g = _toy()
    cfg = MatchedSamplingConfig(n_patches=8, max_nodes=5, seed=3)
    pc = sample_matched(g, "R2", cfg)
    Y = vectorize_collection(g, pc, cfg.max_nodes, "wl")
    # A deterministic toy dictionary is enough to test the readout contract.
    D = np.eye(Y.shape[0], min(6, Y.shape[0]), dtype=np.float64)
    X, err = sparse_code_matrix(D, Y, T=2)
    content = content_readout(X, err)
    perm = np.random.default_rng(11).permutation(X.shape[1])
    channels = relation_channels(g, pc)
    true = relation_readout(X, channels, include_topology=False)
    shuffled = relation_readout(X, channels, perm, include_topology=False)
    graph_only = relation_graph_readout(channels)
    assert np.allclose(content, content_readout(X[:, perm], err[perm]))
    assert true.shape == shuffled.shape
    assert np.all(np.isfinite(true)) and np.all(np.isfinite(shuffled))
    assert graph_only.shape == (6,) and np.all(np.isfinite(graph_only))


def test_wl_relabel_invariance() -> None:
    g = _toy()
    S = set(g.nodes)
    pc = PatchCollection("manual", [S], [0], [[]])
    y = vectorize_collection(g, pc, max_nodes=9, feature_mode="wl")
    old_to_new = np.asarray([4, 2, 8, 1, 6, 0, 7, 3, 5], dtype=np.int64)
    gp = _relabel(g, old_to_new)
    pcp = PatchCollection("manual", [set(gp.nodes)], [int(old_to_new[0])], [[]])
    yp = vectorize_collection(gp, pcp, max_nodes=9, feature_mode="wl")
    assert np.array_equal(y, yp)


def test_tiny_and_isolated_graphs() -> None:
    cases = [from_edges(1, []), from_edges(3, [(0, 1)])]
    cfg = MatchedSamplingConfig(n_patches=8, max_nodes=8, walk_length=16, seed=0)
    for g in cases:
        for method in ["Path", "PPR", "UniformRW", "CoverageRW"]:
            pc = sample_matched(g, method, cfg)
            assert len(pc.node_sets) == cfg.n_patches
            assert all(c in S for c, S in zip(pc.centers, pc.node_sets))


def main() -> int:
    test_sampler_contracts()
    test_relation_shuffle_contract()
    test_wl_relabel_invariance()
    test_tiny_and_isolated_graphs()
    print("sampling_route self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
