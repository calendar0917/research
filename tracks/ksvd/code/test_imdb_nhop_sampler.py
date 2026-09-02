"""Self-tests for the IMDB direct capped n-hop sampler audit."""
from __future__ import annotations

import numpy as np

from .from_scratch_unplanted_representation import relabel_adjacency_and_order
from .imdb_nhop_sampler import (
    SELECTOR_MODES,
    direct_ego_size_feasibility,
    extract_nhop_patch_graphs,
    nhop_feature_matrices,
    nhop_substrate_summary,
    relabel_selector_audit,
    select_fixed_nhop_patch,
)
from .imdb_walk_substrate import TUStructureGraph


def _graph(index: int, label: int) -> TUStructureGraph:
    n = 12
    adjacency = np.zeros((n, n), dtype=np.int8)
    for node in range(1, n):
        adjacency[0, node] = adjacency[node, 0] = 1
    for node in range(1, 7):
        adjacency[node, 1 + node % 6] = 1
        adjacency[1 + node % 6, node] = 1
    if label:
        adjacency[1, 7] = adjacency[7, 1] = 1
    return TUStructureGraph(index=index, label=label, adjacency=adjacency)


def main() -> int:
    graph = _graph(0, 0)
    for mode in SELECTOR_MODES:
        patch = select_fixed_nhop_patch(graph.adjacency, 0, patch_size=7, mode=mode)
        assert patch.node_ids[0] == 0
        assert len(set(patch.node_ids)) == 7
        assert patch.ranked_vector.shape == (21,)
        assert patch.canonical_vector.shape == (21,)

        permutation = np.random.default_rng(9).permutation(12)
        relabeled, mapped_order = relabel_adjacency_and_order(
            graph.adjacency, patch.node_ids, permutation
        )
        mapped_root = mapped_order[0]
        candidate = select_fixed_nhop_patch(relabeled, mapped_root, patch_size=7, mode=mode)
        assert candidate.canonical_vector.shape == patch.canonical_vector.shape

    graphs = tuple(_graph(index, index % 2) for index in range(10))
    feasibility = direct_ego_size_feasibility(graphs, patch_size=7)
    assert feasibility["radii"]["1"]["root_count"] == 120
    examples = extract_nhop_patch_graphs(
        graphs, sampling_seed=20260731, patch_size=7, max_patches_per_graph=6
    )
    assert len(examples) == 10
    substrate = nhop_substrate_summary(examples, patch_size=7)
    assert set(substrate) == set(SELECTOR_MODES)
    for mode in SELECTOR_MODES:
        assert substrate[mode]["patch_count"] == 60
    audit = relabel_selector_audit(
        graphs,
        examples,
        seed=20260801,
        graph_limit=5,
        permutations_per_graph=2,
        patch_size=7,
    )
    assert audit["graphs_checked"] == 5
    for mode in SELECTOR_MODES:
        assert audit["modes"][mode]["comparisons"] == 60
    matrices, labels = nhop_feature_matrices(examples)
    assert matrices["stats"].shape == (10, 12)
    assert matrices["n_signature_canonical_mean_std"].shape == (10, 42)
    assert labels.shape == (10,)

    print("imdb_nhop_sampler self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
