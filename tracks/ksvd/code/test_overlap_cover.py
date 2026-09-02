"""Self-tests for continuous overlapping patch covers."""
from __future__ import annotations

import numpy as np

from .from_scratch_unplanted_representation import ring_lattice_adjacency
from .overlap_cover import (
    audit_cover,
    cover_vectors,
    mapped_replay_relabel_invariance,
    make_slot_persistent_cover,
    patch_budget,
    sample_frontier_cover,
    sample_independent_walk_cover,
    sample_edge_target_bridge_cover,
    sample_multi_chain_target_cover,
    sample_sliding_walk_cover,
)
from .run_overlap_cover_audit import generate_graph


def main() -> int:
    adjacency = ring_lattice_adjacency(n_nodes=30, neighbors_each_side=4)
    budget = patch_budget(adjacency, patch_size=8, target_overlap=4)
    assert budget >= 7

    methods = (
        sample_independent_walk_cover,
        sample_sliding_walk_cover,
        sample_frontier_cover,
    )
    for offset, method in enumerate(methods):
        kwargs = {
            "n_patches": budget,
            "patch_size": 8,
        }
        if method is not sample_independent_walk_cover:
            kwargs["target_overlap"] = 4
        cover = method(adjacency, np.random.default_rng(100 + offset), **kwargs)
        assert len(cover.patches) == budget
        assert cover_vectors(cover).shape == (budget, 28)
        assert all(len(patch.node_ids) == 8 for patch in cover.patches)
        assert all(len(set(patch.node_ids)) == 8 for patch in cover.patches)

        audit = audit_cover(adjacency, cover)
        assert audit["observed_pair_consistency"] == 1.0
        assert audit["observed_pair_accuracy"] == 1.0
        assert 0.0 < audit["node_pair_coverage"] <= 1.0
        assert 0.0 < audit["true_edge_coverage"] <= 1.0

        permutation = np.random.default_rng(900 + offset).permutation(adjacency.shape[0])
        replay = mapped_replay_relabel_invariance(adjacency, cover, permutation)
        assert replay["patch_adjacency_match_rate"] == 1.0
        assert replay["transition_slot_map_match_rate"] == 1.0

    for method in (sample_sliding_walk_cover, sample_frontier_cover):
        cover = method(
            adjacency,
            np.random.default_rng(77),
            n_patches=budget,
            patch_size=8,
            target_overlap=4,
        )
        assert all(transition.overlap_size == 4 for transition in cover.transitions)

    targeted = sample_edge_target_bridge_cover(
        adjacency,
        np.random.default_rng(333),
        n_patches=budget,
        patch_size=8,
        target_overlap=4,
    )
    targeted_audit = audit_cover(adjacency, targeted)
    assert all(transition.overlap_size == 4 for transition in targeted.transitions)
    assert targeted_audit["target_edge_hit_rate"] == 1.0
    assert targeted_audit["continuous_transition_fraction"] == 1.0
    assert targeted_audit["patch_connected_rate"] == 1.0
    persistent = make_slot_persistent_cover(adjacency, targeted)
    persistent_audit = audit_cover(adjacency, persistent)
    assert persistent_audit["continuous_shared_slot_persistence_rate"] == 1.0
    assert all(
        set(left.node_ids) == set(right.node_ids)
        for left, right in zip(targeted.patches, persistent.patches)
    )
    for metric in ("node_coverage", "true_edge_coverage", "node_pair_coverage"):
        assert persistent_audit[metric] == targeted_audit[metric]

    multi = sample_multi_chain_target_cover(
        adjacency,
        np.random.default_rng(444),
        n_patches=budget,
        patch_size=8,
        target_overlap=4,
        segment_length=4,
    )
    multi_audit = audit_cover(adjacency, multi)
    for index, transition in enumerate(multi.transitions, start=1):
        if multi.segment_ids[index] == multi.segment_ids[index - 1]:
            assert transition.overlap_size == 4
    assert multi_audit["target_edge_hit_rate"] == 1.0
    assert 0.70 <= multi_audit["continuous_transition_fraction"] < 1.0
    assert multi_audit["patch_connected_rate"] == 1.0

    for family in ("regular", "small_world", "block"):
        generated = generate_graph(family, 50, 15, seed=12345)
        assert generated.shape == (50, 50)
        assert np.array_equal(generated, generated.T)
        assert np.all(np.diag(generated) == 0)
        if family != "block":
            assert np.all(generated.sum(axis=1) == 15)
    print("overlap_cover self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
