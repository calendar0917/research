"""Self-tests for Beam8 edge-residual rate accounting and decisions."""
from __future__ import annotations

import numpy as np

from .beam8_edge_residual import (
    direct_enumerative_bits,
    identity_proxy_bits,
    prefix_rate_trajectory,
    residual_subset_bits,
    select_checkpoints,
)
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget
from .run_beam8_edge_residual_audit import CHECKPOINTS, classify
from .run_overlap_cover_audit import generate_graph


def _decision_fixture(*, hybrid_gain: bool) -> tuple[list[dict], list[dict]]:
    graph_rows = []
    checkpoint_rows = []
    for seed in (1, 2, 3):
        for graph_index in range(4):
            base_bits = 900
            hybrid_bits = 850 if hybrid_gain else 900
            graph_rows.append(
                {
                    "graph_index": graph_index,
                    "family": "regular",
                    "target_degree": 6,
                    "cover_seed": seed,
                    "maximum_patches": 20,
                    "trajectory_length": 21,
                    "cover_invariants": True,
                    "edge100_reachable": True,
                    "base_hybrid_bits": base_bits,
                    "edge100_patch_only_bits": 1000,
                    "hybrid_after_base_bits": hybrid_bits,
                    "hybrid_after_base_patch_count": 5,
                    "base_budget": 5,
                    "minimum_positive_prefix_bits": 850,
                    "minimum_positive_prefix_patch_count": 1,
                    "direct_enumerative_exact_bits": 800,
                    "residual_preferred": True,
                }
            )
            values = {
                "patch_count": 5,
                "edge_coverage": 0.9,
                "pair_coverage": 0.5,
                "residual_edge_count": 5,
                "raw_zero_fill_rmse": 0.1,
                "canonical_set_identity_bits": 300,
                "prefix_length_bits": 5,
                "ksvd_code_proxy_bits": 200,
                "residual_subset_bits": 400,
                "canonical_hybrid_proxy_bits": base_bits,
                "ordered_hybrid_proxy_bits": base_bits + 100,
                "canonical_raw_exact_bits": 950,
                "direct_bitset_bits": 1225,
                "direct_enumerative_exact_bits": 800,
            }
            for checkpoint in CHECKPOINTS:
                row = dict(values)
                if checkpoint == "HYBRID_AFTER_BASE":
                    row["canonical_hybrid_proxy_bits"] = hybrid_bits
                if checkpoint == "HYBRID_ALL":
                    row["canonical_hybrid_proxy_bits"] = 800
                checkpoint_rows.append(
                    {
                        "graph_index": graph_index,
                        "family": "regular",
                        "target_degree": 6,
                        "cover_seed": seed,
                        "checkpoint": checkpoint,
                        **row,
                    }
                )
    return checkpoint_rows, graph_rows


def main() -> int:
    identity = identity_proxy_bits(
        n_nodes=50, patch_size=10, overlap=3, patch_count=18
    )
    assert 0 < identity["canonical_set_identity_bits"] < identity["ordered_identity_bits"]
    assert identity_proxy_bits(
        n_nodes=50, patch_size=10, overlap=3, patch_count=0
    ) == {"ordered_identity_bits": 0, "canonical_set_identity_bits": 0}
    assert residual_subset_bits(10, 0) == 4
    assert residual_subset_bits(10, 10) == 4
    assert direct_enumerative_bits(10, 3) == 11

    adjacency = generate_graph("small_world", 18, 6, seed=711)
    budget = patch_budget(adjacency, patch_size=6, target_overlap=2)
    cover = sample_marginal_candidate_cover(
        adjacency,
        np.random.default_rng(712),
        n_patches=budget + 3,
        patch_size=6,
        target_overlap=2,
        retained_beam=4,
        candidate_restarts=1,
    )
    trajectory = prefix_rate_trajectory(
        adjacency,
        cover,
        patch_size=6,
        overlap=2,
        n_atoms=8,
        sparsity=2,
        coefficient_bits=6,
    )
    assert len(trajectory) == len(cover.patches) + 1
    assert trajectory[0]["patch_count"] == 0
    assert (
        trajectory[0]["canonical_hybrid_proxy_bits"]
        == trajectory[0]["direct_enumerative_exact_bits"]
        + trajectory[0]["prefix_length_bits"]
    )
    assert all(
        right["covered_edge_count"] >= left["covered_edge_count"]
        and right["observed_pair_count"] >= left["observed_pair_count"]
        and right["residual_edge_count"] <= left["residual_edge_count"]
        for left, right in zip(trajectory, trajectory[1:])
    )
    checkpoints = select_checkpoints(trajectory, base_patch_count=budget)
    assert checkpoints["ZERO"]["patch_count"] == 0
    assert checkpoints["BASE"]["patch_count"] == budget
    assert checkpoints["HYBRID_AFTER_BASE"]["patch_count"] >= budget

    rows, graphs = _decision_fixture(hybrid_gain=True)
    decision = classify(rows, graphs, [1, 2, 3])
    assert decision["invariant_gate"]
    assert decision["residual_repair_gate"]
    assert decision["hybrid_stopping_gate"]
    assert decision["classification"] == "REPAIR_POLICY_FOUND_BUT_NOT_GLOBAL_CODEC"

    rows, graphs = _decision_fixture(hybrid_gain=False)
    for graph in graphs:
        graph["residual_preferred"] = False
        graph["edge100_patch_only_bits"] = 800
    failed = classify(rows, graphs, [1, 2, 3])
    assert not failed["residual_repair_gate"]
    assert not failed["hybrid_stopping_gate"]
    assert failed["classification"] == "NO_RATE_JUSTIFIED_EDGE_REPAIR_POLICY"
    print("beam8_edge_residual self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
