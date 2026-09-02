"""Self-tests for the mentor-subgraph Beam8 pilot decision logic."""
from __future__ import annotations

from .beam8_coverage_operating_point import CHECKPOINTS
from .run_mentor_subgraphs_beam8_pilot import classify


def _graph(geometry: str, stratum: str = "d15_25") -> dict:
    return {
        "geometry": geometry,
        "density_stratum": stratum,
        "sampling_success": True,
        "cover_invariants": True,
        "data_contract": True,
    }


def _checkpoint(geometry: str, checkpoint: str, stratum: str = "d15_25") -> dict:
    return {
        "geometry": geometry,
        "checkpoint": checkpoint,
        "density_stratum": stratum,
        "patch_count": 10,
        "raw_pair_slots": 450,
        "unique_observed_pair_count": 400,
        "node_coverage": 1.0,
        "pair_coverage": 0.5,
        "edge_coverage": 0.96,
        "residual_edge_count": 20,
        "raw_zero_fill_rmse": 0.1,
        "covered_edge_density": 0.8,
        "last_marginal_new_edges": 5,
        "mean_marginal_new_edges": 20,
        "node_incident_recall_mean": 0.95,
        "node_incident_recall_p10": 0.91,
        "node_incident_recall_minimum": 0.8,
        "fully_covered_node_fraction": 0.5,
        "nodes_with_residual_fraction": 0.5,
        "maximum_residual_incident_count": 3,
        "edge_multiplicity_mean": 1.2,
        "edge_multiplicity_cv": 0.2,
        "low_degree_incident_edge_recall": 0.95,
        "high_degree_incident_edge_recall": 0.96,
        "zero_common_neighbor_edge_recall": 0.8,
        "bridge_edge_recall": None,
        "cross_block_edge_recall": None,
        "canonical_set_identity_bits": 300,
        "ksvd_code_proxy_bits": 390,
        "residual_subset_bits": 100,
        "canonical_hybrid_proxy_bits": 796,
        "dictionary_scalars": 1080,
        "code_scalars": 30,
    }


def _stability(**updates: object) -> dict:
    row = {
        "stable_class_match_rate": 1.0,
        "singleton_unique_id_match_rate": 1.0,
        "singleton_trial_count": 50,
        "singleton_match_count": 50,
        "canonical_adjacency_match": 1.0,
        "rooted_vector_row_match": 1.0,
        "rooted_transition_map_match": 1.0,
        "absolute_edge_coverage_delta": 0.0,
        "absolute_pair_coverage_delta": 0.0,
        "exact_ordered_chain_match": 1.0,
        "patch_set_jaccard": 1.0,
    }
    row.update(updates)
    return row


def main() -> int:
    geometry = "s10_o3"
    graphs = [_graph(geometry) for _ in range(25)]
    checkpoints = [
        _checkpoint(geometry, checkpoint) for checkpoint in CHECKPOINTS for _ in range(25)
    ]
    passed = classify(graphs, checkpoints, [_stability()], geometries=[geometry])
    assert passed["classification"] == "READY_FOR_GROUPED_KSVD_RECONSTRUCTION_FOLLOWUP"
    assert passed["stable_gate"]
    assert passed["feasible_geometries"] == [geometry]

    unstable = classify(
        graphs,
        checkpoints,
        [_stability(rooted_vector_row_match=0.8)],
        geometries=[geometry],
    )
    assert unstable["classification"] == "COVERAGE_READY_STABILITY_REQUIRES_REPAIR"

    unreachable = [row for row in checkpoints if row["checkpoint"] != "FAIR95"]
    no_geometry = classify(graphs, unreachable, [_stability()], geometries=[geometry])
    assert no_geometry["classification"] == "STABLE_PREORDER_READY_BEAM8_GEOMETRY_NO_GO"

    bad_data = [dict(row, data_contract=False) for row in graphs]
    failed = classify(bad_data, checkpoints, [_stability()], geometries=[geometry])
    assert failed["classification"] == "FAIL_MENTOR_SUBGRAPH_DATA_CONTRACT"
    print("mentor_subgraphs_beam8_pilot self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
