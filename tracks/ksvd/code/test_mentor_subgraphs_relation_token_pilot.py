"""Synthetic self-tests for the mentor no-label relation-token pilot.

Exercises: dynamic target dimension (2*s+2), target-token exclusion from
context, graph-balanced RMSE aggregation, FINAL-focused gate classification,
root-group fold integrity, and a tiny end-to-end fold evaluation.  All graphs
are synthetic; no real mentor-bank data is touched.  No residual tokens are
built anywhere (they would leak the masked invariant target through context).
"""
from __future__ import annotations

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .mentor_grouped_splits import audit_fold_partition, balanced_group_folds
from .overlap_cover import _make_cover, _make_patch, patch_budget
from .overlap_stitching import make_cover_example
from .run_mentor_subgraphs_relation_token_pilot import (
    BRANCHES,
    FAMILIES,
    RIDGE_ALPHA,
    _run_fold,
    build_masked_matrices,
    classify,
    graph_balanced_metrics,
    invariant_tokens,
)
from .run_overlap_cover_audit import generate_graph
from .run_patch_relation_representation_audit import (
    _relation_arrays,
    all_pairs_shortest_paths,
    bag_features,
    patch_invariant_descriptor,
    pool_tokens,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _tiny_examples(
    n_graphs: int = 2,
    n_nodes: int = 16,
    patch_size: int = 5,
    seed: int = 4101,
    family: str = "d5_10",
) -> list:
    examples = []
    for graph_index in range(n_graphs):
        adjacency = generate_graph("small_world", n_nodes, 5, seed + graph_index)
        budget = patch_budget(adjacency, patch_size=patch_size, target_overlap=2)
        cover = sample_marginal_candidate_cover(
            adjacency,
            np.random.default_rng(5000 + graph_index),
            n_patches=budget,
            patch_size=patch_size,
            target_overlap=2,
            retained_beam=4,
            candidate_restarts=1,
        )
        examples.append(
            make_cover_example(graph_index, family, 5, adjacency, cover)
        )
    return examples


def _path(n_nodes: int) -> np.ndarray:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for node in range(n_nodes - 1):
        adjacency[node, node + 1] = adjacency[node + 1, node] = 1
    return adjacency


def _duplicate_descriptor_example() -> object:
    """Two 3-node path patches on a 6-node path -> identical descriptors."""
    adjacency = _path(6)
    cover = _make_cover(
        "test",
        [
            _make_patch(adjacency, (0, 1, 2), 1),
            _make_patch(adjacency, (3, 4, 5), 4),
        ],
    )
    return make_cover_example(0, "lt5", 2, adjacency, cover)


def _summary(value: float) -> dict[str, float]:
    return {
        "overall_rmse": value,
        "degree_rmse": value,
        "spectrum_rmse": value,
        "density_triangle_rmse": value,
    }


def _fixture_fold(
    fold_index: int,
    *,
    final_true: float,
    final_bag: float,
    final_shuffled: float,
    init_true: float,
    invariant_true: float,
    invariant_shuffled: float,
    integrity: bool = True,
    strata: dict[str, tuple[float, float, float]] | None = None,
) -> dict:
    branches = {
        "FINAL": {
            "BAG": {"summary": _summary(final_bag)},
            "TRUE_RELATION": {"summary": _summary(final_true)},
            "SHUFFLED_RELATION": {"summary": _summary(final_shuffled)},
        },
        "INIT": {
            "BAG": {"summary": _summary(init_true)},
            "TRUE_RELATION": {"summary": _summary(init_true)},
            "SHUFFLED_RELATION": {"summary": _summary(init_true)},
        },
        "INVARIANT": {
            "BAG": {"summary": _summary(invariant_true)},
            "TRUE_RELATION": {"summary": _summary(invariant_true)},
            "SHUFFLED_RELATION": {"summary": _summary(invariant_shuffled)},
        },
    }
    stratum_rows = []
    if strata is not None:
        for stratum, (bag, true, shuffled) in strata.items():
            stratum_rows.extend(
                [
                    {
                        "family": "FINAL",
                        "branch": "BAG",
                        "stratum": stratum,
                        "test_graph_count": 4,
                        "overall_rmse": bag,
                        "degree_rmse": bag,
                        "spectrum_rmse": bag,
                        "density_triangle_rmse": bag,
                    },
                    {
                        "family": "FINAL",
                        "branch": "TRUE_RELATION",
                        "stratum": stratum,
                        "test_graph_count": 4,
                        "overall_rmse": true,
                        "degree_rmse": true,
                        "spectrum_rmse": true,
                        "density_triangle_rmse": true,
                    },
                    {
                        "family": "FINAL",
                        "branch": "SHUFFLED_RELATION",
                        "stratum": stratum,
                        "test_graph_count": 4,
                        "overall_rmse": shuffled,
                        "degree_rmse": shuffled,
                        "spectrum_rmse": shuffled,
                        "density_triangle_rmse": shuffled,
                    },
                ]
            )
    return {
        "fold_index": fold_index,
        "branches": branches,
        "stratum_rows": stratum_rows,
        "invariants": {"passed": integrity},
    }


def _passing_folds() -> list[dict]:
    strata = {
        stratum: (0.10, 0.08, 0.09) for stratum in ("lt5", "d5_10", "d10_15", "d15_25", "ge25")
    }
    return [
        _fixture_fold(
            fold_index,
            final_true=0.08,
            final_bag=0.10,
            final_shuffled=0.09,
            init_true=0.085,
            invariant_true=0.07,
            invariant_shuffled=0.075,
            integrity=True,
            strata=strata,
        )
        for fold_index in range(3)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _test_dynamic_target_dimension() -> None:
    examples = _tiny_examples(n_graphs=2, n_nodes=14, patch_size=5)
    tokens = invariant_tokens(examples[0])
    assert tokens.shape[1] == 2 * 5 + 2
    assert patch_invariant_descriptor(examples[0].cover.patches[0].adjacency).size == 12
    tokens_by_graph = {
        int(example.graph_index): invariant_tokens(example) for example in examples
    }
    features, targets, graph_indices = build_masked_matrices(
        examples, tokens_by_graph, "INVARIANT"
    )
    assert targets.shape[1] == 12
    assert features["BAG"].shape[0] == targets.shape[0]
    assert features["TRUE_RELATION"].shape[0] == targets.shape[0]
    assert graph_indices.shape[0] == targets.shape[0]

    # A different patch size gives a different (dynamic) target dimension.
    examples6 = _tiny_examples(n_graphs=1, n_nodes=14, patch_size=6)
    tokens6 = {
        int(example.graph_index): invariant_tokens(example) for example in examples6
    }
    _features6, targets6, _rows6 = build_masked_matrices(
        examples6, tokens6, "INVARIANT"
    )
    assert targets6.shape[1] == 2 * 6 + 2
    assert targets6.shape[1] != targets.shape[1]


def _test_target_exclusion() -> None:
    examples = _tiny_examples(n_graphs=2, n_nodes=16, patch_size=5)
    for example in examples:
        tokens = invariant_tokens(example)
        distances = all_pairs_shortest_paths(example.adjacency)
        for target_index in range(tokens.shape[0]):
            context = np.delete(tokens, target_index, axis=0)
            assert context.shape[0] == tokens.shape[0] - 1
            assert np.allclose(
                bag_features(tokens, target_index), pool_tokens(context)
            )
            indices, overlap, center = _relation_arrays(
                example.cover, distances, target_index
            )
            assert int(target_index) not in {int(value) for value in indices}
            assert np.all(overlap >= 0.0) and np.all(overlap <= 1.0)
            assert np.all(center >= 0.0)
    # Duplicate descriptors: exclusion is positional, never value-based.
    duplicate = _duplicate_descriptor_example()
    tokens = invariant_tokens(duplicate)
    assert np.array_equal(tokens[0], tokens[1])
    for target_index in range(2):
        context = np.delete(tokens, target_index, axis=0)
        assert context.shape[0] == 1
        assert np.allclose(
            bag_features(tokens, target_index), pool_tokens(context)
        )


def _test_graph_balancing() -> None:
    rng = np.random.default_rng(77)
    targets = rng.standard_normal((10, 4))  # 2 * degree_dim + 2 with degree_dim=1
    predictions = targets + 0.1 * rng.standard_normal((10, 4))
    graph_indices = np.asarray([0] * 8 + [1] * 2, dtype=np.int64)
    rows, summary = graph_balanced_metrics(
        targets, predictions, graph_indices, degree_dim=1
    )
    assert len(rows) == 2
    assert rows[0]["graph_index"] == 0 and rows[1]["graph_index"] == 1
    for key in ("overall_rmse", "degree_rmse", "spectrum_rmse", "density_triangle_rmse"):
        expected = (rows[0][key] + rows[1][key]) / 2.0
        assert abs(summary[key] - expected) < 1e-12
    # The 8-row graph must not dominate the 2-row graph (graph-balanced).
    per_row_mean = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    assert abs(summary["overall_rmse"] - per_row_mean) > 1e-6


def _test_gate_classification() -> None:
    passed = classify(_passing_folds())
    assert passed["classification"] == "MENTOR_RELATION_TOKEN_PILOT_SUPPORTED"
    assert passed["relation_gate"] and passed["invariant_gate"] and passed["final_vs_init_gate"]
    assert passed["checks"]["final_true_vs_bag_gain_at_least_002"]
    assert passed["checks"]["final_true_vs_shuffled_gain_at_least_002"]
    assert passed["checks"]["final_true_wins_both_all_folds"]
    assert passed["checks"]["final_true_wins_both_strata_at_least_4_of_5"]
    assert passed["checks"]["final_true_improves_init_true"]
    assert passed["checks"]["invariant_true_vs_shuffled_gain_at_least_002"]
    assert passed["final_strata_won_count"] == 5

    # Relation + invariant gates pass, but FINAL_TRUE does not improve INIT_TRUE.
    no_final_value = [
        _fixture_fold(
            fold_index,
            final_true=0.085,
            final_bag=0.10,
            final_shuffled=0.09,
            init_true=0.08,
            invariant_true=0.07,
            invariant_shuffled=0.075,
            integrity=True,
            strata={
                stratum: (0.10, 0.085, 0.09)
                for stratum in ("lt5", "d5_10", "d10_15", "d15_25", "ge25")
            },
        )
        for fold_index in range(3)
    ]
    signal = classify(no_final_value)
    assert signal["classification"] == "RELATION_SIGNAL_PRESENT_BUT_NO_FINAL_KSVD_VALUE"
    assert signal["relation_gate"] and signal["invariant_gate"]
    assert not signal["final_vs_init_gate"]

    # Positive TRUE gains but below the 2% thresholds -> below gate.
    below_gate = [
        _fixture_fold(
            fold_index,
            final_true=0.0985,
            final_bag=0.10,
            final_shuffled=0.099,
            init_true=0.08,
            invariant_true=0.07,
            invariant_shuffled=0.075,
            integrity=True,
            strata={
                stratum: (0.10, 0.0985, 0.099)
                for stratum in ("lt5", "d5_10", "d10_15", "d15_25", "ge25")
            },
        )
        for fold_index in range(3)
    ]
    below = classify(below_gate)
    assert below["classification"] == "MENTOR_RELATION_SIGNAL_BELOW_GATE"
    assert below["positive_gains"] and not below["relation_gate"]

    # TRUE not better than matched controls (and no positive gains) -> reject.
    rejected = [
        _fixture_fold(
            fold_index,
            final_true=0.105,
            final_bag=0.10,
            final_shuffled=0.09,
            init_true=0.085,
            invariant_true=0.08,
            invariant_shuffled=0.075,
            integrity=True,
            strata={
                stratum: (0.10, 0.105, 0.09)
                for stratum in ("lt5", "d5_10", "d10_15", "d15_25", "ge25")
            },
        )
        for fold_index in range(3)
    ]
    assert classify(rejected)["classification"] == "REJECT_CURRENT_MENTOR_RELATION_TOKEN"

    # Integrity violation -> reject as well.
    broken = _passing_folds()
    broken[1]["invariants"]["passed"] = False
    broken_decision = classify(broken)
    assert broken_decision["classification"] == "REJECT_CURRENT_MENTOR_RELATION_TOKEN"
    assert not broken_decision["checks"]["integrity_passed"]
    # A failing root-group fold audit also rejects.
    broken_audit = classify(_passing_folds(), fold_audit={"passed": False})
    assert broken_audit["classification"] == "REJECT_CURRENT_MENTOR_RELATION_TOKEN"

    # Fewer than three available folds must raise.
    try:
        classify(_passing_folds()[:2])
        raise AssertionError("expected an error for fewer than three folds")
    except ValueError:
        pass


def _test_root_group_fold_integrity() -> None:
    rng = np.random.default_rng(20260807)
    indices = np.arange(60, dtype=np.int64)
    strata = rng.integers(0, 5, size=60)
    roots = (indices % 11).astype(np.int64)  # 11 shared roots -> 11 groups
    folds = balanced_group_folds(
        indices, strata, roots, n_splits=3, seed=20260807, view="root_candidate_grouped"
    )
    audit = audit_fold_partition(
        folds,
        indices,
        groups_array_by_index=dict(zip(indices, roots)),
        require_group_integrity=True,
    )
    assert audit["passed"]
    assert audit["group_leakage_count"] == 0
    assert len(folds) == 3
    for fold in folds:
        train_groups = {int(roots[i]) for i in fold.train_indices}
        test_groups = {int(roots[i]) for i in fold.test_indices}
        assert not (train_groups & test_groups)
        assert fold.view == "root_candidate_grouped"
    # A group count below the fold count must be rejected.
    try:
        balanced_group_folds(indices, strata, indices % 2, n_splits=3, seed=20260807)
        raise AssertionError("expected a fewer-groups error")
    except ValueError:
        pass


def _test_tiny_fold_evaluation() -> None:
    examples = _tiny_examples(n_graphs=6, n_nodes=16, patch_size=5, seed=5301)
    train = examples[:4]
    test = examples[4:]
    strata_by_source_index = {int(ex.graph_index): ex.family for ex in test}
    fold = _run_fold(
        train,
        test,
        fold_index=0,
        patch_size=5,
        n_atoms=4,
        sparsity=2,
        iterations=2,
        ridge_alpha=RIDGE_ALPHA,
        strata_by_source_index=strata_by_source_index,
    )
    assert fold["train_graph_count"] == 4
    assert fold["test_graph_count"] == 2
    for family in FAMILIES:
        for branch in BRANCHES:
            summary = fold["branches"][family][branch]["summary"]
            for key in (
                "overall_rmse",
                "degree_rmse",
                "spectrum_rmse",
                "density_triangle_rmse",
            ):
                assert np.isfinite(summary[key])
            graphs = fold["branches"][family][branch]["graphs"]
            assert len(graphs) == len(test)
            for row in graphs:
                assert row["density_stratum"] == "d5_10"
                assert np.isfinite(row["overall_rmse"])
    invariants = fold["invariants"]
    assert invariants["passed"]
    assert invariants["target_dimension"] == 2 * 5 + 2
    assert invariants["target_dimension_matches_2s_plus_2"]
    assert invariants["train_test_graph_isolation"]
    assert invariants["target_excluded_from_context"]
    assert invariants["residual_tokens_used"] is False
    assert invariants["graph_balanced_per_graph_rows"]
    assert any(row["stratum"] == "d5_10" for row in fold["stratum_rows"])

    # Deterministic seeds -> bitwise-identical rerun.
    again = _run_fold(
        train,
        test,
        fold_index=0,
        patch_size=5,
        n_atoms=4,
        sparsity=2,
        iterations=2,
        ridge_alpha=RIDGE_ALPHA,
        strata_by_source_index=strata_by_source_index,
    )
    assert again == fold


def main() -> int:
    _test_dynamic_target_dimension()
    _test_target_exclusion()
    _test_graph_balancing()
    _test_gate_classification()
    _test_root_group_fold_integrity()
    _test_tiny_fold_evaluation()
    print("mentor_subgraphs_relation_token_pilot self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
