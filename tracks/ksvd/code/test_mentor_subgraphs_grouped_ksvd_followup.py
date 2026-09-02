"""Self-tests for run_mentor_subgraphs_grouped_ksvd_followup.py.

Runs on synthetic small graphs only (no real data dependency).  The tests
deliberately use noncontiguous source indices and graphs whose stable WL order
differs from the input row order, so the stable-slot -> source-row mapping
(``ids.order``) is exercised rather than assumed to be the identity.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .data_mentor_subgraphs import MentorSubgraphBundle
from .global_stable_ids import compute_global_wl_ids, reorder_by_stable_ids
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .mentor_grouped_splits import make_mentor_folds
from .overlap_stitching import make_cover_example, stack_cover_examples
from .canonical_slots import reorder_cover_structurally
from .run_mentor_subgraphs_grouped_ksvd_followup import (
    CHECKPOINTS,
    CORRECTIONS,
    STAGES,
    _control_gate,
    _covered_source_nodes,
    _mean_stage,
    _optimization_gate,
    _overlap_counts,
    _random_dictionary_seed,
    _run_fold,
    _split_data_checks,
    _verify_pilot_counts,
    _weighted_stage_mean,
    classify,
)
from .run_mentor_subgraphs_grouped_ksvd_followup import PreparedGraph


# ---------------------------------------------------------------------------
# Synthetic graph bank helpers
# ---------------------------------------------------------------------------


def _path_plus_edges(n_nodes: int, seed: int, extra_edges: int) -> np.ndarray:
    """Deterministic connected simple graph: path backbone plus random edges."""
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for index in range(n_nodes - 1):
        adjacency[index, index + 1] = 1
        adjacency[index + 1, index] = 1
    attempts = 0
    added = 0
    while added < extra_edges and attempts < 500:
        attempts += 1
        left, right = rng.integers(0, n_nodes, size=2)
        if left == right or adjacency[left, right]:
            continue
        adjacency[left, right] = 1
        adjacency[right, left] = 1
        added += 1
    return adjacency


def _synthetic_examples(
    indices, *, n_nodes=10, n_patches=6, patch_size=4, overlap=1, seed=7
):
    """Return (source_index, CoverExample, StableNodeIDs) triples."""
    examples = []
    for source_index in indices:
        adjacency = _path_plus_edges(n_nodes, int(source_index) * 1000 + seed, extra_edges=6)
        ids = compute_global_wl_ids(adjacency)
        stable = reorder_by_stable_ids(adjacency, ids)
        cover = sample_marginal_candidate_cover(
            stable,
            np.random.default_rng(int(source_index) + seed),
            n_patches=n_patches,
            patch_size=patch_size,
            target_overlap=overlap,
            retained_beam=8,
            candidate_restarts=1,
            allow_partial=True,
        )
        ordered, _diagnostics = reorder_cover_structurally(stable, cover, "rooted_canonical")
        example = make_cover_example(int(source_index), "test", 4, stable, ordered)
        examples.append((int(source_index), example, ids))
    return examples


def _fake_exposure_row(view, fold_index, **overrides):
    row = {
        "view": view,
        "fold_index": fold_index,
        "test_source_node_seen_fraction": 0.5,
        "source_node_jaccard": 0.2,
        "test_source_edge_seen_fraction": 0.4,
        "source_edge_jaccard": 0.1,
        "test_graph_source_node_exposure": {"mean": 0.5, "p10": 0.3, "minimum": 0.1},
        "test_graph_source_edge_exposure": {"mean": 0.4, "p10": 0.2, "minimum": 0.0},
        "root_intersection_count": 0,
    }
    row.update(overrides)
    return row


def _fake_gate_fold(
    init_error,
    final_error,
    *,
    init_observed=0.10,
    final_observed=0.09,
    nondead=22,
    max_share=0.30,
    test_graph_count=10,
):
    raw_summary = {
        "patch_relative_error": 0.0,
        "observed_pair_rmse": 0.0,
        "repeated_pair_disagreement_std_mean": 0.0,
        "repeated_pair_disagreement_range_mean": 0.0,
    }
    return {
        "test_graph_count": test_graph_count,
        "decomposition_delta": 1e-16,
        "stages": {
            "RAW": {"uncorrected": {"summary": dict(raw_summary)}},
            "PCA3": {
                "uncorrected": {"summary": {"patch_relative_error": 0.30}},
            },
            "RANDOM": {
                "uncorrected": {"summary": {"patch_relative_error": 0.25}},
            },
            "INIT": {
                "uncorrected": {
                    "summary": {
                        "patch_relative_error": init_error,
                        "observed_pair_rmse": init_observed,
                    }
                },
            },
            "FINAL": {
                "uncorrected": {
                    "summary": {
                        "patch_relative_error": final_error,
                        "observed_pair_rmse": final_observed,
                    }
                },
                "dictionary_health": {
                    "train": {
                        "nondead_atom_count": nondead,
                        "maximum_activation_share": max_share,
                    }
                },
            },
        },
    }


def _fake_data_fold():
    return {
        "stages": {
            "RAW": {
                "uncorrected": {
                    "summary": {
                        "patch_relative_error": 0.0,
                        "observed_pair_rmse": 0.0,
                        "repeated_pair_disagreement_std_mean": 0.0,
                        "repeated_pair_disagreement_range_mean": 0.0,
                    }
                },
            },
        },
        "decomposition_delta": 1e-16,
    }


def _fake_branch(folds, **extra):
    payload = {
        "fold_rows": folds,
        "weighted_mean_stages": {
            "FINAL": {
                "uncorrected": {"full_adjacency_rmse": 0.30, "full_edge_recall": 0.8},
                "residual_corrected": {"full_adjacency_rmse": 0.20},
            }
        },
        "graph_balanced_mean_stages": {},
        "stratum_rows": [],
        "dictionary_scalars": 100,
        "code_scalars_per_graph": 20.0,
    }
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# Reconstruction fold
# ---------------------------------------------------------------------------


def test_raw_identity_and_fold_stages():
    examples = _synthetic_examples([3, 7, 11, 15])
    train = [example for _index, example, _ids in examples[:2]]
    test = [example for _index, example, _ids in examples[2:]]
    fold = _run_fold(
        train,
        test,
        view="random_reference",
        fold_index=0,
        n_atoms=4,
        sparsity=2,
        iterations=2,
        random_seed=970301,
    )
    assert set(fold["stages"]) == set(STAGES)
    raw = fold["stages"]["RAW"]["uncorrected"]["summary"]
    assert raw["patch_relative_error"] == 0.0
    assert raw["observed_pair_rmse"] == 0.0
    assert raw["repeated_pair_disagreement_std_mean"] == 0.0
    assert raw["repeated_pair_disagreement_range_mean"] == 0.0
    # RAW with the exact residual sidecar is a perfect full reconstruction.
    assert fold["stages"]["RAW"]["residual_corrected"]["summary"]["full_adjacency_rmse"] == 0.0
    # Squared-error decomposition: uncorrected² = RAW coverage² + corrected².
    assert fold["decomposition_delta"] <= 1e-10
    # training_iterations must come from the ksvd recon_curve (not a stale key).
    assert fold["stages"]["FINAL"]["training"]["training_iterations"] == 2
    assert len(fold["stages"]["FINAL"]["training"]["recon_curve"]) == 2
    for correction in CORRECTIONS:
        for stage in STAGES:
            summary = fold["stages"][stage][correction]["summary"]
            assert summary["patch_relative_error"] >= 0.0
            assert "full_adjacency_rmse" in summary

    # RANDOM dictionary: every atom is a normalized centered train column.
    raw_train = stack_cover_examples(train)
    train_mean = np.mean(raw_train, axis=1, keepdims=True)
    centered_train = raw_train - train_mean
    random_dictionary = np.asarray(fold["dictionaries"]["RANDOM"])
    assert random_dictionary.shape == (raw_train.shape[0], 4)
    for atom in range(4):
        column = random_dictionary[:, atom]
        assert abs(float(np.linalg.norm(column)) - 1.0) < 1e-9
        matches = [
            np.allclose(column, centered_train[:, index] / np.linalg.norm(centered_train[:, index]))
            for index in range(centered_train.shape[1])
        ]
        assert any(matches), f"RANDOM atom {atom} is not a normalized train column"
    # INIT and FINAL dictionaries are unit-norm and the same shape.
    for name in ("INIT", "FINAL"):
        dictionary = np.asarray(fold["dictionaries"][name])
        assert dictionary.shape == (raw_train.shape[0], 4)
        assert np.allclose(np.linalg.norm(dictionary, axis=0), 1.0)
    # Determinism: identical inputs give identical dictionaries and seeds.
    repeat = _run_fold(
        train,
        test,
        view="random_reference",
        fold_index=0,
        n_atoms=4,
        sparsity=2,
        iterations=2,
        random_seed=970301,
    )
    assert np.array_equal(
        np.asarray(fold["dictionaries"]["FINAL"]),
        np.asarray(repeat["dictionaries"]["FINAL"]),
    )
    assert fold["stages"]["RANDOM"]["chosen_train_columns"] == repeat["stages"]["RANDOM"]["chosen_train_columns"]


def test_random_dictionary_seed_deterministic():
    assert _random_dictionary_seed("s8_o2", "BASE", 0) == 970301
    assert _random_dictionary_seed("s8_o2", "BASE", 0) == _random_dictionary_seed("s8_o2", "BASE", 0)
    seeds = {
        _random_dictionary_seed(geometry, checkpoint, fold)
        for geometry in ("s8_o2", "s10_o3", "s12_o4")
        for checkpoint in CHECKPOINTS
        for fold in range(3)
    }
    assert len(seeds) == 3 * 2 * 3


# ---------------------------------------------------------------------------
# Patch-vector overlap and source-node mapping
# ---------------------------------------------------------------------------


def test_vector_overlap_counts():
    train = [b"a", b"a", b"b"]
    test = [b"a", b"a", b"c"]
    counts = _overlap_counts(train, test)
    # occurrence counts multiplicity: 2 of 3 test vectors exactly seen.
    assert counts["occurrence_fraction"] == 2 / 3
    # unique sets {a,b} vs {a,c}: Jaccard 1/3.
    assert counts["unique_overlap"] == 1 / 3
    assert counts["exact_seen_test_vector_count"] == 2
    assert counts["train_unique_vector_count"] == 2
    assert counts["test_unique_vector_count"] == 2


def test_patch_source_node_mapping_uses_stable_order():
    """Stable slot k is source row position ids.order[k]; naive slot mapping is wrong."""
    n_nodes = 6
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for index in range(n_nodes - 1):
        adjacency[index, index + 1] = 1
        adjacency[index + 1, index] = 1
    ids = compute_global_wl_ids(adjacency)
    order = np.asarray(ids.order, dtype=np.int64)
    assert tuple(ids.order) != tuple(range(n_nodes)), "test graph must have nontrivial WL order"
    stable = reorder_by_stable_ids(adjacency, ids)
    cover = sample_marginal_candidate_cover(
        stable,
        np.random.default_rng(1),
        n_patches=2,
        patch_size=4,
        target_overlap=1,
        retained_beam=8,
        candidate_restarts=1,
        allow_partial=True,
    )
    slots = np.unique(
        np.concatenate([np.asarray(patch.node_ids, dtype=np.int64) for patch in cover.patches])
    )
    source_row = np.arange(n_nodes, dtype=np.int32)
    correct = {int(source_row[int(order[int(slot)])]) for slot in slots}
    naive = {int(source_row[int(slot)]) for slot in slots}
    assert correct != naive, "mapping through ids.order must matter for this graph"
    bundle = MentorSubgraphBundle(
        adjacency=adjacency[None, :, :],
        global_node_ids=source_row[None, :],
        vocab=np.arange(n_nodes).astype("U8"),
        roots=np.array([0], dtype=np.int32),
        edge_counts=np.array([n_nodes - 1], dtype=np.int32),
        avg_degrees=np.array([2.0 * (n_nodes - 1) / n_nodes], dtype=np.float64),
        density_percent=np.array([100.0], dtype=np.float64),
        metadata={},
    )
    assert _covered_source_nodes(bundle, 0, ids, slots) == correct


def test_mentor_folds_on_noncontiguous_indices():
    indices = np.asarray([3, 7, 11, 15, 19, 23], dtype=np.int64)
    strata = np.asarray([0, 1, 2, 3, 4, 0], dtype=np.int64)
    roots = np.asarray([10, 20, 10, 30, 40, 50], dtype=np.int64)  # graphs 3 and 11 share root 10
    views = make_mentor_folds(indices, strata, roots, n_splits=3, seed=20260807)
    roots_by_index = {
        int(index): int(root) for index, root in zip(indices, roots)
    }
    for fold in views["root_candidate_grouped"]:
        train_roots = {roots_by_index[int(index)] for index in fold.train_indices}
        test_roots = {roots_by_index[int(index)] for index in fold.test_indices}
        assert not (train_roots & test_roots), "root-grouped folds must not leak roots"
    all_test = set()
    for fold in views["random_reference"]:
        assert fold.train_indices.size > 0 and fold.test_indices.size > 0
        assert not set(int(i) for i in fold.train_indices) & set(int(i) for i in fold.test_indices)
        all_test.update(int(index) for index in fold.test_indices)
    assert all_test == set(int(index) for index in indices)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_optimization_gate_pass_and_fail():
    passing_folds = [
        _fake_gate_fold(0.20, 0.18, test_graph_count=10),
        _fake_gate_fold(0.22, 0.19, test_graph_count=20),
        _fake_gate_fold(0.18, 0.17, test_graph_count=30),
    ]
    random_folds = [
        _fake_gate_fold(0.20, 0.185, test_graph_count=10),
        _fake_gate_fold(0.22, 0.20, test_graph_count=20),
        _fake_gate_fold(0.18, 0.175, test_graph_count=30),
    ]
    branches = {
        "root_candidate_grouped": {"s8_o2_BASE": _fake_branch(passing_folds)},
        "random_reference": {"s8_o2_BASE": _fake_branch(random_folds)},
    }
    gate = _optimization_gate(branches, "s8_o2_BASE", n_folds=3)
    assert gate["passed"], gate
    assert gate["weighted_reduction"] >= 0.05
    assert gate["checks"]["root_not_systematically_reversed"] is True

    # Too small a reduction fails the >= 0.05 gate.
    weak_folds = [
        _fake_gate_fold(0.200, 0.199, test_graph_count=10),
        _fake_gate_fold(0.220, 0.219, test_graph_count=20),
        _fake_gate_fold(0.180, 0.179, test_graph_count=30),
    ]
    weak = {
        "root_candidate_grouped": {"s8_o2_BASE": _fake_branch(weak_folds)},
        "random_reference": {"s8_o2_BASE": _fake_branch(weak_folds)},
    }
    assert not _optimization_gate(weak, "s8_o2_BASE", n_folds=3)["passed"]

    # Degenerate dictionary health fails the gate.
    dead_folds = [
        _fake_gate_fold(0.20, 0.18, nondead=18, test_graph_count=10),
        _fake_gate_fold(0.22, 0.19, nondead=19, test_graph_count=20),
        _fake_gate_fold(0.18, 0.17, nondead=21, test_graph_count=30),
    ]
    dead = {
        "root_candidate_grouped": {"s8_o2_BASE": _fake_branch(dead_folds)},
        "random_reference": {"s8_o2_BASE": _fake_branch(dead_folds)},
    }
    gate = _optimization_gate(dead, "s8_o2_BASE", n_folds=3)
    assert not gate["passed"]
    assert not gate["checks"]["final_nondead_atoms_ge20_all_folds"]

    # A systematic reverse in the random-reference view also fails the
    # protocol's no-systematic-reversal requirement.
    reversed_random = [
        _fake_gate_fold(0.20, 0.21, test_graph_count=10),
        _fake_gate_fold(0.22, 0.23, test_graph_count=20),
        _fake_gate_fold(0.18, 0.19, test_graph_count=30),
    ]
    reversed_views = {
        "root_candidate_grouped": {"s8_o2_BASE": _fake_branch(passing_folds)},
        "random_reference": {"s8_o2_BASE": _fake_branch(reversed_random)},
    }
    gate = _optimization_gate(reversed_views, "s8_o2_BASE", n_folds=3)
    assert not gate["passed"]
    assert not gate["checks"]["root_not_systematically_reversed"]

    # Observed-pair RMSE regression fails the gate.
    regression_folds = [
        _fake_gate_fold(0.20, 0.18, init_observed=0.05, final_observed=0.08, test_graph_count=10),
        _fake_gate_fold(0.22, 0.19, test_graph_count=20),
        _fake_gate_fold(0.18, 0.17, test_graph_count=30),
    ]
    regression = {
        "root_candidate_grouped": {"s8_o2_BASE": _fake_branch(regression_folds)},
        "random_reference": {"s8_o2_BASE": _fake_branch(regression_folds)},
    }
    gate = _optimization_gate(regression, "s8_o2_BASE", n_folds=3)
    assert not gate["passed"]
    assert not gate["checks"]["final_observed_rmse_not_worse_all_folds"]


def test_control_gate():
    folds = [
        _fake_gate_fold(0.20, 0.18, test_graph_count=10),
        _fake_gate_fold(0.22, 0.19, test_graph_count=20),
        _fake_gate_fold(0.18, 0.17, test_graph_count=30),
    ]
    branches = {"root_candidate_grouped": {"s8_o2_BASE": _fake_branch(folds)}}
    gate = _control_gate(branches, "s8_o2_BASE")
    # FINAL 0.18 < RANDOM 0.25 < PCA3 0.30 (weighted by 10/20/30).
    assert gate["passed"]
    assert gate["final_beats_random"] and gate["final_beats_pca3"]

    bad_folds = [
        _fake_gate_fold(0.20, 0.26, test_graph_count=10),
        _fake_gate_fold(0.22, 0.27, test_graph_count=20),
        _fake_gate_fold(0.18, 0.28, test_graph_count=30),
    ]
    bad = {"root_candidate_grouped": {"s8_o2_BASE": _fake_branch(bad_folds)}}
    assert not _control_gate(bad, "s8_o2_BASE")["passed"]


def test_split_data_checks():
    branches = {
        "random_reference": {"s8_o2_BASE": _fake_branch([{**_fake_data_fold(), "test_graph_count": 5}])},
        "root_candidate_grouped": {"s8_o2_BASE": _fake_branch([{**_fake_data_fold(), "test_graph_count": 5}])},
    }
    exposure = [
        _fake_exposure_row("random_reference", 0),
        _fake_exposure_row("random_reference", 1),
        _fake_exposure_row("random_reference", 2),
        _fake_exposure_row("root_candidate_grouped", 0),
        _fake_exposure_row("root_candidate_grouped", 1),
        _fake_exposure_row("root_candidate_grouped", 2),
    ]
    checks = _split_data_checks(
        branches, exposure, views=("random_reference", "root_candidate_grouped"), n_folds=3
    )
    assert checks["passed"], checks

    # Root intersection in the root-grouped view breaks the data gate.
    leaky = [dict(row) for row in exposure]
    leaky[-1] = _fake_exposure_row("root_candidate_grouped", 2, root_intersection_count=1)
    checks = _split_data_checks(
        branches, leaky, views=("random_reference", "root_candidate_grouped"), n_folds=3
    )
    assert not checks["passed"]
    assert not checks["per_view"]["root_candidate_grouped"]["checks"]["root_intersection_zero"]

    # Exposure values outside [0, 1] break the gate.
    invalid = [dict(row) for row in exposure]
    invalid[0] = _fake_exposure_row("random_reference", 0, source_node_jaccard=1.5)
    checks = _split_data_checks(
        branches, invalid, views=("random_reference", "root_candidate_grouped"), n_folds=3
    )
    assert not checks["passed"]

    # Nonzero RAW error breaks the gate.
    bad_raw = {
        "stages": {
            "RAW": {
                "uncorrected": {
                    "summary": {
                        "patch_relative_error": 1e-9,
                        "observed_pair_rmse": 0.0,
                        "repeated_pair_disagreement_std_mean": 0.0,
                        "repeated_pair_disagreement_range_mean": 0.0,
                    }
                },
            },
        },
        "decomposition_delta": 1e-16,
        "test_graph_count": 5,
    }
    branches_bad = {
        "random_reference": {"s8_o2_BASE": _fake_branch([bad_raw])},
        "root_candidate_grouped": {"s8_o2_BASE": _fake_branch([bad_raw])},
    }
    checks = _split_data_checks(
        branches_bad, exposure, views=("random_reference", "root_candidate_grouped"), n_folds=3
    )
    assert not checks["passed"]


def test_classify_labels():
    good_folds = [
        _fake_gate_fold(0.20, 0.18, test_graph_count=10),
        _fake_gate_fold(0.22, 0.19, test_graph_count=20),
        _fake_gate_fold(0.18, 0.17, test_graph_count=30),
    ]
    data_folds = [
        {**_fake_data_fold(), "test_graph_count": 10},
        {**_fake_data_fold(), "test_graph_count": 20},
        {**_fake_data_fold(), "test_graph_count": 30},
    ]

    def make_branches(optimization_folds):
        return {
            "random_reference": {"s8_o2_BASE": _fake_branch(optimization_folds)},
            "root_candidate_grouped": {"s8_o2_BASE": _fake_branch(optimization_folds)},
        }

    exposure = [
        _fake_exposure_row(view, fold)
        for view in ("random_reference", "root_candidate_grouped")
        for fold in range(3)
    ]
    kwargs = dict(
        views=("random_reference", "root_candidate_grouped"),
        geometries=("s8_o2",),
        checkpoints=("BASE",),
        n_folds=3,
    )

    # Data gate failure -> contract failure.
    branches = make_branches(good_folds)
    leaky_exposure = [dict(row) for row in exposure]
    leaky_exposure[-1] = _fake_exposure_row("root_candidate_grouped", 2, root_intersection_count=1)
    decision = classify(branches, leaky_exposure, **kwargs)
    assert decision["classification"] == "FAIL_MENTOR_GROUPED_RECONSTRUCTION_CONTRACT"

    # No branch passes optimization -> keep RAW/INIT baseline.
    weak_folds = [
        _fake_gate_fold(0.200, 0.199, test_graph_count=10),
        _fake_gate_fold(0.220, 0.219, test_graph_count=20),
        _fake_gate_fold(0.180, 0.179, test_graph_count=30),
    ]
    decision = classify(make_branches(weak_folds), exposure, **kwargs)
    assert decision["classification"] == "KEEP_RAW_PATCH_OR_INIT_BASELINE_NO_KSVD_GAIN"

    # Optimization passes but FINAL does not beat RANDOM -> no rate-distortion advantage.
    control_fail_folds = [
        _fake_gate_fold(0.30, 0.27, test_graph_count=10),
        _fake_gate_fold(0.32, 0.28, test_graph_count=20),
        _fake_gate_fold(0.28, 0.26, test_graph_count=30),
    ]
    decision = classify(make_branches(control_fail_folds), exposure, **kwargs)
    assert decision["classification"] == "KSVD_OPTIMIZES_BUT_NO_RATE_DISTORTION_ADVANTAGE"

    # Optimization and control both pass -> ready for relation-token ablation.
    decision = classify(make_branches(good_folds), exposure, **kwargs)
    assert decision["classification"] == "KSVD_GROUPED_RECONSTRUCTION_READY_FOR_RELATION_TOKEN_ABLATION"
    assert decision["ready_branches"] == ["s8_o2_BASE"]
    assert decision["pareto"] == ["s8_o2_BASE"]

    # A single-view run reports incomplete views and falls back to no-gain.
    single = {"random_reference": {"s8_o2_BASE": _fake_branch(good_folds)}}
    decision = classify(
        single,
        [row for row in exposure if row["view"] == "random_reference"],
        views=("random_reference",),
        geometries=("s8_o2",),
        checkpoints=("BASE",),
        n_folds=3,
    )
    assert not decision["views_complete"]
    assert decision["classification"] == "KEEP_RAW_PATCH_OR_INIT_BASELINE_NO_KSVD_GAIN"


def test_verify_pilot_counts():
    pilot = {
        "config": {"pilot_size": 500, "pilot_seed": 20260806, "cover_seed": 970201},
        "pilot_indices": [3, 7, 11],
        "checkpoint_rows": [
            {"source_graph_index": 3, "geometry": "s8_o2", "checkpoint": "BASE", "patch_count": 8},
            {"source_graph_index": 7, "geometry": "s8_o2", "checkpoint": "BASE", "patch_count": 39},
            {"source_graph_index": 11, "geometry": "s8_o2", "checkpoint": "BASE", "patch_count": 17},
        ],
    }
    args = SimpleNamespace(pilot_size=500, pilot_seed=20260806, cover_seed=970201)

    def prepared(patch_counts):
        result = {}
        for index, count in patch_counts.items():
            graph = PreparedGraph(
                source_index=index,
                pilot_position=0,
                density_stratum="lt5",
                average_degree=4.0,
                geometry="s8_o2",
                checkpoint="BASE",
                patch_count=count,
                ids=None,
                example=None,
            )
            result[index] = graph
        return {("s8_o2", "BASE"): result}

    selection = np.asarray([3, 7, 11], dtype=np.int64)
    verification = _verify_pilot_counts(args, pilot, selection, prepared({3: 8, 7: 39, 11: 17}), [])
    assert verification["checked"]
    assert verification["passed"], verification
    assert verification["compared"] == 3

    verification = _verify_pilot_counts(args, pilot, selection, prepared({3: 9, 7: 39, 11: 17}), [])
    assert not verification["passed"]
    assert verification["mismatch_count"] == 1
    assert verification["mismatches"][0]["source_index"] == 3

    # A different selection is not verified against the pilot payload.
    other_selection = np.asarray([1, 2, 3], dtype=np.int64)
    verification = _verify_pilot_counts(args, pilot, other_selection, prepared({3: 8, 7: 39, 11: 17}), [])
    assert not verification["passed"]
    assert not verification["pilot_indices_match"]

    # Non-matching pilot config skips verification.
    args.pilot_seed = 1
    verification = _verify_pilot_counts(args, pilot, selection, prepared({3: 8, 7: 39, 11: 17}), [])
    assert not verification["checked"]


def test_weighted_and_mean_stage_helpers():
    folds = [
        _fake_gate_fold(0.20, 0.18, test_graph_count=10),
        _fake_gate_fold(0.22, 0.19, test_graph_count=30),
    ]
    weighted = _weighted_stage_mean(folds, "FINAL", "uncorrected", "patch_relative_error")
    assert abs(weighted - (0.18 * 10 + 0.19 * 30) / 40) < 1e-12
    mean = _mean_stage(folds, "FINAL", "uncorrected")
    assert abs(mean["patch_relative_error"] - (0.18 + 0.19) / 2) < 1e-12
    # Skipped folds are excluded; an all-skipped fold set yields None.
    skipped = [dict(fold, skipped=True) for fold in folds]
    assert _weighted_stage_mean(skipped, "FINAL", "uncorrected", "patch_relative_error") is None


def main() -> int:
    test_raw_identity_and_fold_stages()
    test_random_dictionary_seed_deterministic()
    test_vector_overlap_counts()
    test_patch_source_node_mapping_uses_stable_order()
    test_mentor_folds_on_noncontiguous_indices()
    test_optimization_gate_pass_and_fail()
    test_control_gate()
    test_split_data_checks()
    test_classify_labels()
    test_verify_pilot_counts()
    test_weighted_and_mean_stage_helpers()
    print("mentor_subgraphs_grouped_ksvd_followup self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
