"""Tests for the raw-graph -> patch-system sufficiency audit.

Static / unit tests.  They never train a neural module, never load official
valid, and never load official test.  Result-dependent tests are skipped when
the frozen result directory does not exist.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from tracks.ksvd.experiments.luyin16 import (
    zinc_raw_graph_patch_sufficiency_audit as audit,
)

RESULTS = audit.RESULTS_DIR
HAS_RESULTS = (RESULTS / "final_decision.json").exists()


# ---------------------------------------------------------------------------
# static protocol locks
# ---------------------------------------------------------------------------


def test_01_official_valid_and_test_never_loaded_in_source():
    source = inspect.getsource(audit)
    assert audit.zlr._load_zinc  # imported loader exists
    assert "_load_zinc(ZINC_ROOT, \"valid\")" not in source
    assert "_load_zinc(ZINC_ROOT, \"test\")" not in source
    assert "v4_records_valid" not in source
    assert "v4_records_test" not in source


def test_02_execution_flags_locked():
    assert audit.WL_ROUNDS == 4
    assert audit.K_PRIMARY == 8
    assert audit.K_ROBUST == (4, 16)
    assert audit.MATERIAL_LB_P3 == 0.002
    assert audit.MATERIAL_DELTA_PRE == 0.02
    assert audit.N_REF == 7200
    assert audit.N_PROBE == 2000
    assert audit.N_SELECTION == 800
    assert audit.PROJECTION_COUNT == 24
    assert len(audit.SKETCH_QUANTILES) == 9


def test_03_progressive_blocks_are_pre_registered():
    assert audit.PROGRESSIVE["PATCH_LOCAL"] == ("B1_identity", "B2_patch_numeric")
    assert audit.PROGRESSIVE["PATCH_PAIR"] == (
        "B1_identity",
        "B2_patch_numeric",
        "B3_pair_relation",
    )
    assert audit.PROGRESSIVE["PATCH_FULL"] == audit.PATCH_BLOCKS
    assert audit.PATCH_BLOCKS[0] == "B1_identity"
    assert audit.PATCH_BLOCKS[-1] == "B4_global_topology"


def test_04_no_trainable_neural_components():
    source = inspect.getsource(audit)
    forbidden = ("torch.optim", "nn.Module", "nn.Linear", "backward()", "optimizer.step")
    for token in forbidden:
        assert token not in source, token


def test_05_split_sizes_and_disjointness():
    ref, sel, probe, report = audit.split_positions()
    assert len(ref) == 7200 and len(sel) == 800 and len(probe) == 2000
    all_idx = np.concatenate([ref, sel, probe])
    assert len(np.unique(all_idx)) == 10000
    assert report["assignment_matches"] is True
    assert report["official_valid_loaded"] is False
    assert report["official_test_loaded"] is False


# ---------------------------------------------------------------------------
# canonicalisation machinery
# ---------------------------------------------------------------------------


def _random_colored_graph(rng, n):
    adjacency = {i: set() for i in range(n)}
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.45:
                adjacency[u].add(v)
                adjacency[v].add(u)
    colors = [("c", int(rng.integers(0, 3))) for _ in range(n)]
    return {k: sorted(v) for k, v in adjacency.items()}, colors


def test_06_canonical_key_relabel_invariance():
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = 9
        adjacency, colors = _random_colored_graph(rng, n)
        key0 = audit.colored_canonical_key(adjacency, colors)
        perm = rng.permutation(n)
        new_adj = {int(perm[u]): sorted(int(perm[v]) for v in adjacency[u]) for u in adjacency}
        new_colors = [None] * n
        for u in range(n):
            new_colors[int(perm[u])] = colors[u]
        assert audit.colored_canonical_key(new_adj, new_colors) == key0


def test_07_canonical_key_separates_different_colors():
    adjacency = {0: [1], 1: [0, 2], 2: [1]}
    key_a = audit.colored_canonical_key(adjacency, [("c", 0), ("c", 0), ("c", 0)])
    key_b = audit.colored_canonical_key(adjacency, [("c", 0), ("c", 1), ("c", 0)])
    assert key_a != key_b


def test_08_canonical_key_separates_structure():
    path = {0: [1], 1: [0, 2], 2: [1]}
    triangle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    colors = [("c", 0)] * 3
    assert audit.colored_canonical_key(path, colors) != audit.colored_canonical_key(
        triangle, colors
    )


def test_09_projected_quantile_sketch_dim_and_determinism():
    rng = np.random.default_rng(0)
    objects = [rng.normal(size=(5, 7)).astype(np.float32) for _ in range(12)]
    fit = np.arange(6)
    projections = rng.normal(size=(audit.PROJECTION_COUNT, 7))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    a, _ = audit.projected_quantile_sketch(objects, fit, projections, "t")
    b, _ = audit.projected_quantile_sketch(objects, fit, projections, "t")
    assert a.shape[1] == audit.PROJECTION_COUNT * len(audit.SKETCH_QUANTILES) + 1
    assert np.array_equal(a, b)


def test_10_progressive_distance_formula():
    block = {
        "B1_identity": np.array([[3.0, 0.0]], dtype=np.float32),
        "B2_patch_numeric": np.array([[4.0, 0.0]], dtype=np.float32),
        "B3_pair_relation": np.array([[0.0, 0.0]], dtype=np.float32),
        "B4_global_topology": np.array([[0.0, 0.0]], dtype=np.float32),
    }
    local = audit._progressive_patch_distance(block, ("B1_identity", "B2_patch_numeric"))
    assert local[0, 0] == pytest.approx(np.sqrt((9.0 + 16.0) / 2.0))


# ---------------------------------------------------------------------------
# frozen-result tests (skipped until the audit has run)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_11_final_decision_is_case_d_and_not_authorized():
    decision = json.loads((RESULTS / "final_decision.json").read_text())
    assert decision["case"] == "D"
    assert decision["verdict"] == "CURRENT_PRE_NEURAL_PATCH_SYSTEM_SUFFICIENCY_NOT_REFUTED"
    assert decision["authorized_for_design"] is False
    assert decision["full_training_authorized"] is False
    assert decision["official_valid_loaded"] is False
    assert decision["official_test_loaded"] is False


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_12_hard_lb_p3_below_material_gate():
    hard = json.loads((RESULTS / "hard_aliasing_target_summary.json").read_text())
    assert hard["per_level"]["P3"]["lb_alias"] < audit.MATERIAL_LB_P3
    assert hard["material_hard_aliasing_signal"] is False


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_13_p3_has_no_raw_nonisomorphic_collisions():
    summary = json.loads((RESULTS / "p3_signature_summary.json").read_text())
    assert summary["raw_isomorphism"]["n_raw_nonisomorphic_classes"] == 0


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_14_delta_pre_negative_and_ci_excludes_zero():
    soft = json.loads((RESULTS / "soft_geometry_bootstrap.json").read_text())
    assert soft["delta_pre"]["point"] < 0.0
    assert soft["delta_pre"]["ci_high"] < 0.0
    assert soft["soft_material_gate"]["passed"] is False


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_15_patch_full_beats_both_raw_references():
    soft = json.loads((RESULTS / "soft_geometry_bootstrap.json").read_text())
    robustness = soft["raw_reference_robustness"]
    assert robustness["patch_minus_wl"] < 0.0
    assert robustness["patch_minus_sp"] < 0.0
    assert robustness["raw_reference_unstable"] is False


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_16_manifests_locked_before_target():
    lock = json.loads((RESULTS / "neighbor_manifest_hashes.json").read_text())
    assert lock["locked_before_target_read"] is True
    for name, digest in lock["manifests"].items():
        assert (RESULTS / name).exists()
        assert len(digest) == 64


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_17_phase_u_integrity_passed():
    integrity = json.loads((RESULTS / "phaseU_integrity.json").read_text())
    assert integrity["passed"] is True
    assert integrity["U0.11_official_valid_not_loaded"] is True
    assert integrity["U0.12_official_test_not_loaded"] is True


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_18_integrity_tests_all_passed():
    integrity = json.loads((RESULTS / "integrity_tests.json").read_text())
    assert integrity["all_passed"] is True
    assert integrity["test3_canonical_relabel_invariance"]["passed"] is True
    assert integrity["test4_p3_collision_exact_isomorphism"]["passed"] is True
    assert integrity["inventory_completeness_passed"] is True


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_19_all_required_outputs_present():
    required = [
        "audit_protocol_lock.json",
        "raw_graph_inventory.json",
        "pre_neural_input_inventory.json",
        "split_inventory.json",
        "hard_signature_lock.json",
        "p0_signature_summary.json",
        "p1_signature_summary.json",
        "p2_signature_summary.json",
        "p3_signature_summary.json",
        "raw_graph_isomorphism_checks.json",
        "functional_collision_checks.json",
        "hard_aliasing_target_summary.json",
        "hard_aliasing_lower_bounds.json",
        "raw_wl_lock.json",
        "raw_sp_lock.json",
        "patch_soft_signature_lock.json",
        "normalization_stats.json",
        "distance_scale_stats.json",
        "neighbor_manifest_hashes.json",
        "phaseU_integrity.json",
        "target_metric_lock.json",
        "random_neighbor_baseline.json",
        "soft_geometry_metrics.csv",
        "soft_geometry_bootstrap.json",
        "collision_consistency.json",
        "knn_predictor_diagnostics.json",
        "final_decision.json",
        "top1_pre_neural_hypothesis.json",
        "answers_q1_q20.json",
    ]
    for name in required:
        assert (RESULTS / name).exists(), name
    for representation in ("raw", "wl", "sp", "patch_local", "patch_pair", "patch_full"):
        for split in ("probe", "selection"):
            assert (RESULTS / f"neighbors_{representation}_{split}.jsonl.gz").exists()


@pytest.mark.skipif(not HAS_RESULTS, reason="audit has not been run")
def test_20_top1_hypothesis_not_authorized():
    hypothesis = json.loads((RESULTS / "top1_pre_neural_hypothesis.json").read_text())
    assert hypothesis["authorized_for_design"] is False
    assert hypothesis["full_training_authorized"] is False
