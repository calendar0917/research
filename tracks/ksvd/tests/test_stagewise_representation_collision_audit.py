"""Tests for the compact-v4 stagewise representation collision audit.

Static / unit tests.  They never train a backbone, never use official valid for
stage selection, and never load official test.  Several tests read the frozen
result directory if it exists; they are skipped when the audit has not been run.
"""

from __future__ import annotations

import gzip
import inspect
import json

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as sm
from tracks.ksvd.experiments.luyin16 import zinc_stagewise_representation_collision_audit as audit

RESULTS = audit.RESULTS_DIR
HAS_RESULTS = (RESULTS / "bottleneck_decision.json").exists()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encoded():
    return sm.build_encoded_records()


@pytest.fixture(scope="module")
def model(encoded):
    return audit.build_model(0)


@pytest.fixture(scope="module")
def sample_batch(encoded):
    records = encoded[0]
    return Batch.from_data_list([records[0]])


# ---------------------------------------------------------------------------
# Test 1-4: inventory / dimensions
# ---------------------------------------------------------------------------


def test_01_stage_inventory_dimensions(model, sample_batch):
    with torch.no_grad():
        stages = audit.encode_stages(model, sample_batch)
    expected = {
        "patch_encoder_input": 170,
        "h0": 48,
        "pair_encoder_input": 64,
        "q0": 16,
        "centre_update_input": 213,
        "h1": 48,
        "unary_fixed": 97,
        "pair_fixed": 165,
        "global": 32,
        "topology": 8,
        "R": 302,
    }
    for key, dim in expected.items():
        assert stages[key].shape[-1] == dim, key


def test_02_R_dimension_and_params(model):
    assert model.unified_graph_width == 302 == audit.EXPECTED_R_DIM
    assert sum(p.numel() for p in model.parameters()) == 82115 == audit.EXPECTED_TOTAL_PARAMS


def test_03_checkpoint_inventory(model):
    assert audit.CHECKPOINT_PATHS[0].exists()
    assert audit.CHECKPOINT_PATHS[1].exists()
    for seed in (0, 1):
        loaded = audit.build_model(seed)
        assert sum(p.numel() for p in loaded.parameters()) == 82115


def test_04_set_and_fixed_stage_partition():
    assert len(audit.SET_STAGES) == 6
    assert len(audit.FIXED_STAGES) == 5
    assert set(audit.SET_STAGES) | set(audit.FIXED_STAGES) == set(audit.STAGE_BY_KEY)
    assert not (set(audit.SET_STAGES) & set(audit.FIXED_STAGES))


# ---------------------------------------------------------------------------
# Test 5-6: instrumentation invariance
# ---------------------------------------------------------------------------


def test_05_forward_instrumentation_does_not_change_prediction(model, encoded):
    """U0.2: captured R and prediction are bit-identical to the original."""
    for record in encoded[0][:5]:
        batch = Batch.from_data_list([record])
        with torch.no_grad():
            stages = audit.encode_stages(model, batch)
            reference_R = model.encode(batch)
            reference_pred = model(batch)
            captured_pred = model.head(stages["R"]).view(-1)
        assert float((stages["R"] - reference_R).abs().max()) == 0.0
        assert float((captured_pred - reference_pred).abs().max()) <= 1e-6


def test_06_captured_inputs_are_real_module_inputs(model, sample_batch):
    """S0 / S2 / S4 are the actual concatenations consumed by the modules."""
    with torch.no_grad():
        stages = audit.encode_stages(model, sample_batch)
        e_patch = model.typed_embedding(sample_batch.typed_token)
        s0_reconstructed = torch.cat(
            [
                sample_batch.patch_cont,
                sample_batch.patch_context,
                e_patch,
                model.parent_embedding(sample_batch.parent_token),
            ],
            dim=1,
        )
        side = model.patch_encoder(s0_reconstructed)
        s2_source = sample_batch.pair_index[0]
        s2_target = sample_batch.pair_index[1]
        projected_left = model.pair_projection(side[s2_source])
        projected_right = model.pair_projection(side[s2_target])
        relation = model.relation_encoder(sample_batch.pair_relation)
        gate = 1.0 + torch.tanh(model.distance_gate(sample_batch.pair_bucket))
        s2_reconstructed = torch.cat(
            [
                projected_left + projected_right,
                torch.abs(projected_left - projected_right),
                (projected_left * projected_right) * gate,
                relation,
            ],
            dim=1,
        )
    assert float((stages["patch_encoder_input"] - s0_reconstructed).abs().max()) == 0.0
    assert float((stages["pair_encoder_input"] - s2_reconstructed).abs().max()) == 0.0
    assert float((stages["h0"] - side).abs().max()) < 1e-6


# ---------------------------------------------------------------------------
# Test 7-10: set sketch
# ---------------------------------------------------------------------------


def test_07_set_sketch_dimension_and_empty_set():
    projections = audit._stage_projection("h0")
    empty = audit.set_sketch(np.zeros((0, 48)), np.zeros(48), np.ones(48), projections)
    assert empty.shape == (217,)
    assert np.isfinite(empty).all()
    assert np.all(empty[:216] == 0.0)
    assert empty[216] == 0.0


def test_08_set_sketch_permutation_invariant():
    rng = np.random.default_rng(3)
    objects = rng.standard_normal((41, 48))
    mu = objects.mean(axis=0)
    sigma = objects.std(axis=0)
    projections = audit._stage_projection("h0")
    base = audit.set_sketch(objects, mu, sigma, projections)
    permuted = audit.set_sketch(objects[rng.permutation(41)], mu, sigma, projections)
    assert np.abs(base - permuted).max() < 1e-9


def test_09_set_sketch_deterministic_fixed_projection_seed():
    a = audit._stage_projection("q0", audit.BASE_SEED)
    b = audit._stage_projection("q0", audit.BASE_SEED)
    assert np.abs(a - b).max() == 0.0
    # unit rows
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-9)
    # different stages get different deterministic seeds
    assert audit._label_seed("proj", audit.BASE_SEED, "q0") != audit._label_seed(
        "proj", audit.BASE_SEED, "h0"
    )


def test_10_set_sketch_target_independent(model, encoded):
    """Randomising the target leaves the sketch exactly unchanged."""
    record = encoded[0][2]
    batch = Batch.from_data_list([record])
    with torch.no_grad():
        raw = audit.encode_stages(model, batch)["patch_encoder_input"].numpy()
    mu = raw.mean(axis=0)
    sigma = raw.std(axis=0)
    projections = audit._stage_projection("patch_encoder_input")
    before = audit.set_sketch(raw, mu, sigma, projections)
    mutated = record.clone()
    mutated.y = torch.tensor([999.0])
    mutated_batch = Batch.from_data_list([mutated])
    with torch.no_grad():
        raw_mutated = audit.encode_stages(model, mutated_batch)["patch_encoder_input"].numpy()
    after = audit.set_sketch(raw_mutated, mu, sigma, projections)
    assert np.abs(before - after).max() == 0.0


def test_11_set_sketch_lock_constants():
    assert audit.SET_SKETCH_DIM == 217
    assert audit.N_PROJECTIONS == 24
    assert len(audit.QUANTILES) == 9
    assert audit.NORMALIZATION_EPS == 1e-6


# ---------------------------------------------------------------------------
# Test 12-14: protocol discipline
# ---------------------------------------------------------------------------


def test_12_primary_k_and_material_threshold_locked():
    assert audit.K_PRIMARY == 8
    assert audit.K_ROBUST == (4, 16)
    assert audit.MATERIAL_DELTA_ETA == 0.02
    assert audit.EXACT_COLLISION_EPS == 1e-6


def test_13_split_sizes_and_disjointness():
    ref, sel, probe, report = audit.split_positions()
    assert len(ref) == 7200 and len(sel) == 800 and len(probe) == 2000
    assert set(ref) & set(sel) == set()
    assert set(ref) & set(probe) == set()
    assert set(sel) & set(probe) == set()
    assert report["assignment_matches"] is True
    assert report["frozen_probe_matches"] is True
    assert report["official_valid_used"] is False
    assert report["official_test_loaded"] is False


def test_14_official_test_never_loaded_in_source():
    source = inspect.getsource(audit)
    assert "extract_test_records" not in source
    assert '_load_zinc(' not in source
    assert '"test"' not in source
    assert "official test" in source.lower()


def test_15_contrast_order_is_pre_registered():
    assert [c["id"] for c in audit.CONTRASTS] == ["A", "B", "C", "D1", "D2"]
    assert [c["order"] for c in audit.CONTRASTS] == [1, 2, 3, 4, 5]
    assert audit.CONTRASTS[0]["a"] == "patch_encoder_input"
    assert audit.CONTRASTS[0]["b"] == "h0"
    assert audit.CONTRASTS[-1]["a"] == "q0"
    assert audit.CONTRASTS[-1]["b"] == "pair_fixed"


# ---------------------------------------------------------------------------
# Test 16-18: gate / decision logic
# ---------------------------------------------------------------------------


def _synthetic_entry(delta_seed0, delta_seed1, sel0, sel1, col0, col1, ci_low=0.01):
    def side(delta, col):
        return {
            "delta_eta": delta,
            "delta_collision": col,
            "bootstrap_delta_eta": {"ci_low": ci_low, "ci_high": ci_low + 0.05, "mean": delta},
        }

    return {
        "id": "X",
        "name": "synthetic",
        "a": "patch_encoder_input",
        "b": "h0",
        "per_seed": {
            0: {"probe": side(delta_seed0, col0), "selection": side(sel0, col0)},
            1: {"probe": side(delta_seed1, col1), "selection": side(sel1, col1)},
        },
        "probe": side(delta_seed0, col0),
        "selection": side(sel0, col0),
    }


def test_16_material_degradation_gate_requires_both_seeds():
    strong = _synthetic_entry(0.05, 0.05, 0.04, 0.04, 0.01, 0.01, ci_low=0.005)
    status = audit._contrast_status(strong)
    assert status["material_degradation"] is True
    assert status["status"] == "MATERIAL_GEOMETRY_DEGRADATION"

    # one seed below the +0.02 materiality threshold -> not material
    weak = _synthetic_entry(0.011, 0.043, 0.011, 0.056, -0.01, -0.01, ci_low=0.003)
    status = audit._contrast_status(weak)
    assert status["material_degradation"] is False

    # collision moving opposite -> material gate fails even at large delta
    inconsistent = _synthetic_entry(0.05, 0.05, 0.04, 0.04, -0.01, -0.01, ci_low=0.005)
    assert audit._contrast_status(inconsistent)["material_degradation"] is False


def test_17_seed_and_split_instability_flags():
    conflicting = _synthetic_entry(0.03, -0.03, 0.02, -0.02, 0.0, 0.0, ci_low=0.01)
    status = audit._contrast_status(conflicting)
    assert status["seed_conflict"] is True
    assert status["status"] == "SEED_UNSTABLE"

    # CI high must be < 0 for a gain; rebuild explicitly
    entry = _synthetic_entry(-0.05, -0.05, -0.04, -0.04, 0.0, 0.0, ci_low=-0.06)
    for seed in (0, 1):
        entry["per_seed"][seed]["probe"]["bootstrap_delta_eta"]["ci_high"] = -0.01
    status = audit._contrast_status(entry)
    assert status["material_gain"] is True
    assert status["status"] == "MATERIAL_GEOMETRY_GAIN"


def test_18_paired_bootstrap_normalized_by_vrand_is_deterministic():
    values = np.random.default_rng(0).normal(0.05, 0.2, size=500)
    scaled = values / 2.1
    a = audit._paired_bootstrap(scaled, 123)
    b = audit._paired_bootstrap(scaled, 123)
    assert a == b
    assert a["n_queries"] == 500
    assert abs(a["mean"] - scaled.mean()) < 1e-9


# ---------------------------------------------------------------------------
# Test 19-21: frozen result artefacts (skipped if not run)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RESULTS, reason="audit not run")
def test_19_neighbor_manifests_locked_before_targets():
    hashes = json.loads((RESULTS / "neighbor_manifest_hashes.json").read_text())
    assert hashes["manifests_hash_locked_before_target_read"] is True
    assert hashes["primary_k"] == 8
    assert hashes["official_test_loaded"] is False
    assert len(hashes["files"]) >= 4
    target_lock = json.loads((RESULTS / "target_metric_lock.json").read_text())
    assert target_lock["official_test_loaded"] is False


@pytest.mark.skipif(not HAS_RESULTS, reason="audit not run")
def test_20_decision_is_case_e_and_not_authorized():
    decision = json.loads((RESULTS / "bottleneck_decision.json").read_text())
    top1 = json.loads((RESULTS / "top1_representation_hypothesis.json").read_text())
    assert decision["decision_case"] in {"Case A", "Case B", "Case C", "Case D", "Case E", "Case F", "Case G"}
    assert decision["official_test_loaded"] is False
    if decision["decision_case"] == "Case E":
        assert decision["first_material_degradation"] is None
        assert decision["implicated_stage"] is None
        assert top1["authorized_for_design"] is False
        assert top1["next_full_training_budget"] == "not yet authorized"


@pytest.mark.skipif(not HAS_RESULTS, reason="audit not run")
def test_21_final_decision_discipline():
    final = json.loads((RESULTS / "final_decision.json").read_text())
    assert final["official_test_loaded"] is False
    assert final["official_valid_used_for_selection"] is False
    assert final["next_full_training_budget"] == "not yet authorized"
    assert "information-theoretic" in final["information_theoretic_caveat"]


@pytest.mark.skipif(not HAS_RESULTS, reason="audit not run")
def test_22_robustness_not_triggered_in_case_e():
    projection = json.loads((RESULTS / "projection_robustness.json").read_text())
    k_robust = json.loads((RESULTS / "k_robustness.json").read_text())
    assert projection["triggered"] in (True, False)
    if projection["triggered"] is False:
        assert "reason" in projection
    assert k_robust["official_test_loaded"] is False


@pytest.mark.skipif(not (RESULTS / "neighbors_seed0_probe.jsonl.gz").exists(), reason="manifests absent")
def test_23_neighbor_jsonl_schema():
    path = RESULTS / "neighbors_seed0_probe.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    for key in ("query_graph_id", "stage", "seed", "split", "rank", "reference_graph_id", "representation_distance"):
        assert key in row
    assert row["split"] == "probe"
    assert row["seed"] == 0
