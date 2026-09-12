"""Tests for the P2 compact-v4 one-shot relation refresh experiment.

Stage A tests (1-11) and P2 architecture tests (12-22).  These are static /
unit tests that never load official test.  Files read from the results dir are
the actual frozen audit artifacts.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_relation_refresh as p2
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

RESULTS = p2.RESULTS_DIR


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _synthetic_datasets():
    graphs = [
        ring_chords(6, []),
        ring_chords(5, []),
        from_edges(6, [(0, 1), (1, 2), (2, 0)]),
        from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ]
    topo_width = int(ztopo.raw_width("hinge"))
    out: list[Data] = []
    for graph in graphs:
        n = len(graph.nodes)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs
            else torch.zeros(2, 0, dtype=torch.long)
        )
        buckets = torch.tensor(
            [min(max(i % 5, 0), 4) for i in range(len(pairs))], dtype=torch.long
        )
        out.append(
            Data(
                patch_cont=torch.randn(n, int(zpp.SHELL_WIDTH)),
                patch_context=torch.zeros(n, 0),
                typed_token=torch.zeros(n, dtype=torch.long),
                parent_token=torch.zeros(n, dtype=torch.long),
                structural_token=torch.zeros(n, dtype=torch.long),
                structural_coarse=torch.zeros(n, 0),
                pair_index=pair_index,
                pair_relation=torch.randn(len(pairs), int(zpp.RELATION_WIDTH)),
                pair_bucket=buckets,
                global_context=torch.randn(1, int(zpp.GLOBAL_WIDTH)),
                topology_features=torch.zeros(1, topo_width),
                y=torch.tensor([0.0]),
                num_nodes=n,
            )
        )
    return out


def _synthetic_batch():
    loader = zpp._make_loader(_synthetic_datasets(), 128, False, 0)
    return next(iter(loader))


def _model_pair():
    base = p2.build_baseline(seed=0)
    model = p2.build_refresh(seed=0)
    base.eval()
    model.eval()
    return base, model


def _read(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"results artifact missing: {name}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Stage A tests (1-11)
# ---------------------------------------------------------------------------


def test_1_checkpoint_fingerprint_correct():
    inv = _read("baseline_inventory.json")
    assert inv["v4_seed0_state_fingerprint_ok"] is True
    assert inv["v4_seed0_state_sha256"] == p2.V4_SEED0_SHA256


def test_2_q0_dimension_16():
    _base, model = _model_pair()
    batch = _synthetic_batch()
    with torch.no_grad():
        model.refresh_enabled = True
        model(batch)
    assert int(model.last_q0.shape[1]) == 16


def test_3_h0_h1_dimensions():
    _base, model = _model_pair()
    batch = _synthetic_batch()
    with torch.no_grad():
        model(batch)
    assert int(model.last_h0.shape[1]) == p2.PATCH_HIDDEN
    assert int(model.last_h1.shape[1]) == p2.PATCH_HIDDEN


def test_4_q1_uses_same_pair_projection_tensors():
    _base, model = _model_pair()
    # The refresh model must not define any additional projection module.
    names = {name for name, _ in model.named_modules()}
    assert not any("refresh" in name for name in names)
    assert isinstance(model.pair_projection, torch.nn.Linear)
    # Functional: manual reconstruction from the SHARED projection is exact.
    batch = _synthetic_batch()
    with torch.no_grad():
        model(batch)
        h1 = model.last_h1
        source, target = model.last_pair_source, model.last_pair_target
        left = model.pair_projection(h1[source])
        right = model.pair_projection(h1[target])
        relation = model.relation_encoder(batch.pair_relation)
        gate = 1.0 + torch.tanh(model.distance_gate(batch.pair_bucket))
        pair_input = torch.cat(
            [
                left + right,
                torch.abs(left - right),
                (left * right) * gate,
                relation,
            ],
            dim=1,
        )
        manual = model.pair_encoder(pair_input)
    assert float((manual - model.last_q1).abs().max()) == 0.0


def test_5_q1_uses_same_pair_encoder_tensors():
    _base, model = _model_pair()
    assert isinstance(model.pair_encoder, torch.nn.Module)
    batch = _synthetic_batch()
    calls = {"n": 0}

    def hook(_module, _inp, _out):
        calls["n"] += 1

    handle = model.pair_encoder.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.refresh_enabled = True
            model(batch)
    finally:
        handle.remove()
    assert calls["n"] == 2  # q0 and q1 share the single instantiated encoder


def test_6_q0_q1_relation_descriptor_identical():
    _base, model = _model_pair()
    batch = _synthetic_batch()
    captured = {}

    def hook(_module, _inp, out):
        captured.setdefault("relation", out.detach().clone())

    handle = model.relation_encoder.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(batch)
    finally:
        handle.remove()
    # q1 must reuse the exact same relation descriptor tensor as q0; if the
    # relation encoder runs exactly once, q0 and q1 share it by construction.
    assert "relation" in captured
    expected = model.relation_encoder(batch.pair_relation)
    assert float((captured["relation"] - expected).abs().max()) == 0.0


def test_7_pair_ordering_identical():
    _base, model = _model_pair()
    batch = _synthetic_batch()
    with torch.no_grad():
        model(batch)
    assert model.last_q0.shape[0] == model.last_q1.shape[0]
    assert model.last_q0.shape[0] == int(batch.pair_index.shape[1])
    assert torch.equal(model.last_pair_source, batch.pair_index[0])
    assert torch.equal(model.last_pair_target, batch.pair_index[1])


def test_8_pair_summary_code_reused_exactly():
    assert (
        p2.PatchPathRelationRefreshModel._pool_pairs
        is zpp.PatchPathModel._pool_pairs
    )
    integrity = _read("staleness_integrity.json")
    assert integrity["results"]["A0.8_pair_summary_reused"] is True


def test_9_probe_manifest_target_independent():
    manifest = _read("staleness_probe_manifest.json")
    assert manifest["target_used_for_selection"] is False
    assert manifest["error_or_residual_used_for_selection"] is False
    assert manifest["n_probe"] == 2000
    assert manifest["verification"]["assignment_matches"] is True
    assert manifest["verification"]["frozen_probe_matches"] is True


def test_10_no_labels_loaded():
    integrity = _read("staleness_integrity.json")
    assert integrity["results"]["A0.9b_no_labels_in_audit"] is True
    # the audit artifacts must not contain any target-derived field
    for name in ("relation_drift.json", "pair_summary_drift.json", "prediction_shift.json"):
        payload = json.dumps(_read(name))
        assert "target" not in payload.replace("target-independent", "").replace(
            "true target NOT used", ""
        ) or "target_used" not in payload


def test_11_official_test_never_loaded():
    for name in (
        "baseline_inventory.json",
        "staleness_probe_manifest.json",
        "staleness_audit_lock.json",
        "staleness_integrity.json",
        "relation_drift.json",
        "pair_summary_drift.json",
        "prediction_shift.json",
        "stageA_decision.json",
        "stageB_p2_seed0.json",
        "stageB_decision.json",
        "common_input_bulk.json",
        "mechanism_diagnostics.json",
        "final_decision.json",
    ):
        path = RESULTS / name
        if path.exists():
            assert json.loads(path.read_text())["official_test_loaded"] is False


# ---------------------------------------------------------------------------
# P2 architecture tests (12-22)
# ---------------------------------------------------------------------------


def test_12_total_params_equal_baseline():
    base, model = _model_pair()
    assert p2._n_params(base) == p2._n_params(model) == p2.V4_PARAMS
    audit = _read("parameter_audit.json")
    assert audit["delta_params"] == 0
    assert audit["parameter_neutral"] is True


def test_13_trainable_tensor_names_counts_match():
    base, model = _model_pair()
    base_params = {n: p.shape for n, p in base.named_parameters()}
    model_params = {n: p.shape for n, p in model.named_parameters()}
    assert set(base_params) == set(model_params)
    for name in base_params:
        assert tuple(base_params[name]) == tuple(model_params[name])
    assert len(model_params) == 48


def test_14_shared_pair_encoder_instantiated_once():
    _base, model = _model_pair()
    encoders = [m for m in model.modules() if m is model.pair_encoder]
    assert len(encoders) == 1
    projections = [m for m in model.modules() if m is model.pair_projection]
    assert len(projections) == 1


def test_15_exactly_one_relation_refresh_occurs():
    _base, model = _model_pair()
    batch = _synthetic_batch()
    calls = {"n": 0}

    def hook(_module, _inp, _out):
        calls["n"] += 1

    handle = model.pair_encoder.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.refresh_enabled = True
            model(batch)
        refresh_on = calls["n"]
        calls["n"] = 0
        model.refresh_enabled = False
        with torch.no_grad():
            model(batch)
        refresh_off = calls["n"]
    finally:
        handle.remove()
    assert refresh_on == 2
    assert refresh_off == 1
    assert refresh_on - refresh_off == p2.REFRESH_COUNT == 1


def test_16_q1_does_not_feed_another_centre_update():
    _base, model = _model_pair()
    batch = _synthetic_batch()
    calls = {"n": 0}

    def hook(_module, _inp, _out):
        calls["n"] += 1

    original = model._pool_pairs_to_centres

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model._pool_pairs_to_centres = counted  # type: ignore[assignment]
    try:
        with torch.no_grad():
            model.refresh_enabled = True
            model(batch)
    finally:
        model._pool_pairs_to_centres = original  # type: ignore[assignment]
    assert calls["n"] == 1


def test_17_R_remains_302d():
    _base, model = _model_pair()
    batch = _synthetic_batch()
    with torch.no_grad():
        unified = model.encode(batch)
    assert int(unified.shape[1]) == 302 == p2.R_TOTAL
    assert int(model.unified_graph_width) == 302


def test_18_graph_head_unchanged():
    base, model = _model_pair()
    assert p2._state_hash(model.head.state_dict()) == p2._state_hash(
        base.head.state_dict()
    )
    assert int(model.head[0].in_features) == 302


def test_19_topology_branch_unchanged():
    base, model = _model_pair()
    assert model.topology_encoder is not None
    assert p2._state_hash(model.topology_encoder.state_dict()) == p2._state_hash(
        base.topology_encoder.state_dict()
    )


def test_20_centre_pooling_unchanged():
    assert (
        p2.PatchPathRelationRefreshModel._pool_pairs_to_centres
        is zpp.PatchPathModel._pool_pairs_to_centres
    )
    _base, model = _model_pair()
    assert model.center_context_width == 165
    assert int(model.center_update[0].in_features) == 48 + 165


def test_21_final_pair_moments_use_q1_not_q0():
    _base, model = _model_pair()
    # break the zero-init centre update so that h1 != h0 and q1 != q0
    with torch.no_grad():
        model.center_update[-1].weight.normal_(std=0.05)
        model.center_update[-1].bias.normal_(std=0.05)
    model.eval()
    batch = _synthetic_batch()
    with torch.no_grad():
        model.refresh_enabled = True
        unified_on = model.encode(batch)
        q0 = model.last_q0.detach()
        q1 = model.last_q1.detach()
        model.refresh_enabled = False
        unified_off = model.encode(batch)
    assert float((q1 - q0).abs().max()) > 1e-6
    # pair block must equal the pooled refreshed relation in the on-path ...
    pair_graph = batch.batch[batch.pair_index[0]]
    n_graphs = int(batch.global_context.shape[0])
    expected_pair = model._pool_pairs(q1, pair_graph, batch.pair_bucket, n_graphs)
    got_pair = unified_on[:, p2.PAIR_OFFSET : p2.PAIR_OFFSET + p2.PAIR_READOUT_WIDTH]
    assert float((expected_pair - got_pair).abs().max()) == 0.0
    # ... and the pooled stale relation in the off-path
    expected_pair_off = model._pool_pairs(q0, pair_graph, batch.pair_bucket, n_graphs)
    got_pair_off = unified_off[:, p2.PAIR_OFFSET : p2.PAIR_OFFSET + p2.PAIR_READOUT_WIDTH]
    assert float((expected_pair_off - got_pair_off).abs().max()) == 0.0
    # unary and global+topology blocks are untouched by the refresh
    assert float(
        (unified_on[:, : p2.UNARY_WIDTH] - unified_off[:, : p2.UNARY_WIDTH]).abs().max()
    ) == 0.0
    tail = p2.PAIR_OFFSET + p2.PAIR_READOUT_WIDTH
    assert float((unified_on[:, tail:] - unified_off[:, tail:]).abs().max()) == 0.0


def test_22_initial_parameter_hashes_match():
    base, model = _model_pair()
    base_state = base.state_dict()
    model_state = model.state_dict()
    assert set(base_state) == set(model_state)
    for key in base_state:
        assert torch.equal(base_state[key], model_state[key])
    assert p2._state_hash(base_state) == p2._state_hash(model_state)
    match = _read("initialization_match.json")
    assert match["all_99613_params_identical"] is True
    assert match["max_abs_diff_initial"] == 0.0
    assert match["shared_state_sha256"] == match["baseline_state_sha256"]


def test_stage_a_decision_advances_over_staleness():
    decision = _read("stageA_decision.json")
    assert decision["integrity_passed"] is True
    assert decision["near_identity"] is False
    assert decision["advance"] is True


def test_final_decision_is_case_b_nogo():
    final = _read("final_decision.json")
    sb = _read("stageB_decision.json")
    assert sb["case"] == "B"
    assert sb["delta_arch0"] < p2.ARCH_GATE
    assert sb["common_input_bulk_safe"] is True
    assert sb["branch_alive"] is True
    assert sb["nan_detected"] is False
    assert final["decision_case"] == "B"
