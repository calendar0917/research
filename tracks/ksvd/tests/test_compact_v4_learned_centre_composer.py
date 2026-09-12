"""Tests for the P1 compact-v4 learned relation-to-centre composer experiment.

Architecture tests (1-15) and conditional capacity-control tests (16-21) from
the P1 task.  These are static / unit tests; they never load official test.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_learned_centre_composer as p1
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_synthetic_graphs():
    return [
        ring_chords(6, []),
        ring_chords(5, []),
        from_edges(6, [(0, 1), (1, 2), (2, 0)]),
        from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ]


def _synthetic_datasets():
    topo_width = int(ztopo.raw_width("hinge"))
    out: list[Data] = []
    for graph in _make_synthetic_graphs():
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
                patch_cont=torch.zeros(n, int(zpp.SHELL_WIDTH)),
                patch_context=torch.zeros(n, 0),
                typed_token=torch.zeros(n, dtype=torch.long),
                parent_token=torch.zeros(n, dtype=torch.long),
                structural_token=torch.zeros(n, dtype=torch.long),
                structural_coarse=torch.zeros(n, 0),
                pair_index=pair_index,
                pair_relation=torch.zeros(len(pairs), int(zpp.RELATION_WIDTH)),
                pair_bucket=buckets,
                global_context=torch.zeros(1, int(zpp.GLOBAL_WIDTH)),
                topology_features=torch.zeros(1, topo_width),
                y=torch.tensor([0.0]),
                num_nodes=n,
            )
        )
    return out


def _synthetic_batch():
    loader = zpp._make_loader(_synthetic_datasets(), 128, False, 0)
    return next(iter(loader))


def _pool_inputs(n_centres=7, n_pairs=40, seed=20260911):
    gen = torch.Generator().manual_seed(int(seed))
    return (
        torch.randn(n_pairs, p1.Q_DIM, generator=gen),
        torch.randint(0, n_centres, (n_pairs,), generator=gen),
        torch.randint(0, n_centres, (n_pairs,), generator=gen),
        torch.randint(0, p1.N_BUCKETS, (n_pairs,), generator=gen),
        int(n_centres),
    )


@pytest.fixture(scope="module")
def p1_model():
    model = p1.build_p1(seed=0)
    model.eval()
    return model


@pytest.fixture(scope="module")
def control_model():
    model = p1.build_control(seed=0)
    model.eval()
    return model


@pytest.fixture(scope="module")
def baseline_model():
    model = p1.build_baseline(seed=0)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# architecture tests (1-15)
# ---------------------------------------------------------------------------


def test_01_baseline_fixed_pooling_unchanged(baseline_model):
    """Test 1 / G0.2: the historical fixed pooling is byte-for-byte reused."""
    value, source, target, bucket, n = _pool_inputs()
    with torch.no_grad():
        out = zpp.PatchPathModel._pool_pairs_to_centres(
            baseline_model, value, source, target, bucket, n
        )
    assert tuple(out.shape) == (n, p1.CENTRE_CONTEXT_WIDTH)
    assert p1.CENTRE_CONTEXT_WIDTH == 165
    # the P1 module must delegate to the same parent implementation
    src = inspect.getsource(p1.PatchPathComposerModel._pool_pairs_to_centres)
    assert "super()._pool_pairs_to_centres(" in src


def test_02_p1_disabled_equals_baseline(p1_model, baseline_model):
    """Test 2 / G0.1: with the residual zero/disabled P1 == baseline forward."""
    state = p1_model.state_dict()
    base_state = baseline_model.state_dict()
    shared = {
        key: val
        for key, val in state.items()
        if key in base_state and base_state[key].shape == val.shape
    }
    baseline_model.load_state_dict(shared, strict=False)
    batch = _synthetic_batch()
    with torch.no_grad():
        p1_out = p1_model.encode(batch)
        base_out = baseline_model.encode(batch)
        p1_model.composer_enabled = False
        p1_disabled = p1_model.encode(batch)
        p1_model.composer_enabled = True
    assert float((p1_out - base_out).abs().max()) == 0.0
    assert float((p1_disabled - base_out).abs().max()) == 0.0


def test_03_q_dimension_16(p1_model):
    assert int(p1_model.pair_hidden) == 16
    assert p1.Q_DIM == 16
    assert int(p1_model.phi[0].in_features) == 16


def test_04_each_bucket_fixed_summary_33(baseline_model):
    assert p1.FIXED_BUCKET_WIDTH == 33
    value, source, target, bucket, n = _pool_inputs()
    with torch.no_grad():
        out = zpp.PatchPathModel._pool_pairs_to_centres(
            baseline_model, value, source, target, bucket, n
        )
    for b in range(p1.N_BUCKETS):
        block = out[:, b * 33 : (b + 1) * 33]
        assert block.shape[1] == 33


def test_05_p1_bucket_summary_33(p1_model):
    value, source, target, bucket, n = _pool_inputs()
    with torch.no_grad():
        out = p1_model._pool_pairs_to_centres(value, source, target, bucket, n)
    assert out.shape[1] == p1.CENTRE_CONTEXT_WIDTH
    for b in range(p1.N_BUCKETS):
        assert out[:, b * 33 : (b + 1) * 33].shape[1] == 33


def test_06_five_buckets_165(p1_model):
    assert p1.N_BUCKETS == 5
    assert int(p1_model.center_context_width) == 165


def test_07_r_remains_302(p1_model):
    assert int(p1_model.unified_graph_width) == 302
    assert p1.R_P1 == 302
    batch = _synthetic_batch()
    with torch.no_grad():
        r = p1_model.encode(batch)
    assert r.shape[1] == 302


def test_08_graph_head_unchanged(p1_model, baseline_model):
    assert p1._state_hash(p1_model.head.state_dict()) == p1._state_hash(
        baseline_model.head.state_dict()
    )
    assert int(p1_model.head[0].in_features) == 302


def test_09_topology_branch_unchanged(p1_model, baseline_model):
    assert p1_model.topology_encoder is not None
    assert p1._state_hash(p1_model.topology_encoder.state_dict()) == p1._state_hash(
        baseline_model.topology_encoder.state_dict()
    )


def test_10_total_params_under_ceiling(p1_model, control_model):
    p1_total = sum(p.numel() for p in p1_model.parameters())
    control_total = sum(p.numel() for p in control_model.parameters())
    assert p1_total <= p1.PARAM_CEILING
    assert control_total <= p1.PARAM_CEILING
    assert p1_total == 102601


def test_11_relation_order_permutation_invariance(p1_model):
    value, source, target, bucket, n = _pool_inputs()
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        ref = p1_model._pool_pairs_to_centres(value, source, target, bucket, n)
        for _ in range(5):
            perm = torch.randperm(value.shape[0], generator=gen)
            out = p1_model._pool_pairs_to_centres(
                value[perm], source[perm], target[perm], bucket[perm], n
            )
            assert float((out - ref).abs().max()) < 1e-6


def test_12_empty_bucket_learned_residual_zero(p1_model, baseline_model):
    gen = torch.Generator().manual_seed(11)
    n, n_pairs = 6, 20
    value = torch.randn(n_pairs, p1.Q_DIM, generator=gen)
    source = torch.randint(0, n, (n_pairs,), generator=gen)
    target = torch.randint(0, n, (n_pairs,), generator=gen)
    bucket = torch.randint(0, p1.N_BUCKETS, (n_pairs,), generator=gen)
    bucket = torch.where(bucket == 2, torch.full_like(bucket, 3), bucket)  # bucket 2 empty
    with torch.no_grad():
        p1_out = p1_model._pool_pairs_to_centres(value, source, target, bucket, n)
        fixed = zpp.PatchPathModel._pool_pairs_to_centres(
            baseline_model, value, source, target, bucket, n
        )
    assert float((p1_out[:, 2 * 33 : 3 * 33] - fixed[:, 2 * 33 : 3 * 33]).abs().max()) == 0.0
    assert float(fixed[:, 2 * 33 : 3 * 33].abs().max()) == 0.0


def test_13_phi_rho_shared_across_buckets(p1_model):
    phis = [m for n, m in p1_model.named_modules() if n == "phi"]
    rhos = [m for n, m in p1_model.named_modules() if n == "rho"]
    assert len(phis) == 1 and len(rhos) == 1
    # activate rho (zero-init final) so the composer is non-degenerate, then
    # perturb the single shared phi and check that all non-empty bucket blocks
    # of the composer output change.
    value, source, target, bucket, n = _pool_inputs()
    rho_backup = p1_model.rho[-1].weight.detach().clone()
    phi_backup = p1_model.phi[0].weight.detach().clone()
    with torch.no_grad():
        p1_model.rho[-1].weight.normal_(mean=0.0, std=0.1, generator=torch.Generator().manual_seed(3))
        ref = p1_model._pool_pairs_to_centres(value, source, target, bucket, n)
        p1_model.phi[0].weight.add_(0.5)
        perturbed = p1_model._pool_pairs_to_centres(value, source, target, bucket, n)
        p1_model.phi[0].weight.copy_(phi_backup)
        p1_model.rho[-1].weight.copy_(rho_backup)
    for b in range(p1.N_BUCKETS):
        block_delta = float(
            (perturbed[:, b * 33 : (b + 1) * 33] - ref[:, b * 33 : (b + 1) * 33])
            .abs()
            .max()
        )
        if bool((bucket == b).any()):
            assert block_delta > 0.0


def test_14_exactly_one_q_stage_no_relation_refresh(p1_model):
    src = inspect.getsource(zpp.PatchPathModel.encode)
    assert src.count("self.pair_encoder(") == 1
    composer_src = inspect.getsource(p1.PatchPathComposerModel._pool_pairs_to_centres)
    assert "pair_encoder" not in composer_src


def test_15_official_test_never_loaded():
    src = inspect.getsource(p1)
    assert '"test"' not in src
    assert "'test'" not in src
    assert '_load_zinc(ZINC_ROOT, "test")' not in src


# ---------------------------------------------------------------------------
# conditional capacity-control tests (16-21)
# ---------------------------------------------------------------------------


def test_16_p1_control_added_params_match(p1_model, control_model):
    base_total = sum(p.numel() for p in p1.build_baseline(seed=0).parameters())
    p1_added = sum(p.numel() for p in p1_model.parameters()) - base_total
    control_added = sum(p.numel() for p in control_model.parameters()) - base_total
    mismatch = abs(p1_added - control_added) / p1_added
    assert mismatch <= p1.CONTROL_PARAM_MISMATCH_TOL
    assert p1_added == 2988 and control_added == 2981


def test_17_control_cannot_access_individual_q(control_model):
    assert not hasattr(control_model, "phi")
    src = inspect.getsource(p1.PatchPathCapacityControlModel._pool_pairs_to_centres)
    # only the fixed summary (and its log-count) is read; individual q rows are
    # never indexed.
    assert "self.phi" not in src
    assert "value[" not in src
    assert "source[" not in src
    assert "target[" not in src


def test_18_control_shared_across_buckets(control_model):
    psis = [m for n, m in control_model.named_modules() if n == "psi"]
    assert len(psis) == 1
    value, source, target, bucket, n = _pool_inputs()
    psi0_backup = control_model.psi[0].weight.detach().clone()
    psi_last_backup = control_model.psi[-1].weight.detach().clone()
    with torch.no_grad():
        control_model.psi[-1].weight.normal_(mean=0.0, std=0.1, generator=torch.Generator().manual_seed(4))
        ref = control_model._pool_pairs_to_centres(value, source, target, bucket, n)
        control_model.psi[0].weight.add_(0.5)
        perturbed = control_model._pool_pairs_to_centres(value, source, target, bucket, n)
        control_model.psi[0].weight.copy_(psi0_backup)
        control_model.psi[-1].weight.copy_(psi_last_backup)
    for b in range(p1.N_BUCKETS):
        block_delta = float(
            (perturbed[:, b * 33 : (b + 1) * 33] - ref[:, b * 33 : (b + 1) * 33])
            .abs()
            .max()
        )
        if bool((bucket == b).any()):
            assert block_delta > 0.0


def test_19_p1_control_shared_tensors_initialize_identically():
    match = p1.initialization_match()
    assert match["p1"]["exact_equal_to_seed0_baseline"]
    assert match["control"]["exact_equal_to_seed0_baseline"]
    assert match["p1"]["max_abs_diff_initial"] == 0.0
    assert match["control"]["max_abs_diff_initial"] == 0.0


def test_20_p1_control_data_order_fingerprint_identical():
    data = _synthetic_datasets()
    loader_p1 = zpp._make_loader(data, 128, True, 0 + 91011)
    loader_control = zpp._make_loader(data, 128, True, 0 + 91011)
    b1 = next(iter(loader_p1))
    b2 = next(iter(loader_control))
    assert torch.equal(b1.batch, b2.batch)
    assert int(b1.num_nodes) == int(b2.num_nodes)


def test_21_control_empty_bucket_residual_zero(control_model, baseline_model):
    gen = torch.Generator().manual_seed(23)
    n, n_pairs = 6, 20
    value = torch.randn(n_pairs, p1.Q_DIM, generator=gen)
    source = torch.randint(0, n, (n_pairs,), generator=gen)
    target = torch.randint(0, n, (n_pairs,), generator=gen)
    bucket = torch.randint(0, p1.N_BUCKETS, (n_pairs,), generator=gen)
    bucket = torch.where(bucket == 4, torch.full_like(bucket, 1), bucket)  # bucket 4 empty
    with torch.no_grad():
        control_out = control_model._pool_pairs_to_centres(
            value, source, target, bucket, n
        )
        fixed = zpp.PatchPathModel._pool_pairs_to_centres(
            baseline_model, value, source, target, bucket, n
        )
    assert float((control_out[:, 4 * 33 : 5 * 33] - fixed[:, 4 * 33 : 5 * 33]).abs().max()) == 0.0


def test_22_official_test_lock_flag():
    """The stage-0 gates must record that official test was never loaded."""
    from pathlib import Path

    gate_path = Path(p1.RESULTS_DIR) / "integrity_gates.json"
    if gate_path.exists():
        import json

        gates = json.loads(gate_path.read_text())
        assert gates["passed"] is True
        assert gates["official_test_loaded"] is False
