"""Tests for the compact-v4 2-round weight-tied recurrent pair--centre variant.

These are static / unit tests for the pre-registered sanity gates.  They never
train to completion and never load official test.  A synthetic batch is used
for the mechanism checks; a small real batch is used only where the data cache
is already available.  The criterion under test is falsification cleanliness:
weight tying, true round-2 dependence on ``h^1``, mask/bucket consistency, and
zero added parameters.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as sh
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _synthetic_batch():
    topo_width = int(ztopo.raw_width("hinge"))
    datasets: list[Data] = []
    for graph in (
        ring_chords(6, []),
        ring_chords(5, []),
        from_edges(6, [(0, 1), (1, 2), (2, 0)]),
    ):
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
        datasets.append(
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
                y=torch.zeros(1),
                num_nodes=n,
            )
        )
    return next(iter(zpp._make_loader(datasets, 128, False, 0)))


@pytest.fixture(scope="module")
def batch():
    return _synthetic_batch()


@pytest.fixture(scope="module")
def baseline():
    model = rec.build_baseline(0)
    model.eval()
    return model


@pytest.fixture(scope="module")
def recurrent():
    model = rec.build_recurrent(0)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test 1-4: parameter accounting / architecture identity
# ---------------------------------------------------------------------------


def test_01_params_equal_baseline(recurrent, baseline):
    assert rec._n_params(recurrent) == rec._n_params(baseline) == 82115
    assert rec.EXPECTED_SMALL_TOTAL == 82115


def test_02_zero_delta_and_head_unchanged(recurrent, baseline):
    assert rec._n_params(recurrent) - rec._n_params(baseline) == 0
    assert rec._n_params(recurrent.head) == rec._n_params(baseline.head) == 4135


def test_03_state_dict_shapes_identical(recurrent, baseline):
    base = {k: tuple(v.shape) for k, v in baseline.state_dict().items()}
    cand = {k: tuple(v.shape) for k, v in recurrent.state_dict().items()}
    assert base == cand
    # No hidden dimension / head expansion sneaked in.
    assert int(recurrent.unified_graph_width) == 302
    assert int(recurrent.pair_hidden) == 16


# ---------------------------------------------------------------------------
# Test 5-6: initialisation and one-round equivalence
# ---------------------------------------------------------------------------


def test_04_shared_init_exact(recurrent, baseline):
    b = baseline.state_dict()
    r = recurrent.state_dict()
    shared = [k for k in b if k in r and r[k].shape == b[k].shape]
    assert len(shared) == len(b) == len(r)
    assert max(float((r[k] - b[k]).abs().max()) for k in shared) == 0.0


def test_05_one_round_bit_identical(recurrent, baseline, batch):
    with torch.no_grad():
        r_parent = baseline.encode(batch)
        r_orig = recurrent.encode_original(batch)
    assert float((r_parent - r_orig).abs().max()) == 0.0


def test_06_init_function_identical(recurrent, baseline, batch):
    # center_update is zero-initialised: the 2-round model must start from
    # exactly the baseline function.
    with torch.no_grad():
        r_parent = baseline.encode(batch)
        r_init = recurrent.encode(batch)
    assert float((r_parent - r_init).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# Test 7-8: weight tying
# ---------------------------------------------------------------------------


def test_07_weight_tying_module_reuse(recurrent, baseline, batch):
    counts_r = recurrent.module_call_counts(batch)
    counts_b = baseline.module_call_counts(batch) if hasattr(
        baseline, "module_call_counts"
    ) else None
    if counts_b is None:
        counts: dict[str, int] = {
            "pair_projection": 0,
            "pair_encoder": 0,
            "center_update": 0,
        }

        def _hook(name):
            def hook(_m, _i, _o):
                counts[name] += 1

            return hook

        handles = [
            baseline.pair_projection.register_forward_hook(_hook("pair_projection")),
            baseline.pair_encoder.register_forward_hook(_hook("pair_encoder")),
            baseline.center_update.register_forward_hook(_hook("center_update")),
        ]
        with torch.no_grad():
            baseline.encode(batch)
        for handle in handles:
            handle.remove()
        counts_b = counts
    assert counts_b == {"pair_projection": 2, "pair_encoder": 1, "center_update": 1}
    assert counts_r == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}


def test_08_no_new_trainable_tensors(recurrent, baseline):
    base_names = {n for n, _ in baseline.named_parameters()}
    cand_names = {n for n, _ in recurrent.named_parameters()}
    assert base_names == cand_names


# ---------------------------------------------------------------------------
# Test 9-11: round-2 genuinely depends on the updated centre state
# ---------------------------------------------------------------------------


def _perturbed_recurrent():
    model = rec.build_recurrent(0)
    model.eval()
    model.capture_diagnostics = True
    with torch.no_grad():
        for parameter in model.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    return model


def test_09_q1_recomputed_from_h1(batch):
    model = _perturbed_recurrent()
    model.encode(batch)
    assert model.last_h0 is not None and model.last_h1 is not None
    h0, h1, h2 = model.last_h0, model.last_h1, model.last_h2
    q0, q1 = model.last_q0, model.last_q1
    assert q0 is not None and q1 is not None and h2 is not None
    assert float((h1 - h0).abs().max()) > 1.0e-6
    assert float((h2 - h1).abs().max()) > 1.0e-6
    assert float((q1 - q0).abs().max()) > 1.0e-6
    grad = torch.autograd.grad(q1.sum(), h1, retain_graph=True)[0]
    assert float(grad.abs().max()) > 0.0


def test_10_zero_update_collapses_to_baseline(batch):
    baseline = rec.build_baseline(0)
    baseline.eval()
    model = _perturbed_recurrent()
    model.force_zero_center_update = True
    with torch.no_grad():
        r_collapsed = model.encode(batch)
        r_baseline = baseline.encode(batch)
    assert float((r_collapsed - r_baseline).abs().max()) == 0.0


def test_11_q1_differs_from_q0_at_init_when_update_active(batch):
    # At exact init the update is zero (q1 == q0, tested via identity).  Once
    # the update is active, q1 must move: this proves recomputation is real.
    model = _perturbed_recurrent()
    with torch.no_grad():
        model.encode(batch)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            model.last_q0, model.last_q1, dim=1, eps=1e-8
        ).mean()
    )
    assert cosine < 1.0 - 1.0e-6


# ---------------------------------------------------------------------------
# Test 12-13: mask / bucket / indexing consistency across rounds
# ---------------------------------------------------------------------------


def test_12_count_block_identical_across_rounds(batch):
    model = _perturbed_recurrent()
    with torch.no_grad():
        model.encode(batch)
    ctx0 = model.last_center_context0
    ctx1 = model.last_center_context1
    assert ctx0 is not None and ctx1 is not None
    width = 2 * rec.Q_DIM + 1
    count_columns = [b * width + 2 * rec.Q_DIM for b in range(zpp.DISTANCE_BUCKETS)]
    assert int(ctx0.shape[1]) == zpp.DISTANCE_BUCKETS * width
    assert bool(torch.isfinite(ctx0).all() and torch.isfinite(ctx1).all())
    assert float((ctx0[:, count_columns] - ctx1[:, count_columns]).abs().max()) == 0.0


def test_13_pair_index_and_bucket_ranges(batch):
    model = _perturbed_recurrent()
    with torch.no_grad():
        model.encode(batch)
    source = model.last_pair_source
    target = model.last_pair_target
    bucket = model.last_pair_bucket
    n_centres = int(model.last_h0.shape[0])
    assert int(source.min()) >= 0 and int(target.min()) >= 0
    assert int(source.max()) < n_centres and int(target.max()) < n_centres
    assert int(bucket.min()) >= 0 and int(bucket.max()) < zpp.DISTANCE_BUCKETS


# ---------------------------------------------------------------------------
# Test 14-16: scope / protocol / test-lock
# ---------------------------------------------------------------------------


def test_14_protocol_inherited_verbatim():
    assert rec.RECURRENCE_ROUNDS == 2
    assert rec.ARCH_GATE == 0.003
    assert sh.OPTIMIZED_PROTOCOL["max_epochs"] == 240
    assert sh.OPTIMIZED_PROTOCOL["patience"] == 40
    assert sh.OPTIMIZED_PROTOCOL["optimizer"] == "Adam"
    assert sh.OPTIMIZED_PROTOCOL["learning_rate"] == 1.0e-3


def test_15_no_forbidden_modules_added():
    source = inspect.getsource(rec)
    for forbidden in (
        "nn.MultiheadAttention",
        "TransformerEncoderLayer",
        "nn.Transformer",
        "path_features",
        "graphormer",
    ):
        assert forbidden not in source
    # The class must reuse the inherited pair/centre modules, not new ones.
    builder = inspect.getsource(rec.PatchPathRecurrentPairCentreModel._pair_value)
    for module in ("self.pair_projection", "self.pair_encoder"):
        assert module in builder


def test_16_official_test_only_in_terminal_stage():
    source = inspect.getsource(rec)
    pattern = "_load_zinc(ZINC_ROOT, " + '"test")'
    assert source.count(pattern) == 0
    # official test is loaded only through the explicit, gated terminal stage
    assert "def terminal_test()" in source
    assert "refusing official test" in source


# ---------------------------------------------------------------------------
# Test 17-19: depth-only stale control
# ---------------------------------------------------------------------------


def test_17_stale_control_matches_baseline_at_init(batch):
    baseline = rec.build_baseline(0)
    baseline.eval()
    stale = rec.build_stale(0)
    stale.eval()
    with torch.no_grad():
        assert float((baseline.encode(batch) - stale.encode(batch)).abs().max()) == 0.0


def test_18_stale_reuses_q0_after_perturbation(batch):
    stale = rec.build_stale(0)
    stale.eval()
    stale.capture_diagnostics = True
    with torch.no_grad():
        for parameter in stale.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    with torch.no_grad():
        stale.encode(batch)
    # In stale mode round 2 keeps q^(0): the stored final pair state is q0.
    assert float((stale.last_q1 - stale.last_q0).abs().max()) == 0.0
    assert float((stale.last_h2 - stale.last_h1).abs().max()) > 1.0e-6


def test_19_stale_control_is_parameter_neutral(batch):
    assert rec._n_params(rec.build_stale(0)) == rec._n_params(rec.build_baseline(0)) == 82115
    assert rec.RECURRENCE_ROUNDS == 2


# ---------------------------------------------------------------------------
# Test 20-25: one-shot late-refresh variant
# ---------------------------------------------------------------------------


def _perturbed_late_refresh():
    model = rec.build_late_refresh(0)
    model.eval()
    model.capture_diagnostics = True
    with torch.no_grad():
        for parameter in model.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    return model


def test_20_late_refresh_params_and_head():
    baseline = rec.build_baseline(0)
    late = rec.build_late_refresh(0)
    recurrent = rec.build_recurrent(0)
    assert rec._n_params(late) == rec._n_params(baseline) == 82115
    assert rec._n_params(late) == rec._n_params(recurrent)
    assert rec._n_params(late.head) == 4135
    assert late.recurrence_mode == "late_refresh"
    assert "late_refresh" in inspect.getsource(rec)


def test_21_late_refresh_shares_A0_and_refreshes_q1(batch):
    model = _perturbed_late_refresh()
    model.encode(batch)
    ctx0 = model.last_center_context0
    ctx1 = model.last_center_context1
    assert ctx0 is not None and ctx1 is not None
    # same object and same values -> one aggregate replayed twice
    assert ctx1 is ctx0
    assert float((ctx1 - ctx0).abs().max()) == 0.0
    h0, h1, h2 = model.last_h0, model.last_h1, model.last_h2
    q0, q1 = model.last_q0, model.last_q1
    assert h0 is not None and h1 is not None and h2 is not None
    assert q0 is not None and q1 is not None
    assert float((h1 - h0).abs().max()) > 1.0e-6
    assert float((h2 - h1).abs().max()) > 1.0e-6
    assert float((q1 - q0).abs().max()) > 1.0e-6
    # q1 is differentiable w.r.t. the final centre state h2
    grad = torch.autograd.grad(q1.sum(), h2, retain_graph=True)[0]
    assert float(grad.abs().max()) > 0.0


def test_22_late_refresh_centre_aggregation_called_once(batch):
    model = rec.build_late_refresh(0)
    model.eval()
    counter = {"n": 0}
    original = model._pool_pairs_to_centres

    def wrapped(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    model._pool_pairs_to_centres = wrapped
    try:
        with torch.no_grad():
            model.encode(batch)
    finally:
        del model._pool_pairs_to_centres
    assert counter["n"] == 1
    counts = model.module_call_counts(batch)
    assert counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}


def test_23_late_refresh_centre_path_equals_stale(batch):
    device = torch.device("cpu")
    stale = rec.build_stale(0).to(device).eval()
    late = rec.build_late_refresh(0).to(device).eval()
    stale.capture_diagnostics = True
    late.capture_diagnostics = True
    shared = {k: v.detach().clone() for k, v in stale.state_dict().items()}
    late.load_state_dict(shared)
    torch.manual_seed(20260912)
    noise = {
        k: 0.01 * torch.randn_like(v)
        for k, v in shared.items()
        if "center_update" in k
    }
    for model in (stale, late):
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if key in noise:
                    value.add_(noise[key])
    with torch.no_grad():
        stale.encode(batch)
        late.encode(batch)
    assert float((late.last_h1 - stale.last_h1).abs().max()) == 0.0
    assert float((late.last_h2 - stale.last_h2).abs().max()) == 0.0
    assert float((late.last_center_context0 - stale.last_center_context0).abs().max()) == 0.0
    assert float((late.last_q1 - stale.last_q1).abs().max()) > 1.0e-6


def test_24_late_refresh_collapses_with_zero_update(batch):
    baseline = rec.build_baseline(0)
    baseline.eval()
    model = _perturbed_late_refresh()
    model.force_zero_center_update = True
    with torch.no_grad():
        assert float((model.encode(batch) - baseline.encode(batch)).abs().max()) == 0.0


def test_25_late_refresh_mode_validation():
    with pytest.raises(ValueError):
        rec.build_recurrent(0, recurrence_mode="bogus")


# ---------------------------------------------------------------------------
# Test 26-31: direct shared-k pair-to-pair composition
# ---------------------------------------------------------------------------


def _active_pair_to_pair(seed: int = 0):
    model = rec.build_pair_to_pair(seed)
    model.eval()
    model.capture_diagnostics = True
    torch.manual_seed(20260913)
    with torch.no_grad():
        for parameter in model.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        for parameter in model.pair_to_pair_psi.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        for parameter in model.pair_to_pair_phi.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    return model


def test_26_pair_to_pair_parameter_accounting():
    baseline = rec.build_baseline(0)
    recurrent = rec.build_recurrent(0)
    composed = rec.build_pair_to_pair(0)
    added = rec._n_params(composed) - rec._n_params(recurrent)
    assert rec._n_params(baseline) == rec._n_params(recurrent) == 82115
    assert added == rec.pair_to_pair_param_count(0)
    assert 2000 <= added <= 5000
    assert rec._n_params(composed) < 90000
    assert rec._n_params(composed.head) == 4135
    # Only psi/phi are new; all shared tensor names are unchanged.
    assert set(recurrent.state_dict()).issubset(set(composed.state_dict()))


def test_27_pair_to_pair_zero_collapses_to_recurrent(batch):
    model = _active_pair_to_pair()
    recurrent = rec.build_recurrent(0)
    recurrent.eval()
    recurrent.load_state_dict(model.state_dict(), strict=False)
    model.force_zero_pair_to_pair = True
    with torch.no_grad():
        assert float((model.encode(batch) - recurrent.encode(batch)).abs().max()) == 0.0


def test_28_pair_to_pair_indexing_no_cross_graph_leak(batch):
    model = _active_pair_to_pair()
    model.encode(batch)
    left = model.last_pair_to_pair_left
    right = model.last_pair_to_pair_right
    target = model.last_pair_to_pair_target
    pair_batch = model.last_pair_batch
    assert left is not None and right is not None and target is not None
    assert int((pair_batch[target] != pair_batch[left]).sum()) == 0
    assert int((pair_batch[target] != pair_batch[right]).sum()) == 0


def test_29_pair_to_pair_endpoints_consistent(batch):
    model = _active_pair_to_pair()
    model.encode(batch)
    left = model.last_pair_to_pair_left
    right = model.last_pair_to_pair_right
    target = model.last_pair_to_pair_target
    source = model.last_pair_source
    pair_target = model.last_pair_target
    a = source[target]
    b = pair_target[target]
    ls, lt = source[left], pair_target[left]
    rs, rt = source[right], pair_target[right]
    left_has_a = (ls == a) | (lt == a)
    right_has_b = (rs == b) | (rt == b)
    left_other = torch.where(ls == a, lt, ls)
    right_other = torch.where(rs == b, rt, rs)
    ok = (
        left_has_a
        & right_has_b
        & (left_other == right_other)
        & (left_other != a)
        & (left_other != b)
    )
    assert bool(ok.all())


def test_30_pair_to_pair_q_star_depends_on_gathered_q(batch):
    model = _active_pair_to_pair()
    model.zero_grad(set_to_none=True)
    model.encode(batch)
    q_raw = model.last_q1_raw
    q_star = model.last_q1
    assert q_raw is not None and q_star is not None
    assert float((q_star - q_raw).abs().max()) > 1.0e-6
    grad = torch.autograd.grad((q_star - q_raw).sum(), q_raw, retain_graph=True)[0]
    assert float(grad.abs().max()) > 0.0


def test_31_pair_to_pair_shape_and_finite(batch):
    model = _active_pair_to_pair()
    with torch.no_grad():
        out = model.encode(batch)
    assert tuple(model.last_q1.shape) == tuple(model.last_q1_raw.shape)
    assert int(model.last_q1.shape[1]) == 16
    assert bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(model.last_q1).all())
