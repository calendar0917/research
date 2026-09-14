"""Static / unit tests for the minimal recurrent residual-magnitude gate.

No training, no official test.  The checks mirror the pre-registered sanity
gates: exactly one added parameter, a single shared ``alpha == 0.5``, the gate
applied to every centre update, non-zero gate gradient, and an untouched ungated
path.
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
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_residual_gate as rg,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as sh
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


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
def ungated():
    model = rec.build_recurrent(0)
    model.eval()
    return model


@pytest.fixture(scope="module")
def gated():
    model = rec.build_gated(0)
    model.eval()
    return model


def _n_params(module):
    return int(sum(p.numel() for p in module.parameters()))


def _perturb_update(model):
    with torch.no_grad():
        for parameter in model.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    return model


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def test_01_gated_adds_exactly_one_parameter(ungated, gated):
    assert _n_params(ungated) == rg.EXPECTED_UNGATED_TOTAL == 82115
    assert _n_params(gated) == rg.EXPECTED_GATED_TOTAL == 82116
    assert _n_params(gated) - _n_params(ungated) == 1
    assert _n_params(gated.head) == rg.EXPECTED_HEAD == 4135


def test_02_gated_named_parameters_differ_only_by_gate(ungated, gated):
    u = {n for n, _ in ungated.named_parameters()}
    g = {n for n, _ in gated.named_parameters()}
    assert g - u == {"update_gate_logit"}
    assert not (u - g)
    assert list(dict(gated.named_parameters())["update_gate_logit"].shape) == []


def test_03_ungated_builder_is_unchanged(gated, ungated):
    assert ungated.update_gate_logit is None
    assert gated.update_gate_logit is not None
    assert rec.build_recurrent(0, gate_update_magnitude=False).update_gate_logit is None


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


def test_04_initial_alpha_is_one_half(gated):
    assert abs(float(gated.update_alpha()) - 0.5) < 1.0e-12
    assert float(gated.update_gate_logit) == 0.0


def test_05_shared_init_bit_identical(ungated, gated):
    u = ungated.state_dict()
    g = gated.state_dict()
    shared = [k for k in u if k in g and g[k].shape == u[k].shape]
    assert len(shared) == len(u)
    assert max(float((g[k] - u[k]).abs().max()) for k in shared) == 0.0


# ---------------------------------------------------------------------------
# gate semantics
# ---------------------------------------------------------------------------


def test_06_removing_gate_reproduces_ungated_bitwise(ungated, gated, batch):
    probe = rec.build_gated(0)
    probe.eval()
    probe.update_gate_logit = None  # deregister -> forward skips the gate
    with torch.no_grad():
        diff = float((probe.encode(batch) - ungated.encode(batch)).abs().max())
    assert diff == 0.0


def test_07_alpha_to_zero_collapses_to_baseline(batch):
    baseline = rec.build_baseline(0).eval()
    model = _perturb_update(rec.build_gated(0)).eval()
    with torch.no_grad():
        model.update_gate_logit.fill_(-30.0)
        assert float(model.update_alpha()) < 1.0e-12
        collapsed = model.encode(batch)
        reference = baseline.encode(batch)
    # alpha ~ 1e-13; a round-2-only ungated path would leave an O(U) gap.
    assert float((collapsed - reference).abs().max()) < 1.0e-9


def test_08_round1_update_scales_with_alpha(batch):
    model = _perturb_update(rec.build_gated(0)).eval()
    model.capture_diagnostics = True
    with torch.no_grad():
        model.update_gate_logit.fill_(0.0)
        model.encode(batch)
        d0 = (model.last_h1 - model.last_h0).clone()
        model.update_gate_logit.fill_(1.0)
        model.encode(batch)
        d1 = (model.last_h1 - model.last_h0).clone()
    expected = float(torch.sigmoid(torch.tensor(0.0)) / torch.sigmoid(torch.tensor(1.0)))
    observed = float(d0.abs().mean() / d1.abs().mean())
    assert abs(observed - expected) < 1.0e-5


def test_09_gate_gradient_is_nonzero_finite():
    model = _perturb_update(rec.build_gated(0))
    model.train()
    batch = _synthetic_batch()
    loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
    loss.backward()
    grad = model.update_gate_logit.grad
    assert grad is not None
    assert bool(torch.isfinite(grad))
    assert float(grad.abs()) > 0.0


def test_10_gated_forward_finite(gated, batch):
    with torch.no_grad():
        out = gated.encode(batch)
    assert bool(torch.isfinite(out).all())
    assert int(out.shape[1]) == 302


# ---------------------------------------------------------------------------
# scope / protocol
# ---------------------------------------------------------------------------


def test_11_single_gate_application_inside_round_loop():
    source = inspect.getsource(rec.PatchPathRecurrentPairCentreModel._encode_core)
    assert source.count("self.update_alpha()") == 1
    assert "for round_index in range(rounds)" in source
    assert rec.RECURRENCE_ROUNDS == 2
    assert rec.Q_DIM == 16


def test_12_protocol_inherited_verbatim():
    assert sh.OPTIMIZED_PROTOCOL["optimizer"] == "Adam"
    assert sh.OPTIMIZED_PROTOCOL["learning_rate"] == 1.0e-3
    assert sh.OPTIMIZED_PROTOCOL["weight_decay"] == 1.0e-5
    assert sh.OPTIMIZED_PROTOCOL["batch_size"] == 128
    assert sh.OPTIMIZED_PROTOCOL["max_epochs"] == 240
    assert sh.OPTIMIZED_PROTOCOL["patience"] == 40
    assert sh.OPTIMIZED_PROTOCOL["scheduler"] == "none"


def test_13_no_forbidden_modules_added():
    source = inspect.getsource(rec)
    for forbidden in (
        "nn.MultiheadAttention",
        "TransformerEncoderLayer",
        "nn.Transformer",
        "graphormer",
    ):
        assert forbidden not in source
    gate_source = inspect.getsource(rec.PatchPathRecurrentPairCentreModel)
    assert "channel" not in gate_source.lower()
    assert "sigmoid" in gate_source


def test_14_soup_average_includes_gate(tmp_path):
    epochs = []
    for index, value in enumerate((0.0, 2.0)):
        model = rec.build_gated(0)
        with torch.no_grad():
            model.update_gate_logit.fill_(value)
        path = tmp_path / f"epoch_{index}.pt"
        torch.save(model.state_dict(), path)
        epochs.append({"epoch": index, "path": str(path)})
    soup = rg.build_soup_state(epochs)
    # arithmetic mean of the gate logits is recovered by the locked soup rule
    assert "update_gate_logit" in soup
    assert abs(float(soup["update_gate_logit"]) - 1.0) < 1.0e-12
    assert set(soup) == set(rec.build_recurrent(0).state_dict()) | {
        "update_gate_logit"
    }
