"""Targeted tests for SDPK-v0 (static dictionary-conditioned pair kernel).

Static/CPU-only, synthetic ``Data``, never loads ZINC data, checkpoints,
official valid, or official test.  The tests enforce the non-negotiable
strict-static contract and the SDPK-specific claims:

* A. architecture: ``center_context=False``, ``center_update is None``, forward
  never calls ``_pool_pairs_to_centres``;
* B. no pair -> local feedback: ``h_i`` / ``alpha_i`` / ``c_i`` bit-identical
  under pair mutation, while the prediction does change (non-vacuous);
* C. every pair is evaluated exactly once; the dictionary is evaluated once;
* D. permutation invariance (pair order, node relabel);
* E. gradient viability of the dictionary coordinates, the dictionary gate and
  the widened pair kernel on a real mini-batch;
* F. no local residual / gamma; dictionary coordinates are a pure function of
  the local state; the dictionary intervention actually shifts predictions.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair_kernel as sdpl


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------


def ztopo_width() -> int:
    from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

    return int(ztopo.raw_width("hinge"))


def _graph(n: int, seed: int) -> Data:
    generator = torch.Generator().manual_seed(int(seed))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.zeros(2, 0, dtype=torch.long)
    )
    return Data(
        patch_cont=torch.randn(n, int(zpp.SHELL_WIDTH), generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.randint(0, sdpl.TYPED_VOCAB_SIZE, (n,), generator=generator),
        parent_token=torch.randint(0, sdpl.PARENT_VOCAB_SIZE, (n,), generator=generator),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 0),
        pair_index=pair_index,
        pair_relation=torch.randn(len(pairs), int(zpp.RELATION_WIDTH), generator=generator),
        pair_bucket=torch.randint(0, zpp.DISTANCE_BUCKETS, (len(pairs),), generator=generator),
        global_context=torch.randn(1, int(zpp.GLOBAL_WIDTH), generator=generator),
        topology_features=torch.randn(1, ztopo_width(), generator=generator),
        y=torch.randn(1, generator=generator),
        num_nodes=n,
    )


@pytest.fixture(scope="module")
def batch():
    graphs = [_graph(n, seed=n) for n in (7, 6, 8)]
    return next(iter(zpp._make_loader(graphs, 8, False, 0)))


@pytest.fixture(scope="module")
def single_batch():
    return next(iter(zpp._make_loader([_graph(7, seed=7)], 8, False, 0)))


@pytest.fixture(scope="module")
def model():
    return sdpl.build_sdpl(0)


# ---------------------------------------------------------------------------
# A. architecture
# ---------------------------------------------------------------------------


def test_a_center_context_disabled(model):
    assert model.center_context is False
    assert model.center_update is None
    assert model.center_context_width == 0


def test_a_forward_never_calls_pool_pairs_to_centres(model, batch, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("strict-static violation: _pool_pairs_to_centres called")

    monkeypatch.setattr(zpp.PatchPathModel, "_pool_pairs_to_centres", _boom)
    model.eval()
    with torch.no_grad():
        prediction = model(batch)
    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()


def test_a_unified_width_and_inherited_small_head(model):
    assert int(model.unified_graph_width) == 302
    assert sdpl._n_params(model.head) == 4135


# ---------------------------------------------------------------------------
# B. no pair -> local feedback
# ---------------------------------------------------------------------------


def test_b_local_state_and_dictionary_bit_identical_under_pair_mutation(model, batch):
    model.eval()
    captured = {}

    def _capture_h(_module, _inputs, output):
        captured["h"] = output.detach().clone()

    handle = model.patch_encoder.register_forward_hook(_capture_h)
    try:
        with torch.no_grad():
            prediction_before = model(batch)
        h_before = captured["h"].clone()
        alpha_before, _, _ = model.dictionary.alpha_from_h(h_before)
        coord_before = model.dictionary.coord_proj(alpha_before)
        mutated = batch.clone()
        mutated.pair_relation = torch.randn_like(mutated.pair_relation)
        with torch.no_grad():
            prediction_after = model(mutated)
        h_after = captured["h"].clone()
        alpha_after, _, _ = model.dictionary.alpha_from_h(h_after)
        coord_after = model.dictionary.coord_proj(alpha_after)
    finally:
        handle.remove()
    assert torch.equal(h_before, h_after), "pair state feeds back into h_i"
    assert torch.equal(alpha_before, alpha_after), "dictionary assignment changed"
    assert torch.equal(coord_before, coord_after), "dictionary coordinate changed"
    assert float((prediction_before - prediction_after).abs().max()) > 1e-8, (
        "pair mutation did not change the graph prediction (vacuous test)"
    )


def test_b_contract_helper_passes(model, batch):
    report = sdpl.static_contract_checks(model, batch)
    assert report["passed"], report
    assert report["h_identical_under_relation_mutation"]
    assert report["alpha_identical_under_relation_mutation"]
    assert report["coord_identical_under_relation_mutation"]
    assert report["max_abs_h_diff"] == 0.0
    assert report["prediction_changes_under_relation_mutation"]


# ---------------------------------------------------------------------------
# C. pair/dictionary evaluated exactly once
# ---------------------------------------------------------------------------


def test_c_pair_and_dictionary_called_once(model, batch):
    model.eval()
    counters = {"pair_encoder": 0, "relation_encoder": 0, "dictionary": 0}
    handles = [
        model.pair_encoder.register_forward_hook(
            lambda *_a, _c=counters: _c.__setitem__("pair_encoder", _c["pair_encoder"] + 1)
        ),
        model.relation_encoder.register_forward_hook(
            lambda *_a, _c=counters: _c.__setitem__(
                "relation_encoder", _c["relation_encoder"] + 1
            )
        ),
        model.dictionary.register_forward_hook(
            lambda *_a, _c=counters: _c.__setitem__("dictionary", _c["dictionary"] + 1)
        ),
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert counters["pair_encoder"] == 1, counters
    assert counters["relation_encoder"] == 1, counters
    assert counters["dictionary"] == 1, counters


# ---------------------------------------------------------------------------
# D. invariance
# ---------------------------------------------------------------------------


def test_d_pair_order_invariance(model, batch):
    permutation = torch.randperm(int(batch.pair_index.shape[1]))
    permuted = batch.clone()
    permuted.pair_index = batch.pair_index[:, permutation]
    permuted.pair_relation = batch.pair_relation[permutation]
    permuted.pair_bucket = batch.pair_bucket[permutation]
    model.eval()
    with torch.no_grad():
        base = model(batch)
        other = model(permuted)
    assert float((base - other).abs().max()) < 1e-5


def test_d_node_relabel_invariance(model, single_batch):
    batch = single_batch
    n = int(batch.num_nodes)
    permutation = torch.randperm(n)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(n)
    relabelled = batch.clone()
    relabelled.patch_cont = batch.patch_cont[permutation]
    relabelled.typed_token = batch.typed_token[permutation]
    relabelled.parent_token = batch.parent_token[permutation]
    relabelled.structural_token = batch.structural_token[permutation]
    relabelled.pair_index = inverse[batch.pair_index]
    model.eval()
    with torch.no_grad():
        base = model(batch)
        other = model(relabelled)
    assert float((base - other).abs().max()) < 1e-5


# ---------------------------------------------------------------------------
# E. gradient viability
# ---------------------------------------------------------------------------


def test_e_dictionary_kernel_receives_gradient(model, batch):
    report = sdpl.gradient_viability_checks(model, batch)
    assert report["all_finite"], report
    assert report["required_positive_ok"], report
    assert report["dictionary_atoms_grad_norm"] > 0.0, report
    assert report["wq_grad_norm"] > 0.0, report
    assert report["u_grad_norm"] > 0.0, report
    assert report["wm_grad_norm"] > 0.0, report
    assert report["pair_encoder_grad_norm"] > 0.0, report


# ---------------------------------------------------------------------------
# F. contract details: no residual / gamma, dictionary width, intervention
# ---------------------------------------------------------------------------


def test_f_no_local_residual_or_gamma(model):
    assert not hasattr(model, "patch_encoder_branch")
    # The strict-static backbone is not wrapped in a residual adapter.
    assert isinstance(model.patch_encoder, zpp._MLPBlock)
    gamma_like = [name for name, _ in model.named_parameters() if "gamma" in name]
    assert gamma_like == [], gamma_like


def test_f_pair_input_layout(model):
    assert int(model.pair_encoder.layers[0].in_features) == 112
    assert int(model.dict_gate.in_features) == 64
    assert int(model.dict_gate.out_features) == 16
    assert int(model.dictionary.coord_proj.out_features) == 16
    assert int(model.dictionary.atoms.shape[0]) == 64


def test_f_tau_and_coordinate_conventions(model):
    assert abs(float(model.dictionary.current_tau()) - sdpl.DICT_TAU_INIT) < 1e-6
    atoms = model.dictionary.normalized_atoms()
    norms = atoms.norm(dim=1)
    assert float((norms - 1.0).abs().max()) < 1e-5
    h = torch.randn(11, int(model.patch_hidden))
    alpha, tau, _atoms = model.dictionary.alpha_from_h(h)
    assert alpha.shape == (11, sdpl.DICT_ATOMS)
    assert float((alpha.sum(dim=1) - 1.0).abs().max()) < 1e-5
    assert float(alpha.min()) >= 0.0
    assert 0.05 < float(tau) < 1.0


def test_f_dictionary_intervention_is_non_vacuous(model, batch):
    model.eval()
    with torch.no_grad():
        normal = model(batch)
    model.set_dict_intervention("neutral")
    try:
        with torch.no_grad():
            neutral = model(batch)
    finally:
        model.set_dict_intervention(None)
    assert float((normal - neutral).abs().max()) > 1e-8


# ---------------------------------------------------------------------------
# parameter / initialization control
# ---------------------------------------------------------------------------


def test_parameter_audit_budget():
    audit = sdpl.parameter_audit()
    assert audit["center_update_is_none"] is True
    assert audit["unified_graph_width"] == 302
    assert audit["pair_input_width"] == 112
    assert audit["params"]["sdpl_total"] == 74996
    assert audit["target_budget"]["within_preferred"], audit["params"]
    assert audit["params"]["sdpl_total"] > audit["params"]["s0_total"]


def test_initialization_match_shared_bit_identical():
    report = sdpl.initialization_match()
    assert report["bit_identical_shared"], report
    assert report["max_abs_diff_s0_vs_sdpl"] == 0.0
    assert report["pair_encoder_widened"]
    assert report["n_shared_tensors"] > 20
