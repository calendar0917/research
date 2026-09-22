"""Targeted tests for the strict-static dictionary-pair round.

These tests are static/CPU-only, use synthetic ``Data`` and never load ZINC
data, checkpoints, official valid, or official test.  They verify the
non-negotiable parts of the architecture contract:

* A. the strict-static architecture (``center_context=False``,
  ``center_update is None``, no ``_pool_pairs_to_centres`` call in forward);
* B. no pair -> local feedback (``h_i`` bit-identical under pair mutation);
* C. every pair is evaluated exactly once, with no relation refresh;
* D. permutation invariance (pair order, graph node relabel);
* E. gradient viability of the residual branches on a real mini-batch.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------


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
        typed_token=torch.randint(0, sdp.TYPED_VOCAB_SIZE, (n,), generator=generator),
        parent_token=torch.randint(0, sdp.PARENT_VOCAB_SIZE, (n,), generator=generator),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 0),
        pair_index=pair_index,
        pair_relation=torch.randn(len(pairs), int(zpp.RELATION_WIDTH), generator=generator),
        pair_bucket=torch.randint(0, zpp.DISTANCE_BUCKETS, (len(pairs),), generator=generator),
        global_context=torch.randn(1, int(zpp.GLOBAL_WIDTH), generator=generator),
        topology_features=torch.randn(
            1, int(ztopo_width()), generator=generator
        ),
        y=torch.randn(1, generator=generator),
        num_nodes=n,
    )


def ztopo_width() -> int:
    from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

    return int(ztopo.raw_width("hinge"))


@pytest.fixture(scope="module")
def batch():
    graphs = [_graph(n, seed=n) for n in (7, 6, 8)]
    return next(iter(zpp._make_loader(graphs, 8, False, 0)))


@pytest.fixture(scope="module")
def single_batch():
    return next(iter(zpp._make_loader([_graph(7, seed=7)], 8, False, 0)))


@pytest.fixture(scope="module")
def models():
    return {arm: sdp.ARMS[arm](seed=0) for arm in ("s0", "dense", "dict")}


# ---------------------------------------------------------------------------
# A. architecture
# ---------------------------------------------------------------------------


def test_a_center_context_disabled(models):
    for arm, model in models.items():
        assert model.center_context is False, arm
        assert model.center_update is None, arm
        assert model.center_context_width == 0, arm


def test_a_forward_never_calls_pool_pairs_to_centres(models, batch, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("strict-static violation: _pool_pairs_to_centres called")

    monkeypatch.setattr(zpp.PatchPathModel, "_pool_pairs_to_centres", _boom)
    for arm, model in models.items():
        model.eval()
        with torch.no_grad():
            prediction = model(batch)
        assert prediction.shape == (3,), arm
        assert torch.isfinite(prediction).all(), arm


def test_a_unified_width_and_small_head(models):
    for arm, model in models.items():
        assert int(model.unified_graph_width) == 302, arm
        assert isinstance(model.head, sdp.GenericReader), arm
        assert sdp._n_params(model.head) == 4135, arm


# ---------------------------------------------------------------------------
# B. no pair -> local feedback
# ---------------------------------------------------------------------------


def test_b_local_state_bit_identical_under_pair_mutation(models, batch):
    for arm, model in models.items():
        model.eval()
        captured = {}

        def _capture_h(_module, _inputs, output):
            captured["h"] = output.detach().clone()

        handle = model.patch_encoder.register_forward_hook(_capture_h)
        try:
            with torch.no_grad():
                prediction_before = model(batch)
            h_before = captured["h"].clone()
            mutated = batch.clone()
            mutated.pair_relation = torch.randn_like(mutated.pair_relation)
            with torch.no_grad():
                prediction_after = model(mutated)
            h_after = captured["h"].clone()
        finally:
            handle.remove()
        assert torch.equal(h_before, h_after), f"{arm}: pair state feeds back into h_i"
        assert float((prediction_before - prediction_after).abs().max()) > 1e-8, (
            f"{arm}: pair mutation did not change the graph prediction "
            "(test would be vacuous)"
        )


def test_b_contract_helper_passes(models, batch):
    for arm, model in models.items():
        report = sdp.static_contract_checks(model, batch)
        assert report["passed"], (arm, report)
        assert report["h_identical_under_relation_mutation"], arm
        assert report["max_abs_h_diff"] == 0.0, arm


# ---------------------------------------------------------------------------
# C. pair evaluated exactly once
# ---------------------------------------------------------------------------


def test_c_pair_encoder_called_once(models, batch):
    for arm, model in models.items():
        model.eval()
        counters = {"pair_encoder": 0, "relation_encoder": 0}
        handles = [
            model.pair_encoder.register_forward_hook(
                lambda *_a, _c=counters: _c.__setitem__(
                    "pair_encoder", _c["pair_encoder"] + 1
                )
            ),
            model.relation_encoder.register_forward_hook(
                lambda *_a, _c=counters: _c.__setitem__(
                    "relation_encoder", _c["relation_encoder"] + 1
                )
            ),
        ]
        try:
            with torch.no_grad():
                model(batch)
        finally:
            for handle in handles:
                handle.remove()
        assert counters["pair_encoder"] == 1, (arm, counters)
        assert counters["relation_encoder"] == 1, (arm, counters)


# ---------------------------------------------------------------------------
# D. invariance
# ---------------------------------------------------------------------------


def test_d_pair_order_invariance(models, batch):
    permutation = torch.randperm(int(batch.pair_index.shape[1]))
    permuted = batch.clone()
    permuted.pair_index = batch.pair_index[:, permutation]
    permuted.pair_relation = batch.pair_relation[permutation]
    permuted.pair_bucket = batch.pair_bucket[permutation]
    for arm, model in models.items():
        model.eval()
        with torch.no_grad():
            base = model(batch)
            other = model(permuted)
        assert float((base - other).abs().max()) < 1e-5, arm


def test_d_node_relabel_invariance(models, single_batch):
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
    for arm, model in models.items():
        model.eval()
        with torch.no_grad():
            base = model(batch)
            other = model(relabelled)
        assert float((base - other).abs().max()) < 1e-5, arm


# ---------------------------------------------------------------------------
# E. gradient viability
# ---------------------------------------------------------------------------


def test_e_dictionary_branch_receives_gradient(models, batch):
    model = models["dict"]
    report = sdp.gradient_viability_checks(model, batch)
    assert report["all_finite"], report
    assert report["dictionary_grad_norm"] > 0.0, report
    assert report["wq_grad_norm"] > 0.0, report
    assert report["wv_grad_norm"] > 0.0, report
    assert report["gamma_grad"] is not None and math.isfinite(report["gamma_grad"]), report


def test_e_dense_branch_receives_gradient(models, batch):
    model = models["dense"]
    report = sdp.gradient_viability_checks(model, batch)
    assert report["all_finite"], report
    assert report["dense_layer0_grad_norm"] > 0.0, report
    assert report["dense_layer1_grad_norm"] > 0.0, report
    assert report["gamma_grad"] is not None and math.isfinite(report["gamma_grad"]), report


# ---------------------------------------------------------------------------
# parameter / initialisation control
# ---------------------------------------------------------------------------


def test_parameter_match_within_one_percent():
    audit = sdp.parameter_audit()
    assert audit["center_update_is_none"] is True
    assert audit["unified_graph_width"] == 302
    match = audit["parameter_match"]
    assert match["within_one_percent"], match
    assert match["dict_branch_params"] == 5201
    assert match["dense_branch_params"] == 5189


def test_initialization_match_bit_identical():
    report = sdp.initialization_match()
    assert report["bit_identical"], report
    assert report["max_abs_diff_dense_vs_dict"] == 0.0
    assert report["max_abs_diff_s0_vs_dict"] == 0.0
    assert report["n_shared_tensors"] > 20


def test_residual_init_conventions(models):
    model = models["dict"]
    branch = model.residual_branch
    assert isinstance(branch, sdp.ResidualDictionaryBranch)
    assert abs(float(branch.current_tau()) - sdp.DICT_TAU_INIT) < 1e-6
    assert abs(float(model.patch_encoder.gamma) - sdp.RESIDUAL_GAMMA_INIT) < 1e-6
    atoms = branch.normalized_atoms()
    norms = atoms.norm(dim=1)
    assert float((norms - 1.0).abs().max()) < 1e-5
    dense = models["dense"]
    assert abs(float(dense.patch_encoder.gamma) - sdp.RESIDUAL_GAMMA_INIT) < 1e-6


def test_dictionary_assignment_is_a_distribution(batch):
    model = sdp.build_dict(0)
    model.eval()
    h0 = torch.randn(11, int(model.patch_hidden))
    alpha, tau, atoms = model.residual_branch.alpha_from_h0(h0)
    assert alpha.shape == (11, sdp.DICT_ATOMS)
    assert float((alpha.sum(dim=1) - 1.0).abs().max()) < 1e-5
    assert float(alpha.min()) >= 0.0
    assert 0.05 < float(tau) < 1.0
