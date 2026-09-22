"""Targeted tests for SRDA-v0 (static relational dictionary algebra).

Static/CPU-only, synthetic ``Data``: never loads ZINC data, checkpoints,
official valid, or official test.

They enforce the non-negotiable strict-static contract and the SRDA-specific
claims:

* A. architecture: standalone module, ``center_context=False``,
  ``center_update is None``, forward succeeds while the historical
  ``_pool_pairs_to_centres`` raises;
* B. local independence: ``z`` / ``q`` / ``alpha`` / ``eps`` bit-identical
  under pair-relation mutation while the prediction changes (non-vacuous);
* C. one single pass: dictionary assignment / relation trunk / pair kernel each
  evaluated exactly once;
* D. no raw endpoint bypass: perturbing ``z`` inside ``ker(Wq)`` shifts ``z``
  and the unary readout materially while the pair-kernel input stays at
  floating-point noise, and the pair input is *exactly* the declared 72D block
  algebra reconstructed from ``alpha`` / ``eps`` / ``rel`` only;
* E. no identity channel: typed-token / parent-token mutation has exactly zero
  effect and no identity lookup parameter exists;
* F. invariance: pair order, endpoint swap, within-graph patch relabel, graph
  order;
* G. gradients: local encoder, ``Wq``, ``D``, ``A``, ``B``, relation trunk,
  ``W_eps`` and the pair encoder all receive nonzero finite gradient;
* H. pooling semantics are the inherited mean/std/log-count moments;
* I. the three cheap inference interventions actually change predictions.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16 import (
    zinc_static_relational_dictionary_algebra as srda,
)


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
        # Present but semantically inert: SRDA must never read these.
        typed_token=torch.randint(0, 4096, (n,), generator=generator),
        parent_token=torch.randint(0, 32, (n,), generator=generator),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 0),
        pair_index=pair_index,
        pair_relation=torch.randn(
            len(pairs), int(zpp.RELATION_WIDTH), generator=generator
        ),
        pair_bucket=torch.randint(
            0, zpp.DISTANCE_BUCKETS, (len(pairs),), generator=generator
        ),
        global_context=torch.randn(1, int(zpp.GLOBAL_WIDTH), generator=generator),
        topology_features=torch.randn(1, ztopo_width(), generator=generator),
        y=torch.randn(1, generator=generator),
        num_nodes=n,
    )


def _batch(graphs=None):
    graphs = graphs or [_graph(n, seed=n) for n in (7, 6, 8)]
    return next(iter(zpp._make_loader(graphs, len(graphs), False, 0)))


def _model() -> srda.SRDAModel:
    model = srda.build_srda(0)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# A. architecture
# ---------------------------------------------------------------------------


def test_architecture_contract_and_runtime_widths():
    model = srda.build_srda(0)
    assert model.center_context is False
    assert model.center_update is None
    # Standalone: the inherited message-passing machinery does not exist here.
    assert not isinstance(model, zpp.PatchPathModel)
    assert not hasattr(model, "_pool_pairs_to_centres")

    audit = srda.runtime_width_audit(model, _batch())
    assert audit["widths_consistent"]
    assert audit["local_input_width"] == srda.LOCAL_INPUT_WIDTH == 146
    assert audit["local_state_width"] == srda.LOCAL_DIM == 64
    assert audit["dictionary_assignment_width"] == srda.DICT_ATOMS == 64
    assert audit["prototype_factor_width"] == srda.TENSOR_RANK == 32
    assert audit["residual_width"] == srda.RESIDUAL_DIM == 8
    assert audit["pair_input_width"] == srda.PAIR_INPUT_WIDTH == 72
    assert audit["relation_input_width"] == zpp.RELATION_WIDTH == 23
    assert audit["unary_width"] == 129
    assert audit["pair_pool_width"] == 325
    assert audit["global_width"] == 32
    assert audit["topology_width"] == 8
    assert audit["graph_width"] == 494


def test_forward_without_pair_to_center():
    model = _model()
    batch = _batch()
    original = zpp.PatchPathModel._pool_pairs_to_centres
    zpp.PatchPathModel._pool_pairs_to_centres = srda._contract_violation
    try:
        with torch.no_grad():
            prediction = model(batch)
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original
    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()


def test_static_contract_checks_pass():
    model = srda.build_srda(0)
    results = srda.static_contract_checks(model, _batch())
    assert results["passed"]
    assert results["pair_encoder_calls_per_forward"] == 1
    assert results["relation_trunk_calls_per_forward"] == 1
    assert results["dictionary_calls_per_forward"] == 1
    assert results["forward_without_pair_to_center"]


# ---------------------------------------------------------------------------
# B/C. local independence + single pass
# ---------------------------------------------------------------------------


def test_local_states_independent_of_pair_relation():
    model = _model()
    batch = _batch()
    prediction, trace = srda._trace_forward(model, batch)

    mutated = batch.clone()
    generator = torch.Generator().manual_seed(1234)
    mutated.pair_relation = torch.randn(
        mutated.pair_relation.shape, generator=generator
    ).to(mutated.pair_relation.device)
    with torch.no_grad():
        mutated_prediction = model(mutated)
    _, mutated_trace = srda._trace_forward(model, mutated)

    for key in ("z", "q_raw", "alpha", "eps"):
        assert torch.equal(trace[key], mutated_trace[key]), key
    # Non-vacuous: the relation really is consumed.
    assert float((prediction - mutated_prediction).abs().max()) > 0.0
    # And the pair path itself does see the mutation.
    assert not torch.equal(trace["pair_input"], mutated_trace["pair_input"])


def test_pair_evaluated_once_and_counted():
    model = _model()
    batch = _batch()
    _, trace = srda._trace_forward(model, batch)
    counters = trace["counters"]
    assert counters == {"pair_encoder": 1, "relation_trunk": 1, "dictionary": 1}


# ---------------------------------------------------------------------------
# D. no raw endpoint bypass
# ---------------------------------------------------------------------------


def test_no_raw_pair_bypass_kernel_of_wq():
    model = _model()
    batch = _batch()
    _, trace = srda._trace_forward(model, batch)

    weight = model.dictionary.query.weight.detach()
    basis, _ = torch.linalg.qr(weight.t())
    generator = torch.Generator().manual_seed(4242)
    vector = torch.randn(int(weight.shape[1]), generator=generator, dtype=weight.dtype)
    delta = vector - basis @ (basis.t() @ vector)
    delta = delta / delta.norm().clamp_min(1.0e-12)
    assert float((weight @ delta).abs().max()) < 1.0e-4

    _, perturbed = srda._trace_forward_perturbed(model, batch, delta)
    z_relative = float(
        (trace["z"] - perturbed["z"]).norm() / trace["z"].norm().clamp_min(1.0e-12)
    )
    pair_relative = float(
        (trace["pair_input"] - perturbed["pair_input"]).abs().max()
        / trace["pair_input"].abs().max().clamp_min(1.0e-12)
    )
    assert z_relative > 0.05
    assert pair_relative < 1.0e-4


def test_pair_input_is_exactly_the_declared_block_algebra():
    model = _model()
    batch = _batch()
    _, trace = srda._trace_forward(model, batch)

    source = batch.pair_index[0].long()
    target = batch.pair_index[1].long()
    with torch.no_grad():
        rel_hidden = model.relation_trunk(batch.pair_relation)
        rel_gate = 1.0 + torch.tanh(
            model.relation_gate(rel_hidden) + model.bucket_embedding(batch.pair_bucket)
        )
        rel_feat = model.relation_feature(rel_hidden)
        proto_pair = 0.5 * (
            trace["a"][source] * trace["b"][target]
            + trace["a"][target] * trace["b"][source]
        )
        p_dict = proto_pair * rel_gate
        e = model.residual_proj(trace["eps"])
        left, right = e[source], e[target]
        p_eps = torch.cat(
            [left + right, torch.abs(left - right), left * right], dim=1
        )
        reconstructed = torch.cat([p_dict, p_eps, rel_feat], dim=1)

    assert torch.allclose(trace["pair_input"], reconstructed, atol=1.0e-6)
    assert trace["pair_input"].shape[1] == 32 + 24 + 16

    # Non-vacuous negative control: ``p_dict`` is a genuine function of the
    # *assignment*, not of raw endpoint state and not a constant.  Re-deriving
    # it from an unrelated patch's assignment moves it materially.
    shuffled_source = (source + 1) % trace["alpha"].shape[0]
    shuffled = 0.5 * (
        trace["a"][shuffled_source] * trace["b"][target]
        + trace["a"][target] * trace["b"][shuffled_source]
    ).detach() * rel_gate.detach()
    assert float((p_dict - shuffled).abs().max()) > 1.0e-3


# ---------------------------------------------------------------------------
# E. no identity channel
# ---------------------------------------------------------------------------


def test_no_identity_channel_params():
    model = srda.build_srda(0)
    audit = srda.identity_channel_audit(model)
    assert audit["forbidden_key_hits"] == []
    assert audit["only_bucket_embedding"]
    assert audit["identity_channel_params"] == 0
    assert audit["uses_typed_token"] is False
    assert audit["uses_parent_token"] is False
    for key in model.state_dict():
        assert "typed" not in key
        assert "parent" not in key


def test_identity_token_mutation_has_zero_effect():
    model = _model()
    batch = _batch()
    with torch.no_grad():
        baseline = model(batch)
        mutated = batch.clone()
        mutated.typed_token = torch.randint(
            0, 4096, mutated.typed_token.shape, generator=torch.Generator().manual_seed(3)
        )
        mutated.parent_token = torch.randint(
            0, 32, mutated.parent_token.shape, generator=torch.Generator().manual_seed(4)
        )
        changed = model(mutated)
    assert torch.equal(baseline, changed)
    assert float((baseline - changed).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# F. invariance
# ---------------------------------------------------------------------------


def test_pair_order_and_endpoint_swap_invariance():
    model = _model()
    batch = _batch()
    with torch.no_grad():
        baseline = model(batch)

        permutation = torch.randperm(
            int(batch.pair_index.shape[1]), generator=torch.Generator().manual_seed(7)
        )
        permuted = batch.clone()
        permuted.pair_index = batch.pair_index[:, permutation]
        permuted.pair_relation = batch.pair_relation[permutation]
        permuted.pair_bucket = batch.pair_bucket[permutation]
        assert torch.allclose(baseline, model(permuted), atol=1.0e-5)

        swapped = batch.clone()
        swapped.pair_index = batch.pair_index.flip(0)
        assert torch.allclose(baseline, model(swapped), atol=1.0e-5)


def test_within_graph_patch_relabel_invariance():
    model = _model()
    batch = _batch()
    generator = torch.Generator().manual_seed(13)
    order = torch.arange(int(batch.patch_cont.shape[0]))
    for graph_id in range(int(batch.global_context.shape[0])):
        members = (batch.batch == graph_id).nonzero(as_tuple=True)[0]
        shuffled = members[torch.randperm(int(members.numel()), generator=generator)]
        order[members] = shuffled
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.shape[0])
    relabeled = batch.clone()
    relabeled.patch_cont = batch.patch_cont[order]
    relabeled.patch_context = batch.patch_context[order]
    relabeled.typed_token = batch.typed_token[order]
    relabeled.parent_token = batch.parent_token[order]
    relabeled.pair_index = inverse[batch.pair_index]
    with torch.no_grad():
        assert torch.allclose(model(batch), model(relabeled), atol=1.0e-5)


def test_graph_order_invariance():
    graphs = [_graph(n, seed=n) for n in (7, 6, 8)]
    model = _model()
    with torch.no_grad():
        forward = model(next(iter(zpp._make_loader(graphs, 3, False, 0))))
        backward = model(
            next(iter(zpp._make_loader(list(reversed(graphs)), 3, False, 0)))
        )
    assert torch.allclose(forward, backward.flip(0), atol=1.0e-5)


# ---------------------------------------------------------------------------
# G. gradients
# ---------------------------------------------------------------------------


def test_gradient_viability():
    model = srda.build_srda(0)
    payload = srda.gradient_viability_checks(model, _batch())
    assert payload["branch_receives_gradient"]
    assert payload["all_finite"]
    for key in (
        "local_encoder_grad_norm",
        "wq_grad_norm",
        "dict_atoms_grad_norm",
        "factor_a_grad_norm",
        "factor_b_grad_norm",
        "residual_proj_grad_norm",
        "relation_trunk_grad_norm",
        "pair_encoder_grad_norm",
    ):
        assert float(payload[key]) > 0.0, key


# ---------------------------------------------------------------------------
# H. inherited pooling semantics
# ---------------------------------------------------------------------------


def test_pooling_matches_inherited_moment_semantics():
    reference = sdp.build_s0(0)
    reference.eval()
    batch = _batch()
    n_graphs = int(batch.global_context.shape[0])

    generator = torch.Generator().manual_seed(21)
    nodes = torch.randn(int(batch.patch_cont.shape[0]), 5, generator=generator)
    pairs = torch.randn(int(batch.pair_index.shape[1]), 3, generator=generator)
    pair_batch = batch.batch[batch.pair_index[0].long()]

    expected_nodes = reference._pool_values(nodes, batch.batch, n_graphs, "mean_std")
    expected_pairs = []
    previous = reference.pair_readout
    reference.pair_readout = "mean_std"
    try:
        expected_pairs.append(
            reference._pool_pairs(pairs, pair_batch, batch.pair_bucket, n_graphs)
        )
    finally:
        reference.pair_readout = previous

    assert torch.allclose(srda.node_moments(nodes, batch.batch, n_graphs), expected_nodes)
    assert torch.allclose(
        srda.bucket_moments(pairs, pair_batch, batch.pair_bucket, n_graphs),
        expected_pairs[0],
    )


# ---------------------------------------------------------------------------
# I. interventions
# ---------------------------------------------------------------------------


def test_inference_interventions_change_predictions():
    model = _model()
    batch = _batch()
    with torch.no_grad():
        baseline = model(batch)
        for mode in ("neutral_dict", "zero_eps", "mean_alpha"):
            model.set_pair_intervention(mode)
            try:
                shifted = model(batch)
            finally:
                model.set_pair_intervention(None)
            assert float((baseline - shifted).abs().max()) > 0.0, mode
        # Default path is restored bit-exactly.
        assert torch.equal(baseline, model(batch))


def test_pair_block_scale_structure():
    model = _model()
    payload = srda.pair_block_scale(model, _batch())
    for key in ("mean_norm_p_dict", "mean_norm_p_eps", "mean_norm_rel_feat"):
        assert float(payload[key]) > 0.0
    weights = payload["pair_encoder_first_layer_block_weight_norms"]
    assert set(weights) == {"p_dict", "p_eps", "rel_feat"}
