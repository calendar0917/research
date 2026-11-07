"""CPU correctness tests for the Conditional S2 and near-matched FFN arms.

The load-bearing claims are: exact parameter counts, a bitwise-identical
original W2 parameter subset and initial output, a genuinely path-conditioned
vector gate (not a re-parameterised scalar), exactly one dropout call, no
iterative propagation, and frozen path vocabularies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import cs2_common as C
import cs2_model as M
import bs_model as BSM

pytestmark = pytest.mark.slow


def test_parameter_counts_are_exact(models):
    expected = {"C2": (10_752, 1_005_076), "F": (10_664, 1_004_988)}
    for key, (new, total) in expected.items():
        model, audit = models[key]
        assert int(sum(p.numel() for p in model.parameters())) == total
        assert audit["new_parameters"] == new
        assert audit["base_parameters"] == C.BASE_PARAMETERS
        assert audit["total_parameters"] == total
        if key == "C2":
            assert model.state_dict()["local.path_cond_v"].shape == (16, 192)
    assert C.C2_ADDED == 10_752
    assert C.FFN_ADDED == 10_664
    assert C.PARAMETER_GAP == 88
    assert abs(C.PARAMETER_GAP / C.NEW_PARAMETERS["C2"]) < 0.01


def test_original_w2_subset_bitwise_equal(models):
    base_state = models["W2"][0].state_dict()
    for key in C.MODEL_ORDER:
        state = models[key][0].state_dict()
        for name, value in base_state.items():
            assert name in state, name
            assert torch.equal(value.cpu(), state[name].cpu()), (key, name)
        assert models[key][1]["original_w2_subset"]["all_bitwise_equal"]


def test_initial_logits_bitwise_equal_to_w2(models, batch):
    base = models["W2"][0]
    with torch.no_grad():
        base.eval()
        ref = base(batch)
        for key in C.MODEL_ORDER:
            model = models[key][0]
            model.eval()
            got = model(batch)
            assert torch.equal(ref, got), key
            assert float((ref - got).abs().max()) == 0.0


def test_c2_gate_is_zero_at_init_and_has_path_width(models, batch):
    _model, audit = models["C2"]
    model = _model
    local = model.local
    assert torch.count_nonzero(local.path_cond_v).item() == 0
    assert tuple(local.path_cond_v.shape) == (16, 192)
    with torch.no_grad():
        gate, delta = local.vector_gate(batch["path_cat"], batch["path_end63"])
    assert tuple(gate.shape) == (int(batch["path_cat"].numel()), 192)
    assert tuple(delta.shape) == tuple(gate.shape)
    assert torch.count_nonzero(delta).item() == 0
    scalar = local.path_gate(batch["path_cat"]).expand(-1, gate.shape[1])
    assert torch.equal(gate, scalar)  # scalar gate preserved at init
    assert audit["new_module_init"]["v_is_zero"]


def test_f_ffn_is_zero_at_init_and_no_extra_width(models, batch):
    _model, audit = models["F"]
    model = _model
    assert tuple(model.ffn.fc1.weight.shape) == (40, 128)
    assert tuple(model.ffn.fc2.weight.shape) == (128, 40)
    assert torch.count_nonzero(model.ffn.fc2.weight).item() == 0
    assert torch.count_nonzero(model.ffn.fc2.bias).item() == 0
    diag = M.module_diagnostics(model, batch)
    assert diag["ffn"]["branch"]["rms"] == 0.0
    assert audit["new_module_init"]["fc2_zero"]


def test_gate_actually_conditions_on_path_type_and_endpoint(models, batch):
    """Changing V must make the *same* path set produce different vectors, and
    two paths with the same path type but different endpoint codes must differ."""
    model = models["C2"][0]
    local = model.local
    saved = local.path_cond_v.detach().clone()
    try:
        with torch.no_grad():
            gen = torch.Generator().manual_seed(7)
            local.path_cond_v.copy_(torch.randn(16, 192, generator=gen) * 0.05)
            gate, delta = local.vector_gate(batch["path_cat"], batch["path_end63"])
        assert torch.count_nonzero(delta).item() > 0
        assert float(delta.std().item()) > 0.0
        assert tuple(gate.shape) == (int(batch["path_cat"].numel()), 192)
        cat = batch["path_cat"].numpy()
        end = batch["path_end63"].numpy()
        pair = None
        for i in range(len(cat)):
            for j in range(i + 1, len(cat)):
                if cat[i] == cat[j] and end[i] != end[j]:
                    pair = (i, j)
                    break
            if pair:
                break
        assert pair is not None
        i, j = pair
        assert not torch.equal(delta[i], delta[j])
        # path type changes also change the gate, for the same endpoint code
        pair2 = None
        for a in range(len(cat)):
            for b in range(a + 1, len(cat)):
                if end[a] == end[b] and cat[a] != cat[b]:
                    pair2 = (a, b)
                    break
            if pair2:
                break
        assert pair2 is not None
        a, b = pair2
        assert not torch.equal(delta[a], delta[b])
    finally:
        with torch.no_grad():
            local.path_cond_v.copy_(saved)


def test_new_module_initialisation_is_deterministic_and_private(spec, norm):
    # The historical W2 builder itself advances the global RNG (it constructs
    # modules before fork_rng); the formal trainer reseeds afterwards.  What
    # must NOT advance the RNG is the new-module construction on top of W2.
    base, _ = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=C.SEED, device="cpu")
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    state_before = torch.get_rng_state().clone()
    first, a1 = M.build_model("C2", spec, norm, C.N_TASKS, seed=C.SEED, device="cpu", base_model=base)
    after_one = torch.get_rng_state().clone()
    second, a2 = M.build_model("F", spec, norm, C.N_TASKS, seed=C.SEED, device="cpu", base_model=base)
    after_two = torch.get_rng_state().clone()
    assert torch.equal(state_before, after_one)
    assert torch.equal(state_before, after_two)
    for name, value in base_state.items():
        assert torch.equal(base.state_dict()[name], value), name
    third, a3 = M.build_model("C2", spec, norm, C.N_TASKS, seed=C.SEED, device="cpu", base_model=base)
    assert torch.equal(state_before, torch.get_rng_state())
    assert a1["new_state_keys"] == a3["new_state_keys"]
    for k in a1["new_state_keys"]:
        assert torch.equal(first.state_dict()[k], third.state_dict()[k]), k
    assert a2["new_module_init"]["private_generator_seed"] == 31
    assert a1["new_module_init"]["private_generator_seed"] == 29


def test_call_counts_and_dropout_once(models, batch):
    for key in C.MODEL_ORDER:
        calls = M.count_module_calls(models[key][0], batch)
        assert calls["atom_encoder"] == 1
        assert calls["bond_encoder"] == 1
        assert calls["local"] == 1
        assert calls["dropout"] == 1
        assert calls["f"] == 1
        assert calls["rho"] == 1
        assert calls["readout"] == 1
        if key == "C2":
            assert calls["path_cond_ec"] == 1
            assert calls["path_cond_ee"] == 1
        else:
            assert calls["new_module"] == 1


def test_post_local_never_needs_bonds_or_paths(models, batch):
    for key in C.MODEL_ORDER:
        model = models[key][0]
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            _a, _b, _scal, h = model.encode_h(batch)
            slim = {
                "n_graphs": batch["n_graphs"],
                "atom_batch": batch["atom_batch"],
                "global": batch["global"],
            }
            only_h = model.post_local(h, slim)
        assert torch.equal(logits, only_h), key


def test_no_iterative_message_passing_and_batch_immutable(models, batch):
    snapshot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for key in C.MODEL_ORDER:
        model = models[key][0]
        model.eval()
        with torch.no_grad():
            model(batch)
        for name, value in snapshot.items():
            if torch.is_tensor(value):
                assert torch.equal(value, batch[name]), (key, name)
        # A single local call is the only structural access; ``post_local``
        # runs from h alone and is checked above.  There is no second encoder.
        calls = M.count_module_calls(model, batch)
        assert calls["atom_encoder"] == 1 and calls["bond_encoder"] == 1 and calls["local"] == 1


def test_atom_renumbering_invariance(models, batch):
    permuted, _p = BSM.permute_atoms(batch, seed=20261108)
    for key in C.MODEL_ORDER:
        model = models[key][0]
        model.eval()
        with torch.no_grad():
            a = model(batch)
            b = model(permuted)
        assert float((a - b).abs().max().item()) <= 2e-5, key


def test_frozen_path_vocabulary_and_no_test_labels(task):
    cache = task.cache
    cat = np.asarray(cache["path_cat"])
    end63 = np.asarray(cache["path_end63"])
    assert int(cat.min()) >= 0 and int(cat.max()) < 400
    assert int(end63.min()) >= 0 and int(end63.max()) < 80
    assert task.spec.n_path_codes == 400
    assert task.spec.n_nbr_codes == 80
    assert not np.isfinite(np.asarray(task.labels[task.test_rows])).any()


def test_new_module_types_are_only_the_declared_ones(models):
    c2 = {type(m).__name__ for n, m in models["C2"][0].named_modules() if n.startswith("local.path_cond")}
    f = {type(m).__name__ for n, m in models["F"][0].named_modules() if n.startswith("ffn")}
    assert c2 == {"Embedding"}
    assert f == {"LocalResidualFFN", "LayerNorm", "Linear"}
    # the historical scalar gate is still there, untouched
    assert models["C2"][0].local.path_gate.weight.shape == (400, 1)
    assert models["F"][0].local.path_gate.weight.shape == (400, 1)
    assert torch.equal(
        models["C2"][0].local.path_gate.weight, models["W2"][0].local.path_gate.weight
    )
    # no sigmoid / attention / extra normalisation in the C2 branch
    names = [n for n, _ in models["C2"][0].local.named_modules()]
    assert not any("sigmoid" in n or "attn" in n or "attention" in n for n in names)
