"""Model: definitions, initialisation, zero residual, symmetry, gradients."""

from __future__ import annotations

import math

import pytest
import torch

import pw_common as C
import pw_model as M



def test_parameter_arithmetic_matches_the_definition():
    expected = (
        3 * C.HIDDEN_WIDTH * C.PAIR_DIM  # w_sum, w_diff, w_prod (bias-free)
        + C.N_BUCKETS * C.PAIR_DIM  # distance embedding [6, 48]
        + 2 * C.PAIR_DIM  # LayerNorm(48)
        + C.N_QUERIES * C.PAIR_DIM  # 16 shared queries
        + 2 * (C.PAIR_DIM * C.PAIR_DIM + C.PAIR_DIM)  # key, value
        + 2 * C.Z_DIM  # LayerNorm(768)
        + (C.Z_DIM * C.COMBINER_HIDDEN + C.COMBINER_HIDDEN)  # Linear(768,128)
        + (C.COMBINER_HIDDEN * C.N_TASKS + C.N_TASKS)  # Linear(128,128)
    )
    assert expected == C.BRANCH_ADDED == 140_768
    assert C.EXPECTED_PARAMETERS == C.BASE_PARAMETERS + expected


def test_model_matches_the_definition(spec, norm, tiny_batch):
    model, audit = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    branch = model.branch
    assert tuple(branch.w_sum.weight.shape) == (C.PAIR_DIM, C.HIDDEN_WIDTH) and branch.w_sum.bias is None
    assert tuple(branch.w_diff.weight.shape) == (C.PAIR_DIM, C.HIDDEN_WIDTH) and branch.w_diff.bias is None
    assert tuple(branch.w_prod.weight.shape) == (C.PAIR_DIM, C.HIDDEN_WIDTH) and branch.w_prod.bias is None
    assert tuple(branch.dist_embedding.shape) == (C.N_BUCKETS, C.PAIR_DIM)
    assert tuple(branch.query.shape) == (C.N_QUERIES, C.PAIR_DIM)
    assert tuple(branch.key.weight.shape) == (C.PAIR_DIM, C.PAIR_DIM)
    assert tuple(branch.value.weight.shape) == (C.PAIR_DIM, C.PAIR_DIM)
    assert tuple(branch.combiner.norm.normalized_shape) == (C.Z_DIM,)
    assert tuple(branch.combiner.fc1.weight.shape) == (C.COMBINER_HIDDEN, C.Z_DIM)
    assert tuple(branch.combiner.fc2.weight.shape) == (C.N_TASKS, C.COMBINER_HIDDEN)
    assert float(branch.scale) == pytest.approx(1.0 / math.sqrt(C.PAIR_DIM))
    assert audit["total_parameters"] == C.EXPECTED_PARAMETERS


def test_init_is_bitwise_w2_and_tail_is_zero(spec, norm):
    model, audit = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    base, _ = C.BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=0, device="cpu")
    state = model.state_dict()
    base_state = base.state_dict()
    assert set(base_state) <= set(state)
    for key in base_state:
        assert torch.equal(state[key], base_state[key]), key
    assert torch.count_nonzero(model.branch.combiner.fc2.weight) == 0
    assert torch.count_nonzero(model.branch.combiner.fc2.bias) == 0
    # the pair trunk itself is *not* zero (a private random init)
    assert torch.count_nonzero(model.branch.w_sum.weight) > 0
    assert torch.count_nonzero(model.branch.query) > 0


def test_init_does_not_move_the_global_rng_beyond_the_w2_build(spec, norm):
    """The inherited W2 builder moves the RNG (a throwaway module is built), so
    the claim is that PW adds *no* further movement: the seed-0 W2 state is
    copied, and every new tensor comes from a private generator.
    """
    torch.manual_seed(1234)
    C.BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=0, device="cpu")
    after_w2 = torch.get_rng_state().clone()
    torch.manual_seed(1234)
    M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    assert torch.equal(after_w2, torch.get_rng_state())


def test_zero_residual_equals_w2(spec, norm, tiny_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    base, _ = C.BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=0, device="cpu")
    model.eval()
    base.eval()
    with torch.no_grad():
        ref = base(tiny_batch)
        logits, aux = model(tiny_batch, return_aux=True)
        zeroed = model(tiny_batch, residual_zero=True)
    assert float(aux["residual_logits"].abs().max()) == 0.0
    assert torch.equal(logits, ref)
    assert torch.equal(zeroed, ref)


def test_pair_token_is_bitwise_symmetric(spec, norm, real_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    model.eval()
    with torch.no_grad():
        h = model.encode_h(real_batch)[3]
        i = real_batch["pair_i"]
        j = real_batch["pair_j"]
        b = real_batch["pair_bucket"]
        token = model.branch.pair_tokens(h, i, j, b)
        swapped = model.branch.pair_tokens(h, j, i, b)
    assert torch.equal(token, swapped)


def test_exchange_invariance_with_nonzero_tail(spec, norm, tiny_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        model.branch.combiner.fc2.weight.normal_(0.0, 0.05, generator=gen)
        model.branch.combiner.fc2.bias.normal_(0.0, 0.05, generator=gen)
    model.eval()
    swapped = dict(tiny_batch)
    swapped["pair_i"] = tiny_batch["pair_j"]
    swapped["pair_j"] = tiny_batch["pair_i"]
    with torch.no_grad():
        a = model(tiny_batch)
        b = model(swapped)
        base = model(tiny_batch, residual_zero=True)
    assert torch.equal(a, b)
    assert float((a - base).abs().max()) > 0.0


def test_molecule_without_pairs_has_zero_residual(spec, norm, tiny_no_pair_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    gen = torch.Generator().manual_seed(4)
    with torch.no_grad():
        model.branch.combiner.fc2.weight.normal_(0.0, 0.05, generator=gen)
        model.branch.combiner.fc2.bias.normal_(0.0, 0.05, generator=gen)
    model.eval()
    base, _ = C.BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=0, device="cpu")
    base.eval()
    with torch.no_grad():
        logits, aux = model(tiny_no_pair_batch, return_aux=True)
        ref = base(tiny_no_pair_batch)
    assert torch.isfinite(logits).all()
    assert float(aux["residual_logits"].abs().max()) == 0.0
    assert torch.equal(logits, ref)


def test_segment_softmax_normalised(spec, norm, real_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    model.eval()
    with torch.no_grad():
        h = model.encode_h(real_batch)[3]
        token = model.branch.pair_tokens(
            h, real_batch["pair_i"], real_batch["pair_j"], real_batch["pair_bucket"]
        )
        z, attn, score = model.branch.segment_attention(
            token, real_batch["pair_graph"], int(real_batch["n_graphs"])
        )
    sums = torch.zeros_like(z[:, :, 0]).index_add(0, real_batch["pair_graph"], attn)
    counts = real_batch["pair_counts"]
    with_pairs = counts > 0
    assert float((sums[with_pairs] - 1.0).abs().max()) <= 1e-5
    assert bool((sums[~with_pairs] == 0).all())
    assert torch.isfinite(attn).all() and torch.isfinite(score).all()
    assert bool((attn >= 0).all())


def test_single_encode_and_no_atom_write(spec, norm, tiny_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    counters = M.count_module_calls_typed(model, tiny_batch)
    assert counters["atom_encoder"] == 1
    assert counters["bond_encoder"] == 1
    assert counters["local"] == 1
    assert counters["drop"] == 1
    assert counters["f"] == 1
    assert counters["rho"] == 1
    assert counters["readout"] == 1
    assert counters["branch"] == 1


def test_every_new_module_receives_gradient(spec, norm, real_batch):
    import bs_train as BT

    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    model.train()
    optimizer, _ = C.BSM.make_optimizer(model)
    y = torch.full((int(real_batch["n_graphs"]), C.N_TASKS), float("nan"))
    y[:, 0] = torch.randint(0, 2, (int(real_batch["n_graphs"]),)).float()
    y[:, 1] = 1.0
    for step in range(3):
        BT.train_step(model, optimizer, real_batch, y, global_step=step, clip=5.0)
    grads = M.branch_grad_norm(model)
    for name in C.BRANCH_PARAMS:
        prefix = f"branch.{name}"
        total = sum(v for k, v in grads.items() if k == prefix or k.startswith(prefix + "."))
        assert total > 0.0, name
    # the documented degeneracy: a per-molecule constant key bias is removed by
    # the softmax, so its gradient is (numerically) zero.
    assert grads.get("branch.key.bias", 0.0) < 1e-8


def test_brute_force_equals_batched_cpu(spec, norm, tiny_batch):
    import pw_checks as CHK

    result = CHK.check_cpu_vs_batched(spec, norm, tiny_batch, device="cpu")
    assert result["passed"], result["details"]


def test_attention_entropy_is_finite_and_bounded(spec, norm, real_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    model.eval()
    entropy = M.attention_entropy(model, real_batch)
    assert entropy["n_pairs"] == int(real_batch["pair_i"].numel())
    assert len(entropy["queries"]) == C.N_QUERIES
    for query in entropy["queries"]:
        assert query["mean_entropy_nats"] >= 0.0
        assert query["mean_normalised_entropy"] <= 1.0 + 1e-6
        assert 0.0 <= query["mean_max_attention_weight"] <= 1.0
