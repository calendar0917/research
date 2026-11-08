"""Architecture invariants of the selective three-hop path branch."""

from __future__ import annotations

import math

import pytest
import torch

import sp_common as C
import sp_model as M


# ---------------------------------------------------------------------------
# parameter budget
# ---------------------------------------------------------------------------


def test_parameter_arithmetic_is_exact():
    assert C.SP_ADDED == 17_888
    assert C.S3_ADDED == 24_208
    assert C.S3_PARAMETERS == 1_018_532
    assert C.SP_PARAMETERS == 1_036_420
    assert C.EXPECTED_PARAMETERS["SP"] == 1_036_420
    assert C.SP_BRANCH["q"]["parameters"] == 9_264
    assert C.SP_BRANCH["k"]["parameters"] == 2_352
    assert C.SP_BRANCH["out"]["parameters"] == 6_272
    assert sum(int(v["parameters"]) for v in C.SP_BRANCH.values()) == 17_888
    assert C.BASE_PARAMETERS + C.S3_ADDED + C.SP_ADDED == C.SP_PARAMETERS
    assert abs(C.SP_ADDED_FRACTION_OF_S3 - 17_888 / 1_018_532) < 1e-12


def test_build_model_reports_exact_parameter_count(spec, norm):
    model, audit = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    assert int(sum(p.numel() for p in model.parameters())) == 1_036_420
    assert audit["total_parameters"] == 1_036_420
    assert audit["new_parameters"] == 17_888
    assert audit["inherited_s3_parameters"] == 1_018_532
    assert audit["base_parameters"] == 994_324
    assert sorted(audit["new_state_keys"]) == [
        "selective.k.bias",
        "selective.k.weight",
        "selective.out.bias",
        "selective.out.weight",
        "selective.q.bias",
        "selective.q.weight",
    ]


def test_selective_output_is_zero_initialised_and_qk_are_not(spec, norm):
    model, audit = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    assert torch.count_nonzero(model.selective.out.weight) == 0
    assert torch.count_nonzero(model.selective.out.bias) == 0
    assert torch.count_nonzero(model.selective.q.weight) > 0
    assert torch.count_nonzero(model.selective.k.weight) > 0
    # default nn.Linear init must not be what produced q/k: it uses its own generator
    assert audit["new_module_init"]["private_generator_seed"] == C.SP_PRIVATE_SEED == 1201
    assert audit["new_module_init"]["global_rng_unchanged"] is True


def test_build_consumes_exactly_the_historical_rng_stream(spec, norm):
    """The inherited W2 construction consumes global RNG; the SP-specific part
    (selective branch + private generator) must consume none.  The net result is
    that the SP build leaves the dropout stream where the historical S3 build
    left it."""
    torch.manual_seed(1234)
    base, _ = M.BSM.build_model("W2", spec, norm, C.N_TASKS, seed=0, device="cpu")
    expected = torch.get_rng_state().clone()
    M.M3.build_model("S3", spec, norm, C.N_TASKS, seed=0, device="cpu", base_model=base)
    assert torch.equal(expected, torch.get_rng_state())

    torch.manual_seed(1234)
    M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu", verify_parameters=False)
    assert torch.equal(expected, torch.get_rng_state())


def test_init_parity_with_history(task):
    model, audit = M.build_model(task.spec, task.norm, C.N_TASKS, seed=0, device="cpu")
    assert audit["base_state_sha256"] == C.S3["base_state_sha256_at_init"]
    assert audit["s3_inherited_state_sha256"] == C.S3["model_state_sha256_at_init"]
    assert audit["base_trunk_init_fingerprint"] == C.W2["trunk_init_fingerprint"]
    assert audit["s3_rebuilt_state_sha256"] == audit["s3_inherited_state_sha256"]
    assert audit["model_state_sha256"] != audit["s3_inherited_state_sha256"]
    assert audit["original_w2_subset"]["all_bitwise_equal"] is True
    assert audit["inherited_w2_plus_s3_subset"]["all_bitwise_equal"] is True


# ---------------------------------------------------------------------------
# grouped softmax
# ---------------------------------------------------------------------------


def test_segment_softmax_normalises_inside_each_root():
    score = torch.tensor([[1.0, 2.0], [0.5, 0.5], [3.0, -1.0], [0.0, 0.0]], dtype=torch.float64)
    index = torch.tensor([0, 0, 1, 2])
    w = M.segment_softmax(score, index, 4)
    sums = torch.zeros(4, 2, dtype=torch.float64).index_add_(0, index, w)
    assert torch.allclose(sums[0], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(sums[1], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(sums[2], torch.ones(2, dtype=torch.float64))
    assert torch.equal(sums[3], torch.zeros(2, dtype=torch.float64))  # empty root


def test_segment_softmax_matches_reference_and_is_group_local():
    torch.manual_seed(0)
    score = torch.randn(9, 2, dtype=torch.float64)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    w = M.segment_softmax(score, index, 4)
    for g in (0, 1, 2):
        mask = index == g
        ref = torch.softmax(score[mask], dim=0)
        assert torch.allclose(w[mask], ref, atol=1e-12)
    # perturbing a different group must not move this group's weights
    score2 = score.clone()
    score2[index == 1] += 5.0
    w2 = M.segment_softmax(score2, index, 4)
    assert torch.equal(w[index == 0], w2[index == 0])
    assert torch.equal(w[index == 2], w2[index == 2])


def test_segment_softmax_is_stable_for_large_scores():
    score = torch.tensor([[1e4, -1e4], [1e4 - 1.0, -1e4]], dtype=torch.float32)
    index = torch.tensor([0, 0])
    w = M.segment_softmax(score, index, 1)
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(0), torch.ones(2))


def test_segment_softmax_empty_input():
    empty = torch.zeros((0, 2))
    w = M.segment_softmax(empty, torch.zeros(0, dtype=torch.long), 3)
    assert w.shape == (0, 2)


# ---------------------------------------------------------------------------
# attn - mean algebra
# ---------------------------------------------------------------------------


def _branch(s3_dim=C.S3_DIM, atom_out=192, width=128, seed=3, zero_out=True):
    torch.manual_seed(seed)
    branch = M.SelectivePathBranch(atom_out=atom_out, s3_dim=s3_dim, n_heads=C.SP_HEADS, width=width)
    if zero_out:
        M.initialize_selective(branch)
    return branch


def test_single_path_root_has_zero_delta_and_zero_residual():
    branch = _branch()
    a = torch.randn(3, 192)
    z = torch.randn(3, C.S3_DIM)
    root = torch.tensor([0, 1, 2])
    mean_z = z.clone()
    counts = torch.ones(3)
    residual, aux = branch(a=a, z=z, root=root, mean_z=mean_z, counts=counts)
    assert torch.equal(aux["selective_delta"], torch.zeros_like(aux["selective_delta"]))
    assert torch.equal(residual, torch.zeros_like(residual))


def test_uniform_attention_gives_zero_delta():
    branch = _branch(zero_out=False)
    with torch.no_grad():
        branch.q.weight.zero_()
        branch.q.bias.zero_()  # identical query for every root -> constant scores
    a = torch.zeros(2, 192)
    z = torch.randn(5, C.S3_DIM)
    root = torch.tensor([0, 0, 1, 1, 1])
    mean = torch.zeros(2, C.S3_DIM).index_add_(0, root, z) / torch.tensor([2.0, 3.0]).unsqueeze(1)
    counts = torch.tensor([2.0, 3.0])
    residual, aux = branch(a=a, z=z, root=root, mean_z=mean, counts=counts)
    assert float(aux["selective_delta"].abs().max()) < 1e-6
    # with delta == 0 the residual collapses to the output bias, nothing else
    expected = branch.out.bias.detach().expand_as(residual)
    assert torch.allclose(residual, expected, atol=1e-6)


def test_empty_root_residual_is_forced_to_zero_after_bias_learning():
    branch = _branch(zero_out=False)
    with torch.no_grad():
        branch.out.bias.fill_(0.5)
        branch.out.weight.normal_()
    a = torch.randn(2, 192)
    z = torch.randn(3, C.S3_DIM)
    root = torch.tensor([0, 0, 0])
    mean = z.mean(0, keepdim=True)
    counts = torch.tensor([3.0, 0.0])
    residual, _aux = branch(a=a, z=z, root=root, mean_z=mean, counts=counts)
    assert torch.equal(residual[1], torch.zeros(C.WIDTHS["width"]))
    assert float(residual[0].abs().max()) > 0.0


def test_attention_weights_sum_to_one_per_root_and_head():
    branch = _branch()
    a = torch.randn(4, 192)
    z = torch.randn(11, C.S3_DIM)
    root = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
    mean = torch.zeros(4, C.S3_DIM).index_add_(0, root, z) / torch.tensor([3.0, 2.0, 6.0, 1.0]).unsqueeze(1)
    counts = torch.tensor([3.0, 2.0, 6.0, 0.0])
    _res, aux = branch(a=a, z=z, root=root, mean_z=mean, counts=counts, return_attention=True)
    w = aux["attention_weight"]
    sums = torch.zeros(4, C.SP_HEADS).index_add_(0, root, w)
    assert float((sums[:3] - 1.0).abs().max()) < 1e-6
    assert float(sums[3].abs().max()) == 0.0
    assert ((w >= 0.0) & (w <= 1.0)).all()


def test_query_comes_from_the_static_atom_encoding_only():
    branch = _branch()
    a = torch.randn(3, 192)
    z = torch.randn(4, C.S3_DIM)
    root = torch.tensor([0, 0, 1, 2])
    counts = torch.tensor([2.0, 1.0, 1.0])
    mean = torch.zeros(3, C.S3_DIM).index_add_(0, root, z) / counts.unsqueeze(1)
    q1 = branch.q(a)
    q2 = branch.q(a)
    assert torch.equal(q1, q2)
    # perturbing z cannot move the query (z is the key/value side only)
    _r, aux_a = branch(a=a, z=z, root=root, mean_z=mean, counts=counts, return_attention=True)
    _r, aux_b = branch(a=a, z=z * 3.0, root=root, mean_z=mean, counts=counts, return_attention=True)
    assert not torch.equal(aux_a["attention_score"], aux_b["attention_score"])
    # heads are independent: shape is [n_paths, 2] and each head normalises
    assert tuple(aux_a["attention_weight"].shape) == (4, 2)


def test_path_order_permutation_is_invariant():
    branch = _branch()
    a = torch.randn(3, 192)
    z = torch.randn(7, C.S3_DIM)
    root = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    counts = torch.tensor([3.0, 2.0, 2.0])
    mean = torch.zeros(3, C.S3_DIM).index_add_(0, root, z) / counts.unsqueeze(1)
    r1, aux1 = branch(a=a, z=z, root=root, mean_z=mean, counts=counts, return_attention=True)

    # a permutation that keeps each root's paths inside that root
    order = torch.tensor([2, 0, 1, 4, 3, 6, 5])
    r2, aux2 = branch(
        a=a, z=z[order], root=root[order], mean_z=mean, counts=counts, return_attention=True
    )
    assert torch.allclose(r1, r2, atol=1e-6)
    assert torch.allclose(aux1["attention_attn"], aux2["attention_attn"], atol=1e-6)


def test_attention_stats_are_finite_and_bounded():
    torch.manual_seed(0)
    w = torch.softmax(torch.randn(9, 2), dim=0)
    root = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    counts = torch.tensor([3.0, 2.0, 4.0, 0.0])
    stats = M.attention_stats(w, root, counts, n_atoms=4)
    assert math.isfinite(stats["entropy_ratio_mean"])
    assert 0.0 <= stats["entropy_ratio_mean"] <= 1.0 + 1e-6
    assert stats["n_weights_outside_unit_interval"] == 0
    assert stats["roots_without_paths"] == 1
    assert stats["finite"] is True


# ---------------------------------------------------------------------------
# forward structure
# ---------------------------------------------------------------------------


def test_forward_calls_each_stage_exactly_once(spec, norm, tiny_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    calls = M.count_module_calls_typed(model, tiny_batch)
    assert calls["atom_encoder"] == 1
    assert calls["bond_encoder"] == 1
    assert calls["local"] == 1
    assert calls["drop"] == 1
    assert calls["s3_encode"] == 1
    assert calls["s3_out"] == 1
    assert calls["selective"] == 1
    assert float(model.drop.p) == 0.10


def test_epoch_zero_logits_equal_w2_and_s3(spec, norm, tiny_batch):
    import bs_model as BSM
    import s3_model as M3

    w2, _ = BSM.build_model("W2", spec, norm, C.N_TASKS, seed=0, device="cpu")
    s3, _ = M3.build_model("S3", spec, norm, C.N_TASKS, seed=0, device="cpu")
    sp, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    for m in (w2, s3, sp):
        m.eval()
    with torch.no_grad():
        l_w2 = w2(tiny_batch)
        l_s3 = s3(tiny_batch)
        l_sp = sp(tiny_batch)
        l_sp_zero = sp(tiny_batch, selective_zero=True)
    assert torch.equal(l_sp, l_w2)
    assert torch.equal(l_sp, l_s3)
    assert torch.equal(l_sp_zero, l_w2)


def test_zero_path_batch_is_finite_and_has_zero_selective_residual(spec, norm, lone_atom_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    with torch.no_grad():
        model.selective.out.weight.normal_()
        model.selective.out.bias.fill_(0.25)
    model.eval()
    with torch.no_grad():
        logits, aux = model(lone_atom_batch, return_aux=True)
    assert torch.isfinite(logits).all()
    assert aux["selective_delta"].numel() == 0 or float(aux["selective_delta"].abs().max()) == 0.0
    assert float(aux["r_selective"].abs().max()) == 0.0


def test_selective_branch_receives_no_propagated_state(spec, norm, tiny_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    seen: dict[str, object] = {}
    handle = model.selective.register_forward_pre_hook(
        lambda _m, args, kwargs: seen.update(kwargs), with_kwargs=True
    )
    try:
        with torch.no_grad():
            model(tiny_batch)
    finally:
        handle.remove()
    assert sorted(seen) == ["a", "counts", "mean_z", "return_attention", "root", "z"]
    assert "h" not in seen and "h_raw" not in seen and "bond_index" not in seen


def test_selective_flops_scale_with_paths_not_atoms():
    few = M.selective_flops(n_paths=1000, n_atoms=400)
    many = M.selective_flops(n_paths=4000, n_atoms=400)
    assert many["flops_total"] > few["flops_total"]
    assert few["flops_atoms_q"] == many["flops_atoms_q"]
    assert many["flops_paths_k"] == pytest.approx(4 * few["flops_paths_k"])
