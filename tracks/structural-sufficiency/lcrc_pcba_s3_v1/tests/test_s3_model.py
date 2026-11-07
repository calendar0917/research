"""Model-side contracts of the two new arms.

The load-bearing properties are (a) the exact parameter arithmetic of the
budget-matched FFN, (b) a bitwise-identical W2 base state for both arms,
(c) the zero-initialised branch, (d) the documented factored per-path encoder,
and (e) that the new parameters actually receive gradients once the output
layer moves away from zero.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import bs_model as BSM
import s3_common as C
import s3_model as M


# ---------------------------------------------------------------------------
# synthetic path tensors (no data cache needed)
# ---------------------------------------------------------------------------


def synthetic_s3_batch(n_atoms: int = 12, n_paths: int = 30, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(int(seed))
    root = torch.randint(0, n_atoms, (n_paths,), generator=g)
    end = torch.randint(0, n_atoms, (n_paths,), generator=g)
    end = torch.where(end == root, (end + 1) % n_atoms, end)
    return {
        "s3_path_atom": root,
        "s3_path_end": end,
        "s3_mid1_type": torch.randint(0, 16, (n_paths,), generator=g),
        "s3_mid2_type": torch.randint(0, 16, (n_paths,), generator=g),
        "s3_bond1": torch.randint(1, 6, (n_paths,), generator=g),
        "s3_bond2": torch.randint(1, 6, (n_paths,), generator=g),
        "s3_bond3": torch.randint(1, 6, (n_paths,), generator=g),
        "s3_dist": torch.randint(1, 4, (n_paths,), generator=g),
    }


def empty_s3_batch() -> dict[str, torch.Tensor]:
    return {k: torch.zeros(0, dtype=torch.long) for k in synthetic_s3_batch(1, 1)}


def fresh_branch(spec, *, zero_out: bool = True) -> M.S3Branch:
    branch = M.S3Branch(spec, width=128, atom_out=192, s3_dim=C.S3_DIM)
    if zero_out:
        with torch.no_grad():
            branch.out.weight.zero_()
            branch.out.bias.zero_()
    return branch


@pytest.fixture(scope="module")
def branch(spec):
    torch.manual_seed(0)
    return fresh_branch(spec)


# ---------------------------------------------------------------------------
# parameter arithmetic
# ---------------------------------------------------------------------------


def test_s3_parameter_count_is_exact(spec, norm):
    model, audit = M.build_model("S3", spec, norm, C.N_TASKS, seed=0, device="cpu")
    assert audit["new_parameters"] == C.S3_ADDED == 24_208
    assert audit["total_parameters"] == C.EXPECTED_PARAMETERS["S3"] == 1_018_532
    assert int(sum(p.numel() for p in model.s3_branch.parameters())) == C.S3_ADDED
    # the three additive blocks of the specification
    we = 192 * 48 + 48
    embeds = 16 * 48 * 2 + 5 * 48 * 3 + 3 * 48
    out = 97 * 128 + 128
    assert we + embeds + out == C.S3_ADDED


def test_ffn_budget_match(spec, norm):
    choice = C.choose_ffn_hidden(C.S3_ADDED)
    assert choice["hidden"] == C.FFN_HIDDEN == 93
    assert C.ffn_parameters(93) == 257 * 93 + 384 == 24_285
    assert C.S3_ADDED - C.ffn_parameters(93) == C.PARAMETER_GAP == -77
    assert choice["absolute_difference"] == 77
    assert C.PARAMETER_GAP_FRACTION <= C.PARAMETER_GAP_MAX_FRACTION
    # the next hidden size would have overshot by more than the gate
    assert abs(C.ffn_parameters(94) - C.S3_ADDED) > abs(C.ffn_parameters(93) - C.S3_ADDED)
    assert C.PARAMETER_GAP_FRACTION == pytest.approx(77 / 24_208, rel=1e-9)
    model, audit = M.build_model("F", spec, norm, C.N_TASKS, seed=0, device="cpu")
    assert audit["new_parameters"] == C.F_ADDED == 24_285
    assert audit["total_parameters"] == C.EXPECTED_PARAMETERS["F"] == 1_018_609
    assert int(model.ffn.fc1.in_features) == 128 and int(model.ffn.fc1.out_features) == 93


def test_recipes_are_identical(spec, norm):
    keys = ("optimizer", "adam_betas", "adam_eps", "learning_rate", "weight_decay", "warmup_epochs",
            "warmup_start_lr", "end_lr", "dropout", "batch_size", "epochs", "gradient_clip", "precision",
            "early_stopping", "seed", "eval_every", "eval_epochs", "widths", "lr_schedule", "model_selection")
    s3 = C.recipe_for_model("S3")
    f = C.recipe_for_model("F")
    for key in keys:
        assert s3[key] == f[key], key
    assert s3["model_selection"] == "argmax official Validation macro AP over epochs 1..40 (earliest epoch wins ties)"
    assert s3["learning_rate"] == 3e-4 and s3["weight_decay"] == 1e-3 and s3["dropout"] == 0.10
    assert s3["gradient_clip"] == 5.0 and s3["batch_size"] == 128 and s3["epochs"] == 40


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


def test_both_arms_share_the_bitwise_w2_base(spec, norm):
    base, base_audit = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=0, device="cpu")
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    base_sha = M.state_sha256(base_state)
    for key in C.MODEL_ORDER:
        model, audit = M.build_model(key, spec, norm, C.N_TASKS, seed=0, device="cpu")
        own = {k: v.detach().clone() for k, v in model.state_dict().items() if k in base_state}
        assert set(own) == set(base_state), key
        assert M.state_sha256(own) == base_sha, key
        assert audit["base_state_sha256"] == base_sha, key
        assert audit["original_w2_subset"]["all_bitwise_equal"], key
        assert audit["base_parameters"] == C.BASE_PARAMETERS, key


def test_new_modules_are_zero_at_init(branch):
    assert torch.count_nonzero(branch.out.weight) == 0
    assert torch.count_nonzero(branch.out.bias) == 0
    batch = synthetic_s3_batch()
    a = torch.randn(12, 192, generator=torch.Generator().manual_seed(3))
    residual, aux = branch(batch, a)
    assert residual.shape == (12, 128)
    assert torch.count_nonzero(residual) == 0
    assert residual.abs().max().item() == 0.0


def test_private_generator_leaves_global_rng_untouched(spec, norm):
    model, audit = M.build_model("S3", spec, norm, C.N_TASKS, seed=0, device="cpu")
    assert audit["new_module_init"]["global_rng_unchanged"] is True
    # the new-module initialiser itself never advances the global torch stream
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    info = M.initialize_new_modules(model, "S3", seed=0)
    assert torch.equal(before, torch.get_rng_state())
    assert info["global_rng_unchanged"] is True
    assert info["out_is_zero"] is True
    # a different private offset gives different new parameters but the same base
    torch.manual_seed(1234)
    model2, audit2 = M.build_model("S3", spec, norm, C.N_TASKS, seed=0, device="cpu")
    assert M.state_sha256({k: v for k, v in model2.state_dict().items() if k in model.state_dict()}) == M.state_sha256(
        {k: v for k, v in model.state_dict().items()}
    )
    assert C.PRIVATE_GENERATOR_OFFSETS["S3"] != C.PRIVATE_GENERATOR_OFFSETS["F"]
    del model2, audit2


# ---------------------------------------------------------------------------
# the documented per-path encoder and rooted aggregation
# ---------------------------------------------------------------------------


def test_path_encoder_matches_the_explicit_formula(branch):
    batch = synthetic_s3_batch(n_atoms=7, n_paths=9, seed=11)
    # roots in 0..2, endpoints in 3..6: the two sets are disjoint by construction
    batch["s3_path_atom"] = torch.arange(9) % 3
    batch["s3_path_end"] = 3 + torch.arange(9) % 4
    a = torch.randn(7, 192, generator=torch.Generator().manual_seed(5))
    z = branch.encode_paths(batch, a)
    gx = branch.we(a)[batch["s3_path_end"]]
    manual = (
        gx
        + branch.e_mid1(batch["s3_mid1_type"])
        + branch.e_mid2(batch["s3_mid2_type"])
        + branch.e_bond1(batch["s3_bond1"] - 1)
        + branch.e_bond2(batch["s3_bond2"] - 1)
        + branch.e_bond3(batch["s3_bond3"] - 1)
        + branch.e_dist(batch["s3_dist"] - 1)
    )
    assert torch.allclose(z, torch.nn.functional.silu(manual), atol=1e-6)
    # the endpoint is the only atom embedding that enters the path encoder
    a2 = a.clone()
    a2[0] += 5.0
    z2 = branch.encode_paths(batch, a2)
    touched = (batch["s3_path_end"] == 0).nonzero().flatten()
    assert touched.numel() == 0 or not torch.allclose(z, z2)
    # the root embedding never enters z (roots disjoint from endpoints here)
    a3 = a.clone()
    roots = torch.unique(batch["s3_path_atom"])
    ends = torch.unique(batch["s3_path_end"])
    assert not bool(set(roots.tolist()) & set(ends.tolist()))
    a3[roots] += 3.0
    assert torch.equal(z, branch.encode_paths(batch, a3))


def test_root_aggregation_matches_the_manual_statistics(branch):
    batch = synthetic_s3_batch(n_atoms=6, n_paths=14, seed=17)
    a = torch.randn(6, 192, generator=torch.Generator().manual_seed(7))
    z = branch.encode_paths(batch, a)
    s_r, counts = branch.root_inputs(z, batch["s3_path_atom"], 6)
    d = C.S3_DIM
    for r in range(6):
        mask = batch["s3_path_atom"] == r
        m = int(mask.sum())
        assert int(counts[r]) == m
        if m == 0:
            assert torch.count_nonzero(s_r[r]) == 0
            continue
        total = z[mask].sum(dim=0)
        assert torch.allclose(s_r[r, :d], total / m, atol=1e-6)
        assert torch.allclose(s_r[r, d : 2 * d], total / np.sqrt(m), atol=1e-6)
        assert torch.allclose(s_r[r, 2 * d :], torch.tensor([np.log1p(m)], dtype=torch.float32), atol=1e-6)
    assert s_r.shape == (6, 2 * C.S3_DIM + 1) == (6, 97)


def test_pathless_roots_return_the_output_bias(spec):
    branch = fresh_branch(spec)
    with torch.no_grad():
        branch.out.bias.fill_(0.375)
        branch.out.weight.normal_(0, 0.02, generator=torch.Generator().manual_seed(9))
    a = torch.randn(4, 192, generator=torch.Generator().manual_seed(11))
    residual, aux = branch(empty_s3_batch(), a)
    assert residual.shape == (4, 128)
    assert aux["n_paths"] == 0
    assert torch.allclose(residual, torch.full((4, 128), 0.375), atol=1e-7)


def test_empty_paths_are_safe_with_zero_init(spec):
    branch = fresh_branch(spec)
    a = torch.randn(5, 192, generator=torch.Generator().manual_seed(13))
    residual, _ = branch(empty_s3_batch(), a)
    assert torch.isfinite(residual).all()
    assert torch.count_nonzero(residual) == 0


def test_gradients_reach_the_encoder_once_the_output_is_nonzero(spec):
    branch = fresh_branch(spec)
    batch = synthetic_s3_batch(n_atoms=8, n_paths=20, seed=23)
    a = torch.randn(8, 192, generator=torch.Generator().manual_seed(29))
    residual, _ = branch(batch, a)
    residual.sum().backward()
    # with a zero output layer the encoder is deliberately frozen at step 0
    assert torch.count_nonzero(branch.we.weight.grad) == 0
    assert torch.count_nonzero(branch.e_bond1.weight.grad) == 0
    branch.zero_grad(set_to_none=True)
    with torch.no_grad():
        branch.out.weight.normal_(0, 0.01, generator=torch.Generator().manual_seed(31))
    branch(batch, a)[0].sum().backward()
    for name, param in (
        ("we", branch.we.weight),
        ("e_mid1", branch.e_mid1.weight),
        ("e_mid2", branch.e_mid2.weight),
        ("e_bond1", branch.e_bond1.weight),
        ("e_bond2", branch.e_bond2.weight),
        ("e_bond3", branch.e_bond3.weight),
        ("e_dist", branch.e_dist.weight),
        ("out", branch.out.weight),
    ):
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert torch.count_nonzero(param.grad) > 0, name


def test_state_hash_is_deterministic_and_sensitive(spec, norm):
    model, _ = M.build_model("F", spec, norm, C.N_TASKS, seed=0, device="cpu")
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    a = M.state_sha256(state)
    assert a == M.state_sha256({k: v.clone() for k, v in state.items()})
    key = sorted(state)[0]
    state[key] = state[key] + 1.0
    assert M.state_sha256(state) != a


@pytest.mark.slow
def test_epoch0_parity_on_a_real_batch(spec):
    """With the residual at zero the model *is* W2: identical logits, bitwise."""
    task = C.load_task_with_s3()
    plan = C.BC.SD.make_plan40(task, seed=int(C.SEED))
    rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)[:8]
    batch = C.collate_with_s3(task, task.s3, rows.tolist())
    model, _ = M.build_model("S3", task.spec, task.norm, C.N_TASKS, seed=0, device="cpu")
    w2, _ = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=0, device="cpu")
    model.eval()
    w2.eval()
    with torch.no_grad():
        assert torch.equal(model(batch, s3_zero=True), w2(batch))
