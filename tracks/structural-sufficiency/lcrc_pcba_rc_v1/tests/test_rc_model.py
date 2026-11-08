"""Architecture invariants of the RC relation-combination branch."""

from __future__ import annotations

import math

import torch

import rc_checks as CHK
import rc_common as C
import rc_model as M


# ---------------------------------------------------------------------------
# parameter budget
# ---------------------------------------------------------------------------


def test_parameter_arithmetic_is_exact():
    expected = {
        "role_proj": 128 * 12 + 12,
        "combiner.norm": 2 * 487,
        "combiner.fc1": 487 * 96 + 96,
        "combiner.fc2": 96 * 32 + 32,
        "task_embedding": 128 * 32,
        "task_bias": 128,
    }
    for name, params in expected.items():
        assert C.BRANCH_PARAMS[name]["parameters"] == params, name
    assert C.BRANCH_ADDED == sum(expected.values()) == 56_698
    assert C.EXPECTED_PARAMETERS == 994_324 + 56_698 == 1_051_022
    assert C.EXPECTED_PARAMETERS < C.PARAMETER_CEILING
    assert 50_000 <= C.BRANCH_ADDED <= 60_000


def test_relation_input_width_arithmetic():
    assert C.TRIU == 78
    assert C.ROLE_BLOCK == 6 * 78 == 468
    assert C.X_DIM == 468 + 6 + 12 + 1 == 487


def test_build_model_reports_exact_parameter_count(spec, norm):
    for key in C.MODEL_ORDER:
        model, audit = M.build_model(key, spec, norm, C.N_TASKS, seed=0, device="cpu")
        assert int(sum(p.numel() for p in model.parameters())) == 1_051_022
        assert audit["total_parameters"] == 1_051_022
        assert audit["new_parameters"] == 56_698
        assert audit["base_parameters"] == 994_324
        assert sorted(audit["new_state_keys"]) == [
            "branch.combiner.fc1.bias",
            "branch.combiner.fc1.weight",
            "branch.combiner.fc2.bias",
            "branch.combiner.fc2.weight",
            "branch.combiner.norm.bias",
            "branch.combiner.norm.weight",
            "branch.role_proj.bias",
            "branch.role_proj.weight",
            "branch.task_bias",
            "branch.task_embedding",
        ]


def test_real_and_blind_share_new_parameter_spec(spec, norm):
    result = CHK.check_init(spec, norm)
    assert result["passed"], result
    d = result["details"]
    assert d["new_state_keys"]["REAL"] == d["new_state_keys"]["BLIND"]
    assert d["new_parameters"]["REAL"] == d["new_parameters"]["BLIND"]
    assert d["total_parameters"]["REAL"] == d["total_parameters"]["BLIND"]
    assert d["w2_prefix_equal"] is True
    assert d["per_arm_base_bitwise_equal"] == {"REAL": True, "BLIND": True}


def test_zero_init_branch_and_private_generator(spec, norm):
    for key in C.MODEL_ORDER:
        model, audit = M.build_model(key, spec, norm, C.N_TASKS, seed=0, device="cpu")
        assert torch.count_nonzero(model.branch.combiner.fc2.weight) == 0
        assert torch.count_nonzero(model.branch.combiner.fc2.bias) == 0
        assert torch.count_nonzero(model.branch.role_proj.weight) > 0
        assert torch.count_nonzero(model.branch.task_embedding) > 0
        assert torch.count_nonzero(model.branch.task_bias) == 0
        assert audit["new_module_init"]["private_generator_seed"] == C.PRIVATE_GENERATOR_OFFSETS[key]
        assert audit["new_module_init"]["global_rng_unchanged"] is True


def test_zero_init_forward_equals_w2(spec, norm, tiny_batch):
    result = CHK.check_zero_init_forward(spec, norm, tiny_batch)
    assert result["passed"], result
    for key, detail in result["details"]["per_arm"].items():
        assert detail["exact"] is True, (key, detail)


# ---------------------------------------------------------------------------
# the W2 forward is untouched
# ---------------------------------------------------------------------------


def test_single_encode_pass_and_no_atom_update(spec, norm, tiny_batch):
    model, _ = M.build_model("REAL", spec, norm, C.N_TASKS, seed=0, device="cpu")
    counts = M.count_module_calls_typed(model, tiny_batch)
    assert counts["atom_encoder"] == 1
    assert counts["bond_encoder"] == 1
    assert counts["local"] == 1
    assert counts["drop"] == 1
    assert counts["branch"] == 1
    assert counts["f"] == 1 and counts["rho"] == 1 and counts["readout"] == 1


def test_branch_reads_the_post_dropout_h(spec, norm, tiny_batch):
    """The branch input must be the *same tensor* W2 feeds into its readout."""
    model, _ = M.build_model("REAL", spec, norm, C.N_TASKS, seed=0, device="cpu")
    seen: dict[str, torch.Tensor] = {}
    handles = [
        model.drop.register_forward_hook(lambda _m, _i, out: seen.__setitem__("h", out)),
        model.branch.register_forward_pre_hook(lambda _m, inp: seen.__setitem__("branch_in", inp[1])),
    ]
    try:
        model.eval()
        with torch.no_grad():
            model(tiny_batch)
    finally:
        for h in handles:
            h.remove()
    assert seen["branch_in"] is seen["h"]


def test_atom_permutation_invariance(spec, norm, tiny_batch):
    result = CHK.check_permutation(spec, norm, tiny_batch)
    assert result["passed"], result


# ---------------------------------------------------------------------------
# role / combiner behaviour
# ---------------------------------------------------------------------------


def test_role_distribution_is_a_probability_vector(spec, norm, tiny_batch):
    model, _ = M.build_model("REAL", spec, norm, C.N_TASKS, seed=0, device="cpu")
    model.eval()
    with torch.no_grad():
        h = torch.randn(int(tiny_batch["atom_feat"].shape[0]), model.width)
        R = model.branch.roles(h)
    assert tuple(R.shape) == (int(tiny_batch["atom_feat"].shape[0]), C.N_ROLES)
    assert torch.all(R >= 0)
    assert torch.allclose(R.sum(dim=1), torch.ones(R.shape[0]), atol=1e-6)


def test_branch_output_shape_and_zero_residual(spec, norm, tiny_batch):
    model, _ = M.build_model("BLIND", spec, norm, C.N_TASKS, seed=0, device="cpu")
    model.eval()
    with torch.no_grad():
        logits, aux = model(tiny_batch, return_aux=True)
    assert tuple(logits.shape) == (int(tiny_batch["n_graphs"]), C.N_TASKS)
    assert float(aux["residual_logits"].abs().max()) == 0.0
    assert torch.equal(aux["base_logits"] + aux["residual_logits"], logits)
    assert tuple(aux["w"].shape) == (int(tiny_batch["n_graphs"]), C.N_BUCKETS)
    assert torch.allclose(aux["w"].sum(dim=1), torch.ones(int(tiny_batch["n_graphs"])), atol=1e-6)


def test_gradient_path_is_active_after_a_few_steps(spec, norm, tiny_batch):
    result = CHK.check_gradient_path(spec, norm, [tiny_batch], steps=3)
    assert result["passed"], result
    d = result["details"]
    assert d["role_projection_grad_nonzero"]
    assert d["combiner_fc1_grad_nonzero"]
    assert d["combiner_fc2_grad_nonzero"]
    assert d["task_embedding_grad_nonzero"]
    # fc2 is zero at step 0, so the trunk of the branch is silent on the first step
    assert d["trace"][0]["branch_grad_norm"]["branch.role_proj.weight"] == 0.0
    assert d["trace"][-1]["branch_grad_norm"]["branch.role_proj.weight"] > 0.0


def test_branch_diagnostics_are_finite(spec, norm, tiny_batch):
    model, _ = M.build_model("REAL", spec, norm, C.N_TASKS, seed=0, device="cpu")
    diag = M.branch_diagnostics(model, tiny_batch)
    assert diag["kind"] == "real"
    assert math.isfinite(diag["residual"]["rms"])
    assert math.isfinite(diag["residual_over_base"])
    assert len(diag["w_mean"]) == C.N_BUCKETS
    assert len(diag["graphs_with_empty_bucket"]) == C.N_BUCKETS
