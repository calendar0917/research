from __future__ import annotations

import torch
import torch.nn.functional as F

import screen_common as C
import screen_models as M


def test_ab_have_identical_initialisation(spec, norm):
    for seed in C.SEEDS:
        a, info_a = M.build_model("A", spec, norm, seed, "cpu")
        b, info_b = M.build_model("B", spec, norm, seed, "cpu")
        assert info_a["state_hash"] == info_b["state_hash"]
        assert info_a["parameters"] == info_b["parameters"] == C.LCRC_PARAMS_REFERENCE
        assert info_b["effective_parameters"] < info_b["parameters"]
        assert info_b["s2_off"]["active_train"] and info_b["s2_off"]["active_eval"]
        assert a._s2_controller is None


def test_s2_slice_targets_the_local_mlp_input(spec, norm, fit_batch):
    batch, _ = fit_batch
    a, info_a = M.build_model("A", spec, norm, 0, "cpu")
    b, info_b = M.build_model("B", spec, norm, 0, "cpu")
    controller = b._s2_controller
    assert (controller.start, controller.end, controller.in_dim) == (320, 520, 529)
    with M.LocalMLPInputCapture(a) as cap_a, M.LocalMLPInputCapture(b) as cap_b:
        a(batch)
        b(batch)
    ia, ib = cap_a.inputs[0], cap_b.inputs[0]
    s2 = slice(controller.start, controller.end)
    assert torch.equal(ib[:, s2], torch.zeros_like(ib[:, s2]))
    assert torch.equal(ia[:, : controller.start], ib[:, : controller.start])
    assert torch.equal(ia[:, controller.end :], ib[:, controller.end :])
    assert float(ia[:, s2].abs().max()) > 0


def test_s2_off_is_invariant_but_full_is_not(spec, norm, fit_batch):
    batch, _ = fit_batch
    a, _ = M.build_model("A", spec, norm, 0, "cpu")
    b, _ = M.build_model("B", spec, norm, 0, "cpu")
    perturbed = dict(batch)
    perturbed["path_cat"] = torch.roll(batch["path_cat"], 1, 0)
    perturbed["path_end63"] = torch.roll(batch["path_end63"], 1, 0)
    perturbed["path_end"] = torch.roll(batch["path_end"], 1, 0)
    perturbed["path_atom"] = torch.roll(batch["path_atom"], 1, 0)
    perturbed["path_closure"] = 1 - batch["path_closure"]
    with torch.no_grad():
        ya, ya_p = a(batch), a(perturbed)
        yb, yb_p = b(batch), b(perturbed)
    assert float((ya_p - ya).abs().max()) > 1e-6
    assert torch.equal(yb_p, yb)


def test_s2_gradients(spec, norm, fit_batch):
    batch, _ = fit_batch
    target = torch.linspace(-1, 1, int(batch["n_graphs"]), dtype=torch.float32)
    for arm in ("A", "B"):
        model, _ = M.build_model(arm, spec, norm, 0, "cpu")
        model.zero_grad(set_to_none=True)
        F.mse_loss(model(batch).view(-1), target).backward()
        names = M.dead_parameter_names_lcrc(model, model._s2_controller)
        params = dict(model.named_parameters())
        s2_grads = {n: float(params[n].grad.abs().sum()) for n in names}
        first_w = params["local.mlp.lins.0.weight"].grad
        if arm == "A":
            assert all(v > 0 for v in s2_grads.values())
            assert float(first_w[:, 320:520].abs().sum()) > 0
        else:
            assert all(v == 0.0 for v in s2_grads.values())
            assert float(first_w[:, 320:520].abs().sum()) == 0.0
        assert float(first_w[:, :320].abs().sum()) > 0


def test_soup_is_safe(spec, norm):
    for arm in C.NN_ARMS:
        model, _ = M.build_model(arm, spec, norm, 0, "cpu")
        report = M.soup_safety_report(model)
        assert report["all_state_entries_float"], report["non_float_entries"]
        assert report["n_state_entries"] > 0


def test_state_hash_is_stable(spec, norm):
    a1, info1 = M.build_model("A", spec, norm, 1, "cpu")
    a2, info2 = M.build_model("A", spec, norm, 1, "cpu")
    assert info1["state_hash"] == info2["state_hash"]
    a3, info3 = M.build_model("A", spec, norm, 2, "cpu")
    assert info3["state_hash"] != info1["state_hash"]
