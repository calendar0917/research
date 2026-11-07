"""CPU/GPU correctness tests for the scaled LCRC-A construction.

The load-bearing claim is scale-1 equivalence: the generalised builder must
reproduce the historical LCRC-A state dict, initialisation and forward pass
bitwise, and must reproduce its epoch-40 Validation AP when the historical
checkpoint is loaded read-only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import bs_common as C
import bs_model as M


def test_parameter_counts_are_exact(spec, norm):
    for scale, expected in C.EXPECTED_PARAMETERS.items():
        model, info = M.build_model(scale, spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device="cpu")
        assert info["total_parameters"] == expected
        assert info["base_parameters"] == C.EXPECTED_BASE_PARAMETERS[scale]
        assert info["readout_parameters"] == C.EXPECTED_READOUT_PARAMETERS[scale]
        assert int(sum(p.numel() for p in model.parameters())) == expected
    assert C.EXPECTED_PARAMETERS == {1: 287_644, 2: 994_324, 3: 2_120_844}


def test_protocol_width_table():
    table = C.width_table()
    assert table["atom_hidden"] == {1: 128, 2: 256, 3: 384}
    assert table["atom_out"] == {1: 96, 2: 192, 3: 288}
    assert table["bond_hidden"] == {1: 160, 2: 320, 3: 480}
    assert table["bond_out"] == {1: 96, 2: 192, 3: 288}
    assert table["stat_rank"] == {1: 32, 2: 64, 3: 96}
    assert table["local_hidden"] == {1: 192, 2: 384, 3: 576}
    assert table["width"] == {1: 64, 2: 128, 3: 192}
    assert table["rho_hidden"] == {1: 64, 2: 128, 3: 192}
    assert table["scorer_hidden"] == {1: 32, 2: 64, 3: 96}
    assert table["pool_hidden"] == {1: 88, 2: 176, 3: 264}
    assert C.FROZEN_WIDTHS == {"n_heads": 4, "n_out": 128}


def test_every_module_width_scales(models):
    for scale, (model, info) in models.items():
        observed = info["width_audit"]["observed"]
        assert observed["atom_hidden"]["out"] == 128 * scale
        assert observed["atom_out"]["out"] == 96 * scale
        assert observed["bond_hidden"]["out"] == 160 * scale
        assert observed["bond_out"]["out"] == 96 * scale
        assert observed["local_hidden"]["out"] == 192 * scale
        assert observed["local_width"]["out"] == 64 * scale
        assert observed["local_norm"] == 64 * scale
        assert observed["rho_hidden"]["out"] == 64 * scale
        assert observed["pool_hidden"]["out"] == 88 * scale
        assert observed["n_heads"] == 4
        assert all(h["out"] == 32 * scale for h in observed["scorer_hidden"])
        assert all(h["in"] == 64 * scale for h in observed["scorer_hidden"])
        assert observed["readout_out"]["out"] == 128
        assert observed["stat_rank"]["nbr_emb"][1] == 32 * scale
        assert observed["stat_rank"]["path_emb"][1] == 32 * scale
        assert observed["stat_rank"]["end_emb"][1] == 32 * scale
        assert observed["stat_rank"]["path_u"][1] == 32 * scale
        assert observed["stat_rank"]["path_v"][1] == 32 * scale
        assert observed["stat_rank"]["joint_proj"]["out"] == 32 * scale
        assert observed["f"][0]["out"] == 64 * scale
        assert observed["f"][1]["out"] == 128
        assert info["width_audit"]["all_match"]


def test_readout_does_not_keep_historical_defaults(models):
    """The 64/32/88 historical constants must not survive at scale 2/3."""
    for scale in (2, 3):
        model, _ = models[scale]
        assert model.readout.fc1.in_features == 4 * 64 * scale
        assert model.readout.fc1.out_features == 88 * scale
        assert model.readout.fc2.in_features == 88 * scale
        for head in model.readout.heads:
            assert head.fc1.in_features == 64 * scale
            assert head.fc1.out_features == 32 * scale
        assert model.local.mlp.lins[1].out_features == 64 * scale
        assert model.rho.lins[0].in_features == 42 + 2 * (64 * scale) + 1
    # and the scale-1 model is exactly the historical shape
    model, _ = models[1]
    assert model.readout.fc1.in_features == 256
    assert model.readout.fc1.out_features == 88
    assert model.local.mlp.lins[1].out_features == 64


def test_local_input_width_formula(spec, models):
    for scale, (model, _) in models.items():
        cfg = C.config_for_scale(scale)
        expected = 3 * 96 * scale + 4 * 32 * scale + 8 + spec.n_scalars
        assert model.local.mlp.lins[0].in_features == expected
        assert 3 * 96 * scale + 4 * 32 * scale + 8 + spec.n_scalars == expected


def test_frozen_input_vocabularies(models, spec):
    for scale, (model, _) in models.items():
        assert model.local.nbr_emb.weight.shape == (80, 32 * scale)
        assert model.local.path_emb.weight.shape == (400, 32 * scale)
        assert model.local.end_emb.weight.shape == (80, 32 * scale)
        assert model.local.path_u.weight.shape == (400, 32 * scale)
        assert model.local.path_v.weight.shape == (80, 32 * scale)
        assert model.local.close_emb.weight.shape == (2, 8)
        assert model.local.path_gate.weight.shape == (400, 1)
        assert model.atom_encoder.lins[0].in_features == spec.atom_feat_dim == 51
        assert model.bond_encoder.lins[0].in_features == 3 * 96 * scale + spec.bond_feat_dim
        assert spec.bond_feat_dim == 13
        assert spec.global_dim == 42
        assert spec.n_scalars == 9


def test_scale1_matches_historical_state_keys(spec, norm, models):
    trunk, trunk_info, _base = C.NM.build_reference(spec, norm, C.N_TASKS, C.MODEL_SEED)
    ref_a = C.NM.NMPModel(spec, norm, C.N_TASKS, "A")
    ref_keys = set(ref_a.state_dict())
    assert set(trunk.state_dict()) < ref_keys
    ours, info = models[1]
    assert set(ours.state_dict()) == ref_keys
    assert info["n_state_tensors"] == len(ref_keys)
    assert trunk_info["parameters"] == 245_184  # the C1 trunk the historical A was built on
    assert info["base_parameters"] == trunk_info["parameters"]
    assert info["total_parameters"] == 287_644


def test_scale1_init_bitwise_equal_historical_A(spec, norm, models):
    _ref_model, _info, base_init = C.NM.build_reference(spec, norm, C.N_TASKS, C.MODEL_SEED)
    ref_a, _audit = C.NM.build_arm(
        "A", spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device="cpu",
        base_init_state=base_init, zero_residual=True,
    )
    ours, _info = models[1]
    ref_state = ref_a.state_dict()
    our_state = ours.state_dict()
    assert set(ref_state) == set(our_state)
    for key in sorted(ref_state):
        assert torch.equal(ref_state[key].cpu(), our_state[key].cpu()), key


def test_scale1_trunk_fingerprint_matches_recorded(models):
    _model, info = models[1]
    assert info["trunk_init_fingerprint"] == C.A0["base_init_fingerprint"]


def test_scale1_forward_equals_historical_A(spec, norm, models, batch):
    ref_a, _audit = C.NM.build_arm(
        "A", spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device="cpu",
        base_init_state=C.NM.build_reference(spec, norm, C.N_TASKS, C.MODEL_SEED)[2],
        zero_residual=True,
    )
    ours, _info = models[1]
    ref_a.eval()
    ours.eval()
    with torch.inference_mode():
        lref = ref_a(batch)
        lour = ours(batch)
    assert float((lref - lour).abs().max().item()) == 0.0


@pytest.mark.slow
def test_scale1_historical_checkpoint_forward(spec, norm, batch):
    payload = torch.load(C.A0_CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    ours = M.build_model(1, spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device="cpu")[0]
    report = ours.load_state_dict(state, strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    ref = C.NM.NMPModel(spec, norm, C.N_TASKS, "A")
    ref.load_state_dict(state, strict=True)
    ref.eval()
    ours.eval()
    with torch.inference_mode():
        assert float((ref(batch) - ours(batch)).abs().max().item()) == 0.0


def test_one_shot_encoding_and_post_local_isolation(models, batch):
    for scale, (model, _) in models.items():
        calls = M.count_module_calls(model, batch)
        assert calls == {"atom_encoder": 1, "bond_encoder": 1, "local": 1, "f": 1, "rho": 1, "readout": 1}
        model.eval()
        with torch.inference_mode():
            _a, _b, _s, h = model.encode_h(batch)
            full = model(batch)
            minimal = {"atom_batch": batch["atom_batch"], "n_graphs": int(batch["n_graphs"]),
                       "global": batch["global"]}
            post = model.post_local(h, minimal)
        assert float((full - post).abs().max().item()) == 0.0


def test_atom_renumbering_invariance(models, batch):
    for scale, (model, _) in models.items():
        perm_batch, p = M.permute_atoms(batch)
        model.eval()
        with torch.inference_mode():
            a = model(batch)
            b = model(perm_batch)
            attr_a = model.f(model.encode_h(batch)[3])
            attr_b = model.f(model.encode_h(perm_batch)[3])
        assert float((a - b).abs().max().item()) <= 2e-5
        assert float((attr_b - attr_a[p]).abs().max().item()) == 0.0
        assert torch.equal(batch["atom_batch"], perm_batch["atom_batch"])


def test_forward_backward_health(spec, norm, batch, task):
    rows = [int(x) for x in task.train_rows[:8]]
    y = torch.as_tensor(np.asarray(task.labels[np.asarray(rows)]), dtype=torch.float32)
    import bs_train as T

    for scale in C.SCALES:
        model, _ = M.build_model(scale, spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device="cpu")
        optimizer, audit = M.make_optimizer(model)
        assert audit["n_parameters"] == C.EXPECTED_PARAMETERS[scale]
        model.train()
        # step 0: the readout output layer is zero, so the whole readout branch
        # has zero gradient until the first update (frozen historical A rule).
        T.train_step(model, optimizer, batch, y, global_step=0, clip=5.0)
        step0_zero = [n for n, p in model.named_parameters() if float(p.grad.abs().max()) == 0.0]
        assert any(n.startswith("readout.heads.0.fc1") for n in step0_zero)
        out = T.train_step(model, optimizer, batch, y, global_step=1, clip=5.0)
        out = T.train_step(model, optimizer, batch, y, global_step=2, clip=5.0)
        assert math.isfinite(out["loss"]) and math.isfinite(out["grad_norm"])
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing
        zeros = [n for n, p in model.named_parameters() if float(p.grad.abs().max()) == 0.0]
        assert all(n.startswith("readout.heads.") and n.endswith("fc2.bias") for n in zeros), zeros


def test_readout_init_keeps_global_rng(spec, norm):
    model = M.BSModel(spec, norm, C.N_TASKS, C.config_for_scale(2))
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    audit = M.initialize_readout(model, seed=0)
    assert audit["global_rng_unchanged"]
    assert torch.equal(before, torch.get_rng_state())
    assert float(model.readout.fc2.weight.abs().max()) == 0.0
    assert float(model.readout.fc2.bias.abs().max()) == 0.0


def test_build_model_is_deterministic(spec, norm):
    a, _ = M.build_model(2, spec, norm, C.N_TASKS, seed=0, device="cpu")
    b, _ = M.build_model(2, spec, norm, C.N_TASKS, seed=0, device="cpu")
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key]), key
    c, _ = M.build_model(2, spec, norm, C.N_TASKS, seed=1, device="cpu")
    assert not torch.equal(a.state_dict()["atom_encoder.lins.0.weight"], c.state_dict()["atom_encoder.lins.0.weight"])


def test_eval_cadence_rule():
    assert C.calc_eval_epochs(1) == list(range(1, 41))
    assert C.calc_eval_epochs(2) == list(range(2, 41, 2))
    original = dict(C.EVAL_CADENCE)
    try:
        with pytest.raises(KeyError):
            C.set_eval_cadence(3, source="test")
        assert C.set_eval_cadence(2, source="test")["every"] == 2
    finally:
        C.EVAL_CADENCE.clear()
        C.EVAL_CADENCE.update(original)
