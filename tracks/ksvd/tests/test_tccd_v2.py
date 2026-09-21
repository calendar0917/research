"""Targeted data-free tests for TCCD-v2 prototype bottleneck."""

from __future__ import annotations

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


def _records():
    layout = T.PatchLayout(capacity=8, n_atom=5, n_bond=3)
    ai = {c: i for i, c in enumerate(range(5))}
    bi = {c: i for i, c in enumerate(range(3))}
    out = []
    for seed in range(4):
        rec = T.build_mol_record(synthetic_zinc_like(seed, 10 + seed), layout, ai, bi)
        rec["y"] = float(seed)
        out.append(rec)
    return out, layout


def test_assignment_is_simplex_and_temperature_is_bounded():
    torch = T._torch()
    records, layout = _records()
    init = np.random.default_rng(0).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init, mode="rel")
    b = V.make_batch(records, [0, 1], "cpu")
    with torch.no_grad():
        _, C, _ = model.forward_padded(b)
        C = C[b["valid"].reshape(-1)]
    assert float(C.min()) >= -1e-7
    assert float((C.sum(1) - 1).abs().max()) <= 1e-6
    assert V.TEMP_MIN < float(model.temperature()) < 1.0


def test_task_gradient_reaches_encoder_prototypes_and_temperature():
    torch = T._torch()
    records, layout = _records()
    init = np.random.default_rng(1).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init, mode="rel")
    b = V.make_batch(records, [0, 1, 2], "cpu")
    pred, _, _ = model.forward_padded(b)
    (pred - b["y"]).abs().mean().backward()
    assert model.W.grad is not None and float(model.W.grad.abs().sum()) > 0
    assert model.P.grad is not None and float(model.P.grad.abs().sum()) > 0
    assert model.temp_logit.grad is not None and float(model.temp_logit.grad.abs().sum()) > 0


def test_joint_permutation_invariance_and_assignment_shuffle_sensitivity():
    torch = T._torch()
    records, layout = _records()
    init = np.random.default_rng(2).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init, mode="rel")
    b = V.make_batch(records, [0], "cpu")
    with torch.no_grad():
        pred, C, h = model.forward_padded(b)
        n = int(records[0]["n"])
        perm = np.random.default_rng(5).permutation(n)
        p = torch.as_tensor(perm)
        bp = dict(b)
        Xp = b["X_pad"][:, :n].index_select(1, p)
        Rp = b["R_pad"][:, :, :n, :n].index_select(2, p).index_select(3, p)
        bp["X_pad"] = torch.nn.functional.pad(Xp, (0, 0, 0, b["X_pad"].shape[1] - n))
        bp["R_pad"] = torch.zeros_like(b["R_pad"])
        bp["R_pad"][:, :, :n, :n] = Rp
        pred_p, _, h_p = model.forward_padded(bp)
        sh = V.shuffle_assignments(C.reshape(1, -1, V.K_PROTO), records, [0], 7, "cpu")
        h_sh = V.compose_padded(sh, b["R_pad"], b["valid"], b["iu0"], b["iu1"])
    assert float((pred - pred_p).abs().max()) <= 1e-5
    assert float((h - h_p).abs().max()) <= 1e-5
    assert float((h - h_sh).abs().max()) > 1e-6


def test_prototype_reader_has_no_latent_bypass():
    records, layout = _records()
    init = np.random.default_rng(3).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init, mode="rel")
    assert model.uses_latent_bypass is False
    assert model.graph_feature_kind == "assignment_composition_only"


def test_regularizer_initial_contributions_are_five_percent():
    records, layout = _records()
    init = np.random.default_rng(4).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init, mode="rel")
    reg = V._initial_regularization(model, records, [0, 1, 2], "cpu", batch=2, log=lambda *a: None)
    assert abs(reg["initial_local_contribution"] / reg["initial_task"] - 0.05) <= 1e-6
    assert abs(reg["initial_balance_contribution"] / reg["initial_task"] - 0.05) <= 1e-6
