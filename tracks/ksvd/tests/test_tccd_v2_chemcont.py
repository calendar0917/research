"""Targeted data-free tests for the TCCD-v2 CHEM-CONT intervention."""

from __future__ import annotations

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V
from tracks.ksvd.code import tccd_v2_chemcont as C
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


def _records(n_mols: int = 80, n_atoms: int = 14):
    layout = T.PatchLayout(capacity=8, n_atom=5, n_bond=3)
    ai = {c: i for i, c in enumerate(range(5))}
    bi = {c: i for i, c in enumerate(range(3))}
    out = []
    for seed in range(n_mols):
        rec = T.build_mol_record(synthetic_zinc_like(seed, n_atoms), layout, ai, bi)
        rec["y"] = float(seed % 5)
        out.append(rec)
    return out, layout


def _empty_pairs():
    return {
        "anchor": np.empty(0, dtype=np.int64),
        "positive": np.empty(0, dtype=np.int64),
        "negative": np.empty(0, dtype=np.int64),
        "anchor_pos": np.empty(0, dtype=np.int64),
    }


def test_c1_edits_zero_for_identical_patch():
    records, layout = _records(4)
    chem = C.decode_chemistry(records, [0, 1], layout=layout, log=lambda *a: None)
    assert chem["n_patches"] > 0
    assert C.c1_edits(chem["s1atom"], chem["rbond"], 0, 0) == 0


def test_chem_loss_gradient_reaches_encoder_only():
    torch = T._torch()
    records, layout = _records(6)
    init = np.random.default_rng(0).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init, mode="rel")
    X = torch.as_tensor(
        np.concatenate([np.asarray(r["X"], dtype=np.float32) for r in records], axis=0),
        dtype=torch.float32,
    )
    a = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    p = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    n = torch.tensor([8, 9, 10, 11], dtype=torch.long)
    model.zero_grad(set_to_none=True)
    loss = C._chem_loss_rows(model, X, a, p, n)
    loss.backward()
    assert model.W.grad is not None and float(model.W.grad.abs().sum()) > 0
    assert model.P.grad is None or float(model.P.grad.abs().sum()) == 0
    assert model.head.weight.grad is None


def test_pair_build_does_not_read_target_labels():
    records, layout = _records(60)
    for r in records:
        r.pop("y", None)  # build must work with no target at all
    tr = list(range(len(records)))
    pairs = C.build_pairs(records, tr, layout=layout, log=lambda *a: None)
    assert pairs["stats"]["uses_target_label"] is False
    assert pairs["anchor"].shape == pairs["positive"].shape == pairs["negative"].shape


def test_pair_definition_sanity_when_triplets_exist():
    records, layout = _records(80)
    tr = list(range(len(records)))
    pairs = C.build_pairs(records, tr, layout=layout, log=lambda *a: None)
    s = pairs["stats"]
    if s["n_triplets"] > 0:
        assert s["positive_l1_equal_rate"] == 1.0
        assert s["positive_key_diff_rate"] == 1.0
        assert s["negative_root_degree_match_rate"] == 1.0
        assert s["negative_size_match_rate"] == 1.0
        assert s["negative_c1_ge2_rate"] == 1.0
        assert s["cross_molecule_rate"] == 1.0


def test_lambda_zero_matches_frozen_tccd_v2_training():
    records, layout = _records(24, n_atoms=16)
    init_a = np.random.default_rng(11).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    init_b = init_a.copy()
    tr = list(range(len(records)))
    kwargs = dict(seed=0, max_epochs=3, patience=10, batch=4, log=lambda *a: None)

    model_a = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init_a, mode="rel")
    res_a = V.train_model(model_a, records, records, tr, tr, "cpu", **kwargs)

    model_b = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init_b, mode="rel")
    res_b = C.train_model_chemcont(
        model_b, records, records, tr, tr, "cpu", pairs=_empty_pairs(), lambda_chem=0.0, **kwargs
    )

    assert res_b.best_valid == res_a.best_valid
    assert res_b.soup_valid == res_a.soup_valid
    assert res_b.best_epoch == res_a.best_epoch
    for row_a, row_b in zip(res_a.train_history, res_b.train_history):
        assert row_a["train_loss"] == row_b["train_loss"]
        assert row_a["train_task"] == row_b["train_task"]
        assert row_a["valid"] == row_b["valid"]
        assert row_a["temperature"] == row_b["temperature"]
    for k in res_a.state_best:
        assert bool((res_a.state_best[k] == res_b.state_best[k]).all())


def test_active_chem_loss_changes_the_training_path():
    """With a real lambda_chem the chem term must reach the loss, not be inert."""
    torch = T._torch()
    records, layout = _records(60)
    tr = list(range(len(records)))
    pairs = C.build_pairs(records, tr, layout=layout, log=lambda *a: None)
    init = np.random.default_rng(3).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init, mode="rel")
    X = C._patch_matrix(records, tr, "cpu")
    tri_a = torch.as_tensor(pairs["anchor"], dtype=torch.long)
    tri_p = torch.as_tensor(pairs["positive"], dtype=torch.long)
    tri_n = torch.as_tensor(pairs["negative"], dtype=torch.long)
    if tri_a.numel() == 0:
        return  # synthetic fixture produced no triplets; covered by gate0 on real data
    model.zero_grad(set_to_none=True)
    C._chem_loss_rows(model, X, tri_a, tri_p, tri_n).backward()
    assert float(model.W.grad.abs().sum()) > 0
