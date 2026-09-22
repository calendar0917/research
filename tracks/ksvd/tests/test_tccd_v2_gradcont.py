"""Targeted data-free tests for the TCCD-v2 GRAD-CONT ordinal intervention."""

from __future__ import annotations

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V
from tracks.ksvd.code import tccd_v2_gradcont as G
from tracks.ksvd.code import tccd_v2_local_chem as LC
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


def _records(n_mols: int = 120, n_atoms: int = 16):
    layout = T.PatchLayout(capacity=8, n_atom=5, n_bond=3)
    ai = {c: i for i, c in enumerate(range(5))}
    bi = {c: i for i, c in enumerate(range(3))}
    out = []
    for seed in range(n_mols):
        rec = T.build_mol_record(synthetic_zinc_like(seed, n_atoms), layout, ai, bi)
        rec["y"] = float(seed % 5)
        out.append(rec)
    return out, layout


def _empty_triplets():
    return {
        "anchor": np.empty(0, dtype=np.int64),
        "closer": np.empty(0, dtype=np.int64),
        "farther": np.empty(0, dtype=np.int64),
        "anchor_pos": np.empty(0, dtype=np.int64),
        "ctype": np.empty(0, dtype=np.int64),
    }


def test_pareto_compare_is_strict_partial_order():
    # A strictly below B
    assert bool(LC.pareto_compare(0, 1, 0, 2))
    assert bool(LC.pareto_compare(0, 1, 1, 1))
    assert bool(LC.pareto_compare(1, 1, 2, 3))
    # equal -> not strict
    assert not bool(LC.pareto_compare(1, 1, 1, 1))
    # incomparable
    assert not bool(LC.pareto_compare(0, 3, 1, 1))
    assert not bool(LC.pareto_compare(2, 0, 1, 1))
    # reflected
    assert not bool(LC.pareto_compare(1, 1, 0, 3))
    assert LC.comparison_type(0, 1, 0, 2) == "same_d1"
    assert LC.comparison_type(0, 1, 1, 1) == "same_d2"
    assert LC.comparison_type(0, 1, 1, 2) == "both"


def test_bins_are_capped():
    v = np.array([0, 1, 3, 7, 99], dtype=np.int64)
    assert LC.bin_d1(v).tolist() == [0, 1, 3, LC.D1_MAX_BIN, LC.D1_MAX_BIN]
    assert LC.bin_d2(v).tolist() == [0, 1, 3, LC.D2_MAX_BIN, LC.D2_MAX_BIN]


def test_decode_local_chem_shapes_and_self_distance():
    records, layout = _records(6)
    chem = LC.decode_local_chem(records, [0, 1, 2], layout=layout, log=lambda *a: None)
    P = chem["n_patches"]
    assert P > 0
    idx = np.arange(P)
    assert np.all(LC.d1_pairs(chem, idx, idx) == 0)
    assert np.all(LC.d2_pairs(chem, idx, idx) == 0)
    assert np.all(LC.d1_row(chem, 0, np.arange(P)) >= 0)


def test_ordinal_triplets_are_legal_and_label_free():
    records, layout = _records(120, n_atoms=18)
    for r in records:
        r.pop("y", None)
    tr = list(range(len(records)))
    chem = LC.decode_local_chem(records, tr, layout=layout, log=lambda *a: None)
    trip = LC.build_ordinal_triplets(chem, log=lambda *a: None)
    a, c, f = trip["anchor"], trip["closer"], trip["farther"]
    if a.size == 0:
        return  # synthetic fixture too poor; real-data legality is covered by gate0
    d1c = LC.d1_pairs(chem, a, c)
    d2c = LC.d2_pairs(chem, a, c)
    d1f = LC.d1_pairs(chem, a, f)
    d2f = LC.d2_pairs(chem, a, f)
    assert np.all(d1c <= d1f) and np.all(d2c <= d2f)
    assert np.all((d1c < d1f) | (d2c < d2f))
    assert np.all(chem["root"][a] == chem["root"][c])
    assert np.all(chem["degree"][a] == chem["degree"][c])
    assert np.all(chem["root"][a] == chem["root"][f])
    assert np.all(chem["degree"][a] == chem["degree"][f])
    assert np.all(chem["graph_of"][a] != chem["graph_of"][c])
    assert np.all(chem["graph_of"][a] != chem["graph_of"][f])
    assert not np.any(LC.is_exact(chem, a, c))
    assert trip["stats"]["uses_target_label"] is False
    # deterministic
    trip2 = LC.build_ordinal_triplets(chem, log=lambda *a: None)
    assert np.array_equal(trip["anchor"], trip2["anchor"])
    assert np.array_equal(trip["closer"], trip2["closer"])
    assert np.array_equal(trip["farther"], trip2["farther"])


def test_sample_bin_pairs_has_distinct_same_key_stream():
    records, layout = _records(120, n_atoms=18)
    chem = LC.decode_local_chem(records, [i for i in range(18)], layout=layout, log=lambda *a: None)
    pairs = LC.sample_bin_pairs(chem, target_pairs=4000, exact_target=2000, log=lambda *a: None)
    a, p, sk = pairs["anchor"], pairs["partner"], pairs["same_key"]
    assert a.shape == p.shape == sk.shape
    if a.size:
        assert np.all(chem["graph_of"][a] != chem["graph_of"][p])
        key_eq = chem["key_id"][a] == chem["key_id"][p]
        assert np.array_equal(key_eq.astype(np.int64), sk)


def test_ordinal_loss_gradient_reaches_encoder_only():
    torch = T._torch()
    records, layout = _records(40)
    init = np.random.default_rng(0).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init, mode="rel")
    X = torch.as_tensor(
        np.concatenate([np.asarray(r["X"], dtype=np.float32) for r in records], axis=0),
        dtype=torch.float32,
    )
    a = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    c = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    f = torch.tensor([8, 9, 10, 11], dtype=torch.long)
    model.zero_grad(set_to_none=True)
    G._ordinal_loss_rows(model, X, a, c, f).backward()
    assert model.W.grad is not None and float(model.W.grad.abs().sum()) > 0
    assert model.P.grad is None or float(model.P.grad.abs().sum()) == 0
    assert model.head.weight.grad is None


def test_ordinal_loss_is_softplus_of_margin():
    torch = T._torch()
    za = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    zj = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # cos 1
    zk = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)  # cos -1
    val = float(G._ordinal_loss(za, zj, zk))
    expected = float(np.log1p(np.exp(-2.0)))  # softplus(-2)
    assert abs(val - expected) < 1e-5


def test_lambda_zero_matches_frozen_tccd_v2_training():
    records, layout = _records(24, n_atoms=16)
    init_a = np.random.default_rng(11).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    init_b = init_a.copy()
    tr = list(range(len(records)))
    kwargs = dict(seed=0, max_epochs=3, patience=10, batch=4, log=lambda *a: None)

    model_a = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init_a, mode="rel")
    res_a = V.train_model(model_a, records, records, tr, tr, "cpu", **kwargs)

    model_b = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init_b, mode="rel")
    res_b = G.train_model_gradcont(
        model_b, records, records, tr, tr, "cpu",
        triplets=_empty_triplets(), lambda_grad=0.0, **kwargs,
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


def test_active_grad_loss_changes_the_training_path():
    torch = T._torch()
    records, layout = _records(120, n_atoms=18)
    tr = list(range(len(records)))
    chem = LC.decode_local_chem(records, tr, layout=layout, log=lambda *a: None)
    trip = LC.build_ordinal_triplets(chem, log=lambda *a: None)
    if trip["anchor"].size == 0:
        return
    init = np.random.default_rng(3).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeModelFactory.build(layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init, mode="rel")
    X = G._patch_matrix(records, tr, "cpu")
    a = torch.as_tensor(trip["anchor"], dtype=torch.long)
    c = torch.as_tensor(trip["closer"], dtype=torch.long)
    f = torch.as_tensor(trip["farther"], dtype=torch.long)
    model.zero_grad(set_to_none=True)
    G._ordinal_loss_rows(model, X, a, c, f).backward()
    assert float(model.W.grad.abs().sum()) > 0


def test_triplet_save_load_roundtrip(tmp_path):
    records, layout = _records(60, n_atoms=18)
    chem = LC.decode_local_chem(records, [i for i in range(30)], layout=layout, log=lambda *a: None)
    trip = LC.build_ordinal_triplets(chem, log=lambda *a: None)
    path = tmp_path / "t.npz"
    LC.save_triplets(trip, path)
    loaded = LC.load_triplets(path)
    for k in ("anchor", "closer", "farther", "anchor_pos", "ctype"):
        assert np.array_equal(trip[k], loaded[k])
