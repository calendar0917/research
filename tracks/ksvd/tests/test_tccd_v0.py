"""Data-free correctness tests for TCCD-v0 (fresh-clone safe, CPU only)."""

from __future__ import annotations

import numpy as np
import pytest

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


def _layout():
    return T.PatchLayout(capacity=8, n_atom=5, n_bond=3)


def _indices():
    return {c: i for i, c in enumerate(range(5))}, {c: i for i, c in enumerate(range(3))}


def test_patch_slot_order_is_permutation_invariant():
    mol = synthetic_zinc_like(3, 11)
    ai, bi = _indices()
    layout = _layout()
    rec = T.build_mol_record(mol, layout, ai, bi)
    rng = np.random.default_rng(0)
    for _ in range(10):
        pm, perm_atom, _ = T.permute_mol_ids(mol, rng)
        rec2 = T.build_mol_record(pm, layout, ai, bi)
        assert np.abs(rec["X"] - rec2["X"][perm_atom]).max() <= 1e-6
        assert np.abs(rec["B"] - rec2["B"][perm_atom][:, perm_atom]).max() <= 1e-6


def test_relation_contraction_joint_permutation_equivariance():
    mol = synthetic_zinc_like(4, 12)
    ai, bi = _indices()
    layout = _layout()
    rec = T.build_mol_record(mol, layout, ai, bi)
    Rint, Rb, Rgeo = T.relation_matrices(rec)
    X = T._torch().as_tensor(rec["X"])
    D = T._torch().as_tensor(
        T.random_normalized_dictionary(layout.feature_dim, T.K_DICT, 0), dtype=T._torch().float32
    )
    C = T.iht_codes(D.double(), X.double())
    iu0, iu1 = T.sym_indices(T.K_DICT)
    h = T.compose_torch(C, T._torch().as_tensor(Rint).double(),
                        [T._torch().as_tensor(r).double() for r in Rb],
                        T._torch().as_tensor(Rgeo).double(), iu0, iu1)
    perm = T._torch().as_tensor(np.random.default_rng(1).permutation(mol.n))
    hp = T.compose_torch(
        C.index_select(0, perm),
        T._torch().as_tensor(Rint).double().index_select(0, perm).index_select(1, perm),
        [T._torch().as_tensor(r).double().index_select(0, perm).index_select(1, perm) for r in Rb],
        T._torch().as_tensor(Rgeo).double().index_select(0, perm).index_select(1, perm),
        iu0, iu1,
    )
    assert float((h - hp).abs().max()) <= 1e-6


def test_relation_sensitivity_to_decoupling():
    mol = synthetic_zinc_like(5, 14)
    ai, bi = _indices()
    layout = _layout()
    rec = T.build_mol_record(mol, layout, ai, bi)
    Rint, Rb, Rgeo = T.relation_matrices(rec)
    X = T._torch().as_tensor(rec["X"])
    D = T._torch().as_tensor(
        T.random_normalized_dictionary(layout.feature_dim, T.K_DICT, 0), dtype=T._torch().float32
    )
    C = T.iht_codes(D, X)
    iu0, iu1 = T.sym_indices(T.K_DICT)
    h = T.compose_torch(C, T._torch().as_tensor(Rint), [T._torch().as_tensor(r) for r in Rb],
                        T._torch().as_tensor(Rgeo), iu0, iu1)
    perm = np.array([1, 0] + list(range(2, mol.n))) if mol.n > 2 else np.array([1, 0])
    hb = T.compose_torch(
        C[T._torch().as_tensor(perm)], T._torch().as_tensor(Rint),
        [T._torch().as_tensor(r) for r in Rb], T._torch().as_tensor(Rgeo), iu0, iu1
    )
    rel = float((h - hb).norm() / (h.norm() + 1e-12))
    assert rel >= 0.05, rel


def test_iht_exact_sparsity_and_gradients():
    torch = T._torch()
    D = torch.as_tensor(T.random_normalized_dictionary(40, 16, 1), dtype=torch.float32).clone()
    D.requires_grad_(True)
    X = torch.randn(7, 40)
    C = T.iht_codes(D, X, s=4, steps=5)
    nnz = (C.abs() > 1e-9).sum(dim=1)
    assert int(nnz.min()) == 4 and int(nnz.max()) == 4
    loss = (((X - C @ D.t()) ** 2).sum(dim=1)).mean()
    g = torch.autograd.grad(loss, D)[0]
    assert float(g.norm()) > 0


def test_hard_threshold_keeps_largest_abs():
    torch = T._torch()
    C = torch.tensor([[3.0, -5.0, 1.0, 4.0, -0.5]])
    out = T.hard_threshold_rows(C, 2)
    assert out[0, 1].item() == -5.0 and out[0, 3].item() == 4.0
    assert (out != 0).sum().item() == 2


def test_omp_ksvd_recovers_planted_sparse_signal():
    rng = np.random.default_rng(0)
    F, K, s, N = 60, 24, 3, 400
    Dtrue = T.normalize_columns(rng.standard_normal((F, K)))
    Ctrue = np.zeros((N, K))
    for i in range(N):
        cols = rng.choice(K, size=s, replace=False)
        Ctrue[i, cols] = rng.standard_normal(s)
    X = (Ctrue @ Dtrue.T).astype(np.float32)
    D0 = T.random_normalized_dictionary(F, K, 7)
    D, _ = T.ksvd_fit(X, D0, s=s, epochs=4, chunk=256, seed=0, log=None)
    err = T.relative_reconstruction_error(D, X)
    err0 = T.relative_reconstruction_error(D0, X)
    assert err < err0 * 0.5


def test_choose_capacity_and_stage0():
    out = T.__dict__  # keep import used
    assert out is not None
    from tracks.ksvd.code.run_tccd_v0 import stage0

    res = stage0(log=lambda *a, **k: None)
    assert res["ok"] is True


def test_coordinate_blocks_are_populated():
    mol = synthetic_zinc_like(6, 10)
    ai, bi = _indices()
    layout = _layout()
    rec = T.build_mol_record(mol, layout, ai, bi)
    sl = layout.block_slices()
    X = rec["X"]
    assert X[:, sl["topology"]].sum() > 0
    assert X[:, sl["bond"]].sum() > 0
    assert X[:, sl["atom"]].sum() > 0
    assert X[:, sl["shell"]].sum() > 0
    assert X[:, sl["mask"]].sum() > 0
    # every patch has its root at slot 0 shell-0 and a positive mask mass
    assert np.all(X[:, sl["mask"]].sum(axis=1) >= 1.0)


def test_relation_geo_matches_shortest_path():
    mol = synthetic_zinc_like(7, 9)
    ai, bi = _indices()
    layout = _layout()
    rec = T.build_mol_record(mol, layout, ai, bi)
    _r, _rb, rgeo = T.relation_matrices(rec)
    adj = T._adjacency(mol)
    for i in range(mol.n):
        d = T.bfs_distances(adj, i, radius=mol.n)
        for j in range(mol.n):
            if j in d:
                assert abs(rgeo[i, j] - np.exp(-d[j] / T.TAU)) < 1e-6
    assert np.allclose(np.diag(rgeo), 1.0)
