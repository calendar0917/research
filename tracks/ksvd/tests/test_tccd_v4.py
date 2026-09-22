"""Targeted correctness tests for TCCD-v4 higher-order assembly."""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v4 as V
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


def _record(seed: int = 0):
    layout = T.PatchLayout(capacity=8, n_atom=5, n_bond=3)
    ai = {c: i for i, c in enumerate(range(5))}
    bi = {c: i for i, c in enumerate(range(3))}
    rec = T.build_mol_record(synthetic_zinc_like(seed, 10 + seed), layout, ai, bi)
    rec["X"] = np.random.default_rng(100 + seed).standard_normal(
        (int(rec["n"]), layout.feature_dim)
    ).astype(np.float32)
    rec["y"] = float(seed) / 3.0
    return rec, layout


def test_gate0_checks_pass():
    checks = V.gate0_checks()
    assert checks["official_test_loaded"] is False
    assert checks["target_not_read_for_r2"] is True
    assert checks["exact_s2"] is True
    assert checks["permutation_invariant"] is True
    assert checks["batching_invariant"] is True
    assert checks["twohop_sensitivity"] is True
    assert checks["mismatch_effective"] is True
    assert checks["official_test_blocked"] is True
    assert checks["all_pass"] is True


def test_safe_normalized_adjacency_and_full_diagonal_r2():
    rec = {"Rb": np.zeros((1, 3, 3), dtype=np.float32), "n": 3}
    assert np.array_equal(V.native_untyped_adjacency(rec), np.zeros((3, 3), dtype=np.float32))
    assert np.array_equal(V.walk_operator(rec, 2), np.zeros((3, 3), dtype=np.float32))

    rec2 = {"Rb": np.zeros((1, 3, 3), dtype=np.float32), "n": 3}
    rec2["Rb"][0, 0, 1] = rec2["Rb"][0, 1, 0] = 1.0
    rec2["Rb"][0, 1, 2] = rec2["Rb"][0, 2, 1] = 1.0
    r2 = V.walk_operator(rec2, 2)
    assert float(r2[0, 0]) > 0.0
    assert float(r2[1, 1]) > 0.0
    assert float(r2[2, 2]) > 0.0


def test_model_adds_exact_twohop_block_and_backpropagates():
    records = [_record(i)[0] for i in range(3)]
    layout = _record(0)[1]
    init = np.random.default_rng(7).standard_normal((layout.feature_dim, V.D_LOCAL)).astype(np.float32)
    model = V.PrototypeAssemblyModelFactory.build(
        layout.feature_dim,
        V.D_LOCAL,
        V.K_PROTO,
        V.N_REL,
        init,
        max_order=2,
        seed=0,
    )
    batch = V.make_batch(records, [0, 1], "cpu", max_order=2)
    pred, C, h = model.forward_padded(batch)
    assert h.shape[1] == V.model_feature_dim(2)
    assert h.shape[1] == V.T.h_dim(V.K_PROTO, V.N_REL) + V.K_PROTO * (V.K_PROTO + 1) // 2
    (pred - batch["y"]).abs().mean().backward()
    assert model.W.grad is not None and float(model.W.grad.abs().sum()) > 0.0
    assert model.P.grad is not None and float(model.P.grad.abs().sum()) > 0.0
    assert model.temp_logit.grad is not None and float(model.temp_logit.grad.abs().sum()) > 0.0
    assert model.head.weight.grad is not None and float(model.head.weight.grad.abs().sum()) > 0.0
    assert model.uses_latent_bypass is False


def test_fixed_permutation_is_deterministic_and_label_free():
    p0 = V.fixed_permutation(7, 12)
    p1 = V.fixed_permutation(7, 12)
    p2 = V.fixed_permutation(7, 13)
    assert np.array_equal(p0, p1)
    assert sorted(p0.tolist()) == list(range(7))
    assert not np.array_equal(p0, p2)
    assert np.array_equal(V.fixed_permutation(0, 3), np.arange(0, dtype=np.int64))
    assert np.array_equal(V.fixed_permutation(1, 3), np.arange(1, dtype=np.int64))


def test_direct_permutation_invariance_of_torch_moment():
    rng = np.random.default_rng(9)
    c = torch.as_tensor(rng.normal(size=(5, V.K_PROTO)).astype(np.float32))
    rec = {"Rb": np.zeros((1, 5, 5), dtype=np.float32), "n": 5}
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        rec["Rb"][0, a, b] = rec["Rb"][0, b, a] = 1.0
    r2 = V.walk_operator(rec, 2)
    p = torch.as_tensor([2, 0, 4, 1, 3])
    rp = torch.as_tensor(r2).index_select(0, p).index_select(1, p)
    iu0, iu1 = [torch.as_tensor(x, dtype=torch.long) for x in T.sym_indices(V.K_PROTO)]
    valid = torch.ones((1, 5), dtype=torch.bool)
    a = V.walk_moment_torch(c.unsqueeze(0), torch.as_tensor(r2).unsqueeze(0), valid, iu0, iu1)
    b = V.walk_moment_torch(c.index_select(0, p).unsqueeze(0), rp.unsqueeze(0), valid, iu0, iu1)
    assert float((a - b).abs().max()) <= 1e-5
