"""Data-free correctness tests for TCCD-v1 (fresh-clone safe, CPU only).

These tests do **not** require the frozen TCCD-v0 dictionary artefact; they
exercise the TCCD-v1 composition features, the shared linear reader and the
frozen gate-decision thresholds on synthetic inputs.
"""

from __future__ import annotations

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v1 as V
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


def _layout():
    return T.PatchLayout(capacity=8, n_atom=5, n_bond=3)


def _indices():
    return {c: i for i, c in enumerate(range(5))}, {c: i for i, c in enumerate(range(3))}


def _record(seed=0):
    mol = synthetic_zinc_like(seed, 12)
    ai, bi = _indices()
    return T.build_mol_record(mol, _layout(), ai, bi), mol


def test_shuffle_perm_matches_v0_rng_semantics():
    torch = T._torch()
    for seed in (0, 1):
        device = "cpu"
        perms = T.fixed_shuffle_perms(
            [{"n": 9}, {"n": 1}, {"n": 13}], [0, 1, 2], seed, device
        )
        for gi, p in enumerate(perms):
            n = [9, 1, 13][gi]
            np_perm = V.shuffle_perm(n, seed, gi)
            assert np.array_equal(p.cpu().numpy(), np_perm)
            assert sorted(np_perm.tolist()) == list(range(n))


def test_compose_feature_matches_torch_composition():
    rec, _ = _record(2)
    K = T.K_DICT
    rng = np.random.default_rng(0)
    D = T.normalize_columns(rng.standard_normal((_layout().feature_dim, K)))
    C = T.omp_codes(D, rec["X"])
    iu0, iu1 = T.sym_indices(K)
    ops = V.rel_operator_list(rec)
    h_np = V.compose_feature(C, ops, iu0, iu1)
    torch = T._torch()
    Rint, Rb, Rgeo = T.relation_matrices(rec)
    h_t = T.compose_torch(
        torch.as_tensor(C), torch.as_tensor(Rint), [torch.as_tensor(r) for r in Rb],
        torch.as_tensor(Rgeo), iu0, iu1,
    ).numpy()
    assert h_np.shape == h_t.shape == (K + V.N_REL * len(iu0),)
    assert np.abs(h_np - h_t).max() <= 1e-4


def test_shuffle_preserves_code_multiset_but_destroys_assignment():
    rec, _ = _record(3)
    K = T.K_DICT
    rng = np.random.default_rng(1)
    D = T.normalize_columns(rng.standard_normal((_layout().feature_dim, K)))
    C = T.omp_codes(D, rec["X"])
    iu0, iu1 = T.sym_indices(K)
    ops = V.rel_operator_list(rec)
    perm = V.shuffle_perm(C.shape[0], 0, 7)
    # same multiset of rows
    assert np.abs(np.sort(C, axis=0) - np.sort(C[perm], axis=0)).max() == 0.0
    # same R operators, but assignment destroyed -> features differ
    h_rel = V.compose_feature(C, ops, iu0, iu1)
    h_shuf = V.compose_feature(C, ops, iu0, iu1, perm=perm)
    assert float(np.abs(h_rel - h_shuf).max()) > 1e-6
    # BAG is invariant to the permutation (it is the code sum) -> primary
    # control must act through the relation contractions, not the bag
    assert np.abs(h_rel[:K] - h_shuf[:K]).max() <= 1e-4


def test_gate_thresholds_are_the_preregistered_ones():
    assert V.gate_a_verdict(0.02, 0) == "PASS"
    assert V.gate_a_verdict(0.015, 0) == "PASS"
    assert V.gate_a_verdict(0.004, 0) == "FAIL"
    assert V.gate_a_verdict(0.005, 0) == "AMBIGUOUS_NEEDS_SEED1"
    assert V.gate_a_verdict(0.009, 0) == "AMBIGUOUS_NEEDS_SEED1"
    assert V.gate_a_verdict(0.010, 1, seed0_delta=0.012) == "PASS"   # mean 0.011
    assert V.gate_a_verdict(0.008, 1, seed0_delta=0.008) == "FAIL"   # mean 0.008

    assert V.gate_b_verdict(0.010, 0) == "PASS"
    assert V.gate_b_verdict(-0.001, 0) == "FAIL"
    assert V.gate_b_verdict(0.0, 0) == "FAIL"
    assert V.gate_b_verdict(0.009, 0) == "AMBIGUOUS_NEEDS_SEED1"
    assert V.gate_b_verdict(0.005, 1, seed0_delta=0.007) == "PASS"   # mean 0.006
    assert V.gate_b_verdict(0.003, 1, seed0_delta=0.003) == "FAIL"   # mean 0.003

    assert V.gate_d_band(-0.01) == "VERY_STRONG"
    assert V.gate_d_band(0.0) == "VERY_STRONG"
    assert V.gate_d_band(0.005) == "STRONG"
    assert V.gate_d_band(0.015) == "COMPETITIVE"
    assert V.gate_d_band(0.0151) == "NOT_VIABLE"


def test_gate_c_margin_direction():
    # PASS iff MAE_TASK-D <= MAE_DENSE + 0.005
    assert (0.100 <= 0.096 + V.GATE_C_MARGIN) is True
    assert (0.100 <= 0.094 + V.GATE_C_MARGIN) is False


def test_ridge_reader_is_deterministic_and_learns_a_linear_target():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 24)).astype(np.float32)
    w = rng.standard_normal(24)
    y = (X @ w).reshape(-1) + 0.3
    kw = dict(log=lambda *a, **k: None)
    r1 = V.ridge_reader(X[:480], y[:480], X[480:], y[480:], **kw)
    r2 = V.ridge_reader(X[:480], y[:480], X[480:], y[480:], **kw)
    assert r1.best_valid == r2.best_valid          # no optimizer / RNG
    assert r1.alpha == 480.0                        # alpha = n_train (frozen rule)
    assert r1.best_valid < 0.75 * float(np.std(y))
    assert set(r1.grid_valid) == {"10", "100", "1000", "10000"}


def test_ridge_reader_standardizes_with_train_stats_only():
    # Shifting a feature by a constant on all splits must not change dev MAE
    # (standardization uses train statistics), i.e. the reader is affine-invariant.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 8)).astype(np.float32)
    y = (X @ rng.standard_normal(8)).reshape(-1)
    kw = dict(log=lambda *a, **k: None, grid=())
    a = V.ridge_reader(X[:320], y[:320], X[320:], y[320:], **kw).best_valid
    b = V.ridge_reader((X + 5.0)[:320], y[:320], (X + 5.0)[320:], y[320:], **kw).best_valid
    assert abs(a - b) <= 1e-6


def test_padded_fast_path_matches_slow_graphwise_path():
    torch = T._torch()
    layout = _layout()
    ai, bi = _indices()
    records = []
    for seed in range(5):
        rec = T.build_mol_record(synthetic_zinc_like(seed, 12), layout, ai, bi)
        rec["y"] = float(seed)
        records.append(rec)
    idx = [0, 1, 2, 3]
    slow = T.make_batch(records, idx, "cpu")
    fast = V.make_padded_batch(records, idx, "cpu")
    D = T.random_normalized_dictionary(layout.feature_dim, T.K_DICT, 9).astype(np.float32)
    model = T.TCCDModel.build(layout.feature_dim, T.K_DICT, V.N_REL, D, dense=False).to("cpu")
    with torch.no_grad():
        C = model.encode(slow["X_all"])
        hs = []
        for gi, graph in enumerate(slow["graphs"]):
            hs.append(T.compose_torch(C[graph["slice"]], graph["Rint"], graph["Rb"],
                                      graph["Rgeo"], slow["iu0"], slow["iu1"]))
        pred_slow = model.head(torch.stack(hs)).reshape(-1)
        pred_fast, C_fast, X_fast = V._fast_forward(model, fast)
        rec_slow = ((slow["X_all"] - C @ model.D.t()) ** 2).sum(1) / (
            (slow["X_all"] ** 2).sum(1) + T.EPS
        )
        rec_fast = V._masked_reconstruction_loss(
            X_fast, C_fast, model.D, fast["valid"].reshape(-1)
        )
    assert float((pred_slow - pred_fast).abs().max()) <= 1e-6
    assert abs(float(rec_slow.mean() - rec_fast)) <= 1e-6
