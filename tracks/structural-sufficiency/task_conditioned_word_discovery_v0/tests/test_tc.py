"""Fast CPU unit tests for task_conditioned_word_discovery_v0."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "code"))

import tc_common as C  # noqa: E402
import tc_cpu as P  # noqa: E402
import tc_neural as N  # noqa: E402
from scipy import sparse  # noqa: E402


def test_signature_key_is_stable():
    sig = [1, 2, 3, -1, 0, 0, 0, -1, -1, -1]
    a = C.signature_key("FULL", 3, sig)
    b = C.signature_key("FULL", 3, sig)
    assert a == b
    assert a.startswith("FULL:3:")
    assert C.signature_key("COARSE", 3, sig) != a


def test_ridge_matches_closed_form_on_standardised_features():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    z, mu, sd = P._standardize_train(x)
    beta = P._ridge_beta(z, y.reshape(-1, 1), 10.0).ravel()
    pred = z @ beta + (y.mean() - z.mean(axis=0) @ beta)
    # explicit normal equations with intercept
    zc = z - z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    beta2 = np.linalg.solve(zc.T @ zc + 10.0 * np.eye(6), zc.T @ yc)
    np.testing.assert_allclose(beta, beta2, atol=1e-10)
    np.testing.assert_allclose(pred, zc @ beta2 + y.mean(), atol=1e-10)


def test_scaffold_folds_are_disjoint_and_cover_fit():
    rows = np.arange(60)
    # every row its own scaffold, plus groups of two
    scaf = {str(i): (f"s{i // 2}" if i < 20 else f"t{i}") for i in rows.tolist()}
    # monkeypatch the frozen map via a temporary config-free function
    old = C.scaffold_map
    C.scaffold_map = lambda split: scaf
    try:
        fold, info = P.scaffold_folds("D0", rows, seed=7)
    finally:
        C.scaffold_map = old
    assert fold.size == rows.size
    assert set(np.unique(fold)) == {0, 1, 2}
    assert sum(info["fold_molecule_counts"]) == rows.size
    for a in range(3):
        for b in range(a + 1, 3):
            assert not (set(rows[fold == a].tolist()) & set(rows[fold == b].tolist()))


def test_frequency_buckets_and_matching():
    n = 24
    df = np.arange(n, dtype=np.int64) + 20
    keys = [f"K{i:03d}" for i in range(n)]
    buckets, info = P.df_quartile_buckets(df, keys)
    assert sorted(info["buckets"][b]["n"] for b in range(4)) == sorted([6, 6, 6, 6])
    # task picks two per bucket, size 3 FULL
    meta = []
    for i in range(n):
        meta.append({"size": 3, "gran": "FULL", "df": int(df[i]), "key": keys[i],
                     "type_id": i, "scaffold_df": 5, "occurrence_fit": 1,
                     "equiv_group": "g", "equiv_members": [i],
                     "equiv_member_keys": [keys[i]]})
    task_cols = []
    for b in range(4):
        idx = [i for i in range(n) if buckets[i] == b][:2]
        task_cols.extend(idx)
    freq_cols, detail = P.frequency_vocabulary(meta, buckets, task_cols)
    assert len(freq_cols) == len(task_cols) == 8
    from collections import Counter
    assert Counter((meta[i]["size"], meta[i]["gran"], buckets[i]) for i in task_cols) == \
        Counter((meta[i]["size"], meta[i]["gran"], buckets[i]) for i in freq_cols)


def test_model_arms_identical_params_and_base_ignores_words():
    atom_dims = [7, 3, 4]
    bond_dims = [4, 5, 3]
    dims = {
        "atom": 8, "bond": 4, "pair_hidden": 16, "pair_out": 8,
        "word_hidden": 16, "word_out": 8, "sentence_input": 24,
        "sentence_hidden": 16, "sentence_out": 12, "readout_input": 45,
        "readout_hidden": 16, "readout_mid": 8, "dropout": 0.0,
    }
    K = 5
    models = {a: N.TCWordModel(atom_dims, bond_dims, K, dims) for a in N.ARMS}
    hs = {a: m.init_parameters(3) for a, m in models.items()}
    assert hs["BASE"] == hs["D_TASK"] == hs["D_FREQ"]
    assert len({m.parameter_ledger()["total"] for m in models.values()}) == 1
    batch = {
        "atom_feat": __import__("torch").tensor([[0, 1, 2], [2, 1, 0]], dtype=__import__("torch").long),
        "bond_feat": __import__("torch").tensor([[0, 1, 0]], dtype=__import__("torch").long),
        "edge_src": __import__("torch").tensor([0, 1], dtype=__import__("torch").long),
        "edge_dst": __import__("torch").tensor([1, 0], dtype=__import__("torch").long),
        "edge_bond": __import__("torch").tensor([0, 0], dtype=__import__("torch").long),
        "atom_mol": __import__("torch").tensor([0, 0], dtype=__import__("torch").long),
        "word_counts": __import__("torch").rand(2, K),
        "n_atoms": __import__("torch").tensor([2.0]),
        "graph_scalars": __import__("torch").zeros(1, 8),
        "n_molecules": 1,
    }
    import torch
    models["BASE"].eval()
    models["D_TASK"].eval()
    with torch.no_grad():
        a = models["BASE"](batch, "BASE")
        b = dict(batch)
        b["word_counts"] = torch.rand_like(batch["word_counts"])
        a2 = models["BASE"](b, "BASE")
        assert torch.allclose(a, a2, atol=1e-7)
        t1 = models["D_TASK"](batch, "D_TASK")
        t2 = models["D_TASK"](b, "D_TASK")
        assert not torch.allclose(t1, t2)


def test_select_candidates_shapes_and_zero_variance_handling():
    rng = np.random.default_rng(4)
    n, p = 60, 9
    x0 = rng.normal(size=(n, 4))
    xw = rng.normal(size=(n, p))
    xw[:, 0] = 0.0  # constant column
    y = rng.normal(size=n)
    fold = np.repeat([0, 1, 2], n // 3)
    meta = [{"size": 3, "gran": "FULL", "df": 20, "key": f"K{i}", "type_id": i,
             "scaffold_df": 5, "occurrence_fit": 10, "equiv_group": f"g{i}",
             "equiv_members": [i], "equiv_member_keys": [f"K{i}"]} for i in range(p)]
    out = P.select_candidates(x0, sparse.csc_matrix(xw), y, fold, meta, 10.0, chunk=4)
    assert out["cond"].shape == (3, p)
    assert out["raw"].shape == (3, p)
    assert not out["consistent"][0]
    assert np.isfinite(out["score"]).any()
    assert np.all(np.isnan(out["score"][~out["consistent"]]))