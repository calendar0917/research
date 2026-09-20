"""Data-free correctness tests for PSCD-TMDL-v0.

These lock the invariants the task+MDL dictionary diagnostic depends on and need
no ZINC data, no GPU and no results artifacts:

* the deterministic discovery/monitor split is reproducible, disjoint and
  covers the canonical train set;
* the first-order two-part MDL proxy is finite and its per-merge delta is
  finite and deterministic;
* the task score equals a hand-computed standardised first-order correlation;
* motif-count features from live tokens equal features read back from the final
  structured codes;
* all three discovery arms reconstruct their corpus exactly (100 % decode).
"""

from __future__ import annotations

import numpy as np
import pytest

from tracks.ksvd.code import run_pscd_tmdl_dictionary as tmdl
from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (
    Mol,
    Vocabulary,
    build_structured_code,
    mol_rank,
    tokenize_frozen,
)


def _mol(n, edges, node_types, bond_types):
    edges = [(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges]
    return Mol(n, edges, np.asarray(bond_types, dtype=np.int64),
               np.asarray(node_types, dtype=np.int64))


def _chain(n, offset=0):
    edges = [(i, i + 1) for i in range(n - 1)]
    nt = [(i + offset) % 3 for i in range(n)]
    bt = [1 + ((i + offset) % 3) for i in range(len(edges))]
    return _mol(n, edges, nt, bt)


def _corpus(n=14):
    mols = []
    for i in range(n):
        mols.append(_chain(8, offset=i % 3))
        mols.append(_mol(
            i % 3 + 3, [(j, (j + 1) % (i % 3 + 3)) for j in range(i % 3 + 3)],
            [j % 2 for j in range(i % 3 + 3)], [1] * (i % 3 + 3)))
    return mols


def test_discovery_split_deterministic_disjoint():
    d1, m1 = tmdl.discovery_split(50, 40, seed=0)
    d2, m2 = tmdl.discovery_split(50, 40, seed=0)
    assert d1 == d2 and m1 == m2
    assert set(d1).isdisjoint(m1)
    assert sorted(d1 + m1) == list(range(50))
    assert len(d1) == 40 and len(m1) == 10


def test_mdl_proxy_finite_and_deterministic(monkeypatch):
    mols = _corpus(10)
    y = np.linspace(0.0, 1.0, len(mols))
    monkeypatch.setattr(tmdl, "MIN_GRAPH_SUPPORT", 1)
    monkeypatch.setattr(tmdl, "TASK_ONLY_TOP", 4)
    monkeypatch.setattr(tmdl, "MDL_TOP", 4)
    r1 = tmdl.discover("tmdl", mols, [tmdl.graph_bonds(m) for m in mols],
                       [mol_rank(m) for m in mols],
                       mols, [tmdl.graph_bonds(m) for m in mols],
                       [mol_rank(m) for m in mols], y, y, steps=3, log=lambda *a: None)
    r2 = tmdl.discover("tmdl", mols, [tmdl.graph_bonds(m) for m in mols],
                       [mol_rank(m) for m in mols],
                       mols, [tmdl.graph_bonds(m) for m in mols],
                       [mol_rank(m) for m in mols], y, y, steps=3, log=lambda *a: None)
    assert [r["key"] for r in r1["rounds"]] == [r["key"] for r in r2["rounds"]]
    for r in r1["rounds"]:
        assert np.isfinite(r["delta_mdl"])
        assert np.isfinite(r["task_score"])


def test_task_score_matches_manual(monkeypatch):
    n = 6
    residual = np.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
    key = b"k"
    key_graphs = {key: [0, 0, 3, 5]}
    z = np.zeros(n)
    for gi in key_graphs[key]:
        z[gi] += 1.0
    zt = (z - z.mean()) / (z.std() + tmdl.STD_EPS)
    manual = abs(float(np.mean(residual * zt)))
    got = tmdl.task_score(key, key_graphs, residual, n)
    assert got == pytest.approx(manual, rel=1e-12)


def test_features_tokens_equal_codes():
    mols = _corpus(8)
    from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (
        learn_vocabulary,
    )
    vocab2 = Vocabulary()
    toks, rounds = learn_vocabulary(mols, vocab2, 4, log=lambda *a: None)
    seq = [bytes.fromhex(r["key"]) for r in rounds]
    X_live = tmdl.motif_count_features(toks, vocab2)
    codes = [build_structured_code(m, tokenize_frozen(m, vocab2, seq, mol_rank(m)),
                                   vocab2, mol_rank(m))["code"] for m in mols]
    X_codes = tmdl.features_from_codes(codes, vocab2)
    assert np.array_equal(X_live, X_codes)


def test_all_arms_exact_reconstruction(monkeypatch):
    mols = _corpus(12)
    y = np.linspace(0.0, 1.0, len(mols))
    monkeypatch.setattr(tmdl, "MIN_GRAPH_SUPPORT", 1)
    monkeypatch.setattr(tmdl, "TASK_ONLY_TOP", 4)
    monkeypatch.setattr(tmdl, "MDL_TOP", 4)
    bonds = [tmdl.graph_bonds(m) for m in mols]
    ranks = [mol_rank(m) for m in mols]
    for mode in ("task", "tmdl"):
        out = tmdl.discover(mode, mols, bonds, ranks, mols, bonds, ranks,
                            y, y, steps=3, log=lambda *a: None)
        rec = tmdl.exact_reconstruction(out["vocab"], mols, out["disc_codes"])
        assert rec["frac"] == 1.0
    # frequency arm via the shared BPE learner
    from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (
        learn_vocabulary,
    )
    vocab = Vocabulary()
    _toks, rounds = learn_vocabulary(mols, vocab, 3, log=lambda *a: None)
    seq = [bytes.fromhex(r["key"]) for r in rounds]
    codes = [build_structured_code(m, tokenize_frozen(m, vocab, seq, mol_rank(m)),
                                   vocab, mol_rank(m))["code"] for m in mols]
    assert tmdl.exact_reconstruction(vocab, mols, codes)["frac"] == 1.0
