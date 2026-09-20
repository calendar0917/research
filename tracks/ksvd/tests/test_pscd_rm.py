"""Data-free correctness tests for PSCD-RM-v0.

These lock the invariants the rate-matched dictionary audit depends on and need
no ZINC data, no GPU and no results artifacts:

* the corpus-level rate statistics (R_occ, R_state) equal a hand-computed
  occurrence/port count on a tiny synthetic code set;
* the frequency prefix rate curve is monotone in compression and equals a fresh
  ``tokenize_frozen`` at every prefix;
* ``truncate_vocabulary`` keeps the mid layout of the frozen frequency
  dictionary restricted to a prefix (same singletons, same first-k motifs);
* the rate-constrained discovery completes under a loose target and records a
  hard **infeasibility** (no relaxation) under an impossible target;
* admissible candidates satisfy the exact hard rate bound they were selected
  under (g_occ >= D_occ / m).
"""

from __future__ import annotations

import numpy as np
import pytest

from tracks.ksvd.code import run_pscd_rm_dictionary as rm
from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (
    Mol,
    StructuredCode,
    Vocabulary,
    apply_rule,
    atom_to_token,
    enumerate_candidates,
    graph_bonds,
    init_tokens,
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
        ring = i % 3 + 3
        mols.append(_mol(i % 3 + 3, [(j, (j + 1) % ring) for j in range(ring)],
                         [j % 2 for j in range(ring)], [1] * ring))
    return mols


def test_corpus_rates_hand_computed():
    # 3 singletons all bonded: 3 occurrences, 3 port pairs (one per occurrence).
    code = StructuredCode(n=3, m=2, occ_motif_ids=[0, 0, 0],
                          comp_edges=[(0, 0, 1, 1, 0), (1, 0, 1, 2, 0)])
    st = rm.corpus_rates([code])
    assert st["N"] == 3 and st["M"] == 3 and st["P"] == 3
    assert st["R_occ"] == pytest.approx(1.0)
    assert st["R_state"] == pytest.approx(2.0)


def _learn_short_frequency(vocab, mols, bonds, ranks, steps=3):
    """Build a short frequency trajectory AND register its motifs in ``vocab``."""
    tokens = [init_tokens(vocab, m) for m in mols]
    sequence = []
    for _ in range(steps):
        tally: dict[bytes, int] = {}
        example: dict[bytes, tuple] = {}
        caches = []
        for gi, m in enumerate(mols):
            idx = atom_to_token(tokens[gi])
            cands = enumerate_candidates(tokens[gi], bonds[gi], m.node_types, idx)
            caches.append(cands)
            for (_ta, _tb, atoms, key, cn, cb) in cands:
                tally[key] = tally.get(key, 0) + 1
                example.setdefault(key, (len(atoms), cn, cb))
        if not tally:
            break
        key = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        size, cn, cb = example[key]
        vocab.add(key, size, cn, cb, singleton=False)
        sequence.append(key)
        for gi in range(len(mols)):
            tokens[gi], _ = apply_rule(tokens[gi], caches[gi], key, ranks[gi])
    return sequence


def test_frequency_curve_matches_tokenize_frozen():
    mols = _corpus(12)
    bonds = [graph_bonds(m) for m in mols]
    ranks = [mol_rank(m) for m in mols]
    vocab = Vocabulary()
    sequence = _learn_short_frequency(vocab, mols, bonds, ranks, steps=3)

    curve = rm.frequency_prefix_rate_curve(vocab, sequence, mols, bonds, ranks,
                                           log=lambda *a: None)
    assert [row["k"] for row in curve] == [0, 1, 2, 3]
    assert curve[0]["R_occ"] == pytest.approx(1.0)
    assert all(curve[i]["R_occ"] >= curve[i + 1]["R_occ"] - 1e-12 for i in range(3))
    for k in range(4):
        fresh = rm.corpus_rates(rm.codes_for(
            vocab, [tokenize_frozen(m, vocab, sequence[:k], rk)
                    for m, rk in zip(mols, ranks)], mols, ranks))
        assert fresh["R_occ"] == pytest.approx(curve[k]["R_occ"])
        assert fresh["R_state"] == pytest.approx(curve[k]["R_state"])


def test_truncate_vocabulary_prefix_layout():
    vocab = Vocabulary()
    vocab.add_singleton(3)
    vocab.add_singleton(6)
    keys = []
    for i in range(4):
        key = b"m%02d" % i
        keys.append(key)
        vocab.add(key, 2, (i % 2, 1), ((0, 1, 1),), singleton=False)
    sub = rm.truncate_vocabulary(vocab, keys, 2)
    assert [mt.singleton for mt in sub.by_id.values()] == [True, True, False, False]
    assert [mt.key for mt in sub.by_id.values()][2:] == keys[:2]
    assert len(sub) == 4


def test_invariance_check_zero_mismatch():
    mols = _corpus(12)
    bonds = [graph_bonds(m) for m in mols]
    ranks = [mol_rank(m) for m in mols]
    vocab = Vocabulary()
    sequence = _learn_short_frequency(vocab, mols, bonds, ranks, steps=3)
    res = rm.invariance_check(vocab, sequence, mols, ranks, n_graphs=6, n_relabels=3)
    assert res["checked"] == 18
    assert res["mismatch"] == 0


def test_discover_rate_completes_under_loose_target(monkeypatch):
    mols = _corpus(12)
    y = np.linspace(0.0, 1.0, len(mols))
    monkeypatch.setattr(rm, "MIN_GRAPH_SUPPORT", 1)
    N = sum(m.n for m in mols)
    merges = 5

    def fake(key, size, cn, cb, **kw):
        return -1.0, merges, 0

    monkeypatch.setattr(rm, "candidate_mdl_and_rates", fake)
    out = rm.discover_rate(mols, [graph_bonds(m) for m in mols],
                           [mol_rank(m) for m in mols],
                           mols, [graph_bonds(m) for m in mols],
                           [mol_rank(m) for m in mols], y, y,
                           r_state_target=2.0, r_occ_target=1.0,
                           steps=2, log=lambda *a: None)
    assert out["completed"] is True
    assert len(out["sequence"]) == 2
    for r in out["rounds"]:
        assert r["g_occ"] == pytest.approx(merges / N)
        assert r["g_state"] == pytest.approx(merges / N)
        assert r["delta_mdl"] == pytest.approx(-1.0)
        assert np.isfinite(r["task_score"])


def test_discover_rate_infeasible_without_relaxation(monkeypatch):
    mols = _corpus(12)
    y = np.linspace(0.0, 1.0, len(mols))
    monkeypatch.setattr(rm, "MIN_GRAPH_SUPPORT", 1)

    def fake(key, size, cn, cb, **kw):
        return -1.0, 5, 0  # tiny compression progress

    monkeypatch.setattr(rm, "candidate_mdl_and_rates", fake)
    out = rm.discover_rate(mols, [graph_bonds(m) for m in mols],
                           [mol_rank(m) for m in mols],
                           mols, [graph_bonds(m) for m in mols],
                           [mol_rank(m) for m in mols], y, y,
                           r_state_target=2.0, r_occ_target=0.0,  # R_occ debt = 1.0
                           steps=2, log=lambda *a: None)
    assert out["completed"] is False
    assert out["sequence"] == []
    info = out["infeasible"]
    assert info["round"] == 1
    assert info["reason"] == "rate_constraint_infeasible"
    assert info["need_occ"] == pytest.approx(0.5)
    assert info["n_mdl_ok"] >= 1 and info["n_rate_ok"] == 0
