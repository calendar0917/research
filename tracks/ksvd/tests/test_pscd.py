"""Data-free correctness tests for the PSCD-v0 compositional-dictionary audit.

These lock the invariants the audit depends on and need no ZINC data, no
checkpoints and no results artifacts:

* graph-BPE merge learning is deterministic and the frozen tokenizer is
  invariant to atom/bond relabelling;
* ``Decode(D, C_G)`` reconstructs the molecule exactly (exact colored canonical
  key) and the full PSCD code is permutation invariant;
* the collision ladder behaves as designed (bag / no-port / orbit-port can
  collide where the exact-port full code does not);
* the MDL proxy returns finite, positive description lengths.
"""

from __future__ import annotations

import numpy as np

from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (
    Mol,
    StructuredCode,
    Vocabulary,
    build_structured_code,
    decode,
    ladder_keys,
    learn_vocabulary,
    mdl_proxy,
    mol_iso_key,
    mol_rank,
    tokenize_frozen,
)
from tracks.ksvd.code.run_aiom_representation_audit import permute_mol


def _mol(n, edges, node_types, bond_types):
    edges = [(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges]
    return Mol(n, edges, np.asarray(bond_types, dtype=np.int64),
               np.asarray(node_types, dtype=np.int64))


def _ring6(sub=()):
    edges = [(i, (i + 1) % 6) for i in range(6)]
    nt = [0] * 6
    bt = [1] * 6
    for s in sub:
        edges.append((s, len(nt)))
        nt.append(1)
        bt.append(1)
    return _mol(len(nt), edges, nt, bt)


def _chain(n):
    edges = [(i, i + 1) for i in range(n - 1)]
    return _mol(n, edges, [i % 3 for i in range(n)], [1 + (i % 3) for i in range(len(edges))])


SYNTH = [_ring6(), _ring6(sub=(0, 1)), _ring6(sub=(0, 3)), _chain(8), _chain(6)]


def test_bpe_learning_deterministic():
    v1 = Vocabulary()
    t1, r1 = learn_vocabulary(SYNTH, v1, 6, log=lambda *a: None)
    v2 = Vocabulary()
    t2, r2 = learn_vocabulary(SYNTH, v2, 6, log=lambda *a: None)
    assert [x["key"] for x in r1] == [x["key"] for x in r2]
    assert [tuple(t.atoms) for t in t1[0]] == [tuple(t.atoms) for t in t2[0]]


def test_tokenizer_and_decode_permutation_invariance():
    vocab = Vocabulary()
    tokens, rounds = learn_vocabulary(SYNTH, vocab, 8, log=lambda *a: None)
    seq = [bytes.fromhex(r["key"]) for r in rounds]
    rng = np.random.default_rng(0)
    for mol in SYNTH:
        rank = mol_rank(mol)
        tok = tokenize_frozen(mol, vocab, seq, rank)
        code = build_structured_code(mol, tok, vocab, rank)["code"]
        assert mol_iso_key(decode(code, vocab)) == mol_iso_key(mol)
        base_d = ladder_keys(code, vocab)["D"]
        for _ in range(4):
            pa = rng.permutation(mol.n)
            pb = rng.permutation(mol.m)
            pm = permute_mol(mol, pa, pb)
            prank = mol_rank(pm)
            ptok = tokenize_frozen(pm, vocab, seq, prank)
            pcode = build_structured_code(pm, ptok, vocab, prank)["code"]
            assert ladder_keys(pcode, vocab)["D"] == base_d
            assert mol_iso_key(decode(pcode, vocab)) == mol_iso_key(pm)


def _ring_motif():
    nt = (0,) * 6
    bonds = ((0, 1, 1), (0, 5, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1))
    return nt, bonds


def test_collision_ladder_bag_noport_orbit_vs_full():
    """Two attachments adjacent vs para on a symmetric ring.

    Bag, no-port composition and orbit-port codes all collide; the full
    exact-port PSCD code must separate them.
    """
    vocab = Vocabulary()
    nt, bonds = _ring_motif()
    ring = vocab.add(b"ring6", 6, nt, bonds, False)
    nit = vocab.add(b"n-single", 1, (1,), (), True)
    # occurrence 0 = ring, 1,2 = nitrogen singletons
    ortho = StructuredCode(8, 7, [ring.mid, nit.mid, nit.mid],
                           [(0, 0, 1, 1, 0), (0, 1, 1, 2, 0)])
    para = StructuredCode(8, 7, [ring.mid, nit.mid, nit.mid],
                          [(0, 0, 1, 1, 0), (0, 3, 1, 2, 0)])
    ka = ladder_keys(ortho, vocab)
    kb = ladder_keys(para, vocab)
    assert ka["A"] == kb["A"]
    assert ka["B"] == kb["B"]
    assert ka["C"] == kb["C"]
    assert ka["D"] != kb["D"]


def test_mdl_proxy_finite_positive():
    vocab = Vocabulary()
    tokens, rounds = learn_vocabulary(SYNTH, vocab, 6, log=lambda *a: None)
    seq = [bytes.fromhex(r["key"]) for r in rounds]
    codes = []
    for mol in SYNTH:
        rank = mol_rank(mol)
        tok = tokenize_frozen(mol, vocab, seq, rank)
        codes.append(build_structured_code(mol, tok, vocab, rank)["code"])
    mdl = mdl_proxy(codes, codes, SYNTH, SYNTH, vocab)
    assert np.isfinite(mdl["pscd_total_mean_bits"]) and mdl["pscd_total_mean_bits"] > 0
    assert np.isfinite(mdl["atom_singleton_mean_bits"]) and mdl["atom_singleton_mean_bits"] > 0
    assert mdl["ratio_total_over_atom"] > 0
