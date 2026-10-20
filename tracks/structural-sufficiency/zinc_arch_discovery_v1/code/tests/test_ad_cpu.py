"""Lightweight CPU correctness checks for the six candidates.

These tests use a tiny synthetic cache built from hand-written molecules; they
do not load ZINC, the P3 object cache, or any checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M  # noqa: E402
import ad_train as T  # noqa: E402


def _mol(atom_types, bonds, idx=0):
    C.ensure_p3_modules()
    import structural_relation_objects_v1 as sro

    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(
        idx,
        np.asarray(atom_types, dtype=np.int64),
        np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64),
        np.asarray(ea, dtype=np.int64),
    )


def _objects():
    # A branched acyclic molecule with a double bond and a triple bond.
    a = _mol(
        [0, 1, 2, 3, 4, 5, 6, 7],
        [(0, 1, 1), (1, 2, 2), (2, 3, 1), (2, 4, 1), (4, 5, 3), (5, 6, 1), (6, 7, 1)],
        0,
    )
    # A six-membered ring with an exocyclic branch.
    b = _mol(
        [0, 0, 1, 1, 2, 2, 3],
        [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1), (0, 6, 1)],
        1,
    )
    # Two rings sharing an atom (fused/spiro-like test of articulation).
    c = _mol(
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 1),
         (0, 6, 1), (6, 7, 1), (7, 8, 1), (8, 1, 1)],
        2,
    )
    return [a, b, c]


@pytest.fixture(scope="module")
def tiny_cache(monkeypatch_module=None):
    objs = _objects()
    old_load = D.load_p3_objects
    old_labels = D.official_labels
    D.load_p3_objects = lambda split: (objs, {"source": "synthetic", "path": None, "sha256": None})
    D.official_labels = lambda split: np.asarray([0.1, -0.3, 0.7], dtype=np.float32)
    try:
        data = D.build_split_v2("valid")
    finally:
        D.load_p3_objects = old_load
        D.official_labels = old_labels
    return data


def _norm():
    return {
        "global_mean": np.zeros(33, dtype=np.float32),
        "global_scale": np.ones(33, dtype=np.float32),
        "root_scalar_mean": np.zeros((3, D.ROOT_SCALARS), dtype=np.float32),
        "root_scalar_scale": np.ones((3, D.ROOT_SCALARS), dtype=np.float32),
        "block_scalar_mean": np.zeros(D.BLOCK_SCALARS, dtype=np.float32),
        "block_scalar_scale": np.ones(D.BLOCK_SCALARS, dtype=np.float32),
    }


def test_category_tables_are_symmetric_and_sparse():
    assert D._PAIR_INDEX.shape == (21, 21)
    for a in range(21):
        for b in range(21):
            assert D._PAIR_INDEX[a, b] == D._PAIR_INDEX[b, a]
            assert 0 <= D._PAIR_INDEX[a, b] < D.PAIR_CATEGORIES
    assert D.JOINT_CATEGORIES == 693
    assert D.TWO_HOP_CATEGORIES == 189


def test_synthetic_cache_audit(tiny_cache):
    audit = D.audit_split("valid", tiny_cache)
    assert audit["n_molecules"] == 3
    assert audit["joint_mass_check_violations_sampled"] == 0
    assert audit["block_joint_mass_ok_sampled"] is True
    assert audit["walk_canonical_sample_ok"] is True
    # full atom composition is conserved, articulation not double counted
    counts = np.asarray(tiny_cache["atom_count_graph"])
    assert np.array_equal(counts.sum(1), np.asarray(tiny_cache["n_atoms_graph"]))
    bond_counts = np.asarray(tiny_cache["bond_count_graph"])
    assert np.array_equal(bond_counts.sum(1), np.asarray(tiny_cache["n_bonds_graph"]))


def test_walk_bank_has_all_lengths_and_reversal_canonical(tiny_cache):
    wa = np.asarray(tiny_cache["walk_atom"])
    wb = np.asarray(tiny_cache["walk_bond"])
    an = np.asarray(tiny_cache["walk_anon"])
    idx = np.asarray(tiny_cache["walk_atom_idx"])
    assert wa.shape[1] == D.WALK_BANK
    for k in range(D.WALK_BANK):
        L = D.WALK_LENGTHS[k // D.WALK_PER_LENGTH]
        for i in range(wa.shape[0]):
            assert np.array_equal(D._anon_pattern(list(idx[i, k, : L + 1])), an[i, k, : L + 1])
            ra = wa[i, k, : L + 1][::-1]
            rb = wb[i, k, :L][::-1]
            rn = D._anon_pattern(list(idx[i, k, : L + 1][::-1]))
            f = (wa[i, k, : L + 1].tobytes(), wb[i, k, :L].tobytes(), an[i, k, : L + 1].tobytes())
            r = (ra.tobytes(), rb.tobytes(), rn.tobytes())
            assert f <= r


@pytest.mark.parametrize("candidate", M.CANDIDATES)
def test_candidate_forward_backward_all_parameters(candidate, tiny_cache):
    model, info = M.build_model(candidate, _norm(), 0, "cpu")
    assert info["parameters"] <= 500_000
    cols = T._walk_cols(torch.Generator().manual_seed(0)) if candidate in ("P0", "P1") else None
    batch = T.collate(tiny_cache, np.arange(3), candidate, valid=False, walk_cols=cols)
    pred = model(batch)
    assert pred.shape == (3,)
    assert torch.isfinite(pred).all()
    pred.mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert torch.count_nonzero(p.grad).item() > 0, name


@pytest.mark.parametrize("candidate", ("R1", "P1", "B"))
def test_attention_cross_graph_isolation(candidate, tiny_cache):
    model, _ = M.build_model(candidate, _norm(), 0, "cpu")
    model.eval()
    cols = T._walk_cols(torch.Generator().manual_seed(0)) if candidate in ("P0", "P1") else None
    batch = T.collate(tiny_cache, np.arange(3), candidate, valid=False, walk_cols=cols)
    with torch.no_grad():
        together = model(batch).numpy()
        separate = []
        for i in range(3):
            b = T.collate(tiny_cache, [i], candidate, valid=False, walk_cols=cols)
            separate.append(float(model(b).numpy()[0]))
    assert np.allclose(together, np.asarray(separate), atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("candidate", M.CANDIDATES)
def test_shuffled_noncontiguous_rows(candidate, tiny_cache):
    model, _ = M.build_model(candidate, _norm(), 0, "cpu")
    cols = T._walk_cols(torch.Generator().manual_seed(0)) if candidate in ("P0", "P1") else None
    batch = T.collate(tiny_cache, [2, 0, 1], candidate, valid=False, walk_cols=cols)
    if candidate == "T":
        assert int(batch["bond_index"].max()) < int(batch["atom_feat"].shape[0])
        assert int(batch["junc_index"].max()) < int(batch["atom_feat"].shape[0])
        assert int(batch["junc_bond"].max()) < int(batch["bond_feat"].shape[0])
    pred = model(batch)
    assert pred.shape == (3,)
    assert torch.isfinite(pred).all()


def test_rooted_relabelling_invariance():
    # Relabel atom indices by a permutation; rooted features mapped back must
    # be unchanged (same graph, same root).
    C.ensure_p3_modules()
    atom_types = [0, 1, 2, 3, 4]
    bonds = [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1)]
    obj = _mol(atom_types, bonds)
    perm = np.asarray([3, 0, 4, 1, 2])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    bonds_p = [(int(inv[u]), int(inv[v]), bt) for u, v, bt in bonds]
    obj_p = _mol([atom_types[int(perm[i])] for i in range(len(atom_types))], bonds_p)

    def feats(o):
        at = D._atom_types(o)
        bt = np.asarray(o.bond_type, dtype=np.int64)
        ra, rb, mr = D._ring_flags(o)
        adj = D._adjacency(int(o.n_atoms), np.asarray(o.bond_index), bt)
        deg = np.asarray([len(adj[i]) for i in range(int(o.n_atoms))])
        dist = D._all_pairs_dist(int(o.n_atoms), adj)
        return D._rooted(o, at, bt, ra, rb, deg, adj, dist)

    j1, s1, c1 = feats(obj)
    j2, s2, c2 = feats(obj_p)
    # root i in the original is root inv[i] in the permuted graph
    for i in range(len(atom_types)):
        k = int(inv[i])
        assert np.array_equal(j1[i], j2[k])
        assert np.array_equal(s1[i], s2[k])
        assert np.allclose(c1[i], c2[k], atol=1e-6)
