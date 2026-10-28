"""Correctness of the raw two-hop enumeration and of the candidate's invariance.

Every path carries root / middle / endpoint / first bond id / second bond id.
The enumeration is compared **element-wise** (not by histogram) against the frozen
LCRC cache for the whole official train and valid splits, plus invariant and
boundary checks for the new statistic.
"""

from __future__ import annotations

import numpy as np
import pytest

import pb_common as P
import pb_paths as PP
import pb_features as F


@pytest.mark.parametrize("split", ["train", "valid"])
def test_full_split_elementwise_agreement_with_frozen_cache(split):
    cache = P.load_cache(split)
    paths = PP.split_paths(cache)
    res = PP.crosscheck_split(cache, paths)
    assert res["passed"], res
    assert res["n_paths"] == res["n_paths_from_cache"]
    assert res["counts"]["root_equals_end"] == 0
    assert res["counts"]["e1_equals_e2"] == 0
    assert res["counts"]["bond_e1_incident_violations"] == 0
    assert res["counts"]["bond_e2_incident_violations"] == 0


def test_bond_ids_are_real_and_distinct():
    cache = P.load_cache("valid")
    paths = PP.split_paths(cache)
    ends = PP.bond_endpoints(cache)
    inc1 = ends[paths["e1"]]
    inc2 = ends[paths["e2"]]
    ok1 = ((inc1[:, 0] == paths["root"]) & (inc1[:, 1] == paths["mid"])) | \
          ((inc1[:, 1] == paths["root"]) & (inc1[:, 0] == paths["mid"]))
    ok2 = ((inc2[:, 0] == paths["mid"]) & (inc2[:, 1] == paths["end"])) | \
          ((inc2[:, 1] == paths["mid"]) & (inc2[:, 0] == paths["end"]))
    assert bool(ok1.all()) and bool(ok2.all())
    assert int(np.sum(paths["e1"] == paths["e2"])) == 0


def test_zero_and_single_path_boundaries():
    cache = PP.synthetic_cache([
        ([0, 0], [(0, 1, 1)]),                                  # 2 atoms: no 2-path
        ([0, 0, 0], [(0, 1, 1), (1, 2, 1)]),                    # chain: one path per end
        ([0, 0, 0, 0, 0, 0], [(0, 1, 1), (1, 2, 1), (2, 3, 1),
                              (3, 4, 1), (4, 5, 1), (5, 0, 1)]),  # 6-ring (closures)
        ([0, 0, 0, 0, 0, 0], [(0, 1, 1), (1, 2, 1), (2, 3, 1),
                              (3, 4, 1), (4, 5, 1)]),           # 6-chain
    ])
    paths = PP.split_paths(cache)
    n = np.bincount(paths["root"], minlength=int(np.asarray(cache["atom_ptr"])[-1]))
    assert (n == 0).sum() == 3  # 2-atom molecule + the middle atom of the 3-chain
    assert (n == 1).sum() == 6  # 2 chain ends + 4 chain ends of the 6-chain
    model, _ = P.load_frozen_a("lcrcA_s0")
    enc = F.encode(model, cache)
    proj = P.projections(P.PRIMARY_SEED)
    st = F.t_stats(paths, *F.path_factors(enc["a"], enc["b"], paths, proj), enc["n_atoms"])
    assert np.isfinite(st["T_real"]).all() and np.isfinite(st["D"]).all()
    assert np.allclose(st["T_real"][n == 0], 0.0)
    assert np.allclose(st["D"][n == 1], 0.0)
    # closure needs a triangle: the 6-ring has none, the fused-ring graph does
    # closure needs a triangle: none in a 6-ring, six in a kite
    ring = PP.synthetic_cache([([0] * 6, [(0, 1, 1), (1, 2, 1), (2, 3, 1),
                                          (3, 4, 1), (4, 5, 1), (5, 0, 1)])])
    kite = PP.synthetic_cache([([0, 1, 2, 3], [(0, 1, 1), (1, 2, 1), (2, 0, 1), (2, 3, 1)])])
    assert int(np.asarray(ring["path_closure"]).sum()) == 0
    assert int(np.asarray(kite["path_closure"]).sum()) == 6


def _single_molecule(va, m):
    ap = np.asarray(va["atom_ptr"])
    bp = np.asarray(va["bond_ptr"])
    a0, a1 = int(ap[m]), int(ap[m + 1])
    b0, b1 = int(bp[m]), int(bp[m + 1])
    return (np.asarray(va["atom_type"][a0:a1], dtype=np.int64),
            [(int(u), int(v), int(bt)) for (u, v), bt in
             zip(np.asarray(va["bond_index"][b0:b1]), np.asarray(va["bond_type"][b0:b1]))])


def test_relabel_and_bond_reversal_invariance():
    va = P.load_cache("valid")
    model, _ = P.load_frozen_a("lcrcA_s0")
    proj = P.projections(P.PRIMARY_SEED)
    rng = np.random.default_rng(1234)
    for m in range(4):
        types, bonds = _single_molecule(va, m)
        _, t_ref = _encode_t(model, PP.synthetic_cache([(types, bonds)]), proj)
        perm = rng.permutation(len(types))
        inv = np.zeros_like(types)
        inv[perm] = types
        relabelled = [(int(perm[u]), int(perm[v]), bt) for u, v, bt in bonds]
        _, t_rel = _encode_t(model, PP.synthetic_cache([(inv, relabelled)]), proj)
        assert np.abs(_rows_sorted(t_rel) - _rows_sorted(t_ref)).max() < 1e-5
        reversed_bonds = [(v, u, bt) for u, v, bt in bonds]
        _, t_rev = _encode_t(model, PP.synthetic_cache([(types, reversed_bonds)]), proj)
        assert np.abs(t_rev - t_ref).max() < 1e-6


def _rows_sorted(a: np.ndarray) -> np.ndarray:
    """Sort rows lexicographically (a graded multiset comparator)."""
    return a[np.lexsort(a.T[::-1])]


def _encode_t(model, cache, proj):
    enc = F.encode(model, cache)
    paths = PP.split_paths(cache)
    st = F.t_stats(paths, *F.path_factors(enc["a"], enc["b"], paths, proj), enc["n_atoms"])
    return enc, st["T_real"]


def test_batch_split_invariance():
    va = P.load_cache("valid")
    model, _ = P.load_frozen_a("lcrcA_s0")
    proj = P.projections(P.PRIMARY_SEED)
    graphs = [_single_molecule(va, m) for m in range(4)]
    joined = PP.synthetic_cache(graphs)
    enc_joined = F.encode(model, joined, batch_size=4)
    paths_joined = PP.split_paths(joined)
    t_joined = F.t_stats(paths_joined, *F.path_factors(enc_joined["a"], enc_joined["b"],
                                                       paths_joined, proj),
                         enc_joined["n_atoms"])["T_real"]
    a_parts, t_parts = [], []
    for g in graphs:
        one = PP.synthetic_cache([g])
        enc_one, t_one = _encode_t(model, one, proj)
        a_parts.append(enc_one["a"])
        t_parts.append(t_one)
    a_sep = np.concatenate(a_parts)
    t_sep = np.concatenate(t_parts)
    assert np.abs(a_sep - enc_joined["a"]).max() < 1e-5
    assert np.abs(t_sep - t_joined).max() < 1e-5
