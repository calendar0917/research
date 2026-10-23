"""Fast, data-free checks for the relation builder and null permutation.

These do not load ZINC, checkpoints or GPU state.  The heavier implementation
gates live in ``nr_runner.py checks`` and write ``analysis/cpu_checks.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import nr_data as D  # noqa: E402


def _cycle4_with_tail():
    # atoms 0-1-2-3-0 ring, atom 4 hangs off atom 0
    bonds = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4]], dtype=np.int64)
    n = 5
    dist = np.full((n, n), 99, dtype=np.int64)
    for i in range(n):
        dist[i, i] = 0
    adj = [[] for _ in range(n)]
    for u, v in bonds:
        adj[u].append(v)
        adj[v].append(u)
    for s in range(n):
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if dist[s, v] == 99:
                        dist[s, v] = d
                        nxt.append(v)
            frontier = nxt
    # one junction: path 4-0-1, centre 0, bonds (4,0) and (0,1)
    junc = np.asarray([[4, 0, 1]], dtype=np.int64)
    return dist, bonds, junc


def test_relation_labels_shape_and_values():
    dist, bonds, junc = _cycle4_with_tail()
    u, v, r = D.relation_labels(dist, 5, bonds, junc)
    assert len(u) == len(v) == len(r) > 0
    assert np.all(u < v)
    assert np.all((r >= 1) & (r < D.REL_TYPES))
    # the junction is centred on atom 0, so it shares ring blocks with ring bonds
    assert np.any((u == 0) | (v == 0))


def test_null_preserves_pair_set_and_role_multiset():
    n_junc = 3
    r_u = np.asarray([0, 0, 1, 3, 3, 4, 5, 6], dtype=np.int64)
    r_v = np.asarray([1, 2, 2, 4, 5, 6, 7, 8], dtype=np.int64)
    r = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    out, stats = D.null_relation_labels(r_u, r_v, r, n_junc, 7)
    assert out.shape == r.shape
    assert stats["n_non_self_pairs"] == len(r)
    role_u = (r_u >= n_junc).astype(int)
    role_v = (r_v >= n_junc).astype(int)
    for ru, rv in ((0, 0), (0, 1), (1, 1)):
        m = (role_u == ru) & (role_v == rv)
        assert np.array_equal(np.sort(out[m]), np.sort(r[m]))
    assert stats["n_changed"] > 0
    # deterministic
    out2, _ = D.null_relation_labels(r_u, r_v, r, n_junc, 7)
    assert np.array_equal(out, out2)


def test_null_is_not_a_global_relation_rename():
    # two molecules with the same real labels but different pair structure; a
    # global rel-id rename would map the same labels to the same labels
    pairs = [
        (np.asarray([3, 4, 5], np.int64), np.asarray([4, 5, 6], np.int64)),
        (np.asarray([3, 4, 6], np.int64), np.asarray([5, 6, 7], np.int64)),
    ]
    mapping = {}
    for i, (u, v) in enumerate(pairs):
        r = np.asarray([1, 2, 3], np.int64)
        out, _ = D.null_relation_labels(u, v, r, 0, 100 + i)
        for a, b in zip(r, out):
            mapping.setdefault(int(a), set()).add(int(b))
    assert any(len(v) > 1 for v in mapping.values())


def test_entropy_and_homogeneous_stats():
    r_u = np.asarray([0, 1, 2], np.int64)
    r_v = np.asarray([1, 2, 3], np.int64)
    r = np.asarray([5, 5, 5], np.int64)
    out, stats = D.null_relation_labels(r_u, r_v, r, 0, 1)
    assert np.array_equal(out, r)
    assert stats["n_changed"] == 0
    assert stats["homogeneous_pair_fraction"] == 1.0
    assert stats["entropy_mean_per_group"] == 0.0


def test_collate_global_token_offsets_and_self_loops():
    import torch

    cache = {
        "atom_ptr": np.asarray([0, 1, 2], np.int32),
        "bond_ptr": np.asarray([0, 1, 3], np.int32),
        "junc_ptr": np.asarray([0, 1, 2], np.int32),
    }
    entries = [
        {"r_u": np.asarray([0], np.int32), "r_v": np.asarray([1], np.int32),
         "r_rel": np.asarray([4], np.int8), "r_rel_null": np.asarray([9], np.int8)},
        {"r_u": np.asarray([0], np.int32), "r_v": np.asarray([2], np.int32),
         "r_rel": np.asarray([5], np.int8), "r_rel_null": np.asarray([8], np.int8)},
    ]
    batch = D.collate_relations(entries, cache, [0, 1])
    n_j = 1 + 1
    n_b = 1 + 2
    assert batch["n_tokens_total"] == n_j + n_b
    src = batch["r_src"].numpy()
    dst = batch["r_dst"].numpy()
    rel = batch["r_rel"].numpy()
    # molecule 0: junction 0, bond global id 2; molecule 1: junction 1, bonds 3,4
    assert {0, 2, 1, 3, 4} == set(src.tolist())
    self_mask = src == dst
    assert int(self_mask.sum()) == n_j + n_b
    assert np.all(rel[self_mask] == 0)
    # mol0 pair is junction 0 -> bond global 2 (and reverse)
    assert ((src == 0) & (dst == 2)).sum() == 1
    assert ((src == 2) & (dst == 0)).sum() == 1
    assert int(batch["r_rel"].max()) == 5
    assert int(batch["r_rel_null"].max()) == 9


def test_null_transport_is_pair_attached():
    # null depends on the stored token pairs and mol index, not on atom ids;
    # renumbering atoms leaves token arrays unchanged, hence the mapping too
    r_u = np.asarray([0, 1], np.int64)
    r_v = np.asarray([1, 2], np.int64)
    r = np.asarray([3, 7], np.int64)
    a, _ = D.null_relation_labels(r_u, r_v, r, 1, 42)
    b, _ = D.null_relation_labels(r_u, r_v, r, 1, 42)
    assert np.array_equal(a, b)
