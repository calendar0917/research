"""Fast unit tests for the SFEC v1 correction and estimator primitives."""

from __future__ import annotations

import numpy as np

import sfec1_data as D


def _contexts(slists, ylists):
    k = np.asarray([len(x) for x in slists], dtype=np.int32)
    off = np.r_[0, np.cumsum(k)].astype(np.int64)
    S = np.concatenate([np.asarray(x, dtype=np.int64) for x in slists])
    y = np.concatenate([np.asarray(x, dtype=np.float64) for x in ylists])
    n_cells = len(S)
    return D.Contexts(
        C=np.arange(len(slists), dtype=np.int32), off=off, k=k, S=S.astype(np.int32),
        y=y, n=np.ones(n_cells, dtype=np.int32), sd=np.zeros(n_cells),
        cell_key=np.arange(n_cells, dtype=np.int64),
        cell_C=np.repeat(np.arange(len(slists)), k).astype(np.int32),
        cell_S=S.astype(np.int32), cell_b=np.ones(n_cells, np.int8),
        cell_mol_first=np.arange(n_cells, dtype=np.int32),
        cell_mol_n=np.ones(n_cells, dtype=np.int32),
    )


def _design(slists, ylists):
    off = [0]
    O_parts, y_parts, pos_parts, k_parts = [], [], [], []
    pos = 0
    for S, y in zip(slists, ylists):
        S = np.asarray(S, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        O_parts.append(S)
        y_parts.append(y)
        pos_parts.append(np.arange(pos, pos + len(S), dtype=np.int64))
        pos += len(S)
        k_parts.append(len(S))
        off.append(pos)
    return D.Design(
        ctx_idx=np.arange(len(slists), dtype=np.int64),
        off=np.asarray(off, dtype=np.int64),
        O=np.concatenate(O_parts), y=np.concatenate(y_parts),
        q=np.zeros(len(np.concatenate(O_parts)), dtype=np.int64),
        k=np.asarray(k_parts, dtype=np.int64),
        cell_pos=np.concatenate(pos_parts),
    )


def test_canonical_orientation_and_delta():
    # ranks deliberately not sorted inside the context
    ctx = _contexts([[0, 1, 2]], [[10.0, 11.0, 13.0]])
    rank = np.array([2, 0, 1], dtype=np.int64)  # S_id -> rank
    pairs = D.build_pairs(ctx, rank, canonical=True)
    assert np.all(pairs["A"] < pairs["B"])
    for a, b, d in zip(pairs["A"], pairs["B"], pairs["delta"]):
        ya = ctx.y[np.where(rank[ctx.S] == a)[0][0]]
        yb = ctx.y[np.where(rank[ctx.S] == b)[0][0]]
        assert abs(float(d) - (yb - ya)) <= 1e-12
    # one observation per exact context and canonical pair
    assert len(set(zip(pairs["ctx"], pairs["A"], pairs["B"]))) == len(pairs["delta"])


def test_canonical_shuffle_invariance():
    ctx = _contexts([[0, 1, 2, 3]], [[1.0, 4.0, 2.0, 8.0]])
    rank = np.array([3, 0, 2, 1], dtype=np.int64)
    ref = D.build_pairs(ctx, rank, canonical=True)
    rng = np.random.default_rng(0)
    for _ in range(10):
        order = rng.permutation(4)
        ctx2 = _contexts([[0, 1, 2, 3]], [ctx.y[order].tolist()])
        # S ids are also permuted with the cells
        ctx2.S = ctx2.S[order]
        ctx2.cell_S = ctx2.cell_S[order]
        got = D.build_pairs(ctx2, rank, canonical=True)
        a = sorted(zip(ref["A"].tolist(), ref["B"].tolist(), ref["delta"].round(12).tolist()))
        b = sorted(zip(got["A"].tolist(), got["B"].tolist(), got["delta"].round(12).tolist()))
        assert a == b


def test_no_reverse_pair_duplicates():
    ctx = _contexts([[0, 1, 2]], [[0.0, 1.0, 2.0]])
    rank = np.array([2, 0, 1], dtype=np.int64)
    pairs = D.build_pairs(ctx, rank, canonical=True)
    keys = set(zip(pairs["A"].tolist(), pairs["B"].tolist()))
    assert all((b, a) not in keys for a, b in keys)


def test_shared_additive_recovery():
    b_true = np.array([0.4, -0.2, 0.7, 0.1, -0.5, 0.3])
    a_true = np.array([1.0, -0.3, 0.2, 0.8])
    S = [[0, 1, 2, 3], [0, 2, 4, 5], [1, 3, 4, 5], [0, 1, 4, 5]]
    y = [[a_true[c] + b_true[s] for s in s_list] for c, s_list in enumerate(S)]
    fit = D.fit_shared(_design(S, y))
    for s_list in S:
        for i in s_list:
            for j in s_list:
                assert abs((fit["b"][j] - fit["b"][i]) - (b_true[j] - b_true[i])) <= 1e-8


def test_strict_mask_indirect_path():
    b_true = np.array([0.5, -0.3, 0.8, 0.1, -0.2])
    a_true = np.array([1.0, 0.2, -0.4, 0.7, -0.1, 0.3, 0.5])
    S = [[0, 2, 3], [0, 2, 4], [0, 1, 3], [1, 2, 3],
         [1, 2, 4], [0, 3, 4], [2, 3, 4]]
    y = [[a_true[c] + b_true[s] for s in s_list] for c, s_list in enumerate(S)]
    design = _design(S, y)
    keep = np.ones(len(design.O), dtype=bool)
    for ci in (0, 1):
        a, b = int(design.off[ci]), int(design.off[ci + 1])
        pos = a + int(np.flatnonzero(design.O[a:b] == 2)[0])
        keep[pos] = False
    fit = D.fit_shared(design, keep=keep)
    assert abs((fit["b"][2] - fit["b"][0]) - (b_true[2] - b_true[0])) <= 1e-8
    assert fit["comp"][0] == fit["comp"][2]
