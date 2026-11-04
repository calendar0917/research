"""Small unit tests for the S2 count construction.  No data or GPU required."""

from __future__ import annotations

import numpy as np

import pc_common as C
import pc_s2 as S


def _chain() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atom = np.zeros((4, 9), dtype=np.int64)
    atom[:, 0] = [5, 6, 7, 8]
    bonds = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    raw = np.zeros((3, 3), dtype=np.int64)
    raw[:, 0] = [0, 1, 2]
    return atom, raw, bonds


def test_chain_type_keys() -> None:
    atom, raw, bonds = _chain()
    paths = S.enumerate_paths(atom, raw, bonds)
    assert paths["n_paths"] == 4
    got = {tuple(int(x) for x in row) for row in S.type_keys(atom, raw, paths).tolist()}
    assert got == {(5, 0, 6, 1, 7), (6, 1, 7, 2, 8), (7, 1, 6, 0, 5), (8, 2, 7, 1, 6)}


def test_triangle_no_root_closure() -> None:
    atom = np.zeros((3, 9), dtype=np.int64)
    atom[:, 0] = 6
    bonds = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    raw = np.zeros((3, 3), dtype=np.int64)
    paths = S.enumerate_paths(atom, raw, bonds)
    assert paths["n_paths"] == 6
    assert np.all(paths["end"] != paths["root"])


def test_full_key_layout() -> None:
    atom, raw, bonds = _chain()
    paths = S.enumerate_paths(atom, raw, bonds)
    full = S.full_keys(atom, raw, paths)
    assert full.shape == (4, 33)
    assert np.array_equal(full[0, :9], atom[0])
    assert np.array_equal(full[0, 9:12], raw[0])
    assert np.array_equal(full[0, 12:21], atom[1])
    assert np.array_equal(full[0, 21:24], raw[1])
    assert np.array_equal(full[0, 24:], atom[2])


def _random_graph(seed: int = 0, n: int = 10) -> dict:
    rng = np.random.default_rng(seed)
    atom = np.zeros((n, 9), dtype=np.int64)
    atom[:, 0] = rng.choice([5, 6, 7, 8], size=n)
    for c, k in enumerate(C.ATOM_VOCABS[1:], start=1):
        atom[:, c] = rng.integers(0, k, size=n)
    pairs = set()
    for i in range(1, n):
        pairs.add((int(rng.integers(0, i)), i))
    while len(pairs) < 14:
        a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
        pairs.add((int(a), int(b)))
    pairs = sorted(pairs)
    bonds = np.asarray(pairs, dtype=np.int64)
    raw = np.stack([rng.integers(0, k, size=len(pairs)) for k in C.BOND_VOCABS], axis=1).astype(
        np.int64
    )
    return {
        "gid": 0,
        "atom_raw": atom,
        "bond_raw": raw,
        "bond_index": bonds,
        "n_paths_cache": S.enumerate_paths(atom, raw, bonds)["n_paths"],
    }


def test_relabel_and_bond_order_invariance() -> None:
    rng = np.random.default_rng(0)
    g = _random_graph()
    base = S.graph_features(g)
    n = g["atom_raw"].shape[0]
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    pairs2, raw2 = [], []
    for (a, b), ba in zip(g["bond_index"].tolist(), g["bond_raw"].tolist()):
        x, y = int(inv[a]), int(inv[b])
        if x > y:
            x, y = y, x
        pairs2.append((x, y))
        raw2.append(ba)
    order = np.lexsort((np.asarray(pairs2)[:, 1], np.asarray(pairs2)[:, 0]))
    pairs2 = [pairs2[i] for i in order]
    raw2 = np.asarray(raw2, dtype=np.int64)[order]
    g2 = {
        "gid": 0,
        "atom_raw": g["atom_raw"][perm],
        "bond_raw": raw2,
        "bond_index": np.asarray(pairs2, dtype=np.int64),
        "n_paths_cache": base["n_paths"],
    }
    rel = S.graph_features(g2)
    order = rng.permutation(g["bond_index"].shape[0])
    g3 = {
        "gid": 0,
        "atom_raw": g["atom_raw"],
        "bond_raw": g["bond_raw"][order],
        "bond_index": g["bond_index"][order],
        "n_paths_cache": base["n_paths"],
    }
    shuf = S.graph_features(g3)
    for key in ("counts_type", "counts_full", "counts_null"):
        assert np.array_equal(base[key], rel[key]), key
        assert np.array_equal(base[key], shuf[key]), key


def test_full_null_marginals() -> None:
    g = _random_graph()
    f = S.graph_features(g)
    assert f["null_audit"]["marginal_first_equal"]
    assert f["null_audit"]["marginal_second_equal"]
    assert f["null_audit"]["permutation_derangement"]
    assert f["null_audit"]["joint_equal_rows"] < f["n_paths"] or f["n_paths"] <= 1
    assert int(f["counts_full"].sum()) == int(f["counts_null"].sum()) == int(f["n_paths"])


def test_hash_is_frozen() -> None:
    keys = [bytes(range(33))] + [
        bytes(((i * 37 + j * 11) % 256) for j in range(33)) for i in range(3)
    ]
    expected = [
        (8342072969736565132, 396),
        (14133181029077935205, 1125),
        (18337858077272252758, 342),
        (12589868814983397228, 1900),
    ]
    assert [(C.stable_digest_int(k), C.stable_bucket(k)) for k in keys] == expected
