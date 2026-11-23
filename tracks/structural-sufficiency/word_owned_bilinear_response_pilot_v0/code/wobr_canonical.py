"""Exact root-fixed canonical certificates for attributed graph words.

A word has at most four atoms.  The root is slot 0; the remaining at most three
slots are relabelled by exhaustive permutation, and the lexicographically
minimal certificate is kept.  Node and edge labels are exact mixed-radix
encodings of all nine OGB atom categorical fields and all three OGB bond
categorical fields.  Equal certificates therefore mean equal rooted attributed
induced graphs; no hash-collision ID is used as an isomorphism certificate.

For size 1 the certificate is the root node label followed by five edge slots
and three unused node slots set to -1.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import wobr_structure as S

_PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_PERMS: dict[int, list[tuple[int, ...]]] = {}


def mixed_radix(values: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    out = np.zeros(values.shape[0], np.int64)
    for i, dim in enumerate(dims):
        out = out * int(dim) + values[:, i].astype(np.int64)
    return out


def _perms_for(size: int) -> list[tuple[int, ...]]:
    if size not in _PERMS:
        import itertools

        _PERMS[size] = [(0,) + p for p in itertools.permutations(range(1, size))]
    return _PERMS[size]


def canonical_certificates(struct: S.MoleculeStruct, atom_dims: Sequence[int],
                           bond_dims: Sequence[int]) -> np.ndarray:
    """Return an exact int64 certificate [n_words, 10] for one molecule."""
    n_words = int(struct.n_words)
    if n_words == 0:
        return np.zeros((0, 10), np.int64)
    support = np.asarray(struct.word_support, np.int64)          # [W,4], -1 pad
    internal = np.asarray(struct.word_internal, np.int64)        # [W,6], -1 pad
    sizes = np.asarray(struct.word_size, np.int64)
    node_label = mixed_radix(np.asarray(struct.atom_feat, np.int64), atom_dims)
    bond_label = mixed_radix(np.asarray(struct.bond_feat, np.int64), bond_dims)

    # node labels per slot; -1 for padding
    node = np.full((n_words, 4), -1, np.int64)
    valid_node = support >= 0
    safe_support = np.where(valid_node, support, 0)
    node[valid_node] = node_label[safe_support[valid_node]]

    # map internal bond endpoints to support slots
    safe_internal = np.where(internal >= 0, internal, 0)
    bu = np.asarray(struct.bond_u, np.int64)[safe_internal]       # [W,6]
    bv = np.asarray(struct.bond_v, np.int64)[safe_internal]
    eq_u = support[:, None, :] == bu[:, :, None]                  # [W,6,4]
    eq_v = support[:, None, :] == bv[:, :, None]
    slot_u = eq_u.argmax(axis=2)
    slot_v = eq_v.argmax(axis=2)
    valid_edge = internal >= 0
    if np.any(valid_edge & (~eq_u.any(axis=2) | ~eq_v.any(axis=2))):
        raise ValueError("internal bond endpoint is not in the word support")

    edge = np.full((n_words, 4, 4), -1, np.int64)
    w_idx = np.arange(n_words)[:, None].repeat(internal.shape[1], axis=1)
    ecode = np.where(valid_edge, bond_label[safe_internal], -1)
    edge[w_idx[valid_edge], slot_u[valid_edge], slot_v[valid_edge]] = ecode[valid_edge]
    edge[w_idx[valid_edge], slot_v[valid_edge], slot_u[valid_edge]] = ecode[valid_edge]

    # The stored support is sorted by atom id, so the root is not necessarily
    # slot 0.  Move the real root to slot 0 before root-fixed canonicalisation.
    root_id = np.asarray(struct.word_root, np.int64)
    root_pos = (support == root_id[:, None]).argmax(axis=1)
    slots = np.arange(4)[None, :].repeat(n_words, axis=0)
    others = np.argsort(np.where(slots == root_pos[:, None], 99, slots), axis=1)
    order = np.concatenate([root_pos[:, None], others[:, :3]], axis=1)
    node = np.take_along_axis(node, order, axis=1)
    edge = edge[np.arange(n_words)[:, None, None], order[:, :, None], order[:, None, :]]

    out = np.zeros((n_words, 10), np.int64)
    for size in (1, 2, 3, 4):
        rows = np.flatnonzero(sizes == size)
        if rows.size == 0:
            continue
        ns = node[rows]
        es = edge[rows]
        best = np.full((rows.size, 10), np.iinfo(np.int64).max, np.int64)
        for perm in _perms_for(size):
            cand = np.empty((rows.size, 10), np.int64)
            for slot in range(4):
                cand[:, slot] = ns[:, perm[slot]] if slot < size else -1
            for t, (a, b) in enumerate(_PAIR_ORDER):
                if a < size and b < size:
                    cand[:, 4 + t] = es[:, perm[a], perm[b]]
                else:
                    cand[:, 4 + t] = -1
            undecided = np.ones(rows.size, bool)
            better = np.zeros(rows.size, bool)
            for col in range(10):
                diff = cand[:, col] < best[:, col]
                better |= undecided & diff
                undecided &= cand[:, col] == best[:, col]
                if not undecided.any():
                    break
            best[better] = cand[better]
        out[rows] = best
    return out


def certificate_key(cert: np.ndarray) -> np.ndarray:
    """Exact bytes view for grouping; not a compressed/hash identifier."""
    cert = np.ascontiguousarray(np.asarray(cert, np.int64))
    if cert.ndim != 2 or cert.shape[1] != 10:
        raise ValueError("certificate must be [n,10]")
    return cert.view(np.dtype((np.void, 10 * 8))).ravel()


def _unmix(value: int, dims: Sequence[int]) -> list[int]:
    digits: list[int] = []
    value = int(value)
    for dim in reversed(list(dims)):
        digits.append(int(value % int(dim)))
        value //= int(dim)
    return list(reversed(digits))


def decode_certificate(row: Sequence[int], size: int, atom_dims: Sequence[int],
                       bond_dims: Sequence[int]) -> dict[str, Any]:
    nodes = []
    for i in range(int(size)):
        nodes.append(_unmix(int(row[i]), atom_dims))
    edges = []
    for t, (a, b) in enumerate(_PAIR_ORDER):
        if a < size and b < size:
            code = int(row[4 + t])
            if code >= 0:
                edges.append({"slots": [a, b], "fields": _unmix(code, bond_dims)})
    return {"size": int(size), "node_fields": nodes, "internal_bonds": edges,
            "root_slot": 0}


def certificate_row(cert: np.ndarray, size: int, atom_dims: Sequence[int],
                    bond_dims: Sequence[int]) -> dict[str, Any]:
    decoded = decode_certificate(cert, size, atom_dims, bond_dims)
    return {
        "size": int(size),
        "certificate": [int(x) for x in cert],
        "node_fields": decoded["node_fields"],
        "internal_bonds": decoded["internal_bonds"],
        "root_slot": 0,
    }