"""Label-free WL features for real-edit fragments (route C).

Three blocks of fixed 512-dim counts (initial token, two WL updates) are
L2-normalised separately; two raw log-size scalars follow and are standardised
with Fit-pair weights inside each kernel fold.  Tokens use the full raw 9/3 OGB
columns, the port flag, ``min(dist_to_port, 3)`` and, **inside the port token**,
the three cut-bond columns.  Hashing is fixed ``blake2b``; Python's salted
``hash`` is never used.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BLOCKS = 3
DIM = 512
FEATURE_DIM = BLOCKS * DIM + 2
PERSON = b"route8h-wl"


# ---------------------------------------------------------------------------
# deterministic serialization / hashing
# ---------------------------------------------------------------------------
def _pack(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return b"b" + len(value).to_bytes(4, "little") + bytes(value)
    if isinstance(value, bool):
        return b"?" + bytes([int(value)])
    if isinstance(value, (int, np.integer)):
        return b"i" + int(value).to_bytes(8, "little", signed=True)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"s" + len(raw).to_bytes(4, "little") + raw
    if isinstance(value, (tuple, list)):
        parts = [_pack(v) for v in value]
        return b"t" + len(parts).to_bytes(4, "little") + b"".join(parts)
    raise TypeError(f"unsupported token part {type(value)!r}")


def digest(value: Any) -> bytes:
    return hashlib.blake2b(_pack(value), digest_size=16, person=PERSON).digest()


def bucket(token: bytes) -> int:
    return int.from_bytes(token[:8], "little") % DIM


# ---------------------------------------------------------------------------
# fragment induction
# ---------------------------------------------------------------------------
def _bfs(adj: Sequence[Sequence[int]], source: int, limit: int | None = None
         ) -> list[int]:
    dist = [-1] * len(adj)
    dist[int(source)] = 0
    frontier = [int(source)]
    depth = 0
    while frontier and (limit is None or depth < limit):
        depth += 1
        nxt = []
        for x in frontier:
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = depth
                    nxt.append(y)
        frontier = nxt
    return dist


@dataclass
class Fragment:
    atom_feat: np.ndarray
    bond_u: np.ndarray
    bond_v: np.ndarray
    bond_feat: np.ndarray
    port: int
    cut: tuple[int, int, int]


def induce(atom_feat: np.ndarray, bond_u: np.ndarray, bond_v: np.ndarray,
           bond_feat: np.ndarray, atoms: Iterable[int], port: int,
           cut: Sequence[int]) -> Fragment:
    atoms_sorted = np.asarray(sorted(int(a) for a in atoms), np.int64)
    remap = {int(a): i for i, a in enumerate(atoms_sorted.tolist())}
    keep = [(k, remap[int(a)], remap[int(b)]) for k, (a, b) in
            enumerate(zip(bond_u.tolist(), bond_v.tolist()))
            if int(a) in remap and int(b) in remap]
    return Fragment(
        atom_feat=np.asarray(atom_feat[atoms_sorted], np.int64),
        bond_u=np.asarray([k[1] for k in keep], np.int64),
        bond_v=np.asarray([k[2] for k in keep], np.int64),
        bond_feat=np.asarray([bond_feat[k[0]] for k in keep], np.int64).reshape(-1, 3),
        port=int(remap[int(port)]),
        cut=tuple(int(x) for x in cut))


def radius_ball(fragment: Fragment, radius: int) -> Fragment:
    adj = [[] for _ in range(fragment.atom_feat.shape[0])]
    for a, b in zip(fragment.bond_u.tolist(), fragment.bond_v.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    dist = _bfs(adj, fragment.port, radius)
    atoms = [i for i, d in enumerate(dist) if 0 <= d <= radius]
    return induce(fragment.atom_feat, fragment.bond_u, fragment.bond_v,
                  fragment.bond_feat, atoms, fragment.port, fragment.cut)


# ---------------------------------------------------------------------------
# WL features
# ---------------------------------------------------------------------------
def wl_features(fragment: Fragment) -> np.ndarray:
    n = int(fragment.atom_feat.shape[0])
    if n == 0:
        return np.zeros(FEATURE_DIM, np.float64)
    adj = [[] for _ in range(n)]
    for k, (a, b) in enumerate(zip(fragment.bond_u.tolist(), fragment.bond_v.tolist())):
        edge = tuple(int(x) for x in fragment.bond_feat[k])
        adj[a].append((b, edge))
        adj[b].append((a, edge))
    dist = _bfs([[y for y, _ in row] for row in adj], fragment.port)
    cut = tuple(int(x) for x in fragment.cut)
    tokens: list[bytes] = []
    for i in range(n):
        node = tuple(int(x) for x in fragment.atom_feat[i])
        port_flag = 1 if i == fragment.port else 0
        extra = cut if port_flag else (-1, -1, -1)
        tokens.append(digest(("node", node, port_flag, min(dist[i], 3), extra)))
    out = np.zeros(FEATURE_DIM, np.float64)
    blocks = []
    for block in range(BLOCKS):
        counts = np.zeros(DIM, np.float64)
        for token in tokens:
            counts[bucket(token)] += 1.0
        norm = float(np.linalg.norm(counts))
        if norm > 0:
            counts /= norm
        blocks.append(counts)
        if block + 1 < BLOCKS:
            new_tokens = []
            for i in range(n):
                neighbourhood = sorted((edge_token, tokens[j])
                                       for j, edge_token in adj[i])
                new_tokens.append(digest(("wl", tokens[i], tuple(neighbourhood))))
            tokens = new_tokens
    out[:BLOCKS * DIM] = np.concatenate(blocks)
    out[BLOCKS * DIM] = math.log1p(n)
    out[BLOCKS * DIM + 1] = math.log1p(int(fragment.bond_u.size))
    return out


def whole_fragment_features(atom_feat: np.ndarray, bond_u: np.ndarray,
                            bond_v: np.ndarray, bond_feat: np.ndarray,
                            atoms: Iterable[int], port: int,
                            cut: Sequence[int]) -> np.ndarray:
    return wl_features(induce(atom_feat, bond_u, bond_v, bond_feat,
                              atoms, port, cut))


def split_scalars(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return features[:BLOCKS * DIM], features[BLOCKS * DIM:]


def standardise(blocks: np.ndarray, scalars: np.ndarray, mean: np.ndarray,
                scale: np.ndarray) -> np.ndarray:
    return np.concatenate([blocks, (scalars - mean) / scale])


def weighted_scalar_stats(scalars: np.ndarray, weights: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit-pair-weighted mean/scale; each endpoint gets half of the pair weight."""
    w = np.asarray(weights, np.float64)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("empty weights")
    mean = (w[:, None] * scalars).sum(axis=0) / total
    var = (w[:, None] * (scalars - mean) ** 2).sum(axis=0) / total
    scale = np.sqrt(np.maximum(var, 0.0))
    degenerate = bool(np.any(scale <= 1e-12))
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale, {"degenerate_scale": degenerate,
                         "mean": mean.tolist(), "scale": scale.tolist(),
                         "total_weight": total}