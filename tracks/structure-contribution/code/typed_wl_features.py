"""Typed-WL / Morgan-style structural count features (label-blind, non-learned).

Protocol: ``configs/strong_representation_readout_screen_v1.yaml``
Round: ``cscl-strong-representation-readout-screen-v1`` — Arm A.

Definition (frozen in the protocol)
-----------------------------------
For every atom ``v`` of the raw attributed molecular graph (PyG ZINC
``x`` = atom category, unique undirected chemical bonds from
``edge_index`` + ``edge_attr`` = bond category):

* ``color_0(v) = atom_type(v)``
* ``color_{r+1}(v) = stable_hash(color_r(v),
  sorted_multiset{(bond_type(v,u), color_r(u))})``

for the fixed radii ``r = 0, 1, 2, 3``.  For every radius the structural
signature ``color_r(v)`` of every rooted fragment occurrence is mapped to a
fixed-width count vector with ``1024`` buckets:

* bucket = ``sha256-derived 64-bit integer("bucket-v1|r|color") mod 1024``;
* the radius is part of the hash input, so radii never share a hash space;
* one count per rooted fragment occurrence (counts, not binary);
* ``X_WL in R^(N x 4096)``.

The hash is SHA256 only — Python's built-in ``hash()`` is never used.  There
is no learned vocabulary, no fitted statistic and no dependence on any label:
the descriptor of a molecule is a pure function of its attributed graph.

This is called **Typed-WL / Morgan-style structural count**, not "the standard
RDKit Morgan fingerprint": the atomic invariants and the bucket assignment are
not RDKit's ECFP definitions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

#: frozen protocol constants
WL_RADII: tuple[int, ...] = (0, 1, 2, 3)
BUCKETS_PER_RADIUS: int = 1024
WL_DIM: int = BUCKETS_PER_RADIUS * len(WL_RADII)  # 4096

HASH_ALGORITHM: str = "sha256"
#: bumped whenever the colour update or bucket rule changes
HASH_VERSION: str = "typed-wl-v1"
BUCKET_SALT: str = "bucket-v1"
COLOR_SALT: str = "color-v1"


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def unique_undirected_bonds(
    edge_index: np.ndarray, edge_attr: np.ndarray
) -> list[tuple[int, int, int]]:
    """Canonicalize PyG-style directed edge lists to unique undirected bonds.

    PyG ZINC stores every physical bond twice, once per direction, with the
    same bond category.  Both directions must agree, else ``ValueError``.
    Self-loops raise (ZINC has none).  Returns ``(lo, hi, bond_type)`` sorted
    by ``(lo, hi)``.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    edge_attr = np.asarray(edge_attr, dtype=np.int64).reshape(-1)
    counts: dict[tuple[int, int], list[int]] = {}
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u == v:
            raise ValueError(f"self-loop at atom {u}: not a physical bond")
        if v < u:
            u, v = v, u
        counts.setdefault((u, v), []).append(int(edge_attr[k]))
    bonds: list[tuple[int, int, int]] = []
    for (u, v), bts in sorted(counts.items()):
        if len(bts) > 2:
            raise ValueError(f"bond ({u},{v}) occurs {len(bts)} times: parallel multi-edge unsupported")
        if len(bts) == 2 and bts[0] != bts[1]:
            raise ValueError(f"inconsistent bond type across directions for ({u},{v}): {bts}")
        bonds.append((u, v, bts[0]))
    return bonds


def stable_sha256_int(payload: str) -> int:
    """SHA256-derived non-negative 64-bit integer (never Python ``hash()``)."""
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def bucket_of(radius: int, color: str) -> int:
    """Fixed-width bucket for a rooted fragment signature (radius is salted)."""
    return stable_sha256_int(f"{BUCKET_SALT}|{int(radius)}|{color}") % BUCKETS_PER_RADIUS


def color_round(radius: int, parent_colors: Sequence[str], adj: Sequence[Sequence[tuple[int, int]]]) -> list[str]:
    """One WL refinement round: ``(own colour, sorted multiset (bond, colour))``.

    ``adj[v] = [(bond_type, neighbour_index), ...]`` for the current molecule.
    """
    out: list[str] = []
    for v in range(len(parent_colors)):
        multiset = sorted(f"{bt:02d}:{parent_colors[j]}" for bt, j in adj[v])
        payload = f"{COLOR_SALT}|r={int(radius)}|{parent_colors[v]}|{'|'.join(multiset)}"
        out.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return out


def colors_per_radius(
    atom_types: Sequence[int], bonds: Sequence[tuple[int, int, int]]
) -> list[list[str]]:
    """Return ``colors[r][v]`` for every frozen radius (0..3)."""
    n = len(atom_types)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, bt in bonds:
        adj[u].append((int(bt), int(v)))
        adj[v].append((int(bt), int(u)))
    colors = [f"a{int(t)}" for t in atom_types]
    out: list[list[str]] = [list(colors)]
    for radius in WL_RADII[1:]:
        colors = color_round(radius, colors, adj)
        out.append(colors)
    return out


def wl_descriptor(atom_types: Sequence[int], bonds: Sequence[tuple[int, int, int]]) -> np.ndarray:
    """``X_WL`` row: per-radius 1024-bucket rooted-fragment counts."""
    if len(WL_RADII) != 4 or BUCKETS_PER_RADIUS != 1024:
        raise AssertionError("frozen Typed-WL dimensions changed")
    vec = np.zeros(WL_DIM, dtype=np.float32)
    rounds = colors_per_radius(atom_types, bonds)
    for r, colors in enumerate(rounds):
        buckets = np.asarray([bucket_of(WL_RADII[r], c) for c in colors], dtype=np.int64)
        np.add.at(vec, r * BUCKETS_PER_RADIUS + buckets, 1.0)
    return vec


# ---------------------------------------------------------------------------
# molecule adapters
# ---------------------------------------------------------------------------


def molecule_typed_wl_from_arrays(
    x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray
) -> np.ndarray:
    """Raw PyG attribute arrays -> Typed-WL row."""
    atom_types = np.asarray(x, dtype=np.int64).reshape(-1)
    bonds = unique_undirected_bonds(edge_index, edge_attr)
    return wl_descriptor(atom_types, bonds)


def molecule_typed_wl(data: Any) -> np.ndarray:
    """PyG ``Data`` -> Typed-WL row (only ``x`` / edges are read)."""
    x = data.x.detach().cpu().numpy() if hasattr(data.x, "detach") else np.asarray(data.x)
    edge_index = (
        data.edge_index.detach().cpu().numpy()
        if hasattr(data.edge_index, "detach")
        else np.asarray(data.edge_index)
    )
    edge_attr = (
        data.edge_attr.detach().cpu().numpy()
        if hasattr(data.edge_attr, "detach")
        else np.asarray(data.edge_attr)
    )
    return molecule_typed_wl_from_arrays(x, edge_index, edge_attr)


# ---------------------------------------------------------------------------
# label-free global structure (30) — historical verified definition
# ---------------------------------------------------------------------------


def global_structure_raw(graph) -> np.ndarray:
    """Raw ``structure_short(15) + structure_long(15)`` of a networkx graph.

    Uses the historical verified definition
    (``zinc_long_range_proxy._global_structure_blocks``); all inputs are graph
    invariants (degrees, density, triangles, clustering, all-pairs distances,
    eccentricities).  No labels, no fitted statistics.
    """
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _global_structure_blocks

    short, long = _global_structure_blocks(graph)
    return np.concatenate([short, long]).astype(np.float32)


def graph_from_data(data: Any):
    """PyG ``Data`` -> networkx graph (deduplicated undirected bonds)."""
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _data_to_graph

    graph, _node_types, _edge_types = _data_to_graph(data)
    return graph


# ---------------------------------------------------------------------------
# diagnostics + self-checks
# ---------------------------------------------------------------------------


def bucket_collision_stats(color_rows: Iterable[Sequence[Sequence[str]]]) -> dict[str, Any]:
    """Distinct signatures vs occupied buckets per radius (diagnostic only).

    ``color_rows`` is an iterable of ``colors_per_radius`` outputs: one entry
    per molecule, each a list over radii of per-atom colour lists.  A collision
    means two distinct rooted-fragment signatures land in the same fixed
    bucket.  This count is reported, never used to change the frozen
    1024-bucket width this round.
    """
    distinct = [set() for _ in WL_RADII]
    occupied = [set() for _ in WL_RADII]
    for colors in color_rows:
        for r, color_list in enumerate(colors):
            for color in color_list:
                distinct[r].add(color)
                occupied[r].add(bucket_of(WL_RADII[r], color))
    out: dict[str, Any] = {}
    for r, radius in enumerate(WL_RADII):
        n_sig = len(distinct[r])
        n_occ = len(occupied[r])
        out[f"r{radius}"] = {
            "distinct_signatures": n_sig,
            "occupied_buckets": n_occ,
            "collision_pairs": max(n_sig - n_occ, 0),
            "collision_ratio": (1.0 - n_occ / n_sig) if n_sig else 0.0,
        }
    return out


def descriptor_nonzero_and_conservation(X: np.ndarray, n_atoms: np.ndarray) -> dict[str, Any]:
    """Nonzero ratio per radius block + per-molecule count conservation."""
    X = np.asarray(X)
    n = X.shape[0]
    out: dict[str, Any] = {}
    for r, radius in enumerate(WL_RADII):
        block = X[:, r * BUCKETS_PER_RADIUS : (r + 1) * BUCKETS_PER_RADIUS]
        out[f"r{radius}"] = {
            "nonzero_ratio": float((block > 0).mean()),
            "mean_count_per_molecule": float(block.sum(axis=1).mean()),
        }
    sums = X.reshape(n, len(WL_RADII), BUCKETS_PER_RADIUS).sum(axis=2)
    conserve = np.allclose(sums, n_atoms[:, None].astype(np.float64), atol=1e-6)
    out["count_conservation"] = bool(conserve)
    out["count_conservation_max_abs_dev"] = float(np.abs(sums - n_atoms[:, None]).max())
    return out


__all__ = [
    "WL_RADII",
    "BUCKETS_PER_RADIUS",
    "WL_DIM",
    "HASH_ALGORITHM",
    "HASH_VERSION",
    "unique_undirected_bonds",
    "stable_sha256_int",
    "bucket_of",
    "color_round",
    "colors_per_radius",
    "wl_descriptor",
    "molecule_typed_wl_from_arrays",
    "molecule_typed_wl",
    "global_structure_raw",
    "graph_from_data",
    "bucket_collision_stats",
    "descriptor_nonzero_and_conservation",
]
