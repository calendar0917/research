"""Exact canonical form for small attributed graphs (deterministic, no hashing).

Round: ``structure-context-support-v1`` (see ``../PROTOCOL.md``).

Why not a WL hash
-----------------
``cscl_units.py`` uses a 4-round labelled-WL *hash* for unit signatures and
documents that distinct non-isomorphic units may collide.  This round must not
merge two different local structures into one "shared identity", so the
identity key has to be an **exact** canonical form:

* invariant: two isomorphic attributed (and anchored) graphs produce the same
  key, whatever the atom order;
* injective: two graphs with the same key are isomorphic as attributed /
  anchored graphs (no false merges).

Algorithm (deterministic, no RNG, no external canonicalisation library)
----------------------------------------------------------------------
1. colour refinement (1-WL) on the attributed graph, where the colour of a
   vertex is recursively ``(colour, sorted multiset of (edge-attr, colour))``;
   colours are *canonically numbered* (sorted by ``repr``) at every round, so
   the refinement itself is isomorphism-equivariant;
2. if the colouring is discrete, read off the canonical certificate;
   otherwise individualise the first non-singleton class, recurse over **all**
   its members (complete enumeration of the search tree) and keep the
   lexicographically smallest certificate.

Every leaf of that tree is a discrete colouring, and the certificate records
the full attributed adjacency in the order induced by the colouring, so a
certificate determines the attributed graph up to isomorphism.  The minimum
over all leaves is therefore an exact canonical form.  Graphs in this round are
small (bond-anchored radius-1 patches: <= 8 patch atoms, extended radius-2
neighbourhoods: <= 20 atoms), so the complete tree is cheap.
"""

from __future__ import annotations

from typing import Hashable, Sequence

__all__ = ["canonical_form", "validate_graph"]

NodeAttr = Hashable
EdgeAttr = Hashable


def validate_graph(
    n: int,
    attrs: Sequence[NodeAttr],
    edges: Sequence[tuple[int, int, EdgeAttr]],
    anchors: Sequence[int] = (),
) -> None:
    if len(attrs) != n:
        raise ValueError(f"attrs length {len(attrs)} != n {n}")
    seen: set[tuple[int, int]] = set()
    for u, v, _ea in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) out of range 0..{n - 1}")
        if u == v:
            raise ValueError(f"self loop at {u}")
        key = (u, v) if u < v else (v, u)
        if key in seen:
            raise ValueError(f"duplicate undirected edge {key}")
        seen.add(key)
    for a in anchors:
        if not 0 <= a < n:
            raise ValueError(f"anchor {a} out of range")


def _rank(signatures: Sequence[tuple]) -> list[int]:
    """Canonical colour numbering: identical signatures -> identical colour id."""
    uniq = sorted(set(signatures), key=repr)
    index = {sig: i for i, sig in enumerate(uniq)}
    return [index[sig] for sig in signatures]


def _refine(colors: list[int], adj: list[list[tuple[int, EdgeAttr]]]) -> list[int]:
    """1-WL colour refinement with canonical colour numbering.

    Terminates on a *partition* stability test (each colour class holds a single
    signature), never on integer-vector equality: ``_rank`` orders colours by
    ``repr``, so an integer-vector comparison can oscillate (e.g. ``10`` sorts
    before ``3`` in string order).  Every returned numbering is a ``_rank``
    output, hence canonical for the signature multiset.
    """
    n = len(colors)
    while True:
        sigs = [
            (colors[i], tuple(sorted(((ea, colors[j]) for j, ea in adj[i]), key=repr)))
            for i in range(n)
        ]
        first: dict[int, tuple] = {}
        stable = True
        for i in range(n):
            seen = first.setdefault(colors[i], sigs[i])
            if seen != sigs[i]:
                stable = False
                break
        if stable:
            return colors
        colors = _rank(sigs)


def _certificate(
    colors: list[int],
    attrs: Sequence[NodeAttr],
    adj: list[list[tuple[int, EdgeAttr]]],
    anchor_set: frozenset[int],
) -> tuple[str, tuple[int, ...]]:
    order = tuple(sorted(range(len(colors)), key=lambda i: colors[i]))
    pos = {v: k for k, v in enumerate(order)}
    parts = []
    for v in order:
        nb = sorted(((pos[j], ea) for j, ea in adj[v]), key=lambda t: (t[0], repr(t[1])))
        nbs = ",".join(f"{ea!r}@{p}" for p, ea in nb)
        parts.append(f"{attrs[v]!r}a{int(v in anchor_set)}:[{nbs}]")
    return "|".join(parts), order


def _search(
    colors: list[int],
    attrs: Sequence[NodeAttr],
    adj: list[list[tuple[int, EdgeAttr]]],
    anchor_set: frozenset[int],
) -> tuple[str, tuple[int, ...]]:
    colors = _refine(colors, adj)
    classes: dict[int, list[int]] = {}
    for v, c in enumerate(colors):
        classes.setdefault(c, []).append(v)
    multi = [c for c, members in classes.items() if len(members) > 1]
    if not multi:
        return _certificate(colors, attrs, adj, anchor_set)
    target = min(multi)
    best: tuple[str, tuple[int, ...]] | None = None
    bump = max(colors) + 1
    for v in classes[target]:
        branch = list(colors)
        branch[v] = bump
        cand = _search(branch, attrs, adj, anchor_set)
        if best is None or cand[0] < best[0]:
            best = cand
    assert best is not None
    return best


def canonical_form(
    n: int,
    attrs: Sequence[NodeAttr],
    edges: Sequence[tuple[int, int, EdgeAttr]],
    anchors: Sequence[int] = (),
    return_order: bool = False,
) -> str | tuple[str, tuple[int, ...]]:
    """Exact canonical key of an attributed undirected graph.

    ``attrs[i]`` is the atom attribute of vertex ``i``; ``edges`` are
    undirected *and must be given with u < v only for a physical bond* (no
    duplicate, no self loop); ``anchors`` is an order-insensitive set of
    distinguished vertices (the two bond endpoints).
    """
    validate_graph(n, attrs, edges, anchors)
    adj: list[list[tuple[int, EdgeAttr]]] = [[] for _ in range(n)]
    for u, v, ea in edges:
        adj[u].append((v, ea))
        adj[v].append((u, ea))
    anchor_set = frozenset(int(a) for a in anchors)
    init = _rank([(attrs[i], 1 if i in anchor_set else 0) for i in range(n)])
    cert, order = _search(init, attrs, adj, anchor_set)
    return (cert, order) if return_order else cert
