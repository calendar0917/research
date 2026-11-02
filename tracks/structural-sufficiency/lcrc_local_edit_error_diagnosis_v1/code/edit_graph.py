"""Labelled graphs, Weisfeiler-Lehman candidate keys and *explicit* isomorphism.

The protocol allows WL hashing only to *generate candidates*; the identity of a
matched core is decided by an explicit labelled-graph isomorphism that also
carries the attachment-point marker.  Nothing in this module is a statistical
procedure - it is exact combinatorics on small graphs (<= ~120 atoms).

Node label:   ``(element, is_aromatic, formal_charge, total_h, is_attachment)``
Edge label:   ``1`` SINGLE, ``2`` DOUBLE, ``3`` TRIPLE, ``4`` AROMATIC, ``5`` OTHER
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Iterable, Sequence

NodeLabel = tuple[str, bool, int, int, bool]
EdgeLabel = int


class LGraph:
    """Small labelled graph with a fixed node order."""

    __slots__ = ("labels", "edges", "adj", "canonical", "_wl")

    def __init__(self, labels: Sequence[NodeLabel], edges: Iterable[tuple[int, int, EdgeLabel]]):
        self.labels = tuple(labels)
        n = len(self.labels)
        edge_list = []
        adj: list[list[tuple[int, EdgeLabel]]] = [[] for _ in range(n)]
        seen = set()
        for u, v, lab in edges:
            u, v = int(u), int(v)
            if u == v:
                raise ValueError("self loop")
            key = (min(u, v), max(u, v), int(lab))
            if key in seen:
                raise ValueError(f"duplicate edge {key}")
            seen.add(key)
            edge_list.append(key)
            adj[u].append((v, int(lab)))
            adj[v].append((u, int(lab)))
        for a in adj:
            a.sort()
        self.edges = tuple(sorted(edge_list))
        self.adj = tuple(tuple(a) for a in adj)
        self.canonical = None
        self._wl = None

    # -- basics ------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.labels)

    def degree(self, v: int) -> int:
        return len(self.adj[v])

    def label_str(self, v: int) -> str:
        e, arom, chg, nh, att = self.labels[v]
        return f"{e}|{int(arom)}|{int(chg)}|{int(nh)}|{int(att)}"

    def edge_key(self, u: int, v: int) -> int | None:
        for w, lab in self.adj[u]:
            if w == v:
                return lab
        return None

    # -- WL candidate key --------------------------------------------------
    def wl_signatures(self, rounds: int = 8) -> tuple[str, ...]:
        """Iterative neighbourhood refinement as graph-independent signature strings.

        Because the signature of a node is built from its own label string and
        the sorted ``(edge label, neighbour signature)`` pairs, the *strings*
        are comparable across graphs (unlike dense colour ids).
        """
        sigs = [self.label_str(v) for v in range(self.n)]
        for _ in range(int(rounds)):
            new = []
            for v in range(self.n):
                nb = sorted((int(lab), sigs[w]) for w, lab in self.adj[v])
                new.append(sigs[v] + "<" + ";".join(f"{l}:{s}" for l, s in nb) + ">")
            if len(set(new)) == len(set(sigs)) and sorted(new) == sorted(sigs):
                sigs = new
                break
            sigs = new
        return tuple(sigs)

    def wl_hash(self) -> str:
        if self._wl is None:
            hist: dict[str, int] = {}
            for s in self.wl_signatures():
                hist[s] = hist.get(s, 0) + 1
            payload = "\n".join(f"{hist[k]}\t{k}" for k in sorted(hist))
            self._wl = hashlib.sha256(f"{self.n}\n{payload}".encode()).hexdigest()
        return self._wl

    # -- explicit isomorphism ---------------------------------------------
    def find_isomorphisms(self, other: "LGraph", cap: int = 4096) -> tuple[list[dict[int, int]], bool]:
        """All label-preserving isomorphisms ``self -> other`` (capped).

        Returns ``(mappings, truncated)``.  A mapping is ``{i_self: j_other}``.
        """
        if self.n != other.n or len(self.edges) != len(other.edges):
            return [], False
        sig_a = self.wl_signatures()
        sig_b = other.wl_signatures()
        if sorted(sig_a) != sorted(sig_b):
            return [], False
        if sorted(self.labels) != sorted(other.labels):
            return [], False
        deg_a = [self.degree(v) for v in range(self.n)]
        deg_b = [other.degree(v) for v in range(other.n)]
        cand = []
        for v in range(self.n):
            pool = [
                w
                for w in range(other.n)
                if sig_b[w] == sig_a[v] and self.labels[v] == other.labels[w] and deg_b[w] == deg_a[v]
            ]
            if not pool:
                return [], False
            cand.append(pool)
        order = sorted(range(self.n), key=lambda v: (len(cand[v]), -deg_a[v], v))

        results: list[dict[int, int]] = []
        used = [False] * other.n
        mapping: dict[int, int] = {}

        def backtrack(idx: int) -> None:
            if len(results) >= cap:
                return
            if idx == len(order):
                results.append(dict(mapping))
                return
            v = order[idx]
            for w in cand[v]:
                if used[w]:
                    continue
                ok = True
                for u, lab in self.adj[v]:
                    if u in mapping:
                        wu = mapping[u]
                        if other.edge_key(w, wu) != lab:
                            ok = False
                            break
                if not ok:
                    continue
                # unmatched neighbours must still have enough compatible partners
                for u, lab in self.adj[v]:
                    if u in mapping or u == v:
                        continue
                    need = [
                        x
                        for x in cand[u]
                        if not used[x] and x != w and other.edge_key(w, x) == lab
                    ]
                    if not need:
                        ok = False
                        break
                if not ok:
                    continue
                mapping[v] = w
                used[w] = True
                backtrack(idx + 1)
                used[w] = False
                del mapping[v]
                if len(results) >= cap:
                    return

        backtrack(0)
        return results, len(results) >= cap

    def is_isomorphic(self, other: "LGraph") -> bool:
        maps, _ = self.find_isomorphisms(other, cap=1)
        return bool(maps)

    # -- automorphisms -----------------------------------------------------
    def automorphisms(self, cap: int = 4096) -> tuple[list[dict[int, int]], bool]:
        return self.find_isomorphisms(self, cap=cap)

    def orbits(self, cap: int = 4096) -> tuple[dict[int, int], int, bool]:
        """Automorphism orbits: ``{node: orbit_representative}, n_automorphisms, truncated``."""
        autos, truncated = self.automorphisms(cap=cap)
        parent = list(range(self.n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for auto in autos:
            for i, j in auto.items():
                union(i, j)
        return {i: find(i) for i in range(self.n)}, len(autos), truncated

    # -- canonical form (complete, not a hash) ----------------------------
    def canonical_form(self, cap_perms: int = 6_000_000) -> str:
        """Deterministic canonical string via exhaustive minimal adjacency code.

        Only used for tie-breaking and for equality of small fragments
        (n <= 5) and of cores after they have already been proven isomorphic.
        For larger graphs an invariant refinement order is used first, which
        still yields a deterministic string for a *given* node order but is not
        guaranteed canonical; equality is therefore always confirmed by
        :meth:`is_isomorphic`.
        """
        if self.canonical is not None:
            return self.canonical
        n = self.n
        labels = [self.label_str(v) for v in range(n)]
        if n <= 5:
            best = None
            for perm in itertools.permutations(range(n)):
                code = _code(self, perm, labels)
                if best is None or code < best:
                    best = code
            self.canonical = best or ""
            return self.canonical
        order = _refinement_order(self)
        self.canonical = _code(self, order, labels)
        return self.canonical


def _code(graph: LGraph, order: Sequence[int], labels: Sequence[str]) -> str:
    pos = {v: i for i, v in enumerate(order)}
    parts = [",".join(labels[v] for v in order)]
    edges = sorted((min(pos[u], pos[v]), max(pos[u], pos[v]), int(lab)) for u, v, lab in graph.edges)
    parts.append(";".join(f"{a}-{b}:{lab}" for a, b, lab in edges))
    return "#".join(parts)


def _refinement_order(graph: LGraph) -> list[int]:
    """Deterministic order from (signature, degree, label, index)."""
    sigs = graph.wl_signatures()
    return sorted(range(graph.n), key=lambda v: (sigs[v], -graph.degree(v), graph.label_str(v), v))


def isomorphism_classify(graphs: Sequence[LGraph], n_isomorphism_checks: list[int] | None = None) -> list[int]:
    """Partition graphs into exact isomorphism classes (representative index)."""
    classes: list[int] = []
    reps: list[LGraph] = []
    for idx, g in enumerate(graphs):
        placed = False
        for rep_idx, rep in enumerate(reps):
            maps, _ = g.find_isomorphisms(rep, cap=1)
            if n_isomorphism_checks is not None:
                n_isomorphism_checks[0] += 1
            if maps:
                classes.append(rep_idx)
                placed = True
                break
        if not placed:
            reps.append(g)
            classes.append(len(reps) - 1)
    return classes


__all__ = ["LGraph", "NodeLabel", "EdgeLabel", "isomorphism_classify"]
