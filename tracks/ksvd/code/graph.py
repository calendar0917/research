from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Graph:
    """Undirected simple graph; nodes are ints 0..n-1 or arbitrary ints."""

    adj: dict[int, set[int]]

    @property
    def nodes(self) -> list[int]:
        return sorted(self.adj.keys())

    @property
    def n(self) -> int:
        return len(self.adj)

    def edges(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for u, nbrs in self.adj.items():
            for v in nbrs:
                if u < v:
                    out.append((u, v))
        return out

    def num_edges(self) -> int:
        return len(self.edges())

    def neighbors(self, u: int) -> set[int]:
        return self.adj.get(u, set())

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.adj.get(u, set())

    def induced(self, S: set[int]) -> Graph:
        adj: dict[int, set[int]] = {u: set() for u in S}
        for u in S:
            for v in self.neighbors(u):
                if v in S:
                    adj[u].add(v)
        return Graph(adj)

    def edge_key(self, u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)


def from_edges(n: int, edges: list[tuple[int, int]]) -> Graph:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return Graph(adj)


def ring_chords(n: int, chords: list[list[int]] | list[tuple[int, int]]) -> Graph:
    edges = [(i, (i + 1) % n) for i in range(n)]
    for c in chords:
        u, v = int(c[0]), int(c[1])
        edges.append((u, v))
    return from_edges(n, edges)


def star_graph(hub: int, leaves: list[int]) -> Graph:
    nodes = [hub] + leaves
    nmap = {x: i for i, x in enumerate(nodes)}
    edges = [(nmap[hub], nmap[L]) for L in leaves]
    return from_edges(len(nodes), edges)
