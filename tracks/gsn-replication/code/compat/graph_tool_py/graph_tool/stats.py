"""graph_tool.stats 最小 shim。"""
from __future__ import annotations

from . import Graph


def _dedup(g: Graph, directed: bool) -> None:
    seen = set()
    out = []
    for u, v in g._edges:
        if u == v:
            continue
        key = (u, v) if directed else (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        out.append((u, v))
    g._edges = out


def remove_self_loops(g: Graph) -> None:
    g._edges = [(u, v) for u, v in g._edges if u != v]


def remove_parallel_edges(g: Graph) -> None:
    _dedup(g, g.directed)
