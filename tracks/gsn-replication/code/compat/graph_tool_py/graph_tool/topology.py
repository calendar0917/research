"""graph_tool.topology 最小 shim（subgraph_isomorphism）。"""
from __future__ import annotations

from networkx.algorithms.isomorphism import GraphMatcher

from . import _VertexMap, _to_nx


def subgraph_isomorphism(sub, host, induced=False, subgraph=True,
                         generator=False, **kwargs):
    """与 graph-tool 等价语义的 pattern->host 单射映射枚举。

    induced=False -> 顶点单射 + pattern 边必须映射到 host 边（motif）
    induced=True  -> 同构子图（graphlet）
    返回 list 或 generator（计数为聚合，顺序无关）。
    """
    sub_nx, host_nx = _to_nx(sub), _to_nx(host)
    matcher = GraphMatcher(host_nx, sub_nx)
    it = (matcher.subgraph_isomorphisms_iter() if induced
          else matcher.subgraph_monomorphisms_iter())

    def gen():
        for m in it:
            yield _VertexMap(m)

    return gen() if generator else list(gen())
