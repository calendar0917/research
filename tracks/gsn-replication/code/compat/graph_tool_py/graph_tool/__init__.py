"""本地冒烟专用：graph_tool 最小兼容 shim（networkx 实现）。

用途：在**没有 conda graph-tool 的机器**上验证官方 GSN 管线（数据加载→结构计数→
编码→模型→训练）的现代栈兼容性。只覆盖官方代码用到的 API 子集：
  gt.Graph / gt.stats.remove_self_loops / gt.stats.remove_parallel_edges /
  gt.topology.subgraph_isomorphism(...)

语义对齐 graph-tool：
  - induced=False -> 顶点单射 + pattern 边必须映射到 host 边（motif 计数，
    等价 networkx subgraph_monomorphisms_iter）
  - induced=True  -> 同构子图（graphlet 计数，等价 subgraph_isomorphisms_iter）
  - 映射对象：可迭代（按 pattern 顶点序给出 host 顶点）+ .get_array()

注意：仅用于管线冒烟；**服务器正式实验用 conda-forge 的真 graph-tool**。
计数语义对 clique 家族 induced 与否等价（K_k 无额外边可加），因此本文配置
(complete_graph) 下 shim 与 graph-tool 结果一致。
"""
from __future__ import annotations

import numpy as np


class Graph:
    """最小 graph-tool Graph：无向/有向边列表容器。"""

    def __init__(self, directed: bool = False):
        self.directed = directed
        self._edges: list[tuple[int, int]] = []

    def add_edge_list(self, edge_list) -> None:
        for u, v in edge_list:
            self._edges.append((int(u), int(v)))

    def num_vertices(self) -> int:
        return max((max(e) for e in self._edges), default=-1) + 1

    def get_vertices(self):
        return range(self.num_vertices())

    def get_edges(self):
        return self._edges


class _VertexMap:
    """graph-tool VertexMap 替身：iter 按 pattern 顶点序，get_array() 同理。"""

    def __init__(self, mapping: dict):
        self._mapping = mapping

    def __iter__(self):
        return iter(self._mapping[i] for i in sorted(self._mapping))

    def get_array(self) -> np.ndarray:
        return np.array(list(self), dtype=np.int64)

    def __len__(self):
        return len(self._mapping)


def _to_nx(g: Graph):
    import networkx as nx

    nxg = nx.Graph() if not g.directed else nx.DiGraph()
    nxg.add_edges_from(g._edges)
    for v in g.get_vertices():
        nxg.add_node(v)
    return nxg


from . import stats as stats  # noqa: E402  (gt.stats.* 可用)
from . import topology as topology  # noqa: E402
