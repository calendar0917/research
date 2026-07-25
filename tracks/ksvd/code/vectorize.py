from __future__ import annotations

from collections import deque

from .graph import Graph


def order_nodes_degree(g: Graph, S: set[int]) -> list[int]:
    return sorted(S, key=lambda u: (-len(g.neighbors(u)), u))


def order_nodes_bfs(g: Graph, S: set[int]) -> list[int]:
    """BFS from max-degree node in S; ties by id. Better local geometry for motifs."""
    if not S:
        return []
    sub_nbr = {u: [v for v in g.neighbors(u) if v in S] for u in S}
    root = max(S, key=lambda u: (len(sub_nbr[u]), -u))
    order: list[int] = []
    seen = {root}
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(sub_nbr[u], key=lambda x: (-len(sub_nbr[x]), x)):
            if v not in seen:
                seen.add(v)
                q.append(v)
    # disconnected pieces
    for u in sorted(S, key=lambda x: (-len(sub_nbr[x]), x)):
        if u not in seen:
            seen.add(u)
            order.append(u)
    return order


def order_nodes(g: Graph, S: set[int], mode: str = "bfs") -> list[int]:
    if mode == "degree":
        return order_nodes_degree(g, S)
    return order_nodes_bfs(g, S)


def adjacency_padded(
    g: Graph,
    S: set[int],
    m: int,
    order_mode: str = "bfs",
) -> list[list[float]]:
    order = order_nodes(g, S, mode=order_mode)[:m]
    idx = {u: i for i, u in enumerate(order)}
    A = [[0.0] * m for _ in range(m)]
    sub = g.induced(set(order))
    for u, v in sub.edges():
        i, j = idx[u], idx[v]
        A[i][j] = 1.0
        A[j][i] = 1.0
    return A


def flatten_upper(A: list[list[float]]) -> list[float]:
    m = len(A)
    out: list[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            out.append(A[i][j])
    return out


def subgraph_extra_features(g: Graph, S: set[int]) -> list[float]:
    """Hand features on induced G[S]: density, triangle count, max degree — local oracle signal."""
    sub = g.induced(S)
    n = sub.n
    e = sub.num_edges()
    dens = (2.0 * e) / (n * (n - 1)) if n > 1 else 0.0
    tri = 0
    nodes = sub.nodes
    for i, u in enumerate(nodes):
        nbrs = sub.neighbors(u)
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            if v not in nbrs:
                continue
            # common neighbors
            tri += len(nbrs & sub.neighbors(v))
    tri //= 3  # each triangle counted 3 times? wait: for each edge uv count common — each tri 3 times
    # Actually for each ordered pair of edges from counting: for each u, pairs of neighbors that connect:
    tri2 = 0
    for u in nodes:
        nbrs = list(sub.neighbors(u))
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                if sub.has_edge(nbrs[a], nbrs[b]):
                    tri2 += 1
    tri2 //= 3
    max_deg = max((len(sub.neighbors(u)) for u in nodes), default=0)
    return [float(n), float(e), dens, float(tri2), float(max_deg)]
