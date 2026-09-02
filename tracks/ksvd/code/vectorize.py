from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

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



def canonical_adjacency_features(
    g: Graph,
    S: set[int],
    max_nodes: int,
    root: int | None = None,
) -> list[float]:
    """Exact canonical adjacency encoding for an unlabelled induced patch.

    Unlike the WL histogram, this retains the complete adjacency matrix up to
    graph isomorphism.  If ``root`` is supplied, isomorphisms must preserve the
    distinguished patch center, so occurrences with the same induced subgraph
    but structurally different centers remain distinguishable.

    The canonical permutation is computed by nauty through ``pynauty``.  The
    returned fixed-width vector contains the upper-triangular adjacency, a node
    mask, and (for rooted patches) the center position in canonical order.
    """
    try:
        import pynauty
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise ImportError(
            "canonical patch features require pynauty; install pynauty==2.8.8.1"
        ) from exc

    nodes = sorted(S)
    if len(nodes) > max_nodes:
        raise ValueError(f"patch has {len(nodes)} nodes but max_nodes={max_nodes}")
    if root is not None and root not in S:
        raise ValueError("root must belong to the patch")
    old_to_local = {u: i for i, u in enumerate(nodes)}
    adjacency = {
        old_to_local[u]: [old_to_local[v] for v in g.neighbors(u) if v in S]
        for u in nodes
    }
    coloring = []
    if root is not None:
        root_local = old_to_local[root]
        rest = set(range(len(nodes))) - {root_local}
        coloring = [{root_local}]
        if rest:
            coloring.append(rest)
    ng = pynauty.Graph(
        number_of_vertices=len(nodes),
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )
    # pynauty returns the original vertex occupying each canonical position.
    canonical_local = list(pynauty.canon_label(ng))
    canonical_nodes = [nodes[i] for i in canonical_local]
    position = {u: i for i, u in enumerate(canonical_nodes)}
    A = [[0.0] * max_nodes for _ in range(max_nodes)]
    for u in canonical_nodes:
        for v in g.neighbors(u):
            if v in position:
                i, j = position[u], position[v]
                A[i][j] = 1.0
    out = flatten_upper(A)
    out.extend([1.0] * len(nodes) + [0.0] * (max_nodes - len(nodes)))
    if root is not None:
        marker = [0.0] * max_nodes
        marker[position[root]] = 1.0
        out.extend(marker)
    return out

def wl_patch_features(
    g: Graph,
    S: set[int],
    max_nodes: int,
    n_bins: int = 64,
    n_iter: int = 3,
) -> list[float]:
    """Permutation-invariant fixed-width features for an induced patch.

    This is intentionally a lightweight WL-style representation for the
    KSVD feasibility test.  It keeps the KSVD interface (one fixed-length
    patch vector) while removing node-ID/BFS tie-breaking noise.
    """
    sub = g.induced(S)
    n = min(sub.n, max_nodes)
    e = sub.num_edges()
    # All patch nodes are retained by the sampler (n <= max_nodes).
    degs = [len(sub.neighbors(u)) for u in sub.nodes]
    out: list[float] = [0.0] * max_nodes
    for d in degs:
        out[min(int(d), max_nodes - 1)] += 1.0

    # Degree-pair edge histogram; invariant and more informative than size.
    pair_dim = max_nodes * (max_nodes + 1) // 2
    pair = [0.0] * pair_dim

    def pair_idx(a: int, b: int) -> int:
        a, b = sorted((min(a, max_nodes - 1), min(b, max_nodes - 1)))
        return a * max_nodes - (a * (a - 1)) // 2 + (b - a)
    for u, v in sub.edges():
        pair[pair_idx(len(sub.neighbors(u)), len(sub.neighbors(v)))] += 1.0
    out.extend(pair)

    # Histogram WL colors into fixed bins. Hashing only controls the output
    # width; the color refinement itself is independent of node ordering.
    colors = {u: len(sub.neighbors(u)) for u in sub.nodes}
    for it in range(n_iter):
        hist = [0.0] * n_bins
        for u in sub.nodes:
            token = (
                it,
                colors[u],
                tuple(sorted(colors[v] for v in sub.neighbors(u))),
            )
            h = hashlib.blake2b(repr(token).encode(), digest_size=8).digest()
            hist[int.from_bytes(h, 'little') % n_bins] += 1.0
        out.extend(hist)
        new_colors = {}
        for u in sub.nodes:
            token = (colors[u], tuple(sorted(colors[v] for v in sub.neighbors(u))))
            h = hashlib.blake2b(repr(token).encode(), digest_size=8).digest()
            new_colors[u] = int.from_bytes(h, 'little')
        colors = new_colors

    dens = (2.0 * e) / (n * (n - 1)) if n > 1 else 0.0
    max_edges = max(1, max_nodes * (max_nodes - 1) // 2)
    out.extend([float(n) / max_nodes, float(e) / max_edges, dens])
    return out



def _stable_bin(token: object, n_bins: int, namespace: str) -> int:
    payload = (namespace + "|" + repr(token)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_bins


def _feat_tuple(feat: np.ndarray | None, idx: int, missing: int = -1) -> tuple[int, ...]:
    if feat is None or idx < 0 or idx >= feat.shape[0]:
        return (missing,)
    row = np.asarray(feat[idx]).reshape(-1)
    return tuple(int(x) for x in row)


def _edge_feat_tuple(
    edge_feat: dict[tuple[int, int], np.ndarray] | None,
    u: int,
    v: int,
) -> tuple[int, ...]:
    if edge_feat is None:
        return (-1,)
    key = (u, v) if u < v else (v, u)
    row = edge_feat.get(key)
    if row is None:
        return (-1,)
    return tuple(int(x) for x in np.asarray(row).reshape(-1))


def labeled_wl_patch_features(
    g: Graph,
    S: set[int],
    max_nodes: int,
    node_feat: np.ndarray | None,
    edge_feat: dict[tuple[int, int], np.ndarray] | None,
    atom_bins: int = 64,
    bond_bins: int = 16,
    wl_bins: int = 64,
    n_iter: int = 3,
) -> list[float]:
    """Permutation-invariant topology + full OGB atom/bond labeled WL.

    Full categorical feature tuples define the WL colors/messages. Hashing is
    used only to obtain fixed-width histograms, not to impose a node order.
    """
    sub = g.induced(S)
    out = wl_patch_features(g, S, max_nodes)

    atom_hist = [0.0] * atom_bins
    atom_labels: dict[int, tuple[int, ...]] = {}
    for u in sub.nodes:
        label = _feat_tuple(node_feat, u)
        atom_labels[u] = label
        atom_hist[_stable_bin(label, atom_bins, "atom")] += 1.0
    out.extend(atom_hist)

    bond_hist = [0.0] * bond_bins
    for u, v in sub.edges():
        label = _edge_feat_tuple(edge_feat, u, v)
        bond_hist[_stable_bin(label, bond_bins, "bond")] += 1.0
    out.extend(bond_hist)

    colors = {
        u: _stable_bin(("initial", atom_labels[u]), 2**63 - 1, "wl-color")
        for u in sub.nodes
    }
    for it in range(n_iter):
        new_colors: dict[int, int] = {}
        hist = [0.0] * wl_bins
        for u in sub.nodes:
            messages = tuple(
                sorted(
                    (_edge_feat_tuple(edge_feat, u, v), colors[v])
                    for v in sub.neighbors(u)
                )
            )
            token = (colors[u], messages)
            color = _stable_bin(token, 2**63 - 1, f"wl-color-{it}")
            new_colors[u] = color
            hist[_stable_bin(color, wl_bins, f"wl-hist-{it}")] += 1.0
        out.extend(hist)
        colors = new_colors
    return out


def _component_count(g: Graph) -> int:
    unseen = set(g.nodes)
    count = 0
    while unseen:
        count += 1
        root = next(iter(unseen))
        unseen.remove(root)
        q = deque([root])
        while q:
            u = q.popleft()
            for v in g.neighbors(u):
                if v in unseen:
                    unseen.remove(v)
                    q.append(v)
    return count


def _shortest_path_without_edge(g: Graph, src: int, dst: int) -> int | None:
    """Shortest src-dst path length after deleting the undirected edge."""
    seen = {src}
    q = deque([(src, 0)])
    while q:
        u, dist = q.popleft()
        for v in g.neighbors(u):
            if (u == src and v == dst) or (u == dst and v == src):
                continue
            if v == dst:
                return dist + 1
            if v not in seen:
                seen.add(v)
                q.append((v, dist + 1))
    return None


def ring_patch_features(
    g: Graph,
    S: set[int],
    max_nodes: int,
    node_feat: np.ndarray | None,
    edge_feat: dict[tuple[int, int], np.ndarray] | None,
) -> list[float]:
    """True induced-cycle and OGB aromatic/ring statistics for a small patch.

    Cycle membership does not rely on triangles: an edge is cyclic iff its
    endpoints remain connected after removing it. Shortest cycle lengths are
    collected per cyclic edge in bins 3..8 plus overflow.
    """
    sub = g.induced(S)
    n = sub.n
    edges = sub.edges()
    e = len(edges)
    components = _component_count(sub) if n else 0
    cycle_rank = max(0, e - n + components)
    max_edges = max(1, max_nodes * (max_nodes - 1) // 2)

    cycle_hist = [0.0] * 7  # lengths 3,4,5,6,7,8,>8
    cyclic_edges = 0
    aromatic_edges = 0
    for u, v in edges:
        bond = _edge_feat_tuple(edge_feat, u, v)
        if bond and bond[0] == 3:  # OGB bond type 3 = aromatic
            aromatic_edges += 1
        alt_dist = _shortest_path_without_edge(sub, u, v)
        if alt_dist is None:
            continue
        cyclic_edges += 1
        cycle_len = alt_dist + 1
        idx = cycle_len - 3 if 3 <= cycle_len <= 8 else 6
        cycle_hist[idx] += 1.0 / max_edges

    aromatic_atoms = 0
    ring_atoms = 0
    if node_feat is not None:
        for u in sub.nodes:
            row = np.asarray(node_feat[u]).reshape(-1)
            aromatic_atoms += int(row.size > 7 and int(row[7]) == 1)
            ring_atoms += int(row.size > 8 and int(row[8]) == 1)

    return [
        float(cycle_rank) / max(1, max_nodes),
        float(cyclic_edges) / max_edges,
        float(cyclic_edges) / max(1, e),
        *cycle_hist,
        float(aromatic_edges) / max(1, e),
        float(aromatic_atoms) / max(1, n),
        float(ring_atoms) / max(1, n),
    ]


def labeled_wl_ring_patch_features(
    g: Graph,
    S: set[int],
    max_nodes: int,
    node_feat: np.ndarray | None,
    edge_feat: dict[tuple[int, int], np.ndarray] | None,
) -> list[float]:
    return labeled_wl_patch_features(
        g, S, max_nodes, node_feat=node_feat, edge_feat=edge_feat
    ) + ring_patch_features(
        g, S, max_nodes, node_feat=node_feat, edge_feat=edge_feat
    )


def patch_feature_dim(max_nodes: int, patch_feat: str) -> int:
    topo_dim = max_nodes * (max_nodes - 1) // 2
    wl_dim = max_nodes + max_nodes * (max_nodes + 1) // 2 + 64 * 3 + 3
    dims = {
        "topo": topo_dim,
        "chem": topo_dim + 16 + 8,
        "wl": wl_dim,
        "wl_chem": wl_dim + 64 + 16 + 64 * 3,
        "wl_chem_ring": wl_dim + 64 + 16 + 64 * 3 + 13,
    }
    if patch_feat not in dims:
        raise ValueError(f"unknown patch_feat={patch_feat!r}; expected one of {sorted(dims)}")
    return dims[patch_feat]


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
