from __future__ import annotations

import numpy as np

from .graph import Graph, from_edges


def make_ring_vs_path(n_per_class: int = 40, n_nodes: int = 12, seed: int = 0):
    """Legacy: cycle vs path — degree/edge count separates perfectly."""
    rng = np.random.default_rng(seed)
    graphs = []
    labels = []
    for _ in range(n_per_class):
        edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
        if rng.random() < 0.3:
            a, b = rng.choice(n_nodes, size=2, replace=False)
            edges.append((int(a), int(b)))
        graphs.append(from_edges(n_nodes, edges))
        labels.append(0)
    for _ in range(n_per_class):
        edges = [(i, i + 1) for i in range(n_nodes - 1)]
        graphs.append(from_edges(n_nodes, edges))
        labels.append(1)
    order = rng.permutation(len(graphs))
    graphs = [graphs[i] for i in order]
    y = np.array([labels[i] for i in order], dtype=np.int64)
    return graphs, y


def _random_tree_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Prüfer-like: grow random tree."""
    if n <= 1:
        return []
    edges: list[tuple[int, int]] = []
    # connect i to random parent in 0..i-1
    for i in range(1, n):
        p = int(rng.integers(0, i))
        edges.append((p, i))
    return edges


def _has_triangle(g: Graph) -> bool:
    for u in g.nodes:
        nbrs = list(g.neighbors(u))
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if g.has_edge(nbrs[i], nbrs[j]):
                    return True
    return False


def _add_triangle_edge(n: int, edges: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    """Add one edge that creates a triangle (connect two neighbors of a degree≥2 node)."""
    g = from_edges(n, edges)
    candidates = []
    for u in g.nodes:
        nbrs = list(g.neighbors(u))
        if len(nbrs) < 2:
            continue
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b = nbrs[i], nbrs[j]
                if not g.has_edge(a, b):
                    candidates.append((a, b))
    if not candidates:
        # fallback: force a wedge then close
        # add star edge if needed
        return edges + [(0, 1)] if not g.has_edge(0, 1) else edges
    a, b = candidates[int(rng.integers(0, len(candidates)))]
    return edges + [(a, b)]


def _add_long_chord(n: int, edges: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    """Add one edge that does NOT create a triangle (prefer distance ≥ 3)."""
    g = from_edges(n, edges)
    # BFS distances from each node — pick pair with dist>=3 and no edge
    from collections import deque

    def dist_from(s: int) -> dict[int, int]:
        d = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in g.neighbors(u):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    pairs = []
    for s in g.nodes:
        dmap = dist_from(s)
        for t, d in dmap.items():
            if t > s and d >= 3 and not g.has_edge(s, t):
                pairs.append((s, t))
    if not pairs:
        # any non-edge non-triangle: try dist>=2 and check
        for s in g.nodes:
            for t in g.nodes:
                if t <= s or g.has_edge(s, t):
                    continue
                trial = from_edges(n, edges + [(s, t)])
                if not _has_triangle(trial):
                    pairs.append((s, t))
        if not pairs:
            return edges + [(0, n - 1)]
    a, b = pairs[int(rng.integers(0, len(pairs)))]
    return edges + [(a, b)]


def make_triangle_vs_longcycle(
    n_per_class: int = 80,
    n_nodes: int = 16,
    seed: int = 0,
) -> tuple[list[Graph], np.ndarray, dict]:
    """
    Both classes: connected graphs with **same n and same |E|=n** (tree + 1 edge).

    - Class 0: extra edge closes a **triangle**
    - Class 1: extra edge creates a **longer cycle**, triangle-free

    Degree/size alone should NOT perfectly separate; motif structure should matter.
    """
    rng = np.random.default_rng(seed)
    graphs: list[Graph] = []
    labels: list[int] = []
    n_tri = 0
    for _ in range(n_per_class):
        base = _random_tree_edges(n_nodes, rng)
        edges = _add_triangle_edge(n_nodes, base, rng)
        g = from_edges(n_nodes, edges)
        graphs.append(g)
        labels.append(0)
        if _has_triangle(g):
            n_tri += 1
    for _ in range(n_per_class):
        # retry until triangle-free
        for _try in range(50):
            base = _random_tree_edges(n_nodes, rng)
            edges = _add_long_chord(n_nodes, base, rng)
            g = from_edges(n_nodes, edges)
            if not _has_triangle(g):
                break
        graphs.append(g)
        labels.append(1)

    order = rng.permutation(len(graphs))
    graphs = [graphs[i] for i in order]
    y = np.array([labels[i] for i in order], dtype=np.int64)
    meta = {
        "task": "triangle_vs_longcycle",
        "n_nodes": n_nodes,
        "edges_per_graph": n_nodes,  # tree + 1
        "n_per_class": n_per_class,
        "class0_triangle_rate": n_tri / max(n_per_class, 1),
    }
    return graphs, y, meta


def _count_c4(g: Graph) -> int:
    """Count 4-cycles (chordless not required): simple O(n*d^3) style."""
    nodes = g.nodes
    count = 0
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if g.has_edge(a, b):
                continue
            # common neighbors pairs form C4 a-x-b-y-a
            common = list(g.neighbors(a) & g.neighbors(b))
            for j in range(len(common)):
                for k in range(j + 1, len(common)):
                    # a-cj-b-ck-a is always C4 if a-b nonedge
                    count += 1
    return count  # each C4 counted once per opposite pair? Actually once per diagonal pair a,b
    # C4 has two diagonals if no chords; for pure C4 one pair of non-adj opposite nodes...
    # For pure C4: opposite pairs: 2 pairs of non-adjacent nodes each with 2 common nbrs → count=2 per C4
    # We'll use as feature without correcting.


def make_c4_vs_longcycle(
    n_per_class: int = 100,
    n_nodes: int = 12,
    long_cycle: int = 8,
    seed: int = 0,
) -> tuple[list[Graph], np.ndarray, dict]:
    """
    Same n, same |E|=n (unicyclic + trees):

    - Class 0: contains a **C4**, remaining nodes attached as trees.
    - Class 1: contains a **C_{long_cycle}** (default 8), remaining as trees; **no C4**.

    Why multi-hop / larger patch:
    - Degree-2 cycle vertex has 1-hop star S with |S|=3 → induced is always a path P3;
      **cannot see the 4th node that closes C4**.
    - RW / wider sampling can cover 4+ nodes so induced G[S] becomes C4 vs path/P4.

    |E|: Ck has k edges + (n-k) tree edges = n for both if long_cycle and 4 used as core sizes.
    """
    assert long_cycle >= 6
    assert n_nodes >= long_cycle
    rng = np.random.default_rng(seed)
    graphs: list[Graph] = []
    labels: list[int] = []

    def attach_trees(edges: list[tuple[int, int]], used: set[int], n: int) -> list[tuple[int, int]]:
        nxt = 0
        while len(used) < n:
            while nxt in used:
                nxt += 1
            parent = int(rng.choice(list(used)))
            edges = edges + [(parent, nxt)]
            used.add(nxt)
        return edges

    for _ in range(n_per_class):
        # C4 on 0,1,2,3
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        used = {0, 1, 2, 3}
        edges = attach_trees(edges, used, n_nodes)
        graphs.append(from_edges(n_nodes, edges))
        labels.append(0)

    for _ in range(n_per_class):
        L = long_cycle
        edges = [(i, (i + 1) % L) for i in range(L)]
        used = set(range(L))
        edges = attach_trees(edges, used, n_nodes)
        g = from_edges(n_nodes, edges)
        # reject accidental C4 (rare with pure long cycle + tree)
        if _count_c4(g) > 0:
            # rebuild once with different tree attachments
            edges = [(i, (i + 1) % L) for i in range(L)]
            used = set(range(L))
            edges = attach_trees(edges, used, n_nodes)
            g = from_edges(n_nodes, edges)
        graphs.append(g)
        labels.append(1)

    order = rng.permutation(len(graphs))
    graphs = [graphs[i] for i in order]
    y = np.array([labels[i] for i in order], dtype=np.int64)
    c4s = [_count_c4(g) for g in graphs]
    es = [g.num_edges() for g in graphs]
    meta = {
        "task": "c4_vs_longcycle",
        "n_nodes": n_nodes,
        "long_cycle": long_cycle,
        "n_per_class": n_per_class,
        "mean_edges": float(np.mean(es)),
        "mean_c4_class0": float(np.mean([c for c, lab in zip(c4s, y) if lab == 0])),
        "mean_c4_class1": float(np.mean([c for c, lab in zip(c4s, y) if lab == 1])),
        "note": "1-hop star is P3 on cycle verts; need patch covering 4 cycle nodes to see C4",
    }
    return graphs, y, meta


def make_distant_wedge(
    n_per_class: int = 80,
    n_nodes: int = 24,
    seed: int = 0,
    path_len: int = 6,
) -> tuple[list[Graph], np.ndarray, dict]:
    """
    Multi-hop structure task (same n, same |E|).

    Both classes: a path backbone of length (path_len) plus identical local noise
    trees attached so that |E| and degree histograms are similar.

    - Class 0: **two triangles** attached near the two ends of a long path
      (motifs separated by path_len hops — 1-hop star cannot see both).
    - Class 1: **two triangles** both attached near the **same** end
      (or one triangle + disjoint edges) so local 1-hop stats can look similar
      but global arrangement differs.

    More precisely class1: two triangles sharing the same path endpoint (adjacent),
    so B0 can see double-triangle locally; class0 needs multi-hop / RW to relate both ends.
    """
    rng = np.random.default_rng(seed)
    graphs: list[Graph] = []
    labels: list[int] = []

    def base_path() -> list[tuple[int, int]]:
        # nodes 0..path_len along a path
        return [(i, i + 1) for i in range(path_len)]

    def add_triangle_on(edges: list[tuple[int, int]], anchor: int, a: int, b: int) -> list[tuple[int, int]]:
        # triangle anchor-a-b
        return edges + [(anchor, a), (anchor, b), (a, b)]

    def fill_noise(edges: list[tuple[int, int]], used: set[int], n: int) -> list[tuple[int, int]]:
        """Attach remaining nodes as trees so n_nodes matched; keep connected."""
        nxt = 0
        while len(used) < n:
            while nxt in used:
                nxt += 1
            parent = int(rng.choice(list(used)))
            edges = edges + [(parent, nxt)]
            used.add(nxt)
        return edges

    # node budget: path_len+1 path nodes + 4 triangle extras + rest noise
    for _ in range(n_per_class):
        # class 0: triangle at node 0 and at node path_len
        edges = base_path()
        used = set(range(path_len + 1))
        # allocate 4 private nodes for two triangles
        t = path_len + 1
        edges = add_triangle_on(edges, 0, t, t + 1)
        edges = add_triangle_on(edges, path_len, t + 2, t + 3)
        used |= {t, t + 1, t + 2, t + 3}
        edges = fill_noise(edges, used, n_nodes)
        graphs.append(from_edges(n_nodes, edges))
        labels.append(0)

    for _ in range(n_per_class):
        # class 1: both triangles on same end (node 0)
        edges = base_path()
        used = set(range(path_len + 1))
        t = path_len + 1
        edges = add_triangle_on(edges, 0, t, t + 1)
        edges = add_triangle_on(edges, 0, t + 2, t + 3)
        used |= {t, t + 1, t + 2, t + 3}
        edges = fill_noise(edges, used, n_nodes)
        graphs.append(from_edges(n_nodes, edges))
        labels.append(1)

    order = rng.permutation(len(graphs))
    graphs = [graphs[i] for i in order]
    y = np.array([labels[i] for i in order], dtype=np.int64)
    # verify edge counts roughly equal
    es = [g.num_edges() for g in graphs]
    meta = {
        "task": "distant_wedge_triangles",
        "n_nodes": n_nodes,
        "path_len": path_len,
        "n_per_class": n_per_class,
        "mean_edges": float(np.mean(es)),
        "std_edges": float(np.std(es)),
        "note": "class0: triangles at both path ends; class1: both triangles at same end",
    }
    return graphs, y, meta


def degree_features(graphs: list[Graph]) -> np.ndarray:
    rows = []
    for g in graphs:
        degs = [len(g.neighbors(u)) for u in g.nodes] or [0]
        rows.append(
            [
                float(np.mean(degs)),
                float(np.std(degs) if len(degs) > 1 else 0.0),
                float(np.max(degs)),
                float(np.min(degs)),
                float(g.n),
                float(g.num_edges()),
            ]
        )
    return np.array(rows, dtype=np.float64)
