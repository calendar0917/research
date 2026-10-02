"""Static typed chordless-cycle objects, NumPy-only research reference.

No ZINC data, target decomposition, RDKit, GNN, attention, or learned neighbor
updates. A ring is read once as a raw atom/bond sequence. All dihedral views
share one MLP and are averaged BEFORE one shared dictionary encoding.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Graph:
    atoms: tuple[int, ...]
    edges: tuple[tuple[int, int, int], ...]

    def __post_init__(self):
        n = len(self.atoms)
        keys = []
        for u, v, b in self.edges:
            if not 0 <= u < v < n or not 0 <= b < 4:
                raise ValueError("expect unique undirected (u<v) edges and bond IDs 0..3")
            keys.append((u, v))
        if len(keys) != len(set(keys)) or any(not 0 <= a < 28 for a in self.atoms):
            raise ValueError("duplicate edge or invalid atom ID")

    def adjacency(self):
        result = [[] for _ in self.atoms]
        for u, v, _ in self.edges:
            result[u].append(v)
            result[v].append(u)
        return [tuple(sorted(x)) for x in result]

    def bonds(self):
        return {(u, v): b for u, v, b in self.edges}


def distances(graph):
    adj = graph.adjacency()
    n = len(adj)
    result = np.full((n, n), n + 1, dtype=int)
    for start in range(n):
        result[start, start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if result[start, v] == n + 1:
                    result[start, v] = result[start, u] + 1
                    queue.append(v)
    return result


def canonical_cycle(vertices):
    vertices = tuple(vertices)
    variants = []
    for direction in (vertices, vertices[::-1]):
        for shift in range(len(direction)):
            variants.append(direction[shift:] + direction[:shift])
    return min(variants)


def chordless_cycles(graph, max_length=10):
    """Enumerate all chordless cycles, not a basis; IDs used only for dedup."""
    adj = graph.adjacency()
    adj_sets = [set(x) for x in adj]
    found = set()
    for start in range(len(adj)):
        def visit(path):
            if len(path) >= max_length:
                return
            for v in adj[path[-1]]:
                if v <= start or v in path:
                    continue
                # A contact with an earlier interior node would be a chord.
                if any(x in adj_sets[v] for x in path[1:-1]):
                    continue
                extended = path + (v,)
                if start in adj_sets[v]:
                    if len(extended) >= 3:
                        found.add(canonical_cycle(extended))
                else:
                    visit(extended)
        for neighbor in adj[start]:
            if neighbor > start:
                visit((start, neighbor))
    return sorted(found, key=lambda c: (len(c), c))


SHELL_PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
MAX_RING_LENGTH = 10
RING_INPUT_DIM = MAX_RING_LENGTH * 28 + MAX_RING_LENGTH * 4 + MAX_RING_LENGTH + 8  # 338


def semantic110(graph):
    """Raw Sem108+size2 precursor. No train-fit standardization required for equality."""
    dist = distances(graph)
    out = np.zeros((len(graph.atoms), 110))
    for root, row in enumerate(dist):
        for v, a in enumerate(graph.atoms):
            if row[v] <= 2:
                out[root, row[v] * 28 + a] += 1
        for u, v, b in graph.edges:
            if row[u] <= 2 and row[v] <= 2:
                pair = tuple(sorted((int(row[u]), int(row[v]))))
                out[root, 84 + 4 * SHELL_PAIRS.index(pair) + b] += 1
        out[root, 108] = np.sum(row <= 2)
        out[root, 109] = sum(row[u] <= 2 and row[v] <= 2 for u, v, _ in graph.edges)
    return out


def dihedral_views(graph, cycle):
    """Atom t and bond (t,t+1) stay aligned under rotations AND reversal."""
    bonds = graph.bonds()
    ell = len(cycle)
    if not 3 <= ell <= MAX_RING_LENGTH:
        raise ValueError("cycle size outside fixed 3..10 range")
    views = []
    for direction in (1, -1):
        for shift in range(ell):
            vertices = [cycle[(shift + direction * t) % ell] for t in range(ell)]
            x = np.zeros(RING_INPUT_DIM)
            for t, v in enumerate(vertices):
                x[t * 28 + graph.atoms[v]] = 1
                next_v = vertices[(t + 1) % ell]
                b = bonds[tuple(sorted((v, next_v)))]
                x[280 + 4 * t + b] = 1
                x[320 + t] = 1
            x[330 + ell - 3] = 1
            views.append(x)
    return np.stack(views)


def ring_mlp_init(seed=3):
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.uniform(-1 / np.sqrt(RING_INPUT_DIM), 1 / np.sqrt(RING_INPUT_DIM), (RING_INPUT_DIM, 64)),
        "b1": rng.uniform(-1 / np.sqrt(RING_INPUT_DIM), 1 / np.sqrt(RING_INPUT_DIM), 64),
        "w2": rng.uniform(-1 / np.sqrt(64), 1 / np.sqrt(64), (64, 48)),
        "b2": rng.uniform(-1 / np.sqrt(64), 1 / np.sqrt(64), 48),
    }


def ring_latents(graph, state, *, linear=False):
    rows = []
    for cycle in chordless_cycles(graph):
        x = dihedral_views(graph, cycle)
        z = x @ state["w1"] + state["b1"]
        if not linear:
            z = z / (1 + np.exp(-z))
        # Nonlinearity happens per oriented view, before orbit average.
        rows.append((z @ state["w2"] + state["b2"]).mean(axis=0))
    return np.stack(rows) if rows else np.empty((0, 48))


def ring_summary(graph, state, D, V, encode):
    h = ring_latents(graph, state)
    if not len(h):
        return np.zeros(97)
    E, _ = encode(h, D, V)
    return np.concatenate([E.sum(axis=0), (E * E).sum(axis=0), [0.0]])


def relabel(graph, permutation):
    """permutation[old] = new."""
    atoms = [0] * len(permutation)
    for old, new in enumerate(permutation):
        atoms[new] = graph.atoms[old]
    edges = []
    for u, v, b in graph.edges:
        a, c = sorted((int(permutation[u]), int(permutation[v])))
        edges.append((a, c, b))
    return Graph(tuple(atoms), tuple(sorted(edges)))


def cube_matching(pairing):
    """All-C cubic cube with one double bond at every vertex (typed-graph witness)."""
    matching = {tuple(sorted(p)) for p in pairing}
    edges = []
    for u in range(8):
        for bit in range(3):
            v = u ^ (1 << bit)
            if u < v:
                edges.append((u, v, int((u, v) in matching)))
    degree = [0] * 8
    for u, v in matching:
        if not any(a == u and c == v for a, c, _ in edges):
            raise ValueError("matching uses a non-edge")
        degree[u] += 1
        degree[v] += 1
    if degree != [1] * 8:
        raise ValueError("expect a perfect matching")
    return Graph((0,) * 8, tuple(sorted(edges)))


def witness_pair():
    parallel = [(0, 1), (2, 3), (4, 5), (6, 7)]
    mixed = [(0, 1), (2, 3), (4, 6), (5, 7)]
    return cube_matching(parallel), cube_matching(mixed)


def square_double_statistics(graph):
    values = []
    bonds = graph.bonds()
    for cycle in chordless_cycles(graph):
        if len(cycle) == 4:
            values.append(sum(bonds[tuple(sorted((cycle[t], cycle[(t + 1) % 4])))] == 1
                              for t in range(4)))
    return sorted(values)
