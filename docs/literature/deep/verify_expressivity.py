"""Verify: plain MPNN (<=1-WL) cannot distinguish two graphs that are 1-WL-equivalent
but differ in triangle count; a structural (GSN-style) feature can.
Uses a computationally-verified 1-WL-inseparable pair found by search (both 4-regular, 7 nodes).

Self-contained since 2026-09: the former `paper/experiments/models.py` (GIN/GSNv) was lost
with the legacy repo, so the forward passes are implemented inline (torch + networkx only).
An unbiased MPNN with mean aggregation on constant node features is 1-WL-equivariant:
identical 1-WL colorings => identical embeddings. Injected per-node triangle counts break
that tie (GSN-style structural encoding).
"""
import torch
import networkx as nx

E0 = [(0, 2), (0, 3), (0, 4), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5),
      (2, 4), (2, 5), (3, 5), (3, 6), (4, 6), (5, 6)]
E1 = [(0, 3), (0, 4), (0, 5), (0, 6), (1, 3), (1, 4), (1, 5), (1, 6),
      (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (5, 6)]

G0 = nx.Graph()
G0.add_edges_from(E0)
G1 = nx.Graph()
G1.add_edges_from(E1)


def triangles(G):
    return sum(nx.triangles(G).values()) // 3


print("isomorphic?", nx.is_isomorphic(G0, G1))
print("triangles:", triangles(G0), triangles(G1))
print("1-WL hash equal?",
      nx.weisfeiler_lehman_graph_hash(G0) == nx.weisfeiler_lehman_graph_hash(G1))
print("degree seq:", sorted(dict(G0.degree()).values()),
      sorted(dict(G1.degree()).values()))


def wl_coloring(G):
    labels = {v: 1 for v in G.nodes()}
    for _ in range(len(G)):
        new = {}
        for v in G.nodes():
            new[v] = (labels[v], tuple(sorted(labels[u] for u in G.neighbors(v))))
        uniq = {}
        for x in sorted(set(new.values())):
            uniq[x] = len(uniq)
        labels = {v: uniq[new[v]] for v in G.nodes()}
    return sorted(labels.values())


print("WL multiset equal?", wl_coloring(G0) == wl_coloring(G1))

# ---- inline 4-layer GIN (mean readout), torch-only ----

def mpnn_embed(G, x, n_layers=4, hidden=64, seed=0):
    """x: [N, d] node features. Returns mean-readout embedding (hidden-dim)."""
    torch.manual_seed(seed)
    Ws, bs = [], []
    d = x.size(1)
    for i in range(n_layers):
        Ws.append(torch.empty(d if i == 0 else hidden, hidden).normal_(0, 0.02))
        bs.append(torch.zeros(hidden))
        d = hidden
    adj = torch.zeros(len(G), len(G))
    for u, v in G.edges():
        adj[u, v] = adj[v, u] = 1.0

    h = x.clone().float()
    for W, b in zip(Ws, bs):
        agg = adj @ h / adj.sum(1, keepdim=True).clamp(min=1)   # mean aggregation
        h = torch.relu(agg @ W + b)
    return h.mean(0)                                            # mean readout

def to_node_matrix(G, features):
    n = len(G)
    idx = {v: i for i, v in enumerate(sorted(G.nodes()))}
    d = len(next(iter(features.values())))
    x = torch.zeros(n, d)
    for v, f in features.items():
        x[idx[v], :] = torch.tensor(f, dtype=torch.float)
    return x


# features from adjacency order: use sorted node order consistently
def feats_constant(G):
    return {v: [1.0] for v in sorted(G.nodes())}


def feats_triangles(G):
    return {v: [1.0, float(nx.triangles(G)[v])] for v in sorted(G.nodes())}


for name, feats_fn in [("GIN (constant feat)", feats_constant),
                       ("GIN + triangle feat (GSN-style)", feats_triangles)]:
    x0 = to_node_matrix(G0, feats_fn(G0))
    x1 = to_node_matrix(G1, feats_fn(G1))
    e0 = mpnn_embed(G0, x0)
    e1 = mpnn_embed(G1, x1)
    same = torch.allclose(e0, e1, atol=1e-6)
    print(f"{name}: graph embeddings equal? {same}  "
          f"(g0={e0.detach().numpy().round(4)}, g1={e1.detach().numpy().round(4)})")

print("triangle per-node feature:", sorted(nx.triangles(G0).values()),
      "vs", sorted(nx.triangles(G1).values()))
