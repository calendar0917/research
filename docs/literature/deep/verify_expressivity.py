"""Verify: plain MPNN (<=1-WL) cannot distinguish two graphs that are 1-WL-equivalent
but differ in triangle count; a structural (GSN-style) feature can.
Uses a computationally-verified 1-WL-inseparable pair found by search (both 4-regular, 7 nodes).
"""
import sys
from pathlib import Path
try:
    _exp = Path(__file__).resolve().parents[4] / 'paper' / 'experiments'
except Exception:
    _exp = Path('/home/calendar/code/paper/experiments')
if _exp.exists():
    sys.path.insert(0, str(_exp))
import networkx as nx, torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from models import GIN, GSNv

E0 = [(0,2),(0,3),(0,4),(0,6),(1,2),(1,3),(1,4),(1,5),(2,4),(2,5),(3,5),(3,6),(4,6),(5,6)]
E1 = [(0,3),(0,4),(0,5),(0,6),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(5,6)]

G0 = nx.Graph(); G0.add_edges_from(E0)
G1 = nx.Graph(); G1.add_edges_from(E1)

def triangles(G): return sum(nx.triangles(G).values())//3
print("isomorphic?", nx.is_isomorphic(G0,G1))
print("triangles:", triangles(G0), triangles(G1))
print("1-WL hash equal?", nx.weisfeiler_lehman_graph_hash(G0)==nx.weisfeiler_lehman_graph_hash(G1))
print("degree seq:", sorted(dict(G0.degree()).values()), sorted(dict(G1.degree()).values()))

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

# build PyG data (no features, use degree or constant)
def to_data(G):
    nodes = list(G.nodes())
    idx = {v:i for i,v in enumerate(nodes)}
    ei = torch.tensor([[idx[u], idx[v]] for u,v in G.edges()] + [[idx[v], idx[u]] for u,v in G.edges()]).t().contiguous()
    x = torch.ones(G.number_of_nodes(), 1)  # constant feature
    return Data(x=x, edge_index=ei)

d0, d1 = to_data(G0), to_data(G1)
batch = Batch.from_data_list([d0, d1])

torch.manual_seed(0)
for name in ['GIN', 'GSNv']:
    if name == 'GIN':
        model = GIN(in_dim=1, hidden=64, num_classes=2, num_layers=4, dropout=0.0)
        out = model(batch.x, batch.edge_index, batch.batch)
    else:
        # GSN-style: triangle count per node as structural feature
        s0 = torch.tensor(list(nx.triangles(G0).values()), dtype=torch.float).unsqueeze(1)
        s1 = torch.tensor(list(nx.triangles(G1).values()), dtype=torch.float).unsqueeze(1)
        s = torch.cat([s0, s1])
        model = GSNv(in_dim=1, struct_dim=1, hidden=64, num_classes=2, num_layers=4, dropout=0.0)
        out = model(batch.x, batch.edge_index, batch.batch, s)
    g0_emb, g1_emb = out[0], out[1]
    same = torch.allclose(g0_emb, g1_emb, atol=1e-6)
    print(f"{name}: graph embeddings equal? {same}  (g0={g0_emb.detach().numpy().round(4)}, g1={g1_emb.detach().numpy().round(4)})")

# Also: train-free check that a triangle-count vector separates them
print("triangle per-node feature:", sorted(nx.triangles(G0).values()), "vs", sorted(nx.triangles(G1).values()))
