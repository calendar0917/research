"""Check whether node-marking (the subgraph-GNN mechanism) separates the WL-inseparable pair."""
import networkx as nx

E0 = [(0,2),(0,3),(0,4),(0,6),(1,2),(1,3),(1,4),(1,5),(2,4),(2,5),(3,5),(3,6),(4,6),(5,6)]
E1 = [(0,3),(0,4),(0,5),(0,6),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(5,6)]
G0 = nx.Graph(); G0.add_edges_from(E0)
G1 = nx.Graph(); G1.add_edges_from(E1)

def wl_coloring(G, marked=None):
    labels = {v: 1 for v in G.nodes()}
    if marked is not None: labels[marked] = 2
    for _ in range(len(G)):
        new = {}
        for v in G.nodes():
            new[v] = (labels[v], tuple(sorted(labels[u] for u in G.neighbors(v))))
        uniq = {}
        for x in sorted(set(new.values())):
            uniq[x] = len(uniq)
        labels = {v: uniq[new[v]] for v in G.nodes()}
    return labels

def marked_multiset(G):
    # multiset over roots of final colorings (counts of each color)
    return sorted(tuple(sorted(wl_coloring(G, r).values())) for r in G.nodes())

def all_colorings_equal(G, H):
    # compare sorted multisets of colorings per root
    return marked_multiset(G) == marked_multiset(H)

print("marked-WL separates?", not all_colorings_equal(G0, G1))

# Also show which roots differ
c0 = marked_multiset(G0); c1 = marked_multiset(G1)
import collections
diff = [x for x in set(c0)|set(c1) if c0.count(x)!=c1.count(x)]
print("distinct colorings:", len(set(c0)), "vs", len(set(c1)), "| differing coloring types:", len(diff))
