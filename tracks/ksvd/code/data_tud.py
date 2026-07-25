from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .graph import Graph
from .pipeline import graph_from_edge_index


def load_tud(
    name: str = "MUTAG",
    root: str | Path | None = None,
) -> tuple[list[Graph], np.ndarray, list[np.ndarray | None], dict[str, Any]]:
    """
    Load TUDataset. Returns graphs, y, node_features list (or None per graph), meta.
    Node features: one-hot or continuous matrix (n_nodes, f) if available.
    """
    from torch_geometric.datasets import TUDataset

    root = Path(root or Path(__file__).resolve().parents[3] / "data" / "TUD")
    root.mkdir(parents=True, exist_ok=True)
    ds = TUDataset(root=str(root), name=name)
    graphs: list[Graph] = []
    labels: list[int] = []
    node_feats: list[np.ndarray | None] = []
    for data in ds:
        n = int(data.num_nodes)
        ei = data.edge_index.cpu().numpy()
        g = graph_from_edge_index(n, ei)
        graphs.append(g)
        y = data.y
        if y.numel() > 1:
            labels.append(int(y.view(-1)[0].item()))
        else:
            labels.append(int(y.item()))
        if getattr(data, "x", None) is not None and data.x is not None:
            node_feats.append(data.x.cpu().numpy().astype(np.float64))
        else:
            node_feats.append(None)
    y_arr = np.array(labels, dtype=np.int64)
    uniq = sorted(set(y_arr.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    y_arr = np.array([remap[int(v)] for v in y_arr], dtype=np.int64)
    has_x = sum(1 for f in node_feats if f is not None)
    feat_dim = int(node_feats[0].shape[1]) if has_x and node_feats[0] is not None else 0
    meta = {
        "name": name,
        "n_graphs": len(graphs),
        "n_classes": len(uniq),
        "mean_n": float(np.mean([g.n for g in graphs])),
        "mean_e": float(np.mean([g.num_edges() for g in graphs])),
        "n_with_node_features": has_x,
        "node_feat_dim": feat_dim,
    }
    return graphs, y_arr, node_feats, meta


def node_feature_readout(x: np.ndarray | None) -> np.ndarray:
    """Graph-level vector from node attributes: mean / max / sum."""
    if x is None or x.size == 0:
        return np.zeros(3, dtype=np.float64)
    # x: (n, f)
    mean = x.mean(axis=0)
    mx = x.max(axis=0)
    sm = x.sum(axis=0)
    return np.concatenate([mean, mx, sm])
