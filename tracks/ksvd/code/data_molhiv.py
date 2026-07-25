"""
ogbg-molhiv loader → internal Graph + labels + OGB scaffold splits.

Requires: ogb, torch, torch_geometric (optional for full dual-channel).
Falls back with clear ImportError if missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .graph import Graph, from_edges


@dataclass
class MolhivBundle:
    graphs: list[Graph]
    y: np.ndarray  # (N,) float/int labels
    split: dict[str, np.ndarray]  # train/valid/test indices
    smiles: list[str] | None
    meta: dict[str, Any]


def _edges_from_pyg(edge_index: np.ndarray, n: int) -> Graph:
    """edge_index (2, E) → undirected Graph."""
    edges: list[tuple[int, int]] = []
    if edge_index.size == 0:
        return from_edges(n, [])
    src, dst = edge_index[0], edge_index[1]
    for u, v in zip(src.tolist(), dst.tolist()):
        u, v = int(u), int(v)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edges.append((u, v))
    # unique
    edges = list(set(edges))
    return from_edges(n, edges)


def load_molhiv(
    root: str | Path | None = None,
    max_graphs: int | None = None,
) -> MolhivBundle:
    """
    Load ogbg-molhiv with official scaffold split.

    root: data directory (default: <repo>/data/ogb)
    max_graphs: if set, take first N graphs and intersect splits (smoke only).
    """
    try:
        from ogb.graphproppred import GraphPropPredDataset
    except ImportError as e:
        raise ImportError(
            "ogb not installed. Run: pip install -r tracks/ksvd/configs/requirements-molhiv.txt"
        ) from e

    repo_root = Path(__file__).resolve().parents[3]
    if root is None:
        root = repo_root / "data" / "ogb"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    split_idx = dataset.get_idx_split()
    n_full = len(dataset)
    n = n_full if max_graphs is None else min(max_graphs, n_full)

    graphs: list[Graph] = []
    labels: list[float] = []
    smiles: list[str] = []
    for i in range(n):
        g_dict, y = dataset[i]
        # g_dict: edge_index, num_nodes, node_feat, edge_feat, ...
        n_nodes = int(g_dict["num_nodes"])
        ei = np.asarray(g_dict["edge_index"])
        graphs.append(_edges_from_pyg(ei, n_nodes))
        # y shape (1,) or scalar
        yy = np.asarray(y).reshape(-1)
        labels.append(float(yy[0]))
        if "smiles" in g_dict:
            smiles.append(str(g_dict["smiles"]))

    y_arr = np.asarray(labels, dtype=np.float64)

    def _clip(idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        return idx[idx < n]

    split = {
        "train": _clip(split_idx["train"]),
        "valid": _clip(split_idx["valid"]),
        "test": _clip(split_idx["test"]),
    }
    meta = {
        "name": "ogbg-molhiv",
        "n_full": n_full,
        "n_used": n,
        "max_graphs": max_graphs,
        "root": str(root),
        "n_train": int(len(split["train"])),
        "n_valid": int(len(split["valid"])),
        "n_test": int(len(split["test"])),
        "pos_rate": float(y_arr.mean()) if n else 0.0,
    }
    return MolhivBundle(
        graphs=graphs,
        y=y_arr,
        split=split,
        smiles=smiles if smiles else None,
        meta=meta,
    )


def degree_hist_features(graphs: list[Graph], max_deg: int = 10) -> np.ndarray:
    """Simple structure baseline: degree histogram (normalized)."""
    X = np.zeros((len(graphs), max_deg + 1), dtype=np.float64)
    for i, g in enumerate(graphs):
        for u in g.nodes:
            d = min(len(g.neighbors(u)), max_deg)
            X[i, d] += 1.0
        s = X[i].sum()
        if s > 0:
            X[i] /= s
    return X


def check_env() -> dict[str, Any]:
    """Report which molhiv deps are available."""
    out: dict[str, Any] = {}
    for name in ("numpy", "sklearn", "torch", "torch_geometric", "ogb"):
        mod = name if name != "sklearn" else "sklearn"
        try:
            m = __import__(mod)
            out[name] = getattr(m, "__version__", "ok")
        except ImportError:
            out[name] = None
    out["ready_structure_probe"] = out.get("ogb") is not None and out.get("numpy") is not None
    out["ready_gine"] = (
        out.get("torch") is not None
        and out.get("torch_geometric") is not None
        and out.get("ogb") is not None
    )
    return out
