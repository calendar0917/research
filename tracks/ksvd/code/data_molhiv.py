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
    # optional OGB atom/bond features aligned with graphs[i] nodes 0..n-1
    node_feats: list[np.ndarray] | None = None  # each (n_i, 9) int
    edge_feats: list[dict[tuple[int, int], np.ndarray]] | None = None  # undirected key → (3,)


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


def _patch_torch_load_weights_only() -> None:
    """OGB pickles need weights_only=False (PyTorch >=2.6 default True)."""
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.load, "_ksvd_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._ksvd_patched = True  # type: ignore[attr-defined]
    torch.load = _load  # type: ignore[assignment]


def load_molhiv(
    root: str | Path | None = None,
    max_graphs: int | None = None,
    seed: int = 0,
    with_features: bool = False,
) -> MolhivBundle:
    """
    Load ogbg-molhiv with official scaffold split.

    root: data directory (default: <repo>/data/ogb)
    max_graphs: if set, stratified subsample **within each split** (smoke only),
      remapped to contiguous 0..n_used-1. Preserves train/valid/test.
    with_features: also return OGB node_feat / edge_feat for chem patch vectors.
    """
    try:
        _patch_torch_load_weights_only()
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

    tr = np.asarray(split_idx["train"], dtype=np.int64)
    va = np.asarray(split_idx["valid"], dtype=np.int64)
    te = np.asarray(split_idx["test"], dtype=np.int64)

    if max_graphs is not None and max_graphs < n_full:
        # keep split proportions, min 1 per split when possible
        rng = np.random.default_rng(seed)
        n_tr_full, n_va_full, n_te_full = len(tr), len(va), len(te)
        frac = max_graphs / n_full
        n_tr = max(1, int(round(n_tr_full * frac)))
        n_va = max(1, int(round(n_va_full * frac)))
        n_te = max(1, int(round(n_te_full * frac)))
        # adjust to exact max_graphs
        total = n_tr + n_va + n_te
        while total > max_graphs and n_tr > 1:
            n_tr -= 1
            total -= 1
        while total < max_graphs and n_tr < n_tr_full:
            n_tr += 1
            total += 1
        tr = rng.choice(tr, size=min(n_tr, n_tr_full), replace=False)
        va = rng.choice(va, size=min(n_va, n_va_full), replace=False)
        te = rng.choice(te, size=min(n_te, n_te_full), replace=False)
        keep = np.unique(np.concatenate([tr, va, te]))
        keep.sort()
        old_to_new = {int(o): i for i, o in enumerate(keep)}
        tr = np.array([old_to_new[int(i)] for i in tr], dtype=np.int64)
        va = np.array([old_to_new[int(i)] for i in va], dtype=np.int64)
        te = np.array([old_to_new[int(i)] for i in te], dtype=np.int64)
        indices = keep
    else:
        indices = np.arange(n_full, dtype=np.int64)
        # tr/va/te already global indices

    graphs: list[Graph] = []
    labels: list[float] = []
    smiles: list[str] = []
    node_feats: list[np.ndarray] | None = [] if with_features else None
    edge_feats: list[dict[tuple[int, int], np.ndarray]] | None = [] if with_features else None
    for oi in indices:
        g_dict, y = dataset[int(oi)]
        n_nodes = int(g_dict["num_nodes"])
        ei = np.asarray(g_dict["edge_index"])
        graphs.append(_edges_from_pyg(ei, n_nodes))
        yy = np.asarray(y).reshape(-1)
        labels.append(float(yy[0]))
        if "smiles" in g_dict:
            smiles.append(str(g_dict["smiles"]))
        if with_features:
            nf = np.asarray(g_dict["node_feat"], dtype=np.int64)
            node_feats.append(nf)  # type: ignore[union-attr]
            ef_map: dict[tuple[int, int], np.ndarray] = {}
            efeat = np.asarray(g_dict["edge_feat"], dtype=np.int64)
            src, dst = ei[0], ei[1]
            for k in range(src.shape[0]):
                u, v = int(src[k]), int(dst[k])
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                if key not in ef_map:
                    ef_map[key] = efeat[k].copy()
            edge_feats.append(ef_map)  # type: ignore[union-attr]

    y_arr = np.asarray(labels, dtype=np.float64)
    n = len(graphs)
    split = {"train": tr, "valid": va, "test": te}
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
        "subsample_seed": seed if max_graphs is not None else None,
        "with_features": with_features,
    }
    return MolhivBundle(
        graphs=graphs,
        y=y_arr,
        split=split,
        smiles=smiles if smiles else None,
        meta=meta,
        node_feats=node_feats,
        edge_feats=edge_feats,
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
