from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .graph import Graph, from_edges
from .ksvd import ksvd, readout_X, _omp
from .sample import SampleConfig, run_method
from .vectorize import adjacency_padded, flatten_upper, subgraph_extra_features


@dataclass
class PipelineConfig:
    method: str = "M0"
    p: float = 1.0
    q: float = 1.0
    walk_length: int = 8
    max_nodes: int = 12
    num_walks: int = 20
    edge_decay: float = 0.7
    no_backtrack: bool = True
    seed: int = 0
    n_atoms: int = 12
    T: int = 3
    T_min: int = 2
    ksvd_iter: int = 8
    order_mode: str = "bfs"  # bfs | degree
    append_local_stats: bool = False  # append triangle etc. to each y (stronger local signal)
    readout_mode: str = "basic"  # basic | rich

def graph_from_edge_index(n: int, edge_index: np.ndarray) -> Graph:
    edges: list[tuple[int, int]] = []
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u != v:
            edges.append((u, v))
    return from_edges(n, edges)


def sample_Y(
    g: Graph,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    sc = SampleConfig(
        p=cfg.p,
        q=cfg.q,
        walk_length=cfg.walk_length,
        max_nodes=cfg.max_nodes,
        num_walks=cfg.num_walks,
        edge_decay=cfg.edge_decay,
        no_backtrack=cfg.no_backtrack,
        seed=cfg.seed,
    )
    bundle = run_method(g, cfg.method, sc)
    cols = []
    for S in bundle.node_sets:
        A = adjacency_padded(g, S, cfg.max_nodes, order_mode=cfg.order_mode)
        vec = flatten_upper(A)
        if cfg.append_local_stats:
            vec = vec + subgraph_extra_features(g, S)
        cols.append(vec)
    if not cols:
        dim = cfg.max_nodes * (cfg.max_nodes - 1) // 2
        if cfg.append_local_stats:
            dim += 5
        Y = np.zeros((dim, 1), dtype=np.float64)
        return Y, {"n_cols": 0, "method": cfg.method}
    Y = np.stack(cols, axis=1).astype(np.float64)
    meta = {
        "n_cols": Y.shape[1],
        "n_features": Y.shape[0],
        "method": cfg.method,
        "mean_|S|": float(np.mean([len(s) for s in bundle.node_sets])),
        "order_mode": cfg.order_mode,
    }
    return Y, meta


def structure_embedding(g: Graph, cfg: PipelineConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-graph dictionary (legacy)."""
    Y, meta = sample_Y(g, cfg)
    norms = np.linalg.norm(Y, axis=0)
    keep = norms > 1e-12
    if not np.any(keep):
        Y = Y + 1e-3 * np.random.default_rng(cfg.seed).standard_normal(Y.shape)
    else:
        Y = Y[:, keep]
    D, X, info = ksvd(
        Y,
        n_atoms=cfg.n_atoms,
        T=cfg.T,
        n_iter=cfg.ksvd_iter,
        seed=cfg.seed,
        T_min=cfg.T_min,
    )
    emb = readout_X(X, mode=cfg.readout_mode)
    out = {**meta, **info, "emb_dim": int(emb.shape[0]), "mode": "per_graph_dict"}
    return emb, out


def embed_dataset(
    graphs: list[Graph],
    cfg: PipelineConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    embs = []
    infos = []
    for i, g in enumerate(graphs):
        c = PipelineConfig(**{**cfg.__dict__, "seed": cfg.seed + i * 17})
        e, info = structure_embedding(g, c)
        embs.append(e)
        infos.append(info)
    d = max(e.shape[0] for e in embs)
    X = np.zeros((len(embs), d), dtype=np.float64)
    for i, e in enumerate(embs):
        X[i, : e.shape[0]] = e
    return X, infos


def _encode_with_D(Y: np.ndarray, D: np.ndarray, T: int, T_min: int) -> np.ndarray:
    n_atoms = D.shape[1]
    N = Y.shape[1]
    X = np.zeros((n_atoms, N), dtype=np.float64)
    for i in range(N):
        X[:, i] = _omp(D, Y[:, i], T)
        if np.count_nonzero(np.abs(X[:, i]) > 1e-10) < T_min:
            corr = np.abs(D.T @ Y[:, i])
            top = np.argsort(-corr)[:T_min]
            Ds = D[:, top]
            coef, _, _, _ = np.linalg.lstsq(Ds, Y[:, i], rcond=None)
            X[:, i] = 0.0
            for c, j in zip(coef, top):
                X[j, i] = c
    return X


def embed_dataset_shared_dict(
    graphs: list[Graph],
    cfg: PipelineConfig,
    *,
    train_idx: np.ndarray | None = None,
    max_train_cols: int = 4000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Fix A: learn one D on (subset of) train graphs' patches, encode all graphs with same D.
    """
    if train_idx is None:
        train_idx = np.arange(len(graphs))
    # collect Y columns from train
    cols = []
    rng = np.random.default_rng(cfg.seed)
    for ti in train_idx:
        g = graphs[int(ti)]
        c = PipelineConfig(**{**cfg.__dict__, "seed": cfg.seed + int(ti) * 17})
        Y, _ = sample_Y(g, c)
        for j in range(Y.shape[1]):
            if np.linalg.norm(Y[:, j]) > 1e-12:
                cols.append(Y[:, j])
    if not cols:
        raise RuntimeError("no training patches")
    if len(cols) > max_train_cols:
        pick = rng.choice(len(cols), size=max_train_cols, replace=False)
        cols = [cols[i] for i in pick]
    Ytr = np.stack(cols, axis=1)
    D, _, info = ksvd(
        Ytr,
        n_atoms=cfg.n_atoms,
        T=cfg.T,
        n_iter=cfg.ksvd_iter,
        seed=cfg.seed,
        T_min=cfg.T_min,
    )
    embs = []
    for i, g in enumerate(graphs):
        c = PipelineConfig(**{**cfg.__dict__, "seed": cfg.seed + i * 17})
        Y, _ = sample_Y(g, c)
        norms = np.linalg.norm(Y, axis=0)
        keep = norms > 1e-12
        if not np.any(keep):
            # dim depends on readout
            dummy = readout_X(np.zeros((cfg.n_atoms, 1)), mode=cfg.readout_mode)
            embs.append(np.zeros_like(dummy))
            continue
        Y = Y[:, keep]
        X = _encode_with_D(Y, D, cfg.T, cfg.T_min)
        embs.append(readout_X(X, mode=cfg.readout_mode))
    Xout = np.stack(embs, axis=0)
    meta = {
        "mode": "shared_dict",
        "n_train_graphs": int(len(train_idx)),
        "n_train_patches": int(Ytr.shape[1]),
        "recon_rel_train": info["recon_rel"],
        "atoms_used_train": info["atoms_used"],
        "order_mode": cfg.order_mode,
        "readout_mode": cfg.readout_mode,
    }
    return Xout, meta


def graph_oracle_features(graphs: list[Graph]) -> np.ndarray:
    """Whole-graph hand features (upper bound / diagnostic)."""
    rows = []
    for g in graphs:
        S = set(g.nodes)
        rows.append(subgraph_extra_features(g, S))
    return np.array(rows, dtype=np.float64)


def patch_pool_features(graphs: list[Graph], cfg: PipelineConfig) -> np.ndarray:
    """
    Fix diagnostic: no KSVD — for each graph, sample patches and mean-pool
    flatten(adj)+local stats. Isolates sampling+vectorize quality.
    """
    rows = []
    for i, g in enumerate(graphs):
        c = PipelineConfig(**{**cfg.__dict__, "seed": cfg.seed + i * 17, "append_local_stats": True})
        Y, _ = sample_Y(g, c)
        if Y.shape[1] == 0:
            rows.append(np.zeros(Y.shape[0] if Y.size else 1))
        else:
            rows.append(Y.mean(axis=1))
    d = max(r.shape[0] for r in rows)
    X = np.zeros((len(rows), d), dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, : r.shape[0]] = r
    return X
