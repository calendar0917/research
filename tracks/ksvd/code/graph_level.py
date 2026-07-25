"""
Default **graph-level** RW→KSVD pipeline (luyin10 main line).

Algorithm (fixed default):
  1. Coverage-driven RW patches (sparse seeds, soft edge decay, no hard delete)
  2. Vectorize each induced G[S] → columns of Y
  3. Shared D on train graphs' patches; encode each graph's Y → X; readout → s_G
  4. Classify with logistic regression (process + synthetic graph task)

Not for node-level GIN fusion (see node_struct.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coverage_sample import CoverageConfig, sample_coverage, trajectory_cover_metrics
from .graph import Graph
from .ksvd import _omp, ksvd, readout_X
from .sample import SampleConfig, run_method
from .vectorize import adjacency_padded, flatten_upper


@dataclass
class GraphLevelConfig:
    """Default graph-level sampler + KSVD."""

    # --- sampler (coverage RW) ---
    p: float = 0.5
    q: float = 2.0
    walk_length: int = 8
    max_nodes: int = 8
    max_walks: int = 12
    edge_decay: float = 0.7
    seed_policy: str = "degree_stratified"
    cover_target: float | None = 0.95
    no_backtrack: bool = True
    # --- KSVD ---
    n_atoms: int = 12
    T: int = 3
    T_min: int = 2
    ksvd_iter: int = 6
    readout_mode: str = "rich"
    seed: int = 0
    order_mode: str = "bfs"
    max_train_patches: int = 6000


def _cov_cfg(cfg: GraphLevelConfig, seed: int | None = None) -> CoverageConfig:
    return CoverageConfig(
        p=cfg.p,
        q=cfg.q,
        walk_length=cfg.walk_length,
        max_nodes=cfg.max_nodes,
        max_walks=cfg.max_walks,
        edge_decay=cfg.edge_decay,
        no_backtrack=cfg.no_backtrack,
        seed=cfg.seed if seed is None else seed,
        seed_policy=cfg.seed_policy,  # type: ignore
        cover_target=cfg.cover_target,
        hard_delete=False,
    )


def sample_patches_graph_level(g: Graph, cfg: GraphLevelConfig, seed: int | None = None):
    """Default sampler: coverage RW. Returns SampleBundle + traj metrics."""
    bundle = sample_coverage(g, _cov_cfg(cfg, seed))
    met = trajectory_cover_metrics(g, bundle)
    return bundle, met


def sample_patches_B0(g: Graph, max_nodes: int = 8, seed: int = 0):
    sc = SampleConfig(max_nodes=max_nodes, seed=seed)
    return run_method(g, "B0", sc)


def bundle_to_Y(
    g: Graph,
    bundle,
    max_nodes: int,
    order_mode: str = "bfs",
) -> tuple[np.ndarray, dict[str, Any]]:
    cols = []
    for S in bundle.node_sets:
        y = np.array(
            flatten_upper(adjacency_padded(g, S, max_nodes, order_mode=order_mode)),
            dtype=np.float64,
        )
        if np.linalg.norm(y) > 1e-12:
            cols.append(y)
    if not cols:
        dim = max_nodes * (max_nodes - 1) // 2
        return np.zeros((dim, 1), dtype=np.float64), {"n_cols": 0}
    Y = np.stack(cols, axis=1)
    return Y, {"n_cols": int(Y.shape[1]), "n_features": int(Y.shape[0])}


def collect_train_Y(
    graphs: list[Graph],
    train_idx: np.ndarray,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
) -> tuple[np.ndarray, dict[str, Any]]:
    cols = []
    covers, repeats, nwalks = [], [], []
    rng = np.random.default_rng(cfg.seed)
    for ti in train_idx:
        g = graphs[int(ti)]
        if mode == "B0":
            b = sample_patches_B0(g, cfg.max_nodes, cfg.seed + int(ti))
            met = {"traj_edge_cover": np.nan, "traj_edge_repeat": np.nan, "n_walks": g.n}
        else:
            b, met = sample_patches_graph_level(g, cfg, seed=cfg.seed + int(ti))
        covers.append(met.get("traj_edge_cover", np.nan))
        repeats.append(met.get("traj_edge_repeat", np.nan))
        nwalks.append(met.get("n_walks", len(b.node_sets)))
        Y, _ = bundle_to_Y(g, b, cfg.max_nodes, cfg.order_mode)
        for j in range(Y.shape[1]):
            if np.linalg.norm(Y[:, j]) > 1e-12:
                cols.append(Y[:, j])
    if not cols:
        raise RuntimeError("no training patches")
    n_raw = len(cols)
    if len(cols) > cfg.max_train_patches:
        pick = rng.choice(len(cols), size=cfg.max_train_patches, replace=False)
        cols = [cols[i] for i in pick]
    Ytr = np.stack(cols, axis=1)
    stats = {
        "n_patches_raw": n_raw,
        "n_patches_used": Ytr.shape[1],
        "mean_traj_cover": float(np.nanmean(covers)),
        "mean_traj_repeat": float(np.nanmean(repeats)),
        "mean_n_walks": float(np.mean(nwalks)),
    }
    return Ytr, stats


def learn_shared_D_graph_level(
    graphs: list[Graph],
    train_idx: np.ndarray,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
) -> tuple[np.ndarray, dict[str, Any]]:
    Ytr, pstats = collect_train_Y(graphs, train_idx, cfg, mode=mode)
    D, X, info = ksvd(
        Ytr,
        n_atoms=cfg.n_atoms,
        T=cfg.T,
        n_iter=cfg.ksvd_iter,
        seed=cfg.seed,
        T_min=cfg.T_min,
    )
    return D, {**info, **pstats, "mode": mode}


def encode_graph(
    g: Graph,
    D: np.ndarray,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """s_G from patches of one graph under fixed D."""
    if mode == "B0":
        b = sample_patches_B0(g, cfg.max_nodes, cfg.seed if seed is None else seed)
        met = {"n_walks": g.n, "traj_edge_cover": np.nan, "traj_edge_repeat": np.nan}
    else:
        b, met = sample_patches_graph_level(g, cfg, seed=seed)
    Y, ymeta = bundle_to_Y(g, b, cfg.max_nodes, cfg.order_mode)
    n_atoms = D.shape[1]
    N = Y.shape[1]
    X = np.zeros((n_atoms, N), dtype=np.float64)
    for j in range(N):
        y = Y[:, j]
        if np.linalg.norm(y) < 1e-12:
            continue
        x = _omp(D, y, cfg.T)
        if np.count_nonzero(np.abs(x) > 1e-10) < cfg.T_min:
            corr = np.abs(D.T @ y)
            top = np.argsort(-corr)[: cfg.T_min]
            Ds = D[:, top]
            coef, _, _, _ = np.linalg.lstsq(Ds, y, rcond=None)
            x = np.zeros(n_atoms, dtype=np.float64)
            for c, k in zip(coef, top):
                x[k] = c
        X[:, j] = x
    # energy-style readout (luyin10)
    emb = readout_X(X, mode=cfg.readout_mode)
    # also explicit energy vector
    energy = (X**2).sum(axis=1)
    usage = (np.abs(X) > 1e-10).mean(axis=1)
    s = np.concatenate([emb, energy, usage])
    R = Y - D @ X
    recon = float(np.linalg.norm(R) / (np.linalg.norm(Y) + 1e-12))
    meta = {
        **met,
        **ymeta,
        "recon_rel": recon,
        "emb_dim": int(s.shape[0]),
    }
    return s, meta


def encode_dataset(
    graphs: list[Graph],
    D: np.ndarray,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows, metas = [], []
    for i, g in enumerate(graphs):
        s, m = encode_graph(g, D, cfg, mode=mode, seed=cfg.seed + i * 13)
        rows.append(s)
        metas.append(m)
    # pad
    d = max(r.shape[0] for r in rows)
    X = np.zeros((len(rows), d), dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, : r.shape[0]] = r
    return X, metas
