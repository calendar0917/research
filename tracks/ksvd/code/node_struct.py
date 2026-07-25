from __future__ import annotations

"""Node-level structure coefficients via shared dictionary."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .graph import Graph
from .ksvd import _omp, ksvd
from .sample import SampleConfig, node2vec_walk
from .vectorize import adjacency_padded, flatten_upper


@dataclass
class NodeStructConfig:
    """How to build patch(es) for each node, then encode with shared D."""

    mode: str = "B0"  # B0 = 1-hop star; RW = random walk(s) from node
    max_nodes: int = 8
    walk_length: int = 6
    num_walks: int = 1  # r walks per node (RW mode); pooled into s_v
    pool: str = "mean"  # mean | max | mean_max  over walk coefficients
    p: float = 1.0
    q: float = 1.0
    no_backtrack: bool = True
    order_mode: str = "bfs"
    n_atoms: int = 12
    T: int = 3
    T_min: int = 2
    ksvd_iter: int = 6
    seed: int = 0
    feat: str = "abs_coef"  # abs_coef | soft | abs_soft


def _one_rw_set(g: Graph, v: int, cfg: NodeStructConfig, walk_id: int) -> set[int]:
    import random

    sc = SampleConfig(
        p=cfg.p,
        q=cfg.q,
        walk_length=cfg.walk_length,
        max_nodes=cfg.max_nodes,
        num_walks=1,
        edge_decay=1.0,
        no_backtrack=cfg.no_backtrack,
        seed=cfg.seed + int(v) + 10007 * walk_id,
    )
    r = random.Random(cfg.seed + 17 * int(v) + 97 * walk_id)
    ew = {e: 1.0 for e in g.edges()}
    walk, _ = node2vec_walk(g, v, sc, r, ew)
    S: list[int] = []
    for u in walk:
        if u not in S:
            S.append(u)
        if len(S) >= cfg.max_nodes:
            break
    return set(S) if S else {v}


def patches_for_node(g: Graph, v: int, cfg: NodeStructConfig) -> list[set[int]]:
    """Return list of node-sets (one for B0; r for RW)."""
    if cfg.mode.upper() == "B0":
        S = {v} | set(g.neighbors(v))
        if len(S) > cfg.max_nodes:
            others = sorted(S - {v}, key=lambda u: (-len(g.neighbors(u)), u))
            S = {v} | set(others[: cfg.max_nodes - 1])
        return [S]
    r = max(1, int(cfg.num_walks))
    return [_one_rw_set(g, v, cfg, i) for i in range(r)]


def patch_for_node(g: Graph, v: int, cfg: NodeStructConfig) -> set[int]:
    """Back-compat: first / only patch."""
    return patches_for_node(g, v, cfg)[0]


def vectorize_patch(g: Graph, S: set[int], cfg: NodeStructConfig) -> np.ndarray:
    A = adjacency_padded(g, S, cfg.max_nodes, order_mode=cfg.order_mode)
    return np.array(flatten_upper(A), dtype=np.float64)


def encode_y(D: np.ndarray, y: np.ndarray, cfg: NodeStructConfig) -> np.ndarray:
    """Sparse code one patch vector y -> coefficient x (n_atoms,)."""
    if np.linalg.norm(y) < 1e-12:
        return np.zeros(D.shape[1], dtype=np.float64)
    x = _omp(D, y, cfg.T)
    if np.count_nonzero(np.abs(x) > 1e-10) < cfg.T_min:
        corr = np.abs(D.T @ y)
        top = np.argsort(-corr)[: cfg.T_min]
        Ds = D[:, top]
        coef, _, _, _ = np.linalg.lstsq(Ds, y, rcond=None)
        x = np.zeros(D.shape[1], dtype=np.float64)
        for c, j in zip(coef, top):
            x[j] = c
    return x


def pool_coefs(xs: list[np.ndarray], pool: str) -> np.ndarray:
    """Pool list of coefficient vectors into one."""
    if not xs:
        raise ValueError("empty coef list")
    if len(xs) == 1:
        return xs[0]
    A = np.stack(xs, axis=0)  # (r, n_atoms)
    if pool == "max":
        return np.max(np.abs(A), axis=0) * np.sign(A[np.argmax(np.abs(A), axis=0), np.arange(A.shape[1])])
    if pool == "mean_max":
        return np.concatenate([A.mean(axis=0), np.max(np.abs(A), axis=0)])
    # mean (default): mean of raw coefs
    return A.mean(axis=0)


def collect_train_patches(
    graphs: list[Graph],
    train_idx: np.ndarray,
    cfg: NodeStructConfig,
    max_patches: int = 8000,
) -> tuple[np.ndarray, dict[str, float]]:
    """Y: (n_features, n_patches). All walks of all train nodes (subsample if huge)."""
    cols: list[np.ndarray] = []
    sizes: list[int] = []
    rng = np.random.default_rng(cfg.seed)
    for ti in train_idx:
        g = graphs[int(ti)]
        for v in g.nodes:
            for S in patches_for_node(g, v, cfg):
                y = vectorize_patch(g, S, cfg)
                if np.linalg.norm(y) > 1e-12:
                    cols.append(y)
                    sizes.append(len(S))
    if not cols:
        raise RuntimeError("no node patches")
    n_before = len(cols)
    if len(cols) > max_patches:
        pick = rng.choice(len(cols), size=max_patches, replace=False)
        cols = [cols[i] for i in pick]
        sizes = [sizes[i] for i in pick]
    Y = np.stack(cols, axis=1)
    stats = {
        "n_patches_raw": float(n_before),
        "n_patches_used": float(Y.shape[1]),
        "mean_|S|": float(np.mean(sizes)),
        "max_|S|": float(np.max(sizes)),
    }
    return Y, stats


def learn_shared_D(
    graphs: list[Graph],
    train_idx: np.ndarray,
    cfg: NodeStructConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    Y, pstats = collect_train_patches(graphs, train_idx, cfg)
    D, X, info = ksvd(
        Y,
        n_atoms=cfg.n_atoms,
        T=cfg.T,
        n_iter=cfg.ksvd_iter,
        seed=cfg.seed,
        T_min=cfg.T_min,
    )
    # per-column recon for diagnostics (same Y distribution)
    R = Y - D @ X
    col_rel = np.linalg.norm(R, axis=0) / (np.linalg.norm(Y, axis=0) + 1e-12)
    return D, {
        **info,
        **pstats,
        "n_train_patches": int(Y.shape[1]),
        "n_features": int(Y.shape[0]),
        "recon_rel": float(info["recon_rel"]),
        "recon_col_mean": float(col_rel.mean()),
        "recon_col_std": float(col_rel.std()),
        "num_walks": int(cfg.num_walks) if cfg.mode.upper() == "RW" else 1,
        "pool": cfg.pool if cfg.mode.upper() == "RW" else "none",
    }


def coef_to_feat(x: np.ndarray, feat: str) -> np.ndarray:
    ax = np.abs(x)
    if feat == "abs_coef":
        return ax
    soft = ax / (ax.sum() + 1e-12)
    if feat == "soft":
        return soft
    return np.concatenate([ax, soft])


def encode_graph_nodes(
    g: Graph,
    D: np.ndarray,
    cfg: NodeStructConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    s_v from r walks: encode each patch, pool coefs, then abs_coef feature.
    Returns (n_nodes, d), plus graph-level recon diagnostics on those patches.
    """
    n = g.n
    # output dim
    n_atoms = D.shape[1]
    if cfg.pool == "mean_max" and cfg.mode.upper() == "RW" and cfg.num_walks > 1:
        base_dim = 2 * n_atoms
    else:
        base_dim = n_atoms
    if cfg.feat == "abs_soft":
        out_dim = 2 * base_dim
    else:
        out_dim = base_dim

    out = np.zeros((n, out_dim), dtype=np.float64)
    recon_list: list[float] = []
    size_list: list[int] = []

    for v in g.nodes:
        xs = []
        for S in patches_for_node(g, v, cfg):
            y = vectorize_patch(g, S, cfg)
            size_list.append(len(S))
            x = encode_y(D, y, cfg)
            xs.append(x)
            if np.linalg.norm(y) > 1e-12:
                recon_list.append(float(np.linalg.norm(y - D @ x) / (np.linalg.norm(y) + 1e-12)))
        pooled = pool_coefs(xs, cfg.pool if len(xs) > 1 else "mean")
        # if mean_max, pooled is already 2*n_atoms; feat abs_coef applies abs to whole
        if cfg.feat == "abs_coef":
            feat = np.abs(pooled)
        elif cfg.feat == "soft":
            ax = np.abs(pooled)
            feat = ax / (ax.sum() + 1e-12)
        else:
            ax = np.abs(pooled)
            feat = np.concatenate([ax, ax / (ax.sum() + 1e-12)])
        out[int(v), : feat.shape[0]] = feat

    diag = {
        "mean_|S|": float(np.mean(size_list)) if size_list else 0.0,
        "patch_recon_mean": float(np.mean(recon_list)) if recon_list else 0.0,
        "patch_recon_std": float(np.std(recon_list)) if recon_list else 0.0,
        "n_patches_encoded": float(len(size_list)),
    }
    return out, diag


def encode_dataset_nodes(
    graphs: list[Graph],
    D: np.ndarray,
    cfg: NodeStructConfig,
) -> tuple[list[np.ndarray], dict[str, float]]:
    structs = []
    recons, sizes = [], []
    for g in graphs:
        s, d = encode_graph_nodes(g, D, cfg)
        structs.append(s)
        recons.append(d["patch_recon_mean"])
        sizes.append(d["mean_|S|"])
    agg = {
        "encode_patch_recon_mean": float(np.mean(recons)),
        "encode_patch_recon_std": float(np.std(recons)),
        "encode_mean_|S|": float(np.mean(sizes)),
    }
    return structs, agg


def zscore_node_features(
    structs: list[np.ndarray],
    train_idx: np.ndarray,
) -> list[np.ndarray]:
    chunks = [structs[int(i)] for i in train_idx]
    all_tr = np.vstack(chunks)
    mu = all_tr.mean(axis=0)
    sig = all_tr.std(axis=0) + 1e-6
    return [(s - mu) / sig for s in structs]
