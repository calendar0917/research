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
from .vectorize import (
    adjacency_padded,
    flatten_upper,
    labeled_wl_patch_features,
    labeled_wl_ring_patch_features,
    patch_feature_dim,
    wl_patch_features,
)


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
    ring_boost: float = 0.0  # >0: prefer triangle-rich edges in RW
    # --- KSVD ---
    n_atoms: int = 12
    T: int = 3
    T_min: int = 2
    ksvd_iter: int = 6
    readout_mode: str = "rich"  # basic | rich | pool
    pool: str = "mean"  # mean | max | sum | attn (MIL)
    seed: int = 0
    order_mode: str = "bfs"
    max_train_patches: int = 6000
    # --- patch vector ---
    # wl_chem(_ring): permutation-invariant full OGB labels (+ true cycles).
    patch_feat: str = "topo"  # topo | wl | chem | wl_chem | wl_chem_ring
    normalize_patches: bool = False
    max_patches_per_graph: int | None = None
    # Radius used by atom-centered samplers.  Coverage/B0 samplers ignore it;
    # keeping it in the shared config makes localized protocols explicit.
    radius: int = 2
    # Standard historical initialization caps K at the input width.  An
    # explicit opt-in permits a genuinely overcomplete dictionary (K > d).
    allow_overcomplete: bool = False


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
        ring_boost=cfg.ring_boost,
    )


def _chem_hist_for_patch(
    S: set[int],
    node_feat: np.ndarray | None,
    edge_feat: dict[tuple[int, int], np.ndarray] | None,
    g: Graph,
    n_atom_bins: int = 16,
    n_bond_bins: int = 8,
) -> np.ndarray:
    """Coarse atom/bond type histograms on induced patch (OGB first channel)."""
    atom_h = np.zeros(n_atom_bins, dtype=np.float64)
    bond_h = np.zeros(n_bond_bins, dtype=np.float64)
    if node_feat is not None:
        for u in S:
            if 0 <= u < node_feat.shape[0]:
                t = int(node_feat[u, 0]) % n_atom_bins
                atom_h[t] += 1.0
        s = atom_h.sum()
        if s > 0:
            atom_h /= s
    if edge_feat is not None:
        for u in S:
            for v in g.neighbors(u):
                if v in S and u < v:
                    key = (u, v)
                    ef = edge_feat.get(key)
                    if ef is not None and ef.size:
                        t = int(ef[0]) % n_bond_bins
                        bond_h[t] += 1.0
        s = bond_h.sum()
        if s > 0:
            bond_h /= s
    return np.concatenate([atom_h, bond_h])


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
    patch_feat: str = "topo",
    node_feat: np.ndarray | None = None,
    edge_feat: dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cols = []
    for S in bundle.node_sets:
        y = np.array(
            flatten_upper(adjacency_padded(g, S, max_nodes, order_mode=order_mode)),
            dtype=np.float64,
        )
        if patch_feat == "wl":
            y = np.asarray(wl_patch_features(g, S, max_nodes), dtype=np.float64)
        elif patch_feat == "chem":
            y = np.concatenate(
                [y, _chem_hist_for_patch(S, node_feat, edge_feat, g)]
            )
        elif patch_feat == "wl_chem":
            if node_feat is None or edge_feat is None:
                raise ValueError("wl_chem requires node_feat and edge_feat")
            y = np.asarray(
                labeled_wl_patch_features(
                    g, S, max_nodes, node_feat=node_feat, edge_feat=edge_feat
                ),
                dtype=np.float64,
            )
        elif patch_feat == "wl_chem_ring":
            if node_feat is None or edge_feat is None:
                raise ValueError("wl_chem_ring requires node_feat and edge_feat")
            y = np.asarray(
                labeled_wl_ring_patch_features(
                    g, S, max_nodes, node_feat=node_feat, edge_feat=edge_feat
                ),
                dtype=np.float64,
            )
        elif patch_feat != "topo":
            raise ValueError(f"unknown patch_feat={patch_feat!r}")
        if np.linalg.norm(y) > 1e-12:
            cols.append(y)
    if not cols:
        dim = patch_feature_dim(max_nodes, patch_feat)
        return np.zeros((dim, 1), dtype=np.float64), {"n_cols": 0}
    Y = np.stack(cols, axis=1)
    return Y, {"n_cols": int(Y.shape[1]), "n_features": int(Y.shape[0])}


def collect_train_Y(
    graphs: list[Graph],
    train_idx: np.ndarray,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
    node_feats: list[np.ndarray] | None = None,
    edge_feats: list[dict[tuple[int, int], np.ndarray]] | None = None,
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
        nf = node_feats[int(ti)] if node_feats is not None else None
        ef = edge_feats[int(ti)] if edge_feats is not None else None
        Y, _ = bundle_to_Y(
            g,
            b,
            cfg.max_nodes,
            cfg.order_mode,
            patch_feat=cfg.patch_feat,
            node_feat=nf,
            edge_feat=ef,
        )
        if cfg.normalize_patches:
            norms = np.linalg.norm(Y, axis=0, keepdims=True)
            Y = Y / np.maximum(norms, 1e-12)
        if cfg.max_patches_per_graph is not None and Y.shape[1] > cfg.max_patches_per_graph:
            local = rng.choice(Y.shape[1], size=cfg.max_patches_per_graph, replace=False)
            Y = Y[:, local]
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
        "patch_feat": cfg.patch_feat,
    }
    return Ytr, stats


def learn_shared_D_graph_level(
    graphs: list[Graph],
    train_idx: np.ndarray,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
    node_feats: list[np.ndarray] | None = None,
    edge_feats: list[dict[tuple[int, int], np.ndarray]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    Ytr, pstats = collect_train_Y(
        graphs, train_idx, cfg, mode=mode, node_feats=node_feats, edge_feats=edge_feats
    )
    D, X, info = ksvd(
        Ytr,
        n_atoms=cfg.n_atoms,
        T=cfg.T,
        n_iter=cfg.ksvd_iter,
        seed=cfg.seed,
        T_min=cfg.T_min,
    )
    return D, {**info, **pstats, "mode": mode}


def sparse_code_patch_matrix(
    Y: np.ndarray,
    D: np.ndarray,
    cfg: GraphLevelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized patches and their OMP codes under a fixed dictionary.

    Keeping sparse coding separate from graph readout lets experiments compare
    multiple pooling/statistical readouts without resampling patches or rerunning
    OMP.  The function uses exactly the same T/T_min behavior as the historical
    :func:`encode_patch_matrix` path.
    """
    if Y.shape[0] != D.shape[0]:
        raise ValueError(f"feature mismatch: Y has {Y.shape[0]} rows, D has {D.shape[0]}")
    Yn = np.asarray(Y, dtype=np.float64)
    if cfg.normalize_patches:
        norms = np.linalg.norm(Yn, axis=0, keepdims=True)
        Yn = Yn / np.maximum(norms, 1e-12)

    n_atoms = D.shape[1]
    N = Yn.shape[1]
    X = np.zeros((n_atoms, N), dtype=np.float64)
    for j in range(N):
        y = Yn[:, j]
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
    return Yn, X


def sparse_code_readouts(
    X: np.ndarray,
    patch_errors: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fixed graph statistics used by the MolHIV readout experiments.

    The summaries are deliberately label-free and dictionary-agnostic.  They
    can therefore be used unchanged for KSVD and matched dictionary controls.
    """
    A = np.abs(X)
    k, n = A.shape
    zero = np.zeros(k, dtype=np.float64)
    if n:
        mean = A.mean(axis=1)
        maxv = A.max(axis=1)
        std = A.std(axis=1)
        usage = (A > 1e-10).mean(axis=1)
        q75, q90 = np.quantile(A, [0.75, 0.90], axis=1)
        top = np.sort(A, axis=1)[:, -min(3, n) :].mean(axis=1)
        energy = (X * X).mean(axis=1)
        signed_mean = X.mean(axis=1)
        winner = np.bincount(np.argmax(A, axis=0), minlength=k).astype(np.float64) / n
    else:
        mean = maxv = std = usage = q75 = q90 = top = energy = signed_mean = winner = zero
    errors = np.asarray([] if patch_errors is None else patch_errors, dtype=np.float64)
    if errors.size:
        recon = np.array(
            [errors.mean(), errors.std(), *np.quantile(errors, [0.50, 0.75, 0.90]), errors.max()],
            dtype=np.float64,
        )
    else:
        recon = np.zeros(6, dtype=np.float64)
    count = np.array([float(n), np.log1p(n)], dtype=np.float64)
    basic = np.concatenate([mean, maxv, usage])
    tail = np.concatenate([top, q75, q90])
    moments = np.concatenate([mean, std, energy])
    rich_no_recon = np.concatenate(
        [mean, maxv, top, std, usage, q75, q90, energy, signed_mean, winner]
    )
    return {
        "max": maxv,
        "basic": basic,
        "tail": tail,
        "moments": moments,
        "recon": np.concatenate([recon, count]),
        "rich_no_recon": rich_no_recon,
        "rich": np.concatenate([rich_no_recon, recon, count]),
    }


def encode_patch_matrix(
    Y: np.ndarray,
    D: np.ndarray,
    cfg: GraphLevelConfig,
    return_patch_errors: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sparse-code one graph's already-vectorized patch matrix."""
    Y, X = sparse_code_patch_matrix(Y, D, cfg)

    emb = readout_X(X, mode=cfg.readout_mode, pool=cfg.pool, seed=cfg.seed)
    energy = (X**2).sum(axis=1)
    usage = (np.abs(X) > 1e-10).mean(axis=1)
    s = emb if cfg.readout_mode == "pool" else np.concatenate([emb, energy, usage])
    R = Y - D @ X
    meta = {
        "recon_rel": float(np.linalg.norm(R) / (np.linalg.norm(Y) + 1e-12)),
        "emb_dim": int(s.shape[0]),
        "readout_mode": cfg.readout_mode,
        "pool": cfg.pool,
    }
    if return_patch_errors:
        denom = np.maximum(np.linalg.norm(Y, axis=0), 1e-12)
        meta["patch_recon_errors"] = (np.linalg.norm(R, axis=0) / denom).tolist()
    return s, meta


def encode_graph(
    g: Graph,
    D: np.ndarray,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
    seed: int | None = None,
    node_feat: np.ndarray | None = None,
    edge_feat: dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """s_G from patches of one graph under fixed D."""
    if mode == "B0":
        b = sample_patches_B0(g, cfg.max_nodes, cfg.seed if seed is None else seed)
        met = {"n_walks": g.n, "traj_edge_cover": np.nan, "traj_edge_repeat": np.nan}
    else:
        b, met = sample_patches_graph_level(g, cfg, seed=seed)
    Y, ymeta = bundle_to_Y(
        g,
        b,
        cfg.max_nodes,
        cfg.order_mode,
        patch_feat=cfg.patch_feat,
        node_feat=node_feat,
        edge_feat=edge_feat,
    )
    s, code_meta = encode_patch_matrix(Y, D, cfg)
    return s, {**met, **ymeta, **code_meta}


def encode_dataset(
    graphs: list[Graph],
    D: np.ndarray,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
    node_feats: list[np.ndarray] | None = None,
    edge_feats: list[dict[tuple[int, int], np.ndarray]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if node_feats is not None and len(node_feats) != len(graphs):
        raise ValueError("node_feats and graphs must have the same length")
    if edge_feats is not None and len(edge_feats) != len(graphs):
        raise ValueError("edge_feats and graphs must have the same length")
    rows, metas = [], []
    for i, g in enumerate(graphs):
        nf = node_feats[i] if node_feats is not None else None
        ef = edge_feats[i] if edge_feats is not None else None
        s, m = encode_graph(
            g,
            D,
            cfg,
            mode=mode,
            seed=cfg.seed + i * 13,
            node_feat=nf,
            edge_feat=ef,
        )
        rows.append(s)
        metas.append(m)
    # pad
    d = max(r.shape[0] for r in rows)
    X = np.zeros((len(rows), d), dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, : r.shape[0]] = r
    return X, metas
