"""Build train-only dictionaries and localized node-token caches for MolHIV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.extmath import randomized_svd
from ogb.utils.features import get_atom_feature_dims

from .data_molhiv import (
    _edges_from_pyg,
    _patch_torch_load_weights_only,
    load_molhiv,
)
from .ksvd import _omp, ksvd
from .molhiv_node_tokens import (
    centered_ego_vector,
    graph_node_offsets,
    iter_centered_vectors,
)


def _csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _reservoir_train_patches(
    graphs,
    train_idx,
    node_feats,
    edge_feats,
    radius: int,
    max_nodes: int,
    capacity: int,
    seed: int,
    max_patches_per_graph: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a train-only patch reservoir with optional graph balancing.

    ``max_patches_per_graph=0`` preserves the historical node-uniform stream.
    A positive cap samples up to that many centers uniformly inside each train
    graph before reservoir sampling, preventing large molecules from dominating
    the unsupervised dictionary solely through their node count.
    """
    if capacity <= 0:
        raise ValueError("patch reservoir capacity must be positive")
    if max_patches_per_graph < 0:
        raise ValueError("max_patches_per_graph must be nonnegative")

    reservoir_rng = np.random.default_rng(seed)
    center_rng = np.random.default_rng(seed + 104729)
    reservoir: np.ndarray | None = None
    reservoir_graphs = np.full(capacity, -1, dtype=np.int64)
    seen = 0
    available = 0
    contributing_graphs = 0

    def offer(graph_i: int, vec: np.ndarray) -> None:
        nonlocal reservoir, seen
        vec = np.asarray(vec, dtype=np.float64)
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
        if reservoir is None:
            reservoir = np.zeros((vec.size, capacity), dtype=np.float64)
        if vec.size != reservoir.shape[0]:
            raise ValueError("atom-centered patch dimension changed within the dataset")
        if seen < capacity:
            slot = seen
        else:
            slot = int(reservoir_rng.integers(0, seen + 1))
        if slot < capacity:
            reservoir[:, slot] = vec
            reservoir_graphs[slot] = graph_i
        seen += 1

    if max_patches_per_graph == 0:
        for graph_i, _, vec in iter_centered_vectors(
            graphs,
            train_idx,
            node_feats,
            edge_feats,
            radius=radius,
            max_nodes=max_nodes,
        ):
            offer(int(graph_i), vec)
        available = seen
        contributing_graphs = int(len(train_idx))
    else:
        for raw_i in train_idx:
            i = int(raw_i)
            g = graphs[i]
            nodes = np.asarray(list(g.nodes), dtype=np.int64)
            available += int(nodes.size)
            if nodes.size == 0:
                continue
            n_take = min(int(nodes.size), max_patches_per_graph)
            if n_take < nodes.size:
                centers = center_rng.choice(nodes, size=n_take, replace=False)
                centers.sort()
            else:
                centers = nodes
            contributing_graphs += 1
            for raw_u in centers:
                u = int(raw_u)
                vec = centered_ego_vector(
                    g, u, node_feats[i], edge_feats[i],
                    radius=radius, max_nodes=max_nodes,
                )
                offer(i, vec)

    if reservoir is None or seen == 0:
        raise RuntimeError("no atom-centered train patches")
    used = min(seen, capacity)
    selected_graphs = reservoir_graphs[:used]
    unique_graphs, selected_counts = np.unique(selected_graphs, return_counts=True)
    return reservoir[:, :used], {
        "sampling": (
            "node_uniform" if max_patches_per_graph == 0 else "graph_capped_uniform_centers"
        ),
        "max_patches_per_graph": int(max_patches_per_graph),
        "n_train_node_patches_available": int(available),
        "n_train_node_patches_after_graph_cap": int(seen),
        # Historical key retained for downstream report compatibility.
        "n_train_node_patches_raw": int(seen),
        "n_train_node_patches_used": int(used),
        "n_train_graphs_contributing_candidates": int(contributing_graphs),
        "n_train_graphs_represented_in_reservoir": int(unique_graphs.size),
        "selected_patches_per_represented_graph_mean": float(selected_counts.mean()),
        "selected_patches_per_represented_graph_max": int(selected_counts.max()),
        "feature_dim": int(reservoir.shape[0]),
    }



def _external_official_train_patches(
    development_original_indices: np.ndarray,
    root: str | Path,
    radius: int,
    max_nodes: int,
    capacity: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample one unlabeled patch per development-disjoint official-train graph.

    The external pool is restricted to full-dataset official-train molecules that
    are absent from the complete current development subset.  Official-valid and
    official-test molecules are executable exclusion invariants, and labels are
    neither read for sampling nor used in vectorization/dictionary fitting.
    """
    if capacity <= 0:
        raise ValueError("external official-train patch capacity must be positive")

    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset

    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    split_idx = dataset.get_idx_split()
    full_train = np.asarray(split_idx["train"], dtype=np.int64)
    full_valid = np.asarray(split_idx["valid"], dtype=np.int64)
    full_test = np.asarray(split_idx["test"], dtype=np.int64)
    development = np.unique(
        np.asarray(development_original_indices, dtype=np.int64)
    )

    candidates = np.setdiff1d(full_train, development, assume_unique=False)
    if capacity > candidates.size:
        raise ValueError(
            f"requested {capacity} external patches but only {candidates.size} "
            "development-disjoint official-train graphs are available"
        )

    graph_rng = np.random.default_rng(seed + 32452843)
    center_rng = np.random.default_rng(seed + 49979687)
    selected = graph_rng.choice(candidates, size=capacity, replace=False).astype(
        np.int64
    )

    development_set = set(development.tolist())
    valid_set = set(full_valid.tolist())
    test_set = set(full_test.tolist())
    selected_set = set(selected.tolist())
    overlap_development = len(selected_set & development_set)
    overlap_valid = len(selected_set & valid_set)
    overlap_test = len(selected_set & test_set)
    if overlap_development or overlap_valid or overlap_test:
        raise AssertionError(
            "external official-train isolation failed: "
            f"development={overlap_development}, valid={overlap_valid}, "
            f"test={overlap_test}"
        )

    reservoir: np.ndarray | None = None
    selected_centers = np.empty(capacity, dtype=np.int64)
    for out_j, original_i in enumerate(selected.tolist()):
        # GraphPropPredDataset returns the label alongside the graph.  Bind it to
        # an explicitly unused value: no labels enter selection or fitting.
        g_dict, _unused_label = dataset[int(original_i)]
        n_nodes = int(g_dict["num_nodes"])
        if n_nodes <= 0:
            raise RuntimeError(f"external graph {original_i} has no nodes")
        edge_index = np.asarray(g_dict["edge_index"])
        graph = _edges_from_pyg(edge_index, n_nodes)
        node_feat = np.asarray(g_dict["node_feat"], dtype=np.int64)
        raw_edge_feat = np.asarray(g_dict["edge_feat"], dtype=np.int64)
        edge_feat: dict[tuple[int, int], np.ndarray] = {}
        src, dst = edge_index[0], edge_index[1]
        for k in range(src.shape[0]):
            u, v = int(src[k]), int(dst[k])
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key not in edge_feat:
                edge_feat[key] = raw_edge_feat[k].copy()

        center = int(center_rng.integers(0, n_nodes))
        selected_centers[out_j] = center
        vec = centered_ego_vector(
            graph,
            center,
            node_feat,
            edge_feat,
            radius=radius,
            max_nodes=max_nodes,
        )
        vec = np.asarray(vec, dtype=np.float64)
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
        if reservoir is None:
            reservoir = np.zeros((vec.size, capacity), dtype=np.float64)
        if vec.size != reservoir.shape[0]:
            raise ValueError("external atom-centered patch dimension changed")
        reservoir[:, out_j] = vec
        if (out_j + 1) % 1000 == 0 or out_j + 1 == capacity:
            print(
                f"external official-train patches {out_j + 1}/{capacity}",
                flush=True,
            )

    assert reservoir is not None
    import hashlib

    return reservoir, {
        "sampling": "graph_uniform_one_random_center_without_replacement",
        "labels_used": False,
        "full_official_train_count": int(full_train.size),
        "full_official_valid_count": int(full_valid.size),
        "full_official_test_count": int(full_test.size),
        "development_subset_count": int(development.size),
        "development_official_train_count": int(
            np.intersect1d(development, full_train).size
        ),
        "full_official_train_candidate_count": int(candidates.size),
        "selected_external_graph_count": int(selected.size),
        "external_patch_count": int(reservoir.shape[1]),
        "feature_dim": int(reservoir.shape[0]),
        "selected_original_indices_sha256": hashlib.sha256(
            selected.tobytes()
        ).hexdigest(),
        "selected_centers_sha256": hashlib.sha256(
            selected_centers.tobytes()
        ).hexdigest(),
        "overlap_with_development_subset": int(overlap_development),
        "overlap_with_official_valid": int(overlap_valid),
        "overlap_with_official_test": int(overlap_test),
        "graph_selection_seed": int(seed + 32452843),
        "center_selection_seed": int(seed + 49979687),
    }


def _scaffold_balanced_train_patches(
    graphs,
    train_idx,
    node_feats,
    edge_feats,
    radius: int,
    max_nodes: int,
    capacity: int,
    seed: int,
    scaffold_by_graph: dict[int, str],
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample patches with inverse scaffold-frequency weighting.

    Each center in scaffold group ``g`` receives weight ``|g|^{-alpha}``,
    where ``|g|`` is the number of candidate atom centers in the inner-train
    portion of that scaffold.  Alpha=0 is node-uniform; alpha=1 makes total
    scaffold mass uniform.  We use 0<alpha<1 as a group-robust compromise.
    Scaffold identity is used only to fit the unsupervised dictionary and is
    never encoded into model inputs or evaluation graphs.
    """
    if capacity <= 0:
        raise ValueError("patch reservoir capacity must be positive")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("scaffold balance alpha must be in (0, 1]")

    center_graphs: list[int] = []
    center_nodes: list[int] = []
    center_groups: list[str] = []
    for raw_i in train_idx:
        i = int(raw_i)
        if i not in scaffold_by_graph:
            raise KeyError(f"missing scaffold group for fit graph {i}")
        group = scaffold_by_graph[i]
        for raw_u in graphs[i].nodes:
            center_graphs.append(i)
            center_nodes.append(int(raw_u))
            center_groups.append(group)
    n_available = len(center_graphs)
    if n_available == 0:
        raise RuntimeError("no atom-centered train patches")

    groups, inverse = np.unique(np.asarray(center_groups), return_inverse=True)
    group_counts = np.bincount(inverse).astype(np.int64)
    center_weights = np.power(group_counts[inverse].astype(np.float64), -alpha)
    center_weights /= center_weights.sum()
    rng = np.random.default_rng(seed + 161803)
    n_take = min(capacity, n_available)
    selected = rng.choice(
        n_available, size=n_take, replace=False, p=center_weights
    )

    reservoir: np.ndarray | None = None
    selected_group_ids = inverse[selected]
    for out_j, center_j in enumerate(selected.tolist()):
        i = center_graphs[center_j]
        u = center_nodes[center_j]
        vec = centered_ego_vector(
            graphs[i], u, node_feats[i], edge_feats[i],
            radius=radius, max_nodes=max_nodes,
        )
        vec = np.asarray(vec, dtype=np.float64)
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
        if reservoir is None:
            reservoir = np.zeros((vec.size, n_take), dtype=np.float64)
        reservoir[:, out_j] = vec
    assert reservoir is not None

    selected_counts = np.bincount(
        selected_group_ids, minlength=len(groups)
    )
    represented = selected_counts > 0
    return reservoir, {
        "sampling": "scaffold_inverse_frequency_weighted_centers",
        "scaffold_balance_alpha": float(alpha),
        "n_train_node_patches_available": int(n_available),
        "n_train_node_patches_after_graph_cap": int(n_available),
        "n_train_node_patches_raw": int(n_available),
        "n_train_node_patches_used": int(n_take),
        "n_train_graphs_contributing_candidates": int(len(train_idx)),
        "n_train_scaffolds": int(len(groups)),
        "n_train_scaffolds_represented_in_reservoir": int(represented.sum()),
        "selected_patches_per_represented_scaffold_mean": float(
            selected_counts[represented].mean()
        ),
        "selected_patches_per_represented_scaffold_max": int(
            selected_counts.max()
        ),
        "largest_scaffold_candidate_centers": int(group_counts.max()),
        "feature_dim": int(reservoir.shape[0]),
    }


def _random_patch_dictionary(Y: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(Y.shape[1], size=n_atoms, replace=Y.shape[1] < n_atoms)
    D = Y[:, idx].astype(np.float64).copy()
    return D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)


def _cluster_train_patches(
    Y: np.ndarray, n_clusters: int, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unsupervised spherical-ish patch domains, fitted on train patches only."""
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if Y.shape[1] < n_clusters:
        raise ValueError("fewer train patches than requested clusters")
    # Columns are already unit normalized, so Euclidean clustering is equivalent
    # to cosine clustering up to a constant.  float32 limits memory/runtime.
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(1024, Y.shape[1]),
        n_init=10,
        max_iter=200,
        reassignment_ratio=0.0,
    )
    labels = model.fit_predict(Y.T.astype(np.float32, copy=False)).astype(np.int64)
    counts = np.bincount(labels, minlength=n_clusters)
    if np.any(counts == 0):
        raise RuntimeError(f"empty train-patch cluster: counts={counts.tolist()}")
    return labels, {
        "method": "train_only_minibatch_kmeans_on_unit_patches",
        "n_clusters": int(n_clusters),
        "cluster_counts": counts.tolist(),
        "cluster_fractions": (counts / counts.sum()).tolist(),
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
    }


def _cluster_balanced_columns(
    Y: np.ndarray, labels: np.ndarray, n_clusters: int, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample a fixed-size, equal-mass pool from learned patch domains."""
    rng = np.random.default_rng(seed + 271828)
    total = Y.shape[1]
    quotas = np.full(n_clusters, total // n_clusters, dtype=np.int64)
    quotas[: total % n_clusters] += 1
    selected: list[np.ndarray] = []
    unique_selected = 0
    replacement_clusters: list[int] = []
    for c, quota in enumerate(quotas):
        members = np.flatnonzero(labels == c)
        replace = members.size < int(quota)
        if replace:
            replacement_clusters.append(c)
        draw = rng.choice(members, size=int(quota), replace=replace)
        selected.append(np.asarray(draw, dtype=np.int64))
        unique_selected += int(np.unique(draw).size)
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return Y[:, indices], {
        "balanced_counts": quotas.tolist(),
        "replacement_clusters": replacement_clusters,
        "n_unique_source_patches": int(unique_selected),
        "n_balanced_patches": int(indices.size),
    }


def _fit_ksvd_mixture(
    Y: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    n_atoms: int,
    sparsity: int,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, list[slice], dict[str, Any]]:
    if n_atoms % n_clusters != 0:
        raise ValueError("mixture requires n_atoms divisible by n_clusters")
    atoms_per = n_atoms // n_clusters
    if sparsity > atoms_per:
        raise ValueError("mixture sparsity cannot exceed atoms per subdictionary")
    blocks: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    slices: list[slice] = []
    start = 0
    for c in range(n_clusters):
        Yc = Y[:, labels == c]
        if Yc.shape[1] < atoms_per:
            raise ValueError(f"cluster {c} has only {Yc.shape[1]} patches for {atoms_per} atoms")
        D, _, info = ksvd(
            Yc, n_atoms=atoms_per, T=sparsity, T_min=1,
            n_iter=n_iter, seed=seed + 1009 * c,
        )
        blocks.append(D)
        stop = start + D.shape[1]
        slices.append(slice(start, stop))
        infos.append({"cluster": c, "n_patches": int(Yc.shape[1]), **info})
        start = stop
    return np.concatenate(blocks, axis=1), slices, {
        "routing": "minimum_omp_reconstruction_error",
        "n_subdictionaries": int(n_clusters),
        "atoms_per_subdictionary": int(atoms_per),
        "subdictionaries": infos,
    }


def _encode_ksvd_mixture(
    D: np.ndarray, y: np.ndarray, slices: list[slice], sparsity: int,
) -> tuple[np.ndarray, int, float]:
    best_x: np.ndarray | None = None
    best_cluster = -1
    best_error = float("inf")
    for c, block in enumerate(slices):
        Dc = D[:, block]
        xc = _omp(Dc, y, sparsity)
        err = float(np.dot(y - Dc @ xc, y - Dc @ xc))
        if err < best_error:
            best_error = err
            best_cluster = c
            best_x = xc
    if best_x is None:
        raise RuntimeError("mixture routing failed")
    out = np.zeros(D.shape[1], dtype=np.float64)
    out[slices[best_cluster]] = best_x
    return out, best_cluster, best_error


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=8000, help="0 means full dataset")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument("--n-atoms", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--ksvd-iter", type=int, default=3)
    ap.add_argument(
        "--patch-view",
        choices=[
            "full", "no_ring", "rooted_topology",
            "center_residual_no_ring",
        ],
        default="full",
        help=(
            "full historical chemistry+topology vector, or a no-ring/no-chemistry "
            "rooted topology view for complementary structural KSVD"
        ),
    )
    ap.add_argument(
        "--patch-metric",
        choices=[
            "raw", "pca", "pca_whiten",
            "masked_autoencoder", "masked_block_autoencoder",
        ],
        default="raw",
        help=(
            "train-only patch geometry used by every dictionary family; "
            "pca denoises and pca_whiten additionally equalizes retained axes"
        ),
    )
    ap.add_argument(
        "--patch-latent-dim", type=int, default=64,
        help="latent dimensions for PCA or masked-autoencoder patch metrics",
    )
    ap.add_argument(
        "--patch-whiten-floor", type=float, default=1e-3,
        help="relative singular-value floor for stable PCA whitening",
    )
    ap.add_argument("--patch-ae-epochs", type=int, default=30)
    ap.add_argument("--patch-ae-batch-size", type=int, default=256)
    ap.add_argument("--patch-ae-lr", type=float, default=2e-3)
    ap.add_argument("--patch-ae-mask-prob", type=float, default=0.15)
    ap.add_argument("--patch-ae-weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-train-patches", type=int, default=6000)
    ap.add_argument(
        "--external-official-train-patches", type=int, default=0,
        help=(
            "number of graph-uniform unlabeled patches from full official-train "
            "graphs disjoint from the complete current development subset"
        ),
    )
    ap.add_argument(
        "--ksvd-external-residual-atoms", type=int, default=0,
        help=(
            "append this many KSVD atoms learned on external-patch residuals "
            "after fitting the --n-atoms core dictionary on inner-fold patches"
        ),
    )
    ap.add_argument(
        "--max-patches-per-graph", type=int, default=0,
        help="0 keeps node-uniform sampling; positive values cap candidate centers per train graph",
    )
    ap.add_argument(
        "--scaffold-balance-alpha", type=float, default=0.0,
        help=(
            "0 disables scaffold balancing; values in (0,1] inverse-weight "
            "inner-train patch centers by scaffold frequency during dictionary fit"
        ),
    )
    ap.add_argument(
        "--ksvd-clusters", type=int, default=4,
        help="learned patch domains for ksvd_cluster_balanced/ksvd_mixture",
    )
    ap.add_argument("--families", default="random_patch,pca,ksvd")
    ap.add_argument(
        "--fit-indices-cache", default="",
        help="optional fold archive; fit dictionaries only on one external inner-train fold",
    )
    ap.add_argument("--fit-fold", type=int, default=-1)
    ap.add_argument(
        "--encode-official-valid", action="store_true",
        help="also encode official-valid graphs (disabled for fold-selection caches)",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.external_official_train_patches < 0:
        raise ValueError("--external-official-train-patches must be nonnegative")
    if args.ksvd_external_residual_atoms < 0:
        raise ValueError("--ksvd-external-residual-atoms must be nonnegative")
    if (
        args.ksvd_external_residual_atoms > 0
        and args.external_official_train_patches <= 0
    ):
        raise ValueError(
            "--ksvd-external-residual-atoms requires "
            "--external-official-train-patches > 0"
        )
    if not 0.0 <= args.scaffold_balance_alpha <= 1.0:
        raise ValueError("--scaffold-balance-alpha must be in [0, 1]")
    if args.patch_latent_dim <= 0:
        raise ValueError("--patch-latent-dim must be positive")
    if args.patch_whiten_floor <= 0.0:
        raise ValueError("--patch-whiten-floor must be positive")
    if args.patch_ae_epochs <= 0 or args.patch_ae_batch_size <= 0:
        raise ValueError("patch autoencoder epochs/batch size must be positive")
    if args.patch_ae_lr <= 0.0 or args.patch_ae_weight_decay < 0.0:
        raise ValueError("invalid patch autoencoder optimizer configuration")
    if not 0.0 <= args.patch_ae_mask_prob < 1.0:
        raise ValueError("--patch-ae-mask-prob must be in [0,1)")
    if args.scaffold_balance_alpha > 0.0 and not args.fit_indices_cache:
        raise ValueError("scaffold balancing requires --fit-indices-cache")
    if args.scaffold_balance_alpha > 0.0 and args.max_patches_per_graph > 0:
        raise ValueError("scaffold balancing and graph patch caps are mutually exclusive")

    t0 = time.time()
    families = _csv(args.families)
    unknown = set(families) - {
        "random_patch", "pca", "ksvd", "ksvd_cluster_balanced", "ksvd_mixture"
    }
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    bundle = load_molhiv(max_graphs=max_graphs, seed=args.data_seed, with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV chemical features were not loaded")
    graphs = bundle.graphs
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)
    fit_indices = tr
    fit_protocol = "official-train"
    scaffold_by_graph: dict[int, str] | None = None
    if args.fit_indices_cache:
        if args.fit_fold < 0:
            raise ValueError("--fit-indices-cache requires --fit-fold >= 0")
        with np.load(args.fit_indices_cache, allow_pickle=False) as folds:
            key = f"fold_{args.fit_fold}_train_indices"
            if key not in folds.files:
                raise KeyError(f"{key} missing from {args.fit_indices_cache}")
            fit_indices = np.asarray(folds[key], dtype=np.int64)
            if "official_train_indices" in folds.files and not np.array_equal(
                np.asarray(folds["official_train_indices"], dtype=np.int64), tr
            ):
                raise ValueError("fold cache official train does not match dataset")
            if args.scaffold_balance_alpha > 0.0:
                if "train_scaffold_groups" not in folds.files:
                    raise KeyError("fold cache lacks train_scaffold_groups")
                groups = np.asarray(folds["train_scaffold_groups"]).astype(str)
                if groups.shape != tr.shape:
                    raise ValueError("train scaffold groups are not aligned to official train")
                scaffold_by_graph = {
                    int(graph_i): str(group)
                    for graph_i, group in zip(tr.tolist(), groups.tolist())
                }
        if not set(fit_indices.tolist()).issubset(set(tr.tolist())):
            raise ValueError("dictionary fit indices contain non-official-train graphs")
        fit_protocol = "external-official-train-fold"
    elif args.fit_fold >= 0:
        raise ValueError("--fit-fold requires --fit-indices-cache")

    if args.scaffold_balance_alpha > 0.0:
        assert scaffold_by_graph is not None
        Ytr, patch_stats = _scaffold_balanced_train_patches(
            graphs, fit_indices, bundle.node_feats, bundle.edge_feats,
            radius=args.radius, max_nodes=args.max_nodes,
            capacity=args.max_train_patches, seed=args.dict_seed,
            scaffold_by_graph=scaffold_by_graph,
            alpha=args.scaffold_balance_alpha,
        )
        fit_protocol += "-scaffold-balanced"
    else:
        Ytr, patch_stats = _reservoir_train_patches(
            graphs,
            fit_indices,
            bundle.node_feats,
            bundle.edge_feats,
            radius=args.radius,
            max_nodes=args.max_nodes,
            capacity=args.max_train_patches,
            seed=args.dict_seed,
            max_patches_per_graph=args.max_patches_per_graph,
        )
    print(f"train patch pool {Ytr.shape}; raw={patch_stats['n_train_node_patches_raw']}", flush=True)

    external_patch_stats: dict[str, Any] | None = None
    inner_patch_count_for_dictionary = int(Ytr.shape[1])
    if args.external_official_train_patches > 0:
        inner_patch_count = inner_patch_count_for_dictionary
        Yexternal, external_patch_stats = _external_official_train_patches(
            development_original_indices=np.asarray(
                bundle.meta["original_indices"], dtype=np.int64
            ),
            root=bundle.meta["root"],
            radius=args.radius,
            max_nodes=args.max_nodes,
            capacity=args.external_official_train_patches,
            seed=args.dict_seed,
        )
        if Yexternal.shape[0] != Ytr.shape[0]:
            raise ValueError(
                "inner and external patch dimensions differ: "
                f"{Ytr.shape[0]} vs {Yexternal.shape[0]}"
            )
        Ytr = np.concatenate([Ytr, Yexternal], axis=1)
        patch_stats["n_inner_train_patches_used"] = inner_patch_count
        patch_stats["n_external_official_train_patches_used"] = int(
            Yexternal.shape[1]
        )
        patch_stats["n_dictionary_patches_used"] = int(Ytr.shape[1])
        patch_stats["external_official_train"] = external_patch_stats
        fit_protocol += (
            "+external-official-train-development-disjoint-unlabeled"
        )
        print(
            f"combined inner+external patch pool {Ytr.shape}; "
            f"external={Yexternal.shape[1]}",
            flush=True,
        )

    # Patch views test whether the dictionary should model the full chemistry
    # vector or only a complementary structural/context residual.  Every
    # statistic below is fit strictly on the fold's dictionary-fit patches.
    patch_view_indices: np.ndarray | None = None
    patch_condition_keys: np.ndarray | None = None
    patch_condition_means: np.ndarray | None = None
    patch_condition_global_mean: np.ndarray | None = None
    patch_condition_lookup: dict[tuple[int, ...], int] = {}
    unknown_condition_count = 0
    patch_view_info: dict[str, Any] = {"name": args.patch_view}
    Yview = Ytr

    wl_topology_dim = (
        args.max_nodes
        + args.max_nodes * (args.max_nodes + 1) // 2
        + 64 * 3
        + 3
    )
    labeled_base_dim = wl_topology_dim + 64 + 16 + 64 * 3
    explicit_ring_dim = 13
    center_start = labeled_base_dim + explicit_ring_dim
    center_dims = list(get_atom_feature_dims())
    center_end = center_start + sum(center_dims)
    rooted_count_dim = 2 * (args.radius + 1)

    if args.patch_view == "no_ring":
        patch_view_indices = np.concatenate([
            np.arange(labeled_base_dim, dtype=np.int64),
            np.arange(center_start, Ytr.shape[0], dtype=np.int64),
        ])
        Yview = Ytr[patch_view_indices].copy()
        view_norms = np.linalg.norm(Yview, axis=0)
        nonzero_view = view_norms > 1e-12
        Yview[:, nonzero_view] /= view_norms[nonzero_view]
        Yview[:, ~nonzero_view] = 0.0
        patch_view_info.update({
            "input_dim": int(Ytr.shape[0]),
            "output_dim": int(Yview.shape[0]),
            "explicit_ring_features": False,
            "atom_or_bond_label_features": True,
            "center_atom_one_hot_in_dictionary_input": True,
            "zero_view_train_patches": int((~nonzero_view).sum()),
        })
        print(
            f"patch view no_ring: {Ytr.shape[0]} -> {Yview.shape[0]}",
            flush=True,
        )
    elif args.patch_view == "rooted_topology":
        patch_view_indices = np.concatenate([
            np.arange(wl_topology_dim, dtype=np.int64),
            np.arange(Ytr.shape[0] - rooted_count_dim, Ytr.shape[0], dtype=np.int64),
        ])
        Yview = Ytr[patch_view_indices].copy()
        view_norms = np.linalg.norm(Yview, axis=0)
        nonzero_view = view_norms > 1e-12
        Yview[:, nonzero_view] /= view_norms[nonzero_view]
        Yview[:, ~nonzero_view] = 0.0
        patch_view_info.update({
            "input_dim": int(Ytr.shape[0]),
            "output_dim": int(Yview.shape[0]),
            "unlabeled_wl_topology_dim": int(wl_topology_dim),
            "rooted_shell_count_dim": int(rooted_count_dim),
            "explicit_ring_features": False,
            "atom_or_bond_label_features": False,
            "zero_view_train_patches": int((~nonzero_view).sum()),
        })
        print(
            f"patch view rooted_topology: {Ytr.shape[0]} -> {Yview.shape[0]} "
            "(no explicit ring or chemistry labels)",
            flush=True,
        )
    elif args.patch_view == "center_residual_no_ring":
        # Retain labeled topology/chemistry and centered shell summaries, but
        # remove explicit cycle statistics and the center atom one-hot itself.
        # Conditioning on the center type then removes its train-set mean
        # context, so KSVD models deviations specific to that atom chemistry.
        patch_view_indices = np.concatenate([
            np.arange(labeled_base_dim, dtype=np.int64),
            np.arange(center_end, Ytr.shape[0], dtype=np.int64),
        ])
        Ycontext = Ytr[patch_view_indices].copy()

        def center_keys(matrix: np.ndarray) -> np.ndarray:
            rows = []
            offset = center_start
            for width in center_dims:
                rows.append(np.argmax(matrix[offset : offset + width], axis=0))
                offset += width
            return np.stack(rows, axis=1).astype(np.int64)

        train_keys = center_keys(Ytr)
        patch_condition_keys, inverse = np.unique(
            train_keys, axis=0, return_inverse=True
        )
        patch_condition_means = np.zeros(
            (len(patch_condition_keys), Ycontext.shape[0]), dtype=np.float64
        )
        condition_counts = np.bincount(
            inverse, minlength=len(patch_condition_keys)
        ).astype(np.int64)
        for condition_i in range(len(patch_condition_keys)):
            patch_condition_means[condition_i] = Ycontext[
                :, inverse == condition_i
            ].mean(axis=1)
        patch_condition_global_mean = Ycontext.mean(axis=1)
        patch_condition_lookup = {
            tuple(int(x) for x in key): i
            for i, key in enumerate(patch_condition_keys.tolist())
        }
        Yview = Ycontext - patch_condition_means[inverse].T
        view_norms = np.linalg.norm(Yview, axis=0)
        nonzero_view = view_norms > 1e-12
        Yview[:, nonzero_view] /= view_norms[nonzero_view]
        Yview[:, ~nonzero_view] = 0.0
        patch_view_info.update({
            "input_dim": int(Ytr.shape[0]),
            "output_dim": int(Yview.shape[0]),
            "n_center_conditions": int(len(patch_condition_keys)),
            "min_condition_count": int(condition_counts.min()),
            "median_condition_count": float(np.median(condition_counts)),
            "max_condition_count": int(condition_counts.max()),
            "singleton_condition_count": int((condition_counts == 1).sum()),
            "explicit_ring_features": False,
            "center_atom_one_hot_in_dictionary_input": False,
            "conditioning": "train-only exact OGB center atom tuple mean residual",
            "zero_view_train_patches": int((~nonzero_view).sum()),
        })
        print(
            f"patch view center_residual_no_ring: {Ytr.shape[0]} -> "
            f"{Yview.shape[0]}, center conditions={len(patch_condition_keys)}",
            flush=True,
        )

    # The historical metric runs KSVD directly in the normalized patch vector
    # space.  A train-only PCA metric tests whether noisy/correlated
    # coordinates, rather than sparse coding itself, limit scaffold transfer.
    # The transform is fit solely on the fold's dictionary-fit patches and is
    # frozen before any held-out graph is encoded.
    patch_transform_mean: np.ndarray | None = None
    patch_transform_basis: np.ndarray | None = None
    patch_transform_scale: np.ndarray | None = None
    patch_ae_weight: np.ndarray | None = None
    patch_ae_bias: np.ndarray | None = None
    patch_ae_latent_mean: np.ndarray | None = None
    patch_metric_info: dict[str, Any] = {"name": args.patch_metric}
    Ydict = Yview
    if args.patch_metric in {"pca", "pca_whiten"}:
        latent_dim = min(
            int(args.patch_latent_dim), Yview.shape[0], Yview.shape[1] - 1
        )
        if latent_dim <= 0:
            raise RuntimeError("not enough train patches for latent patch metric")
        patch_transform_mean = Yview.mean(axis=1)
        centered = Yview - patch_transform_mean[:, None]
        basis, singular_values, _ = randomized_svd(
            centered, n_components=latent_dim, n_iter=7,
            random_state=args.dict_seed + 104729,
        )
        patch_transform_basis = np.asarray(basis, dtype=np.float64)
        projected = patch_transform_basis.T @ centered
        patch_transform_scale = np.ones(latent_dim, dtype=np.float64)
        if args.patch_metric == "pca_whiten":
            floor = max(
                float(singular_values[0]) * args.patch_whiten_floor, 1e-12
            )
            patch_transform_scale = (
                np.sqrt(max(Yview.shape[1] - 1, 1))
                / np.maximum(singular_values, floor)
            )
            projected *= patch_transform_scale[:, None]
        norms = np.linalg.norm(projected, axis=0)
        nonzero = norms > 1e-12
        projected[:, nonzero] /= norms[nonzero]
        projected[:, ~nonzero] = 0.0
        Ydict = projected
        total_centered_energy = float(np.square(centered).sum())
        retained_energy = float(np.square(singular_values).sum())
        patch_metric_info.update({
            "latent_dim": int(latent_dim),
            "randomized_svd_seed": int(args.dict_seed + 104729),
            "retained_variance_fraction": (
                retained_energy / max(total_centered_energy, 1e-12)
            ),
            "singular_value_max": float(singular_values[0]),
            "singular_value_min": float(singular_values[-1]),
            "whiten_scale_min": float(patch_transform_scale.min()),
            "whiten_scale_max": float(patch_transform_scale.max()),
            "zero_projected_train_patches": int((~nonzero).sum()),
        })
        print(
            f"patch metric {args.patch_metric}: {Yview.shape[0]} -> "
            f"{latent_dim}, retained variance="
            f"{patch_metric_info['retained_variance_fraction']:.4f}",
            flush=True,
        )
    elif args.patch_metric in {"masked_autoencoder", "masked_block_autoencoder"}:
        latent_dim = min(int(args.patch_latent_dim), Yview.shape[0])
        if latent_dim <= 0:
            raise RuntimeError("invalid autoencoder latent dimension")
        # A deliberately small nonlinear denoising bottleneck.  It sees no HIV
        # labels and, with --patch-view no_ring, no manually enumerated cycle
        # statistics.  KSVD is fit only after the encoder is frozen.
        torch.manual_seed(args.dict_seed + 130363)
        x_train = torch.tensor(Yview.T, dtype=torch.float32)
        encoder = torch.nn.Linear(Yview.shape[0], latent_dim)
        decoder = torch.nn.Linear(latent_dim, Yview.shape[0])
        torch.nn.init.xavier_uniform_(encoder.weight)
        torch.nn.init.zeros_(encoder.bias)
        torch.nn.init.xavier_uniform_(decoder.weight)
        torch.nn.init.zeros_(decoder.bias)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=args.patch_ae_lr, weight_decay=args.patch_ae_weight_decay,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.dict_seed + 433494437)
        block_slices: list[tuple[str, int, int]] = []
        if args.patch_metric == "masked_block_autoencoder":
            if args.patch_view != "no_ring":
                raise ValueError(
                    "masked_block_autoencoder currently requires --patch-view no_ring"
                )
            # The no-ring view preserves this exact semantic ordering.  Mask one
            # complete block per sample so corruption removes actual chemistry
            # or topology rather than mostly selecting the 92% zero coordinates.
            block_widths = [
                ("unlabeled_topology_wl", wl_topology_dim),
                ("labeled_chemistry_wl", labeled_base_dim - wl_topology_dim),
                ("center_atom", sum(center_dims)),
                ("shell_atom", (args.radius + 1) * 32),
                ("shell_bond", (args.radius + 1) * 16),
                ("rooted_counts", rooted_count_dim),
            ]
            block_start = 0
            for block_name, block_width in block_widths:
                block_slices.append(
                    (block_name, block_start, block_start + block_width)
                )
                block_start += block_width
            if block_start != Yview.shape[0]:
                raise AssertionError(
                    f"semantic block width {block_start} != view dim {Yview.shape[0]}"
                )
        loss_curve: list[float] = []
        clean_curve: list[float] = []
        n_train = int(x_train.shape[0])
        for epoch in range(1, args.patch_ae_epochs + 1):
            order = torch.randperm(n_train, generator=generator)
            epoch_loss = 0.0
            seen = 0
            encoder.train(); decoder.train()
            for lo in range(0, n_train, args.patch_ae_batch_size):
                idx = order[lo : lo + args.patch_ae_batch_size]
                target = x_train[idx]
                keep = torch.ones_like(target, dtype=torch.bool)
                if block_slices:
                    block_ids = torch.randint(
                        len(block_slices), (len(idx),), generator=generator
                    )
                    for block_i, (_, start, stop) in enumerate(block_slices):
                        rows = block_ids == block_i
                        if bool(rows.any()):
                            keep[rows, start:stop] = False
                if args.patch_ae_mask_prob > 0.0:
                    keep &= torch.rand(
                        target.shape, generator=generator, dtype=target.dtype
                    ) >= args.patch_ae_mask_prob
                corrupted = target * keep
                latent = F.gelu(encoder(corrupted), approximate="tanh")
                reconstruction = decoder(latent)
                mse = F.mse_loss(reconstruction, target)
                cosine_loss = 1.0 - F.cosine_similarity(
                    reconstruction, target, dim=-1, eps=1e-8
                ).mean()
                loss = mse + 0.05 * cosine_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(idx)
                seen += len(idx)
            loss_curve.append(epoch_loss / max(seen, 1))
            encoder.eval(); decoder.eval()
            with torch.no_grad():
                clean_latent = F.gelu(encoder(x_train), approximate="tanh")
                clean_reconstruction = decoder(clean_latent)
                clean_mse = float(F.mse_loss(clean_reconstruction, x_train))
            clean_curve.append(clean_mse)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.patch_ae_epochs:
                print(
                    f"patch AE ep={epoch:02d} denoise={loss_curve[-1]:.6f} "
                    f"clean_mse={clean_mse:.6f}", flush=True,
                )
        with torch.no_grad():
            latent_train = F.gelu(encoder(x_train), approximate="tanh").numpy()
        patch_ae_latent_mean = latent_train.mean(axis=0).astype(np.float64)
        latent_train = latent_train.astype(np.float64) - patch_ae_latent_mean
        latent_norms = np.linalg.norm(latent_train, axis=1)
        nonzero = latent_norms > 1e-12
        latent_train[nonzero] /= latent_norms[nonzero, None]
        latent_train[~nonzero] = 0.0
        Ydict = latent_train.T
        patch_ae_weight = encoder.weight.detach().numpy().astype(np.float64)
        patch_ae_bias = encoder.bias.detach().numpy().astype(np.float64)
        patch_metric_info.update({
            "latent_dim": int(latent_dim),
            "fit_seed": int(args.dict_seed + 130363),
            "mask_seed": int(args.dict_seed + 433494437),
            "epochs": int(args.patch_ae_epochs),
            "batch_size": int(args.patch_ae_batch_size),
            "learning_rate": float(args.patch_ae_lr),
            "mask_probability": float(args.patch_ae_mask_prob),
            "weight_decay": float(args.patch_ae_weight_decay),
            "corruption_mode": (
                "semantic_block_plus_feature_mask"
                if block_slices else "independent_feature_mask"
            ),
            "semantic_blocks": [
                {"name": name, "start": int(start), "stop": int(stop)}
                for name, start, stop in block_slices
            ],
            "train_nonzero_coordinate_fraction": float(
                np.count_nonzero(Yview) / max(Yview.size, 1)
            ),
            "trainable_parameters": int(
                sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))
            ),
            "denoising_loss_curve": loss_curve,
            "clean_mse_curve": clean_curve,
            "zero_latent_train_patches": int((~nonzero).sum()),
        })
        print(
            f"patch metric {args.patch_metric}: {Yview.shape[0]} -> "
            f"{latent_dim}; final clean_mse={clean_curve[-1]:.6f}",
            flush=True,
        )

    def transform_patch(y: np.ndarray) -> np.ndarray:
        nonlocal unknown_condition_count
        original_y = y
        if patch_view_indices is not None:
            y = y[patch_view_indices]
            if patch_condition_means is not None:
                assert patch_condition_global_mean is not None
                key_values = []
                offset = center_start
                for width in center_dims:
                    key_values.append(int(np.argmax(original_y[offset : offset + width])))
                    offset += width
                condition_i = patch_condition_lookup.get(tuple(key_values))
                if condition_i is None:
                    y = y - patch_condition_global_mean
                    unknown_condition_count += 1
                else:
                    y = y - patch_condition_means[condition_i]
            y = y / max(float(np.linalg.norm(y)), 1e-12)
        if patch_ae_weight is not None:
            assert patch_ae_bias is not None
            assert patch_ae_latent_mean is not None
            pre = patch_ae_weight @ y + patch_ae_bias
            # Match torch GELU(approximate="tanh") used during fitting.
            z = 0.5 * pre * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (pre + 0.044715 * np.power(pre, 3))
            ))
            z = z - patch_ae_latent_mean
            return z / max(float(np.linalg.norm(z)), 1e-12)
        if patch_transform_basis is None:
            return y
        assert patch_transform_mean is not None
        assert patch_transform_scale is not None
        z = patch_transform_basis.T @ (y - patch_transform_mean)
        z = z * patch_transform_scale
        norm = float(np.linalg.norm(z))
        return z / max(norm, 1e-12)

    dictionaries: dict[str, np.ndarray] = {}
    dictionary_info: dict[str, Any] = {}
    mixture_slices: dict[str, list[slice]] = {}
    clustered_families = {"ksvd_cluster_balanced", "ksvd_mixture"} & set(families)
    cluster_labels: np.ndarray | None = None
    cluster_info: dict[str, Any] | None = None
    if clustered_families:
        cluster_labels, cluster_info = _cluster_train_patches(
            Ydict, n_clusters=args.ksvd_clusters, seed=args.dict_seed
        )
        print(
            f"learned {args.ksvd_clusters} train-only patch domains: "
            f"{cluster_info['cluster_counts']}",
            flush=True,
        )
    if "random_patch" in families:
        dictionaries["random_patch"] = _random_patch_dictionary(Ydict, args.n_atoms, args.dict_seed)
        dictionary_info["random_patch"] = {"matched_column_draw_seed": args.dict_seed}
    if "pca" in families:
        U, _, _ = randomized_svd(
            Ydict,
            n_components=args.n_atoms,
            n_iter=5,
            random_state=args.dict_seed,
        )
        dictionaries["pca"] = np.asarray(U, dtype=np.float64)
        dictionary_info["pca"] = {"randomized_svd_seed": args.dict_seed}
    if "ksvd" in families:
        if args.ksvd_external_residual_atoms > 0:
            Ycore = Ydict[:, :inner_patch_count_for_dictionary]
            Yexternal = Ydict[:, inner_patch_count_for_dictionary:]
            if Yexternal.shape[1] != args.external_official_train_patches:
                raise ValueError(
                    "external transformed patch count mismatch: "
                    f"{Yexternal.shape[1]} vs "
                    f"{args.external_official_train_patches}"
                )
            Dcore, _, core_info = ksvd(
                Ycore,
                n_atoms=args.n_atoms,
                T=args.sparsity,
                T_min=1,
                n_iter=args.ksvd_iter,
                seed=args.dict_seed,
            )
            external_codes = np.column_stack([
                _omp(Dcore, Yexternal[:, j], args.sparsity)
                for j in range(Yexternal.shape[1])
            ])
            residual = Yexternal - Dcore @ external_codes
            residual_norms = np.linalg.norm(residual, axis=0)
            usable = residual_norms > 1e-10
            if int(usable.sum()) < args.ksvd_external_residual_atoms:
                raise RuntimeError(
                    "too few nonzero external residuals for requested atoms"
                )
            residual_train = residual[:, usable].copy()
            residual_train /= residual_norms[usable][None, :]
            residual_sparsity = min(
                args.sparsity, args.ksvd_external_residual_atoms
            )
            Dresidual, _, residual_info = ksvd(
                residual_train,
                n_atoms=args.ksvd_external_residual_atoms,
                T=residual_sparsity,
                T_min=1,
                n_iter=args.ksvd_iter,
                seed=args.dict_seed + 67867967,
            )
            D = np.concatenate([Dcore, Dresidual], axis=1)
            dictionaries["ksvd"] = D
            dictionary_info["ksvd"] = {
                "mode": "inner_core_plus_external_residual_atoms",
                "core_atoms": int(args.n_atoms),
                "external_residual_atoms": int(
                    args.ksvd_external_residual_atoms
                ),
                "total_atoms": int(D.shape[1]),
                "inner_core_patch_count": int(Ycore.shape[1]),
                "external_patch_count": int(Yexternal.shape[1]),
                "usable_external_residual_count": int(usable.sum()),
                "external_core_reconstruction_mse": float(
                    np.mean(np.square(residual))
                ),
                "external_core_relative_residual_norm_mean": float(
                    residual_norms.mean()
                ),
                "core_ksvd": core_info,
                "external_residual_ksvd": residual_info,
                "external_residual_seed": int(args.dict_seed + 67867967),
            }
            fit_protocol += (
                f"+ksvd-inner-core-external-residual-{args.ksvd_external_residual_atoms}"
            )
        else:
            D, _, info = ksvd(
                Ydict,
                n_atoms=args.n_atoms,
                T=args.sparsity,
                T_min=1,
                n_iter=args.ksvd_iter,
                seed=args.dict_seed,
            )
            dictionaries["ksvd"] = D
            dictionary_info["ksvd"] = info
    if "ksvd_cluster_balanced" in families:
        assert cluster_labels is not None and cluster_info is not None
        Ybal, balance_info = _cluster_balanced_columns(
            Ydict, cluster_labels, args.ksvd_clusters, args.dict_seed
        )
        D, _, info = ksvd(
            Ybal, n_atoms=args.n_atoms, T=args.sparsity, T_min=1,
            n_iter=args.ksvd_iter, seed=args.dict_seed,
        )
        dictionaries["ksvd_cluster_balanced"] = D
        dictionary_info["ksvd_cluster_balanced"] = {
            "cluster_model": cluster_info,
            "balancing": balance_info,
            "ksvd": info,
        }
    if "ksvd_mixture" in families:
        assert cluster_labels is not None and cluster_info is not None
        D, slices, info = _fit_ksvd_mixture(
            Ydict, cluster_labels, args.ksvd_clusters, args.n_atoms,
            args.sparsity, args.ksvd_iter, args.dict_seed,
        )
        dictionaries["ksvd_mixture"] = D
        mixture_slices["ksvd_mixture"] = slices
        dictionary_info["ksvd_mixture"] = {
            "cluster_model": cluster_info,
            **info,
        }
    print("learned dictionaries " + ", ".join(f"{k}:{v.shape}" for k, v in dictionaries.items()), flush=True)

    offsets = graph_node_offsets(graphs)
    total_nodes = int(offsets[-1])
    tokens = {
        family: np.zeros((total_nodes, D.shape[1]), dtype=np.float32)
        for family, D in dictionaries.items()
    }
    # Fold-development caches encode all official-train graphs so the held-out
    # inner fold can be scored, but never expose those patches to dictionary fit.
    encode_indices = (
        np.concatenate([tr, va]) if args.encode_official_valid else tr.copy()
    )
    encoded_nodes = 0
    mixture_route_counts = {
        family: np.zeros(len(slices), dtype=np.int64)
        for family, slices in mixture_slices.items()
    }
    mixture_error_sums = {family: 0.0 for family in mixture_slices}
    for count, raw_i in enumerate(encode_indices, 1):
        i = int(raw_i)
        g = graphs[i]
        for u in g.nodes:
            vec = centered_ego_vector(
                g,
                int(u),
                bundle.node_feats[i],
                bundle.edge_feats[i],
                radius=args.radius,
                max_nodes=args.max_nodes,
            )
            y = np.asarray(vec, dtype=np.float64)
            y /= max(float(np.linalg.norm(y)), 1e-12)
            y = transform_patch(y)
            row = int(offsets[i] + u)
            for family, D in dictionaries.items():
                if family in mixture_slices:
                    code, route, error = _encode_ksvd_mixture(
                        D, y, mixture_slices[family], args.sparsity
                    )
                    tokens[family][row] = code.astype(np.float32)
                    mixture_route_counts[family][route] += 1
                    mixture_error_sums[family] += error
                else:
                    tokens[family][row] = _omp(D, y, args.sparsity).astype(np.float32)
            encoded_nodes += 1
        if count % 500 == 0 or count == len(encode_indices):
            print(f"encoded graphs {count}/{len(encode_indices)}; nodes={encoded_nodes}", flush=True)

    for family, counts in mixture_route_counts.items():
        dictionary_info[family]["encoded_route_counts"] = counts.tolist()
        dictionary_info[family]["encoded_route_fractions"] = (counts / counts.sum()).tolist()
        dictionary_info[family]["encoded_mean_squared_reconstruction_error"] = float(
            mixture_error_sums[family] / max(int(counts.sum()), 1)
        )

    # Executable isolation invariant: official-test token rows are untouched.
    for i in te:
        lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
        for family in families:
            if np.any(tokens[family][lo:hi] != 0):
                raise AssertionError(f"{family} encoded official-test nodes")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {
        "offsets": offsets,
        "original_indices": np.asarray(bundle.meta["original_indices"], dtype=np.int64),
        "train_indices": tr,
        "valid_indices": va,
        "test_indices": te,
    }
    if patch_transform_basis is not None:
        assert patch_transform_mean is not None
        assert patch_transform_scale is not None
        archive["patch_transform_mean"] = patch_transform_mean.astype(np.float32)
        archive["patch_transform_basis"] = patch_transform_basis.astype(np.float32)
        archive["patch_transform_scale"] = patch_transform_scale.astype(np.float32)
    if patch_ae_weight is not None:
        assert patch_ae_bias is not None
        assert patch_ae_latent_mean is not None
        archive["patch_ae_weight"] = patch_ae_weight.astype(np.float32)
        archive["patch_ae_bias"] = patch_ae_bias.astype(np.float32)
        archive["patch_ae_latent_mean"] = patch_ae_latent_mean.astype(np.float32)
    if patch_condition_keys is not None:
        assert patch_condition_means is not None
        assert patch_condition_global_mean is not None
        archive["patch_condition_keys"] = patch_condition_keys.astype(np.int64)
        archive["patch_condition_means"] = patch_condition_means.astype(np.float32)
        archive["patch_condition_global_mean"] = patch_condition_global_mean.astype(np.float32)
    for family, D in dictionaries.items():
        archive[f"dictionary_{family}"] = D.astype(np.float32)
        archive[f"tokens_{family}"] = tokens[family]
    np.savez_compressed(output, **archive)
    if patch_condition_keys is not None:
        patch_view_info["unknown_encoded_center_count"] = int(unknown_condition_count)
    meta = {
        "protocol_id": "molhiv-localized-ksvd-node-tokens-v1",
        "test_policy": "official test node patches were not vectorized or sparse-coded",
        "config": vars(args),
        "data_meta": bundle.meta,
        "patch_stats": patch_stats,
        "patch_view": patch_view_info,
        "patch_metric": patch_metric_info,
        "dictionary_fit_protocol": fit_protocol,
        "dictionary_fit_n_graphs": int(len(fit_indices)),
        "dictionary_fit_n_external_graphs": int(
            0 if external_patch_stats is None
            else external_patch_stats["selected_external_graph_count"]
        ),
        "external_official_train_audit": external_patch_stats,
        "dictionary_fit_indices_sha256": __import__("hashlib").sha256(
            np.asarray(fit_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "encoded_official_valid": bool(args.encode_official_valid),
        "dictionary_info": dictionary_info,
        "families": families,
        "total_nodes": total_nodes,
        "encoded_nodes": encoded_nodes,
        "encoded_train_valid_nodes": encoded_nodes,
        "elapsed_sec": time.time() - t0,
        "archive": str(output),
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {output} and {meta_path}", flush=True)


if __name__ == "__main__":
    main()
