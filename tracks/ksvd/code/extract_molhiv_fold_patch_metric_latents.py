"""Fit the raw-patch PCA metric on one outer fold and encode official-train only.

This is a lean replacement for building a KSVD token cache and then replaying
its PCA transform.  The current real-prototype route uses the train-fit patch
metric but not the KSVD dictionary, so this utility fits exactly the same
reservoir/PCA transform and records a direct fit-split audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.build_molhiv_node_tokens import _reservoir_train_patches
from code.data_molhiv import load_molhiv
from code.molhiv_node_tokens import centered_ego_vector, graph_node_offsets


def sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--metric-seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--max-train-patches", type=int, default=6000)
    ap.add_argument("--max-patches-per-graph", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.fold < 0 or min(args.max_nodes, args.latent_dim, args.max_train_patches) <= 0:
        raise ValueError("invalid fold/metric configuration")

    started = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("molecular features were not loaded")
    graphs = bundle.graphs
    offsets = graph_node_offsets(graphs)
    original_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        fold_original = np.asarray(folds["original_indices"], dtype=np.int64)
        fold_train = np.asarray(folds["official_train_indices"], dtype=np.int64)
    if not np.array_equal(fold_original, original_indices):
        raise ValueError("fold original indices mismatch")
    if not np.array_equal(fold_train, official_train):
        raise ValueError("fold official train mismatch")
    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("outer fit/heldout overlap")
    if set(np.concatenate([fit_indices, heldout_indices]).tolist()) != set(official_train.tolist()):
        raise AssertionError("outer fold does not partition official train")

    train_patches, patch_stats = _reservoir_train_patches(
        graphs,
        fit_indices,
        bundle.node_feats,
        bundle.edge_feats,
        radius=args.radius,
        max_nodes=args.max_nodes,
        capacity=args.max_train_patches,
        seed=args.metric_seed,
        max_patches_per_graph=args.max_patches_per_graph,
    )
    latent_dim = min(args.latent_dim, train_patches.shape[0], train_patches.shape[1] - 1)
    if latent_dim <= 0:
        raise RuntimeError("not enough patches for PCA metric")
    mean = train_patches.mean(axis=1)
    centered = train_patches - mean[:, None]
    basis, singular_values, _ = randomized_svd(
        centered,
        n_components=latent_dim,
        n_iter=7,
        random_state=args.metric_seed + 104729,
    )
    basis = np.asarray(basis, dtype=np.float64)
    scale = np.ones(latent_dim, dtype=np.float64)
    retained_variance = float(
        np.sum(singular_values ** 2) / max(np.sum(centered ** 2), 1e-20)
    )
    print(
        f"fit fold metric patches={train_patches.shape}, latent={latent_dim}, "
        f"retained_variance={retained_variance:.6f}",
        flush=True,
    )

    latents = np.zeros((int(offsets[-1]), latent_dim), dtype=np.float32)
    zero_norm = 0
    encoded_nodes = 0
    for count, raw_i in enumerate(official_train, start=1):
        i = int(raw_i)
        graph = graphs[i]
        for raw_u in graph.nodes:
            u = int(raw_u)
            y = np.asarray(
                centered_ego_vector(
                    graph,
                    u,
                    bundle.node_feats[i],
                    bundle.edge_feats[i],
                    radius=args.radius,
                    max_nodes=args.max_nodes,
                ),
                dtype=np.float64,
            )
            y /= max(float(np.linalg.norm(y)), 1e-12)
            z = basis.T @ (y - mean)
            z *= scale
            norm = float(np.linalg.norm(z))
            if norm <= 1e-12:
                zero_norm += 1
                z[:] = 0.0
            else:
                z /= norm
            latents[int(offsets[i]) + u] = z.astype(np.float32)
            encoded_nodes += 1
        if count % 500 == 0 or count == len(official_train):
            print(
                f"encoded outer-train graphs {count}/{len(official_train)}; nodes={encoded_nodes}",
                flush=True,
            )

    official_nontrain_rows = np.concatenate(
        [
            np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
            for i in np.concatenate([official_valid, official_test])
        ]
    )
    if np.any(latents[official_nontrain_rows] != 0):
        raise AssertionError("official valid/test rows were encoded")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        offsets=offsets,
        original_indices=original_indices,
        train_indices=official_train,
        valid_indices=official_valid,
        test_indices=official_test,
        fit_indices=fit_indices,
        heldout_indices=heldout_indices,
        patch_transform_mean=mean.astype(np.float32),
        patch_transform_basis=basis.astype(np.float32),
        patch_transform_scale=scale.astype(np.float32),
        latents=latents,
    )
    meta = {
        "protocol_id": "molhiv-direct-fold-fit-raw-patch-pca-latents-v1",
        "date": "2026-07-28",
        "representation": (
            "unit-normalized permutation-invariant radius-2 descriptor, then outer-fit-only "
            "PCA projection and unit normalization"
        ),
        "fit_policy": "PCA fitted only on the requested outer-fit fold",
        "fit_indices_sha256": sha256(fit_indices),
        "heldout_indices_sha256": sha256(heldout_indices),
        "supervised_gnn_layers": 0,
        "unsupervised_gnn_layers": 0,
        "used_labels": False,
        "official_valid_encoded": False,
        "official_test_encoded": False,
        "config": vars(args),
        "n_graphs": int(len(graphs)),
        "n_official_train_graphs": int(len(official_train)),
        "n_outer_fit_graphs": int(len(fit_indices)),
        "n_outer_heldout_graphs": int(len(heldout_indices)),
        "n_encoded_nodes": int(encoded_nodes),
        "latent_dim": int(latent_dim),
        "zero_norm_nodes": int(zero_norm),
        "retained_variance": retained_variance,
        "patch_stats": patch_stats,
        "offsets_sha256": sha256(offsets),
        "transform_mean_sha256": sha256(mean.astype(np.float32)),
        "transform_basis_sha256": sha256(basis.astype(np.float32)),
        "latents_sha256": sha256(latents),
        "elapsed_sec": time.time() - started,
        "output": str(output),
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
