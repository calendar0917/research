"""Mechanism gate for relation-regularized K-SVD on real MolHIV patches.

The gate is intentionally upstream of graph-label classification.  All methods
receive identical chemistry-aware patches and emit the same code dimension.
Held-out relation probes see content-only codes; no held-out relation is used
while encoding a patch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig, bundle_to_Y, sample_patches_graph_level
from .ksvd import _omp, ksvd
from .relational_ksvd import relational_ksvd, sparse_codes


RELATION_NAMES = ("either", "overlap", "bridge_bond")


def _relations(g, patches: list[set[int]]) -> np.ndarray:
    p = len(patches)
    out = np.zeros((p, p, len(RELATION_NAMES)), dtype=np.float64)
    for i in range(p):
        for j in range(i + 1, p):
            left, right = patches[i], patches[j]
            overlap = bool(left & right)
            left_only, right_only = left - right, right - left
            bridge = any(v in right_only for u in left_only for v in g.neighbors(u))
            values = np.array([overlap or bridge, overlap, bridge], dtype=np.float64)
            out[i, j] = out[j, i] = values
    return out


def _graph_patches(g, cfg, node_feat, edge_feat):
    bundle, _ = sample_patches_graph_level(g, cfg, seed=cfg.seed)
    vectors: list[np.ndarray] = []
    patches: list[set[int]] = []
    for patch in bundle.node_sets[: cfg.max_patches_per_graph]:
        Y, meta = bundle_to_Y(
            g,
            SimpleNamespace(node_sets=[patch]),
            cfg.max_nodes,
            cfg.order_mode,
            patch_feat=cfg.patch_feat,
            node_feat=node_feat,
            edge_feat=edge_feat,
        )
        if meta["n_cols"]:
            y = Y[:, 0]
            norm = float(np.linalg.norm(y))
            if norm > 1e-12:
                vectors.append(y / norm)
                patches.append(set(patch))
    if len(vectors) < 2:
        return None
    return np.stack(vectors, axis=1), _relations(g, patches)


def _collect(bundle, indices, cfg, max_graphs: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(idx)
    graphs = []
    for raw in idx:
        i = int(raw)
        item = _graph_patches(
            bundle.graphs[i], cfg, bundle.node_feats[i], bundle.edge_feats[i]
        )
        if item is not None:
            graphs.append(item)
        if len(graphs) >= max_graphs:
            break
    if not graphs:
        raise RuntimeError("no patch graphs were collected")
    return graphs


def _flatten_training(graphs, shuffle: bool, seed: int):
    columns = []
    pair_index = []
    labels = []
    offset = 0
    rng = np.random.default_rng(seed)
    for Y, relation in graphs:
        p = Y.shape[1]
        columns.append(Y)
        permutation = rng.permutation(p) if shuffle else np.arange(p)
        rel = relation[np.ix_(permutation, permutation)]
        for i in range(p):
            for j in range(i + 1, p):
                pair_index.append((offset + i, offset + j))
                labels.append(rel[i, j])
        offset += p
    return (
        np.concatenate(columns, axis=1),
        np.asarray(pair_index, dtype=np.int64),
        np.asarray(labels, dtype=np.float64),
    )


def _pair_dataset(graphs, encoder):
    features = []
    labels = []
    content_distances = []
    for Y, relation in graphs:
        Z = np.asarray(encoder(Y), dtype=np.float64).T
        for i in range(Z.shape[0]):
            for j in range(i + 1, Z.shape[0]):
                features.append(np.concatenate([np.abs(Z[i] - Z[j]), Z[i] * Z[j]]))
                labels.append(relation[i, j])
                content_distances.append(float(np.linalg.norm(Y[:, i] - Y[:, j])))
    return np.asarray(features), np.asarray(labels), np.asarray(content_distances)


def _matched_indices(distance, labels, bin_edges, seed):
    rng = np.random.default_rng(seed)
    bins = np.clip(np.digitize(distance, bin_edges[1:-1]), 0, len(bin_edges) - 2)
    selected = []
    for bin_id in range(len(bin_edges) - 1):
        inside = np.where(bins == bin_id)[0]
        positive = inside[labels[inside] > 0.5]
        negative = inside[labels[inside] <= 0.5]
        take = min(len(positive), len(negative))
        if take:
            selected.extend(rng.choice(positive, size=take, replace=False).tolist())
            selected.extend(rng.choice(negative, size=take, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def _probe(Xtr, ytr, dtr, Xva, yva, dva):
    rows = {}
    for r, name in enumerate(RELATION_NAMES):
        if len(np.unique(ytr[:, r])) < 2 or len(np.unique(yva[:, r])) < 2:
            rows[name] = {"roc_auc": float("nan"), "average_precision": float("nan")}
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=0, C=1.0
            ),
        )
        model.fit(Xtr, ytr[:, r])
        score = model.predict_proba(Xva)[:, 1]
        rows[name] = {
            "roc_auc": float(roc_auc_score(yva[:, r], score)),
            "average_precision": float(average_precision_score(yva[:, r], score)),
            "valid_positive_rate": float(yva[:, r].mean()),
        }
        # Harder diagnostic: balance positives and negatives separately inside
        # raw-content-distance bins.  A method that only detects that overlapping
        # patches look similar should lose most of its advantage here.
        edges = np.unique(np.quantile(dtr, np.linspace(0.0, 1.0, 11)))
        if len(edges) >= 3:
            itr = _matched_indices(dtr, ytr[:, r], edges, seed=100 + r)
            iva = _matched_indices(dva, yva[:, r], edges, seed=200 + r)
        else:
            itr = iva = np.empty(0, dtype=np.int64)
        if len(itr) and len(iva):
            matched_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=0, C=1.0
                ),
            )
            matched_model.fit(Xtr[itr], ytr[itr, r])
            matched_score = matched_model.predict_proba(Xva[iva])[:, 1]
            rows[name]["content_matched_roc_auc"] = float(
                roc_auc_score(yva[iva, r], matched_score)
            )
            rows[name]["content_matched_average_precision"] = float(
                average_precision_score(yva[iva, r], matched_score)
            )
            rows[name]["content_matched_valid_pairs"] = int(len(iva))
        else:
            rows[name]["content_matched_roc_auc"] = float("nan")
            rows[name]["content_matched_average_precision"] = float("nan")
            rows[name]["content_matched_valid_pairs"] = 0
    rows["mean_roc_auc"] = float(
        np.nanmean([rows[name]["roc_auc"] for name in RELATION_NAMES])
    )
    rows["mean_average_precision"] = float(
        np.nanmean([rows[name]["average_precision"] for name in RELATION_NAMES])
    )
    rows["mean_content_matched_roc_auc"] = float(
        np.nanmean([rows[name]["content_matched_roc_auc"] for name in RELATION_NAMES])
    )
    rows["mean_content_matched_average_precision"] = float(
        np.nanmean(
            [rows[name]["content_matched_average_precision"] for name in RELATION_NAMES]
        )
    )
    return rows


def _dictionary_reconstruction(graphs, D, sparsity):
    errs = []
    for Y, _ in graphs:
        X = sparse_codes(D, Y, sparsity)
        errs.append(np.linalg.norm(Y - D @ X) ** 2)
    denom = sum(np.linalg.norm(Y) ** 2 for Y, _ in graphs)
    return float(np.sqrt(sum(errs) / max(denom, 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold-cache",
        default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fit-graphs", type=int, default=600)
    ap.add_argument("--eval-graphs", type=int, default=600)
    ap.add_argument("--n-atoms", type=int, default=12)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--relation-weight", type=float, default=0.1)
    ap.add_argument("--outer-iter", type=int, default=3)
    ap.add_argument("--code-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    started = time.time()
    bundle = load_molhiv(with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV atom and bond features are required")
    fold_cache = np.load(args.fold_cache)
    train_idx = fold_cache[f"fold_{args.fold}_train_indices"]
    valid_idx = fold_cache[f"fold_{args.fold}_valid_indices"]
    cfg = GraphLevelConfig(
        seed=0,
        max_patches_per_graph=8,
        patch_feat="wl_chem_ring",
        normalize_patches=True,
    )
    train_graphs = _collect(bundle, train_idx, cfg, args.fit_graphs, args.seed + 11)
    valid_graphs = _collect(bundle, valid_idx, cfg, args.eval_graphs, args.seed + 29)
    Ytr, pair_index, pair_labels = _flatten_training(
        train_graphs, shuffle=False, seed=args.seed
    )
    _, _, shuffled_labels = _flatten_training(
        train_graphs, shuffle=True, seed=args.seed + 101
    )

    ordinary_D, _, ordinary_info = ksvd(
        Ytr,
        n_atoms=args.n_atoms,
        T=args.sparsity,
        T_min=1,
        n_iter=4,
        seed=args.seed,
    )
    relational_D, relational_info = relational_ksvd(
        Ytr,
        pair_index,
        pair_labels,
        n_atoms=args.n_atoms,
        sparsity=args.sparsity,
        relation_weight=args.relation_weight,
        outer_iter=args.outer_iter,
        code_steps=args.code_steps,
        seed=args.seed,
    )
    shuffled_D, shuffled_info = relational_ksvd(
        Ytr,
        pair_index,
        shuffled_labels,
        n_atoms=args.n_atoms,
        sparsity=args.sparsity,
        relation_weight=args.relation_weight,
        outer_iter=args.outer_iter,
        code_steps=args.code_steps,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(Ytr.shape[1], size=args.n_atoms, replace=Ytr.shape[1] < args.n_atoms)
    random_patch_D = Ytr[:, picked].copy()
    random_patch_D /= np.maximum(
        np.linalg.norm(random_patch_D, axis=0, keepdims=True), 1e-12
    )
    pca = PCA(n_components=args.n_atoms, random_state=args.seed).fit(Ytr.T)
    random_basis, _ = np.linalg.qr(rng.standard_normal((Ytr.shape[0], args.n_atoms)))

    encoders = {
        "pca": lambda Y: pca.transform(Y.T).T,
        "random_projection": lambda Y: random_basis.T @ Y,
        "random_patch_dictionary": lambda Y: sparse_codes(
            random_patch_D, Y, args.sparsity
        ),
        "ordinary_ksvd": lambda Y: sparse_codes(ordinary_D, Y, args.sparsity),
        "relational_ksvd": lambda Y: sparse_codes(relational_D, Y, args.sparsity),
        "shuffled_relational_ksvd": lambda Y: sparse_codes(
            shuffled_D, Y, args.sparsity
        ),
    }
    result = {
        "protocol_id": "molhiv-relational-ksvd-content-only-relation-gate-v1",
        "config": vars(args),
        "n_train_graphs": len(train_graphs),
        "n_valid_graphs": len(valid_graphs),
        "n_train_patches": int(Ytr.shape[1]),
        "n_train_pairs": int(pair_index.shape[0]),
        "relation_positive_rates": dict(
            zip(RELATION_NAMES, pair_labels.mean(axis=0).astype(float).tolist())
        ),
        "methods": {},
        "fit": {
            "ordinary": ordinary_info,
            "relational": relational_info,
            "shuffled_relational": shuffled_info,
        },
    }
    dictionary_map = {
        "random_patch_dictionary": random_patch_D,
        "ordinary_ksvd": ordinary_D,
        "relational_ksvd": relational_D,
        "shuffled_relational_ksvd": shuffled_D,
    }
    for name, encoder in encoders.items():
        Xtr, ytr, dtr = _pair_dataset(train_graphs, encoder)
        Xva, yva, dva = _pair_dataset(valid_graphs, encoder)
        row = {"relation_probe": _probe(Xtr, ytr, dtr, Xva, yva, dva)}
        if name in dictionary_map:
            row["valid_reconstruction_relative_error"] = _dictionary_reconstruction(
                valid_graphs, dictionary_map[name], args.sparsity
            )
        elif name == "pca":
            error = sum(
                np.linalg.norm(Y.T - pca.inverse_transform(pca.transform(Y.T))) ** 2
                for Y, _ in valid_graphs
            )
            denom = sum(np.linalg.norm(Y) ** 2 for Y, _ in valid_graphs)
            row["valid_reconstruction_relative_error"] = float(
                np.sqrt(error / max(denom, 1e-12))
            )
        else:
            error = sum(
                np.linalg.norm(Y - random_basis @ (random_basis.T @ Y)) ** 2
                for Y, _ in valid_graphs
            )
            denom = sum(np.linalg.norm(Y) ** 2 for Y, _ in valid_graphs)
            row["valid_reconstruction_relative_error"] = float(
                np.sqrt(error / max(denom, 1e-12))
            )
        result["methods"][name] = row
        print(name, json.dumps(row), flush=True)

    ordinary_auc = result["methods"]["ordinary_ksvd"]["relation_probe"]["mean_roc_auc"]
    relational_auc = result["methods"]["relational_ksvd"]["relation_probe"]["mean_roc_auc"]
    shuffled_auc = result["methods"]["shuffled_relational_ksvd"]["relation_probe"]["mean_roc_auc"]
    result["paired_summary"] = {
        "relational_minus_ordinary_mean_roc_auc": relational_auc - ordinary_auc,
        "relational_minus_shuffled_mean_roc_auc": relational_auc - shuffled_auc,
    }
    result["elapsed_sec"] = time.time() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
