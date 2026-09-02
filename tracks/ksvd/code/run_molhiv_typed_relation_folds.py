"""Chemistry-typed patch relations on fixed official-train scaffold folds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data_molhiv import load_molhiv
from .graph_level import (
    GraphLevelConfig,
    bundle_to_Y,
    learn_shared_D_graph_level,
    sparse_code_patch_matrix,
    sparse_code_readouts,
    sample_patches_graph_level,
)
from .run_molhiv_next_round import fit_auc, size_feat


ATOM_BINS = 16
BOND_BINS = 8


def _typed_relation(
    g,
    patches,
    X: np.ndarray,
    node_feat: np.ndarray,
    edge_feat: dict[tuple[int, int], np.ndarray],
    shuffle_seed: int | None = None,
) -> np.ndarray:
    """Return compact relation summaries for geometry, atom, and bond channels."""
    A = np.abs(np.asarray(X, dtype=np.float64))
    k, n = A.shape
    n_channels = 1 + ATOM_BINS + BOND_BINS
    W = np.zeros((n_channels, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = patches[i], patches[j]
            overlap = sorted(si & sj)
            cross: list[tuple[int, int, int]] = []
            for u in si:
                for v in g.neighbors(u):
                    if v in sj and u < v:
                        key = (u, v)
                        ef = edge_feat.get(key)
                        bt = int(ef[0]) % BOND_BINS if ef is not None and ef.size else 0
                        cross.append((u, v, bt))
            if not overlap and not cross:
                continue
            base = len(overlap) / max(1, min(len(si), len(sj)))
            base += len(cross) / max(1, len(si) + len(sj))
            W[0, i, j] = W[0, j, i] = base
            for u in overlap:
                at = int(node_feat[u, 0]) % ATOM_BINS
                w = 1.0 / max(1, min(len(si), len(sj)))
                W[1 + at, i, j] += w
                W[1 + at, j, i] += w
            for _, _, bt in cross:
                w = 1.0 / max(1, len(si) + len(sj))
                W[1 + ATOM_BINS + bt, i, j] += w
                W[1 + ATOM_BINS + bt, j, i] += w
    if shuffle_seed is not None and n > 1:
        p = np.random.default_rng(shuffle_seed).permutation(n)
        W = W[:, p][:, :, p]

    feats = []
    for c in range(n_channels):
        total = float(W[c].sum())
        if total <= 1e-12:
            feats.append(np.zeros(3 * k, dtype=np.float64))
            continue
        R = (A @ W[c] @ A.T) / total
        R = 0.5 * (R + R.T)
        feats.append(np.concatenate([np.diag(R), R.sum(axis=1), np.linalg.eigvalsh(R)]))
    return np.concatenate(feats)


def _encode_one(g, D, cfg, node_feat, edge_feat, shuffle_seed=None):
    bundle, _ = sample_patches_graph_level(g, cfg, seed=cfg.seed)
    Y, _ = bundle_to_Y(
        g, bundle, cfg.max_nodes, cfg.order_mode,
        patch_feat=cfg.patch_feat, node_feat=node_feat, edge_feat=edge_feat,
    )
    _, X = sparse_code_patch_matrix(Y, D, cfg)
    bag = sparse_code_readouts(X)["rich_no_recon"]
    patches = bundle.node_sets[: X.shape[1]]
    rel = _typed_relation(g, patches, X, node_feat, edge_feat)
    shuf = _typed_relation(g, patches, X, node_feat, edge_feat, shuffle_seed=shuffle_seed)
    return bag, rel, shuf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--n-atoms", type=int, default=8)
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument("--max-train-patches", type=int, default=8000)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    bundle = load_molhiv(with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV atom and bond features are required")
    z = np.load(args.fold_cache)
    folds = [args.fold] if args.fold >= 0 else [0, 1, 2]
    cfg = GraphLevelConfig(
        n_atoms=args.n_atoms, T=args.sparsity, T_min=1, ksvd_iter=args.ksvd_iter,
        seed=0, max_train_patches=args.max_train_patches, max_patches_per_graph=8,
        patch_feat="wl_chem_ring", normalize_patches=True, readout_mode="pool", pool="max",
    )
    sizes = size_feat(bundle.graphs)
    out = {"protocol_id": "molhiv-chemistry-typed-relation-folds-v1", "config": vars(args), "folds": {}}
    for fold in folds:
        tr = np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)
        va = np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)
        D, dinfo = learn_shared_D_graph_level(bundle.graphs, tr, cfg, node_feats=bundle.node_feats, edge_feats=bundle.edge_feats)
        rows = {k: [] for k in ("bag", "bag_typed_relation", "bag_shuffled_typed_relation", "size")}
        for gi in np.concatenate([tr, va]):
            i = int(gi)
            bag, rel, shuf = _encode_one(bundle.graphs[i], D, cfg, bundle.node_feats[i], bundle.edge_feats[i], shuffle_seed=7919 + i)
            rows["bag"].append(bag)
            rows["bag_typed_relation"].append(np.concatenate([bag, rel]))
            rows["bag_shuffled_typed_relation"].append(np.concatenate([bag, shuf]))
            rows["size"].append(sizes[i])
        ntr = len(tr)
        Xs = {k: np.stack(v, axis=0) for k, v in rows.items()}
        fold_out = {"n_train": int(len(tr)), "n_valid": int(len(va)), "dictionary": dinfo, "relation_dim": int(Xs["bag_typed_relation"].shape[1] - Xs["bag"].shape[1]), "variants": {}}
        for name, X in Xs.items():
            m = fit_auc(X[:ntr], bundle.y[tr], X[ntr:], bundle.y[va], X[ntr:], bundle.y[va], 0)
            fold_out["variants"][name] = {"train_auc": m["train_auc"], "valid_auc": m["valid_auc"]}
            print("fold", fold, name, json.dumps(fold_out["variants"][name]), flush=True)
        out["folds"][str(fold)] = fold_out
    path = Path(args.output) if args.output else Path(__file__).resolve().parents[1] / "results" / "molhiv" / "chemistry_typed_relation_folds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out["elapsed_sec"] = time.time() - t0
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
