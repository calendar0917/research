"""Train-only low-rank readout for chemistry-typed KSVD relations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig, learn_shared_D_graph_level
from .run_molhiv_next_round import fit_auc, size_feat
from .run_molhiv_typed_relation_folds import _encode_one


def _fit_pca(train_rel: np.ndarray, all_rel: np.ndarray, rank: int):
    scaler = StandardScaler().fit(train_rel)
    ztr = scaler.transform(train_rel)
    za = scaler.transform(all_rel)
    r = min(rank, ztr.shape[0], ztr.shape[1])
    pca = PCA(n_components=r, random_state=0).fit(ztr)
    return pca.transform(ztr), pca.transform(za), {
        "rank": int(r),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--rank", type=int, default=16)
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
    out = {"protocol_id": "molhiv-chemistry-typed-relation-pca-folds-v1", "config": vars(args), "folds": {}}

    for fold in folds:
        tr = np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)
        va = np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)
        D, dinfo = learn_shared_D_graph_level(bundle.graphs, tr, cfg, node_feats=bundle.node_feats, edge_feats=bundle.edge_feats)
        bags, rels, shufs, size_rows = [], [], [], []
        for gi in np.concatenate([tr, va]):
            i = int(gi)
            bag, rel, shuf = _encode_one(bundle.graphs[i], D, cfg, bundle.node_feats[i], bundle.edge_feats[i], shuffle_seed=7919 + i)
            bags.append(bag); rels.append(rel); shufs.append(shuf); size_rows.append(sizes[i])
        ntr = len(tr)
        bag_all = np.stack(bags, axis=0)
        rel_all = np.stack(rels, axis=0)
        shuf_all = np.stack(shufs, axis=0)
        bag_tr, bag_va = bag_all[:ntr], bag_all[ntr:]
        rel_tr, rel_proj, pca_meta = _fit_pca(rel_all[:ntr], rel_all, args.rank)
        shuf_tr, shuf_proj, shuf_pca_meta = _fit_pca(shuf_all[:ntr], shuf_all, args.rank)
        Xs = {
            "bag": bag_all,
            "bag_typed_relation_pca": np.concatenate([bag_all, rel_proj], axis=1),
            "bag_shuffled_typed_relation_pca": np.concatenate([bag_all, shuf_proj], axis=1),
            "size": np.stack(size_rows, axis=0),
        }
        fold_out = {"n_train": int(len(tr)), "n_valid": int(len(va)), "dictionary": dinfo, "pca": {"correct": pca_meta, "shuffled": shuf_pca_meta}, "variants": {}}
        for name, X in Xs.items():
            m = fit_auc(X[:ntr], bundle.y[tr], X[ntr:], bundle.y[va], X[ntr:], bundle.y[va], 0)
            fold_out["variants"][name] = {"train_auc": m["train_auc"], "valid_auc": m["valid_auc"]}
            print("fold", fold, name, json.dumps(fold_out["variants"][name]), flush=True)
        out["folds"][str(fold)] = fold_out

    path = Path(args.output) if args.output else Path(__file__).resolve().parents[1] / "results" / "molhiv" / "chemistry_typed_relation_pca_folds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out["elapsed_sec"] = time.time() - t0
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
