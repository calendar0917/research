"""Fixed official-train scaffold-fold audit for chemistry-aware KSVD relations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig, learn_shared_D_graph_level
from .run_molhiv_chemistry_relation_probe import _encode_graph
from .run_molhiv_next_round import fit_auc, size_feat


def _low_relation(rel: np.ndarray, n_atoms: int) -> np.ndarray:
    """Compact symmetric readout of the atom-pair matrix."""
    R = np.asarray(rel, dtype=np.float64).reshape(n_atoms, n_atoms)
    R = 0.5 * (R + R.T)
    eig = np.linalg.eigvalsh(R)
    return np.concatenate([np.diag(R), R.sum(axis=1), eig])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--fold", type=int, default=-1, help="-1 runs all three folds")
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
        n_atoms=args.n_atoms,
        T=args.sparsity,
        T_min=1,
        ksvd_iter=args.ksvd_iter,
        seed=0,
        max_train_patches=args.max_train_patches,
        max_patches_per_graph=8,
        patch_feat="wl_chem_ring",
        normalize_patches=True,
        readout_mode="pool",
        pool="max",
    )
    sizes = size_feat(bundle.graphs)
    out = {"protocol_id": "molhiv-chemistry-relation-official-train-folds-v1", "config": vars(args), "folds": {}}

    for fold in folds:
        tr = np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)
        va = np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)
        D, dinfo = learn_shared_D_graph_level(
            bundle.graphs,
            tr,
            cfg,
            node_feats=bundle.node_feats,
            edge_feats=bundle.edge_feats,
        )
        rows = {k: [] for k in ("bag", "relation_low", "bag_relation_low", "bag_shuffled_relation_low", "size")}
        # Encode only official-train internal train/valid graphs. The dictionary
        # is fit on tr only, so no held-out graph enters KSVD.
        for i in np.concatenate([tr, va]):
            gi = int(i)
            bag, rel, _, _ = _encode_graph(
                bundle.graphs[gi], D, cfg, bundle.node_feats[gi], bundle.edge_feats[gi]
            )
            _, shuffled, _, _ = _encode_graph(
                bundle.graphs[gi], D, cfg, bundle.node_feats[gi], bundle.edge_feats[gi], relation_seed=7919 + gi
            )
            rows["bag"].append(bag)
            rows["relation_low"].append(_low_relation(rel, args.n_atoms))
            rows["bag_relation_low"].append(np.concatenate([bag, _low_relation(rel, args.n_atoms)]))
            rows["bag_shuffled_relation_low"].append(np.concatenate([bag, _low_relation(shuffled, args.n_atoms)]))
            rows["size"].append(sizes[gi])
        ntr = len(tr)
        Xs = {k: np.stack(v, axis=0) for k, v in rows.items()}
        fold_out = {"n_train": int(len(tr)), "n_valid": int(len(va)), "dictionary": dinfo, "variants": {}}
        for name, X in Xs.items():
            metrics = fit_auc(X[:ntr], bundle.y[tr], X[ntr:], bundle.y[va], X[ntr:], bundle.y[va], 0)
            fold_out["variants"][name] = {"train_auc": metrics["train_auc"], "valid_auc": metrics["valid_auc"]}
            print("fold", fold, name, json.dumps(fold_out["variants"][name]), flush=True)
        out["folds"][str(fold)] = fold_out

    if args.output:
        path = Path(args.output)
    else:
        path = Path(__file__).resolve().parents[1] / "results" / "molhiv" / "chemistry_relation_official_train_folds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out["elapsed_sec"] = time.time() - t0
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
