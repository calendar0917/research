"""MolHIV patch-content and patch-relation ablation.

This is a small, label-free readout experiment.  A shared KSVD dictionary is
fit on train patches only.  For every graph we compare:

* bag: statistics of patch sparse codes;
* relation: atom-by-atom co-occurrence accumulated over adjacent/overlapping
  patch pairs;
* bag+relation;
* bag+shuffled-relation.

The same patch sample and dictionary are used for all readouts.  The relation
is permutation invariant: it is a sum over unordered patch pairs, not a token
sequence.  ``wl`` is topology-only; ``wl_chem_ring`` includes OGB atom/bond
attributes through a permutation-invariant rooted patch descriptor.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _pair_weight(g, a: set[int], b: set[int]) -> float:
    """Structural relation between two patches, independent of node IDs."""
    overlap = len(a & b)
    cross = 0
    for u in a:
        for v in g.neighbors(u):
            if v in b:
                cross += 1
    # The two patches are related if they overlap or touch by an original edge.
    # Normalize so large patches do not dominate solely by cardinality.
    if overlap == 0 and cross == 0:
        return 0.0
    return float(overlap / max(1, min(len(a), len(b))) + cross / max(1, len(a) + len(b)))


def _relation_feature(X: np.ndarray, patches, g, rng: np.random.Generator | None = None):
    """Return atom-pair co-occurrence and diagnostics for one graph."""
    A = np.abs(np.asarray(X, dtype=np.float64))
    n_atoms, n_patches = A.shape
    if n_patches <= 1:
        return np.zeros(n_atoms * n_atoms, dtype=np.float64), {"n_pairs": 0, "weight": 0.0}

    W = np.zeros((n_patches, n_patches), dtype=np.float64)
    for i in range(n_patches):
        for j in range(i + 1, n_patches):
            w = _pair_weight(g, patches[i], patches[j])
            W[i, j] = W[j, i] = w
    if rng is not None and n_patches > 1:
        p = rng.permutation(n_patches)
        W = W[np.ix_(p, p)]

    total = float(W.sum())
    if total <= 1e-12:
        return np.zeros(n_atoms * n_atoms, dtype=np.float64), {"n_pairs": 0, "weight": 0.0}
    R = (A @ W @ A.T) / total
    # Scale removes the effect of the number of sampled patches.  Keeping the
    # full K x K matrix makes the relation directly inspectable.
    return R.reshape(-1), {
        "n_pairs": int(np.count_nonzero(np.triu(W, 1))),
        "weight": total,
    }


def _encode_graph(g, D, cfg, node_feat, edge_feat, relation_seed=None):
    bundle, _ = sample_patches_graph_level(g, cfg, seed=cfg.seed)
    Y, _ = bundle_to_Y(
        g,
        bundle,
        cfg.max_nodes,
        cfg.order_mode,
        patch_feat=cfg.patch_feat,
        node_feat=node_feat,
        edge_feat=edge_feat,
    )
    Yn, X = sparse_code_patch_matrix(Y, D, cfg)
    bag = sparse_code_readouts(X)["rich_no_recon"]
    rng = None if relation_seed is None else np.random.default_rng(relation_seed)
    rel, met = _relation_feature(X, bundle.node_sets[: X.shape[1]], g, rng=rng)
    return bag, rel, met, Yn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=3000, help="0 means full official split")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--patch-feat", choices=("wl", "wl_chem_ring"), default="wl_chem_ring")
    ap.add_argument("--n-atoms", type=int, default=8)
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument("--max-train-patches", type=int, default=3000)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    need_features = args.patch_feat == "wl_chem_ring"
    bundle = load_molhiv(max_graphs=max_graphs, seed=args.data_seed, with_features=need_features)
    graphs, y = bundle.graphs, bundle.y
    tr, va, te = (bundle.split[k] for k in ("train", "valid", "test"))
    cfg = GraphLevelConfig(
        n_atoms=args.n_atoms,
        T=args.sparsity,
        T_min=1,
        ksvd_iter=args.ksvd_iter,
        seed=0,
        max_train_patches=args.max_train_patches,
        max_patches_per_graph=8,
        patch_feat=args.patch_feat,
        normalize_patches=True,
        readout_mode="pool",
        pool="max",
    )
    D, dinfo = learn_shared_D_graph_level(
        graphs,
        tr,
        cfg,
        node_feats=bundle.node_feats,
        edge_feats=bundle.edge_feats,
    )

    rows = {k: [] for k in ("bag", "relation", "bag_relation", "bag_shuffled_relation", "size")}
    relation_rows = []
    for i, g in enumerate(graphs):
        nf = bundle.node_feats[i] if bundle.node_feats is not None else None
        ef = bundle.edge_feats[i] if bundle.edge_feats is not None else None
        bag, rel, met, _ = _encode_graph(g, D, cfg, nf, ef)
        _, shuffled, _, _ = _encode_graph(g, D, cfg, nf, ef, relation_seed=7919 + i)
        rows["bag"].append(bag)
        rows["relation"].append(rel)
        rows["bag_relation"].append(np.concatenate([bag, rel]))
        rows["bag_shuffled_relation"].append(np.concatenate([bag, shuffled]))
        relation_rows.append(met)
    sizes = size_feat(graphs)
    rows["size"] = sizes
    Xs = {k: np.stack(v, axis=0) for k, v in rows.items()}

    result = {
        "protocol_id": "molhiv-ksvd-chemistry-relation-probe-v1",
        "meta": bundle.meta,
        "config": vars(args),
        "dictionary": dinfo,
        "relation": {
            "mean_pairs": float(np.mean([m["n_pairs"] for m in relation_rows])),
            "mean_weight": float(np.mean([m["weight"] for m in relation_rows])),
        },
        "variants": {},
    }
    for name, X in Xs.items():
        result["variants"][name] = fit_auc(X[tr], y[tr], X[va], y[va], X[te], y[te], 0)
        print(name, json.dumps(result["variants"][name]), flush=True)

    if args.output:
        out = Path(args.output)
    else:
        suffix = "full" if max_graphs is None else f"n{max_graphs}"
        out = Path(__file__).resolve().parents[1] / "results" / "molhiv" / f"chemistry_relation_{args.patch_feat}_{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result["elapsed_sec"] = time.time() - t0
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
