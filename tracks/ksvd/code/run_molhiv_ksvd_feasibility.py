"""Fast feasibility check for the MolHIV KSVD route.

The probe keeps KSVD as the method core, while testing two prerequisites:
1. patch vectors are permutation invariant (WL-style fixed-width vectors);
2. KSVD beats or at least differs from random/PCA dictionaries under one
   fixed official split/subset.

Examples:
  python -m code.run_molhiv_ksvd_feasibility --max-graphs 5000
  python -m code.run_molhiv_ksvd_feasibility --max-graphs 0 --dict-seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data_molhiv import load_molhiv
from .graph import Graph
from .graph_level import (
    GraphLevelConfig,
    bundle_to_Y,
    collect_train_Y,
    encode_patch_matrix,
    sample_patches_graph_level,
)
from .ksvd import ksvd
from .run_molhiv_next_round import fit_auc, size_feat
from .vectorize import (
    adjacency_padded,
    flatten_upper,
    labeled_wl_patch_features,
    labeled_wl_ring_patch_features,
    wl_patch_features,
)


def _parse_seeds(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _random_dictionary(n_features: int, n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n_features, n_atoms))
    return D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)


def _random_patch_dictionary(Y: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(Y.shape[1], size=n_atoms, replace=Y.shape[1] < n_atoms)
    D = Y[:, idx].astype(np.float64).copy()
    return D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)


def _selected_patch_vector(
    g: Graph,
    S: set[int],
    cfg: GraphLevelConfig,
    node_feat: np.ndarray | None,
    edge_feat: dict[tuple[int, int], np.ndarray] | None,
) -> np.ndarray:
    if cfg.patch_feat == "wl":
        values = wl_patch_features(g, S, cfg.max_nodes)
    elif cfg.patch_feat == "wl_chem":
        values = labeled_wl_patch_features(
            g, S, cfg.max_nodes, node_feat=node_feat, edge_feat=edge_feat
        )
    elif cfg.patch_feat == "wl_chem_ring":
        values = labeled_wl_ring_patch_features(
            g, S, cfg.max_nodes, node_feat=node_feat, edge_feat=edge_feat
        )
    else:
        raise ValueError("permutation check only supports invariant WL patch features")
    return np.asarray(values, dtype=np.float64)


def _permutation_check(
    graphs: list[Graph],
    cfg: GraphLevelConfig,
    seed: int,
    node_feats: list[np.ndarray] | None = None,
    edge_feats: list[dict[tuple[int, int], np.ndarray]] | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_total = n_legacy_changed = n_selected_changed = 0
    for gi, g in enumerate(graphs[: min(150, len(graphs))]):
        nf = node_feats[gi] if node_feats is not None else None
        ef = edge_feats[gi] if edge_feats is not None else None
        bundle, _ = sample_patches_graph_level(g, cfg, seed=seed + gi)
        for S in bundle.node_sets[:3]:
            legacy = np.asarray(flatten_upper(adjacency_padded(g, S, cfg.max_nodes)))
            selected = _selected_patch_vector(g, S, cfg, nf, ef)

            old_nodes = g.nodes
            new_nodes = old_nodes.copy()
            rng.shuffle(new_nodes)
            mapping = dict(zip(old_nodes, new_nodes))
            relabeled = Graph(
                {mapping[u]: {mapping[v] for v in nbrs} for u, nbrs in g.adj.items()}
            )
            S2 = {mapping[u] for u in S}
            nf2 = None
            if nf is not None:
                nf2 = np.empty_like(nf)
                for old, new in mapping.items():
                    nf2[new] = nf[old]
            ef2 = None
            if ef is not None:
                ef2 = {}
                for (u, v), row in ef.items():
                    u2, v2 = mapping[u], mapping[v]
                    ef2[(u2, v2) if u2 < v2 else (v2, u2)] = row.copy()

            legacy2 = np.asarray(
                flatten_upper(adjacency_padded(relabeled, S2, cfg.max_nodes))
            )
            selected2 = _selected_patch_vector(relabeled, S2, cfg, nf2, ef2)
            n_legacy_changed += int(not np.array_equal(legacy, legacy2))
            n_selected_changed += int(not np.array_equal(selected, selected2))
            n_total += 1
    return {
        "patch_feat": cfg.patch_feat,
        "n_patches": n_total,
        "legacy_changed": n_legacy_changed,
        "legacy_changed_rate": n_legacy_changed / max(1, n_total),
        "selected_changed": n_selected_changed,
        "selected_changed_rate": n_selected_changed / max(1, n_total),
    }


def _encode_all_dictionaries(
    graphs: list[Graph],
    dictionaries: list[tuple[str, np.ndarray]],
    cfg: GraphLevelConfig,
    node_feats: list[np.ndarray] | None,
    edge_feats: list[dict[tuple[int, int], np.ndarray]] | None,
) -> dict[str, np.ndarray]:
    """Vectorize/sample each graph once, then encode it under every D."""
    rows: dict[str, list[np.ndarray]] = {name: [] for name, _ in dictionaries}
    for i, g in enumerate(graphs):
        nf = node_feats[i] if node_feats is not None else None
        ef = edge_feats[i] if edge_feats is not None else None
        patches, _ = sample_patches_graph_level(g, cfg, seed=cfg.seed + i * 13)
        Y, _ = bundle_to_Y(
            g,
            patches,
            cfg.max_nodes,
            cfg.order_mode,
            patch_feat=cfg.patch_feat,
            node_feat=nf,
            edge_feat=ef,
        )
        for name, D in dictionaries:
            embedding, _ = encode_patch_matrix(Y, D, cfg)
            rows[name].append(embedding)
    return {name: np.stack(values, axis=0) for name, values in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=5000, help="0 means full split")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--dict-seeds", type=str, default="0,1,2")
    ap.add_argument("--n-atoms", type=int, default=8)
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument("--max-train-patches", type=int, default=4000)
    ap.add_argument(
        "--patch-feat",
        choices=("wl", "wl_chem", "wl_chem_ring"),
        default="wl_chem_ring",
    )
    ap.add_argument(
        "--families",
        default="random,random_patch,pca,ksvd",
        help="comma-separated subset of random,random_patch,pca,ksvd",
    )
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    seeds = _parse_seeds(args.dict_seeds)
    families = {x.strip() for x in args.families.split(",") if x.strip()}
    unknown_families = families - {"random", "random_patch", "pca", "ksvd"}
    if unknown_families:
        raise ValueError(f"unknown dictionary families: {sorted(unknown_families)}")
    needs_features = args.patch_feat in {"wl_chem", "wl_chem_ring"}
    bundle = load_molhiv(
        max_graphs=max_graphs, seed=args.data_seed, with_features=needs_features
    )
    graphs, y = bundle.graphs, bundle.y
    tr, va, te = (bundle.split[k] for k in ("train", "valid", "test"))
    size = size_feat(graphs)
    size_result = fit_auc(size[tr], y[tr], size[va], y[va], size[te], y[te], 0)

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
    permutation = _permutation_check(
        graphs,
        cfg,
        args.data_seed,
        node_feats=bundle.node_feats,
        edge_feats=bundle.edge_feats,
    )
    Y, patch_stats = collect_train_Y(
        graphs,
        tr,
        cfg,
        node_feats=bundle.node_feats,
        edge_feats=bundle.edge_feats,
    )

    dictionaries: list[tuple[str, np.ndarray]] = []
    for seed in seeds:
        if "random" in families:
            dictionaries.append(
                (f"random_seed{seed}", _random_dictionary(Y.shape[0], args.n_atoms, seed))
            )
        if "random_patch" in families:
            dictionaries.append(
                (f"random_patch_seed{seed}", _random_patch_dictionary(Y, args.n_atoms, seed))
            )
    if "pca" in families:
        U, _, _ = np.linalg.svd(Y, full_matrices=False)
        dictionaries.append(("pca", U[:, : args.n_atoms]))
    if "ksvd" in families:
        for seed in seeds:
            D, _, _ = ksvd(
                Y,
                n_atoms=args.n_atoms,
                T=args.sparsity,
                T_min=1,
                n_iter=args.ksvd_iter,
                seed=seed,
            )
            dictionaries.append((f"ksvd_seed{seed}", D))
    if not dictionaries:
        raise ValueError("at least one dictionary family is required")

    out = {
        "protocol_id": "molhiv-ksvd-feasibility-v2",
        "meta": bundle.meta,
        "config": vars(args),
        "permutation": permutation,
        "patch_stats": patch_stats,
        "size": size_result,
        "dictionaries": {},
    }
    encode_t0 = time.time()
    encoded = _encode_all_dictionaries(
        graphs,
        dictionaries,
        cfg,
        node_feats=bundle.node_feats,
        edge_feats=bundle.edge_feats,
    )
    shared_encode_sec = time.time() - encode_t0
    for name, _ in dictionaries:
        t0 = time.time()
        X = encoded[name]
        only = fit_auc(X[tr], y[tr], X[va], y[va], X[te], y[te], 0)
        X_size = np.hstack([X, size])
        plus_size = fit_auc(
            X_size[tr], y[tr], X_size[va], y[va], X_size[te], y[te], 0
        )
        row = {
            "only": only,
            "plus_size": plus_size,
            "delta_valid_vs_size": plus_size["valid_auc"] - size_result["valid_auc"],
            "delta_test_vs_size": plus_size["test_auc"] - size_result["test_auc"],
            "fit_elapsed_sec": time.time() - t0,
            "shared_encode_sec": shared_encode_sec,
        }
        out["dictionaries"][name] = row
        print(name, json.dumps(row), flush=True)

    if args.output:
        output = Path(args.output)
    else:
        suffix = "full" if max_graphs is None else f"n{max_graphs}"
        output = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "molhiv"
            / f"ksvd_feasibility_{args.patch_feat}_{suffix}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
