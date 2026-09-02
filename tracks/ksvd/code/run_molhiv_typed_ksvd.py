"""Typed chemical K-SVD feasibility probe for ogbg-molhiv.

Patches are split into fused/aromatic/ring/tree vocabularies, with one K-SVD
per type. All dictionary families use identical patches, dimensions, OMP and
readout so K-SVD is compared against random-patch and PCA controls fairly.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .data_molhiv import load_molhiv
from .graph import Graph
from .graph_level import GraphLevelConfig, encode_patch_matrix, sample_patches_graph_level
from .ksvd import ksvd
from .run_molhiv_ksvd_feasibility import _random_patch_dictionary
from .run_molhiv_next_round import fit_auc, size_feat
from .vectorize import labeled_wl_ring_patch_features

PATCH_TYPES = ("fused", "aromatic", "ring", "tree")


def _parse_seeds(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _patch_type(
    g: Graph,
    S: set[int],
    node_feat: np.ndarray,
    edge_feat: dict[tuple[int, int], np.ndarray],
) -> str:
    sub = g.induced(S)
    n = sub.n
    e = sub.num_edges()
    unseen = set(sub.nodes)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            u = stack.pop()
            for v in sub.neighbors(u):
                if v in unseen:
                    unseen.remove(v)
                    stack.append(v)
    cycle_rank = max(0, e - n + components)
    if cycle_rank > 1:
        return "fused"
    aromatic = any(
        int(edge_feat.get(g.edge_key(u, v), np.array([-1]))[0]) == 3
        for u, v in sub.edges()
    )
    if aromatic:
        return "aromatic"
    atom_in_ring = any(
        u < node_feat.shape[0]
        and node_feat.shape[1] > 8
        and int(node_feat[u, 8]) == 1
        for u in S
    )
    if cycle_rank == 1 or atom_in_ring:
        return "ring"
    return "tree"


def _patch_vector(
    g: Graph,
    S: set[int],
    cfg: GraphLevelConfig,
    node_feat: np.ndarray,
    edge_feat: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    y = np.asarray(
        labeled_wl_ring_patch_features(
            g, S, cfg.max_nodes, node_feat=node_feat, edge_feat=edge_feat
        ),
        dtype=np.float64,
    )
    return y / max(np.linalg.norm(y), 1e-12)


def _sample_sets(g: Graph, cfg: GraphLevelConfig, graph_idx: int) -> list[set[int]]:
    bundle, _ = sample_patches_graph_level(g, cfg, seed=cfg.seed + graph_idx * 13)
    sets = bundle.node_sets
    limit = cfg.max_patches_per_graph
    if limit is not None and len(sets) > limit:
        rng = np.random.default_rng(cfg.seed + graph_idx * 1009)
        pick = rng.choice(len(sets), size=limit, replace=False)
        sets = [sets[int(i)] for i in sorted(pick)]
    return sets


def _collect_typed_train(
    graphs: list[Graph],
    train_idx: np.ndarray,
    node_feats: list[np.ndarray],
    edge_feats: list[dict[tuple[int, int], np.ndarray]],
    cfg: GraphLevelConfig,
    max_per_type: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    cols: dict[str, list[np.ndarray]] = {t: [] for t in PATCH_TYPES}
    for raw_idx in train_idx:
        i = int(raw_idx)
        for S in _sample_sets(graphs[i], cfg, i):
            typ = _patch_type(graphs[i], S, node_feats[i], edge_feats[i])
            cols[typ].append(_patch_vector(graphs[i], S, cfg, node_feats[i], edge_feats[i]))
    raw_counts = {t: len(cols[t]) for t in PATCH_TYPES}
    rng = np.random.default_rng(cfg.seed)
    out: dict[str, np.ndarray] = {}
    for typ in PATCH_TYPES:
        values = cols[typ]
        if len(values) > max_per_type:
            pick = rng.choice(len(values), size=max_per_type, replace=False)
            values = [values[int(i)] for i in pick]
        if not values:
            raise RuntimeError(f"no training patches for type {typ}")
        out[typ] = np.stack(values, axis=1)
    return out, raw_counts


def _learn_families(
    typed_Y: dict[str, np.ndarray],
    seeds: list[int],
    n_atoms: int,
    sparsity: int,
    n_iter: int,
) -> dict[str, dict[str, np.ndarray]]:
    families: dict[str, dict[str, np.ndarray]] = {}
    for seed in seeds:
        families[f"random_patch_seed{seed}"] = {
            typ: _random_patch_dictionary(Y, n_atoms, seed) for typ, Y in typed_Y.items()
        }
        learned: dict[str, np.ndarray] = {}
        for type_idx, (typ, Y) in enumerate(typed_Y.items()):
            D, _, _ = ksvd(
                Y,
                n_atoms=n_atoms,
                T=sparsity,
                T_min=1,
                n_iter=n_iter,
                seed=seed + type_idx * 1009,
            )
            learned[typ] = D
        families[f"ksvd_seed{seed}"] = learned
    pca: dict[str, np.ndarray] = {}
    for typ, Y in typed_Y.items():
        U, _, _ = np.linalg.svd(Y, full_matrices=False)
        pca[typ] = U[:, :n_atoms]
    families["pca"] = pca
    return families


def _encode_families(
    graphs: list[Graph],
    node_feats: list[np.ndarray],
    edge_feats: list[dict[tuple[int, int], np.ndarray]],
    cfg: GraphLevelConfig,
    families: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    rows = {name: [] for name in families}
    type_rows: list[np.ndarray] = []
    total_counts: Counter[str] = Counter()
    for i, g in enumerate(graphs):
        grouped: dict[str, list[np.ndarray]] = {t: [] for t in PATCH_TYPES}
        for S in _sample_sets(g, cfg, i):
            typ = _patch_type(g, S, node_feats[i], edge_feats[i])
            grouped[typ].append(_patch_vector(g, S, cfg, node_feats[i], edge_feats[i]))
            total_counts[typ] += 1
        counts = np.asarray([len(grouped[t]) for t in PATCH_TYPES], dtype=np.float64)
        type_rows.append(np.concatenate([counts / max(1.0, counts.sum()), (counts > 0).astype(float)]))
        for name, dicts in families.items():
            pieces: list[np.ndarray] = []
            for typ in PATCH_TYPES:
                if grouped[typ]:
                    Y = np.stack(grouped[typ], axis=1)
                    emb, _ = encode_patch_matrix(Y, dicts[typ], cfg)
                else:
                    emb = np.zeros(dicts[typ].shape[1], dtype=np.float64)
                pieces.append(emb)
            rows[name].append(np.concatenate(pieces))
    return (
        {name: np.stack(values, axis=0) for name, values in rows.items()},
        np.stack(type_rows, axis=0),
        dict(total_counts),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=5000, help="0 means full split")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--dict-seeds", type=str, default="0,1,2")
    ap.add_argument("--atoms-per-type", type=int, default=8)
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument("--max-train-patches-per-type", type=int, default=2000)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    seeds = _parse_seeds(args.dict_seeds)
    bundle = load_molhiv(max_graphs=max_graphs, seed=args.data_seed, with_features=True)
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    graphs, y = bundle.graphs, bundle.y
    tr, va, te = (bundle.split[k] for k in ("train", "valid", "test"))
    size = size_feat(graphs)
    size_result = fit_auc(size[tr], y[tr], size[va], y[va], size[te], y[te], 0)

    cfg = GraphLevelConfig(
        n_atoms=args.atoms_per_type,
        T=args.sparsity,
        T_min=1,
        ksvd_iter=args.ksvd_iter,
        seed=0,
        max_patches_per_graph=8,
        patch_feat="wl_chem_ring",
        normalize_patches=False,  # vectors are normalized once before training/encoding
        readout_mode="pool",
        pool="max",
    )
    typed_Y, train_type_counts = _collect_typed_train(
        graphs,
        tr,
        bundle.node_feats,
        bundle.edge_feats,
        cfg,
        args.max_train_patches_per_type,
    )
    families = _learn_families(
        typed_Y, seeds, args.atoms_per_type, args.sparsity, args.ksvd_iter
    )
    t0 = time.time()
    encoded, type_X, all_type_counts = _encode_families(
        graphs, bundle.node_feats, bundle.edge_feats, cfg, families
    )
    encode_sec = time.time() - t0
    type_size = np.hstack([type_X, size])
    type_result = fit_auc(
        type_size[tr], y[tr], type_size[va], y[va], type_size[te], y[te], 0
    )

    out = {
        "protocol_id": "molhiv-typed-ksvd-v1",
        "meta": bundle.meta,
        "config": vars(args),
        "patch_types": PATCH_TYPES,
        "train_type_counts": train_type_counts,
        "used_train_patches": {t: int(Y.shape[1]) for t, Y in typed_Y.items()},
        "all_type_counts": all_type_counts,
        "size": size_result,
        "type_plus_size": type_result,
        "shared_encode_sec": encode_sec,
        "dictionaries": {},
    }
    for name, X in encoded.items():
        X_type_size = np.hstack([X, type_X, size])
        result = fit_auc(
            X_type_size[tr], y[tr], X_type_size[va], y[va], X_type_size[te], y[te], 0
        )
        row = {
            "plus_type_size": result,
            "delta_valid_vs_type_size": result["valid_auc"] - type_result["valid_auc"],
            "delta_test_vs_type_size": result["test_auc"] - type_result["test_auc"],
            "delta_valid_vs_size": result["valid_auc"] - size_result["valid_auc"],
            "delta_test_vs_size": result["test_auc"] - size_result["test_auc"],
        }
        out["dictionaries"][name] = row
        print(name, json.dumps(row), flush=True)

    if args.output:
        output = Path(args.output)
    else:
        suffix = "full" if max_graphs is None else f"n{max_graphs}"
        output = Path(__file__).resolve().parents[1] / "results" / "molhiv" / f"typed_ksvd_{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
