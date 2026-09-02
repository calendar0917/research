"""Train-only label-aware patch reweighting for chemical MolHIV K-SVD.

Labels affect only which *training* patches enter dictionary learning.  Graph
encoding, OMP, pooling and the downstream classifier remain unchanged.  The
script intentionally evaluates train/official-valid only; test is not loaded
into the model-selection report.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_molhiv import MolhivBundle, load_molhiv
from .graph import Graph
from .graph_level import (
    GraphLevelConfig,
    bundle_to_Y,
    encode_patch_matrix,
    sample_patches_graph_level,
)
from .ksvd import ksvd
from .run_molhiv_ksvd_feasibility import _random_patch_dictionary
from .run_molhiv_next_round import size_feat
from .vectorize import ring_patch_features


@dataclass(frozen=True)
class PatchRecord:
    graph_idx: int
    nodes: tuple[int, ...]
    label: int


def _parse_seeds(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_fractions(raw: str) -> list[str]:
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    for value in values:
        if value == "natural":
            continue
        fraction = float(value)
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"positive fraction must be in (0,1), got {value}")
    return values


def _fit_train_valid(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    seed: int = 0,
) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, random_state=seed, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)

    def auc(X: np.ndarray, y: np.ndarray) -> float:
        probability = clf.predict_proba(X)
        classes = list(clf.named_steps["logisticregression"].classes_)
        positive_col = classes.index(1.0) if 1.0 in classes else classes.index(1)
        return float(roc_auc_score(y, probability[:, positive_col]))

    return {"train_auc": auc(Xtr, ytr), "valid_auc": auc(Xva, yva)}


def _collect_patch_pool(
    bundle: MolhivBundle,
    cfg: GraphLevelConfig,
) -> tuple[np.ndarray, list[PatchRecord], dict[str, float]]:
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    columns: list[np.ndarray] = []
    records: list[PatchRecord] = []
    per_graph_counts: list[int] = []
    for raw_idx in bundle.split["train"]:
        graph_idx = int(raw_idx)
        graph = bundle.graphs[graph_idx]
        patches, _ = sample_patches_graph_level(
            graph, cfg, seed=cfg.seed + graph_idx
        )
        node_sets = patches.node_sets
        if (
            cfg.max_patches_per_graph is not None
            and len(node_sets) > cfg.max_patches_per_graph
        ):
            rng = np.random.default_rng(cfg.seed + graph_idx * 1009)
            selected = rng.choice(
                len(node_sets), size=cfg.max_patches_per_graph, replace=False
            )
            node_sets = [node_sets[int(i)] for i in sorted(selected)]
        Y, _ = bundle_to_Y(
            graph,
            type("PatchBundle", (), {"node_sets": node_sets})(),
            cfg.max_nodes,
            cfg.order_mode,
            patch_feat=cfg.patch_feat,
            node_feat=bundle.node_feats[graph_idx],
            edge_feat=bundle.edge_feats[graph_idx],
        )
        if cfg.normalize_patches:
            Y = Y / np.maximum(np.linalg.norm(Y, axis=0, keepdims=True), 1e-12)
        kept = 0
        for column_idx, nodes in enumerate(node_sets):
            if column_idx >= Y.shape[1] or np.linalg.norm(Y[:, column_idx]) < 1e-12:
                continue
            columns.append(Y[:, column_idx])
            records.append(
                PatchRecord(
                    graph_idx=graph_idx,
                    nodes=tuple(sorted(nodes)),
                    label=int(bundle.y[graph_idx] > 0.5),
                )
            )
            kept += 1
        per_graph_counts.append(kept)
    Y_pool = np.stack(columns, axis=1)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    stats = {
        "n_pool": int(Y_pool.shape[1]),
        "n_positive_pool": int(labels.sum()),
        "positive_pool_fraction": float(labels.mean()),
        "mean_patches_per_train_graph": float(np.mean(per_graph_counts)),
    }
    return Y_pool, records, stats


def _select_training_patches(
    Y_pool: np.ndarray,
    records: list[PatchRecord],
    fraction_spec: str,
    n_patches: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    n_patches = min(n_patches, len(records))
    if fraction_spec == "natural":
        selected = rng.choice(len(records), size=n_patches, replace=False)
    else:
        target = float(fraction_spec)
        n_positive = int(round(n_patches * target))
        n_negative = n_patches - n_positive
        pos_pick = rng.choice(
            positive, size=n_positive, replace=n_positive > len(positive)
        )
        neg_pick = rng.choice(
            negative, size=n_negative, replace=n_negative > len(negative)
        )
        selected = np.concatenate([pos_pick, neg_pick])
        rng.shuffle(selected)
    selected_labels = labels[selected]
    return Y_pool[:, selected], selected, {
        "n_selected": int(len(selected)),
        "n_positive_selected": int(selected_labels.sum()),
        "positive_selected_fraction": float(selected_labels.mean()),
        "n_unique_selected": int(len(np.unique(selected))),
    }


def _encode_indices(
    bundle: MolhivBundle,
    indices: np.ndarray,
    dictionaries: dict[str, np.ndarray],
    cfg: GraphLevelConfig,
) -> dict[str, np.ndarray]:
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    rows: dict[str, list[np.ndarray]] = {name: [] for name in dictionaries}
    for raw_idx in indices:
        graph_idx = int(raw_idx)
        graph = bundle.graphs[graph_idx]
        patches, _ = sample_patches_graph_level(
            graph, cfg, seed=cfg.seed + graph_idx * 13
        )
        Y, _ = bundle_to_Y(
            graph,
            patches,
            cfg.max_nodes,
            cfg.order_mode,
            patch_feat=cfg.patch_feat,
            node_feat=bundle.node_feats[graph_idx],
            edge_feat=bundle.edge_feats[graph_idx],
        )
        for name, dictionary in dictionaries.items():
            embedding, _ = encode_patch_matrix(Y, dictionary, cfg)
            rows[name].append(embedding)
    return {name: np.stack(values, axis=0) for name, values in rows.items()}


def _dictionary_stability(
    dictionaries: dict[str, np.ndarray],
    fraction: str,
    seeds: list[int],
) -> dict:
    available = [seed for seed in seeds if f"ksvd_pos{fraction}_seed{seed}" in dictionaries]
    if len(available) < 2:
        return {"n_seeds": len(available)}
    reference = dictionaries[f"ksvd_pos{fraction}_seed{available[0]}"]
    alignments = []
    all_scores = []
    for seed in available[1:]:
        candidate = dictionaries[f"ksvd_pos{fraction}_seed{seed}"]
        similarities = np.abs(reference.T @ candidate)
        row, col = linear_sum_assignment(-similarities)
        scores = similarities[row, col]
        all_scores.extend(scores.tolist())
        alignments.append(
            {
                "seed": seed,
                "mean_abs_cosine": float(scores.mean()),
                "min_abs_cosine": float(scores.min()),
                "n_ge_0_8": int(np.sum(scores >= 0.8)),
                "assignment": col.tolist(),
                "scores": scores.tolist(),
            }
        )
    return {
        "reference_seed": available[0],
        "n_seeds": len(available),
        "mean_abs_cosine": float(np.mean(all_scores)),
        "median_abs_cosine": float(np.median(all_scores)),
        "alignments": alignments,
    }


def _patch_signature(
    graph: Graph,
    nodes: tuple[int, ...],
    node_feat: np.ndarray,
    edge_feat: dict[tuple[int, int], np.ndarray],
    max_nodes: int,
) -> dict:
    S = set(nodes)
    sub = graph.induced(S)
    atom_types = Counter(int(node_feat[u, 0]) for u in nodes)
    bond_types = Counter(
        int(edge_feat[graph.edge_key(u, v)][0])
        for u, v in sub.edges()
        if graph.edge_key(u, v) in edge_feat
    )
    ring = ring_patch_features(graph, S, max_nodes, node_feat, edge_feat)
    return {
        "n_nodes": sub.n,
        "n_edges": sub.num_edges(),
        "atom_type_counts": dict(sorted(atom_types.items())),
        "bond_type_counts": dict(sorted(bond_types.items())),
        "cycle_rank_scaled": ring[0],
        "cyclic_edge_fraction": ring[2],
        "aromatic_edge_fraction": ring[-3],
        "aromatic_atom_fraction": ring[-2],
        "ring_atom_fraction": ring[-1],
    }


def _interpret_atoms(
    bundle: MolhivBundle,
    dictionary: np.ndarray,
    Y_train: np.ndarray,
    selected: np.ndarray,
    records: list[PatchRecord],
    max_nodes: int,
    top_k: int = 5,
) -> list[dict]:
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    similarities = np.abs(dictionary.T @ Y_train)
    atoms = []
    for atom_idx in range(dictionary.shape[1]):
        top = np.argsort(-similarities[atom_idx])[:top_k]
        examples = []
        for local_idx in top:
            record = records[int(selected[int(local_idx)])]
            examples.append(
                {
                    "cosine": float(similarities[atom_idx, local_idx]),
                    "graph_idx": record.graph_idx,
                    "label": record.label,
                    "nodes": list(record.nodes),
                    "signature": _patch_signature(
                        bundle.graphs[record.graph_idx],
                        record.nodes,
                        bundle.node_feats[record.graph_idx],
                        bundle.edge_feats[record.graph_idx],
                        max_nodes,
                    ),
                }
            )
        atoms.append(
            {
                "atom": atom_idx,
                "top_positive_fraction": float(np.mean([x["label"] for x in examples])),
                "examples": examples,
            }
        )
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-graphs", type=int, default=0, help="0 means full split")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--dict-seeds", type=str, default="0")
    parser.add_argument(
        "--positive-fractions", type=str, default="natural,0.10,0.25,0.50"
    )
    parser.add_argument("--n-atoms", type=int, default=8)
    parser.add_argument("--sparsity", type=int, default=2)
    parser.add_argument("--ksvd-iter", type=int, default=4)
    parser.add_argument("--max-train-patches", type=int, default=4000)
    parser.add_argument("--selection-seed", type=int, default=1729)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dictionary-output", type=str, default=None)
    args = parser.parse_args()

    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    seeds = _parse_seeds(args.dict_seeds)
    fractions = _parse_fractions(args.positive_fractions)
    bundle = load_molhiv(
        max_graphs=max_graphs, seed=args.data_seed, with_features=True
    )
    train_idx = np.asarray(bundle.split["train"], dtype=np.int64)
    valid_idx = np.asarray(bundle.split["valid"], dtype=np.int64)
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

    pool_started = time.time()
    Y_pool, records, pool_stats = _collect_patch_pool(bundle, cfg)
    pool_stats["elapsed_sec"] = time.time() - pool_started
    selected_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    selection_stats = {}
    dictionaries: dict[str, np.ndarray] = {}
    training_info = {}
    for fraction_idx, fraction in enumerate(fractions):
        Y_train, selected, stats = _select_training_patches(
            Y_pool,
            records,
            fraction,
            args.max_train_patches,
            args.selection_seed + fraction_idx * 1009,
        )
        selected_data[fraction] = (Y_train, selected)
        selection_stats[fraction] = stats
        for seed in seeds:
            random_name = f"random_patch_pos{fraction}_seed{seed}"
            dictionaries[random_name] = _random_patch_dictionary(
                Y_train, args.n_atoms, seed
            )
            ksvd_name = f"ksvd_pos{fraction}_seed{seed}"
            D, _, info = ksvd(
                Y_train,
                n_atoms=args.n_atoms,
                T=args.sparsity,
                T_min=1,
                n_iter=args.ksvd_iter,
                seed=seed,
            )
            dictionaries[ksvd_name] = D
            training_info[ksvd_name] = info

    encode_idx = np.concatenate([train_idx, valid_idx])
    encode_started = time.time()
    encoded = _encode_indices(bundle, encode_idx, dictionaries, cfg)
    encode_sec = time.time() - encode_started
    n_train = len(train_idx)
    y_train = bundle.y[train_idx]
    y_valid = bundle.y[valid_idx]
    size = size_feat(bundle.graphs)
    size_train, size_valid = size[train_idx], size[valid_idx]
    size_result = _fit_train_valid(size_train, y_train, size_valid, y_valid)

    results = {}
    for name, X in encoded.items():
        X_train, X_valid = X[:n_train], X[n_train:]
        only = _fit_train_valid(X_train, y_train, X_valid, y_valid)
        plus_size = _fit_train_valid(
            np.hstack([X_train, size_train]),
            y_train,
            np.hstack([X_valid, size_valid]),
            y_valid,
        )
        results[name] = {
            "only": only,
            "plus_size": plus_size,
            "delta_valid_vs_size": plus_size["valid_auc"] - size_result["valid_auc"],
        }
        print(name, json.dumps(results[name]), flush=True)

    stability = {
        fraction: _dictionary_stability(dictionaries, fraction, seeds)
        for fraction in fractions
    }
    atom_examples = {}
    if seeds:
        for fraction in fractions:
            Y_train, selected = selected_data[fraction]
            name = f"ksvd_pos{fraction}_seed{seeds[0]}"
            atom_examples[name] = _interpret_atoms(
                bundle,
                dictionaries[name],
                Y_train,
                selected,
                records,
                cfg.max_nodes,
            )

    out = {
        "protocol_id": "molhiv-label-aware-ksvd-v1-valid-only",
        "meta": bundle.meta,
        "config": vars(args),
        "pool_stats": pool_stats,
        "selection_stats": selection_stats,
        "size": size_result,
        "shared_encode_sec": encode_sec,
        "training_info": training_info,
        "dictionaries": results,
        "stability": stability,
        "atom_examples": atom_examples,
        "test_policy": "test split was not encoded or evaluated by this script",
    }
    suffix = "full" if max_graphs is None else f"n{max_graphs}"
    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parents[1]
        / "results"
        / "molhiv"
        / f"label_aware_ksvd_{suffix}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    dictionary_output = (
        Path(args.dictionary_output)
        if args.dictionary_output
        else output.with_suffix(".npz")
    )
    np.savez_compressed(dictionary_output, **dictionaries)
    print(f"wrote {output}")
    print(f"wrote {dictionary_output}")


if __name__ == "__main__":
    main()
