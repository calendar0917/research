"""A natural 624-D typed-slot reconstruction proxy for the mentor route.

The proxy uses 8 atom slots x 64 atom bins plus 28 undirected slot pairs x 4
bond bins.  It is deliberately not presented as the mentor's exact feature
builder; its purpose is to test whether preserving typed slot alignment, rather
than adding more global distribution statistics, changes the attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ksvd_research.core import ksvd
from ksvd_research.data import MolhivBundle, load_molhiv
from ksvd_research.evaluation import GraphLevelConfig, sparse_code_patch_matrix
from tracks.ksvd.experiments.luyin16.mentor_concept_replication import (
    REPO_ROOT,
    _resolve_repo_path,
    _write_json,
    _write_records,
    build_composition_features,
    collect_train_patch_matrix,
    fit_xgboost_views,
    graph_level_config,
)
from ksvd_research.evaluation.graph_level import sample_patches_graph_level


NODE_BINS = 64
BOND_BINS = 4
PATCH_SIZE = 8
PAIR_COUNT = PATCH_SIZE * (PATCH_SIZE - 1) // 2
TYPED_SLOT_DIM = PATCH_SIZE * NODE_BINS + PAIR_COUNT * BOND_BINS


def _typed_slot_order(graph: Any, nodes: set[int], node_features: np.ndarray) -> list[int]:
    """Deterministic typed BFS order; IDs only break residual exact ties."""
    if not nodes:
        return []
    induced_degree = {u: sum(v in nodes for v in graph.neighbors(u)) for u in nodes}
    root = max(nodes, key=lambda u: (induced_degree[u], -int(node_features[u, 0]), -u))
    order: list[int] = []
    seen = {root}
    queue = [root]
    while queue:
        current = queue.pop(0)
        order.append(current)
        neighbors = [v for v in graph.neighbors(current) if v in nodes and v not in seen]
        neighbors.sort(key=lambda v: (-induced_degree[v], int(node_features[v, 0]) % NODE_BINS, v))
        seen.update(neighbors)
        queue.extend(neighbors)
    for node in sorted(nodes, key=lambda u: (-induced_degree[u], int(node_features[u, 0]) % NODE_BINS, u)):
        if node not in seen:
            seen.add(node)
            order.append(node)
    return order[:PATCH_SIZE]


def typed_slot_vector(
    graph: Any,
    nodes: set[int],
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
) -> np.ndarray:
    order = _typed_slot_order(graph, nodes, node_features)
    output = np.zeros(TYPED_SLOT_DIM, dtype=np.float64)
    for slot, node in enumerate(order):
        output[slot * NODE_BINS + (int(node_features[node, 0]) % NODE_BINS)] = 1.0
    pair_offset = PATCH_SIZE * NODE_BINS
    pair_index = 0
    for left in range(PATCH_SIZE):
        for right in range(left + 1, PATCH_SIZE):
            if left < len(order) and right < len(order) and graph.has_edge(order[left], order[right]):
                key = graph.edge_key(order[left], order[right])
                values = edge_features.get(key)
                bond_type = int(values[0]) % BOND_BINS if values is not None and values.size else 0
                output[pair_offset + pair_index * BOND_BINS + bond_type] = 1.0
            pair_index += 1
    return output


def vectorize_graphs(
    bundle: MolhivBundle,
    cfg: GraphLevelConfig,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("typed-slot proxy requires node and edge features")
    matrices: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for graph_index, graph in enumerate(bundle.graphs):
        local_seed = cfg.seed + graph_index * 13
        sampled, sample_meta = sample_patches_graph_level(graph, cfg, seed=local_seed)
        columns = [
            typed_slot_vector(graph, nodes, bundle.node_feats[graph_index], bundle.edge_feats[graph_index])
            for nodes in sampled.node_sets
        ]
        matrix = np.stack(columns, axis=1) if columns else np.zeros((TYPED_SLOT_DIM, 0), dtype=np.float64)
        if matrix.shape[1]:
            norms = np.linalg.norm(matrix, axis=0, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
            if cfg.max_patches_per_graph is not None and matrix.shape[1] > cfg.max_patches_per_graph:
                rng = np.random.default_rng(local_seed)
                chosen = np.sort(
                    rng.choice(matrix.shape[1], size=cfg.max_patches_per_graph, replace=False)
                )
                matrix = matrix[:, chosen]
        matrices.append(matrix)
        metadata.append(
            {
                "graph_index": graph_index,
                "local_seed": local_seed,
                "n_patches": int(matrix.shape[1]),
                **sample_meta,
            }
        )
    return matrices, metadata


def _graph_readout(
    patch_matrices: Sequence[np.ndarray],
    dictionary: np.ndarray | None,
    cfg: GraphLevelConfig,
    aggregation: str = "mean",
) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    errors: list[float] = []
    for matrix in patch_matrices:
        if dictionary is None:
            represented = matrix
        elif matrix.shape[1]:
            encoded_matrix, codes = sparse_code_patch_matrix(matrix, dictionary, cfg)
            reconstructed = dictionary @ codes
            if aggregation == "mean":
                represented = reconstructed
            elif aggregation == "atom_signed":
                # Preserve dictionary atom orientation while removing
                # coefficient-sign cancellation.  This binds D's content to
                # graph-specific |X| and remains in the original 624-D space.
                represented = dictionary @ np.abs(codes)
            elif aggregation == "atom_abs":
                # Magnitude-only atom contribution; useful when signs are an
                # arbitrary coding convention rather than semantic polarity.
                represented = np.abs(dictionary) @ np.abs(codes)
            elif aggregation == "abs_mean":
                represented = np.abs(dictionary @ codes)
            else:
                raise ValueError(f"unknown aggregation={aggregation!r}")
            errors.append(
                float(
                    # Reconstruction error is always Y-DX. Alternative
                    # readouts such as D|X| must not redefine this metric.
                    np.linalg.norm(encoded_matrix - reconstructed)
                    / max(np.linalg.norm(encoded_matrix), 1e-12)
                )
            )
        else:
            represented = np.zeros((TYPED_SLOT_DIM, 0), dtype=np.float64)
            errors.append(0.0)
        if represented.shape[1]:
            if aggregation in {"mean", "atom_signed", "atom_abs", "abs_mean"}:
                row = represented.mean(axis=1)
            elif aggregation == "sum":
                row = represented.sum(axis=1)
            elif aggregation == "max":
                row = represented.max(axis=1)
            else:
                raise ValueError(f"unknown aggregation={aggregation!r}")
        else:
            row = np.zeros(TYPED_SLOT_DIM)
        rows.append(row)
    features = np.stack(rows, axis=0).astype(np.float32)
    return features, {
        "dimension": int(features.shape[1]),
        "readout": f"{aggregation} over slot-aligned patch object",
        "input_dimension": TYPED_SLOT_DIM,
        "mean_graph_reconstruction_error": None if dictionary is None else float(np.mean(errors)),
    }


def run(config_path: Path, result_dir: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))))
    data_config = dict(config["data"])
    root = _resolve_repo_path(data_config["root"])
    bundle = load_molhiv(
        root=root,
        max_graphs=None if data_config.get("max_graphs") is None else int(data_config["max_graphs"]),
        seed=int(data_config.get("subsample_seed", 0)),
        with_features=True,
    )
    cfg = graph_level_config(config)
    composition, _composition_names, _composition_meta = build_composition_features(bundle)
    patch_matrices, sampling_meta = vectorize_graphs(bundle, cfg)
    train_patches, leakage_audit = collect_train_patch_matrix(
        patch_matrices, bundle.split["train"], cfg.max_train_patches, cfg.seed
    )
    initial, _, initial_info = ksvd(
        train_patches, n_atoms=cfg.n_atoms, T=cfg.T, T_min=cfg.T_min, n_iter=0, seed=cfg.seed
    )
    final, _, final_info = ksvd(
        train_patches,
        n_atoms=initial.shape[1],
        T=cfg.T,
        T_min=cfg.T_min,
        n_iter=cfg.ksvd_iter,
        seed=cfg.seed,
        initial_dictionary=initial,
    )
    aggregation = str(config.get("readout", "mean"))
    raw, raw_meta = _graph_readout(patch_matrices, None, cfg, aggregation)
    init, init_meta = _graph_readout(patch_matrices, initial, cfg, aggregation)
    fin, final_meta = _graph_readout(patch_matrices, final, cfg, aggregation)
    init_atom_signed, init_atom_signed_meta = _graph_readout(
        patch_matrices, initial, cfg, "atom_signed"
    )
    final_atom_signed, final_atom_signed_meta = _graph_readout(
        patch_matrices, final, cfg, "atom_signed"
    )
    init_atom_abs, init_atom_abs_meta = _graph_readout(
        patch_matrices, initial, cfg, "atom_abs"
    )
    final_atom_abs, final_atom_abs_meta = _graph_readout(
        patch_matrices, final, cfg, "atom_abs"
    )
    init_abs_mean, init_abs_mean_meta = _graph_readout(
        patch_matrices, initial, cfg, "abs_mean"
    )
    final_abs_mean, final_abs_mean_meta = _graph_readout(
        patch_matrices, final, cfg, "abs_mean"
    )
    available = {
        "s": composition,
        "r_raw": raw,
        "r_init": init,
        "r_final": fin,
        "r_init_atom_signed": init_atom_signed,
        "r_final_atom_signed": final_atom_signed,
        "r_init_atom_abs": init_atom_abs,
        "r_final_atom_abs": final_atom_abs,
        "r_init_abs_mean": init_abs_mean,
        "r_final_abs_mean": final_abs_mean,
        # Keep the view names semantically literal: S is the explicit
        # topology/composition block (205D), and R is the 624D typed-slot
        # readout.  The previous implementation accidentally duplicated R,
        # making the 1248D rows R_raw+R_* rather than S+R_*.
        "s_r_raw": np.concatenate([composition, raw], axis=1),
        "s_r_init": np.concatenate([composition, init], axis=1),
        "s_r_final": np.concatenate([composition, fin], axis=1),
        "s_r_final_atom_signed": np.concatenate([composition, final_atom_signed], axis=1),
        "s_r_final_atom_abs": np.concatenate([composition, final_atom_abs], axis=1),
        "s_r_final_abs_mean": np.concatenate([composition, final_abs_mean], axis=1),
    }
    requested = list(config["views"])
    views = {name: available[name] for name in requested}
    records, fitted = fit_xgboost_views(
        views, bundle.y, bundle.split, dict(config["classifier"]), evaluate_test=False
    )
    manifest = {
        "protocol_id": str(config["protocol_id"]),
        "status": "full" if data_config.get("max_graphs") is None else "development",
        "proxy": "typed-slot-8x64-plus-pair-28x4-v2",
        "not_exact_mentor_feature_replication": True,
        "resolved_config": config,
        "data": bundle.meta,
        "leakage_audit": leakage_audit,
        "sampling": {
            "n_graphs": len(sampling_meta),
            "mean_patches": float(np.mean([item["n_patches"] for item in sampling_meta])),
        },
        "dictionary_info": {"initial": initial_info, "final": final_info},
        "features": {
            "r_raw_v2": raw_meta,
            "r_init_v2": init_meta,
            "r_final_v2": final_meta,
            "r_init_atom_signed_v2": init_atom_signed_meta,
            "r_final_atom_signed_v2": final_atom_signed_meta,
            "r_init_atom_abs_v2": init_atom_abs_meta,
            "r_final_atom_abs_v2": final_atom_abs_meta,
            "r_init_abs_mean_v2": init_abs_mean_meta,
            "r_final_abs_mean_v2": final_abs_mean_meta,
        },
        "classifier": fitted["summary"],
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_records(result_dir / "records.csv", records)
    _write_json(result_dir / "summary.json", manifest)
    # Persist only train+official-validation rows for downstream classifier
    # studies.  This keeps the feature freeze reproducible without encoding or
    # exporting any official-test representation.
    train_valid = np.concatenate(
        [np.asarray(bundle.split["train"], dtype=np.int64), np.asarray(bundle.split["valid"], dtype=np.int64)]
    )
    feature_payload = {
        "dataset_indices": train_valid,
        "labels": np.asarray(bundle.y[train_valid], dtype=np.int64),
        "train_count": np.asarray([len(bundle.split["train"])], dtype=np.int64),
    }
    for name, matrix in views.items():
        feature_payload[name] = np.asarray(matrix[train_valid], dtype=np.float32)
    np.savez_compressed(result_dir / "feature_views_train_valid.npz", **feature_payload)
    (result_dir / "TYPED_SLOT_PROXY_SUMMARY.md").write_text(_render_summary(manifest), encoding="utf-8")
    return manifest


def _render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Typed-slot reconstruction proxy v2",
        "",
        f"Protocol: `{payload['protocol_id']}`",
        "",
        "This is not the mentor's exact feature builder. It tests a natural 624-D slot-aligned object: 8×64 atom bins + 28×4 bond bins.",
        "",
        "| view | dim | valid ROC-AUC |",
        "|---|---:|---:|",
    ]
    for name, result in payload["classifier"]["per_view"].items():
        lines.append(f"| {name} | {result['dimension']} | {result['valid']['mean']:.6f} ± {result['valid']['std_sample']:.6f} |")
    lines.extend(
        [
            "",
            f"Best by validation only: `{payload['classifier']['best_by_valid_only']}`.",
            "",
            "Primary checks: `r_final-r_init`, `r_final-r_raw`, and `s_r_final-s`.",
            "Test evaluation is disabled.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = REPO_ROOT / "tracks/ksvd/configs/luyin16/mentor_typed_slot_proxy_v2.yaml"
    if args.result_dir is None:
        args.result_dir = REPO_ROOT / "tracks/ksvd/results/luyin16/mentor_typed_slot_proxy_v2"
    result = run(args.config.expanduser().resolve(), args.result_dir.expanduser().resolve())
    print(
        json.dumps(
            {
                "protocol_id": result["protocol_id"],
                "status": result["status"],
                "best_by_valid_only": result["classifier"]["best_by_valid_only"],
                "view_dimensions": result["classifier"]["per_view"],
                "test_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
