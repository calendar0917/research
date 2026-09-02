"""Mentor-shaped S/T/A ablation with a label-free ring-context proxy.

The mentor script explicitly consumes composition[69], recon_typed[624], and
five selected ring-context mass blocks of size K plus five coverage scalars.
This runner reproduces that shape and attribution question with local,
auditable proxies. It does not claim the unknown upstream feature schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ksvd_research.core import ksvd
from ksvd_research.data import MolhivBundle, load_molhiv
from ksvd_research.evaluation import (
    GraphLevelConfig,
    sparse_code_patch_matrix,
    sparse_code_readouts,
)
from ksvd_research.evaluation.graph_level import sample_patches_graph_level
from ksvd_research.features import RING_CONTEXT_NAMES, build_ring_context_index
from tracks.ksvd.experiments.luyin16.mentor_concept_replication import (
    _resolve_repo_path,
    _write_json,
    _write_records,
    collect_train_patch_matrix,
    fit_xgboost_views,
    graph_level_config,
)
from tracks.ksvd.experiments.luyin16.pure_structural_fusion_proxy import structural_69
from tracks.ksvd.experiments.luyin16.typed_slot_reconstruction_proxy import (
    TYPED_SLOT_DIM,
    typed_slot_vector,
)


SELECTED_CONTEXT_NAMES = (
    "ring5",
    "ring6",
    "aromatic_ring",
    "multi_ring",
    "ring_boundary",
)


def context_mass_readout(
    codes: np.ndarray,
    patch_contexts: np.ndarray,
    selected_names: Sequence[str] = SELECTED_CONTEXT_NAMES,
) -> np.ndarray:
    """Return context-conditioned absolute activation mass plus coverage.

    For each selected context and atom, activation mass is normalized by the
    graph's total absolute sparse-code mass. This preserves both which atoms
    fire and where they fire. The final scalar per context is patch coverage.
    """
    if patch_contexts.ndim != 2 or patch_contexts.shape[1] != len(RING_CONTEXT_NAMES):
        raise ValueError(f"invalid patch context shape: {patch_contexts.shape}")
    if codes.ndim != 2 or codes.shape[1] != patch_contexts.shape[0]:
        raise ValueError(
            f"code/context mismatch: codes={codes.shape}, contexts={patch_contexts.shape}"
        )
    absolute = np.abs(codes)
    denominator = max(float(absolute.sum()), 1e-12)
    name_to_index = {name: index for index, name in enumerate(RING_CONTEXT_NAMES)}
    blocks: list[np.ndarray] = []
    coverage: list[float] = []
    for name in selected_names:
        mask = patch_contexts[:, name_to_index[name]]
        if np.any(mask):
            blocks.append(absolute[:, mask].sum(axis=1) / denominator)
        else:
            blocks.append(np.zeros(codes.shape[0], dtype=np.float64))
        coverage.append(float(mask.mean()) if mask.size else 0.0)
    return np.concatenate([*blocks, np.asarray(coverage, dtype=np.float64)])


def _vectorize_graphs_with_context(
    bundle: MolhivBundle,
    cfg: GraphLevelConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("typed reconstruction and ring contexts require OGB features")
    matrices: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for graph_index, graph in enumerate(bundle.graphs):
        local_seed = cfg.seed + graph_index * 13
        sampled, sample_meta = sample_patches_graph_level(graph, cfg, seed=local_seed)
        node_features = bundle.node_feats[graph_index]
        edge_features = bundle.edge_feats[graph_index]
        context_index = build_ring_context_index(graph, node_features)
        columns = [
            typed_slot_vector(graph, nodes, node_features, edge_features)
            for nodes in sampled.node_sets
        ]
        masks = [context_index.patch_mask(nodes) for nodes in sampled.node_sets]
        matrix = (
            np.stack(columns, axis=1)
            if columns
            else np.zeros((TYPED_SLOT_DIM, 0), dtype=np.float64)
        )
        context_matrix = (
            np.stack(masks, axis=0)
            if masks
            else np.zeros((0, len(RING_CONTEXT_NAMES)), dtype=bool)
        )
        if matrix.shape[1]:
            norms = np.linalg.norm(matrix, axis=0, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
            if (
                cfg.max_patches_per_graph is not None
                and matrix.shape[1] > cfg.max_patches_per_graph
            ):
                rng = np.random.default_rng(local_seed)
                chosen = np.sort(
                    rng.choice(
                        matrix.shape[1],
                        size=cfg.max_patches_per_graph,
                        replace=False,
                    )
                )
                matrix = matrix[:, chosen]
                context_matrix = context_matrix[chosen]
        matrices.append(matrix)
        contexts.append(context_matrix)
        metadata.append(
            {
                "graph_index": graph_index,
                "n_patches": int(matrix.shape[1]),
                "context_coverage": {
                    name: (
                        float(context_matrix[:, index].mean())
                        if context_matrix.shape[0]
                        else 0.0
                    )
                    for index, name in enumerate(RING_CONTEXT_NAMES)
                },
                **sample_meta,
            }
        )
    return matrices, contexts, metadata


def _dictionary_t_views(
    patch_matrices: Sequence[np.ndarray],
    patch_contexts: Sequence[np.ndarray],
    dictionary: np.ndarray,
    cfg: GraphLevelConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows: dict[str, list[np.ndarray]] = {
        "raw": [],
        "reconstruction": [],
        "atom_signed": [],
        "atom_abs": [],
        "abs_mean": [],
        "code_mean": [],
        "abs_code_mean": [],
        "rich_code": [],
        "residual_abs": [],
        "residual_signed": [],
    }
    context_rows: list[np.ndarray] = []
    errors: list[float] = []
    for matrix, contexts in zip(patch_matrices, patch_contexts, strict=True):
        if matrix.shape[1]:
            encoded, codes = sparse_code_patch_matrix(matrix, dictionary, cfg)
            reconstructed = dictionary @ codes
            residual = encoded - reconstructed
            code_readouts = sparse_code_readouts(codes)
            rows["raw"].append(encoded.mean(axis=1))
            rows["reconstruction"].append(reconstructed.mean(axis=1))
            rows["atom_signed"].append((dictionary @ np.abs(codes)).mean(axis=1))
            rows["atom_abs"].append((np.abs(dictionary) @ np.abs(codes)).mean(axis=1))
            rows["abs_mean"].append(np.abs(reconstructed).mean(axis=1))
            rows["code_mean"].append(codes.mean(axis=1))
            rows["abs_code_mean"].append(np.abs(codes).mean(axis=1))
            rows["rich_code"].append(code_readouts["rich_no_recon"])
            rows["residual_abs"].append(np.abs(residual).mean(axis=1))
            rows["residual_signed"].append(residual.mean(axis=1))
            context_rows.append(context_mass_readout(codes, contexts))
            errors.append(
                float(
                    np.linalg.norm(residual)
                    / max(np.linalg.norm(encoded), 1e-12)
                )
            )
        else:
            rows["raw"].append(np.zeros(TYPED_SLOT_DIM, dtype=np.float64))
            rows["reconstruction"].append(np.zeros(TYPED_SLOT_DIM, dtype=np.float64))
            rows["atom_signed"].append(np.zeros(TYPED_SLOT_DIM, dtype=np.float64))
            rows["atom_abs"].append(np.zeros(TYPED_SLOT_DIM, dtype=np.float64))
            rows["abs_mean"].append(np.zeros(TYPED_SLOT_DIM, dtype=np.float64))
            rows["code_mean"].append(np.zeros(dictionary.shape[1], dtype=np.float64))
            rows["abs_code_mean"].append(np.zeros(dictionary.shape[1], dtype=np.float64))
            rows["rich_code"].append(np.zeros(10 * dictionary.shape[1], dtype=np.float64))
            rows["residual_abs"].append(np.zeros(TYPED_SLOT_DIM, dtype=np.float64))
            rows["residual_signed"].append(np.zeros(TYPED_SLOT_DIM, dtype=np.float64))
            context_rows.append(
                np.zeros(
                    len(SELECTED_CONTEXT_NAMES) * dictionary.shape[1]
                    + len(SELECTED_CONTEXT_NAMES)
                )
            )
            errors.append(0.0)
    result = {name: np.stack(values).astype(np.float32) for name, values in rows.items()}
    context_mass = np.stack(context_rows).astype(np.float32)
    result["context_mass"] = context_mass
    return result, {
        "raw_dimension": int(result["raw"].shape[1]),
        "reconstruction_dimension": int(result["reconstruction"].shape[1]),
        "code_dimension": int(result["code_mean"].shape[1]),
        "rich_code_dimension": int(result["rich_code"].shape[1]),
        "context_dimension": int(context_mass.shape[1]),
        "context_schema": f"{len(SELECTED_CONTEXT_NAMES)} contexts x K + coverage",
        "selected_contexts": list(SELECTED_CONTEXT_NAMES),
        "mean_graph_reconstruction_error": float(np.mean(errors)),
    }


def _dictionary_views(
    patch_matrices: Sequence[np.ndarray],
    patch_contexts: Sequence[np.ndarray],
    dictionary: np.ndarray,
    cfg: GraphLevelConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Backward-compatible reconstruction/context view wrapper."""
    views, metadata = _dictionary_t_views(patch_matrices, patch_contexts, dictionary, cfg)
    return views["reconstruction"], views["context_mass"], metadata


def assemble_mentor_views(
    composition: np.ndarray,
    reconstruction: np.ndarray,
    context_mass: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the mentor script's seven S/T/A aliases."""
    return {
        "s": composition,
        "t": reconstruction,
        "a": context_mass,
        "st": np.concatenate([composition, reconstruction], axis=1),
        "sa": np.concatenate([composition, context_mass], axis=1),
        "ta": np.concatenate([reconstruction, context_mass], axis=1),
        "sta": np.concatenate([composition, reconstruction, context_mass], axis=1),
    }


def run(config_path: Path, result_dir: Path) -> dict[str, Any]:
    import yaml

    config = json.loads(json.dumps(yaml.safe_load(config_path.read_text(encoding="utf-8"))))
    data_config = dict(config["data"])
    bundle = load_molhiv(
        root=_resolve_repo_path(data_config["root"]),
        max_graphs=(
            None if data_config.get("max_graphs") is None else int(data_config["max_graphs"])
        ),
        seed=int(data_config.get("subsample_seed", 0)),
        with_features=True,
    )
    cfg = graph_level_config(config)
    composition = np.stack([structural_69(graph) for graph in bundle.graphs])
    patch_matrices, patch_contexts, sampling_meta = _vectorize_graphs_with_context(
        bundle, cfg
    )
    train_patches, leakage_audit = collect_train_patch_matrix(
        patch_matrices,
        bundle.split["train"],
        cfg.max_train_patches,
        cfg.seed,
    )
    initial, _, initial_info = ksvd(
        train_patches,
        n_atoms=cfg.n_atoms,
        T=cfg.T,
        T_min=cfg.T_min,
        n_iter=0,
        seed=cfg.seed,
    )
    final, _, final_info = ksvd(
        train_patches,
        n_atoms=cfg.n_atoms,
        T=cfg.T,
        T_min=cfg.T_min,
        n_iter=cfg.ksvd_iter,
        seed=cfg.seed,
        initial_dictionary=initial,
    )
    init_views, init_meta = _dictionary_t_views(
        patch_matrices, patch_contexts, initial, cfg
    )
    final_views, final_meta = _dictionary_t_views(
        patch_matrices, patch_contexts, final, cfg
    )
    t_final = final_views["reconstruction"]
    a_final = final_views["context_mass"]
    t_init = init_views["reconstruction"]
    a_init = init_views["context_mass"]
    available = assemble_mentor_views(composition, t_final, a_final)
    available.update(
        {
            "t_init": t_init,
            "a_init": a_init,
            "st_init": np.concatenate([composition, t_init], axis=1),
            "sa_init": np.concatenate([composition, a_init], axis=1),
            "sta_init": np.concatenate([composition, t_init, a_init], axis=1),
        }
    )
    for prefix, feature_views in (("init", init_views), ("final", final_views)):
        for name, matrix in feature_views.items():
            available[f"t_{prefix}_{name}"] = matrix
    available.update(
        {
            "s_t_init_raw": np.concatenate([composition, init_views["raw"]], axis=1),
            "s_t_final_raw": np.concatenate([composition, final_views["raw"]], axis=1),
            "s_t_init_reconstruction": np.concatenate(
                [composition, init_views["reconstruction"]], axis=1
            ),
            "s_t_final_reconstruction": np.concatenate(
                [composition, final_views["reconstruction"]], axis=1
            ),
            "s_t_init_rich_code": np.concatenate(
                [composition, init_views["rich_code"]], axis=1
            ),
            "s_t_final_rich_code": np.concatenate(
                [composition, final_views["rich_code"]], axis=1
            ),
            "s_t_init_residual_abs": np.concatenate(
                [composition, init_views["residual_abs"]], axis=1
            ),
            "s_t_final_residual_abs": np.concatenate(
                [composition, final_views["residual_abs"]], axis=1
            ),
        }
    )
    requested = [str(name) for name in config["views"]]
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unknown views: {missing}")
    views = {name: available[name] for name in requested}
    records, fitted = fit_xgboost_views(
        views,
        bundle.y,
        bundle.split,
        dict(config["classifier"]),
        evaluate_test=False,
    )
    context_dim = len(SELECTED_CONTEXT_NAMES) * cfg.n_atoms + len(SELECTED_CONTEXT_NAMES)
    manifest = {
        "protocol_id": str(config["protocol_id"]),
        "status": "full" if data_config.get("max_graphs") is None else "development",
        "official_test_evaluated": False,
        "mentor_shape_alignment": {
            "composition": 69,
            "recon_typed": TYPED_SLOT_DIM,
            "K": cfg.n_atoms,
            "selected_context_mass": context_dim,
            "sta": int(available["sta"].shape[1]),
        },
        "not_exact_mentor_feature_replication": True,
        "known_schema_differences": [
            "S is a pure-structure 69D proxy, not the unknown mentor composition schema",
            "T is the local 8-slot typed reconstruction proxy",
            "A uses explicit local definitions for the five mentor-named ring contexts",
        ],
        "resolved_config": config,
        "data": bundle.meta,
        "leakage_audit": leakage_audit,
        "sampling": {
            "mean_patches": float(np.mean([matrix.shape[1] for matrix in patch_matrices])),
            "mean_context_coverage": {
                name: float(
                    np.mean([item["context_coverage"][name] for item in sampling_meta])
                )
                for name in RING_CONTEXT_NAMES
            },
        },
        "dictionary_info": {"initial": initial_info, "final": final_info},
        "features": {"initial": init_meta, "final": final_meta},
        "classifier": fitted["summary"],
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_records(result_dir / "records.csv", records)
    _write_json(result_dir / "summary.json", manifest)
    train_valid = np.concatenate(
        [bundle.split["train"], bundle.split["valid"]]
    ).astype(np.int64)
    payload: dict[str, np.ndarray] = {
        "dataset_indices": train_valid,
        "labels": bundle.y[train_valid].astype(np.int64),
        "train_count": np.asarray([len(bundle.split["train"])], dtype=np.int64),
    }
    payload.update({name: matrix[train_valid] for name, matrix in views.items()})
    np.savez_compressed(result_dir / "feature_views_train_valid.npz", **payload)
    (result_dir / "MENTOR_RING_CONTEXT_PROXY.md").write_text(
        _render_summary(manifest), encoding="utf-8"
    )
    return manifest


def _render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Mentor-shaped ring-context proxy",
        "",
        f"Protocol: `{payload['protocol_id']}`",
        "",
        "This is a shape-aligned proxy, not an exact reproduction of the unknown upstream feature builders.",
        "",
        "| view | dim | valid ROC-AUC |",
        "|---|---:|---:|",
    ]
    for name, result in payload["classifier"]["per_view"].items():
        valid = result["valid"]
        lines.append(
            f"| {name} | {result['dimension']} | "
            f"{valid['mean']:.6f} ± {valid['std_sample']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Primary attribution: `st-s`, `sa-s`, `sta-max(st,sa)`, and `sta-sta_init`.",
            "Official test is not evaluated.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.config.expanduser().resolve(), args.result_dir.expanduser().resolve())
    print(
        json.dumps(
            {
                "protocol_id": result["protocol_id"],
                "status": result["status"],
                "dimensions": result["mentor_shape_alignment"],
                "best_by_valid_only": result["classifier"]["best_by_valid_only"],
                "test_evaluated": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
