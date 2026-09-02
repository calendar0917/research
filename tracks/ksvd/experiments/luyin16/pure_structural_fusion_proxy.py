"""Dimension-controlled pure-structure S + K-SVD proxy.

This experiment deliberately removes atom/bond chemistry from both sides:
``S_struct`` is a 69-D graph-only descriptor and ``R_struct`` is a 624-D
readout made from 26 fixed sparse-code summaries for each of 24 K-SVD atoms.
The 624=26x24 factorization is a testable proxy motivated by the historical
K24/T3 settings, not a claim about the mentor's unknown schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ksvd_research.core import ksvd
from ksvd_research.data import load_molhiv
from ksvd_research.evaluation import GraphLevelConfig, sparse_code_patch_matrix
from tracks.ksvd.experiments.luyin16.mentor_concept_replication import (
    REPO_ROOT,
    _resolve_repo_path,
    _topology_row,
    collect_train_patch_matrix,
    vectorize_graphs,
)


STRUCT_DIM = 69
READOUT_DIM = 624
N_ATOMS = 24
SPARSITY = 3
SUMMARY_DIM = 26


def _degree_bin(value: int) -> int:
    return min(max(int(value), 0), 4)


def _distance_hist(graph: Any) -> np.ndarray:
    nodes = list(graph.nodes)
    out = np.zeros(15, dtype=np.float64)
    if len(nodes) < 2:
        return out
    pairs = 0
    for source in nodes:
        distances = {source: 0}
        queue = [source]
        for current in queue:
            for nxt in graph.neighbors(current):
                if nxt not in distances:
                    distances[nxt] = distances[current] + 1
                    queue.append(nxt)
        for target in nodes:
            if target <= source:
                continue
            pairs += 1
            distance = distances.get(target)
            if distance is None:
                out[-1] += 1.0
            else:
                out[min(max(distance, 1), 14) - 1] += 1.0
    return out / max(float(pairs), 1.0)


def structural_69(graph: Any) -> np.ndarray:
    """Return 18 topology + 21 degree histogram + 15 degree-pair + 15 distance bins."""
    degrees = {node: len(graph.neighbors(node)) for node in graph.nodes}
    degree_hist = np.zeros(21, dtype=np.float64)
    for degree in degrees.values():
        degree_hist[min(int(degree), 20)] += 1.0
    degree_hist /= max(float(len(degrees)), 1.0)

    pair_hist = np.zeros(15, dtype=np.float64)
    pair_index = 0
    for left_bin in range(5):
        for right_bin in range(left_bin, 5):
            value = 0.0
            for left, right in graph.edges():
                a, b = sorted((_degree_bin(degrees[left]), _degree_bin(degrees[right])))
                if (a, b) == (left_bin, right_bin):
                    value += 1.0
            pair_hist[pair_index] = value
            pair_index += 1
    pair_hist /= max(float(graph.num_edges()), 1.0)

    row = np.concatenate([_topology_row(graph), degree_hist, pair_hist, _distance_hist(graph)])
    if row.shape != (STRUCT_DIM,) or not np.all(np.isfinite(row)):
        raise RuntimeError(f"invalid structural descriptor shape={row.shape}")
    return row.astype(np.float32)


def _coefficient_summary(codes: np.ndarray, errors: np.ndarray | None) -> np.ndarray:
    """26 fixed per-atom sparse-code summaries; no chemistry or labels."""
    atoms, count = codes.shape
    out = np.zeros((atoms, SUMMARY_DIM), dtype=np.float64)
    if count == 0:
        return out.reshape(-1)
    absolute = np.abs(codes)
    quantiles = np.quantile(codes, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1)
    abs_quantiles = np.quantile(absolute, [0.50, 0.75, 0.90], axis=1)
    top_sorted = np.sort(absolute, axis=1)
    top = [top_sorted[:, -min(k, count) :].mean(axis=1) for k in (1, 3, 5)]
    out[:, 0] = codes.mean(axis=1)
    out[:, 1] = absolute.mean(axis=1)
    out[:, 2] = codes.std(axis=1)
    out[:, 3] = absolute.std(axis=1)
    out[:, 4:9] = quantiles.T
    out[:, 9:12] = abs_quantiles.T
    out[:, 12] = absolute.min(axis=1)
    out[:, 13] = absolute.max(axis=1)
    out[:, 14] = (absolute > 1e-10).mean(axis=1)
    out[:, 15] = (codes > 1e-10).mean(axis=1)
    out[:, 16] = (codes < -1e-10).mean(axis=1)
    out[:, 17] = np.square(codes).mean(axis=1)
    out[:, 18] = np.sqrt(np.square(codes).mean(axis=1))
    out[:, 19] = top[0]
    out[:, 20] = top[1]
    out[:, 21] = top[2]
    out[:, 22] = np.bincount(np.argmax(absolute, axis=0), minlength=atoms) / float(count)
    out[:, 23] = np.sum(absolute, axis=1) / float(count)
    out[:, 24] = np.mean(np.sign(codes) * absolute, axis=1)
    if errors is not None and errors.size:
        out[:, 25] = float(np.mean(errors))
    return out.reshape(-1)


def sparse_readout_624(patch_matrices: Sequence[np.ndarray], dictionary: np.ndarray, cfg: GraphLevelConfig) -> np.ndarray:
    rows: list[np.ndarray] = []
    for matrix in patch_matrices:
        if matrix.shape[1]:
            encoded, codes = sparse_code_patch_matrix(matrix, dictionary, cfg)
            errors = np.linalg.norm(encoded - dictionary @ codes, axis=0) / np.maximum(np.linalg.norm(encoded, axis=0), 1e-12)
        else:
            codes = np.zeros((dictionary.shape[1], 0), dtype=np.float64)
            errors = np.empty(0, dtype=np.float64)
        row = _coefficient_summary(codes, errors)
        if row.shape != (READOUT_DIM,):
            raise RuntimeError(f"invalid readout shape={row.shape}")
        rows.append(row)
    return np.stack(rows, axis=0).astype(np.float32)


def run(result_dir: Path) -> dict[str, Any]:
    bundle = load_molhiv(root=_resolve_repo_path("data/ogb"), with_features=True)
    cfg = GraphLevelConfig(
        p=0.5, q=2.0, walk_length=8, max_nodes=8, max_walks=8,
        edge_decay=0.7, seed_policy="degree_stratified", cover_target=0.9,
        no_backtrack=True, n_atoms=N_ATOMS, T=SPARSITY, T_min=1,
        ksvd_iter=5, seed=0, max_train_patches=6000, patch_feat="topo",
        normalize_patches=True, max_patches_per_graph=8,
    )
    structural = np.stack([structural_69(graph) for graph in bundle.graphs])
    patch_matrices, sampling_meta = vectorize_graphs(bundle, cfg, mode="coverage")
    train_patches, leakage = collect_train_patch_matrix(patch_matrices, bundle.split["train"], cfg.max_train_patches, cfg.seed)
    initial, _, initial_info = ksvd(train_patches, n_atoms=N_ATOMS, T=SPARSITY, T_min=1, n_iter=0, seed=0)
    final, _, final_info = ksvd(train_patches, n_atoms=N_ATOMS, T=SPARSITY, T_min=1, n_iter=5, seed=0, initial_dictionary=initial)
    r_init = sparse_readout_624(patch_matrices, initial, cfg)
    r_final = sparse_readout_624(patch_matrices, final, cfg)
    views = {
        "s_struct": structural,
        "r_init_struct": r_init,
        "r_final_struct": r_final,
        "s_r_init_struct": np.concatenate([structural, r_init], axis=1),
        "s_r_final_struct": np.concatenate([structural, r_final], axis=1),
    }
    train_valid = np.concatenate([bundle.split["train"], bundle.split["valid"]]).astype(np.int64)
    payload: dict[str, np.ndarray] = {
        "dataset_indices": train_valid,
        "labels": bundle.y[train_valid].astype(np.int64),
        "train_count": np.asarray([len(bundle.split["train"])], dtype=np.int64),
    }
    payload.update({name: matrix[train_valid] for name, matrix in views.items()})
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(result_dir / "feature_views_train_valid.npz", **payload)
    manifest = {
        "protocol_id": "mentor-pure-structural-s-plus-ksvd-k24-t3-v1",
        "official_test_evaluated": False,
        "dimensions": {name: int(matrix.shape[1]) for name, matrix in views.items()},
        "structural_schema": "18 topology + 21 degree histogram + 15 degree-pair histogram + 15 distance histogram",
        "readout_schema": "26 fixed coefficient summaries x 24 atoms = 624",
        "config": {"n_atoms": N_ATOMS, "sparsity": SPARSITY, "ksvd_iter": 5, "patch_feat": "topo"},
        "leakage_audit": leakage,
        "dictionary_info": {"initial": initial_info, "final": final_info},
        "mean_patches": float(np.mean([m.shape[1] for m in patch_matrices])),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=REPO_ROOT / "tracks/ksvd/results/luyin16/pure_structural_fusion_k24_t3")
    args = parser.parse_args()
    run(args.result_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
