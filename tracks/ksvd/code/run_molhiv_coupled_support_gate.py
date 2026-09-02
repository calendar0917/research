"""Mechanism gate for a structure--chemistry shared-support dictionary.

This is intentionally *not* a classifier experiment.  It asks whether paired
structural and chemical patch descriptions have a reproducible shared local
vocabulary before adding a downstream learner.  All fitting uses the training
side of one fixed MolHIV scaffold fold; the validation side is never used to
choose a dictionary.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .coupled_support_dictionary import (
    coupled_sparse_encode,
    fit_coupled_support_dictionary,
    relative_reconstruction,
    support_agreement,
)
from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig, bundle_to_Y, sample_patches_graph_level
from .ksvd import _omp, ksvd
from .vectorize import patch_feature_dim


@dataclass
class TwoViewGraph:
    index: int
    label: float
    structure: np.ndarray
    chemistry: np.ndarray


def _resolve(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    return Path(__file__).resolve().parents[3] / path


def _pick(indices: np.ndarray, labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Deterministically downsample while retaining MolHIV's label balance."""
    indices = np.asarray(indices, dtype=np.int64)
    if maximum >= len(indices):
        return indices.copy()
    rng = np.random.default_rng(seed)
    pos, neg = indices[labels[indices] > 0.5], indices[labels[indices] <= 0.5]
    take_pos = min(len(pos), int(round(maximum * len(pos) / len(indices))))
    take_neg = min(len(neg), maximum - take_pos)
    take_pos = min(len(pos), maximum - take_neg)
    answer = np.concatenate([
        rng.choice(pos, size=take_pos, replace=False),
        rng.choice(neg, size=take_neg, replace=False),
    ])
    rng.shuffle(answer)
    return answer


def _one_graph(bundle, index: int, cfg: GraphLevelConfig) -> TwoViewGraph | None:
    graph = bundle.graphs[index]
    sampled, _ = sample_patches_graph_level(graph, cfg, seed=cfg.seed + index)
    patches = sampled.node_sets[: cfg.max_patches_per_graph]
    if len(patches) < 2:
        return None
    holder = SimpleNamespace(node_sets=patches)
    structure, structure_meta = bundle_to_Y(
        graph, holder, cfg.max_nodes, cfg.order_mode, patch_feat="wl"
    )
    chemistry_full, chemistry_meta = bundle_to_Y(
        graph,
        holder,
        cfg.max_nodes,
        cfg.order_mode,
        patch_feat="wl_chem_ring",
        node_feat=bundle.node_feats[index],
        edge_feat=bundle.edge_feats[index],
    )
    if not structure_meta["n_cols"] or not chemistry_meta["n_cols"]:
        return None
    if structure.shape[1] != chemistry_full.shape[1]:
        raise RuntimeError("the two patch views lost their one-to-one alignment")
    # The first wl-width coordinates of wl_chem_ring contain no atom/bond
    # labels.  The remainder is the chemical view: atom/bond histograms,
    # label-aware local messages, and ring/aromatic facts.
    structural_width = patch_feature_dim(cfg.max_nodes, "wl")
    chemistry = chemistry_full[structural_width:]
    if chemistry.shape[0] <= 0:
        raise RuntimeError("chemical view unexpectedly has no coordinates")
    return TwoViewGraph(index, float(bundle.y[index]), structure, chemistry)


def _collect(bundle, indices, cfg, maximum: int, seed: int) -> list[TwoViewGraph]:
    records = []
    for index in _pick(np.asarray(indices), bundle.y, maximum, seed):
        record = _one_graph(bundle, int(index), cfg)
        if record is not None:
            records.append(record)
    if len(records) < 2:
        raise RuntimeError("too few usable molecules")
    return records


def _columns(records: list[TwoViewGraph]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([record.structure for record in records], axis=1),
        np.concatenate([record.chemistry for record in records], axis=1),
    )


def _fit_scaler(structure: np.ndarray, chemistry: np.ndarray) -> dict[str, np.ndarray | float]:
    return {
        "structure_mean": structure.mean(axis=1, keepdims=True),
        "chemistry_mean": chemistry.mean(axis=1, keepdims=True),
        "structure_rms": max(float(np.sqrt(np.mean(np.sum((structure - structure.mean(axis=1, keepdims=True)) ** 2, axis=0)))), 1e-12),
        "chemistry_rms": max(float(np.sqrt(np.mean(np.sum((chemistry - chemistry.mean(axis=1, keepdims=True)) ** 2, axis=0)))), 1e-12),
    }


def _scale(values: np.ndarray, mean: np.ndarray, rms: float) -> np.ndarray:
    centered = (values - mean) / rms
    return centered / np.maximum(np.linalg.norm(centered, axis=0, keepdims=True), 1e-12)


def _apply_scaler(
    records: list[TwoViewGraph], scaler: dict[str, np.ndarray | float]
) -> tuple[np.ndarray, np.ndarray]:
    structure, chemistry = _columns(records)
    return (
        _scale(structure, scaler["structure_mean"], float(scaler["structure_rms"])),
        _scale(chemistry, scaler["chemistry_mean"], float(scaler["chemistry_rms"])),
    )


def _codes(dictionary: np.ndarray, values: np.ndarray, sparsity: int) -> np.ndarray:
    return np.stack([_omp(dictionary, values[:, i], sparsity) for i in range(values.shape[1])], axis=1)


def _derangement(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(count)
    if count < 2:
        return order
    for _ in range(32):
        trial = rng.permutation(count)
        if not np.any(trial == order):
            return trial
    return np.roll(order, 1)


def _independent_alignment(ds: np.ndarray, dc: np.ndarray, S: np.ndarray, C: np.ndarray, sparsity: int, seed: int) -> dict[str, float]:
    zs, zc = _codes(ds, S, sparsity), _codes(dc, C, sparsity)
    true = support_agreement(zs, zc)
    shuffled = support_agreement(zs, zc[:, _derangement(C.shape[1], seed)])
    return {"true": true, "shuffled": shuffled, "true_minus_shuffled": true - shuffled}


def _joint_mapping_gap(ds: np.ndarray, dc: np.ndarray, S: np.ndarray, C: np.ndarray, sparsity: int, seed: int) -> dict[str, object]:
    zs, zc = coupled_sparse_encode(S, C, ds, dc, sparsity)
    true = relative_reconstruction(S, C, ds, dc, zs, zc)
    C_wrong = C[:, _derangement(C.shape[1], seed)]
    zs_wrong, zc_wrong = coupled_sparse_encode(S, C_wrong, ds, dc, sparsity)
    wrong = relative_reconstruction(S, C_wrong, ds, dc, zs_wrong, zc_wrong)
    return {
        "true": true,
        "shuffled": wrong,
        "shuffled_minus_true_joint": float(wrong["joint"] - true["joint"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--fit-graphs", type=int, default=800)
    parser.add_argument("--eval-graphs", type=int, default=800)
    parser.add_argument("--n-atoms", type=int, default=16)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--ksvd-iter", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()
    bundle = load_molhiv(with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV atom and bond features are required")
    folds = np.load(_resolve(args.fold_cache))
    train_indices = folds[f"fold_{args.fold}_train_indices"]
    valid_indices = folds[f"fold_{args.fold}_valid_indices"]
    cfg = GraphLevelConfig(seed=args.seed, max_patches_per_graph=8, normalize_patches=False)
    train = _collect(bundle, train_indices, cfg, args.fit_graphs, args.seed + 101)
    valid = _collect(bundle, valid_indices, cfg, args.eval_graphs, args.seed + 202)
    S_train_raw, C_train_raw = _columns(train)
    scaler = _fit_scaler(S_train_raw, C_train_raw)
    S_train, C_train = _apply_scaler(train, scaler)
    S_valid, C_valid = _apply_scaler(valid, scaler)

    # Baseline 1: independent two-view KSVD.  Its atom indices are unrelated,
    # so an agreement score should not arise just from the coding procedure.
    Ds_ind, _, ind_s_info = ksvd(S_train, n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=args.seed)
    Dc_ind, _, ind_c_info = ksvd(C_train, n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=args.seed + 1)

    # Baseline 2: ordinary concatenation.  Its single code is deliberately
    # stricter than the proposed model because it equates coefficient values.
    D_concat, _, concat_info = ksvd(np.vstack([S_train, C_train]), n_atoms=args.n_atoms, T=args.sparsity, T_min=1, n_iter=args.ksvd_iter, seed=args.seed + 2)
    split = S_train.shape[0]
    Ds_initial, Dc_initial = D_concat[:split], D_concat[split:]

    Ds_coupled, Dc_coupled, coupled_info = fit_coupled_support_dictionary(
        S_train, C_train, n_atoms=args.n_atoms, sparsity=args.sparsity,
        n_iter=args.ksvd_iter, seed=args.seed + 3,
        initial_structure=Ds_initial / np.maximum(np.linalg.norm(Ds_initial, axis=0, keepdims=True), 1e-12),
        initial_chemistry=Dc_initial / np.maximum(np.linalg.norm(Dc_initial, axis=0, keepdims=True), 1e-12),
    )
    shuffled = _derangement(C_train.shape[1], args.seed + 404)
    Ds_shuffle, Dc_shuffle, shuffle_info = fit_coupled_support_dictionary(
        S_train, C_train[:, shuffled], n_atoms=args.n_atoms, sparsity=args.sparsity,
        n_iter=args.ksvd_iter, seed=args.seed + 3,
        initial_structure=Ds_initial / np.maximum(np.linalg.norm(Ds_initial, axis=0, keepdims=True), 1e-12),
        initial_chemistry=Dc_initial[:,] / np.maximum(np.linalg.norm(Dc_initial, axis=0, keepdims=True), 1e-12),
    )

    def concat_reconstruction(S: np.ndarray, C: np.ndarray) -> dict[str, float]:
        Z = _codes(D_concat, np.vstack([S, C]), args.sparsity)
        return relative_reconstruction(S, C, D_concat[:split], D_concat[split:], Z, Z)

    def independent_reconstruction(S: np.ndarray, C: np.ndarray) -> dict[str, float]:
        return relative_reconstruction(S, C, Ds_ind, Dc_ind, _codes(Ds_ind, S, args.sparsity), _codes(Dc_ind, C, args.sparsity))

    def coupled_reconstruction(Ds: np.ndarray, Dc: np.ndarray, S: np.ndarray, C: np.ndarray) -> dict[str, float]:
        zs, zc = coupled_sparse_encode(S, C, Ds, Dc, args.sparsity)
        return relative_reconstruction(S, C, Ds, Dc, zs, zc)

    result = {
        "protocol_id": "molhiv-structure-chemistry-shared-support-gate-v1",
        "config": vars(args),
        "design": {
            "structure_view": "unlabeled WL local topology; no atom or bond labels",
            "chemistry_view": "atom/bond histograms, label-aware local messages, and ring/aromatic facts; the unlabeled topology prefix is removed",
            "proposal": "paired KSVD atoms share selected identities but have separate structural and chemical coefficients",
            "fit_guard": "all scalers and dictionaries use training-fold molecules only",
            "negative_control": "the training patch-to-chemistry correspondence is deranged with no fixed patch; both view bags are otherwise unchanged",
        },
        "counts": {
            "train_graphs": len(train), "valid_graphs": len(valid),
            "train_patches": int(S_train.shape[1]), "valid_patches": int(S_valid.shape[1]),
            "structure_dim": int(S_train.shape[0]), "chemistry_dim": int(C_train.shape[0]),
        },
        "fit": {
            "independent_structure": ind_s_info, "independent_chemistry": ind_c_info,
            "concatenated": concat_info, "coupled": coupled_info,
            "coupled_shuffled_training": shuffle_info,
        },
        "validation": {
            "independent_reconstruction": independent_reconstruction(S_valid, C_valid),
            "concatenated_shared_coefficient_reconstruction": concat_reconstruction(S_valid, C_valid),
            "coupled_shared_support_reconstruction": coupled_reconstruction(Ds_coupled, Dc_coupled, S_valid, C_valid),
            "coupled_shuffled_training_reconstruction": coupled_reconstruction(Ds_shuffle, Dc_shuffle, S_valid, C_valid),
            "independent_support_alignment": _independent_alignment(Ds_ind, Dc_ind, S_valid, C_valid, args.sparsity, args.seed + 505),
            "coupled_support_alignment": _independent_alignment(Ds_coupled, Dc_coupled, S_valid, C_valid, args.sparsity, args.seed + 506),
            "coupled_shuffled_training_support_alignment": _independent_alignment(Ds_shuffle, Dc_shuffle, S_valid, C_valid, args.sparsity, args.seed + 507),
            "coupled_true_mapping_gap": _joint_mapping_gap(Ds_coupled, Dc_coupled, S_valid, C_valid, args.sparsity, args.seed + 606),
            "coupled_shuffled_training_mapping_gap": _joint_mapping_gap(Ds_shuffle, Dc_shuffle, S_valid, C_valid, args.sparsity, args.seed + 607),
        },
        "elapsed_sec": time.time() - started,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["validation"], indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
