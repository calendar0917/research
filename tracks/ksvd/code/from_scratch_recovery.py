"""Minimal recovery experiments for the from-scratch K-SVD route.

This module deliberately excludes graph sampling, permutation handling,
classification, and real datasets.  It tests only whether a shared sparse
basis and its codes can be recovered when the data-generating process is
known.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

import numpy as np

from .ksvd import _omp, ksvd


EPS = 1e-12


@dataclass(frozen=True)
class RecoveryDataset:
    name: str
    D_true: np.ndarray
    X_train_true: np.ndarray
    Y_train: np.ndarray
    X_test_true: np.ndarray
    Y_test: np.ndarray
    metadata: dict[str, Any]
    binary_graph_signal: bool = False
    true_atom_edge_supports: tuple[tuple[int, ...], ...] | None = None


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64).copy()
    scales = np.linalg.norm(out, axis=0)
    valid = scales > EPS
    out[:, valid] /= scales[valid][None, :]
    out[:, ~valid] = 0.0
    return out


def _ensure_atom_singletons(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    coefficient_scales: np.ndarray | None = None,
    signed: bool,
) -> None:
    """Place one guaranteed singleton for every atom at the front of X."""
    n_atoms = X.shape[0]
    scales = (
        np.ones(n_atoms, dtype=np.float64)
        if coefficient_scales is None
        else np.asarray(coefficient_scales, dtype=np.float64)
    )
    if X.shape[1] < n_atoms:
        raise ValueError("need at least one sample per atom")
    for atom in range(n_atoms):
        sign = float(rng.choice([-1.0, 1.0])) if signed else 1.0
        X[:, atom] = 0.0
        X[atom, atom] = sign * scales[atom]


def sample_sparse_codes(
    *,
    n_atoms: int,
    n_samples: int,
    max_sparsity: int,
    rng: np.random.Generator,
    singleton_probability: float = 0.5,
    coefficient_scales: np.ndarray | None = None,
    signed: bool = True,
    random_amplitude: bool = True,
) -> np.ndarray:
    if not 1 <= max_sparsity <= n_atoms:
        raise ValueError("max_sparsity must lie in [1, n_atoms]")
    if not 0.0 <= singleton_probability <= 1.0:
        raise ValueError("singleton_probability must lie in [0, 1]")
    scales = (
        np.ones(n_atoms, dtype=np.float64)
        if coefficient_scales is None
        else np.asarray(coefficient_scales, dtype=np.float64)
    )
    if scales.shape != (n_atoms,):
        raise ValueError("coefficient_scales has the wrong shape")

    X = np.zeros((n_atoms, n_samples), dtype=np.float64)
    for sample in range(n_samples):
        if max_sparsity == 1 or rng.random() < singleton_probability:
            size = 1
        else:
            size = int(rng.integers(2, max_sparsity + 1))
        support = rng.choice(n_atoms, size=size, replace=False)
        amplitudes = (
            rng.uniform(0.5, 1.5, size=size)
            if random_amplitude
            else np.ones(size, dtype=np.float64)
        )
        signs = rng.choice([-1.0, 1.0], size=size) if signed else np.ones(size)
        X[support, sample] = scales[support] * amplitudes * signs

    _ensure_atom_singletons(
        X,
        rng,
        coefficient_scales=scales,
        signed=signed,
    )
    return X


def make_e0_dataset(
    *,
    seed: int = 20260731,
    n_features: int = 15,
    n_atoms: int = 4,
    max_sparsity: int = 2,
    n_train: int = 1000,
    n_test: int = 300,
    noise_std: float = 0.0,
) -> RecoveryDataset:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_features, n_atoms))
    D_true, _ = np.linalg.qr(raw, mode="reduced")
    D_true = normalize_columns(D_true)
    X_train = sample_sparse_codes(
        n_atoms=n_atoms,
        n_samples=n_train,
        max_sparsity=max_sparsity,
        rng=rng,
        singleton_probability=0.5,
        signed=True,
        random_amplitude=True,
    )
    X_test = sample_sparse_codes(
        n_atoms=n_atoms,
        n_samples=n_test,
        max_sparsity=max_sparsity,
        rng=rng,
        singleton_probability=0.5,
        signed=True,
        random_amplitude=True,
    )
    Y_train = D_true @ X_train
    Y_test = D_true @ X_test
    if noise_std > 0.0:
        Y_train = Y_train + noise_std * rng.standard_normal(Y_train.shape)
        Y_test = Y_test + noise_std * rng.standard_normal(Y_test.shape)
    return RecoveryDataset(
        name="E0_numeric_orthogonal",
        D_true=D_true,
        X_train_true=X_train,
        Y_train=Y_train,
        X_test_true=X_test,
        Y_test=Y_test,
        metadata={
            "seed": seed,
            "n_features": n_features,
            "n_atoms": n_atoms,
            "max_sparsity": max_sparsity,
            "n_train": n_train,
            "n_test": n_test,
            "noise_std": noise_std,
            "dictionary": "random_QR_orthonormal",
            "singleton_probability": 0.5,
        },
    )


def upper_triangle_edges(n_nodes: int) -> tuple[tuple[int, int], ...]:
    return tuple((u, v) for u in range(n_nodes) for v in range(u + 1, n_nodes))


def make_fixed_slot_graph_atoms(
    n_nodes: int = 6,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...], dict[str, Any]]:
    if n_nodes != 6:
        raise ValueError("the frozen E1 v0 atom design requires n_nodes=6")
    edges = upper_triangle_edges(n_nodes)
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    named_edges: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
        ("A_triangle_012", ((0, 1), (0, 2), (1, 2))),
        ("B_path_234", ((2, 3), (3, 4))),
        ("C_branch_5_to_0_3", ((0, 5), (3, 5))),
        ("D_cross_14_25", ((1, 4), (2, 5))),
    )
    masks = np.zeros((len(edges), len(named_edges)), dtype=np.float64)
    supports: list[tuple[int, ...]] = []
    for atom, (_name, atom_edges) in enumerate(named_edges):
        idx = tuple(edge_to_index[tuple(sorted(edge))] for edge in atom_edges)
        masks[list(idx), atom] = 1.0
        supports.append(idx)
    if np.any(masks.sum(axis=1) > 1.0):
        raise RuntimeError("frozen E1 v0 atoms must have disjoint edge supports")
    scales = np.linalg.norm(masks, axis=0)
    D_true = masks / scales[None, :]
    metadata = {
        "n_nodes": n_nodes,
        "vector_dimension": len(edges),
        "upper_triangle_edges": [list(edge) for edge in edges],
        "atom_names": [name for name, _ in named_edges],
        "atom_edges": [[list(edge) for edge in atom_edges] for _, atom_edges in named_edges],
        "edge_supports_are_disjoint": True,
    }
    return D_true, scales, tuple(supports), metadata


def make_e1_dataset(
    *,
    seed: int = 20260731,
    max_sparsity: int = 2,
    n_train: int = 1000,
    n_test: int = 300,
) -> RecoveryDataset:
    rng = np.random.default_rng(seed)
    D_true, scales, supports, atom_meta = make_fixed_slot_graph_atoms(6)
    n_atoms = D_true.shape[1]
    X_train = sample_sparse_codes(
        n_atoms=n_atoms,
        n_samples=n_train,
        max_sparsity=max_sparsity,
        rng=rng,
        singleton_probability=0.5,
        coefficient_scales=scales,
        signed=False,
        random_amplitude=False,
    )
    X_test = sample_sparse_codes(
        n_atoms=n_atoms,
        n_samples=n_test,
        max_sparsity=max_sparsity,
        rng=rng,
        singleton_probability=0.5,
        coefficient_scales=scales,
        signed=False,
        random_amplitude=False,
    )
    Y_train = D_true @ X_train
    Y_test = D_true @ X_test
    if not np.all((Y_train == 0.0) | (Y_train == 1.0)):
        raise RuntimeError("E1 train signals must be binary adjacency vectors")
    if not np.all((Y_test == 0.0) | (Y_test == 1.0)):
        raise RuntimeError("E1 test signals must be binary adjacency vectors")
    return RecoveryDataset(
        name="E1_fixed_slot_disjoint_edges",
        D_true=D_true,
        X_train_true=X_train,
        Y_train=Y_train,
        X_test_true=X_test,
        Y_test=Y_test,
        metadata={
            "seed": seed,
            "n_atoms": n_atoms,
            "max_sparsity": max_sparsity,
            "n_train": n_train,
            "n_test": n_test,
            "singleton_probability": 0.5,
            "coefficient_rule": "sqrt(atom_edge_count)",
            **atom_meta,
        },
        binary_graph_signal=True,
        true_atom_edge_supports=supports,
    )


def sparse_code_matrix(D: np.ndarray, Y: np.ndarray, T: int) -> np.ndarray:
    X = np.zeros((D.shape[1], Y.shape[1]), dtype=np.float64)
    for sample in range(Y.shape[1]):
        X[:, sample] = _omp(D, Y[:, sample], T)
    return X


def optimal_atom_alignment(
    D_true: np.ndarray,
    D_learned: np.ndarray,
) -> dict[str, Any]:
    true_n = normalize_columns(D_true)
    learned_n = normalize_columns(D_learned)
    if true_n.shape[1] != learned_n.shape[1]:
        raise ValueError("atom alignment currently requires equal dictionary sizes")
    n_atoms = true_n.shape[1]
    if n_atoms > 8:
        raise ValueError("exhaustive matching is intentionally limited to <=8 atoms")
    similarity = np.abs(true_n.T @ learned_n)
    best_perm: tuple[int, ...] | None = None
    best_score = -np.inf
    for perm in permutations(range(n_atoms)):
        score = float(np.mean([similarity[i, perm[i]] for i in range(n_atoms)]))
        if score > best_score:
            best_score = score
            best_perm = tuple(int(x) for x in perm)
    assert best_perm is not None
    signs = np.asarray(
        [1.0 if float(true_n[:, i] @ learned_n[:, best_perm[i]]) >= 0.0 else -1.0 for i in range(n_atoms)],
        dtype=np.float64,
    )
    matched = np.asarray([similarity[i, best_perm[i]] for i in range(n_atoms)])
    return {
        "true_to_learned": list(best_perm),
        "signs": signs.tolist(),
        "matched_cosines": matched.tolist(),
        "mean_atom_cosine": float(matched.mean()),
        "minimum_atom_cosine": float(matched.min()),
    }


def align_codes_to_truth(X_learned: np.ndarray, alignment: dict[str, Any]) -> np.ndarray:
    perm = np.asarray(alignment["true_to_learned"], dtype=np.int64)
    signs = np.asarray(alignment["signs"], dtype=np.float64)
    return signs[:, None] * X_learned[perm, :]


def support_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    *,
    absolute_threshold: float = 1e-6,
    relative_threshold: float = 1e-3,
) -> dict[str, float]:
    """Compare sparse supports while ignoring numerical OMP residue.

    A coefficient is active when it exceeds both a fixed numerical floor and
    ``relative_threshold`` times the largest coefficient in that sample.
    Ground-truth coefficients in the frozen protocol are at least 0.5, so this
    tolerance cannot erase a real activation, but it does avoid counting
    1e-4-level least-squares residue as a semantic atom occurrence.
    """
    true_scale = np.max(np.abs(X_true), axis=0, keepdims=True)
    pred_scale = np.max(np.abs(X_pred), axis=0, keepdims=True)
    true_cutoff = np.maximum(absolute_threshold, relative_threshold * true_scale)
    pred_cutoff = np.maximum(absolute_threshold, relative_threshold * pred_scale)
    truth = np.abs(X_true) >= true_cutoff
    pred = np.abs(X_pred) >= pred_cutoff
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    return {
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "support_tp": tp,
        "support_fp": fp,
        "support_fn": fn,
        "support_absolute_threshold": float(absolute_threshold),
        "support_relative_threshold": float(relative_threshold),
    }


def binary_reconstruction_metrics(
    Y_true: np.ndarray,
    Y_reconstructed: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    truth = Y_true >= threshold
    pred = Y_reconstructed >= threshold
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    exact = float(np.mean(np.all(truth == pred, axis=0)))
    return {
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "exact_patch_recovery": exact,
    }


def atom_edge_support_metrics(
    D_learned: np.ndarray,
    alignment: dict[str, Any],
    true_supports: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    learned_n = normalize_columns(D_learned)
    perm = np.asarray(alignment["true_to_learned"], dtype=np.int64)
    per_atom: list[float] = []
    predicted_supports: list[list[int]] = []
    for true_atom, support in enumerate(true_supports):
        learned_atom = learned_n[:, perm[true_atom]]
        size = len(support)
        predicted = np.argsort(-np.abs(learned_atom))[:size]
        overlap = len(set(int(x) for x in predicted) & set(support))
        # Equal-size sets imply precision = recall = F1 = overlap / size.
        per_atom.append(float(overlap / max(size, 1)))
        predicted_supports.append([int(x) for x in predicted])
    return {
        "atom_edge_support_f1_per_atom": per_atom,
        "mean_atom_edge_support_f1": float(np.mean(per_atom)),
        "minimum_atom_edge_support_f1": float(np.min(per_atom)),
        "predicted_edge_supports": predicted_supports,
    }


def evaluate_dictionary(
    dataset: RecoveryDataset,
    D: np.ndarray,
    *,
    T: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    X_test = sparse_code_matrix(D, dataset.Y_test, T)
    reconstruction = D @ X_test
    rel = float(
        np.linalg.norm(dataset.Y_test - reconstruction, "fro")
        / max(np.linalg.norm(dataset.Y_test, "fro"), EPS)
    )
    nmse = float(
        np.linalg.norm(dataset.Y_test - reconstruction, "fro") ** 2
        / max(np.linalg.norm(dataset.Y_test, "fro") ** 2, EPS)
    )
    alignment = optimal_atom_alignment(dataset.D_true, D)
    aligned_X = align_codes_to_truth(X_test, alignment)
    coefficient_rel = float(
        np.linalg.norm(dataset.X_test_true - aligned_X, "fro")
        / max(np.linalg.norm(dataset.X_test_true, "fro"), EPS)
    )
    metrics: dict[str, Any] = {
        **alignment,
        "test_reconstruction_relative": rel,
        "test_nmse": nmse,
        "coefficient_relative_error": coefficient_rel,
        **support_metrics(dataset.X_test_true, aligned_X),
    }
    if dataset.binary_graph_signal:
        metrics.update(binary_reconstruction_metrics(dataset.Y_test, reconstruction))
        assert dataset.true_atom_edge_supports is not None
        metrics.update(
            atom_edge_support_metrics(
                D,
                alignment,
                dataset.true_atom_edge_supports,
            )
        )
    return X_test, metrics


def fit_and_evaluate(
    dataset: RecoveryDataset,
    *,
    learner_seed: int,
    T: int = 2,
    n_iter: int = 25,
) -> tuple[np.ndarray, dict[str, Any]]:
    D, _X_train, info = ksvd(
        dataset.Y_train,
        n_atoms=dataset.D_true.shape[1],
        T=T,
        n_iter=n_iter,
        seed=learner_seed,
        T_min=1,
    )
    _X_test, metrics = evaluate_dictionary(dataset, D, T=T)
    metrics["learner_seed"] = int(learner_seed)
    metrics["train_reconstruction_relative"] = float(info["recon_rel"])
    metrics["train_atoms_used"] = int(info["atoms_used"])
    metrics["train_mean_nnz"] = float(info["mean_nnz"])
    return D, metrics


def pairwise_dictionary_stability(dictionaries: list[np.ndarray]) -> dict[str, Any]:
    values: list[float] = []
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(range(len(dictionaries)), 2):
        alignment = optimal_atom_alignment(dictionaries[left], dictionaries[right])
        value = float(alignment["mean_atom_cosine"])
        values.append(value)
        pairs.append({"left": left, "right": right, "matched_atom_cosine": value})
    return {
        "pair_count": len(values),
        "pairwise_matched_atom_cosine_mean": float(np.mean(values)) if values else 1.0,
        "pairwise_matched_atom_cosine_minimum": float(np.min(values)) if values else 1.0,
        "pairs": pairs,
    }
