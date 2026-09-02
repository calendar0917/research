"""Empirically constrained K-SVD variants.

The learned dictionary is constrained to normalized columns observed in the
training patch matrix.  This makes every atom legal and directly decodable as
an actual training patch.  The constraint is unsupervised and must be fitted
inside each outer-training fold.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import norm, svd
from scipy.optimize import linear_sum_assignment

from .ksvd import _omp, ksvd


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64).copy()
    scales = np.linalg.norm(out, axis=0)
    valid = scales > 1e-12
    out[:, valid] /= scales[valid][None, :]
    out[:, ~valid] = 0.0
    return out


def _unique_legal_candidates(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized, nonzero, signature-distinct training columns.

    `indices[j]` is the original column index represented by candidate `j`.
    Exact byte identity is appropriate here because all candidates come from
    the same deterministic vectorizer and are normalized by the same routine.
    """
    candidates = _normalize_columns(Y)
    keep: list[int] = []
    seen: set[bytes] = set()
    for j in range(candidates.shape[1]):
        col = np.ascontiguousarray(candidates[:, j])
        if norm(col) <= 1e-12:
            continue
        key = col.tobytes()
        if key in seen:
            continue
        seen.add(key)
        keep.append(j)
    if not keep:
        raise ValueError("Y contains no nonzero legal patch candidates")
    idx = np.asarray(keep, dtype=np.int64)
    return candidates[:, idx], idx


def project_dictionary_to_real_patches(
    D: np.ndarray,
    Y: np.ndarray,
    *,
    require_unique_signatures: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project atoms jointly to distinct real training patches.

    Hungarian maximum-weight matching is used on absolute cosine similarity,
    so K-SVD's arbitrary atom sign does not affect the selected legal patch.
    Atom order is preserved.  By default, selected atoms have distinct vector
    signatures, not merely different occurrence indices.
    """
    Dn = _normalize_columns(D)
    if require_unique_signatures:
        candidates, original_indices = _unique_legal_candidates(Y)
    else:
        all_candidates = _normalize_columns(Y)
        valid = np.linalg.norm(all_candidates, axis=0) > 1e-12
        candidates = all_candidates[:, valid]
        original_indices = np.flatnonzero(valid).astype(np.int64)
    n_atoms = Dn.shape[1]
    if candidates.shape[1] < n_atoms:
        raise ValueError(
            f"need at least {n_atoms} distinct legal candidates, got {candidates.shape[1]}"
        )

    similarity = np.abs(Dn.T @ candidates)
    rows, cols = linear_sum_assignment(-similarity)
    if len(rows) != n_atoms:
        raise RuntimeError("Hungarian projection did not assign every atom")
    order = np.empty(n_atoms, dtype=np.int64)
    order[rows] = cols
    selected = candidates[:, order]
    selected_indices = original_indices[order]
    matched = similarity[np.arange(n_atoms), order]
    return selected, {
        "candidate_count": int(candidates.shape[1]),
        "selected_training_indices": selected_indices.tolist(),
        "selected_unique_signatures": int(np.unique(order).size),
        "mean_preprojection_absolute_cosine": float(matched.mean()),
        "minimum_preprojection_absolute_cosine": float(matched.min()),
        "maximum_preprojection_absolute_cosine": float(matched.max()),
        "projectability": 1.0,
    }


def _sparse_code(
    D: np.ndarray,
    Y: np.ndarray,
    T: int,
    T_min: int,
) -> np.ndarray:
    X = np.zeros((D.shape[1], Y.shape[1]), dtype=np.float64)
    for i in range(Y.shape[1]):
        X[:, i] = _omp(D, Y[:, i], T)
        if np.count_nonzero(np.abs(X[:, i]) > 1e-10) < T_min:
            corr = np.abs(D.T @ Y[:, i])
            top = np.argsort(-corr)[:T_min]
            coef, _, _, _ = np.linalg.lstsq(D[:, top], Y[:, i], rcond=None)
            X[:, i] = 0.0
            X[top, i] = coef
    return X


def _dictionary_info(
    D: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_iter: int,
    T: int,
    recon_curve: list[float],
    projection_history: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    offdiag = np.abs(D.T @ D)[~np.eye(D.shape[1], dtype=bool)]
    nnz = np.count_nonzero(np.abs(X) > 1e-10, axis=0)
    return {
        "variant": variant,
        "recon_rel": float(norm(Y - D @ X, "fro") / max(norm(Y, "fro"), 1e-12)),
        "recon_curve": recon_curve,
        "atoms_used": int(np.sum(np.any(np.abs(X) > 1e-10, axis=1))),
        "mean_nnz": float(nnz.mean()) if nnz.size else 0.0,
        "n_atoms": int(D.shape[1]),
        "T": int(T),
        "n_iter": int(n_iter),
        "projectability": 1.0,
        "all_atoms_are_training_patches": True,
        "selected_unique_signatures": int(projection_history[-1]["selected_unique_signatures"]),
        "mean_absolute_offdiagonal_coherence": float(offdiag.mean()) if offdiag.size else 0.0,
        "maximum_absolute_offdiagonal_coherence": float(offdiag.max()) if offdiag.size else 0.0,
        "projection_history": projection_history,
    }


def final_projected_ksvd(
    Y: np.ndarray,
    n_atoms: int = 12,
    T: int = 3,
    n_iter: int = 10,
    seed: int = 0,
    T_min: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run ordinary K-SVD, then project its final atoms to legal patches."""
    Dfree, _Xfree, free_info = ksvd(
        Y,
        n_atoms=n_atoms,
        T=T,
        n_iter=n_iter,
        seed=seed,
        T_min=T_min,
    )
    D, projection = project_dictionary_to_real_patches(Dfree, Y)
    T_eff = min(max(T, T_min), D.shape[1])
    T_min_eff = min(T_min, T_eff)
    X = _sparse_code(D, Y, T_eff, T_min_eff)
    recon = float(norm(Y - D @ X, "fro") / max(norm(Y, "fro"), 1e-12))
    info = _dictionary_info(
        D,
        X,
        Y,
        n_iter=n_iter,
        T=T_eff,
        recon_curve=[recon],
        projection_history=[projection],
        variant="final_projected_ksvd",
    )
    info["unconstrained_recon_rel"] = float(free_info["recon_rel"])
    info["projection_reconstruction_penalty"] = recon - float(free_info["recon_rel"])
    return D, X, info


def iterative_projected_ksvd(
    Y: np.ndarray,
    n_atoms: int = 12,
    T: int = 3,
    n_iter: int = 10,
    seed: int = 0,
    T_min: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """K-SVD with a projection to distinct legal patches after every update."""
    rng = np.random.default_rng(seed)
    n_features, n_samples = Y.shape
    n_atoms_eff = min(n_atoms, max(n_features, 2), max(n_samples, 2))
    candidates, original_indices = _unique_legal_candidates(Y)
    if candidates.shape[1] < n_atoms_eff:
        raise ValueError(
            f"need at least {n_atoms_eff} distinct legal candidates, got {candidates.shape[1]}"
        )
    init_local = rng.choice(candidates.shape[1], size=n_atoms_eff, replace=False)
    D = candidates[:, init_local].copy()
    init_projection = {
        "candidate_count": int(candidates.shape[1]),
        "selected_training_indices": original_indices[init_local].tolist(),
        "selected_unique_signatures": int(n_atoms_eff),
        "mean_preprojection_absolute_cosine": 1.0,
        "minimum_preprojection_absolute_cosine": 1.0,
        "maximum_preprojection_absolute_cosine": 1.0,
        "projectability": 1.0,
        "stage": "initialization",
    }
    T_eff = min(max(T, T_min), n_atoms_eff)
    T_min_eff = min(T_min, T_eff)
    X = np.zeros((n_atoms_eff, n_samples), dtype=np.float64)
    recon_curve: list[float] = []
    history: list[dict[str, Any]] = [init_projection]

    for iteration in range(n_iter):
        X = _sparse_code(D, Y, T_eff, T_min_eff)
        for j in range(n_atoms_eff):
            omega = np.where(np.abs(X[j, :]) > 1e-10)[0]
            if omega.size == 0:
                D[:, j] = Y[:, int(rng.integers(0, n_samples))]
                nj = norm(D[:, j])
                if nj > 1e-12:
                    D[:, j] /= nj
                continue
            E = Y[:, omega] - D @ X[:, omega] + np.outer(D[:, j], X[j, omega])
            try:
                U, S, Vt = svd(E, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            D[:, j] = U[:, 0]
            X[j, omega] = S[0] * Vt[0, :]
        D, projection = project_dictionary_to_real_patches(D, Y)
        projection["stage"] = f"iteration_{iteration + 1}"
        history.append(projection)
        # Projection changes every atom, so stale SVD coefficients are invalid.
        X = _sparse_code(D, Y, T_eff, T_min_eff)
        recon_curve.append(float(norm(Y - D @ X, "fro") / max(norm(Y, "fro"), 1e-12)))

    if n_iter == 0:
        X = _sparse_code(D, Y, T_eff, T_min_eff)
        recon_curve.append(float(norm(Y - D @ X, "fro") / max(norm(Y, "fro"), 1e-12)))
    info = _dictionary_info(
        D,
        X,
        Y,
        n_iter=n_iter,
        T=T_eff,
        recon_curve=recon_curve,
        projection_history=history,
        variant="iterative_projected_ksvd",
    )
    return D, X, info
