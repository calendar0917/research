"""Sparse dictionaries with shared supports and view-specific strengths.

Each local patch has a structural view ``S`` and a chemical view ``C``.  A
dictionary atom is therefore a pair ``(d_structure, d_chemical)``.  When a
patch is encoded, both views must choose the same small set of atom *indices*,
but each view gets its own least-squares coefficients.  This is deliberately
different from concatenating the views: concatenation forces the coefficients
themselves to be identical.
"""

from __future__ import annotations

import numpy as np


def _unit_columns(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    return out / np.maximum(np.linalg.norm(out, axis=0, keepdims=True), 1e-12)


def _joint_residual_error(
    structure: np.ndarray,
    chemistry: np.ndarray,
    dictionary_structure: np.ndarray,
    dictionary_chemistry: np.ndarray,
    codes_structure: np.ndarray,
    codes_chemistry: np.ndarray,
) -> float:
    left = np.linalg.norm(structure - dictionary_structure @ codes_structure) ** 2
    right = np.linalg.norm(chemistry - dictionary_chemistry @ codes_chemistry) ** 2
    denom = np.linalg.norm(structure) ** 2 + np.linalg.norm(chemistry) ** 2
    return float(np.sqrt((left + right) / max(denom, 1e-12)))


def coupled_sparse_encode(
    structure: np.ndarray,
    chemistry: np.ndarray,
    dictionary_structure: np.ndarray,
    dictionary_chemistry: np.ndarray,
    sparsity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily choose a common support, then refit both coefficient vectors.

    Correlations are normalized by each residual norm before being added.  This
    keeps the choice from being dominated merely because one view has more
    coordinates or a larger numerical scale.
    """
    S = np.asarray(structure, dtype=np.float64)
    C = np.asarray(chemistry, dtype=np.float64)
    Ds = np.asarray(dictionary_structure, dtype=np.float64)
    Dc = np.asarray(dictionary_chemistry, dtype=np.float64)
    if S.ndim != 2 or C.ndim != 2 or S.shape[1] != C.shape[1]:
        raise ValueError("structure and chemistry must be paired column matrices")
    if Ds.ndim != 2 or Dc.ndim != 2 or Ds.shape[1] != Dc.shape[1]:
        raise ValueError("paired dictionaries must have the same atom count")
    if S.shape[0] != Ds.shape[0] or C.shape[0] != Dc.shape[0]:
        raise ValueError("dictionary and data dimensions do not match")

    atoms, patches = Ds.shape[1], S.shape[1]
    count = min(max(1, int(sparsity)), atoms)
    Zs = np.zeros((atoms, patches), dtype=np.float64)
    Zc = np.zeros((atoms, patches), dtype=np.float64)
    for column in range(patches):
        chosen: list[int] = []
        residual_s, residual_c = S[:, column].copy(), C[:, column].copy()
        for _ in range(count):
            score_s = np.abs(Ds.T @ residual_s) / max(np.linalg.norm(residual_s), 1e-12)
            score_c = np.abs(Dc.T @ residual_c) / max(np.linalg.norm(residual_c), 1e-12)
            score = 0.5 * (score_s + score_c)
            if chosen:
                score[np.asarray(chosen, dtype=np.int64)] = -np.inf
            picked = int(np.argmax(score))
            chosen.append(picked)
            selected = np.asarray(chosen, dtype=np.int64)
            coef_s, *_ = np.linalg.lstsq(Ds[:, selected], S[:, column], rcond=None)
            coef_c, *_ = np.linalg.lstsq(Dc[:, selected], C[:, column], rcond=None)
            residual_s = S[:, column] - Ds[:, selected] @ coef_s
            residual_c = C[:, column] - Dc[:, selected] @ coef_c
        Zs[np.asarray(chosen), column] = coef_s
        Zc[np.asarray(chosen), column] = coef_c
    return Zs, Zc


def fit_coupled_support_dictionary(
    structure: np.ndarray,
    chemistry: np.ndarray,
    *,
    n_atoms: int,
    sparsity: int,
    n_iter: int,
    seed: int,
    initial_structure: np.ndarray | None = None,
    initial_chemistry: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Alternating paired K-SVD updates with a common support per patch.

    An atom update uses the same selected patch columns in both views, while
    the two singular-value updates retain separate coefficient magnitudes.
    Thus atom identity is shared; the strength of its structural and chemical
    expression is not.
    """
    S = np.asarray(structure, dtype=np.float64)
    C = np.asarray(chemistry, dtype=np.float64)
    if S.ndim != 2 or C.ndim != 2 or S.shape[1] != C.shape[1]:
        raise ValueError("structure and chemistry must be paired column matrices")
    if S.shape[1] < 2:
        raise ValueError("at least two paired patches are required")
    atoms = min(int(n_atoms), S.shape[1])
    if atoms < 1:
        raise ValueError("n_atoms must be positive")
    rng = np.random.default_rng(seed)
    if initial_structure is None or initial_chemistry is None:
        picked = rng.choice(S.shape[1], size=atoms, replace=False)
        Ds, Dc = _unit_columns(S[:, picked]), _unit_columns(C[:, picked])
    else:
        Ds, Dc = _unit_columns(initial_structure), _unit_columns(initial_chemistry)
        if Ds.shape != (S.shape[0], atoms) or Dc.shape != (C.shape[0], atoms):
            raise ValueError("initial paired dictionaries have unexpected shape")

    history: list[float] = []
    for _ in range(max(1, int(n_iter))):
        Zs, Zc = coupled_sparse_encode(S, C, Ds, Dc, sparsity)
        for atom in range(atoms):
            active = np.flatnonzero((np.abs(Zs[atom]) > 1e-10) | (np.abs(Zc[atom]) > 1e-10))
            if not len(active):
                source = int(rng.integers(S.shape[1]))
                Ds[:, atom] = S[:, source] / max(np.linalg.norm(S[:, source]), 1e-12)
                Dc[:, atom] = C[:, source] / max(np.linalg.norm(C[:, source]), 1e-12)
                continue
            residual_s = S[:, active] - Ds @ Zs[:, active] + np.outer(Ds[:, atom], Zs[atom, active])
            residual_c = C[:, active] - Dc @ Zc[:, active] + np.outer(Dc[:, atom], Zc[atom, active])
            us, ss, vhs = np.linalg.svd(residual_s, full_matrices=False)
            uc, sc, vhc = np.linalg.svd(residual_c, full_matrices=False)
            Ds[:, atom] = us[:, 0]
            Dc[:, atom] = uc[:, 0]
            Zs[atom, active] = ss[0] * vhs[0]
            Zc[atom, active] = sc[0] * vhc[0]
        Zs, Zc = coupled_sparse_encode(S, C, Ds, Dc, sparsity)
        history.append(_joint_residual_error(S, C, Ds, Dc, Zs, Zc))
    return Ds, Dc, {
        "reconstruction_relative": float(history[-1]),
        "reconstruction_history": history,
        "n_atoms": atoms,
        "sparsity": min(max(1, int(sparsity)), atoms),
        "n_iter": max(1, int(n_iter)),
    }


def support_agreement(
    codes_structure: np.ndarray,
    codes_chemistry: np.ndarray,
) -> float:
    """Mean Jaccard agreement of the two independently inferred supports."""
    left = np.abs(np.asarray(codes_structure)) > 1e-10
    right = np.abs(np.asarray(codes_chemistry)) > 1e-10
    if left.shape != right.shape:
        raise ValueError("support matrices must have equal shape")
    union = np.sum(left | right, axis=0)
    intersection = np.sum(left & right, axis=0)
    return float(np.mean(intersection / np.maximum(union, 1)))


def relative_reconstruction(
    structure: np.ndarray,
    chemistry: np.ndarray,
    dictionary_structure: np.ndarray,
    dictionary_chemistry: np.ndarray,
    codes_structure: np.ndarray,
    codes_chemistry: np.ndarray,
) -> dict[str, float]:
    """Report each view and the balanced joint reconstruction error."""
    S, C = np.asarray(structure), np.asarray(chemistry)
    return {
        "structure": float(np.linalg.norm(S - dictionary_structure @ codes_structure) / max(np.linalg.norm(S), 1e-12)),
        "chemistry": float(np.linalg.norm(C - dictionary_chemistry @ codes_chemistry) / max(np.linalg.norm(C), 1e-12)),
        "joint": _joint_residual_error(S, C, dictionary_structure, dictionary_chemistry, codes_structure, codes_chemistry),
    }
