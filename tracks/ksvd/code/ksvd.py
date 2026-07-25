from __future__ import annotations

import numpy as np
from numpy.linalg import norm, svd


def _omp(D: np.ndarray, y: np.ndarray, T: int) -> np.ndarray:
    """Orthogonal matching pursuit; D is (n, k), y is (n,). Return x (k,)."""
    n, k = D.shape
    x = np.zeros(k, dtype=np.float64)
    residual = y.copy()
    support: list[int] = []
    for _ in range(min(T, k)):
        corr = D.T @ residual
        # mask already selected
        for j in support:
            corr[j] = 0.0
        j = int(np.argmax(np.abs(corr)))
        if abs(corr[j]) < 1e-12:
            break
        support.append(j)
        Ds = D[:, support]
        # least squares
        coef, _, _, _ = np.linalg.lstsq(Ds, y, rcond=None)
        residual = y - Ds @ coef
        x = np.zeros(k, dtype=np.float64)
        for c, idx in zip(coef, support):
            x[idx] = c
        if norm(residual) < 1e-8:
            break
    return x


def ksvd(
    Y: np.ndarray,
    n_atoms: int = 12,
    T: int = 3,
    n_iter: int = 10,
    seed: int = 0,
    T_min: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Sparse dictionary learning (K-SVD style).
    Y: (n_features, n_samples)
    Returns D (n_features, n_atoms), X (n_atoms, n_samples), info dict.
    """
    rng = np.random.default_rng(seed)
    n, N = Y.shape
    n_atoms = min(n_atoms, max(n, 2), max(N, 2))
    T = max(T, T_min)
    T = min(T, n_atoms)

    # init: random columns from Y + noise, normalize
    idx = rng.choice(N, size=n_atoms, replace=N < n_atoms)
    D = Y[:, idx].astype(np.float64).copy()
    D += 1e-3 * rng.standard_normal(D.shape)
    for j in range(n_atoms):
        nj = norm(D[:, j])
        if nj < 1e-12:
            D[:, j] = rng.standard_normal(n)
            nj = norm(D[:, j])
        D[:, j] /= nj

    X = np.zeros((n_atoms, N), dtype=np.float64)
    errs: list[float] = []

    for _it in range(n_iter):
        # sparse coding
        for i in range(N):
            X[:, i] = _omp(D, Y[:, i], T)
            # enforce min nonzeros soft: if too sparse, take top-T_min correlations
            if np.count_nonzero(np.abs(X[:, i]) > 1e-10) < T_min:
                corr = np.abs(D.T @ Y[:, i])
                top = np.argsort(-corr)[:T_min]
                Ds = D[:, top]
                coef, _, _, _ = np.linalg.lstsq(Ds, Y[:, i], rcond=None)
                X[:, i] = 0.0
                for c, j in zip(coef, top):
                    X[j, i] = c

        # dictionary update (sequential SVD)
        for j in range(n_atoms):
            omega = np.where(np.abs(X[j, :]) > 1e-10)[0]
            if omega.size == 0:
                # reinit dead atom
                i = int(rng.integers(0, N))
                D[:, j] = Y[:, i] + 1e-3 * rng.standard_normal(n)
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

        # normalize atoms
        for j in range(n_atoms):
            nj = norm(D[:, j])
            if nj > 1e-12:
                D[:, j] /= nj
                X[j, :] *= nj

        R = Y - D @ X
        errs.append(float(norm(R, "fro") / max(norm(Y, "fro"), 1e-12)))

    atoms_used = int(np.sum(np.any(np.abs(X) > 1e-10, axis=1)))
    nnz_per = np.array([np.count_nonzero(np.abs(X[:, i]) > 1e-10) for i in range(N)])
    info = {
        "recon_rel": errs[-1] if errs else 1.0,
        "recon_curve": errs,
        "atoms_used": atoms_used,
        "mean_nnz": float(nnz_per.mean()) if N else 0.0,
        "n_atoms": n_atoms,
        "T": T,
        "n_iter": n_iter,
    }
    return D, X, info


def readout_X(X: np.ndarray, mode: str = "basic") -> np.ndarray:
    """
    Graph-level vector from coefficient matrix (n_atoms, n_samples).
    mode:
      basic — mean|x|, max|x|, usage
      rich  — + std, sum, energy, quantiles per atom, top-atom soft histogram
    """
    absX = np.abs(X)
    mean_abs = absX.mean(axis=1)
    max_abs = absX.max(axis=1)
    usage = (absX > 1e-10).mean(axis=1)
    if mode == "basic":
        return np.concatenate([mean_abs, max_abs, usage])
    std_abs = absX.std(axis=1)
    sum_abs = absX.sum(axis=1)
    energy = (X**2).mean(axis=1)
    q25 = np.quantile(absX, 0.25, axis=1)
    q75 = np.quantile(absX, 0.75, axis=1)
    # soft assignment histogram: which atom dominates each column
    n_atoms, N = X.shape
    hist = np.zeros(n_atoms, dtype=np.float64)
    if N > 0:
        winners = np.argmax(absX, axis=0)
        for j in winners:
            hist[j] += 1.0
        hist /= N
    return np.concatenate([mean_abs, max_abs, usage, std_abs, sum_abs, energy, q25, q75, hist])
