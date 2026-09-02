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
    initial_dictionary: np.ndarray | None = None,
    coherence_step: float = 0.0,
    anchor_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Sparse dictionary learning (K-SVD style).
    Y: (n_features, n_samples)
    Returns D (n_features, n_atoms), X (n_atoms, n_samples), info dict.
    """
    rng = np.random.default_rng(seed)
    n, N = Y.shape
    if initial_dictionary is None:
        n_atoms = min(n_atoms, max(n, 2), max(N, 2))
    else:
        n_atoms = int(n_atoms)
        if n_atoms < 1:
            raise ValueError("n_atoms must be positive")
    T = max(T, T_min)
    T = min(T, n_atoms)
    if not np.isfinite(coherence_step) or coherence_step < 0.0:
        raise ValueError("coherence_step must be finite and non-negative")
    if not np.isfinite(anchor_strength) or not 0.0 <= anchor_strength < 1.0:
        raise ValueError("anchor_strength must be finite and in [0, 1)")

    # Default initialization matches the historical implementation exactly.
    # A supplied dictionary enables deterministic warm starts without changing
    # the subsequent sparse coding or atom-wise K-SVD updates.
    if initial_dictionary is None:
        idx = rng.choice(N, size=n_atoms, replace=N < n_atoms)
        D = Y[:, idx].astype(np.float64).copy()
        D += 1e-3 * rng.standard_normal(D.shape)
        initialization = "random_training_columns"
    else:
        supplied = np.asarray(initial_dictionary, dtype=np.float64)
        if supplied.shape != (n, n_atoms):
            raise ValueError(
                "initial_dictionary must have shape "
                f"{(n, n_atoms)}, got {supplied.shape}"
            )
        if not np.all(np.isfinite(supplied)):
            raise ValueError("initial_dictionary contains non-finite values")
        D = supplied.copy()
        initialization = "provided"
    for j in range(n_atoms):
        nj = norm(D[:, j])
        if nj < 1e-12:
            D[:, j] = rng.standard_normal(n)
            nj = norm(D[:, j])
        D[:, j] /= nj
    anchor_dictionary = D.copy() if anchor_strength > 0.0 else None

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

        # Optional empirical-anchor proximal regularization.  Atom indices are
        # inherited from the initialization throughout sequential K-SVD; sign
        # alignment removes the rank-one SVD sign ambiguity before shrinking
        # each learned atom toward its original real training patch.
        if anchor_strength > 0.0:
            assert anchor_dictionary is not None
            for j in range(n_atoms):
                anchor = anchor_dictionary[:, j]
                if float(D[:, j] @ anchor) < 0.0:
                    anchor = -anchor
                D[:, j] = (
                    (1.0 - anchor_strength) * D[:, j]
                    + anchor_strength * anchor
                )
                nj = norm(D[:, j])
                if nj < 1e-12:
                    raise RuntimeError(
                        "anchor update produced a degenerate atom"
                    )
                D[:, j] /= nj

        # Optional projected gradient step on the off-diagonal Gram penalty.
        # The zero-step path deliberately skips this entire block so historical
        # dictionaries and sparse codes remain byte-identical.
        if coherence_step > 0.0 and n_atoms > 1:
            gram_offdiag = D.T @ D
            np.fill_diagonal(gram_offdiag, 0.0)
            D = D - coherence_step * (D @ gram_offdiag)
            for j in range(n_atoms):
                nj = norm(D[:, j])
                if nj < 1e-12:
                    raise RuntimeError(
                        "incoherence update produced a degenerate atom"
                    )
                D[:, j] /= nj

        if anchor_strength > 0.0 or coherence_step > 0.0:
            for i in range(N):
                X[:, i] = _omp(D, Y[:, i], T)
                if np.count_nonzero(np.abs(X[:, i]) > 1e-10) < T_min:
                    corr = np.abs(D.T @ Y[:, i])
                    top = np.argsort(-corr)[:T_min]
                    Ds = D[:, top]
                    coef, _, _, _ = np.linalg.lstsq(Ds, Y[:, i], rcond=None)
                    X[:, i] = 0.0
                    for c, atom in zip(coef, top):
                        X[atom, i] = c

        R = Y - D @ X
        errs.append(float(norm(R, "fro") / max(norm(Y, "fro"), 1e-12)))

    # For a zero-update control, still report the OMP reconstruction of the
    # supplied initialization.  The dictionary itself is left unchanged.
    if n_iter == 0:
        for i in range(N):
            X[:, i] = _omp(D, Y[:, i], T)
        R = Y - D @ X
        errs.append(float(norm(R, "fro") / max(norm(Y, "fro"), 1e-12)))

    atoms_used = int(np.sum(np.any(np.abs(X) > 1e-10, axis=1)))
    nnz_per = np.array([np.count_nonzero(np.abs(X[:, i]) > 1e-10) for i in range(N)])
    gram_absolute = np.abs(D.T @ D)
    offdiagonal_mask = ~np.eye(n_atoms, dtype=bool)
    offdiagonal = gram_absolute[offdiagonal_mask]
    info = {
        "recon_rel": errs[-1] if errs else 1.0,
        "recon_curve": errs,
        "atoms_used": atoms_used,
        "mean_nnz": float(nnz_per.mean()) if N else 0.0,
        "n_atoms": n_atoms,
        "T": T,
        "n_iter": n_iter,
        "initialization": initialization,
        "coherence_step": float(coherence_step),
        "anchor_strength": float(anchor_strength),
        "mean_absolute_anchor_cosine": (
            float(np.mean(np.abs(np.sum(D * anchor_dictionary, axis=0))))
            if anchor_dictionary is not None else None
        ),
        "minimum_absolute_anchor_cosine": (
            float(np.min(np.abs(np.sum(D * anchor_dictionary, axis=0))))
            if anchor_dictionary is not None else None
        ),
        "mean_absolute_offdiagonal_coherence": (
            float(offdiagonal.mean()) if offdiagonal.size else 0.0
        ),
        "maximum_absolute_offdiagonal_coherence": (
            float(offdiagonal.max()) if offdiagonal.size else 0.0
        ),
    }
    return D, X, info


def mil_attention_weights(
    X: np.ndarray,
    seed: int = 0,
    n_iter: int = 40,
    lr: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Instance-level MIL attention over patch columns of X (n_atoms, N).

    Learns w,b so a_j = softmax(w^T tanh(x_j) + b); returns (weights a, attended vec).
    Unsupervised: maximize attended energy under entropy regularizer (no labels).
    """
    n_atoms, N = X.shape
    if N == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(n_atoms, dtype=np.float64)
    if N == 1:
        return np.ones(1, dtype=np.float64), X[:, 0].copy()

    rng = np.random.default_rng(seed)
    # features for score: tanh of coefficients
    H = np.tanh(X)  # (k, N)
    w = rng.standard_normal(n_atoms) * 0.1
    b = 0.0
    for _ in range(n_iter):
        scores = w @ H + b  # (N,)
        scores = scores - scores.max()
        exp_s = np.exp(scores)
        a = exp_s / (exp_s.sum() + 1e-12)
        # objective: attended ||x||^2 + small entropy (prefer peaked)
        # dL/dscores via soft attention
        att = X @ a  # (k,)
        energy = float(att @ att)
        # encourage energy of attended vector
        # grad wrt a ≈ 2 (X^T att); through softmax
        g_a = 2.0 * (X.T @ att)  # (N,)
        # entropy bonus on a: +eps * (-sum a log a) → prefer less flat
        eps = 0.05
        g_a = g_a - eps * (np.log(a + 1e-12) + 1.0)
        # softmax Jacobian: da_i/ds_j
        # g_s = a ⊙ (g_a - 1^T (a ⊙ g_a))
        g_s = a * (g_a - float(a @ g_a))
        g_w = H @ g_s
        g_b = float(g_s.sum())
        w = w + lr * g_w
        b = b + lr * g_b
        # light weight decay
        w *= 0.999

    scores = w @ H + b
    scores = scores - scores.max()
    exp_s = np.exp(scores)
    a = exp_s / (exp_s.sum() + 1e-12)
    att = X @ a
    return a.astype(np.float64), att.astype(np.float64)


def pool_X(
    X: np.ndarray,
    pool: str = "mean",
    seed: int = 0,
) -> np.ndarray:
    """
    Pool patch coefficients X (n_atoms, N) → graph vector base.

    pool:
      mean | max | sum — elementwise over patches
      attn | mil — MIL attention (unsupervised), returns attended x + a-stats
    """
    absX = np.abs(X)
    n_atoms, N = X.shape
    if N == 0:
        return np.zeros(n_atoms * 3, dtype=np.float64)

    pool = pool.lower()
    if pool in ("attn", "mil"):
        a, att = mil_attention_weights(X, seed=seed)
        # attended coef + usage of top instances + entropy of a
        top_k = min(3, N)
        top_idx = np.argsort(-a)[:top_k]
        top_x = absX[:, top_idx].mean(axis=1) if top_k else np.zeros(n_atoms)
        ent = float(-(a * np.log(a + 1e-12)).sum()) if N else 0.0
        peak = float(a.max()) if N else 0.0
        return np.concatenate([att, top_x, np.array([ent, peak, float(N)], dtype=np.float64)])
    if pool == "max":
        return absX.max(axis=1)
    if pool == "sum":
        return absX.sum(axis=1)
    # mean default
    return absX.mean(axis=1)


def readout_X(X: np.ndarray, mode: str = "basic", pool: str = "mean", seed: int = 0) -> np.ndarray:
    """
    Graph-level vector from coefficient matrix (n_atoms, n_samples).
    mode:
      basic — mean|x|, max|x|, usage
      rich  — + std, sum, energy, quantiles per atom, top-atom soft histogram
      pool  — use pool_X only (for ablating mean/max/attn)
    pool: mean|max|sum|attn (used when mode=='pool' or appended in rich+attn)
    """
    absX = np.abs(X)
    n_atoms, N = X.shape
    if mode == "pool":
        return pool_X(X, pool=pool, seed=seed)

    mean_abs = absX.mean(axis=1) if N else np.zeros(n_atoms)
    max_abs = absX.max(axis=1) if N else np.zeros(n_atoms)
    usage = (absX > 1e-10).mean(axis=1) if N else np.zeros(n_atoms)
    if mode == "basic":
        return np.concatenate([mean_abs, max_abs, usage])

    std_abs = absX.std(axis=1) if N else np.zeros(n_atoms)
    sum_abs = absX.sum(axis=1) if N else np.zeros(n_atoms)
    energy = (X**2).mean(axis=1) if N else np.zeros(n_atoms)
    q25 = np.quantile(absX, 0.25, axis=1) if N else np.zeros(n_atoms)
    q75 = np.quantile(absX, 0.75, axis=1) if N else np.zeros(n_atoms)
    hist = np.zeros(n_atoms, dtype=np.float64)
    if N > 0:
        winners = np.argmax(absX, axis=0)
        for j in winners:
            hist[j] += 1.0
        hist /= N
    base = np.concatenate([mean_abs, max_abs, usage, std_abs, sum_abs, energy, q25, q75, hist])
    if pool in ("attn", "mil"):
        return np.concatenate([base, pool_X(X, pool="attn", seed=seed)])
    return base
