"""Physical-bond line graph, lazy one-step walk and the row-sum-matched null.

Pure numpy/float64 reference implementation.  The torch path in ``rcc_models``
must agree with this to ~1e-6 relative; that agreement is a CPU gate.

Definitions (frozen, identical to the repo's bond random-walk rounds):
    A_L  : 0/1 adjacency of the physical-bond line graph (share an atom)
    P    : (I + D^-1 A_L) / 2, zero-degree rows set to the one-hot row e_i
    K_R  : sym(P) = (P + P^T)/2
    d    : K_R @ 1
    K_D  : d d^T / sum(d)              (rank one, exact row-sum match)
    B    : K_R - K_D                   (B @ 1 = 0 exactly in exact arithmetic)
"""

from __future__ import annotations

import numpy as np

SELF_LOOP = 0.5


def line_graph_adjacency(bond_index: np.ndarray) -> np.ndarray:
    bi = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    n = bi.shape[0]
    A = np.zeros((n, n), dtype=np.float64)
    if n == 0:
        return A
    incident: dict[int, list[int]] = {}
    for b, (u, v) in enumerate(bi):
        incident.setdefault(int(u), []).append(b)
        incident.setdefault(int(v), []).append(b)
    for bonds in incident.values():
        for i in range(len(bonds)):
            for j in range(i + 1, len(bonds)):
                a, c = bonds[i], bonds[j]
                A[a, c] = 1.0
                A[c, a] = 1.0
    return A


def lazy_transition(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    deg = A.sum(axis=1)
    P = np.zeros((n, n), dtype=np.float64)
    eye = np.eye(n, dtype=np.float64)
    for i in range(n):
        if deg[i] > 1e-12:
            P[i] = SELF_LOOP * eye[i] + (1.0 - SELF_LOOP) * A[i] / deg[i]
        else:
            P[i, i] = 1.0
    return P


def symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(P, dtype=np.float64) + np.asarray(P, dtype=np.float64).T)


def null_from_row_sums(K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(K_D, d, total mass).  Zero mass returns zeros."""
    K = np.asarray(K, dtype=np.float64)
    d = K.sum(axis=1)
    tot = float(d.sum())
    if tot <= 0.0:
        return np.zeros_like(K), d, 0.0
    return np.outer(d, d) / tot, d, tot


def build_kernel(bond_index: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """float64 (K_R, d) for one molecule."""
    import numpy as _np

    bi = _np.asarray(bond_index, dtype=_np.int64).reshape(-1, 2)
    if bi.size == 0:
        return _np.zeros((0, 0)), _np.zeros(0)
    if bi.min() < 0 or bi.max() >= int(n_atoms):
        raise ValueError("bond index outside the atom range")
    A = line_graph_adjacency(bi)
    K = symmetrize(lazy_transition(A))
    d = K.sum(axis=1)
    return K, d


def structure_stats(K: np.ndarray, n_bonds: int, n_atoms: int, n_junc: int) -> np.ndarray:
    d = K.sum(axis=1) if K.size else np.zeros(0)
    fro = float(np.sqrt((K * K).sum())) if K.size else 0.0
    d2 = float(np.sqrt((d * d).sum())) if d.size else 0.0
    return np.array(
        [
            np.log1p(n_bonds),
            np.log1p(n_atoms),
            np.log1p(n_junc),
            np.log1p(d2),
            np.log1p(fro),
        ],
        dtype=np.float32,
    )


def qform(Z: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Z^T M Z for Z [n, r] and M [n, n] (float64 reference)."""
    return np.asarray(Z, dtype=np.float64).T @ np.asarray(M, dtype=np.float64) @ np.asarray(
        Z, dtype=np.float64
    )


def upper_tri(M: np.ndarray) -> np.ndarray:
    r = M.shape[0]
    idx = np.triu_indices(r)
    return np.asarray(M, dtype=np.float64)[idx]


def double_sum(Z: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Explicit double sum, used only as an independent float64 reference."""
    Z = np.asarray(Z, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    n, r = Z.shape
    out = np.zeros((r, r), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            out += M[i, j] * np.outer(Z[i], Z[j])
    return out
