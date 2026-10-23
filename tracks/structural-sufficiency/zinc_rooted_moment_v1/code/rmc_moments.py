"""Exact rooted-moment mathematics for RMC.

The readout
-----------
For one root ``i``, one scale ``k`` and one view ``v``:

    W = B^k,  m_i = sum_j W_ij,  p_i(j) = W_ij / m_i,  B = I + A          (A: undirected simple adjacency)
    mu   = E_p[u]
    C2   = E_p[(u - mu)(u - mu)^T]
    C3   = E_p[(u - mu) (x) (u - mu) (x) (u - mu)]

    kappa(q, u) = (1 + q.u)^2 + eps,   a = 1 + q.mu
    Z(q)        = a^2 + q^T C2 q + eps = E_p[kappa]
    F(q)        = mu + [2 a C2 q + C3[q, q]] / Z
                = sum_j p(j) kappa(q, u_j) u_j / sum_j p(j) kappa(q, u_j)

Derivation (used by ``tests/test_rmc_math.py`` against direct summation)::

    E[kappa] = 1 + eps + 2 q.mu + q^T E[u u^T] q
             = (1 + q.mu)^2 + q^T C2 q + eps
    E[kappa u]_a = (1+eps) mu_a + 2 (E[u u^T] q)_a + E[u_a (q.u)^2]
    E[u_a (q.u)^2] = M3[q,q]_a
                   = C3[q,q]_a + mu_a q^T C2 q + 2 (C2 q)_a (q.mu) + mu_a (q.mu)^2
    E[kappa u]_a = mu_a Z + 2 a (C2 q)_a + C3[q,q]_a

so ``F = E[kappa u] / E[kappa]`` is the same number, computed only from cached
``mu / C2 / C3 / log m``.  With ``C3 = 0`` the readout is exactly the value the
same positive kernel takes under the same-mean, same-covariance Gaussian
closure (whose support is not the empirical sample, so its conditional mean may
leave the empirical convex hull -- that is the definition of the control, not a
bug).

Packed layouts
--------------
``C2`` is stored packed over ``a <= b`` (78), ``C3`` over ``a <= b <= c`` (364,
mixed terms included).  ``q^T C2 q`` and ``C3[q,q]_a`` both use the same 78-slot
Gram vector with off-diagonal multiplicity 2, and ``C3[q,q]_a`` reads the packed
tensor through a precomputed 936-entry (a, b<=c) -> packed-triple map.  The
packed and full layouts are checked to be numerically identical.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import rmc_common as C  # noqa: E402

DIM = C.PROJ_DIM
N_PAIR = C.N_PAIR
N_TRIPLE = C.N_TRIPLE
N_C3_FULL = C.C3_FULL_TERMS
EPS = C.EPS


# ---------------------------------------------------------------------------
# index maps (frozen; no data dependence)
# ---------------------------------------------------------------------------


def pair_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(i, j, mult)`` for the 78 packed ``i <= j`` pairs (row-major)."""
    i, j, m = [], [], []
    for a in range(DIM):
        for b in range(a, DIM):
            i.append(a)
            j.append(b)
            m.append(1.0 if a == b else 2.0)
    return (
        np.asarray(i, dtype=np.int64),
        np.asarray(j, dtype=np.int64),
        np.asarray(m, dtype=np.float64),
    )


def triple_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j, k = [], [], []
    for a in range(DIM):
        for b in range(a, DIM):
            for c in range(b, DIM):
                i.append(a)
                j.append(b)
                k.append(c)
    return tuple(np.asarray(v, dtype=np.int64) for v in (i, j, k))  # type: ignore[return-value]


def triple_pos() -> dict[tuple[int, int, int], int]:
    out: dict[tuple[int, int, int], int] = {}
    idx = 0
    for a in range(DIM):
        for b in range(a, DIM):
            for c in range(b, DIM):
                out[(a, b, c)] = idx
                idx += 1
    return out


def c2_full_index() -> np.ndarray:
    """144-entry map ``(a, b) -> packed a<=b`` for the full symmetric C2."""
    pi, pj, _ = pair_arrays()
    pack = {(int(a), int(b)): p for p, (a, b) in enumerate(zip(pi, pj))}
    return np.asarray([pack[(a, b)] if a <= b else pack[(b, a)] for a in range(DIM) for b in range(DIM)],
                      dtype=np.int64)


def c3_full_index() -> np.ndarray:
    """936-entry map ``(a, b<=c) -> packed sorted triple``.

    Entry ``a * 78 + p`` is the packed value of the ordered triple
    ``(a, b, c)`` with ``(b, c)`` the ``p``-th packed pair.  Because C3 is
    symmetric in all three indices this is a pure gather of the 364 entries.
    """
    pi, pj, _ = pair_arrays()
    pos = triple_pos()
    out = []
    for a in range(DIM):
        for b, c in zip(pi, pj):
            out.append(pos[tuple(sorted((a, int(b), int(c))))])
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# graph powers / moments (numpy, float64; the construction path)
# ---------------------------------------------------------------------------


def walk_powers(n_atoms: int, bonds: np.ndarray, scales: Sequence[int] = C.SCALES) -> list[np.ndarray]:
    """``W_k = (I + A)^k`` for each requested scale, from the undirected bonds."""
    n = int(n_atoms)
    A = np.zeros((n, n), dtype=np.float64)
    for u, v in np.asarray(bonds, dtype=np.int64):
        if u == v:  # a self loop in the raw data would double count; P3 has none
            raise ValueError("self loop in bond_index")
        A[u, v] = 1.0
        A[v, u] = 1.0
    B = A + np.eye(n, dtype=np.float64)
    out: list[np.ndarray] = []
    W = np.eye(n, dtype=np.float64)
    need = sorted(set(int(s) for s in scales))
    if need[0] < 0:
        raise ValueError("negative scale")
    present = 0
    for k in need:
        while present < k:
            W = W @ B
            present += 1
        out.append(W.copy())
    order = {k: i for i, k in enumerate(need)}
    return [out[order[int(s)]] for s in scales]


def outer3_packed(u: np.ndarray) -> np.ndarray:
    """``u_a u_b u_c`` for the 364 packed ``a <= b <= c`` triples."""
    ti, tj, tk = triple_arrays()
    return u[:, ti] * u[:, tj] * u[:, tk]


def moments_from_walk(u: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rooted moments of the row-normalised walk measure.

    ``u``: ``(n, 12)`` endpoint features, ``W``: ``(n, n)`` non-negative walk
    matrix.  Returns ``mu (n,12)``, ``c2 (n,78)``, ``c3 (n,364)``, ``logm (n,)``.
    """
    u = np.asarray(u, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = u.shape[0]
    if W.shape != (n, n):
        raise ValueError(f"walk matrix {W.shape} does not match {n} endpoints")
    mass = W.sum(axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("non-positive walk mass (I + A)^k should guarantee positivity")
    P = W / mass[:, None]

    mu = P @ u
    U2 = (u[:, :, None] * u[:, None, :]).reshape(n, DIM * DIM)
    M2 = (P @ U2).reshape(n, DIM, DIM)
    c2_full = M2 - mu[:, :, None] * mu[:, None, :]

    U3 = outer3_packed(u)
    M3 = P @ U3

    pi, pj, _ = pair_arrays()
    ti, tj, tk = triple_arrays()
    c2_ab = c2_full[:, ti, tj]
    c2_ac = c2_full[:, ti, tk]
    c2_bc = c2_full[:, tj, tk]
    mu_a, mu_b, mu_c = mu[:, ti], mu[:, tj], mu[:, tk]
    c3_packed = (
        M3
        - mu_c * c2_ab
        - mu_b * c2_ac
        - mu_a * c2_bc
        - mu_a * mu_b * mu_c
    )
    pack2 = c2_full[:, pi, pj]
    return mu, pack2, c3_packed, np.log(mass)


def moments_direct(u: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force central moments from an explicit probability vector."""
    u = np.asarray(u, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if not np.isclose(p.sum(), 1.0, atol=1e-12):
        raise ValueError("probabilities must sum to 1")
    mu = (p[:, None] * u).sum(0)
    d = u - mu
    c2_full = (p[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    c3_full = (p[:, None, None, None] * d[:, :, None, None] * d[:, None, :, None] * d[:, None, None, :]).sum(0)
    pi, pj, _ = pair_arrays()
    ti, tj, tk = triple_arrays()
    return mu, c2_full[pi, pj], c3_full[ti, tj, tk]


# ---------------------------------------------------------------------------
# readout (torch; one implementation for model + reference checks)
# ---------------------------------------------------------------------------


def _torch_maps(device, dtype):
    import torch

    key = (str(device), str(dtype))
    cache = getattr(_torch_maps, "_cache", None)
    if cache is None:
        cache = {}
        _torch_maps._cache = cache  # type: ignore[attr-defined]
    if key not in cache:
        pi, pj, mult = pair_arrays()
        cache[key] = {
            "pi": torch.as_tensor(pi, device=device),
            "pj": torch.as_tensor(pj, device=device),
            "mult": torch.as_tensor(mult, device=device, dtype=dtype),
            "c2": torch.as_tensor(c2_full_index(), device=device),
            "c3": torch.as_tensor(c3_full_index(), device=device),
        }
    return cache[key]


def gram_pairs(q):
    """``q_a q_b * mult`` for the 78 packed pairs -> ``(..., 78)``."""
    maps = _torch_maps(q.device, q.dtype)
    return q[..., maps["pi"]] * q[..., maps["pj"]] * maps["mult"]


def readout_batched(mu, c2, c3, q, *, eps: float = EPS, use_c3: bool = True):
    """Exact positive-kernel readout from cached moments.

    ``mu``: ``(..., S, 12)``; ``c2``: ``(..., S, 78)``; ``c3``: ``(..., S, 364)``;
    ``q``: ``(..., H, 12)``; leading dimensions must broadcast.

    Returns ``F (..., S, H, 12)`` and ``Z (..., S, H)``.
    """
    import torch

    maps = _torch_maps(q.device, q.dtype)
    qq = gram_pairs(q)  # (..., H, 78)
    a = 1.0 + torch.einsum("...sc,...hc->...sh", mu, q)
    qc2q = torch.einsum("...sp,...hp->...sh", c2, qq)
    c2_full = c2[..., maps["c2"]].reshape(*c2.shape[:-1], DIM, DIM)
    c2q = torch.einsum("...sab,...hb->...sha", c2_full, q)
    if use_c3:
        c3_full = c3[..., maps["c3"]].reshape(*c3.shape[:-1], DIM, N_PAIR)
        c3qq = torch.einsum("...sap,...hp->...sha", c3_full, qq)
    else:
        c3qq = torch.zeros_like(c2q)
    Z = a * a + qc2q + eps
    F = mu.unsqueeze(-2) + (2.0 * a.unsqueeze(-1) * c2q + c3qq) / Z.unsqueeze(-1)
    return F, Z


def readout_direct(u, logp, q, *, eps: float = EPS):
    """Direct weighted sum ``sum_j p_j kappa(q,u_j) u_j / sum_j p_j kappa`` (torch)."""
    import torch

    p = torch.exp(logp)
    if p.dim() == 1:
        p = p.unsqueeze(0).expand(u.shape[0], -1)
    qu = torch.einsum("nc,jc->nj", q, u)
    kap = (1.0 + qu) ** 2 + eps
    w = p * kap
    den = w.sum(-1)
    num = torch.einsum("nj,jc->nc", w, u)
    return num / den.unsqueeze(-1), den


# ---------------------------------------------------------------------------
# small analytic witness (numpy; documentation + unit test)
# ---------------------------------------------------------------------------


def witness() -> dict[str, Any]:
    """One-dimensional two-point witness that the C3 term changes the readout."""
    dists = {
        "A": (np.asarray([-0.5, 0.5]), np.asarray([0.5, 0.5])),
        "B": (np.asarray([-math.sqrt(1.0 / 8.0), math.sqrt(0.5)]), np.asarray([2.0 / 3.0, 1.0 / 3.0])),
    }
    out: dict[str, Any] = {"eps": EPS, "q": 1.0, "distributions": {}}
    for name, (u1, p1) in dists.items():
        u = u1.reshape(-1)
        mu = float((p1 * u).sum())
        c2 = float((p1 * (u - mu) ** 2).sum())
        m3 = float((p1 * (u - mu) ** 3).sum())
        a = 1.0 + mu
        Z = a * a + c2 + EPS
        F_m = float(mu + 2.0 * a * c2 / Z)
        F_g = F_m  # C3 = 0
        F_m_c3 = float(mu + (2.0 * a * c2 + m3) / Z)
        kappa = (1.0 + u) ** 2 + EPS
        F_direct = float((p1 * kappa * u).sum() / (p1 * kappa).sum())
        out["distributions"][name] = {
            "u": u.tolist(),
            "p": p1.tolist(),
            "mean": mu,
            "variance": c2,
            "third_central_moment": m3,
            "F_moment_with_C3": F_m_c3,
            "F_gaussian_closure": F_g,
            "F_direct_sum": F_direct,
        }
    return out
