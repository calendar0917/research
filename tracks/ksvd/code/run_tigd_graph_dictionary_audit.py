#!/usr/bin/env python
"""TIGD-v0 — Typed Incidence Graph Dictionary unmixing feasibility audit.

Question (graph-space dictionary, no task-driven / sparse / GNN / RFF)
---------------------------------------------------------------------
Previous rounds removed variable-size / permutation / graph structure *before*
the dictionary.  TIGD instead makes graph matching part of dictionary inference:

    (X_V, X_E, B)  ->  (P_V, P_E, alpha_G)

Each dictionary atom is one *whole attributed incidence graph*

    D_k = (A_k^V, A_k^E, R_k),
    A_k^V[a,:] = softmax(Theta_k^V[a,:]),  A_k^E[b,:] = softmax(Theta_k^E[b,:]),
    R_k[:,b]   = 2 * softmax_a(Xi_k[:,b])   (sum_a R_k[a,b] = 2 exactly).

A graph is explained by a mixture code ``alpha_G in Delta_K`` and graph-specific
soft transport ``P_V in Pi(1/n, 1/24)``, ``P_E in Pi(1/m, 1/24)``.  The energy

    L_V = sum_{v,a} P_V[v,a] ||X_V[v] - A_V_bar[a]||^2
    L_E = sum_{e,b} P_E[e,b] ||X_E[e] - A_E_bar[b]||^2
    L_B = sum_{v,e,a,b} P_V[v,a] P_E[e,b] (B_ve - R_bar_ab)^2
    E_G = L_V + L_E + L_B

couples the *same* ``P_V[v,:]`` across all bonds incident to ``v`` (unlike DOI,
which keeps only the incidence-pair distribution as a multiset).

Discipline
----------
* Only official ZINC ``train`` (10000) and ``valid`` (1000) are loaded; the
  official ``test`` split is never read, instantiated or referenced.
* The dictionary is learned label-free (block coordinate, detached inference).
* ``y`` is used only in the frozen-code probe.
* This is a feasibility audit: no sparse penalty, no task-driven dictionary,
  no unrolled backprop, no GNN / RFF / learned lift, no K/q/eps sweep.

Reuse
-----
The ZINC loader, ``Mol`` container, permutation / rewiring perturbations, probe
and codebook helpers come from ``run_aiom_representation_audit``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_aiom_representation_audit import (  # noqa: E402
    Mol,
    build_codebook,
    linear_ridge,
    load_mols,
    permute_mol,
    repo_loader,
    train_probe,
    two_switch,
)

C_V = 21
C_E = 3
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tigd_graph_dictionary"

# Frozen reference MAE for the frozen-code probe (from prior rounds).
REF_DOI_MAE = 0.3865
REF_SCATTER_MAE = 0.3657
REF_ICRATE_MAE = 0.399

PILOT_GRAPHS = 2048
PILOT_EPOCHS = 40
FULL_GRAPHS = 10_000
FULL_EPOCHS = 60
HOLDOUT_GRAPHS = 256


# ===========================================================================
# 0. Config
# ===========================================================================
@dataclass
class TIGDConfig:
    K: int = 16
    qV: int = 24
    qE: int = 24
    eps: float = 0.05
    sinkhorn_iters: int = 50
    T_alt: int = 8
    alpha_steps: int = 5
    alpha_backtrack: int = 12
    lam_V: float = 1.0
    lam_E: float = 1.0
    lam_B: float = 1.0
    alpha_tol: float = 1e-10


# ===========================================================================
# 1. Data -> arrays
# ===========================================================================
def make_codebook(mols: Sequence[Mol]):
    atoms, bonds = build_codebook(mols)
    assert len(atoms) == C_V, f"expected C_V={C_V}, got {len(atoms)}"
    assert len(bonds) == C_E, f"expected C_E={C_E}, got {len(bonds)}"
    return {c: i for i, c in enumerate(atoms)}, {c: i for i, c in enumerate(bonds)}


def mol_arrays(mol: Mol, node_index, bond_index):
    """Return (B (n,m), X_V (n,C_V), X_E (m,C_E)) as float64."""
    n, m = mol.n, mol.m
    B = np.zeros((n, m), dtype=np.float64)
    X_V = np.zeros((n, C_V), dtype=np.float64)
    X_E = np.zeros((m, C_E), dtype=np.float64)
    for v in range(n):
        X_V[v, node_index[int(mol.node_types[v])]] = 1.0
    for e, (a, b) in enumerate(mol.bonds):
        B[a, e] = 1.0
        B[b, e] = 1.0
        X_E[e, bond_index[int(mol.bond_types[e])]] = 1.0
    return B, X_V, X_E


# ===========================================================================
# 2. Numpy CPU float64 reference (single graph)
# ===========================================================================
def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn_np(C: np.ndarray, r: np.ndarray, c: np.ndarray, eps: float,
                iters: int):
    n, m = C.shape
    lr = np.log(r)
    lc = np.log(c)
    f = np.zeros(n, dtype=np.float64)
    g = np.zeros(m, dtype=np.float64)
    for _ in range(iters):
        f = eps * lr - eps * _logsumexp((g[None, :] - C) / eps, axis=1)
        g = eps * lc - eps * _logsumexp((f[:, None] - C) / eps, axis=0)
    P = np.exp((f[:, None] + g[None, :] - C) / eps)
    return P


def proj_simplex_np(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u + (1.0 - css) / (np.arange(1, len(v) + 1)) > 0)[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _attr_cost_np(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    # X (N,C), A (q,C) -> (N,q)
    x2 = (X ** 2).sum(-1, keepdims=True)
    xdot = X @ A.T
    a2 = (A ** 2).sum(-1)[None, :]
    return x2 - 2.0 * xdot + a2


def mixture_np(A_V: np.ndarray, A_E: np.ndarray, R: np.ndarray, alpha: np.ndarray):
    AV = np.einsum("k,kac->ac", alpha, A_V)
    AE = np.einsum("k,kbc->bc", alpha, A_E)
    RB = np.einsum("k,kab->ab", alpha, R)
    return AV, AE, RB


def components_np(alpha, A_V, A_E, R, P_V, P_E, B, X_V, X_E, lam_B: float = 1.0):
    """Return (L_V, L_E, L_B, total, grad_alpha)."""
    AV, AE, RB = mixture_np(A_V, A_E, R, alpha)
    cV = _attr_cost_np(X_V, AV)
    cE = _attr_cost_np(X_E, AE)
    L_V = float((P_V * cV).sum())
    L_E = float((P_E * cE).sum())
    Usq = (1.0 - RB) ** 2
    Wsq = RB ** 2
    BPE = B @ P_E                     # (n, qE)
    Q_U = P_V.T @ BPE                 # (qV, qE)
    Q_W = P_V.T @ ((1.0 - B) @ P_E)
    L_B = float((Q_U * Usq).sum() + (Q_W * Wsq).sum())
    total = L_V + L_E + lam_B * L_B

    # gradient wrt alpha
    PVt_XV = P_V.T @ X_V
    margV = P_V.sum(0)
    dLV = -2.0 * (PVt_XV - margV[:, None] * AV)
    PEt_XE = P_E.T @ X_E
    margE = P_E.sum(0)
    dLE = -2.0 * (PEt_XE - margE[:, None] * AE)
    dLB = -2.0 * (1.0 - RB) * Q_U + 2.0 * RB * Q_W
    grad = (np.einsum("ac,kac->k", dLV, A_V)
            + np.einsum("bc,kbc->k", dLE, A_E)
            + lam_B * np.einsum("ab,kab->k", dLB, R))
    return L_V, L_E, L_B, total, grad


def infer_reference_np(mol: Mol, atoms: dict, node_index, bond_index,
                       cfg: TIGDConfig, lam_B: float = 1.0):
    """Single-graph CPU float64 reference inference (spec section 14-18)."""
    A_V, A_E, R = atoms["A_V"], atoms["A_E"], atoms["R"]
    K, qV, qE = A_V.shape[0], A_V.shape[1], A_E.shape[1]
    B, X_V, X_E = mol_arrays(mol, node_index, bond_index)
    n, m = mol.n, mol.m
    alpha = np.full(K, 1.0 / K)
    P_V = np.full((n, qV), 1.0 / (qV * n))
    P_E = np.full((m, qE), 1.0 / (qE * m))
    r_V = np.full(n, 1.0 / n)
    c_col = np.full(qV, 1.0 / qV)
    r_E = np.full(m, 1.0 / m)
    c_ecol = np.full(qE, 1.0 / qE)
    failures = 0
    marg_err = 0.0
    for _ in range(cfg.T_alt):
        AV, AE, RB = mixture_np(A_V, A_E, R, alpha)
        Usq = (1.0 - RB) ** 2
        Wsq = RB ** 2
        # --- P_V ---
        cV = _attr_cost_np(X_V, AV)
        C_V = cV + lam_B * (B @ (P_E @ Usq.T) + (1.0 - B) @ (P_E @ Wsq.T))
        P_V = sinkhorn_np(C_V, r_V, c_col, cfg.eps, cfg.sinkhorn_iters)
        marg_err = max(marg_err,
                       float(np.abs(P_V.sum(1) - r_V).max()),
                       float(np.abs(P_V.sum(0) - c_col).max()))
        # --- P_E ---
        cE = _attr_cost_np(X_E, AE)
        C_E = cE + lam_B * (B.T @ (P_V @ Usq) + (1.0 - B).T @ (P_V @ Wsq))
        P_E = sinkhorn_np(C_E, r_E, c_ecol, cfg.eps, cfg.sinkhorn_iters)
        # --- alpha (projected gradient, backtracking) ---
        for _step in range(cfg.alpha_steps):
            _, _, _, E0, grad = components_np(alpha, A_V, A_E, R, P_V, P_E, B,
                                              X_V, X_E, lam_B)
            eta = 1.0
            accepted = False
            for _bt in range(cfg.alpha_backtrack):
                u = proj_simplex_np(alpha - eta * grad)
                _, _, _, Enew, _ = components_np(u, A_V, A_E, R, P_V, P_E, B,
                                                 X_V, X_E, lam_B)
                if Enew <= E0 + cfg.alpha_tol:
                    alpha = u
                    accepted = True
                    break
                eta *= 0.5
            if not accepted:
                failures += 1
    L_V, L_E, L_B, total, _ = components_np(alpha, A_V, A_E, R, P_V, P_E, B,
                                            X_V, X_E, lam_B)
    return {
        "alpha": alpha, "P_V": P_V, "P_E": P_E,
        "L_V": L_V, "L_E": L_E, "L_B": L_B, "total": total,
        "marginal_err": marg_err, "alpha_failures": failures,
    }


def energy_reference_np(mol, atoms, node_index, bond_index, cfg, alpha, P_V, P_E,
                        lam_B=1.0):
    A_V, A_E, R = atoms["A_V"], atoms["A_E"], atoms["R"]
    B, X_V, X_E = mol_arrays(mol, node_index, bond_index)
    return components_np(alpha, A_V, A_E, R, P_V, P_E, B, X_V, X_E, lam_B)[3]


def atoms_from_logits_np(theta_V, theta_E, xi):
    V = np.exp(theta_V - theta_V.max(-1, keepdims=True))
    A_V = V / V.sum(-1, keepdims=True)
    E = np.exp(theta_E - theta_E.max(-1, keepdims=True))
    A_E = E / E.sum(-1, keepdims=True)
    Xs = np.exp(xi - xi.max(1, keepdims=True))
    R = 2.0 * Xs / Xs.sum(1, keepdims=True)
    return {"A_V": A_V, "A_E": A_E, "R": R}


# ===========================================================================
# 3. Torch batched solver
# ===========================================================================
def _torch():
    import torch  # noqa

    return torch


def atoms_from_logits(torch, theta_V, theta_E, xi):
    A_V = torch.softmax(theta_V, dim=-1)
    A_E = torch.softmax(theta_E, dim=-1)
    R = 2.0 * torch.softmax(xi, dim=1)
    return A_V, A_E, R


def _mixture_t(A_V, A_E, R, alpha):
    torch = _torch()
    AV = torch.einsum("gk,kac->gac", alpha, A_V)
    AE = torch.einsum("gk,kbc->gbc", alpha, A_E)
    RB = torch.einsum("gk,kab->gab", alpha, R)
    return AV, AE, RB


def _attr_cost_t(X, A):
    # X (G,N,C), A (G,q,C) -> (G,N,q)
    torch = _torch()
    x2 = (X ** 2).sum(-1, keepdim=True)
    xdot = torch.einsum("gnc,gac->gna", X, A)
    a2 = (A ** 2).sum(-1).unsqueeze(1)
    return x2 - 2.0 * xdot + a2


def sinkhorn_batch(C, r, c, eps: float, iters: int):
    """Log-domain Sinkhorn with zero-mass rows handled via -inf."""
    torch = _torch()
    G, N, M = C.shape
    neg_inf = torch.full_like(r, float("-inf"))
    lr = torch.where(r > 0, torch.log(torch.clamp(r, min=1e-300)), neg_inf)
    neg_inf_c = torch.full_like(c, float("-inf"))
    lc = torch.where(c > 0, torch.log(torch.clamp(c, min=1e-300)), neg_inf_c)
    f = torch.zeros(G, N, dtype=C.dtype, device=C.device)
    g = torch.zeros(G, M, dtype=C.dtype, device=C.device)
    for _ in range(iters):
        f = eps * lr - eps * torch.logsumexp((g[:, None, :] - C) / eps, dim=2)
        g = eps * lc - eps * torch.logsumexp((f[:, :, None] - C) / eps, dim=1)
    P = torch.exp((f[:, :, None] + g[:, None, :] - C) / eps)
    row = P.sum(2)
    col = P.sum(1)
    row_err = (row - r).abs().max().item()
    col_err = (col - c).abs().max().item()
    return P, max(row_err, col_err)


def energy_batch(alpha, A_V, A_E, R, P_V, P_E, B, X_V, X_E, lam_B=1.0):
    """Return (total, L_V, L_E, L_B) with per-graph rows.

    ``alpha`` (G,K); atoms (K,...) or (G,...) broadcast handled by einsum.
    """
    torch = _torch()
    if A_V.dim() == 3:
        AV = torch.einsum("gk,kac->gac", alpha, A_V)
        AE = torch.einsum("gk,kbc->gbc", alpha, A_E)
        RB = torch.einsum("gk,kab->gab", alpha, R)
    else:
        AV = torch.einsum("gk,gkac->gac", alpha, A_V)
        AE = torch.einsum("gk,gkbc->gbc", alpha, A_E)
        RB = torch.einsum("gk,gkab->gab", alpha, R)
    cV = _attr_cost_t(X_V, AV)
    cE = _attr_cost_t(X_E, AE)
    L_V = (P_V * cV).sum((1, 2))
    L_E = (P_E * cE).sum((1, 2))
    Usq = (1.0 - RB) ** 2
    Wsq = RB ** 2
    BPE = torch.bmm(B, P_E)
    Q_U = torch.bmm(P_V.transpose(1, 2), BPE)
    Q_W = torch.bmm(P_V.transpose(1, 2), torch.bmm(1.0 - B, P_E))
    L_B = (Q_U * Usq).sum((1, 2)) + (Q_W * Wsq).sum((1, 2))
    total = L_V + L_E + lam_B * L_B
    return total, L_V, L_E, lam_B * L_B


def alpha_grad_batch(alpha, A_V, A_E, R, P_V, P_E, B, X_V, X_E, lam_B=1.0):
    torch = _torch()
    if A_V.dim() == 3:
        AV = torch.einsum("gk,kac->gac", alpha, A_V)
        AE = torch.einsum("gk,kbc->gbc", alpha, A_E)
        RB = torch.einsum("gk,kab->gab", alpha, R)
    else:
        AV = torch.einsum("gk,gkac->gac", alpha, A_V)
        AE = torch.einsum("gk,gkbc->gbc", alpha, A_E)
        RB = torch.einsum("gk,gkab->gab", alpha, R)
    PVt_XV = torch.bmm(P_V.transpose(1, 2), X_V)
    margV = P_V.sum(1)
    dLV = -2.0 * (PVt_XV - margV[:, :, None] * AV)
    PEt_XE = torch.bmm(P_E.transpose(1, 2), X_E)
    margE = P_E.sum(1)
    dLE = -2.0 * (PEt_XE - margE[:, :, None] * AE)
    BPE = torch.bmm(B, P_E)
    Q_U = torch.bmm(P_V.transpose(1, 2), BPE)
    Q_W = torch.bmm(P_V.transpose(1, 2), torch.bmm(1.0 - B, P_E))
    dLB = -2.0 * (1.0 - RB) * Q_U + 2.0 * RB * Q_W
    if A_V.dim() == 3:
        grad = (torch.einsum("gac,kac->gk", dLV, A_V)
                + torch.einsum("gbc,kbc->gk", dLE, A_E)
                + lam_B * torch.einsum("gab,kab->gk", dLB, R))
    else:
        grad = (torch.einsum("gac,gkac->gk", dLV, A_V)
                + torch.einsum("gbc,gkbc->gk", dLE, A_E)
                + lam_B * torch.einsum("gab,gkab->gk", dLB, R))
    return grad


def proj_simplex_batch(v):
    torch = _torch()
    K = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = torch.cumsum(u, dim=-1)
    k = torch.arange(1, K + 1, dtype=v.dtype, device=v.device)
    cond = u - (css - 1.0) / k > 0
    rho = cond.sum(-1) - 1
    css_rho = css.gather(-1, rho.unsqueeze(-1)).squeeze(-1)
    theta = (css_rho - 1.0) / (rho + 1).to(v.dtype)
    return torch.clamp(v - theta.unsqueeze(-1), min=0.0)


def _expand_rows(t, M):
    return t.repeat_interleave(M, dim=0)


def alpha_line_search(alpha, grad, A_V, A_E, R, P_V, P_E, B, X_V, X_E, cfg,
                      lam_B=1.0):
    torch = _torch()
    G, K = alpha.shape
    M = cfg.alpha_backtrack
    total0, _, _, _ = energy_batch(alpha, A_V, A_E, R, P_V, P_E, B, X_V, X_E, lam_B)
    etas = (0.5 ** torch.arange(M, dtype=alpha.dtype, device=alpha.device))
    u = alpha[:, None, :] - etas[None, :, None] * grad[:, None, :]
    u = proj_simplex_batch(u.reshape(G * M, K))
    P_V_e = _expand_rows(P_V, M)
    P_E_e = _expand_rows(P_E, M)
    B_e = _expand_rows(B, M)
    X_V_e = _expand_rows(X_V, M)
    X_E_e = _expand_rows(X_E, M)
    total, _, _, _ = energy_batch(u, A_V, A_E, R, P_V_e, P_E_e, B_e, X_V_e, X_E_e, lam_B)
    total = total.reshape(G, M)
    accept = total <= total0[:, None] + cfg.alpha_tol
    any_acc = accept.any(1)
    first = accept.to(torch.int64).argmax(1)
    idx = torch.arange(G, device=alpha.device) * M + first
    new = u[idx]
    alpha_new = torch.where(any_acc[:, None], new, alpha)
    return alpha_new, (~any_acc)


def infer_batch(B, X_V, X_E, n, m, atom_valid, bond_valid, A_V, A_E, R, cfg,
                lam_B=1.0):
    """Batched inference.  Returns dict with alpha, P_V, P_E, components."""
    torch = _torch()
    G, N, _ = B.shape
    qV, qE, K = A_V.shape[1], A_E.shape[1], A_V.shape[0]
    alpha = torch.full((G, K), 1.0 / K, dtype=B.dtype, device=B.device)
    P_V = (atom_valid / (qV * n[:, None])).unsqueeze(-1).expand(-1, -1, qV).contiguous()
    P_E = (bond_valid / (qE * m[:, None])).unsqueeze(-1).expand(-1, -1, qE).contiguous()
    c_Vcol = torch.full((G, qV), 1.0 / qV, dtype=B.dtype, device=B.device)
    c_Ecol = torch.full((G, qE), 1.0 / qE, dtype=B.dtype, device=B.device)
    r_V = atom_valid / n[:, None]
    r_E = bond_valid / m[:, None]
    marg_err = 0.0
    failures = torch.zeros(G, dtype=torch.bool, device=B.device)
    one_minus_B = 1.0 - B
    for _ in range(cfg.T_alt):
        AV = torch.einsum("gk,kac->gac", alpha, A_V)
        AE = torch.einsum("gk,kbc->gbc", alpha, A_E)
        RB = torch.einsum("gk,kab->gab", alpha, R)
        Usq = (1.0 - RB) ** 2
        Wsq = RB ** 2
        # --- P_V ---
        cV = _attr_cost_t(X_V, AV)
        T_U = torch.bmm(P_E, Usq.transpose(1, 2))
        T_W = torch.bmm(P_E, Wsq.transpose(1, 2))
        C_V = cV + lam_B * (torch.bmm(B, T_U) + torch.bmm(one_minus_B, T_W))
        P_V, e1 = sinkhorn_batch(C_V, r_V, c_Vcol, cfg.eps, cfg.sinkhorn_iters)
        marg_err = max(marg_err, e1)
        # --- P_E ---
        cE = _attr_cost_t(X_E, AE)
        S_U = torch.bmm(P_V, Usq)
        S_W = torch.bmm(P_V, Wsq)
        C_E = cE + lam_B * (torch.bmm(B.transpose(1, 2), S_U)
                            + torch.bmm(one_minus_B.transpose(1, 2), S_W))
        P_E, e2 = sinkhorn_batch(C_E, r_E, c_Ecol, cfg.eps, cfg.sinkhorn_iters)
        marg_err = max(marg_err, e2)
        # --- alpha ---
        for _step in range(cfg.alpha_steps):
            grad = alpha_grad_batch(alpha, A_V, A_E, R, P_V, P_E, B, X_V, X_E, lam_B)
            alpha, failed = alpha_line_search(alpha, grad, A_V, A_E, R, P_V, P_E,
                                              B, X_V, X_E, cfg, lam_B)
            failures = failures | failed
    total, L_V, L_E, L_B = energy_batch(alpha, A_V, A_E, R, P_V, P_E, B, X_V, X_E, lam_B)
    return {
        "alpha": alpha, "P_V": P_V, "P_E": P_E,
        "total": total, "L_V": L_V, "L_E": L_E, "L_B": L_B,
        "marginal_err": marg_err, "alpha_failures": failures,
    }


# ===========================================================================
# 4. Collate molecules -> padded batch
# ===========================================================================
def collate(mols: Sequence[Mol], node_index, bond_index, device="cpu",
            dtype=None):
    torch = _torch()
    if dtype is None:
        dtype = torch.float32
    G = len(mols)
    N = max(mol.n for mol in mols)
    M = max(mol.m for mol in mols)
    B = np.zeros((G, N, M), dtype=np.float64)
    X_V = np.zeros((G, N, C_V), dtype=np.float64)
    X_E = np.zeros((G, M, C_E), dtype=np.float64)
    av = np.zeros((G, N), dtype=np.float64)
    bv = np.zeros((G, M), dtype=np.float64)
    n = np.zeros(G, dtype=np.float64)
    m = np.zeros(G, dtype=np.float64)
    for i, mol in enumerate(mols):
        b, xv, xe = mol_arrays(mol, node_index, bond_index)
        B[i, :mol.n, :mol.m] = b
        X_V[i, :mol.n] = xv
        X_E[i, :mol.m] = xe
        av[i, :mol.n] = 1.0
        bv[i, :mol.m] = 1.0
        n[i] = mol.n
        m[i] = mol.m
    def t(a):
        return torch.as_tensor(a, dtype=dtype, device=device)
    return {
        "B": t(B), "X_V": t(X_V), "X_E": t(X_E),
        "atom_valid": t(av), "bond_valid": t(bv), "n": t(n), "m": t(m),
    }


# ===========================================================================
# 5. Dictionary init
# ===========================================================================
def empirical_freqs(mols, node_index, bond_index):
    cv = np.zeros(C_V, dtype=np.float64)
    ce = np.zeros(C_E, dtype=np.float64)
    for mol in mols:
        for c in mol.node_types.tolist():
            cv[node_index[int(c)]] += 1.0
        for c in mol.bond_types.tolist():
            ce[bond_index[int(c)]] += 1.0
    return cv / cv.sum(), ce / ce.sum()


def init_dictionary(mols, node_index, bond_index, cfg: TIGDConfig, seed: int = 0):
    rng = np.random.default_rng(seed)
    fV, fE = empirical_freqs(mols, node_index, bond_index)
    baseV = np.log(np.maximum(fV, 1e-12))
    baseE = np.log(np.maximum(fE, 1e-12))
    theta_V = baseV[None, None, :] + 0.25 * rng.standard_normal((cfg.K, cfg.qV, C_V))
    theta_E = baseE[None, None, :] + 0.25 * rng.standard_normal((cfg.K, cfg.qE, C_E))
    xi = 0.5 * rng.standard_normal((cfg.K, cfg.qV, cfg.qE))
    return {"theta_V": theta_V, "theta_E": theta_E, "xi": xi}, {"fV": fV, "fE": fE}


def atoms_np_from_dict(params):
    return atoms_from_logits_np(params["theta_V"], params["theta_E"], params["xi"])


# ===========================================================================
# 6. Code / transport / dictionary statistics
# ===========================================================================
def code_stats(alpha: np.ndarray) -> dict:
    a = np.clip(alpha, 1e-12, 1.0)
    H = -(a * np.log(a)).sum(1)
    mean_alpha = alpha.mean(0)
    ma = np.clip(mean_alpha, 1e-12, 1.0)
    counts = np.bincount(alpha.argmax(1), minlength=alpha.shape[1])
    return {
        "entropy_mean": float(H.mean()),
        "effective_support_mean": float(np.exp(H).mean()),
        "max_coef_mean": float(alpha.max(1).mean()),
        "max_coef_p90": float(np.percentile(alpha.max(1), 90)),
        "mean_alpha": mean_alpha.tolist(),
        "mean_alpha_effective_support": float(np.exp(-(ma * np.log(ma)).sum())),
        "top1_util_frac": float(counts.max() / len(alpha)),
        "argmax_counts": counts.tolist(),
    }


def transport_stats_batch(out, atom_valid, bond_valid, n, m, qV, qE):
    """Per-batch transport diagnostics.  Returns scalars + per-graph arrays."""
    torch = _torch()
    with torch.no_grad():
        pv = out["P_V"] * n[:, None, None]           # rows sum to 1
        pe = out["P_E"] * m[:, None, None]
        eps = 1e-12
        hv = -(pv.clamp_min(eps).log() * pv).sum(-1)
        he = -(pe.clamp_min(eps).log() * pe).sum(-1)
        mask_v = atom_valid > 0
        mask_e = bond_valid > 0
        ent_v = (hv * mask_v).sum(1) / mask_v.sum(1).clamp_min(1)
        ent_e = (he * mask_e).sum(1) / mask_e.sum(1).clamp_min(1)
        max_v = ((pv.max(-1).values * mask_v).sum(1) / mask_v.sum(1).clamp_min(1))
        max_e = ((pe.max(-1).values * mask_e).sum(1) / mask_e.sum(1).clamp_min(1))
    return {
        "transport_entropy_V": float(ent_v.mean()),
        "transport_entropy_E": float(ent_e.mean()),
        "transport_max_entry_V": float(max_v.mean()),
        "transport_max_entry_E": float(max_e.mean()),
    }


def dict_distances(atoms: dict) -> dict:
    A_V, A_E, R = atoms["A_V"], atoms["A_E"], atoms["R"]
    K = A_V.shape[0]
    flat = np.concatenate([A_V.reshape(K, -1), A_E.reshape(K, -1), R.reshape(K, -1)],
                          axis=1)
    d = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
    iu = np.triu_indices(K, k=1)
    off = d[iu]
    return {
        "min_pair_dist": float(off.min()),
        "mean_pair_dist": float(off.mean()),
        "max_pair_dist": float(off.max()),
        "pair_dist_matrix": d.tolist(),
        "near_duplicate_frac": float((off < 1e-2).mean()),
    }


# ===========================================================================
# 7. Training (block coordinate, detached inference)
# ===========================================================================
def infer_mols(mols, node_index, bond_index, A_V, A_E, R, cfg, device="cpu",
               batch_size=128, lam_B=1.0, dtype=None, log=print):
    """Batched inference over a molecule list; returns numpy summaries."""
    torch = _torch()
    if dtype is None:
        dtype = torch.float32
    alphas = []
    L_V, L_E, L_B, total = [], [], [], []
    marg_err = 0.0
    fails = []
    ent_v = ent_e = mx_v = mx_e = 0.0
    n_codes = 0
    with torch.no_grad():
        for start in range(0, len(mols), batch_size):
            chunk = mols[start:start + batch_size]
            b = collate(chunk, node_index, bond_index, device=device, dtype=dtype)
            out = infer_batch(b["B"], b["X_V"], b["X_E"], b["n"], b["m"],
                              b["atom_valid"], b["bond_valid"], A_V, A_E, R, cfg,
                              lam_B=lam_B)
            alphas.append(out["alpha"].detach().cpu().numpy())
            L_V.append(out["L_V"].detach().cpu().numpy())
            L_E.append(out["L_E"].detach().cpu().numpy())
            L_B.append(out["L_B"].detach().cpu().numpy())
            total.append(out["total"].detach().cpu().numpy())
            fails.append(out["alpha_failures"].detach().cpu().numpy())
            marg_err = max(marg_err, float(out["marginal_err"]))
            ts = transport_stats_batch(out, b["atom_valid"], b["bond_valid"],
                                       b["n"], b["m"], cfg.qV, cfg.qE)
            g = len(chunk)
            ent_v += ts["transport_entropy_V"] * g
            ent_e += ts["transport_entropy_E"] * g
            mx_v += ts["transport_max_entry_V"] * g
            mx_e += ts["transport_max_entry_E"] * g
            n_codes += g
    alpha = np.concatenate(alphas, axis=0)
    fails = np.concatenate(fails, axis=0)
    res = {
        "alpha": alpha,
        "L_V": float(np.mean(np.concatenate(L_V))),
        "L_E": float(np.mean(np.concatenate(L_E))),
        "L_B": float(np.mean(np.concatenate(L_B))),
        "total": float(np.mean(np.concatenate(total))),
        "marginal_err": marg_err,
        "backtrack_failure_rate": float(fails.mean()),
        "transport_entropy_V": ent_v / max(n_codes, 1),
        "transport_entropy_E": ent_e / max(n_codes, 1),
        "transport_max_entry_V": mx_v / max(n_codes, 1),
        "transport_max_entry_E": mx_e / max(n_codes, 1),
    }
    res.update(code_stats(alpha))
    res["code_pairwise_mean"] = _pairwise_mean(alpha)
    return res


def train_stage(mols_train, mols_eval, node_index, bond_index, params, cfg,
                *, epochs, batch_size=128, lr=1e-3, seed=0, device="cpu",
                log=print, eval_every=5):
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    theta_V = torch.nn.Parameter(torch.as_tensor(params["theta_V"], dtype=torch.float32,
                                                 device=device))
    theta_E = torch.nn.Parameter(torch.as_tensor(params["theta_E"], dtype=torch.float32,
                                                 device=device))
    xi = torch.nn.Parameter(torch.as_tensor(params["xi"], dtype=torch.float32,
                                            device=device))
    opt = torch.optim.Adam([theta_V, theta_E, xi], lr=lr, weight_decay=0.0)
    history = []
    n_train = len(mols_train)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_train)
        ep_total = 0.0
        ep_L = np.zeros(3)
        nb = 0
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            mols = [mols_train[i] for i in idx]
            b = collate(mols, node_index, bond_index, device=device)
            A_V, A_E, R = atoms_from_logits(torch, theta_V, theta_E, xi)
            with torch.no_grad():
                out = infer_batch(b["B"], b["X_V"], b["X_E"], b["n"], b["m"],
                                  b["atom_valid"], b["bond_valid"], A_V, A_E, R, cfg,
                                  lam_B=cfg.lam_B)
                alpha = out["alpha"].detach()
                P_V = out["P_V"].detach()
                P_E = out["P_E"].detach()
            A_V2, A_E2, R2 = atoms_from_logits(torch, theta_V, theta_E, xi)
            total, L_V, L_E, L_B = energy_batch(alpha, A_V2, A_E2, R2, P_V, P_E,
                                                b["B"], b["X_V"], b["X_E"], cfg.lam_B)
            loss = total.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_total += float(total.mean().detach())
            ep_L += np.array([float(L_V.mean()), float(L_E.mean()), float(L_B.mean())])
            nb += 1
        rec = {"epoch": epoch, "train_total": ep_total / max(nb, 1),
               "train_L_V": float(ep_L[0] / max(nb, 1)),
               "train_L_E": float(ep_L[1] / max(nb, 1)),
               "train_L_B": float(ep_L[2] / max(nb, 1))}
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            A_V, A_E, R = atoms_from_logits(torch, theta_V, theta_E, xi)
            ev = infer_mols(mols_eval, node_index, bond_index,
                            A_V.detach(), A_E.detach(), R.detach(), cfg,
                            device=device, batch_size=batch_size, log=log)
            atoms_d = {
                "A_V": A_V.detach().cpu().double().numpy(),
                "A_E": A_E.detach().cpu().double().numpy(),
                "R": R.detach().cpu().double().numpy(),
            }
            dd = dict_distances(atoms_d)
            rec.update({
                "eval_total": ev["total"], "eval_L_V": ev["L_V"],
                "eval_L_E": ev["L_E"], "eval_L_B": ev["L_B"],
                "code_entropy": ev["entropy_mean"],
                "code_effective_support": ev["effective_support_mean"],
                "code_max_coef": ev["max_coef_mean"],
                "code_max_coef_p90": ev["max_coef_p90"],
                "code_pairwise_mean": ev["code_pairwise_mean"],
                "mean_alpha_effective_support": ev["mean_alpha_effective_support"],
                "top1_util_frac": ev["top1_util_frac"],
                "argmax_counts": ev["argmax_counts"],
                "mean_alpha": ev["mean_alpha"],
                "transport_entropy_V": ev["transport_entropy_V"],
                "transport_entropy_E": ev["transport_entropy_E"],
                "transport_max_entry_V": ev["transport_max_entry_V"],
                "transport_max_entry_E": ev["transport_max_entry_E"],
                "sinkhorn_marginal_err": ev["marginal_err"],
                "backtrack_failure_rate": ev["backtrack_failure_rate"],
                "atom_min_pair_dist": dd["min_pair_dist"],
                "atom_mean_pair_dist": dd["mean_pair_dist"],
                "near_duplicate_frac": dd["near_duplicate_frac"],
                "dead_atoms": int(np.sum(np.array(ev["argmax_counts"]) == 0)),
            })
            log(f"[epoch {epoch:3d}] total={rec['train_total']:.4f} "
                f"L_V={rec['train_L_V']:.4f} L_E={rec['train_L_E']:.4f} "
                f"L_B={rec['train_L_B']:.4f} | codeH={rec['code_entropy']:.3f} "
                f"supp={rec['code_effective_support']:.2f} "
                f"util_dead={rec['dead_atoms']} | marg={rec['sinkhorn_marginal_err']:.2e}")
        history.append(rec)
    result_params = {
        "theta_V": theta_V.detach().cpu().double().numpy(),
        "theta_E": theta_E.detach().cpu().double().numpy(),
        "xi": xi.detach().cpu().double().numpy(),
    }
    return result_params, history


# ===========================================================================
# 8. Mechanism audit (holdout: original / permutation / rewiring / random)
# ===========================================================================
def size_matched_partner(rng, mol, pool, used_idx):
    cand = sorted(range(len(pool)), key=lambda j: abs(pool[j].n - mol.n) + abs(pool[j].m - mol.m))
    top = [j for j in cand[:20] if j not in used_idx][:20] or cand[:20]
    return pool[int(top[int(rng.integers(0, len(top)))])]


def mechanism_audit(mols_holdout, mols_pool, node_index, bond_index, atoms, cfg,
                    device="cpu", batch_size=128, seed: int = 0, log=print):
    torch = _torch()
    rng = np.random.default_rng(seed)
    A_V = torch.as_tensor(atoms["A_V"], dtype=torch.float32, device=device)
    A_E = torch.as_tensor(atoms["A_E"], dtype=torch.float32, device=device)
    R = torch.as_tensor(atoms["R"], dtype=torch.float32, device=device)

    def infer_one_list(mols, lam_B=1.0):
        res = infer_mols(mols, node_index, bond_index, A_V, A_E, R, cfg,
                         device=device, batch_size=batch_size, lam_B=lam_B,
                         log=lambda *a: None)
        return res

    originals, perms, rewires, randoms = [], [], [], []
    rewire_ok = np.zeros(len(mols_holdout), dtype=bool)
    for i, mol in enumerate(mols_holdout):
        originals.append(mol)
        pa = rng.permutation(mol.n)
        pbs = rng.permutation(mol.m)
        perms.append(permute_mol(mol, pa, pbs))
        rw = two_switch(rng, mol)
        if rw is None:
            rewires.append(mol)
        else:
            rewire_ok[i] = True
            rewires.append(rw)
        randoms.append(size_matched_partner(rng, mol, mols_pool, {i}))

    def infer(mols, lam_B=1.0):
        return infer_mols(mols, node_index, bond_index, A_V, A_E, R, cfg,
                          device=device, batch_size=batch_size, lam_B=lam_B,
                          log=lambda *a: None)

    o = infer(originals)
    p = infer(perms)
    r = infer(rewires)
    q = infer(randoms)
    o_attr = infer(originals, lam_B=0.0)
    r_attr = infer(rewires, lam_B=0.0)

    def d1(a, b):
        return np.abs(a - b).sum(1)

    d_perm = d1(o["alpha"], p["alpha"])
    d_rewire = d1(o["alpha"], r["alpha"])
    d_random = d1(o["alpha"], q["alpha"])
    d_struct = d1(o["alpha"], o_attr["alpha"])
    d_rewire_attr = d1(o_attr["alpha"], r_attr["alpha"])
    return {
        "n_holdout": len(mols_holdout),
        "rewire_success_rate": float(rewire_ok.mean()),
        "original": {k: o[k] for k in ("L_V", "L_E", "L_B", "total", "marginal_err",
                                       "entropy_mean", "effective_support_mean")},
        "permutation": {k: p[k] for k in ("L_V", "L_E", "L_B", "total")},
        "rewire": {k: r[k] for k in ("L_V", "L_E", "L_B", "total")},
        "random": {k: q[k] for k in ("L_V", "L_E", "L_B", "total")},
        "alpha_delta": {
            "perm_mean": float(d_perm.mean()), "perm_p90": float(np.percentile(d_perm, 90)),
            "rewire_mean": float(d_rewire.mean()), "rewire_p90": float(np.percentile(d_rewire, 90)),
            "random_mean": float(d_random.mean()), "random_p90": float(np.percentile(d_random, 90)),
            "struct_off_mean": float(d_struct.mean()),
            "struct_off_p90": float(np.percentile(d_struct, 90)),
            "rewire_attr_mean": float(d_rewire_attr.mean()),
        },
        "code_pairwise_dist": {
            "original_mean": _pairwise_mean(o["alpha"]),
            "attr_only_mean": _pairwise_mean(o_attr["alpha"]),
        },
        "struct_off": {k: o_attr[k] for k in ("L_V", "L_E", "L_B", "total")},
        "struct_off_rewire": {k: r_attr[k] for k in ("L_V", "L_E", "L_B", "total")},
        "codes_original": o["alpha"], "codes_perm": p["alpha"],
        "codes_rewire": r["alpha"], "codes_random": q["alpha"],
        "codes_attr": o_attr["alpha"],
    }


def _pairwise_mean(alpha):
    if len(alpha) < 2:
        return 0.0
    # subsample for speed
    a = alpha if len(alpha) <= 512 else alpha[:512]
    d = np.abs(a[:, None, :] - a[None, :, :]).sum(2)
    iu = np.triu_indices(len(a), k=1)
    return float(d[iu].mean())


# ===========================================================================
# 9. Frozen-code target probe (only use of y)
# ===========================================================================
def run_probe(alpha_train, y_train, alpha_valid, y_valid, seed=0):
    out = {}
    out["linear_ridge"] = linear_ridge(alpha_train, y_train, alpha_valid, y_valid)
    out["linear_probe"] = train_probe(alpha_train, y_train, alpha_valid, y_valid,
                                      hidden=None, seed=seed)
    out["mlp_probe"] = train_probe(alpha_train, y_train, alpha_valid, y_valid,
                                   hidden=32, seed=seed)
    return out


# ===========================================================================
# 10. Visualisation
# ===========================================================================
def save_atom_vis(atoms: dict, alpha_mean: np.ndarray, out_path: Path, top_k: int = 4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        return {"saved": False, "reason": str(exc)}
    A_V, A_E, R = atoms["A_V"], atoms["A_E"], atoms["R"]
    order = np.argsort(-np.asarray(alpha_mean))[:top_k]
    fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4.2))
    if len(order) == 1:
        axes = [axes]
    qV, qE = A_V.shape[1], A_E.shape[1]
    for ax, k in zip(axes, order):
        ax.set_title(f"atom {k} (mean alpha={alpha_mean[k]:.3f})")
        for a in range(qV):
            for b in range(qE):
                w = R[k, a, b]
                if w > 0.25:
                    ax.plot([0, 1], [a, b], color="tab:blue", alpha=min(1.0, w / 2.0),
                            lw=0.6)
        ax.scatter([0] * qV, range(qV), s=6, c="tab:orange")
        ax.scatter([1] * qE, range(qE), s=6, c="tab:green")
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-1, max(qV, qE))
        ax.set_yticks([])
        ax.set_xticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {"saved": True, "path": str(out_path)}


# ===========================================================================
# 11. Gates
# ===========================================================================
def pilot_gate(history: list, audit: dict, cfg: TIGDConfig) -> dict:
    last = history[-1]
    d = audit["alpha_delta"]
    checks = {
        "dictionary_no_collapse": (last["near_duplicate_frac"] < 0.5
                                   and last["atom_min_pair_dist"] > 0.05
                                   and last["dead_atoms"] < cfg.K
                                   and last["top1_util_frac"] < 0.8),
        "code_diversity": (last["code_pairwise_mean"] > 0.1),
        "permutation_invariance": (d["perm_mean"] < 1e-3),
        "rewiring_changes_code": (d["rewire_mean"] > 10 * max(d["perm_mean"], 1e-6)),
        "random_ge_rewiring": (d["random_mean"] >= d["rewire_mean"]),
        "structure_learned": (last["eval_L_B"] < history[0].get("eval_L_B", np.inf)),
        "structure_off_changes_code": (d["struct_off_mean"] > 1e-3),
        "transport_nonuniform": (last["transport_max_entry_V"] > 1.2 / cfg.qV
                                 and last["transport_entropy_V"] < math.log(cfg.qV) * 0.98),
        "recon_separation": (audit["random"]["total"] > audit["original"]["total"] * 1.01),
        "solver_stable": (np.isfinite(last["sinkhorn_marginal_err"])
                          and last["sinkhorn_marginal_err"] < 1e-2),
    }
    checks["all_pass"] = all(v for k, v in checks.items() if k != "all_pass")
    return checks


# ===========================================================================
# 12. Orchestration
# ===========================================================================
def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return None


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float)
                    + "\n", encoding="utf-8")


def select_train(mols, count, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(mols))[:count]
    return [mols[i] for i in idx], idx.tolist()


def load_labels(root: Path, split: str, limit: int | None = None):
    """Load only the regression targets (official test is refused)."""
    if split == "test":
        raise RuntimeError("TIGD never loads the official test split")
    load_zinc, _ = repo_loader()
    ds = load_zinc(root, "val" if split == "valid" else split)
    n = len(ds) if limit is None else min(int(limit), len(ds))
    return np.array([float(ds[i].y.reshape(-1)[0]) for i in range(n)], dtype=np.float64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--stage", choices=["pilot", "full", "both"], default="pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pilot-graphs", type=int, default=PILOT_GRAPHS)
    parser.add_argument("--full-graphs", type=int, default=FULL_GRAPHS)
    parser.add_argument("--holdout-graphs", type=int, default=HOLDOUT_GRAPHS)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.time()
    cfg = TIGDConfig()
    def log(*a):
        print(*a, flush=True)
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    log("loading ZINC train / valid (official test never loaded)")
    mols_train = load_mols(args.data_root, "train")
    mols_valid = load_mols(args.data_root, "valid")
    node_index, bond_index = make_codebook(mols_train)
    assert (len(node_index), len(bond_index)) == (C_V, C_E)
    if args.limit:
        mols_train = mols_train[:args.limit]
        mols_valid = mols_valid[:args.limit]
    log(f"train={len(mols_train)} valid={len(mols_valid)} "
        f"C_V={len(node_index)} C_E={len(bond_index)}")

    stages = [args.stage] if args.stage != "both" else ["pilot", "full"]
    summary: dict[str, Any] = {
        "round": "TIGD-v0",
        "config": cfg.__dict__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain"),
        "python": platform.python_version(),
        "host": platform.node(),
        "device": args.device,
        "train_size": len(mols_train),
        "valid_size": len(mols_valid),
        "frozen_refs": {"doi_mae": REF_DOI_MAE, "scatter_mae": REF_SCATTER_MAE,
                        "icrate_mae": REF_ICRATE_MAE},
    }

    pilot_gate_result = None
    for stage in stages:
        if stage == "pilot":
            n_graphs = min(args.pilot_graphs, len(mols_train))
            epochs = args.epochs or PILOT_EPOCHS
            if args.smoke:
                n_graphs, epochs = min(64, len(mols_train)), 3
        else:
            if args.skip_full:
                continue
            if pilot_gate_result is not None and not pilot_gate_result["all_pass"]:
                log("pilot gate failed; skipping full stage")
                break
            n_graphs = min(args.full_graphs, len(mols_train))
            epochs = args.epochs or FULL_EPOCHS
        stage_dir = out_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== stage {stage}: {n_graphs} graphs, {epochs} epochs ===")

        sub, sub_idx = select_train(mols_train, n_graphs, seed=args.seed)
        params, _freqs = init_dictionary(sub, node_index, bond_index, cfg, seed=args.seed)
        eval_mols = sub if len(sub) <= 512 else [sub[i] for i in
                                                 np.random.default_rng(1).permutation(len(sub))[:512]]
        params, history = train_stage(sub, eval_mols, node_index, bond_index, params,
                                      cfg, epochs=epochs, batch_size=args.batch_size,
                                      seed=args.seed, device=args.device, log=log)
        atoms = atoms_np_from_dict(params)
        np.savez(stage_dir / "dictionary.npz", **params)
        write_json(stage_dir / "history.json", {"history": history, "config": cfg.__dict__})
        dd = dict_distances(atoms)
        log(f"dictionary pairwise distance: min={dd['min_pair_dist']:.4f} "
            f"mean={dd['mean_pair_dist']:.4f} near_dup={dd['near_duplicate_frac']:.3f}")

        # --- holdout mechanism audit ---
        used = set(sub_idx)
        holdout_pool = [i for i in range(len(mols_train)) if i not in used]
        if len(holdout_pool) >= args.holdout_graphs:
            rng = np.random.default_rng(args.seed + 7)
            holdout_pool = list(rng.permutation(holdout_pool)[:args.holdout_graphs])
            holdout = [mols_train[i] for i in holdout_pool]
            partner_pool = [mols_train[i] for i in holdout_pool[:2000]] or mols_train[:512]
        else:
            # full stage uses all train graphs -> perturb a valid holdout instead
            rng = np.random.default_rng(args.seed + 7)
            holdout_pool = list(rng.permutation(len(mols_valid))[:args.holdout_graphs])
            holdout = [mols_valid[i] for i in holdout_pool]
            partner_pool = mols_valid[:512]
        log(f"mechanism audit on {len(holdout)} holdout graphs")
        audit = mechanism_audit(holdout, partner_pool, node_index, bond_index, atoms,
                                cfg, device=args.device, batch_size=args.batch_size,
                                seed=args.seed, log=log)
        codes = {k: v for k, v in audit.items() if k.startswith("codes_")}
        audit_light = {k: v for k, v in audit.items() if not k.startswith("codes_")}
        np.savez(stage_dir / "codes_audit.npz", **codes)
        write_json(stage_dir / "mechanism_audit.json", audit_light)
        log("alpha delta: perm=%.3e rewire=%.3e random=%.3e struct_off=%.3e"
            % (audit["alpha_delta"]["perm_mean"], audit["alpha_delta"]["rewire_mean"],
               audit["alpha_delta"]["random_mean"], audit["alpha_delta"]["struct_off_mean"]))
        log("energy: orig=%.4f rewire=%.4f random=%.4f"
            % (audit["original"]["total"], audit["rewire"]["total"],
               audit["random"]["total"]))

        # --- full inference for probe ---
        A_Vt = _torch().as_tensor(atoms["A_V"], dtype=_torch().float32, device=args.device)
        A_Et = _torch().as_tensor(atoms["A_E"], dtype=_torch().float32, device=args.device)
        Rt = _torch().as_tensor(atoms["R"], dtype=_torch().float32, device=args.device)
        tr_inf = infer_mols(sub, node_index, bond_index, A_Vt, A_Et, Rt, cfg,
                            device=args.device, batch_size=args.batch_size, log=log)
        va_inf = infer_mols(mols_valid, node_index, bond_index, A_Vt, A_Et, Rt, cfg,
                            device=args.device, batch_size=args.batch_size, log=log)
        np.savez(stage_dir / "codes.npz", alpha_train=tr_inf["alpha"],
                 alpha_valid=va_inf["alpha"])
        write_json(stage_dir / "inference_summary.json",
                   {"train": {k: v for k, v in tr_inf.items() if k != "alpha"},
                    "valid": {k: v for k, v in va_inf.items() if k != "alpha"}})

        if not args.skip_probe:
            # y is loaded here and only here (frozen-code probe)
            y_train_all = load_labels(args.data_root, "train", limit=len(mols_train))
            y_valid_all = load_labels(args.data_root, "valid", limit=len(mols_valid))
            y_sub = y_train_all[np.asarray(sub_idx, dtype=np.int64)]
            log("running frozen-code probe (only use of y)")
            probe = run_probe(tr_inf["alpha"], y_sub, va_inf["alpha"],
                              y_valid_all, seed=args.seed)
            write_json(stage_dir / "probe.json", probe)
            log("probe valid MAE: linear=%.4f linear_head=%.4f mlp=%.4f"
                % (probe["linear_ridge"]["valid_mae"], probe["linear_probe"]["valid_mae"],
                   probe["mlp_probe"]["valid_mae"]))

        vis = save_atom_vis(atoms, tr_inf["mean_alpha"], stage_dir / "atoms.png")
        write_json(stage_dir / "dict_stats.json",
                   {"distances": {k: v for k, v in dd.items() if k != "pair_dist_matrix"},
                    "mean_alpha": tr_inf["mean_alpha"],
                    "argmax_counts": tr_inf["argmax_counts"], "vis": vis})

        summary[stage] = {
            "n_graphs": n_graphs, "epochs": epochs,
            "final": history[-1], "dict_distances": {k: v for k, v in dd.items()
                                                     if k != "pair_dist_matrix"},
            "mechanism_audit": audit_light,
            "train_inference": {k: v for k, v in tr_inf.items() if k != "alpha"},
            "valid_inference": {k: v for k, v in va_inf.items() if k != "alpha"},
        }
        if not args.skip_probe:
            summary[stage]["probe"] = probe
        if stage == "pilot":
            pilot_gate_result = pilot_gate(history, audit, cfg)
            summary["pilot_gate"] = pilot_gate_result
            write_json(stage_dir / "pilot_gate.json", pilot_gate_result)
            log("pilot gate: %s" % json.dumps(pilot_gate_result, sort_keys=True))
        write_json(stage_dir / "summary.json", summary)

    summary["wall_seconds"] = time.time() - t0
    write_json(out_root / "SUMMARY.json", summary)
    log(f"done in {summary['wall_seconds']:.1f}s -> {out_root/'SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
