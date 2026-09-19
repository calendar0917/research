#!/usr/bin/env python
"""AIOM representation-domain analysis — Attributed Incidence Operator Moments.

Question (pre-dictionary, no model is trained as a contribution)
---------------------------------------------------------------
Can a *variable-size*, *permutation-free*, *attribute-carrying* molecular graph
be mapped by an almost-learning-free deterministic lift into a common
finite-dimensional **linear** space that keeps enough structure and attribute
information for sparse dictionary learning to be a reasonable next step?

The candidate lift is

    M_t(G) = X_G^T S_G^t X_G ,   t = 0..T_max ,   T_max = 8

with the atom--bond incidence graph, its symmetric normalised operator
``S_G = D^{-1/2} A D^{-1/2}`` and the *raw* one-hot object matrix ``X_G``
(no learned projection ``W``).  Taking ``vech`` of each symmetric moment and
stacking gives the fixed-dimensional whole-graph vector ``Phi_T(G)``.

Discipline
----------
* Only the official ZINC ``train`` and ``valid`` splits are ever loaded.  The
  official ``test`` split is never read, instantiated or referenced.
* ``y`` is never used except in the final capacity probe (Test F).
* No dictionary, no ISTA, no K-SVD, no learned projection, no architecture
  candidate is implemented.
* This note/script does not modify `STATE.yaml` or any historical record.

Reuse
-----
Exact colored-incidence canonicalization and the ZINC loader are reused from
the previous round (``run_wholegraph_canonical_registration_audit.py`` /
``tracks/ksvd/experiments/luyin16/zinc_long_range_proxy.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

# Analysis T schedule (prefixes of one T_max=8 computation).
T_MAIN = [0, 1, 2, 3, 4, 6, 8]
T_NN = [0, 1, 2, 4, 8]
T_MAX = 8

RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/aiom_representation"


# ===========================================================================
# 0. Repository loader + molecule container
# ===========================================================================
def repo_loader():
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (  # type: ignore
        _data_to_graph,
        _load_zinc,
    )

    return _load_zinc, _data_to_graph


@dataclass
class Mol:
    """Undirected attributed molecule with atom ids 0..n-1.

    ``bonds[i]`` is an undirected endpoint pair (a, b) with a < b and
    ``bond_types[i]`` is its bond category; ``node_types[v]`` is the atom
    category.  A bond object is one real chemical bond (never a directed copy).
    """

    n: int
    bonds: list[tuple[int, int]]
    bond_types: np.ndarray
    node_types: np.ndarray

    @property
    def m(self) -> int:
        return len(self.bonds)


def mol_from_graph_parts(graph, node_types, edge_types) -> Mol:
    bonds = sorted((int(a), int(b)) for (a, b) in edge_types.keys())
    bond_types = np.array([int(edge_types[b]) for b in bonds], dtype=np.int64)
    node_types = np.asarray(node_types, dtype=np.int64).reshape(-1)
    return Mol(int(graph.n), bonds, bond_types, node_types)


def mol_from_pyg(data, data_to_graph) -> Mol:
    graph, node_types, edge_types = data_to_graph(data)
    return mol_from_graph_parts(graph, node_types, edge_types)


def graph_and_edge_dict(mol: Mol):
    from tracks.ksvd.code.graph import from_edges  # type: ignore

    graph = from_edges(mol.n, mol.bonds)
    edge_types = {b: int(mol.bond_types[i]) for i, b in enumerate(mol.bonds)}
    return graph, edge_types


# ===========================================================================
# 1. Exact colored-incidence canonical key + typed WL fingerprint (reused)
# ===========================================================================
def canonical_key_bytes(mol: Mol) -> bytes:
    from tracks.ksvd.code.run_wholegraph_canonical_registration_audit import (  # type: ignore
        canonical_atom_order,
    )

    graph, edge_types = graph_and_edge_dict(mol)
    _order, key = canonical_atom_order(graph, mol.node_types, edge_types)
    return key


def iso_key_hex(mol: Mol) -> str:
    return hashlib.sha256(canonical_key_bytes(mol)).hexdigest()


def wl_fingerprint(mol: Mol, rounds: int = 3) -> dict:
    from tracks.ksvd.code.run_wholegraph_canonical_registration_audit import (  # type: ignore
        attributed_wl_fingerprint,
    )

    graph, edge_types = graph_and_edge_dict(mol)
    return attributed_wl_fingerprint(graph, mol.node_types, edge_types, rounds=rounds)


def permute_mol(mol: Mol, perm_atom: Sequence[int], perm_bond: Sequence[int]) -> Mol:
    """Relabel atoms by ``perm_atom[v]`` and reorder bonds by ``perm_bond[e]``."""
    perm_atom = np.asarray(perm_atom, dtype=np.int64)
    new_node_types = np.empty_like(mol.node_types)
    new_node_types[perm_atom] = mol.node_types
    bonds = [(int(perm_atom[a]), int(perm_atom[b])) for (a, b) in mol.bonds]
    order = np.asarray(perm_bond, dtype=np.int64)
    bonds = [bonds[i] for i in order]
    bond_types = mol.bond_types[order]
    return Mol(mol.n, bonds, bond_types, new_node_types)


# ===========================================================================
# 2. AIOM lift
# ===========================================================================
def incidence_matrix(mol: Mol, C_V: int, C_E: int, node_index, bond_index):
    """Return (A_G incidence bipartite adjacency, X_G raw one-hot)."""
    n, m = mol.n, mol.m
    N = n + m
    A = np.zeros((N, N), dtype=np.float64)
    X = np.zeros((N, C_V + C_E), dtype=np.float64)
    for v in range(n):
        X[v, node_index[int(mol.node_types[v])]] = 1.0
    for e, (a, b) in enumerate(mol.bonds):
        ev = n + e
        A[a, ev] = 1.0
        A[ev, a] = 1.0
        A[b, ev] = 1.0
        A[ev, b] = 1.0
        X[ev, C_V + bond_index[int(mol.bond_types[e])]] = 1.0
    return A, X


def aiom_moments(
    mol: Mol,
    C_V: int,
    C_E: int,
    node_index,
    bond_index,
    T_max: int = T_MAX,
) -> np.ndarray:
    """Return ``M_0..M_{T_max}`` of shape ``(T_max+1, C, C)``."""
    A, X = incidence_matrix(mol, C_V, C_E, node_index, bond_index)
    deg = A.sum(axis=1)
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    S = (dinv[:, None] * A) * dinv[None, :]
    H = X
    moments = np.empty((T_max + 1, X.shape[1], X.shape[1]), dtype=np.float64)
    for t in range(T_max + 1):
        moments[t] = X.T @ H
        if t < T_max:
            H = S @ H
    return moments


def phi_from_moments(moments: np.ndarray, triu) -> np.ndarray:
    return np.concatenate([moments[t][triu] for t in range(moments.shape[0])])


def upper_tri_indices(C: int):
    return np.triu_indices(C)


def prefix_slice(phi_full: np.ndarray, T: int, dim_per_moment: int) -> np.ndarray:
    return phi_full[: (T + 1) * dim_per_moment]


# ===========================================================================
# 3. Feature extraction over a split (cached)
# ===========================================================================
def load_mols(root: Path, split: str, limit: int | None = None) -> list[Mol]:
    if split == "test":
        raise RuntimeError("AIOM analysis never loads the official test split")
    load_zinc, data_to_graph = repo_loader()
    pyg_split = "val" if split == "valid" else split
    dataset = load_zinc(root, pyg_split)
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    return [mol_from_pyg(dataset[i], data_to_graph) for i in range(n)]


def build_codebook(mols: Sequence[Mol]):
    atoms = sorted({int(c) for mol in mols for c in mol.node_types.tolist()})
    bonds = sorted({int(c) for mol in mols for c in mol.bond_types.tolist()})
    return atoms, bonds


def encode_phis(mols: Sequence[Mol], C_V: int, C_E: int, node_index, bond_index,
                triu, T_max: int = T_MAX) -> np.ndarray:
    rows = []
    for mol in mols:
        moments = aiom_moments(mol, C_V, C_E, node_index, bond_index, T_max)
        rows.append(phi_from_moments(moments, triu))
    return np.asarray(rows, dtype=np.float64)


# ===========================================================================
# 4. Standardisation + distance / NN helpers
# ===========================================================================
def fit_standardizer(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def apply_standardizer(F: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (F - mu) / sd


def nearest_non_iso(
    F: np.ndarray,
    iso_ids: np.ndarray,
    *,
    top_k: int = 10,
    chunk: int = 256,
):
    """Nearest *non-isomorphic* neighbours in standardized feature space.

    Returns ``(nn_index, nn_dist)`` of shape ``(n, top_k)`` and ``(n, top_k)``.
    """
    n, d = F.shape
    F = np.ascontiguousarray(F, dtype=np.float32)
    sq = np.einsum("ij,ij->i", F, F)
    nn_index = np.full((n, top_k), -1, dtype=np.int64)
    nn_dist = np.full((n, top_k), np.inf, dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = F[start:stop] @ F.T  # (b, n)
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * block
        np.maximum(d2, 0.0, out=d2)
        for r in range(stop - start):
            i = start + r
            d2[r, iso_ids == iso_ids[i]] = np.inf  # exclude own isomorphism class
            k = min(top_k, n - 1)
            if k <= 0:
                continue
            idx = np.argpartition(d2[r], k - 1)[:k]
            order = np.argsort(d2[r, idx])
            idx = idx[order]
            nn_index[i, :k] = idx
            nn_dist[i, :k] = np.sqrt(d2[r, idx])
    return nn_index, nn_dist


def cosine_similarity_matrix(W: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Wn = W / norms
    return Wn @ Wn.T


# ===========================================================================
# 5. Test A — permutation invariance
# ===========================================================================
def stage_invariance(mols, C_V, C_E, node_index, bond_index, triu,
                     n_graphs=500, n_perms=20, seed=20260919, log=print):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(mols), size=min(n_graphs, len(mols)), replace=False)
    worst = 0.0
    worst_at = None
    checks = 0
    for i in idx:
        mol = mols[i]
        base = phi_from_moments(
            aiom_moments(mol, C_V, C_E, node_index, bond_index, T_MAX), triu
        )
        for _ in range(n_perms):
            pa = rng.permutation(mol.n)
            pb = rng.permutation(mol.m) if mol.m > 0 else np.zeros(0, dtype=np.int64)
            pm = permute_mol(mol, pa, pb)
            other = phi_from_moments(
                aiom_moments(pm, C_V, C_E, node_index, bond_index, T_MAX), triu
            )
            err = float(np.max(np.abs(other - base)))
            checks += 1
            if err > worst:
                worst = err
                worst_at = int(i)
    log(f"[Test A] {checks} checks, max abs err = {worst:.3e} (graph {worst_at})")
    return {"checks": checks, "max_abs_err": worst, "worst_graph": worst_at,
            "pass": worst < 1e-10}


# ===========================================================================
# 6. Test B — collisions
# ===========================================================================
def feature_hash(row: np.ndarray, decimals: int) -> str:
    q = np.round(np.ascontiguousarray(row, dtype=np.float64), decimals)
    q = q + 0.0  # normalise -0.0
    return hashlib.sha256(q.tobytes()).hexdigest()


def collision_statistics(phi_T: np.ndarray, iso_ids: np.ndarray, decimals: int):
    n = phi_T.shape[0]
    groups: dict[str, list[int]] = defaultdict(list)
    for i in range(n):
        groups[feature_hash(phi_T[i], decimals)].append(i)
    dup_only_groups = 0
    collision_groups = 0
    graphs_in_collision: set[int] = set()
    graphs_in_dup_only: set[int] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        classes = {int(iso_ids[i]) for i in members}
        if len(classes) > 1:
            collision_groups += 1
            graphs_in_collision.update(members)
        else:
            dup_only_groups += 1
            graphs_in_dup_only.update(members)
    n_coll_pairs = 0
    # pairwise within groups only (cheap)
    for members in groups.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                if iso_ids[members[a]] != iso_ids[members[b]]:
                    n_coll_pairs += 1
    return {
        "n_graphs": int(n),
        "n_unique_feature_hashes": int(len(groups)),
        "n_duplicate_only_groups": int(dup_only_groups),
        "n_collision_groups": int(collision_groups),
        "n_graphs_in_collision": int(len(graphs_in_collision)),
        "n_graphs_in_duplicate_only": int(len(graphs_in_dup_only)),
        "collision_graph_fraction": float(len(graphs_in_collision) / max(n, 1)),
        "n_noniso_collision_pairs": int(n_coll_pairs),
        "decimals": int(decimals),
    }


# ===========================================================================
# 7. Test C — rank / redundancy
# ===========================================================================
def rank_metrics(F_train_std: np.ndarray):
    s = np.linalg.svd(F_train_std, full_matrices=False, compute_uv=False)
    s = np.asarray(s, dtype=np.float64)
    energy = s ** 2
    total = float(energy.sum())
    if total <= 0:
        return {"numerical_rank": 0, "effective_rank_entropy": 0.0,
                "effective_rank_participation": 0.0,
                "dims_90": 0, "dims_95": 0, "dims_99": 0, "n_singular": int(s.size)}
    cum = np.cumsum(energy) / total
    def dims_at(frac):
        return int(np.searchsorted(cum, frac) + 1)
    p = energy / total
    p = p[p > 0]
    eff_entropy = float(np.exp(-(p * np.log(p)).sum()))
    eff_part = float((energy.sum() ** 2) / float((energy ** 2).sum()))
    tol = s[0] * max(F_train_std.shape) * np.finfo(np.float64).eps
    return {
        "numerical_rank": int((s > tol).sum()),
        "numerical_rank_tol": float(tol),
        "effective_rank_entropy": eff_entropy,
        "effective_rank_participation": eff_part,
        "dims_90": dims_at(0.90),
        "dims_95": dims_at(0.95),
        "dims_99": dims_at(0.99),
        "n_singular": int(s.size),
        "top_singular_values": [float(x) for x in s[:20]],
    }


# ===========================================================================
# 8. Test D — controlled perturbations (label-free)
# ===========================================================================
def perturb_atom_substitution(rng, mol: Mol, atom_categories) -> Mol | None:
    if mol.n == 0:
        return None
    v = int(rng.integers(0, mol.n))
    choices = [c for c in atom_categories if c != int(mol.node_types[v])]
    if not choices:
        return None
    nt = mol.node_types.copy()
    nt[v] = int(choices[int(rng.integers(0, len(choices)))])
    return Mol(mol.n, list(mol.bonds), mol.bond_types.copy(), nt)


def perturb_bond_type_swap(rng, mol: Mol) -> Mol | None:
    if mol.m < 2:
        return None
    order = rng.permutation(mol.m)
    for a in range(mol.m):
        for b in range(a + 1, mol.m):
            i, j = int(order[a]), int(order[b])
            if mol.bond_types[i] != mol.bond_types[j]:
                bt = mol.bond_types.copy()
                bt[i], bt[j] = bt[j], bt[i]
                return Mol(mol.n, list(mol.bonds), bt, mol.node_types.copy())
    return None


def two_switch(rng, mol: Mol) -> Mol | None:
    """Degree-preserving, feature-preserving endpoint rewiring of two bonds."""
    if mol.m < 2:
        return None
    edges = set(mol.bonds)
    m = mol.m
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    rng.shuffle(pairs)
    for i, j in pairs:
        a, b = mol.bonds[i]
        c, d = mol.bonds[j]
        if len({a, b, c, d}) < 4:
            continue
        candidates = [
            ((a, d), (c, b)),
            ((a, c), (b, d)),
        ]
        for e1, e2 in candidates:
            p1 = (min(e1), max(e1))
            p2 = (min(e2), max(e2))
            if p1[0] == p1[1] or p2[0] == p2[1]:
                continue
            if p1 == p2:
                continue
            if p1 in edges - {mol.bonds[i], mol.bonds[j]}:
                continue
            if p2 in edges - {mol.bonds[i], mol.bonds[j]}:
                continue
            new_bonds = list(mol.bonds)
            new_bonds[i] = p1
            new_bonds[j] = p2
            new_types = mol.bond_types.copy()
            new_types[i], new_types[j] = mol.bond_types[i], mol.bond_types[j]
            return Mol(mol.n, new_bonds, new_types, mol.node_types.copy())
    return None


def relative_delta(a: np.ndarray, b: np.ndarray, T: int, dim: int) -> float:
    pa = prefix_slice(a, T, dim)
    pb = prefix_slice(b, T, dim)
    return float(np.linalg.norm(pb - pa) / (np.linalg.norm(pa) + 1e-12))


# ===========================================================================
# 9. Test F — capacity probes
# ===========================================================================
def _torch():
    import torch  # noqa

    return torch


def train_probe(
    X_train, y_train, X_valid, y_valid, *, hidden: int | None, seed: int = 0,
    max_epochs: int = 240, patience: int = 40, batch_size: int = 128,
    lr: float = 1e-3, wd: float = 1e-5, clip: float = 5.0, log=print,
):
    torch = _torch()
    torch.manual_seed(seed)
    d = X_train.shape[1]
    if hidden is None:
        net = torch.nn.Sequential(torch.nn.Linear(d, 1))
    else:
        net = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1)
        )
    xtr = torch.as_tensor(X_train, dtype=torch.float32)
    ytr = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    xva = torch.as_tensor(X_valid, dtype=torch.float32)
    yva = torch.as_tensor(y_valid, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed + 91011)
    n = xtr.shape[0]
    best = float("inf")
    best_state = None
    best_epoch = 1
    stale = 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.randperm(n, generator=g).tolist()
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb = xtr[idx]
            yb = ytr[idx]
            pred = net(xb)
            loss = torch.nn.functional.l1_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
        net.eval()
        with torch.no_grad():
            vmae = float((net(xva) - yva).abs().mean())
        if vmae < best:
            best = vmae
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        final_train = float((net(xtr) - ytr).abs().mean())
        final_valid = float((net(xva) - yva).abs().mean())
    return {"train_mae": final_train, "valid_mae": final_valid,
            "best_epoch": best_epoch, "params": int(sum(p.numel() for p in net.parameters()))}


def linear_ridge(X_train, y_train, X_valid, y_valid, alpha: float = 1e-3):
    n, d = X_train.shape
    Xb = np.concatenate([X_train, np.ones((n, 1))], axis=1)
    G = Xb.T @ Xb + alpha * np.eye(d + 1)
    coef = np.linalg.solve(G, Xb.T @ y_train)
    def pred(X):
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) @ coef
    return {
        "train_mae": float(np.abs(pred(X_train) - y_train).mean()),
        "valid_mae": float(np.abs(pred(X_valid) - y_valid).mean()),
        "alpha": float(alpha),
    }


# ===========================================================================
# 10. Plot helpers
# ===========================================================================
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_plots(out: Path, collision_rows, rank_rows, perturbation_rows,
               nn_rows, probe_rows, log=print):
    plt = _mpl()
    out.mkdir(parents=True, exist_ok=True)

    # collision rate vs T
    fig, ax = plt.subplots(figsize=(5, 3.5))
    Ts = [r["T"] for r in collision_rows]
    ax.plot(Ts, [r["collision_graph_fraction"] for r in collision_rows], "o-")
    ax.set_xlabel("T")
    ax.set_ylabel("non-iso collision graph fraction")
    ax.set_title("AIOM exact collision rate vs T")
    fig.tight_layout()
    fig.savefig(out / "collision_rate_vs_T.png", dpi=130)
    plt.close(fig)

    # effective rank vs T
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(Ts, [r["effective_rank_entropy"] for r in rank_rows], "o-", label="entropy")
    ax.plot(Ts, [r["effective_rank_participation"] for r in rank_rows], "s--", label="participation")
    ax.set_xlabel("T")
    ax.set_ylabel("effective rank")
    ax.legend()
    ax.set_title("AIOM effective rank vs T")
    fig.tight_layout()
    fig.savefig(out / "effective_rank_vs_T.png", dpi=130)
    plt.close(fig)

    # perturbation ECDFs at a few T
    if perturbation_rows:
        fig, axes = plt.subplots(1, len(T_NN), figsize=(3.2 * len(T_NN), 3.0),
                                 sharey=True)
        cats = ["atom_sub", "bond_type_swap", "two_switch", "permutation", "random_pair"]
        for ax, T in zip(np.atleast_1d(axes), T_NN):
            for cat in cats:
                vals = sorted(r["delta"] for r in perturbation_rows
                              if r["T"] == T and r["kind"] == cat)
                if not vals:
                    continue
                y = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, y, label=cat)
            ax.set_title(f"T={T}")
            ax.set_xlabel(r"$\delta_T$")
        np.atleast_1d(axes)[0].set_ylabel("ECDF")
        np.atleast_1d(axes)[-1].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "perturbation_ecdf.png", dpi=130)
        plt.close(fig)

    # NN WL similarity vs T
    if nn_rows:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot([r["T"] for r in nn_rows], [r["top1_wl_sim"] for r in nn_rows], "o-", label="top-1")
        ax.plot([r["T"] for r in nn_rows], [r["top10_wl_sim"] for r in nn_rows], "s--", label="top-10 mean")
        ax.axhline(nn_rows[0]["random_control_wl_sim"], color="gray", ls=":",
                   label="size-matched random")
        ax.set_xlabel("T")
        ax.set_ylabel("WL cosine similarity")
        ax.legend()
        ax.set_title("Phi-space neighbours: WL structural similarity")
        fig.tight_layout()
        fig.savefig(out / "nn_wl_similarity_vs_T.png", dpi=130)
        plt.close(fig)

    # probe MAE vs T
    if probe_rows:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        lin = [r for r in probe_rows if r["probe"] == "linear"]
        mlp = [r for r in probe_rows if r["probe"] == "mlp"]
        ax.plot([r["T"] for r in lin], [r["train_mae"] for r in lin], "o-", label="linear train")
        ax.plot([r["T"] for r in lin], [r["valid_mae"] for r in lin], "o--", label="linear valid")
        ax.plot([r["T"] for r in mlp], [r["train_mae"] for r in mlp], "s-", label="MLP train")
        ax.plot([r["T"] for r in mlp], [r["valid_mae"] for r in mlp], "s--", label="MLP valid")
        ax.axhline(0.15, color="green", ls=":", alpha=0.6)
        ax.axhline(0.20, color="red", ls=":", alpha=0.6)
        ax.set_xlabel("T")
        ax.set_ylabel("MAE")
        ax.legend(fontsize=7)
        ax.set_title("AIOM dense capacity probe (raw y MAE)")
        fig.tight_layout()
        fig.savefig(out / "probe_mae_vs_T.png", dpi=130)
        plt.close(fig)
    log(f"[plots] saved to {out}")


# ===========================================================================
# 11. Orchestration
# ===========================================================================
def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:  # pragma: no cover
        return f"<error: {exc}>"


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
                    encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke run on a subset of graphs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-features", action="store_true",
                        help="reuse cached features npz")
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    t0 = time.time()

    limit = args.limit if args.limit is not None else (300 if args.quick else None)

    log_path = out / "run.log"
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log("=== AIOM representation audit start ===")
    provenance = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "data_root": str(args.data_root),
        "limit": limit,
        "seed": args.seed,
    }
    write_json(out / "provenance.json", provenance)
    log(f"provenance: commit={provenance['git_commit']}")

    train_mols = load_mols(args.data_root, "train", limit)
    valid_mols = load_mols(args.data_root, "valid", limit)
    log(f"loaded train={len(train_mols)} valid={len(valid_mols)} (test never loaded)")

    atom_cats, bond_cats = build_codebook(train_mols + valid_mols)
    C_V, C_E = len(atom_cats), len(bond_cats)
    node_index = {c: i for i, c in enumerate(atom_cats)}
    bond_index = {c: i for i, c in enumerate(bond_cats)}
    C = C_V + C_E
    triu = upper_tri_indices(C)
    dim = len(triu[0])
    log(f"codebook: C_V={C_V} C_E={C_E} C={C} dim_per_moment={dim}")

    # --- graph metadata ---
    sizes = np.array([m.n for m in train_mols] + [m.n for m in valid_mols])
    degrees = []
    for m in train_mols + valid_mols:
        deg = np.zeros(m.n, dtype=np.int64)
        for a, b in m.bonds:
            deg[a] += 1
            deg[b] += 1
        degrees.extend(deg.tolist())
    metadata = {
        "C_V": C_V, "C_E": C_E, "C": C, "dim_per_moment": dim,
        "atom_categories": atom_cats, "bond_categories": bond_cats,
        "train_graphs": len(train_mols), "valid_graphs": len(valid_mols),
        "graph_size": {"min": int(sizes.min()), "median": float(np.median(sizes)),
                       "p95": float(np.percentile(sizes, 95)), "max": int(sizes.max()),
                       "mean": float(sizes.mean())},
        "atom_degree": {"min": int(min(degrees)), "median": float(np.median(degrees)),
                        "max": int(max(degrees))},
        "T_max": T_MAX,
        "feature_dim_T8": (T_MAX + 1) * dim,
    }
    write_json(out / "graph_metadata.json", metadata)

    # directed-edge semantics audit on a sample
    load_zinc, data_to_graph = repo_loader()
    dataset = load_zinc(args.data_root, "train")
    n_audit = min(500, len(dataset))
    directed_ok = 0
    self_loops = 0
    multi_edges = 0
    inconsistent = 0
    for i in range(n_audit):
        data = dataset[i]
        ei = data.edge_index.detach().cpu().numpy()
        ea = data.edge_attr.detach().cpu().numpy().reshape(-1)
        seen = {}
        mult = defaultdict(int)
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u == v:
                self_loops += 1
                continue
            key = (min(u, v), max(u, v))
            mult[key] += 1
            if key in seen and seen[key] != int(ea[k]):
                inconsistent += 1
            seen[key] = int(ea[k])
        multi_edges += sum(1 for c in mult.values() if c > 2)
        directed_ok += 1 if all(c == 2 for c in mult.values()) else 0
    metadata["edge_semantics"] = {
        "sample_graphs": n_audit,
        "graphs_all_bonds_directed_twice": directed_ok,
        "self_loops": self_loops,
        "multi_edges_gt2": multi_edges,
        "inconsistent_directed_bond_attr": inconsistent,
    }
    write_json(out / "graph_metadata.json", metadata)
    log(f"edge semantics: {metadata['edge_semantics']}")

    # --- features (cached) ---
    cache = out / "aiom_features_T8.npz"
    if args.skip_features and cache.exists():
        d = np.load(cache, allow_pickle=True)
        phi_train = d["phi_train"]
        phi_valid = d["phi_valid"]
        iso_train = d["iso_train"]
        iso_valid = d["iso_valid"]
        log("loaded cached features")
    else:
        log("encoding train features ...")
        phi_train = encode_phis(train_mols, C_V, C_E, node_index, bond_index, triu)
        log("encoding valid features ...")
        phi_valid = encode_phis(valid_mols, C_V, C_E, node_index, bond_index, triu)
        log("computing exact iso keys (pynauty) ...")
        key_to_id: dict[str, int] = {}
        def iso_ids(mols):
            ids = np.empty(len(mols), dtype=np.int64)
            for i, mol in enumerate(mols):
                h = iso_key_hex(mol)
                if h not in key_to_id:
                    key_to_id[h] = len(key_to_id)
                ids[i] = key_to_id[h]
            return ids
        iso_train = iso_ids(train_mols)
        iso_valid = iso_ids(valid_mols)
        np.savez_compressed(cache, phi_train=phi_train, phi_valid=phi_valid,
                            iso_train=iso_train, iso_valid=iso_valid)
        log("features cached")

    phi_all = np.concatenate([phi_train, phi_valid], axis=0)
    iso_all = np.concatenate([iso_train, iso_valid])
    n_train = phi_train.shape[0]
    is_train = np.zeros(phi_all.shape[0], dtype=bool)
    is_train[:n_train] = True
    log(f"iso classes train={len(set(iso_train.tolist()))} valid_new={len(set(iso_valid.tolist()))}")

    # --- Test A ---
    inv = stage_invariance(train_mols, C_V, C_E, node_index, bond_index, triu,
                           n_graphs=(100 if args.quick else 500),
                           n_perms=(5 if args.quick else 20), seed=args.seed, log=log)
    write_json(out / "invariance.json", inv)

    # --- Test B collisions ---
    collision_rows = []
    for T in T_MAIN:
        phi_T = phi_all[:, : (T + 1) * dim]
        for dec in (10, 12):
            st = collision_statistics(phi_T, iso_all, dec)
            st["T"] = T
            collision_rows.append(st)
    write_json(out / "collision_metrics.json", collision_rows)
    log("collision metrics: " + ", ".join(
        f"T={r['T']}/1e-{r['decimals']}:{r['collision_graph_fraction']:.4f}"
        for r in collision_rows if r["decimals"] == 10))

    # near collisions + NN structural fidelity (shared computation)
    nn_rows = []
    near_records = []
    wl_hists = [wl_fingerprint(m, rounds=3) for m in train_mols + valid_mols]
    wl_keys = sorted({k for h in wl_hists for k in h}, key=repr)
    wl_index = {k: i for i, k in enumerate(wl_keys)}
    W = np.zeros((len(wl_hists), len(wl_keys)), dtype=np.float64)
    for r, h in enumerate(wl_hists):
        for k, v in h.items():
            W[r, wl_index[k]] = v
    wl_sim = cosine_similarity_matrix(W)

    size_all = np.array([m.n for m in train_mols + valid_mols])
    rng = np.random.default_rng(args.seed)
    # size-matched random control pairs
    rand_pairs = []
    for _ in range(20000):
        i = int(rng.integers(0, len(size_all)))
        cand = np.where(np.abs(size_all - size_all[i]) <= 1)[0]
        cand = cand[cand != i]
        if cand.size == 0:
            continue
        j = int(cand[rng.integers(0, cand.size)])
        rand_pairs.append((i, j))
    rand_pairs = np.array(rand_pairs, dtype=np.int64)
    log(f"size-matched random pairs: {len(rand_pairs)}")

    for T in T_NN:
        mu, sd = fit_standardizer(phi_train[:, : (T + 1) * dim])
        Z = apply_standardizer(phi_all[:, : (T + 1) * dim], mu, sd)
        nni, nnd = nearest_non_iso(Z, iso_all, top_k=10, chunk=(128 if args.quick else 256))
        valid_mask = nni >= 0
        top1_sim = np.array([wl_sim[i, nni[i, 0]] if valid_mask[i, 0] else np.nan
                             for i in range(len(iso_all))])
        top10_sim = np.array([
            np.nanmean([wl_sim[i, nni[i, k]] for k in range(10) if valid_mask[i, k]])
            for i in range(len(iso_all))
        ])
        # distances of nearest non-iso neighbours
        first = nnd[:, 0]
        first = first[np.isfinite(first)]
        # random-pair WL similarity control
        rand_wl = wl_sim[rand_pairs[:, 0], rand_pairs[:, 1]]
        row = {
            "T": T,
            "top1_wl_sim": float(np.nanmean(top1_sim)),
            "top10_wl_sim": float(np.nanmean(top10_sim)),
            "random_control_wl_sim": float(rand_wl.mean()),
            "nn_dist_min": float(np.min(first)) if first.size else None,
            "nn_dist_p1": float(np.percentile(first, 1)) if first.size else None,
            "nn_dist_median": float(np.median(first)) if first.size else None,
            "nn_dist_p95": float(np.percentile(first, 95)) if first.size else None,
        }
        nn_rows.append(row)
        # save nearest pairs sample (smallest distances) for manual inspection
        order = np.argsort(np.where(np.isfinite(nnd[:, 0]), nnd[:, 0], np.inf))
        for i in order[:40]:
            if not np.isfinite(nnd[i, 0]):
                continue
            near_records.append({
                "T": T, "i": int(i), "j": int(nni[i, 0]),
                "dist": float(nnd[i, 0]),
                "wl_sim": float(wl_sim[i, nni[i, 0]]),
                "same_iso": bool(iso_all[i] == iso_all[nni[i, 0]]),
            })
        log(f"[Test B/E] T={T} top1_wl={row['top1_wl_sim']:.3f} "
            f"top10_wl={row['top10_wl_sim']:.3f} rand={row['random_control_wl_sim']:.3f} "
            f"nn_med={row['nn_dist_median']:.3f}")
    write_json(out / "nearest_neighbor_metrics.json", nn_rows)
    # near collision pairs CSV
    with (out / "near_collision_pairs.csv").open("w", encoding="utf-8") as fh:
        fh.write("T,i,j,std_euclidean_dist,wl_cosine_sim,same_iso_class\n")
        for r in near_records:
            fh.write(f"{r['T']},{r['i']},{r['j']},{r['dist']:.6f},{r['wl_sim']:.6f},"
                     f"{int(r['same_iso'])}\n")

    # rank metrics
    rank_rows = []
    for T in T_MAIN:
        mu, sd = fit_standardizer(phi_train[:, : (T + 1) * dim])
        Ztr = apply_standardizer(phi_train[:, : (T + 1) * dim], mu, sd)
        st = rank_metrics(Ztr)
        st["T"] = T
        st["feature_dim"] = (T + 1) * dim
        rank_rows.append(st)
        log(f"[Test C] T={T} dim={st['feature_dim']} numrank={st['numerical_rank']} "
            f"eff_entropy={st['effective_rank_entropy']:.2f} dims99={st['dims_99']}")
    write_json(out / "rank_metrics.json", rank_rows)

    # --- Test D perturbations ---
    prng = np.random.default_rng(args.seed + 7)
    n_pert = 100 if args.quick else 500
    sample_idx = prng.choice(n_train, size=min(n_pert, n_train), replace=False)
    perturbation_rows = []
    success = defaultdict(int)
    for i in sample_idx:
        mol = train_mols[int(i)]
        base = phi_from_moments(
            aiom_moments(mol, C_V, C_E, node_index, bond_index, T_MAX), triu
        )
        constructions = {
            "atom_sub": lambda: perturb_atom_substitution(prng, mol, atom_cats),
            "bond_type_swap": lambda: perturb_bond_type_swap(prng, mol),
            "two_switch": lambda: two_switch(prng, mol),
        }
        for kind, fn in constructions.items():
            pm = fn()
            if pm is None:
                continue
            success[kind] += 1
            other = phi_from_moments(
                aiom_moments(pm, C_V, C_E, node_index, bond_index, T_MAX), triu
            )
            for T in T_MAIN:
                perturbation_rows.append({
                    "kind": kind, "T": T, "graph": int(i),
                    "delta": relative_delta(base, other, T, dim),
                })
        # permutation negative control
        pa = prng.permutation(mol.n)
        pb = prng.permutation(mol.m) if mol.m > 0 else np.zeros(0, dtype=np.int64)
        pm = permute_mol(mol, pa, pb)
        other = phi_from_moments(
            aiom_moments(pm, C_V, C_E, node_index, bond_index, T_MAX), triu
        )
        for T in T_MAIN:
            perturbation_rows.append({"kind": "permutation", "T": T,
                                      "graph": int(i),
                                      "delta": relative_delta(base, other, T, dim)})
        # random size-matched pair
        cand = np.where(np.abs(size_all[:n_train] - mol.n) <= 1)[0]
        cand = cand[cand != int(i)]
        if cand.size:
            j = int(cand[prng.integers(0, cand.size)])
            other_mol = train_mols[j]
            other = phi_from_moments(
                aiom_moments(other_mol, C_V, C_E, node_index, bond_index, T_MAX), triu
            )
            for T in T_MAIN:
                perturbation_rows.append({"kind": "random_pair", "T": T,
                                          "graph": int(i),
                                          "delta": relative_delta(base, other, T, dim)})
    log(f"perturbation successes: {dict(success)}")
    with (out / "perturbation_metrics.csv").open("w", encoding="utf-8") as fh:
        fh.write("kind,T,graph,delta\n")
        for r in perturbation_rows:
            fh.write(f"{r['kind']},{r['T']},{r['graph']},{r['delta']:.8f}\n")
    # per kind/T summary
    pert_summary = []
    for kind in ("permutation", "atom_sub", "bond_type_swap", "two_switch", "random_pair"):
        for T in T_MAIN:
            vals = np.array([r["delta"] for r in perturbation_rows
                             if r["kind"] == kind and r["T"] == T])
            if vals.size == 0:
                continue
            pert_summary.append({
                "kind": kind, "T": T, "n": int(vals.size),
                "mean": float(vals.mean()), "median": float(np.median(vals)),
                "p5": float(np.percentile(vals, 5)), "p95": float(np.percentile(vals, 95)),
                "max": float(vals.max()),
            })
    write_json(out / "perturbation_summary.json", pert_summary)

    # --- Test E spearman between d_phi and d_wl on random pairs ---
    from scipy.stats import spearmanr
    spear_rows = []
    for T in T_NN:
        mu, sd = fit_standardizer(phi_train[:, : (T + 1) * dim])
        Z = apply_standardizer(phi_all[:, : (T + 1) * dim], mu, sd)
        sub = rand_pairs[:5000]
        dphi = np.linalg.norm(Z[sub[:, 0]] - Z[sub[:, 1]], axis=1)
        dwl = 1.0 - wl_sim[sub[:, 0], sub[:, 1]]
        rho = float(spearmanr(dphi, dwl).correlation)
        spear_rows.append({"T": T, "spearman_dphi_dwl": rho, "n_pairs": int(len(sub))})
        log(f"[Test E] T={T} spearman(d_phi,d_wl)={rho:.3f}")
    write_json(out / "nn_wl_spearman.json", spear_rows)

    # --- Test F probes ---
    probe_rows = []
    if not args.skip_probe:
        load_zinc, _ = repo_loader()
        train_ds = load_zinc(args.data_root, "train")
        valid_ds = load_zinc(args.data_root, "val")
        y_train = np.array([float(train_ds[i].y.reshape(-1)[0])
                            for i in range(phi_train.shape[0])], dtype=np.float64)
        y_valid = np.array([float(valid_ds[i].y.reshape(-1)[0])
                            for i in range(phi_valid.shape[0])], dtype=np.float64)
        log(f"target y: train mean={y_train.mean():.4f} std={y_train.std():.4f} "
            f"valid mean={y_valid.mean():.4f}")
        for T in T_MAIN:
            mu, sd = fit_standardizer(phi_train[:, : (T + 1) * dim])
            Ztr = apply_standardizer(phi_train[:, : (T + 1) * dim], mu, sd)
            Zva = apply_standardizer(phi_valid[:, : (T + 1) * dim], mu, sd)
            ridge = linear_ridge(Ztr, y_train, Zva, y_valid)
            lin = train_probe(Ztr, y_train, Zva, y_valid, hidden=None, seed=0, log=log)
            mlp = train_probe(Ztr, y_train, Zva, y_valid, hidden=64, seed=0, log=log)
            for name, res in (("linear", lin), ("mlp", mlp), ("ridge", ridge)):
                row = {"T": T, "probe": name, "seed": 0, "dim": (T + 1) * dim, **res}
                probe_rows.append(row)
            log(f"[Test F] T={T} linear tr/va={lin['train_mae']:.4f}/{lin['valid_mae']:.4f} "
                f"mlp tr/va={mlp['train_mae']:.4f}/{mlp['valid_mae']:.4f} "
                f"ridge va={ridge['valid_mae']:.4f}")
        # seed stability on two Ts
        for T in (2, 8):
            mu, sd = fit_standardizer(phi_train[:, : (T + 1) * dim])
            Ztr = apply_standardizer(phi_train[:, : (T + 1) * dim], mu, sd)
            Zva = apply_standardizer(phi_valid[:, : (T + 1) * dim], mu, sd)
            for seed in (1, 2):
                mlp = train_probe(Ztr, y_train, Zva, y_valid, hidden=64, seed=seed, log=log)
                probe_rows.append({"T": T, "probe": "mlp", "seed": seed,
                                   "dim": (T + 1) * dim, **mlp})
        write_json(out / "probe_results.json", probe_rows)

    # plots
    try:
        save_plots(out / "plots", [r for r in collision_rows if r["decimals"] == 10],
                   rank_rows, perturbation_rows, nn_rows, probe_rows, log=log)
    except Exception as exc:  # pragma: no cover
        log(f"[plots] failed: {exc!r}")

    summary = {
        "provenance": provenance,
        "metadata": metadata,
        "invariance": inv,
        "collision": [r for r in collision_rows if r["decimals"] == 10],
        "rank": rank_rows,
        "nn": nn_rows,
        "probe": probe_rows,
        "wall_seconds": time.time() - t0,
    }
    write_json(out / "SUMMARY.json", summary)
    log(f"=== done in {time.time() - t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
