"""Bond random-walk structural organization (cscl-bond-random-walk-organization-v1).

Track: ``tracks/structure-contribution`` · protocol:
``configs/bond_random_walk_organization_v1.yaml``.

Part A (label-blind structure audit)
------------------------------------
* physical bonds come from :func:`cscl_units.unique_undirected_bonds` (one
  record per real undirected bond, PyG double storage removed);
* the **bond line graph** ``A_L`` has one node per physical bond and an edge
  between two distinct bonds that share at least one atom;
* the **lazy random walk** is ``P = (I + D^{-1} A_L) / 2`` with the row rule
  ``P[i] = e_i`` for zero-degree nodes, so every row sums to exactly 1;
* ``P^1..P^4`` are computed by deterministic matrix multiplication (float64,
  no sampled paths); the frozen model relation is
  ``K = (P + P^2 + P^3) / 3`` (symmetrised in the model, see below);
* communities are computed on the same line graph with a self-contained
  deterministic Walktrap implementation (:func:`walktrap_partition`), the
  networkx greedy-modularity reference, and the TASU-v1 2ECC/bridge partition
  mapped to bond labels.

Part B (one-shot Bond-Bond relation composer)
---------------------------------------------
:class:`BondRWComposer` keeps the frozen BJOC Atom/Bond encoders and the BJOC
Composer/Reader shapes, removes the learned Junction encoder entirely, and adds
one shared Pair MLP::

    phi(b_e, b_f) = MLP([b_e + b_f, |b_e - b_f|, b_e * b_f])
    r_e           = sum_f K_ef phi(b_e, b_f)
    composer      = [b_e, r_e, 0, 0, raw bond] -> 160 -> 96
    reader        = [z, sum_a, sum_a^2, global33] -> 256 -> 256 -> 1

``phi`` is exactly symmetric, so ``sum_f K_ef phi_ef == sum_f Ks_ef phi_ef``
with ``Ks = (K + K^T)/2``.  The stored pair list is the unordered ``e <= f``
half of ``Ks`` with the diagonal weight halved, which reproduces the exact
double-sum in the ``index_add_`` accumulation (unit-tested).  ``K`` itself is
therefore used as specified; nothing is dropped.

The two GPU arms differ only in the pair weights handed to the model:

* ``R`` structured: real ``Ks`` of ``K = (P + P^2 + P^3)/3``;
* ``U`` unstructured: ``K_ef = 1 / N_bond``.

Discipline: no GNN/attention/iteration; no learned Junction encoder; ``y`` is
never read by any function in this module except the supervised training loop
(which receives Fit raw ``y`` from the driver).
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402
import task_adaptive_structure_units_v1 as tasu  # noqa: E402

PROTOCOL_ID = "cscl-bond-random-walk-organization-v1"

#: frozen walk constants (no search after the freeze)
MAX_K = 4
KERNEL_KS = (1, 2, 3)
WALKTRAP_STEPS = 4
SELF_LOOP_P = 0.5

#: frozen model widths
PAIR_HIDDEN = 128
EXPECTED_PARAMETERS = 349_985  # atom 26336 + bond 62816 + pair 49376 + composer 62976 + reader 148481

FORBIDDEN_TOKENS = (
    "MessagePassing",
    "GCN",
    "GIN",
    "GAT",
    "SAGEConv",
    "Transformer",
    "MultiheadAttention",
    "self_attention",
    "TransformerEncoder",
    "GRU",
    "LSTM",
    "RNN",
    "GraphConv",
    "ChebConv",
    "TAGConv",
)


# ===========================================================================
# Part A — line graph, lazy walk, kernels
# ===========================================================================


def line_graph_adjacency(bond_index: Sequence[Sequence[int]]) -> np.ndarray:
    """0/1 adjacency of the physical-bond line graph (float64).

    Two distinct bonds are adjacent iff they share at least one atom.  The
    function reads only bond endpoint ids (label-blind).
    """
    bi = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    n = bi.shape[0]
    A = np.zeros((n, n), dtype=np.float64)
    if n == 0:
        return A
    incident: dict[int, list[int]] = {}
    for b, (u, v) in enumerate(bi):
        incident.setdefault(int(u), []).append(b)
        incident.setdefault(int(v), []).append(b)
    for _, bonds in incident.items():
        for i in range(len(bonds)):
            for j in range(i + 1, len(bonds)):
                a, c = bonds[i], bonds[j]
                A[a, c] = 1.0
                A[c, a] = 1.0
    return A


def lazy_transition(A: np.ndarray) -> np.ndarray:
    """``P = (I + D^{-1} A) / 2`` with zero-degree rows set to one-hot."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    deg = A.sum(axis=1)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if deg[i] > 1e-12:
            P[i] = SELF_LOOP_P * (np.eye(1, n, i)[0]) + (1.0 - SELF_LOOP_P) * A[i] / deg[i]
        else:
            P[i, i] = 1.0
    return P


def kernel_powers(A: np.ndarray, max_k: int = MAX_K) -> dict[int, np.ndarray]:
    """Deterministic ``P^k`` for k = 1..max_k (float64 matrix products)."""
    P = lazy_transition(A)
    out: dict[int, np.ndarray] = {}
    cur = P.copy()
    for k in range(1, int(max_k) + 1):
        if k > 1:
            cur = cur @ P
        out[k] = cur.copy()
    return out


def relation_kernel(powers: Mapping[int, np.ndarray], ks: Sequence[int] = KERNEL_KS) -> np.ndarray:
    """``K = mean_k P^k`` over the frozen kernel scales."""
    ks = [int(k) for k in ks]
    if not ks:
        raise ValueError("empty kernel scales")
    K = np.zeros_like(powers[ks[0]], dtype=np.float64)
    for k in ks:
        K = K + powers[k]
    return K / float(len(ks))


def symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T)


def unordered_triplets(Ks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unordered ``e <= f`` triplets of a symmetric kernel.

    Weights are stored exactly as ``Ks``; the diagonal must receive a single
    contribution while off-diagonal entries are added to both endpoints, which
    the model implements by halving the weight when ``src == dst``.
    """
    Ks = np.asarray(Ks, dtype=np.float64)
    n = Ks.shape[0]
    src, dst, w = [], [], []
    for e in range(n):
        for f in range(e, n):
            src.append(e)
            dst.append(f)
            w.append(float(Ks[e, f]))
    return (
        np.asarray(src, dtype=np.int32),
        np.asarray(dst, dtype=np.int32),
        np.asarray(w, dtype=np.float32),
    )


def uniform_triplets(n_bonds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unordered triplets of ``K_ef = 1 / n_bonds`` (the U control)."""
    n = int(n_bonds)
    src, dst = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    keep = src <= dst
    s = src[keep].astype(np.int32)
    d = dst[keep].astype(np.int32)
    w = np.full(s.shape[0], 1.0 / max(n, 1), dtype=np.float32)
    return s, d, w


def kernel_triplets_from_objects(obj: sro.MolObjects) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R triplet cache for one molecule (label-blind)."""
    A = line_graph_adjacency(obj.bond_index)
    powers = kernel_powers(A, MAX_K)
    K = relation_kernel(powers)
    return unordered_triplets(symmetrize(K))


# ---------------------------------------------------------------------------
# Part A — descriptive statistics
# ---------------------------------------------------------------------------


def _tv_rows_to_uniform(M: np.ndarray) -> float:
    n = M.shape[0]
    if n == 0:
        return 0.0
    return float(np.mean(0.5 * np.abs(M - 1.0 / n).sum(axis=1)))


def _tv_rows_to_stationary(M: np.ndarray, A: np.ndarray) -> float:
    n = M.shape[0]
    if n == 0:
        return 0.0
    deg = A.sum(axis=1)
    if deg.sum() <= 0:
        return 0.0
    pi = deg / deg.sum()
    return float(np.mean(0.5 * np.abs(M - pi[None, :]).sum(axis=1)))


def _row_entropy_norm(M: np.ndarray) -> float:
    n = M.shape[0]
    if n <= 1:
        return 0.0
    p = np.clip(M, 1e-300, None)
    return float(np.mean(-(p * np.log(p)).sum(axis=1)) / math.log(n))


def _reachable_fraction(M: np.ndarray, tol: float = 1e-12) -> float:
    n = M.shape[0]
    if n == 0:
        return 0.0
    return float(np.mean((M > tol).sum(axis=1) / n))


def _effective_rank(S: np.ndarray) -> float:
    vals = np.linalg.svd(S, compute_uv=False)
    denom = float((vals**2).sum())
    if denom <= 0:
        return 0.0
    participation = float((vals**2).sum() ** 2 / float((vals**4).sum()))
    return participation


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = float((a * a).sum()), float((b * b).sum())
    if aa <= 0 or bb <= 0:
        return 0.0
    return float((a * b).sum() / math.sqrt(aa * bb))


def molecule_walk_stats(obj: sro.MolObjects) -> dict[str, Any]:
    """Label-blind per-molecule random-walk / line-graph statistics."""
    A = line_graph_adjacency(obj.bond_index)
    n = A.shape[0]
    powers = kernel_powers(A, MAX_K)
    K = relation_kernel(powers)
    U = np.full((n, n), 1.0 / max(n, 1))
    stats: dict[str, Any] = {
        "mol_index": int(obj.mol_index),
        "n_atoms": int(obj.n_atoms),
        "n_bonds": int(n),
        "n_junc": int(obj.junc_index.shape[0]),
        "line_graph_edges": float(A.sum() / 2.0),
        "isolated_line_nodes": int((A.sum(axis=1) <= 0).sum()),
    }
    for k in range(1, MAX_K + 1):
        M = powers[k]
        stats[f"tv_uniform_k{k}"] = _tv_rows_to_uniform(M)
        stats[f"tv_stationary_k{k}"] = _tv_rows_to_stationary(M, A)
        stats[f"entropy_norm_k{k}"] = _row_entropy_norm(M)
        stats[f"reachable_k{k}"] = _reachable_fraction(M)
        stats[f"self_prob_k{k}"] = float(np.mean(np.diag(M)))
    d1 = (powers[1] - U).reshape(-1)
    d2 = (powers[2] - U).reshape(-1)
    d3 = (powers[3] - U).reshape(-1)
    stats["increment_cos_12"] = _cosine(d1, d2)
    stats["increment_cos_23"] = _cosine(d2, d3)
    stats["scale_distinctness_p1p3"] = float(
        np.mean(0.5 * np.abs(powers[3] - powers[1]).sum(axis=1))
    )
    stats["scale_distinctness_raw_p1p3"] = float(
        np.abs(powers[3] - powers[1]).sum() / (2.0 * max(n * n, 1))
    )
    stats["K_tv_uniform"] = _tv_rows_to_uniform(K)
    stats["K_effective_rank"] = _effective_rank(K)
    stats["K_min"] = float(K.min())
    stats["K_max"] = float(K.max())
    row_cos = []
    for e in range(n):
        row_cos.append(_cosine(powers[2][e], powers[3][e]))
    stats["row_cos_p2_p3"] = float(np.mean(row_cos)) if row_cos else 0.0
    return stats


def _molecule_category(obj: sro.MolObjects) -> str:
    """Coarse label-free topology class from the frozen global features."""
    names = list(sro.GLOBAL_FEAT_NAMES)
    gf = {name: float(obj.global_feat[i]) for i, name in enumerate(names)}
    n_ring_systems = gf["n_ring_systems"]
    max_degree = gf["max_degree"]
    ring_bond_fraction = gf["ring_bond_fraction"]
    if n_ring_systems < 0.5:
        return "acyclic_chain" if max_degree <= 2.0 else "acyclic_branch"
    if ring_bond_fraction >= 0.999:
        return "ring_only"
    return "ring_chain"


def _bond_class(obj: sro.MolObjects, e: int) -> str:
    ring = "ring" if obj.bond_feat[e, sro.BOND_CATEGORIES] > 0.5 else "nonring"
    return f"{ring}_type{int(obj.bond_type[e])}"


def atom_relabel_invariance(obj: sro.MolObjects, rng: np.random.Generator) -> float:
    """Max abs kernel/triplet difference after an atom relabel (bitwise expected)."""
    n = obj.n_atoms
    perm = rng.permutation(n)
    bi = np.asarray(obj.bond_index, dtype=np.int64)
    bi2 = perm[bi]  # new ids; bond slot order unchanged
    A1 = line_graph_adjacency(bi)
    A2 = line_graph_adjacency(bi2)
    K1 = relation_kernel(kernel_powers(A1, MAX_K))
    K2 = relation_kernel(kernel_powers(A2, MAX_K))
    return float(np.abs(K1 - K2).max())


def bond_permutation_equivariance(obj: sro.MolObjects, rng: np.random.Generator) -> dict[str, float]:
    """Permute bond slots; K must transform by conjugation, statistics invariant."""
    n = obj.bond_index.shape[0]
    perm = rng.permutation(n)
    A = line_graph_adjacency(obj.bond_index)
    A2 = A[np.ix_(perm, perm)]
    K = relation_kernel(kernel_powers(A, MAX_K))
    K2 = relation_kernel(kernel_powers(A2, MAX_K))
    Kc = K[np.ix_(perm, perm)]
    stats1 = molecule_walk_stats(obj)
    obj2 = sro.MolObjects(
        mol_index=obj.mol_index,
        smiles=obj.smiles,
        n_atoms=obj.n_atoms,
        atom_feat=obj.atom_feat,
        bond_index=obj.bond_index[perm],
        bond_type=obj.bond_type[perm],
        bond_feat=obj.bond_feat[perm],
        junc_index=obj.junc_index,
        junc_bond=obj.junc_bond,
        junc_feat=obj.junc_feat,
        global_feat=obj.global_feat,
    )
    stats2 = molecule_walk_stats(obj2)
    keep = [k for k in stats1 if k not in ("mol_index", "n_atoms", "n_junc")]
    max_stat = max(abs(float(stats1[k]) - float(stats2[k])) for k in keep)
    return {
        "kernel_conjugation_max_abs": float(np.abs(K2 - Kc).max()),
        "statistic_max_abs": max_stat,
    }


# ---------------------------------------------------------------------------
# Part A — Walktrap (Pons & Latapy 2005), deterministic
# ---------------------------------------------------------------------------


def walktrap_partition(A: np.ndarray, steps: int = WALKTRAP_STEPS) -> dict[str, Any]:
    """Exact deterministic Walktrap on a small 0/1 graph.

    Communities are sets of nodes; at every merge the community transition row
    is the exact mean of the member rows of ``P0^steps`` (``P0 = D^{-1}A`` with
    zero-degree rows one-hot).  The merge minimising
    ``(n1 n2 / (n1 + n2)) r^2(C1,C2)`` is applied; the returned partition is the
    one maximising modularity along the deterministic merge path (first
    occurrence on ties).
    """
    n = A.shape[0]
    if n == 0:
        return {"labels": np.zeros(0, dtype=np.int64), "n_communities": 0, "modularity": 0.0, "merge_path": []}
    deg = A.sum(axis=1)
    P0 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if deg[i] > 0:
            P0[i] = A[i] / deg[i]
        else:
            P0[i, i] = 1.0
    Pt = P0.copy()
    for _ in range(1, int(steps)):
        Pt = Pt @ P0
    labels = np.arange(n, dtype=np.int64)
    groups = [[i] for i in range(n)]
    rows = [Pt[i].copy() for i in range(n)]

    def modularity_of(labels_now: np.ndarray) -> float:
        import networkx as nx

        if A.sum() <= 0:
            return 0.0
        G = nx.from_numpy_array(A)
        parts = [set(np.where(labels_now == c)[0].tolist()) for c in np.unique(labels_now)]
        return float(nx.community.modularity(G, parts)) if parts else 0.0

    best_mod = modularity_of(labels)
    best_labels = labels.copy()
    path = [{"n_communities": int(len(groups)), "modularity": best_mod}]
    while len(groups) > 1:
        k = len(groups)
        dist = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(i + 1, k):
                diff = rows[i] - rows[j]
                with np.errstate(divide="ignore", invalid="ignore"):
                    r2 = float(np.sum((diff**2) / np.where(deg > 0, deg, 1.0)))
                dist[i, j] = dist[j, i] = (len(groups[i]) * len(groups[j]) / (len(groups[i]) + len(groups[j]))) * r2
        np.fill_diagonal(dist, np.inf)
        flat = int(np.argmin(dist))
        i, j = divmod(flat, k)
        if i > j:
            i, j = j, i
        merged = groups[i] + groups[j]
        new_row = (len(groups[i]) * rows[i] + len(groups[j]) * rows[j]) / len(merged)
        keep = [g for g in range(k) if g not in (i, j)]
        groups = [groups[g] for g in keep] + [merged]
        rows = [rows[g] for g in keep] + [new_row]
        labels = np.zeros(n, dtype=np.int64)
        for c, members in enumerate(groups):
            for node in members:
                labels[node] = c
        mod = modularity_of(labels)
        path.append({"n_communities": int(len(groups)), "modularity": mod})
        if mod > best_mod + 1e-12:
            best_mod, best_labels = mod, labels.copy()
    return {
        "labels": best_labels,
        "n_communities": int(len(np.unique(best_labels))),
        "modularity": float(best_mod),
        "merge_path": path,
        "steps": int(steps),
    }


def tasu_bond_labels(obj: sro.MolObjects) -> np.ndarray:
    """TASU-v1 2ECC/bridge partition mapped to bond labels.

    A bond inside one 2-edge-connected block gets that block id; every bridge
    gets its own unique label.  Labels are canonicalised by first appearance.
    """
    bonds = [(int(u), int(v), int(bt)) for (u, v), bt in zip(obj.bond_index, obj.bond_type)]
    topo = tasu.build_structure_topology(int(obj.mol_index), bonds)
    labels = -np.ones(obj.bond_index.shape[0], dtype=np.int64)
    bridge_key = {(min(int(u), int(v)), max(int(u), int(v))): k for k, (u, v, _) in enumerate(bonds)}
    bridge_bonds = set(int(b) for b in topo.bridge_bond)
    next_label = 0
    for b, (u, v, _) in enumerate(bonds):
        if b in bridge_bonds:
            labels[b] = next_label
            next_label += 1
        else:
            labels[b] = int(topo.block_of_atom[int(u)])
    canon: dict[int, int] = {}
    out = np.zeros_like(labels)
    for b, val in enumerate(labels):
        if int(val) not in canon:
            canon[int(val)] = len(canon)
        out[b] = canon[int(val)]
    return out


def _ari(a: Sequence[int], b: Sequence[int]) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))


def community_analysis_molecule(
    obj: sro.MolObjects,
    *,
    relabel_trials: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Community partitions and their descriptors for one molecule."""
    import networkx as nx

    A = line_graph_adjacency(obj.bond_index)
    n = A.shape[0]
    G = nx.from_numpy_array(A)
    wt = walktrap_partition(A, steps=WALKTRAP_STEPS)
    gm = nx.community.greedy_modularity_communities(G) if n else []
    cc = list(nx.connected_components(G)) if n else []
    tasu_lab = tasu_bond_labels(obj)

    wt_lab = wt["labels"]
    gm_lab = _lab_from_parts(gm, n)
    cc_lab = _lab_from_parts(cc, n)

    # atom boundary sharing: how many atoms have incident bonds in >1 community
    incident: dict[int, list[int]] = {}
    for b, (u, v) in enumerate(obj.bond_index):
        incident.setdefault(int(u), []).append(b)
        incident.setdefault(int(v), []).append(b)
    multi = boundary = 0
    for _, bonds in incident.items():
        if len(bonds) > 1:
            multi += 1
            if len({int(wt_lab[b]) for b in bonds}) > 1:
                boundary += 1

    # community descriptors for the Walktrap partition
    n_comms = int(len(np.unique(wt_lab))) if n else 0
    sizes = [int((wt_lab == c).sum()) for c in range(n_comms)]
    single = sum(1 for c in range(n_comms) if (wt_lab == c).sum() == 1)
    ring_like = 0
    connected = 0
    for c in range(n_comms):
        members = np.where(wt_lab == c)[0]
        sub = G.subgraph(members.tolist())
        if nx.is_connected(sub) if len(members) > 1 else True:
            connected += 1
        if all(obj.bond_feat[b, sro.BOND_CATEGORIES] > 0.5 for b in members):
            ring_like += 1

    # relabel stability: permute line-graph node order, re-run Walktrap
    rng = np.random.default_rng(seed + int(obj.mol_index))
    relabel_aris = []
    for _ in range(int(relabel_trials)):
        perm = rng.permutation(n)
        wt2 = walktrap_partition(A[np.ix_(perm, perm)], steps=WALKTRAP_STEPS)
        relabel_aris.append(_ari(wt_lab, wt2["labels"][np.argsort(perm)]))
    return {
        "mol_index": int(obj.mol_index),
        "n_bonds": int(n),
        "category": _molecule_category(obj),
        "walktrap": {
            "n_communities": int(wt["n_communities"]),
            "sizes": sizes,
            "modularity": float(wt["modularity"]),
            "singleton_communities": int(single),
            "connected_communities": int(connected),
            "ringlike_communities": int(ring_like),
            "relabel_ari_min": float(min(relabel_aris)) if relabel_aris else 1.0,
            "relabel_aris": [float(a) for a in relabel_aris],
            "partition_covers_all_bonds": bool(
                n == 0 or (len(wt_lab) == n and int(wt_lab.min()) >= 0 and int(np.max(wt_lab)) + 1 == int(len(np.unique(wt_lab))))
            ),
            "unique_bond_membership": bool(n == 0 or sum(sizes) == n),
        },
        "greedy_modularity": {
            "n_communities": int(len(gm)),
            "sizes": sorted(int(len(c)) for c in gm),
            "modularity": float(nx.community.modularity(G, gm)) if (n and G.number_of_edges() > 0) else 0.0,
        },
        "connected_components": {"n_communities": int(len(cc)), "sizes": sorted(int(len(c)) for c in cc)},
        "tasu_2ecc_bridge": {
            "n_communities": int(len(np.unique(tasu_lab))) if n else 0,
            "n_bridges": int(0 if n == 0 else _count_bridges(obj)),
        },
        "ari": {
            "walktrap_vs_tasu": _ari(wt_lab, tasu_lab),
            "walktrap_vs_greedy": _ari(wt_lab, gm_lab),
            "walktrap_vs_cc": _ari(wt_lab, cc_lab),
            "greedy_vs_tasu": _ari(gm_lab, tasu_lab),
        },
        "atom_boundary": {
            "atoms_with_multiple_incident_bonds": int(multi),
            "boundary_atoms": int(boundary),
        },
    }


def _lab_from_parts(parts, n: int) -> np.ndarray:
    out = -np.ones(n, dtype=np.int64)
    for c, members in enumerate(parts):
        for node in members:
            out[int(node)] = c
    return out


def summarize_walk_stats(per_mol: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-molecule Part-A records + the pre-frozen gate."""
    keys = [
        "tv_uniform_k1", "tv_uniform_k2", "tv_uniform_k3", "tv_uniform_k4",
        "tv_stationary_k1", "tv_stationary_k3",
        "entropy_norm_k1", "entropy_norm_k3",
        "reachable_k1", "reachable_k3",
        "increment_cos_12", "increment_cos_23",
        "scale_distinctness_p1p3", "scale_distinctness_raw_p1p3", "K_tv_uniform", "K_effective_rank", "row_cos_p2_p3",
    ]
    agg = {k: float(np.mean([float(m[k]) for m in per_mol])) for k in keys}
    by_cat: dict[str, Any] = {}
    cats = sorted({str(m["category"]) for m in per_mol})
    for cat in cats:
        sel = [m for m in per_mol if str(m["category"]) == cat]
        by_cat[cat] = {
            "n_molecules": len(sel),
            "mean_n_bonds": float(np.mean([m["n_bonds"] for m in sel])),
            "mean_tv_uniform_k3": float(np.mean([m["tv_uniform_k3"] for m in sel])),
            "mean_K_tv_uniform": float(np.mean([m["K_tv_uniform"] for m in sel])),
            "mean_isolated": float(np.mean([m["isolated_line_nodes"] for m in sel])),
        }
    gate = {
        "mean_tv_p3_ge_min": bool(agg["tv_uniform_k3"] >= 0.03),
        "mean_tv_K_ge_min": bool(agg["K_tv_uniform"] >= 0.05),
        "scale_distinctness_ge_min": bool(agg["scale_distinctness_p1p3"] >= 0.05),
    }
    return {"aggregate": agg, "by_category": by_cat, "gate_conditions": gate}


def relation_reassignment_witness(
    model: BondRWComposer,
    objs: Sequence[sro.MolObjects],
    triplets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device | str,
    *,
    n_molecules: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """C2: fixed representations/kernels; reassign K and measure r / prediction deltas.

    Keeps the trained encoders and Pair MLP frozen; for each molecule the bond
    representation ``b`` and the Pair MLP outputs are computed once, then the
    relation weights are replaced by (i) the uniform control, (ii) a random
    row-stochastic reassignment, (iii) a random bond-slot permutation of the
    real kernel.  The per-bond marginal set of ``b`` never changes.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(objs), size=min(int(n_molecules), len(objs)), replace=False)
    dev = torch.device(device)
    deltas = {"uniform": [], "row_perm": [], "bond_perm": [], "zero": []}
    pred_deltas = {"uniform": [], "row_perm": [], "bond_perm": [], "zero": []}
    with torch.no_grad():
        for i in idx:
            obj = objs[int(i)]
            src, dst, w = triplets[int(i)]
            batch = move_batch(batch_rw_objects([obj], [(src, dst, w)]), dev)
            pred0, feats = model(batch, return_features=True)
            b = feats["b"]
            b0 = pred0.detach().cpu().item()
            bs, bd = b[src], b[dst]
            phi = model.pair_mlp(torch.cat([bs + bd, (bs - bd).abs(), bs * bd], dim=1))
            n = b.shape[0]
            variants = {}
            wu = np.full_like(w, 1.0 / n)
            variants["uniform"] = (torch.as_tensor(src, dtype=torch.long, device=dev),
                                   torch.as_tensor(dst, dtype=torch.long, device=dev), wu)
            perm = rng.permutation(n)
            variants["row_perm"] = (
                torch.as_tensor(src, dtype=torch.long, device=dev),
                torch.as_tensor(perm[dst], dtype=torch.long, device=dev),
                w,
            )
            variants["zero"] = (torch.as_tensor(src, dtype=torch.long, device=dev),
                                torch.as_tensor(dst, dtype=torch.long, device=dev),
                                np.zeros_like(w))
            bp_perm = rng.permutation(n)
            variants["bond_perm"] = (
                torch.as_tensor(bp_perm[src], dtype=torch.long, device=dev),
                torch.as_tensor(bp_perm[dst], dtype=torch.long, device=dev),
                w,
            )
            for name, (s2, d2, w2) in variants.items():
                w_t = torch.as_tensor(w2, dtype=torch.float32, device=dev)
                w_eff = torch.where(s2 == d2, w_t * 0.5, w_t).unsqueeze(1)
                r2 = torch.zeros_like(b)
                r2.index_add_(0, s2, phi * w_eff)
                r2.index_add_(0, d2, phi * w_eff)
                deltas[name].append(float((r2 - feats["r"]).abs().mean()))
                zero = torch.zeros_like(feats["r"])
                zero_scalar = torch.zeros((n, 1), device=dev, dtype=b.dtype)
                compose_in = torch.cat([b, r2, zero, zero_scalar, batch["bond_feat"]], dim=1)
                c2 = model.group_composer(compose_in)
                z2 = torch.zeros((1, c2.shape[1]), device=dev, dtype=c2.dtype)
                z2.index_add_(0, batch["bond_batch"], c2)
                a_sum = torch.zeros((1, feats["a"].shape[1]), device=dev, dtype=feats["a"].dtype)
                a_sum.index_add_(0, batch["atom_batch"], feats["a"])
                a_sq = torch.zeros_like(a_sum)
                a_sq.index_add_(0, batch["atom_batch"], feats["a"] * feats["a"])
                g = (batch["global_feat"].to(feats["a"].dtype) - model.global_mean) / model.global_scale
                reader_in2 = torch.cat([z2, a_sum, a_sq, g], dim=1)
                pred2 = model.reader(reader_in2).item()
                pred_deltas[name].append(abs(pred2 - b0))
    return {
        "n_molecules": int(len(idx)),
        "mean_r_delta": {k: float(np.mean(v)) for k, v in deltas.items()},
        "mean_pred_delta": {k: float(np.mean(v)) for k, v in pred_deltas.items()},
        "max_pred_delta": {k: float(np.max(v)) for k, v in pred_deltas.items()},
        "note": "fixed b and frozen phi; only the relation weights are reassigned",
    }


def _count_bridges(obj: sro.MolObjects) -> int:
    bonds = [(int(u), int(v), int(bt)) for (u, v), bt in zip(obj.bond_index, obj.bond_type)]
    topo = tasu.build_structure_topology(int(obj.mol_index), bonds)
    return int(len(topo.bridge_bond))


def summarize_communities(per_mol: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate community records (descriptive only; no chemistry claims)."""
    wt_n = [m["walktrap"]["n_communities"] for m in per_mol]
    wt_sizes = [s for m in per_mol for s in m["walktrap"]["sizes"]]
    tasu_n = [m["tasu_2ecc_bridge"]["n_communities"] for m in per_mol]
    ari_keys = list(per_mol[0]["ari"].keys()) if per_mol else []
    connected = sum(m["walktrap"]["connected_communities"] for m in per_mol)
    total_comms = sum(m["walktrap"]["n_communities"] for m in per_mol)
    multi_atoms = sum(m["atom_boundary"]["atoms_with_multiple_incident_bonds"] for m in per_mol)
    boundary_atoms = sum(m["atom_boundary"]["boundary_atoms"] for m in per_mol)
    relabel_all = [a for m in per_mol for a in m["walktrap"].get("relabel_aris", [])]
    return {
        "n_molecules": len(per_mol),
        "walktrap": {
            "mean_communities": float(np.mean(wt_n)) if wt_n else 0.0,
            "median_communities": float(np.median(wt_n)) if wt_n else 0.0,
            "size_histogram": {str(s): int(wt_sizes.count(s)) for s in sorted(set(wt_sizes))},
            "singleton_fraction": float(
                sum(m["walktrap"]["singleton_communities"] for m in per_mol) / max(total_comms, 1)
            ),
            "connected_fraction": float(connected / max(total_comms, 1)),
            "ringlike_fraction": float(
                sum(m["walktrap"]["ringlike_communities"] for m in per_mol) / max(total_comms, 1)
            ),
            "relabel_ari_min": float(min(m["walktrap"]["relabel_ari_min"] for m in per_mol)),
            "relabel_ari_mean": float(np.mean(relabel_all)) if relabel_all else 1.0,
            "relabel_ari_median": float(np.median(relabel_all)) if relabel_all else 1.0,
            "relabel_ari_frac_ge_0p9": float(np.mean([a >= 0.9 for a in relabel_all])) if relabel_all else 1.0,
            "all_partitions_cover_bonds": bool(all(m["walktrap"]["partition_covers_all_bonds"] for m in per_mol)),
            "all_bonds_unique_membership": bool(all(m["walktrap"]["unique_bond_membership"] for m in per_mol)),
        },
        "mean_communities": {
            "walktrap": float(np.mean(wt_n)) if wt_n else 0.0,
            "tasu_2ecc_bridge": float(np.mean(tasu_n)) if tasu_n else 0.0,
            "greedy_modularity": float(np.mean([m["greedy_modularity"]["n_communities"] for m in per_mol])),
            "connected_components": float(np.mean([m["connected_components"]["n_communities"] for m in per_mol])),
        },
        "mean_ari": {k: float(np.mean([m["ari"][k] for m in per_mol])) for k in ari_keys},
        "atom_boundary": {
            "atoms_with_multiple_incident_bonds": int(multi_atoms),
            "boundary_atoms": int(boundary_atoms),
            "boundary_fraction": float(boundary_atoms / max(multi_atoms, 1)),
        },
    }


# ===========================================================================
# Part B — model
# ===========================================================================


class BondRWConfig:
    """Frozen widths: BJOC encoders/composer/reader + one Pair MLP."""

    atom_hidden = 128
    atom_out = 96
    bond_hidden = 160
    bond_out = 96
    pair_hidden = PAIR_HIDDEN
    pair_out = 96
    compose_hidden = 160
    compose_out = 96
    reader_hidden = 256
    reader_layers = 2
    dropout = 0.0

    @property
    def atom_in(self) -> int:
        return sro.ATOM_FEAT_DIM

    @property
    def bond_in(self) -> int:
        return 3 * self.bond_out + sro.BOND_FEAT_DIM

    @property
    def pair_in(self) -> int:
        return 3 * self.bond_out

    @property
    def compose_in(self) -> int:
        return self.bond_out + 2 * self.compose_out + 1 + sro.BOND_FEAT_DIM

    @property
    def reader_in(self) -> int:
        return self.compose_out + 2 * self.atom_out + sro.GLOBAL_DIM

    def inventory(self) -> dict[str, int]:
        def mlp(i: int, h: int, o: int) -> int:
            return i * h + h + h * o + o

        def lin(i: int, o: int) -> int:
            return i * o + o

        atom = mlp(self.atom_in, self.atom_hidden, self.atom_out)
        bond = mlp(self.bond_in, self.bond_hidden, self.bond_out)
        pair = mlp(self.pair_in, self.pair_hidden, self.pair_out)
        compose = mlp(self.compose_in, self.compose_hidden, self.compose_out)
        reader = lin(self.reader_in, self.reader_hidden)
        for _ in range(self.reader_layers - 1):
            reader += lin(self.reader_hidden, self.reader_hidden)
        reader += lin(self.reader_hidden, 1)
        return {
            "atom_encoder": atom,
            "bond_encoder": bond,
            "pair_mlp": pair,
            "group_composer": compose,
            "reader": reader,
            "total": atom + bond + pair + compose + reader,
        }


class BondRWComposer(nn.Module):
    """One-shot Bond-Bond relation composer (no learned Junction encoder)."""

    def __init__(self, config: BondRWConfig, global_mean: np.ndarray, global_scale: np.ndarray):
        super().__init__()
        self.config = config
        self.atom_encoder = src._MLP(config.atom_in, config.atom_hidden, config.atom_out, 1, config.dropout)
        self.bond_encoder = src._MLP(config.bond_in, config.bond_hidden, config.bond_out, 1, config.dropout)
        self.pair_mlp = src._MLP(config.pair_in, config.pair_hidden, config.pair_out, 1, config.dropout)
        self.group_composer = src._MLP(config.compose_in, config.compose_hidden, config.compose_out, 1, config.dropout)
        self.reader = src._MLP(config.reader_in, config.reader_hidden, 1, config.reader_layers, config.dropout)
        self.register_buffer("global_mean", torch.as_tensor(global_mean, dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(global_scale, dtype=torch.float32))

    def encode_atoms(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.atom_encoder(batch["atom_feat"])

    def encode_bonds(self, a: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        feats = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        return self.bond_encoder(feats)

    def pair_aggregate(self, b: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        src, dst, w = batch["pair_src"], batch["pair_dst"], batch["pair_w"]
        bs, bd = b[src], b[dst]
        phi = self.pair_mlp(torch.cat([bs + bd, (bs - bd).abs(), bs * bd], dim=1))
        w_eff = torch.where(src == dst, w * 0.5, w).unsqueeze(1)
        r = torch.zeros_like(b)
        r.index_add_(0, src, phi * w_eff)
        r.index_add_(0, dst, phi * w_eff)
        return r

    def forward_features(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        r = self.pair_aggregate(b, batch)
        n_bonds = int(b.shape[0])
        zeros = torch.zeros((n_bonds, self.config.compose_out), device=b.device, dtype=b.dtype)
        zero_scalar = torch.zeros((n_bonds, 1), device=b.device, dtype=b.dtype)
        compose_in = torch.cat([b, r, zeros, zero_scalar, batch["bond_feat"]], dim=1)
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        return {"a": a, "b": b, "r": r, "c": c, "z": z, "reader_in": reader_in}

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        return_features: bool = False,
    ):
        feats = self.forward_features(batch)
        pred = self.reader(feats["reader_in"]).view(-1)
        if return_features:
            return pred, feats
        return pred


def build_model(seed: int, norm: Mapping[str, Any], device: torch.device | str = "cpu") -> BondRWComposer:
    """Shared scratch init for R and U (the arm never touches initialisation)."""
    cfg = BondRWConfig()
    torch.manual_seed(int(seed))
    model = BondRWComposer(
        cfg,
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
    )
    return model.to(device)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def state_dict_hash(model: nn.Module) -> str:
    return src.state_dict_hash(model)


def encoder_group_hash(model: BondRWComposer) -> str:
    return bp.state_group_sha256(
        {**{f"atom:{k}": v for k, v in model.atom_encoder.state_dict().items()},
         **{f"bond:{k}": v for k, v in model.bond_encoder.state_dict().items()}}
    )


def load_ssl_encoders(model: BondRWComposer, ssl_state: Mapping[str, Any], device: torch.device | str) -> None:
    """Load only the atom/bond encoder groups (no Junction encoder exists)."""
    model.atom_encoder.load_state_dict({k: v.to(device) for k, v in ssl_state["atom_encoder"].items()})
    model.bond_encoder.load_state_dict({k: v.to(device) for k, v in ssl_state["bond_encoder"].items()})


def forbidden_module_scan(model: nn.Module) -> list[str]:
    return src.forbidden_module_scan(model)


def batch_rw_objects(
    objs: Sequence[sro.MolObjects],
    triplets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Standard object batch + flat unordered pair list with global bond ids."""
    batch = sro.batch_objects(objs)
    srcs, dsts, ws = [], [], []
    offset = 0
    for m, (src, dst, w) in enumerate(triplets):
        n_b = int(objs[m].bond_index.shape[0])
        srcs.append(torch.as_tensor(np.asarray(src) + offset, dtype=torch.long))
        dsts.append(torch.as_tensor(np.asarray(dst) + offset, dtype=torch.long))
        ws.append(torch.as_tensor(np.asarray(w), dtype=torch.float32))
        offset += n_b
    batch["pair_src"] = torch.cat(srcs, 0) if srcs else torch.zeros(0, dtype=torch.long)
    batch["pair_dst"] = torch.cat(dsts, 0) if dsts else torch.zeros(0, dtype=torch.long)
    batch["pair_w"] = torch.cat(ws, 0) if ws else torch.zeros(0)
    return batch


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_model(
    model: BondRWComposer,
    objs: Sequence[sro.MolObjects],
    triplets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device | str,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), int(batch_size)):
            chunk_objs = list(objs[i : i + int(batch_size)])
            chunk_trip = list(triplets[i : i + int(batch_size)])
            batch = move_batch(batch_rw_objects(chunk_objs, chunk_trip), torch.device(device))
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros(0)


# ===========================================================================
# Part B — training loop (frozen R3 recipe; identical to the BJOC P3 loop)
# ===========================================================================


def train_supervised_rw(
    *,
    arm: str,
    seed: int,
    train_objs: Sequence[sro.MolObjects],
    train_triplets: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    device: torch.device,
    ssl_state: Mapping[str, Any] | None = None,
    recipe: Mapping[str, Any] | None = None,
    smoke_epochs: int | None = None,
    log_fn=print,
) -> tuple[BondRWComposer, dict[str, Any]]:
    """One frozen R3 trajectory; the only arm difference is ``train_triplets``."""
    rec = dict(bp.RECIPES["R3"] if recipe is None else recipe)
    rec.setdefault("batch_size", bp.BATCH_SIZE)
    rec.setdefault("grad_clip", bp.GRAD_CLIP)
    rec.setdefault("shuffle_offset", bp.TRAIN_SHUFFLE_OFFSET)
    rec.setdefault("soup_last", bp.SOUP_LAST)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")
    model = build_model(seed, norm, device)
    cfg = BondRWConfig()
    audit = src.parameter_audit(cfg, model)  # type: ignore[arg-type]
    if not audit["parameter_exact"] or audit["actual_parameters"] != EXPECTED_PARAMETERS:
        raise SystemExit(f"parameter audit failed: {audit}")
    init_full_sha = state_dict_hash(model)
    init_encoder_sha = encoder_group_hash(model)
    if ssl_state is not None:
        load_ssl_encoders(model, ssl_state, device)
    encoder_used_sha = encoder_group_hash(model)

    epochs = int(rec["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(train_objs))
    batch_size = int(rec["batch_size"])
    steps_per_epoch = math.ceil(n / batch_size)
    total_updates = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=float(rec["lr"]), weight_decay=float(rec["weight_decay"]))
    scheduler = None
    if rec["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates, eta_min=float(rec["lr_min"])
        )
    elif rec["schedule"] != "constant":
        raise SystemExit(f"unknown schedule {rec['schedule']!r}")

    gen = torch.Generator().manual_seed(int(seed) + int(rec["shuffle_offset"]))
    states: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    grad_norm_max = 0.0
    clip_events = 0
    import time as _time

    started = _time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_seen, steps = 0.0, 0, 0
        t_ep = _time.perf_counter()
        for indices in bp.shuffle_batches(n, batch_size, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = move_batch(
                batch_rw_objects([train_objs[i] for i in indices], [train_triplets[i] for i in indices]),
                device,
            )
            yb = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(rec["grad_clip"])))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > float(rec["grad_clip"]):
                clip_events += 1
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float((pred.view(-1) - yb.view(-1)).abs().sum())
            n_seen += len(indices)
            steps += 1
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        curve.append(
            {
                "epoch": epoch,
                "train_mae": total / max(n_seen, 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": _time.perf_counter() - t_ep,
            }
        )
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} lr={curve[-1]['lr']:.3e} {curve[-1]['seconds']:.1f}s")
    wall = _time.perf_counter() - started

    members = list(range(epochs - len(states) + 1, epochs + 1))[-int(rec["soup_last"]):]
    soup_states = states[-int(rec["soup_last"]):]
    soup = {k: torch.stack([s[k].float() for s in soup_states], 0).mean(0) for k in soup_states[0].keys()}
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_model(model, train_objs, train_triplets, device)
    expected_order = bp.batch_order_hash(seed, n, batch_size, epochs)
    info = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "recipe": rec,
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {rec['soup_last']} epoch states",
        "soup_members": members,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()) if len(fit_pred) else None,
        "init_full_sha256": init_full_sha,
        "init_encoder_sha256": init_encoder_sha,
        "encoder_used_sha256": encoder_used_sha,
        "parameter_audit": audit,
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "full_hash_independent_recomputation": expected_order,
            "full_matches": order_hash.hexdigest() == expected_order,
        },
        "anomalies": {"grad_clip_events": int(clip_events), "grad_norm_max": grad_norm_max},
        "curve": curve,
        "wall_seconds": wall,
        "molecules_seen": int(epochs * n),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "labels_used": "Fit rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    return model, info


# ===========================================================================
# Part C — synthetic witnesses
# ===========================================================================


def junction_moment_counterexample() -> dict[str, Any]:
    """Abstract object pair: identical raw marginals, different bond relations.

    System A incidence (4 bonds, 4 junctions, every bond touches two junctions)::

        j1={b1,b2} j2={b2,b3} j3={b3,b4} j4={b4,b1}   (line graph = C4)

    System B incidence (same junction multiset, same per-bond counts)::

        j1={b1,b2} j2={b1,b2} j3={b3,b4} j4={b3,b4}   (line graph = 2 disjoint edges)

    The two systems have identical bond multiset, identical junction multiset
    (X,X,Y,Y), identical per-bond junction counts (2,2,2,2) and therefore
    identical *marginal* (BJOC-Null style) summaries; only the incidence
    organisation, hence the bond line graph and its random-walk kernel, differs.

    **Not a legal molecular graph**: B would require two distinct junctions
    with the same unordered bond pair, i.e. two parallel paths, so this is an
    explicitly labelled abstract counterexample (the task allows non-molecules
    if marked).  A separate real-molecule search is reported in the audit.
    """
    A_inc = np.zeros((4, 4), dtype=np.float64)
    for e, f in ((0, 1), (1, 2), (2, 3), (3, 0)):
        A_inc[e, f] = A_inc[f, e] = 1.0
    A_inc_wrong = np.zeros((4, 4), dtype=np.float64)
    for e, f in ((0, 1), (2, 3)):
        A_inc_wrong[e, f] = A_inc_wrong[f, e] = 1.0
    K_A = relation_kernel(kernel_powers(A_inc, MAX_K))
    K_B = relation_kernel(kernel_powers(A_inc_wrong, MAX_K))
    marg_A = (4, (0, 0, 1, 1), (2, 2, 2, 2))
    return {
        "legal_molecule": False,
        "warning": "abstract incidence systems, not realisable as molecules",
        "raw_marginals_equal": marg_A,
        "null_projection_equal": True,
        "K_max_abs_diff": float(np.abs(K_A - K_B).max()),
        "K_A_tv_uniform": _tv_rows_to_uniform(K_A),
        "K_B_tv_uniform": _tv_rows_to_uniform(K_B),
        "bond_line_graph": {"A_edges": float(A_inc.sum() / 2), "B_edges": float(A_inc_wrong.sum() / 2)},
    }


def pair_block_ablation(
    phi_net: nn.Module,
    *,
    n: int = 32,
    seed: int = 0,
    dim: int = 8,
) -> dict[str, Any]:
    """C2: does the Pair MLP use the joint (multiplicative) input block?"""
    rng = np.random.default_rng(seed)
    b = torch.as_tensor(rng.normal(size=(n, dim)), dtype=torch.float32)
    bs, bd = b[:, None, :], b[None, :, :]
    blocks = {"sum": bs + bd, "diff": (bs - bd).abs(), "prod": bs * bd}
    with torch.no_grad():
        full = phi_net(torch.cat([blocks["sum"], blocks["diff"], blocks["prod"]], dim=-1))
        out: dict[str, Any] = {}
        for idx, name in enumerate(blocks):
            parts = [blocks[k] for k in blocks]
            parts[idx] = torch.zeros_like(parts[idx])
            ab = phi_net(torch.cat(parts, dim=-1))
            out[f"zero_{name}_mean_abs_dphi"] = float((ab - full).abs().mean())
            out[f"zero_{name}_rel"] = float((ab - full).abs().mean() / full.abs().mean().clamp(min=1e-12))
        out["phi_mean_abs"] = float(full.abs().mean())
    return out


def pair_linear_summary_residual(
    phi_net: nn.Module,
    *,
    n_molecules: int = 64,
    seed: int = 0,
    dim: int = 96,
) -> dict[str, Any]:
    """C2: can the pair output be reduced to a linear reweighting of ``b_f``?

    A pure linear reweighting produces ``r_e = W (K b)_e + c * rowsum_e``.
    Fit exactly that surrogate to the frozen Pair MLP outputs on random bond
    representations and kernels and report the relative residual.
    """
    rng = np.random.default_rng(seed)
    feats, targets = [], []
    for _ in range(int(n_molecules)):
        nb = int(rng.integers(5, 15))
        b = rng.normal(size=(nb, dim)).astype(np.float32)
        A = np.zeros((nb, nb), dtype=np.float64)
        for v in range(1, nb):
            u = int(rng.integers(0, v))
            A[u, v] = A[v, u] = 1.0
        K = relation_kernel(kernel_powers(A, MAX_K))
        with torch.no_grad():
            bt = torch.as_tensor(b)
            bs, bd = bt[:, None, :], bt[None, :, :]
            phi = phi_net(torch.cat([bs + bd, (bs - bd).abs(), bs * bd], dim=-1))
            r = torch.einsum("ef,efk->ek", torch.as_tensor(K, dtype=torch.float32), phi)
        kb = torch.as_tensor(K, dtype=torch.float32) @ bt
        rowsum = torch.as_tensor(K.sum(axis=1), dtype=torch.float32)
        feats.append(torch.cat([kb, rowsum.unsqueeze(1)], dim=1))
        targets.append(r)
    X = torch.cat(feats, 0)
    Y = torch.cat(targets, 0)
    lam = 1e-3
    XtX = X.T @ X + lam * torch.eye(X.shape[1])
    W = torch.linalg.solve(XtX, X.T @ Y)
    pred = X @ W
    resid = float((pred - Y).pow(2).mean())
    signal = float(Y.pow(2).mean())
    return {
        "n_random_graphs": int(n_molecules),
        "n_rows": int(X.shape[0]),
        "relative_residual": resid / max(signal, 1e-12),
        "signal_mean_square": signal,
        "surrogate": "r_e ~ W (K b)_e + c * rowsum_e (all K-dependence flows through K b)",
        "verdict": "residual << 1 would mean the pair channel could be a linear reweighting of b_f",
    }


def synthetic_relation_task(
    *,
    seed: int = 0,
    n_train: int = 160,
    n_test: int = 80,
    n_bonds_range: tuple[int, int] = (6, 14),
    epochs: int = 400,
    pair_hidden: int = PAIR_HIDDEN,
) -> dict[str, Any]:
    """Controlled pair-interaction target: real K vs uniform K vs linear summary.

    Graphs are random connected line graphs (trees + a few cycles).  Bond
    representations ``b_e`` are fixed random vectors.  The target is

        target = sum_ef K_ef (b_e . b_f)^2 + sum_e h(b_e)

    which depends on *which* pairs carry the relation weight.  A pair-composer
    trained with real K should beat the uniform-K control; a purely linear
    reweighting of ``b_f`` cannot represent the pair term.
    """

    def make_graph(rng):
        nb = int(rng.integers(n_bonds_range[0], n_bonds_range[1] + 1))
        A = np.zeros((nb, nb), dtype=np.float64)
        for v in range(1, nb):
            u = int(rng.integers(0, v))
            A[u, v] = A[v, u] = 1.0
        for _ in range(int(rng.integers(0, 3))):
            u, v = rng.integers(0, nb, size=2)
            u, v = int(min(u, v)), int(max(u, v))
            if u != v:
                A[u, v] = A[v, u] = 1.0
        return A

    rng = np.random.default_rng(seed)

    def sample():
        A = make_graph(rng)
        b = rng.normal(size=(A.shape[0], 8))
        K = relation_kernel(kernel_powers(A, MAX_K))
        pair = np.einsum("ef,ek,fk->", K, b, b)  # sum K_ef (b_e.b_f)
        target = pair + b.sum() * 0.1
        return A, b, target

    train = [sample() for _ in range(n_train)]
    test = [sample() for _ in range(n_test)]
    model_real = _TinyPairRegressor(pair_hidden)
    model_unif = _TinyPairRegressor(pair_hidden)
    model_lin = _TinyLinearRegressor()
    for model, kind in ((model_real, "real"), (model_unif, "uniform")):
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        for _ in range(epochs):
            order = rng.permutation(len(train))
            for idx in order:
                A, b, target = train[idx]
                K = relation_kernel(kernel_powers(A, MAX_K))
                if kind == "uniform":
                    K = np.full_like(K, 1.0 / A.shape[0])
                pred = model(torch.as_tensor(K, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32))
                loss = (pred - float(target)) ** 2
                opt.zero_grad()
                loss.backward()
                opt.step()
    opt = torch.optim.Adam(model_lin.parameters(), lr=5e-3)
    for _ in range(epochs):
        for idx in rng.permutation(len(train)):
            A, b, target = train[idx]
            K = relation_kernel(kernel_powers(A, MAX_K))
            pred = model_lin(torch.as_tensor(K, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32))
            loss = (pred - float(target)) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()

    def eval_model(model, uniform=False):
        errs = []
        with torch.no_grad():
            for A, b, target in test:
                K = relation_kernel(kernel_powers(A, MAX_K))
                if uniform:
                    K = np.full_like(K, 1.0 / A.shape[0])
                pred = float(model(torch.as_tensor(K, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)))
                errs.append(abs(pred - float(target)))
        return float(np.mean(errs))

    return {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "target": "sum_ef K_ef (b_e.b_f) + 0.1*sum_e b_e ; b fixed random 8-D",
        "test_mae_real_K_trained": eval_model(model_real, uniform=False),
        "test_mae_real_K_model_evaluated_uniform": eval_model(model_real, uniform=True),
        "test_mae_uniform_K_trained_uniform": eval_model(model_unif, uniform=True),
        "test_mae_linear_summary": eval_model(model_lin, uniform=False),
        "interpretation": (
            "if the real-K model beats both the uniform-K control and the linear summary, "
            "the pair channel uses actual pair organisation; this is a controlled relation "
            "target, not evidence about chemistry"
        ),
    }


class _TinyPairRegressor(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(24, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, K: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n = b.shape[0]
        bs, bd = b[:, None, :], b[None, :, :]
        phi = self.mlp(torch.cat([bs + bd, (bs - bd).abs(), bs * bd], dim=-1)).squeeze(-1)
        return (K * phi).sum()


class _TinyLinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 1)

    def forward(self, K: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        summary = (K @ b).mean(0)
        return self.lin(summary).view(())
