#!/usr/bin/env python
"""Nonlinearity Placement Audit — where must nonlinearity enter the structural lift?

Question (pre-dictionary; no dictionary, no ISTA/K-SVD, no GNN backbone)
-----------------------------------------------------------------------
DOI/AIOM keep the structural processing *linear*:

    H_t = S^t X ,   S = D^{-1/2} A_G D^{-1/2}

i.e. many propagation steps are stacked with **no nonlinear discrimination**
between them.  Under the DOI conclusion the fixed-lift premise (not the RFF
aggregation) caps the readout.  This round asks exactly one thing:

    Is adding nonlinearity *between* propagation steps already enough?

and, if not,

    must those nonlinear structural coordinates be **task-coupled**?

Three lifts, same graph semantics and the same downstream distributional
readout (DOI's ``Phi = [mu_V; mu_E; mu_I; log(1+n); log(1+m)]``):

* **Lift A — Frozen Linear DOI** (existing, frozen): ``H_t = S^t X``.
* **Lift B — Fixed Nonlinear Scattering DOI**: fixed graph wavelets
  ``Psi_0 = I-P``, ``Psi_1 = P-P^2``, ``Psi_2 = P^2-P^4``, low-pass ``P^4``
  with ``P = (I+S)/2``, **pointwise modulus** interleaved with filtering
  (scattering transform), **no trainable structural parameter**.
* **Lift C — Tiny Task-Coupled Incidence Lift**: a ~2.5K-parameter atom--bond
  alternating ``tanh`` network (``h=16``, ``L=2``) producing only local
  coordinates ``{h_v}``, ``{e_e}`` — no pooling, no graph token.

The learned lift never bypasses the distributional readout: the *only* thing
the prediction head sees is ``Phi_theta(G)``.

Discipline
----------
* Only official ZINC ``train`` (10000) and ``valid`` (1000) are loaded.  The
  official ``test`` split is never read, instantiated or referenced.
* ``y`` is used *only* for the capacity probe (and the learned lift's task loss).
* No dictionary, ISTA, K-SVD, attention, learned pooling, graph token,
  canonical IDs, shortest paths or hand-crafted graph statistics.
* Frozen AIOM/DOI results are reused; they are not retuned here.

Reuse
-----
The ZINC loader, ``Mol``, incidence operator, perturbation protocol, probe,
canonicalization and WL fingerprint come from ``run_aiom_representation_audit``;
the distributional RFF/KME machinery comes from
``run_doi_representation_audit``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_aiom_representation_audit import (  # noqa: E402
    Mol,
    apply_standardizer,
    build_codebook,
    cosine_similarity_matrix,
    fit_standardizer,
    incidence_matrix,
    load_mols,
    permute_mol,
    perturb_atom_substitution,
    perturb_bond_type_swap,
    repo_loader,
    train_probe,
    two_switch,
    wl_fingerprint,
)
from tracks.ksvd.code.run_doi_representation_audit import (  # noqa: E402
    D_MAX_RFF,
    KERNELS,
    Population,
    assemble,
    exact_kernel_calibration,
    extract_responses as doi_extract_responses,
    kernel_mean_embeddings,
    median_heuristic,
    relative_change,
    sample_rff,
    select_dimension,
    standardized_distance,
)

# ---------------------------------------------------------------------------
# Fixed configuration (no search)
# ---------------------------------------------------------------------------
H_DIM = 16                      # tiny lift hidden width (frozen)
N_ROUNDS = 2                    # atom-bond alternating rounds (frozen)

D_RFF_PRIMARY = 1024            # primary RFF dimension per distribution
D_RFF_CANDIDATES = [1024, 2048]
N_SCATTER_BLOCKS = 8            # [X, U0, U1, U2, U01, U02, U12, L]

MEDIAN_PAIRS = 50_000
CAL_GRAPHS = 300
CAL_PAIRS = 2_000
SCAT_RFF_SEED_BASE = 91600
LEARN_RFF_SEED_BASE = 92600
EPS = 1e-12

SMOKE_GRAPHS = 64
SMOKE_EPOCHS = 40

N_PERT = 500                    # graphs for the unified geometry audit
N_INVARIANCE = 500
N_INV_PERM = 20

EXPECTED_C_V = 21
EXPECTED_C_E = 3

LIFTS = ("linear_doi", "fixed_scattering", "learned_incidence")
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/nonlinearity_placement"


# ===========================================================================
# 1. Generic distributional context (train-only scaling + bandwidth + RFF)
# ===========================================================================
def fit_rms_scale(resp_list, d: int, which: str):
    """RMS scale ``s_j = sqrt(E_train[r_j^2])`` for one object type."""
    sumsq = np.zeros(d, dtype=np.float64)
    count = 0
    for R_V, R_E, _, _ in resp_list:
        R = R_V if which == "V" else R_E
        sub = R[:, :d]
        sumsq += (sub * sub).sum(axis=0)
        count += sub.shape[0]
    return np.sqrt(sumsq / max(count, 1)), count


def safe_scales(s: np.ndarray, d: int) -> np.ndarray:
    out = s[:d] + EPS
    return np.where(s[:d] == 0.0, 1.0, out)


def build_population(resp_list, d: int, s_V: np.ndarray, s_E: np.ndarray,
                     kind: str) -> Population:
    """Empirical object population, mirroring the DOI construction exactly."""
    sv_scale = safe_scales(s_V, d)
    se_scale = safe_scales(s_E, d)
    parts: list[np.ndarray] = []
    counts: list[int] = []
    for R_V, R_E, pv, pe in resp_list:
        sv = R_V[:, :d] / sv_scale
        se = R_E[:, :d] / se_scale
        if kind == "V":
            block = sv
        elif kind == "E":
            block = se
        else:  # "I"
            if pv.size:
                block = np.concatenate([sv[pv], se[pe]], axis=1)
            else:
                block = np.zeros((0, 2 * d), dtype=np.float64)
        parts.append(np.ascontiguousarray(block))
        counts.append(block.shape[0])
    width = (2 * d) if kind == "I" else d
    objs = np.concatenate(parts, axis=0) if parts else np.zeros((0, width))
    counts_arr = np.asarray(counts, dtype=np.int64)
    offsets = np.zeros(len(counts_arr), dtype=np.int64)
    if len(counts_arr) > 1:
        offsets[1:] = np.cumsum(counts_arr)[:-1]
    return Population(objs, offsets, counts_arr)


@dataclass
class DistributionContext:
    d: int
    s_V: np.ndarray
    s_E: np.ndarray
    sigmas: dict[str, float]
    rffs: dict[str, dict]
    D: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def fit_distribution_context(train_resp, d: int, *, seed: int,
                             rff_seed_base: int) -> DistributionContext:
    """Train-only RMS scales, median-heuristic bandwidths and fixed-seed RFF.

    The RFF dimension is the smallest candidate meeting the label-free
    approximation thresholds (median<0.02, p95<0.05 for all three kernels).
    """
    s_V, _ = fit_rms_scale(train_resp, d, "V")
    s_E, _ = fit_rms_scale(train_resp, d, "E")
    popV = build_population(train_resp, d, s_V, s_E, "V")
    popE = build_population(train_resp, d, s_V, s_E, "E")
    popI = build_population(train_resp, d, s_V, s_E, "I")

    rng_bw = np.random.default_rng(seed)
    sigmas: dict[str, float] = {}
    bw_stats: dict[str, Any] = {}
    for key, pop in (("V", popV), ("E", popE), ("I", popI)):
        st = median_heuristic(pop.objs, rng_bw)
        bw_stats[key] = st
        sigmas[key] = float(st["sigma"])

    rffs: dict[str, dict] = {}
    for key, pop in (("V", popV), ("E", popE), ("I", popI)):
        d_eff = pop.objs.shape[1]
        seed_k = rff_seed_base + {"V": 1, "E": 2, "I": 3}[key]
        omega, b = sample_rff(d_eff, sigmas[key], D_MAX_RFF, seed_k)
        rffs[key] = {"omega": omega, "b": b, "sigma": sigmas[key],
                     "d": d_eff, "seed": seed_k}

    # label-free calibration for the RFF-dimension rule
    n_train = len(train_resp)
    n_cal = min(CAL_GRAPHS, n_train)
    cal_idx = np.random.default_rng(seed + 7).choice(n_train, size=n_cal, replace=False)
    cal_resp = [train_resp[i] for i in cal_idx]
    cal_pops = {k: build_population(cal_resp, d, s_V, s_E, k) for k in KERNELS}
    n_pairs = min(CAL_PAIRS, n_cal * (n_cal - 1))
    prng = np.random.default_rng(seed + 11)
    pairs = np.stack([prng.integers(0, n_cal, size=n_pairs),
                      prng.integers(0, n_cal, size=n_pairs)], axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    cal_positions = np.arange(n_cal, dtype=np.int64)

    exact = {k: exact_kernel_calibration(cal_pops[k], cal_positions, sigmas[k], pairs)
             for k in KERNELS}
    cal_emb = {f"mu{k}": kernel_mean_embeddings(
        cal_pops[k], rffs[k]["omega"], rffs[k]["b"], D_MAX_RFF)[0] for k in KERNELS}
    D_sel, per_kernel, ok = select_dimension(cal_emb, cal_positions, pairs, exact,
                                             D_RFF_CANDIDATES)

    diagnostics = {
        "d": d,
        "rms_V": {"min": float(s_V.min()), "max": float(s_V.max()),
                  "mean": float(s_V.mean()), "n_zero": int((s_V == 0).sum())},
        "rms_E": {"min": float(s_E.min()), "max": float(s_E.max()),
                  "mean": float(s_E.mean()), "n_zero": int((s_E == 0).sum())},
        "bandwidth": bw_stats,
        "D_selected": D_sel,
        "D_candidates": D_RFF_CANDIDATES,
        "approximation_ok": bool(ok),
        "per_kernel_D_error": per_kernel,
    }
    return DistributionContext(d=d, s_V=s_V, s_E=s_E, sigmas=sigmas, rffs=rffs,
                               D=D_sel, diagnostics=diagnostics)


def materialize_features(resp_list, ctx: DistributionContext):
    """Return ``(features_dict, populations, mus)`` for one split."""
    pops = {k: build_population(resp_list, ctx.d, ctx.s_V, ctx.s_E, k)
            for k in KERNELS}
    mus = {}
    sums = {}
    for k in KERNELS:
        mu, s = kernel_mean_embeddings(pops[k], ctx.rffs[k]["omega"],
                                       ctx.rffs[k]["b"], D_MAX_RFF)
        mus[k], sums[k] = mu, s
    n = np.array([r[0].shape[0] for r in resp_list], dtype=np.int64)
    m = np.array([r[1].shape[0] for r in resp_list], dtype=np.int64)
    feats = assemble(mus["V"], mus["E"], mus["I"], sums["V"], sums["E"], sums["I"],
                     n, m, ctx.D)
    return feats, pops, mus


def kernel_approx_audit(pops: dict, mus: dict, sigmas: dict[str, float], D: int,
                        pairs: np.ndarray) -> dict[str, Any]:
    """Exact Gaussian KME inner product vs RFF approximation on graph pairs."""
    from scipy.stats import spearmanr

    scale = math.sqrt(D_MAX_RFF / D)
    out: dict[str, Any] = {}
    for k in KERNELS:
        pop = pops[k]
        exact = exact_kernel_calibration(pop, np.arange(len(pop.counts)), sigmas[k], pairs)
        emb = mus[k][:, :D] * scale
        approx = np.einsum("ij,ij->i", emb[pairs[:, 0]], emb[pairs[:, 1]])
        err = np.abs(approx - exact)
        out[k] = {
            "median_abs_err": float(np.median(err)),
            "mean_abs_err": float(err.mean()),
            "p95_abs_err": float(np.percentile(err, 95)),
            "spearman": float(spearmanr(approx, exact).correlation),
        }
    out["D"] = int(D)
    out["n_pairs"] = int(len(pairs))
    out["pass"] = bool(all(out[k]["median_abs_err"] < 0.02
                           and out[k]["p95_abs_err"] < 0.05 for k in KERNELS))
    return out


# ===========================================================================
# 2. Lift B — fixed nonlinear scattering responses (no trainable parameters)
# ===========================================================================
def scattering_blocks(mol: Mol, C_V: int, C_E: int, node_index, bond_index):
    """Fixed scattering response blocks.

    ``P = (I+S)/2``; wavelets ``Psi_0=I-P``, ``Psi_1=P-P^2``, ``Psi_2=P^2-P^4``;
    low-pass ``P^4``.  First order ``|Psi_j X|``; second order (ordered
    increasing paths) ``|Psi_{j2} |Psi_{j1} X||``.  Every block keeps the full
    ``N=(n+m)`` incidence-node rows so atom and bond objects are handled together.
    """
    A, X = incidence_matrix(mol, C_V, C_E, node_index, bond_index)
    N = A.shape[0]
    deg = A.sum(axis=1)
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    S = (dinv[:, None] * A) * dinv[None, :]
    eye = np.eye(N, dtype=np.float64)
    P = 0.5 * (eye + S)
    P2 = P @ P
    P4 = P2 @ P2
    Psi0 = eye - P
    Psi1 = P - P2
    Psi2 = P2 - P4
    U0 = np.abs(Psi0 @ X)
    U1 = np.abs(Psi1 @ X)
    U2 = np.abs(Psi2 @ X)
    U01 = np.abs(Psi1 @ U0)
    U02 = np.abs(Psi2 @ U0)
    U12 = np.abs(Psi2 @ U1)
    L = P4 @ X
    blocks = np.concatenate([X, U0, U1, U2, U01, U02, U12, L], axis=1)
    assert blocks.shape[1] == N_SCATTER_BLOCKS * X.shape[1]
    return blocks


def extract_scattering(mols: Sequence[Mol], C_V: int, C_E: int,
                       node_index, bond_index):
    """Return per-molecule ``(R_V, R_E, pair_v, pair_e)`` (undefined, real incidences)."""
    out = []
    for mol in mols:
        B = scattering_blocks(mol, C_V, C_E, node_index, bond_index)
        n, m = mol.n, mol.m
        R_V = np.ascontiguousarray(B[:n])
        R_E = np.ascontiguousarray(B[n:])
        pv = np.empty(2 * m, dtype=np.int64)
        pe = np.empty(2 * m, dtype=np.int64)
        for e, (a, b) in enumerate(mol.bonds):
            pv[2 * e] = a
            pe[2 * e] = e
            pv[2 * e + 1] = b
            pe[2 * e + 1] = e
        out.append((R_V, R_E, pv, pe))
    return out


# ===========================================================================
# 3. Lift C — tiny task-coupled incidence lift (torch, vectorized)
# ===========================================================================
class TinyIncidenceLift(nn.Module):
    """Strictly minimal atom--bond alternating ``tanh`` network.

    ``h_v^0 = W_Vin onehot(x_v)``, ``e_e^0 = W_Ein onehot(x_e)``;
    per round: ``e_e <- tanh(W_EE e_e + W_VE (h_u+h_v) + b_E)`` then
    ``h_v <- tanh(W_VV h_v + W_EV sum_{e ni v} e_e + b_V)``.
    No residual, no LayerNorm, no dropout, no graph-level state.
    """

    def __init__(self, C_V: int, C_E: int, h: int = H_DIM, rounds: int = N_ROUNDS):
        super().__init__()
        g = 1.0 / math.sqrt(h)
        self.W_Vin = nn.Parameter(torch.randn(h, C_V) * g)
        self.W_Ein = nn.Parameter(torch.randn(h, C_E) * g)
        self.W_EE = nn.ParameterList([nn.Parameter(torch.randn(h, h) * g) for _ in range(rounds)])
        self.W_VE = nn.ParameterList([nn.Parameter(torch.randn(h, h) * g) for _ in range(rounds)])
        self.b_E = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(rounds)])
        self.W_VV = nn.ParameterList([nn.Parameter(torch.randn(h, h) * g) for _ in range(rounds)])
        self.W_EV = nn.ParameterList([nn.Parameter(torch.randn(h, h) * g) for _ in range(rounds)])
        self.b_V = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(rounds)])
        self.rounds = rounds

    def forward(self, node_oh, bond_oh, bond_u, bond_v, inc_atom, inc_bond):
        h = node_oh @ self.W_Vin.T
        e = bond_oh @ self.W_Ein.T
        layers = [(h, e)]
        for r in range(self.rounds):
            c = h[bond_u] + h[bond_v]
            e = torch.tanh(e @ self.W_EE[r].T + c @ self.W_VE[r].T + self.b_E[r])
            m = torch.zeros_like(h).index_add_(0, inc_atom, e[inc_bond])
            h = torch.tanh(h @ self.W_VV[r].T + m @ self.W_EV[r].T + self.b_V[r])
            layers.append((h, e))
        return h, e, layers

    def structural_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


@dataclass
class LiftData:
    """Precomputed index/one-hot arrays for a set of graphs (fixed order)."""

    node_oh: np.ndarray          # (Ntot, C_V)
    bond_oh: np.ndarray          # (Mtot, C_E)
    bond_u: np.ndarray           # (Mtot,) global atom id
    bond_v: np.ndarray           # (Mtot,) global atom id
    inc_atom: np.ndarray         # (2Mtot,) global atom id
    inc_bond: np.ndarray         # (2Mtot,) global bond id
    node_lo: np.ndarray          # (G+1,)
    bond_lo: np.ndarray          # (G+1,)
    inc_lo: np.ndarray           # (G+1,)
    n_atoms: np.ndarray          # (G,)
    n_bonds: np.ndarray          # (G,)
    n_graphs: int

    @property
    def ntot(self) -> int:
        return int(self.node_lo[-1])

    @property
    def mtot(self) -> int:
        return int(self.bond_lo[-1])


def make_lift_data(mols: Sequence[Mol], C_V: int, C_E: int, node_index, bond_index) -> LiftData:
    node_oh, bond_oh, bu, bv, ia, ib = [], [], [], [], [], []
    node_lo, bond_lo, inc_lo = [0], [0], [0]
    n_atoms, n_bonds = [], []
    for mol in mols:
        n, m = mol.n, mol.m
        arr = np.zeros((n, C_V), dtype=np.float32)
        arr[np.arange(n), [node_index[int(t)] for t in mol.node_types]] = 1.0
        node_oh.append(arr)
        arr2 = np.zeros((m, C_E), dtype=np.float32)
        if m:
            arr2[np.arange(m), [bond_index[int(t)] for t in mol.bond_types]] = 1.0
        bond_oh.append(arr2)
        base_a, base_b = node_lo[-1], bond_lo[-1]
        for e, (a, b) in enumerate(mol.bonds):
            bu.append(base_a + a)
            bv.append(base_a + b)
            ia.append(base_a + a)
            ib.append(base_b + e)
            ia.append(base_a + b)
            ib.append(base_b + e)
        node_lo.append(node_lo[-1] + n)
        bond_lo.append(bond_lo[-1] + m)
        inc_lo.append(ia.__len__())
        n_atoms.append(n)
        n_bonds.append(m)
    return LiftData(
        node_oh=np.concatenate(node_oh, axis=0) if node_oh else np.zeros((0, C_V), np.float32),
        bond_oh=np.concatenate(bond_oh, axis=0) if bond_oh else np.zeros((0, C_E), np.float32),
        bond_u=np.asarray(bu, dtype=np.int64), bond_v=np.asarray(bv, dtype=np.int64),
        inc_atom=np.asarray(ia, dtype=np.int64), inc_bond=np.asarray(ib, dtype=np.int64),
        node_lo=np.asarray(node_lo, dtype=np.int64), bond_lo=np.asarray(bond_lo, dtype=np.int64),
        inc_lo=np.asarray(inc_lo, dtype=np.int64),
        n_atoms=np.asarray(n_atoms, dtype=np.int64), n_bonds=np.asarray(n_bonds, dtype=np.int64),
        n_graphs=len(mols),
    )


def _concat_ranges(lo: np.ndarray, hi: np.ndarray, glist: Sequence[int]) -> np.ndarray:
    if not len(glist):
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(lo[g], hi[g]) for g in glist])


def batch_tensors(data: LiftData, glist: Sequence[int], device="cpu"):
    """Slice ``glist`` graphs out of a ``LiftData`` into a batched structure."""
    glist = list(glist)
    node_ids = _concat_ranges(data.node_lo[:-1], data.node_lo[1:], glist)
    bond_ids = _concat_ranges(data.bond_lo[:-1], data.bond_lo[1:], glist)
    inc_ids = _concat_ranges(data.inc_lo[:-1], data.inc_lo[1:], glist)

    local_node = np.full(data.ntot, -1, dtype=np.int64)
    local_node[node_ids] = np.arange(node_ids.size)
    local_bond = np.full(data.mtot, -1, dtype=np.int64)
    local_bond[bond_ids] = np.arange(bond_ids.size)
    local_inc = np.full(int(data.inc_lo[-1]), -1, dtype=np.int64)
    local_inc[inc_ids] = np.arange(inc_ids.size)

    node_gid = np.repeat(np.arange(len(glist)), data.n_atoms[glist])
    bond_gid = np.repeat(np.arange(len(glist)), data.n_bonds[glist])
    inc_gid = np.repeat(np.arange(len(glist)), 2 * data.n_bonds[glist])

    bond_u = local_node[data.bond_u[bond_ids]]
    bond_v = local_node[data.bond_v[bond_ids]]
    inc_atom = local_node[data.inc_atom[inc_ids]]
    inc_bond = local_bond[data.inc_bond[inc_ids]]

    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return {
        "node_oh": t(data.node_oh[node_ids]),
        "bond_oh": t(data.bond_oh[bond_ids]),
        "bond_u": t(bond_u, torch.long), "bond_v": t(bond_v, torch.long),
        "inc_atom": t(inc_atom, torch.long), "inc_bond": t(inc_bond, torch.long),
        "node_gid": t(node_gid, torch.long), "bond_gid": t(bond_gid, torch.long),
        "inc_gid": t(inc_gid, torch.long),
        "n_atoms": t(data.n_atoms[glist], torch.long),
        "n_bonds": t(data.n_bonds[glist], torch.long),
        "G": len(glist),
    }


def _segment_mean(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = values.new_zeros((n, values.shape[1]))
    out.index_add_(0, index, values)
    counts = values.new_zeros(n)
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(1)


def phi_from_objects(batch: dict, h: torch.Tensor, e: torch.Tensor,
                     rff: dict, D: int) -> torch.Tensor:
    """Fixed KME readout ``Phi = [mu_V; mu_E; mu_I; log(1+n); log(1+m)]``."""
    G = batch["G"]
    scale = math.sqrt(2.0 / D)
    psiV = torch.cos(h @ rff["V"]["omega"][:D].T + rff["V"]["b"][:D]) * scale
    psiE = torch.cos(e @ rff["E"]["omega"][:D].T + rff["E"]["b"][:D]) * scale
    q = torch.cat([h[batch["inc_atom"]], e[batch["inc_bond"]]], dim=1)
    psiI = torch.cos(q @ rff["I"]["omega"][:D].T + rff["I"]["b"][:D]) * scale
    muV = _segment_mean(psiV, batch["node_gid"], G)
    muE = _segment_mean(psiE, batch["bond_gid"], G)
    muI = _segment_mean(psiI, batch["inc_gid"], G)
    n = batch["n_atoms"].to(h.dtype)
    m = batch["n_bonds"].to(h.dtype)
    return torch.cat([muV, muE, muI, torch.log1p(n).unsqueeze(1),
                      torch.log1p(m).unsqueeze(1)], dim=1)


def make_rff_tensors(rffs_np: dict, device="cpu") -> dict:
    out = {}
    for k, r in rffs_np.items():
        out[k] = {"omega": torch.as_tensor(r["omega"], dtype=torch.float32, device=device),
                  "b": torch.as_tensor(r["b"], dtype=torch.float32, device=device)}
    return out


@torch.no_grad()
def learned_objects_and_phi(model: TinyIncidenceLift, rff_t: dict, data: LiftData,
                            D: int, chunk: int = 256, device="cpu"):
    """Evaluate the learned lift: graph features, per-object responses, layer states."""
    model.eval()
    phis, objsV, objsE, objsI = [], [], [], []
    g = 0
    while g < data.n_graphs:
        glist = list(range(g, min(g + chunk, data.n_graphs)))
        b = batch_tensors(data, glist, device)
        h, e, _ = model(b["node_oh"], b["bond_oh"], b["bond_u"], b["bond_v"],
                        b["inc_atom"], b["inc_bond"])
        phi = phi_from_objects(b, h, e, rff_t, D)
        phis.append(phi.cpu().numpy())
        hn, en = h.cpu().numpy(), e.cpu().numpy()
        for gi, gg in enumerate(glist):
            nm = (b["node_gid"] == gi)
            bm = (b["bond_gid"] == gi)
            im = (b["inc_gid"] == gi)
            hv = hn[nm.numpy()]
            ev = en[bm.numpy()]
            objsV.append(hv)
            objsE.append(ev)
            a_loc = b["inc_atom"][im].numpy()
            b_loc = b["inc_bond"][im].numpy()
            objsI.append(np.concatenate([hn[a_loc], en[b_loc]], axis=1))
        g += chunk
    return np.concatenate(phis, axis=0), objsV, objsE, objsI


def populations_from_objects(objsV, objsE, objsI) -> dict[str, Population]:
    def make(parts):
        counts = np.asarray([p.shape[0] for p in parts], dtype=np.int64)
        objs = np.concatenate(parts, axis=0) if parts else np.zeros((0, 1))
        offsets = np.zeros(len(counts), dtype=np.int64)
        if len(counts) > 1:
            offsets[1:] = np.cumsum(counts)[:-1]
        return Population(objs, offsets, counts)
    return {"V": make(objsV), "E": make(objsE), "I": make(objsI)}


def bandwidth_from_objects(objsV, objsE, objsI, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pops = populations_from_objects(objsV, objsE, objsI)
    return {k: median_heuristic(pops[k].objs, rng) for k in KERNELS}


def sample_rff_from_sigmas(dims: dict[str, int], sigmas: dict[str, float],
                           seed_base: int) -> dict[str, dict]:
    rffs = {}
    for k in KERNELS:
        seed_k = seed_base + {"V": 1, "E": 2, "I": 3}[k]
        omega, b = sample_rff(dims[k], sigmas[k], D_MAX_RFF, seed_k)
        rffs[k] = {"omega": omega, "b": b, "sigma": sigmas[k], "d": dims[k],
                   "seed": seed_k}
    return rffs


# ===========================================================================
# 4. Learned-lift end-to-end training (identical protocol to the DOI probe)
# ===========================================================================
def train_learned_lift(model, head, rff_t, data_train, y_train, data_valid, y_valid,
                       *, seed=0, D, max_epochs=240, patience=40, batch_size=128,
                       lr=1e-3, wd=1e-5, clip=5.0, device="cpu", log=print):
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    ytr = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    yva = torch.as_tensor(y_valid, dtype=torch.float32).reshape(-1, 1)
    gen = torch.Generator().manual_seed(seed + 91011)

    def eval_mae(data, y):
        model.eval()
        head.eval()
        with torch.no_grad():
            g = 0
            preds = []
            while g < data.n_graphs:
                glist = list(range(g, min(g + 256, data.n_graphs)))
                b = batch_tensors(data, glist, device)
                h, e, _ = model(b["node_oh"], b["bond_oh"], b["bond_u"], b["bond_v"],
                                b["inc_atom"], b["inc_bond"])
                phi = phi_from_objects(b, h, e, rff_t, D)
                preds.append(head(phi).cpu())
                g += 256
        pred = torch.cat(preds, dim=0)
        return float((pred - y).abs().mean())

    n = data_train.n_graphs
    best = float("inf")
    best_epoch = 1
    stale = 0
    best_state = None
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        head.train()
        order = torch.randperm(n, generator=gen).tolist()
        for start in range(0, n, batch_size):
            glist = order[start:start + batch_size]
            b = batch_tensors(data_train, glist, device)
            h, e, _ = model(b["node_oh"], b["bond_oh"], b["bond_u"], b["bond_v"],
                            b["inc_atom"], b["inc_bond"])
            phi = phi_from_objects(b, h, e, rff_t, D)
            pred = head(phi)
            loss = torch.nn.functional.l1_loss(pred, ytr[glist])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
        vmae = eval_mae(data_valid, yva)
        history.append({"epoch": epoch, "valid_mae": vmae})
        if vmae < best:
            best = vmae
            best_epoch = epoch
            best_state = {
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        if epoch % 10 == 0 or epoch == 1:
            log(f"    learned epoch {epoch}: valid MAE {vmae:.4f} (best {best:.4f})")
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        head.load_state_dict(best_state["head"])
    train_mae = eval_mae(data_train, ytr)
    valid_mae = eval_mae(data_valid, yva)
    return {"train_mae": train_mae, "valid_mae": valid_mae, "best_epoch": best_epoch,
            "epochs_run": len(history), "history": history,
            "structural_params": model.structural_params(),
            "head_params": int(sum(p.numel() for p in head.parameters()))}


def smoke_overfit(model, head, rff_t, data, y, *, D, epochs=SMOKE_EPOCHS,
                  seed=0, device="cpu", log=print) -> dict[str, Any]:
    """64-graph smoke: gradients flow, loss descends, no NaN (not a gate)."""
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()),
                           lr=1e-3, weight_decay=1e-5)
    yt = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    losses = []
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        head.train()
        order = torch.randperm(data.n_graphs, generator=g).tolist()
        for start in range(0, data.n_graphs, 32):
            glist = order[start:start + 32]
            b = batch_tensors(data, glist, device)
            h, e, _ = model(b["node_oh"], b["bond_oh"], b["bond_u"], b["bond_v"],
                            b["inc_atom"], b["inc_bond"])
            phi = phi_from_objects(b, h, e, rff_t, D)
            loss = torch.nn.functional.l1_loss(head(phi), yt[glist])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
    finite = bool(np.all(np.isfinite(losses)))
    return {"epochs": epochs, "first_loss": losses[0], "last_loss": losses[-1],
            "min_loss": float(np.min(losses)), "finite": finite,
            "decreased": bool(losses[-1] < losses[0])}


# ===========================================================================
# 5. Geometry audit helpers (shared across the three lifts)
# ===========================================================================
def build_perturbed(rng, mol: Mol, kind: str, atom_cats) -> Mol | None:
    if kind == "permutation":
        pa = rng.permutation(mol.n)
        pb = rng.permutation(mol.m) if mol.m > 0 else np.zeros(0, dtype=np.int64)
        return permute_mol(mol, pa, pb)
    if kind == "atom_sub":
        return perturb_atom_substitution(rng, mol, atom_cats)
    if kind == "bond_type_swap":
        return perturb_bond_type_swap(rng, mol)
    if kind == "two_switch":
        return two_switch(rng, mol)
    raise ValueError(kind)


def perturb_geometry(F_base: np.ndarray, F_pert: dict[str, np.ndarray],
                     base_idx: np.ndarray) -> dict[str, Any]:
    """Standardized perturbation distances + random size-matched control."""
    mu, sd = fit_standardizer(F_base)
    Z = apply_standardizer(F_base, mu, sd)
    out = {"std": {}, "relative": {}}
    for kind, Fp in F_pert.items():
        if Fp is None or len(Fp) == 0:
            continue
        b = F_base[base_idx]
        out["std"][kind] = standardized_distance(b, Fp, sd)
        out["relative"][kind] = relative_change(b, Fp)
    return out, Z


def summarize_vals(vals) -> dict[str, Any]:
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def wl_fidelity(phi_std: np.ndarray, wl_sim: np.ndarray, top_k: int = 10) -> dict[str, Any]:
    """Nearest-neighbour WL fidelity + representation/WL distance correlation."""
    from scipy.stats import spearmanr

    n = phi_std.shape[0]
    d2 = np.sum(phi_std ** 2, axis=1)[:, None] + np.sum(phi_std ** 2, axis=1)[None, :] \
        - 2.0 * (phi_std @ phi_std.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :top_k]
    top1 = np.mean([wl_sim[i, nn[i, 0]] for i in range(n)])
    top10 = np.mean([np.mean([wl_sim[i, nn[i, k]] for k in range(top_k)]) for i in range(n)])
    rng = np.random.default_rng(0)
    ii = rng.integers(0, n, size=5000)
    jj = rng.integers(0, n, size=5000)
    mask = ii != jj
    dphi = np.sqrt(d2[ii[mask], jj[mask]])
    dwl = 1.0 - wl_sim[ii[mask], jj[mask]]
    rho = float(spearmanr(dphi, dwl).correlation)
    rand_wl = float(wl_sim[ii[mask], jj[mask]].mean())
    return {"top1_wl_sim": float(top1), "top10_wl_sim": float(top10),
            "random_control_wl_sim": rand_wl, "spearman_dphi_dwl": rho}


# ===========================================================================
# 6. Plot helpers
# ===========================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_plots(out: Path, results: dict, log=print):
    plt = _mpl()
    pdir = out / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    table = results["performance"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    names = [r["lift"] for r in table]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [r["train_mae"] for r in table], width=0.4, label="train")
    ax.bar(xs + 0.2, [r["valid_mae"] for r in table], width=0.4, label="valid")
    ax.axhline(0.3865, color="gray", ls=":", label="frozen DOI valid")
    ax.axhline(0.30, color="red", ls=":", alpha=0.5)
    ax.axhline(0.25, color="orange", ls=":", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=12, fontsize=8)
    ax.set_ylabel("MAE")
    ax.legend(fontsize=7)
    ax.set_title("Nonlinearity placement: capacity probe")
    fig.tight_layout()
    fig.savefig(pdir / "mae_comparison.png", dpi=130)
    plt.close(fig)

    geo = results["geometry"]
    lifts = [lift for lift in LIFTS if lift in geo]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = np.arange(len(lifts))
    rew = [geo[lift]["mean_two_switch"] for lift in lifts]
    rnd = [geo[lift]["mean_random_pair"] for lift in lifts]
    ax.bar(xs - 0.2, rew, width=0.4, label="degree-preserving 2-switch")
    ax.bar(xs + 0.2, rnd, width=0.4, label="random size-matched pair")
    for i, lift in enumerate(lifts):
        ax.text(i, max(rew[i], rnd[i]), f"R={geo[lift]['random_over_rewiring']:.1f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(lifts, rotation=12, fontsize=8)
    ax.set_ylabel("mean standardized $\\Delta_\\Phi$")
    ax.legend(fontsize=7)
    ax.set_title("Rewiring vs random-pair distance ($R_{topo}$)")
    fig.tight_layout()
    fig.savefig(pdir / "rewiring_vs_random.png", dpi=130)
    plt.close(fig)

    wl = results["wl"]
    lifts = [lift for lift in LIFTS if lift in wl]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = np.arange(len(lifts))
    ax.bar(xs - 0.25, [wl[lift]["top1_wl_sim"] for lift in lifts], width=0.25, label="top-1")
    ax.bar(xs, [wl[lift]["top10_wl_sim"] for lift in lifts], width=0.25, label="top-10 mean")
    ax.bar(xs + 0.25, [wl[lift]["random_control_wl_sim"] for lift in lifts], width=0.25,
           label="size-matched random")
    ax2 = ax.twinx()
    ax2.plot(xs, [wl[lift]["spearman_dphi_dwl"] for lift in lifts], "k^--",
             label="Spearman $d_\\Phi,d_{WL}$")
    ax2.set_ylabel("Spearman")
    ax.set_xticks(xs)
    ax.set_xticklabels(lifts, rotation=12, fontsize=8)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("WL cosine similarity")
    ax.legend(fontsize=6, loc="lower left")
    ax.set_title("WL neighbour fidelity")
    fig.tight_layout()
    fig.savefig(pdir / "wl_fidelity.png", dpi=130)
    plt.close(fig)

    mech = results.get("learned_mechanism")
    if mech:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        labels = ["h0", "e0", "e1", "h1", "e2", "h2"]
        keys = ["delta_h0", "delta_e0", "delta_e1", "delta_h1", "delta_e2", "delta_h2"]
        ax.bar(labels, [mech[k]["mean"] for k in keys])
        ax.set_ylabel("mean L2 change under within-graph rewiring")
        ax.set_title("Learned incidence lift: layer-wise rewiring propagation")
        fig.tight_layout()
        fig.savefig(pdir / "learned_layerwise_rewiring.png", dpi=130)
        plt.close(fig)

    pert = results.get("perturbation")
    if pert:
        plifts = [lift for lift in LIFTS if lift in pert]
        fig, axes = plt.subplots(1, len(plifts), figsize=(3.4 * len(plifts), 3.2), sharey=True)
        kinds = ["permutation", "atom_sub", "bond_type_swap", "two_switch", "random_pair"]
        for ax, lift in zip(np.atleast_1d(axes), plifts):
            for kind in kinds:
                vals = sorted(pert[lift]["std"].get(kind, []))
                if not vals:
                    continue
                y = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, y, label=kind)
            ax.set_title(lift, fontsize=8)
            ax.set_xlabel("standardized $\\Delta_\\Phi$")
        np.atleast_1d(axes)[0].set_ylabel("ECDF")
        np.atleast_1d(axes)[-1].legend(fontsize=6)
        fig.suptitle("Geometry: scattering vs DOI vs learned")
        fig.tight_layout()
        fig.savefig(pdir / "geometry_ecdf.png", dpi=130)
        plt.close(fig)
    log(f"[plots] saved to {pdir}")


# ===========================================================================
# 7. Orchestration
# ===========================================================================
def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception as exc:  # pragma: no cover
        return f"<error: {exc}>"


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
                    encoding="utf-8")


def load_doi_context(results_dir: Path) -> DistributionContext:
    """Rebuild the *recorded* DOI hyperparameters (no re-tuning)."""
    sc = np.load(results_dir / "scales_T4.npz")
    s_V, s_E = sc["s_V"], sc["s_E"]
    bw = json.loads((results_dir / "bandwidth_stats.json").read_text(encoding="utf-8"))["4"]
    meta = json.loads((results_dir / "rff_metadata.json").read_text(encoding="utf-8"))["4"]
    approx = json.loads((results_dir / "rff_approximation.json").read_text(encoding="utf-8"))["4"]
    sigmas = {k: float(bw[k]["sigma"]) for k in KERNELS}
    rffs = {}
    for k in KERNELS:
        omega, b = sample_rff(int(meta[k]["d"]), sigmas[k], D_MAX_RFF, int(meta[k]["seed"]))
        rffs[k] = {"omega": omega, "b": b, "sigma": sigmas[k], "d": int(meta[k]["d"]),
                   "seed": int(meta[k]["seed"])}
    return DistributionContext(d=120, s_V=s_V, s_E=s_E, sigmas=sigmas, rffs=rffs,
                               D=int(approx["D_selected"]), diagnostics={"source": "recorded"})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--doi-results", type=Path,
                        default=REPO_ROOT / "tracks/ksvd/results/doi_representation")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    t0 = time.time()
    quick = args.quick
    log_path = out / "run.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log("=== Nonlinearity Placement Audit start ===")
    provenance = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "platform": platform.platform(),
        "seed": args.seed, "quick": quick, "device": args.device,
        "H_DIM": H_DIM, "N_ROUNDS": N_ROUNDS,
        "D_RFF_PRIMARY": D_RFF_PRIMARY, "D_RFF_CANDIDATES": D_RFF_CANDIDATES,
    }
    write_json(out / "provenance.json", provenance)
    log(f"provenance commit={provenance['git_commit']}")

    limit = args.limit if args.limit is not None else (300 if quick else None)
    train_mols = load_mols(args.data_root, "train", limit)
    valid_mols = load_mols(args.data_root, "valid", limit)
    log(f"loaded train={len(train_mols)} valid={len(valid_mols)} (test never loaded)")

    atom_cats, bond_cats = build_codebook(train_mols + valid_mols)
    C_V, C_E = len(atom_cats), len(bond_cats)
    node_index = {c: i for i, c in enumerate(atom_cats)}
    bond_index = {c: i for i, c in enumerate(bond_cats)}
    if not quick:
        assert C_V == EXPECTED_C_V and C_E == EXPECTED_C_E, f"C_V={C_V} C_E={C_E}"
    log(f"codebook C_V={C_V} C_E={C_E}")

    # edge semantics audit
    self_loops = multi_edges = 0
    for mol in train_mols[:min(500, len(train_mols))]:
        seen: dict[tuple[int, int], int] = {}
        for (a, b) in mol.bonds:
            if a == b:
                self_loops += 1
            seen[(a, b)] = seen.get((a, b), 0) + 1
        multi_edges += sum(1 for v in seen.values() if v > 1)
    edge_semantics = {"C_V": C_V, "C_E": C_E, "self_loops": self_loops,
                      "multi_edges": multi_edges}
    write_json(out / "edge_semantics.json", edge_semantics)
    log(f"edge semantics {edge_semantics}")

    train_resp_doi = doi_extract_responses(train_mols, C_V, C_E, node_index, bond_index)
    valid_resp_doi = doi_extract_responses(valid_mols, C_V, C_E, node_index, bond_index)

    # =====================================================================
    # Lift B — Fixed Nonlinear Scattering DOI
    # =====================================================================
    log("extracting scattering responses ...")
    train_resp_scat = extract_scattering(train_mols, C_V, C_E, node_index, bond_index)
    valid_resp_scat = extract_scattering(valid_mols, C_V, C_E, node_index, bond_index)
    d_scat = train_resp_scat[0][0].shape[1]
    log(f"scattering response dim d={d_scat} (8 blocks x {C_V + C_E})")

    scat_ctx = fit_distribution_context(train_resp_scat, d_scat, seed=args.seed,
                                        rff_seed_base=SCAT_RFF_SEED_BASE)
    log(f"scattering D_selected={scat_ctx.D} ok={scat_ctx.diagnostics['approximation_ok']} "
        f"sigmas={ {k: round(v, 3) for k, v in scat_ctx.sigmas.items()} }")
    scat_feat_train, scat_pops_train, scat_mus_train = materialize_features(
        train_resp_scat, scat_ctx)
    scat_feat_valid, scat_pops_valid, scat_mus_valid = materialize_features(
        valid_resp_scat, scat_ctx)
    write_json(out / "scattering_metadata.json", scat_ctx.diagnostics)
    log(f"scattering feature dims full={scat_feat_train['full'].shape[1]}")

    # scattering RFF approximation audit (label-free)
    rng_cal = np.random.default_rng(args.seed + 31)
    n_cal = min(CAL_GRAPHS, len(train_resp_scat))
    cal_idx = np.random.default_rng(args.seed + 33).choice(len(train_resp_scat), n_cal, replace=False)
    cal_resp = [train_resp_scat[i] for i in cal_idx]
    cal_ctx_pops = {k: build_population(cal_resp, d_scat, scat_ctx.s_V, scat_ctx.s_E, k)
                    for k in KERNELS}
    cal_mus = {k: kernel_mean_embeddings(cal_ctx_pops[k], scat_ctx.rffs[k]["omega"],
                                         scat_ctx.rffs[k]["b"], D_MAX_RFF)[0] for k in KERNELS}
    n_ap = min(CAL_PAIRS, n_cal * (n_cal - 1))
    ap_pairs = np.stack([rng_cal.integers(0, n_cal, size=n_ap),
                         rng_cal.integers(0, n_cal, size=n_ap)], axis=1)
    ap_pairs = ap_pairs[ap_pairs[:, 0] != ap_pairs[:, 1]]
    scat_approx = kernel_approx_audit(cal_ctx_pops, cal_mus, scat_ctx.sigmas,
                                      scat_ctx.D, ap_pairs)
    write_json(out / "scattering_rff_approximation.json", scat_approx)
    log(f"scattering RFF approx pass={scat_approx['pass']} "
        f"worst median={max(scat_approx[k]['median_abs_err'] for k in KERNELS):.4f} "
        f"worst p95={max(scat_approx[k]['p95_abs_err'] for k in KERNELS):.4f}")

    # =====================================================================
    # Lift A — reproduce the frozen DOI features (recorded hyperparameters)
    # =====================================================================
    # The recorded DOI hyperparameters (d=120, C_V=21/C_E=3) are only valid on
    # the full codebook; ``--quick`` (a smoke mode) skips the frozen lift.
    doi_ctx = None
    if not quick:
        log("materializing frozen linear DOI features ...")
        doi_ctx = load_doi_context(args.doi_results)
        doi_feat_train, _, _ = materialize_features(train_resp_doi, doi_ctx)
        doi_feat_valid, _, _ = materialize_features(valid_resp_doi, doi_ctx)
        # verify against the pulled DOI artifact (train graphs, first 8 rows)
        try:
            ref = np.load(args.doi_results / "doi_features_T4.npz")
            err = float(np.max(np.abs(doi_feat_train["full"][:8] - ref["full_train"][:8])))
            log(f"DOI reproduction check max abs err (first 8 rows) = {err:.3e}")
            write_json(out / "doi_reproduction_check.json", {"max_abs_err_first8": err})
        except Exception as exc:  # pragma: no cover
            log(f"DOI reproduction check skipped: {exc!r}")
    else:
        log("quick mode: skipping frozen DOI materialization")

    # =====================================================================
    # Capacity probes: frozen DOI + scattering
    # =====================================================================
    load_zinc, _ = repo_loader()
    train_ds = load_zinc(args.data_root, "train")
    valid_ds = load_zinc(args.data_root, "val")
    n_train = len(train_mols)
    n_valid = len(valid_mols)
    y_train = np.array([float(train_ds[i].y.reshape(-1)[0]) for i in range(n_train)])
    y_valid = np.array([float(valid_ds[i].y.reshape(-1)[0]) for i in range(n_valid)])
    log(f"y train mean={y_train.mean():.4f} std={y_train.std():.4f}")

    performance_rows = [{"lift": "linear_doi", "structural_params": 0,
                         "train_mae": 0.3273, "valid_mae": 0.3865,
                         "source": "frozen DOI audit (decision-4b8e2d1f)"}]
    probe_detail: dict[str, Any] = {}

    log("[probe] fixed scattering ...")
    mu, sd = fit_standardizer(scat_feat_train["full"])
    Ztr = apply_standardizer(scat_feat_train["full"], mu, sd)
    Zva = apply_standardizer(scat_feat_valid["full"], mu, sd)
    scat_mlp = train_probe(Ztr, y_train, Zva, y_valid, hidden=64, seed=0, log=log)
    scat_lin = train_probe(Ztr, y_train, Zva, y_valid, hidden=None, seed=0, log=log)
    probe_detail["fixed_scattering"] = {"mlp": scat_mlp, "linear": scat_lin}
    performance_rows.append({"lift": "fixed_scattering", "structural_params": 0,
                             "train_mae": scat_mlp["train_mae"],
                             "valid_mae": scat_mlp["valid_mae"],
                             "dim": int(scat_feat_train["full"].shape[1]),
                             "source": "this run"})
    log(f"  fixed_scattering MLP train/valid={scat_mlp['train_mae']:.4f}/{scat_mlp['valid_mae']:.4f}")

    # =====================================================================
    # Lift C — tiny task-coupled incidence lift
    # =====================================================================
    learned_result = None
    learned_phi_train = learned_phi_valid = None
    if not args.skip_learned:
        log("[learned] initializing tiny incidence lift (seed 0) ...")
        torch.manual_seed(0)
        np.random.seed(0)
        model = TinyIncidenceLift(C_V, C_E, H_DIM, N_ROUNDS)
        head = nn.Sequential(nn.Linear(3 * D_RFF_PRIMARY + 2, 64), nn.SiLU(),
                             nn.Linear(64, 1))
        log(f"  structural params={model.structural_params()} "
            f"head params={sum(p.numel() for p in head.parameters())}")

        data_train = make_lift_data(train_mols, C_V, C_E, node_index, bond_index)
        data_valid = make_lift_data(valid_mols, C_V, C_E, node_index, bond_index)

        # fixed bandwidths from the initialized model (train-only, label-free)
        log("  estimating fixed learned-lift bandwidths at init ...")
        _, init_v, init_e, init_i = learned_objects_and_phi(
            model, make_rff_tensors(sample_rff_from_sigmas(
                {"V": H_DIM, "E": H_DIM, "I": 2 * H_DIM},
                {"V": 1.0, "E": 1.0, "I": 1.0}, LEARN_RFF_SEED_BASE),
                args.device), data_train, D_RFF_PRIMARY, chunk=512, device=args.device)
        bw = bandwidth_from_objects(init_v, init_e, init_i, args.seed)
        learn_sigmas = {k: float(bw[k]["sigma"]) for k in KERNELS}
        learn_rffs = sample_rff_from_sigmas(
            {"V": H_DIM, "E": H_DIM, "I": 2 * H_DIM}, learn_sigmas, LEARN_RFF_SEED_BASE)
        rff_t = make_rff_tensors(learn_rffs, args.device)

        # learned RFF dimension is fixed at 1024 (spec section 18); we still run
        # the *label-free* approximation audit at init for reporting.
        n_lc = min(CAL_GRAPHS, data_train.n_graphs)
        lc_graphs = list(np.random.default_rng(args.seed + 41).choice(
            data_train.n_graphs, n_lc, replace=False))
        lc_data = make_lift_data([train_mols[g] for g in lc_graphs], C_V, C_E,
                                 node_index, bond_index)
        _, lc_v, lc_e, lc_i = learned_objects_and_phi(model, rff_t, lc_data,
                                                      D_RFF_PRIMARY, device=args.device)
        lc_pops = populations_from_objects(lc_v, lc_e, lc_i)
        lc_mus = {}
        for k in KERNELS:
            mu_, _ = kernel_mean_embeddings(lc_pops[k], learn_rffs[k]["omega"],
                                            learn_rffs[k]["b"], D_MAX_RFF)
            lc_mus[k] = mu_
        learn_pairs = np.stack([rng_cal.integers(0, n_lc, size=CAL_PAIRS),
                                rng_cal.integers(0, n_lc, size=CAL_PAIRS)], axis=1)
        learn_pairs = learn_pairs[learn_pairs[:, 0] != learn_pairs[:, 1]]
        D_learn = D_RFF_PRIMARY
        learn_init_approx = kernel_approx_audit(lc_pops, lc_mus, learn_sigmas,
                                                D_learn, learn_pairs)
        log(f"  learned D={D_learn} (fixed) init approx median="
            f"{max(learn_init_approx[k]['median_abs_err'] for k in KERNELS):.4f} "
            f"p95={max(learn_init_approx[k]['p95_abs_err'] for k in KERNELS):.4f} "
            f"sigmas={ {k: round(v, 3) for k, v in learn_sigmas.items()} }")
        write_json(out / "learned_metadata.json", {
            "structural_params": model.structural_params(),
            "head_params": int(sum(p.numel() for p in head.parameters())),
            "H_DIM": H_DIM, "N_ROUNDS": N_ROUNDS,
            "sigmas": learn_sigmas, "bandwidth_stats": bw,
            "rff": {k: {"seed": learn_rffs[k]["seed"], "d": learn_rffs[k]["d"],
                        "sigma": learn_sigmas[k]} for k in KERNELS},
            "D": D_learn,
            "init_approximation": learn_init_approx,
            "weight_matrices": {name: list(p.shape) for name, p in model.named_parameters()},
        })

        # 64-graph smoke (gradients / descent / finiteness — not a gate)
        n_smoke = min(SMOKE_GRAPHS, data_train.n_graphs)
        smoke_src = list(np.random.default_rng(args.seed + 51).choice(
            data_train.n_graphs, n_smoke, replace=False))
        smoke_data = make_lift_data([train_mols[g] for g in smoke_src], C_V, C_E,
                                    node_index, bond_index)
        smoke = smoke_overfit(copy.deepcopy(model), copy.deepcopy(head), rff_t,
                              smoke_data, y_train[smoke_src], D=D_learn, log=log)
        write_json(out / "learned_smoke.json", smoke)
        log(f"  smoke {smoke}")

        # end-to-end training on the full train split
        log("  training learned lift end-to-end (MAE only) ...")
        learned_result = train_learned_lift(
            model, head, rff_t, data_train, y_train, data_valid, y_valid,
            seed=0, D=D_learn, device=args.device, log=log)
        performance_rows.append({
            "lift": "learned_incidence",
            "structural_params": learned_result["structural_params"],
            "train_mae": learned_result["train_mae"],
            "valid_mae": learned_result["valid_mae"],
            "dim": int(3 * D_learn + 2), "source": "this run",
        })
        log(f"  learned MLP train/valid={learned_result['train_mae']:.4f}/"
            f"{learned_result['valid_mae']:.4f}")
        write_json(out / "learned_probe.json", learned_result)

        # post-training materialization (features + object responses)
        learned_phi_train, objsV, objsE, objsI = learned_objects_and_phi(
            model, rff_t, data_train, D_learn, chunk=256, device=args.device)
        learned_phi_valid, _, _, _ = learned_objects_and_phi(
            model, rff_t, data_valid, D_learn, chunk=256, device=args.device)

        # post-training kernel approximation audit (train-only)
        n_ka = min(CAL_GRAPHS, data_train.n_graphs)
        ka_graphs = list(np.random.default_rng(args.seed + 61).choice(
            data_train.n_graphs, n_ka, replace=False))
        ka_pops = populations_from_objects([objsV[g] for g in ka_graphs],
                                           [objsE[g] for g in ka_graphs],
                                           [objsI[g] for g in ka_graphs])
        ka_mus = {}
        for k in KERNELS:
            mu_, _ = kernel_mean_embeddings(ka_pops[k], learn_rffs[k]["omega"],
                                            learn_rffs[k]["b"], D_MAX_RFF)
            ka_mus[k] = mu_
        ka_pairs = np.stack([rng_cal.integers(0, n_ka, size=CAL_PAIRS),
                             rng_cal.integers(0, n_ka, size=CAL_PAIRS)], axis=1)
        ka_pairs = ka_pairs[ka_pairs[:, 0] != ka_pairs[:, 1]]
        learn_post_approx = kernel_approx_audit(ka_pops, ka_mus, learn_sigmas, D_learn, ka_pairs)
        learn_post_approx["confounded"] = bool(
            max(learn_post_approx[k]["p95_abs_err"] for k in KERNELS) > 0.10)
        write_json(out / "learned_rff_approximation.json", learn_post_approx)
        log(f"  learned post-training RFF approx pass={learn_post_approx['pass']} "
            f"confounded={learn_post_approx['confounded']}")

        np.savez_compressed(
            out / "learned_features.npz",
            train=learned_phi_train.astype(np.float32),
            valid=learned_phi_valid.astype(np.float32))
        torch.save({"model": model.state_dict(), "head": head.state_dict(),
                    "D": D_learn, "sigmas": learn_sigmas},
                   out / "learned_model.pt")
        expected_dim = 3 * D_RFF_PRIMARY + 2
        if D_learn != D_RFF_PRIMARY:  # pragma: no cover - learned D is fixed
            log(f"  NOTE: learned RFF dimension raised to {D_learn} "
                f"({expected_dim} -> {3 * D_learn + 2})")

    # =====================================================================
    # Unified geometry audit (500 graphs): DOI vs scattering vs learned
    # =====================================================================
    geometry: dict[str, Any] = {}
    wl_metrics: dict[str, Any] = {}
    perturbation: dict[str, Any] = {}
    learned_mechanism = None
    if not args.skip_geometry:
        n_pert = min(N_PERT, n_train)
        geo_rng = np.random.default_rng(args.seed + 71)
        geo_idx = geo_rng.choice(n_train, size=n_pert, replace=False)
        base_mols = [train_mols[int(i)] for i in geo_idx]
        kinds = ["permutation", "atom_sub", "bond_type_swap", "two_switch"]
        pert_mols: dict[str, list[Mol]] = {k: [] for k in kinds}
        pert_keep: dict[str, list[int]] = {k: [] for k in kinds}
        for loc, mol in enumerate(base_mols):
            for kind in kinds:
                pm = build_perturbed(geo_rng, mol, kind, atom_cats)
                if pm is not None:
                    pert_mols[kind].append(pm)
                    pert_keep[kind].append(loc)
        for kind in kinds:
            pert_keep[kind] = np.asarray(pert_keep[kind], dtype=np.int64)

        def sci_phi(mols):
            resp = extract_scattering(mols, C_V, C_E, node_index, bond_index)
            feats, _, _ = materialize_features(resp, scat_ctx)
            return feats["full"]

        def doi_phi(mols):
            resp = doi_extract_responses(mols, C_V, C_E, node_index, bond_index)
            feats, _, _ = materialize_features(resp, doi_ctx)
            return feats["full"]

        base_features = {
            "fixed_scattering": sci_phi(base_mols),
        }
        if doi_ctx is not None:
            base_features["linear_doi"] = doi_phi(base_mols)
        lifts_available = ["fixed_scattering"] + (["linear_doi"] if doi_ctx is not None else [])
        pert_features = {lift: {} for lift in lifts_available}
        for kind in kinds:
            if not pert_mols[kind]:
                continue
            pert_features["fixed_scattering"][kind] = sci_phi(pert_mols[kind])
            if doi_ctx is not None:
                pert_features["linear_doi"][kind] = doi_phi(pert_mols[kind])

        # WL histograms on the base sample
        wl_hists = [wl_fingerprint(m, rounds=3) for m in base_mols]
        wl_keys = sorted({k for h in wl_hists for k in h}, key=repr)
        wl_index = {k: i for i, k in enumerate(wl_keys)}
        W = np.zeros((len(wl_hists), len(wl_keys)))
        for r, h in enumerate(wl_hists):
            for k, v in h.items():
                W[r, wl_index[k]] = v
        wl_sim = cosine_similarity_matrix(W)

        if learned_phi_train is not None:
            # learned features for the same sample (fixed order = train order)
            base_features["learned_incidence"] = learned_phi_train[geo_idx]
            lifts_available = list(base_features.keys())
            pert_features["learned_incidence"] = {}
            for kind in kinds:
                if not pert_mols[kind]:
                    continue
                pm_data = make_lift_data(pert_mols[kind], C_V, C_E, node_index, bond_index)
                phi_p, _, _, _ = learned_objects_and_phi(
                    model, rff_t, pm_data, D_learn, chunk=256, device=args.device)
                pert_features["learned_incidence"][kind] = phi_p

        for lift in lifts_available:
            F = base_features[lift]
            gg = {}
            for kind in kinds:
                if kind not in pert_features[lift]:
                    continue
                gg[kind] = (np.asarray(pert_keep[kind]), pert_features[lift][kind])
            mu, sd = fit_standardizer(F)
            Z = apply_standardizer(F, mu, sd)
            pert_std: dict[str, list] = {}
            pert_rel: dict[str, list] = {}
            for kind, (keep, Fp) in gg.items():
                b = F[keep]
                pert_std[kind] = standardized_distance(b, Fp, sd).tolist()
                pert_rel[kind] = relative_change(b, Fp).tolist()
            # random size-matched pairs within the sample
            sizes = np.array([m.n for m in base_mols])
            rnd_d, rnd_r = [], []
            rr = np.random.default_rng(args.seed + 73)
            for loc in range(n_pert):
                cand = np.where(np.abs(sizes - sizes[loc]) <= 1)[0]
                cand = cand[cand != loc]
                if cand.size == 0:
                    continue
                j = int(cand[rr.integers(0, cand.size)])
                rnd_d.append(float(np.linalg.norm((Z[loc] - Z[j]) / 1.0)))
                rnd_r.append(float(np.linalg.norm(F[loc] - F[j]) / (np.linalg.norm(F[loc]) + EPS)))
            pert_std["random_pair"] = rnd_d
            pert_rel["random_pair"] = rnd_r
            perturbation[lift] = {"std": pert_std, "relative": pert_rel}
            rew_mean = float(np.mean(pert_std.get("two_switch", [np.nan])))
            rnd_mean = float(np.mean(rnd_d)) if rnd_d else float("nan")
            geometry[lift] = {
                "dim": int(F.shape[1]),
                "mean_permutation": float(np.mean(pert_std.get("permutation", [0.0]))),
                "mean_atom_sub": float(np.mean(pert_std.get("atom_sub", [np.nan]))),
                "mean_bond_type_swap": float(np.mean(pert_std.get("bond_type_swap", [np.nan]))),
                "mean_two_switch": rew_mean,
                "mean_random_pair": rnd_mean,
                "random_over_rewiring": (rnd_mean / rew_mean) if rew_mean else None,
                "relative_two_switch_mean": float(np.mean(pert_rel.get("two_switch", [np.nan]))),
                "relative_random_mean": float(np.mean(pert_rel.get("random_pair", [np.nan]))),
            }
            wl_metrics[lift] = {"dim": int(F.shape[1]), **wl_fidelity(Z, wl_sim)}
            log(f"  geo {lift}: R_topo={geometry[lift]['random_over_rewiring']:.2f} "
                f"2switch={rew_mean:.3f} random={rnd_mean:.3f} "
                f"top1WL={wl_metrics[lift]['top1_wl_sim']:.4f} "
                f"rho={wl_metrics[lift]['spearman_dphi_dwl']:.3f}")

        # learned layer-wise mechanism audit
        if learned_phi_train is not None:
            log("  learned layer-wise rewiring mechanism audit ...")
            n_mech = min(200, n_pert)
            mech_rng = np.random.default_rng(args.seed + 81)
            mech_idx = mech_rng.choice(n_pert, size=n_mech, replace=False)
            acc = {k: [] for k in ["delta_h0", "delta_e0", "delta_e1", "delta_h1",
                                   "delta_e2", "delta_h2", "delta_phi"]}
            model.eval()
            for loc in mech_idx:
                mol = base_mols[int(loc)]
                pm = two_switch(mech_rng, mol)
                if pm is None:
                    continue
                d0 = make_lift_data([mol], C_V, C_E, node_index, bond_index)
                d1 = make_lift_data([pm], C_V, C_E, node_index, bond_index)
                with torch.no_grad():
                    b0 = batch_tensors(d0, [0], args.device)
                    b1 = batch_tensors(d1, [0], args.device)
                    h0, e0, lay0 = model(b0["node_oh"], b0["bond_oh"], b0["bond_u"],
                                         b0["bond_v"], b0["inc_atom"], b0["inc_bond"])
                    h1, e1, lay1 = model(b1["node_oh"], b1["bond_oh"], b1["bond_u"],
                                         b1["bond_v"], b1["inc_atom"], b1["inc_bond"])
                    phi0 = phi_from_objects(b0, h0, e0, rff_t, D_learn).cpu().numpy()[0]
                    phi1 = phi_from_objects(b1, h1, e1, rff_t, D_learn).cpu().numpy()[0]
                pairs = [("delta_h0", 0, 0), ("delta_e0", 0, 1), ("delta_e1", 1, 1),
                         ("delta_h1", 1, 0), ("delta_e2", 2, 1), ("delta_h2", 2, 0)]
                for name, layer, which in pairs:
                    a0 = lay0[layer][which].cpu().numpy()
                    a1 = lay1[layer][which].cpu().numpy()
                    if a0.shape != a1.shape:
                        continue
                    acc[name].append(float(np.linalg.norm(a1 - a0, axis=1).mean()))
                acc["delta_phi"].append(float(np.linalg.norm(phi1 - phi0)))
            learned_mechanism = {k: summarize_vals(v) for k, v in acc.items()}
            write_json(out / "learned_mechanism.json", learned_mechanism)
            log(f"  mechanism: { {k: round(v['mean'], 4) for k, v in learned_mechanism.items() if v['n']} }")

    results = {"performance": performance_rows, "geometry": geometry,
               "wl": wl_metrics, "perturbation": perturbation,
               "learned_mechanism": learned_mechanism}

    # =====================================================================
    # Gated no-message control (only if learned beats DOI by > 0.10 valid)
    # =====================================================================
    no_message = {"triggered": False}
    learned_row = next((r for r in performance_rows if r["lift"] == "learned_incidence"), None)
    doi_row = next(r for r in performance_rows if r["lift"] == "linear_doi")
    if (learned_row is not None and not args.skip_learned
            and (doi_row["valid_mae"] - learned_row["valid_mae"]) > 0.10):
        log("[control] learned lift beats DOI by >0.10 valid; running no-message control ...")
        torch.manual_seed(0)
        nm_model = TinyIncidenceLift(C_V, C_E, H_DIM, N_ROUNDS)
        with torch.no_grad():
            for r in range(N_ROUNDS):
                nm_model.W_VE[r].zero_()
                nm_model.W_EV[r].zero_()
        nm_head = nn.Sequential(nn.Linear(3 * D_learn + 2, 64), nn.SiLU(), nn.Linear(64, 1))
        nm_ctx = sample_rff_from_sigmas({"V": H_DIM, "E": H_DIM, "I": 2 * H_DIM},
                                        learn_sigmas, LEARN_RFF_SEED_BASE)
        nm_rff_t = make_rff_tensors(nm_ctx, args.device)
        # keep W_VE/W_EV zero throughout training
        nm_res = train_learned_lift(model=nm_model, head=nm_head, rff_t=nm_rff_t,
                                    data_train=data_train, y_train=y_train,
                                    data_valid=data_valid, y_valid=y_valid,
                                    seed=0, D=D_learn, device=args.device, log=log)
        for r in range(N_ROUNDS):
            nm_res[f"W_VE[{r}]_norm"] = float(nm_model.W_VE[r].norm())
        no_message = {"triggered": True, **{k: v for k, v in nm_res.items() if k != "history"}}
        write_json(out / "no_message_control.json", no_message)
        log(f"  no-message control valid MAE={nm_res['valid_mae']:.4f} "
            f"(learned {learned_row['valid_mae']:.4f})")

    summary = {
        "provenance": provenance,
        "edge_semantics": edge_semantics,
        "performance": performance_rows,
        "scattering_metadata": scat_ctx.diagnostics,
        "scattering_rff_approximation": scat_approx,
        "doi_context": (doi_ctx.diagnostics if doi_ctx is not None else None),
        "probe_detail": probe_detail,
        "learned": {
            "metadata": json.loads((out / "learned_metadata.json").read_text(encoding="utf-8"))
            if (out / "learned_metadata.json").exists() else None,
            "probe": learned_result,
            "post_approximation": json.loads(
                (out / "learned_rff_approximation.json").read_text(encoding="utf-8"))
            if (out / "learned_rff_approximation.json").exists() else None,
        },
        "geometry": geometry,
        "wl": wl_metrics,
        "learned_mechanism": learned_mechanism,
        "no_message_control": no_message,
        "primary_T": 4,
        "wall_seconds": time.time() - t0,
    }
    write_json(out / "SUMMARY.json", summary)

    try:
        save_plots(out, results, log=log)
    except Exception as exc:  # pragma: no cover
        log(f"[plots] failed: {exc!r}")

    log("=== performance table ===")
    for r in performance_rows:
        log(f"  {r['lift']:20s} params={r['structural_params']:6d} "
            f"train={r['train_mae']:.4f} valid={r['valid_mae']:.4f}")
    log(f"=== done in {time.time() - t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
