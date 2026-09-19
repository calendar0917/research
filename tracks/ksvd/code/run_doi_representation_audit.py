#!/usr/bin/env python
"""DOI representation-domain audit — Distributional Operator-Incidence Lift.

Question (pre-dictionary; no dictionary, no ISTA/K-SVD, no learned encoder)
--------------------------------------------------------------------------
AIOM failed because ``M_t = X^T S^t X`` aggregates object-level heterogeneity
*before* any nonlinear representation.  This round keeps the *per-object*
multiscale operator response and embeds its empirical distribution, plus the
empirical atom--bond *incidence-pair* distribution:

    r_i^(T) = [ H_0[i,:] ; H_1[i,:] ; ... ; H_T[i,:] ] ,   H_t = S_G^t X_G
    q_ve    = [ r_v ; r_e ]                 (real incidence pairs)

    mu_V(G) = mean_{v in V} psi_V(r_v)
    mu_E(G) = mean_{e in E} psi_E(r_e)
    mu_I(G) = mean_{(v,e): B_ve=1} psi_I(q_ve)

    Phi_DOI(G) = [ mu_V ; mu_E ; mu_I ; log(1+n) ; log(1+m) ]

with Gaussian kernels ``k_V, k_E, k_I`` (independent train-only median-heuristic
bandwidths) approximated by Random Fourier Features.  The operator ``S_G`` and
the incidence graph are *exactly* the AIOM ones, so AIOM -> DOI differs only in
the aggregation (per-object distribution vs. global moments).

The script runs the full audit (Test A-F): permutation invariance, RFF
approximation sanity, representation geometry / near-collision, controlled
perturbation sensitivity, nearest-neighbour structural fidelity (typed 3-round
WL), multiscale contribution and a dense capacity probe (the only use of ``y``).

Discipline
----------
* Only official ZINC ``train`` (10 000) and ``valid`` (1 000) are loaded.  The
  official ``test`` split is never read, instantiated or referenced.
* ``y`` is used *only* in the capacity probe (Test F) and its controls.
* All bandwidth / normalisation / RFF decisions are train-only and label-free.
* No dictionary, ISTA, K-SVD, GNN, attention, learned structural encoder,
  learned projection or learned bandwidth is implemented.
* AIOM results are a frozen baseline; they are not retuned here.

Reuse
-----
Exact colored-incidence canonicalization, the ZINC loader, the AIOM operator,
perturbation protocol and the dense probe are reused from the previous round
(``run_aiom_representation_audit.py`` /
``run_wholegraph_canonical_registration_audit.py``).
"""

from __future__ import annotations

import argparse
import json
import math
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_aiom_representation_audit import (  # noqa: E402
    Mol,
    aiom_moments,
    apply_standardizer,
    build_codebook,
    cosine_similarity_matrix,
    fit_standardizer,
    incidence_matrix,
    iso_key_hex,
    linear_ridge,
    load_mols,
    nearest_non_iso,
    permute_mol,
    perturb_atom_substitution,
    perturb_bond_type_swap,
    phi_from_moments,
    repo_loader,
    train_probe,
    two_switch,
    upper_tri_indices,
    wl_fingerprint,
)

# ---------------------------------------------------------------------------
# Fixed schedule (no search)
# ---------------------------------------------------------------------------
T_MAX = 4                       # main representation
T_LIST = [0, 2, 4]              # prefix responses for the multiscale study
D_CANDIDATES = [512, 1024, 2048]
D_MAX_RFF = 2048
MEDIAN_PAIRS = 50_000
CAL_GRAPHS = 300
CAL_PAIRS = 2_000
RFF_SEED_BASE = 71400
EPS = 1e-12

KERNELS = ("V", "E", "I")
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/doi_representation"


# ===========================================================================
# 1. Per-object multiscale operator responses (same operator as AIOM)
# ===========================================================================
def extract_responses(mols: Sequence[Mol], C_V: int, C_E: int, node_index,
                      bond_index, T_max: int = T_MAX):
    """Return a list of ``(R_V, R_E, pair_v, pair_e)`` per molecule.

    ``R_V[:, (t*C):(t+1)*C] = H_t[atom]`` with ``H_t = S^t X`` using the AIOM
    operator ``S = D^{-1/2} A_G D^{-1/2}``.  Prefix ``T`` keeps the first
    ``(T+1)*C`` columns.  ``pair_v``/``pair_e`` enumerate the real incidences
    ``(v, e)`` (2 per undirected bond).
    """
    out = []
    for mol in mols:
        A, X = incidence_matrix(mol, C_V, C_E, node_index, bond_index)
        deg = A.sum(axis=1)
        dinv = np.zeros_like(deg)
        nz = deg > 0
        dinv[nz] = deg[nz] ** -0.5
        S = (dinv[:, None] * A) * dinv[None, :]
        H = X.copy()
        rows = []
        for t in range(T_max + 1):
            rows.append(H)
            if t < T_max:
                H = S @ H
        Hcat = np.concatenate(rows, axis=1)  # (N, (T_max+1)*C)
        n, m = mol.n, mol.m
        R_V = np.ascontiguousarray(Hcat[:n])
        R_E = np.ascontiguousarray(Hcat[n:])
        pv = np.empty(2 * m, dtype=np.int64)
        pe = np.empty(2 * m, dtype=np.int64)
        for e, (a, b) in enumerate(mol.bonds):
            pv[2 * e] = a
            pe[2 * e] = e
            pv[2 * e + 1] = b
            pe[2 * e + 1] = e
        out.append((R_V, R_E, pv, pe))
    return out


def response_dims(C: int) -> dict[int, int]:
    return {T: (T + 1) * C for T in T_LIST}


# ===========================================================================
# 2. Train-only RMS scaling + object populations
# ===========================================================================
def fit_rms_scales(resp_list, T: int, C: int, which: str):
    """RMS scale ``s_j = sqrt(E_train[r_j^2])`` for one object type."""
    d = (T + 1) * C
    sumsq = np.zeros(d, dtype=np.float64)
    count = 0
    for R_V, R_E, _, _ in resp_list:
        R = R_V if which == "V" else R_E
        sub = R[:, :d]
        sumsq += (sub * sub).sum(axis=0)
        count += sub.shape[0]
    rms = np.sqrt(sumsq / max(count, 1))
    return rms, count, d


@dataclass
class Population:
    objs: np.ndarray      # (N_total, d) scaled responses, contiguous per graph
    offsets: np.ndarray   # (G,) first row index of each graph
    counts: np.ndarray    # (G,) number of objects per graph


def build_population(resp_list, T: int, C: int, s_V: np.ndarray, s_E: np.ndarray,
                     kind: str) -> Population:
    d = (T + 1) * C
    sv_scale = s_V[:d] + EPS
    se_scale = s_E[:d] + EPS
    sv_scale = np.where(s_V[:d] == 0.0, 1.0, sv_scale)
    se_scale = np.where(s_E[:d] == 0.0, 1.0, se_scale)
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


# ===========================================================================
# 3. Train-only median-heuristic bandwidth
# ===========================================================================
def median_heuristic(objs: np.ndarray, rng: np.random.Generator,
                     max_pairs: int = MEDIAN_PAIRS) -> dict[str, Any]:
    n = objs.shape[0]
    if n < 2:
        return {"n_objects": int(n), "n_pairs": 0, "median": None, "p10": None,
                "p90": None, "sigma": None, "degenerate": True}
    ii = rng.integers(0, n, size=max_pairs)
    jj = rng.integers(0, n, size=max_pairs)
    mask = ii != jj
    ii, jj = ii[mask], jj[mask]
    dd = np.linalg.norm(objs[ii] - objs[jj], axis=1)
    med = float(np.median(dd))
    positive = dd[dd > 0.0]
    if med <= 0.0:
        # Degenerate population (e.g. T=0 one-hot: carbon is ~70% of atoms so
        # >50% of random pairs are identical).  Report it, but keep the audit
        # runnable with a documented fallback bandwidth.
        sigma = float(np.median(positive)) if positive.size else 1.0
        source = "positive_median_fallback"
    else:
        sigma = med
        source = "median"
    return {
        "n_objects": int(n),
        "n_pairs": int(dd.size),
        "n_zero_dist": int((dd == 0.0).sum()),
        "median": med,
        "p10": float(np.percentile(dd, 10)),
        "p90": float(np.percentile(dd, 90)),
        "mean": float(dd.mean()),
        "sigma": sigma,
        "sigma_source": source,
        "degenerate": bool(med <= 0.0),
    }


# ===========================================================================
# 4. Random Fourier Features
# ===========================================================================
def sample_rff(d: int, sigma: float, D: int, seed: int):
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 1.0 / sigma, size=(D, d))
    b = rng.uniform(0.0, 2.0 * np.pi, size=D)
    return omega, b


def kernel_mean_embeddings(pop: Population, omega: np.ndarray, b: np.ndarray,
                           D: int, chunk: int = 40_000):
    """Return ``(mean_embedding, sum_embedding)`` of shape ``(G, D)``."""
    factor = math.sqrt(2.0 / D)
    om = omega[:D]
    bb = b[:D]
    G = len(pop.counts)
    sums = np.zeros((G, D), dtype=np.float64)
    obj = pop.objs
    g = 0
    while g < G:
        count = 0
        g2 = g
        while g2 < G and (count + int(pop.counts[g2]) <= chunk or g2 == g):
            count += int(pop.counts[g2])
            g2 += 1
        a = int(pop.offsets[g])
        bnd = a + count
        Z = obj[a:bnd] @ om.T + bb
        psi = np.cos(Z) * factor
        rel = (pop.offsets[g:g2] - a).astype(np.int64)
        sums[g:g2] = np.add.reduceat(psi, rel, axis=0)
        g = g2
    counts = pop.counts.astype(np.float64).reshape(-1, 1)
    counts_safe = np.where(counts == 0, 1.0, counts)
    mu = np.where(counts == 0, 0.0, sums / counts_safe)
    return mu, sums


def gaussian_kernel_mean(K: np.ndarray, L: np.ndarray, sigma: float) -> float:
    kn = np.einsum("ij,ij->i", K, K)[:, None]
    ln = np.einsum("ij,ij->i", L, L)[None, :]
    d2 = kn + ln - 2.0 * (K @ L.T)
    np.maximum(d2, 0.0, out=d2)
    return float(np.exp(-d2 / (2.0 * sigma * sigma)).mean())


# ===========================================================================
# 5. Feature assembly
# ===========================================================================
def assemble(muV, muE, muI, sumV, sumE, sumI, n, m, D, D_max=D_MAX_RFF):
    k = math.sqrt(D_max / D)
    mV = muV[:, :D] * k
    mE = muE[:, :D] * k
    mI = muI[:, :D] * k
    sV = sumV[:, :D] * k
    sE = sumE[:, :D] * k
    sI = sumI[:, :D] * k
    ln = np.log1p(n.astype(np.float64)).reshape(-1, 1)
    lm = np.log1p(m.astype(np.float64)).reshape(-1, 1)
    return {
        "full": np.concatenate([mV, mE, mI, ln, lm], axis=1),
        "obj": np.concatenate([mV, mE, ln, lm], axis=1),
        "inc": np.concatenate([mI, ln, lm], axis=1),
        "sum": np.concatenate([sV, sE, sI], axis=1),
        "muV": mV, "muE": mE, "muI": mI,
        "sumV": sV, "sumE": sE, "sumI": sI,
        "n": n, "m": m,
    }


def compute_embeddings(resp_list, T: int, C: int, s_V, s_E, rffs,
                       D_max: int = D_MAX_RFF):
    popV = build_population(resp_list, T, C, s_V, s_E, "V")
    popE = build_population(resp_list, T, C, s_V, s_E, "E")
    popI = build_population(resp_list, T, C, s_V, s_E, "I")
    muV, sumV = kernel_mean_embeddings(popV, rffs["V"]["omega"], rffs["V"]["b"], D_max)
    muE, sumE = kernel_mean_embeddings(popE, rffs["E"]["omega"], rffs["E"]["b"], D_max)
    muI, sumI = kernel_mean_embeddings(popI, rffs["I"]["omega"], rffs["I"]["b"], D_max)
    n = np.array([r[0].shape[0] for r in resp_list], dtype=np.int64)
    m = np.array([r[1].shape[0] for r in resp_list], dtype=np.int64)
    return {"muV": muV, "muE": muE, "muI": muI,
            "sumV": sumV, "sumE": sumE, "sumI": sumI, "n": n, "m": m}


# ===========================================================================
# 6. RFF dimension selection (label-free, train-only calibration)
# ===========================================================================
def exact_kernel_calibration(pop: Population, cal_positions: np.ndarray,
                             sigma: float, pairs: np.ndarray):
    """Exact Gaussian-kernel mean inner product for a set of graph pairs."""
    vals = np.empty(len(pairs), dtype=np.float64)
    for p, (gi, hi) in enumerate(pairs):
        g = int(cal_positions[gi])
        h = int(cal_positions[hi])
        Kg = pop.objs[pop.offsets[g]:pop.offsets[g] + pop.counts[g]]
        Kh = pop.objs[pop.offsets[h]:pop.offsets[h] + pop.counts[h]]
        vals[p] = gaussian_kernel_mean(Kg, Kh, sigma)
    return vals


def select_dimension(cal_emb: dict, cal_positions: np.ndarray, pairs: np.ndarray,
                     exact: dict, D_candidates=D_CANDIDATES):
    """Choose the smallest D whose median abs err < 0.02 and p95 < 0.05 for all
    three kernels.  Returns (D, per-kernel metrics, flag)."""
    from scipy.stats import spearmanr

    per_kernel: dict[str, dict] = {}
    for key in KERNELS:
        mu = cal_emb[f"mu{key}"]
        ev = exact[key]
        rows = {}
        for D in D_candidates:
            k = math.sqrt(D_MAX_RFF / D)
            emb = mu[:, :D] * k
            approx = np.einsum("ij,ij->i", emb[pairs[:, 0]], emb[pairs[:, 1]])
            err = np.abs(approx - ev)
            rho = float(spearmanr(approx, ev).correlation)
            rows[D] = {
                "mean_abs_err": float(err.mean()),
                "median_abs_err": float(np.median(err)),
                "p95_abs_err": float(np.percentile(err, 95)),
                "spearman": rho,
            }
        per_kernel[key] = rows
    chosen = None
    for D in D_candidates:
        if all(per_kernel[key][D]["median_abs_err"] < 0.02
               and per_kernel[key][D]["p95_abs_err"] < 0.05 for key in KERNELS):
            chosen = D
            break
    ok = chosen is not None
    if chosen is None:
        chosen = D_candidates[-1]
    return chosen, per_kernel, ok


# ===========================================================================
# 7. Perturbation helpers (identical protocol to AIOM)
# ===========================================================================
def relative_change(base: np.ndarray, other: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(other - base, axis=1)
    den = np.linalg.norm(base, axis=1) + EPS
    return num / den


def standardized_distance(base: np.ndarray, other: np.ndarray, sd: np.ndarray):
    return np.linalg.norm((other - base) / (sd + EPS), axis=1)


# ===========================================================================
# 8. Plot helpers
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

    # (1) exact vs RFF kernel scatter (selected D) for V/E/I at T=4
    cal = results["calibration"]
    if cal:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, key in zip(axes, KERNELS):
            ev = np.asarray(cal["exact"][key])
            ap = np.asarray(cal["approx_selected"][key])
            ax.scatter(ev, ap, s=4, alpha=0.35)
            lo, hi = 0.0, max(1.0, float(ev.max()))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_title(f"{key}: D={cal['D_selected']}")
            ax.set_xlabel("exact KME")
            ax.set_ylabel("RFF KME")
        fig.suptitle("Exact vs RFF kernel mean (T=4 calibration pairs)")
        fig.tight_layout()
        fig.savefig(pdir / "exact_vs_rff_kernel.png", dpi=130)
        plt.close(fig)

    # (2) perturbation distance distributions (standardized) at T=4
    pert = results["perturbation"]
    if pert:
        kinds = ["permutation", "atom_sub", "bond_type_swap", "two_switch", "random_pair"]
        reps = ["aiom_T4", "full", "obj", "inc"]
        fig, axes = plt.subplots(1, len(reps), figsize=(3.3 * len(reps), 3.2), sharey=True)
        for ax, rep in zip(np.atleast_1d(axes), reps):
            for kind in kinds:
                vals = sorted(pert["std"][rep].get(kind, []))
                if not vals:
                    continue
                y = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, y, label=kind)
            ax.set_title(rep)
            ax.set_xlabel("standardized $\\Delta_\\Phi$")
        np.atleast_1d(axes)[0].set_ylabel("ECDF")
        np.atleast_1d(axes)[-1].legend(fontsize=6)
        fig.suptitle("Perturbation sensitivity (T=4)")
        fig.tight_layout()
        fig.savefig(pdir / "perturbation_ecdf.png", dpi=130)
        plt.close(fig)

        # rewiring vs random distance summary
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        xs = np.arange(len(reps))
        rew = [np.mean(pert["std"][rep]["two_switch"]) for rep in reps]
        rnd = [np.mean(pert["std"][rep]["random_pair"]) for rep in reps]
        ax.bar(xs - 0.2, rew, width=0.4, label="degree-preserving 2-switch")
        ax.bar(xs + 0.2, rnd, width=0.4, label="random size-matched pair")
        ax.set_xticks(xs)
        ax.set_xticklabels(reps, rotation=15)
        ax.set_ylabel("mean standardized $\\Delta_\\Phi$")
        ax.legend(fontsize=7)
        ax.set_title("Rewiring vs random-pair distance")
        fig.tight_layout()
        fig.savefig(pdir / "rewiring_vs_random.png", dpi=130)
        plt.close(fig)

    # (3) WL neighbour fidelity
    nn = results["nn_wl"]
    if nn:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        reps = list(nn.keys())
        top1 = [nn[r]["top1_wl_sim"] for r in reps]
        top10 = [nn[r]["top10_wl_sim"] for r in reps]
        rand = [nn[r]["random_control_wl_sim"] for r in reps]
        xs = np.arange(len(reps))
        ax.bar(xs - 0.25, top1, width=0.25, label="top-1")
        ax.bar(xs, top10, width=0.25, label="top-10 mean")
        ax.bar(xs + 0.25, rand, width=0.25, label="size-matched random")
        ax.set_xticks(xs)
        ax.set_xticklabels(reps, rotation=15)
        ax.set_ylim(0.8, 1.0)
        ax.set_ylabel("WL cosine similarity")
        ax.legend(fontsize=7)
        ax.set_title("Neighbour structural fidelity")
        fig.tight_layout()
        fig.savefig(pdir / "wl_neighbor_fidelity.png", dpi=130)
        plt.close(fig)

    # (4) probe MAE vs T (Full DOI) and (5) mechanism comparison
    probes = results["probes"]
    if probes:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for variant, style in (("full", "o-"), ("obj", "s--"), ("inc", "^:")):
            rows = [p for p in probes if p["variant"] == variant and p["probe"] == "mlp"]
            rows = sorted(rows, key=lambda r: r["T"])
            if rows:
                ax.plot([r["T"] for r in rows], [r["valid_mae"] for r in rows],
                        style, label=f"{variant} MLP valid")
        aiom = [p for p in probes if p["variant"] == "aiom" and p["probe"] == "mlp"]
        if aiom:
            ax.axhline(aiom[0]["valid_mae"], color="gray", ls=":", label="AIOM T=4")
        ax.set_xlabel("T")
        ax.set_ylabel("valid MAE")
        ax.legend(fontsize=7)
        ax.set_title("Dense capacity probe vs T")
        fig.tight_layout()
        fig.savefig(pdir / "probe_mae_vs_T.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        t4 = [p for p in probes if p["T"] == 4 and p["probe"] == "mlp"]
        xs = np.arange(len(t4))
        ax.bar(xs - 0.2, [p["train_mae"] for p in t4], width=0.4, label="train")
        ax.bar(xs + 0.2, [p["valid_mae"] for p in t4], width=0.4, label="valid")
        ax.set_xticks(xs)
        ax.set_xticklabels([p["variant"] for p in t4], rotation=15)
        ax.set_ylabel("MAE")
        ax.legend(fontsize=7)
        ax.set_title("T=4 mechanism comparison (MLP)")
        fig.tight_layout()
        fig.savefig(pdir / "mechanism_comparison_T4.png", dpi=130)
        plt.close(fig)

    # (6) DOI vs AIOM geometry
    geo = results["geometry"]
    if geo:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        reps = list(geo.keys())
        med = [geo[r]["nn_dist_median"] for r in reps]
        p5 = [geo[r]["nn_dist_p5"] for r in reps]
        xs = np.arange(len(reps))
        ax.bar(xs - 0.2, med, width=0.4, label="median")
        ax.bar(xs + 0.2, p5, width=0.4, label="p5")
        ax.set_xticks(xs)
        ax.set_xticklabels(reps, rotation=15)
        ax.set_ylabel("nearest non-iso standardized distance")
        ax.legend(fontsize=7)
        ax.set_title("Representation geometry")
        fig.tight_layout()
        fig.savefig(pdir / "geometry_nn_distance.png", dpi=130)
        plt.close(fig)

    log(f"[plots] saved to {pdir}")


# ===========================================================================
# 9. Orchestration
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-iso", action="store_true",
                        help="skip pynauty iso keys (geometry uses WL only)")
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

    log("=== DOI representation audit start ===")
    provenance = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "data_root": str(args.data_root),
        "seed": args.seed,
        "quick": quick,
        "T_MAX": T_MAX,
        "T_LIST": T_LIST,
        "D_candidates": D_CANDIDATES,
    }
    write_json(out / "provenance.json", provenance)
    log(f"provenance commit={provenance['git_commit']}")

    limit = args.limit if args.limit is not None else (400 if quick else None)
    train_mols = load_mols(args.data_root, "train", limit)
    valid_mols = load_mols(args.data_root, "valid", limit)
    log(f"loaded train={len(train_mols)} valid={len(valid_mols)} (test never loaded)")

    atom_cats, bond_cats = build_codebook(train_mols + valid_mols)
    C_V, C_E = len(atom_cats), len(bond_cats)
    C = C_V + C_E
    node_index = {c: i for i, c in enumerate(atom_cats)}
    bond_index = {c: i for i, c in enumerate(bond_cats)}
    dims = response_dims(C)
    log(f"codebook C_V={C_V} C_E={C_E} C={C} dims={dims}")

    # ---- edge semantics audit (assert the known semantics) ----
    n_audit = min(500, len(train_mols))
    self_loops = 0
    multi_edges = 0
    for mol in train_mols[:n_audit]:
        assert mol.n > 0
        seen: dict[tuple[int, int], int] = {}
        for (a, b) in mol.bonds:
            if a == b:
                self_loops += 1
            key = (a, b)
            seen[key] = seen.get(key, 0) + 1
        multi_edges += sum(1 for v in seen.values() if v > 1)
    assert C_E == 3, f"unexpected bond codebook C_E={C_E}"
    if args.limit is None and not quick:
        assert C_V == 21 and C_E == 3, f"unexpected codebook C_V={C_V} C_E={C_E}"
    elif C_V != 21:
        log(f"WARNING: limited run codebook C_V={C_V} C_E={C_E} "
            f"(full dataset has C_V=21,C_E=3)")
    edge_semantics = {
        "sample_graphs": n_audit,
        "C_V": C_V, "C_E": C_E, "C": C,
        "self_loops": self_loops,
        "multi_edges": multi_edges,
    }
    write_json(out / "edge_semantics.json", edge_semantics)
    log(f"edge semantics {edge_semantics}")

    # ---- responses (all graphs, T_MAX) ----
    log("extracting responses ...")
    train_resp = extract_responses(train_mols, C_V, C_E, node_index, bond_index)
    valid_resp = extract_responses(valid_mols, C_V, C_E, node_index, bond_index)

    n_train = len(train_mols)
    n_valid = len(valid_mols)
    n_all = np.array([m.n for m in train_mols] + [m.n for m in valid_mols])

    # ---- exact iso keys (pynauty colored incidence) ----
    if args.skip_iso:
        iso_train = np.arange(n_train)
        iso_valid = np.arange(n_valid)
    else:
        log("computing exact iso keys ...")
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
    iso_all = np.concatenate([iso_train, iso_valid])
    log(f"iso classes train={len(set(iso_train.tolist()))}")

    # ---- typed 3-round WL histogram (external structural metric) ----
    log("computing WL fingerprints ...")
    wl_hists = [wl_fingerprint(m, rounds=3) for m in train_mols + valid_mols]
    wl_keys = sorted({k for h in wl_hists for k in h}, key=repr)
    wl_index = {k: i for i, k in enumerate(wl_keys)}
    W = np.zeros((len(wl_hists), len(wl_keys)), dtype=np.float64)
    for r, h in enumerate(wl_hists):
        for k, v in h.items():
            W[r, wl_index[k]] = v
    wl_sim = cosine_similarity_matrix(W)

    # ---- AIOM T=4 frozen baseline ----
    log("computing AIOM T=4 baseline ...")
    triu = upper_tri_indices(C)
    aiom_dim = len(triu[0])

    def aiom_phi(mols):
        rows = []
        for mol in mols:
            moments = aiom_moments(mol, C_V, C_E, node_index, bond_index, T_max=4)
            rows.append(phi_from_moments(moments, triu))
        return np.asarray(rows, dtype=np.float64)

    aiom_train = aiom_phi(train_mols)
    aiom_valid = aiom_phi(valid_mols)
    assert aiom_train.shape[1] == 5 * aiom_dim

    # ---- size-matched random pairs ----
    rng = np.random.default_rng(args.seed)
    rand_pairs = []
    n_rand = 5_000 if quick else 20_000
    for _ in range(n_rand):
        i = int(rng.integers(0, n_all.shape[0]))
        cand = np.where(np.abs(n_all - n_all[i]) <= 1)[0]
        cand = cand[cand != i]
        if cand.size == 0:
            continue
        j = int(cand[rng.integers(0, cand.size)])
        rand_pairs.append((i, j))
    rand_pairs = np.array(rand_pairs, dtype=np.int64)

    # =====================================================================
    # Per-T pipeline: scales, populations, bandwidths, RFF, D selection
    # =====================================================================
    state: dict[int, dict] = {}
    scaling_stats: dict[str, Any] = {}
    bandwidth_stats: dict[str, Any] = {}
    rff_metadata: dict[str, Any] = {}
    approx_metrics: dict[str, Any] = {}
    calibration: dict[str, Any] = {}

    n_T = 2 if quick else len(T_LIST)
    T_run = T_LIST[:n_T] if quick else T_LIST

    for T in T_run:
        d = dims[T]
        log(f"--- T={T} (d={d}) ---")
        s_V, cnt_V, _ = fit_rms_scales(train_resp, T, C, "V")
        s_E, cnt_E, _ = fit_rms_scales(train_resp, T, C, "E")
        scaling_stats[str(T)] = {
            "d": d,
            "n_train_atoms": int(cnt_V),
            "n_train_bonds": int(cnt_E),
            "rms_V": {"min": float(s_V.min()), "max": float(s_V.max()),
                      "mean": float(s_V.mean()),
                      "n_zero": int((s_V == 0).sum()),
                      "top5": [float(x) for x in np.sort(s_V)[-5:]]},
            "rms_E": {"min": float(s_E.min()), "max": float(s_E.max()),
                      "mean": float(s_E.mean()),
                      "n_zero": int((s_E == 0).sum()),
                      "top5": [float(x) for x in np.sort(s_E)[-5:]]},
        }
        np.savez_compressed(out / f"scales_T{T}.npz", s_V=s_V, s_E=s_E)

        # train populations (for bandwidth + full embeddings)
        popV = build_population(train_resp, T, C, s_V, s_E, "V")
        popE = build_population(train_resp, T, C, s_V, s_E, "E")
        popI = build_population(train_resp, T, C, s_V, s_E, "I")

        # bandwidths (train-only median heuristic)
        rng_bw = np.random.default_rng(args.seed + 100 * T)
        bw = {}
        for key, pop in (("V", popV), ("E", popE), ("I", popI)):
            st = median_heuristic(pop.objs, rng_bw)
            bw[key] = st
            if st["degenerate"]:
                log(f"  WARNING: median heuristic is 0 for T={T} kernel={key}: "
                    f"degenerate descriptor; falling back to positive-pair median "
                    f"sigma={st['sigma']:.4f}")
        bandwidth_stats[str(T)] = bw
        log("  bandwidths " + ", ".join(
            f"{k}={bw[k]['sigma']:.4f}(med={bw[k]['median']:.4f})" for k in KERNELS))

        # RFF at max dimension (prefixes reused for D candidates)
        rffs = {}
        for key, pop in (("V", popV), ("E", popE), ("I", popI)):
            d_eff = pop.objs.shape[1]
            sigma = bw[key]["sigma"]
            seed = RFF_SEED_BASE + 1000 * T + {"V": 1, "E": 2, "I": 3}[key]
            omega, b = sample_rff(d_eff, sigma, D_MAX_RFF, seed)
            rffs[key] = {"omega": omega, "b": b, "sigma": sigma,
                         "d": d_eff, "seed": seed}
        rff_metadata[str(T)] = {
            k: {"D": D_MAX_RFF, "d": rffs[k]["d"], "sigma": rffs[k]["sigma"],
                "seed": rffs[k]["seed"]} for k in KERNELS
        }

        # full train embeddings at D_MAX
        emb_train = compute_embeddings(train_resp, T, C, s_V, s_E, rffs)

        # calibration subset for D selection (train-only, label-free)
        n_cal = min(CAL_GRAPHS, n_train)
        cal_idx = np.random.default_rng(args.seed + 7).choice(n_train, size=n_cal,
                                                              replace=False)
        cal_resp = [train_resp[i] for i in cal_idx]
        cal_V = build_population(cal_resp, T, C, s_V, s_E, "V")
        cal_E = build_population(cal_resp, T, C, s_V, s_E, "E")
        cal_I = build_population(cal_resp, T, C, s_V, s_E, "I")
        cal_positions = np.arange(n_cal, dtype=np.int64)
        n_pairs = min(CAL_PAIRS, n_cal * (n_cal - 1))
        prng = np.random.default_rng(args.seed + 11)
        pairs = np.stack([prng.integers(0, n_cal, size=n_pairs),
                          prng.integers(0, n_cal, size=n_pairs)], axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]

        exact = {}
        for key, pop in (("V", cal_V), ("E", cal_E), ("I", cal_I)):
            exact[key] = exact_kernel_calibration(pop, cal_positions,
                                                  bw[key]["sigma"], pairs)
        cal_emb = {}
        for key, pop in (("V", cal_V), ("E", cal_E), ("I", cal_I)):
            mu, _ = kernel_mean_embeddings(pop, rffs[key]["omega"],
                                           rffs[key]["b"], D_MAX_RFF)
            cal_emb[f"mu{key}"] = mu
        D_sel, per_kernel, ok = select_dimension(cal_emb, cal_positions, pairs, exact)
        log(f"  D_selected={D_sel} ok={ok}")
        approx_metrics[str(T)] = {"D_selected": D_sel, "ok": ok,
                                  "per_kernel": per_kernel}
        # store selected-D approx values for the scatter plot
        approx_selected = {}
        for key in KERNELS:
            k = math.sqrt(D_MAX_RFF / D_sel)
            emb = cal_emb[f"mu{key}"][:, :D_sel] * k
            approx_selected[key] = np.einsum(
                "ij,ij->i", emb[pairs[:, 0]], emb[pairs[:, 1]])
        if T == T_MAX or (quick and T == T_run[-1]):
            calibration = {"T": T, "D_selected": D_sel,
                           "exact": {k: exact[k].tolist() for k in KERNELS},
                           "approx_selected": {k: approx_selected[k].tolist()
                                               for k in KERNELS}}

        # features at selected D
        D = D_sel
        feats_train = assemble(emb_train["muV"], emb_train["muE"], emb_train["muI"],
                               emb_train["sumV"], emb_train["sumE"], emb_train["sumI"],
                               emb_train["n"], emb_train["m"], D)
        emb_valid = compute_embeddings(valid_resp, T, C, s_V, s_E, rffs)
        feats_valid = assemble(emb_valid["muV"], emb_valid["muE"], emb_valid["muI"],
                               emb_valid["sumV"], emb_valid["sumE"], emb_valid["sumI"],
                               emb_valid["n"], emb_valid["m"], D)

        state[T] = {
            "d": d, "s_V": s_V, "s_E": s_E, "rffs": rffs, "D": D,
            "train": {k: feats_train[k] for k in ("full", "obj", "inc", "sum")},
            "valid": {k: feats_valid[k] for k in ("full", "obj", "inc", "sum")},
            "emb_train": emb_train,
        }
        log(f"  feature dims: full={feats_train['full'].shape[1]} "
            f"obj={feats_train['obj'].shape[1]} inc={feats_train['inc'].shape[1]} "
            f"sum={feats_train['sum'].shape[1]}")

    primary_T = T_MAX if T_MAX in state else T_run[-1]
    D_primary = state[primary_T]["D"]
    log(f"primary T={primary_T} D={D_primary}")

    # save RFF metadata / approximations
    write_json(out / "response_scaling.json", scaling_stats)
    write_json(out / "bandwidth_stats.json", bandwidth_stats)
    write_json(out / "rff_metadata.json", rff_metadata)
    write_json(out / "rff_approximation.json", approx_metrics)
    if calibration:
        write_json(out / "calibration_sample.json",
                   {k: (v if k != "T" else v) for k, v in calibration.items()})

    # =====================================================================
    # Test A — permutation invariance (all three views)
    # =====================================================================
    log("[Test A] permutation invariance ...")
    n_inv = 40 if quick else 500
    n_perm = 5 if quick else 20
    inv_rng = np.random.default_rng(args.seed + 3)
    inv_idx = inv_rng.choice(n_train, size=min(n_inv, n_train), replace=False)
    variants = ("full", "obj", "inc")
    worst = {v: 0.0 for v in variants}
    checks = 0
    st = state[primary_T]
    for i in inv_idx:
        mol = train_mols[int(i)]
        base = {v: st["train"][v][i] for v in variants}
        for _ in range(n_perm):
            pa = inv_rng.permutation(mol.n)
            pb = inv_rng.permutation(mol.m) if mol.m > 0 else np.zeros(0, dtype=np.int64)
            pm = permute_mol(mol, pa, pb)
            rep = extract_responses([pm], C_V, C_E, node_index, bond_index)
            emb = compute_embeddings(rep, primary_T, C, st["s_V"], st["s_E"], st["rffs"])
            feats = assemble(emb["muV"], emb["muE"], emb["muI"],
                             emb["sumV"], emb["sumE"], emb["sumI"],
                             emb["n"], emb["m"], st["D"])
            for v in variants:
                err = float(np.max(np.abs(feats[v][0] - base[v])))
                worst[v] = max(worst[v], err)
            checks += 1
    invariance = {"checks": checks, "n_perm": n_perm,
                  "worst_abs_err": worst,
                  "pass": all(worst[v] < 1e-10 for v in variants)}
    for v in variants:
        log(f"  {v}: max abs err {worst[v]:.3e}")
    write_json(out / "invariance.json", invariance)

    # =====================================================================
    # Test C / E — geometry, near-collision, WL neighbour fidelity
    # =====================================================================
    log("[Test C/E] geometry + WL fidelity ...")

    def cap_neighbors(Z, top_k=10, chunk=256, iso=None):
        return nearest_non_iso(Z, iso_all if iso is None else iso, top_k=top_k, chunk=chunk)

    wl_dist_all = 1.0 - wl_sim
    # WL top-10 neighbours (exclude self + same iso class) for overlap
    def wl_topk(i, k=10):
        d = wl_dist_all[i].copy()
        d[iso_all == iso_all[i]] = np.inf
        idx = np.argpartition(d, min(k, len(d) - 1))[:k]
        return set(idx.tolist())

    rep_features = {
        "aiom_T4": (aiom_train, aiom_valid),
        f"full_T{primary_T}": (state[primary_T]["train"]["full"],
                               state[primary_T]["valid"]["full"]),
        f"obj_T{primary_T}": (state[primary_T]["train"]["obj"],
                              state[primary_T]["valid"]["obj"]),
        f"inc_T{primary_T}": (state[primary_T]["train"]["inc"],
                              state[primary_T]["valid"]["inc"]),
    }
    geometry: dict[str, Any] = {}
    nn_wl: dict[str, Any] = {}
    near_pairs_saved: list[dict] = []
    for name, (Ftr, Fva) in rep_features.items():
        mu, sd = fit_standardizer(Ftr)
        Z = apply_standardizer(np.concatenate([Ftr, Fva], axis=0), mu, sd)
        nni, nnd = cap_neighbors(Z, top_k=10,
                                 chunk=(128 if quick else 256))
        first = nnd[:, 0]
        first = first[np.isfinite(first)]
        geometry[name] = {
            "dim": int(Ftr.shape[1]),
            "nn_dist_min": float(np.min(first)) if first.size else None,
            "nn_dist_p1": float(np.percentile(first, 1)) if first.size else None,
            "nn_dist_p5": float(np.percentile(first, 5)) if first.size else None,
            "nn_dist_median": float(np.median(first)) if first.size else None,
            "nn_dist_p95": float(np.percentile(first, 95)) if first.size else None,
        }
        # WL fidelity of the *train-only* index ordering (train feature space)
        Ztr = Z[:n_train]
        nni_tr, nnd_tr = cap_neighbors(Ztr, top_k=10, chunk=(128 if quick else 256),
                                       iso=iso_all[:n_train])
        vm = nni_tr >= 0
        top1 = np.array([wl_sim[i, nni_tr[i, 0]] if vm[i, 0] else np.nan
                         for i in range(n_train)])
        top10 = np.array([np.nanmean([wl_sim[i, nni_tr[i, k]] for k in range(10)
                                      if vm[i, k]]) for i in range(n_train)])
        rand_wl = wl_sim[rand_pairs[:, 0], rand_pairs[:, 1]]
        sub = rand_pairs[:5000]
        dphi = np.linalg.norm(Z[sub[:, 0]] - Z[sub[:, 1]], axis=1)
        dwl = wl_dist_all[sub[:, 0], sub[:, 1]]
        from scipy.stats import spearmanr
        rho = float(spearmanr(dphi, dwl).correlation)
        # top-k overlap with WL neighbours
        overlaps = []
        for i in range(0, n_train, max(1, n_train // 500)):
            ours = set(nni_tr[i, :10][nni_tr[i, :10] >= 0].tolist())
            theirs = wl_topk(i, 10)
            if theirs:
                overlaps.append(len(ours & theirs) / 10.0)
        nn_wl[name] = {
            "dim": int(Ftr.shape[1]),
            "top1_wl_sim": float(np.nanmean(top1)),
            "top10_wl_sim": float(np.nanmean(top10)),
            "random_control_wl_sim": float(rand_wl.mean()),
            "spearman_dphi_dwl": rho,
            "top10_overlap_wl": float(np.mean(overlaps)) if overlaps else None,
        }
        log(f"  {name}: nn_med={geometry[name]['nn_dist_median']:.3f} "
            f"top1_wl={nn_wl[name]['top1_wl_sim']:.3f} "
            f"top10_wl={nn_wl[name]['top10_wl_sim']:.3f} rho={rho:.3f} "
            f"overlap={nn_wl[name]['top10_overlap_wl']}")
        if name.endswith(f"full_T{primary_T}"):
            order = np.argsort(np.where(np.isfinite(nnd_tr[:, 0]), nnd_tr[:, 0], np.inf))
            for i in order[:50]:
                if not np.isfinite(nnd_tr[i, 0]):
                    continue
                near_pairs_saved.append({
                    "i": int(i), "j": int(nni_tr[i, 0]), "dist": float(nnd_tr[i, 0]),
                    "wl_sim": float(wl_sim[i, nni_tr[i, 0]]),
                    "same_iso": bool(iso_all[i] == iso_all[nni_tr[i, 0]]),
                })
    write_json(out / "geometry_metrics.json", geometry)
    write_json(out / "nn_wl_metrics.json", nn_wl)
    with (out / "nearest_pairs_full.csv").open("w", encoding="utf-8") as fh:
        fh.write("i,j,std_dist,wl_cosine,same_iso\n")
        for r in near_pairs_saved:
            fh.write(f"{r['i']},{r['j']},{r['dist']:.6f},{r['wl_sim']:.6f},"
                     f"{int(r['same_iso'])}\n")

    # =====================================================================
    # Test D — controlled perturbation sensitivity
    # =====================================================================
    log("[Test D] perturbations ...")
    n_pert = 60 if quick else 500
    pert_rng = np.random.default_rng(args.seed + 5)
    pert_idx = pert_rng.choice(n_train, size=min(n_pert, n_train), replace=False)

    def build_perturbed(kind, mol):
        if kind == "permutation":
            pa = pert_rng.permutation(mol.n)
            pb = pert_rng.permutation(mol.m) if mol.m > 0 else np.zeros(0, dtype=np.int64)
            return permute_mol(mol, pa, pb)
        if kind == "atom_sub":
            return perturb_atom_substitution(pert_rng, mol, atom_cats)
        if kind == "bond_type_swap":
            return perturb_bond_type_swap(pert_rng, mol)
        if kind == "two_switch":
            return two_switch(pert_rng, mol)
        raise ValueError(kind)

    kinds = ["permutation", "atom_sub", "bond_type_swap", "two_switch"]
    # base features for the sampled graphs
    st = state[primary_T]
    base_feats = {
        "full": st["train"]["full"][pert_idx],
        "obj": st["train"]["obj"][pert_idx],
        "inc": st["train"]["inc"][pert_idx],
    }
    # standardizers (train-only) for each rep
    standardizers = {}
    for rep in ("full", "obj", "inc"):
        mu, sd = fit_standardizer(st["train"][rep])
        standardizers[rep] = (mu, sd)
    aiom_mu, aiom_sd = fit_standardizer(aiom_train)
    base_aiom = aiom_train[pert_idx]

    pert_std = {r: {k: [] for k in kinds + ["random_pair"]}
                for r in ("full", "obj", "inc", "aiom_T4")}
    pert_rel = {r: {k: [] for k in kinds + ["random_pair"]}
                for r in ("full", "obj", "inc", "aiom_T4")}
    construction = defaultdict(int)

    per_graph_pert = {k: [] for k in kinds}
    for loc, gi in enumerate(pert_idx):
        mol = train_mols[int(gi)]
        for kind in kinds:
            pm = build_perturbed(kind, mol)
            per_graph_pert[kind].append(pm)
            if pm is not None and kind != "permutation":
                construction[kind] += 1

    for kind in kinds:
        mols_k = [pm for pm in per_graph_pert[kind] if pm is not None]
        idx_k = [loc for loc, pm in enumerate(per_graph_pert[kind]) if pm is not None]
        if not mols_k:
            continue
        rep_resp = extract_responses(mols_k, C_V, C_E, node_index, bond_index)
        emb = compute_embeddings(rep_resp, primary_T, C, st["s_V"], st["s_E"], st["rffs"])
        feats = assemble(emb["muV"], emb["muE"], emb["muI"], emb["sumV"], emb["sumE"],
                         emb["sumI"], emb["n"], emb["m"], st["D"])
        aiom_p = aiom_phi(mols_k)
        for rep in ("full", "obj", "inc"):
            b = base_feats[rep][idx_k]
            o = feats[rep]
            mu, sd = standardizers[rep]
            pert_std[rep][kind].extend(
                standardized_distance(b, o, sd).tolist())
            pert_rel[rep][kind].extend(relative_change(b, o).tolist())
        pert_std["aiom_T4"][kind].extend(
            standardized_distance(base_aiom[idx_k], aiom_p, aiom_sd).tolist())
        pert_rel["aiom_T4"][kind].extend(
            relative_change(base_aiom[idx_k], aiom_p).tolist())

    # random size-matched pairs among the sampled graphs (train feature space)
    for rep in ("full", "obj", "inc"):
        F = st["train"][rep]
        mu, sd = standardizers[rep]
        Z = apply_standardizer(F, mu, sd)
        for loc, gi in enumerate(pert_idx):
            cand = np.where(np.abs(n_all[:n_train] - n_all[gi]) <= 1)[0]
            cand = cand[cand != gi]
            if cand.size == 0:
                continue
            j = int(cand[pert_rng.integers(0, cand.size)])
            pert_std[rep]["random_pair"].append(
                float(np.linalg.norm((Z[gi] - Z[j]) / 1.0)))
            pert_rel[rep]["random_pair"].append(
                float(np.linalg.norm(F[gi] - F[j]) / (np.linalg.norm(F[gi]) + EPS)))
    Zaiom = apply_standardizer(aiom_train, aiom_mu, aiom_sd)
    for loc, gi in enumerate(pert_idx):
        cand = np.where(np.abs(n_all[:n_train] - n_all[gi]) <= 1)[0]
        cand = cand[cand != gi]
        if cand.size == 0:
            continue
        j = int(cand[pert_rng.integers(0, cand.size)])
        pert_std["aiom_T4"]["random_pair"].append(float(np.linalg.norm(Zaiom[gi] - Zaiom[j])))
        pert_rel["aiom_T4"]["random_pair"].append(
            float(np.linalg.norm(aiom_train[gi] - aiom_train[j])
                  / (np.linalg.norm(aiom_train[gi]) + EPS)))

    def summarize(vals):
        if not vals:
            return {"n": 0}
        v = np.array(vals)
        return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
                "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                "max": float(v.max())}

    pert_summary = {"std": {}, "relative": {}}
    for metric, store in (("std", pert_std), ("relative", pert_rel)):
        for rep, kinds_d in store.items():
            pert_summary[metric][rep] = {k: summarize(v) for k, v in kinds_d.items()}
    write_json(out / "perturbation_summary.json", pert_summary)
    # ratio of random/rewiring
    ratios = {}
    for rep in pert_std:
        rew = np.mean(pert_std[rep]["two_switch"]) if pert_std[rep]["two_switch"] else np.nan
        rnd = np.mean(pert_std[rep]["random_pair"]) if pert_std[rep]["random_pair"] else np.nan
        ratios[rep] = {"mean_rewiring": float(rew), "mean_random": float(rnd),
                       "random_over_rewiring": float(rnd / rew) if rew and rew > 0 else None}
    write_json(out / "rewiring_ratio.json", ratios)
    log("  rewiring ratios: " + ", ".join(f"{r}={ratios[r]['random_over_rewiring']}"
                                          for r in ratios))

    # =====================================================================
    # Test F — multiscale contribution (rewiring + WL at T=0,2,4)
    # =====================================================================
    multiscale = {}
    for T in T_run:
        stt = state[T]
        # rewiring std distance
        mols_k = [pm for pm in per_graph_pert["two_switch"] if pm is not None]
        idx_k = [loc for loc, pm in enumerate(per_graph_pert["two_switch"]) if pm is not None]
        if mols_k:
            emb = compute_embeddings(extract_responses(mols_k, C_V, C_E, node_index, bond_index),
                                     T, C, stt["s_V"], stt["s_E"], stt["rffs"])
            feats = assemble(emb["muV"], emb["muE"], emb["muI"], emb["sumV"], emb["sumE"],
                             emb["sumI"], emb["n"], emb["m"], stt["D"])
            mu, sd = fit_standardizer(stt["train"]["full"])
            b = stt["train"]["full"][pert_idx][idx_k]
            rew = float(np.mean(standardized_distance(b, feats["full"], sd)))
        else:
            rew = None
        # WL fidelity of full DOI at this T
        Ftr = stt["train"]["full"]
        mu, sd = fit_standardizer(Ftr)
        Ztr = apply_standardizer(Ftr, mu, sd)
        nni_tr, _ = cap_neighbors(Ztr, top_k=10, chunk=(128 if quick else 256),
                                  iso=iso_all[:n_train])
        vm = nni_tr >= 0
        top1 = np.nanmean([wl_sim[i, nni_tr[i, 0]] if vm[i, 0] else np.nan
                           for i in range(n_train)])
        top10 = np.nanmean([np.nanmean([wl_sim[i, nni_tr[i, k]] for k in range(10)
                                        if vm[i, k]]) for i in range(n_train)])
        multiscale[str(T)] = {"rewiring_std": rew, "D": stt["D"],
                              "full_dim": int(Ftr.shape[1]),
                              "top1_wl_sim": float(top1), "top10_wl_sim": float(top10)}
        log(f"  T={T}: rewiring={rew} top1_wl={top1:.3f} top10_wl={top10:.3f}")
    write_json(out / "multiscale.json", multiscale)

    # =====================================================================
    # Capacity probe (only use of y)
    # =====================================================================
    probe_rows: list[dict] = []
    if not args.skip_probe:
        log("[Test F] capacity probes ...")
        load_zinc, _ = repo_loader()
        train_ds = load_zinc(args.data_root, "train")
        valid_ds = load_zinc(args.data_root, "val")
        y_train = np.array([float(train_ds[i].y.reshape(-1)[0]) for i in range(n_train)])
        y_valid = np.array([float(valid_ds[i].y.reshape(-1)[0]) for i in range(n_valid)])
        log(f"  y train mean={y_train.mean():.4f} std={y_train.std():.4f}")

        def run_probe(variant, T, features, hidden=64, seed=0, do_ridge=True):
            Ftr, Fva = features
            mu, sd = fit_standardizer(Ftr)
            Ztr = apply_standardizer(Ftr, mu, sd)
            Zva = apply_standardizer(Fva, mu, sd)
            res = {"variant": variant, "T": T, "dim": int(Ftr.shape[1])}
            lin = train_probe(Ztr, y_train, Zva, y_valid, hidden=None, seed=seed, log=log)
            mlp = train_probe(Ztr, y_train, Zva, y_valid, hidden=hidden, seed=seed, log=log)
            res["linear"] = lin
            res["mlp"] = mlp
            row = {"variant": variant, "T": T, "probe": "linear", "seed": seed,
                   "dim": int(Ftr.shape[1]), **lin}
            probe_rows.append(row)
            row = {"variant": variant, "T": T, "probe": "mlp", "seed": seed,
                   "dim": int(Ftr.shape[1]), **mlp}
            probe_rows.append(row)
            if do_ridge:
                ridge = linear_ridge(Ztr, y_train, Zva, y_valid)
                probe_rows.append({"variant": variant, "T": T, "probe": "ridge",
                                   "seed": seed, "dim": int(Ftr.shape[1]), **ridge})
            log(f"  [{variant} T={T}] linear tr/va={lin['train_mae']:.4f}/"
                f"{lin['valid_mae']:.4f} mlp tr/va={mlp['train_mae']:.4f}/"
                f"{mlp['valid_mae']:.4f} dim={Ftr.shape[1]}")
            return mlp

        # AIOM baseline T=4
        run_probe("aiom", 4, (aiom_train, aiom_valid))
        # multiscale Full DOI
        for T in T_run:
            run_probe("full", T, (state[T]["train"]["full"], state[T]["valid"]["full"]))
        # mechanism at primary T
        st = state[primary_T]
        run_probe("obj", primary_T, (st["train"]["obj"], st["valid"]["obj"]))
        run_probe("inc", primary_T, (st["train"]["inc"], st["valid"]["inc"]))
        run_probe("sum", primary_T, (st["train"]["sum"], st["valid"]["sum"]))

        write_json(out / "probe_results.json", probe_rows)

        # probe ceiling (only if Full DOI T=primary beats AIOM by >0.05)
        aiom_mlp = next((p for p in probe_rows if p["variant"] == "aiom"
                         and p["probe"] == "mlp"), None)
        full_mlp = next((p for p in probe_rows if p["variant"] == "full"
                         and p["T"] == primary_T and p["probe"] == "mlp"), None)
        ceiling = {"triggered": False}
        if aiom_mlp and full_mlp and (aiom_mlp["valid_mae"] - full_mlp["valid_mae"]) > 0.05:
            ceiling["triggered"] = True
            wide = run_probe("full_h256", primary_T,
                             (st["train"]["full"], st["valid"]["full"]),
                             hidden=256, seed=0, do_ridge=False)
            ceiling["wide_valid_mae"] = wide["valid_mae"]
            ceiling["wide_train_mae"] = wide["train_mae"]
            ceiling["improvement_over_h64"] = float(full_mlp["valid_mae"] - wide["valid_mae"])
        write_json(out / "probe_ceiling.json", ceiling)
        log(f"  ceiling check: {ceiling}")

    # =====================================================================
    # Save features + summary + plots
    # =====================================================================
    log("saving features ...")
    for T in T_run:
        st = state[T]
        np.savez_compressed(
            out / f"doi_features_T{T}.npz",
            full_train=st["train"]["full"].astype(np.float32),
            full_valid=st["valid"]["full"].astype(np.float32),
            obj_train=st["train"]["obj"].astype(np.float32),
            obj_valid=st["valid"]["obj"].astype(np.float32),
            inc_train=st["train"]["inc"].astype(np.float32),
            inc_valid=st["valid"]["inc"].astype(np.float32),
            sum_train=st["train"]["sum"].astype(np.float32),
            sum_valid=st["valid"]["sum"].astype(np.float32),
            n_train=st["emb_train"]["n"], m_train=st["emb_train"]["m"],
        )
    np.savez_compressed(out / "aiom_T4.npz",
                        aiom_train=aiom_train.astype(np.float32),
                        aiom_valid=aiom_valid.astype(np.float32))

    results = {
        "invariance": invariance,
        "geometry": geometry,
        "nn_wl": nn_wl,
        "perturbation": {"std": {r: {k: list(v) for k, v in d.items()}
                                 for r, d in pert_std.items()},
                         "relative": {r: {k: list(v) for k, v in d.items()}
                                      for r, d in pert_rel.items()}},
        "probes": probe_rows,
        "calibration": calibration,
        "multiscale": multiscale,
    }
    try:
        save_plots(out, results, log=log)
    except Exception as exc:  # pragma: no cover
        log(f"[plots] failed: {exc!r}")

    summary = {
        "provenance": provenance,
        "edge_semantics": edge_semantics,
        "scaling": scaling_stats,
        "bandwidth": bandwidth_stats,
        "rff": rff_metadata,
        "rff_approximation": approx_metrics,
        "invariance": invariance,
        "geometry": geometry,
        "nn_wl": nn_wl,
        "perturbation_summary": pert_summary,
        "rewiring_ratio": ratios,
        "multiscale": multiscale,
        "probes": probe_rows,
        "D_selected": {str(T): state[T]["D"] for T in T_run},
        "primary_T": primary_T,
        "wall_seconds": time.time() - t0,
    }
    write_json(out / "SUMMARY.json", summary)
    log(f"=== done in {time.time() - t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
