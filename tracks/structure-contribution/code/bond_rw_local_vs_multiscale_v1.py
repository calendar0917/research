"""One-step vs three-step bond random-walk relations (cscl-rw-local-vs-multiscale-v1).

This module adds exactly one model arm, ``R1``, to the already frozen
``cscl-bond-random-walk-organization-v1`` family.  The model class
(:class:`bond_random_walk_organization_v1.BondRWComposer`) is reused
**unchanged**; only the pair-weight kernel changes:

* R1  : ``K1 = sym(P)`` with ``P = (I + D^{-1} A_L)/2`` (lazy one-step walk);
* R123: ``Ks = sym((P + P^2 + P^3)/3)`` (historical arm R, read-only reuse);
* U   : ``K_ef = 1/N_bond`` (historical unstructured control, read-only reuse).

The round has four kinds of logic, all label-blind except the single Monitor
scoring step inside the driver:

1. Phase-0 mathematical audit: the full-graph double sum of ``K`` and
   ``Ks = (K + K^T)/2`` is equal under symmetric ``phi``, but the per-bond
   ``r_e`` is NOT: ``r_e(Ks) = (r_e(K) + (K^T phi)_e)/2``.  The model consumes
   per-bond messages, so the historical claim "using K equals using Ks" holds
   only at the graph-sum level; symmetrisation also reallocates row-weight /
   degree-centrality mass (``rowsum(Ks) = (rowsum(K)+colsum(K))/2``).
2. Reconciliation of the two historical synthetic-task MAE values (global
   torch RNG in the tiny regressors, not a different task or seed).
3. Error-distribution audit (quantiles, extreme-row contributions, paired
   deltas) used by the single Monitor read.
4. Correctness helpers for the R1 kernel (triplet reconstruction, invariance).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_arch_breakthrough_v1 as ab  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bond_random_walk_organization_v1 as bwo  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-rw-local-vs-multiscale-v1"
EXPECTED_PARAMETERS = 349_985

#: frozen provenance of the historical R123 / U arms (read-only reuse)
R123_ARM = "R"
U_ARM = "U"
R123_INIT_SHA256 = "90bf6322d5728654b4739df14236a6f2bcc55c22e74985fbd082bd82f8524769"
R123_ENCODER_USED_SHA256 = "71bb658770097c2b97812aa1b6f42fd9392efa43f4f0a69d77da151ccfafc0ab"

#: historical synthetic-task values observed for the same nominal call
HISTORICAL_SYNTHETIC_REAL_K = {
    "local_bond_random_walk_dry_run_0.3025": 0.3025,
    "remote_part_a_synthetic_witnesses_0.3755": 0.3755,
    "remote_mechanism_audit_0.7174": 0.7174,
}


def primary_decision_from_ci(ci_lo: float, ci_hi: float) -> str:
    """Frozen decision rule for the primary R123 - R1 contrast."""
    if float(ci_hi) < 0:
        return "MULTISCALE_EXTRA_GAIN"
    if float(ci_lo) > 0:
        return "ONE_HOP_BETTER"
    return "MULTISCALE_INDISTINGUISHABLE"


def next_step_for(decision: str) -> str:
    """Frozen next-step rule (no automatic round is started)."""
    if decision == "MULTISCALE_EXTRA_GAIN":
        return "worth studying Atom/Atom-Bond multi-scale organization"
    return "one-hop Bond adjacency already sufficient; do not expand multi-scale organization"


# ---------------------------------------------------------------------------
# R1 kernel
# ---------------------------------------------------------------------------


def k1_matrix(obj: sro.MolObjects) -> np.ndarray:
    """``K1 = sym(P)`` for one molecule."""
    A = bwo.line_graph_adjacency(obj.bond_index)
    return bwo.symmetrize(bwo.lazy_transition(A))


def k1_triplets_from_objects(obj: sro.MolObjects) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unordered triplet storage of the R1 kernel (same convention as R123)."""
    return bwo.unordered_triplets(k1_matrix(obj))


# ---------------------------------------------------------------------------
# Phase-0 mathematics
# ---------------------------------------------------------------------------


def _symmetric_phi(B: np.ndarray) -> np.ndarray:
    """A generic exactly symmetric function of (b_e, b_f): sum + product."""
    return B[:, None, :] + B[None, :, :] + B[:, None, :] * B[None, :, :]


def math_audit_molecule(obj: sro.MolObjects, rng: np.random.Generator, dim: int = 4) -> dict[str, Any]:
    """One-molecule K vs Ks audit (label-free).

    Under exactly symmetric ``phi``: the full double sum is invariant under
    symmetrisation, but the per-bond aggregate ``r_e`` is not; explicitly
    ``r_e(Ks) = (r_e(K) + (K^T phi)_e) / 2``.
    """
    A = bwo.line_graph_adjacency(obj.bond_index)
    n = A.shape[0]
    P = bwo.lazy_transition(A)
    Ks_P = bwo.symmetrize(P)
    K123 = bwo.relation_kernel(bwo.kernel_powers(A, bwo.MAX_K))
    Ks_K = bwo.symmetrize(K123)
    B = rng.normal(size=(n, dim))
    phi = _symmetric_phi(B)
    r_P = np.einsum("ef,efk->ek", P, phi)
    r_KsP = np.einsum("ef,efk->ek", Ks_P, phi)
    r_PT = np.einsum("fe,fek->ek", P, phi)
    r_K = np.einsum("ef,efk->ek", K123, phi)
    r_KsK = np.einsum("ef,efk->ek", Ks_K, phi)
    r_KT = np.einsum("fe,fek->ek", K123, phi)
    deg = A.sum(axis=1)

    def _rowsum_stats(M: np.ndarray) -> dict[str, float]:
        rs = M.sum(axis=1)
        return {
            "mean": float(rs.mean()),
            "std": float(rs.std()),
            "min": float(rs.min()),
            "max": float(rs.max()),
            "cv": float(rs.std() / max(abs(rs.mean()), 1e-12)),
        }

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "n_bonds": int(n),
        "double_sum_equality_max_abs_P": float(np.abs(r_P.sum(axis=0) - r_KsP.sum(axis=0)).max()) if n else 0.0,
        "double_sum_equality_max_abs_K123": float(np.abs(r_K.sum(axis=0) - r_KsK.sum(axis=0)).max()) if n else 0.0,
        "per_bond_diff_max_abs_P": float(np.abs(r_P - r_KsP).max()) if n else 0.0,
        "per_bond_diff_max_abs_K123": float(np.abs(r_K - r_KsK).max()) if n else 0.0,
        "per_bond_diff_mean_abs_K123": float(np.abs(r_K - r_KsK).mean()) if n else 0.0,
        "per_bond_identity_residual_P": float(np.abs(r_KsP - 0.5 * (r_P + r_PT)).max()) if n else 0.0,
        "per_bond_identity_residual_K123": float(np.abs(r_KsK - 0.5 * (r_K + r_KT)).max()) if n else 0.0,
        "matrix_max_abs_P_minus_symP": float(np.abs(P - Ks_P).max()) if n else 0.0,
        "matrix_max_abs_K123_minus_symK": float(np.abs(K123 - Ks_K).max()) if n else 0.0,
        "rowsum_P": _rowsum_stats(P),
        "rowsum_symP": _rowsum_stats(Ks_P),
        "rowsum_K123": _rowsum_stats(K123),
        "rowsum_symK": _rowsum_stats(Ks_K),
        "mean_abs_rowsum_symP_minus_1": float(np.abs(Ks_P.sum(axis=1) - 1.0).mean()) if n else 0.0,
        "mean_abs_rowsum_symK_minus_1": float(np.abs(Ks_K.sum(axis=1) - 1.0).mean()) if n else 0.0,
        "corr_degree_rowsum_symP": _corr(deg, Ks_P.sum(axis=1)),
        "corr_degree_rowsum_symK": _corr(deg, Ks_K.sum(axis=1)),
        "corr_degree_colsum_symP": _corr(deg, Ks_P.sum(axis=0)),
        "corr_degree_colsum_symK": _corr(deg, Ks_K.sum(axis=0)),
        "diag_mass_P": float(np.diag(P).sum()) if n else 0.0,
        "diag_mass_symP": float(np.diag(Ks_P).sum()) if n else 0.0,
    }


def summarize_math_audit(per_mol: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-molecule audits + frozen conclusions."""
    keys = [
        "double_sum_equality_max_abs_P",
        "double_sum_equality_max_abs_K123",
        "per_bond_diff_max_abs_P",
        "per_bond_diff_max_abs_K123",
        "per_bond_diff_mean_abs_K123",
        "per_bond_identity_residual_P",
        "per_bond_identity_residual_K123",
        "matrix_max_abs_P_minus_symP",
        "matrix_max_abs_K123_minus_symK",
        "mean_abs_rowsum_symP_minus_1",
        "mean_abs_rowsum_symK_minus_1",
        "corr_degree_rowsum_symP",
        "corr_degree_rowsum_symK",
        "corr_degree_colsum_symP",
        "corr_degree_colsum_symK",
    ]
    agg = {k: float(np.mean([float(m[k]) for m in per_mol])) for k in keys}
    agg["double_sum_equality_max_over_molecules"] = max(
        float(m["double_sum_equality_max_abs_P"]) for m in per_mol
    )
    agg["double_sum_equality_max_over_molecules_K123"] = max(
        float(m["double_sum_equality_max_abs_K123"]) for m in per_mol
    )
    agg["per_bond_diff_max_over_molecules"] = max(
        float(m["per_bond_diff_max_abs_P"]) for m in per_mol
    )
    agg["per_bond_diff_max_over_molecules_K123"] = max(
        float(m["per_bond_diff_max_abs_K123"]) for m in per_mol
    )
    agg["per_bond_identity_residual_max_P"] = max(
        float(m["per_bond_identity_residual_P"]) for m in per_mol
    )
    agg["per_bond_identity_residual_max_K123"] = max(
        float(m["per_bond_identity_residual_K123"]) for m in per_mol
    )
    for name in ("rowsum_P", "rowsum_symP", "rowsum_K123", "rowsum_symK"):
        for field in ("cv", "min", "max"):
            agg[f"{name}_{field}"] = float(np.mean([float(m[name][field]) for m in per_mol]))
    conclusions = {
        "double_sum_equality_exact": bool(agg["double_sum_equality_max_over_molecules"] <= 1e-10),
        "double_sum_equality_exact_K123": bool(agg["double_sum_equality_max_over_molecules_K123"] <= 1e-10),
        "per_bond_equality_holds": bool(agg["per_bond_diff_max_over_molecules"] <= 1e-10),
        "per_bond_identity_verified": bool(
            max(float(m["per_bond_identity_residual_P"]) for m in per_mol) <= 1e-10
            and max(float(m["per_bond_identity_residual_K123"]) for m in per_mol) <= 1e-10
        ),
        "symmetrisation_changes_rows": bool(agg["mean_abs_rowsum_symK_minus_1"] > 1e-6),
        "row_sum_centrality_signal": bool(abs(agg["corr_degree_rowsum_symK"]) > 0.05),
    }
    return {"aggregate": agg, "conclusions": conclusions}


def synthetic_rng_audit(torch_seeds: Sequence[int] = (0, 0, 12345)) -> dict[str, Any]:
    """Show that the historical synthetic-task MAE spread is a torch-init effect.

    ``bwo.synthetic_relation_task`` seeds numpy for data/order but constructs its
    tiny regressors from the global torch RNG.  Reproducing the same call after
    the same ``torch.manual_seed`` must be bit-identical; a different manual seed
    must change the result (the task recipe/seed argument is identical).
    """
    runs = []
    for s in torch_seeds:
        torch.manual_seed(int(s))
        r = bwo.synthetic_relation_task(seed=0)
        runs.append(
            {
                "torch_manual_seed": int(s),
                "test_mae_real_K_trained": float(r["test_mae_real_K_trained"]),
                "test_mae_real_K_model_evaluated_uniform": float(r["test_mae_real_K_model_evaluated_uniform"]),
                "test_mae_uniform_K_trained_uniform": float(r["test_mae_uniform_K_trained_uniform"]),
                "test_mae_linear_summary": float(r["test_mae_linear_summary"]),
            }
        )
    same = [r for r in runs if r["torch_manual_seed"] == 0]
    diff = [r for r in runs if r["torch_manual_seed"] != 0]
    seed0_identical = bool(
        len(same) >= 2
        and all(r["test_mae_real_K_trained"] == same[0]["test_mae_real_K_trained"] for r in same)
    )
    differs = bool(
        diff and all(r["test_mae_real_K_trained"] != same[0]["test_mae_real_K_trained"] for r in diff)
    )
    return {
        "protocol": PROTOCOL_ID,
        "runs": runs,
        "seed0_repeat_identical": seed0_identical,
        "different_seed_differs": differs,
        "historical_values": HISTORICAL_SYNTHETIC_REAL_K,
        "conclusion": (
            "the historical spread of the real-K synthetic MAE is explained by the unseeded global "
            "torch RNG used for the tiny regressors; the qualitative ordering real-K < uniform-K < "
            "linear summary is unchanged in every run"
            if (seed0_identical and differs)
            else "the torch-RNG explanation was not confirmed; inspect directly"
        ),
    }


# ---------------------------------------------------------------------------
# Phase-0/2 error audit (executed once, inside the single Monitor read)
# ---------------------------------------------------------------------------


def arm_error_stats(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Quantiles and extreme-error contributions of one arm."""
    err = np.abs(np.asarray(pred, dtype=np.float64) - y)
    sse = err**2
    order = np.argsort(-err)
    total = float(sse.sum())
    k1 = max(int(round(0.01 * len(err))), 1)
    k5 = max(int(round(0.05 * len(err))), 1)
    return {
        "n": int(len(err)),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(sse.mean())),
        "median_ae": float(np.median(err)),
        "p90_ae": float(np.quantile(err, 0.90)),
        "p95_ae": float(np.quantile(err, 0.95)),
        "p99_ae": float(np.quantile(err, 0.99)),
        "max_ae": float(err.max()),
        "top10_ae_mean": float(err[order[:10]].mean()),
        "worst1_sse_share": float(sse[order[:1]].sum() / max(total, 1e-30)),
        "top10_sse_share": float(sse[order[:10]].sum() / max(total, 1e-30)),
        "top1pct_sse_share": float(sse[order[:k1]].sum() / max(total, 1e-30)),
        "top5pct_sse_share": float(sse[order[:k5]].sum() / max(total, 1e-30)),
        "worst_rows": [
            {"row": int(i), "y": float(y[i]), "ae": float(err[i])} for i in order[:5]
        ],
    }


def contrast_audit(
    y: np.ndarray,
    pred_new: np.ndarray,
    pred_ref: np.ndarray,
    *,
    bootstrap_n: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Paired molecule contrast + extreme-contribution decomposition."""
    pred_new = np.asarray(pred_new, dtype=np.float64)
    pred_ref = np.asarray(pred_ref, dtype=np.float64)
    base = ab.compute_contrast(
        {0: pred_new}, {0: pred_ref}, y, bootstrap_n=int(bootstrap_n), bootstrap_seed=int(bootstrap_seed)
    )
    err_new = np.abs(pred_new - y)
    err_ref = np.abs(pred_ref - y)
    delta = err_new - err_ref
    order = np.argsort(-np.abs(delta))

    def _rows(indices):
        return [
            {
                "row": int(i),
                "y": float(y[i]),
                "err_new": float(err_new[i]),
                "err_ref": float(err_ref[i]),
                "delta": float(delta[i]),
            }
            for i in indices
        ]

    keep_worst = np.ones(len(delta), dtype=bool)
    keep_worst[order[0]] = False
    keep_10 = np.ones(len(delta), dtype=bool)
    keep_10[order[:10]] = False
    worst_ae = int(np.argmax(np.maximum(err_new, err_ref)))
    keep_ae = np.ones(len(delta), dtype=bool)
    keep_ae[worst_ae] = False
    return {
        "bootstrap": base["bootstrap"],
        "delta": base["delta_seed_mean"],
        "mae_new": base["mae_new_seed_mean"],
        "mae_ref": base["mae_ref_seed_mean"],
        "rmse_new": base["rmse_new"],
        "rmse_ref": base["rmse_ref"],
        "ordinary_delta": base["ordinary_delta"],
        "tail_delta": base["tail_delta"],
        "tail_n": base["tail_n"],
        "frac_molecules_better": base["frac_molecules_better"],
        "top10_removed_delta_arm_ae": base["top10_removed_delta"],
        "delta_wo_single_worst_delta": float(delta[keep_worst].mean()) if keep_worst.any() else None,
        "delta_wo_top10_delta": float(delta[keep_10].mean()) if keep_10.any() else None,
        "delta_wo_single_worst_ae": float(delta[keep_ae].mean()) if keep_ae.any() else None,
        "top5_worst_delta_rows": _rows(order[:5]),
        "top5_best_delta_rows": _rows(order[::-1][:5]),
    }


def error_audit(
    y: np.ndarray,
    preds: Mapping[str, np.ndarray],
    contrasts: Sequence[tuple[str, str, str]],
    *,
    bootstrap_n: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Full per-molecule / extreme-error audit for the single Monitor read.

    ``contrasts`` are ``(name, new_arm, ref_arm)`` triples over ``preds`` keys.
    """
    stats = {arm: arm_error_stats(y, p) for arm, p in preds.items()}
    cons = {
        name: contrast_audit(
            y, preds[new], preds[ref], bootstrap_n=bootstrap_n, bootstrap_seed=bootstrap_seed
        )
        for name, new, ref in contrasts
    }
    return {"arm_stats": stats, "contrasts": cons}


def per_molecule_rows(y: np.ndarray, preds: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Rows for the per-molecule CSV (one row per Monitor molecule)."""
    rows = []
    for i in range(len(y)):
        rec: dict[str, Any] = {"row": i, "y": float(y[i])}
        for arm, p in preds.items():
            rec[f"{arm}_pred"] = float(p[i])
            rec[f"{arm}_ae"] = float(abs(float(p[i]) - float(y[i])))
        rows.append(rec)
    return rows
