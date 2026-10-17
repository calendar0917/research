"""Real one-hop Bond relation vs a row-sum-matched incidence-free Null.

Protocol ``cscl-bond-relation-attribution-v1`` (see
``results/bond_relation_attribution_v1/PROTOCOL.md``).

Arms (all share the unchanged ``bond_random_walk_organization_v1.BondRWComposer``,
349,985 parameters, one Fit-only masked-SSL initialisation per seed and the frozen
R3 recipe):

* **R**  ``K^R = sym(P)``, ``P = (I + D^-1 A_L)/2`` on the physical-bond line
  graph (historical R1 kernel, reused through
  ``bond_rw_local_vs_multiscale_v1.k1_matrix``).
* **D**  ``s_i = sum_j K^R_ij``, ``K^D_ij = s_i s_j / sum_k s_k``.  Exactly the
  same bond objects, the same per-row weight mass and the same total weight as R,
  with no real one-hop incidence.  Dense of rank one, no new parameters.
* **U**  ``K^U_ef = 1/N_bond`` (historical uniform control, unchanged).

The module is deliberately small: the kernels and the model/training loop are
imported from the historical modules so that no refactor can change their
computational meaning.  Everything here is label-free.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bond_random_walk_organization_v1 as bwo  # noqa: E402
import bond_rw_local_vs_multiscale_v1 as lm  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bond-relation-attribution-v1"
SPLIT_TAG = "E"
SPLIT_SALT = "cscl-bond-relation-attribution-v1"
EXPECTED_PARAMETERS = 349_985
P3_EXPECTED_PARAMETERS = 413_505
SEEDS = (0, 1)

#: the four rows where the committed canonical SMILES contradicts the official
#: PyG constitution (reproduced by ``structure_context_support_v1.pairing_verification``)
EXCLUDED_ROWS = (941, 2210, 6071, 9766)
N_USABLE = 10_000 - len(EXCLUDED_ROWS)

FIT_TARGET = 7_996
MONITOR_TARGET = 1_000
LOCKED_TARGET = 1_000

#: frozen scoring constants (PROTOCOL.md §5)
MEANINGFUL_DELTA = 0.004
CASE_A_MAX_GAP_TO_P3 = 0.010
CASE_B_MAX_GAP_TO_P3 = 0.020
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
BOOTSTRAP_N = 5_000
BOOTSTRAP_SEED = 0

ARMS = ("R", "D", "U", "P3")
RDU_ARMS = ("R", "D", "U")
ZONES = ("monitor", "locked")
PRIMARY_ZONE = "locked"
REPLICATION_ZONE = "monitor"

#: historical file hashes that must remain byte-identical (PROTOCOL.md §3.5)
REUSE_HASHES = {
    "bond_random_walk_organization_v1.py": "efda1dd72ac4465583b0e913c348a949c0c3e5ec9d92c965cd0631a89e39e7f0",
    "bond_rw_local_vs_multiscale_v1.py": "6638b238e36b601f829b83cf30078926430c8f1eab8cf2dbfa0b60f735b46c81",
    "structural_relation_composer_v1.py": "1c83a97ece60a151c6fa0c1ad5d2022b470ede9356b23d8d7b1e6b62daa71f4f",
    "structural_relation_objects_v1.py": "8f4ba44dc3c047d674a1a6832f7ed8265eaa2457b15e820b3973a4cbd3a0861c",
    "bjoc_perfmech_long_v1.py": "cb9796cb45f6fe048f2fd63e96798937cdbd802b28dbad34b2aafdbef0cbafe8",
    "bjoc_arch_breakthrough_v1.py": "31294de0526215e3c998a822346d4dac5baceb24ebd986b61122f78e65d55377",
    "bjoc_masked_pretraining_v1.py": "f27dbfe36ac5a9a0a1915320dba7c955115ccfa36e6b4755f2d2d62d8efb1489",
}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def r_matrix(obj: sro.MolObjects) -> np.ndarray:
    """R kernel: the historical R1 matrix ``sym(P)`` (delegated, not reimplemented)."""
    return lm.k1_matrix(obj)


def rowsum_null_matrix(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-sum-matched, incidence-free Null ``K^D_ij = s_i s_j / sum_k s_k``.

    ``K`` must be a symmetric non-negative relation matrix.  The returned matrix
    has exactly the same row sums as ``K`` (``s``) and the same total weight
    ``sum(s)``, and is dense and rank one by construction.

    ``K^D = s s^T / sum(s)`` is the unique matrix of the form ``s a^T`` whose
    row sums are ``s``; it therefore depends on ``K`` only through ``s``.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("relation matrix must be square")
    if np.abs(K - K.T).max(initial=0.0) > 1e-9:
        raise ValueError("relation matrix must be symmetric")
    if (K < -1e-12).any():
        raise ValueError("relation matrix must be non-negative")
    s = K.sum(axis=1)
    total = float(s.sum())
    if total <= 0.0:
        return np.zeros_like(K), s
    return np.outer(s, s) / total, s


def d_matrix(obj: sro.MolObjects) -> tuple[np.ndarray, np.ndarray]:
    """D kernel for one molecule (plus its row sums)."""
    return rowsum_null_matrix(r_matrix(obj))


def u_matrix(n_bonds: int) -> np.ndarray:
    n = int(n_bonds)
    return np.full((n, n), 1.0 / n, dtype=np.float64) if n else np.zeros((0, 0), dtype=np.float64)


def kernel_triplets(K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unordered ``e <= f`` triplet storage (same convention as every arm)."""
    return bwo.unordered_triplets(K)


def r_triplets_from_objects(obj: sro.MolObjects):
    return kernel_triplets(r_matrix(obj))


def d_triplets_from_objects(obj: sro.MolObjects):
    K_D, _ = d_matrix(obj)
    return kernel_triplets(K_D)


def u_triplets_from_objects(obj: sro.MolObjects):
    """Uniform triplets through the unchanged historical helper."""
    n = int(np.asarray(obj.bond_index).reshape(-1, 2).shape[0])
    return bwo.uniform_triplets(n)


def triplets_for(arm: str, obj: sro.MolObjects):
    if arm == "R":
        return r_triplets_from_objects(obj)
    if arm == "D":
        return d_triplets_from_objects(obj)
    if arm == "U":
        return u_triplets_from_objects(obj)
    raise ValueError(f"no pair kernel for arm {arm!r}")


# ---------------------------------------------------------------------------
# kernel statistics / audits
# ---------------------------------------------------------------------------


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() <= 1e-12 or b.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def kernel_statistics(K: np.ndarray, degree: np.ndarray | None = None) -> dict[str, Any]:
    """Descriptive statistics of one symmetric non-negative relation kernel."""
    K = np.asarray(K, dtype=np.float64)
    n = int(K.shape[0])
    if n == 0:
        return {"n_bonds": 0}
    row = K.sum(axis=1)
    total = float(row.sum())
    vals = K[np.triu_indices(n, 0)]
    eig = np.linalg.eigvalsh(K)
    eig = np.sort(np.abs(eig))[::-1]
    top_share = float(eig[0] ** 2 / max(float((eig**2).sum()), 1e-30)) if eig.size else 0.0
    def _entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        if p.size == 0:
            return 0.0
        return float(-(p * np.log(p)).sum() / max(np.log(p.size), 1e-30))

    out = {
        "n_bonds": n,
        "symmetry_max_abs_err": float(np.abs(K - K.T).max()),
        "min_entry": float(K.min()),
        "rowsum_mean": float(row.mean()),
        "rowsum_std": float(row.std()),
        "rowsum_cv": float(row.std() / max(abs(row.mean()), 1e-12)),
        "rowsum_min": float(row.min()),
        "rowsum_max": float(row.max()),
        "total_weight": total,
        "nonzero_fraction": float((K > 1e-12).mean()),
        "rank": int(np.linalg.matrix_rank(K, tol=1e-9)),
        "top_eigen_share": top_share,
        "tv_rows_to_uniform": float(0.5 * np.abs(K - 1.0 / n).sum(axis=1).mean()),
        "row_normalised_entropy": float(np.mean([_entropy(K[i] / max(row[i], 1e-30)) for i in range(n)])),
        "diagonal_mass": float(np.diag(K).sum() / max(total, 1e-30)),
        "weight_q50": float(np.quantile(vals, 0.50)),
        "weight_q90": float(np.quantile(vals, 0.90)),
        "weight_q99": float(np.quantile(vals, 0.99)),
        "weight_max": float(vals.max()),
        "weight_mean": float(vals.mean()),
    }
    if degree is not None:
        out["corr_degree_rowsum"] = _safe_corr(np.asarray(degree), row)
        out["corr_degree_total_incident"] = _safe_corr(np.asarray(degree), K.sum(axis=0) + K.sum(axis=1))
    return out


def relation_audit_molecule(obj: sro.MolObjects) -> dict[str, Any]:
    """Label-free R/D/U audit for one molecule (PROTOCOL.md §3.1 / §3.3)."""
    A = bwo.line_graph_adjacency(obj.bond_index)
    deg = A.sum(axis=1)
    K_R = r_matrix(obj)
    K_D, s = rowsum_null_matrix(K_R)
    n = int(K_R.shape[0])
    K_U = u_matrix(n)
    s_sum = float(s.sum())
    outer_formula_err = float(
        np.abs(K_D - np.outer(s, s) / s_sum).max() if (n and s_sum > 0) else 0.0
    )
    row_R = K_R.sum(axis=1)
    row_D = K_D.sum(axis=1)
    row_U = K_U.sum(axis=1) if n else np.zeros(0)
    rec = {
        "mol_index": int(obj.mol_index),
        "n_bonds": n,
        "R": kernel_statistics(K_R, deg),
        "D": kernel_statistics(K_D, deg),
        "U": kernel_statistics(K_U, deg),
        "checks": {
            "row_sums_equal_R_D": float(np.abs(row_R - row_D).max()) if n else 0.0,
            "total_weight_R": float(row_R.sum()),
            "total_weight_D": float(row_D.sum()),
            "total_weight_equal": float(abs(row_R.sum() - row_D.sum())),
            "s_equals_rowsum_R": float(np.abs(s - row_R).max()) if n else 0.0,
            "rank_one_formula_err": outer_formula_err,
            "max_abs_D_minus_R": float(np.abs(K_D - K_R).max()) if n else 0.0,
            "max_abs_D_minus_U": float(np.abs(K_D - K_U).max()) if n else 0.0,
            "corr_R_D": _safe_corr(K_R.reshape(-1), K_D.reshape(-1)) if n > 1 else 1.0,
            "is_D_equal_U": bool(float(np.abs(K_D - K_U).max()) <= 1e-12) if n else True,
            "is_R_equal_U": bool(float(np.abs(K_R - K_U).max()) <= 1e-12) if n else True,
            "D_mass_on_R_zero_support": float(
                K_D[K_R <= 1e-12].sum() / max(s_sum, 1e-30)
            ) if n else 0.0,
            "R_mass_on_diagonal": float(np.diag(K_R).sum() / max(s_sum, 1e-30)) if n else 0.0,
            "U_rowsum_max_abs_err": float(np.abs(row_U - 1.0).max()) if n else 0.0,
        },
    }
    if n >= 2:
        perm = np.arange(n)[::-1].copy()
        A2 = A[np.ix_(perm, perm)]
        K2 = bwo.symmetrize(bwo.lazy_transition(A2))
        K2_D, _ = rowsum_null_matrix(K2)
        rec["checks"]["bond_permutation_D_max_abs"] = float(
            np.abs(K2_D - K_D[np.ix_(perm, perm)]).max()
        )
    else:
        rec["checks"]["bond_permutation_D_max_abs"] = 0.0
    return rec


def summarize_relation_audit(per_mol: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean(path: str) -> float:
        vals = []
        for rec in per_mol:
            a, b = path.split(".", 1)
            vals.append(float(rec[a][b]))
        return float(np.mean(vals))

    def cmax(key: str) -> float:
        return float(max(float(r["checks"][key]) for r in per_mol))

    n_ge2 = [r for r in per_mol if int(r["n_bonds"]) >= 2]
    single = [r for r in per_mol if int(r["n_bonds"]) == 1]
    agg = {
        "n_molecules": len(per_mol),
        "n_single_bond": len(single),
        "n_bonds_ge2": len(n_ge2),
        "R_rowsum_cv": mean("R.rowsum_cv"),
        "D_rowsum_cv": mean("D.rowsum_cv"),
        "R_nonzero_fraction": mean("R.nonzero_fraction"),
        "D_nonzero_fraction": mean("D.nonzero_fraction"),
        "U_nonzero_fraction": mean("U.nonzero_fraction"),
        "R_top_eigen_share": mean("R.top_eigen_share"),
        "D_top_eigen_share": mean("D.top_eigen_share"),
        "U_top_eigen_share": mean("U.top_eigen_share"),
        "R_tv_to_uniform": mean("R.tv_rows_to_uniform"),
        "D_tv_to_uniform": mean("D.tv_rows_to_uniform"),
        "R_diagonal_mass": mean("R.diagonal_mass"),
        "D_diagonal_mass": mean("D.diagonal_mass"),
        "corr_R_D": mean("checks.corr_R_D"),
        "D_mass_on_R_zero_support": mean("checks.D_mass_on_R_zero_support"),
        "corr_degree_rowsum_R": mean("R.corr_degree_rowsum"),
        "corr_degree_rowsum_D": mean("D.corr_degree_rowsum"),
        "row_sums_equal_max_err": cmax("row_sums_equal_R_D"),
        "total_weight_equal_max_err": cmax("total_weight_equal"),
        "rank_one_formula_max_err": cmax("rank_one_formula_err"),
        "max_abs_D_minus_R_over_molecules": max(
            float(r["checks"]["max_abs_D_minus_R"]) for r in per_mol
        ) if per_mol else 0.0,
        "max_abs_D_minus_R_min_over_ge2": min(
            float(r["checks"]["max_abs_D_minus_R"]) for r in n_ge2
        ) if n_ge2 else 0.0,
        "max_abs_D_minus_U_over_ge2": max(
            float(r["checks"]["max_abs_D_minus_U"]) for r in n_ge2
        ) if n_ge2 else 0.0,
        "n_D_equals_U": int(sum(1 for r in per_mol if r["checks"]["is_D_equal_U"])),
        "n_D_equals_U_ge2": int(sum(1 for r in n_ge2 if r["checks"]["is_D_equal_U"])),
        "frac_D_equals_U": float(np.mean([1.0 if r["checks"]["is_D_equal_U"] else 0.0 for r in per_mol]))
        if per_mol else 0.0,
        "frac_D_equals_U_ge2": float(np.mean([1.0 if r["checks"]["is_D_equal_U"] else 0.0 for r in n_ge2]))
        if n_ge2 else 0.0,
        "n_R_equals_U": int(sum(1 for r in per_mol if r["checks"]["is_R_equal_U"])),
        "frac_R_equals_U": float(np.mean([1.0 if r["checks"]["is_R_equal_U"] else 0.0 for r in per_mol]))
        if per_mol else 0.0,
        "bond_permutation_D_max_abs": cmax("bond_permutation_D_max_abs"),
        "U_rowsum_max_abs_err": cmax("U_rowsum_max_abs_err"),
        "D_rank_equals_1_all": bool(all(int(r["D"]["rank"]) == 1 for r in per_mol)),
        "R_rank_gt_1_all_ge2": bool(all(int(r["R"]["rank"]) > 1 for r in n_ge2)),
    }
    conclusions = {
        "row_sums_match": bool(agg["row_sums_equal_max_err"] <= 1e-10),
        "total_weight_matches": bool(agg["total_weight_equal_max_err"] <= 1e-10),
        "D_is_rank_one": bool(agg["D_rank_equals_1_all"] and agg["rank_one_formula_max_err"] <= 1e-9),
        "D_differs_from_R": bool(agg["max_abs_D_minus_R_min_over_ge2"] > 1e-6),
        "D_differs_from_U": bool(agg["max_abs_D_minus_U_over_ge2"] > 1e-6),
        "D_is_dense": bool(abs(agg["D_nonzero_fraction"] - 1.0) <= 1e-12),
        "bond_permutation_invariant": bool(agg["bond_permutation_D_max_abs"] <= 1e-9),
    }
    return {"aggregate": agg, "conclusions": conclusions}


# ---------------------------------------------------------------------------
# split construction (identity-keyed; the identity itself is computed by the driver)
# ---------------------------------------------------------------------------


def build_identity_split(
    rows: Sequence[int],
    identity_keys: Sequence[str],
    *,
    salt: str = SPLIT_SALT,
    fit_target: int = FIT_TARGET,
    monitor_target: int = MONITOR_TARGET,
) -> dict[str, Any]:
    """Deterministic identity-grouped split (same greedy rule as every prior round).

    ``identity_keys[i]`` is the verifiable molecular identity of ``rows[i]``.
    Identical identities never cross zones.  Group order is
    ``sha256(salt|identity)``; rows inside a group are sorted.
    """
    rows = [int(r) for r in rows]
    if len(rows) != len(identity_keys):
        raise ValueError("rows and identity_keys length mismatch")
    groups: dict[str, list[int]] = {}
    for r, key in zip(rows, identity_keys):
        groups.setdefault(str(key), []).append(int(r))
    order = sorted(groups, key=lambda k: hashlib.sha256(f"{salt}|{k}".encode()).hexdigest())
    fit: list[int] = []
    monitor: list[int] = []
    locked: list[int] = []
    for key in order:
        members = sorted(groups[key])
        if len(fit) < int(fit_target):
            fit.extend(members)
        elif len(monitor) < int(monitor_target):
            monitor.extend(members)
        else:
            locked.extend(members)
    side_of = {r: "fit" for r in fit}
    side_of.update({r: "monitor" for r in monitor})
    side_of.update({r: "locked" for r in locked})
    crossing = sum(1 for members in groups.values() if len({side_of[r] for r in members}) > 1)
    group_id: dict[str, int] = {key: i for i, key in enumerate(order)}
    return {
        "salt": salt,
        "n_groups": len(groups),
        "fit_rows": sorted(fit),
        "monitor_rows": sorted(monitor),
        "locked_rows": sorted(locked),
        "duplicate_groups": {k: sorted(v) for k, v in groups.items() if len(v) > 1},
        "duplicate_groups_cross_sides": int(crossing),
        "group_id": {str(r): group_id[str(key)] for r, key in zip(rows, identity_keys)},
        "identity_of_row": {int(r): str(key) for r, key in zip(rows, identity_keys)},
    }


# ---------------------------------------------------------------------------
# frozen decision logic
# ---------------------------------------------------------------------------


def contrast_label(
    delta_seed_mean: float,
    per_seed_delta: Mapping[str, float],
    ci_lo: float,
    ci_hi: float,
    *,
    threshold: float = MEANINGFUL_DELTA,
) -> str:
    """Frozen per-contrast label (PROTOCOL.md §5.1)."""
    vals = [float(v) for v in per_seed_delta.values()]
    if (
        float(delta_seed_mean) <= -float(threshold)
        and all(v < 0 for v in vals)
        and float(ci_hi) < 0
    ):
        return "GAIN"
    if (
        float(delta_seed_mean) >= float(threshold)
        and all(v > 0 for v in vals)
        and float(ci_lo) > 0
    ):
        return "LOSS"
    return "NEUTRAL"


def replication_status(monitor_per_seed: Mapping[str, float]) -> dict[str, Any]:
    """Frozen replication rule: negative seed-mean and every seed negative."""
    vals = [float(v) for v in monitor_per_seed.values()]
    mean = float(np.mean(vals)) if vals else 0.0
    return {
        "seed_mean_delta": mean,
        "per_seed_delta": {k: float(v) for k, v in monitor_per_seed.items()},
        "replicated": bool(vals and mean < 0 and all(v < 0 for v in vals)),
    }


def overall_case(
    *,
    p1_locked_label: str,
    p1_monitor: Mapping[str, float],
    p2_locked_label: str,
    p2_monitor: Mapping[str, float],
    p1_locked_delta: float,
    p1_monitor_delta: float,
    p2_locked_delta: float,
    r_minus_p3_locked: float,
    threshold: float = MEANINGFUL_DELTA,
    case_a_gap: float = CASE_A_MAX_GAP_TO_P3,
    case_b_gap: float = CASE_B_MAX_GAP_TO_P3,
) -> dict[str, Any]:
    """Frozen Case A / B / C rule (PROTOCOL.md §5.2).  Ordered, first match wins."""
    p1_rep = replication_status(p1_monitor)
    p2_rep = replication_status(p2_monitor)
    if (
        p1_locked_label == "GAIN"
        and p1_rep["replicated"]
        and p2_locked_label == "GAIN"
        and p2_rep["replicated"]
        and float(r_minus_p3_locked) <= float(case_a_gap)
    ):
        case = "CASE_A_RELATION_CANDIDATE"
        reasons = ["p1_gain_locked", "p1_replicated_monitor", "p2_gain_locked", "p2_replicated_monitor", "p3_gap_ok"]
    else:
        b: list[str] = []
        if (
            p1_locked_label == "NEUTRAL"
            and abs(float(p1_locked_delta)) < float(threshold)
            and abs(float(p1_monitor_delta)) < float(threshold)
        ):
            b.append("B1_ROWSUNM_ABSORBS")
        if float(p2_locked_delta) >= 0.0:
            b.append("B2_NO_GAIN_OVER_UNIFORM")
        if float(r_minus_p3_locked) > float(case_b_gap):
            b.append("B3_FAR_BELOW_STRONG_REFERENCE")
        if p1_locked_label == "LOSS":
            b.append("B4_D_BETTER_THAN_R")
        case = ("CASE_" + b[0]) if b else "CASE_C_INCONCLUSIVE"
        reasons = b or ["no_condition_met"]
    return {
        "case": case,
        "reasons": reasons,
        "p1_monitor_replicated": p1_rep,
        "p2_monitor_replicated": p2_rep,
        "thresholds": {
            "meaningful_delta": float(threshold),
            "case_a_max_gap_to_p3": float(case_a_gap),
            "case_b_max_gap_to_p3": float(case_b_gap),
        },
    }


# ---------------------------------------------------------------------------
# small helpers used by the driver and tests
# ---------------------------------------------------------------------------


def identity_group_signature(rows: Sequence[int], group_ids: Sequence[int]) -> dict[str, Any]:
    """Zone-independent verifiable summary of a committed split payload."""
    rows = [int(r) for r in rows]
    group_ids = [int(g) for g in group_ids]
    if len(rows) != len(group_ids):
        raise ValueError("rows/group_ids length mismatch")
    by_group: dict[int, list[int]] = {}
    for r, g in zip(rows, group_ids):
        by_group.setdefault(g, []).append(r)
    sizes = sorted((len(v) for v in by_group.values()), reverse=True)
    return {
        "n_rows": len(rows),
        "n_groups": len(by_group),
        "n_duplicate_groups": int(sum(1 for s in sizes if s > 1)),
        "max_group_size": int(sizes[0]) if sizes else 0,
        "group_sizes_digest": hashlib.sha256(
            np.asarray(sizes, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
