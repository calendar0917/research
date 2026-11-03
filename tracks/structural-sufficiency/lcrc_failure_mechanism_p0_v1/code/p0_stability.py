"""Seed stability and tail-error diagnostics on Lipophilicity random122.

All tail definitions are anchored on BASE: a molecule belongs to the tail of a
given seed if its BASE absolute error is in that seed's top 10%.  Candidate
models are never used to define the tail.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from p0_common import BOOTSTRAP_SEED, N_BOOT, TOP_FRACTION
from p0_sources import Group
from p0_stats import (
    ClusterIndex,
    bootstrap_summary,
    cluster_bootstrap,
    overlap_summary,
    sign_agreement,
    top_error_indices,
    metrics,
)

SEEDS = (0, 1, 2)


def _matrix(group: Group, split: str, arm: str, seeds: Sequence[int] = SEEDS) -> np.ndarray:
    suffix = "_test" if split == "test" else ""
    rows = []
    for seed in seeds:
        key = f"{arm}_s{seed}{suffix}"
        if key not in group.predictions:
            raise KeyError(f"missing prediction {key}")
        rows.append(group.predictions[key].pred)
    return np.vstack(rows)


def _subset_metrics(y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    return metrics(y[idx], pred[idx])


def _tail_metrics(
    y: np.ndarray, base_pred: np.ndarray, other_pred: np.ndarray, *, fraction: float = TOP_FRACTION
) -> dict[str, Any]:
    idx = top_error_indices(np.abs(base_pred - y), fraction)
    return {
        "n": int(len(idx)),
        "definition": f"top {int(round(100 * fraction))}% of BASE absolute error within this seed",
        "base": metrics(y[idx], base_pred[idx]),
        "candidate": metrics(y[idx], other_pred[idx]),
        "delta_rmse_candidate_minus_base": float(
            metrics(y[idx], other_pred[idx])["rmse"] - metrics(y[idx], base_pred[idx])["rmse"]
        ),
        "delta_mae_candidate_minus_base": float(
            metrics(y[idx], other_pred[idx])["mae"] - metrics(y[idx], base_pred[idx])["mae"]
        ),
        "tail_indices": idx.tolist(),
    }


def _seed_mean_metrics(split_metrics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = ("rmse", "mae", "p90_abs", "sse", "mse")
    out = {}
    for key in keys:
        values = [float(m[key]) for m in split_metrics]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_sd"] = float(np.std(values, ddof=0))
        out[f"{key}_min"] = float(np.min(values))
        out[f"{key}_max"] = float(np.max(values))
    return out


def analyse_split(
    group: Group,
    split: str,
    scaffolds: Sequence[str],
    *,
    n_boot: int = N_BOOT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = np.asarray(group.labels[split], dtype=np.float64)
    n = len(y)
    base = _matrix(group, split, "base")
    combo = _matrix(group, split, "combo_chem_contrib")
    if base.shape != combo.shape or base.shape[1] != n:
        raise ValueError("base/combo matrices are not aligned")

    cluster = ClusterIndex.from_values(scaffolds)
    if cluster.n != n:
        raise ValueError("scaffold vector length does not match split size")

    residual_base = base - y[None, :]
    residual_combo = combo - y[None, :]
    se_base = residual_base**2
    se_combo = residual_combo**2
    ae_base = np.abs(residual_base)
    ae_combo = np.abs(residual_combo)

    per_seed_metrics = {
        "base": {f"s{seed}": metrics(y, base[i]) for i, seed in enumerate(SEEDS)},
        "combo_chem_contrib": {f"s{seed}": metrics(y, combo[i]) for i, seed in enumerate(SEEDS)},
    }
    seed_mean = {
        "base": _seed_mean_metrics([per_seed_metrics["base"][f"s{s}"] for s in SEEDS]),
        "combo_chem_contrib": _seed_mean_metrics(
            [per_seed_metrics["combo_chem_contrib"][f"s{s}"] for s in SEEDS]
        ),
    }
    delta_by_seed: dict[str, Any] = {}
    for i, seed in enumerate(SEEDS):
        b = per_seed_metrics["base"][f"s{seed}"]
        c = per_seed_metrics["combo_chem_contrib"][f"s{seed}"]
        delta_by_seed[f"s{seed}"] = {
            "delta_rmse_combo_minus_base": float(c["rmse"] - b["rmse"]),
            "delta_mae_combo_minus_base": float(c["mae"] - b["mae"]),
            "delta_sse_combo_minus_base": float(c["sse"] - b["sse"]),
            "delta_p90_abs_combo_minus_base": float(c["p90_abs"] - b["p90_abs"]),
            "both_better": bool(c["rmse"] < b["rmse"] and c["mae"] < b["mae"]),
        }
    delta_rmse_values = [delta_by_seed[f"s{s}"]["delta_rmse_combo_minus_base"] for s in SEEDS]
    delta_mae_values = [delta_by_seed[f"s{s}"]["delta_mae_combo_minus_base"] for s in SEEDS]
    seed_mean_delta = {
        "delta_rmse_mean_over_seeds": float(np.mean(delta_rmse_values)),
        "delta_rmse_sd_over_seeds": float(np.std(delta_rmse_values, ddof=0)),
        "delta_mae_mean_over_seeds": float(np.mean(delta_mae_values)),
        "delta_mae_sd_over_seeds": float(np.std(delta_mae_values, ddof=0)),
        "n_seeds_combo_better_rmse": int(sum(v < 0 for v in delta_rmse_values)),
        "n_seeds_combo_better_mae": int(sum(v < 0 for v in delta_mae_values)),
        "direction_reversal": bool(
            any(v < 0 for v in delta_rmse_values) and any(v > 0 for v in delta_rmse_values)
        ),
    }

    # per-molecule
    rows = []
    base_tail_sets = []
    for i, seed in enumerate(SEEDS):
        tail_idx = top_error_indices(ae_base[i], TOP_FRACTION)
        base_tail_sets.append(set(int(x) for x in tail_idx))
    for j in range(n):
        sign = sign_agreement(residual_base[:, j])
        rows.append(
            {
                "group": group.key,
                "split": split,
                "exposure": "FROZEN_VALIDATION_ONLY"
                if split == "valid"
                else "POST_HOC_EXPOSED_DISCOVERY",
                "row_id": int(group.rows[split][j]),
                "y": float(y[j]),
                "mean_pred_base": float(np.mean(base[:, j])),
                "mean_signed_residual_base": float(np.mean(residual_base[:, j])),
                "mean_squared_error_base": float(np.mean(se_base[:, j])),
                "pred_std_base": float(np.std(base[:, j], ddof=0)),
                "sign_agreement_base": sign["sign_agreement"],
                "sign_agreement_count_base": sign["sign_agreement_count"],
                "all_same_sign_base": sign["all_same_sign"],
                "mean_squared_error_combo": float(np.mean(se_combo[:, j])),
                "pred_std_combo": float(np.std(combo[:, j], ddof=0)),
                "sign_agreement_combo": sign_agreement(residual_combo[:, j])["sign_agreement"],
                "mean_delta_se_combo_minus_base": float(np.mean(se_combo[:, j] - se_base[:, j])),
                "mean_delta_ae_combo_minus_base": float(np.mean(ae_combo[:, j] - ae_base[:, j])),
                "n_seeds_in_base_top10": int(sum(j in s for s in base_tail_sets)),
            }
        )
        for i, seed in enumerate(SEEDS):
            rows[-1][f"pred_base_s{seed}"] = float(base[i, j])
            rows[-1][f"pred_combo_s{seed}"] = float(combo[i, j])
            rows[-1][f"se_base_s{seed}"] = float(se_base[i, j])
            rows[-1][f"se_combo_s{seed}"] = float(se_combo[i, j])
            rows[-1][f"delta_se_combo_minus_base_s{seed}"] = float(se_combo[i, j] - se_base[i, j])
            rows[-1][f"delta_ae_combo_minus_base_s{seed}"] = float(ae_combo[i, j] - ae_base[i, j])
            rows[-1][f"in_base_top10_s{seed}"] = bool(j in base_tail_sets[i])
    per_molecule = pd.DataFrame(rows)

    overlap = overlap_summary([sorted(s) for s in base_tail_sets], n)
    overlap.pop("appearance_counts", None)

    tail_subset: dict[str, Any] = {}
    for i, seed in enumerate(SEEDS):
        tail_subset[f"s{seed}"] = _tail_metrics(y, base[i], combo[i])

    stable_tail_idx = np.asarray(
        [j for j in range(n) if sum(j in s for s in base_tail_sets) >= 2], dtype=np.int64
    )
    stable_tail = {
        "definition": "molecules in the BASE top-10% tail in at least 2 of 3 seeds",
        "n": int(len(stable_tail_idx)),
        "fraction_of_split": float(len(stable_tail_idx) / n),
        "per_seed_metrics": {
            f"s{seed}": {
                "base": metrics(y[stable_tail_idx], base[i][stable_tail_idx]),
                "combo_chem_contrib": metrics(y[stable_tail_idx], combo[i][stable_tail_idx]),
            }
            for i, seed in enumerate(SEEDS)
        }
        if len(stable_tail_idx)
        else {},
    }

    # bootstrap: paired combo - base, cluster = scaffold
    def _compute(idx: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for i, seed in enumerate(SEEDS):
            b = metrics(y[idx], base[i][idx])
            c = metrics(y[idx], combo[i][idx])
            out[f"s{seed}_delta_rmse"] = c["rmse"] - b["rmse"]
            out[f"s{seed}_delta_mae"] = c["mae"] - b["mae"]
            out[f"s{seed}_delta_sse"] = c["sse"] - b["sse"]
            k = max(1, int(round(TOP_FRACTION * len(idx))))
            tail = np.argsort(-np.abs(base[i][idx] - y[idx]), kind="stable")[:k]
            out[f"s{seed}_base_tail_delta_rmse"] = float(
                metrics(y[idx][tail], combo[i][idx][tail])["rmse"]
                - metrics(y[idx][tail], base[i][idx][tail])["rmse"]
            )
            out[f"s{seed}_base_tail_delta_mae"] = float(
                metrics(y[idx][tail], combo[i][idx][tail])["mae"]
                - metrics(y[idx][tail], base[i][idx][tail])["mae"]
            )
        rmse_deltas = [out[f"s{s}_delta_rmse"] for s in SEEDS]
        mae_deltas = [out[f"s{s}_delta_mae"] for s in SEEDS]
        out["seed_mean_delta_rmse"] = float(np.mean(rmse_deltas))
        out["seed_mean_delta_mae"] = float(np.mean(mae_deltas))
        return out

    point = _compute(np.arange(n))
    draws = cluster_bootstrap(cluster, _compute, n_boot=n_boot, seed=BOOTSTRAP_SEED)
    bootstrap = bootstrap_summary(point, draws)

    payload = {
        "split": split,
        "exposure": "FROZEN_VALIDATION_ONLY"
        if split == "valid"
        else "POST_HOC_EXPOSED_DISCOVERY",
        "n": int(n),
        "seeds": list(SEEDS),
        "tail_fraction": TOP_FRACTION,
        "tail_definition": "per seed, top 10% of BASE absolute error",
        "overall_metrics": per_seed_metrics,
        "seed_mean_metrics": seed_mean,
        "delta_by_seed": delta_by_seed,
        "seed_mean_delta": seed_mean_delta,
        "per_molecule_summary": {
            "frac_all_same_sign_base": float(np.mean([r["all_same_sign_base"] for r in rows])),
            "mean_sign_agreement_base": float(np.mean([r["sign_agreement_base"] for r in rows])),
            "mean_pred_std_base": float(np.mean([r["pred_std_base"] for r in rows])),
            "median_pred_std_base": float(np.median([r["pred_std_base"] for r in rows])),
            "mean_pred_std_combo": float(np.mean([r["pred_std_combo"] for r in rows])),
            "median_pred_std_combo": float(np.median([r["pred_std_combo"] for r in rows])),
            "mean_abs_signed_residual_base": float(
                np.mean(np.abs([r["mean_signed_residual_base"] for r in rows]))
            ),
            "mean_delta_se_combo_minus_base": float(
                np.mean([r["mean_delta_se_combo_minus_base"] for r in rows])
            ),
            "mean_delta_ae_combo_minus_base": float(
                np.mean([r["mean_delta_ae_combo_minus_base"] for r in rows])
            ),
            "n_molecules_combo_better_mean_se": int(
                sum(r["mean_delta_se_combo_minus_base"] < 0 for r in rows)
            ),
            "n_molecules_combo_worse_mean_se": int(
                sum(r["mean_delta_se_combo_minus_base"] > 0 for r in rows)
            ),
        },
        "base_tail_overlap": overlap,
        "base_tail_subset": tail_subset,
        "stable_tail_ge2_seeds": stable_tail,
        "bootstrap": {
            "unit": "Bemis-Murcko scaffold",
            "n_boot": int(n_boot),
            "cluster_profile": cluster.profile(),
            "intervals": bootstrap,
            "molecule_independence_used": False,
            "note": (
                "Percentile intervals from resampling whole scaffold clusters with replacement; "
                "no molecule-independent interval is used to narrow any interval."
            ),
        },
    }
    return payload, per_molecule


def analyse_group_a_single_seed(group: Group) -> dict[str, Any]:
    """Group A has one seed per model, so seed stability is not estimable."""
    lcrc = group.predictions["lcrc_a_r3_s0"]
    pna = group.predictions["pna_native_s0"]
    y = lcrc.y
    d = (lcrc.pred - y) ** 2 - (pna.pred - y) ** 2
    return {
        "status": "NOT_APPLICABLE_SINGLE_SEED",
        "reason": "LCRC-A-R3 and PNA-native have exactly one frozen seed each on random42",
        "n": int(len(y)),
        "mean_d_i": float(np.mean(d)),
        "median_d_i": float(np.median(d)),
        "frac_pna_better": float(np.mean(d > 0)),
        "frac_lcrc_better": float(np.mean(d < 0)),
        "note": "seed variance cannot be separated from the model gap on this group",
    }
