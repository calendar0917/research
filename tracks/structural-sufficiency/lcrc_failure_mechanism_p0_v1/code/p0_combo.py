"""COMBO / CHEM8 / MULTICONTRIB attribution on the frozen random122 predictions.

Everything is computed per split (Validation420 and historical Test420) and per
seed.  Test is ``POST_HOC_EXPOSED_DISCOVERY`` and never selects a checkpoint.

The single-model effect (overall metric differences), the conditional subset
effect (pre-defined groups) and the post-hoc extreme cases (top SSE contributors)
are reported as separate quantities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from p0_common import BOOTSTRAP_SEED, LOW_SUPPORT_N, N_BOOT, TOP_FRACTION
from p0_sources import Group, align_by_row_id
from p0_stats import (
    ClusterIndex,
    bootstrap_summary,
    cluster_bootstrap,
    metrics,
    quartile_bins,
    top_error_indices,
)

SPLITS = ("valid", "test")
SEEDS = (0, 1, 2)
ARMS = ("base", "chem8", "multicontrib", "combo_chem_contrib")
ARM_LABEL = {
    "base": "BASE",
    "chem8": "CHEM8",
    "multicontrib": "MULTICONTRIB",
    "combo_chem_contrib": "COMBO (CHEM8+MULTICONTRIB)",
}
EXPOSURE = {
    "valid": "FROZEN_VALIDATION_ONLY",
    "test": "POST_HOC_EXPOSED_DISCOVERY",
}

#: pre-declared groupings; support/size/TPSA quartiles are fit on Validation420
GROUPINGS = (
    ("scaffold_seen", ("unseen", "seen")),
    ("unseen_atom_frac_q", ("Q1_lowest", "Q2", "Q3", "Q4_highest")),
    ("rare_atom_frac_q", ("Q1_lowest", "Q2", "Q3", "Q4_highest")),
    ("median_log1p_df_q", ("Q1_lowest", "Q2", "Q3", "Q4_highest")),
    ("p10_log1p_df_q", ("Q1_lowest", "Q2", "Q3", "Q4_highest")),
    ("heavy_atoms_q", ("Q1_lowest", "Q2", "Q3", "Q4_highest")),
    ("tpsa_q", ("Q1_lowest", "Q2", "Q3", "Q4_highest")),
)


def _prediction(group: Group, split: str, arm: str, seed: int) -> np.ndarray:
    suffix = "_test" if split == "test" else ""
    key = f"{arm}_s{seed}{suffix}"
    pred = group.predictions[key]
    return align_by_row_id(pred.rows, pred.pred, group.rows[split], label=f"combo/{key}")


def _available_arms(group: Group, split: str, seed: int) -> list[str]:
    suffix = "_test" if split == "test" else ""
    return [arm for arm in ARMS if f"{arm}_s{seed}{suffix}" in group.predictions]


def _group_labels(valid_frame: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    labels: dict[str, np.ndarray] = {}
    edges_record: dict[str, Any] = {}
    for column, name in (
        ("heavy_atoms", "heavy_atoms_q"),
        ("tpsa", "tpsa_q"),
        ("unseen_atom_frac", "unseen_atom_frac_q"),
        ("rare_atom_frac", "rare_atom_frac_q"),
        ("median_log1p_df", "median_log1p_df_q"),
        ("p10_log1p_df", "p10_log1p_df_q"),
    ):
        _, edges = quartile_bins(valid_frame[column].to_numpy(dtype=np.float64), 4)
        labels[name] = np.searchsorted(edges, frame[column].to_numpy(dtype=np.float64), side="right")
        edges_record[name] = edges.tolist()
    labels["scaffold_seen"] = frame["scaffold_seen"].astype(bool).to_numpy().astype(np.int64)
    edges_record["scaffold_seen"] = ["False_unseen", "True_seen"]
    return labels, edges_record


def _subset(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {"n": 0}
    return metrics(y[mask], pred[mask])


def _contribution_table(
    y: np.ndarray, base_pred: np.ndarray, arm_pred: np.ndarray
) -> dict[str, Any]:
    """SSE contributions ``c_i = SE_base - SE_arm`` (positive = arm better)."""
    c = (base_pred - y) ** 2 - (arm_pred - y) ** 2
    positive = np.sort(c[c > 0])[::-1]
    negative = np.sort(c[c < 0])
    net = float(np.sum(c))
    top5_pos = float(np.sum(positive[:5]))
    top10_pos = float(np.sum(positive[:10]))
    top5_neg = float(np.sum(negative[:5]))
    top10_neg = float(np.sum(negative[:10]))
    gross_pos = float(np.sum(positive))
    gross_neg = float(np.sum(negative))

    def _ratio(value: float, denom: float) -> float | None:
        return float(value / denom) if abs(denom) > 1e-15 else None

    return {
        "n": int(len(c)),
        "net_sse_reduction_base_minus_arm": net,
        "gross_positive": gross_pos,
        "gross_negative": gross_neg,
        "n_positive": int(np.sum(c > 0)),
        "n_negative": int(np.sum(c < 0)),
        "top5_positive_sum": top5_pos,
        "top10_positive_sum": top10_pos,
        "top5_negative_sum": top5_neg,
        "top10_negative_sum": top10_neg,
        "top5_positive_frac_of_net": _ratio(top5_pos, net),
        "top10_positive_frac_of_net": _ratio(top10_pos, net),
        "top5_negative_frac_of_net": _ratio(top5_neg, net),
        "top10_negative_frac_of_net": _ratio(top10_neg, net),
        "top5_positive_frac_of_gross_positive": _ratio(top5_pos, gross_pos),
        "top10_positive_frac_of_gross_positive": _ratio(top10_pos, gross_pos),
        "top5_negative_frac_of_gross_negative": _ratio(top5_neg, gross_neg),
        "top10_negative_frac_of_gross_negative": _ratio(top10_neg, gross_neg),
        "post_hoc_selected_extremes": True,
        "note": (
            "The top-5/top-10 molecules are selected after seeing the errors; they describe "
            "concentration of the SSE change and are not an independent effect estimate."
        ),
    }


def analyse_combo(group: Group, support: Any) -> dict[str, Any]:
    query = support.query.reset_index(drop=True)
    valid_frame = query[query["split"] == "valid"].reset_index(drop=True)
    if len(valid_frame) != len(group.rows["valid"]):
        raise ValueError("support frame missing Validation420")

    records: list[dict[str, Any]] = []
    overall: dict[str, Any] = {split: {} for split in SPLITS}
    base_tail: dict[str, Any] = {split: {} for split in SPLITS}
    improved: dict[str, Any] = {split: {} for split in SPLITS}
    contribution: dict[str, Any] = {split: {} for split in SPLITS}
    interaction: dict[str, Any] = {split: {} for split in SPLITS}
    bootstrap_payload: dict[str, Any] = {split: {} for split in SPLITS}
    quartile_edges: dict[str, Any] = {}
    group_sizes: dict[str, Any] = {}

    for split in SPLITS:
        frame = query[query["split"] == split].reset_index(drop=True)
        if not np.array_equal(frame["row_id"].to_numpy(), group.rows[split]):
            raise ValueError(f"{split}: support frame is not aligned to the frozen split order")
        y = np.asarray(group.labels[split], dtype=np.float64)
        labels, edges = _group_labels(valid_frame, frame)
        quartile_edges[split] = edges
        group_sizes[split] = {
            name: {str(label): int(np.sum(labels[name] == i)) for i, label in enumerate(names)}
            for name, names in GROUPINGS
        }

        preds: dict[tuple[int, str], np.ndarray] = {}
        for seed in SEEDS:
            for arm in _available_arms(group, split, seed):
                preds[(seed, arm)] = _prediction(group, split, arm, seed)

        for seed in SEEDS:
            if (seed, "base") not in preds:
                continue
            base_pred = preds[(seed, "base")]
            base_metrics = metrics(y, base_pred)
            overall[split][f"s{seed}"] = {}
            base_tail[split][f"s{seed}"] = {}
            improved[split][f"s{seed}"] = {}
            contribution[split][f"s{seed}"] = {}
            tail_idx = top_error_indices(np.abs(base_pred - y), TOP_FRACTION)
            for arm in _available_arms(group, split, seed):
                arm_pred = preds[(seed, arm)]
                arm_metrics = metrics(y, arm_pred)
                overall[split][f"s{seed}"][arm] = arm_metrics
                records.append(
                    {
                        "split": split,
                        "exposure": EXPOSURE[split],
                        "seed": seed,
                        "arm": arm,
                        "grouping": "overall",
                        "group_label": "all",
                        "n": int(len(y)),
                        "low_support": bool(len(y) < LOW_SUPPORT_N),
                        **arm_metrics,
                        "delta_rmse_vs_base": float(arm_metrics["rmse"] - base_metrics["rmse"]),
                        "delta_mae_vs_base": float(arm_metrics["mae"] - base_metrics["mae"]),
                        "delta_sse_vs_base": float(arm_metrics["sse"] - base_metrics["sse"]),
                    }
                )
                for grouping, names in GROUPINGS:
                    values = labels[grouping]
                    for i, name in enumerate(names):
                        mask = values == i
                        subset = _subset(y, arm_pred, mask)
                        if subset["n"] == 0:
                            continue
                        base_subset = _subset(y, base_pred, mask)
                        records.append(
                            {
                                "split": split,
                                "exposure": EXPOSURE[split],
                                "seed": seed,
                                "arm": arm,
                                "grouping": grouping,
                                "group_label": name,
                                "n": int(subset["n"]),
                                "low_support": bool(subset["n"] < LOW_SUPPORT_N),
                                **subset,
                                "delta_rmse_vs_base": float(subset["rmse"] - base_subset["rmse"]),
                                "delta_mae_vs_base": float(subset["mae"] - base_subset["mae"]),
                                "delta_sse_vs_base": float(subset["sse"] - base_subset["sse"]),
                            }
                        )
                tail_arm = _subset(y, arm_pred, np.isin(np.arange(len(y)), tail_idx))
                base_tail[split][f"s{seed}"][arm] = tail_arm
                se_base = (base_pred - y) ** 2
                se_arm = (arm_pred - y) ** 2
                ae_base = np.abs(base_pred - y)
                ae_arm = np.abs(arm_pred - y)
                improved[split][f"s{seed}"][arm] = {
                    "n": int(len(y)),
                    "n_se_improved": int(np.sum(se_arm < se_base)),
                    "frac_se_improved": float(np.mean(se_arm < se_base)),
                    "n_ae_improved": int(np.sum(ae_arm < ae_base)),
                    "frac_ae_improved": float(np.mean(ae_arm < ae_base)),
                    "n_unchanged": int(np.sum(se_arm == se_base)),
                    "definition": "improved = squared error strictly smaller than same-seed BASE",
                }
                contribution[split][f"s{seed}"][arm] = _contribution_table(y, base_pred, arm_pred)
            base_tail[split][f"s{seed}"]["definition"] = (
                f"top {int(round(100 * TOP_FRACTION))}% of same-seed BASE absolute error, "
                "fixed before looking at candidate errors"
            )
            base_tail[split][f"s{seed}"]["n"] = int(len(tail_idx))
            base_tail[split][f"s{seed}"]["tail_indices"] = tail_idx.tolist()

        # ---- 2x2 interaction on the MSE scale (all four arms available at seed 0)
        seed = 0
        if all((seed, arm) in preds for arm in ARMS):
            b = preds[(seed, "base")]
            c8 = preds[(seed, "chem8")]
            mc = preds[(seed, "multicontrib")]
            cb = preds[(seed, "combo_chem_contrib")]

            def _mse_of(pred: np.ndarray, mask: np.ndarray) -> float:
                return float(np.mean((pred[mask] - y[mask]) ** 2))

            def _tail_mask(idx: np.ndarray) -> np.ndarray:
                k = max(1, int(round(TOP_FRACTION * len(idx))))
                order = np.argsort(-np.abs(b[idx] - y[idx]), kind="stable")[:k]
                mask = np.zeros(len(idx), dtype=bool)
                mask[order] = True
                return mask

            all_mask = np.ones(len(y), dtype=bool)
            tail_mask = _tail_mask(np.arange(len(y)))
            interaction[split]["s0"] = {"available": True}
            for name, mask in (
                ("overall", all_mask),
                ("base_top10_subset", tail_mask),
                ("scaffold_seen", labels["scaffold_seen"] == 1),
                ("scaffold_unseen", labels["scaffold_seen"] == 0),
            ):
                mse = {
                    "base": _mse_of(b, mask),
                    "chem8": _mse_of(c8, mask),
                    "multicontrib": _mse_of(mc, mask),
                    "combo_chem_contrib": _mse_of(cb, mask),
                }
                inter = mse["combo_chem_contrib"] - mse["chem8"] - mse["multicontrib"] + mse["base"]
                interaction[split]["s0"][name] = {
                    "n": int(np.sum(mask)),
                    "mse": mse,
                    "interaction_mse_combo_minus_chem8_minus_multicontrib_plus_base": float(inter),
                    "definition": (
                        "MSE(COMBO) - MSE(CHEM8) - MSE(MULTICONTRIB) + MSE(BASE); "
                        "positive means the joint change is worse than the sum of single changes"
                    ),
                }
            interaction[split]["s0"]["tail_definition"] = (
                "BASE top 10% absolute error recomputed inside each sample"
            )
        for seed_other in (1, 2):
            interaction[split][f"s{seed_other}"] = {
                "available": False,
                "reason": "CHEM8 and MULTICONTRIB have no seed %d prediction file" % seed_other,
            }

        # ---- scaffold-cluster bootstrap for the key paired quantities
        scaffolds = frame["scaffold"].astype(str).to_numpy()
        cluster = ClusterIndex.from_values(scaffolds)
        seed0_tail = (
            top_error_indices(np.abs(preds[(0, "base")] - y), TOP_FRACTION)
            if (0, "base") in preds
            else np.asarray([], dtype=np.int64)
        )

        def _compute(idx: np.ndarray) -> dict[str, float]:
            out: dict[str, float] = {}
            yv = y[idx]
            for s in SEEDS:
                if (s, "base") not in preds:
                    continue
                bv = preds[(s, "base")][idx]
                bm = metrics(yv, bv)
                k = max(1, int(round(TOP_FRACTION * len(idx))))
                tail = np.argsort(-np.abs(bv - yv), kind="stable")[:k]
                for arm in _available_arms(group, split, s):
                    av = preds[(s, arm)][idx]
                    am = metrics(yv, av)
                    out[f"s{s}_{arm}_delta_rmse"] = am["rmse"] - bm["rmse"]
                    out[f"s{s}_{arm}_delta_mae"] = am["mae"] - bm["mae"]
                    out[f"s{s}_{arm}_delta_sse"] = am["sse"] - bm["sse"]
                    out[f"s{s}_{arm}_base_tail_delta_rmse"] = float(
                        metrics(yv[tail], av[tail])["rmse"] - metrics(yv[tail], bv[tail])["rmse"]
                    )
                    out[f"s{s}_{arm}_base_tail_delta_mae"] = float(
                        metrics(yv[tail], av[tail])["mae"] - metrics(yv[tail], bv[tail])["mae"]
                    )
            # seed-mean combo deltas
            combo_rmse, combo_mae = [], []
            for s in SEEDS:
                if (s, "base") in preds and (s, "combo_chem_contrib") in preds:
                    bm = metrics(yv, preds[(s, "base")][idx])
                    cm = metrics(yv, preds[(s, "combo_chem_contrib")][idx])
                    combo_rmse.append(cm["rmse"] - bm["rmse"])
                    combo_mae.append(cm["mae"] - bm["mae"])
            out["combo_seed_mean_delta_rmse"] = float(np.mean(combo_rmse)) if combo_rmse else float("nan")
            out["combo_seed_mean_delta_mae"] = float(np.mean(combo_mae)) if combo_mae else float("nan")
            # 2x2 interaction overall and on the BASE-tail of seed 0
            if all((0, arm) in preds for arm in ARMS):
                tail0 = np.isin(np.arange(len(idx)), seed0_tail)
                for tag, mask in (
                    ("overall", np.ones(len(idx), dtype=bool)),
                    ("base_tail", tail0),
                ):
                    vals = {
                        arm: float(np.mean((preds[(0, arm)][idx][mask] - yv[mask]) ** 2))
                        for arm in ARMS
                    }
                    out[f"interaction_{tag}_mse"] = (
                        vals["combo_chem_contrib"]
                        - vals["chem8"]
                        - vals["multicontrib"]
                        + vals["base"]
                    )
            return out

        point = _compute(np.arange(len(y)))
        draws = cluster_bootstrap(cluster, _compute, n_boot=N_BOOT, seed=BOOTSTRAP_SEED)
        bootstrap_payload[split] = {
            "unit": "Bemis-Murcko scaffold",
            "n_boot": N_BOOT,
            "cluster_profile": cluster.profile(),
            "intervals": bootstrap_summary(point, draws),
            "molecule_independence_used": False,
        }

    # explicit record of empty bins (e.g. p10_log1p_df has a point mass at zero)
    degenerate: dict[str, Any] = {}
    for split in SPLITS:
        degenerate[split] = {}
        for name, names in GROUPINGS:
            empty = [label for label, n in group_sizes[split][name].items() if n == 0]
            if empty:
                degenerate[split][name] = {
                    "empty_bins": empty,
                    "n_effective_bins": int(sum(1 for n in group_sizes[split][name].values() if n > 0)),
                    "reason": "quantile boundaries tied at a point mass; empty bins are dropped, not merged",
                }

    # direction summary that cannot hide a reversal
    direction = {}
    for split in SPLITS:
        per_seed = {}
        for seed in SEEDS:
            key = f"s{seed}"
            if key in overall[split] and "combo_chem_contrib" in overall[split][key]:
                b = overall[split][key]["base"]
                c = overall[split][key]["combo_chem_contrib"]
                per_seed[key] = {
                    "delta_rmse": float(c["rmse"] - b["rmse"]),
                    "delta_mae": float(c["mae"] - b["mae"]),
                    "delta_sse": float(c["sse"] - b["sse"]),
                }
        values = [v["delta_rmse"] for v in per_seed.values()]
        direction[split] = {
            "per_seed": per_seed,
            "n_seeds_combo_better_rmse": int(sum(v < 0 for v in values)),
            "n_seeds_combo_worse_rmse": int(sum(v > 0 for v in values)),
            "direction_reversal": bool(any(v < 0 for v in values) and any(v > 0 for v in values)),
        }

    return {
        "definition": {
            "delta_convention": (
                "delta_*_vs_base = candidate - BASE; negative means the candidate has lower error. "
                "SSE contributions use c_i = SE_BASE - SE_candidate, so positive means the candidate is better."
            ),
            "tail": f"top {int(round(100 * TOP_FRACTION))}% of same-seed BASE absolute error",
            "support_quartiles": "quartile edges fitted on Validation420 only and applied unchanged to Test420",
            "groupings": [name for name, _ in GROUPINGS],
        },
        "exposure": EXPOSURE,
        "group_sizes": group_sizes,
        "degenerate_bins": degenerate,
        "quartile_edges": quartile_edges,
        "overall": overall,
        "direction_summary_combo": direction,
        "group_table": records,
        "base_tail_subset": base_tail,
        "improved_fraction": improved,
        "contribution": contribution,
        "interaction_2x2": interaction,
        "bootstrap": bootstrap_payload,
        "missing_cells": [
            {"seed": seed, "arm": arm, "reason": "no frozen per-molecule prediction"}
            for seed in SEEDS
            for arm in ARMS
            if arm not in _available_arms(group, "valid", seed)
        ],
        "post_hoc_warning": (
            "Test metrics are POST_HOC_EXPOSED_DISCOVERY. Top-5/top-10 contributor tables are "
            "post-hoc selected extremes. Single-model effects, conditional-subset effects and "
            "post-hoc extremes must not be combined into a single headline number."
        ),
    }
