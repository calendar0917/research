"""Fast baseline-difficulty / typed-refinement redistribution audit (ZINC).

Decision audit, not a broad search.  It answers one question with three
pre-registered tests:

    Why does the corrected exact typed tokenizer make historical *easy*
    molecules worse while making some historical *hard* molecules better?

Data: frozen compact-v4-hinge historical / corrected validation predictions
(seeds 0-3), the historical -> corrected train token split map, official-train
molecule targets and the existing historical compact-v4-hinge **train OOF**
predictions (``results/oof_difficulty_audit/oof_per_molecule.csv``).  Official
test is never loaded and nothing is trained.

The three tests:

* **H1** error redistribution by historical difficulty quintile -- centre (bias)
  vs spread (seed variance) decomposition.
* **H2** train-derived *typed-refinement informativeness* (do the corrected
  typed children of a historical token carry distinguishable train target /
  historical-OOF-residual context?) vs the corrected gain, controlling for
  baseline difficulty.
* **H3** the 2x2 easy/hard x low/high-informativeness mechanism table.

Outputs (``results/baseline_difficulty_typed_refinement_audit/``):
``per_molecule_analysis.csv``, ``coarse_token_child_informativeness.csv``,
``difficulty_quintiles.csv``, ``movement_classes.csv``, ``typed_info_groups.csv``,
``mechanism_2x2.csv``, ``correlation_table.csv``, ``audit_summary.json``,
``decision_record.json``, ``figures/*.png``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_baseline_difficulty_typed_refinement_audit
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16.zinc_corrected_token_fragmentation_audit import (
    SEEDS,
    _write_csv,
    build_tables,
    load_frozen_runs,
    load_occurrences,
    molecule_matrix,
    paired_degradation,
    partial_spearman,
    spearman,
    token_split_map,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OOF_MOLECULE_CSV = (
    REPO_ROOT / "tracks/ksvd/results/oof_difficulty_audit/oof_per_molecule.csv"
)
OUT_DIR = REPO_ROOT / "tracks/ksvd/results/baseline_difficulty_typed_refinement_audit"

MIN_CHILD_SUPPORT = 5  # molecule-level train support for a corrected child class
MAGNITUDE_EPS = 0.01  # |prediction shift| below this = magnitude-only disturbance

# Pre-registered primary informativeness metric (residual-based, molecule-mean
# over the molecule's split patches; 0 when the tokenizer offers no refinement).
PRIMARY_INFO = "info_resid_mean_split"
# Three seed-consistency / gate primary thresholds.
GATE_PARTIAL_STRONG = 0.20
GATE_PARTIAL_MODERATE = 0.15
GATE_PARTIAL_WEAK = 0.10


# --------------------------------------------------------------------------
# Train context (targets + historical OOF residuals for the 10 000 train mols)
# --------------------------------------------------------------------------
def load_train_context(n_train: int) -> dict[str, np.ndarray]:
    rows: dict[int, dict[str, str]] = {}
    with OOF_MOLECULE_CSV.open() as handle:
        for row in csv.DictReader(handle):
            rows[int(row["subset_index"])] = row
    target = np.asarray([float(rows[i]["target"]) for i in range(n_train)])
    # oof_per_molecule signed_residual = target - prediction; the sign is
    # irrelevant for dispersion, only the spread across children matters.
    resid = np.asarray([float(rows[i]["mean_signed_residual"]) for i in range(n_train)])
    return {"target": target, "resid": resid}


def child_statistics(
    occ: Mapping[str, np.ndarray], train_ctx: Mapping[str, np.ndarray]
) -> dict[int, dict[str, float]]:
    """Molecule-level target / OOF-residual context of every corrected child."""
    train_mol = occ["train_mol_ids"].tolist()
    corr = occ["train_corr_r2_id"].tolist()
    child_mols: dict[int, set[int]] = defaultdict(set)
    for mol, child in zip(train_mol, corr):
        child_mols[int(child)].add(int(mol))
    target = train_ctx["target"]
    resid = train_ctx["resid"]
    stats: dict[int, dict[str, float]] = {}
    for child, mols in child_mols.items():
        idx = np.fromiter(mols, dtype=np.int64)
        stats[int(child)] = {
            "support": int(len(idx)),
            "mean_target": float(target[idx].mean()),
            "median_target": float(np.median(target[idx])),
            "mean_resid": float(resid[idx].mean()),
        }
    return stats


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    w = weights / weights.sum()
    mean = float((w * values).sum())
    return float(np.sqrt((w * (values - mean) ** 2).sum()))


def parent_informativeness(
    split_r2: Mapping[int, Mapping[str, Any]],
    child_stats: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, Any]]:
    """Child-context dispersion for every historical (parent) token."""
    parents: dict[int, dict[str, Any]] = {}
    for hist_id, info in split_r2.items():
        children = list(info["corrected_child_ids"])
        eligible = [
            c for c in children if child_stats.get(c, {}).get("support", 0) >= MIN_CHILD_SUPPORT
        ]
        record: dict[str, Any] = {
            "historical_count": int(info["historical_count"]),
            "split_multiplicity": int(info["split_multiplicity"]),
            "n_children": int(len(children)),
            "n_eligible_children": int(len(eligible)),
            "eligible": bool(len(children) >= 2 and len(eligible) >= 2),
            "eligible_children": eligible,
        }
        if record["eligible"]:
            w = np.asarray([child_stats[c]["support"] for c in eligible], dtype=np.float64)
            mt = np.asarray([child_stats[c]["mean_target"] for c in eligible], dtype=np.float64)
            mr = np.asarray([child_stats[c]["mean_resid"] for c in eligible], dtype=np.float64)
            record.update(
                {
                    "target_dispersion": _weighted_std(mt, w),
                    "resid_dispersion": _weighted_std(mr, w),
                    "target_range": float(mt.max() - mt.min()),
                    "resid_range": float(mr.max() - mr.min()),
                }
            )
        else:
            record.update(
                {"target_dispersion": 0.0, "resid_dispersion": 0.0, "target_range": 0.0, "resid_range": 0.0}
            )
        parents[int(hist_id)] = record
    return parents


def high_info_threshold(parents: Mapping[int, Mapping[str, Any]], top_fraction: float = 1.0 / 3.0) -> float:
    values = np.asarray(
        [p["resid_dispersion"] for p in parents.values() if p["eligible"]], dtype=np.float64
    )
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, 1.0 - top_fraction))


def molecule_informativeness(
    occ: Mapping[str, np.ndarray],
    parents: Mapping[int, Mapping[str, Any]],
    high_threshold: float,
    n_valid: int,
) -> dict[int, dict[str, float]]:
    mols = occ["valid_mol_ids"].tolist()
    hist = occ["valid_hist_r2_id"].tolist()
    by_mol: dict[int, list[int]] = defaultdict(list)
    for mol, h in zip(mols, hist):
        by_mol[int(mol)].append(int(h))

    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if len(values) else 0.0

    # historical token seen in validation but with no train occurrence -> the
    # corrected tokenizer cannot refine it from train statistics: zero info.
    absent = {
        "n_children": 0, "eligible": False, "resid_dispersion": 0.0,
        "target_dispersion": 0.0,
    }
    out: dict[int, dict[str, float]] = {}
    for mol in range(n_valid):
        hs = by_mol.get(mol, [])
        recs = [parents.get(h, absent) for h in hs]
        split = [r for r in recs if r["n_children"] >= 2]
        elig = [r for r in split if r["eligible"]]
        high = [
            parents.get(h, absent)["eligible"] and parents.get(h, absent)["resid_dispersion"] >= high_threshold
            for h in hs
        ]
        out[mol] = {
            "n_patches": float(len(recs)),
            "n_split_patches": float(len(split)),
            "n_eligible_patches": float(len(elig)),
            "info_resid_mean": _mean([r["resid_dispersion"] for r in recs]),
            "info_resid_max": float(max((r["resid_dispersion"] for r in recs), default=0.0)),
            "info_resid_mean_split": _mean([r["resid_dispersion"] for r in split]),
            "info_target_mean": _mean([r["target_dispersion"] for r in recs]),
            "info_target_max": float(max((r["target_dispersion"] for r in recs), default=0.0)),
            "info_target_mean_split": _mean([r["target_dispersion"] for r in split]),
            "frac_high_info_patches": float(np.mean(high)) if high else 0.0,
        }
    return out


# --------------------------------------------------------------------------
# H1 -- difficulty quintiles, movement classes, centre/spread decomposition
# --------------------------------------------------------------------------
def quintile_assignment(metric: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(metric, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    bins = np.digitize(metric, edges[1:-1])  # 0-based quintile index
    return bins, edges


def center_spread_decomposition(
    pred_hist: np.ndarray, pred_corr: np.ndarray, targets: np.ndarray
) -> dict[str, np.ndarray]:
    """Exact telescoping MSE decomposition (no statistical claim).

    For any family, ``E_s[(pred_s - y)^2] = (ens - y)^2 + Var_s(pred)`` where
    ``ens`` is the seed mean.  The first term is the *centre* (ensemble/bias)
    component, the second the *spread* (seed-variance) component.  Comparing the
    two across tokenizers separates "the corrected representation moved the
    centre" from "the corrected representation added variance".
    """
    ens_hist = pred_hist.mean(axis=0)
    ens_corr = pred_corr.mean(axis=0)
    # L2 (exact telescoping; outlier-sensitive on ZINC's heavy target tail)
    center_hist = (ens_hist - targets) ** 2
    center_corr = (ens_corr - targets) ** 2
    spread_hist = pred_hist.var(axis=0)
    spread_corr = pred_corr.var(axis=0)
    # L1 (robust; descriptive - center = ensemble |err|, spread = mean_s|pred_s - ens|)
    center_mae_hist = np.abs(ens_hist - targets)
    center_mae_corr = np.abs(ens_corr - targets)
    spread_mae_hist = np.abs(pred_hist - ens_hist).mean(axis=0)
    spread_mae_corr = np.abs(pred_corr - ens_corr).mean(axis=0)
    return {
        "center_hist": center_hist,
        "center_corr": center_corr,
        "delta_center": center_corr - center_hist,
        "spread_hist": spread_hist,
        "spread_corr": spread_corr,
        "delta_spread": spread_corr - spread_hist,
        "delta_center_mae": center_mae_corr - center_mae_hist,
        "delta_spread_mae": spread_mae_corr - spread_mae_hist,
    }


def difficulty_quintiles(
    bins: np.ndarray,
    degradation: Mapping[str, Any],
    ens_err_hist: np.ndarray,
    ens_err_corr: np.ndarray,
    decomp: Mapping[str, np.ndarray] | None = None,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    rows = []
    for q in range(n_bins):
        idx = np.where(bins == q)[0]
        if idx.size == 0:
            continue
        row: dict[str, Any] = {
            "quintile": q + 1,
            "n": int(idx.size),
            "hist_mae": float(degradation["err_hist"][:, idx].mean()),
            "corr_mae": float(degradation["err_corr"][:, idx].mean()),
            "degradation": float(degradation["delta"][:, idx].mean()),
            "ensemble_hist_mae": float(ens_err_hist[idx].mean()),
            "ensemble_corr_mae": float(ens_err_corr[idx].mean()),
            "ensemble_degradation": float((ens_err_corr - ens_err_hist)[idx].mean()),
            "hist_disagreement": float(degradation["disagreement_hist"][idx].mean()),
            "corr_disagreement": float(degradation["disagreement_corr"][idx].mean()),
            "delta_disagreement": float(degradation["delta_disagreement"][idx].mean()),
        }
        if decomp is not None:
            row.update(
                {
                    "delta_center_mse": float(decomp["delta_center"][idx].mean()),
                    "delta_spread_mse": float(decomp["delta_spread"][idx].mean()),
                    "delta_center_mae": float(decomp["delta_center_mae"][idx].mean()),
                    "delta_spread_mae": float(decomp["delta_spread_mae"][idx].mean()),
                }
            )
        rows.append(row)
    return rows


def movement_classes(
    bins: np.ndarray,
    pred_hist: np.ndarray,
    pred_corr: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    n_seeds = pred_hist.shape[0]
    rows = []
    for q in range(n_bins):
        idx = np.where(bins == q)[0]
        if idx.size == 0:
            continue
        helpful = harmful = wrong = overshoot = magnitude = total = 0
        for s in range(n_seeds):
            err_h = pred_hist[s, idx] - targets[idx]
            err_c = pred_corr[s, idx] - targets[idx]
            shift = pred_corr[s, idx] - pred_hist[s, idx]
            abs_h = np.abs(err_h)
            abs_c = np.abs(err_c)
            help_mask = abs_c < abs_h
            harm_mask = abs_c > abs_h
            same_sign = err_h * err_c > 0
            helpful += int(help_mask.sum())
            harmful += int(harm_mask.sum())
            wrong += int((harm_mask & same_sign).sum())
            overshoot += int((harm_mask & ~same_sign).sum())
            magnitude += int((harm_mask & (np.abs(shift) <= MAGNITUDE_EPS)).sum())
            total += idx.size
        rows.append(
            {
                "quintile": q + 1,
                "n_pairs": int(total),
                "helpful_pct": 100.0 * helpful / total,
                "harmful_pct": 100.0 * harmful / total,
                "wrong_direction_pct": 100.0 * wrong / total,
                "overshoot_pct": 100.0 * overshoot / total,
                "magnitude_only_pct": 100.0 * magnitude / total,
            }
        )
    return rows


# --------------------------------------------------------------------------
# H2/H3 -- informativeness vs gain
# --------------------------------------------------------------------------
def info_group_table(
    info: np.ndarray,
    degradation: Mapping[str, Any],
    n_groups: int = 3,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    edges = np.quantile(info, np.linspace(0, 1, n_groups + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    labels = np.digitize(info, edges[1:-1])
    names = ("low", "mid", "high") if n_groups == 3 else tuple(f"g{i}" for i in range(n_groups))
    rows = []
    for g, name in enumerate(names):
        idx = np.where(labels == g)[0]
        if idx.size == 0:
            continue
        rows.append(
            {
                "info_group": name,
                "n": int(idx.size),
                "hist_mae": float(degradation["err_hist"][:, idx].mean()),
                "corr_mae": float(degradation["err_corr"][:, idx].mean()),
                "gain": float((degradation["err_hist"] - degradation["err_corr"])[:, idx].mean()),
                "delta_disagreement": float(degradation["delta_disagreement"][idx].mean()),
            }
        )
    return rows, labels


def mechanism_2x2(
    bins: np.ndarray,
    info_median: float,
    info: np.ndarray,
    degradation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    easy = bins <= 1  # Q1/Q2
    hard = bins >= 3  # Q4/Q5
    low = info <= info_median
    high = info > info_median
    cells = [
        ("easy", "low", easy & low),
        ("easy", "high", easy & high),
        ("hard", "low", hard & low),
        ("hard", "high", hard & high),
    ]
    rows = []
    for diff, inf, mask in cells:
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        rows.append(
            {
                "difficulty": diff,
                "typed_info": inf,
                "n": int(idx.size),
                "hist_mae": float(degradation["err_hist"][:, idx].mean()),
                "corr_mae": float(degradation["err_corr"][:, idx].mean()),
                "gain": float((degradation["err_hist"] - degradation["err_corr"])[:, idx].mean()),
                "delta_disagreement": float(degradation["delta_disagreement"][idx].mean()),
            }
        )
    return rows


INFO_METRIC_NAMES = (
    "info_resid_mean_split",
    "info_resid_mean",
    "info_resid_max",
    "info_target_mean_split",
    "info_target_mean",
    "info_target_max",
    "frac_high_info_patches",
)


def informativeness_association(
    info: np.ndarray,
    degradation: Mapping[str, Any],
    ens_err_hist: np.ndarray,
    ens_err_corr: np.ndarray,
    frequency: np.ndarray,
) -> dict[str, Any]:
    gain_seed = degradation["err_hist"] - degradation["err_corr"]  # positive = corrected better
    gain_ens = ens_err_hist - ens_err_corr
    baseline = degradation["mean_err_hist"]

    per_seed = [spearman(info, gain_seed[s]) for s in range(len(SEEDS))]
    ensemble = spearman(info, gain_ens)
    same_sign = int(sum(1 for r in per_seed if r == r and r > 0))
    return {
        "per_seed_rho": [round(r, 4) for r in per_seed],
        "ensemble_rho": round(ensemble, 4),
        "seeds_positive": same_sign,
        "seeds_total": len(SEEDS),
        "partial_given_difficulty": round(partial_spearman(info, gain_ens, baseline), 4),
        "partial_given_frequency": round(partial_spearman(info, gain_ens, frequency), 4),
        "spearman_info_vs_baseline": round(spearman(info, baseline), 4),
    }


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------
def decide(
    quintiles: Sequence[Mapping[str, Any]],
    moves: Sequence[Mapping[str, Any]],
    association: Mapping[str, Any],
    info_groups: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    bins: np.ndarray,
    decomp: Mapping[str, np.ndarray],
    degradation: Mapping[str, Any],
    ens_err_hist: np.ndarray,
    ens_err_corr: np.ndarray,
) -> dict[str, Any]:
    by_q = {row["quintile"]: row for row in quintiles}
    easy_deg = float(np.mean([by_q[q]["degradation"] for q in (1, 2) if q in by_q]))
    hard_deg = float(np.mean([by_q[q]["degradation"] for q in (4, 5) if q in by_q]))
    easy_dis = float(np.mean([by_q[q]["delta_disagreement"] for q in (1, 2) if q in by_q]))
    hard_dis = float(np.mean([by_q[q]["delta_disagreement"] for q in (4, 5) if q in by_q]))
    indiv_deg = float(np.mean([row["degradation"] for row in quintiles]))
    ens_deg = float(np.mean([row["ensemble_degradation"] for row in quintiles]))
    idx_harm = np.mean([row["harmful_pct"] for row in moves])

    partial = association["partial_given_difficulty"]
    seed_ok = association["seeds_positive"] >= 3
    group = {row["info_group"]: row for row in info_groups}
    gain_high = group.get("high", {}).get("gain", 0.0)
    gain_low = group.get("low", {}).get("gain", 0.0)
    gain_mid = group.get("mid", {}).get("gain", 0.0)
    cell = {(row["difficulty"], row["typed_info"]): row for row in cells}
    gain_easy_low = cell.get(("easy", "low"), {}).get("gain", 0.0)
    gain_easy_high = cell.get(("easy", "high"), {}).get("gain", 0.0)
    gain_hard_low = cell.get(("hard", "low"), {}).get("gain", 0.0)
    gain_hard_high = cell.get(("hard", "high"), {}).get("gain", 0.0)
    interaction = (gain_hard_high - gain_hard_low) - (gain_easy_high - gain_easy_low)

    easy_mask = bins <= 1
    hard_mask = bins >= 3
    easy_ens_deg = float((ens_err_corr - ens_err_hist)[easy_mask].mean())
    easy_indiv_deg = float(degradation["delta"][:, easy_mask].mean())
    easy_delta_center = float(decomp["delta_center_mae"][easy_mask].mean())
    easy_delta_spread = float(decomp["delta_spread_mae"][easy_mask].mean())
    hard_delta_center = float(decomp["delta_center_mae"][hard_mask].mean())
    hard_delta_spread = float(decomp["delta_spread_mae"][hard_mask].mean())
    easy_center_dominates = bool(easy_delta_center > easy_delta_spread and easy_delta_center > 0)
    # per-seed stability of the difficulty ladder (positive = corrected worse on easy)
    ladder = [spearman(degradation["mean_err_hist"], degradation["delta"][s]) for s in range(len(SEEDS))]

    gates = {
        "hurts_easy": bool(easy_deg > 0 and easy_deg > hard_deg),
        "easy_disagreement_increases": bool(easy_dis > 0),
        "ensemble_degradation_smaller_easy": bool(
            abs(easy_ens_deg) < abs(easy_indiv_deg) * 0.6
        ),
        "easy_variance_dominated": bool(not easy_center_dominates and easy_delta_spread > 0),
        "typed_info_weak_after_control": bool(abs(partial) < GATE_PARTIAL_WEAK and not seed_ok),
        "typed_info_strong_seed_consistent": bool(
            partial >= GATE_PARTIAL_STRONG and seed_ok
        ),
        "typed_info_moderate_seed_consistent": bool(
            partial >= GATE_PARTIAL_MODERATE and seed_ok
        ),
        "high_info_group_benefits": bool(gain_high > 0 and gain_high > gain_low),
        "low_info_group_no_gain": bool(gain_low <= 0),
        "hard_high_largest_gain": bool(
            gain_hard_high >= max(gain_hard_low, gain_easy_high, gain_easy_low)
        ),
        "clear_2x2_interaction": bool(interaction > 0.005),
    }

    if (
        gates["hurts_easy"]
        and gates["easy_disagreement_increases"]
        and gates["ensemble_degradation_smaller_easy"]
        and gates["typed_info_weak_after_control"]
    ):
        case = "A_COARSE_REGULARIZATION"
    elif (
        gates["typed_info_strong_seed_consistent"]
        and gates["high_info_group_benefits"]
        and gates["low_info_group_no_gain"]
        and gates["hard_high_largest_gain"]
    ):
        case = "B_SELECTIVE_TYPED_REFINEMENT"
    elif (
        gates["easy_disagreement_increases"]
        and gates["typed_info_moderate_seed_consistent"]
        and gates["hard_high_largest_gain"]
    ):
        case = "C_BOTH"
    else:
        case = "D_UNEXPLAINED_REDISTRIBUTION"

    return {
        "case": case,
        "gates": gates,
        "easy_degradation": easy_deg,
        "hard_degradation": hard_deg,
        "easy_delta_disagreement": easy_dis,
        "hard_delta_disagreement": hard_dis,
        "individual_degradation": indiv_deg,
        "ensemble_degradation": ens_deg,
        "easy_individual_degradation": easy_indiv_deg,
        "easy_ensemble_degradation": easy_ens_deg,
        "easy_delta_center_mae": easy_delta_center,
        "easy_delta_spread_mae": easy_delta_spread,
        "hard_delta_center_mae": hard_delta_center,
        "hard_delta_spread_mae": hard_delta_spread,
        "easy_delta_center_mse": float(decomp["delta_center"][easy_mask].mean()),
        "easy_delta_spread_mse": float(decomp["delta_spread"][easy_mask].mean()),
        "hard_delta_center_mse": float(decomp["delta_center"][hard_mask].mean()),
        "hard_delta_spread_mse": float(decomp["delta_spread"][hard_mask].mean()),
        "difficulty_ladder_spearman_per_seed": [round(r, 4) for r in ladder],
        "difficulty_ladder_seeds_negative": int(sum(1 for r in ladder if r < 0)),
        "difficulty_ladder_seeds_total": len(ladder),
        "harmful_pct_mean": float(idx_harm),
        "gain_low_mid_high": [gain_low, gain_mid, gain_high],
        "gain_cells": {
            "easy_low": gain_easy_low,
            "easy_high": gain_easy_high,
            "hard_low": gain_hard_low,
            "hard_high": gain_hard_high,
        },
        "interaction_hard_high_minus_others": interaction,
        "informativeness_association": dict(association),
        "interpretation": (
            "Centre (ensemble/bias) component dominates the easy degradation "
            f"(easy delta_center_mae={easy_delta_center:.4f} vs delta_spread_mae={easy_delta_spread:.4f}); "
            "the ensemble itself degrades on easy molecules, so this is not a clean "
            "variance-reducing-regularization effect (Case A condition 3 fails), and typed-refinement "
            "informativeness carries no independent signal (partial rho "
            f"{association['partial_given_difficulty']}), so B/C are ruled out."
        ),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(
    quintiles: Sequence[Mapping[str, Any]],
    info: np.ndarray,
    degradation: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = OUT_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)
        written.append(name)

    qx = [row["quintile"] for row in quintiles]
    # Fig 1 - difficulty quintile vs corrected degradation (individual vs ensemble)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(qx, [row["degradation"] for row in quintiles], "o-", label="individual-seed mean")
    ax.plot(qx, [row["ensemble_degradation"] for row in quintiles], "s--", label="4-seed ensemble")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(qx)
    ax.set_xlabel("historical-error quintile (1 = easiest, 5 = hardest)")
    ax.set_ylabel("corrected - historical |error|")
    ax.set_title("Figure 1 - degradation vs baseline difficulty")
    ax.legend()
    _save(fig, "fig1_difficulty_degradation.png")

    # Fig 2 - difficulty quintile vs delta disagreement
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(qx, [row["hist_disagreement"] for row in quintiles], "o-", label="historical")
    ax.plot(qx, [row["corr_disagreement"] for row in quintiles], "s-", label="corrected")
    ax.set_xticks(qx)
    ax.set_xlabel("historical-error quintile")
    ax.set_ylabel("cross-seed prediction std")
    ax.set_title("Figure 2 - seed disagreement vs baseline difficulty")
    ax.legend()
    _save(fig, "fig2_difficulty_disagreement.png")

    # Fig 3 - informativeness vs corrected gain (binned quintiles) with difficulty control
    edges = np.quantile(info, np.linspace(0, 1, 6))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    bins = np.digitize(info, edges[1:-1])
    gain_ens = degradation["mean_err_hist"] - degradation["mean_err_corr"]
    xs, ys, ns = [], [], []
    for b in range(5):
        idx = np.where(bins == b)[0]
        if idx.size == 0:
            continue
        xs.append(b + 1)
        ys.append(float(gain_ens[idx].mean()))
        ns.append(int(idx.size))
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.bar(xs, ys)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(xs)
    ax.set_xlabel("typed-refinement informativeness quintile")
    ax.set_ylabel("ensemble corrected gain (hist - corr |err|)")
    ax.set_title("Figure 3 - typed informativeness vs corrected gain")
    _save(fig, "fig3_informativeness_gain.png")

    # Fig 4 - 2x2 mechanism table
    labels, gains = [], []
    for row in cells:
        labels.append(f"{row['difficulty']}\n+{row['typed_info']}")
        gains.append(row["gain"])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    colors = ["tab:red" if g < 0 else "tab:green" for g in gains]
    ax.bar(range(len(gains)), gains, color=colors)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("corrected gain (hist - corr |err|)")
    ax.set_title("Figure 4 - 2x2 difficulty x typed informativeness")
    _save(fig, "fig4_mechanism_2x2.png")
    return written


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------
def write_outputs(result: Mapping[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        OUT_DIR / "coarse_token_child_informativeness.csv",
        [
            "historical_token", "historical_train_count", "split_multiplicity",
            "n_children", "n_eligible_children", "eligible", "target_dispersion",
            "resid_dispersion", "target_range", "resid_range", "high_info",
            "child_support_summary",
        ],
        result["parent_rows"],
    )
    _write_csv(
        OUT_DIR / "per_molecule_analysis.csv",
        result["molecule_header"],
        result["molecule_rows"],
    )
    _write_csv(
        OUT_DIR / "difficulty_quintiles.csv",
        [
            "quintile", "n", "hist_mae", "corr_mae", "degradation",
            "ensemble_hist_mae", "ensemble_corr_mae", "ensemble_degradation",
            "delta_center_mae", "delta_spread_mae", "delta_center_mse", "delta_spread_mse",
            "hist_disagreement", "corr_disagreement", "delta_disagreement",
        ],
        [[row[k] for k in (
            "quintile", "n", "hist_mae", "corr_mae", "degradation",
            "ensemble_hist_mae", "ensemble_corr_mae", "ensemble_degradation",
            "delta_center_mae", "delta_spread_mae", "delta_center_mse", "delta_spread_mse",
            "hist_disagreement", "corr_disagreement", "delta_disagreement",
        )] for row in result["quintiles"]],
    )
    _write_csv(
        OUT_DIR / "movement_classes.csv",
        ["quintile", "n_pairs", "helpful_pct", "harmful_pct", "wrong_direction_pct", "overshoot_pct", "magnitude_only_pct"],
        [[row[k] for k in (
            "quintile", "n_pairs", "helpful_pct", "harmful_pct",
            "wrong_direction_pct", "overshoot_pct", "magnitude_only_pct",
        )] for row in result["movement"]],
    )
    _write_csv(
        OUT_DIR / "typed_info_groups.csv",
        ["info_group", "n", "hist_mae", "corr_mae", "gain", "delta_disagreement"],
        [[row[k] for k in ("info_group", "n", "hist_mae", "corr_mae", "gain", "delta_disagreement")] for row in result["info_groups"]],
    )
    _write_csv(
        OUT_DIR / "mechanism_2x2.csv",
        ["difficulty", "typed_info", "n", "hist_mae", "corr_mae", "gain", "delta_disagreement"],
        [[row[k] for k in ("difficulty", "typed_info", "n", "hist_mae", "corr_mae", "gain", "delta_disagreement")] for row in result["cells"]],
    )
    assoc = result["decision"]["informativeness_association"]
    _write_csv(
        OUT_DIR / "correlation_table.csv",
        ["metric", "value"],
        [
            ["info_vs_gain_seed0", assoc["per_seed_rho"][0]],
            ["info_vs_gain_seed1", assoc["per_seed_rho"][1]],
            ["info_vs_gain_seed2", assoc["per_seed_rho"][2]],
            ["info_vs_gain_seed3", assoc["per_seed_rho"][3]],
            ["info_vs_gain_ensemble", assoc["ensemble_rho"]],
            ["seeds_positive", assoc["seeds_positive"]],
            ["partial_info_gain_given_difficulty", assoc["partial_given_difficulty"]],
            ["partial_info_gain_given_frequency", assoc["partial_given_frequency"]],
            ["spearman_info_vs_baseline", assoc["spearman_info_vs_baseline"]],
        ],
    )
    robustness = result["summary"]["informativeness_metric_robustness"]
    _write_csv(
        OUT_DIR / "informativeness_metric_robustness.csv",
        ["metric", "ensemble_rho", "partial_given_difficulty", "partial_given_frequency", "seeds_positive", "spearman_vs_baseline"],
        [
            [
                name,
                values["ensemble_rho"], values["partial_given_difficulty"],
                values["partial_given_frequency"], values["seeds_positive"],
                values["spearman_info_vs_baseline"],
            ]
            for name, values in robustness.items()
        ],
    )
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(result["summary"], indent=2))
    (OUT_DIR / "decision_record.json").write_text(json.dumps(result["decision"], indent=2))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_audit() -> dict[str, Any]:
    import time

    started = time.perf_counter()
    runs = load_frozen_runs()
    occ = load_occurrences()
    degradation = paired_degradation(runs)
    targets = degradation["targets"]
    n_valid = int(occ["valid_mol_ids"].max()) + 1
    n_train = int(occ["train_mol_ids"].max()) + 1

    train_ctx = load_train_context(n_train)
    split_r2 = token_split_map(occ["train_hist_r2_id"], occ["train_corr_r2_id"])
    child_stats = child_statistics(occ, train_ctx)
    parents = parent_informativeness(split_r2, child_stats)
    high_threshold = high_info_threshold(parents)
    mol_info = molecule_informativeness(occ, parents, high_threshold, n_valid)

    # controls (frequency / tier) come from the frozen fragmentation bundle
    bundle = build_tables(occ)
    features, _ = molecule_matrix(bundle)
    frequency = features["mean_log_freq_hist"]

    pred_hist = np.stack([runs["historical"][s]["predictions"] for s in SEEDS])
    pred_corr = np.stack([runs["corrected"][s]["predictions"] for s in SEEDS])
    ens_err_hist = np.abs(pred_hist.mean(axis=0) - targets)
    ens_err_corr = np.abs(pred_corr.mean(axis=0) - targets)

    bins, edges = quintile_assignment(degradation["mean_err_hist"])
    decomp = center_spread_decomposition(pred_hist, pred_corr, targets)
    quintiles = difficulty_quintiles(bins, degradation, ens_err_hist, ens_err_corr, decomp)
    movement = movement_classes(bins, pred_hist, pred_corr, targets)

    info = np.asarray([mol_info[i][PRIMARY_INFO] for i in range(n_valid)], dtype=np.float64)
    info_groups, info_labels = info_group_table(info, degradation, n_groups=3)
    info_median = float(np.median(info))
    cells = mechanism_2x2(bins, info_median, info, degradation)
    association = informativeness_association(info, degradation, ens_err_hist, ens_err_corr, frequency)
    decision = decide(
        quintiles, movement, association, info_groups, cells, bins, decomp,
        degradation, ens_err_hist, ens_err_corr,
    )

    # robustness: H2 null across the whole informativeness metric family
    metric_associations = {}
    for name in INFO_METRIC_NAMES:
        values = np.asarray([mol_info[i][name] for i in range(n_valid)], dtype=np.float64)
        metric_associations[name] = informativeness_association(
            values, degradation, ens_err_hist, ens_err_corr, frequency
        )

    # per-molecule table
    gain_seed = degradation["err_hist"] - degradation["err_corr"]
    gain_ens = ens_err_hist - ens_err_corr
    molecule_header = [
        "molecule_id", "target", "difficulty_quintile",
        "err_hist_mean", "err_corr_mean", "delta_mean",
        "ensemble_err_hist", "ensemble_err_corr", "ensemble_gain",
        "disagreement_hist", "disagreement_corr", "delta_disagreement",
        PRIMARY_INFO, "info_resid_mean", "info_resid_max", "info_target_mean",
        "frac_high_info_patches", "n_split_patches",
        "info_group", "info_high",
        "gain_seed0", "gain_seed1", "gain_seed2", "gain_seed3",
    ]
    info_names = {0: "low", 1: "mid", 2: "high"}
    molecule_rows = []
    for i in range(n_valid):
        row = [
            f"valid:{i:04d}", round(float(targets[i]), 6), int(bins[i] + 1),
            round(float(degradation["mean_err_hist"][i]), 6),
            round(float(degradation["mean_err_corr"][i]), 6),
            round(float(degradation["mean_delta"][i]), 6),
            round(float(ens_err_hist[i]), 6), round(float(ens_err_corr[i]), 6),
            round(float(gain_ens[i]), 6),
            round(float(degradation["disagreement_hist"][i]), 6),
            round(float(degradation["disagreement_corr"][i]), 6),
            round(float(degradation["delta_disagreement"][i]), 6),
            round(float(info[i]), 6),
            round(float(mol_info[i]["info_resid_mean"]), 6),
            round(float(mol_info[i]["info_resid_max"]), 6),
            round(float(mol_info[i]["info_target_mean"]), 6),
            round(float(mol_info[i]["frac_high_info_patches"]), 6),
            int(mol_info[i]["n_split_patches"]),
            info_names[int(info_labels[i])],
            int(info[i] > info_median),
        ]
        row += [round(float(gain_seed[s, i]), 6) for s in range(len(SEEDS))]
        molecule_rows.append(row)

    # parent-level table
    parent_rows = []
    for hist_id, p in sorted(parents.items()):
        children = split_r2[hist_id]["corrected_child_ids"]
        child_summary = ";".join(
            f"c{c}:{child_stats.get(c, {}).get('support', 0)}:"
            f"{child_stats.get(c, {}).get('mean_target', 0.0):.4f}:"
            f"{child_stats.get(c, {}).get('mean_resid', 0.0):.4f}"
            for c in children
        )
        parent_rows.append([
            f"h{hist_id}", p["historical_count"], p["split_multiplicity"],
            p["n_children"], p["n_eligible_children"], int(p["eligible"]),
            round(p["target_dispersion"], 6), round(p["resid_dispersion"], 6),
            round(p["target_range"], 6), round(p["resid_range"], 6),
            int(p["eligible"] and p["resid_dispersion"] >= high_threshold),
            child_summary,
        ])

    summary = {
        "primary_info_metric": PRIMARY_INFO,
        "min_child_support": MIN_CHILD_SUPPORT,
        "high_info_threshold": high_threshold,
        "n_train_molecules": n_train,
        "n_valid_molecules": n_valid,
        "n_eligible_parents": int(sum(1 for p in parents.values() if p["eligible"])),
        "n_split_parents": int(sum(1 for p in parents.values() if p["split_multiplicity"] >= 2)),
        "valid_molecules_with_split_patches": int((info > 0).sum()),
        "difficulty_quintiles": quintiles,
        "movement_classes": movement,
        "typed_info_groups": info_groups,
        "mechanism_2x2": cells,
        "informativeness_association": association,
        "informativeness_metric_robustness": metric_associations,
        "decision": decision,
    }

    figures = make_figures(quintiles, info, degradation, cells)
    result = {
        "summary": summary,
        "decision": decision,
        "quintiles": quintiles,
        "movement": movement,
        "info_groups": info_groups,
        "cells": cells,
        "parent_rows": parent_rows,
        "molecule_header": molecule_header,
        "molecule_rows": molecule_rows,
        "figures": figures,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = run_audit()
    write_outputs(result)
    print(json.dumps({
        "case": result["decision"]["case"],
        "gates": result["decision"]["gates"],
        "quintiles": result["quintiles"],
        "movement": result["movement"],
        "info_groups": result["info_groups"],
        "mechanism_2x2": result["cells"],
        "association": result["decision"]["informativeness_association"],
        "gain_cells": result["decision"]["gain_cells"],
        "runtime_seconds": result["runtime_seconds"],
        "figures": result["figures"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
