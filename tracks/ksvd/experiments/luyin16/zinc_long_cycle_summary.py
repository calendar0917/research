"""ZINC long-cycle audit - final summary tables (A-E), refined analyses.

Reads the stage outputs of ``zinc_long_cycle_audit`` and produces the final
tables used by the audit note:

* Table A  - extreme target decomposition (label-implied cycle values)
* Table B  - validation error by label-implied cycle severity
* Table C  - basis instability of extreme-tail molecules
* Table D  - cycle statistic comparison (invariant vs basis artifact)
* Table E  - oracle results (copy; curated in the note)
* split composition by label-implied cycle, learned-fraction (lambda) analysis
* updated figures 2/4 (label-based), extended permutation checks

All fits are train-only; validation is evaluate-only; test is descriptive.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error

AUDIT_ROOT = Path(__file__).resolve().parents[2] / "results" / "zinc_long_cycle_audit"


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(AUDIT_ROOT / name)
    if "molecule_id" in df.columns:
        df["molecule_id"] = df["molecule_id"].astype(str)
    return df


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(AUDIT_ROOT / name, index=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> int:
    refine = json.loads((AUDIT_ROOT / "stage_refine.json").read_text())
    sC = refine["constants_fitted"]["sigma_cycle"]
    muC = refine["constants_fitted"]["mu_cycle"]

    lab = _load("label_effective_cycle.csv")
    lab["label_excess"] = -lab["label_effective_cycle_snapped"].clip(upper=0)
    lab["label_cycle_component"] = lab["label_normalized_cycle_component"]

    # exact longest simple cycle for ALL molecules (invariant ground truth)
    exact_all = _load("exact_longest_all.csv")
    exact_all = exact_all.rename(columns={
        "longest_simple_cycle_length": "longest_simple_cycle_length_exact_all"})

    # randomized-max-basis distribution (invariant estimator of the order-
    # dependent label statistic), K=10 random insertion orders per molecule
    rnd = _load("randomized_maxbasis.csv")

    splits = {}
    for split in ("train", "valid", "test"):
        df = _load(f"{split}_cycle_audit.csv")
        df = df.merge(
            lab[["molecule_id", "label_effective_cycle_snapped", "label_excess",
                 "label_cycle_component"]],
            on="molecule_id", how="left",
        )
        df["y_without_cycle_label"] = df["target"] - df["label_cycle_component"]
        df = df.merge(exact_all[["molecule_id", "longest_simple_cycle_length_exact_all"]],
                      on="molecule_id", how="left")
        df["longest_simple_cycle_length"] = df["longest_simple_cycle_length_exact_all"]
        df = df.merge(rnd[["molecule_id", "mean_maxbasis", "max_maxbasis",
                           "min_maxbasis", "frac_ge7", "frac_ge9"]],
                      on="molecule_id", how="left")
        splits[split] = df
        _save(df, f"{split}_cycle_audit_label.csv")

    # ---------------- Table A: extreme target decomposition -----------------
    ta_rows = []
    for split, df in splits.items():
        sub = df.sort_values("target").head(50)
        ta_rows.append(sub)
        sub2 = df[df["target"] < -4]
        if len(sub2):
            ta_rows.append(sub2.sort_values("target"))
    ta = pd.concat(ta_rows, ignore_index=True).drop_duplicates("molecule_id").copy()
    ta["cycle_component_label"] = ta["label_cycle_component"]
    ta["y_without_cycle"] = ta["y_without_cycle_label"]
    cols = [
        "molecule_id", "split", "target", "max_basis_cycle_length",
        "excess_basis_cycle_length", "label_effective_cycle_snapped",
        "label_cycle_component", "y_without_cycle_label",
        "longest_simple_cycle_length", "rdkit_max_ring_size",
    ]
    _save(ta[cols], "table_a_extreme_decomposition.csv")

    # ------- extreme-tail: how much of it is cycle vs chemical -------------
    tail_rows = []
    for split, df in splits.items():
        for thr in (-4, -6, -10):
            sub = df[df["target"] < thr]
            tail_rows.append({
                "split": split,
                "threshold": thr,
                "n": int(len(sub)),
                "n_with_label_cycle_lt_0": int(
                    (sub["label_effective_cycle_snapped"] < 0).sum()),
                "n_with_label_cycle_eq_0": int(
                    (sub["label_effective_cycle_snapped"] == 0).sum()),
                "n_with_label_cycle_le_m2": int(
                    (sub["label_effective_cycle_snapped"] <= -2).sum()),
                "min_y": float(sub["target"].min()) if len(sub) else np.nan,
                "max_label_excess": float(sub["label_excess"].max()) if len(sub) else np.nan,
            })
    _save(pd.DataFrame(tail_rows), "extreme_tail_composition.csv")

    # -------------- Table B: valid error by label cycle severity -----------
    valid = splits["valid"]
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 100.5]
    labels = ["excess=0", "excess=1", "excess=2", "excess=3", "excess>=4"]
    valid["excess_label_group"] = pd.cut(
        valid["label_excess"], bins=bins, labels=labels
    )
    b_rows = []
    for group, sub in valid.groupby("excess_label_group", observed=True):
        b_rows.append({
            "excess_group": str(group),
            "n": int(len(sub)),
            "target_mean": float(sub["target"].mean()),
            "prediction_mean": float(sub["compact_v2_prediction"].mean()),
            "signed_residual_mean": float(sub["signed_residual"].mean()),
            "MAE": float(mean_absolute_error(sub["target"].to_numpy(),
                                             sub["compact_v2_prediction"].to_numpy())),
            "mae_mass_share": float(sub["absolute_error"].sum()
                                    / valid["absolute_error"].sum()),
            "target_min": float(sub["target"].min()),
            "cycle_comp_mean": float(sub["label_cycle_component"].mean()),
            "lambda_learned": float(
                1.0 - sub["signed_residual"].mean() / sub["label_cycle_component"].mean()
            ) if abs(sub["label_cycle_component"].mean()) > 1e-9 else np.nan,
        })
    _save(pd.DataFrame(b_rows), "table_b_valid_error_by_label_excess.csv")

    # ------------------- Table C: basis instability -------------------------
    perm = _load("permutation_sensitivity.csv")
    perm = perm.merge(
        lab[["molecule_id", "label_effective_cycle_snapped", "label_excess"]],
        on="molecule_id", how="left",
    )
    perm["perm_instability"] = (perm["num_unique_basis_lengths"] > 1).astype(int)

    def max_excess_variation(row: Any) -> float:
        # excess span implied by perm max/min basis lengths
        hi = max(row["perm_max"] - 6.0, 0.0)
        lo = max(row["perm_min"] - 6.0, 0.0)
        return hi - lo

    perm["excess_span_over_perms"] = perm.apply(max_excess_variation, axis=1)
    _save(perm, "table_c_basis_instability.csv")

    # ------------------ Table D: statistic comparison -----------------------
    all_df = pd.concat(splits.values(), ignore_index=True)
    d_rows = []
    stats_defs = [
        ("max_basis_cycle_length", "benchmark/candidate max basis-cycle (stored order)",
         False),
        ("cycle_score_gvae_order", "GVAE-replicated max basis-cycle (candidate)",
         False),
        ("label_effective_cycle_snapped", "label-implied cycle value (decomposition)",
         False),
        ("max_min_basis_cycle_length", "max min-cycle-basis length", True),
        ("total_min_basis_cycle_length", "total min-cycle-basis length", True),
        ("longest_simple_cycle_length", "longest simple cycle (exact, all 12k)",
         True),
        ("rdkit_max_ring_size", "RDKit max ring size", True),
        ("mean_maxbasis", "mean of max basis-cycle over 10 random orders",
         True),
        ("frac_ge7", "fraction of random orders with max basis cycle >= 7",
         True),
    ]
    for col, label, invariant in stats_defs:
        a = all_df[col].astype(float)
        ok = ~np.isnan(a)
        if ok.sum() < 20:
            continue
        r = scipy_stats.pearsonr(all_df.loc[ok, "target"], a[ok])
        rs = scipy_stats.spearmanr(all_df.loc[ok, "target"], a[ok])
        v = valid.copy()
        v["_stat"] = v[col].astype(float)
        okv = v["_stat"].notna() & v["absolute_error"].notna()
        rv_abs = scipy_stats.pearsonr(v.loc[okv, "absolute_error"], v.loc[okv, "_stat"]) if okv.sum() > 20 else (np.nan, np.nan)
        d_rows.append({
            "statistic": label,
            "column": col,
            "corr_with_y_pearson": float(r[0]),
            "corr_with_y_spearman": float(rs[0]),
            "corr_with_abs_residual_valid_pearson": float(rv_abs[0]),
            "n_corr": int(ok.sum()),
            "permutation_invariant": "YES" if invariant else ("NO" if col in (
                "max_basis_cycle_length", "cycle_score_gvae_order") else "?")
        })
    # empirical invariance verdicts from the perm stage
    d_rows.append({
        "statistic": "max basis-cycle (empirical, 655x100 perms)",
        "column": "-",
        "corr_with_y_pearson": np.nan,
        "corr_with_y_spearman": np.nan,
        "corr_with_abs_residual_valid_pearson": np.nan,
        "n_corr": 655,
        "permutation_invariant": f"NO (unstable on {int(perm['perm_instability'].sum())}/{len(perm)})",
    })
    d_rows.append({
        "statistic": "max min-cycle-basis (empirical, 20x20 perms)",
        "column": "max_min_basis_cycle_length",
        "corr_with_y_pearson": np.nan,
        "corr_with_y_spearman": np.nan,
        "corr_with_abs_residual_valid_pearson": np.nan,
        "n_corr": 20,
        "permutation_invariant": "YES (0 unstable)",
    })
    d_rows.append({
        "statistic": "longest simple cycle (empirical, 5x10 perms)",
        "column": "longest_simple_cycle_length",
        "corr_with_y_pearson": np.nan,
        "corr_with_y_spearman": np.nan,
        "corr_with_abs_residual_valid_pearson": np.nan,
        "n_corr": 5,
        "permutation_invariant": "YES (0 unstable)",
    })
    _save(pd.DataFrame(d_rows), "table_d_cycle_statistic_comparison.csv")

    # ------------ Table E: oracle (curated copy) ----------------------------
    oracle = _load("oracle_results.csv")
    _save(oracle, "table_e_oracle.csv")

    # ------- split composition by label-implied excess ----------------------
    comp_rows = []
    for split, df in splits.items():
        exc = df["label_excess"].to_numpy()
        comp_rows.append({
            "split": split,
            "n": int(len(df)),
            "excess_ge_1": int((exc >= 1).sum()),
            "excess_ge_2": int((exc >= 2).sum()),
            "excess_ge_4": int((exc >= 4).sum()),
            "excess_ge_6": int((exc >= 6).sum()),
            "excess_ge_8": int((exc >= 8).sum()),
            "max_excess": float(exc.max()) if len(exc) else np.nan,
            "frac_excess_ge_1": float((exc >= 1).mean()),
            "cycle_term_sum": float(df["label_cycle_component"].sum()),
        })
    _save(pd.DataFrame(comp_rows), "split_cycle_composition_label.csv")

    # ----------------- lambda (learned fraction) analysis -------------------
    lam_rows = []
    for group, sub in valid.groupby("excess_label_group", observed=True):
        comp = sub["label_cycle_component"].mean()
        resid = sub["signed_residual"].mean()
        lam_rows.append({
            "excess_group": str(group),
            "n": int(len(sub)),
            "cycle_component_mean": float(comp),
            "signed_residual_mean": float(resid),
            "lambda_learned": float(1.0 - resid / comp) if abs(comp) > 1e-9 else np.nan,
        })
    _save(pd.DataFrame(lam_rows), "lambda_learned_by_severity.csv")

    # ------------------- updated figures 2 + 4 ------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
    colors = {"train": "#4C72B0", "valid": "#DD8452", "test": "#55A868"}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for split, sub in all_df.groupby("split"):
        ax.scatter(sub["label_cycle_component"], sub["target"], s=7, alpha=0.35,
                   color=colors[split], label=split)
    ax.set_xlabel("label-implied normalized cycle component")
    ax.set_ylabel("target y")
    ax.set_title("Figure 2 (label-based): target y vs normalized cycle component")
    ax.legend()
    fig.tight_layout()
    fig.savefig(AUDIT_ROOT / "figures" / "fig2_target_vs_cycle_component_label.png",
                bbox_inches="tight")
    plt.close(fig)

    b = pd.read_csv(AUDIT_ROOT / "table_b_valid_error_by_label_excess.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(range(len(b)), b["MAE"], color=colors["valid"], alpha=0.8)
    for i, row in b.iterrows():
        ax.text(i, row["MAE"] + 0.01 * max(b["MAE"]), f'n={int(row["n"])}',
                ha="center", fontsize=8)
    ax.set_xticks(range(len(b)))
    ax.set_xticklabels(b["excess_group"], rotation=15)
    ax.set_ylabel("validation MAE")
    ax.set_title("Figure 4 (label-based): validation MAE by label cycle excess")
    fig.tight_layout()
    fig.savefig(AUDIT_ROOT / "figures" / "fig4_valid_mae_by_excess_label.png",
                bbox_inches="tight")
    plt.close(fig)

    summary = {
        "tables": [str(AUDIT_ROOT / n) for n in (
            "table_a_extreme_decomposition.csv",
            "table_b_valid_error_by_label_excess.csv",
            "table_c_basis_instability.csv",
            "table_d_cycle_statistic_comparison.csv",
            "table_e_oracle.csv",
            "extreme_tail_composition.csv",
            "split_cycle_composition_label.csv",
            "lambda_learned_by_severity.csv",
        )],
        "label_cycle_counts": {
            "cycle_value_lattice": sorted(
                set(lab["label_effective_cycle_snapped"].tolist())
            ),
            "n_with_cycle_zero": int((lab["label_effective_cycle_snapped"] == 0).sum()),
            "n_with_cycle_negative": int((lab["label_effective_cycle_snapped"] < 0).sum()),
        },
    }
    (AUDIT_ROOT / "LONG_CYCLE_AUDIT_SUMMARY.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
