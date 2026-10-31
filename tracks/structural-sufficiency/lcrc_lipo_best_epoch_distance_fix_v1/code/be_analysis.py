"""Cross-seed tables, paired differences and the stability analysis.

Main ranking is the 3-seed mean Validation420 RMSE.  The 3-seed mean Test420
RMSE is shown in full next to it; a Validation/Test reversal is reported
explicitly and never used to change a training decision.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import be_common as C

#: Historical seed-0 runs whose training curve must be reproduced epoch-by-epoch
#: (only the checkpoint rule changes this round).
HISTORY_CURVES = {
    "base_s0": C.HIST_CONTEXT / "analysis/runs/base_s0",
    "roleblind_s0": C.HIST_MECH / "analysis/runs/roleblind_s0",
    "roleblind_s1": C.HIST_MECH / "analysis/runs/roleblind_s1",
    "g2l_loo_s0": C.HIST_COMPOSITION / "analysis/runs/g2l_loo_s0",
    "g2l_all_s0": C.HIST_COMPOSITION / "analysis/runs/g2l_all_s0",
    "dist_blind_s0": C.HIST_COMPOSITION / "analysis/runs/dist_blind_s0",
    "base_s1": C.HIST_MECH / "analysis/runs/base_s1",
}


def _read_curve(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec: dict[str, Any] = {}
            for key, value in row.items():
                if key == "is_best":
                    rec[key] = str(value) in ("True", "true", "1")
                else:
                    rec[key] = float(value)
            out.append(rec)
    return out


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _sd(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


# ---------------------------------------------------------------------------
# historical curve reproduction
# ---------------------------------------------------------------------------


def history_curve_check() -> dict[str, Any]:
    arms: dict[str, Any] = {}
    all_ok = True
    for arm, hist_dir in HISTORY_CURVES.items():
        new_curve_path = C.run_dir(arm) / "curve.csv"
        hist_curve_path = hist_dir / "curve.csv"
        if not new_curve_path.exists() or not hist_curve_path.exists():
            arms[arm] = {"available": False, "new": new_curve_path.exists(), "historical": hist_curve_path.exists()}
            all_ok = False
            continue
        new = _read_curve(new_curve_path)
        old = _read_curve(hist_curve_path)
        n = min(len(new), len(old))
        valid_diff = float(
            np.max(np.abs(np.asarray([r["valid_rmse_raw"] for r in new[:n]]) - np.asarray([r["valid_rmse_raw"] for r in old[:n]])))
        )
        train_diff = float(
            np.max(
                np.abs(
                    np.asarray([r["train_rmse_standardized"] for r in new[:n]])
                    - np.asarray([r["train_rmse_standardized"] for r in old[:n]])
                )
            )
        )
        new_best_epoch = int(np.argmin([r["valid_rmse_raw"] for r in new])) + 1
        old_best_epoch = int(np.argmin([r["valid_rmse_raw"] for r in old])) + 1
        identical = bool(len(new) == len(old) and valid_diff == 0.0 and train_diff == 0.0)
        all_ok &= identical
        arms[arm] = {
            "available": True,
            "historical_run": C.track_rel(hist_dir),
            "epochs_new": len(new),
            "epochs_historical": len(old),
            "max_abs_diff_valid_rmse": valid_diff,
            "max_abs_diff_train_rmse_standardized": train_diff,
            "bitwise_identical_curve": identical,
            "best_epoch_new": new_best_epoch,
            "best_epoch_historical": old_best_epoch,
            "best_valid_rmse_new": float(min(r["valid_rmse_raw"] for r in new)),
            "best_valid_rmse_historical": float(min(r["valid_rmse_raw"] for r in old)),
            "last_epoch_valid_rmse_historical": float(old[-1]["valid_rmse_raw"]),
        }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "scope": "validation420_only",
        "explanation": (
            "The new runs use the frozen R3 recipe and the frozen batch order, so every "
            "epoch-1..360 curve must equal the historical curve of the same family/seed; "
            "only the checkpoint rule (best single epoch instead of last-5 soup) differs. "
            "A bitwise-identical curve therefore proves the comparison isolates the "
            "checkpoint rule, and the historical soup value is exactly the last epoch "
            "column of that same curve."
        ),
        "all_bitwise_identical": bool(all_ok),
        "arms": arms,
    }
    C.write_json(C.HISTORY_CURVE_CHECK, payload)
    return payload


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _rows() -> list[dict[str, Any]]:
    test = C.read_json(C.EVAL_TEST / "metrics.json")["models"]
    out = []
    for arm in C.ARMS:
        train = C.load_train_json(arm)
        curve = _read_curve(C.run_dir(arm) / "curve.csv")
        valid_curve = np.asarray([r["valid_rmse_raw"] for r in curve])
        train_curve = np.asarray([r["train_rmse_standardized"] for r in curve])
        best_epoch = int(train["checkpoint"]["best_epoch"])
        after_best = valid_curve[best_epoch:]
        out.append(
            {
                "arm": arm,
                "model": C.MODEL_DISPLAY[train["family"]],
                "family": train["family"],
                "seed": int(train["seed"]),
                "best_epoch": best_epoch,
                "train_rmse": float(train["metrics"]["train"]["rmse"]),
                "valid_rmse": float(train["metrics"]["valid"]["rmse"]),
                "valid_mae": float(train["metrics"]["valid"]["mae"]),
                "valid_r2": float(train["metrics"]["valid"]["r2"]),
                "test_rmse": float(test[arm]["test"]["rmse"]),
                "test_mae": float(test[arm]["test"]["mae"]),
                "test_r2": float(test[arm]["test"]["r2"]),
                "parameters": int(train["model_audit"]["parameters"]),
                "train_wall_seconds": float(train["timing"]["train_wall_seconds"]),
                "valid_rmse_at_epoch_360": float(valid_curve[-1]),
                "train_rmse_std_at_epoch_360": float(train_curve[-1]),
                "valid_rmse_minus_at_360": float(valid_curve[-1] - valid_curve.min()),
                "late_epoch_valid_rise": float(after_best.max() - after_best.min()) if after_best.size else 0.0,
                "n_epochs_after_best": int(after_best.size),
                "best_epoch_fraction": float(best_epoch / len(curve)),
                "n_improvements": int(train["checkpoint"]["n_improvements"]),
                "checkpoint_is_final_epoch": bool(train["checkpoint"]["checkpoint_is_final_epoch_state"]),
                "curve": curve,
            }
        )
    return out


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def analyze() -> dict[str, Any]:
    history = history_curve_check()
    n_smoke = sum(1 for arm in C.ARMS if C.load_train_json(arm).get("smoke") is True)
    rows = _rows()
    by_arm = {row["arm"]: row for row in rows}

    _write_csv(
        C.TABLE_MAIN,
        [
            "model",
            "arm",
            "seed",
            "best_epoch",
            "train_rmse",
            "valid_rmse",
            "test_rmse",
            "valid_mae",
            "test_mae",
            "valid_r2",
            "test_r2",
            "parameters",
            "valid_rmse_at_epoch_360",
            "train_rmse_std_at_epoch_360",
        ],
        rows,
    )

    # ---------------- model level summary ----------------
    summary_rows = []
    base_by_seed = {row["seed"]: row for row in rows if row["family"] == "base"}
    for family in C.FAMILIES:
        fam_rows = [row for row in rows if row["family"] == family]
        valid = [row["valid_rmse"] for row in fam_rows]
        test = [row["test_rmse"] for row in fam_rows]
        summary_rows.append(
            {
                "model": C.MODEL_DISPLAY[family],
                "family": family,
                "valid_rmse_mean": _mean(valid),
                "valid_rmse_sd": _sd(valid),
                "valid_rmse_min": float(np.min(valid)),
                "valid_rmse_max": float(np.max(valid)),
                "test_rmse_mean": _mean(test),
                "test_rmse_sd": _sd(test),
                "test_rmse_min": float(np.min(test)),
                "test_rmse_max": float(np.max(test)),
                "valid_mae_mean": _mean([row["valid_mae"] for row in fam_rows]),
                "test_mae_mean": _mean([row["test_mae"] for row in fam_rows]),
                "valid_r2_mean": _mean([row["valid_r2"] for row in fam_rows]),
                "test_r2_mean": _mean([row["test_r2"] for row in fam_rows]),
                "best_epoch_min": int(min(row["best_epoch"] for row in fam_rows)),
                "best_epoch_median": float(np.median([row["best_epoch"] for row in fam_rows])),
                "best_epoch_max": int(max(row["best_epoch"] for row in fam_rows)),
                "best_epochs": "/".join(str(row["best_epoch"]) for row in sorted(fam_rows, key=lambda r: r["seed"])),
                "wins_valid_vs_base": int(
                    sum(1 for row in fam_rows if row["valid_rmse"] < base_by_seed[row["seed"]]["valid_rmse"])
                ),
                "wins_test_vs_base": int(
                    sum(1 for row in fam_rows if row["test_rmse"] < base_by_seed[row["seed"]]["test_rmse"])
                ),
            }
        )
    base_summary = next(row for row in summary_rows if row["family"] == "base")
    for row in summary_rows:
        row["vs_base_valid_mean_delta"] = row["valid_rmse_mean"] - base_summary["valid_rmse_mean"]
        row["vs_base_test_mean_delta"] = row["test_rmse_mean"] - base_summary["test_rmse_mean"]
    summary_rows.sort(key=lambda row: row["valid_rmse_mean"])
    _write_csv(
        C.TABLE_SUMMARY,
        [
            "model",
            "family",
            "valid_rmse_mean",
            "valid_rmse_sd",
            "test_rmse_mean",
            "test_rmse_sd",
            "valid_mae_mean",
            "test_mae_mean",
            "valid_r2_mean",
            "test_r2_mean",
            "vs_base_valid_mean_delta",
            "vs_base_test_mean_delta",
            "wins_valid_vs_base",
            "wins_test_vs_base",
            "best_epochs",
            "best_epoch_min",
            "best_epoch_median",
            "best_epoch_max",
            "valid_rmse_min",
            "valid_rmse_max",
            "test_rmse_min",
            "test_rmse_max",
        ],
        summary_rows,
    )

    # ---------------- paired per-seed differences ----------------
    pair_rows = []
    pair_summary: dict[str, Any] = {}
    for ref_family, cand_family in C.PAIRS:
        label = C.PAIR_LABEL[(ref_family, cand_family)]
        per_seed = []
        for seed in C.SEEDS:
            ref = by_arm[C.arm_key(ref_family, seed)]
            cand = by_arm[C.arm_key(cand_family, seed)]
            rec = {
                "pair": label,
                "ref_model": C.MODEL_DISPLAY[ref_family],
                "cand_model": C.MODEL_DISPLAY[cand_family],
                "seed": seed,
                "ref_valid_rmse": ref["valid_rmse"],
                "cand_valid_rmse": cand["valid_rmse"],
                "valid_gain": ref["valid_rmse"] - cand["valid_rmse"],
                "ref_test_rmse": ref["test_rmse"],
                "cand_test_rmse": cand["test_rmse"],
                "test_gain": ref["test_rmse"] - cand["test_rmse"],
                "ref_best_epoch": ref["best_epoch"],
                "cand_best_epoch": cand["best_epoch"],
            }
            per_seed.append(rec)
            pair_rows.append(rec)
        valid_gains = [row["valid_gain"] for row in per_seed]
        test_gains = [row["test_gain"] for row in per_seed]
        pair_summary[label] = {
            "ref_model": C.MODEL_DISPLAY[ref_family],
            "cand_model": C.MODEL_DISPLAY[cand_family],
            "valid_gain_per_seed": valid_gains,
            "valid_gain_mean": _mean(valid_gains),
            "valid_gain_sd": _sd(valid_gains),
            "valid_gain_min": float(np.min(valid_gains)),
            "valid_gain_max": float(np.max(valid_gains)),
            "valid_wins": int(sum(1 for g in valid_gains if g > 0)),
            "test_gain_per_seed": test_gains,
            "test_gain_mean": _mean(test_gains),
            "test_gain_sd": _sd(test_gains),
            "test_gain_min": float(np.min(test_gains)),
            "test_gain_max": float(np.max(test_gains)),
            "test_wins": int(sum(1 for g in test_gains if g > 0)),
            "sign_agreement_valid_vs_test": bool(
                np.sign(_mean(valid_gains)) == np.sign(_mean(test_gains))
            ),
        }
    _write_csv(
        C.TABLE_PAIRS,
        [
            "pair",
            "seed",
            "ref_model",
            "cand_model",
            "ref_valid_rmse",
            "cand_valid_rmse",
            "valid_gain",
            "ref_test_rmse",
            "cand_test_rmse",
            "test_gain",
            "ref_best_epoch",
            "cand_best_epoch",
        ],
        pair_rows,
    )

    # ---------------- seed / stability table ----------------
    stability_rows = []
    for family in C.FAMILIES:
        fam_rows = sorted([row for row in rows if row["family"] == family], key=lambda r: r["seed"])
        stability_rows.append(
            {
                "model": C.MODEL_DISPLAY[family],
                "family": family,
                "best_epochs": "/".join(str(row["best_epoch"]) for row in fam_rows),
                "best_epoch_min": int(min(row["best_epoch"] for row in fam_rows)),
                "best_epoch_max": int(max(row["best_epoch"] for row in fam_rows)),
                "valid_rmse_spread": float(max(row["valid_rmse"] for row in fam_rows) - min(row["valid_rmse"] for row in fam_rows)),
                "test_rmse_spread": float(max(row["test_rmse"] for row in fam_rows) - min(row["test_rmse"] for row in fam_rows)),
                "valid_rmse_mean": _mean([row["valid_rmse"] for row in fam_rows]),
                "valid_rmse_sd": _sd([row["valid_rmse"] for row in fam_rows]),
                "train_rmse_at_best_epoch_mean": _mean([row["train_rmse"] for row in fam_rows]),
                "train_rmse_std_at_360_mean": _mean([row["train_rmse_std_at_epoch_360"] for row in fam_rows]),
                "valid_at_360_mean": _mean([row["valid_rmse_at_epoch_360"] for row in fam_rows]),
                "valid_at_360_minus_best_mean": _mean([row["valid_rmse_minus_at_360"] for row in fam_rows]),
                "late_epoch_valid_rise_mean": _mean([row["late_epoch_valid_rise"] for row in fam_rows]),
                "n_runs_where_360_is_worse_than_best": int(
                    sum(1 for row in fam_rows if row["valid_rmse_minus_at_360"] > 0)
                ),
                "n_runs_where_best_is_final_epoch": int(sum(1 for row in fam_rows if row["checkpoint_is_final_epoch"])),
                "n_improvements_mean": _mean([row["n_improvements"] for row in fam_rows]),
            }
        )
    stability_rows.sort(key=lambda row: row["valid_rmse_mean"])
    _write_csv(
        C.TABLE_STABILITY,
        [
            "model",
            "family",
            "best_epochs",
            "best_epoch_min",
            "best_epoch_max",
            "valid_rmse_spread",
            "test_rmse_spread",
            "valid_rmse_mean",
            "valid_rmse_sd",
            "train_rmse_at_best_epoch_mean",
            "train_rmse_std_at_360_mean",
            "valid_at_360_mean",
            "valid_at_360_minus_best_mean",
            "late_epoch_valid_rise_mean",
            "n_runs_where_360_is_worse_than_best",
            "n_runs_where_best_is_final_epoch",
            "n_improvements_mean",
        ],
        stability_rows,
    )
    _write_csv(
        C.TABLE_SEED,
        [
            "model",
            "arm",
            "seed",
            "best_epoch",
            "best_epoch_fraction",
            "valid_rmse",
            "valid_rmse_at_epoch_360",
            "valid_rmse_minus_at_360",
            "late_epoch_valid_rise",
            "train_rmse",
            "train_rmse_std_at_epoch_360",
            "test_rmse",
            "n_improvements",
            "checkpoint_is_final_epoch",
        ],
        [dict(row) for row in rows],
    )

    # ---------------- interpretation helpers ----------------
    mean_valid = {row["family"]: row["valid_rmse_mean"] for row in summary_rows}
    mean_test = {row["family"]: row["test_rmse_mean"] for row in summary_rows}
    valid_order = [row["family"] for row in summary_rows]
    test_order = [
        row["family"] for row in sorted(summary_rows, key=lambda r: r["test_rmse_mean"])
    ]
    rank_inversions = [
        {"family": fam, "valid_rank": i + 1, "test_rank": test_order.index(fam) + 1}
        for i, fam in enumerate(valid_order)
        if test_order.index(fam) != i
    ]
    pair_reversals = [
        label
        for label, info in pair_summary.items()
        if not info["sign_agreement_valid_vs_test"]
    ]
    best_single_seed = {}
    for family in C.FAMILIES:
        fam_rows = sorted([row for row in rows if row["family"] == family], key=lambda r: r["valid_rmse"])
        best = fam_rows[0]
        best_single_seed[C.MODEL_DISPLAY[family]] = {
            "best_seed_by_valid": best["seed"],
            "valid_rmse_best_seed": best["valid_rmse"],
            "test_rmse_at_best_valid_seed": best["test_rmse"],
            "valid_rmse_3seed_mean": mean_valid[family],
            "test_rmse_3seed_mean": mean_test[family],
            "valid_optimism_of_single_seed": mean_valid[family] - best["valid_rmse"],
            "test_optimism_of_single_seed": mean_test[family] - best["test_rmse"],
            "min_test_rmse_across_seeds": float(min(row["test_rmse"] for row in rows if row["family"] == family)),
            "test_seed_at_min_test": int(
                min((row for row in rows if row["family"] == family), key=lambda r: r["test_rmse"])["seed"]
            ),
        }

    answers = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "scope": "EXPOSED_TEST_EXPLORATORY",
        "main_ranking_metric": "3_seed_mean_validation420_rmse",
        "model_summary": {
            C.MODEL_DISPLAY[row["family"]]: {
                "valid_rmse_mean": row["valid_rmse_mean"],
                "valid_rmse_sd": row["valid_rmse_sd"],
                "test_rmse_mean": row["test_rmse_mean"],
                "test_rmse_sd": row["test_rmse_sd"],
                "valid_mae_mean": row["valid_mae_mean"],
                "test_mae_mean": row["test_mae_mean"],
                "vs_base_valid_mean_delta": row["vs_base_valid_mean_delta"],
                "vs_base_test_mean_delta": row["vs_base_test_mean_delta"],
                "wins_valid_vs_base": row["wins_valid_vs_base"],
                "wins_test_vs_base": row["wins_test_vs_base"],
                "best_epochs": row["best_epochs"],
            }
            for row in summary_rows
        },
        "paired": pair_summary,
        "ranking": {
            "by_mean_valid": [C.MODEL_DISPLAY[f] for f in valid_order],
            "by_mean_test": [C.MODEL_DISPLAY[f] for f in test_order],
            "rank_inversions": rank_inversions,
            "pairs_with_sign_reversal": pair_reversals,
        },
        "single_seed_versus_3seed": best_single_seed,
        "stability": {row["model"]: row for row in stability_rows},
        "history_curve_check_passed": bool(history["all_bitwise_identical"]),
        "history_curves": history["arms"],
        "best_epoch_distribution": {
            C.MODEL_DISPLAY[family]: [
                {"seed": row["seed"], "best_epoch": row["best_epoch"]}
                for row in sorted([r for r in rows if r["family"] == family], key=lambda r: r["seed"])
            ]
            for family in C.FAMILIES
        },
        "n_smoke_runs": int(n_smoke),
        "references": {
            "pna_valid_rmse_historical": C.PNA_NATIVE_VALID_RMSE,
            "pna_test_rmse_historical": C.PNA_NATIVE_TEST_RMSE,
            "pna_note": "historical protocol (native training, PNA architecture); not a matched control",
        },
    }
    C.write_json(C.ANSWERS, answers)
    return answers


__all__ = ["analyze", "history_curve_check"]
