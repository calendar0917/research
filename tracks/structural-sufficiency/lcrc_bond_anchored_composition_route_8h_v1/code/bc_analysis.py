"""Stage A/B aggregation: prediction tables, paired bootstrap and branch utilisation.

Reads only ``results/<institution>/<task>/<variant>_s<seed>/`` plus the frozen
historical prediction tables (read-only).  Writes

* ``analysis/prediction_table.csv``    per-molecule predictions of every unit
* ``analysis/paired_comparisons.json`` paired molecule bootstrap (2 000 draws)
* ``analysis/branch_utilization.json`` residual size, branch-off identity, concentration
* ``analysis/stage_decisions.json``    the pre-registered trigger rule
* ``analysis/compute_ledger.json``     per-unit elapsed seconds / GPU·h

Nothing is trained here and no Test number is ever used for a decision.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import bc_common as C

#: Historical anchors (read-only prediction tables of the sprint round).
HIST_LIPO = "L_lipophilicity"
HIST_WC = "L_WC_LOGP"
HIST_GINE = "G0_lipophilicity"

COMPARISONS = {
    "lipo": [("JOINT", "MARGINAL", "gain_null"), ("JOINT", "C1", "gain_base")],
    "wc_logp": [("JOINT", "MARGINAL", "gain_null"), ("JOINT", "C1", "gain_base")],
}


# ---------------------------------------------------------------------------
# unit discovery
# ---------------------------------------------------------------------------


def discover() -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for status_path in sorted(C.RESULTS_DIR.glob("*/*/*/status.json")):
        status = C.read_json(status_path)
        if not status.get("formal", False):
            continue
        if status.get("epochs_run", 0) <= 2:
            continue
        status["dir"] = str(status_path.parent)
        units.append(status)
    return units


def load_predictions(unit: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    path = Path(unit["dir"]) / "predictions.npz"
    with np.load(path, allow_pickle=False) as z:
        keys = [k for k in z.files if k.startswith(f"{split}_")]
        return {k[len(split) + 1 :]: np.asarray(z[k]) for k in keys}


def hist_predictions(name: str) -> dict[str, dict[str, np.ndarray]]:
    XR = C.modules()["xr_common"]
    path = XR.RESULTS_DIR / "replay" / f"{name}_predictions.npz"
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as z:
        for split in C.SPLITS:
            out[split] = {
                "row_id": np.asarray(z[f"{split}_row_id"]),
                "y_raw": np.asarray(z[f"{split}_y_raw"]),
                "pred_raw": np.asarray(z[f"{split}_pred_raw"]),
            }
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def bootstrap_paired(
    err_a: np.ndarray,
    err_b: np.ndarray,
    *,
    n_draws: int = C.BOOTSTRAP_DRAWS,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """``stat = RMSE(b) - RMSE(a)``; positive means ``a`` is better."""
    err_a = np.asarray(err_a, dtype=np.float64)
    err_b = np.asarray(err_b, dtype=np.float64)
    if err_a.shape != err_b.shape or err_a.size == 0:
        raise ValueError("paired residual vectors must have the same non-zero length")
    rng = np.random.default_rng(int(seed))
    n = err_a.size
    point_a = float(np.sqrt(np.mean(err_a**2)))
    point_b = float(np.sqrt(np.mean(err_b**2)))
    draws = np.empty(int(n_draws), dtype=np.float64)
    for d in range(int(n_draws)):
        idx = rng.integers(0, n, n)
        draws[d] = np.sqrt(np.mean(err_b[idx] ** 2)) - np.sqrt(np.mean(err_a[idx] ** 2))
    return {
        "n": int(n),
        "rmse_a": point_a,
        "rmse_b": point_b,
        "delta": float(point_b - point_a),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "prob_a_better": float(np.mean(draws > 0.0)),
        "draws": int(n_draws),
        "seed": int(seed),
        "mean_draw": float(draws.mean()),
    }


def concentration(err_a: np.ndarray, err_b: np.ndarray) -> dict[str, Any]:
    """Is a change driven by a handful of molecules?"""
    sa = np.asarray(err_a, dtype=np.float64) ** 2
    sb = np.asarray(err_b, dtype=np.float64) ** 2
    reduction = sb - sa  # negative => a better for that molecule
    total = float(np.sum(np.abs(reduction)))
    out: dict[str, Any] = {"total_abs_squared_error_change": total}
    if total == 0.0:
        return out | {"degenerate": True}
    order = np.argsort(-np.abs(reduction))
    n = reduction.size
    for frac in (0.01, 0.05):
        k = max(1, int(round(frac * n)))
        out[f"top_{int(frac*100)}pct_share_of_abs_change"] = float(
            np.sum(np.abs(reduction[order[:k]])) / total
        )
    n_better = int(np.sum(reduction < 0))
    out["n_molecules_better"] = n_better
    out["fraction_molecules_better"] = float(n_better / n)
    out["median_squared_error_change"] = float(np.median(reduction))
    # leave-the-worst-out: drop the 1% largest |change| and re-measure the gap
    k = max(1, int(round(0.01 * n)))
    keep = np.ones(n, dtype=bool)
    keep[order[:k]] = False
    out["delta_after_dropping_top_1pct"] = float(
        np.sqrt(np.mean(sb[keep])) - np.sqrt(np.mean(sa[keep]))
    )
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def build_tables() -> dict[str, Any]:
    units = discover()
    hist = {name: hist_predictions(name) for name in (HIST_LIPO, HIST_WC, HIST_GINE)}
    rows: list[dict[str, Any]] = []
    for unit in units:
        for split in C.SPLITS:
            pred = load_predictions(unit, split)
            n = int(pred["row_id"].size)
            for i in range(n):
                rows.append(
                    {
                        "unit": unit["unit"],
                        "institution": unit["institution"],
                        "task": unit["task"],
                        "variant": unit["variant"],
                        "seed": int(unit["seed"]),
                        "split": split,
                        "row_id": int(pred["row_id"][i]),
                        "y_raw": float(pred["y_raw"][i]),
                        "pred_raw": float(pred["pred_raw"][i]),
                        "base_raw": float(pred["base_raw"][i]),
                        "delta_raw": float(pred["delta_raw"][i]),
                        "n_paths3": int(pred["n_paths3"][i]),
                        "rmse_unit_split": float(
                            np.sqrt(np.mean((pred["pred_raw"] - pred["y_raw"]) ** 2))
                        ),
                    }
                )
    C.write_csv(
        C.ANALYSIS_DIR / "prediction_table.csv",
        rows,
        columns=["unit", "institution", "task", "variant", "seed", "split", "row_id", "y_raw",
                 "pred_raw", "base_raw", "delta_raw", "n_paths3", "rmse_unit_split"],
    )

    # paired comparisons
    comparisons: list[dict[str, Any]] = []
    index = {(u["institution"], u["task"], u["variant"], int(u["seed"])): u for u in units}
    anchors = {
        ("lipo", "C1"): hist[HIST_LIPO],
        ("wc_logp", "C1"): hist[HIST_WC],
        ("lipo", "GINE"): hist[HIST_GINE],
    }
    for institution in C.INSTITUTIONS:
        for task in C.TASKS:
            for seed in sorted({s for (inst, t, _v, s) in index if inst == institution and t == task}):
                pair: dict[str, Any] = {"institution": institution, "task": task, "seed": int(seed)}
                joint = index.get((institution, task, "JOINT", seed))
                marginal = index.get((institution, task, "MARGINAL", seed))
                if joint is None or marginal is None:
                    continue
                for split in ("valid", "test"):
                    jp = load_predictions(joint, split)
                    mp = load_predictions(marginal, split)
                    assert np.array_equal(jp["row_id"], mp["row_id"]), "paired rows must match"
                    err_j = jp["pred_raw"] - jp["y_raw"]
                    err_m = mp["pred_raw"] - mp["y_raw"]
                    entry = {
                        "split": split,
                        "joint_vs_marginal": bootstrap_paired(err_j, err_m),
                        "concentration": concentration(err_j, err_m),
                        "rmse_joint": float(np.sqrt(np.mean(err_j**2))),
                        "rmse_marginal": float(np.sqrt(np.mean(err_m**2))),
                        "rmse_epoch0": float(
                            np.sqrt(np.mean((jp["base_raw"] - jp["y_raw"]) ** 2))
                        ),
                    }
                    anchor = anchors.get((task, "C1"))
                    if anchor is not None and split in anchor:
                        ids = anchor[split]["row_id"]
                        if np.array_equal(ids, jp["row_id"]):
                            err_c1 = anchor[split]["pred_raw"] - anchor[split]["y_raw"]
                            entry["joint_vs_c1"] = bootstrap_paired(err_j, err_c1)
                            entry["rmse_c1"] = float(np.sqrt(np.mean(err_c1**2)))
                            entry["concentration_vs_c1"] = concentration(err_j, err_c1)
                    if task == "lipo" and split in hist[HIST_GINE]:
                        g = hist[HIST_GINE][split]
                        if np.array_equal(g["row_id"], jp["row_id"]):
                            err_g = g["pred_raw"] - g["y_raw"]
                            entry["rmse_gine"] = float(np.sqrt(np.mean(err_g**2)))
                            entry["joint_vs_gine"] = bootstrap_paired(err_j, err_g)
                    pair[split] = entry
                comparisons.append(pair)

    # branch utilisation / identity tables
    utilisation: list[dict[str, Any]] = []
    for unit in units:
        valid = load_predictions(unit, "valid")
        train = load_predictions(unit, "train")
        test = load_predictions(unit, "test")
        branch_off = {
            "rmse_valid_raw": float(np.sqrt(np.mean((valid["branch_off_pred_raw"] - valid["y_raw"]) ** 2))),
            "max_abs_diff_from_base": float(np.max(np.abs(valid["branch_off_pred_raw"] - valid["base_raw"]))),
        }
        joint = index.get((unit["institution"], unit["task"], "JOINT", int(unit["seed"])))
        marginal = index.get((unit["institution"], unit["task"], "MARGINAL", int(unit["seed"])))
        nonzero_fraction = None
        if joint is not None and marginal is not None and unit["variant"] == "JOINT":
            mp = load_predictions(marginal, "valid")
            nonzero_fraction = float(np.mean(np.abs(valid["pred_raw"] - mp["pred_raw"]) > 1e-8))
        utilisation.append(
            {
                "unit": unit["unit"],
                "institution": unit["institution"],
                "task": unit["task"],
                "variant": unit["variant"],
                "seed": int(unit["seed"]),
                "selected_epoch": int(unit["selected_epoch"]),
                "epochs_run": int(unit["epochs_run"]),
                "delta_rms_std_valid": float(unit["delta_rms_std_valid"]),
                "delta_rms_std_train": float(unit["delta_rms_std_train"]),
                "delta_over_teacher_std_valid": float(unit["delta_over_teacher_std"]),
                "teacher_valid_std": float(unit["teacher_valid_std"]),
                "rmse_valid": float(np.sqrt(np.mean((valid["pred_raw"] - valid["y_raw"]) ** 2))),
                "rmse_train": float(np.sqrt(np.mean((train["pred_raw"] - train["y_raw"]) ** 2))),
                "rmse_test": float(np.sqrt(np.mean((test["pred_raw"] - test["y_raw"]) ** 2))),
                "rmse_base_valid": float(np.sqrt(np.mean((valid["base_raw"] - valid["y_raw"]) ** 2))),
                "branch_off": branch_off,
                "mean_abs_delta_raw_valid": float(np.mean(np.abs(valid["delta_raw"]))),
                "molecules_with_nonzero_delta": int(np.sum(np.abs(valid["delta_raw"]) > 1e-9)),
                "n_valid": int(valid["row_id"].size),
                "fraction_molecules_joint_differs_from_marginal": nonzero_fraction,
                "events_per_molecule_mean_valid": float(np.mean(valid["n_paths3"])),
                "parameter_report": unit["parameter_report"],
                "gpu_hours": float(unit["gpu_hours"]),
                "elapsed_seconds": float(unit["elapsed_seconds"]),
                "git_commit": unit.get("git_commit"),
                "device": unit.get("device"),
                "paths3_sha256": unit.get("paths3_sha256"),
                "status_path": str(Path(unit["dir"]) / "status.json"),
            }
        )

    ledger = {
        "round_id": C.ROUND_ID,
        "units": [
            {
                "unit": u["unit"],
                "institution": u["institution"],
                "task": u["task"],
                "variant": u["variant"],
                "seed": int(u["seed"]),
                "elapsed_seconds": float(u["elapsed_seconds"]),
                "gpu_hours": float(u["gpu_hours"]),
                "device": u.get("device"),
                "epochs_run": int(u["epochs_run"]),
                "selected_epoch": int(u["selected_epoch"]),
                "git_commit": u.get("git_commit"),
                "paths3_sha256": u.get("paths3_sha256"),
                "checkpoint_best": str(Path(u["dir"]) / "checkpoint_best_val.pt"),
            }
            for u in units
        ],
        "slurm_or_process_failures_are_billed": True,
        "note": "elapsed_seconds is the wall clock of the whole unit process (model build + path3 verify + training + prediction)",
    }
    ledger["total_gpu_hours"] = float(sum(u["gpu_hours"] for u in ledger["units"]))
    ledger["total_elapsed_hours"] = float(sum(u["elapsed_seconds"] for u in ledger["units"]) / 3600.0)

    C.atomic_write_json(C.ANALYSIS_DIR / "paired_comparisons.json", {"comparisons": comparisons,
                                                                     "bootstrap": {"draws": C.BOOTSTRAP_DRAWS,
                                                                                   "seed": C.BOOTSTRAP_SEED,
                                                                                   "statistic": "RMSE(b)-RMSE(a); positive => a better"}})
    C.atomic_write_json(C.ANALYSIS_DIR / "branch_utilization.json", {"units": utilisation})
    C.atomic_write_json(C.ANALYSIS_DIR / "compute_ledger.json", ledger)
    return {"units": units, "comparisons": comparisons, "utilisation": utilisation, "ledger": ledger}


# ---------------------------------------------------------------------------
# the pre-registered trigger rule
# ---------------------------------------------------------------------------


def stage_decisions(built: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = built["comparisons"]
    triggers: dict[str, Any] = {"per_institution": {}, "detail": []}
    for institution in C.INSTITUTIONS:
        rows = [c for c in comparisons if c["institution"] == institution and c["task"] == "lipo"]
        if not rows:
            continue
        per_seed = []
        for row in rows:
            valid = row["valid"]
            vm = valid["joint_vs_marginal"]
            vc = valid.get("joint_vs_c1")
            util = next(
                (u for u in built["utilisation"]
                 if u["institution"] == institution and u["task"] == "lipo"
                 and u["variant"] == "JOINT" and u["seed"] == row["seed"]),
                None,
            )
            cond_a = vm["delta"] >= C.STAGE_C_JOINT_VS_MARGINAL
            cond_b = bool(vc is not None and vc["delta"] >= C.STAGE_C_JOINT_VS_C1)
            cond_c = vm["prob_a_better"] >= C.STAGE_C_BOOTSTRAP_PROB
            nondegenerate = bool(util is not None and util["mean_abs_delta_raw_valid"] > 1e-6
                                 and util["epochs_run"] > 1)
            train_gain = None
            train_gain_train = None
            reversed_ = None
            if util is not None:
                train_gain = util["rmse_base_valid"] - util["rmse_valid"]
                train_gain_train = util["rmse_base_train"] - util["rmse_train"]
                # "clearly reversed": the branch buys Train and loses more than
                # the pre-registered tolerance on Valid, or the mirror image.
                if train_gain_train is not None:
                    same_sign = (train_gain_train >= 0.0) == (train_gain >= 0.0)
                    reversed_ = bool((not same_sign) and abs(train_gain) >= C.REVERSAL_TOLERANCE)
            cond_d = bool(nondegenerate and reversed_ is False)
            per_seed.append(
                {
                    "seed": int(row["seed"]),
                    "joint_vs_marginal_valid_delta": vm["delta"],
                    "joint_vs_marginal_ci95": vm["ci95"],
                    "joint_vs_marginal_prob": vm["prob_a_better"],
                    "joint_vs_c1_valid_delta": None if vc is None else vc["delta"],
                    "joint_vs_c1_prob": None if vc is None else vc["prob_a_better"],
                    "branch_nondegenerate": nondegenerate,
                    "valid_gain_over_epoch0": train_gain,
                    "train_gain_over_epoch0": train_gain_train,
                    "clearly_reversed": reversed_,
                    "conditions": {"A_marginal_ge_0.010": cond_a, "B_c1_ge_0.015": cond_b,
                                   "C_bootstrap_prob_ge_0.80": cond_c, "D_nondegenerate_positive": cond_d},
                    "triggered": bool(cond_a or cond_b or cond_c or cond_d),
                }
            )
        triggers["per_institution"][institution] = {
            "seeds": per_seed,
            "triggered": bool(any(s["triggered"] for s in per_seed)),
            "rule": "for E2E at least one of seed0/seed1 must satisfy; all seeds are reported",
        }
    e2e = triggers["per_institution"].get("E2E", {}).get("triggered", False)
    frozen = triggers["per_institution"].get("FROZEN", {}).get("triggered", False)
    triggers["stage_c_required"] = bool(e2e or frozen)
    triggers["stage_c_institution"] = "E2E" if e2e else ("FROZEN" if frozen else None)
    triggers["rule"] = (
        "E2E confirmed when it triggers; FROZEN only when E2E does not; both -> E2E"
    )
    triggers["thresholds"] = {
        "joint_vs_marginal": C.STAGE_C_JOINT_VS_MARGINAL,
        "joint_vs_c1": C.STAGE_C_JOINT_VS_C1,
        "bootstrap_prob": C.STAGE_C_BOOTSTRAP_PROB,
    }
    triggers["validation_only"] = True
    triggers["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    C.atomic_write_json(C.ANALYSIS_DIR / "stage_decisions.json", triggers)
    return triggers


def main() -> None:
    built = build_tables()
    print(f"[analysis] {len(built['units'])} formal units, {len(built['comparisons'])} paired comparisons")
    triggers = stage_decisions(built)
    print("[analysis] stage C required:", triggers["stage_c_required"],
          "institution:", triggers["stage_c_institution"])
    print("[analysis] total GPU hours:", built["ledger"]["total_gpu_hours"])


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
