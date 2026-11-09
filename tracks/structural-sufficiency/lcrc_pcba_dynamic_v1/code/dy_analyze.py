"""Local analysis: the frozen comparison table, the per-task structure of the
difference and the pre-registered decision label.

Reads only artifacts already on disk (no training, no GPU, no new inference):
``runs/W2_DYNAMIC/*``, the historical W2 / RC-REAL run artifacts, the frozen BOC
reference predictions and this round's ``analysis/`` files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import dy_common as C  # noqa: E402

SCORABLE_MIN_SUPPORT = int(C.SUPPORT_MIN)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def _read_curve(path: Path) -> list[dict[str, Any]]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


def per_task_at(path: Path, epoch: int) -> dict[str, Any]:
    ap: list[float | None] = []
    positive: list[int] = []
    observed: list[int] = []
    strata: list[str] = []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if int(row["epoch"]) != int(epoch):
            continue
        ap.append(None if row["ap"] == "" else float(row["ap"]))
        positive.append(int(row["positive"]))
        observed.append(int(row["observed"]))
        strata.append(row["stratum"])
    if len(ap) != C.N_TASKS:
        raise RuntimeError(f"{path} epoch {epoch}: {len(ap)} tasks != {C.N_TASKS}")
    return {
        "per_task_ap": ap,
        "valid_positive": positive,
        "valid_observed": observed,
        "valid_stratum": strata,
    }


def load_run(run_dir: Path) -> dict[str, Any]:
    run = C.read_json(run_dir / "run.json")
    curve = _read_curve(C.REPO / run["curve"])
    best_epoch = int(run["best_epoch"])
    final_epoch = int(run["final_epoch"])
    return {
        "run": run,
        "curve": curve,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_valid_ap": float(run["best_valid_ap"]),
        "valid_ap_at_40": float(run["fixed_epoch_reports"][str(final_epoch)]["ap"]),
        "best": per_task_at(C.REPO / run["per_task_ap"], best_epoch),
        "at40": per_task_at(C.REPO / run["per_task_ap"], final_epoch),
        "gpu_hours": float(run["gpu_hours"]),
        "parameters": int(run["parameters"]),
    }


# ---------------------------------------------------------------------------
# task-level arithmetic
# ---------------------------------------------------------------------------


def delta_vector(a: Sequence[Any], b: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    deltas: list[float] = []
    idx: list[int] = []
    for t, (x, y) in enumerate(zip(a, b)):
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        deltas.append(float(x) - float(y))
        idx.append(t)
    return np.asarray(deltas, dtype=np.float64), np.asarray(idx, dtype=np.int64)


def subset_stats(
    deltas: np.ndarray,
    mask: np.ndarray,
    *,
    n_scored_total: int | None = None,
) -> dict[str, Any]:
    sel = deltas[mask]
    if sel.size == 0:
        return {"n_tasks": 0}
    out = {
        "n_tasks": int(sel.size),
        "mean_delta": float(sel.mean()),
        "median_delta": float(np.median(sel)),
        "improved": int((sel > 0).sum()),
        "degraded": int((sel < 0).sum()),
        "unchanged": int((sel == 0).sum()),
        "min_delta": float(sel.min()),
        "max_delta": float(sel.max()),
        "sum_delta": float(sel.sum()),
    }
    if n_scored_total:
        out["macro_contribution"] = float(sel.sum() / float(n_scored_total))
    out["sum_abs_delta"] = float(np.abs(sel).sum())
    return out


def extremes(
    rows: Sequence[Mapping[str, Any]], *, key: str = "delta_vs_w2", k: int = 5
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: r[key])
    return {
        "worst": ordered[: int(k)],
        "best": list(reversed(ordered[-int(k) :])),
    }


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(comparison: Mapping[str, Any]) -> dict[str, Any]:
    rules = C.DECISION_RULES
    best = comparison["headline"]
    high = comparison["test_high_support"]
    low_like = comparison["test_low_support"]
    usage = comparison["dynamic_usage"]
    used = bool(usage["dynamic_branch_used"])
    delta_test = float(best["delta_test_best"])
    delta_valid = float(best["delta_valid_best"])
    checks = {
        "delta_test_ge_0.010": delta_test >= rules["strong"]["delta_test_min"],
        "improved_high_support_ge_70": int(high["improved"]) >= rules["strong"]["improved_high_support_min"],
        "high_support_mean_ge_0.006": (high.get("mean_delta") or -1.0)
        >= rules["strong"]["high_support_mean_delta_min"],
        "validation_direction_positive": delta_valid > 0.0,
        "dynamic_branch_used": bool(used),
    }
    if not used:
        label = "MECHANISM_OR_OPTIMIZATION_INVALID"
        reason = "the dynamic branch is not numerically used"
    elif all(checks.values()):
        label = rules["strong"]["label"]
        reason = "all STRONG conditions met"
    elif delta_test >= rules["possible"]["delta_test_min"]:
        label = rules["possible"]["label"]
        reason = "overall gain in the 0.005..0.010 band or offset by low-support tasks"
    elif float(rules["none"]["delta_test_band"][0]) <= delta_test <= float(rules["none"]["delta_test_band"][1]):
        label = rules["none"]["label"]
        reason = "delta inside the pre-registered no-gain band"
    elif delta_test <= float(rules["regression"]["delta_test_max"]):
        broad = (high.get("improved") or 0) >= 40 or (high.get("degraded") or 0) >= 40
        label = rules["regression"]["label"] if broad else rules["none"]["label"]
        reason = (
            "negative delta with broad degradation"
            if broad
            else "negative delta driven by very-low-support tasks only"
        )
    else:
        label = rules["none"]["label"]
        reason = "delta between the pre-registered bands; report as measured"
    return {
        "label": label,
        "reason": reason,
        "measured": {
            "delta_test_best": delta_test,
            "delta_valid_best": delta_valid,
            "high_support_improved": int(high["improved"]),
            "high_support_tasks": int(high["n_tasks"]),
            "high_support_mean_delta": high.get("mean_delta"),
            "low_support_mean_delta": low_like.get("mean_delta"),
            "dynamic_branch_used": bool(used),
            "Rtotal_valid": usage["valid_best"]["Rtotal"],
        },
        "conditions": checks,
        "thresholds": dict(rules),
        "caveat": (
            "one seed, exposed Test split; the thresholds are engineering decisions, not "
            "significance tests, and this round has no capacity-matched STATIC control"
        ),
    }


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------


def build(analysis_dir: Path) -> dict[str, Any]:
    dyn = load_run(C.RUNS / "W2_DYNAMIC")
    w2 = load_run(C.W2_SEED0["run_json"].parent)
    rc = load_run(C.RC_ROUND / "runs/REAL")

    scores = C.read_json(analysis_dir / "test_scores.json")
    mech = C.read_json(analysis_dir / "mechanism.json")
    preflight = C.read_json(analysis_dir / "preflight.json")
    models = scores["models"]

    dyn_best_label = "W2_DYNAMIC_best_valid"
    dyn40_label = "W2_DYNAMIC_epoch40"
    test_ap = {
        "W2-DYNAMIC": models[dyn_best_label]["test_ap_official"],
        "W2-DYNAMIC@40": models[dyn40_label]["test_ap_official"],
        "W2": models["W2_best_valid"]["test_ap_official"],
        "W2@40": models["W2_epoch40"]["test_ap_official"],
        "RC-REAL": models["RC_REAL_best_valid"]["test_ap_official"],
        "RC-REAL@40": models["RC_REAL_epoch40"]["test_ap_official"],
    }
    test_independent = {
        "W2-DYNAMIC": models[dyn_best_label]["test_ap_independent"],
        "W2-DYNAMIC@40": models[dyn40_label]["test_ap_independent"],
        "W2": models["W2_best_valid"]["test_ap_independent"],
        "W2@40": models["W2_epoch40"]["test_ap_independent"],
        "RC-REAL": models["RC_REAL_best_valid"]["test_ap_independent"],
        "RC-REAL@40": models["RC_REAL_epoch40"]["test_ap_independent"],
    }

    # --- headline -----------------------------------------------------------------
    headline = {
        "delta_test_best": test_ap["W2-DYNAMIC"] - test_ap["W2"],
        "delta_test_best_vs_rc": test_ap["W2-DYNAMIC"] - test_ap["RC-REAL"],
        "delta_test_40": test_ap["W2-DYNAMIC@40"] - test_ap["W2@40"],
        "delta_test_40_vs_rc40": test_ap["W2-DYNAMIC@40"] - test_ap["RC-REAL@40"],
        "delta_valid_best": dyn["best_valid_ap"] - w2["best_valid_ap"],
        "delta_valid_best_vs_rc": dyn["best_valid_ap"] - rc["best_valid_ap"],
        "delta_valid_40": dyn["valid_ap_at_40"] - w2["valid_ap_at_40"],
        "test_ap": test_ap,
        "test_ap_independent": test_independent,
        "valid_ap": {
            "W2-DYNAMIC": dyn["best_valid_ap"],
            "W2-DYNAMIC@40": dyn["valid_ap_at_40"],
            "W2": w2["best_valid_ap"],
            "W2@40": w2["valid_ap_at_40"],
            "RC-REAL": rc["best_valid_ap"],
            "RC-REAL@40": rc["valid_ap_at_40"],
        },
        "best_epoch": {
            "W2-DYNAMIC": dyn["best_epoch"],
            "W2": w2["best_epoch"],
            "RC-REAL": rc["best_epoch"],
        },
        "gpu_hours": {
            "W2-DYNAMIC": dyn["gpu_hours"],
            "W2": w2["gpu_hours"],
            "RC-REAL": rc["gpu_hours"],
        },
        "parameters": {
            "W2-DYNAMIC": dyn["parameters"],
            "W2": w2["parameters"],
            "RC-REAL": rc["parameters"],
        },
        "official_vs_independent_max_abs_delta": float(
            max(abs(models[k]["test_ap_official"] - models[k]["test_ap_independent"]) for k in models)
        ),
    }

    # --- per task -----------------------------------------------------------------
    test_pos = np.asarray(models["W2_best_valid"]["test_positive"], dtype=np.int64)
    d_w2, idx = delta_vector(models[dyn_best_label]["test_per_task_ap"], models["W2_best_valid"]["test_per_task_ap"])
    d_rc, idx_rc = delta_vector(models[dyn_best_label]["test_per_task_ap"], models["RC_REAL_best_valid"]["test_per_task_ap"])
    if not np.array_equal(idx, idx_rc):
        raise RuntimeError("W2 and RC-REAL task universes differ")
    n_scored = int(idx.size)
    support = test_pos[idx]

    valid_pos_dyn = np.asarray(dyn["best"]["valid_positive"], dtype=np.int64)[idx]
    valid_pos_w2 = np.asarray(w2["best"]["valid_positive"], dtype=np.int64)[idx]

    high = support >= SCORABLE_MIN_SUPPORT
    mid = (support >= 5) & (support < SCORABLE_MIN_SUPPORT)
    low = support < 5
    common = high & (valid_pos_dyn >= SCORABLE_MIN_SUPPORT) & (valid_pos_w2 >= SCORABLE_MIN_SUPPORT)

    d_valid, idx_v = delta_vector(dyn["best"]["per_task_ap"], w2["best"]["per_task_ap"])
    valid_support = np.asarray(dyn["best"]["valid_positive"], dtype=np.int64)[idx_v]
    valid_high = valid_support >= SCORABLE_MIN_SUPPORT

    test_high_support = subset_stats(d_w2, high, n_scored_total=n_scored)
    test_mid_support = subset_stats(d_w2, mid, n_scored_total=n_scored)
    test_low_support = subset_stats(d_w2, low, n_scored_total=n_scored)
    all_stats = subset_stats(d_w2, np.ones_like(high), n_scored_total=n_scored)
    common_stats = subset_stats(d_w2, common, n_scored_total=n_scored)
    rc_high = subset_stats(d_rc, high, n_scored_total=n_scored)
    rc_all = subset_stats(d_rc, np.ones_like(high), n_scored_total=n_scored)

    rows = [
        {
            "task": int(t),
            "test_positive": int(support[i]),
            "test_observed": int(models["W2_best_valid"]["test_observed"][t]),
            "valid_positive_dynamic": int(valid_pos_dyn[i]),
            "valid_positive_w2": int(valid_pos_w2[i]),
            "delta_vs_w2": float(d_w2[i]),
            "delta_vs_rc": float(d_rc[i]),
            "test_ap_dynamic_best": models[dyn_best_label]["test_per_task_ap"][t],
            "test_ap_w2_best": models["W2_best_valid"]["test_per_task_ap"][t],
            "test_ap_rc_best": models["RC_REAL_best_valid"]["test_per_task_ap"][t],
            "stratum": "ge20" if support[i] >= 20 else "5_19" if support[i] >= 5 else "lt5",
        }
        for i, t in enumerate(idx.tolist())
    ]

    # --- dynamic usage ----------------------------------------------------------
    usage = mech["results"][dyn_best_label]["full_validation_dynamic_usage"]
    diag_epoch40 = mech["results"][dyn40_label]["full_validation_dynamic_usage"]
    curve_usage = [
        {
            "epoch": int(r["epoch"]),
            "train_bce": float(r["train_bce"]),
            "valid_ap": None if r["valid_ap"] == "" else float(r["valid_ap"]),
            "valid_bce": None if r["valid_bce"] == "" else float(r["valid_bce"]),
            "R1": None if r["R1"] == "" else float(r["R1"]),
            "R2": None if r["R2"] == "" else float(r["R2"]),
            "Rtotal": None if r["Rtotal"] == "" else float(r["Rtotal"]),
            "fc2_norm_blocks0": None if r["fc2_norm_blocks0"] == "" else float(r["fc2_norm_blocks0"]),
            "grad_norm_bond_proj": None if r["grad_norm_bond_proj"] == "" else float(r["grad_norm_bond_proj"]),
            "grad_norm_blocks0_fc1": None if r["grad_norm_blocks0_fc1"] == "" else float(r["grad_norm_blocks0_fc1"]),
        }
        for r in dyn["curve"]
    ]
    w2_curve_valid = [
        {"epoch": int(r["epoch"]), "valid_ap": float(r["valid_ap"])} for r in w2["curve"]
    ]
    usage_summary = {
        "valid_best": usage,
        "valid_epoch40": diag_epoch40,
        "curve_R": curve_usage,
        "epoch0": dyn["run"]["epoch0_diagnostic"]["dynamic_usage"],
        "suppressed": bool((usage["Rtotal"] or 0.0) < float(C.DECISION_RULES["suppressed"]["r_total_max"])),
        "dynamic_branch_used": bool(
            (usage["Rtotal"] or 0.0) > float(C.DECISION_RULES["suppressed"]["r_total_max"])
            and preflight.get("gradient_and_profile", {})
            .get("checks", {})
            .get("all_new_tensors_receive_gradient", False)
        ),
        "gradient_audit": preflight.get("gradient_and_profile", {}).get("per_step_summary", {}),
        "gradient_gate_passed": bool(
            preflight.get("gradient_and_profile", {})
            .get("checks", {})
            .get("all_new_tensors_receive_gradient", False)
        ),
    }

    comparison = {
        "schema": "lcrc-pcba-dynamic-comparison-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_role": C.TEST_ROLE,
        "headline": headline,
        "n_tasks_scored_test": n_scored,
        "test_support_counts": {
            "ge20": int(high.sum()),
            "5_19": int(mid.sum()),
            "lt5": int(low.sum()),
        },
        "validation_support_counts": {
            "ge20": int(valid_high.sum()),
            "lt20": int((~valid_high).sum()),
            "n_tasks_scored_validation": int(idx_v.size),
        },
        "test_all_scored": all_stats,
        "test_high_support": test_high_support,
        "test_mid_support": test_mid_support,
        "test_low_support": test_low_support,
        "test_common_validation_and_test_ge20": {
            **common_stats,
            "n_tasks_expected": int(common.sum()),
            "definition": "official Test positives >= 20 and both Validation positive counts >= 20",
        },
        "test_all_scored_vs_rc": rc_all,
        "test_high_support_vs_rc": rc_high,
        "validation_high_support": subset_stats(d_valid, valid_high, n_scored_total=int(idx_v.size)),
        "validation_all_scored": subset_stats(d_valid, np.ones_like(valid_high), n_scored_total=int(idx_v.size)),
        "extremes_vs_w2": extremes(rows, key="delta_vs_w2", k=5),
        "extremes_vs_rc": extremes(rows, key="delta_vs_rc", k=5),
        "per_task": rows,
        "low_support_dominance": {
            "lt5_sum_delta_share_of_total": float(
                test_low_support.get("sum_delta", 0.0) / all_stats["sum_delta"]
            )
            if all_stats.get("sum_delta")
            else None,
            "lt5_abs_share_of_total_abs": float(
                np.abs(d_w2[low]).sum() / np.abs(d_w2).sum()
            )
            if low.any()
            else None,
            "lt5_n_tasks": int(low.sum()),
            "ge20_macro_contribution": test_high_support.get("macro_contribution"),
            "5_19_macro_contribution": test_mid_support.get("macro_contribution"),
            "lt5_macro_contribution": test_low_support.get("macro_contribution"),
        },
        "dynamic_usage": usage_summary,
        "mechanism": {
            "label": "post-training Validation-only input intervention (not a trained control)",
            "valid_ap": {
                label: {
                    mode: mech["results"][label]["variants"][mode]["valid_ap"]
                    for mode in ("real", "message_off", "message_shuffle")
                }
                for label in mech["results"]
            },
            "logits_change": {
                label: {
                    mode: mech["results"][label]["variants"][mode]["logits_change_vs_real"]
                    for mode in ("message_off", "message_shuffle")
                }
                for label in mech["results"]
            },
            "boundary": mech["interpretation_boundary"],
        },
        "history": {
            "W2_brief_test_ap_best": C.W2_SEED0["test_ap_best"],
            "W2_boc_replay_test_ap_best": models["W2_best_valid"]["test_ap_official"],
            "RC_brief_test_ap_best": C.RC_REAL["test_ap_best"],
            "reference_brief_vs_recomputed_delta": {
                "W2": models["W2_best_valid"]["test_ap_official"] - C.W2_SEED0["test_ap_best"],
                "RC-REAL": models["RC_REAL_best_valid"]["test_ap_official"] - C.RC_REAL["test_ap_best"],
            },
        },
        "preflight_verdict": preflight.get("verdict"),
        "curve": {
            "W2-DYNAMIC": curve_usage,
            "W2": w2_curve_valid,
        },
    }
    comparison["robustness"] = robustness_block(comparison)
    comparison["robustness"]["validation_vs_test_per_task"] = _valid_test_agreement(
        models[dyn_best_label]["valid_per_task_ap"],
        models["W2_best_valid"]["valid_per_task_ap"],
        models[dyn_best_label]["test_per_task_ap"],
        models["W2_best_valid"]["test_per_task_ap"],
        valid_pos_dyn_arr=np.asarray(dyn["best"]["valid_positive"], dtype=np.int64),
    )
    comparison["decision"] = decide(comparison)
    return comparison


def _valid_test_agreement(
    valid_dyn: Sequence[Any],
    valid_w2: Sequence[Any],
    test_dyn: Sequence[Any],
    test_w2: Sequence[Any],
    *,
    valid_pos_dyn_arr: np.ndarray,
) -> dict[str, Any]:
    """Do the Validation and Test per-task deltas point the same way?"""
    vd: list[float] = []
    td: list[float] = []
    vp: list[int] = []
    for t in range(C.N_TASKS):
        a, b = valid_dyn[t], valid_w2[t]
        c, d = test_dyn[t], test_w2[t]
        if None in (a, b, c, d):
            continue
        vd.append(float(a) - float(b))
        td.append(float(c) - float(d))
        vp.append(int(valid_pos_dyn_arr[t]))
    vd_a = np.asarray(vd, dtype=np.float64)
    td_a = np.asarray(td, dtype=np.float64)
    vp_a = np.asarray(vp, dtype=np.int64)
    hi = vp_a >= SCORABLE_MIN_SUPPORT
    return {
        "n_tasks": int(vd_a.size),
        "validation_improved": int((vd_a > 0).sum()),
        "test_improved": int((td_a > 0).sum()),
        "sign_agreement": float(((vd_a > 0) == (td_a > 0)).mean()),
        "pearson": float(np.corrcoef(vd_a, td_a)[0, 1]),
        "spearman": float(
            np.corrcoef(
                np.argsort(np.argsort(vd_a)).astype(np.float64),
                np.argsort(np.argsort(td_a)).astype(np.float64),
            )[0, 1]
        ),
        "validation_macro_delta": float(vd_a.mean()),
        "test_macro_delta": float(td_a.mean()),
        "validation_high_support_mean_delta": float(vd_a[hi].mean()),
        "validation_high_support_improved": int((vd_a[hi] > 0).sum()),
        "validation_high_support_tasks": int(hi.sum()),
        "note": "descriptive agreement between the two splits, not a leakage or significance test",
    }


def robustness_block(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Descriptive robustness of the macro-AP difference (no new inference).

    Never a replacement for the official macro AP: it only says how much of the
    measured difference is concentrated in a few tasks.
    """
    rows = list(comparison["per_task"])
    d = np.asarray([r["delta_vs_w2"] for r in rows], dtype=np.float64)
    pos = np.asarray([r["test_positive"] for r in rows], dtype=np.int64)
    vd = np.asarray([r["valid_positive_dynamic"] for r in rows], dtype=np.int64)
    vw = np.asarray([r["valid_positive_w2"] for r in rows], dtype=np.int64)
    hi = pos >= SCORABLE_MIN_SUPPORT
    common = hi & (vd >= SCORABLE_MIN_SUPPORT) & (vw >= SCORABLE_MIN_SUPPORT)
    task_ids = np.asarray([int(r["task"]) for r in rows], dtype=np.int64)
    drop = {}
    for k in (1, 3, 5, 10):
        idx = np.argsort(-np.abs(d))[:k]
        keep = np.setdiff1d(np.arange(d.size), idx)
        drop[f"drop_top_abs_{k}"] = {
            "macro_delta_remaining": float(d[keep].mean()),
            "removed_tasks": [int(task_ids[i]) for i in idx.tolist()],
            "removed_mean_delta": float(d[idx].mean()),
        }
    return {
        "n_scored": int(d.size),
        "macro_delta": float(d.mean()),
        "macro_delta_high_support": float(d[hi].mean()),
        "macro_delta_common_106": float(d[common].mean()),
        "high_support_improve_gt_0.01": int(((d > 0.01) & hi).sum()),
        "high_support_improve_gt_0.02": int(((d > 0.02) & hi).sum()),
        "high_support_degrade_gt_0.01": int(((d < -0.01) & hi).sum()),
        "high_support_degrade_gt_0.02": int(((d < -0.02) & hi).sum()),
        "macro_sensitivity_drop_extremes": drop,
        "scored_task_ids": {
            "n": int(task_ids.size),
            "unscorable": sorted(set(range(C.N_TASKS)) - set(task_ids.tolist())),
            "note": (
                "a task is scored only if at least one labelled positive exists in "
                "the split; the excluded id is never part of any macro average"
            ),
        },
        "single_low_support_task": {
            "task": int(task_ids[pos < 5][0]) if (pos < 5).any() else None,
            "test_positive": int(pos[pos < 5][0]) if (pos < 5).any() else None,
            "delta": float(d[pos < 5][0]) if (pos < 5).any() else None,
        },
        "note": (
            "descriptive only. The official macro AP is the reported number; these "
            "decompositions only show whether the difference is broad or a few-task artefact."
        ),
    }


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def write_per_task_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "task",
        "stratum",
        "test_positive",
        "test_observed",
        "valid_positive_dynamic",
        "valid_positive_w2",
        "delta_vs_w2",
        "delta_vs_rc",
        "test_ap_dynamic_best",
        "test_ap_w2_best",
        "test_ap_rc_best",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: ("" if row.get(k) is None else row[k])
                    for k in fields
                }
            )
    return C.sha256_file(path)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def report_tables(comparison: Mapping[str, Any]) -> str:
    h = comparison["headline"]
    t = h["test_ap"]
    v = h["valid_ap"]
    lines: list[str] = []
    lines.append("### Headline comparison (official Test = EXPOSED_TEST_ENGINEERING_COMPARISON_SET)\n")
    lines.append("| Model | Params | Best Valid AP | Test AP best-Valid | Valid@40 | Test@40 | GPU-hour |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, params, bv, tv, v40, t40, gh in (
        ("W2 historical", h["parameters"]["W2"], v["W2"], t["W2"], v["W2@40"], t["W2@40"], h["gpu_hours"]["W2"]),
        (
            "RC-REAL historical",
            h["parameters"]["RC-REAL"],
            v["RC-REAL"],
            t["RC-REAL"],
            v["RC-REAL@40"],
            t["RC-REAL@40"],
            h["gpu_hours"]["RC-REAL"],
        ),
        (
            "W2-DYNAMIC",
            h["parameters"]["W2-DYNAMIC"],
            v["W2-DYNAMIC"],
            t["W2-DYNAMIC"],
            v["W2-DYNAMIC@40"],
            t["W2-DYNAMIC@40"],
            h["gpu_hours"]["W2-DYNAMIC"],
        ),
    ):
        lines.append(
            f"| {name} | {params} | {_fmt(bv)} | {_fmt(tv)} | {_fmt(v40)} | {_fmt(t40)} | {_fmt(gh, 4)} |"
        )
    lines.append("")
    lines.append("### Deltas\n")
    lines.append("| Quantity | Value |")
    lines.append("|---|---:|")
    for label, key in (
        ("Dynamic − W2, Test AP (best-Valid, 127 tasks)", "delta_test_best"),
        ("Dynamic − RC-REAL, Test AP (best-Valid, 127 tasks)", "delta_test_best_vs_rc"),
        ("Dynamic − W2, Validation AP (best-Valid)", "delta_valid_best"),
        ("Dynamic − RC-REAL, Validation AP (best-Valid)", "delta_valid_best_vs_rc"),
        ("Dynamic − W2, Test AP @40", "delta_test_40"),
        ("Dynamic − RC-REAL, Test AP @40", "delta_test_40_vs_rc40"),
        ("Dynamic − W2, Validation AP @40", "delta_valid_40"),
    ):
        lines.append(f"| {label} | {_fmt(h[key])} |")
    lines.append("")
    lines.append("### Support strata (Test positives, best-Valid checkpoints)\n")
    lines.append("| Stratum | n tasks | mean Δ | median Δ | improved | degraded | macro contribution |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, key in (
        (f"Test positives ≥ {SCORABLE_MIN_SUPPORT}", "test_high_support"),
        ("Test positives 5–19", "test_mid_support"),
        ("Test positives < 5", "test_low_support"),
        ("all scored tasks", "test_all_scored"),
        ("Validation & Test ≥ 20 (common)", "test_common_validation_and_test_ge20"),
        ("Validation ≥ 20 (Validation AP)", "validation_high_support"),
    ):
        s = comparison[key]
        lines.append(
            f"| {name} | {s['n_tasks']} | {_fmt(s['mean_delta'])} | {_fmt(s['median_delta'])} | "
            f"{s.get('improved', '—')} | {s.get('degraded', '—')} | {_fmt(s.get('macro_contribution'))} |"
        )
    lines.append("")
    lines.append(
        f"Test positives ≥20 tasks: **{comparison['test_support_counts']['ge20']}**; "
        f"5–19: {comparison['test_support_counts']['5_19']}; <5: {comparison['test_support_counts']['lt5']}; "
        f"scored: {comparison['n_tasks_scored_test']}."
    )
    u = comparison["dynamic_usage"]["valid_best"]
    lines.append("")
    lines.append("### Dynamic usage on the full official Validation split\n")
    lines.append("| Quantity | Value |")
    lines.append("|---|---:|")
    for key in (
        "rms_h0",
        "rms_h1_minus_h0",
        "rms_h2_minus_h1",
        "rms_h2_minus_h0",
        "R1",
        "R2",
        "Rtotal",
        "finite_fraction",
    ):
        lines.append(f"| {key} | {_fmt(u[key])} |")
    lines.append("")
    lines.append("### Post-training Validation-only interventions (best-Valid checkpoint)\n")
    lines.append("| Mode | Validation AP | Δ vs real | rel. RMS logits change |")
    lines.append("|---|---:|---:|---:|")
    for mode in ("real", "message_off", "message_shuffle"):
        ap = comparison["mechanism"]["valid_ap"]["W2_DYNAMIC_best_valid"][mode]
        delta = None if mode == "real" else (
            ap - comparison["mechanism"]["valid_ap"]["W2_DYNAMIC_best_valid"]["real"]
        )
        change = (
            None
            if mode == "real"
            else comparison["mechanism"]["logits_change"]["W2_DYNAMIC_best_valid"][mode]["relative_rms_change"]
        )
        lines.append(f"| {mode} | {_fmt(ap)} | {_fmt(delta)} | {_fmt(change)} |")
    lines.append("")
    d = comparison["decision"]
    lines.append("### Decision\n")
    lines.append(f"**`{d['label']}`** — {d['reason']}\n")
    lines.append("| Condition | Value | Met |")
    lines.append("|---|---:|:---:|")
    for key, value in d["measured"].items():
        lines.append(f"| {key} | {_fmt(value) if not isinstance(value, bool) else value} | — |")
    for key, value in d["conditions"].items():
        lines.append(f"| {key} | {value} | {'yes' if value else 'no'} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default=str(C.ANALYSIS))
    args = parser.parse_args(argv)
    analysis_dir = Path(args.analysis_dir)
    comparison = build(analysis_dir)
    C.write_json(analysis_dir / "comparison.json", comparison)
    sha = write_per_task_csv(analysis_dir / "per_task_deltas.csv", comparison["per_task"])
    comparison["per_task_csv_sha256"] = sha
    C.write_json(analysis_dir / "comparison.json", comparison)
    (analysis_dir / "report_tables.md").write_text(report_tables(comparison), encoding="utf-8")
    C.log(
        f"[analyze] decision {comparison['decision']['label']} "
        f"deltaTest {comparison['headline']['delta_test_best']:+.6f} "
        f"deltaValid {comparison['headline']['delta_valid_best']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
