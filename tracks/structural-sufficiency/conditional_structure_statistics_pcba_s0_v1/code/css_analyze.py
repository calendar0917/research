"""Pre-registered analysis of the E/L (and, if purchased, S) comparison.

Everything here is fixed in ``PROTOCOL.md`` section 9 before the formal runs:

* the main table (parameters, step/epoch/eval cost, GPU-hours, best epoch,
  best-Validation and epoch-40 three-split metrics, last-5-epoch means),
* the full per-epoch Train/Validation/Test curves (read from ``curve.csv``),
* E-L per-task AP differences, improved/degraded task counts, medians, largest
  positive/negative task contributions and the macro difference with the largest
  absolute task removed,
* the descriptive high-support slice (Validation **and** Test positives >= 20),
* a 1,000-draw common-task bootstrap of the macro-AP difference (a fixed-model
  sensitivity statement, never a significance test and never a replacement for a
  second seed),
* two read-only mechanism checks on the frozen best-Validation state over the
  first 4,096 official Validation rows: a row permutation of ``qt`` and a zeroed
  ``Wc * c_G`` context term,
* the pre-registered S purchase rule.

No training happens here and no checkpoint/product is modified.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import css_common as C
import css_data as D
import css_model as M


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def arm_dir(arm: str) -> Path:
    return C.RUNS / C.ARMS[arm]["name"]


def load_curve(arm: str) -> list[dict[str, Any]]:
    path = arm_dir(arm) / "train_curve.csv"
    if not path.is_file():
        raise RuntimeError(f"missing curve for arm {arm}: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def load_run(arm: str) -> dict[str, Any]:
    path = arm_dir(arm) / "run.json"
    if not path.is_file():
        raise RuntimeError(f"missing run.json for arm {arm}: {path}")
    return C.read_json(path)


def _last5(curve: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(r[key]) for r in curve[-5:] if r.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def main_table(arms: Sequence[str]) -> list[dict[str, Any]]:
    table = []
    for arm in arms:
        run = load_run(arm)
        curve = load_curve(arm)
        best = run["best_epoch"]
        best_row = next(r for r in curve if int(r["epoch"]) == int(best))
        final_row = curve[-1]
        epochs = [int(r["epoch"]) for r in curve]
        table.append(
            {
                "arm": arm,
                "arm_name": C.ARMS[arm]["name"],
                "parameters": int(run["parameters"]),
                "epochs_run": int(run["epochs_run"]),
                "best_epoch": int(best),
                "best_valid_ap": float(run["best_valid_ap"]),
                "best_state": {
                    "train_ap": float(best_row["train_ap"]),
                    "valid_ap": float(best_row["valid_ap"]),
                    "test_ap": float(best_row["test_ap"]),
                    "train_bce": float(best_row["train_eval_bce"]),
                    "valid_bce": float(best_row["valid_bce"]),
                    "test_bce": float(best_row["test_bce"]),
                },
                "epoch40_state": {
                    "train_ap": float(final_row["train_ap"]),
                    "valid_ap": float(final_row["valid_ap"]),
                    "test_ap": float(final_row["test_ap"]),
                    "train_bce": float(final_row["train_eval_bce"]),
                    "valid_bce": float(final_row["valid_bce"]),
                    "test_bce": float(final_row["test_bce"]),
                },
                "last5": {
                    "train_ap": _last5(curve, "train_ap"),
                    "valid_ap": _last5(curve, "valid_ap"),
                    "test_ap": _last5(curve, "test_ap"),
                },
                "last_epoch_valid_minus_best": float(final_row["valid_ap"] - run["best_valid_ap"]),
                "still_rising_last5": bool(
                    np.polyfit([float(e) for e in epochs[-5:]], [float(r["valid_ap"]) for r in curve[-5:]], 1)[0]
                    > 0
                ),
                "timing": {
                    "wall_seconds": float(run["wall_seconds"]),
                    "gpu_hours": float(run["wall_seconds"]) / 3600.0,
                    "epoch_seconds_mean": float(np.mean([float(r["epoch_seconds"]) for r in curve])),
                    "train_seconds_mean": float(np.mean([float(r["train_seconds"]) for r in curve])),
                    "eval_three_split_seconds_mean": float(
                        np.mean(
                            [
                                float(r["eval_train_seconds"])
                                + float(r["eval_valid_seconds"])
                                + float(r["eval_test_seconds"])
                                for r in curve
                            ]
                        )
                    ),
                    "step_ms_median_mean": float(np.mean([float(r["step_ms_mean"]) for r in curve])),
                    "gpu_peak_reserved_mb": float(max(float(r["gpu_peak_reserved_bytes"]) for r in curve)) / 1e6,
                },
                "steps_completed": int(run["steps_completed"]),
                "device": run["device"],
                "all_epochs_completed": bool(run["all_epochs_completed"]),
            }
        )
    return table


# ---------------------------------------------------------------------------
# prediction products
# ---------------------------------------------------------------------------


def load_predictions(arm: str, state: str, split: str) -> dict[str, np.ndarray]:
    path = arm_dir(arm) / f"predictions_{state}_{split}.npz"
    if not path.is_file():
        raise RuntimeError(f"missing predictions: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {k: np.asarray(payload[k]) for k in payload.files}


def per_task_ap_from_predictions(arm: str, state: str, split: str) -> dict[str, Any]:
    payload = load_predictions(arm, state, split)
    y = payload["labels"]
    logits = payload["logits"]
    probs = M.sigmoid(logits)
    report = C.MET.per_task_ap(y, probs)
    official = C.MET.official_ap(y, probs)
    return {
        "per_task_ap": report["per_task"],
        "positive": report["positive"],
        "negative": report["negative"],
        "observed": report["observed"],
        "official_ap": official["value"],
        "sklearn_macro_ap": report["mean_over_scored_tasks"],
        "rows": int(payload["row_ids"].size),
        "rows_sha256": C.sha256_int64(payload["row_ids"]),
    }


def paired_task_difference(
    arm_a: str,
    arm_b: str,
    *,
    state: str,
    split: str,
    support_min: int = C.SUPPORT_MIN_POSITIVE,
) -> dict[str, Any]:
    a = per_task_ap_from_predictions(arm_a, state, split)
    b = per_task_ap_from_predictions(arm_b, state, split)
    rows = []
    diffs = []
    for t in range(C.N_TASKS):
        va, vb = a["per_task_ap"][t], b["per_task_ap"][t]
        if va is None or vb is None:
            continue
        d = float(va) - float(vb)
        diffs.append(d)
        rows.append(
            {
                "task": t,
                "ap_a": float(va),
                "ap_b": float(vb),
                "delta": d,
                "positive_a": int(a["positive"][t]),
                "positive_b": int(b["positive"][t]),
            }
        )
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    order = np.argsort(diffs_arr)
    high_support = [
        r
        for r in rows
        if r["positive_a"] >= int(support_min) and r["positive_b"] >= int(support_min)
    ]
    high_diffs = np.asarray([r["delta"] for r in high_support], dtype=np.float64)
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "state": state,
        "split": split,
        "n_common_tasks": int(diffs_arr.size),
        "macro_delta": float(diffs_arr.mean()),
        "macro_delta_official_difference": float(a["official_ap"] - b["official_ap"]),
        "median_delta": float(np.median(diffs_arr)),
        "n_improved": int((diffs_arr > 0).sum()),
        "n_degraded": int((diffs_arr < 0).sum()),
        "n_tied": int((diffs_arr == 0).sum()),
        "largest_positive": rows[int(order[-1])] if rows else None,
        "largest_negative": rows[int(order[0])] if rows else None,
        "macro_delta_without_largest_abs": float(
            np.delete(diffs_arr, int(np.argmax(np.abs(diffs_arr)))).mean()
        ),
        "high_support": {
            "definition": f"Validation and Test positives >= {support_min}",
            "n_tasks": int(high_diffs.size),
            "macro_delta": float(high_diffs.mean()) if high_diffs.size else None,
            "n_improved": int((high_diffs > 0).sum()) if high_diffs.size else 0,
            "n_degraded": int((high_diffs < 0).sum()) if high_diffs.size else 0,
            "task_ids": [int(r["task"]) for r in high_support],
        },
        "per_task": rows,
        "arm_a_official_ap": float(a["official_ap"]),
        "arm_b_official_ap": float(b["official_ap"]),
        "arm_a_sklearn_macro_ap": float(a["sklearn_macro_ap"]),
        "arm_b_sklearn_macro_ap": float(b["sklearn_macro_ap"]),
    }


def bootstrap_task_delta(
    arm_a: str,
    arm_b: str,
    *,
    state: str,
    split: str,
    draws: int = C.BOOTSTRAP_DRAWS,
    seed: int = C.BOOTSTRAP_SEED,
    task_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Resample tasks, not molecules.

    ``task_ids`` restricts the resampling to a pre-specified task set (this round
    uses the well-supported tasks: >= 20 positives in both Validation and Test),
    so a reader can see how much of a macro difference survives dropping the
    sparse tail where a single mis-ranking moves AP by 0.05.
    """
    a = per_task_ap_from_predictions(arm_a, state, split)
    b = per_task_ap_from_predictions(arm_b, state, split)
    keep = None if task_ids is None else {int(t) for t in task_ids}
    diffs = []
    for t in range(C.N_TASKS):
        va, vb = a["per_task_ap"][t], b["per_task_ap"][t]
        if va is None or vb is None:
            continue
        if keep is not None and t not in keep:
            continue
        diffs.append(float(va) - float(vb))
    diffs = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n = diffs.size
    means = np.empty(int(draws), dtype=np.float64)
    for i in range(int(draws)):
        means[i] = diffs[rng.integers(0, n, size=n)].mean()
    return {
        "draws": int(draws),
        "seed": int(seed),
        "n_tasks": int(n),
        "macro_delta": float(diffs.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p_delta_positive": float((means > 0).mean()),
        "task_subset": "all common tasks" if task_ids is None else "well-supported tasks only",
        "interpretation": (
            "fixed-model sensitivity of the macro-AP difference to the task composition; "
            "not a molecule-level CI, not training variance, and not a substitute for a second seed"
        ),
    }


# ---------------------------------------------------------------------------
# read-only mechanism checks on the frozen best-Validation state
# ---------------------------------------------------------------------------


def _state_hash(model: M.CSSModel) -> str:
    return C.state_fingerprint(model.state_dict())


def mechanism_checks(
    arm: str,
    dev: D.DevTask,
    *,
    device: torch.device | str = "cpu",
    state: str = "best",
    n_rows: int = C.MECHANISM_VALID_ROWS,
) -> dict[str, Any]:
    import torch

    device = torch.device(device)
    ckpt_path = arm_dir(arm) / ("checkpoint_best.pt" if state == "best" else "checkpoint_epoch40.pt")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, audit = M.build_model(arm, dev.task.norm)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    before = _state_hash(model)
    rows = np.asarray(dev.valid_rows[: int(n_rows)], dtype=np.int64)
    collate = lambda r: D.collate_batch(dev, r)  # noqa: E731

    base_diag = M.ForwardDiagnostics()
    base = M.predict_logits(model, collate, rows, device, diagnostics=base_diag)
    base_probs = M.sigmoid(base)

    out: dict[str, Any] = {
        "arm": arm,
        "state": state,
        "checkpoint": C.track_rel(ckpt_path),
        "parameters": model.n_parameters(),
        "n_rows": int(rows.size),
        "rows_sha256": C.sha256_int64(rows),
        "row_slice": "first 4,096 official Validation rows (fixed before the check)",
        "baseline": {
            "query_norm": float(model.q0.detach().norm()),
            "gate_mean": base_diag.summary()["gate_mean"],
            "gate_std": base_diag.summary()["gate_std"],
            "query_task_spread": base_diag.summary()["query_norm_task_spread"],
            "logits_abs_mean": float(np.abs(base).mean()),
        },
        "state_hash_before": before,
    }

    # (1) permute the task rows of qt, keep the original task output head
    if hasattr(model, "qt"):
        perm = torch.as_tensor(np.random.default_rng(C.BOOTSTRAP_SEED).permutation(C.N_TASKS))
        perm_diag = M.ForwardDiagnostics()
        shuffled = M.predict_logits(
            model, collate, rows, device, overrides=M.ForwardOverrides(qt_permutation=perm), diagnostics=perm_diag
        )
        out["qt_row_permutation"] = {
            "applicable": True,
            "max_abs_logit_change": float(np.abs(shuffled - base).max()),
            "mean_abs_logit_change": float(np.abs(shuffled - base).mean()),
            "mean_abs_prob_change": float(np.abs(M.sigmoid(shuffled) - base_probs).mean()),
            "gate_mean": perm_diag.summary()["gate_mean"],
            "gate_std": perm_diag.summary()["gate_std"],
            "gate_mean_change": perm_diag.summary()["gate_mean"] - base_diag.summary()["gate_mean"],
        }
    else:
        out["qt_row_permutation"] = {"applicable": False, "reason": "S has no task query table"}

    # (2) zero the Wc * c_G context term, keep every other input
    ctx_diag = M.ForwardDiagnostics()
    off = M.predict_logits(
        model, collate, rows, device, overrides=M.ForwardOverrides(context_off=True), diagnostics=ctx_diag
    )
    out["context_term_off"] = {
        "applicable": True,
        "max_abs_logit_change": float(np.abs(off - base).max()),
        "mean_abs_logit_change": float(np.abs(off - base).mean()),
        "mean_abs_prob_change": float(np.abs(M.sigmoid(off) - base_probs).mean()),
        "gate_mean": ctx_diag.summary()["gate_mean"],
        "gate_std": ctx_diag.summary()["gate_std"],
        "gate_mean_change": ctx_diag.summary()["gate_mean"] - base_diag.summary()["gate_mean"],
    }
    after = _state_hash(model)
    out["state_hash_after"] = after
    out["state_unchanged"] = bool(before == after)
    out["notes"] = [
        "read-only: no gradient step, no checkpoint write, no prediction product overwritten",
        "a gate / prediction change is a usage measurement, NOT a performance gain and NOT a causal explanation",
        "the checks are reported for whichever state is analysed and are never used to select a model",
    ]
    return out


# ---------------------------------------------------------------------------
# S purchase rule
# ---------------------------------------------------------------------------


def s_gate(
    *,
    e_best_valid_ap: float,
    e_epoch40_valid_ap: float,
    l_epoch40_valid_ap: float,
    used_gpu_hours: float,
    projected_s_gpu_hours: float,
) -> dict[str, Any]:
    gate = C.S_GATE
    d40 = float(e_epoch40_valid_ap) - float(l_epoch40_valid_ap)
    conditions = {
        "c1_d40_or_e_best": bool(
            d40 >= gate["d40_early_min"] or float(e_best_valid_ap) >= gate["e_best_high_min"]
        ),
        "c2_e_best_floor": bool(float(e_best_valid_ap) >= gate["e_best_floor"]),
        "c3_budget": bool(
            (float(used_gpu_hours) + gate["cost_multiplier"] * float(projected_s_gpu_hours))
            <= gate["gpu_hard_hours"]
        ),
    }
    purchased = all(conditions.values())
    reasons = []
    if not conditions["c1_d40_or_e_best"]:
        reasons.append(
            f"d40 = {d40:.6f} < {gate['d40_early_min']} and E best-Val {e_best_valid_ap:.6f} < {gate['e_best_high_min']}"
        )
    if not conditions["c2_e_best_floor"]:
        reasons.append(f"E best-Val {e_best_valid_ap:.6f} < {gate['e_best_floor']}")
    if not conditions["c3_budget"]:
        reasons.append(
            f"used {used_gpu_hours:.2f} + 1.2 x {projected_s_gpu_hours:.2f} GPU-h > hard budget {gate['gpu_hard_hours']}"
        )
    return {
        "purchased": bool(purchased),
        "label": "PURCHASED" if purchased else "NOT_PURCHASED",
        "d40_valid_ap_E_minus_L": d40,
        "e_best_valid_ap": float(e_best_valid_ap),
        "e_epoch40_valid_ap": float(e_epoch40_valid_ap),
        "l_epoch40_valid_ap": float(l_epoch40_valid_ap),
        "used_gpu_hours": float(used_gpu_hours),
        "projected_s_gpu_hours": float(projected_s_gpu_hours),
        "rule": dict(gate),
        "conditions": conditions,
        "reasons": reasons,
        "note": (
            "the rule buys one extra explanatory trajectory (task-query increment); it is not a "
            "significance threshold. Continuous values are always reported next to the decision."
        ),
    }


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def run_analysis(
    *,
    arms: Sequence[str] | None = None,
    device: str = "cpu",
    mechanism_rows: int = C.MECHANISM_VALID_ROWS,
    write: bool = True,
) -> dict[str, Any]:
    available = [a for a in C.ARM_ORDER if (arm_dir(a) / "run.json").is_file()]
    arms = [a for a in (arms or available) if a in available]
    if "E" not in arms or "L" not in arms:
        raise RuntimeError(f"both primary arms must be complete; found {available}")
    dev = D.load_task(with_cpc=True)
    table = main_table(arms)
    comparisons = {}
    for other in [a for a in arms if a != "E"]:
        comparisons[f"E_{other}"] = {
            state: paired_task_difference("E", other, state=state, split="valid")
            for state in ("best", "epoch40")
        }
        comparisons[f"E_{other}_test"] = {
            state: paired_task_difference("E", other, state=state, split="test")
            for state in ("best", "epoch40")
        }
    bootstrap = {
        f"E_{other}": {
            split: bootstrap_task_delta("E", other, state="epoch40", split=split)
            for split in ("valid", "test")
        }
        for other in [a for a in arms if a != "E"]
    }
    # the same resampling restricted to the tasks this round pre-specifies as
    # well supported (>= 20 positives in both Validation and Test)
    bootstrap_high_support = {
        f"E_{other}": {
            split: bootstrap_task_delta(
                "E",
                other,
                state="epoch40",
                split=split,
                task_ids=comparisons[f"E_{other}" if split == "valid" else f"E_{other}_test"]["epoch40"][
                    "high_support"
                ]["task_ids"],
            )
            for split in ("valid", "test")
        }
        for other in [a for a in arms if a != "E"]
    }
    mechanisms = {arm: mechanism_checks(arm, dev, device=device, n_rows=mechanism_rows) for arm in arms}
    by_arm = {row["arm"]: row for row in table}
    used_gpu_hours = float(sum(row["timing"]["gpu_hours"] for row in table))
    projected_s = float(by_arm["E"]["timing"]["gpu_hours"])
    gate = s_gate(
        e_best_valid_ap=by_arm["E"]["best_valid_ap"],
        e_epoch40_valid_ap=by_arm["E"]["epoch40_state"]["valid_ap"],
        l_epoch40_valid_ap=by_arm["L"]["epoch40_state"]["valid_ap"],
        used_gpu_hours=used_gpu_hours,
        projected_s_gpu_hours=projected_s,
    )
    payload = {
        "schema": "css-pcba-s0-v1-analysis-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "seed": int(C.SEED),
        "test_role": C.TEST_ROLE,
        "main_table": table,
        "epoch_comparisons": comparisons,
        "task_bootstrap": bootstrap,
        "task_bootstrap_high_support": bootstrap_high_support,
        "mechanism_checks": mechanisms,
        "s_gate": gate,
        "historical_anchors": dict(C.HISTORICAL),
        "reuse_note": "historical anchors are cited from existing run.json / analysis files, never retrained",
        "label_view": dev.label_view_audit(),
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
    }
    if write:
        C.atomic_write_json(C.ANALYSIS / "analysis.json", payload)
    return payload


__all__ = [
    "arm_dir",
    "bootstrap_task_delta",
    "load_curve",
    "load_predictions",
    "load_run",
    "main_table",
    "mechanism_checks",
    "paired_task_difference",
    "per_task_ap_from_predictions",
    "run_analysis",
    "s_gate",
]
