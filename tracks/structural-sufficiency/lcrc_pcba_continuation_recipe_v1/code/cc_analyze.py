"""Fixed-epoch analysis, Stage-2 eligibility, Validation selection and ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import cc_common as C  # noqa: E402
import cc_data as D  # noqa: E402


def _run_dir(stage: str, model: str, recipe: str) -> Path:
    return C.RUNS / f"{stage}_{model}_{recipe}"


def _load_run(stage: str, model: str, recipe: str) -> dict[str, Any] | None:
    path = _run_dir(stage, model, recipe) / "run.json"
    if not path.is_file():
        return None
    return C.read_json(path)


def _checkpoint_path(run: Mapping[str, Any]) -> Path | None:
    p = run.get("checkpoint")
    if not p:
        return None
    path = Path(str(p))
    return path if path.is_absolute() else C.REPO / path


def _checkpoint_sha(run: Mapping[str, Any]) -> str | None:
    p = _checkpoint_path(run)
    if p is None or not p.is_file():
        return None
    return C.sha256_file(p)


def _ap(run: Mapping[str, Any], epoch: int) -> float:
    return float(run["fixed_epoch_reports"][str(epoch)]["ap"])


def _report(run: Mapping[str, Any], epoch: int) -> Mapping[str, Any]:
    return run["fixed_epoch_reports"][str(epoch)]


def _train_ap(run: Mapping[str, Any]) -> float | None:
    te = run.get("train_eval_final")
    return None if not te else float(te["ap"])


def _train_bce(run: Mapping[str, Any]) -> float | None:
    te = run.get("train_eval_final")
    return None if not te else float(te["bce"])


def _stratum(positive: int) -> str:
    if int(positive) < 5:
        return "valid_positive_lt5"
    if int(positive) < 20:
        return "valid_positive_5_19"
    return "valid_positive_ge20"


def _task_wins(a: Sequence[Any], b: Sequence[Any]) -> dict[str, int]:
    wins = losses = ties = 0
    for x, y in zip(a, b):
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        if float(x) > float(y) + 1e-12:
            wins += 1
        elif float(x) < float(y) - 1e-12:
            losses += 1
        else:
            ties += 1
    return {"better": wins, "worse": losses, "tied": ties, "scored": wins + losses + ties}


def _strata_summary(
    ap_a: Sequence[Any],
    ap_b: Sequence[Any],
    positive: Sequence[int],
    *,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stratum in ("valid_positive_lt5", "valid_positive_5_19", "valid_positive_ge20"):
        idx = [t for t in range(len(positive)) if _stratum(int(positive[t])) == stratum]
        a = [ap_a[t] for t in idx]
        b = [ap_b[t] for t in idx]
        finite = [
            (t, float(x), float(y))
            for t, x, y in zip(idx, a, b)
            if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)
        ]
        deltas = [y - x for _, x, y in finite]
        wins = sum(1 for d in deltas if d > 1e-12)
        losses = sum(1 for d in deltas if d < -1e-12)
        out[stratum] = {
            "n_tasks": int(len(idx)),
            "n_scored": int(len(finite)),
            f"mean_ap_{label_a}": float(np.mean([x for _, x, _ in finite])) if finite else None,
            f"mean_ap_{label_b}": float(np.mean([y for _, _, y in finite])) if finite else None,
            f"mean_delta_{label_b}_minus_{label_a}": float(np.mean(deltas)) if deltas else None,
            f"n_tasks_{label_b}_better": int(wins),
            f"n_tasks_{label_b}_worse": int(losses),
        }
    return out


def stage1_comparison() -> dict[str, Any]:
    preflight = C.read_json(C.ANALYSIS / "preflight_gpu.json")
    sources = C.read_json(C.ANALYSIS / "source_checkpoint_audit.json")
    out: dict[str, Any] = {
        "schema": "lcrc-pcba-continuation-epoch60-comparison-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "protocol": C.DECISION["stage1"],
        "models": {},
        "cross_model": {},
        "notes": [
            "D_L = AP_L60 - AP40, D_R = AP_R60 - AP40, D_RL = AP_R60 - AP_L60.",
            "Task-win counts use the 127 Validation tasks scored in both recipes.",
            "Pairwise L/R comparisons are within a model; cross-architecture levels are not causal.",
        ],
    }
    for model_id in C.MODEL_ORDER:
        pf = preflight["models"][model_id]
        ap40 = float(pf["epoch40_replay"]["official_valid_ap"])
        ap40_per_task = pf["epoch40_replay"].get("per_task_ap")
        valid_positive = pf["epoch40_replay"]["positive"]
        source_run = C.read_json(C.SOURCE[model_id]["run_json"])
        source_train = source_run.get("train_eval_final") or {}
        source_train_ap = (
            float(source_train["ap"]) if source_train.get("ap") is not None else None
        )
        source_train_bce = (
            float(source_train["bce"]) if source_train.get("bce") is not None else None
        )
        source_valid40 = (source_run.get("fixed_epoch_reports") or {}).get("40", {})
        model: dict[str, Any] = {
            "source_epoch40": {
                "checkpoint": C.SOURCE[model_id]["path"].relative_to(C.REPO).as_posix(),
                "sha256": C.SOURCE[model_id]["sha256"],
                "valid_ap": ap40,
                "valid_bce": source_valid40.get("valid_bce"),
                "historical_valid_ap": float(C.SOURCE[model_id]["valid_ap"]),
                "replay_abs_delta": abs(ap40 - float(C.SOURCE[model_id]["valid_ap"])),
                "per_task_ap": ap40_per_task,
                "positive": valid_positive,
                "train_ap": source_train_ap,
                "train_bce": source_train_bce,
                "train_valid_ap_gap": (
                    None if source_train_ap is None else float(source_train_ap - ap40)
                ),
            },
            "recipes": {},
        }
        for recipe in C.RECIPE_ORDER:
            run = _load_run("stage1", model_id, recipe)
            if run is None:
                model["recipes"][recipe] = {"status": "MISSING"}
                continue
            if run.get("status") != "COMPLETED":
                model["recipes"][recipe] = {
                    "status": run.get("status"),
                    "stop_reason": run.get("stop_reason"),
                    "run": str(_run_dir("stage1", model_id, recipe).relative_to(C.REPO)),
                }
                continue
            rep60 = _report(run, C.EPOCH60)
            curve60 = [
                r
                for r in _curve_rows(_run_dir("stage1", model_id, recipe) / "train_curve.csv")
                if int(r["epoch"]) == C.EPOCH60
            ][0]
            model["recipes"][recipe] = {
                "status": "COMPLETED",
                "run_dir": str(_run_dir("stage1", model_id, recipe).relative_to(C.REPO)),
                "checkpoint": run.get("checkpoint"),
                "checkpoint_sha256": _checkpoint_sha(run),
                "ap40": ap40,
                "ap50": _ap(run, C.EPOCH50),
                "ap60": float(rep60["ap"]),
                "valid_bce50": _report(run, C.EPOCH50)["valid_bce"],
                "valid_bce60": float(rep60["valid_bce"]),
                "train_bce60": _train_bce(run),
                "train_ap60": _train_ap(run),
                "train_ap_delta_40_60": (
                    None
                    if _train_ap(run) is None or source_train_ap is None
                    else float(_train_ap(run) - source_train_ap)
                ),
                "train_valid_ap_gap40": (
                    None if source_train_ap is None else float(source_train_ap - ap40)
                ),
                "train_valid_ap_gap60": (
                    None if _train_ap(run) is None else float(_train_ap(run) - float(rep60["ap"]))
                ),
                "n_tasks_scored": int(rep60["n_tasks_scored"]),
                "per_task_ap60": rep60["per_task_ap"],
                "valid_positive": rep60["positive"],
                "lr_epoch60": {
                    "first": float(curve60["lr_first"]),
                    "last": float(curve60["lr_last"]),
                    "mean": float(curve60["lr_mean"]),
                },
                "gpu_hours": float(run.get("gpu_hours", float("nan"))),
                "gpu_seconds": float(run.get("gpu_seconds", float("nan"))),
                "global_steps_run": int(run.get("global_steps_run", -1)),
                "steps_per_second_mean": float(run.get("steps_per_second_mean", float("nan"))),
                "first_step": run.get("first_step"),
                "grad_clip_events": int(run.get("grad_clip_events", -1)),
                "non_finite_events": int(run.get("non_finite_events", -1)),
                "restore_audit_passed": bool(run.get("restore_audit", {}).get("state_dict_bitwise_equal")),
            }
        l = model["recipes"].get("L", {})
        r = model["recipes"].get("R", {})
        if l.get("status") == "COMPLETED" and r.get("status") == "COMPLETED":
            d_l = float(l["ap60"]) - ap40
            d_r = float(r["ap60"]) - ap40
            d_rl = float(r["ap60"]) - float(l["ap60"])
            wins = _task_wins(r["per_task_ap60"], l["per_task_ap60"])
            wins_l = _task_wins(l["per_task_ap60"], r["per_task_ap60"])
            n_scored = int(wins["scored"])
            if d_l < C.DECISION["stage1"]["min_gain_over_ap40"] and d_r < C.DECISION["stage1"]["min_gain_over_ap40"]:
                label = "NO_CONTINUATION_GAIN"
            elif (
                d_rl >= C.DECISION["stage1"]["task_win_min"]
                and wins["better"] >= C.DECISION["stage1"]["task_win_threshold"]
            ):
                label = "RESTART_PROMISING"
            elif (
                d_rl <= -C.DECISION["stage1"]["task_win_min"]
                and wins_l["better"] >= C.DECISION["stage1"]["task_win_threshold"]
            ):
                label = "LOW_LR_PROMISING"
            else:
                label = "NO_CLEAR_RECIPE_WINNER"
            model["comparison"] = {
                "D_L": d_l,
                "D_R": d_r,
                "D_RL": d_rl,
                "n_tasks_scored_both": n_scored,
                "n_tasks_R_better_than_L": int(wins["better"]),
                "n_tasks_L_better_than_R": int(wins_l["better"]),
                "n_tasks_tied": int(wins["tied"]),
                "recipe_label": label,
                "thresholds": C.DECISION["stage1"],
                "strata_R_vs_L": _strata_summary(
                    l["per_task_ap60"], r["per_task_ap60"], l["valid_positive"],
                    label_a="L", label_b="R",
                ),
                "strata_L_vs_epoch40": _strata_summary(
                    ap40_per_task, l["per_task_ap60"], valid_positive,
                    label_a="epoch40", label_b="L",
                ) if ap40_per_task is not None else None,
                "strata_R_vs_epoch40": _strata_summary(
                    ap40_per_task, r["per_task_ap60"], valid_positive,
                    label_a="epoch40", label_b="R",
                ) if ap40_per_task is not None else None,
                "train_ap_rose": {
                    recipe_key: bool(
                        rec.get("train_ap_delta_40_60") is not None
                        and rec["train_ap_delta_40_60"] > 0
                    )
                    for recipe_key, rec in (("L", l), ("R", r))
                },
            }
            # Stage-2 eligibility per recipe.
            elig: dict[str, Any] = {}
            for recipe_key, rec in (("L", l), ("R", r)):
                cond_ap = float(rec["ap60"]) >= ap40 + C.DECISION["stage1"]["min_gain_over_ap40"]
                cond_late = float(rec["ap60"]) >= float(rec["ap50"]) - C.DECISION["stage1"]["late_stability_slack"]
                cond_checks = (
                    rec.get("status") == "COMPLETED"
                    and bool(pf.get("passed"))
                    and int(rec.get("non_finite_events", 1)) == 0
                    and bool(rec.get("restore_audit_passed"))
                )
                elig[recipe_key] = {
                    "AP60_ge_AP40_plus_0.002": bool(cond_ap),
                    "AP60_ge_AP50_minus_0.001": bool(cond_late),
                    "correctness_checks_passed": bool(cond_checks),
                    "eligible": bool(cond_ap and cond_late and cond_checks),
                }
            eligible_recipes = [k for k in ("L", "R") if elig[k]["eligible"]]
            if eligible_recipes:
                best = max(eligible_recipes, key=lambda k: (float((l if k == "L" else r)["ap60"]), k == "L"))
                ap_best = float((l if best == "L" else r)["ap60"])
                close = [
                    k
                    for k in eligible_recipes
                    if abs(float((l if k == "L" else r)["ap60"]) - ap_best)
                    <= C.DECISION["stage1"]["tie_break_ap"]
                ]
                chosen = "L" if "L" in close else best
            else:
                chosen = None
            model["stage2"] = {
                "eligible": bool(eligible_recipes),
                "qualifying_recipes": eligible_recipes,
                "per_recipe_conditions": elig,
                "selected_recipe": chosen,
                "selection_rule": (
                    "highest AP60 among recipes that pass both the gain and late-stability "
                    "gates; ties within 1e-4 choose L"
                ),
            }
        else:
            model["comparison"] = {"status": "INCOMPLETE"}
            model["stage2"] = {"eligible": False, "reason": "Stage-1 runs incomplete"}
        out["models"][model_id] = model

    # Cross-model levels (descriptive, not a causal LR comparison).
    levels: dict[str, Any] = {}
    for epoch_key, source in (("epoch40", "source_epoch40"), ("epoch60_L", "L"), ("epoch60_R", "R")):
        row: dict[str, Any] = {}
        for model_id in C.MODEL_ORDER:
            rec = out["models"][model_id]
            if epoch_key == "epoch40":
                row[model_id] = rec["source_epoch40"]["valid_ap"]
            else:
                r = rec["recipes"].get(source, {})
                row[model_id] = r.get("ap60") if r.get("status") == "COMPLETED" else None
        levels[epoch_key] = row
        levels[f"gap_V_minus_A_{epoch_key}"] = (
            None if row.get("V") is None or row.get("A") is None else float(row["V"] - row["A"])
        )
        levels[f"gap_V_minus_G_{epoch_key}"] = (
            None if row.get("V") is None or row.get("G") is None else float(row["V"] - row["G"])
        )
    out["cross_model"] = {
        "levels": levels,
        "note": (
            "A/G/V are different architectures.  These gaps describe where the "
            "continuation checkpoints landed; they are not a pure learning-rate effect."
        ),
    }
    return out


def _curve_rows(path: Path) -> list[dict[str, Any]]:
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stage2_continuation(stage1: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if stage1 is None:
        stage1 = C.read_json(C.ANALYSIS / "epoch60_recipe_comparison.json")
    out: dict[str, Any] = {
        "schema": "lcrc-pcba-continuation-epoch80-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "rule": C.DECISION["stage2"],
        "models": {},
        "notes": [
            "Only models that passed the frozen Stage-2 eligibility rule were continued.",
            "A model stopped at epoch 60 is not an epoch-80 result.",
            "AP70->AP80 is the last-ten-epoch AP change at the fixed evaluation points.",
        ],
    }
    for model_id in C.MODEL_ORDER:
        rec = stage1["models"][model_id]
        s2 = rec.get("stage2", {})
        recipe = s2.get("selected_recipe")
        entry: dict[str, Any] = {
            "stage2_executed": False,
            "eligibility": s2,
            "selected_recipe": recipe,
        }
        if not s2.get("eligible") or recipe is None:
            entry["reason"] = "did not pass the frozen Stage-2 eligibility rule"
            out["models"][model_id] = entry
            continue
        run = _load_run("stage2", model_id, recipe)
        if run is None:
            entry["reason"] = "eligible but the Stage-2 run is missing"
            out["models"][model_id] = entry
            continue
        if run.get("status") != "COMPLETED":
            entry.update(
                {
                    "reason": f"Stage-2 run status {run.get('status')}",
                    "stop_reason": run.get("stop_reason"),
                    "run_dir": str(_run_dir("stage2", model_id, recipe).relative_to(C.REPO)),
                }
            )
            out["models"][model_id] = entry
            continue
        ap40 = float(rec["source_epoch40"]["valid_ap"])
        ap50 = float(rec["recipes"][recipe]["ap50"])
        ap60 = float(rec["recipes"][recipe]["ap60"])
        ap70 = _ap(run, C.EPOCH70)
        ap80 = _ap(run, C.EPOCH80)
        d_60_80 = ap80 - ap60
        d_70_80 = ap80 - ap70
        if ap80 >= ap60 + C.DECISION["stage2"]["late_gain_min"]:
            conv = "LATE_GAIN"
        elif ap80 > ap60:
            conv = "SMALL_LATE_GAIN"
        else:
            conv = "PLATEAU_OR_REGRESSION"
        rep60 = rec["recipes"][recipe]
        rep80 = _report(run, C.EPOCH80)
        curve80 = [
            r for r in _curve_rows(_run_dir("stage2", model_id, recipe) / "train_curve.csv")
            if int(r["epoch"]) == C.EPOCH80
        ][0]
        entry = {
            "stage2_executed": True,
            "run_dir": str(_run_dir("stage2", model_id, recipe).relative_to(C.REPO)),
            "recipe": recipe,
            "checkpoint": run.get("checkpoint"),
            "checkpoint_sha256": _checkpoint_sha(run),
            "ap40": ap40,
            "ap50": ap50,
            "ap60": ap60,
            "ap70": ap70,
            "ap80": ap80,
            "delta_40_60": ap60 - ap40,
            "delta_60_80": d_60_80,
            "delta_40_80": ap80 - ap40,
            "delta_70_80": d_70_80,
            "last_ten_epoch_ap_delta": d_70_80,
            "valid_bce60": rep60["valid_bce60"],
            "valid_bce80": float(rep80["valid_bce"]),
            "valid_bce_delta_60_80": float(rep80["valid_bce"]) - float(rep60["valid_bce60"]),
            "train_ap60": rep60["train_ap60"],
            "train_ap80": _train_ap(run),
            "train_valid_ap_gap60": rep60["train_valid_ap_gap60"],
            "train_valid_ap_gap80": (
                None if _train_ap(run) is None else float(_train_ap(run) - ap80)
            ),
            "train_valid_gap_delta_60_80": (
                None
                if _train_ap(run) is None or rep60["train_ap60"] is None
                else float((_train_ap(run) - ap80) - (rep60["train_ap60"] - ap60))
            ),
            "n_tasks_scored80": int(rep80["n_tasks_scored"]),
            "per_task_ap60": rep60["per_task_ap60"],
            "per_task_ap80": rep80["per_task_ap"],
            "valid_positive": rep80["positive"],
            "task_coverage_80_vs_60": _task_wins(rep80["per_task_ap"], rep60["per_task_ap60"]),
            "strata_80_vs_60": _strata_summary(
                rep60["per_task_ap60"], rep80["per_task_ap"], rep80["positive"],
                label_a="60", label_b="80",
            ),
            "lr_epoch80": {
                "first": float(curve80["lr_first"]),
                "last": float(curve80["lr_last"]),
                "mean": float(curve80["lr_mean"]),
            },
            "gpu_hours": float(run.get("gpu_hours", float("nan"))),
            "gpu_seconds": float(run.get("gpu_seconds", float("nan"))),
            "global_steps_run": int(run.get("global_steps_run", -1)),
            "convergence_label": conv,
        }
        out["models"][model_id] = entry
    return out


def selection(stage1: Mapping[str, Any] | None = None, stage2: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if stage1 is None:
        stage1 = C.read_json(C.ANALYSIS / "epoch60_recipe_comparison.json")
    if stage2 is None:
        stage2 = C.read_json(C.ANALYSIS / "epoch80_continuation.json")
    out: dict[str, Any] = {
        "schema": "lcrc-pcba-continuation-validation-selection-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_read": False,
        "rule": {
            "metric": "official OGB Validation macro AP",
            "candidates": ["epoch40", "L_epoch60", "R_epoch60", "selected_recipe_epoch80 if executed"],
            "tie_break": "within 1e-4 choose fewer training epochs; then L over R",
            "test_never_used": True,
        },
        "models": {},
    }
    for model_id in C.MODEL_ORDER:
        rec = stage1["models"][model_id]
        candidates: list[dict[str, Any]] = []
        src = rec["source_epoch40"]
        candidates.append(
            {
                "label": "epoch40",
                "model": model_id,
                "recipe": "none",
                "epoch": C.EPOCH40,
                "valid_ap": float(src["valid_ap"]),
                "path": src["checkpoint"],
                "sha256": src["sha256"],
                "source": "historical epoch-40 checkpoints, replayed",
            }
        )
        for recipe in C.RECIPE_ORDER:
            r = rec["recipes"].get(recipe, {})
            if r.get("status") == "COMPLETED":
                candidates.append(
                    {
                        "label": f"{recipe}_epoch60",
                        "model": model_id,
                        "recipe": recipe,
                        "epoch": C.EPOCH60,
                        "valid_ap": float(r["ap60"]),
                        "path": r["checkpoint"],
                        "sha256": r["checkpoint_sha256"],
                        "source": f"Stage-1 {recipe} run",
                    }
                )
        s2 = stage2["models"].get(model_id, {})
        if s2.get("stage2_executed"):
            candidates.append(
                {
                    "label": f"{s2['recipe']}_epoch80",
                    "model": model_id,
                    "recipe": s2["recipe"],
                    "epoch": C.EPOCH80,
                    "valid_ap": float(s2["ap80"]),
                    "path": s2["checkpoint"],
                    "sha256": s2["checkpoint_sha256"],
                    "source": f"Stage-2 {s2['recipe']} run",
                }
            )
        best_ap = max(float(c["valid_ap"]) for c in candidates)
        close = [
            c for c in candidates if best_ap - float(c["valid_ap"]) <= C.DECISION["selection"]["tie_break_ap"]
        ]
        priority = {"L": 0, "R": 1, "none": 0}
        close.sort(key=lambda c: (int(c["epoch"]), priority.get(c["recipe"], 2)))
        chosen = close[0]
        out["models"][model_id] = {
            "candidates": candidates,
            "selected": chosen,
            "reason": (
                f"max Validation AP {best_ap:.9f}; within 1e-4 candidates="
                f"{[c['label'] for c in close]}; selected {chosen['label']} "
                f"(epoch {chosen['epoch']}, {chosen['valid_ap']:.9f})"
            ),
            "alias_of_epoch40": chosen["epoch"] == C.EPOCH40,
        }
    return out


def _rr_status(experiment: str) -> dict[str, Any] | None:
    """Read the latest pulled rr metadata for an experiment.

    ``rr status`` has no JSON mode, and the pulled ``meta.json`` is the same
    provenance record that rr writes at launch.  Reading it keeps the ledger
    reproducible offline and avoids parsing human-oriented status text.
    """
    pulled = C.REPO / ".rr" / "pulled"
    best: dict[str, Any] | None = None
    for meta_path in pulled.glob("*/meta.json"):
        try:
            meta = C.read_json(meta_path)
        except Exception:  # noqa: BLE001
            continue
        if meta.get("experiment") != experiment:
            continue
        if meta.get("state") != "completed":
            continue
        if best is None or str(meta.get("created_at", "")) > str(best.get("created_at", "")):
            best = meta
    if best is None:
        return None
    runtime = best.get("runtime") or {}
    return {
        "run_id": best.get("run_id"),
        "state": best.get("state"),
        "pool": best.get("pool"),
        "node": runtime.get("node"),
        "job": best.get("slurm_job_id"),
        "gpu_count": runtime.get("gpu_count"),
        "cuda_visible_devices": runtime.get("cuda_visible_devices"),
        "driver_version": runtime.get("driver_version"),
        "torch_version": runtime.get("torch_version"),
        "torch_cuda": runtime.get("torch_cuda"),
        "commit": best.get("git_commit"),
        "git_dirty": best.get("git_dirty"),
        "command": best.get("command"),
        "created_at": best.get("created_at"),
        "started_at": runtime.get("started_at"),
        "completed_at": runtime.get("completed_at"),
        "exit_code": runtime.get("exit_code"),
    }


def ledger() -> tuple[dict[str, Any], dict[str, Any]]:
    experiments: list[tuple[str, str, str, str]] = []
    for model_id in C.MODEL_ORDER:
        for recipe in C.RECIPE_ORDER:
            experiments.append(("stage1", model_id, recipe, f"cc-s1-{model_id}-{recipe}"))
    s1 = C.read_json(C.ANALYSIS / "epoch60_recipe_comparison.json")
    for model_id in C.MODEL_ORDER:
        recipe = s1["models"][model_id].get("stage2", {}).get("selected_recipe")
        if recipe and s1["models"][model_id].get("stage2", {}).get("eligible"):
            experiments.append(("stage2", model_id, recipe, f"cc-s2-{model_id}-{recipe}"))
    registry: dict[str, Any] = {
        "schema": "lcrc-pcba-continuation-run-registry-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "source_checkpoints": {
            m: {
                "path": C.SOURCE[m]["path"].relative_to(C.REPO).as_posix(),
                "sha256": C.SOURCE[m]["sha256"],
                "epoch": C.EPOCH40,
                "valid_ap": C.SOURCE[m]["valid_ap"],
                "source_commit": C.SOURCE[m]["source_commit"],
            }
            for m in C.TEST_MODEL_ORDER
        },
        "runs": [],
    }
    jobs: list[dict[str, Any]] = []
    for stage, model_id, recipe, experiment in experiments:
        run = _load_run(stage, model_id, recipe)
        status = _rr_status(experiment)
        record: dict[str, Any] = {
            "stage": stage,
            "model": model_id,
            "recipe": recipe,
            "experiment": experiment,
            "run_dir": str(_run_dir(stage, model_id, recipe).relative_to(C.REPO)),
            "status": None if run is None else run.get("status"),
            "stop_reason": None if run is None else run.get("stop_reason"),
            "checkpoint": None if run is None else run.get("checkpoint"),
            "checkpoint_sha256": None if run is None else _checkpoint_sha(run),
            "source_checkpoint": None if run is None else run.get("source_checkpoint"),
            "source_checkpoint_sha256": None if run is None else run.get("source_checkpoint_sha256"),
            "batch_plan_sha256": None
            if run is None
            else (run.get("batch_plan") or {}).get("global_plan_sha256"),
            "batch_plan_prefix_sha256": None
            if run is None
            else (run.get("batch_plan") or {}).get("prefix_sha256"),
            "lr_schedule_sha256": None
            if run is None
            else (run.get("lr_schedule") or {}).get("schedule_sha256"),
            "recipe_lr_schedule_sha256": None
            if run is None
            else ((run.get("lr_schedule") or {}).get("schedule_sha256") or {}).get(recipe),
            "rng_state_start_sha256": None
            if run is None
            else (run.get("rng_state_start") or {}).get("combined_sha256"),
            "first_step": None if run is None else run.get("first_step"),
            "restore_audit": None
            if run is None
            else {
                "state_dict_bitwise_equal": run.get("restore_audit", {}).get("state_dict_bitwise_equal"),
                "optimizer_state_bitwise_equal": run.get("restore_audit", {}).get("optimizer_state_bitwise_equal"),
                "optimizer_step_values": run.get("restore_audit", {}).get("optimizer_step_values"),
            },
            "gpu_hours": None if run is None else run.get("gpu_hours"),
            "gpu_seconds": None if run is None else run.get("gpu_seconds"),
            "started_utc": None if run is None else run.get("started_utc"),
            "finished_utc": None if run is None else run.get("finished_utc"),
            "git_commit": None if run is None else run.get("git_commit"),
            "rr": None
            if status is None
            else {
                "run_id": status.get("run_id"),
                "state": status.get("state"),
                "pool": status.get("pool"),
                "node": status.get("node"),
                "job": status.get("job"),
                "gpu_count": status.get("gpu_count"),
                "cuda_visible_devices": status.get("cuda_visible_devices"),
                "driver_version": status.get("driver_version"),
                "torch_version": status.get("torch_version"),
                "torch_cuda": status.get("torch_cuda"),
                "commit": status.get("commit"),
                "git_dirty": status.get("git_dirty"),
                "command": status.get("command"),
                "created_at": status.get("created_at"),
                "started_at": status.get("started_at"),
                "completed_at": status.get("completed_at"),
                "exit_code": status.get("exit_code"),
            },
        }
        registry["runs"].append(record)
        if run is not None:
            jobs.append(
                {
                    "stage": stage,
                    "experiment": experiment,
                    "model": model_id,
                    "recipe": recipe,
                    "gpu_hours_process": float(run.get("gpu_hours", 0.0)),
                    "status": run.get("status"),
                    "checkpoint_sha256": _checkpoint_sha(run),
                }
            )
    superseded_path = C.ANALYSIS / "preflight_gpu_superseded.json"
    if superseded_path.is_file():
        superseded = C.read_json(superseded_path)
        jobs.append(
            {
                "stage": "stage0",
                "experiment": "cc-preflight (superseded)",
                "model": "A+G+V",
                "recipe": "none",
                "gpu_hours_process": float(superseded.get("total_gpu_hours", 0.0)),
                "status": "COMPLETED",
                "note": (
                    "First Stage-0 GPU preflight at commit "
                    f"{str(superseded.get('git_commit', ''))[:8]}; superseded by the "
                    "PinnedStager preflight.  Counted in the GPU budget."
                ),
            }
        )
    preflight_path = C.ANALYSIS / "preflight_gpu.json"
    preflight_hours = 0.0
    if preflight_path.is_file():
        preflight_hours = float(C.read_json(preflight_path).get("total_gpu_hours", 0.0))
        jobs.append(
            {
                "stage": "stage0",
                "experiment": "cc-preflight",
                "model": "A+G+V",
                "recipe": "none",
                "gpu_hours_process": preflight_hours,
                "status": "COMPLETED",
            }
        )
    test_hours = 0.0
    freeze_path = C.ANALYSIS / "test_prediction_freeze.json"
    if freeze_path.is_file():
        test_hours = float(C.read_json(freeze_path).get("total_gpu_hours", 0.0))
        jobs.append(
            {
                "stage": "test",
                "experiment": "cc-test-freeze",
                "model": "7 logical models",
                "recipe": "none",
                "gpu_hours_process": test_hours,
                "status": "COMPLETED",
            }
        )
    # Auxiliary GPU jobs and maintenance actions, kept separate from the
    # paired training registry but visible for the resource audit.
    auxiliary: list[dict[str, Any]] = []
    for stage, experiment, model, run_dir in (
        ("stage0", "cc-preflight", "A+G+V", C.RUNS / "preflight"),
        ("test", "cc-test-freeze", "C/A/G/V + selected", C.RUNS / "test_freeze"),
    ):
        status = _rr_status(experiment)
        run_json = run_dir / "run.json"
        gpu_hours = None
        if experiment == "cc-preflight" and (C.ANALYSIS / "preflight_gpu.json").is_file():
            gpu_hours = float(C.read_json(C.ANALYSIS / "preflight_gpu.json").get("total_gpu_hours", 0.0))
        elif experiment == "cc-test-freeze" and (C.ANALYSIS / "test_prediction_freeze.json").is_file():
            gpu_hours = float(
                C.read_json(C.ANALYSIS / "test_prediction_freeze.json").get("total_gpu_hours", 0.0)
            )
        auxiliary.append(
            {
                "stage": stage,
                "experiment": experiment,
                "model": model,
                "run_dir": str(run_dir.relative_to(C.REPO)),
                "gpu_hours": gpu_hours,
                "rr": None
                if status is None
                else {
                    "run_id": status.get("run_id"),
                    "state": status.get("state"),
                    "pool": status.get("pool"),
                    "node": status.get("node"),
                    "commit": status.get("commit"),
                    "git_dirty": status.get("git_dirty"),
                    "started_at": status.get("started_at"),
                    "completed_at": status.get("completed_at"),
                    "exit_code": status.get("exit_code"),
                },
            }
        )
    auxiliary.append(
        {
            "stage": "test",
            "experiment": "cc-test-score",
            "model": "6 unique / 7 logical",
            "run_dir": str((C.ANALYSIS / "test_results.json").relative_to(C.REPO)),
            "gpu_hours": 0.0,
            "rr": None,
            "note": "CPU-only post-freeze scoring; the freeze job produced the Test logits.",
        }
    )
    registry["auxiliary"] = auxiliary

    total = float(sum(j["gpu_hours_process"] for j in jobs))
    gpu_ledger = {
        "schema": "lcrc-pcba-continuation-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "budget": C.BUDGET,
        "jobs": jobs,
        "total_gpu_hours": total,
        "within_target": bool(total <= C.BUDGET["gpu_target_hours"]),
        "within_hard_limit": bool(total <= C.BUDGET["gpu_hard_hours"]),
        "note": (
            "gpu_hours_process is the full training-process elapsed time (includes data "
            "load, validation and final Train evaluation); it is an upper bound on GPU "
            "kernel time and counts all preflight/test inference."
        ),
    }
    return registry, gpu_ledger


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("stage1")
    sub.add_parser("stage2")
    sub.add_parser("select")
    sub.add_parser("ledger")
    sub.add_parser("all")
    args = parser.parse_args(argv)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    if args.mode in ("stage1", "all"):
        s1 = stage1_comparison()
        C.write_json(C.ANALYSIS / "epoch60_recipe_comparison.json", s1)
        C.log("wrote epoch60_recipe_comparison.json")
    if args.mode in ("stage2", "all"):
        s2 = stage2_continuation()
        C.write_json(C.ANALYSIS / "epoch80_continuation.json", s2)
        C.log("wrote epoch80_continuation.json")
    if args.mode in ("select", "all"):
        sel = selection()
        C.write_json(C.ANALYSIS / "validation_selection.json", sel)
        C.log("wrote validation_selection.json")
    if args.mode in ("ledger", "all"):
        reg, gpu = ledger()
        C.write_json(C.ANALYSIS / "run_registry.json", reg)
        C.write_json(C.ANALYSIS / "gpu_ledger.json", gpu)
        C.log(f"wrote run_registry.json / gpu_ledger.json total_gpu_hours={gpu['total_gpu_hours']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
