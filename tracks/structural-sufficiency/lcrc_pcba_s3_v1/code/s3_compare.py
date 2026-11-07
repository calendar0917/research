"""Comparison, decision and ledger for the LCRC-S3 round.

Reads (git-ignored) run directories plus the audits and writes the analysis
artifacts of the round:

    analysis/s3_selected.json              S3 best-Validation-AP record
    analysis/ffn_selected.json             F best-Validation-AP record
    analysis/best_valid_comparison.json    D_best_S3 / D_best_F / D_mech + per-task strata
    analysis/epoch40_comparison.json       the same at the fixed epoch-40 endpoint
    analysis/per_task_deltas.csv           128 rows (127 scorable)
    analysis/decision.json                 the pre-registered label
    analysis/gpu_ledger.json               every job charged to this round
    analysis/training_parity.json          plan / recipe / init consistency of the two runs

The per-task comparisons use *each model's own selected best-Validation
checkpoint* for the primary table and the epoch-40 endpoint for the auxiliary
table.  Nothing here reads an official Test target.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import s3_common as C  # noqa: E402

SCORABLE_TASKS = 127


# ---------------------------------------------------------------------------
# small statistics
# ---------------------------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra = _rank(a)
    rb = _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranked = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and ranked[j + 1] == ranked[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def delta_stats(a: Sequence[float | None], b: Sequence[float | None], support: Sequence[int]) -> dict[str, Any]:
    """Per-task ``a - b`` statistics over the scorable tasks."""
    rows = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x is None or y is None:
            continue
        rows.append((i, float(x) - float(y), int(support[i])))
    d = np.asarray([r[1] for r in rows], dtype=np.float64)
    s = np.asarray([r[2] for r in rows], dtype=np.int64)
    out: dict[str, Any] = {
        "n_scorable": int(d.size),
        "improved": int((d > 0).sum()),
        "regressed": int((d < 0).sum()),
        "unchanged": int((d == 0).sum()),
        "mean_delta": float(d.mean()) if d.size else 0.0,
        "median_delta": float(np.median(d)) if d.size else 0.0,
        "std_delta": float(d.std()) if d.size else 0.0,
        "min_delta": float(d.min()) if d.size else 0.0,
        "max_delta": float(d.max()) if d.size else 0.0,
        "spearman_delta_vs_support": spearman(d, s.astype(np.float64)),
    }
    pos = np.sort(d[d > 0])[::-1]
    total_pos = float(pos.sum())
    out["positive_delta_total"] = total_pos
    for k in (5, 10):
        out[f"top{k}_positive_share"] = float(pos[:k].sum() / total_pos) if total_pos > 0 else 0.0
    for lo, hi, name in (
        (20, 10**9, "support_ge20"),
        (5, 19, "support_5_19"),
        (0, 4, "support_lt5"),
    ):
        mask = (s >= lo) & (s <= hi)
        out[name] = {
            "n": int(mask.sum()),
            "improved": int((d[mask] > 0).sum()),
            "mean_delta": float(d[mask].mean()) if mask.any() else 0.0,
            "median_delta": float(np.median(d[mask])) if mask.any() else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# run loading
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run["_dir"] = C.track_rel(run_dir)
    curve = run_dir / "train_curve.csv"
    if curve.is_file():
        with open(curve, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        run["_curve"] = [
            {
                "epoch": int(r["epoch"]),
                "valid_ap": float(r["valid_ap"]) if r["valid_ap"] else None,
                "valid_bce": float(r["valid_bce"]) if r["valid_bce"] else None,
                "train_bce": float(r["train_bce"]) if r["train_bce"] else None,
                "learning_rate": float(r["learning_rate"]) if r["learning_rate"] else None,
                "epoch_seconds": float(r["epoch_seconds"]) if r["epoch_seconds"] else None,
                "step_ms_p50": float(r["step_ms_p50"]) if r["step_ms_p50"] else None,
                "branch_residual_over_h_raw": float(r["branch_residual_over_h_raw"])
                if r.get("branch_residual_over_h_raw")
                else None,
                "new_param_grad_norm": float(r["new_param_grad_norm"]) if r.get("new_param_grad_norm") else None,
                "gpu_peak_allocated_bytes": int(r["gpu_peak_allocated_bytes"])
                if r.get("gpu_peak_allocated_bytes")
                else 0,
            }
            for r in rows
        ]
    return run


def selected_record(run: Mapping[str, Any], *, key: str) -> dict[str, Any]:
    """The best-Validation-AP checkpoint record of one arm."""
    epoch = int(run["best_epoch"])
    report = run["fixed_epoch_reports"][str(epoch)]
    curve = run.get("_curve", [])
    best_ap = float(run["best_valid_ap"])
    worst_after = None
    after = [r["valid_ap"] for r in curve if r["epoch"] > epoch and r["valid_ap"] is not None]
    if after:
        worst_after = {
            "epoch_range": [int(epoch) + 1, int(max(r["epoch"] for r in curve))],
            "min_ap": float(min(after)),
            "max_ap": float(max(after)),
            "drop_from_best": float(best_ap - min(after)),
        }
    final_epoch = int(run["final_epoch"])
    e40 = run["fixed_epoch_reports"].get(str(final_epoch))
    return {
        "arm": key,
        "model": run["model"],
        "model_name": run["model_name"],
        "role": run["role"],
        "seed": int(run["seed"]),
        "parameters": int(run["parameters"]),
        "base_parameters": int(run["base_parameters"]),
        "new_parameters": int(run["new_parameters"]),
        "status": run["status"],
        "eval_every": int(run["eval_every"]),
        "eval_epochs_completed": int(len(run["eval_epochs_completed"])),
        "selection_rule": run.get("selection_rule"),
        "best_epoch": int(epoch),
        "best_valid_ap": best_ap,
        "best_valid_bce": float(report["valid_bce"]),
        "best_per_task_ap": report["per_task_ap"],
        "best_valid_positive": report["positive"],
        "best_valid_observed": report["observed"],
        "best_checkpoint": run["best_checkpoint"],
        "best_checkpoint_sha256": run["best_checkpoint_sha256"],
        "epoch40_ap": float(e40["ap"]) if e40 else None,
        "epoch40_valid_bce": float(e40["valid_bce"]) if e40 else None,
        "epoch40_per_task_ap": e40["per_task_ap"] if e40 else None,
        "epoch40_checkpoint": run["epoch40_checkpoint"],
        "epoch40_checkpoint_sha256": run["epoch40_checkpoint_sha256"],
        "train_eval_final": run.get("train_eval_final"),
        "gpu_hours": float(run["gpu_hours"]),
        "epoch_seconds_mean": float(run["epoch_seconds_mean"]),
        "steps_per_second_mean": float(run["steps_per_second_mean"]),
        "gpu_peak_allocated_bytes": int(run["gpu_peak_allocated_bytes"]),
        "gradient_clip_events": int(run["grad_clip_events"]),
        "batch_plan": run["batch_plan"],
        "valid_curve": [{"epoch": r["epoch"], "valid_ap": r["valid_ap"], "valid_bce": r["valid_bce"]} for r in run.get("_curve", [])],
        "post_best_behaviour": worst_after,
        "branch_residual_over_h_raw_final": (
            run.get("_curve", [{}])[-1].get("branch_residual_over_h_raw") if run.get("_curve") else None
        ),
        "epoch0_diagnostic": run.get("epoch0_diagnostic"),
        "s3_cache_identity": run.get("s3_cache_identity"),
        "git_commit": run.get("git_commit"),
        "test_evaluated": bool(run.get("test_evaluated", False)),
    }


# ---------------------------------------------------------------------------
# W2 replay record (read from analysis/w2_best_replay.json)
# ---------------------------------------------------------------------------


def w2_reference(replay_path: Path) -> dict[str, Any]:
    replay = json.loads(Path(replay_path).read_text(encoding="utf-8"))
    if not replay.get("verified"):
        raise RuntimeError(f"W2 replay is not verified: {replay.get('verification')}")
    best = replay["best"]
    e40 = replay["epoch40"]
    return {
        "arm": "W2",
        "model": "W2",
        "model_name": "LCRC-A-Wide-2x",
        "role": "historical_reference_read_only",
        "seed": 0,
        "parameters": int(C.BASE_PARAMETERS),
        "base_parameters": int(C.BASE_PARAMETERS),
        "new_parameters": 0,
        "status": "COMPLETED",
        "best_epoch": int(best["epoch"]),
        "best_valid_ap": float(best["ap"]),
        "best_valid_bce": float(best["valid_bce"]),
        "best_per_task_ap": best["per_task_ap"],
        "best_valid_positive": best["positive"],
        "best_valid_observed": best["observed"],
        "best_checkpoint": best["checkpoint"],
        "best_checkpoint_sha256": best["checkpoint_sha256"],
        "epoch40_ap": float(e40["ap"]),
        "epoch40_valid_bce": float(e40["valid_bce"]),
        "epoch40_per_task_ap": e40["per_task_ap"],
        "epoch40_checkpoint": e40["checkpoint"],
        "epoch40_checkpoint_sha256": e40["checkpoint_sha256"],
        "gpu_hours": float(C.W2["gpu_hours"]),
        "epoch_seconds_mean": float(C.W2["epoch_seconds_mean"]),
        "recorded_best_valid_ap": float(C.W2["best_valid_ap"]),
        "recorded_best_epoch": int(C.W2["best_epoch"]),
        "recorded_ap40": float(C.W2["ap40"]),
        "replay_ap_delta_best": float(best["ap"]) - float(C.W2["best_valid_ap"]),
        "replay_ap_delta_epoch40": float(e40["ap"]) - float(C.W2["ap40"]),
        "train_eval_final": replay.get("train_eval"),
    }


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(
    *,
    d_best_s3: float,
    d_best_f: float,
    d_mech: float,
    task_stats_ge20: Mapping[str, Any],
) -> dict[str, Any]:
    rules = C.DECISION_RULES
    promising = rules["promising"]
    mean_ge20 = float(task_stats_ge20["mean_delta"])
    improved_ge20 = int(task_stats_ge20["improved"])
    conditions = {
        "d_best_s3_min": float(promising["d_best_s3_min"]),
        "d_best_s3": float(d_best_s3),
        "d_best_s3_pass": bool(d_best_s3 >= float(promising["d_best_s3_min"])),
        "d_mech_min": float(promising["d_mech_min"]),
        "d_mech": float(d_mech),
        "d_mech_pass": bool(d_mech >= float(promising["d_mech_min"])),
        "mean_delta_support_ge20_min": float(promising["mean_delta_support_ge20_min"]),
        "mean_delta_support_ge20": mean_ge20,
        "mean_delta_support_ge20_pass": bool(mean_ge20 >= float(promising["mean_delta_support_ge20_min"])),
        "improved_support_ge20_min": int(promising["improved_support_ge20_min"]),
        "improved_support_ge20": improved_ge20,
        "improved_support_ge20_pass": bool(improved_ge20 >= int(promising["improved_support_ge20_min"])),
    }
    if all(conditions[k] for k in ("d_best_s3_pass", "d_mech_pass", "mean_delta_support_ge20_pass", "improved_support_ge20_pass")):
        label = promising["label"]
        interpretation = (
            "S3 shows a structural-extension signal worth confirming on the current official "
            "Validation split; this is not evidence of improved generalisation."
        )
    elif d_best_s3 <= float(rules["regression"]["d_best_s3_max"]):
        label = rules["regression"]["label"]
        interpretation = (
            "S3 loses at least 0.002 best-Validation AP against W2; recorded as a regression for "
            "this exact recipe, not as a proof that explicit three-hop structure is useless."
        )
    elif d_best_s3 < float(rules["no_clear_gain"]["d_best_s3_max"]) and not (
        mean_ge20 >= 0.003 and improved_ge20 >= 60
    ):
        label = rules["no_clear_gain"]["label"]
        interpretation = (
            "S3's best-Validation gain over W2 is below +0.002 with no broad high-support benefit; "
            "the S3 route stops here."
        )
    elif d_best_f > 0 and d_mech <= 0.003:
        label = rules["capacity_only"]["label"]
        interpretation = (
            "Both S3 and the near-matched FFN improve on W2, but S3 has no material advantage over "
            "the FFN: this is an ordinary capacity effect, not a structural one."
        )
    else:
        label = rules["weak_or_mixed"]["label"]
        interpretation = (
            "S3 shows a small improvement over W2 that does not reach the full promising gate; the "
            "actual gain and its task distribution are reported without further experiments."
        )
    return {
        "label": label,
        "interpretation": interpretation,
        "promising_conditions": conditions,
        "rule_order": [
            "PROMISING_S3_SIGNAL (all four conditions)",
            "REGRESSION (d_best_s3 <= -0.002)",
            "NO_CLEAR_GAIN (d_best_s3 < +0.002 and no broad high-support benefit)",
            "CAPACITY_ONLY (F also improves and S3 - F <= +0.003)",
            "WEAK_OR_MIXED (remaining case)",
        ],
        "thresholds_are_research_priorities_not_significance_tests": True,
        "test_accessed": False,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_analysis(
    *,
    s3_run_dir: Path,
    f_run_dir: Path,
    replay_path: Path,
    out_dir: Path,
    rr_jobs: Path | None = None,
) -> dict[str, Any]:
    s3_run = load_run(s3_run_dir)
    f_run = load_run(f_run_dir)
    s3 = selected_record(s3_run, key="S3")
    f = selected_record(f_run, key="F")
    w2 = w2_reference(replay_path)

    support = [int(x) for x in w2["best_valid_positive"]]
    strata = {
        "ge20": [i for i, s in enumerate(support) if s >= 20],
        "5_19": [i for i, s in enumerate(support) if 5 <= s < 20],
        "lt5": [i for i, s in enumerate(support) if s < 5],
    }

    d_best_s3 = float(s3["best_valid_ap"]) - float(w2["best_valid_ap"])
    d_best_f = float(f["best_valid_ap"]) - float(w2["best_valid_ap"])
    d_mech = float(s3["best_valid_ap"]) - float(f["best_valid_ap"])

    best_tasks = {
        "S3_minus_W2": delta_stats(s3["best_per_task_ap"], w2["best_per_task_ap"], support),
        "S3_minus_F": delta_stats(s3["best_per_task_ap"], f["best_per_task_ap"], support),
        "F_minus_W2": delta_stats(f["best_per_task_ap"], w2["best_per_task_ap"], support),
    }
    e40_d_s3 = float(s3["epoch40_ap"]) - float(w2["epoch40_ap"])
    e40_d_f = float(f["epoch40_ap"]) - float(w2["epoch40_ap"])
    e40_tasks = {
        "S3_minus_W2": delta_stats(s3["epoch40_per_task_ap"], w2["epoch40_per_task_ap"], support),
        "S3_minus_F": delta_stats(s3["epoch40_per_task_ap"], f["epoch40_per_task_ap"], support),
        "F_minus_W2": delta_stats(f["epoch40_per_task_ap"], w2["epoch40_per_task_ap"], support),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json(out_dir / "s3_selected.json", s3)
    C.write_json(out_dir / "ffn_selected.json", f)

    comparison = {
        "primary_metric": "official Validation macro AP at each model's own best-Validation epoch",
        "selection_rule": "argmax over epochs 1..40, earliest epoch wins ties",
        "models": {
            "W2": {"parameters": w2["parameters"], "best_epoch": w2["best_epoch"], "best_valid_ap": w2["best_valid_ap"], "checkpoint_sha256": w2["best_checkpoint_sha256"], "source": "read-only replay of the historical run"},
            "S3": {"parameters": s3["parameters"], "best_epoch": s3["best_epoch"], "best_valid_ap": s3["best_valid_ap"], "checkpoint_sha256": s3["best_checkpoint_sha256"]},
            "F": {"parameters": f["parameters"], "best_epoch": f["best_epoch"], "best_valid_ap": f["best_valid_ap"], "checkpoint_sha256": f["best_checkpoint_sha256"]},
        },
        "deltas": {
            "D_best_S3": d_best_s3,
            "D_best_F": d_best_f,
            "D_mech": d_mech,
            "D_mech_note": "S3 - F, the mechanism contrast at near-equal parameter count",
        },
        "per_task": best_tasks,
        "task_support_strata": {
            "n_scorable": int(sum(1 for x in w2["best_per_task_ap"] if x is not None)),
            "ge20": len(strata["ge20"]),
            "5_19": len(strata["5_19"]),
            "lt5": len(strata["lt5"]),
            "unscorable": [i for i, x in enumerate(w2["best_per_task_ap"]) if x is None],
        },
        "curves": {"W2": None, "S3": s3["valid_curve"], "F": f["valid_curve"]},
        "post_best": {"S3": s3["post_best_behaviour"], "F": f["post_best_behaviour"]},
        "best_epoch_ap_gap_epoch40": {
            "W2": float(w2["best_valid_ap"]) - float(w2["epoch40_ap"]),
            "S3": float(s3["best_valid_ap"]) - float(s3["epoch40_ap"]),
            "F": float(f["best_valid_ap"]) - float(f["epoch40_ap"]),
        },
        "test_accessed": False,
    }
    C.write_json(out_dir / "best_valid_comparison.json", comparison)

    epoch40 = {
        "primary_metric": "official Validation macro AP at the fixed epoch-40 endpoint (auxiliary)",
        "models": {
            "W2": {"ap40": w2["epoch40_ap"], "valid_bce": w2["epoch40_valid_bce"]},
            "S3": {"ap40": s3["epoch40_ap"], "valid_bce": s3["epoch40_valid_bce"]},
            "F": {"ap40": f["epoch40_ap"], "valid_bce": f["epoch40_valid_bce"]},
        },
        "deltas": {"D_epoch40_S3": e40_d_s3, "D_epoch40_F": e40_d_f, "D_epoch40_S3_minus_F": float(s3["epoch40_ap"]) - float(f["epoch40_ap"])},
        "per_task": e40_tasks,
        "train": {
            "W2": {"ap40": C.W2["train_ap40"], "bce40": C.W2["train_bce40"]},
            "S3": s3["train_eval_final"],
            "F": f["train_eval_final"],
        },
        "note": (
            "the epoch-40 endpoint is reported as a diagnostic; it is NOT used to select a model, "
            "and no epoch-40 result is allowed to replace the pre-registered best-Validation rule"
        ),
        "test_accessed": False,
    }
    C.write_json(out_dir / "epoch40_comparison.json", epoch40)

    # per-task CSV (best-selected and epoch-40 side by side)
    fields = [
        "task",
        "valid_positive",
        "valid_observed",
        "stratum",
        "w2_best_ap",
        "s3_best_ap",
        "f_best_ap",
        "delta_s3_minus_w2",
        "delta_s3_minus_f",
        "delta_f_minus_w2",
        "w2_epoch40_ap",
        "s3_epoch40_ap",
        "f_epoch40_ap",
        "delta_s3_minus_w2_epoch40",
        "delta_s3_minus_f_epoch40",
        "delta_f_minus_w2_epoch40",
    ]
    rows = []
    for t in range(C.N_TASKS):
        def _v(x):
            return "" if x is None else float(x)

        w2b, s3b, fb = w2["best_per_task_ap"][t], s3["best_per_task_ap"][t], f["best_per_task_ap"][t]
        w2e, s3e, fe = w2["epoch40_per_task_ap"][t], s3["epoch40_per_task_ap"][t], f["epoch40_per_task_ap"][t]
        rows.append(
            {
                "task": t,
                "valid_positive": int(w2["best_valid_positive"][t]),
                "valid_observed": int(w2["best_valid_observed"][t]),
                "stratum": "unscorable"
                if w2b is None
                else ("support_ge20" if w2["best_valid_positive"][t] >= 20 else "support_5_19" if w2["best_valid_positive"][t] >= 5 else "support_lt5"),
                "w2_best_ap": _v(w2b),
                "s3_best_ap": _v(s3b),
                "f_best_ap": _v(fb),
                "delta_s3_minus_w2": "" if (w2b is None or s3b is None) else float(s3b) - float(w2b),
                "delta_s3_minus_f": "" if (fb is None or s3b is None) else float(s3b) - float(fb),
                "delta_f_minus_w2": "" if (w2b is None or fb is None) else float(fb) - float(w2b),
                "w2_epoch40_ap": _v(w2e),
                "s3_epoch40_ap": _v(s3e),
                "f_epoch40_ap": _v(fe),
                "delta_s3_minus_w2_epoch40": "" if (w2e is None or s3e is None) else float(s3e) - float(w2e),
                "delta_s3_minus_f_epoch40": "" if (fe is None or s3e is None) else float(s3e) - float(fe),
                "delta_f_minus_w2_epoch40": "" if (w2e is None or fe is None) else float(fe) - float(w2e),
            }
        )
    path = out_dir / "per_task_deltas.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    per_task_sha = C.sha256_file(path)

    decision = decide(
        d_best_s3=d_best_s3,
        d_best_f=d_best_f,
        d_mech=d_mech,
        task_stats_ge20=best_tasks["S3_minus_W2"]["support_ge20"],
    )
    decision.update(
        {
            "round_id": C.ROUND_ID,
            "d_best_s3": d_best_s3,
            "d_best_f": d_best_f,
            "d_mech": d_mech,
            "epoch40": {"D_epoch40_S3": e40_d_s3, "D_epoch40_F": e40_d_f},
            "best_epochs": {"W2": w2["best_epoch"], "S3": s3["best_epoch"], "F": f["best_epoch"]},
            "best_valid_ap": {
                "W2": w2["best_valid_ap"],
                "S3": s3["best_valid_ap"],
                "F": f["best_valid_ap"],
            },
            "epoch40_ap": {"W2": w2["epoch40_ap"], "S3": s3["epoch40_ap"], "F": f["epoch40_ap"]},
            "task_distribution": {
                "improved": best_tasks["S3_minus_W2"]["improved"],
                "regressed": best_tasks["S3_minus_W2"]["regressed"],
                "mean_delta": best_tasks["S3_minus_W2"]["mean_delta"],
                "median_delta": best_tasks["S3_minus_W2"]["median_delta"],
                "top10_positive_share": best_tasks["S3_minus_W2"]["top10_positive_share"],
                "support_ge20": best_tasks["S3_minus_W2"]["support_ge20"],
            },
            "per_task_csv_sha256": per_task_sha,
            "next_step_if_promising": (
                "train-only scaffold-grouped development split/folds, retraining W2 and S3 from "
                "scratch under the same best-validation epoch rule (NOT executed in this round)"
            ),
            "next_step_if_not_promising": (
                "stop the S3 hypothesis and return to W2 supervision / representation / generalisation "
                "mechanisms; do not add S4/S5, multi-hop interactions or bigger ranks"
            ),
        }
    )
    C.write_json(out_dir / "decision.json", decision)

    parity = _training_parity(s3_run, f_run, s3=s3, f=f)
    C.write_json(out_dir / "training_parity.json", parity)

    ledger = _ledger(s3=s3, f=f, rr_jobs=rr_jobs)
    C.write_json(out_dir / "gpu_ledger.json", ledger)

    return {
        "s3": s3,
        "f": f,
        "w2": w2,
        "comparison": comparison,
        "epoch40": epoch40,
        "decision": decision,
        "parity": parity,
        "ledger": ledger,
    }


def _training_parity(s3_run: Mapping[str, Any], f_run: Mapping[str, Any], *, s3: Mapping[str, Any], f: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "same_seed": int(s3_run["seed"]) == int(f_run["seed"]),
        "same_global_plan_sha256": s3_run["batch_plan"]["global_plan_sha256"] == f_run["batch_plan"]["global_plan_sha256"] == C.W2["global_plan_sha256"],
        "same_consumed40_prefix": s3_run["batch_plan"]["prefix_sha256"] == f_run["batch_plan"]["prefix_sha256"] == C.W2["consumed40_prefix_sha256"],
        "same_optimizer_recipe": s3_run["recipe"]["optimizer"] == f_run["recipe"]["optimizer"],
        "same_learning_rate": s3_run["recipe"]["learning_rate"] == f_run["recipe"]["learning_rate"],
        "same_weight_decay": s3_run["recipe"]["weight_decay"] == f_run["recipe"]["weight_decay"],
        "same_warmup": s3_run["recipe"]["warmup_epochs"] == f_run["recipe"]["warmup_epochs"],
        "same_dropout": s3_run["recipe"]["dropout"] == f_run["recipe"]["dropout"],
        "same_grad_clip": float(s3_run["grad_clip"]) == float(f_run["grad_clip"]),
        "same_epochs": int(s3_run["final_epoch"]) == int(f_run["final_epoch"]) == C.TOTAL_EPOCHS,
        "same_batch_size": s3_run["batch_plan"]["batch_size"] == f_run["batch_plan"]["batch_size"] == C.BATCH_SIZE,
        "same_base_parameters": int(s3_run["base_parameters"]) == int(f_run["base_parameters"]) == C.BASE_PARAMETERS,
        "same_base_init_state_sha256": s3_run["model_audit"]["base_state_sha256"] == f_run["model_audit"]["base_state_sha256"],
        "same_base_trunk_init_fingerprint": s3_run["model_audit"]["base_trunk_init_fingerprint"]
        == f_run["model_audit"]["base_trunk_init_fingerprint"]
        == C.W2["trunk_init_fingerprint"],
        "same_eval_cadence": int(s3_run["eval_every"]) == int(f_run["eval_every"]),
        "same_eval_epochs_completed": s3_run["eval_epochs_completed"] == f_run["eval_epochs_completed"],
        "no_test_access": (not s3_run["test_evaluated"]) and (not f_run["test_evaluated"]),
        "epoch0_ap_diff": abs(float(s3_run["epoch0_diagnostic"]["ap"]) - float(f_run["epoch0_diagnostic"]["ap"])),
        "parameter_gap": int(s3["new_parameters"]) - int(f["new_parameters"]),
        "parameter_gap_fraction": float(abs(int(s3["new_parameters"]) - int(f["new_parameters"])) / int(s3["new_parameters"])),
    }
    checks["epoch0_parity_pass"] = bool(checks["epoch0_ap_diff"] <= 1e-4)
    checks["parameter_match_pass"] = bool(checks["parameter_gap_fraction"] <= C.PARAMETER_GAP_MAX_FRACTION)
    checks["all_pass"] = bool(all(v for k, v in checks.items() if isinstance(v, bool)))
    return {
        "checks": checks,
        "s3_run": s3["_dir"] if "_dir" in s3 else None,
        "f_run": f["_dir"] if "_dir" in f else None,
        "note": "both arms start from the same untrained W2 seed-0 state, consume the same batch order and the same recipe",
    }


def _ledger(*, s3: Mapping[str, Any], f: Mapping[str, Any], rr_jobs: Path | None) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    if rr_jobs is not None and Path(rr_jobs).is_file():
        raw = json.loads(Path(rr_jobs).read_text(encoding="utf-8"))
        for job in raw.get("jobs", []):
            jobs.append(
                {
                    "experiment": job.get("experiment"),
                    "run_id": job.get("run_id"),
                    "pool": job.get("pool"),
                    "state": job.get("state"),
                    "commit": job.get("commit"),
                    "created": job.get("created"),
                    "finished": job.get("finished"),
                    "seconds": job.get("duration_seconds") or job.get("seconds"),
                }
            )
    formal = {
        "S3": float(s3["gpu_hours"]),
        "F": float(f["gpu_hours"]),
    }
    audit_hours = 0.0
    for job in jobs:
        if job["experiment"] and str(job["experiment"]).startswith("s3-") and not str(job["experiment"]).startswith(("s3-s3", "s3-f")):
            secs = job.get("seconds")
            if isinstance(secs, (int, float)):
                audit_hours += float(secs) / 3600.0
    total = sum(formal.values()) + audit_hours
    return {
        "formal_runs": formal,
        "audit_jobs_gpu_hours": float(audit_hours),
        "total_gpu_hours": float(total),
        "budget": dict(C.BUDGET),
        "within_target": bool(total <= float(C.BUDGET["gpu_target_hours"])),
        "within_hard_cap": bool(total <= float(C.BUDGET["gpu_hard_hours"])),
        "jobs": jobs,
        "note": "wall-clock GPU-hours are charged from the run records; failed or abandoned jobs are included",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-run", default=str(C.RUNS / "S3"))
    parser.add_argument("--f-run", default=str(C.RUNS / "F"))
    parser.add_argument("--replay", default=str(C.ANALYSIS / "w2_best_replay.json"))
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    parser.add_argument("--rr-jobs", default=str(C.ANALYSIS / "rr_jobs.json"))
    args = parser.parse_args(argv)
    result = build_analysis(
        s3_run_dir=Path(args.s3_run),
        f_run_dir=Path(args.f_run),
        replay_path=Path(args.replay),
        out_dir=Path(args.out_dir),
        rr_jobs=Path(args.rr_jobs) if args.rr_jobs else None,
    )
    C.log(
        f"decision {result['decision']['label']}  D_best_S3={result['decision']['d_best_s3']:+.6f} "
        f"D_best_F={result['decision']['d_best_f']:+.6f} D_mech={result['decision']['d_mech']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
