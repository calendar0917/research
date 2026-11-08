"""Per-task sign structure of the REAL - W2 Test AP delta (read-only analysis).

This script trains nothing and evaluates nothing: it reads the already frozen
per-task AP table of the RC round and answers the three questions the operator
asked after the interim peek, plus an audit of the headline "0.33%".

* sign distribution of ``d_t = AP_REAL(t) - AP_W2(t)`` (best-Validation and
  epoch-40 pairs, separately);
* share of the five largest *positive* deltas in the total positive mass, and
  what the five largest *absolute* deltas contribute (they are not the same
  five tasks -- the largest single deviation is negative);
* mean gain and improved fraction on the 108 high-support (test positive >= 20)
  tasks;
* where "0.33%" comes from and what it is *not*.

The official OGB macro AP is the unweighted mean of the per-task AP over the
scored tasks (equal task weights), which is why the mean of ``d_t`` equals the
difference of the two official scores to ~1e-16; that identity is checked
rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
ROUND = CODE.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]
ANALYSIS = ROUND / "analysis"

DEFAULT_CSV = ANALYSIS / "per_task_valid_test.csv"
FALLBACK_CSV = ANALYSIS / "per_task_valid_test_peek_real.csv"
OUT_PATH = ANALYSIS / "task_delta_analysis.json"

PAIRS = (
    ("test_ap_REAL_best_valid", "test_ap_W2_best_valid_epoch24", "REAL_best_valid_minus_W2_best_valid_epoch24"),
    ("test_ap_REAL_epoch40", "test_ap_W2_epoch40", "REAL_epoch40_minus_W2_epoch40"),
)


def _read(path: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty per-task table: {path}")
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values: list[float] = []
        numeric = True
        for row in rows:
            raw = row[key]
            if raw in ("", None):
                values.append(float("nan"))
                continue
            try:
                values.append(float(raw))
            except ValueError:
                numeric = False
                break
        if numeric:
            columns[key] = np.asarray(values, dtype=np.float64)
        else:
            columns[key] = np.asarray([row[key] for row in rows], dtype=object)
    return rows, columns


def _r(x: float | None) -> float | None:
    if x is None:
        return None
    value = float(x)
    return None if not math.isfinite(value) else round(value, 12)


def _task_list(values: np.ndarray, tasks: np.ndarray) -> list[dict[str, Any]]:
    return [{"task": int(t), "delta": _r(d)} for t, d in zip(tasks, values)]


def sign_structure(delta: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    """Positive/negative mass, concentration and the top-|d| sign pattern."""
    scored = np.isfinite(delta)
    d = delta[scored]
    tasks = np.arange(delta.size)[scored]
    pos = d > 0.0
    neg = d < 0.0
    zero = d == 0.0

    pos_sum = float(d[pos].sum())
    neg_sum = float(d[neg].sum())
    net = float(d.sum())
    mean = float(d.mean())

    pos_order = np.argsort(-d, kind="stable")
    pos_idx = pos_order[pos_order < d.size][:5]
    pos_idx = pos_idx[d[pos_idx] > 0.0]
    top5_pos_sum = float(d[pos_idx].sum())

    abs_order = np.argsort(-np.abs(d), kind="stable")[:5]
    top5_abs_net = float(d[abs_order].sum())

    ge20 = support[scored] >= 20
    d20 = d[ge20]
    return {
        "n_tasks_scored": int(d.size),
        "n_positive": int(pos.sum()),
        "n_negative": int(neg.sum()),
        "n_zero": int(zero.sum()),
        "positive_fraction": _r(pos.sum() / d.size),
        "mean_delta": _r(mean),
        "median_delta": _r(float(np.median(d))),
        "sum_delta": _r(net),
        "sum_positive": _r(pos_sum),
        "sum_negative": _r(neg_sum),
        "mean_positive_delta": _r(float(d[pos].mean())) if pos.any() else None,
        "mean_negative_delta": _r(float(d[neg].mean())) if neg.any() else None,
        "positive_over_negative_abs": _r(pos_sum / abs(neg_sum)) if neg_sum else None,
        "gross_mass": _r(pos_sum + abs(neg_sum)),
        "net_over_gross": _r(net / (pos_sum + abs(neg_sum))) if (pos_sum or neg_sum) else None,
        "top5_positive": {
            "tasks": _task_list(d[pos_idx], tasks[pos_idx]),
            "sum": _r(top5_pos_sum),
            "share_of_positive_mass": _r(top5_pos_sum / pos_sum) if pos_sum else None,
            "share_of_net": _r(top5_pos_sum / net) if net else None,
            "share_of_gross": _r(top5_pos_sum / (pos_sum + abs(neg_sum))),
        },
        "top5_absolute": {
            "tasks": _task_list(d[abs_order], tasks[abs_order]),
            "signs": ["positive" if d[i] > 0 else "negative" for i in abs_order],
            "sum_abs_deviation": _r(float(np.abs(d[abs_order]).sum())),
            "net_contribution": _r(top5_abs_net),
            "share_of_net": _r(top5_abs_net / net) if net else None,
            "share_of_gross": _r(float(np.abs(d[abs_order]).sum()) / (pos_sum + abs(neg_sum))),
            "leaves_net_without_them": _r(net - top5_abs_net),
        },
        "high_support_ge20": {
            "n_tasks": int(d20.size),
            "mean_delta": _r(float(d20.mean())) if d20.size else None,
            "median_delta": _r(float(np.median(d20))) if d20.size else None,
            "sum_delta": _r(float(d20.sum())) if d20.size else None,
            "improved": int((d20 > 0).sum()),
            "degraded": int((d20 < 0).sum()),
            "improved_fraction": _r(float((d20 > 0).mean())) if d20.size else None,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)

    path = Path(args.csv) if args.csv else (DEFAULT_CSV if DEFAULT_CSV.is_file() else FALLBACK_CSV)
    rows, col = _read(path)
    support = col["test_positive"]
    support_stratum = col.get("support_stratum")

    results: dict[str, Any] = {
        "schema": "lcrc-pcba-rc-task-delta-v1",
        "round_id": "lcrc-pcba-rc-v1",
        "source_table": str(path.relative_to(REPO)) if REPO in path.parents else str(path),
        "n_rows": len(rows),
        "support_column": "test_positive (official Test positives per task)",
        "pairs": {},
        "zero_and_missing": {
            "tasks_with_support_column": int(np.isfinite(support).sum()),
            "tasks_with_test_positive_lt2": [
                {"task": int(t), "test_positive": int(support[t]), "support_stratum": str(support_stratum[t])}
                for t in np.arange(support.size)[support < 2]
            ]
            if support_stratum is not None
            else [{"task": int(t), "test_positive": int(support[t])} for t in np.arange(support.size)[support < 2]],
        },
    }
    if support_stratum is not None:
        results["support_strata_counts"] = {
            str(s): int(np.sum(support_stratum == s)) for s in sorted(set(support_stratum.tolist()))
        }

    for new_col, old_col, label in PAIRS:
        if new_col not in col or old_col not in col:
            results["pairs"][label] = {"available": False}
            continue
        new = col[new_col]
        old = col[old_col]
        delta = new - old
        block = sign_structure(delta, support)
        block["available"] = True
        block["mean_new"] = _r(float(np.nanmean(new)))
        block["mean_old"] = _r(float(np.nanmean(old)))
        block["mean_gap_check"] = _r(float(np.nanmean(new) - np.nanmean(old) - np.nanmean(delta)))
        results["pairs"][label] = block

    # ---- where does "0.33%" come from, and what is it not? -----------------
    best = results["pairs"].get("REAL_best_valid_minus_W2_best_valid_epoch24", {})
    if best.get("available"):
        d = col["test_ap_REAL_best_valid"] - col["test_ap_W2_best_valid_epoch24"]
        w2 = col["test_ap_W2_best_valid_epoch24"]
        ge20 = support >= 20
        sc = np.isfinite(d)
        mean_all = float(d[sc].mean())
        mean20 = float(d[sc & ge20].mean())
        w2_all = float(w2[sc].mean())
        w2_20 = float(w2[sc & ge20].mean())
        official_delta = mean_all  # equal task weights
        results["percentage_audit"] = {
            "official_macro_ap_delta_absolute": _r(official_delta),
            "official_macro_ap_delta_in_ap_percentage_points": _r(100.0 * official_delta),
            "official_macro_ap_delta_relative_percent": _r(100.0 * official_delta / w2_all),
            "high_support_mean_delta_absolute": _r(mean20),
            "high_support_delta_in_ap_percentage_points": _r(100.0 * mean20),
            "high_support_relative_percent_of_w2_high_support_mean": _r(100.0 * mean20 / w2_20),
            "w2_macro_ap_mean_over_scored_tasks": _r(w2_all),
            "w2_mean_ap_on_high_support_tasks": _r(w2_20),
            "reproduction_of_quoted_0_33_percent": {
                "quantity": "100 * mean_{t in ge20}(AP_REAL(t) - AP_W2(t)) / mean_{t in ge20} AP_W2(t)",
                "value_percent": _r(100.0 * mean20 / w2_20),
                "matches_0_33_within_0_01pp": bool(abs(100.0 * mean20 / w2_20 - 0.33) < 0.01),
            },
            "denominators": {
                "note": (
                    "The official OGB macro AP is an equal-weight mean over the 127 scored tasks, so "
                    "the macro delta is exactly the mean per-task delta. A relative number is only "
                    "meaningful together with its denominator: the ge20 subset mean is ~0.273, the "
                    "full scored set is ~0.254, and the delta itself is 0.0023 AP (0.23 AP points)."
                ),
                "n_tasks_all": int(sc.sum()),
                "n_tasks_ge20": int((sc & ge20).sum()),
            },
            "sign_cancellation": {
                "gross_positive_mass": best["sum_positive"],
                "gross_negative_mass": best["sum_negative"],
                "net": best["sum_delta"],
                "net_over_gross": best["net_over_gross"],
                "note": (
                    "The net +0.0023 AP is a small residual of +/-1.5 AP of task-level movement; the "
                    "reported percentage describes the residual, not a uniform shift."
                ),
            },
        }

    out = Path(args.out)
    import json

    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
