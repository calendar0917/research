"""P4 — task support vs macro-AP cancellation.

Recomputes the per-task AP of every frozen model on Validation and Test and
decomposes the macro-AP difference against W2 into task-level contributions,
support strata and the largest single-task effects.

The "drop top-k absolute changes" numbers are descriptive sensitivity only; the
official macro AP stays the primary number and is never replaced.

Outputs: ``analysis/task_support_effects.csv``, ``analysis/macro_ap_decomposition.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import lc_common as C
import lc_data as D

CSV_PATH = C.ANALYSIS / "task_support_effects.csv"
SUMMARY_PATH = C.ANALYSIS / "macro_ap_decomposition.json"


def _per_task_tables() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in ("valid", "test"):
        y = D.split_labels(split)
        out[split] = {
            "support": [D.task_support(y[:, t]) for t in range(C.N_TASKS)],
            "ap": {},
        }
        for label in C.MODEL_ORDER:
            logits = D.load_logits(label, split)
            out[split]["ap"][label] = [D.task_ap(y[:, t], logits[:, t]) for t in range(C.N_TASKS)]
    return out


def _delta_stats(deltas: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    """Summary statistics for one (pair, split) over the scorable tasks."""
    n_pos = support
    keep = np.isfinite(deltas)
    d = deltas[keep]
    p = n_pos[keep]
    out: dict[str, Any] = {
        "n_tasks_scored": int(d.size),
        "mean_delta_ap": float(d.mean()) if d.size else None,
        "median_delta_ap": float(np.median(d)) if d.size else None,
        "positive_sum": float(d[d > 0].sum()),
        "negative_sum": float(d[d < 0].sum()),
        "net_sum": float(d.sum()),
        "n_improved": int((d > 0).sum()),
        "n_degraded": int((d < 0).sum()),
        "n_tied": int((d == 0).sum()),
        "strata": {},
    }
    for stratum, lo, hi in (("ge20", 20, None), ("5_19", 5, 19), ("1_4", 1, 4)):
        m = p >= lo if hi is None else (p >= lo) & (p <= hi)
        sub = d[m]
        out["strata"][stratum] = {
            "n_tasks": int(sub.size),
            "mean_delta_ap": float(sub.mean()) if sub.size else None,
            "median_delta_ap": float(np.median(sub)) if sub.size else None,
            "sum_delta_ap": float(sub.sum()) if sub.size else None,
            "n_improved": int((sub > 0).sum()),
            "n_degraded": int((sub < 0).sum()),
            "mean_test_positive": float(p[m].mean()) if m.any() else None,
            "share_of_net": (float(sub.sum() / d.sum()) if d.sum() != 0 else None),
            "share_of_abs_movement": (
                float(np.abs(sub).sum() / np.abs(d).sum()) if np.abs(d).sum() != 0 else None
            ),
        }
    out["drop_top_abs"] = {}
    order = np.argsort(-np.abs(d))
    for k in (1, 3, 5):
        keep_idx = order[k:]
        out["drop_top_abs"][f"drop_top_{k}"] = {
            "n_tasks": int(keep_idx.size),
            "mean_delta_ap": float(d[keep_idx].mean()) if keep_idx.size else None,
        }
    return out


def _top_effects(tasks: np.ndarray, deltas: np.ndarray, support: np.ndarray, k: int = 5) -> dict[str, Any]:
    fin = np.isfinite(deltas)
    idx = np.flatnonzero(fin)
    order_pos = idx[np.argsort(-deltas[idx])]
    order_neg = idx[np.argsort(deltas[idx])]
    order_abs = idx[np.argsort(-np.abs(deltas[idx]))]

    def pack(order: np.ndarray) -> list[dict[str, Any]]:
        out = []
        for i in order[:k]:
            out.append(
                {
                    "task": int(tasks[i]),
                    "positive": int(support[i]),
                    "delta_ap": float(deltas[i]),
                    "contribution_to_mean": float(deltas[i] / fin.sum()),
                }
            )
        return out

    return {
        "top_positive": pack(order_pos),
        "top_negative": pack(order_neg),
        "top_absolute": pack(order_abs),
    }


def run() -> dict[str, Any]:
    tables = _per_task_tables()
    task_sets = C.read_json(C.ANALYSIS / "task_sets.json")
    high_tasks = set(int(t) for t in task_sets["HIGH_SUPPORT"])

    # ---- per-task CSV -------------------------------------------------------
    csv_rows: list[dict[str, Any]] = []
    for t in range(C.N_TASKS):
        row: dict[str, Any] = {"task": t, "high_support": t in high_tasks}
        for split in ("valid", "test"):
            s = tables[split]["support"][t]
            row[f"{split}_observed"] = s["observed"]
            row[f"{split}_positive"] = s["positive"]
            row[f"{split}_negative"] = s["negative"]
            row[f"{split}_positive_rate"] = (
                float(s["positive"] / s["observed"]) if s["observed"] else None
            )
            row[f"{split}_support_stratum"] = C.support_stratum(s["positive"], s["negative"])
            for label in C.MODEL_ORDER:
                row[f"{split}_ap_{label}"] = tables[split]["ap"][label][t]
        for pair in C.PRIMARY_PAIRS + (("BOC_BOND", "W2"),):
            a, b = pair
            key = C.pair_key(a, b)
            for split in ("valid", "test"):
                va = tables[split]["ap"][a][t]
                vb = tables[split]["ap"][b][t]
                row[f"{split}_delta_{key}"] = (None if va is None or vb is None else float(va - vb))
        csv_rows.append(row)
    C.save_csv(CSV_PATH, csv_rows)

    summary: dict[str, Any] = {
        "schema": "lcrc-pcba-bottleneck-diagnosis-macro-ap-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "created_utc": C.utcnow(),
        "official_macro_ap": {
            split: {label: _macro_ap(tables[split]["ap"][label]) for label in C.MODEL_ORDER}
            for split in ("valid", "test")
        },
        "definitions": {
            "delta_ap": "AP(new) - AP(W2) per task on observed labels of the same split",
            "mean_delta_ap": "unweighted mean over the 127 scorable tasks (the macro-AP difference)",
            "support_strata": {"ge20": "positive >=20", "5_19": "5..19", "1_4": "1..4"},
            "drop_top_abs": "descriptive sensitivity only; never a replacement for the official macro AP",
        },
        "pairs": {},
        "valid_test_reversals": {},
    }

    for pair in C.PRIMARY_PAIRS + (("BOC_BOND", "W2"),):
        a, b = pair
        pkey = C.pair_key(a, b)
        if a == "W2":
            continue
        entry: dict[str, Any] = {"new_model": a, "reference_model": b, "splits": {}}
        for split in ("valid", "test"):
            ap_new = np.array([np.nan if v is None else v for v in tables[split]["ap"][a]], dtype=np.float64)
            ap_ref = np.array([np.nan if v is None else v for v in tables[split]["ap"][b]], dtype=np.float64)
            deltas = ap_new - ap_ref
            support = np.array([s["positive"] for s in tables[split]["support"]], dtype=np.float64)
            stats = _delta_stats(deltas, support)
            stats["top_effects"] = _top_effects(np.arange(C.N_TASKS), deltas, support)
            entry["splits"][split] = stats

        # Support asymmetry verdict (PROTOCOL section 8), Test based.
        st = entry["splits"]["test"]
        low = st["strata"]["5_19"]["sum_delta_ap"] + st["strata"]["1_4"]["sum_delta_ap"]
        low_abs_share = st["strata"]["5_19"]["share_of_abs_movement"] + st["strata"]["1_4"]["share_of_abs_movement"]
        net = st["net_sum"]
        high_sum = st["strata"]["ge20"]["sum_delta_ap"]
        if low_abs_share >= C.LOW_SUPPORT_SHARE_STRONG and np.sign(low) != np.sign(high_sum):
            label = "H3_SUPPORTED_BY_SUPPORT_ASYMMETRY"
        elif low_abs_share < C.LOW_SUPPORT_SHARE_WEAK:
            label = "H3_NOT_SUPPORTED_BY_SUPPORT_ASYMMETRY"
        else:
            label = "H3_MIXED"
        entry["support_asymmetry"] = {
            "label": label,
            "test_low_support_sum": float(low),
            "test_high_support_sum": float(high_sum),
            "test_net_sum": float(net),
            "test_low_support_share_of_abs_movement": float(low_abs_share),
            "test_low_support_share_of_net": (float(low / net) if net != 0 else None),
        }

        # Valid/Test reversal accounting on tasks scorable on both splits.
        def _deltas(split: str) -> np.ndarray:
            return np.array(
                [
                    np.nan
                    if tables[split]["ap"][a][t] is None or tables[split]["ap"][b][t] is None
                    else tables[split]["ap"][a][t] - tables[split]["ap"][b][t]
                    for t in range(C.N_TASKS)
                ],
                dtype=np.float64,
            )

        dv = _deltas("valid")
        dt = _deltas("test")
        both = np.isfinite(dv) & np.isfinite(dt)
        high_mask = np.array([t in high_tasks for t in range(C.N_TASKS)])
        rev = {
            "n_tasks_both": int(both.sum()),
            "n_valid_positive_test_positive": int(np.sum(both & (dv > 0) & (dt > 0))),
            "n_valid_positive_test_negative": int(np.sum(both & (dv > 0) & (dt < 0))),
            "n_valid_negative_test_positive": int(np.sum(both & (dv < 0) & (dt > 0))),
            "n_valid_negative_test_negative": int(np.sum(both & (dv < 0) & (dt < 0))),
            "n_valid_positive_test_negative_high_support": int(
                np.sum(both & (dv > 0) & (dt < 0) & high_mask)
            ),
            "mean_delta_valid": float(np.nanmean(dv)),
            "mean_delta_test": float(np.nanmean(dt)),
        }
        summary["valid_test_reversals"][pkey] = rev
        summary["pairs"][pkey] = entry

    C.write_json(SUMMARY_PATH, summary)
    return summary


def _macro_ap(per_task: list[float | None]) -> float | None:
    vals = [float(v) for v in per_task if v is not None]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    s = run()
    print("macro AP:", {split: {k: round(v, 6) for k, v in ap.items()} for split, ap in s["official_macro_ap"].items()})
    for pkey, p in s["pairs"].items():
        t = p["splits"]["test"]
        v = p["splits"]["valid"]
        print(
            f"{pkey:24s} valid mean {v['mean_delta_ap']:+.5f} ({v['n_improved']}/{v['n_tasks_scored']}) | "
            f"test mean {t['mean_delta_ap']:+.5f} ({t['n_improved']}/{t['n_tasks_scored']}) | "
            f"hi-support test mean {t['strata']['ge20']['mean_delta_ap']:+.5f} "
            f"({t['strata']['ge20']['n_improved']}/{t['strata']['ge20']['n_tasks']}) | "
            f"low-abs-share {p['support_asymmetry']['test_low_support_share_of_abs_movement']:.3f} -> "
            f"{p['support_asymmetry']['label']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
