"""Comparison + report numbers for the LCRC-RC round.

Reads only frozen artifacts:

* the historical W2 run (``lcrc_pcba_backbone_scaling_v1/runs/W2``),
* this round's ``runs/REAL`` and ``runs/BLIND``,
* ``analysis/test_scores.json`` (post-freeze official Test scores),
* ``analysis/profile.json`` / ``analysis/inference_timing.json``.

Writes ``analysis/comparison.json`` (the numbers quoted in ``REPORT.md``).
No model is loaded and nothing is retrained here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import rc_common as C  # noqa: E402

COMPARISON_PATH = C.ANALYSIS / "comparison.json"
COMPARISON_CSV = C.ANALYSIS / "comparison_table.csv"

#: label in test_scores.json for each arm / checkpoint pair
TEST_LABELS = {
    "W2": ("W2_best_valid_epoch24", "W2_epoch40"),
    "REAL": ("REAL_best_valid", "REAL_epoch40"),
    "BLIND": ("BLIND_best_valid", "BLIND_epoch40"),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_curve(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _curve_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    ap40 = _num(rows[-1].get("valid_ap"))
    best = max(
        (r for r in rows if _num(r.get("valid_ap")) is not None),
        key=lambda r: float(r["valid_ap"]),
    )
    tail = [r for r in rows if _num(r.get("residual_over_base")) is not None]
    return {
        "epochs": len(rows),
        "valid_ap_at_40": ap40,
        "best_epoch": int(best["epoch"]),
        "best_valid_ap": _num(best["valid_ap"]),
        "train_bce_at_40": _num(rows[-1].get("train_bce")),
        "epoch_seconds_mean": float(np.mean([float(r["epoch_seconds"]) for r in rows])),
        "step_ms_p50_mean": float(np.mean([_num(r.get("step_ms_p50")) or np.nan for r in rows])),
        "gpu_hours_sum": float(sum(float(r["epoch_seconds"]) for r in rows) / 3600.0),
        "branch": {
            "residual_over_base_last": _num(tail[-1]["residual_over_base"]) if tail else None,
            "residual_over_base_mean": float(
                np.mean([_num(r["residual_over_base"]) for r in tail])
            )
            if tail
            else None,
            "residual_rms_last": _num(tail[-1]["residual_rms"]) if tail else None,
            "role_entropy_last": _num(tail[-1]["role_entropy"]) if tail else None,
            "role_max_mean_last": _num(tail[-1]["role_max_mean"]) if tail else None,
            "branch_module_norm_last": _num(tail[-1]["branch_module_norm"]) if tail else None,
            "branch_grad_norm_last": _num(tail[-1]["branch_grad_norm"]) if tail else None,
        }
        if tail
        else {},
    }


def _arm(model_key: str) -> dict[str, Any]:
    if model_key == "W2":
        run_json = C.W2_RUN_JSON
        curve = C.W2_CURVE
        per_task = C.W2_PER_TASK
        train_eval = C.W2_TRAIN_EVAL
    else:
        run_json = C.RUNS / model_key / "run.json"
        curve = C.RUNS / model_key / "train_curve.csv"
        per_task = C.RUNS / model_key / "per_task_ap.csv"
        train_eval = C.RUNS / model_key / f"train_eval_epoch{C.TOTAL_EPOCHS}.json"
    if not Path(run_json).is_file():
        raise RuntimeError(f"{model_key}: run.json missing ({run_json})")
    run = _read_json(Path(run_json))
    curve_rows = _read_curve(Path(curve))
    summary = _curve_summary(curve_rows)
    out: dict[str, Any] = {
        "model": model_key,
        "available": True,
        "run_dir": C.track_rel(Path(run_json).parent),
        "status": run.get("status"),
        "parameters": int(run["parameters"]),
        "new_parameters": int(run.get("new_parameters", 0) or 0),
        "seed": int(run["seed"]),
        "best_epoch": int(run["best_epoch"]),
        "best_valid_ap": float(run["best_valid_ap"]),
        "valid_ap_at_40": float(run["fixed_epoch_reports"][str(C.TOTAL_EPOCHS)]["ap"]),
        "gpu_hours": float(run["gpu_hours"]),
        "train_seconds": float(run["train_seconds"]),
        "valid_seconds": float(run["valid_seconds"]),
        "gpu_peak_allocated_bytes": int(run.get("gpu_peak_allocated_bytes") or 0),
        "curve": summary,
        "distance_cache": run.get("distance_cache"),
        "host_facts": {k: run["host_facts"][k] for k in ("python", "torch", "torch_cuda", "cudnn") if k in run.get("host_facts", {})},
        "gpu_names": run.get("host_facts", {}).get("gpu_names"),
        "git_commit": run.get("git_commit"),
    }
    if Path(train_eval).is_file():
        te = _read_json(Path(train_eval))
        out["train_eval"] = {
            "epoch": int(te["epoch"]),
            "ap": float(te["ap"]),
            "bce": float(te["bce"]),
            "n_tasks_scored": int(te["n_tasks_scored"]),
            "evaluated_rows": int(te["evaluated_rows"]),
            "seconds": float(te["seconds"]),
        }
    if Path(per_task).is_file():
        with open(per_task, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        last = [r for r in rows if int(r["epoch"]) == int(run["best_epoch"])]
        out["per_task_valid_ap_at_best"] = [
            _num(r["ap"]) for r in sorted(last, key=lambda r: int(r["task"]))
        ]
    return out


def _wins(a: Sequence[Any], b: Sequence[Any], support: Sequence[int], *, min_support: int = 0,
          max_support: int | None = None) -> dict[str, Any]:
    deltas = []
    up = down = 0
    for x, y, s in zip(a, b, support):
        if x is None or y is None:
            continue
        s = int(s)
        if s < int(min_support):
            continue
        if max_support is not None and s > int(max_support):
            continue
        d = float(x) - float(y)
        deltas.append(d)
        up += int(d > 0)
        down += int(d < 0)
    if not deltas:
        return {"n_tasks": 0, "mean_delta": None, "improved": 0, "degraded": 0}
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "n_tasks": int(arr.size),
        "mean_delta": float(arr.mean()),
        "median_delta": float(np.median(arr)),
        "improved": int(up),
        "degraded": int(down),
        "worst_delta": float(arr.min()),
        "best_delta": float(arr.max()),
        "sum_delta": float(arr.sum()),
    }


def _sparse_dominance(deltas: list[float | None], support: Sequence[int]) -> dict[str, Any]:
    """How much of the macro-AP difference comes from a handful of tasks?"""
    pairs = [
        (abs(float(d)), float(d), int(s), i)
        for i, (d, s) in enumerate(zip(deltas, support))
        if d is not None
    ]
    total = float(sum(p[1] for p in pairs))
    ordered = sorted(pairs, key=lambda p: -p[0])
    top = ordered[:5]
    abs_total = float(sum(p[0] for p in pairs))
    top5_abs_share = float(sum(p[0] for p in top) / abs_total) if abs_total else 0.0
    top5_signed_share = float(sum(p[1] for p in top) / total) if total else 0.0
    n_large = int(sum(1 for p in pairs if p[0] >= 0.01))
    return {
        "n_tasks_scored": len(pairs),
        "sum_of_signed_deltas": total,
        "mean_signed_delta": total / len(pairs) if pairs else None,
        "top5_abs_share": top5_abs_share,
        "top5_signed_share_of_total": top5_signed_share,
        "top5": [
            {"task": int(idx), "delta": float(d), "test_support": int(s)} for _, d, s, idx in top
        ],
        "n_tasks_with_abs_delta_ge_0.01": n_large,
        "abs_delta_p50": float(np.median([p[0] for p in pairs])) if pairs else None,
        "abs_delta_p90": float(np.percentile([p[0] for p in pairs], 90)) if pairs else None,
        "max_abs_delta": float(ordered[0][0]) if ordered else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scores", default=str(C.ANALYSIS / "test_scores.json"))
    parser.add_argument("--out", default=str(COMPARISON_PATH))
    parser.add_argument("--profile", default=str(C.ANALYSIS / "profile.json"))
    parser.add_argument("--inference", default=str(C.ANALYSIS / "inference_timing.json"))
    args = parser.parse_args(argv)

    arms = {key: _arm(key) for key in ("W2", "REAL", "BLIND")}
    scores = _read_json(Path(args.test_scores))
    profile = _read_json(Path(args.profile)) if Path(args.profile).is_file() else None
    inference = _read_json(Path(args.inference)) if Path(args.inference).is_file() else None

    def test_entry(label: str) -> dict[str, Any]:
        entry = scores["models"][label]
        return {
            "label": label,
            "epoch": int(entry["epoch"]),
            "valid_ap": float(entry["valid_ap"]),
            "test_ap_official": float(entry["test_ap_official"]),
            "test_ap_independent": float(entry["test_ap_independent"]),
            "test_bce": float(entry["test_bce"]),
            "n_tasks_scored": int(entry["n_tasks_scored"]),
            "per_task_ap": entry["test_per_task_ap"],
            "test_positive": entry["test_positive"],
            "test_observed": entry["test_observed"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
        }

    test = {}
    for key, (best_label, epoch40_label) in TEST_LABELS.items():
        test[key] = {
            "best_valid": test_entry(best_label),
            "epoch40": test_entry(epoch40_label),
        }

    table = []
    for key in ("W2", "REAL", "BLIND"):
        arm = arms[key]
        table.append(
            {
                "model": key,
                "parameters": arm["parameters"],
                "best_valid_ap": arm["best_valid_ap"],
                "best_valid_epoch": arm["best_epoch"],
                "valid_ap_at_40": arm["valid_ap_at_40"],
                "test_ap_best_valid": test[key]["best_valid"]["test_ap_official"],
                "test_ap_epoch40": test[key]["epoch40"]["test_ap_official"],
                "gpu_hours": arm["gpu_hours"],
                "train_ap_at_40": (arm.get("train_eval") or {}).get("ap"),
                "train_bce_at_40": (arm.get("train_eval") or {}).get("bce"),
                "test_bce_best_valid": test[key]["best_valid"]["test_bce"],
            }
        )

    support = np.asarray(test["REAL"]["best_valid"]["test_positive"], dtype=np.int64)
    support_40 = np.asarray(test["W2"]["epoch40"]["test_positive"], dtype=np.int64)
    if not np.array_equal(support, support_40):
        raise RuntimeError("per-task Test support differs between checkpoints")

    comparisons: dict[str, Any] = {}
    for tag, new, old in (
        ("REAL_minus_W2", "REAL", "W2"),
        ("BLIND_minus_W2", "BLIND", "W2"),
        ("REAL_minus_BLIND", "REAL", "BLIND"),
    ):
        for which in ("best_valid", "epoch40"):
            a = test[new][which]["per_task_ap"]
            b = test[old][which]["per_task_ap"]
            entry = {
                "delta_test_ap": test[new][which]["test_ap_official"] - test[old][which]["test_ap_official"],
                "delta_valid_ap": test[new][which]["valid_ap"] - test[old][which]["valid_ap"],
                "overall": _wins(a, b, support),
                "support_ge20": _wins(a, b, support, min_support=20),
                "support_lt5": _wins(a, b, support, max_support=4),
                "support_5_19": _wins(a, b, support, min_support=5, max_support=19),
                "sparse_dominance": _sparse_dominance(a, support),
            }
            comparisons[f"{tag}_{which}"] = entry
        comparisons[f"{tag}_valid_ap_best"] = (
            arms[new]["best_valid_ap"] - arms[old]["best_valid_ap"]
        )
        comparisons[f"{tag}_valid_ap_40"] = arms[new]["valid_ap_at_40"] - arms[old]["valid_ap_at_40"]

    decision: dict[str, Any] = {}
    for tag in ("REAL_minus_W2", "BLIND_minus_W2", "REAL_minus_BLIND"):
        delta = comparisons[f"{tag}_best_valid"]["delta_test_ap"]
        rules = C.DECISION_RULES
        if delta >= rules["promising"]["delta_test_ap_min"]:
            label = rules["promising"]["label"]
        elif delta >= rules["weak"]["delta_test_ap_min"]:
            label = rules["weak"]["label"]
        elif delta > rules["none"]["delta_test_ap_min"]:
            label = rules["none"]["label"]
        else:
            label = rules["regression"]["label"]
        decision[tag] = {"delta_test_ap_best_valid": delta, "label": label}

    out = {
        "schema": C.SCHEMA_COMPARISON,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_role": C.TEST_ROLE,
        "official_evaluator": scores.get("official_evaluator"),
        "n_tasks": int(C.N_TASKS),
        "n_tasks_scored_test": int(scores["n_tasks_scored_test"]),
        "n_tasks_scored_valid": int(scores["n_tasks_scored_valid"]),
        "table": table,
        "valid_ap": {
            key: {
                "best_valid_ap": arms[key]["best_valid_ap"],
                "best_valid_epoch": arms[key]["best_epoch"],
                "valid_ap_at_40": arms[key]["valid_ap_at_40"],
            }
            for key in ("W2", "REAL", "BLIND")
        },
        "test_ap": {
            key: {
                "best_valid": test[key]["best_valid"]["test_ap_official"],
                "epoch40": test[key]["epoch40"]["test_ap_official"],
            }
            for key in ("W2", "REAL", "BLIND")
        },
        "comparisons": comparisons,
        "decision": decision,
        "arms": arms,
        "test_detail": {key: {w: {k: v for k, v in test[key][w].items() if k != "per_task_ap"} for w in test[key]} for key in test},
        "test_support": {
            "ge20_tasks": int((support >= 20).sum()),
            "ge5_lt20_tasks": int(((support >= 5) & (support < 20)).sum()),
            "lt5_tasks": int((support < 5).sum()),
            "n_scored": int(len(support)),
        },
        "profile": profile,
        "inference": inference,
        "disclaimer": (
            "single seed (0), one formal run per arm; differences are descriptive, "
            "not a statistical-significance claim"
        ),
    }
    C.write_json(Path(args.out), out)

    fields = [
        "model",
        "parameters",
        "best_valid_ap",
        "best_valid_epoch",
        "valid_ap_at_40",
        "test_ap_best_valid",
        "test_ap_epoch40",
        "gpu_hours",
        "train_ap_at_40",
        "test_bce_best_valid",
    ]
    with open(COMPARISON_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({k: row.get(k) for k in fields})

    for row in table:
        C.log(
            f"[compare] {row['model']:<6} params {row['parameters']} bestValid {row['best_valid_ap']:.9f} "
            f"valid40 {row['valid_ap_at_40']:.9f} test(best) {row['test_ap_best_valid']:.9f} "
            f"test@40 {row['test_ap_epoch40']:.9f} gpu_h {row['gpu_hours']:.4f}"
        )
    for tag, entry in decision.items():
        C.log(f"[compare] {tag}: delta test AP {entry['delta_test_ap_best_valid']:+.6f} -> {entry['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
