"""Build the LCRC-PW comparison table from the frozen artefacts.

Everything here is arithmetic over already-frozen JSON/CSV files: the PW
Validation/Test scores, the historical W2 reference and the RC round's
REAL / BLIND scores.  Nothing is re-trained, re-scored or re-selected, and the
RC artefacts are opened read-only.

Outputs
-------
``analysis/comparison.json``      full comparison record
``analysis/comparison_table.csv`` the flat table used by REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import pw_common as C  # noqa: E402

COMPARISON_PATH = C.ANALYSIS / "comparison.json"
TABLE_PATH = C.ANALYSIS / "comparison_table.csv"
SCORES_PATH = C.ANALYSIS / "test_scores.json"
ENTROPY_PATH = C.ANALYSIS / "query_entropy.json"
INFERENCE_PATH = C.ANALYSIS / "inference_timing.json"


def _r(value: Any, digits: int = 12) -> Any:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(v):
        return None
    return round(v, digits)


def _load_rc() -> dict[str, Any]:
    """RC-REAL / RC-BLIND / W2 Test scores from the RC round (read-only)."""
    if not C.RC_SCORES.is_file():
        return {}
    rc = C.read_json(C.RC_SCORES)
    out: dict[str, Any] = {"scores": rc.get("models", {}), "comparisons": rc.get("comparisons", {})}
    if C.RC_TASK_DELTA.is_file():
        out["task_delta_analysis"] = C.read_json(C.RC_TASK_DELTA)
    out["path"] = C.track_rel(C.RC_SCORES)
    return out


def _per_task_columns() -> dict[str, Any]:
    with open(C.PER_TASK_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, Any] = {"n_rows": len(rows), "columns": {}}
    if not rows:
        return out
    for key in rows[0]:
        values: list[Any] = []
        numeric = True
        for row in rows:
            raw = row[key]
            if raw == "":
                values.append(None)
                continue
            try:
                values.append(float(raw))
            except ValueError:
                numeric = False
                break
        if numeric:
            out["columns"][key] = values
        else:
            out["columns"][key] = [row[key] for row in rows]
    return out


def _paired(
    a: Sequence[Any], b: Sequence[Any], support: Sequence[float], *, lo: int = 0, hi: int | None = None
) -> list[tuple[int, float]]:
    """``(task_id, delta)`` for every task scored in both arms inside ``[lo, hi)``."""
    out: list[tuple[int, float]] = []
    for task, (x, y, s) in enumerate(zip(a, b, support)):
        if x is None or y is None:
            continue
        if float(s) < float(lo) or (hi is not None and float(s) >= float(hi)):
            continue
        out.append((int(task), float(x) - float(y)))
    return out


def _summary(pairs: Sequence[tuple[int, float]]) -> dict[str, Any]:
    if not pairs:
        return {"n_tasks": 0}
    tasks = [t for t, _ in pairs]
    arr = np.asarray([d for _, d in pairs], dtype=np.float64)
    pos = arr[arr > 0]
    neg = arr[arr < 0]
    abs_order = np.argsort(-np.abs(arr), kind="stable")[:5]
    pos_order = np.argsort(-arr, kind="stable")
    top_pos = [int(pos_order[i]) for i in range(arr.size) if arr[pos_order[i]] > 0][:5]
    return {
        "n_tasks": int(arr.size),
        "mean_delta": _r(arr.mean()),
        "median_delta": _r(float(np.median(arr))),
        "sum_delta": _r(arr.sum()),
        "improved": int((arr > 0).sum()),
        "degraded": int((arr < 0).sum()),
        "improved_fraction": _r(float((arr > 0).sum()) / arr.size),
        "sum_positive": _r(pos.sum()) if pos.size else 0.0,
        "sum_negative": _r(neg.sum()) if neg.size else 0.0,
        "top5_positive_share_of_positive_mass": _r(
            float(pos[np.argsort(-pos)[:5]].sum()) / float(pos.sum()) if pos.size else None
        ),
        "top5_absolute": [{"task": tasks[int(i)], "delta": _r(arr[int(i)])} for i in abs_order],
        "top5_absolute_net": _r(float(arr[abs_order].sum())),
        "top5_absolute_share_of_net": _r(float(arr[abs_order].sum()) / float(arr.sum())) if float(arr.sum()) else None,
        "top5_positive": [{"task": tasks[int(i)], "delta": _r(arr[int(i)])} for i in top_pos],
        "worst_delta": _r(arr.min()),
        "best_delta": _r(arr.max()),
        "worst_task": tasks[int(np.argmin(arr))],
        "best_task": tasks[int(np.argmax(arr))],
    }


def _pair_stats(a: Sequence[Any], b: Sequence[Any], support: Sequence[float], *, min_support: int = 0) -> dict[str, Any]:
    return _summary(_paired(a, b, support, lo=int(min_support)))


def build() -> dict[str, Any]:
    if not SCORES_PATH.is_file():
        raise RuntimeError(f"PW Test scores missing: {SCORES_PATH}")
    pw = C.read_json(SCORES_PATH)
    rc = _load_rc()
    per_task = _per_task_columns()

    models: dict[str, Any] = {}
    for label, entry in pw["models"].items():
        models[label] = {
            "model": entry["model"],
            "epoch": int(entry["epoch"]),
            "parameters": None,
            "valid_ap": _r(entry["valid_ap"]),
            "test_ap_official": _r(entry["test_ap_official"]),
            "test_ap_independent": _r(entry["test_ap_independent"]),
            "test_bce": _r(entry["test_bce"]),
            "n_tasks_scored": int(entry["n_tasks_scored"]),
            "role": entry["role"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
        }
    run_path = C.RUNS / C.MODEL_KEY / "run.json"
    run = C.read_json(run_path) if run_path.is_file() else {}
    for key in ("PW_best_valid", "PW_epoch40"):
        models[key]["parameters"] = int(run.get("parameters", C.EXPECTED_PARAMETERS))
        models[key]["gpu_hours"] = _r(run.get("gpu_hours"))
        models[key]["new_parameters"] = int(run.get("new_parameters", C.BRANCH_ADDED))

    w2 = rc.get("scores", {})
    for label, source in (
        ("W2_best_valid_epoch24", "W2_best_valid_epoch24"),
        ("W2_epoch40", "W2_epoch40"),
    ):
        if source in w2:
            entry = w2[source]
            models[label] = {
                "model": "W2",
                "epoch": int(entry["epoch"]),
                "parameters": int(C.W2["parameters"]),
                "valid_ap": _r(entry["valid_ap"]),
                "test_ap_official": _r(entry["test_ap_official"]),
                "test_ap_independent": _r(entry.get("test_ap_independent")),
                "n_tasks_scored": int(entry.get("n_tasks_scored", 0)),
                "gpu_hours": _r(C.W2["gpu_hours"]),
                "role": "historical W2 reference (same scoring pipeline)",
            }
    for label, source in (
        ("RC_REAL_best_valid", "REAL_best_valid"),
        ("RC_REAL_epoch40", "REAL_epoch40"),
        ("RC_BLIND_best_valid", "BLIND_best_valid"),
        ("RC_BLIND_epoch40", "BLIND_epoch40"),
    ):
        if source in w2:
            entry = w2[source]
            models[label] = {
                "model": source.split("_")[0],
                "epoch": int(entry["epoch"]),
                "parameters": 1_051_022,
                "valid_ap": _r(entry["valid_ap"]),
                "test_ap_official": _r(entry["test_ap_official"]),
                "test_ap_independent": _r(entry.get("test_ap_independent")),
                "test_bce": _r(entry.get("test_bce")),
                "n_tasks_scored": int(entry.get("n_tasks_scored", 0)),
                "role": "RC round reference (read-only)",
            }

    rc_runs = {}
    for key in ("REAL", "BLIND"):
        path = C.RC_ROUND / "runs" / key / "run.json"
        if path.is_file():
            payload = C.read_json(path)
            rc_runs[key] = {"gpu_hours": _r(payload.get("gpu_hours")), "parameters": int(payload.get("parameters", 0))}
            for label, suffix in ((f"RC_{key}_best_valid", "best"), (f"RC_{key}_epoch40", "e40")):
                if label in models:
                    models[label]["gpu_hours"] = rc_runs[key]["gpu_hours"]
                    models[label]["parameters"] = rc_runs[key]["parameters"]

    # ---- deltas ------------------------------------------------------------
    compare_pairs = [
        ("PW_vs_W2_best", "PW_best_valid", "W2_best_valid_epoch24"),
        ("PW_vs_W2_epoch40", "PW_epoch40", "W2_epoch40"),
        ("PW_vs_RC_REAL_best", "PW_best_valid", "RC_REAL_best_valid"),
        ("PW_vs_RC_BLIND_best", "PW_best_valid", "RC_BLIND_best_valid"),
        ("PW_vs_RC_REAL_epoch40", "PW_epoch40", "RC_REAL_epoch40"),
        ("PW_vs_RC_BLIND_epoch40", "PW_epoch40", "RC_BLIND_epoch40"),
        ("RC_REAL_vs_W2", "RC_REAL_best_valid", "W2_best_valid_epoch24"),
        ("RC_BLIND_vs_W2", "RC_BLIND_best_valid", "W2_best_valid_epoch24"),
        ("RC_REAL_vs_RC_BLIND", "RC_REAL_best_valid", "RC_BLIND_best_valid"),
    ]
    deltas: dict[str, Any] = {}
    for name, new, old in compare_pairs:
        if new not in models or old not in models:
            continue
        deltas[name] = {
            "new": new,
            "old": old,
            "delta_test_ap": _r(models[new]["test_ap_official"] - models[old]["test_ap_official"]),
            "delta_valid_ap": _r(models[new]["valid_ap"] - models[old]["valid_ap"]),
            "relative_test_percent": _r(
                100.0
                * (models[new]["test_ap_official"] - models[old]["test_ap_official"])
                / models[old]["test_ap_official"]
            ),
        }

    # ---- per-task analysis on the joined table -----------------------------
    col = per_task["columns"]
    support = col.get("test_positive", [])
    per_task_stats: dict[str, Any] = {}
    for name, a, b in (
        ("PW_minus_W2", "test_ap_PW_best_valid", "test_ap_W2_best_valid_epoch24"),
        ("PW_minus_RC_REAL", "test_ap_PW_best_valid", "RC_REAL_best_valid"),
        ("PW_minus_RC_BLIND", "test_ap_PW_best_valid", "RC_BLIND_best_valid"),
        ("PW_e40_minus_W2_e40", "test_ap_PW_epoch40", "test_ap_W2_epoch40"),
    ):
        if a not in col or b not in col:
            per_task_stats[name] = {"available": False}
            continue
        per_task_stats[name] = {
            "available": True,
            "all_tasks": _pair_stats(col[a], col[b], support),
            "support_ge20": _pair_stats(col[a], col[b], support, min_support=20),
            "support_5_19": _pair_stats_band(col[a], col[b], support, lo=5, hi=20),
            "support_lt5": _pair_stats_band(col[a], col[b], support, lo=0, hi=5),
            "strata_note": (
                "support_ge20 uses the >=20 filter, support_5_19 is the 5..19 band and "
                "support_lt5 the 0..4 band; each is computed from the same per-task table"
            ),
        }

    # ---- train / valid gap -------------------------------------------------
    gaps: dict[str, Any] = {}
    if run and run.get("train_eval_final"):
        train_ap = float(run["train_eval_final"]["ap"])
        e40 = run["fixed_epoch_reports"].get(str(C.TOTAL_EPOCHS), {})
        best_rep = run["fixed_epoch_reports"].get(str(run["best_epoch"]), {})
        gaps["PW"] = {
            "train_ap_epoch40": _r(train_ap),
            "train_bce_epoch40": _r(run["train_eval_final"]["bce"]),
            "valid_ap_epoch40": _r(e40.get("ap")),
            "valid_ap_best": _r(run.get("best_valid_ap")),
            "valid_bce_epoch40": _r(e40.get("valid_bce")),
            "valid_bce_best": _r(best_rep.get("valid_bce")),
            "train_minus_valid_ap": _r(train_ap - float(e40.get("ap", float("nan")))),
            "valid_minus_test_best": _r(models["PW_best_valid"]["valid_ap"] - models["PW_best_valid"]["test_ap_official"]),
        }
    w2_train = C.W2.get("train_ap40")
    if w2_train is not None:
        gaps["W2"] = {
            "train_ap_epoch40": _r(w2_train),
            "train_bce_epoch40": _r(C.W2.get("train_bce40")),
            "valid_ap_epoch40": _r(C.W2.get("ap40")),
            "valid_bce_epoch40": _r(C.W2.get("valid_bce40")),
            "train_minus_valid_ap": _r(float(w2_train) - float(C.W2["ap40"])),
        }
    real_run = C.RC_ROUND / "runs" / "REAL" / "run.json"
    if real_run.is_file():
        payload = C.read_json(real_run)
        gaps["RC_REAL"] = _train_gap(payload)
    blind_run = C.RC_ROUND / "runs" / "BLIND" / "run.json"
    if blind_run.is_file():
        payload = C.read_json(blind_run)
        gaps["RC_BLIND"] = _train_gap(payload)

    # ---- attention entropy ------------------------------------------------
    entropy = C.read_json(ENTROPY_PATH) if ENTROPY_PATH.is_file() else {}
    inference = C.read_json(INFERENCE_PATH) if INFERENCE_PATH.is_file() else {}

    # ---- decision ---------------------------------------------------------
    best_valid = "PW_best_valid"
    reference_candidates = {
        k: models[k]["test_ap_official"]
        for k in ("W2_best_valid_epoch24", "RC_REAL_best_valid", "RC_BLIND_best_valid")
        if k in models
    }
    strongest_reference = max(reference_candidates, key=reference_candidates.get)
    strongest_value = float(reference_candidates[strongest_reference])
    delta_vs_strongest = float(models[best_valid]["test_ap_official"]) - strongest_value
    decision = _decision(delta_vs_strongest)

    table_rows = []
    for label, entry in models.items():
        table_rows.append(
            {
                "model": label,
                "family": entry["model"],
                "epoch": entry.get("epoch"),
                "parameters": entry.get("parameters"),
                "best_valid_ap": entry["valid_ap"],
                "test_ap_best": entry["test_ap_official"],
                "test_bce": entry.get("test_bce"),
                "gpu_hours": entry.get("gpu_hours"),
            }
        )

    out = {
        "schema": C.SCHEMA_COMPARISON,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_role": C.TEST_ROLE,
        "sources": {
            "pw_scores": C.track_rel(SCORES_PATH),
            "pw_per_task": C.track_rel(C.PER_TASK_PATH),
            "pw_run": C.track_rel(run_path) if run_path.is_file() else None,
            "rc_scores": rc.get("path"),
            "rc_per_task": C.track_rel(C.RC_PER_TASK) if C.RC_PER_TASK.is_file() else None,
            "query_entropy": C.track_rel(ENTROPY_PATH) if ENTROPY_PATH.is_file() else None,
            "inference": C.track_rel(INFERENCE_PATH) if INFERENCE_PATH.is_file() else None,
        },
        "models": models,
        "deltas": deltas,
        "per_task": per_task_stats,
        "train_valid_gap": gaps,
        "attention_entropy": entropy,
        "inference": inference,
        "rc_task_delta_analysis": rc.get("task_delta_analysis"),
        "decision": {
            "rule": "best-Validation Test AP delta against the strongest existing LCRC (W2 / RC-REAL / RC-BLIND)",
            "threshold_consistency": _band_consistency(),
            "reference_model": strongest_reference,
            "reference_test_ap": _r(strongest_value),
            "pw_test_ap": _r(models[best_valid]["test_ap_official"]),
            "delta_test_ap": _r(delta_vs_strongest),
            "label": decision["label"],
            "bands": C.DECISION_RULES,
            "interpretation": decision["interpretation"],
        },
        "n_tasks_scored": per_task.get("n_rows"),
        "descriptive_only": True,
    }
    C.write_json(COMPARISON_PATH, out)
    _write_table(TABLE_PATH, table_rows, out)
    return out


def _pair_stats_band(
    a: Sequence[Any], b: Sequence[Any], support: Sequence[float], *, lo: int, hi: int
) -> dict[str, Any]:
    return _summary(_paired(a, b, support, lo=int(lo), hi=int(hi)))


def _train_gap(payload: Mapping[str, Any]) -> dict[str, Any]:
    train = payload.get("train_eval_final") or {}
    e40 = (payload.get("fixed_epoch_reports") or {}).get(str(C.TOTAL_EPOCHS), {})
    return {
        "train_ap_epoch40": _r(train.get("ap")),
        "train_bce_epoch40": _r(train.get("bce")),
        "valid_ap_epoch40": _r(e40.get("ap")),
        "valid_bce_epoch40": _r(e40.get("valid_bce")),
        "valid_ap_best": _r(payload.get("best_valid_ap")),
        "train_minus_valid_ap": _r(float(train.get("ap")) - float(e40.get("ap")))
        if train.get("ap") is not None and e40.get("ap") is not None
        else None,
    }


def _band_consistency() -> dict[str, Any]:
    """The PW bands must use exactly the RC thresholds (labels differ by design)."""
    pairs = {}
    for key in ("promising", "weak", "none"):
        inherited = float(C.INHERITED_RC_DECISION_RULES[key]["delta_test_ap_min"])
        current = float(C.DECISION_RULES[key]["delta_test_ap_min"])
        pairs[key] = {"inherited": inherited, "pw": current, "equal": bool(inherited == current)}
    return {"thresholds": pairs, "all_equal": bool(all(v["equal"] for v in pairs.values()))}


def _decision(delta: float) -> dict[str, Any]:
    if delta >= C.DECISION_RULES["promising"]["delta_test_ap_min"]:
        return {
            "label": C.DECISION_RULES["promising"]["label"],
            "interpretation": "PW Test AP is at least +0.005 above the strongest existing LCRC.",
        }
    if delta >= C.DECISION_RULES["weak"]["delta_test_ap_min"]:
        return {
            "label": C.DECISION_RULES["weak"]["label"],
            "interpretation": "PW Test AP is +0.002..+0.005 above the strongest existing LCRC.",
        }
    if delta > C.DECISION_RULES["none"]["delta_test_ap_min"]:
        return {
            "label": C.DECISION_RULES["none"]["label"],
            "interpretation": "PW Test AP is within +/-0.002 of the strongest existing LCRC.",
        }
    return {
        "label": C.DECISION_RULES["regression"]["label"],
        "interpretation": "PW Test AP is at least 0.002 below the strongest existing LCRC.",
    }


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]], full: Mapping[str, Any]) -> None:
    fields = ["model", "family", "epoch", "parameters", "best_valid_ap", "test_ap_best", "test_bce", "gpu_hours"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(COMPARISON_PATH))
    parser.add_argument("--table", default=str(TABLE_PATH))
    args = parser.parse_args(argv)
    out = build()
    if Path(args.out) != COMPARISON_PATH or Path(args.table) != TABLE_PATH:
        C.write_json(Path(args.out), out)
    print(f"wrote {COMPARISON_PATH} and {TABLE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
