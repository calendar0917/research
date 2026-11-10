"""Comparison, per-task deltas, curves, decision labels and figures.

Reads the two pulled run directories plus the read-only historical records, and
writes ``analysis/comparison.json``, ``analysis/comparison.csv``,
``analysis/curves_epoch.csv``, ``analysis/per_task_delta.csv``,
``analysis/decision.json`` and the figures under ``analysis/figures/``.

The primary metric is Validation AP; Test is reported as
``EXPOSED_TEST_EXPLORATORY`` and never used to pick a checkpoint.
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

import rd_common as C  # noqa: E402

ANALYSIS = C.ANALYSIS
FIGURES = C.FIGURES

COMPARISON_FIELDS = [
    "model",
    "kind",
    "params",
    "dropout",
    "weight_decay",
    "best_epoch",
    "best_valid_ap",
    "train_ap_at_best",
    "test_ap_at_best",
    "train_ap_40",
    "valid_ap_40",
    "test_ap_40",
    "d_best_vs_w3",
    "d_ap40_vs_w3",
    "d_best_vs_w2",
    "d_ap40_vs_w2",
    "gpu_hours",
    "label",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return C.sha256_file(path)


def _f(x: Any) -> float:
    return float(x)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)[0])


def _ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, float)
    ranks[order] = np.arange(arr.size, dtype=float)
    # average ties
    sorted_vals = arr[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return None
    return float((rx * ry).sum() / denom)


def lr_budget() -> dict[str, Any]:
    """Total AdamW decay budget over the 40-epoch schedule (lr-summed)."""
    steps = C.TOTAL_STEPS_40
    total = float(sum(C.scheduled_lr(s, phase=3, kind="all") for s in range(steps)))
    return {
        "steps": steps,
        "sum_lr": total,
        "mean_lr": total / steps,
        "decay_budget": {
            key: {
                "weight_decay": float(C.RUNS[key]["weight_decay"]),
                "decay_sum": float(C.RUNS[key]["weight_decay"]) * total,
                "multiplicative_shrink": float(np.exp(-float(C.RUNS[key]["weight_decay"]) * total)),
                "percent_shrink": float(100.0 * (1.0 - np.exp(-float(C.RUNS[key]["weight_decay"]) * total))),
            }
            for key in C.RUN_ORDER
        },
        "historical_w3": {
            "weight_decay": 1.0e-3,
            "decay_sum": 1.0e-3 * total,
            "percent_shrink": float(100.0 * (1.0 - np.exp(-1.0e-3 * total))),
        },
    }


def load_run(run_key: str) -> dict[str, Any]:
    run_dir = C.run_dir(run_key)
    run = C.read_json(run_dir / "run.json")
    metrics = _read_csv(run_dir / "epoch_metrics.csv")
    logs = _read_csv(run_dir / "epoch_log.csv")
    tasks = _read_csv(run_dir / "per_task_ap.csv")
    series: dict[str, dict[int, dict[str, float]]] = {s: {} for s in C.SPLITS}
    per_epoch: dict[int, dict[str, dict[str, float]]] = {}
    for row in metrics:
        epoch = int(float(row["epoch"]))
        split = row["split"]
        rec = {
            "ap": _f(row["ap_official"]),
            "ap_sklearn": _f(row["ap_sklearn"]),
            "ap_abs_delta": _f(row["ap_abs_delta"]),
            "bce": _f(row["bce"]),
            "n_tasks_scored": int(float(row["n_tasks_scored"])),
            "rows": int(float(row["rows"])),
            "eval_seconds": _f(row["eval_seconds"]),
            "infer_seconds": _f(row["infer_seconds"]),
            "observed_entries": int(float(row["observed_entries"])),
            "positive_entries": int(float(row["positive_entries"])),
        }
        series[split][epoch] = rec
        per_epoch.setdefault(epoch, {})[split] = rec
    task_ap: dict[tuple[int, str], list[float | None]] = {}
    task_support: dict[tuple[int, str], dict[str, list[int]]] = {}
    for row in tasks:
        key = (int(float(row["epoch"])), row["split"])
        task_ap.setdefault(key, [None] * C.N_TASKS)
        task_support.setdefault(key, {"observed": [0] * C.N_TASKS, "positive": [0] * C.N_TASKS})
        t = int(float(row["task"]))
        task_ap[key][t] = None if row["ap"] in ("", "None") else _f(row["ap"])
        task_support[key]["observed"][t] = int(float(row["observed"]))
        task_support[key]["positive"][t] = int(float(row["positive"]))
    return {
        "run_key": run_key,
        "run": run,
        "run_dir": run_dir,
        "metrics": metrics,
        "logs": logs,
        "series": series,
        "per_epoch": per_epoch,
        "task_ap": task_ap,
        "task_support": task_support,
        "best_epoch": int(run["best_epoch"]),
        "best_valid_ap": float(run["best_valid_ap"]),
    }


def load_historical_test() -> dict[str, Any]:
    path = ANALYSIS / "historical_test.json"
    if not path.is_file():
        return {"available": False}
    data = C.read_json(path)
    return {
        "available": True,
        "path": C.track_rel(path),
        "test_ap_best": data.get("historical_test_ap_best_valid"),
        "test_ap_epoch40": data.get("historical_test_ap_epoch40"),
        "best_epoch_task_ap": data["checkpoints"]["best_valid"].get("test_per_task_ap"),
        "epoch40_task_ap": data["checkpoints"]["epoch40"].get("test_per_task_ap"),
        "n_tasks_scored": data["checkpoints"]["best_valid"].get("test_n_tasks_scored"),
        "valid_alignment": data.get("valid_alignment"),
    }


def historical_valid_task_ap(epoch: int) -> list[float | None] | None:
    path = C.BS_RUNS / "W3" / "per_task_ap.csv"
    if not path.is_file():
        return None
    rows = [r for r in _read_csv(path) if int(float(r["epoch"])) == int(epoch)]
    if not rows:
        return None
    out: list[float | None] = [None] * C.N_TASKS
    for row in rows:
        out[int(float(row["task"]))] = None if row["ap"] == "" else _f(row["ap"])
    return out


def _stratified_deltas(
    ours: Sequence[float | None],
    theirs: Sequence[float | None],
    support: Sequence[int],
) -> dict[str, Any]:
    strata = {
        "all_scored": lambda p: p >= 1,
        "support_ge20": lambda p: p >= 20,
        "support_5_19": lambda p: 5 <= p < 20,
        "support_lt5": lambda p: p < 5,
    }
    out: dict[str, Any] = {}
    for name, rule in strata.items():
        deltas = []
        improved = 0
        worse = 0
        for t in range(C.N_TASKS):
            if ours[t] is None or theirs[t] is None:
                continue
            if not rule(int(support[t])):
                continue
            d = float(ours[t]) - float(theirs[t])
            deltas.append(d)
            improved += int(d > 0)
            worse += int(d < 0)
        out[name] = {
            "n": len(deltas),
            "mean_delta": _mean(deltas),
            "median_delta": _median(deltas),
            "improved": improved,
            "worse": worse,
            "unchanged": len(deltas) - improved - worse,
            "min_delta": float(min(deltas)) if deltas else None,
            "max_delta": float(max(deltas)) if deltas else None,
        }
    return out


def _top_concentration(deltas: Mapping[int, float], k: int = 10) -> dict[str, Any]:
    pos = {t: d for t, d in deltas.items() if d > 0}
    total = float(sum(pos.values()))
    top = sorted(pos.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return {
        "sum_positive_delta": total,
        f"top{k}_sum": float(sum(d for _, d in top)),
        f"top{k}_share": (float(sum(d for _, d in top)) / total) if total > 0 else None,
        f"top{k}_tasks": [int(t) for t, _ in top],
        f"top{k}_deltas": [float(d) for _, d in top],
    }


def analyse_run(run_key: str, hist_test: Mapping[str, Any]) -> dict[str, Any]:
    data = load_run(run_key)
    run = data["run"]
    best_epoch = data["best_epoch"]
    series = data["series"]
    epochs = sorted(series["train"])
    if epochs != list(range(1, C.TOTAL_EPOCHS + 1)):
        raise RuntimeError(f"{run_key}: expected 40 evaluated epochs, got {len(epochs)}")

    s_best = {s: series[s][best_epoch] for s in C.SPLITS}
    s_40 = {s: series[s][C.TOTAL_EPOCHS] for s in C.SPLITS}
    at_report_epochs = {
        str(e): {s: series[s][e]["ap"] for s in C.SPLITS} for e in C.REPORT_EPOCHS
    }
    tail = [e for e in epochs if e >= 20]
    slopes = {s: _slope(tail, [series[s][e]["ap"] for e in tail]) for s in C.SPLITS}

    valid_support = data["task_support"][(best_epoch, "valid")]["positive"]
    valid_ours = data["task_ap"][(best_epoch, "valid")]
    hist_valid_best = historical_valid_task_ap(C.HISTORICAL["W3"]["best_epoch"])
    hist_valid_40 = historical_valid_task_ap(40)
    strat_valid_best = (
        _stratified_deltas(valid_ours, hist_valid_best, valid_support) if hist_valid_best else None
    )
    strat_valid_40 = (
        _stratified_deltas(data["task_ap"][(40, "valid")], hist_valid_40, valid_support)
        if hist_valid_40
        else None
    )

    test_ours = data["task_ap"][(best_epoch, "test")]
    hist_test_best = hist_test.get("best_epoch_task_ap") if hist_test.get("available") else None
    test_support = data["task_support"][(best_epoch, "test")]["positive"]
    strat_test_best = (
        _stratified_deltas(test_ours, hist_test_best, test_support) if hist_test_best else None
    )

    # Train-Valid gap: own-scorable macro AP and the common scorable task set
    train_scorable = [t for t in range(C.N_TASKS) if data["task_ap"][(best_epoch, "train")][t] is not None]
    valid_scorable = [t for t in range(C.N_TASKS) if data["task_ap"][(best_epoch, "valid")][t] is not None]
    common = sorted(set(train_scorable) & set(valid_scorable))
    gap_common_best = {
        "n_common_tasks": len(common),
        "train_macro_common": _mean([data["task_ap"][(best_epoch, "train")][t] for t in common]),
        "valid_macro_common": _mean([data["task_ap"][(best_epoch, "valid")][t] for t in common]),
    }
    gap_common_best["gap_common"] = gap_common_best["train_macro_common"] - gap_common_best["valid_macro_common"]
    gap_own_best = {
        "train_macro_own": _mean([data["task_ap"][(best_epoch, "train")][t] for t in train_scorable]),
        "valid_macro_own": _mean([data["task_ap"][(best_epoch, "valid")][t] for t in valid_scorable]),
        "n_train_scored": len(train_scorable),
        "n_valid_scored": len(valid_scorable),
    }
    gap_own_best["gap_own"] = gap_own_best["train_macro_own"] - gap_own_best["valid_macro_own"]

    def _gap(epoch: int) -> dict[str, Any]:
        tr = data["task_ap"][(epoch, "train")]
        va = data["task_ap"][(epoch, "valid")]
        tr_tasks = [t for t in range(C.N_TASKS) if tr[t] is not None]
        va_tasks = [t for t in range(C.N_TASKS) if va[t] is not None]
        common_t = sorted(set(tr_tasks) & set(va_tasks))
        own = {
            "train_macro_own": _mean([tr[t] for t in tr_tasks]),
            "valid_macro_own": _mean([va[t] for t in va_tasks]),
            "n_train_scored": len(tr_tasks),
            "n_valid_scored": len(va_tasks),
        }
        own["gap_own"] = own["train_macro_own"] - own["valid_macro_own"]
        com = {
            "n_common_tasks": len(common_t),
            "train_macro_common": _mean([tr[t] for t in common_t]),
            "valid_macro_common": _mean([va[t] for t in common_t]),
        }
        com["gap_common"] = com["train_macro_common"] - com["valid_macro_common"]
        return {"own_scorable": own, "common_scorable": com}

    gap_at_40 = _gap(40)

    last10 = [e for e in epochs if e >= 31]
    noise_floor = {}
    for split in C.SPLITS:
        vals = [series[split][e]["ap"] for e in last10]
        diffs = np.diff([series[split][e]["ap"] for e in tail])
        noise_floor[split] = {
            "last10_mean": float(np.mean(vals)),
            "last10_std": float(np.std(vals)),
            "last10_range": float(max(vals) - min(vals)),
            "step_diff_std_20_40": float(np.std(diffs)),
        }

    def _delta_vs_hist(epoch: int, split: str) -> dict[int, float]:
        ours = data["task_ap"][(epoch, split)]
        theirs = historical_valid_task_ap(epoch) if split == "valid" else None
        if split == "valid" and theirs:
            return {t: float(ours[t]) - float(theirs[t]) for t in range(C.N_TASKS) if ours[t] is not None and theirs[t] is not None}
        return {}

    spearman = {}
    for label, epoch, split in (("valid_best", best_epoch, "valid"), ("valid_40", 40, "valid")):
        deltas = _delta_vs_hist(epoch, split)
        if deltas:
            support_v = data["task_support"][(epoch, "valid")]["positive"]
            spearman[label] = _spearman([float(support_v[t]) for t in deltas], list(deltas.values()))
    if hist_test.get("available") and hist_test.get("best_epoch_task_ap"):
        ours = data["task_ap"][(best_epoch, "test")]
        theirs = hist_test["best_epoch_task_ap"]
        deltas = {
            t: float(ours[t]) - float(theirs[t])
            for t in range(C.N_TASKS)
            if ours[t] is not None and theirs[t] is not None
        }
        support_t = data["task_support"][(best_epoch, "test")]["positive"]
        spearman["test_best"] = _spearman([float(support_t[t]) for t in deltas], list(deltas.values()))

    delta_best = s_best["valid"]["ap"] - C.AP_W3_BEST
    delta_40 = s_40["valid"]["ap"] - C.AP_W3_40
    delta_best_w2 = s_best["valid"]["ap"] - C.AP_W2_BEST
    delta_40_w2 = s_40["valid"]["ap"] - C.AP_W2_40
    test_delta_best = (
        s_best["test"]["ap"] - float(hist_test["test_ap_best"])
        if hist_test.get("available") and hist_test.get("test_ap_best") is not None
        else None
    )
    test_delta_40 = (
        s_40["test"]["ap"] - float(hist_test["test_ap_epoch40"])
        if hist_test.get("available") and hist_test.get("test_ap_epoch40") is not None
        else None
    )

    # persistence of the peak: epochs strictly above the historical best
    above = [e for e in epochs if series["valid"][e]["ap"] > C.AP_W3_BEST]
    runs: list[tuple[int, int]] = []
    for e in above:
        if runs and e == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((e, e))
    longest = max((b - a + 1 for a, b in runs), default=0)

    train_drop = s_best["train"]["ap"] - C.TRAIN_AP_W3_40
    train_drop_40 = s_40["train"]["ap"] - C.TRAIN_AP_W3_40

    label, reasons = decide(
        delta_best=delta_best,
        delta_40=delta_40,
        test_delta_best=test_delta_best,
        train_drop_40=train_drop_40,
        strat_valid_best=strat_valid_best,
        longest_above=longest,
    )

    return {
        "run": run_key,
        "run_dir": C.track_rel(data["run_dir"]),
        "status": run["status"],
        "parameters": int(run["parameters"]),
        "dropout": float(run["dropout"]),
        "weight_decay": float(run["weight_decay"]),
        "best_epoch": int(best_epoch),
        "best_valid_ap": float(data["best_valid_ap"]),
        "best_epoch_ap": {
            "train": s_best["train"]["ap"],
            "valid": s_best["valid"]["ap"],
            "test": s_best["test"]["ap"],
        },
        "best_epoch_bce": {"train": s_best["train"]["bce"], "valid": s_best["valid"]["bce"], "test": s_best["test"]["bce"]},
        "epoch40_ap": {"train": s_40["train"]["ap"], "valid": s_40["valid"]["ap"], "test": s_40["test"]["ap"]},
        "epoch40_bce": {"valid": s_40["valid"]["bce"], "test": s_40["test"]["bce"]},
        "report_epochs_ap": at_report_epochs,
        "slope_20_40": slopes,
        "delta_vs_w3": {"best": delta_best, "ap40": delta_40, "test_best": test_delta_best, "test_40": test_delta_40},
        "delta_vs_w2": {"best": delta_best_w2, "ap40": delta_40_w2},
        "train_ap_change_vs_w3_40": {"at_best": train_drop, "at_40": train_drop_40},
        "gap": {
            "own_scorable_at_best": gap_own_best,
            "common_scorable_at_best": gap_common_best,
            "at_epoch40": gap_at_40,
        },
        "peak_persistence": {
            "epochs_above_historical_best": above,
            "n_epochs_above": len(above),
            "longest_consecutive": longest,
            "first_epoch_above": above[0] if above else None,
            "last_epoch_above": above[-1] if above else None,
        },
        "per_task": {
            "valid_best": {
                "n_scored": sum(1 for x in valid_ours if x is not None),
                "mean_delta_vs_w3": _mean(
                    [float(valid_ours[t]) - float(hist_valid_best[t]) for t in range(C.N_TASKS)
                     if valid_ours[t] is not None and hist_valid_best and hist_valid_best[t] is not None]
                )
                if hist_valid_best
                else None,
                "strata": strat_valid_best,
                "concentration": _top_concentration(
                    {
                        t: float(valid_ours[t]) - float(hist_valid_best[t])
                        for t in range(C.N_TASKS)
                        if valid_ours[t] is not None and hist_valid_best and hist_valid_best[t] is not None
                    }
                )
                if hist_valid_best
                else None,
            },
            "valid_40": {"strata": strat_valid_40},
            "test_best": {"strata": strat_test_best},
        },
        "noise_floor": noise_floor,
        "spearman_delta_vs_support": spearman,
        "lr_budget": lr_budget(),
        "metric_cross_check": {
            "max_abs_delta_official_vs_sklearn": max(
                series[s][e]["ap_abs_delta"] for s in C.SPLITS for e in epochs
            )
        },
        "timing": {
            "gpu_hours": float(run["timing"]["gpu_hours"]),
            "train_seconds": float(run["timing"]["train_seconds"]),
            "eval_seconds_total": float(run["timing"]["eval_seconds_total"]),
            "per_split": run["timing"]["per_split"],
            "epoch_seconds_mean": float(run["timing"]["epoch_seconds_mean"]),
            "steps_per_second_mean": float(run["timing"]["steps_per_second_mean"]),
            "peak_gpu_allocated_bytes": int(run["resources"]["gpu_peak_allocated_bytes"]),
            "peak_rss_kb": int(run["resources"]["peak_rss_kb"]),
        },
        "eval_integrity_all_epochs": bool(run["eval_state_integrity_all_epochs"]),
        "checkpoints": {
            "best": run["best_checkpoint"],
            "best_sha256": run["best_checkpoint_sha256"],
            "epoch40": run["final_checkpoint"],
            "epoch40_sha256": run["final_checkpoint_sha256"],
            "last_sha256": run["last_checkpoint_sha256"],
        },
        "label": label,
        "label_reasons": reasons,
    }


def decide(
    *,
    delta_best: float,
    delta_40: float,
    test_delta_best: float | None,
    train_drop_40: float,
    strat_valid_best: Mapping[str, Any] | None,
    longest_above: int,
) -> tuple[str, list[str]]:
    """The PROTOCOL section 10 labels, operationalised with the pinned numbers."""
    reasons: list[str] = []
    reasons.append(f"d_best {delta_best:+.4f}, d_ap40 {delta_40:+.4f}")
    broad = False
    if strat_valid_best:
        ge20 = strat_valid_best["support_ge20"]
        broad = bool(ge20["mean_delta"] > 0 and ge20["improved"] > ge20["n"] / 2)
        reasons.append(
            f"support>=20: mean delta {ge20['mean_delta']:+.4f}, "
            f"{ge20['improved']}/{ge20['n']} improved"
        )
    if test_delta_best is not None:
        reasons.append(f"Test at best-Valid epoch delta vs W3 best-Valid: {test_delta_best:+.4f}")
    test_ok = True if test_delta_best is None else test_delta_best >= -0.003
    peak_threshold = 0.002

    if delta_best >= C.DECISION["best_delta_min"] and delta_40 >= C.DECISION["ap40_delta_min"] and broad and test_ok:
        return "PROMISING_SINGLE_SEED_SIGNAL", reasons + [
            "d_best >= +0.005, d_ap40 >= +0.003, gain broad on the >=20-positive stratum, "
            "Test at the best-Valid checkpoint not regressing by more than 0.003"
        ]
    if delta_best >= peak_threshold and (delta_40 <= 0.0 or longest_above < C.DECISION["peak_narrow_epochs"]):
        return "PEAK_ONLY_OR_UNSTABLE_SIGNAL", reasons + [
            f"peak is real (d_best >= +0.002) but d_ap40 <= 0 or only {longest_above} consecutive "
            "epoch(s) above the historical W3 best"
        ]
    if train_drop_40 <= C.DECISION["train_ap_suppression_delta"] and delta_40 < 0.002:
        return "FIT_SUPPRESSION_WITHOUT_GENERALIZATION_GAIN", reasons + [
            f"Train AP@40 is {train_drop_40:+.4f} below historical W3 with no Validation gain"
        ]
    if abs(delta_best) < peak_threshold and abs(delta_40) < peak_threshold:
        return "NO_CLEAR_GAIN", reasons + [
            "both Validation differences are inside +/-0.002"
        ]
    return "MIXED_SIGNAL", reasons + [
        "no single pre-registered condition applies; gains and losses offset or Valid/Test disagree"
    ]


def build_curves(runs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, data in runs.items():
        for epoch in sorted(data["series"]["train"]):
            row: dict[str, Any] = {"run": key, "epoch": int(epoch)}
            for split in C.SPLITS:
                row[f"{split}_ap"] = data["series"][split][epoch]["ap"]
                row[f"{split}_bce"] = data["series"][split][epoch]["bce"]
            rows.append(row)
    for epoch in sorted(C.HISTORICAL["W3"]["valid_curve"]):
        rows.append(
            {
                "run": "W3_historical",
                "epoch": int(epoch),
                "valid_ap": C.HISTORICAL["W3"]["valid_curve"][epoch],
                "valid_bce": C.HISTORICAL["W3"]["valid_bce_curve"][epoch],
            }
        )
    return rows


def build_task_deltas(runs: Mapping[str, Mapping[str, Any]], hist_test: Mapping[str, Any]) -> list[dict[str, Any]]:
    hist_valid_best = historical_valid_task_ap(C.HISTORICAL["W3"]["best_epoch"])
    hist_valid_40 = historical_valid_task_ap(40)
    hist_test_best = hist_test.get("best_epoch_task_ap") if hist_test.get("available") else None
    rows: list[dict[str, Any]] = []
    for key, data in runs.items():
        best = data["best_epoch"]
        support = data["task_support"][(best, "valid")]["positive"]
        test_support = data["task_support"][(best, "test")]["positive"]
        for t in range(C.N_TASKS):
            row: dict[str, Any] = {
                "run": key,
                "task": t,
                "valid_positive": int(support[t]),
                "test_positive": int(test_support[t]),
                "stratum": (
                    "support_ge20"
                    if support[t] >= 20
                    else "support_5_19"
                    if support[t] >= 5
                    else "support_lt5"
                ),
                "valid_ap_best": data["task_ap"][(best, "valid")][t],
                "valid_ap_40": data["task_ap"][(40, "valid")][t],
                "train_ap_best": data["task_ap"][(best, "train")][t],
                "test_ap_best": data["task_ap"][(best, "test")][t],
                "w3_valid_ap_best": hist_valid_best[t] if hist_valid_best else None,
                "w3_valid_ap_40": hist_valid_40[t] if hist_valid_40 else None,
                "w3_test_ap_best": hist_test_best[t] if hist_test_best else None,
            }
            for split in ("valid_best", "valid_40", "test_best"):
                ref = {"valid_best": "w3_valid_ap_best", "valid_40": "w3_valid_ap_40", "test_best": "w3_test_ap_best"}[
                    split
                ]
                mine = {
                    "valid_best": "valid_ap_best",
                    "valid_40": "valid_ap_40",
                    "test_best": "test_ap_best",
                }[split]
                if row[mine] is not None and row[ref] is not None:
                    row[f"delta_{split}"] = float(row[mine]) - float(row[ref])
                else:
                    row[f"delta_{split}"] = None
            rows.append(row)
    return rows


def make_figures(runs: Mapping[str, Mapping[str, Any]], figures_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return [f"matplotlib unavailable: {exc}"]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    colors = {"W3_WD10": "#1f77b4", "W3_DO20": "#d62728", "W3_historical": "#7f7f7f"}
    for split in ("valid", "test", "train"):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for key, data in runs.items():
            xs = sorted(data["series"][split])
            ax.plot(xs, [data["series"][split][e]["ap"] for e in xs], label=key, color=colors.get(key))
            ax.scatter([data["best_epoch"]], [data["series"][split][data["best_epoch"]]["ap"]], s=18,
                       color=colors.get(key), zorder=3)
        if split == "valid":
            xs = sorted(C.HISTORICAL["W3"]["valid_curve"])
            ax.plot(xs, [C.HISTORICAL["W3"]["valid_curve"][e] for e in xs], "--", label="W3 historical",
                    color=colors["W3_historical"])
            ax.axhline(C.AP_W3_BEST, ls=":", lw=1, color=colors["W3_historical"])
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{split} macro AP (official OGB)")
        ax.set_title(f"LCRC-W3 regularization diagnostic - {split} AP")
        ax.grid(alpha=0.25)
        ax.legend()
        path = Path(figures_dir) / f"{split}_ap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(C.track_rel(path))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for key, data in runs.items():
        xs = sorted(data["series"]["train"])
        ax.plot(xs, [data["series"]["train"][e]["ap"] - data["series"]["valid"][e]["ap"] for e in xs],
                label=f"{key} train-valid", color=colors.get(key))
    ax.set_xlabel("epoch")
    ax.set_ylabel("Train AP - Valid AP")
    ax.set_title("Train-Valid gap (own scorable-task macro AP)")
    ax.grid(alpha=0.25)
    ax.legend()
    path = Path(figures_dir) / "train_valid_gap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(C.track_rel(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=",".join(C.RUN_ORDER))
    parser.add_argument("--out-dir", default=str(ANALYSIS))
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_keys = [k.strip() for k in str(args.runs).split(",") if k.strip()]
    hist_test = load_historical_test()
    runs: dict[str, Any] = {}
    for key in run_keys:
        C.log(f"[compare] loading {key}")
        runs[key] = load_run(key)

    analysis = {key: analyse_run(key, hist_test) for key in run_keys}
    curves = build_curves(runs)
    per_task = build_task_deltas(runs, hist_test)
    figures = make_figures(runs, out / "figures")

    hist_w3 = {
        "parameters": C.HISTORICAL["W3"]["parameters"],
        "best_epoch": C.HISTORICAL["W3"]["best_epoch"],
        "best_valid_ap": C.HISTORICAL["W3"]["best_valid_ap"],
        "train_ap_at_best": None,
        "train_ap_40": C.HISTORICAL["W3"]["train_ap40"],
        "valid_ap_40": C.HISTORICAL["W3"]["valid_ap40"],
        "test_ap_at_best": hist_test.get("test_ap_best"),
        "test_ap_40": hist_test.get("test_ap_epoch40"),
        "gpu_hours": 1.3880,
    }
    hist_w2 = {
        "parameters": C.HISTORICAL["W2"]["parameters"],
        "best_epoch": C.HISTORICAL["W2"]["best_epoch"],
        "best_valid_ap": C.HISTORICAL["W2"]["best_valid_ap"],
        "train_ap_40": C.HISTORICAL["W2"]["train_ap40"],
        "valid_ap_40": C.HISTORICAL["W2"]["valid_ap40"],
        "test_ap_at_best": C.HISTORICAL["W2"]["test_ap_best"],
        "test_ap_40": C.HISTORICAL["W2"]["test_ap40"],
        "gpu_hours": 1.3949,
    }

    comparison_rows: list[dict[str, Any]] = []
    for model, kind, payload in (
        ("W2_historical", "historical", hist_w2),
        ("W3_historical", "historical", hist_w3),
    ):
        comparison_rows.append(
            {
                "model": model,
                "kind": kind,
                "params": payload["parameters"],
                "dropout": 0.10,
                "weight_decay": 0.001,
                "best_epoch": payload["best_epoch"],
                "best_valid_ap": payload["best_valid_ap"],
                "train_ap_at_best": None,
                "test_ap_at_best": payload["test_ap_at_best"],
                "train_ap_40": payload["train_ap_40"],
                "valid_ap_40": payload["valid_ap_40"],
                "test_ap_40": payload["test_ap_40"],
                "d_best_vs_w3": payload["best_valid_ap"] - C.AP_W3_BEST,
                "d_ap40_vs_w3": payload["valid_ap_40"] - C.AP_W3_40,
                "d_best_vs_w2": payload["best_valid_ap"] - C.AP_W2_BEST,
                "d_ap40_vs_w2": payload["valid_ap_40"] - C.AP_W2_40,
                "gpu_hours": payload["gpu_hours"],
                "label": "HISTORICAL_READ_ONLY",
            }
        )
    for key in run_keys:
        a = analysis[key]
        comparison_rows.append(
            {
                "model": key,
                "kind": "this_round",
                "params": a["parameters"],
                "dropout": a["dropout"],
                "weight_decay": a["weight_decay"],
                "best_epoch": a["best_epoch"],
                "best_valid_ap": a["best_valid_ap"],
                "train_ap_at_best": a["best_epoch_ap"]["train"],
                "test_ap_at_best": a["best_epoch_ap"]["test"],
                "train_ap_40": a["epoch40_ap"]["train"],
                "valid_ap_40": a["epoch40_ap"]["valid"],
                "test_ap_40": a["epoch40_ap"]["test"],
                "d_best_vs_w3": a["delta_vs_w3"]["best"],
                "d_ap40_vs_w3": a["delta_vs_w3"]["ap40"],
                "d_best_vs_w2": a["delta_vs_w2"]["best"],
                "d_ap40_vs_w2": a["delta_vs_w2"]["ap40"],
                "gpu_hours": a["timing"]["gpu_hours"],
                "label": a["label"],
            }
        )

    historical_noise: dict[str, Any] = {"available": False}
    hist_run_path = C.BS_RUNS / "W3" / "run.json"
    if hist_run_path.is_file():
        hist_run = C.read_json(hist_run_path)
        w3 = hist_run.get("fixed_epoch_reports") or {}
        vals = [float(w3[str(e)]["ap"]) for e in range(31, 41) if str(e) in w3]
        if vals:
            historical_noise = {
                "available": True,
                "source": C.track_rel(hist_run_path),
                "valid_last10_mean": float(np.mean(vals)),
                "valid_last10_std": float(np.std(vals)),
                "valid_last10_range": float(max(vals) - min(vals)),
            }

    comparison = {
        "schema": "lcrc-pcba-w3-regularization-comparison-v1",
        "round_id": C.ROUND_ID,
        "captured_utc": C.utcnow(),
        "primary_metric": "official OGB Validation macro AP",
        "primary_checkpoint": "max Validation AP, ties -> earlier epoch",
        "test_policy": "EXPOSED_TEST_EXPLORATORY (never used for selection)",
        "historical_reference": {
            "W3": {k: v for k, v in C.HISTORICAL["W3"].items() if k not in ("best_checkpoint", "epoch40_checkpoint")},
            "W2": dict(C.HISTORICAL["W2"]),
            "historical_test": {
                "source": hist_test.get("path"),
                "available": hist_test.get("available"),
                "W3_test_ap_best_valid": hist_test.get("test_ap_best"),
                "W3_test_ap_epoch40": hist_test.get("test_ap_epoch40"),
                "valid_replay_alignment": hist_test.get("valid_alignment"),
            },
        },
        "thresholds": dict(C.DECISION),
        "historical_noise_floor": historical_noise,
        "runs": analysis,
        "table": comparison_rows,
        "curves": C.track_rel(out / "curves_epoch.csv"),
        "per_task": C.track_rel(out / "per_task_delta.csv"),
        "figures": figures,
    }
    C.write_json(out / "comparison.json", comparison)
    _write_csv(out / "comparison.csv", COMPARISON_FIELDS, comparison_rows)
    _write_csv(
        out / "curves_epoch.csv",
        ["run", "epoch", "train_ap", "train_bce", "valid_ap", "valid_bce", "test_ap", "test_bce"],
        curves,
    )
    _write_csv(
        out / "per_task_delta.csv",
        [
            "run",
            "task",
            "valid_positive",
            "test_positive",
            "stratum",
            "valid_ap_best",
            "valid_ap_40",
            "train_ap_best",
            "test_ap_best",
            "w3_valid_ap_best",
            "w3_valid_ap_40",
            "w3_test_ap_best",
            "delta_valid_best",
            "delta_valid_40",
            "delta_test_best",
        ],
        per_task,
    )
    decision = {
        "schema": "lcrc-pcba-w3-regularization-decision-v1",
        "round_id": C.ROUND_ID,
        "captured_utc": C.utcnow(),
        "thresholds": dict(C.DECISION),
        "runs": {
            key: {
                "label": analysis[key]["label"],
                "reasons": analysis[key]["label_reasons"],
                "d_best_vs_w3": analysis[key]["delta_vs_w3"]["best"],
                "d_ap40_vs_w3": analysis[key]["delta_vs_w3"]["ap40"],
                "test_delta_best": analysis[key]["delta_vs_w3"]["test_best"],
                "peak_persistence": analysis[key]["peak_persistence"],
                "train_ap_at_best": analysis[key]["best_epoch_ap"]["train"],
                "train_ap_40": analysis[key]["epoch40_ap"]["train"],
            }
            for key in run_keys
        },
        "statistical_claim": "none; single seed, thresholds are direction/resource decisions",
    }
    C.write_json(out / "decision.json", decision)
    C.log(f"[compare] wrote {C.track_rel(out / 'comparison.json')}")
    for key in run_keys:
        a = analysis[key]
        C.log(
            f"[compare] {key}: best e{a['best_epoch']} valid {a['best_valid_ap']:.6f} "
            f"(d {a['delta_vs_w3']['best']:+.6f}) @40 {a['epoch40_ap']['valid']:.6f} "
            f"(d {a['delta_vs_w3']['ap40']:+.6f}) train@40 {a['epoch40_ap']['train']:.6f} "
            f"test@best {a['best_epoch_ap']['test']:.6f} -> {a['label']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
