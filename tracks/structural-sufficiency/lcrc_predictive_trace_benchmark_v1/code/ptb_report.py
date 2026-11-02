"""Render the deliverable markdown tables straight from ``analysis/*.csv|json``.

Every number in ``REPORT.md`` can be regenerated with this script, so no figure is
transcribed by hand.  Output: ``analysis/report_tables.md`` (plus the same text on
stdout).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import ptb_common as C

DATASETS = list(C.MOLECULE_ORDER)
FROZEN = "Lipophilicity"


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(value: Any, digits: int = 4) -> str:
    if value in (None, "", "None"):
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def molecule_summary(mol: Sequence[Mapping[str, str]]) -> str:
    lines = []
    header = ["dataset", "metric", "A0 mean±sd", "C1 mean±sd", "Δ C1−A0", "Δ 95% CI (rows)", "n", "status"]
    body = []
    stats = {}
    stats_path = C.ANALYSIS / "statistics.json"
    if stats_path.exists():
        stats = C.read_json(stats_path).get("molecule_paired", {})
    for dataset in DATASETS:
        rows = [r for r in mol if r.get("dataset") == dataset and r.get("test_primary") not in (None, "")]
        if not rows:
            body.append([dataset, "—", "—", "—", "—", "—", 0, "NOT_RUN_BUDGET"])
            continue
        a0 = [float(r["test_primary"]) for r in rows if r["model"] == "A0"]
        c1 = [float(r["test_primary"]) for r in rows if r["model"] == "C1"]
        metric = C.MOLECULE_DATASETS[dataset]["primary_metric"]
        stat = stats.get(dataset, {})
        delta = (np.mean(c1) - np.mean(a0)) if a0 and c1 else float("nan")
        per = stat.get("per_seed", [])
        lo = min((p["boot_lo"] for p in per), default=None)
        hi = max((p["boot_hi"] for p in per), default=None)
        ci = f"[{_f(lo)}, {_f(hi)}]" if lo is not None and hi is not None else "—"
        body.append(
            [
                dataset,
                metric,
                f"{_f(np.mean(a0))} ± {_f(np.std(a0, ddof=1) if len(a0) > 1 else 0.0)}",
                f"{_f(np.mean(c1))} ± {_f(np.std(c1, ddof=1) if len(c1) > 1 else 0.0)}",
                f"{delta:+.4f}",
                ci,
                f"{len(a0)}/{len(c1)}",
                rows[0].get("result_label", ""),
            ]
        )
    lines.append(_table(header, body))
    frozen_rows = [r for r in mol if r.get("dataset") == FROZEN and r.get("status") == "FROZEN_HISTORY"]
    if frozen_rows:
        body = []
        for repeat in sorted({int(r["repeat"]) for r in frozen_rows}):
            a0 = next((r for r in frozen_rows if int(r["repeat"]) == repeat and r["model"] == "A0"), None)
            c1 = next((r for r in frozen_rows if int(r["repeat"]) == repeat and r["model"] == "C1"), None)
            if not a0 or not c1:
                continue
            body.append(
                [
                    f"repeat {repeat}",
                    "valid RMSE",
                    _f(a0["valid_primary"]),
                    _f(c1["valid_primary"]),
                    f"{float(c1['valid_primary']) - float(a0['valid_primary']):+.4f}",
                    "frozen history",
                ]
            )
            body.append(
                [
                    f"repeat {repeat}",
                    "test RMSE",
                    _f(a0["test_primary"]),
                    _f(c1["test_primary"]),
                    f"{float(c1['test_primary']) - float(a0['test_primary']):+.4f}",
                    "EXPOSED_TEST_EXPLORATORY",
                ]
            )
        lines.append("\n**Lipophilicity — read-only frozen history (no retraining)**\n")
        lines.append(_table(["unit", "metric", "A0", "C1", "Δ C1−A0", "label"], body))
    return "\n\n".join(lines)


def molecule_units(mol: Sequence[Mapping[str, str]]) -> str:
    body = []
    for dataset in DATASETS:
        for model in C.MODELS:
            for seed in C.MOLECULE_SEEDS:
                r = next(
                    (
                        x
                        for x in mol
                        if x.get("dataset") == dataset
                        and x.get("model") == model
                        and str(x.get("seed")) == str(seed)
                    ),
                    None,
                )
                if r is None:
                    body.append([dataset, model, seed, "—", "—", "—", "—", "—", "NOT_RUN_BUDGET"])
                    continue
                body.append(
                    [
                        dataset,
                        model,
                        seed,
                        r.get("best_epoch", "—"),
                        _f(r.get("valid_primary")),
                        _f(r.get("test_primary")),
                        _f(r.get("train_minus_valid_rmse") or r.get("train_minus_valid_roc_auc") or r.get("train_minus_valid_macro_roc_auc")),
                        _f(r.get("gpu_hours"), 3),
                        r.get("status", ""),
                    ]
                )
    return _table(
        ["dataset", "model", "seed", "best epoch", "valid", "test", "train−valid", "GPU-h", "status"],
        body,
    )


def tu_summary(tu: Sequence[Mapping[str, str]]) -> str:
    if not tu:
        return "_TU 结果尚未产生。_"
    body = []
    stats = C.ANALYSIS / "statistics.json"
    tu_stats = C.read_json(stats).get("tu_paired", []) if stats.exists() else []
    for dataset in C.TU_DATASETS:
        for model in C.MODELS:
            rows = [r for r in tu if r.get("dataset") == dataset and r.get("model") == model and r.get("test_primary")]
            if not rows:
                body.append([dataset, model, "—", "—", "—", "—", "NOT_RUN_BUDGET"])
                continue
            vals = [float(r["test_primary"]) for r in rows]
            body.append(
                [
                    dataset,
                    model,
                    f"{_f(np.mean(vals))} ± {_f(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)}",
                    f"{min(vals):.4f}/{max(vals):.4f}",
                    str(len(rows)),
                    _f(np.mean([float(r["best_epoch"]) for r in rows]), 1)
                    if rows[0].get("best_epoch")
                    else "—",
                    rows[0].get("result_label", ""),
                ]
            )
        stat = next((s for s in tu_stats if s.get("dataset") == dataset), None)
        if stat:
            body.append(
                [
                    f"**{dataset} Δ**",
                    "C1−A0",
                    f"{stat['delta_test_mean']:+.4f}",
                    f"[{stat['delta_test_boot_lo']:+.4f}, {stat['delta_test_boot_hi']:+.4f}]",
                    f"{stat.get('n_folds', '—')} folds",
                    "outer-fold bootstrap",
                    C.TU_RESULT_LABEL,
                ]
            )
    return _table(["dataset", "model", "test mean±sd", "min/max", "n units", "best epoch", "label"], body)


def efficiency(path: Path) -> str:
    rows = _rows(path)
    if not rows:
        return "_效率表尚未产生。_"
    body = [
        [
            r["dataset"],
            r["model"],
            r["params_trainable"],
            _f(r["param_mb_fp32"], 3),
            _f(r["epoch_seconds"], 3),
            _f(r["graphs_per_second"], 1),
            _f(r["cuda_peak_allocated_mb"], 1),
            _f(r["train_cache_mb"], 1),
            _f(r["cache_build_seconds"], 2),
            _f(r["long_tail_max_over_min"], 2),
        ]
        for r in rows
    ]
    return _table(
        [
            "dataset",
            "model",
            "trainable params",
            "param MB",
            "epoch s",
            "graphs/s",
            "CUDA peak MB",
            "cache MB",
            "cache build s",
            "long-tail ratio",
        ],
        body,
    )


def gates() -> str:
    path = C.ANALYSIS / "correctness_gates.json"
    if not path.exists():
        return "_门禁尚未产生。_"
    payload = C.read_json(path)
    rows = []
    for gate in payload.get("gates", payload if isinstance(payload, list) else []):
        rows.append([gate.get("name", "?"), "PASS" if gate.get("ok") else "FAIL", (gate.get("detail") or "")[:110]])
    return _table(["gate", "result", "detail"], rows)


def ledger() -> str:
    path = C.ANALYSIS / "compute_ledger.json"
    if not path.exists():
        return "_预算账本尚未回传。_"
    payload = C.read_json(path)
    units = payload.get("units", [])
    by_kind: dict[str, list[float]] = {}
    for u in units:
        by_kind.setdefault(str(u.get("kind", "?")), []).append(float(u.get("gpu_hours", 0.0)))
    body = [
        [kind, len(vals), f"{sum(vals):.3f}", f"{np.mean(vals):.3f}"]
        for kind, vals in sorted(by_kind.items())
    ]
    body.append(["**total**", len(units), f"**{sum(sum(v) for v in by_kind.values()):.3f}**", "—"])
    return _table(["kind", "units", "GPU-h", "mean GPU-h"], body)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(C.ANALYSIS / "report_tables.md"))
    args = ap.parse_args(argv)
    mol = _rows(C.ANALYSIS / "molecular_results.csv")
    tu = _rows(C.ANALYSIS / "tu_results.csv")
    parts = [
        "# 自动生成的表格（`code/ptb_report.py`）",
        f"生成时间 {C.utcnow()}，commit `{C.git_commit()}`。",
        "## A. 分子任务 Test 主指标（每格 3 seeds）",
        molecule_summary(mol),
        "## B. 36 个分子单元逐条",
        molecule_units(mol),
        "## C. TU 结果",
        tu_summary(tu),
        "## D. 效率",
        efficiency(C.ANALYSIS / "efficiency_table.csv"),
        "## E. 正确性门禁",
        gates(),
        "## F. GPU 账本",
        ledger(),
    ]
    text = "\n\n".join(parts) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
