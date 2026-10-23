"""Assemble the report tables from the analysis JSONs.

Reads ``main_table.json`` (valid, all runs), ``config_summary.json`` and
``test_table.json`` (terminal read) and writes ``analysis/report_tables.md``.
No model is loaded and no split is touched; this is pure formatting.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ANALYSIS = _HERE.parent / "analysis"


def _load(name):
    p = ANALYSIS / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    main_t = _load("main_table.json") or {"records": []}
    recs = [r for r in main_t["records"] if r.get("predictions")]
    conf = _load("config_summary.json") or {}
    test = _load("test_table.json")

    lines = ["# Report tables", ""]

    # 1) discovery: runs with seed 0 or 1 only
    lines += ["## Discovery stage (seeds 0-1)", "",
              "| config | seed | params | best-valid | ep | W20 | final | train@best | d(row172) | d(top10) | gpu-h | source |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(recs, key=lambda r: (r["arm"], r["recipe"], r["seed"])):
        if r["seed"] > 1:
            continue

        def f(x, fmt="{:.6f}"):
            return "-" if x is None else fmt.format(x)
        lines.append(
            f"| {r['arm']}-{r['recipe']} | {r['seed']} | {r['params']:,} | {f(r['best_valid'])} | "
            f"{r['best_epoch']} | {f(r['w20'])} | {f(r['final_valid'])} | {f(r['train_mae_at_best'])} | "
            f"{f(r.get('mae_drop_row172'))} | {f(r.get('mae_drop_top10'))} | "
            f"{f(r['gpu_hours'], '{:.3f}')} | {r['source']} |")

    # 2) config summary (all seeds present)
    lines += ["", "## Config summary over available seeds", "",
              "| config | seeds | best-valid mean | sd | range | W20 mean | mean best epoch | GPU-h |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in sorted(conf.items(), key=lambda kv: kv[1]["mean"]):
        sd = "-" if v["sd"] is None else format(v["sd"], ".6f")
        w20 = "-" if v["w20_mean"] is None else format(v["w20_mean"], ".6f")
        gpu = "-" if v["gpu_hours"] is None else format(v["gpu_hours"], ".3f")
        seeds = ",".join(str(s) for s in v["seeds"])
        lines.append(f"| {k} | {v['n']} ({seeds}) | {v['mean']:.6f} | {sd} | "
                     f"[{v['range'][0]:.6f}, {v['range'][1]:.6f}] | {w20} | "
                     f"{v['mean_epoch']:.0f} | {gpu} |")

    # 3) frozen groups (valid)
    order = ["all_1000", "seen_749", "unseen_251", "ordinary_935", "tail_65",
             "unseen_ordinary_225", "seen_ordinary_710", "unseen_tail_26"]
    lines += ["", "## Valid MAE on frozen groups", "",
              "| run | " + " | ".join(order) + " |",
              "|---|" + "---|" * len(order)]
    for r in sorted(recs, key=lambda r: (r["arm"], r["recipe"], r["seed"])):
        g = r.get("groups") or {}
        lines.append(f"| {r['arm']}-{r['recipe']} s{r['seed']} | "
                     + " | ".join(f"{g[k]['mae']:.5f}" if k in g else "-" for k in order) + " |")

    # 4) terminal test
    if test:
        lines += ["", "## Terminal official test (frozen shortlist)", "",
                  "| model | seed | policy | epoch | valid MAE | test MAE | RMSE | p90 AE | drop row172 | drop top10 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for d in test["test"]:
            lines.append(
                f"| {d['name']} | {d['seed']} | {d['policy']} | {d['epoch']} | {d['valid_mae']:.6f} | "
                f"{d['mae']:.6f} | {d['rmse']:.4f} | {d['p90_ae']:.4f} | "
                f"{d['mae_drop_row172']:.6f} | {d['mae_drop_top10']:.6f} |")
        by = {}
        for d in test["test"]:
            by.setdefault(d["name"], []).append(d["mae"])
        lines += ["", "| model family | seeds | test mean | test range |", "|---|---:|---:|---:|"]
        for k, v in sorted(by.items()):
            lines.append(f"| {k} | {len(v)} | {sum(v)/len(v):.6f} | [{min(v):.6f}, {max(v):.6f}] |")
        if test.get("bootstrap"):
            lines += ["", "| a | b | delta test MAE (a-b) | 95% CI | P(a better) |", "|---|---|---:|---|---:|"]
            for b in test["bootstrap"]:
                lines.append(f"| {b['a']} | {b['b']} | {b['delta_mae']:+.6f} | "
                             f"[{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] | {b['p_a_better']:.3f} |")

    out = ANALYSIS / "report_tables.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
