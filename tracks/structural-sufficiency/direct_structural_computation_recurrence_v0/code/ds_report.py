"""Render the DSCR v0 report tables straight from the frozen artefacts.

Writes ``analysis/report_tables.md``; the numbers in ``REPORT.md`` are pasted from
this file so that the narrative cannot drift from the artefacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import ds_common as C  # noqa: E402


def f(v: Any, nd: int = 6, sign: bool = False) -> str:
    if v is None or v == "":
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if sign:
        return f"{x:+.{nd}f}"
    return f"{x:.{nd}f}"


def main_table() -> str:
    path = C.ANALYSIS / "main_table.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = [
        "| 单位 | seed | H | best ep | Train AP | Valid AP | Test AP | Valid 高支持 AP (n) | 末 5 epoch Valid 均值 | Train−Valid | GPU·h | 来源 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        src = "本轮" if "new run" in r["source"] else "历史只读"
        out.append(
            f"| {r['model']} | {r['seed']} | {r['horizon']} | {r['best_epoch']} | {f(r['train_ap'])} | "
            f"**{f(r['valid_ap'])}** | {f(r['test_ap'])} | {f(r['valid_high_support_ap'])} ({r['valid_high_support_tasks']}) | "
            f"{f(r['last5_valid_ap_mean'])} | {f(r['train_minus_valid_ap'], 3, True)} | {f(r['gpu_hours'], 3)} | {src} |"
        )
    return "\n".join(out)


def comparisons() -> str:
    data = C.read_json(C.ANALYSIS / "paired_comparisons.json")
    comps = data["comparisons"]
    out = [
        "| 对照 (A−B) | split | 共同任务 | Δ macro AP | 任务 bootstrap 95% CI | 改善/退化 | 中位 Δ | 高支持 Δ (n) | 低支持贡献 | 去 top-3 后 Δ | 分子 bootstrap 95% CI |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, comp in comps.items():
        for split in ("valid", "test"):
            s = comp["splits"].get(split) or {}
            if not s.get("available"):
                continue
            ci = s.get("task_bootstrap", {}).get("ci95") or [None, None]
            mol = s.get("molecule_bootstrap") or {}
            mci = mol.get("ci95") or [None, None]
            hi = s.get("high_support") or {}
            lo = s.get("low_support") or {}
            out.append(
                f"| {comp['a']} − {comp['b']} | {split} | {s['n_tasks_common']} | {f(s['delta_macro_ap'], 6, True)} | "
                f"[{f(ci[0], 6, True)}, {f(ci[1], 6, True)}] | {s['n_improved']}/{s['n_degraded']} | "
                f"{f(s['median_delta'], 6, True)} | {f(hi.get('mean_delta'), 6, True)} ({hi.get('n_tasks')}) | "
                f"{f(lo.get('contribution_to_macro'), 6, True)} | {f(s['sensitivity_remove_top3']['macro_delta_remaining'], 6, True)} | "
                f"{('[' + f(mci[0], 6, True) + ', ' + f(mci[1], 6, True) + ']') if mci[0] is not None else '—'} |"
            )
    return "\n".join(out)


def mechanism() -> str:
    d = C.read_json(C.ANALYSIS / "mechanism_diagnostics.json")
    out = ["| 权重 | 读出算术 | Valid AP | Test AP | Δ Valid vs 自身 |", "|---|---|---:|---:|---:|"]
    for key, v in d["variants"].items():
        label = v["variant"] if v["gamma"] is None else f"{v['variant']} (gamma={v['gamma']})"
        out.append(
            f"| {key.split('_')[0]} | {label} | {f(v['splits']['valid']['ap'])} | "
            f"{f(v['splits']['test']['ap'])} | {f(v.get('valid_delta_vs_own'), 6, True)} |"
        )
    return "\n".join(out)


def routes() -> str:
    s = C.read_json(C.ANALYSIS / "summary.json")
    out = ["| 路线 | 机制角色 | 支持证据 | 反对证据 | 未决问题 | 判定标签 |", "|---|---|---|---|---|---|"]
    for row in s["route_table"]:
        out.append(
            f"| {row['route']} | {row['role']} | {'; '.join(row['evidence_for'])} | "
            f"{'; '.join(row['evidence_against'])} | {'; '.join(row['open'])} | {row['priority']} |"
        )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS / "report_tables.md"))
    args = parser.parse_args(argv)
    s = C.read_json(C.ANALYSIS / "summary.json")
    parts = [
        "# DSCR v0 报告表（由 artefacts 直接渲染，勿手工编辑）",
        "",
        f"- round_id: `{s['round_id']}`  H = {s['horizon']}  C\\* = {s['candidate_selection']['C*']}  K\\* = {s['candidate_selection']['K*']}",
        f"- 判定标签: `{json.dumps(s['decision']['labels'], ensure_ascii=False)}`",
        "",
        "## 表 1 · 单位主表",
        "",
        main_table(),
        "",
        "## 表 2 · 预注册配对对照",
        "",
        comparisons(),
        "",
        "## 表 3 · 冻结权重机制诊断",
        "",
        mechanism(),
        "",
        "## 表 4 · 路线判定表",
        "",
        routes(),
        "",
    ]
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    C.log(f"[report] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
