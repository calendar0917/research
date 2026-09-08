#!/usr/bin/env python3
"""汇总 results/（官方协议）与 results/strict/（严格协议）结果。

用法: uv run python code/audit_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

PAPER = {
    "IMDBBINARY--gsn-e": (77.8, 3.3), "IMDBBINARY--gsn-v": (76.8, 2.0),
    "IMDBMULTI--gsn-e": (54.3, 3.3), "IMDBMULTI--gsn-v": (52.6, 3.6),
    "COLLAB--gsn-e": (85.5, 1.2), "COLLAB--gsn-v": (82.7, 1.5),
}


def show_official() -> None:
    rows = []
    for p in sorted(RESULTS.glob("*__seed*.json")):
        r = json.loads(p.read_text())
        m = r.get("metrics", {})
        rows.append((r.get("config"), r.get("seed"), m.get("best_test_mean"),
                     m.get("best_test_std"), r.get("in_paper", True)))
    if not rows:
        return
    print("== 官方乐观协议 (gsn-official-social-v1) ==")
    for cfg, seed, mean, std, in_paper in rows:
        exp = PAPER.get(cfg)
        paper_s = f"{exp[0]:.1f}±{exp[1]:.1f}" if exp else "n/a"
        note = "" if in_paper else "(扩展)"
        m_s = f"{mean*100:.1f}" if mean is not None else "?"
        s_s = f"{std*100:.1f}" if std is not None else "?"
        print(f"  {cfg:<22} seed={seed:<4} {m_s}±{s_s}   paper={paper_s} {note}")


def show_strict() -> None:
    rows = []
    for p in sorted((RESULTS / "strict").glob("*/summary.json")):
        s = json.loads(p.read_text())
        rows.append((s["config"], s["fold_level"], s["seed_level"], s["n_folds_total"]))
    if not rows:
        return
    print("== 严格协议 (gsn-strict-social-v1, fold-level mean±std over all folds) ==")
    for config, fl, sl, n in rows:
        exp = PAPER.get(config)
        paper_s = f"{exp[0]:.1f}±{exp[1]:.1f}" if exp else "n/a"
        print(f"  {config:<22} {fl['mean']*100:6.2f}±{fl['std']*100:5.2f}"
              f" (n={n:>5})  seed-level {sl['mean_of_means']*100:.2f}±"
              f"{sl['std_of_means']*100:.2f}  paper={paper_s}")


def main() -> None:
    show_official()
    show_strict()
    if not any(RESULTS.iterdir()):
        print("no results yet")


if __name__ == "__main__":
    main()
