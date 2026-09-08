#!/usr/bin/env python3
"""汇总 results/*.json 并与论文数字并排打印。

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


def main() -> None:
    rows = []
    for p in sorted(RESULTS.glob("*__seed*.json")):
        cfg = p.name.replace("__seed", "--").replace(".json", "").rsplit("--", 1)[0]
        r = json.loads(p.read_text())
        m = r.get("metrics", {})
        rows.append((cfg, r.get("seed"), m.get("best_test_mean"), m.get("best_test_std"),
                     r.get("in_paper", True), r.get("elapsed_s")))
    if not rows:
        print("no results yet")
        return
    print(f"{'config':<22}{'seed':>5}{'ours mean':>12}{'±':>2}{'std':>7}{'paper':>13}{'note':>8}")
    for cfg, seed, mean, std, in_paper, elapsed in rows:
        exp = PAPER.get(cfg)
        paper_s = f"{exp[0]:.1f}±{exp[1]:.1f}" if exp else "n/a"
        note = "" if in_paper else "(扩展)"
        m_s = f"{mean*100:.1f}" if mean is not None else "?"
        s_s = f"{std*100:.1f}" if std is not None else "?"
        print(f"{cfg:<22}{seed:>5}{m_s:>12}{'±':>2}{s_s:>7}{paper_s:>13}{note:>8}")


if __name__ == "__main__":
    main()
