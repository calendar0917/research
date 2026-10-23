"""Export every run's valid/train curve into one long CSV + a sampled table."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
RUNS = EXP / "runs"

SOURCES = [
    ("A-R0-s0", _SS / "zinc_recipe_trunk_v2/runs/xBssl_P3N_s0"),
    ("A-R0-s1", RUNS / "A_R0_s1"),
    ("A-R0-s2", RUNS / "A_R0_s2"),
    ("A-R0-s3", RUNS / "A_R0_s3"),
    ("A-R1-s0", RUNS / "A_R1_ema_s0"),
    ("A-R0L-s0", RUNS / "A_R0L_s0"),
    ("A-R0L-s1", RUNS / "A_R0L_s1"),
    ("B-R0-s0", _SS / "zinc_dual_channel_composer_v1/runs/L_s0"),
    ("B-R0-s1", RUNS / "B_R0_s1"),
    ("B-R0-s2", RUNS / "B_R0_s2"),
    ("B-R0-s3", RUNS / "B_R0_s3"),
    ("B-R1-s0", RUNS / "B_R1_ema_s0"),
    ("C-DBT600-s0", RUNS / "dbtc_/dbt_s0"),
    ("C-DBT600-s1", RUNS / "dbtc_/dbt_s1"),
]


def main() -> int:
    rows = []
    for name, run in SOURCES:
        info = C.read_json(run / "train.json")
        for e in info["curve"]:
            rows.append({"run": name, "epoch": int(e["epoch"]),
                         "valid_mae": float(e["valid_mae"]),
                         "train_mae": float(e.get("online_train_mae", float("nan"))),
                         "lr": float(e.get("lr", float("nan"))),
                         "seconds": float(e.get("seconds", float("nan")))})
    out = ANALYSIS / "curves_valid.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "epoch", "valid_mae", "train_mae", "lr", "seconds"])
        w.writeheader()
        w.writerows(rows)

    marks = list(range(50, 1201, 50))
    lines = ["# Valid MAE at fixed epochs (sampled from curves_valid.csv)", "",
             "| run | " + " | ".join(str(m) for m in marks) + " |",
             "|---|" + "---|" * len(marks)]
    for name, _ in SOURCES:
        vals = {r["epoch"]: r["valid_mae"] for r in rows if r["run"] == name}
        lines.append(f"| {name} | " + " | ".join(
            f"{vals[m]:.4f}" if m in vals else "-" for m in marks) + " |")
    (ANALYSIS / "curves_valid.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(rows)} rows) and curves_valid.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
