"""Curve / main-table analysis for the DBT-PA short run.

Historical DBT curve is used only as a *short-horizon progress reference*.  Its
cosine schedule is different (1,500 epochs), so no row here is a causal adapter
effect estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

ANALYSIS = _EXP / "analysis"
RUN = _EXP / "runs" / "pa_s0"
DBT_ANALYSIS = _SSUF / "zinc_directed_bond_transformer_v1" / "analysis"


def _curve(path: Path) -> list[dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _at_epoch(curve: list[dict[str, float]], epoch: int) -> dict[str, float]:
    for row in curve:
        if int(row["epoch"]) == int(epoch):
            return row
    raise KeyError(f"epoch {epoch} not in curve")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(RUN))
    args = p.parse_args(argv)
    run = Path(args.run)
    info = C.read_json(run / "train.json")
    test = C.read_json(ANALYSIS / "test_report.json")
    curve = _curve(run / "curve.csv")
    dbt_curve = _curve(DBT_ANALYSIS / "curve.csv")
    E = int(info["recipe"]["epochs"])
    best = info["best_valid"]
    best_epoch = int(best["epoch"])
    valid = np.asarray([r["valid_mae"] for r in curve])
    online = np.asarray([r["online_train_mae"] for r in curve])
    dbt_valid = np.asarray([r["valid_mae"] for r in dbt_curve])
    dbt_same = _at_epoch(dbt_curve, E)
    dbt_prefix_best_i = int(np.argmin(dbt_valid[:E]))
    dbt_prefix_best = float(dbt_valid[:E].min())
    lo = max(0, best_epoch - 20)
    hi = min(len(valid), best_epoch + 20)
    observations: dict[str, Any] = {
        "new": {
            "E": E,
            "total_updates": int(info["recipe"]["total_updates"]),
            "best_valid": float(best["valid_mae"]),
            "best_epoch": best_epoch,
            "best_global_step": int(best["global_step"]),
            "eval_train_at_best": float(info["fixed_objects"]["best"]["train_mae_eval_mode"]),
            "final_valid": float(info["fixed_objects"]["final"]["valid_mae"]),
            "final_train_eval": float(info["fixed_objects"]["final"]["train_mae_eval_mode"]),
            "test_mae": float(test["test"]["mae"]),
            "valid_mae_last_20_mean": float(valid[-20:].mean()),
            "valid_mae_last_20_min": float(valid[-20:].min()),
            "valid_mae_last_50_mean": float(valid[-50:].mean()),
            "valid_mae_best_pm20_mean": float(valid[lo:hi].mean()),
            "online_train_last": float(online[-1]),
            "wall_seconds": float(info["wall_seconds"]),
            "gpu_hours": info["gpu_hours"],
            "peak_gpu_mem_mb": info["peak_gpu_mem_mb"],
        },
        "dbt_reference_different_cosine": {
            "epoch_at_same_updates": E,
            "valid_at_same_epoch": float(dbt_same["valid_mae"]),
            "online_train_at_same_epoch": float(dbt_same["online_train_mae"]),
            "best_valid_within_first_E": dbt_prefix_best,
            "best_epoch_within_first_E": dbt_prefix_best_i + 1,
            "valid_last_20_mean_within_first_E": float(dbt_valid[max(0, E - 20):E].mean()),
            "note": "1,500-epoch cosine vs E-epoch cosine; short-horizon reference only",
        },
        "delta_at_same_epoch_new_minus_dbt": float(valid[E - 1] - dbt_same["valid_mae"]),
        "delta_best_new_minus_dbt_same_epoch": float(best["valid_mae"] - dbt_same["valid_mae"]),
        "delta_best_new_minus_dbt_firstE_best": float(best["valid_mae"] - dbt_prefix_best),
        "test_contrasts": test["test_contrasts"],
        "valid_contrasts": test["valid_contrasts"],
    }
    C.write_json(ANALYSIS / "observations.json", observations)

    # Plot new curve and the DBT curve over the same first-E window.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot([r["epoch"] for r in curve], valid, label=f"DBT-PA s0 (E={E})", color="#2b6cb0")
    ax.plot([r["epoch"] for r in dbt_curve[:E]], dbt_valid[:E],
            label="DBT s0 first-E window (different cosine)", color="#a0aec0", alpha=0.9)
    ax.axvline(best_epoch, color="#2b6cb0", linestyle=":", linewidth=1)
    ax.scatter([best_epoch], [best["valid_mae"]], color="#2b6cb0", s=18, zorder=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("valid MAE (raw y)")
    ax.set_title("DBT-PA short schedule vs DBT first-E progress")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS / "curves.png", dpi=150)
    plt.close(fig)

    rows = test.get("main_table", {}).get("rows")
    if rows is None:
        rows = C.read_json(ANALYSIS / "main_table.json")["rows"]
    lines = ["| model | params | best valid | best ep | eval-train@best | test MAE | updates | status | GPU-h |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        gh = f"{r['gpu_hours']:.4f}" if r.get("gpu_hours") is not None else "—"
        ev = r.get("eval_train_at_best")
        ev_s = f"{ev:.6f}" if ev is not None else "—"
        lines.append(
            f"| {r['model']} | {r['params']:,} | {r['best_valid']:.9f} | "
            f"{r.get('best_epoch', '—')} | {ev_s} | "
            f"{r['test_mae']:.9f} | {r.get('updates', '—')} | {r['status']} | {gh} |"
        )
    (ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.log(json.dumps(observations, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
