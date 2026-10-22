"""Local report assembly for zinc-branching-fragment-v1 (reads run artifacts)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
RUN = EXP / "runs" / "bf_s0"

REFERENCES = {
    "P3B_s0": {"params": 413505, "valid": 0.07622375367576023, "test": 0.069864841,
               "epoch": 377, "updates": None, "gpu_hours": None, "status": "reused"},
    "L_s0": {"params": 494193, "valid": 0.078537152, "test": 0.066266555,
             "epoch": None, "updates": None, "gpu_hours": None, "status": "reused"},
    "DBT_s0": {"params": 452523, "valid": 0.077768402, "test": 0.069401494,
               "epoch": 949, "updates": 235500, "gpu_hours": 3.6918, "status": "reused"},
    "DBT-PA_s0": {"params": 457443, "valid": 0.140224125, "test": 0.107658585,
                  "epoch": 287, "updates": 48670, "gpu_hours": 0.7257, "status": "reused"},
}


def _csv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def curve_enriched(train: dict) -> list[dict]:
    rows = []
    for entry in train["curve"]:
        rows.append({
            "epoch": int(entry["epoch"]),
            "global_step": int(entry["epoch"]) * int(train["recipe"]["steps_per_epoch"]),
            "online_train_mae": entry["online_train_mae"],
            "valid_mae": entry["valid_mae"],
            "lr_inherited": entry["lr_inherited"],
            "lr_new": entry["lr_new"],
            "seconds": entry["seconds"],
            "reference_valid": entry["reference_valid"],
            "patience": entry["patience"],
            "is_best": bool(entry["is_best"]),
            "early_stop": bool(entry["early_stop"]),
        })
    return rows


def write_curves(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = np.asarray([r["epoch"] for r in rows])
    valid = np.asarray([r["valid_mae"] for r in rows], dtype=np.float64)
    online = np.asarray([np.nan if r["online_train_mae"] is None else float(r["online_train_mae"])
                         for r in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axes[0].plot(ep, valid, color="#1f77b4", lw=1.4, label="valid MAE")
    axes[0].axhline(valid[0], color="black", ls="--", lw=1.0,
                    label=f"epoch 0 (P3B) {valid[0]:.6f}")
    best_ep = int(np.argmin(valid))
    axes[0].scatter([ep[best_ep]], [valid[best_ep]], color="red", zorder=3,
                    label=f"best @{int(ep[best_ep])} {valid[best_ep]:.6f}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("valid MAE (raw y)")
    axes[0].set_title("P3-BF valid")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].plot(ep[1:], online[1:], color="#d62728", lw=1.4, label="online train MAE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("train MAE (raw y)")
    axes[1].set_title("P3-BF online train")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ANALYSIS / "curves.png", dpi=150)
    plt.close(fig)


def main() -> int:
    train = C.read_json(RUN / "train.json")
    timing = C.read_json(ANALYSIS / "timing.json")
    plan = C.read_json(ANALYSIS / "plan.json")
    freeze_doc = C.read_json(ANALYSIS / "freeze.json")
    test = C.read_json(ANALYSIS / "test_report.json")
    groups = C.read_json(ANALYSIS / "groups_rows.json")
    rep = C.read_json(ANALYSIS / "representation_check.json")
    checks = C.read_json(ANALYSIS / "cpu_checks.json")

    rows = curve_enriched(train)
    C.write_curve_csv(ANALYSIS / "curve_enriched.csv", rows)
    write_curves(rows)

    best = train["scored"]["best"]
    final = train["scored"]["final"]
    model_row = {
        "model": "P3-BF_s0",
        "params": int(train["parameter_count"]),
        "best_valid": float(best["valid_mae"]),
        "best_epoch": int(best["epoch"]),
        "eval_train_at_best": float(best["train_mae_eval_mode"]),
        "test_mae": float(test["test"]["mae"]),
        "updates": int(train["recipe"]["actual_global_steps"]),
        "epochs_run": int(train["recipe"]["epochs"]),
        "planned_epochs": int(train["recipe"]["planned_epochs"]),
        "stopped_early": bool(train["recipe"]["early_stop"]["triggered"]),
        "gpu_hours": float(train["gpu_hours"]),
        "status": "complete (early stop by declared rule)",
    }
    table = []
    for name, ref in REFERENCES.items():
        table.append({
            "model": name, "params": ref["params"], "best_valid": ref["valid"],
            "best_epoch": ref["epoch"], "eval_train_at_best": None,
            "test_mae": ref["test"], "updates": ref["updates"],
            "epochs_run": None, "planned_epochs": None, "stopped_early": None,
            "gpu_hours": ref["gpu_hours"], "status": ref["status"],
        })
    table.append(model_row)
    C.write_json(ANALYSIS / "main_table.json", {
        "rows": table,
        "p3_bf_vs_p3b_test_delta": float(test["overall"]["contrasts"]["P3B_s0"]["mean_delta_ae"]),
        "p3_bf_vs_l_test_delta": float(test["overall"]["contrasts"]["L_s0"]["mean_delta_ae"]),
        "p3_bf_vs_dbt_test_delta": float(test["overall"]["contrasts"]["DBT_s0"]["mean_delta_ae"]),
        "p3b_function_same_epoch0": test["p3b_function_same_epoch0"],
        "final": {"epoch": int(final["epoch"]), "valid_mae": float(final["valid_mae"]),
                  "eval_train": float(final["train_mae_eval_mode"])},
    })
    lines = [
        "| model | params | best valid | ep | eval-train@best | test MAE | updates | status | GPU-h |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in table:
        ev = "—" if r["eval_train_at_best"] is None else f"{r['eval_train_at_best']:.9f}"
        gh = "—" if r["gpu_hours"] is None else f"{r['gpu_hours']:.4f}"
        lines.append(
            f"| {r['model']} | {r['params']} | {r['best_valid']:.9f} | "
            f"{'—' if r['best_epoch'] is None else r['best_epoch']} | {ev} | "
            f"{r['test_mae']:.9f} | {'—' if r['updates'] is None else r['updates']} | "
            f"{r['status']} | {gh} |"
        )
    (ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    glines = ["| split | group | n | P3-BF | P3B_s0 | L_s0 | DBT_s0 | Δ vs P3B |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("valid", "test"):
        for r in groups[split]:
            m = r["mae"]
            glines.append(
                f"| {split} | {r['group']} | {r['n']} | {m['P3-BF']:.9f} | {m['P3B_s0']:.9f} | "
                f"{m['L_s0']:.9f} | {m['DBT_s0']:.9f} | {r['delta_vs_P3B_s0']:+.3e} |"
            )
    (ANALYSIS / "groups.md").write_text(
        "# Fixed four-group raw-y MAE (label split is analysis-only; row 172 kept)\n\n"
        + "\n".join(glines) + "\n", encoding="utf-8"
    )

    summary = {
        "protocol": "zinc-branching-fragment-v1",
        "candidate": "P3-BF",
        "seed": 0,
        "git_commit_run": freeze_doc["git_commit"],
        "parameters": int(train["parameter_count"]),
        "epoch0_valid": float(train["epoch0"]["valid_mae"]),
        "best_valid": float(best["valid_mae"]),
        "best_epoch": int(best["epoch"]),
        "best_is_epoch0": bool(freeze_doc["best_is_epoch0"]),
        "best_min_new_training_valid": float(min(e["valid_mae"] for e in rows[1:])),
        "best_min_new_training_epoch": int(min(rows[1:], key=lambda e: e["valid_mae"])["epoch"]),
        "final_valid": float(final["valid_mae"]),
        "final_eval_train": float(final["train_mae_eval_mode"]),
        "test_mae": float(test["test"]["mae"]),
        "test_delta_vs_p3b": float(test["overall"]["contrasts"]["P3B_s0"]["mean_delta_ae"]),
        "test_delta_vs_l": float(test["overall"]["contrasts"]["L_s0"]["mean_delta_ae"]),
        "test_delta_vs_dbt": float(test["overall"]["contrasts"]["DBT_s0"]["mean_delta_ae"]),
        "test_ci_vs_p3b": test["overall"]["contrasts"]["P3B_s0"]["ci95"],
        "epochs_planned": int(plan["epochs"]),
        "epochs_run": int(train["recipe"]["epochs"]),
        "updates_planned": int(plan["total_updates"]),
        "updates_run": int(train["recipe"]["actual_global_steps"]),
        "early_stop": bool(train["recipe"]["early_stop"]["triggered"]),
        "steady_epoch_seconds": float(timing["steady_epoch_seconds"]),
        "formal_train_seconds": float(train["formal_train_seconds"]),
        "gpu_job_wall_seconds": float(train["wall_seconds"]),
        "gpu_hours_job": float(train["gpu_hours"]),
        "checkpoint": {
            "best_sha256": freeze_doc["best_sha256"],
            "final_sha256": train["scored"]["final_sha256"],
            "init_sha256": freeze_doc["init_checkpoint"]["sha256"],
            "best_is_p3b_function": bool(freeze_doc["best_is_epoch0"]),
            "test_prediction_max_abs_diff_vs_P3B": float(
                test["p3b_function_same_epoch0"]["max_abs_diff_vs_P3B_reference_test"]),
        },
        "representation": {
            "families": ["S3", "S4", "F4"],
            "train_occurrences": rep["train_valid_content"]["per_family"],
            "roots_total": int(rep["refinement"]["n_roots"]),
            "roots_in_multi_raw108_groups": int(rep["refinement"]["n_roots_in_multi_groups"]),
            "subdivided_groups": int(rep["refinement"]["n_groups_subdivided"]),
            "subdivided_roots": int(rep["refinement"]["n_roots_in_subdivided_groups"]),
            "fraction_roots_subdivided": float(rep["refinement"]["fraction_roots_subdivided"]),
            "fraction_molecules_subdivided": float(rep["refinement"]["fraction_molecules_subdivided"]),
            "pure_content_subdivisions": int(rep["refinement"]["pure_content_subdivisions"]),
            "pure_multiplicity_subdivisions": int(rep["refinement"]["pure_multiplicity_subdivisions"]),
            "ab_example": rep["ab_example"],
        },
        "cpu_checks_passed": all(v["pass"] for v in checks["checks"].values()),
        "cpu_check_count": len(checks["checks"]),
    }
    C.write_json(ANALYSIS / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "representation"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
