"""Machine-generated tables and the headline numbers of the round.

Writes ``analysis/tables/*.md`` and ``analysis/summary.json``.  The prose verdict
in ``REPORT.md`` / ``DECISION.md`` is written by the operator and must agree with
``analysis/summary.json``; nothing here decides anything by itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import bc_common as C

ANCHORS = {
    "lipo": {"C1": (0.7069, 0.7293), "GINE": (0.6161, 0.6800)},
    "wc_logp": {"C1": (0.7958, 0.9534), "GINE": (0.8587, 0.8902)},
}


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], header: Sequence[str] | None = None) -> str:
    head = list(header or columns)
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(row.get(c), 4) if not isinstance(row.get(c), str) else str(row.get(c)) for c in columns) + " |")
    return "\n".join(out) + "\n"


def build() -> dict[str, Any]:
    tbl = C.ANALYSIS_DIR / "tables"
    tbl.mkdir(parents=True, exist_ok=True)
    util = C.read_json(C.ANALYSIS_DIR / "branch_utilization.json")["units"]
    comps = C.read_json(C.ANALYSIS_DIR / "paired_comparisons.json")["comparisons"]
    decisions = C.read_json(C.ANALYSIS_DIR / "stage_decisions.json")
    ledger = C.read_json(C.ANALYSIS_DIR / "compute_ledger.json")

    units_md = _table(
        util,
        ["unit", "institution", "task", "variant", "seed", "selected_epoch", "epochs_run",
         "rmse_train", "rmse_valid", "rmse_test", "rmse_base_valid", "mean_abs_delta_raw_valid"],
        ["unit", "inst", "task", "variant", "seed", "epoch*", "epochs", "Train RMSE",
         "Valid RMSE", "Test RMSE", "Valid RMSE (branch-off)", "mean |Δ| raw"],
    )
    (tbl / "units.md").write_text(units_md)

    for split in ("valid", "test"):
        rows = []
        for c in comps:
            e = c.get(split)
            if e is None:
                continue
            rows.append(
                {
                    "unit": f"{c['institution']}/{c['task']}/s{c['seed']}",
                    "rmse_joint": e["rmse_joint"],
                    "rmse_marginal": e["rmse_marginal"],
                    "rmse_c1": e.get("rmse_c1"),
                    "rmse_gine": e.get("rmse_gine"),
                    "d_marginal": e["joint_vs_marginal"]["delta"],
                    "ci_marginal": f"[{e['joint_vs_marginal']['ci95'][0]:+.4f},{e['joint_vs_marginal']['ci95'][1]:+.4f}]",
                    "p_marginal": e["joint_vs_marginal"]["prob_a_better"],
                    "d_c1": None if "joint_vs_c1" not in e else e["joint_vs_c1"]["delta"],
                    "p_c1": None if "joint_vs_c1" not in e else e["joint_vs_c1"]["prob_a_better"],
                    "top1pct": e["concentration"].get("top_1pct_share_of_abs_change"),
                    "frac_better": e["concentration"]["fraction_molecules_better"],
                    "clean": "yes" if e["degenerate"]["clean_pairing"] else (
                        "control dead" if e["degenerate"]["marginal_degenerate"] else "joint dead"),
                }
            )
        (tbl / f"paired_{split}.md").write_text(
            _table(rows,
                   ["unit", "rmse_joint", "rmse_marginal", "rmse_c1", "rmse_gine", "d_marginal",
                    "ci_marginal", "p_marginal", "d_c1", "p_c1", "top1pct", "frac_better", "clean"],
                   ["unit", "JOINT", "MARGINAL", "C1 anchor", "GINE anchor",
                    "Δ(J vs M)", "95% CI", "P(J<M)", "Δ(J vs C1)", "P(J<C1)",
                    "top1% share", "frac better", "pairing"])
            + "\nΔ = RMSE(MARGINAL) - RMSE(JOINT): **positive means JOINT is better**.\n"
              "P(J<M) is the bootstrap probability that JOINT has the lower RMSE.\n")

    summary: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "units": [
            {
                "unit": u["unit"], "institution": u["institution"], "task": u["task"],
                "variant": u["variant"], "seed": u["seed"],
                "rmse_train": u["rmse_train"], "rmse_valid": u["rmse_valid"], "rmse_test": u["rmse_test"],
                "rmse_base_valid": u["rmse_base_valid"], "epochs_run": u["epochs_run"],
                "selected_epoch": u["selected_epoch"],
                "branch_off_valid_max_abs_diff": u["branch_off"]["max_abs_diff_from_base"],
                "delta_over_teacher_std_valid": u["delta_over_teacher_std_valid"],
            }
            for u in util
        ],
        "anchors": ANCHORS,
        "stage_decisions": decisions,
        "compute": {"total_gpu_hours": ledger["total_gpu_hours"],
                    "total_elapsed_hours": ledger["total_elapsed_hours"],
                    "units": len(ledger["units"])},
    }
    C.atomic_write_json(C.ANALYSIS_DIR / "summary.json", summary)
    return summary


def main() -> None:
    s = build()
    print(json.dumps(s["compute"], indent=2))
    print("stage C required:", s["stage_decisions"]["stage_c_required"],
          s["stage_decisions"]["stage_c_institution"])


if __name__ == "__main__":
    main()
