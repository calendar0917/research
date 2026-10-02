"""Local paired analysis for the node-binding residual pair v1.

Reads the two pulled arm exports (``control/`` and ``residual/``) and produces
the paired CSV, group table, compact summary JSON and a draft REPORT.  Uses the
frozen identity-aligned group labels (G0 / G1 / G172) from the Small/Full
cycle-alignment round.  Signed error convention: ``pred - y``.  No test access.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
PAIR = HERE
GROUPS_CSV = REPO_ROOT / "tracks/ksvd/results/zinc_small_full_cycle_alignment_v1/valid_per_graph_aligned.csv"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_valid(arm_dir: Path) -> dict:
    summary = load_json(arm_dir / "summary.json")
    rows = list(csv.DictReader((arm_dir / "valid_predictions.csv").open(encoding="utf-8")))
    return {
        "summary": summary,
        "graph_id": [int(r["graph_id"]) for r in rows],
        "y": [float(r["y"]) for r in rows],
        "raw": [float(r["pred_raw"]) for r in rows],
        "cal": [float(r["pred_calibrated"]) for r in rows],
        "bias": float(rows[0]["train_fitted_bias"]),
    }


def load_groups() -> dict[int, str]:
    rows = list(csv.DictReader(GROUPS_CSV.open(encoding="utf-8")))
    return {int(r["graph_id"]): str(r["group"]) for r in rows}


def mae(pred: list[float], y: list[float]) -> float:
    return sum(abs(p - t) for p, t in zip(pred, y)) / len(y)


def main() -> int:
    control = load_valid(PAIR / "control")
    residual = load_valid(PAIR / "residual")
    groups = load_groups()

    y = control["y"]
    assert control["graph_id"] == residual["graph_id"], "arm row order mismatch"
    assert y == residual["y"], "arm target mismatch"
    n = len(y)

    group_of = [groups.get(gid, "UNKNOWN") for gid in control["graph_id"]]
    row = {"n": n}
    row["control_cal_valid_mae"] = mae(control["cal"], y)
    row["residual_cal_valid_mae"] = mae(residual["cal"], y)
    row["control_raw_valid_mae"] = mae(control["raw"], y)
    row["residual_raw_valid_mae"] = mae(residual["raw"], y)
    row["calibrated_gain"] = row["control_cal_valid_mae"] - row["residual_cal_valid_mae"]
    row["raw_gain"] = row["control_raw_valid_mae"] - row["residual_raw_valid_mae"]
    row["control_bias"] = control["bias"]
    row["residual_bias"] = residual["bias"]
    row["bias_diff"] = control["bias"] - residual["bias"]

    keep = [i for i, gid in enumerate(control["graph_id"]) if gid != 172]
    row["gain_without172"] = mae([control["cal"][i] for i in keep], [y[i] for i in keep]) - mae(
        [residual["cal"][i] for i in keep], [y[i] for i in keep]
    )

    group_table = []
    total_c = total_r = 0.0
    for group in ("G0", "G1", "G172"):
        idx = [i for i, g in enumerate(group_of) if g == group]
        if not idx:
            continue
        c_mae = mae([control["cal"][i] for i in idx], [y[i] for i in idx])
        r_mae = mae([residual["cal"][i] for i in idx], [y[i] for i in idx])
        c_contrib = sum(abs(control["cal"][i] - y[i]) for i in idx) / n
        r_contrib = sum(abs(residual["cal"][i] - y[i]) for i in idx) / n
        total_c += c_contrib
        total_r += r_contrib
        group_table.append(
            {
                "group": group,
                "n": len(idx),
                "control_mae": c_mae,
                "residual_mae": r_mae,
                "gain": c_mae - r_mae,
                "control_contribution": c_contrib,
                "residual_contribution": r_contrib,
                "contribution_delta": r_contrib - c_contrib,
            }
        )
    row["group_contribution_sum_control"] = total_c
    row["group_contribution_sum_residual"] = total_r
    row["group_contribution_closes"] = bool(
        abs(total_c - row["control_cal_valid_mae"]) < 1e-6
        and abs(total_r - row["residual_cal_valid_mae"]) < 1e-6
    )
    g0 = next((g for g in group_table if g["group"] == "G0"), None)
    row["g0_contribution_worsening"] = (g0["contribution_delta"] if g0 else None)
    row["gate_calibrated_gain_ge_0.003"] = bool(row["calibrated_gain"] >= 0.003)
    row["gate_gain_without172_positive"] = bool(row["gain_without172"] > 0.0)
    row["gate_g0_worsening_le_0.001"] = bool(
        g0 is not None and g0["contribution_delta"] <= 0.001
    )
    row["purchase_gate_passed"] = bool(
        row["gate_calibrated_gain_ge_0.003"]
        and row["gate_gain_without172_positive"]
        and row["gate_g0_worsening_le_0.001"]
    )

    # health trajectory comparison
    def health_rows(arm_dir: Path) -> list[dict]:
        payload = load_json(arm_dir / "health.json")
        return payload["health"]

    health_c = health_rows(PAIR / "control")
    health_r = health_rows(PAIR / "residual")
    row["health"] = {}
    for label, rows in (("control", health_c), ("residual", health_r)):
        traj = []
        for h in rows:
            traj.append(
                {
                    "epoch": h["epoch"],
                    "slot_rms": h["node_slot"]["rms"],
                    "slot_zero_frac": h["node_slot"]["zero_frac"],
                    "slot_per_graph_rms_mean": h["node_slot"].get("per_graph_rms_mean"),
                    "node_out_rms": h["node_out"]["rms"],
                    "node_out_constant_frac": h["node_out"]["constant_column_frac"],
                    "node_out_zero_frac": h["node_out"]["zero_column_frac"],
                    "W_A_S_rel_change": h["params"]["W_A_S"]["rel_change"],
                    "W_A_S_norm": h["params"]["W_A_S"]["norm"],
                    "W_A_C_rel_change": h["params"]["W_A_C"]["rel_change"],
                    "node_encoder_first_rel_change": h["params"]["node_encoder_first"]["rel_change"],
                    "grad_W_A_S": h["task_grad"]["W_A_S"]["grad_norm"],
                    "grad_W_A_C": h["task_grad"]["W_A_C"]["grad_norm"],
                    "grad_node_encoder_first": h["task_grad"]["node_encoder_first"]["grad_norm"],
                    "product_rms": h["terms"]["product_rms"],
                    "structural_additive_rms": h["terms"]["structural_additive_rms"],
                    "atomic_additive_rms": h["terms"]["atomic_additive_rms"],
                    "cross_mix_delta_rms": h["terms"]["cross_mix_delta_rms"],
                }
            )
        row["health"][label] = traj

    (PAIR / "pair_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (PAIR / "group_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(group_table[0].keys()))
        writer.writeheader()
        writer.writerows(group_table)
    with (PAIR / "pair_valid_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["graph_id", "group", "y", "control_raw", "control_cal", "residual_raw", "residual_cal", "residual_minus_control_cal"]
        )
        for i in range(n):
            writer.writerow(
                [
                    control["graph_id"][i],
                    group_of[i],
                    y[i],
                    control["raw"][i],
                    control["cal"][i],
                    residual["raw"][i],
                    residual["cal"][i],
                    residual["cal"][i] - control["cal"][i],
                ]
            )
    print(json.dumps({k: v for k, v in row.items() if k != "health"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())