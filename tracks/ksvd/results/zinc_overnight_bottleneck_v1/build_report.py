"""Build the deliverable tables for zinc_overnight_bottleneck_v1.

Reads the pulled per-arm exports + analysis.json and writes:
  group_table.csv, pair_valid_predictions.csv, health_2x2.csv, main_table.csv

Run:
    uv run python -m tracks.ksvd.results.zinc_overnight_bottleneck_v1.build_report
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT
from tracks.ksvd.results.zinc_overnight_bottleneck_v1 import analyze_overnight as an

OUT = an.OUT


def severity_masks() -> dict[str, np.ndarray]:
    """Stratify the 1000 official-valid rows by the snapped cycle penalty label."""
    with an.GROUP_CSV.open() as handle:
        rows = list(csv.DictReader(handle))
    pen = np.array([float(r["label_effective_cycle_snapped"]) for r in rows])
    return {
        "penalty_0": pen == 0.0,
        "penalty_-1": pen == -1.0,
        "penalty_le_-2": pen <= -2.0,
        "penalty_any_neg": pen < 0.0,
    }


def write_severity_table(arms: dict[str, dict[str, np.ndarray]]) -> None:
    masks = severity_masks()
    with (OUT / "severity_table.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["arm", "stratum", "n", "mae", "contribution"])
        for tag, a in sorted(arms.items()):
            err = np.abs(a["cal"] - a["y"])
            for name, mask in masks.items():
                w.writerow([
                    tag,
                    name,
                    int(mask.sum()),
                    float(err[mask].mean()) if mask.any() else float("nan"),
                    float(err[mask].sum() / 1000.0),
                ])


def arm_predictions() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for (arm, seed), path in an.discover().items():
        a = an.load_arm(path)
        out[a["tag"]] = {"y": a["y"], "raw": a["raw"], "cal": a["cal"], "group": np.array(an.groups())}
    return out


def write_group_table(arms: dict[str, dict[str, np.ndarray]], g: list[str]) -> None:
    with (OUT / "group_table.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["arm", "group", "n", "mae", "contribution"])
        for tag, a in sorted(arms.items()):
            err = np.abs(a["cal"] - a["y"])
            for name in ("G0", "G1", "G172"):
                mask = np.array([x == name for x in g])
                w.writerow([tag, name, int(mask.sum()), float(err[mask].mean()), float(err[mask].sum() / 1000.0)])
            w.writerow([tag, "overall", int(len(err)), float(err.mean()), float(err.sum() / 1000.0)])


def write_pair_predictions(arms: dict[str, dict[str, np.ndarray]], g: list[str]) -> None:
    tags = sorted(arms)
    with (OUT / "pair_valid_predictions.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        header = ["graph_id", "group", "y"]
        for tag in tags:
            header += [f"{tag}_raw", f"{tag}_cal"]
        header += ["abs_err_" + tags[0]]
        w.writerow(header)
        for i in range(len(g)):
            row: list[Any] = [i, g[i], float(arms[tags[0]]["y"][i])]
            for tag in tags:
                row += [float(arms[tag]["raw"][i]), float(arms[tag]["cal"][i])]
            row.append(float(abs(arms[tags[0]]["cal"][i] - arms[tags[0]]["y"][i])))
            w.writerow(row)


def write_health() -> None:
    tags = sorted(p.parent.name for p in OUT.glob("*/health.json"))
    with (OUT / "health_2x2.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["arm_tag", "epoch", "slot_rms_a", "slot_zero_a", "node_out_std_a", "node_out_alive_a",
                    "node_out_std_b", "node_out_alive_b", "grad_W_A_S", "grad_W_A_C", "grad_node_enc",
                    "product_rms", "dependency_delta", "weak_task_use", "gate_healthy"])
        for tag in tags:
            for r in an.node_timeline(tag):
                w.writerow([tag, r["epoch"], r["slot_rms_a"], r["slot_zero_a"], r["node_out_std_a"], r["node_out_alive_a"],
                            r["node_out_std_b"], r["node_out_alive_b"], r["grad_W_A_S"], r["grad_W_A_C"], r["grad_node_enc"],
                            r["product_rms"], r["dependency_delta"], r["weak_task_use"], r["gate_healthy"]])


def write_main_table(report: dict[str, Any]) -> None:
    rows = []
    for tag, info in sorted(report["arms"].items()):
        gt = info["group_table"]
        s = info["summary"]
        rows.append({
            "arm": tag,
            "cal_valid_mae": gt["overall"]["mae"],
            "raw_valid_mae": s.get("raw_valid_mae"),
            "train_fitted_bias": s.get("train_fitted_bias"),
            "raw_train_mae": s.get("raw_train_mae"),
            "cal_train_mae": s.get("calibrated_train_mae"),
            "eval_gap": s.get("eval_gap"),
            "G0_mae": gt["G0"]["mae"],
            "G0_contribution": gt["G0"]["contribution"],
            "G1_mae": gt["G1"]["mae"],
            "G1_contribution": gt["G1"]["contribution"],
            "G172_mae": gt["G172"]["mae"],
            "G172_contribution": gt["G172"]["contribution"],
            "epochs": s.get("epochs"),
            "members": s.get("members"),
            "kappa": s.get("kappa"),
            "node_weight_decay": s.get("node_weight_decay"),
            "init_state_sha256": s.get("init_state_sha256"),
        })
    with (OUT / "main_table.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["arm"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    g = an.groups()
    arms = arm_predictions()
    write_group_table(arms, g)
    write_pair_predictions(arms, g)
    write_severity_table(arms)
    write_health()
    report = json.loads((OUT / "analysis.json").read_text()) if (OUT / "analysis.json").exists() else {"arms": {}}
    if report.get("arms"):
        write_main_table(report)
    print("wrote group_table.csv, pair_valid_predictions.csv, health_2x2.csv", "main_table.csv" if report.get("arms") else "")


if __name__ == "__main__":
    main()