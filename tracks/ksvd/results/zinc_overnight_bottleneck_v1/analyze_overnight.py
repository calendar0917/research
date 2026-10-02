"""Analysis for zinc_overnight_bottleneck_v1 (read-only over pulled artifacts).

Scans every exported arm directory, recomputes the group-accounted calibrated
MAE, the paired gains vs the matched control, the frozen gates and the
node-health timelines.  Never touches official test.

Run:
    uv run python -m tracks.ksvd.results.zinc_overnight_bottleneck_v1.analyze_overnight
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

OUT = REPO_ROOT / "tracks/ksvd/results/zinc_overnight_bottleneck_v1"
GROUP_CSV = REPO_ROOT / "tracks/ksvd/results/zinc_small_full_cycle_alignment_v1/valid_per_graph_aligned.csv"
HISTORICAL_CONTROL = 0.10904212416528025
ID172 = 172

NODE_GAIN_MIN = 0.003
G0_WORSENING_MAX = 0.001
T_GAIN_WITHOUT172_MIN = -0.0005
CONFIRM_MEAN_GAIN_MIN = 0.003


def groups() -> list[str]:
    with GROUP_CSV.open() as handle:
        return [row["group"] for row in csv.DictReader(handle)]


def discover() -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    for path in sorted(OUT.glob("*/valid_predictions.csv")):
        match = re.fullmatch(r"([A-Za-z0-9]+)_s(\d+)", path.parent.name)
        if match:
            found[(match.group(1), int(match.group(2)))] = path
    return found


def load_arm(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {
        "tag": path.parent.name,
        "arm": summary.get("arm", path.parent.name),
        "y": np.array([float(r["y"]) for r in rows]),
        "raw": np.array([float(r["pred_raw"]) for r in rows]),
        "cal": np.array([float(r["pred_calibrated"]) for r in rows]),
        "bias": float(rows[0]["train_fitted_bias"]) if rows else float("nan"),
        "summary": summary,
    }


def group_table(arm: dict[str, Any], g: list[str]) -> dict[str, Any]:
    err = np.abs(arm["cal"] - arm["y"])
    out: dict[str, Any] = {}
    for name in ("G0", "G1", "G172"):
        mask = np.array([x == name for x in g])
        out[name] = {"n": int(mask.sum()), "mae": float(err[mask].mean()) if mask.any() else float("nan"), "contribution": float(err[mask].sum() / 1000.0)}
    out["overall"] = {"n": int(len(err)), "mae": float(err.mean()), "contribution": float(err.sum() / 1000.0), "gap_to_0.09": float(err.mean() - 0.09)}
    return out


def paired(control: dict[str, Any], cand: dict[str, Any], g: list[str]) -> dict[str, Any]:
    gc, gk = group_table(control, g), group_table(cand, g)
    mask172 = np.arange(len(g)) != ID172
    y = control["y"]
    return {
        "control_tag": control["tag"],
        "candidate_tag": cand["tag"],
        "control_cal_mae": gc["overall"]["mae"],
        "candidate_cal_mae": gk["overall"]["mae"],
        "candidate_gap_to_0.09": gk["overall"]["gap_to_0.09"],
        "calibrated_gain": float(gc["overall"]["mae"] - gk["overall"]["mae"]),
        "raw_gain": float(np.abs(control["raw"] - y).mean() - np.abs(cand["raw"] - y).mean()),
        "gain_without172": float(np.abs(control["cal"][mask172] - y[mask172]).mean() - np.abs(cand["cal"][mask172] - y[mask172]).mean()),
        "g0_contribution_worsening": float(gk["G0"]["contribution"] - gc["G0"]["contribution"]),
        "g0_contribution_gain": float(gc["G0"]["contribution"] - gk["G0"]["contribution"]),
        "g1_contribution_gain": float(gc["G1"]["contribution"] - gk["G1"]["contribution"]),
        "g172_contribution_gain": float(gc["G172"]["contribution"] - gk["G172"]["contribution"]),
        "control": gc,
        "candidate": gk,
        "bias_control": control["bias"],
        "bias_candidate": cand["bias"],
        "vs_historical_control": float(HISTORICAL_CONTROL - gk["overall"]["mae"]),
    }


def node_gate(p: dict[str, Any]) -> dict[str, bool]:
    return {
        "cal_gain_ge_0.003": bool(p["calibrated_gain"] >= NODE_GAIN_MIN),
        "gain_without172_positive": bool(p["gain_without172"] > 0.0),
        "g0_worsening_le_0.001": bool(p["g0_contribution_worsening"] <= G0_WORSENING_MAX),
    }


def topology_gate(p: dict[str, Any]) -> dict[str, bool]:
    return {
        "cal_gain_ge_0.003": bool(p["calibrated_gain"] >= NODE_GAIN_MIN),
        "g0_worsening_le_0.001": bool(p["g0_contribution_worsening"] <= G0_WORSENING_MAX),
        "gain_without172_ge_-0.0005": bool(p["gain_without172"] >= T_GAIN_WITHOUT172_MIN),
        "broad_paired_signal": bool(p["gain_without172"] > 0.0),
        "outlier_dominated_signal": bool(p["calibrated_gain"] > 0.0 and p["gain_without172"] <= 0.0),
    }


def node_timeline(tag: str) -> list[dict[str, Any]]:
    path = OUT / tag / "health.json"
    if not path.exists():
        return []
    rows = []
    for probe in json.loads(path.read_text())["health"]:
        a, b = probe["batch_a"], probe["batch_b"]
        rows.append(
            {
                "epoch": probe["epoch"],
                "slot_rms_a": a["node_slot"]["rms"],
                "slot_zero_a": a["node_slot"]["zero_frac"],
                "node_out_std_a": a["node_out"]["max_col_std"],
                "node_out_alive_a": a["node_out"]["alive"],
                "node_out_std_b": b["node_out"]["max_col_std"],
                "node_out_alive_b": b["node_out"]["alive"],
                "grad_W_A_S": a["task_grad"]["W_A_S"]["grad_norm"],
                "grad_W_A_C": a["task_grad"]["W_A_C"]["grad_norm"],
                "grad_node_enc": a["task_grad"]["node_encoder_first"]["grad_norm"],
                "product_rms": a["terms"]["product_rms"],
                "dependency_delta": a["dependency"]["delta_rms"],
                "weak_task_use": a["dependency"]["weak_task_use"],
                "gate_healthy": probe["gate"]["healthy"],
            }
        )
    return rows


def main() -> None:
    g = groups()
    found = discover()
    arms = {key: load_arm(path) for key, path in found.items()}
    report: dict[str, Any] = {
        "historical_control_cal_mae": HISTORICAL_CONTROL,
        "official_test_loaded": False,
        "arms": {},
        "paired": {},
        "node_timeline": {},
    }
    for key, arm in arms.items():
        report["arms"][arm["tag"]] = {"group_table": group_table(arm, g), "summary": arm["summary"]}
    for (cand_arm, seed), _ in sorted(arms.items()):
        if cand_arm == "N0":
            continue
        control = arms.get(("N0", seed))
        if control is None:
            continue
        p = paired(control, arms[(cand_arm, seed)], g)
        p["gate"] = topology_gate(p) if cand_arm == "T0" else node_gate(p)
        report["paired"][f"{cand_arm}_s{seed}"] = p
    for tag in sorted({a["tag"] for a in arms.values()} | {p.parent.name for p in OUT.glob("*/health.json")}):
        report["node_timeline"][tag] = node_timeline(tag)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== arms (calibrated valid) ===")
    for arm in arms.values():
        gt = report["arms"][arm["tag"]]["group_table"]
        print(f"{arm['tag']:>12}: cal={gt['overall']['mae']:.6f} raw={arm['summary'].get('raw_valid_mae')} bias={arm['bias']:+.6f} "
              f"G0={gt['G0']['mae']:.6f} G1={gt['G1']['mae']:.6f} G172={gt['G172']['mae']:.4f}")
    print("=== paired vs matched N0 ===")
    for tag, p in report["paired"].items():
        print(f"{tag:>8}: gain={p['calibrated_gain']:+.6f} gain_wo172={p['gain_without172']:+.6f} "
              f"G0worsen={p['g0_contribution_worsening']:+.6f} gate={p['gate']}")
    print("=== node timelines ===")
    for tag, rows in report["node_timeline"].items():
        if not rows:
            continue
        alive = [r["epoch"] for r in rows if r["gate_healthy"]]
        print(f"{tag:>16}: epochs={[r['epoch'] for r in rows]} healthy_epochs={alive}")
        for r in rows:
            print(f"   ep{r['epoch']:>3} slot_rms={r['slot_rms_a']:.3e} out_std_a={r['node_out_std_a']:.3e} out_std_b={r['node_out_std_b']:.3e} "
                  f"gS={r['grad_W_A_S']:.2e} gC={r['grad_W_A_C']:.2e} gE={r['grad_node_enc']:.2e} dep={r['dependency_delta']:.2e} healthy={r['gate_healthy']}")


if __name__ == "__main__":
    main()