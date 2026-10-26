#!/usr/bin/env python
"""Combine frozen B4 seed metrics into the five-class decision.

Reads only JSON/arrays produced after the B4 label read; no data access.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CLASS_NAMES = {
    "A": "NONREDUNDANT_AND_TASK_RELEVANT",
    "B": "NONREDUNDANT_BUT_NOT_PREDICTIVE",
    "C": "REDUNDANT_OR_RECONSTRUCTIBLE",
    "D": "CAPACITY_OR_CALIBRATION_ONLY",
    "E": "INCONCLUSIVE",
}


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_metrics(paths):
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        out[int(d["seed"])] = d
    return out


def classify(metrics, redundancy):
    seeds = sorted(metrics)
    gate_stop = bool(redundancy.get("gate", {}).get("stop_before_new_gpu", False))
    tags = []
    if gate_stop:
        return "C", {"reason": "A4 gate was STOP_BEFORE_NEW_GPU"}, tags
    # A4 is non-redundant by the frozen gate outcome.
    nonredundant = True
    gain_j = {s: float(metrics[s]["gains"]["Gain_J"]) for s in seeds}
    gain_n = {s: float(metrics[s]["gains"]["Gain_vs_N"]) for s in seeds}
    gain_z = {s: float(metrics[s]["gains"]["Gain_Z_over_raw"]) for s in seeds}
    boot_j = {s: metrics[s]["paired_bootstrap"]["Gain_J"] for s in seeds}
    boot_n = {s: metrics[s]["paired_bootstrap"]["Gain_vs_N"] for s in seeds}
    extreme = {s: float(metrics[s]["gain_worst_rows"]["gain_without_top10"]) for s in seeds}
    both_pos = all(gain_j[s] > 0 for s in seeds)
    both_n_pos = all(gain_n[s] > 0 for s in seeds)
    both_practical = all(gain_j[s] >= 0.003 for s in seeds)
    both_ci_pos = all(boot_j[s]["ci95"][0] > 0 for s in seeds)
    extreme_ok = all(extreme[s] > 0 for s in seeds)
    signs = {s: (gain_j[s] > 0) for s in seeds}
    mixed_signs = len(set(signs.values())) > 1
    finite_ok = all(_finite(metrics[s]["arms"][a]["mae"])
                    for s in seeds for a in ("raw", "Z", "ZJ", "ZN"))
    if not finite_ok:
        return "E", {"reason": "non-finite metric"}, tags
    if mixed_signs:
        return "E", {"reason": "seeds disagree on Gain_J direction",
                     "gain_j": gain_j}, tags
    if both_pos and both_n_pos and both_practical and both_ci_pos and extreme_ok:
        return "A", {"reason": "all pre-registered class-A criteria met",
                     "gain_j": gain_j, "gain_n": gain_n,
                     "boot_j": boot_j, "extreme_without_top10": extreme}, tags
    if both_pos and not both_n_pos:
        tags.append("ZN_CONTROL_CAPTURES_GAIN")
        return "D", {"reason": "ZJ improves over Z, but the dim-matched ZN control is not clearly worse",
                     "gain_j": gain_j, "gain_n": gain_n}, tags
    if both_pos and both_n_pos and not both_practical:
        return "B", {"reason": "Gain_J is positive on both seeds but below the 0.003 practical band",
                     "gain_j": gain_j, "gain_n": gain_n}, tags
    if both_pos and both_n_pos and not both_ci_pos:
        # Diagnostic tags are collected independently of the class branch.
        if not extreme_ok:
            tags.append("SINGLE_MOLECULE_DRIVEN")
        if any(gain_n[s] < 0.003 for s in seeds):
            tags.append("ZN_CONTROL_NEAR_GAIN")
        if any(metrics[s]["gains"].get("Gain_huber_J", 0.0) * metrics[s]["gains"].get("Gain_J", 0.0) < 0
               for s in seeds):
            tags.append("HUBER_SENSITIVITY_SIGN_DISAGREES")
        if any(metrics[s]["arms"].get("Z", {}).get("mae", 0) > metrics[s]["arms"].get("raw", {}).get("mae", 0)
               for s in seeds):
            tags.append("AUDIT_HEAD_CALIBRATION_FAILURE")
        return "B", {"reason": "Gain_J is positive but a paired 95% interval includes 0; the gain is not stable across seeds/rows",
                     "gain_j": gain_j, "boot_j": boot_j, "extreme_without_top10": extreme}, tags
    if both_pos and both_n_pos and not extreme_ok:
        tags.append("SINGLE_MOLECULE_DRIVEN")
        return "B", {"reason": "Gain_J does not survive removal of the 10 largest gain rows",
                     "gain_without_top10": extreme}, tags
    if not any(gain_j[s] > 0 for s in seeds):
        return "B", {"reason": "no positive Gain_J on either seed", "gain_j": gain_j}, tags
    return "E", {"reason": "fallback: evidence does not satisfy a stable class",
                 "gain_j": gain_j, "gain_n": gain_n}, tags


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--redundancy", default="analysis/redundancy.json")
    p.add_argument("--out", default="decision.json")
    args = p.parse_args(argv)
    metrics = load_metrics(args.metrics)
    redundancy = json.loads(Path(args.redundancy).read_text())
    a4_mlp = {}
    for target, feats in redundancy.get("targets", {}).items():
        for feat, v in feats.items():
            if feat.startswith("z_") and "mlp" in v and "probe" in v["mlp"]:
                a4_mlp[f"{target}:{feat}"] = {"probe_r2_block": v["mlp"]["probe"].get("r2_block")}
    cls, why, tags = classify(metrics, redundancy)
    # Diagnostic tags from the frozen strata, never overriding the class.
    tail_share = None
    for s, m in metrics.items():
        st = m.get("strata", {}).get("tail", {})
        tail = st.get("tail_y_le_-3")
        ordinary = st.get("ordinary_y_gt_-3")
        if tail and ordinary and m.get("arms", {}).get("ZJ"):
            n_t = tail["n"]
            n_o = ordinary["n"]
            mae_t = tail["ZJ"]["mean"]
            mae_o = ordinary["ZJ"]["mean"]
            total_num = n_t * mae_t + n_o * mae_o
            share = (n_t * mae_t / total_num) if total_num > 0 else None
            tail_share = share if tail_share is None else max(tail_share, share)
    if tail_share is not None and tail_share >= 0.5:
        tags.append("BENCHMARK_TAIL_DOMINATED")
    # A4 ProbeFit outlier sensitivity is a known property of this split.
    gate_rows = redundancy.get("gate", {}).get("rows", [])
    if any(r.get("target", "").startswith("J_latent") and r.get("block_r2_probe", 0) < 0
           for r in gate_rows):
        tags.append("PROBEFIT_OUTLIER_SENSITIVE")
    # ProbeFit residual-SSE concentration (ProbeFit labels only, diagnosis).
    probe_out = None
    probe_path = Path(args.redundancy).parent / "b" / "probe_outlier_diagnostic.json"
    if probe_path.exists():
        probe_out = json.loads(probe_path.read_text())
        shares = [v.get("sse_share_top2", 0.0) for v in probe_out.get("seeds", {}).values()]
        if shares and max(shares) >= 0.5:
            tags.append("AUDIT_HEAD_OUTLIER_DOMINATED")
    tags = sorted(set(tags))
    out = {
        "protocol": "zinc-dcc-information-audit-v1",
        "class": cls,
        "class_name": CLASS_NAMES[cls],
        "no_justified_new_model": cls != "A",
        "next_round": ("estimation/data mechanisms; no new structured-composition model "
                       "unless a calibrated, pre-registered replication shows a stable gain"),
        "tags": tags,
        "rationale": why,
        "a4_gate_stop": bool(redundancy.get("gate", {}).get("stop_before_new_gpu", False)),
        "a4_gate_rows": gate_rows,
        "a4_mlp_sensitivity": a4_mlp,
        "seed_metrics": {str(s): {
            "gains": metrics[s].get("gains"),
            "paired_bootstrap": metrics[s].get("paired_bootstrap"),
            "gain_worst_rows": metrics[s].get("gain_worst_rows"),
            "train_fit_vs_audit": metrics[s].get("train_fit_vs_audit"),
            "arms": metrics[s].get("arms"),
        } for s in sorted(metrics)},
        "tail_error_share_max": tail_share,
        "probe_outlier_diagnostic": probe_out,
        "interpretation": (
            "Class A supports only next-round joint-structure work; it is not an "
            "information-theoretic insufficiency proof and not evidence of reaching "
            "the 0.05x target."
        ),
        "official_valid_used": False,
        "official_test_used": False,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"class": cls, "class_name": CLASS_NAMES[cls], "tags": tags,
                      "rationale": why}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
