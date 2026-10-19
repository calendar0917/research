"""Phase 1c analysis: absolute table, paired bootstrap, pre-registered labels.

Reads only the Phase 1c run JSONs (Fit/Monitor labels and Monitor predictions)
plus the frozen Phase 1b anchor JSONs.  No Reserved or official valid/test
labels are read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PHASE1B = HERE.parent / "phase1b"
for p in (HERE, PHASE1B):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_phase1b import paired_bootstrap  # noqa: E402

RUNS = REPO / "tracks/structural-sufficiency/runs/phase1c"
RUNS_1B = REPO / "tracks/structural-sufficiency/runs/phase1b"
OUT = REPO / "tracks/structural-sufficiency/results/phase1c/analysis_phase1c.json"
BANDS = {"negligible": 0.01, "meaningful": 0.02}
PHASE1C_MODELS = ["S_cap_size", "R1_size", "R1_marginal_size"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def run_table(path: Path) -> dict[str, Any]:
    d = load_json(path)
    labels = np.asarray(d["monitor_y"], dtype=np.float64)
    pred = np.asarray(d["monitor_pred_best"], dtype=np.float64)
    err = pred - labels
    return {
        "path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
        "model": d["model"],
        "seed": int(d["seed"]),
        "params": int(d["parameters"]),
        "flops_forward": d.get("flops", {}).get("flops_forward"),
        "step_time_ms": d.get("step_time_ms_median"),
        "gpu_seconds": d.get("gpu_seconds"),
        "train_seconds": d.get("train_seconds"),
        "peak_gpu_mb": d.get("peak_gpu_memory_mb_allocated"),
        "best_epoch": int(d["best_epoch"]),
        "epochs_planned": int(d.get("config", {}).get("epochs", -1)),
        "epochs_run": len(d.get("curves", [])),
        "train_mae": float(d["best_metrics"]["train"]["mae"]),
        "monitor_mae": mae(pred, labels),
        "monitor_rmse": rmse(pred, labels),
        "median_abs_error": float(np.median(np.abs(pred - labels))),
        "p90_abs_error": float(np.quantile(np.abs(pred - labels), 0.90)),
        "train_monitor_gap": float(d["best_metrics"]["monitor"]["mae"] - d["best_metrics"]["train"]["mae"]),
        "monitor_mol_index": [int(x) for x in d["monitor_mol_index"]],
        "monitor_y": labels.tolist(),
        "monitor_pred": pred.tolist(),
    }


def compare(a: Mapping[str, Any], b: Mapping[str, Any], n_boot: int) -> dict[str, Any]:
    """Positive delta = b better (MAE(a)-MAE(b))."""
    ya = np.asarray(a["monitor_y"], dtype=np.float64)
    yb = np.asarray(b["monitor_y"], dtype=np.float64)
    assert np.array_equal(ya, yb)
    pa = np.asarray(a["monitor_pred"], dtype=np.float64)
    pb = np.asarray(b["monitor_pred"], dtype=np.float64)
    boot = paired_bootstrap(pa - ya, pb - yb, n_boot=n_boot)
    rmse_a = rmse(pa, ya)
    rmse_b = rmse(pb, yb)
    rmse_delta = rmse_a - rmse_b
    boot["rmse_delta"] = float(rmse_delta)
    boot["rmse_agrees"] = bool(np.sign(boot["delta"]) == np.sign(rmse_delta))
    boot["mae_a"] = float(np.mean(np.abs(pa - ya)))
    boot["mae_b"] = float(np.mean(np.abs(pb - yb)))
    boot["win_rate_b_better"] = float(np.mean(np.abs(pb - yb) < np.abs(pa - ya)))
    boot["median_abs_error_a"] = float(np.median(np.abs(pa - ya)))
    boot["median_abs_error_b"] = float(np.median(np.abs(pb - yb)))
    boot["p90_abs_error_a"] = float(np.quantile(np.abs(pa - ya), 0.90))
    boot["p90_abs_error_b"] = float(np.quantile(np.abs(pb - yb), 0.90))
    return boot


def decide(comp: Mapping[str, Any], best_mae: float, gref_mae: float) -> dict[str, Any]:
    pair = comp["gain_pair_marginal_minus_r1size"]
    cap = comp["gain_over_capacity_scap_minus_r1size"]
    readout_r1 = comp["gain_readout_vs_R1"]

    def meaningful(c: Mapping[str, Any]) -> bool:
        return bool(c["delta"] >= BANDS["meaningful"] and c["ci95"][0] > 0 and c["rmse_agrees"])

    def negligible(c: Mapping[str, Any]) -> bool:
        return bool(abs(c["delta"]) < BANDS["negligible"])

    pair_m = meaningful(pair)
    cap_m = meaningful(cap)
    pair_neg = pair["delta"] < BANDS["negligible"]
    cap_neg = negligible(cap)
    readout_m = meaningful(readout_r1)
    gap = best_mae - gref_mae

    labels: list[str] = []
    if pair_m and cap_m:
        labels.append("CORRESPONDENCE_SURVIVES_CAPACITY")
    elif pair_m and not cap_m:
        labels.append("CORRESPONDENCE_VS_MARGINAL_ONLY")
    if cap_neg and not pair_m:
        labels.append("CAPACITY_READOUT_EXPLAINS")
    if readout_m and abs(cap["delta"]) < BANDS["meaningful"]:
        labels.append("READOUT_BOTTLENECK_DOMINATES")
    if pair_neg:
        labels.append("NO_ENDPOINT_CORRESPONDENCE_GAIN")
    if gap >= BANDS["meaningful"]:
        labels.append("ABSOLUTE_GAP_PERSISTS")
    if (not pair_m) and (not pair_neg) and not (0.01 <= pair["delta"] < 0.02):
        labels.append("INSUFFICIENT_EVIDENCE")
    if 0.01 <= pair["delta"] < 0.02:
        labels.append("INSUFFICIENT_EVIDENCE")

    priority = [
        "NO_ENDPOINT_CORRESPONDENCE_GAIN",
        "CORRESPONDENCE_SURVIVES_CAPACITY",
        "CORRESPONDENCE_VS_MARGINAL_ONLY",
        "READOUT_BOTTLENECK_DOMINATES",
        "CAPACITY_READOUT_EXPLAINS",
        "INSUFFICIENT_EVIDENCE",
    ]
    primary = next((x for x in priority if x in labels), "UNRESOLVED")
    return {
        "labels": labels,
        "primary": primary,
        "pair_meaningful": pair_m,
        "capacity_meaningful": cap_m,
        "capacity_negligible": cap_neg,
        "pair_negligible_or_reversed": pair_neg,
        "readout_vs_R1_meaningful": readout_m,
        "gap_best_minus_Gref": float(gap),
        "absolute_gap_persists": bool(gap >= BANDS["meaningful"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--runs-1b", default=str(RUNS_1B))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    runs = Path(args.runs)
    runs_1b = Path(args.runs_1b)

    seed0: dict[str, Any] = {}
    for model in PHASE1C_MODELS:
        path = runs / "zinc" / f"{model}_seed0.json"
        if not path.exists():
            raise SystemExit(f"missing Phase 1c run {path}")
        seed0[model] = run_table(path)
    assert len({tuple(r["monitor_mol_index"]) for r in seed0.values()}) == 1

    anchors: dict[str, Any] = {}
    for name, fname in [
        ("S0", "S0_seed0.json"),
        ("R1", "R1_seed0.json"),
        ("M2", "M2_seed0.json"),
        ("G_ref", "G_ref_seed0.json"),
    ]:
        path = runs_1b / "zinc" / fname
        if path.exists():
            anchors[name] = run_table(path)

    comp: dict[str, Any] = {}
    comp["gain_pair_marginal_minus_r1size"] = compare(seed0["R1_marginal_size"], seed0["R1_size"], args.n_boot)
    comp["gain_over_capacity_scap_minus_r1size"] = compare(seed0["S_cap_size"], seed0["R1_size"], args.n_boot)
    comp["gain_over_capacity_scap_minus_marginal"] = compare(seed0["S_cap_size"], seed0["R1_marginal_size"], args.n_boot)
    for anchor in ["S0", "R1", "G_ref"]:
        if anchor in anchors:
            comp[f"gain_{anchor}_minus_Scap"] = compare(anchors[anchor], seed0["S_cap_size"], args.n_boot)
            comp[f"gain_{anchor}_minus_R1size"] = compare(anchors[anchor], seed0["R1_size"], args.n_boot)
            comp[f"gain_{anchor}_minus_marginal"] = compare(anchors[anchor], seed0["R1_marginal_size"], args.n_boot)
            comp[f"gap_R1size_minus_{anchor}"] = compare(seed0["R1_size"], anchors[anchor], args.n_boot)
    if "G_ref" in anchors:
        comp["gap_Scap_minus_Gref"] = compare(seed0["S_cap_size"], anchors["G_ref"], args.n_boot)
        comp["gap_marginal_minus_Gref"] = compare(seed0["R1_marginal_size"], anchors["G_ref"], args.n_boot)
    if "R1" in anchors:
        comp["gain_readout_vs_R1"] = compare(anchors["R1"], seed0["R1_size"], args.n_boot)
        comp["gain_readout_vs_R1_marginal"] = compare(anchors["R1"], seed0["R1_marginal_size"], args.n_boot)
    if "S0" in anchors:
        comp["gain_readout_vs_S0"] = compare(anchors["S0"], seed0["S_cap_size"], args.n_boot)

    best_name = min(seed0, key=lambda k: seed0[k]["monitor_mae"])
    best_mae = seed0[best_name]["monitor_mae"]
    gref_mae = anchors["G_ref"]["monitor_mae"] if "G_ref" in anchors else float("nan")
    decision = decide(comp, best_mae, gref_mae)
    decision["best_model"] = best_name
    decision["best_mae"] = float(best_mae)

    seeds: dict[str, Any] = {}
    for seed in (1,):
        s1: dict[str, Any] = {}
        for model in ["R1_size", "R1_marginal_size"]:
            path = runs / "zinc" / f"{model}_seed{seed}.json"
            if path.exists():
                s1[model] = run_table(path)
        if len(s1) == 2:
            assert tuple(s1["R1_size"]["monitor_mol_index"]) == tuple(seed0["R1_size"]["monitor_mol_index"])
            seeds[f"seed{seed}"] = {
                "models": {k: {kk: vv for kk, vv in v.items() if kk not in {"monitor_y", "monitor_pred", "monitor_mol_index"}} for k, v in s1.items()},
                "gain_pair_marginal_minus_r1size": compare(s1["R1_marginal_size"], s1["R1_size"], args.n_boot),
            }
        elif s1:
            seeds[f"seed{seed}"] = {"incomplete": True, "models_present": sorted(s1)}
    if seeds:
        pair0 = comp["gain_pair_marginal_minus_r1size"]["delta"]
        pairs = [pair0] + [v["gain_pair_marginal_minus_r1size"]["delta"] for v in seeds.values() if "gain_pair_marginal_minus_r1size" in v]
        signs = [p > 0 for p in pairs]
        seeds["summary"] = {
            "pair_gains": pairs,
            "mean_pair_gain": float(np.mean(pairs)),
            "same_positive_sign": bool(all(signs)),
            "n_seeds": len(pairs),
        }

    keep_edge: dict[str, Any] = {}
    mk = runs / "zinc" / "M_keep_edge_seed0.json"
    r1l = runs / "zinc" / "R1_size_seed0_long.json"
    if mk.exists() and r1l.exists():
        mkd = run_table(mk)
        r1d = run_table(r1l)
        r1base = run_table(runs / "zinc" / "R1_size_seed0.json")
        upd = compare(r1d, mkd, args.n_boot)
        same_recipe = compare(r1base, mkd, args.n_boot)
        if upd["delta"] >= BANDS["meaningful"] and upd["ci95"][0] > 0 and upd["rmse_agrees"]:
            pre_label = "UPDATE_HELPS_WITH_DIRECT_EDGE_READOUT"
        elif upd["delta"] < BANDS["negligible"]:
            pre_label = "NO_UPDATE_GAIN_WITH_DIRECT_EDGE_READOUT"
        else:
            pre_label = "INSUFFICIENT_EVIDENCE"
        # The longer cosine schedule is a different optimization problem, not a pure
        # compute match.  If it failed to fit, the pre-registered contrast is confounded.
        schedule_confounded = (
            r1d["train_mae"] - r1base["train_mae"] > 0.03
            or r1d["best_epoch"] < 0.6 * r1d["epochs_run"]
        )
        if schedule_confounded:
            upd_label = "MATCHED_COMPUTE_ARM_OPTIMIZATION_CONFOUNDED"
        elif same_recipe["delta"] > BANDS["meaningful"]:
            upd_label = "UPDATE_HURTS_SAME_RECIPE"
        elif same_recipe["delta"] < -BANDS["meaningful"] and same_recipe["ci95"][1] < 0:
            upd_label = "UPDATE_HINT_BUT_COMPUTE_PARAM_CONFOUNDED"
        elif abs(same_recipe["delta"]) < BANDS["meaningful"]:
            upd_label = "NO_UPDATE_GAIN_WITH_DIRECT_EDGE_READOUT"
        else:
            upd_label = "INSUFFICIENT_EVIDENCE"
        keep_edge = {
            "M_keep_edge": {k: v for k, v in mkd.items() if k not in {"monitor_y", "monitor_pred", "monitor_mol_index"}},
            "R1_size_long": {k: v for k, v in r1d.items() if k not in {"monitor_y", "monitor_pred", "monitor_mol_index"}},
            "gain_update_keep_edge_matched": upd,
            "same_recipe_gain_r1size_minus_mkeep": same_recipe,
            "pre_registered_label": pre_label,
            "schedule_confounded": schedule_confounded,
            "r1_long_train_mae_delta_vs_r1size": r1d["train_mae"] - r1base["train_mae"],
            "decision": upd_label,
            "compute": {
                "r1_size_long_planned_epochs": r1d["epochs_planned"],
                "m_keep_edge_planned_epochs": mkd["epochs_planned"],
                "r1_size_long_epochs_run": r1d["epochs_run"],
                "m_keep_edge_epochs_run": mkd["epochs_run"],
                "r1_size_long_gpu_seconds": r1d["gpu_seconds"],
                "m_keep_edge_gpu_seconds": mkd["gpu_seconds"],
                "r1_size_long_flops_per_epoch": r1d["flops_forward"],
                "m_keep_edge_flops_per_epoch": mkd["flops_forward"],
                "flop_ratio_per_epoch": (mkd["flops_forward"] / r1d["flops_forward"]) if r1d["flops_forward"] else None,
                "actual_flop_ratio": (
                    (mkd["flops_forward"] * mkd["epochs_run"])
                    / (r1d["flops_forward"] * r1d["epochs_run"])
                ) if r1d["flops_forward"] and r1d["epochs_run"] else None,
            },
        }
    else:
        keep_edge = {"status": "not_run", "reason": "follow-up gate not triggered or not yet purchased"}

    out = {
        "protocol_id": "ssrc-phase1c-zinc-v1",
        "bands": BANDS,
        "seed0": {k: {kk: vv for kk, vv in v.items() if kk not in {"monitor_y", "monitor_pred", "monitor_mol_index"}} for k, v in seed0.items()},
        "anchors": {k: {kk: vv for kk, vv in v.items() if kk not in {"monitor_y", "monitor_pred", "monitor_mol_index"}} for k, v in anchors.items()},
        "comparisons": comp,
        "decision": decision,
        "seed_replication": seeds,
        "m_keep_edge": keep_edge,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print("== Phase 1c seed 0 ==")
    print(f"{'model':18s} {'params':>8s} {'MAE':>8s} {'RMSE':>8s} {'train':>8s} {'best_ep':>7s} {'eps':>4s} {'gpu_s':>8s}")
    for name, v in seed0.items():
        print(f"{name:18s} {v['params']:8d} {v['monitor_mae']:8.4f} {v['monitor_rmse']:8.4f} {v['train_mae']:8.4f} {v['best_epoch']:7d} {v['epochs_run']:4d} {float(v['gpu_seconds'] or 0):8.1f}")
    print("\ncomparisons (positive = right-hand model better):")
    for name, v in comp.items():
        print(f"  {name:42s} {v['delta']:+.4f} CI95 [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] rmse={v['rmse_delta']:+.4f}")
    print(f"\nDECISION: {decision['primary']}")
    print(f"  labels: {decision['labels']}")
    print(f"  best={decision['best_model']} MAE={decision['best_mae']:.4f} vs G_ref={gref_mae:.4f} gap={decision['gap_best_minus_Gref']:+.4f}")
    if seeds:
        print(f"\nseed replication: {json.dumps(seeds.get('summary', {'status': 'incomplete'}), indent=2)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
