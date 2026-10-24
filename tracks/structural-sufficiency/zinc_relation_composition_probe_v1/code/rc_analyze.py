"""Analysis for the D/S relation-composition probe (valid only; no test).

Produces, under ``analysis/``:

* ``summary.json``            per-run best-valid / W20 / T100 / final / train@best,
                              the D-S contrast, the D/S vs historical B-R0
                              reference, and the frozen group description
* ``curves.csv``              the two valid curves in one long table
* ``per_molecule_valid.csv``  row, y_raw, predictions and absolute errors
* ``intervention.json``       the one fixed-weight intervention on the trained D
                              (relation-write rule switched D -> S, full re-forward)

The comparison to B-R0 uses the already-reported B-R0 seed-0 checkpoint
predictions from ``zinc_performance_search_v1``; it is a *historical* reference
produced in an earlier round, not a re-training in this environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

import rc_models as RM  # noqa: E402,F401
import rc_data as RD  # noqa: E402
import rc_runner as RR  # noqa: E402
import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
ZPS = _SS / "zinc_performance_search_v1"
GENERALIZATION = _SS / "zinc_generalization_audit_v1" / "analysis" / "per_molecule.csv"
B_REFERENCE = {
    "name": "B-R0-s0 (historical)",
    "valid_mae": 0.07853714964160463,
    "best_epoch": 489,
    "w20": 0.07908245333650848,
    "final": 0.07907641669729491,
    "train_mae_at_best": 0.005509941598307341,
    "groups": {
        "all_1000": 0.0785371522932536, "seen_749": 0.05426827371593307,
        "unseen_251": 0.15095703298812643, "ordinary_935": 0.06553710583988859,
        "tail_65": 0.26553782050704255, "unseen_ordinary_225": 0.13077343996946705,
        "seen_ordinary_710": 0.04486361967206444, "unseen_tail_26": 0.32562274180344797,
    },
    "params": 494193,
    "predictions": ZPS / "analysis" / "valid_pred_B-R0-s0_s0.npz",
    "source": "zinc_dual_channel_composer_v1/runs/L_s0 (via zinc_performance_search_v1)",
}
ARM_SEED = {"D": 0, "S": 0}


def _rel(path):
    try:
        return str(Path(path).relative_to(_HERE.parents[2]))
    except ValueError:
        return str(path)


def _load_labels():
    import csv

    y, unseen = [], []
    with open(GENERALIZATION) as fh:
        for row in csv.DictReader(fh):
            y.append(float(row["y_raw"]))
            unseen.append(float(row["r2_unseen_atom_frac"]) > 0)
    return np.asarray(y, dtype=np.float64), np.asarray(unseen, dtype=bool)


def GROUPS(y, unseen):
    return {
        "all_1000": np.ones(len(y), dtype=bool),
        "seen_749": ~unseen,
        "unseen_251": unseen,
        "ordinary_935": y > -3.0,
        "tail_65": y <= -3.0,
        "unseen_ordinary_225": unseen & (y > -3.0),
        "seen_ordinary_710": (~unseen) & (y > -3.0),
        "unseen_tail_26": unseen & (y <= -3.0),
    }


def _curve_metrics(curve, key="valid_mae"):
    vals = [float(e[key]) for e in curve if e.get(key) is not None]
    w20 = None
    if len(vals) >= 20:
        w20 = float(min(float(np.median(vals[i:i + 20])) for i in range(len(vals) - 19)))
    t100 = float(np.median(vals[-100:])) if len(vals) >= 100 else float(np.median(vals))
    return w20, t100, float(vals[-1])


def _run_record(arm: str, seed: int) -> dict:
    out = RR.run_dir(arm, seed)
    info = C.read_json(out / "train.json")
    curve = info["curve"]
    w20, t100, final = _curve_metrics(curve)
    npz = np.load(out / "predictions" / "valid_best.npz")
    pred = npz["pred"].astype(np.float64)
    return {
        "arm": arm,
        "seed": int(seed),
        "run_dir": _rel(out),
        "parameters": info["parameter_count"],
        "epochs": info["recipe"]["epochs"],
        "updates": info["recipe"]["updates"],
        "best_valid": float(info["best_valid"]["valid_mae"]),
        "best_epoch": int(info["best_valid"]["epoch"]),
        "w20": w20,
        "t100": t100,
        "final_valid": final,
        "train_mae_at_best_eval_mode": float(info["fixed_objects"]["best"]["train_mae_eval_mode"]),
        "train_mae_at_final_eval_mode": float(info["fixed_objects"]["final"]["train_mae_eval_mode"]),
        "gpu_hours": float(info.get("wall_seconds", 0.0)) / 3600.0,
        "curve": curve,
        "pred": pred,
        "initial_state_sha256": info["model_info"].get("initial_state_sha256"),
        "ssl_sha256": info["model_info"].get("ssl", {}).get("sha256"),
        "git_commit": info.get("git_commit"),
    }


# ---------------------------------------------------------------------------
# the one fixed-weight intervention
# ---------------------------------------------------------------------------
def intervention(arm: str = "D", seed: int = 0, batch_size: int = 128) -> dict:
    """Load the trained D checkpoint; re-forward valid with the S write rule."""
    out = RR.run_dir(arm, seed)
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    y, _unseen = _load_labels()
    collator, _stats = RR._collator()
    valid_cache = D.load_cache("valid")
    norm = D.load_norm()
    stats = RD.load_split("train")["stats"]
    model, _ = RM.build_model("RC", norm, seed, "cpu", arm=arm, stats=stats)
    C.load_state(model, ckpt["state_dict"], "cpu")
    model.eval()
    rows = np.arange(len(y))
    preds = {}
    rel_states = {}
    for rule in ("D", "S"):
        model.set_arm(rule)
        chunks = []
        with torch.no_grad(), C.isolated_rng():
            for s in range(0, len(rows), batch_size):
                chunk = rows[s:s + batch_size]
                batch = T1._to_device(collator(valid_cache, chunk, "RC", valid=True), "cpu")
                chunks.append(model(batch).numpy())
        preds[rule] = np.concatenate(chunks).astype(np.float64)
    # relation-state difference after the first layer, on a fixed small batch
    batch = T1._to_device(collator(valid_cache, list(range(16)), "RC", valid=True), "cpu")
    with torch.no_grad():
        cg, cl = RM.composer_pad(model, batch)
        rel_diff = {}
        for rule in ("D", "S"):
            model.set_arm(rule)
            h, e, pm = RM.layer0_inputs(model, batch, cg, cl)
            h1, e1 = model.core.layers[0](h, e, pm, batch["bond_mask"], batch["log_line_deg"])
            rel_diff[rule] = e1
        d_rel = float((rel_diff["D"] - rel_diff["S"]).abs().max())
        d_rel_mean = float((rel_diff["D"] - rel_diff["S"]).abs().mean())
    ae = {k: np.abs(preds[k] - y) for k in preds}
    delta = preds["S"] - preds["D"]
    reported = float(C.read_json(out / "train.json")["best_valid"]["valid_mae"])
    reported_final = float(C.read_json(out / "train.json")["fixed_objects"]["final"]["valid_mae"])
    return {
        "reported_best_valid_checkpoint": reported,
        "reproduction_absdiff_vs_reported": abs(ae["D"].mean() - reported),
        "trained_arm": arm,
        "seed": int(seed),
        "checkpoint": _rel(out / "best.pt"),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "reported_final_valid": reported_final,
        "valid_mae_with_D_rule": float(ae["D"].mean()),
        "valid_mae_with_S_rule_same_weights": float(ae["S"].mean()),
        "mae_difference_S_minus_D": float(ae["S"].mean() - ae["D"].mean()),
        "prediction_change_abs_mean": float(np.abs(delta).mean()),
        "prediction_change_abs_p50": float(np.median(np.abs(delta))),
        "prediction_change_abs_p90": float(np.percentile(np.abs(delta), 90)),
        "prediction_change_abs_max": float(np.abs(delta).max()),
        "layer0_relation_state_max_abs_diff": d_rel,
        "layer0_relation_state_mean_abs_diff": d_rel_mean,
        "note": ("fixed weights of the trained D model; only the relation-write rule "
                 "changes; the real-token attention is untouched; a full re-forward "
                 "from the inputs is used"),
    }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# markdown tables (repo convention: analysis/report_tables.md)
# ---------------------------------------------------------------------------
def _md(summary, records, aerr, groups, y) -> str:
    f = lambda v, n=6: ("n/a" if v is None else f"{v:.{n}f}")
    L = []
    L.append("### 训练结果（valid only，官方 test 未加载）\n")
    L.append("| arm | seed | params | epochs | best-valid | ep | W20 | T100 | final | train@best(eval) | gpu-h |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, r in records.items():
        L.append(f"| {arm} | {r['seed']} | {r['parameters']} | {r['epochs']} | {f(r['best_valid'])} | "
                 f"{r['best_epoch']} | {f(r['w20'])} | {f(r['t100'])} | {f(r['final_valid'])} | "
                 f"{f(r['train_mae_at_best_eval_mode'])} | {f(r['gpu_hours'], 3)} |")
    b = summary["b_reference"]
    L.append(f"| B-R0 (历史) | 0 | {b['params']} | 600 | {f(b['valid_mae'])} | {b['best_epoch']} | "
             f"{f(b['w20'])} | n/a | {f(b['final'])} | n/a | n/a |")
    if summary.get("d_minus_s"):
        d = summary["d_minus_s"]
        L.append("")
        L.append("### D − S（同 seed 0，逐指标）\n")
        L.append("| 指标 | D − S |")
        L.append("|---|---|")
        for k, v in d.items():
            L.append(f"| {k} | {v:+.6f} |")
    L.append("")
    L.append("### 与 B-R0 的差（负 = 更好）\n")
    L.append("| arm | best-valid − B | W20 − B | final − B |")
    L.append("|---|---|---|---|")
    for arm, v in summary["vs_b_reference"].items():
        L.append(f"| {arm} | {v['best_valid_minus_B']:+.6f} | {v['w20_minus_B']:+.6f} | "
                 f"{v['final_minus_B']:+.6f} |")
    L.append("")
    L.append("### 配对 bootstrap（valid 行重采样，固定 checkpoint，描述性）\n")
    L.append("| 对比 | ΔMAE | 95% CI | 前者更好的行占比 |")
    L.append("|---|---|---|---|")
    for k, v in summary["contrasts_bootstrap"].items():
        L.append(f"| {k} | {v['mean_delta_mae']:+.6f} | [{v['ci95_low']:+.6f}, {v['ci95_high']:+.6f}] | "
                 f"{v['share_a_better']:.3f} |")
    L.append("")
    L.append("### 稳健性（valid）\n")
    L.append("| arm | MAE | 去 row172 | 去自身 top10 |")
    L.append("|---|---|---|---|")
    for arm, r in records.items():
        L.append(f"| {arm} | {f(r['valid_mae_recomputed'])} | {f(r['mae_drop_row172'])} | {f(r['mae_drop_top10'])} |")
    L.append("")
    L.append("### 冻结分组（n / MAE）\n")
    gnames = list(groups)
    L.append("| arm | " + " | ".join(gnames) + " |")
    L.append("|" + "---|" * (len(gnames) + 1))
    for arm, r in records.items():
        L.append(f"| {arm} | " + " | ".join(
            f"{r['groups'][g]['mae']:.4f}" if r['groups'][g]['n'] else "n/a" for g in gnames) + " |")
    L.append(f"| n | " + " | ".join(str(int(m.sum())) for m in groups.values()) + " |")
    iv = summary.get("intervention")
    if iv and "error" not in iv:
        L.append("")
        L.append("### 固定权重干预（训练后的 D，规则换成 S，完整重前向）\n")
        L.append("| 量 | 值 |")
        L.append("|---|---|")
        for k, v in iv.items():
            if k == "note":
                continue
            L.append(f"| {k} | {v if not isinstance(v, float) else f'{v:.6g}'} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--arms", nargs="+", default=["D", "S"])
    args = p.parse_args(argv)
    y, unseen = _load_labels()
    groups = GROUPS(y, unseen)
    records = {}
    for arm in args.arms:
        path = RR.run_dir(arm, ARM_SEED[arm]) / "train.json"
        if path.exists():
            records[arm] = _run_record(arm, ARM_SEED[arm])
        else:
            print(f"[analyze] missing run for arm {arm}: {path}")
    if not records:
        print("[analyze] nothing to do")
        return 0

    b_npz = np.load(B_REFERENCE["predictions"])
    b_pred = b_npz["pred"].astype(np.float64)
    aerr = {}
    for arm, rec in records.items():
        pred = rec["pred"]
        rec["valid_mae_recomputed"] = float(np.abs(pred - y).mean())
        aerr[f"{arm}-s0"] = np.abs(pred - y)
        rec["mae_drop_row172"] = float(np.delete(aerr[f"{arm}-s0"], 172).mean())
        top10 = np.argsort(-aerr[f"{arm}-s0"])[:10]
        keep = np.ones(len(y), dtype=bool)
        keep[top10] = False
        rec["mae_drop_top10"] = float(aerr[f"{arm}-s0"][keep].mean())
        rec["groups"] = {g: {"n": int(m.sum()), "mae": float(aerr[f"{arm}-s0"][m].mean())}
                         for g, m in groups.items()}
    aerr["B-R0-s0"] = np.abs(b_pred - y)

    # paired bootstrap over valid rows (descriptive, fixed checkpoints)
    rng = np.random.default_rng(20261024)
    boot = rng.integers(0, len(y), size=(int(args.bootstrap), len(y)))
    contrasts = {}
    keys = sorted(aerr)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            d = aerr[a] - aerr[b]
            bs = d[boot].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            contrasts[f"{a} minus {b}"] = {
                "mean_delta_mae": float(d.mean()),
                "ci95_low": float(lo), "ci95_high": float(hi),
                "share_a_better": float((d < 0).mean()),
            }

    summary = {
        "round": RR.ROUND_ID,
        "official_test_loaded": False,
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in records.items()},
        "b_reference": {k: v for k, v in B_REFERENCE.items() if k != "predictions"},
        "d_minus_s": {
            "best_valid": records["D"]["best_valid"] - records["S"]["best_valid"],
            "w20": records["D"]["w20"] - records["S"]["w20"],
            "t100": records["D"]["t100"] - records["S"]["t100"],
            "final_valid": records["D"]["final_valid"] - records["S"]["final_valid"],
        } if {"D", "S"} <= set(records) else None,
        "vs_b_reference": {
            arm: {
                "best_valid_minus_B": rec["best_valid"] - B_REFERENCE["valid_mae"],
                "w20_minus_B": rec["w20"] - B_REFERENCE["w20"],
                "final_minus_B": rec["final_valid"] - B_REFERENCE["final"],
            } for arm, rec in records.items()
        },
        "contrasts_bootstrap": contrasts,
        "groups_definition": {
            "source": "zinc_generalization_audit_v1/analysis/per_molecule.csv (frozen)",
            "unseen": "r2_unseen_atom_frac > 0 (251)", "ordinary": "y > -3 (935)",
            "tail": "y <= -3 (65)", "unseen_ordinary": "225",
        },
        "w20_definition": "min over contiguous 20-epoch windows of the window median valid MAE",
        "t100_definition": "median of the last 100 epochs' valid MAE",
        "bootstrap_note": ("row bootstrap on the exposed valid split at fixed checkpoints; "
                           "describes sample sensitivity, not training repeatability"),
        "single_seed_boundary": (sorted(set(r["seed"] for r in records.values())) == [0]),
    }

    # intervention on the trained D
    try:
        summary["intervention"] = intervention("D", 0)
    except Exception as exc:  # pragma: no cover
        summary["intervention"] = {"error": repr(exc)}

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "summary.json", summary)
    if "intervention" in summary and "error" not in summary["intervention"]:
        C.write_json(ANALYSIS / "intervention.json", summary["intervention"])

    # curves
    import csv as _csv

    with open(ANALYSIS / "curves.csv", "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["arm", "seed", "epoch", "online_train_mae", "valid_mae", "lr", "seconds"])
        for arm, rec in records.items():
            for e in rec["curve"]:
                w.writerow([arm, rec["seed"], e["epoch"], e["online_train_mae"],
                            e["valid_mae"], e["lr"], e["seconds"]])

    # per-molecule valid
    with open(ANALYSIS / "per_molecule_valid.csv", "w", newline="") as fh:
        w = _csv.writer(fh)
        cols = ["row", "y_raw"] + [f"pred_{k}" for k in keys] + [f"ae_{k}" for k in keys]
        w.writerow(cols)
        preds = {}
        for k in keys:
            arm = k.split("-")[0]
            preds[k] = records[arm]["pred"] if arm in records else b_pred
        for i in range(len(y)):
            w.writerow([i, y[i]] + [preds[k][i] for k in keys] + [aerr[k][i] for k in keys])
    (ANALYSIS / "report_tables.md").write_text(_md(summary, records, aerr, groups, y))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
