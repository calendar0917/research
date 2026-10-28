"""Matched valid-1000 analysis for non-local runs vs frozen LCRC-A seed 0.

No test access.  Primary metric is full raw-y MAE; every secondary metric is
reported alongside it and never replaces it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

import nl_common as N

A_RUN = N.LCRC_DIR / "runs" / "lcrcA_s0"
A_TABLE = N.LCRC_DIR / "analysis" / "per_molecule_valid.csv"


def load_rows():
    rows = list(csv.DictReader(open(A_TABLE)))
    y = np.asarray([float(r["y"]) for r in rows], dtype=np.float64)
    unseen = np.asarray([float(r["r2_unseen_atom_frac"]) > 0 for r in rows])
    rings = np.asarray([int(r["n_rings"]) for r in rows])
    idx = np.asarray([int(r["row"]) for r in rows])
    order = np.argsort(idx)
    assert np.all(idx[order] == np.arange(len(idx)))
    return {"y": y, "unseen": unseen, "rings": rings, "row": idx}


def load_pred(path):
    z = np.load(path)
    rows = z["row"]
    pred = z["pred"].astype(np.float64)
    out = np.zeros(len(rows))
    out[rows] = pred
    return out


def metrics(pred, y, unseen, rings):
    ae = np.abs(pred - y)
    groups = {
        "all": np.ones(len(y), dtype=bool),
        "ordinary": y > -3.0,
        "tail": y <= -3.0,
        "r2_unseen": unseen,
        "r2_unseen_ordinary": unseen & (y > -3.0),
        "seen_ordinary": (~unseen) & (y > -3.0),
        "ring_rich": rings >= 4,
        "ring_rich_unseen": (rings >= 4) & unseen,
    }
    worst10 = np.argsort(-ae)[:10]
    row172 = np.where(np.abs(y + 20.340498) < 1e-6)[0]
    if len(row172) == 0:
        row172 = np.asarray([172])
    keep172 = np.ones(len(y), dtype=bool); keep172[row172] = False
    keepw10 = np.ones(len(y), dtype=bool); keepw10[worst10] = False
    out = {"full_mae": float(ae.mean())}
    for k, m in groups.items():
        out[k] = {"n": int(m.sum()), "mae": float(ae[m].mean()),
                  "ae_share": float(ae[m].sum() / ae.sum())}
    out["drop_row172_mae"] = float(ae[keep172].mean())
    out["drop_fixed_worst10_mae"] = float(ae[keepw10].mean())
    out["fixed_worst10_rows"] = worst10.tolist()
    out["row172_row"] = int(row172[0])
    out["row172_ae"] = float(ae[row172].mean())
    out["median_ae"] = float(np.median(ae))
    out["top10_ae_share"] = float(ae[worst10].sum() / ae.sum())
    return out, ae


def bootstrap_delta(pred_a, pred_b, y, n_boot=10000, seed=20261029):
    ae_a = np.abs(pred_a - y)
    ae_b = np.abs(pred_b - y)
    d = ae_a - ae_b          # negative => a better
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(1)
    return {
        "delta_mae": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "first_better_rate": float((d < 0).mean()),
        "median_delta_ae": float(np.median(d)),
    }


def curve_summary(path):
    rows = list(csv.DictReader(open(path)))
    ep = np.asarray([int(r["epoch"]) for r in rows])
    v = np.asarray([float(r["valid_mae"]) for r in rows])
    tr = np.asarray([float(r["online_train_mae"]) for r in rows])
    out = {"epochs": len(ep), "best_epoch": int(ep[int(np.argmin(v))]),
           "best_valid": float(v.min()), "final_valid": float(v[-1]),
           "T100": float(np.median(v[-100:])), "train_final": float(tr[-1])}
    for e in (100, 150, 200, 250, 300, 350, 400, 450, 500, 600):
        if len(ep) >= e:
            out[f"valid@{e}"] = float(v[e - 1])
            out[f"train@{e}"] = float(tr[e - 1])
    if len(v) >= 50:
        out["slope_last50"] = float(np.polyfit(np.arange(50), v[-50:], 1)[0])
    out["curve"] = {"epoch": ep.tolist(), "valid": v.tolist(), "train": tr.tolist()}
    return out


def analyse(run_dirs, ref_a=None):
    ref_a = ref_a or A_RUN
    meta = load_rows()
    y, unseen, rings = meta["y"], meta["unseen"], meta["rings"]
    a_best = load_pred(ref_a / "predictions" / "valid_best.npz")
    a_final = load_pred(ref_a / "predictions" / "valid_final.npz")
    a_m, a_ae = metrics(a_best, y, unseen, rings)
    ref = {"run": str(ref_a), "best": a_m,
           "curve": curve_summary(ref_a / "curve.csv")}
    fixed_worst10 = set(a_m["fixed_worst10_rows"])
    row172 = a_m["row172_row"]
    keep172 = np.ones(len(y), dtype=bool); keep172[row172] = False
    keepw10 = np.ones(len(y), dtype=bool)
    for r in fixed_worst10:
        keepw10[r] = False
    out = {"reference_A_s0": ref, "runs": {}}
    for d in run_dirs:
        d = Path(d)
        tj = d / "train.json"
        if not tj.exists():
            out["runs"][str(d)] = {"status": "incomplete", "has_train_json": False}
            continue
        t = N.L.read_json(tj)
        pb = d / "predictions" / "valid_best.npz"
        pf = d / "predictions" / "valid_final.npz"
        p_best = load_pred(pb) if pb.exists() else None
        p_final = load_pred(pf) if pf.exists() else None
        rec = {"status": t.get("status"), "run": t.get("run"), "head": t.get("head"),
               "mode": t.get("mode"), "seed": t.get("seed"), "parameters": t.get("parameters"),
               "best": t.get("best"), "best_valid_mae_eval": t.get("best_valid_mae_eval"),
               "final_valid_mae_eval": t.get("final_valid_mae_eval"),
               "best_train_mae_eval": t.get("best_train_mae_eval"),
               "gpu_hours": t.get("gpu_hours"), "git_head": t.get("git_head"),
               "step_logs": t.get("step_logs", [])}
        rec["train_curve"] = curve_summary(d / "curve.csv") if (d / "curve.csv").exists() else None
        if p_best is not None:
            m, ae = metrics(p_best, y, unseen, rings)
            m["delta_vs_A"] = bootstrap_delta(p_best, a_best, y)
            m["delta_vs_A_drop172"] = float(ae[keep172].mean() - a_ae[keep172].mean())
            m["delta_vs_A_dropfixed10"] = float(ae[keepw10].mean() - a_ae[keepw10].mean())
            # group-level paired deltas
            grp = {"ordinary": y > -3, "r2_unseen_ordinary": unseen & (y > -3),
                   "seen_ordinary": (~unseen) & (y > -3), "tail": y <= -3}
            m["group_deltas"] = {
                k: float((ae[v] - a_ae[v]).mean()) for k, v in grp.items()
            }
            m["wins_against_A"] = float((ae < a_ae).mean())
            rec["best"] = m
        if p_final is not None:
            mf, _ = metrics(p_final, y, unseen, rings)
            mf["delta_vs_A"] = bootstrap_delta(p_final, a_final, y)
            rec["final"] = mf
        out["runs"][str(d)] = rec
    return out


def _f(x, fmt=".6f"):
    return "—" if x is None else format(float(x), fmt)


def markdown(out, path):
    lines = ["# Non-local tree — matched valid-1000 table", "",
             f"Reference A_s0 best valid: {out['reference_A_s0']['best']['full_mae']:.6f}", "",
             "| run | mode | seed | best valid | Δ vs A | Δ drop172 | Δ dropfixed10 | ordinary Δ | "
             "unseen-ord Δ | seen-ord Δ | tail Δ | best ep | final | T100 | params | GPU·h |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d, r in out["runs"].items():
        if "best" not in r or "full_mae" not in r.get("best", {}):
            lines.append(f"| {Path(d).name} | — | — | incomplete | | | | | | | | | | | | |")
            continue
        m, c = r["best"], r.get("train_curve") or {}
        g = m["group_deltas"]
        lines.append(
            f"| {r.get('run', Path(d).name)} | {r.get('mode')} | {r.get('seed')} | "
            f"{m['full_mae']:.6f} | {m['delta_vs_A']['delta_mae']:+.6f} | "
            f"{m['delta_vs_A_drop172']:+.6f} | {m['delta_vs_A_dropfixed10']:+.6f} | "
            f"{g['ordinary']:+.6f} | {g['r2_unseen_ordinary']:+.6f} | "
            f"{g['seen_ordinary']:+.6f} | {g['tail']:+.6f} | "
            f"{c.get('best_epoch')} | {_f(r.get('final_valid_mae_eval'))} | "
            f"{_f(c.get('T100'))} | {r.get('parameters')} | {_f(r.get('gpu_hours'), '.4f')} |"
        )
    lines += ["", "Δ is (run − A_s0); negative is better. Row bootstrap CIs are in the JSON.", ""]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=str(N.ANALYSIS / "main_table"))
    args = ap.parse_args()
    runs = args.runs or sorted(glob.glob(str(N.RUNS / "*")))
    out = analyse(runs)
    out["official_test_loaded"] = False
    N.L.write_json(Path(args.out).with_suffix(".json"), out)
    markdown(out, Path(args.out).with_suffix(".md"))
    print(Path(args.out).with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
