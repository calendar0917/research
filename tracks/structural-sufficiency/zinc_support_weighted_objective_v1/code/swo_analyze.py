"""Analysis for the support-weighted-objective pilot (official valid only).

Reports, for every arm: overall-1000, ordinary / tail, the three radius-2 support
tiers, rings>=4, the anchor's fixed worst-10 and its 990-row complement,
``drop_row172``, median/P90 AE, top-10 error share, W20/T100, train@best/final,
GPU-hours, and paired bootstrap contrasts with a top-1-contributor decomposition.

Contrasts that matter:
  W_s0 - E_s0        same-session, same commit, only the loss weighting differs
  W_s0 - E_anchor    against the frozen historical DCC-L seed-0 checkpoint
  E_s0 - E_anchor    pure execution drift (init bit-identical), the noise floor

Official test is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import swo_common as S

N_BOOT = S.N_BOOT


def _run_record(tag: str) -> dict:
    d = S.RUNS / tag
    rec = S.read_json(d / "train.json")
    best = rec.get("best_valid") or rec.get("best") or {}
    swo = S.read_json(d / "swo.json") if (d / "swo.json").exists() else {}
    preds = {}
    for ckpt in ("best", "final"):
        preds[ckpt] = S.load_pred(d, ckpt)
    return {"tag": tag, "dir": d, "best_epoch": int(best["epoch"]),
            "best_valid": float(best["valid_mae"]),
            "W20": rec.get("W20"), "T100": rec.get("T100"),
            "parameters": int(rec["parameter_count"]),
            "gpu_hours": rec.get("gpu_hours"),
            "train_at_best": (rec.get("fixed_objects") or {}).get("best", {}).get("train_mae_eval_mode"),
            "train_at_final": (rec.get("fixed_objects") or {}).get("final", {}).get("train_mae_eval_mode"),
            "init": (rec.get("init") or {}).get("initial_state_sha256"),
            "arm": rec.get("arm") or swo.get("arm"),
            "seed": int(rec["seed"]),
            "weights": rec.get("sample_weighting"),
            "pred": preds, "rec": rec}


def _anchor() -> dict:
    art = S.read_json(S.E_ANCHOR / "artifacts.json")
    pred = S.load_pred(S.E_ANCHOR, "best")
    if pred is None:
        raise SystemExit("anchor predictions missing")
    fixed = art.get("fixed_objects") or {}
    return {"tag": "E_anchor(L_s0)", "dir": S.E_ANCHOR,
            "best_epoch": int(art["best_valid"]["epoch"]),
            "best_valid": float(art["best_valid"]["valid_mae"]),
            "W20": art.get("W20"), "T100": art.get("T100"),
            "parameters": int(art["parameter_count"]), "gpu_hours": art.get("gpu_hours"),
            "train_at_best": (fixed.get("best") or {}).get("train_mae_eval_mode"),
            "train_at_final": (fixed.get("final") or {}).get("train_mae_eval_mode"),
            "init": art.get("initial_state_sha256"), "arm": "E", "seed": 0,
            "weights": None, "pred": {"best": pred, "final": None}, "rec": None}


def _boot(delta, rng):
    idx = rng.integers(0, len(delta), size=(N_BOOT, len(delta)))
    m = delta[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", default="E_s0,W_s0")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    g = S.valid_groups()
    y, masks = g["y"], g["masks"]
    rng = np.random.default_rng(20261027)

    runs = {}
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        if (S.RUNS / tag / "train.json").exists():
            runs[tag] = _run_record(tag)
    anchor = _anchor()

    # fixed worst-10 defined by the frozen anchor (diagnostic only)
    ae_anchor = np.abs(anchor["pred"]["best"] - y)
    worst10 = np.argsort(-ae_anchor)[:10]
    masks = dict(masks)
    masks["anchor_worst10"] = np.zeros(len(y), dtype=bool)
    masks["anchor_worst10"][worst10] = True
    masks["anchor_complement_990"] = ~masks["anchor_worst10"]

    table, aerr = {}, {}
    for rec in list(runs.values()) + [anchor]:
        tag = rec["tag"]
        entry = {"arm": rec["arm"], "seed": rec["seed"], "parameters": rec["parameters"],
                 "gpu_hours": rec["gpu_hours"], "best_epoch": rec["best_epoch"],
                 "W20": rec["W20"], "T100": rec["T100"],
                 "weights": rec["weights"], "init": rec["init"]}
        for ckpt, pred in rec["pred"].items():
            if pred is None:
                continue
            ae = np.abs(pred - y)
            aerr[(tag, ckpt)] = ae
            entry[ckpt] = {
                "valid_mae": float(ae.mean()),
                "groups": {k: float(ae[m].mean()) for k, m in masks.items() if m.any()},
                "n": {k: int(m.sum()) for k, m in masks.items()},
                "median_ae": float(np.median(ae)), "p90_ae": float(np.quantile(ae, 0.90)),
                "top10_share": float(np.sort(ae)[-10:].sum() / ae.sum()),
                "row172_ae": float(ae[172]),
                "train_mae": (rec["train_at_best"] if ckpt == "best" else rec["train_at_final"]),
            }
        table[tag] = entry

    # paired contrasts
    contrasts = {}
    pairs = [("W_s0", "E_s0"), ("W_s0", "E_anchor(L_s0)"), ("E_s0", "E_anchor(L_s0)"),
             ("E_s1", "W_s1"), ("W_s1", "E_s1"), ("W_s1", "W_s0"), ("E_s1", "E_s0")]
    for a, b in pairs:
        if a not in aerr and a not in table:
            continue
        for ckpt in ("best", "final"):
            if (a, ckpt) not in aerr or (b, ckpt) not in aerr:
                continue
            d = aerr[(a, ckpt)] - aerr[(b, ckpt)]
            lo, hi = _boot(d, rng)
            order = np.argsort(-np.maximum(aerr[(a, ckpt)], aerr[(b, ckpt)]))[:10]
            keep = np.ones(len(d), dtype=bool)
            keep[order] = False
            top1 = int(np.argmax(np.abs(d)))
            contrasts[f"{a} − {b}|{ckpt}"] = {
                "first": a, "second": b, "checkpoint": ckpt,
                "delta_mae": float(d.mean()), "ci95": [lo, hi],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "first_better_fraction": float(np.mean(d < 0)),
                "delta_by_group": {k: float(d[m].mean()) for k, m in masks.items() if m.any()},
                "delta_drop_top10": float(d[keep].mean()),
                "delta_drop_row172": float(np.delete(d, 172).mean()),
                "top1_row": top1, "top1_delta": float(d[top1]),
                "delta_drop_top1": float(np.delete(d, top1).mean()),
                "n_rows_improved_gt_0.01": int((d < -0.01).sum()),
                "n_rows_worse_gt_0.01": int((d > 0.01).sum()),
            }

    # mechanistic diagnostic: does W help the molecules it up-weighted?
    mech = {}
    if ("W_s0", "best") in aerr and ("E_s0", "best") in aerr:
        import ad_data as D, bjm_common as BC
        vc = D.load_cache("valid")
        sup = BC.radius2_support(BC.molecules_from_cache(D.load_cache("train")),
                                 BC.molecules_from_cache(vc))
        low4 = np.asarray(sup["low4_frac"], dtype=np.float64)
        unseen = np.asarray(sup["unseen_frac"], dtype=np.float64)
        d = aerr[("W_s0", "best")] - aerr[("E_s0", "best")]
        q = np.quantile(low4, [1 / 3, 2 / 3])
        for name, m in {"low4_tertile_1": low4 <= q[0], "low4_tertile_2": (low4 > q[0]) & (low4 <= q[1]),
                        "low4_tertile_3": low4 > q[1],
                        "unseen_frac_0": unseen == 0.0, "unseen_frac_gt_0": unseen > 0.0}.items():
            mech[name] = {"n": int(m.sum()), "delta_mae": float(d[m].mean())}
        mech["corr_deltaAE_low4"] = float(np.corrcoef(d, low4)[0, 1])
    # valid-curve noise / selection noise
    curves = {}
    for tag, rec in runs.items():
        c = np.asarray([e["valid_mae"] for e in rec["rec"]["curve"]], dtype=np.float64)
        curves[tag] = {"sd_last_100_epochs": float(c[-100:].std()),
                       "min": float(c.min()), "argmin_epoch": int(c.argmin() + 1),
                       "final": float(c[-1]),
                       "min_minus_final": float(c.min() - c[-1]),
                       "n_epochs_below_final_minus_0.005": int((c < c[-1] - 0.005).sum())}

    payload = {
        "round": S.ROUND_ID, "written": S.now(), "n_valid": int(len(y)),
        "metric": "raw-y MAE", "official_test_loaded": False,
        "main_table": table, "contrasts": contrasts, "mechanism_diagnostics": mech,
        "valid_curve_noise": curves, "group_sizes": {k: int(m.sum()) for k, m in masks.items()},
        "weights": S.read_json(S.CONFIG)["weights_frozen"],
        "support_definition": g["support"],
    }
    out = Path(args.out) if args.out else S.ANALYSIS / "summary_swo.json"
    S.write_json(out, payload)

    # markdown
    lines = ["# SWO v1 — official-valid tables", "",
             f"written {payload['written']} · n_valid=1000 · official test never loaded", "",
             "## Main table (raw-y MAE)", "",
             "| arm | params | best@ep | overall | ordinary | tail | r2 all-sup | r2 low-sup | r2 unseen | rings>=4 | anchor worst-10 | complement 990 | drop172 | median | P90 | top10 share | train@best | GPU·h |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, t in table.items():
        b = t.get("best")
        if not b:
            continue
        gr = b["groups"]
        lines.append(
            f"| {tag} | {t['parameters']} | {b['valid_mae']:.6f}@{t['best_epoch']} "
            f"| {b['valid_mae']:.6f} | {gr['ordinary_y_gt_-3']:.6f} | {gr['tail_y_le_-3']:.6f} "
            f"| {gr['radius2_all_supported']:.6f} | {gr['radius2_low_support']:.6f} "
            f"| {gr['radius2_any_unseen']:.6f} | {gr['rings_ge4']:.6f} "
            f"| {gr['anchor_worst10']:.6f} | {gr['anchor_complement_990']:.6f} "
            f"| {gr['drop_row172']:.6f} | {b['median_ae']:.6f} | {b['p90_ae']:.6f} "
            f"| {b['top10_share']:.3f} | "
            f"{(b['train_mae'] if b['train_mae'] is not None else float('nan')):.6f} "
            f"| {(t['gpu_hours'] or float('nan')):.3f} |")
    lines += ["", "## Paired contrasts (negative = first arm better)", "",
              "| contrast | ckpt | ΔMAE | 95% CI | excludes 0 | ordinary | tail | r2 low-sup | r2 unseen | rings>=4 | anchor worst-10 | drop172 | drop top-1 (row) |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, c in contrasts.items():
        gr = c["delta_by_group"]
        lines.append(
            f"| {name} | {c['checkpoint']} | {c['delta_mae']:+.6f} "
            f"| [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | {c['excludes_zero']} "
            f"| {gr['ordinary_y_gt_-3']:+.6f} | {gr['tail_y_le_-3']:+.6f} "
            f"| {gr['radius2_low_support']:+.6f} | {gr['radius2_any_unseen']:+.6f} "
            f"| {gr['rings_ge4']:+.6f} | {gr['anchor_worst10']:+.6f} "
            f"| {c['delta_drop_row172']:+.6f} | {c['delta_drop_top1']:+.6f} (row {c['top1_row']}) |")
    if mech:
        lines += ["", "## Does W help the molecules it up-weighted? (diagnostic)", "",
                  "| valid subset | n | ΔMAE (W − E, negative better) |", "|---|---:|---:|"]
        for k, v in mech.items():
            if isinstance(v, dict):
                lines.append(f"| {k} | {v['n']} | {v['delta_mae']:+.6f} |")
        lines.append(f"| corr(ΔAE, low4) | — | {mech['corr_deltaAE_low4']:+.4f} |")
    if curves:
        lines += ["", "## Valid-curve noise and checkpoint-selection leverage", "",
                  "| run | sd(last 100 ep) | min | @ep | final | min−final | epochs < final−0.005 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for tag, c in curves.items():
            lines.append(f"| {tag} | {c['sd_last_100_epochs']:.5f} | {c['min']:.6f} "
                         f"| {c['argmin_epoch']} | {c['final']:.6f} "
                         f"| {c['min_minus_final']:+.5f} | {c['n_epochs_below_final_minus_0.005']} |")
    md = out.with_name(f"tables_{out.stem.split('_')[-1]}.md")
    md.write_text("\n".join(lines) + "\n")
    print(f"[swo-analyze] {len(table)} arms, {len(contrasts)} contrasts -> {S.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
