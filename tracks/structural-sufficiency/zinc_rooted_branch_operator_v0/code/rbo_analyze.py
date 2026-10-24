"""Score the frozen RBO checkpoint against the frozen P3-minus reference.

Reads only train-split artifacts: M (1000 target-key-free monitor rows),
Q_target (350) and Q_background (650) from
``zinc_composition_transfer_v1/cache/support_sets.npz``.  The P3-minus
reference is the frozen ``sup_minus_s0/best_monitor.pt`` of that round (same
7,549 training molecules, same recipe, same encoder init), recomputed here from
its checkpoint so the comparison is per-molecule paired.

Also runs the mechanism probes (three-body term off, relation term off, typed
histogram off) on the same frozen checkpoint.  No test split is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_V2 = _SSUF / "zinc_recipe_trunk_v2" / "code"
_CT = _SSUF / "zinc_composition_transfer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_CT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import rbo_model as R  # noqa: E402
import zct_train as ZT  # noqa: E402

TASK = _TASK
OUT = TASK / "analysis"
CT = _SSUF / "zinc_composition_transfer_v1"


def predict(model, rows, cache, prep, kind, device="cpu", batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            chunk = np.asarray(rows)[s:s + batch]
            if kind == "rbo":
                b = R.collate(chunk, cache, prep, device)
            else:
                b = T1._to_device(T1.collate(cache, chunk, "P3N"), device)
            out.append(model(b).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0).astype(np.float64)


def compare(pred_a, pred_b, y, n_boot=10000, seed=20261024):
    a = np.abs(pred_a - y)
    b = np.abs(pred_b - y)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(1)
    order = np.argsort(-np.abs(d))
    drop = order[:5]
    keep = np.ones(len(d), dtype=bool)
    keep[drop] = False
    better = float((d < 0).mean())
    # sign test (two-sided, normal approximation)
    n_pos = int((d < 0).sum())
    n_eff = int((d != 0).sum())
    if n_eff:
        z = (n_pos - n_eff / 2) / np.sqrt(n_eff * 0.25)
        from math import erfc
        p = erfc(abs(z) / np.sqrt(2))
    else:
        p = 1.0
    return {
        "n": int(len(d)),
        "mae_a": float(a.mean()), "mae_b": float(b.mean()),
        "delta_mean": float(d.mean()),
        "delta_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "delta_median": float(np.median(d)),
        "frac_better": better,
        "sign_test_p": float(p),
        "trimmed_mean_delta_drop5": float(d[keep].mean()),
        "drop5_rows": [int(i) for i in drop],
        "drop5_delta": float(d[drop].mean()),
        "p90_ae_a": float(np.percentile(a, 90)), "p90_ae_b": float(np.percentile(b, 90)),
        "max_ae_a": float(a.max()), "max_ae_b": float(b.max()),
        "n_ae_gt_0p5_a": int((a > 0.5).sum()), "n_ae_gt_0p5_b": int((b > 0.5).sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbo", default=str(TASK / "runs" / "rbo_minus_s0" / "best_monitor.pt"))
    ap.add_argument("--p3", default=str(CT / "runs" / "sup_minus_s0" / "best_monitor.pt"))
    ap.add_argument("--p3-final", default=str(CT / "runs" / "sup_minus_s0" / "final.pt"))
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    sup = dict(np.load(CT / "cache" / "support_sets.npz"))
    manifest = C.read_json(CT / "cache" / "support_manifest.json")
    cache = D.load_cache("train")
    y_all = np.asarray(cache["labels"], dtype=np.float64)
    norm = D.load_norm()
    prep = torch.load(TASK / "cache" / "rbo_train.pt", map_location="cpu", weights_only=False)
    groups = {k: np.asarray(sup[k], dtype=np.int64)
              for k in ("monitor", "q_target", "q_background")}

    # ---- frozen RBO checkpoint -------------------------------------------
    ck = torch.load(args.rbo, map_location="cpu", weights_only=False)
    rbo, rbo_info = R.build_model(norm, 0, "cpu")
    rbo.load_state_dict(ck["state_dict"])
    rbo.eval()
    rbo_preds = {k: predict(rbo, rows, cache, prep, "rbo") for k, rows in groups.items()}
    rbo_mae = {k: float(np.abs(p - y_all[rows]).mean()) for k, (p, rows) in
               ((k, (rbo_preds[k], groups[k])) for k in groups)}

    # ---- frozen P3-minus reference ---------------------------------------
    p3, _ = ZT.build_model(norm, 0, "cpu")
    st = torch.load(args.p3, map_location="cpu", weights_only=False)
    p3.load_state_dict(st["state_dict"])
    p3_preds = {k: predict(p3, rows, cache, prep, "p3") for k, rows in groups.items()}
    p3_mae = {k: float(np.abs(p - y_all[rows]).mean()) for k, (p, rows) in
              ((k, (p3_preds[k], groups[k])) for k in groups)}
    p3_epoch = int(st.get("epoch", -1))
    p3_ref = {"recorded_monitor_mae": 0.0804237261582166,
              "recomputed_monitor_mae": p3_mae["monitor"],
              "match": abs(p3_mae["monitor"] - 0.0804237261582166) < 2e-6}

    # optional epoch-600 P3-minus (no selection use)
    p3f_mae = {}
    if Path(args.p3_final).exists():
        p3f, _ = ZT.build_model(norm, 0, "cpu")
        stf = torch.load(args.p3_final, map_location="cpu", weights_only=False)
        p3f.load_state_dict(stf["state_dict"])
        p3f.eval()
        p3f_mae = {k: float(np.abs(predict(p3f, rows, cache, prep, "p3") - y_all[rows]).mean())
                   for k, rows in groups.items()}

    # ---- eval-mode fit error at the selected checkpoint -------------------
    fit_rows_eval = np.asarray(sup["fit_minus"], dtype=np.int64)
    fit_pred = predict(rbo, fit_rows_eval, cache, prep, "rbo")
    fit_mae = float(np.abs(fit_pred - y_all[fit_rows_eval]).mean())

    # ---- paired comparisons ---------------------------------------------
    comps = {}
    for k in groups:
        yy = y_all[groups[k]]
        comps[f"rbo_minus_p3_{k}"] = compare(rbo_preds[k], p3_preds[k], yy)

    # ---- strata by radius-2-unseen atom fraction (vs the fit_minus pool) --
    kz = np.load(CT / "cache" / "keys.npz")
    k2 = np.asarray(kz["atom_key2"])
    ap = np.asarray(cache["atom_ptr"])
    fit_rows = np.asarray(sup["fit_minus"], dtype=np.int64)
    k2_fit = {int(x) for r in fit_rows for x in k2[ap[r]:ap[r + 1]]}
    def unseen_frac(rows):
        return np.asarray([
            float(np.mean([int(x) not in k2_fit for x in k2[ap[r]:ap[r + 1]]]))
            if ap[r + 1] > ap[r] else 0.0
            for r in rows
        ])
    unseen = {k: unseen_frac(rows) for k, rows in groups.items()}
    ust = {}
    for label, lo, hi in (("<0.05", -1, 0.05), ("0.05-0.15", 0.05, 0.15),
                          ("0.15-0.30", 0.15, 0.30), (">=0.30", 0.30, 1.01)):
        entry = {}
        for k in groups:
            sel = np.flatnonzero((unseen[k] > lo) & (unseen[k] <= hi))
            if len(sel) == 0:
                continue
            yy = y_all[groups[k][sel]]
            entry[k] = {
                "n": int(len(sel)),
                "rbo_mae": float(np.abs(rbo_preds[k][sel] - yy).mean()),
                "p3_mae": float(np.abs(p3_preds[k][sel] - yy).mean()),
                "delta": float((np.abs(rbo_preds[k][sel] - yy)
                                - np.abs(p3_preds[k][sel] - yy)).mean()),
            }
        ust[label] = entry

    # ---- strata by target-key training support --------------------------
    keys = [int(x) for x in manifest["target_keys"]]
    key_support = {int(k): int(v) for k, v in manifest["key_support_train"].items()}
    kz = np.load(CT / "cache" / "keys.npz")
    koff = np.asarray(kz["mol_k2_off"])
    kmol = np.asarray(kz["mol_k2"])
    mol_keys = {}
    for m in groups["q_target"].tolist():
        mol_keys[m] = set(kmol[int(koff[m]):int(koff[m + 1])].tolist()) & set(keys)
    strata = {}
    for lo, hi, label in ((4, 7, "support_4_7"), (8, 15, "support_8_15"),
                          (16, 25, "support_16_25")):
        ksel = {k for k in keys if lo <= key_support.get(k, -1) <= hi}
        rows = np.asarray([m for m in groups["q_target"].tolist() if mol_keys[m] & ksel])
        if len(rows) == 0:
            continue
        pos = {int(m): i for i, m in enumerate(groups["q_target"].tolist())}
        ii = np.asarray([pos[int(m)] for m in rows])
        yy = y_all[rows]
        strata[label] = {
            "n_molecules": int(len(rows)),
            "n_keys": int(len(ksel)),
            "rbo_mae": float(np.abs(rbo_preds["q_target"][ii] - yy).mean()),
            "p3_mae": float(np.abs(p3_preds["q_target"][ii] - yy).mean()),
            "delta": float((np.abs(rbo_preds["q_target"][ii] - yy)
                            - np.abs(p3_preds["q_target"][ii] - yy)).mean()),
        }
    per_key = []
    pos = {int(m): i for i, m in enumerate(groups["q_target"].tolist())}
    for k in keys:
        ms = [m for m in groups["q_target"].tolist() if k in mol_keys[m]]
        ii = np.asarray([pos[m] for m in ms])
        if len(ii) == 0:
            continue
        yy = y_all[np.asarray(ms)]
        per_key.append({
            "key": int(k), "n_query": int(len(ms)), "support_train": key_support.get(k),
            "rbo": float(np.abs(rbo_preds["q_target"][ii] - yy).mean()),
            "p3": float(np.abs(p3_preds["q_target"][ii] - yy).mean()),
        })
    pk = np.asarray([[r["rbo"], r["p3"]] for r in per_key])
    nq = np.asarray([r["n_query"] for r in per_key], dtype=np.float64)
    key_summary = {
        "n_keys": int(len(per_key)),
        "keys_rbo_better": int((pk[:, 0] < pk[:, 1]).sum()),
        "unweighted_mean_key_delta": float((pk[:, 0] - pk[:, 1]).mean()),
        "median_key_delta": float(np.median(pk[:, 0] - pk[:, 1])),
        # unweighted key means give singleton keys the same weight as a key
        # with 14 query molecules; the n-weighted number is the honest one
        "query_weighted_mean_key_delta": float(
            ((pk[:, 0] - pk[:, 1]) * nq).sum() / nq.sum()),
        "n_query_min_median_max": [int(nq.min()), float(np.median(nq)), int(nq.max())],
    }

    # ---- mechanism probes (frozen checkpoint, no retraining) -------------
    probes = {}
    with torch.no_grad():
        for name, setter in (
            ("three_body_off", lambda: setattr(rbo, "three_body_scale", torch.tensor(0.0))),
            ("relation_off", lambda: setattr(rbo, "rel_scale", torch.tensor(0.0))),
        ):
            rbo.three_body_scale.fill_(1.0)
            rbo.rel_scale.fill_(1.0)
            setter()
            pr = {k: predict(rbo, rows, cache, prep, "rbo") for k, rows in groups.items()}
            probes[name] = {
                "monitor_mae": float(np.abs(pr["monitor"] - y_all[groups["monitor"]]).mean()),
                "q_target_mae": float(np.abs(pr["q_target"] - y_all[groups["q_target"]]).mean()),
                "q_background_mae": float(np.abs(pr["q_background"] - y_all[groups["q_background"]]).mean()),
                "mean_abs_pred_change_monitor": float(np.abs(
                    pr["monitor"] - rbo_preds["monitor"]).mean()),
            }
        rbo.three_body_scale.fill_(1.0)
        rbo.rel_scale.fill_(1.0)
        # typed histogram off: zero the branch histogram input
        old_collate = R.collate
        def collate_nohist(rows, cache_, prep_, device):
            b = old_collate(rows, cache_, prep_, device)
            b["hist"] = torch.zeros_like(b["hist"])
            return b
        R.collate = collate_nohist
        try:
            pr = {k: predict(rbo, rows, cache, prep, "rbo") for k, rows in groups.items()}
        finally:
            R.collate = old_collate
        probes["typed_hist_off"] = {
            "monitor_mae": float(np.abs(pr["monitor"] - y_all[groups["monitor"]]).mean()),
            "q_target_mae": float(np.abs(pr["q_target"] - y_all[groups["q_target"]]).mean()),
            "q_background_mae": float(np.abs(pr["q_background"] - y_all[groups["q_background"]]).mean()),
            "mean_abs_pred_change_monitor": float(np.abs(pr["monitor"] - rbo_preds["monitor"]).mean()),
        }

    summary = {
        "rbo": {
            "checkpoint": args.rbo, "epoch": int(ck["epoch"]),
            "monitor_mae_at_best": float(ck.get("monitor_mae", np.nan)),
            "sha256": C.sha256_file(args.rbo), "parameters": rbo_info["parameters"],
            "mae": rbo_mae, "fit_minus_eval_mae": fit_mae,
        },
        "p3_minus": {"checkpoint": args.p3, "epoch": p3_epoch, "mae": p3_mae,
                     "reference_check": p3_ref},
        "p3_minus_final_600": p3f_mae,
        "comparisons": comps,
        "unseen_fraction": {k: {"mean": float(v.mean()),
                                "n_gt_0.15": int((v > 0.15).sum())}
                            for k, v in unseen.items()},
        "strata_unseen_fraction": ust,
        "strata_q_target": strata,
        "per_key": per_key,
        "key_summary": key_summary,
        "probes": probes,
    }
    C.write_json(OUT / "summary.json", summary)
    C.write_json(OUT / "per_key.json", per_key)
    # per-molecule csv
    lines = ["group,row,y,rbo_pred,p3_pred,rbo_ae,p3_ae,delta"]
    for k, rows in groups.items():
        for i, r in enumerate(rows.tolist()):
            ae_r = abs(rbo_preds[k][i] - y_all[r])
            ae_p = abs(p3_preds[k][i] - y_all[r])
            lines.append(f"{k},{r},{y_all[r]:.6f},{rbo_preds[k][i]:.6f},{p3_preds[k][i]:.6f},"
                         f"{ae_r:.6f},{ae_p:.6f},{ae_r - ae_p:.6f}")
    (OUT / "per_molecule.csv").write_text("\n".join(lines) + "\n")

    md = ["# RBO v0 — frozen-checkpoint comparison (train-only evaluation sets)", "",
          f"RBO checkpoint: `{args.rbo}` (epoch {ck['epoch']}), parameters {rbo_info['parameters']}",
          f"P3-minus reference: `{args.p3}` (epoch {p3_epoch})", "",
          "| set | n | RBO MAE | P3-minus MAE | Δ (RBO−P3) | 95% CI | median Δ | better | sign p |",
          "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for k in groups:
        c = comps[f"rbo_minus_p3_{k}"]
        md.append(f"| {k} | {c['n']} | {c['mae_a']:.6f} | {c['mae_b']:.6f} | "
                  f"{c['delta_mean']:+.6f} | [{c['delta_ci95'][0]:+.6f}, {c['delta_ci95'][1]:+.6f}] | "
                  f"{c['delta_median']:+.6f} | {c['frac_better']:.3f} | {c['sign_test_p']:.4f} |")
    if p3f_mae:
        md += ["", "P3-minus epoch-600 (secondary, no selection): " +
               ", ".join(f"{k} {v:.6f}" for k, v in p3f_mae.items())]
    md += ["", "| r2-unseen atom fraction | set | n | RBO | P3-minus | Δ |",
           "|---|---|---:|---:|---:|---:|"]
    for lab, entry in ust.items():
        for k, v in entry.items():
            md.append(f"| {lab} | {k} | {v['n']} | {v['rbo_mae']:.6f} | {v['p3_mae']:.6f} | "
                      f"{v['delta']:+.6f} |")
    md += ["", "| stratum | n mol | n keys | RBO | P3-minus | Δ |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in strata.items():
        md.append(f"| {k} | {v['n_molecules']} | {v['n_keys']} | {v['rbo_mae']:.6f} | "
                  f"{v['p3_mae']:.6f} | {v['delta']:+.6f} |")
    md += ["", f"key level: {key_summary['keys_rbo_better']}/{key_summary['n_keys']} keys better, "
           f"unweighted mean Δ {key_summary['unweighted_mean_key_delta']:+.6f}, "
           f"query-weighted mean Δ {key_summary['query_weighted_mean_key_delta']:+.6f}, "
           f"median Δ {key_summary['median_key_delta']:+.6f}",
           "", "| probe (frozen checkpoint) | M MAE | Q_target MAE | Q_background MAE | mean |Δpred| on M |",
           "|---|---:|---:|---:|---:|",
           f"| full | {rbo_mae['monitor']:.6f} | {rbo_mae['q_target']:.6f} | {rbo_mae['q_background']:.6f} | – |"]
    for k, v in probes.items():
        md.append(f"| {k} | {v['monitor_mae']:.6f} | {v['q_target_mae']:.6f} | "
                  f"{v['q_background_mae']:.6f} | {v['mean_abs_pred_change_monitor']:.6f} |")
    (OUT / "main_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
