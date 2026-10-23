"""Final tables, machine-readable summary and the per-molecule deliverable."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import zst_common as Z  # noqa: E402
import zst_head as H  # noqa: E402
import ad_common as C  # noqa: E402


def arms_of(rec, kind, tag):
    return [e[tag] for e in rec["arms"][kind]]


def cell(rec, qname, kind, tag):
    entries = rec["arms"][kind]
    ms = [e[tag][qname] for e in entries]
    return {
        "mae_mean": float(np.mean([m["mae_corrected"] for m in ms])),
        "mae_by_init": [float(m["mae_corrected"]) for m in ms],
        "delta_mean": float(np.mean([m["delta"] for m in ms])),
        "delta_by_init": [float(m["delta"]) for m in ms],
        "ci_by_init": [m["boot"]["molecule_ci"] for m in ms],
        "key_ci_by_init": [m["boot"].get("key_cluster_ci") for m in ms],
        "n_key_clusters": ms[0]["boot"].get("n_key_clusters"),
        "frac_improved_mean": float(np.mean([m["frac_improved"] for m in ms])),
        "mean_abs_correction": float(np.mean([m["mean_abs_correction"] for m in ms])),
        "trimmed_delta_mean": float(np.mean([m["trimmed_delta"] for m in ms])),
        "corr_with_residual": float(np.mean([m["corr_signed_with_residual"] for m in ms])),
        "selected_step": [e["steps_used"] for e in entries],
        "dev_mae_raw": [e.get("dev_mae") for e in entries],
        "dev_mae_trim5": [e.get("dev_mae_trim5") for e in entries],
        "fit_mae_cfit_gated": [e["fit_mae_cfit_gated"] for e in entries],
    }


def summarise(records: dict, stage: str) -> dict:
    out = {"stage": stage, "seeds": {}}
    for seed_s, rec in records.items():
        s = {"n_fit": rec.get("n_fit"), "base_ckpt_sha256": rec.get("base_ckpt_sha256"),
             "support_hash": rec.get("support_hash"),
             "c_fit": rec.get("c_fit") or rec.get("c_fit"),
             "c_dev": dict(rec.get("c_dev") or {}, base_mae_trim5=rec.get("c_dev", {}).get("base_mae_trim5")),
             "residual_gated": rec.get("c_fit_residual_gated"),
             "wd_selection": rec.get("wd_selection"), "cells": {},
             "baselines": rec.get("baselines", {})}
        for qname in rec["base_metrics"]:
            b = rec["base_metrics"][qname]
            c = {"seed": int(seed_s), "package": qname, "n": b["n"],
                 "n_gated": b.get("n_gated"), "base_mae": b["mae"],
                 "base_mae_gated": b.get("mae_gated_subset"),
                 "mean_residual": b.get("mean_residual")}
            for kind in ("G", "S"):
                c[kind] = cell(rec, qname, kind, "q")
                c[kind + "_endpoint"] = cell(rec, qname, kind, "q_endpoint")
            c["baselines"] = rec.get("baselines", {}).get("q", {}).get(qname)
            s["cells"][qname] = c
        out["seeds"][seed_s] = s
    return out


HEADER = None


def write_per_molecule(records, path, stage, header=True):
    global HEADER
    n = 0
    with open(path, "w" if header else "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if header:
            w.writerow(["molecule_id", "stage", "role", "target_family", "seed", "y",
                    "base_pred", "G_pred_selected", "S_pred_selected",
                    "G_pred_endpoint", "S_pred_endpoint", "n_unseen", "unseen_frac",
                    "min_n1", "mean_n1", "min_n2", "mean_n2", "gated",
                    "base_ckpt_sha256", "support_hash", "n_head_inits"])
        for seed_s, rec in records.items():
            seed = int(seed_s)
            for qname, b in rec["base_metrics"].items():
                fam = qname.split("_")[-1]
                pm = b["per_molecule"]
                base = np.asarray(pm["base"])
                y = np.asarray(pm["y"])
                rows = pm["row"]
                sup = pm["support"]
                sel = {k: np.mean([e["q"][qname]["per_molecule"]["corrected"]
                                   for e in rec["arms"][k]], axis=0) for k in ("G", "S")}
                end = {k: np.mean([e["q_endpoint"][qname]["per_molecule"]["corrected"]
                                   for e in rec["arms"][k]], axis=0) for k in ("G", "S")}
                for i, mid in enumerate(rows):
                    w.writerow([mid, stage, qname, fam, seed, f"{y[i]:.6f}", f"{base[i]:.6f}",
                                f"{sel['G'][i]:.6f}", f"{sel['S'][i]:.6f}",
                                f"{end['G'][i]:.6f}", f"{end['S'][i]:.6f}",
                                int(sup["n_unseen"][i]), f"{sup['unseen_frac'][i]:.6f}",
                                f"{sup['min_n1'][i]:.4f}", f"{sup['mean_n1'][i]:.4f}",
                                f"{sup['min_n2'][i]:.4f}", f"{sup['mean_n2'][i]:.4f}",
                                int(sup["n_unseen"][i] > 0),
                                rec.get("base_ckpt_sha256") or "",
                                rec.get("support_hash") or "", 3])
                    n += 1
    return n


def md(summary, old=None) -> str:
    L = []
    L.append("# Main table — B / G / S, new holdout\n")
    L.append("`delta` = corrected MAE − base MAE (negative favours the correction).")
    L.append("The **C_dev-selected** head is the primary arm; every one of the 24 heads returned")
    L.append("`steps_used = 0`, i.e. the selection rule chooses *no correction*, so the selected")
    L.append("G and S columns equal B by construction. The `endpoint` columns are the descriptive")
    L.append("no-selection read at 1,500 steps (the fitted head actually applied).\n")
    L.append("| seed | package | n | n gated | B MAE | G sel | S sel | G end | S end | G−B end | S−B end | S−G end | S improves end | |h_S| end |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed_s, s in summary["seeds"].items():
        for q in s["cells"]:
            c = s["cells"][q]
            g, sc = c["G"], c["S"]
            ge, se = c["G_endpoint"], c["S_endpoint"]
            sg = float(np.mean([a - b for a, b in zip(se["delta_by_init"], ge["delta_by_init"])]))
            L.append(f"| {seed_s} | {q} | {c['n']} | {c['n_gated']} | {c['base_mae']:.4f} | "
                     f"{g['mae_mean']:.4f} | {sc['mae_mean']:.4f} | {ge['mae_mean']:.4f} | "
                     f"{se['mae_mean']:.4f} | {ge['delta_mean']:+.4f} | {se['delta_mean']:+.4f} | "
                     f"{sg:+.4f} | {se['frac_improved_mean']:.2f} | {se['mean_abs_correction']:.4f} |")
    L.append("\n## Bootstrap intervals on the endpoint deltas (per head init)\n")
    L.append("| seed | package | G−B [95% mol] | S−B [95% mol] | S−B [95% key-cluster] | clusters |")
    L.append("|---|---|---|---|---|---:|")
    for seed_s, s in summary["seeds"].items():
        for q in s["cells"]:
            c = s["cells"][q]
            ge, se = c["G_endpoint"], c["S_endpoint"]
            L.append(f"| {seed_s} | {q} | {fmt_ci(ge['ci_by_init'])} | {fmt_ci(se['ci_by_init'])} | "
                     f"{fmt_ci(se['key_ci_by_init'])} | {se['n_key_clusters'] or '-'} |")
    L.append("\n## Head fitting and selection\n")
    L.append("Raw (`dev_mae`) and pre-registered (`dev_mae_trim5`) curves are both shown. Every "
             "listed arm (the C_dev-selected wd) has its minimum at step 0 under **both**, so the "
             "null is not a trim5 artefact. The other wd only kept its trim5 level (also equal to "
             "the base), not its raw curve.")
    L.append("| seed | head | wd | C_fit gated MAE | C_dev base raw | C_dev best raw (post-hoc) | C_dev base trim5 | C_dev best trim5 | selected step |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for seed_s, s in summary["seeds"].items():
        for kind in ("G", "S"):
            c = s["cells"]["Q_A"][kind]
            dv = s["c_dev"]
            L.append(f"| {seed_s} | {kind} | {s['wd_selection'][kind]['selected_wd']} | "
                     f"{np.mean(c['fit_mae_cfit_gated']):.4f} | {dv['base_mae']:.4f} | "
                     f"{np.mean(c['dev_mae_raw']):.4f} | {dv['base_mae_trim5']:.4f} | "
                     f"{np.mean(c['dev_mae_trim5']):.4f} | {c['selected_step']} |")
    L.append("\n## Coarse baselines (no local structure at all)\n")
    L.append("| seed | package | constant (mean) Δ | constant (median) Δ | linear in f_base Δ |")
    L.append("|---|---|---:|---:|---:|")
    for seed_s, s in summary["seeds"].items():
        for q in s["cells"]:
            b = s["cells"][q]["baselines"]
            if not b:
                continue
            L.append(f"| {seed_s} | {q} | {b['constant_mean']['delta']:+.4f} | "
                     f"{b['constant_median']['delta']:+.4f} | {b['linear_f_base']['delta']:+.4f} |")
    dg = (summary.get("diagnostics") or {}).get("within_key") or {}
    if dg:
        L.append("\n## Within-key diagnostic (molecule-disjoint 80/20 inside every C key family)\n")
        L.append("Held-out molecules share the exact radius-2 keys with their training molecules.")
        L.append("The **selected** column is the pre-registered rule (trim5). The post-hoc raw")
        L.append("minimum is shown for completeness and is **not** a selection rule.\n")
        L.append("| seed | head | base hold MAE | selected step | selected Δ | raw best step (post-hoc) | raw best Δ (post-hoc) |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for seed_s, v in dg.items():
            for kind in ("G", "S"):
                x = v[kind]
                L.append(f"| {seed_s} | {kind} | {v['base_mae_hold']:.4f} | {x['best_step']} | "
                         f"{x['hold_delta']:+.4f} | {x['raw_best_step']} | "
                         f"{x['raw_best_delta_vs_base']:+.4f} |")
        L.append("\nOracle per-package constant (only bounds whole-package translations):\n")
        L.append("| cell | base MAE | best constant | gain |")
        L.append("|---|---:|---:|---:|")
        for k, v in (summary["diagnostics"].get("oracle_constant") or {}).items():
            L.append(f"| {k} | {v['base_mae']:.4f} | {v['best_constant']:+.4f} | {v['gain']:.6f} |")
    if old:
        L.append("\n## Old supervision-support package (exploration, `sup_minus` epoch-600 base)\n")
        L.append("| seed | package | n | base MAE | base MAE gated | S selected Δ | G endpoint Δ | S endpoint Δ |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for seed_s, rec in old["seeds"].items():
            for q in rec["base_metrics"]:
                c = rec["base_metrics"][q]
                sp = rec["arms"]["S"][0]["q"][q]["delta"]
                ge = float(np.mean([e["q_endpoint"][q]["delta"] for e in rec["arms"]["G"]]))
                se = float(np.mean([e["q_endpoint"][q]["delta"] for e in rec["arms"]["S"]]))
                L.append(f"| {seed_s} | {q} | {c['n']} | {c['mae']:.4f} | "
                         f"{c['mae_gated_subset'] or 0:.4f} | {sp:+.4f} | {ge:+.4f} | {se:+.4f} |")
    return "\n".join(L) + "\n"


def fmt_ci(ci):
    rows = []
    for item in ci:
        if item is None:
            continue
        a, b = item
        rows.append(f"[{a:+.4f}, {b:+.4f}]")
    return " / ".join(rows) if rows else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(Z.RUNS / "records" / "head_records_full.json"))
    ap.add_argument("--final", action="store_true", help="write summary.json + main_table.md")
    ap.add_argument("--stage", default="new_holdout")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if not Path(args.records).exists():
        raise SystemExit(f"missing {args.records}; run code/zst_run_head.py first")
    records = json.load(open(args.records, encoding="utf-8"))
    summary = summarise(records, args.stage)
    old = None
    p = Z.ANALYSIS / "oldpkg_records_v2.json"
    if p.exists():
        old = json.load(open(p, encoding="utf-8"))
        summary["old_package"] = {
            "note": "exploration on the old package; sup_minus epoch-600 base, no new training",
            "seeds": {s: {"c_fit": r.get("c_fit"), "c_dev": r.get("c_dev"),
                          "wd_selection": r.get("wd_selection"),
                          "cells": {q: {"base_mae": r["base_metrics"][q]["mae"],
                                        "n": r["base_metrics"][q]["n"],
                                        "n_gated": r["base_metrics"][q]["n_gated"],
                                        "G_selected_delta": float(np.mean(
                                            [e["q"][q]["delta"] for e in r["arms"]["G"]])),
                                        "S_selected_delta": float(np.mean(
                                            [e["q"][q]["delta"] for e in r["arms"]["S"]])),
                                        "G_endpoint_delta": float(np.mean(
                                            [e["q_endpoint"][q]["delta"] for e in r["arms"]["G"]])),
                                        "S_endpoint_delta": float(np.mean(
                                            [e["q_endpoint"][q]["delta"] for e in r["arms"]["S"]])),
                                        "selected_steps": [e["steps_used"] for e in r["arms"]["S"]]}
                                    for q in r["base_metrics"]}}
                      for s, r in old["seeds"].items()}}
        summary["old_package"]["components"] = old.get("components")
        summary["old_package"]["historical_exposure"] = old.get("historical_exposure")
    diag = Z.ANALYSIS / "diagnostics.json"
    if diag.exists():
        summary["diagnostics"] = json.load(open(diag, encoding="utf-8"))
    summary["seed_models"] = {s: {k: v.get(k) for k in ("n_fit", "base_ckpt_sha256", "support_hash")}
                              for s, v in summary["seeds"].items()}
    tag = args.tag
    Z.write_json(Z.ANALYSIS / f"summary{tag}.json", summary)
    (Z.ANALYSIS / f"main_table{tag}.md").write_text(md(summary, old), encoding="utf-8")
    n = write_per_molecule(records, Z.ANALYSIS / f"per_molecule{tag}.csv", args.stage)
    for seed in records:
        p = Z.ANALYSIS / f"base_rows_s{seed}.csv"
        if p.exists():
            rows = p.read_text(encoding="utf-8").splitlines()
            with open(Z.ANALYSIS / f"per_molecule{tag}.csv", "a", encoding="utf-8") as fh:
                fh.write("\n".join(rows[1:]) + "\n")
            n += len(rows) - 1
    full_old = Z.RUNS / "records" / "oldpkg_records_full.json"
    if full_old.exists():
        old_full = json.load(open(full_old, encoding="utf-8"))
        n += write_per_molecule(old_full["seeds"], Z.ANALYSIS / f"per_molecule{tag}.csv",
                                "old_package", header=False)
    print(f"[report] {n} per-molecule rows")
    if args.final:
        (Z.ANALYSIS / "main_table.md").write_text(md(summary, old), encoding="utf-8")
        Z.write_json(Z.ANALYSIS / "summary.json", summary)
    print((Z.ANALYSIS / f"main_table{tag}.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
