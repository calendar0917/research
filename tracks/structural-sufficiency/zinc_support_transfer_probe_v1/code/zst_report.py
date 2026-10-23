"""Tables, machine-readable summary and the per-molecule deliverable."""

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
import ad_common as C  # noqa: E402


def arm_cells(rec, kind):
    out = []
    for entry in rec["arms"][kind]:
        out.append(entry)
    return out


def summarise(records: dict, stage: str = "new_holdout") -> dict:
    out = {"stage": stage, "seeds": {}, "cells": []}
    for seed_s, rec in records.items():
        seed = int(seed_s)
        s = {"n_fit": rec.get("n_fit"), "base_ckpt_sha256": rec.get("base_ckpt_sha256"),
             "support_hash": rec.get("support_hash"),
             "roles": rec.get("q_roles"), "c_fit": rec.get("c_fit"),
             "c_dev": rec.get("c_dev"),
             "wd_selection": rec.get("wd_selection"), "cells": {}}
        for qname in ("Q_A", "Q_B", "Q_bg"):
            b = rec["base_metrics"][qname]
            cell = {"seed": seed, "package": qname, "base_mae": b["mae"],
                    "base_mae_gated": b.get("mae_gated_subset"),
                    "n": b["n"], "n_gated": b["n_gated"]}
            for kind in ("G", "S"):
                entries = rec["arms"][kind]
                deltas = [e["q"][qname]["delta"] for e in entries]
                maes = [e["q"][qname]["mae_corrected"] for e in entries]
                cell[kind] = {
                    "mae_mean": float(np.mean(maes)),
                    "mae_by_init": [float(x) for x in maes],
                    "delta_mean": float(np.mean(deltas)),
                    "delta_by_init": [float(x) for x in deltas],
                    "delta_ci_by_init": [e["q"][qname]["boot"]["molecule_ci"] for e in entries],
                    "delta_key_ci_by_init": [e["q"][qname]["boot"].get("key_cluster_ci")
                                             for e in entries],
                    "frac_improved_mean": float(np.mean([e["q"][qname]["frac_improved"]
                                                         for e in entries])),
                    "mean_abs_correction": float(np.mean([e["q"][qname]["mean_abs_correction"]
                                                          for e in entries])),
                    "trimmed_delta_mean": float(np.mean([e["q"][qname]["trimmed_delta"]
                                                         for e in entries])),
                    "fit_mae_cfit_gated": float(np.mean([e["fit_mae_cfit_gated"] for e in entries])),
                    "dev_mae": float(np.mean([e["dev_mae"] for e in entries])),
                }
                if qname == "Q_bg":
                    cell[kind]["delta_gated_subset"] = float(
                        np.mean([e["q"][qname]["delta_gated_subset"] for e in entries]))
                    cell[kind]["identity_max_abs_diff"] = float(
                        np.max([e["q"][qname]["identity_max_abs_diff_unsupported"] for e in entries]))
            mean_g = np.mean([e["q"][qname]["per_molecule"]["corrected"]
                              for e in rec["arms"]["G"]], axis=0)
            mean_s = np.mean([e["q"][qname]["per_molecule"]["corrected"]
                              for e in rec["arms"]["S"]], axis=0)
            base = np.asarray(b["per_molecule"]["base"])
            y = np.asarray(b["per_molecule"]["y"])
            sg = np.abs(mean_s - y) - np.abs(base - y)
            cell["S_minus_G"] = {
                "delta_mean": float(np.mean(np.abs(mean_s - y) - np.abs(mean_g - y))),
                "delta_by_init": [float(np.mean(
                    np.abs(e_s["q"][qname]["per_molecule"]["corrected"] - y)
                    - np.abs(e_g["q"][qname]["per_molecule"]["corrected"] - y)))
                    for e_s, e_g in zip(rec["arms"]["S"], rec["arms"]["G"])],
                "S_delta_mean": float(sg.mean()),
            }
            s["cells"][qname] = cell
            out["cells"].append(cell)
        out["seeds"][seed_s] = s
    return out


def write_per_molecule(records: dict, path: Path, stage: str) -> int:
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["molecule_id", "stage", "role", "target_family", "seed", "y",
                    "base_pred", "G_pred", "S_pred", "n_unseen", "unseen_frac",
                    "min_n1", "mean_n1", "min_n2", "mean_n2", "gated",
                    "base_ckpt_sha256", "support_hash", "n_head_inits"])
        for seed_s, rec in records.items():
            seed = int(seed_s)
            for qname in ("Q_A", "Q_B", "Q_bg"):
                fam = qname.split("_")[-1]
                g = rec["arms"]["G"]
                s = rec["arms"]["S"]
                base = np.asarray(rec["base_metrics"][qname]["per_molecule"]["base"])
                y = np.asarray(rec["base_metrics"][qname]["per_molecule"]["y"])
                row_ids = rec["base_metrics"][qname]["per_molecule"]["row"]
                sup = rec["base_metrics"][qname]["per_molecule"]["support"]
                g_mean = np.mean([e["q"][qname]["per_molecule"]["corrected"] for e in g], axis=0)
                s_mean = np.mean([e["q"][qname]["per_molecule"]["corrected"] for e in s], axis=0)
                for i, mid in enumerate(row_ids):
                    w.writerow([
                        mid, stage, qname, fam, seed, float(y[i]), float(base[i]),
                        float(g_mean[i]), float(s_mean[i]),
                        int(sup["n_unseen"][i]), float(sup["unseen_frac"][i]),
                        float(sup["min_n1"][i]), float(sup["mean_n1"][i]),
                        float(sup["min_n2"][i]), float(sup["mean_n2"][i]),
                        int(sup["n_unseen"][i] > 0),
                        rec.get("base_ckpt_sha256") or "", rec.get("support_hash") or "",
                        len(s)])
                    rows += 1
    return rows


def markdown(summary: dict) -> str:
    lines = ["# Main table — B / G / S on the two query packages and the background",
             "", "`delta` = corrected MAE − base MAE (negative favours the correction).",
             "Each cell is the mean over the three head initialisations; per-init values are in",
             "`analysis/summary.json` and `analysis/head_records*.json`.", ""]
    lines.append("| seed | package | n | n gated | B MAE | G MAE | S MAE | G−B | S−B | S−G | S improves | S trimmed Δ | mean |h_S| |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in summary["cells"]:
        g, s = cell["G"], cell["S"]
        lines.append(
            f"| {cell['seed']} | {cell['package']} | {cell['n']} | {cell['n_gated']} | "
            f"{cell['base_mae']:.4f} | {g['mae_mean']:.4f} | {s['mae_mean']:.4f} | "
            f"{g['delta_mean']:+.4f} | {s['delta_mean']:+.4f} | {cell['S_minus_G']['delta_mean']:+.4f} | "
            f"{s['frac_improved_mean']:.2f} | {s['trimmed_delta_mean']:+.4f} | "
            f"{s['mean_abs_correction']:.4f} |")
    lines += ["", "## Head fitting (C_fit) and selection (C_dev)", ""]
    lines.append("| seed | head | wd | C_fit gated MAE | C_dev MAE | base C_dev MAE |")
    lines.append("|---|---|---|---:|---:|---:|")
    for seed_s, s in summary["seeds"].items():
        for kind in ("G", "S"):
            entries = None
            for cell in summary["cells"]:
                if cell["seed"] == int(seed_s) and cell["package"] == "Q_A":
                    entries = cell
            g = entries["G"] if kind == "G" else entries["S"]
            lines.append(f"| {seed_s} | {kind} | {s['wd_selection'][kind]['selected_wd']} | "
                         f"{g['fit_mae_cfit_gated']:.4f} | {g['dev_mae']:.4f} | "
                         f"{s['c_dev']['base_mae']:.4f} |".replace("0.001 |", "1e-3 |"))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(Z.ANALYSIS / "head_records.json"))
    ap.add_argument("--stage", default="new_holdout")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    records = json.load(open(args.records, encoding="utf-8"))
    summary = summarise(records, stage=args.stage)
    tag = f"_{args.tag}" if args.tag else ""
    Z.write_json(Z.ANALYSIS / f"summary{tag}.json", summary)
    (Z.ANALYSIS / f"main_table{tag}.md").write_text(markdown(summary), encoding="utf-8")
    n = write_per_molecule(records, Z.ANALYSIS / f"per_molecule{tag}.csv", args.stage)
    print(f"[report] {n} per-molecule rows -> analysis/per_molecule{tag}.csv")
    print((Z.ANALYSIS / f"main_table{tag}.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
