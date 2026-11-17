"""Small machine-readable summary + failure-mode table."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import xr_common as C


def main() -> int:
    pairs = list(csv.DictReader(open(C.ANALYSIS_DIR / "measured_pair_results.csv", encoding="utf-8")))
    oracle = list(csv.DictReader(open(C.ANALYSIS_DIR / "oracle_localization_results.csv", encoding="utf-8")))
    pair_table = C.read_json(C.ANALYSIS_DIR / "pair_table.json")["pairs"]
    decision = C.read_json(C.ANALYSIS_DIR / "decision.json")

    modes = ["accurate_localized", "accurate_mislocalized", "inaccurate_localized", "inaccurate_mislocalized"]
    scopes = {
        "all": [True] * len(pairs),
        "train_train": [r["stratum"] == "train_train" for r in pairs],
        "at_least_one_nontrain": [r["stratum"] != "train_train" for r in pairs],
        "direction_evaluable": [str(r["direction_evaluable"]) in ("1", "True", "true") for r in pairs],
    }
    rows = []
    for scope, mask in scopes.items():
        for model in ("LCRC", "GINE"):
            vals = [r[f"failure_mode_{model}"] for r, keep in zip(pairs, mask) if keep]
            counts = Counter(vals)
            n = len(vals)
            for mode in modes:
                rows.append({"scope": scope, "model": model, "failure_mode": mode, "n": int(counts.get(mode, 0)), "fraction": (counts.get(mode, 0) / n if n else None)})
            rows.append({"scope": scope, "model": model, "failure_mode": "total", "n": n, "fraction": 1.0})
    path = C.ANALYSIS_DIR / "failure_modes.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["scope", "model", "failure_mode", "n", "fraction"])
        w.writeheader()
        for row in rows:
            w.writerow({**row, "fraction": "" if row["fraction"] is None else f"{row['fraction']:.6g}"})

    core = np.asarray([p["wc"]["core_abs_sum"] for p in pair_table], dtype=np.float64)
    va = np.asarray([p["wc"]["var_a_abs_sum"] for p in pair_table], dtype=np.float64)
    vb = np.asarray([p["wc"]["var_b_abs_sum"] for p in pair_table], dtype=np.float64)
    total = core + va + vb
    frac_var = (va + vb) / total
    zero = {}
    for model in ("LCRC", "GINE", "LOCAL_LCRC"):
        vals = np.asarray([float(r["oracle_enrichment"]) for r in oracle if r["model"] == model and r["oracle_enrichment"] != ""], dtype=np.float64)
        rand = np.asarray([float(r["random_median_capture"]) for r in oracle if r["model"] == model and r["random_median_capture"] != ""], dtype=np.float64)
        top = np.asarray([float(r["top1_capture"]) for r in oracle if r["model"] == model and r["top1_capture"] != ""], dtype=np.float64)
        zero[model] = {
            "n": int(vals.size),
            "median_enrichment": float(np.median(vals)) if vals.size else None,
            "mean_enrichment": float(np.mean(vals)) if vals.size else None,
            "fraction_top1_gt_random_median": float((top > rand).mean()) if vals.size else None,
            "fraction_enrichment_gt_0": float((vals > 0).mean()) if vals.size else None,
        }
    summary = {
        "round_id": C.ROUND_ID,
        "decision_labels": decision["labels"],
        "label_scope": decision.get("label_scope"),
        "pair_support": C.read_json(C.ANALYSIS_DIR / "pair_support.json"),
        "prediction": decision["prediction"],
        "oracle": {m: decision["oracle_detail"][m] for m in ("LCRC", "GINE")},
        "real_label": {m: decision["real_label_detail"][m] for m in ("LCRC", "GINE")},
        "local_heuristic": decision.get("local_heuristic"),
        "contribution_change_fraction": {
            "mean_variable": float(frac_var.mean()),
            "median_variable": float(np.median(frac_var)),
            "q25_variable": float(np.quantile(frac_var, 0.25)),
            "q75_variable": float(np.quantile(frac_var, 0.75)),
            "mean_core": float(np.mean(core / total)),
            "median_core": float(np.median(core / total)),
            "fraction_pairs_variable_gt_0.9": float((frac_var > 0.9).mean()),
            "fraction_pairs_core_zero": float((core == 0).mean()),
        },
        "zero_inflation": zero,
        "failure_modes_csv": {"path": C.relpath(path), "sha256": C.sha256_file(path)},
    }
    C.write_json(C.ANALYSIS_DIR / "summary_metrics.json", summary)
    print(json.dumps(summary, indent=1)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
