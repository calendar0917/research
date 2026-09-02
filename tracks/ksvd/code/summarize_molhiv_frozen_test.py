"""Summarize the predeclared frozen MolHIV final evaluation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "values": values,
        "mean": float(arr.mean()),
        "sample_std": float(arr.std(ddof=1)),
        "sem": float(arr.std(ddof=1) / np.sqrt(arr.size)),
        "minimum": float(arr.min()),
        "maximum": float(arr.max()),
    }


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    for p, row in zip(paths, rows):
        if row.get("official_test_evaluations") != 1:
            raise ValueError(f"{p} is not a single frozen test evaluation")
        if row.get("official_valid_evaluations") != 1:
            raise ValueError(f"{p} lacks its single official-valid evaluation")
    return rows


def _ensemble(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    ys = [np.asarray(row[f"official_{prefix}_y"], dtype=np.int64) for row in rows]
    preds = [
        np.asarray(row[f"official_{prefix}_predictions"], dtype=np.float64)
        for row in rows
    ]
    if not all(np.array_equal(ys[0], y) for y in ys[1:]):
        raise ValueError(f"{prefix} labels differ across seeds")
    if not all(pred.shape == preds[0].shape for pred in preds):
        raise ValueError(f"{prefix} prediction shapes differ across seeds")
    mean_pred = np.mean(np.stack(preds, axis=0), axis=0)
    return {
        "rocauc": float(roc_auc_score(ys[0], mean_pred)),
        "n_models": len(rows),
        "aggregation": "arithmetic mean of per-seed positive-class probabilities",
    }


def _paired(left: list[float], right: list[float]) -> dict[str, Any]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    result = {
        "deltas": delta.tolist(),
        "mean": float(delta.mean()),
        "sample_std": float(delta.std(ddof=1)),
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
    }
    try:
        from scipy.stats import ttest_rel

        result["two_sided_paired_ttest_p"] = float(
            ttest_rel(np.asarray(left), np.asarray(right)).pvalue
        )
    except Exception:
        result["two_sided_paired_ttest_p"] = None
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/molhiv")
    ap.add_argument(
        "--output",
        default="results/molhiv/frozen_test_5seeds_summary.json",
    )
    args = ap.parse_args()
    root = Path(args.results_dir)
    specs = {
        "gine_h64": {
            "paths": [root / f"frozen_test_gine_h64_seed{s}.json" for s in range(5)],
            "trainable_parameters": 37380,
        },
        "gine_h70_parameter_matched": {
            "paths": [root / f"frozen_test_gine_h70_seed{s}.json" for s in range(5)],
            "trainable_parameters": 43404,
        },
        "ksvd_h64_d32_t3": {
            "paths": [root / f"frozen_test_ksvd_seed{s}.json" for s in range(5)],
            "trainable_parameters": 43655,
            "fixed_dictionary_values": 27136,
        },
    }
    output: dict[str, Any] = {
        "protocol_id": "molhiv-localized-node-token-frozen-test-5seed-summary-v1",
        "frozen_config": "results/molhiv/node_token_frozen_config_v1.json",
        "disclosure": "earlier feasibility files in this repository inspected official test; this is a post-freeze controlled terminal evaluation, not an untouched-test claim",
        "families": {},
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, spec in specs.items():
        rows = _load(spec["paths"])
        loaded[name] = rows
        valid = [float(row["official_valid_auc"]) for row in rows]
        test = [float(row["official_test_auc"]) for row in rows]
        output["families"][name] = {
            "paths": [str(p) for p in spec["paths"]],
            "trainable_parameters": spec["trainable_parameters"],
            "fixed_dictionary_values": spec.get("fixed_dictionary_values", 0),
            "selected_epochs": [int(row["selected_epoch"]) for row in rows],
            "official_valid": _stats(valid),
            "official_test": _stats(test),
            "official_valid_probability_ensemble": _ensemble(rows, "valid"),
            "official_test_probability_ensemble": _ensemble(rows, "test"),
        }
    k = output["families"]["ksvd_h64_d32_t3"]
    g64 = output["families"]["gine_h64"]
    g70 = output["families"]["gine_h70_parameter_matched"]
    output["paired"] = {
        "ksvd_minus_gine_h64_valid": _paired(
            k["official_valid"]["values"], g64["official_valid"]["values"]
        ),
        "ksvd_minus_gine_h64_test": _paired(
            k["official_test"]["values"], g64["official_test"]["values"]
        ),
        "ksvd_minus_parameter_matched_gine_h70_valid": _paired(
            k["official_valid"]["values"], g70["official_valid"]["values"]
        ),
        "ksvd_minus_parameter_matched_gine_h70_test": _paired(
            k["official_test"]["values"], g70["official_test"]["values"]
        ),
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
