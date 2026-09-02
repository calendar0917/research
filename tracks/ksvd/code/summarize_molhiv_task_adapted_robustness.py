"""Aggregate task-adapted dictionary results across scaffold folds and seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--controls", default="frozen_ksvd,adapt_ksvd,frozen_random")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    controls = [x.strip() for x in args.controls.split(",") if x.strip()]
    cells: dict[tuple[int, int, str], dict] = {}
    sources: dict[tuple[int, int, str], str] = {}
    for path_raw in args.inputs:
        path = Path(path_raw)
        report = json.loads(path.read_text(encoding="utf-8"))
        fold = int(report["fold"])
        seed = int(report["config"]["seed"])
        for control, row in report["results"].items():
            if control not in controls:
                continue
            key = (fold, seed, control)
            if key in cells:
                old = float(cells[key]["heldout"]["auc"])
                new = float(row["heldout"]["auc"])
                if abs(old - new) > 1e-12:
                    raise ValueError(f"conflicting duplicate {key}: {old} vs {new}")
                continue
            cells[key] = row
            sources[key] = str(path)
    folds = sorted({key[0] for key in cells})
    seeds = sorted({key[1] for key in cells})
    missing = [
        (fold, seed, control)
        for fold in folds for seed in seeds for control in controls
        if (fold, seed, control) not in cells
    ]
    if missing:
        raise ValueError(f"missing cells: {missing}")

    aggregate = {}
    for control in controls:
        matrix = np.asarray([
            [cells[(fold, seed, control)]["heldout"]["auc"] for fold in folds]
            for seed in seeds
        ], dtype=np.float64)
        aggregate[control] = {
            "seed_by_fold_auc": matrix.tolist(),
            "per_seed_scaffold_mean_auc": matrix.mean(axis=1).tolist(),
            "per_fold_seed_mean_auc": matrix.mean(axis=0).tolist(),
            "grand_mean_auc": float(matrix.mean()),
            "std_over_9_cells": float(matrix.std(ddof=1)),
        }

    def paired(left: str, right: str) -> dict:
        a = np.asarray(aggregate[left]["seed_by_fold_auc"])
        b = np.asarray(aggregate[right]["seed_by_fold_auc"])
        delta = a - b
        return {
            "left": left,
            "right": right,
            "seed_by_fold_delta": delta.tolist(),
            "per_seed_mean_delta": delta.mean(axis=1).tolist(),
            "per_fold_mean_delta": delta.mean(axis=0).tolist(),
            "grand_mean_delta": float(delta.mean()),
            "wins_out_of_9": int((delta > 0).sum()),
            "ties_out_of_9": int((delta == 0).sum()),
        }

    comparisons = {
        "adapt_minus_frozen_ksvd": paired("adapt_ksvd", "frozen_ksvd"),
        "frozen_random_minus_frozen_ksvd": paired("frozen_random", "frozen_ksvd"),
        "adapt_ksvd_minus_frozen_random": paired("adapt_ksvd", "frozen_random"),
    }
    promotion = {
        "adapt_mean_gain_at_least_005": bool(
            comparisons["adapt_minus_frozen_ksvd"]["grand_mean_delta"] >= 0.005
        ),
        "adapt_wins_at_least_6_of_9": bool(
            comparisons["adapt_minus_frozen_ksvd"]["wins_out_of_9"] >= 6
        ),
        "adapt_beats_frozen_random": bool(
            comparisons["adapt_ksvd_minus_frozen_random"]["grand_mean_delta"] > 0
        ),
    }
    promotion["promote_to_shallow_gnn"] = all(promotion.values())
    summary = {
        "protocol_id": "molhiv-task-adapted-dictionary-primary-mil-3fold-3seed-v1",
        "date": "2026-07-28",
        "folds": folds,
        "seeds": seeds,
        "controls": controls,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "promotion": promotion,
        "sources": {str(key): value for key, value in sorted(sources.items())},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
