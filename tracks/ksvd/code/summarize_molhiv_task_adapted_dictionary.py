"""Aggregate the three fixed-epoch task-adapted dictionary scaffold screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    reports.sort(key=lambda row: int(row["fold"]))
    folds = [int(row["fold"]) for row in reports]
    if len(set(folds)) != len(folds):
        raise ValueError("duplicate folds")
    controls = list(reports[0]["results"])
    if any(set(row["results"]) != set(controls) for row in reports):
        raise ValueError("reports do not contain matched controls")

    aggregate = {}
    for control in controls:
        aucs = np.asarray([
            row["results"][control]["heldout"]["auc"] for row in reports
        ], dtype=np.float64)
        rec = np.asarray([
            row["results"][control]["heldout"]["reconstruction_mse_per_node"]
            for row in reports
        ], dtype=np.float64)
        movement = np.asarray([
            row["results"][control]["dictionary"]["mean_atom_l2_movement"]
            for row in reports
        ], dtype=np.float64)
        aggregate[control] = {
            "fold_auc": aucs.tolist(),
            "mean_auc": float(aucs.mean()),
            "std_auc": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "mean_heldout_reconstruction_mse_per_node": float(rec.mean()),
            "mean_atom_l2_movement": float(movement.mean()),
        }

    adapt = np.asarray(aggregate["adapt_ksvd"]["fold_auc"])
    frozen = np.asarray(aggregate["frozen_ksvd"]["fold_auc"])
    random = np.asarray(aggregate["adapt_random"]["fold_auc"])
    delta_frozen = adapt - frozen
    delta_random = adapt - random
    promotion = {
        "mean_adapt_minus_frozen_auc": float(delta_frozen.mean()),
        "fold_wins_over_frozen": int((delta_frozen > 0).sum()),
        "mean_adapt_minus_random_auc": float(delta_random.mean()),
        "fold_wins_over_random": int((delta_random > 0).sum()),
        "mean_gain_at_least_005": bool(delta_frozen.mean() >= 0.005),
        "at_least_2_of_3_frozen_wins": bool((delta_frozen > 0).sum() >= 2),
        "beats_adapt_random_mean": bool(delta_random.mean() > 0),
    }
    promotion["promote_to_shallow_gnn"] = bool(
        promotion["mean_gain_at_least_005"]
        and promotion["at_least_2_of_3_frozen_wins"]
        and promotion["beats_adapt_random_mean"]
    )
    summary = {
        "protocol_id": "molhiv-task-adapted-dictionary-primary-mil-scaffold-summary-v1",
        "date": "2026-07-28",
        "folds": folds,
        "inputs": args.inputs,
        "aggregate": aggregate,
        "promotion": promotion,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
