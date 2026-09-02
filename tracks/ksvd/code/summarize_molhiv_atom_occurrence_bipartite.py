"""Summarize train-only scaffold-fold atom--occurrence bipartite pilots."""

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
    reports.sort(key=lambda row: (int(row["fold"]), int(row["config"]["seed"])))
    protocols = {row["protocol_id"] for row in reports}
    if protocols != {"molhiv-atom-occurrence-bipartite-scaffold-v1"}:
        raise ValueError(f"unexpected protocols: {protocols}")
    controls = sorted(set.intersection(*[set(row["results"]) for row in reports]))

    aggregate: dict[str, dict[str, object]] = {}
    for control in controls:
        aucs = np.asarray([
            row["results"][control]["heldout"]["auc"] for row in reports
        ], dtype=np.float64)
        aggregate[control] = {
            "aucs": aucs.tolist(),
            "mean": float(aucs.mean()),
            "std": float(aucs.std()),
            "min": float(aucs.min()),
            "max": float(aucs.max()),
            "trainable_parameters": sorted({
                int(row["results"][control]["trainable_parameters"])
                for row in reports
            }),
        }

    pairs = {
        "farthest_bipartite_vs_node_mil": ("farthest_bipartite", "farthest_node_mil"),
        "ksvd_bipartite_vs_node_mil": ("ksvd_bipartite", "ksvd_node_mil"),
        "ksvd_vs_farthest_bipartite": ("ksvd_bipartite", "farthest_bipartite"),
        "ksvd_vs_pca_bipartite": ("ksvd_bipartite", "pca_bipartite"),
        "ksvd_vs_random_direction_bipartite": (
            "ksvd_bipartite", "random_direction_bipartite"
        ),
        "ksvd_vs_shuffled_id": (
            "ksvd_bipartite", "ksvd_bipartite_shuffled_id"
        ),
        "ksvd_vs_no_id": ("ksvd_bipartite", "ksvd_bipartite_no_id"),
        "ksvd_vs_medoid_bipartite": (
            "ksvd_bipartite", "ksvd_medoid_bipartite"
        ),
    }
    comparisons: dict[str, dict[str, object]] = {}
    for name, (candidate, baseline) in pairs.items():
        if candidate not in aggregate or baseline not in aggregate:
            continue
        deltas = np.asarray(aggregate[candidate]["aucs"]) - np.asarray(
            aggregate[baseline]["aucs"]
        )
        comparisons[name] = {
            "deltas": deltas.tolist(),
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std()),
            "wins": int(np.sum(deltas > 0)),
            "ties": int(np.sum(deltas == 0)),
            "n": int(len(deltas)),
        }

    farthest_gain = comparisons.get("farthest_bipartite_vs_node_mil", {})
    ksvd_gain = comparisons.get("ksvd_bipartite_vs_node_mil", {})
    ksvd_controls = [
        comparisons.get("ksvd_vs_farthest_bipartite", {}),
        comparisons.get("ksvd_vs_pca_bipartite", {}),
        comparisons.get("ksvd_vs_random_direction_bipartite", {}),
    ]
    promotion = {
        "bipartite_gain_threshold": 0.005,
        "minimum_wins": max(2, int(np.ceil(len(reports) * 2 / 3))),
        "farthest_bipartite_gain_pass": bool(
            farthest_gain
            and farthest_gain["mean_delta"] >= 0.005
            and farthest_gain["wins"] >= max(2, int(np.ceil(len(reports) * 2 / 3)))
        ),
        "ksvd_bipartite_gain_pass": bool(
            ksvd_gain
            and ksvd_gain["mean_delta"] >= 0.005
            and ksvd_gain["wins"] >= max(2, int(np.ceil(len(reports) * 2 / 3)))
        ),
        "ksvd_beats_all_geometry_controls_mean": bool(
            ksvd_controls and all(row and row["mean_delta"] > 0 for row in ksvd_controls)
        ),
        "ksvd_identity_beats_shuffled_mean": bool(
            comparisons.get("ksvd_vs_shuffled_id", {}).get("mean_delta", -np.inf) > 0
        ),
        "ksvd_identity_beats_no_id_mean": bool(
            comparisons.get("ksvd_vs_no_id", {}).get("mean_delta", -np.inf) > 0
        ),
    }
    promotion["strict_ksvd_promotion_pass"] = bool(
        promotion["ksvd_bipartite_gain_pass"]
        and promotion["ksvd_beats_all_geometry_controls_mean"]
        and promotion["ksvd_identity_beats_shuffled_mean"]
        and promotion["ksvd_identity_beats_no_id_mean"]
    )

    output = {
        "protocol_id": "molhiv-atom-occurrence-bipartite-summary-v1",
        "date": "2026-07-28",
        "selection_only": True,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "inputs": args.inputs,
        "fold_seed_order": [
            {"fold": int(row["fold"]), "seed": int(row["config"]["seed"])}
            for row in reports
        ],
        "aggregate": aggregate,
        "comparisons": comparisons,
        "promotion": promotion,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
