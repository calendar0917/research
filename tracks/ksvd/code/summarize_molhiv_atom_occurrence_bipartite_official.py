"""Summarize the frozen 3-seed farthest real-patch official evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def paired_stratified_bootstrap_auc_delta(
    y: np.ndarray, bip: np.ndarray, node: np.ndarray, *, replicates: int, seed: int
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        idx = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        yy = y[idx]
        deltas[i] = roc_auc_score(yy, bip[idx]) - roc_auc_score(yy, node[idx])
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_delta_gt_zero": float(np.mean(deltas > 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = ap.parse_args()
    reports = [json.loads(Path(p).read_text()) for p in args.inputs]
    reports.sort(key=lambda row: int(row["config"]["seed"]))
    seeds = [int(row["config"]["seed"]) for row in reports]
    if seeds != [0, 1, 2]:
        raise ValueError(f"expected seeds 0/1/2, got {seeds}")

    controls = ["farthest_node_mil", "farthest_bipartite"]
    splits = ["official_valid", "official_test"]
    prototype_hashes = {row["prototype_sha256"]["farthest"] for row in reports}
    if len(prototype_hashes) != 1:
        raise AssertionError("farthest prototype vocabulary changed across model seeds")

    aggregate: dict[str, dict] = {}
    ensemble_predictions: dict[tuple[str, str], np.ndarray] = {}
    shared_labels: dict[str, np.ndarray] = {}
    for split in splits:
        labels = [
            np.asarray(row["results"][controls[0]][split]["labels"], dtype=np.int64)
            for row in reports
        ]
        if not all(np.array_equal(labels[0], y) for y in labels[1:]):
            raise AssertionError(f"{split} labels differ across seeds")
        for row in reports:
            other = np.asarray(row["results"][controls[1]][split]["labels"], dtype=np.int64)
            if not np.array_equal(labels[0], other):
                raise AssertionError(f"{split} labels differ across controls")
        shared_labels[split] = labels[0]

    for control in controls:
        control_row: dict[str, object] = {
            "trainable_parameters": sorted({
                int(row["results"][control]["trainable_parameters"]) for row in reports
            }),
            "per_seed": [],
        }
        for row in reports:
            seed_row = {"seed": int(row["config"]["seed"])}
            for split in splits:
                seed_row[split + "_auc"] = float(row["results"][control][split]["auc"])
            control_row["per_seed"].append(seed_row)
        for split in splits:
            aucs = [float(row["results"][control][split]["auc"]) for row in reports]
            probabilities = [
                np.asarray(row["results"][control][split]["probabilities"], dtype=np.float64)
                for row in reports
            ]
            ensemble = np.mean(np.stack(probabilities), axis=0)
            ensemble_predictions[(control, split)] = ensemble
            control_row[split] = {
                "single_seed_mean_auc": float(np.mean(aucs)),
                "single_seed_sample_std_auc": sample_std(aucs),
                "ensemble_auc": float(roc_auc_score(shared_labels[split], ensemble)),
                "ensemble_probability_sha256": sha256(ensemble),
                "n_graphs": int(len(ensemble)),
                "n_positive": int(shared_labels[split].sum()),
            }
        aggregate[control] = control_row

    comparisons = {}
    for split in splits:
        bip = aggregate["farthest_bipartite"][split]
        node = aggregate["farthest_node_mil"][split]
        per_seed_delta = [
            float(row["results"]["farthest_bipartite"][split]["auc"])
            - float(row["results"]["farthest_node_mil"][split]["auc"])
            for row in reports
        ]
        comparisons[split] = {
            "bipartite_minus_node_per_seed": per_seed_delta,
            "mean_delta": float(np.mean(per_seed_delta)),
            "wins": int(sum(x > 0 for x in per_seed_delta)),
            "ensemble_delta": float(bip["ensemble_auc"] - node["ensemble_auc"]),
            "paired_stratified_bootstrap": paired_stratified_bootstrap_auc_delta(
                shared_labels[split],
                ensemble_predictions[("farthest_bipartite", split)],
                ensemble_predictions[("farthest_node_mil", split)],
                replicates=args.bootstrap_replicates,
                seed=20260728 + (0 if split == "official_valid" else 1),
            ),
        }

    out = {
        "protocol_id": "molhiv-realpatch-atom-occurrence-bipartite-full-official-ensemble-v1",
        "date": "2026-07-28",
        "policy": {
            "development": "8000-graph official-train-only 3 scaffold folds x 3 seeds",
            "development_mean_gain": 0.01596775,
            "development_pairwise_wins": "6/9",
            "full_dataset_graphs": int(reports[0]["n_train"] + reports[0]["n_valid"] + reports[0]["n_test"]),
            "members": "fixed farthest real-patch vocabulary; model seeds 0/1/2",
            "ensemble": "predeclared arithmetic mean of probabilities",
            "official_test_status": "controlled frozen evaluation, not untouched because repository previously inspected official test",
            "post_test_tuning_permitted": False,
        },
        "inputs": args.inputs,
        "prototype_sha256": next(iter(prototype_hashes)),
        "label_sha256": {split: sha256(y) for split, y in shared_labels.items()},
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
