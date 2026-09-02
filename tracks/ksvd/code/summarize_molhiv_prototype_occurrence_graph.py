"""Aggregate the strict three-fold prototype occurrence-graph screen.

The base and gated runs may be stored in separate JSON files.  Reports are
merged by outer scaffold fold and checked to ensure that every control uses the
same held-out split within a fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_CONTROLS = (
    "random_node_mil",
    "random_occ_mil",
    "random_occ_gine",
    "random_occ_gated_gine",
    "ksvd_occ_mil",
    "ksvd_occ_gine",
    "ksvd_occ_gated_gine",
    "ksvd_occ_gine_shuffled_id",
    "ksvd_occ_gine_no_id",
)


def _pair(aggregate: dict[str, Any], left: str, right: str) -> dict[str, Any]:
    delta = np.asarray(aggregate[left]["fold_auc"], dtype=np.float64) - np.asarray(
        aggregate[right]["fold_auc"], dtype=np.float64
    )
    return {
        "left": left,
        "right": right,
        "fold_delta_auc": delta.tolist(),
        "mean_delta_auc": float(delta.mean()),
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "min_delta_auc": float(delta.min()),
        "max_delta_auc": float(delta.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--severe-collapse-threshold",
        type=float,
        default=-0.02,
        help="A paired held-out AUC delta at or below this value is severe.",
    )
    args = ap.parse_args()

    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    by_fold: dict[int, dict[str, Any]] = {}
    input_by_fold: dict[int, list[str]] = {}
    for path, report in zip(args.inputs, reports):
        fold = int(report["fold"])
        input_by_fold.setdefault(fold, []).append(path)
        if fold not in by_fold:
            by_fold[fold] = {
                "heldout_indices_sha256": report["heldout_indices_sha256"],
                "fit_indices_sha256": report["fit_indices_sha256"],
                "results": {},
                "occurrence_stats": report.get("occurrence_stats", {}),
            }
        merged = by_fold[fold]
        if report["heldout_indices_sha256"] != merged["heldout_indices_sha256"]:
            raise ValueError(f"fold {fold} held-out split mismatch")
        if report["fit_indices_sha256"] != merged["fit_indices_sha256"]:
            raise ValueError(f"fold {fold} fit split mismatch")
        overlap = set(merged["results"]).intersection(report["results"])
        if overlap:
            raise ValueError(f"fold {fold} duplicate controls: {sorted(overlap)}")
        merged["results"].update(report["results"])
        if not merged["occurrence_stats"] and report.get("occurrence_stats"):
            merged["occurrence_stats"] = report["occurrence_stats"]

    folds = sorted(by_fold)
    if folds != [0, 1, 2]:
        raise ValueError(f"expected folds [0, 1, 2], got {folds}")
    for fold in folds:
        missing = set(EXPECTED_CONTROLS).difference(by_fold[fold]["results"])
        extra = set(by_fold[fold]["results"]).difference(EXPECTED_CONTROLS)
        if missing or extra:
            raise ValueError(
                f"fold {fold} control mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )

    aggregate: dict[str, Any] = {}
    for control in EXPECTED_CONTROLS:
        rows = [by_fold[fold]["results"][control] for fold in folds]
        aucs = np.asarray([row["heldout"]["auc"] for row in rows], dtype=np.float64)
        fit_aucs = np.asarray([row["fit"]["auc"] for row in rows], dtype=np.float64)
        gates = [row.get("final_gine_gate") for row in rows]
        aggregate[control] = {
            "fold_auc": aucs.tolist(),
            "mean_auc": float(aucs.mean()),
            "std_auc": float(aucs.std(ddof=1)),
            "fold_fit_auc": fit_aucs.tolist(),
            "mean_fit_auc": float(fit_aucs.mean()),
            "fold_final_tanh_gine_gate": gates,
            "mean_final_tanh_gine_gate": (
                float(np.mean(gates)) if all(gate is not None for gate in gates) else None
            ),
        }

    pair_specs = (
        ("random_occ_mil", "random_node_mil"),
        ("random_occ_gine", "random_occ_mil"),
        ("random_occ_gated_gine", "random_occ_mil"),
        ("random_occ_gated_gine", "random_occ_gine"),
        ("ksvd_occ_gine", "ksvd_occ_mil"),
        ("ksvd_occ_gated_gine", "ksvd_occ_mil"),
        ("ksvd_occ_gated_gine", "ksvd_occ_gine"),
        ("ksvd_occ_gine", "ksvd_occ_gine_shuffled_id"),
        ("ksvd_occ_gine", "ksvd_occ_gine_no_id"),
        ("ksvd_occ_gated_gine", "random_occ_gated_gine"),
        ("ksvd_occ_mil", "random_occ_mil"),
    )
    comparisons = {
        f"{left}_minus_{right}": _pair(aggregate, left, right)
        for left, right in pair_specs
    }

    random_gain = comparisons["random_occ_gated_gine_minus_random_occ_mil"]
    ksvd_gain = comparisons["ksvd_occ_gated_gine_minus_ksvd_occ_mil"]
    ksvd_vs_random = comparisons[
        "ksvd_occ_gated_gine_minus_random_occ_gated_gine"
    ]
    identity_shuffled = comparisons[
        "ksvd_occ_gine_minus_ksvd_occ_gine_shuffled_id"
    ]
    identity_no_id = comparisons["ksvd_occ_gine_minus_ksvd_occ_gine_no_id"]

    def family_gate(pair: dict[str, Any]) -> dict[str, Any]:
        return {
            "mean_gain_at_least_005": bool(pair["mean_delta_auc"] >= 0.005),
            "at_least_2_of_3_wins": bool(pair["wins"] >= 2),
            "no_severe_collapse": bool(
                pair["min_delta_auc"] > args.severe_collapse_threshold
            ),
            "severe_collapse_threshold": args.severe_collapse_threshold,
        }

    random_promotion = family_gate(random_gain)
    random_promotion["passes"] = bool(all(
        random_promotion[key]
        for key in (
            "mean_gain_at_least_005",
            "at_least_2_of_3_wins",
            "no_severe_collapse",
        )
    ))
    ksvd_promotion = family_gate(ksvd_gain)
    ksvd_promotion.update({
        "beats_random_gated_mean": bool(ksvd_vs_random["mean_delta_auc"] > 0),
        "ungated_identity_beats_shuffled_all_folds": bool(identity_shuffled["wins"] == 3),
        "ungated_identity_beats_no_id_all_folds": bool(identity_no_id["wins"] == 3),
    })
    ksvd_promotion["passes"] = bool(all(
        ksvd_promotion[key]
        for key in (
            "mean_gain_at_least_005",
            "at_least_2_of_3_wins",
            "no_severe_collapse",
            "beats_random_gated_mean",
            "ungated_identity_beats_shuffled_all_folds",
            "ungated_identity_beats_no_id_all_folds",
        )
    ))

    occurrence_stats: dict[str, Any] = {}
    for family in ("random", "ksvd"):
        keys = sorted(by_fold[0]["occurrence_stats"][family])
        occurrence_stats[family] = {}
        for key in keys:
            values = [by_fold[fold]["occurrence_stats"][family][key] for fold in folds]
            occurrence_stats[family][f"fold_{key}"] = values
            if all(isinstance(value, (int, float)) for value in values):
                occurrence_stats[family][f"mean_{key}"] = float(np.mean(values))

    summary = {
        "protocol_id": "molhiv-prototype-occurrence-graph-scaffold3-seed0-summary-v1",
        "date": "2026-07-28",
        "folds": folds,
        "inputs_by_fold": {str(fold): input_by_fold[fold] for fold in folds},
        "aggregate": aggregate,
        "comparisons": comparisons,
        "occurrence_stats": occurrence_stats,
        "promotion": {
            "random_gated_interaction": random_promotion,
            "ksvd_gated_interaction": ksvd_promotion,
            "promote_ksvd_to_multiseed": ksvd_promotion["passes"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
