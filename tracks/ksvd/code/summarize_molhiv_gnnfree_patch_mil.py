"""Summarize the strict 3-fold x 3-seed GNN-free raw-patch PCA MIL screen.

The new models are evaluated exactly once after epoch 30 on official-train-only
Bemis--Murcko scaffold folds.  An older epoch-selected GINE matrix can be
included as an explicitly optimistic reference.  Optional fixed-epoch GINE
reports provide the protocol-matched architectural comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CONTROLS = ("random_node_mil", "ksvd_node_mil")


def _matrix_stats(matrix: np.ndarray, fit_matrix: np.ndarray | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "fold_by_seed_auc": matrix.tolist(),
        "grand_mean_auc": float(matrix.mean()),
        "grand_std_auc": float(matrix.std(ddof=1)),
        "fold_mean_auc": matrix.mean(axis=1).tolist(),
        "fold_std_auc": matrix.std(axis=1, ddof=1).tolist(),
        "seed_mean_auc": matrix.mean(axis=0).tolist(),
        "seed_std_auc": matrix.std(axis=0, ddof=1).tolist(),
        "min_auc": float(matrix.min()),
        "max_auc": float(matrix.max()),
    }
    if fit_matrix is not None:
        gap = fit_matrix - matrix
        out.update({
            "fold_by_seed_fit_auc": fit_matrix.tolist(),
            "grand_mean_fit_auc": float(fit_matrix.mean()),
            "fold_by_seed_fit_minus_heldout_auc": gap.tolist(),
            "mean_fit_minus_heldout_auc": float(gap.mean()),
        })
    return out


def _comparison(left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> dict[str, Any]:
    delta = left - right
    return {
        "left": left_name,
        "right": right_name,
        "fold_by_seed_delta_auc": delta.tolist(),
        "mean_delta_auc": float(delta.mean()),
        "std_delta_auc": float(delta.std(ddof=1)),
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
        "n_pairs": int(delta.size),
        "fold_mean_delta_auc": delta.mean(axis=1).tolist(),
        "seed_mean_delta_auc": delta.mean(axis=0).tolist(),
        "min_delta_auc": float(delta.min()),
        "max_delta_auc": float(delta.max()),
    }


def _load_fixed_gine(paths: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    heldout = np.full((3, 3), np.nan, dtype=np.float64)
    fit = np.full((3, 3), np.nan, dtype=np.float64)
    inputs: dict[str, list[str]] = {}
    for path in paths:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        fold, seed = int(row["fold"]), int(row["seed"])
        if not (0 <= fold < 3 and 0 <= seed < 3):
            raise ValueError(f"fixed GINE fold/seed outside 0..2: {path}")
        if np.isfinite(heldout[fold, seed]):
            raise ValueError(f"duplicate fixed GINE cell fold={fold}, seed={seed}")
        if row.get("official_valid_evaluations") != 0 or row.get("official_test_evaluations") != 0:
            raise ValueError(f"nonzero official evaluation count in {path}")
        heldout[fold, seed] = float(row["heldout"]["auc"])
        fit[fold, seed] = float(row["fit"]["auc"])
        inputs.setdefault(str(fold), []).append(path)
    if not np.isfinite(heldout).all():
        raise ValueError(f"incomplete fixed GINE matrix:\n{heldout}")
    return heldout, fit, inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--legacy-gine-summary")
    ap.add_argument("--fixed-gine", nargs="*", default=[])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    heldout = {name: np.full((3, 3), np.nan, dtype=np.float64) for name in CONTROLS}
    fit = {name: np.full((3, 3), np.nan, dtype=np.float64) for name in CONTROLS}
    inputs_by_cell: dict[str, list[str]] = {}
    split_hashes: dict[int, tuple[str, str]] = {}

    for path in args.inputs:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        fold = int(report["fold"])
        seed = int(report["config"]["seed"])
        if not (0 <= fold < 3 and 0 <= seed < 3):
            raise ValueError(f"fold/seed outside 0..2: {path}")
        if report["selection_policy"].get("official_valid_evaluations") != 0:
            raise ValueError(f"official-valid was evaluated in {path}")
        if report["selection_policy"].get("official_test_evaluations") != 0:
            raise ValueError(f"official-test was evaluated in {path}")
        hashes = (report["fit_indices_sha256"], report["heldout_indices_sha256"])
        if fold in split_hashes and hashes != split_hashes[fold]:
            raise ValueError(f"split hash mismatch in fold {fold}")
        split_hashes[fold] = hashes
        inputs_by_cell.setdefault(f"fold{fold}_seed{seed}", []).append(path)
        for name, row in report["results"].items():
            if name not in CONTROLS:
                continue
            if np.isfinite(heldout[name][fold, seed]):
                raise ValueError(f"duplicate {name}, fold={fold}, seed={seed}")
            heldout[name][fold, seed] = float(row["heldout"]["auc"])
            fit[name][fold, seed] = float(row["fit"]["auc"])

    for name in CONTROLS:
        if not np.isfinite(heldout[name]).all():
            raise ValueError(f"incomplete {name} matrix:\n{heldout[name]}")

    aggregate = {
        name: _matrix_stats(heldout[name], fit[name]) for name in CONTROLS
    }
    comparisons: dict[str, Any] = {
        "random_node_mil_minus_ksvd_node_mil": _comparison(
            heldout["random_node_mil"], heldout["ksvd_node_mil"],
            "random_node_mil", "ksvd_node_mil",
        )
    }
    references: dict[str, Any] = {}

    if args.legacy_gine_summary:
        legacy = json.loads(Path(args.legacy_gine_summary).read_text(encoding="utf-8"))
        matrix = np.full((3, 3), np.nan, dtype=np.float64)
        epoch_matrix = np.full((3, 3), np.nan, dtype=np.float64)
        for row in legacy["nine_pair_confirmation"]["rows"]:
            matrix[int(row["fold"]), int(row["seed"])] = float(row["gine_auc"])
            epoch_matrix[int(row["fold"]), int(row["seed"])] = int(row["gine_epoch"])
        if not np.isfinite(matrix).all():
            raise ValueError("legacy GINE matrix incomplete")
        references["legacy_epoch_selected_gine"] = {
            **_matrix_stats(matrix),
            "selected_epoch_fold_by_seed": epoch_matrix.astype(int).tolist(),
            "comparison_caveat": (
                "This older reference selected the best held-out epoch and is therefore "
                "optimistic relative to the fixed-epoch MIL protocol."
            ),
            "source": args.legacy_gine_summary,
        }
        comparisons["random_node_mil_minus_legacy_epoch_selected_gine"] = _comparison(
            heldout["random_node_mil"], matrix,
            "random_node_mil", "legacy_epoch_selected_gine",
        )
        comparisons["ksvd_node_mil_minus_legacy_epoch_selected_gine"] = _comparison(
            heldout["ksvd_node_mil"], matrix,
            "ksvd_node_mil", "legacy_epoch_selected_gine",
        )

    if args.fixed_gine:
        fixed, fixed_fit, fixed_inputs = _load_fixed_gine(args.fixed_gine)
        references["fixed_epoch30_original_node_gine"] = {
            **_matrix_stats(fixed, fixed_fit),
            "inputs_by_fold": fixed_inputs,
            "comparison_caveat": (
                "Protocol-matched fixed 30-epoch baseline; architecture uses three "
                "supervised original-node GINE layers and no KSVD features."
            ),
        }
        comparisons["random_node_mil_minus_fixed_epoch30_gine"] = _comparison(
            heldout["random_node_mil"], fixed,
            "random_node_mil", "fixed_epoch30_original_node_gine",
        )
        comparisons["ksvd_node_mil_minus_fixed_epoch30_gine"] = _comparison(
            heldout["ksvd_node_mil"], fixed,
            "ksvd_node_mil", "fixed_epoch30_original_node_gine",
        )

    random_minus_ksvd = comparisons["random_node_mil_minus_ksvd_node_mil"]
    decision = {
        "gnn_free_random_node_mil_is_promising": bool(
            aggregate["random_node_mil"]["grand_mean_auc"] >= 0.72
        ),
        "random_real_prototypes_beat_ksvd_majority": bool(
            random_minus_ksvd["wins"] >= 5
        ),
        "random_real_prototypes_beat_ksvd_mean": bool(
            random_minus_ksvd["mean_delta_auc"] > 0
        ),
        "ksvd_specific_superiority_established": bool(
            random_minus_ksvd["mean_delta_auc"] < 0 and random_minus_ksvd["losses"] >= 5
        ),
    }
    if "random_node_mil_minus_fixed_epoch30_gine" in comparisons:
        matched = comparisons["random_node_mil_minus_fixed_epoch30_gine"]
        decision.update({
            "gnn_free_random_beats_matched_gine_mean": bool(matched["mean_delta_auc"] > 0),
            "gnn_free_random_beats_matched_gine_majority": bool(matched["wins"] >= 5),
        })

    summary = {
        "protocol_id": "molhiv-gnnfree-rawpatch-pca64-node-mil-scaffold3-seed3-summary-v1",
        "date": "2026-07-28",
        "matrix_layout": "rows=folds 0..2, columns=seeds 0..2",
        "data_policy": {
            "data": "official-train only",
            "heldout": "three fixed Bemis-Murcko scaffold folds",
            "epoch_policy": "single evaluation after fixed epoch 30",
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
        },
        "architecture": {
            "local_representation": "raw permutation-invariant radius-2 descriptor plus fold-fit label-free PCA64",
            "supervised_message_passing_layers": 0,
            "unsupervised_message_passing_layers": 0,
            "prototype_count": 32,
            "readout": "node-level prototype MIL with attention plus mean plus max",
        },
        "inputs_by_cell": inputs_by_cell,
        "split_hashes": {str(k): {"fit": v[0], "heldout": v[1]} for k, v in split_hashes.items()},
        "aggregate": aggregate,
        "references": references,
        "comparisons": comparisons,
        "decision": decision,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
