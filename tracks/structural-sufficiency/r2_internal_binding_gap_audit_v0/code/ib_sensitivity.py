"""Frozen EGO2 input-sensitivity: remove distance-2--2 or distance-1--1 edges.

No gradient, no parameter update.  Only the packed ego connectivity changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import ib_common as C


def _root_changes(orig: np.ndarray, var: np.ndarray) -> dict[str, np.ndarray]:
    diff = np.asarray(var, np.float64) - np.asarray(orig, np.float64)
    l2 = np.linalg.norm(diff, axis=1)
    absmean = np.mean(np.abs(diff), axis=1)
    n0 = np.linalg.norm(orig, axis=1)
    n1 = np.linalg.norm(var, axis=1)
    denom = n0 * n1
    dot = np.sum(orig * var, axis=1)
    cos = np.ones_like(dot, dtype=np.float64)
    nz = denom > 0
    cos[nz] = dot[nz] / denom[nz]
    return {"l2": l2, "absmean": absmean, "cosine": cos}


def _stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _summary(orig_pred: np.ndarray, var_pred: np.ndarray, y: np.ndarray,
             orig_roots: np.ndarray, var_roots: np.ndarray,
             root_row_id: np.ndarray, root_any_removed: np.ndarray,
             molecule_any_removed: np.ndarray, removed_total: np.ndarray,
             total_edges: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pred_diff = np.asarray(var_pred, np.float64) - np.asarray(orig_pred, np.float64)
    abs_pred = np.abs(pred_diff)
    root = _root_changes(orig_roots, var_roots)
    rmse_before = C.rmse(orig_pred, y)
    rmse_after = C.rmse(var_pred, y)
    coverage = {
        "n_molecules": int(molecule_any_removed.size),
        "molecules_with_removed_edge": int(molecule_any_removed.sum()),
        "molecule_coverage": float(molecule_any_removed.mean()),
        "n_roots": int(root_any_removed.size),
        "roots_with_removed_edge": int(root_any_removed.sum()),
        "root_coverage": float(root_any_removed.mean()),
        "edges_removed": int(removed_total.sum()),
        "ego_edge_occurrences": int(total_edges.sum()),
        "edge_fraction_removed": (float(removed_total.sum() / total_edges.sum())
                                  if total_edges.sum() else 0.0),
        "mean_removed_per_molecule": float(removed_total.mean()),
        "median_removed_per_molecule": float(np.median(removed_total)),
        "max_removed_per_molecule": int(removed_total.max()),
    }
    affected_mol = molecule_any_removed
    affected_root = root_any_removed
    zero_mol = ~molecule_any_removed
    zero_root = ~root_any_removed
    summary = {
        "coverage": coverage,
        "replay": {
            "rmse_before": rmse_before,
            "rmse_after": rmse_after,
            "delta_rmse": rmse_after - rmse_before,
        },
        "prediction_change": {
            "all": _stats(abs_pred),
            "affected_molecules": _stats(abs_pred[affected_mol]),
            "zero_removal_molecules_max_abs": (
                float(abs_pred[zero_mol].max()) if np.any(zero_mol) else 0.0
            ),
        },
        "root_change": {
            "l2_all": _stats(root["l2"]),
            "l2_affected_roots": _stats(root["l2"][affected_root]),
            "absmean_all": _stats(root["absmean"]),
            "absmean_affected_roots": _stats(root["absmean"][affected_root]),
            "cosine_all": _stats(root["cosine"]),
            "cosine_affected_roots": _stats(root["cosine"][affected_root]),
            "zero_removal_roots_max_l2": (
                float(root["l2"][zero_root].max()) if np.any(zero_root) else 0.0
            ),
        },
        "coverage_sufficient": bool(coverage["molecule_coverage"] >= 0.10),
        "interpret_performance": bool(coverage["molecule_coverage"] >= 0.10),
    }
    arrays = {
        "pred_diff": pred_diff,
        "root_l2": root["l2"],
        "root_absmean": root["absmean"],
        "root_cosine": root["cosine"],
        "root_row_id": np.asarray(root_row_id, np.int64),
        "root_any_removed": np.asarray(root_any_removed, bool),
    }
    return summary, arrays


def main() -> None:
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    C.assert_inputs()
    bundles = {split: C.load_bundle(split) for split in C.SPLITS}
    summary: dict[str, Any] = {
        "schema": "r2-internal-binding-gap-audit-sensitivity-v1",
        "created_utc": C.now_utc(),
        "note": ("Only the ego connectivity tensors change.  The perturbed graphs "
                 "are not legal chemical edits and their performance change is not "
                 "interpreted as a real bond-deletion effect."),
        "cells": {},
    }
    molecule_rows: list[dict[str, Any]] = []

    for split, seed in C.CELLS:
        bundle = bundles[split]
        rows = np.asarray(bundle.eval_rows, np.int64)
        run_id = C.RUNS[(split, seed)]["EGO2"]
        stored = C.load_predictions(run_id, bundle)
        model, ck = C.load_ego2_model(run_id, bundle)
        import sr_struct as SS

        structs = SS.build_structures(bundle, rows.tolist())
        original, variants, edge_stats = C.build_ego_variants(structs, rows)
        pred_orig, roots_orig = C.predict_ego2(model, rows, original, bundle,
                                               return_roots=True)
        replay_max = float(np.max(np.abs(pred_orig - stored["pred"])))
        if replay_max >= 1e-5:
            raise AssertionError(
                f"{split}/S{seed}: original replay max |delta| {replay_max} >= 1e-5"
            )
        root_rows = C.root_row_ids(original, rows)
        cell: dict[str, Any] = {
            "split": split,
            "seed": int(seed),
            "run_ego2": run_id,
            "checkpoint_epoch": int(ck["epoch"]),
            "n_molecules": int(rows.size),
            "replay_max_abs": replay_max,
            "variants": {},
        }
        for drop in ("22", "11"):
            pred_var, roots_var = C.predict_ego2(model, rows, variants[drop], bundle,
                                                 return_roots=True)
            removed_total = np.asarray([
                int(edge_stats[drop][int(r)]["removed_per_root"].sum()) for r in rows
            ], np.int64)
            total_edges = np.asarray([
                int(edge_stats[drop][int(r)]["total_per_root"].sum()) for r in rows
            ], np.int64)
            any_removed = removed_total > 0
            root_any_removed = np.repeat(any_removed, [
                int(original[int(r)].n_atoms) for r in rows
            ])
            rec, arrays = _summary(
                pred_orig, pred_var, stored["y"], roots_orig, roots_var,
                root_rows, root_any_removed, any_removed, removed_total, total_edges
            )
            rec["coverage"]["removed_edges_by_type"] = drop
            cell["variants"][drop] = rec
            for i, row in enumerate(rows.tolist()):
                molecule_rows.append({
                    "split": split,
                    "seed": int(seed),
                    "variant": drop,
                    "row_id": int(row),
                    "n_atoms": int(original[int(row)].n_atoms),
                    "n_roots": int(original[int(row)].n_atoms),
                    "removed_edges": int(removed_total[i]),
                    "total_ego_edge_occurrences": int(total_edges[i]),
                    "removed_roots": int(np.sum(
                        edge_stats[drop][int(row)]["removed_per_root"] > 0)),
                    "any_removed": bool(any_removed[i]),
                    "pred_orig": float(pred_orig[i]),
                    "pred_var": float(pred_var[i]),
                    "pred_diff": float(arrays["pred_diff"][i]),
                    "abs_pred_diff": float(abs(arrays["pred_diff"][i])),
                    "label": float(stored["y"][i]),
                })
            np.savez_compressed(
                C.ANALYSIS_DIR / f"sensitivity_roots_{split}_S{seed}_drop{drop}.npz",
                **{k: v for k, v in arrays.items() if k.startswith("root_")},
            )
        summary["cells"][f"{split}_S{seed}"] = cell

    C.write_json(C.ANALYSIS_DIR / "sensitivity_summary.json", summary)
    C.write_csv(C.ANALYSIS_DIR / "sensitivity_molecules.csv", molecule_rows)

    print("frozen EGO2 topology sensitivity")
    for split, seed in C.CELLS:
        cell = summary["cells"][f"{split}_S{seed}"]
        print(f"  {split}/S{seed}: replay max |delta|={cell['replay_max_abs']:.2e}")
        for drop in ("22", "11"):
            rec = cell["variants"][drop]
            cov = rec["coverage"]
            pc = rec["prediction_change"]["all"]
            rc = rec["root_change"]["l2_all"]
            print(f"    drop{drop}: molecule coverage={cov['molecule_coverage']:.3f}, "
                  f"edge fraction={cov['edge_fraction_removed']:.4f}, "
                  f"mean |dpred|={pc['mean']:.6f}, "
                  f"RMSE {rec['replay']['rmse_before']:.6f} -> "
                  f"{rec['replay']['rmse_after']:.6f} "
                  f"({rec['replay']['delta_rmse']:+.6f}); "
                  f"root L2 mean={rc['mean']:.6f}; "
                  f"interpret={rec['interpret_performance']}")


if __name__ == "__main__":
    main()