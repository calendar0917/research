"""Driver for the whole round: Phase A witness, Phase B Train-only test, artifacts.

Usage
-----
    uv run python code/nb_run.py            # run Phase A then Phase B
    uv run python code/nb_run.py --phase a   # Phase A only
    uv run python code/nb_run.py --phase b   # Phase B only (Phase A artifacts must exist)

Phase A failure stops the round *before* Phase B (pre-registered stopping rule).
No property label is read in Phase A.  No Valid/Test label or index is read in
either phase.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import nb_common as C
import nb_cv as CV
import nb_data as D
import nb_features as F
import nb_structural as S


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def phase_a(ledger: C.CpuLedger) -> dict[str, Any]:
    with ledger.timed("phase A: constructive witness + production forward"):
        witness = S.run_phase_a()
    C.write_json(C.ANALYSIS / "structural_witness.json", witness)
    C.log(f"phase A verdict: {witness['verdict']['summary']}")
    return witness


# ---------------------------------------------------------------------------
# feature / data correctness gates
# ---------------------------------------------------------------------------


def correctness_gates(
    data: Mapping[str, Any],
    cache: Mapping[str, Any],
    g: np.ndarray,
    feats: Mapping[str, Any],
    folds: Mapping[str, Any],
    scaffolds: Mapping[str, Any],
) -> dict[str, Any]:
    margins = F.margin_residuals(feats)
    n = int(len(data["labels"]))
    heavy_plus_h = (
        np.asarray(feats["n_heavy_atoms"], dtype=np.int64) + np.asarray(feats["n_hydrogen_nodes_removed"], dtype=np.int64)
    )
    cache_atoms = np.diff(np.asarray(cache["atom_ptr"], dtype=np.int64))
    checks = {
        "official_train_rows_match_frozen_benchmark": {
            "value": bool(data["audit"]["rows_match_frozen_benchmark"]),
            "sha256": data["audit"]["rows_sha256"],
            "frozen": data["audit"]["frozen_rows_sha256"],
            "passed": bool(data["audit"]["rows_match_frozen_benchmark"]),
        },
        "train_size": {
            "value": n,
            "expected": C.OGB_TRAIN_ROWS,
            "passed": bool(n == C.OGB_TRAIN_ROWS),
        },
        "labels_binary": {
            "classes": sorted(np.unique(data["labels"]).tolist()),
            "n_positive": data["audit"]["n_positive"],
            "n_negative": data["audit"]["n_negative"],
            "passed": bool(set(np.unique(data["labels"]).tolist()) == {0, 1}),
        },
        "labels_restricted_to_official_train": {
            "n_labels": int(np.asarray(data["labels"]).size),
            "n_rows": n,
            "valid_test_labels_read": bool(data["audit"]["valid_test_labels_read"]),
            "valid_test_index_files_opened": bool(data["audit"]["valid_test_index_files_opened"]),
            "passed": bool(
                np.asarray(data["labels"]).size == n
                and not data["audit"]["valid_test_labels_read"]
                and not data["audit"]["valid_test_index_files_opened"]
            ),
        },
        "structure_cache_matches_frozen_benchmark": {
            "value": str(cache["meta"].get("content_fingerprint")),
            "frozen": D.FROZEN_TRAIN_CACHE_FINGERPRINT,
            "n_molecules": int(cache["n_molecules"]),
            "labels_present": bool(cache["meta"].get("labels_present")),
            "passed": bool(
                str(cache["meta"].get("content_fingerprint")) == D.FROZEN_TRAIN_CACHE_FINGERPRINT
                and int(cache["n_molecules"]) == n
                and not cache["meta"].get("labels_present")
            ),
        },
        "structure_cache_rows_equal_official_train_rows": {
            "passed": bool([int(x) for x in np.asarray(cache["rows"]).tolist()] == [int(x) for x in data["rows"]]),
        },
        "heavy_atom_graph_reproduces_frozen_sample_set": {
            "n_rows_checked": int(n),
            "n_mismatch": int((heavy_plus_h != cache_atoms).sum()),
            "passed": bool(np.array_equal(heavy_plus_h, cache_atoms)),
            "note": "heavy atoms + explicit hydrogen nodes == the frozen lipo_v1 atom count for every Train molecule",
        },
        "g42_shape_and_finiteness": {
            "shape": list(g.shape),
            "finite": bool(np.isfinite(g).all()),
            "n_columns_with_variation": int((g.std(axis=0) > 0).sum()),
            "passed": bool(g.shape == (n, C.GLOBAL_DIM) and np.isfinite(g).all()),
        },
        "pair_features_shape": {
            "real": list(np.asarray(feats["real"]).shape),
            "null": list(np.asarray(feats["null"]).shape),
            "passed": bool(
                np.asarray(feats["real"]).shape == (n, C.N_PAIR_FEATURES)
                and np.asarray(feats["null"]).shape == (n, C.N_PAIR_FEATURES)
            ),
        },
        "pair_features_finite_nonnegative": {
            "finite": bool(np.isfinite(feats["real"]).all() and np.isfinite(feats["null"]).all()),
            "min": float(min(np.asarray(feats["real"]).min(), np.asarray(feats["null"]).min())),
            "passed": bool(
                np.isfinite(feats["real"]).all()
                and np.isfinite(feats["null"]).all()
                and np.asarray(feats["real"]).min() >= 0.0
                and np.asarray(feats["null"]).min() >= 0.0
            ),
        },
        "null_margin_conservation": {**margins, "passed": bool(
            margins["row_margin_conserved"]
            and margins["col_margin_conserved"]
            and margins["total_conserved"]
            and margins["zero_pair_rows_all_zero"]
            and margins["raw_row_margin_matches_counts"]
            and margins["raw_col_margin_matches_counts"]
            and margins["raw_total_matches_counts"]
        )},
        "ecfp4_binary": {
            "shape": list(np.asarray(feats["ecfp4"]).shape),
            "unique_values": sorted(np.unique(feats["ecfp4"]).tolist()),
            "n_zero_molecules": int((np.asarray(feats["ecfp4"]).sum(axis=1) == 0).sum()),
            "passed": bool(
                np.asarray(feats["ecfp4"]).shape == (n, C.MORGAN_NBITS)
                and set(np.unique(feats["ecfp4"]).tolist()) <= {0, 1}
            ),
        },
        "scaffold_groups_defined": {
            "n_groups": int(scaffolds["n_groups"]),
            "kind_counts": scaffolds["kind_counts"],
            "empty_groups": int(sum(1 for gname in scaffolds["groups"] if gname == "")),
            "passed": bool(
                scaffolds["n_groups"] > 1 and all(gname != "" for gname in scaffolds["groups"])
            ),
        },
        "folds_feasible_and_partitioned": {
            "status": folds["status"],
            "partition_ok": folds["outer_valid_partition_covers_all_rows"],
            "passed": bool(folds["status"] == "OK" and folds["outer_valid_partition_covers_all_rows"]),
        },
        "no_cross_fold_scaffold_leakage": {
            "checked": "outer train/valid and both inner train/valid splits of every outer fold",
            "passed": bool(
                all(
                    not set(
                        np.unique(scaffolds["codes"][np.asarray(f["train_rows"])]).tolist()
                    ).intersection(set(np.unique(scaffolds["codes"][np.asarray(f["valid_rows"])]).tolist()))
                    for f in folds["folds"]
                )
                and all(
                    not set(
                        np.unique(scaffolds["codes"][np.asarray(g["train_rows"])]).tolist()
                    ).intersection(set(np.unique(scaffolds["codes"][np.asarray(g["valid_rows"])]).tolist()))
                    for f in folds["folds"]
                    for g in f["inner"]
                )
            ),
        },
    }
    all_passed = all(bool(v.get("passed")) for v in checks.values())
    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": int(sum(1 for v in checks.values() if v.get("passed"))),
        "all_passed": bool(all_passed),
        "labels_used_for_features": False,
        "feature_source": "official Train SMILES only; every feature is label-free",
    }


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------


def feature_coverage(feats: Mapping[str, Any]) -> dict[str, Any]:
    real = np.asarray(feats["real"], dtype=np.float64)
    null = np.asarray(feats["null"], dtype=np.float64)
    real3 = real.reshape(-1, C.N_TYPE_PAIRS, C.N_DIST_BINS)
    null3 = null.reshape(-1, C.N_TYPE_PAIRS, C.N_DIST_BINS)
    nonzero_rows = np.asarray(feats["totals"]) > 0
    per_column_real = (real > 0).mean(axis=0)
    per_column_null = (null > 0).mean(axis=0)
    pair_coverage = {
        "C": "C", "N": "N", "O": "O", "S": "S", "X": "X", "OTHER": "OTHER",
    }
    pair_rows = []
    for p, (i, j) in enumerate(C.TYPE_PAIRS):
        block_real = real3[:, p, :]
        block_null = null3[:, p, :]
        pair_rows.append(
            {
                "index": int(p),
                "classes": [C.ELEMENT_CLASSES[i], C.ELEMENT_CLASSES[j]],
                "n_molecules_nonzero_real": int((block_real.sum(axis=1) > 0).sum()),
                "n_molecules_nonzero_null": int((block_null.sum(axis=1) > 0).sum()),
                "real_total_mass": float(block_real.sum()),
                "null_total_mass": float(block_null.sum()),
            }
        )
    bin_rows = []
    for k, label in enumerate(C.DIST_BIN_LABELS):
        bin_rows.append(
            {
                "index": int(k),
                "distance_bin": label,
                "n_molecules_nonzero_real": int((real3[:, :, k].sum(axis=1) > 0).sum()),
                "n_molecules_nonzero_null": int((null3[:, :, k].sum(axis=1) > 0).sum()),
                "real_total_mass": float(real3[:, :, k].sum()),
            }
        )
    return {
        "n_molecules": int(real.shape[0]),
        "n_molecules_with_pairs": int(nonzero_rows.sum()),
        "nonzero_rate_pairs": float(nonzero_rows.mean()),
        "feature_nonzero_rate_real": {
            "overall": float((real > 0).mean()),
            "mean_per_column": float(per_column_real.mean()),
            "min_per_column": float(per_column_real.min()),
            "max_per_column": float(per_column_real.max()),
            "n_columns_never_nonzero": int((per_column_real == 0).sum()),
        },
        "feature_nonzero_rate_null": {
            "overall": float((null > 0).mean()),
            "mean_per_column": float(per_column_null.mean()),
            "min_per_column": float(per_column_null.min()),
            "max_per_column": float(per_column_null.max()),
            "n_columns_never_nonzero": int((per_column_null == 0).sum()),
        },
        "type_pairs_with_any_real_mass": int(sum(1 for r in pair_rows if r["n_molecules_nonzero_real"] > 0)),
        "distance_bins_with_any_real_mass": int(sum(1 for r in bin_rows if r["n_molecules_nonzero_real"] > 0)),
        "per_type_pair": pair_rows,
        "per_distance_bin": bin_rows,
        "total_pairs_ge5": int(np.asarray(feats["totals"], dtype=np.int64).sum()),
        "pairs_ge5_mean_per_molecule": float(np.asarray(feats["totals"], dtype=np.float64).mean()),
        "pairs_ge5_max_per_molecule": int(np.asarray(feats["totals"], dtype=np.int64).max()),
        "class_legend": pair_coverage,
        "distance_bin_lower_bounds": list(C.DIST_BIN_LOWER),
    }


def phase_b(ledger: C.CpuLedger) -> dict[str, Any]:
    with ledger.timed("phase B: official Train-only load"):
        data = D.load_train_only()
        C.log(f"train rows={data['audit']['n_train']} pos={data['audit']['n_positive']} neg={data['audit']['n_negative']}")
        cache = D.load_train_cache(data["smiles"], data["rows"])
        g = D.g42_matrix(cache)

    with ledger.timed("phase B: REAL/NULL/ECFP4 feature construction"):
        feats = F.build_feature_matrices(data["smiles"], label="molhiv-train")
        feats["g"] = g  # frozen 42-d lipo_v1 global block (raw, pre-normalisation)
        coverage = feature_coverage(feats)

    with ledger.timed("phase B: scaffolds + frozen folds"):
        scaf = D.scaffolds(data["smiles"])
        folds = D.build_folds(np.asarray(data["labels"], dtype=np.int64), scaf["codes"])
        C.log(f"folds status={folds['status']} scaffolds={folds['n_scaffolds_total']}")

    correctness = correctness_gates(data, cache, g, feats, folds, scaf)
    parse_modes = [str(m) for m in np.asarray(feats["parse_modes"]).tolist()]
    C.write_json(C.ANALYSIS / "feature_margin_checks.json", {
        "round_id": C.ROUND_ID,
        "phase": "B",
        "created_utc": C.utcnow(),
        "parse_modes": {
            "counts": {m: int(parse_modes.count(m)) for m in sorted(set(parse_modes))},
            "non_strict_rows": {
                "train_row_ids": [int(r) for r, m in zip(data["rows"], parse_modes) if m != "strict"],
                "train_positions": [int(i) for i, m in enumerate(parse_modes) if m != "strict"],
                "smiles": [str(s) for s, m in zip(data["smiles"], parse_modes) if m != "strict"],
                "reason": "invalid-valence metal complexes (Al valence 7, Ge valence 5); see PROTOCOL.md §5.1",
            },
            "rows_dropped": 0,
        },
        "feature_definition": {
            "element_classes": list(C.ELEMENT_CLASSES),
            "halogens_in_X": sorted(C.HALOGENS),
            "n_type_pairs": C.N_TYPE_PAIRS,
            "type_pairs": [[C.ELEMENT_CLASSES[i], C.ELEMENT_CLASSES[j]] for i, j in C.TYPE_PAIRS],
            "distance_bins": list(C.DIST_BIN_LABELS),
            "minimum_pair_distance": C.MIN_PAIR_DISTANCE,
            "unreachable_pairs": "not counted (component-restricted)",
            "normalisation": "P = R / max(1, total_number_of_pairs_with_distance_>=5); all-zero when no such pair",
            "null": "NULL[p,k] = row_margin[p] * col_margin[k] / N, computed on raw counts, then divided by the same max(1,total)",
            "n_features": C.N_PAIR_FEATURES,
        },
        "margin_checks": F.margin_residuals(feats),
        "coverage": coverage,
        "correctness_gates": correctness,
        "labels_used": False,
    })
    C.write_json(C.ANALYSIS / "folds.json", {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "n_train": int(len(data["labels"])),
        "n_positive": int((np.asarray(data["labels"]) == 1).sum()),
        "n_negative": int((np.asarray(data["labels"]) == 0).sum()),
        "scaffolds": {
            "n_groups": int(scaf["n_groups"]),
            "kind_counts": scaf["kind_counts"],
            "mode_counts": scaf["mode_counts"],
            "largest_group_size": int(scaf["largest_group_size"]),
            "singleton_groups": int(scaf["singleton_groups"]),
            "group_key_sha256": C.sha256_bytes(
                np.asarray(scaf["codes"], dtype=np.int64).tobytes()
            ),
        },
        **folds,
    })

    if not correctness["all_passed"]:
        return {"status": "INFEASIBLE", "correctness": correctness, "metrics": None, "bootstrap": None}

    with ledger.timed("phase B: nested grouped CV (3 outer x 2 inner, three models)"):
        fold_results = []
        for fold in folds["folds"]:
            selection = CV.select_c(fold, np.asarray(data["labels"], dtype=np.int64), feats)
            result = CV.run_outer_fold(fold, np.asarray(data["labels"], dtype=np.int64), feats, selection)
            result["valid_rows"] = fold["valid_rows"]
            result["train_rows"] = fold["train_rows"]
            result["selection"] = selection
            fold_results.append(result)
            C.log(
                f"outer fold {result['outer_fold']}: C={result['selected_C']} "
                f"auc BASE={result['branches']['BASE']['roc_auc']:.6f} "
                f"NULL={result['branches']['BASE_NULL']['roc_auc']:.6f} "
                f"REAL={result['branches']['BASE_REAL']['roc_auc']:.6f}"
            )

    y = np.asarray(data["labels"], dtype=np.int64)
    with ledger.timed("phase B: aggregate metrics"):
        metrics = CV.aggregate(fold_results, y)
        metrics["feature_widths"] = CV.feature_widths(C.MORGAN_NBITS)
        metrics["estimator"] = dict(C.LOGREG)
        metrics["C_grid"] = list(C.C_GRID)
        metrics["branch_definitions"] = {
            "BASE": "ECFP4(0/1) + standardised g42",
            "BASE_NULL": "BASE + standardised NULL126",
            "BASE_REAL": "BASE + standardised REAL126",
        }
        metrics["preprocessing"] = {
            "ecfp4": "binary 0/1, not standardised",
            "g42": "per-partition mean/std; zero-variance columns keep denominator 1; then clipped to [-5,5]",
            "pair": "one shared mean/std fitted on the partition's REAL and NULL concatenated; then clipped to [-5,5]",
            "fit_partitions": "inner folds fit on inner Train; outer folds fit on outer Train",
        }
        metrics["selections"] = {
            str(r["outer_fold"]): r["selection"] for r in fold_results
        }
        metrics["n_models_fitted"] = int(
            sum(1 for r in fold_results for _ in C.BRANCHES) + len(C.C_GRID) * C.INNER_SPLITS * len(fold_results)
        )

    with ledger.timed("phase B: paired scaffold bootstrap (1000 replicates)"):
        bootstrap = CV.paired_scaffold_bootstrap(fold_results, y, scaf["codes"])
    bootstrap["created_utc"] = C.utcnow()

    gates = CV.evaluate_gates(metrics, bootstrap, correctness)

    # artifacts
    C.write_json(C.ANALYSIS / "metrics.json", {
        "round_id": C.ROUND_ID,
        "phase": "B",
        "created_utc": C.utcnow(),
        "dataset": "ogbg-molhiv",
        "split": "official Train only (3-fold StratifiedGroupKFold by Bemis-Murcko scaffold)",
        "labels": {
            "n_train": int(y.size),
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
        },
        "metrics": metrics,
        "per_fold_records": [
            {
                "outer_fold": r["outer_fold"],
                "train_rows_sha256": C.rows_sha256(r["train_rows"]),
                "valid_rows_sha256": C.rows_sha256(r["valid_rows"]),
                "branches": {
                    b: {k: v for k, v in r["branches"][b].items()} for b in C.BRANCHES
                },
            }
            for r in fold_results
        ],
        "gates": gates,
    })
    C.write_json(C.ANALYSIS / "bootstrap.json", {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        **bootstrap,
    })

    # out-of-fold predictions (fixed fits), one array per branch and fold
    payload: dict[str, np.ndarray] = {
        "train_rows": np.asarray(data["rows"], dtype=np.int64),
        "y": y,
        "scaffold_codes": np.asarray(scaf["codes"], dtype=np.int64),
        "scaffold_groups": np.asarray(scaf["groups"], dtype=object),
        "selected_C": np.asarray([r["selected_C"] for r in fold_results], dtype=np.float64),
    }
    oof = {b: np.full(y.size, np.nan, dtype=np.float64) for b in C.BRANCHES}
    for r in fold_results:
        rows = np.asarray(r["valid_rows"], dtype=np.int64)
        payload[f"fold{r['outer_fold']}_valid_rows"] = rows
        payload[f"fold{r['outer_fold']}_y"] = y[rows]
        for b in C.BRANCHES:
            payload[f"fold{r['outer_fold']}_pred_{b}"] = np.asarray(r["scores"][b], dtype=np.float64)
            oof[b][rows] = np.asarray(r["scores"][b], dtype=np.float64)
    for b in C.BRANCHES:
        payload[f"oof_{b}"] = oof[b]
    np.savez_compressed(C.ANALYSIS / "oof_predictions.npz", **payload)

    return {
        "status": "OK",
        "data_audit": data["audit"],
        "data_files": data["files"],
        "correctness": correctness,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "gates": gates,
        "coverage": coverage,
        "fold_count": len(fold_results),
        "n_scaffolds": int(scaf["n_groups"]),
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b", "all"), default="all")
    args = parser.parse_args(argv)

    ledger = C.CpuLedger()
    C.log(f"round {C.ROUND_ID} starting")

    provenance: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_head": C.git_head(),
        "git_dirty_tracked": C.git_dirty(),
        "reused_modules": C.reused_module_hashes(),
        "code": C.code_hashes(),
        "environment": C.env_info(),
        "feature_spec_hash": C.FEATURE_SPEC_HASH,
        "gpu_hours_planned": C.GPU_BUDGET_HOURS,
        "frozen_design": {
            "pair_features": {
                "element_classes": list(C.ELEMENT_CLASSES),
                "distance_bins": list(C.DIST_BIN_LABELS),
                "n_features": C.N_PAIR_FEATURES,
            },
            "branches": list(C.BRANCHES),
            "folds": {
                "outer_splits": C.OUTER_SPLITS,
                "inner_splits": C.INNER_SPLITS,
                "seed": C.SPLIT_SEED,
                "inner_seed_rule": "SPLIT_SEED + outer_fold",
                "group_key": "Bemis-Murcko scaffold, canonical-SMILES fallback",
            },
            "logreg": dict(C.LOGREG),
            "C_grid": list(C.C_GRID),
            "clip": C.CLIP,
            "bootstrap": {
                "replicates": C.BOOTSTRAP_REPLICATES,
                "max_attempts": C.BOOTSTRAP_MAX_ATTEMPTS,
                "seed": C.BOOTSTRAP_SEED,
            },
            "gates": {
                "real_minus_null_auc_mean": C.GATE_REAL_MINUS_NULL_AUC_MEAN,
                "real_minus_base_auc_mean": C.GATE_REAL_MINUS_BASE_AUC_MEAN,
                "real_minus_null_ap_mean": C.GATE_REAL_MINUS_NULL_AP_MEAN,
                "all_fold_diffs_positive": C.GATE_ALL_FOLD_DIFFS_POSITIVE,
                "bootstrap_ci_lower": "> 0",
            },
            "phase_a": {
                "ring_size": C.RING_SIZE,
                "nitrogen_index_A": C.NITROGEN_INDEX_A,
                "nitrogen_index_B": C.NITROGEN_INDEX_B,
                "variants": list(C.FORWARD_VARIANTS),
                "seeds": list(C.FORWARD_SEEDS),
                "prediction_tolerance": C.PREDICTION_TOL,
            },
        },
        "command": "uv run python tracks/structural-sufficiency/lcrc_nonlocal_binding_bottleneck_v1/code/nb_run.py",
    }

    witness = None
    if args.phase in ("a", "all"):
        witness = phase_a(ledger)
        if not witness["gates"]["forward_equivalent"]:
            provenance["stopped_at"] = "phase A"
            provenance["stop_reason"] = "production forward equivalence failed"
            C.write_json(C.ANALYSIS / "provenance.json", provenance)
            C.write_json(C.ANALYSIS / "cpu_ledger.json", ledger.to_dict(phase="A", stop_reason="phase A failure"))
            C.log("STOP: phase A forward equivalence failed; Phase B not run")
            return 2
        if not all(witness["gates"].values()):
            failed = [k for k, v in witness["gates"].items() if not v]
            provenance["stopped_at"] = "phase A"
            provenance["stop_reason"] = f"phase A structural gates failed: {failed}"
            C.write_json(C.ANALYSIS / "provenance.json", provenance)
            C.write_json(C.ANALYSIS / "cpu_ledger.json", ledger.to_dict(phase="A", stop_reason="phase A gate failure"))
            C.log(f"STOP: phase A gates failed: {failed}")
            return 2

    result = None
    if args.phase in ("b", "all"):
        if witness is None:
            path = C.ANALYSIS / "structural_witness.json"
            if not path.exists():
                raise SystemExit("Phase B requires Phase A artifacts; run --phase a first")
            witness = C.read_json(path)
            if not witness["gates"]["forward_equivalent"]:
                raise SystemExit("Phase A failed; the pre-registered rule forbids running Phase B")
        result = phase_b(ledger)
        if result["status"] != "OK":
            provenance["stopped_at"] = "phase B correctness gates"
            provenance["stop_reason"] = "data/feature correctness gates failed"
            C.write_json(C.ANALYSIS / "provenance.json", provenance)
            C.write_json(C.ANALYSIS / "cpu_ledger.json", ledger.to_dict(phase="B", stop_reason="gate failure"))
            C.log("STOP: phase B correctness gates failed -> INFEASIBLE")
            return 3

    provenance["phase_a_gates"] = None if witness is None else witness["gates"]
    if result is not None:
        provenance["correctness_gates"] = {k: bool(v.get("passed")) for k, v in result["correctness"]["checks"].items()}
        provenance["data"] = {
            "official_train_rows_sha256": result["data_audit"]["rows_sha256"],
            "n_train": result["data_audit"]["n_train"],
            "files": result["data_files"],
            "valid_test_labels_read": result["data_audit"]["valid_test_labels_read"],
            "valid_test_index_files_opened": result["data_audit"]["valid_test_index_files_opened"],
            "structure_cache_fingerprint": result["correctness"]["checks"]["structure_cache_matches_frozen_benchmark"]["value"],
            "feature_spec_hash": C.FEATURE_SPEC_HASH,
        }
        provenance["decision"] = result["gates"]["decision"]
        provenance["model_init_state_hashes"] = (
            {} if witness is None else {k: v["init_state_hash"] for k, v in witness["forward"].items()}
        )
    C.write_json(C.ANALYSIS / "provenance.json", provenance)
    C.write_json(C.ANALYSIS / "cpu_ledger.json", ledger.to_dict(phase=args.phase))

    if result is not None:
        g = result["gates"]
        C.log(
            f"decision={g['decision']} ({g['reason']}); passed {g['n_passed']}/{g['n_checks']} gates; "
            f"REAL-NULL mean AUC diff={g['checks']['g1_mean_auc_real_minus_null_ge_0.015']['value']:.6f}"
        )
    C.log(f"cpu wall={ledger.to_dict()['cpu_wall_seconds_total']:.1f}s (budget {C.CPU_SOFT_CAP_HOURS} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
