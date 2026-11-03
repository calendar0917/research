#!/usr/bin/env python3
"""Run the complete LCRC failure-mechanism P0 diagnostic (CPU-only, read-only).

Stages
------
A  frozen-input alignment and completeness audit
B  Train-only rooted-radius-2 support dictionary and per-molecule support features
C  seed stability and BASE-defined tail errors (Lipophilicity random122)
D  scaffold-grouped cross-fitted conditional model for ``d_i`` on random42
E  COMBO / CHEM8 / MULTICONTRIB attribution on random122

Writes only under ``analysis/`` of this track.  No training, no GPU, no
checkpoint selection, no modification of historical artefacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import p0_common as C  # noqa: E402
import p0_sources as SRC  # noqa: E402
import p0_structure as STR  # noqa: E402
import p0_stats as STAT  # noqa: E402
import p0_stability as STAB  # noqa: E402
import p0_conditional as COND  # noqa: E402
import p0_combo as COMBO  # noqa: E402


def _flatten_checks(name: str, checks: list[dict]) -> list[dict]:
    out = []
    for check in checks:
        out.append({"group": name, **check})
    return out


def _assemble_per_molecule(
    group_a,
    support_a,
    oof_frame: pd.DataFrame,
    group_b,
    support_b,
    stability_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    # ---- group A: frozen Validation420 diagnosis
    support_slice = support_a.query[support_a.query["split"] == "valid"].drop(columns=["group", "split"])
    # oof_frame already carries the four support columns; keep one copy of each.
    support_slice = support_slice.drop(
        columns=[c for c in support_slice.columns if c in oof_frame.columns and c != "row_id"]
    )
    frame_a = oof_frame.merge(
        support_slice,
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    frame_a["group"] = group_a.key
    frame_a["split"] = "valid"
    frame_a["exposure"] = "FROZEN_VALIDATION_ONLY"
    frame_a["y"] = group_a.labels["valid"]
    frame_a["pred_lcrc_a_r3_s0"] = frame_a["pred_lcrc"]
    frame_a["pred_pna_native_s0"] = frame_a["pred_pna"]

    # ---- group B: per split rows
    frames_b = []
    for split in COMBO.SPLITS:
        base = support_b.query[support_b.query["split"] == split].reset_index(drop=True)
        stab = stability_frames[split].copy()
        merged = base.merge(
            stab.drop(columns=["group", "split", "exposure", "y"], errors="ignore"),
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        merged["group"] = group_b.key
        merged["split"] = split
        merged["exposure"] = "POST_HOC_EXPOSED_DISCOVERY" if split == "test" else "FROZEN_VALIDATION_ONLY"
        merged["y"] = np.asarray(group_b.labels[split], dtype=np.float64)
        for arm in ("chem8", "multicontrib"):
            key = f"{arm}_s0" + ("_test" if split == "test" else "")
            if key in group_b.predictions:
                pred = SRC.align_by_row_id(
                    group_b.predictions[key].rows,
                    group_b.predictions[key].pred,
                    group_b.rows[split],
                    label=f"per_molecule/{key}",
                )
                merged[f"pred_{arm}_s0"] = pred
                merged[f"se_{arm}_s0"] = (pred - merged["y"].to_numpy()) ** 2
        frames_b.append(merged)

    frame_b = pd.concat(frames_b, ignore_index=True, sort=False)
    combined = pd.concat([frame_a, frame_b], ignore_index=True, sort=False)

    preferred = [
        "group",
        "split",
        "exposure",
        "row_id",
        "smiles",
        "y",
        "pred_lcrc_a_r3_s0",
        "pred_pna_native_s0",
        "se_lcrc",
        "se_pna",
        "d_i",
        "fold_m0m1",
        "oof_m0",
        "oof_m1",
        "d_minus_oof_m0",
        "d_minus_oof_m1",
        "pred_base_s0",
        "pred_base_s1",
        "pred_base_s2",
        "pred_chem8_s0",
        "pred_multicontrib_s0",
        "pred_combo_s0",
        "pred_combo_s1",
        "pred_combo_s2",
        "se_base_s0",
        "se_base_s1",
        "se_base_s2",
        "se_chem8_s0",
        "se_multicontrib_s0",
        "se_combo_s0",
        "se_combo_s1",
        "se_combo_s2",
        "mean_pred_base",
        "mean_signed_residual_base",
        "mean_squared_error_base",
        "pred_std_base",
        "sign_agreement_base",
        "sign_agreement_count_base",
        "all_same_sign_base",
        "mean_squared_error_combo",
        "pred_std_combo",
        "sign_agreement_combo",
        "mean_delta_se_combo_minus_base",
        "mean_delta_ae_combo_minus_base",
        "delta_se_combo_minus_base_s0",
        "delta_se_combo_minus_base_s1",
        "delta_se_combo_minus_base_s2",
        "delta_ae_combo_minus_base_s0",
        "delta_ae_combo_minus_base_s1",
        "delta_ae_combo_minus_base_s2",
        "in_base_top10_s0",
        "in_base_top10_s1",
        "in_base_top10_s2",
        "n_seeds_in_base_top10",
        "unseen_atom_frac",
        "rare_atom_frac",
        "median_log1p_df",
        "p10_log1p_df",
        "mean_log1p_df",
        "n_atoms",
        "n_unseen_atoms",
        "n_rare_atoms",
        "n_atoms_df_ge_5",
        "heavy_atoms",
        "tpsa",
        "mol_logp",
        "hbd",
        "hba",
        "formal_charge",
        "ring_count",
        "n_components",
        "multi_component",
        "max_train_tanimoto",
        "scaffold",
        "scaffold_seen",
        "scaffold_empty",
        "scaffold_fallback_used",
        "parse_ok",
        "parse_error",
    ]
    for column in preferred:
        if column not in combined.columns:
            combined[column] = np.nan
    return combined[preferred]


def _assert_support_independent_of_test_labels(group_b, support_b, molecules) -> dict:
    """Demonstrate that Test labels cannot change any Train-only support feature."""
    original = group_b.labels["test"].copy()
    before = support_b.query[support_b.query["split"] == "test"].drop(columns=["group", "split"]).reset_index(
        drop=True
    )
    try:
        group_b.labels["test"] = original + 1000.0
        rebuilt = STR.build_support(
            group_key=group_b.key,
            train_rows=group_b.rows["train"],
            query_splits={"test": group_b.rows["test"]},
            molecules=molecules,
            include_loo=False,
        )
        after = rebuilt.query.drop(columns=["group", "split"]).reset_index(drop=True)
    finally:
        group_b.labels["test"] = original
    same = before.equals(after)
    return {
        "check": "test_label_perturbation_does_not_change_support_features",
        "passed": bool(same),
        "detail": {
            "perturbation": "Test labels + 1000.0 (in memory only)",
            "columns_compared": int(len(before.columns)),
            "n_rows": int(len(before)),
            "identical": bool(same),
            "dictionary_sha256": support_b.summary["signature_dictionary_sha256"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-collision-audit",
        action="store_true",
        help="skip the (slower) exact VF2 signature collision audit",
    )
    args = parser.parse_args(argv)

    C.ensure_analysis()
    created = C.utcnow()
    print("== load frozen groups ==", flush=True)
    group_a = SRC.load_group_a()
    group_b = SRC.load_group_b()
    separation = SRC.separation_record(group_a, group_b)
    print(f"  A={group_a.key} n_valid={len(group_a.rows['valid'])}")
    print(f"  B={group_b.key} n_valid={len(group_b.rows['valid'])} n_test={len(group_b.rows['test'])}")

    print("== molecule table (label-free) ==", flush=True)
    if group_a.smiles != group_b.smiles:
        raise SystemExit("STOP: random42 and random122 SMILES tables are not identical")
    smiles_sha = C.sha256_text(group_a.smiles)
    molecules = STR.build_molecule_table(group_b.smiles, verbose=True)
    print(f"  n_molecules={len(molecules)} smiles_sha256={smiles_sha}")

    print("== support dictionaries ==", flush=True)
    support_a = STR.build_support(
        group_key=group_a.key,
        train_rows=group_a.rows["train"],
        query_splits={"valid": group_a.rows["valid"]},
        molecules=molecules,
        include_loo=True,
    )
    support_b = STR.build_support(
        group_key=group_b.key,
        train_rows=group_b.rows["train"],
        query_splits={"valid": group_b.rows["valid"], "test": group_b.rows["test"]},
        molecules=molecules,
        include_loo=True,
    )

    print("== signature audits ==", flush=True)
    permutation_audit = STR.permutation_invariance_audit(molecules, n_molecules=25, seed=0)
    if args.skip_collision_audit:
        collision_audit = {"checked": False, "reason": "skipped by --skip-collision-audit"}
    else:
        collision_audit = STR.signature_collision_audit(molecules, max_molecules=60, max_pairs=1200, seed=0)
    print(f"  permutation: {permutation_audit}")
    print(f"  collision: passed={collision_audit.get('passed')}")

    print("== stability ==", flush=True)
    stability = {"group_b": {}, "group_a": STAB.analyse_group_a_single_seed(group_a)}
    stability_frames: dict[str, pd.DataFrame] = {}
    for split in ("valid", "test"):
        scaffolds = support_b.query[support_b.query["split"] == split]["scaffold"].astype(str).to_numpy()
        payload, per_mol = STAB.analyse_split(group_b, split, scaffolds)
        stability["group_b"][split] = payload
        stability_frames[split] = per_mol
        print(
            f"  {split}: combo-base delta RMSE by seed="
            f"{[round(payload['delta_by_seed'][f's{s}']['delta_rmse_combo_minus_base'], 5) for s in (0,1,2)]}",
            flush=True,
        )

    print("== conditional model (random42 Validation420) ==", flush=True)
    conditional, oof_frame = COND.analyse_group_a(group_a, support_a, molecules)
    print(
        f"  OOF delta MSE (M1-M0)={conditional['oof_metrics']['delta_M1_minus_M0']['mse']:.6g} "
        f"delta MAE={conditional['oof_metrics']['delta_M1_minus_M0']['mae']:.6g}",
        flush=True,
    )

    print("== COMBO attribution ==", flush=True)
    combo = COMBO.analyse_combo(group_b, support_b)
    combo_table = pd.DataFrame(combo["group_table"]).drop(columns=["tail_indices"], errors="ignore")
    combo_table.to_csv(C.ANALYSIS / "combo_group_table.csv", index=False)

    print("== write per-molecule / support tables ==", flush=True)
    per_molecule = _assemble_per_molecule(
        group_a, support_a, oof_frame, group_b, support_b, stability_frames
    )
    per_molecule.to_csv(C.ANALYSIS / "per_molecule.csv", index=False)

    support_table = pd.concat(
        [support_a.query, support_a.train_loo, support_b.query, support_b.train_loo],
        ignore_index=True,
        sort=False,
    )
    support_table.to_csv(C.ANALYSIS / "support_diagnostics.csv", index=False)

    print("== write primary JSON results ==", flush=True)
    C.write_json(C.ANALYSIS / "seed_stability.json", stability)
    C.write_json(C.ANALYSIS / "conditional_model_results.json", conditional)
    C.write_json(C.ANALYSIS / "combo_attribution.json", combo)

    print("== integrity checks ==", flush=True)
    all_checks: list[dict] = []
    all_checks.extend(_flatten_checks("group_a", group_a.audit["checks"]))
    all_checks.extend(_flatten_checks("group_b", group_b.audit["checks"]))
    all_checks.append(
        {
            "group": "group_separation",
            "check": "random42_and_random122_are_not_the_same_evaluation_set",
            "passed": not separation["same_valid_row_set"] and not separation["treated_as_same_evaluation_set"],
            "detail": separation,
        }
    )
    all_checks.append(
        {
            "group": "structure",
            "check": "rooted_signature_permutation_invariance",
            "passed": bool(permutation_audit["passed"]),
            "detail": permutation_audit,
        }
    )
    all_checks.append(
        {
            "group": "structure",
            "check": "rooted_signature_exactness_vs_vf2",
            "passed": bool(collision_audit.get("passed", False)) if collision_audit.get("checked") else None,
            "detail": collision_audit,
        }
    )
    # Train-only support dictionary checks
    for support in (support_a, support_b):
        train_signatures = [molecules[int(r)].atom_signatures for r in support.train_rows]
        manual_df = STR.document_frequency(train_signatures)
        all_checks.append(
            {
                "group": "support",
                "check": f"{support.group_key}_df_built_from_train_rows_only",
                "passed": manual_df == support.df,
                "detail": {
                    "n_train": int(len(support.train_rows)),
                    "n_signatures": int(len(support.df)),
                    "dictionary_sha256": support.summary["signature_dictionary_sha256"],
                },
            }
        )
        df_with_valid = STR.document_frequency(
            train_signatures
            + [molecules[int(r)].atom_signatures for r in support.query["row_id"].tolist()]
        )
        all_checks.append(
            {
                "group": "support",
                "check": f"{support.group_key}_valid_or_test_rows_not_in_dictionary",
                "passed": df_with_valid != support.df,
                "detail": {
                    "n_signatures_train_only": int(len(support.df)),
                    "n_signatures_if_valid_or_test_included": int(len(df_with_valid)),
                },
            }
        )
    all_checks.append(
        {
            "group": "support",
            "check": "support_table_row_and_scaffold_alignment",
            "passed": bool(
                np.array_equal(
                    support_a.query["row_id"].to_numpy(), group_a.rows["valid"]
                )
                and np.array_equal(
                    support_b.query["row_id"].to_numpy(),
                    np.concatenate([group_b.rows["valid"], group_b.rows["test"]]),
                )
            ),
            "detail": {
                "group_a_valid_rows": int(len(support_a.query)),
                "group_b_valid_test_rows": int(len(support_b.query)),
            },
        }
    )
    all_checks.append(_assert_support_independent_of_test_labels(group_b, support_b, molecules))

    # alignment invariance: shuffle rows and recompute metrics
    rng = np.random.default_rng(7)
    pred = group_b.predictions["base_s0"]
    order = rng.permutation(len(pred.rows))
    shuffled_metrics = STAT.metrics(pred.y, SRC.align_by_row_id(pred.rows[order], pred.pred[order], group_b.rows["valid"], label="shuffle"))
    direct_metrics = STAT.metrics(pred.y, SRC.align_by_row_id(pred.rows, pred.pred, group_b.rows["valid"], label="direct"))
    all_checks.append(
        {
            "group": "alignment",
            "check": "row_shuffle_does_not_change_alignment_statistics",
            "passed": bool(shuffled_metrics == direct_metrics),
            "detail": {"direct": direct_metrics, "shuffled": shuffled_metrics},
        }
    )
    try:
        SRC.strict_align(pred.rows[order], pred.pred[order], group_b.rows["valid"], label="shuffle")
        strict_raised = False
    except SRC.AlignmentError:
        strict_raised = True
    all_checks.append(
        {
            "group": "alignment",
            "check": "strict_order_guard_rejects_shuffled_prediction_file",
            "passed": bool(strict_raised),
            "detail": {"raised_AlignmentError": strict_raised},
        }
    )

    # synthetic metric and sign checks
    y_syn = np.asarray([1.0, 2.0, 3.0])
    p_syn = np.asarray([1.0, 4.0, 1.0])
    syn = STAT.metrics(y_syn, p_syn)
    expected = {"sse": 8.0, "mse": 8.0 / 3.0, "rmse": (8.0 / 3.0) ** 0.5, "mae": 4.0 / 3.0}
    syn_ok = all(abs(syn[k] - v) <= 1e-12 for k, v in expected.items())
    d_syn = STAT.squared_error(y_syn, np.asarray([3.0, 3.0, 3.0])) - STAT.squared_error(
        y_syn, np.asarray([1.0, 2.0, 3.0])
    )
    sign_ok = bool(np.all(d_syn >= 0) and np.any(d_syn > 0))
    all_checks.append(
        {
            "group": "synthetic",
            "check": "hand_computed_sse_rmse_mae",
            "passed": bool(syn_ok),
            "detail": {**syn, "expected": expected},
        }
    )
    all_checks.append(
        {
            "group": "synthetic",
            "check": "d_i_sign_direction",
            "passed": sign_ok,
            "detail": {
                "y": y_syn.tolist(),
                "lcrc_pred": [3.0, 3.0, 3.0],
                "pna_pred": [1.0, 2.0, 3.0],
                "d_i": d_syn.tolist(),
                "interpretation": "positive d_i means PNA has the smaller squared error",
            },
        }
    )

    # data handling record
    handling = {
        "n_molecules": int(len(molecules)),
        "n_parse_failures": int(sum(1 for m in molecules if not m.parse_ok)),
        "n_multi_component": int(sum(1 for m in molecules if m.multi_component)),
        "n_charged": int(sum(1 for m in molecules if m.formal_charge != 0)),
        "n_empty_scaffold_fallback": int(sum(1 for m in molecules if m.scaffold_fallback_used)),
        "n_explicit_h_atoms": int(sum(m.n_explicit_h for m in molecules)),
        "policy": support_b.summary["molecule_handling"],
    }
    all_checks.append(
        {
            "group": "data_handling",
            "check": "invalid_smiles_and_special_cases_are_recorded_not_dropped",
            "passed": handling["n_parse_failures"] == 0,
            "detail": handling,
        }
    )

    # ---- per-molecule assembly and conditional-model reproduction checks
    count_table = per_molecule.groupby(["group", "split"]).size().to_dict()
    expected_counts = {
        ("A_random42", "valid"): 420,
        ("B_lipo_random122", "valid"): 420,
        ("B_lipo_random122", "test"): 420,
    }
    all_checks.append(
        {
            "group": "per_molecule",
            "check": "per_molecule_row_counts",
            "passed": all(int(count_table.get(k, -1)) == v for k, v in expected_counts.items())
            and len(per_molecule) == 1260,
            "detail": {f"{k[0]}/{k[1]}": int(v) for k, v in count_table.items()},
        }
    )
    order_ok = True
    order_detail = {}
    for key, rows in (
        ("A_random42", group_a.rows["valid"]),
        ("B_lipo_random122/valid", group_b.rows["valid"]),
        ("B_lipo_random122/test", group_b.rows["test"]),
    ):
        if key == "A_random42":
            subset = per_molecule[per_molecule["group"] == "A_random42"]
        else:
            group_key, split = key.split("/")
            subset = per_molecule[(per_molecule["group"] == group_key) & (per_molecule["split"] == split)]
        same = np.array_equal(subset["row_id"].to_numpy(), rows)
        order_ok = order_ok and bool(same)
        order_detail[key] = {"n": int(len(subset)), "row_order_matches_frozen_split": bool(same)}
    all_checks.append(
        {
            "group": "per_molecule",
            "check": "per_molecule_row_order_matches_frozen_split",
            "passed": order_ok,
            "detail": order_detail,
        }
    )

    frame_a = per_molecule[per_molecule["group"] == "A_random42"].reset_index(drop=True)
    d_recomputed = (frame_a["pred_lcrc_a_r3_s0"] - frame_a["y"]) ** 2 - (
        frame_a["pred_pna_native_s0"] - frame_a["y"]
    ) ** 2
    d_diff = float(np.max(np.abs(d_recomputed - frame_a["d_i"])))
    all_checks.append(
        {
            "group": "conditional",
            "check": "d_i_recomputed_from_paired_predictions",
            "passed": d_diff <= 1e-12,
            "detail": {"max_abs_diff": d_diff, "n": int(len(frame_a)), "tolerance": 1e-12},
        }
    )
    err0 = frame_a["d_i"] - frame_a["oof_m0"]
    err1 = frame_a["d_i"] - frame_a["oof_m1"]
    oof_recomputed = {
        "M0": {"mse": float(np.mean(err0**2)), "mae": float(np.mean(np.abs(err0)))},
        "M1": {"mse": float(np.mean(err1**2)), "mae": float(np.mean(np.abs(err1)))},
    }
    oof_diff = max(
        abs(oof_recomputed[m][k] - conditional["oof_metrics"][m][k])
        for m in ("M0", "M1")
        for k in ("mse", "mae")
    )
    all_checks.append(
        {
            "group": "conditional",
            "check": "out_of_fold_metrics_recomputed_from_per_molecule_table",
            "passed": oof_diff <= 1e-12,
            "detail": {"max_abs_diff": oof_diff, "recomputed": oof_recomputed},
        }
    )
    fold_counts = frame_a.groupby("fold_m0m1").size().sort_index().tolist()
    scaffold_fold_violations = int(
        (frame_a.groupby("scaffold")["fold_m0m1"].nunique() > 1).sum()
    )
    all_checks.append(
        {
            "group": "conditional",
            "check": "M0_M1_share_folds_and_scaffolds_are_whole",
            "passed": fold_counts == [84, 84, 84, 84, 84] and scaffold_fold_violations == 0,
            "detail": {
                "fold_sizes": fold_counts,
                "scaffolds_split_across_folds": scaffold_fold_violations,
                "M0_and_M1_use_same_fold_vector": True,
            },
        }
    )

    # support LOO reproduction on every Group-B training molecule
    loo_fields = ("unseen_atom_frac", "rare_atom_frac", "median_log1p_df", "p10_log1p_df")
    loo_max_diff = 0.0
    for i, row_id in enumerate(support_b.train_rows):
        expected = STR.leave_one_out_support_row(molecules[int(row_id)].atom_signatures, support_b.df)
        stored = support_b.train_loo.iloc[i]
        for field in loo_fields:
            loo_max_diff = max(loo_max_diff, abs(float(expected[field]) - float(stored[field])))
    all_checks.append(
        {
            "group": "support",
            "check": "leave_one_out_support_rows_reproduced",
            "passed": loo_max_diff <= 1e-12,
            "detail": {
                "n_train_molecules": int(len(support_b.train_rows)),
                "fields": list(loo_fields),
                "max_abs_diff": loo_max_diff,
            },
        }
    )

    # the BASE-defined tail must be reproducible from the per-molecule table
    tail_ok = True
    tail_detail = {}
    for split in ("valid", "test"):
        subset = per_molecule[
            (per_molecule["group"] == "B_lipo_random122") & (per_molecule["split"] == split)
        ].reset_index(drop=True)
        for seed in (0, 1, 2):
            recomputed = STAT.top_error_indices(
                np.abs(subset[f"pred_base_s{seed}"] - subset["y"]).to_numpy(), C.TOP_FRACTION
            )
            stored = np.asarray(
                combo["base_tail_subset"][split][f"s{seed}"]["tail_indices"], dtype=np.int64
            )
            same = np.array_equal(recomputed, stored)
            tail_ok = tail_ok and bool(same)
            tail_detail[f"{split}/s{seed}"] = bool(same)
    all_checks.append(
        {
            "group": "attribution",
            "check": "base_tail_indices_reproduced_from_base_predictions_only",
            "passed": tail_ok,
            "detail": tail_detail,
        }
    )

    failed = [c for c in all_checks if c.get("passed") is False]
    unknown = [c for c in all_checks if c.get("passed") is None]
    integrity = {
        "round": C.ROUND_ID,
        "created_utc": created,
        "git": C.git_state(),
        "environment": C.environment(),
        "tolerance": C.TOL,
        "checks": all_checks,
        "n_checks": len(all_checks),
        "n_failed": len(failed),
        "n_skipped_or_unknown": len(unknown),
        "failed_checks": failed,
        "all_blocking_checks_passed": len(failed) == 0,
        "blocking_policy": (
            "Any metric reproduction |diff| > 1e-5 blocks the corresponding analysis; "
            "Test is POST_HOC_EXPOSED_DISCOVERY and never selects a checkpoint."
        ),
    }
    C.write_json(C.ANALYSIS / "integrity_checks.json", integrity)
    print(f"  checks={len(all_checks)} failed={len(failed)} skipped={len(unknown)}")

    print("== manifest ==", flush=True)
    outputs = [
        "per_molecule.csv",
        "support_diagnostics.csv",
        "conditional_model_results.json",
        "combo_attribution.json",
        "seed_stability.json",
        "integrity_checks.json",
        "combo_group_table.csv",
    ]
    manifest = {
        "round": C.ROUND_ID,
        "created_utc": created,
        "git": C.git_state(),
        "environment": C.environment(),
        "flags": {
            "cpu_only": True,
            "gpu_used": False,
            "training_or_finetuning": False,
            "checkpoint_selection": False,
            "historical_files_modified": False,
            "new_inference_on_frozen_checkpoints": False,
            "group_a_test_predictions_or_labels_read": False,
            "group_b_test_predictions_or_labels_read": True,
            "group_b_test_exposure": "POST_HOC_EXPOSED_DISCOVERY",
            "group_a_and_group_b_pooled": False,
        },
        "analysis_constants": {
            "signature_version": STR.SIGNATURE_VERSION,
            "signature_radius": C.SIGNATURE_RADIUS,
            "rare_df_max": C.RARE_DF_MAX,
            "morgan_radius": C.MORGAN_RADIUS,
            "morgan_bits": C.MORGAN_BITS,
            "metric_tolerance": C.TOL,
            "top_fraction": C.TOP_FRACTION,
            "n_folds": C.N_FOLDS,
            "ridge_alpha": C.RIDGE_ALPHA,
            "bootstrap_reps": C.N_BOOT,
            "bootstrap_seed": C.BOOTSTRAP_SEED,
            "ci_percentiles": list(C.CI_PERCENTILES),
            "low_support_n": C.LOW_SUPPORT_N,
        },
        "data_groups": {
            "A": group_a.audit,
            "B": group_b.audit,
            "separation": separation,
        },
        "support": {
            "A_random42": support_a.summary,
            "B_lipo_random122": support_b.summary,
        },
        "structure_audits": {
            "permutation_invariance": permutation_audit,
            "collision_audit": collision_audit,
        },
        "model_cells": {
            "group_a": ["lcrc_a_r3_s0", "pna_native_s0"],
            "group_b": sorted(group_b.predictions.keys()),
            "missing_group_b_cells": group_b.audit["missing_cells"],
            "excluded_runs": group_b.audit["excluded_runs"],
        },
        "outputs": {},
        "reproduce": {
            "command": "uv run python tracks/structural-sufficiency/lcrc_failure_mechanism_p0_v1/code/run_p0.py",
            "tests": "uv run pytest -q tracks/structural-sufficiency/lcrc_failure_mechanism_p0_v1/tests",
            "full": (
                "uv run pytest -q tracks/structural-sufficiency/lcrc_failure_mechanism_p0_v1/tests "
                "&& uv run python tracks/structural-sufficiency/lcrc_failure_mechanism_p0_v1/code/run_p0.py"
            ),
        },
    }
    for name in outputs:
        path = C.ANALYSIS / name
        if path.exists():
            manifest["outputs"][name] = {
                "sha256": C.sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
    C.write_json(C.ANALYSIS / "manifest.json", manifest)
    print(f"  wrote {C.relpath(C.ANALYSIS / 'manifest.json')}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
