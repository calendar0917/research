"""Build ``input_manifest.json`` for the follow-up round.

Only inputs actually used by the three analyses are recorded.  No Test target,
Test prediction file or Test inference is touched.  Existing run records and
the original frozen round are read-only.
"""

from __future__ import annotations

import argparse
import subprocess
from typing import Any, Sequence

import numpy as np

import followup_common as F

ORIGINAL_CODE_FILES = (
    "dissect_common.py",
    "dissect_tables.py",
    "dissect_model.py",
    "dissect_intervene.py",
    "dissect_data.py",
    "dissect_source.py",
    "dissect_freeze.py",
    "dissect_compact.py",
    "dissect_score.py",
    "dissect_train.py",
)
ORIGINAL_ANALYSIS_FILES = (
    "tables/main_runs.csv",
    "tables/contrasts_valid.json",
    "tables/contrasts_test_exposed.json",
    "tables/cross_repeat.json",
    "bootstrap.json",
    "compact_selection.json",
    "protocol_freeze.json",
    "test_access_log.json",
    "freeze/A_r122.json",
    "freeze/B_r122.json",
    "freeze/C_r122.json",
    "freeze/cross_r123.json",
    "freeze/cross_r124.json",
)
ORIGINAL_TOP_FILES = ("REPORT.md", "MODEL.md", "PROTOCOL.md")
ERROR_RUNS = tuple((v, s) for v in F.ERROR_VARIANTS for s in F.SEEDS)
CHECKPOINT_RUNS = tuple((v, s) for v in F.INTERVENTION_VARIANTS for s in F.SEEDS)


def _source_hashes() -> dict[str, Any]:
    import dissect_source as S

    src = S.load_lipop(122, labels="train_valid")
    split = F.C.CHEMBENCH_LIPOP / F.C.SPLIT_FILES[122]
    return {
        "csv": F.hash_entry(F.C.LIPOP_CSV, "ChemBench Lipophilicity CSV", extra={"sha256_expected": F.C.LIPOP_CSV_SHA256, "sha256_matches_frozen_constant": F.sha256_file(F.C.LIPOP_CSV) == F.C.LIPOP_CSV_SHA256}),
        "split_122": F.hash_entry(split, "public122 split pickle", extra={"sha256_expected": F.C.SPLIT_FILE_SHA256[122], "sha256_matches_frozen_constant": F.sha256_file(split) == F.C.SPLIT_FILE_SHA256[122], "split_rows_sha256": list(F.C.SPLIT_ROWS_SHA256[122]), "sizes": src["sizes"]}),
        "raw_artifact": F.hash_entry(F.C.PILOT / "analysis/source/raw_attributes_v1.json.gz", "label-free RDKit raw-attribute artifact"),
        "raw_artifact_manifest": F.hash_entry(F.C.PILOT / "analysis/source/raw_artifact_manifest.json", "raw-attribute artifact manifest"),
        "structure_cache_train": F.hash_entry(F.C.CACHE / "Lipophilicity_r122_train.pt", "Train structure cache (label-free)"),
        "structure_cache_valid": F.hash_entry(F.C.CACHE / "Lipophilicity_r122_valid.pt", "Valid structure cache (label-free)"),
        # Test cache is deliberately absent from this manifest.
    }


def _run_entries() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    for variant, seed in ERROR_RUNS:
        rec = F.load_train_record(122, variant, seed)
        identity = (int(rec["repeat"]), str(rec["variant"]), int(rec["seed"]))
        if identity in seen:
            raise ValueError(f"duplicate run identity {identity}")
        seen.add(identity)
        run_files = [("train.json", "run training record"), ("valid_predictions.npz", "existing Valid predictions"), ("curve.csv", "existing Valid-RMSE curve")]
        if (variant, seed) in CHECKPOINT_RUNS:
            run_files.append(("checkpoint.pt", "existing best-Valid-RMSE checkpoint"))
        file_entries = []
        for name, role in run_files:
            entry = F.hash_entry(F.run_dir(122, variant, seed) / name, role, extra={"run": F.C.run_name(122, variant, seed), "repeat": 122, "variant": variant, "seed": seed})
            file_entries.append(entry)
            files.append(entry)
        records.append(
            {
                "run": F.C.run_name(122, variant, seed),
                "repeat": 122,
                "variant": variant,
                "seed": seed,
                "status": rec["status"],
                "best_epoch": int(rec["checkpoint"]["best_epoch"]),
                "parameters": int(rec["model"]["parameters"]),
                "target_norm": rec["data"]["target_norm"],
                "train_norm_fingerprint": rec["data"]["train_norm_fingerprint"],
                "train_cache_fingerprint": rec["data"]["train_cache_fingerprint"],
                "valid_cache_fingerprint": rec["data"]["valid_cache_fingerprint"],
                "valid_metrics": rec["metrics"]["valid"],
                "test_labels_read": bool(rec.get("test_labels_read")),
                "git_commit": rec.get("git_commit"),
                "git_dirty_tracked": rec.get("git_dirty_tracked"),
                "device": rec.get("device"),
                "torch": rec.get("torch"),
                "python": rec.get("python"),
                "valid_predictions_hash": next(x["sha256"] for x in file_entries if x["path"].endswith("valid_predictions.npz")),
                "checkpoint_hash": next((x["sha256"] for x in file_entries if x["path"].endswith("checkpoint.pt")), None),
            }
        )
    return records, files


def _protocol_hash_check() -> dict[str, Any]:
    freeze = F.read_json(F.ORIG_ANALYSIS / "protocol_freeze.json")
    expected = freeze.get("code_hashes", {})
    checks = []
    for name in ORIGINAL_CODE_FILES:
        path = F.ORIG_CODE / name
        key = f"code/{name}"
        actual = F.sha256_file(path)
        checks.append(
            {
                "file": key,
                "actual_sha256": actual,
                "protocol_freeze_sha256": expected.get(key),
                "matches": expected.get(key) == actual,
            }
        )
    mismatches = [x for x in checks if not x["matches"]]
    working_tree_clean_for_round = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", F.rel(F.ORIG)],
        cwd=F.REPO,
        check=False,
    ).returncode == 0
    return {
        "source_round_working_tree_matches_commit": bool(working_tree_clean_for_round),
        "all_protocol_freeze_hashes_match_source": not mismatches,
        "checks": checks,
        "mismatches": mismatches,
        "protocol_freeze_git_commit": freeze.get("git_commit"),
        "protocol_freeze_created_utc": freeze.get("created_utc"),
        "post_freeze_difference_explanation": (
            "protocol_freeze.json was written before the final round commit. Two code hashes are therefore "
            "stale relative to the checked-out source at 4a393476: dissect_tables.py changed cross_repeat "
            "to read compact_selection['selected'] instead of ['compact']; dissect_intervene.py gained an "
            "--out-tag option for path_loo_compact.json. Neither change touches _seed_diffs, contrast_table, "
            "the per-run records, or the Valid/Test scalar columns used here. The actual source at the "
            "recorded commit is used and no alternate code version is substituted."
        ),
        "analysis_decision": (
            "Continue the three independent analyses: (1) the main_runs.csv scalars and per-run Valid "
            "predictions used by A/B are unaffected by these two post-freeze edits; (2) the buggy "
            "_seed_diffs code is present in both the frozen hash version and the checked-out source, so the "
            "A correction is not sensitive to the stale hashes; (3) no Test prediction is used."
        ),
    }


def build_manifest() -> dict[str, Any]:
    rows = F.load_main_rows()
    identity = F.check_main_rows(rows)
    protocol_check = _protocol_hash_check()
    run_records, run_files = _run_entries()
    ref = F.valid_reference()
    view = ref["view"]
    calc_norm_fingerprint = F.D.PREP_DATA.norm_fingerprint(view.train_norm)
    target_norm = view.target_norm
    norm_matches = []
    for rec in run_records:
        norm_matches.append(
            {
                "run": rec["run"],
                "recorded_target_norm": rec["target_norm"],
                "reconstructed_target_norm": target_norm,
                "target_norm_matches": rec["target_norm"] == target_norm,
                "recorded_train_norm_fingerprint": rec["train_norm_fingerprint"],
                "reconstructed_train_norm_fingerprint": calc_norm_fingerprint,
                "train_norm_fingerprint_matches": rec["train_norm_fingerprint"] == calc_norm_fingerprint,
            }
        )
    if not all(x["target_norm_matches"] and x["train_norm_fingerprint_matches"] for x in norm_matches):
        raise SystemExit("Train normalization differs from the frozen run records; stop dependent analyses")

    original_files = [
        F.hash_entry(F.ORIG / name, f"original round {name}") for name in ORIGINAL_TOP_FILES
    ]
    original_analysis = [
        F.hash_entry(F.ORIG_ANALYSIS / name, f"original round analysis/{name}") for name in ORIGINAL_ANALYSIS_FILES
    ]
    original_code = [F.hash_entry(F.ORIG_CODE / name, f"original round code/{name}") for name in ORIGINAL_CODE_FILES]
    followup_code = [F.hash_entry(F.CODE / name, f"follow-up code/{name}") for name in (
        "followup_common.py",
        "corrected_tables.py",
        "error_concentration.py",
        "frozen_jc_interventions.py",
        "test_followup.py",
        "build_input_manifest.py",
    )]

    missing = []
    for entry in [*original_files, *original_analysis, *original_code, *run_files, *followup_code]:
        if not entry.get("exists"):
            missing.append(entry["path"])

    return {
        "schema": "lcrc-lipo-followup-input-manifest-v1",
        "created_utc": F.utcnow(),
        "round": "lcrc_lipo_model_dissection_followup_v1",
        "new_training_runs": 0,
        "new_test_inference_or_rescoring": 0,
        "test_inputs_used": [],
        "test_prediction_files_read": 0,
        "test_targets_read": 0,
        "git": {
            "repository": F.rel(F.REPO),
            "head_commit": F.git_commit(),
            "expected_source_round_commit": "4a393476aace617666642d87e0443ea3c3af87e0",
            "head_equals_expected_source_commit": F.git_commit() == "4a393476aace617666642d87e0443ea3c3af87e0",
            "status_short_at_manifest_build": F.git_status_short(),
            "note": "The source round files are read at the checked-out revision above; their hashes are also checked against protocol_freeze.json.",
        },
        "software_environment": F.env_info(),
        "main_runs_table": {
            "path": F.rel(F.MAIN_RUNS),
            "sha256": F.sha256_file(F.MAIN_RUNS),
            "identity_check": identity,
            "columns_used": ["repeat", "variant", "seed", "valid_rmse", "valid_mae", "test_rmse", "test_mae", "best_epoch", "params_trainable"],
            "test_use": "Already-reported Test RMSE/MAE scalar columns only; no Test prediction file, target or inference.",
        },
        "original_round_source_commit_check": protocol_check,
        "original_round_files": {"top_level": original_files, "analysis": original_analysis, "code": original_code},
        "source_data": _source_hashes(),
        "valid_reference": {
            "repeat": 122,
            "rows": int(ref["rows"].size),
            "row_id_sha256": F.sha256_array(ref["rows"]),
            "y_sha256": F.sha256_array(ref["y"]),
            "rows_equal_published_split": True,
            "train_normalisation": {
                "target_norm": target_norm,
                "train_norm_fingerprint": calc_norm_fingerprint,
                "source": "Train structures/labels only; reconstructed values match every run record",
                "valid_or_test_used": False,
                "matches": norm_matches,
            },
        },
        "runs": run_records,
        "run_files": run_files,
        "followup_code_files": followup_code,
        "missing_inputs": missing,
        "missing_input_policy": "No missing input is filled by retraining, borrowing another repeat, or silently using another version.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    manifest = build_manifest()
    path = F.write_json(F.FOLLOWUP / "input_manifest.json", manifest)
    print(f"wrote {F.rel(path)}; missing={len(manifest['missing_inputs'])}")
    print(f"source tree matches commit: {manifest['original_round_source_commit_check']['source_round_working_tree_matches_commit']}")
    print(f"all protocol_freeze code hashes match: {manifest['original_round_source_commit_check']['all_protocol_freeze_hashes_match_source']}")
    print(f"run identities: {manifest['main_runs_table']['identity_check']['n_unique_identities']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
