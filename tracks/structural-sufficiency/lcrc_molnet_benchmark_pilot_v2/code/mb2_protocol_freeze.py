"""Freeze the round's protocol *before* the first GPU run.

``analysis/protocol_freeze.json`` records, and hashes, everything that must not
change after a score is seen:

* the seven datasets, their sources, split files and row hashes;
* the frozen R3 recipe and the four pre-registered P3 candidates;
* the single frozen XGBoost baseline configuration;
* the frozen feature spec and every reused module;
* the P0 audit and the source manifest;
* the GPU budget and the Test-access discipline.

Run it after ``mb2_p0.py`` and before ``mb2_train.py``.  It refuses to run if the
P0 audit is not ``PASS``.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import mb2_common as C


def build() -> dict[str, Any]:
    p0_path = C.ANALYSIS / "p0_audit.json"
    if not p0_path.exists():
        raise SystemExit(f"{C.track_rel(p0_path)} missing: run mb2_p0.py first")
    p0 = C.read_json(p0_path)
    if p0.get("status") != "PASS":
        raise SystemExit(f"P0 status is {p0.get('status')!r}; the protocol cannot be frozen on a failed audit")
    dataset_manifest_path = C.ANALYSIS / "dataset_manifest.json"
    dataset_manifest = C.read_json(dataset_manifest_path)
    spec = dataset_manifest["feature_spec"]

    return {
        "schema": "lcrc-molnet-v2-protocol-freeze-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit_at_freeze": C.git_commit(),
        "git_dirty_tracked_at_freeze": C.git_dirty(),
        "written_before_any_gpu_training": True,
        "frozen_statement": (
            "Every quantity in this file was fixed before the first formal GPU run of the round. "
            "No split, seed, epoch count, learning rate, loss, checkpoint rule, metric or baseline "
            "configuration may change after a Validation or Test number has been seen."
        ),
        "datasets": {
            key: {
                "source_kind": entry["source_kind"],
                "task": entry["task"],
                "label_columns": entry["label_columns"],
                "units": entry["units"],
                "split_method": entry["split_method"],
                "n_rows": entry["n_rows"],
                "csv_sha256": entry["csv_sha256"],
                "split_file": entry["split_file"],
                "split_sha256": entry["split_sha256"],
                "rows_sha256": entry["rows_sha256"],
                "p0_status": entry["status"],
                "test_status_at_freeze": (
                    "EXPOSED (Sprint 01)" if key in C.EXPOSED_TEST_DATASETS else "NOT_YET_READ"
                ),
            }
            for key, entry in dataset_manifest["datasets"].items()
        },
        "dataset_order": list(C.DATASET_ORDER),
        "split_policy": {
            "chembench_repeat_index": C.SPLIT_REPEAT_INDEX,
            "chembench_repeat_file_index": C.SPLIT_REPEAT_FILE_INDEX,
            "chembench_rule": (
                "repeat 0 = the lexicographically first published split file (suffix 122); glob order is "
                "never used"
            ),
            "ogb_rule": "official split/scaffold/{train,valid,test}.csv.gz row ids, published order preserved",
            "repair_or_resplit": False,
            "drop_or_impute_samples": False,
        },
        "feature_spec": spec,
        "reused_module_hashes": C.reused_module_hashes(),
        "code_hashes": C.code_hashes(),
        "recipe_R3": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in C.RECIPE.items()},
            "note": "identical to Sprint 01 (lcrc-molnet-benchmark-pilot-v1) and to the earlier R3 rounds",
        },
        "task_adaptation": {
            "regression": "MSE on the Train-standardised target; RMSE on raw units; lowest-Validation-RMSE checkpoint",
            "binary": "BCEWithLogitsLoss on a single logit; sigmoid probability; highest-Validation-ROC-AUC checkpoint (tie -> lower BCE -> earlier epoch)",
            "multitask_tox21": (
                "BCEWithLogitsLoss(reduction='none') with the observed-label mask; "
                "loss = (per_label_loss*mask).sum()/mask.sum(); highest-Validation-macro-ROC-AUC checkpoint "
                "(tie -> lower masked BCE -> earlier epoch)"
            ),
            "tox21_head_change": (
                "only the last output layer of f and of rho widens from 1 to 12 outputs; the shared encoder, "
                "S1/S2 modules, local MLP, dropout position and readout form are unchanged"
            ),
            "tox21_parameters": 248_536,
            "single_task_parameters": 247_106,
            "class_weights": False,
            "focal_loss": False,
            "threshold_tuning": False,
        },
        "tox21_endpoints": list(C.TOX21_TASKS),
        "tox21_missing_policy": {
            "missing_value": "NaN (never 0, never imputed, never treated as a negative)",
            "loss": "masked BCE, denominator = number of observed cells",
            "metrics": "per-endpoint ROC-AUC/PR-AUC on observed cells; macro mean over definable endpoints",
            "undefined_endpoint_rule": "reported as not definable; the macro-average coverage count is reported with it",
        },
        "probe_candidates_P3": {
            name: {
                **C.PROBE_CANDIDATES[name],
                "recipe_overrides": dict(C.PROBE_CANDIDATES[name]["overrides"]),
                "validation_only": True,
                "test_inference": False,
            }
            for name in C.PROBE_ORDER
        },
        "probe_rules": {
            "max_extra_runs": 4,
            "epochs_per_candidate": 360,
            "seed": C.TRAIN_SEED,
            "read_test_labels": False,
            "run_test_inference": False,
            "report_test_metric": False,
            "selection_uses_validation_only": True,
            "note": (
                "FreeSolv and BACE Test sets were already read in Sprint 01, so the probes are Validation-only "
                "diagnostics and are never merged into the seven-dataset main table"
            ),
        },
        "baseline_P2": {
            "frozen_configuration": dict(C.BASELINE_CONFIG),
            "datasets": list(C.DATASET_ORDER),
            "same_rows_as_lcrc": True,
            "uses_gpu": False,
            "hyperparameter_search": False,
            "tox21": "one boosting model per endpoint, trained only on that endpoint's labelled Train rows",
        },
        "evaluation_discipline": {
            "lcrc_test": "one shot per new dataset, after analysis/checkpoint_manifest.json is committed",
            "baseline_test": "one shot per dataset, after analysis/baseline_manifest.json is committed",
            "exposed_test_datasets": list(C.EXPOSED_TEST_DATASETS),
            "exposed_test_label": "EXPOSED_TEST_EXPLORATORY",
            "test_used_for_any_selection": False,
            "literature_grades": ["A EXACT_SPLIT_MATCH", "B PROTOCOL_MATCH_ONLY", "C NOT_COMPARABLE"],
        },
        "budget": {
            "gpu": "NVIDIA A100 (res / a100-2)",
            "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
            "planned_gpu_runs": {"P1": 3, "P3": 4, "total_formal": 7, "P2": "CPU only"},
            "stop_conditions": [
                "if the projected HIV run would exceed the remaining cap, report BUDGET_BLOCKED and skip HIV",
                "P3 is dropped first if the budget gets tight; P1 and P2 keep priority",
                "no epoch reduction, batch-size inflation or recipe change may be used to fit the budget",
                "stop after the authorised runs; no extra dataset, seed, architecture or S2 ablation",
            ],
        },
        "inputs": {
            "source_manifest": {
                "path": C.track_rel(C.SOURCE_MANIFEST),
                "sha256": C.sha256_file(C.SOURCE_MANIFEST),
            },
            "raw_artifact_manifest": {
                "path": C.track_rel(C.RAW_ARTIFACT_MANIFEST),
                "sha256": C.sha256_file(C.RAW_ARTIFACT_MANIFEST),
            },
            "p0_audit": {"path": C.track_rel(p0_path), "sha256": C.sha256_file(p0_path)},
            "dataset_manifest": {
                "path": C.track_rel(dataset_manifest_path),
                "sha256": C.sha256_file(dataset_manifest_path),
            },
        },
        "environment_at_freeze": C.public_env_info(),
        "test_labels_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freeze the v2 protocol before GPU training")
    parser.add_argument("--out", default=str(C.ANALYSIS / "protocol_freeze.json"))
    args = parser.parse_args(argv)
    payload = build()
    C.write_json(args.out, payload)
    C.log(
        f"froze protocol -> {C.track_rel(args.out)} "
        f"(P0 PASS, {len(payload['datasets'])} datasets, cap {payload['budget']['gpu_hard_cap_hours']} A100-h)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
