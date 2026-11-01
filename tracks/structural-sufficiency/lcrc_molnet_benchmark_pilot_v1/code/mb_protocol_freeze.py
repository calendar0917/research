"""Pre-registration freeze: binds data, splits, spec, recipe and budget.

This file is written and committed *before* any GPU training.  It records the
hashes of every artifact the round depends on, so that a later reader can prove
that nothing (split, feature spec, recipe, checkpoint rule, budget) moved after
results were seen.

Usage
-----
    uv run python .../code/mb_protocol_freeze.py
"""

from __future__ import annotations

from typing import Any

import mb_common as C
import mb_data as D
import mb_raw_artifact as RAW
import mb_source as S
import mb_train as T


def build() -> dict[str, Any]:
    spec = D.frozen_spec()
    audit = C.read_json(C.ANALYSIS / "p0_audit.json")
    dataset_manifest = C.read_json(C.ANALYSIS / "dataset_manifest.json")
    if audit.get("status") != "PASS":
        raise SystemExit(f"P0 audit status is {audit.get('status')}; refusing to freeze a protocol")
    source_manifest = S.load_source_manifest(verify=True)

    split_manifests = {}
    for key in C.DATASET_ORDER:
        path = C.ANALYSIS / f"split_manifest_{key}.json"
        if not path.exists():
            raise SystemExit(f"missing split manifest for {key}")
        payload = C.read_json(path)
        split_manifests[key] = {
            "path": C.track_rel(path),
            "sha256": C.sha256_file(path),
            "selected_split_file": payload["selected_split_file"],
            "selected_split_sha256": payload["selected_split_sha256"],
            "sizes": payload["sizes"],
            "rows_sha256": payload["rows_sha256"],
            "cross_split_duplicate_groups": len(payload["cross_split_duplicate_groups"]),
        }

    blocked = {}
    for key in C.DATASET_ORDER:
        if audit["datasets"][key]["status"] != "PASS":
            blocked[key] = audit["datasets"][key]["status"]

    return {
        "schema": "lcrc-molnet-protocol-freeze-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "frozen_before_training": True,
        "question": (
            "What does frozen LCRC-A score on four MoleculeNet tasks under published, "
            "traceable splits, and does it transfer from physicochemical regression to "
            "bioactivity classification?"
        ),
        "datasets": {
            key: {
                "task": C.DATASETS[key]["task"],
                "label_column": C.DATASETS[key]["label_column"],
                "units": C.DATASETS[key]["units"],
                "split_method": C.DATASETS[key]["split_method"],
                "primary_metric": C.DATASETS[key]["primary_metric"],
                "metric_direction": C.DATASETS[key]["metric_direction"],
                "expected_split_sizes": list(C.DATASETS[key]["expected_split_sizes"]),
                "status": "READY" if key not in blocked else blocked[key],
            }
            for key in C.DATASET_ORDER
        },
        "blocked_datasets": blocked,
        "data_source": {
            "repository": S.CHEMBENCH_URL,
            "commit": S.CHEMBENCH_COMMIT,
            "commit_date": S.CHEMBENCH_COMMIT_DATE,
            "vendored_manifest": C.track_rel(C.SOURCE_MANIFEST),
            "vendored_manifest_sha256": C.sha256_file(C.SOURCE_MANIFEST),
            "file_hashes": {k: v["sha256"] for k, v in source_manifest["files"].items()},
            "split_repeat_index": C.SPLIT_REPEAT_INDEX,
            "split_repeat_file_index": C.SPLIT_REPEAT_FILE_INDEX,
            "split_repeat_selection_rule": source_manifest["datasets"]["ESOL"]["selection_rule"],
            "alternative_sources_considered": [
                {
                    "name": "MoleculeNet / DeepChem loaders (current default splitters)",
                    "decision": "not used as a split source",
                    "reason": (
                        "chembench is the public source that publishes the actual split indices; "
                        "DeepChem's current loader defaults are not guaranteed to reproduce the "
                        "published protocol, and the task book forbids treating them as the official split"
                    ),
                },
                {
                    "name": "chemprop splits.tar.gz (ChEMBL/BACE indices used by ChemBench for BACE)",
                    "decision": "not used directly",
                    "reason": (
                        "ChemBench already redistributes those BACE scaffold indices at a fixed commit; "
                        "reviewing a second source adds no information for the four selected tasks"
                    ),
                },
            ],
        },
        "raw_artifact": {
            "path": C.track_rel(RAW.ARTIFACT),
            "sha256": C.sha256_file(RAW.ARTIFACT),
            "manifest": C.track_rel(RAW.MANIFEST),
            "manifest_sha256": C.sha256_file(RAW.MANIFEST),
            "schema": RAW.SCHEMA,
            "label_free": True,
            "reason": (
                "RDKit is local-only; the GPU hosts rebuild the identical lipo_v1 caches from this "
                "committed label-free attribute artifact, and P0 proves it equals a fresh RDKit derivation"
            ),
        },
        "split_manifests": split_manifests,
        "feature_spec": {
            "name": spec.name,
            "hash": spec.hash(),
            "source": "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/feature_spec.py::lipo_v1_spec",
            "frozen_json": C.track_rel(C.FROZEN_SPEC_JSON),
            "frozen_json_sha256": C.sha256_file(C.FROZEN_SPEC_JSON),
            "atom_feat_dim": spec.atom_feat_dim,
            "bond_feat_dim": spec.bond_feat_dim,
            "n_scalars": spec.n_scalars,
            "global_dim": spec.global_dim,
            "n_atom_types": spec.n_atom_types,
            "n_bond_types": spec.n_bond_types,
            "n_nbr_codes": spec.n_nbr_codes,
            "n_path_codes": spec.n_path_codes,
        },
        "model": {
            "name": "LCRC-A",
            "source": "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_model.py",
            "parameters": 247_106,
            "config": {
                "width": 64,
                "stat_rank": 32,
                "local_hidden": 192,
                "atom_hidden": 128,
                "atom_out": 96,
                "bond_hidden": 160,
                "bond_out": 96,
                "rho_hidden": 64,
            },
            "structure": "explicit S1 adjacency chemistry statistics + explicit S2 typed 2-hop paths + atom MLP + sum of per-atom contributions + global rho",
            "iterative_hidden_message_passing": False,
            "attention": False,
            "extra_descriptors": False,
            "extra_global_features": False,
            "task_adaptation_allowed": [
                "loss (MSE on standardised target vs BCEWithLogits)",
                "output interpretation (denormalised value vs sigmoid probability)",
                "metric (RMSE/MAE/R2 vs ROC-AUC/PR-AUC)",
                "checkpoint criterion (Validation RMSE vs Validation ROC-AUC)",
            ],
        },
        "training": {
            "recipe_id": "R3/ESOL-BASE",
            "recipe": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in C.RECIPE.items()
            },
            "train_seed": C.TRAIN_SEED,
            "split_repeat_index": C.SPLIT_REPEAT_INDEX,
            "runs_per_dataset": 1,
            "checkpoint_rule": {
                "regression": "lowest Validation RMSE; strict improvement so a tie keeps the earlier epoch",
                "classification": (
                    "highest Validation ROC-AUC; tie -> lower Validation BCE; tie -> earlier epoch"
                ),
            },
            "normalisation": "Train-only structure standardisation and Train-only target mean/std; metrics in raw units",
            "validation_use": "checkpoint selection only; no early stopping; no Test-based decision",
            "test_use": "single one-shot evaluation after the checkpoint manifest is frozen",
        },
        "budget": {
            "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
            "formal_runs": len(C.DATASET_ORDER),
            "smoke": "one 2-epoch engineering smoke per code path, excluded from results",
        },
        "p0": {
            "path": C.track_rel(C.ANALYSIS / "p0_audit.json"),
            "sha256": C.sha256_file(C.ANALYSIS / "p0_audit.json"),
            "status": audit["status"],
        },
        "dataset_manifest": {
            "path": C.track_rel(C.ANALYSIS / "dataset_manifest.json"),
            "sha256": C.sha256_file(C.ANALYSIS / "dataset_manifest.json"),
        },
        "code_hashes": T.code_hashes(),
        "stopping_rules": [
            "no second seed",
            "no split change after seeing any score",
            "no hyperparameter search",
            "no S2BLIND/SCRAMBLE ablation in this round",
            "no G2L/attention/message passing in this round",
            "no additional descriptors or pretraining",
            "no BACE loss or threshold search",
            "no external model training",
            "stop after this four-dataset batch; BBBP/Tox21/HIV require a new decision",
        ],
        "test_labels_read": False,
    }


def main() -> int:
    payload = build()
    C.write_json(C.ANALYSIS / "protocol_freeze.json", payload)
    C.log(f"wrote {C.track_rel(C.ANALYSIS / 'protocol_freeze.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
