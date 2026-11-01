"""Pre-register Sprint 03 before the first GPU job.

Writes ``analysis/protocol_freeze.json``.  Everything that could otherwise be
tuned after seeing a score is fixed here: the two extra Lipophilicity repeats,
the three BBBP inner-CV folds, the two task objectives, the R3 recipe, the
checkpoint rule, the run list, the budget and the Test policy.

This file is written *after* the CPU audits (they are inputs, not results) and
*before* deployment.  It refuses to run if any of its inputs is missing.

Usage
-----
    uv run python code/ta_protocol_freeze.py
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import ta_common as C  # noqa: I001

OUT = C.PROTOCOL_FREEZE

REQUIRED_INPUTS = {
    "p0_information_flow": C.ANALYSIS / "p0_information_flow.json",
    "feature_coverage": C.ANALYSIS / "feature_coverage.json",
    "feature_collision_tests": C.ANALYSIS / "feature_collision_tests.json",
    "split_freeze": C.SPLIT_FREEZE,
    "taskaware_loss_probes_p0": C.ANALYSIS / "taskaware_loss_probes_p0.json",
    "baseline_repeat_manifest": C.ANALYSIS / "baseline_repeat_manifest.json",
}


def build() -> dict[str, Any]:
    missing = [k for k, p in REQUIRED_INPUTS.items() if not p.exists()]
    if missing:
        raise SystemExit(f"cannot freeze: missing inputs {missing}")
    split = C.read_json(C.SPLIT_FREEZE)
    audit = C.read_json(C.ANALYSIS / "p0_information_flow.json")
    coverage = C.read_json(C.ANALYSIS / "feature_coverage.json")
    collisions = C.read_json(C.ANALYSIS / "feature_collision_tests.json")
    loss_probe = C.read_json(C.ANALYSIS / "taskaware_loss_probes_p0.json")
    baseline = C.read_json(C.ANALYSIS / "baseline_repeat_manifest.json")

    payload = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_commit_note": (
            "HEAD when this file was written; the commit that contains it is recorded per run "
            "as protocol_freeze_committed_at in each train.json"
        ),
        "git_dirty_tracked": C.git_dirty(),
        "code_hashes": C.code_hashes(),
        "inputs": {
            name: {"path": C.track_rel(path), "sha256": C.sha256_file(path)}
            for name, path in REQUIRED_INPUTS.items()
        },
        "p0_summary": {
            "information_item_status_counts": audit["status_counts"],
            "fields_with_no_forward_response": audit["sensitivity"]["fields_with_no_response"],
            "joint_code_gates": {
                "s1": audit["joint_codes"]["s1"]["passed"],
                "s2": audit["joint_codes"]["s2"]["passed"],
                "s2_pairing": audit["joint_codes"]["s2_pairing"]["passed"],
            },
            "tetrahedral_collision": collisions["constructive"]["tetrahedral"]["verdict"],
            "frozen_model_same_output": collisions["constructive"]["frozen_model_effect"]["outputs_identical"],
            "ez_changes_only": collisions["constructive"]["ez"]["tensors_that_change"],
            "identical_input_pairs": {
                d: v["n_pairs_with_identical_model_input"] for d, v in collisions["real_data"].items()
            },
            "chiral_molecule_fraction": {
                d: v["rdkit_audit"]["fraction_molecules_with_assigned_chiral_center"]
                for d, v in coverage["datasets"].items()
            },
        },
        "p1_lipophilicity_repeats": {
            "datasets": ["Lipophilicity"],
            "splits": {
                str(suffix): {
                    "file": split["lipophilicity_repeats"][str(suffix)]["file"],
                    "sha256": split["lipophilicity_repeats"][str(suffix)]["sha256"],
                    "rows_sha256": split["lipophilicity_repeats"][str(suffix)]["rows_sha256"],
                    "sizes": split["lipophilicity_repeats"][str(suffix)]["sizes"],
                }
                for suffix in (122, 123, 124)
            },
            "reference_repeat": 122,
            "new_repeats": [123, 124],
            "model": "frozen lipo_v1 LCRC-A, R3 recipe, training seed 0",
            "baseline": "frozen ECFP4+217 RDKit descriptors+XGBoost (Sprint 02 configuration, no tuning)",
            "selection_metric": "Validation RMSE",
            "reported_metric": "Test RMSE, one read per repeat after the checkpoint freeze",
            "exposure_label": "PUBLIC_SPLIT_ROBUSTNESS -- not an independent blind test",
            "gpu_runs": 2,
        },
        "p2_bbbp_inner_cv": {
            "source": "BBBP.scaffold122 Train rows only (1631)",
            "n_folds": C.BBBP_CV_FOLDS,
            "salt": C.BBBP_CV_SALT,
            "folds": {
                k: {
                    "n": v["n"],
                    "n_positive": v["n_positive"],
                    "n_negative": v["n_negative"],
                    "rows_sha256": v["rows_sha256"],
                    "n_scaffold_groups": v["n_scaffold_groups"],
                }
                for k, v in split["bbbp_inner_cv"]["folds"].items()
            },
            "algorithm": split["bbbp_inner_cv"]["algorithm"],
            "acyclic_policy": split["bbbp_inner_cv"]["acyclic_policy"],
            "model": "frozen lipo_v1 LCRC-A, R3 recipe, training seed 0, one run per fold",
            "selection_metric": "ROC-AUC on the held-out fold",
            "published_valid_test_labels": "not read",
            "gpu_runs": 3,
        },
        "p3_objectives": {
            "tox21_equal_task_loss": {
                **{k: v for k, v in loss_probe["tox21_equal_task_loss"].items()},
                "gpu_runs": 1,
                "reference": "lcrc_molnet_benchmark_pilot_v2/analysis/runs/Tox21_s0 (frozen masked-cell run)",
                "selection_metric": "Validation macro ROC-AUC (unchanged checkpoint rule)",
                "test_read": False,
            },
            "hiv_sqrt_pos_weight": {
                **{k: v for k, v in loss_probe["hiv_sqrt_pos_weight"].items()},
                "gpu_runs": 1,
                "reference": "lcrc_molnet_benchmark_pilot_v2/analysis/runs/ogbg-molhiv_s0",
                "selection_metric": "Validation ROC-AUC (unchanged checkpoint rule)",
                "test_read": False,
            },
            "loss_probe_gates": loss_probe["gates"],
            "baseline_repeat_valid_rmse": {
                k: v["valid_rmse"] for k, v in baseline["entries"].items()
            },
        },
        "recipe": {
            **C.recipe(),
            "adam_betas": [float(x) for x in C.recipe()["adam_betas"]],
            "note": "byte-identical to Sprint 02 R3; only the objective (P3) or the row list (P1/P2) changes",
        },
        "checkpoint_rule": {
            "regression": "lowest Validation RMSE, strict improvement, ties keep the earlier epoch",
            "binary": "highest Validation ROC-AUC, tie -> lower Validation BCE, then earlier epoch",
            "multitask": "highest Validation macro ROC-AUC, tie -> lower masked BCE, then earlier epoch",
        },
        "run_plan": list(C.PLANNED_RUNS),
        "max_formal_runs": C.MAX_FORMAL_RUNS,
        "budget": {
            "hard_cap_a100_gpu_hours": C.GPU_HARD_CAP_HOURS,
            "expected_hours": {
                "lipo_r123": 0.10,
                "lipo_r124": 0.10,
                "bbbp_cv_f0": 0.03,
                "bbbp_cv_f1": 0.03,
                "bbbp_cv_f2": 0.03,
                "tox21_equal_task": 0.19,
                "hiv_sqrt_pos": 0.83,
            },
            "expected_total_hours": 1.31,
            "priority_order": ["lipo_r123", "lipo_r124", "bbbp_cv_f0", "bbbp_cv_f1", "bbbp_cv_f2", "tox21_equal_task", "hiv_sqrt_pos"],
            "no_epoch_reduction_to_fit_budget": True,
        },
        "not_done_this_round": [
            "no new FeatureSpec",
            "no augmented LCRC model",
            "no representation change of any kind",
            "no second training seed",
            "no hyper-parameter search",
            "no new dataset",
            "no re-scoring of any Sprint-01/02 Test split",
        ],
        "test_policy": {
            "test_reads_this_round": ["Lipophilicity random123", "Lipophilicity random124"],
            "everything_else": "not read, not predicted, not evaluated",
            "selection": "Validation only (P1/P3) or held-out inner-CV fold (P2)",
            "sprint01_02_test_numbers": "EXPOSED_TEST_EXPLORATORY; carried forward read-only, never re-optimised",
        },
        "engineering_gates": [
            "source split bytes re-verified against the Sprint-02 manifest before training",
            "structure caches rebuilt for the new row lists, SMILES re-checked against the published source",
            "Train-only standardiser fitted once on the frozen Train rows of each run",
            "driver equivalence: a 2-epoch CPU smoke of the new driver matches Sprint 02's driver bit-for-bit on the published Tox21 split",
            "no Test cache is constructed before the checkpoint freeze",
        ],
    }
    C.write_json(OUT, payload)
    C.log(f"wrote {C.track_rel(OUT)}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freeze the Sprint-03 protocol")
    parser.parse_args(argv)
    payload = build()
    print(f"protocol frozen: {payload['protocol_id']} at commit {payload['git_commit'][:8]}")
    print(f"inputs: {list(payload['inputs'])}")
    print(f"planned GPU runs: {[r['key'] for r in payload['run_plan']]}")
    print(f"budget cap: {payload['budget']['hard_cap_a100_gpu_hours']} A100 h, expected {payload['budget']['expected_total_hours']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
