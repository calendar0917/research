"""Pre-registration freeze: hash every scientific artefact before training.

The freeze is written *before* the first formal training unit and never edited
afterwards.  It contains

* the full scientific payload (methods, recipes, seeds, thresholds, stage order,
  budget, engineering fallbacks) so the design is readable without the code;
* the SHA-256 of every file that can change a number (``code/*.py``,
  ``configs/frozen.yaml``, ``tests/*.py``, ``PROTOCOL.md``);
* the parameter counts and the three-bond-path cache digests.

``bc_freeze.verify()`` re-checks the hashes and is run by every training unit, so
a silently edited model or recipe cannot enter a result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import bc_common as C

#: Explicit file list: every file that can change a *training* number.  Analysis
#: and reporting modules are deliberately not listed, so adding them later cannot
#: disturb the freeze; changing any file below does.
FROZEN_FILES = (
    "PROTOCOL.md",
    "configs/frozen.yaml",
    "tests/test_bc_round.py",
    "code/bc_common.py",
    "code/bc_path3.py",
    "code/bc_data.py",
    "code/bc_models.py",
    "code/bc_train.py",
    "code/bc_checks.py",
    "code/bc_freeze.py",
    "code/bc_prepare.py",
    "code/bc_run.py",
)
FROZEN_DIRS: tuple[str, ...] = ()
FREEZE_PATH = C.ANALYSIS_DIR / "protocol_freeze.json"


def _file_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in FROZEN_FILES:
        path = C.ROUND_DIR / name
        if path.is_file():
            out[name] = C.sha256_file(path)
    for directory in FROZEN_DIRS:
        for path in sorted((C.ROUND_DIR / directory).glob("*.py")):
            out[f"{directory}/{path.name}"] = C.sha256_file(path)
    return out


def payload() -> dict[str, Any]:
    return {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "dataset": C.DATASET,
        "splits": list(C.SPLITS),
        "expected_split_sizes": C.EXPECTED_SPLIT_SIZES,
        "tasks": {task: {"target": C.TASK_TARGET_NAME[task], "teacher": C.TASK_TEACHER[task]} for task in C.TASKS},
        "variants": {variant: C.VARIANT_NOTE[variant] for variant in C.VARIANTS},
        "institutions": list(C.INSTITUTIONS),
        "branch": {
            "u": f"SiLU(Linear({C.S2_EVIDENCE_DIM},{C.PAIR_HIDDEN}))",
            "v": f"SiLU(Linear({C.BOND_EVIDENCE_DIM},{C.BOND_HIDDEN}))",
            "pair_input_dim": C.PAIR_INPUT_DIM,
            "psi": f"Linear({C.PSI_HIDDEN},{C.PSI_OUT}) . SiLU . Linear({C.PAIR_INPUT_DIM},{C.PSI_HIDDEN})",
            "readout": "Linear(32,1,bias=False), weight initialised to exactly zero",
            "normaliser": "1/sqrt(max(1,n_paths3(G)))",
            "nonlinearity_placement": "before the per-path aggregation",
            "reversal": "z_P = 0.5*(psi(x(p,q,e)) + psi(x(p_rev,q_rev,e)))",
            "branch_dropout": C.BRANCH_DROPOUT,
            "dropout_stream": "isolated torch.Generator seeded per epoch; the C1 stream is untouched",
            "params_expected": 17392,
            "max_fraction_of_c1": C.BRANCH_MAX_FRACTION,
        },
        "c1": {
            "parameters": C.C1_PARAMETERS,
            "trunk_parameters": C.C1_TRUNK_PARAMETERS,
            "readout_parameters": C.C1_READOUT_PARAMETERS,
            "preserved_modules": [
                "atom encoder", "bond encoder", "S1 neighbour embedding", "typed S2 type/end/joint/closure/content",
                "LocalMLP", "LayerNorm", "dropout", "sum_v f(h_v)", "rho([g, mean h, mean h^2, log1p n])",
            ],
        },
        "recipes": {"FROZEN": C.RECIPE_FROZEN, "E2E": C.RECIPE_E2E},
        "seeds": {
            "branch_init": C.BRANCH_INIT_SEED,
            "branch_dropout": C.BRANCH_DROPOUT_SEED,
            "train_default": C.TRAIN_SEED_DEFAULT,
            "e2e_repeats": [0, 1],
            "bootstrap": C.BOOTSTRAP_SEED,
            "oracle_control": C.ORACLE_CONTROL_SEED,
        },
        "statistics": {
            "bootstrap_draws": C.BOOTSTRAP_DRAWS,
            "paired": "same molecules, same official row ids",
            "test_policy": "each frozen model reads DEVELOPMENT Test exactly once, after selection",
        },
        "decision_scales": {
            "stage_c_joint_vs_marginal": C.STAGE_C_JOINT_VS_MARGINAL,
            "stage_c_joint_vs_c1": C.STAGE_C_JOINT_VS_C1,
            "stage_c_bootstrap_prob": C.STAGE_C_BOOTSTRAP_PROB,
            "practical_joint_vs_c1": C.PRACTICAL_JOINT_VS_C1,
            "practical_joint_vs_marginal": C.PRACTICAL_JOINT_VS_MARGINAL,
            "min_pairing_support": C.MIN_PAIRING_SUPPORT,
            "min_oracle_eligible_fraction": C.MIN_ORACLE_ELIGIBLE_FRACTION,
        },
        "stage_order": [
            "stage0_preflight_and_freeze",
            "stageA_frozen_branch_lipo_and_wc_logp",
            "stageB_end_to_end_two_seeds",
            "stageB_trigger_rule_validation_only",
            "stageC_only_if_triggered",
            "stageD_read_only_integration",
        ],
        "stageC_selection_rule": [
            "if E2E triggers, confirm E2E",
            "else if FROZEN triggers, confirm FROZEN",
            "both trigger -> E2E",
        ],
        "budget": {
            "wall_clock_soft_hours": 7.5,
            "wall_clock_hard_hours": 8.0,
            "gpu_hours_soft": 14.0,
            "gpu_hours_hard": 16.0,
            "gpus": ["res-gpu0", "res-gpu1"],
            "one_gpu_process_per_card": True,
        },
        "engineering_fallbacks": [
            "exact event chunking when a batch is large; never sampling or truncation",
            "WC_TEACHER_UNAVAILABLE if the WC_LOGP C1 checkpoint fails identity verification",
            "a failed unit is re-run as a whole under a new id; results are never spliced",
        ],
    }


def write(*, pairing_support: dict[str, Any] | None = None) -> dict[str, Any]:
    hashes = _file_hashes()
    body = payload()
    freeze = {
        "round_id": C.ROUND_ID,
        "protocol_hash": C.sha256_json(body),
        "body": body,
        "file_hashes": hashes,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": C.git_commit(),
        "git_branch": C.git_branch(),
        "git_dirty_tracked": C.git_dirty_tracked(),
        "uv_lock_sha256": C.lockfile_sha256(),
        "notes": [
            "this file is written before the first formal training unit and is never edited afterwards",
            "any later change is an engineering fix, recorded in analysis/compute_ledger.json with the affected units",
        ],
    }
    if pairing_support is not None:
        freeze["pairing_support"] = {
            "status": pairing_support.get("status"),
            "min_valid": pairing_support.get("min_valid_support"),
            "by_split": pairing_support.get("pairing_support"),
            "paths3_sha256": pairing_support.get("paths3_sha256"),
        }
    C.ensure_dir(C.ANALYSIS_DIR)
    C.atomic_write_json(FREEZE_PATH, freeze)
    return freeze


def verify(*, strict: bool = True) -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        if strict:
            raise RuntimeError(f"no protocol freeze at {C.relpath(FREEZE_PATH)}; refusing to train")
        return {"present": False}
    freeze = C.read_json(FREEZE_PATH)
    expected = freeze.get("file_hashes", {})
    current = _file_hashes()
    changed = sorted(name for name, digest in expected.items() if current.get(name) != digest)
    added = sorted(name for name in current if name not in expected)
    body = payload()
    body_hash = C.sha256_json(body)
    return {
        "present": True,
        "protocol_hash_expected": freeze.get("protocol_hash"),
        "protocol_hash_current": body_hash,
        "protocol_hash_match": bool(body_hash == freeze.get("protocol_hash")),
        "changed_files": changed,
        "added_files": added,
        "clean": bool(not changed and not added and body_hash == freeze.get("protocol_hash")),
        "frozen_at": freeze.get("frozen_at"),
    }


def main() -> None:
    support = C.read_json(C.ANALYSIS_DIR / "path3_support.json") if (C.ANALYSIS_DIR / "path3_support.json").is_file() else None
    freeze = write(pairing_support=support)
    print(json.dumps({k: v for k, v in freeze.items() if k != "body"}, indent=2, ensure_ascii=False)[:3000])
    print("files hashed:", len(freeze["file_hashes"]))
    print("protocol_hash:", freeze["protocol_hash"])


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
