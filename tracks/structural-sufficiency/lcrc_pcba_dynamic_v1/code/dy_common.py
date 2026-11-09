"""Frozen definitions for the LCRC-W2-DYNAMIC v1 diagnostic boundary round.

Round: ``lcrc-pcba-dynamic-v1``

This round trains **one** model: the historical LCRC-W2 (width-128 single-shot
S1/S2 encode) with two additional layers of real chemical-bond message passing
that read the neighbours' learned hidden states.

Frozen, read-only inheritance (never modified, never re-defaulted):

* ``lcrc_pcba_backbone_scaling_v1/code/bs_model.py`` -- ``BSModel`` (= W2 at
  scale 2), ``build_model`` (the official seed-0 W2 constructor),
  ``permute_atoms``, ``post_local`` readout, ``make_optimizer``, ``set_lrs``.
* ``lcrc_pcba_backbone_scaling_v1/code/bs_common.py`` -- W2 widths, the frozen
  from-scratch recipe (AdamW, 1-epoch linear warmup 3e-5 -> 3e-4, per-step
  cosine to 1e-5 over 109,520 updates, dropout 0.10, clip 5.0), data identities,
  the 40-epoch order plan, the official evaluator, provenance helpers.
* ``lcrc_pcba_structural_gain_sprint_v1/code/sg_model.py`` -- ``MessageFFN``,
  the already-validated pure directed-bond message formula, reused at width 128
  / hidden 256, and the private-generator new-module initialisation rule.
* ``lcrc_pcba_boc_v1/analysis/test_prediction_manifest.json`` -- the frozen
  W2 / RC-REAL Test logits (reused, never regenerated).
* ``lcrc_pcba_continuation_recipe_v1/code/cc_data.py`` -- the only path that may
  read official Test labels, and the freeze seal.

No official Test target is read anywhere in this module; the training label
artifact has every Test row set to NaN and the inherited loader refuses it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# paths / inherited code (read-only)
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
SG_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1"
BOC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_boc_v1"
RC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_rc_v1"
CC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1"

BS_CODE = BS_ROUND / "code"
SG_CODE = SG_ROUND / "code"
CC_CODE = CC_ROUND / "code"

for _p in (str(BS_CODE), str(SG_CODE), str(CC_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bs_common as BC  # noqa: E402  (W2 widths, recipe, data, evaluator, provenance)
import bs_model as BM  # noqa: E402  (the frozen W2 model and its constructor)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
PREDICTIONS = ANALYSIS / "test_predictions"

ROUND_ID = "lcrc-pcba-dynamic-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-dynamic-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-dynamic-checkpoint-v1"
SCHEMA_PREFLIGHT = "lcrc-pcba-dynamic-preflight-v1"
SCHEMA_MECHANISM = "lcrc-pcba-dynamic-mechanism-v1"
SCHEMA_TEST_FREEZE = "lcrc-pcba-test-prediction-freeze-v1"  # inherited seal schema
SCHEMA_TEST_RESULTS = "lcrc-pcba-dynamic-test-scores-v1"

TEST_ROLE = "EXPOSED_TEST_ENGINEERING_COMPARISON_SET"

MODEL_NAME = "W2-DYNAMIC"
W2_KEY = "W2"

# ---------------------------------------------------------------------------
# data / recipe (inherited, never re-derived)
# ---------------------------------------------------------------------------

DATASET = BC.DATASET
N_TASKS = BC.N_TASKS
N_MOLECULES = BC.N_MOLECULES
N_TRAIN = BC.N_TRAIN
N_VALID = BC.N_VALID
N_TEST = BC.N_TEST
BATCH_SIZE = BC.BATCH_SIZE
TOTAL_EPOCHS = BC.TOTAL_EPOCHS
STEPS_PER_EPOCH = BC.STEPS_PER_EPOCH
TOTAL_STEPS_40 = BC.TOTAL_STEPS_40
#: Re-exported inherited machinery (single import surface for the round).
MET = BC.MET
PC = BC.PC
SD = BC.SD
NM = BC.NM
TU = BC.TU
config_for_scale = BC.config_for_scale
MODELS = BC.MODELS
set_lrs = BM.set_lrs
make_optimizer = BM.make_optimizer
optimizer_state_cpu = BM.optimizer_state_cpu
state_cpu = BM.state_cpu
restore_rng_state = BM.restore_rng_state

SEED = BC.SEED  # 0
MODEL_SEED = BC.MODEL_SEED  # 0
RECIPE = dict(BC.RECIPE)  # SCRATCH_RECIPE: AdamW / warmup+cosine / dropout 0.10 / clip 5.0
scheduled_lr = BC.scheduled_lr

SOURCE = dict(BC.SOURCE)
FROZEN_INPUTS = dict(BC.FROZEN_INPUTS)

#: The frozen 40-epoch batch-order plan of the historical W2 seed-0 run.
FROZEN_PLAN = {
    "global_plan_sha256": BC.A0["global_plan_sha256"],
    "consumed40_prefix_sha256": BC.A0["consumed40_prefix_sha256"],
    "order_seed": 0,
    "batch_size": BATCH_SIZE,
    "steps_per_epoch": STEPS_PER_EPOCH,
}

#: W2 seed-0 initialisation identity, recorded by the backbone-scaling round.
W2_SEED0 = {
    "run_json": BS_ROUND / "runs/W2/run.json",
    "checkpoint_best": BS_ROUND / "runs/W2/checkpoint_best.pt",
    "checkpoint_epoch40": BS_ROUND / "runs/W2/checkpoint_epoch40.pt",
    "best_checkpoint_sha256": "d86fe5e56e898d382acf66eb0533beb915f8ca39144c6c1c74e25e147667d2bd",
    "epoch40_checkpoint_sha256": "7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a",
    "parameters": 994_324,
    "base_parameters": 848_096,
    "readout_parameters": 146_228,
    "trunk_init_fingerprint": "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d",
    "full_trunk_init_fingerprint": "109756bd24b2ab8589012031e3cf0b0389b0d275227b35044e9381a45c00a5db",
    "rng_start": "681a4f7b74eec5ef2b23f4f964299121bd55a784aa0bf79e0c641c1e0069ec6a",
    "epoch0_valid_ap": 0.029063331894071648,
    "best_epoch": 24,
    "best_valid_ap": 0.26527331610028143,
    "valid_ap_at_40": 0.26186691813350904,
    "test_ap_best": 0.2536164787414065,
    "test_ap_at_40": 0.250985,
    "test_ap_at_40_boc_replay": 0.25098520156897997,
    "gpu_hours": 1.3949014246219305,
}

#: RC-REAL historical reference (frozen; never retrained here).
RC_REAL = {
    "parameters": 1_051_022,
    "best_checkpoint": RC_ROUND / "runs/REAL/checkpoint_best.pt",
    "best_checkpoint_sha256": "5a94dc4b386fa13d0d74e0391e10dd61a97e37040e7c1301125c96cf5b4646b1",
    "best_epoch": 33,
    "best_valid_ap": 0.2677177024413342,
    "valid_ap_at_40": 0.26695028589805453,
    "test_ap_best": 0.25594109880999005,
    "test_ap_at_40": 0.2561664520457869,
    "gpu_hours": 1.9185818754296957,
}

#: Frozen Test predictions from the BOC round (label-free, already sealed).
BOC_MANIFEST = BOC_ROUND / "analysis/test_prediction_manifest.json"
BOC_SELECTION = BOC_ROUND / "analysis/selected_checkpoint.json"
BOC_FROZEN_PREDICTIONS = {
    "W2_best_valid": {
        "path": BOC_ROUND / "runs/freeze/test_predictions/W2_best_valid.npy",
        "sha256": "777397bb2e44aef2e2af11480e24c9efba40e835492285948da5f52a6848e7ff",
    },
    "W2_epoch40": {
        "path": BOC_ROUND / "runs/freeze/test_predictions/W2_epoch40.npy",
        "sha256": "341bcbbf48f7e2f10356b15428dbb4d9ae619fdb6bd82f84bd23fd8e52f20bcf",
    },
    "RC_REAL_best_valid": {
        "path": BOC_ROUND / "runs/freeze/test_predictions/RC_REAL_best_valid.npy",
        "sha256": "9a40636406223c286d78dc7414e603e87df1be95656ae935f02b3d273f829779",
    },
    "RC_REAL_epoch40": {
        "path": BOC_ROUND / "runs/freeze/test_predictions/RC_REAL_epoch40.npy",
        "sha256": "e10400103613f46ed891930bea382fe537f7613d6b8f76c5fcdacc792a710d6b",
    },
}
BOC_TEST_ROWS_NPY_SHA256 = "9171aa54ed4070dcbd6697b68d57560c7015c118168de4fa63b5e05a02c3cd62"

# ---------------------------------------------------------------------------
# the one new model
# ---------------------------------------------------------------------------

DYNAMIC: dict[str, Any] = {
    "name": MODEL_NAME,
    "base_model": W2_KEY,
    "width": 128,          # == W2 width; never changed
    "hidden": 256,         # == 2 * width
    "n_layers": 2,
    "norm": "LayerNorm(width) before the message sum (sg_model.MessageFFN)",
    "activation": "SiLU",
    "message_source": "batch['bond_index'] (each physical bond once) expanded to both directions",
    "message_feature": "bond_proj(concat(bond_feat, bond_feat)) with a single shared Linear(13,128)",
    "aggregation": "index_add_ over dst, neighbours summed, no degree normalisation",
    "residual": "h_next = h + fc2(SiLU(fc1(z + msg)))",
    "graph_context": "none (the historical V context feedback is explicitly excluded)",
    "attention_in_blocks": "none",
    "extra_pooling_or_head": "none",
    "dropout_calls": "exactly one (the inherited W2 drop after local)",
    "local_calls": "exactly one",
    "init_rule": (
        "W2 part from bs_model.build_model('W2', seed=0); new modules from a private "
        "generator seed*1_000_003+17 (Linear U(+-1/sqrt(fan_in)), LayerNorm 1/0); "
        "both blocks' fc2 weight and bias zeroed; the global torch RNG is never advanced"
    ),
    "layer2_input": "h1 (the layer-1 output), never h0",
}

EXPECTED_PARAMETERS: dict[str, int] = {
    "w2_total": 994_324,
    "w2_base": 848_096,
    "w2_readout": 146_228,
    "message_ffn_one": 66_176,   # norm 256 + fc1 33,024 + fc2 32,896
    "message_ffn_two": 132_352,
    "bond_proj": 1_792,          # 13*128 + 128
    "new_total": 134_144,
    "total": 1_128_468,
}

NEW_MODULE_PREFIXES = ("blocks.", "bond_proj.")

# ---------------------------------------------------------------------------
# budget / diagnostics
# ---------------------------------------------------------------------------

BUDGET = {
    "host": "res",
    "preferred_pool": "res-gpu1",
    "gpu_target_hours": 2.5,
    "gpu_hard_hours": 3.0,
    "max_concurrent_training_jobs": 1,
    "profile_real_batches": 100,
    "profile_warmup_steps": 10,
    "forbidden": [
        "20-epoch training",
        "fewer training samples",
        "structural change to fit the budget",
        "a second model in parallel",
    ],
}

#: Fixed small Validation batch for the per-epoch dynamic-usage diagnostics.
DIAG_VALID_ROWS = 32
DIAG_BATCH_SIZE = 512  # official Validation inference batch (inherited history)

#: Gradient-audit steps (step indices are 1-based optimizer steps).
GRAD_AUDIT_STEPS = (1, 3, 10, 100)

#: Mechanism interventions after training (Validation only, no retraining).
MECHANISM = {
    "interventions": ["real", "message_off", "message_shuffle"],
    "shuffle_seed": 0,
    "note": (
        "message_off / message_shuffle create out-of-distribution inputs; they measure "
        "sensitivity to the real message correspondence, not a causal generalisation gain"
    ),
}

# ---------------------------------------------------------------------------
# decision rules (fixed in advance)
# ---------------------------------------------------------------------------

SUPPORT_MIN = 20
DECISION_RULES = {
    "strong": {
        "delta_test_min": 0.010,
        "improved_high_support_min": 70,
        "high_support_tasks": 106,
        "high_support_mean_delta_min": 0.006,
        "validation_direction_positive": True,
        "dynamic_branch_used": True,
        "label": "STRONG_DYNAMIC_SIGNAL",
    },
    "possible": {
        "delta_test_min": 0.005,
        "delta_test_max": 0.010,
        "label": "POSSIBLE_DYNAMIC_SIGNAL",
    },
    "none": {"delta_test_band": (-0.002, 0.002), "label": "NO_CLEAR_DYNAMIC_GAIN"},
    "regression": {"delta_test_max": -0.002, "label": "DYNAMIC_REGRESSION"},
    "invalid": {"label": "MECHANISM_OR_OPTIMIZATION_INVALID"},
    "suppressed": {"r_total_max": 0.01, "label": "DYNAMIC_BRANCH_SUPPRESSED"},
}

# ---------------------------------------------------------------------------
# IO / provenance helpers (inherited wrappers, no re-implementation)
# ---------------------------------------------------------------------------

utcnow = BC.utcnow
write_json = BC.write_json
read_json = BC.read_json
sha256_file = BC.sha256_file
sha256_bytes = BC.sha256_bytes
log = BC.log
environment = BC.environment
seed_everything = BC.seed_everything
capture_rng_state = BC.capture_rng_state
rng_state_hashes = BC.rng_state_hashes
git_commit = BC.git_commit
git_state = BC.git_state
track_rel = BC.track_rel
state_fingerprint = BC.state_fingerprint
gpu_occupancy = BC.gpu_occupancy
host_facts = BC.host_facts


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def is_new_parameter(name: str) -> bool:
    return str(name).startswith(NEW_MODULE_PREFIXES)


def state_subset(state: Mapping[str, Any], *, new: bool) -> dict[str, Any]:
    """Split a W2-DYNAMIC state dict into (W2-only | new-module-only) tensors."""
    return {
        k: v for k, v in state.items() if (is_new_parameter(k) == bool(new))
    }


def trunk_subset(state: Mapping[str, Any]) -> dict[str, Any]:
    """The W2 *trunk*: every W2 tensor except the attention readout (the
    historical ``bs_model._base_state_keys`` fingerprint, plus the new-module
    exclusion)."""
    return {k: v for k, v in state.items() if not is_new_parameter(k) and not k.startswith("readout.")}


def recipe() -> dict[str, Any]:
    """The complete frozen recipe of the single formal run."""
    out = dict(RECIPE)
    out.update(
        {
            "round": ROUND_ID,
            "model": MODEL_NAME,
            "base_model": W2_KEY,
            "role": "diagnostic_boundary_experiment_two_layer_real_bond_mp",
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_cadence": "every_epoch_official_validation_ap",
            "widths": dict(BC.config_for_scale(2)),
            "dynamic": dict(DYNAMIC),
            "parameters": int(EXPECTED_PARAMETERS["total"]),
            "parameter_breakdown": dict(EXPECTED_PARAMETERS),
            "lr_schedule": {
                "steps_per_epoch": STEPS_PER_EPOCH,
                "total_steps_40_epochs": TOTAL_STEPS_40,
                "step0": scheduled_lr(0, phase=3, kind="all"),
                "warmup_last": scheduled_lr(STEPS_PER_EPOCH - 1, phase=3, kind="all"),
                "post_warmup_first": scheduled_lr(STEPS_PER_EPOCH, phase=3, kind="all"),
                "step40e": scheduled_lr(TOTAL_STEPS_40, phase=3, kind="all"),
                "formula": (
                    "steps 0..2737 linear 3e-5->3e-4; then "
                    "1e-5+(3e-4-1e-5)*0.5*(1+cos(pi*(s-2738)/(109520-2738)))"
                ),
                "single_parameter_group": True,
            },
            "batch_plan": dict(FROZEN_PLAN),
            "init_rule": DYNAMIC["init_rule"],
            "test_policy": (
                "official Test targets are NaN in the training label artifact and are "
                "never indexed before the prediction freeze"
            ),
        }
    )
    return out


def freeze_seal(freeze: Mapping[str, Any]) -> str:
    """The inherited BOC/continuation freeze seal, re-used verbatim."""
    payload = {
        "logical_models": freeze.get("logical_models"),
        "predictions": freeze.get("predictions"),
        "test_rows_sha256": freeze.get("test_rows_sha256"),
        "selection_manifest_sha256": freeze.get("selection_manifest_sha256"),
    }
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=BC._json_default).encode()
    )


def verify_frozen_reference_logits() -> dict[str, Any]:
    """Hash-verify the frozen W2 / RC-REAL Test logits (no Test label involved)."""
    out: dict[str, Any] = {}
    for label, rec in BOC_FROZEN_PREDICTIONS.items():
        path = Path(rec["path"])
        if not path.is_file():
            raise RuntimeError(f"frozen reference logits missing: {path}")
        sha = sha256_file(path)
        if sha != rec["sha256"]:
            raise RuntimeError(f"{label} frozen logits sha {sha} != recorded {rec['sha256']}")
        out[label] = {
            "path": track_rel(path),
            "sha256": sha,
            "shape": list(__import__("numpy").load(path, mmap_mode="r").shape),
            "verified": True,
            "source_round": "lcrc-pcba-boc-v1",
        }
    return out


def test_rows_sha256_check(rows: Any) -> str:
    """The inherited ``sha256_int64`` identity of the official Test row ids."""
    sha = BC.SC.sha256_int64(np.asarray(rows, dtype=np.int64).tolist())
    if sha != BOC_TEST_ROWS_NPY_SHA256:
        raise RuntimeError(f"official Test row ids {sha} != frozen {BOC_TEST_ROWS_NPY_SHA256}")
    return sha


__all__ = [
    "ANALYSIS",
    "BATCH_SIZE",
    "BOC_FROZEN_PREDICTIONS",
    "BOC_MANIFEST",
    "BOC_SELECTION",
    "BOC_TEST_ROWS_NPY_SHA256",
    "BUDGET",
    "DATASET",
    "DECISION_RULES",
    "DIAG_BATCH_SIZE",
    "DIAG_VALID_ROWS",
    "DYNAMIC",
    "EXPECTED_PARAMETERS",
    "FROZEN_INPUTS",
    "FROZEN_PLAN",
    "GRAD_AUDIT_STEPS",
    "MECHANISM",
    "MET",
    "MODEL_NAME",
    "MODELS",
    "NM",
    "PC",
    "SD",
    "MODEL_SEED",
    "NEW_MODULE_PREFIXES",
    "N_MOLECULES",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PREDICTIONS",
    "PROTOCOL_ID",
    "RC_REAL",
    "RECIPE",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_MECHANISM",
    "SCHEMA_PREFLIGHT",
    "SCHEMA_RUN",
    "SCHEMA_TEST_FREEZE",
    "SCHEMA_TEST_RESULTS",
    "SEED",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "SUPPORT_MIN",
    "TESTS",
    "TEST_ROLE",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "W2_KEY",
    "W2_SEED0",
    "capture_rng_state",
    "config_for_scale",
    "environment",
    "freeze_seal",
    "git_commit",
    "git_state",
    "gpu_occupancy",
    "host_facts",
    "is_new_parameter",
    "make_optimizer",
    "log",
    "parameter_count",
    "optimizer_state_cpu",
    "read_json",
    "recipe",
    "restore_rng_state",
    "rng_state_hashes",
    "scheduled_lr",
    "seed_everything",
    "set_lrs",
    "sha256_bytes",
    "sha256_file",
    "state_cpu",
    "state_fingerprint",
    "state_subset",
    "trunk_subset",
    "test_rows_sha256_check",
    "track_rel",
    "utcnow",
    "verify_frozen_reference_logits",
    "write_json",
]
