"""Frozen definitions for the LCRC-CPC v1 round.

Round: ``lcrc-pcba-cpc-v1``.  One model, one seed, 40 epochs:

``LCRC-CPC`` = the historical LCRC-W2 seed-0 model plus a **one-shot, rooted,
two-bond path conditioning residual**.  The residual reads the same ``h0`` and
``b`` exactly once, jointly combines the learned root/middle/endpoint states of
every real simple path ``i-j-k``, aggregates them at the root, and writes back
once before the unchanged W2 ``post_local`` readout.

Everything historical is imported read-only:

* ``lcrc_pcba_backbone_scaling_v1/code/bs_common.py`` -- W2 recipe, frozen
  40-epoch order plan, data identities, official evaluator, provenance helpers.
* ``lcrc_pcba_backbone_scaling_v1/code/bs_model.py`` -- ``BSModel`` (= W2 at
  scale 2), the official seed-0 W2 constructor, ``permute_atoms``,
  ``make_optimizer``, ``set_lrs``.
* ``lcrc_pcba_structural_gain_sprint_v1/code/sg_data.py`` -- the frozen task,
  official split, NaN-masked Train/Valid label artifact, official AP.
* ``lcrc_pcba_continuation_recipe_v1/code/cc_data.py`` -- the only gated reader
  of official Test labels, used after the Test prediction freeze.

No official Test target is read by this module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# paths / inherited code
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
S3_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_v1"
BOC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_boc_v1"
RC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_rc_v1"
DYNAMIC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_dynamic_v1"
CC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1"

for _p in (str(BS_ROUND / "code"), str(CC_ROUND / "code"), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bs_common as BC  # noqa: E402
import bs_model as BM  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
PREDICTIONS = ANALYSIS / "test_predictions"

ROUND_ID = "lcrc-pcba-cpc-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-cpc-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-cpc-checkpoint-v1"
SCHEMA_PREFLIGHT = "lcrc-pcba-cpc-preflight-v1"
SCHEMA_CACHE = "lcrc-pcba-cpc-structural-cache-v1"
SCHEMA_TEST_FREEZE = "lcrc-pcba-test-prediction-freeze-v1"  # inherited seal
SCHEMA_TEST_RESULTS = "lcrc-pcba-cpc-test-scores-v1"

MODEL_NAME = "LCRC-CPC"
W2_KEY = "W2"
TEST_ROLE = "EXPOSED_TEST_ENGINEERING_COMPARISON_SET"

# ---------------------------------------------------------------------------
# inherited data / recipe
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
permute_atoms = BM.permute_atoms

SEED = BC.SEED
MODEL_SEED = BC.MODEL_SEED
RECIPE = dict(BC.RECIPE)
scheduled_lr = BC.scheduled_lr
SOURCE = dict(BC.SOURCE)
FROZEN_INPUTS = dict(BC.FROZEN_INPUTS)

FROZEN_PLAN = {
    "global_plan_sha256": BC.A0["global_plan_sha256"],
    "consumed40_prefix_sha256": BC.A0["consumed40_prefix_sha256"],
    "order_seed": 0,
    "batch_size": BATCH_SIZE,
    "steps_per_epoch": STEPS_PER_EPOCH,
}

W2_WIDTHS = BC.config_for_scale(2)
WIDTH = int(W2_WIDTHS["width"])           # 128
BOND_WIDTH = int(W2_WIDTHS["bond_out"])  # 192

#: Historical W2 seed-0 identity (read-only; never retrained here).
W2_SEED0 = {
    "run_json": BS_ROUND / "runs/W2/run.json",
    "best_checkpoint": BS_ROUND / "runs/W2/checkpoint_best.pt",
    "epoch40_checkpoint": BS_ROUND / "runs/W2/checkpoint_epoch40.pt",
    "best_checkpoint_sha256": "d86fe5e56e898d382acf66eb0533beb915f8ca39144c6c1c74e25e147667d2bd",
    "epoch40_checkpoint_sha256": "7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a",
    "parameters": 994_324,
    "best_epoch": 24,
    "best_valid_ap": 0.26527331610028143,
    "valid_ap_at_40": 0.26186691813350904,
    "test_ap_best": 0.2536168632180464,
    "test_ap_at_40": 0.25098584629059617,
    "trunk_init_fingerprint": "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d",
    "epoch0_valid_ap": 0.029063331894071648,
    "gpu_hours": 1.3949014246219305,
}

RC_REAL = {
    "parameters": 1_051_022,
    "best_epoch": 33,
    "best_valid_ap": 0.2677177024413342,
    "valid_ap_at_40": 0.26695028589805453,
    "test_ap_best": 0.2559409168065797,
    "test_ap_at_40": 0.2561671655569764,
    "best_checkpoint_sha256": "5a94dc4b386fa13d0d74e0391e10dd61a97e37040e7c1301125c96cf5b4646b1",
    "gpu_hours": 1.9185818754296957,
}

#: Historical Dynamic reference.  These are the values named in the brief.
DYNAMIC = {
    "parameters": 1_128_468,
    "best_epoch": 25,
    "best_valid_ap": 0.2750671855066329,
    "valid_ap_at_40": 0.26879111577186443,
    "test_ap_best": 0.2622260489867648,
    "test_ap_at_40": 0.2601363152112446,
    "best_checkpoint_sha256": "da80e7f5460355cb68341029291c63b43e74629d195121bb409848bb28445343",
    "epoch40_checkpoint_sha256": "a8f58c5142e1c50cf22f03524b369eac5cad6e3c216597ab9db6943e571fbcd6",
    "gpu_hours": 1.734047997343328,
}

#: The pre-registered recovery endpoints of the brief (6-dp form).
RECOVERY = {
    "valid_w2": 0.265273,
    "valid_dynamic": 0.275067,
    "test_w2": 0.253617,
    "test_dynamic": 0.262226,
    "target_fraction": 0.75,
    "target_valid_ap": 0.27262,
    "target_test_ap": 0.26007,
    "high_support_tasks": 106,
}

#: Frozen BOC-round Test logits for W2 and RC-REAL (no Test labels involved).
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
DYNAMIC_MANIFEST = DYNAMIC_ROUND / "analysis/test_prediction_manifest.json"

# ---------------------------------------------------------------------------
# the one new model
# ---------------------------------------------------------------------------

CPC: dict[str, Any] = {
    "name": MODEL_NAME,
    "base_model": W2_KEY,
    "path": "rooted simple two-bond path (i, j, k), i != j, j != k, i != k",
    "path_scope": "every root, every middle in N(i), every endpoint in N(j) except i",
    "enumeration": "complete; no sampling, truncation, dedup or shortest-path filter",
    "path_fields": ["cpc_ptr", "cpc_middle", "cpc_end", "cpc_bond_ij", "cpc_bond_jk"],
    "root": "implicit, path_atom from cpc_ptr",
    "role_dim": 32,
    "bond_dim": 16,
    "token_hidden": 96,
    "token_out": 64,
    "token_input": 192,
    "gate_input": 96,
    "context_dim": 64,
    "delta_input": 65,
    "delta_output": 128,
    "gate": "independent sigmoid gate per path, never a cross-path softmax",
    "aggregation": "sum_p gate_p * z_p / max(count_root, 1), one root scatter",
    "zero_path": "delta[i] = 0 exactly (the final Linear bias cannot leak)",
    "second_propagation": "none; h_cpc is never read by CPC and post_local runs once",
    "init_rule": (
        "historical bs_model.build_model('W2', seed=0) state loaded bitwise; CPC "
        "constructed inside torch.random.fork_rng and re-initialised from a private "
        "generator seed*1_000_003+1001; the Output Linear(65,128) weight and bias "
        "are zeroed, so epoch-0 logits == historical untrained W2 logits"
    ),
}

#: Exact tensor arithmetic (LayerNorm + biases + zeroed output).
EXPECTED_PARAMETERS: dict[str, int] = {
    "w2_total": 994_324,
    "w2_base": 848_096,
    "w2_readout": 146_228,
    "norm": 256,                 # LayerNorm(128)
    "role_projections": 12_384,  # 3 * (128*32 + 32)
    "bond_projection": 3_088,    # 192*16 + 16
    "path_mlp_1": 18_528,        # 192*96 + 96
    "path_mlp_2": 6_208,         # 96*64 + 64
    "gate": 97,                  # 96*1 + 1
    "output": 8_448,             # 65*128 + 128
    "cpc_total": 49_009,
    "total": 1_043_333,
}
NEW_MODULE_PREFIX = "cpc."

PRIVATE_GENERATOR_OFFSET = 1001

# ---------------------------------------------------------------------------
# budget / diagnostics
# ---------------------------------------------------------------------------

BUDGET = {
    "host": "res",
    "preferred_pool": "res-gpu1",
    "gpu_target_hours": 2.8,
    "gpu_hard_hours": 3.5,
    "cache_target_gib": 1.0,
    "profile_real_batches": 100,
    "profile_warmup_steps": 10,
}

DIAG_VALID_ROWS = 32
DIAG_BATCH_SIZE = 256
GRAD_AUDIT_STEPS = (1, 3, 10, 100)

MECHANISM = {
    "interventions": ["real", "delta_zero", "pairing_shuffle"],
    "shuffle_seed": 0,
    "note": (
        "delta_zero and pairing_shuffle are out-of-distribution interventions. "
        "They measure reliance on the trained CPC path, not a causal generalisation gain."
    ),
}

# ---------------------------------------------------------------------------
# decision rules (pre-registered)
# ---------------------------------------------------------------------------

SUPPORT_MIN = 20
DECISION_RULES = {
    "matches": {
        "recovery_valid_min": 1.0,
        "recovery_test_min": 1.0,
        "improved_high_support_min": 70,
        "high_support_tasks": RECOVERY["high_support_tasks"],
        "label": "CPC_MATCHES_DYNAMIC_RANGE",
    },
    "most": {
        "recovery_valid_min": 0.75,
        "recovery_test_min": 0.75,
        "improved_high_support_min": 70,
        "high_support_tasks": RECOVERY["high_support_tasks"],
        "label": "CPC_RECOVERS_MOST_DYNAMIC_GAIN",
    },
    "partial": {
        "test_and_valid_above_w2": True,
        "improved_high_support_min": 53,
        "high_support_mean_delta_positive": True,
        "high_support_tasks": RECOVERY["high_support_tasks"],
        "label": "CPC_PARTIAL_GAIN",
    },
    "none": {"label": "CPC_NO_CLEAR_GAIN"},
    "invalid": {"label": "CPC_MECHANISM_INVALID"},
}

# ---------------------------------------------------------------------------
# helpers (thin inherited wrappers)
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
    return str(name).startswith(NEW_MODULE_PREFIX)


def state_subset(state: Mapping[str, Any], *, new: bool) -> dict[str, Any]:
    return {k: v for k, v in state.items() if is_new_parameter(k) == bool(new)}


def trunk_subset(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in state.items()
        if not is_new_parameter(k) and not k.startswith("readout.")
    }


def recipe() -> dict[str, Any]:
    out = dict(RECIPE)
    out.update(
        {
            "round": ROUND_ID,
            "model": MODEL_NAME,
            "base_model": W2_KEY,
            "role": "one_shot_rooted_two_bond_path_conditioning",
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_cadence": "every_epoch_official_validation_ap",
            "widths": dict(W2_WIDTHS),
            "cpc": dict(CPC),
            "parameters": int(EXPECTED_PARAMETERS["total"]),
            "parameter_breakdown": dict(EXPECTED_PARAMETERS),
            "lr_schedule": BC.lr_audit(),
            "batch_plan": dict(FROZEN_PLAN),
            "test_policy": (
                "official Test targets are NaN in the training label artifact; "
                "Test logits are sealed before any Test label is opened"
            ),
        }
    )
    return out


def freeze_seal(freeze: Mapping[str, Any]) -> str:
    """Inherited BOC/continuation freeze seal, reused verbatim."""
    payload = {
        "logical_models": freeze.get("logical_models"),
        "predictions": freeze.get("predictions"),
        "test_rows_sha256": freeze.get("test_rows_sha256"),
        "selection_manifest_sha256": freeze.get("selection_manifest_sha256"),
    }
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=BC._json_default).encode()
    )


def test_rows_sha256_check(rows: Any) -> str:
    sha = BC.SC.sha256_int64(np.asarray(rows, dtype=np.int64).tolist())
    if sha != BOC_TEST_ROWS_NPY_SHA256:
        raise RuntimeError(f"official Test row ids {sha} != frozen {BOC_TEST_ROWS_NPY_SHA256}")
    return sha


def verify_frozen_reference_logits() -> dict[str, Any]:
    """Hash-verify W2 / RC-REAL / W2-DYNAMIC frozen Test logits."""
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
            "shape": list(np.load(path, mmap_mode="r").shape),
            "verified": True,
            "source_round": "lcrc-pcba-boc-v1",
        }
    manifest = read_json(DYNAMIC_MANIFEST)
    entry = manifest["predictions"]["W2_DYNAMIC_best_valid"]
    path = REPO / entry["test_logits_path"]
    if not path.is_file():
        raise RuntimeError(f"frozen W2-DYNAMIC logits missing: {path}")
    sha = sha256_file(path)
    if sha != entry["test_logits_sha256"]:
        raise RuntimeError(f"W2-DYNAMIC logits sha {sha} != manifest {entry['test_logits_sha256']}")
    out["W2_DYNAMIC_best_valid"] = {
        "path": track_rel(path),
        "sha256": sha,
        "shape": list(np.load(path, mmap_mode="r").shape),
        "verified": True,
        "source_round": "lcrc-pcba-dynamic-v1",
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "epoch": int(entry["epoch"]),
        "valid_ap": float(entry["valid_ap_replay"]),
    }
    entry40 = manifest["predictions"]["W2_DYNAMIC_epoch40"]
    path40 = REPO / entry40["test_logits_path"]
    if not path40.is_file():
        raise RuntimeError(f"frozen W2-DYNAMIC epoch40 logits missing: {path40}")
    sha40 = sha256_file(path40)
    if sha40 != entry40["test_logits_sha256"]:
        raise RuntimeError(f"W2-DYNAMIC epoch40 logits sha {sha40} != manifest {entry40['test_logits_sha256']}")
    out["W2_DYNAMIC_epoch40"] = {
        "path": track_rel(path40),
        "sha256": sha40,
        "shape": list(np.load(path40, mmap_mode="r").shape),
        "verified": True,
        "source_round": "lcrc-pcba-dynamic-v1",
        "checkpoint_sha256": entry40["checkpoint_sha256"],
        "epoch": int(entry40["epoch"]),
        "valid_ap": float(entry40["valid_ap_replay"]),
    }
    return out


__all__ = [name for name in dir() if not name.startswith("_")]
