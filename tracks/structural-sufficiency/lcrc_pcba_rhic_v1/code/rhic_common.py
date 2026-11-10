"""Frozen definitions for the LCRC-RHIC v1 round.

Round: ``lcrc-pcba-rhic-v1``.  One model, one seed, 40 epochs:

``LCRC-RHIC`` = the historical LCRC-W2 seed-0 model plus a **one-shot,
root-conditioned, two-level hierarchical path composition** residual.

Before the root summary, paths are grouped by their first-hop branch
``(root, middle)`` -- a directed physical bond.  Inside a branch the model keeps
the mean token and the **pairwise co-occurrence mean of distinct path tokens**;
across the branches of one root it keeps the mean branch value and the pairwise
co-occurrence mean of distinct branch values.  The result is written back to the
root atom once, before the unchanged W2 ``post_local`` readout.

Everything historical is imported read-only and never retrained here:

* ``lcrc_pcba_backbone_scaling_v1/code/bs_common.py`` (via ``cpc_common``) --
  W2 recipe, frozen 40-epoch order plan, data identities, official evaluator,
  provenance helpers, W2 / RC-REAL / W2-DYNAMIC reference numbers.
* ``lcrc_pcba_backbone_scaling_v1/code/bs_model.py`` -- ``BSModel`` (W2 at
  scale 2) and its official seed-0 constructor.
* ``lcrc_pcba_cpc_v1/code/cpc_paths.py`` -- the audited rooted two-bond path
  cache and the vectorised collate.  This round does not rebuild it.
* ``lcrc_pcba_cpc_v1/code/cpc_model.py`` -- the audited CPC path-token
  encoder (reused verbatim as ``rhic.token``) and state helpers.
* ``lcrc_pcba_continuation_recipe_v1/code/cc_data.py`` -- the only gated
  reader of official Test labels, used after the Test prediction freeze.

No official Test target is read by this module.
"""

from __future__ import annotations

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

CPC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_cpc_v1"
CPC_CODE = CPC_ROUND / "code"

for _p in (str(CPC_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cpc_common as IC  # noqa: E402  (W2/CPC/Dynamic identities + inherited helpers)
import cpc_model as CM  # noqa: E402  (audited CPC token encoder, state helpers)
import cpc_paths as P  # noqa: E402  (audited path cache + collate)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
PREDICTIONS = ANALYSIS / "test_predictions"

ROUND_ID = "lcrc-pcba-rhic-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-rhic-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-rhic-checkpoint-v1"
SCHEMA_PREFLIGHT = "lcrc-pcba-rhic-preflight-v1"
SCHEMA_STRUCTURE = "lcrc-pcba-rhic-structure-v1"
SCHEMA_SELECTION = "lcrc-pcba-rhic-selection-v1"
SCHEMA_TEST_RESULTS = "lcrc-pcba-rhic-test-scores-v1"
SCHEMA_MECHANISM = "lcrc-pcba-rhic-mechanism-v1"
SCHEMA_DECISION = "lcrc-pcba-rhic-decision-v1"
#: inherited BOC/CC freeze schema; the ``cc_data`` Test-label gate requires it
SCHEMA_TEST_FREEZE = IC.SCHEMA_TEST_FREEZE

MODEL_NAME = "LCRC-RHIC"
BASE_MODEL = "W2"
TEST_ROLE = IC.TEST_ROLE

# ---------------------------------------------------------------------------
# inherited data / recipe
# ---------------------------------------------------------------------------

DATASET = IC.DATASET
N_TASKS = IC.N_TASKS
N_MOLECULES = IC.N_MOLECULES
N_TRAIN = IC.N_TRAIN
N_VALID = IC.N_VALID
N_TEST = IC.N_TEST
BATCH_SIZE = IC.BATCH_SIZE
TOTAL_EPOCHS = IC.TOTAL_EPOCHS
STEPS_PER_EPOCH = IC.STEPS_PER_EPOCH
TOTAL_STEPS_40 = IC.TOTAL_STEPS_40

MET = IC.MET
PC = IC.PC
SD = IC.SD
NM = IC.NM
TU = IC.TU
config_for_scale = IC.config_for_scale
MODELS = IC.MODELS
set_lrs = IC.set_lrs
make_optimizer = IC.make_optimizer
optimizer_state_cpu = IC.optimizer_state_cpu
state_cpu = IC.state_cpu
restore_rng_state = IC.restore_rng_state
permute_atoms = IC.permute_atoms

SEED = IC.SEED
MODEL_SEED = IC.MODEL_SEED
RECIPE = dict(IC.RECIPE)
scheduled_lr = IC.scheduled_lr
SOURCE = dict(IC.SOURCE)
FROZEN_INPUTS = dict(IC.FROZEN_INPUTS)

FROZEN_PLAN = dict(IC.FROZEN_PLAN)

W2_WIDTHS = IC.config_for_scale(2)
WIDTH = int(W2_WIDTHS["width"])           # 128
BOND_WIDTH = int(W2_WIDTHS["bond_out"])   # 192

#: Historical references (read-only; never retrained here).
W2_SEED0 = dict(IC.W2_SEED0)
RC_REAL = dict(IC.RC_REAL)
DYNAMIC = dict(IC.DYNAMIC)
CPC = {
    "parameters": 1_043_333,
    "best_epoch": 36,
    "best_valid_ap": 0.2659443284254145,
    "valid_ap_at_40": 0.2639930249927749,
    "test_ap_best": 0.26235601072953935,
    "test_ap_at_40": 0.26149230849854166,
    "best_checkpoint_sha256": "7957f2aa4bc524950960f4560cce0c3cacc46090aaf770d85305d018d71cf27a",
    "epoch40_checkpoint_sha256": "031d2fa636da490bb47ab66a1933ce7b0cab3898dec55bf22c636832940ac453",
    "train_ap_40": 0.5929580206432022,
    "train_bce_40": 0.02455431676606988,
    "valid_bce_40": 0.04668954378071157,
    "gpu_hours": 1.7620175250588606,
    "source_round": "lcrc-pcba-cpc-v1",
    "test_scores": CPC_ROUND / "analysis/test_scores.json",
}

#: train/valid BCE + train AP at the frozen epoch-40 endpoint of every reference
REFERENCE_ENDPOINTS = {
    "W2": {
        "train_ap_40": 0.5818745570283522,
        "train_bce_40": 0.025215364422066445,
        "valid_bce_40": 0.04653792303564479,
    },
    "RC-REAL": {
        "train_ap_40": 0.6042811271102507,
        "train_bce_40": 0.0238952797628516,
        "valid_bce_40": 0.04716892753801324,
    },
    "LCRC-CPC": {
        "train_ap_40": CPC["train_ap_40"],
        "train_bce_40": CPC["train_bce_40"],
        "valid_bce_40": CPC["valid_bce_40"],
    },
    "W2-DYNAMIC": {
        "train_ap_40": 0.6542595185290578,
        "train_bce_40": 0.022108199368544947,
        "valid_bce_40": 0.048016634077238544,
    },
}

BOC_FROZEN_PREDICTIONS = dict(IC.BOC_FROZEN_PREDICTIONS)
BOC_TEST_ROWS_NPY_SHA256 = IC.BOC_TEST_ROWS_NPY_SHA256
DYNAMIC_MANIFEST = IC.DYNAMIC_MANIFEST

# ---------------------------------------------------------------------------
# the one new model (frozen geometry)
# ---------------------------------------------------------------------------

RHIC: dict[str, Any] = {
    "name": MODEL_NAME,
    "base_model": BASE_MODEL,
    "path_source": "lcrc_pcba_cpc_v1 rooted simple two-bond path (i, j, k)",
    "path_scope": "every root, every middle in N(i), every endpoint in N(j) except i",
    "path_cache": "reused verbatim: cpc_path_atom/cpc_middle/cpc_end/cpc_bond_ij/cpc_bond_jk",
    "branch_definition": "directed physical bond (root, middle); (i,j) != (j,i)",
    "branch_id": "2 * physical_bond_id + dir, dir=0 for storage u->v, dir=1 for v->u",
    "branch_id_runtime": "derived from cpc_bond_ij + root/middle vs bond_index; no new cache",
    "token": "CPC v1 path token, reused verbatim (LayerNorm + 3x Linear(128,32) + Linear(192,16))",
    "token_input": 192,
    "token_hidden": 96,
    "token_out": 64,
    "gate_input": 96,
    "branch_input": 209,
    "branch_hidden": 96,
    "branch_out": 64,
    "branch_input_parts": ["r[root] 32", "m[middle] 32", "e[bond_ij] 16", "mu_b 64", "q_b 64", "log1p(n_b) 1"],
    "root_input": 162,
    "root_hidden": 192,
    "root_out": 128,
    "root_input_parts": ["r[root] 32", "mean_branch 64", "cross_branch 64", "log1p(path_count) 1", "log1p(d_root) 1"],
    "first_level": "sum_p v_p and sum_p v_p*v_p per branch; mu_b = S_b/n_b; q_b = pairwise mean of distinct tokens",
    "second_level": "sum_b branch_value and sum_b branch_value^2 per root; cross_branch = pairwise mean of distinct branches",
    "empty_group": "n_b = 0 branches are never encoded and never counted at the root; d_root = 0 -> delta = 0",
    "second_propagation": "none; h = h0 + delta is never read by RHIC and post_local runs once",
    "init_rule": (
        "historical bs_model.build_model('W2', seed=0) state loaded bitwise; RHIC constructed "
        "inside torch.random.fork_rng and initialised from a private generator "
        "seed*1_000_003+2002; the final Linear(192,128) weight and bias are zeroed, so "
        "epoch-0 logits == historical untrained W2 logits"
    ),
}

#: Exact tensor arithmetic (LayerNorm + biases + zeroed final Linear).
EXPECTED_PARAMETERS: dict[str, int] = {
    "w2_total": 994_324,
    "w2_base": 848_096,
    "w2_readout": 146_228,
    "token_norm": 256,            # LayerNorm(128)
    "token_roles": 12_384,        # 3 * (128*32 + 32)
    "token_bond": 3_088,          # 192*16 + 16
    "token_path1": 18_528,        # 192*96 + 96
    "token_path2": 6_208,         # 96*64 + 64
    "token_gate": 97,             # 96*1 + 1
    "token_total": 40_561,
    "branch1": 20_160,            # 209*96 + 96
    "branch2": 6_208,             # 96*64 + 64
    "branch_total": 26_368,
    "root1": 31_296,              # 162*192 + 192
    "root2": 24_704,              # 192*128 + 128
    "root_total": 56_000,
    "rhic_total": 122_929,
    "total": 1_117_253,
}
NEW_MODULE_PREFIX = "rhic."
TOKEN_PREFIX = "rhic.token."
TOKEN_LAYER_PREFIXES = (
    "rhic.norm.",
    "rhic.root_proj.",
    "rhic.middle_proj.",
    "rhic.end_proj.",
    "rhic.bond_proj.",
    "rhic.path1.",
    "rhic.path2.",
    "rhic.gate.",
)
BRANCH_PREFIX = ("rhic.branch1.", "rhic.branch2.")
ROOT_PREFIX = ("rhic.root1.", "rhic.root2.")
PRIVATE_GENERATOR_OFFSET = 2002

#: frozen structural expectations (from the CPC round, read-only)
CPC_CACHE = {
    "n_rows": 437_929,
    "n_paths": 34_369_144,
    "cache_sha256": "caad3e90dd76d1f1483d2a01501eba23413072a7b5a2fc4b62780c315aa59035",
    "meta_sha256": "ef3246c1b3dbda3618549fd69cac9e2452bdd54879e108385509da2e3bcae503",
    "enumeration_schema_version": "rooted-two-bond-simple-path-v1",
    "rows_sha256": "85b8d38b10293ddd5f431c6feae3aa9c22ec56f19a3a2367c4da2c170d9c1642",
}
CPC_STRUCTURE_ARTIFACT = CPC_ROUND / "analysis/structure_cache.json"

# ---------------------------------------------------------------------------
# budget / diagnostics
# ---------------------------------------------------------------------------

BUDGET = {
    "host": "res",
    "preferred_pool": "res-gpu1",
    "gpu_target_hours": 3.0,
    "gpu_hard_hours": 4.0,
    "profile_real_batches": 100,
    "profile_warmup_steps": 10,
}

DIAG_VALID_ROWS = 32
DIAG_BATCH_SIZE = 256
GRAD_AUDIT_STEPS = (1, 3, 10, 100)

MECHANISM = {
    "interventions": ["real", "delta_zero", "branch_assignment_shuffle"],
    "shuffle_seed": 0,
    "note": (
        "delta_zero and the within-root branch-assignment shuffle are out-of-distribution "
        "interventions on a jointly trained network.  They measure reliance on the trained "
        "RHIC branch structure, not an independent causal generalisation gain.  The shuffle "
        "keeps every path token, root membership and branch count fixed."
    ),
}

# ---------------------------------------------------------------------------
# decision rules (pre-registered)
# ---------------------------------------------------------------------------

SUPPORT_MIN = 20
#: Fixed 106-task high-support set and its 22 low-support complement, frozen by
#: the CPC round from **official** Valid and Test positives >= 20 each.  Task 45
#: is unscorable on Test.  This round never recomputes the set from Test labels
#: before the freeze; it re-reads the published list and verifies its identity.
CPC_DECISION_ARTIFACT = CPC_ROUND / "analysis/decision.json"
HIGH_SUPPORT_TASKS = (
    0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 29, 30, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47,
    48, 49, 50, 51, 52, 53, 56, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 70, 71,
    73, 74, 75, 76, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
    94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110,
    111, 112, 113, 114, 115, 116, 119, 121, 122, 124, 126, 127,
)
LOW_SUPPORT_TASKS = (
    2, 9, 27, 28, 31, 34, 44, 45, 54, 55, 57, 62, 69, 72, 77, 80, 108, 117, 118,
    120, 123, 125,
)
if len(HIGH_SUPPORT_TASKS) != 106 or len(LOW_SUPPORT_TASKS) != 22:
    raise RuntimeError("high/low-support task counts must be 106 / 22")
if set(HIGH_SUPPORT_TASKS) & set(LOW_SUPPORT_TASKS):
    raise RuntimeError("high- and low-support task sets overlap")
if set(HIGH_SUPPORT_TASKS) | set(LOW_SUPPORT_TASKS) != set(range(128)):
    raise RuntimeError("high+low support does not cover all 128 tasks")
if not CPC_DECISION_ARTIFACT.is_file():
    raise RuntimeError(f"frozen CPC decision artifact missing: {CPC_DECISION_ARTIFACT}")
_CPC_DECISION = IC.read_json(CPC_DECISION_ARTIFACT)
if tuple(int(x) for x in _CPC_DECISION["high_support_task_ids"]) != HIGH_SUPPORT_TASKS:
    raise RuntimeError("embedded high-support set disagrees with the frozen CPC decision artifact")
if tuple(int(x) for x in _CPC_DECISION["low_support_task_ids"]) != LOW_SUPPORT_TASKS:
    raise RuntimeError("embedded low-support set disagrees with the frozen CPC decision artifact")
HIGH_SUPPORT_SET = frozenset(HIGH_SUPPORT_TASKS)
LOW_SUPPORT_SET = frozenset(LOW_SUPPORT_TASKS)

DECISION_RULES = {
    "targets": {
        "valid_ap": 0.280,
        "test_ap": 0.270,
        "support_min": SUPPORT_MIN,
        "high_support_tasks": 106,
    },
    #: a broad, real advance: both official splits at the pre-registered targets
    "advance": {
        "valid_best_min": 0.280,
        "test_best_min": 0.270,
        "high_support_mean_delta_vs_cpc_min": 0.003,
        "improved_high_support_vs_cpc_min": 70,
        "label": "PERFORMANCE_ADVANCE",
    },
    #: one split materially better, or both better but far below target
    "partial": {
        "valid_strong_min": 0.273,
        "test_strong_min": 0.265,
        "both_improved_min": 0.002,
        "high_support_mean_delta_vs_cpc_positive": True,
        "label": "PARTIAL_ADVANCE",
    },
    "regression": {
        "valid_max": 0.262944,   # CPC best - 0.003
        "test_max": 0.259356,    # CPC best - 0.003
        "high_support_mean_delta_vs_cpc_max": -0.003,
        "label": "REGRESSION",
    },
    "none": {"label": "NO_CLEAR_ADVANCE"},
    "invalid": {"label": "MECHANISM_INVALID"},
}

# ---------------------------------------------------------------------------
# helpers (thin inherited wrappers)
# ---------------------------------------------------------------------------

utcnow = IC.utcnow
write_json = IC.write_json
read_json = IC.read_json
sha256_file = IC.sha256_file
sha256_bytes = IC.sha256_bytes
log = IC.log
environment = IC.environment
seed_everything = IC.seed_everything
capture_rng_state = IC.capture_rng_state
rng_state_hashes = IC.rng_state_hashes
git_commit = IC.git_commit
git_state = IC.git_state
track_rel = IC.track_rel
state_fingerprint = IC.state_fingerprint
gpu_occupancy = IC.gpu_occupancy
host_facts = IC.host_facts
freeze_seal = IC.freeze_seal
test_rows_sha256_check = IC.test_rows_sha256_check
verify_frozen_reference_logits = IC.verify_frozen_reference_logits


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def is_new_parameter(name: str) -> bool:
    return str(name).startswith(NEW_MODULE_PREFIX)


def is_token_parameter(name: str) -> bool:
    return str(name).startswith(TOKEN_LAYER_PREFIXES)


def is_branch_parameter(name: str) -> bool:
    return str(name).startswith(BRANCH_PREFIX)


def is_root_parameter(name: str) -> bool:
    return str(name).startswith(ROOT_PREFIX)


def state_subset(state: Mapping[str, Any], *, new: bool) -> dict[str, Any]:
    return {k: v for k, v in state.items() if is_new_parameter(k) == bool(new)}


def trunk_subset(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in state.items()
        if not is_new_parameter(k) and not k.startswith("readout.")
    }


def ensure_cpc_cache(task: Any, cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Load the frozen CPC structure cache, rebuilding it only if it is absent.

    The rebuild uses the inherited CPC builder (deterministic, label-free) and is
    accepted only if its identity matches the frozen CPC round exactly.
    """
    base = Path(cache_dir) if cache_dir is not None else Path(P.CPC_CACHE_DIR)
    if not (base / "meta.json").is_file():
        log(f"[cache] frozen CPC structure cache absent at {track_rel(base)}; rebuilding")
        P.build_cpc_cache(task, base, log_every=25_000)
    cpc = P.load_cpc_cache(base)
    meta = dict(cpc["meta"])
    checks = {
        "n_rows": int(meta["n_rows"]) == int(CPC_CACHE["n_rows"]),
        "n_paths": int(meta["n_paths"]) == int(CPC_CACHE["n_paths"]),
        "rows_sha256": meta["rows_sha256"] == CPC_CACHE["rows_sha256"],
        "cache_sha256": meta["cache_sha256"] == CPC_CACHE["cache_sha256"],
        "schema": meta["enumeration_schema_version"] == CPC_CACHE["enumeration_schema_version"],
        "no_labels": bool(meta.get("labels_present")) is False,
        "no_test_labels": bool(meta.get("test_labels_present")) is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CPC cache identity mismatch: {checks}")
    return cpc


def recipe() -> dict[str, Any]:
    out = dict(RECIPE)
    out.update(
        {
            "round": ROUND_ID,
            "model": MODEL_NAME,
            "base_model": BASE_MODEL,
            "role": "one_shot_root_conditioned_hierarchical_path_composition",
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_cadence": "every_epoch_official_validation_ap",
            "widths": dict(W2_WIDTHS),
            "rhic": dict(RHIC),
            "parameters": int(EXPECTED_PARAMETERS["total"]),
            "parameter_breakdown": dict(EXPECTED_PARAMETERS),
            "lr_schedule": IC.BC.lr_audit(),
            "batch_plan": dict(FROZEN_PLAN),
            "test_policy": (
                "official Test targets are NaN in the training label artifact; "
                "Test logits are sealed before any Test label is opened"
            ),
        }
    )
    return out


__all__ = [name for name in dir() if not name.startswith("_")]
