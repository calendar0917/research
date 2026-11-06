"""Frozen definitions for the LCRC PCBA explicit-performance sprint v1.

Round: ``lcrc-pcba-explicit-performance-sprint-v1``.

Everything about data, labels, split, normalization, batch order, loss,
optimizer, schedule, evaluation epochs and the (single) adaptive COMBO rule is
pre-registered in ``PROTOCOL.md`` and encoded here.  The code inherits the
already-audited PCBA assets and the frozen LCRC-C1/A architecture; it does not
re-derive any of them.

Official Test targets are unreachable: the frozen label artifact has every Test
row set to NaN, ``sg_data.load_task`` refuses finite Test labels, and no code
path in this round indexes the Test split.

Model ladder (all share the original C1 local encoder and the original 4-head
graph attention readout of NMPModel(arm="A")):

* ``A0``  -- exactly the original A (strict same-round control),
* ``A1``  -- A + task-conditioned attention residual (128 tasks x 2 queries),
* ``A2``  -- A + 10 physicochemical descriptors + missing flag residual head,
* ``A3``  -- A + explicit rooted 2-hop branch tokens (hashed, 8192 buckets),
* ``COMBO`` -- A + the two best eligible single modules, same definitions.

No added module may read ``bond_index`` / ``bond_feat`` / ``nbr_*`` / ``path_*``
or perform any inter-atom hidden-state update.  A3's adjacency is used only in
the offline deterministic feature builder.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PERF = REPO / "tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1"
LR_PROBE = REPO / "tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1"
SG_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1"
NMP_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1"

PILOT_CODE = PILOT / "code"
PERF_CODE = PERF / "code"
LR_PROBE_CODE = LR_PROBE / "code"
SG_CODE = SG_SPRINT / "code"
NMP_CODE = NMP_SPRINT / "code"

for _p in (str(PILOT_CODE), str(PERF_CODE), str(LR_PROBE_CODE), str(SG_CODE), str(NMP_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sg_common as SC  # noqa: E402  (inherited frozen PCBA constants)
import nmp_common as NC  # noqa: E402  (inherited no-MP arms / recipe / parity)

ANALYSIS = ROUND / "analysis"
CACHE = ROUND / "cache"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-explicit-performance-sprint-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-explicit-performance-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-explicit-performance-checkpoint-v1"
SCHEMA_PREFLIGHT = "lcrc-pcba-explicit-performance-preflight-v1"

DATASET = SC.DATASET
N_TASKS = SC.N_TASKS
N_MOLECULES = SC.N_MOLECULES
N_TRAIN = SC.N_TRAIN
N_VALID = SC.N_VALID
N_TEST = SC.N_TEST
BATCH_SIZE = SC.BATCH_SIZE
TOTAL_EPOCHS = 40
STEPS_PER_EPOCH = SC.STEPS_PER_EPOCH
TOTAL_STEPS_40 = SC.TOTAL_STEPS_40
SEED = 0

EVAL_EPOCHS = (10, 20, 30, 40)
FINAL_EPOCH = 40

SOURCE = dict(SC.SOURCE)

# ---------------------------------------------------------------------------
# environment-free identities inherited from the no-MP sprint
# ---------------------------------------------------------------------------

BASE_PARAMETERS = 245_184
A0_PARAMETERS = 287_644
HISTORICAL_A0 = {
    "valid_ap40": 0.2463560956,  # per the formal nmp Phase-C seed-0 A run
    "parameters": 287_644,
    "new_parameters": 42_460,
    "run": "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A/run.json",
    "base_init_fingerprint": "1b1f1ad35ab33f4107bc600166fce838d5e16ff5def2348743c73c0211dcf7c3",
    "global_plan_sha256": "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
    "consumed40_prefix_sha256": "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58",
}
HISTORICAL_PARITY = {0: dict(NC.HISTORICAL_PARITY[0])}

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

#: Modules added on top of the original A.  Order is fixed; COMBO composes a
#: subset of them with exactly the single-model definition.
MODULES = {
    "A1": {
        "name": "task_conditioned_attention",
        "new_parameters": 41_344,
        "reads": ["h", "atom_batch", "n_graphs"],
    },
    "A2": {
        "name": "physchem_descriptor_residual",
        "new_parameters": 9_088,
        "reads": ["physchem"],
    },
    "A3": {
        "name": "explicit_rooted_branch_tokens",
        "new_parameters": 266_304,
        "reads": ["h", "bt_ptr", "bt_bucket"],
    },
}

ARMS: dict[str, dict[str, Any]] = {
    "A0": {"name": "A0_original_A", "modules": (), "task": "control", "role": "strict_control", "assigned_pool": "res-gpu1"},
    "A1": {"name": "A1_task_conditioned", "modules": ("A1",), "task": "Q1", "role": "experiment", "assigned_pool": "res-gpu1"},
    "A2": {"name": "A2_physchem", "modules": ("A2",), "task": "Q2", "role": "experiment", "assigned_pool": "res-gpu1"},
    "A3": {"name": "A3_branch_tokens", "modules": ("A3",), "task": "Q3", "role": "experiment", "assigned_pool": "res-gpu1"},
}
ARM_ORDER = ("A0", "A1", "A2", "A3")
ALLOWED_COMBOS = (("A1", "A2"), ("A1", "A3"), ("A2", "A3"))
COMBO_PRIORITY = ("A1", "A3", "A2")


def arm_modules(arm: str, combine: str = "") -> tuple[str, ...]:
    """Resolve the module set for an arm name, validating the fixed order."""
    if arm in ARMS:
        if combine:
            raise ValueError(f"--combine is only valid for COMBO, not {arm}")
        return tuple(ARMS[arm]["modules"])
    if arm != "COMBO":
        raise KeyError(arm)
    parts = tuple(x.strip() for x in str(combine).split(",") if x.strip())
    if parts not in ALLOWED_COMBOS:
        raise ValueError(f"COMBO modules {parts} not allowed; allowed={ALLOWED_COMBOS}")
    return parts


def expected_parameters(modules: tuple[str, ...]) -> int:
    return int(A0_PARAMETERS + sum(int(MODULES[m]["new_parameters"]) for m in modules))


def arm_label(arm: str, modules: tuple[str, ...]) -> str:
    return arm if arm != "COMBO" else "COMBO_" + "+".join(modules)


# ---------------------------------------------------------------------------
# inherited frozen recipe (identical object values to the no-MP sprint)
# ---------------------------------------------------------------------------

SCRATCH_RECIPE = dict(SC.SCRATCH_RECIPE)
RECIPE = SCRATCH_RECIPE


def scheduled_lr(global_step: int) -> float:
    return float(SC.scheduled_lr(int(global_step), phase=3, kind="all"))


def lr_audit() -> dict[str, Any]:
    return {
        "steps_per_epoch": int(STEPS_PER_EPOCH),
        "total_steps_40_epochs": int(TOTAL_STEPS_40),
        "step0": scheduled_lr(0),
        "warmup_last": scheduled_lr(STEPS_PER_EPOCH - 1),
        "post_warmup_first": scheduled_lr(STEPS_PER_EPOCH),
        "step10e": scheduled_lr(10 * STEPS_PER_EPOCH),
        "step20e": scheduled_lr(20 * STEPS_PER_EPOCH),
        "step30e": scheduled_lr(30 * STEPS_PER_EPOCH),
        "step40e": scheduled_lr(40 * STEPS_PER_EPOCH),
        "formula": (
            "steps 0..2737 linear 3e-5->3e-4; then "
            "1e-5+(3e-4-1e-5)*0.5*(1+cos(pi*(s-2738)/(109520-2738)))"
        ),
    }


# ---------------------------------------------------------------------------
# pre-registered decision rules (never edited after seeing results)
# ---------------------------------------------------------------------------

DECISION = {
    "combo_eligible_delta_min": 0.003,
    "combo_strong_delta_min": 0.005,
    "combo_min_eligible": 2,
    "combo_max_count": 1,
    "tie_order": list(COMBO_PRIORITY),
    "screen_positive_delta_min": 0.005,
    "weak_signal_delta_min": 0.002,
    "main_metric": "epoch40_official_validation_macro_ap",
    "support_strata": {"ge20": 20, "s5_19": 5, "lt5": 5},
    "budget": {
        "gpu_target_hours": 8.0,
        "gpu_hard_hours": 10.0,
        "max_concurrent_gpus": 2,
        "note": "preflight, failed attempts and CPU prep below GPU-hour accounting; CPU is 0 GPU-h",
    },
}

# ---------------------------------------------------------------------------
# feature definitions (frozen; hashes recorded at build time)
# ---------------------------------------------------------------------------

N_BRANCH_BUCKETS = 8192
BRANCH_TOKEN_DIM = 32
BRANCH_ALGO = "ep-rooted-branch-token-v1"
PHYSCHEM_NAMES = (
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "log1p_MolWt",
    "Crippen_MolLogP",
    "NumRotatableBonds",
    "TotalFormalCharge",
    "NumAromaticRings",
    "MaxRingAtomCount",
    "FractionCSP3",
)
N_PHYSCHEM = len(PHYSCHEM_NAMES)
PHYSCHEM_DIM = N_PHYSCHEM + 1  # + missing flag
PHYSCHEM_ALGO = "ep-rdkit-physchem-v1"
PHYSCHEM_ZERO_STD_POLICY = "std == 0 -> scale 1.0"

CACHE_A2 = CACHE / "physchem_v1.npz"
CACHE_A2_META = CACHE / "physchem_v1.json"
CACHE_A3 = CACHE / "branch_tokens_v1.npz"
CACHE_A3_META = CACHE / "branch_tokens_v1.json"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

utcnow = SC.utcnow
write_json = SC.write_json
read_json = SC.read_json
sha256_file = SC.sha256_file
sha256_bytes = SC.sha256_bytes
sha256_batches = SC.sha256_batches
track_rel = SC.track_rel
log = SC.log
environment = SC.environment
seed_everything = SC.seed_everything
capture_rng_state = SC.capture_rng_state
rng_state_hashes = SC.rng_state_hashes
git_commit = SC.git_commit
_json_default = SC._json_default
state_fingerprint = NC.state_fingerprint
parameter_count = NC.parameter_count


def array_sha(arr: Any, dtype: str = "float32") -> str:
    import numpy as np

    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    return sha256_bytes(a.tobytes())


def recipe_for(arm: str, modules: tuple[str, ...]) -> dict[str, Any]:
    recipe = dict(SCRATCH_RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "arm": arm,
            "arm_name": arm_label(arm, modules),
            "modules": list(modules),
            "seed": int(SEED),
            "final_epoch": int(FINAL_EPOCH),
            "eval_epochs": list(EVAL_EPOCHS),
            "lr_schedule": lr_audit(),
        }
    )
    return recipe


def digest_module_spec(modules: tuple[str, ...]) -> str:
    payload = "|".join(
        [
            ROUND_ID,
            BRANCH_ALGO,
            PHYSCHEM_ALGO,
            str(N_BRANCH_BUCKETS),
            str(BRANCH_TOKEN_DIM),
            ",".join(PHYSCHEM_NAMES),
            ",".join(modules),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "ALLOWED_COMBOS",
    "ANALYSIS",
    "ARMS",
    "ARM_ORDER",
    "BASE_PARAMETERS",
    "BATCH_SIZE",
    "BRANCH_ALGO",
    "BRANCH_TOKEN_DIM",
    "CACHE",
    "CACHE_A2",
    "CACHE_A2_META",
    "CACHE_A3",
    "CACHE_A3_META",
    "CODE_DIR",
    "DATASET",
    "DECISION",
    "EVAL_EPOCHS",
    "FINAL_EPOCH",
    "HISTORICAL_A0",
    "HISTORICAL_PARITY",
    "MODULES",
    "N_BRANCH_BUCKETS",
    "N_MOLECULES",
    "N_PHYSCHEM",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PHYSCHEM_ALGO",
    "PHYSCHEM_DIM",
    "PHYSCHEM_NAMES",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_PREFLIGHT",
    "SCHEMA_RUN",
    "SCRATCH_RECIPE",
    "SEED",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TRACK",
    "arm_label",
    "arm_modules",
    "array_sha",
    "capture_rng_state",
    "digest_module_spec",
    "environment",
    "expected_parameters",
    "git_commit",
    "log",
    "lr_audit",
    "parameter_count",
    "read_json",
    "recipe_for",
    "rng_state_hashes",
    "scheduled_lr",
    "seed_everything",
    "sha256_batches",
    "sha256_bytes",
    "sha256_file",
    "state_fingerprint",
    "track_rel",
    "utcnow",
    "write_json",
]
