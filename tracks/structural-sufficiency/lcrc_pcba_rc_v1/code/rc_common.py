"""Frozen definitions for the LCRC-RC round v1 (relation-combination, no message passing).

Round: ``lcrc-pcba-rc-v1``

Base model: the historical **LCRC-W2** (2x LCRC-A, 994,324 parameters).  This
round adds exactly one graph-level readout branch on top of the *post-dropout*
atom state ``h`` of W2:

    R       = softmax(Linear(128,12)(h), -1)                     [N_atoms, 12]
    T_d     = 144 * (R^T A_d R) / max(N_d,1)                      [12, 12]
    T_all   = 144 * (R^T A_all R) / max(N_all,1)                   [12, 12]
    F_d     = sqrt(w_d) * T_d          (REAL)                     [12, 12]
    F_d     = sqrt(w_d) * T_all        (BLIND, 0 when N_d == 0)   [12, 12]
    x       = concat(triu(F_0..F_5) [6*78], w [6], mean_a R [12], log1p n [1])
    z       = Linear(96,32)(Dropout(0.10)(SiLU(Linear(487,96)(LayerNorm(x)))))
    logits  = W2.post_local(h, batch) + z @ TaskEmbedding^T + task_bias

``A_d`` is the *undirected* atom-pair adjacency of the shortest-bond-path
distance bucket ``d`` inside one connected component of the original chemical
bond graph.  No atom state is ever updated, nothing is propagated along a bond,
and the branch only ever produces a per-graph (molecule-level) statistic.

Inheritance is read-only: ``bs_common`` (frozen data, recipe, order plan,
official OGB evaluator, W2 identity), ``bs_model`` (the W2 forward),
``pc_metrics`` (masked BCE + official AP), ``pc_model`` (collate / predict),
``cc_data`` (the official Test label gate used *after* the prediction freeze).
Nothing historical is edited.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

import sys as _sys  # noqa: E402

BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
BS_CODE = BS_ROUND / "code"
S3SEL_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1"
CC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1"
for _p in (str(BS_CODE), str(CODE_DIR)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import bs_common as BC  # noqa: E402  (adds the inherited code dirs, frozen identities)
import bs_model as BSM  # noqa: E402
import pc_metrics as MET  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-rc-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-rc-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-rc-checkpoint-v1"
SCHEMA_DIST_CACHE = "lcrc-pcba-rc-distance-cache-v1"
SCHEMA_PREFLIGHT = "lcrc-pcba-rc-preflight-v1"
SCHEMA_PROFILE = "lcrc-pcba-rc-profile-v1"
SCHEMA_TEST_FREEZE = "lcrc-pcba-rc-test-freeze-v1"
SCHEMA_TEST_RESULTS = "lcrc-pcba-rc-test-results-v1"
SCHEMA_COMPARISON = "lcrc-pcba-rc-comparison-v1"

# ---------------------------------------------------------------------------
# inherited, read-only identities
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
SEED = BC.SEED
SOURCE = BC.SOURCE
RECIPE = BC.RECIPE
scheduled_lr = BC.scheduled_lr
lr_audit = BC.lr_audit

#: Frozen W2 width dict (scale 2); passed explicitly to every constructor.
WIDTHS = BC.config_for_scale(2)
BASE_MODEL_KEY = "W2"
BASE_PARAMETERS = 994_324
HIDDEN_WIDTH = int(WIDTHS["width"])  # 128

# ---------------------------------------------------------------------------
# the historical W2 reference (read-only; never retrained in this round)
# ---------------------------------------------------------------------------

W2_RUN_DIR = BS_ROUND / "runs" / BASE_MODEL_KEY
W2_RUN_JSON = W2_RUN_DIR / "run.json"
W2_CURVE = W2_RUN_DIR / "train_curve.csv"
W2_PER_TASK = W2_RUN_DIR / "per_task_ap.csv"
W2_TRAIN_EVAL = W2_RUN_DIR / "train_eval_epoch40.json"
W2_CHECKPOINT_BEST = W2_RUN_DIR / "checkpoint_best.pt"
W2_CHECKPOINT_EPOCH40 = W2_RUN_DIR / "checkpoint_epoch40.pt"

W2 = {
    "round": "lcrc-pcba-backbone-scaling-v1",
    "arm": "W2",
    "seed": 0,
    "epochs": 40,
    "parameters": BASE_PARAMETERS,
    "best_epoch": 24,
    "best_valid_ap": 0.26527331610028143,
    "ap40": 0.26186691813350904,
    "train_ap40": 0.5818745570283522,
    "train_bce40": 0.02628940533386738,
    "valid_bce40": 0.04653792303564479,
    "best_checkpoint_sha256": "d86fe5e56e898d382acf66eb0533beb915f8ca39144c6c1c74e25e147667d2bd",
    "epoch40_checkpoint_sha256": "7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a",
    "global_plan_sha256": "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
    "consumed40_prefix_sha256": "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58",
    "trunk_init_fingerprint": "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d",
    "full_trunk_init_fingerprint": "109756bd24b2ab8589012031e3cf0b0389b0d275227b35044e9381a45c00a5db",
    "base_state_sha256_at_init": "6a3df38f00a5a4754c310a82d4ad5f71d7f89e3392f1c914aacc4113a3d6f53b",
    "gpu_hours": 1.3949014246219305,
    "epoch_seconds_mean": 124.85945256503764,
}
W2_TEST_AP_OFFICIAL = 0.2536164787414065  # s3-selective analysis/test_scores.json (best Valid epoch 24)

#: Official Test has been read before (continuation round).  This round treats
#: Test AP as the primary *engineering* comparison metric, not a blind estimate.
TEST_ROLE = "EXPOSED_TEST_PERFORMANCE_BENCHMARK"

# ---------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------

N_ROLES = 12
N_BUCKETS = 6
TRIU = N_ROLES * (N_ROLES + 1) // 2  # 78 (diagonal included)
ROLE_BLOCK = N_BUCKETS * TRIU  # 468
X_DIM = ROLE_BLOCK + N_BUCKETS + N_ROLES + 1  # 487
HIDDEN = 96
Z_DIM = 32
BRANCH_DROPOUT = 0.10

#: Distance buckets of the *shortest bond-path* distance inside a component.
#: ``(label, lo, hi)`` with inclusive bounds; ``hi=None`` means "and above".
BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("d1", 1, 1),
    ("d2", 2, 2),
    ("d3", 3, 3),
    ("d4", 4, 4),
    ("d5_6", 5, 6),
    ("d7_plus", 7, None),
)
#: Packed-array value for "different connected component / unreachable".
UNREACHABLE = 255

#: New-parameter arithmetic (exact; checked by sum()).
BRANCH_PARAMS: dict[str, dict[str, int]] = {
    "role_proj": {"in": HIDDEN_WIDTH, "out": N_ROLES, "parameters": HIDDEN_WIDTH * N_ROLES + N_ROLES},
    "combiner.norm": {"in": X_DIM, "out": X_DIM, "parameters": 2 * X_DIM},
    "combiner.fc1": {"in": X_DIM, "out": HIDDEN, "parameters": X_DIM * HIDDEN + HIDDEN},
    "combiner.fc2": {"in": HIDDEN, "out": Z_DIM, "parameters": HIDDEN * Z_DIM + Z_DIM},
    "task_embedding": {"in": N_TASKS, "out": Z_DIM, "parameters": N_TASKS * Z_DIM},
    "task_bias": {"in": N_TASKS, "out": 1, "parameters": N_TASKS},
}
BRANCH_ADDED = int(sum(int(v["parameters"]) for v in BRANCH_PARAMS.values()))
EXPECTED_PARAMETERS = int(BASE_PARAMETERS + BRANCH_ADDED)
PARAMETER_CEILING = 1_100_000

INIT_RULE = (
    "torch.manual_seed(0) -> historical W2 seed-0 build (bs_model.build_model, "
    "which itself uses torch.manual_seed(0) + FULL->C1 remap + the private "
    "readout generator seed*1000003+17) -> the frozen W2 state is copied "
    "bitwise into the RC model -> the new tensors are drawn from a private CPU "
    "generator (seed*1000003+offset, offset 1301 for REAL and 1302 for BLIND): "
    "the same kaiming-uniform rule the trunk Linears use, embedding-style "
    "normal(0, 1/sqrt(32)) for TaskEmbedding, zeros for task_bias, and the "
    "combiner's Linear(96,32) is zeroed so the residual logits are exactly 0 at "
    "initialisation.  No step advances the global torch RNG."
)
PRIVATE_GENERATOR_OFFSETS = {"REAL": 1301, "BLIND": 1302}

MODELS: dict[str, dict[str, Any]] = {
    "REAL": {
        "name": "LCRC-RC-REAL",
        "role": "relation_combination_real_bucket_role_binding",
        "kind": "real",
        "new_parameters": BRANCH_ADDED,
        "parameters": EXPECTED_PARAMETERS,
        "assigned_pool": "res-gpu1",
    },
    "BLIND": {
        "name": "LCRC-RC-BLIND",
        "role": "relation_combination_bucket_binding_destroyed_control",
        "kind": "blind",
        "new_parameters": BRANCH_ADDED,
        "parameters": EXPECTED_PARAMETERS,
        "assigned_pool": "res-gpu0",
    },
}
MODEL_ORDER = ("REAL", "BLIND")

#: Fixed engineering decision bands (resource/interpretation rules, not tests).
DECISION_RULES = {
    "promising": {"delta_test_ap_min": 0.005, "label": "PROMISING_RC_SIGNAL"},
    "weak": {"delta_test_ap_min": 0.002, "delta_test_ap_max": 0.005, "label": "WEAK_RC_SIGNAL"},
    "none": {"delta_test_ap_min": -0.002, "delta_test_ap_max": 0.002, "label": "NO_CLEAR_RC_GAIN"},
    "regression": {"delta_test_ap_max": -0.002, "label": "RC_REGRESSION"},
    "support_strata": {"ge20": 20, "ge5_lt20": 5},
}

BUDGET = {"gpu_target_hours_per_arm": 2.25, "gpu_hard_hours_per_arm": 3.0, "max_concurrent_gpus": 2}

# ---------------------------------------------------------------------------
# frozen distance cache location
# ---------------------------------------------------------------------------

DIST_CACHE_DIR = RUNS / "dist_cache"
DIST_CACHE_META = DIST_CACHE_DIR / "meta.json"

# ---------------------------------------------------------------------------
# thin IO helpers (inherited, already audited)
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
host_facts = BC.host_facts
gpu_occupancy = BC.gpu_occupancy
gpu_peak = BC.gpu_peak
state_fingerprint = BC.state_fingerprint


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def recipe_for_model(model_key: str) -> dict[str, Any]:
    if model_key not in MODELS:
        raise KeyError(model_key)
    m = MODELS[model_key]
    recipe = dict(RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "model": model_key,
            "model_name": m["name"],
            "role": m["role"],
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_cadence": 1,
            "widths": dict(WIDTHS),
            "parameters": int(EXPECTED_PARAMETERS),
            "new_parameters": int(BRANCH_ADDED),
            "lr_schedule": lr_audit(),
            "init_rule": INIT_RULE,
            "branch": {
                "n_roles": N_ROLES,
                "n_buckets": N_BUCKETS,
                "x_dim": X_DIM,
                "hidden": HIDDEN,
                "z_dim": Z_DIM,
                "dropout": BRANCH_DROPOUT,
                "buckets": [list(b) for b in BUCKETS],
                "triu_entries": TRIU,
            },
            "no_message_passing": True,
            "atom_state_updates": 0,
            "test_role": TEST_ROLE,
        }
    )
    return recipe


__all__ = [
    "ANALYSIS",
    "BASE_MODEL_KEY",
    "BASE_PARAMETERS",
    "BATCH_SIZE",
    "BC",
    "BRANCH_ADDED",
    "BRANCH_DROPOUT",
    "BRANCH_PARAMS",
    "BSM",
    "BS_CODE",
    "BS_ROUND",
    "BUCKETS",
    "BUDGET",
    "CC_ROUND",
    "CODE_DIR",
    "DATASET",
    "DECISION_RULES",
    "DIST_CACHE_DIR",
    "DIST_CACHE_META",
    "EXPECTED_PARAMETERS",
    "HIDDEN",
    "HIDDEN_WIDTH",
    "INIT_RULE",
    "MET",
    "MODELS",
    "MODEL_ORDER",
    "N_BUCKETS",
    "N_MOLECULES",
    "N_ROLES",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PARAMETER_CEILING",
    "PRIVATE_GENERATOR_OFFSETS",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "ROLE_BLOCK",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "S3SEL_ROUND",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_COMPARISON",
    "SCHEMA_DIST_CACHE",
    "SCHEMA_PREFLIGHT",
    "SCHEMA_PROFILE",
    "SCHEMA_RUN",
    "SCHEMA_TEST_FREEZE",
    "SCHEMA_TEST_RESULTS",
    "SEED",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TEST_ROLE",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TRIU",
    "TRACK",
    "UNREACHABLE",
    "W2",
    "W2_CHECKPOINT_BEST",
    "W2_CHECKPOINT_EPOCH40",
    "W2_CURVE",
    "W2_PER_TASK",
    "W2_RUN_DIR",
    "W2_RUN_JSON",
    "W2_TEST_AP_OFFICIAL",
    "W2_TRAIN_EVAL",
    "WIDTHS",
    "X_DIM",
    "Z_DIM",
    "capture_rng_state",
    "environment",
    "git_commit",
    "git_state",
    "gpu_occupancy",
    "gpu_peak",
    "host_facts",
    "log",
    "lr_audit",
    "parameter_count",
    "read_json",
    "recipe_for_model",
    "rng_state_hashes",
    "scheduled_lr",
    "seed_everything",
    "sha256_bytes",
    "sha256_file",
    "state_fingerprint",
    "track_rel",
    "utcnow",
    "write_json",
]
