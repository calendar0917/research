"""Frozen definitions of the LCRC-PW round v1 (pair-interaction readout).

Round: ``lcrc-pcba-pw-v1``

The trunk is the historical **LCRC-W2** (994,324 parameters: one atom encode,
one bond encode, one-shot S1/S2 local statistics, per-atom ``f``, global
``rho``, four-head attention pooling).  This round adds exactly one graph-level
branch that reads the *single* post-dropout atom state ``h`` of that trunk:

    D_ij    = shortest-bond-path distance of the unordered pair i<j inside one
              connected component, binned into 1,2,3,4,5-6,>=7 (precomputed)
    p_ij    = LN( SiLU( W_sum(h_i+h_j) + W_diff|h_i-h_j| + W_prod(h_i*h_j)
                       + E[bucket_ij] ) )                       [P, 48]
    k_ij    = Linear(48,48)(p_ij) ,  v_ij = Linear(48,48)(p_ij)
    s_kij   = <q_k, k_ij> / sqrt(48)                            [P, 16]
    a_kij   = softmax over the pairs *of one molecule* of s_kij
    z_k     = sum_ij a_kij v_ij                                 [G, 16, 48]
    r       = Linear(128,128)( Dropout(0.10)( SiLU( Linear(768,128)(
                  LayerNorm(z.reshape(G, 768)) ) ) ) )          [G, 128]
    logits  = W2_logits + r * 1[pair_count > 0]

Hard constraints (audited, not asserted):

* ``h`` is produced once, by the frozen W2 ``local`` block, and is never
  updated: no message passing, no adjacency propagation, no atom-state write,
  no second atom encode, one pair-attention layer only, no pair-token
  iteration;
* the pair token is symmetric in ``i <-> j`` by construction and verified
  bitwise;
* the final ``Linear(128,128)`` is zero-initialised, so at step 0 the model
  logits are exactly the W2 logits, and a molecule without a single legal pair
  keeps a residual of exactly zero (an explicit per-graph mask, not a
  numerical accident);
* every original W2 tensor is bitwise equal to the historical seed-0 initial
  state; all new tensors come from a private CPU generator that never advances
  the global torch RNG and never changes the W2 seed state.

Inheritance is read-only: ``bs_common`` (frozen data, recipe, order plan,
official evaluator, W2 identity), ``bs_model`` (the W2 forward), ``bs_train``
(the training step), ``pc_metrics`` (masked BCE + official AP),
``cc_data`` (the official Test label gate used only *after* a prediction
freeze).
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
RC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_rc_v1"
RC_CODE = RC_ROUND / "code"
CC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1"
for _p in (str(BS_CODE), str(RC_CODE), str(CODE_DIR)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import bs_common as BC  # noqa: E402  (adds the inherited code dirs, frozen identities)
import bs_model as BSM  # noqa: E402
import pc_metrics as MET  # noqa: E402
import rc_common as RCC  # noqa: E402  (frozen W2 reference + recipe identities)
import rc_dist as RD  # noqa: E402  (the inherited Train-only task gate)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-pw-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-pw-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-pw-checkpoint-v1"
SCHEMA_PAIR_INDEX = "lcrc-pcba-pw-pair-index-v1"
SCHEMA_PREFLIGHT = "lcrc-pcba-pw-preflight-v1"
SCHEMA_PROFILE = "lcrc-pcba-pw-profile-v1"
SCHEMA_TEST_FREEZE = RCC.SCHEMA_TEST_FREEZE
SCHEMA_TEST_RESULTS = "lcrc-pcba-pw-test-results-v1"
SCHEMA_COMPARISON = "lcrc-pcba-pw-comparison-v1"

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

WIDTHS = BC.config_for_scale(2)
BASE_MODEL_KEY = "W2"
BASE_PARAMETERS = RCC.BASE_PARAMETERS  # 994,324
HIDDEN_WIDTH = int(WIDTHS["width"])  # 128

W2_RUN_DIR = BS_ROUND / "runs" / BASE_MODEL_KEY
W2_CHECKPOINT_BEST = W2_RUN_DIR / "checkpoint_best.pt"
W2_CHECKPOINT_EPOCH40 = W2_RUN_DIR / "checkpoint_epoch40.pt"
W2 = RCC.W2
W2_TEST_AP_OFFICIAL = RCC.W2_TEST_AP_OFFICIAL

#: The LCRC-RC round is the direct predecessor and supplies the RC-REAL /
#: RC-BLIND reference numbers.  It is read-only here: its scores, its per-task
#: table and its frozen Test logits are never modified.
RC_ANALYSIS = RC_ROUND / "analysis"
RC_SCORES = RC_ANALYSIS / "test_scores.json"
RC_PER_TASK = RC_ANALYSIS / "per_task_valid_test.csv"
RC_PREDICTIONS = RC_ANALYSIS / "test_predictions"
RC_TASK_DELTA = RC_ANALYSIS / "task_delta_analysis.json"

#: Official Test has been read before in this track; PW treats it as an
#: *exposed engineering comparison set*, never as a model-selection signal.
TEST_ROLE = "EXPOSED_TEST_ENGINEERING_COMPARISON_SET"

# ---------------------------------------------------------------------------
# the PW branch
# ---------------------------------------------------------------------------

N_BUCKETS = RCC.N_BUCKETS  # 6
BUCKETS = RCC.BUCKETS  # (1,2,3,4,5-6,>=7); inherited verbatim
UNREACHABLE = RCC.UNREACHABLE  # 255 sentinel for "different component"

PAIR_DIM = 48
N_QUERIES = 16
Z_DIM = N_QUERIES * PAIR_DIM  # 768
COMBINER_HIDDEN = 128
BRANCH_DROPOUT = 0.10
SCORE_SCALE = PAIR_DIM ** -0.5

#: New-parameter arithmetic (exact; checked against the built model).
BRANCH_PARAMS: dict[str, dict[str, Any]] = {
    "w_sum": {"in": HIDDEN_WIDTH, "out": PAIR_DIM, "bias": False,
              "parameters": HIDDEN_WIDTH * PAIR_DIM},
    "w_diff": {"in": HIDDEN_WIDTH, "out": PAIR_DIM, "bias": False,
               "parameters": HIDDEN_WIDTH * PAIR_DIM},
    "w_prod": {"in": HIDDEN_WIDTH, "out": PAIR_DIM, "bias": False,
               "parameters": HIDDEN_WIDTH * PAIR_DIM},
    "dist_embedding": {"in": N_BUCKETS, "out": PAIR_DIM, "bias": False,
                       "parameters": N_BUCKETS * PAIR_DIM},
    "norm": {"in": PAIR_DIM, "out": PAIR_DIM, "bias": True,
             "parameters": 2 * PAIR_DIM},
    "query": {"in": N_QUERIES, "out": PAIR_DIM, "hidden_dim": True,
              "parameters": N_QUERIES * PAIR_DIM},
    "key": {"in": PAIR_DIM, "out": PAIR_DIM, "bias": True,
            "parameters": PAIR_DIM * PAIR_DIM + PAIR_DIM},
    "value": {"in": PAIR_DIM, "out": PAIR_DIM, "bias": True,
              "parameters": PAIR_DIM * PAIR_DIM + PAIR_DIM},
    "combiner.norm": {"in": Z_DIM, "out": Z_DIM, "bias": True,
                      "parameters": 2 * Z_DIM},
    "combiner.fc1": {"in": Z_DIM, "out": COMBINER_HIDDEN, "bias": True,
                     "parameters": Z_DIM * COMBINER_HIDDEN + COMBINER_HIDDEN},
    "combiner.fc2": {"in": COMBINER_HIDDEN, "out": N_TASKS, "bias": True,
                     "parameters": COMBINER_HIDDEN * N_TASKS + N_TASKS},
}
BRANCH_ADDED = int(sum(int(v["parameters"]) for v in BRANCH_PARAMS.values()))
EXPECTED_PARAMETERS = int(BASE_PARAMETERS + BRANCH_ADDED)
PARAMETER_CEILING = 1_200_000

INIT_RULE = (
    "torch.manual_seed(0) -> historical W2 seed-0 build (bs_model.build_model) -> "
    "the frozen W2 state is copied bitwise into the PW model -> every new tensor is "
    "drawn from a private CPU generator (seed*1000003+1401) that never advances the "
    "global torch RNG: kaiming_uniform(a=sqrt(5)) + fan_in bias for the pair and "
    "combiner Linears, normal(0, 1/sqrt(48)) for the distance embedding and the 16 "
    "shared queries, default LayerNorm(1,0), and the final Linear(128,128) weight and "
    "bias are zeroed so the residual logits are exactly 0 at initialisation."
)
PRIVATE_GENERATOR_OFFSET = 1401

MODEL_KEY = "PW"
MODEL = {
    "key": MODEL_KEY,
    "name": "LCRC-PW-v1",
    "role": "one_shot_pair_interaction_residual_on_w2_atom_state",
    "new_parameters": BRANCH_ADDED,
    "parameters": EXPECTED_PARAMETERS,
    "assigned_pool": "res-gpu1",
}
MODEL_ORDER = (MODEL_KEY,)

#: Fixed engineering decision bands (identical to the RC round so the two
#: rounds are read with the same ruler; they are decision labels, not tests).
#: PW decision bands (PROTOCOL.md §7).  The thresholds are the RC round's, the
#: labels are PW's own so a report can never confuse a PW verdict with an RC one.
DECISION_RULES = {
    "promising": {"label": "PROMISING_PW_SIGNAL", "delta_test_ap_min": 0.005},
    "weak": {"label": "WEAK_PW_SIGNAL", "delta_test_ap_min": 0.002},
    "none": {"label": "NO_CLEAR_PW_GAIN", "delta_test_ap_min": -0.002},
    "regression": {"label": "PW_REGRESSION", "delta_test_ap_min": float("-inf")},
}
#: The inherited RC bands, kept for the cross-round consistency audit only.
INHERITED_RC_DECISION_RULES = RCC.DECISION_RULES
BUDGET = {
    "gpu_hard_hours": 3.0,
    "gpu_target_hours": 2.5,
    "profile_before_training": True,
    "no_pair_sampling": True,
}

#: Reference numbers used by ``pw_compare`` (all read-only, all same pipeline).
REFERENCE_W2 = {
    "model": "W2",
    "parameters": int(W2["parameters"]),
    "best_epoch": int(W2["best_epoch"]),
    "best_valid_ap_recorded": float(W2["best_valid_ap"]),
    "valid_ap_at_40": float(W2["ap40"]),
    "gpu_hours": float(W2["gpu_hours"]),
}

# ---------------------------------------------------------------------------
# pair-index cache location
# ---------------------------------------------------------------------------

PAIR_CACHE_DIR = RUNS / "pair_cache"
PAIR_CACHE_META = PAIR_CACHE_DIR / "meta.json"

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


def recipe_for_model(model_key: str = MODEL_KEY) -> dict[str, Any]:
    if model_key != MODEL_KEY:
        raise KeyError(model_key)
    recipe = dict(RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "model": model_key,
            "model_name": MODEL["name"],
            "role": MODEL["role"],
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_cadence": 1,
            "widths": dict(WIDTHS),
            "parameters": int(EXPECTED_PARAMETERS),
            "new_parameters": int(BRANCH_ADDED),
            "lr_schedule": lr_audit(),
            "init_rule": INIT_RULE,
            "branch": {
                "kind": "one_shot_pair_attention_residual",
                "pair_dim": PAIR_DIM,
                "n_queries": N_QUERIES,
                "z_dim": Z_DIM,
                "combiner_hidden": COMBINER_HIDDEN,
                "dropout": BRANCH_DROPOUT,
                "buckets": [list(b) for b in BUCKETS],
                "score_scale": SCORE_SCALE,
                "softmax": "query-wise over the legal atom pairs of one molecule",
                "pair_layers": 1,
            },
            "no_message_passing": True,
            "atom_state_updates": 0,
            "pair_state_updates": 0,
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
    "COMBINER_HIDDEN",
    "DATASET",
    "DECISION_RULES",
    "INHERITED_RC_DECISION_RULES",
    "EXPECTED_PARAMETERS",
    "HIDDEN_WIDTH",
    "INIT_RULE",
    "MET",
    "MODEL",
    "MODEL_KEY",
    "MODEL_ORDER",
    "N_BUCKETS",
    "N_MOLECULES",
    "N_QUERIES",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PAIR_CACHE_DIR",
    "PAIR_CACHE_META",
    "PAIR_DIM",
    "PARAMETER_CEILING",
    "PRIVATE_GENERATOR_OFFSET",
    "PROTOCOL_ID",
    "RC_ANALYSIS",
    "RC_CODE",
    "RC_PER_TASK",
    "RC_PREDICTIONS",
    "RC_ROUND",
    "RC_SCORES",
    "RC_TASK_DELTA",
    "RCC",
    "RD",
    "RECIPE",
    "REFERENCE_W2",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_COMPARISON",
    "SCHEMA_PAIR_INDEX",
    "SCHEMA_PREFLIGHT",
    "SCHEMA_PROFILE",
    "SCHEMA_RUN",
    "SCHEMA_TEST_FREEZE",
    "SCHEMA_TEST_RESULTS",
    "SCORE_SCALE",
    "SEED",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TEST_ROLE",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TRACK",
    "UNREACHABLE",
    "W2",
    "W2_CHECKPOINT_BEST",
    "W2_CHECKPOINT_EPOCH40",
    "W2_TEST_AP_OFFICIAL",
    "WIDTHS",
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
