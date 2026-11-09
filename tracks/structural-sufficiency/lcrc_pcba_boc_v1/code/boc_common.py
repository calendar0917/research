"""Frozen definitions for the LCRC-BOC round v1 (bond objects + ring systems).

Round: ``lcrc-pcba-boc-v1``

Question: can a *single-shot*, learnable joint composition of

* bond-centred local instances ``P_uv = G[{u,v} union N(u) union N(v)]`` and
* ring-system instances (connected components of the non-bridge subgraph)

improve the official MolPCBA scaffold-split generalisation of LCRC-W2 when the
composition is inserted into the **atom main representation** (not added as a
graph-level logit residual)?

Inheritance is read-only.  The following historical implementations are the
only basis of this round:

* ``lcrc_pcba_backbone_scaling_v1/code/bs_model.py`` -- W2 trunk / readout,
* ``lcrc_pcba_backbone_scaling_v1/code/bs_common.py`` -- frozen recipe,
  data identities, official evaluator, order plan, W2 initialisation,
* ``lcrc_pcba_backbone_scaling_v1/code/bs_train.py`` -- the exact training step,
* ``lcrc_tu_quick_screen_v1/code/tu_model.py`` -- ``TULCRC`` / ``MLP``,
* ``lcrc_pcba_s3_selective_v1/code/sp_test.py`` -- freeze/score discipline,
* ``lcrc_pcba_continuation_recipe_v1/code/cc_data.py`` -- the official Test
  label gate.

No historical file is edited.  The official Test split is already exposed; this
round therefore treats Test as ``EXPOSED_TEST_ENGINEERING_COMPARISON_SET``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
BS_CODE = BS_ROUND / "code"
RC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_rc_v1"
RC_CODE = RC_ROUND / "code"
CC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1"
CC_CODE = CC_ROUND / "code"
S3SEL_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1"

for _p in (str(BS_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bs_common as BC  # noqa: E402  (sets up inherited code dirs / frozen data)
import bs_model as BSM  # noqa: E402
import pc_metrics as MET  # noqa: E402
import tu_model as TU  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-boc-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-boc-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-boc-checkpoint-v1"
SCHEMA_STRUCT_CACHE = "lcrc-pcba-boc-structure-cache-v1"
SCHEMA_AUDIT = "lcrc-pcba-boc-structure-audit-v1"
SCHEMA_PROFILE = "lcrc-pcba-boc-profile-v1"
SCHEMA_TEST_FREEZE = "lcrc-pcba-boc-test-freeze-v1"
SCHEMA_TEST_RESULTS = "lcrc-pcba-boc-test-results-v1"
SCHEMA_COMPARISON = "lcrc-pcba-boc-comparison-v1"

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

#: Frozen W2 width dict (scale 2), passed explicitly to the historical model.
WIDTHS = BC.config_for_scale(2)
BASE_MODEL_KEY = "W2"
BASE_PARAMETERS = 994_324
ATOM_WIDTH = int(WIDTHS["atom_out"])  # 192
HIDDEN_WIDTH = int(WIDTHS["width"])  # 128
BOND_WIDTH = int(WIDTHS["bond_out"])  # 192

#: Object widths fixed by the round brief.
OBJ = 64
RING_PAIR = 32
RING_DIST_EMB = 16
RING_BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, None))
N_RING_BUCKETS = len(RING_BUCKETS)

#: The two experimental arms.  They instantiate the same modules; only the
#: ring-system content that reaches the main representation differs.
MODELS: dict[str, dict[str, Any]] = {
    "FULL": {
        "name": "BOC-FULL",
        "role": "bond_objects_plus_ring_systems",
        "kind": "full",
        "ring_input": "real ring-system composition",
        "private_generator_offset": 811_001,
    },
    "BOND": {
        "name": "BOC-BOND",
        "role": "bond_objects_only_ring_channel_zero",
        "kind": "bond",
        "ring_input": "exact zeros; ring modules instantiated but bypassed",
        "private_generator_offset": 811_001,
    },
}
MODEL_ORDER = ("FULL", "BOND")

#: Exact parameter target, measured from the frozen module shapes and re-checked
#: at every build: 994,324 (W2) + 358,272 (bond/ring/fusion).
EXPECTED_PARAMETERS: dict[str, int] = {"FULL": 1_352_596, "BOND": 1_352_596}
PARAMETER_BAND = (1_200_000, 1_600_000)

# ---------------------------------------------------------------------------
# historical references (read-only; never retrained in this round)
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
#: The round brief's historical Test AP (s3-selective analysis).
W2_TEST_AP_BRIEF = 0.2536164787414065
#: Same-pipeline replay in the RC round (used only as a cross-check).
W2_TEST_AP_RC_REPLAY = 0.2536319543673203
W2_TEST_AP_40_BRIEF = 0.250985
W2_TEST_AP_40_RC_REPLAY = 0.25098520156897997

RC = {
    "round": "lcrc-pcba-rc-v1",
    "arm": "REAL",
    "model": "RC-REAL",
    "seed": 0,
    "epochs": 40,
    "parameters": 1_051_022,
    "best_epoch": 33,
    "best_valid_ap": 0.2677177024413342,
    "ap40": 0.26695028589805453,
    "gpu_hours": 1.9185818754296957,
    "best_checkpoint": RC_ROUND / "runs" / "REAL" / "checkpoint_best.pt",
    "best_checkpoint_sha256": "5a94dc4b386fa13d0d74e0391e10dd61a97e37040e7c1301125c96cf5b4646b1",
    "epoch40_checkpoint": RC_ROUND / "runs" / "REAL" / "checkpoint_epoch40.pt",
    "epoch40_checkpoint_sha256": "f0e3c87ffa531f1215a6ec215d5456e3970cd13f7647086db9b4e601c9217151",
    "test_ap_best": 0.25594109880999005,
    "test_ap_40": 0.2561664520457869,
}
#: Re-read from the RC analysis in ``reference_identities``; the hashes above
#: are checked against the files at freeze time.

#: Round-brief historical Test APs, used verbatim in REPORT.md.
HISTORICAL_TEST_AP = {
    "W2": W2_TEST_AP_BRIEF,
    "RC-REAL": 0.25594109880999005,
}

TEST_ROLE = "EXPOSED_TEST_ENGINEERING_COMPARISON_SET"

#: Structural-cache location (gitignored, rebuilt by a label-free CPU job).
STRUCT_CACHE_DIR = RUNS / "structure_cache"

#: Engineering decision bands from the round brief (not significance tests).
DECISION_BANDS = {
    "strong": 0.270,
    "meaningful": 0.265,
    "small": 0.258,
}

# ---------------------------------------------------------------------------
# IO / environment helpers (thin wrappers over already-audited inherited code)
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
_json_default = BC._json_default


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def state_sha256(state: Mapping[str, Any]) -> str:
    """Stable state fingerprint (key + dtype + shape + bytes)."""
    import hashlib

    import torch

    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("utf-8"))
        h.update(str(tuple(value.shape)).encode("utf-8"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def fitted_lr_audit() -> dict[str, Any]:
    return BC.lr_audit()


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
            "widths": dict(WIDTHS),
            "base_parameters": int(BASE_PARAMETERS),
            "parameters": int(EXPECTED_PARAMETERS.get(model_key, -1)),
            "lr_schedule": lr_audit(),
            "init_rule": (
                "W2 trunk: historical seed-0 state exactly; bond/ring/fusion "
                "modules: private CPU generator, standard Linear init, no zeroed "
                "output layer"
            ),
            "test_policy": (
                "Train/Valid labels only in training; Test structure is label-free; "
                "official Test labels read only after the prediction freeze"
            ),
        }
    )
    return recipe


__all__ = [
    "ANALYSIS",
    "ATOM_WIDTH",
    "BASE_MODEL_KEY",
    "BASE_PARAMETERS",
    "BOND_WIDTH",
    "BATCH_SIZE",
    "BSM",
    "CC_CODE",
    "CC_ROUND",
    "CODE_DIR",
    "DATASET",
    "DECISION_BANDS",
    "EXPECTED_PARAMETERS",
    "HIDDEN_WIDTH",
    "HISTORICAL_TEST_AP",
    "MODELS",
    "MODEL_ORDER",
    "MET",
    "N_MOLECULES",
    "N_RING_BUCKETS",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "OBJ",
    "PARAMETER_BAND",
    "PROTOCOL_ID",
    "RC",
    "RC_CODE",
    "RC_ROUND",
    "RECIPE",
    "REPO",
    "RING_BUCKETS",
    "RING_DIST_EMB",
    "RING_PAIR",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "S3SEL_ROUND",
    "SCHEMA_AUDIT",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_COMPARISON",
    "SCHEMA_PROFILE",
    "SCHEMA_RUN",
    "SCHEMA_STRUCT_CACHE",
    "SCHEMA_TEST_FREEZE",
    "SCHEMA_TEST_RESULTS",
    "SEED",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "STRUCT_CACHE_DIR",
    "TESTS",
    "TEST_ROLE",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TU",
    "W2",
    "W2_CHECKPOINT_BEST",
    "W2_CHECKPOINT_EPOCH40",
    "W2_CURVE",
    "W2_PER_TASK",
    "W2_RUN_DIR",
    "W2_RUN_JSON",
    "W2_TRAIN_EVAL",
    "W2_TEST_AP_40_BRIEF",
    "W2_TEST_AP_40_RC_REPLAY",
    "W2_TEST_AP_BRIEF",
    "W2_TEST_AP_RC_REPLAY",
    "WIDTHS",
    "capture_rng_state",
    "environment",
    "fitted_lr_audit",
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
    "state_sha256",
    "track_rel",
    "utcnow",
    "write_json",
]
