"""Frozen definitions for the PCBA masked atom/bond SSL pilot.

Round: ``lcrc-pcba-atom-bond-ssl-pilot-v1``

This module only names what was fixed *before* the first formal GPU step:

* the inherited, read-only PCBA assets (``pcba_v1`` cache, official split,
  feature spec, Train-only normalisation, the 40-epoch seed-0 order plan);
* the two modules that are allowed to move (``atom_encoder``, ``bond_encoder``)
  and the two temporary reconstruction heads that are discarded;
* the anatomy of the two masked views (which cells are zeroed, which cells are
  targets, which cells are *not visible* to the network);
* the frozen SSL recipe, the pretext Probe rule and the transfer rule;
* the frozen historical reference and the pre-registered decision bands.

Nothing here is fitted and no number depends on a Validation result.

Official Test targets are unreachable in this round: no code path in this
round imports the label artifact for pretraining, and the supervised path
reuses the already-audited ``sg_data.load_task`` refusal gate (Test labels are
NaN in the frozen artifact and every Test row is refused).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# paths and inherited code (read-only imports)
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
SG_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1"
NMP_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1"
EP_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_explicit_performance_sprint_v1"

PILOT_CODE = PILOT / "code"
SG_CODE = SG_SPRINT / "code"
NMP_CODE = NMP_SPRINT / "code"
EP_CODE = EP_SPRINT / "code"

for _p in (str(NMP_CODE), str(SG_CODE), str(PILOT_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nmp_common as C  # noqa: E402  (sets up the whole inherited import tree)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-atom-bond-ssl-pilot-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-atom-bond-ssl-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-atom-bond-ssl-checkpoint-v1"
SCHEMA_PROBE = "lcrc-pcba-atom-bond-ssl-probe-v1"

DATASET = C.DATASET
N_TASKS = C.N_TASKS
N_MOLECULES = C.N_MOLECULES
N_TRAIN = C.N_TRAIN
N_VALID = C.N_VALID
N_TEST = C.N_TEST
BATCH_SIZE = C.BATCH_SIZE
TOTAL_EPOCHS = C.TOTAL_EPOCHS
STEPS_PER_EPOCH = C.STEPS_PER_EPOCH
TOTAL_STEPS_40 = C.TOTAL_STEPS_40

SOURCE = C.SOURCE  # frozen cache / split / label / spec identities
CACHE_PATH = PILOT / "cache" / "pcba_v1"
SPEC_PATH = PILOT / "analysis" / "pcba_spec.json"
SPLIT_PATH = PILOT / "analysis" / "pcba_split.json"
CACHE_MANIFEST = PILOT / "analysis" / "cache_manifest.json"

#: The frozen label artifact path is recorded *only* so that the label-access
#: audit can prove it was never opened.  Pretraining never reads it.
LABEL_ARTIFACT_NAME = "pcba_labels_trainvalid.npy"
LABEL_ARTIFACT_PATH = PILOT / "analysis" / LABEL_ARTIFACT_NAME

# ---------------------------------------------------------------------------
# model geometry (inherited, not redefined)
# ---------------------------------------------------------------------------

WIDTH = C.WIDTH                      # 64
ATOM_FEAT_DIM = 51
BOND_FEAT_DIM = 13
ATOM_HIDDEN = 128
ATOM_OUT = 96
BOND_HIDDEN = 160
BOND_OUT = 96
N_ATOM_CLASSES = 16                  # lipo_v1 atom_type 0..15
N_BOND_CLASSES = 5                   # lipo_v1 bond_type - 1, 0..4

ATOM_ENCODER_PARAMETERS = ATOM_FEAT_DIM * ATOM_HIDDEN + ATOM_HIDDEN + ATOM_HIDDEN * ATOM_OUT + ATOM_OUT
BOND_ENCODER_PARAMETERS = (
    (3 * ATOM_OUT + BOND_FEAT_DIM) * BOND_HIDDEN + BOND_HIDDEN + BOND_HIDDEN * BOND_OUT + BOND_OUT
)
ENCODER_PARAMETERS = ATOM_ENCODER_PARAMETERS + BOND_ENCODER_PARAMETERS

#: Reconstruction-head input widths and hidden width (frozen).
ATOM_HEAD_INPUT = 3 * ATOM_OUT            # a_v ++ mean_b(v) ++ mean_a(v) = 288
BOND_HEAD_INPUT = ATOM_OUT + 2 * (2 * ATOM_OUT)  # b_e ++ (ctx_u + ctx_v) ++ |ctx_u - ctx_v| = 480
HEAD_HIDDEN = 160

#: Downstream model identity (unchanged original LCRC-A).
EXPECTED_PARAMETERS = C.EXPECTED_PARAMETERS["A"]      # 287,644
EXPECTED_BASE_PARAMETERS = C.EXPECTED_PARAMETERS["C"]  # 245,184
EXPECTED_NEW_PARAMETERS = C.ARMS["A"]["new_parameters"]  # 42,460
BASE_STATE_TENSORS = 37
TRANSFERRED_PREFIXES = ("atom_encoder.", "bond_encoder.")
NON_TRANSFERRED_PREFIXES = ("local.", "f.", "rho.", "readout.")
MODEL_NAME = "A_SSL_INIT_S0"

# ---------------------------------------------------------------------------
# pretext Probe
# ---------------------------------------------------------------------------

PROBE_SALT = "pcba-ssl-probe-v1"
PROBE_ENTROPY = f"{PROBE_SALT}|"
N_PROBE = 4096
PROBE_VISIT_INDEX = 0

# ---------------------------------------------------------------------------
# mask definition
# ---------------------------------------------------------------------------

SSL_SEED = 0
MASK_RATIO_NUM = 3          # 15% = 3/20, rounded half up
MASK_RATIO_DEN = 20
MASK_ATOM_TAG = "pcba-ssl-mask-v1|atom"
MASK_BOND_TAG = "pcba-ssl-mask-v1|bond"
SHUFFLE_SEED = 20261107     # data-shuffle generator, isolated from mask entropy
HEAD_INIT_SEED = 20261108   # private head initialisation, isolated from encoders

# ---------------------------------------------------------------------------
# frozen SSL recipe
# ---------------------------------------------------------------------------

PRETRAIN: dict[str, Any] = {
    "seed": SSL_SEED,
    "updates": 10000,
    "batch_size": 128,
    "optimizer": "Adam",
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "gradient_clip_norm": 5.0,
    "dropout": 0.0,
    "precision": "fp32",
    "atom_mask_ratio": 0.15,
    "bond_mask_ratio": 0.15,
    "loss": "CE(masked atoms) + CE(masked physical bonds)",
    "eval_every_updates": 2000,
    "checkpoint_every_updates": 1000,
    "final_update_for_export": 10000,
    "early_stop_on_probe": False,
}

# ---------------------------------------------------------------------------
# frozen supervised recipe (identical to the historical scratch A)
# ---------------------------------------------------------------------------

SUPERVISED: dict[str, Any] = dict(C.SCRATCH_RECIPE)
SUPERVISED.update(
    {
        "model_name": MODEL_NAME,
        "arm": "A",
        "seed": 0,
        "epochs": TOTAL_EPOCHS,
        "eval_epochs": [10, 20, 30, 40],
        "primary_epoch": 40,
        "selection": "fixed epoch 40; no best-epoch substitution, no early stopping",
        "init": "historical seed-0 scratch init, then atom_encoder.*/bond_encoder.* replaced by SSL state",
        "optimizer_state": "freshly created AdamW; SSL optimizer state is never transferred",
        "trainable": "all original A parameters (encoders are NOT frozen)",
    }
)

# ---------------------------------------------------------------------------
# historical reference (read-only, never retrained here)
# ---------------------------------------------------------------------------

HISTORICAL_A0: dict[str, Any] = {
    "name": "original LCRC-A (historical run, not retrained this round)",
    "run_dir": "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A",
    "analysis_file": "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/analysis/phase_c_seed0_full.json",
    "seed": 0,
    "epoch": 40,
    "validation_ap": 0.24635609609821457,
    "parameters": 287644,
    "train_ap": 0.4196570175055804,
    "train_bce": 0.030857778624664715,
    "valid_bce": 0.04577261283603607,
    "commit": "b1b917dc4f72d9815b8f00813a11d95db3868298",
}
AP40_A0 = float(HISTORICAL_A0["validation_ap"])

#: Frozen reference values that must be reproduced bitwise before training.
HISTORICAL_PARITY = C.HISTORICAL_PARITY[0]

# ---------------------------------------------------------------------------
# decision bands (pre-registered resource bands, not significance tests)
# ---------------------------------------------------------------------------

DECISION_BANDS = (
    (0.005, "PROMISING_SINGLE_SEED"),
    (0.002, "WEAK_SIGNAL"),
    (-0.002, "NO_CLEAR_GAIN"),
)


def decision_label(delta_ap40: float) -> str:
    value = float(delta_ap40)
    if value >= 0.005:
        return "PROMISING_SINGLE_SEED"
    if value >= 0.002:
        return "WEAK_SIGNAL"
    if value > -0.002:
        return "NO_CLEAR_GAIN"
    return "NEGATIVE_TRANSFER"


#: Frozen pretext quality gate (fixed before the run; not tuned afterwards).
PRETEXT_GATE: dict[str, Any] = {
    "min_prior_ce_margin_nats": 0.05,
    "rule": (
        "probe atom CE <= SSL-fit atom prior CE - 0.05 nats AND "
        "probe bond CE <= SSL-fit bond prior CE - 0.05 nats AND "
        "both encoders moved away from the seed-0 scratch state"
    ),
    "failure_label": "PRETEXT_INVALID_OR_UNLEARNED",
    "note": (
        "a margin in nats below the class-marginal prior is the frozen notion of "
        "'clearly below the prior'; it is a resource gate, not a significance test"
    ),
}

# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

BUDGET: dict[str, Any] = {
    "gpu_target_hours": 5.0,
    "gpu_hard_hours": 7.0,
    "note": "includes preflight, failed attempts, pretraining and supervised finetuning",
}

# ---------------------------------------------------------------------------
# inherited helpers
# ---------------------------------------------------------------------------

utcnow = C.utcnow
write_json = C.write_json
read_json = C.read_json
sha256_file = C.sha256_file
sha256_bytes = C.sha256_bytes
sha256_int64 = C.sha256_int64
sha256_batches = C.sha256_batches
track_rel = C.track_rel
log = C.log
environment = C.environment
seed_everything = C.seed_everything
capture_rng_state = C.capture_rng_state
rng_state_hashes = C.rng_state_hashes
git_commit = C.git_commit
scheduled_lr = C.scheduled_lr
state_fingerprint = C.state_fingerprint


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def tensor_sha256(tensor: Any) -> str:
    import numpy as np

    return hashlib.sha256(
        np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32).tobytes()
    ).hexdigest()


def mask_seed_bytes(view: str, seed: int, row_id: int, visit_index: int) -> bytes:
    payload = f"{view}|{int(seed)}|{int(row_id)}|{int(visit_index)}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def n_masked(n_items: int) -> int:
    """Frozen mask count: round half up on 15%, at least one object."""
    n = int(n_items)
    if n <= 0:
        return 0
    return max(1, (MASK_RATIO_NUM * n + MASK_RATIO_DEN // 2) // MASK_RATIO_DEN)


__all__ = [
    "ANALYSIS",
    "ATOM_ENCODER_PARAMETERS",
    "ATOM_FEAT_DIM",
    "ATOM_HEAD_INPUT",
    "ATOM_HIDDEN",
    "ATOM_OUT",
    "BASE_STATE_TENSORS",
    "BOND_ENCODER_PARAMETERS",
    "BOND_FEAT_DIM",
    "BOND_HEAD_INPUT",
    "BOND_HIDDEN",
    "BOND_OUT",
    "BUDGET",
    "CACHE_MANIFEST",
    "CACHE_PATH",
    "CODE_DIR",
    "DATASET",
    "DECISION_BANDS",
    "ENCODER_PARAMETERS",
    "EP_CODE",
    "EP_SPRINT",
    "EXPECTED_BASE_PARAMETERS",
    "EXPECTED_NEW_PARAMETERS",
    "EXPECTED_PARAMETERS",
    "HEAD_HIDDEN",
    "HEAD_INIT_SEED",
    "HISTORICAL_A0",
    "HISTORICAL_PARITY",
    "LABEL_ARTIFACT_NAME",
    "LABEL_ARTIFACT_PATH",
    "MODEL_NAME",
    "MASK_ATOM_TAG",
    "MASK_BOND_TAG",
    "N_ATOM_CLASSES",
    "N_BOND_CLASSES",
    "NMP_CODE",
    "NMP_SPRINT",
    "NON_TRANSFERRED_PREFIXES",
    "N_MOLECULES",
    "N_PROBE",
    "N_TASKS",
    "PILOT",
    "PILOT_CODE",
    "PRETEXT_GATE",
    "PRETRAIN",
    "PROBE_ENTROPY",
    "PROBE_SALT",
    "PROBE_VISIT_INDEX",
    "PROTOCOL_ID",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_PROBE",
    "SCHEMA_RUN",
    "SG_CODE",
    "SG_SPRINT",
    "SHUFFLE_SEED",
    "SOURCE",
    "SPEC_PATH",
    "SPLIT_PATH",
    "SSL_SEED",
    "SUPERVISED",
    "TESTS",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TRANSFERRED_PREFIXES",
    "WIDTH",
    "capture_rng_state",
    "decision_label",
    "environment",
    "git_commit",
    "log",
    "mask_seed_bytes",
    "n_masked",
    "parameter_count",
    "read_json",
    "rng_state_hashes",
    "scheduled_lr",
    "seed_everything",
    "sha256_batches",
    "sha256_bytes",
    "sha256_file",
    "sha256_int64",
    "state_fingerprint",
    "tensor_sha256",
    "track_rel",
    "utcnow",
    "write_json",
]
