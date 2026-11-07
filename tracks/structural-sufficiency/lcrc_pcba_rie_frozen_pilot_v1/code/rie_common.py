"""Frozen definitions for the LCRC-RIE frozen pilot (round 1).

Round: ``lcrc-pcba-rie-frozen-pilot-v1``

The round asks exactly one question:

    On top of the frozen original LCRC-A encoder, do *real* motif-motif
    topological relations add Validation AP evidence that is not already
    supplied by the same motif composition, the same atomic states and the same
    model capacity?

Everything this round needs is frozen here before the first GPU run:

* the historical **original LCRC-A seed-0 epoch-40** checkpoint and its SHA-256
  (no SSL / A1 / A2 / A3 / G / V checkpoint is admissible),
* the frozen Validation AP of that checkpoint,
* the 8 motif families (6 RDKit ``ChemicalFeatures`` families + 2 explicit
  SMARTS families), the three distance bins and the 165-dimensional pair
  feature,
* the REAL / SHUFFLE control definition and its per-row RNG,
* the tiny task-specific residual head (8,784 trainable parameters) and its
  optimisation recipe,
* the fixed evaluation epochs (0, 2, 4, 8) and the epoch-8 primary comparison,
* the decision rules and the resource budget.

Official Test is never read: the inherited label artifact has every Test row set
to NaN, ``lr_data.load_task`` refuses a finite Test label, and no code path in
this round indexes ``task.test_rows``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# paths / inherited code (read-only)
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PERF = REPO / "tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1"
LR_PROBE = REPO / "tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1"
SG_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1"
NMP_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1"

NMP_CODE = NMP_SPRINT / "code"
for _p in (str(NMP_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ``nmp_common`` puts the pilot / perf / lr-probe / structural-gain code dirs on
# ``sys.path`` and re-exports the frozen PCBA identities this round inherits.
import nmp_common as NMP  # noqa: E402  (import side effect: inherited paths)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-rie-frozen-pilot-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-rie-frozen-pilot-run-v1"

DATASET = NMP.DATASET
N_TASKS = NMP.N_TASKS
N_MOLECULES = NMP.N_MOLECULES
N_TRAIN = NMP.N_TRAIN
N_VALID = NMP.N_VALID
N_TEST = NMP.N_TEST
WIDTH = NMP.WIDTH  # frozen LCRC-A atom-state width (64)
SEED = 0

# ---------------------------------------------------------------------------
# frozen source checkpoint: original LCRC-A, seed 0, epoch 40
# ---------------------------------------------------------------------------

SOURCE: dict[str, Any] = {
    "round": "lcrc-pcba-no-mp-sprint-v1",
    "arm": "A",
    "arm_name": "A_attention_pooling",
    "mode": "attention_pool",
    "run_dir": NMP_SPRINT / "runs" / "phaseC_seed0_A",
    "checkpoint": NMP_SPRINT / "runs" / "phaseC_seed0_A" / "checkpoint_epoch40.pt",
    "checkpoint_sha256": "f110935a6055f3c087f9d0b40580506a512cdeb09f6af67a56bde787cc93a437",
    "checkpoint_schema": "lcrc-pcba-no-mp-checkpoint-v1",
    "checkpoint_kind": "phaseC_epoch40",
    "phase": "C",
    "seed": 0,
    "epoch": 40,
    "global_step": 109520,
    "n_parameters": 287_644,
    "new_parameters": 42_460,
    "baseline_parameters": 245_184,
    "state_dict_tensors": 57,
    "git_commit": "b1b917dc4f72d9815b8f00813a11d95db3868298",
    "valid_ap_stored": 0.24635609609821457,
    "valid_bce_stored": 0.04577261283603607,
    "valid_logits_sha256_stored": "7d1b6042485e1493bad4912d8915ccf3b2eeada87eb351a14d6f46345a145c51",
    "valid_probs_sha256_stored": "77ebfa995e9d72893bfc2001773c7f3edbecf6b1ac34ec4e302e7438234b46c1",
}

#: Decision anchor fixed by the round instruction (and equal to the run registry
#: value up to 1e-9).  ``VALID_AP_TOL`` is the frozen reproduction tolerance.
AP_A = 0.24635609609821457
AP_A_INSTRUCTION = 0.2463560956
VALID_AP_TOL = 1e-4
#: The checkpoint's own stored Validation AP must agree with ``AP_A`` too.
AP_A_STORED_TOL = 1e-9

#: Inherited frozen data identities (refuse to run on anything else).
SOURCE_HASHES = {
    "feature_spec_hash": "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee",
    "norm_fingerprint": "6ce50f989fc0789d9573f7f787a44a49a282531ac5a2db09c6600c67dc3293fb",
    "split_partition_sha256": "c872bd6311796826db9365c5c739c4cee94c3baa78fbd18006616feefc67bf4f",
    "label_artifact_sha256": "5c3b5fbe642d80113522832fe01090097b860010868c59f8696bc247d5027d01",
}
N_SCORABLE_TASKS_EXPECTED = 127

# ---------------------------------------------------------------------------
# motif vocabulary: 8 chemical families (frozen; never extended)
# ---------------------------------------------------------------------------

MOTIF_FAMILIES: tuple[dict[str, Any], ...] = (
    {"id": 0, "name": "Donor", "source": "rdkit_ChemicalFeatures", "smarts": None},
    {"id": 1, "name": "Acceptor", "source": "rdkit_ChemicalFeatures", "smarts": None},
    {"id": 2, "name": "Aromatic", "source": "rdkit_ChemicalFeatures", "smarts": None},
    {"id": 3, "name": "PosIonizable", "source": "rdkit_ChemicalFeatures", "smarts": None},
    {"id": 4, "name": "NegIonizable", "source": "rdkit_ChemicalFeatures", "smarts": None},
    {"id": 5, "name": "Hydrophobe", "source": "rdkit_ChemicalFeatures", "smarts": None},
    {"id": 6, "name": "Carbonyl", "source": "smarts", "smarts": "[CX3]=[OX1]"},
    {"id": 7, "name": "Halogen", "source": "smarts", "smarts": "[F,Cl,Br,I]"},
)
N_MOTIF_FAMILIES = len(MOTIF_FAMILIES)
FAMILY_NAMES: tuple[str, ...] = tuple(f["name"] for f in MOTIF_FAMILIES)
FAMILY_NAME_TO_ID: dict[str, int] = {f["name"]: int(f["id"]) for f in MOTIF_FAMILIES}
CARBONYL_SMARTS = "[CX3]=[OX1]"
HALOGEN_SMARTS = "[F,Cl,Br,I]"

#: RDKit ``BaseFeatures.fdef`` is the standard definition source for families
#: 0-5.  The digest is recorded in every coverage/manifest artifact.
RDKIT_FDEF_SHA256 = "766f9790513f8f94c67ec6d1c8b6d5bede59b81f6da8c2081f57e49fb28d9784"

#: Unordered family pair index, ``a <= b``: ``36 = 8*9/2`` columns.
N_FAMILY_PAIRS = N_MOTIF_FAMILIES * (N_MOTIF_FAMILIES + 1) // 2  # 36

# ---------------------------------------------------------------------------
# frozen relation definition
# ---------------------------------------------------------------------------

PAIR_FEATURE_DIM = 2 * WIDTH + N_FAMILY_PAIRS + 1  # 64 + 64 + 36 + 1 = 165
#: (min_distance, max_distance_inclusive); ``None`` = unbounded above.
DISTANCE_BINS: tuple[tuple[int, int | None], ...] = ((3, 4), (5, 6), (7, None))
N_BINS = len(DISTANCE_BINS)  # 3
MIN_PAIR_DISTANCE = 3
RELATION_SHAPE = (N_BINS, PAIR_FEATURE_DIM)  # (3, 165)
RELATION_DIM_FLAT = N_BINS * PAIR_FEATURE_DIM  # 495

SHUFFLE_TAG = "rie-distance-shuffle-v1"
SHUFFLE_SEED = 0
RING_POLICY = (
    "min physical bond-graph shortest-path distance over all atom pairs of the two "
    "motifs; pairs with distance < 3 or in different connected components are dropped"
)

#: RMS normalisation of the relation features (REAL Train only).
RMS_EPS = 1e-4
RMS_FLOOR_SCALE = 1.0

# ---------------------------------------------------------------------------
# residual head (the only trainable object in this round)
# ---------------------------------------------------------------------------

HEAD: dict[str, Any] = {
    "input_shape": [N_BINS, PAIR_FEATURE_DIM],
    "shared_projection": {"in": PAIR_FEATURE_DIM, "out": 16, "bias": False},
    "relation_evidence_dim": N_BINS * 16,  # 48
    "task_weight": {"in": N_BINS * 16, "out": N_TASKS, "bias": False},
    "parameter_count": PAIR_FEATURE_DIM * 16 + N_BINS * 16 * N_TASKS,  # 8784
    "forward": "logits = frozen_A_logits + W_task @ flatten_b( W_shared @ S_b )",
    "initialisation": {
        "shared_relation_projection": "PyTorch default nn.Linear init (kaiming_uniform_, a=sqrt(5)) "
        "under an explicit torch.Generator(manual_seed(0))",
        "task_weight": "exact zeros",
        "both_arms": "bitwise identical initial state",
    },
}
HEAD_PARAMETERS = int(HEAD["parameter_count"])
assert HEAD_PARAMETERS == 165 * 16 + 48 * 128 == 8784

BATCH_SIZE = 1024
EPOCHS = 8
EVAL_EPOCHS: tuple[int, ...] = (0, 2, 4, 8)
MAIN_EPOCH = 8
STEPS_PER_EPOCH = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE  # 343

RECIPE: dict[str, Any] = {
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "optimizer": "AdamW",
    "learning_rate": 3.0e-4,
    "lr_schedule": "constant",
    "weight_decay": 1.0e-3,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "gradient_clip_norm": 5.0,
    "precision": "fp32",
    "dropout": 0.0,
    "scheduler": "none",
    "early_stopping": False,
    "loss": "observed-entry masked BCEWithLogits (+ 1e-5 * mean |W_task|)",
    "l1_coefficient": 1.0e-5,
    "l1_scope": "task_weight.weight only (128 x 48); not the shared projection, not frozen A",
    "row_order": "torch.Generator().manual_seed(0) + torch.randperm(n_train) per epoch; "
    "the identical plan feeds REAL and SHUFFLE",
    "eval_epochs": list(EVAL_EPOCHS),
    "main_epoch": MAIN_EPOCH,
    "trainable_parameters": HEAD_PARAMETERS,
    "frozen_A_parameters": int(SOURCE["n_parameters"]),
    "test_policy": "official Test targets are NaN and are never indexed",
}

ARMS: dict[str, dict[str, Any]] = {
    "real": {"name": "REAL_relation", "role": "true motif-pair topology", "assigned_pool": "res-gpu1"},
    "shuffle": {
        "name": "SHUFFLE_relation",
        "role": "distance-bin permutation control (same motifs, pairs, features, bin counts)",
        "assigned_pool": "res-gpu1",
    },
}
ARM_ORDER = ("real", "shuffle")

# ---------------------------------------------------------------------------
# decision rules (fixed before the first formal run)
# ---------------------------------------------------------------------------

DECISION: dict[str, Any] = {
    "promising": {
        "label": "PROMISING_RELATION_SIGNAL",
        "d_base_min": 0.005,
        "d_ctrl_min": 0.003,
        "d_ge20_min": 0.003,
        "improved_ge20_min": 60,
    },
    "weak": {"label": "WEAK_OR_MIXED_SIGNAL", "d_base_min": 0.002},
    "no_signal": {"label": "NO_RELATION_SIGNAL"},
    "invalid": {
        "source_checkpoint": "SOURCE_CHECKPOINT_UNAVAILABLE_OR_MISMATCH",
        "intervention": "INSUFFICIENT_RELATION_INTERVENTION",
        "resource": "INCOMPLETE_RESOURCE",
        "incomplete": "INCOMPLETE",
    },
    "min_intervention_fraction": 0.15,
    "ge20_threshold": 20,
    "strata": {"ge20": 20, "mid": (5, 19), "lt5": 5},
    "main_epoch": MAIN_EPOCH,
    "interpretation_guardrails": [
        "A gain over A without a gain over SHUFFLE is readout capacity, not relation evidence.",
        "Single-seed, single-epoch differences are resource-decision evidence, not significance.",
        "Validation labels have driven earlier model selection; they are not a pristine holdout.",
        "This round never authorises the full LCRC-RIE joint training by itself.",
    ],
}

BUDGET: dict[str, Any] = {
    "gpu_hours_target": 0.5,
    "gpu_hours_hard": 1.0,
    "structures_soft_seconds": 3600.0,
    "features_soft_seconds": 1800.0,
    "train_soft_seconds": 1200.0,
    "audit_soft_seconds": 900.0,
    "timing_probe_molecules": 1024,
}

EXTRACT: dict[str, Any] = {
    "batch_size": 512,
    "smoke_molecules": 8,
    "model_eval": True,
    "requires_grad": False,
    "inference_mode": True,
    "atom_state_width": WIDTH,
    "saved_atom_states": "not persisted; h is consumed per batch to build graph-level relations",
    "logits": "raw frozen LCRC-A logits, one forward per molecule, stored as float32",
    "rms_fit_split": "real train only",
    "valid_rows_used_for_rms_fit": 0,
}


# ---------------------------------------------------------------------------
# inherited helpers (thin aliases over already-audited code)
# ---------------------------------------------------------------------------

utcnow = NMP.utcnow
write_json = NMP.write_json
read_json = NMP.read_json
sha256_file = NMP.sha256_file
sha256_bytes = NMP.sha256_bytes
sha256_int64 = NMP.sha256_int64
track_rel = NMP.track_rel
log = NMP.log
environment = NMP.environment
seed_everything = NMP.seed_everything
capture_rng_state = NMP.capture_rng_state
rng_state_hashes = NMP.rng_state_hashes
git_commit = NMP.git_commit
state_fingerprint = NMP.state_fingerprint
json_default = NMP._json_default


def sha256_array(arr: np.ndarray) -> str:
    """SHA-256 of dtype + shape + contiguous payload (content fingerprint)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def family_pair_index(a: int, b: int) -> int:
    """Unordered family-pair column index in ``[0, 36)``; symmetric in a, b."""
    a, b = int(a), int(b)
    if not (0 <= a < N_MOTIF_FAMILIES and 0 <= b < N_MOTIF_FAMILIES):
        raise ValueError(f"family ids out of range: {(a, b)}")
    if a > b:
        a, b = b, a
    return a * (2 * N_MOTIF_FAMILIES - a + 1) // 2 + (b - a)


def bin_for_distance(distance: int) -> int:
    """Frozen distance -> bin map; ``-1`` means 'not eligible' (dropped)."""
    d = int(distance)
    if d < MIN_PAIR_DISTANCE:
        return -1
    for idx, (lo, hi) in enumerate(DISTANCE_BINS):
        if d >= lo and (hi is None or d <= hi):
            return idx
    return -1


def shuffle_seed_sequence(official_row_id: int) -> np.random.SeedSequence:
    """Deterministic per-molecule RNG: seed 0 + SHA-256 of the official row id."""
    digest = hashlib.sha256(f"{SHUFFLE_TAG}|{int(official_row_id)}".encode()).hexdigest()
    return np.random.SeedSequence(entropy=int(digest, 16), spawn_key=(SHUFFLE_SEED,))


def shuffle_shares_motif_pairs(*_args: Any, **_kwargs: Any) -> bool:  # pragma: no cover - doc only
    raise NotImplementedError("documentation helper; see rie_features.shuffle_bins")


def gpu_peak(device: Any) -> dict[str, int]:
    import torch

    if getattr(device, "type", str(device)) != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def assert_inherited_hashes(task: Any) -> dict[str, Any]:
    """Fail hard if the frozen data / spec / norm identities drifted."""
    found = {
        "feature_spec_hash": str(task.spec.hash()),
        "norm_fingerprint": str(task.norm["norm_fingerprint"]),
        "split_partition_sha256": str(task.split_partition_sha256),
        "label_artifact_sha256": str(task.label_artifact_sha256),
    }
    mismatches = {
        k: {"expected": SOURCE_HASHES[k], "found": found[k]}
        for k in SOURCE_HASHES
        if found[k] != SOURCE_HASHES[k]
    }
    if mismatches:
        raise RuntimeError(f"inherited PCBA identities drifted: {mismatches}")
    if int(np.asarray(task.train_rows).size) != N_TRAIN or int(np.asarray(task.valid_rows).size) != N_VALID:
        raise RuntimeError("official split sizes drifted")
    return {"found": found, "expected": dict(SOURCE_HASHES), "passed": True}


__all__ = [
    "ANALYSIS",
    "AP_A",
    "AP_A_INSTRUCTION",
    "AP_A_STORED_TOL",
    "ARMS",
    "ARM_ORDER",
    "BATCH_SIZE",
    "BUDGET",
    "CARBONYL_SMARTS",
    "CODE_DIR",
    "DATASET",
    "DECISION",
    "DISTANCE_BINS",
    "EPOCHS",
    "EVAL_EPOCHS",
    "EXTRACT",
    "FAMILY_NAMES",
    "FAMILY_NAME_TO_ID",
    "HALOGEN_SMARTS",
    "HEAD",
    "HEAD_PARAMETERS",
    "MAIN_EPOCH",
    "MIN_PAIR_DISTANCE",
    "MOTIF_FAMILIES",
    "N_BINS",
    "N_FAMILY_PAIRS",
    "N_MOLECULES",
    "N_MOTIF_FAMILIES",
    "N_SCORABLE_TASKS_EXPECTED",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PAIR_FEATURE_DIM",
    "PROTOCOL_ID",
    "RDKIT_FDEF_SHA256",
    "RECIPE",
    "RELATION_DIM_FLAT",
    "RELATION_SHAPE",
    "REPO",
    "RING_POLICY",
    "RMS_EPS",
    "RMS_FLOOR_SCALE",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCHEMA_RUN",
    "SEED",
    "SHUFFLE_SEED",
    "SHUFFLE_TAG",
    "SOURCE",
    "SOURCE_HASHES",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TRACK",
    "VALID_AP_TOL",
    "WIDTH",
    "assert_inherited_hashes",
    "bin_for_distance",
    "capture_rng_state",
    "environment",
    "family_pair_index",
    "git_commit",
    "gpu_peak",
    "json_default",
    "log",
    "read_json",
    "rng_state_hashes",
    "seed_everything",
    "sha256_array",
    "sha256_bytes",
    "sha256_file",
    "sha256_int64",
    "shuffle_seed_sequence",
    "state_fingerprint",
    "track_rel",
    "utcnow",
    "write_json",
]
