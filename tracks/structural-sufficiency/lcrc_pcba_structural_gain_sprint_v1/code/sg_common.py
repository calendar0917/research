"""Frozen definitions for the LCRC PCBA structural-gain sprint v1.

Round: ``lcrc-pcba-structural-gain-sprint-v1``

The round is a pre-registered performance screen.  All arm definitions,
thresholds, evaluation epochs and the two-phase promotion rules are fixed in
``PROTOCOL.md`` before any formal run.  This module only encodes them.

Frozen, read-only inheritance:

* ``lcrc_pcba_scale_pilot_v1``  -- feature cache, official split, labels, spec,
  Train-only normalization and the frozen LCRC-C1 model,
* ``lcrc_pcba_pipeline_perf_v1`` -- the already-gated ``PinnedStager``,
* ``lcrc_pcba_lr_probe_v1`` -- the epoch-9 warm-start checkpoint with AdamW
  moments.

Official Test targets are unreachable: the label artifact has every Test row
set to NaN and every training/evaluation path refuses a Test row.  No Test
metric is produced anywhere in this round.
"""

from __future__ import annotations

import hashlib
import math
import os
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

PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PILOT_CODE = PILOT / "code"
PERF = REPO / "tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1"
PERF_CODE = PERF / "code"
LR_PROBE = REPO / "tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1"
LR_PROBE_CODE = LR_PROBE / "code"
for _p in (str(PILOT_CODE), str(PERF_CODE), str(LR_PROBE_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lr_common as LC  # noqa: E402  (asserts the inherited PCBA constants)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-structural-gain-sprint-v1"
PROTOCOL_ID = ROUND_ID

DATASET = "ogbg-molpcba"
N_TASKS = 128
N_MOLECULES = 437_929
N_TRAIN = 350_343
N_VALID = 43_793
N_TEST = 43_793
SEED = 0
BATCH_SIZE = 128
TOTAL_EPOCHS = 40
PHASE1_EPOCHS = 12
STEPS_PER_EPOCH = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE  # 2738
TOTAL_STEPS_40 = TOTAL_EPOCHS * STEPS_PER_EPOCH

CACHE_PATH = PILOT / "cache" / "pcba_v1"
LABELS_PATH = PILOT / "analysis" / "pcba_labels_trainvalid.npy"
SPEC_PATH = PILOT / "analysis" / "pcba_spec.json"
SPLIT_PATH = PILOT / "analysis" / "pcba_split.json"
CACHE_MANIFEST = PILOT / "analysis" / "cache_manifest.json"

#: Frozen source: the epoch-9 best checkpoint of the low-LR probe.  Both model
#: weights and AdamW moments exist and are restored exactly.
SOURCE = {
    "round": "lcrc-pcba-lr-probe-v1",
    "group": "L",
    "group_name": "L_low_lr",
    "checkpoint": LR_PROBE / "runs" / "L_low_lr" / "checkpoint_best.pt",
    "checkpoint_sha256": "b158f1669f6d0ccbf5fc05f510050c69ef2fba9f6258cbe59d4d3d9a938eb6c5",
    "best_epoch": 9,
    "best_valid_ap": 0.2374902968831333,
    "optimizer_step_per_param": 9 * STEPS_PER_EPOCH,  # 24642
    "feature_spec_hash": "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee",
    "norm_fingerprint": "6ce50f989fc0789d9573f7f787a44a49a282531ac5a2db09c6600c67dc3293fb",
    "split_partition_sha256": "c872bd6311796826db9365c5c739c4cee94c3baa78fbd18006616feefc67bf4f",
    "label_artifact_sha256": "5c3b5fbe642d80113522832fe01090097b860010868c59f8696bc247d5027d01",
    "cache_sha256": "caad3e90dd76d1f1483d2a01501eba23413072a7b5a2fc4b62780c315aa59035",
    "n_parameters": 245_184,
    "n_state_tensors": 37,
    "n_optimizer_params": 31,
}

#: Historical references that are reported, never used as a replacement for the
#: pre-registered fixed-epoch comparisons.
AP0 = float(SOURCE["best_valid_ap"])
AP_READOUT = 0.2403392810

#: The four arms.  Pool assignment is the protocol assignment; the actual
#: device is read from the environment at run time and recorded.
ARMS: dict[str, dict[str, Any]] = {
    "C": {
        "name": "C_control",
        "role": "control_c1",
        "blocks": "none",
        "assigned_pool": "res-gpu0",
    },
    "F": {
        "name": "F_ffn2",
        "role": "experiment_ffn2",
        "blocks": "ffn",
        "assigned_pool": "res-gpu1",
    },
    "M": {
        "name": "M_mp2",
        "role": "experiment_mp2",
        "blocks": "mp",
        "assigned_pool": "res-gpu0",
    },
    "V": {
        "name": "V_mp2_graph_context",
        "role": "experiment_mp2_graph_context",
        "blocks": "mp_context",
        "assigned_pool": "res-gpu1",
    },
}
ARM_ORDER = ("C", "F", "M", "V")
BLOCK_WIDTH = 64
BLOCK_HIDDEN = 128

#: Shared recipe for Phase 1 and Phase 2.
RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "weight_decay": 1.0e-3,
    "base_lr_start": 5.0e-5,
    "base_lr_end": 1.0e-5,
    "new_lr_start": 3.0e-4,
    "new_lr_end": 3.0e-5,
    "scheduler": "per_step_cosine_over_40_epoch_horizon",
    "warmup": "none (restored AdamW moments are not re-warmed)",
    "dropout": 0.10,
    "batch_size": BATCH_SIZE,
    "gradient_clip": 5.0,
    "precision": "fp32",
    "epochs": TOTAL_EPOCHS,
    "phase1_epochs": PHASE1_EPOCHS,
    "early_stopping": False,
    "seed": SEED,
    "loss": "masked BCEWithLogitsLoss over observed entries only",
    "selection": "pre-registered fixed epochs; best-AP checkpoint is auxiliary",
    "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
}

#: Phase-3 from-scratch recipe.
SCRATCH_RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "weight_decay": 1.0e-3,
    "learning_rate": 3.0e-4,
    "warmup_epochs": 1.0,
    "warmup_start_lr": 3.0e-5,
    "end_lr": 1.0e-5,
    "scheduler": "1-epoch linear warmup then per-step cosine over the 40-epoch horizon",
    "dropout": 0.10,
    "batch_size": BATCH_SIZE,
    "gradient_clip": 5.0,
    "precision": "fp32",
    "epochs": TOTAL_EPOCHS,
    "early_stopping": False,
    "loss": "masked BCEWithLogitsLoss over observed entries only",
    "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
}

#: Fixed evaluation epochs.  Phase 2 additionally inherits 0/4/8/12 from Phase 1.
EVAL_EPOCHS: dict[int, tuple[int, ...]] = {
    1: (0, 4, 8, 12),
    2: (20, 30, 40),
    3: (10, 20, 30, 40),
}

#: Pre-registered promotion numbers.
DECISION: dict[str, Any] = {
    "ap0": AP0,
    "ap_readout_reference": AP_READOUT,
    "phase1": {
        "completed_epochs": 12,
        "d_base_min": 0.004,
        "d_control_min": 0.004,
        "n_tasks_expected": 127,
        "n_tasks_improved_min": 70,
        "support_subset_min_positive": 20,
        "tie_break_ap": 1e-5,
        "tie_break_order": ["F", "M", "V"],
    },
    "phase2": {
        "completed_epochs": 40,
        "ap_w40_min": 0.245,
        "d40_min": 0.006,
        "n_tasks_improved_min": 70,
        "support_subset_min_positive": 20,
    },
    "phase3": {
        "w_ap40_min": 0.250,
        "per_seed_improvement_min": 0.005,
    },
    "statistical_claim": "none; thresholds are resource decisions fixed in advance",
}

#: Wall-clock / GPU budgets (target; hard limits are enforced by the agent).
BUDGET: dict[str, Any] = {
    "wall_target_hours": 7.5,
    "wall_hard_hours": 8.0,
    "gpu_target_hours": 15.0,
    "gpu_hard_hours": 16.0,
    "no_new_gpu_stage_below_minutes": 30,
    # per-job soft/hard wall guards are passed on the command line
}

#: Fixed smoke/preflight sizes.
PREFLIGHT_SMOKE_STEPS = 30
PREFLIGHT_GRAD_STEPS = 3
PREFLIGHT_IDENTITY_MOLECULES = 8


# ---------------------------------------------------------------------------
# scheduler (pure functions of the global optimizer-step index)
# ---------------------------------------------------------------------------


def scheduled_lr(global_step: int, *, phase: int | str, kind: str) -> float:
    """LR applied *before* optimizer step ``global_step`` (0-based).

    ``kind`` is ``"base"`` (restored C1 parameters) or ``"new"`` (added
    parameters).  Phase 1/2 use the common 40-epoch cosine; Phase 3 uses the
    from-scratch warmup + cosine.
    """
    step = max(0, int(global_step))
    if str(phase) in ("1", "2"):
        frac = min(1.0, step / float(TOTAL_STEPS_40))
        if kind == "base":
            hi, lo = RECIPE["base_lr_start"], RECIPE["base_lr_end"]
        elif kind == "new":
            hi, lo = RECIPE["new_lr_start"], RECIPE["new_lr_end"]
        else:  # pragma: no cover - defensive
            raise KeyError(kind)
        return float(lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * frac)))
    if str(phase) == "3":
        if kind != "all":
            raise KeyError("phase 3 has a single parameter group (kind='all')")
        warmup = STEPS_PER_EPOCH
        hi = float(SCRATCH_RECIPE["learning_rate"])
        lo = float(SCRATCH_RECIPE["end_lr"])
        start = float(SCRATCH_RECIPE["warmup_start_lr"])
        if step < warmup:
            return float(start + (hi - start) * (step + 1) / warmup)
        frac = min(1.0, (step - warmup) / float(max(1, TOTAL_STEPS_40 - warmup)))
        return float(lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * frac)))
    raise KeyError(f"unknown phase {phase!r}")


def lr_audit() -> dict[str, Any]:
    """Freeze the exact LR endpoints / schedule shape reached by the rules."""
    p1 = {
        "base_step0": scheduled_lr(0, phase=1, kind="base"),
        "base_step12e": scheduled_lr(PHASE1_EPOCHS * STEPS_PER_EPOCH, phase=1, kind="base"),
        "base_step40e": scheduled_lr(TOTAL_STEPS_40, phase=1, kind="base"),
        "new_step0": scheduled_lr(0, phase=1, kind="new"),
        "new_step12e": scheduled_lr(PHASE1_EPOCHS * STEPS_PER_EPOCH, phase=1, kind="new"),
        "new_step40e": scheduled_lr(TOTAL_STEPS_40, phase=1, kind="new"),
    }
    p3 = {
        "step0": scheduled_lr(0, phase=3, kind="all"),
        "warmup_last": scheduled_lr(STEPS_PER_EPOCH - 1, phase=3, kind="all"),
        "post_warmup_first": scheduled_lr(STEPS_PER_EPOCH, phase=3, kind="all"),
        "step40e": scheduled_lr(TOTAL_STEPS_40, phase=3, kind="all"),
    }
    return {
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_steps_40_epochs": TOTAL_STEPS_40,
        "phase12": p1,
        "phase3": p3,
        "formula_phase12": (
            "lr(s) = lo + (hi-lo)*0.5*(1+cos(pi*s/S40)); base 5e-5->1e-5, new 3e-4->3e-5"
        ),
        "formula_phase3": (
            "steps 0..2737 linear 3e-5->3e-4; then lo+(hi-lo)*0.5*(1+cos(pi*(s-2738)/(S40-2738)))"
        ),
    }


# ---------------------------------------------------------------------------
# IO helpers (thin wrappers around the inherited, already-audited helpers)
# ---------------------------------------------------------------------------

utcnow = LC.utcnow
write_json = LC.write_json
read_json = LC.read_json
sha256_file = LC.sha256_file
sha256_bytes = LC.sha256_bytes
sha256_int64 = LC.sha256_int64
sha256_batches = LC.sha256_batches
track_rel = LC.track_rel
log = LC.log
environment = LC.environment
seed_everything = LC.seed_everything
capture_rng_state = LC.capture_rng_state
rng_state_hashes = LC.rng_state_hashes
git_commit = LC.git_commit
_json_default = LC._json_default


def recipe_for_arm(arm: str, phase: int) -> dict[str, Any]:
    if arm not in ARMS:
        raise KeyError(arm)
    recipe = dict(RECIPE)
    recipe.update(
        {
            "arm": arm,
            "arm_name": ARMS[arm]["name"],
            "role": ARMS[arm]["role"],
            "blocks": ARMS[arm]["blocks"],
            "assigned_pool": ARMS[arm]["assigned_pool"],
            "phase": int(phase),
            "eval_epochs": list(EVAL_EPOCHS[int(phase)]),
            "lr_schedule": lr_audit(),
        }
    )
    if int(phase) == 3:
        recipe = dict(SCRATCH_RECIPE)
        recipe.update(
            {
                "arm": arm,
                "arm_name": ARMS[arm]["name"],
                "role": ARMS[arm]["role"],
                "blocks": ARMS[arm]["blocks"],
                "assigned_pool": ARMS[arm]["assigned_pool"],
                "phase": 3,
                "eval_epochs": list(EVAL_EPOCHS[3]),
                "lr_schedule": lr_audit(),
            }
        )
    return recipe


__all__ = [
    "ANALYSIS",
    "AP0",
    "AP_READOUT",
    "ARMS",
    "ARM_ORDER",
    "BATCH_SIZE",
    "BUDGET",
    "CACHE_MANIFEST",
    "CACHE_PATH",
    "CODE_DIR",
    "DATASET",
    "DECISION",
    "EVAL_EPOCHS",
    "LABELS_PATH",
    "LC",
    "N_MOLECULES",
    "N_TASKS",
    "PHASE1_EPOCHS",
    "PILOT",
    "PILOT_CODE",
    "PERF",
    "PERF_CODE",
    "PREFLIGHT_GRAD_STEPS",
    "PREFLIGHT_IDENTITY_MOLECULES",
    "PREFLIGHT_SMOKE_STEPS",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCRATCH_RECIPE",
    "SEED",
    "SOURCE",
    "SPEC_PATH",
    "SPLIT_PATH",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "capture_rng_state",
    "environment",
    "git_commit",
    "log",
    "lr_audit",
    "read_json",
    "recipe_for_arm",
    "rng_state_hashes",
    "scheduled_lr",
    "seed_everything",
    "sha256_batches",
    "sha256_bytes",
    "sha256_file",
    "sha256_int64",
    "track_rel",
    "utcnow",
    "write_json",
]
