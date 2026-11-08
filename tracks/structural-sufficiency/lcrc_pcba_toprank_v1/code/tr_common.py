"""Frozen definitions for the LCRC-W2 TopRank round v1.

Round: ``lcrc-pcba-toprank-v1``

One model only: the historical **LCRC-W2** (994,324 parameters) built by the
frozen ``bs_model.build_model("W2", ...)`` code path.  No layer, width, readout,
feature, normaliser or data row changes.  The only change is the training
objective:

    loss_total = masked_bce_loss(logits, y) + lambda_rank * rank_loss(logits, y)

``rank_loss`` is a per-assay hard-negative ranking surrogate (log-sum-exp over
the batch's observed negatives, softplus against the batch's observed
positives).  It is *not* an exact Average-Precision loss.

Everything inherited is read-only: ``bs_common`` (frozen data, recipe, order
plan, official OGB evaluator, W2 initialisation), ``pc_metrics`` (masked BCE and
the official AP), ``bs_model`` (the W2 forward).  Official Test targets are NaN
in the local label artifact; they become readable only through
``tr_test.py score`` after the prediction freeze.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

import sys as _sys  # noqa: E402

BS_CODE = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/code"
if str(BS_CODE) not in _sys.path:
    _sys.path.insert(0, str(BS_CODE))

import bs_common as BC  # noqa: E402  (adds the inherited code dirs, frozen identities)
import bs_model as BSM  # noqa: E402
import pc_metrics as MET  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-toprank-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-toprank-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-toprank-checkpoint-v1"
SCHEMA_CALIBRATION = "lcrc-pcba-toprank-calibration-v1"
SCHEMA_PREFLIGHT = "lcrc-pcba-toprank-preflight-v1"
SCHEMA_SELECTION = "lcrc-pcba-toprank-selection-v1"
SCHEMA_TEST_RESULTS = "lcrc-pcba-toprank-test-results-v1"

DATASET = BC.DATASET
N_TASKS = BC.N_TASKS
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
SCHEDULED_LR = BC.scheduled_lr
GLOBAL_PLAN_SHA256 = "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6"

MODEL_KEY = "W2"
MODEL_PARAMETERS = 994_324

#: Historical W2 reference (never retrained in this round).  The Test numbers
#: are the frozen values from ``lcrc_pcba_s3_selective_v1/analysis/test_scores.json``
#: and the checkpoint identity from the backbone-scaling W2 run.
W2 = {
    "round": "lcrc-pcba-backbone-scaling-v1",
    "run_dir": REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/runs/W2",
    "checkpoint_best": REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/runs/W2/checkpoint_best.pt",
    "checkpoint_epoch40": REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/runs/W2/checkpoint_epoch40.pt",
    "run_json": REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/runs/W2/run.json",
    "best_epoch": 24,
    "best_valid_ap": 0.26527331610028143,
    "ap40": 0.261866918,
    "checkpoint_best_sha256": "d86fe5e56e898d382acf66eb0533beb915f8ca39144c6c1c74e25e147667d2bd",
    "parameters": MODEL_PARAMETERS,
    "base_parameters": 848_096,
    "readout_parameters": 146_228,
    "init_state_sha256": "6a3df38f00a5a4754c310a82d4ad5f71d7f89e3392f1c914aacc4113a3d6f53b",
    "trunk_init_fingerprint": "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d",
    "full_trunk_init_fingerprint": "109756bd24b2ab8589012031e3cf0b0389b0d275227b35044e9381a45c00a5db",
    "epoch0_valid_ap": 0.029063331894071648,
    "epoch0_valid_bce": 1.9391586913627363,
    "gpu_hours": 1.3949014246219305,
    "test_ap_official": 0.2536164787414065,
    "test_ap_independent": 0.2536164787414066,
    "test_bce": 0.04858394826016791,
    "test_logits_sha256": "13a55b2729a11ebcea9b9d9af4ee49759ec80736f7b632f141ef4fbfb6270d98",
    "test_scores_source": (
        "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1/analysis/test_scores.json"
    ),
    "test_prediction_manifest": (
        "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1/analysis/test_prediction_manifest.json"
    ),
}

#: Official Test has been read before (continuation round, 2026-11-06).  This
#: round is a descriptive benchmark of an already-exposed split.
TEST_ROLE = "EXPOSED_TEST_PERFORMANCE_BENCHMARK"

#: One-shot, pre-registered coefficient calibration (PROTOCOL.md section 4).
CALIBRATION = {
    "n_batches": 16,
    "epoch_index": 0,
    "batch_index_rule": "floor(k * n_batches_epoch / 16) for k = 0..15 on epoch 1",
    "factor": 0.25,
    "lambda_min": 1e-4,
    "lambda_max": 0.1,
    "min_valid_batches": 8,
    "checkpoint": str(W2["checkpoint_best"]),
    "checkpoint_sha256": W2["checkpoint_best_sha256"],
    "gradient_norm": "L2 norm over every entry of d loss / d logits (float64 reduction of float32 gradients)",
    "model_mode": "eval (dropout off, no model update, no RNG draw)",
}

#: Pre-registered engineering target (resource decision, not a significance test).
WORK_TARGET = {
    "test_ap_min": 0.256616,
    "high_support_min_test_positives": 20,
    "high_support_tasks_historical": 108,
    "high_support_mean_delta_min": 0.002,
    "high_support_improved_min": 60,
}

BUDGET = {"gpu_target_hours": 2.5, "gpu_hard_hours": 4.0, "max_concurrent_gpus": 1}


# ---------------------------------------------------------------------------
# IO / identity helpers (inherited)
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


def sha256_int64(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array, dtype=np.int64).tobytes())


def state_sha256(state: Mapping[str, Any]) -> str:
    """Per-tensor identity over the whole state dict (key + dtype + shape + bytes)."""
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("utf-8"))
        h.update(str(tuple(value.shape)).encode("utf-8"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def relative_or_absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


# ---------------------------------------------------------------------------
# data gate
# ---------------------------------------------------------------------------


def load_task_gated() -> Any:
    """The frozen task, with the round's own isolation checks.

    ``bs_common.SD.load_task`` already verifies the feature-spec hash, the
    Train-only normaliser fingerprint, the split partition and the label
    artifact, and refuses a finite official Test label.  The explicit checks
    below are the round's independent restatement of that gate.
    """
    task = BC.SD.load_task()
    checks = {
        "feature_spec_hash": task.spec.hash() == SOURCE["feature_spec_hash"],
        "norm_fingerprint": task.norm["norm_fingerprint"] == SOURCE["norm_fingerprint"],
        "split_partition_sha256": task.split_partition_sha256 == SOURCE["split_partition_sha256"],
        "label_artifact_sha256": task.label_artifact_sha256 == SOURCE["label_artifact_sha256"],
        "sizes": (
            int(task.train_rows.size),
            int(task.valid_rows.size),
            int(task.test_rows.size),
        )
        == (N_TRAIN, N_VALID, N_TEST),
        "test_labels_masked": not bool(np.isfinite(task.labels[task.test_rows]).any()),
        "no_split_overlap": (
            np.intersect1d(task.train_rows, task.valid_rows).size == 0
            and np.intersect1d(task.train_rows, task.test_rows).size == 0
            and np.intersect1d(task.valid_rows, task.test_rows).size == 0
        ),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"frozen data identity drifted: {failed}")
    return task


def make_plan40(task: Any, *, seed: int = SEED) -> list[dict[str, Any]]:
    plan = BC.SD.make_plan40(task, seed=int(seed))
    if plan[0]["global_plan_sha256"] != GLOBAL_PLAN_SHA256:
        raise RuntimeError(
            f"order plan {plan[0]['global_plan_sha256']} != frozen {GLOBAL_PLAN_SHA256}"
        )
    return plan


def plan_meta(plan: Sequence[Mapping[str, Any]], consumed: int, order_seed: int) -> dict[str, Any]:
    prefix = [p["row_order_sha256"] for p in plan[: int(consumed)]]
    return {
        "global_plan_sha256": plan[0]["global_plan_sha256"],
        "order_seed": int(order_seed),
        "batch_size": int(BATCH_SIZE),
        "epochs": int(len(plan)),
        "consumed_epochs": int(consumed),
        "prefix_sha256": sha256_bytes("|".join(prefix).encode()),
    }


def calibration_batch_indices(plan: Sequence[Mapping[str, Any]]) -> list[int]:
    n = int(plan[0]["n_batches"])
    count = int(CALIBRATION["n_batches"])
    indices = [(k * n) // count for k in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError(f"calibration batch indices are not distinct: {indices}")
    return indices


def calibration_batches(
    task: Any, plan: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The frozen 16 Train batches, with row hashes and an isolation check."""
    train_rows = np.asarray(task.train_rows, dtype=np.int64)
    train_set = np.unique(train_rows)
    forbidden = np.union1d(
        np.asarray(task.valid_rows, dtype=np.int64), np.asarray(task.test_rows, dtype=np.int64)
    )
    out: list[dict[str, Any]] = []
    for index in calibration_batch_indices(plan):
        rows = np.asarray(plan[0]["batches"][index], dtype=np.int64)
        if np.setdiff1d(rows, train_set).size:
            raise RuntimeError(f"calibration batch {index} contains a non-Train row")
        if np.intersect1d(rows, forbidden).size:
            raise RuntimeError(f"calibration batch {index} contains a Valid/Test row")
        out.append(
            {
                "batch_index": int(index),
                "rows": rows,
                "rows_sha256": sha256_int64(rows),
                "rows_sha256_float32_order": sha256_bytes(
                    np.ascontiguousarray(rows, dtype=np.int64).tobytes()
                ),
                "n_rows": int(rows.size),
            }
        )
    combined = sha256_bytes("|".join(b["rows_sha256"] for b in out).encode())
    if combined != sha256_bytes("|".join(b["rows_sha256"] for b in out).encode()):
        raise RuntimeError("calibration batch identity is not deterministic")
    for b in out:
        b["combined_rows_sha256"] = combined
    return out


# ---------------------------------------------------------------------------
# model / recipe helpers (inherited, never re-implemented)
# ---------------------------------------------------------------------------


def build_w2(task: Any, *, device: Any = "cpu") -> tuple[Any, dict[str, Any]]:
    model, audit = BSM.build_model(
        MODEL_KEY, task.spec, task.norm, N_TASKS, seed=SEED, device=device
    )
    if int(audit["total_parameters"]) != MODEL_PARAMETERS:
        raise RuntimeError(f"W2 parameters {audit['total_parameters']} != {MODEL_PARAMETERS}")
    if int(parameter_count(model)) != MODEL_PARAMETERS:
        raise RuntimeError("W2 parameter count differs from the recorded audit")
    return model, audit


def make_optimizer(model: Any) -> tuple[Any, dict[str, Any]]:
    return BSM.make_optimizer(model)


def set_lrs(optimizer: Any, *, step: int) -> float:
    return BSM.set_lrs(optimizer, step=int(step))


def state_cpu(model: Any) -> dict[str, Any]:
    return BSM.state_cpu(model)


def optimizer_state_cpu(optimizer: Any) -> dict[str, Any]:
    return BSM.optimizer_state_cpu(optimizer)


def restore_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return BSM.restore_rng_state(state)


def lr_audit() -> dict[str, Any]:
    return BC.lr_audit()


__all__ = [
    "ANALYSIS",
    "BATCH_SIZE",
    "BC",
    "BSM",
    "BUDGET",
    "CALIBRATION",
    "DATASET",
    "GLOBAL_PLAN_SHA256",
    "MET",
    "MODEL_KEY",
    "MODEL_PARAMETERS",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCHEMA_CALIBRATION",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_PREFLIGHT",
    "SCHEMA_RUN",
    "SCHEMA_SELECTION",
    "SCHEMA_TEST_RESULTS",
    "SEED",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TEST_ROLE",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TRACK",
    "W2",
    "WORK_TARGET",
    "build_w2",
    "calibration_batch_indices",
    "calibration_batches",
    "capture_rng_state",
    "environment",
    "git_commit",
    "git_state",
    "gpu_occupancy",
    "gpu_peak",
    "host_facts",
    "load_task_gated",
    "log",
    "lr_audit",
    "make_optimizer",
    "make_plan40",
    "optimizer_state_cpu",
    "parameter_count",
    "plan_meta",
    "read_json",
    "relative_or_absolute",
    "restore_rng_state",
    "rng_state_hashes",
    "seed_everything",
    "set_lrs",
    "sha256_bytes",
    "sha256_file",
    "sha256_int64",
    "state_cpu",
    "state_sha256",
    "track_rel",
    "utcnow",
    "write_json",
]
