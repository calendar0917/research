"""Round ``fixed-source-structural-regularization-scratch-v1`` (FSRP-SCRATCH v1).

One arm only: ``STRUCT_DROP_SCRATCH``.

The *unmodified* DSCR FIXED1 architecture (inherited read-only) trained from the
official FIXED1 seed-0 random initialisation for the first 25 epochs of the
frozen 40-epoch recipe, with structural-unit message dropout active from the
very first training step (q = 0.10, one Bernoulli draw per real physical
undirected bond per training batch, independent mask RNG, seed 20261126).

Nothing new is designed here.  The structural mask arithmetic is inherited
verbatim from
``fixed_source_structural_regularization_pilot_v0/code/sr_model.py``; the model,
the optimizer, the order plan, the LR schedule and the evaluation protocol are
inherited from ``direct_structural_computation_recurrence_v0``.

Test is never loaded, scored or referenced by any decision in this round.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

DSCR_ROUND = TRACK / "direct_structural_computation_recurrence_v0"
DSCR_CODE = DSCR_ROUND / "code"
FSRP_ROUND = TRACK / "fixed_source_structural_regularization_pilot_v0"
FSRP_CODE = FSRP_ROUND / "code"

for _p in (str(DSCR_CODE), str(FSRP_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ds_common as C  # noqa: E402  (frozen data / evaluation / recipe)
import ds_model as DM  # noqa: E402  (frozen FIXED1 architecture + official constructor)
import ds_train as DT  # noqa: E402  (frozen optimizer / train step / watchdog)
import sr_model as SM  # noqa: E402  (inherited STRUCT_DROP arithmetic)

ROUND_ID = "fixed-source-structural-regularization-scratch-v1"
PROTOCOL_ID = "fsrp-scratch-v1"
ARM = "STRUCT_DROP_SCRATCH"
SCHEMA_RUN = "fsrp-scratch-run-v1"
SCHEMA_CHECKPOINT = "fsrp-scratch-checkpoint-v1"
SCHEMA_CHECKS = "fsrp-scratch-checks-v1"
SCHEMA_PREFLIGHT = "fsrp-scratch-preflight-v1"
SCHEMA_SUMMARY = "fsrp-scratch-summary-v1"
SCHEMA_MANIFEST = "fsrp-scratch-manifest-v1"

ANALYSIS = ROUND / "analysis"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "runs"
RESULTS = ROUND / "results"
TESTS = ROUND / "tests"

SEED = 0
ORDER_SEED = 0
EPOCHS = 25                      # first 25 epochs of the frozen 40-epoch recipe
TRAIN_EVAL_EPOCHS = (12, 20, 25)  # complete official Train AP/BCE
STEPS_PER_EPOCH = int(C.STEPS_PER_EPOCH)
TOTAL_STEPS_40 = int(C.TOTAL_STEPS_40)
Q = 0.10
MASK_SEED = 20261126
PARAMS_TOTAL = 1_128_468

#: Hard gates inherited from the historical DSCR FIXED1 seed-0 run.
EXPECTED_INITIAL_STATE_SHA256 = "3d313189bd322eb6c2b1c9075851b3b5f31b7c2e82318c5b4217319d4bd50e5c"
EXPECTED_NEW_STATE_SHA256 = "f7c24ee48bc71c9c7d1d958bac6b249b583762c97b11ca07af21a62fe5441177"
EXPECTED_W2_TRUNK_SHA256 = "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d"
EXPECTED_PLAN_GLOBAL_SHA256 = "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6"
EXPECTED_PLAN_PREFIX25_SHA256 = "8c857251a58b4534f287338052fa0be259cf01f341da81d55716a3c2a6b29b40"
N_STATE_TENSORS = 71
N_PARAMETER_TENSORS = 65

#: Frozen schedule endpoints (``scheduled_lr``, phase 3).
EXPECTED_LR_FIRST_STEP = float(C.scheduled_lr(0, phase=3, kind="all"))
EXPECTED_LR_LAST_STEP = float(C.scheduled_lr((EPOCHS - 1) * STEPS_PER_EPOCH + STEPS_PER_EPOCH - 1, phase=3, kind="all"))

#: Historical FIXED1 seed-0 25-epoch prefix (read-only; never retrained here).
HIST_DIR = DSCR_ROUND / "results/runs/FIXED1_seed0"
HIST_CURVE = HIST_DIR / "train_curve.csv"
HIST_PER_TASK = HIST_DIR / "per_task_ap.csv"
HIST_BEST_EVAL = HIST_DIR / "best_checkpoint_eval.json"
HIST_MAIN_TABLE = DSCR_ROUND / "analysis/main_table.csv"
HIST = {
    "round_id": "direct-structural-computation-recurrence-v0",
    "arm": "FIXED1",
    "seed": 0,
    "horizon": 25,
    "parameters": PARAMS_TOTAL,
    "best_epoch": 25,
    "best_valid_ap": 0.27584697853384577,      # best-checkpoint re-score
    "best_valid_bce": 0.04637687617786432,
    "best_train_ap": 0.5995324907052489,
    "best_train_bce": 0.024222211976071915,
    "endpoint_valid_ap": 0.27584552136653284,
    "endpoint_valid_bce": 0.04637687616613202,
    "endpoint_train_ap": 0.5995356788033605,
    "endpoint_train_bce": 0.024222211982811284,
    "last5_mean_valid_ap": 0.2733584884518717,
    "high_support_valid_ap": 0.298486642,
    "high_support_tasks": 107,
    "train_ap_e12": 0.4479189031400282,
    "train_ap_e20": 0.5541792648992719,
    "train_ap_e25": 0.5995356788033605,
    "train_bce_e12": 0.029789790741576318,
    "train_bce_e20": 0.02589244308455447,
    "train_bce_e25": 0.024222211982811284,
    "valid_seconds_mean": 5.62,
    "epoch_train_seconds_mean": 120.65,
    "gpu_hours_full_eval_every_epoch": 1.230772778,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return C.utcnow()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    C.write_json(Path(path), dict(payload))


def read_json(path: Path) -> dict[str, Any]:
    return C.read_json(Path(path))


def sha256_file(path: Path) -> str:
    return C.sha256_file(Path(path))


def nested_fingerprint(obj: Any) -> str:
    import hashlib

    h = hashlib.sha256()

    def _walk(value: Any, key: str) -> None:
        h.update(key.encode())
        if torch.is_tensor(value):
            arr = value.detach().to("cpu").contiguous().numpy()
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        elif isinstance(value, Mapping):
            for k in sorted(value.keys(), key=lambda x: str(x)):
                _walk(value[k], f"{key}/{k}")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _walk(v, f"{key}/{i}")
        else:
            h.update(repr(value).encode())

    _walk(obj, "root")
    return h.hexdigest()


def model_state_fingerprint(model: torch.nn.Module) -> str:
    return C.state_fingerprint(C.state_cpu(model))


def optimizer_fingerprint(optimizer: torch.optim.Optimizer) -> str:
    return nested_fingerprint(C.optimizer_state_cpu(optimizer))


def arm_run_dir() -> Path:
    return RUNS / f"{ARM}_seed{SEED}"


def load_task_without_test() -> Any:
    """Official PCBA task; the frozen loader refuses exposed Test labels."""
    return C.SD.load_task()


def plan_and_identity(task: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan, audit = C.plan_for_seed(task, ORDER_SEED)
    return plan, audit


def verify_plan_identity(plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Both the 40-epoch global plan and the consumed 25-epoch prefix must match FIXED1."""
    meta = C.plan_meta(plan, EPOCHS, ORDER_SEED)
    if meta["global_plan_sha256"] != EXPECTED_PLAN_GLOBAL_SHA256:
        raise RuntimeError(f"global plan {meta['global_plan_sha256']} != FIXED1 {EXPECTED_PLAN_GLOBAL_SHA256}")
    if meta["prefix_sha256"] != EXPECTED_PLAN_PREFIX25_SHA256:
        raise RuntimeError(f"25-epoch prefix {meta['prefix_sha256']} != FIXED1 {EXPECTED_PLAN_PREFIX25_SHA256}")
    if int(meta["batch_size"]) != int(C.BATCH_SIZE):
        raise RuntimeError(f"plan batch size {meta['batch_size']} != {C.BATCH_SIZE}")
    if len(plan[0]["batches"]) != STEPS_PER_EPOCH:
        raise RuntimeError(f"steps per epoch {len(plan[0]['batches'])} != {STEPS_PER_EPOCH}")
    return {"plan_meta": meta, "matches_historical_fixed1": True}


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mask_rng_state: Any,
    plan: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    valid_ap: float | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA_CHECKPOINT,
        "round_id": ROUND_ID,
        "protocol_id": PROTOCOL_ID,
        "arm": ARM,
        "seed": SEED,
        "order_seed": ORDER_SEED,
        "horizon": EPOCHS,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": str(kind),
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "state_dict": C.state_cpu(model),
        "optimizer_state_dict": C.optimizer_state_cpu(optimizer),
        "optimizer": "AdamW",
        "scheduler_state": DT.scheduler_state(global_step, optimizer),
        "rng_state": C.capture_rng_state(),
        "rng_state_hashes": C.rng_state_hashes(C.capture_rng_state()),
        "mask_rng_state": mask_rng_state,
        "mask_rng_fingerprint": None if mask_rng_state is None else nested_fingerprint(mask_rng_state),
        "batch_plan": dict(plan),
        "model_audit": dict(model_audit),
        "initial_state_sha256": model_audit.get("initial_state_sha256"),
        "feature_spec_hash": C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
        "cache_sha256_manifest": C.SOURCE["cache_sha256"],
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "selection_metric": "full official Valid macro AP (official OGB Evaluator)",
        "recipe": C.recipe("FIXED1", SEED, EPOCHS),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "created_utc": utcnow(),
        "test_role": "NO_TEST_ACCESS_IN_THIS_ROUND",
        "test_used_for_selection": False,
        "test_scored": False,
    }
    payload["recipe"]["round"] = ROUND_ID
    payload["recipe"]["arm"] = ARM
    payload["recipe"]["structural_dropout"] = {
        "q": Q,
        "mask_seed": MASK_SEED,
        "scope": "directed message term, both FIXED1 blocks, shared per physical bond, from step 0",
    }
    if extra:
        payload.update(dict(extra))
    return payload


def atomic_save(payload: Mapping[str, Any], path: Path) -> str:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return sha256_file(path)


# ---------------------------------------------------------------------------
# historical FIXED1 record (read-only reader + consistency gate)
# ---------------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, fields: list[str], rows: list[Mapping[str, Any]]) -> str:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return sha256_file(path)


def load_historical_fixed1() -> dict[str, Any]:
    """Re-read the FIXED1 seed-0 25-epoch prefix from its own records and gate on HIST."""
    rows = read_csv_rows(HIST_CURVE)
    if len(rows) != EPOCHS:
        raise RuntimeError(f"historical curve has {len(rows)} epochs, expected {EPOCHS}")
    curve = {
        int(r["epoch"]): {
            "global_step": int(r["global_step"]),
            "learning_rate": float(r["learning_rate"]),
            "valid_ap": float(r["valid_ap"]),
            "valid_bce": float(r["valid_bce"]),
            "train_ap": float(r["train_ap"]),
            "train_bce": float(r["train_bce"]),
        }
        for r in rows
    }
    best = json.loads(HIST_BEST_EVAL.read_text())
    checks = {
        "n_epochs": len(curve) == EPOCHS,
        "final_global_step": curve[EPOCHS]["global_step"] == EPOCHS * STEPS_PER_EPOCH,
        "best_epoch": int(best["epoch"]) == HIST["best_epoch"],
        "best_valid_ap": abs(float(best["valid"]["ap"]) - HIST["best_valid_ap"]) < 1e-12,
        "best_train_ap": abs(float(best["train"]["ap"]) - HIST["best_train_ap"]) < 1e-12,
        "endpoint_valid_ap": abs(curve[EPOCHS]["valid_ap"] - HIST["endpoint_valid_ap"]) < 1e-12,
        "last5_mean": abs(
            float(np.mean([curve[e]["valid_ap"] for e in range(EPOCHS - 4, EPOCHS + 1)])) - HIST["last5_mean_valid_ap"]
        ) < 1e-9,
        "train_ap_e12": abs(curve[12]["train_ap"] - HIST["train_ap_e12"]) < 1e-12,
        "train_ap_e20": abs(curve[20]["train_ap"] - HIST["train_ap_e20"]) < 1e-12,
        "train_ap_e25": abs(curve[25]["train_ap"] - HIST["train_ap_e25"]) < 1e-12,
        "lr_first": abs(curve[1]["learning_rate"] - float(C.scheduled_lr(STEPS_PER_EPOCH - 1, phase=3, kind="all"))) < 1e-12,
        "lr_last": abs(curve[EPOCHS]["learning_rate"] - EXPECTED_LR_LAST_STEP) < 1e-12,
    }
    if not all(checks.values()):
        raise RuntimeError(f"historical FIXED1 record failed verification: {checks}")
    per_task = read_csv_rows(HIST_PER_TASK)
    valid = {int(r["task"]): float(r["ap"]) for r in per_task if r["split"] == "valid" and int(r["epoch"]) == EPOCHS}
    train = {int(r["task"]): float(r["ap"]) for r in per_task if r["split"] == "train" and int(r["epoch"]) == EPOCHS}
    for t, v in enumerate(best["valid"]["per_task_ap"]):
        if v is not None:
            valid[int(t)] = float(v)
    for t, v in enumerate(best["train"]["per_task_ap"]):
        if v is not None:
            train[int(t)] = float(v)
    return {
        "curve": curve,
        "best": best,
        "valid_per_task": valid,
        "train_per_task": train,
        "positive": [int(x) for x in best["valid"]["positive"]],
        "negative": [int(x) for x in best["valid"]["negative"]],
        "verification": checks,
        "sources": {
            "curve": C.track_rel(HIST_CURVE),
            "curve_sha256": sha256_file(HIST_CURVE),
            "best_checkpoint_eval": C.track_rel(HIST_BEST_EVAL),
            "best_checkpoint_eval_sha256": sha256_file(HIST_BEST_EVAL),
            "per_task_ap": C.track_rel(HIST_PER_TASK),
            "per_task_ap_sha256": sha256_file(HIST_PER_TASK),
            "main_table": C.track_rel(HIST_MAIN_TABLE),
            "main_table_sha256": sha256_file(HIST_MAIN_TABLE),
        },
    }


__all__ = [
    "ANALYSIS",
    "ARM",
    "CONFIGS",
    "C",
    "DM",
    "DT",
    "EPOCHS",
    "EXPECTED_INITIAL_STATE_SHA256",
    "EXPECTED_LR_FIRST_STEP",
    "EXPECTED_LR_LAST_STEP",
    "EXPECTED_NEW_STATE_SHA256",
    "EXPECTED_PLAN_GLOBAL_SHA256",
    "EXPECTED_PLAN_PREFIX25_SHA256",
    "EXPECTED_W2_TRUNK_SHA256",
    "HIST",
    "HIST_BEST_EVAL",
    "HIST_CURVE",
    "HIST_MAIN_TABLE",
    "HIST_PER_TASK",
    "MASK_SEED",
    "N_PARAMETER_TENSORS",
    "N_STATE_TENSORS",
    "ORDER_SEED",
    "PARAMS_TOTAL",
    "PROTOCOL_ID",
    "Q",
    "REPO",
    "RESULTS",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SEED",
    "SM",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TOTAL_STEPS_40",
    "TRAIN_EVAL_EPOCHS",
    "TRACK",
    "arm_run_dir",
    "atomic_save",
    "checkpoint_payload",
    "load_historical_fixed1",
    "load_task_without_test",
    "model_state_fingerprint",
    "nested_fingerprint",
    "optimizer_fingerprint",
    "plan_and_identity",
    "read_csv_rows",
    "sha256_file",
    "utcnow",
    "verify_plan_identity",
    "write_csv",
    "write_json",
]
