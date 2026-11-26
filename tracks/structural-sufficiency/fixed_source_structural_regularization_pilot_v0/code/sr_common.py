"""Round ``fixed-source-structural-regularization-pilot-v0`` (FSRP v0).

FIXED1 structural-unit message dropout (``STRUCT_DROP``) versus a plain
continuation (``BASE_CONT``) from the same frozen FIXED1 seed-0 epoch-25 state.

Both arms are the *original* FIXED1 computation (inherited read-only from
``direct_structural_computation_recurrence_v0``); the only difference is a
per-training-batch Bernoulli mask on the directed message term of both
structural blocks, shared per physical bond.

This module only holds round identities, the frozen base-checkpoint loader,
checkpoint IO and small finger-print helpers.  No arithmetic is redefined here.
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
BASE_ROUND = TRACK / "direct_structural_computation_recurrence_v0"
BASE_CODE = BASE_ROUND / "code"

for _p in (str(CODE_DIR), str(BASE_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ds_common as C  # noqa: E402  (frozen definitions / data / evaluation)
import ds_model as DM  # noqa: E402  (frozen FIXED1 model)
import ds_train as DT  # noqa: E402  (frozen optimizer / restore / train step)

ROUND_ID = "fixed-source-structural-regularization-pilot-v0"
PROTOCOL_ID = "fsrp-v0"
SCHEMA_RUN = "fsrp-run-v1"
SCHEMA_CHECKPOINT = "fsrp-checkpoint-v1"
SCHEMA_CHECKS = "fsrp-checks-v1"
SCHEMA_PREFLIGHT = "fsrp-preflight-v1"
SCHEMA_SUMMARY = "fsrp-summary-v1"
SCHEMA_MANIFEST = "fsrp-checkpoint-manifest-v1"

ANALYSIS = ROUND / "analysis"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "runs"
RESULTS = ROUND / "results"
TESTS = ROUND / "tests"

SEED = 0
ORDER_SEED = 0
START_EPOCH = 25
END_EPOCH = 33
TRAIN_EPOCHS = tuple(range(START_EPOCH + 1, END_EPOCH + 1))  # 26..33
TRAIN_EVAL_EPOCHS = (START_EPOCH, 29, 33)
HORIZON = END_EPOCH
Q = 0.10
MASK_SEED = 20261126
ARMS = ("BASE_CONT", "STRUCT_DROP")
ARM_ROLE = {
    "BASE_CONT": "original FIXED1 continuation, no training constraint",
    "STRUCT_DROP": "FIXED1 + structural-unit message dropout (q=0.10) during training only",
}

BASE_CHECKPOINT = BASE_ROUND / "runs/FIXED1_seed0/checkpoint_last.pt"
BASE_CHECKPOINT_FALLBACKS = (
    BASE_ROUND / "runs/FIXED1_seed0/checkpoint_epoch25.pt",
    BASE_ROUND / "runs/FIXED1_seed0/checkpoint_best.pt",
)
EXPECTED_BASE_EPOCH = 25
EXPECTED_BASE_STEP = 68450
EXPECTED_BASE_VALID_AP = 0.27584552136653284
EXPECTED_BASE_HORIZON = 25
PARAMS_TOTAL = 1_128_468
EXPECTED_LR_AT_RESTORE = 0.00010358628017094336  # scheduler lr_now at completed step 68450 (used up to step 68449)
EXPECTED_LR_FIRST_STEP = 0.00010358229137883232  # frozen 40-epoch cosine at step 68450


# ---------------------------------------------------------------------------
# small helpers
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
    """Order-independent fingerprint of nested tensor/dict/list state."""
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


def arm_run_dir(arm: str) -> Path:
    if arm not in ARMS:
        raise KeyError(arm)
    return RUNS / f"{arm}_seed{SEED}"


# ---------------------------------------------------------------------------
# frozen base checkpoint (read only)
# ---------------------------------------------------------------------------


def resolve_base_checkpoint(explicit: str | Path | None = None) -> Path:
    candidates = ([Path(explicit)] if explicit else []) + [BASE_CHECKPOINT, *BASE_CHECKPOINT_FALLBACKS]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "no FIXED1 epoch-25 checkpoint found; checked: " + ", ".join(str(p) for p in candidates)
    )


def verify_base_payload(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Hard gate: the inherited state really is the FIXED1 seed-0 epoch-25 state."""
    errors: list[str] = []
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        errors.append(f"schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if str(payload.get("round_id")) != C.ROUND_ID:
        errors.append(f"round_id {payload.get('round_id')} != {C.ROUND_ID}")
    if str(payload.get("arm")) != "FIXED1":
        errors.append(f"arm {payload.get('arm')} != FIXED1")
    if int(payload.get("seed", -1)) != SEED:
        errors.append(f"seed {payload.get('seed')} != {SEED}")
    if int(payload.get("epoch", -1)) != EXPECTED_BASE_EPOCH:
        errors.append(f"epoch {payload.get('epoch')} != {EXPECTED_BASE_EPOCH}")
    if int(payload.get("global_step", -1)) != EXPECTED_BASE_STEP:
        errors.append(f"global_step {payload.get('global_step')} != {EXPECTED_BASE_STEP}")
    if int(payload.get("horizon", -1)) != EXPECTED_BASE_HORIZON:
        errors.append(f"horizon {payload.get('horizon')} != {EXPECTED_BASE_HORIZON}")
    if int(payload.get("parameters", -1)) != PARAMS_TOTAL:
        errors.append(f"parameters {payload.get('parameters')} != {PARAMS_TOTAL}")
    if abs(float(payload.get("valid_ap", -1.0)) - EXPECTED_BASE_VALID_AP) > 1e-9:
        errors.append(f"valid_ap {payload.get('valid_ap')} != {EXPECTED_BASE_VALID_AP}")
    if "optimizer_state_dict" not in payload or "rng_state" not in payload:
        errors.append("optimizer/RNG state missing")
    if not payload.get("state_dict"):
        errors.append("state_dict missing")
    plan = payload.get("batch_plan") or {}
    if int(plan.get("consumed_epochs", -1)) != EXPECTED_BASE_EPOCH:
        errors.append(f"plan consumed_epochs {plan.get('consumed_epochs')} != {EXPECTED_BASE_EPOCH}")
    sched = payload.get("scheduler_state") or {}
    if int(sched.get("global_step", -1)) != EXPECTED_BASE_STEP:
        errors.append(f"scheduler global_step {sched.get('global_step')} != {EXPECTED_BASE_STEP}")
    if errors:
        raise RuntimeError(f"base checkpoint {path} failed verification: {errors}")
    return {
        "path": C.track_rel(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "schema": payload["schema"],
        "round_id": payload["round_id"],
        "arm": payload["arm"],
        "seed": int(payload["seed"]),
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "horizon": int(payload["horizon"]),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "valid_ap": float(payload["valid_ap"]),
        "parameters": int(payload["parameters"]),
        "optimizer_state_present": True,
        "rng_state_present": True,
        "rng_state_hashes": payload.get("rng_state_hashes"),
        "batch_plan": dict(plan),
        "scheduler_state": dict(sched),
        "state_dict_fingerprint": C.state_fingerprint(payload["state_dict"]),
        "optimizer_fingerprint": nested_fingerprint(payload["optimizer_state_dict"]),
        "test_role_recorded": payload.get("test_role"),
        "test_used_for_selection_recorded": payload.get("test_used_for_selection"),
    }


def load_base_checkpoint(explicit: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = resolve_base_checkpoint(explicit)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    audit = verify_base_payload(payload, path)
    return payload, audit, path


# ---------------------------------------------------------------------------
# order plan / schedule gates
# ---------------------------------------------------------------------------


def load_task_without_test() -> Any:
    """Official PCBA task; the frozen loader refuses exposed Test labels."""
    task = C.SD.load_task()
    return task


def plan_and_identity(task: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan, audit = C.plan_for_seed(task, ORDER_SEED)
    return plan, audit


def verify_plan_prefix(payload: Mapping[str, Any], plan: list[dict[str, Any]], consumed: int) -> dict[str, Any]:
    saved = payload.get("batch_plan") or {}
    now = C.plan_meta(plan, int(consumed), ORDER_SEED)
    if now["global_plan_sha256"] != saved.get("global_plan_sha256"):
        raise RuntimeError("resumed global plan hash changed")
    if now["prefix_sha256"] != saved.get("prefix_sha256"):
        raise RuntimeError("resumed consumed-prefix hash changed")
    return {"plan_meta": now, "matches_checkpoint": True}


def current_lr_at(optimizer: torch.optim.Optimizer, global_step: int) -> float:
    C.set_lrs(optimizer, step=int(global_step))
    return float(optimizer.param_groups[0]["lr"])


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def checkpoint_payload(
    *,
    arm: str,
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
        "arm": str(arm),
        "seed": SEED,
        "order_seed": ORDER_SEED,
        "horizon": HORIZON,
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
        "feature_spec_hash": C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
        "cache_sha256_manifest": C.SOURCE["cache_sha256"],
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "selection_metric": "full official Valid macro AP (official OGB Evaluator)",
        "recipe": C.recipe("FIXED1", SEED, HORIZON),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "created_utc": utcnow(),
        "test_role": "NO_TEST_ACCESS_IN_THIS_ROUND",
        "test_used_for_selection": False,
        "test_scored": False,
    }
    payload["recipe"]["round"] = ROUND_ID
    payload["recipe"]["arm"] = str(arm)
    payload["recipe"]["structural_dropout"] = (
        None if arm == "BASE_CONT" else {"q": Q, "scope": "directed message term, both blocks, shared per physical bond"}
    )
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


def load_json_or(path: Path, default: Any) -> Any:
    if not Path(path).is_file():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


__all__ = [
    "ANALYSIS",
    "ARMS",
    "ARM_ROLE",
    "BASE_CHECKPOINT",
    "BASE_CHECKPOINT_FALLBACKS",
    "BASE_ROUND",
    "CONFIGS",
    "END_EPOCH",
    "EXPECTED_BASE_EPOCH",
    "EXPECTED_BASE_STEP",
    "EXPECTED_BASE_VALID_AP",
    "EXPECTED_LR_AT_RESTORE",
    "EXPECTED_LR_FIRST_STEP",
    "HORIZON",
    "MASK_SEED",
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
    "START_EPOCH",
    "TRAIN_EPOCHS",
    "TRAIN_EVAL_EPOCHS",
    "TRACK",
    "TESTS",
    "arm_run_dir",
    "atomic_save",
    "checkpoint_payload",
    "current_lr_at",
    "load_base_checkpoint",
    "load_json_or",
    "load_task_without_test",
    "model_state_fingerprint",
    "nested_fingerprint",
    "optimizer_fingerprint",
    "plan_and_identity",
    "resolve_base_checkpoint",
    "sha256_file",
    "utcnow",
    "verify_base_payload",
    "verify_plan_prefix",
    "write_json",
    "C",
    "DM",
    "DT",
]
