"""Checkpoint payloads and strict continuation restore for no-MP runs."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import torch

import nmp_common as C
import nmp_model as M


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def save_checkpoint(payload: Mapping[str, Any], path: Path | str) -> str:
    return _atomic_torch_save(payload, Path(path))


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint {path} is not a dict")
    payload["_path"] = C.track_rel(path)
    return payload


def scheduler_state(global_step: int, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "kind": "scratch_warmup_cosine_40epoch",
        "global_step": int(global_step),
        "total_steps_40_epochs": int(C.TOTAL_STEPS_40),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        "audit": C.lr_audit(),
    }


def checkpoint_payload(
    *,
    phase: str,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    valid_ap: float | None,
    valid_report: Mapping[str, Any] | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "phase": phase,
        "arm": arm,
        "arm_name": C.ARMS[arm]["name"],
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": C.N_TASKS,
        "variant": "C1 + " + C.ARMS[arm]["mode"],
        "state_dict": M.state_cpu(model),
        "optimizer_state_dict": M.optimizer_state_cpu(optimizer),
        "optimizer": "AdamW",
        "scheduler_state": scheduler_state(global_step, optimizer),
        "rng_state": rng_state,
        "rng_state_hashes": C.rng_state_hashes(rng_state),
        "batch_plan": dict(plan),
        "model_audit": dict(model_audit),
        "feature_spec_hash": C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
        "cache_sha256_manifest": C.SOURCE["cache_sha256"],
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "valid_report_summary": None
        if valid_report is None
        else {
            "ap": float(valid_report["ap"]),
            "valid_bce": float(valid_report["valid_bce"]),
            "n_tasks_scored": int(valid_report["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(valid_report["n_tasks_with_observed_labels"]),
            "valid_logits_sha256": valid_report.get("valid_logits_sha256"),
            "valid_probs_sha256": valid_report.get("valid_probs_sha256"),
        },
        "recipe": C.recipe_for_arm(arm, phase),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def restore_continuation(
    model: M.NMPModel,
    optimizer: torch.optim.AdamW,
    payload: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if payload.get("arm") != arm:
        raise RuntimeError(f"checkpoint arm {payload.get('arm')} != requested {arm}")
    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(f"checkpoint seed {payload.get('seed')} != requested {seed}")
    if payload.get("phase") != "B":
        raise RuntimeError(f"continuation source phase {payload.get('phase')} != 'B'")
    if int(payload.get("epoch", -1)) != 20:
        raise RuntimeError(f"continuation epoch {payload.get('epoch')} != 20")
    expected_step = 20 * C.STEPS_PER_EPOCH
    if int(payload.get("global_step", -1)) != expected_step:
        raise RuntimeError(f"continuation global_step {payload.get('global_step')} != {expected_step}")

    model.load_state_dict(payload["state_dict"], strict=True)
    saved_opt = payload["optimizer_state_dict"]
    if len(saved_opt.get("param_groups", [])) != len(optimizer.param_groups):
        raise RuntimeError("continuation optimizer groups differ from the rebuilt optimizer")
    optimizer.load_state_dict(copy.deepcopy(dict(saved_opt)))
    loaded = M.optimizer_state_cpu(optimizer)
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in loaded["state"].values()
        }
    )
    if steps != [float(expected_step)]:
        raise RuntimeError(f"restored AdamW steps {steps} != {expected_step}")
    if len(loaded["param_groups"]) != 1:
        raise RuntimeError("continuation optimizer is not a single parameter group")
    sched = payload["scheduler_state"]
    if int(sched["global_step"]) != expected_step:
        raise RuntimeError("scheduler state global_step mismatch")
    if not (0.0 < C.scheduled_lr(int(sched["global_step"])) <= float(C.SCRATCH_RECIPE["learning_rate"])):
        raise RuntimeError("scheduler state does not map to an in-range learning rate")
    saved_plan = payload.get("batch_plan") or {}
    return {
        "resumed_from": payload.get("_path", ""),
        "source_phase": payload.get("phase"),
        "source_epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "scheduler_state": sched,
        "optimizer_state_restored": True,
        "optimizer_step_values": steps,
        "batch_plan": saved_plan,
        "rng_state_hashes": payload.get("rng_state_hashes"),
    }


__all__ = [
    "checkpoint_payload",
    "load_checkpoint",
    "restore_continuation",
    "save_checkpoint",
    "scheduler_state",
]
