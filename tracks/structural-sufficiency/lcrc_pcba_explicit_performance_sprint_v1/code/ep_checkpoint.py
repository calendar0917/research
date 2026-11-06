"""Checkpoint payloads for the explicit-performance sprint.

The epoch-40 checkpoint keeps the complete optimizer state, RNG state, batch
plan identity, feature-cache hashes and the model audit so an infrastructure
failure can be recovered without resetting the optimizer or re-initialising.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

import ep_common as C


def save_checkpoint(payload: Mapping[str, Any], path: Path | str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint {path} is not a dict")
    payload["_path"] = C.track_rel(path)
    return payload


def state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    import copy

    state = optimizer.state_dict()
    out: dict[str, Any] = {"state": {}, "param_groups": copy.deepcopy(state["param_groups"])}
    for param_id, values in state["state"].items():
        out["state"][int(param_id)] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in values.items()
        }
    return out


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
    arm: str,
    modules: tuple[str, ...],
    seed: int,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    features: Mapping[str, Any],
    valid_ap: float | None,
    valid_report: Mapping[str, Any] | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "arm": arm,
        "arm_label": C.arm_label(arm, modules),
        "modules": list(modules),
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "state_dict": state_cpu(model),
        "optimizer_state_dict": optimizer_state_cpu(optimizer),
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
        "feature_hashes": dict(features),
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
        "recipe": C.recipe_for(arm, modules),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def restore_for_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: Mapping[str, Any],
    *,
    arm: str,
    modules: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    """Infrastructure-failure recovery from a complete epoch-N checkpoint."""
    if payload.get("schema") != C.SCHEMA_CHECKPOINT or payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError("checkpoint schema/round mismatch")
    if tuple(payload.get("modules", ())) != tuple(modules) or payload.get("arm") != arm:
        raise RuntimeError("checkpoint arm/modules mismatch")
    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError("checkpoint seed mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(dict(payload["optimizer_state_dict"]))
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in optimizer.state_dict()["state"].values()
        }
    )
    return {
        "resumed_from": payload.get("_path", ""),
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "optimizer_steps": steps,
        "rng_state_hashes": payload.get("rng_state_hashes"),
    }


__all__ = [
    "checkpoint_payload",
    "load_checkpoint",
    "optimizer_state_cpu",
    "restore_for_resume",
    "save_checkpoint",
    "scheduler_state",
    "state_cpu",
]
