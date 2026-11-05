"""Warm-start checkpoint audit, continuation payloads and atomic saves."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import sg_common as C
import sg_model as M


def load_source_checkpoint(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path or C.SOURCE["checkpoint"])
    sha = C.sha256_file(path)
    if sha != C.SOURCE["checkpoint_sha256"]:
        raise RuntimeError(f"warm-start sha256 {sha} != frozen {C.SOURCE['checkpoint_sha256']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError("warm start is not a dict with state_dict")
    payload["_path"] = C.track_rel(path)
    return payload


def verify_source_metadata(payload: Mapping[str, Any], ref_model: torch.nn.Module) -> dict[str, Any]:
    recipe = payload.get("recipe") or {}
    checks = {
        "schema": payload.get("schema") == "lcrc-pcba-lr-probe-checkpoint-v1",
        "round": payload.get("round_id") == C.SOURCE["round"],
        "group": payload.get("group") == C.SOURCE["group"],
        "group_name": payload.get("group_name") == C.SOURCE["group_name"],
        "epoch": int(payload.get("epoch", -1)) == int(C.SOURCE["best_epoch"]),
        "valid_ap": abs(float(payload.get("valid_ap", float("nan"))) - C.AP0) <= 1e-12,
        "dataset": payload.get("dataset") == C.DATASET,
        "variant": payload.get("variant") == "C1",
        "n_outputs": int(payload.get("n_outputs", -1)) == C.N_TASKS,
        "feature_spec_hash": payload.get("feature_spec_hash") == C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": payload.get("norm_fingerprint") == C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": payload.get("split_partition_sha256") == C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": payload.get("label_artifact_sha256") == C.SOURCE["label_artifact_sha256"],
        "cache_sha256_manifest": payload.get("cache_sha256_manifest") == C.SOURCE["cache_sha256"],
        "optimizer_is_adamw": payload.get("optimizer") == "AdamW",
        "test_evaluated_false": payload.get("test_evaluated") is False,
        "has_optimizer_state": isinstance(payload.get("optimizer_state_dict"), dict),
        "has_rng_state": isinstance(payload.get("rng_state"), dict),
        "recipe_learning_rate": float(recipe.get("learning_rate", -1.0)) == 5.0e-5,
        "recipe_wd": float(recipe.get("weight_decay", -1.0)) == 1.0e-3,
        "recipe_betas": tuple(recipe.get("adam_betas", ())) == (0.9, 0.999),
        "recipe_eps": float(recipe.get("adam_eps", -1.0)) == 1.0e-8,
        "recipe_dropout": float(recipe.get("dropout", -1.0)) == 0.10,
        "recipe_batch_size": int(recipe.get("batch_size", -1)) == C.BATCH_SIZE,
        "recipe_clip": float(recipe.get("gradient_clip", -1.0)) == 5.0,
    }
    failed = {k: v for k, v in checks.items() if not v}
    if failed:
        raise RuntimeError(f"warm-start metadata mismatch: {failed}")

    state = payload["state_dict"]
    if len(state) != C.SOURCE["n_state_tensors"]:
        raise RuntimeError(f"source state_dict has {len(state)} tensors, expected {C.SOURCE['n_state_tensors']}")
    ref_state = ref_model.state_dict()
    if set(state) != set(ref_state):
        raise RuntimeError(
            f"source state keys differ from the frozen C1: only_source={sorted(set(state)-set(ref_state))} "
            f"only_ref={sorted(set(ref_state)-set(state))}"
        )
    shape_rows = []
    for key in sorted(state):
        got, want = state[key], ref_state[key]
        if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
            raise RuntimeError(
                f"source tensor {key} shape/dtype {tuple(got.shape)}/{got.dtype} != "
                f"frozen C1 {tuple(want.shape)}/{want.dtype}"
            )
        shape_rows.append({"name": key, "shape": list(got.shape), "dtype": str(got.dtype)})
    fp = M.state_fingerprint(state)
    init = payload.get("initialization") or {}
    # ``init`` describes the epoch-29 checkpoint this run was warm-started from;
    # these are the epoch-9 weights after nine training epochs, so only the
    # frozen-init metadata (parameter count / model shape) is comparable.
    if int(payload.get("n_outputs", 0)) != 128:
        raise RuntimeError("source checkpoint is not 128-task")
    if int(init.get("parameter_count", -1)) != C.SOURCE["n_parameters"]:
        raise RuntimeError("source parameter count mismatch")

    opt = payload["optimizer_state_dict"]
    groups = opt.get("param_groups") or []
    if len(groups) != 1 or list(groups[0].get("params") or []) != list(range(C.SOURCE["n_optimizer_params"])):
        raise RuntimeError("source AdamW state is not one group of parameters 0..30")
    if set(opt.get("state", {})) != set(range(C.SOURCE["n_optimizer_params"])):
        raise RuntimeError("source AdamW state does not cover exactly 31 parameters")
    params = [p for p in ref_model.parameters()]
    if len(params) != C.SOURCE["n_optimizer_params"]:
        raise RuntimeError(f"frozen C1 has {len(params)} parameters, expected 31")
    moment_rows = []
    for pid, param in enumerate(params):
        values = opt["state"][pid]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if key not in values:
                raise RuntimeError(f"source AdamW state[{pid}] missing {key}")
        for key in ("exp_avg", "exp_avg_sq"):
            if tuple(values[key].shape) != tuple(param.shape):
                raise RuntimeError(
                    f"source AdamW {key}[{pid}] shape {tuple(values[key].shape)} != parameter {tuple(param.shape)}"
                )
            if values[key].dtype != torch.float32:
                raise RuntimeError(f"source AdamW {key}[{pid}] is not FP32")
        step = values["step"]
        step_value = float(step.detach().cpu().item()) if torch.is_tensor(step) else float(step)
        if step_value != float(C.SOURCE["optimizer_step_per_param"]):
            raise RuntimeError(
                f"source AdamW step[{pid}]={step_value} != {C.SOURCE['optimizer_step_per_param']}"
            )
        moment_rows.append({"param": pid, "shape": list(param.shape), "step": step_value})
    rng = payload["rng_state"]
    for key in ("python_random", "numpy", "torch_cpu", "torch_cuda"):
        if key not in rng:
            raise RuntimeError(f"source RNG state missing {key}")
    if len(rng["torch_cuda"]) != 1:
        raise RuntimeError(f"source RNG has {len(rng['torch_cuda'])} CUDA states, expected 1")

    return {
        "source_checkpoint": payload["_path"],
        "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
        "source_epoch": int(payload["epoch"]),
        "source_valid_ap": float(payload["valid_ap"]),
        "metadata_checks": {k: bool(v) for k, v in checks.items()},
        "state_dict_tensors": int(len(state)),
        "state_dict_keys_shapes_match_frozen_c1": True,
        "state_dict_tensor_shapes": shape_rows,
        "state_dict_fingerprint": fp,
        "parameter_count": int(C.SOURCE["n_parameters"]),
        "n_outputs": int(payload["n_outputs"]),
        "adamw_param_groups": len(groups),
        "adamw_param_ids": list(groups[0]["params"]),
        "adamw_step_values": sorted({row["step"] for row in moment_rows}),
        "adamw_moments": moment_rows,
        "rng_cuda_states": int(len(rng["torch_cuda"])),
        "test_evaluated": False,
    }


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    return M.state_fingerprint(state)


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


def checkpoint_payload(
    *,
    phase: int,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    source_audit: Mapping[str, Any] | None,
    valid_ap: float | None,
    valid_report: Mapping[str, Any] | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = M.state_cpu(model)
    payload: dict[str, Any] = {
        "schema": "lcrc-pcba-structural-gain-checkpoint-v1",
        "round_id": C.ROUND_ID,
        "phase": int(phase),
        "arm": arm,
        "arm_name": C.ARMS[arm]["name"],
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": C.N_TASKS,
        "variant": "C1 + " + C.ARMS[arm]["blocks"],
        "state_dict": state,
        "optimizer_state_dict": M.optimizer_state_cpu(optimizer),
        "optimizer": "AdamW",
        "scheduler_state": dict(scheduler_state),
        "rng_state": rng_state,
        "rng_state_hashes": C.rng_state_hashes(rng_state),
        "batch_plan": dict(plan),
        "model_audit": dict(model_audit),
        "source_audit": dict(source_audit) if source_audit is not None else None,
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
            "n_tasks_scored": int(valid_report["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(valid_report["n_tasks_with_observed_labels"]),
            "valid_logits_sha256": valid_report.get("valid_logits_sha256"),
            "valid_probs_sha256": valid_report.get("valid_probs_sha256"),
        },
        "recipe": C.recipe_for_arm(arm, int(phase)),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


__all__ = [
    "checkpoint_payload",
    "load_checkpoint",
    "load_source_checkpoint",
    "save_checkpoint",
    "state_fingerprint",
    "verify_source_metadata",
]
