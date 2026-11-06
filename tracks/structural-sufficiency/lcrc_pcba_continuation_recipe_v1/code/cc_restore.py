"""Strict model/optimizer/RNG restore and continuation checkpoint payloads."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import torch

import cc_common as C
import nmp_model as NM
import sg_model as SM
import pc_model as PC


# ---------------------------------------------------------------------------
# model construction (architecture exactly inherited from the source round)
# ---------------------------------------------------------------------------


def build_model(model_id: str, task: Any, device: torch.device | str = "cpu"):
    if model_id in ("A", "G"):
        _, _, base_init = NM.build_reference(task.spec, task.norm, C.N_TASKS, C.SEED)
        model, audit = NM.build_arm(
            model_id,
            task.spec,
            task.norm,
            C.N_TASKS,
            seed=C.SEED,
            device="cpu",
            base_init_state=base_init,
            zero_residual=True,
        )
        return model, audit
    if model_id == "V":
        _, _, base_init = SM.build_reference(task.spec, task.norm, C.N_TASKS, C.SEED)
        model, audit = SM.build_arm(
            "V",
            task.spec,
            task.norm,
            C.N_TASKS,
            seed=C.SEED,
            device="cpu",
            base_init_state=base_init,
            warm_start_state=None,
            zero_residual=True,
        )
        return model, audit
    if model_id == "C":
        model, info = PC.build_c1(task.spec, task.norm, C.N_TASKS, C.SEED, "cpu")
        return model, {
            "arm": "C",
            "arm_name": "C_control",
            "block_mode": "none",
            "total_parameter_count": int(PC.parameter_count(model)),
            "model_info": info,
        }
    raise KeyError(model_id)


def make_optimizer(model: torch.nn.Module, *, lr: float = C.LR_FLOOR) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr),
        betas=tuple(C.COMMON_RECIPE["adam_betas"]),
        eps=float(C.COMMON_RECIPE["adam_eps"]),
        weight_decay=float(C.COMMON_RECIPE["weight_decay"]),
    )


def optimizer_step_values(optimizer: torch.optim.Optimizer) -> list[float]:
    state = SM.optimizer_state_cpu(optimizer)
    values = set()
    for values_dict in state["state"].values():
        step = values_dict["step"]
        values.add(float(step.detach().cpu().item()) if torch.is_tensor(step) else float(step))
    return sorted(values)


def optimizer_fingerprint(optimizer: torch.optim.Optimizer) -> str:
    return SM.optimizer_state_fingerprint(SM.optimizer_state_cpu(optimizer))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> float:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
    return float(lr)


def restore_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return SM.restore_rng_state(state)


def _abs_path(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else C.REPO / p


# ---------------------------------------------------------------------------
# strict checkpoint restore
# ---------------------------------------------------------------------------


def restore_model_and_optimizer(
    model_id: str,
    payload: Mapping[str, Any],
    task: Any,
    *,
    device: torch.device | str,
) -> tuple[torch.nn.Module, torch.optim.AdamW, dict[str, Any]]:
    model, model_audit = build_model(model_id, task, device="cpu")
    state = payload["state_dict"]
    result = model.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:  # pragma: no cover - strict=True raises
        raise RuntimeError(f"state_dict load mismatch: {result}")

    bitwise = True
    max_abs = 0.0
    worst = None
    after_load = model.state_dict()
    if set(after_load) != set(state):
        raise RuntimeError("loaded state_dict key set differs from the checkpoint")
    for key in sorted(state):
        saved = state[key].detach().cpu()
        got = after_load[key].detach().cpu()
        if tuple(saved.shape) != tuple(got.shape) or saved.dtype != got.dtype:
            raise RuntimeError(f"state tensor {key} shape/dtype mismatch")
        if not torch.equal(saved, got):
            bitwise = False
            diff = float((saved.float() - got.float()).abs().max().item())
            if diff > max_abs:
                max_abs = diff
                worst = key
    if not bitwise:
        raise RuntimeError(f"state_dict is not bitwise equal (worst={worst}, max_abs={max_abs})")

    saved_opt = payload["optimizer_state_dict"]
    optimizer = make_optimizer(model)
    if len(saved_opt.get("param_groups", [])) != len(optimizer.param_groups):
        raise RuntimeError("checkpoint optimizer groups differ from the rebuilt optimizer")
    optimizer.load_state_dict(copy.deepcopy(dict(saved_opt)))
    loaded_opt = SM.optimizer_state_cpu(optimizer)
    eq = SM.compare_optimizer_states(loaded_opt, saved_opt)
    if not eq.get("equal", False):
        raise RuntimeError(f"restored AdamW state differs from checkpoint: {eq}")

    steps = optimizer_step_values(optimizer)
    expected_step = int(payload["global_step"])
    if steps != [float(expected_step)]:
        raise RuntimeError(f"restored AdamW steps {steps} != checkpoint global_step {expected_step}")
    groups = loaded_opt["param_groups"]
    if len(groups) != 1:
        raise RuntimeError(f"checkpoint optimizer has {len(groups)} groups, expected 1")
    expected_ids = list(range(int(payload["n_optimizer_parameters"]))) if "n_optimizer_parameters" in payload else list(range(len(list(model.parameters()))))
    if list(groups[0].get("params", [])) != expected_ids:
        raise RuntimeError(
            f"optimizer parameter ids {groups[0].get('params')} != contiguous {expected_ids}"
        )
    if set(loaded_opt["state"]) != set(expected_ids):
        raise RuntimeError("optimizer state does not cover exactly the model parameters")
    hyper = {
        "betas": tuple(groups[0].get("betas", ())),
        "eps": float(groups[0].get("eps", -1.0)),
        "weight_decay": float(groups[0].get("weight_decay", -1.0)),
    }
    if hyper["betas"] != tuple(C.COMMON_RECIPE["adam_betas"]):
        raise RuntimeError(f"optimizer betas {hyper['betas']} != frozen")
    if hyper["eps"] != float(C.COMMON_RECIPE["adam_eps"]):
        raise RuntimeError(f"optimizer eps {hyper['eps']} != frozen")
    if hyper["weight_decay"] != float(C.COMMON_RECIPE["weight_decay"]):
        raise RuntimeError(f"optimizer weight_decay {hyper['weight_decay']} != frozen")

    model.to(torch.device(device))
    # Move the moment tensors to the target device; keep the scalar ``step`` on
    # CPU, matching torch.optim.AdamW's own state convention.
    for values in optimizer.state.values():
        for key, value in list(values.items()):
            if torch.is_tensor(value) and key != "step":
                values[key] = value.to(torch.device(device))

    audit = {
        "model": model_id,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "checkpoint_sha256": C.sha256_file(_abs_path(payload["_path"])) if payload.get("_path") else None,
        "state_dict_tensors": int(len(state)),
        "state_dict_keys": sorted(state),
        "state_dict_bitwise_equal": True,
        "state_dict_fingerprint": SM.state_fingerprint(state),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "n_parameter_tensors": int(len(list(model.parameters()))),
        "model_audit": model_audit,
        "optimizer": "AdamW",
        "optimizer_param_groups": 1,
        "optimizer_param_ids": list(groups[0]["params"]),
        "optimizer_state_ids": sorted(int(x) for x in loaded_opt["state"]),
        "optimizer_step_values": steps,
        "optimizer_state_bitwise_equal": True,
        "optimizer_state_fingerprint": SM.optimizer_state_fingerprint(loaded_opt),
        "optimizer_hyperparameters": hyper,
    }
    return model, optimizer, audit


# ---------------------------------------------------------------------------
# checkpoint save
# ---------------------------------------------------------------------------


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
    model_id: str,
    recipe: str,
    stage: str,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    source_audit: Mapping[str, Any] | None,
    valid_ap: float | None,
    valid_report: Mapping[str, Any] | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "stage": stage,
        "model": model_id,
        "model_name": C.SOURCE[model_id]["arm_name"],
        "recipe": recipe,
        "recipe_name": C.RECIPE_NAMES[recipe],
        "seed": C.SEED,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": C.N_TASKS,
        "variant": C.SOURCE[model_id].get("mode"),
        "state_dict": SM.state_cpu(model),
        "optimizer_state_dict": SM.optimizer_state_cpu(optimizer),
        "optimizer": "AdamW",
        "scheduler_state": {
            "kind": "continuation_recipe",
            "recipe": recipe,
            "global_step": int(global_step),
            "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
            "lr_schedule_sha256": C.schedule_sha256(recipe),
            "lr_at_global_step": C.scheduled_lr(recipe, int(global_step)),
            "definition": C.schedule_audit()["definitions"][recipe],
        },
        "rng_state": rng_state,
        "rng_state_hashes": C.rng_state_hashes(rng_state),
        "batch_plan": dict(plan),
        "model_audit": dict(model_audit),
        "source_audit": dict(source_audit) if source_audit else None,
        "feature_spec_hash": C.FEATURE_SPEC_HASH,
        "norm_fingerprint": C.NORM_FINGERPRINT,
        "split_partition_sha256": C.SPLIT_PARTITION_SHA256,
        "label_artifact_sha256": C.LABEL_ARTIFACT_SHA256,
        "cache_sha256_manifest": C.CACHE_SHA256,
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "valid_report_summary": None
        if valid_report is None
        else {
            "ap": float(valid_report["ap"]),
            "valid_bce": float(valid_report["valid_bce"]),
            "n_tasks_scored": int(valid_report["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(valid_report["n_tasks_with_observed_labels"]),
            "valid_logits_sha256": valid_report.get("valid_logits_sha256"),
        },
        "recipe_config": C.recipe_config(recipe),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "test_evaluated": False,
        "test_policy": "official Test labels are read only after the prediction freeze",
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


__all__ = [
    "build_model",
    "checkpoint_payload",
    "load_checkpoint",
    "make_optimizer",
    "optimizer_fingerprint",
    "optimizer_step_values",
    "restore_model_and_optimizer",
    "restore_rng_state",
    "save_checkpoint",
    "set_lr",
]
