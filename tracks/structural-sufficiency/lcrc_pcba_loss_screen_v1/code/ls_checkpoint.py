"""Checkpoint, frozen-model and AdamW-state loading for the loss screen.

The source checkpoint is the L-low-lr epoch-9 best checkpoint.  Its model
weights and AdamW moments are restored in both arms; no moment is
re-initialised.  The checkpoint is read-only and its SHA-256 is checked before
any tensor is touched.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import ls_common as C
import pc_metrics as MET
import pc_model as BM


def _tensor_digest(value: Any) -> str:
    if torch.is_tensor(value):
        arr = value.detach().cpu().contiguous().numpy()
    else:
        arr = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(tuple(arr.shape)).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def load_source_checkpoint(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path or C.SOURCE["checkpoint"])
    sha = C.sha256_file(path)
    if sha != C.SOURCE["checkpoint_sha256"]:
        raise RuntimeError(f"source checkpoint sha256 {sha} != frozen {C.SOURCE['checkpoint_sha256']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError("source checkpoint is not a dict with state_dict")
    verify_source_metadata(payload)
    return payload


def verify_source_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "schema_is_lr_probe": payload.get("schema") == "lcrc-pcba-lr-probe-checkpoint-v1",
        "round_id": payload.get("round_id") == C.SOURCE["round"],
        "group": payload.get("group") == C.SOURCE["group"],
        "group_name": payload.get("group_name") == C.SOURCE["group_name"],
        "epoch": int(payload.get("epoch", -1)) == int(C.SOURCE["best_epoch"]),
        "valid_ap": abs(float(payload.get("valid_ap", float("nan"))) - float(C.SOURCE["best_valid_ap"]))
        <= 1e-12,
        "feature_spec_hash": payload.get("feature_spec_hash") == C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": payload.get("norm_fingerprint") == C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": payload.get("split_partition_sha256")
        == C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": payload.get("label_artifact_sha256")
        == C.SOURCE["label_artifact_sha256"],
        "cache_sha256_manifest": payload.get("cache_sha256_manifest") == C.SOURCE["cache_sha256"],
        "test_evaluated_false": payload.get("test_evaluated") is False,
        "has_optimizer_state": isinstance(payload.get("optimizer_state_dict"), dict),
        "has_rng_state": isinstance(payload.get("rng_state"), dict),
        "optimizer_is_adamw": payload.get("optimizer") == "AdamW",
        "recipe_lr": float((payload.get("recipe") or {}).get("learning_rate", -1.0)) == C.RECIPE["learning_rate"],
        "recipe_wd": float((payload.get("recipe") or {}).get("weight_decay", -1.0))
        == C.RECIPE["weight_decay"],
        "recipe_betas": tuple((payload.get("recipe") or {}).get("adam_betas", ()))
        == tuple(C.RECIPE["adam_betas"]),
        "recipe_eps": float((payload.get("recipe") or {}).get("adam_eps", -1.0)) == C.RECIPE["adam_eps"],
        "recipe_dropout": float((payload.get("recipe") or {}).get("dropout", -1.0)) == C.RECIPE["dropout"],
        "recipe_batch": int((payload.get("recipe") or {}).get("batch_size", -1)) == C.BATCH_SIZE,
        "recipe_clip": float((payload.get("recipe") or {}).get("gradient_clip", -1.0))
        == C.RECIPE["gradient_clip"],
    }
    if not all(checks.values()):
        failed = {k: v for k, v in checks.items() if not v}
        raise RuntimeError(f"source checkpoint metadata mismatch: {failed}")

    state = payload["state_dict"]
    if not isinstance(state, dict) or len(state) != 37:
        raise RuntimeError(f"source state_dict has {len(state) if isinstance(state, dict) else type(state)} entries, expected 37")
    opt = payload["optimizer_state_dict"]
    groups = opt.get("param_groups") or []
    if len(groups) != 1:
        raise RuntimeError(f"source optimizer has {len(groups)} param groups, expected 1")
    params = list(groups[0].get("params") or [])
    if params != list(range(31)):
        raise RuntimeError(f"source optimizer param ids are not 0..30: {params}")
    if set(opt.get("state", {})) != set(range(31)):
        raise RuntimeError("source optimizer state does not cover exactly 31 parameters")
    steps = []
    for pid in range(31):
        values = opt["state"][pid]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if key not in values:
                raise RuntimeError(f"source optimizer state[{pid}] missing {key}")
        step = values["step"]
        if torch.is_tensor(step):
            step_value = float(step.detach().cpu().item())
        else:
            step_value = float(step)
        steps.append(step_value)
    expected_step = float(C.SOURCE["optimizer_step_per_param"])
    if any(s != expected_step for s in steps):
        raise RuntimeError(
            f"source optimizer steps {sorted(set(steps))} != frozen epoch-9 step {expected_step}"
        )
    rng = payload["rng_state"]
    for key in ("python_random", "numpy", "torch_cpu", "torch_cuda"):
        if key not in rng:
            raise RuntimeError(f"source RNG state missing {key}")
    if len(rng["torch_cuda"]) != 1:
        raise RuntimeError(f"source RNG has {len(rng['torch_cuda'])} CUDA states, expected 1")
    return {
        "checks": {k: bool(v) for k, v in checks.items()},
        "state_tensors": int(len(state)),
        "optimizer_param_ids": params,
        "optimizer_steps": sorted(set(steps)),
        "optimizer_step_expected": expected_step,
        "rng_cuda_states": int(len(rng["torch_cuda"])),
    }


def _state_equal(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if set(a) != set(b):
        return {
            "equal": False,
            "reason": "key sets differ",
            "only_a": sorted(set(a) - set(b)),
            "only_b": sorted(set(b) - set(a)),
        }
    max_diff = 0.0
    worst = None
    all_equal = True
    for key in sorted(a):
        av, bv = a[key], b[key]
        if tuple(av.shape) != tuple(bv.shape) or av.dtype != bv.dtype:
            return {
                "equal": False,
                "reason": f"shape/dtype mismatch at {key}",
                "a": [list(av.shape), str(av.dtype)],
                "b": [list(bv.shape), str(bv.dtype)],
            }
        if not torch.equal(av, bv):
            all_equal = False
            diff = float((av.float() - bv.float()).abs().max().item())
            if diff > max_diff:
                max_diff = diff
                worst = key
    return {
        "equal": bool(all_equal),
        "max_abs_diff": float(max_diff),
        "worst_key": worst,
        "n_tensors": len(a),
    }


def build_model_from_checkpoint(
    task: Any,
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
    group: str,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    model, model_info = BM.build_c1(task.spec, task.norm, C.N_TASKS, C.SEED, device)
    checks = {
        "variant_C1": model_info.get("variant") == "C1",
        "parameters_245184": int(model_info.get("parameters", -1)) == C.SOURCE["n_parameters"],
        "no_b_mean": "b_mean" not in model_info.get("local_layout", {}),
        "dropout_0p10": float(getattr(model, "drop").p) == C.RECIPE["dropout"],
        "fp32": str(next(model.parameters()).dtype) == "torch.float32",
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen-model checks failed: {checks}")

    source_state = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(source_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={missing} unexpected={unexpected}")
    loaded_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    source_local = {k: v.detach().cpu().clone() for k, v in source_state.items()}
    equality = _state_equal(loaded_state, source_local)
    loaded_fp = BM.state_fingerprint(loaded_state)
    source_fp = BM.state_fingerprint(source_local)
    if not equality["equal"] or loaded_fp != source_fp:
        raise RuntimeError(
            f"loaded model state is not bitwise equal to source: equality={equality} "
            f"loaded_fp={loaded_fp} source_fp={source_fp}"
        )

    hist_info = checkpoint.get("model_info", {})
    info_checks = {
        "variant_equal": model_info.get("variant") == hist_info.get("variant") == "C1",
        "parameters_equal": model_info.get("parameters") == hist_info.get("parameters"),
        "local_layout_equal": model_info.get("local_layout") == hist_info.get("local_layout"),
        "local_input_dim_equal": model_info.get("local_input_dim") == hist_info.get("local_input_dim"),
        "input_widths_equal": model_info.get("input_widths") == hist_info.get("input_widths"),
        "init_state_hash_equal": model_info.get("init_state_hash") == hist_info.get("init_state_hash"),
        "full_init_state_hash_equal": model_info.get("full_init_state_hash")
        == hist_info.get("full_init_state_hash"),
    }
    if not all(info_checks.values()):
        raise RuntimeError(f"rebuilt model info mismatch: {info_checks}")

    audit = {
        "group": group,
        "device": str(device),
        "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
        "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
        "source_epoch": int(C.SOURCE["best_epoch"]),
        "source_valid_ap": float(C.SOURCE["best_valid_ap"]),
        "loaded_parameter_fingerprint": loaded_fp,
        "source_parameter_fingerprint": source_fp,
        "parameter_fingerprints_equal": bool(loaded_fp == source_fp),
        "state_dict_bitwise_equal": bool(equality["equal"]),
        "state_dict_max_abs_diff": float(equality["max_abs_diff"]),
        "state_dict_tensors": int(equality["n_tensors"]),
        "parameter_count": int(model_info["parameters"]),
        "model_dropout_p": float(getattr(model, "drop").p),
        "model_dtype": str(next(model.parameters()).dtype),
        "model_info_checks": {k: bool(v) for k, v in info_checks.items()},
        "model_info": model_info,
        "model_checks": {k: bool(v) for k, v in checks.items()},
    }
    return model, model_info, audit


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.RECIPE["learning_rate"]),
        betas=tuple(float(x) for x in C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )


def optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    out: dict[str, Any] = {
        "state": {},
        "param_groups": copy.deepcopy(state["param_groups"]),
    }
    for param_id, values in state["state"].items():
        out["state"][param_id] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in values.items()
        }
    return out


def _recursive_equal(a: Any, b: Any) -> dict[str, Any]:
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return {"equal": False, "reason": "tensor vs non-tensor"}
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            return {
                "equal": False,
                "reason": "shape/dtype mismatch",
                "a": [list(a.shape), str(a.dtype)],
                "b": [list(b.shape), str(b.dtype)],
            }
        exact = bool(torch.equal(a, b))
        diff = float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0
        return {"equal": exact, "max_abs_diff": diff}
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return {
                "equal": False,
                "reason": "key mismatch",
                "only_a": sorted(map(str, set(a) - set(b))),
                "only_b": sorted(map(str, set(b) - set(a))),
            }
        rows = {str(k): _recursive_equal(a[k], b[k]) for k in a}
        return {
            "equal": all(v["equal"] for v in rows.values()),
            "max_abs_diff": max((v.get("max_abs_diff", 0.0) for v in rows.values()), default=0.0),
            "children": rows,
        }
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return {"equal": False, "reason": "length mismatch"}
        rows = [_recursive_equal(x, y) for x, y in zip(a, b)]
        return {
            "equal": all(v["equal"] for v in rows),
            "max_abs_diff": max((v.get("max_abs_diff", 0.0) for v in rows), default=0.0),
        }
    try:
        same = bool(a == b)
    except Exception:
        same = False
    return {"equal": same}


def optimizer_state_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    return _recursive_equal(a, b)


def optimizer_state_fingerprint(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    groups = state.get("param_groups", [])
    h.update(repr(C._json_default(groups)).encode())
    for pid in sorted(state.get("state", {}), key=lambda x: int(x)):
        values = state["state"][pid]
        h.update(str(int(pid)).encode())
        for key in sorted(values):
            h.update(key.encode())
            h.update(_tensor_digest(values[key]).encode())
    return h.hexdigest()


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    source_state: Mapping[str, Any],
    model: torch.nn.Module,
) -> dict[str, Any]:
    params = [p for p in model.parameters() if p.requires_grad]
    groups = source_state.get("param_groups") or []
    if len(groups) != 1:
        raise RuntimeError(f"source optimizer has {len(groups)} groups, expected 1")
    ids = list(groups[0].get("params") or [])
    if ids != list(range(len(params))):
        raise RuntimeError(f"source param ids {ids} != 0..{len(params) - 1}")
    if set(source_state.get("state", {})) != set(range(len(params))):
        raise RuntimeError("source optimizer state does not cover all model parameters")
    for pid, param in enumerate(params):
        values = source_state["state"][pid]
        for key in ("exp_avg", "exp_avg_sq"):
            shape = tuple(values[key].shape)
            if shape != tuple(param.shape):
                raise RuntimeError(f"optimizer {key}[{pid}] shape {shape} != param {tuple(param.shape)}")
    optimizer.load_state_dict(source_state)
    loaded = optimizer_state_cpu(optimizer)
    source_cpu = {
        "state": {
            int(pid): {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                for k, v in values.items()
            }
            for pid, values in source_state["state"].items()
        },
        "param_groups": copy.deepcopy(source_state["param_groups"]),
    }
    equality = optimizer_state_equal(loaded, source_cpu)
    source_fp = optimizer_state_fingerprint(source_cpu)
    loaded_fp = optimizer_state_fingerprint(loaded)
    if not equality["equal"] or source_fp != loaded_fp:
        raise RuntimeError(
            f"loaded AdamW state is not identical to source: equality={equality} "
            f"loaded_fp={loaded_fp} source_fp={source_fp}"
        )
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in loaded["state"].values()
        }
    )
    audit = {
        "optimizer": "AdamW",
        "learning_rate": float(C.RECIPE["learning_rate"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "betas": list(C.RECIPE["adam_betas"]),
        "eps": float(C.RECIPE["adam_eps"]),
        "n_param_groups": len(loaded["param_groups"]),
        "n_state_entries": len(loaded["state"]),
        "source_step_values": steps,
        "source_optimizer_state_fingerprint": source_fp,
        "loaded_optimizer_state_fingerprint": loaded_fp,
        "state_identical_to_source": bool(equality["equal"]),
        "state_max_abs_diff": float(equality.get("max_abs_diff", 0.0)),
        "moments_reinitialised": False,
    }
    return audit


def _array_sha256(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return C.sha256_bytes(a.tobytes())


def evaluate_validation(
    model: torch.nn.Module,
    task: Any,
    device: torch.device,
    *,
    reason: str = "",
) -> dict[str, Any]:
    logits = BM.predict_logits(model, task.cache, task.valid_rows, device, batch_size=256)
    probs = BM.sigmoid(logits)
    report = MET.validation_report(task.labels[task.valid_rows], probs)
    report.update(
        {
            "reason": reason,
            "valid_logits_sha256": _array_sha256(logits),
            "valid_probs_sha256": _array_sha256(probs),
            "valid_logits_shape": list(logits.shape),
            "inference_transfer": "pc_model.predict_logits (pageable, frozen inference path)",
        }
    )
    return report


def parameter_value_summary(model: torch.nn.Module) -> dict[str, Any]:
    rows = []
    for name, param in model.named_parameters():
        arr = param.detach().cpu().contiguous().numpy()
        rows.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": C.sha256_bytes(np.ascontiguousarray(arr).tobytes()),
            }
        )
    return {
        "n_parameters": len(rows),
        "parameter_fingerprint": BM.state_fingerprint(model.state_dict()),
        "parameters": rows,
    }


__all__ = [
    "build_model_from_checkpoint",
    "evaluate_validation",
    "load_optimizer_state",
    "load_source_checkpoint",
    "make_optimizer",
    "optimizer_state_cpu",
    "optimizer_state_equal",
    "optimizer_state_fingerprint",
    "parameter_value_summary",
    "verify_source_metadata",
]
