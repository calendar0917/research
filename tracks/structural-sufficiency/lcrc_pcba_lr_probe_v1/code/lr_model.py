"""Checkpoint loading, parameter fingerprinting and official Validation AP."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import lr_common as C
import pc_metrics as MET
import pc_model as BM


def load_checkpoint(path: Path | str = C.HIST["checkpoint"]) -> dict[str, Any]:
    path = Path(path)
    sha = C.sha256_file(path)
    if sha != C.HIST["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint sha {sha} != frozen {C.HIST['checkpoint_sha256']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError("checkpoint is not the expected dict with state_dict")
    C.verify_historical_checkpoint_metadata(payload)
    return payload


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


def build_from_checkpoint(
    task: Any,
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
    group: str,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Build the frozen LCRC-C1, load the historical weights, audit the result."""
    model, model_info = BM.build_c1(task.spec, task.norm, C.N_TASKS, C.SEED, device)
    if model_info.get("variant") != "C1":
        raise RuntimeError(f"built variant {model_info.get('variant')} != C1")
    if int(model_info.get("parameters", -1)) != 245_184:
        raise RuntimeError(f"parameter count {model_info.get('parameters')} != 245,184")
    if "b_mean" in model_info.get("local_layout", {}):
        raise RuntimeError("C1 must not contain the b_mean block")
    if float(getattr(model, "drop").p) != 0.10:
        raise RuntimeError(f"model dropout p={getattr(model, 'drop').p} != 0.10")
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("model is not FP32")

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
            f"loaded state does not bitwise equal the historical state: "
            f"equality={equality} loaded_fp={loaded_fp} source_fp={source_fp}"
        )

    hist_info = checkpoint.get("model_info", {})
    info_checks = {
        "variant_equal": model_info.get("variant") == hist_info.get("variant") == "C1",
        "init_state_hash_equal": model_info.get("init_state_hash") == hist_info.get("init_state_hash"),
        "full_init_state_hash_equal": model_info.get("full_init_state_hash")
        == hist_info.get("full_init_state_hash"),
        "local_input_dim_equal": model_info.get("local_input_dim") == hist_info.get("local_input_dim"),
        "local_layout_equal": model_info.get("local_layout") == hist_info.get("local_layout"),
        "input_widths_equal": model_info.get("input_widths") == hist_info.get("input_widths"),
        "parameters_equal": model_info.get("parameters") == hist_info.get("parameters"),
    }
    if not all(info_checks.values()):
        raise RuntimeError(f"rebuilt model info does not match historical model_info: {info_checks}")

    audit = {
        "group": group,
        "device": str(device),
        "source_checkpoint": C.track_rel(C.HIST["checkpoint"]),
        "source_checkpoint_sha256": C.HIST["checkpoint_sha256"],
        "source_feature_spec_hash": checkpoint.get("feature_spec_hash"),
        "source_norm_fingerprint": checkpoint.get("norm_fingerprint"),
        "loaded_parameter_fingerprint": loaded_fp,
        "source_parameter_fingerprint": source_fp,
        "parameter_fingerprints_equal": bool(loaded_fp == source_fp),
        "state_dict_bitwise_equal": bool(equality["equal"]),
        "state_dict_max_abs_diff": float(equality["max_abs_diff"]),
        "state_dict_tensors": int(equality["n_tensors"]),
        "parameter_count": int(model_info["parameters"]),
        "model_dropout_p": float(getattr(model, "drop").p),
        "model_dtype": str(next(model.parameters()).dtype),
        "model_info_equal_to_historical": True,
        "model_info_checks": info_checks,
        "model_info": model_info,
    }
    return model, model_info, audit


def make_optimizer(model: torch.nn.Module, *, learning_rate: float) -> torch.optim.Optimizer:
    recipe = C.PROBE_RECIPE
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(learning_rate),
        betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]),
        weight_decay=float(recipe["weight_decay"]),
    )


def optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Deep-copy optimizer state to CPU for an immutable checkpoint snapshot."""
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
    """The only selection metric: official OGB Validation AP, FP32, pageable inference."""
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
    """Cheap per-parameter digest for human auditing (not used for selection)."""
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
    "build_from_checkpoint",
    "evaluate_validation",
    "load_checkpoint",
    "make_optimizer",
    "optimizer_state_cpu",
    "parameter_value_summary",
]
