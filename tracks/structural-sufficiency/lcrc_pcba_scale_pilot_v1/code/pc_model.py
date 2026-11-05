"""Thin adapter to the frozen LCRC-C1 module (no new module, no new parameter)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pc_common as C
import bp_data as D
import bp_model as BM
from feature_spec import FeatureSpec


def build_c1(
    spec: FeatureSpec, norm: Mapping[str, Any], n_out: int, seed: int, device: torch.device | str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the historical C1 model for ``n_out`` tasks (128 for PCBA)."""
    if int(n_out) != C.N_TASKS:
        raise ValueError(f"this round only trains {C.N_TASKS} outputs, got {n_out}")
    model, info = BM.build_c1(spec, norm, int(n_out), int(seed), device, cont_slice=None)
    expected = int(C.FROZEN_WIDTHS["width"])
    if int(info["local_input_dim"]) <= 0 or expected <= 0:  # pragma: no cover - defensive
        raise RuntimeError("unexpected local layout")
    if "b_mean" in info["local_layout"]:
        raise RuntimeError("C1 must not contain the b_mean local block")
    return model, info


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    return BM.state_fingerprint(state)


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return D.collate(cache, rows)


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_logits(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    rows = np.asarray(rows, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = to_device(collate(cache, chunk.tolist()), device)
            logits = model(batch)
            if logits.shape[1] != C.N_TASKS:
                raise RuntimeError(f"model returned {logits.shape[1]} outputs, expected {C.N_TASKS}")
            out.append(logits.detach().cpu().numpy().astype(np.float32))
    logits = np.concatenate(out, axis=0) if out else np.zeros((0, C.N_TASKS), dtype=np.float32)
    if logits.shape[0] != len(rows):
        raise RuntimeError(f"prediction rows {logits.shape[0]} != {len(rows)}")
    return logits


def sigmoid(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


__all__ = [
    "build_c1",
    "collate",
    "parameter_count",
    "predict_logits",
    "sigmoid",
    "state_fingerprint",
    "to_device",
]
