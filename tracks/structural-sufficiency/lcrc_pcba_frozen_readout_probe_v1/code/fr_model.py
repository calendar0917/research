"""Read-only access to the frozen LCRC-C1 encoder and the ``rho`` input capture.

The only new machinery in this module is a forward **pre-hook** on the existing
``model.rho`` module.  It records the tensor the model itself passes to ``rho``;
no mathematical implementation is edited, and the hook is removed as soon as the
extraction finishes.  ``rho_in`` is additionally reconstructed independently
from the batch and the encoder output ``h`` so a silent layout change would be
caught instead of silently cached.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import torch

import fr_common as C
import pc_model as BM
import ls_checkpoint as CK
import tu_model as TU


class RhoInputCapture:
    """Forward pre-hook that records the exact input tensor of ``module``."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.calls = 0
        self._pending: torch.Tensor | None = None
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):  # noqa: ANN001 - torch signature
        if module is not self.module:  # pragma: no cover - defensive
            raise RuntimeError("rho hook fired for a different module")
        if len(args) != 1:
            raise RuntimeError(f"rho was called with {len(args)} positional arguments, expected 1")
        self.calls += 1
        self._pending = args[0]

    def take(self) -> torch.Tensor:
        if self._pending is None:
            raise RuntimeError("rho hook did not fire on the last forward pass")
        value = self._pending
        self._pending = None
        return value

    def discard(self) -> None:
        self._pending = None

    def close(self) -> None:
        self._handle.remove()
        self._pending = None


def rho_dimension_audit(model: torch.nn.Module, spec: Any) -> dict[str, Any]:
    """Verify the ``rho`` input width against the model and the frozen layout."""
    width = int(C.RHO_LAYOUT["mean_h"])
    layout_raw = TU.rho_layout(spec, width)
    layout = {str(k): [int(v[0]), int(v[1])] for k, v in layout_raw.items()}
    widths = {str(k): int(v[1] - v[0]) for k, v in layout_raw.items()}
    inferred = int(layout_raw["log1p_n"][1])
    first = model.rho.lins[0]
    hidden = model.rho.lins[1]
    checks = {
        "layout_widths_match_frozen": widths == {k: int(v) for k, v in C.RHO_LAYOUT.items()},
        "layout_total_matches_frozen": inferred == C.RHO_IN_DIM_EXPECTED,
        "layout_total_matches_rho_first_layer": int(first.weight.shape[1]) == inferred,
        "rho_output_is_128": int(model.rho.lins[-1].weight.shape[0]) == C.N_TASKS,
        "rho_hidden_is_64": int(first.weight.shape[0]) == 64 and int(hidden.weight.shape[0]) == C.N_TASKS,
        "rho_has_two_linears": len(model.rho.lins) == 2,
    }
    if not all(checks.values()):
        raise RuntimeError(f"rho input layout gate failed: {checks}")
    return {
        "rho_input_dim": inferred,
        "expected_from_round_instruction": C.RHO_IN_DIM_EXPECTED,
        "layout": layout,
        "widths": widths,
        "first_layer_weight_shape": list(first.weight.shape),
        "second_layer_weight_shape": list(hidden.weight.shape),
        "checks": {k: bool(v) for k, v in checks.items()},
        "source": "tu_model.rho_layout(spec, 64) compared with model.rho.lins[0].weight.shape[1]",
    }


def load_frozen_model(task: Any, *, device: torch.device, label: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the historical checkpoint and freeze every parameter before any step."""
    checkpoint = CK.load_source_checkpoint()
    model, model_info, load_audit = CK.build_model_from_checkpoint(
        task, checkpoint, device=device, group=label
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fp = BM.state_fingerprint(state)
    audit = {
        "label": label,
        "device": str(device),
        "model_eval": not bool(model.training),
        "requires_grad_any": bool(any(p.requires_grad for p in model.parameters())),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "parameter_fingerprint": fp,
        "state_dict_tensors": int(len(state)),
        "rho_dimension": rho_dimension_audit(model, task.spec),
        "checkpoint_load_audit": load_audit,
        "model_info": model_info,
        "dtype": str(next(model.parameters()).dtype),
    }
    if audit["requires_grad_any"] or not audit["model_eval"]:
        raise RuntimeError("frozen model gate failed: unfrozen parameter or training mode")
    if audit["parameters"] != int(C.SOURCE["n_parameters"]):
        raise RuntimeError(f"parameter count {audit['parameters']} != frozen {C.SOURCE['n_parameters']}")
    return model, audit


def reconstruct_rho_in(model: torch.nn.Module, batch: Mapping[str, torch.Tensor], h: torch.Tensor) -> torch.Tensor:
    """Independent, read-only reconstruction of ``[g, mean_h, mean_h2, log1p(n)]``."""
    n_graphs = int(batch["n_graphs"])
    atom_batch = batch["atom_batch"]
    width = int(h.shape[1])
    count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
    denom = count.clamp(min=1.0).unsqueeze(1)
    mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
    mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
    g = (batch["global"] - model.global_mean) / model.global_scale
    return torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)


def extract_split(
    model: torch.nn.Module,
    task: Any,
    rows: np.ndarray,
    device: torch.device,
    *,
    split: str,
    batch_size: int = int(C.EXTRACT["batch_size"]),
    rho_dim: int,
) -> dict[str, Any]:
    """One pass over ``rows``: cache the frozen logits and the ``rho`` inputs.

    Buffers are allocated **outside** ``inference_mode`` so the cached tensors
    stay usable as ordinary autograd inputs later; values are copied in place.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n = int(rows.size)
    rho_buf = torch.empty((n, int(rho_dim)), dtype=torch.float32, device=device)
    logit_buf = torch.empty((n, C.N_TASKS), dtype=torch.float32, device=device)
    capture = RhoInputCapture(model.rho)
    max_reconstruct_diff = 0.0
    max_replay_diff = 0.0
    max_contrib_diff = 0.0
    n_batches = 0
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, n, int(batch_size)):
                stop = min(start + int(batch_size), n)
                chunk = rows[start:stop]
                batch = BM.to_device(BM.collate(task.cache, chunk.tolist()), device)
                calls_before = capture.calls
                logits, aux = model(batch, return_aux=True)
                if capture.calls != calls_before + 1:
                    raise RuntimeError(
                        f"rho hook fired {capture.calls - calls_before} times in one forward pass"
                    )
                rho_in = capture.take()
                if tuple(logits.shape) != (stop - start, C.N_TASKS):
                    raise RuntimeError(f"logits shape {tuple(logits.shape)} != {(stop - start, C.N_TASKS)}")
                if tuple(rho_in.shape) != (stop - start, int(rho_dim)):
                    raise RuntimeError(f"rho_in shape {tuple(rho_in.shape)} != {(stop - start, int(rho_dim))}")
                if rho_in.dtype != torch.float32 or logits.dtype != torch.float32:
                    raise RuntimeError("extraction produced non-fp32 tensors")
                if not bool(torch.isfinite(rho_in).all()) or not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"non-finite features or logits at rows {start}:{stop}")

                recon = reconstruct_rho_in(model, batch, aux["h"])
                max_reconstruct_diff = max(
                    max_reconstruct_diff, float((recon - rho_in).abs().max().item())
                )
                replay = model.rho(rho_in)
                if capture.calls != calls_before + 2:
                    raise RuntimeError("unexpected hook count after the rho replay check")
                capture.discard()
                max_replay_diff = max(max_replay_diff, float((replay - aux["rho"]).abs().max().item()))
                max_contrib_diff = max(
                    max_contrib_diff,
                    float((aux["contrib"] + aux["rho"] - logits).abs().max().item()),
                )
                rho_buf[start:stop].copy_(rho_in)
                logit_buf[start:stop].copy_(logits)
                n_batches += 1
    finally:
        capture.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = float(time.perf_counter() - t0)
    return {
        "split": split,
        "rows": n,
        "rho_in": rho_buf,
        "logits": logit_buf,
        "audit": {
            "split": split,
            "rows": n,
            "rho_in_shape": list(rho_buf.shape),
            "logits_shape": list(logit_buf.shape),
            "dtype": "float32",
            "batch_size": int(batch_size),
            "n_batches": int(n_batches),
            "hook_calls": int(capture.calls),
            "hook_calls_expected": int(2 * n_batches),
            "seconds": seconds,
            "molecules_per_second": float(n / seconds) if seconds > 0 else None,
            "max_abs_rho_reconstruction_diff": float(max_reconstruct_diff),
            "max_abs_rho_replay_diff": float(max_replay_diff),
            "max_abs_contrib_plus_rho_minus_logits": float(max_contrib_diff),
            "rho_reconstruction_gate": bool(max_reconstruct_diff <= C.RHO_RECONSTRUCT_TOL),
            "rho_reconstruction_tolerance": float(C.RHO_RECONSTRUCT_TOL),
            "rho_replay_gate": bool(max_replay_diff <= C.RHO_REPLAY_TOL),
            "rho_replay_tolerance": float(C.RHO_REPLAY_TOL),
            "logits_identity_gate": bool(max_contrib_diff == 0.0),
            "cross_call_nondeterminism_note": (
                "mean_h/mean_h2 use CUDA atomic index_add_; a repeated forward is not "
                "bitwise identical (see runs/replay_diag/replay_diag.json)"
            ),
            "rho_in_sha256": C.sha256_array(rho_buf.cpu().numpy()),
            "logits_sha256": C.sha256_array(logit_buf.cpu().numpy()),
            "rows_sha256": C.sha256_int64(rows),
            "row_order": "ascending official OGB row ids restricted to this split",
            "first_rows": [int(x) for x in rows[:5]],
            "last_rows": [int(x) for x in rows[-5:]],
        },
    }


def parameter_digests(model: torch.nn.Module) -> dict[str, Any]:
    """Per-parameter digests, so "unchanged" is checked value by value."""
    rows = []
    for name, param in model.named_parameters():
        arr = param.detach().cpu().contiguous().numpy()
        rows.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": C.sha256_array(arr),
            }
        )
    return {
        "n_parameter_tensors": len(rows),
        "n_parameters": int(sum(int(np.prod(r["shape"])) for r in rows)),
        "parameter_fingerprint": BM.state_fingerprint(model.state_dict()),
        "parameters": rows,
    }


def replay_logits(
    model: torch.nn.Module,
    task: Any,
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Independent inference through the historical ``pc_model.predict_logits`` path."""
    return BM.predict_logits(model, task.cache, np.asarray(rows, dtype=np.int64), device, batch_size=batch_size)


__all__ = [
    "RhoInputCapture",
    "extract_split",
    "load_frozen_model",
    "parameter_digests",
    "reconstruct_rho_in",
    "replay_logits",
    "rho_dimension_audit",
]
