"""The one tiny task-specific residual head used by both arms (8,784 params).

    S(G)            : (B, 3, 165)   frozen relation features
    R   = W_shared @ S              (165 -> 16, no bias, shared across bins)
    ev  = flatten_b(R)              (3 * 16 = 48)
    out = frozen_A_logits + W_task @ ev   (48 -> 128, no bias)

``W_task`` is initialised to exact zeros, so epoch 0 reproduces the frozen A
logits bitwise.  Both arms (REAL, SHUFFLE) start from a bitwise identical head
state built from the same private generator; neither arm inherits the other's
weights.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import rie_common as C


class RelationHead(nn.Module):
    """Fixed, tiny, one-shot relation readout (no message passing, no attention)."""

    def __init__(self, *, in_dim: int = C.PAIR_FEATURE_DIM, hidden: int = 16, n_bins: int = C.N_BINS,
                 n_tasks: int = C.N_TASKS):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.n_bins = int(n_bins)
        self.n_tasks = int(n_tasks)
        self.shared_relation_projection = nn.Linear(self.in_dim, self.hidden, bias=False)
        self.task_weight = nn.Linear(self.n_bins * self.hidden, self.n_tasks, bias=False)

    def forward(self, relation: torch.Tensor) -> torch.Tensor:
        if relation.ndim != 3 or relation.shape[1:] != (self.n_bins, self.in_dim):
            raise ValueError(f"relation tensor {tuple(relation.shape)} != (B,{self.n_bins},{self.in_dim})")
        evidence = self.shared_relation_projection(relation)  # (B, n_bins, hidden)
        return self.task_weight(evidence.reshape(relation.shape[0], self.n_bins * self.hidden))

    def relation_evidence(self, relation: torch.Tensor) -> torch.Tensor:
        evidence = self.shared_relation_projection(relation)
        return evidence.reshape(relation.shape[0], self.n_bins * self.hidden)


def head_parameter_names() -> tuple[str, ...]:
    return ("shared_relation_projection.weight", "task_weight.weight")


def head_parameter_count(head: nn.Module | None = None) -> int:
    if head is None:
        return int(C.HEAD_PARAMETERS)
    return int(sum(p.numel() for p in head.parameters()))


def initial_head_state(*, seed: int = C.SEED) -> dict[str, torch.Tensor]:
    """Frozen initial state: default Linear init for the projection, zeros for W_task.

    ``PyTorch``'s default ``nn.Linear`` init is ``kaiming_uniform_(a=sqrt(5))``,
    i.e. ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``; it is reproduced here from an
    explicit private generator so the global RNG stream is never touched.
    """
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        # constructing nn.Linear draws from the global stream; the fork keeps the
        # public RNG stream untouched so the paired plan/RNG audit stays exact
        model = RelationHead()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            bound = 1.0 / math.sqrt(float(model.shared_relation_projection.in_features))
            model.shared_relation_projection.weight.uniform_(-bound, bound, generator=gen)
            model.task_weight.weight.zero_()
    if not torch.equal(before, torch.get_rng_state()):
        raise RuntimeError("private head initialisation changed the global torch RNG state")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if set(state) != set(head_parameter_names()):
        raise RuntimeError(f"unexpected head state keys {sorted(state)}")
    return state


def build_head(state: Mapping[str, torch.Tensor] | None = None, *, device: torch.device | str = "cpu") -> RelationHead:
    with torch.random.fork_rng(devices=[]):
        head = RelationHead()
    resolved = initial_head_state() if state is None else {k: v.detach().cpu().clone() for k, v in state.items()}
    missing, unexpected = head.load_state_dict(resolved, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"head load mismatch: {missing} / {unexpected}")
    head.to(torch.device(device))
    if head_parameter_count(head) != C.HEAD_PARAMETERS:
        raise RuntimeError(f"head parameters {head_parameter_count(head)} != {C.HEAD_PARAMETERS}")
    return head


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Deterministic digest over the head state (name, shape, dtype, bytes)."""
    import hashlib

    h = hashlib.sha256()
    for name in sorted(state):
        tensor = np.ascontiguousarray(state[name].detach().cpu().numpy())
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.tobytes())
    return h.hexdigest()


def as_tensor(block: Any) -> torch.Tensor:
    """Read-only-safe float32 tensor copy (memmap slices are not writable)."""
    arr = np.asarray(block, dtype=np.float32)
    if not arr.flags.writeable:
        arr = arr.copy()
    return torch.from_numpy(np.ascontiguousarray(arr))


def predict_logits(
    head: RelationHead,
    relation: np.ndarray,
    base_logits: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """``frozen logits + head(relation)`` over an ``(N,3,165)`` block."""
    relation = np.asarray(relation)
    base_logits = np.asarray(base_logits, dtype=np.float32)
    if relation.shape[0] != base_logits.shape[0]:
        raise ValueError(f"relation rows {relation.shape[0]} != logits rows {base_logits.shape[0]}")
    head.eval()
    out = np.empty_like(base_logits)
    with torch.inference_mode():
        for start in range(0, relation.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), relation.shape[0])
            block = as_tensor(relation[start:stop]).to(device)
            delta = head(block)
            base = as_tensor(base_logits[start:stop]).to(device)
            out[start:stop] = (base + delta).detach().cpu().numpy()
    return out


__all__ = [
    "RelationHead",
    "as_tensor",
    "build_head",
    "head_parameter_count",
    "head_parameter_names",
    "initial_head_state",
    "predict_logits",
    "state_hash",
]
