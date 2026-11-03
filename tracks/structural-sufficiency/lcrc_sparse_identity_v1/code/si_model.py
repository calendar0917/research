"""A0 plus the three pre-declared wide arms.

The A0 graph is *not* re-implemented: every arm embeds the frozen
``ptb_model.PTBLCRC`` variant ``A0`` object built by ``ptb_model.build_model``,
which is the exact construction path the historical PTB MolHIV A0 used (including
its two constructor calls, so the global RNG stream after construction is the
same as A0's).

The only addition is one bias-free ``nn.Linear(in_dim, 1)`` whose weight is
zero-initialised inside a forked RNG context.  Consequences:

* A0 parameters are bitwise identical to the same-seed A0 (gate 12);
* the branch consumes no global RNG, so shuffling and dropout masks are the same
  as A0's (gate 10);
* at initialisation the branch contributes exactly zero (gate 10);
* WIDE_TRUE and WIDE_NULL have the same trainable parameter count (gate 6).

``in_dim`` is 512 for WIDE_TRUE / WIDE_NULL and 64 for FOLD64.  The pre-computed
branch input matrix is carried in the batch dict under ``si_feat``; A0 never
reads it.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import si_common as C

import ptb_model as PM
from feature_spec import FeatureSpec


class WideLCRC(nn.Module):
    """Frozen A0 plus a zero-initialised linear residual on ``batch['si_feat']``."""

    def __init__(self, base: PM.PTBLCRC, kind: str, in_dim: int):
        super().__init__()
        if kind not in C.WIDE:
            raise KeyError(kind)
        self.kind = kind
        self.in_dim = int(in_dim)
        self.base = base
        with torch.random.fork_rng(devices=[]):
            # Deterministic placeholder allocation; the weight is then zeroed, so
            # no draw matters.  The fork keeps the global stream untouched.
            torch.manual_seed(0)
            self.branch = nn.Linear(self.in_dim, 1, bias=False)
        nn.init.zeros_(self.branch.weight)

    def forward(self, batch: Mapping[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        y = self.base(batch, **kwargs)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        feat = batch.get("si_feat")
        if feat is None:
            if float(self.branch.weight.abs().sum()) != 0.0:
                raise RuntimeError("wide branch has non-zero weight but no feature matrix was supplied")
            return y
        if feat.shape[1] != self.in_dim:
            raise ValueError(f"branch input width {feat.shape[1]} != {self.in_dim}")
        return y + self.branch(feat.to(dtype=y.dtype)).reshape(-1)

    def branch_contribution(self, feat: np.ndarray, device) -> np.ndarray:
        w = self.branch.weight.detach().reshape(-1)
        f = torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device)
        with torch.no_grad():
            return torch.mv(f, w).float().cpu().numpy().astype(np.float64)


def build_model(
    kind: str,
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    seed: int,
    device,
    *,
    in_dim: int | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build one arm with the frozen same-seed A0 initialisation."""
    if kind not in C.NEURAL:
        raise KeyError(kind)
    base, info = PM.build_model(spec, norm, C.MODEL_A0, int(seed), device, n_tasks=C.N_TASKS)
    if kind == C.MODEL_A0:
        info = dict(info)
        info.update({"arm": C.MODEL_A0, "branch_dim": 0, "branch_parameters": 0, "branch_init": None})
        return base, info
    dim = int(C.BRANCH_DIM[kind] if in_dim is None else in_dim)
    model = WideLCRC(base, kind, dim).to(device)
    info = dict(info)
    info.update(
        {
            "arm": kind,
            "branch_dim": dim,
            "branch_parameters": int(model.branch.weight.numel()),
            "branch_init": "zeros",
            "branch_bias": False,
            "base_parameters": int(sum(p.numel() for p in base.parameters())),
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }
    )
    return model, info


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    return PM.state_fingerprint(state)


def load_model(
    kind: str,
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    device,
    *,
    in_dim: int | None = None,
) -> nn.Module:
    """Rebuild an arm from a saved state dict (used by gates and Test scoring)."""
    base = PM.PTBLCRC(spec, norm, C.MODEL_A0, n_tasks=C.N_TASKS, head_seed=0)
    if kind == C.MODEL_A0:
        model: nn.Module = base
    else:
        model = WideLCRC(base, kind, int(C.BRANCH_DIM[kind] if in_dim is None else in_dim))
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    model.to(device)
    model.eval()
    return model


def branch_norms(model: nn.Module) -> dict[str, float] | None:
    if not isinstance(model, WideLCRC):
        return None
    w = model.branch.weight.detach().reshape(-1).double()
    return {
        "l1": float(w.abs().sum()),
        "l2": float(torch.sqrt((w * w).sum())),
        "linf": float(w.abs().max()) if w.numel() else 0.0,
        "n_parameters": int(w.numel()),
        "n_nonzero": int((w != 0).sum()),
    }


def contribution_stats(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    a = np.abs(v)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean_abs": float(a.mean()),
        "sd_abs": float(a.std(ddof=0)),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90)),
        "p95_abs": float(np.percentile(a, 95)),
        "p99_abs": float(np.percentile(a, 99)),
        "max_abs": float(a.max()),
        "mean_signed": float(v.mean()),
        "n_positive": int((v > 0).sum()),
        "n_negative": int((v < 0).sum()),
        "n_zero": int((v == 0).sum()),
    }


__all__ = [
    "WideLCRC",
    "build_model",
    "load_model",
    "state_fingerprint",
    "branch_norms",
    "contribution_stats",
]
