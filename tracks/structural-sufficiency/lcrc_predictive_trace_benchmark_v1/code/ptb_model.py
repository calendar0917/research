"""The two models under test: ``A0`` (frozen LCRC-A) and ``C1`` (NO_BOND_MEAN).

The computational graph is *not* re-implemented here.  ``A0``/``C1`` are the
frozen ``lcrc_lipo_model_dissection_v1`` ``DissectLCRC`` objects, with exactly one
authorised addition: the two output heads ``f`` and ``rho`` may widen from 1 to
``n_tasks`` outputs so that Tox21 (12 observed-label endpoints) can be trained
with one shared encoder.  That widening adds the same number of parameters to
both variants, so the frozen 18 432-parameter gap is preserved (asserted).

Single-task forwards delegate to the frozen parent ``forward`` unchanged, so the
numerical behaviour is bit-identical to the upstream implementation (verified in
``analysis/correctness_gates.json`` by replaying the frozen public122
Lipophilicity checkpoint).

Shared initialisation: for one training seed, ``C1`` is built by deleting the
96-d ``b_mean`` block from the ``A0`` state through the frozen column map
(``dissect_common.variant_cols`` / ``dissect_model.remap_state``).  Every shared
tensor is bitwise identical and recorded per run.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import ptb_common as C

import dissect_model as DM
from feature_spec import FeatureSpec


class PTBLCRC(DM.DissectLCRC):
    """Frozen A0/C1 graph with an optional ``n_tasks``-wide output head."""

    def __init__(
        self,
        spec: FeatureSpec,
        norm: Mapping[str, Any],
        variant: str,
        n_tasks: int = 1,
        cfg: Mapping[str, Any] | None = None,
        head_seed: int = 0,
    ):
        super().__init__(spec, norm, variant, cfg)
        self.n_tasks = int(n_tasks)
        if self.n_tasks < 1:
            raise ValueError("n_tasks must be >= 1")
        self.head_seed = int(head_seed)
        if self.n_tasks != 1:
            width = int(self.cfg["width"])
            # ``f``/``rho`` are rebuilt with the frozen head shapes except for the
            # final output width; the local encoder is untouched.  A forked RNG
            # keeps the widened heads a deterministic function of the seed and
            # leaves the global stream untouched.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(head_seed) * 1_000_003 + 17)
                self.f = DM.MLP([width, width, self.n_tasks])
                self.rho = DM.MLP([len(self.rho_cols), int(self.cfg["rho_hidden"]), self.n_tasks])

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_aux: bool = False,
        zero_g: bool = False,
        zero_contrib: bool = False,
        zero_mean_h2: bool = False,
        zero_s2: bool = False,
        s2_agg_override: str | None = None,
    ):
        if self.n_tasks == 1:
            return super().forward(
                batch,
                return_aux=return_aux,
                zero_g=zero_g,
                zero_contrib=zero_contrib,
                zero_mean_h2=zero_mean_h2,
                zero_s2=zero_s2,
                s2_agg_override=s2_agg_override,
            )
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        a, b, scal = self._encode(batch)
        h, local_aux = self.local(batch, a, b, scal, s2_agg_override=s2_agg_override, zero_s2=zero_s2)
        h = self.drop(h)  # exactly one dropout call, shared by both readouts
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib_atom = self.f(h).view(-1, self.n_tasks)
        contrib = h.new_zeros((n_graphs, self.n_tasks)).index_add_(0, atom_batch, contrib_atom)
        y_contrib = torch.zeros_like(contrib) if zero_contrib else contrib
        v = C.DC.VARIANT_DEFS[self.variant]
        parts = []
        g = (batch["global"] - self.global_mean) / self.global_scale
        if zero_g:
            g = torch.zeros_like(g)
        parts.append(g)
        parts.append(mean_h)
        parts.append(torch.zeros_like(mean_h2) if zero_mean_h2 else mean_h2)
        parts.append(torch.log1p(count).unsqueeze(1))
        rho_in = torch.cat(parts, dim=1)
        if rho_in.shape[1] != len(self.rho_cols):
            raise RuntimeError(f"rho width {rho_in.shape[1]} != frozen column count {len(self.rho_cols)}")
        rho = self.rho(rho_in).view(-1, self.n_tasks)
        y = y_contrib + rho
        if return_aux:
            return y, {
                "contrib": contrib,
                "contrib_used": y_contrib,
                "rho": rho,
                "contrib_atom": contrib_atom,
                "mean_h": mean_h,
                "mean_h2": mean_h2,
                "h": h,
                "count": count,
                "a": a,
                "b": b,
                "rho_input": rho_in,
                **local_aux,
            }
        return y


# ---------------------------------------------------------------------------
# construction with the frozen shared initialisation
# ---------------------------------------------------------------------------


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = __import__("hashlib").sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def full_state(spec: FeatureSpec, norm: Mapping[str, Any], seed: int, n_tasks: int = 1) -> dict[str, torch.Tensor]:
    """One complete A0 initial state for a training seed (CPU RNG only)."""
    torch.manual_seed(int(seed))
    model = PTBLCRC(spec, norm, C.MODEL_A0, n_tasks=int(n_tasks), head_seed=int(seed))
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def variant_state(
    variant: str, full: Mapping[str, torch.Tensor], n_tasks: int = 1
) -> dict[str, torch.Tensor]:
    """Frozen column-deletion mapping A0 -> variant (C1 drops the b_mean columns)."""
    state = DM.remap_state(variant, full)
    if variant == C.MODEL_C1:
        local_cols, _ = C.DC.variant_cols(variant)
        if state["local.mlp.lins.0.weight"].shape[1] != len(local_cols):
            raise RuntimeError("remapped local MLP width does not equal the frozen C1 column count")
        if state["local.mlp.lins.0.weight"].shape[1] != 433:
            raise RuntimeError(f"C1 local MLP input width is {state['local.mlp.lins.0.weight'].shape[1]}, expected 433")
    return state


def build_model(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    variant: str,
    seed: int,
    device,
    *,
    n_tasks: int = 1,
    dropout: float | None = None,
) -> tuple[PTBLCRC, dict[str, Any]]:
    """Build A0/C1 with the frozen shared initialisation; returns (model, info)."""
    if variant not in C.MODELS:
        raise KeyError(variant)
    fs = full_state(spec, norm, int(seed), int(n_tasks))
    state = variant_state(variant, fs, int(n_tasks))
    model = PTBLCRC(spec, norm, variant, n_tasks=int(n_tasks), head_seed=int(seed))
    model.load_state_dict(state, strict=True)
    if dropout is not None:
        model.drop.p = float(dropout)
    model.to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    info = {
        "variant": variant,
        "label": C.MODEL_LABELS[variant],
        "seed": int(seed),
        "n_tasks": int(n_tasks),
        "parameters": n_params,
        "parameters_breakdown": DM.module_breakdown(model),
        "init_state_hash": state_fingerprint(state),
        "full_init_state_hash": state_fingerprint(fs),
        "shared_state_equal_to_full": DM.shared_state_equal(variant, fs, state),
        "local_input_cols": len(C.DC.variant_cols(variant)[0]),
        "rho_input_cols": len(C.DC.variant_cols(variant)[1]),
        "dropout": float(model.drop.p),
        "has_iterative_message_passing": False,
        "has_attention": False,
    }
    if dropout is not None:
        info["dropout_override"] = float(dropout)
    return model, info


def load_checkpoint_model(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    variant: str,
    state: Mapping[str, torch.Tensor],
    device,
    *,
    n_tasks: int = 1,
) -> PTBLCRC:
    model = PTBLCRC(spec, norm, variant, n_tasks=int(n_tasks), head_seed=0)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    model.to(device)
    model.eval()
    return model


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


__all__ = [
    "PTBLCRC",
    "build_model",
    "load_checkpoint_model",
    "full_state",
    "variant_state",
    "state_fingerprint",
    "parameter_count",
]
