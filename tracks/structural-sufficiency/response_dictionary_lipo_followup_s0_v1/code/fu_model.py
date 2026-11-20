"""The JOINT arm: same parameters and initial state as RESP, different fusion.

S0 RESP::

    q = tanh(U z_S),  k = tanh(V z_C)
    T_RESP = sum_j w[j] q[j] k[j] / sqrt(16)

JOINT (this round, the only new unit)::

    h_joint = tanh((U z_S + V z_C) / sqrt(2))
    T_JOINT = sum_j w[j] h_joint[j] / sqrt(16)
    s_JOINT(C,S) = b(C) + a(S) + T_JOINT

This is the bias-free ``concat(z_S, z_C) -> 16 (tanh) -> 1`` joint MLP with a
``w``-weighted output: it can express cross-branch non-linear interaction but
says nothing about a product.  It is *not* ADD: ADD is ``w_j (q_j + k_j)``.

The class subclasses the frozen S0 model so that the encoder, ``a_head``,
``b_head``, ``U``, ``V``, ``w`` and ``fallback_head`` are constructed by exactly
the same code with exactly the same shapes; the parent's constructor argument
``mode`` is only used for the RESP/ADD operator choice, which this subclass
overrides.  The canonical S0 ``manual_seed(0)`` state therefore loads strictly
and keeps hash ``ccc8171d...``; with ``w = 0`` the fusion term is exactly zero,
so initial predictions coincide with S0 RESP/ADD.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch

import fu_common as FC  # noqa: F401  (sets sys.path for rd_*)
import rd_model as RM


class JointResponseModel(RM.ResponseModel):
    """``h = tanh((U z_S + V z_C)/sqrt(2))``; ``T = <w, h>/sqrt(16)``."""

    fusion_mode = "JOINT"

    def __init__(self, dims: Mapping[str, Any], config: RM.ModelConfig) -> None:
        parent_config = replace(config, mode="RESP")
        super().__init__(dims, parent_config)
        self.requested_mode = config.mode

    def terms(self, z_s: torch.Tensor, z_c: torch.Tensor, *, detach_w: bool = False):
        a = self.a_head(z_s).view(-1)
        b = self.b_head(z_c).view(-1)
        u = self.U(z_s)
        v = self.V(z_c)
        q = torch.tanh(u)
        k = torch.tanh(v)
        h = torch.tanh((u + v) / (2.0 ** 0.5))
        w = self.w.detach() if detach_w else self.w
        t = (h * w.view(1, -1)).sum(dim=1) / (self.config.latent ** 0.5)
        return a, b, q, k, t


def build_model(dims: Mapping[str, Any], config: RM.ModelConfig,
                state: Mapping[str, torch.Tensor]) -> RM.ResponseModel:
    """``mode`` selects the fusion operator; parameter set is identical for all modes."""
    if config.mode == "JOINT":
        model = JointResponseModel(dims, config)
    elif config.mode in ("RESP", "ADD"):
        model = RM.ResponseModel(dims, config)
    else:
        raise ValueError(f"unknown mode {config.mode!r}")
    model.load_state_dict({k: v.clone() for k, v in state.items()}, strict=True)
    return model


def model_config(config: Mapping[str, Any], mode: str) -> RM.ModelConfig:
    return RM.ModelConfig(
        hidden=int(config["hidden"]),
        latent=int(config["latent"]),
        n_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
        mode=mode,
    )


def load_canonical_state(path, expected_hash: str):
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items()}
    got = RM.state_hash(state)
    if got != expected_hash:
        raise RuntimeError(f"canonical init state hash {got} != expected {expected_hash}")
    return state
