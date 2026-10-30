"""v2 models: capacity-matched E, LCRC-A, GINE and the dropout-at-h hook.

Capacity arithmetic (this is the central audit of Workstream A)
---------------------------------------------------------------
A ``lipo_v1`` LCRC-A has 247,106 parameters.  Its LocalMLP input is 529-wide::

    [a(96) | b_sum(96) | b_mean(96) | S1joint(32) | S2(200) | scalars(9)]

The old S2-off arm B zeroes the 200-wide S2 slot in train and eval.  It
therefore loses 32,192 typed-S2 module parameters *and* makes the
``lins[0].weight[:, 320:520]`` block (38,400 parameters) dead: 70,592 lost
effective parameters, 176,514 effective parameters remaining.

The task book specifies a two-layer S1-context replacement with::

    320*135 + 135 + 135*200 + 200 = 70,535

and a target effective count of ``176,514 + 70,535 = 247,049`` (A minus 57).
The literal placement of that 200-wide output into the original 529-wide
LocalMLP input would make the 38,400 S2 columns live again, giving 285,449
*live* parameters, not 247,049.  The task book's arithmetic is only consistent
if the 200-wide S2 input slot remains dead.

This module therefore implements the capacity-consistent realization:

* the original LocalMLP module (529 -> 192 -> 64 + LayerNorm) is unchanged;
* its 200-wide S2 input slot is zeroed, exactly as in B, so the 38,400 columns
  stay dead and the effective budget is B + the new network;
* the new two-layer network consumes the 320-wide S1 context and its 200-wide
  output is injected as a parameter-free fixed residual into the 192-wide
  LocalMLP hidden state (the last 8 channels are folded into the first 8).

The residual map has no trainable parameters, so every one of the 70,535 new
parameters receives a gradient from the supervised loss.  This is recorded as
an implementation resolution of the task-book arithmetic, not as a new
scientific mechanism.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import v2_common as C

prep_model = C.prep_model
V1GINE = C.v1m.GINERegressor


# ---------------------------------------------------------------------------
# capacity-matched local content
# ---------------------------------------------------------------------------


def _fixed_project_200_to_192(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free 200 -> 192 map that uses all 200 input channels.

    The first 192 channels pass through; the last 8 are folded into the first
    8 outputs.  Every second-layer output row therefore receives gradient.
    """
    if x.shape[-1] != 200:
        raise ValueError(f"expected 200 replacement channels, got {tuple(x.shape)}")
    out = x[..., :192]
    return out + torch.cat(
        [x[..., 192:200], x.new_zeros(x.shape[:-1] + (184,))], dim=-1
    )


class LocalContentCapacityMatched(nn.Module):
    """S1 context + two-layer replacement; typed-S2 tensors are never read."""

    def __init__(self, spec, cfg: Mapping[str, Any]):
        super().__init__()
        self.spec = spec
        rank = int(cfg["stat_rank"])
        a_out = int(cfg["atom_out"])
        self.atom_out = a_out
        self.rank = rank
        self.nbr_emb = nn.Embedding(spec.n_nbr_codes, rank)
        self.replacement = nn.Sequential(
            nn.Linear(3 * a_out + rank, 135),
            nn.SiLU(),
            nn.Linear(135, 3 * rank + 8 + a_out),
        )
        in_dim = 3 * a_out + rank + (3 * rank + 8 + a_out) + spec.n_scalars
        self.mlp = prep_model.MLP([in_dim, int(cfg["local_hidden"]), int(cfg["width"])])
        self.norm = nn.LayerNorm(int(cfg["width"]))
        self.replacement_out = 3 * rank + 8 + a_out
        self.effective_dead_inputs = self.replacement_out

    def _s1_context(self, batch, a, b):
        n_atoms = a.shape[0]
        width = a.shape[1]
        bond_index = batch["bond_index"]
        if bond_index.numel():
            u, v = bond_index[:, 0], bond_index[:, 1]
            degree = a.new_zeros(n_atoms)
            degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
            degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
            b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
        else:
            degree = a.new_zeros(n_atoms)
            b_sum = a.new_zeros((n_atoms, width))
        b_mean = b_sum / degree.clamp(min=1.0).unsqueeze(1)
        if batch["nbr_code"].numel():
            nbr = a.new_zeros((n_atoms, self.rank)).index_add_(
                0, batch["nbr_atom"], self.nbr_emb(batch["nbr_code"])
            )
        else:
            nbr = a.new_zeros((n_atoms, self.rank))
        return b_sum, b_mean, nbr

    def forward(self, batch, a, b, scal):
        # Deliberately no access to path_cat/path_end63/path_end/path_closure.
        b_sum, b_mean, nbr = self._s1_context(batch, a, b)
        ctx = torch.cat([a, b_sum, b_mean, nbr], dim=1)
        replacement = self.replacement(ctx)
        # The original 200-wide S2 slot stays zero (B's effective-capacity
        # accounting); the replacement enters the LocalMLP hidden instead.
        zeros = a.new_zeros((a.shape[0], self.replacement_out))
        x = torch.cat([a, b_sum, b_mean, nbr, zeros, scal], dim=1)
        hidden = F.silu(self.mlp.lins[0](x))
        hidden = hidden + _fixed_project_200_to_192(replacement)
        hidden = self.mlp.lins[1](hidden)
        return self.norm(hidden)


class LCRCCapacityMatched(prep_model.LCRC):
    """LCRC-A with the typed-S2 branch replaced by the S1-context network."""

    def __init__(self, spec, norm: Mapping[str, Any], cfg: Mapping[str, Any] | None = None):
        super().__init__(spec, norm, cfg)
        self.local = LocalContentCapacityMatched(spec, self.cfg)


# ---------------------------------------------------------------------------
# dropout at the final atom-level local representation
# ---------------------------------------------------------------------------


class DropoutAtLocal:
    """Apply dropout to the output of the local encoder, before readout.

    For A/E the local encoder is ``model.local``; for C it is the last GINE
    LayerNorm.  The dropout module is registered on the model so ``train()`` and
    ``eval()`` propagate.  ``p=0`` is the identity, so R0 is numerically
    equivalent to the original no-dropout forward.
    """

    def __init__(self, model: nn.Module, arm: str, p: float):
        if p < 0 or p >= 1:
            raise ValueError("dropout p must be in [0, 1)")
        self.arm = arm
        self.module = nn.Dropout(float(p))
        model.add_module("_v2_dropout_head", self.module)
        if arm in ("A", "E"):
            target = model.local
        elif arm == "C":
            target = model.norms[-1]
        else:
            raise ValueError(f"unknown arm {arm!r}")
        self.target = target
        self.handle = target.register_forward_hook(self._hook)
        self.calls = 0

    def _hook(self, module, inputs, output):
        self.calls += 1
        if not torch.is_tensor(output):
            raise TypeError("dropout hook expected a tensor output")
        return self.module(output)

    def remove(self) -> None:
        self.handle.remove()


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_a(spec, norm, seed: int, device="cpu"):
    torch.manual_seed(int(seed))
    return prep_model.LCRC(spec, norm, prep_model.DEFAULT_CONFIG).to(device)


def build_e(spec, norm, seed: int, device="cpu"):
    """Build E with shared-module init bit-identical to A for the same seed."""
    torch.manual_seed(int(seed))
    base = prep_model.LCRC(spec, norm, prep_model.DEFAULT_CONFIG)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    torch.manual_seed(int(seed))
    model = LCRCCapacityMatched(spec, norm, prep_model.DEFAULT_CONFIG)
    own = model.state_dict()
    shared = {}
    for key, value in base_state.items():
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            shared[key] = value
    model.load_state_dict({**own, **shared}, strict=True)
    model.to(device)
    return model, base, shared


def build_c(spec, norm, seed: int, device="cpu"):
    torch.manual_seed(int(seed))
    return V1GINE(spec, norm, hidden=C.v1c.GINE_HIDDEN, layers=C.v1c.GINE_LAYERS).to(device)


def build_model(arm: str, spec, norm, seed: int, device="cpu", *, dropout: float = 0.0):
    if arm == "A":
        model = build_a(spec, norm, seed, device)
        shared = None
    elif arm == "E":
        model, base, shared = build_e(spec, norm, seed, device)
    elif arm == "C":
        model = build_c(spec, norm, seed, device)
        shared = None
    else:
        raise ValueError(f"unknown arm {arm!r}")
    hook = DropoutAtLocal(model, arm, dropout)
    return model, {"arm": arm, "dropout": float(dropout), "dropout_hook": hook, "shared_with_A": shared}


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def module_breakdown(model: nn.Module) -> dict[str, int]:
    return {name: int(sum(p.numel() for p in mod.parameters())) for name, mod in model.named_children() if name != "_v2_dropout_head"}


def dead_parameter_names_e(model: "LCRCCapacityMatched") -> set[str]:
    return {"local.mlp.lins.0.weight"}


def effective_parameters_e(model: "LCRCCapacityMatched") -> int:
    """Nominal minus the 200 input columns of the first LocalMLP layer."""
    dead = int(model.local.mlp.lins[0].weight[:, 320:520].numel())
    return int(parameter_count(model) - dead)


def e_capacity_audit(model: "LCRCCapacityMatched", spec) -> dict[str, Any]:
    total = parameter_count(model)
    dead = int(model.local.mlp.lins[0].weight[:, 320:520].numel())
    replacement = int(sum(p.numel() for p in model.local.replacement.parameters()))
    a_reference = int(C.v1c.LCRC_PARAMS_REFERENCE)
    b_effective = 176_514
    return {
        "nominal_parameters": total,
        "a_reference_parameters": a_reference,
        "dead_localmlp_s2_input_columns": dead,
        "effective_parameters": total - dead,
        "delta_effective_vs_A": (total - dead) - a_reference,
        "delta_fraction_vs_A": ((total - dead) - a_reference) / a_reference,
        "replacement_parameters": replacement,
        "old_B_effective_parameters": b_effective,
        "effective_equals_B_plus_replacement": (total - dead) == b_effective + replacement,
        "literal_529_input_live_parameters_if_slot_not_dead": total,
        "wiring": "zeroed original S2 slot + parameter-free 200->192 fixed residual into LocalMLP hidden",
        "dead_parameter_names": sorted(dead_parameter_names_e(model)),
        "typed_s2_inputs_used": False,
        "iterative_hidden_message_passing": False,
    }


def state_dict_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def model_state_hash(model: nn.Module) -> str:
    return state_dict_fingerprint(model.state_dict())


def shared_init_audit(base: prep_model.LCRC, model: "LCRCCapacityMatched") -> dict[str, Any]:
    a_state = base.state_dict()
    e_state = model.state_dict()
    shared = []
    mismatched = []
    for key, value in a_state.items():
        if key in e_state and tuple(e_state[key].shape) == tuple(value.shape):
            same = torch.equal(value, e_state[key])
            shared.append({"key": key, "same": bool(same)})
            if not same:
                mismatched.append(key)
    return {
        "n_shared_entries": len(shared),
        "all_shared_bitwise_identical": not mismatched,
        "mismatched": mismatched,
        "shared_keys": [x["key"] for x in shared],
        "e_only_keys": sorted(set(e_state) - set(a_state)),
        "a_only_keys": sorted(set(a_state) - set(e_state)),
    }


# ---------------------------------------------------------------------------
# restricted batch / E invariance helpers
# ---------------------------------------------------------------------------

TYPED_S2_KEYS = ("path_cat", "path_end63", "path_end", "path_closure", "path_graph", "path_atom")


class RestrictedBatch(dict):
    """A batch dict whose access to typed-S2 keys raises."""

    def __getitem__(self, key):
        if key in TYPED_S2_KEYS:
            raise KeyError(f"E must not read typed-S2 tensor {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in TYPED_S2_KEYS:
            raise KeyError(f"E must not read typed-S2 tensor {key!r}")
        return super().get(key, default)


def restricted_batch(batch: Mapping[str, Any]) -> RestrictedBatch:
    return RestrictedBatch({k: v for k, v in batch.items() if k != "atom_type"})


__all__ = [
    "LocalContentCapacityMatched",
    "LCRCCapacityMatched",
    "DropoutAtLocal",
    "build_a",
    "build_e",
    "build_c",
    "build_model",
    "parameter_count",
    "module_breakdown",
    "effective_parameters_e",
    "e_capacity_audit",
    "dead_parameter_names_e",
    "state_dict_fingerprint",
    "model_state_hash",
    "shared_init_audit",
    "restricted_batch",
    "RestrictedBatch",
    "TYPED_S2_KEYS",
]
