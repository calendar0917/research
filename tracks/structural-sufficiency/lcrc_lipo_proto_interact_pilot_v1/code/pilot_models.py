"""The two new readouts on top of the frozen LCRC-A graph.

BASE
    ``lcrc_model.LCRC`` with the frozen ``lipo_v1`` spec and the untouched
    64-wide local representation.  Prediction stays
    ``y = sum_v f(h_v) + rho(G)``; only the v2 ``R3`` dropout (0.10, applied to
    the final atom-level ``h``, i.e. the output of ``model.local``) is added, in
    exactly the position the v2 round used.

PROTO (``PROTO_K16_TOP4``)
    ``BASE`` with ``self.f`` replaced by a shared-prototype readout: 16 learned
    prototype keys ``D``, 16 shared local linear experts ``(W, b)``, cosine
    routing at temperature 0.25 restricted to the top-4 prototypes.  No MLP
    residual bypass, no auxiliary loss.  Every other module is bit-identical to
    ``BASE`` at the same seed.

PAIR
    ``BASE`` plus a graph-level residual ``delta(G)`` built from the mean
    product of a 16-d tanh projection of ``h`` over atom pairs at graph
    distance 3, 4 and 5.  The original ``f`` and ``rho`` are kept intact.
    ``PAIR_TRUE`` and ``PAIR_NULL`` share this module; they differ only in the
    distance-class labels supplied by the batch (see :mod:`pilot_pairs`).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import pilot_common as C

prep_model = C.prep_model


# ---------------------------------------------------------------------------
# dropout at the final atom-level local representation (v2 R3 position)
# ---------------------------------------------------------------------------


def attach_dropout(model: nn.Module, p: float) -> Any:
    """Register the v2 ``DropoutAtLocal`` hook on ``model.local``."""
    return C.V2M.DropoutAtLocal(model, "A", float(p))


# ---------------------------------------------------------------------------
# PROTO readout
# ---------------------------------------------------------------------------


class ProtoReadout(nn.Module):
    """Shared-prototype local contribution ``f_proto(h_v)``.

    ``s_vk = cos(h_v, D_k) / tau``; only the top-``topk`` logits survive,
    everything else is masked to ``-inf`` before the softmax.  The atom
    contribution is the routing-weighted mixture of the shared linear experts
    ``e_vk = W_k . h_v + b_k``.  There is no residual MLP: the prototypes carry
    the whole atom contribution.
    """

    def __init__(self, width: int, n_prototypes: int, topk: int, tau: float):
        super().__init__()
        if topk > n_prototypes:
            raise ValueError("topk must not exceed the number of prototypes")
        self.width = int(width)
        self.n_prototypes = int(n_prototypes)
        self.topk = int(topk)
        self.tau = float(tau)
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        self.prototypes = nn.Parameter(torch.empty(self.n_prototypes, self.width))
        self.experts = nn.Parameter(torch.empty(self.n_prototypes, self.width))
        self.bias = nn.Parameter(torch.zeros(self.n_prototypes))
        # Plain default-style initialisation (no tuning): random unit-scale-free
        # key directions, Linear-default expert weights, zero biases.
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        bound = 1.0 / (self.width ** 0.5)
        nn.init.uniform_(self.experts, -bound, bound)

    def route(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the masked soft routing weights and the top-k indices."""
        h_hat = F.normalize(h, dim=-1)
        d_hat = F.normalize(self.prototypes, dim=-1)
        similarity = h_hat @ d_hat.t() / self.tau
        top_idx = similarity.topk(self.topk, dim=-1).indices
        masked = torch.full_like(similarity, float("-inf"))
        masked.scatter_(1, top_idx, similarity.gather(1, top_idx))
        weights = torch.softmax(masked, dim=-1)
        return weights, top_idx

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights, _ = self.route(h)
        experts = h @ self.experts.t() + self.bias
        return (weights * experts).sum(dim=-1, keepdim=True)


class ProtoLCRC(prep_model.LCRC):
    """``LCRC`` whose atom contribution is the shared-prototype readout."""

    def __init__(
        self,
        spec,
        norm: Mapping[str, Any],
        cfg: Mapping[str, Any] | None = None,
        *,
        n_prototypes: int = C.N_PROTOTYPES,
        topk: int = C.PROTO_TOPK,
        tau: float = C.PROTO_TAU,
    ):
        super().__init__(spec, norm, cfg)
        self.f = ProtoReadout(int(self.cfg["width"]), n_prototypes, topk, tau)


# ---------------------------------------------------------------------------
# PAIR residual
# ---------------------------------------------------------------------------


class PairLCRC(prep_model.LCRC):
    """``LCRC`` plus a distance-binned mid-range atom-pair residual."""

    def __init__(
        self,
        spec,
        norm: Mapping[str, Any],
        cfg: Mapping[str, Any] | None = None,
        *,
        z_dim: int = C.PAIR_Z_DIM,
        hidden: int = C.PAIR_HIDDEN,
        distances: tuple[int, ...] = C.PAIR_DISTANCES,
    ):
        super().__init__(spec, norm, cfg)
        width = int(self.cfg["width"])
        self.z_dim = int(z_dim)
        self.distances = tuple(int(d) for d in distances)
        self.pair_z = nn.Linear(width, self.z_dim)
        self.pair_head = nn.Sequential(
            nn.Linear(len(self.distances) * self.z_dim, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), 1),
        )
        last = self.pair_head[2]
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last.bias)

    def pair_moments(self, h: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``M[g, d, r] = mean_{u<v, dist=d} z_ur z_vr`` (strict zero when empty)."""
        n_graphs = int(batch["n_graphs"])
        z = torch.tanh(self.pair_z(h))
        u, v = batch["pair_u"], batch["pair_v"]
        graph, pair_bin = batch["pair_graph"], batch["pair_bin"]
        blocks = []
        for k in range(len(self.distances)):
            mask = pair_bin == k
            if int(mask.numel()) and bool(mask.any()):
                prod = z.index_select(0, u[mask]) * z.index_select(0, v[mask])
                acc = z.new_zeros((n_graphs, self.z_dim)).index_add_(0, graph[mask], prod)
                count = z.new_zeros(n_graphs).index_add_(
                    0, graph[mask], torch.ones_like(graph[mask], dtype=z.dtype)
                )
                mean = acc / count.clamp(min=1.0).unsqueeze(1)
                blocks.append(torch.where((count > 0).unsqueeze(1), mean, torch.zeros_like(mean)))
            else:
                blocks.append(z.new_zeros((n_graphs, self.z_dim)))
        return torch.cat(blocks, dim=1)

    def delta(self, h: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.pair_head(self.pair_moments(h, batch)).view(-1)

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        y_base, aux = super().forward(batch, return_aux=True)
        delta = self.delta(aux["h"], batch)
        y = y_base + delta
        aux["y_base"] = y_base
        aux["delta"] = delta
        aux["y"] = y
        if return_aux:
            return y, aux
        return y


# ---------------------------------------------------------------------------
# builders (shared-initialisation discipline)
# ---------------------------------------------------------------------------


def build_base(spec, norm, seed: int, device="cpu"):
    """Exactly the v2 ``build_a`` initialisation for seed 0."""
    model = C.V2M.build_a(spec, norm, int(seed), device)
    return model


def build_proto(spec, norm, seed: int, device="cpu"):
    """PROTO with every shared tensor bit-identical to BASE at the same seed."""
    torch.manual_seed(int(seed))
    base = prep_model.LCRC(spec, norm, prep_model.DEFAULT_CONFIG)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    torch.manual_seed(int(seed))
    model = ProtoLCRC(spec, norm, prep_model.DEFAULT_CONFIG)
    own = model.state_dict()
    shared = {
        key: value
        for key, value in base_state.items()
        if key in own and tuple(own[key].shape) == tuple(value.shape)
    }
    model.load_state_dict({**own, **shared}, strict=True)
    model.to(device)
    return model, base


def build_pair(spec, norm, seed: int, device="cpu"):
    torch.manual_seed(int(seed))
    model = PairLCRC(spec, norm, prep_model.DEFAULT_CONFIG)
    model.to(device)
    return model


def build_arm(arm: str, spec, norm, seed: int, device="cpu", *, dropout: float = C.DROPOUT):
    """Build one arm and attach the v2 R3 dropout hook."""
    if arm in C.BASE_ARMS:
        model = build_base(spec, norm, seed, device)
        extra = {}
    elif arm in C.PROTO_ARMS:
        model, base = build_proto(spec, norm, seed, device)
        extra = {"base_reference": base}
    elif arm in C.PAIR_ARMS:
        model = build_pair(spec, norm, seed, device)
        extra = {}
    else:
        raise ValueError(f"unknown arm {arm!r}")
    hook = attach_dropout(model, dropout)
    return model, {"arm": arm, "dropout": float(dropout), "hook": hook, **extra}


# ---------------------------------------------------------------------------
# parameter / initialisation audits
# ---------------------------------------------------------------------------

MODULE_ORDER = ("atom_encoder", "bond_encoder", "local", "f", "rho", "pair_z", "pair_head")


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def module_breakdown(model: nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, child in model.named_children():
        if name == "_v2_dropout_head":
            continue
        out[name] = parameter_count(child)
    out["total"] = parameter_count(model)
    return out


def shared_init_audit(base: nn.Module, model: nn.Module) -> dict[str, Any]:
    a_state = base.state_dict()
    b_state = model.state_dict()
    shared_keys, mismatched = [], []
    for key, value in a_state.items():
        if key in b_state and tuple(b_state[key].shape) == tuple(value.shape):
            same = torch.equal(value, b_state[key])
            shared_keys.append(key)
            if not same:
                mismatched.append(key)
    return {
        "n_shared_entries": len(shared_keys),
        "all_shared_bitwise_identical": not mismatched,
        "mismatched": mismatched,
        "shared_keys": shared_keys,
        "model_only_keys": sorted(set(b_state) - set(a_state)),
        "base_only_keys": sorted(set(a_state) - set(b_state)),
    }


def pair_state_equality(a: nn.Module, b: nn.Module) -> dict[str, Any]:
    sa, sb = a.state_dict(), b.state_dict()
    if sorted(sa) != sorted(sb):
        return {"same_keys": False, "identical": False, "mismatched": sorted(set(sa) ^ set(sb))}
    mismatched = [k for k in sa if not torch.equal(sa[k], sb[k])]
    return {
        "same_keys": True,
        "identical": not mismatched,
        "mismatched": mismatched,
        "n_tensors": len(sa),
        "parameter_count_a": parameter_count(a),
        "parameter_count_b": parameter_count(b),
    }


def proto_stats(module: ProtoReadout) -> dict[str, torch.Tensor]:
    return {
        "prototypes": module.prototypes.detach().clone(),
        "experts": module.experts.detach().clone(),
        "bias": module.bias.detach().clone(),
    }


__all__ = [
    "attach_dropout",
    "ProtoReadout",
    "ProtoLCRC",
    "PairLCRC",
    "build_base",
    "build_proto",
    "build_pair",
    "build_arm",
    "parameter_count",
    "module_breakdown",
    "shared_init_audit",
    "pair_state_equality",
    "proto_stats",
]
