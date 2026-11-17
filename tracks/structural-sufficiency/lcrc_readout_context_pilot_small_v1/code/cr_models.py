"""The three readouts of this round and the parameter-budget rule.

Common local encoder (frozen, read-only semantics)
--------------------------------------------------
``C1`` = the repository ``NO_BOND_MEAN`` LCRC local encoder: the 96-d
``b_mean`` input block is deleted, everything else (atom/bond encoders, ``b_sum``,
``S1joint``, typed ``S2`` type/end/joint/closure/content, the nine atom scalars,
``LocalMLP`` 433->192->64, ``LayerNorm``) is unchanged.  ``h_v`` is the resulting
64-d local representation, and **one** ``dropout(0.10)`` call on ``h`` is shared
by every local head of the arm.

Readouts
--------
A  ``sum_v f(h_v) + rho([z_g, mean_v h_v, mean_v h_v^2, log1p(n_atoms)])``
   (the original C1 readout, re-used unchanged: ``f`` 4225, ``rho`` 11073)

B  ``b_global + sum_v c0(h_v) + sum_{j in J} u_j(z_j)``

C  ``b_global + sum_v c0(h_v) + sum_{j in J} u_j(z_j)
              + a_size(z_size) * sum_v c_size(h_v)
              + a_ring(z_ring) * sum_v c_ring(h_v)``

``J`` is the Train-only set of global columns whose raw span exceeds ``1e-8``
(``M = |J|``); every other column is fixed to exactly zero in B/C.  Each single
variable ``u_j`` / ``a_k`` is the frozen 48-parameter form

    q(z) = w2 . [ SiLU(w1 * z + b1) - SiLU(b1) ]      (w1, b1, w2 in R^16)

so ``q(0) = 0`` by construction and there is **no output bias at all** (the
centring is exact and cannot absorb a dead bias).

Parameter budget
----------------
    P_B(H) = 66 * H + 48 * M + 2
    P_C(H) = 68 * H + 48 * M + 98

and ``H`` is the positive integer in ``1..1024`` minimising
``|P_x(H) - 15298|`` (ties -> smaller ``H``).  That is an analytic budget match,
never a width search: no arm is compared at a second width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cr_common as C
import cr_data as D

MOD = D.modules()
DM = MOD["dissect_model"]

WIDTH = 64                     # frozen local representation h_v
STAT_RANK = 32
LOCAL_HIDDEN = 192
ATOM_HIDDEN = 128
ATOM_OUT = 96
BOND_HIDDEN = 160
BOND_OUT = 96
RHO_HIDDEN = 64
DROPOUT = 0.10

#: Frozen original readout budget that B and C must match as closely as possible.
READOUT_BUDGET = 15298
F_PARAMS = 4225                # MLP 64 -> 64 -> 1
RHO_PARAMS = 11073             # MLP 171 -> 64 -> 1
TRUNK_PARAMS = 213376          # C1 minus f minus rho
FULL_C1_PARAMS = 228674        # trunk + f + rho
SINGLE_WIDTH = 16              # w1/b1/w2 of one single-variable function
SINGLE_PARAMS = 3 * SINGLE_WIDTH
HIDDEN_SEARCH_MAX = 1024

#: Parameter counts frozen by the protocol (a mismatch is an implementation bug).
EXPECTED_COUNTS = {
    "full_c1": FULL_C1_PARAMS,
    "trunk": TRUNK_PARAMS,
    "readout_A": F_PARAMS + RHO_PARAMS,
    "f": F_PARAMS,
    "rho": RHO_PARAMS,
}

TRUNK_PREFIXES = ("atom_encoder.", "bond_encoder.", "local.")
TRUNK_BUFFERS = ("global_mean", "global_scale", "scalar_mean", "scalar_scale")
#: Every arm nests the shared local encoder under one submodule.
ARM_TRUNK_PREFIX = "trunk."


# ---------------------------------------------------------------------------
# budget arithmetic
# ---------------------------------------------------------------------------


def readout_params(arm: str, hidden: int, n_global: int) -> int:
    if arm == "A":
        return F_PARAMS + RHO_PARAMS
    if arm == "B":
        return 66 * int(hidden) + 48 * int(n_global) + 2
    if arm == "C":
        return 68 * int(hidden) + 48 * int(n_global) + 98
    raise KeyError(arm)


def choose_hidden(arm: str, n_global: int, budget: int = READOUT_BUDGET) -> dict[str, Any]:
    """Analytic budget match: smallest-|P(H)-budget| H in 1..1024, ties -> smaller H."""
    if arm not in ("B", "C"):
        raise KeyError(arm)
    best = None
    for hidden in range(1, HIDDEN_SEARCH_MAX + 1):
        params = readout_params(arm, hidden, n_global)
        delta = abs(params - budget)
        if best is None or delta < best[1]:
            best = (hidden, delta, params)
    hidden, delta, params = best
    return {
        "arm": arm,
        "n_active_global": int(n_global),
        "hidden": int(hidden),
        "nominal_readout_params": int(params),
        "budget": int(budget),
        "abs_delta": int(delta),
        "relative_delta": float(delta / budget),
        "formula": ("66*H + 48*M + 2" if arm == "B" else "68*H + 48*M + 98"),
        "search": "analytic argmin over H in 1..1024, ties to the smaller H (no fitting, no validation)",
    }


# ---------------------------------------------------------------------------
# canonical initial state
# ---------------------------------------------------------------------------


def canonical_c1_state(spec, norm: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """The one canonical C1 initial state of this round (CPU, seed before construction)."""
    C.set_global_seeds(C.INIT_SEED_CANONICAL)
    model = DM.DissectLCRC(spec, norm, "C1")
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def trunk_keys(state: Mapping[str, Any]) -> list[str]:
    keys = [k for k in state if k.startswith(TRUNK_PREFIXES)]
    keys += [k for k in TRUNK_BUFFERS if k in state]
    return sorted(keys)


def state_hashes(state: Mapping[str, Any]) -> dict[str, str]:
    keys = trunk_keys(state)
    return {
        "full_state_hash": C.sha256_state(state),
        "trunk_hash": C.sha256_state(state, keys=keys),
        "trunk_keys": keys,
        "readout_keys": sorted(k for k in state if k not in set(keys)),
    }


# ---------------------------------------------------------------------------
# isolated initialisation streams
# ---------------------------------------------------------------------------


def _stream(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _uniform(shape: Sequence[int], bound: float, generator: torch.Generator) -> torch.Tensor:
    tensor = torch.empty(*[int(s) for s in shape], dtype=torch.float32)
    nn.init.uniform_(tensor, -float(bound), float(bound), generator=generator)
    return tensor


def _linear_weight(out_features: int, in_features: int, generator: torch.Generator) -> torch.Tensor:
    """``nn.Linear`` default weight init (kaiming_uniform, a=sqrt(5)) on a fixed stream."""
    weight = torch.empty(int(out_features), int(in_features), dtype=torch.float32)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
    return weight


def _linear_bias(out_features: int, fan_in: int, generator: torch.Generator) -> torch.Tensor:
    """``nn.Linear`` default bias init (U(-1/sqrt(fan_in), 1/sqrt(fan_in)))."""
    bound = 1.0 / math.sqrt(int(fan_in)) if fan_in > 0 else 0.0
    bias = torch.empty(int(out_features), dtype=torch.float32)
    nn.init.uniform_(bias, -bound, bound, generator=generator)
    return bias


# ---------------------------------------------------------------------------
# the single-variable function q(z)
# ---------------------------------------------------------------------------


class SingleVariableFunctions(nn.Module):
    """``n`` independent 48-parameter functions ``q(z) = w2 . [SiLU(w1 z + b1) - SiLU(b1)]``.

    Initialisation follows the PyTorch default of the two-layer ``1 -> 16 -> 1``
    MLP each function stands for: ``w1``/``b1`` are the ``Linear(1, 16)``
    weight/bias (fan-in 1, uniform bound 1.0), ``w2`` is the bias-free
    ``Linear(16, 1)`` weight (fan-in 16, uniform bound 0.25).  Draw order is
    ``w1, b1, w2`` on the caller's isolated stream.
    """

    def __init__(self, n_functions: int, *, generator: torch.Generator | None = None) -> None:
        super().__init__()
        n = int(n_functions)
        self.n_functions = n
        g = generator if generator is not None else _stream(0)
        self.w1 = nn.Parameter(_uniform((n, SINGLE_WIDTH), 1.0, g))
        self.b1 = nn.Parameter(_uniform((n, SINGLE_WIDTH), 1.0, g))
        self.w2 = nn.Parameter(_uniform((n, SINGLE_WIDTH), 1.0 / math.sqrt(SINGLE_WIDTH), g))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: ``(batch, n)`` -> ``(batch, n)`` function values."""
        if z.dim() != 2 or z.shape[1] != self.n_functions:
            raise ValueError(f"expected z of shape (B, {self.n_functions}), got {tuple(z.shape)}")
        hidden = F.silu(self.w1.unsqueeze(0) * z.unsqueeze(-1) + self.b1.unsqueeze(0))
        centred = hidden - F.silu(self.b1).unsqueeze(0)
        return (centred * self.w2.unsqueeze(0)).sum(dim=-1)

    def scalar_loop(self, z: torch.Tensor) -> torch.Tensor:
        """Explicit per-function Python loop (used to cross-check the vectorised path)."""
        out = []
        for j in range(self.n_functions):
            hidden = F.silu(float(1.0) * z[:, j : j + 1] * self.w1[j].unsqueeze(0) + self.b1[j].unsqueeze(0))
            centred = hidden - F.silu(self.b1[j]).unsqueeze(0)
            out.append((centred * self.w2[j].unsqueeze(0)).sum(dim=-1, keepdim=True))
        return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# common local trunk (bitwise the C1 local path)
# ---------------------------------------------------------------------------


class LocalTrunk(nn.Module):
    """``C1`` local encoder -> ``h_v`` with one shared dropout call."""

    def __init__(self, spec, norm: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.cfg = {
            "width": WIDTH,
            "stat_rank": STAT_RANK,
            "local_hidden": LOCAL_HIDDEN,
            "atom_hidden": ATOM_HIDDEN,
            "atom_out": ATOM_OUT,
            "bond_hidden": BOND_HIDDEN,
            "bond_out": BOND_OUT,
            "rho_hidden": RHO_HIDDEN,
        }
        if cfg:
            self.cfg.update(dict(cfg))
        # Creation order matches ``DissectLCRC`` so the canonical state transfers.
        self.atom_encoder = DM.MLP([spec.atom_feat_dim, int(self.cfg["atom_hidden"]), int(self.cfg["atom_out"])])
        self.bond_encoder = DM.MLP(
            [3 * int(self.cfg["atom_out"]) + spec.bond_feat_dim, int(self.cfg["bond_hidden"]), int(self.cfg["bond_out"])]
        )
        self.local = DM.LocalContent(spec, "C1", self.cfg)
        self.drop = nn.Dropout(DROPOUT)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))

    # -- state transfer ----------------------------------------------------
    def load_canonical(self, state: Mapping[str, Any]) -> None:
        subset = {k: state[k] for k in trunk_keys(state) if k in self.state_dict()}
        missing = set(self.state_dict()) - set(subset)
        if missing:
            raise RuntimeError(f"trunk state is missing keys: {sorted(missing)}")
        self.load_state_dict({k: v.clone() for k, v in subset.items()}, strict=True)

    # -- forward -----------------------------------------------------------
    def encode(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        else:
            bond_input = torch.zeros(
                (0, 3 * a.shape[1] + batch["bond_feat"].shape[1]), dtype=a.dtype, device=a.device
            )
        b = self.bond_encoder(bond_input)
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        return a, b, scal

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b, scal = self.encode(batch)
        h, _aux = self.local(batch, a, b, scal)
        return self.drop(h)  # exactly one dropout call, shared by every head

    # -- standardised global inputs ---------------------------------------
    def z_global(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (batch["global"] - self.global_mean) / self.global_scale


def _counts(atom_batch: torch.Tensor, n_graphs: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return atom_batch.new_zeros(n_graphs, dtype=dtype).index_add_(
        0, atom_batch, torch.ones_like(atom_batch, dtype=dtype)
    )


def _sum_per_graph(values: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    return values.new_zeros(n_graphs).index_add_(0, atom_batch, values)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


class ArmA(nn.Module):
    """The original C1 readout, re-implemented (checked against the frozen module)."""

    name = "A"

    def __init__(self, spec, norm: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.trunk = LocalTrunk(spec, norm, cfg)
        self.f = DM.MLP([WIDTH, WIDTH, 1])
        self.rho_dim = int(spec.global_dim) + 2 * WIDTH + 1
        self.rho = DM.MLP([self.rho_dim, RHO_HIDDEN, 1])

    def load_canonical(self, state: Mapping[str, Any]) -> None:
        self.trunk.load_canonical(state)
        for prefix, module in (("f.", self.f), ("rho.", self.rho)):
            subset = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            module.load_state_dict({k: v.clone() for k, v in subset.items()}, strict=True)

    def parts(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        h = self.trunk(batch)
        count = _counts(atom_batch, n_graphs, dtype=h.dtype)
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, WIDTH)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, WIDTH)).index_add_(0, atom_batch, h * h) / denom
        contrib_atom = self.f(h).view(-1)
        contrib = contrib_atom.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        g = self.trunk.z_global(batch)
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        if rho_in.shape[1] != self.rho.lins[0].in_features:
            raise RuntimeError(f"rho width {rho_in.shape[1]} != {self.rho.lins[0].in_features}")
        rho = self.rho(rho_in).view(-1)
        return {
            "pred_std": contrib + rho,
            "h": h,
            "count": count,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "contrib_atom": contrib_atom,
            "local_sum": contrib,
            "rho": rho,
            "rho_input": rho_in,
            "term_names": ["local_sum", "rho"],
        }

    def forward(self, batch, *, return_aux: bool = False):
        parts = self.parts(batch)
        return (parts["pred_std"], parts) if return_aux else parts["pred_std"]


class _AdditiveBase(nn.Module):
    """Shared plumbing for B and C: trunk, active-global mask, additive terms."""

    def __init__(
        self,
        spec,
        norm: Mapping[str, Any],
        *,
        arm: str,
        hidden: int,
        active_indices: Sequence[int],
        context_indices: Sequence[int],
        cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.name = arm
        self.active_indices = tuple(int(i) for i in active_indices)
        self.context_indices = tuple(int(i) for i in context_indices)
        self.n_active = len(self.active_indices)
        self.hidden = int(hidden)
        self.trunk = LocalTrunk(spec, norm, cfg)

        local_gen = _stream(C.INIT_STREAM_LOCAL)
        # Fixed draw order on stream 1003: hidden layer, local output head,
        # then (arm C only) the two bias-free context heads.
        self.w_hidden = nn.Parameter(_linear_weight(self.hidden, WIDTH, local_gen))
        self.b_hidden = nn.Parameter(_linear_bias(self.hidden, WIDTH, local_gen))
        self.w_base = nn.Parameter(_linear_weight(1, self.hidden, local_gen))
        self.b_local = nn.Parameter(_linear_bias(1, self.hidden, local_gen))
        if arm == "C":
            self.w_context = nn.Parameter(_linear_weight(len(C.CONTEXT_FIELDS), self.hidden, local_gen))

        self.global_functions = SingleVariableFunctions(self.n_active, generator=_stream(C.INIT_STREAM_GLOBAL))
        self.b_global = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    # -- shared pieces -----------------------------------------------------
    def load_canonical(self, state: Mapping[str, Any]) -> None:
        self.trunk.load_canonical(state)

    def _active_z(self, batch) -> torch.Tensor:
        z = self.trunk.z_global(batch)
        index = torch.as_tensor(self.active_indices, dtype=torch.long, device=z.device)
        return z.index_select(1, index)

    def _context_z(self, batch) -> torch.Tensor:
        z = self.trunk.z_global(batch)
        index = torch.as_tensor(self.context_indices, dtype=torch.long, device=z.device)
        return z.index_select(1, index)

    def _global_terms(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        z_active = self._active_z(batch)
        u_terms = self.global_functions(z_active)
        return z_active, u_terms

    def module_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, child in self.named_children():
            out[name] = int(sum(p.numel() for p in child.parameters()))
        out["readout_total"] = int(
            sum(p.numel() for k, p in self.named_parameters() if not k.startswith(ARM_TRUNK_PREFIX))
        )
        out["trunk_total"] = int(
            sum(p.numel() for k, p in self.named_parameters() if k.startswith(ARM_TRUNK_PREFIX))
        )
        out["total"] = int(sum(p.numel() for p in self.parameters()))
        return out


class ArmB(_AdditiveBase):
    """Local additive base + one single-variable function per active global column."""

    def __init__(self, spec, norm, *, hidden: int, active_indices: Sequence[int], cfg=None) -> None:
        super().__init__(
            spec, norm, arm="B", hidden=hidden, active_indices=active_indices, context_indices=(), cfg=cfg
        )

    def parts(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        h = self.trunk(batch)
        hidden = F.silu(F.linear(h, self.w_hidden, self.b_hidden))
        base_root = F.linear(hidden, self.w_base, self.b_local).view(-1)
        sum_base = _sum_per_graph(base_root, atom_batch, n_graphs)
        z_active, u_terms = self._global_terms(batch)
        u_sum = u_terms.sum(dim=1)
        pred = self.b_global.expand(n_graphs) + sum_base + u_sum
        return {
            "pred_std": pred,
            "h": h,
            "hidden": hidden,
            "base_root": base_root,
            "sum_base": sum_base,
            "z_active": z_active,
            "u_terms": u_terms,
            "u_sum": u_sum,
            "intercept": self.b_global.expand(n_graphs),
            "term_names": ["sum_base", "u_sum", "b_global"],
        }

    def forward(self, batch, *, return_aux: bool = False):
        parts = self.parts(batch)
        return (parts["pred_std"], parts) if return_aux else parts["pred_std"]


class ArmC(_AdditiveBase):
    """B plus two explicit ``local structure x global context`` interactions."""

    def __init__(
        self,
        spec,
        norm,
        *,
        hidden: int,
        active_indices: Sequence[int],
        context_indices: Sequence[int],
        cfg=None,
    ) -> None:
        super().__init__(
            spec,
            norm,
            arm="C",
            hidden=hidden,
            active_indices=active_indices,
            context_indices=context_indices,
            cfg=cfg,
        )
        self.context_functions = SingleVariableFunctions(
            len(C.CONTEXT_FIELDS), generator=_stream(C.INIT_STREAM_CONTEXT)
        )

    def parts(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        h = self.trunk(batch)
        hidden = F.silu(F.linear(h, self.w_hidden, self.b_hidden))
        base_root = F.linear(hidden, self.w_base, self.b_local).view(-1)
        context_root = F.linear(hidden, self.w_context, None)               # (n_atoms, K)
        sum_base = _sum_per_graph(base_root, atom_batch, n_graphs)
        context_sums = torch.stack(
            [_sum_per_graph(context_root[:, k], atom_batch, n_graphs) for k in range(context_root.shape[1])],
            dim=1,
        )                                                                   # (n_graphs, K)
        z_active, u_terms = self._global_terms(batch)
        u_sum = u_terms.sum(dim=1)
        z_context = self._context_z(batch)
        a_values = self.context_functions(z_context)                        # (n_graphs, K)
        interaction = a_values * context_sums                               # (n_graphs, K)
        pred = self.b_global.expand(n_graphs) + sum_base + u_sum + interaction.sum(dim=1)
        return {
            "pred_std": pred,
            "h": h,
            "hidden": hidden,
            "base_root": base_root,
            "sum_base": sum_base,
            "context_root": context_root,
            "context_sums": context_sums,
            "a_values": a_values,
            "interaction": interaction,
            "interaction_root": context_root * a_values.index_select(0, atom_batch),
            "z_active": z_active,
            "u_terms": u_terms,
            "u_sum": u_sum,
            "z_context": z_context,
            "intercept": self.b_global.expand(n_graphs),
            "term_names": ["sum_base", "u_sum", "interaction_size", "interaction_ring", "b_global"],
        }

    def forward(self, batch, *, return_aux: bool = False):
        parts = self.parts(batch)
        return (parts["pred_std"], parts) if return_aux else parts["pred_std"]


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


@dataclass
class ArmSpec:
    arm: str
    hidden: int | None
    n_active: int
    readout_params_nominal: int
    budget: dict[str, Any]


def arm_specs(bundle: D.Bundle) -> dict[str, ArmSpec]:
    n_active = int(bundle.active_global["n_active"])
    out = {}
    out["A"] = ArmSpec(
        arm="A",
        hidden=None,
        n_active=n_active,
        readout_params_nominal=readout_params("A", 0, n_active),
        budget={
            "arm": "A",
            "formula": "f(64->64->1) + rho(171->64->1), unchanged from C1",
            "hidden": None,
            "nominal_readout_params": readout_params("A", 0, n_active),
            "n_active_global": n_active,
        },
    )
    for arm in ("B", "C"):
        chosen = choose_hidden(arm, n_active)
        out[arm] = ArmSpec(
            arm=arm,
            hidden=int(chosen["hidden"]),
            n_active=n_active,
            readout_params_nominal=int(chosen["nominal_readout_params"]),
            budget=chosen,
        )
    return out


def build_arm(
    arm: str,
    spec,
    norm: Mapping[str, Any],
    canonical_state: Mapping[str, Any],
    *,
    n_active: int,
    hidden: int | None,
    active_indices: Sequence[int],
    context_indices: Sequence[int],
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Construct one arm and load the *same* canonical trunk into it."""
    if arm == "A":
        model = ArmA(spec, norm)
    elif arm == "B":
        model = ArmB(spec, norm, hidden=int(hidden), active_indices=active_indices)
    elif arm == "C":
        model = ArmC(
            spec,
            norm,
            hidden=int(hidden),
            active_indices=active_indices,
            context_indices=context_indices,
        )
    else:
        raise KeyError(arm)
    model.load_canonical(canonical_state)
    return model.to(device)


def parameter_report(model: nn.Module) -> dict[str, Any]:
    named = list(model.named_parameters())
    trunk = sum(p.numel() for k, p in named if k.startswith(ARM_TRUNK_PREFIX))
    readout = sum(p.numel() for k, p in named if not k.startswith(ARM_TRUNK_PREFIX))
    breakdown: dict[str, int] = {}
    for key, value in named:
        top = key.split(".")[0]
        breakdown[top] = breakdown.get(top, 0) + int(value.numel())
    unused = [k for k, p in named if not p.requires_grad]
    return {
        "arm": getattr(model, "name", "?"),
        "total": int(sum(p.numel() for _, p in named)),
        "trunk": int(trunk),
        "readout": int(readout),
        "breakdown": breakdown,
        "non_trainable_parameters": unused,
        "n_parameter_tensors": len(named),
        "readout_parameter_names": sorted(k for k, _ in named if not k.startswith(ARM_TRUNK_PREFIX)),
    }


__all__ = [
    "WIDTH",
    "DROPOUT",
    "READOUT_BUDGET",
    "EXPECTED_COUNTS",
    "SingleVariableFunctions",
    "LocalTrunk",
    "ArmA",
    "ArmB",
    "ArmC",
    "ArmSpec",
    "readout_params",
    "choose_hidden",
    "canonical_c1_state",
    "trunk_keys",
    "state_hashes",
    "arm_specs",
    "build_arm",
    "parameter_report",
]
