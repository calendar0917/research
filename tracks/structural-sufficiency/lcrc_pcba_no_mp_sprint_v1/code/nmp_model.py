"""Arms S, A and G on top of the frozen LCRC-C1.

The frozen C1 class is subclassed, not copied: ``AtomEncoder``, ``BondEncoder``,
S1/S2 ``LocalContent``, dropout, ``f``, ``rho``, the global descriptor and the
Train-only continuous normalisation are inherited unchanged.  The new modules
are appended as extra submodules and read only ``h``, ``atom_batch`` and the
molecule count; they never touch ``bond_index``, ``bond_feat``, ``nbr_*`` or
``path_*``.

Each arm is zero-residual at initialisation: the last Linear of every residual
block (or the graph readout head for A) is zero, so step 0 reproduces C1.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import nmp_common as C
import pc_model as PC
import bp_model as BPM


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


class StaticFFNBlock(nn.Module):
    """Pre-norm residual FFN, 64 -> 160 -> 64, no inter-atom access."""

    def __init__(self, width: int = C.WIDTH, hidden: int = 160):
        super().__init__()
        self.norm = nn.LayerNorm(int(width))
        self.fc1 = nn.Linear(int(width), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        return self.fc2(F.silu(self.fc1(z)))


class GraphContextBlock(nn.Module):
    """Pre-norm residual block with a per-graph mean-context broadcast."""

    def __init__(self, width: int = C.WIDTH, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(int(width))
        self.ctx = nn.Linear(int(width), int(width))
        self.fc1 = nn.Linear(int(width), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(width))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        return self.fc2(F.silu(self.fc1(z + context)))


class AttentionScorer(nn.Module):
    """One permutation-invariant attention head: 64 -> 32 -> 1."""

    def __init__(self, width: int = C.WIDTH, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(int(width), int(hidden))
        self.fc2 = nn.Linear(int(hidden), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class AttentionPoolReadout(nn.Module):
    """Four-head graph-level pooling plus a graph-level residual head."""

    def __init__(self, width: int = C.WIDTH, hidden: int = 32, n_heads: int = 4,
                 pool_hidden: int = 88, n_out: int = C.N_TASKS):
        super().__init__()
        self.n_heads = int(n_heads)
        self.heads = nn.ModuleList([AttentionScorer(width, hidden) for _ in range(self.n_heads)])
        self.fc1 = nn.Linear(self.n_heads * int(width), int(pool_hidden))
        self.fc2 = nn.Linear(int(pool_hidden), int(n_out))

    def segment_softmax(self, scores: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int):
        """Numerically stable softmax over atoms of the same molecule, per head."""
        G = int(n_graphs)
        H = int(scores.shape[1])
        shift_batch = atom_batch.unsqueeze(1).expand(-1, H)
        maxv = scores.new_full((G, H), float("-inf"))
        maxv.scatter_reduce_(0, shift_batch, scores, reduce="amax", include_self=False)
        maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
        e = (scores - maxv[atom_batch]).exp()
        den = scores.new_zeros((G, H)).index_add_(0, atom_batch, e)
        alpha = e / den[atom_batch].clamp_min(1e-30)
        return alpha

    def forward(
        self,
        h: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
        *,
        return_attention: bool = False,
    ):
        scores = torch.cat([head(h) for head in self.heads], dim=1)  # (N, 4)
        alpha = self.segment_softmax(scores, atom_batch, int(n_graphs))
        pooled = []
        for k in range(self.n_heads):
            pooled.append(
                h.new_zeros((int(n_graphs), h.shape[1])).index_add_(
                    0, atom_batch, alpha[:, k : k + 1] * h
                )
            )
        z = torch.cat(pooled, dim=1)  # (G, 256)
        residual = self.fc2(F.silu(self.fc1(z)))
        if return_attention:
            return residual, alpha, z
        return residual, None, None


# ---------------------------------------------------------------------------
# arm model
# ---------------------------------------------------------------------------


class NMPModel(BPM.C1Model):
    """Frozen LCRC-C1 with one of the three no-bond-MP additions."""

    def __init__(self, spec: Any, norm: Mapping[str, Any], n_out: int, arm: str):
        if arm not in C.ARMS:
            raise KeyError(arm)
        # Fork the global RNG around the *whole* construction: the base module
        # constructor draws random values that are immediately overwritten by
        # the frozen C1 state, and the new modules are set from a private
        # generator later.  Building an arm therefore never advances the public
        # dropout stream.
        with torch.random.fork_rng(devices=[]):
            super().__init__(spec, norm, int(n_out))
            self.arm = arm
            self.mode = C.ARMS[arm]["mode"]
            if self.mode == "static_ffn":
                self.blocks: nn.ModuleList | None = nn.ModuleList([StaticFFNBlock() for _ in range(2)])
                self.heads = None
                self.readout = None
            elif self.mode == "graph_context":
                self.blocks = nn.ModuleList([GraphContextBlock() for _ in range(2)])
                self.heads = None
                self.readout = None
            elif self.mode == "attention_pool":
                self.blocks = None
                self.readout = AttentionPoolReadout(n_out=int(n_out))
                self.heads = None  # heads live inside ``readout`` and are listed for clarity
            else:  # pragma: no cover - defensive
                raise KeyError(self.mode)

    # -- new-parameter helpers ------------------------------------------------

    def new_parameter_names(self) -> list[str]:
        return [
            n for n, _ in self.named_parameters()
            if n.startswith(("blocks.", "readout."))
        ]

    def base_parameter_names(self) -> list[str]:
        new = set(self.new_parameter_names())
        return [n for n, _ in self.named_parameters() if n not in new]

    def new_modules(self) -> Iterable[nn.Module]:
        mods: list[nn.Module] = []
        if self.blocks is not None:
            mods.extend(self.blocks)
        if self.readout is not None:
            mods.append(self.readout)
        return mods

    def new_parameters(self) -> list[nn.Parameter]:
        names = set(self.new_parameter_names())
        return [p for n, p in self.named_parameters() if n in names]

    def base_parameters(self) -> list[nn.Parameter]:
        names = set(self.new_parameter_names())
        return [p for n, p in self.named_parameters() if n not in names]

    # -- graph helpers --------------------------------------------------------

    @staticmethod
    def _atom_count(atom_batch: torch.Tensor, n_graphs: int, dtype: torch.dtype) -> torch.Tensor:
        count = atom_batch.new_zeros(int(n_graphs), dtype=dtype).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=dtype)
        )
        return count.clamp(min=1.0).unsqueeze(1)

    def _graph_mean(self, values: torch.Tensor, atom_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        denom = self._atom_count(atom_batch, int(n_graphs), values.dtype)
        return values.new_zeros((int(n_graphs), values.shape[1])).index_add_(
            0, atom_batch, values
        ) / denom

    # -- arm branches ---------------------------------------------------------

    def propagate(
        self, h: torch.Tensor, batch: Mapping[str, torch.Tensor], *, return_attention: bool = False
    ):
        """Apply the arm branch.  Returns ``(h, attention, pooled, residual)``."""
        atom_batch = batch["atom_batch"]
        n_graphs = int(batch["n_graphs"])
        attention = pooled = residual = None
        if self.mode == "static_ffn":
            for block in self.blocks:
                h = h + block(h)
        elif self.mode == "graph_context":
            for block in self.blocks:
                z = block.norm(h)
                g = self._graph_mean(z, atom_batch, n_graphs)
                context = block.ctx(g)[atom_batch]
                h = h + block.fc2(F.silu(block.fc1(z + context)))
        elif self.mode == "attention_pool":
            residual, attention, pooled = self.readout(
                h, atom_batch, n_graphs, return_attention=return_attention
            )
        else:  # pragma: no cover - defensive
            raise KeyError(self.mode)
        return h, attention, pooled, residual

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        a, b, scal = self._encode(batch)
        h = self.drop(self.local(batch, a, b, scal))
        want_attention = bool(return_aux and self.mode == "attention_pool")
        h, attention, pooled, residual = self.propagate(
            h, batch, return_attention=want_attention
        )
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib = h.new_zeros(n_graphs, self.n_classes).index_add_(0, atom_batch, self.f(h))
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in)
        logits = contrib + rho
        if residual is not None:
            logits = logits + residual
        if return_aux:
            return logits, {
                "h": h,
                "contrib": contrib,
                "rho": rho,
                "mean_h": mean_h,
                "mean_h2": mean_h2,
                "attention_weights": attention,
                "attention_pooled": pooled,
                "residual_logits": residual,
            }
        return logits


# ---------------------------------------------------------------------------
# construction / initialisation
# ---------------------------------------------------------------------------

BASE_STATE_TENSORS = 37
BASE_PARAMETERS = 31


def build_reference(
    spec: Any, norm: Mapping[str, Any], n_out: int, seed: int
) -> tuple[nn.Module, dict[str, Any], dict[str, torch.Tensor]]:
    """Build the frozen C1 identity and return its CPU state dict."""
    ref, info = PC.build_c1(spec, norm, int(n_out), int(seed), "cpu")
    if info.get("variant") != "C1" or int(info.get("parameters", -1)) != C.EXPECTED_PARAMETERS["C"]:
        raise RuntimeError(f"reference C1 is not the frozen model: {info.get('parameters')}")
    state = {k: v.detach().cpu().clone() for k, v in ref.state_dict().items()}
    if len(state) != BASE_STATE_TENSORS:
        raise RuntimeError(f"reference state has {len(state)} tensors, expected {BASE_STATE_TENSORS}")
    return ref, info, state


def _new_state_keys(model: NMPModel) -> set[str]:
    keys = set(model.state_dict())
    base = {k for k in keys if not k.startswith(("blocks.", "readout."))}
    return keys - base


def initialize_new_modules(model: NMPModel, *, seed: int, zero_residual: bool = True) -> dict[str, Any]:
    """Private-generator init; the global torch RNG is never touched."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + 17)
    before = torch.get_rng_state().clone()
    n_linear = 0
    n_norm = 0
    with torch.no_grad():
        for module in model.new_modules():
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    bound = 1.0 / math.sqrt(float(sub.in_features))
                    sub.weight.uniform_(-bound, bound, generator=gen)
                    if sub.bias is not None:
                        sub.bias.uniform_(-bound, bound, generator=gen)
                    n_linear += 1
                elif isinstance(sub, nn.LayerNorm):
                    nn.init.ones_(sub.weight)
                    nn.init.zeros_(sub.bias)
                    n_norm += 1
    if zero_residual:
        with torch.no_grad():
            if model.mode in ("static_ffn", "graph_context"):
                for block in model.blocks:
                    block.fc2.weight.zero_()
                    block.fc2.bias.zero_()
            elif model.mode == "attention_pool":
                model.readout.fc2.weight.zero_()
                model.readout.fc2.bias.zero_()
    after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError("private initialisation changed the global torch RNG state")
    return {
        "private_generator_seed": int(seed) * 1_000_003 + 17,
        "linear_layers_initialised": n_linear,
        "layernorm_layers_initialised": n_norm,
        "residual_output_zeroed": bool(zero_residual),
        "global_rng_unchanged": True,
    }


def build_arm(
    arm: str,
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    seed: int,
    device: torch.device | str,
    base_init_state: Mapping[str, torch.Tensor],
    zero_residual: bool = True,
) -> tuple[NMPModel, dict[str, Any]]:
    model = NMPModel(spec, norm, int(n_out), arm)
    missing, unexpected = model.load_state_dict(dict(base_init_state), strict=False)
    expected_missing = _new_state_keys(model)
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"frozen init load mismatch: missing={sorted(missing)} "
            f"expected_new={sorted(expected_missing)} unexpected={sorted(unexpected)}"
        )
    base_fp_init = C.state_fingerprint(
        {k: v for k, v in model.state_dict().items() if not k.startswith(("blocks.", "readout."))}
    )
    init_audit = initialize_new_modules(model, seed=int(seed), zero_residual=bool(zero_residual))
    base_fp_after = C.state_fingerprint(
        {k: v for k, v in model.state_dict().items() if not k.startswith(("blocks.", "readout."))}
    )
    if base_fp_after != base_fp_init:
        raise RuntimeError("new-module initialisation changed the frozen base state")
    model.to(torch.device(device))

    names = model.base_parameter_names()
    new_names = model.new_parameter_names()
    base_elements = int(sum(p.numel() for p in model.base_parameters()))
    new_elements = int(sum(p.numel() for p in model.new_parameters()))
    total = C.parameter_count(model)
    expected = C.EXPECTED_PARAMETERS[arm]
    if int(base_elements) != C.EXPECTED_PARAMETERS["C"] or len(names) != BASE_PARAMETERS:
        raise RuntimeError(f"base parameter count {len(names)} / {base_elements} != 31 / 245184")
    if total != expected or new_elements != C.ARMS[arm]["new_parameters"]:
        raise RuntimeError(
            f"arm {arm} parameters {total} (new {new_elements}) != expected {expected} "
            f"(new {C.ARMS[arm]['new_parameters']})"
        )
    audit = {
        "arm": arm,
        "arm_name": C.ARMS[arm]["name"],
        "mode": model.mode,
        "base_parameter_tensors": len(names),
        "base_parameter_elements": base_elements,
        "new_parameter_tensors": len(new_names),
        "new_parameter_count": new_elements,
        "new_parameter_names": new_names,
        "total_parameter_count": total,
        "base_init_fingerprint": base_fp_init,
        "base_state_fingerprint_after_load": base_fp_after,
        "new_module_init": init_audit,
        "zero_residual_at_init": bool(zero_residual),
        "variant": "C1 + " + model.mode,
    }
    return model, audit


# ---------------------------------------------------------------------------
# optimizer / state helpers
# ---------------------------------------------------------------------------


def make_optimizer_scratch(model: NMPModel) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.SCRATCH_RECIPE["warmup_start_lr"]),
        betas=tuple(C.SCRATCH_RECIPE["adam_betas"]),
        eps=float(C.SCRATCH_RECIPE["adam_eps"]),
        weight_decay=float(C.SCRATCH_RECIPE["weight_decay"]),
    )
    audit = {
        "optimizer": "AdamW",
        "mode": "from_scratch",
        "n_param_groups": len(optimizer.param_groups),
        "n_parameters": int(sum(p.numel() for p in params)),
        "learning_rate_target": float(C.SCRATCH_RECIPE["learning_rate"]),
        "warmup_start_lr": float(C.SCRATCH_RECIPE["warmup_start_lr"]),
        "moments_reinitialised": True,
        "betas": list(C.SCRATCH_RECIPE["adam_betas"]),
        "eps": float(C.SCRATCH_RECIPE["adam_eps"]),
        "weight_decay": float(C.SCRATCH_RECIPE["weight_decay"]),
    }
    return optimizer, audit


def set_lrs(optimizer: torch.optim.AdamW, *, step: int) -> float:
    lr = C.scheduled_lr(int(step))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    import copy

    state = optimizer.state_dict()
    out: dict[str, Any] = {"state": {}, "param_groups": copy.deepcopy(state["param_groups"])}
    for param_id, values in state["state"].items():
        out["state"][int(param_id)] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in values.items()
        }
    return out


def restore_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    import random

    import numpy as np

    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        states = [s.detach().cpu() for s in state["torch_cuda"]]
        if len(states) != torch.cuda.device_count():
            raise RuntimeError(
                f"RNG state has {len(states)} CUDA devices, host has {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(states)
    return C.rng_state_hashes(C.capture_rng_state())


__all__ = [
    "AttentionPoolReadout",
    "AttentionScorer",
    "BASE_PARAMETERS",
    "BASE_STATE_TENSORS",
    "GraphContextBlock",
    "NMPModel",
    "StaticFFNBlock",
    "build_arm",
    "build_reference",
    "initialize_new_modules",
    "make_optimizer_scratch",
    "optimizer_state_cpu",
    "restore_rng_state",
    "set_lrs",
    "state_cpu",
]
