"""DBT-PA: frozen local SSL stack + post-encoder rank-8 adapters.

The frozen part is exactly the P3 atom/bond/junction encoder stack
(``P3Encoders``) from the DBT round, initialised from the seed-0 train-only
masked-SSL checkpoint.  Its *final* outputs ``a0``, ``b0``, ``w0`` get three
independent residual adapters::

    a = a0 + U_a SiLU(D_a a0)
    b = b0 + U_b SiLU(D_b b0)
    w = w0 + U_w SiLU(D_w w0)

with ``D: 96 -> 8`` normal-initialised and ``U: 8 -> 96`` zero-initialised.
The frozen stack internally still uses unadapted intermediates; nothing is
fed back into the bond/junction encoders.  The adapted outputs go to the DBT
token builder and reader, whose token set, relations, four global-attention
layers and reader are unchanged.

Construction order matters: all original DBT modules are created first (so
their seed-0 random state is the same as the original DBT), then the adapters
are appended.  The original DBT modules are never loaded from the supervised
``best.pt``; only the local stack comes from the SSL checkpoint.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import dbt_data as BD  # noqa: E402
import dbt_models as BM  # noqa: E402

ADAPTER_RANK = 8


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Stable hash of a state dict (names + dtype + bytes)."""
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def encoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        for key, value in getattr(model.encoders, name).state_dict().items():
            state[f"{name}.{key}"] = value
    return state


def encoder_sha256(model: nn.Module) -> str:
    return state_dict_sha256(encoder_state(model))


class PostEncoderAdapter(nn.Module):
    """Residual ``x + U SiLU(D x)`` with zero-initialised ``U``."""

    def __init__(self, dim: int = BD.ATOM_OUT, rank: int = ADAPTER_RANK):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.down = nn.Linear(self.dim, self.rank)
        self.up = nn.Linear(self.rank, self.dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.silu(self.down(x)))


class DirectedBondTransformerPA(BM.DirectedBondTransformer):
    """DBT with a frozen local stack and three post-encoder adapters."""

    def __init__(self, norm: Mapping[str, Any], *, adapter_rank: int = ADAPTER_RANK,
                 **kwargs: Any):
        # All original DBT modules are created in the original order first.
        super().__init__(norm, **kwargs)
        self.adapter_rank = int(adapter_rank)
        # The adapters are appended after the original modules, so the seed-0
        # random state of every original DBT module is unchanged.
        self.adapter_a = PostEncoderAdapter(BD.ATOM_OUT, self.adapter_rank)
        self.adapter_b = PostEncoderAdapter(BD.BOND_OUT, self.adapter_rank)
        self.adapter_w = PostEncoderAdapter(BD.ATOM_OUT, self.adapter_rank)
        self.freeze_local_stack()

    # -- freezing ------------------------------------------------------------
    def freeze_local_stack(self) -> None:
        for p in self.encoders.parameters():
            p.requires_grad_(False)
        # The encoder stack must stay in eval mode even when the surrounding
        # model is switched to train(); its dropout is already 0, this makes
        # the invariant explicit.
        self.encoders.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoders.eval()
        for p in self.encoders.parameters():
            p.requires_grad_(False)
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # -- adapter application -------------------------------------------------
    def adapt_local(self, a0: torch.Tensor, b0: torch.Tensor,
                    w0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.adapter_a(a0), self.adapter_b(b0), self.adapter_w(w0)

    def forward_cached(self, batch: Mapping[str, torch.Tensor],
                       a0: torch.Tensor, b0: torch.Tensor, w0: torch.Tensor
                       ) -> torch.Tensor:
        """Forward from cached *base* outputs (adapters + DBT downstream)."""
        a, b, w = self.adapt_local(a0, b0, w0)
        return self.forward_from_local(batch, a, b, w)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Direct inference: run the frozen base live, then the adapters."""
        with torch.no_grad():
            a0, b0, w0 = self.encoders(batch)
        return self.forward_cached(batch, a0, b0, w0)


def build_model_pa(norm: Mapping[str, Any], seed: int, device, *,
                   graph_scales: Mapping[str, Any] | None = None,
                   dropout: float = 0.05,
                   adapter_rank: int = ADAPTER_RANK,
                   ssl_path: Path | str | None = None
                   ) -> tuple[DirectedBondTransformerPA, dict[str, Any]]:
    """Deterministic seed-0 construction + explicit SSL initialisation.

    There is deliberately no random-initialisation fallback: a missing SSL
    checkpoint is a hard error.
    """
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    model = DirectedBondTransformerPA(
        norm, dropout=dropout, graph_scales=graph_scales, adapter_rank=adapter_rank
    )
    model = model.to(device)
    path = Path(ssl_path) if ssl_path is not None else (
        C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
    )
    if not path.exists():
        raise FileNotFoundError(f"required SSL checkpoint is missing: {path}")
    loaded = model.load_ssl(path, device)
    ssl_state = torch.load(path, map_location="cpu", weights_only=False)
    matches = {}
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        have = getattr(model.encoders, name).state_dict()
        want = ssl_state[name]
        if set(have) != set(want):
            raise RuntimeError(f"SSL key mismatch in {name}: {sorted(have)} vs {sorted(want)}")
        matches[name] = all(torch.equal(have[k].cpu(), want[k].cpu()) for k in want)
    if not all(matches.values()):
        raise RuntimeError(f"SSL tensors were not loaded exactly: {matches}")

    total = C.parameter_count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    info: dict[str, Any] = {
        "seed": int(seed),
        "architecture": "directed_bond_transformer_post_adapters_v1",
        "adapter_rank": int(adapter_rank),
        "ssl": loaded,
        "ssl_tensor_match": matches,
        "encoder_sha256": encoder_sha256(model),
        "parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
        "module_breakdown": {
            name: int(sum(p.numel() for p in mod.parameters()))
            for name, mod in model.named_children()
        },
    }
    return model, info
