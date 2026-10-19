"""SSRC Phase 2a — atom-level local computation inserted after the BJOC AtomEncoder.

Protocol: ``notes/phase2a_protocol.md`` (``ssrc-phase2a-zinc-v1``).
Track: ``tracks/structural-sufficiency``.

The insertion point is the only architectural change relative to the frozen P3E
model (``structural_relation_composer_v1.BJOC``, ``organization_mode="null"``)::

    atom_feat -> AtomEncoder -> [ ARM F / ARM M ] -> a'
    a' -> BondEncoder -> JunctionEncoder -> NullComposer -> Reader      (unchanged)

Both arms share one block shape, so the only difference is what enters the
pre-norm LayerNorm:

* ``node_ffn`` — ``z = LayerNorm(h)``; per-atom, never touches ``batch``.
* ``local_mp``  — ``z = LayerNorm(h + msg)`` with
  ``msg_v = sum_{u->v in physical bonds} SiLU(h_u + Wb @ bond_feat_uv)``,
  two rounds, one shared bond projection, both edge directions.

There is no gate, no zero-initialised residual, no bond-state update, no global
attention, and no new raw descriptor.  Parameters are matched to +1.54 %.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
_REPO = _CODE_DIR.parents[3]
_STRUCT_CODE = _REPO / "tracks/structure-contribution/code"
for _p in (_CODE_DIR, _STRUCT_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

KIND_NONE = "none"
KIND_NODE_FFN = "node_ffn"
KIND_LOCAL_MP = "local_mp"
LOCAL_KINDS = (KIND_NONE, KIND_NODE_FFN, KIND_LOCAL_MP)

#: frozen design constants (protocol §2)
LOCAL_HIDDEN = 128
LOCAL_ROUNDS = 2
LAYER_INIT_OFFSET = 1013
BONDPROJ_INIT_OFFSET = 2027

#: block-internal parameter shape
DIM = 96  # == src.BJOCConfig().atom_out


def _linear_params(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim + out_dim


def block_parameters(dim: int = DIM, hidden: int = LOCAL_HIDDEN) -> int:
    """LayerNorm(dim) + Linear(dim,hidden) + Linear(hidden,dim)."""
    return 2 * dim + _linear_params(dim, hidden) + _linear_params(hidden, dim)


def local_parameters(kind: str, dim: int = DIM, hidden: int = LOCAL_HIDDEN, rounds: int = LOCAL_ROUNDS) -> int:
    if kind == KIND_NONE:
        return 0
    base = rounds * block_parameters(dim, hidden)
    if kind == KIND_LOCAL_MP:
        base += _linear_params(sro.BOND_FEAT_DIM, dim)
    return base


@dataclass(frozen=True)
class LocalConfig(src.BJOCConfig):
    """P3E widths (untouched) + the phase-2a local module."""

    kind: str = KIND_NODE_FFN
    local_hidden: int = LOCAL_HIDDEN
    local_rounds: int = LOCAL_ROUNDS

    def local_inventory(self) -> dict[str, int]:
        if self.kind == KIND_NONE:
            return {"per_block": 0, "blocks": 0, "bond_projection": 0, "total": 0}
        per_block = block_parameters(self.atom_out, self.local_hidden)
        out = {
            "per_block": per_block,
            "blocks": self.local_rounds * per_block,
            "bond_projection": (
                _linear_params(sro.BOND_FEAT_DIM, self.atom_out) if self.kind == KIND_LOCAL_MP else 0
            ),
        }
        out["total"] = out["blocks"] + out["bond_projection"]
        return out

    def inventory(self) -> dict[str, int]:
        base = super().inventory()
        local = self.local_inventory()["total"]
        return {**base, "local": local, "total": base["total"] + local}


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


class _LocalBlock(nn.Module):
    """``h + W2 SiLU(W1 LayerNorm(z))`` — pre-norm residual FFN."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.lin2(F.silu(self.lin1(self.norm(z))))


class NodeFFN2(nn.Module):
    """Two per-atom residual blocks.  Reads no adjacency, no edge attributes."""

    def __init__(self, dim: int, hidden: int, rounds: int = LOCAL_ROUNDS):
        super().__init__()
        self.blocks = nn.ModuleList([_LocalBlock(dim, hidden) for _ in range(rounds)])

    def forward(self, a: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        h = a
        for blk in self.blocks:
            h = h + blk(h)
        return h


class LocalMP2(nn.Module):
    """Two GINE-style rounds on the true physical bond list (both directions)."""

    def __init__(self, dim: int, hidden: int, bond_in: int, rounds: int = LOCAL_ROUNDS):
        super().__init__()
        self.bond_proj = nn.Linear(bond_in, dim)
        self.blocks = nn.ModuleList([_LocalBlock(dim, hidden) for _ in range(rounds)])

    def directed_edges(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (src, dst) for both directions of every undirected bond."""
        bi = batch["bond_index"]
        src = torch.cat([bi[:, 0], bi[:, 1]], dim=0)
        dst = torch.cat([bi[:, 1], bi[:, 0]], dim=0)
        return src, dst

    def messages(self, h: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        src, dst = self.directed_edges(batch)
        e = self.bond_proj(batch["bond_feat"].to(h.dtype))
        e2 = torch.cat([e, e], dim=0)
        msg = h.new_zeros((h.shape[0], h.shape[1]))
        msg.index_add_(0, dst, F.silu(h[src] + e2))
        return msg

    def forward(self, a: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        h = a
        for blk in self.blocks:
            h = h + blk(h + self.messages(h, batch))
        return h


def build_local(kind: str, dim: int = DIM, hidden: int = LOCAL_HIDDEN, rounds: int = LOCAL_ROUNDS) -> nn.Module | None:
    if kind == KIND_NONE:
        return None
    if kind == KIND_NODE_FFN:
        return NodeFFN2(dim, hidden, rounds)
    if kind == KIND_LOCAL_MP:
        return LocalMP2(dim, hidden, sro.BOND_FEAT_DIM, rounds)
    raise ValueError(f"unknown local kind {kind!r}")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class P3ELocal(src.BJOC):
    """Frozen BJOC-Null trunk + an atom-level local module after the AtomEncoder."""

    def __init__(
        self,
        config: LocalConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        organization_mode: str = src.NULL,
    ):
        super().__init__(config, global_mean, global_scale, organization_mode)
        self.local = build_local(config.kind, config.atom_out, config.local_hidden, config.local_rounds)

    @property
    def local_kind(self) -> str:
        return self.config.kind

    def encode_atoms(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a = self.atom_encoder(batch["atom_feat"])
        if self.local is None:
            return a
        return self.local(a, batch)


# ---------------------------------------------------------------------------
# deterministic initialisation
# ---------------------------------------------------------------------------


def _default_linear_init(lin: nn.Linear, gen: torch.Generator) -> None:
    """Exactly ``nn.Linear.reset_parameters`` but on a private generator."""
    nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5), generator=gen)
    bound = 1.0 / math.sqrt(lin.in_features)
    nn.init.uniform_(lin.bias, -bound, bound, generator=gen)


def init_local(module: nn.Module | None, seed: int) -> None:
    """Initialise the new module from dedicated generators (never the global RNG).

    Blocks use one generator (so Arm F and Arm M have identical block weights at
    init); the bond projection uses another.  LayerNorm stays at its default
    (weight=1, bias=0), which is deterministic.
    """
    if module is None:
        return
    g = torch.Generator().manual_seed(int(seed) + LAYER_INIT_OFFSET)
    for blk in module.blocks:
        _default_linear_init(blk.lin1, g)
        _default_linear_init(blk.lin2, g)
    if hasattr(module, "bond_proj"):
        g2 = torch.Generator().manual_seed(int(seed) + BONDPROJ_INIT_OFFSET)
        _default_linear_init(module.bond_proj, g2)


def _copy_trunk_(model: src.BJOC, base: src.BJOC) -> None:
    """Bit-identical copy of every published BJOC tensor (trunk only)."""
    base_sd = base.state_dict()
    own = model.state_dict()
    missing = [k for k in base_sd if k not in own]
    if missing:
        raise RuntimeError(f"trunk keys missing in extended model: {missing}")
    with torch.no_grad():
        for key, value in base_sd.items():
            if own[key].shape != value.shape:
                raise RuntimeError(f"shape mismatch for {key}: {tuple(own[key].shape)} vs {tuple(value.shape)}")
            own[key].copy_(value)


def build_local_model(
    kind: str,
    *,
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str = "cpu",
    hidden: int = LOCAL_HIDDEN,
    rounds: int = LOCAL_ROUNDS,
) -> tuple[src.BJOC, P3ELocal, dict[str, Any]]:
    """P3E scratch trunk (exactly ``bp.build_bjoc_null``) + deterministic local module."""
    import bjoc_perfmech_long_v1 as bp

    if kind not in LOCAL_KINDS:
        raise SystemExit(f"unknown kind {kind!r}")
    base = bp.build_bjoc_null(seed, norm, "cpu")
    cfg = LocalConfig(kind=kind, local_hidden=hidden, local_rounds=rounds)
    torch.manual_seed(int(seed))
    model = P3ELocal(
        cfg,
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        src.NULL,
    )
    _copy_trunk_(model, base)
    init_local(model.local, seed)
    model = model.to(device)
    info = {
        "kind": kind,
        "seed": int(seed),
        "local_hidden": hidden,
        "local_rounds": rounds,
        "base_scratch_init_full_sha256": src.state_dict_hash(base),
        "base_scratch_init_encoder_sha256": bp.encoder_group_hash(base),
        "base_scratch_init_group_state_sha256": bp.state_group_sha256(
            {**{f"group_composer:{k}": v for k, v in base.group_composer.state_dict().items()},
             **{f"reader:{k}": v for k, v in base.reader.state_dict().items()}}
        ),
        "extended_trunk_state_sha256": trunk_state_sha256(model),
        "extended_trunk_param_sha256": trunk_param_sha256(model),
        "trunk_matches_base_init": trunk_param_sha256(model) == src.state_dict_hash(base),
        "local_init_sha256": local_state_sha256(model.local),
        "local_block_init_sha256": local_block_state_sha256(model.local),
    }
    return base, model, info


def trunk_state_sha256(model: src.BJOC) -> str:
    """Hash of every non-``local`` tensor (buffers included)."""
    import hashlib

    h = hashlib.sha256()
    sd = model.state_dict()
    for key in sorted(sd):
        if key.startswith("local."):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(sd[key].detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def trunk_param_sha256(model: src.BJOC) -> str:
    """Same hashing rule as ``structural_relation_composer_v1.state_dict_hash``."""
    import hashlib

    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith("local."):
            continue
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def local_state_sha256(module: nn.Module | None) -> str:
    import hashlib

    if module is None:
        return "none"
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def local_block_state_sha256(module: nn.Module | None) -> str:
    """Hash of the block weights only (excludes the bond projection)."""
    import hashlib

    if module is None:
        return "none"
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        if key.startswith("bond_proj."):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    cfg = model.config
    expected = cfg.inventory()
    local = getattr(model, "local", None)
    actual = int(sum(p.numel() for p in model.parameters()))
    groups = {
        "atom_encoder": sum(p.numel() for p in model.atom_encoder.parameters()),
        "bond_encoder": sum(p.numel() for p in model.bond_encoder.parameters()),
        "junction_encoder": sum(p.numel() for p in model.junction_encoder.parameters()),
        "group_composer": sum(p.numel() for p in model.group_composer.parameters()),
        "reader": sum(p.numel() for p in model.reader.parameters()),
        "local": 0 if local is None else sum(p.numel() for p in local.parameters()),
    }
    expected_local = cfg.local_inventory() if hasattr(cfg, "local_inventory") else {"total": 0}
    return {
        "expected": expected,
        "expected_local": expected_local,
        "actual_parameters": actual,
        "actual_groups": groups,
        "parameter_exact": actual == expected["total"]
        and groups
        == (
            {
                k: expected[k]
                for k in ("atom_encoder", "bond_encoder", "junction_encoder", "group_composer", "reader")
            }
            | {"local": expected_local["total"]}
        ),
    }


def forward_gemm_flops(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Count matmul FLOPs of one forward pass (scatter-adds not counted)."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(batch)
    table = counter.get_flop_counts()
    prefix = type(model).__name__
    by_module: dict[str, int] = {}
    for name, _child in model.named_children():
        key = f"{prefix}.{name}"
        if key in table:
            by_module[name] = int(sum(int(v) for v in table[key].values()))
    total = int(sum(by_module.values()))
    mode_total = int(counter.get_total_flops())
    if total != mode_total:
        raise RuntimeError(f"per-module FLOPs sum {total} != counter total {mode_total}")
    return {"total": total, "by_module": by_module}


def local_branch_shift(a: torch.Tensor, a_local: torch.Tensor) -> dict[str, float]:
    """How far the local branch moves the atom state (relative L2 and cosine)."""
    num = float(torch.linalg.vector_norm(a_local - a))
    den = float(torch.linalg.vector_norm(a)) + 1e-12
    cos = float(
        torch.nn.functional.cosine_similarity(a_local.reshape(1, -1), a.reshape(1, -1), dim=1).item()
    )
    return {"relative_shift": num / den, "cosine": cos, "abs_shift": num}
