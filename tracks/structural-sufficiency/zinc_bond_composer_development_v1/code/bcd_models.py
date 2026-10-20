"""BCD models: original P3 (B0) and two controlled per-bond branches.

The only architectural intervention is after the shared ``group_composer``:

    c_e = P3_composer(b_e, W_e, V_e, n_e, raw_bond_e)          (96)
    h0_e = Linear(96 -> 48)(c_e)
    B1: MIX(x) = gamma * FFN_extra(x)          (48 -> 96 -> 48)
    B2: MIX(x) = 4-head Q/K/V/output MHA       (head dim 12, all bonds of
                                                the same molecule visible)
    U  = H + MIX(LN1(H))
    H1 = U + FFN(LN2(U))                       (48 -> 96 -> 48)
    c'_e = c_e + Linear_out(48 -> 96)(h1_e)
    z    = sum_e c'_e

Everything else in P3 is retained: atom/bond/junction encoders, Null grouping,
composer, the 321 -> 256 -> 256 -> 1 reader, and every original parameter.
``Linear_out`` is zero-initialised, so B1/B2 are exact P3 functions at step 0.

``c_e`` contains the graph-level Null context by construction.  It is *not* a
strictly local or causal per-bond contribution.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STRUCT_CODE = REPO / "tracks/structure-contribution/code"
for _p in (HERE, STRUCT_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bcd_common as C  # noqa: E402
import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402

DIM = 96
HIDDEN = 48
FFN_HIDDEN = 96
HEADS = 4
HEAD_DIM = 12
LN_EPS = 1e-5
KINDS = ("B0", "B1", "B2")
BRANCH_KINDS = ("B1", "B2")

#: Dedicated, kind-independent initialisation streams.  B1 and B2 therefore
#: receive the same input projection, LayerNorms, second FFN and zero output
#: projection; only MIX differs.
INIT_INPUT = 5101
INIT_BLOCK = 5102
INIT_MIX = 5103


def _linear_params(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim + out_dim


def branch_parameter_inventory() -> dict[str, int]:
    input_proj = _linear_params(DIM, HIDDEN)
    ln1 = 2 * HIDDEN
    ln2 = 2 * HIDDEN
    ffn = _linear_params(HIDDEN, FFN_HIDDEN) + _linear_params(FFN_HIDDEN, HIDDEN)
    mix_ffn = _linear_params(HIDDEN, FFN_HIDDEN) + _linear_params(FFN_HIDDEN, HIDDEN) + HIDDEN
    mix_mha = _linear_params(HIDDEN, 3 * HIDDEN) + _linear_params(HIDDEN, HIDDEN)
    output_proj = _linear_params(HIDDEN, DIM)
    block = ln1 + mix_ffn + ln2 + ffn
    return {
        "input_projection": input_proj,
        "ln1": ln1,
        "mix_b1": mix_ffn,
        "mix_b2": mix_mha,
        "ln2": ln2,
        "ffn": ffn,
        "block": block,
        "output_projection": output_proj,
        "total": input_proj + block + output_proj,
    }


def _default_linear_init(lin: nn.Linear, gen: torch.Generator) -> None:
    nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5), generator=gen)
    bound = 1.0 / math.sqrt(lin.in_features)
    nn.init.uniform_(lin.bias, -bound, bound, generator=gen)


class _FFNExtra(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, FFN_HIDDEN, bias=True)
        self.fc2 = nn.Linear(FFN_HIDDEN, HIDDEN, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class _MHA(nn.Module):
    """Standard 4-head scaled dot-product self-attention, head dim 12."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=True)
        self.out = nn.Linear(HIDDEN, HIDDEN, bias=True)

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 48]; key_padding: [B, N] bool, True = valid key
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, HEADS, HEAD_DIM)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, H, N, D]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        # SDPA bool mask: True means the key is allowed.
        mask = key_padding[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(b, n, HEADS * HEAD_DIM)
        return self.out(y)


class BondBranch(nn.Module):
    """One independent pre-LN block on the per-bond composer state."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in BRANCH_KINDS:
            raise ValueError(f"unknown branch kind {kind!r}")
        self.kind = kind
        self.input_proj = nn.Linear(DIM, HIDDEN, bias=True)
        self.ln1 = nn.LayerNorm(HIDDEN, eps=LN_EPS, elementwise_affine=True)
        if kind == "B1":
            self.mix = _FFNExtra()
            self.gamma = nn.Parameter(torch.ones(HIDDEN))
        else:
            self.mix = _MHA()
        self.ln2 = nn.LayerNorm(HIDDEN, eps=LN_EPS, elementwise_affine=True)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN, FFN_HIDDEN, bias=True),
            nn.SiLU(),
            nn.Linear(FFN_HIDDEN, HIDDEN, bias=True),
        )
        self.output_proj = nn.Linear(HIDDEN, DIM, bias=True)

    def mix_forward(self, x: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        if self.kind == "B1":
            return self.gamma * self.mix(x)
        return self.mix(x, key_padding)

    def forward_packed(self, h: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        u = h + self.mix_forward(self.ln1(h), key_padding)
        h1 = u + self.ffn(self.ln2(u))
        return self.output_proj(h1)


def _pack(h: torch.Tensor, bond_batch: torch.Tensor, n_graphs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack variable-length per-molecule bond rows into [B, Nmax, 48]."""
    device = h.device
    if h.shape[0] == 0:
        return h.new_zeros((n_graphs, 0, HIDDEN)), torch.zeros((n_graphs, 0), dtype=torch.bool, device=device)
    counts = torch.bincount(bond_batch, minlength=n_graphs)
    max_n = int(counts.max().item())
    offsets = torch.cumsum(counts, 0) - counts
    pos = torch.arange(h.shape[0], device=device) - offsets[bond_batch]
    packed = h.new_zeros((n_graphs, max_n, HIDDEN))
    valid = torch.zeros((n_graphs, max_n), dtype=torch.bool, device=device)
    packed[bond_batch, pos] = h
    valid[bond_batch, pos] = True
    return packed, valid


def _unpack(packed: torch.Tensor, valid: torch.Tensor, bond_batch: torch.Tensor) -> torch.Tensor:
    counts = valid.sum(1)
    offsets = torch.cumsum(counts, 0) - counts
    pos = torch.arange(bond_batch.shape[0], device=packed.device) - offsets[bond_batch]
    return packed[bond_batch, pos]


class DevBJOC(src.BJOC):
    """P3 with an optional per-bond branch on the composer output ``c``."""

    def __init__(
        self,
        config: src.BJOCConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        branch_kind: str = "B0",
    ):
        super().__init__(config, global_mean, global_scale, src.NULL)
        if branch_kind not in KINDS:
            raise ValueError(f"unknown branch kind {branch_kind!r}")
        self.branch_kind = branch_kind
        self.branch = BondBranch(branch_kind) if branch_kind in BRANCH_KINDS else None

    def branch_delta(self, c: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        assert self.branch is not None
        n_graphs = int(batch["n_graphs"])
        h0 = self.branch.input_proj(c)
        packed, valid = _pack(h0, batch["bond_batch"], n_graphs)
        if packed.shape[1] == 0:
            return c.new_zeros(c.shape)
        # For molecules with no bonds there is no real query.  Giving them one
        # dummy valid zero key avoids a fully masked SDPA row; the resulting
        # padded outputs are discarded below and never enter the molecule sum.
        key_valid = valid.clone()
        empty = ~valid.any(1)
        if bool(empty.any()):
            key_valid[empty, 0] = True
        delta_packed = self.branch.forward_packed(packed, key_valid)
        delta = _unpack(delta_packed, valid, batch["bond_batch"])
        return torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.branch is None:
            return super().forward_features(batch, organization_mode)
        feats = super().forward_features(batch, organization_mode)
        c = feats["c"]
        delta = self.branch_delta(c, batch)
        c_prime = c + delta
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c_prime)
        a = feats["a"]
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        return {
            **feats,
            "c_base": c,
            "delta": delta,
            "c_prime": c_prime,
            "z_base": feats["z"],
            "z": z,
            "reader_in": reader_in,
        }


# ---------------------------------------------------------------------------
# construction / hashing
# ---------------------------------------------------------------------------


def _copy_trunk_(model: src.BJOC, base: src.BJOC) -> None:
    base_sd = base.state_dict()
    own = model.state_dict()
    missing = [k for k in base_sd if k not in own and not k.startswith("branch.")]
    if missing:
        raise RuntimeError(f"trunk keys missing in extended model: {missing}")
    with torch.no_grad():
        for key, value in base_sd.items():
            if key.startswith("branch."):
                continue
            if own[key].shape != value.shape:
                raise RuntimeError(f"shape mismatch for {key}")
            own[key].copy_(value)


def init_branch(branch: BondBranch, seed: int) -> None:
    g_in = torch.Generator().manual_seed(int(seed) + INIT_INPUT)
    _default_linear_init(branch.input_proj, g_in)
    # LayerNorm defaults are deterministic (weight=1, bias=0).
    g_block = torch.Generator().manual_seed(int(seed) + INIT_BLOCK)
    _default_linear_init(branch.ffn[0], g_block)
    _default_linear_init(branch.ffn[2], g_block)
    g_mix = torch.Generator().manual_seed(int(seed) + INIT_MIX)
    if branch.kind == "B1":
        _default_linear_init(branch.mix.fc1, g_mix)
        _default_linear_init(branch.mix.fc2, g_mix)
        nn.init.ones_(branch.gamma)
    else:
        _default_linear_init(branch.mix.qkv, g_mix)
        _default_linear_init(branch.mix.out, g_mix)
    nn.init.zeros_(branch.output_proj.weight)
    nn.init.zeros_(branch.output_proj.bias)


def build_model(
    kind: str,
    *,
    seed: int,
    norm: Mapping[str, Any],
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> tuple[src.BJOC, DevBJOC, dict[str, Any]]:
    """Canonical P3 post-SSL trunk plus an optional, independently initialised branch."""
    if kind not in KINDS:
        raise SystemExit(f"unknown kind {kind!r}")
    base = bp.build_bjoc_null(seed, norm, "cpu")
    mp.load_encoder_state(base, ssl_state, "cpu")
    canonical_param = src.state_dict_hash(base)
    canonical_encoder = bp.encoder_group_hash(base)
    canonical_group = mp.group_state_hash(base)
    if canonical_encoder != C.CANONICAL_ENCODER_SHA:
        raise SystemExit(f"canonical encoder hash mismatch: {canonical_encoder}")
    if canonical_group != C.CANONICAL_GROUP_SHA:
        raise SystemExit(f"canonical composer/reader hash mismatch: {canonical_group}")

    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        model = DevBJOC(
            src.BJOCConfig(),
            np.asarray(norm["global_mean"], dtype=np.float32),
            np.asarray(norm["global_scale"], dtype=np.float32),
            kind,
        )
    finally:
        torch.set_rng_state(rng_state)
    _copy_trunk_(model, base)
    if model.branch is not None:
        init_branch(model.branch, seed)
    model = model.to(device)

    info = {
        "kind": kind,
        "seed": int(seed),
        "canonical_param_sha256": canonical_param,
        "canonical_encoder_sha256": canonical_encoder,
        "canonical_group_sha256": canonical_group,
        "trunk_param_sha256": trunk_param_sha256(model),
        "trunk_state_sha256": trunk_state_sha256(model),
        "branch_param_sha256": branch_param_sha256(model.branch),
        "branch_common_param_sha256": branch_common_param_sha256(model.branch),
        "params": parameter_count(model),
    }
    info["trunk_matches_canonical"] = info["trunk_param_sha256"] == canonical_param
    if not info["trunk_matches_canonical"]:
        raise SystemExit("extended model trunk does not match canonical P3")
    return base, model, info


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def parameter_audit(model: DevBJOC) -> dict[str, Any]:
    groups: dict[str, int] = {}
    branch: dict[str, int] = {}
    for name, p in model.named_parameters():
        root = name.split(".")[0]
        groups[root] = groups.get(root, 0) + int(p.numel())
        if root == "branch":
            parts = name.split(".")
            key = parts[1]
            if key == "mix":
                key = "mix_b1" if model.branch_kind == "B1" else "mix_b2"
            branch[key] = branch.get(key, 0) + int(p.numel())
    total = parameter_count(model)
    return {
        "kind": model.branch_kind,
        "p3_parameters": total if model.branch is None else total - sum(branch.values()),
        "branch_parameters": sum(branch.values()),
        "expected_p3": bp.EXPECTED_PARAMETERS,
        "expected_total": C.EXPECTED_PARAMS[model.branch_kind],
        "expected_branch": 0 if model.branch_kind == "B0" else C.BRANCH_PARAMS,
        "total": total,
        "groups": groups,
        "branch_groups": branch,
        "parameter_exact": total == C.EXPECTED_PARAMS[model.branch_kind],
        "reader_shape": [list(x.shape) for x in model.reader.lins[0].parameters()],
        "reader_first_weight": list(model.reader.lins[0].weight.shape),
    }


def truncate_hash(h: str) -> str:
    return h[:16]


def trunk_param_sha256(model: nn.Module) -> str:
    """Hash every non-branch parameter, matching ``src.state_dict_hash``."""
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith("branch."):
            continue
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def trunk_state_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith("branch."):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def branch_param_sha256(branch: BondBranch | None) -> str:
    if branch is None:
        return "none"
    h = hashlib.sha256()
    for key, value in sorted(branch.state_dict().items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def branch_common_param_sha256(branch: BondBranch | None) -> str:
    """Hash the branch modules that B1 and B2 are required to share bitwise."""
    if branch is None:
        return "none"
    h = hashlib.sha256()
    for key, value in sorted(branch.state_dict().items()):
        if key.startswith("mix.") or key == "gamma":
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def zero_output_projection(model: DevBJOC) -> None:
    if model.branch is not None:
        with torch.no_grad():
            model.branch.output_proj.weight.zero_()
            model.branch.output_proj.bias.zero_()


def forward_gemm_flops(model: DevBJOC, batch: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Analytic forward matmul FLOPs for one batch (scatter/softmax excluded)."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(batch)
        total = int(counter.get_total_flops())
    finally:
        model.train(was_training)
    return {"total": total}
