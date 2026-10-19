"""ZINC structural-token attention v1 — model definitions.

Round: ``zinc-structural-token-attention-v1``
Protocol: ``results/zinc_structural_token_attention_v1/PROTOCOL.md``

Design (frozen before any GPU second of this round)
---------------------------------------------------
Keep the frozen P3 ``BJOC-Null`` atom / bond / junction encoders, the Null
grouping path and the group composer.  Remove the old 321->256->256->1 reader.
Build one token per *explicit object* (atom / physical bond / junction):

    H0 = concat_rows(a, b, w) + type_embedding[type]

with no atom index, no sequence position, no truncation.  Two independent
pre-LN residual blocks act only inside one molecule:

    U    = H + MIX(LN1(H))
    Hnext = U + FFN(LN2(U))

``MIX`` is the only arm-specific module:

* ``S``  gamma * FFN_extra(x)         (capacity / readout control)
* ``T``  standard 4-head Q/K/V/output self-attention
* ``R``  T + zero-initialised structural bias
         Btype[type_i, type_j, h] + Bdist[dist_bin, h] + Boverlap[overlap, h]

Finally

    Z = [ z_P3(321), sum_A H2(96), sum_B H2(96), sum_J H2(96),
          log1p(N_A), log1p(N_B), log1p(N_J) ]              (612)
    y_hat = reader(Z),  reader = 612 -> 96 -> 96 -> 1 (SiLU)

The P3 statistics bypass ``z_P3`` is computed by the *original* Null grouping
and composer code path, not by a frozen P3 prediction.  The old reader is
deleted, not retained.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO = _CODE_DIR.parents[3]
for _p in (_CODE_DIR, REPO / "tracks/structure-contribution/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import structural_relation_composer_v1 as src  # noqa: E402

DIM = 96
HEADS = 4
HEAD_DIM = DIM // HEADS
FFN_HIDDEN = 192
LN_EPS = 1e-5
TYPE_NAMES = ("A", "B", "J")
N_TYPES = 3
N_DIST_BINS = 6  # 0,1,2,3,>=4,disconnected
N_OVERLAP_BINS = 4  # 0,1,2,3
N_TYPE_PAIRS = 9
VARIANTS = ("S", "T", "R")
N_BLOCKS = 2

P3_TOTAL = 413505
P3_READER = 148481
P3_NONREADER = P3_TOTAL - P3_READER  # 265024

#: 2 * (4*96*96 + 4*96) = 2 * 37248
MIX_PARAMETERS = 37248
#: 96*192+192 + 192*96+96 = 37152
FFN_PARAMETERS = 96 * FFN_HIDDEN + FFN_HIDDEN + FFN_HIDDEN * DIM + DIM
LN_PARAMETERS = 2 * DIM
BLOCK_PARAMETERS = MIX_PARAMETERS + 2 * LN_PARAMETERS + FFN_PARAMETERS
TYPE_PARAMETERS = N_TYPES * DIM
READER_PARAMETERS = 612 * DIM + DIM + DIM * DIM + DIM + DIM * 1 + 1
RELATION_PARAMETERS = N_BLOCKS * HEADS * (N_TYPE_PAIRS + N_DIST_BINS + N_OVERLAP_BINS)
EXPECTED_PARAMETERS = {
    "S": P3_NONREADER + N_BLOCKS * BLOCK_PARAMETERS + TYPE_PARAMETERS + READER_PARAMETERS,
    "T": P3_NONREADER + N_BLOCKS * BLOCK_PARAMETERS + TYPE_PARAMETERS + READER_PARAMETERS,
    "R": P3_NONREADER
    + N_BLOCKS * BLOCK_PARAMETERS
    + TYPE_PARAMETERS
    + READER_PARAMETERS
    + RELATION_PARAMETERS,
}

INIT_OFFSET = 9119

FORBIDDEN_TOKENS = (
    "MessagePassing",
    "GINE",
    "GCN",
    "GAT",
    "SAGEConv",
    "MultiheadAttention",
    "Transformer",
    "GRU",
    "LSTM",
    "RNN",
    "BatchNorm",
    "Sequential_state",
)
ALLOWED_EXTRA_CLASSES = (
    "STAObjectModel",
    "TokenBlock",
    "AttnMix",
    "ExtraFFN",
    "ResidualFFN",
    "StructuralBias",
    "NewReader",
)


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------


class ResidualFFN(nn.Module):
    """96 -> 192 -> 96, SiLU, both layers biased, no dropout."""

    def __init__(self, dim: int = DIM, hidden: int = FFN_HIDDEN):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden, bias=True)
        self.lin2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(F.silu(self.lin1(x)))


class ExtraFFN(nn.Module):
    """S-arm MIX: gamma * FFN_extra(x), gamma initialised to 1."""

    def __init__(self, dim: int = DIM, hidden: int = FFN_HIDDEN):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden, bias=True)
        self.lin2 = nn.Linear(hidden, dim, bias=True)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, **_unused: Any) -> torch.Tensor:
        return self.gamma * self.lin2(F.silu(self.lin1(x)))


class StructuralBias(nn.Module):
    """Per-layer, per-head additive bias tables (zero initialised)."""

    def __init__(self, heads: int = HEADS):
        super().__init__()
        self.heads = int(heads)
        self.type_bias = nn.Parameter(torch.zeros(N_TYPES, N_TYPES, self.heads))
        self.dist_bias = nn.Parameter(torch.zeros(N_DIST_BINS, self.heads))
        self.overlap_bias = nn.Parameter(torch.zeros(N_OVERLAP_BINS, self.heads))

    def forward(
        self,
        type_ids: torch.Tensor,  # [G, T] long
        rel_dist: torch.Tensor,  # [G, T, T] long, 0..5
        rel_overlap: torch.Tensor,  # [G, T, T] long, 0..3
    ) -> torch.Tensor:
        g, t = type_ids.shape
        ti = type_ids[:, :, None].expand(g, t, t)
        tj = type_ids[:, None, :].expand(g, t, t)
        pair = ti * N_TYPES + tj
        b = F.embedding(pair, self.type_bias.reshape(N_TYPE_PAIRS, self.heads))
        b = b + F.embedding(rel_dist, self.dist_bias)
        b = b + F.embedding(rel_overlap, self.overlap_bias)
        return b.permute(0, 3, 1, 2)  # [G, H, T, T]


class AttnMix(nn.Module):
    """Standard 4-head Q/K/V/output self-attention, optional relation bias."""

    def __init__(self, dim: int = DIM, heads: int = HEADS, relation: bool = False):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.relation = bool(relation)
        self.struct_bias = StructuralBias(self.heads) if self.relation else None

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        type_ids: torch.Tensor,
        rel_dist: torch.Tensor | None = None,
        rel_overlap: torch.Tensor | None = None,
        need_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        g, t, _ = x.shape
        q = self.q_proj(x).view(g, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(g, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(g, t, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        if self.relation:
            if rel_dist is None or rel_overlap is None:
                raise ValueError("R arm requires rel_dist and rel_overlap")
            logits = logits + self.struct_bias(type_ids, rel_dist, rel_overlap)
        # Key padding is the only mask.  Every molecule has at least one token,
        # so no query row can be all-masked.  Padding rows are zeroed by the
        # caller after the block and never enter a pooled statistic.
        key_mask = ~valid
        logits = logits.masked_fill(key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(g, t, self.dim)
        out = self.out_proj(out)
        return out, (attn if need_attention else None)


class TokenBlock(nn.Module):
    """One independent pre-LN residual block.

    The two blocks in a model are separate module instances; no layer object is
    reused.  Padding rows are zeroed after the block.
    """

    def __init__(self, variant: str, dim: int = DIM, heads: int = HEADS):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.norm1 = nn.LayerNorm(dim, eps=LN_EPS, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(dim, eps=LN_EPS, elementwise_affine=True)
        if variant == "S":
            self.mix = ExtraFFN(dim)
        elif variant == "T":
            self.mix = AttnMix(dim, heads, relation=False)
        else:
            self.mix = AttnMix(dim, heads, relation=True)
        self.ffn = ResidualFFN(dim)

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        type_ids: torch.Tensor,
        rel_dist: torch.Tensor | None = None,
        rel_overlap: torch.Tensor | None = None,
        need_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n1 = self.norm1(x)
        if self.variant == "S":
            mix_out, attn = self.mix(n1), None
        else:
            mix_out, attn = self.mix(
                n1, valid, type_ids, rel_dist, rel_overlap, need_attention=need_attention
            )
        u = x + mix_out
        h = u + self.ffn(self.norm2(u))
        if not bool(valid.all()):
            h = h * valid.unsqueeze(-1).to(h.dtype)
        return h, attn


class NewReader(nn.Module):
    """612 -> 96 -> 96 -> 1, SiLU, no dropout / BN."""

    def __init__(self, in_dim: int = 612, hidden: int = DIM):
        super().__init__()
        self.lins = nn.ModuleList(
            [
                nn.Linear(in_dim, hidden, bias=True),
                nn.Linear(hidden, hidden, bias=True),
                nn.Linear(hidden, 1, bias=True),
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.silu(self.lins[0](z))
        z = F.silu(self.lins[1](z))
        return self.lins[2](z).view(-1)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class STAObjectModel(src.BJOC):
    """BJOC-Null trunk + explicit object tokens + two token blocks + new reader."""

    def __init__(
        self,
        config: src.BJOCConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        variant: str,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        super().__init__(config, global_mean, global_scale, src.NULL)
        self.variant = variant
        # Remove the old 321->...->1 reader completely.
        del self.reader
        self.type_embedding = nn.Parameter(torch.empty(N_TYPES, DIM))
        self.blocks = nn.ModuleList([TokenBlock(variant) for _ in range(N_BLOCKS)])
        self.reader = NewReader(612, DIM)

    # -- token encoder -------------------------------------------------------
    def encode_tokens(
        self,
        h0: torch.Tensor,  # [G, T, 96], padding rows may be zero
        valid: torch.Tensor,  # [G, T] bool
        type_ids: torch.Tensor,  # [G, T] long
        rel_dist: torch.Tensor | None = None,
        rel_overlap: torch.Tensor | None = None,
        need_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None]]:
        h = h0
        attns: list[torch.Tensor | None] = []
        for block in self.blocks:
            h, attn = block(h, valid, type_ids, rel_dist, rel_overlap, need_attention=need_attention)
            attns.append(attn)
        # Type-specific sums; absent types stay exactly zero.
        sums = []
        for t in range(N_TYPES):
            mask = valid & (type_ids == t)
            sums.append((h * mask.unsqueeze(-1).to(h.dtype)).sum(dim=1))
        return h, torch.stack(sums, dim=1), attns  # [G,3,96]

    # -- forward -------------------------------------------------------------
    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        need_attention: bool = False,
    ) -> dict[str, Any]:
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)

        # ---- P3 statistics bypass (original Null grouping + composer) ------
        W, V, n_b = self.group_to_bonds(w, batch, src.NULL)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]],
            dim=1,
        )
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        reader_in_p3 = torch.cat([z, a_sum, a_sq, g], dim=1)

        # ---- object-token sequence ----------------------------------------
        token_type = batch["token_type"]
        h0_flat = torch.cat([a, b, w], dim=0) + self.type_embedding[token_type]
        tmax = int(batch["token_valid"].shape[1])
        h0 = h0_flat.new_zeros((n_graphs, tmax, DIM))
        if h0_flat.numel() > 0:
            h0[batch["token_batch"], batch["token_slot"]] = h0_flat
        valid = batch["token_valid"]
        type_ids = batch["token_type_padded"]
        rel_dist = batch["rel_dist"].long() if "rel_dist" in batch else None
        rel_overlap = batch["rel_overlap"].long() if "rel_overlap" in batch else None
        h2, type_sums, attns = self.encode_tokens(
            h0, valid, type_ids, rel_dist, rel_overlap, need_attention=need_attention
        )
        counts = batch["token_counts"].to(dtype=a.dtype)
        reader_in = torch.cat(
            [
                reader_in_p3,
                type_sums[:, 0],
                type_sums[:, 1],
                type_sums[:, 2],
                torch.log1p(counts[:, 0]).unsqueeze(1),
                torch.log1p(counts[:, 1]).unsqueeze(1),
                torch.log1p(counts[:, 2]).unsqueeze(1),
            ],
            dim=1,
        )
        return {
            "a": a,
            "b": b,
            "w": w,
            "W": W,
            "V": V,
            "n_b": n_b,
            "c": c,
            "z": z,
            "reader_in_p3": reader_in_p3,
            "h0": h0,
            "h2": h2,
            "type_sums": type_sums,
            "reader_in": reader_in,
            "attention": attns,
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        return_features: bool = False,
        need_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        feats = self.forward_features(batch, need_attention=need_attention)
        pred = self.reader(feats["reader_in"])
        if return_features:
            return pred, feats
        return pred


# ---------------------------------------------------------------------------
# deterministic initialisation / construction
# ---------------------------------------------------------------------------


def _generator(seed: int, tag: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) * 100003 + int(tag) * 7919 + INIT_OFFSET)


def _init_linear(lin: nn.Linear, seed: int, tag: int) -> None:
    fan_in = int(lin.weight.shape[1])
    bound = 1.0 / math.sqrt(float(fan_in))
    gen = _generator(seed, tag)
    with torch.no_grad():
        lin.weight.uniform_(-bound, bound, generator=gen)
        if lin.bias is not None:
            lin.bias.zero_()


def _init_ln(ln: nn.LayerNorm) -> None:
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.zero_()


def init_new_modules(model: STAObjectModel, seed: int) -> None:
    """Initialise every new tensor from dedicated generators.

    Common modules receive the same tag for S/T/R; T/R MHA projections receive
    the same tags; the R bias tables are exactly zero.  This function never
    touches the global RNG or any P3 encoder tensor.
    """
    with torch.no_grad():
        bound = 1.0 / math.sqrt(float(DIM))
        model.type_embedding.uniform_(-bound, bound, generator=_generator(seed, 10))
        for i, lin in enumerate(model.reader.lins):
            _init_linear(lin, seed, 20 + i)
        for li, block in enumerate(model.blocks):
            _init_ln(block.norm1)
            _init_ln(block.norm2)
            _init_linear(block.ffn.lin1, seed, 500 + li * 10 + 0)
            _init_linear(block.ffn.lin2, seed, 500 + li * 10 + 1)
            if model.variant == "S":
                _init_linear(block.mix.lin1, seed, 400 + li * 10 + 0)
                _init_linear(block.mix.lin2, seed, 400 + li * 10 + 1)
                block.mix.gamma.fill_(1.0)
            else:
                _init_linear(block.mix.q_proj, seed, 300 + li * 10 + 0)
                _init_linear(block.mix.k_proj, seed, 300 + li * 10 + 1)
                _init_linear(block.mix.v_proj, seed, 300 + li * 10 + 2)
                _init_linear(block.mix.out_proj, seed, 300 + li * 10 + 3)
                if model.variant == "R":
                    block.mix.struct_bias.type_bias.zero_()
                    block.mix.struct_bias.dist_bias.zero_()
                    block.mix.struct_bias.overlap_bias.zero_()


def copy_shared_(model: STAObjectModel, base: src.BJOC) -> None:
    """Copy the P3 non-reader tensors (four modules + two buffers) bit-exactly."""
    own = model.state_dict()
    prefixes = ("atom_encoder.", "bond_encoder.", "junction_encoder.", "group_composer.")
    copied = 0
    with torch.no_grad():
        for key, value in base.state_dict().items():
            if not (key.startswith(prefixes) or key in ("global_mean", "global_scale")):
                continue
            if key not in own:
                raise RuntimeError(f"shared tensor {key} missing in STA model")
            if own[key].shape != value.shape:
                raise RuntimeError(f"shape mismatch for {key}: {own[key].shape} vs {value.shape}")
            own[key].copy_(value)
            copied += 1
    expected = sum(
        1 for key in base.state_dict() if key.startswith(prefixes) or key in ("global_mean", "global_scale")
    )
    if copied != expected:
        raise RuntimeError(f"copied {copied} shared tensors, expected {expected}")


def build_sta_model(
    variant: str,
    *,
    seed: int,
    norm: Mapping[str, Any],
    ssl_state: Mapping[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[src.BJOC, STAObjectModel, dict[str, Any]]:
    """Build the exact P3 supervised start, load the seed-matched SSL encoder,
    then add this round's arm without perturbing any shared tensor."""
    import bjoc_masked_pretraining_v1 as mp
    import bjoc_perfmech_long_v1 as bp

    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}")
    rng_before = torch.get_rng_state()
    base = bp.build_bjoc_null(seed, norm, "cpu")
    scratch = {
        "init_full_sha256": src.state_dict_hash(base),
        "init_group_state_sha256": mp.group_state_hash(base),
        "init_encoder_sha256": bp.encoder_group_hash(base),
    }
    if ssl_state is not None:
        mp.load_encoder_state(base, ssl_state, "cpu")
    encoder_used = bp.encoder_group_hash(base)

    torch.manual_seed(int(seed))
    model = STAObjectModel(
        src.BJOCConfig(),
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        variant,
    )
    torch.set_rng_state(rng_before)
    copy_shared_(model, base)
    init_new_modules(model, seed)
    model = model.to(device)

    n_params = int(sum(p.numel() for p in model.parameters()))
    expected = EXPECTED_PARAMETERS[variant]
    if n_params != expected:
        raise SystemExit(f"{variant} parameter count {n_params} != {expected}")
    info = {
        "variant": variant,
        "seed": int(seed),
        "parameters": n_params,
        "expected_parameters": expected,
        "scratch": scratch,
        "encoder_used_sha256": encoder_used,
        "shared_matches_base": shared_matches_base(model, base),
        "shared_state_sha256": shared_state_sha256(model),
        "extra_state_sha256": extra_state_sha256(model),
        "global_rng_untouched": bool(torch.equal(rng_before, torch.get_rng_state())),
    }
    return base, model, info


# ---------------------------------------------------------------------------
# audits / hashes
# ---------------------------------------------------------------------------


def shared_matches_base(model: STAObjectModel, base: src.BJOC) -> bool:
    own = model.state_dict()
    prefixes = ("atom_encoder.", "bond_encoder.", "junction_encoder.", "group_composer.")
    for key, value in base.state_dict().items():
        if key.startswith(prefixes) or key in ("global_mean", "global_scale"):
            if key not in own or not torch.equal(own[key], value):
                return False
    return True


def _hash_tensors(items: Sequence[tuple[str, torch.Tensor]]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(items, key=lambda kv: kv[0]):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def shared_state_sha256(model: STAObjectModel) -> str:
    prefixes = ("atom_encoder.", "bond_encoder.", "junction_encoder.", "group_composer.")
    items = [
        (k, v)
        for k, v in model.state_dict().items()
        if k.startswith(prefixes) or k in ("global_mean", "global_scale")
    ]
    return _hash_tensors(items)


def extra_state_sha256(model: STAObjectModel) -> str:
    prefixes = ("type_embedding", "blocks.", "reader.")
    items = [(k, v) for k, v in model.state_dict().items() if k.startswith(prefixes)]
    return _hash_tensors(items)


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32).tobytes()
    ).hexdigest()


def parameter_audit(model: STAObjectModel) -> dict[str, Any]:
    cfg = src.BJOCConfig()
    inv = cfg.inventory()
    shared_keys = ("atom_encoder", "bond_encoder", "junction_encoder", "group_composer")
    groups = {
        "atom_encoder": sum(p.numel() for p in model.atom_encoder.parameters()),
        "bond_encoder": sum(p.numel() for p in model.bond_encoder.parameters()),
        "junction_encoder": sum(p.numel() for p in model.junction_encoder.parameters()),
        "group_composer": sum(p.numel() for p in model.group_composer.parameters()),
        "reader": sum(p.numel() for p in model.reader.parameters()),
        "type_embedding": int(model.type_embedding.numel()),
        "blocks": sum(p.numel() for p in model.blocks.parameters()),
        "struct_bias": sum(p.numel() for name, p in model.named_parameters() if ".mix.struct_bias." in name),
    }
    groups["mix"] = sum(
        p.numel() for name, p in model.named_parameters() if name.startswith("blocks.") and ".mix." in name
    )
    groups["token_ffn"] = sum(
        p.numel() for name, p in model.named_parameters() if name.startswith("blocks.") and ".ffn." in name
    )
    actual = int(sum(p.numel() for p in model.parameters()))
    shared_ok = all(groups[k] == inv[k] for k in shared_keys)
    return {
        "groups": groups,
        "actual_parameters": actual,
        "expected_parameters": EXPECTED_PARAMETERS[model.variant],
        "p3_nonreader_parameters": P3_NONREADER,
        "p3_reader_removed_parameters": P3_READER,
        "shared_groups_match_p3": shared_ok,
        "old_reader_present": any(
            k.startswith("reader.lins.") and k.endswith(".weight") and tuple(v.shape) == (256, 321)
            for k, v in model.state_dict().items()
        ),
        "parameter_exact": actual == EXPECTED_PARAMETERS[model.variant] and shared_ok,
        "reader_in_dim": 612,
        "z_p3_dim": 321,
    }


def module_audit(model: STAObjectModel) -> dict[str, Any]:
    classes: list[str] = []
    offending: list[str] = []
    extra: list[str] = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        classes.append(f"{name}:{cls}")
        for token in FORBIDDEN_TOKENS:
            if token in cls:
                offending.append(f"{name}:{cls}")
        if cls in ALLOWED_EXTRA_CLASSES:
            extra.append(f"{name}:{cls}")
    allowed = {
        "_MLP",
        "Linear",
        "ModuleList",
        "LayerNorm",
        "SiLU",
        "STAObjectModel",
        "TokenBlock",
        "AttnMix",
        "ExtraFFN",
        "ResidualFFN",
        "StructuralBias",
        "NewReader",
    }
    leaks = [c for c in classes if c.split(":")[-1] not in allowed]
    body = Path(__file__).read_text(encoding="utf-8")
    for token in (
        "nn." + "MultiheadAttention",
        "Message" + "Passing(",
        "torch" + "_geometric",
        "nn." + "Batch" + "Norm",
    ):
        if token in body:
            offending.append(f"source:{token}")
    return {
        "n_modules": len(classes),
        "classes": sorted({c.split(":")[-1] for c in classes}),
        "extra_modules": extra,
        "offending": offending,
        "unexpected_classes": leaks,
        "pass": not offending and not leaks,
    }


def relation_parameter_audit(model: STAObjectModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for li, block in enumerate(model.blocks):
        if model.variant != "R":
            continue
        sb = block.mix.struct_bias
        out[f"layer{li}"] = {
            "type_bias": {
                "shape": list(sb.type_bias.shape),
                "nonzero": int((sb.type_bias != 0).sum()),
                "abs_sum": float(sb.type_bias.detach().abs().sum()),
                "l2": float(sb.type_bias.detach().norm()),
            },
            "dist_bias": {
                "shape": list(sb.dist_bias.shape),
                "nonzero": int((sb.dist_bias != 0).sum()),
                "abs_sum": float(sb.dist_bias.detach().abs().sum()),
                "l2": float(sb.dist_bias.detach().norm()),
                "values": [float(x) for x in sb.dist_bias.detach().cpu().reshape(-1)],
            },
            "overlap_bias": {
                "shape": list(sb.overlap_bias.shape),
                "nonzero": int((sb.overlap_bias != 0).sum()),
                "abs_sum": float(sb.overlap_bias.detach().abs().sum()),
                "l2": float(sb.overlap_bias.detach().norm()),
                "values": [float(x) for x in sb.overlap_bias.detach().cpu().reshape(-1)],
            },
        }
    out["total_parameters"] = int(
        sum(p.numel() for name, p in model.named_parameters() if ".mix.struct_bias." in name)
    )
    return out
