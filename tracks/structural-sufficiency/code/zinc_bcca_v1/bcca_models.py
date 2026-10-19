"""ZINC bond-conditioned context attention v1 — model definitions.

Round: ``zinc-bond-conditioned-context-attention-v1``
Protocol: ``results/zinc_bond_conditioned_context_attention_v1/PROTOCOL.md``
Track: ``tracks/structural-sufficiency``

This file is isolated from every frozen model file.  It imports (read-only)
``structural_relation_composer_v1`` (BJOC), ``structural_relation_objects_v1``,
``bjoc_perfmech_long_v1`` (P3/R3 builder) and ``bjoc_masked_pretraining_v1``
(SSL encoder loader), and defines exactly one new class hierarchy.

The frozen P3 Null grouping inside the composer is::

    n_e = deg(u) + deg(v) - 2                     (junction count of bond e)
    W_e = n_e * mean_j(w_j)                       (j over the molecule's junctions)
    V_e = n_e * mean_j(w_j^2)                     (per-dimension raw 2nd moment)

Only the W/V slots of the composer input are replaced:

* ``C`` (conditional)  W_e = n_e * sum_j A_ej w_j ,  V_e = n_e * sum_j A_ej w_j^2
* ``G`` (global)       A is computed exactly as in C, then marginalised to
                       p_j = sum_e n_e A_ej / sum_e n_e and shared by all bonds.
* ``F`` (capacity)     P3's Null W/V are kept; only the composer's ``b`` slot is
                       replaced by ``b + U SiLU(T LN(b))``.
* ``P3``               the untouched frozen Null grouping (used for the loop
                       parity gate and, only if the execution regime changed, a
                       same-round baseline).

Attention (single head, no bias, no dropout, no positional/distance/ownership
mask, no V/output projection, no gate, no extra reader channel)::

    q_e = Q LN(b_e),  k_j = K LN(w_j),  A_ej = softmax_j(q_e.k_j / sqrt(32))

with ``Q in R^{32x96}``, ``K in R^{32x96}`` and non-affine LayerNorm(96, eps=1e-5).
All pairs are restricted to the same molecule; there is no virtual node and no
cross-molecule interaction.
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

_CODE_DIR = Path(__file__).resolve().parent
REPO = _CODE_DIR.parents[3]
for _p in (_CODE_DIR, REPO / "tracks/structure-contribution/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

#: frozen architecture constants (PROTOCOL §3)
DIM = 96            # == src.BJOCConfig().compose_out
ATTN_DIM = 32       # r
LN_EPS = 1e-5
INIT_OFFSET = 9119  # dedicated generator for the extra module

VARIANTS = ("P3", "C", "G", "F")
NEW_VARIANTS = ("C", "G", "F")
ATTENTION_VARIANTS = ("C", "G")
MODES = ("P3", "C", "G", "uniform", "F")

#: extra parameters: 2 * 96 * 32 = 6144, independent of the variant
EXTRA_PARAMETERS = 2 * DIM * ATTN_DIM
EXPECTED_PARAMETERS = {
    "P3": 413505,
    "F": 413505 + EXTRA_PARAMETERS,
    "G": 413505 + EXTRA_PARAMETERS,
    "C": 413505 + EXTRA_PARAMETERS,
}

#: module-class tokens that must never appear in this round's models
FORBIDDEN_TOKENS = (
    "MessagePassing",
    "GINE",
    "GCN",
    "GAT",
    "SAGEConv",
    "Transformer",
    "MultiheadAttention",
    "GRU",
    "LSTM",
    "RNN",
    "BatchNorm",
    "Sequential_state",
)
#: the only extra module classes this round may introduce
ALLOWED_EXTRA_CLASSES = ("BondJunctionAttention", "BondFFN")


# ---------------------------------------------------------------------------
# per-molecule pair index (block-diagonal bond x junction, no cross-molecule)
# ---------------------------------------------------------------------------


def bond_junction_counts(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """``n_e`` exactly as in the frozen Null builder (same index_add_ call)."""
    n_bonds = int(batch["bond_feat"].shape[0])
    jb_flat = batch["junc_bond"].reshape(-1)
    n_b = torch.zeros(n_bonds, device=jb_flat.device, dtype=torch.float32)
    n_b.index_add_(0, jb_flat, torch.ones_like(jb_flat, dtype=torch.float32))
    return n_b


def pair_index(batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """All (bond, junction) pairs that share a molecule, as flat indices.

    ``batch_objects`` emits bonds and junctions grouped by molecule, so each
    molecule's junctions occupy one contiguous index range.  A bond in molecule
    ``g`` is paired with every junction of ``g`` and nothing else.
    """
    bb = batch["bond_batch"]
    jb = batch["junc_batch"]
    n_graphs = int(batch["n_graphs"])
    n_b, n_j = int(bb.numel()), int(jb.numel())
    if n_b == 0 or n_j == 0:
        return None
    jcount = torch.bincount(jb, minlength=n_graphs)
    joff = torch.cumsum(jcount, 0) - jcount
    per_bond = jcount[bb]
    total = int(per_bond.sum().item())
    if total == 0:
        return None
    row = torch.repeat_interleave(torch.arange(n_b, device=bb.device), per_bond)
    start = torch.repeat_interleave(joff[bb], per_bond)
    inside = torch.repeat_interleave(torch.cumsum(per_bond, 0) - per_bond, per_bond)
    col = start + (torch.arange(total, device=bb.device) - inside)
    return row, col


def segment_softmax(scores: torch.Tensor, row: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Softmax over the variable-length junction list of each bond.

    Bonds with no junction produce no pair row; they are only read through
    scatter-adds, which stay exactly zero.  No all-masked softmax is ever
    evaluated, so no NaN can be produced.
    """
    smax = torch.full((n_rows,), float("-inf"), device=scores.device, dtype=scores.dtype)
    smax = smax.scatter_reduce(0, row, scores, reduce="amax", include_self=True)
    e = torch.exp(scores - smax[row])
    den = torch.zeros(n_rows, device=scores.device, dtype=scores.dtype)
    den.index_add_(0, row, e)
    return e / den[row].clamp_min(1e-30)


# ---------------------------------------------------------------------------
# extra modules
# ---------------------------------------------------------------------------


class BondJunctionAttention(nn.Module):
    """Single-head, bias-free bond->junction cross-attention over one molecule.

    Returns the composer's W/V slots.  ``mode`` selects the mapping:

    * ``C``       per-bond distribution ``A_ej`` (conditional)
    * ``G``       graph marginal ``p_j`` shared by every bond (global)
    * ``uniform`` ``A_ej = 1/J_g`` under the same code path (P3-equivalent
                  mapping; mechanism diagnostic only)
    """

    def __init__(self, dim: int = DIM, attn_dim: int = ATTN_DIM):
        super().__init__()
        self.dim = int(dim)
        self.attn_dim = int(attn_dim)
        self.norm_b = nn.LayerNorm(dim, eps=LN_EPS, elementwise_affine=False)
        self.norm_w = nn.LayerNorm(dim, eps=LN_EPS, elementwise_affine=False)
        self.q = nn.Linear(dim, attn_dim, bias=False)
        self.k = nn.Linear(dim, attn_dim, bias=False)
        self.register_buffer("temperature", torch.tensor(math.sqrt(float(attn_dim))))

    def scores(self, b: torch.Tensor, w: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        q = self.q(self.norm_b(b))
        k = self.k(self.norm_w(w))
        return (q[row] * k[col]).sum(-1) / self.temperature

    def forward(
        self,
        b: torch.Tensor,
        w: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        n_b: torch.Tensor,
        mode: str = "C",
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        if mode not in ("C", "G", "uniform"):
            raise ValueError(f"unknown attention mode {mode!r}")
        n_bonds = int(b.shape[0])
        n_graphs = int(batch["n_graphs"])
        n_junc = int(batch["junc_feat"].shape[0])
        weight = w.new_zeros((n_bonds, self.dim))
        weight_sq = w.new_zeros((n_bonds, self.dim))
        idx = pair_index(batch)
        if idx is None:
            if return_attention:
                return weight, weight_sq, {
                    "row": torch.zeros(0, dtype=torch.long),
                    "col": torch.zeros(0, dtype=torch.long),
                    "A": w.new_zeros(0),
                    "p": w.new_zeros(n_junc),
                }
            return weight, weight_sq, None

        row, col = idx
        if mode == "uniform":
            jcount = torch.bincount(batch["junc_batch"], minlength=n_graphs)
            a = 1.0 / jcount[batch["bond_batch"]][row].to(w.dtype)
        else:
            scores = self.scores(b, w, row, col)
            a = segment_softmax(scores, row, n_bonds)

        p = None
        if mode in ("C", "uniform"):
            weight.index_add_(0, row, w[col] * a.unsqueeze(1))
            weight_sq.index_add_(0, row, (w[col] * w[col]) * a.unsqueeze(1))
            weight = n_b.unsqueeze(1) * weight
            weight_sq = n_b.unsqueeze(1) * weight_sq
        else:
            jb = batch["junc_batch"]
            num = w.new_zeros(n_junc)
            num.index_add_(0, col, n_b[row] * a)
            den = w.new_zeros(n_graphs)
            den.index_add_(0, batch["bond_batch"], n_b)
            p = num / den[jb].clamp_min(1e-12)
            gw = w.new_zeros((n_graphs, self.dim))
            gw.index_add_(0, jb, w * p.unsqueeze(1))
            weight = n_b.unsqueeze(1) * gw[batch["bond_batch"]]
            gws = w.new_zeros((n_graphs, self.dim))
            gws.index_add_(0, jb, (w * w) * p.unsqueeze(1))
            weight_sq = n_b.unsqueeze(1) * gws[batch["bond_batch"]]

        if return_attention:
            if p is None:
                # the graph marginal of the same A (needed for D = ||A_e - p||);
                # this is a diagnostic-only reduction and never enters the model.
                jb = batch["junc_batch"]
                num = w.new_zeros(n_junc)
                num.index_add_(0, col, n_b[row] * a)
                den = w.new_zeros(n_graphs)
                den.index_add_(0, batch["bond_batch"], n_b)
                p = num / den[jb].clamp_min(1e-12)
            return weight, weight_sq, {
                "row": row.detach(),
                "col": col.detach(),
                "A": a.detach(),
                "p": p.detach(),
            }
        return weight, weight_sq, None


class BondFFN(nn.Module):
    """Parameter-matched capacity control: ``b + U SiLU(T LN(b))``.

    ``T: [32, 96]``, ``U: [96, 32]``, no bias, non-affine LayerNorm.  Applied
    only to the composer's ``b`` slot; ``w`` is encoded from the untouched
    ``b`` and nothing is fed back to the junction encoder.
    """

    def __init__(self, dim: int = DIM, hidden: int = ATTN_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=LN_EPS, elementwise_affine=False)
        self.t = nn.Linear(dim, hidden, bias=False)
        self.u = nn.Linear(hidden, dim, bias=False)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return b + self.u(F.silu(self.t(self.norm(b))))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class BondContextBJOC(src.BJOC):
    """BJOC-Null with the composer's context moments replaced (or untouched)."""

    def __init__(
        self,
        config: src.BJOCConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        variant: str = "P3",
        attn_dim: int = ATTN_DIM,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        super().__init__(config, global_mean, global_scale, src.NULL)
        self.variant = variant
        self.attn = BondJunctionAttention(config.compose_out, attn_dim) if variant in ATTENTION_VARIANTS else None
        self.ffn = BondFFN(config.compose_out, attn_dim) if variant == "F" else None

    # -- context mapping -----------------------------------------------------
    def context(
        self,
        b: torch.Tensor,
        w: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        mode: str,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if self.variant in ("P3", "F"):
            W, V, n_b = self.group_to_bonds(w, batch, src.NULL)
            return W, V, n_b, None
        n_b = bond_junction_counts(batch).to(w.dtype)
        attention_mode = {"C": "C", "G": "G", "uniform": "uniform"}.get(mode, "C")
        W, V, aux = self.attn(b, w, batch, n_b, mode=attention_mode, return_attention=return_attention)
        return W, V, n_b, aux

    # -- forward -------------------------------------------------------------
    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        mode: str | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        mode = self.variant if mode is None else mode
        if self.variant == "F" and mode not in ("F",):
            raise ValueError("the F arm has no alternative context mapping")
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)
        W, V, n_b, aux = self.context(b, w, batch, mode, return_attention=return_attention)
        b_compose = self.ffn(b) if self.variant == "F" else b
        compose_in = torch.cat(
            [b_compose, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]],
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
        reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        feats = {
            "a": a,
            "b": b,
            "b_compose": b_compose,
            "w": w,
            "W": W,
            "V": V,
            "n_b": n_b,
            "c": c,
            "z": z,
            "reader_in": reader_in,
        }
        if aux is not None:
            feats["attn"] = aux
        return feats

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        return_features: bool = False,
        mode: str | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = self.forward_features(
            batch, organization_mode, mode=mode, return_attention=return_attention
        )
        pred = self.reader(feats["reader_in"]).view(-1)
        if return_features:
            return pred, feats
        return pred


# ---------------------------------------------------------------------------
# builders and deterministic initialisation
# ---------------------------------------------------------------------------


def _uniform_matrix(seed: int, out_dim: int = ATTN_DIM, in_dim: int = DIM) -> torch.Tensor:
    """``nn.Linear`` default weight range on a dedicated generator."""
    g = torch.Generator().manual_seed(int(seed) + INIT_OFFSET)
    bound = 1.0 / math.sqrt(float(in_dim))
    return torch.empty((out_dim, in_dim), dtype=torch.float32).uniform_(-bound, bound, generator=g)


def init_extra(model: BondContextBJOC, seed: int) -> dict[str, Any]:
    """Q=0 (U=0) and K/T from the shared dedicated generator."""
    with torch.no_grad():
        if model.variant in ATTENTION_VARIANTS:
            model.attn.q.weight.zero_()
            model.attn.k.weight.copy_(_uniform_matrix(seed))
        elif model.variant == "F":
            model.ffn.t.weight.copy_(_uniform_matrix(seed))
            model.ffn.u.weight.zero_()
    return extra_init_record(model)


def copy_shared_(model: BondContextBJOC, base: src.BJOC) -> None:
    """Copy every frozen BJOC tensor (five modules + buffers) bit-exactly."""
    own = model.state_dict()
    missing = [k for k in base.state_dict() if k not in own]
    if missing:
        raise RuntimeError(f"shared tensors missing in the extended model: {missing}")
    with torch.no_grad():
        for key, value in base.state_dict().items():
            if own[key].shape != value.shape:
                raise RuntimeError(f"shape mismatch for {key}")
            own[key].copy_(value)


def build_variant(
    variant: str,
    *,
    seed: int,
    norm: Mapping[str, Any],
    ssl_state: Mapping[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[src.BJOC, BondContextBJOC, dict[str, Any]]:
    """Build the exact P3 supervised init, optionally load SSL, then add the arm.

    Order matters: the frozen base (and its SSL encoder) is fully built first;
    the extra module is constructed afterwards under a saved/restored global RNG
    state and initialised from its own generator.  No shared tensor, and no
    batch-order RNG stream, can be perturbed by the extra module.
    """
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
    model = BondContextBJOC(
        src.BJOCConfig(),
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        variant,
    )
    torch.set_rng_state(rng_before)
    copy_shared_(model, base)
    init_extra(model, seed)
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
        "shared_matches_base": shared_state_sha256(model) == src.state_dict_hash(base),
        "shared_state_sha256": shared_state_sha256(model),
        "base_state_sha256": src.state_dict_hash(base),
        "extra_init": extra_init_record(model),
        "global_rng_untouched": bool(torch.equal(rng_before, torch.get_rng_state())),
    }
    return base, model, info


# ---------------------------------------------------------------------------
# hashing / audits
# ---------------------------------------------------------------------------


def shared_state_sha256(model: src.BJOC) -> str:
    """Hashing rule of ``src.state_dict_hash`` over the shared five modules only."""
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(("attn.", "ffn.")):
            continue
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def extra_state_sha256(model: BondContextBJOC) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if not key.startswith(("attn.", "ffn.")):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def extra_init_record(model: BondContextBJOC) -> dict[str, Any]:
    rec: dict[str, Any] = {"variant": model.variant}
    if model.variant in ATTENTION_VARIANTS:
        rec["q_weight_abs_sum"] = float(model.attn.q.weight.detach().abs().sum())
        rec["k_weight_sha256"] = tensor_sha256(model.attn.k.weight.detach())
        rec["q_parameters"] = int(model.attn.q.weight.numel())
        rec["k_parameters"] = int(model.attn.k.weight.numel())
    elif model.variant == "F":
        rec["u_weight_abs_sum"] = float(model.ffn.u.weight.detach().abs().sum())
        rec["t_weight_sha256"] = tensor_sha256(model.ffn.t.weight.detach())
        rec["t_parameters"] = int(model.ffn.t.weight.numel())
        rec["u_parameters"] = int(model.ffn.u.weight.numel())
    rec["extra_state_sha256"] = extra_state_sha256(model)
    return rec


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32).tobytes()).hexdigest()


def parameter_audit(model: BondContextBJOC) -> dict[str, Any]:
    groups = {
        "atom_encoder": sum(p.numel() for p in model.atom_encoder.parameters()),
        "bond_encoder": sum(p.numel() for p in model.bond_encoder.parameters()),
        "junction_encoder": sum(p.numel() for p in model.junction_encoder.parameters()),
        "group_composer": sum(p.numel() for p in model.group_composer.parameters()),
        "reader": sum(p.numel() for p in model.reader.parameters()),
        "attn": 0 if model.attn is None else sum(p.numel() for p in model.attn.parameters()),
        "ffn": 0 if model.ffn is None else sum(p.numel() for p in model.ffn.parameters()),
    }
    expected = src.BJOCConfig().inventory()
    actual = int(sum(p.numel() for p in model.parameters()))
    return {
        "groups": groups,
        "actual_parameters": actual,
        "expected_parameters": EXPECTED_PARAMETERS[model.variant],
        "extra_parameters": int(groups["attn"] + groups["ffn"]),
        "parameter_exact": actual == EXPECTED_PARAMETERS[model.variant]
        and all(groups[k] == expected[k] for k in
                ("atom_encoder", "bond_encoder", "junction_encoder", "group_composer", "reader")),
        "reader_in_dim": int(model.config.reader_in),
    }


def module_audit(model: BondContextBJOC) -> dict[str, Any]:
    """Allow this round's attention/FFN; forbid MP, extra statistics, bypasses."""
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
    src_path = Path(__file__)
    body = src_path.read_text(encoding="utf-8")
    usage = [
        "nn." + "MultiheadAttention",
        "Message" + "Passing(",
        "Transformer" + "Encoder(",
        "nn." + "Transformer",
        "torch" + "_geometric",
    ]
    for token in usage:
        if token in body:
            offending.append(f"source:{token}")
    allowed = {"_MLP", "Linear", "ModuleList", "Dropout", "LayerNorm",
               "BondContextBJOC", "BondJunctionAttention", "BondFFN"}
    leaks = [c for c in classes if c.split(":")[-1] not in allowed]
    return {
        "n_modules": len(classes),
        "classes": sorted({c.split(":")[-1] for c in classes}),
        "extra_modules": extra,
        "offending": offending,
        "unexpected_classes": leaks,
        "pass": not offending and not leaks,
    }


# ---------------------------------------------------------------------------
# attention diagnostics (label-free)
# ---------------------------------------------------------------------------


def attention_diagnostics(aux: Mapping[str, Any], batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Per-molecule D, normalised entropy, max weight (label-free).

    ``D = sum_e n_e ||A_e - p||_1 / sum_e n_e`` is reported per molecule and
    molecule-weighted.  ``p`` is the graph marginal from the same forward pass.
    """
    n_graphs = int(batch["n_graphs"])
    n_b = int(batch["bond_feat"].shape[0])
    if n_b == 0 or int(aux["A"].numel()) == 0:
        return {"D_molecule_mean": None, "D_molecule_weighted": None,
                "entropy_norm_mean": None, "max_weight_mean": None,
                "n_molecules": n_graphs, "n_pairs": 0}
    row, col, A = aux["row"], aux["col"], aux["A"]
    bb = batch["bond_batch"]
    jb = batch["junc_batch"]
    n_b_t = bond_junction_counts(batch).to(A.device).to(A.dtype)
    jcount = torch.bincount(jb, minlength=n_graphs).to(A.dtype)

    denom = torch.zeros(n_graphs, device=A.device, dtype=torch.float64)
    denom.index_add_(0, bb, n_b_t.double())

    d_mol: float | None = None
    d_weighted: float | None = None
    if aux.get("p") is not None:
        p = aux["p"]
        absdiff = (A.double() - p.double()[col]).abs()
        w = n_b_t.double()[row]
        per_bond = torch.zeros(n_b, device=A.device, dtype=torch.float64)
        per_bond.index_add_(0, row, w * absdiff)
        num = torch.zeros(n_graphs, device=A.device, dtype=torch.float64)
        num.index_add_(0, bb, per_bond)
        d_mol = float((num / denom.clamp_min(1e-12)).mean())
        d_weighted = float(num.sum() / denom.sum().clamp_min(1e-12))

    logj = torch.log(torch.clamp(jcount, min=2.0)).to(A.dtype)
    ent = -(A * torch.log(A.clamp_min(1e-30)))
    per_bond_ent = torch.zeros(n_b, device=A.device, dtype=A.dtype)
    per_bond_ent.index_add_(0, row, ent)
    norm_ent = per_bond_ent / logj[bb]

    has_pair = torch.zeros(n_b, device=A.device, dtype=torch.bool)
    has_pair[row] = True
    m = torch.full((n_b,), float("-inf"), device=A.device, dtype=A.dtype)
    m = m.scatter_reduce(0, row, A, reduce="amax", include_self=True)
    return {
        "D_molecule_mean": d_mol,
        "D_molecule_weighted": d_weighted,
        "entropy_norm_mean": float(norm_ent[has_pair].mean()),
        "max_weight_mean": float(m[has_pair].mean()),
        "n_molecules": n_graphs,
        "n_pairs": int(A.numel()),
    }


def content_dispersion(W: torch.Tensor, V: torch.Tensor, n_b: torch.Tensor,
                       batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Relative across-bond spread of ``W_e/n_e`` (and V) inside each molecule.

    Under the global-marginal mapping the content ``W_e/n_e`` is identical for
    every bond of a molecule, so this is ~0 for G and positive for C.  ``n_e``
    is divided out so the spread is not just molecular size.
    """
    n_graphs = int(batch["n_graphs"])
    n_bonds = int(W.shape[0])
    if n_bonds == 0:
        return {"W_content_dispersion": 0.0, "V_content_dispersion": 0.0}
    safe = n_b.clamp_min(1.0).unsqueeze(1)
    bb = batch["bond_batch"]
    out: dict[str, float] = {}
    for name, X in (("W", W / safe), ("V", V / safe)):
        # float64 and centered residuals: the naive E[x^2]-E[x]^2 form loses
        # everything to cancellation when the content is (by construction)
        # identical across bonds, which is exactly the G control.
        norm = torch.linalg.vector_norm(X.double(), dim=1)
        s = torch.zeros(n_graphs, device=X.device, dtype=torch.float64)
        cnt = torch.zeros(n_graphs, device=X.device, dtype=torch.float64)
        s.index_add_(0, bb, norm)
        cnt.index_add_(0, bb, torch.ones_like(norm))
        mean = s / cnt.clamp_min(1.0)
        resid = norm - mean[bb]
        s2 = torch.zeros(n_graphs, device=X.device, dtype=torch.float64)
        s2.index_add_(0, bb, resid * resid)
        std = torch.sqrt(s2 / cnt.clamp_min(1.0))
        out[f"{name}_content_dispersion"] = float((std / mean.clamp_min(1e-12)).mean())
    return out


def moment_conservation(W1: torch.Tensor, V1: torch.Tensor,
                        W2: torch.Tensor, V2: torch.Tensor,
                        batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Per-molecule relative L1 error of the total first/second moments."""
    n_graphs = int(batch["n_graphs"])
    bb = batch["bond_batch"]
    out: dict[str, float] = {}
    for name, A, B in (("W", W1, W2), ("V", V1, V2)):
        a = torch.zeros((n_graphs, A.shape[1]), device=A.device, dtype=torch.float64)
        b = torch.zeros_like(a)
        a.index_add_(0, bb, A.double())
        b.index_add_(0, bb, B.double())
        denom = b.abs().sum(dim=1).clamp_min(1e-12)
        rel = (a - b).abs().sum(dim=1) / denom
        out[f"{name}_max_relative_error"] = float(rel.max())
        out[f"{name}_mean_relative_error"] = float(rel.mean())
    return out


# ---------------------------------------------------------------------------
# FLOPs / op counts
# ---------------------------------------------------------------------------


def gemm_flops(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
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


def attention_op_counts(batch: Mapping[str, torch.Tensor], dim: int = DIM,
                        attn_dim: int = ATTN_DIM) -> dict[str, int]:
    """Analytic pair-level op counts (the GEMM counter cannot see gather/mul ops)."""
    idx = pair_index(batch)
    pairs = 0 if idx is None else int(idx[0].numel())
    return {
        "pairs": pairs,
        "score_macs": pairs * attn_dim,
        "weighted_first_moment_macs": pairs * dim,
        "weighted_second_moment_macs": pairs * dim,
        "marginal_reduction_adds": pairs,
    }


# ---------------------------------------------------------------------------
# toy batch helper
# ---------------------------------------------------------------------------


def build_minimal_batch(
    atom_feat: np.ndarray,
    bond_index: np.ndarray,
    bond_feat: np.ndarray,
    junc_index: np.ndarray,
    junc_bond: np.ndarray,
    junc_feat: np.ndarray,
    global_feat: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Single-molecule batch dict (toy algebra / gradient tests)."""
    n_a = int(atom_feat.shape[0])
    n_b = int(bond_index.shape[0])
    n_j = int(junc_index.shape[0])
    return {
        "n_graphs": 1,
        "atom_feat": torch.as_tensor(atom_feat, dtype=torch.float32),
        "atom_batch": torch.zeros(n_a, dtype=torch.long),
        "bond_feat": torch.as_tensor(bond_feat, dtype=torch.float32),
        "bond_batch": torch.zeros(n_b, dtype=torch.long),
        "bond_index": torch.as_tensor(bond_index, dtype=torch.long),
        "junc_feat": torch.as_tensor(junc_feat, dtype=torch.float32),
        "junc_batch": torch.zeros(n_j, dtype=torch.long),
        "junc_index": torch.as_tensor(junc_index, dtype=torch.long),
        "junc_bond": torch.as_tensor(junc_bond, dtype=torch.long),
        "global_feat": torch.zeros((1, sro.GLOBAL_DIM), dtype=torch.float32)
        if global_feat is None
        else torch.as_tensor(global_feat, dtype=torch.float32).reshape(1, sro.GLOBAL_DIM),
    }
