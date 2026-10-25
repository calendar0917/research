"""P3-Chromatic Bond Transformer v0.

One physical bond = one token.  Five pre-LN global interaction layers over all
bond pairs of the same molecule (key padding is the only mask).  The only C/S
difference is whether the 48 relation-channel logits are used per output
feature channel (C) or averaged within each head before the key softmax (S).

The local carrier (P3 encoders + DCC-L global/local composers) and its initial
state are copied from ``dcc_models.build_initial_state``.  No supervised DCC-L
checkpoint is loaded.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SS = _EXP.parent
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_V2 = _SS / "zinc_recipe_trunk_v2" / "code"
_DCC = _SS / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M  # noqa: E402
import dcc_models as DCC  # noqa: E402

WIDTH = 48
HEADS = 4
HEAD_DIM = WIDTH // HEADS
DEPTH = 5
FFN_HIDDEN = 96
REL_DIM = 22
KSTEPS = 21
BOND_FEAT_DIM = D.BOND_FEAT_DIM
GLOBAL_DIM = D.GLOBAL_DIM
LOCAL_HIDDEN = 80
GROUP_HIDDEN = 160
TOKEN_IN = 3 * 96 + 2
READER_IN = 48 + 48 + 96 + 96 + GLOBAL_DIM + 2
RES_SCALE = 1.0 / math.sqrt(5.0)


class ChromaticLayer(nn.Module):
    """One pre-LN full-graph chromatic/scalar interaction block."""

    def __init__(self, *, d: int = WIDTH, heads: int = HEADS, rel_dim: int = REL_DIM,
                 ffn_hidden: int = FFN_HIDDEN):
        super().__init__()
        if d % heads:
            raise ValueError("width must be divisible by heads")
        self.d = int(d)
        self.heads = int(heads)
        self.head_dim = self.d // self.heads
        self.ln1 = nn.LayerNorm(self.d)
        self.q = nn.Linear(self.d, self.d, bias=False)
        self.k = nn.Linear(self.d, self.d, bias=False)
        self.v = nn.Linear(self.d, self.d, bias=False)
        self.rel_head = nn.Linear(rel_dim, self.heads, bias=False)
        self.rel_channel = nn.Linear(rel_dim, self.d, bias=False)
        self.rel_value = nn.Linear(rel_dim, self.d, bias=False)
        self.out = nn.Linear(2 * self.d, self.d, bias=True)
        self.ln2 = nn.LayerNorm(self.d)
        self.ff = nn.Sequential(
            nn.Linear(self.d, ffn_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(ffn_hidden, self.d, bias=True),
        )

    def forward(self, h: torch.Tensor, rel: torch.Tensor, mask: torch.Tensor,
                scale: torch.Tensor, arm: str, collect: bool = False):
        b, e, _ = h.shape
        hh, hd = self.heads, self.head_dim
        valid = mask
        x = self.ln1(h) * valid.unsqueeze(-1).to(h.dtype)
        q = self.q(x).view(b, e, hh, hd)
        k = self.k(x).view(b, e, hh, hd)
        v = self.v(x).view(b, e, hh, hd)
        qk = torch.einsum("bihd,bjhd->bijh", q, k) / math.sqrt(float(hd))
        rh = self.rel_head(rel).view(b, e, e, hh)
        rc_raw = self.rel_channel(rel).view(b, e, e, hh, hd)
        if arm == "C":
            rc = rc_raw
        elif arm == "S":
            rc = rc_raw.mean(dim=-1, keepdim=True).expand_as(rc_raw)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        logits = qk.unsqueeze(-1) + rh.unsqueeze(-1) + rc
        logits = logits.masked_fill(~valid[:, None, :, None, None], -1.0e9)
        attn = torch.softmax(logits, dim=2)          # softmax over keys f
        rv = self.rel_value(rel).view(b, e, e, hh, hd)
        msg = torch.einsum("bijhc,bjhc->bihc", attn, v)
        msg = msg + torch.einsum("bijhc,bijhc->bihc", attn, rv)
        msg = msg.reshape(b, e, self.d)
        sc = scale.unsqueeze(-1)
        res = self.out(torch.cat([msg, msg * sc], dim=-1))
        h = h + RES_SCALE * res
        h = h * valid.unsqueeze(-1).to(h.dtype)
        h = h + RES_SCALE * self.ff(self.ln2(h))
        h = h * valid.unsqueeze(-1).to(h.dtype)

        stats = None
        if collect:
            with torch.no_grad():
                a = attn.detach()
                n_keys = valid.sum(dim=1).to(a.dtype).clamp_min(1.0)
                denom = valid.sum().to(a.dtype).clamp_min(1.0) * hh * hd
                ent = -(a.clamp_min(1.0e-12).log() * a).sum(dim=2)       # [B,E,H,hd]
                mean_entropy = float((ent * valid[:, :, None, None]).sum() / denom)
                row_err = (a.sum(dim=2) - 1.0).abs()
                row_sum_max_error = float((row_err * valid[:, :, None, None]).max()) if bool(valid.any()) else 0.0
                finite = bool(torch.isfinite(a).all() and torch.isfinite(rc_raw).all()
                              and torch.isfinite(qk).all())
                maxw = a.max(dim=2).values
                mean_max = float((maxw * valid[:, :, None, None]).sum() / denom)
                am = a.mean(dim=-1, keepdim=True)
                tv = 0.5 * (a - am).abs().sum(dim=2)                     # [B,E,H,hd]
                mean_tv = float((tv * valid[:, :, None, None]).sum() / denom)
                pair_mask = valid[:, :, None] & valid[:, None, :]
                pair_count = pair_mask.sum().to(a.dtype).clamp_min(1.0) * hh
                rc_std_raw = rc_raw.detach().std(dim=-1)                 # [B,E,E,H]
                rc_std_eff = rc.detach().std(dim=-1)
                mean_rc_raw = float((rc_std_raw * pair_mask[:, :, :, None]).sum() / pair_count)
                mean_rc_eff = float((rc_std_eff * pair_mask[:, :, :, None]).sum() / pair_count)
                adj = rel[..., 0] >= 0.25
                eye = torch.eye(e, device=h.device, dtype=torch.bool).unsqueeze(0)
                nonadj = (~adj) & (~eye) & pair_mask
                na_mass = (a * nonadj[:, :, :, None, None]).sum(dim=2)   # [B,E,H,hd]
                mean_na = float((na_mass * valid[:, :, None, None]).sum() / denom)
                stats = {
                    "mean_entropy": mean_entropy,
                    "row_sum_max_error": row_sum_max_error,
                    "finite": finite,
                    "mean_log_keys": float(torch.log(n_keys).mean()),
                    "mean_max_weight": mean_max,
                    "channel_dev_tv": mean_tv,
                    "rc_logit_std_raw": mean_rc_raw,
                    "rc_logit_std_effective": mean_rc_eff,
                    "nonadjacent_attention_mass": mean_na,
                    "mean_scale": float(scale[valid].mean()) if bool(valid.any()) else 0.0,
                    "n_pairs": int(pair_mask.sum()),
                }
        return h, stats


class ChromaticBondTransformer(nn.Module):
    """P3 local carrier + full-graph chromatic bond interaction + new reader."""

    def __init__(self, norm: Mapping[str, Any], *, train_mean_log1p_line_degree: float,
                 depth: int = DEPTH, width: int = WIDTH, heads: int = HEADS,
                 ffn_hidden: int = FFN_HIDDEN, rel_dim: int = REL_DIM,
                 dropout: float = 0.0):
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.arm = "C"
        self.encoders = M.P3Encoders(dropout=float(dropout))
        self.group_composer = M.MLP(
            96 + 2 * 96 + 1 + BOND_FEAT_DIM, GROUP_HIDDEN, 96, 1, float(dropout)
        )
        self.local_composer = M.MLP(
            96 + 2 * 96 + 1 + BOND_FEAT_DIM, LOCAL_HIDDEN, 96, 1, float(dropout)
        )
        self.token_proj = nn.Linear(TOKEN_IN, self.width, bias=True)
        self.token_norm = nn.LayerNorm(self.width)
        self.layers = nn.ModuleList(
            [ChromaticLayer(d=self.width, heads=heads, rel_dim=rel_dim,
                            ffn_hidden=ffn_hidden) for _ in range(self.depth)]
        )
        self.reader = nn.Sequential(
            nn.Linear(READER_IN, 128, bias=True),
            nn.SiLU(),
            nn.Linear(128, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, 1, bias=True),
        )
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("line_degree_scale",
                             torch.tensor(float(train_mean_log1p_line_degree), dtype=torch.float32))

    # -- exact DCC-L local carrier -----------------------------------------
    def group_to_bonds(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        n_graphs = int(batch["n_graphs"])
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        w_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w)
        w2_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w * w)
        cnt = torch.bincount(batch["junc_batch"], minlength=n_graphs).clamp(min=1).to(w.dtype)
        mean_w = w_sum / cnt.unsqueeze(1)
        mean_w2 = w2_sum / cnt.unsqueeze(1)
        bond_batch = batch["bond_batch"]
        return n_b.unsqueeze(1) * mean_w[bond_batch], n_b.unsqueeze(1) * mean_w2[bond_batch], n_b

    def local_grouping(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        W = torch.zeros((int(batch["bond_feat"].shape[0]), w.shape[1]), device=w.device, dtype=w.dtype)
        V = torch.zeros_like(W)
        W.index_add_(0, jb[:, 0], w)
        W.index_add_(0, jb[:, 1], w)
        w2 = w * w
        V.index_add_(0, jb[:, 0], w2)
        V.index_add_(0, jb[:, 1], w2)
        return W, V, n_b

    def local_parts(self, batch: Mapping[str, torch.Tensor]):
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        c_glob = self.group_composer(
            torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        Wl, Vl, _ = self.local_grouping(w, batch)
        c_loc = self.local_composer(
            torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        return a, b, w, c_glob, c_loc, n_b

    # -- initialisation -----------------------------------------------------
    def load_dcc_parts(self, d0_state: Mapping[str, torch.Tensor]) -> None:
        prefixes = ("encoders.", "group_composer.", "local_composer.")
        sub = {k: v for k, v in d0_state.items() if k.startswith(prefixes)}
        own = {k: v for k, v in self.state_dict().items() if k.startswith(prefixes)}
        if set(sub) != set(own):
            raise RuntimeError(f"DCC part key mismatch: missing={sorted(set(own)-set(sub))} "
                               f"unexpected={sorted(set(sub)-set(own))}")
        missing, unexpected = self.load_state_dict(sub, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected DCC keys: {unexpected}")
        if any(k.startswith(prefixes) for k in missing):
            raise RuntimeError(f"unexpected missing DCC keys: "
                               f"{[k for k in missing if k.startswith(prefixes)]}")
        with torch.no_grad():
            self.global_mean.copy_(torch.as_tensor(d0_state["global_mean"], dtype=torch.float32))
            self.global_scale.copy_(torch.as_tensor(d0_state["global_scale"], dtype=torch.float32))

    def zero_rel_channel(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                layer.rel_channel.weight.zero_()

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, collect: bool = False):
        a, b, w, c_glob, c_loc, n_b = self.local_parts(batch)
        bcount = int(batch["n_graphs"])
        bb = int(batch["n_bonds_pad"])
        bond_batch = batch["bond_batch"]
        n_bonds_per_bond = batch["n_bonds"].to(b.dtype)[bond_batch]
        x = torch.cat(
            [b, c_glob, c_loc, torch.log1p(n_b).unsqueeze(1),
             torch.log1p(n_bonds_per_bond).unsqueeze(1)],
            dim=1,
        )
        h0 = self.token_norm(self.token_proj(x))
        h = h0.new_zeros((bcount, bb, self.width))
        scale = torch.log1p(n_b) / self.line_degree_scale.clamp_min(1.0e-6)
        scale_dense = h.new_zeros((bcount, bb))
        if bb:
            slot = bond_batch.to(torch.long) * bb + batch["bond_local"].to(torch.long)
            h_flat = h.reshape(bcount * bb, self.width).index_copy(0, slot, h0)
            h = h_flat.view(bcount, bb, self.width)
            scale_flat = scale_dense.reshape(bcount * bb).index_copy(0, slot, scale)
            scale_dense = scale_flat.view(bcount, bb)

        layer_stats = []
        if bb:
            rel = batch["rel"]
            mask = batch["rel_mask"]
            for layer in self.layers:
                h, st = layer(h, rel, mask, scale_dense, self.arm, collect=collect)
                if collect:
                    st["rel_channel_weight_rms"] = float(
                        layer.rel_channel.weight.detach().pow(2).mean().sqrt()
                    )
                    layer_stats.append(st)
        else:
            h = h * 1.0

        h_sum = h.new_zeros((bcount, self.width)).index_add_(0, bond_batch, h[bond_batch, batch["bond_local"]]) if bb else h.new_zeros((bcount, self.width))
        h_sq = h.new_zeros((bcount, self.width)).index_add_(0, bond_batch, (h[bond_batch, batch["bond_local"]]) ** 2) if bb else h.new_zeros((bcount, self.width))
        a_sum = a.new_zeros((bcount, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((bcount, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        gstd = (batch["global"] - self.global_mean) / self.global_scale
        z = torch.cat(
            [h_sum, h_sq, a_sum, a_sq, gstd,
             torch.log1p(batch["n_bonds"]).unsqueeze(1).to(a.dtype),
             torch.log1p(batch["n_atoms"]).unsqueeze(1).to(a.dtype)],
            dim=1,
        )
        y = self.reader(z).view(-1)
        if collect:
            return y, {"layers": layer_stats}
        return y


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


_SHARED: dict[tuple[int, float], tuple[ChromaticBondTransformer, dict[str, Any]]] = {}


def _shared_initial_state(norm: Mapping[str, Any], seed: int,
                          train_mean_log1p_line_degree: float
                          ) -> tuple[ChromaticBondTransformer, dict[str, Any]]:
    key = (int(seed), float(train_mean_log1p_line_degree))
    if key in _SHARED:
        return _SHARED[key]
    # D0: the exact P3 + DCC-L seed-matched shared initial state.  This loads the
    # train-only masked-SSL encoder and never a supervised DCC-L checkpoint.
    dual, base_info = DCC.build_initial_state(norm, int(seed))
    d0_state = C.cpu_state(dual)
    torch.manual_seed(int(seed) + 172_941)
    model = ChromaticBondTransformer(
        norm, train_mean_log1p_line_degree=float(train_mean_log1p_line_degree)
    )
    model.load_dcc_parts(d0_state)
    model.zero_rel_channel()
    info = {
        "seed": int(seed),
        "architecture": "p3_chromatic_bond_v0",
        "arm_neutral_builder": True,
        "ssl": base_info["ssl"],
        "p3_reference_state_sha256": base_info["p3_reference_state_sha256"],
        "dcc_d0_state_sha256": base_info["initial_state_sha256"],
        "initial_state_sha256": C.state_hash(model),
        "parameters": _param_count(model),
        "parameters_breakdown": {
            "encoders": _param_count(model.encoders),
            "group_composer": _param_count(model.group_composer),
            "local_composer": _param_count(model.local_composer),
            "token_fusion": _param_count(model.token_proj) + _param_count(model.token_norm),
            "layers": _param_count(model.layers),
            "reader": _param_count(model.reader),
        },
        "line_degree_scale": float(train_mean_log1p_line_degree),
        "depth": int(model.depth),
        "width": int(model.width),
        "heads": int(model.layers[0].heads) if len(model.layers) else 0,
        "head_dim": int(model.layers[0].head_dim) if len(model.layers) else 0,
        "rel_dim": int(REL_DIM),
        "dropout": 0.0,
        "batchnorm": False,
        "official_test_loaded": False,
    }
    _SHARED[key] = (model, info)
    return model, info


def build_initial_state(norm: Mapping[str, Any], seed: int, *,
                        train_mean_log1p_line_degree: float
                        ) -> tuple[ChromaticBondTransformer, dict[str, Any]]:
    model, info = _shared_initial_state(norm, seed, train_mean_log1p_line_degree)
    return copy.deepcopy(model), dict(info)


def build_model(arm: str, norm: Mapping[str, Any], seed: int, device, *,
                train_mean_log1p_line_degree: float):
    if arm not in ("C", "S"):
        raise ValueError(f"unknown arm {arm!r}")
    torch.manual_seed(int(seed) + 172_941)
    torch.cuda.manual_seed_all(int(seed) + 172_941)
    model, info = _shared_initial_state(norm, int(seed), float(train_mean_log1p_line_degree))
    model = copy.deepcopy(model)
    model.arm = arm
    model.to(device)
    info = dict(info)
    info.update({
        "candidate": f"CHR_{arm}",
        "arm": arm,
        "parameters": _param_count(model),
        "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
        "parameters_total": sum(1 for p in model.parameters()),
    })
    return model, info
