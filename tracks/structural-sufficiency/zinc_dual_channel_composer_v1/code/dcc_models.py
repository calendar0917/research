"""P3 dual-channel local/global composer for the ZINC generalization follow-up.

The base path is the exact P3 Null computation graph.  A parallel local path
feeds a second composer with either:

L arm:  Wl_b = sum_{j touches b} w_j,      Vl_b = sum_{j touches b} w_j^2
C arm:  Wl_b = n_b * graph_mean(w),        Vl_b = n_b * graph_mean(w^2)

Both arms share every parameter shape and the same initial state.  The reader
is widened to [sum c_glob, sum a, sum a^2, global33, sum c_loc, sum c_loc^2]
with zero new columns, so the initial output equals P3.  There is no recursive
local message passing: the local path is one static incidence aggregation and
one shared MLP per bond.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_SUFF = _HERE.parents[1]
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_V2 = _SUFF / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ad2_models as A2  # noqa: E402

WIDTH = 96
BOND_FEAT_DIM = D.BOND_FEAT_DIM
GLOBAL_DIM = D.GLOBAL_DIM
LOCAL_HIDDEN = 80
READER_HIDDEN = 256


class DualChannel(nn.Module):
    """P3 Null base plus one parallel local composer and a widened reader."""

    def __init__(self, norm: Mapping[str, object], *, dropout: float = 0.0):
        super().__init__()
        self.encoders = M1.P3Encoders(dropout=dropout)
        self.group_composer = M1.MLP(
            WIDTH + 2 * WIDTH + 1 + BOND_FEAT_DIM, 160, WIDTH, 1, dropout
        )
        self.local_composer = M1.MLP(
            WIDTH + 2 * WIDTH + 1 + BOND_FEAT_DIM, LOCAL_HIDDEN, WIDTH, 1, dropout
        )
        # 96 (base bonds) + 2*96 (atom sum/moment) + 33 (global) +
        # 2*96 (local bond sum/moment) = 513.
        self.reader = M1.MLP(
            3 * WIDTH + GLOBAL_DIM + 2 * WIDTH,
            READER_HIDDEN,
            1,
            n_hidden=2,
            dropout=dropout,
        )
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        # Arm switch: True = organized local ownership, False = matched Null control.
        self.organized_local = True

    # -- exact P3 Null grouping ------------------------------------------------
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

    # -- L/C local grouping ----------------------------------------------------
    def local_grouping(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor], *, organized: bool):
        if not organized:
            return self.group_to_bonds(w, batch)
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

    # -- initialization --------------------------------------------------------
    def load_p3_state(self, p3_state: Mapping[str, torch.Tensor]) -> None:
        """Load all P3 weights; widen the first reader layer with zero columns."""
        base = {k: v for k, v in p3_state.items() if not k.startswith("reader.")}
        missing, unexpected = self.load_state_dict(base, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected P3 keys: {unexpected}")
        if any(not k.startswith(("reader.", "local_composer.")) for k in missing):
            raise RuntimeError(f"unexpected missing P3 keys: {missing}")
        old = self.reader.lins
        with torch.no_grad():
            old[0].weight.zero_()
            old[0].weight[:, : p3_state["reader.lins.0.weight"].shape[1]].copy_(
                p3_state["reader.lins.0.weight"]
            )
            old[0].bias.copy_(p3_state["reader.lins.0.bias"])
            for i in range(1, len(old)):
                old[i].weight.copy_(p3_state[f"reader.lins.{i}.weight"])
                old[i].bias.copy_(p3_state[f"reader.lins.{i}.bias"])

    # -- forward ---------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        c_glob = self.group_composer(
            torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        Wl, Vl, _ = self.local_grouping(w, batch, organized=bool(self.organized_local))
        c_loc = self.local_composer(
            torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        B = int(batch["n_graphs"])
        z = c_glob.new_zeros((B, c_glob.shape[1])).index_add_(0, batch["bond_batch"], c_glob)
        zl = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc)
        zl2 = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc * c_loc)
        a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        return self.reader(torch.cat([z, a_sum, a_sq, g, zl, zl2], dim=1)).view(-1)


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def build_initial_state(norm: Mapping[str, object], seed: int) -> tuple[DualChannel, dict]:
    """Build the exact P3 seed-0 reference, then the shared dual initial state.

    The P3 reference is built first with ``torch.manual_seed(seed)`` exactly as
    ``ad2_models.build_model("P3N", ...)`` does.  Only after that is the local
    branch initialized, with a fixed second seed.  The returned model is the
    single shared state from which both L and C are deep-copied; no arm-specific
    constructor consumes RNG.
    """
    torch.manual_seed(int(seed))
    p3 = A2.P3NullModel(norm)
    ssl_path = C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
    ssl_info = A2._load_encoders(p3, ssl_path, torch.device("cpu"))
    p3_state = C.cpu_state(p3)

    torch.manual_seed(int(seed) + 90731)
    dual = DualChannel(norm)
    dual.load_p3_state(p3_state)
    info = {
        "arm_neutral_builder": True,
        "seed": int(seed),
        "ssl": ssl_info,
        "p3_reference_state_sha256": C.state_hash(p3),
        "initial_state_sha256": C.state_hash(dual),
        "parameters": _param_count(dual),
        "parameters_breakdown": {
            "p3_base_total": _param_count(p3),
            "encoders": _param_count(dual.encoders),
            "group_composer": _param_count(dual.group_composer),
            "local_composer": _param_count(dual.local_composer),
            "reader": _param_count(dual.reader),
        },
    }
    return dual, info


def build_model(candidate: str, norm: Mapping[str, object], seed: int, device,
                *, organized_local: bool):
    """Builder compatible with ``ad_train.train``."""
    shared, base_info = build_initial_state(norm, seed)
    model = copy.deepcopy(shared)
    model.organized_local = bool(organized_local)
    model.to(device)
    info = dict(base_info)
    info.update(
        {
            "candidate": candidate,
            "arm": "L" if organized_local else "C",
            "organized_local": bool(organized_local),
            "parameters": _param_count(model),
            "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
            "parameters_total": sum(1 for p in model.parameters()),
        }
    )
    return model, info
