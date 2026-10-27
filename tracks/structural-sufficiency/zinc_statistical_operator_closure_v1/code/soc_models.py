"""SOC-v1 — Statistical Operator Closure on the bond/junction incidence graph.

The base backbone is the frozen DCC-L ``DualChannel`` (P3 encoders -> global /
local composers -> 513-dim reader).  The only new computation is a *weight-tied*
moment-closure iteration over the explicit bond<->junction incidence structure:

    (a, b, w)  ->  T(a, b, w)  ->  T(...)  ->  (a, b, w)^K

where each object state is updated from

    [self, mean(group1), mean(group1^2), mean(group2), mean(group2^2),
     log1p count1, log1p count2, role_embedding]

and the *same* projection + core MLP is used for every object role and every
iteration.  ``group1`` / ``group2`` are the explicit incidence groups:

    atom      : (incident bonds, empty)
    bond      : (two endpoint atoms, incident junctions)
    junction  : (three patch atoms, two incident bonds)

The update is a residual with a per-role sigmoid gate initialised at ~0.0067, so
at initialisation the model is numerically DCC-L.  No attention, no global token
mixing, no learned pairwise relation: the state transition is a moment statistic
of the incidence graph, applied repeatedly with the same parameters.

Two readouts are supported and pre-registered separately:

    final : use (a^K, b^K, w^K) exactly where DCC-L uses (a, b, w)
    mean  : use the time mean over t = 0..K (a trajectory statistic)

Both are identical to DCC-L at gate = 0?  No: ``mean`` uses the arithmetic mean
of K+1 identical states, which is the same vector, so yes, both equal DCC-L at
gate = 0 to floating-point round-off.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SUFF = _HERE.parents[1]
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_DCC = _SUFF / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import dcc_models as DM  # noqa: E402

WIDTH = DM.WIDTH
LOCAL_HIDDEN = DM.LOCAL_HIDDEN
READER_HIDDEN = DM.READER_HIDDEN
DEFAULT_K = 3
DEFAULT_RANK = 24
DEFAULT_HIDDEN = 48
DEFAULT_GATE_BIAS = -5.0


def _role_index(n: int, role: int, device) -> torch.Tensor:
    return torch.full((n,), int(role), dtype=torch.long, device=device)


def _stats(src: torch.Tensor, idx: torch.Tensor, n_out: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First / second moments and count of ``src`` scattered to ``idx``.

    ``src`` must already be gathered along the incidence (one row per message),
    so the caller decides whether an object has one or several incident groups.
    """
    d = src.shape[1]
    s = src.new_zeros((n_out, d)).index_add_(0, idx, src)
    q = src.new_zeros((n_out, d)).index_add_(0, idx, src * src)
    cnt = src.new_zeros((n_out,)).index_add_(0, idx, torch.ones(idx.numel(), device=src.device, dtype=src.dtype))
    inv = 1.0 / cnt.clamp(min=1.0)
    return s * inv.unsqueeze(1), q * inv.unsqueeze(1), cnt


class MomentClosure(nn.Module):
    """One shared statistical operator, called K times.

    The module owns no per-iteration parameters.  ``role`` distinguishes the
    three object types (the same weights are used for all of them); the
    projection sees the object's own state plus the two incidence groups.
    """

    def __init__(self, *, width: int = WIDTH, rank: int = DEFAULT_RANK,
                 hidden: int = DEFAULT_HIDDEN, K: int = DEFAULT_K,
                 gate_bias: float = DEFAULT_GATE_BIAS, dropout: float = 0.0):
        super().__init__()
        self.width = int(width)
        self.rank = int(rank)
        self.K = int(K)
        in_dim = 5 * self.width + 2 + 8
        self.role = nn.Parameter(torch.randn(3, 8) * 0.02)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, self.rank)
        self.core = nn.Sequential(
            nn.Linear(self.rank, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.width),
        )
        self.dropout = float(dropout)
        self.gate = nn.Parameter(torch.full((3,), float(gate_bias)))

    def _update(self, h: torch.Tensor, g1: tuple, g2: tuple, role: int) -> torch.Tensor:
        m1, s1, c1 = g1
        m2, s2, c2 = g2
        b = h.shape[0]
        role_tok = self.role[role].view(1, 8).expand(b, 8)
        x = torch.cat(
            [h, m1, s1, m2, s2, torch.log1p(c1).unsqueeze(1), torch.log1p(c2).unsqueeze(1), role_tok],
            dim=1,
        )
        u = self.core(self.proj(self.norm(x)))
        if self.dropout > 0.0:
            u = F.dropout(u, p=self.dropout, training=self.training)
        g = torch.sigmoid(self.gate[role])
        return h + g * u

    def forward(self, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor,
                batch: Mapping[str, torch.Tensor], *, return_trajectory: bool = False):
        """Return the K-step closure (and optionally every intermediate state)."""
        bi = batch["bond_index"]
        jb = batch["junc_bond"]
        ji = batch["junc_index"]
        na, nb, nw = a.shape[0], b.shape[0], w.shape[0]

        # incidence message sources (gathered once per step)
        atom_idx = bi.reshape(-1)                      # 2*nb
        atom_src_b = b[atom_idx]
        bond_g1_idx = torch.arange(nb, device=a.device).repeat_interleave(2)
        bond_g1_src = a[bi.reshape(-1)]
        bond_g2_idx = jb.reshape(-1)                   # 2*nw, values are bond ids
        bond_g2_src = w.repeat_interleave(2, dim=0)
        junc_g1_idx = torch.arange(nw, device=a.device).repeat_interleave(3)
        junc_g1_src = a[ji.reshape(-1)]
        junc_g2_idx = torch.arange(nw, device=a.device).repeat_interleave(2)
        junc_g2_src = b[jb.reshape(-1)]

        zero_a = a.new_zeros((na, self.width))
        zero_b = b.new_zeros((nb, self.width))
        zero_w = w.new_zeros((nw, self.width))
        zero_ca = a.new_zeros((na,))
        zero_cb = b.new_zeros((nb,))
        zero_cw = w.new_zeros((nw,))

        traj = [(a, b, w)] if return_trajectory else None
        for _ in range(self.K):
            # Jacobi step: all statistics use the same pre-step states.
            a_g1 = _stats(atom_src_b, atom_idx, na)
            b_g1 = _stats(bond_g1_src, bond_g1_idx, nb)
            b_g2 = _stats(bond_g2_src, bond_g2_idx, nb)
            w_g1 = _stats(junc_g1_src, junc_g1_idx, nw)
            w_g2 = _stats(junc_g2_src, junc_g2_idx, nw)
            a_new = self._update(a, a_g1, (zero_a, zero_a, zero_ca), 0)
            b_new = self._update(b, b_g1, b_g2, 1)
            w_new = self._update(w, w_g1, w_g2, 2)
            a, b, w = a_new, b_new, w_new
            if return_trajectory:
                traj.append((a, b, w))
        if return_trajectory:
            return (a, b, w), traj
        return a, b, w


class SOCModel(DM.DualChannel):
    """DCC-L backbone + weight-tied moment closure before the composers."""

    def __init__(self, norm: Mapping[str, object], *, K: int = DEFAULT_K,
                 readout: str = "final", rank: int = DEFAULT_RANK,
                 hidden: int = DEFAULT_HIDDEN, dropout: float = 0.0):
        super().__init__(norm, dropout=dropout)
        if readout not in ("final", "mean"):
            raise ValueError(f"unknown readout {readout!r}")
        self.readout = str(readout)
        self.closure = MomentClosure(width=WIDTH, rank=rank, hidden=hidden,
                                     K=K, dropout=dropout)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b, w = self.encoders(batch)
        traj = None
        if self.readout == "mean":
            (a, b, w), traj = self.closure(a, b, w, batch, return_trajectory=True)
            a = torch.stack([t[0] for t in traj], 0).mean(0)
            b = torch.stack([t[1] for t in traj], 0).mean(0)
            w = torch.stack([t[2] for t in traj], 0).mean(0)
        else:
            a, b, w = self.closure(a, b, w, batch)

        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        c_glob = self.group_composer(
            torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        Wl, Vl, _ = self.local_grouping(w, batch, organized=True)
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


def build_initial_state(norm: Mapping[str, object], seed: int, *, K: int = DEFAULT_K,
                        readout: str = "final", rank: int = DEFAULT_RANK,
                        hidden: int = DEFAULT_HIDDEN):
    """Build the exact DCC-L seed state, then a deterministic closure on top.

    The shared (backbone) tensors are loaded bit-for-bit from the frozen DCC-L
    initial state.  Only the closure tensors depend on ``seed + 55117``.  At
    gate bias ``-5`` the initial prediction differs from DCC-L by <1%.
    """
    shared, base_info = DM.build_initial_state(norm, seed)
    torch.manual_seed(int(seed) + 55117)
    model = SOCModel(norm, K=K, readout=readout, rank=rank, hidden=hidden)
    missing, unexpected = model.load_state_dict(shared.state_dict(), strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected shared keys: {unexpected}")
    if any(not k.startswith("closure.") for k in missing):
        raise RuntimeError(f"unexpected missing keys: {missing}")
    info = dict(base_info)
    info.update(
        {
            "architecture": "SOC",
            "K": int(K),
            "readout": str(readout),
            "rank": int(rank),
            "hidden": int(hidden),
            "parameters": _param_count(model),
            "parameters_closure": _param_count(model.closure),
            "gate_init": float(torch.sigmoid(model.closure.gate).detach().mean()),
        }
    )
    return model, info


def build_model(candidate: str, norm: Mapping[str, object], seed: int, device,
                *, K: int = DEFAULT_K, readout: str = "final",
                rank: int = DEFAULT_RANK, hidden: int = DEFAULT_HIDDEN):
    model, info = build_initial_state(norm, seed, K=K, readout=readout,
                                      rank=rank, hidden=hidden)
    model.to(device)
    info["candidate"] = candidate
    return model, info


def build_control_state(norm: Mapping[str, object], seed: int, device, *,
                        reader_hidden: int = 274):
    """Parameter-matched DCC-L control with a wider reader and no closure.

    The extra reader units are dead at initialisation (their outgoing columns in
    the last layer are zero), so the control starts numerically at DCC-L and can
    only diverge if the extra capacity itself buys something.  ``reader_hidden=274``
    matches SOC-v1's 512,888 parameters to +0.03%.
    """
    shared, info = DM.build_initial_state(norm, seed)
    torch.manual_seed(int(seed) + 77123)
    model = DM.DualChannel(norm)
    # replace the reader with the wider one (same three-layer MLP shape)
    model.reader = DM.M1.MLP(3 * WIDTH + DM.GLOBAL_DIM + 2 * WIDTH,
                             int(reader_hidden), 1, n_hidden=2)
    base = shared.state_dict()
    own = model.state_dict()
    for k, v in base.items():
        if not k.startswith("reader."):
            own[k].copy_(v)
    with torch.no_grad():
        old = shared.reader.lins
        new = model.reader.lins
        new[0].weight.zero_()
        new[0].weight[: old[0].weight.shape[0]].copy_(old[0].weight)
        new[0].bias.zero_()
        new[0].bias[: old[0].bias.shape[0]].copy_(old[0].bias)
        new[1].weight.zero_()
        new[1].weight[: old[1].weight.shape[0], : old[1].weight.shape[1]].copy_(old[1].weight)
        new[1].bias.zero_()
        new[1].bias[: old[1].bias.shape[0]].copy_(old[1].bias)
        new[2].weight.zero_()
        new[2].weight[:, : old[2].weight.shape[1]].copy_(old[2].weight)
        new[2].bias.copy_(old[2].bias)
    model.to(device)
    info = dict(info)
    info.update({"architecture": "CTRL", "reader_hidden": int(reader_hidden),
                 "parameters": _param_count(model)})
    return model, info
