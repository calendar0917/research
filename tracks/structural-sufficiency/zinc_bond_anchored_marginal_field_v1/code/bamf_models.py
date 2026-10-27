"""BAMF-v1 model: DCC-L with an exact global marginal field in the local composer.

The original DCC-L computation is preserved.  The only architectural change is
that the first linear layer of ``local_composer`` receives 12 extra columns from
a four-channel binary latent field whose node/edge potentials are produced from
the frozen P3 atom/bond states.  The G arm computes exact whole-molecule edge
marginals; the I arm computes the same number of features from independent
two-node edge marginals with the *same* potential networks and parameters.

No recursive learned message passing is used.  The exact inference is a
junction-tree sum-product computation over the physical-bond graph; its
messages are algorithmic, not learned hidden states.
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
_DCC = _SUFF / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import dcc_models as DCC  # noqa: E402
from bamf_inference import (  # noqa: E402
    BatchPlan,
    GlobalPlan,
    edge_log_potential,
    exact_marginals,
    independent_marginals,
    marginal_features,
)

WIDTH = 96
BOND_FEAT_DIM = D.BOND_FEAT_DIM
GLOBAL_DIM = D.GLOBAL_DIM
LOCAL_HIDDEN = 80
READER_HIDDEN = 256
CHANNELS = 4
STATES = 2
MARGINAL_DIM = 12
LOCAL_INPUT = WIDTH + 2 * WIDTH + 1 + BOND_FEAT_DIM + MARGINAL_DIM


class BAMFModel(nn.Module):
    """DCC-L plus the bond-anchored marginal field (G or I arm)."""

    def __init__(self, norm: Mapping[str, object], *, arm: str = "G", dropout: float = 0.0):
        super().__init__()
        if arm not in ("G", "I"):
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = str(arm)
        self.encoders = M1.P3Encoders(dropout=dropout)
        self.group_composer = M1.MLP(
            WIDTH + 2 * WIDTH + 1 + BOND_FEAT_DIM, 160, WIDTH, 1, dropout
        )
        self.local_composer = M1.MLP(LOCAL_INPUT, LOCAL_HIDDEN, WIDTH, 1, dropout)
        self.reader = M1.MLP(
            3 * WIDTH + GLOBAL_DIM + 2 * WIDTH,
            READER_HIDDEN,
            1,
            n_hidden=2,
            dropout=dropout,
        )
        self.atom_potential = nn.Linear(WIDTH, CHANNELS * STATES)
        self.bond_potential = nn.Linear(WIDTH, CHANNELS * 3)
        self.register_buffer(
            "global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32)
        )
        self.plan: GlobalPlan | None = None
        # Diagnostic counters filled by forward when requested by the runner.
        self.last_marginal_stats: dict[str, float] = {}

    # -- exact P3 grouping ---------------------------------------------------
    def group_to_bonds(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        n_graphs = int(batch["n_graphs"])
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(
            0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype)
        )
        w_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(
            0, batch["junc_batch"], w
        )
        w2_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(
            0, batch["junc_batch"], w * w
        )
        cnt = torch.bincount(batch["junc_batch"], minlength=n_graphs).clamp(min=1).to(w.dtype)
        mean_w = w_sum / cnt.unsqueeze(1)
        mean_w2 = w2_sum / cnt.unsqueeze(1)
        bond_batch = batch["bond_batch"]
        return n_b.unsqueeze(1) * mean_w[bond_batch], n_b.unsqueeze(1) * mean_w2[bond_batch], n_b

    def local_grouping(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(
            0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype)
        )
        W = torch.zeros(
            (int(batch["bond_feat"].shape[0]), w.shape[1]), device=w.device, dtype=w.dtype
        )
        V = torch.zeros_like(W)
        W.index_add_(0, jb[:, 0], w)
        W.index_add_(0, jb[:, 1], w)
        w2 = w * w
        V.index_add_(0, jb[:, 0], w2)
        V.index_add_(0, jb[:, 1], w2)
        return W, V, n_b

    # -- initialization ------------------------------------------------------
    def load_dcc_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Copy every DCC-L tensor; widen only ``local_composer.lins.0``.

        The 12 new columns are strictly zero.  All original columns and all
        other tensors are copied from the frozen DCC-L initial state.
        """
        own = self.state_dict()
        unexpected = [k for k in state if k not in own]
        if unexpected:
            raise RuntimeError(f"unexpected DCC keys: {unexpected}")
        with torch.no_grad():
            for key, value in state.items():
                if key == "local_composer.lins.0.weight":
                    if value.shape[0] != own[key].shape[0]:
                        raise RuntimeError("local composer first-layer row mismatch")
                    own[key].zero_()
                    own[key][:, : value.shape[1]].copy_(value)
                else:
                    if own[key].shape != value.shape:
                        raise RuntimeError(
                            f"shape mismatch for {key}: {tuple(value.shape)} vs {tuple(own[key].shape)}"
                        )
                    own[key].copy_(value)
        self.load_state_dict(own)

    # -- plan ----------------------------------------------------------------
    def set_plan(self, plan: GlobalPlan | None) -> None:
        self.plan = plan
        if plan is not None:
            plan.to(next(self.parameters()).device)

    # -- forward -------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        c_glob = self.group_composer(
            torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        Wl, Vl, _ = self.local_grouping(w, batch)
        # four binary channels, shared potential networks across atoms/bonds
        log_phi = torch.tanh(self.atom_potential(a)).reshape(-1, CHANNELS, STATES)
        log_psi_raw = torch.tanh(self.bond_potential(b)).reshape(-1, CHANNELS, 3)
        log_psi = edge_log_potential(log_psi_raw)
        if self.arm == "G":
            plan = batch.get("bamf_plan")
            if plan is None:
                raise RuntimeError("G arm requires batch['bamf_plan']")
            _, edge_marg = exact_marginals(log_phi, log_psi, plan)
        else:
            _, edge_marg = independent_marginals(log_phi, log_psi, batch["bond_index"])
        m_e = marginal_features(edge_marg)
        c_loc = self.local_composer(
            torch.cat(
                [b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"], m_e], dim=1
            )
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


def _named_parameter_table(model: nn.Module) -> dict[str, int]:
    return {name: int(p.numel()) for name, p in model.named_parameters()}


def build_initial_state(norm: Mapping[str, object], seed: int, arm: str):
    """Build the shared BAMF initial state for one supervised seed.

    The DCC-L base is built by the historical builder (same P3/SSL/random
    skeleton and the same ``seed + 90731`` local-composer draw).  The two
    potential heads are drawn after that with a fixed, seed-dependent stream;
    G and I therefore share every tensor bit-for-bit.
    """
    shared_dcc, dcc_info = DCC.build_initial_state(norm, seed)

    # fixed stream for the new potential heads; it is consumed after the DCC
    # skeleton has been built, so the DCC tensors are unaffected.
    torch.manual_seed(int(seed) + 424242)
    model = BAMFModel(norm, arm=arm)
    model.load_dcc_state(C.cpu_state(shared_dcc))

    info = {
        "arm": str(arm),
        "seed": int(seed),
        "ssl": dcc_info["ssl"],
        "dcc_initial_state_sha256": dcc_info["initial_state_sha256"],
        "dcc_parameter_count": int(dcc_info["parameters"]),
        "parameters": _param_count(model),
        "parameters_by_tensor": _named_parameter_table(model),
        "parameters_breakdown": {
            "dcc_total": int(dcc_info["parameters"]),
            "atom_potential": _param_count(model.atom_potential),
            "bond_potential": _param_count(model.bond_potential),
            "local_composer_extra": MARGINAL_DIM * LOCAL_HIDDEN,
        },
        "initial_state_sha256": C.state_hash(model),
        "marginal_dim": MARGINAL_DIM,
        "channels": CHANNELS,
        "states": STATES,
    }
    return model, info


def build_model(candidate: str, norm: Mapping[str, object], seed: int, device, *,
                arm: str, plan: GlobalPlan | None = None):
    model, info = build_initial_state(norm, seed, arm)
    if plan is not None:
        model.set_plan(plan)
    model.to(device)
    info = dict(info)
    info.update(
        {
            "candidate": candidate,
            "parameters": _param_count(model),
            "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
            "parameters_total": sum(1 for p in model.parameters()),
        }
    )
    return model, info


def clone_for_arm(model: BAMFModel, arm: str) -> BAMFModel:
    out = copy.deepcopy(model)
    out.arm = str(arm)
    return out
