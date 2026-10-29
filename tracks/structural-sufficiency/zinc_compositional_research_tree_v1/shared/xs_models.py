"""The fixed shared skeleton (task §1) plus tiny helpers.

    a_v = atom_encoder(raw108)                      108 -> 128 -> 96  (SSL init)
    b_e = bond_encoder([a_u+a_v, |a_u-a_v|, a_u a_v, bond7])   295 -> 160 -> 96
    g_base = [sum a, sum a^2, sum b, sum b^2, global33, log1p(n_atoms), log1p(n_bonds)]
    rho_base: 419 -> 128 -> 64 -> 1                                 (SiLU, no dropout)
    rho_event: 225 -> 128 -> 1                                      (output zero-init)

``rho_event`` is shared by both families; only its 225-dim input layout differs
(CMA: ``[b_e, m_e, m_e^2, log1p n_e]``, SPC: ``[a_v, t_v, t_v^2, log1p n_pairs]``).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import xs_common as C

WIDTH = 96
N_ATOM_FEAT = C.ATOM_FEAT_DIM       # 108
BOND_FEAT = C.BOND_FEAT_DIM         # 7
GLOBAL_DIM = C.GLOBAL_DIM           # 33

BOND_IN = 3 * WIDTH + BOND_FEAT     # 295
BASE_IN = 4 * WIDTH + GLOBAL_DIM + 2  # 419
EVENT_IN = 225


class MLP(nn.Module):
    """Explicit-width SiLU MLP with ``lins.<i>.{weight,bias}`` naming.

    The naming matches the frozen SSL checkpoints, so the atom/bond encoders can
    be loaded verbatim.
    """

    def __init__(self, dims: Sequence[int], dropout: float = 0.0):
        super().__init__()
        dims = [int(d) for d in dims]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                   for i in range(len(dims) - 1)])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
                if self.dropout > 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SharedSkeleton(nn.Module):
    """Atom/bond encoders + the two readers fixed by the task."""

    def __init__(self, norm: Mapping[str, object], dropout: float = 0.0):
        super().__init__()
        self.atom_encoder = MLP([N_ATOM_FEAT, 128, WIDTH], dropout)
        self.bond_encoder = MLP([BOND_IN, 160, WIDTH], dropout)
        self.rho_base = MLP([BASE_IN, 128, 64, 1], dropout)
        self.rho_event = MLP([EVENT_IN, 128, 1], dropout)
        # zero-init event output layer (task §1)
        nn.init.zeros_(self.rho_event.lins[-1].weight)
        nn.init.zeros_(self.rho_event.lins[-1].bias)
        self.register_buffer("global_mean",
                             torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale",
                             torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    # -- encoders ------------------------------------------------------------
    def encode(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        b = self.bond_encoder(torch.cat([au + av, (au - av).abs(), au * av,
                                         batch["bond_feat"]], dim=1))
        return a, b

    # -- global readout ------------------------------------------------------
    def base_readout(self, batch: Mapping[str, torch.Tensor], a, b) -> torch.Tensor:
        B = int(batch["n_graphs"])
        ab, bb = batch["atom_batch"], batch["bond_batch"]
        za = a.new_zeros((B, WIDTH)).index_add_(0, ab, a)
        za2 = a.new_zeros((B, WIDTH)).index_add_(0, ab, a * a)
        zb = b.new_zeros((B, WIDTH)).index_add_(0, bb, b)
        zb2 = b.new_zeros((B, WIDTH)).index_add_(0, bb, b * b)
        g = torch.cat([za, za2, zb, zb2, batch["global_feat"],
                       torch.log1p(batch["n_atoms"].to(a.dtype)).view(B, 1),
                       torch.log1p(batch["n_bonds"].to(a.dtype)).view(B, 1)], dim=1)
        if g.shape[1] != BASE_IN:
            raise RuntimeError(f"g_base dim {g.shape[1]} != {BASE_IN}")
        return self.rho_base(g).view(-1)

    # -- event readout -------------------------------------------------------
    def event_readout(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != EVENT_IN:
            raise RuntimeError(f"event input dim {x.shape[1]} != {EVENT_IN}")
        return self.rho_event(x).view(-1)

    @staticmethod
    def scatter_events(values: torch.Tensor, batch_index: torch.Tensor, n_graphs: int) -> torch.Tensor:
        out = values.new_zeros((int(n_graphs),))
        if values.numel() > 0:
            out.index_add_(0, batch_index, values)
        return out


def load_ssl(skeleton: SharedSkeleton, path, device="cpu") -> dict:
    """Load the frozen train-only SSL atom/bond encoders (read-only)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    loaded = []
    for name in ("atom_encoder", "bond_encoder"):
        getattr(skeleton, name).load_state_dict({k: v.to(device) for k, v in state[name].items()})
        loaded.append(name)
    return {
        "source": str(path),
        "sha256": C.sha256_file(path),
        "keys_loaded": loaded,
        "junction_encoder_loaded": False,
    }


def parameter_breakdown(model) -> dict:
    out = {}
    for name, module in model.named_children():
        out[name] = int(sum(p.numel() for p in module.parameters()))
        if isinstance(module, SharedSkeleton):
            for sub, m in module.named_children():
                out[f"{name}.{sub}"] = int(sum(p.numel() for p in m.parameters()))
    out["total"] = C.parameter_count(model)
    return out


def build_seeded(factory, seed: int):
    """Deterministic construction so REAL/NULL (and the shared parts of CMA/SPC)
    start from bit-identical tensors for the same seed."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = factory()
    return model
