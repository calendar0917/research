"""SCE and GEF heads on the frozen LCRC-A trunk.

Base forward (copied verbatim from ``lc_models.LCRC.forward`` so the trunk
output is bit-identical to historical LCRC-A):

    y_base = sum_v f(h_v) + rho(global33, mean h, mean h^2, log(1+|V|))

SCE adds   sum_bridge I_e / sqrt(1+n_bridge)
GEF adds   sum_v [g(h_v,c_v) - g(h_v,0)]
with exactly one field solve per forward and no iterative state update.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

import nl_common as N


class SCEHead(nn.Module):
    """Separator-conditioned bridge interaction (endpoint-exchange invariant)."""

    def __init__(self, width: int, bond_dim: int, side: int = 32, hidden: int = 64,
                 mode: str = "real"):
        super().__init__()
        if mode not in ("real", "marginal"):
            raise ValueError(mode)
        self.mode = mode
        self.side = side
        self.proj = nn.Linear(width, side)
        in_dim = 3 * side + 2 * width + bond_dim + 2
        self.mlp = N.M.MLP([in_dim, hidden, 1])
        nn.init.normal_(self.mlp.lins[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.mlp.lins[-1].bias)

    def forward(self, h, b, batch, bridge, n_graphs: int):
        n_atoms = h.shape[0]
        phi = self.proj(h)
        e = bridge["n_bridges"]
        m_a = h.new_zeros((0, self.side))
        m_b = h.new_zeros((0, self.side))
        if e:
            m_a = h.new_zeros((e, self.side)).index_add_(
                0, bridge["side_a_seg"],
                phi[bridge["side_a_atom"]] * bridge["side_a_w"][bridge["side_a_seg"]].unsqueeze(1)
            )
            m_b = h.new_zeros((e, self.side)).index_add_(
                0, bridge["side_b_seg"],
                phi[bridge["side_b_atom"]] * bridge["side_b_w"][bridge["side_b_seg"]].unsqueeze(1)
            )
            if self.mode == "marginal":
                # Molecule-level content mean field, same per-side normalisation
                # and same parameter structure.
                count = h.new_zeros(n_graphs).index_add_(
                    0, batch["atom_batch"], torch.ones_like(batch["atom_batch"], dtype=h.dtype)
                )
                mu = h.new_zeros((n_graphs, self.side)).index_add_(
                    0, batch["atom_batch"], phi[:, :]
                ) / count.clamp(min=1.0).unsqueeze(1)
                mu_e = mu[bridge["bridge_graph"]]
                m_a = mu_e
                m_b = mu_e
        bond = batch["bond_index"][bridge["bridge_bond"]] if e else torch.zeros((0, 2), dtype=torch.long)
        ha = h[bond[:, 0]] if e else h.new_zeros((0, h.shape[1]))
        hb = h[bond[:, 1]] if e else h.new_zeros((0, h.shape[1]))
        be = b[bridge["bridge_bond"]] if e else h.new_zeros((0, b.shape[1]))
        feat = torch.cat(
            [
                m_a + m_b,
                (m_a - m_b).abs(),
                m_a * m_b,
                ha + hb,
                (ha - hb).abs(),
                be,
                # symmetric side-size features: strict endpoint-exchange
                # invariance forbids an ordered (size_A, size_B) slot pair.
                torch.log1p(bridge["side_a_size"]).unsqueeze(1)
                + torch.log1p(bridge["side_b_size"]).unsqueeze(1),
                (torch.log1p(bridge["side_a_size"])
                 - torch.log1p(bridge["side_b_size"])).abs().unsqueeze(1),
            ],
            dim=1,
        )
        i_e = self.mlp(feat).view(-1)
        total = h.new_zeros(n_graphs).index_add_(0, bridge["bridge_graph"], i_e)
        n_b = bridge["n_bridges_per_graph"].to(h.dtype)
        term = total / torch.sqrt(1.0 + n_b)
        return term, {"I_e": i_e, "bridge_term": term, "m_a": m_a, "m_b": m_b}


class GEFHead(nn.Module):
    """One exact graph-equilibrium field, read atom-wise with subtraction."""

    def __init__(self, width: int, bond_dim: int, rank: int = 12, hidden: int = 64,
                 lam: float = 0.5, mode: str = "real", pad_max: int = 64):
        super().__init__()
        if mode not in ("real", "fixed", "local"):
            raise ValueError(mode)
        self.mode = mode
        self.rank = rank
        self.lam = float(lam)
        self.pad_max = int(pad_max)
        self.z_proj = nn.Linear(width, rank)
        self.w_mlp = N.M.MLP([2 * width + bond_dim, 32, 1])
        self.g = N.M.MLP([width + rank, hidden, 1])
        # g(h,c)-g(h,0) is invariant to an output bias, so the final bias would
        # be an exactly-dead parameter; remove it rather than carry a zero grad.
        self.g.lins[-1].bias = None
        nn.init.normal_(self.g.lins[-1].weight, std=1.0e-3)

    def conduction(self, h, b, batch):
        idx = batch["bond_index"]
        hu, hv = h[idx[:, 0]], h[idx[:, 1]]
        if self.mode == "fixed":
            return hu.new_ones(idx.shape[0])
        raw = self.w_mlp(torch.cat([hu + hv, (hu - hv).abs(), b], dim=1)).view(-1)
        return torch.nn.functional.softplus(raw) + 1.0e-3

    def field(self, h, z, b, batch, n_graphs: int):
        device = h.device
        n_atoms = h.shape[0]
        counts = batch["atom_counts"].to(device)
        n_pad = int(self.pad_max)
        atom_batch = batch["atom_batch"]
        atom_off = batch["atom_off"]
        local = torch.arange(n_atoms, device=device) - atom_off[atom_batch]
        valid = torch.zeros((n_graphs, n_pad), dtype=torch.bool, device=device)
        valid[atom_batch, local] = True

        w = self.conduction(h, b, batch)
        idx = batch["bond_index"]
        bb = batch["bond_batch"]
        off = atom_off[bb]
        lu = idx[:, 0] - off
        lv = idx[:, 1] - off
        # symmetric dense weighted adjacency (out-of-place scatter_add)
        base = torch.zeros((n_graphs, n_pad, n_pad), dtype=h.dtype, device=device)
        ii = (bb * n_pad + lu) * n_pad + lv
        jj = (bb * n_pad + lv) * n_pad + lu
        flat = base.view(-1)
        flat = flat.scatter_add(0, ii, w)
        flat = flat.scatter_add(0, jj, w)
        W = flat.view(n_graphs, n_pad, n_pad)
        deg = W.sum(-1)
        eye = torch.eye(n_pad, dtype=h.dtype, device=device).unsqueeze(0)
        A = self.lam * eye + torch.diag_embed(deg) - W
        # padded rows become well-conditioned and are masked out later
        pad_mask = (~valid).unsqueeze(-1) & (~valid).unsqueeze(-2)
        A = A + pad_mask.to(h.dtype) * eye
        K = torch.linalg.solve(A, eye)

        z_pad = h.new_zeros((n_graphs, n_pad, self.rank)).index_put(
            (atom_batch, local), z
        )
        off = 1.0 - eye
        M = K * off * valid.unsqueeze(-1).to(h.dtype) * valid.unsqueeze(-2).to(h.dtype)
        denom = M.sum(-1)
        zsum = z_pad.sum(1)                       # (B, rank)
        n = counts.to(h.dtype).unsqueeze(1)
        num = torch.bmm(M, z_pad)                 # (B, N, rank)
        z_self = z_pad
        other_mean = (zsum.unsqueeze(1) - z_self) / (n - 1.0).clamp(min=1.0).unsqueeze(-1)
        c_pad = num / denom.clamp(min=1.0e-8).unsqueeze(-1) - other_mean
        single = (counts == 1).unsqueeze(1)
        c_pad = torch.where(single.unsqueeze(-1), torch.zeros_like(c_pad), c_pad)
        c_pad = c_pad * valid.unsqueeze(-1).to(h.dtype)
        c = c_pad[atom_batch, local]
        return c, {"K": K, "w": w, "M": M, "valid": valid, "c": c}

    def forward(self, h, b, batch, n_graphs: int):
        z = self.z_proj(h)
        if self.mode == "local":
            # Same reader, same rank; local one-hop mean field instead of the
            # global equilibrium.  Not the c=0 identity.
            e = batch["bond_index"].shape[0]
            zbar = z.new_zeros((h.shape[0], self.rank)).index_add_(
                0, batch["bond_index"][:, 0], z[batch["bond_index"][:, 1]]
            ).index_add_(0, batch["bond_index"][:, 1], z[batch["bond_index"][:, 0]])
            deg = h.new_zeros(h.shape[0]).index_add_(
                0, batch["bond_index"][:, 0], torch.ones(e, dtype=h.dtype, device=h.device)
            ).index_add_(
                0, batch["bond_index"][:, 1], torch.ones(e, dtype=h.dtype, device=h.device)
            )
            local_mean = zbar / deg.clamp(min=1.0).unsqueeze(1)
            glob = h.new_zeros((n_graphs, self.rank)).index_add_(0, batch["atom_batch"], z)
            cnt = batch["atom_counts"].to(h.dtype).unsqueeze(1)
            c = local_mean - (glob / cnt)[batch["atom_batch"]]
            aux_extra = {"c": c, "w": torch.ones(e, dtype=h.dtype, device=h.device)}
        else:
            c, aux_extra = self.field(h, z, b, batch, n_graphs)
        delta = self.g(torch.cat([h, c], dim=1)).view(-1) - self.g(
            torch.cat([h, torch.zeros_like(c)], dim=1)
        ).view(-1)
        term = h.new_zeros(n_graphs).index_add_(0, batch["atom_batch"], delta)
        return term, {"delta_atom": delta, **aux_extra}


class NLModel(nn.Module):
    """LCRC-A trunk (bit-identical init) plus one non-local head."""

    def __init__(self, head: str, norm: Mapping[str, Any], cfg: Mapping[str, Any] | None = None,
                 *, mode: str = "real", lam: float = 0.5):
        super().__init__()
        self.head_kind = head
        self.mode = mode
        self.trunk = N.M.LCRC("A", norm, cfg)
        self.cfg = self.trunk.cfg
        width = int(self.cfg["width"])
        bond_dim = int(self.cfg["bond_out"])
        self.sce: SCEHead | None = None
        self.gef: GEFHead | None = None
        if head == "sce":
            self.sce = SCEHead(width, bond_dim, mode=mode)
        elif head == "gef":
            self.gef = GEFHead(width, bond_dim, lam=lam, mode=mode)
        elif head != "base":
            raise ValueError(head)

    # -- base readout, copied from LCRC.forward -----------------------------
    def _base(self, batch, a, b, h):
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / count.clamp(min=1.0).unsqueeze(1)
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / count.clamp(min=1.0).unsqueeze(1)
        contrib_atom = self.trunk.f(h).view(-1)
        contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        g = (batch["global"] - self.trunk.global_mean) / self.trunk.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.trunk.rho(rho_in).view(-1)
        return contrib + rho, contrib, rho

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        a, b, h = self.trunk._encode(batch)
        y, contrib, rho = self._base(batch, a, b, h)
        aux = {"contrib": contrib, "rho": rho, "h": h, "baseline_y": y}
        if self.sce is not None:
            term, h_aux = self.sce(h, b, batch, batch["bridge"], n_graphs)
        elif self.gef is not None:
            term, h_aux = self.gef(h, b, batch, n_graphs)
        else:
            term, h_aux = torch.zeros(n_graphs, device=y.device, dtype=y.dtype), {}
        y = y + term
        aux["term"] = term
        aux.update(h_aux)
        return (y, aux) if return_aux else y


def _params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _state_hash(state) -> str:
    import hashlib
    h_ = hashlib.sha256()
    for key, value in sorted(state.items()):
        h_.update(key.encode())
        h_.update(value.detach().cpu().numpy().astype("float32").tobytes())
    return h_.hexdigest()


def build_model(head: str, norm, seed: int, device, cfg=None, *, mode="real", lam=0.5,
                ssl_path=None):
    """Shared trunk first (seed, SSL), then head under seed+90731.

    The trunk construction is the same call sequence as ``lc_models.build_model('A')``
    so its state is bit-identical to historical LCRC-A seed 0.
    """
    torch.manual_seed(int(seed))
    c = dict(N.M.DEFAULT_CONFIG if cfg is None else {**N.M.DEFAULT_CONFIG, **dict(cfg)})
    model = NLModel(head, norm, c, mode=mode, lam=lam)
    ssl_source = N.M.ssl_path(int(seed)) if ssl_path is None else ssl_path
    ssl = N.M.load_ssl_encoders(model.trunk, ssl_source)
    torch.manual_seed(int(seed) + 90731)
    # rebuild head parameters under the derived seed: modules were constructed
    # before the manual_seed call, so re-initialise them deterministically.
    if model.sce is not None:
        for mod in model.sce.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, a=5 ** 0.5)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.normal_(model.sce.mlp.lins[-1].weight, std=1.0e-3)
        nn.init.zeros_(model.sce.mlp.lins[-1].bias)
    if model.gef is not None:
        for mod in model.gef.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, a=5 ** 0.5)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.normal_(model.gef.g.lins[-1].weight, std=1.0e-3)
    model.to(device)
    trunk_keys = {k: v.detach().clone() for k, v in model.trunk.state_dict().items()}
    info = {
        "head": head,
        "mode": mode,
        "lambda": lam,
        "seed": int(seed),
        "config": model.cfg,
        "ssl": ssl,
        "parameters": _params(model),
        "parameters_trunk": _params(model.trunk),
        "parameters_head": 0 if (model.sce is None and model.gef is None)
        else _params(model.sce or model.gef),
        "trunk_state_sha256": _state_hash(trunk_keys),
        "initial_state_sha256": N.L.C.state_hash(model),
        "has_iterative_message_passing": False,
    }
    if info["parameters"] > 500_000:
        raise RuntimeError(f"{head}/{mode}: {info['parameters']} params exceeds cap")
    return model, info


def load_model(head: str, norm, state, device, cfg=None, *, mode="real", lam=0.5):
    model = NLModel(head, norm, cfg, mode=mode, lam=lam)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.to(device)
    return model


__all__ = ["SCEHead", "GEFHead", "NLModel", "build_model", "load_model"]
