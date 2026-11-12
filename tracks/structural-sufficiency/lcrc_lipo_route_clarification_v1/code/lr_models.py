"""The six frozen architectures of the round.

``B``  the frozen C1 local-content model (``NO_BOND_MEAN``, 228,674 parameters),
       re-implemented bit-for-bit so the residual-block variants can be built on
       top of exactly the same sub-state;
``S``  static capacity: two residual blocks with ``c_t = h_t`` -- the new
       modules never read a neighbour state;
``O``  fixed context: ``c_0 = T h0``, ``c_1 = T(T h0)`` -- a two-step mean walk
       of the *initial* node state only;
``D``  dynamic composition: ``c_0 = T h0``, ``c_1 = T h1`` -- the second block
       reads the updated state;
``F``  Morgan-2048 (+ g42) MLP, 207,009 parameters;
``G``  GINE-style graph reference (explicit implementation, no library
       defaults), 251,781 parameters.

``T`` is the row-normalised mean over the true undirected bond graph with **no
self-loop**; isolated atoms (and disconnected components) receive the zero
vector and never mix across molecules.

Block::

    r_t   = Wout_t( SiLU( Win_t( concat( LN_root_t(h_t), LN_context_t(c_t) ) ) ) )
    h_t+1 = h_t + r_t

``Win`` is 128->128 and ``Wout`` is 128->64, both with bias; the two layer norms
have affine parameters; ``Wout`` weight/bias start at exactly zero, everything
else uses the fixed-seed default initialisation.  No new dropout inside a block.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import lr_common as C

import dissect_common as DC
import dissect_model as M
from feature_spec import FeatureSpec


# ---------------------------------------------------------------------------
# neighbour-mean operator
# ---------------------------------------------------------------------------


def neighbour_mean_index(atom_batch: torch.Tensor, bond_index: torch.Tensor, n_atoms: int):
    """Return ``(dst, src, inv_deg)`` for ``(T x)_i = mean_{j in N(i)} x_j``."""
    if bond_index.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=atom_batch.device)
        return empty, empty, torch.zeros(n_atoms, dtype=torch.float32, device=atom_batch.device)
    u, v = bond_index[:, 0], bond_index[:, 1]
    dst = torch.cat([u, v])
    src = torch.cat([v, u])
    deg = torch.zeros(n_atoms, dtype=torch.float32, device=atom_batch.device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    inv_deg = torch.where(deg > 0, 1.0 / deg.clamp(min=1.0), torch.zeros_like(deg))
    return dst, src, inv_deg


def neighbour_mean(x: torch.Tensor, dst: torch.Tensor, src: torch.Tensor, inv_deg: torch.Tensor) -> torch.Tensor:
    if dst.numel() == 0:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    return out * inv_deg.to(x.dtype).unsqueeze(1)


# ---------------------------------------------------------------------------
# residual block
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, width: int = 64, hidden: int = 128):
        super().__init__()
        self.ln_root = nn.LayerNorm(width)
        self.ln_ctx = nn.LayerNorm(width)
        self.win = nn.Linear(2 * width, hidden, bias=True)
        self.wout = nn.Linear(hidden, width, bias=True)
        nn.init.zeros_(self.wout.weight)
        nn.init.zeros_(self.wout.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.ln_root(h), self.ln_ctx(c)], dim=1)
        return self.wout(F.silu(self.win(x)))


# ---------------------------------------------------------------------------
# B / S / O / D
# ---------------------------------------------------------------------------
def apply_blocks(kind: str, blocks, h0: torch.Tensor, dst, src, inv_deg) -> tuple[torch.Tensor, list[float]]:
    """Run the two frozen residual blocks with the branch-specific context."""
    h_t = h0
    norms: list[float] = []
    for t, block in enumerate(blocks):
        if kind == "S":
            ctx = h_t
        elif kind == "O":
            ctx = h0
            for _ in range(t + 1):
                ctx = neighbour_mean(ctx, dst, src, inv_deg)
        elif kind == "D":
            src_state = h0 if t == 0 else h_t
            ctx = neighbour_mean(src_state, dst, src, inv_deg)
        else:
            raise RuntimeError(kind)
        r = block(h_t, ctx)
        h_t = h_t + r
        norms.append(float(r.detach().norm()))
    return h_t, norms




class RouteModel(nn.Module):
    """C1 base plus the frozen residual blocks (``kind`` in B/S/O/D)."""

    VARIANTS = ("B", "S", "O", "D")

    def __init__(self, spec: FeatureSpec, norm: Mapping[str, Any], kind: str,
                 cfg: Mapping[str, Any] | None = None):
        super().__init__()
        if kind not in self.VARIANTS:
            raise KeyError(kind)
        self.kind = kind
        self.spec = spec
        self.cfg = dict(M.DEFAULT_CONFIG if cfg is None else {**M.DEFAULT_CONFIG, **dict(cfg)})
        width = int(self.cfg["width"])
        a_out = int(self.cfg["atom_out"])
        # creation order identical to ``DissectLCRC('C1')`` so the same
        # ``torch.manual_seed`` reproduces the frozen C1 initial state
        self.atom_encoder = M.MLP([spec.atom_feat_dim, int(self.cfg["atom_hidden"]), a_out])
        self.bond_encoder = M.MLP([3 * a_out + spec.bond_feat_dim, int(self.cfg["bond_hidden"]),
                                   int(self.cfg["bond_out"])])
        self.local = M.LocalContent(spec, "C1", self.cfg)
        self.drop = nn.Dropout(float(C.RECIPE["dropout"]))
        self.f = M.MLP([width, width, 1])
        self.rho = M.MLP([DC.RHO_DIM, int(self.cfg["rho_hidden"]), 1])
        self.blocks = nn.ModuleList()
        if kind != "B":
            self.blocks = nn.ModuleList([ResidualBlock(width, 128) for _ in range(2)])
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))
        if kind != "B" and len(self.blocks) != 2:
            raise RuntimeError("the frozen design has exactly two residual blocks")

    # -- shared encoding -------------------------------------------------
    def encode_local(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        else:
            bond_input = torch.zeros((0, 3 * a.shape[1] + batch["bond_feat"].shape[1]),
                                     dtype=a.dtype, device=a.device)
        b = self.bond_encoder(bond_input)
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        return a, b, scal

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        a, b, scal = self.encode_local(batch)
        h, local_aux = self.local(batch, a, b, scal)
        h = self.drop(h)  # the frozen single dropout call
        h0 = h
        block_norms: list[float] = []
        if self.blocks:
            dst, src, inv_deg = neighbour_mean_index(atom_batch, batch["bond_index"], h.shape[0])
            h, block_norms = apply_blocks(self.kind, self.blocks, h0, dst, src, inv_deg)

        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib_atom = self.f(h).view(-1)
        contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in).view(-1)
        y = contrib + rho
        if return_aux:
            return y, {
                "h": h, "h0": h0, "contrib": contrib, "rho": rho, "mean_h": mean_h,
                "mean_h2": mean_h2, "count": count, "local_input": local_aux["local_input"],
                "block_norm": block_norms,
            }
        return y


def build_route_model(spec, norm, kind: str, seed: int, device=None, cfg=None) -> RouteModel:
    """Construct ``kind`` with the frozen C1 initial state for the shared part."""
    full_state = M.build_full_state(spec, norm, int(seed))
    init = M.remap_state("C1", full_state)
    torch.manual_seed(int(seed))
    model = RouteModel(spec, norm, kind, cfg)
    state = model.state_dict()
    merged = {k: (init[k].clone() if k in init else state[k].clone()) for k in state}
    model.load_state_dict(merged, strict=True)
    if device is not None:
        model.to(device)
    return model


def shared_init_hash(model: RouteModel) -> dict[str, Any]:
    c1_keys = [k for k in model.state_dict() if not k.startswith("blocks.")]
    return {"n_shared_keys": len(c1_keys), "n_block_keys": len(model.state_dict()) - len(c1_keys)}


# ---------------------------------------------------------------------------
# F: Morgan-2048 + g42 MLP
# ---------------------------------------------------------------------------


class FingerprintMLP(nn.Module):
    def __init__(self, spec: FeatureSpec, norm: Mapping[str, Any], fp_mean, fp_scale,
                 n_bits: int = 2048):
        super().__init__()
        self.n_bits = int(n_bits)
        self.register_buffer("fp_mean", torch.as_tensor(fp_mean, dtype=torch.float32))
        self.register_buffer("fp_scale", torch.as_tensor(fp_scale, dtype=torch.float32))
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        d = self.n_bits + int(spec.global_dim)
        self.net = nn.Sequential(
            nn.Linear(d, 96), nn.SiLU(),
            nn.Linear(96, 64), nn.SiLU(),
            nn.Dropout(float(C.RECIPE["dropout"])),
            nn.Linear(64, 1),
        )

    def feature(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        fp = (batch["fp"] - self.fp_mean) / self.fp_scale
        g = (batch["global"] - self.global_mean) / self.global_scale
        return torch.cat([fp, g], dim=1)

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        x = self.feature(batch)
        y = self.net(x).view(-1)
        if return_aux:
            return y, {"h": None, "rho": None, "contrib": y}
        return y


def build_fingerprint_model(spec, norm, fp_mean, fp_scale, seed: int, device=None) -> FingerprintMLP:
    torch.manual_seed(int(seed))
    model = FingerprintMLP(spec, norm, fp_mean, fp_scale)
    if device is not None:
        model.to(device)
    return model


# ---------------------------------------------------------------------------
# G: GINE-style reference
# ---------------------------------------------------------------------------


class GINELayer(nn.Module):
    def __init__(self, hidden: int = 128, bond_dim: int = 13):
        super().__init__()
        self.edge = nn.Linear(bond_dim, hidden, bias=True)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, u: torch.Tensor, dst, src, bond_feat: torch.Tensor) -> torch.Tensor:
        if dst.numel():
            msg = torch.zeros_like(u)
            msg.index_add_(0, dst, F.relu(u[src] + self.edge(bond_feat)))
        else:
            msg = torch.zeros_like(u)
        s = (1.0 + self.eps) * u + msg
        return self.norm(u + self.mlp(s))


class GINEReference(nn.Module):
    def __init__(self, spec: FeatureSpec, norm: Mapping[str, Any], n_layers: int = 3, hidden: int = 128):
        super().__init__()
        self.spec = spec
        self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        d_in = int(spec.atom_feat_dim) + int(spec.n_scalars)
        self.input = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([GINELayer(hidden, int(spec.bond_feat_dim)) for _ in range(n_layers)])
        self.to_h = nn.Linear(hidden, 64)
        self.drop = nn.Dropout(float(C.RECIPE["dropout"]))
        self.f = M.MLP([64, 64, 1])
        self.rho = M.MLP([DC.RHO_DIM, 64, 1])

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        u = self.input(torch.cat([batch["atom_feat"], scal], dim=1))
        idx = batch["bond_index"]
        if idx.numel():
            uu, vv = idx[:, 0], idx[:, 1]
            dst = torch.cat([uu, vv])
            src = torch.cat([vv, uu])
            bf = torch.cat([batch["bond_feat"], batch["bond_feat"]], dim=0)
        else:
            dst = src = torch.zeros(0, dtype=torch.long, device=u.device)
            bf = torch.zeros((0, int(self.spec.bond_feat_dim)), dtype=u.dtype, device=u.device)
        for layer in self.layers:
            u = layer(u, dst, src, bf)
        h = self.drop(self.to_h(u))
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, self.f(h).view(-1))
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in).view(-1)
        y = contrib + rho
        if return_aux:
            return y, {"h": h, "rho": rho, "contrib": contrib}
        return y


def build_gine_model(spec, norm, seed: int, device=None) -> GINEReference:
    """GINE reference with the C1 readout (``f``/``rho``) initialised from the C1 state."""
    full_state = M.build_full_state(spec, norm, int(seed))
    init = M.remap_state("C1", full_state)
    torch.manual_seed(int(seed))
    model = GINEReference(spec, norm)
    state = model.state_dict()
    merged = dict(state)
    for key in ("f.lins.0.weight", "f.lins.0.bias", "f.lins.1.weight", "f.lins.1.bias",
                "rho.lins.0.weight", "rho.lins.0.bias", "rho.lins.1.weight", "rho.lins.1.bias"):
        merged[key] = init[key].clone()
    for key in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        merged[key] = init[key].clone()
    model.load_state_dict(merged, strict=True)
    if device is not None:
        model.to(device)
    return model


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def build_model(kind: str, spec, norm, seed: int, device=None, extra: Mapping[str, Any] | None = None):
    if kind in RouteModel.VARIANTS:
        model = build_route_model(spec, norm, kind, seed, device)
    elif kind == "F":
        extra = dict(extra or {})
        model = build_fingerprint_model(spec, norm, extra["fp_mean"], extra["fp_scale"], seed, device)
    elif kind == "G":
        model = build_gine_model(spec, norm, seed, device)
    else:
        raise KeyError(kind)
    got = parameter_count(model)
    want = C.EXPECTED_PARAMS[kind]
    if got != want:
        raise RuntimeError(f"{kind}: {got} trainable parameters != frozen {want}")
    return model


__all__ = [n for n in dir() if not n.startswith("_")]
