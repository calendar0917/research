"""The five LCRC arms (B / S / R1 / R2 / D2) plus the D-MPNN reference (G).

Base ``B`` is the frozen ``C1`` (``NO_BOND_MEAN``) computation of
``lcrc_lipo_model_dissection_v1``: ``width=64``, the original
``f: 64 -> 64 -> 1`` and ``rho`` readout, no ``b_mean`` column, no PCBA
attention readout.  ``B`` has 228,674 trainable parameters and is *not* modified
here; every context arm wraps it and adds exactly 34,304 new parameters:

* ``bond_proj``: ``Linear(13, 64)`` (896)
* two independent ``MessageFFN(64, 128)`` blocks (16,704 each)

All four context arms start from the *same* B state and the *same* private
initialisation of the new modules, with both ``fc2`` layers zeroed, so at global
step 0 every arm reproduces B bit-for-bit.

Common forward (identical for every arm)::

    a, b, scal = B._encode(batch)
    h0, _ = B.local(batch, a, b, scal)
    h0 = B.drop(h0)                       # exactly one dropout call
    h1, h2 = the two context layers       # B: no layers, y reads h0
    y = sum_i B.f(h2_i) + B.rho([g, mean(h2), mean(h2^2), log1p(n)])

Layer formula (physical undirected bonds expanded to both directions, no extra
self loops, ``q_ji = bond_proj(bond_feat_ji)``)::

    z_i   = LN_k(h_{k-1,i})
    m_i   = sum_{j in N(i)} SiLU(v_{j->i}^{k} + q_ji)
    h_{k,i} = h_{k-1,i} + fc2_k(SiLU(fc1_k(z_i + m_i)))

with the source state ``v`` fixed per arm:

==========  ==========================================  ==========================
arm         layer 1 ``v``                               layer 2 ``v``
==========  ==========================================  ==========================
``S``       ``LN_1(h0_i)`` (the target itself)          ``LN_2(h1_i)``
``R1``      ``LN_1(h0_j)``                              ``LN_2(h0_j)``
``R2``      ``LN_1(h0_j)``                              ``LN_2(c_j)``
``D2``      ``LN_1(h0_j)``                              ``LN_2(h1_j)``
==========  ==========================================  ==========================

``c_j = (h0_j + sum_{u in N(j)} h0_u) / (1 + deg(j))`` is the fixed 1-hop
extended source (never detached).  A molecule with no bonds has ``m = 0`` and
``c = h0``, so every arm returns a finite prediction for it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lp_common as C

import dissect_model as DM
import sg_model as SGM


# ---------------------------------------------------------------------------
# message block
# ---------------------------------------------------------------------------


class ArmMessageFFN(SGM.MessageFFN):
    """``MessageFFN`` semantics with an explicit *source state* tensor.

    ``sg_model.MessageFFN`` derives the source vector from the same tensor as
    the root (``z = norm(x); msg = index_add(dst, silu(z[src] + bond_msg))``).
    R1 and R2 must feed a *different* tensor for the source (``h0`` / ``c``),
    so this subclass keeps the inherited geometry (LayerNorm-64 -> 64->128 ->
    128->64, zero-residual capable ``fc2``) and takes both tensors explicitly.
    With ``src_state is x`` and ``src`` the physical neighbour index it is
    numerically identical to ``sg_model.MessageFFN`` (preflight check).
    """

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        src_state: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        bond_msg: torch.Tensor,
    ) -> torch.Tensor:
        z = self.norm(x)
        if src.numel():
            v = self.norm(src_state)
            msg = torch.zeros_like(z).index_add_(0, dst, F.silu(v[src] + bond_msg))
            z = z + msg
        return self.fc2(F.silu(self.fc1(z)))

    def mixed_forward(
        self,
        x: torch.Tensor,
        *,
        src: torch.Tensor,
        dst: torch.Tensor,
        bond_msg: torch.Tensor,
        primary_state: torch.Tensor,
        alternative_state: torch.Tensor,
        alternative_dst_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-destination-target source switch (intervention only).

        Edges whose ``dst`` atom is flagged in ``alternative_dst_mask`` take the
        alternative source state; every other edge keeps the primary one.
        """
        z = self.norm(x)
        if src.numel():
            take = alternative_dst_mask[dst]
            msg = torch.zeros_like(z)
            if bool(take.any()):
                v_alt = self.norm(alternative_state)
                msg.index_add_(0, dst[take], F.silu(v_alt[src[take]] + bond_msg[take]))
            keep = ~take
            if bool(keep.any()):
                v_pri = self.norm(primary_state)
                msg.index_add_(0, dst[keep], F.silu(v_pri[src[keep]] + bond_msg[keep]))
            z = z + msg
        return self.fc2(F.silu(self.fc1(z)))


# ---------------------------------------------------------------------------
# arm model
# ---------------------------------------------------------------------------


class LipoContextModel(nn.Module):
    """B (frozen C1) wrapped with the arm's context layers."""

    def __init__(self, base: DM.DissectLCRC, arm: str):
        super().__init__()
        if arm not in C.ARM_SPECS:
            raise KeyError(arm)
        self.arm = arm
        self.spec = dict(C.ARM_SPECS[arm])
        self.kind = self.spec["kind"]
        self.base = base
        if self.kind == "context":
            self.bond_proj = nn.Linear(C.BOND_FEAT_DIM, C.WIDTH)
            self.blocks = nn.ModuleList(
                [ArmMessageFFN(C.WIDTH, C.BLOCK_HIDDEN) for _ in range(C.N_CONTEXT_LAYERS)]
            )

    # -- construction -----------------------------------------------------
    def new_parameters(self) -> list[tuple[str, nn.Parameter]]:
        if self.kind != "context":
            return []
        return [(f"bond_proj.{n}", p) for n, p in self.bond_proj.named_parameters()] + [
            (f"blocks.{i}.{n}", p) for i, block in enumerate(self.blocks) for n, p in block.named_parameters()
        ]

    def init_new_modules(self, generator: torch.Generator | None = None) -> dict[str, Any]:
        """Private-CPU-Generator initialisation of every new module.

        Every new ``Linear`` weight/bias is drawn ``U(-a, a)`` with
        ``a = 1/sqrt(in_features)``; both ``fc2`` (weight and bias) are then set
        to zero and both LayerNorms to ``(1, 0)``.  The drawn values are hashed
        so the round can prove all four arms share them.
        """
        if self.kind != "context":
            return {"drawn": {}, "zeroed": [], "layer_norm": [], "generator_seed": None}
        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(C.PRIVATE_INIT_SEED)
        drawn: dict[str, torch.Tensor] = {}
        order = ["bond_proj"] + [f"blocks.{i}" for i in range(C.N_CONTEXT_LAYERS)]
        modules = {"bond_proj": self.bond_proj}
        modules.update({f"blocks.{i}": block for i, block in enumerate(self.blocks)})
        with torch.no_grad():
            for name in order:
                module = modules[name]
                linears = [("", module)] if isinstance(module, nn.Linear) else []
                if not linears:
                    linears = [("fc1", module.fc1), ("fc2", module.fc2)]
                for prefix, linear in linears:
                    bound = 1.0 / math.sqrt(float(linear.in_features))
                    weight = torch.empty_like(linear.weight).uniform_(-bound, bound, generator=generator)
                    bias = torch.empty_like(linear.bias).uniform_(-bound, bound, generator=generator)
                    linear.weight.copy_(weight)
                    linear.bias.copy_(bias)
                    key = f"{name}.{prefix}" if prefix else name
                    drawn[f"{key}.weight"] = weight.clone()
                    drawn[f"{key}.bias"] = bias.clone()
            zeroed: list[str] = []
            for i, block in enumerate(self.blocks):
                block.fc2.weight.zero_()
                block.fc2.bias.zero_()
                block.norm.weight.fill_(1.0)
                block.norm.bias.zero_()
                zeroed += [f"blocks.{i}.fc2.weight", f"blocks.{i}.fc2.bias"]
        return {
            "drawn": drawn,
            "zeroed": zeroed,
            "layer_norm": [f"blocks.{i}.norm" for i in range(C.N_CONTEXT_LAYERS)],
            "generator_seed": int(C.PRIVATE_INIT_SEED),
            "generator_device": "cpu",
        }

    # -- forward ----------------------------------------------------------
    def _structure(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        idx = batch["bond_index"]
        n_atoms = int(batch["atom_feat"].shape[0])
        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            src = torch.cat([u, v])
            dst = torch.cat([v, u])
            bond_feat = torch.cat([batch["bond_feat"], batch["bond_feat"]], dim=0)
            deg = torch.zeros(n_atoms, dtype=torch.float32, device=idx.device).index_add_(
                0, torch.cat([u, v]), torch.ones(2 * int(u.numel()), dtype=torch.float32, device=idx.device)
            )
        else:
            src = dst = torch.zeros(0, dtype=torch.long, device=batch["atom_feat"].device)
            bond_feat = torch.zeros((0, C.BOND_FEAT_DIM), dtype=torch.float32, device=batch["atom_feat"].device)
            deg = torch.zeros(n_atoms, dtype=torch.float32, device=batch["atom_feat"].device)
        return {"src": src, "dst": dst, "bond_feat": bond_feat, "deg": deg, "n_atoms": n_atoms}

    def _readout(self, batch: Mapping[str, torch.Tensor], h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exactly the frozen C1 readout: ``sum_v f(h_v) + rho(...)``."""
        base = self.base
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        contrib_atom = base.f(h).view(-1)
        contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        g = (batch["global"] - base.global_mean) / base.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = base.rho(rho_in).view(-1)
        return {
            "pred": contrib + rho,
            "contrib": contrib,
            "rho": rho,
            "contrib_atom": contrib_atom,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "count": count,
            "rho_input": rho_in,
            "global_norm": g,
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        source_override: str | None = None,
        layer2_alt_state: str | None = None,
        layer2_alt_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """One forward pass.

        ``source_override`` (intervention only) replaces the layer-2 source for
        *every* edge: ``"reuse"`` (R1 semantics, ``h0``) or ``"fixed"`` (R2
        semantics, ``c``).  ``layer2_alt_state`` + ``layer2_alt_mask`` replace
        the layer-2 source only for the directed edges whose destination atom is
        flagged (``"reuse"`` uses ``h0``).
        """
        base = self.base
        a, b, scal = base._encode(batch)
        h0, _local_aux = base.local(batch, a, b, scal)
        h0 = base.drop(h0)  # exactly one dropout call per forward

        aux: dict[str, torch.Tensor] = {"h0": h0}
        if self.kind == "base":
            out = self._readout(batch, h0)
            aux.update(out)
            aux["h1"] = h0
            aux["h2"] = h0
            return (out["pred"], aux) if return_aux else out["pred"]

        struct = self._structure(batch)
        source = self.spec["source"]
        layers = self.message_layers(h0, struct, source)
        h1, h2 = layers["h1"], layers["h2"]

        if source_override is not None or (layer2_alt_state is not None and layer2_alt_mask is not None):
            h2 = self._override_layer2(
                h0=h0,
                h1=h1,
                struct=struct,
                bond_msg=layers["bond_msg"],
                source=source,
                source_override=source_override,
                layer2_alt_state=layer2_alt_state,
                layer2_alt_mask=layer2_alt_mask,
            )

        out = self._readout(batch, h2)
        if return_aux:
            aux.update(out)
            aux["h1"] = h1
            aux["h2"] = h2
            aux["src"] = struct["src"]
            aux["dst"] = struct["dst"]
            aux["bond_msg"] = layers["bond_msg"]
            aux["deg"] = struct["deg"]
            aux["extended"] = self._extended_source(h0, struct["src"], struct["dst"], struct["deg"])
            return out["pred"], aux
        return out["pred"]

    # -- helpers ----------------------------------------------------------
    def message_layers(
        self, h0: torch.Tensor, struct: Mapping[str, torch.Tensor], source: str
    ) -> dict[str, torch.Tensor]:
        """The two context layers on an explicit ``h0`` (used by preflight + interventions)."""
        if self.kind != "context":
            raise ValueError("the base arm has no context layers")
        src, dst = struct["src"], struct["dst"]
        bond_feat = struct["bond_feat"]
        deg = struct["deg"]
        if bond_feat.shape[0]:
            feats = bond_feat
            if feats.shape[0] != src.numel():           # physical -> directed expansion
                feats = torch.cat([feats, feats], dim=0)
            bond_msg = self.bond_proj(feats)
        else:
            bond_msg = bond_feat.new_zeros((0, C.WIDTH))
        if source == "same_node":
            h1 = h0 + self.blocks[0](h0, src_state=h0, src=dst, dst=dst, bond_msg=bond_msg)
            h2 = h1 + self.blocks[1](h1, src_state=h1, src=dst, dst=dst, bond_msg=bond_msg)
        else:
            h1 = h0 + self.blocks[0](h0, src_state=h0, src=src, dst=dst, bond_msg=bond_msg)
            if source == "repeat_h0":
                h2 = h1 + self.blocks[1](h1, src_state=h0, src=src, dst=dst, bond_msg=bond_msg)
            elif source == "fixed_extended":
                extended = self._extended_source(h0, src, dst, deg)
                h2 = h1 + self.blocks[1](h1, src_state=extended, src=src, dst=dst, bond_msg=bond_msg)
            else:
                h2 = h1 + self.blocks[1](h1, src_state=h1, src=src, dst=dst, bond_msg=bond_msg)
        return {"h1": h1, "h2": h2, "bond_msg": bond_msg}

    @staticmethod
    def _extended_source(
        h0: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, deg: torch.Tensor
    ) -> torch.Tensor:
        """``c_j = (h0_j + sum_{u in N(j)} h0_u) / (1 + deg(j))`` (never detached).

        ``src``/``dst`` are the *directed* expansion (both directions), so the
        undirected neighbour sum is accumulated over unique ``(min, max)`` pairs;
        accumulating over both directions would double count every neighbour.
        """
        n_atoms = int(h0.shape[0])
        nbr_sum = torch.zeros_like(h0)
        if src.numel():
            low = torch.minimum(src, dst)
            high = torch.maximum(src, dst)
            unique = torch.unique(low * n_atoms + high)
            left = torch.div(unique, n_atoms, rounding_mode="floor")
            right = unique - left * n_atoms
            nbr_sum.index_add_(0, left, h0[right])
            nbr_sum.index_add_(0, right, h0[left])
        return (h0 + nbr_sum) / (1.0 + deg).unsqueeze(1)

    def _override_layer2(
        self,
        *,
        h0,
        h1,
        struct,
        bond_msg,
        source,
        source_override,
        layer2_alt_state,
        layer2_alt_mask,
    ):
        src, dst = struct["src"], struct["dst"]
        extended = self._extended_source(h0, src, dst, struct["deg"])
        primary = self._layer2_source_state(source, h0=h0, h1=h1, extended=extended)
        if source_override is not None:
            alternative = {"reuse": h0, "fixed": extended}[source_override]
            return h1 + self.blocks[1](h1, src_state=alternative, src=src, dst=dst, bond_msg=bond_msg)
        alternative = {"reuse": h0, "fixed": extended}[layer2_alt_state]
        return h1 + self.blocks[1].mixed_forward(
            h1,
            src=src,
            dst=dst,
            bond_msg=bond_msg,
            primary_state=primary,
            alternative_state=alternative,
            alternative_dst_mask=layer2_alt_mask,
        )

    def _layer2_source_state(self, source: str, *, h0, h1, extended):
        return {
            "same_node": h1,
            "repeat_h0": h0,
            "fixed_extended": extended,
            "dynamic": h1,
        }[source]


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_arm(spec, norm: Mapping[str, Any], arm: str) -> LipoContextModel:
    """Construct the arm with the *shared* C1 initial state (no trained weights)."""
    if arm not in C.ARM_SPECS:
        raise KeyError(arm)
    if arm == C.DM_ARM:
        raise ValueError("G is the D-MPNN reference, not a C1 arm")
    base = DM.DissectLCRC(spec, norm, "C1")
    model = LipoContextModel(base, arm)
    return model


def shared_c1_init_state(spec, norm: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """One untrained FULL state -> remapped C1 state, shared by every arm.

    ``build_full_state(..., seed=0)`` then ``remap_state('C1', ...)``.  No
    trained checkpoint is loaded, and the private init generator never touches
    this path.
    """
    full_state = DM.build_full_state(spec, norm, int(C.TRAIN_SEED))
    return DM.remap_state("C1", full_state)


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


__all__ = [name for name in dir() if not name.startswith("_")]
