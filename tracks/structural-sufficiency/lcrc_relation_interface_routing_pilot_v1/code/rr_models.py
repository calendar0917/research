"""R1 / R2: block-token + interface-token readouts on the frozen C1 local encoder.

Shared structure
----------------
Trunk (bitwise the frozen C1 ``NO_BOND_MEAN`` local encoder, loaded from the one
canonical C1 initial state of the readout-context round):

    a_v = AtomEncoder(atom_feat_v)
    b_e = BondEncoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bond_feat_e])
    ... the frozen typed S2 embeddings / path_gate / LocalMLP / LayerNorm ...

Block-conditioned local input (width 433, the frozen C1 column layout):

    x_(v,i) = [ a_v,
                sum_{e incident v, owner(e)=i} b_e,
                sum_{S1 occurrence of v with owner=i} nbr_emb(code),
                sum_{S2 internal to i rooted at v} [type,end,joint,closure],
                sum_{S2 internal to i rooted at v} a_end * path_gate(code),
                atom_scalars_v ]

    h_(v,i) = LayerNorm(LocalMLP(x_(v,i)))          (shared frozen LocalMLP)
    one shared dropout(0.10) on h
    z_i     = sum_v w(v,i) h_(v,i)                   (frozen routing weights)
    Z_i     = [z_i, log1p(n_bonds_i), log1p(n_atoms_i)]   (66-d block token)

Interface token from the *cross-boundary* S2 occurrences of the unordered block
pair ``{i,j}`` (each occurrence enters exactly one interface):

    t_ij = MLP_iface( sum_{occ in (i,j)} [type,end,joint,closure,content] )   (200 -> 64 -> 64)

Readout:

    c_i   = f_block(Z_i)                       MLP 66 -> 64 -> 1
    c_ij  = f_interface(t_ij)                  R1, MLP 64 -> 64 -> 1
    c_ij  = f_interface(t_ij + ctx_ij)         R2, ctx_ij = MLP_context([Z_i+Z_j, |Z_i-Z_j|, Z_i*Z_j])
    y     = bias + sum_i c_i + sum_{i<j} c_ij + rho_g(g, log1p(n_atoms))

``rho_g`` is the only whole-graph term: it never sees ``z``, ``c_i`` or ``c_ij``.
No mean(h), no mean(h^2), no other hidden global shortcut.

The equality ``sum_i [sum_v w(v,i) ...]`` is *conditional evidence routing*, not a
claim that a chemical atom is deleted when a block is switched off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import rr_common as C

M = C.modules()
DM = M["dissect_model"]
CM = M["cr_models"]

WIDTH = C.WIDTH
LOCAL_INPUT_DIM = 433
TRUNK_PREFIXES = ("atom_encoder.", "bond_encoder.", "local.")
TRUNK_BUFFERS = ("global_mean", "global_scale", "scalar_mean", "scalar_scale")
ARM_PREFIX = "trunk."


# ---------------------------------------------------------------------------
# initialisation streams
# ---------------------------------------------------------------------------


def _stream(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _init_mlp(dims: Sequence[int], generator: torch.Generator) -> Any:
    """``DM.MLP`` with ``nn.Linear`` default init drawn on an isolated stream."""
    module = DM.MLP([int(d) for d in dims])
    for lin in module.lins:
        fan_in = int(lin.weight.shape[1])
        with torch.no_grad():
            lin.weight.copy_(CM._linear_weight(int(lin.weight.shape[0]), fan_in, generator))
            lin.bias.copy_(CM._linear_bias(int(lin.bias.shape[0]), fan_in, generator))
    return module


# ---------------------------------------------------------------------------
# trunk
# ---------------------------------------------------------------------------


class RoutingTrunk(nn.Module):
    """The frozen C1 local path; identical parameter names/shapes as ``LocalTrunk``."""

    def __init__(self, spec, norm: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = CM.LocalTrunk(spec, norm).cfg
        self.spec = spec
        self.cfg = dict(cfg)
        self.atom_encoder = DM.MLP([spec.atom_feat_dim, int(cfg["atom_hidden"]), int(cfg["atom_out"])])
        self.bond_encoder = DM.MLP(
            [3 * int(cfg["atom_out"]) + spec.bond_feat_dim, int(cfg["bond_hidden"]), int(cfg["bond_out"])]
        )
        self.local = DM.LocalContent(spec, "C1", self.cfg)
        self.drop = nn.Dropout(C.DROPOUT)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))
        width = int(self.local.mlp.lins[0].in_features)
        if width != LOCAL_INPUT_DIM:
            raise RuntimeError(f"frozen C1 local input width {width} != {LOCAL_INPUT_DIM}")

    def load_canonical(self, state: Mapping[str, Any]) -> None:
        subset = {key: state[key] for key in CM.trunk_keys(state) if key in self.state_dict()}
        missing = set(self.state_dict()) - set(subset)
        if missing:
            raise RuntimeError(f"trunk state is missing keys: {sorted(missing)}")
        self.load_state_dict({k: v.clone() for k, v in subset.items()}, strict=True)

    def encode(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        else:
            bond_input = torch.zeros(
                (0, 3 * a.shape[1] + batch["bond_feat"].shape[1]), dtype=a.dtype, device=a.device
            )
        b = self.bond_encoder(bond_input)
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        return a, b, scal

    def z_global(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (batch["global"] - self.global_mean) / self.global_scale


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    variant: str
    n_global: int
    block_token_dim: int = C.BLOCK_TOKEN_DIM

    @property
    def context_dim(self) -> int:
        return 3 * self.block_token_dim


class RoutingModel(nn.Module):
    def __init__(self, spec, norm: Mapping[str, Any], variant: str) -> None:
        super().__init__()
        if variant not in C.MODELS:
            raise KeyError(variant)
        self.variant = variant
        self.spec = spec
        self.trunk = RoutingTrunk(spec, norm)
        self.n_global = int(spec.global_dim)
        self.model_spec = ModelSpec(variant=variant, n_global=self.n_global)

        # Frozen draw order per module on isolated streams.  R1 and R2 use the
        # *same* streams and shapes for every shared module.
        self.f_block = _init_mlp([C.BLOCK_TOKEN_DIM, C.IFACE_WIDTH, 1], _stream(C.INIT_STREAM_BLOCK))
        self.iface_mlp = _init_mlp(
            [C.IFACE_EVIDENCE_DIM, C.IFACE_WIDTH, C.IFACE_WIDTH], _stream(C.INIT_STREAM_IFACE)
        )
        self.f_interface = _init_mlp([C.IFACE_WIDTH, C.IFACE_WIDTH, 1], _stream(C.INIT_STREAM_IFACE + 1))
        self.rho_g = _init_mlp(
            [self.n_global + 1, C.RHO_HIDDEN, 1], _stream(C.INIT_STREAM_RHO)
        )
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        if variant == "R2":
            self.context = _init_mlp(
                [self.model_spec.context_dim, C.CONTEXT_WIDTH, C.CONTEXT_WIDTH],
                _stream(C.INIT_STREAM_CONTEXT),
            )

    # -- construction helpers ---------------------------------------------
    def load_canonical(self, state: Mapping[str, Any]) -> None:
        self.trunk.load_canonical(state)

    def parameter_report(self) -> dict[str, Any]:
        named = list(self.named_parameters())
        trunk = sum(p.numel() for k, p in named if k.startswith(ARM_PREFIX))
        readout = sum(p.numel() for k, p in named if not k.startswith(ARM_PREFIX))
        breakdown: dict[str, int] = {}
        for key, value in named:
            top = ".".join(key.split(".")[:2]) if key.startswith(ARM_PREFIX) else key.split(".")[0]
            breakdown[top] = breakdown.get(top, 0) + int(value.numel())
        return {
            "variant": self.variant,
            "total": int(sum(p.numel() for _, p in named)),
            "trunk": int(trunk),
            "readout": int(readout),
            "breakdown": dict(sorted(breakdown.items())),
            "readout_parameter_names": sorted(k for k, _ in named if not k.startswith(ARM_PREFIX)),
        }

    def readout_state(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().clone()
            for k, v in self.state_dict().items()
            if not k.startswith(ARM_PREFIX) and k not in TRUNK_BUFFERS
        }

    # -- forward -----------------------------------------------------------
    def _local_parts(self, batch: Mapping[str, torch.Tensor], a, b, scal):
        """Block-conditioned 433-d local inputs from the shared frozen encoders."""
        local = self.trunk.local
        n_bel = int(batch["belong_weight"].numel())
        rank = int(local.rank)
        width = a.shape[1]
        device, dtype = a.device, a.dtype

        bond_u = batch["bond_bucket_u"]
        bond_v = batch["bond_bucket_v"]
        b_sum = a.new_zeros((n_bel, width))
        if b.numel():
            b_sum = b_sum.index_add_(0, bond_u, b).index_add_(0, bond_v, b)

        nbr = a.new_zeros((n_bel, rank))
        if batch["nbr_code"].numel():
            nbr = nbr.index_add_(0, batch["nbr_bucket"], local.nbr_emb(batch["nbr_code"]))

        # whole-occurrence typed S2 evidence (type, end, joint, closure) and content
        if batch["path_cat"].numel():
            path_typed = torch.cat(
                [
                    local.path_emb(batch["path_cat"]),
                    local.end_emb(batch["path_end63"]),
                    local.joint_proj(local.path_u(batch["path_cat"]) * local.path_v(batch["path_end63"])),
                    local.close_emb(batch["path_closure"]),
                ],
                dim=1,
            )
            content = a[batch["path_end"]] * local.path_gate(batch["path_cat"])
        else:
            path_typed = a.new_zeros((0, 3 * rank + 8))
            content = a.new_zeros((0, width))
        evidence = torch.cat([path_typed, content], dim=1)
        if evidence.shape[1] != C.IFACE_EVIDENCE_DIM:
            raise RuntimeError(f"interface evidence width {evidence.shape[1]} != {C.IFACE_EVIDENCE_DIM}")

        path_bucket = batch["path_bucket"]
        internal = path_bucket >= 0
        path_agg = a.new_zeros((n_bel, path_typed.shape[1]))
        content_agg = a.new_zeros((n_bel, width))
        if bool(internal.any()):
            path_agg = path_agg.index_add_(0, path_bucket[internal], path_typed[internal])
            content_agg = content_agg.index_add_(0, path_bucket[internal], content[internal])

        iface_ev = a.new_zeros((int(batch["n_interfaces"]), C.IFACE_EVIDENCE_DIM))
        cross = batch["path_iface"] >= 0
        if bool(cross.any()):
            iface_ev = iface_ev.index_add_(0, batch["path_iface"][cross], evidence[cross])

        atom_rows = batch["belong_atom"]
        x = torch.cat([a[atom_rows], b_sum, nbr, path_agg, content_agg, scal[atom_rows]], dim=1)
        if x.shape[1] != LOCAL_INPUT_DIM:
            raise RuntimeError(f"block-conditioned local input width {x.shape[1]} != {LOCAL_INPUT_DIM}")
        return x, iface_ev

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        a, b, scal = self.trunk.encode(batch)
        x, iface_ev = self._local_parts(batch, a, b, scal)
        h = self.trunk.local.norm(self.trunk.local.mlp(x))
        h = self.trunk.drop(h)          # the one shared dropout of the local path

        n_blocks = int(batch["n_blocks"])
        n_graphs = int(batch["n_graphs"])
        z = h.new_zeros((n_blocks, WIDTH)).index_add_(
            0, batch["belong_block"], h * batch["belong_weight"].unsqueeze(1)
        )
        tokens = torch.cat(
            [
                z,
                torch.log1p(batch["blk_n_bonds"].to(z.dtype)).unsqueeze(1),
                torch.log1p(batch["blk_n_atoms"].to(z.dtype)).unsqueeze(1),
            ],
            dim=1,
        )
        c_block = self.f_block(tokens).view(-1)

        t_iface = self.iface_mlp(iface_ev)
        ctx = None
        if self.variant == "R2":
            za = tokens.index_select(0, batch["iface_a"])
            zb = tokens.index_select(0, batch["iface_b"])
            ctx = self.context(
                torch.cat([za + zb, (za - zb).abs(), za * zb], dim=1)
            )
            c_iface = self.f_interface(t_iface + ctx).view(-1)
        else:
            c_iface = self.f_interface(t_iface).view(-1)

        block_sum = c_block.new_zeros(n_graphs).index_add_(0, batch["block_graph"], c_block)
        iface_sum = c_iface.new_zeros(n_graphs).index_add_(0, batch["iface_graph"], c_iface)
        g = self.trunk.z_global(batch)
        rho_in = torch.cat([g, torch.log1p(batch["atom_counts"].to(g.dtype)).unsqueeze(1)], dim=1)
        if rho_in.shape[1] != self.n_global + 1:
            raise RuntimeError(f"rho_g width {rho_in.shape[1]} != {self.n_global + 1}")
        rho = self.rho_g(rho_in).view(-1)
        pred = self.bias.expand(n_graphs) + block_sum + iface_sum + rho

        if not return_aux:
            return pred
        return pred, {
            "pred_std": pred,
            "bias": self.bias.expand(n_graphs),
            "rho": rho,
            "block_sum": block_sum,
            "iface_sum": iface_sum,
            "c_block": c_block,
            "c_iface": c_iface,
            "z": z,
            "tokens": tokens,
            "t_iface": t_iface,
            "iface_context": ctx if ctx is not None else t_iface.new_zeros(t_iface.shape),
            "h_cost": h,
            "n_blocks_per_graph": batch["n_blocks_per_graph"],
            "n_ifaces_per_graph": batch["n_ifaces_per_graph"],
            "term_names": ["bias", "block_sum", "iface_sum", "rho"],
        }


# ---------------------------------------------------------------------------
# native block gate (architecture-defined module shutdown)
# ---------------------------------------------------------------------------


def gated_readout(parts: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], keep: torch.Tensor):
    """``y(m) = bias + rho + sum_i m_i c_i + sum_{i<j} m_i m_j c_ij``.

    ``keep`` is a 0/1 (or bool) tensor over the blocks of the batch.  Interfaces
    survive only when **both** endpoint blocks are kept, which is exactly the
    semantics used by the block-gating analysis.
    """
    c_block = parts["c_block"]
    c_iface = parts["c_iface"]
    k = keep.to(c_block.dtype)
    block_sum = c_block.new_zeros(int(batch["n_graphs"])).index_add_(
        0, batch["block_graph"], c_block * k
    )
    ka = k.index_select(0, batch["iface_a"])
    kb = k.index_select(0, batch["iface_b"])
    iface_sum = c_iface.new_zeros(int(batch["n_graphs"])).index_add_(
        0, batch["iface_graph"], c_iface * (ka * kb)
    )
    return parts["bias"] + parts["rho"] + block_sum + iface_sum


# ---------------------------------------------------------------------------
# construction / parameter accounting
# ---------------------------------------------------------------------------


def canonical_trunk_state(spec, norm) -> dict[str, torch.Tensor]:
    """The single canonical C1 initial state of the read-only source round."""
    return CM.canonical_c1_state(spec, norm)


def build_model(
    variant: str,
    spec,
    norm: Mapping[str, Any],
    canonical_state: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> RoutingModel:
    C.set_global_seeds(C.TRAIN_SEED)
    model = RoutingModel(spec, norm, variant)
    model.load_canonical(canonical_state)
    return model.to(device)


def expected_parameter_counts(n_global: int) -> dict[str, int]:
    """Frozen analytic counts (checked against the built modules)."""
    trunk = int(CM.TRUNK_PARAMS)
    f_block = C.BLOCK_TOKEN_DIM * C.IFACE_WIDTH + C.IFACE_WIDTH + C.IFACE_WIDTH + 1
    iface_mlp = C.IFACE_EVIDENCE_DIM * C.IFACE_WIDTH + C.IFACE_WIDTH + C.IFACE_WIDTH * C.IFACE_WIDTH + C.IFACE_WIDTH
    f_interface = C.IFACE_WIDTH * C.IFACE_WIDTH + C.IFACE_WIDTH + C.IFACE_WIDTH + 1
    rho = (int(n_global) + 1) * C.RHO_HIDDEN + C.RHO_HIDDEN + C.RHO_HIDDEN + 1
    context = 3 * C.BLOCK_TOKEN_DIM * C.CONTEXT_WIDTH + C.CONTEXT_WIDTH + C.CONTEXT_WIDTH * C.CONTEXT_WIDTH + C.CONTEXT_WIDTH
    r1 = trunk + f_block + iface_mlp + f_interface + rho + 1
    return {
        "trunk": trunk,
        "f_block": f_block,
        "iface_mlp": iface_mlp,
        "f_interface": f_interface,
        "rho_g": rho,
        "bias": 1,
        "context_R2": context,
        "R1_total": r1,
        "R2_total": r1 + context,
        "c1_total": int(CM.FULL_C1_PARAMS),
        "r1_over_c1": float(r1) / float(CM.FULL_C1_PARAMS),
        "r2_over_c1": float(r1 + context) / float(CM.FULL_C1_PARAMS),
        "r2_over_r1": float(r1 + context) / float(r1),
    }


__all__ = [
    "WIDTH",
    "LOCAL_INPUT_DIM",
    "RoutingTrunk",
    "RoutingModel",
    "ModelSpec",
    "canonical_trunk_state",
    "build_model",
    "expected_parameter_counts",
    "gated_readout",
]
