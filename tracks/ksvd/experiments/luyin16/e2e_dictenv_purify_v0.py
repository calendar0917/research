"""E2E-DictEnv-Purify-v0 — four-layer clean dictionary environment.

Round ``e2e_dictenv_purify_v0`` · study ``zinc-context-gap``.
Architecture audit: ``tracks/ksvd/notes/e2e_dictenv_purify_v0_architecture_audit.md``
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_purify_v0_preregistration.md``

The canonical H1 reference (E2E-DictEnv-P2-ABS winner) is reorganised into four
concept layers, and exactly one architectural hypothesis is exposed on top of
them:

```text
StructuralBasis          structure defines the basis
AttributedLocalMeasure   attributes value the basis
StaticComposer           static composition builds the molecule
PropertyReader           final prediction only
```

Two flavours share every line of logic:

``reference``
    exact H1 arithmetic: 62-D primitive anchor ``p_i``, ``alpha_v W_A_S`` and
    ``g_uv W_E_S``.

``purified``
    the one authorized change of this round — a fixed constant structural
    channel is concatenated to the structural coordinate and the anchor is
    reduced to the two pure-structural size scalars::

        beta_v    = [1 ; alpha_v]                                  in R^33
        u_v       = ((beta_v W_A_S) . (q_v W_A_C)) / sqrt(96)       W_A_S: 33 x 96
        gamma_uv  = [1 ; g_uv]                                     in R^97
        g_uv      = [au+av ; |au-av| ; au o av]                    in R^96
        u_uv      = ((gamma_uv W_E_S) . (b_uv W_E_C)) / sqrt(48)    W_E_S: 97 x 48
        anchor    = [log1p|V_i|, log1p|E_i|] (train-standardized)   in R^2

    The ``1`` is a fixed constant structural channel: it is **not** a dictionary
    atom, **not** part of ``alpha``, **not** reconstructed, **not** counted
    toward the top-8 sparsity constraint, and has no parameters of its own.

Everything else is frozen: phi65, K=32, s=8, IHT=10, the SDB-v0 K-SVD
dictionary, the tied reconstruction objective, the 15-D pure-topology pair
relation, the 5 distance buckets, the distance gate, the pair encoder, the
global structural invariant (``global_context[0:30]``), the graph-level
zeroth-order attributed measure (``global_context[30:62]``), the graph topology
block and the reader.  No message passing, no recurrence, no attention, no
pair->centre, no handcrafted shell x chemistry histogram.  Official ZINC test is
never loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

PROTOCOL_VERSION = "e2e_dictenv_purify_v0"

# frozen geometry (identical to the H1 reference)
PHI_DIM = int(p1.PHI_DIM)  # 65
K_ATOMS = int(p1.K_ATOMS)  # 32
SPARSITY = int(p1.SPARSITY)  # 8
IHT_STEPS = int(p1.IHT_STEPS)  # 10
ATOM_CATEGORIES = int(p1.ATOM_CATEGORIES)  # 28
BOND_CATEGORIES = int(p1.BOND_CATEGORIES)  # 4
N_SHELLS = int(p1.N_SHELLS)  # 3
SHELLPAIR_CLASSES = int(p1.SHELLPAIR_CLASSES)  # 6
ENV_DIM = int(p1.ENV_DIM)  # 48
D_A = int(p1.D_A)  # 96
D_E = int(p1.D_E)  # 48
PAIR_HIDDEN = int(p1.PAIR_HIDDEN)  # 16
DISTANCE_BUCKETS = int(p1.DISTANCE_BUCKETS)  # 5
RELATION_WIDTH = int(p1.RELATION_WIDTH)  # 15
P1_RELATION_INDICES: tuple[int, ...] = tuple(p1.P1_RELATION_INDICES)
GLOBAL_WIDTH = int(p1.GLOBAL_WIDTH)  # 62
TOPOLOGY_IN = int(p1.TOPOLOGY_IN)  # 25
TOPOLOGY_HIDDEN = int(p1.TOPOLOGY_HIDDEN)  # 16
TOPOLOGY_OUT = int(p1.TOPOLOGY_OUT)  # 8
READER_HIDDEN = p1.READER_HIDDEN  # (13, 13)
BACKEND_DROPOUT = float(p1.BACKEND_DROPOUT)  # 0.05
EPS = float(v0.EPS)

#: anchor layout of the reference, in order (identical to P1)
ANCHOR_DIM_REFERENCE = int(p1.ANCHOR_DIM)  # 62
ANCHOR_ROOT = p1.ANCHOR_ROOT  # slice(0, 28)
ANCHOR_ATOM_MASS = p1.ANCHOR_ATOM_MASS  # slice(28, 56)
ANCHOR_BOND_MASS = p1.ANCHOR_BOND_MASS  # slice(56, 60)
ANCHOR_SIZE = p1.ANCHOR_SIZE  # slice(60, 62)
ANCHOR_DIM_PURIFIED = 2

#: global context composition (verified against zinc_long_range_proxy.global_feature_views)
GLOBAL_STRUCTURAL_INVARIANT = slice(0, 30)  # pure topology: short + long structural blocks
GLOBAL_ZEROTH_ORDER_ATTRIBUTE = slice(30, 62)  # whole-graph atom marginal 28 + bond marginal 4

#: H1 decoder topology
H1_NODE_HIDDEN = 64
H1_NODE_OUT = 48
H1_EDGE_HIDDEN = 48
H1_EDGE_OUT = 32
H1_ANCHOR_OUT = 32
H1_FUSION_HIDDEN = 128

#: frozen H1/T1 reconstruction weight (no recalibration, no sweep)
LAMBDA_REC = float(p1.LAMBDA_REC)  # 33.95873017865987

#: constant-channel initialization: deterministic, separate generator, never
#: displaces a shared tensor (pre-registration section 5)
CONSTANT_INIT_MULT = 1000003
CONSTANT_INIT_OFFSET_NODE = 17
CONSTANT_INIT_OFFSET_EDGE = 29

FLAVORS = ("reference", "purified")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PurifyConfig:
    """Frozen configuration of one arm of the round."""

    flavor: str
    K: int = K_ATOMS
    s: int = SPARSITY
    iht_steps: int = IHT_STEPS
    d_a: int = D_A
    d_e: int = D_E
    lambda_rec: float = LAMBDA_REC
    horizon: int = 320

    def __post_init__(self) -> None:
        if self.flavor not in FLAVORS:
            raise ValueError(f"unknown flavor {self.flavor!r}")

    @property
    def constant_node_channel(self) -> bool:
        return self.flavor == "purified"

    @property
    def constant_edge_channel(self) -> bool:
        return self.flavor == "purified"

    @property
    def anchor_dim(self) -> int:
        return ANCHOR_DIM_PURIFIED if self.flavor == "purified" else ANCHOR_DIM_REFERENCE

    @property
    def node_structure_dim(self) -> int:
        return int(self.K) + (1 if self.constant_node_channel else 0)

    @property
    def edge_structure_dim(self) -> int:
        return 3 * int(self.K) + (1 if self.constant_edge_channel else 0)

    @property
    def fusion_in(self) -> int:
        return H1_ANCHOR_OUT + N_SHELLS * H1_NODE_OUT + SHELLPAIR_CLASSES * H1_EDGE_OUT

    def as_dict(self) -> dict[str, Any]:
        return {
            "flavor": self.flavor,
            "K": int(self.K),
            "s": int(self.s),
            "iht_steps": int(self.iht_steps),
            "d_a": int(self.d_a),
            "d_e": int(self.d_e),
            "lambda_rec": float(self.lambda_rec),
            "horizon": int(self.horizon),
            "anchor_dim": int(self.anchor_dim),
            "constant_node_channel": bool(self.constant_node_channel),
            "constant_edge_channel": bool(self.constant_edge_channel),
            "node_structure_dim": int(self.node_structure_dim),
            "edge_structure_dim": int(self.edge_structure_dim),
            "fusion_in": int(self.fusion_in),
        }


def reference_config(**overrides: Any) -> PurifyConfig:
    return PurifyConfig(flavor="reference", **overrides)


def purified_config(**overrides: Any) -> PurifyConfig:
    return PurifyConfig(flavor="purified", **overrides)


# ---------------------------------------------------------------------------
# layer 1 — StructuralBasis : structure defines the basis
# ---------------------------------------------------------------------------


class StructuralBasis(nn.Module):
    """``phi_v in R^65 -> trainable dictionary D -> sparse coordinate alpha_v``.

    Math (frozen, unchanged from P1/P2)::

        Dbar_k = D_k / max(||D_k||_2, eps)
        alpha_v = H_s[ alpha_v + eta (phi_v - alpha_v Dbar^T) Dbar ]   (10 unrolled
                  tied-IHT steps, eta = 1/(sigma^2 + eps), exact top-s)
        hat_phi_v = alpha_v Dbar^T
        L_rec = E_v ||phi_v - hat_phi_v||^2 / (||phi_v||^2 + eps)

    The basis is pure topology: neither ``D`` nor ``alpha`` ever reads an atom or
    bond attribute.  ``K``, ``s``, the IHT step count and the reconstruction
    objective are frozen.
    """

    def __init__(self, dictionary: np.ndarray, K: int = K_ATOMS, s: int = SPARSITY, iht_steps: int = IHT_STEPS) -> None:
        super().__init__()
        self.K = int(K)
        self.s = int(s)
        self.iht_steps = int(iht_steps)
        self.D = nn.Parameter(torch.as_tensor(np.asarray(dictionary, dtype=np.float32)).clone())

    def normalized_dictionary(self) -> torch.Tensor:
        return v0.normalized_dictionary(self.D)

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        """Exact top-``s`` tied-IHT sparse structural coordinate."""
        return v0.tied_iht_codes(self.normalized_dictionary(), phi, s=self.s, steps=self.iht_steps)

    def reconstruct(self, coord: torch.Tensor) -> torch.Tensor:
        return coord @ self.normalized_dictionary().t()

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        phi_hat = self.reconstruct(coord)
        numerator = ((phi - phi_hat) ** 2).sum(dim=1)
        denominator = (phi ** 2).sum(dim=1) + EPS
        return (numerator / denominator).mean()


# ---------------------------------------------------------------------------
# layer 2 — AttributedLocalMeasure : attributes value the basis
# ---------------------------------------------------------------------------


@dataclass
class Interventions:
    """Inference-only interventions (mechanism audit + constant-channel ablation).

    One parameter object instead of scattered flags: the semantics are explicit
    and identical for both flavours.
    """

    coord_zero: bool = False
    occ_coord_node: torch.Tensor | None = None
    bond_u: torch.Tensor | None = None
    bond_v: torch.Tensor | None = None
    constant_node_scale: float = 1.0
    constant_edge_scale: float = 1.0

    @property
    def constant_node_off(self) -> bool:
        return float(self.constant_node_scale) == 0.0

    @property
    def constant_edge_off(self) -> bool:
        return float(self.constant_edge_scale) == 0.0


class AttributedLocalMeasure(nn.Module):
    """Structure coordinate x attribute valuation, routed into shell slots.

    Node (per occurrence ``v`` of root ``i`` in shell ``s``)::

        reference:  u_v = ((alpha_v W_A_S) . (q_v W_A_C)) / sqrt(96)
        purified:   u_v = (([1 ; alpha_v] W_A_S) . (q_v W_A_C)) / sqrt(96)
        A_i = [ sum_{s=0}^{2} sum_{v: s_iv=s} u_v ]      in R^{288}

    Edge (per bond ``uv`` of root ``i`` in shell-pair class ``p``)::

        g_uv = [alpha_u + alpha_v ; |alpha_u - alpha_v| ; alpha_u o alpha_v]
        reference:  u_uv = ((g_uv W_E_S) . (b_uv W_E_C)) / sqrt(48)
        purified:   u_uv = (([1 ; g_uv] W_E_S) . (b_uv W_E_C)) / sqrt(48)
        E_i^edge = [ sum_{p=0}^{5} sum_{uv: p_i(uv)=p} u_uv ]   in R^{288}

    Shell and shell-pair are routing indices only.  The attribute matrices
    ``W_A_C`` (28 x 96) and ``W_E_C`` (4 x 48) are the only place where a raw
    atom/bond category enters a local measure; in the purified flavour the
    constant structural rows of ``W_A_S`` / ``W_E_S`` carry the zeroth-order
    attribute content through the same elementwise product.
    """

    def __init__(self, K: int, d_a: int, d_e: int, constant_node_channel: bool, constant_edge_channel: bool) -> None:
        super().__init__()
        self.K = int(K)
        self.d_a = int(d_a)
        self.d_e = int(d_e)
        self.constant_node_channel = bool(constant_node_channel)
        self.constant_edge_channel = bool(constant_edge_channel)
        node_rows = self.K + (1 if self.constant_node_channel else 0)
        edge_rows = 3 * self.K + (1 if self.constant_edge_channel else 0)
        self.W_A_S = nn.Parameter(torch.empty(node_rows, self.d_a))
        self.W_A_C = nn.Parameter(torch.empty(ATOM_CATEGORIES, self.d_a))
        self.W_E_S = nn.Parameter(torch.empty(edge_rows, self.d_e))
        self.W_E_C = nn.Parameter(torch.empty(BOND_CATEGORIES, self.d_e))
        for parameter in (self.W_A_S, self.W_A_C, self.W_E_S, self.W_E_C):
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5.0))

    # -- structure x attribute ------------------------------------------------

    def _node_structure(self, coord: torch.Tensor, scale: float) -> torch.Tensor:
        if not self.constant_node_channel:
            return coord
        ones = torch.ones((coord.shape[0], 1), device=coord.device, dtype=coord.dtype) * float(scale)
        return torch.cat([ones, coord], dim=1)

    def _edge_structure(self, left: torch.Tensor, right: torch.Tensor, scale: float) -> torch.Tensor:
        g = torch.cat([left + right, torch.abs(left - right), left * right], dim=1)
        if not self.constant_edge_channel:
            return g
        ones = torch.ones((g.shape[0], 1), device=g.device, dtype=g.dtype) * float(scale)
        return torch.cat([ones, g], dim=1)

    def node_units(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        occ_coord_node: torch.Tensor | None = None,
        constant_scale: float = 1.0,
    ) -> torch.Tensor:
        """Per-occurrence node units ``u_v`` before shell aggregation."""
        gather = data.env_occ_node if occ_coord_node is None else occ_coord_node
        q = F.one_hot(data.dict_atom, num_classes=ATOM_CATEGORIES).to(coord.dtype)
        gather = gather.to(coord.device)
        structure = self._node_structure(coord[gather], constant_scale)
        attribute = q[data.env_occ_node.to(coord.device)].to(coord.dtype)
        return (structure @ self.W_A_S) * (attribute @ self.W_A_C) / math.sqrt(float(self.d_a))

    def edge_units(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
        constant_scale: float = 1.0,
    ) -> torch.Tensor:
        """Per-bond units ``u_uv`` before shell-pair aggregation."""
        u_index = data.env_bond_u if bond_u is None else bond_u
        v_index = data.env_bond_v if bond_v is None else bond_v
        u_index = u_index.to(coord.device)
        v_index = v_index.to(coord.device)
        structure = self._edge_structure(coord[u_index], coord[v_index], constant_scale)
        attribute = F.one_hot(data.env_bond_type, num_classes=BOND_CATEGORIES).to(coord.dtype)
        return (structure @ self.W_E_S) * (attribute @ self.W_E_C) / math.sqrt(float(self.d_e))

    # -- routing into the local slots ----------------------------------------

    def node_slots(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        occ_coord_node: torch.Tensor | None = None,
        constant_scale: float = 1.0,
    ) -> torch.Tensor:
        """``A_i`` in R^{3 x 96}: 3 shell slots, shell is a routing index."""
        n = int(coord.shape[0])
        u = self.node_units(coord, data, occ_coord_node=occ_coord_node, constant_scale=constant_scale)
        flat = torch.zeros((n * N_SHELLS, self.d_a), device=u.device, dtype=u.dtype)
        flat.index_add_(
            0,
            data.env_occ_root.to(coord.device) * N_SHELLS + data.env_occ_shell.to(coord.device),
            u,
        )
        return flat.view(n, N_SHELLS, self.d_a)

    def edge_slots(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
        constant_scale: float = 1.0,
    ) -> torch.Tensor:
        """``E_i^edge`` in R^{6 x 48}: 6 shell-pair slots, class is a routing index."""
        n = int(coord.shape[0])
        u = self.edge_units(coord, data, bond_u=bond_u, bond_v=bond_v, constant_scale=constant_scale)
        flat = torch.zeros((n * SHELLPAIR_CLASSES, self.d_e), device=u.device, dtype=u.dtype)
        flat.index_add_(
            0,
            data.env_bond_root.to(coord.device) * SHELLPAIR_CLASSES + data.env_bond_shellpair.to(coord.device),
            u,
        )
        return flat.view(n, SHELLPAIR_CLASSES, self.d_e)

    def slots(self, coord: torch.Tensor, data: Any, interventions: Interventions) -> tuple[torch.Tensor, torch.Tensor]:
        node = self.node_slots(
            coord,
            data,
            occ_coord_node=interventions.occ_coord_node,
            constant_scale=interventions.constant_node_scale,
        )
        edge = self.edge_slots(
            coord,
            data,
            bond_u=interventions.bond_u,
            bond_v=interventions.bond_v,
            constant_scale=interventions.constant_edge_scale,
        )
        return node, edge


# ---------------------------------------------------------------------------
# layer 3 — StaticComposer : static composition builds the molecule
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(int(in_dim), int(hidden)), nn.SiLU(), nn.Linear(int(hidden), int(out_dim)))


class StaticComposer(nn.Module):
    """Local environment construction + static pair composition + graph pooling.

    Frozen H1 layout (widths must not change)::

        node slot  96 -> 64 -> 48   (shared over the 3 shell slots)
        edge slot  48 -> 48 -> 32   (shared over the 6 shellpair slots)
        anchor      a -> 32 -> 32   (a = 62 reference / 2 purified)
        fusion  = [anchor_out 32 ; node_out 144 ; edge_out 192] = 368 -> 128 -> 48
        E_i = fusion(...)  in R^48

    Static composition (unchanged)::

        u_i = W_P E_i in R^16 (bias-free)
        pair_input = [u_i+u_j ; |u_i-u_j| ; u_i o u_j o gate(d_ij) ; rho_ij]   (64)
        gate(d) = 1 + tanh(Embedding(d))          (5 distance buckets)
        rho_ij = 15-D pure-topology relation (indices 0:14 and 18)
        pair value = _MLPBlock(64, 64, 16)
        unary      = pool_moments(E)                    (sum ; sum of squares ; log1p n)
        pair read  = per-bucket pool of the pair value  (5 x (16+16+1))
        global     = _MLPBlock(62 -> 32 -> 32) on
                     [GlobalStructuralInvariant 30 ; GlobalZerothOrderAttributeMeasure 32]
        topology   = 25 -> 16 -> 8

    No message passing, no recurrence, no attention, no pair->centre: ``E_i`` is
    formed once and never updated.
    """

    def __init__(self, config: PurifyConfig) -> None:
        super().__init__()
        self.node_encoder = _mlp(config.d_a, H1_NODE_HIDDEN, H1_NODE_OUT)
        self.edge_encoder = _mlp(config.d_e, H1_EDGE_HIDDEN, H1_EDGE_OUT)
        self.anchor_encoder = _mlp(config.anchor_dim, H1_ANCHOR_OUT, H1_ANCHOR_OUT)
        self.fusion = _mlp(config.fusion_in, H1_FUSION_HIDDEN, ENV_DIM)
        self.pair_projection = nn.Linear(ENV_DIM, PAIR_HIDDEN, bias=False)
        self.relation_encoder = zpp._MLPBlock(
            RELATION_WIDTH, max(PAIR_HIDDEN, 32), PAIR_HIDDEN, float(BACKEND_DROPOUT)
        )
        self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, PAIR_HIDDEN)
        self.pair_encoder = zpp._MLPBlock(
            4 * PAIR_HIDDEN, max(2 * PAIR_HIDDEN, 64), PAIR_HIDDEN, float(BACKEND_DROPOUT)
        )
        self.global_encoder = zpp._MLPBlock(
            GLOBAL_WIDTH, max(ENV_DIM // 2, 32), H1_ANCHOR_OUT, float(BACKEND_DROPOUT)
        )
        self.topology_encoder = nn.Sequential(
            nn.Linear(TOPOLOGY_IN, TOPOLOGY_HIDDEN),
            nn.ReLU(),
            nn.Linear(TOPOLOGY_HIDDEN, TOPOLOGY_OUT),
        )

    # -- local environment ---------------------------------------------------

    def local_anchor(self, data: Any, anchor_dim: int) -> torch.Tensor:
        """``[p_i]`` (reference, 62-D) or ``[log1p|V_i|, log1p|E_i|]`` (purified, 2-D).

        The purified flavour slices the two size coordinates out of the frozen
        train-standardized 62-D anchor (indices 60:62), so the size path is
        bit-identical to the reference's size path.
        """
        anchor = data.anchor
        if int(anchor_dim) == ANCHOR_DIM_REFERENCE:
            return anchor
        return anchor[:, ANCHOR_SIZE]

    def environment(self, anchor: torch.Tensor, node_slots: torch.Tensor, edge_slots: torch.Tensor) -> torch.Tensor:
        """``E_i in R^48`` — frozen after formation."""
        n = int(anchor.shape[0])
        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        anchor_out = self.anchor_encoder(anchor)
        fused = torch.cat([anchor_out, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1)
        return self.fusion(fused)

    # -- static composition --------------------------------------------------

    def compose(self, E: torch.Tensor, data: Any) -> dict[str, torch.Tensor]:
        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = v0.pool_moments(E, batch, n_graphs)

        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        left = u[source]
        right = u[target]
        relation = self.relation_encoder(data.pair_relation[:, list(P1_RELATION_INDICES)])
        product = left * right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_input = torch.cat([left + right, torch.abs(left - right), product * gate, relation], dim=1)
        pair_value = self.pair_encoder(pair_input)
        relation_readout = v0.pool_pair_moments(pair_value, batch[source], data.pair_bucket, n_graphs)

        global_out = self.global_encoder(data.global_context)
        topology_out = self.topology_encoder(data.topology_features)
        unified = torch.cat([unary, relation_readout, global_out, topology_out], dim=1)
        return {
            "unary": unary,
            "pair_value": pair_value,
            "relation_readout": relation_readout,
            "global_out": global_out,
            "topology_out": topology_out,
            "unified": unified,
        }


# ---------------------------------------------------------------------------
# layer 4 — PropertyReader : final prediction only
# ---------------------------------------------------------------------------


class PropertyReader(nn.Module):
    """``302 -> (13, 13) -> 1`` (``GenericReader``), frozen: no widening, no
    deepening, no activation change, no normalization, no dropout, no attention."""

    def __init__(self, in_dim: int | None = None) -> None:
        super().__init__()
        width = UNIFIED_DIM if in_dim is None else int(in_dim)
        self.reader = GenericReader(width, READER_HIDDEN)

    def forward(self, unified: torch.Tensor) -> torch.Tensor:
        return self.reader(unified)


UNIFIED_DIM = (
    (ENV_DIM + ENV_DIM + 1)
    + DISTANCE_BUCKETS * (2 * PAIR_HIDDEN + 1)
    + H1_ANCHOR_OUT
    + TOPOLOGY_OUT
)  # 97 + 165 + 32 + 8 = 302


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class PurifyV0Model(nn.Module):
    """Four-layer clean dictionary environment (``reference`` / ``purified``)."""

    def __init__(self, config: PurifyConfig, dictionary: np.ndarray) -> None:
        super().__init__()
        self.config = config
        self.basis = StructuralBasis(dictionary, K=config.K, s=config.s, iht_steps=config.iht_steps)
        self.measure = AttributedLocalMeasure(
            K=config.K,
            d_a=config.d_a,
            d_e=config.d_e,
            constant_node_channel=config.constant_node_channel,
            constant_edge_channel=config.constant_edge_channel,
        )
        self.composer = StaticComposer(config)
        self.reader = PropertyReader()

    # -- structural basis ----------------------------------------------------

    @property
    def D(self) -> nn.Parameter:
        return self.basis.D

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        return self.basis.code(phi)

    def reconstruct(self, coord: torch.Tensor) -> torch.Tensor:
        """``hat_phi = alpha Dbar^T`` — identical definition for every flavour."""
        return self.basis.reconstruct(coord)

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        return self.basis.reconstruction_loss(phi, coord)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        data: Any,
        *,
        interventions: Interventions | None = None,
        return_aux: bool = False,
        **kwargs: Any,
    ):
        if interventions is None:
            interventions, kwargs = interventions_from_kwargs(**kwargs)
        if kwargs:
            raise TypeError(f"unexpected keyword arguments {sorted(kwargs)}")
        phi = data.dict_phi
        coord = self.code(phi)
        if interventions.coord_zero:
            coord = torch.zeros_like(coord)
        node_slots, edge_slots = self.measure.slots(coord, data, interventions)
        anchor = self.composer.local_anchor(data, self.config.anchor_dim).to(coord.dtype)
        E = self.composer.environment(anchor, node_slots, edge_slots)
        composed = self.composer.compose(E, data)
        prediction = self.reader(composed["unified"]).view(-1)
        if return_aux:
            aux = {
                "E": E,
                "coord": coord,
                "phi": phi,
                "anchor": anchor,
                "node_slots": node_slots,
                "edge_slots": edge_slots,
                **composed,
            }
            return prediction, aux
        return prediction


#: legacy keyword plumbing kept so that shared callers (P1-style evaluate helpers)
#: can drive the model without flavour-specific branches
_LEGACY_INTERVENTION_KEYS = ("coord_zero", "occ_coord_node", "bond_u", "bond_v")


def interventions_from_kwargs(**kwargs: Any) -> tuple[Interventions, dict[str, Any]]:
    """Split legacy flat kwargs into an :class:`Interventions` object."""
    payload = {key: kwargs.pop(key) for key in list(kwargs) if key in _LEGACY_INTERVENTION_KEYS}
    return Interventions(**payload), kwargs


# ---------------------------------------------------------------------------
# initialization (matched reference / purified pair)
# ---------------------------------------------------------------------------


def constant_row(dim: int, seed: int, offset: int) -> torch.Tensor:
    """Deterministic candidate-only constant structural row."""
    generator = torch.Generator().manual_seed(int(seed) * CONSTANT_INIT_MULT + int(offset))
    row = torch.empty(1, int(dim))
    nn.init.kaiming_uniform_(row, a=math.sqrt(5.0), generator=generator)
    return row


def build_model(config: PurifyConfig, dictionary: np.ndarray, seed: int = 0) -> PurifyV0Model:
    """Build one arm under ``seed`` with the repository-standard init order."""
    torch.manual_seed(int(seed))
    return PurifyV0Model(config, dictionary)


def shared_keys(reference: PurifyV0Model, candidate: PurifyV0Model) -> dict[str, str]:
    """Same-name tensors whose shapes agree between the two flavours."""
    ref_state = reference.state_dict()
    cand_state = candidate.state_dict()
    return {
        key: key
        for key, value in cand_state.items()
        if key in ref_state and ref_state[key].shape == value.shape
    }


def build_matched_pair(
    dictionary: np.ndarray,
    seed: int = 0,
    *,
    reference: PurifyConfig | None = None,
    purified: PurifyConfig | None = None,
) -> tuple[PurifyV0Model, PurifyV0Model, dict[str, Any]]:
    """Matched-init pair used by the formal runs.

    * the reference model is built with ``torch.manual_seed(seed)`` and is
      bit-identical to ``e2e_dictenv_p2_abs.build_model`` for the H1 config;
    * every same-shape shared tensor is copied into the candidate
      (``max_abs_diff`` exactly 0);
    * the purified anchor encoder takes the size slice (columns ``60:62``) of the
      reference anchor encoder with the reference bias;
    * the candidate-only constant rows are deterministic draws from a separate
      generator.
    """
    reference = reference_config() if reference is None else reference
    purified = purified_config() if purified is None else purified
    reference_model = build_model(reference, dictionary, seed=seed)
    candidate_model = build_model(purified, dictionary, seed=seed)
    ref_state = reference_model.state_dict()
    candidate_state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        constant_node = constant_row(D_A, seed, CONSTANT_INIT_OFFSET_NODE)[0]
        constant_edge = constant_row(D_E, seed, CONSTANT_INIT_OFFSET_EDGE)[0]
        for key, value in candidate_model.state_dict().items():
            if key == "measure.W_A_S":
                candidate_state[key] = torch.cat([constant_node.unsqueeze(0), ref_state["measure.W_A_S"]], dim=0)
            elif key == "measure.W_E_S":
                candidate_state[key] = torch.cat([constant_edge.unsqueeze(0), ref_state["measure.W_E_S"]], dim=0)
            elif key == "composer.anchor_encoder.0.weight":
                # keep the reference weights of the two size coordinates
                candidate_state[key] = ref_state[key][:, ANCHOR_SIZE].clone()
            elif key in ref_state and ref_state[key].shape == value.shape:
                candidate_state[key] = ref_state[key]
            else:  # pragma: no cover - defensive
                candidate_state[key] = value
    candidate_model.load_state_dict(candidate_state)
    report = shared_init_report(reference_model, candidate_model)
    return reference_model, candidate_model, report


def shared_init_report(reference: PurifyV0Model, candidate: PurifyV0Model) -> dict[str, Any]:
    """Verify the frozen init-fairness claims (max_abs_diff == 0 on shared tensors)."""
    ref_state = reference.state_dict()
    cand_state = candidate.state_dict()
    shared = shared_keys(reference, candidate)
    diffs: dict[str, float] = {}
    for key in shared:
        left = ref_state[key].float()
        right = cand_state[key].float()
        diffs[key] = float((left - right).abs().max().item()) if left.numel() else 0.0
    size_weight_ok = bool(
        torch.equal(
            cand_state["composer.anchor_encoder.0.weight"],
            ref_state["composer.anchor_encoder.0.weight"][:, ANCHOR_SIZE],
        )
    )
    return {
        "n_shared_tensors": int(len(shared)),
        "shared_keys": sorted(shared),
        "max_abs_diff": float(max(diffs.values())) if diffs else 0.0,
        "per_tensor_max_abs_diff": diffs,
        "all_shared_identical": bool(all(value == 0.0 for value in diffs.values())),
        "anchor_size_slice_identical": size_weight_ok,
        "constant_node_row_deterministic": True,
        "constant_edge_row_deterministic": True,
    }


# ---------------------------------------------------------------------------
# legacy reference bridge (Stage-0 equivalence)
# ---------------------------------------------------------------------------

_P2_KEY_PREFIX = {
    "D": "basis.D",
    "W_A_S": "measure.W_A_S",
    "W_A_C": "measure.W_A_C",
    "W_E_S": "measure.W_E_S",
    "W_E_C": "measure.W_E_C",
    "node_encoder": "composer.node_encoder",
    "edge_encoder": "composer.edge_encoder",
    "anchor_encoder": "composer.anchor_encoder",
    "fusion": "composer.fusion",
    "pair_projection": "composer.pair_projection",
    "relation_encoder": "composer.relation_encoder",
    "distance_gate": "composer.distance_gate",
    "pair_encoder": "composer.pair_encoder",
    "global_encoder": "composer.global_encoder",
    "topology_encoder": "composer.topology_encoder",
    "reader": "reader.reader",
}


def p2_key_to_purify(key: str) -> str:
    """Map a ``e2e_dictenv_p2_abs.P2Model`` state key onto this module's layout."""
    head, _, rest = key.partition(".")
    if head not in _P2_KEY_PREFIX:
        raise KeyError(f"unknown P2 state key {key!r}")
    target = _P2_KEY_PREFIX[head]
    return target if not rest else f"{target}.{rest}"


def load_p2_state(model: PurifyV0Model, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Load an H1 reference checkpoint (P2Model or P1Model layout) verbatim."""
    mapped = {p2_key_to_purify(key): value for key, value in state.items()}
    current = model.state_dict()
    missing = sorted(set(current) - set(mapped))
    unexpected = sorted(set(mapped) - set(current))
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing} unexpected={unexpected}")
    for key, value in mapped.items():
        if current[key].shape != value.shape:
            raise RuntimeError(f"shape mismatch at {key}: {tuple(current[key].shape)} vs {tuple(value.shape)}")
    model.load_state_dict(mapped)
    return {"n_tensors": int(len(mapped)), "keys": sorted(mapped)}


# ---------------------------------------------------------------------------
# parameter ledger / purity audit / information flow
# ---------------------------------------------------------------------------


def _decoder_ledger(config: PurifyConfig) -> dict[str, Any]:
    node_in = config.d_a
    node = (node_in * H1_NODE_HIDDEN + H1_NODE_HIDDEN) + (H1_NODE_HIDDEN * H1_NODE_OUT + H1_NODE_OUT)
    edge = (config.d_e * H1_EDGE_HIDDEN + H1_EDGE_HIDDEN) + (H1_EDGE_HIDDEN * H1_EDGE_OUT + H1_EDGE_OUT)
    anchor = (config.anchor_dim * H1_ANCHOR_OUT + H1_ANCHOR_OUT) + (H1_ANCHOR_OUT * H1_ANCHOR_OUT + H1_ANCHOR_OUT)
    fusion = (config.fusion_in * H1_FUSION_HIDDEN + H1_FUSION_HIDDEN) + (H1_FUSION_HIDDEN * ENV_DIM + ENV_DIM)
    return {
        "node_slot_decoder": int(node),
        "edge_slot_decoder": int(edge),
        "anchor_decoder": int(anchor),
        "fusion_decoder": int(fusion),
        "subtotal": int(node + edge + anchor + fusion),
    }


def _backend_ledger() -> dict[str, int]:
    return {key: int(value) for key, value in p1.backend_parameter_count().items()}


def parameter_ledger(config: PurifyConfig) -> dict[str, Any]:
    """Exact trainable-parameter ledger of one flavour."""
    dictionary = PHI_DIM * int(config.K)
    node_binding = int(config.node_structure_dim) * config.d_a + ATOM_CATEGORIES * config.d_a
    edge_binding = int(config.edge_structure_dim) * config.d_e + BOND_CATEGORIES * config.d_e
    decoder = _decoder_ledger(config)
    backend = _backend_ledger()
    local = dictionary + node_binding + edge_binding + decoder["subtotal"]
    whole = local + backend["subtotal"]
    return {
        "flavor": config.flavor,
        "modules": {
            "StructuralBasis.dictionary": int(dictionary),
            "AttributedLocalMeasure.node_binding": int(node_binding),
            "AttributedLocalMeasure.edge_binding": int(edge_binding),
            "StaticComposer.node_slot_decoder": decoder["node_slot_decoder"],
            "StaticComposer.edge_slot_decoder": decoder["edge_slot_decoder"],
            "StaticComposer.anchor_decoder": decoder["anchor_decoder"],
            "StaticComposer.fusion_decoder": decoder["fusion_decoder"],
            "StaticComposer.pair_projection": int(backend["pair_projection"]),
            "StaticComposer.relation_encoder": int(backend["relation_encoder"]),
            "StaticComposer.distance_gate": int(backend["distance_gate"]),
            "StaticComposer.pair_encoder": int(backend["pair_encoder"]),
            "StaticComposer.global_encoder": int(backend["global_encoder"]),
            "StaticComposer.topology_encoder": int(backend["topology_encoder"]),
            "PropertyReader.reader": int(backend["reader"]),
        },
        "layers": {
            "StructuralBasis": int(dictionary),
            "AttributedLocalMeasure": int(node_binding + edge_binding),
            "StaticComposer": int(
                decoder["subtotal"]
                + backend["pair_projection"]
                + backend["relation_encoder"]
                + backend["distance_gate"]
                + backend["pair_encoder"]
                + backend["global_encoder"]
                + backend["topology_encoder"]
            ),
            "PropertyReader": int(backend["reader"]),
        },
        "dictionary": int(dictionary),
        "node_binding": int(node_binding),
        "edge_binding": int(edge_binding),
        "decoder": int(decoder["subtotal"]),
        "local_subtotal": int(local),
        "backend_subtotal": int(backend["subtotal"]),
        "whole_model": int(whole),
    }


def architecture_ledger() -> dict[str, Any]:
    """Ledger of both arms plus the frozen comparison (pre-registration section 4.4)."""
    reference = parameter_ledger(reference_config())
    purified = parameter_ledger(purified_config())
    delta = int(purified["whole_model"] - reference["whole_model"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "reference": reference,
        "purified": purified,
        "delta": {
            "absolute": delta,
            "percent": float(100.0 * delta / reference["whole_model"]),
            "within_parameter_rule": bool(purified["whole_model"] <= reference["whole_model"]),
        },
        "module_deltas": {
            key: int(purified["modules"].get(key, 0) - reference["modules"].get(key, 0))
            for key in sorted(set(reference["modules"]) | set(purified["modules"]))
        },
        "layer_deltas": {
            key: int(purified["layers"].get(key, 0) - reference["layers"].get(key, 0))
            for key in sorted(set(reference["layers"]) | set(purified["layers"]))
        },
        "official_test_loaded": False,
    }


def purity_audit(config: PurifyConfig) -> dict[str, Any]:
    """Structural purity accounting of one flavour (pre-registration section 14)."""
    ledger = parameter_ledger(config)
    purified = config.flavor == "purified"
    local_entries = [
        {"name": "node structure x atom attribute", "source": "alpha_v x q_v", "structure_conditioned": True},
        {"name": "edge structure relation x bond attribute", "source": "g_uv x b_uv", "structure_conditioned": True},
    ]
    local_raw_bypasses: list[dict[str, Any]] = []
    if not purified:
        local_raw_bypasses = [
            {"name": "anchor root atom identity", "dim": 28, "source": "q_i", "structure_conditioned": False},
            {"name": "anchor patch atom marginal", "dim": 28, "source": "sum_{v in P_i} q_v", "structure_conditioned": False},
            {"name": "anchor patch bond marginal", "dim": 4, "source": "sum_{e in E_i} b_e", "structure_conditioned": False},
        ]
    structural_scalar_slot = {"name": "local structural size", "dim": 2, "source": "[log1p|V_i|, log1p|E_i|]"}
    chemistry_params = ATOM_CATEGORIES * config.d_a + BOND_CATEGORIES * config.d_e
    if not purified:
        chemistry_params += ledger["modules"]["StaticComposer.anchor_decoder"]
    chemistry_params += ledger["modules"]["StaticComposer.global_encoder"] * 32 / GLOBAL_WIDTH
    dictionary_params = (
        ledger["modules"]["StructuralBasis.dictionary"]
        + int(config.K) * config.d_a
        + 3 * int(config.K) * config.d_e
    )
    return {
        "flavor": config.flavor,
        "local_chemistry_entry_points": len(local_entries) + len(local_raw_bypasses),
        "local_chemistry_structure_conditioned_entries": local_entries,
        "local_raw_chemistry_bypasses": local_raw_bypasses,
        "local_raw_chemistry_bypass_count": len(local_raw_bypasses),
        "global_chemistry_entry_points": 1,
        "global_chemistry_entry": {
            "name": "GlobalZerothOrderAttributeMeasure",
            "dims": 32,
            "source": "whole-graph atom marginal 28 + whole-graph bond marginal 4",
            "encoder": "shared 62 -> 32 -> 32 global encoder (fused with the 30-D structural invariant)",
        },
        "global_structural_invariant": {
            "name": "GlobalStructuralInvariant",
            "dims": 30,
            "source": "short + long pure-topology structural blocks",
        },
        "local_structural_scalar_slot": structural_scalar_slot,
        "trainable_parameters": ledger["whole_model"],
        "parameter_bearing_modules": len([v for v in ledger["modules"].values() if v > 0]),
        "chemistry_specific_parameters": float(chemistry_params),
        "dictionary_specific_parameters": int(dictionary_params),
        "constant_structural_channel": {
            "present": bool(purified),
            "node_channel": "[1 ; alpha_v] with the 1 frozen at 1.0",
            "edge_channel": "[1 ; g_uv] with the 1 frozen at 1.0",
            "is_dictionary_atom": False,
            "part_of_alpha": False,
            "reconstructed": False,
            "counted_toward_top_s": False,
            "has_parameters_of_its_own": False,
        },
        "official_test_loaded": False,
    }


def information_flow() -> dict[str, Any]:
    """Machine-readable description of the reference information flow."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "reference": "e2e_dictenv_p2_abs H1 (P2-ABS winner), params 97487",
        "edges": [
            {"from": "phi65", "to": "dictionary D", "kind": "pure topology", "dims": [65, 32]},
            {"from": "dictionary D", "to": "alpha", "kind": "tied IHT top-8 (10 steps)", "dims": [32]},
            {"from": "alpha", "to": "node structure x atom attribute", "kind": "shell routing", "dims": [96, 3]},
            {"from": "alpha endpoints", "to": "edge structure relation x bond attribute", "kind": "shellpair routing", "dims": [48, 6]},
            {"from": "raw anchor 62", "to": "anchor decoder 62->32->32", "kind": "raw chemistry bypass", "dims": [62, 32]},
            {"from": "slot decoders + anchor decoder", "to": "fusion 368->128->48", "kind": "local environment", "dims": [368, 48]},
            {"from": "local environment", "to": "pair projection (16)", "kind": "static pair composition", "dims": [48, 16]},
            {"from": "pair relation 15-D", "to": "pair encoder", "kind": "pure topology", "dims": [15]},
            {"from": "distance bucket 5", "to": "distance gate", "kind": "graph distance", "dims": [5, 16]},
            {"from": "local environment", "to": "unary pooling", "kind": "sum, sumsq, log1p count", "dims": [97]},
            {"from": "pair value 16", "to": "per-bucket pooling", "kind": "sum, sumsq, log1p count", "dims": [165]},
            {"from": "global_context[0:30]", "to": "GlobalStructuralInvariant", "kind": "pure topology", "dims": [30]},
            {"from": "global_context[30:62]", "to": "GlobalZerothOrderAttributeMeasure", "kind": "graph-level zeroth-order attributes", "dims": [32]},
            {"from": "GlobalStructuralInvariant + GlobalZerothOrderAttributeMeasure", "to": "global encoder 62->32->32", "kind": "fused encoder (frozen)", "dims": [62, 32]},
            {"from": "topology_features 25", "to": "topology encoder 25->16->8", "kind": "cycle / hinge global statistics", "dims": [25, 8]},
            {"from": "unary + pair readout + global + topology", "to": "reader 302->(13,13)->1", "kind": "property prediction", "dims": [302, 1]},
        ],
        "chemistry_entry_points": {
            "local_structure_conditioned": [
                {"tensor": "W_A_C", "shape": [28, 96], "path": "alpha_v x q_v"},
                {"tensor": "W_E_C", "shape": [4, 48], "path": "g_uv x b_uv"},
            ],
            "local_raw_bypass": [
                {"block": "anchor root atom identity", "dim": 28},
                {"block": "anchor patch atom marginal", "dim": 28},
                {"block": "anchor patch bond marginal", "dim": 4},
            ],
            "global": [{"block": "whole-graph atom marginal 28 + bond marginal 4", "dim": 32}],
        },
        "not_redundancy_candidates": [
            "global_context[0:30] pure topology (radius-2 local dictionary cannot recover long-range invariants)",
            "topology_features 25-D cycle/hinge statistics",
        ],
        "forbidden_paths": [
            "message passing",
            "recurrence",
            "pair->centre",
            "attention",
            "handcrafted shell x chemistry histogram (patch_cont / atom_shell / bond_shell)",
        ],
    }


def reference_inventory(dictionary_sha256: str | None = None) -> dict[str, Any]:
    """Inventory of the canonical reference this round re-trains."""
    ledger = parameter_ledger(reference_config())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "reference": "E2E-DictEnv-P2-ABS winner H1",
        "source_commit": "122db5dc (p2_abs formal run) / de136504 (this round's base)",
        "config": reference_config().as_dict(),
        "historical_context_only": {
            "soup_valid_mae": 0.12354862861608853,
            "soup_members": [293, 309, 313, 316, 317],
            "best_valid_mae": 0.129284,
            "best_epoch": 313,
        },
        "dictionary": {
            "kind": "sdb32",
            "source": "tracks/ksvd/results/sdb_v0/dictionary.pt",
            "shape": [PHI_DIM, K_ATOMS],
            "tensor_sha256": dictionary_sha256,
            "K": K_ATOMS,
            "s": SPARSITY,
            "iht_steps": IHT_STEPS,
        },
        "parameter_ledger": ledger,
        "module_table": information_flow()["edges"],
        "purity_audit": purity_audit(reference_config()),
        "official_test_loaded": False,
    }


__all__ = [
    "PROTOCOL_VERSION",
    "PHI_DIM",
    "K_ATOMS",
    "SPARSITY",
    "IHT_STEPS",
    "ATOM_CATEGORIES",
    "BOND_CATEGORIES",
    "N_SHELLS",
    "SHELLPAIR_CLASSES",
    "ENV_DIM",
    "D_A",
    "D_E",
    "ANCHOR_DIM_REFERENCE",
    "ANCHOR_DIM_PURIFIED",
    "ANCHOR_SIZE",
    "GLOBAL_STRUCTURAL_INVARIANT",
    "GLOBAL_ZEROTH_ORDER_ATTRIBUTE",
    "LAMBDA_REC",
    "UNIFIED_DIM",
    "FLAVORS",
    "PurifyConfig",
    "reference_config",
    "purified_config",
    "StructuralBasis",
    "AttributedLocalMeasure",
    "StaticComposer",
    "PropertyReader",
    "PurifyV0Model",
    "Interventions",
    "build_model",
    "build_matched_pair",
    "shared_keys",
    "shared_init_report",
    "constant_row",
    "p2_key_to_purify",
    "load_p2_state",
    "parameter_ledger",
    "architecture_ledger",
    "purity_audit",
    "information_flow",
    "reference_inventory",
]
