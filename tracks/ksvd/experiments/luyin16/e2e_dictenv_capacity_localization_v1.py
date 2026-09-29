"""Capacity localization + one-winner round on the closed ZINC dictionary line.

Round ``e2e_dictenv_capacity_localization_v1`` (Workstream Z, ZINC).
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_capacity_localization_v1_preregistration.md``.

The frozen capacity base is **CAP-BASE = CSSD-q1** (C6 clean mask + common
structural coordinate ``c1`` + q1-orthogonal residual sparse dictionary +
paired node/edge structure-semantic binding + full relation + first/second
moment readout; seed-0 Top-5 soup valid MAE 0.130028).

Three isolated, parameter-comparable capacity candidates are warm-screened for
40 epochs from the same CAP-BASE soup checkpoint against a continuation
control (M0):

* ``F`` — multi-rank factorized structure-semantic fusion (node + edge);
* ``R`` — two residual relation-conditioned environment-composition blocks;
* ``G`` — gated DeepSets (learned permutation-invariant) readout summaries.

Each candidate touches exactly one stage of the frozen forward pass; everything
else is bit-identical to CAP-BASE.  At most one qualifying candidate receives a
single from-scratch 320-epoch seed-0 run.  CPU only; the official ZINC test
split is never loaded (``official_test_loaded = false`` everywhere).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

PROTOCOL_VERSION = "e2e_dictenv_capacity_localization_v1"

#: frozen capacity base and the historical references it is compared against.
CAP_BASE = "CSSD-q1"
CAP_BASE_SOUP_MAE = 0.13002798487985273  # re-verified from the frozen checkpoint
FINAL_CLEAN_SPARSE_SOUP_MAE = float(cssd.REF_SPARSE_SEED0_SOUP_MAE)  # 0.12849851670576026

KINDS: tuple[str, ...] = ("M0", "F", "R", "G")

# ---------------------------------------------------------------------------
# frozen candidate architecture constants (pre-registration section 5)
# ---------------------------------------------------------------------------

#: Candidate F — multi-rank factorized structure-semantic residual fusion.
FUSION_HEADS = 4
FUSION_HEAD_DIM = 24
FUSION_PROJ_INIT = 0.01
FUSION_INIT_SEED = 20261001

#: Candidate R — relation-conditioned residual environment-composition blocks.
RELATION_BLOCKS = 2
RELATION_HIDDEN = 256
RELATION_RESIDUAL_INIT = 0.05
RELATION_INIT_SEED = 20261002

#: Candidate G — gated DeepSets readout summaries.
READOUT_SUMMARY_DIM = 192
READOUT_READER_INIT = 0.01
READOUT_SUMMARY_EPS = 1.0e-6
READOUT_INIT_SEED = 20261003

#: Screening protocol (pre-registration sections 6-8).
SCREEN_EPOCHS = 40
SCREEN_SOUP_WINDOW: tuple[int, int] = (21, 40)
SCREEN_SOUP_K = 5
SCREEN_LAST10 = 10

#: Gate (pre-registration section 8).
GATE_SOUP_DELTA = -0.004
GATE_LAST10_DELTA = -0.003
STRONG_SOUP_DELTA = -0.010
TIE_TOLERANCE = 0.002
TIE_ORDER: tuple[str, ...] = ("F", "R", "G")

#: Full run (pre-registration section 9).
FULL_EPOCHS = 320
FULL_SEED = 0

#: Full-run interpretation bands (pre-registration section 10).
FULL_NEUTRAL_ABOVE = 0.127
FULL_USEFUL_AT = 0.125
FULL_STRONG_AT = 0.120
FULL_VERY_STRONG_AT = 0.110

#: Parameter-budget rule (pre-registration section 4).
BUDGET_PREFERRED = (120_000, 160_000)
BUDGET_HARD_CEILING = 250_000
BUDGET_RATIO_MAX = 1.5


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


def cpu_only_guard(device: Any) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise RuntimeError(f"capacity-localization round is CPU-only, got device={resolved}")


def capacity_spec(kind: str) -> "CapacitySpec":
    kind = str(kind).upper()
    if kind not in KINDS:
        raise ValueError(f"unknown capacity kind {kind!r}")
    return CAPACITY_SPECS[kind]


@dataclass(frozen=True)
class CapacitySpec:
    """Declarative candidate identity (frozen architecture, no search axis)."""

    kind: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "description": self.description}


CAPACITY_SPECS: dict[str, CapacitySpec] = {
    "M0": CapacitySpec("M0", "CAP-BASE continuation control (no capacity module)"),
    "F": CapacitySpec(
        "F",
        "multi-rank factorized structure-semantic residual fusion "
        f"(H={FUSION_HEADS}, d_h={FUSION_HEAD_DIM})",
    ),
    "R": CapacitySpec(
        "R",
        "two residual relation-conditioned pair-composition blocks "
        f"(hidden={RELATION_HIDDEN}, FiLM on encoded relation)",
    ),
    "G": CapacitySpec(
        "G",
        "gated DeepSets readout summaries " f"(dim={READOUT_SUMMARY_DIM}, node + pair)",
    ),
}


# ---------------------------------------------------------------------------
# deterministic local initialization helpers
# ---------------------------------------------------------------------------


@contextmanager
def preserved_global_rng() -> Iterator[None]:
    """Run a block without advancing the global torch RNG stream.

    Candidate construction must not perturb the global RNG stream that the
    frozen training protocol consumes for dropout; the base parameters are
    constructed before this context, and every new parameter is initialized
    from an explicit local generator.
    """
    state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(state)


def _kaiming_uniform_generator(
    tensor: torch.Tensor, generator: torch.Generator, a: float = math.sqrt(5.0)
) -> None:
    """Reproduce ``nn.init.kaiming_uniform_(tensor, a)`` on a local generator."""
    fan_in = int(tensor.shape[1]) if tensor.dim() >= 2 else int(tensor.shape[0])
    gain = math.sqrt(2.0 / (1.0 + float(a) * float(a)))
    std = gain / math.sqrt(max(fan_in, 1))
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound, generator=generator)


def _linear_default_generator(linear: nn.Linear, generator: torch.Generator) -> None:
    """Reproduce the default ``nn.Linear`` initialization on a local generator."""
    _kaiming_uniform_generator(linear.weight, generator)
    if linear.bias is not None:
        bound = 1.0 / math.sqrt(max(int(linear.weight.shape[1]), 1))
        with torch.no_grad():
            linear.bias.uniform_(-bound, bound, generator=generator)


def _small_uniform_generator(tensor: torch.Tensor, generator: torch.Generator, scale: float) -> None:
    with torch.no_grad():
        tensor.uniform_(-float(scale), float(scale), generator=generator)


# ---------------------------------------------------------------------------
# the capacity model: CSSD + three isolated hook points
# ---------------------------------------------------------------------------


class CapacityModel(cssd.CSSDModel):
    """CSSD model with one optional capacity module and three hook points.

    ``forward`` is the frozen ``AuditModel`` forward with the environment,
    pair-composition and graph-readout steps routed through hooks.  With the
    default hooks and ``capacity_off`` the model reproduces ``CSSDModel``
    behaviour bit-for-bit on the same weights (asserted by focused tests).

    Hook contracts (all single-call, no message passing, no recurrence):

    * ``capacity_environments(coord, data, mask, fill) -> E``
    * ``capacity_pair(pair_input, relation, data) -> pair_value``
    * ``capacity_graph_repr(unary, relation_readout, graph_hidden, topology, ctx) -> repr``
    """

    CAPACITY_KIND = "M0"
    CAPACITY_NEW_PREFIXES: tuple[str, ...] = ()

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        self.capacity_off = False
        self.capacity_diagnostics: dict[str, Any] = {}

    # -- hooks (default = frozen CAP-BASE path) ------------------------------

    def capacity_environments(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: audit.AuditMask,
        fill: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        return super().environments_masked(coord, data, mask, fill)

    def capacity_pair(
        self, pair_input: torch.Tensor, relation: torch.Tensor, data: Any
    ) -> torch.Tensor:
        return self.pair_encoder(pair_input)

    def capacity_graph_repr(
        self,
        unary: torch.Tensor,
        relation_readout: torch.Tensor,
        graph_hidden: torch.Tensor,
        topology: torch.Tensor,
        ctx: Mapping[str, Any],
    ) -> torch.Tensor:
        return torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)

    # -- forward (frozen AuditModel forward, hooks only) ---------------------

    def forward(
        self,
        data: Any,
        *,
        mask: audit.AuditMask | None = None,
        fill: Mapping[str, torch.Tensor] | None = None,
        readout_perm: tuple[Sequence[str], torch.Tensor] | None = None,
        return_aux: bool = False,
        **_kwargs: Any,
    ):
        effective = audit.AuditMask() if mask is None else mask
        coord = self.code(data.dict_phi)
        if effective.coord_zero:
            coord = torch.zeros_like(coord)
        E = self.capacity_environments(coord, data, effective, fill)

        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = audit.pool_moments_masked(E, batch, n_graphs, effective.unary_zero_blocks, fill)

        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        if effective.pair_projection_zero:
            u = torch.zeros_like(u)
        left = u[source]
        right = u[target]
        relation_input = data.pair_relation[:, list(p1.P1_RELATION_INDICES)]
        if effective.relation_zero_groups:
            relation_input = audit._replace_grouped_columns(
                relation_input, audit.RELATION_GROUPS, effective.relation_zero_groups, fill, "relation:"
            )
        relation = self.relation_encoder(relation_input)
        product = left * right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        if effective.gate_off:
            gate = torch.ones_like(gate)
        pair_input = torch.cat([left + right, torch.abs(left - right), product * gate, relation], dim=1)
        pair_value = self.capacity_pair(pair_input, relation, data)
        pair_batch = batch[source]
        relation_readout = audit.pool_pair_moments_masked(
            pair_value, pair_batch, data.pair_bucket, n_graphs, effective.pair_zero_blocks, fill
        )
        if readout_perm is not None:
            blocks, permutation = readout_perm
            unary, relation_readout = audit.apply_readout_permutation(
                unary, relation_readout, blocks, permutation
            )

        global_input = data.global_context
        if effective.global_zero_groups:
            global_input = audit._replace_grouped_columns(
                global_input, audit.GLOBAL_GROUPS, effective.global_zero_groups, fill, "global:"
            )
        graph_hidden = self.global_encoder(global_input)
        if effective.graph_hidden_zero:
            graph_hidden = torch.zeros_like(graph_hidden)
        topology_input = data.topology_features
        if effective.topology_zero:
            if fill is not None and "topology" in fill:
                topology_input = (
                    fill["topology"]
                    .to(topology_input.device, topology_input.dtype)
                    .reshape(1, -1)
                    .expand_as(topology_input)
                    .contiguous()
                )
            else:
                topology_input = torch.zeros_like(topology_input)
        topology = self.topology_encoder(topology_input)

        ctx = {
            "E": E,
            "coord": coord,
            "pair_value": pair_value,
            "pair_batch": pair_batch,
            "batch": batch,
            "n_graphs": n_graphs,
        }
        unified = self.capacity_graph_repr(unary, relation_readout, graph_hidden, topology, ctx)
        prediction = self.reader(unified).view(-1)
        if return_aux:
            return prediction, {
                "E": E,
                "coord": coord,
                "phi": data.dict_phi,
                "unary": unary,
                "relation_readout": relation_readout,
                "pair_value": pair_value,
            }
        return prediction

    # -- capacity bookkeeping ------------------------------------------------

    def capacity_parameter_names(self) -> list[str]:
        names: list[str] = []
        for name, _parameter in self.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in self.CAPACITY_NEW_PREFIXES):
                names.append(name)
        return names

    def capacity_parameter_count(self) -> int:
        return int(sum(dict(self.named_parameters())[name].numel() for name in self.capacity_parameter_names()))


class FiLMResidualBlock(nn.Module):
    """``h <- h + W2 silu((1 + tanh(gamma(r))) odot W1 LN(h) + beta(r))``."""

    def __init__(self, width: int, hidden: int, relation_width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(width))
        self.fc1 = nn.Linear(int(width), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(width))
        self.gamma = nn.Linear(int(relation_width), int(hidden))
        self.beta = nn.Linear(int(relation_width), int(hidden))

    def forward(self, h: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        z = self.fc1(self.norm(h))
        z = (1.0 + torch.tanh(self.gamma(relation))) * z + self.beta(relation)
        return h + self.fc2(F.silu(z))


class FusionCapacityModel(CapacityModel):
    """Candidate F — multi-rank factorized structure-semantic residual fusion.

    Node:  ``u = u_base + P_N [ (c W^h_S) odot (q W^h_C) / sqrt(d_h) ]_h``
    Edge:  ``e = e_base + P_E [ (g W^h_ES) odot (b W^h_EC) / sqrt(d_h) ]_h``

    The base product path is untouched and the residual projections start
    near zero, so step-0 predictions stay close to CAP-BASE while every new
    module still receives gradient.
    """

    CAPACITY_KIND = "F"
    CAPACITY_NEW_PREFIXES = ("F_NS", "F_NC", "F_NP", "F_ES", "F_EC", "F_EP")

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        heads: int = FUSION_HEADS,
        head_dim: int = FUSION_HEAD_DIM,
        init_seed: int = FUSION_INIT_SEED,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        self.fusion_heads = int(heads)
        self.fusion_head_dim = int(head_dim)
        coord_width = int(self.W_A_S.shape[0])          # q + K atoms
        node_semantic = int(self.W_A_C.shape[0])        # atom categories
        edge_structure = int(self.W_E_S.shape[0])       # 3 * (q + K)
        bond_semantic = int(self.W_E_C.shape[0])        # bond categories
        node_out = int(self.W_A_S.shape[1])
        edge_out = int(self.W_E_S.shape[1])
        generator = torch.Generator().manual_seed(int(init_seed))
        with preserved_global_rng():
            self.F_NS = nn.ParameterList(
                [nn.Linear(coord_width, self.fusion_head_dim, bias=False) for _ in range(self.fusion_heads)]
            )
            self.F_NC = nn.ParameterList(
                [nn.Linear(node_semantic, self.fusion_head_dim, bias=False) for _ in range(self.fusion_heads)]
            )
            self.F_NP = nn.Linear(self.fusion_heads * self.fusion_head_dim, node_out, bias=False)
            self.F_ES = nn.ParameterList(
                [nn.Linear(edge_structure, self.fusion_head_dim, bias=False) for _ in range(self.fusion_heads)]
            )
            self.F_EC = nn.ParameterList(
                [nn.Linear(bond_semantic, self.fusion_head_dim, bias=False) for _ in range(self.fusion_heads)]
            )
            self.F_EP = nn.Linear(self.fusion_heads * self.fusion_head_dim, edge_out, bias=False)
        with torch.no_grad():
            for head in range(self.fusion_heads):
                _kaiming_uniform_generator(self.F_NS[head].weight, generator)
                _kaiming_uniform_generator(self.F_NC[head].weight, generator)
                _kaiming_uniform_generator(self.F_ES[head].weight, generator)
                _kaiming_uniform_generator(self.F_EC[head].weight, generator)
            _small_uniform_generator(self.F_NP.weight, generator, FUSION_PROJ_INIT)
            _small_uniform_generator(self.F_EP.weight, generator, FUSION_PROJ_INIT)

    # -- occurrence-level factorized interaction ------------------------------

    def _node_occurrence(self, occ: torch.Tensor, qc: torch.Tensor) -> torch.Tensor:
        base = (occ @ self.W_A_S) * (qc @ self.W_A_C) / math.sqrt(float(p2.D_A))
        if self.capacity_off:
            return base
        heads = [
            (self.F_NS[head](occ) * self.F_NC[head](qc)) / math.sqrt(float(self.fusion_head_dim))
            for head in range(self.fusion_heads)
        ]
        return base + self.F_NP(torch.cat(heads, dim=1))

    def _edge_occurrence(self, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        base = (g @ self.W_E_S) * (b @ self.W_E_C) / math.sqrt(float(self.W_E_S.shape[1]))
        if self.capacity_off:
            return base
        heads = [
            (self.F_ES[head](g) * self.F_EC[head](b)) / math.sqrt(float(self.fusion_head_dim))
            for head in range(self.fusion_heads)
        ]
        return base + self.F_EP(torch.cat(heads, dim=1))

    def head_products(self, coord: torch.Tensor, data: Any) -> dict[str, torch.Tensor]:
        """Per-head factorized products at node and edge occurrences (diagnostics)."""
        q = F.one_hot(data.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        occ = coord[data.env_occ_node.to(coord.device)]
        qc = q[data.env_occ_node.to(coord.device)]
        cu = coord[data.env_bond_u.to(coord.device)]
        cv = coord[data.env_bond_v.to(coord.device)]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        b = F.one_hot(data.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(coord.dtype)
        node = torch.stack(
            [self.F_NS[h](occ) * self.F_NC[h](qc) for h in range(self.fusion_heads)], dim=0
        )
        edge = torch.stack(
            [self.F_ES[h](g) * self.F_EC[h](b) for h in range(self.fusion_heads)], dim=0
        )
        return {"node": node, "edge": edge}

    # -- masked environment (frozen AuditModel structure + F occurrence) -----

    def capacity_environments(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: audit.AuditMask,
        fill: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        n = int(coord.shape[0])
        anchor = data.anchor.to(coord.dtype)
        if int(anchor.shape[1]) != int(audit.ANCHOR_DIM_EXPECTED):
            raise RuntimeError(f"anchor width {int(anchor.shape[1])} != {audit.ANCHOR_DIM_EXPECTED}")
        if mask.anchor_zero_groups:
            anchor = audit._replace_grouped_columns(
                anchor, audit.ANCHOR_GROUPS, mask.anchor_zero_groups, fill, "anchor:"
            )

        q = F.one_hot(data.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        occ_coord_node = data.env_occ_node
        if mask.use_node_shuffle:
            occ_coord_node = getattr(data, "env_occ_coord_node", None)
            if occ_coord_node is None:
                raise RuntimeError("node-shuffle intervention requires data.env_occ_coord_node")
        occ = coord[occ_coord_node.to(coord.device)]
        qc = q[data.env_occ_node.to(coord.device)]
        u = self._node_occurrence(occ, qc)
        if mask.node_binding_zero:
            u = torch.zeros_like(u)
        flat = torch.zeros((n * int(p2.N_SHELLS), int(u.shape[1])), device=u.device, dtype=u.dtype)
        flat.index_add_(
            0,
            data.env_occ_root.to(coord.device) * int(p2.N_SHELLS) + data.env_occ_shell.to(coord.device),
            u,
        )
        node_slots = flat.view(n, int(p2.N_SHELLS), int(u.shape[1]))

        bond_u = data.env_bond_u
        bond_v = data.env_bond_v
        if mask.use_edge_shuffle:
            bond_u = getattr(data, "env_bond_u_shuffled", None)
            bond_v = getattr(data, "env_bond_v_shuffled", None)
            if bond_u is None or bond_v is None:
                raise RuntimeError("edge-shuffle intervention requires shuffled endpoint fields")
        bond_u = bond_u.to(coord.device)
        bond_v = bond_v.to(coord.device)
        cu = coord[bond_u]
        cv = coord[bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        if mask.edge_binding_zero:
            g = torch.zeros_like(g)
        b = F.one_hot(data.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(coord.dtype)
        ue = self._edge_occurrence(g, b)
        flat_e = torch.zeros(
            (n * int(p2.SHELLPAIR_CLASSES), int(ue.shape[1])), device=ue.device, dtype=ue.dtype
        )
        flat_e.index_add_(
            0,
            data.env_bond_root.to(coord.device) * int(p2.SHELLPAIR_CLASSES)
            + data.env_bond_shellpair.to(coord.device),
            ue,
        )
        edge_slots = flat_e.view(n, int(p2.SHELLPAIR_CLASSES), int(ue.shape[1]))

        if self.env_mlp is not None:
            z = torch.cat([anchor, node_slots.reshape(n, -1), edge_slots.reshape(n, -1)], dim=1)
            return self.env_mlp(z)
        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        anchor_out = self.anchor_encoder(anchor)
        fused = torch.cat([anchor_out, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1)
        return self.fusion(fused)


class RelationCapacityModel(CapacityModel):
    """Candidate R — two residual relation-conditioned pair-composition blocks.

    Only ``(E_i, E_j, r_ij) -> q_ij`` is enriched.  Each forward computes the
    pair state exactly once and never writes back to the environments.
    """

    CAPACITY_KIND = "R"
    CAPACITY_NEW_PREFIXES = ("pair_blocks",)

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        blocks: int = RELATION_BLOCKS,
        hidden: int = RELATION_HIDDEN,
        init_seed: int = RELATION_INIT_SEED,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        self.relation_blocks = int(blocks)
        self.relation_hidden = int(hidden)
        generator = torch.Generator().manual_seed(int(init_seed))
        with preserved_global_rng():
            self.pair_blocks = nn.ModuleList(
                [
                    FiLMResidualBlock(int(p2.PAIR_HIDDEN), self.relation_hidden, int(p2.PAIR_HIDDEN))
                    for _ in range(self.relation_blocks)
                ]
            )
        with torch.no_grad():
            for block in self.pair_blocks:
                _kaiming_uniform_generator(block.fc1.weight, generator)
                _kaiming_uniform_generator(block.fc1.bias, generator)
                _kaiming_uniform_generator(block.fc2.weight, generator)
                block.fc2.weight.mul_(RELATION_RESIDUAL_INIT)
                _kaiming_uniform_generator(block.fc2.bias, generator)
                block.fc2.bias.mul_(RELATION_RESIDUAL_INIT)
                block.gamma.weight.zero_()
                block.gamma.bias.zero_()
                block.beta.weight.zero_()
                block.beta.bias.zero_()

    def capacity_pair(
        self, pair_input: torch.Tensor, relation: torch.Tensor, data: Any
    ) -> torch.Tensor:
        h = self.pair_encoder(pair_input)
        if self.capacity_off:
            return h
        for block in self.pair_blocks:
            h = block(h, relation)
        return h

    def pair_block_contributions(
        self, pair_input: torch.Tensor, relation: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Per-block output / FiLM modulation statistics (diagnostics)."""
        h = self.pair_encoder(pair_input)
        rows: list[dict[str, torch.Tensor]] = []
        for block in self.pair_blocks:
            z = block.fc1(block.norm(h))
            modulation = torch.tanh(block.gamma(relation))
            shift = block.beta(relation)
            y = block.fc2(F.silu((1.0 + modulation) * z + shift))
            rows.append({"contribution": y, "modulation": modulation, "shift": shift})
            h = h + y
        return rows


class ReadoutCapacityModel(CapacityModel):
    """Candidate G — gated DeepSets readout summaries.

    ``H_E`` / ``H_P`` are appended to the frozen first/second moment readout;
    the reader's original input columns are copied bit-identically and the new
    columns start near zero.  Permutation invariant, no attention.
    """

    CAPACITY_KIND = "G"
    CAPACITY_NEW_PREFIXES = (
        "summary_gate_env",
        "summary_value_env",
        "summary_gate_pair",
        "summary_value_pair",
    )

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        dim: int = READOUT_SUMMARY_DIM,
        init_seed: int = READOUT_INIT_SEED,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        self.summary_dim = int(dim)
        self.disabled_summaries: set[str] = set()
        generator = torch.Generator().manual_seed(int(init_seed))
        with preserved_global_rng():
            self.summary_gate_env = nn.Linear(int(p2.ENV_DIM), self.summary_dim)
            self.summary_value_env = nn.Linear(int(p2.ENV_DIM), self.summary_dim)
            self.summary_gate_pair = nn.Linear(int(p2.PAIR_HIDDEN), self.summary_dim)
            self.summary_value_pair = nn.Linear(int(p2.PAIR_HIDDEN), self.summary_dim)
            old_reader = self.reader
            widened = p2.GenericReader(
                int(audit.READER_IN_DIM) + 2 * self.summary_dim, tuple(p2.READER_HIDDEN)
            )
        with torch.no_grad():
            for module in (
                self.summary_gate_env,
                self.summary_value_env,
                self.summary_gate_pair,
                self.summary_value_pair,
            ):
                _linear_default_generator(module, generator)
            # bit-identical transfer of the CAP-BASE reader into the first block
            widened.net[0].weight[:, : int(audit.READER_IN_DIM)].copy_(old_reader.net[0].weight)
            widened.net[0].bias.copy_(old_reader.net[0].bias)
            widened.net[2].weight.copy_(old_reader.net[2].weight)
            widened.net[2].bias.copy_(old_reader.net[2].bias)
            widened.net[4].weight.copy_(old_reader.net[4].weight)
            widened.net[4].bias.copy_(old_reader.net[4].bias)
            _small_uniform_generator(
                widened.net[0].weight[:, int(audit.READER_IN_DIM) :], generator, READOUT_READER_INIT
            )
        self.reader = widened

    @staticmethod
    def gated_summary(
        values: torch.Tensor,
        batch: torch.Tensor,
        gate: nn.Linear,
        value: nn.Linear,
        n_graphs: int,
        eps: float = READOUT_SUMMARY_EPS,
    ) -> torch.Tensor:
        """``sum_i sigmoid(g(v_i)) odot value(v_i) / (sum_i sigmoid(g(v_i)) + eps)``."""
        activation = torch.sigmoid(gate(values))
        weighted = activation * value(values)
        numerator = torch.zeros(
            (int(n_graphs), int(weighted.shape[1])), device=values.device, dtype=values.dtype
        )
        denominator = torch.zeros(
            (int(n_graphs), int(activation.shape[1])), device=values.device, dtype=values.dtype
        )
        if int(values.shape[0]):
            numerator.index_add_(0, batch, weighted)
            denominator.index_add_(0, batch, activation)
        return numerator / (denominator + float(eps))

    def summary_values(
        self, E: torch.Tensor, batch: torch.Tensor, pair_value: torch.Tensor, pair_batch: torch.Tensor, n_graphs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_env = self.gated_summary(E, batch, self.summary_gate_env, self.summary_value_env, n_graphs)
        h_pair = self.gated_summary(
            pair_value, pair_batch, self.summary_gate_pair, self.summary_value_pair, n_graphs
        )
        return h_env, h_pair

    def capacity_graph_repr(
        self,
        unary: torch.Tensor,
        relation_readout: torch.Tensor,
        graph_hidden: torch.Tensor,
        topology: torch.Tensor,
        ctx: Mapping[str, Any],
    ) -> torch.Tensor:
        h_env, h_pair = self.summary_values(
            ctx["E"], ctx["batch"], ctx["pair_value"], ctx["pair_batch"], int(ctx["n_graphs"])
        )
        if self.capacity_off or "env" in self.disabled_summaries:
            h_env = torch.zeros_like(h_env)
        if self.capacity_off or "pair" in self.disabled_summaries:
            h_pair = torch.zeros_like(h_pair)
        return torch.cat([unary, relation_readout, graph_hidden, topology, h_env, h_pair], dim=1)


MODEL_BY_KIND: dict[str, type[CapacityModel]] = {
    "M0": CapacityModel,
    "F": FusionCapacityModel,
    "R": RelationCapacityModel,
    "G": ReadoutCapacityModel,
}


def build_capacity_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    kind: str,
    *,
    spec: cm.CleanMechSpec | None = None,
) -> CapacityModel:
    """Build a candidate with the exact CAP-BASE initialization stream.

    The global RNG stream advances exactly as it does for ``CSSDModel``: the
    base parameters consume the frozen stream and every capacity parameter is
    initialized from an explicit local generator inside
    :func:`preserved_global_rng`.
    """
    kind = str(kind).upper()
    if kind not in MODEL_BY_KIND:
        raise ValueError(f"unknown capacity kind {kind!r}")
    spec = cssd.CSSD_SPEC if spec is None else spec
    torch.manual_seed(int(seed))
    model = MODEL_BY_KIND[kind](
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        node_binding=spec.node_binding,
        edge_binding=spec.edge_binding,
        coding=spec.coding,
    )
    if model.CAPACITY_KIND != kind:
        raise RuntimeError(f"model kind mismatch: built {model.CAPACITY_KIND!r} for {kind!r}")
    return model


# ---------------------------------------------------------------------------
# warm start from the CAP-BASE soup checkpoint
# ---------------------------------------------------------------------------


def load_capacity_warm_state(
    model: CapacityModel, state: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Load the CAP-BASE soup into a candidate, checking bit-identity.

    Every tensor present in ``state`` that is not owned by the capacity module
    must be copied exactly; the capacity module keeps its frozen local
    initialization.  For candidate G the reader's original input columns are
    copied bit-identically and only the appended summary columns stay at their
    near-zero initialization.
    """
    model_state = model.state_dict()
    new_prefixes = tuple(model.CAPACITY_NEW_PREFIXES)
    report: dict[str, Any] = {
        "kind": model.CAPACITY_KIND,
        "shared_keys": 0,
        "reader_columns_copied": 0,
        "unexpected_keys": [],
        "bit_identical": True,
    }
    with torch.no_grad():
        for key, value in state.items():
            tensor = value.detach().to(torch.float32) if torch.is_tensor(value) else value
            if not torch.is_tensor(tensor):
                raise RuntimeError(f"non-tensor state entry {key!r}")
            owned = any(key == prefix or key.startswith(prefix + ".") for prefix in new_prefixes)
            if key in model_state and tuple(model_state[key].shape) == tuple(tensor.shape):
                if owned:
                    continue
                model_state[key].copy_(tensor)
                if not torch.equal(model_state[key], tensor):
                    report["bit_identical"] = False
                report["shared_keys"] += 1
                continue
            # candidate G: the reader's first block is wider than CAP-BASE.
            if key == "reader.net.0.weight" and key in model_state:
                target = model_state[key]
                if int(tensor.shape[1]) > int(target.shape[1]):
                    raise RuntimeError("warm-state reader is wider than the model reader")
                target[:, : int(tensor.shape[1])].copy_(tensor)
                if not torch.equal(target[:, : int(tensor.shape[1])], tensor):
                    report["bit_identical"] = False
                report["shared_keys"] += 1
                report["reader_columns_copied"] = int(tensor.shape[1])
                continue
            report["unexpected_keys"].append(key)
    if report["unexpected_keys"]:
        raise RuntimeError(f"unexpected warm-state keys: {report['unexpected_keys']}")
    if not report["bit_identical"]:
        raise RuntimeError("warm-state load is not bit-identical for CAP-BASE parameters")
    return report


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


def predictions_for(
    model: CapacityModel,
    loader: Any,
    device: torch.device,
    mask: audit.AuditMask | None = None,
) -> np.ndarray:
    cpu_only_guard(device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            chunks.append(model(batch, mask=mask).view(-1).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0,))


def evaluate_mae(
    model: CapacityModel, loader: Any, device: torch.device, mask: audit.AuditMask | None = None
) -> float:
    result = audit.evaluate_mask(model, loader, device, mask)
    return float(result["mae"])


def prediction_shift(
    model: CapacityModel, loader: Any, device: torch.device, baseline: np.ndarray
) -> dict[str, float]:
    predictions = predictions_for(model, loader, device, cssd.CSSD_MASK)
    if predictions.shape != baseline.shape:
        raise RuntimeError(f"prediction shape mismatch {predictions.shape} vs {baseline.shape}")
    delta = np.abs(predictions - baseline)
    return {
        "mean_abs_prediction_delta": float(delta.mean()),
        "max_abs_prediction_delta": float(delta.max()),
        "molecules": int(predictions.shape[0]),
    }


def new_module_gradient_norms(
    model: CapacityModel, batch: Any, device: torch.device
) -> dict[str, float]:
    """Frozen single-batch step-0 gradient norms of every capacity parameter."""
    cpu_only_guard(device)
    model.train(False)
    batch = batch.to(device)
    prediction, aux = model(batch, mask=cssd.CSSD_MASK, return_aux=True)
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(cm.H1_LAMBDA) * rec
    model.zero_grad(set_to_none=True)
    loss.backward()
    norms: dict[str, float] = {}
    for name in model.capacity_parameter_names():
        parameter = dict(model.named_parameters())[name]
        value = parameter.grad
        norms[name] = float(value.detach().norm()) if value is not None else 0.0
    model.zero_grad(set_to_none=True)
    model.train(True)
    return norms


# ---------------------------------------------------------------------------
# screening loop (frozen CAP-BASE training semantics, 40-epoch window soup)
# ---------------------------------------------------------------------------


def _soup_state(states: Mapping[int, Mapping[str, torch.Tensor]], members: Sequence[int]) -> dict[str, torch.Tensor]:
    members = list(members)
    return {
        key: torch.stack([states[epoch][key].float() for epoch in members], dim=0).mean(0)
        for key in states[members[0]]
    }


def train_screen(
    *,
    tag: str,
    model: CapacityModel,
    dictionary: np.ndarray,
    subspace: cssd.CommonSubspace,
    epochs: int,
    threads: int,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    seed: int = 0,
    soup_window: tuple[int, int] = SCREEN_SOUP_WINDOW,
    soup_k: int = SCREEN_SOUP_K,
    last_k: int = SCREEN_LAST10,
    log: bool = True,
    return_soup_state: bool = False,
) -> dict[str, Any]:
    """Warm-screen one arm with the frozen optimizer/loss/data-order protocol.

    Identical to ``cssd.train_cssd`` on the training path (fresh Adam, lr/wd
    from the frozen protocol, batch 128, clip 5.0, same shuffle seeds) except
    that the retained weight soup is the Top-``soup_k`` of the window
    ``soup_window`` instead of the whole curve, and no per-epoch checkpoints
    are written.  The model must already be warm-loaded; the caller has already
    seeded the global RNG exactly as ``train_cssd`` does.
    """
    import time

    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    device = audit.attach_cpu(int(threads))
    cpu_only_guard(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY)
    )
    loader = p1.make_env_loader(
        train_data, int(p2run.BATCH_SIZE), True, int(seed) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    eval_loader = p1.make_env_loader(
        valid_data, int(p2run.BATCH_SIZE), False, int(seed) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    mask = cssd.CSSD_MASK
    lam = float(cm.H1_LAMBDA)
    low, high = (int(soup_window[0]), int(soup_window[1]))
    window_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        task_sum = rec_sum = rec_term_sum = 0.0
        n_mol = n_nodes = n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            rec_term_sum += float(rec.detach())
            n_batches += 1
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float(
                (((phi - phi_hat) ** 2).sum(dim=1) / ((phi**2).sum(dim=1) + float(v0.EPS))).sum()
            )
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = audit._evaluate_model(model, eval_loader, device, mask)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "train_rec": train_rec,
                "train_rec_term": float(rec_term_sum / max(n_batches, 1)),
                "valid_mae": float(valid["mae"]),
                "d_norm": float(model.D.detach().norm()),
            }
        )
        if low <= epoch <= high:
            window_states[int(epoch)] = {
                key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()
            }
        if log and (epoch == 1 or epoch % 5 == 0 or epoch == int(epochs)):
            print(
                f"[{tag}] epoch={epoch:03d} train={train_mae:.6f} valid={float(valid['mae']):.6f}",
                flush=True,
            )
    wall = float(time.perf_counter() - started)
    if not window_states:
        raise RuntimeError(f"soup window {soup_window} produced no retained states")
    window_rows = [row for row in curve if low <= int(row["epoch"]) <= high]
    members = sorted(
        int(row["epoch"])
        for row in sorted(window_rows, key=lambda row: float(row["valid_mae"]))[: int(soup_k)]
    )
    soup_state = _soup_state(window_states, members)
    soup_model = build_capacity_model(dictionary, int(seed), subspace, model.CAPACITY_KIND)
    soup_model.load_state_dict(soup_state)
    soup_model = soup_model.to(device)
    soup_mae = evaluate_mae(soup_model, eval_loader, device, mask)
    last_rows = curve[-int(last_k) :]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "kind": model.CAPACITY_KIND,
        "device": "cpu",
        "threads": int(threads),
        "seed": int(seed),
        "epochs_budget": int(epochs),
        "epochs_run": int(len(curve)),
        "official_test_loaded": False,
        "soup_window": [low, high],
        "soup_members": members,
        "soup_member_valid_mae": [float(curve[epoch - 1]["valid_mae"]) for epoch in members],
        "soup_valid_mae": float(soup_mae),
        "best_valid_mae": float(min(row["valid_mae"] for row in curve)),
        "best_epoch": int(min(curve, key=lambda row: float(row["valid_mae"]))["epoch"]),
        "last10_mean_valid_mae": float(np.mean([row["valid_mae"] for row in last_rows])),
        "last_valid_mae": float(curve[-1]["valid_mae"]),
        "final_train_mae": float(curve[-1]["train_mae"]),
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve), 1)),
        "official_test_loaded_guard": False,
        "curve": curve,
    }
    if return_soup_state:
        payload["soup_state"] = soup_state
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# gate and winner selection
# ---------------------------------------------------------------------------


def screening_deltas(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Delta vs M0 for every non-control candidate."""
    control = rows["M0"]
    deltas: dict[str, dict[str, float]] = {}
    for kind in ("F", "R", "G"):
        row = rows[kind]
        deltas[kind] = {
            "delta_soup_vs_M0": float(row["soup_valid_mae"]) - float(control["soup_valid_mae"]),
            "delta_last10_vs_M0": float(row["last10_mean_valid_mae"])
            - float(control["last10_mean_valid_mae"]),
        }
    return deltas


def capacity_gate(row: Mapping[str, Any], delta: Mapping[str, float]) -> dict[str, Any]:
    soup_ok = float(delta["delta_soup_vs_M0"]) <= GATE_SOUP_DELTA
    last10_ok = float(delta["delta_last10_vs_M0"]) <= GATE_LAST10_DELTA
    stable = bool(np.isfinite(float(row["soup_valid_mae"]))) and bool(
        np.isfinite(float(row["last10_mean_valid_mae"]))
    )
    passed = bool(soup_ok and last10_ok and stable)
    verdict = "CAPACITY_SIGNAL" if passed else "NO_CAPACITY_SIGNAL"
    if passed and float(delta["delta_soup_vs_M0"]) <= STRONG_SOUP_DELTA:
        verdict = "STRONG_CAPACITY_SIGNAL"
    return {
        "soup_ok": soup_ok,
        "last10_ok": last10_ok,
        "stable": stable,
        "passed": passed,
        "verdict": verdict,
        "thresholds": {"soup_delta": GATE_SOUP_DELTA, "last10_delta": GATE_LAST10_DELTA},
    }


def select_winner(
    rows: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Frozen winner rule: gate survivors, then soup MAE, then F > R > G."""
    passing = [kind for kind in TIE_ORDER if bool(gate[kind]["passed"])]
    if not passing:
        return {
            "winner": None,
            "reason": "NO_CLEAR_CAPACITY_LOCALIZATION",
            "passing": [],
            "frozen_tie_order": list(TIE_ORDER),
        }
    ranked = sorted(passing, key=lambda kind: (float(rows[kind]["soup_valid_mae"]), TIE_ORDER.index(kind)))
    best = ranked[0]
    tied = [
        kind
        for kind in ranked
        if abs(float(rows[kind]["soup_valid_mae"]) - float(rows[best]["soup_valid_mae"])) <= TIE_TOLERANCE
    ]
    if len(tied) > 1:
        best = min(tied, key=lambda kind: TIE_ORDER.index(kind))
        reason = "TIE_BREAK_F_R_G"
    else:
        reason = "SCREENING_SOUP_MAE"
    return {
        "winner": best,
        "reason": reason,
        "passing": passing,
        "ranked": ranked,
        "tied_within_tolerance": tied,
        "frozen_tie_order": list(TIE_ORDER),
    }


def full_interpretation(soup_mae: float) -> dict[str, Any]:
    if soup_mae <= FULL_VERY_STRONG_AT:
        band = "MAJOR_CAPACITY_BOTTLENECK_IDENTIFIED"
    elif soup_mae <= FULL_STRONG_AT:
        band = "NEW_PERFORMANCE_BAND_SINGLE_SEED"
    elif soup_mae <= FULL_USEFUL_AT:
        band = "CAPACITY_DIRECTION_SUPPORTED_SINGLE_SEED"
    elif soup_mae <= FULL_NEUTRAL_ABOVE:
        band = "FULL_CAPACITY_GAIN_NOT_ESTABLISHED"
    else:
        band = "FULL_CAPACITY_GAIN_NOT_ESTABLISHED"
    return {
        "band": band,
        "soup_valid_mae": float(soup_mae),
        "thresholds": {
            "neutral_above": FULL_NEUTRAL_ABOVE,
            "useful_at": FULL_USEFUL_AT,
            "strong_at": FULL_STRONG_AT,
            "very_strong_at": FULL_VERY_STRONG_AT,
        },
        "delta_vs_cap_base": float(soup_mae) - CAP_BASE_SOUP_MAE,
        "delta_vs_final_clean_sparse": float(soup_mae) - FINAL_CLEAN_SPARSE_SOUP_MAE,
    }


# ---------------------------------------------------------------------------
# parameter budget
# ---------------------------------------------------------------------------


def parameter_budget(subspace: cssd.CommonSubspace | None = None) -> dict[str, Any]:
    """Exact construction-time parameter audit for CAP-BASE and all candidates."""
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    if subspace is None:
        subspace = cssd.CommonSubspace(
            components=np.eye(int(cssd.PHI_DIM), 1), rms=np.ones(1), kind="q1"
        )
    dictionary, dict_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    rows: dict[str, Any] = {}
    for kind in KINDS:
        model = build_capacity_model(dictionary, 0, subspace, kind)
        total = int(sum(parameter.numel() for parameter in model.parameters()))
        added = int(model.capacity_parameter_count())
        rows[kind] = {
            "kind": kind,
            "kind_description": CAPACITY_SPECS[kind].description,
            "total_params": total,
            "capacity_params": added,
            "capacity_parameter_names": model.capacity_parameter_names(),
        }
    base_total = int(rows["M0"]["total_params"])
    for kind in KINDS:
        rows[kind]["added_params_vs_cap_base"] = int(rows[kind]["total_params"]) - base_total
        rows[kind]["relative_increase"] = float(rows[kind]["total_params"]) / float(base_total) - 1.0
        rows[kind]["within_preferred"] = bool(
            BUDGET_PREFERRED[0] <= rows[kind]["total_params"] <= BUDGET_PREFERRED[1]
        )
        rows[kind]["within_hard_ceiling"] = bool(rows[kind]["total_params"] <= BUDGET_HARD_CEILING)
    candidate_added = [max(int(rows[kind]["added_params_vs_cap_base"]), 1) for kind in ("F", "R", "G")]
    ratio = float(max(candidate_added)) / float(min(candidate_added))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cap_base": CAP_BASE,
        "dictionary_sha256": dict_sha,
        "cap_base_params": base_total,
        "budget_preferred": list(BUDGET_PREFERRED),
        "budget_hard_ceiling": int(BUDGET_HARD_CEILING),
        "budget_ratio_max": float(BUDGET_RATIO_MAX),
        "rows": rows,
        "candidate_added_params": {kind: int(rows[kind]["added_params_vs_cap_base"]) for kind in ("F", "R", "G")},
        "added_params_ratio": ratio,
        "added_params_ratio_ok": bool(ratio <= BUDGET_RATIO_MAX),
        "official_test_loaded": False,
    }
    return payload


__all__ = [
    "PROTOCOL_VERSION",
    "CAP_BASE",
    "CAP_BASE_SOUP_MAE",
    "FINAL_CLEAN_SPARSE_SOUP_MAE",
    "KINDS",
    "FUSION_HEADS",
    "FUSION_HEAD_DIM",
    "FUSION_PROJ_INIT",
    "RELATION_BLOCKS",
    "RELATION_HIDDEN",
    "RELATION_RESIDUAL_INIT",
    "READOUT_SUMMARY_DIM",
    "READOUT_SUMMARY_EPS",
    "READOUT_READER_INIT",
    "SCREEN_EPOCHS",
    "SCREEN_SOUP_WINDOW",
    "SCREEN_SOUP_K",
    "SCREEN_LAST10",
    "GATE_SOUP_DELTA",
    "GATE_LAST10_DELTA",
    "STRONG_SOUP_DELTA",
    "TIE_TOLERANCE",
    "TIE_ORDER",
    "FULL_EPOCHS",
    "FULL_SEED",
    "FULL_NEUTRAL_ABOVE",
    "FULL_USEFUL_AT",
    "FULL_STRONG_AT",
    "FULL_VERY_STRONG_AT",
    "BUDGET_PREFERRED",
    "BUDGET_HARD_CEILING",
    "BUDGET_RATIO_MAX",
    "official_test_blocker",
    "cpu_only_guard",
    "CapacitySpec",
    "CAPACITY_SPECS",
    "capacity_spec",
    "preserved_global_rng",
    "CapacityModel",
    "FusionCapacityModel",
    "RelationCapacityModel",
    "ReadoutCapacityModel",
    "FiLMResidualBlock",
    "MODEL_BY_KIND",
    "build_capacity_model",
    "load_capacity_warm_state",
    "predictions_for",
    "evaluate_mae",
    "prediction_shift",
    "new_module_gradient_norms",
    "train_screen",
    "screening_deltas",
    "capacity_gate",
    "select_winner",
    "full_interpretation",
    "parameter_budget",
]
