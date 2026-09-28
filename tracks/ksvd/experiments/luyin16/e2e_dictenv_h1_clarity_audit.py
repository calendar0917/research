"""E2E-DictEnv-H1-Clarity-Audit — information-flow / mechanism audit core.

Round ``e2e_dictenv_h1_clarity_audit`` (Workstream Z).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_h1_clarity_audit_preregistration.md``.

This module contains **no training infrastructure of its own beyond the two
declared Phase-C protocols**, and it never modifies the frozen P1 / P2-ABS
implementation.  It provides:

* ``AuditMask`` — a frozen, declarative set of channel masks over the *already
  trained* H1 forward pass;
* ``AuditModel`` — a ``P2Model`` subclass whose masked forward is line-by-line
  the H1 forward with the declared blocks zeroed (bit-identical to
  ``P2Model.forward`` when the mask is absent, asserted by a focused CPU test);
* ``interventions()`` — the pre-registered frozen-intervention registry;
* provenance verification helpers for ``global_context`` / ``topology_features``
  / ``pair_relation`` coordinate groups;
* the static information-flow inventory (Phase A);
* CPU-only ``evaluate`` / ``baseline replay`` / ``frozen intervention`` helpers;
* ``train_adaptation`` (Tier 1 warm-start) and ``train_from_scratch`` (Tier 2
  matched CPU) — both follow the frozen H1 hyper-parameters imported from
  ``zinc_e2e_dictenv_p2_abs`` (never re-typed constants).

CPU only: every entry point asserts ``device.type == "cpu"`` and sets
``torch.set_num_threads`` explicitly.  ``official_test_loaded`` is ``False``
everywhere.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

PROTOCOL_VERSION = "e2e_dictenv_h1_clarity_audit"

#: the frozen P2-ABS winner.
H1_CONFIG = p2.P2Config("H1", "h1", 48, 32, 8, 0.25, 320, "sdb32")
H1_SOUP_MEMBERS = (293, 309, 313, 316, 317)
H1_HISTORICAL_SOUP_MAE = 0.12354862861608853
H1_LAMBDA = 33.95873017865987

# ---------------------------------------------------------------------------
# verified coordinate groups (see preregistration section 1)
# ---------------------------------------------------------------------------

#: ``e2e_dictenv_p1.build_anchor_raw`` layout (62-D, standardized with train stats).
ANCHOR_GROUPS: dict[str, tuple[int, int]] = {
    "root": (0, 28),  # root atom identity one-hot q_i
    "atom_mass": (28, 56),  # unconditioned patch atom-type mass
    "bond_mass": (56, 60),  # unconditioned patch bond-type mass
    "size": (60, 62),  # [log1p|V_i|, log1p|E_i|]
}
ANCHOR_DIM_EXPECTED = 62

#: ``zinc_long_range_proxy.global_feature_views(...)["global_all"]`` (62-D).
GLOBAL_GROUPS: dict[str, tuple[int, int]] = {
    "structure_short": (0, 15),  # degree / density / triangle / clustering summary
    "structure_long": (15, 30),  # shortest-path + eccentricity summary
    "atom_histogram": (30, 58),  # whole-molecule atom-type frequency
    "bond_histogram": (58, 62),  # whole-molecule bond-type frequency
}
GLOBAL_CHEMISTRY_GROUPS = ("atom_histogram", "bond_histogram")
GLOBAL_STRUCTURE_GROUPS = ("structure_short", "structure_long")
GLOBAL_DIM_EXPECTED = 62

#: ``zinc_topology_features.raw_vector(mode="hinge")`` (25-D, standardized).
TOPOLOGY_DIM_EXPECTED = 25

#: positions inside ``pair_relation[:, p1.P1_RELATION_INDICES]`` (15-D used slice).
RELATION_GROUPS: dict[str, tuple[int, int]] = {
    "distance": (0, 6),  # 5-bucket one-hot + log shortest-path distance
    "overlap": (6, 11),  # patch-overlap block (5)
    "boundary": (11, 14),  # patch-boundary block (3)
    "path_count": (14, 15),  # log number of shortest paths
}
RELATION_DIM_EXPECTED = 15
#: raw 23-D relation layout, for provenance checks only.
RELATION_RAW_LAYOUT = {
    "distance_one_hot": (0, 5),
    "log_distance": (5, 6),
    "overlap": (6, 11),
    "boundary": (11, 14),
    "path_bond_mean": (14, 18),  # dropped by P1_RELATION_INDICES
    "log_path_count": (18, 19),
    "adjacent_bond_type": (19, 23),  # dropped by P1_RELATION_INDICES
}

#: unary pool blocks (``v0.pool_moments``: sum | sum of squares | log1p(count)).
UNARY_BLOCKS: dict[str, tuple[int, int]] = {
    "first": (0, p2.ENV_DIM),
    "second": (p2.ENV_DIM, 2 * p2.ENV_DIM),
    "count": (2 * p2.ENV_DIM, 2 * p2.ENV_DIM + 1),
}
UNARY_DIM = 2 * p2.ENV_DIM + 1  # 97
#: per-distance-bucket pair pool blocks (``v0.pool_pair_moments``).
PAIR_BLOCKS: dict[str, tuple[int, int]] = {
    "first": (0, p2.PAIR_HIDDEN),
    "second": (p2.PAIR_HIDDEN, 2 * p2.PAIR_HIDDEN),
    "count": (2 * p2.PAIR_HIDDEN, 2 * p2.PAIR_HIDDEN + 1),
}
PAIR_BLOCK_DIM = 2 * p2.PAIR_HIDDEN + 1  # 33
RELATION_READOUT_DIM = p2.DISTANCE_BUCKETS * PAIR_BLOCK_DIM  # 165
READER_IN_DIM = UNARY_DIM + RELATION_READOUT_DIM + 32 + p2.TOPOLOGY_OUT  # 302


# ---------------------------------------------------------------------------
# audit mask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditMask:
    """Declarative channel mask over the trained H1 forward pass.

    Every field defaults to "no intervention"; ``AuditMask()`` is the identity.
    Zeroing is applied *inside* the forward pass at the declared coordinate, so
    parameter shapes never change.
    """

    anchor_zero_groups: tuple[str, ...] = ()
    global_zero_groups: tuple[str, ...] = ()
    topology_zero: bool = False
    node_binding_zero: bool = False
    edge_binding_zero: bool = False
    use_node_shuffle: bool = False
    use_edge_shuffle: bool = False
    coord_zero: bool = False
    relation_zero_groups: tuple[str, ...] = ()
    unary_zero_blocks: tuple[str, ...] = ()
    pair_zero_blocks: tuple[str, ...] = ()
    graph_hidden_zero: bool = False
    gate_off: bool = False
    pair_projection_zero: bool = False

    def __post_init__(self) -> None:
        for name in self.anchor_zero_groups:
            if name not in ANCHOR_GROUPS:
                raise ValueError(f"unknown anchor group {name!r}")
        for name in self.global_zero_groups:
            if name not in GLOBAL_GROUPS:
                raise ValueError(f"unknown global group {name!r}")
        for name in self.relation_zero_groups:
            if name not in RELATION_GROUPS:
                raise ValueError(f"unknown relation group {name!r}")
        for name in self.unary_zero_blocks:
            if name not in UNARY_BLOCKS:
                raise ValueError(f"unknown unary block {name!r}")
        for name in self.pair_zero_blocks:
            if name not in PAIR_BLOCKS:
                raise ValueError(f"unknown pair block {name!r}")
        if self.use_node_shuffle and self.use_edge_shuffle:
            pass  # combined shuffle is explicitly allowed (N5)

    def is_identity(self) -> bool:
        return (
            not self.anchor_zero_groups
            and not self.global_zero_groups
            and not self.topology_zero
            and not self.node_binding_zero
            and not self.edge_binding_zero
            and not self.use_node_shuffle
            and not self.use_edge_shuffle
            and not self.coord_zero
            and not self.relation_zero_groups
            and not self.unary_zero_blocks
            and not self.pair_zero_blocks
            and not self.graph_hidden_zero
            and not self.gate_off
            and not self.pair_projection_zero
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            key: (list(value) if isinstance(value, tuple) else value) for key, value in self.__dict__.items()
        }
        payload["is_identity"] = self.is_identity()
        return payload

    def signature(self) -> str:
        """Stable short signature used for artifact names / hashes."""
        items = []
        for key, value in sorted(self.__dict__.items()):
            if isinstance(value, tuple):
                if value:
                    items.append(f"{key}={'+'.join(value)}")
            elif value:
                items.append(f"{key}=1")
        return ";".join(items) if items else "identity"


def _is_audit_mask_layout_sane() -> None:
    assert p1.ANCHOR_DIM == ANCHOR_DIM_EXPECTED, p1.ANCHOR_DIM
    assert p1.ANCHOR_ROOT == slice(*ANCHOR_GROUPS["root"])
    assert p1.ANCHOR_ATOM_MASS == slice(*ANCHOR_GROUPS["atom_mass"])
    assert p1.ANCHOR_BOND_MASS == slice(*ANCHOR_GROUPS["bond_mass"])
    assert p1.ANCHOR_SIZE == slice(*ANCHOR_GROUPS["size"])
    assert p1.GLOBAL_WIDTH == GLOBAL_DIM_EXPECTED
    assert p1.TOPOLOGY_IN == TOPOLOGY_DIM_EXPECTED
    assert p1.RELATION_WIDTH == RELATION_DIM_EXPECTED
    assert p1.P1_RELATION_INDICES == tuple(range(14)) + (18,)


_is_audit_mask_layout_sane()


# ---------------------------------------------------------------------------
# masked pooling (mirrors v0.pool_moments / v0.pool_pair_moments exactly)
# ---------------------------------------------------------------------------


def pool_moments_masked(
    value: torch.Tensor,
    batch: torch.Tensor,
    n_graphs: int,
    zero_blocks: Sequence[str] = (),
    fill: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    counts = torch.bincount(batch, minlength=int(n_graphs)).to(value.dtype).unsqueeze(1)
    total = torch.zeros((int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype)
    squared = torch.zeros_like(total)
    total.index_add_(0, batch, value)
    squared.index_add_(0, batch, value * value)
    if "first" in zero_blocks:
        total = _block_fill_or_zero(total, "unary:first", fill, value.shape[1])
    if "second" in zero_blocks:
        squared = _block_fill_or_zero(squared, "unary:second", fill, value.shape[1])
    count_block = torch.log1p(counts)
    if "count" in zero_blocks:
        count_block = _block_fill_or_zero(count_block, "unary:count", fill, 1)
    return torch.cat([total, squared, count_block], dim=1)


def pool_pair_moments_masked(
    value: torch.Tensor,
    pair_batch: torch.Tensor,
    pair_bucket: torch.Tensor,
    n_graphs: int,
    zero_blocks: Sequence[str] = (),
    fill: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    for bucket in range(p2.DISTANCE_BUCKETS):
        mask = pair_bucket == int(bucket)
        current = value[mask]
        current_batch = pair_batch[mask]
        total = torch.zeros((int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype)
        squared = torch.zeros_like(total)
        counts = torch.zeros((int(n_graphs), 1), device=value.device, dtype=value.dtype)
        if current.numel():
            total.index_add_(0, current_batch, current)
            squared.index_add_(0, current_batch, current * current)
            counts.index_add_(
                0,
                current_batch,
                torch.ones((current_batch.shape[0], 1), device=value.device, dtype=value.dtype),
            )
        if "first" in zero_blocks:
            total = _bucket_block_fill_or_zero(total, "pair:first", fill, value.shape[1], bucket)
        if "second" in zero_blocks:
            squared = _bucket_block_fill_or_zero(squared, "pair:second", fill, value.shape[1], bucket)
        count_block = torch.log1p(counts)
        if "count" in zero_blocks:
            count_block = _bucket_block_fill_or_zero(count_block, "pair:count", fill, 1, bucket)
        blocks.append(torch.cat([total, squared, count_block], dim=1))
    return torch.cat(blocks, dim=1)


def _bucket_block_fill_or_zero(
    block: torch.Tensor, key: str, fill: Mapping[str, torch.Tensor] | None, width: int, bucket: int
) -> torch.Tensor:
    if fill is not None and key in fill:
        table = fill[key]
        if table.dim() != 2 or int(table.shape[0]) != p2.DISTANCE_BUCKETS:
            raise ValueError(f"fill[{key!r}] must be [{p2.DISTANCE_BUCKETS}, W], got {tuple(table.shape)}")
        if int(table.shape[1]) != int(width):
            raise ValueError(f"fill[{key!r}] width {int(table.shape[1])} != {width}")
        replacement = table[int(bucket)].to(device=block.device, dtype=block.dtype).reshape(1, -1)
        return replacement.expand(block.shape[0], -1).contiguous()
    return torch.zeros_like(block)


def _replace_grouped_columns(
    value: torch.Tensor,
    groups: Mapping[str, tuple[int, int]],
    names: Sequence[str],
    fill: Mapping[str, torch.Tensor] | None = None,
    prefix: str = "",
) -> torch.Tensor:
    """Return ``value`` with the named column groups zeroed or mean-filled.

    ``fill`` (when given) maps ``f"{prefix}{name}"`` to a per-coordinate vector;
    masked blocks present in ``fill`` are replaced by that vector instead of by
    zero, which keeps the intervention closer to the data distribution.
    """
    out = value.clone()
    for name in names:
        low, high = groups[name]
        width = int(high) - int(low)
        key = f"{prefix}{name}"
        if fill is not None and key in fill:
            replacement = fill[key].to(device=out.device, dtype=out.dtype).reshape(-1)
            if int(replacement.numel()) != width:
                raise ValueError(f"fill[{key!r}] width {int(replacement.numel())} != {width}")
            out[:, int(low) : int(high)] = replacement.reshape(1, width)
        else:
            out[:, int(low) : int(high)] = 0.0
    return out


def _zero_grouped_columns(
    value: torch.Tensor, groups: Mapping[str, tuple[int, int]], names: Sequence[str]
) -> torch.Tensor:
    """Backwards-compatible alias for zero-mode grouped replacement."""
    return _replace_grouped_columns(value, groups, names, None, "")


def _block_fill_or_zero(
    block: torch.Tensor, key: str, fill: Mapping[str, torch.Tensor] | None, width: int
) -> torch.Tensor:
    if fill is not None and key in fill:
        replacement = fill[key].to(device=block.device, dtype=block.dtype).reshape(-1)
        if int(replacement.numel()) != int(width):
            raise ValueError(f"fill[{key!r}] width {int(replacement.numel())} != {width}")
        return replacement.reshape(1, -1).expand(block.shape[0], -1).contiguous()
    return torch.zeros_like(block)


#: readout blocks addressable by the cross-graph row-shuffle probe.
READOUT_SHUFFLE_BLOCKS = (
    "unary_first",
    "unary_second",
    "unary_count",
    "pair_first",
    "pair_second",
    "pair_count",
)


def apply_readout_permutation(
    unary: torch.Tensor,
    relation_readout: torch.Tensor,
    blocks: Sequence[str],
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-graph row shuffle of pooled readout blocks (distribution-preserving).

    Each selected block's rows are replaced by the same block of another graph
    in the batch, so the block marginal is untouched while its graph pairing is
    destroyed.  This is the cleanest probe for "does this graph's own pooled
    statistic matter?".
    """
    unary_out = unary
    readout_out = relation_readout
    if any(name.startswith("unary") for name in blocks):
        unary_out = unary.clone()
        for name in blocks:
            if name == "unary_first":
                low, high = UNARY_BLOCKS["first"]
                unary_out[:, low:high] = unary[permutation][:, low:high]
            elif name == "unary_second":
                low, high = UNARY_BLOCKS["second"]
                unary_out[:, low:high] = unary[permutation][:, low:high]
            elif name == "unary_count":
                low, high = UNARY_BLOCKS["count"]
                unary_out[:, low:high] = unary[permutation][:, low:high]
    if any(name.startswith("pair") for name in blocks):
        n_graphs = int(relation_readout.shape[0])
        view = relation_readout.reshape(n_graphs, p2.DISTANCE_BUCKETS, PAIR_BLOCK_DIM)
        permuted = view[permutation]
        readout_out = relation_readout.clone().reshape(n_graphs, p2.DISTANCE_BUCKETS, PAIR_BLOCK_DIM)
        for name in blocks:
            if name.startswith("pair_"):
                low, high = PAIR_BLOCKS[name[len("pair_") :]]
                readout_out[:, :, low:high] = permuted[:, :, low:high]
        readout_out = readout_out.reshape(n_graphs, p2.DISTANCE_BUCKETS * PAIR_BLOCK_DIM)
    return unary_out, readout_out


# ---------------------------------------------------------------------------
# masked H1 model
# ---------------------------------------------------------------------------


class AuditModel(p2.P2Model):
    """H1 architecture + a declarative input-channel mask.

    ``forward(data, mask=None)`` delegates to ``P2Model.forward`` unchanged.
    ``forward(data, mask=AuditMask(...))`` executes exactly the same operations
    in exactly the same order, with the masked blocks zeroed at their declared
    coordinates.  With ``AuditMask()`` the two paths are bit-identical (asserted
    by a focused CPU test).
    """

    # -- local environment ---------------------------------------------------

    def environments_masked(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: AuditMask,
        fill: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        n = int(coord.shape[0])
        anchor = data.anchor.to(coord.dtype)
        if int(anchor.shape[1]) != ANCHOR_DIM_EXPECTED:
            raise RuntimeError(f"anchor width {int(anchor.shape[1])} != {ANCHOR_DIM_EXPECTED}")
        if mask.anchor_zero_groups:
            anchor = _replace_grouped_columns(anchor, ANCHOR_GROUPS, mask.anchor_zero_groups, fill, "anchor:")

        q = torch.nn.functional.one_hot(data.dict_atom, num_classes=p2.ATOM_CATEGORIES).to(coord.dtype)
        occ_coord_node = data.env_occ_node
        if mask.use_node_shuffle:
            occ_coord_node = getattr(data, "env_occ_coord_node", None)
            if occ_coord_node is None:
                raise RuntimeError("node-shuffle intervention requires data.env_occ_coord_node")
        c = coord[occ_coord_node.to(coord.device)]
        qc = q[data.env_occ_node.to(coord.device)]
        u = (c @ self.W_A_S) * (qc @ self.W_A_C) / math.sqrt(float(p2.D_A))
        if mask.node_binding_zero:
            u = torch.zeros_like(u)
        flat = torch.zeros((n * p2.N_SHELLS, p2.D_A), device=u.device, dtype=u.dtype)
        flat.index_add_(
            0,
            data.env_occ_root.to(coord.device) * p2.N_SHELLS + data.env_occ_shell.to(coord.device),
            u,
        )
        node_slots = flat.view(n, p2.N_SHELLS, p2.D_A)

        d_e = int(self.config.d_e)
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
        b = torch.nn.functional.one_hot(data.env_bond_type, num_classes=p2.BOND_CATEGORIES).to(coord.dtype)
        ue = (g @ self.W_E_S) * (b @ self.W_E_C) / math.sqrt(float(d_e))
        flat_e = torch.zeros((n * p2.SHELLPAIR_CLASSES, d_e), device=ue.device, dtype=ue.dtype)
        flat_e.index_add_(
            0,
            data.env_bond_root.to(coord.device) * p2.SHELLPAIR_CLASSES
            + data.env_bond_shellpair.to(coord.device),
            ue,
        )
        edge_slots = flat_e.view(n, p2.SHELLPAIR_CLASSES, d_e)

        if self.env_mlp is not None:
            z = torch.cat([anchor, node_slots.reshape(n, -1), edge_slots.reshape(n, -1)], dim=1)
            return self.env_mlp(z)
        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        anchor_out = self.anchor_encoder(anchor)
        fused = torch.cat([anchor_out, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1)
        return self.fusion(fused)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        data: Any,
        *,
        mask: AuditMask | None = None,
        fill: Mapping[str, torch.Tensor] | None = None,
        readout_perm: tuple[Sequence[str], torch.Tensor] | None = None,
        return_aux: bool = False,
        **_kwargs: Any,
    ):
        if mask is None:
            return super().forward(data, return_aux=return_aux)

        coord = self.code(data.dict_phi)
        if mask.coord_zero:
            coord = torch.zeros_like(coord)
        E = self.environments_masked(coord, data, mask, fill)

        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = pool_moments_masked(E, batch, n_graphs, mask.unary_zero_blocks, fill)

        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        if mask.pair_projection_zero:
            u = torch.zeros_like(u)
        left = u[source]
        right = u[target]
        relation_input = data.pair_relation[:, list(p1.P1_RELATION_INDICES)]
        if mask.relation_zero_groups:
            relation_input = _replace_grouped_columns(
                relation_input, RELATION_GROUPS, mask.relation_zero_groups, fill, "relation:"
            )
        relation = self.relation_encoder(relation_input)
        product = left * right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        if mask.gate_off:
            gate = torch.ones_like(gate)
        pair_input = torch.cat([left + right, torch.abs(left - right), product * gate, relation], dim=1)
        pair_value = self.pair_encoder(pair_input)
        pair_batch = batch[source]
        relation_readout = pool_pair_moments_masked(
            pair_value, pair_batch, data.pair_bucket, n_graphs, mask.pair_zero_blocks, fill
        )
        if readout_perm is not None:
            blocks, permutation = readout_perm
            unary, relation_readout = apply_readout_permutation(unary, relation_readout, blocks, permutation)

        global_input = data.global_context
        if mask.global_zero_groups:
            global_input = _replace_grouped_columns(
                global_input, GLOBAL_GROUPS, mask.global_zero_groups, fill, "global:"
            )
        graph_hidden = self.global_encoder(global_input)
        if mask.graph_hidden_zero:
            graph_hidden = torch.zeros_like(graph_hidden)
        topology_input = data.topology_features
        if mask.topology_zero:
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
        unified = torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)
        prediction = self.reader(unified).view(-1)
        if return_aux:
            return prediction, {
                "E": E,
                "coord": coord,
                "phi": data.dict_phi,
                "unary": unary,
                "relation_readout": relation_readout,
            }
        return prediction


def build_audit_model(dictionary: np.ndarray, seed: int = 0) -> AuditModel:
    """Build the H1 model as an ``AuditModel`` (identical parameter init)."""
    torch.manual_seed(int(seed))
    return AuditModel(H1_CONFIG, dictionary)


def load_h1_soup_state(path: Any) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "D" not in state:
        raise RuntimeError(f"{path}: unexpected H1 soup checkpoint payload")
    return {
        key: value.detach().cpu().float() if torch.is_tensor(value) else value for key, value in state.items()
    }


# ---------------------------------------------------------------------------
# interventions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Intervention:
    name: str
    category: str
    mask: AuditMask = field(default_factory=AuditMask)
    notes: str = ""
    shuffle_kind: str | None = None  # "node" | "edge" | "both"
    seeds: tuple[int, ...] = ()
    #: when True the masked input blocks are replaced by their per-coordinate
    #: official-valid mean (distribution-aware probe) instead of by zero.
    use_fill: bool = False
    #: cross-molecule row shuffle of a graph-level block: "global" | "topology" | "both".
    graph_shuffle: str | None = None
    #: cross-graph row shuffle of pooled readout blocks (``READOUT_SHUFFLE_BLOCKS``).
    readout_shuffle: tuple[str, ...] = ()
    #: cross-pair row shuffle of relation groups (``RELATION_SHUFFLE_KINDS``).
    relation_shuffle: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "mask": self.mask.as_dict(),
            "notes": self.notes,
            "shuffle_kind": self.shuffle_kind,
            "seeds": list(self.seeds),
            "use_fill": bool(self.use_fill),
            "graph_shuffle": self.graph_shuffle,
            "readout_shuffle": list(self.readout_shuffle),
            "relation_shuffle": self.relation_shuffle,
        }


#: P1 mechanism-audit shuffle seeds (kept identical for comparability).
SHUFFLE_SEEDS = (101, 202, 303, 404, 505)


def interventions(include_extended: bool = True) -> list[Intervention]:
    """The pre-registered frozen-intervention registry (preregistration §3)."""
    rows: list[Intervention] = [
        # -- anchor ---------------------------------------------------------
        Intervention("A0", "anchor", AuditMask(), "control: audit path, identity mask"),
        Intervention(
            "A1",
            "anchor",
            AuditMask(anchor_zero_groups=("root", "atom_mass", "bond_mass", "size")),
            "zero full anchor62",
        ),
        Intervention(
            "A2",
            "anchor",
            AuditMask(anchor_zero_groups=("atom_mass", "bond_mass")),
            "keep root atom + size only",
        ),
        Intervention("A3", "anchor", AuditMask(anchor_zero_groups=("root",)), "remove root atom identity"),
        Intervention("A4", "anchor", AuditMask(anchor_zero_groups=("atom_mass",)), "remove patch atom mass"),
        Intervention("A5", "anchor", AuditMask(anchor_zero_groups=("bond_mass",)), "remove patch bond mass"),
        Intervention("A6", "anchor", AuditMask(anchor_zero_groups=("size",)), "remove size"),
        # -- graph-level global context -------------------------------------
        Intervention("G0", "global", AuditMask(), "control: identity mask"),
        Intervention(
            "G1",
            "global",
            AuditMask(global_zero_groups=GLOBAL_CHEMISTRY_GROUPS),
            "zero graph-level chemistry marginal (atom+bond histogram)",
        ),
        Intervention(
            "G2",
            "global",
            AuditMask(global_zero_groups=GLOBAL_STRUCTURE_GROUPS),
            "zero graph-level topology part (short+long)",
        ),
        Intervention(
            "G3",
            "global",
            AuditMask(global_zero_groups=GLOBAL_STRUCTURE_GROUPS + GLOBAL_CHEMISTRY_GROUPS),
            "zero full global62",
        ),
        # -- topology25 bypass ----------------------------------------------
        Intervention("T0", "topology25", AuditMask(), "control: identity mask"),
        Intervention("T1", "topology25", AuditMask(topology_zero=True), "zero topology25"),
        # -- node / edge binding --------------------------------------------
        Intervention("N0", "binding", AuditMask(), "control: identity mask"),
        Intervention("N1", "binding", AuditMask(node_binding_zero=True), "zero node-binding slot input"),
        Intervention(
            "N2",
            "binding",
            AuditMask(edge_binding_zero=True),
            "zero edge-binding role input (bond type kept)",
        ),
        Intervention(
            "N3",
            "binding",
            AuditMask(use_node_shuffle=True),
            "node assignment shuffle",
            shuffle_kind="node",
            seeds=SHUFFLE_SEEDS,
        ),
        Intervention(
            "N4",
            "binding",
            AuditMask(use_edge_shuffle=True),
            "edge assignment shuffle",
            shuffle_kind="edge",
            seeds=SHUFFLE_SEEDS,
        ),
        Intervention(
            "N5",
            "binding",
            AuditMask(use_node_shuffle=True, use_edge_shuffle=True),
            "node + edge assignment shuffle",
            shuffle_kind="both",
            seeds=tuple((s, 1000 + s) for s in SHUFFLE_SEEDS),
        ),
        Intervention("N6", "binding", AuditMask(coord_zero=True), "zero dictionary coordinate alpha"),
        # -- pair relation ---------------------------------------------------
        Intervention("R0", "relation", AuditMask(), "control: identity mask"),
        Intervention(
            "R1",
            "relation",
            AuditMask(relation_zero_groups=("overlap", "boundary", "path_count")),
            "distance-only relation",
        ),
        Intervention("R2", "relation", AuditMask(relation_zero_groups=("overlap",)), "remove overlap block"),
        Intervention(
            "R3", "relation", AuditMask(relation_zero_groups=("boundary",)), "remove boundary block"
        ),
        Intervention(
            "R4", "relation", AuditMask(relation_zero_groups=("path_count",)), "remove log path count"
        ),
        Intervention(
            "R5",
            "relation",
            AuditMask(relation_zero_groups=("distance", "overlap", "boundary", "path_count")),
            "no explicit relation",
        ),
        # -- statistical readout ---------------------------------------------
        Intervention("P0", "readout", AuditMask(), "control: identity mask"),
        Intervention("P1", "readout", AuditMask(unary_zero_blocks=("second",)), "zero unary second moment"),
        Intervention("P2", "readout", AuditMask(pair_zero_blocks=("second",)), "zero pair second moments"),
        Intervention(
            "P3",
            "readout",
            AuditMask(unary_zero_blocks=("second",), pair_zero_blocks=("second",)),
            "zero unary + pair second moments",
        ),
        Intervention(
            "P4",
            "readout",
            AuditMask(unary_zero_blocks=("count",), pair_zero_blocks=("count",)),
            "zero unary + pair count terms",
        ),
    ]
    if not include_extended:
        return rows
    rows.extend(
        [
            Intervention(
                "EA1",
                "anchor_ext",
                AuditMask(anchor_zero_groups=("root", "atom_mass", "bond_mass")),
                "keep size only",
            ),
            Intervention(
                "EA2",
                "anchor_ext",
                AuditMask(anchor_zero_groups=("root", "atom_mass", "size")),
                "keep patch bond mass only",
            ),
            Intervention(
                "EG1",
                "global_ext",
                AuditMask(global_zero_groups=("atom_histogram",)),
                "zero atom histogram only",
            ),
            Intervention(
                "EG2",
                "global_ext",
                AuditMask(global_zero_groups=("bond_histogram",)),
                "zero bond histogram only",
            ),
            Intervention(
                "ER1",
                "relation_ext",
                AuditMask(relation_zero_groups=("overlap", "boundary", "path_count")),
                "distance-only relation (duplicate of R1 for extended table)",
            ),
            Intervention(
                "EP1", "readout_ext", AuditMask(unary_zero_blocks=("first",)), "zero unary first moment only"
            ),
            Intervention(
                "EP2",
                "readout_ext",
                AuditMask(unary_zero_blocks=("first", "second", "count")),
                "zero full unary pool",
            ),
            Intervention(
                "EP3",
                "readout_ext",
                AuditMask(pair_zero_blocks=("first", "second", "count")),
                "zero full relation readout",
            ),
            Intervention(
                "EB1", "backend_ext", AuditMask(graph_hidden_zero=True), "zero global encoder output"
            ),
            Intervention("EB2", "backend_ext", AuditMask(gate_off=True), "distance gate off (gate=1)"),
            Intervention(
                "EB3", "backend_ext", AuditMask(pair_projection_zero=True), "zero pair-projection output"
            ),
        ]
    )
    return rows


# ---------------------------------------------------------------------------
# data preparation for assignment shuffles (P1 semantics, unchanged)
# ---------------------------------------------------------------------------


def prepare_shuffles(data_list: Sequence[Any], kind: str, seed: Any) -> None:
    """Attach P1-semantics shuffled assignment fields (preserves multisets)."""
    if kind in ("node", "both"):
        node_seed = int(seed[0]) if isinstance(seed, tuple) else int(seed)
        for data in data_list:
            data.env_occ_coord_node = p1.shuffled_occ_node_for_molecule(
                data.env_occ_node, data.env_occ_root, data.env_occ_shell, node_seed
            )
    if kind in ("edge", "both"):
        edge_seed = int(seed[1]) if isinstance(seed, tuple) else int(seed)
        for data in data_list:
            su, sv = p1.shuffled_bond_endpoints_for_molecule(
                data.env_bond_root, data.env_bond_shellpair, data.env_bond_u, data.env_bond_v, edge_seed
            )
            data.env_bond_u_shuffled = su
            data.env_bond_v_shuffled = sv


def clear_shuffles(data_list: Sequence[Any]) -> None:
    for data in data_list:
        data.env_occ_coord_node = None
        data.env_bond_u_shuffled = None
        data.env_bond_v_shuffled = None


#: seeds used for the cross-molecule graph-level row shuffle.
GRAPH_SHUFFLE_SEEDS = (4242, 5150, 6262)

#: cross-molecule row-shuffle targets: ``kind -> ((attribute, low, high), ...)``.
GRAPH_SHUFFLE_KINDS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "global_chemistry": (("global_context", 30, 62),),
    "global_structure": (("global_context", 0, 30),),
    "global_all": (("global_context", 0, 62),),
    "topology": (("topology_features", 0, 25),),
    "global_topology": (("global_context", 0, 62), ("topology_features", 0, 25)),
}


def permute_graph_rows(data_list: Sequence[Any], kind: str, seed: int):
    """Cross-molecule row shuffle of graph-level blocks (distribution-preserving).

    The marginal distribution of each shuffled block is untouched; only the
    molecule it belongs to changes.  This is the cleanest frozen probe for
    "is this graph's own value used?" on graph-level variables.  Returns a
    ``restore()`` callable that puts the original rows back.
    """
    if kind not in GRAPH_SHUFFLE_KINDS:
        raise ValueError(f"unknown graph-shuffle kind {kind!r}")
    generator = torch.Generator().manual_seed(int(seed))
    n = len(data_list)
    saved: list[tuple[Any, str, torch.Tensor]] = []
    if n == 0:
        return lambda: None
    for attribute, low, high in GRAPH_SHUFFLE_KINDS[kind]:
        stacked = torch.stack([getattr(data, attribute).reshape(-1) for data in data_list])
        permutation = torch.randperm(n, generator=generator)
        shuffled = stacked[permutation]
        for index, data in enumerate(data_list):
            original = getattr(data, attribute)
            replacement = original.clone()
            replacement[..., int(low) : int(high)] = shuffled[index, int(low) : int(high)]
            saved.append((data, attribute, original))
            setattr(data, attribute, replacement)

    def restore() -> None:
        for data, attribute, original in saved:
            setattr(data, attribute, original)

    return restore


def build_fill_policy(
    model: "AuditModel", loader: Any, device: torch.device, mask: AuditMask | None = None
) -> dict[str, torch.Tensor]:
    """Per-coordinate official-valid means of every mappable input/output block.

    Computed from an identity-mask pass, so the means are the values the trained
    H1 forward actually sees.  Used as the distribution-aware replacement for
    masked blocks (``Intervention.use_fill``).
    """
    mask = AuditMask() if mask is None else mask
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    n_nodes = 0
    n_graphs = 0
    n_pairs = 0

    def _accumulate(key: str, value: torch.Tensor) -> None:
        value = value.detach().double()
        if key in sums:
            sums[key] = sums[key] + value
        else:
            sums[key] = value.clone()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _prediction, aux = model(batch, mask=mask, return_aux=True)
            anchor = batch.anchor
            for name, (low, high) in ANCHOR_GROUPS.items():
                _accumulate(f"anchor:{name}", anchor[:, low:high].sum(dim=0))
            n_nodes += int(anchor.shape[0])
            for name, (low, high) in GLOBAL_GROUPS.items():
                _accumulate(f"global:{name}", batch.global_context[:, low:high].sum(dim=0))
            _accumulate("topology", batch.topology_features.sum(dim=0))
            n_graphs += int(batch.global_context.shape[0])
            relation = batch.pair_relation[:, list(p1.P1_RELATION_INDICES)]
            for name, (low, high) in RELATION_GROUPS.items():
                _accumulate(f"relation:{name}", relation[:, low:high].sum(dim=0))
            n_pairs += int(relation.shape[0])
            unary = aux["unary"]
            _accumulate(
                "unary:first", unary[:, UNARY_BLOCKS["first"][0] : UNARY_BLOCKS["first"][1]].sum(dim=0)
            )
            _accumulate(
                "unary:second", unary[:, UNARY_BLOCKS["second"][0] : UNARY_BLOCKS["second"][1]].sum(dim=0)
            )
            _accumulate(
                "unary:count", unary[:, UNARY_BLOCKS["count"][0] : UNARY_BLOCKS["count"][1]].sum(dim=0)
            )
            readout = aux["relation_readout"].reshape(
                int(batch.global_context.shape[0]), p2.DISTANCE_BUCKETS, PAIR_BLOCK_DIM
            )
            for block in ("first", "second", "count"):
                low, high = PAIR_BLOCKS[block]
                _accumulate(f"pair:{block}", readout[:, :, low:high].sum(dim=0))

    policy: dict[str, torch.Tensor] = {}
    for key, value in sums.items():
        if key.startswith("anchor:"):
            denominator = max(n_nodes, 1)
        elif key.startswith("relation:"):
            denominator = max(n_pairs, 1)
        else:
            denominator = max(n_graphs, 1)
        policy[key] = (value / float(denominator)).float()
    policy["pair:first"] = policy["pair:first"].reshape(p2.DISTANCE_BUCKETS, p2.PAIR_HIDDEN)
    policy["pair:second"] = policy["pair:second"].reshape(p2.DISTANCE_BUCKETS, p2.PAIR_HIDDEN)
    policy["pair:count"] = policy["pair:count"].reshape(p2.DISTANCE_BUCKETS, 1)
    return policy


# ---------------------------------------------------------------------------
# CPU evaluation
# ---------------------------------------------------------------------------


def attach_cpu(threads: int) -> torch.device:
    device = torch.device("cpu")
    torch.set_num_threads(int(threads))
    return device


def evaluate_mask(
    model: AuditModel,
    loader: Any,
    device: torch.device,
    mask: AuditMask | None = None,
    fill: Mapping[str, torch.Tensor] | None = None,
    readout_shuffle: tuple[Sequence[str], int] | None = None,
) -> dict[str, Any]:
    """Prediction-only CPU evaluation with a mask (no reconstruction readout).

    ``readout_shuffle=(blocks, seed)`` additionally cross-graph row-shuffles the
    named pooled readout blocks (``READOUT_SHUFFLE_BLOCKS``).
    """
    if device.type != "cpu":
        raise RuntimeError(f"clarity audit is CPU-only, got device={device}")
    model.eval()
    generator = torch.Generator().manual_seed(int(readout_shuffle[1])) if readout_shuffle else None
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            readout_perm = None
            if readout_shuffle is not None:
                n_graphs = int(batch.y.numel())
                readout_perm = (tuple(readout_shuffle[0]), torch.randperm(n_graphs, generator=generator))
            prediction = model(batch, mask=mask, fill=fill, readout_perm=readout_perm)
            predictions.append(prediction.view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    target = np.concatenate(targets).astype(np.float64)
    pred = np.concatenate(predictions).astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(target - pred))),
        "n_molecules": int(target.shape[0]),
        "predictions": pred,
        "targets": target,
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
    }


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def run_intervention(
    model: AuditModel,
    loader: Any,
    data_list: Sequence[Any],
    device: torch.device,
    intervention: Intervention,
    baseline_predictions: np.ndarray | None = None,
    fill_policy: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Run one frozen intervention (averaging over shuffle seeds when relevant)."""
    started = time.perf_counter()
    multiple = bool(
        intervention.shuffle_kind
        or intervention.graph_shuffle
        or intervention.readout_shuffle
        or intervention.relation_shuffle
    )
    seeds: Iterable[Any] = intervention.seeds if multiple else (None,)
    fill = fill_policy if intervention.use_fill else None
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        restore = None
        relation_restore = None
        try:
            if intervention.shuffle_kind:
                prepare_shuffles(data_list, intervention.shuffle_kind, seed)
            if intervention.graph_shuffle:
                restore = permute_graph_rows(data_list, intervention.graph_shuffle, int(seed))
            if intervention.relation_shuffle:
                relation_restore = permute_pair_rows(data_list, intervention.relation_shuffle, int(seed))
            readout_shuffle = (
                (intervention.readout_shuffle, int(seed)) if intervention.readout_shuffle else None
            )
            result = evaluate_mask(model, loader, device, intervention.mask, fill, readout_shuffle)
        finally:
            if restore is not None:
                restore()
            if relation_restore is not None:
                relation_restore()
            if intervention.shuffle_kind:
                clear_shuffles(data_list)
        row: dict[str, Any] = {
            "seed": None if seed is None else (list(seed) if isinstance(seed, tuple) else int(seed)),
            "mae": float(result["mae"]),
            "prediction_mean": float(result["prediction_mean"]),
            "prediction_std": float(result["prediction_std"]),
        }
        if baseline_predictions is not None:
            row["mean_abs_prediction_delta"] = float(
                np.mean(np.abs(result["predictions"] - baseline_predictions))
            )
            row["max_abs_prediction_delta"] = float(
                np.max(np.abs(result["predictions"] - baseline_predictions))
            )
            row["prediction_correlation"] = _corr(result["predictions"], baseline_predictions)
        rows.append(row)
    runtime = float(time.perf_counter() - started)
    payload = {
        "intervention": intervention.name,
        "category": intervention.category,
        "mask": intervention.mask.as_dict(),
        "notes": intervention.notes,
        "shuffle_kind": intervention.shuffle_kind,
        "graph_shuffle": intervention.graph_shuffle,
        "readout_shuffle": list(intervention.readout_shuffle),
        "relation_shuffle": intervention.relation_shuffle,
        "use_fill": bool(intervention.use_fill),
        "seeds": [row["seed"] for row in rows],
        "intervention_mae": float(np.mean([row["mae"] for row in rows])),
        "intervention_mae_std": float(np.std([row["mae"] for row in rows])),
        "intervention_mae_rows": [float(row["mae"]) for row in rows],
        "prediction_mean": float(np.mean([row["prediction_mean"] for row in rows])),
        "prediction_std": float(np.mean([row["prediction_std"] for row in rows])),
        "runtime_seconds": runtime,
    }
    for key in ("mean_abs_prediction_delta", "max_abs_prediction_delta", "prediction_correlation"):
        if key in rows[0]:
            payload[key] = float(np.mean([row[key] for row in rows]))
    return payload


def summarise_interventions(rows: Sequence[Mapping[str, Any]], baseline_mae: float) -> list[dict[str, Any]]:
    """Attach ``baseline_mae`` / ``delta_mae`` / load-bearing class to each row."""
    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["baseline_mae"] = float(baseline_mae)
        enriched["delta_mae"] = float(row["intervention_mae"] - baseline_mae)
        delta = float(enriched["delta_mae"])
        absolute = abs(float(row.get("mean_abs_prediction_delta", float("nan"))))
        if delta >= 0.02 or absolute >= 0.05:
            enriched["load_bearing"] = "strongly_load_bearing"
        elif delta >= 0.005 or absolute >= 0.01:
            enriched["load_bearing"] = "moderately_used"
        elif delta >= 0.002 or absolute >= 0.003:
            enriched["load_bearing"] = "weakly_used"
        else:
            enriched["load_bearing"] = "weak_or_dormant"
        out.append(enriched)
    return out


# ---------------------------------------------------------------------------
# provenance verification (Phase A)
# ---------------------------------------------------------------------------


def verify_global_context_provenance(n_molecules: int = 64) -> dict[str, Any]:
    """Rank-correlate the stored standardized ``global_context`` with raw features.

    The stored cache holds ``standardizer.transform(raw)``; a per-coordinate
    affine map preserves Pearson correlation up to sign, so ``|corr| == 1`` for
    every coordinate proves the stored slice ``k`` really is raw feature ``k``.
    """
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
    from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr

    valid = p1run.load_split("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(p1run.ZINC_ROOT, "val"))[: int(n_molecules)]
    views = zlr.global_feature_views(raw)
    raw_matrix = np.asarray(views["global_all"], dtype=np.float64)
    stored = np.stack([data.global_context.numpy().reshape(-1) for data in valid]).astype(np.float64)
    if stored.shape != raw_matrix.shape:
        raise RuntimeError(f"global_context shape mismatch {stored.shape} vs {raw_matrix.shape}")
    correlations = []
    n_constant = 0
    for column in range(stored.shape[1]):
        left = stored[:, column]
        right = raw_matrix[:, column]
        if float(left.std()) == 0.0 and float(right.std()) == 0.0:
            # coordinate is constant on this molecule subset: no evidence either
            # way, but also no contradiction.
            n_constant += 1
            correlations.append(float("nan"))
            continue
        if float(left.std()) == 0.0 or float(right.std()) == 0.0:
            correlations.append(0.0)  # one side varies, the other does not: contradiction
            continue
        correlations.append(float(np.corrcoef(left, right)[0, 1]))
    correlations = np.asarray(correlations)
    informative = correlations[~np.isnan(correlations)]
    groups = {
        name: {
            "slice": list(span),
            "min_abs_corr": float(np.nanmin(np.abs(correlations[span[0] : span[1]])))
            if not np.all(np.isnan(correlations[span[0] : span[1]]))
            else None,
        }
        for name, span in GLOBAL_GROUPS.items()
    }
    return {
        "n_molecules": int(n_molecules),
        "n_constant_coordinates": int(n_constant),
        "min_abs_corr_informative": float(np.min(np.abs(informative))) if informative.size else None,
        "all_coordinates_affine_consistent": bool(
            informative.size == 0 or np.min(np.abs(informative)) >= 0.999
        ),
        "groups": groups,
        "provenance": "zinc_long_range_proxy.global_feature_views(...)['global_all'] "
        "= concat([structure_short(15), structure_long(15), atom_hist(28), bond_hist(4)])",
        "official_test_loaded": False,
    }


def verify_relation_groups(n_pairs: int = 200_000) -> dict[str, Any]:
    """Empirical checks of the 23-D raw / 15-D used pair-relation grouping."""
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run

    valid = p1run.load_split("valid")
    relation = torch.cat([data.pair_relation for data in valid], dim=0).numpy()
    bucket = torch.cat([data.pair_bucket for data in valid], dim=0).numpy()
    n = min(int(n_pairs), int(relation.shape[0]))
    relation = relation[:n]
    bucket = bucket[:n]
    one_hot = relation[:, 0:5]
    checks: dict[str, Any] = {}
    checks["distance_one_hot_rows_sum_to_one"] = bool(np.allclose(one_hot.sum(axis=1), 1.0, atol=1e-5))
    checks["bucket_equals_argmax_one_hot"] = bool(np.all(bucket == one_hot.argmax(axis=1)))
    exact = bucket < 4
    checks["log_distance_matches_bucket_when_exact"] = bool(
        np.allclose(relation[exact, 5], np.log1p(bucket[exact] + 1.0), atol=1e-5)
    )
    overlap = relation[:, 6:11]
    checks["overlap_block_within_unit_interval"] = bool(
        np.all(overlap >= -1e-6) and np.all(overlap <= 1.0 + 1e-6)
    )
    boundary = relation[:, 11:14]
    checks["boundary_indicator_is_binary"] = bool(np.all(np.isin(np.round(boundary[:, 2]), (0.0, 1.0))))
    log_path_count = relation[:, 18]
    path_count = np.expm1(log_path_count)
    checks["path_count_is_log1p_integer"] = bool(np.allclose(path_count, np.round(path_count), atol=1e-3))
    checks["path_count_at_least_one"] = bool(np.all(path_count >= 1.0 - 1e-3))
    adjacent = relation[:, 19:23]
    checks["adjacent_bond_type_one_hot_or_zero"] = bool(
        np.all(adjacent.sum(axis=1) <= 1.0 + 1e-5) and np.all(adjacent >= -1e-6)
    )
    checks["path_bond_mean_block_present"] = bool(relation.shape[1] >= 23)
    return {
        "n_pairs_checked": int(n),
        "raw_layout": {name: list(span) for name, span in RELATION_RAW_LAYOUT.items()},
        "used_indices": list(p1.P1_RELATION_INDICES),
        "used_groups": {name: list(span) for name, span in RELATION_GROUPS.items()},
        "checks": checks,
        "all_passed": bool(all(checks.values())),
        "official_test_loaded": False,
    }


def verify_topology_groups() -> dict[str, Any]:
    """Correlate the stored standardized ``topology_features`` with raw hinge rows."""

    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
    from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

    frame = ztopo.load_cache("valid")
    if frame is None:
        return {"available": False, "reason": "topology cache missing", "official_test_loaded": False}
    raw = np.stack([ztopo.raw_vector(frame.iloc[i], "hinge") for i in range(len(frame))]).astype(np.float64)
    valid = p1run.load_split("valid")
    stored = np.stack([data.topology_features.numpy().reshape(-1) for data in valid]).astype(np.float64)
    if stored.shape != raw.shape:
        raise RuntimeError(f"topology shape mismatch {stored.shape} vs {raw.shape}")
    correlations = []
    for column in range(stored.shape[1]):
        left, right = stored[:, column], raw[:, column]
        correlations.append(
            0.0
            if (float(left.std()) == 0.0 or float(right.std()) == 0.0)
            else float(np.corrcoef(left, right)[0, 1])
        )
    correlations = np.asarray(correlations)
    names = ztopo.feature_names("hinge")
    return {
        "available": True,
        "n_graphs": int(stored.shape[0]),
        "min_abs_corr": float(np.min(np.abs(correlations))),
        "all_coordinates_affine_consistent": bool(np.min(np.abs(correlations)) >= 0.999),
        "feature_names": list(names),
        "correlations": [float(value) for value in correlations],
        "provenance": "zinc_topology_features.raw_vector(mode='hinge'): "
        "[longest, n3..n10, n>10, mcb_count, mcb_max, mcb_mean, mcb_total, cycle_rank] "
        "+ [longest, longest^2, ReLU(longest-3)..ReLU(longest-10)]",
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# Phase A: static information-flow inventory
# ---------------------------------------------------------------------------


#: seeds used for the cross-graph pooled-readout row shuffle.
READOUT_SHUFFLE_SEEDS = (9101, 9202)


def interventions_readout_shuffle() -> list[Intervention]:
    """Cross-graph row-shuffle probes for the pooled readout blocks."""

    def row(name: str, blocks: tuple[str, ...], note: str) -> Intervention:
        return Intervention(
            name,
            "readout_shuffle",
            AuditMask(),
            note,
            seeds=READOUT_SHUFFLE_SEEDS,
            readout_shuffle=blocks,
        )

    return [
        row("PS1", ("unary_first",), "shuffle unary first-moment rows across graphs"),
        row("PS2", ("unary_second",), "shuffle unary second-moment rows across graphs"),
        row("PS3", ("unary_count",), "shuffle unary count rows across graphs"),
        row("PS4", ("pair_first",), "shuffle pair first-moment rows across graphs"),
        row("PS5", ("pair_second",), "shuffle pair second-moment rows across graphs"),
        row("PS6", ("pair_count",), "shuffle pair count rows across graphs"),
        row("PS7", ("unary_first", "unary_second", "unary_count"), "shuffle full unary pool rows"),
        row("PS8", ("pair_first", "pair_second", "pair_count"), "shuffle full pair readout rows"),
        row("PS9", ("unary_second", "pair_second"), "shuffle both second-moment blocks"),
    ]


def interventions_graph_shuffle() -> list[Intervention]:
    """Distribution-preserving graph-level probes (cross-molecule row shuffle)."""
    return [
        Intervention(
            "GS1",
            "global_shuffle",
            AuditMask(),
            "shuffle graph atom+bond histogram rows",
            graph_shuffle="global_chemistry",
            seeds=GRAPH_SHUFFLE_SEEDS,
        ),
        Intervention(
            "GS2",
            "global_shuffle",
            AuditMask(),
            "shuffle graph structure (short+long) rows",
            graph_shuffle="global_structure",
            seeds=GRAPH_SHUFFLE_SEEDS,
        ),
        Intervention(
            "GS3",
            "global_shuffle",
            AuditMask(),
            "shuffle full global62 rows",
            graph_shuffle="global_all",
            seeds=GRAPH_SHUFFLE_SEEDS,
        ),
        Intervention(
            "GS4",
            "global_shuffle",
            AuditMask(),
            "shuffle topology25 rows",
            graph_shuffle="topology",
            seeds=GRAPH_SHUFFLE_SEEDS,
        ),
        Intervention(
            "GS5",
            "global_shuffle",
            AuditMask(),
            "shuffle global62 + topology25 rows",
            graph_shuffle="global_topology",
            seeds=GRAPH_SHUFFLE_SEEDS,
        ),
    ]


def interventions_fill() -> list[Intervention]:
    """Mean-fill variants of every input-block intervention (distribution-aware).

    Concrete (external) input blocks are replaced by their official-valid
    per-coordinate mean instead of by zero; internal activation masks
    (``node_binding_zero``, ``edge_binding_zero``, ``coord_zero``,
    ``graph_hidden_zero``, ``gate_off``, ``pair_projection_zero``) have no
    distribution-matched replacement and are therefore excluded.
    """
    excluded = {
        "N1",
        "N2",
        "N6",
        "EB1",
        "EB2",
        "EB3",
        "N0",
        "A0",
        "G0",
        "T0",
        "R0",
        "P0",
    }
    fillable = [
        intervention
        for intervention in interventions()
        if intervention.name not in excluded and intervention.mask is not None
    ]
    rows: list[Intervention] = []
    for intervention in fillable:
        mask = intervention.mask
        touches_input = bool(
            mask.anchor_zero_groups
            or mask.global_zero_groups
            or mask.topology_zero
            or mask.relation_zero_groups
            or mask.unary_zero_blocks
            or mask.pair_zero_blocks
        )
        if not touches_input:
            continue
        rows.append(
            Intervention(
                intervention.name,
                intervention.category + "_fill",
                mask,
                intervention.notes + " [mean-fill probe]",
                shuffle_kind=intervention.shuffle_kind,
                seeds=intervention.seeds,
                use_fill=True,
                graph_shuffle=intervention.graph_shuffle,
            )
        )
    return rows


#: species of relation-row-shuffle targets: ``kind -> raw column slices``.
RELATION_SHUFFLE_KINDS: dict[str, tuple[tuple[int, int], ...]] = {
    "distance": ((0, 6),),
    "overlap": ((6, 11),),
    "boundary": ((11, 14),),
    "path_count": ((18, 19),),
    "all": ((0, 14), (18, 19)),
}


def permute_pair_rows(data_list: Sequence[Any], kind: str, seed: int):
    """Cross-pair row shuffle of relation columns (distribution-preserving).

    The marginal distribution of each shuffled relation block is untouched;
    only the pair it belongs to changes.  Returns a ``restore()`` callable.
    """
    if kind not in RELATION_SHUFFLE_KINDS:
        raise ValueError(f"unknown relation-shuffle kind {kind!r}")
    generator = torch.Generator().manual_seed(int(seed))
    saved: list[tuple[Any, torch.Tensor]] = []
    if not data_list:
        return lambda: None
    for low, high in RELATION_SHUFFLE_KINDS[kind]:
        concatenated = torch.cat([data.pair_relation for data in data_list], dim=0)
        permutation = torch.randperm(int(concatenated.shape[0]), generator=generator)
        shuffled = concatenated[permutation]
        offset = 0
        for data in data_list:
            original = data.pair_relation
            rows = int(original.shape[0])
            replacement = original.clone()
            replacement[:, int(low) : int(high)] = shuffled[offset : offset + rows, int(low) : int(high)]
            offset += rows
            saved.append((data, original))
            data.pair_relation = replacement

    def restore() -> None:
        for data, original in saved:
            data.pair_relation = original

    return restore


def interventions_relation_shuffle() -> list[Intervention]:
    """Cross-pair row-shuffle probes for the 15-D used relation groups."""
    return [
        Intervention(
            "RS1",
            "relation_shuffle",
            AuditMask(),
            "shuffle distance block rows across pairs",
            seeds=READOUT_SHUFFLE_SEEDS,
            relation_shuffle="distance",
        ),
        Intervention(
            "RS2",
            "relation_shuffle",
            AuditMask(),
            "shuffle overlap block rows across pairs",
            seeds=READOUT_SHUFFLE_SEEDS,
            relation_shuffle="overlap",
        ),
        Intervention(
            "RS3",
            "relation_shuffle",
            AuditMask(),
            "shuffle boundary block rows across pairs",
            seeds=READOUT_SHUFFLE_SEEDS,
            relation_shuffle="boundary",
        ),
        Intervention(
            "RS4",
            "relation_shuffle",
            AuditMask(),
            "shuffle log-path-count rows across pairs",
            seeds=READOUT_SHUFFLE_SEEDS,
            relation_shuffle="path_count",
        ),
        Intervention(
            "RS5",
            "relation_shuffle",
            AuditMask(),
            "shuffle all used relation rows across pairs",
            seeds=READOUT_SHUFFLE_SEEDS,
            relation_shuffle="all",
        ),
    ]


def _relation_shuffle_kind(name: str) -> str:
    return {"RS1": "distance", "RS2": "overlap", "RS3": "boundary", "RS4": "path_count", "RS5": "all"}[name]


def _entry(
    name: str,
    dim: str,
    source: str,
    module: str,
    *,
    is_structure: bool = False,
    is_chemistry: bool = False,
    correspondence: bool = False,
    dictionary_mediated: bool = False,
    handcrafted: bool = False,
    bypasses_dictionary: bool = False,
    bypasses_local_binding: bool = False,
    note: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "dim": dim,
        "source": source,
        "entry_module": module,
        "is_structure": bool(is_structure),
        "is_chemistry": bool(is_chemistry),
        "is_structure_chemistry_correspondence": bool(correspondence),
        "dictionary_mediated": bool(dictionary_mediated),
        "is_handcrafted_statistic": bool(handcrafted),
        "bypasses_dictionary": bool(bypasses_dictionary),
        "bypasses_local_structure_semantic_binding": bool(bypasses_local_binding),
        "note": note,
    }


def information_inventory() -> dict[str, Any]:
    """Complete machine-readable information-flow inventory (preregistration §2)."""
    rows: list[dict[str, Any]] = [
        # -- structural dictionary path -------------------------------------
        _entry(
            "phi65",
            "65",
            "fsar_r2_ar0.build_phi (pure topology, no chemistry)",
            "code()",
            is_structure=True,
            handcrafted=True,
            note="radius-2 rooted operator basis; chemistry-free by construction",
        ),
        _entry(
            "dictionary_D",
            "65x32",
            "sdb_v0 K-SVD init, trainable",
            "code()",
            is_structure=True,
            note="column-normalized tied dictionary; K32/s8/IHT10 frozen",
        ),
        _entry(
            "alpha_coord",
            "32",
            "tied IHT10(Dbar, phi65), exact top-8",
            "code()",
            is_structure=True,
            dictionary_mediated=True,
            note="the only fine-grained learned structural coordinate",
        ),
        _entry(
            "dict_atom_one_hot_q",
            "28",
            "raw ZINC node type (encoded cache)",
            "environments()",
            is_chemistry=True,
            note="per-occurrence atom one-hot, not dictionary-mediated",
        ),
        _entry(
            "bond_type_b",
            "4",
            "raw ZINC edge_attr (encoded cache)",
            "environments()",
            is_chemistry=True,
            note="per-bond-occurrence bond one-hot",
        ),
        _entry(
            "shell_s_iv",
            "3 (routing)",
            "e2e_dictenv_v0.env_incidence BFS distance",
            "environments()",
            is_structure=True,
            handcrafted=True,
            note="root-relative BFS shell / routing index, no chemistry",
        ),
        _entry(
            "shellpair_p_uv",
            "6 (routing)",
            "e2e_dictenv_v0.env_incidence sorted shell pair",
            "environments()",
            is_structure=True,
            handcrafted=True,
            note="root-relative shellpair / routing index",
        ),
        # -- node structure-semantic binding ---------------------------------
        _entry(
            "node_slot_u_v",
            "96",
            "(alpha_v W_A^S) * (q_v W_A^C) / sqrt(96)",
            "environments()",
            is_structure=True,
            is_chemistry=True,
            correspondence=True,
            dictionary_mediated=True,
            note="node structure-semantic binding (outer product)",
        ),
        _entry(
            "node_slots_A_i",
            "3x96=288",
            "per-(root,shell) index_add of node_slot_u_v",
            "node_encoder",
            is_structure=True,
            is_chemistry=True,
            correspondence=True,
            dictionary_mediated=True,
            note="shell-routed; each slot is a sum over occurrences",
        ),
        _entry(
            "node_encoder_out",
            "3x48=144",
            "shared MLP 96->64->48",
            "fusion",
            note="per-shell learned re-encoding, shared across shells",
        ),
        # -- edge structure-semantic binding ---------------------------------
        _entry(
            "edge_role_g_uv",
            "96",
            "[a_u+a_v; |a_u-a_v|; a_u*a_v]",
            "environments()",
            is_structure=True,
            dictionary_mediated=True,
            note="symmetric dictionary-role pair descriptor",
        ),
        _entry(
            "edge_slot_ue_uv",
            "48",
            "(g_uv W_E^S) * (b_uv W_E^C) / sqrt(48)",
            "environments()",
            is_structure=True,
            is_chemistry=True,
            correspondence=True,
            dictionary_mediated=True,
            note="edge structure-semantic binding",
        ),
        _entry(
            "edge_slots_E_i",
            "6x48=288",
            "per-(root,shellpair) index_add of edge_slot_ue_uv",
            "edge_encoder",
            is_structure=True,
            is_chemistry=True,
            correspondence=True,
            dictionary_mediated=True,
            note="shellpair-routed",
        ),
        _entry(
            "edge_encoder_out",
            "6x32=192",
            "shared MLP 48->48->32",
            "fusion",
            note="per-shellpair learned re-encoding, shared across shellpairs",
        ),
        # -- anchor ----------------------------------------------------------
        _entry(
            "anchor_root_identity",
            "28",
            "one-hot q_i (root atom)",
            "anchor_encoder",
            is_chemistry=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="zeroth-order identity, no patch context",
        ),
        _entry(
            "anchor_patch_atom_mass",
            "28",
            "sum_{v in P_i} q_v (whole patch, unconditioned)",
            "anchor_encoder",
            is_chemistry=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="patch-level atom marginal; shell information is summed away",
        ),
        _entry(
            "anchor_patch_bond_mass",
            "4",
            "sum_{e in E_i} b_e (whole patch, unconditioned)",
            "anchor_encoder",
            is_chemistry=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="patch-level bond marginal",
        ),
        _entry(
            "anchor_size",
            "2",
            "[log1p|V_i|, log1p|E_i|]",
            "anchor_encoder",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="root-relative patch size only",
        ),
        # -- decoder ---------------------------------------------------------
        _entry("anchor_encoder_out", "32", "MLP 62->32->32", "fusion", note="anchor re-encoding"),
        _entry("fusion_input", "368", "cat[anchor32, node144, edge192]", "fusion()", note="H1 slot fusion"),
        _entry(
            "local_environment_E_i",
            "48",
            "fusion MLP 368->128->48",
            "pair/graph backend",
            is_structure=True,
            is_chemistry=True,
            correspondence=True,
            dictionary_mediated=True,
            note="frozen after formation; the only local representation the backend sees",
        ),
        # -- relation --------------------------------------------------------
        _entry(
            "pair_relation_raw",
            "23",
            "zinc_patch_path_pooling._pair_relation",
            "relation_encoder",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="pure topology (path-bond-mean and adjacent-bond blocks dropped by P1)",
        ),
        _entry(
            "pair_relation_distance",
            "6",
            "raw [0:5] one-hot + [5] log distance",
            "relation_encoder",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="used coords 0..5",
        ),
        _entry(
            "pair_relation_overlap",
            "5",
            "raw [6:11]",
            "relation_encoder",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="used coords 6..10; patch node-set overlap ratios",
        ),
        _entry(
            "pair_relation_boundary",
            "3",
            "raw [11:14]",
            "relation_encoder",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="used coords 11..13; boundary overlap + centre containment indicator",
        ),
        _entry(
            "pair_relation_path_count",
            "1",
            "raw [18] log1p(number of shortest paths)",
            "relation_encoder",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="used coord 14",
        ),
        _entry(
            "pair_bucket",
            "5 cats",
            "argmax distance one-hot",
            "distance_gate",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="also selects the pair-pool bucket",
        ),
        # -- graph-level -----------------------------------------------------
        _entry(
            "global_structure_short",
            "15",
            "global_feature_views short block",
            "global_encoder",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="degree/density/triangle/clustering summary",
        ),
        _entry(
            "global_structure_long",
            "15",
            "global_feature_views long block",
            "global_encoder",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="all-pairs shortest-path + eccentricity summary",
        ),
        _entry(
            "global_atom_histogram",
            "28",
            "whole-molecule atom-type frequency",
            "global_encoder",
            is_chemistry=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="whole-molecule chemistry marginal",
        ),
        _entry(
            "global_bond_histogram",
            "4",
            "whole-molecule bond-type frequency",
            "global_encoder",
            is_chemistry=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="whole-molecule chemistry marginal",
        ),
        _entry(
            "topology_features_25",
            "25",
            "zinc_topology_features raw_vector('hinge')",
            "topology_encoder",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
            note="cycle spectrum / MCB / longest-cycle hinge features; graph-level structure bypass",
        ),
        # -- pools -----------------------------------------------------------
        _entry(
            "unary_first_moment",
            "48",
            "sum_v E_i over graph nodes",
            "reader",
            is_structure=True,
            is_chemistry=True,
            correspondence=True,
            dictionary_mediated=True,
        ),
        _entry(
            "unary_second_moment",
            "48",
            "sum_v E_i^2",
            "reader",
            is_structure=True,
            is_chemistry=True,
            correspondence=True,
            dictionary_mediated=True,
            note="explicit second-order statistics",
        ),
        _entry(
            "unary_count",
            "1",
            "log1p(number of nodes)",
            "reader",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
        ),
        _entry(
            "pair_first_moment",
            "5x16=80",
            "sum over pairs per distance bucket of pair_value",
            "reader",
            is_structure=True,
            dictionary_mediated=True,
        ),
        _entry(
            "pair_second_moment",
            "5x16=80",
            "sum of pair_value^2 per bucket",
            "reader",
            is_structure=True,
            dictionary_mediated=True,
            note="explicit second-order statistics",
        ),
        _entry(
            "pair_count",
            "5",
            "log1p(number of pairs) per bucket",
            "reader",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
        ),
        _entry(
            "graph_hidden",
            "32",
            "global_encoder MLP 62->32->32 on global_context",
            "reader",
            is_chemistry=True,
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
        ),
        _entry(
            "topology_hidden",
            "8",
            "topology_encoder MLP 25->16->8",
            "reader",
            is_structure=True,
            handcrafted=True,
            bypasses_dictionary=True,
            bypasses_local_binding=True,
        ),
        _entry(
            "reader_input",
            "302 = 97 + 165 + 32 + 8",
            "cat[unary, relation_readout, graph_hidden, topology]",
            "reader",
        ),
        _entry("prediction", "1", "GenericReader 302->13->13->1", "output"),
    ]
    flags = (
        "is_structure",
        "is_chemistry",
        "is_structure_chemistry_correspondence",
        "dictionary_mediated",
        "is_handcrafted_statistic",
        "bypasses_dictionary",
        "bypasses_local_structure_semantic_binding",
    )
    summary = {
        "n_entries": len(rows),
        "counts": {flag: int(sum(1 for row in rows if row[flag])) for flag in flags},
        "non_dictionary_bypasses": [
            row["name"]
            for row in rows
            if row["bypasses_dictionary"] and row["bypasses_local_structure_semantic_binding"]
        ],
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "device": "cpu",
        "baseline": {
            "checkpoint": "results/e2e_dictenv_p2_abs/states/H1_soup_state.pt",
            "config": H1_CONFIG.as_dict(),
            "soup_members": list(H1_SOUP_MEMBERS),
            "historical_soup_valid_mae": H1_HISTORICAL_SOUP_MAE,
        },
        "coordinate_groups": {
            "anchor": {k: list(v) for k, v in ANCHOR_GROUPS.items()},
            "global_context": {k: list(v) for k, v in GLOBAL_GROUPS.items()},
            "pair_relation_used": {k: list(v) for k, v in RELATION_GROUPS.items()},
            "unary_pool": {k: list(v) for k, v in UNARY_BLOCKS.items()},
            "pair_pool": {k: list(v) for k, v in PAIR_BLOCKS.items()},
        },
        "entries": rows,
        "summary": summary,
        "reader_input_dim": READER_IN_DIM,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# CPU phase C: training protocols
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return float(pages) * 4096.0 / (1024.0**2)
    except Exception:  # pragma: no cover - non-Linux fallback
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _evaluate_model(
    model: AuditModel, loader: Any, device: torch.device, mask: AuditMask | None
) -> dict[str, Any]:
    result = evaluate_mask(model, loader, device, mask)
    return {"mae": float(result["mae"]), "n_molecules": int(result["n_molecules"])}


def train_cpu(
    *,
    tag: str,
    mask: AuditMask | None,
    epochs: int,
    threads: int,
    out_dir: Any,
    init_state: Mapping[str, torch.Tensor] | None = None,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    seed: int = 0,
    save_states: bool = True,
    log: bool = True,
    model_factory: Callable[[np.ndarray, int], "AuditModel"] | None = None,
    arm_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One CPU training run of the frozen H1 protocol with an optional mask.

    ``init_state=None`` trains from scratch (Tier 2 matched protocol).
    ``init_state=<H1 soup state>`` warm-starts (Tier 1 adaptation), with a fresh
    Adam state in both cases.  All hyper-parameters come from
    ``zinc_e2e_dictenv_p2_abs`` / ``zinc_e2e_dictenv_p1``; none is re-typed.
    """
    from pathlib import Path

    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    device = attach_cpu(threads)
    if device.type != "cpu":
        raise RuntimeError("audit training is CPU-only")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionary, dict_sha = p2run.load_dictionary(H1_CONFIG.dict_kind)
    p2run._seed_everything(int(seed))
    factory = build_audit_model if model_factory is None else model_factory
    model = factory(dictionary, int(seed))
    if init_state is not None:
        missing = model.load_state_dict({k: v.float() for k, v in init_state.items()})
        if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
            raise RuntimeError(f"warm-start state mismatch: {missing}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY)
    )
    loader = p1.make_env_loader(
        train_data, int(p2run.BATCH_SIZE), True, int(seed) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    eval_loader = p1.make_env_loader(
        valid_data, int(p2run.BATCH_SIZE), False, int(seed) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    lam = float(H1_LAMBDA)
    soup_k = int(p2run.SOUP_K)
    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    rss_start = _rss_mb()
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        task_sum = rec_sum = 0.0
        n_mol = n_nodes = 0
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
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi**2).sum(dim=1) + v0.EPS)).sum())
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = _evaluate_model(model, eval_loader, device, mask)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "train_rec": train_rec,
                "valid_mae": float(valid["mae"]),
                "d_norm": float(model.D.detach().norm()),
            }
        )
        epoch_states[int(epoch)] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        keep = {i + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:soup_k]}
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 10 == 0 or epoch == int(epochs)):
            print(
                f"[{tag}] epoch={epoch:03d} train={train_mae:.6f} rec={train_rec:.2e} "
                f"valid={float(valid['mae']):.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
    wall = float(time.perf_counter() - started)
    assert best_state is not None
    members = sorted(
        int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:soup_k]
    )
    soup_state = {
        k: torch.stack([epoch_states[e][k].float() for e in members]).mean(0)
        for k in epoch_states[members[0]]
    }
    soup_model = factory(dictionary, int(seed))
    soup_model.load_state_dict(soup_state)
    soup_valid = _evaluate_model(soup_model, eval_loader, device, mask)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "device": "cpu",
        "threads": int(threads),
        "seed": int(seed),
        "mask": mask.as_dict() if mask is not None else None,
        "warm_started": bool(init_state is not None),
        "config": H1_CONFIG.as_dict(),
        "dictionary_sha256": dict_sha,
        "lambda_rec": lam,
        "epoch_budget": int(epochs),
        "epochs_run": int(len(curve)),
        "parameters": p2.total_parameter_count(H1_CONFIG),
        "actual_params": int(sum(p.numel() for p in model.parameters())),
        "curve": curve,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(curve[-1]["valid_mae"]),
        "train_min_mae": float(min(row["train_mae"] for row in curve)),
        "soup": {
            "members": members,
            "member_valid_mae": [float(curve[e - 1]["valid_mae"]) for e in members],
            "soup_valid_mae": float(soup_valid["mae"]),
        },
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve), 1)),
        "rss_start_mb": float(rss_start),
        "rss_end_mb": float(_rss_mb()),
        "peak_rss_mb": _peak_rss_mb(),
        "official_test_loaded": False,
    }
    if save_states:
        torch.save(best_state, out_dir / f"{tag}_raw_state.pt")
        torch.save(soup_state, out_dir / f"{tag}_soup_state.pt")
        torch.save(
            {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
            out_dir / f"{tag}_final_state.pt",
        )
    payload["best_state_sha256"] = state_sha256(best_state)
    payload["soup_state_sha256"] = state_sha256(soup_state)
    if arm_spec is not None:
        payload["arm_spec"] = dict(arm_spec)
    return payload


def _peak_rss_mb() -> float:
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


# ---------------------------------------------------------------------------
# Phase C candidate registry (pre-registered order; selection is decision-gated)
# ---------------------------------------------------------------------------

#: ``name -> (mask, notes)``.  ``M0`` is the matched continuation control:
#: the untouched H1 forward continued for the same budget.
PHASE_C_CANDIDATES: dict[str, tuple[AuditMask, str]] = {
    "M0": (AuditMask(), "continuation control: untouched H1 forward"),
    "C1": (
        AuditMask(global_zero_groups=GLOBAL_CHEMISTRY_GROUPS),
        "remove graph-level chemistry marginal (atom+bond histogram)",
    ),
    "C2": (
        AuditMask(anchor_zero_groups=("atom_mass", "bond_mass")),
        "minimal anchor: root atom + size only",
    ),
    "C3": (
        AuditMask(global_zero_groups=GLOBAL_CHEMISTRY_GROUPS, anchor_zero_groups=("atom_mass", "bond_mass")),
        "clean bypass model: C1 + C2",
    ),
    "C4": (
        AuditMask(
            global_zero_groups=GLOBAL_CHEMISTRY_GROUPS,
            anchor_zero_groups=("atom_mass", "bond_mass"),
            relation_zero_groups=("overlap", "boundary", "path_count"),
        ),
        "clean model + distance-only relation",
    ),
    "C5": (
        AuditMask(
            global_zero_groups=GLOBAL_CHEMISTRY_GROUPS,
            anchor_zero_groups=("atom_mass", "bond_mass"),
            unary_zero_blocks=("second",),
            pair_zero_blocks=("second",),
        ),
        "clean model + no explicit second moments",
    ),
    # Not in the original preregistration list: Phase B's distribution-preserving
    # probes classified four blocks as dormant (unary count, pair count, relation
    # log-path-count, global bond histogram), so this is the minimal "delete only
    # what is dormant" candidate.  Declared before the first adaptation run and
    # applied to every candidate identically.
    "C6": (
        AuditMask(
            global_zero_groups=GLOBAL_CHEMISTRY_GROUPS + ("bond_histogram",),
            unary_zero_blocks=("count",),
            pair_zero_blocks=("count",),
            relation_zero_groups=("path_count",),
        ),
        "remove graph chemistry marginal + the four frozen-dormant statistic blocks",
    ),
}


def candidate_mask(name: str) -> AuditMask:
    if name not in PHASE_C_CANDIDATES:
        raise KeyError(f"unknown Phase-C candidate {name!r}; known={sorted(PHASE_C_CANDIDATES)}")
    return PHASE_C_CANDIDATES[name][0]


__all__ = [
    "PROTOCOL_VERSION",
    "H1_CONFIG",
    "H1_SOUP_MEMBERS",
    "H1_HISTORICAL_SOUP_MAE",
    "H1_LAMBDA",
    "ANCHOR_GROUPS",
    "GLOBAL_GROUPS",
    "GLOBAL_CHEMISTRY_GROUPS",
    "GLOBAL_STRUCTURE_GROUPS",
    "RELATION_GROUPS",
    "RELATION_RAW_LAYOUT",
    "UNARY_BLOCKS",
    "PAIR_BLOCKS",
    "READER_IN_DIM",
    "SHUFFLE_SEEDS",
    "GRAPH_SHUFFLE_SEEDS",
    "GRAPH_SHUFFLE_KINDS",
    "RELATION_SHUFFLE_KINDS",
    "READOUT_SHUFFLE_SEEDS",
    "READOUT_SHUFFLE_BLOCKS",
    "AuditMask",
    "Intervention",
    "AuditModel",
    "PHASE_C_CANDIDATES",
    "build_audit_model",
    "load_h1_soup_state",
    "interventions",
    "interventions_fill",
    "interventions_graph_shuffle",
    "interventions_readout_shuffle",
    "apply_readout_permutation",
    "candidate_mask",
    "build_fill_policy",
    "permute_graph_rows",
    "permute_pair_rows",
    "interventions_relation_shuffle",
    "prepare_shuffles",
    "clear_shuffles",
    "attach_cpu",
    "evaluate_mask",
    "run_intervention",
    "summarise_interventions",
    "information_inventory",
    "verify_global_context_provenance",
    "verify_relation_groups",
    "verify_topology_groups",
    "pool_moments_masked",
    "pool_pair_moments_masked",
    "_replace_grouped_columns",
    "_zero_grouped_columns",
    "train_cpu",
    "state_sha256",
]
