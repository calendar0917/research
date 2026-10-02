"""E2E-DictEnv-Typed-Cycle-v1 — static typed chordless-cycle object into the shared task dictionary.

Round ``zinc_e2e_dictenv_typed_cycle_v1`` (Workstream Z, ZINC).
Pre-registration:
``tracks/ksvd/notes/zinc_e2e_dictenv_typed_cycle_v1_preregistration.md``.
Phase-A screen core: ``typed_cycle_probe_v1.py``.

This module keeps the complete frozen **Small** ``LatentBridgeSEM108``
(106,925 parameters: Sem108 body 97,709 + shared ``D_L``/``V_L`` 9,216) and adds
**one** static object type read through the *same* task dictionary:

* every chordless ring of length 3..10 is read once as an ordered atom/bond
  sequence; all ``2L`` dihedral views are encoded by one shared MLP
  (``338 -> 64 -> 48``, SiLU) and averaged **after** the nonlinearity;
* that per-ring latent ``h_cycle`` is passed through the *existing*
  ``local_dictionary_bridge`` (same ``D_L``/``V_L``, same 16-step ISTA,
  ``lambda1=0.05``, ``lambda2=0.01``); there is no second dictionary;
* the decoded ring values are pooled as
  ``[sum(E_cycle), sum(E_cycle^2), zero_count_slot]`` (97) and concatenated to
  the frozen 302-D unified readout, entering the **same** reader
  (``399 -> 13 -> 13 -> 1``).

No message passing, no attention, no node/edge hidden state, no write-back.
Acyclic graphs contribute 97 exact zeros; the count slot is always zero (no
count shortcut).  CPU only; ``official_test_loaded = False`` always.

Parameter budget: ``106925 + 24816 + 1261 = 133002``, new task-dictionary
parameters = 0.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader

PROTOCOL_VERSION = "e2e_dictenv_typed_cycle_v1"

#: frozen ring-view layout (matches typed_cycle_reference.dihedral_views).
MAX_RING_LENGTH = 10
RING_ATOM_DIM = MAX_RING_LENGTH * 28  # 280
RING_BOND_DIM = MAX_RING_LENGTH * 4  # 40
RING_MASK_DIM = MAX_RING_LENGTH  # 10
RING_LENGTH_DIM = 8  # lengths 3..10
RING_INPUT_DIM = RING_ATOM_DIM + RING_BOND_DIM + RING_MASK_DIM + RING_LENGTH_DIM  # 338
RING_HIDDEN = 64
RING_OUT = int(p2.ENV_DIM)  # 48
RING_POOL_DIM = 2 * RING_OUT + 1  # 97

#: unchanged frozen Small widths.
SMALL_READER_IN = 302
TYPED_READER_IN = SMALL_READER_IN + RING_POOL_DIM  # 399
READER_HIDDEN = tuple(int(width) for width in p2.READER_HIDDEN)  # (13, 13)

#: exact parameter contract.
RING_ENCODER_PARAMETERS = RING_INPUT_DIM * RING_HIDDEN + RING_HIDDEN + RING_HIDDEN * RING_OUT + RING_OUT
READER_PARAMETERS_TYPED = TYPED_READER_IN * READER_HIDDEN[0] + READER_HIDDEN[0] + READER_HIDDEN[0] * READER_HIDDEN[1] + READER_HIDDEN[1] + READER_HIDDEN[1] + 1
READER_PARAMETERS_SMALL = SMALL_READER_IN * READER_HIDDEN[0] + READER_HIDDEN[0] + READER_HIDDEN[0] * READER_HIDDEN[1] + READER_HIDDEN[1] + READER_HIDDEN[1] + 1
READER_INCREMENT = READER_PARAMETERS_TYPED - READER_PARAMETERS_SMALL
SMALL_PARAMETERS = lb.CANDIDATE_PARAMETERS  # 106,925
EXPECTED_PARAMETERS = SMALL_PARAMETERS + RING_ENCODER_PARAMETERS + READER_INCREMENT  # 133,002

#: ring field names attached to each molecule.
RING_VIEW_FIELDS = (
    "ring_view_atom",
    "ring_view_bond",
    "ring_view_mask",
    "ring_view_length",
    "ring_view_cycle",
)
RING_CYCLE_FIELDS = ("ring_cycle_anchor", "ring_cycle_length")


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


# ---------------------------------------------------------------------------
# static ring view construction (integer view set; mirrors the NumPy reference)
# ---------------------------------------------------------------------------
def dihedral_view_components(ref_graph: Any, cycle: Sequence[int]):
    """Return the integer components of all ``2L`` dihedral views of one ring.

    Mirrors ``typed_cycle_reference.dihedral_views`` exactly: atom ``t`` and the
    outgoing bond ``(vertex_t, vertex_{t+1 mod L})`` stay aligned under rotation
    **and** reversal, and the reversed bond type is re-read from the original
    graph bond table (never by flipping a bond array).
    """
    bonds = ref_graph.bonds()
    ell = len(cycle)
    if not 3 <= ell <= MAX_RING_LENGTH:
        raise ValueError(f"cycle size {ell} outside fixed 3..{MAX_RING_LENGTH}")
    atom = np.zeros((2 * ell, MAX_RING_LENGTH), dtype=np.int64)
    bond = np.zeros((2 * ell, MAX_RING_LENGTH), dtype=np.int64)
    mask = np.zeros((2 * ell, MAX_RING_LENGTH), dtype=np.float64)
    length = np.full((2 * ell,), ell, dtype=np.int64)
    view = 0
    for direction in (1, -1):
        for shift in range(ell):
            vertices = [cycle[(shift + direction * t) % ell] for t in range(ell)]
            for t, vertex in enumerate(vertices):
                atom[view, t] = int(ref_graph.atoms[vertex])
                next_vertex = vertices[(t + 1) % ell]
                key = tuple(sorted((vertex, next_vertex)))
                bond[view, t] = int(bonds[key])
                mask[view, t] = 1.0
            view += 1
    return atom, bond, mask, length


def assemble_views(
    view_atom: torch.Tensor,
    view_bond: torch.Tensor,
    view_mask: torch.Tensor,
    view_length: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Assemble the 338-D view matrix from integer components.

    Masked positions stay exact zeros (no atom/bond value is written where the
    position mask is 0), matching the reference construction.
    """
    n_views = int(view_atom.shape[0])
    if dtype is None:
        dtype = torch.float32
    device = view_atom.device
    x = torch.zeros((n_views, RING_INPUT_DIM), dtype=dtype, device=device)
    if n_views == 0:
        return x
    position = torch.arange(MAX_RING_LENGTH, device=device, dtype=torch.long).unsqueeze(0)
    atom_col = position * 28 + view_atom.long()
    bond_col = RING_ATOM_DIM + position * 4 + view_bond.long()
    src = view_mask.to(dtype)
    x.scatter_(1, atom_col, src)
    x.scatter_(1, bond_col, src)
    x[:, RING_ATOM_DIM + RING_BOND_DIM : RING_ATOM_DIM + RING_BOND_DIM + RING_MASK_DIM] = src
    length_col = RING_ATOM_DIM + RING_BOND_DIM + RING_MASK_DIM + (view_length.long() - 3).unsqueeze(1)
    x.scatter_(1, length_col, torch.ones_like(length_col, dtype=dtype))
    return x


# ---------------------------------------------------------------------------
# custom collate: reuse the frozen env offsets, then add the ring offsets
# ---------------------------------------------------------------------------
def typed_cycle_collate(data_list: Sequence[Any]) -> Any:
    """``p1.env_collate`` plus ring offsets (no double offset).

    ``Batch.from_data_list`` already auto-increments any key containing
    ``index``; the ring fields deliberately avoid that token, so their offsets
    are applied exactly once here.  ``ring_view_cycle`` receives the cumulative
    **cycle** count and ``ring_cycle_anchor`` the node ``ptr``; the per-view
    atom/bond/mask/length and per-cycle length fields need no offset.
    """
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1

    data_list = list(data_list)
    base = p1.env_collate(data_list)
    if not data_list or not hasattr(data_list[0], "ring_view_atom"):
        return base
    view_counts = [int(d.ring_view_atom.shape[0]) for d in data_list]
    cycle_counts = [int(d.ring_cycle_anchor.shape[0]) for d in data_list]
    n_views, n_cycles = sum(view_counts), sum(cycle_counts)
    if n_views:
        view_graph = np.repeat(np.arange(len(data_list)), view_counts)
        cycle_ptr = np.concatenate([[0], np.cumsum(cycle_counts)])
        offset = torch.as_tensor(cycle_ptr[view_graph], dtype=base.ring_view_cycle.dtype)
        base.ring_view_cycle = base.ring_view_cycle + offset
    if n_cycles:
        anchor_graph = np.repeat(np.arange(len(data_list)), cycle_counts)
        node_ptr = np.concatenate([[0], np.cumsum([int(d.num_nodes) for d in data_list])])
        offset = torch.as_tensor(node_ptr[anchor_graph], dtype=base.ring_cycle_anchor.dtype)
        base.ring_cycle_anchor = base.ring_cycle_anchor + offset
    return base


def typed_cycle_loader(data_list: Sequence[Any], batch_size: int, shuffle: bool, seed: int) -> Any:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return torch.utils.data.DataLoader(
        list(data_list),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=typed_cycle_collate,
    )


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class TypedCycleSEM108(lb.LatentBridgeSEM108):
    """Frozen Small + one typed-cycle object path through the shared bridge.

    All parent parameters keep the exact ``build_latent_bridge_model`` RNG
    stream; the ring encoder and the widened reader are drawn from a private,
    isolated generator.  The new reader columns start at zero, so at
    initialisation the old function is exactly contained.
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: Any,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        ring_seed: int = 1,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        self.ring_seed = int(ring_seed)
        self.ring_encoder = nn.Sequential(
            nn.Linear(RING_INPUT_DIM, RING_HIDDEN),
            nn.SiLU(),
            nn.Linear(RING_HIDDEN, RING_OUT),
        )
        generator = torch.Generator().manual_seed(int(ring_seed))
        for module in self.ring_encoder:
            sc._reset_module_from_generator(module, generator)

        old_reader = self.reader
        new_reader = GenericReader(TYPED_READER_IN, READER_HIDDEN)
        sc._reset_module_from_generator(new_reader, generator)
        with torch.no_grad():
            new_reader.net[0].weight.zero_()
            new_reader.net[0].weight[:, :SMALL_READER_IN].copy_(old_reader.net[0].weight)
            new_reader.net[0].bias.copy_(old_reader.net[0].bias)
            new_reader.net[2].weight.copy_(old_reader.net[2].weight)
            new_reader.net[2].bias.copy_(old_reader.net[2].bias)
            new_reader.net[4].weight.copy_(old_reader.net[4].weight)
            new_reader.net[4].bias.copy_(old_reader.net[4].bias)
        self.reader = new_reader

        #: per-forward slot for the ring pool consumed by the reader pre-hook.
        self._pending_ring_pool: torch.Tensor | None = None
        self._zero_ring = False

        def _append_ring_pool(_module: nn.Module, args: tuple[Any, ...]):
            pool = self._pending_ring_pool
            if pool is None:
                raise RuntimeError(
                    "typed-cycle reader called without a pending ring pool; "
                    "a direct reader call is not part of the model forward"
                )
            self._pending_ring_pool = None
            unified = args[0]
            if int(unified.shape[1]) != SMALL_READER_IN:
                raise RuntimeError(
                    f"frozen unified readout width {int(unified.shape[1])} != {SMALL_READER_IN}"
                )
            return (torch.cat([unified, pool.to(unified.dtype)], dim=1),) + tuple(args[1:])

        self.reader.register_forward_pre_hook(_append_ring_pool)

    # -- reader seam -----------------------------------------------------------

    # -- ring object -----------------------------------------------------------
    def ring_pool(self, data: Any) -> torch.Tensor:
        n_graphs = int(data.global_context.shape[0])
        device = data.global_context.device
        if not hasattr(data, "ring_view_atom") or int(data.ring_view_atom.shape[0]) == 0:
            return torch.zeros((n_graphs, RING_POOL_DIM), device=device, dtype=data.global_context.dtype)
        views = assemble_views(
            data.ring_view_atom,
            data.ring_view_bond,
            data.ring_view_mask,
            data.ring_view_length,
            dtype=torch.float32,
        )
        h = self.ring_encoder(views)
        cycle_ids = data.ring_view_cycle.long()
        n_cycles = int(data.ring_cycle_anchor.shape[0])
        sums = torch.zeros((n_cycles, RING_OUT), device=h.device, dtype=h.dtype)
        sums.index_add_(0, cycle_ids, h)
        counts = torch.bincount(cycle_ids, minlength=n_cycles).clamp_min(1).to(h.dtype)
        h_cycle = sums / counts.unsqueeze(1)
        E_cycle = self.local_dictionary_bridge(h_cycle)
        if self._zero_ring:
            E_cycle = torch.zeros_like(E_cycle)
        graph_of_cycle = data.batch[data.ring_cycle_anchor.to(data.batch.device)].to(E_cycle.device)
        total = torch.zeros((n_graphs, RING_OUT), device=E_cycle.device, dtype=E_cycle.dtype)
        squared = torch.zeros_like(total)
        total.index_add_(0, graph_of_cycle, E_cycle)
        squared.index_add_(0, graph_of_cycle, E_cycle * E_cycle)
        count_slot = torch.zeros((n_graphs, 1), device=E_cycle.device, dtype=E_cycle.dtype)
        return torch.cat([total, squared, count_slot], dim=1)

    # -- forward ---------------------------------------------------------------
    def forward(self, data: Any, **kwargs: Any):
        self._pending_ring_pool = self.ring_pool(data)
        try:
            return super().forward(data, **kwargs)
        finally:
            self._pending_ring_pool = None

    # -- inference-only intervention -------------------------------------------
    def set_zero_ring(self, enabled: bool) -> None:
        self._zero_ring = bool(enabled)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


# attach the owner reference used by the reader pre-hook (registered above).


def build_typed_cycle_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: Any,
    *,
    ring_seed: int = 1,
) -> TypedCycleSEM108:
    """Build the candidate with the exact parent RNG stream for shared params."""
    torch.manual_seed(int(seed))
    model = TypedCycleSEM108(
        lb.cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        node_binding=lb.cssd.CSSD_SPEC.node_binding,
        edge_binding=lb.cssd.CSSD_SPEC.edge_binding,
        coding=lb.cssd.CSSD_SPEC.coding,
        ring_seed=int(ring_seed),
    )
    return model


# ---------------------------------------------------------------------------
# parameter audit
# ---------------------------------------------------------------------------
def typed_cycle_parameter_audit(model: TypedCycleSEM108) -> dict[str, Any]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    ring = int(sum(parameter.numel() for parameter in model.ring_encoder.parameters()))
    reader = int(sum(parameter.numel() for parameter in model.reader.parameters()))
    parent_reader = int(sc._reader_count(SMALL_READER_IN, READER_HIDDEN[0]))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "total_parameters": total,
        "expected_parameters": EXPECTED_PARAMETERS,
        "parameter_exact": bool(total == EXPECTED_PARAMETERS),
        "ring_encoder_parameters": ring,
        "expected_ring_encoder_parameters": RING_ENCODER_PARAMETERS,
        "reader_parameters": reader,
        "small_reader_parameters": parent_reader,
        "reader_increment": int(reader - parent_reader),
        "expected_reader_increment": READER_INCREMENT,
        "ring_encoder_exact": bool(ring == RING_ENCODER_PARAMETERS),
        "reader_increment_exact": bool(reader - parent_reader == READER_INCREMENT),
        "new_task_dictionary_parameters": 0,
        "shared_bridge_atoms": int(model.local_dictionary_bridge.n_atoms),
        "ring_pool_dim": int(RING_POOL_DIM),
        "typed_reader_input": int(TYPED_READER_IN),
    }
    return payload


__all__ = [
    "PROTOCOL_VERSION",
    "MAX_RING_LENGTH",
    "RING_INPUT_DIM",
    "RING_HIDDEN",
    "RING_OUT",
    "RING_POOL_DIM",
    "SMALL_READER_IN",
    "TYPED_READER_IN",
    "READER_HIDDEN",
    "RING_ENCODER_PARAMETERS",
    "READER_INCREMENT",
    "SMALL_PARAMETERS",
    "EXPECTED_PARAMETERS",
    "RING_VIEW_FIELDS",
    "RING_CYCLE_FIELDS",
    "official_test_blocker",
    "dihedral_view_components",
    "assemble_views",
    "typed_cycle_collate",
    "typed_cycle_loader",
    "TypedCycleSEM108",
    "build_typed_cycle_model",
    "typed_cycle_parameter_audit",
]