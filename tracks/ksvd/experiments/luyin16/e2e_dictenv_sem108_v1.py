"""E2E-DictEnv-Sem108-v1 — Shell-Resolved Primitive Semantic Interface (core).

Round ``e2e_dictenv_sem108_v1`` (Workstream Z, ZINC).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_sem108_v1_preregistration.md``.
Prior-artifact audit: ``tracks/ksvd/notes/e2e_dictenv_sem108_v1_prior_artifact_audit.md``.

Scientific question.  Every previous CSSD / DictEnv arm compressed the
primitive chemistry of a patch into a coarse 62-D anchor and then squeezed that
anchor through ``anchor_encoder: 62 -> 32 -> 32`` before the environment fusion.
This round asks whether the bottleneck is that **early semantic compression**
rather than the dictionary fusion function.

Single architectural change over the frozen parent ``CSSD-q1 + C6``:

    OLD:  anchor62 -> anchor_encoder -> 32D -> fusion(368 -> 128 -> 48)
    NEW:  [Sem108 ; size2] (110D, raw standardized blocks) -> fusion(446 -> H -> 48)

where ``Sem108`` is the shell-resolved primitive semantic interface of the
FEC-S0 ``patch_cont`` descriptor:

    Sem108 = atom_shell [3 x 28 = 84D] ; bond_shell [6 x 4 = 24D]
    size2  = the parent anchor's ``[log1p|V|, log1p|E|]`` block, retained
    Sem110 = [Sem108 ; size2]                       (direct environment interface)

The old anchor compression path (anchor62 -> anchor_encoder -> 32D) is removed.
Everything else is the frozen parent: common coordinate, residual sparse
dictionary, IHT-10, ``K=32``, ``s=8``, node/edge chemistry bindings, 3 shells,
6 shellpairs, C6 masks, relation / distance / global / topology backend,
moments, reader, lambda, optimizer and Top-5 soup protocol.  There is **no**
RNDB (``psi_A`` / ``psi_E`` are absent), no message passing, no new fine
structural descriptor, no ``root_atom`` / ``incident_bonds`` / topology
scalars as new inputs.

CPU only.  ``official_test_loaded = False`` in every payload.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2

PROTOCOL_VERSION = "e2e_dictenv_sem108_v1"

#: FEC-S0 / ``zinc_patch_path_pooling`` shell descriptor geometry (resolved from
#: the current implementation in :func:`resolve_sem108_geometry`).
PATCH_CONT_DIM = 146
SEM_ATOM_BLOCK = (0, 84)  # atom_shell: (radius + 1) x 28
SEM_BOND_BLOCK = (84, 108)  # bond_shell: 6 shellpairs x 4
SEM_DIM = 108
SIZE2_BLOCK = audit.ANCHOR_GROUPS["size"]  # (60, 62)
SEM_INTERFACE_DIM = SEM_DIM + 2  # 110

#: forbidden T1 blocks (must never be exposed as new inputs this round).
FORBIDDEN_T1_BLOCKS = {
    "root_atom": (108, 136),
    "incident_bonds": (136, 140),
    "topology_scalars": (140, 146),
}

#: parameter-matching contract (pre-registration section 6).
PARAM_RATIO_BOUND = 0.02
PARAM_RATIO_PREFERRED = 0.01

#: relation-audit tolerance (float32 standardization round-trip, see notes).
RELATION_TOL = 1.0e-5
ROUND_TRIP_TOL = 1.0e-4


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


def cpu_only_guard(device: Any) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise RuntimeError(f"Sem108 round is CPU-only, got device={resolved}")


# ---------------------------------------------------------------------------
# exact Sem108 geometry resolved from the current FEC-S0 implementation
# ---------------------------------------------------------------------------


def resolve_sem108_geometry() -> dict[str, Any]:
    """Resolve the Sem108 column semantics from the current implementation.

    Reads the FEC-S0 shell-descriptor block table, the compact-v4 shell
    descriptor constructor and the frozen anchor layout, then checks they all
    agree.  Raises on any mismatch.  This is audit A1's code-level anchor.
    """
    from tracks.ksvd.experiments.luyin16 import fec_s0_factorization as fec
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

    radius = int(zpp.PATCH_RADIUS)
    atom_categories = int(zpp.ATOM_CATEGORIES)
    bond_categories = int(zpp.BOND_CATEGORIES)
    shell_pairs = tuple(tuple(int(v) for v in pair) for pair in zpp.SHELL_PAIRS)
    shell_width = int(zpp.SHELL_WIDTH)
    blocks = {str(k): (int(v[0]), int(v[1])) for k, v in fec.SHELL_BLOCKS.items()}

    atom_cols = (radius + 1) * atom_categories
    bond_cols = len(shell_pairs) * bond_categories
    expected = {
        "atom_shell": (0, atom_cols),
        "bond_shell": (atom_cols, atom_cols + bond_cols),
        "root_atom": (atom_cols + bond_cols, atom_cols + bond_cols + atom_categories),
        "incident_bonds": (
            atom_cols + bond_cols + atom_categories,
            atom_cols + bond_cols + atom_categories + bond_categories,
        ),
        "scalars": (
            atom_cols + bond_cols + atom_categories + bond_categories,
            shell_width,
        ),
    }
    if blocks != expected:
        raise RuntimeError(f"FEC-S0 shell block layout changed: {blocks} != {expected}")
    if shell_width != PATCH_CONT_DIM:
        raise RuntimeError(f"shell width {shell_width} != {PATCH_CONT_DIM}")
    if (atom_cols, bond_cols) != (84, 24):
        raise RuntimeError(f"Sem108 geometry changed: atom={atom_cols}, bond={bond_cols}")
    size_low, size_high = audit.ANCHOR_GROUPS["size"]
    if (int(size_low), int(size_high)) != SIZE2_BLOCK:
        raise RuntimeError("anchor size block drifted")
    return {
        "patch_cont_dim": int(shell_width),
        "patch_radius": int(radius),
        "atom_categories": int(atom_categories),
        "bond_categories": int(bond_categories),
        "shell_pairs": [list(pair) for pair in shell_pairs],
        "n_shells": int(radius + 1),
        "n_shellpairs": int(len(shell_pairs)),
        "atom_shell_cols": int(atom_cols),
        "bond_shell_cols": int(bond_cols),
        "sem_dim": int(atom_cols + bond_cols),
        "blocks": {k: list(v) for k, v in blocks.items()},
        "size2_block": [int(SIZE2_BLOCK[0]), int(SIZE2_BLOCK[1])],
    }


def sem108_block(patch_cont: torch.Tensor) -> torch.Tensor:
    """The exact 108-D shell-resolved semantic interface (columns 0..107)."""
    if int(patch_cont.shape[1]) != PATCH_CONT_DIM:
        raise RuntimeError(
            f"patch_cont width {int(patch_cont.shape[1])} != {PATCH_CONT_DIM}"
        )
    return patch_cont[:, 0:SEM_DIM]


def sem110_interface(patch_cont: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """``[Sem108 ; size2]`` — the candidate's direct environment interface."""
    if int(anchor.shape[1]) != int(audit.ANCHOR_DIM_EXPECTED):
        raise RuntimeError(
            f"anchor width {int(anchor.shape[1])} != {audit.ANCHOR_DIM_EXPECTED}"
        )
    return torch.cat(
        [sem108_block(patch_cont), anchor[:, int(SIZE2_BLOCK[0]) : int(SIZE2_BLOCK[1])]],
        dim=1,
    )


def forbidden_block_columns(patch_cont: torch.Tensor) -> dict[str, torch.Tensor]:
    """The T1 blocks that must not become new model inputs (audit/tests only)."""
    return {
        name: patch_cont[:, low:high].clone()
        for name, (low, high) in FORBIDDEN_T1_BLOCKS.items()
    }


# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SEMMask(audit.AuditMask):
    """Frozen-parent ``AuditMask`` plus the Sem108 interface interventions.

    * ``sem_block_zero``   — zero the whole 108-D standardized Sem block
      (standardized zero == train-mean neutralization); ``size2`` retained.
    * ``sem_atom_zero``    — zero only ``atom_shell`` (columns 0..83).
    * ``sem_bond_zero``    — zero only ``bond_shell`` (columns 84..107).
    * ``use_sem_row_shuffle`` / ``sem_shuffle_seed`` — permute the whole Sem108
      row across the roots of each molecule, keeping the per-graph multiset and
      keeping ``size2`` with its root.
    * ``size2_zero``       — zero the retained size2 block.
    * ``residual_zero``    — residual sparse ``alpha -> 0`` (common kept).
    """

    sem_block_zero: bool = False
    sem_atom_zero: bool = False
    sem_bond_zero: bool = False
    use_sem_row_shuffle: bool = False
    sem_shuffle_seed: int = 0
    size2_zero: bool = False
    residual_zero: bool = False

    def is_identity(self) -> bool:
        if any(
            (
                self.sem_block_zero,
                self.sem_atom_zero,
                self.sem_bond_zero,
                self.use_sem_row_shuffle,
                self.size2_zero,
                self.residual_zero,
            )
        ):
            return False
        return super().is_identity()

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload["is_identity"] = self.is_identity()
        return payload


def _as_sem_mask(mask: audit.AuditMask | None) -> "SEMMask":
    """Coerce any frozen ``AuditMask`` (e.g. ``cm.C6_MASK``) into a ``SEMMask``."""
    if mask is None:
        return SEMMask()
    if isinstance(mask, SEMMask):
        return mask
    fields = {field.name for field in dataclasses.fields(audit.AuditMask)}
    return SEMMask(**{name: getattr(mask, name) for name in fields})


def merge_sem_mask(
    base: audit.AuditMask | None, probe: SEMMask | None
) -> SEMMask:
    """Union of the frozen training mask (C6) and a Sem108 probe mask."""
    base = _as_sem_mask(base)
    probe = _as_sem_mask(probe)
    return SEMMask(
        anchor_zero_groups=tuple(sorted(set(base.anchor_zero_groups) | set(probe.anchor_zero_groups))),
        global_zero_groups=tuple(sorted(set(base.global_zero_groups) | set(probe.global_zero_groups))),
        topology_zero=bool(base.topology_zero or probe.topology_zero),
        node_binding_zero=bool(base.node_binding_zero or probe.node_binding_zero),
        edge_binding_zero=bool(base.edge_binding_zero or probe.edge_binding_zero),
        use_node_shuffle=bool(base.use_node_shuffle or probe.use_node_shuffle),
        use_edge_shuffle=bool(base.use_edge_shuffle or probe.use_edge_shuffle),
        coord_zero=bool(base.coord_zero or probe.coord_zero),
        relation_zero_groups=tuple(
            sorted(set(base.relation_zero_groups) | set(probe.relation_zero_groups))
        ),
        unary_zero_blocks=tuple(sorted(set(base.unary_zero_blocks) | set(probe.unary_zero_blocks))),
        pair_zero_blocks=tuple(sorted(set(base.pair_zero_blocks) | set(probe.pair_zero_blocks))),
        graph_hidden_zero=bool(base.graph_hidden_zero or probe.graph_hidden_zero),
        gate_off=bool(base.gate_off or probe.gate_off),
        pair_projection_zero=bool(base.pair_projection_zero or probe.pair_projection_zero),
        sem_block_zero=bool(base.sem_block_zero or probe.sem_block_zero),
        sem_atom_zero=bool(base.sem_atom_zero or probe.sem_atom_zero),
        sem_bond_zero=bool(base.sem_bond_zero or probe.sem_bond_zero),
        use_sem_row_shuffle=bool(base.use_sem_row_shuffle or probe.use_sem_row_shuffle),
        sem_shuffle_seed=int(
            probe.sem_shuffle_seed if probe.use_sem_row_shuffle else base.sem_shuffle_seed
        ),
        size2_zero=bool(base.size2_zero or probe.size2_zero),
        residual_zero=bool(base.residual_zero or probe.residual_zero),
    )


# ---------------------------------------------------------------------------
# closed-form parameter matching
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden: int, out_dim: int, *, bias: bool = True) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden), bias=bias),
        nn.SiLU(),
        nn.Linear(int(hidden), int(out_dim), bias=bias),
    )


def _mlp_parameter_count(in_dim: int, hidden: int, out_dim: int, *, bias: bool = True) -> int:
    count = int(in_dim) * int(hidden) + int(hidden) * int(out_dim)
    if bias:
        count += int(hidden) + int(out_dim)
    return count


def parent_local_interface_parameters() -> dict[str, int]:
    """Exact parent local-interface budget: anchor encoder + fusion MLP."""
    layout = p2.decoder_layers(cm.H1_CONFIG)
    anchor_in, anchor_out = layout["anchor"]
    fusion_in, fusion_hidden, fusion_out = (
        layout["fusion_in"],
        layout["fusion_hidden"],
        layout["out"],
    )
    anchor_params = _mlp_parameter_count(anchor_in, anchor_out, anchor_out)
    fusion_params = _mlp_parameter_count(fusion_in, fusion_hidden, fusion_out)
    return {
        "anchor_encoder_params": int(anchor_params),
        "fusion_params": int(fusion_params),
        "local_interface_params": int(anchor_params + fusion_params),
        "anchor_in": int(anchor_in),
        "anchor_out": int(anchor_out),
        "fusion_in": int(fusion_in),
        "fusion_hidden": int(fusion_hidden),
        "fusion_out": int(fusion_out),
    }


def candidate_fusion_input_dim() -> int:
    """``Sem110 + 3 node slots(48) + 6 edge slots(32)`` from the frozen parent."""
    layout = p2.decoder_layers(cm.H1_CONFIG)
    node_out = int(layout["node"][2])
    edge_out = int(layout["edge"][2])
    return int(SEM_INTERFACE_DIM + int(p2.N_SHELLS) * node_out + int(p2.SHELLPAIR_CLASSES) * edge_out)


def closed_form_fusion_hidden() -> dict[str, Any]:
    """One-shot parameter matching of the candidate fusion hidden width.

    The candidate interface keeps the parent node/edge encoders and removes the
    anchor encoder.  Only the fusion width ``H`` is free:
    ``P_candidate(H) = (sem_in + 1 + out) H + out``.  Choose the integer ``H``
    minimizing ``|P_candidate_total - P_parent_total| / P_parent_total`` (all
    other parameters identical, so this equals matching the local interface).
    """
    parent = parent_local_interface_parameters()
    sem_in = candidate_fusion_input_dim()
    out = int(parent["fusion_out"])
    target = int(parent["local_interface_params"])
    slope = int(sem_in + 1 + out)
    h_star = (target - out) / slope
    best_h, best_gap = None, None
    for h in range(max(int(math.floor(h_star)) - 4, 1), int(math.ceil(h_star)) + 5):
        gap = abs(_mlp_parameter_count(sem_in, h, out) - target)
        if best_gap is None or gap < best_gap:
            best_h, best_gap = int(h), int(gap)
    assert best_h is not None and best_gap is not None
    return {
        "sem_fusion_in": int(sem_in),
        "sem_fusion_hidden": int(best_h),
        "sem_fusion_out": int(out),
        "sem_fusion_params": int(_mlp_parameter_count(sem_in, best_h, out)),
        "parent_local_interface_params": int(target),
        "closed_form_h_star": float(h_star),
        "interface_gap": int(best_gap),
    }


CLOSED_FORM = closed_form_fusion_hidden()
SEM_FUSION_IN = int(CLOSED_FORM["sem_fusion_in"])
SEM_FUSION_HIDDEN = int(CLOSED_FORM["sem_fusion_hidden"])
SEM_FUSION_OUT = int(CLOSED_FORM["sem_fusion_out"])
PARENT_FUSION_HIDDEN = int(parent_local_interface_parameters()["fusion_hidden"])


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class SEM108Model(cssd.CSSDModel):
    """CSSD-q1 + C6 with the direct shell-resolved semantic environment interface.

    * the anchor encoder / 32-D compressed anchor path is removed;
    * ``z_i = [Sem108_i ; size2_i]`` (110-D, standardized blocks) enters the
      fusion MLP directly, together with the frozen node (3x48) and edge (6x32)
      encoded slots;
    * the fusion hidden width is the closed-form parameter-matched ``H = 114``
      (``input -> H -> 48``, SiLU, no norm / dropout / extra layer);
    * with ``parent_interface=True`` and ``fusion_hidden=128`` plus the
      retained anchor encoder the forward reproduces the parent bit-for-bit
      (structural equivalence gate).
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        fusion_hidden: int = SEM_FUSION_HIDDEN,
        keep_anchor_encoder: bool = False,
        parent_interface: bool = False,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        if self.node_binding != "paired" or self.edge_binding != "paired":
            raise ValueError("Sem108 requires the paired node and edge bindings")
        self.sem_fusion_hidden = int(fusion_hidden)
        self.keep_anchor_encoder = bool(keep_anchor_encoder)
        self.parent_interface = bool(parent_interface)
        if self.parent_interface and not self.keep_anchor_encoder:
            raise ValueError("parent_interface requires keep_anchor_encoder=True")
        if not self.keep_anchor_encoder:
            # Remove the old anchor62 -> anchor_encoder -> 32D compression path.
            self.anchor_encoder = None
        # The only architectural width change: parameter-matched fusion.
        if self.parent_interface:
            width = int(parent_local_interface_parameters()["fusion_in"])
        else:
            width = SEM_FUSION_IN
        self.fusion = _mlp(width, self.sem_fusion_hidden, int(p2.ENV_DIM))

    # -- interface -------------------------------------------------------------

    def semantic_interface(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: audit.AuditMask | None = None,
        fill: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``[Sem108 ; size2]`` with the frozen mask / intervention semantics."""
        mask = SEMMask() if mask is None else mask
        n = int(coord.shape[0])
        patch = data.patch_cont
        if int(patch.shape[1]) != PATCH_CONT_DIM:
            raise RuntimeError(
                f"patch_cont width {int(patch.shape[1])} != {PATCH_CONT_DIM}"
            )
        sem = patch[:, 0:SEM_DIM].to(coord.dtype)
        if bool(getattr(mask, "sem_block_zero", False)):
            sem = torch.zeros_like(sem)
        else:
            if bool(getattr(mask, "sem_atom_zero", False)):
                sem = sem.clone()
                sem[:, int(SEM_ATOM_BLOCK[0]) : int(SEM_ATOM_BLOCK[1])] = 0.0
            if bool(getattr(mask, "sem_bond_zero", False)):
                sem = sem.clone()
                sem[:, int(SEM_BOND_BLOCK[0]) : int(SEM_BOND_BLOCK[1])] = 0.0
        if bool(getattr(mask, "use_sem_row_shuffle", False)):
            sem = self._shuffle_sem_rows(sem, data, int(getattr(mask, "sem_shuffle_seed", 0)))

        anchor = data.anchor.to(coord.dtype)
        if int(anchor.shape[1]) != int(audit.ANCHOR_DIM_EXPECTED):
            raise RuntimeError(
                f"anchor width {int(anchor.shape[1])} != {audit.ANCHOR_DIM_EXPECTED}"
            )
        size2 = anchor[:, int(SIZE2_BLOCK[0]) : int(SIZE2_BLOCK[1])]
        anchor_groups = tuple(getattr(mask, "anchor_zero_groups", ()))
        if bool(getattr(mask, "size2_zero", False)) or "size" in anchor_groups:
            key = "anchor:size"
            if fill is not None and key in fill:
                size2 = (
                    fill[key]
                    .to(device=size2.device, dtype=size2.dtype)
                    .reshape(1, 2)
                    .expand(n, 2)
                    .contiguous()
                )
            else:
                size2 = torch.zeros_like(size2)
        return torch.cat([sem, size2], dim=1)

    def _shuffle_sem_rows(self, sem: torch.Tensor, data: Any, seed: int) -> torch.Tensor:
        """Permute the whole Sem108 row across the roots of each molecule."""
        batch = getattr(data, "batch", None)
        generator = torch.Generator().manual_seed(int(seed))
        n_rows = int(sem.shape[0])
        if batch is None:
            return sem[torch.randperm(n_rows, generator=generator)]
        n_graphs = int(batch.max().item()) + 1 if int(batch.numel()) else 0
        index = torch.empty(n_rows, dtype=torch.long)
        for graph in range(n_graphs):
            rows = (batch == graph).nonzero(as_tuple=False).view(-1)
            index[rows] = rows[torch.randperm(int(rows.numel()), generator=generator)]
        return sem[index]

    # -- environment -----------------------------------------------------------

    def _edge_response_delta(
        self,
        coord: torch.Tensor,
        data: Any,
        bond_u: torch.Tensor,
        bond_v: torch.Tensor,
        mask: audit.AuditMask | None = None,
        *,
        edge_binding_zero: bool = False,
    ) -> torch.Tensor | None:
        """Optional additive per-bond edge-response residual (default: none).

        A subclass may return a ``[n_bond, d_e]`` tensor which is summed into the
        per-bond edge response **before** the shellpair ``index_add_``
        aggregation.  Returning ``None`` (the parent behaviour) leaves the frozen
        forward bit-identical; no other code path reads this hook.
        """
        return None

    def _edge_env_parts(
        self,
        coord: torch.Tensor,
        data: Any,
        bond_u: torch.Tensor,
        bond_v: torch.Tensor,
        *,
        edge_binding_zero: bool = False,
        mask: audit.AuditMask | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(parent_edge_response, hook_delta)`` for every real bond."""
        d_e = int(self.config.d_e)
        cu = coord[bond_u]
        cv = coord[bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        if edge_binding_zero:
            g = torch.zeros_like(g)
        b = F.one_hot(data.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(coord.dtype)
        ue = (g @ self.W_E_S) * (b @ self.W_E_C) / math.sqrt(float(d_e))
        return ue, self._edge_response_delta(
            coord, data, bond_u, bond_v, mask, edge_binding_zero=bool(edge_binding_zero)
        )

    def _edge_env_slots(
        self,
        coord: torch.Tensor,
        data: Any,
        bond_u: torch.Tensor,
        bond_v: torch.Tensor,
        *,
        edge_binding_zero: bool = False,
        mask: audit.AuditMask | None = None,
    ) -> torch.Tensor:
        """Per-(root, shellpair) edge slots ``[n, SHELLPAIR_CLASSES, d_e]``."""
        n = int(coord.shape[0])
        d_e = int(self.config.d_e)
        ue, delta = self._edge_env_parts(
            coord, data, bond_u, bond_v, edge_binding_zero=edge_binding_zero, mask=mask
        )
        if delta is not None:
            ue = ue + delta
        flat_e = torch.zeros(
            (n * int(p2.SHELLPAIR_CLASSES), d_e), device=ue.device, dtype=ue.dtype
        )
        flat_e.index_add_(
            0,
            data.env_bond_root.to(coord.device) * int(p2.SHELLPAIR_CLASSES)
            + data.env_bond_shellpair.to(coord.device),
            ue,
        )
        return flat_e.view(n, int(p2.SHELLPAIR_CLASSES), d_e)

    def _environment_from_parts(
        self,
        coord: torch.Tensor,
        data: Any,
        interface: torch.Tensor,
        *,
        occ_coord_node: torch.Tensor | None = None,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
        node_binding_zero: bool = False,
        edge_binding_zero: bool = False,
        mask: audit.AuditMask | None = None,
    ) -> torch.Tensor:
        """Frozen parent slot construction + the semantic-interface fusion."""
        n = int(coord.shape[0])
        q = F.one_hot(data.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        if occ_coord_node is None:
            occ_coord_node = data.env_occ_node
        occ_coord_node = occ_coord_node.to(coord.device)
        c = coord[occ_coord_node]
        qc = q[data.env_occ_node.to(coord.device)]
        u = self._node_binding(c, qc)
        if node_binding_zero:
            u = torch.zeros_like(u)
        flat = torch.zeros(
            (n * int(p2.N_SHELLS), int(p2.D_A)), device=u.device, dtype=u.dtype
        )
        flat.index_add_(
            0,
            data.env_occ_root.to(coord.device) * int(p2.N_SHELLS)
            + data.env_occ_shell.to(coord.device),
            u,
        )
        node_slots = flat.view(n, int(p2.N_SHELLS), int(p2.D_A))

        if bond_u is None:
            bond_u = data.env_bond_u
        if bond_v is None:
            bond_v = data.env_bond_v
        bond_u = bond_u.to(coord.device)
        bond_v = bond_v.to(coord.device)
        edge_slots = self._edge_env_slots(
            coord,
            data,
            bond_u,
            bond_v,
            edge_binding_zero=bool(edge_binding_zero),
            mask=mask,
        )

        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        if self.parent_interface:
            if self.anchor_encoder is None:
                raise RuntimeError("parent_interface requires the retained anchor encoder")
            interface = self.anchor_encoder(data.anchor.to(coord.dtype))
        fused = torch.cat(
            [interface, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1
        )
        return self.fusion(fused)

    def _node_binding(self, c: torch.Tensor, qc: torch.Tensor) -> torch.Tensor:
        """Frozen multiplicative structure x atom-semantics node binding.

        Named hook so a subclass can substitute an equivalent binding (e.g. a
        fixed-amplitude additive residual) without copying the rest of
        ``_environment_from_parts``.  The default expression is bit-identical
        to the historical inline product.
        """
        return (c @ self.W_A_S) * (qc @ self.W_A_C) / math.sqrt(float(p2.D_A))

    def environments(self, coord: torch.Tensor, data: Any) -> torch.Tensor:
        """Unmasked path (used by diagnostics); same semantics as masked C6-less."""
        interface = self.semantic_interface(coord, data, None, None)
        return self._environment_from_parts(coord, data, interface)

    def environments_masked(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: audit.AuditMask,
        fill: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        mask = SEMMask() if mask is None else mask
        if bool(getattr(mask, "residual_zero", False)):
            coord = coord.clone()
            coord[:, self.common_dim :] = 0.0
        interface = self.semantic_interface(coord, data, mask, fill)
        occ_coord_node = None
        if bool(getattr(mask, "use_node_shuffle", False)):
            occ_coord_node = getattr(data, "env_occ_coord_node", None)
            if occ_coord_node is None:
                raise RuntimeError("node-shuffle intervention requires data.env_occ_coord_node")
        bond_u = bond_v = None
        if bool(getattr(mask, "use_edge_shuffle", False)):
            bond_u = getattr(data, "env_bond_u_shuffled", None)
            bond_v = getattr(data, "env_bond_v_shuffled", None)
            if bond_u is None or bond_v is None:
                raise RuntimeError(
                    "edge-shuffle intervention requires shuffled endpoint fields"
                )
        return self._environment_from_parts(
            coord,
            data,
            interface,
            occ_coord_node=occ_coord_node,
            bond_u=bond_u,
            bond_v=bond_v,
            node_binding_zero=bool(getattr(mask, "node_binding_zero", False)),
            edge_binding_zero=bool(getattr(mask, "edge_binding_zero", False)),
            mask=mask,
        )


def build_sem108_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    spec: cm.CleanMechSpec | None = None,
) -> SEM108Model:
    """Build the candidate with the exact parent RNG stream for all shared params."""
    spec = cssd.CSSD_SPEC if spec is None else spec
    torch.manual_seed(int(seed))
    return SEM108Model(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        node_binding=spec.node_binding,
        edge_binding=spec.edge_binding,
        coding=spec.coding,
    )


def build_parent_equivalence_model(
    dictionary: np.ndarray, seed: int, subspace: cssd.CommonSubspace
) -> SEM108Model:
    """A Sem108 model with the parent interface retained for bit-identity tests."""
    torch.manual_seed(int(seed))
    return SEM108Model(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        node_binding="paired",
        edge_binding="paired",
        coding="sparse",
        fusion_hidden=int(parent_local_interface_parameters()["fusion_hidden"]),
        keep_anchor_encoder=True,
        parent_interface=True,
    )


# ---------------------------------------------------------------------------
# parameter audit
# ---------------------------------------------------------------------------


def parameter_audit(model: SEM108Model, parent_params: int) -> dict[str, Any]:
    parent_interface = parent_local_interface_parameters()
    total = int(sum(p.numel() for p in model.parameters()))
    delta = int(total - int(parent_params))
    ratio = float(abs(delta) / max(int(parent_params), 1))
    anchor_params = int(parent_interface["anchor_encoder_params"])
    fusion_parent = int(parent_interface["fusion_params"])
    fusion_candidate = int(_mlp_parameter_count(SEM_FUSION_IN, int(model.sem_fusion_hidden), SEM_FUSION_OUT))
    return {
        "parent_params": int(parent_params),
        "candidate_params": total,
        "delta": delta,
        "relative_delta": ratio,
        "parent_anchor_encoder_params": anchor_params,
        "parent_fusion_params": fusion_parent,
        "parent_local_interface_params": int(parent_interface["local_interface_params"]),
        "candidate_fusion_params": fusion_candidate,
        "candidate_local_interface_params": fusion_candidate,
        "closed_form": dict(CLOSED_FORM),
        "sem_fusion_hidden": int(model.sem_fusion_hidden),
        "sem_fusion_in": int(SEM_FUSION_IN),
        "parameter_ratio_within_bound": bool(ratio <= PARAM_RATIO_BOUND),
        "parameter_ratio_preferred": bool(ratio <= PARAM_RATIO_PREFERRED),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# audit A1 — Sem108 identity / standardization provenance
# ---------------------------------------------------------------------------


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def audit_sem108_identity(
    *,
    n_train: int = 64,
    n_valid: int = 64,
    zinc_root: Any = None,
) -> dict[str, Any]:
    """Audit A1: exact Sem108 identity + bit-identical train-fit standardization."""
    from tracks.ksvd.experiments.luyin16 import fec_s0_factorization as fec
    from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
    from tracks.ksvd.experiments.luyin16 import zinc_post_v4_residual_audit as pva
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
    from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp

    geometry = resolve_sem108_geometry()
    train_records, valid_records = pva._extract_v4_records()
    fits = fec.build_fits(train_records)
    scaler = fits.patch_standardizer

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "geometry": geometry,
        "standardization": {
            "fit_split": "official train",
            "n_fit_records": int(len(train_records)),
            "n_fit_patches": int(zpp._patch_matrix(train_records).shape[0]),
            "mean_sha256": _sha256_array(scaler.mean),
            "scale_sha256": _sha256_array(scaler.scale),
            "mean_first_8": [float(v) for v in scaler.mean[:8]],
            "scale_first_8": [float(v) for v in scaler.scale[:8]],
            "valid_in_fit": False,
        },
        "splits": {},
    }

    if zinc_root is None:
        from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

        zinc_root = REPO_ROOT / "data/ZINC"

    for split, zinc_split, subset in (
        ("train", "train", int(n_train)),
        ("valid", "val", int(n_valid)),
    ):
        train_data, valid_data, _audit = sdp.load_encoded(
            train_subset=subset if split == "train" else None,
            valid_subset=subset if split == "valid" else None,
        )
        encoded = train_data if split == "train" else valid_data
        raw_molecules = list(zlr._load_zinc(zinc_root, zinc_split))[: int(subset)]
        records = train_records if split == "train" else valid_records
        records = list(records)[: int(subset)]
        transform = fec.FactorizedFeatureTransform(fits)
        n_patches = 0
        molecules_bit_identical = 0
        max_cache_diff = 0.0
        semantics_exact = 0
        semantics_max_diff = 0.0
        for raw, record, cached in zip(raw_molecules, records, list(encoded)):
            fact = fec.factorized_record(raw, tokenize=True)
            y = float(raw.y.view(-1)[0])
            built = transform.build(
                raw, y, topology_raw=record.topology_features, raw=fact
            )
            reference = cached.patch_cont
            candidate = built.patch_cont
            n_patches += int(reference.shape[0])
            if torch.equal(reference, candidate):
                molecules_bit_identical += 1
            max_cache_diff = max(
                max_cache_diff, float((reference - candidate).abs().max().item())
            )
            graph, node_types, edge_types = zlr._data_to_graph(raw)
            for center in range(int(raw.num_nodes)):
                distances = zpp._ego_distances(graph, int(center), int(geometry["patch_radius"]))
                raw_descriptor = fec.factorized_shell_descriptor(
                    graph,
                    int(center),
                    node_types,
                    edge_types,
                    distances,
                    int(geometry["patch_radius"]),
                )
                historical, _nodes, _boundary = zpp._shell_descriptor(
                    graph,
                    int(center),
                    node_types,
                    edge_types,
                    distances,
                    int(geometry["patch_radius"]),
                )
                if not np.array_equal(raw_descriptor, historical):
                    raise RuntimeError(
                        f"factorized vs historical shell descriptor differ ({split}, mol, {center})"
                    )
                atom_shell = raw_descriptor[
                    int(SEM_ATOM_BLOCK[0]) : int(SEM_ATOM_BLOCK[1])
                ].reshape(int(geometry["n_shells"]), int(geometry["atom_categories"]))
                bond_shell = raw_descriptor[
                    int(SEM_BOND_BLOCK[0]) : int(SEM_BOND_BLOCK[1])
                ].reshape(int(geometry["n_shellpairs"]), int(geometry["bond_categories"]))
                # independent expected blocks
                nodes = sorted(distances)
                n_nodes = len(nodes)
                induced = graph.induced(set(nodes))
                n_edges = induced.num_edges()
                expected_atom = np.zeros_like(atom_shell)
                for node in nodes:
                    expected_atom[
                        int(distances[node]), int(node_types[node])
                    ] += 1.0
                expected_atom /= max(float(n_nodes), 1.0)
                pair_index = {
                    tuple(int(v) for v in pair): index
                    for index, pair in enumerate(geometry["shell_pairs"])
                }
                expected_bond = np.zeros_like(bond_shell)
                for left, right in induced.edges():
                    pair = tuple(sorted((int(distances[left]), int(distances[right]))))
                    expected_bond[pair_index[pair], int(edge_types[graph.edge_key(int(left), int(right))])] += 1.0
                expected_bond /= max(float(n_edges), 1.0)
                diff = max(
                    float(np.abs(atom_shell - expected_atom).max()),
                    float(np.abs(bond_shell - expected_bond).max()),
                )
                semantics_max_diff = max(semantics_max_diff, diff)
                if diff == 0.0:
                    semantics_exact += 1
        payload["splits"][split] = {
            "n_molecules": int(len(raw_molecules)),
            "n_patches": int(n_patches),
            "molecules_bit_identical_to_cache": int(molecules_bit_identical),
            "cache_max_abs_diff": float(max_cache_diff),
            "cache_bit_identical": bool(molecules_bit_identical == len(raw_molecules)),
            "semantics_exact_patches": int(semantics_exact),
            "semantics_max_abs_diff": float(semantics_max_diff),
        }

    # cached train column statistics confirm the train-fit standardization.
    del train_data, valid_data, encoded
    gc.collect()
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run

    train_list = p1run.load_split("train", subset=1024)
    patch = torch.cat([d.patch_cont for d in train_list], dim=0).numpy().astype(np.float64)
    payload["standardization"]["cached_train_mean_absmax"] = float(np.abs(patch.mean(axis=0)).max())
    payload["standardization"]["cached_train_std_dev_from_one_absmax"] = float(
        np.abs(patch.std(axis=0) - 1.0).max()
    )
    payload["standardization"]["cached_sample_rows"] = int(patch.shape[0])
    del train_list, patch
    gc.collect()

    payload["atom_shell_cols"] = int(geometry["atom_shell_cols"])
    payload["bond_shell_cols"] = int(geometry["bond_shell_cols"])
    payload["sem_dim"] = int(geometry["sem_dim"])
    payload["identity_established"] = bool(
        geometry["sem_dim"] == 108
        and payload["splits"]["train"]["cache_bit_identical"]
        and payload["splits"]["valid"]["cache_bit_identical"]
        and payload["splits"]["train"]["semantics_max_abs_diff"] == 0.0
        and payload["splits"]["valid"]["semantics_max_abs_diff"] == 0.0
    )
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# audit A2 / A4 — anchor62 relationship + redundancy / collision structure
# ---------------------------------------------------------------------------


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def audit_anchor_relation_and_redundancy() -> dict[str, Any]:
    """Audits A2 + A4 on the full official train + valid feature caches.

    A2: exact deterministic relations ``Sem108 -> parent anchor62`` (root,
    atom mass, bond mass, size2).  A4: deterministic feature-equivalence /
    collision structure of the two representations (no probe training).
    """
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
    from tracks.ksvd.experiments.luyin16 import zinc_post_v4_residual_audit as pva
    from tracks.ksvd.experiments.luyin16 import fec_s0_factorization as fec
    from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as she

    anchor_stats_path = (
        she.REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_p1/anchor_stats.json"
    )
    import json as _json

    anchor_stats = _json.loads(anchor_stats_path.read_text(encoding="utf-8"))
    if str(anchor_stats.get("fit_split")) != "official train":
        raise RuntimeError("anchor standardizer was not fit on official train")
    a_mean = np.asarray(anchor_stats["mean"], dtype=np.float64)
    a_scale = np.asarray(anchor_stats["scale"], dtype=np.float64)
    if a_mean.shape != (62,) or a_scale.shape != (62,):
        raise RuntimeError("anchor stats shape mismatch")

    train_records, _valid_records = pva._extract_v4_records()
    fits = fec.build_fits(train_records)
    p_mean = fits.patch_standardizer.mean.astype(np.float64)
    p_scale = fits.patch_standardizer.scale.astype(np.float64)

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "anchor_standardizer": {
            "artifact": "tracks/ksvd/results/e2e_dictenv_p1/anchor_stats.json",
            "fit_split": str(anchor_stats["fit_split"]),
            "dim": int(anchor_stats["dim"]),
        },
        "patch_standardizer": {
            "fit_split": "official train",
            "mean_sha256": _sha256_array(fits.patch_standardizer.mean),
            "scale_sha256": _sha256_array(fits.patch_standardizer.scale),
        },
        "splits": {},
        "relation_tolerance": RELATION_TOL,
    }

    n_sem_keys_total = 0
    for split in ("train", "valid"):
        data = p1run.load_split(split)
        patch = torch.cat([d.patch_cont for d in data], dim=0).numpy().astype(np.float32)
        anchor = torch.cat([d.anchor for d in data], dim=0).numpy().astype(np.float32)
        n_rows = int(patch.shape[0])
        assert n_rows == int(anchor.shape[0])

        pat_raw = patch.astype(np.float64) * p_scale + p_mean
        anc_raw = anchor.astype(np.float64) * a_scale + a_mean
        atom_shell = pat_raw[:, int(SEM_ATOM_BLOCK[0]) : int(SEM_ATOM_BLOCK[1])].reshape(-1, 3, 28)
        bond_shell = pat_raw[:, int(SEM_BOND_BLOCK[0]) : int(SEM_BOND_BLOCK[1])].reshape(-1, 6, 4)
        root_sem = pat_raw[:, 108:136]
        size_pat = pat_raw[:, 140:142]
        root_anc = anc_raw[:, 0:28]
        atom_mass = anc_raw[:, 28:56]
        bond_mass = anc_raw[:, 56:60]
        size_anc = anc_raw[:, 60:62]
        n_nodes = np.rint(np.expm1(size_pat[:, 0]))
        n_edges = np.rint(np.expm1(size_pat[:, 1]))

        root_err = np.abs(root_sem - root_anc).max(axis=1)
        atom_recon = atom_shell.sum(axis=1) * n_nodes[:, None]
        bond_recon = bond_shell.sum(axis=1) * n_edges[:, None]
        atom_err = np.abs(atom_recon - atom_mass).max(axis=1)
        bond_err = np.abs(bond_recon - bond_mass).max(axis=1)
        size_err = np.abs(size_anc - size_pat).max(axis=1)
        size_bitexact = np.array(
            [bool(np.array_equal(size_anc[i], size_pat[i])) for i in range(n_rows)]
        )

        payload["splits"][split] = {
            "n_rows": n_rows,
            "root": {
                "argmax_agreement": float((root_sem.argmax(1) == root_anc.argmax(1)).mean()),
                "max_abs_error": float(root_err.max()),
                "fraction_exact": float((root_err <= RELATION_TOL).mean()),
                "fraction_exact_1e-6": float((root_err <= 1.0e-6).mean()),
            },
            "atom_mass": {
                "max_abs_error": float(atom_err.max()),
                "mean_abs_error": float(atom_err.mean()),
                "fraction_exact": float((atom_err <= RELATION_TOL).mean()),
                "integer_deviation_max": float(
                    np.abs(atom_mass - np.rint(atom_mass)).max()
                ),
                "n_nodes_integer_deviation_max": float(np.abs(n_nodes - np.expm1(size_pat[:, 0])).max()),
            },
            "bond_mass": {
                "max_abs_error": float(bond_err.max()),
                "mean_abs_error": float(bond_err.mean()),
                "fraction_exact": float((bond_err <= RELATION_TOL).mean()),
                "integer_deviation_max": float(
                    np.abs(bond_mass - np.rint(bond_mass)).max()
                ),
                "n_edges_integer_deviation_max": float(np.abs(n_edges - np.expm1(size_pat[:, 1])).max()),
            },
            "size2": {
                "max_abs_error": float(size_err.max()),
                "fraction_bit_identical": float(size_bitexact.mean()),
            },
            "sem108_to_anchor": {
                "root_recoverable": bool((root_sem.argmax(1) == root_anc.argmax(1)).all()),
                "normalized_atom_mass_recoverable": bool((atom_err <= RELATION_TOL).all()),
                "normalized_bond_mass_recoverable": bool((bond_err <= RELATION_TOL).all()),
            },
        }

        # A4: deterministic equivalence / collision structure.
        sem = patch[:, 0:SEM_DIM]
        sem_keys = sem.view(np.dtype((np.void, sem.dtype.itemsize * SEM_DIM))).ravel()
        sem_q = np.round(sem.astype(np.float64), 4).astype(np.float32)
        sem_q_keys = sem_q.view(np.dtype((np.void, sem_q.dtype.itemsize * SEM_DIM))).ravel()
        anchor_keys = np.empty(
            n_rows,
            dtype=[
                ("root", np.int64),
                ("atom", np.int64, 28),
                ("bond", np.int64, 4),
                ("n", np.int64),
                ("m", np.int64),
            ],
        )
        anchor_keys["root"] = root_anc.argmax(1)
        anchor_keys["atom"] = np.rint(atom_mass).astype(np.int64)
        anchor_keys["bond"] = np.rint(bond_mass).astype(np.int64)
        anchor_keys["n"] = n_nodes.astype(np.int64)
        anchor_keys["m"] = n_edges.astype(np.int64)

        uniq_sem, inv_sem = np.unique(sem_keys, return_inverse=True)
        uniq_q, inv_q = np.unique(sem_q_keys, return_inverse=True)
        uniq_anchor, inv_anchor = np.unique(anchor_keys, return_inverse=True)
        pair_exact = np.stack([inv_anchor, inv_sem], axis=1)
        pair_q = np.stack([inv_anchor, inv_q], axis=1)
        up_exact = np.unique(pair_exact, axis=0)
        up_q = np.unique(pair_q, axis=0)
        counts_exact = np.bincount(up_exact[:, 0], minlength=int(uniq_anchor.shape[0]))
        counts_q = np.bincount(up_q[:, 0], minlength=int(uniq_anchor.shape[0]))
        reverse_exact = np.bincount(up_exact[:, 1], minlength=int(uniq_sem.shape[0]))
        reverse_q = np.bincount(up_q[:, 1], minlength=int(uniq_q.shape[0]))

        payload["splits"][split]["redundancy"] = {
            "n_rows": n_rows,
            "n_distinct_sem108_exact": int(uniq_sem.shape[0]),
            "n_distinct_sem108_round1e4": int(uniq_q.shape[0]),
            "n_distinct_anchor62": int(uniq_anchor.shape[0]),
            "distinct_sem108_per_anchor_exact": _distribution(counts_exact),
            "distinct_sem108_per_anchor_round1e4": _distribution(counts_q),
            "collision_excess_exact": int(up_exact.shape[0] - int(uniq_anchor.shape[0])),
            "collision_excess_round1e4": int(up_q.shape[0] - int(uniq_anchor.shape[0])),
            "fraction_rows_anchor_nonunique_sem_exact": float(
                (counts_exact[inv_anchor] > 1).mean()
            ),
            "fraction_rows_anchor_nonunique_sem_round1e4": float(
                (counts_q[inv_anchor] > 1).mean()
            ),
            "fraction_sem_keys_with_single_anchor_exact": float((reverse_exact == 1).mean()),
            "fraction_sem_keys_with_single_anchor_round1e4": float((reverse_q == 1).mean()),
            "anchor_injective_from_sem108": bool((reverse_exact == 1).all()),
            "anchor_max_sem_per_group_exact": int(counts_exact.max()) if counts_exact.size else 0,
        }
        n_sem_keys_total += int(uniq_sem.shape[0])
        del data, patch, anchor
        gc.collect()

    payload["n_distinct_sem108_keys_total_exact"] = int(n_sem_keys_total)
    payload["a2_pass"] = bool(
        all(
            payload["splits"][split][kind]["fraction_exact"] >= 1.0
            for split in ("train", "valid")
            for kind in ("root", "atom_mass", "bond_mass")
        )
        and all(
            payload["splits"][split]["size2"]["fraction_bit_identical"] >= 1.0
            for split in ("train", "valid")
        )
    )
    payload["a4_pass"] = bool(
        all(
            payload["splits"][split]["redundancy"]["anchor_injective_from_sem108"]
            for split in ("train", "valid")
        )
    )
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# audit A3 — frozen T1 coarse146 block attribution (state availability)
# ---------------------------------------------------------------------------


def audit_t1_block_ablation() -> dict[str, Any]:
    """Audit A3: frozen T1 block ablation if the exact soup state exists.

    The durable 2026-09-24 T1 run persisted its epoch states only under the
    git-ignored ``results/e2e_dictenv_t1/states/`` directory.  When those
    tensors are absent the audit records ``T1_BLOCK_AUDIT_UNAVAILABLE`` and
    never retrains T1 (the round is zero-training before the formal candidate).
    """
    from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

    track_root = REPO_ROOT / "tracks/ksvd"
    state_dir = track_root / "results/e2e_dictenv_t1/states"
    candidates = (
        "stage_b_lambda025_soup_state.pt",
        "stage_b_lambda025_selection_state.pt",
        "final_sparse_seed1_soup_state.pt",
        "final_sparse_seed1_selection_state.pt",
    )
    found = {name: (state_dir / name).exists() for name in candidates}
    decision_path = track_root / "results/e2e_dictenv_t1/tuning_decision.json"
    import json as _json

    decision = _json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "state_dir": str(state_dir.relative_to(REPO_ROOT)),
        "state_files_checked": found,
        "durable_facts": {
            "winner_candidate_id": decision.get("candidate_id", "A2_COARSE146_SLOT48"),
            "winner_soup_valid_mae": decision.get("best_tuned_soup_valid_mae", 0.12576500436564675),
            "interface": "coarse146 + per-shell dictionary slot code (48 x 3), decoder 290 -> 135 -> 48",
            "note": (
                "T1's gain was obtained with the full coarse146 descriptor present; its exact"
                " checkpoint tensors are not durable, so a per-block attribution of that gain"
                " (atom84 / bond24 / Sem108 / topology6) cannot be measured without retraining"
                " T1, which this round forbids."
            ),
        },
    }
    if any(found.values()):
        raise RuntimeError(
            "T1 state tensors unexpectedly present; the frozen block ablation was"
            " pre-registered only for the unavailable-state case and must be"
            " implemented under a new pre-registration"
        )
    payload["status"] = "T1_BLOCK_AUDIT_UNAVAILABLE"
    payload["reason"] = "exact T1 selected/soup state tensors are not durable (git-ignored and deleted)"
    payload["retrained_t1"] = False
    payload["limits"] = [
        "no Delta_sem108 / Delta_atom84 / Delta_bond24 / Delta_topology6 block attribution",
        "the round proceeds on interface history + non-redundant shell-semantic representation",
    ]
    official_test_blocker(payload)
    return payload


def audit_decision(
    identity: Mapping[str, Any],
    relation: Mapping[str, Any],
    t1: Mapping[str, Any],
) -> dict[str, Any]:
    """Phase-A stop rule (pre-registration section 5)."""
    checks = {
        "sem108_identity_established": bool(identity.get("identity_established")),
        "feature_scaler_provenance_valid": bool(
            identity["standardization"]["fit_split"] == "official train"
            and not identity["standardization"]["valid_in_fit"]
        ),
        "official_valid_not_in_scaler_fit": True,
        "anchor_relation_deterministic": bool(relation.get("a2_pass")),
        "redundancy_audit_pass": bool(relation.get("a4_pass")),
        "t1_block_audit": str(t1.get("status")),
        "t1_strongly_rejects_shell_semantics": False,
    }
    proceed = bool(
        checks["sem108_identity_established"]
        and checks["feature_scaler_provenance_valid"]
        and checks["anchor_relation_deterministic"]
        and not checks["t1_strongly_rejects_shell_semantics"]
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "checks": checks,
        "t1_limitation": str(t1.get("reason", "")),
        "decision": "PROCEED" if proceed else "STOP",
        "rationale": (
            "Sem108 columns are established from the current implementation and the cached"
            " descriptor is a bit-identical train-fit standardized image of the raw blocks;"
            " the parent anchor's root/atom/bond marginals and size are deterministic"
            " functions of Sem108 (+size2); the T1 block audit is unavailable because the"
            " exact T1 state tensors were not durable, which is recorded as a limitation"
            " rather than a rejection (no T1 retraining this round)."
            if proceed
            else "a Phase-A stop rule fired"
        ),
    }
    official_test_blocker(payload)
    return payload


__all__ = [
    "PROTOCOL_VERSION",
    "PATCH_CONT_DIM",
    "SEM_ATOM_BLOCK",
    "SEM_BOND_BLOCK",
    "SEM_DIM",
    "SIZE2_BLOCK",
    "SEM_INTERFACE_DIM",
    "FORBIDDEN_T1_BLOCKS",
    "PARAM_RATIO_BOUND",
    "PARAM_RATIO_PREFERRED",
    "RELATION_TOL",
    "ROUND_TRIP_TOL",
    "SEMMask",
    "SEM108Model",
    "SEM_FUSION_IN",
    "SEM_FUSION_HIDDEN",
    "SEM_FUSION_OUT",
    "PARENT_FUSION_HIDDEN",
    "CLOSED_FORM",
    "official_test_blocker",
    "cpu_only_guard",
    "resolve_sem108_geometry",
    "sem108_block",
    "sem110_interface",
    "forbidden_block_columns",
    "merge_sem_mask",
    "parent_local_interface_parameters",
    "candidate_fusion_input_dim",
    "closed_form_fusion_hidden",
    "build_sem108_model",
    "build_parent_equivalence_model",
    "parameter_audit",
    "audit_sem108_identity",
    "audit_anchor_relation_and_redundancy",
    "audit_t1_block_ablation",
    "audit_decision",
]
