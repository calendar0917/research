"""E2E-DictEnv-Hier-Relation-v1: one model, one relation dictionary, no write-back.

Round ``zinc-e2e-dictenv-hier-relation-v1`` (see
``tracks/ksvd/notes/zinc_e2e_dictenv_hier_relation_v1_preregistration.md``).

This module owns only the frozen mathematics and the model:

* the node input layout ``x_i = [Joint709 ; common1 ; size2]`` (``d_x = 712``);
* the node dictionary ``D_N in R^{712 x 128}`` (all 128 coordinates kept, no
  top-k, no message passing, no node write-back);
* the within-molecule same-atom-type shared environment ``mu_i`` and the
  individual deviation ``delta_i``;
* the fixed ``+-1/sqrt(32)`` projection ``P`` (seed ``20261001``, buffer);
* the symmetric relation-object algebra ``r_uv`` (228 dims) and its frozen
  train-only four-block scaler;
* the relation dictionary ``D_R in R^{228 x 64}`` with tied-IHT ``s=8/steps=10``
  (the repository-validated :func:`e2e_dictenv_v0.tied_iht_codes`);
* the graph readout (node pooling 384 + relation pooling 192 + counts 2 +
  topology 8 = 586) and the single prediction head.

Contract enforced structurally: the relation object may **read** one
atom-type-grouped within-graph aggregate of the node coordinates, but node
coordinates are never written back, no relation tensor is stacked into a second
relation layer, and the pooled node / relation coordinates go to one shared
head (never two independently trained models).

Batching is explicit (global index arrays, no PyG ``Batch``), so a new field can
never be silently double-offset.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

# ---------------------------------------------------------------------------
# frozen layout constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "e2e_dictenv_hier_relation_v1"
ROUND = "zinc-e2e-dictenv-hier-relation-v1"

JOINT_DIM = 709
COMMON_DIM = 1
SIZE_DIM = 2
EXTRA_SIZE_DIM = COMMON_DIM + SIZE_DIM  # 3
D_X = JOINT_DIM + EXTRA_SIZE_DIM  # 712
MAX_INPUT_DIM = 712

JOINT_SLICES: dict[str, tuple[int, int]] = {
    "struct": (0, 65),
    "sem": (65, 173),
    "corr": (173, 709),
}
EXTRA_SLICES: dict[str, tuple[int, int]] = {
    "common": (0, 1),
    "size": (1, 3),
}

NODE_ATOMS = 128
ATOM_CATEGORIES = int(v0.ATOM_CATEGORIES)  # 28
BOND_CATEGORIES = int(v0.BOND_CATEGORIES)  # 4

PROJ_DIM = 32
PROJ_SEED = 20261001

ATTACH_DIM = 3 * PROJ_DIM  # 96
REL_INPUT_DIM = ATTACH_DIM * 2 + PROJ_DIM + BOND_CATEGORIES  # 228
REL_BLOCKS: dict[str, tuple[int, int]] = {
    "mu": (0, 96),
    "delta": (96, 192),
    "mu_delta": (192, 224),
    "bond": (224, 228),
}
REL_BLOCK_ORDER: tuple[str, ...] = ("mu", "delta", "mu_delta", "bond")

REL_ATOMS = 64
REL_SPARSITY = 8
REL_IHT_STEPS = int(v0.IHT_STEPS)  # 10

NODE_POOL_DIM = 3 * NODE_ATOMS  # 384
REL_POOL_DIM = 3 * REL_ATOMS  # 192
COUNT_DIM = 2
TOPO_IN = 25
TOPO_HIDDEN = 16
TOPO_OUT = 8
READOUT_DIM = NODE_POOL_DIM + REL_POOL_DIM + COUNT_DIM + TOPO_OUT  # 586
HEAD_HIDDEN = (64, 32)

DEPTH = 1  # one relation layer only: no stacking, no refresh
MESSAGE_PASSING = False
NODE_WRITE_BACK = False

LAMBDA_NODE = 0.05
LAMBDA_REL = 0.02
STD_EPS = 1e-8
REL_RECON_EPS = 1e-12

#: fixed protocol knobs (mirrored into the control-plane config)
SEED = 0
TRAIN_EPOCHS = 320
BATCH_SIZE = 128
SMOKE_EPOCHS = 3
SMOKE_TRAIN_SUBSET = 2048
SMOKE_VALID_SUBSET = 512
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
SOUP_K = 5
DROPOUT = 0.05

#: deterministic spectral-initialisation sampling (recorded, never re-drawn)
INIT_SAMPLE_MAX = 32768
NODE_INIT_SAMPLE_SEED = 20261002
EDGE_INIT_SAMPLE_SEED = 20261003
SUPPLEMENT_SEED = 0

#: intervention permutation seeds (pre-registered)
REL_PERM_SEEDS = (11, 22)

#: activity-probe epochs (pre-registered)
PROBE_EPOCHS = (0, 5, 20, 80, 160, 240, 320)
PROBE_MOLECULES = 256

SCALER_FLOOR = float(a1.SCALER_FLOOR)
SCALER_EPS = float(a1.SCALER_EPS)


class ContractViolation(RuntimeError):
    """Raised when a structural contract of this round would be broken."""


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    return sha256_array(tensor.detach().cpu().contiguous().numpy())


# ---------------------------------------------------------------------------
# fixed projection
# ---------------------------------------------------------------------------


def build_projection(
    dim_in: int = NODE_ATOMS, dim_out: int = PROJ_DIM, seed: int = PROJ_SEED
) -> torch.Tensor:
    """Fixed elementwise ``+-1/sqrt(dim_out)`` projection (own CPU generator)."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(0, 2, (int(dim_in), int(dim_out)), generator=generator)
    return ((2.0 * signs - 1.0) / math.sqrt(float(dim_out))).to(torch.float32)


def projection_sha256(proj: torch.Tensor) -> str:
    return sha256_tensor(proj)


# ---------------------------------------------------------------------------
# scalers (train-only, frozen once fitted)
# ---------------------------------------------------------------------------


@dataclass
class FrozenScaler:
    """Per-column RMS + zero-RMS mask + block-energy weight (train only).

    ``gain`` is the pre-multiplied float32 per-column gain used at forward time;
    ``scale`` / ``mask`` / ``weight`` are kept verbatim for the provenance
    record and for the hand-recomputation tests.
    """

    name: str
    dim: int
    scale: np.ndarray
    mask: np.ndarray
    weight: np.ndarray
    block_order: tuple[str, ...]
    block_slices: Mapping[str, tuple[int, int]]
    rms_raw: np.ndarray
    block_energy: Mapping[str, float]
    fit_rows: int

    @property
    def gain(self) -> np.ndarray:
        per_col = self.mask / self.scale
        out = np.ones(int(self.dim), dtype=np.float64)
        for index, block in enumerate(self.block_order):
            lo, hi = self.block_slices[block]
            out[lo:hi] = per_col[lo:hi] * float(self.weight[index])
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "dim": int(self.dim),
            "fit_rows": int(self.fit_rows),
            "blocks": {
                block: {
                    "slice": [int(self.block_slices[block][0]), int(self.block_slices[block][1])],
                    "weight": float(self.weight[index]),
                    "block_energy": float(self.block_energy[block]),
                    "masked_coordinates": int(
                        np.sum(
                            self.mask[
                                self.block_slices[block][0] : self.block_slices[block][1]
                            ]
                            == 0.0
                        )
                    ),
                    "rms_min": float(
                        np.min(
                            self.rms_raw[
                                self.block_slices[block][0] : self.block_slices[block][1]
                            ]
                        )
                    ),
                    "rms_max": float(
                        np.max(
                            self.rms_raw[
                                self.block_slices[block][0] : self.block_slices[block][1]
                            ]
                        )
                    ),
                }
                for index, block in enumerate(self.block_order)
            },
            "block_order": list(self.block_order),
            "scale": [float(x) for x in self.scale.tolist()],
            "mask": [int(x) for x in self.mask.tolist()],
            "gain": [float(x) for x in self.gain.tolist()],
            "fit_split": "official train",
            "mean_subtracted": False,
            "official_test_loaded": False,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "FrozenScaler":
        order = tuple(payload["block_order"])
        return cls(
            name=str(payload["name"]),
            dim=int(payload["dim"]),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            mask=np.asarray(payload["mask"], dtype=np.float64),
            weight=np.asarray([payload["blocks"][b]["weight"] for b in order], dtype=np.float64),
            block_order=order,
            block_slices={
                b: (int(payload["blocks"][b]["slice"][0]), int(payload["blocks"][b]["slice"][1]))
                for b in order
            },
            rms_raw=np.zeros(int(payload["dim"]), dtype=np.float64),
            block_energy={b: float(payload["blocks"][b]["block_energy"]) for b in order},
            fit_rows=int(payload["fit_rows"]),
        )


def fit_frozen_scaler(
    name: str, block_slices: Mapping[str, tuple[int, int]], block: np.ndarray
) -> FrozenScaler:
    """Fit one multi-block frozen scaler with the validated ``a1`` math."""
    block = np.asarray(block, dtype=np.float64)
    dim = int(block.shape[1])
    order = tuple(block_slices)
    if dim != max(hi for _lo, hi in block_slices.values()):
        raise ContractViolation("block slices do not cover the declared width")
    scale = np.ones(dim, dtype=np.float64)
    mask = np.ones(dim, dtype=np.float64)
    weight: list[float] = []
    energy: dict[str, float] = {}
    rms_raw = np.zeros(dim, dtype=np.float64)
    for block_name in order:
        lo, hi = block_slices[block_name]
        entry = a1.fit_block_scaler(block_name, block[:, lo:hi])
        scale[lo:hi] = entry.scale
        mask[lo:hi] = entry.mask
        rms_raw[lo:hi] = entry.rms_raw
        weight.append(float(entry.weight))
        energy[block_name] = float(entry.block_energy)
    return FrozenScaler(
        name=str(name),
        dim=dim,
        scale=scale,
        mask=mask,
        weight=np.asarray(weight, dtype=np.float64),
        block_order=order,
        block_slices={b: tuple(block_slices[b]) for b in order},
        rms_raw=rms_raw,
        block_energy=energy,
        fit_rows=int(block.shape[0]),
    )


def scaled_block_energy(block: np.ndarray, scaler: FrozenScaler) -> dict[str, float]:
    """Post-hoc check that each block's mean squared norm is ~1 on fitted rows."""
    block = np.asarray(block, dtype=np.float64)
    out: dict[str, float] = {}
    for block_name in scaler.block_order:
        lo, hi = scaler.block_slices[block_name]
        scaled = block[:, lo:hi] * scaler.gain[lo:hi][None, :]
        out[block_name] = float(np.mean(np.sum(scaled * scaled, axis=1)))
    return out


# ---------------------------------------------------------------------------
# relation object algebra
# ---------------------------------------------------------------------------


def symmetric_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``S(a,b) = [a+b ; |a-b| ; a*b]`` (endpoint-swap invariant, 3*d wide)."""
    return torch.cat([a + b, (a - b).abs(), a * b], dim=-1)


def relation_objects(
    m_u: torch.Tensor,
    d_u: torch.Tensor,
    m_v: torch.Tensor,
    d_v: torch.Tensor,
    bond_type: torch.Tensor,
) -> torch.Tensor:
    """``r_uv`` (228 dims) built from the per-endpoint projected (m, d)."""
    bond = F.one_hot(bond_type.long(), num_classes=BOND_CATEGORIES).to(m_u.dtype)
    return torch.cat(
        [
            symmetric_pair(m_u, m_v),
            symmetric_pair(d_u, d_v),
            m_u * d_v + m_v * d_u,
            bond,
        ],
        dim=-1,
    )


def relation_block_energy(r: torch.Tensor) -> dict[str, float]:
    """Mean squared norm of each of the four relation blocks (no grad)."""
    with torch.no_grad():
        out: dict[str, float] = {}
        for block in REL_BLOCK_ORDER:
            lo, hi = REL_BLOCKS[block]
            part = r[:, lo:hi]
            out[block] = float(part.pow(2).sum(dim=1).mean()) if part.numel() else 0.0
        return out


# ---------------------------------------------------------------------------
# grouping / pooling helpers
# ---------------------------------------------------------------------------


def group_key(graph_id: torch.Tensor, atom_type: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Within-graph atom-type group key ``g * 28 + t`` (never crosses graphs)."""
    key = graph_id.long() * ATOM_CATEGORIES + atom_type.long()
    if int(key.min()) < 0 or int(key.max()) >= int(n_graphs) * ATOM_CATEGORIES:
        raise ContractViolation("group key outside [0, n_graphs * ATOM_CATEGORIES)")
    return key


def grouped_mean(
    value: torch.Tensor, key: torch.Tensor, n_groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment mean plus the per-row group size (functional ``index_add``)."""
    sums = torch.zeros(int(n_groups), value.shape[1], device=value.device, dtype=value.dtype)
    sums = sums.index_add(0, key, value)
    counts = torch.zeros(int(n_groups), device=value.device, dtype=value.dtype)
    counts = counts.index_add(0, key, torch.ones_like(value[:, 0]))
    return sums / counts.clamp(min=1.0)[:, None], counts


def shared_and_deviation(
    z: torch.Tensor, graph_id: torch.Tensor, atom_type: torch.Tensor, n_graphs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mu_i, delta_i)`` from the within-molecule same-atom-type group."""
    key = group_key(graph_id, atom_type, int(n_graphs))
    means, _counts = grouped_mean(z, key, int(n_graphs) * ATOM_CATEGORIES)
    mu = means.index_select(0, key)
    return mu, z - mu


def in_group_permutation(
    graph_id: torch.Tensor, atom_type: torch.Tensor, n_graphs: int, seed: int
) -> torch.Tensor:
    """Deterministic permutation that only moves nodes inside their own group.

    Within every ``(graph_id, atom_type)`` group the members are rolled by a
    seeded offset in ``[0, size-1]``; singleton groups are the identity.  The
    group *mean* is therefore invariant (the multiset is unchanged) while the
    individual deviations are permuted.
    """
    key = group_key(graph_id, atom_type, int(n_graphs))
    order = torch.argsort(key.to(torch.float64), stable=True)
    sorted_key = key.index_select(0, order)
    change = torch.ones_like(sorted_key, dtype=torch.bool)
    change[1:] = sorted_key[1:] != sorted_key[:-1]
    starts = torch.nonzero(change, as_tuple=False).flatten()
    sizes = torch.diff(
        torch.cat([starts, torch.tensor([sorted_key.numel()], dtype=starts.dtype)])
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    draws = torch.rand(int(starts.numel()), generator=generator)
    offsets = torch.floor(draws * sizes.to(torch.float64)).to(torch.long).clamp(min=0)
    group_starts = starts.repeat_interleave(sizes)
    group_sizes = sizes.repeat_interleave(sizes)
    position = torch.arange(sorted_key.numel(), dtype=torch.long) - group_starts
    source = group_starts + (position + offsets.repeat_interleave(sizes)) % group_sizes
    perm = torch.empty_like(order)
    perm[order] = order.index_select(0, source)
    return perm


def pool_sum_mean_std(value: torch.Tensor, index: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """``[sum ; mean ; population std]`` per graph (empty rows stay exactly 0)."""
    dim = int(value.shape[1])
    counts = torch.zeros(int(n_graphs), device=value.device, dtype=value.dtype)
    counts = counts.index_add(0, index, torch.ones_like(value[:, 0]))
    total = torch.zeros(int(n_graphs), dim, device=value.device, dtype=value.dtype)
    total = total.index_add(0, index, value)
    empty = counts <= 0
    safe = counts.clamp(min=1.0)
    mean = total / safe[:, None]
    sq = torch.zeros(int(n_graphs), dim, device=value.device, dtype=value.dtype)
    sq = sq.index_add(0, index, value * value)
    var = (sq / safe[:, None] - mean * mean).clamp(min=0.0)
    std = torch.sqrt(var + STD_EPS)
    total = torch.where(empty[:, None], torch.zeros_like(total), total)
    mean = torch.where(empty[:, None], torch.zeros_like(mean), mean)
    std = torch.where(empty[:, None], torch.zeros_like(std), std)
    return torch.cat([total, mean, std], dim=1)


# ---------------------------------------------------------------------------
# spectral initialisation
# ---------------------------------------------------------------------------


def second_moment_basis(
    matrix: np.ndarray, atoms: int, *, seed: int, log: Any = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Top-``atoms`` eigenvectors of the uncentred train second moment.

    ``matrix`` is ``[n, d]`` float64 (train-only, deterministically sampled).
    Signs are fixed by the largest-|component| convention.  If the effective
    rank is below ``atoms`` the remaining columns are deterministic normalised
    supplements (``seed``), orthonormalised against the fitted eigenvectors;
    no run is ever padded by a silent zero column.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n, d = matrix.shape
    if int(atoms) > d:
        raise ContractViolation(f"cannot take {atoms} eigenvectors of a {d}-dim second moment")
    moment = (matrix.T @ matrix) / float(max(n, 1))
    values, vectors = np.linalg.eigh(moment)
    order = np.argsort(values)[::-1][: int(atoms)]
    basis = vectors[:, order].copy()
    scale_ref = float(values[order[0]]) if float(values[order[0]]) > 0.0 else 1.0
    effective = int(np.sum(values[order] > scale_ref * 1e-12))
    supplemented = 0
    if effective < int(atoms):
        generator = np.random.default_rng(int(seed))
        for index in range(effective, int(atoms)):
            candidate = generator.normal(size=d)
            for previous in range(index):
                candidate -= float(candidate @ basis[:, previous]) * basis[:, previous]
            norm = float(np.linalg.norm(candidate))
            if norm < 1e-9:
                candidate = np.zeros(d, dtype=np.float64)
                candidate[index % d] = 1.0
                for previous in range(index):
                    candidate -= float(candidate @ basis[:, previous]) * basis[:, previous]
                norm = float(np.linalg.norm(candidate))
            basis[:, index] = candidate / max(norm, 1e-12)
            supplemented += 1
    for index in range(basis.shape[1]):
        column = basis[:, index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            basis[:, index] = -column
    info = {
        "rows": int(n),
        "dim": int(d),
        "atoms": int(atoms),
        "effective_rank": int(effective),
        "supplemented_columns": int(supplemented),
        "top_eigenvalue": float(values[order[0]]),
        "atoms_eigenvalue_min": float(values[order[int(atoms) - 1]]),
        "orthonormality_max_error": float(np.abs(basis.T @ basis - np.eye(int(atoms))).max()),
        "sign_convention": "largest_abs_component_positive",
        "sample_seed": int(seed),
        "basis_sha256_f32": sha256_array(basis.astype(np.float32)),
    }
    if log is not None:
        log(
            f"[spectral-init] rows={n} dim={d} atoms={atoms} "
            f"effective_rank={effective} supplemented={supplemented} "
            f"orth_err={info['orthonormality_max_error']:.3e}"
        )
    return basis.astype(np.float32), info


def sample_indices(total: int, limit: int, seed: int) -> np.ndarray:
    """Deterministic without-replacement train sample (capped at ``limit``)."""
    total = int(total)
    if total <= int(limit):
        return np.arange(total, dtype=np.int64)
    return np.sort(
        np.random.default_rng(int(seed)).choice(total, size=int(limit), replace=False)
    ).astype(np.int64)


# ---------------------------------------------------------------------------
# prepared split tensors (explicit global indices; no PyG batching)
# ---------------------------------------------------------------------------


@dataclass
class SplitTensors:
    """Prepared, frozen per-split tensors (CPU) shared by every batch."""

    x: torch.Tensor  # [N, 712] frozen scaled node input
    node_ptr: torch.Tensor  # [G+1]
    atom_type: torch.Tensor  # [N]
    edge_index: torch.Tensor  # [2, M] global node ids (u < v)
    edge_type: torch.Tensor  # [M]
    edge_ptr: torch.Tensor  # [G+1]
    topology: torch.Tensor  # [G, 25]
    y: torch.Tensor  # [G]
    mol_id: torch.Tensor  # [G]
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_type.numel())

    @property
    def n_graphs(self) -> int:
        return int(self.y.numel())

    def node_count(self, slots: torch.Tensor) -> torch.Tensor:
        return self.node_ptr.index_select(0, slots + 1) - self.node_ptr.index_select(0, slots)

    def edge_count(self, slots: torch.Tensor) -> torch.Tensor:
        return self.edge_ptr.index_select(0, slots + 1) - self.edge_ptr.index_select(0, slots)

    def batch(self, slots: Sequence[int] | torch.Tensor) -> "HierBatch":
        """Materialise one batch from explicit molecule slots.

        Node and edge ids are **global**; the local remap is built per batch, so
        no offset can be applied twice.
        """
        slots = torch.as_tensor(list(slots), dtype=torch.long)
        graphs = int(slots.numel())
        node_counts = self.node_count(slots)
        edge_counts = self.edge_count(slots)
        contiguous = bool(
            torch.equal(
                slots, torch.arange(int(slots[0]), int(slots[0]) + graphs, dtype=torch.long)
            )
        )
        if contiguous:
            node_ids = torch.arange(
                int(self.node_ptr[slots[0]]),
                int(self.node_ptr[slots[-1] + 1]),
                dtype=torch.long,
            )
            edge_ids = torch.arange(
                int(self.edge_ptr[slots[0]]),
                int(self.edge_ptr[slots[-1] + 1]),
                dtype=torch.long,
            )
        else:
            node_ids = torch.cat(
                [
                    torch.arange(int(self.node_ptr[s]), int(self.node_ptr[s + 1]))
                    for s in slots.tolist()
                ]
            )
            edge_ids = torch.cat(
                [
                    torch.arange(int(self.edge_ptr[s]), int(self.edge_ptr[s + 1]))
                    for s in slots.tolist()
                ]
            )
        local_graph = torch.repeat_interleave(
            torch.arange(graphs, dtype=torch.long), node_counts
        )
        edge_local = torch.repeat_interleave(torch.arange(graphs, dtype=torch.long), edge_counts)
        local_of_global = torch.full(
            (int(self.x.shape[0]),), -1, dtype=torch.long
        )
        local_of_global[node_ids] = torch.arange(int(node_ids.numel()), dtype=torch.long)
        edges = self.edge_index.index_select(1, edge_ids)
        local_edges = local_of_global.index_select(0, edges.reshape(-1)).reshape(2, -1)
        if bool((local_edges < 0).any()):
            raise ContractViolation("batch edge references a node outside the batch")
        return HierBatch(
            x=self.x.index_select(0, node_ids),
            graph_id=local_graph,
            atom_type=self.atom_type.index_select(0, node_ids),
            edge_index=local_edges,
            edge_type=self.edge_type.index_select(0, edge_ids),
            edge_graph=edge_local,
            topology=self.topology.index_select(0, slots),
            y=self.y.index_select(0, slots),
            mol_index=self.mol_id.index_select(0, slots),
            n_graphs=graphs,
        )


@dataclass
class HierBatch:
    """One explicit batch; every index is local to the batch."""

    x: torch.Tensor  # [N, 712] scaled node input
    graph_id: torch.Tensor  # [N] local graph index 0..G-1
    atom_type: torch.Tensor  # [N] atom category
    edge_index: torch.Tensor  # [2, M] local node ids (u < v)
    edge_type: torch.Tensor  # [M] bond category
    edge_graph: torch.Tensor  # [M] local graph index
    topology: torch.Tensor  # [G, 25]
    y: torch.Tensor  # [G]
    mol_index: torch.Tensor  # [G] original molecule ids
    n_graphs: int = 0

    def to(self, device: Any) -> "HierBatch":
        return HierBatch(
            **{
                name: (value.to(device) if torch.is_tensor(value) else value)
                for name, value in self.__dict__.items()
            }
        )


def make_batches(
    n_graphs: int, batch_size: int, *, shuffle: bool, seed: int
) -> list[torch.Tensor]:
    """Deterministic contiguous molecule slots (optionally shuffled once)."""
    slots = np.arange(int(n_graphs), dtype=np.int64)
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(slots)
    size = max(1, int(batch_size))
    return [
        torch.as_tensor(part, dtype=torch.long)
        for part in np.array_split(slots, max(1, int(np.ceil(len(slots) / size))))
        if len(part)
    ]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class HierRelationModel(nn.Module):
    """One model: node dictionary -> shared/deviation relation object -> one head."""

    def __init__(
        self,
        *,
        d_x: int = D_X,
        node_atoms: int = NODE_ATOMS,
        rel_atoms: int = REL_ATOMS,
        proj: torch.Tensor | None = None,
        node_input_mask: torch.Tensor | None = None,
        extra_gain: torch.Tensor | None = None,
        rel_gain: torch.Tensor | None = None,
        dropout: float = DROPOUT,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if int(d_x) > MAX_INPUT_DIM:
            raise ContractViolation(f"d_x={d_x} exceeds the frozen maximum {MAX_INPUT_DIM}")
        self.d_x = int(d_x)
        self.node_atoms = int(node_atoms)
        self.rel_atoms = int(rel_atoms)
        self.D_node = nn.Parameter(torch.zeros(self.d_x, self.node_atoms))
        self.D_rel = nn.Parameter(torch.zeros(REL_INPUT_DIM, self.rel_atoms))
        nn.init.normal_(self.D_node, std=float(init_scale))
        nn.init.normal_(self.D_rel, std=float(init_scale))
        projection = (
            build_projection(self.node_atoms, PROJ_DIM, PROJ_SEED) if proj is None else proj
        )
        if tuple(projection.shape) != (self.node_atoms, PROJ_DIM):
            raise ContractViolation("projection shape mismatch")
        self.register_buffer("projection", projection.clone().to(torch.float32))
        mask = (
            torch.ones(self.d_x, dtype=torch.float32)
            if node_input_mask is None
            else torch.as_tensor(node_input_mask, dtype=torch.float32).flatten()
        )
        if int(mask.numel()) != self.d_x:
            raise ContractViolation("node input mask width mismatch")
        self.register_buffer("node_input_mask", mask)
        gain = (
            torch.ones(EXTRA_SIZE_DIM, dtype=torch.float32)
            if extra_gain is None
            else torch.as_tensor(extra_gain, dtype=torch.float32).flatten()
        )
        if int(gain.numel()) != EXTRA_SIZE_DIM:
            raise ContractViolation("extra gain width mismatch")
        self.register_buffer("extra_gain", gain)
        rel = (
            torch.ones(REL_INPUT_DIM, dtype=torch.float32)
            if rel_gain is None
            else torch.as_tensor(rel_gain, dtype=torch.float32).flatten()
        )
        if int(rel.numel()) != REL_INPUT_DIM:
            raise ContractViolation("relation gain width mismatch")
        self.register_buffer("rel_gain", rel)
        self.topology_encoder = nn.Sequential(
            nn.Linear(TOPO_IN, TOPO_HIDDEN),
            nn.ReLU(),
            nn.Linear(TOPO_HIDDEN, TOPO_OUT),
        )
        self.head = nn.Sequential(
            nn.Linear(READOUT_DIM, HEAD_HIDDEN[0]),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(HEAD_HIDDEN[0], HEAD_HIDDEN[1]),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(HEAD_HIDDEN[1], 1),
        )

    # -- frozen structure -------------------------------------------------
    @property
    def node_dictionary(self) -> torch.Tensor:
        return v0.normalized_dictionary(self.D_node)

    @property
    def relation_dictionary(self) -> torch.Tensor:
        return v0.normalized_dictionary(self.D_rel)

    def node_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen input mask (only fixed zero-RMS columns are dropped).

        The 709-D joint block is already scaled by the frozen train-only joint
        scaler; the appended common1 / size2 block is scaled here with the frozen
        train-only per-column gains.
        """
        masked = x * self.node_input_mask[None, :]
        extra = masked[:, JOINT_DIM:] * self.extra_gain[None, :]
        return torch.cat([masked[:, :JOINT_DIM], extra], dim=1)

    def relation_codes(
        self, r_scaled: torch.Tensor, dbar_rel: torch.Tensor | None = None
    ) -> torch.Tensor:
        dictionary = self.relation_dictionary if dbar_rel is None else dbar_rel
        return v0.tied_iht_codes(
            dictionary,
            r_scaled,
            s=REL_SPARSITY,
            steps=REL_IHT_STEPS,
        )

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        batch: HierBatch,
        *,
        mode: str = "none",
        perm_seed: int | None = None,
    ) -> dict[str, Any]:
        """Run with the model's own column-normalised dictionaries."""
        return self.forward_with_dictionaries(
            batch, self.node_dictionary, self.relation_dictionary, mode=mode, perm_seed=perm_seed
        )

    def forward_with_dictionaries(
        self,
        batch: HierBatch,
        dbar_node: torch.Tensor,
        dbar_rel: torch.Tensor,
        *,
        mode: str = "none",
        perm_seed: int | None = None,
    ) -> dict[str, Any]:
        """Explicit-dictionary forward (used by the MAE-only direction-gradient probe)."""
        if mode not in (
            "none",
            "zero_beta",
            "zero_node_pool",
            "zero_delta_rel",
            "permute_endpoints",
        ):
            raise ContractViolation(f"unknown intervention mode {mode!r}")
        if mode == "permute_endpoints" and perm_seed is None:
            raise ContractViolation("permute_endpoints requires an explicit seed")
        n_graphs = int(batch.n_graphs)
        x = self.node_input(batch.x)
        z = x @ dbar_node  # [N, 128] all coordinates, no top-k
        x_hat = z @ dbar_node.t()

        mu, delta = shared_and_deviation(z, batch.graph_id, batch.atom_type, n_graphs)
        if mode == "permute_endpoints":
            perm = in_group_permutation(
                batch.graph_id, batch.atom_type, n_graphs, int(perm_seed)
            )
            z_rel = z.index_select(0, perm)
            mu_rel, delta_rel = shared_and_deviation(
                z_rel, batch.graph_id, batch.atom_type, n_graphs
            )
        else:
            mu_rel, delta_rel = mu, delta
        if mode == "zero_delta_rel":
            delta_rel = torch.zeros_like(delta_rel)

        m = mu_rel @ self.projection
        d = delta_rel @ self.projection
        u, v = batch.edge_index[0], batch.edge_index[1]
        r_raw = relation_objects(
            m.index_select(0, u),
            d.index_select(0, u),
            m.index_select(0, v),
            d.index_select(0, v),
            batch.edge_type,
        )
        r_scaled = r_raw * self.rel_gain[None, :]

        edges = int(r_scaled.shape[0])
        if edges > 0:
            beta = self.relation_codes(r_scaled, dbar_rel)
        else:
            beta = r_scaled.new_zeros((0, self.rel_atoms))
        if mode == "zero_beta":
            # switch the relation-code channel off for both the readout and the
            # reconstruction term; the node branch is untouched
            beta = torch.zeros_like(beta)
        if edges > 0:
            r_hat = beta @ dbar_rel.t()
            mean_recon = (r_scaled - r_hat).pow(2).sum() / edges
            mean_ref = r_scaled.pow(2).sum() / edges
            r_rel = mean_recon / (mean_ref + REL_RECON_EPS)
        else:
            r_hat = r_scaled.new_zeros(r_scaled.shape)
            r_rel = r_scaled.new_zeros(())

        node_pool = pool_sum_mean_std(z, batch.graph_id, n_graphs)
        if mode == "zero_node_pool":
            node_pool = torch.zeros_like(node_pool)
        rel_pool = pool_sum_mean_std(beta, batch.edge_graph, n_graphs)

        node_counts = torch.zeros(n_graphs, device=z.device, dtype=z.dtype)
        node_counts = node_counts.index_add(0, batch.graph_id, torch.ones_like(z[:, 0]))
        edge_counts = torch.zeros(n_graphs, device=z.device, dtype=z.dtype)
        if beta.shape[0] > 0:
            edge_counts = edge_counts.index_add(
                0, batch.edge_graph, torch.ones_like(beta[:, 0])
            )
        counts = torch.stack([torch.log1p(node_counts), torch.log1p(edge_counts)], dim=1)
        topo = self.topology_encoder(batch.topology)
        features = torch.cat([node_pool, rel_pool, counts, topo], dim=1)
        if int(features.shape[1]) != READOUT_DIM:
            raise ContractViolation(f"readout width {int(features.shape[1])} != {READOUT_DIM}")
        prediction = self.head(features).reshape(-1)

        mean_node_recon = (x - x_hat).pow(2).sum() / x.shape[0]
        mean_node_ref = x.pow(2).sum() / x.shape[0]
        r_node = mean_node_recon / (mean_node_ref + REL_RECON_EPS)

        return {
            "pred": prediction,
            "x": x,
            "z": z,
            "x_hat": x_hat,
            "mu": mu,
            "delta": delta,
            "r_scaled": r_scaled,
            "r_hat": r_hat,
            "beta": beta,
            "node_pool": node_pool,
            "rel_pool": rel_pool,
            "counts": counts,
            "features": features,
            "r_node": r_node,
            "r_rel": r_rel,
            "edge_counts": edge_counts,
            "node_dictionary": dbar_node,
            "relation_dictionary": dbar_rel,
        }

    def loss(
        self,
        batch: HierBatch,
        output: Mapping[str, Any] | None = None,
        *,
        mode: str = "none",
        perm_seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``MAE + 0.05 * R_N + 0.02 * R_R`` on the real inputs / reconstructions."""
        out = self(batch, mode=mode, perm_seed=perm_seed) if output is None else output
        task = F.l1_loss(out["pred"].view(-1), batch.y.view(-1))
        penalty = out["r_node"] * LAMBDA_NODE + out["r_rel"] * LAMBDA_REL
        return task + penalty, {
            "task": task.detach(),
            "r_node": out["r_node"].detach(),
            "r_rel": out["r_rel"].detach(),
            "loss": (task + penalty).detach(),
        }


def parameter_report(model: HierRelationModel) -> dict[str, Any]:
    """Exact trainable/buffer accounting (P and the scaler gains are buffers)."""
    named = {name: int(p.numel()) for name, p in model.named_parameters()}
    buffers = {name: int(b.numel()) for name, b in model.named_buffers()}
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total = int(sum(p.numel() for p in model.parameters()))
    expected = (
        model.d_x * NODE_ATOMS
        + REL_INPUT_DIM * REL_ATOMS
        + (TOPO_IN * TOPO_HIDDEN + TOPO_HIDDEN + TOPO_HIDDEN * TOPO_OUT + TOPO_OUT)
        + (
            READOUT_DIM * HEAD_HIDDEN[0]
            + HEAD_HIDDEN[0]
            + HEAD_HIDDEN[0] * HEAD_HIDDEN[1]
            + HEAD_HIDDEN[1]
            + HEAD_HIDDEN[1]
            + 1
        )
    )
    return {
        "trainable": trainable,
        "total_parameters": total,
        "parameters": named,
        "buffers": buffers,
        "buffer_elements": int(sum(buffers.values())),
        "expected_trainable": int(expected),
        "matches_expected": bool(trainable == int(expected)),
        "readout_dim": int(READOUT_DIM),
        "node_pool_dim": int(NODE_POOL_DIM),
        "relation_pool_dim": int(REL_POOL_DIM),
        "d_x": int(model.d_x),
        "node_atoms": int(model.node_atoms),
        "relation_input_dim": int(REL_INPUT_DIM),
        "relation_atoms": int(model.rel_atoms),
        "lambda_node": float(LAMBDA_NODE),
        "lambda_rel": float(LAMBDA_REL),
    }


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to(torch.float32).cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "ROUND",
    "PROTOCOL_VERSION",
    "D_X",
    "JOINT_DIM",
    "EXTRA_SIZE_DIM",
    "NODE_ATOMS",
    "REL_INPUT_DIM",
    "REL_ATOMS",
    "REL_SPARSITY",
    "REL_IHT_STEPS",
    "READOUT_DIM",
    "REL_BLOCKS",
    "REL_BLOCK_ORDER",
    "PROBE_EPOCHS",
    "ContractViolation",
    "FrozenScaler",
    "HierBatch",
    "HierRelationModel",
    "SplitTensors",
    "build_projection",
    "fit_frozen_scaler",
    "group_key",
    "grouped_mean",
    "in_group_permutation",
    "make_batches",
    "parameter_report",
    "pool_sum_mean_std",
    "projection_sha256",
    "relation_block_energy",
    "relation_objects",
    "sample_indices",
    "scaled_block_energy",
    "second_moment_basis",
    "sha256_array",
    "sha256_tensor",
    "shared_and_deviation",
    "state_sha256",
    "symmetric_pair",
]
