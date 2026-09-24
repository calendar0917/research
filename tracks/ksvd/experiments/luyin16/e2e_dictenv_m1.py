"""E2E-DictEnv-M1 — primitive-only dictionary-core architecture transfer to OGBG-MolHIV.

Round: ``e2e_dictenv_m1``.  Study ``zinc-context-gap`` (Workstream M).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_m1_preregistration.md``.

This is an **architecture** transfer of the P1 clean dictionary core, not a
vocabulary transfer.  The dictionary is re-fit label-free on the MolHIV
official-train pure-topology ``phi_v^65``; the ZINC ``D`` is not used as the M1
initialization.

Local environment (chemistry enters only as zeroth-order primitives + OGB
categorical embeddings, and local structural organisation passes through the
dictionary code)::

    phi_v^65 --IHT_{s=8}(Dbar)--> alpha_v^32
    q_v^chem = sum_field Embedding(field)(chem)          in R^24
    b_e^chem = sum_field Embedding(field)(chem)          in R^12
    anchor  p_i = [q_i ; sum_P q_v ; sum_E b_e ; log sizes]        in R^62
    node    (alpha_v W_A^S) o (q_v W_A^C) / sqrt(96)   -> per-shell SUM -> A_i (288)
    edge    g_uv = [a_u+a_v; |a_u-a_v|; a_u o a_v]     -> per-shellpair SUM -> E_i (288)
    z_i = [p_i ; A_i ; E_i] (638) --SiLU MLP 638->102->48--> E_i
    global  = [topology^30 ; mean atom chem^24 ; mean bond chem^12] (66 -> 32 -> 32)
    topology= 25 -> 16 -> 8
    reader  = GenericReader(302, (13, 13)) -> one HIV logit

Hard purity: no ``patch_cont`` / ``atom_shell`` / ``bond_shell`` or any
handcrafted shell x chemistry histogram; no pair bond-chemistry relation; no
message passing; no recurrence; no pair->centre; no attention; no context
writeback.

``official_test_loaded = false`` always in this module.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen configuration (pre-registered; no sweep)
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "e2e_dictenv_m1"

PHI_DIM = int(r2.PHI_DIM)  # 65
K_ATOMS = 32
SPARSITY = 8
IHT_STEPS = 10
N_SHELLS = 3
PATCH_RADIUS = int(r2.PATCH_RADIUS)  # 2
SHELLPAIR_CLASSES = 6
ENV_DIM = 48

# OGB chemistry widths (sizes read at runtime from ogb.utils.features)
ATOM_CHEM_DIM = 24
BOND_CHEM_DIM = 12

#: zeroth-order primitive anchor layout:
#:   root chem(24) | patch atom mass(24) | patch bond mass(12) | size(2)
ANCHOR_DIM = ATOM_CHEM_DIM + ATOM_CHEM_DIM + BOND_CHEM_DIM + 2  # 62
ANCHOR_ROOT = slice(0, ATOM_CHEM_DIM)
ANCHOR_ATOM_MASS = slice(ATOM_CHEM_DIM, 2 * ATOM_CHEM_DIM)
ANCHOR_BOND_MASS = slice(2 * ATOM_CHEM_DIM, 2 * ATOM_CHEM_DIM + BOND_CHEM_DIM)
ANCHOR_SIZE = slice(2 * ATOM_CHEM_DIM + BOND_CHEM_DIM, ANCHOR_DIM)

D_A = 96
D_E = 48

ENV_MLP_IN = ANCHOR_DIM + N_SHELLS * D_A + SHELLPAIR_CLASSES * D_E  # 638
ENV_MLP_HIDDEN = 102

# backend (exact P1 static read-only composer, chemistry-free relation)
PAIR_HIDDEN = int(v0.PAIR_HIDDEN)  # 16
DISTANCE_BUCKETS = 5  # P1 semantics: 1,2,3,4,5+ (disconnected clipped into 5+)
RELATION_WIDTH = DISTANCE_BUCKETS + 1 + 5 + 3 + 1  # 15
GLOBAL_TOPOLOGY_WIDTH = 30  # MolHIV short(15) + long(15) pure topology
GLOBAL_WIDTH = GLOBAL_TOPOLOGY_WIDTH + ATOM_CHEM_DIM + BOND_CHEM_DIM  # 66
TOPOLOGY_IN = 25
TOPOLOGY_HIDDEN = 16
TOPOLOGY_OUT = 8
READER_HIDDEN = v0.READER_HIDDEN  # (13, 13)
BACKEND_DROPOUT = float(v0.BACKEND_DROPOUT)  # 0.05

EPS = float(v0.EPS)

#: frozen relative reconstruction setting (P1 0.25 relative); absolute lambda is
#: calibrated per the pre-registration on the first 512 official-train graphs.
LAMBDA_FACTOR = 0.25

# decision thresholds (pre-registration)
PARAM_BUDGET_MIN = 80000
PARAM_BUDGET_MAX = 130000

BAND_STRONG = 0.80
BAND_COMPETITIVE = 0.78
BAND_PLAUSIBLE = 0.72

VERDICTS = {
    "strong": "M1_TRANSFER_STRONG",
    "competitive": "M1_TRANSFER_COMPETITIVE",
    "plausible": "M1_TRANSFER_PLAUSIBLE",
    "weak": "M1_TRANSFER_WEAK",
}


def ogb_feature_dims() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Runtime OGB categorical schema (no hand-written category counts)."""
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom_dims = tuple(int(value) for value in get_atom_feature_dims())
    bond_dims = tuple(int(value) for value in get_bond_feature_dims())
    return atom_dims, bond_dims


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def chemistry_parameter_count(
    atom_dims: Sequence[int] | None = None, bond_dims: Sequence[int] | None = None
) -> dict[str, int]:
    if atom_dims is None or bond_dims is None:
        atom_dims, bond_dims = ogb_feature_dims()
    atom = sum(int(d) * ATOM_CHEM_DIM for d in atom_dims)
    bond = sum(int(d) * BOND_CHEM_DIM for d in bond_dims)
    return {"atom_embeddings": int(atom), "bond_embeddings": int(bond), "subtotal": int(atom + bond)}


def local_parameter_count(
    atom_dims: Sequence[int] | None = None, bond_dims: Sequence[int] | None = None
) -> dict[str, int]:
    chem = chemistry_parameter_count(atom_dims, bond_dims)
    dictionary = PHI_DIM * K_ATOMS
    node_binding = K_ATOMS * D_A + ATOM_CHEM_DIM * D_A
    edge_binding = (3 * K_ATOMS) * D_E + BOND_CHEM_DIM * D_E
    decoder = ENV_MLP_IN * ENV_MLP_HIDDEN + ENV_MLP_HIDDEN + ENV_MLP_HIDDEN * ENV_DIM + ENV_DIM
    return {
        "chemistry": int(chem["subtotal"]),
        "dictionary": int(dictionary),
        "node_binding": int(node_binding),
        "edge_binding": int(edge_binding),
        "decoder": int(decoder),
        "subtotal": int(chem["subtotal"] + dictionary + node_binding + edge_binding + decoder),
    }


def backend_parameter_count() -> dict[str, int]:
    pair_projection = ENV_DIM * PAIR_HIDDEN
    relation_encoder = RELATION_WIDTH * 32 + 32 + 2 * 32 + 32 * PAIR_HIDDEN + PAIR_HIDDEN
    distance_gate = DISTANCE_BUCKETS * PAIR_HIDDEN
    pair_encoder = 4 * PAIR_HIDDEN * 64 + 64 + 2 * 64 + 64 * PAIR_HIDDEN + PAIR_HIDDEN
    global_encoder = GLOBAL_WIDTH * 32 + 32 + 2 * 32 + 32 * 32 + 32
    topology_encoder = (
        TOPOLOGY_IN * TOPOLOGY_HIDDEN + TOPOLOGY_HIDDEN + TOPOLOGY_HIDDEN * TOPOLOGY_OUT + TOPOLOGY_OUT
    )
    reader = (
        (97 + DISTANCE_BUCKETS * (2 * PAIR_HIDDEN + 1) + 32 + TOPOLOGY_OUT) * 13
        + 13
        + 13 * 13
        + 13
        + 13 * 1
        + 1
    )
    return {
        "pair_projection": int(pair_projection),
        "relation_encoder": int(relation_encoder),
        "distance_gate": int(distance_gate),
        "pair_encoder": int(pair_encoder),
        "global_encoder": int(global_encoder),
        "topology_encoder": int(topology_encoder),
        "reader": int(reader),
        "subtotal": int(
            pair_projection
            + relation_encoder
            + distance_gate
            + pair_encoder
            + global_encoder
            + topology_encoder
            + reader
        ),
    }


def total_parameter_count(
    atom_dims: Sequence[int] | None = None, bond_dims: Sequence[int] | None = None
) -> dict[str, int]:
    local = local_parameter_count(atom_dims, bond_dims)
    backend = backend_parameter_count()
    whole = int(local["subtotal"] + backend["subtotal"])
    return {
        "local_total": int(local["subtotal"]),
        "backend_total": int(backend["subtotal"]),
        "whole_model": whole,
        "p1_reference": 97865,
        "budget_min": int(PARAM_BUDGET_MIN),
        "budget_max": int(PARAM_BUDGET_MAX),
        "within_budget": bool(PARAM_BUDGET_MIN <= whole <= PARAM_BUDGET_MAX),
    }


# ---------------------------------------------------------------------------
# zeroth-order primitive anchor
# ---------------------------------------------------------------------------


def build_anchor_raw(
    root_chem: torch.Tensor,
    occ_root: torch.Tensor,
    atom_chem: torch.Tensor,
    occ_node: torch.Tensor,
    bond_root: torch.Tensor,
    bond_chem: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """``p_i in R^62`` from raw chemistry primitives, one row per root.

    * root chemistry           = ``q_i^chem``                          (24)
    * patch atom mass          = sum_{v in P_i} ``q_v^chem``            (24)
    * patch bond mass          = sum_{e in E_i} ``b_e^chem``            (12)
    * size                     = [log1p|V_i|, log1p|E_i|]               (2)

    Nothing is shell- or shellpair-conditioned.
    """
    dtype = root_chem.dtype
    n = int(n_nodes)
    atom_mass = torch.zeros((n, ATOM_CHEM_DIM), dtype=dtype, device=root_chem.device)
    atom_mass.index_add_(0, occ_root.long(), atom_chem[occ_node.long()])
    bond_mass = torch.zeros((n, BOND_CHEM_DIM), dtype=dtype, device=root_chem.device)
    bond_mass.index_add_(0, bond_root.long(), bond_chem)
    node_count = torch.zeros(n, dtype=dtype, device=root_chem.device).index_add_(
        0, occ_root.long(), torch.ones_like(occ_root, dtype=dtype)
    )
    edge_count = torch.zeros(n, dtype=dtype, device=root_chem.device).index_add_(
        0, bond_root.long(), torch.ones_like(bond_root, dtype=dtype)
    )
    size = torch.stack([torch.log1p(node_count), torch.log1p(edge_count)], dim=1)
    return torch.cat([root_chem, atom_mass, bond_mass, size], dim=1)


def fit_anchor_scaler(anchor_raw: torch.Tensor, floor: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor]:
    values = anchor_raw.to(torch.float64)
    mean = values.mean(dim=0)
    scale = torch.clamp(values.std(dim=0, unbiased=False), min=float(floor))
    return mean.to(torch.float32), scale.to(torch.float32)


def standardize_anchor(anchor_raw: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    dtype = anchor_raw.dtype
    device = anchor_raw.device
    return (anchor_raw - mean.to(device=device, dtype=dtype)) / scale.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# pure-topology pair relation (P1 5-bucket clipping semantics)
# ---------------------------------------------------------------------------


def pure_topology_pair_relation(
    graph: Any,
    patch_nodes: Sequence[Any],
    boundaries: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """MolHIV analogue of P1's 15-D pure-topology relation.

    ``patch_nodes`` / ``boundaries`` are per-root node sets.  Returns
    ``(pair_index [2, n_pairs], relation [n_pairs, 15], clipping_stats)``.
    P1's 5 distance buckets are reused; distance > 5 and disconnected salt
    pairs are clipped into the top (``5+``) bucket.
    """
    from collections import deque

    nodes = [int(node) for node in graph.nodes]
    n = len(nodes)
    dist: list[dict[int, int]] = []
    path_count: list[dict[int, float]] = []
    for source in nodes:
        d = {source: 0}
        pc = {source: 1.0}
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            for neighbour in sorted(graph.neighbors(node)):
                neighbour = int(neighbour)
                if neighbour not in d:
                    d[neighbour] = d[node] + 1
                    pc[neighbour] = 0.0
                    queue.append(neighbour)
                if d[neighbour] == d[node] + 1:
                    pc[neighbour] += pc[node]
        dist.append(d)
        path_count.append(pc)

    sources: list[int] = []
    targets: list[int] = []
    relations: list[np.ndarray] = []
    clipped_disconnected = 0
    clipped_far = 0
    total_pairs = 0
    for left in range(n):
        left_nodes = set(int(v) for v in patch_nodes[left])
        left_boundary = set(int(v) for v in boundaries[left])
        for right in range(left + 1, n):
            total_pairs += 1
            right_nodes = set(int(v) for v in patch_nodes[right])
            right_boundary = set(int(v) for v in boundaries[right])
            if right in dist[left]:
                distance = int(dist[left][right])
                count = float(path_count[left][right])
            else:
                distance = 0
                count = 0.0
                clipped_disconnected += 1
            if distance > DISTANCE_BUCKETS:
                clipped_far += 1
            bucket = min(max(distance, 1), DISTANCE_BUCKETS) - 1
            distance_one_hot = np.zeros(DISTANCE_BUCKETS, dtype=np.float32)
            distance_one_hot[bucket] = 1.0
            intersection = len(left_nodes & right_nodes)
            union = len(left_nodes | right_nodes)
            left_size = len(left_nodes)
            right_size = len(right_nodes)
            overlap = np.asarray(
                [
                    float(intersection) / 20.0,
                    float(intersection) / max(float(union), 1.0),
                    float(intersection) / max(float(min(left_size, right_size)), 1.0),
                    float(intersection) / max(float(max(left_size, right_size)), 1.0),
                    float(abs(left_size - right_size)) / 20.0,
                ],
                dtype=np.float32,
            )
            boundary_intersection = len(left_boundary & right_boundary)
            boundary_union = len(left_boundary | right_boundary)
            boundary_min = min(len(left_boundary), len(right_boundary))
            boundary = np.asarray(
                [
                    float(boundary_intersection) / max(float(boundary_union), 1.0),
                    float(boundary_intersection) / max(float(boundary_min), 1.0),
                    float(boundary_intersection > 0),
                ],
                dtype=np.float32,
            )
            relation = np.concatenate(
                [
                    distance_one_hot,
                    np.asarray([np.log1p(float(distance))], dtype=np.float32),
                    overlap,
                    boundary,
                    np.asarray([np.log1p(float(count))], dtype=np.float32),
                ]
            ).astype(np.float32, copy=False)
            if relation.shape != (RELATION_WIDTH,):
                raise RuntimeError(f"relation width changed: {relation.shape}")
            sources.append(left)
            targets.append(right)
            relations.append(relation)
    pair_index = np.asarray([sources, targets], dtype=np.int64)
    if relations:
        pair_relation = np.stack(relations, axis=0).astype(np.float32, copy=False)
        pair_bucket = np.asarray(
            [
                min(max(int(np.argmax(row[:DISTANCE_BUCKETS])), 0), DISTANCE_BUCKETS - 1)
                for row in relations
            ],
            dtype=np.int64,
        )
    else:
        pair_relation = np.zeros((0, RELATION_WIDTH), dtype=np.float32)
        pair_bucket = np.zeros((0,), dtype=np.int64)
    stats = {
        "n_pairs": int(total_pairs),
        "clipped_disconnected": int(clipped_disconnected),
        "clipped_distance_gt_5": int(clipped_far),
        "clipped_total": int(clipped_disconnected + clipped_far),
        "clipped_fraction": float((clipped_disconnected + clipped_far) / max(total_pairs, 1)),
    }
    return pair_index, pair_relation, stats


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class M1Model(nn.Module):
    """Primitive-only dictionary chemical environment for OGBG-MolHIV."""

    def __init__(self, dictionary: np.ndarray, atom_dims: Sequence[int] | None = None, bond_dims: Sequence[int] | None = None) -> None:
        super().__init__()
        if atom_dims is None or bond_dims is None:
            atom_dims, bond_dims = ogb_feature_dims()
        self.atom_dims = tuple(int(v) for v in atom_dims)
        self.bond_dims = tuple(int(v) for v in bond_dims)
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(int(d), ATOM_CHEM_DIM) for d in self.atom_dims]
        )
        self.bond_embeddings = nn.ModuleList(
            [nn.Embedding(int(d), BOND_CHEM_DIM) for d in self.bond_dims]
        )
        self.D = nn.Parameter(torch.as_tensor(np.asarray(dictionary, dtype=np.float32)).clone())
        self.W_A_S = nn.Parameter(torch.empty(K_ATOMS, D_A))
        self.W_A_C = nn.Parameter(torch.empty(ATOM_CHEM_DIM, D_A))
        self.W_E_S = nn.Parameter(torch.empty(3 * K_ATOMS, D_E))
        self.W_E_C = nn.Parameter(torch.empty(BOND_CHEM_DIM, D_E))
        for parameter in (self.W_A_S, self.W_A_C, self.W_E_S, self.W_E_C):
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5.0))
        self.env_mlp = nn.Sequential(
            nn.Linear(ENV_MLP_IN, ENV_MLP_HIDDEN),
            nn.SiLU(),
            nn.Linear(ENV_MLP_HIDDEN, ENV_DIM),
        )
        self.pair_projection = nn.Linear(ENV_DIM, PAIR_HIDDEN, bias=False)
        self.relation_encoder = zpp._MLPBlock(
            RELATION_WIDTH, max(PAIR_HIDDEN, 32), PAIR_HIDDEN, float(BACKEND_DROPOUT)
        )
        self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, PAIR_HIDDEN)
        self.pair_encoder = zpp._MLPBlock(
            4 * PAIR_HIDDEN, max(2 * PAIR_HIDDEN, 64), PAIR_HIDDEN, float(BACKEND_DROPOUT)
        )
        self.global_encoder = zpp._MLPBlock(
            GLOBAL_WIDTH, max(ENV_DIM // 2, 32), 32, float(BACKEND_DROPOUT)
        )
        self.topology_encoder = nn.Sequential(
            nn.Linear(TOPOLOGY_IN, TOPOLOGY_HIDDEN),
            nn.ReLU(),
            nn.Linear(TOPOLOGY_HIDDEN, TOPOLOGY_OUT),
        )
        self.reader = GenericReader(
            97 + DISTANCE_BUCKETS * (2 * PAIR_HIDDEN + 1) + 32 + TOPOLOGY_OUT, READER_HIDDEN
        )

    # -- chemistry -----------------------------------------------------------

    def atom_chem(self, fields: torch.Tensor) -> torch.Tensor:
        out = None
        for index, embedding in enumerate(self.atom_embeddings):
            value = embedding(fields[:, index].long())
            out = value if out is None else out + value
        assert out is not None
        return out

    def bond_chem(self, fields: torch.Tensor) -> torch.Tensor:
        out = None
        for index, embedding in enumerate(self.bond_embeddings):
            value = embedding(fields[:, index].long())
            out = value if out is None else out + value
        assert out is not None
        return out

    # -- coding --------------------------------------------------------------

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        Dbar = v0.normalized_dictionary(self.D)
        return v0.tied_iht_codes(Dbar, phi, s=SPARSITY, steps=IHT_STEPS)

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        return coord @ v0.normalized_dictionary(self.D).t()

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        phi_hat = self.reconstruct(phi, coord)
        numerator = ((phi - phi_hat) ** 2).sum(dim=1)
        denominator = (phi ** 2).sum(dim=1) + EPS
        return (numerator / denominator).mean()

    # -- local environment ---------------------------------------------------

    def anchor(self, q: torch.Tensor, b: torch.Tensor, data: Any) -> torch.Tensor:
        n = int(q.shape[0])
        raw = build_anchor_raw(
            q,
            data.env_occ_root,
            q,
            data.env_occ_node,
            data.env_bond_root,
            b,
            n,
        )
        mean = getattr(self, "anchor_mean", None)
        scale = getattr(self, "anchor_scale", None)
        if mean is None or scale is None:
            return raw
        return standardize_anchor(raw, mean, scale)

    def environments(self, coord: torch.Tensor, data: Any, q: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n = int(coord.shape[0])
        anchor = self.anchor(q, b, data)
        # node dictionary-chemistry binding, per-shell SUM
        c = coord[data.env_occ_node.to(coord.device)]
        qc = q[data.env_occ_node.to(coord.device)]
        u = (c @ self.W_A_S) * (qc @ self.W_A_C) / math.sqrt(float(D_A))
        flat = torch.zeros((n * N_SHELLS, D_A), device=u.device, dtype=u.dtype)
        flat.index_add_(0, data.env_occ_root.to(coord.device) * N_SHELLS + data.env_occ_shell.to(coord.device), u)
        A = flat.view(n, N_SHELLS * D_A)
        # edge dictionary-role x bond-chemistry binding, per-shellpair SUM
        bond_u = data.env_bond_u.to(coord.device)
        bond_v = data.env_bond_v.to(coord.device)
        cu = coord[bond_u]
        cv = coord[bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        bo = b
        ue = (g @ self.W_E_S) * (bo @ self.W_E_C) / math.sqrt(float(D_E))
        flat_e = torch.zeros((n * SHELLPAIR_CLASSES, D_E), device=ue.device, dtype=ue.dtype)
        flat_e.index_add_(
            0,
            data.env_bond_root.to(coord.device) * SHELLPAIR_CLASSES + data.env_bond_shellpair.to(coord.device),
            ue,
        )
        E_edge = flat_e.view(n, SHELLPAIR_CLASSES * D_E)
        z = torch.cat([anchor, A, E_edge], dim=1)
        return self.env_mlp(z)

    # -- forward --------------------------------------------------------------

    def forward(self, data: Any, *, return_aux: bool = False):
        phi = data.dict_phi
        coord = self.code(phi)
        q = self.atom_chem(data.dict_atom)
        b = self.bond_chem(data.env_bond_fields)
        E = self.environments(coord, data, q, b)

        n_graphs = int(data.global_topology.shape[0])
        batch = data.batch
        unary = v0.pool_moments(E, batch, n_graphs)
        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        left = u[source]
        right = u[target]
        relation = self.relation_encoder(data.pair_relation)
        product = left * right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_input = torch.cat(
            [left + right, torch.abs(left - right), product * gate, relation], dim=1
        )
        pair_value = self.pair_encoder(pair_input)
        pair_batch = batch[source]
        relation_readout = v0.pool_pair_moments(pair_value, pair_batch, data.pair_bucket, n_graphs)

        # whole-molecule chemistry-only aggregates (no shell-conditioned stats)
        atom_counts = torch.bincount(batch, minlength=n_graphs).to(q.dtype).clamp(min=1.0).unsqueeze(1)
        atom_sum = torch.zeros((n_graphs, ATOM_CHEM_DIM), device=q.device, dtype=q.dtype)
        atom_sum.index_add_(0, batch, q)
        atom_mean = atom_sum / atom_counts
        mol_bond_graph = data.mol_bond_graph.to(q.device)
        b_mol = self.bond_chem(data.mol_bond_fields)
        bond_counts = torch.bincount(mol_bond_graph, minlength=n_graphs).to(b_mol.dtype).clamp(min=1.0).unsqueeze(1)
        bond_sum = torch.zeros((n_graphs, BOND_CHEM_DIM), device=b_mol.device, dtype=b_mol.dtype)
        bond_sum.index_add_(0, mol_bond_graph, b_mol)
        bond_mean = bond_sum / bond_counts
        global_context = torch.cat([data.global_topology.to(q.dtype), atom_mean, bond_mean], dim=1)

        graph_hidden = self.global_encoder(global_context)
        topology = self.topology_encoder(data.topology_features)
        unified = torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)
        prediction = self.reader(unified).view(-1)
        if return_aux:
            return prediction, {"E": E, "coord": coord, "phi": phi, "anchor": None}
        return prediction


def build_model(
    dictionary: np.ndarray,
    seed: int = 0,
    anchor_mean: torch.Tensor | None = None,
    anchor_scale: torch.Tensor | None = None,
    atom_dims: Sequence[int] | None = None,
    bond_dims: Sequence[int] | None = None,
) -> M1Model:
    torch.manual_seed(int(seed))
    model = M1Model(dictionary, atom_dims=atom_dims, bond_dims=bond_dims)
    if anchor_mean is not None:
        model.register_buffer("anchor_mean", anchor_mean.detach().to(torch.float32).clone())
    if anchor_scale is not None:
        model.register_buffer("anchor_scale", anchor_scale.detach().to(torch.float32).clone())
    return model


# ---------------------------------------------------------------------------
# batch container / collate (self-contained; no PyG Batch surprises)
# ---------------------------------------------------------------------------

_NODE_KEYS = ("dict_phi", "dict_atom")
_OCC_NODE_KEYS = ("env_occ_node", "env_occ_root", "env_occ_shell")
_OCC_BOND_KEYS = (
    "env_bond_root",
    "env_bond_shellpair",
    "env_bond_u",
    "env_bond_v",
    "env_bond_fields",
)
_GRAPH_KEYS = ("global_topology", "topology_features")
_PAIR_KEYS = ("pair_relation", "pair_bucket")


class MolhivBatch:
    """Minimal concatenated batch for the M1 local environment."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def to(self, device: Any) -> "MolhivBatch":
        device = torch.device(device)
        for key, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)
        return self

    def __len__(self) -> int:
        return int(self.y.shape[0])


def env_collate(data_list: Sequence[Any]) -> MolhivBatch:
    """Concatenate per-graph M1 payloads, offsetting local indices by graph."""
    data_list = list(data_list)
    counts = [int(d.dict_phi.shape[0]) for d in data_list]
    occ_counts = [int(d.env_occ_node.shape[0]) for d in data_list]
    bond_counts = [int(d.env_bond_root.shape[0]) for d in data_list]
    mol_bond_counts = [int(d.mol_bond_fields.shape[0]) for d in data_list]
    ptr: list[int] = [0]
    for count in counts:
        ptr.append(ptr[-1] + count)

    def _cat(name: str) -> torch.Tensor:
        return torch.cat([getattr(d, name) for d in data_list], dim=0)

    def _offset(name: str, sizes: Sequence[int]) -> torch.Tensor:
        out = _cat(name)
        position = 0
        for index, size in enumerate(sizes):
            size = int(size)
            if size:
                out[position : position + size] += int(ptr[index])
            position += size
        return out

    batch = torch.repeat_interleave(
        torch.arange(len(data_list), dtype=torch.long), torch.as_tensor(counts, dtype=torch.long)
    )
    mol_bond_graph = torch.repeat_interleave(
        torch.arange(len(data_list), dtype=torch.long),
        torch.as_tensor(mol_bond_counts, dtype=torch.long),
    )
    pair_index = torch.cat([d.pair_index for d in data_list], dim=1).clone()
    pair_offset = torch.repeat_interleave(
        torch.arange(len(data_list), dtype=torch.long),
        torch.as_tensor([int(d.pair_index.shape[1]) for d in data_list], dtype=torch.long),
    )
    pair_ptr = torch.as_tensor(ptr, dtype=torch.long)[pair_offset]
    pair_index = pair_index + pair_ptr.unsqueeze(0)

    fields: dict[str, Any] = {
        "dict_phi": _cat("dict_phi"),
        "dict_atom": _cat("dict_atom"),
        "env_occ_node": _offset("env_occ_node", occ_counts),
        "env_occ_root": _offset("env_occ_root", occ_counts),
        "env_occ_shell": _cat("env_occ_shell"),
        "env_bond_root": _offset("env_bond_root", bond_counts),
        "env_bond_shellpair": _cat("env_bond_shellpair"),
        "env_bond_u": _offset("env_bond_u", bond_counts),
        "env_bond_v": _offset("env_bond_v", bond_counts),
        "env_bond_fields": _cat("env_bond_fields"),
        "mol_bond_fields": _cat("mol_bond_fields"),
        "mol_bond_graph": mol_bond_graph,
        "global_topology": torch.stack([d.global_topology for d in data_list], dim=0),
        "topology_features": torch.stack([d.topology_features for d in data_list], dim=0),
        "pair_index": pair_index,
        "pair_relation": _cat("pair_relation"),
        "pair_bucket": _cat("pair_bucket"),
        "y": torch.stack([d.y.reshape(()) for d in data_list], dim=0),
        "batch": batch,
        "ptr": torch.as_tensor(ptr, dtype=torch.long),
    }
    return MolhivBatch(**fields)


def make_env_loader(graphs: Sequence[Any], batch_size: int, shuffle: bool, seed: int) -> Any:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return torch.utils.data.DataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=env_collate,
    )


__all__ = [
    "PROTOCOL_VERSION",
    "PHI_DIM",
    "K_ATOMS",
    "SPARSITY",
    "IHT_STEPS",
    "ATOM_CHEM_DIM",
    "BOND_CHEM_DIM",
    "ANCHOR_DIM",
    "D_A",
    "D_E",
    "ENV_MLP_IN",
    "ENV_MLP_HIDDEN",
    "ENV_DIM",
    "RELATION_WIDTH",
    "DISTANCE_BUCKETS",
    "GLOBAL_WIDTH",
    "GLOBAL_TOPOLOGY_WIDTH",
    "TOPOLOGY_IN",
    "TOPOLOGY_OUT",
    "LAMBDA_FACTOR",
    "PARAM_BUDGET_MIN",
    "PARAM_BUDGET_MAX",
    "VERDICTS",
    "ogb_feature_dims",
    "chemistry_parameter_count",
    "local_parameter_count",
    "backend_parameter_count",
    "total_parameter_count",
    "build_anchor_raw",
    "fit_anchor_scaler",
    "standardize_anchor",
    "pure_topology_pair_relation",
    "M1Model",
    "build_model",
    "env_collate",
    "make_env_loader",
]
