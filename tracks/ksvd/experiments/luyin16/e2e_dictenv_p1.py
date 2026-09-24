"""E2E-DictEnv-P1 — Primitive-Only End-to-End Dictionary Chemical Environment.

Round: ``e2e_dictenv_p1``.  Study ``zinc-context-gap``.
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_p1_preregistration.md``.
Prior-artifact audit:
``tracks/ksvd/notes/e2e_dictenv_p1_prior_artifact_audit.md``.

P1 removes the T1 handcrafted structural-chemistry bypass and asks whether a
shared, task-coupled sparse structural dictionary can *be* the organizer of the
local chemical environment:

    pure topology phi_v^65 -> D -> alpha_v^32
    (alpha_v, q_v, shell)        -> node role x chemistry -> per-shell slots
    (alpha_u, alpha_v, b_uv, sp) -> edge role x bond type  -> per-shellpair slots
    p_i^62 (zeroth-order identity / mass / size)  +  A_i^288  +  E_i^288
        -> SiLU MLP 638 -> 102 -> 48 -> E_i

Hard purity contract: no ``patch_cont`` / ``atom_shell`` / ``bond_shell`` (or any
handcrafted shell x chemistry histogram) anywhere in the model, no pair
chemistry in the pair relation, no message passing, no recurrence, no
pair->centre, no attention, no context writeback.  ``E_i`` is frozen after
formation and the static read-only composer only ever sees ``E_i``, ``E_j`` and
the 15-D pure-topology relation.

Arms (everything matched except the coding operator)::

    S (SparseDictEnv):  alpha = IHT_{s=8}(Dbar, phi), 10 steps, exact top-s
    D (DenseTiedEnv):   z     = phi @ Pbar            (dense tied coordinate)

Both arms: 97,865 total parameters; ``P_init == D_init`` == the SDB-v0 frozen
K-SVD artifact (``results/sdb_v0/dictionary.pt``, 65x32, sha256
``925d573a58083c3c20f36980dec9bba77dff592b30ea1abf11098ca3a7b7487a``).

``official_test_loaded = false`` always.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_sdb_v0 as zsdb
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen configuration (pre-registered; no sweep)
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "e2e_dictenv_p1"

PHI_DIM = int(v0.PHI_DIM)  # 65
K_ATOMS = int(v0.K_ATOMS)  # 32
SPARSITY = int(v0.SPARSITY)  # 8
IHT_STEPS = int(v0.IHT_STEPS)  # 10
ATOM_CATEGORIES = int(v0.ATOM_CATEGORIES)  # 28
BOND_CATEGORIES = int(v0.BOND_CATEGORIES)  # 4
N_SHELLS = int(v0.N_SHELLS)  # 3
PATCH_RADIUS = int(v0.PATCH_RADIUS)  # 2
SHELLPAIR_CLASSES = int(v0.SHELLPAIR_CLASSES)  # 6
ENV_DIM = int(v0.ENV_DIM)  # 48

#: exact zeroth-order primitive anchor, in order:
#:   root identity (28) | patch atom mass (28) | patch bond mass (4) | size (2)
ANCHOR_DIM = ATOM_CATEGORIES + ATOM_CATEGORIES + BOND_CATEGORIES + 2  # 62
ANCHOR_ROOT = slice(0, ATOM_CATEGORIES)
ANCHOR_ATOM_MASS = slice(ATOM_CATEGORIES, 2 * ATOM_CATEGORIES)
ANCHOR_BOND_MASS = slice(2 * ATOM_CATEGORIES, 2 * ATOM_CATEGORIES + BOND_CATEGORIES)
ANCHOR_SIZE = slice(2 * ATOM_CATEGORIES + BOND_CATEGORIES, ANCHOR_DIM)

#: node dictionary-chemistry binding width
D_A = 96
#: edge dictionary-role / bond-chemistry binding width
D_E = 48

ENV_MLP_IN = ANCHOR_DIM + N_SHELLS * D_A + SHELLPAIR_CLASSES * D_E  # 638
ENV_MLP_HIDDEN = 102

# backend (exact FEC-S1 strict-static backend, chemistry-free pair relation)
PAIR_HIDDEN = int(v0.PAIR_HIDDEN)  # 16
DISTANCE_BUCKETS = int(v0.DISTANCE_BUCKETS)  # 5
RELATION_WIDTH_RAW = int(v0.RELATION_WIDTH)  # 23 (historical)
#: historical pure-topology relation coordinates: distance one-hot, log distance,
#: overlap, boundary, log path count (drops path-bond-mean 14:18 and adjacent 19:23)
P1_RELATION_INDICES: tuple[int, ...] = tuple(range(14)) + (18,)
RELATION_WIDTH = len(P1_RELATION_INDICES)  # 15
GLOBAL_WIDTH = int(v0.GLOBAL_WIDTH)  # 62
TOPOLOGY_IN = int(v0.TOPOLOGY_IN)  # 25
TOPOLOGY_HIDDEN = int(v0.TOPOLOGY_HIDDEN)  # 16
TOPOLOGY_OUT = int(v0.TOPOLOGY_OUT)  # 8
READER_HIDDEN = v0.READER_HIDDEN  # (13, 13)
BACKEND_DROPOUT = float(v0.BACKEND_DROPOUT)  # 0.05

EPS = float(v0.EPS)

SPARSE_ARM = v0.SPARSE_ARM
DENSE_ARM = v0.DENSE_ARM
ARMS = v0.ARMS

#: frozen T1 winner reconstruction weight (no recalibration, no sweep)
LAMBDA_REC = 33.95873017865987

# decision thresholds (pre-registration)
GATE_DICT_SPECIFIC = 0.003
GATE_ZERO = 0.030
GATE_NODE_SHUFFLE = 0.010
GATE_ALL_SHUFFLE = 0.015
BAND_STRONG_MAX = 0.135
BAND_VIABLE_MAX = 0.145
HEALTH_MIN_ACTIVE = 24
HEALTH_MIN_EFFECTIVE = 8
HEALTH_MAX_ATOM_SHARE = 0.50
HEALTH_MIN_INITIAL_RETENTION = 0.80
HEALTH_MAX_VALID_RECONSTRUCTION = 0.01
PARAM_BUDGET_MIN = 90000
PARAM_BUDGET_MAX = 105000

VERDICTS = {
    "strong": "P1_STRONG_CLEAN_DICTIONARY_CORE_SUPPORTED",
    "viable": "P1_VIABLE_CLEAN_DICTIONARY_CORE_SUPPORTED",
    "mechanism_absolute_weak": "P1_CLEAN_CORE_MECHANISM_SUPPORTED_ABSOLUTE_WEAK",
    "not_specific": "P1_CLEAN_ENVIRONMENT_VIABLE_BUT_DICTIONARY_NOT_SPECIFIC",
    "not_load_bearing": "P1_DICTIONARY_NOT_LOAD_BEARING",
    "mechanism_collapsed": "P1_MECHANISM_COLLAPSED",
    "specificity_unstable": "P1_DICTIONARY_SPECIFICITY_UNSTABLE",
}


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def local_parameter_count() -> dict[str, int]:
    dictionary = PHI_DIM * K_ATOMS
    node_binding = K_ATOMS * D_A + ATOM_CATEGORIES * D_A
    edge_binding = (2 * K_ATOMS + K_ATOMS) * D_E + BOND_CATEGORIES * D_E
    decoder = ENV_MLP_IN * ENV_MLP_HIDDEN + ENV_MLP_HIDDEN + ENV_MLP_HIDDEN * ENV_DIM + ENV_DIM
    return {
        "dictionary": int(dictionary),
        "node_binding": int(node_binding),
        "edge_binding": int(edge_binding),
        "decoder": int(decoder),
        "subtotal": int(dictionary + node_binding + edge_binding + decoder),
    }


def backend_parameter_count() -> dict[str, int]:
    pair_projection = ENV_DIM * PAIR_HIDDEN  # Linear(48,16,bias=False)
    relation_encoder = (
        RELATION_WIDTH * 32 + 32 + 2 * 32 + 32 * PAIR_HIDDEN + PAIR_HIDDEN
    )  # _MLPBlock(15,32,16) incl. LayerNorm
    distance_gate = DISTANCE_BUCKETS * PAIR_HIDDEN
    pair_encoder = (
        4 * PAIR_HIDDEN * 64 + 64 + 2 * 64 + 64 * PAIR_HIDDEN + PAIR_HIDDEN
    )  # _MLPBlock(64,64,16)
    global_encoder = GLOBAL_WIDTH * 32 + 32 + 2 * 32 + 32 * 32 + 32  # _MLPBlock(62,32,32)
    topology_encoder = (
        TOPOLOGY_IN * TOPOLOGY_HIDDEN + TOPOLOGY_HIDDEN + TOPOLOGY_HIDDEN * TOPOLOGY_OUT + TOPOLOGY_OUT
    )
    reader = (97 + DISTANCE_BUCKETS * (2 * PAIR_HIDDEN + 1) + 32 + TOPOLOGY_OUT) * 13 + 13 + 13 * 13 + 13 + 13 * 1 + 1
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


def total_parameter_count() -> dict[str, int]:
    local = local_parameter_count()
    backend = backend_parameter_count()
    whole = int(local["subtotal"] + backend["subtotal"])
    return {
        "local_total": int(local["subtotal"]),
        "backend_total": int(backend["subtotal"]),
        "whole_model": whole,
        "fec_s1_reference": 66170,
        "difference_vs_fec_s1": int(whole - 66170),
        "budget_min": int(PARAM_BUDGET_MIN),
        "budget_max": int(PARAM_BUDGET_MAX),
        "within_budget": bool(PARAM_BUDGET_MIN <= whole <= PARAM_BUDGET_MAX),
    }


# ---------------------------------------------------------------------------
# zeroth-order primitive anchor (root identity / unconditioned mass / size)
# ---------------------------------------------------------------------------


def build_anchor_raw(
    atom: torch.Tensor,
    occ_node: torch.Tensor,
    occ_root: torch.Tensor,
    bond_root: torch.Tensor,
    bond_type: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Build ``p_i in R^62`` from raw primitives, one row per root (= node).

    * root identity    = one-hot ``q_i``                       (28)
    * patch atom mass  = sum_{v in P_i} one-hot ``q_v``       (28)  (unconditioned)
    * patch bond mass  = sum_{e in E_i} one-hot ``b_e``       (4)   (unconditioned)
    * size             = [log1p|V_i|, log1p|E_i|]              (2)

    Nothing here is shell- or shellpair-conditioned: the mass sums run over the
    whole patch.  ``n_nodes`` roots are produced; row ``i`` corresponds to the
    local node index ``i``.
    """
    dtype = torch.float32
    n = int(n_nodes)
    q = F.one_hot(atom.long(), num_classes=ATOM_CATEGORIES).to(dtype)  # [N, 28]
    root_identity = q
    atom_mass = torch.zeros((n, ATOM_CATEGORIES), dtype=dtype, device=atom.device)
    atom_mass.index_add_(0, occ_root.long(), q[occ_node.long()])
    bond_mass = torch.zeros((n, BOND_CATEGORIES), dtype=dtype, device=atom.device)
    bond_mass.index_add_(
        0,
        bond_root.long(),
        F.one_hot(bond_type.long(), num_classes=BOND_CATEGORIES).to(dtype),
    )
    node_count = torch.zeros(n, dtype=dtype, device=atom.device).index_add_(
        0, occ_root.long(), torch.ones_like(occ_root, dtype=dtype)
    )
    edge_count = torch.zeros(n, dtype=dtype, device=atom.device).index_add_(
        0, bond_root.long(), torch.ones_like(bond_root, dtype=dtype)
    )
    size = torch.stack([torch.log1p(node_count), torch.log1p(edge_count)], dim=1)
    return torch.cat([root_identity, atom_mass, bond_mass, size], dim=1)


def fit_anchor_scaler(anchor_raw: torch.Tensor, floor: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Train-only per-coordinate ``(mean, scale)`` standardizer for the anchor."""
    values = anchor_raw.to(torch.float64)
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False)
    scale = torch.clamp(scale, min=float(floor))
    return mean.to(torch.float32), scale.to(torch.float32)


def standardize_anchor(anchor_raw: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (anchor_raw - mean.to(anchor_raw.dtype)) / scale.to(anchor_raw.dtype)


# ---------------------------------------------------------------------------
# root-local incidence helpers for the edge branch / interventions
# ---------------------------------------------------------------------------


def shuffled_occ_node_for_molecule(
    occ_node: torch.Tensor,
    occ_root: torch.Tensor,
    occ_shell: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Permute the node used for the *code* gather inside every (root, shell).

    ``q_v`` and the shell stay tied to the original occurrence, so the
    ``alpha_v <-> q_v`` correspondence is broken while the node set, the alpha
    multiset, the q multiset and the shell grouping are all preserved.
    """
    occ_node_np = occ_node.numpy()
    occ_root_np = occ_root.numpy()
    occ_shell_np = occ_shell.numpy()
    out = occ_node_np.copy()
    rng = np.random.default_rng(int(seed))
    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(len(occ_node_np)):
        groups.setdefault((int(occ_root_np[index]), int(occ_shell_np[index])), []).append(index)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        permuted = rng.permutation(len(positions))
        original_nodes = occ_node_np[positions]
        out[positions] = original_nodes[permuted]
    return torch.as_tensor(out, dtype=torch.long)


def shuffled_bond_endpoints_for_molecule(
    bond_root: torch.Tensor,
    bond_shellpair: torch.Tensor,
    bond_u: torch.Tensor,
    bond_v: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute the dictionary-role endpoints inside every (root, shellpair).

    The bond-type at each position is untouched, so the structural edge-role
    multiset ``{g_uv}`` and the bond-type multiset are both preserved while the
    ``g_uv <-> b_uv`` correspondence is broken.
    """
    root_np = bond_root.numpy()
    sp_np = bond_shellpair.numpy()
    u_np = bond_u.numpy()
    v_np = bond_v.numpy()
    u_out = u_np.copy()
    v_out = v_np.copy()
    rng = np.random.default_rng(int(seed))
    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(len(root_np)):
        groups.setdefault((int(root_np[index]), int(sp_np[index])), []).append(index)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        permuted = rng.permutation(len(positions))
        u_out[positions] = u_np[positions][permuted]
        v_out[positions] = v_np[positions][permuted]
    return (
        torch.as_tensor(u_out, dtype=torch.long),
        torch.as_tensor(v_out, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class P1Model(nn.Module):
    """Primitive-only dictionary chemical environment (Sparse / DenseTied)."""

    def __init__(self, arm: str = SPARSE_ARM) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = str(arm)
        D_ksvd, _D_rand, _pca = zsdb.load_dictionary()
        self.D = nn.Parameter(
            torch.as_tensor(np.asarray(D_ksvd, dtype=np.float32).copy(), dtype=torch.float32).clone()
        )
        # node dictionary-chemistry binding (bias-free)
        self.W_A_S = nn.Parameter(torch.empty(K_ATOMS, D_A))
        self.W_A_C = nn.Parameter(torch.empty(ATOM_CATEGORIES, D_A))
        # edge dictionary-role / bond-chemistry binding (bias-free)
        self.W_E_S = nn.Parameter(torch.empty(3 * K_ATOMS, D_E))
        self.W_E_C = nn.Parameter(torch.empty(BOND_CATEGORIES, D_E))
        for parameter in (self.W_A_S, self.W_A_C, self.W_E_S, self.W_E_C):
            self._init_factor_(parameter)
        # environment decoder 638 -> 102 -> 48 (SiLU)
        self.env_mlp = nn.Sequential(
            nn.Linear(ENV_MLP_IN, ENV_MLP_HIDDEN),
            nn.SiLU(),
            nn.Linear(ENV_MLP_HIDDEN, ENV_DIM),
        )
        # static composition backend (exact FEC-S1 strict-static, chemistry-free relation)
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

    @staticmethod
    def _init_factor_(parameter: nn.Parameter) -> None:
        nn.init.kaiming_uniform_(parameter, a=math.sqrt(5.0))

    # -- coding (identical to v0 semantics) ----------------------------------

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        Dbar = v0.normalized_dictionary(self.D)
        if self.arm == SPARSE_ARM:
            return v0.tied_iht_codes(Dbar, phi, s=SPARSITY, steps=IHT_STEPS)
        return phi @ Dbar

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        Dbar = v0.normalized_dictionary(self.D)
        return coord @ Dbar.t()

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        phi_hat = self.reconstruct(phi, coord)
        numerator = ((phi - phi_hat) ** 2).sum(dim=1)
        denominator = (phi ** 2).sum(dim=1) + EPS
        return (numerator / denominator).mean()

    # -- local environment ---------------------------------------------------

    def node_environment(self, coord: torch.Tensor, data: Any, occ_coord_node: torch.Tensor | None) -> torch.Tensor:
        """Dictionary-role x atom-chemistry per-shell slots ``A_i in R^288``."""
        n = int(coord.shape[0])
        q = F.one_hot(data.dict_atom, num_classes=ATOM_CATEGORIES).to(coord.dtype)
        if occ_coord_node is None:
            occ_coord_node = data.env_occ_node
        occ_coord_node = occ_coord_node.to(coord.device)
        c = coord[occ_coord_node]
        qc = q[data.env_occ_node]
        u = (c @ self.W_A_S) * (qc @ self.W_A_C) / math.sqrt(float(D_A))
        flat = torch.zeros(
            (n * N_SHELLS, D_A), device=u.device, dtype=u.dtype
        )
        flat.index_add_(0, data.env_occ_root * N_SHELLS + data.env_occ_shell, u)
        return flat.view(n, N_SHELLS * D_A)

    def edge_environment(
        self,
        coord: torch.Tensor,
        data: Any,
        bond_u: torch.Tensor | None,
        bond_v: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dictionary-derived endpoint role x bond chemistry ``E_i in R^288``."""
        n = int(coord.shape[0])
        if bond_u is None:
            bond_u = data.env_bond_u
        if bond_v is None:
            bond_v = data.env_bond_v
        bond_u = bond_u.to(coord.device)
        bond_v = bond_v.to(coord.device)
        cu = coord[bond_u]
        cv = coord[bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)  # [m, 96]
        b = F.one_hot(data.env_bond_type, num_classes=BOND_CATEGORIES).to(coord.dtype)
        u = (g @ self.W_E_S) * (b @ self.W_E_C) / math.sqrt(float(D_E))
        flat = torch.zeros(
            (n * SHELLPAIR_CLASSES, D_E), device=u.device, dtype=u.dtype
        )
        flat.index_add_(0, data.env_bond_root * SHELLPAIR_CLASSES + data.env_bond_shellpair, u)
        return flat.view(n, SHELLPAIR_CLASSES * D_E)

    def environments(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        occ_coord_node: torch.Tensor | None = None,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        anchor = data.anchor.to(coord.dtype)
        if int(anchor.shape[1]) != ANCHOR_DIM:
            raise RuntimeError(f"anchor width {int(anchor.shape[1])} != {ANCHOR_DIM}")
        A = self.node_environment(coord, data, occ_coord_node)
        E_edge = self.edge_environment(coord, data, bond_u, bond_v)
        z = torch.cat([anchor, A, E_edge], dim=1)
        return self.env_mlp(z)

    # -- forward --------------------------------------------------------------

    def forward(
        self,
        data: Any,
        *,
        coord_zero: bool = False,
        occ_coord_node: torch.Tensor | None = None,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        phi = data.dict_phi
        coord = self.code(phi)
        if coord_zero:
            coord = torch.zeros_like(coord)
        E = self.environments(
            coord,
            data,
            occ_coord_node=occ_coord_node,
            bond_u=bond_u,
            bond_v=bond_v,
        )

        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = v0.pool_moments(E, batch, n_graphs)

        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        projected_left = u[source]
        projected_right = u[target]
        relation = self.relation_encoder(data.pair_relation[:, list(P1_RELATION_INDICES)])
        product = projected_left * projected_right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_input = torch.cat(
            [
                projected_left + projected_right,
                torch.abs(projected_left - projected_right),
                product * gate,
                relation,
            ],
            dim=1,
        )
        pair_value = self.pair_encoder(pair_input)
        pair_batch = batch[source]
        relation_readout = v0.pool_pair_moments(pair_value, pair_batch, data.pair_bucket, n_graphs)

        graph_hidden = self.global_encoder(data.global_context)
        topology = self.topology_encoder(data.topology_features)
        unified = torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)
        prediction = self.reader(unified)
        aux = {"E": E, "coord": coord, "phi": phi}
        if return_aux:
            return prediction, aux
        return prediction


def build_model(
    arm: str,
    seed: int = 0,
    reference_state: Mapping[str, torch.Tensor] | None = None,
) -> P1Model:
    """Build an arm; when ``reference_state`` is given, copy every shared key."""
    torch.manual_seed(int(seed))
    model = P1Model(arm=arm)
    if reference_state is not None:
        with torch.no_grad():
            state = model.state_dict()
            for key, value in reference_state.items():
                if key in state and state[key].shape == value.shape:
                    state[key].copy_(value)
    return model


# ---------------------------------------------------------------------------
# custom collate: local occurrence / endpoint indices -> global node indices
# ---------------------------------------------------------------------------

_OCC_OFFSET_BY_COUNT = {
    "env_occ_node": "occ",
    "env_occ_root": "occ",
    "env_occ_coord_node": "occ",
    "env_bond_root": "bond",
    "env_bond_u": "bond",
    "env_bond_v": "bond",
    "env_bond_u_shuffled": "bond",
    "env_bond_v_shuffled": "bond",
}


def _offset_occurrences(value: torch.Tensor, counts: Sequence[int], ptr: Sequence[int]) -> torch.Tensor:
    out = value.clone()
    position = 0
    for index, count in enumerate(counts):
        count = int(count)
        if count:
            out[position : position + count] += int(ptr[index])
        position += count
    return out


def env_collate(data_list: Sequence[Any]) -> Any:
    """PyG batch with per-graph node offsets applied to occurrence/endpoint fields."""
    from torch_geometric.data import Batch

    data_list = list(data_list)
    base = Batch.from_data_list(data_list)
    ptr = base.ptr.tolist()
    occ_counts = [int(d.env_occ_node.shape[0]) for d in data_list]
    bond_counts = [int(d.env_bond_root.shape[0]) for d in data_list]
    for name, kind in _OCC_OFFSET_BY_COUNT.items():
        if not hasattr(base, name) or getattr(base, name) is None:
            continue
        counts = bond_counts if kind == "bond" else occ_counts
        setattr(base, name, _offset_occurrences(getattr(base, name), counts, ptr))
    return base


def make_env_loader(
    graphs: Sequence[Any], batch_size: int, shuffle: bool, seed: int
) -> Any:
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
    "ATOM_CATEGORIES",
    "BOND_CATEGORIES",
    "N_SHELLS",
    "PATCH_RADIUS",
    "SHELLPAIR_CLASSES",
    "ENV_DIM",
    "ANCHOR_DIM",
    "D_A",
    "D_E",
    "ENV_MLP_IN",
    "ENV_MLP_HIDDEN",
    "RELATION_WIDTH_RAW",
    "RELATION_WIDTH",
    "P1_RELATION_INDICES",
    "LAMBDA_REC",
    "SPARSE_ARM",
    "DENSE_ARM",
    "ARMS",
    "GATE_DICT_SPECIFIC",
    "GATE_ZERO",
    "GATE_NODE_SHUFFLE",
    "GATE_ALL_SHUFFLE",
    "BAND_STRONG_MAX",
    "BAND_VIABLE_MAX",
    "VERDICTS",
    "local_parameter_count",
    "backend_parameter_count",
    "total_parameter_count",
    "build_anchor_raw",
    "fit_anchor_scaler",
    "standardize_anchor",
    "shuffled_occ_node_for_molecule",
    "shuffled_bond_endpoints_for_molecule",
    "P1Model",
    "build_model",
    "env_collate",
    "make_env_loader",
]
