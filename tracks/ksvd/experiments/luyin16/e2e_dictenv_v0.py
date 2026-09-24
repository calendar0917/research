"""E2E-DictEnv-v0 — End-to-End Sparse Dictionary-Core Chemical Environment.

Round: ``e2e_dictenv_v0``.  Study ``zinc-context-gap``.
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_v0_preregistration.md``.
Prior-artifact audit: ``tracks/ksvd/notes/e2e_dictenv_v0_prior_artifact_audit.md``.

The only fine-grained learned structural coordinate inside local environment
formation is the tied sparse dictionary code (or, in the matched control, the
tied dense coordinate)::

    phi_v^65 -> D -> alpha_v^32 -> (alpha_v, q_v, shell_iv) -> E_i -> static composition -> y

Hard purity contract: no message passing, no recurrence, no pair->centre, no
relation refresh, no attention, no context writeback, no typed/parent lookup,
no FEC-S1 146->214->24 adapter, no 146-D patch_cont anywhere in the model.
Environment E_i is frozen after formation; pairs are composed once, read-only.

Arms (everything matched except the coding operator)::

    S (SparseDictEnv):      alpha = IHT_{s=8}(Dbar, phi), 10 steps, exact top-s
    D (DenseTiedEnv):       z     = phi @ Pbar            (dense tied coordinate)

Both arms: 66,158 total parameters; P init == D init == the SDB-v0 frozen
K-SVD artifact (``results/sdb_v0/dictionary.pt``, 65x32, sha256 begins
``925d573a5808...``).

``official_test_loaded = false`` always.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.code import tccd_v0
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_sdb_v0 as zsdb
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from torch_geometric.data import Batch

# ---------------------------------------------------------------------------
# frozen configuration (pre-registered; no sweep)
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "e2e_dictenv_v0"

PHI_DIM = int(sdb.PHI_DIM)  # 65
K_ATOMS = int(sdb.K_ATOMS)  # 32
SPARSITY = int(sdb.SPARSITY)  # 8
IHT_STEPS = 10
ATOM_CATEGORIES = int(sdb.ATOM_CATEGORIES)  # 28
BOND_CATEGORIES = 4
N_SHELLS = 3
PATCH_RADIUS = 2
SHELLPAIR_CLASSES = 6

ANCHOR_DIM = 1  # the constant-1 chemistry-marginal anchor
R_ATOM = 64
R_BOND = 16
ENV_MLP_IN = R_ATOM + R_BOND + 6  # 86
ENV_MLP_HIDDEN = 329
ENV_DIM = 48

# backend (exact FEC-S1 strict-static backend)
PAIR_HIDDEN = 16
DISTANCE_BUCKETS = int(zpp.DISTANCE_BUCKETS)  # 5
RELATION_WIDTH = int(zpp.RELATION_WIDTH)  # 23
GLOBAL_WIDTH = int(zpp.GLOBAL_WIDTH)  # 62
TOPOLOGY_IN = 25
TOPOLOGY_HIDDEN = 16
TOPOLOGY_OUT = 8
READER_HIDDEN = (13, 13)
BACKEND_DROPOUT = 0.05

EPS = 1.0e-12

SPARSE_ARM = "sparse"
DENSE_ARM = "dense_tied"
ARMS = (SPARSE_ARM, DENSE_ARM)

# decision thresholds (pre-registration)
GATE_DICT_SPECIFIC = 0.003  # G_sparse = M_D - M_S >= 0.003
GATE_ZERO = 0.010  # M_zero - M_S >= 0.010
GATE_SHUFFLE = 0.010  # M_shuffle - M_S >= 0.010
BAND_STRONG_MAX = 0.135
BAND_VIABLE_MAX = 0.145
HEALTH_MIN_ACTIVE = 24
HEALTH_MIN_EFFECTIVE = 8
HEALTH_MAX_ATOM_SHARE = 0.50

VERDICTS = {
    "strong": "E2E_DICTENV_STRONG_DICTIONARY_CORE_SUPPORTED",
    "specific_weak": "E2E_DICTENV_DICTIONARY_SPECIFIC_BUT_ABSOLUTE_WEAK",
    "viable_not_specific": "E2E_DICTENV_VIABLE_BUT_DICTIONARY_NOT_SPECIFIC",
    "absolute_weak": "E2E_DICTENV_ABSOLUTE_WEAK",
    "gain_not_dictionary": "E2E_DICTENV_GAIN_NOT_DICTIONARY_MEDIATED",
    "mechanism_collapsed": "E2E_DICTENV_MECHANISM_COLLAPSED",
}


# ---------------------------------------------------------------------------
# parameter accounting (must equal 66,158 exactly)
# ---------------------------------------------------------------------------


def backend_parameter_count() -> dict[str, int]:
    """Exact FEC-S1 strict-static backend (patch_hidden=48, pair_hidden=16)."""
    pair_projection = ENV_DIM * PAIR_HIDDEN  # Linear(48,16,bias=False)
    relation_encoder = (
        RELATION_WIDTH * 32 + 32 + 2 * 32 + 32 * PAIR_HIDDEN + PAIR_HIDDEN
    )  # _MLPBlock(23,32,16) incl. LayerNorm
    distance_gate = DISTANCE_BUCKETS * PAIR_HIDDEN
    pair_encoder = (
        4 * PAIR_HIDDEN * 64 + 64 + 2 * 64 + 64 * PAIR_HIDDEN + PAIR_HIDDEN
    )  # _MLPBlock(64,64,16)
    global_encoder = (
        GLOBAL_WIDTH * 32 + 32 + 2 * 32 + 32 * 32 + 32
    )  # _MLPBlock(62,32,32)
    topology_encoder = (
        TOPOLOGY_IN * TOPOLOGY_HIDDEN + TOPOLOGY_HIDDEN + TOPOLOGY_HIDDEN * TOPOLOGY_OUT + TOPOLOGY_OUT
    )
    reader = (
        (97 + 165 + 32 + 8) * 13 + 13 + 13 * 13 + 13 + 13 * 1 + 1
    )  # GenericReader(302,(13,13))
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


def local_parameter_count() -> dict[str, int]:
    dictionary = PHI_DIM * K_ATOMS
    w_r = (ANCHOR_DIM + K_ATOMS) * R_ATOM  # 33 x 64
    w_c = ATOM_CATEGORIES * R_ATOM  # 28 x 64
    shell = N_SHELLS * R_ATOM  # 3 x 64
    w_b = BOND_CATEGORIES * R_BOND  # 4 x 16
    p_shell = SHELLPAIR_CLASSES * R_BOND  # 6 x 16
    env_mlp = ENV_MLP_IN * ENV_MLP_HIDDEN + ENV_MLP_HIDDEN + ENV_MLP_HIDDEN * ENV_DIM + ENV_DIM
    return {
        "dictionary": int(dictionary),
        "W_R": int(w_r),
        "W_C": int(w_c),
        "S": int(shell),
        "W_B": int(w_b),
        "P_shell": int(p_shell),
        "env_mlp": int(env_mlp),
        "subtotal": int(dictionary + w_r + w_c + shell + w_b + p_shell + env_mlp),
    }


def total_parameter_count() -> dict[str, int]:
    local = local_parameter_count()
    backend = backend_parameter_count()
    return {
        "local_total": int(local["subtotal"]),
        "backend_total": int(backend["subtotal"]),
        "whole_model": int(local["subtotal"] + backend["subtotal"]),
        "fec_s1_reference": 66170,
        "difference_vs_fec_s1": int(local["subtotal"] + backend["subtotal"] - 66170),
    }


# ---------------------------------------------------------------------------
# root-relative incidence (pure topology; built once per molecule)
# ---------------------------------------------------------------------------


def bfs_distances(graph: Any, root: int, radius: int = PATCH_RADIUS) -> dict[int, int]:
    """BFS distance buckets inside the radius-``radius`` ball of ``root``."""
    distances: dict[int, int] = {int(root): 0}
    queue: deque[int] = deque([int(root)])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(radius):
            continue
        for neighbor in sorted(graph.neighbors(node)):
            neighbor = int(neighbor)
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def env_incidence(
    graph: Any,
    edge_types: Mapping[Any, int],
    radius: int = PATCH_RADIUS,
) -> dict[str, torch.Tensor]:
    """Per-molecule root-relative environment incidence.

    Returns local (unbatched) tensors:

    * ``occ_node``   [n_occ]      node index (0..n-1) of each occurrence (i,v)
    * ``occ_root``   [n_occ]      root index of each occurrence
    * ``occ_shell``  [n_occ]      shell s_iv in {0,1,2}
    * ``bond_root``  [n_bond_occ] root index of each bond occurrence
    * ``bond_shellpair`` [n_bond_occ]  shellpair class in {0..5}
    * ``bond_type``  [n_bond_occ] primitive bond category in {0..3}
    * ``bond_u`` / ``bond_v`` [n_bond_occ] local endpoint node indices of each
      bond occurrence (additive; used by E2E-DictEnv-P1, ignored by v0/T1)

    Node order is assumed to be ``0..n-1`` (the audited graph convention).
    """
    shell_pairs = zpp._shell_pairs_for_radius(int(radius))
    sp_index = {pair: index for index, pair in enumerate(shell_pairs)}
    nodes = list(graph.nodes)
    n = len(nodes)
    occ_node: list[int] = []
    occ_root: list[int] = []
    occ_shell: list[int] = []
    bond_root: list[int] = []
    bond_shellpair: list[int] = []
    bond_type: list[int] = []
    bond_u: list[int] = []
    bond_v: list[int] = []
    for root in nodes:
        root = int(root)
        distances = bfs_distances(graph, root, int(radius))
        patch = sorted(distances)
        for v in patch:
            occ_root.append(root)
            occ_node.append(int(v))
            occ_shell.append(int(distances[int(v)]))
        induced = graph.induced(set(patch))
        for a, b in induced.edges():
            pair = tuple(sorted((int(distances[int(a)]), int(distances[int(b)]))))
            sp = sp_index[pair]
            bond = int(edge_types[graph.edge_key(int(a), int(b))])
            bond_root.append(root)
            bond_shellpair.append(int(sp))
            bond_type.append(bond)
            bond_u.append(int(a))
            bond_v.append(int(b))
    out: dict[str, torch.Tensor] = {}
    out["occ_node"] = torch.as_tensor(occ_node, dtype=torch.long) if occ_node else torch.empty(0, dtype=torch.long)
    out["occ_root"] = torch.as_tensor(occ_root, dtype=torch.long) if occ_root else torch.empty(0, dtype=torch.long)
    out["occ_shell"] = torch.as_tensor(occ_shell, dtype=torch.long) if occ_shell else torch.empty(0, dtype=torch.long)
    out["bond_root"] = torch.as_tensor(bond_root, dtype=torch.long) if bond_root else torch.empty(0, dtype=torch.long)
    out["bond_shellpair"] = torch.as_tensor(bond_shellpair, dtype=torch.long) if bond_shellpair else torch.empty(0, dtype=torch.long)
    out["bond_type"] = torch.as_tensor(bond_type, dtype=torch.long) if bond_type else torch.empty(0, dtype=torch.long)
    out["bond_u"] = torch.as_tensor(bond_u, dtype=torch.long) if bond_u else torch.empty(0, dtype=torch.long)
    out["bond_v"] = torch.as_tensor(bond_v, dtype=torch.long) if bond_v else torch.empty(0, dtype=torch.long)
    out["n"] = torch.tensor(n, dtype=torch.long)
    return out


def shuffled_occ_node_for_molecule(
    occ_node: torch.Tensor,
    occ_root: torch.Tensor,
    occ_shell: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Within each (root, shell) group, permute the occurrence node mapping.

    The returned tensor has the same length as ``occ_node``; position ``t``
    receives the node that position ``t``'s code is taken from.  The q (and
    shell) of position ``t`` stay tied to the original ``occ_node[t]``, so the
    alpha <-> q correspondence is broken inside every root/shell while the
    node set, alpha multiset and q multiset are preserved.
    """
    occ_node = occ_node.numpy()
    occ_root = occ_root.numpy()
    occ_shell = occ_shell.numpy()
    out = occ_node.copy()
    rng = np.random.default_rng(int(seed))
    # occurrences are grouped by root (construction order) and, within a root,
    # shells were appended in BFS discovery order; regroup by (root, shell).
    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(len(occ_node)):
        groups.setdefault((int(occ_root[index]), int(occ_shell[index])), []).append(index)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        permuted = rng.permutation(len(positions))
        original_nodes = occ_node[positions]
        out[positions] = original_nodes[permuted]
    return torch.as_tensor(out, dtype=torch.long)


# ---------------------------------------------------------------------------
# tied coding operators (both arms share D/P initialization)
# ---------------------------------------------------------------------------


def normalized_dictionary(D: torch.Tensor) -> torch.Tensor:
    """Column normalization ``Dbar_k = D_k / max(||D_k||_2, eps)``."""
    return F.normalize(D, dim=0, eps=EPS)


def tied_iht_codes(Dbar: torch.Tensor, X: torch.Tensor, s: int = SPARSITY, steps: int = IHT_STEPS) -> torch.Tensor:
    """Tied unrolled IHT over the column-normalized dictionary (exact top-s).

    ``C^{t+1} = H_s[ C^t + eta (X - C^t Dbar^T) Dbar ]`` with the same
    deterministic power-iteration step size as the repository-validated
    ``tccd_v0.iht_codes``.
    """
    sigma = tccd_v0.power_iter_sigma(Dbar)
    eta = 1.0 / (sigma * sigma + EPS)
    C = torch.zeros(X.shape[0], Dbar.shape[1], device=X.device, dtype=X.dtype)
    for _ in range(int(steps)):
        C = C + eta * ((X - C @ Dbar.t()) @ Dbar)
        C = tccd_v0.hard_threshold_rows(C, int(s))
    return C


# ---------------------------------------------------------------------------
# pooling helpers (exact FEC-S1 ``moments`` semantics)
# ---------------------------------------------------------------------------


def pool_moments(value: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    counts = torch.bincount(batch, minlength=int(n_graphs)).to(value.dtype).unsqueeze(1)
    total = torch.zeros((int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype)
    squared = torch.zeros_like(total)
    total.index_add_(0, batch, value)
    squared.index_add_(0, batch, value * value)
    return torch.cat([total, squared, torch.log1p(counts)], dim=1)


def pool_pair_moments(value: torch.Tensor, pair_batch: torch.Tensor, pair_bucket: torch.Tensor, n_graphs: int) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    for bucket in range(DISTANCE_BUCKETS):
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
        blocks.append(torch.cat([total, squared, torch.log1p(counts)], dim=1))
    return torch.cat(blocks, dim=1)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class E2EDictEnvModel(nn.Module):
    """SparseDictEnv / DenseTiedEnv shared architecture (66,158 params)."""

    def __init__(self, arm: str = SPARSE_ARM) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = str(arm)
        D_ksvd, _D_rand, _pca = zsdb.load_dictionary()
        self.D = nn.Parameter(
            torch.as_tensor(
                np.asarray(D_ksvd, dtype=np.float32).copy(), dtype=torch.float32
            ).clone()
        )
        # atom trilinear factors
        self.W_R = nn.Parameter(torch.empty(ANCHOR_DIM + K_ATOMS, R_ATOM))
        self.W_C = nn.Parameter(torch.empty(ATOM_CATEGORIES, R_ATOM))
        self.S = nn.Parameter(torch.empty(N_SHELLS, R_ATOM))
        self._init_factor_(self.W_R)
        self._init_factor_(self.W_C)
        self._init_factor_(self.S)
        # bond factors
        self.W_B = nn.Parameter(torch.empty(BOND_CATEGORIES, R_BOND))
        self.P = nn.Parameter(torch.empty(SHELLPAIR_CLASSES, R_BOND))
        self._init_factor_(self.W_B)
        self._init_factor_(self.P)
        # environment decoder 86 -> 329 -> 48 (SiLU)
        self.env_mlp = nn.Sequential(
            nn.Linear(ENV_MLP_IN, ENV_MLP_HIDDEN),
            nn.SiLU(),
            nn.Linear(ENV_MLP_HIDDEN, ENV_DIM),
        )
        # static composition backend (exact FEC-S1)
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
        self.reader = GenericReader(97 + DISTANCE_BUCKETS * (2 * PAIR_HIDDEN + 1) + 32 + TOPOLOGY_OUT, READER_HIDDEN)

    @staticmethod
    def _init_factor_(parameter: nn.Parameter) -> None:
        nn.init.kaiming_uniform_(parameter, a=math.sqrt(5.0))

    # -- coding --------------------------------------------------------------

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        """The arm's tied coordinate: ``[N, 32]``."""
        Dbar = normalized_dictionary(self.D)
        if self.arm == SPARSE_ARM:
            return tied_iht_codes(Dbar, phi, s=SPARSITY, steps=IHT_STEPS)
        return phi @ Dbar

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        """``hat_phi = coord @ Dbar^T`` (tied; same D/P both directions)."""
        Dbar = normalized_dictionary(self.D)
        return coord @ Dbar.t()

    # -- local environment ---------------------------------------------------

    def atom_marginal(self, coord: torch.Tensor, data: Any, occ_coord_node: torch.Tensor, n: int) -> torch.Tensor:
        """``m_i^V = sum_{v in P_i} (r_v W_R)*(q_v W_C)*S_{s_iv} / sqrt(64)`` (SUM)."""
        q = F.one_hot(data.dict_atom, num_classes=ATOM_CATEGORIES).to(coord.dtype)
        rv = torch.cat([torch.ones(coord.shape[0], 1, device=coord.device, dtype=coord.dtype), coord], dim=1)
        r_occ = rv[occ_coord_node]
        q_occ = q[data.env_occ_node]
        shell = data.env_occ_shell
        u = (r_occ @ self.W_R) * (q_occ @ self.W_C) * self.S[shell]
        u = u / math.sqrt(float(R_ATOM))
        m = torch.zeros((int(n), R_ATOM), device=u.device, dtype=u.dtype)
        m.index_add_(0, data.env_occ_root, u)
        return m

    def bond_marginal(self, data: Any, n: int) -> torch.Tensor:
        """``m_i^E = sum_e (b_e W_B)*(P_{p_ie}) / sqrt(16)`` (SUM)."""
        b = F.one_hot(data.env_bond_type, num_classes=BOND_CATEGORIES).to(self.W_B.dtype)
        w = (b @ self.W_B) * self.P[data.env_bond_shellpair]
        w = w / math.sqrt(float(R_BOND))
        m = torch.zeros((int(n), R_BOND), device=w.device, dtype=w.dtype)
        m.index_add_(0, data.env_bond_root, w)
        return m

    def environments(self, coord: torch.Tensor, data: Any, *, occ_coord_node: torch.Tensor | None = None) -> torch.Tensor:
        """Form ``E_i in R^48`` for every root; E is frozen after this."""
        n = int(coord.shape[0])
        if occ_coord_node is None:
            occ_coord_node = data.env_occ_node
        m_v = self.atom_marginal(coord, data, occ_coord_node, n)
        m_e = self.bond_marginal(data, n)
        z = torch.cat([m_v, m_e, data.env_scalars.to(m_v.dtype)], dim=1)
        return self.env_mlp(z)

    # -- forward --------------------------------------------------------------

    def forward(
        self,
        data: Any,
        *,
        coord_zero: bool = False,
        occ_coord_node: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Full forward: environment -> static composition -> prediction.

        ``coord_zero`` forces the dictionary coordinate to zero (mechanism
        intervention A).  ``occ_coord_node`` replaces the occurrence node used
        for the *code* gather (mechanism intervention B).
        """
        phi = data.dict_phi
        coord = self.code(phi)
        if coord_zero:
            coord = torch.zeros_like(coord)
        if occ_coord_node is not None:
            occ_coord_node = occ_coord_node.to(phi.device)
        E = self.environments(coord, data, occ_coord_node=occ_coord_node)

        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = pool_moments(E, batch, n_graphs)

        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        projected_left = u[source]
        projected_right = u[target]
        relation = self.relation_encoder(data.pair_relation)
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
        relation_readout = pool_pair_moments(pair_value, pair_batch, data.pair_bucket, n_graphs)

        graph_hidden = self.global_encoder(data.global_context)
        topology = self.topology_encoder(data.topology_features)
        unified = torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)
        prediction = self.reader(unified)
        aux = {
            "E": E,
            "coord": coord,
            "phi": phi,
        }
        if return_aux:
            return prediction, aux
        return prediction

    # -- losses ----------------------------------------------------------------

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        """``L_rec = E_v ||phi - hat_phi||^2 / (||phi||^2 + eps)``."""
        phi_hat = self.reconstruct(phi, coord)
        numerator = ((phi - phi_hat) ** 2).sum(dim=1)
        denominator = (phi ** 2).sum(dim=1) + EPS
        return (numerator / denominator).mean()


def build_model(arm: str, seed: int = 0, reference_state: Mapping[str, torch.Tensor] | None = None) -> E2EDictEnvModel:
    """Build an arm; when ``reference_state`` is given, copy every shared key.

    ``seed`` is used for reproducibility of the shared random initialization;
    the dictionary tensor is always the SDB K-SVD artifact (arm-independent).
    """
    torch.manual_seed(int(seed))
    model = E2EDictEnvModel(arm=arm)
    if reference_state is not None:
        with torch.no_grad():
            state = model.state_dict()
            for key, value in reference_state.items():
                if key in state and state[key].shape == value.shape:
                    state[key].copy_(value)
    return model


# ---------------------------------------------------------------------------
# custom collate: local occurrence indices -> global node indices
# ---------------------------------------------------------------------------

#: node-indexed occurrence attributes that must be shifted by the per-graph
#: node offset when molecules are batched.
_OCC_OFFSET_NAMES = (
    "env_occ_node",
    "env_occ_root",
    "env_bond_root",
    "env_occ_coord_node",
)


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
    """PyG batch with per-graph node offsets applied to the occurrence fields."""
    data_list = list(data_list)
    base = Batch.from_data_list(data_list)
    ptr = base.ptr.tolist()
    occ_counts = [int(d.env_occ_node.shape[0]) for d in data_list]
    bond_counts = [int(d.env_bond_root.shape[0]) for d in data_list]
    for name in _OCC_OFFSET_NAMES:
        if not hasattr(base, name) or getattr(base, name) is None:
            continue
        counts = bond_counts if name == "env_bond_root" else occ_counts
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
    "R_ATOM",
    "R_BOND",
    "ENV_MLP_IN",
    "ENV_MLP_HIDDEN",
    "ENV_DIM",
    "PAIR_HIDDEN",
    "SPARSE_ARM",
    "DENSE_ARM",
    "ARMS",
    "GATE_DICT_SPECIFIC",
    "GATE_ZERO",
    "GATE_SHUFFLE",
    "BAND_STRONG_MAX",
    "BAND_VIABLE_MAX",
    "VERDICTS",
    "backend_parameter_count",
    "local_parameter_count",
    "total_parameter_count",
    "bfs_distances",
    "env_incidence",
    "shuffled_occ_node_for_molecule",
    "normalized_dictionary",
    "tied_iht_codes",
    "pool_moments",
    "pool_pair_moments",
    "E2EDictEnvModel",
    "build_model",
    "env_collate",
    "make_env_loader",
]
