"""E2E-DictEnv-T1 — tuned sparse dictionary-core chemical environment.

Round: ``e2e_dictenv_t1``.  Study ``zinc-context-gap``.
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_t1_preregistration.md``.
Prior-artifact audit: ``tracks/ksvd/notes/e2e_dictenv_t1_prior_artifact_audit.md``.

T1 keeps the E2E-DictEnv-v0 dictionary exactly as confirmed
(``phi65 -> D -> alpha32``, K=32, s=8, 10 tied-IHT steps, exact top-8,
column-normalized, task-coupled, SDB K-SVD init) and changes **only** the
dictionary -> environment interface:

    x_i^coarse (146, FEC-S0 bit-identical train-fit standardized patch_cont)
    m_i^D      (learned sparse role x chemistry environment code)
    z_i = [x_i^coarse ; m_i^D] -> SiLU MLP -> E_i in R^48

Three registered interface candidates:

* ``A1_COARSE146_POOLED64``: pooled ``(alpha_v W_R) o (q_v W_C) o S_{s_iv}``
  code in R^64, decoder ``210 -> 172 -> 48``.
* ``A2_COARSE146_SLOT48``: per-shell slot code in R^48, concatenated R^144,
  decoder ``290 -> 135 -> 48``.
* ``A3_COARSE146_SLOT64``: per-shell slot code in R^64, concatenated R^192,
  decoder ``338 -> 116 -> 48``.

No message passing, no recurrence, no pair->centre, no relation refresh, no
attention, no context writeback, no typed/parent lookup, no FEC-S1
``146 -> 214 -> 24`` adapter, no separate bond marginal branch (the v0 bond
chemistry is carried by the coarse descriptor's ``bond_shell`` and
``incident_bonds`` blocks).  The only learned fine structural coordinate is
``alpha_v = IHT(D, phi_v)``.  ``official_test_loaded = false`` always.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_sdb_v0 as zsdb
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

PROTOCOL_VERSION = "e2e_dictenv_t1"

# frozen dictionary / environment constants (inherited from v0)
PHI_DIM = int(v0.PHI_DIM)  # 65
K_ATOMS = int(v0.K_ATOMS)  # 32
SPARSITY = int(v0.SPARSITY)  # 8
IHT_STEPS = int(v0.IHT_STEPS)  # 10
ATOM_CATEGORIES = int(v0.ATOM_CATEGORIES)  # 28
N_SHELLS = int(v0.N_SHELLS)  # 3
ENV_DIM = int(v0.ENV_DIM)  # 48
PATCH_RADIUS = int(v0.PATCH_RADIUS)

#: exact FEC-S0 factorized coarse descriptor width
COARSE_DIM = 146

# backend constants (exact FEC-S1 strict-static backend, identical to v0)
RELATION_WIDTH = int(v0.RELATION_WIDTH)  # 23
GLOBAL_WIDTH = int(v0.GLOBAL_WIDTH)  # 62
TOPOLOGY_IN = int(v0.TOPOLOGY_IN)  # 25
TOPOLOGY_HIDDEN = int(v0.TOPOLOGY_HIDDEN)  # 16
TOPOLOGY_OUT = int(v0.TOPOLOGY_OUT)  # 8
PAIR_HIDDEN = int(v0.PAIR_HIDDEN)  # 16
DISTANCE_BUCKETS = int(v0.DISTANCE_BUCKETS)  # 5
READER_HIDDEN = v0.READER_HIDDEN
BACKEND_DROPOUT = float(v0.BACKEND_DROPOUT)
EPS = float(v0.EPS)

SPARSE_ARM = v0.SPARSE_ARM
DENSE_ARM = v0.DENSE_ARM
ARMS = v0.ARMS

# frozen v0 reconstruction anchor and the registered Stage-B multiples
LAMBDA_V0 = 135.83492071463948
LAMBDA_SCALES = (1.0, 0.5, 0.25)

# decision thresholds (pre-registration)
GATE_DICT_SPECIFIC = 0.003
GATE_ZERO = 0.010
GATE_SHUFFLE = 0.010
BAND_STRONG_MAX = 0.135
BAND_VIABLE_MAX = 0.145
HEALTH_MIN_ACTIVE = 24
HEALTH_MIN_EFFECTIVE = 8
HEALTH_MAX_ATOM_SHARE = 0.50
HEALTH_MIN_INITIAL_RETENTION = 0.80
HEALTH_MAX_VALID_RECONSTRUCTION = 0.01

TUNING_MATERIAL_DELTA = 0.002

VERDICTS = {
    "strong_specific": "E2E_DICTENV_TUNED_STRONG_AND_DICTIONARY_SPECIFIC",
    "viable_specific": "E2E_DICTENV_TUNED_VIABLE_AND_DICTIONARY_SPECIFIC",
    "perf_gain_specific_unstable": "E2E_DICTENV_TUNED_PERFORMANCE_GAIN_DICTIONARY_SPECIFIC_UNSTABLE",
    "no_material_gain": "E2E_DICTENV_TUNING_NO_MATERIAL_ABSOLUTE_GAIN",
    "mechanism_lost": "E2E_DICTENV_TUNED_MECHANISM_LOST",
}


# ---------------------------------------------------------------------------
# candidates and exact parameter accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One registered Stage-A dictionary -> environment interface."""

    key: str
    candidate_id: str
    r_dict: int
    mode: str  # "pooled" | "slot"
    decoder_hidden: int

    @property
    def decoder_input(self) -> int:
        extra = self.r_dict if self.mode == "pooled" else N_SHELLS * self.r_dict
        return COARSE_DIM + extra


CANDIDATES: dict[str, Candidate] = {
    "a1": Candidate("a1", "A1_COARSE146_POOLED64", 64, "pooled", 172),
    "a2": Candidate("a2", "A2_COARSE146_SLOT48", 48, "slot", 135),
    "a3": Candidate("a3", "A3_COARSE146_SLOT64", 64, "slot", 116),
}

CANDIDATE_ORDER = ("a1", "a2", "a3")


def binding_parameter_count(candidate: Candidate) -> int:
    base = K_ATOMS * candidate.r_dict + ATOM_CATEGORIES * candidate.r_dict
    if candidate.mode == "pooled":
        base += N_SHELLS * candidate.r_dict
    return int(base)


def decoder_parameter_count(candidate: Candidate) -> int:
    d = candidate.decoder_input
    h = candidate.decoder_hidden
    return int(d * h + h + h * ENV_DIM + ENV_DIM)


def local_parameter_count(candidate: Candidate) -> dict[str, int]:
    dictionary = PHI_DIM * K_ATOMS
    binding = binding_parameter_count(candidate)
    decoder = decoder_parameter_count(candidate)
    return {
        "dictionary": int(dictionary),
        "binding": int(binding),
        "decoder": int(decoder),
        "subtotal": int(dictionary + binding + decoder),
    }


def total_parameter_count(candidate: Candidate) -> dict[str, int]:
    local = local_parameter_count(candidate)
    backend = v0.backend_parameter_count()
    return {
        "candidate_id": candidate.candidate_id,
        "mode": candidate.mode,
        "r_dict": int(candidate.r_dict),
        "decoder_hidden": int(candidate.decoder_hidden),
        "decoder_input": int(candidate.decoder_input),
        "local_total": int(local["subtotal"]),
        "backend_total": int(backend["subtotal"]),
        "whole_model": int(local["subtotal"] + backend["subtotal"]),
        "fec_s1_reference": 66170,
        "difference_vs_fec_s1": int(local["subtotal"] + backend["subtotal"] - 66170),
    }


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class T1Model(nn.Module):
    """SparseDictEnv / DenseTiedEnv on a registered interface candidate."""

    def __init__(self, arm: str = SPARSE_ARM, candidate: Candidate = CANDIDATES["a1"]) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = str(arm)
        self.candidate = candidate
        D_ksvd, _D_rand, _pca = zsdb.load_dictionary()
        self.D = nn.Parameter(
            torch.as_tensor(np.asarray(D_ksvd, dtype=np.float32).copy(), dtype=torch.float32).clone()
        )
        r = int(candidate.r_dict)
        self.W_R = nn.Parameter(torch.empty(K_ATOMS, r))
        self.W_C = nn.Parameter(torch.empty(ATOM_CATEGORIES, r))
        self._init_factor_(self.W_R)
        self._init_factor_(self.W_C)
        if candidate.mode == "pooled":
            self.S = nn.Parameter(torch.empty(N_SHELLS, r))
            self._init_factor_(self.S)
        else:
            self.register_parameter("S", None)
        self.env_mlp = nn.Sequential(
            nn.Linear(int(candidate.decoder_input), int(candidate.decoder_hidden)),
            nn.SiLU(),
            nn.Linear(int(candidate.decoder_hidden), ENV_DIM),
        )
        # exact FEC-S1 strict-static backend (identical to v0)
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

    # -- dictionary x chemistry environment code ------------------------------

    def dictionary_environment(self, coord: torch.Tensor, data: Any, occ_coord_node: torch.Tensor | None) -> torch.Tensor:
        r = int(self.candidate.r_dict)
        q = torch.nn.functional.one_hot(data.dict_atom, num_classes=ATOM_CATEGORIES).to(coord.dtype)
        if occ_coord_node is None:
            occ_coord_node = data.env_occ_node
        c = coord[occ_coord_node]
        qc = q[data.env_occ_node]
        wr = c @ self.W_R
        wc = qc @ self.W_C
        n = int(coord.shape[0])
        if self.candidate.mode == "pooled":
            u = wr * wc * self.S[data.env_occ_shell]
            u = u / math.sqrt(float(r))
            m = torch.zeros((n, r), device=u.device, dtype=u.dtype)
            m.index_add_(0, data.env_occ_root, u)
            return m
        parts: list[torch.Tensor] = []
        for shell in range(N_SHELLS):
            mask = data.env_occ_shell == shell
            u = (wr[mask] * wc[mask]) / math.sqrt(float(r))
            ms = torch.zeros((n, r), device=u.device, dtype=u.dtype)
            ms.index_add_(0, data.env_occ_root[mask], u)
            parts.append(ms)
        return torch.cat(parts, dim=1)

    def environments(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        occ_coord_node: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = self.dictionary_environment(coord, data, occ_coord_node)
        coarse = data.patch_cont.to(m.dtype)
        if int(coarse.shape[1]) != COARSE_DIM:
            raise RuntimeError(f"coarse descriptor width {int(coarse.shape[1])} != {COARSE_DIM}")
        z = torch.cat([coarse, m], dim=1)
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
        phi = data.dict_phi
        coord = self.code(phi)
        if coord_zero:
            coord = torch.zeros_like(coord)
        if occ_coord_node is not None:
            occ_coord_node = occ_coord_node.to(phi.device)
        E = self.environments(coord, data, occ_coord_node=occ_coord_node)

        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = v0.pool_moments(E, batch, n_graphs)

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
    candidate: Candidate,
    seed: int = 0,
    reference_state: Mapping[str, torch.Tensor] | None = None,
) -> T1Model:
    torch.manual_seed(int(seed))
    model = T1Model(arm=arm, candidate=candidate)
    if reference_state is not None:
        with torch.no_grad():
            state = model.state_dict()
            for key, value in reference_state.items():
                if key in state and state[key].shape == value.shape:
                    state[key].copy_(value)
    return model


__all__ = [
    "PROTOCOL_VERSION",
    "PHI_DIM",
    "K_ATOMS",
    "SPARSITY",
    "IHT_STEPS",
    "ATOM_CATEGORIES",
    "N_SHELLS",
    "COARSE_DIM",
    "ENV_DIM",
    "SPARSE_ARM",
    "DENSE_ARM",
    "ARMS",
    "LAMBDA_V0",
    "LAMBDA_SCALES",
    "GATE_DICT_SPECIFIC",
    "GATE_ZERO",
    "GATE_SHUFFLE",
    "BAND_STRONG_MAX",
    "BAND_VIABLE_MAX",
    "HEALTH_MIN_ACTIVE",
    "HEALTH_MIN_EFFECTIVE",
    "HEALTH_MAX_ATOM_SHARE",
    "HEALTH_MIN_INITIAL_RETENTION",
    "HEALTH_MAX_VALID_RECONSTRUCTION",
    "TUNING_MATERIAL_DELTA",
    "VERDICTS",
    "Candidate",
    "CANDIDATES",
    "CANDIDATE_ORDER",
    "binding_parameter_count",
    "decoder_parameter_count",
    "local_parameter_count",
    "total_parameter_count",
    "T1Model",
    "build_model",
]
