"""E2E-DictEnv-P2-ABS — configurable clean dictionary-core model for ZINC absolute tuning.

Round ``e2e_dictenv_p2_abs`` (Workstream Z).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_p2_abs_preregistration.md``.

The P1 clean primitive-only environment is frozen.  This module exposes exactly
the pre-registered candidate axes:

* reconstruction balance (``lambda_factor``) and horizon (runner-level);
* dictionary capacity ``K`` / sparsity ``s``;
* the dictionary-mediated decoder ``p1`` / ``h1`` / ``h2``;
* the edge binding width ``d_e``.

With ``decoder='p1', d_e=48, K=32, s=8`` and the SDB-v0 dictionary the model is
bit-identical to ``e2e_dictenv_p1.P1Model`` (verified by a CPU test).

Purity is unchanged: no ``patch_cont`` / ``atom_shell`` / ``bond_shell``, no
pair bond-chemistry relation, no message passing, no recurrence, no attention,
no pair->centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

PROTOCOL_VERSION = "e2e_dictenv_p2_abs"

PHI_DIM = int(p1.PHI_DIM)
ATOM_CATEGORIES = int(p1.ATOM_CATEGORIES)
BOND_CATEGORIES = int(p1.BOND_CATEGORIES)
N_SHELLS = int(p1.N_SHELLS)
SHELLPAIR_CLASSES = int(p1.SHELLPAIR_CLASSES)
ENV_DIM = int(p1.ENV_DIM)
ANCHOR_DIM = int(p1.ANCHOR_DIM)
D_A = int(p1.D_A)
IHT_STEPS = int(p1.IHT_STEPS)

PAIR_HIDDEN = int(p1.PAIR_HIDDEN)
DISTANCE_BUCKETS = int(p1.DISTANCE_BUCKETS)
RELATION_WIDTH = int(p1.RELATION_WIDTH)
GLOBAL_WIDTH = int(p1.GLOBAL_WIDTH)
TOPOLOGY_IN = int(p1.TOPOLOGY_IN)
TOPOLOGY_HIDDEN = int(p1.TOPOLOGY_HIDDEN)
TOPOLOGY_OUT = int(p1.TOPOLOGY_OUT)
READER_HIDDEN = p1.READER_HIDDEN
BACKEND_DROPOUT = float(p1.BACKEND_DROPOUT)

PARAM_BUDGET_MIN = 80000
PARAM_BUDGET_MAX = 130000

DECODERS = ("p1", "h1", "h2")


@dataclass(frozen=True)
class P2Config:
    tag: str
    decoder: str
    d_e: int
    K: int
    s: int
    lambda_factor: float
    horizon: int
    dict_kind: str = "sdb32"  # "sdb32" | "k64s8" | "k64s12"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "decoder": self.decoder,
            "d_e": int(self.d_e),
            "K": int(self.K),
            "s": int(self.s),
            "lambda_factor": float(self.lambda_factor),
            "horizon": int(self.horizon),
            "dict_kind": self.dict_kind,
        }


def env_input_dim(config: P2Config) -> int:
    return ANCHOR_DIM + N_SHELLS * D_A + SHELLPAIR_CLASSES * int(config.d_e)


def decoder_layers(config: P2Config) -> dict[str, Any]:
    """Frozen decoder layout per the pre-registration."""
    if config.decoder == "p1":
        return {"kind": "p1", "in": env_input_dim(config), "hidden": 102, "out": ENV_DIM}
    if config.decoder == "h1":
        return {
            "kind": "h1",
            "node": (D_A, 64, 48),
            "edge": (int(config.d_e), 48, 32),
            "anchor": (ANCHOR_DIM, 32),
            "fusion_in": 32 + N_SHELLS * 48 + SHELLPAIR_CLASSES * 32,
            "fusion_hidden": 128,
            "out": ENV_DIM,
        }
    if config.decoder == "h2":
        return {
            "kind": "h2",
            "node": (D_A, 96, 64),
            "edge": (int(config.d_e), 64, 48),
            "anchor": (ANCHOR_DIM, 48),
            "fusion_in": 48 + N_SHELLS * 64 + SHELLPAIR_CLASSES * 48,
            "fusion_hidden": 96,
            "out": ENV_DIM,
        }
    raise ValueError(f"unknown decoder {config.decoder!r}")


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(int(in_dim), int(hidden)), nn.SiLU(), nn.Linear(int(hidden), int(out_dim)))


class P2Model(nn.Module):
    """Configurable clean dictionary-core environment (Sparse tied-IHT only)."""

    def __init__(self, config: P2Config, dictionary: np.ndarray) -> None:
        super().__init__()
        self.config = config
        self.D = nn.Parameter(torch.as_tensor(np.asarray(dictionary, dtype=np.float32)).clone())
        self.W_A_S = nn.Parameter(torch.empty(int(config.K), D_A))
        self.W_A_C = nn.Parameter(torch.empty(ATOM_CATEGORIES, D_A))
        self.W_E_S = nn.Parameter(torch.empty(3 * int(config.K), int(config.d_e)))
        self.W_E_C = nn.Parameter(torch.empty(BOND_CATEGORIES, int(config.d_e)))
        for parameter in (self.W_A_S, self.W_A_C, self.W_E_S, self.W_E_C):
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5.0))
        layout = decoder_layers(config)
        if layout["kind"] == "p1":
            self.env_mlp = nn.Sequential(
                nn.Linear(layout["in"], layout["hidden"]),
                nn.SiLU(),
                nn.Linear(layout["hidden"], layout["out"]),
            )
            self.node_encoder = None
            self.edge_encoder = None
            self.anchor_encoder = None
            self.fusion = None
        else:
            node_in, node_h, node_out = layout["node"]
            edge_in, edge_h, edge_out = layout["edge"]
            anchor_in, anchor_out = layout["anchor"]
            self.env_mlp = None
            self.node_encoder = _mlp(node_in, node_h, node_out)
            self.edge_encoder = _mlp(edge_in, edge_h, edge_out)
            self.anchor_encoder = _mlp(anchor_in, anchor_out, anchor_out)
            self.fusion = _mlp(layout["fusion_in"], layout["fusion_hidden"], layout["out"])
        self.pair_projection = nn.Linear(ENV_DIM, PAIR_HIDDEN, bias=False)
        self.relation_encoder = zpp._MLPBlock(
            RELATION_WIDTH, max(PAIR_HIDDEN, 32), PAIR_HIDDEN, float(BACKEND_DROPOUT)
        )
        self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, PAIR_HIDDEN)
        self.pair_encoder = zpp._MLPBlock(
            4 * PAIR_HIDDEN, max(2 * PAIR_HIDDEN, 64), PAIR_HIDDEN, float(BACKEND_DROPOUT)
        )
        self.global_encoder = zpp._MLPBlock(GLOBAL_WIDTH, max(ENV_DIM // 2, 32), 32, float(BACKEND_DROPOUT))
        self.topology_encoder = nn.Sequential(
            nn.Linear(TOPOLOGY_IN, TOPOLOGY_HIDDEN),
            nn.ReLU(),
            nn.Linear(TOPOLOGY_HIDDEN, TOPOLOGY_OUT),
        )
        self.reader = GenericReader(
            97 + DISTANCE_BUCKETS * (2 * PAIR_HIDDEN + 1) + 32 + TOPOLOGY_OUT, READER_HIDDEN
        )

    # -- coding --------------------------------------------------------------

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        Dbar = v0.normalized_dictionary(self.D)
        return v0.tied_iht_codes(Dbar, phi, s=int(self.config.s), steps=IHT_STEPS)

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        return coord @ v0.normalized_dictionary(self.D).t()

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        phi_hat = self.reconstruct(phi, coord)
        numerator = ((phi - phi_hat) ** 2).sum(dim=1)
        denominator = (phi ** 2).sum(dim=1) + v0.EPS
        return (numerator / denominator).mean()

    # -- local environment ---------------------------------------------------

    def environments(self, coord: torch.Tensor, data: Any) -> torch.Tensor:
        n = int(coord.shape[0])
        anchor = data.anchor.to(coord.dtype)
        q = torch.nn.functional.one_hot(data.dict_atom, num_classes=ATOM_CATEGORIES).to(coord.dtype)
        c = coord[data.env_occ_node.to(coord.device)]
        qc = q[data.env_occ_node.to(coord.device)]
        u = (c @ self.W_A_S) * (qc @ self.W_A_C) / math.sqrt(float(D_A))
        flat = torch.zeros((n * N_SHELLS, D_A), device=u.device, dtype=u.dtype)
        flat.index_add_(0, data.env_occ_root.to(coord.device) * N_SHELLS + data.env_occ_shell.to(coord.device), u)
        node_slots = flat.view(n, N_SHELLS, D_A)

        d_e = int(self.config.d_e)
        bond_u = data.env_bond_u.to(coord.device)
        bond_v = data.env_bond_v.to(coord.device)
        cu = coord[bond_u]
        cv = coord[bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        b = torch.nn.functional.one_hot(data.env_bond_type, num_classes=BOND_CATEGORIES).to(coord.dtype)
        ue = (g @ self.W_E_S) * (b @ self.W_E_C) / math.sqrt(float(d_e))
        flat_e = torch.zeros((n * SHELLPAIR_CLASSES, d_e), device=ue.device, dtype=ue.dtype)
        flat_e.index_add_(
            0,
            data.env_bond_root.to(coord.device) * SHELLPAIR_CLASSES + data.env_bond_shellpair.to(coord.device),
            ue,
        )
        edge_slots = flat_e.view(n, SHELLPAIR_CLASSES, d_e)

        if self.env_mlp is not None:
            z = torch.cat([anchor, node_slots.reshape(n, -1), edge_slots.reshape(n, -1)], dim=1)
            return self.env_mlp(z)
        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        anchor_out = self.anchor_encoder(anchor)
        fused = torch.cat([anchor_out, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1)
        return self.fusion(fused)

    # -- forward --------------------------------------------------------------

    def forward(self, data: Any, *, return_aux: bool = False, **kwargs: Any):
        coord = self.code(data.dict_phi)
        E = self.environments(coord, data)
        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = v0.pool_moments(E, batch, n_graphs)
        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        left = u[source]
        right = u[target]
        relation = self.relation_encoder(data.pair_relation[:, list(p1.P1_RELATION_INDICES)])
        product = left * right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_input = torch.cat([left + right, torch.abs(left - right), product * gate, relation], dim=1)
        pair_value = self.pair_encoder(pair_input)
        pair_batch = batch[source]
        relation_readout = v0.pool_pair_moments(pair_value, pair_batch, data.pair_bucket, n_graphs)
        graph_hidden = self.global_encoder(data.global_context)
        topology = self.topology_encoder(data.topology_features)
        unified = torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)
        prediction = self.reader(unified).view(-1)
        if return_aux:
            return prediction, {"E": E, "coord": coord, "phi": data.dict_phi}
        return prediction


def build_model(config: P2Config, dictionary: np.ndarray, seed: int = 0) -> P2Model:
    torch.manual_seed(int(seed))
    return P2Model(config, dictionary)


def parameter_breakdown(config: P2Config) -> dict[str, int]:
    dictionary = PHI_DIM * int(config.K)
    node_binding = int(config.K) * D_A + ATOM_CATEGORIES * D_A
    edge_binding = 3 * int(config.K) * int(config.d_e) + BOND_CATEGORIES * int(config.d_e)
    layout = decoder_layers(config)
    if layout["kind"] == "p1":
        decoder = layout["in"] * layout["hidden"] + layout["hidden"] + layout["hidden"] * layout["out"] + layout["out"]
    else:
        node_in, node_h, node_out = layout["node"]
        edge_in, edge_h, edge_out = layout["edge"]
        anchor_in, anchor_out = layout["anchor"]
        decoder = (
            (node_in * node_h + node_h + node_h * node_out + node_out)
            + (edge_in * edge_h + edge_h + edge_h * edge_out + edge_out)
            + (anchor_in * anchor_out + anchor_out + anchor_out * anchor_out + anchor_out)
            + (layout["fusion_in"] * layout["fusion_hidden"] + layout["fusion_hidden"] + layout["fusion_hidden"] * layout["out"] + layout["out"])
        )
    backend = int(p1.backend_parameter_count()["subtotal"])
    local = int(dictionary + node_binding + edge_binding + decoder)
    return {
        "dictionary": int(dictionary),
        "node_binding": int(node_binding),
        "edge_binding": int(edge_binding),
        "decoder": int(decoder),
        "local_subtotal": int(local),
        "backend_subtotal": int(backend),
        "whole_model": int(local + backend),
    }


def total_parameter_count(config: P2Config) -> dict[str, Any]:
    breakdown = parameter_breakdown(config)
    whole = int(breakdown["whole_model"])
    return {
        **breakdown,
        "p1_reference": int(p1.total_parameter_count()["whole_model"]),
        "budget_min": int(PARAM_BUDGET_MIN),
        "budget_max": int(PARAM_BUDGET_MAX),
        "within_budget": bool(PARAM_BUDGET_MIN <= whole <= PARAM_BUDGET_MAX),
    }


#: lambda_base (P1 Stage-B calibration on the first 512 official-train graphs)
LAMBDA_BASE_P1 = 135.83492071463948

#: ZINC Stage-A absolute lambdas (frozen, pre-registered)
LAMBDA_Z1 = 0.25 * LAMBDA_BASE_P1
LAMBDA_Z2 = 0.125 * LAMBDA_BASE_P1
LAMBDA_Z3 = 0.0625 * LAMBDA_BASE_P1

__all__ = [
    "PROTOCOL_VERSION",
    "P2Config",
    "P2Model",
    "DECODERS",
    "PARAM_BUDGET_MIN",
    "PARAM_BUDGET_MAX",
    "LAMBDA_BASE_P1",
    "LAMBDA_Z1",
    "LAMBDA_Z2",
    "LAMBDA_Z3",
    "env_input_dim",
    "decoder_layers",
    "build_model",
    "parameter_breakdown",
    "total_parameter_count",
]
