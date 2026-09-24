"""E2E-DictEnv-A1 — Invariant Attributed Dictionary Core (objects + model).

Round ``e2e_dictenv_a1`` (study ``zinc-context-gap``).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_a1_preregistration.md``.
Prior-artifact audit: ``tracks/ksvd/notes/e2e_dictenv_a1_prior_artifact_audit.md``.

The round changes exactly one thing relative to the frozen H1 clean
dictionary core: the dictionary **input**.  Instead of the chemistry-free
``phi65``, the attributed arms feed a permutation-invariant, rooted,
structure x attribute **joint** statistic

    chi_REAL_i = [ phi_i (65) ; vec(J^V_i) (308) ; vec(J^E_i) (60) ]   in R^433
    J^V_i = sum_{v in P_i} b^V_{iv} q_v^T        b^V in R^11, q_v in R^28
    J^E_i = sum_{e in P_i} b^E_{ie} r_e^T        b^E in R^15, r_e in R^4

and the exact assignment-independent analytic null of the same marginals

    chi_INDEP_i = [ phi_i ; vec(P^V_i) ; vec(P^E_i) ]                  in R^433
    P^V_i = (1/n_i) (sum_v b^V_{iv}) (sum_v q_v)^T
    P^E_i = (1/m_i) (sum_e b^E_{ie}) (sum_e r_e)^T   (all zeros when m_i = 0)

Everything else (H1 decoder, backend, anchor, post-code node/edge chemistry
binding, 15-D pure-topology pair relation, K=32, s=8, tied IHT, Top-5 soup,
lambda_rec) is inherited unchanged from ``e2e_dictenv_p2_abs`` / P1.

Forbidden here (see the audit): canonical node-slot patch, node-ID dependent
slots, typed-WL slot ordering as a Euclidean coordinate, ``patch_cont146`` /
``atom_shell`` / ``bond_shell`` descriptors, learned embeddings, message passing,
recurrence, ring/cycle features, target ``y``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb

PROTOCOL_VERSION = "e2e_dictenv_a1"

# ---------------------------------------------------------------------------
# frozen dimensions
# ---------------------------------------------------------------------------

PHI_DIM = int(r2.PHI_DIM)                    # 65
NODE_BASIS_DIM = int(r2.NODE_BASIS_DIM)      # 11
EDGE_BASIS_DIM = int(r2.EDGE_BASIS_DIM)      # 15
ATOM_CATEGORIES = int(r2.ATOM_CATEGORIES)    # 28
BOND_CATEGORIES = int(r2.BOND_CATEGORIES)    # 4
PATCH_RADIUS = int(r2.PATCH_RADIUS)          # 2

JOINT_V_DIM = NODE_BASIS_DIM * ATOM_CATEGORIES   # 308
JOINT_E_DIM = EDGE_BASIS_DIM * BOND_CATEGORIES   # 60
BLOCK_SLICES: dict[str, slice] = {
    "S": slice(0, PHI_DIM),
    "V": slice(PHI_DIM, PHI_DIM + JOINT_V_DIM),
    "E": slice(PHI_DIM + JOINT_V_DIM, PHI_DIM + JOINT_V_DIM + JOINT_E_DIM),
}
A1_DIM = PHI_DIM + JOINT_V_DIM + JOINT_E_DIM     # 433

BLOCKS = ("S", "V", "E")

#: frozen reconstruction normalisation constant (same as P1/P2 ``v0.EPS``)
EPS = float(v0.EPS)
SCALER_FLOOR = float(r2.SCALER_FLOOR)   # 1.0e-9
SCALER_EPS = float(r2.SCALER_EPS)       # 1.0e-12

#: frozen scalar reconstruction weight (P2-ABS Stage-A winner, inherited as is)
LAMBDA_REC = float(p2.LAMBDA_Z1)        # 33.95873017865987
HORIZON = 320

#: frozen H1 dictionary-core configuration for every A1 arm
DICT_K = int(sdb.K_ATOMS)               # 32
DICT_S = int(sdb.SPARSITY)              # 8
DICE_D_E = 48
DECODER = "h1"

#: Stage-2 frozen IHT candidate step counts
IHT_CANDIDATE_STEPS: tuple[int, ...] = (10, 30, 100, 200)
#: Stage-2 frozen qualification threshold (normalised reconstruction error)
IHT_QUALIFY_MAX_REC = 0.002
#: deterministic official-train row subset used by the coder qualification
IHT_DIAG_ROWS = 50000
IHT_DIAG_SEED = 20260924

#: material decision threshold (all gates)
MATERIAL = 0.003
#: Stage-0 continuity gate (REAL code space) and the TCCD pool geometry
CONTINUITY_POOL = 1500
CONTINUITY_NEAR_PAIRS = 800
CONTINUITY_ROUNDS = 3
CONTINUITY_SEED_OFFSET = 9
CONTINUITY_AUC_PASS = 0.70
#: code-pairing-removal mechanism threshold
MECHANISM_THRESHOLD = 0.010

#: parameter budget inherited from P1/P2
PARAM_BUDGET_MIN = int(p2.PARAM_BUDGET_MIN)
PARAM_BUDGET_MAX = int(p2.PARAM_BUDGET_MAX)

ARMS = ("TOPO", "INDEP", "REAL")
#: (arm, coordinate kind) — ``phi`` is bit-identical reuse, ``real``/``indep``
#: are the two 433-D objects.
ARM_COORDINATE = {"TOPO": "phi", "INDEP": "indep", "REAL": "real"}
ARM_INPUT_DIM = {"TOPO": PHI_DIM, "INDEP": A1_DIM, "REAL": A1_DIM}

CODING_IHT = "iht"
CODING_OMP_FROZEN = "omp_frozen"
CODING_DENSE_TIED = "dense_tied"
CODINGS = (CODING_IHT, CODING_OMP_FROZEN, CODING_DENSE_TIED)


# ---------------------------------------------------------------------------
# 1. rooted patch objects
# ---------------------------------------------------------------------------


def one_hot_rows(index: np.ndarray, categories: int) -> np.ndarray:
    """``[k, categories]`` float64 one-hot; out-of-range rows stay zero."""
    idx = np.asarray(index, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], int(categories)), dtype=np.float64)
    valid = (idx >= 0) & (idx < int(categories))
    out[np.arange(idx.shape[0])[valid], idx[valid]] = 1.0
    return out


def joint_from_basis(
    node_basis: np.ndarray, edge_basis: np.ndarray, q: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(J^V, J^E)`` for one rooted patch.

    ``node_basis [n, 11]`` rows follow the patch node order; ``q [n, 28]`` are
    the atom one-hots of the same rows; ``edge_basis [m, 15]`` follows the
    sorted undirected induced edge list and ``r [m, 4]`` the bond one-hots.
    """
    node_basis = np.asarray(node_basis, dtype=np.float64)
    edge_basis = np.asarray(edge_basis, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if node_basis.shape[1] != NODE_BASIS_DIM or q.shape[1] != ATOM_CATEGORIES:
        raise RuntimeError("node basis / atom one-hot width mismatch")
    if edge_basis.shape[1] != EDGE_BASIS_DIM or r.shape[1] != BOND_CATEGORIES:
        raise RuntimeError("edge basis / bond one-hot width mismatch")
    if node_basis.shape[0] != q.shape[0] or edge_basis.shape[0] != r.shape[0]:
        raise RuntimeError("basis / attribute row-count mismatch")
    return node_basis.T @ q, edge_basis.T @ r


def marginal_from_basis(
    node_basis: np.ndarray, edge_basis: np.ndarray, q: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(P^V, P^E)`` — the assignment-independent analytic null.

    ``P^E`` is exactly zero when the patch has no induced undirected edge
    (``m = 0``); the zero-edge case never divides by ``m``.
    """
    node_basis = np.asarray(node_basis, dtype=np.float64)
    edge_basis = np.asarray(edge_basis, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = int(node_basis.shape[0])
    m = int(edge_basis.shape[0])
    node_basis_sum = node_basis.sum(axis=0)
    atom_sum = q.sum(axis=0)
    joint_v = np.outer(node_basis_sum, atom_sum)
    if n > 0:
        joint_v = joint_v / float(n)
    if m > 0:
        joint_e = np.outer(edge_basis.sum(axis=0), r.sum(axis=0)) / float(m)
    else:
        joint_e = np.zeros((EDGE_BASIS_DIM, BOND_CATEGORIES), dtype=np.float64)
    return joint_v, joint_e


def patch_blocks(
    graph: Any,
    center: int,
    atom_types: np.ndarray,
    edge_types: Mapping[Any, int],
    radius: int = PATCH_RADIUS,
) -> dict[str, Any]:
    """All A1 per-root blocks for one patch, from the audited basis builder.

    Returns exactly the quantities the cache stores, so that a rebuild is
    bit-reproducible and the G0.2 permutation tests can recompute blocks from
    permuted attribute rows.
    """
    nodes, index, node_basis, edge_basis, edges = v2._explicit_basis_for_patch(
        graph, int(center), int(radius)
    )
    atom_types = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    atom_rows = atom_types[np.asarray(nodes, dtype=np.int64)]
    q = one_hot_rows(atom_rows, ATOM_CATEGORIES)
    if edges:
        bond_rows = np.asarray(
            [int(edge_types[graph.edge_key(int(a), int(b))]) for a, b in edges],
            dtype=np.int64,
        )
    else:
        bond_rows = np.zeros(0, dtype=np.int64)
    r = one_hot_rows(bond_rows, BOND_CATEGORIES)
    joint_v, joint_e = joint_from_basis(node_basis, edge_basis, q, r)
    marginal_v, marginal_e = marginal_from_basis(node_basis, edge_basis, q, r)
    return {
        "nodes": nodes,
        "index": index,
        "node_basis": node_basis,
        "edge_basis": edge_basis,
        "edges": edges,
        "q": q,
        "r": r,
        "joint_v": joint_v,
        "joint_e": joint_e,
        "marginal_v": marginal_v,
        "marginal_e": marginal_e,
        "n_patch": int(node_basis.shape[0]),
        "m_patch": int(edge_basis.shape[0]),
    }


# ---------------------------------------------------------------------------
# 2. raw cache record -> stacked blocks
# ---------------------------------------------------------------------------

RAW_FIELDS = (
    "joint_v",   # [N, 308] float32
    "joint_e",   # [N, 60]  float32
    "marginal_v",  # [N, 308] float32
    "marginal_e",  # [N, 60]  float32
)


def concatenate_blocks(raw: Mapping[str, np.ndarray], coordinate: str) -> np.ndarray:
    """``[N, 433]`` raw (unscaled) object for ``coordinate`` in {real, indep}."""
    coordinate = str(coordinate)
    if coordinate == "real":
        node_key, edge_key = "joint_v", "joint_e"
    elif coordinate == "indep":
        node_key, edge_key = "marginal_v", "marginal_e"
    else:
        raise ValueError(f"unknown coordinate {coordinate!r}")
    parts = [
        np.asarray(raw["phi"], dtype=np.float64),
        np.asarray(raw[node_key], dtype=np.float64),
        np.asarray(raw[edge_key], dtype=np.float64),
    ]
    out = np.concatenate(parts, axis=1)
    if out.shape[1] != A1_DIM:
        raise RuntimeError(f"coordinate width {out.shape[1]} != {A1_DIM}")
    return out


# ---------------------------------------------------------------------------
# 3. train-only scaling (identical procedure for REAL and INDEP)
# ---------------------------------------------------------------------------


@dataclass
class BlockScaler:
    """Per-coordinate RMS scaler + mask + block-energy weight (train only)."""

    name: str
    scale: np.ndarray
    mask: np.ndarray
    weight: float
    rms_raw: np.ndarray
    block_energy: float
    masked_coordinates: int
    dim: int

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": int(self.dim),
            "weight": float(self.weight),
            "block_energy": float(self.block_energy),
            "masked_coordinates": int(self.masked_coordinates),
            "scale": [float(x) for x in self.scale.tolist()],
            "mask": [int(x) for x in self.mask.tolist()],
            "rms_min": float(np.min(self.rms_raw)),
            "rms_max": float(np.max(self.rms_raw)),
        }


@dataclass
class ObjectScaler:
    """Three block scalers for one dictionary object (REAL or INDEP)."""

    coordinate: str
    blocks: dict[str, BlockScaler] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate,
            "blocks": {name: block.to_json() for name, block in self.blocks.items()},
        }


def fit_block_scaler(name: str, block: np.ndarray) -> BlockScaler:
    """Layer 1 (train-RMS + zero-RMS mask) and layer 2 (block energy)."""
    block = np.asarray(block, dtype=np.float64)
    scale, mask = sdb.fit_rms_scaler(block, floor=SCALER_FLOOR)
    scaled = (block / scale[None, :]) * mask[None, :]
    energy = float(np.mean(np.sum(scaled * scaled, axis=1)))
    weight = float(1.0 / math.sqrt(energy + SCALER_EPS))
    rms_raw = np.sqrt(np.mean(block * block, axis=0))
    return BlockScaler(
        name=str(name),
        scale=np.asarray(scale, dtype=np.float64),
        mask=np.asarray(mask, dtype=np.float64),
        weight=weight,
        rms_raw=rms_raw,
        block_energy=energy,
        masked_coordinates=int(np.sum(mask == 0.0)),
        dim=int(block.shape[1]),
    )


def fit_object_scaler(coordinate: str, raw: Mapping[str, np.ndarray]) -> ObjectScaler:
    """Fit all three block scalers of one object on official train only."""
    x = concatenate_blocks(raw, coordinate)
    blocks = {name: fit_block_scaler(name, x[:, BLOCK_SLICES[name]]) for name in BLOCKS}
    return ObjectScaler(coordinate=str(coordinate), blocks=blocks)


def apply_block_scaler(
    scaler: ObjectScaler,
    phi: np.ndarray,
    node_block: np.ndarray,
    edge_block: np.ndarray,
) -> np.ndarray:
    """Build a scaled 433-D object from explicit block contents.

    Used for the objects themselves and for the mechanism intervention, where
    only the *content* of the node/edge blocks is substituted while the REAL
    scaler (scale / mask / block weight) stays untouched.
    """
    blocks = {
        "S": np.asarray(phi, dtype=np.float64),
        "V": np.asarray(node_block, dtype=np.float64),
        "E": np.asarray(edge_block, dtype=np.float64),
    }
    parts = []
    for name in BLOCKS:
        block = blocks[name]
        s = scaler.blocks[name]
        if block.shape[1] != int(s.dim):
            raise RuntimeError(f"block {name} width {block.shape[1]} != {s.dim}")
        parts.append(float(s.weight) * (block / s.scale[None, :]) * s.mask[None, :])
    out = np.concatenate(parts, axis=1)
    if out.shape[1] != A1_DIM:
        raise RuntimeError(f"scaled width {out.shape[1]} != {A1_DIM}")
    return out


def apply_object_scaler(scaler: ObjectScaler, raw: Mapping[str, np.ndarray]) -> np.ndarray:
    """Scaled ``[N, 433]`` object; identical formula for REAL and INDEP."""
    coordinate = str(scaler.coordinate)
    if coordinate == "real":
        node_key, edge_key = "joint_v", "joint_e"
    elif coordinate == "indep":
        node_key, edge_key = "marginal_v", "marginal_e"
    else:
        raise ValueError(f"unknown coordinate {coordinate!r}")
    return apply_block_scaler(
        scaler,
        np.asarray(raw["phi"], dtype=np.float64),
        np.asarray(raw[node_key], dtype=np.float64),
        np.asarray(raw[edge_key], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# 4. the model: P2/H1 core with a swappable dictionary input + coder
# ---------------------------------------------------------------------------


class _EnvOverride:
    """Read-only proxy that substitutes a few batch tensors for intervention.

    Used only by :meth:`A1Model.environments`: the shuffled role tensors replace
    ``env_occ_node`` / ``env_bond_u`` / ``env_bond_v`` (the *dictionary-role*
    indices) while ``dict_atom`` / ``env_bond_type`` (chemistry) stay untouched,
    which is exactly the inherited P1 shuffle semantics.
    """

    __slots__ = ("_data", "_overrides")

    def __init__(self, data: Any, overrides: Mapping[str, torch.Tensor]) -> None:
        self._data = data
        self._overrides = dict(overrides)

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._data, name)


class A1Model(p2.P2Model):
    """H1 clean dictionary core whose dictionary input and coder are explicit.

    ``code`` is the only place the arm enters; ``environments``/``forward``
    reproduce the frozen P2-ABS H1 body verbatim and add the inherited P1
    inference-only intervention hooks (``coord_zero`` / ``occ_coord_node`` /
    ``bond_u`` / ``bond_v``) so that the shared P1 ``evaluate`` instrumentation
    is *effective* here instead of being silently swallowed.
    """

    def __init__(
        self,
        config: p2.P2Config,
        dictionary: np.ndarray,
        *,
        input_dim: int,
        coding_mode: str = CODING_IHT,
        iht_steps: int | None = None,
        freeze_dictionary: bool = False,
    ) -> None:
        super().__init__(config, dictionary)
        if str(coding_mode) not in CODINGS:
            raise ValueError(f"unknown coding mode {coding_mode!r}")
        if int(dictionary.shape[0]) != int(input_dim):
            raise RuntimeError(
                f"dictionary feature width {int(dictionary.shape[0])} != input_dim {int(input_dim)}"
            )
        self.input_dim = int(input_dim)
        self.coding_mode = str(coding_mode)
        self.iht_steps = int(p2.IHT_STEPS if iht_steps is None else iht_steps)
        if freeze_dictionary:
            self.D.requires_grad_(False)

    def code(self, phi: torch.Tensor, precomputed: torch.Tensor | None = None) -> torch.Tensor:
        if precomputed is not None:
            return precomputed
        Dbar = v0.normalized_dictionary(self.D)
        if self.coding_mode == CODING_DENSE_TIED:
            return phi @ Dbar
        if self.coding_mode != CODING_IHT:
            raise RuntimeError(
                "A1Model with coding_mode='omp_frozen' requires a precomputed code tensor"
            )
        return v0.tied_iht_codes(Dbar, phi, s=int(self.config.s), steps=int(self.iht_steps))

    def environments(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        occ_coord_node: torch.Tensor | None = None,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        overrides: dict[str, torch.Tensor] = {}
        if occ_coord_node is not None:
            overrides["env_occ_node"] = occ_coord_node
        if bond_u is not None:
            overrides["env_bond_u"] = bond_u
        if bond_v is not None:
            overrides["env_bond_v"] = bond_v
        if overrides:
            data = _EnvOverride(data, overrides)
        return super().environments(coord, data)

    def forward(
        self,
        data: Any,
        *,
        coord_zero: bool = False,
        occ_coord_node: torch.Tensor | None = None,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
        return_aux: bool = False,
        **kwargs: Any,
    ):
        if kwargs:
            raise TypeError(f"A1Model.forward got unexpected keyword arguments {sorted(kwargs)}")
        coord = self.code(data.dict_phi, getattr(data, "precomputed_coord", None))
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


def build_model(
    arm: str,
    dictionary: np.ndarray,
    *,
    coding_mode: str = CODING_IHT,
    iht_steps: int | None = None,
    freeze_dictionary: bool = False,
    d_e: int = DICE_D_E,
    decoder: str = DECODER,
    seed: int = 0,
    reference_state: Mapping[str, torch.Tensor] | None = None,
) -> A1Model:
    arm = str(arm)
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    config = p2.P2Config(
        tag=f"A1_{arm}",
        decoder=str(decoder),
        d_e=int(d_e),
        K=DICT_K,
        s=DICT_S,
        lambda_factor=0.25,
        horizon=HORIZON,
        dict_kind="a1",
    )
    torch.manual_seed(int(seed))
    model = A1Model(
        config,
        dictionary,
        input_dim=ARM_INPUT_DIM[arm],
        coding_mode=coding_mode,
        iht_steps=iht_steps,
        freeze_dictionary=freeze_dictionary,
    )
    if reference_state is not None:
        with torch.no_grad():
            state = model.state_dict()
            for key, value in reference_state.items():
                if key in state and state[key].shape == value.shape:
                    state[key].copy_(value)
    return model


def parameter_breakdown(arm: str, *, d_e: int = DICE_D_E, decoder: str = DECODER) -> dict[str, int]:
    """Exact parameter accounting (must equal the built model)."""
    arm = str(arm)
    config = p2.P2Config(
        tag=f"A1_{arm}",
        decoder=str(decoder),
        d_e=int(d_e),
        K=DICT_K,
        s=DICT_S,
        lambda_factor=0.25,
        horizon=HORIZON,
        dict_kind="a1",
    )
    breakdown = p2.parameter_breakdown(config)
    input_dim = ARM_INPUT_DIM[arm]
    breakdown = dict(breakdown)
    breakdown["dictionary"] = int(input_dim * DICT_K)
    breakdown["input_dim"] = int(input_dim)
    local = int(
        breakdown["dictionary"]
        + breakdown["node_binding"]
        + breakdown["edge_binding"]
        + breakdown["decoder"]
    )
    breakdown["local_subtotal"] = local
    breakdown["whole_model"] = int(local + breakdown["backend_subtotal"])
    return breakdown


def total_parameter_count(arm: str, **kwargs: Any) -> dict[str, Any]:
    breakdown = parameter_breakdown(arm, **kwargs)
    whole = int(breakdown["whole_model"])
    return {
        **breakdown,
        "arm": str(arm),
        "budget_min": int(PARAM_BUDGET_MIN),
        "budget_max": int(PARAM_BUDGET_MAX),
        "within_budget": bool(PARAM_BUDGET_MIN <= whole <= PARAM_BUDGET_MAX),
    }


def reconstruction_objective(model: A1Model, x: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    """Arm-independent normalised reconstruction objective (same formula all arms)."""
    return model.reconstruction_loss(x, coord)


__all__ = [
    "PROTOCOL_VERSION",
    "PHI_DIM",
    "NODE_BASIS_DIM",
    "EDGE_BASIS_DIM",
    "ATOM_CATEGORIES",
    "BOND_CATEGORIES",
    "JOINT_V_DIM",
    "JOINT_E_DIM",
    "BLOCK_SLICES",
    "A1_DIM",
    "BLOCKS",
    "ARM_COORDINATE",
    "ARM_INPUT_DIM",
    "ARMS",
    "CODING_IHT",
    "CODING_OMP_FROZEN",
    "CODING_DENSE_TIED",
    "CODINGS",
    "LAMBDA_REC",
    "HORIZON",
    "DICT_K",
    "DICT_S",
    "IHT_CANDIDATE_STEPS",
    "IHT_QUALIFY_MAX_REC",
    "IHT_DIAG_ROWS",
    "IHT_DIAG_SEED",
    "MATERIAL",
    "MECHANISM_THRESHOLD",
    "CONTINUITY_POOL",
    "CONTINUITY_NEAR_PAIRS",
    "CONTINUITY_ROUNDS",
    "CONTINUITY_SEED_OFFSET",
    "CONTINUITY_AUC_PASS",
    "one_hot_rows",
    "joint_from_basis",
    "marginal_from_basis",
    "patch_blocks",
    "concatenate_blocks",
    "BlockScaler",
    "ObjectScaler",
    "fit_block_scaler",
    "fit_object_scaler",
    "apply_block_scaler",
    "apply_object_scaler",
    "A1Model",
    "build_model",
    "parameter_breakdown",
    "total_parameter_count",
    "reconstruction_objective",
]
