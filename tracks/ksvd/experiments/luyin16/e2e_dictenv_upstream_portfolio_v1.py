"""E2E-DictEnv-Upstream-Portfolio-v1 — three bounded upstream candidates.

Round ``upstream_portfolio_v1`` (Workstream Z, ZINC).
Pre-registration: ``tracks/ksvd/notes/zinc_upstream_portfolio_v1_preregistration.md``.

One immutable ``ScaleSpec`` family (``e2e_dictenv_scale_v1``) with exactly three
single-variation upstream arms plus one shared control.  No message passing, no
Transformer/attention, no node/edge hidden state, no write-back; the shared task
dictionary, its 16-step ISTA solver settings and the structural dictionary are
untouched.

Arms
----
``control``  the frozen ``LatentScaleSEM108`` Full architecture (no change).
``code``     append the native task-dictionary code moments ``[sum_i c_i,
             sum_i c_i^2]`` (``c_i = rho_i * alpha_i``, ``K = 2d``) to the
             existing graph-reader input.  Reader ``R -> 39 -> 39 -> 1``
             becomes ``R + 2K -> 39 -> 39 -> 1`` with the old input columns and
             both later layers reused and the new columns zero-initialised.
``drop``     training-only shared dictionary-atom dropout on the solved code,
             ``E = (c * mask / (1 - p)) @ V_L`` with ``p = 0.1``; no new
             parameters; eval is exactly the parent path.
``r3``       append 36 exact shell-3 local statistics (28 atom counts in shell
             3, 4 bond counts between shells 2 and 3, 4 bond counts within
             shell 3) to the end of the 446-D fusion input, keeping every old
             column; fusion ``446 -> 342 -> 144`` becomes ``482 -> 342 -> 144``.

Nothing here reads or instantiates the official ZINC test split.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem

PROTOCOL_VERSION = "e2e_dictenv_upstream_portfolio_v1"

#: the three candidates and the one shared control; order is fixed by the prompt.
ARMS: tuple[str, ...] = ("control", "code", "drop", "r3")
CANDIDATE_ARMS: tuple[str, ...] = ("code", "drop", "r3")

#: ``drop`` arm fixed probability (pre-registered; not swept).
DROP_P = 0.10
#: private drop RNG seed, independent of the shared batch/RNG stream.
DROP_SEED = 20261002

#: the 36-D shell-3 block layout (atoms 28 | (2,3) bonds 4 | (3,3) bonds 4).
R3_ATOM_BLOCK = (0, 28)
R3_BOND23_BLOCK = (28, 32)
R3_BOND33_BLOCK = (32, 36)
R3_DIM = 36
#: shell radius of the appended statistics.
R3_RADIUS = 3
#: unfiltered 28/4 category vocabulary of the frozen Sem108 interface.
R3_ATOM_CATEGORIES = int(p2.ATOM_CATEGORIES)  # 28
R3_BOND_CATEGORIES = int(p2.BOND_CATEGORIES)  # 4


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


# ---------------------------------------------------------------------------
# closed-form parameter accounting (verified against real modules)
# ---------------------------------------------------------------------------


def reader_input_dim(arm: str, spec: sc.ScaleSpec) -> int:
    base = int(spec.reader_input)
    if arm == "code":
        return base + 2 * int(spec.k)
    return base


def fusion_in_dim(arm: str, spec: sc.ScaleSpec) -> int:
    base = int(spec.fusion_in)
    if arm == "r3":
        return base + R3_DIM
    return base


def expected_parameters(arm: str, spec: sc.ScaleSpec) -> int:
    """Closed-form parameter count for one arm at one ``ScaleSpec``."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}")
    total = int(spec.parameters())
    if arm == "code":
        total += int(spec.reader_hidden) * 2 * int(spec.k)
    elif arm == "r3":
        total += int(spec.fusion_hidden) * R3_DIM
    return int(total)


def expected_inventory(arm: str, spec: sc.ScaleSpec) -> dict[str, int]:
    inventory = dict(spec.inventory())
    if arm == "code":
        inventory["reader"] = sc._reader_count(reader_input_dim(arm, spec), spec.reader_hidden)
    elif arm == "r3":
        inventory["fusion"] = sc._mlp_count(fusion_in_dim(arm, spec), spec.fusion_hidden, spec.d)
    return inventory


# ---------------------------------------------------------------------------
# R3 exact shell-3 statistics (reference: upstream_portfolio_v1_reference)
# ---------------------------------------------------------------------------


def undirected_edges(edge_index: np.ndarray, edge_attr: np.ndarray) -> list[tuple[int, int, int]]:
    """Physical undirected edges, de-duplicated; each directed pair counted once."""
    seen: dict[tuple[int, int], int] = {}
    for index in range(int(edge_index.shape[1])):
        u, v = int(edge_index[0, index]), int(edge_index[1, index])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key not in seen:
            seen[key] = int(edge_attr[index])
    return [(u, v, bond) for (u, v), bond in seen.items()]


def shell3_features(n: int, atoms: Sequence[int], edges: Sequence[tuple[int, int, int]]) -> np.ndarray:
    """Exact ``[n, 36]`` shell-3 counts (atoms | (2,3) bonds | (3,3) bonds)."""
    adjacency: list[list[int]] = [[] for _ in range(int(n))]
    for u, v, _bond in edges:
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))
    features = np.zeros((int(n), R3_DIM), dtype=np.int64)
    for root in range(int(n)):
        distance = [-1] * int(n)
        distance[root] = 0
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            if distance[node] >= R3_RADIUS:
                continue
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        for node in range(int(n)):
            if distance[node] == R3_RADIUS:
                features[root, int(atoms[node])] += 1
        for u, v, bond in edges:
            left, right = sorted((distance[int(u)], distance[int(v)]))
            if (left, right) == (2, 3):
                features[root, R3_BOND23_BLOCK[0] + int(bond)] += 1
            elif (left, right) == (3, 3):
                features[root, R3_BOND33_BLOCK[0] + int(bond)] += 1
    return features


def r3_block_scales(features: torch.Tensor) -> tuple[float, float, float]:
    """Train-only per-block RMS scales; an all-zero block uses scale 1."""
    values = features.to(torch.float64)
    scales: list[float] = []
    for start, stop in (R3_ATOM_BLOCK, R3_BOND23_BLOCK, R3_BOND33_BLOCK):
        block = values[:, start:stop]
        rms = float(torch.sqrt((block * block).mean())) if block.numel() else 0.0
        scales.append(1.0 if (not math.isfinite(rms) or rms <= 1e-12) else rms)
    return float(scales[0]), float(scales[1]), float(scales[2])


# ---------------------------------------------------------------------------
# arm models
# ---------------------------------------------------------------------------


class CodeScaleSEM108(sc.LatentScaleSEM108):
    """Full/Small scale model with native code-moment readout appended.

    ``c_i = rho_i * alpha_i`` is produced by the *same single* bridge call used
    for ``E``; the reader input gets ``[sum_i c_i, sum_i c_i^2] / scale`` in one
    extra concatenation.  No new dictionary, no second ISTA call, no raw-feature
    bypass, no write-back.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        spec = self.scale_spec
        old_reader = self.reader
        new_input = int(spec.reader_input) + 2 * int(spec.k)
        new_reader = sc.GenericReader(new_input, (spec.reader_hidden, spec.reader_hidden))
        with torch.no_grad():
            new_reader.net[0].weight[:, : int(spec.reader_input)].copy_(old_reader.net[0].weight)
            new_reader.net[0].weight[:, int(spec.reader_input) :].zero_()
            new_reader.net[0].bias.copy_(old_reader.net[0].bias)
            new_reader.net[2].load_state_dict(old_reader.net[2].state_dict())
            new_reader.net[4].load_state_dict(old_reader.net[4].state_dict())
        self.reader = new_reader
        self._code_c: torch.Tensor | None = None
        self._code_batch: torch.Tensor | None = None
        self.code_scale_1 = 1.0
        self.code_scale_2 = 1.0
        self.reader.register_forward_pre_hook(self._append_code_moments)

    # -- scales ---------------------------------------------------------------

    def set_code_scales(self, scale_1: float, scale_2: float) -> None:
        self.code_scale_1 = float(max(scale_1, 1e-12))
        self.code_scale_2 = float(max(scale_2, 1e-12))

    # -- single shared bridge call -------------------------------------------

    def _environment_from_parts(self, coord: torch.Tensor, data: Any, interface: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h = sem.SEM108Model._environment_from_parts(self, coord, data, interface, **kwargs)
        E, aux = self.local_dictionary_bridge(h, return_aux=True)
        self._code_c = aux["scale"] * aux["alpha"]
        self._code_batch = None if getattr(data, "batch", None) is None else data.batch
        return E

    def _graph_code_moments(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._code_c is None or self._code_batch is None:
            raise RuntimeError("code moments requested before the bridge produced a code")
        c = self._code_c
        batch = self._code_batch.to(c.device)
        n_graphs = int(batch.max().item()) + 1 if int(batch.numel()) else 0
        first = c.new_zeros((n_graphs, c.shape[1]))
        second = c.new_zeros((n_graphs, c.shape[1]))
        first.index_add_(0, batch, c)
        second.index_add_(0, batch, c * c)
        z = torch.cat([first / self.code_scale_1, second / self.code_scale_2], dim=1)
        return z.to(device=device, dtype=dtype)

    def _append_code_moments(self, _module: nn.Module, inputs: Sequence[torch.Tensor]):
        unified = inputs[0]
        z = self._graph_code_moments(unified.device, unified.dtype)
        return (torch.cat([unified, z], dim=1),)


class DropScaleSEM108(sc.LatentScaleSEM108):
    """Training-only shared dictionary-atom dropout on the solved code."""

    def __init__(self, *args: Any, dropout: float = DROP_P, drop_seed: int = DROP_SEED, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout probability must lie in [0, 1)")
        self.atom_dropout = float(dropout)
        self.drop_seed = int(drop_seed)
        self._drop_generator: torch.Generator | None = None

    def _generator_for(self, device: torch.device) -> torch.Generator:
        if self._drop_generator is None or self._drop_generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.drop_seed)
            self._drop_generator = generator
        return self._drop_generator

    def _dropout_decode(self, h: torch.Tensor) -> torch.Tensor:
        bridge = self.local_dictionary_bridge
        scale = torch.sqrt(h.pow(2).mean(dim=1, keepdim=True) + lb.BRIDGE_EPS)
        x = h / scale
        dictionary = bridge.normalized_dictionary()
        step = bridge.solver_step(dictionary)
        alpha = bridge.codes(x, dictionary, step)
        generator = self._generator_for(alpha.device)
        keep = (torch.rand(alpha.shape[1], generator=generator, device=alpha.device, dtype=alpha.dtype) >= self.atom_dropout).to(alpha.dtype)
        return scale * ((alpha * keep / (1.0 - self.atom_dropout)) @ bridge.V_L)

    def _environment_from_parts(self, coord: torch.Tensor, data: Any, interface: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h = sem.SEM108Model._environment_from_parts(self, coord, data, interface, **kwargs)
        if self.training and self.atom_dropout > 0.0:
            return self._dropout_decode(h)
        return self.local_dictionary_bridge(h)


class R3ScaleSEM108(sc.LatentScaleSEM108):
    """Scale model with the 36-D exact shell-3 statistics appended to fusion."""

    def __init__(self, *args: Any, block_scales: tuple[float, float, float] = (1.0, 1.0, 1.0), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        spec = self.scale_spec
        old_fusion = self.fusion
        new_fusion = sem._mlp(int(spec.fusion_in) + R3_DIM, spec.fusion_hidden, spec.d)
        with torch.no_grad():
            new_fusion[0].weight[:, : int(spec.fusion_in)].copy_(old_fusion[0].weight)
            new_fusion[0].weight[:, int(spec.fusion_in) :].zero_()
            new_fusion[0].bias.copy_(old_fusion[0].bias)
            new_fusion[2].load_state_dict(old_fusion[2].state_dict())
        self.fusion = new_fusion
        self.block_scales = tuple(float(value) for value in block_scales)

    def _r3_block(self, data: Any, dtype: torch.dtype) -> torch.Tensor:
        raw = getattr(data, "r3_features", None)
        if raw is None:
            raise RuntimeError("R3 arm requires data.r3_features (attach_r3 was not run)")
        values = raw.to(dtype=dtype)
        if int(values.shape[1]) != R3_DIM:
            raise RuntimeError(f"r3_features width {int(values.shape[1])} != {R3_DIM}")
        first, second, third = self.block_scales
        return torch.cat(
            [
                values[:, R3_ATOM_BLOCK[0] : R3_ATOM_BLOCK[1]] / first,
                values[:, R3_BOND23_BLOCK[0] : R3_BOND23_BLOCK[1]] / second,
                values[:, R3_BOND33_BLOCK[0] : R3_BOND33_BLOCK[1]] / third,
            ],
            dim=1,
        )

    def _environment_from_parts(self, coord: torch.Tensor, data: Any, interface: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Faithful copy of SEM108Model._environment_from_parts with the frozen
        # 36-D shell-3 block appended at the *end* of the fusion input.
        n = int(coord.shape[0])
        q = F.one_hot(data.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        occ_coord_node = kwargs.get("occ_coord_node")
        if occ_coord_node is None:
            occ_coord_node = data.env_occ_node
        occ_coord_node = occ_coord_node.to(coord.device)
        c = coord[occ_coord_node]
        qc = q[data.env_occ_node.to(coord.device)]
        u = (c @ self.W_A_S) * (qc @ self.W_A_C) / math.sqrt(float(p2.D_A))
        if kwargs.get("node_binding_zero", False):
            u = torch.zeros_like(u)
        flat = torch.zeros((n * int(p2.N_SHELLS), int(p2.D_A)), device=u.device, dtype=u.dtype)
        flat.index_add_(
            0,
            data.env_occ_root.to(coord.device) * int(p2.N_SHELLS) + data.env_occ_shell.to(coord.device),
            u,
        )
        node_slots = flat.view(n, int(p2.N_SHELLS), int(p2.D_A))

        bond_u = kwargs.get("bond_u")
        bond_v = kwargs.get("bond_v")
        if bond_u is None:
            bond_u = data.env_bond_u
        if bond_v is None:
            bond_v = data.env_bond_v
        edge_slots = self._edge_env_slots(
            coord,
            data,
            bond_u.to(coord.device),
            bond_v.to(coord.device),
            edge_binding_zero=bool(kwargs.get("edge_binding_zero", False)),
            mask=kwargs.get("mask"),
        )
        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        if self.parent_interface:
            if self.anchor_encoder is None:
                raise RuntimeError("parent_interface requires the retained anchor encoder")
            interface = self.anchor_encoder(data.anchor.to(coord.dtype))
        fused = torch.cat(
            [interface, node_out.reshape(n, -1), edge_out.reshape(n, -1), self._r3_block(data, coord.dtype)],
            dim=1,
        )
        return self.local_dictionary_bridge(self.fusion(fused))


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

_ARM_CLASSES: dict[str, type] = {"code": CodeScaleSEM108, "drop": DropScaleSEM108, "r3": R3ScaleSEM108}


def build_arm_model(
    arm: str,
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    spec: sc.ScaleSpec = sc.FULL,
    *,
    scale_seed: int = 0,
    r3_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
    drop_p: float = DROP_P,
) -> sc.LatentScaleSEM108:
    """Build one arm; shared parameters follow the exact ``build_scale_model`` stream."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known {list(ARMS)}")
    clean_spec = cssd.CSSD_SPEC
    torch.manual_seed(int(seed))
    if arm == "control":
        return sc.LatentScaleSEM108(
            cm.H1_CONFIG,
            dictionary,
            subspace=subspace,
            spec=spec,
            node_binding=clean_spec.node_binding,
            edge_binding=clean_spec.edge_binding,
            coding=clean_spec.coding,
            scale_seed=int(scale_seed),
        )
    cls = _ARM_CLASSES[arm]
    extra: dict[str, Any] = {}
    if arm == "r3":
        extra["block_scales"] = r3_scales
    if arm == "drop":
        extra["dropout"] = float(drop_p)
    return cls(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        spec=spec,
        node_binding=clean_spec.node_binding,
        edge_binding=clean_spec.edge_binding,
        coding=clean_spec.coding,
        scale_seed=int(scale_seed),
        **extra,
    )


def load_parent_soup(model: sc.LatentScaleSEM108, arm: str, soup_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Warm-start ``model`` from the frozen Full soup, widening new columns.

    Old reader / fusion columns, both later layers and every bias are copied
    exactly; genuinely new input columns start at zero.
    """
    target = model.state_dict()
    loaded: list[str] = []
    expanded: list[str] = []
    skipped: list[str] = []
    with torch.no_grad():
        for key, value in soup_state.items():
            value = value.float()
            if key not in target:
                skipped.append(key)
                continue
            if target[key].shape == value.shape:
                target[key].copy_(value)
                loaded.append(key)
                continue
            if key == "reader.net.0.weight" and arm == "code":
                width = value.shape[1]
                target[key][:, :width].copy_(value)
                target[key][:, width:].zero_()
                expanded.append(key)
                continue
            if key == "fusion.0.weight" and arm == "r3":
                width = value.shape[1]
                target[key][:, :width].copy_(value)
                target[key][:, width:].zero_()
                expanded.append(key)
                continue
            skipped.append(key)
    model.load_state_dict(target)
    return {"loaded": len(loaded), "expanded": expanded, "skipped": skipped}


def arm_parameter_audit(model: sc.LatentScaleSEM108, arm: str) -> dict[str, Any]:
    spec = model.scale_spec
    expected = expected_inventory(arm, spec)
    actual = sc._parameter_inventory(model)
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    mismatches = {key: [expected[key], actual[key]] for key in expected if expected[key] != actual[key]}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "official_test_loaded": False,
        "spec": spec.as_dict(),
        "expected_inventory": expected,
        "actual_inventory": actual,
        "inventory_mismatches": mismatches,
        "inventory_exact": bool(not mismatches),
        "expected_parameters": expected_parameters(arm, spec),
        "actual_parameters": total,
        "trainable_parameters": trainable,
        "parameter_exact": bool(total == expected_parameters(arm, spec)),
        "reader_input": int(model.reader.net[0].weight.shape[1]),
        "fusion_input": int(model.fusion[0].weight.shape[1]),
    }


__all__ = [
    "PROTOCOL_VERSION",
    "ARMS",
    "CANDIDATE_ARMS",
    "DROP_P",
    "DROP_SEED",
    "R3_DIM",
    "R3_ATOM_BLOCK",
    "R3_BOND23_BLOCK",
    "R3_BOND33_BLOCK",
    "R3_ATOM_CATEGORIES",
    "R3_BOND_CATEGORIES",
    "reader_input_dim",
    "fusion_in_dim",
    "expected_parameters",
    "expected_inventory",
    "undirected_edges",
    "shell3_features",
    "r3_block_scales",
    "CodeScaleSEM108",
    "DropScaleSEM108",
    "R3ScaleSEM108",
    "build_arm_model",
    "load_parent_soup",
    "arm_parameter_audit",
    "official_test_blocker",
]
