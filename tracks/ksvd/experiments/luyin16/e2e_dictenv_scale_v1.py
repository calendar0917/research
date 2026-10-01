"""E2E-DictEnv-Scale-v1 — unified Small/Full task-dictionary scaling core.

Round ``e2e_dictenv_scale_v1`` (Workstream Z, ZINC).
Pre-registration:
``tracks/ksvd/notes/zinc_e2e_dictenv_scale_v1_preregistration.md``.
Delivered handoff package (vendored):
``tracks/ksvd/experiments/luyin16/latent_bridge_reference/v1_20261001/``
and ``.../dictionary_scaling/v1_20261001/`` (see the pre-registration).

This module implements **one** model family driven by a single immutable
``ScaleSpec`` (multiplier ``m``):

* ``m = 1`` — the frozen ``LatentBridgeSEM108`` (Sem108 body 97,709 + D_L/V_L
  9,216 = 106,925 parameters).  Built through the untouched parent
  constructor, so every name, initial value and RNG draw is identical.
* ``m = 3`` — Full: widen the task path only
  (fusion ``446 -> 342 -> 144``; task dictionary ``D_L [144, 288]`` /
  ``V_L [288, 144]``; relation endpoints ``144 -> 48``; relation encoder
  ``15 -> 96 -> 48``; static pair encoder ``192 -> 192 -> 48``; graph reader
  ``814 -> 39 -> 39 -> 1``) = 408,651 parameters.
* ``m = 2`` — **audit only** (233,203 parameters); never a training candidate.

Everything outside the task path is byte-identical to the frozen Sem108 /
latent-bridge construction: the structural dictionary ``D`` (K32/s8/IHT10),
the common coordinate ``U`` / ``common_rms``, the node binding (96) and edge
binding (48), the node slot encoder ``96 -> 64 -> 48``, the edge slot encoder
``48 -> 48 -> 32``, the 3 node shells / 6 edge shell-pairs, the semantic
interface ``[Sem108 ; size2]`` (110), the global encoder ``62 -> 32 -> 32``,
the topology encoder ``25 -> 16 -> 8`` and the C6 mask.

The task dictionary is the existing 16-step unrolled soft-threshold ISTA
(``rho`` row-RMS normalisation, column-normalised ``D_L``, detached
conservative step, ``lambda1 = 0.05``, ``lambda2 = 0.01``, ``alpha^0 = 0``,
``E = rho * alpha @ V_L``); both ``D_L`` and ``V_L`` receive the task
gradient.  No graph index, no message passing, no node/edge hidden state, no
pair -> node write-back; the 16 iterations run inside one object only.

CPU only.  ``official_test_loaded = False`` always.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

PROTOCOL_VERSION = "e2e_dictenv_scale_v1"

#: frozen task-path base widths (the ``m = 1`` preset).
BASE_DIM = int(sem.SEM_FUSION_OUT)  # 48
BASE_PAIR = int(p2.PAIR_HIDDEN)  # 16
BASE_FUSION_HIDDEN = int(sem.SEM_FUSION_HIDDEN)  # 114
BASE_RELATION_HIDDEN = int(max(p2.PAIR_HIDDEN, 32))  # 32
BASE_PAIR_HIDDEN = int(max(2 * p2.PAIR_HIDDEN, 64))  # 64
BASE_READER_HIDDEN = int(p2.READER_HIDDEN[0])  # 13
#: frozen edge-binding width (H1 config ``d_e``), never multiplied.
BASE_D_E = int(cm.H1_CONFIG.d_e)  # 48

#: task-path budget for the scaled candidate (Full <= 420k, pre-registered).
PARAM_BUDGET_MAX = 420_000

#: fixed-module parameter prefixes / names that must never change with ``m``.
FIXED_PARAMETER_NAMES = (
    "D",
    "W_A_S",
    "W_A_C",
    "W_E_S",
    "W_E_C",
    "node_encoder.",
    "edge_encoder.",
    "global_encoder.",
    "topology_encoder.",
)
#: fixed buffers (common coordinate + residual RMS) that must never change.
FIXED_BUFFER_NAMES = ("U", "common_rms")


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


# ---------------------------------------------------------------------------
# exact parameter accounting (closed form; verified against real modules)
# ---------------------------------------------------------------------------


def _linear_count(in_dim: int, out_dim: int, *, bias: bool = True) -> int:
    return int(in_dim) * int(out_dim) + (int(out_dim) if bias else 0)


def _mlp_count(in_dim: int, hidden: int, out_dim: int) -> int:
    return _linear_count(in_dim, hidden) + _linear_count(hidden, out_dim)


def _mlpblock_count(in_dim: int, hidden: int, out_dim: int) -> int:
    # Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> ReLU
    return _mlp_count(in_dim, hidden, out_dim) + 2 * int(hidden)


def _reader_count(in_dim: int, hidden: int) -> int:
    return (
        _linear_count(in_dim, hidden)
        + _linear_count(hidden, hidden)
        + _linear_count(hidden, 1)
    )


@dataclass(frozen=True)
class ScaleSpec:
    """Immutable width multiplier for the unified Small/Full model family."""

    name: str
    multiplier: int

    def __post_init__(self) -> None:
        if int(self.multiplier) not in (1, 2, 3):
            raise ValueError(
                f"ScaleSpec multiplier must be 1, 2 or 3 (2 is audit-only), got {self.multiplier!r}"
            )

    @property
    def m(self) -> int:
        return int(self.multiplier)

    @property
    def d(self) -> int:
        """Fusion output / task-dictionary decoded environment width."""
        return BASE_DIM * self.m

    @property
    def k(self) -> int:
        """Task-dictionary atom count (two orthonormal bases)."""
        return 2 * self.d

    @property
    def p(self) -> int:
        """Static-relation endpoint / pair output width."""
        return BASE_PAIR * self.m

    @property
    def fusion_in(self) -> int:
        return int(sem.SEM_FUSION_IN)  # 446

    @property
    def fusion_hidden(self) -> int:
        return BASE_FUSION_HIDDEN * self.m

    @property
    def relation_hidden(self) -> int:
        return BASE_RELATION_HIDDEN * self.m

    @property
    def pair_hidden(self) -> int:
        return BASE_PAIR_HIDDEN * self.m

    @property
    def reader_hidden(self) -> int:
        return BASE_READER_HIDDEN * self.m

    @property
    def reader_input(self) -> int:
        """``(2d + 1) + 5 * (2p + 1) + 32 + 8``."""
        return (
            2 * self.d
            + 1
            + int(p2.DISTANCE_BUCKETS) * (2 * self.p + 1)
            + 32
            + int(p2.TOPOLOGY_OUT)
        )

    @property
    def unit_dictionary_parameters(self) -> int:
        """``D_L [d, 2d]`` + ``V_L [2d, d]``."""
        return 2 * self.d * self.k

    def inventory(self) -> dict[str, int]:
        """Expected parameter count per named component (pre-registered)."""
        return {
            "structural_dictionary": int(cssd.PHI_DIM) * int(cssd.K_ATOMS),
            "node_binding": (1 + int(cssd.K_ATOMS) + int(p2.ATOM_CATEGORIES)) * int(p2.D_A),
            "edge_binding": (
                3 * (1 + int(cssd.K_ATOMS)) + int(p2.BOND_CATEGORIES)
            )
            * BASE_D_E,
            "node_slot_encoder": _mlp_count(int(p2.D_A), 64, int(p2.ENV_DIM)),
            "edge_slot_encoder": _mlp_count(BASE_D_E, 48, 32),
            "fusion": _mlp_count(self.fusion_in, self.fusion_hidden, self.d),
            "task_dictionary_D_V": self.unit_dictionary_parameters,
            "pair_projection": self.d * self.p,
            "relation_encoder": _mlpblock_count(
                int(p2.RELATION_WIDTH), self.relation_hidden, self.p
            ),
            "distance_gate": int(p2.DISTANCE_BUCKETS) * self.p,
            "pair_encoder": _mlpblock_count(
                4 * self.p, self.pair_hidden, self.p
            ),
            "global_encoder": _mlpblock_count(int(p2.GLOBAL_WIDTH), 32, 32),
            "topology_encoder": _mlp_count(
                int(p2.TOPOLOGY_IN), int(p2.TOPOLOGY_HIDDEN), int(p2.TOPOLOGY_OUT)
            ),
            "reader": _reader_count(self.reader_input, self.reader_hidden),
        }

    def parameters(self) -> int:
        return int(sum(self.inventory().values()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "multiplier": self.m,
            "d": self.d,
            "k": self.k,
            "p": self.p,
            "fusion": [self.fusion_in, self.fusion_hidden, self.d],
            "task_dictionary": [self.d, self.k],
            "value_table": [self.k, self.d],
            "pair_projection": [self.d, self.p],
            "relation_encoder": [int(p2.RELATION_WIDTH), self.relation_hidden, self.p],
            "distance_gate": [int(p2.DISTANCE_BUCKETS), self.p],
            "pair_encoder": [4 * self.p, self.pair_hidden, self.p],
            "reader": [self.reader_input, self.reader_hidden, self.reader_hidden],
            "reader_input": self.reader_input,
            "parameters": self.parameters(),
        }


#: public presets — only Small (m=1) and Full (m=3) are training candidates.
SMALL = ScaleSpec("small", 1)
FULL = ScaleSpec("full", 3)
#: m = 2 exists for zero-training dimension / parameter audits only.
AUDIT_ONLY_M2 = ScaleSpec("audit-m2", 2)
PRESETS: dict[str, ScaleSpec] = {"small": SMALL, "full": FULL}
TRAIN_PRESETS: dict[str, ScaleSpec] = {"small": SMALL, "full": FULL}


def scale_spec(name_or_multiplier: str | int) -> ScaleSpec:
    if isinstance(name_or_multiplier, str):
        key = name_or_multiplier.strip().lower()
        if key not in PRESETS:
            raise KeyError(f"unknown scale preset {name_or_multiplier!r}; known: {sorted(PRESETS)}")
        return PRESETS[key]
    return ScaleSpec(f"m{int(name_or_multiplier)}", int(name_or_multiplier))


# ---------------------------------------------------------------------------
# readout block layouts derived from d / p (never the frozen 48/16 constants)
# ---------------------------------------------------------------------------


def unary_block_layout(spec: ScaleSpec) -> dict[str, tuple[int, int]]:
    d = spec.d
    return {"first": (0, d), "second": (d, 2 * d), "count": (2 * d, 2 * d + 1)}


def pair_block_layout(spec: ScaleSpec) -> dict[str, tuple[int, int]]:
    p = spec.p
    return {"first": (0, p), "second": (p, 2 * p), "count": (2 * p, 2 * p + 1)}


def pair_block_dim(spec: ScaleSpec) -> int:
    return 2 * spec.p + 1


def relation_readout_dim(spec: ScaleSpec) -> int:
    return int(p2.DISTANCE_BUCKETS) * pair_block_dim(spec)


def readout_layout(spec: ScaleSpec) -> dict[str, Any]:
    """Full reader-input layout with 48/16 replaced by the spec-derived d/p."""
    return {
        "unary": unary_block_layout(spec),
        "pair_per_bucket": pair_block_layout(spec),
        "pair_block_dim": pair_block_dim(spec),
        "relation_readout_dim": relation_readout_dim(spec),
        "global": (2 * spec.d + 1 + relation_readout_dim(spec), 2 * spec.d + 1 + relation_readout_dim(spec) + 32),
        "topology": (
            2 * spec.d + 1 + relation_readout_dim(spec) + 32,
            spec.reader_input,
        ),
        "reader_input": spec.reader_input,
    }


# ---------------------------------------------------------------------------
# fresh width-specific initialisation stream (never the parent torch stream)
# ---------------------------------------------------------------------------


def _reset_module_from_generator(module: nn.Module, generator: torch.Generator) -> None:
    """Re-initialise a module with the parent's distribution, private stream.

    ``nn.Linear`` (weight and bias) uses ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``
    (the default ``kaiming_uniform_(a=sqrt(5))`` result), ``nn.Embedding`` uses
    ``N(0, 1)`` and ``nn.LayerNorm`` keeps the identity affine.  All draws come
    from the explicitly passed private generator; the parent torch stream is
    never consumed.
    """
    if isinstance(module, nn.Linear):
        fan_in = int(module.weight.shape[1])
        bound = 1.0 / math.sqrt(float(max(fan_in, 1)))
        with torch.no_grad():
            module.weight.uniform_(-bound, bound, generator=generator)
            if module.bias is not None:
                module.bias.uniform_(-bound, bound, generator=generator)
        return
    if isinstance(module, nn.Embedding):
        with torch.no_grad():
            module.weight.normal_(0.0, 1.0, generator=generator)
        return
    if isinstance(module, nn.LayerNorm):
        with torch.no_grad():
            module.weight.fill_(1.0)
            module.bias.fill_(0.0)
        return
    for child in module.children():
        _reset_module_from_generator(child, generator)


# ---------------------------------------------------------------------------
# the unified model
# ---------------------------------------------------------------------------


class LatentScaleSEM108(lb.LatentBridgeSEM108):
    """Sem108 + task dictionary with a single ``ScaleSpec`` width multiplier.

    ``spec = SMALL`` is constructed through the untouched
    ``LatentBridgeSEM108`` constructor: identical parameter names, identical
    initial values and identical RNG consumption as
    ``build_latent_bridge_model``.  ``spec = FULL`` starts from that same Small
    skeleton and replaces only the task-path modules in a separate, recorded
    torch generator stream (``scale_seed``); every fixed module keeps its
    Small initial value.
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        spec: ScaleSpec = SMALL,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
        scale_seed: int = 0,
    ) -> None:
        if not isinstance(spec, ScaleSpec):
            raise TypeError("spec must be a ScaleSpec")
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        self.scale_spec = spec
        self.scale_seed = int(scale_seed)
        if spec.m != 1:
            self._widen_task_path_(spec, int(scale_seed))

    # -- widening --------------------------------------------------------------

    def _widen_task_path_(self, spec: ScaleSpec, seed: int) -> None:
        generator = torch.Generator().manual_seed(int(seed))
        d, p = spec.d, spec.p
        # The bridge itself is width-generic and uses its own private NumPy
        # stream (seed 0), exactly like the Small reference frame.
        self.local_dictionary_bridge = lb.LatentDictionaryBridge(
            dim=d,
            n_atoms=spec.k,
            seed=lb.BRIDGE_SEED,
            lambda1=lb.BRIDGE_LAMBDA1,
            lambda2=lb.BRIDGE_LAMBDA2,
            steps=lb.BRIDGE_STEPS,
        )
        self.fusion = sem._mlp(spec.fusion_in, spec.fusion_hidden, d)
        self.pair_projection = nn.Linear(d, p, bias=False)
        self.relation_encoder = zpp._MLPBlock(
            int(p2.RELATION_WIDTH), spec.relation_hidden, p, float(p2.BACKEND_DROPOUT)
        )
        self.distance_gate = nn.Embedding(int(p2.DISTANCE_BUCKETS), p)
        self.pair_encoder = zpp._MLPBlock(
            4 * p, spec.pair_hidden, p, float(p2.BACKEND_DROPOUT)
        )
        self.reader = GenericReader(
            spec.reader_input, (spec.reader_hidden, spec.reader_hidden)
        )
        for module in (
            self.fusion,
            self.pair_projection,
            self.relation_encoder,
            self.distance_gate,
            self.pair_encoder,
            self.reader,
        ):
            _reset_module_from_generator(module, generator)
        # inference-only bridge interventions stay available on the new bridge.
        self._bridge_permute_seed = None

    # -- diagnostics -----------------------------------------------------------

    @property
    def spec(self) -> ScaleSpec:
        return self.scale_spec

    def environment_parts_with_aux(self, coord: torch.Tensor, data: Any, **kwargs: Any):
        """``(E, bridge_aux)`` on the plain path (diagnostics / audits only)."""
        h = sem.SEM108Model._environment_from_parts(self, coord, data, **kwargs)
        return self.local_dictionary_bridge(h, return_aux=True)


def build_scale_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    spec: ScaleSpec = FULL,
    *,
    clean_spec: Any = None,
    scale_seed: int = 0,
) -> LatentScaleSEM108:
    """Build the scaled candidate with the exact parent RNG stream for shared params.

    ``m = 1`` is byte-identical to ``build_latent_bridge_model``; ``m = 3``
    replaces the task-path modules in the private ``scale_seed`` stream.
    """
    clean_spec = cssd.CSSD_SPEC if clean_spec is None else clean_spec
    torch.manual_seed(int(seed))
    return LatentScaleSEM108(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        spec=spec,
        node_binding=clean_spec.node_binding,
        edge_binding=clean_spec.edge_binding,
        coding=clean_spec.coding,
        scale_seed=int(scale_seed),
    )


def build_small_scale_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    *,
    clean_spec: Any = None,
) -> LatentScaleSEM108:
    return build_scale_model(dictionary, seed, subspace, SMALL, clean_spec=clean_spec)


# ---------------------------------------------------------------------------
# parameter audit
# ---------------------------------------------------------------------------


def _parameter_inventory(model: LatentScaleSEM108) -> dict[str, int]:
    bridge = model.local_dictionary_bridge
    return {
        "structural_dictionary": int(model.D.numel()),
        "node_binding": int(model.W_A_S.numel() + model.W_A_C.numel()),
        "edge_binding": int(model.W_E_S.numel() + model.W_E_C.numel()),
        "node_slot_encoder": int(sum(parameter.numel() for parameter in model.node_encoder.parameters())),
        "edge_slot_encoder": int(sum(parameter.numel() for parameter in model.edge_encoder.parameters())),
        "fusion": int(sum(parameter.numel() for parameter in model.fusion.parameters())),
        "task_dictionary_D_V": int(bridge.D_L.numel() + bridge.V_L.numel()),
        "pair_projection": int(model.pair_projection.weight.numel()),
        "relation_encoder": int(sum(parameter.numel() for parameter in model.relation_encoder.parameters())),
        "distance_gate": int(model.distance_gate.weight.numel()),
        "pair_encoder": int(sum(parameter.numel() for parameter in model.pair_encoder.parameters())),
        "global_encoder": int(sum(parameter.numel() for parameter in model.global_encoder.parameters())),
        "topology_encoder": int(sum(parameter.numel() for parameter in model.topology_encoder.parameters())),
        "reader": int(sum(parameter.numel() for parameter in model.reader.parameters())),
    }


def scale_parameter_audit(model: LatentScaleSEM108) -> dict[str, Any]:
    """Exact per-component accounting against the ``ScaleSpec`` closed form."""
    spec = model.scale_spec
    expected = spec.inventory()
    actual = _parameter_inventory(model)
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    bridge = actual["task_dictionary_D_V"]
    body = int(total - bridge)
    mismatches = {key: [expected[key], actual[key]] for key in expected if expected[key] != actual[key]}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "spec": spec.as_dict(),
        "expected_inventory": expected,
        "actual_inventory": actual,
        "inventory_mismatches": mismatches,
        "inventory_exact": bool(not mismatches),
        "expected_parameters": spec.parameters(),
        "actual_parameters": total,
        "trainable_parameters": trainable,
        "all_trainable": bool(trainable == total),
        "task_dictionary_parameters": bridge,
        "body_parameters": body,
        "parameter_exact": bool(total == spec.parameters()),
        "within_budget": bool(total <= PARAM_BUDGET_MAX),
        "budget_max": int(PARAM_BUDGET_MAX),
    }


def fixed_module_state(model: LatentScaleSEM108) -> dict[str, torch.Tensor]:
    """The parameters / buffers that must never depend on the multiplier."""
    state: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name.startswith(FIXED_PARAMETER_NAMES):
            state[name] = parameter.detach().cpu().clone()
    for name, buffer in model.named_buffers():
        if name in FIXED_BUFFER_NAMES:
            state[name] = buffer.detach().cpu().clone()
    return state


def compare_fixed_modules(
    small: LatentScaleSEM108, full: LatentScaleSEM108
) -> dict[str, Any]:
    """Bit-exact comparison of every multiplier-independent parameter / buffer."""
    small_state = fixed_module_state(small)
    full_state = fixed_module_state(full)
    name_mismatch = sorted(set(small_state) ^ set(full_state))
    value_mismatch = [
        name
        for name in sorted(set(small_state) & set(full_state))
        if not torch.equal(small_state[name], full_state[name])
    ]
    return {
        "n_fixed_tensors": int(len(small_state)),
        "name_mismatch": name_mismatch,
        "value_mismatch": value_mismatch,
        "identical": bool(not name_mismatch and not value_mismatch),
    }


# ---------------------------------------------------------------------------
# eval-mode function-containment witness (Small embedded into Full)
# ---------------------------------------------------------------------------


def _repeated_map(size: int, multiplier: int) -> np.ndarray:
    return np.tile(np.arange(int(size)), int(multiplier))


def _expand_linear(
    weight: np.ndarray, bias: np.ndarray | None, input_map: np.ndarray, output_map: np.ndarray
):
    """Eval-mode Net2Wider embedding for a PyTorch-layout weight ``[out, in]``.

    Replicated incoming weights are divided by their multiplicity so the
    widened layer reproduces the Small activation exactly.
    """
    input_map = np.asarray(input_map, dtype=np.int64)
    output_map = np.asarray(output_map, dtype=np.int64)
    multiplicity = np.bincount(input_map, minlength=weight.shape[1])
    divisor = np.maximum(multiplicity[input_map], 1).astype(weight.dtype)
    wide = weight[np.ix_(output_map, input_map)] / divisor[None, :]
    return wide, None if bias is None else bias[output_map]


def reader_input_map(multiplier: int) -> np.ndarray:
    """Map widened sum / square / bucket blocks back to Small; counts stay scalar."""
    result: list[np.ndarray] = []
    offset = 0
    for width in [BASE_DIM, BASE_DIM, 1] + [BASE_PAIR, BASE_PAIR, 1] * int(p2.DISTANCE_BUCKETS) + [32, int(p2.TOPOLOGY_OUT)]:
        copies = int(multiplier) if width in (BASE_DIM, BASE_PAIR) else 1
        result.append(np.tile(np.arange(offset, offset + width), copies))
        offset += width
    return np.asarray(np.concatenate(result), dtype=np.int64)


def pair_input_map(multiplier: int) -> np.ndarray:
    """Map the widened 4-block pair input back to the Small 4 x 16 layout."""
    pm = _repeated_map(BASE_PAIR, multiplier)
    return np.concatenate([pm + BASE_PAIR * index for index in range(4)])


def embed_small_into_scale_state(
    small_state: Mapping[str, torch.Tensor], spec: ScaleSpec
) -> dict[str, torch.Tensor]:
    """Eval-mode containment witness: a Small state mapped into the wide model.

    This is **not** a training initialiser: it is symmetric by construction
    (each wide block is a copy of the Small block) and exists only to prove
    that the wide function family contains the Small one.  Formal Full
    training uses ``build_scale_model`` with fresh width-specific draws.
    """
    if spec.m == 1:
        return {key: value.clone() for key, value in small_state.items()}
    if spec.m not in (2, 3):
        raise ValueError("containment witness supports multiplier 2 or 3")
    m = int(spec.m)
    dm = _repeated_map(BASE_DIM, m)
    pm = _repeated_map(BASE_PAIR, m)
    fusion_hidden = _repeated_map(BASE_FUSION_HIDDEN, m)
    relation_hidden = _repeated_map(BASE_RELATION_HIDDEN, m)
    pair_hidden = _repeated_map(BASE_PAIR_HIDDEN, m)
    reader_hidden = _repeated_map(BASE_READER_HIDDEN, m)
    out: dict[str, torch.Tensor] = {}

    # unchanged entries are copied verbatim.
    for key, value in small_state.items():
        if key.startswith("local_dictionary_bridge."):
            continue
        if key.startswith("fusion.") or key.startswith("pair_projection."):
            continue
        if key.startswith("relation_encoder.") or key.startswith("pair_encoder."):
            continue
        if key.startswith("distance_gate.") or key.startswith("reader."):
            continue
        out[key] = value.clone()

    def _numpy(key: str) -> np.ndarray:
        return small_state[key].detach().cpu().numpy()

    # fusion: 446 -> 114 -> 48  ==>  446 -> 114m -> 48m
    weight, bias = _expand_linear(
        _numpy("fusion.0.weight"), _numpy("fusion.0.bias"), np.arange(int(sem.SEM_FUSION_IN)), fusion_hidden
    )
    out["fusion.0.weight"] = torch.from_numpy(weight)
    out["fusion.0.bias"] = torch.from_numpy(bias)
    weight, bias = _expand_linear(
        _numpy("fusion.2.weight"), _numpy("fusion.2.bias"), fusion_hidden, dm
    )
    out["fusion.2.weight"] = torch.from_numpy(weight)
    out["fusion.2.bias"] = torch.from_numpy(bias)

    # task dictionary: block-diagonal embedding of the Small D_L / V_L.
    d_small = _numpy("local_dictionary_bridge.D_L")
    v_small = _numpy("local_dictionary_bridge.V_L")
    d_wide = np.zeros((spec.d, spec.k), dtype=d_small.dtype)
    v_wide = np.zeros((spec.k, spec.d), dtype=v_small.dtype)
    for block in range(m):
        rows = slice(BASE_DIM * block, BASE_DIM * (block + 1))
        cols = slice(BASE_DIM * 2 * block, BASE_DIM * 2 * (block + 1))
        d_wide[rows, cols] = d_small
        v_wide[cols, rows] = v_small
    out["local_dictionary_bridge.D_L"] = torch.from_numpy(d_wide)
    out["local_dictionary_bridge.V_L"] = torch.from_numpy(v_wide)

    # pair projection: 48 -> 16  ==>  48m -> 16m
    weight, _ = _expand_linear(
        _numpy("pair_projection.weight"), None, dm, pm
    )
    out["pair_projection.weight"] = torch.from_numpy(weight)

    # relation encoder (_MLPBlock): 15 -> 32 -> 16 with LayerNorm.
    weight, bias = _expand_linear(
        _numpy("relation_encoder.layers.0.weight"),
        _numpy("relation_encoder.layers.0.bias"),
        np.arange(int(p2.RELATION_WIDTH)),
        relation_hidden,
    )
    out["relation_encoder.layers.0.weight"] = torch.from_numpy(weight)
    out["relation_encoder.layers.0.bias"] = torch.from_numpy(bias)
    out["relation_encoder.layers.1.weight"] = torch.from_numpy(
        _numpy("relation_encoder.layers.1.weight")[relation_hidden]
    )
    out["relation_encoder.layers.1.bias"] = torch.from_numpy(
        _numpy("relation_encoder.layers.1.bias")[relation_hidden]
    )
    weight, bias = _expand_linear(
        _numpy("relation_encoder.layers.4.weight"),
        _numpy("relation_encoder.layers.4.bias"),
        relation_hidden,
        pm,
    )
    out["relation_encoder.layers.4.weight"] = torch.from_numpy(weight)
    out["relation_encoder.layers.4.bias"] = torch.from_numpy(bias)

    # distance gate: [5, 16] -> [5, 16m]
    out["distance_gate.weight"] = torch.from_numpy(_numpy("distance_gate.weight")[:, pm])

    # pair encoder (_MLPBlock): 4*16 -> 64 -> 16  ==>  4*16m -> 64m -> 16m
    weight, bias = _expand_linear(
        _numpy("pair_encoder.layers.0.weight"),
        _numpy("pair_encoder.layers.0.bias"),
        pair_input_map(m),
        pair_hidden,
    )
    out["pair_encoder.layers.0.weight"] = torch.from_numpy(weight)
    out["pair_encoder.layers.0.bias"] = torch.from_numpy(bias)
    out["pair_encoder.layers.1.weight"] = torch.from_numpy(
        _numpy("pair_encoder.layers.1.weight")[pair_hidden]
    )
    out["pair_encoder.layers.1.bias"] = torch.from_numpy(
        _numpy("pair_encoder.layers.1.bias")[pair_hidden]
    )
    weight, bias = _expand_linear(
        _numpy("pair_encoder.layers.4.weight"),
        _numpy("pair_encoder.layers.4.bias"),
        pair_hidden,
        pm,
    )
    out["pair_encoder.layers.4.weight"] = torch.from_numpy(weight)
    out["pair_encoder.layers.4.bias"] = torch.from_numpy(bias)

    # reader (GenericReader): R -> 13 -> 13 -> 1  ==>  R_m -> 13m -> 13m -> 1
    weight, bias = _expand_linear(
        _numpy("reader.net.0.weight"),
        _numpy("reader.net.0.bias"),
        reader_input_map(m),
        reader_hidden,
    )
    out["reader.net.0.weight"] = torch.from_numpy(weight)
    out["reader.net.0.bias"] = torch.from_numpy(bias)
    weight, bias = _expand_linear(
        _numpy("reader.net.2.weight"),
        _numpy("reader.net.2.bias"),
        reader_hidden,
        reader_hidden,
    )
    out["reader.net.2.weight"] = torch.from_numpy(weight)
    out["reader.net.2.bias"] = torch.from_numpy(bias)
    weight, bias = _expand_linear(
        _numpy("reader.net.4.weight"),
        _numpy("reader.net.4.bias"),
        reader_hidden,
        np.arange(1),
    )
    out["reader.net.4.weight"] = torch.from_numpy(weight)
    out["reader.net.4.bias"] = torch.from_numpy(bias)
    return out


def containment_witness_delta(
    small: LatentScaleSEM108,
    wide: LatentScaleSEM108,
    batch: Any,
    *,
    mask: Any = None,
) -> dict[str, float]:
    """Load the embedded Small state into ``wide`` and compare predictions."""
    embedded = embed_small_into_scale_state(small.state_dict(), wide.scale_spec)
    wide.load_state_dict(embedded)
    small.eval()
    wide.eval()
    with torch.no_grad():
        plain_small = small(batch).view(-1).double()
        plain_wide = wide(batch).view(-1).double()
        masked_small = small(batch, mask=mask).view(-1).double()
        masked_wide = wide(batch, mask=mask).view(-1).double()
    return {
        "plain_prediction_max_delta": float((plain_small - plain_wide).abs().max()),
        "masked_prediction_max_delta": float((masked_small - masked_wide).abs().max()),
    }


# ---------------------------------------------------------------------------
# width capture on a real batch
# ---------------------------------------------------------------------------


def capture_task_path_widths(
    model: LatentScaleSEM108, batch: Any, *, mask: Any = None
) -> dict[str, int]:
    """Capture h / E / alpha / u / pair / reader-input widths on one forward."""
    captured: dict[str, int] = {}
    handles: list[Any] = []
    bridge = model.local_dictionary_bridge
    handles.append(
        bridge.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("h", int(inputs[0].shape[1]))
        )
    )
    handles.append(
        bridge.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__(
                "E", int((output[0] if isinstance(output, tuple) else output).shape[1])
            )
        )
    )
    handles.append(
        model.pair_projection.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("u", int(output.shape[1]))
        )
    )
    handles.append(
        model.pair_encoder.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("pair", int(output.shape[1]))
        )
    )
    handles.append(
        model.reader.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("reader_input", int(inputs[0].shape[1]))
        )
    )
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            if mask is None:
                model(batch)
            else:
                model(batch, mask=mask)
        with torch.no_grad():
            h = _capture_bridge_input(model, batch, mask)
            _e, aux = model.local_dictionary_bridge(h, return_aux=True)
        captured["alpha"] = int(aux["alpha"].shape[1])
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return captured


def _capture_bridge_input(model: LatentScaleSEM108, batch: Any, mask: Any | None) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}
    handle = model.local_dictionary_bridge.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("h", inputs[0].detach())
    )
    try:
        with torch.no_grad():
            if mask is None:
                model(batch)
            else:
                model(batch, mask=mask)
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError("bridge was not called during the forward pass")
    return captured["h"]


__all__ = [
    "PROTOCOL_VERSION",
    "BASE_DIM",
    "BASE_PAIR",
    "BASE_FUSION_HIDDEN",
    "PARAM_BUDGET_MAX",
    "FIXED_PARAMETER_NAMES",
    "FIXED_BUFFER_NAMES",
    "ScaleSpec",
    "SMALL",
    "FULL",
    "AUDIT_ONLY_M2",
    "PRESETS",
    "TRAIN_PRESETS",
    "scale_spec",
    "unary_block_layout",
    "pair_block_layout",
    "pair_block_dim",
    "relation_readout_dim",
    "readout_layout",
    "LatentScaleSEM108",
    "build_scale_model",
    "build_small_scale_model",
    "scale_parameter_audit",
    "fixed_module_state",
    "compare_fixed_modules",
    "reader_input_map",
    "pair_input_map",
    "embed_small_into_scale_state",
    "containment_witness_delta",
    "capture_task_path_widths",
    "official_test_blocker",
]
