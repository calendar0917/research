"""FEC-S0 -- Factorized Environment-Composition S0 (zero-training equivalence audit).

Round: ``fec_s0``.  Pre-registration: ``notes/fec_s0_preregistration.md``.
Prior-artifact audit: ``notes/fec_s0_prior_artifact_audit.md``.

This module is an **audit / factorized re-implementation**, not a new model and
not a training run.  It reconstructs every S0 input tensor that reaches the
prediction from *raw molecular primitives* (atom categories, bond categories and
the untyped graph) through explicit role x primitive bindings, then compares the
reconstructed tensors, the intermediate model states and the final prediction
against the historical strict-static S0 execution.

No S0 weight is modified.  The original ``StrictStaticPairModel`` is reused
verbatim; only the feature construction that feeds it is replaced.

Path classification (frozen in the pre-registration):

* ``FACTORIZED_SHARED`` -- shared role descriptor x shared primitive descriptor,
  operator identical for every patch (``patch_cont`` blocks, ``pair_relation``
  topology/chemistry blocks, ``global_context`` marginals).
* ``EXPLICIT_BINDING_LOOKUP`` -- deterministic canonical key of an explicit
  binding followed by a data-fit vocabulary table (the historical aliased typed
  / parent token).  Reconstructible, but a per-joint-configuration memory, not a
  shared role x primitive factorization.
* ``PURE_TOPOLOGY`` -- depends only on the untyped graph.
* ``NOT_REACHABLE`` / ``OPAQUE``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.fec_s0_factorization all
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch, Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _global_structure_blocks,
    _load_zinc,
)
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
    _extract_v4_records,
)
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    build_colored_incidence,
    historical_certificate,
)
from tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair import (
    CANONICAL_V4_CONFIG,
    StrictStaticPairModel,
    build_s0,
    load_encoded,
)

# ---------------------------------------------------------------------------
# frozen constants (inherited, never re-derived)
# ---------------------------------------------------------------------------

TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/fec_s0"
ZINC_ROOT = REPO_ROOT / "data/ZINC"
S0_STATE_PATH = TRACK_ROOT / "results/zinc_static_dictionary_pair/states/s0_seed0_selection_state.pt"
S0_RUN_JSON = TRACK_ROOT / "results/zinc_static_dictionary_pair/runs/s0_seed0.json"

ATOM_CATEGORIES = zpp.ATOM_CATEGORIES  # 28
BOND_CATEGORIES = zpp.BOND_CATEGORIES  # 4
PATCH_RADIUS = zpp.PATCH_RADIUS  # 2
DISTANCE_BUCKETS = zpp.DISTANCE_BUCKETS  # 5
SHELL_PAIRS = zpp.SHELL_PAIRS  # 6 classes, including the empty (0,0)
SHELL_WIDTH = zpp.SHELL_WIDTH  # 146
RELATION_WIDTH = zpp.RELATION_WIDTH  # 23
GLOBAL_WIDTH = zpp.GLOBAL_WIDTH  # 62
ATOM_BINS = 28
BOND_BINS = 4

#: pre-registered tolerances (frozen in notes/fec_s0_preregistration.md §6)
TOL_RAW = 1.0e-7
TOL_STANDARDIZED = 1.0e-7
TOL_INTERMEDIATE = 1.0e-6
TOL_PREDICTION = 1.0e-6
TOKENIZER_VERSION = "typed_tokenizer_v1_historical"

#: frozen block boundaries of the 146-D shell descriptor
SHELL_BLOCKS = {
    "atom_shell": (0, 84),
    "bond_shell": (84, 108),
    "root_atom": (108, 136),
    "incident_bonds": (136, 140),
    "scalars": (140, 146),
}
RELATION_BLOCKS = {
    "topology_distance_logdist": (0, 6),
    "topology_overlap_boundary": (6, 14),
    "chemistry_path_bond_mean": (14, 18),
    "topology_log_path_count": (18, 19),
    "chemistry_adjacent_bond": (19, 23),
}
GLOBAL_BLOCKS = {
    "topology_structure": (0, 30),
    "atom_marginal": (30, 58),
    "bond_marginal": (58, 62),
}


def _onehot(index: int, width: int) -> np.ndarray:
    value = np.zeros(int(width), dtype=np.float32)
    value[int(index)] = 1.0
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


# ---------------------------------------------------------------------------
# factorized raw feature builders (explicit role x primitive)
# ---------------------------------------------------------------------------


def factorized_shell_descriptor(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    distances: Mapping[int, int],
    radius: int,
) -> np.ndarray:
    """Reconstruct the historical 146-D ``patch_cont`` from raw primitives.

    Explicit role x primitive bindings::

        atom_shell[k, a] = sum_v 1[shell_i(v)=k] * 1[atom(v)=a] / n_nodes
        bond_shell[p, b] = sum_e 1[shellpair_i(e)=p] * 1[bond(e)=b] / n_edges
        root_atom[a]     = 1[atom(root)=a]
        incident[b]      = sum_e incident 1[bond(e)=b] / deg(root)
        scalars          = [log1p(n), log1p(m), boundary_frac, cycle_rank/n,
                            deg(root)/4, mean_degree/4]

    The four chemistry-bearing blocks are pure role x primitive sums; the six
    scalars are pure topology.  The arithmetic mirrors the historical
    ``zinc_patch_path_pooling._shell_descriptor`` exactly (same accumulation
    cells, same denominators) so the comparison can be bit-identical.
    """
    radius = int(radius)
    shell_pairs = zpp._shell_pairs_for_radius(radius)
    nodes = frozenset(int(node) for node in distances)
    induced = graph.induced(set(nodes))
    n_nodes = len(nodes)
    n_edges = induced.num_edges()

    atom_shell = np.zeros((radius + 1, ATOM_CATEGORIES), dtype=np.float32)
    for node in nodes:
        atom_shell += np.outer(
            _onehot(int(distances[node]), radius + 1),
            _onehot(int(node_types[node]), ATOM_CATEGORIES),
        )
    atom_shell /= max(float(n_nodes), 1.0)

    bond_shell = np.zeros((len(shell_pairs), BOND_CATEGORIES), dtype=np.float32)
    shell_pair_index = {pair: index for index, pair in enumerate(shell_pairs)}
    for left, right in induced.edges():
        pair = tuple(sorted((int(distances[left]), int(distances[right]))))
        bond_shell += np.outer(
            _onehot(shell_pair_index[pair], len(shell_pairs)),
            _onehot(int(edge_types[graph.edge_key(int(left), int(right))]), BOND_CATEGORIES),
        )
    bond_shell /= max(float(n_edges), 1.0)

    root_atom = _onehot(int(node_types[int(center)]), ATOM_CATEGORIES)

    incident_bonds = np.zeros(BOND_CATEGORIES, dtype=np.float32)
    for neighbor in graph.neighbors(int(center)):
        incident_bonds += _onehot(
            int(edge_types[graph.edge_key(int(center), int(neighbor))]), BOND_CATEGORIES
        )
    incident_bonds /= max(float(len(graph.neighbors(int(center)))), 1.0)

    cycle_rank = max(int(n_edges) - int(n_nodes) + 1, 0)
    degrees = np.asarray([len(graph.neighbors(node)) for node in nodes], dtype=np.float32)
    scalars = np.asarray(
        [
            np.log1p(float(n_nodes)),
            np.log1p(float(n_edges)),
            float(len(nodes) and sum(int(distance) == radius for distance in distances.values()))
            / max(float(n_nodes), 1.0),
            float(cycle_rank) / max(float(n_nodes), 1.0),
            float(len(graph.neighbors(int(center)))) / 4.0,
            float(degrees.mean()) / 4.0 if degrees.size else 0.0,
        ],
        dtype=np.float32,
    )
    descriptor = np.concatenate(
        [atom_shell.reshape(-1), bond_shell.reshape(-1), root_atom, incident_bonds, scalars]
    ).astype(np.float32, copy=False)
    if descriptor.shape != (SHELL_WIDTH,):
        raise RuntimeError(f"factorized shell width {descriptor.shape} != {(SHELL_WIDTH,)}")
    return descriptor


def factorized_pair_relation(
    left_nodes: frozenset[int],
    left_boundary: frozenset[int],
    right_nodes: frozenset[int],
    right_boundary: frozenset[int],
    path_summary: tuple[int, float, np.ndarray],
    adjacent_bond: int | None,
    patch_radius: int,
) -> np.ndarray:
    """Reconstruct the historical 23-D ``pair_relation`` as topology + chemistry.

    Historical ordering preserved exactly::

        [ distance one-hot(5) | log distance(1) | overlap(5) | boundary(3)
          | path_bond_mean(4) | log path count(1) | adjacent bond(4) ]

    ``path_bond_mean`` and ``adjacent bond`` are the composition-level chemical
    relation primitives; every other coordinate is pure untyped-graph topology.
    """
    distance, path_count, path_bond_mean = path_summary
    bucket = min(max(int(distance), 1), DISTANCE_BUCKETS) - 1

    distance_one_hot = np.zeros(DISTANCE_BUCKETS, dtype=np.float32)
    distance_one_hot[bucket] = 1.0

    intersection = len(left_nodes & right_nodes)
    union = len(left_nodes | right_nodes)
    left_size = len(left_nodes)
    right_size = len(right_nodes)
    denominator = max(float(int(patch_radius) * int(patch_radius) + 10), 1.0)
    overlap = np.asarray(
        [
            float(intersection) / denominator,
            float(intersection) / max(float(union), 1.0),
            float(intersection) / max(float(min(left_size, right_size)), 1.0),
            float(intersection) / max(float(max(left_size, right_size)), 1.0),
            float(abs(left_size - right_size)) / denominator,
        ],
        dtype=np.float32,
    )
    boundary_intersection = len(left_boundary & right_boundary)
    boundary_union = len(left_boundary | right_boundary)
    boundary_min = min(len(left_boundary), len(right_boundary))
    boundary_features = np.asarray(
        [
            float(boundary_intersection) / max(float(boundary_union), 1.0),
            float(boundary_intersection) / max(float(boundary_min), 1.0),
            float(boundary_intersection > 0),
        ],
        dtype=np.float32,
    )
    topology_block = np.concatenate(
        [
            distance_one_hot,
            np.asarray([np.log1p(float(distance))], dtype=np.float32),
            overlap,
            boundary_features,
        ]
    )
    chemistry_path_block = np.asarray(path_bond_mean, dtype=np.float32)
    topology_count_block = np.asarray([np.log1p(float(path_count))], dtype=np.float32)
    adjacent = np.zeros(BOND_CATEGORIES, dtype=np.float32)
    if adjacent_bond is not None:
        adjacent[int(adjacent_bond)] = 1.0

    relation = np.concatenate([topology_block, chemistry_path_block, topology_count_block, adjacent]).astype(
        np.float32, copy=False
    )
    if relation.shape != (RELATION_WIDTH,):
        raise RuntimeError(f"factorized relation width {relation.shape} != {(RELATION_WIDTH,)}")
    return relation


def factorized_global_context(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
) -> np.ndarray:
    """Reconstruct the 62-D ``global_context`` as topology + atom/bond marginals."""
    short, long = _global_structure_blocks(graph)
    atom_counts = np.bincount(np.asarray(node_types, dtype=np.int64) % ATOM_BINS, minlength=ATOM_BINS).astype(
        np.float64
    )
    atom_hist = atom_counts / max(atom_counts.sum(), 1.0)
    bond_values = np.asarray(list(edge_types.values()), dtype=np.int64)
    bond_counts = np.bincount(bond_values % BOND_BINS, minlength=BOND_BINS).astype(np.float64)
    bond_hist = bond_counts / max(bond_counts.sum(), 1.0)
    context = np.concatenate([short, long, atom_hist, bond_hist]).astype(np.float32, copy=False)
    if context.shape != (GLOBAL_WIDTH,):
        raise RuntimeError(f"factorized global width {context.shape} != {(GLOBAL_WIDTH,)}")
    return context


def reconstruct_typed_key(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    radius: int,
) -> bytes:
    """Rebuild the historical aliased token from raw primitives.

    The explicit binding puts structure and chemistry together for the first
    time: node color ``(is_root, distance, atom_type)`` and incidence-edge color
    ``(bond_type)``.  The historical ``pynauty`` certificate is a
    **non-injective** projection of that binding (it does not serialise the
    color labels), i.e. an ``EXPLICIT_BINDING_LOOKUP`` key, not a shared
    role x primitive factorization.
    """
    incidence = build_colored_incidence(graph, int(center), node_types, edge_types, int(radius))
    return historical_certificate(incidence)


@dataclass(frozen=True)
class FactorizedRawRecord:
    patch_cont_raw: np.ndarray  # (n, 146)
    pair_index: np.ndarray  # (2, E)
    pair_relation_raw: np.ndarray  # (E, 23)
    pair_bucket: np.ndarray  # (E,)
    global_raw: np.ndarray  # (62,)
    typed_keys: tuple[bytes, ...]
    parent_keys: tuple[bytes, ...]


def factorized_record(data: Any, tokenize: bool = True) -> FactorizedRawRecord:
    """Build every raw S0 input for one molecule from raw primitives."""
    graph, node_types, edge_types = _data_to_graph(data)
    centers = list(graph.nodes)
    patch_raw: list[np.ndarray] = []
    typed_keys: list[bytes] = []
    parent_keys: list[bytes] = []
    patch_nodes: list[frozenset[int]] = []
    patch_boundary: list[frozenset[int]] = []
    for center in centers:
        distances = zpp._ego_distances(graph, int(center), PATCH_RADIUS)
        patch_raw.append(
            factorized_shell_descriptor(graph, int(center), node_types, edge_types, distances, PATCH_RADIUS)
        )
        patch_nodes.append(frozenset(int(node) for node in distances))
        patch_boundary.append(
            frozenset(int(node) for node, distance in distances.items() if int(distance) == PATCH_RADIUS)
        )
        if tokenize:
            typed_keys.append(reconstruct_typed_key(graph, int(center), node_types, edge_types, PATCH_RADIUS))
            parent_keys.append(
                reconstruct_typed_key(graph, int(center), node_types, edge_types, max(1, PATCH_RADIUS - 1))
            )

    shortest_paths = {
        int(source): zpp._shortest_path_summary(graph, int(source), edge_types) for source in centers
    }
    pair_sources: list[int] = []
    pair_targets: list[int] = []
    pair_relations: list[np.ndarray] = []
    pair_buckets: list[int] = []
    for left_index, left_center in enumerate(centers):
        for right_index in range(left_index + 1, len(centers)):
            right_center = int(centers[right_index])
            path_summary = shortest_paths[int(left_center)][right_center]
            adjacent_bond = None
            if int(path_summary[0]) == 1:
                adjacent_bond = int(edge_types[graph.edge_key(int(left_center), right_center)])
            pair_relations.append(
                factorized_pair_relation(
                    patch_nodes[left_index],
                    patch_boundary[left_index],
                    patch_nodes[right_index],
                    patch_boundary[right_index],
                    path_summary,
                    adjacent_bond,
                    PATCH_RADIUS,
                )
            )
            pair_sources.append(int(left_index))
            pair_targets.append(int(right_index))
            pair_buckets.append(min(max(int(path_summary[0]), 1), DISTANCE_BUCKETS) - 1)

    pair_index = np.asarray([pair_sources, pair_targets], dtype=np.int64)
    pair_relation = (
        np.stack(pair_relations, axis=0).astype(np.float32, copy=False)
        if pair_relations
        else np.zeros((0, RELATION_WIDTH), dtype=np.float32)
    )
    return FactorizedRawRecord(
        patch_cont_raw=np.stack(patch_raw, axis=0).astype(np.float32, copy=False),
        pair_index=pair_index,
        pair_relation_raw=pair_relation,
        pair_bucket=np.asarray(pair_buckets, dtype=np.int64),
        global_raw=factorized_global_context(graph, node_types, edge_types),
        typed_keys=tuple(typed_keys),
        parent_keys=tuple(parent_keys),
    )


# ---------------------------------------------------------------------------
# frozen fitted transforms (train-only, historical semantics)
# ---------------------------------------------------------------------------


@dataclass
class FecFits:
    patch_standardizer: zpp.Standardizer
    context_standardizer: zpp.Standardizer
    topology_standardizer: zpp.Standardizer
    typed_vocabulary: Mapping[bytes, int]
    parent_vocabulary: Mapping[bytes, int]


def _canonical_config() -> dict[str, Any]:
    config = yaml.safe_load(CANONICAL_V4_CONFIG.read_text(encoding="utf-8"))
    config["test_policy"] = "no_test"
    config["model"]["device"] = "cpu"
    return config


def build_fits(train_records: Sequence[Any], config: Mapping[str, Any] | None = None) -> FecFits:
    """Reproduce the historical train-only fits with the historical functions."""
    config = _canonical_config() if config is None else config
    representation = config["representation"]
    typed_vocabulary = zpp._fit_vocabulary(
        train_records,
        "typed_certificate",
        int(representation.get("max_typed_tokens", 8192)),
        int(representation.get("minimum_typed_frequency", 1)),
    )
    parent_vocabulary = zpp._fit_vocabulary(
        train_records,
        "parent_certificate",
        int(representation.get("max_parent_tokens", 2048)),
        int(representation.get("minimum_parent_frequency", 1)),
    )
    patch_standardizer = zpp.Standardizer.fit(zpp._patch_matrix(train_records))
    context_standardizer = zpp.Standardizer.fit(zpp._context_matrix(train_records))
    topology_matrix = np.stack([record.topology_features for record in train_records], axis=0).astype(
        np.float32, copy=False
    )
    topology_standardizer = zpp.Standardizer.fit(topology_matrix)
    return FecFits(
        patch_standardizer=patch_standardizer,
        context_standardizer=context_standardizer,
        topology_standardizer=topology_standardizer,
        typed_vocabulary=typed_vocabulary,
        parent_vocabulary=parent_vocabulary,
    )


class FactorizedFeatureTransform:
    """Build a model-ready ``Data`` from raw molecular primitives.

    Reuses the frozen S0 fitted transforms and vocabularies.  ``topology_raw``
    is the pure-topology hinge row (retained verbatim, labelled).
    """

    def __init__(self, fits: FecFits) -> None:
        self.fits = fits

    def build(
        self,
        raw_data: Any,
        y: float,
        topology_raw: np.ndarray | None = None,
        raw: FactorizedRawRecord | None = None,
    ) -> Data:
        record = factorized_record(raw_data, tokenize=True) if raw is None else raw
        patch_cont = self.fits.patch_standardizer.transform(record.patch_cont_raw)
        global_context = self.fits.context_standardizer.transform(record.global_raw[None, :])
        n_patches = int(record.patch_cont_raw.shape[0])
        typed = np.asarray(
            [self.fits.typed_vocabulary.get(key, 0) for key in record.typed_keys],
            dtype=np.int64,
        )
        parent = np.asarray(
            [self.fits.parent_vocabulary.get(key, 0) for key in record.parent_keys],
            dtype=np.int64,
        )
        if topology_raw is None:
            topology = np.zeros((1, 0), dtype=np.float32)
        else:
            topology = self.fits.topology_standardizer.transform(
                np.asarray(topology_raw, dtype=np.float32)[None, :]
            )
        return Data(
            patch_cont=torch.from_numpy(patch_cont),
            patch_context=torch.zeros((n_patches, 0), dtype=torch.float32),
            typed_token=torch.from_numpy(typed),
            parent_token=torch.from_numpy(parent),
            structural_token=torch.zeros(n_patches, dtype=torch.int64),
            structural_coarse=torch.zeros((n_patches, 4), dtype=torch.float32),
            pair_index=torch.from_numpy(record.pair_index),
            pair_relation=torch.from_numpy(record.pair_relation_raw),
            pair_bucket=torch.from_numpy(record.pair_bucket),
            global_context=torch.from_numpy(global_context),
            topology_features=torch.from_numpy(topology),
            y=torch.tensor([float(y)], dtype=torch.float32),
            num_nodes=n_patches,
            batch=torch.zeros(n_patches, dtype=torch.long),
            attribute_atom_type=torch.zeros(0, dtype=torch.int64),
            attribute_atom_role=torch.zeros((0, 8), dtype=torch.float32),
            attribute_atom_patch_index=torch.zeros(0, dtype=torch.int64),
            attribute_bond_type=torch.zeros(0, dtype=torch.int64),
            attribute_bond_role_left=torch.zeros((0, 8), dtype=torch.float32),
            attribute_bond_role_right=torch.zeros((0, 8), dtype=torch.float32),
            attribute_bond_patch_index=torch.zeros(0, dtype=torch.int64),
        )


class FactorizedStaticEnvironmentCompositionS0(torch.nn.Module):
    """Function-preserving wrapper: raw graph -> factorized features -> S0."""

    def __init__(self, model: StrictStaticPairModel, transform: FactorizedFeatureTransform) -> None:
        super().__init__()
        self.model = model
        self.transform = transform

    def build(self, raw_data: Any, y: float, topology_raw: np.ndarray | None = None) -> Data:
        return self.transform.build(raw_data, y, topology_raw=topology_raw)

    def forward(self, raw_data: Any, y: float, topology_raw: np.ndarray | None = None) -> torch.Tensor:
        return self.model(self.build(raw_data, y, topology_raw=topology_raw)).view(-1)


# ---------------------------------------------------------------------------
# comparison + capture utilities
# ---------------------------------------------------------------------------


def _tensor_stats(reference: torch.Tensor, candidate: torch.Tensor, tol: float) -> dict[str, Any]:
    reference = reference.detach().to(torch.float64)
    candidate = candidate.detach().to(torch.float64)
    if reference.shape != candidate.shape:
        return {
            "shape_mismatch": [list(reference.shape), list(candidate.shape)],
            "max_abs": None,
            "mean_abs": None,
            "allclose": False,
            "bit_equal_fraction": 0.0,
        }
    diff = (reference - candidate).abs()
    exact = (reference == candidate).to(torch.float64)
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "allclose": bool(torch.all(diff <= tol).item()) if diff.numel() else True,
        "bit_equal_fraction": float(exact.mean().item()) if exact.numel() else 1.0,
        "tolerance": float(tol),
    }


def _array_stats(reference: np.ndarray, candidate: np.ndarray, tol: float) -> dict[str, Any]:
    return _tensor_stats(
        torch.from_numpy(np.ascontiguousarray(reference)),
        torch.from_numpy(np.ascontiguousarray(candidate)),
        tol,
    )


def _block_stats(
    reference: np.ndarray, candidate: np.ndarray, blocks: Mapping[str, tuple[int, int]], tol: float
) -> dict[str, Any]:
    return {
        name: _array_stats(reference[:, start:end], candidate[:, start:end], tol)
        for name, (start, end) in blocks.items()
    }


def _ensure_batch(data: Data) -> Data:
    """Give an un-batched historical molecule a single-graph ``batch`` vector."""
    if not hasattr(data, "batch") or data.batch is None:
        data = data.clone()
        data.batch = torch.zeros(int(data.num_nodes), dtype=torch.long)
    return data


def _capture(model: StrictStaticPairModel, data: Data) -> dict[str, torch.Tensor]:
    caps: dict[str, torch.Tensor] = {}
    data = _ensure_batch(data)

    def hook(name: str):
        def fn(_module, _inputs, output):
            caps[name] = output.detach().clone()

        return fn

    handles = [
        model.patch_encoder.register_forward_hook(hook("patch")),
        model.pair_encoder.register_forward_hook(hook("pair_value")),
        model.relation_encoder.register_forward_hook(hook("relation")),
        model.global_encoder.register_forward_hook(hook("graph_hidden")),
    ]
    try:
        with torch.no_grad():
            representation = model.encode(data)
            prediction = model(data)
    finally:
        for handle in handles:
            handle.remove()
    caps["R"] = representation.detach().clone()
    caps["pred"] = prediction.detach().clone()
    return caps


def load_s0_model() -> StrictStaticPairModel:
    model = build_s0(0)
    state = torch.load(S0_STATE_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def _module_param_counts(model: StrictStaticPairModel) -> dict[str, int]:
    return {name: int(sum(p.numel() for p in module.parameters())) for name, module in model.named_children()}


# ---------------------------------------------------------------------------
# access audit
# ---------------------------------------------------------------------------


def access_audit(
    model: StrictStaticPairModel,
    data: Data,
) -> dict[str, Any]:
    """Trace exactly which Data tensors the S0 prediction graph reads."""
    calls = {"pair_encoder": 0, "relation_encoder": 0, "patch_encoder": 0}
    handles = [
        model.pair_encoder.register_forward_hook(
            lambda *_a: calls.__setitem__("pair_encoder", calls["pair_encoder"] + 1)
        ),
        model.relation_encoder.register_forward_hook(
            lambda *_a: calls.__setitem__("relation_encoder", calls["relation_encoder"] + 1)
        ),
        model.patch_encoder.register_forward_hook(
            lambda *_a: calls.__setitem__("patch_encoder", calls["patch_encoder"] + 1)
        ),
    ]
    try:
        with torch.no_grad():
            model(data)
    finally:
        for handle in handles:
            handle.remove()

    inputs = {
        "patch_cont": {
            "width": int(data.patch_cont.shape[1]),
            "raw_sources": "shell_descriptor: atom_shell(3x28) bond_shell(6x4) root_atom(28) incident(4) scalars(6)",
            "chemistry": "yes (4 of 5 blocks)",
            "topology": "yes (roles + 6 scalars)",
            "mixed": "yes",
            "reaches_prediction": True,
            "classification": "FACTORIZED_SHARED",
        },
        "typed_token": {
            "width": 16,
            "raw_sources": "pynauty certificate(v1_historical) of the rooted typed incidence graph",
            "chemistry": "yes (joint atom+bond)",
            "topology": "yes (rooted incidence)",
            "mixed": "yes",
            "reaches_prediction": True,
            "classification": "EXPLICIT_BINDING_LOOKUP",
            "note": "aliased non-injective key; data-fit 6785-row table, independent params per key",
        },
        "parent_token": {
            "width": 8,
            "raw_sources": "same certificate at radius 1",
            "chemistry": "yes",
            "topology": "yes",
            "mixed": "yes",
            "reaches_prediction": True,
            "classification": "EXPLICIT_BINDING_LOOKUP",
            "note": "same aliased family as typed_token",
        },
        "pair_relation": {
            "width": int(data.pair_relation.shape[1]),
            "raw_sources": "untyped topology + path bond composition + adjacent bond one-hot",
            "chemistry": "yes (path_bond_mean 4 + adjacent 4)",
            "topology": "yes (distance/overlap/boundary/log path count 15)",
            "mixed": "yes (explicit concatenation)",
            "reaches_prediction": True,
            "classification": "FACTORIZED_SHARED",
        },
        "pair_bucket": {
            "width": 1,
            "raw_sources": "shortest-path distance bucket",
            "chemistry": "no",
            "topology": "yes",
            "mixed": "no",
            "reaches_prediction": True,
            "classification": "PURE_TOPOLOGY",
        },
        "global_context": {
            "width": int(data.global_context.shape[1]),
            "raw_sources": "30D pure topology + 28D atom marginal + 4D bond marginal",
            "chemistry": "yes (marginals)",
            "topology": "yes",
            "mixed": "yes (explicit concatenation)",
            "reaches_prediction": True,
            "classification": "FACTORIZED_SHARED",
        },
        "topology_features": {
            "width": int(data.topology_features.shape[1]),
            "raw_sources": "exact simple-cycle spectrum + hinge basis of the untyped graph",
            "chemistry": "no",
            "topology": "yes",
            "mixed": "no",
            "reaches_prediction": True,
            "classification": "PURE_TOPOLOGY",
        },
    }
    return {
        "protocol_version": "fec_s0_v1",
        "reachable_data_keys": sorted(data.keys()),
        "dormant_paths": {
            "patch_context": "width 0 (radius-3 outer context disabled)",
            "structural_token/structural_coarse": "structural_context_mode=none",
            "attribute_*": "attribute_mode=none",
            "center_update": "center_context=False -> center_update is None",
            "direct_token_readout": "disabled",
        },
        "forward_hook_calls_per_forward": calls,
        "input_path_table": inputs,
        "module_parameter_counts": _module_param_counts(model),
        "model": {
            "class": type(model).__name__,
            "params": int(sum(p.numel() for p in model.parameters())),
            "unified_graph_width": int(model.unified_graph_width),
            "center_update_is_none": bool(model.center_update is None),
            "residual_mode": str(model.residual_mode),
        },
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# equivalence stages
# ---------------------------------------------------------------------------


def _subset(seq: Sequence[Any], limit: int | None) -> list[Any]:
    return list(seq) if limit is None else list(seq)[: int(limit)]


def build_factorized_records(raw_dataset: Sequence[Any], limit: int | None) -> list[FactorizedRawRecord]:
    started = time.perf_counter()
    records = [factorized_record(data, tokenize=True) for data in _subset(raw_dataset, limit)]
    print(f"[fec-s0] factorized {len(records)} molecules in {time.perf_counter()-started:.1f}s", flush=True)
    return records


def local_descriptor_equivalence(
    records: Sequence[Any],
    encoded: Sequence[Data],
    factorized: Sequence[FactorizedRawRecord],
    fits: FecFits,
    limit: int | None,
) -> dict[str, Any]:
    recs = _subset(records, limit)
    encs = _subset(encoded, limit)
    facts = _subset(factorized, limit)
    hist_raw = np.concatenate([np.stack([p.shell_descriptor for p in r.patches]) for r in recs])
    fact_raw = np.concatenate([f.patch_cont_raw for f in facts])
    hist_std = np.concatenate([d.patch_cont.numpy() for d in encs])
    fact_std = np.concatenate([fits.patch_standardizer.transform(f.patch_cont_raw) for f in facts])
    n_patches = int(hist_raw.shape[0])
    return {
        "protocol_version": "fec_s0_v1",
        "n_molecules": int(len(recs)),
        "n_patches": n_patches,
        "raw_descriptor": {
            "overall": _array_stats(hist_raw, fact_raw, TOL_RAW),
            "per_block": _block_stats(hist_raw, fact_raw, SHELL_BLOCKS, TOL_RAW),
        },
        "scaled_descriptor": {
            "overall": _array_stats(hist_std, fact_std, TOL_STANDARDIZED),
            "per_block": _block_stats(hist_std, fact_std, SHELL_BLOCKS, TOL_STANDARDIZED),
        },
        "standardizer_source": "train-fit Standardizer (refit with zpp.Standardizer.fit)",
        "official_test_loaded": False,
    }


def pair_relation_equivalence(
    records: Sequence[Any],
    factorized: Sequence[FactorizedRawRecord],
    limit: int | None,
) -> dict[str, Any]:
    recs = _subset(records, limit)
    facts = _subset(factorized, limit)
    hist = np.concatenate([r.pair_relation for r in recs])
    fact = np.concatenate([f.pair_relation_raw for f in facts])
    return {
        "protocol_version": "fec_s0_v1",
        "n_molecules": int(len(recs)),
        "n_pairs": int(hist.shape[0]),
        "overall": _array_stats(hist, fact, TOL_RAW),
        "per_block": _block_stats(hist, fact, RELATION_BLOCKS, TOL_RAW),
        "official_test_loaded": False,
    }


def global_equivalence(
    records: Sequence[Any],
    encoded: Sequence[Data],
    factorized: Sequence[FactorizedRawRecord],
    fits: FecFits,
    limit: int | None,
) -> dict[str, Any]:
    recs = _subset(records, limit)
    encs = _subset(encoded, limit)
    facts = _subset(factorized, limit)
    hist_raw = np.stack([r.global_context for r in recs])
    fact_raw = np.stack([f.global_raw for f in facts])
    hist_std = np.concatenate([d.global_context.numpy() for d in encs])
    fact_std = np.concatenate([fits.context_standardizer.transform(f.global_raw[None, :]) for f in facts])
    return {
        "protocol_version": "fec_s0_v1",
        "n_molecules": int(len(recs)),
        "raw_descriptor": {
            "overall": _array_stats(hist_raw, fact_raw, TOL_RAW),
            "per_block": _block_stats(hist_raw, fact_raw, GLOBAL_BLOCKS, TOL_RAW),
        },
        "scaled_descriptor": {
            "overall": _array_stats(hist_std, fact_std, TOL_STANDARDIZED),
            "per_block": _block_stats(hist_std, fact_std, GLOBAL_BLOCKS, TOL_STANDARDIZED),
        },
        "official_test_loaded": False,
    }


def token_equivalence(
    records: Sequence[Any],
    encoded: Sequence[Data],
    factorized: Sequence[FactorizedRawRecord],
    fits: FecFits,
    limit: int | None,
) -> dict[str, Any]:
    recs = _subset(records, limit)
    encs = _subset(encoded, limit)
    facts = _subset(factorized, limit)
    typed_key_exact = 0
    parent_key_exact = 0
    typed_id_exact = 0
    parent_id_exact = 0
    typed_total = 0
    parent_total = 0
    for record, data, fact in zip(recs, encs, facts):
        hist_typed = [p.typed_certificate for p in record.patches]
        hist_parent = [p.parent_certificate for p in record.patches]
        typed_key_exact += int(tuple(hist_typed) == tuple(fact.typed_keys))
        parent_key_exact += int(tuple(hist_parent) == tuple(fact.parent_keys))
        typed_ids = np.asarray([fits.typed_vocabulary.get(key, 0) for key in fact.typed_keys], dtype=np.int64)
        parent_ids = np.asarray(
            [fits.parent_vocabulary.get(key, 0) for key in fact.parent_keys], dtype=np.int64
        )
        typed_id_exact += int(np.array_equal(typed_ids, data.typed_token.numpy()))
        parent_id_exact += int(np.array_equal(parent_ids, data.parent_token.numpy()))
        typed_total += len(hist_typed)
        parent_total += len(hist_parent)
    return {
        "protocol_version": "fec_s0_v1",
        "n_molecules": int(len(recs)),
        "n_patches": int(typed_total),
        "typed_tokenizer_version": TOKENIZER_VERSION,
        "typed_certificate_reconstruction": {
            "molecules_exact": typed_key_exact,
            "molecule_fraction": typed_key_exact / max(len(recs), 1),
        },
        "parent_certificate_reconstruction": {
            "molecules_exact": parent_key_exact,
            "molecule_fraction": parent_key_exact / max(len(recs), 1),
        },
        "typed_token_id_match": {
            "molecules_exact": typed_id_exact,
            "molecule_fraction": typed_id_exact / max(len(recs), 1),
        },
        "parent_token_id_match": {
            "molecules_exact": parent_id_exact,
            "molecule_fraction": parent_id_exact / max(len(recs), 1),
        },
        "class": "EXPLICIT_BINDING_LOOKUP (reconstructible, non-injective alias)",
        "official_test_loaded": False,
    }


def token_alias_audit(records: Sequence[Any], limit: int | None) -> dict[str, Any]:
    """Quantify the historical certificate's non-injectivity on the audit split."""
    recs = _subset(records, limit)
    buckets: dict[bytes, set[int]] = {}
    occurrences: dict[bytes, int] = {}
    total = 0
    for record in recs:
        for patch in record.patches:
            root_atom = int(np.argmax(patch.shell_descriptor[108:136]))
            buckets.setdefault(patch.typed_certificate, set()).add(root_atom)
            occurrences[patch.typed_certificate] = occurrences.get(patch.typed_certificate, 0) + 1
            total += 1
    aliased = sum(1 for key, atoms in buckets.items() if len(atoms) > 1)
    shared_occurrences = sum(count for key, count in occurrences.items() if len(buckets[key]) > 1)
    return {
        "n_patches": int(total),
        "n_unique_tokens": int(len(buckets)),
        "tokens_with_multiple_root_atoms": int(aliased),
        "token_fraction_aliased": aliased / max(len(buckets), 1),
        "occurrences_in_aliased_tokens": int(shared_occurrences),
        "occurrence_fraction_aliased": shared_occurrences / max(total, 1),
        "note": "historical pynauty certificate omits color labels -> distinct rooted typed bindings can share one token id",
        "official_test_loaded": False,
    }


def forward_equivalence(
    model: StrictStaticPairModel,
    raw_dataset: Sequence[Any],
    records: Sequence[Any],
    encoded: Sequence[Data],
    factorized: Sequence[FactorizedRawRecord],
    transform: FactorizedFeatureTransform,
    limit: int | None,
) -> dict[str, Any]:
    raws = _subset(raw_dataset, limit)
    recs = _subset(records, limit)
    encs = _subset(encoded, limit)
    facts = _subset(factorized, limit)
    layer_names = ["patch", "pair_value", "relation", "graph_hidden", "R", "pred"]
    accum: dict[str, list[dict[str, float]]] = {name: [] for name in layer_names}
    input_stats = {"patch_cont": [], "pair_relation": [], "global_context": [], "topology_features": []}
    hist_preds: list[float] = []
    fact_preds: list[float] = []
    targets: list[float] = []
    for raw, record, enc, fact in zip(raws, recs, encs, facts):
        y = float(raw.y.view(-1)[0])
        topology_raw = None if record.topology_features is None else record.topology_features
        fact_data = transform.build(raw, y, topology_raw=topology_raw, raw=fact)
        input_stats["patch_cont"].append(
            _tensor_stats(enc.patch_cont, fact_data.patch_cont, TOL_STANDARDIZED)
        )
        input_stats["pair_relation"].append(
            _tensor_stats(enc.pair_relation, fact_data.pair_relation, TOL_RAW)
        )
        input_stats["global_context"].append(
            _tensor_stats(enc.global_context, fact_data.global_context, TOL_STANDARDIZED)
        )
        input_stats["topology_features"].append(
            _tensor_stats(enc.topology_features, fact_data.topology_features, TOL_STANDARDIZED)
        )
        hist_caps = _capture(model, enc)
        fact_caps = _capture(model, fact_data)
        for name in layer_names:
            tol = TOL_PREDICTION if name == "pred" else TOL_INTERMEDIATE
            accum[name].append(_tensor_stats(hist_caps[name], fact_caps[name], tol))
        hist_preds.append(float(hist_caps["pred"].view(-1)[0]))
        fact_preds.append(float(fact_caps["pred"].view(-1)[0]))
        targets.append(y)
    hist_pred = np.asarray(hist_preds, dtype=np.float64)
    fact_pred = np.asarray(fact_preds, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    return {
        "protocol_version": "fec_s0_v1",
        "n_molecules": int(len(raws)),
        "inputs": {name: _aggregate(stats) for name, stats in input_stats.items()},
        "layers": {name: _aggregate(stats) for name, stats in accum.items()},
        "predictions": {
            "historical_mae": float(np.mean(np.abs(target - hist_pred))),
            "factorized_mae": float(np.mean(np.abs(target - fact_pred))),
            "recorded_checkpoint_best_valid_mae": 0.14567435123870381,
            "historical_vs_factorized": _tensor_stats(
                torch.from_numpy(hist_pred), torch.from_numpy(fact_pred), TOL_PREDICTION
            ),
        },
        "official_test_loaded": False,
    }


def _aggregate(stats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _max(key: str) -> float:
        values = [float(s[key]) for s in stats if s.get(key) is not None]
        return max(values) if values else 0.0

    def _mean(key: str) -> float:
        values = [float(s[key]) for s in stats if s.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    return {
        "max_abs": _max("max_abs"),
        "mean_abs": _mean("mean_abs"),
        "allclose_fraction": float(np.mean([bool(s["allclose"]) for s in stats])) if stats else 1.0,
        "bit_equal_fraction": _mean("bit_equal_fraction"),
    }


# ---------------------------------------------------------------------------
# purity contract
# ---------------------------------------------------------------------------


def _relabel_raw(raw: Any, seed: int = 20260924) -> Any:
    n = int(raw.num_nodes)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(n, generator=generator)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(n)
    relabeled = raw.clone()
    relabeled.x = raw.x[permutation]
    relabeled.edge_index = inverse[raw.edge_index]
    if raw.edge_attr is not None:
        relabeled.edge_attr = raw.edge_attr.clone()
    return relabeled


def purity_contract(
    model: StrictStaticPairModel,
    raw_dataset: Sequence[Any],
    records: Sequence[Any],
    factorized: Sequence[FactorizedRawRecord],
    transform: FactorizedFeatureTransform,
    limit: int | None,
) -> dict[str, Any]:
    raws = _subset(raw_dataset, limit)
    recs = _subset(records, limit)
    facts = _subset(factorized, limit)

    factorized_data = []
    for raw, record, fact in zip(raws, recs, facts):
        topology_raw = None if record.topology_features is None else record.topology_features
        factorized_data.append(
            transform.build(raw, float(raw.y.view(-1)[0]), topology_raw=topology_raw, raw=fact)
        )
    batch = Batch.from_data_list(factorized_data)

    # (A) pair -> centre never called; (B) h invariant under pair mutation;
    # (C) pair/relation encoders exactly once per forward.
    from tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair import (
        static_contract_checks,
    )

    contract = static_contract_checks(model, batch)

    # (D) environment formation strictly before pair composition (call order).
    order: list[str] = []
    handles = [
        model.patch_encoder.register_forward_pre_hook(lambda *_a: order.append("patch")),
        model.pair_projection.register_forward_pre_hook(lambda *_a: order.append("pair")),
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    environment_before_pair = bool(order and order[0] == "patch" and "pair" in order)

    # (E) no recurrence: encoders called exactly once per forward.
    counts = {"pair_encoder": 0, "relation_encoder": 0}
    handles = [
        model.pair_encoder.register_forward_hook(
            lambda *_a: counts.__setitem__("pair_encoder", counts["pair_encoder"] + 1)
        ),
        model.relation_encoder.register_forward_hook(
            lambda *_a: counts.__setitem__("relation_encoder", counts["relation_encoder"] + 1)
        ),
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    no_recurrence = counts["pair_encoder"] == 1 and counts["relation_encoder"] == 1

    # (F) relabel invariance of the factorized builder + model.
    relabel_diffs: list[float] = []
    for raw, record, fact in zip(raws[: min(len(raws), 8)], recs[:8], facts[:8]):
        y = float(raw.y.view(-1)[0])
        topology_raw = None if record.topology_features is None else record.topology_features
        base = transform.build(raw, y, topology_raw=topology_raw, raw=fact)
        relabeled_raw = _relabel_raw(raw)
        relabeled_fact = factorized_record(relabeled_raw, tokenize=True)
        relabeled = transform.build(relabeled_raw, y, topology_raw=topology_raw, raw=relabeled_fact)
        with torch.no_grad():
            p0 = model(base).view(-1)
            p1 = model(relabeled).view(-1)
        relabel_diffs.append(float((p0 - p1).abs().max().item()))
    relabel_max = max(relabel_diffs) if relabel_diffs else 0.0

    return {
        "protocol_version": "fec_s0_v1",
        "static_contract": contract,
        "environment_before_pair": environment_before_pair,
        "call_order_head": order[:6],
        "no_recurrence": no_recurrence,
        "encoder_calls_per_forward": counts,
        "relabel_invariance_max_abs_pred_diff": relabel_max,
        "relabel_invariance_n_molecules": len(relabel_diffs),
        "passed": bool(
            contract["passed"] and environment_before_pair and no_recurrence and relabel_max <= TOL_PREDICTION
        ),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_audit(
    train_limit: int | None = 256,
    valid_limit: int | None = 256,
    write: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)

    print("[fec-s0] loading historical records / encoded inputs", flush=True)
    train_records, valid_records = _extract_v4_records()
    train_encoded, valid_encoded, audit = load_encoded()
    if int(audit["typed_vocabulary_size_with_oov"]) != 6785:
        raise RuntimeError(f"typed vocabulary drift: {audit['typed_vocabulary_size_with_oov']}")
    if int(audit["parent_vocabulary_size_with_oov"]) != 32:
        raise RuntimeError("parent vocabulary drift")
    train_raw = _load_zinc(ZINC_ROOT, "train")
    valid_raw = _load_zinc(ZINC_ROOT, "val")
    print(
        f"[fec-s0] records train={len(train_records)} valid={len(valid_records)}",
        flush=True,
    )

    fits = build_fits(train_records)
    transform = FactorizedFeatureTransform(fits)
    model = load_s0_model()

    # Full-train standardizer/vocabulary reproduction check (cheap, no pynauty).
    hist_train_patch = zpp._patch_matrix(train_records)
    hist_train_global = zpp._context_matrix(train_records)
    cache_train_patch = torch.cat([d.patch_cont for d in train_encoded], dim=0).numpy()
    cache_train_global = torch.cat([d.global_context for d in train_encoded], dim=0).numpy()
    hist_typed_ids = np.concatenate(
        [
            np.asarray(
                [fits.typed_vocabulary.get(p.typed_certificate, 0) for p in r.patches],
                dtype=np.int64,
            )
            for r in train_records
        ]
    )
    cache_typed_ids = torch.cat([d.typed_token for d in train_encoded]).numpy()
    hist_parent_ids = np.concatenate(
        [
            np.asarray(
                [fits.parent_vocabulary.get(p.parent_certificate, 0) for p in r.patches],
                dtype=np.int64,
            )
            for r in train_records
        ]
    )
    cache_parent_ids = torch.cat([d.parent_token for d in train_encoded]).numpy()
    recon = {
        "patch_standardizer_reproduces_cache": _array_stats(
            cache_train_patch,
            fits.patch_standardizer.transform(hist_train_patch),
            TOL_STANDARDIZED,
        ),
        "global_standardizer_reproduces_cache": _array_stats(
            cache_train_global,
            fits.context_standardizer.transform(hist_train_global),
            TOL_STANDARDIZED,
        ),
        "typed_vocabulary_size": len(fits.typed_vocabulary) + 1,
        "parent_vocabulary_size": len(fits.parent_vocabulary) + 1,
        "typed_token_id_match_fraction": float(np.mean(hist_typed_ids == cache_typed_ids)),
        "parent_token_id_match_fraction": float(np.mean(hist_parent_ids == cache_parent_ids)),
    }
    del hist_train_patch, hist_train_global

    print("[fec-s0] building factorized train/valid records", flush=True)
    train_factorized = build_factorized_records(train_raw, train_limit)
    valid_factorized = build_factorized_records(valid_raw, valid_limit)

    # First molecule for access audit.
    first_record = valid_records[0]
    first_fact = valid_factorized[0]
    access_data = transform.build(
        valid_raw[0],
        float(valid_raw[0].y.view(-1)[0]),
        topology_raw=first_record.topology_features,
        raw=first_fact,
    )
    access = access_audit(model, access_data)

    local = local_descriptor_equivalence(train_records, train_encoded, train_factorized, fits, train_limit)
    local_valid = local_descriptor_equivalence(
        valid_records, valid_encoded, valid_factorized, fits, valid_limit
    )
    pair = pair_relation_equivalence(train_records, train_factorized, train_limit)
    pair_valid = pair_relation_equivalence(valid_records, valid_factorized, valid_limit)
    glob = global_equivalence(train_records, train_encoded, train_factorized, fits, train_limit)
    glob_valid = global_equivalence(valid_records, valid_encoded, valid_factorized, fits, valid_limit)
    tokens = token_equivalence(valid_records, valid_encoded, valid_factorized, fits, valid_limit)
    aliasing = token_alias_audit(valid_records, valid_limit)
    forward = forward_equivalence(
        model, valid_raw, valid_records, valid_encoded, valid_factorized, transform, valid_limit
    )
    purity = purity_contract(model, valid_raw, valid_records, valid_factorized, transform, valid_limit)

    classification = _classify(access)
    payload = {
        "protocol_version": "fec_s0_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "limits": {"train": train_limit, "valid": valid_limit},
        "reproduction": recon,
        "classification": classification,
        "stages": {
            "access_audit": access,
            "local_descriptor_equivalence_train": local,
            "local_descriptor_equivalence_valid": local_valid,
            "pair_relation_equivalence_train": pair,
            "pair_relation_equivalence_valid": pair_valid,
            "global_equivalence_train": glob,
            "global_equivalence_valid": glob_valid,
            "token_equivalence_valid": tokens,
            "token_alias_audit_valid": aliasing,
            "forward_equivalence_valid": forward,
            "purity_contract": purity,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    if write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(RESULTS_DIR / "access_audit.json", access)
        _write_json(
            RESULTS_DIR / "local_descriptor_equivalence.json",
            {"train": local, "valid": local_valid},
        )
        _write_json(
            RESULTS_DIR / "pair_relation_equivalence.json",
            {"train": pair, "valid": pair_valid},
        )
        _write_json(
            RESULTS_DIR / "global_equivalence.json",
            {"train": glob, "valid": glob_valid},
        )
        _write_json(
            RESULTS_DIR / "intermediate_equivalence.json",
            {"forward_equivalence_valid": forward, "layers": forward["layers"]},
        )
        _write_json(
            RESULTS_DIR / "prediction_equivalence.json",
            forward["predictions"],
        )
        _write_json(RESULTS_DIR / "purity_contract.json", purity)
        _write_json(RESULTS_DIR / "token_equivalence.json", tokens)
        _write_json(RESULTS_DIR / "token_alias_audit.json", aliasing)
        _write_json(RESULTS_DIR / "full_audit.json", payload)
    return payload


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _classify(access: Mapping[str, Any]) -> dict[str, Any]:
    table = access["input_path_table"]
    local_blockers = [
        key
        for key, entry in table.items()
        if entry.get("classification") == "EXPLICIT_BINDING_LOOKUP" and entry.get("reaches_prediction")
    ]
    factorized = [key for key, entry in table.items() if entry.get("classification") == "FACTORIZED_SHARED"]
    pure_topology = [key for key, entry in table.items() if entry.get("classification") == "PURE_TOPOLOGY"]
    if local_blockers:
        verdict = "FEC_S0_LOCAL_FACTORIZATION_BLOCKED"
    else:
        verdict = "FEC_S0_FUNCTIONALLY_EQUIVALENT"
    return {
        "factorized_shared_paths": factorized,
        "pure_topology_paths": pure_topology,
        "local_blocking_paths": local_blockers,
        "verdict": verdict,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FEC-S0 factorized equivalence audit")
    parser.add_argument("stage", nargs="?", default="all", choices=["all"])
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--valid-limit", type=int, default=256)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    payload = run_audit(
        train_limit=args.train_limit,
        valid_limit=args.valid_limit,
        write=not args.no_write,
    )
    print(json.dumps(_jsonable(payload["classification"]), indent=2))
    print("forward layers:")
    for name, stats in payload["stages"]["forward_equivalence_valid"]["layers"].items():
        print(f"  {name:12s} max_abs={stats['max_abs']:.3e} allclose={stats['allclose_fraction']:.3f}")
    print("purity passed:", payload["stages"]["purity_contract"]["passed"])
    print(f"elapsed={payload['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
