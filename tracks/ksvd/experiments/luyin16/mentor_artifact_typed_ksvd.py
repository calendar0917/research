"""Artifact-aligned MolHIV typed-patch + K-SVD feature reproduction.

The recovered mentor artifacts describe the following feature blocks:

``composition[69] + recon_typed[624] + context_mass[5*K+5]``.

The original upstream builders and payload are missing.  This runner therefore
reconstructs the documented mechanism with an explicit, invariant schema:

* every atom is the root of its complete radius-2 induced ego graph;
* each root is represented by a 208-D typed descriptor containing compact OGB
  atom/bond semantics and topology summaries;
* a shared train-only K-SVD dictionary encodes those descriptors;
* raw/INIT/FINAL descriptor populations are pooled with mean/std/max, yielding
  exactly 624 dimensions;
* five root-level ring contexts produce ``5*K+5`` activation-mass features.
* the optional ``cross_cov`` block is the centre-population covariance between
  FINAL atom-use coordinates and the five ring-context indicators.

Unlike the historical 52-D proxy, no coordinate depends on a node-id tie break,
no radius-2 ego is truncated to eight nodes, and no atom category is reduced by
modulo.  The result is still a documented proxy rather than the unavailable
mentor payload, and the manifest says so explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.cluster import kmeans_plusplus
from sklearn.linear_model import orthogonal_mp

from ksvd_research.data import MolhivBundle, load_molhiv
from ksvd_research.features import build_ring_context_index
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    ATOM_DIM,
    BOND_DIM,
    compact_atom_semantics,
    compact_bond_semantics,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
PATCH_DIM = 208
TYPED_POOL_DIM = 3 * PATCH_DIM
SELECTED_CONTEXT_NAMES = (
    "ring5",
    "ring6",
    "aromatic_ring",
    "multi_ring",
    "ring_boundary",
)
COMPOSITION_MACRO_NAMES = (
    "log1p_num_nodes",
    "log1p_num_edges",
    "density",
    "degree_mean",
    "degree_std",
    "degree_max",
    "cycle_rank_per_node",
    "triangle_count_per_node",
)
COMPOSITION_DIM = ATOM_DIM + BOND_DIM + len(COMPOSITION_MACRO_NAMES)

VIEW_BLOCKS: dict[str, tuple[str, ...]] = {
    "s": ("composition",),
    "t_raw": ("typed_raw",),
    "t_init": ("typed_init",),
    "t_final": ("typed_final",),
    "a_init": ("context_init",),
    "a_final": ("context_final",),
    "st_raw": ("composition", "typed_raw"),
    "st_init": ("composition", "typed_init"),
    "st_final": ("composition", "typed_final"),
    "sta_init": ("composition", "typed_init", "context_init"),
    "sta_final": ("composition", "typed_final", "context_final"),
    "sta_cross_cov": (
        "composition",
        "typed_final",
        "context_final",
        "cross_cov_final",
    ),
    "st_raw_final": ("composition", "typed_raw", "typed_final"),
    "sta_raw_final": (
        "composition",
        "typed_raw",
        "typed_final",
        "context_final",
    ),
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    for key in ("protocol_id", "data", "patch", "dictionary"):
        if key not in payload:
            raise KeyError(f"missing configuration key {key!r}")
    patch = payload["patch"]
    if int(patch.get("radius", 2)) != 2:
        raise ValueError("the artifact-aligned descriptor is fixed to radius=2")
    dictionary = payload["dictionary"]
    if int(dictionary["n_atoms"]) < 2:
        raise ValueError("dictionary.n_atoms must be at least two")
    if int(dictionary["sparsity"]) < 1:
        raise ValueError("dictionary.sparsity must be positive")
    return payload


def _ego_shells(graph: Any, center: int) -> tuple[list[int], list[int]]:
    center = int(center)
    shell1 = sorted(int(node) for node in graph.neighbors(center))
    shell1_set = set(shell1)
    shell2: set[int] = set()
    for node in shell1:
        shell2.update(int(neighbor) for neighbor in graph.neighbors(node))
    shell2.discard(center)
    shell2.difference_update(shell1_set)
    return shell1, sorted(shell2)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def typed_radius2_descriptor(
    graph: Any,
    center: int,
    atom_semantics: np.ndarray,
    bond_semantics: Mapping[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Return the invariant 208-D descriptor for one complete radius-2 ego."""
    center = int(center)
    shell1, shell2 = _ego_shells(graph, center)
    nodes = [center, *shell1, *shell2]
    node_set = set(nodes)
    shell_of = {center: 0}
    shell_of.update({node: 1 for node in shell1})
    shell_of.update({node: 2 for node in shell2})

    root_atom = np.asarray(atom_semantics[center], dtype=np.float32)
    shell1_atom = (
        np.mean(atom_semantics[np.asarray(shell1, dtype=np.int64)], axis=0, dtype=np.float32)
        if shell1
        else np.zeros(ATOM_DIM, dtype=np.float32)
    )
    shell2_atom = (
        np.mean(atom_semantics[np.asarray(shell2, dtype=np.int64)], axis=0, dtype=np.float32)
        if shell2
        else np.zeros(ATOM_DIM, dtype=np.float32)
    )

    bond_sums = np.zeros((4, BOND_DIM), dtype=np.float32)
    bond_counts = np.zeros(4, dtype=np.int64)
    induced_edges: list[tuple[int, int]] = []
    for left in nodes:
        for right in graph.neighbors(left):
            right = int(right)
            if right not in node_set or left >= right:
                continue
            induced_edges.append((int(left), right))
            pair = tuple(sorted((shell_of[int(left)], shell_of[right])))
            category = {
                (0, 1): 0,
                (1, 1): 1,
                (1, 2): 2,
                (2, 2): 3,
            }.get(pair)
            if category is not None:
                bond_sums[category] += bond_semantics[graph.edge_key(int(left), right)]
                bond_counts[category] += 1
    bond_means = np.zeros_like(bond_sums)
    active = bond_counts > 0
    bond_means[active] = bond_sums[active] / bond_counts[active, None]

    valid_count = len(nodes)
    edge_count = len(induced_edges)
    possible_edges = valid_count * (valid_count - 1) / 2
    root_degree = sum(1 for node in shell1 if graph.has_edge(center, node))
    shell1_internal = int(bond_counts[1])
    shell12 = int(bond_counts[2])
    shell2_internal = int(bond_counts[3])
    shell1_possible = len(shell1) * (len(shell1) - 1) / 2
    shell2_possible = len(shell2) * (len(shell2) - 1) / 2
    shell12_possible = len(shell1) * len(shell2)
    cycle_rank = max(edge_count - valid_count + 1, 0)

    triangle_edges = 0
    for left, right in induced_edges:
        triangle_edges += len(
            graph.neighbors(left).intersection(graph.neighbors(right)).intersection(node_set)
        )
    triangles = triangle_edges / 3.0

    topology = np.asarray(
        [
            math.log1p(valid_count),
            _safe_ratio(edge_count, possible_edges),
            _safe_ratio(root_degree, valid_count - 1),
            _safe_ratio(len(shell1), valid_count - 1),
            _safe_ratio(len(shell2), valid_count - 1),
            _safe_ratio(shell1_internal, shell1_possible),
            _safe_ratio(shell12, shell12_possible),
            _safe_ratio(shell2_internal, shell2_possible),
            _safe_ratio(cycle_rank, valid_count),
            _safe_ratio(triangles, valid_count),
        ],
        dtype=np.float32,
    )
    shell_sizes = np.asarray([math.log1p(len(shell1)), math.log1p(len(shell2))], dtype=np.float32)
    descriptor = np.concatenate(
        [
            root_atom,
            shell1_atom,
            shell2_atom,
            bond_means.reshape(-1),
            topology,
            shell_sizes,
        ]
    ).astype(np.float32, copy=False)
    if descriptor.shape != (PATCH_DIM,) or not np.all(np.isfinite(descriptor)):
        raise RuntimeError(f"invalid typed descriptor shape/value: {descriptor.shape}")
    return descriptor


def pool_typed_descriptors(descriptors: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[1] != PATCH_DIM:
        raise ValueError(f"expected [n,{PATCH_DIM}], got {descriptors.shape}")
    if not descriptors.shape[0]:
        return np.zeros(TYPED_POOL_DIM, dtype=np.float32)
    pooled = np.concatenate(
        [
            descriptors.mean(axis=0),
            descriptors.std(axis=0),
            descriptors.max(axis=0),
        ]
    ).astype(np.float32, copy=False)
    if pooled.shape != (TYPED_POOL_DIM,):
        raise RuntimeError(f"typed pool shape mismatch: {pooled.shape}")
    return pooled


def _triangle_count(graph: Any) -> int:
    edge_memberships = 0
    for left, right in graph.edges():
        edge_memberships += len(graph.neighbors(left).intersection(graph.neighbors(right)))
    return edge_memberships // 3


def composition_69(
    graph: Any,
    atom_semantics: np.ndarray,
    bond_semantics: Mapping[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Return 48 atom + 13 bond + 8 topology coordinates."""
    atom_mean = (
        atom_semantics.mean(axis=0, dtype=np.float32)
        if atom_semantics.shape[0]
        else np.zeros(ATOM_DIM, dtype=np.float32)
    )
    bond_rows = list(bond_semantics.values())
    bond_mean = (
        np.mean(np.stack(bond_rows, axis=0), axis=0, dtype=np.float32)
        if bond_rows
        else np.zeros(BOND_DIM, dtype=np.float32)
    )
    degrees = np.asarray([len(graph.neighbors(node)) for node in graph.nodes], dtype=np.float32)
    n_nodes = int(graph.n)
    n_edges = int(graph.num_edges())
    density = _safe_ratio(2 * n_edges, n_nodes * (n_nodes - 1))
    macro = np.asarray(
        [
            math.log1p(n_nodes),
            math.log1p(n_edges),
            density,
            float(degrees.mean()) if degrees.size else 0.0,
            float(degrees.std()) if degrees.size else 0.0,
            float(degrees.max()) if degrees.size else 0.0,
            _safe_ratio(max(n_edges - n_nodes + 1, 0), n_nodes),
            _safe_ratio(_triangle_count(graph), n_nodes),
        ],
        dtype=np.float32,
    )
    row = np.concatenate([atom_mean, bond_mean, macro]).astype(np.float32, copy=False)
    if row.shape != (COMPOSITION_DIM,) or not np.all(np.isfinite(row)):
        raise RuntimeError(f"invalid composition row: {row.shape}")
    return row


def _root_context_matrix(graph: Any, node_features: np.ndarray) -> np.ndarray:
    index = build_ring_context_index(graph, node_features)
    ring_nodes = set(index.ring_nodes)
    boundary_nodes: set[int] = set()
    for left, right in index.boundary_edges:
        if left in ring_nodes and right not in ring_nodes:
            boundary_nodes.add(int(right))
        elif right in ring_nodes and left not in ring_nodes:
            boundary_nodes.add(int(left))
    rows = np.zeros((graph.n, len(SELECTED_CONTEXT_NAMES)), dtype=bool)
    for node in graph.nodes:
        rows[int(node)] = (
            node in index.ring5_nodes,
            node in index.ring6_nodes,
            node in index.aromatic_ring_nodes,
            node in index.multi_ring_nodes,
            node in boundary_nodes,
        )
    return rows


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(values, axis=0), 1e-12)
    return values / norms[None, :]


def omp_encode(dictionary: np.ndarray, samples: np.ndarray, sparsity: int) -> np.ndarray:
    dictionary64 = np.asarray(dictionary, dtype=np.float64, order="F")
    samples64 = np.asarray(samples, dtype=np.float64, order="F")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        codes = orthogonal_mp(
            dictionary64,
            samples64,
            n_nonzero_coefs=min(int(sparsity), dictionary64.shape[1]),
            tol=None,
            precompute=True,
            copy_X=True,
            return_path=False,
        )
    codes = np.asarray(codes, dtype=np.float32)
    if codes.ndim == 1:
        codes = codes[:, None]
    return codes


def _leading_rank1(
    residual: np.ndarray,
    power_iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray]:
    if residual.shape[1] == 1:
        column = residual[:, 0]
        sigma = float(np.linalg.norm(column))
        if sigma <= 1e-12:
            return np.zeros_like(column), 0.0, np.ones(1, dtype=np.float32)
        return column / sigma, sigma, np.ones(1, dtype=np.float32)
    right = rng.standard_normal(residual.shape[1]).astype(np.float32)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    left = np.zeros(residual.shape[0], dtype=np.float32)
    for _ in range(max(1, int(power_iterations))):
        left = residual @ right
        left_norm = float(np.linalg.norm(left))
        if left_norm <= 1e-12:
            return left, 0.0, right
        left /= left_norm
        right = residual.T @ left
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1e-12:
            return left, 0.0, right
        right /= right_norm
    sigma = float(left @ (residual @ right))
    if sigma < 0:
        left = -left
        sigma = -sigma
    return left.astype(np.float32), sigma, right.astype(np.float32)


def _ksvd_update(
    samples: np.ndarray,
    dictionary: np.ndarray,
    codes: np.ndarray,
    power_iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(samples, dtype=np.float32)
    learned = np.asarray(dictionary, dtype=np.float32).copy()
    coefficients = np.asarray(codes, dtype=np.float32).copy()
    reconstruction = learned @ coefficients
    revived = 0
    for atom in rng.permutation(learned.shape[1]):
        active = np.flatnonzero(np.abs(coefficients[atom]) > 1e-10)
        if active.size == 0:
            residual = values - reconstruction
            sample = int(np.argmax(np.sum(np.square(residual), axis=0)))
            candidate = residual[:, sample]
            norm = float(np.linalg.norm(candidate))
            if norm > 1e-12:
                learned[:, atom] = candidate / norm
                coefficients[atom] = 0.0
                coefficients[atom, sample] = norm
                reconstruction[:, sample] += learned[:, atom] * norm
                revived += 1
            continue
        old_atom = learned[:, atom].copy()
        old_coefficients = coefficients[atom, active].copy()
        restricted = (
            values[:, active] - reconstruction[:, active] + old_atom[:, None] * old_coefficients[None, :]
        )
        left, sigma, right = _leading_rank1(restricted, power_iterations, rng)
        if sigma <= 1e-10 or float(np.linalg.norm(left)) <= 1e-10:
            continue
        learned[:, atom] = left
        coefficients[atom, active] = sigma * right
        reconstruction[:, active] = (
            values[:, active] - restricted + learned[:, atom, None] * coefficients[atom, active][None, :]
        )
    return _normalize_columns(learned), coefficients, revived


def fit_ksvd(
    training_rows: np.ndarray,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    power_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit train-only coordinate scaling, INIT and FINAL dictionaries."""
    rows = np.asarray(training_rows, dtype=np.float32)
    scale = np.sqrt(np.mean(np.square(rows), axis=0, dtype=np.float64)).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-4))
    scaled = rows / scale[None, :]
    scaled_norms = np.maximum(np.linalg.norm(scaled, axis=1), 1e-12)
    samples = (scaled / scaled_norms[:, None]).T.astype(np.float32)

    _, indices = kmeans_plusplus(
        samples.T,
        n_clusters=int(n_atoms),
        random_state=int(seed),
    )
    initial = _normalize_columns(samples[:, np.asarray(indices, dtype=np.int64)])
    final = initial.copy()
    rng = np.random.default_rng(int(seed) + 1009)

    initial_codes = omp_encode(initial, samples, sparsity)
    initial_error = float(
        np.linalg.norm(samples - initial @ initial_codes) / max(float(np.linalg.norm(samples)), 1e-12)
    )
    curve: list[float] = []
    revived_total = 0
    for iteration in range(int(iterations)):
        codes = omp_encode(final, samples, sparsity)
        final, _, revived = _ksvd_update(
            samples,
            final,
            codes,
            power_iterations=power_iterations,
            rng=rng,
        )
        recoded = omp_encode(final, samples, sparsity)
        error = float(np.linalg.norm(samples - final @ recoded) / max(float(np.linalg.norm(samples)), 1e-12))
        curve.append(error)
        revived_total += int(revived)
        print(
            f"  K-SVD iteration {iteration + 1}/{iterations}: "
            f"relative reconstruction={error:.6f}, revived={revived}",
            flush=True,
        )
    info = {
        "training_patch_count": int(rows.shape[0]),
        "patch_dimension": int(rows.shape[1]),
        "n_atoms": int(n_atoms),
        "sparsity": int(sparsity),
        "iterations": int(iterations),
        "initial_relative_reconstruction": initial_error,
        "final_relative_reconstruction": curve[-1] if curve else initial_error,
        "reconstruction_curve": curve,
        "revived_atoms_total": revived_total,
        "initialization": "sklearn kmeans++ indices over real train patches",
        "coordinate_scaling": "train-only coordinate RMS, then per-patch L2 normalization",
    }
    return initial, final, scale, info


def _sample_train_rows(
    offsets: np.ndarray,
    train_graphs: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    train_graphs = np.asarray(train_graphs, dtype=np.int64)
    selected: list[int] = []
    round_id = 0
    while len(selected) < int(maximum):
        progressed = False
        for graph_position in rng.permutation(train_graphs):
            start = int(offsets[int(graph_position)])
            end = int(offsets[int(graph_position) + 1])
            if start >= end:
                continue
            local = int(rng.integers(start, end))
            selected.append(local)
            progressed = True
            if len(selected) >= int(maximum):
                break
        if not progressed:
            break
        round_id += 1
        if round_id > 1024:
            raise RuntimeError("train patch sampler failed to terminate")
    return np.asarray(selected, dtype=np.int64)


def _clip_reconstructed_descriptors(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    clipped[:, : 3 * ATOM_DIM + 4 * BOND_DIM] = np.clip(clipped[:, : 3 * ATOM_DIM + 4 * BOND_DIM], 0.0, 1.0)
    # density/ratio coordinates, excluding log1p(size) and triangle-per-node.
    for coordinate in (197, 198, 199, 200, 201, 202, 203, 204):
        clipped[:, coordinate] = np.clip(clipped[:, coordinate], 0.0, 1.0)
    return clipped


def _context_mass(atom_use: np.ndarray, masks: np.ndarray) -> np.ndarray:
    atom_use = np.asarray(atom_use, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    if atom_use.shape[0] != masks.shape[0]:
        raise ValueError(f"atom/context mismatch: {atom_use.shape}/{masks.shape}")
    blocks: list[np.ndarray] = []
    coverage: list[float] = []
    for context in range(masks.shape[1]):
        selected = masks[:, context]
        blocks.append(
            atom_use[selected].sum(axis=0) / max(int(atom_use.shape[0]), 1)
            if np.any(selected)
            else np.zeros(atom_use.shape[1], dtype=np.float32)
        )
        coverage.append(float(selected.mean()) if selected.size else 0.0)
    return np.concatenate([*blocks, np.asarray(coverage, dtype=np.float32)])


def _context_cross_cov(atom_use: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return Cov_v(atom-use_v, context_v) with population normalization."""
    values = np.asarray(atom_use, dtype=np.float32)
    indicators = np.asarray(masks, dtype=np.float32)
    if values.ndim != 2 or indicators.ndim != 2 or values.shape[0] != indicators.shape[0]:
        raise ValueError(f"atom/context mismatch: {values.shape}/{indicators.shape}")
    n_rows = values.shape[0]
    if n_rows <= 1:
        return np.zeros(values.shape[1] * indicators.shape[1], dtype=np.float32)
    values_centered = values - values.mean(axis=0, keepdims=True)
    indicators_centered = indicators - indicators.mean(axis=0, keepdims=True)
    return ((values_centered.T @ indicators_centered) / float(n_rows)).reshape(-1).astype(
        np.float32, copy=False
    )


def dictionary_graph_readouts(
    descriptors: np.ndarray,
    context_masks: np.ndarray,
    offsets: np.ndarray,
    dictionary: np.ndarray,
    scale: np.ndarray,
    sparsity: int,
    patch_batch_size: int,
    *,
    return_cross_cov: bool = False,
) -> tuple[np.ndarray, ...]:
    """Encode graph batches and return typed/context readouts.

    The historical three-item return is retained unless ``return_cross_cov``
    is requested.  The extended return appends the flattened centre covariance
    block and is used by the artifact-aligned full build.
    """
    n_graphs = len(offsets) - 1
    n_atoms = int(dictionary.shape[1])
    typed = np.zeros((n_graphs, TYPED_POOL_DIM), dtype=np.float32)
    context = np.zeros(
        (n_graphs, len(SELECTED_CONTEXT_NAMES) * n_atoms + len(SELECTED_CONTEXT_NAMES)),
        dtype=np.float32,
    )
    cross_cov = np.zeros(
        (n_graphs, n_atoms * len(SELECTED_CONTEXT_NAMES)), dtype=np.float32
    )
    residual_energy = 0.0
    input_energy = 0.0
    graph_start = 0
    while graph_start < n_graphs:
        graph_end = graph_start + 1
        row_start = int(offsets[graph_start])
        while graph_end < n_graphs:
            candidate_end = int(offsets[graph_end + 1])
            if candidate_end - row_start > int(patch_batch_size) and graph_end > graph_start:
                break
            graph_end += 1
            if candidate_end - row_start >= int(patch_batch_size):
                break
        row_end = int(offsets[graph_end])
        raw = np.asarray(descriptors[row_start:row_end], dtype=np.float32)
        scaled = raw / scale[None, :]
        norms = np.maximum(np.linalg.norm(scaled, axis=1), 1e-12)
        encoded = (scaled / norms[:, None]).T.astype(np.float32)
        codes = omp_encode(dictionary, encoded, sparsity)
        normalized_reconstruction = (dictionary @ codes).T
        residual_energy += float(np.square(encoded.T - normalized_reconstruction).sum(dtype=np.float64))
        input_energy += float(np.square(encoded).sum(dtype=np.float64))
        reconstructed = normalized_reconstruction * norms[:, None] * scale[None, :]
        reconstructed = _clip_reconstructed_descriptors(reconstructed)
        atom_use = np.abs(codes.T)
        atom_denominator = atom_use.sum(axis=1, keepdims=True)
        np.divide(
            atom_use,
            atom_denominator,
            out=atom_use,
            where=atom_denominator > 1e-12,
        )

        for graph_position in range(graph_start, graph_end):
            local_start = int(offsets[graph_position]) - row_start
            local_end = int(offsets[graph_position + 1]) - row_start
            local_atom_use = atom_use[local_start:local_end]
            local_masks = context_masks[row_start + local_start : row_start + local_end]
            typed[graph_position] = pool_typed_descriptors(reconstructed[local_start:local_end])
            context[graph_position] = _context_mass(
                local_atom_use,
                local_masks,
            )
            cross_cov[graph_position] = _context_cross_cov(local_atom_use, local_masks)
        graph_start = graph_end
        if graph_start % 2000 == 0 or graph_start == n_graphs:
            print(f"  encoded {graph_start}/{n_graphs} graphs", flush=True)
    encoding = {
        "normalized_patch_relative_reconstruction": math.sqrt(
            residual_energy / max(input_energy, 1e-12)
        )
    }
    if return_cross_cov:
        return typed, context, cross_cov, encoding
    return typed, context, encoding


def assemble_view(payload: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in VIEW_BLOCKS:
        raise ValueError(f"unknown view {name!r}; expected one of {sorted(VIEW_BLOCKS)}")
    blocks = [np.asarray(payload[key], dtype=np.float32) for key in VIEW_BLOCKS[name]]
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)


def build_features(config_path: Path, result_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    data_config = dict(config["data"])
    dictionary_config = dict(config["dictionary"])
    started = time.time()
    bundle: MolhivBundle = load_molhiv(
        root=_resolve(data_config.get("root", "data/ogb")),
        max_graphs=(None if data_config.get("max_graphs") is None else int(data_config["max_graphs"])),
        seed=int(data_config.get("subsample_seed", 0)),
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV atom/bond features were not loaded")

    offsets = np.zeros(len(bundle.graphs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([graph.n for graph in bundle.graphs], dtype=np.int64)
    total_patches = int(offsets[-1])
    descriptors = np.empty((total_patches, PATCH_DIM), dtype=np.float32)
    context_masks = np.zeros((total_patches, len(SELECTED_CONTEXT_NAMES)), dtype=bool)
    composition = np.empty((len(bundle.graphs), COMPOSITION_DIM), dtype=np.float32)

    print(
        f"Building invariant typed radius-2 descriptors for {len(bundle.graphs)} graphs "
        f"({total_patches} atom-centred patches)",
        flush=True,
    )
    for graph_position, graph in enumerate(bundle.graphs):
        node_features = bundle.node_feats[graph_position]
        edge_features = bundle.edge_feats[graph_position]
        atom_semantics = compact_atom_semantics(node_features)
        bond_semantics = {
            graph.edge_key(left, right): compact_bond_semantics(values)
            for (left, right), values in edge_features.items()
        }
        composition[graph_position] = composition_69(graph, atom_semantics, bond_semantics)
        start = int(offsets[graph_position])
        for local_position, center in enumerate(graph.nodes):
            descriptors[start + local_position] = typed_radius2_descriptor(
                graph,
                center,
                atom_semantics,
                bond_semantics,
            )
        context_masks[start : int(offsets[graph_position + 1])] = _root_context_matrix(graph, node_features)
        if (graph_position + 1) % 1000 == 0 or graph_position + 1 == len(bundle.graphs):
            print(f"  vectorized {graph_position + 1}/{len(bundle.graphs)} graphs", flush=True)

    typed_raw = np.stack(
        [
            pool_typed_descriptors(descriptors[int(offsets[i]) : int(offsets[i + 1])])
            for i in range(len(bundle.graphs))
        ],
        axis=0,
    ).astype(np.float32)

    train_rows = _sample_train_rows(
        offsets,
        bundle.split["train"],
        maximum=int(dictionary_config["max_train_patches"]),
        seed=int(dictionary_config.get("seed", 0)),
    )
    initial, final, scale, dictionary_info = fit_ksvd(
        descriptors[train_rows],
        n_atoms=int(dictionary_config["n_atoms"]),
        sparsity=int(dictionary_config["sparsity"]),
        iterations=int(dictionary_config["iterations"]),
        power_iterations=int(dictionary_config.get("power_iterations", 6)),
        seed=int(dictionary_config.get("seed", 0)),
    )
    print("Encoding INIT dictionary", flush=True)
    typed_init, context_init, cross_cov_init, init_encoding = dictionary_graph_readouts(
        descriptors,
        context_masks,
        offsets,
        initial,
        scale,
        sparsity=int(dictionary_config["sparsity"]),
        patch_batch_size=int(dictionary_config.get("encoding_patch_batch_size", 8192)),
        return_cross_cov=True,
    )
    print("Encoding FINAL dictionary", flush=True)
    typed_final, context_final, cross_cov_final, final_encoding = dictionary_graph_readouts(
        descriptors,
        context_masks,
        offsets,
        final,
        scale,
        sparsity=int(dictionary_config["sparsity"]),
        patch_batch_size=int(dictionary_config.get("encoding_patch_batch_size", 8192)),
        return_cross_cov=True,
    )

    dataset_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    feature_path = result_dir / "features_full.npz"
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        dataset_indices=dataset_indices,
        labels=bundle.y.astype(np.int64),
        train_indices=np.asarray(bundle.split["train"], dtype=np.int64),
        valid_indices=np.asarray(bundle.split["valid"], dtype=np.int64),
        test_indices=np.asarray(bundle.split["test"], dtype=np.int64),
        graph_offsets=offsets,
        train_patch_rows=train_rows,
        composition=composition,
        typed_raw=typed_raw,
        typed_init=typed_init,
        typed_final=typed_final,
        context_init=context_init,
        context_final=context_final,
        cross_cov_init=cross_cov_init,
        cross_cov_final=cross_cov_final,
        dictionary_initial=initial,
        dictionary_final=final,
        coordinate_scale=scale,
    )
    sampled_graphs = np.searchsorted(offsets, train_rows, side="right") - 1
    manifest = {
        "protocol_id": str(config["protocol_id"]),
        "status": "full" if data_config.get("max_graphs") is None else "development",
        "artifact_alignment": {
            "composition_dimension": COMPOSITION_DIM,
            "typed_patch_dimension": PATCH_DIM,
            "typed_pool_dimension": TYPED_POOL_DIM,
            "typed_pooling": "mean/std/max",
            "context_dimension": int(context_final.shape[1]),
            "context_names": list(SELECTED_CONTEXT_NAMES),
            "cross_cov_dimension": int(cross_cov_final.shape[1]),
            "cross_cov_definition": (
                "population Cov_v(|FINAL K-SVD code_v| normalized by per-centre code mass, "
                "ring-context indicator_v), flattened atom x context"
            ),
        },
        "not_exact_mentor_payload": True,
        "known_upstream_gaps": [
            "the mentor's exact 69-D composition macro schema is unavailable",
            "the mentor's fixed-coordinate structural cache and reference payload are unavailable",
            "ring5/ring6/multi-ring definitions use the repository's explicit proxy",
        ],
        "correctness_changes_vs_historical_52d_proxy": [
            "complete all-centre radius-2 ego; no max_nodes=8 truncation",
            "permutation-invariant shell/bond/topology aggregation; no node-id tie break",
            "compact 48-D atom and 13-D bond semantics; no category modulo",
            "dictionary fit and coordinate scaling use official train rows only",
        ],
        "data": bundle.meta,
        "config": config,
        "feature_file": str(feature_path.resolve()),
        "feature_file_sha256": _sha256(feature_path),
        "implementation_sha256": _sha256(Path(__file__)),
        "dictionary": dictionary_info,
        "encoding": {"initial": init_encoding, "final": final_encoding},
        "split_positive_counts": {
            split: int(bundle.y[indices].sum()) for split, indices in bundle.split.items()
        },
        "leakage_audit": {
            "dictionary_graph_scope": "official train only",
            "dictionary_patch_rows": int(train_rows.size),
            "dictionary_rows_all_from_train": bool(
                np.all(np.isin(sampled_graphs, np.asarray(bundle.split["train"])))
            ),
            "test_labels_used_for_features_or_dictionary": False,
        },
        "runtime_seconds": float(time.time() - started),
    }
    _write_json(result_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "protocol_id": manifest["protocol_id"],
                "status": manifest["status"],
                "data_sizes": {name: int(len(indices)) for name, indices in bundle.split.items()},
                "total_patches": total_patches,
                "dictionary": dictionary_info,
                "encoding": manifest["encoding"],
                "feature_file": manifest["feature_file"],
                "runtime_seconds": manifest["runtime_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build_features(args.config.expanduser().resolve(), args.result_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
