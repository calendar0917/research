"""Unified center-token route for the three follow-up directions.

This entry point implements the complete proposal in the attached note on one
frozen radius-2 rooted-WL substrate:

* conditional role--attribute statistics, including a matched centre-shuffle
  null;
* a train-fold-only KSVD dictionary over center tokens ``z_c=[t_c,a_c]`` and
  graph-level sparse-code readouts;
* distribution pooling over the centre-token population (quantiles, tails,
  entropy/Gini, rare centres/roles and top-k activation).

The protocol has two explicit stages.  ``--stage tune`` performs all
feature/schema/dictionary fitting inside official-train scaffold folds,
searches XGBoost parameters on those folds, and reports official validation
once after the search.  ``--stage test`` checks the frozen hashes and evaluates
all pre-registered views on official test without selecting a view or tuning a
parameter.  The test split is therefore never used by the exploratory stage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
from sklearn.decomposition import sparse_encode
from sklearn.metrics import roc_auc_score
import yaml

from ksvd_research.core import ksvd
from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.cross_center_interaction_screen import (
    _frozen_s_rows,
    _resolve,
    _sha256,
)
from tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal import (
    _official_train_scaffold_splits,
)
from tracks.ksvd.experiments.luyin16.structural_role_fusion_screen import (
    patch_feature_rows,
    patch_roles,
)
from tracks.ksvd.experiments.luyin16.task_aligned_interaction_screen import (
    XGB_SEARCH_KEYS,
    _fit_xgb_predict,
    _suggest_xgb_params,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/center_token_route.yaml"

NODE_ROLE_WIDTH = 64
EDGE_ROLE_WIDTH = 32
NODE_ATTRIBUTE_WIDTH = 40
EDGE_ATTRIBUTE_WIDTH = 13
ROLE_WIDTH = NODE_ROLE_WIDTH + EDGE_ROLE_WIDTH
ATTRIBUTE_WIDTH = NODE_ATTRIBUTE_WIDTH + EDGE_ATTRIBUTE_WIDTH
TOKEN_WIDTH = ROLE_WIDTH + ATTRIBUTE_WIDTH
N_ROOT_ROLES = NODE_ROLE_WIDTH

# Fixed readout dimensions.  Keeping these in one place is important for the
# empty-graph path and for the frozen schema recorded in the manifest.
CONDITIONAL_ROLE_SCALAR_WIDTH = 8
CONDITIONAL_TOP_ROLE_K = 8
CONDITIONAL_COVARIANCE_SUMMARY_WIDTH = ATTRIBUTE_WIDTH + 5 + 8
CONDITIONAL_COMPACT_WIDTH = (
    N_ROOT_ROLES * CONDITIONAL_ROLE_SCALAR_WIDTH
    + CONDITIONAL_TOP_ROLE_K * 3
    + CONDITIONAL_COVARIANCE_SUMMARY_WIDTH
)
CONDITIONAL_FULL_WIDTH = (
    N_ROOT_ROLES
    + 3 * N_ROOT_ROLES * ATTRIBUTE_WIDTH
    + CONDITIONAL_COVARIANCE_SUMMARY_WIDTH
)
DISTRIBUTION_TOP_FEATURE_WIDTH = 21
DISTRIBUTION_TOP_VALUE_WIDTH = 12
DISTRIBUTION_WIDTH = (
    11 * TOKEN_WIDTH
    + N_ROOT_ROLES
    + DISTRIBUTION_TOP_FEATURE_WIDTH
    + DISTRIBUTION_TOP_VALUE_WIDTH
)

# The same names are used in summaries and in the frozen manifest.  The order
# is deliberately fixed so feature provenance can be audited from a row.
VIEW_NAMES = (
    "s_marginal",
    "s_distribution",
    "s_conditional",
    "s_conditional_pca",
    "s_ksvd_init",
    "s_ksvd_final",
    "s_center_token_all",
)
NULL_NAMES = (
    "conditional_shuffled",
    "ksvd_final_shuffled",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _strip_search_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: params[key] for key in XGB_SEARCH_KEYS if key in params}


def _in_range(params: Mapping[str, Any], ranges: Mapping[str, Sequence[float]]) -> bool:
    return all(
        key in params
        and float(ranges[key][0]) <= float(params[key]) <= float(ranges[key][1])
        for key in XGB_SEARCH_KEYS
    )


def _canonical_lex_order(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"lexicographic order expects 2-D rows, got {array.shape}")
    if array.shape[0] <= 1:
        return np.arange(array.shape[0], dtype=np.int64)
    keys = tuple(array[:, column] for column in range(array.shape[1] - 1, -1, -1))
    return np.asarray(np.lexsort(keys), dtype=np.int64)


def _stable_attribute_shuffle(
    topology: np.ndarray,
    attributes: np.ndarray,
    roles: np.ndarray,
    *,
    seed: int,
    repeat: int = 0,
) -> np.ndarray:
    """Shuffle centre attributes while keeping topology and role frequencies.

    Both bags are canonicalized before the permutation.  Consequently the
    null is a function of semantic rows rather than internal node ids and is
    invariant under arbitrary graph relabeling.
    """
    topo = np.asarray(topology, dtype=np.float32)
    attrs = np.asarray(attributes, dtype=np.float32)
    root_roles = np.asarray(roles, dtype=np.int64)
    if topo.ndim != 2 or attrs.ndim != 2 or root_roles.shape != (topo.shape[0],):
        raise ValueError("unaligned centre rows for matched shuffle")
    if topo.shape[0] != attrs.shape[0] or topo.shape[0] <= 1:
        return attrs.copy()
    topo_order = np.asarray(
        sorted(
            range(topo.shape[0]),
            key=lambda index: (
                int(root_roles[index]),
                tuple(float(value) for value in topo[index]),
                tuple(float(value) for value in attrs[index]),
            ),
        ),
        dtype=np.int64,
    )
    attr_order = _canonical_lex_order(attrs)
    digest = hashlib.blake2b(digest_size=8)
    digest.update(b"center-token-matched-shuffle-v1")
    digest.update(np.asarray([int(seed), int(repeat)], dtype=np.int64).tobytes())
    digest.update(root_roles[topo_order].tobytes())
    digest.update(topo[topo_order].tobytes())
    digest.update(attrs[attr_order].tobytes())
    rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
    shuffled = attrs[attr_order][rng.permutation(attrs.shape[0])]
    output = np.empty_like(attrs)
    output[topo_order] = shuffled
    if not np.array_equal(np.sort(attrs, axis=0), np.sort(output, axis=0)):
        raise RuntimeError("centre attribute shuffle changed the attribute bag")
    return output


def _centre_token_rows(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    representation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return topology rows, attribute rows, root roles and graph context."""
    rows = patch_feature_rows(
        graph,
        node_features,
        edge_features,
        "rooted_wl",
        representation,
    )
    topology = np.concatenate([rows["node_role"], rows["edge_role"]], axis=1).astype(
        np.float32, copy=False
    )
    attributes = np.concatenate(
        [rows["node_attribute"], rows["edge_attribute"]], axis=1
    ).astype(np.float32, copy=False)
    root_roles: list[int] = []
    for center in graph.nodes:
        roles = patch_roles(graph, int(center), "rooted_wl", representation)
        positions = np.flatnonzero(np.asarray(roles.nodes, dtype=np.int64) == int(center))
        if positions.size != 1:
            raise RuntimeError("root centre is absent or duplicated in rooted-WL patch")
        root_roles.append(int(roles.node_ids[int(positions[0])]))
    root = np.asarray(root_roles, dtype=np.int64)
    context = np.asarray(rows["context"], dtype=np.float32).reshape(-1)
    if topology.shape[0] != attributes.shape[0] or topology.shape[0] != root.size:
        raise RuntimeError("centre token row alignment changed")
    return topology, attributes, root, context


def _canonical_center_order(
    topology: np.ndarray,
    attributes: np.ndarray,
    roles: np.ndarray,
) -> np.ndarray:
    """Return a node-label-independent order for centre-token rows.

    Centre rows are a population, not an ordered sequence.  Canonical packing
    is nevertheless useful because the train-token reservoir and dictionary
    initialization consume a sequence.  Equal semantic rows are interchangeable
    and therefore need no node-id tie breaker.
    """
    topo = np.asarray(topology, dtype=np.float32)
    attrs = np.asarray(attributes, dtype=np.float32)
    root = np.asarray(roles, dtype=np.int64)
    if topo.ndim != 2 or attrs.ndim != 2 or root.shape != (topo.shape[0],) or attrs.shape[0] != topo.shape[0]:
        raise ValueError("centre-token rows are not aligned for canonical ordering")
    return np.asarray(
        sorted(
            range(topo.shape[0]),
            key=lambda index: (
                int(root[index]),
                tuple(float(value) for value in topo[index]),
                tuple(float(value) for value in attrs[index]),
            ),
        ),
        dtype=np.int64,
    )


def _graph_token_features(
    bundle,
    index: int,
    representation: Mapping[str, Any],
    *,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("center-token route requires node and edge features")
    topology, attributes, roles, context = _centre_token_rows(
        bundle.graphs[int(index)],
        bundle.node_feats[int(index)],
        bundle.edge_feats[int(index)],
        representation,
    )
    shuffled = _stable_attribute_shuffle(
        topology, attributes, roles, seed=int(shuffle_seed)
    )
    order = _canonical_center_order(topology, attributes, roles)
    topology = topology[order]
    attributes = attributes[order]
    roles = roles[order]
    shuffled = shuffled[order]
    return {
        "topology": topology,
        "attributes": attributes,
        "tokens": np.concatenate([topology, attributes], axis=1).astype(
            np.float32, copy=False
        ),
        "roles": roles,
        "shuffled_attributes": shuffled,
        "context": context,
    }


def _distribution_mean_std(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"mean/std readout expects 2-D rows, got {values.shape}")
    if values.shape[0] == 0:
        return np.zeros(2 * values.shape[1], dtype=np.float32)
    return np.concatenate([values.mean(axis=0), values.std(axis=0)]).astype(
        np.float32, copy=False
    )


def _build_token_cache(
    bundle,
    indices: np.ndarray,
    representation: Mapping[str, Any],
    cache_path: Path,
    *,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    """Build a packed, variable-length centre-token cache."""
    selected = np.asarray(indices, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("cannot build an empty center-token cache")
    signature_payload = {
        "schema": "rooted_wl_center_token_route_v1",
        "indices": selected.tolist(),
        "representation": dict(representation),
        "shuffle_seed": int(shuffle_seed),
        "encoder_sha256": _sha256(Path(__file__).resolve()),
        "role_encoder_sha256": _sha256(
            REPO_ROOT / "tracks/ksvd/experiments/luyin16/structural_role_fusion_screen.py"
        ),
    }
    signature = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached = str(np.asarray(archive["signature"]).reshape(-1)[0])
            if cached != signature:
                raise ValueError(f"center-token cache signature mismatch: {cache_path}")
            return {name: np.asarray(archive[name]) for name in archive.files if name != "signature"}

    offsets = np.zeros(selected.size + 1, dtype=np.int64)
    context = np.zeros((selected.size, 5), dtype=np.float32)
    marginal = np.zeros((selected.size, 2 * TOKEN_WIDTH), dtype=np.float32)
    token_parts: list[np.ndarray] = []
    role_parts: list[np.ndarray] = []
    shuffled_attribute_parts: list[np.ndarray] = []
    for position, raw_index in enumerate(selected):
        values = _graph_token_features(
            bundle, int(raw_index), representation, shuffle_seed=int(shuffle_seed)
        )
        token_parts.append(values["tokens"])
        role_parts.append(values["roles"])
        shuffled_attribute_parts.append(values["shuffled_attributes"])
        offsets[position + 1] = offsets[position] + values["tokens"].shape[0]
        context[position] = values["context"]
        marginal[position] = _distribution_mean_std(values["tokens"])
        if position and position % 500 == 0:
            print(f"center-token feature graphs: {position}/{selected.size}", flush=True)
    tokens = np.concatenate(token_parts, axis=0).astype(np.float32, copy=False)
    roles = np.concatenate(role_parts, axis=0).astype(np.int16, copy=False)
    shuffled_attributes = np.concatenate(shuffled_attribute_parts, axis=0).astype(
        np.float32, copy=False
    )
    arrays = {
        "dataset_indices": selected,
        "labels": np.asarray(bundle.y[selected], dtype=np.int64),
        "graph_offsets": offsets,
        "tokens": tokens,
        "roles": roles,
        "shuffled_attributes": shuffled_attributes,
        "marginal": marginal,
        "context": context,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, signature=np.asarray([signature]), **arrays)
    temporary.replace(cache_path)
    return arrays


def _graph_slice(cache: Mapping[str, np.ndarray], row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(cache["graph_offsets"], dtype=np.int64)
    start, stop = int(offsets[int(row)]), int(offsets[int(row) + 1])
    tokens = np.asarray(cache["tokens"][start:stop], dtype=np.float32)
    roles = np.asarray(cache["roles"][start:stop], dtype=np.int64)
    shuffled = np.asarray(cache["shuffled_attributes"][start:stop], dtype=np.float32)
    topology = tokens[:, :ROLE_WIDTH]
    attributes = tokens[:, ROLE_WIDTH:]
    if tokens.shape[0] != roles.size or shuffled.shape != attributes.shape:
        raise RuntimeError("packed token cache slice is not aligned")
    return topology, attributes, roles, shuffled


def _gini(values: np.ndarray) -> float:
    vector = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if vector.size == 0 or float(vector.sum()) <= 1.0e-12:
        return 0.0
    ordered = np.sort(vector)
    n = ordered.size
    return float((2.0 * np.arange(1, n + 1) @ ordered) / (n * ordered.sum()) - (n + 1.0) / n)


def _entropy(values: np.ndarray) -> float:
    vector = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    total = float(vector.sum())
    if total <= 1.0e-12:
        return 0.0
    p = vector / total
    p = p[p > 1.0e-12]
    return float(-np.sum(p * np.log(p)))


def _top_fraction(values: np.ndarray, top_k: int) -> float:
    vector = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if vector.size == 0 or float(vector.sum()) <= 1.0e-12:
        return 0.0
    k = min(int(top_k), vector.size)
    return float(np.sort(vector)[-k:].sum() / vector.sum())


def _fit_distribution_reference(tokens: np.ndarray, roles: np.ndarray) -> dict[str, np.ndarray | float]:
    values = np.asarray(tokens, dtype=np.float32)
    root = np.asarray(roles, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != TOKEN_WIDTH or root.ndim != 1:
        raise ValueError("invalid train centre-token reference arrays")
    if values.shape[0] != root.size:
        raise ValueError("train centre-token reference rows are not aligned")
    return {
        "q10": np.quantile(values, 0.10, axis=0).astype(np.float32),
        "q90": np.quantile(values, 0.90, axis=0).astype(np.float32),
        "norm_q90": np.asarray([np.quantile(np.linalg.norm(values, axis=1), 0.90)], dtype=np.float32),
        "role_counts": np.bincount(root, minlength=N_ROOT_ROLES).astype(np.float32),
        "rare_role_threshold": float(max(2.0, np.quantile(np.bincount(root, minlength=N_ROOT_ROLES), 0.25))),
    }


def _distribution_pool(tokens: np.ndarray, roles: np.ndarray, reference: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float32)
    root = np.asarray(roles, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != TOKEN_WIDTH or root.shape != (values.shape[0],):
        raise ValueError("invalid graph centre-token rows for distribution pooling")
    if values.shape[0] == 0:
        return np.zeros(DISTRIBUTION_WIDTH, dtype=np.float32)
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    q10 = np.asarray(reference["q10"], dtype=np.float32)
    q90 = np.asarray(reference["q90"], dtype=np.float32)
    high_tail = np.mean(values >= q90[None, :], axis=0)
    low_tail = np.mean(values <= q10[None, :], axis=0)
    norms = np.linalg.norm(values, axis=1)
    norm_q90 = float(np.asarray(reference["norm_q90"]).reshape(-1)[0])
    role_counts = np.bincount(root, minlength=N_ROOT_ROLES).astype(np.float32)
    role_mass = role_counts / max(float(root.size), 1.0)
    role_train_counts = np.asarray(reference["role_counts"], dtype=np.float32)
    rare_roles = np.asarray(role_train_counts < float(reference["rare_role_threshold"]), dtype=np.float32)
    rare_center_count = float(np.sum(norms >= norm_q90)) / max(float(norms.size), 1.0)
    top_norms = np.sort(norms)[::-1]
    top_features = np.asarray(
        [
            float(norms.mean()),
            float(norms.std()),
            float(np.quantile(norms, 0.50)),
            float(np.quantile(norms, 0.90)),
            float(norms.max(initial=0.0)),
            _entropy(norms),
            _gini(norms),
            _top_fraction(norms, 1),
            _top_fraction(norms, 3),
            _top_fraction(norms, 5),
            rare_center_count,
            float(np.sum((role_counts > 0) & (role_counts <= 1))),
            float(np.sum((role_counts > 0) & (role_counts <= 2))),
            float(np.sum(role_counts > 0)),
            _entropy(role_counts),
            _gini(role_counts),
            _top_fraction(role_counts, 1),
            _top_fraction(role_counts, 3),
            _top_fraction(role_counts, 5),
            float(np.sum(role_mass * rare_roles)),
            float(np.mean(role_mass > 0.0)),
        ],
        dtype=np.float32,
    )
    # Top-k norm values and the corresponding token mass retain the extreme
    # centre environments without introducing a node-id-dependent ordering.
    top_values = np.zeros(12, dtype=np.float32)
    for position, value in enumerate(top_norms[:6]):
        top_values[position] = float(value)
    if top_norms.size:
        order = np.argsort(-norms)[: min(6, norms.size)]
        top_values[6 : 6 + order.size] = np.asarray(values[order, :].mean(axis=1), dtype=np.float32)
    return np.concatenate(
        [
            values.mean(axis=0),
            values.std(axis=0),
            values.min(axis=0),
            values.max(axis=0),
            quantiles.reshape(-1),
            high_tail,
            low_tail,
            role_mass,
            top_features,
            top_values,
        ]
    ).astype(np.float32, copy=False)


def _conditional_features(
    topology: np.ndarray,
    attributes: np.ndarray,
    roles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return compact and full ``P(attribute | root role)`` statistics."""
    topo = np.asarray(topology, dtype=np.float32)
    attrs = np.asarray(attributes, dtype=np.float32)
    root = np.asarray(roles, dtype=np.int64)
    if topo.ndim != 2 or attrs.ndim != 2 or root.shape != (topo.shape[0],) or attrs.shape[0] != topo.shape[0]:
        raise ValueError("conditional rows are not aligned")
    n = max(topo.shape[0], 1)
    counts = np.bincount(root, minlength=N_ROOT_ROLES).astype(np.float32)
    freq = counts / float(n)
    global_attr = attrs.mean(axis=0, dtype=np.float32) if attrs.shape[0] else np.zeros(ATTRIBUTE_WIDTH, dtype=np.float32)
    means = np.zeros((N_ROOT_ROLES, ATTRIBUTE_WIDTH), dtype=np.float32)
    stds = np.zeros_like(means)
    for role in range(N_ROOT_ROLES):
        selected = attrs[root == role]
        if selected.size:
            means[role] = selected.mean(axis=0, dtype=np.float32)
            stds[role] = selected.std(axis=0, dtype=np.float32)
    residual = means - global_attr[None, :]
    weighted = means - np.sum(freq[:, None] * means, axis=0, keepdims=True)
    covariance = (weighted * freq[:, None]).T @ weighted
    covariance = covariance.astype(np.float32, copy=False)
    eig = np.linalg.eigvalsh(covariance.astype(np.float64, copy=False))[::-1]
    covariance_summary = np.concatenate(
        [
            np.diag(covariance),
            np.asarray(
                [
                    float(np.linalg.norm(covariance)),
                    float(np.trace(covariance)),
                    float(np.sum(np.abs(covariance))),
                    float(_entropy(eig)),
                    float(_gini(eig)),
                ],
                dtype=np.float32,
            ),
            eig[:8].astype(np.float32, copy=False),
        ]
    ).astype(np.float32, copy=False)
    role_scalars = []
    for role in range(N_ROOT_ROLES):
        row = residual[role]
        std_row = stds[role]
        role_scalars.extend(
            [
                float(freq[role]),
                float(np.linalg.norm(means[role])),
                float(np.mean(std_row)),
                float(np.linalg.norm(row)),
                float(np.max(np.abs(row), initial=0.0)),
                float(_entropy(np.abs(means[role]))),
                float(_gini(row)),
                float(_top_fraction(row, 3)),
            ]
        )
    # The compact route is deliberately not a second copy of the full
    # role-by-attribute table.  It keeps interpretable per-role scalars and a
    # sorted set of the strongest role deviations; the complete conditional
    # table is exposed separately and can be projected with train-only PCA.
    strength = np.linalg.norm(residual, axis=1)
    top_roles = np.argsort(-strength, kind="stable")[:CONDITIONAL_TOP_ROLE_K]
    top_role_summary = np.concatenate(
        [
            strength[top_roles],
            np.linalg.norm(means[top_roles], axis=1),
            np.mean(stds[top_roles], axis=1),
        ]
    ).astype(np.float32, copy=False)
    compact = np.concatenate(
        [
            np.asarray(role_scalars, dtype=np.float32),
            top_role_summary,
            covariance_summary,
        ]
    ).astype(np.float32, copy=False)
    full = np.concatenate(
        [freq, means.reshape(-1), stds.reshape(-1), residual.reshape(-1), covariance_summary]
    ).astype(np.float32, copy=False)
    return compact, full, {
        "n_roles": N_ROOT_ROLES,
        "attribute_width": ATTRIBUTE_WIDTH,
        "compact_dimension": int(compact.size),
        "full_dimension": int(full.size),
        "covariance_dimension": int(covariance_summary.size),
        "top_role_k": CONDITIONAL_TOP_ROLE_K,
    }


def _empty_conditional_features() -> tuple[np.ndarray, np.ndarray]:
    """Return schema-sized zero blocks for an empty centre population."""
    return (
        np.zeros(CONDITIONAL_COMPACT_WIDTH, dtype=np.float32),
        np.zeros(CONDITIONAL_FULL_WIDTH, dtype=np.float32),
    )


def _code_readout(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"code readout expects [n_centres,k], got {values.shape}")
    k = values.shape[1]
    if values.shape[0] == 0:
        return np.zeros(3 * k + k * k + 10, dtype=np.float32)
    absolute = np.abs(values)
    mass = absolute.sum(axis=0)
    histogram = mass / max(float(mass.sum()), 1.0e-12)
    covariance = np.cov(values.astype(np.float64, copy=False), rowvar=False, bias=True)
    covariance = np.atleast_2d(covariance).astype(np.float32, copy=False)
    if covariance.shape != (k, k):
        covariance = np.zeros((k, k), dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    extras = np.asarray(
        [
            float(np.mean(np.any(absolute > 1.0e-10, axis=1))),
            float(_entropy(histogram)),
            float(_gini(histogram)),
            float(_top_fraction(histogram, 1)),
            float(_top_fraction(histogram, 3)),
            float(np.mean(norms)),
            float(np.std(norms)),
            float(np.quantile(norms, 0.90)),
            float(norms.max(initial=0.0)),
            float(values.shape[0]),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [histogram, values.mean(axis=0), values.std(axis=0), covariance.reshape(-1), extras]
    ).astype(np.float32, copy=False)


def _reservoir_tokens(
    cache: Mapping[str, np.ndarray],
    graph_rows: np.ndarray,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    reservoir = np.zeros((int(maximum), TOKEN_WIDTH), dtype=np.float32)
    sources = np.full(int(maximum), -1, dtype=np.int64)
    seen = 0
    filled = 0
    for row in np.asarray(graph_rows, dtype=np.int64):
        topology, attributes, _roles, _shuffled = _graph_slice(cache, int(row))
        tokens = np.concatenate([topology, attributes], axis=1).astype(np.float32, copy=False)
        for token in tokens:
            seen += 1
            if filled < int(maximum):
                slot = filled
                filled += 1
            else:
                slot = int(rng.integers(0, seen))
                if slot >= int(maximum):
                    continue
            reservoir[slot] = token
            sources[slot] = int(row)
    if filled == 0:
        raise ValueError("center-token reservoir is empty")
    return reservoir[:filled], sources[:filled]


def _learn_token_dictionaries(
    tokens: np.ndarray,
    dictionary_config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(tokens, dtype=np.float64).T
    n_atoms = int(dictionary_config["n_atoms"])
    sparsity = int(dictionary_config["sparsity"])
    initial, _, init_info = ksvd(
        values,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=1,
        n_iter=0,
        seed=int(seed),
    )
    final, _, final_info = ksvd(
        values,
        n_atoms=initial.shape[1],
        T=sparsity,
        T_min=1,
        n_iter=int(dictionary_config["iterations"]),
        seed=int(seed),
        initial_dictionary=initial,
    )
    return initial.astype(np.float32), final.astype(np.float32), {
        "initial": init_info,
        "final": final_info,
        "n_train_tokens": int(tokens.shape[0]),
        "seed": int(seed),
    }


def _encode_graph_codes(
    tokens: np.ndarray,
    dictionary: np.ndarray,
    *,
    sparsity: int,
    n_jobs: int,
) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float32)
    if values.shape[0] == 0:
        return np.zeros((0, dictionary.shape[1]), dtype=np.float32)
    codes = sparse_encode(
        values,
        np.asarray(dictionary, dtype=np.float32).T,
        algorithm="omp",
        n_nonzero_coefs=min(int(sparsity), int(dictionary.shape[1])),
        n_jobs=int(n_jobs),
        check_input=False,
    )
    return np.asarray(codes, dtype=np.float32)


def _batch_code_readouts(
    token_rows: np.ndarray,
    offsets: np.ndarray,
    dictionary: np.ndarray,
    *,
    sparsity: int,
    n_jobs: int,
) -> np.ndarray:
    """Encode one packed token population and split readouts by graph.

    ``sparse_encode`` is substantially faster when called once for the packed
    population than when called once per molecule.  The offsets are the cache
    offsets for the requested graph order and therefore preserve graph-level
    alignment exactly.
    """
    values = np.asarray(token_rows, dtype=np.float32)
    boundaries = np.asarray(offsets, dtype=np.int64)
    if boundaries.ndim != 1 or boundaries.size == 0 or boundaries[0] != 0:
        raise ValueError("invalid packed token offsets")
    if int(boundaries[-1]) != int(values.shape[0]):
        raise ValueError("packed token offsets do not cover token rows")
    if values.shape[0] == 0:
        return np.zeros((boundaries.size - 1, _code_readout(np.zeros((0, dictionary.shape[1]), dtype=np.float32)).size), dtype=np.float32)
    codes = _encode_graph_codes(
        values,
        dictionary,
        sparsity=int(sparsity),
        n_jobs=int(n_jobs),
    )
    readouts = [
        _code_readout(codes[int(boundaries[row]) : int(boundaries[row + 1])])
        for row in range(boundaries.size - 1)
    ]
    return np.stack(readouts, axis=0).astype(np.float32, copy=False)


def _dictionary_atom_metadata(dictionary: np.ndarray, train_tokens: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(dictionary, dtype=np.float32)
    tokens = np.asarray(train_tokens, dtype=np.float32)
    output: list[dict[str, Any]] = []
    for atom in range(values.shape[1]):
        norm = max(float(np.linalg.norm(values[:, atom])), 1.0e-12)
        similarities = tokens @ (values[:, atom] / norm)
        nearest = int(np.argmax(similarities)) if similarities.size else -1
        row = values[:, atom]
        output.append(
            {
                "atom": int(atom),
                "norm": float(np.linalg.norm(row)),
                "topology_mass": float(np.sum(np.abs(row[:ROLE_WIDTH]))),
                "attribute_mass": float(np.sum(np.abs(row[ROLE_WIDTH:]))),
                "nearest_train_token": nearest,
                "nearest_cosine": float(similarities[nearest]) if nearest >= 0 else 0.0,
                "topology_argmax": int(np.argmax(np.abs(row[:ROLE_WIDTH]))),
                "attribute_argmax": int(np.argmax(np.abs(row[ROLE_WIDTH:]))),
            }
        )
    return output


def _build_graph_blocks(
    cache: Mapping[str, np.ndarray],
    graph_rows: np.ndarray,
    *,
    distribution_reference: Mapping[str, Any],
    dictionaries: tuple[np.ndarray, np.ndarray] | None,
    dictionary_config: Mapping[str, Any],
    shuffle_seed: int,
    include_full_conditional: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows = np.asarray(graph_rows, dtype=np.int64)
    conditional_compact: list[np.ndarray] = []
    conditional_full: list[np.ndarray] = []
    distribution: list[np.ndarray] = []
    conditional_shuffled: list[np.ndarray] = []
    true_token_parts: list[np.ndarray] = []
    shuffled_token_parts: list[np.ndarray] = []
    token_offsets = np.zeros(rows.size + 1, dtype=np.int64)
    contexts = np.asarray(cache["context"][rows], dtype=np.float32)
    for position, row in enumerate(rows):
        topology, attributes, root_roles, shuffled_attributes = _graph_slice(cache, int(row))
        compact, full, _meta = _conditional_features(topology, attributes, root_roles)
        shuffled_compact, _shuffled_full, _ = _conditional_features(
            topology, shuffled_attributes, root_roles
        )
        conditional_compact.append(compact)
        if include_full_conditional:
            conditional_full.append(full)
            conditional_shuffled.append(_shuffled_full)
        distribution.append(_distribution_pool(np.concatenate([topology, attributes], axis=1), root_roles, distribution_reference))
        if dictionaries is not None:
            true_tokens = np.concatenate([topology, attributes], axis=1)
            shuffled_tokens = np.concatenate([topology, shuffled_attributes], axis=1)
            true_token_parts.append(true_tokens)
            shuffled_token_parts.append(shuffled_tokens)
            token_offsets[position + 1] = token_offsets[position] + true_tokens.shape[0]
    blocks = {
        "conditional_compact": np.stack(conditional_compact, axis=0).astype(np.float32),
        "distribution": np.stack(distribution, axis=0).astype(np.float32),
        "context": contexts,
    }
    metadata = {
        "conditional_compact_dimension": int(blocks["conditional_compact"].shape[1]),
        "distribution_dimension": int(blocks["distribution"].shape[1]),
        "context_dimension": int(contexts.shape[1]),
    }
    if include_full_conditional:
        blocks["conditional_full"] = np.stack(conditional_full, axis=0).astype(np.float32)
        blocks["conditional_shuffled"] = np.stack(conditional_shuffled, axis=0).astype(np.float32)
        blocks["conditional_shuffled_compact"] = np.stack(
            [
                _conditional_features(
                    _graph_slice(cache, int(row))[0],
                    _graph_slice(cache, int(row))[3],
                    _graph_slice(cache, int(row))[2],
                )[0]
                for row in rows
            ],
            axis=0,
        ).astype(np.float32)
        metadata["conditional_full_dimension"] = int(blocks["conditional_full"].shape[1])
    if dictionaries is not None:
        initial, final = dictionaries
        packed_true = np.concatenate(true_token_parts, axis=0) if true_token_parts else np.zeros((0, TOKEN_WIDTH), dtype=np.float32)
        packed_shuffled = np.concatenate(shuffled_token_parts, axis=0) if shuffled_token_parts else np.zeros((0, TOKEN_WIDTH), dtype=np.float32)
        encode_jobs = int(dictionary_config.get("encode_n_jobs", -1))
        blocks["ksvd_init"] = _batch_code_readouts(
            packed_true, token_offsets, initial,
            sparsity=int(dictionary_config["sparsity"]), n_jobs=encode_jobs,
        )
        blocks["ksvd_final"] = _batch_code_readouts(
            packed_true, token_offsets, final,
            sparsity=int(dictionary_config["sparsity"]), n_jobs=encode_jobs,
        )
        blocks["ksvd_final_shuffled"] = _batch_code_readouts(
            packed_shuffled, token_offsets, final,
            sparsity=int(dictionary_config["sparsity"]), n_jobs=encode_jobs,
        )
        metadata["ksvd_readout_dimension"] = int(blocks["ksvd_final"].shape[1])
    return blocks, metadata


def _fit_projection(train: np.ndarray, valid: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA

    values = np.asarray(train, dtype=np.float32)
    heldout = np.asarray(valid, dtype=np.float32)
    n_components = min(int(rank), values.shape[0], values.shape[1])
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    train_out = model.fit_transform(values).astype(np.float32, copy=False)
    valid_out = model.transform(heldout).astype(np.float32, copy=False)
    return train_out, valid_out, {
        "n_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(model.explained_variance_ratio_)),
        "model": model,
    }


def _fit_projection_model(train: np.ndarray, rank: int) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Fit PCA on one allowed scope and return its model plus coordinates."""
    from sklearn.decomposition import PCA

    values = np.asarray(train, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("PCA fit requires non-empty two-dimensional training rows")
    n_components = min(int(rank), values.shape[0], values.shape[1])
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    projected = model.fit_transform(values).astype(np.float32, copy=False)
    return model, projected, {
        "n_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(model.explained_variance_ratio_)),
    }


def _project_conditional_blocks(
    train_full: np.ndarray,
    all_full: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit train-only PCA and project an arbitrary aligned block."""
    model, _train_projected, meta = _fit_projection_model(train_full, rank)
    return model.transform(np.asarray(all_full, dtype=np.float32)).astype(np.float32, copy=False), meta


def _assemble_views(
    base: np.ndarray,
    blocks: Mapping[str, np.ndarray],
    *,
    conditional_projection: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    base_values = np.asarray(base, dtype=np.float32)
    views = {
        "s_marginal": base_values,
        "s_distribution": np.concatenate([base_values, blocks["distribution"]], axis=1).astype(np.float32),
        "s_conditional": np.concatenate([base_values, blocks["conditional_compact"]], axis=1).astype(np.float32),
    }
    if conditional_projection is not None:
        views["s_conditional_pca"] = np.concatenate([base_values, conditional_projection], axis=1).astype(np.float32)
    else:
        views["s_conditional_pca"] = views["s_conditional"]
    if "ksvd_init" in blocks:
        views["s_ksvd_init"] = np.concatenate([base_values, blocks["ksvd_init"]], axis=1).astype(np.float32)
        views["s_ksvd_final"] = np.concatenate([base_values, blocks["ksvd_final"]], axis=1).astype(np.float32)
        views["s_center_token_all"] = np.concatenate(
            [base_values, blocks["distribution"], blocks["conditional_compact"], blocks["ksvd_final"]], axis=1
        ).astype(np.float32)
    else:
        views["s_ksvd_init"] = base_values
        views["s_ksvd_final"] = base_values
        views["s_center_token_all"] = np.concatenate(
            [base_values, blocks["distribution"], blocks["conditional_compact"]], axis=1
        ).astype(np.float32)
    return views


def _fixed_classifier(classifier: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(classifier)
    params.update({"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "random_state": 0})
    return params


def _score_view(
    fold_matrices: Sequence[tuple[np.ndarray, np.ndarray]],
    fold_labels: Sequence[tuple[np.ndarray, np.ndarray]],
    params: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    fold_rows = []
    if len(fold_matrices) != len(fold_labels):
        raise ValueError("fold matrices and fold labels must have equal length")
    for fold, ((x_train, x_valid), (y_train, y_valid)) in enumerate(zip(fold_matrices, fold_labels, strict=True)):
        seed_auc = []
        for seed in seeds:
            current = dict(params)
            current["random_state"] = int(seed)
            prediction = _fit_xgb_predict(x_train, y_train, x_valid, current)
            seed_auc.append(float(roc_auc_score(y_valid, prediction)))
        fold_rows.append({"fold": int(fold), "seed_auc": seed_auc, "mean_auc": float(np.mean(seed_auc)), "std_auc": float(np.std(seed_auc))})
    means = [row["mean_auc"] for row in fold_rows]
    return {"folds": fold_rows, "mean_auc": float(np.mean(means)), "fold_std": float(np.std(means))}


def _tune_view(
    fold_matrices: Sequence[tuple[np.ndarray, np.ndarray]],
    fold_labels: Sequence[tuple[np.ndarray, np.ndarray]],
    tuning: Mapping[str, Any],
    classifier: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
    ranges = tuning["ranges"]
    fixed = _strip_search_params(classifier)
    if _in_range(fixed, ranges):
        study.enqueue_trial(fixed)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial, tuning, seed=seed)
        try:
            return float(_score_view(fold_matrices, fold_labels, params, [0])["mean_auc"])
        finally:
            gc.collect()

    study.optimize(objective, n_trials=int(tuning["n_trials"]), show_progress_bar=False)
    best = dict(study.best_trial.params)
    best.update({"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "random_state": 0, "n_jobs": int(tuning.get("n_jobs", -1))})
    fold_score = _score_view(fold_matrices, fold_labels, best, [0])
    return {
        "n_trials": int(tuning["n_trials"]),
        "best_trial": int(study.best_trial.number),
        "best_scaffold_auc": float(fold_score["mean_auc"]),
        "best_scaffold_fold_auc": [row["mean_auc"] for row in fold_score["folds"]],
        "best_params": best,
        "trials": [
            {"trial": int(trial.number), "value": None if trial.value is None else float(trial.value), "params": dict(trial.params), "state": str(trial.state)}
            for trial in study.trials
        ],
    }


def _evaluate_frozen(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    params: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    predictions = []
    records = []
    for seed in seeds:
        current = dict(params)
        current["random_state"] = int(seed)
        prediction = _fit_xgb_predict(matrix[train_rows], labels[train_rows], matrix[valid_rows], current)
        predictions.append(prediction)
        records.append({"seed": int(seed), "auc": float(roc_auc_score(labels[valid_rows], prediction))})
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    return {"rows": records, "mean_auc": float(np.mean([row["auc"] for row in records])), "std_auc": float(np.std([row["auc"] for row in records])), "seed_ensemble_auc": float(roc_auc_score(labels[valid_rows], ensemble))}


def _load_indices(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        original = np.asarray(archive["original_indices"], dtype=np.int64)
        train = original[np.asarray(archive["official_train_indices"], dtype=np.int64)]
        valid = original[np.asarray(archive["official_valid_indices"], dtype=np.int64)]
        test = original[np.asarray(archive["official_test_indices"], dtype=np.int64)]
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    return train, valid, test, payload


def _invariance_audit(bundle, indices: np.ndarray, representation: Mapping[str, Any], seed: int, tolerance: float) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16.patch_object_audit import relabel_graph_features

    rng = np.random.default_rng(int(seed))
    drifts = []
    selected = np.asarray(indices, dtype=np.int64)
    for raw_index in selected:
        index = int(raw_index)
        base = _graph_token_features(bundle, index, representation, shuffle_seed=seed)
        permutation = rng.permutation(bundle.graphs[index].n)
        changed_graph, changed_nodes, changed_edges = relabel_graph_features(bundle.graphs[index], bundle.node_feats[index], bundle.edge_feats[index], permutation)

        class OneGraph:
            graphs = [changed_graph]
            node_feats = [changed_nodes]
            edge_feats = [changed_edges]
            y = np.asarray([0], dtype=np.int64)

        changed = _graph_token_features(OneGraph(), 0, representation, shuffle_seed=seed)
        drifts.extend(
            [
                float(np.max(np.abs(base["topology"] - changed["topology"]), initial=0.0)),
                float(np.max(np.abs(base["attributes"] - changed["attributes"]), initial=0.0)),
                float(np.max(np.abs(base["roles"] - changed["roles"]), initial=0.0)),
                float(np.max(np.abs(base["shuffled_attributes"] - changed["shuffled_attributes"]), initial=0.0)),
            ]
        )
    maximum = max(drifts, default=0.0)
    return {"n_graphs": int(selected.size), "maximum_abs_drift": float(maximum), "tolerance": float(tolerance), "pass": bool(maximum <= float(tolerance))}


def _build_base(cache: Mapping[str, np.ndarray], frozen: Mapping[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    s = _frozen_s_rows(frozen, indices)
    cache_indices = np.asarray(cache["dataset_indices"], dtype=np.int64)
    mapping = {int(index): position for position, index in enumerate(cache_indices)}
    rows = np.asarray([mapping[int(index)] for index in indices], dtype=np.int64)
    return np.concatenate([s, cache["marginal"][rows], cache["context"][rows]], axis=1).astype(np.float32, copy=False)


def _build_reference(cache: Mapping[str, np.ndarray], graph_rows: np.ndarray) -> dict[str, Any]:
    tokens = np.asarray(cache["tokens"], dtype=np.float32)
    roles = np.asarray(cache["roles"], dtype=np.int64)
    offsets = np.asarray(cache["graph_offsets"], dtype=np.int64)
    selected = np.asarray(graph_rows, dtype=np.int64)
    parts = [tokens[int(offsets[row]) : int(offsets[row + 1])] for row in selected]
    role_parts = [roles[int(offsets[row]) : int(offsets[row + 1])] for row in selected]
    return _fit_distribution_reference(np.concatenate(parts, axis=0), np.concatenate(role_parts, axis=0))


def _rows_for_indices(available: np.ndarray, requested: np.ndarray) -> np.ndarray:
    mapping = {int(index): position for position, index in enumerate(np.asarray(available, dtype=np.int64))}
    return np.asarray([mapping[int(index)] for index in requested], dtype=np.int64)


def _fold_blocks(
    cache: Mapping[str, np.ndarray],
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    *,
    dictionary_config: Mapping[str, Any],
    representation: Mapping[str, Any],
    fold_seed: int,
    conditional_rank: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    all_rows = np.concatenate([train_rows, valid_rows]).astype(np.int64)
    reference = _build_reference(cache, train_rows)
    train_tokens, sources = _reservoir_tokens(cache, train_rows, int(dictionary_config["max_train_tokens"]), int(fold_seed))
    initial, final, dictionary_meta = _learn_token_dictionaries(train_tokens, dictionary_config, seed=int(fold_seed))
    blocks_all, block_meta = _build_graph_blocks(
        cache,
        all_rows,
        distribution_reference=reference,
        dictionaries=(initial, final),
        dictionary_config=dictionary_config,
        shuffle_seed=int(fold_seed),
    )
    # Fit the conditional PCA on true fold-train rows only.  It is an optional
    # capacity-matched view; the compact block remains directly interpretable.
    cond_train, cond_valid, cond_meta = _fit_projection(
        blocks_all["conditional_full"][: train_rows.size],
        blocks_all["conditional_full"][train_rows.size :],
        int(conditional_rank),
    )
    blocks_all["conditional_pca"] = np.concatenate([cond_train, cond_valid], axis=0).astype(np.float32, copy=False)
    block_meta.update({"dictionary": dictionary_meta, "conditional_pca": {key: value for key, value in cond_meta.items() if key != "model"}, "reservoir_sources": {"n": int(sources.size), "n_graphs": int(np.unique(sources).size)}})
    return blocks_all, block_meta


def _run_tune(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["data"]
    representation = config["representation"]
    train_indices, valid_indices, _test_indices, fold_archive = _load_indices(_resolve(data["scaffold_folds"]))
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    labels = np.asarray(bundle.y, dtype=np.int64)
    dev_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64, copy=False)
    with np.load(_resolve(data["frozen_features"]), allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    cache = _build_token_cache(bundle, dev_indices, representation, _resolve(data["dev_token_cache"]), shuffle_seed=int(config["feature_seed"]))
    if not np.array_equal(np.asarray(cache["dataset_indices"], dtype=np.int64), dev_indices):
        raise RuntimeError("dev center-token cache row order mismatch")
    audit_count = min(int(config["audit"]["n_graphs"]), int(train_indices.size))
    audit = _invariance_audit(bundle, train_indices[:audit_count], representation, int(config["audit"]["seed"]), float(config["audit"]["tolerance"]))
    if not audit["pass"]:
        raise RuntimeError(f"center-token invariance audit failed: {audit}")
    train_rows = np.arange(train_indices.size, dtype=np.int64)
    valid_rows = np.arange(train_indices.size, dev_indices.size, dtype=np.int64)
    splits, split_meta = _official_train_scaffold_splits(fold_archive, train_indices)
    local_splits = splits
    fold_labels = [
        (labels[train_indices[np.asarray(fold_train, dtype=np.int64)]], labels[train_indices[np.asarray(fold_valid, dtype=np.int64)]])
        for fold_train, fold_valid in local_splits
    ]
    # Build each outer fold once; all candidate models consume the same honest
    # fold-specific conditional schema and dictionary coordinates.
    candidate_fold_matrices: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in VIEW_NAMES}
    null_fold_matrices: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in NULL_NAMES}
    fold_metadata: list[dict[str, Any]] = []
    for fold_id, (fold_train, fold_valid) in enumerate(local_splits):
        blocks, metadata = _fold_blocks(
            cache,
            np.asarray(fold_train, dtype=np.int64),
            np.asarray(fold_valid, dtype=np.int64),
            dictionary_config=config["dictionary"],
            representation=representation,
            fold_seed=int(config["feature_seed"]) + 1009 * fold_id,
            conditional_rank=int(config["conditional_pca_rank"]),
        )
        fold_base = _build_base(cache, frozen, train_indices[np.concatenate([fold_train, fold_valid])])
        views = _assemble_views(fold_base, blocks, conditional_projection=blocks["conditional_pca"])
        null_views = {
            "conditional_shuffled": np.concatenate([fold_base, blocks["conditional_shuffled_compact"]], axis=1).astype(np.float32),
            "ksvd_final_shuffled": np.concatenate([fold_base, blocks["ksvd_final_shuffled"]], axis=1).astype(np.float32),
        }
        # ``fold_train``/``fold_valid`` are local rows in official-train space;
        # the matrices above are ordered fold-train then fold-valid.
        for name in VIEW_NAMES:
            candidate_fold_matrices[name].append((views[name][: fold_train.size], views[name][fold_train.size :]))
        null_fold_matrices["conditional_shuffled"].append((null_views["conditional_shuffled"][: fold_train.size], null_views["conditional_shuffled"][fold_train.size :]))
        null_fold_matrices["ksvd_final_shuffled"].append((null_views["ksvd_final_shuffled"][: fold_train.size], null_views["ksvd_final_shuffled"][fold_train.size :]))
        fold_metadata.append({"fold": int(fold_id), "n_train": int(fold_train.size), "n_valid": int(fold_valid.size), **metadata})
        print(f"fold {fold_id}: built center-token conditional/distribution/KSVD blocks", flush=True)
        del blocks, views, null_views
        gc.collect()
    tuning = config["tuning"]
    classifier = config["classifier"]
    seeds = [int(value) for value in config.get("model_seeds", [0, 1, 2])]
    view_results: dict[str, Any] = {}
    start = time.perf_counter()
    for view_id, name in enumerate(VIEW_NAMES):
        search = _tune_view(candidate_fold_matrices[name], fold_labels, tuning, classifier, seed=int(tuning["seed"]) + 1009 * view_id)
        fixed = _score_view(candidate_fold_matrices[name], fold_labels, _fixed_classifier(classifier), seeds)
        tuned = _score_view(candidate_fold_matrices[name], fold_labels, search["best_params"], seeds)
        view_results[name] = {"dimension": int(candidate_fold_matrices[name][0][0].shape[1]), "search": search, "fixed_scaffold": fixed, "tuned_scaffold": tuned}
        print(f"{name}: tuned={search['best_scaffold_auc']:.6f}; fixed={fixed['mean_auc']:.6f}; dim={view_results[name]['dimension']}", flush=True)
    null_results = {}
    for name in NULL_NAMES:
        null_results[name] = {"dimension": int(null_fold_matrices[name][0][0].shape[1]), "fixed_scaffold": _score_view(null_fold_matrices[name], fold_labels, _fixed_classifier(classifier), seeds)}
        print(f"{name}: fixed={null_results[name]['fixed_scaffold']['mean_auc']:.6f}", flush=True)
    # Full official-train fit for the one-time official validation report.  No
    # validation row affects the schemas or dictionary.
    full_blocks, full_meta = _fold_blocks(
        cache,
        train_rows,
        valid_rows,
        dictionary_config=config["dictionary"],
        representation=representation,
        fold_seed=int(config["feature_seed"]),
        conditional_rank=int(config["conditional_pca_rank"]),
    )
    dev_base = _build_base(cache, frozen, dev_indices)
    dev_views = _assemble_views(dev_base, full_blocks, conditional_projection=full_blocks["conditional_pca"])
    valid_report: dict[str, Any] = {}
    for name in VIEW_NAMES:
        valid_report[name] = _evaluate_frozen(dev_views[name], labels[dev_indices], train_rows, valid_rows, view_results[name]["search"]["best_params"], seeds)
        view_results[name]["official_valid"] = valid_report[name]
        print(f"official-valid {name}: mean={valid_report[name]['mean_auc']:.6f}; ensemble={valid_report[name]['seed_ensemble_auc']:.6f}", flush=True)
    null_valid = {
        "conditional_shuffled": _evaluate_frozen(np.concatenate([dev_base, full_blocks["conditional_shuffled_compact"]], axis=1), labels[dev_indices], train_rows, valid_rows, view_results["s_conditional"]["search"]["best_params"], seeds),
        "ksvd_final_shuffled": _evaluate_frozen(np.concatenate([dev_base, full_blocks["ksvd_final_shuffled"]], axis=1), labels[dev_indices], train_rows, valid_rows, view_results["s_ksvd_final"]["search"]["best_params"], seeds),
    }
    for name in NULL_NAMES:
        null_results[name]["official_valid"] = null_valid[name]
    selected = max(VIEW_NAMES, key=lambda name: (float(view_results[name]["search"]["best_scaffold_auc"]), int(name == "s_center_token_all")))
    output = {
        "protocol_id": config["protocol_id"],
        "stage": "candidate_freeze_before_terminal_test",
        "official_validation_or_test_evaluated": True,
        "official_test_evaluated": False,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(Path(__file__)),
        "role_encoder_sha256": _sha256(REPO_ROOT / "tracks/ksvd/experiments/luyin16/structural_role_fusion_screen.py"),
        "audit_boundary": {"search_fit_scope": "official-train scaffold folds only", "official_validation_used_once_for_frozen_reporting": True, "official_test_encoded": False, "official_test_evaluated": False},
        "data": {"dataset": data["dataset"], "scaffold_folds": str(_resolve(data["scaffold_folds"])), "scaffold_folds_sha256": _sha256(_resolve(data["scaffold_folds"])), "frozen_features": str(_resolve(data["frozen_features"])), "full_features": str(_resolve(data["full_features"])), "dev_token_cache": str(_resolve(data["dev_token_cache"])), "test_token_cache": str(_resolve(data["test_token_cache"])), "n_train": int(train_indices.size), "n_valid": int(valid_indices.size), "split_metadata": split_meta},
        "representation": dict(representation),
        "feature_schema": {"token": "[node_role(64), edge_role(32), node_attribute(40), edge_attribute(13)]", "conditional": "rooted-WL root-role conditional mean/std/residual/frequency + role covariance", "distribution": "coordinate quantiles/tails + centre norm and role entropy/Gini/rare/top-k", "ksvd": "train-token reservoir, INIT/FINAL sparse-code histogram/mean/std/covariance"},
        "invariance_audit": audit,
        "dictionary": dict(config["dictionary"]),
        "conditional_pca_rank": int(config["conditional_pca_rank"]),
        "model_seeds": seeds,
        "views": view_results,
        "null_views": null_results,
        "fold_metadata": fold_metadata,
        "full_fit_metadata": full_meta,
        "selection": {"rule": "highest official-train scaffold CV; s_center_token_all wins ties", "candidate_views": list(VIEW_NAMES), "selected_view": selected},
        "runtime": {"seconds": float(time.perf_counter() - start), "python": platform.python_version(), "platform": platform.platform()},
    }
    output_json = _resolve(config["output"]["frozen_manifest_json"])
    output_md = _resolve(config["output"]["frozen_manifest_markdown"])
    _write_json_atomic(output_json, output)
    lines = ["# Center-token route candidate freeze", "", f"Protocol: `{config['protocol_id']}`", "", "Official test was not encoded or evaluated in this stage.", "", "| view | dim | train scaffold CV | official-valid mean | valid ensemble |", "|---|---:|---:|---:|---:|"]
    for name in VIEW_NAMES:
        row = view_results[name]
        lines.append(f"| `{name}` | {row['dimension']} | {row['search']['best_scaffold_auc']:.6f} | {row['official_valid']['mean_auc']:.6f} | {row['official_valid']['seed_ensemble_auc']:.6f} |")
    lines.extend(["", f"Selected by train-only scaffold CV: `{selected}`.", "", "Null controls:"])
    for name in NULL_NAMES:
        lines.append(f"- `{name}` fixed scaffold mean {null_results[name]['fixed_scaffold']['mean_auc']:.6f}; official-valid mean {null_results[name]['official_valid']['mean_auc']:.6f}.")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _run_test(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _resolve(config["output"]["frozen_manifest_json"])
    output_path = _resolve(config["output"]["terminal_test_json"])
    if output_path.exists():
        raise FileExistsError(f"refusing repeated center-token terminal test: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != _sha256(config_path) or manifest.get("implementation_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("center-token frozen manifest hash mismatch")
    data = config["data"]
    train_indices, valid_indices, test_indices, _fold_archive = _load_indices(_resolve(data["scaffold_folds"]))
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    labels = np.asarray(bundle.y, dtype=np.int64)
    dev_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64, copy=False)
    with np.load(_resolve(data["frozen_features"]), allow_pickle=False) as archive:
        frozen_dev = {name: np.asarray(archive[name]) for name in archive.files}
    # The frozen development file intentionally contains only train+valid.  A
    # separate label-free composition file supplies S for test; verify the
    # overlapping development rows byte-for-byte before touching test labels.
    full_features_path = _resolve(data["full_features"])
    with np.load(full_features_path, allow_pickle=False) as archive:
        full_s = np.asarray(archive["composition"], dtype=np.float32)
        full_s_indices = np.arange(full_s.shape[0], dtype=np.int64)
    frozen_dev_s = _frozen_s_rows(frozen_dev, dev_indices)
    if not np.array_equal(full_s[dev_indices], frozen_dev_s):
        raise RuntimeError("full composition S disagrees with frozen development S")
    frozen_all = {"dataset_indices": full_s_indices, "s": full_s}
    dev_cache = _build_token_cache(bundle, dev_indices, config["representation"], _resolve(data["dev_token_cache"]), shuffle_seed=int(config["feature_seed"]))
    test_cache = _build_token_cache(bundle, test_indices, config["representation"], _resolve(data["test_token_cache"]), shuffle_seed=int(config["feature_seed"]))
    test_rows = np.arange(test_indices.size, dtype=np.int64)
    model_seeds = [int(value) for value in manifest["model_seeds"]]
    results: dict[str, Any] = {}
    dev_labels = labels[dev_indices]
    for scope, fit_rows in (("train_only", np.arange(train_indices.size, dtype=np.int64)), ("train_valid_refit", np.arange(dev_indices.size, dtype=np.int64))):
        scope_seed = int(config["feature_seed"]) + (0 if scope == "train_only" else 777)
        reference = _build_reference(dev_cache, fit_rows)
        fit_tokens, sources = _reservoir_tokens(
            dev_cache,
            fit_rows,
            int(config["dictionary"]["max_train_tokens"]),
            scope_seed,
        )
        initial, final, dictionary_meta = _learn_token_dictionaries(
            fit_tokens, config["dictionary"], seed=scope_seed
        )
        dev_blocks, dev_meta = _build_graph_blocks(
            dev_cache,
            np.arange(dev_indices.size, dtype=np.int64),
            distribution_reference=reference,
            dictionaries=(initial, final),
            dictionary_config=config["dictionary"],
            shuffle_seed=scope_seed,
        )
        test_blocks, test_meta = _build_graph_blocks(
            test_cache,
            test_rows,
            distribution_reference=reference,
            dictionaries=(initial, final),
            dictionary_config=config["dictionary"],
            shuffle_seed=scope_seed,
        )
        # Fit the conditional projection on the permitted development scope;
        # test rows are transformed only after the projection is fixed.
        pca, _cond_fit, cond_meta = _fit_projection_model(
            dev_blocks["conditional_full"][fit_rows], int(config["conditional_pca_rank"])
        )
        dev_blocks["conditional_pca"] = pca.transform(dev_blocks["conditional_full"]).astype(np.float32, copy=False)
        test_blocks["conditional_pca"] = pca.transform(test_blocks["conditional_full"]).astype(np.float32, copy=False)
        dev_base = _build_base(dev_cache, frozen_dev, dev_indices)
        test_base = _build_base(test_cache, frozen_all, test_indices)
        dev_views = _assemble_views(dev_base, dev_blocks, conditional_projection=dev_blocks["conditional_pca"])
        test_views = _assemble_views(test_base, test_blocks, conditional_projection=test_blocks["conditional_pca"])
        null_dev_views = {
            "conditional_shuffled": np.concatenate([dev_base, dev_blocks["conditional_shuffled_compact"]], axis=1).astype(np.float32),
            "ksvd_final_shuffled": np.concatenate([dev_base, dev_blocks["ksvd_final_shuffled"]], axis=1).astype(np.float32),
        }
        null_test_views = {
            "conditional_shuffled": np.concatenate([test_base, test_blocks["conditional_shuffled_compact"]], axis=1).astype(np.float32),
            "ksvd_final_shuffled": np.concatenate([test_base, test_blocks["ksvd_final_shuffled"]], axis=1).astype(np.float32),
        }
        for name in VIEW_NAMES:
            params = manifest["views"][name]["search"]["best_params"]
            scored = []
            predictions = []
            for seed in model_seeds:
                current = dict(params)
                current["random_state"] = int(seed)
                prediction = _fit_xgb_predict(dev_views[name][fit_rows], dev_labels[fit_rows], test_views[name], current)
                predictions.append(prediction)
                scored.append({"seed": int(seed), "test_auc": float(roc_auc_score(labels[test_indices], prediction))})
            ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
            results.setdefault(name, {})[scope] = {"rows": scored, "mean_auc": float(np.mean([row["test_auc"] for row in scored])), "std_auc": float(np.std([row["test_auc"] for row in scored])), "seed_ensemble_auc": float(roc_auc_score(labels[test_indices], ensemble)), "dimension": int(test_views[name].shape[1])}
            print(f"test {scope} {name}: mean={results[name][scope]['mean_auc']:.6f}; ensemble={results[name][scope]['seed_ensemble_auc']:.6f}", flush=True)
        null_params = {
            "conditional_shuffled": manifest["views"]["s_conditional"]["search"]["best_params"],
            "ksvd_final_shuffled": manifest["views"]["s_ksvd_final"]["search"]["best_params"],
        }
        for name in NULL_NAMES:
            predictions = []
            scored = []
            for seed in model_seeds:
                current = dict(null_params[name])
                current["random_state"] = int(seed)
                prediction = _fit_xgb_predict(null_dev_views[name][fit_rows], dev_labels[fit_rows], null_test_views[name], current)
                predictions.append(prediction)
                scored.append({"seed": int(seed), "test_auc": float(roc_auc_score(labels[test_indices], prediction))})
            ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
            results.setdefault(name, {})[scope] = {"rows": scored, "mean_auc": float(np.mean([row["test_auc"] for row in scored])), "std_auc": float(np.std([row["test_auc"] for row in scored])), "seed_ensemble_auc": float(roc_auc_score(labels[test_indices], ensemble)), "dimension": int(null_test_views[name].shape[1])}
            print(f"test {scope} {name}: mean={results[name][scope]['mean_auc']:.6f}; ensemble={results[name][scope]['seed_ensemble_auc']:.6f}", flush=True)
        del dev_blocks, test_blocks, dev_views, test_views, null_dev_views, null_test_views
        gc.collect()
    output = {"protocol_id": config["protocol_id"], "stage": "controlled_frozen_terminal_test", "official_test_evaluated": True, "test_used_for_selection_or_tuning": False, "historical_test_already_seen_by_older_routes": True, "source_manifest": str(manifest_path), "source_manifest_sha256": _sha256(manifest_path), "split_sizes": {"train": int(train_indices.size), "valid": int(valid_indices.size), "test": int(test_indices.size), "test_positive": int(labels[test_indices].sum())}, "views": results, "model_seeds": model_seeds}
    _write_json_atomic(output_path, output)
    output_md = _resolve(config["output"]["terminal_test_markdown"])
    lines = ["# Center-token route frozen terminal test", "", "Test was not used for view/parameter selection.", ""]
    for scope in ("train_only", "train_valid_refit"):
        lines.extend([f"## {scope}", "", "| view | five-seed mean | seed ensemble |", "|---|---:|---:|"])
        for name in VIEW_NAMES:
            row = results[name][scope]
            lines.append(f"| `{name}` | {row['mean_auc']:.6f} | {row['seed_ensemble_auc']:.6f} |")
        lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("tune", "test"), required=True)
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = _run_tune(config_path, config) if args.stage == "tune" else _run_test(config_path, config)
    print(json.dumps({"protocol_id": result["protocol_id"], "stage": result["stage"], "selected_view": result.get("selection", {}).get("selected_view"), "views": {name: {"valid": result["views"][name].get("official_valid", {}).get("mean_auc"), "test_train_only": result["views"][name].get("train_only", {}).get("mean_auc")} for name in result.get("views", {})}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
