"""Low-cost invariance and object-scope audit for the MolHIV radius-2 proxy."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from xgboost import XGBClassifier

from ksvd_research.core import from_edges
from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.mentor_concept_replication import (
    _radius2_rooted_chem_vector,
    _radius2_rooted_topology_vector,
    _radius_ego_nodes,
    _rooted_bfs_order,
    typed_matrix_distribution_readout,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/patch_object_audit.yaml"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def relabel_graph_features(graph, node_features, edge_features, permutation: np.ndarray):
    """Relabel a graph with ``permutation[old_id] = new_id``."""
    permutation = np.asarray(permutation, dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(graph.n)):
        raise ValueError("permutation must contain every node id exactly once")
    changed = from_edges(
        graph.n,
        [
            (int(permutation[left]), int(permutation[right]))
            for left, right in graph.edges()
        ],
    )
    changed_nodes = np.empty_like(node_features)
    changed_nodes[permutation] = node_features
    changed_edges: dict[tuple[int, int], np.ndarray] = {}
    for (left, right), values in edge_features.items():
        mapped_left, mapped_right = int(permutation[left]), int(permutation[right])
        key = (
            (mapped_left, mapped_right)
            if mapped_left < mapped_right
            else (mapped_right, mapped_left)
        )
        changed_edges[key] = np.asarray(values).copy()
    return changed, changed_nodes, changed_edges


def _matched_scope_chem_vector(
    graph,
    nodes: set[int],
    center: int,
    max_nodes: int,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, set[int]]:
    """Historical 52D schema, but all blocks use the retained adjacency nodes."""
    order = _rooted_bfs_order(graph, nodes, center)[:max_nodes]
    kept = set(order)
    position = {node: index for index, node in enumerate(order)}
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    induced = graph.induced(kept)
    for left, right in induced.edges():
        i, j = position[left], position[right]
        adjacency[i, j] = adjacency[j, i] = 1.0
    upper = np.asarray(
        [adjacency[i, j] for i in range(max_nodes) for j in range(i + 1, max_nodes)],
        dtype=np.float64,
    )
    atom_hist = np.zeros(16, dtype=np.float64)
    for node in kept:
        atom_hist[int(node_features[node, 0]) % 16] += 1.0
    if atom_hist.sum() > 0:
        atom_hist /= atom_hist.sum()
    bond_hist = np.zeros(8, dtype=np.float64)
    for left, right in induced.edges():
        key = (left, right) if left < right else (right, left)
        values = edge_features.get(key)
        if values is not None and values.size:
            bond_hist[int(values[0]) % 8] += 1.0
    if bond_hist.sum() > 0:
        bond_hist /= bond_hist.sum()
    return np.concatenate([upper, atom_hist, bond_hist]), kept


def build_graph_patch_views(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    *,
    radius: int,
    max_nodes: int,
    centers: Sequence[int] | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """Build historical and matched-scope graph readouts for one molecule."""
    ordered_centers = list(graph.nodes if centers is None else centers)
    legacy_columns: list[np.ndarray] = []
    matched_columns: list[np.ndarray] = []
    topology_columns: list[np.ndarray] = []
    full_sizes: list[int] = []
    kept_sizes: list[int] = []
    dropped_edges: list[int] = []
    for center in ordered_centers:
        nodes = _radius_ego_nodes(graph, int(center), radius)
        legacy_columns.append(
            _radius2_rooted_chem_vector(
                graph,
                nodes,
                int(center),
                max_nodes,
                node_features,
                edge_features,
            )
        )
        matched, kept = _matched_scope_chem_vector(
            graph,
            nodes,
            int(center),
            max_nodes,
            node_features,
            edge_features,
        )
        matched_columns.append(matched)
        topology_columns.append(
            _radius2_rooted_topology_vector(graph, nodes, int(center), max_nodes)
        )
        full_sizes.append(len(nodes))
        kept_sizes.append(len(kept))
        dropped_edges.append(graph.induced(nodes).num_edges() - graph.induced(kept).num_edges())

    def matrix(columns: Sequence[np.ndarray]) -> np.ndarray:
        value = np.stack(columns, axis=1).astype(np.float64)
        if normalize:
            value /= np.maximum(np.linalg.norm(value, axis=0, keepdims=True), 1e-12)
        return value

    legacy_matrix = matrix(legacy_columns)
    matched_matrix = matrix(matched_columns)
    topology_matrix = matrix(topology_columns)
    return {
        "legacy_matrix": legacy_matrix,
        "matched_matrix": matched_matrix,
        "topology_matrix": topology_matrix,
        "legacy_readout": typed_matrix_distribution_readout(legacy_matrix),
        "matched_readout": typed_matrix_distribution_readout(matched_matrix),
        "topology_readout": typed_matrix_distribution_readout(topology_matrix),
        "full_sizes": np.asarray(full_sizes, dtype=np.int64),
        "kept_sizes": np.asarray(kept_sizes, dtype=np.int64),
        "dropped_edges": np.asarray(dropped_edges, dtype=np.int64),
    }


def _select_rows(
    labels: np.ndarray,
    candidates: np.ndarray,
    n_graphs: int,
    positive_fraction: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = candidates[labels[candidates] == 1]
    negative = candidates[labels[candidates] == 0]
    n_positive = min(len(positive), int(round(n_graphs * positive_fraction)))
    n_negative = min(len(negative), n_graphs - n_positive)
    selected = np.concatenate(
        [
            rng.choice(positive, n_positive, replace=False),
            rng.choice(negative, n_negative, replace=False),
        ]
    )
    rng.shuffle(selected)
    return selected.astype(np.int64)


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12))


def _collision_summary(counts: Counter[int], modulus: int) -> dict[str, Any]:
    bins: dict[int, list[int]] = {}
    for category in sorted(counts):
        bins.setdefault(int(category) % modulus, []).append(int(category))
    colliding = {key: values for key, values in bins.items() if len(values) > 1}
    collided_occurrences = sum(
        counts[category] for values in colliding.values() for category in values
    )
    total = sum(counts.values())
    return {
        "modulus": modulus,
        "observed_categories": sorted(counts),
        "n_observed_categories": len(counts),
        "colliding_bins": colliding,
        "n_colliding_bins": len(colliding),
        "collided_occurrence_fraction": float(collided_occurrences / max(total, 1)),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    invariant = result["relabel_invariance"]
    scope = result["object_scope"]
    collision = result["category_encoding"]
    decisions = result["decisions"]
    return "\n".join(
        [
            "# MolHIV radius-2 patch-object audit",
            "",
            f"Protocol: `{result['protocol_id']}`",
            "",
            "## Decision",
            "",
            f"- relabel invariance: **{decisions['relabel_invariance']}**",
            f"- matched topology/attribute scope: **{decisions['matched_object_scope']}**",
            f"- modulo category encoding: **{decisions['category_encoding']}**",
            f"- proceed to fold attribution: **{decisions['proceed_to_fold_attribution']}**",
            "",
            "## Relabel audit",
            "",
            f"- audited graphs: {result['sample']['n_graphs']}",
            f"- comparisons: {invariant['typed_feature_max_abs']['n']}",
            f"- typed readout maximum drift: {invariant['typed_feature_max_abs']['maximum']:.8g}",
            f"- topology readout maximum drift: {invariant['topology_feature_max_abs']['maximum']:.8g}",
            f"- frozen prediction maximum drift: {invariant['prediction_abs_difference']['maximum']:.8g}",
            f"- cached-feature reproduction maximum error: {invariant['cached_reproduction_max_abs']:.8g}",
            "",
            "## Object scope",
            "",
            f"- truncated patch fraction: {scope['truncated_patch_fraction']:.6f}",
            f"- graphs with truncation: {scope['graphs_with_truncation_fraction']:.6f}",
            f"- legacy vs retained-node readout maximum relative L2: {scope['legacy_vs_matched_relative_l2']['maximum']:.6f}",
            f"- mean dropped nodes per patch: {scope['mean_dropped_nodes_per_patch']:.6f}",
            f"- mean dropped edges per patch: {scope['mean_dropped_edges_per_patch']:.6f}",
            "",
            "## Category encoding",
            "",
            f"- atom raw categories: {collision['atom']['n_observed_categories']}; colliding modulo bins: {collision['atom']['n_colliding_bins']}",
            f"- atom occurrences in colliding bins: {collision['atom']['collided_occurrence_fraction']:.6f}",
            f"- bond raw categories: {collision['bond']['n_observed_categories']}; colliding modulo bins: {collision['bond']['n_colliding_bins']}",
            "",
            "## Interpretation",
            "",
            "This audit changes no model selection and never evaluates official test. A failed gate means the current object must be corrected before RAW/INIT/FINAL attribution.",
            "",
        ]
    )


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    rep_cfg = config["representation"]
    audit_cfg = config["audit"]
    feature_path = _resolve(config["frozen_features"])
    search_path = _resolve(config["frozen_search"])
    with np.load(feature_path, allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    train_count = int(frozen["train_count"][0])
    labels = frozen["labels"].astype(np.int64)
    if str(data_cfg["split"]) != "valid":
        raise ValueError("the initial audit is restricted to frozen official validation")
    candidates = np.arange(train_count, labels.shape[0], dtype=np.int64)
    selected_rows = _select_rows(
        labels,
        candidates,
        int(audit_cfg["n_graphs"]),
        float(audit_cfg["positive_fraction"]),
        int(audit_cfg["seed"]),
    )
    dataset_indices = frozen["dataset_indices"].astype(np.int64)
    bundle = load_molhiv(root=_resolve(data_cfg["root"]), with_features=True)
    assert bundle.node_feats is not None and bundle.edge_feats is not None

    search = json.loads(search_path.read_text(encoding="utf-8"))
    params = dict(search["views"]["s_r_raw"]["best_params"])
    params["random_state"] = 0
    train_labels = labels[:train_count]
    weight = float(np.sum(train_labels == 0) / max(1, np.sum(train_labels == 1)))
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=weight,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(frozen["s_r_raw"][:train_count], train_labels)

    rng = np.random.default_rng(int(audit_cfg["seed"]))
    typed_max: list[float] = []
    typed_relative: list[float] = []
    topology_max: list[float] = []
    topology_relative: list[float] = []
    scope_relative: list[float] = []
    original_features: list[np.ndarray] = []
    permuted_features: list[np.ndarray] = []
    comparison_rows: list[dict[str, Any]] = []
    cached_reproduction_max = 0.0
    total_patches = 0
    truncated_patches = 0
    graphs_with_truncation = 0
    dropped_nodes = 0
    dropped_edges = 0

    for row in selected_rows:
        dataset_index = int(dataset_indices[row])
        graph = bundle.graphs[dataset_index]
        node_features = bundle.node_feats[dataset_index]
        edge_features = bundle.edge_feats[dataset_index]
        base = build_graph_patch_views(
            graph,
            node_features,
            edge_features,
            radius=int(rep_cfg["radius"]),
            max_nodes=int(rep_cfg["max_nodes"]),
            normalize=bool(rep_cfg["normalize_patches"]),
        )
        cached_reproduction_max = max(
            cached_reproduction_max,
            float(np.max(np.abs(base["legacy_readout"] - frozen["r_raw"][row]))),
        )
        graph_truncated = bool(np.any(base["full_sizes"] > int(rep_cfg["max_nodes"])))
        graphs_with_truncation += int(graph_truncated)
        total_patches += int(base["full_sizes"].size)
        truncated_patches += int(np.sum(base["full_sizes"] > int(rep_cfg["max_nodes"])))
        dropped_nodes += int(np.sum(base["full_sizes"] - base["kept_sizes"]))
        dropped_edges += int(np.sum(base["dropped_edges"]))
        scope_relative.append(_relative_l2(base["legacy_readout"], base["matched_readout"]))

        base_full = np.concatenate([frozen["s"][row], base["legacy_readout"]])
        for repeat in range(int(audit_cfg["permutations_per_graph"])):
            permutation = rng.permutation(graph.n)
            changed_graph, changed_nodes, changed_edges = relabel_graph_features(
                graph, node_features, edge_features, permutation
            )
            mapped_centers = [int(permutation[center]) for center in graph.nodes]
            changed = build_graph_patch_views(
                changed_graph,
                changed_nodes,
                changed_edges,
                radius=int(rep_cfg["radius"]),
                max_nodes=int(rep_cfg["max_nodes"]),
                centers=mapped_centers,
                normalize=bool(rep_cfg["normalize_patches"]),
            )
            typed_delta = np.abs(base["legacy_readout"] - changed["legacy_readout"])
            topology_delta = np.abs(base["topology_readout"] - changed["topology_readout"])
            typed_max.append(float(typed_delta.max()))
            typed_relative.append(_relative_l2(base["legacy_readout"], changed["legacy_readout"]))
            topology_max.append(float(topology_delta.max()))
            topology_relative.append(
                _relative_l2(base["topology_readout"], changed["topology_readout"])
            )
            original_features.append(base_full)
            permuted_features.append(
                np.concatenate([frozen["s"][row], changed["legacy_readout"]])
            )
            comparison_rows.append(
                {
                    "dataset_index": dataset_index,
                    "label": int(labels[row]),
                    "repeat": repeat,
                    "n_nodes": graph.n,
                    "truncated_patch_fraction": float(
                        np.mean(base["full_sizes"] > int(rep_cfg["max_nodes"]))
                    ),
                    "typed_feature_max_abs": typed_max[-1],
                    "typed_feature_relative_l2": typed_relative[-1],
                    "topology_feature_max_abs": topology_max[-1],
                    "topology_feature_relative_l2": topology_relative[-1],
                }
            )

    original_array = np.stack(original_features).astype(np.float32)
    permuted_array = np.stack(permuted_features).astype(np.float32)
    original_predictions = model.predict_proba(original_array)[:, 1]
    permuted_predictions = model.predict_proba(permuted_array)[:, 1]
    prediction_delta = np.abs(original_predictions - permuted_predictions)
    for row, delta in zip(comparison_rows, prediction_delta):
        row["prediction_abs_difference"] = float(delta)
    worst = sorted(
        comparison_rows,
        key=lambda row: (row["prediction_abs_difference"], row["typed_feature_max_abs"]),
        reverse=True,
    )[:20]

    atom_counts: Counter[int] = Counter()
    bond_counts: Counter[int] = Counter()
    for dataset_index in dataset_indices:
        node_features = bundle.node_feats[int(dataset_index)]
        atom_counts.update(int(value) for value in node_features[:, 0])
        for values in bundle.edge_feats[int(dataset_index)].values():
            if values.size:
                bond_counts[int(values[0])] += 1

    feature_tolerance = float(audit_cfg["feature_tolerance"])
    prediction_tolerance = float(audit_cfg["prediction_tolerance"])
    relabel_pass = max(typed_max, default=0.0) <= feature_tolerance and float(
        prediction_delta.max(initial=0.0)
    ) <= prediction_tolerance
    scope_pass = truncated_patches == 0 or max(scope_relative, default=0.0) <= feature_tolerance
    atom_collision = _collision_summary(atom_counts, int(rep_cfg["atom_bins"]))
    bond_collision = _collision_summary(bond_counts, int(rep_cfg["bond_bins"]))
    category_pass = atom_collision["n_colliding_bins"] == 0 and bond_collision["n_colliding_bins"] == 0

    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None

    result = {
        "protocol_id": config["protocol_id"],
        "config": config,
        "audit_boundary": {
            "official_test_evaluated": False,
            "model_selection_performed": False,
            "frozen_feature_sha256": _sha256(feature_path),
            "frozen_search_sha256": _sha256(search_path),
            "config_sha256": _sha256(config_path),
            "git_head": git_head,
        },
        "sample": {
            "split": data_cfg["split"],
            "n_graphs": int(selected_rows.size),
            "n_positive": int(labels[selected_rows].sum()),
            "n_negative": int(selected_rows.size - labels[selected_rows].sum()),
            "permutations_per_graph": int(audit_cfg["permutations_per_graph"]),
            "dataset_indices_sha256": hashlib.sha256(
                dataset_indices[selected_rows].astype(np.int64).tobytes()
            ).hexdigest(),
        },
        "relabel_invariance": {
            "typed_feature_max_abs": _summary(typed_max),
            "typed_feature_relative_l2": _summary(typed_relative),
            "topology_feature_max_abs": _summary(topology_max),
            "topology_feature_relative_l2": _summary(topology_relative),
            "prediction_abs_difference": _summary(prediction_delta.tolist()),
            "cached_reproduction_max_abs": cached_reproduction_max,
            "worst_comparisons": worst,
        },
        "object_scope": {
            "n_patches": total_patches,
            "n_truncated_patches": truncated_patches,
            "truncated_patch_fraction": float(truncated_patches / max(total_patches, 1)),
            "graphs_with_truncation_fraction": float(
                graphs_with_truncation / max(selected_rows.size, 1)
            ),
            "mean_dropped_nodes_per_patch": float(dropped_nodes / max(total_patches, 1)),
            "mean_dropped_edges_per_patch": float(dropped_edges / max(total_patches, 1)),
            "legacy_vs_matched_relative_l2": _summary(scope_relative),
        },
        "category_encoding": {"atom": atom_collision, "bond": bond_collision},
        "decisions": {
            "relabel_invariance": "PASS" if relabel_pass else "FAIL",
            "matched_object_scope": "PASS" if scope_pass else "FAIL",
            "category_encoding": "PASS" if category_pass else "FAIL",
            "proceed_to_fold_attribution": bool(relabel_pass and scope_pass and category_pass),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = run(config_path)
    output_json = _resolve(config["output_json"])
    output_markdown = _resolve(config["output_markdown"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(_render_markdown(result), encoding="utf-8")
    print(json.dumps({"decisions": result["decisions"], "output": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
