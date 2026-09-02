"""Train-only screen for invariant patch-level structure--attribute fusion.

This protocol isolates the mechanism behind the historical 52-D typed patch
result without retaining its node-ID ordered adjacency slots.  Every atom-
centred radius-2 patch is represented by four permutation-invariant blocks:

* topology-only rooted-WL node and edge role marginals;
* strict atom and bond attribute marginals.

The blocks are fused *before* graph readout.  The screen compares shared L2
normalisation, a compact bilinear binding, PCA reconstruction, and matched
INIT/FINAL K-SVD reconstruction.  A deterministic within-graph permutation of
whole attribute patch rows preserves the separate structure and attribute
distributions while breaking their patch-level co-occurrence.

All fitted transforms are learned independently inside each official-train
scaffold fold.  Official validation and test are never encoded or evaluated.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.decomposition import PCA, sparse_encode

from ksvd_research.core import ksvd
from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.patch_object_audit import (
    relabel_graph_features,
)
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    REPO_ROOT,
    _aggregate_folds,
    _delta,
    _fit_auc_views,
    _fold_indices,
    _frozen_s_rows,
    _resolve,
    _sha256,
    _summary,
    _write_json,
)
from tracks.ksvd.experiments.luyin16.structural_role_fusion_screen import (
    BOND_DIM,
    STRICT_ATOM_DIM,
    patch_feature_rows,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/invariant_joint_reconstruction_screen.yaml"
)
LOCAL_BLOCKS = ("node_role", "edge_role", "node_attribute", "edge_attribute")
READOUT_BLOCKS = ("mean", "std")


def _representation_dimensions(config: Mapping[str, Any]) -> dict[str, int]:
    return {
        "node_role": int(config["node_role_bins"]),
        "edge_role": int(config["edge_role_bins"]),
        "node_attribute": STRICT_ATOM_DIM,
        "edge_attribute": BOND_DIM,
    }


def _block_slices(config: Mapping[str, Any]) -> dict[str, slice]:
    dimensions = _representation_dimensions(config)
    output: dict[str, slice] = {}
    start = 0
    for name in LOCAL_BLOCKS:
        stop = start + dimensions[name]
        output[name] = slice(start, stop)
        start = stop
    return output


def _balanced_rows(rows: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Give each categorical block comparable mass before shared L2 norm."""
    values = np.asarray(rows, dtype=np.float32).copy()
    slices = _block_slices(config)
    # strict_atom_semantics concatenates seven one-hot fields; compact bond
    # semantics concatenates three.  Dividing by those fixed masses makes all
    # four patch blocks sum to one when the corresponding entities exist.
    values[:, slices["node_attribute"]] /= 7.0
    values[:, slices["edge_attribute"]] /= 3.0
    return values


def _shared_l2_rows(rows: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    values = _balanced_rows(rows, config)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1.0e-12)


def _mean_std_readout(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected [n_patches,width], got {values.shape}")
    if values.shape[0] == 0:
        return np.zeros(2 * values.shape[1], dtype=np.float32)
    return np.concatenate(
        [values.mean(axis=0), values.std(axis=0)]
    ).astype(np.float32, copy=False)


def _lexicographic_order(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("lexicographic row ordering expects a matrix")
    if values.shape[0] <= 1:
        return np.arange(values.shape[0], dtype=np.int64)
    keys = tuple(values[:, column] for column in range(values.shape[1] - 1, -1, -1))
    return np.asarray(np.lexsort(keys), dtype=np.int64)


def _shuffle_patch_attributes(
    rows: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    """Break patch-level S/A pairing while preserving both row multisets.

    Sorting both modalities first makes the fixed random permutation invariant
    to arbitrary node relabelling, which only changes the order of centres.
    """
    values = np.asarray(rows, dtype=np.float32)
    slices = _block_slices(config)
    structure = values[:, : slices["node_attribute"].start]
    attributes = values[:, slices["node_attribute"].start :]
    if values.shape[0] <= 1:
        return values.copy()
    structure = structure[_lexicographic_order(structure)]
    attributes = attributes[_lexicographic_order(attributes)]
    rng = np.random.default_rng(int(seed))
    attributes = attributes[rng.permutation(attributes.shape[0])]
    return np.concatenate([structure, attributes], axis=1).astype(
        np.float32, copy=False
    )


def _patch_rows_for_graph(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    representation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    blocks = patch_feature_rows(
        graph,
        node_features,
        edge_features,
        "rooted_wl",
        representation,
    )
    rows = np.concatenate([blocks[name] for name in LOCAL_BLOCKS], axis=1)
    context = np.asarray(blocks["context"], dtype=np.float32).reshape(-1)
    expected = sum(_representation_dimensions(representation).values())
    if rows.shape[1] != expected:
        raise RuntimeError(f"joint patch width changed: {rows.shape[1]} != {expected}")
    return rows.astype(np.float32, copy=False), context


def _cache_signature(
    indices: np.ndarray,
    representation: Mapping[str, Any],
) -> str:
    encoder = Path(__file__).resolve().parent / "structural_role_fusion_screen.py"
    payload = {
        "schema": "rooted_wl_strict_patch_marginals_v1",
        "indices": np.asarray(indices, dtype=np.int64).tolist(),
        "representation": dict(representation),
        "encoder_sha256": _sha256(encoder),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _build_patch_cache(
    bundle,
    indices: np.ndarray,
    representation: Mapping[str, Any],
    cache_path: Path,
) -> tuple[dict[str, np.ndarray], bool]:
    selected = np.asarray(indices, dtype=np.int64)
    signature = _cache_signature(selected, representation)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_signature = str(np.asarray(archive["signature"]).reshape(-1)[0])
            if cached_signature != signature:
                raise ValueError(f"patch cache signature mismatch: {cache_path}")
            return {name: np.asarray(archive[name]) for name in archive.files}, True
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("typed patch extraction requires node and edge features")
    rows: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    offsets = [0]
    for position, raw_index in enumerate(selected):
        index = int(raw_index)
        local, context = _patch_rows_for_graph(
            bundle.graphs[index],
            bundle.node_feats[index],
            bundle.edge_feats[index],
            representation,
        )
        rows.append(local)
        contexts.append(context)
        offsets.append(offsets[-1] + local.shape[0])
        if position and position % 500 == 0:
            print(f"patch cache graphs: {position}/{selected.size}", flush=True)
    payload = {
        "signature": np.asarray([signature]),
        "dataset_indices": selected,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "rows": np.concatenate(rows, axis=0).astype(np.float32, copy=False),
        "context": np.stack(contexts, axis=0).astype(np.float32, copy=False),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(cache_path)
    return payload, False


def _cache_graph_rows(cache: Mapping[str, np.ndarray], position: int) -> np.ndarray:
    offsets = np.asarray(cache["offsets"], dtype=np.int64)
    start, stop = int(offsets[position]), int(offsets[position + 1])
    return np.asarray(cache["rows"][start:stop], dtype=np.float32)


def _batch_for_positions(
    cache: Mapping[str, np.ndarray],
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lengths: list[int] = []
    rows: list[np.ndarray] = []
    for position in np.asarray(positions, dtype=np.int64):
        current = _cache_graph_rows(cache, int(position))
        rows.append(current)
        lengths.append(current.shape[0])
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
    )
    return np.concatenate(rows, axis=0), offsets


def _shuffled_batch(
    cache: Mapping[str, np.ndarray],
    positions: np.ndarray,
    dataset_indices: np.ndarray,
    representation: Mapping[str, Any],
    *,
    base_seed: int,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    lengths: list[int] = []
    for position in np.asarray(positions, dtype=np.int64):
        index = int(dataset_indices[int(position)])
        current = _shuffle_patch_attributes(
            _cache_graph_rows(cache, int(position)),
            representation,
            seed=int(base_seed) + 1000003 * (int(repeat) + 1) + 7919 * index,
        )
        rows.append(current)
        lengths.append(current.shape[0])
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
    )
    return np.concatenate(rows, axis=0), offsets


def _readout_batch(rows: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            _mean_std_readout(rows[int(offsets[i]) : int(offsets[i + 1])])
            for i in range(offsets.size - 1)
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _sample_pool(rows: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.shape[0] <= int(maximum):
        return values
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(values.shape[0], size=int(maximum), replace=False))
    return values[chosen]


def _sparse_reconstruction(
    rows: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    n_jobs: int,
) -> np.ndarray:
    codes = sparse_encode(
        np.asarray(rows, dtype=np.float64),
        np.asarray(dictionary, dtype=np.float64).T,
        algorithm="omp",
        n_nonzero_coefs=int(sparsity),
        n_jobs=int(n_jobs),
    )
    return (codes @ np.asarray(dictionary, dtype=np.float64).T).astype(
        np.float32, copy=False
    )


def _relative_error(original: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(original) - np.asarray(reconstruction))
        / max(np.linalg.norm(np.asarray(original)), 1.0e-12)
    )


def _fit_transforms(
    train_joint_rows: np.ndarray,
    representation: Mapping[str, Any],
    transform_config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    pool = _sample_pool(
        train_joint_rows,
        int(transform_config["max_train_patches"]),
        int(seed),
    )
    pca_rank = min(
        int(transform_config["pca_rank"]), pool.shape[0], pool.shape[1]
    )
    pca = PCA(
        n_components=pca_rank,
        svd_solver="randomized",
        random_state=int(seed),
    ).fit(pool)

    slices = _block_slices(representation)
    structure_stop = int(slices["node_attribute"].start)
    structure_pool = pool[:, :structure_stop]
    attribute_pool = pool[:, structure_stop:]
    structure_rank = min(
        int(transform_config["bilinear_structure_rank"]),
        structure_pool.shape[0],
        structure_pool.shape[1],
    )
    attribute_rank = min(
        int(transform_config["bilinear_attribute_rank"]),
        attribute_pool.shape[0],
        attribute_pool.shape[1],
    )
    structure_pca = PCA(
        n_components=structure_rank,
        svd_solver="randomized",
        random_state=int(seed) + 1,
    ).fit(structure_pool)
    attribute_pca = PCA(
        n_components=attribute_rank,
        svd_solver="randomized",
        random_state=int(seed) + 2,
    ).fit(attribute_pool)

    n_atoms = int(transform_config["ksvd_atoms"])
    sparsity = int(transform_config["ksvd_sparsity"])
    initial, _, initial_info = ksvd(
        np.asarray(pool, dtype=np.float64).T,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=int(transform_config["ksvd_minimum_sparsity"]),
        n_iter=0,
        seed=int(seed),
    )
    final, _, final_info = ksvd(
        np.asarray(pool, dtype=np.float64).T,
        n_atoms=initial.shape[1],
        T=sparsity,
        T_min=int(transform_config["ksvd_minimum_sparsity"]),
        n_iter=int(transform_config["ksvd_iterations"]),
        seed=int(seed),
        initial_dictionary=initial,
    )
    return {
        "pool_size": int(pool.shape[0]),
        "pca": pca,
        "structure_pca": structure_pca,
        "attribute_pca": attribute_pca,
        "dictionary_initial": initial,
        "dictionary_final": final,
        "dictionary_initial_info": initial_info,
        "dictionary_final_info": final_info,
    }


def _bilinear_rows(
    joint_rows: np.ndarray,
    structure_pca: PCA,
    attribute_pca: PCA,
    representation: Mapping[str, Any],
) -> np.ndarray:
    slices = _block_slices(representation)
    structure_stop = int(slices["node_attribute"].start)
    structure = structure_pca.transform(joint_rows[:, :structure_stop])
    attribute = attribute_pca.transform(joint_rows[:, structure_stop:])
    return np.einsum("ni,nj->nij", structure, attribute).reshape(
        joint_rows.shape[0], -1
    ).astype(np.float32, copy=False)


def _audit_patch_multiset_invariance(
    bundle,
    indices: np.ndarray,
    representation: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("invariance audit requires typed graph features")
    rng = np.random.default_rng(int(audit["seed"]))
    candidates = np.asarray(indices, dtype=np.int64)
    if candidates.size > int(audit["n_graphs"]):
        candidates = np.sort(
            rng.choice(candidates, size=int(audit["n_graphs"]), replace=False)
        )
    true_drift: list[float] = []
    shuffled_drift: list[float] = []
    for raw_index in candidates:
        index = int(raw_index)
        base, _ = _patch_rows_for_graph(
            bundle.graphs[index],
            bundle.node_feats[index],
            bundle.edge_feats[index],
            representation,
        )
        base_sorted = base[_lexicographic_order(base)]
        base_shuffled = _shuffle_patch_attributes(
            base,
            representation,
            seed=int(audit["seed"]) + 7919 * index,
        )
        base_shuffled = base_shuffled[_lexicographic_order(base_shuffled)]
        for _ in range(int(audit["permutations_per_graph"])):
            permutation = rng.permutation(bundle.graphs[index].n)
            changed_graph, changed_nodes, changed_edges = relabel_graph_features(
                bundle.graphs[index],
                bundle.node_feats[index],
                bundle.edge_feats[index],
                permutation,
            )
            changed, _ = _patch_rows_for_graph(
                changed_graph,
                changed_nodes,
                changed_edges,
                representation,
            )
            changed_sorted = changed[_lexicographic_order(changed)]
            changed_shuffled = _shuffle_patch_attributes(
                changed,
                representation,
                seed=int(audit["seed"]) + 7919 * index,
            )
            changed_shuffled = changed_shuffled[
                _lexicographic_order(changed_shuffled)
            ]
            true_drift.append(
                float(np.max(np.abs(base_sorted - changed_sorted), initial=0.0))
            )
            shuffled_drift.append(
                float(
                    np.max(
                        np.abs(base_shuffled - changed_shuffled), initial=0.0
                    )
                )
            )
    tolerance = float(audit["tolerance"])
    return {
        "n_graphs": int(candidates.size),
        "permutations_per_graph": int(audit["permutations_per_graph"]),
        "true_multiset_drift": _summary(true_drift),
        "shuffled_multiset_drift": _summary(shuffled_drift),
        "tolerance": tolerance,
        "pass": bool(
            max(true_drift or [0.0]) <= tolerance
            and max(shuffled_drift or [0.0]) <= tolerance
        ),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MolHIV invariant joint reconstruction screen",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "Official-train scaffold folds only; fixed XGBoost; official validation/test not encoded.",
        "",
        f"Patch multiset relabel invariance: **{'PASS' if result['object_audit']['pass'] else 'FAIL'}**",
        "",
        "| view | mean validation ROC-AUC | fold std |",
        "|---|---:|---:|",
    ]
    for name, row in result["aggregate"].items():
        lines.append(
            f"| `{name}` | {row['mean_auc']:.6f} | {row['std_auc']:.6f} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, gate in result["gates"].items():
        lines.append(
            f"- `{name}`: {gate['mean_delta']:+.6f}, wins "
            f"{gate['fold_wins']}/{result['n_folds']} — "
            f"**{'PASS' if gate['passed'] else 'FAIL'}**"
        )
    lines.extend(["", f"Decision: **{result['decision']}**", ""])
    return "\n".join(lines)


def _view(
    frozen_s: np.ndarray,
    local: np.ndarray,
    context: np.ndarray,
) -> np.ndarray:
    return np.concatenate([frozen_s, local, context], axis=1).astype(
        np.float32, copy=False
    )


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    representation = config["representation"]
    screen = config["screen"]
    classifier = config["classifier"]
    transform_config = config["transforms"]
    result_json = _resolve(config["output_json"])
    result_markdown = _resolve(config["output_markdown"])
    cache_path = _resolve(config["patch_cache"])
    frozen_path = _resolve(data["frozen_features"])
    folds_path = _resolve(data["scaffold_folds"])
    with np.load(frozen_path, allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    labels = np.asarray(bundle.y, dtype=np.int64)
    start = time.perf_counter()

    fold_ids = sorted(
        int(name.removeprefix("fold_").removesuffix("_train_indices"))
        for name in fold_archive
        if name.startswith("fold_") and name.endswith("_train_indices")
    )
    selections = [
        _fold_indices(fold_archive, fold, labels, screen) for fold in fold_ids
    ]
    selected_union = np.unique(
        np.concatenate([np.concatenate(pair) for pair in selections])
    ).astype(np.int64)
    print(
        f"selected graphs={selected_union.size}; folds={fold_ids}; "
        f"patch representation width={sum(_representation_dimensions(representation).values())}",
        flush=True,
    )
    original = np.asarray(fold_archive["original_indices"], dtype=np.int64)
    audit_candidates = original[
        np.asarray(fold_archive["official_train_indices"], dtype=np.int64)
    ]
    object_audit = _audit_patch_multiset_invariance(
        bundle,
        audit_candidates,
        representation,
        config["audit"],
    )
    if not object_audit["pass"]:
        raise RuntimeError("joint patch object failed relabel invariance")
    cache, cache_hit = _build_patch_cache(
        bundle, selected_union, representation, cache_path
    )
    cache_indices = np.asarray(cache["dataset_indices"], dtype=np.int64)
    index_to_position = {
        int(index): position for position, index in enumerate(cache_indices)
    }
    shuffle_repeats = int(screen["shuffle_repeats"])
    model_seeds = [int(value) for value in classifier.get("model_seeds", [0])]
    folds: list[dict[str, Any]] = []
    transform_rows: list[dict[str, Any]] = []

    for fold, (train_indices, valid_indices) in zip(
        fold_ids, selections, strict=True
    ):
        fold_indices = np.concatenate([train_indices, valid_indices]).astype(
            np.int64
        )
        positions = np.asarray(
            [index_to_position[int(index)] for index in fold_indices],
            dtype=np.int64,
        )
        train_positions = positions[: train_indices.size]
        true_rows, offsets = _batch_for_positions(cache, positions)
        train_rows, _ = _batch_for_positions(cache, train_positions)
        true_joint = _shared_l2_rows(true_rows, representation)
        train_joint = _shared_l2_rows(train_rows, representation)
        transforms = _fit_transforms(
            train_joint,
            representation,
            transform_config,
            seed=int(transform_config["seed"]) + 1009 * int(fold),
        )
        context = np.asarray(cache["context"][positions], dtype=np.float32)
        frozen_s = _frozen_s_rows(frozen, fold_indices)
        marginal_readout = _readout_batch(true_rows, offsets)
        joint_readout = _readout_batch(true_joint, offsets)
        bilinear = _bilinear_rows(
            true_joint,
            transforms["structure_pca"],
            transforms["attribute_pca"],
            representation,
        )
        bilinear_readout = _readout_batch(bilinear, offsets)
        pca_reconstruction = transforms["pca"].inverse_transform(
            transforms["pca"].transform(true_joint)
        ).astype(np.float32, copy=False)
        pca_readout = _readout_batch(pca_reconstruction, offsets)
        initial_reconstruction = _sparse_reconstruction(
            true_joint,
            transforms["dictionary_initial"],
            int(transform_config["ksvd_sparsity"]),
            int(transform_config["n_jobs"]),
        )
        final_reconstruction = _sparse_reconstruction(
            true_joint,
            transforms["dictionary_final"],
            int(transform_config["ksvd_sparsity"]),
            int(transform_config["n_jobs"]),
        )
        initial_readout = _readout_batch(initial_reconstruction, offsets)
        final_readout = _readout_batch(final_reconstruction, offsets)

        views: dict[str, np.ndarray] = {
            "s": frozen_s,
            "s_marginal_mean_std": _view(
                frozen_s, marginal_readout, context
            ),
            "s_joint_l2": _view(frozen_s, joint_readout, context),
            "s_bilinear": _view(
                frozen_s,
                np.concatenate([marginal_readout, bilinear_readout], axis=1),
                context,
            ),
            "s_pca_reconstruction": _view(frozen_s, pca_readout, context),
            "s_ksvd_initial": _view(frozen_s, initial_readout, context),
            "s_ksvd_final": _view(frozen_s, final_readout, context),
        }
        for repeat in range(shuffle_repeats):
            shuffled_rows, shuffled_offsets = _shuffled_batch(
                cache,
                positions,
                cache_indices,
                representation,
                base_seed=int(screen["seed"]),
                repeat=repeat,
            )
            if not np.array_equal(offsets, shuffled_offsets):
                raise RuntimeError("shuffle changed graph patch counts")
            shuffled_joint = _shared_l2_rows(shuffled_rows, representation)
            shuffled_joint_readout = _readout_batch(
                shuffled_joint, shuffled_offsets
            )
            shuffled_bilinear = _bilinear_rows(
                shuffled_joint,
                transforms["structure_pca"],
                transforms["attribute_pca"],
                representation,
            )
            shuffled_bilinear_readout = _readout_batch(
                shuffled_bilinear, shuffled_offsets
            )
            shuffled_pca = transforms["pca"].inverse_transform(
                transforms["pca"].transform(shuffled_joint)
            ).astype(np.float32, copy=False)
            shuffled_initial = _sparse_reconstruction(
                shuffled_joint,
                transforms["dictionary_initial"],
                int(transform_config["ksvd_sparsity"]),
                int(transform_config["n_jobs"]),
            )
            shuffled_final = _sparse_reconstruction(
                shuffled_joint,
                transforms["dictionary_final"],
                int(transform_config["ksvd_sparsity"]),
                int(transform_config["n_jobs"]),
            )
            views[f"s_joint_l2_shuffled_{repeat}"] = _view(
                frozen_s, shuffled_joint_readout, context
            )
            views[f"s_bilinear_shuffled_{repeat}"] = _view(
                frozen_s,
                np.concatenate(
                    [marginal_readout, shuffled_bilinear_readout], axis=1
                ),
                context,
            )
            views[f"s_pca_reconstruction_shuffled_{repeat}"] = _view(
                frozen_s, _readout_batch(shuffled_pca, shuffled_offsets), context
            )
            views[f"s_ksvd_initial_shuffled_{repeat}"] = _view(
                frozen_s,
                _readout_batch(shuffled_initial, shuffled_offsets),
                context,
            )
            views[f"s_ksvd_final_shuffled_{repeat}"] = _view(
                frozen_s,
                _readout_batch(shuffled_final, shuffled_offsets),
                context,
            )

        scores, by_seed = _fit_auc_views(
            views,
            labels[fold_indices],
            len(train_indices),
            classifier,
            model_seeds,
        )
        folds.append(
            {
                "fold": int(fold),
                "n_train": int(train_indices.size),
                "n_valid": int(valid_indices.size),
                "scores": scores,
                "scores_by_model_seed": by_seed,
            }
        )
        valid_start = int(offsets[int(train_indices.size)])
        transform_rows.append(
            {
                "fold": int(fold),
                "train_patch_pool": int(transforms["pool_size"]),
                "pca_explained_variance_ratio": float(
                    np.sum(transforms["pca"].explained_variance_ratio_)
                ),
                "pca_valid_reconstruction_error": _relative_error(
                    true_joint[valid_start:], pca_reconstruction[valid_start:]
                ),
                "ksvd_initial_info": transforms["dictionary_initial_info"],
                "ksvd_final_info": transforms["dictionary_final_info"],
                "ksvd_initial_valid_reconstruction_error": _relative_error(
                    true_joint[valid_start:], initial_reconstruction[valid_start:]
                ),
                "ksvd_final_valid_reconstruction_error": _relative_error(
                    true_joint[valid_start:], final_reconstruction[valid_start:]
                ),
            }
        )
        print(
            f"fold {fold}: marginal={scores['s_marginal_mean_std']['valid_auc']:.6f}; "
            f"bilinear={scores['s_bilinear']['valid_auc']:.6f}; "
            f"pca={scores['s_pca_reconstruction']['valid_auc']:.6f}; "
            f"ksvd_init/final={scores['s_ksvd_initial']['valid_auc']:.6f}/"
            f"{scores['s_ksvd_final']['valid_auc']:.6f}",
            flush=True,
        )

    aggregate = {
        name: _aggregate_folds(folds, name) for name in folds[0]["scores"]
    }
    gates = {
        "joint_l2_vs_marginal": _delta(
            folds, "s_joint_l2", "s_marginal_mean_std"
        ),
        "bilinear_vs_marginal": _delta(
            folds, "s_bilinear", "s_marginal_mean_std"
        ),
        "pca_vs_joint_l2": _delta(
            folds, "s_pca_reconstruction", "s_joint_l2"
        ),
        "ksvd_initial_vs_joint_l2": _delta(
            folds, "s_ksvd_initial", "s_joint_l2"
        ),
        "ksvd_final_vs_joint_l2": _delta(
            folds, "s_ksvd_final", "s_joint_l2"
        ),
        "ksvd_final_vs_initial": _delta(
            folds, "s_ksvd_final", "s_ksvd_initial"
        ),
        "ksvd_final_vs_pca": _delta(
            folds, "s_ksvd_final", "s_pca_reconstruction"
        ),
    }
    for name, true_view in (
        ("joint_l2_vs_shuffle", "s_joint_l2"),
        ("bilinear_vs_shuffle", "s_bilinear"),
        ("pca_vs_shuffle", "s_pca_reconstruction"),
        ("ksvd_initial_vs_shuffle", "s_ksvd_initial"),
        ("ksvd_final_vs_shuffle", "s_ksvd_final"),
    ):
        controls = [
            f"{true_view}_shuffled_{repeat}" for repeat in range(shuffle_repeats)
        ]
        fold_deltas = []
        for row in folds:
            candidate = float(row["scores"][true_view]["valid_auc"])
            baseline = float(
                np.mean(
                    [row["scores"][control]["valid_auc"] for control in controls]
                )
            )
            fold_deltas.append(candidate - baseline)
        gates[name] = {
            "candidate": true_view,
            "baseline": "mean(" + ",".join(controls) + ")",
            "fold_deltas": fold_deltas,
            "mean_delta": float(np.mean(fold_deltas)),
            "fold_wins": int(np.sum(np.asarray(fold_deltas) > 0.0)),
        }
    minimum_delta = float(screen["minimum_delta"])
    minimum_wins = int(screen["minimum_fold_wins"])
    for gate in gates.values():
        gate["minimum_mean_delta"] = minimum_delta
        gate["minimum_fold_wins"] = minimum_wins
        gate["passed"] = bool(
            gate["mean_delta"] >= minimum_delta
            and gate["fold_wins"] >= minimum_wins
        )

    binding_pass = bool(
        gates["bilinear_vs_marginal"]["passed"]
        and gates["bilinear_vs_shuffle"]["passed"]
    )
    reconstruction_pass = bool(
        (
            gates["pca_vs_joint_l2"]["passed"]
            and gates["pca_vs_shuffle"]["passed"]
        )
        or (
            gates["ksvd_initial_vs_joint_l2"]["passed"]
            and gates["ksvd_initial_vs_shuffle"]["passed"]
        )
        or (
            gates["ksvd_final_vs_joint_l2"]["passed"]
            and gates["ksvd_final_vs_shuffle"]["passed"]
        )
    )
    ksvd_update_pass = bool(gates["ksvd_final_vs_initial"]["passed"])
    if binding_pass and reconstruction_pass and ksvd_update_pass:
        decision = "BINDING_AND_RECONSTRUCTION_PASS_KSVD_UPDATE_PASS"
    elif binding_pass and reconstruction_pass:
        decision = "BINDING_AND_GENERIC_RECONSTRUCTION_PASS_KSVD_UPDATE_FAIL"
    elif binding_pass:
        decision = "LOW_RANK_BINDING_ONLY_PASS"
    elif reconstruction_pass and ksvd_update_pass:
        decision = "RECONSTRUCTION_PASS_KSVD_UPDATE_PASS"
    elif reconstruction_pass:
        decision = "GENERIC_RECONSTRUCTION_ONLY_PASS"
    else:
        decision = "INVARIANT_PATCH_EARLY_FUSION_NO_GO"

    result = {
        "protocol_id": str(config["protocol_id"]),
        "config": config,
        "data": {
            "dataset": str(data["dataset"]),
            "selected_graphs": int(selected_union.size),
            "split": "official-train-only scaffold folds",
            "official_validation_encoded_or_evaluated": False,
            "official_test_encoded_or_evaluated": False,
            "patch_cache": str(cache_path),
            "patch_cache_hit": bool(cache_hit),
        },
        "audit_boundary": {
            "config_sha256": _sha256(config_path),
            "frozen_s_sha256": _sha256(frozen_path),
            "scaffold_folds_sha256": _sha256(folds_path),
            "roles_use_attributes": False,
            "strict_atom_excludes": ["degree", "is_in_ring"],
            "transform_fit": "independently on each outer-train patch pool",
            "shuffle": "whole attribute patch rows permuted within graph after invariant sorting",
        },
        "representation": {
            "radius": int(representation["radius"]),
            "centres": "all graph nodes",
            "patch_blocks": list(LOCAL_BLOCKS),
            "patch_dimensions": _representation_dimensions(representation),
            "readout": list(READOUT_BLOCKS),
            "shared_normalization": "attribute field-mass balancing then whole-patch L2",
            "bilinear": "outer product of train-only structure-PCA and attribute-PCA patch scores",
        },
        "object_audit": object_audit,
        "n_folds": len(fold_ids),
        "folds": folds,
        "aggregate": aggregate,
        "transform_diagnostics": transform_rows,
        "gates": gates,
        "decision": decision,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_json(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "gates": result["gates"],
                "runtime_seconds": result["runtime"]["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
