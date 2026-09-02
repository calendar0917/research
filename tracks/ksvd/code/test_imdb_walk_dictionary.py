"""Self-tests for the IMDB WALK R0-D split and reconstruction utilities."""
from __future__ import annotations

import numpy as np

from .imdb_walk_dictionary import (
    FoldSplit,
    audit_fold_splits,
    graph_balanced_reconstruction_metrics,
    grouped_isomorphism_folds,
    run_imdb_dictionary_fold,
    stratified_graph_folds,
)
from .imdb_walk_substrate import IMDBPatchGraph, IsomorphismGroup, TUStructureGraph


def _structure(index: int, label: int) -> TUStructureGraph:
    adjacency = np.asarray([[0, 1], [1, 0]], dtype=np.int8)
    return TUStructureGraph(index=index, label=label, adjacency=adjacency)


def _patch_graph(index: int, label: int, rng: np.random.Generator) -> IMDBPatchGraph:
    # Non-degenerate 21-D binary-like patches with fold-independent construction.
    patches = (rng.random((5 + index % 3, 21)) > (0.62 - 0.05 * label)).astype(np.float64)
    return IMDBPatchGraph(
        graph_index=index,
        label=label,
        n_nodes=7,
        n_edges=int(patches[0].sum()),
        n_patches=int(patches.shape[0]),
        root_coverage=1.0,
        walk_vectors=patches,
        canonical_vectors=patches.copy(),
        edge_counts=np.sum(patches, axis=1).astype(np.int64),
        features={},
        graph_statistics=np.zeros(7, dtype=np.float64),
    )


def main() -> int:
    graphs = tuple(_structure(index, index % 2) for index in range(20))
    stratified_a = stratified_graph_folds(graphs, n_splits=5, seed=731301)
    stratified_b = stratified_graph_folds(graphs, n_splits=5, seed=731301)
    assert stratified_a == stratified_b
    stratified_audit = audit_fold_splits(graphs, stratified_a)
    assert stratified_audit["passes_partition_gate"]
    for row in stratified_audit["folds"]:
        assert row["test_class_counts"] == {"0": 2, "1": 2}

    groups = tuple(
        IsomorphismGroup(
            group_id=group_id,
            member_indices=(2 * group_id, 2 * group_id + 1),
            labels=(0, 1),
            signature=f"group-{group_id}",
        )
        for group_id in range(10)
    )
    grouped_a = grouped_isomorphism_folds(graphs, groups, n_splits=5, seed=731301)
    grouped_b = grouped_isomorphism_folds(graphs, groups, n_splits=5, seed=731301)
    assert grouped_a == grouped_b
    grouped_audit = audit_fold_splits(graphs, grouped_a, groups=groups)
    assert grouped_audit["passes_partition_gate"]
    assert grouped_audit["total_group_leakage_count"] == 0
    for group in groups:
        containing_folds = [
            fold.fold_index
            for fold in grouped_a
            if set(group.member_indices) & set(fold.test_indices)
        ]
        assert len(containing_folds) == 1

    values = np.asarray([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]])
    reconstruction = values * 0.5
    balanced = graph_balanced_reconstruction_metrics(
        values,
        reconstruction,
        ((0, 0, 0, 2), (1, 1, 2, 4)),
    )
    assert np.isclose(balanced["mean_relative_reconstruction_error"], 0.5)
    assert balanced["class_mean_relative_reconstruction_error"] == {"0": 0.5, "1": 0.5}

    rng = np.random.default_rng(17)
    examples = tuple(_patch_graph(index, index % 2, rng) for index in range(20))
    result, dictionaries = run_imdb_dictionary_fold(
        examples,
        FoldSplit(
            fold_index=0,
            train_indices=tuple(range(15)),
            test_indices=tuple(range(15, 20)),
        ),
        n_atoms=6,
        sparsity=2,
        minimum_sparsity=1,
        n_iterations=2,
    )
    assert result["train_graph_count"] == 15
    assert result["test_graph_count"] == 5
    assert dictionaries["init"].shape == (21, 6)
    assert dictionaries["final"].shape == (21, 6)
    assert dictionaries["pca12"].shape == (21, 6)
    for stage in ("init", "final", "fixed_gaussian", "medoid"):
        for split_name in ("train", "test"):
            stage_result = result["stages"][stage][split_name]
            assert stage_result["graph_balanced"]["graph_count"] == (
                15 if split_name == "train" else 5
            )
            assert stage_result["dictionary_health"]["minimum_nonzeros_per_patch"] >= 1
    for split_name in ("train", "test"):
        assert result["stages"]["pca12"][split_name]["patch_weighted"][
            "relative_reconstruction_error"
        ] >= 0.0

    print("imdb_walk_dictionary self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
