"""Self-tests for statistics-conditioned residual KSVD utilities."""
from __future__ import annotations

import numpy as np

from .imdb_walk_conditional import (
    build_conditional_targets,
    fit_statistics_residualizer,
    predict_statistics_patch_mean,
    run_conditional_fold,
)
from .imdb_walk_dictionary import FoldSplit
from .imdb_walk_substrate import IMDBPatchGraph, TUStructureGraph


def _graph(index: int, label: int) -> TUStructureGraph:
    adjacency = np.zeros((7, 7), dtype=np.int8)
    for node in range(6):
        adjacency[node, node + 1] = adjacency[node + 1, node] = 1
    return TUStructureGraph(index=index, label=label, adjacency=adjacency)


def _example(index: int, label: int, rng: np.random.Generator) -> IMDBPatchGraph:
    count = 4 + index % 3
    base = np.full((count, 21), 0.15 * index, dtype=np.float64)
    values = base + (rng.random((count, 21)) < (0.2 + 0.1 * label)).astype(np.float64)
    return IMDBPatchGraph(
        graph_index=index,
        label=label,
        n_nodes=7,
        n_edges=6,
        n_patches=count,
        root_coverage=1.0,
        walk_vectors=values,
        canonical_vectors=values.copy(),
        edge_counts=np.sum(values, axis=1).astype(np.int64),
        features={"walk_mean_std": np.concatenate([values.mean(axis=0), values.std(axis=0)])},
        graph_statistics=np.asarray(
            [7.0, 6.0 + index, 0.2, 2.0, 0.5, 1.0, 3.0, label, 0.0, 2.0, 5.0, 1.0]
        ),
    )


def main() -> int:
    rng = np.random.default_rng(31)
    examples = tuple(_example(index, index % 2, rng) for index in range(30))
    residualizer = fit_statistics_residualizer(examples[:20])
    predictions = predict_statistics_patch_mean(examples[20:], residualizer)
    assert predictions.shape == (10, 21)
    residual, slices, raw, predicted = build_conditional_targets(examples[20:], residualizer)
    assert residual.shape == raw.shape == predicted.shape
    assert len(slices) == 10

    graphs = tuple(_graph(index, index % 2) for index in range(30))
    outer = FoldSplit(0, tuple(range(24)), tuple(range(24, 30)))
    inner = FoldSplit(0, tuple(range(20)), tuple(range(20, 24)))
    result, dictionaries = run_conditional_fold(
        graphs,
        examples,
        outer,
        inner,
        n_atoms=4,
        sparsity=2,
        minimum_sparsity=1,
        n_iterations=2,
        graph_shuffle_seed=731623,
        label_shuffle_seed=731633,
    )
    assert dictionaries["standard_init"].shape == (21, 4)
    assert dictionaries["residual_final"].shape == (21, 4)
    assert result["residualizer"]["stats_active_dimension"] > 0
    assert result["config"]["residual_target"].startswith("raw_patch_minus")
    assert result["full_patch_reconstruction"]["test"][
        "residual_final_graph_balanced_mean_relative_error"
    ] >= 0.0
    assert "stats_plus_residual_final" in result["evaluations"]
    assert 0.0 <= result["evaluations"]["stats_plus_residual_final"]["test_balanced_accuracy"] <= 1.0
    assert result["stages"]["residual"]["final"]["test"]["dictionary_health"]["minimum_nonzeros_per_patch"] >= 1

    print("imdb_walk_conditional self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
