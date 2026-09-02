"""Self-tests for IMDB WALK R0-A downstream attribution utilities."""
from __future__ import annotations

import numpy as np

from .imdb_walk_dictionary import FoldSplit
from .imdb_walk_downstream import (
    _permute_rows,
    make_inner_train_validation_split,
    run_imdb_downstream_fold,
    variable_graph_pair_readout,
    variable_graph_code_readout,
)
from .imdb_walk_substrate import IMDBPatchGraph, IsomorphismGroup, TUStructureGraph


def _graph(index: int, label: int) -> TUStructureGraph:
    n = 7
    adjacency = np.zeros((n, n), dtype=np.int8)
    for node in range(n - 1):
        adjacency[node, node + 1] = adjacency[node + 1, node] = 1
    if label:
        adjacency[0, 2] = adjacency[2, 0] = 1
    return TUStructureGraph(index=index, label=label, adjacency=adjacency)


def _example(index: int, label: int, rng: np.random.Generator) -> IMDBPatchGraph:
    count = 5 + index % 4
    values = (rng.random((count, 21)) < (0.25 + 0.25 * label)).astype(np.float64)
    stats = np.asarray(
        [7.0, 6.0 + label, 0.2, 2.0, 0.5, 1.0, 3.0, label, 0.0, 2.0, 5.0, 1.0]
    )
    walk_summary = np.concatenate([values.mean(axis=0), values.std(axis=0)])
    return IMDBPatchGraph(
        graph_index=index,
        label=label,
        n_nodes=7,
        n_edges=int(6 + label),
        n_patches=count,
        root_coverage=1.0,
        walk_vectors=values,
        canonical_vectors=values.copy(),
        edge_counts=np.sum(values, axis=1).astype(np.int64),
        features={"walk_mean_std": walk_summary},
        graph_statistics=stats,
    )


def main() -> int:
    codes = np.asarray(
        [[1.0, 0.0, 2.0, 0.0, 0.0], [0.0, -2.0, 0.0, 3.0, 4.0]],
        dtype=np.float64,
    )
    pooled = variable_graph_code_readout(
        codes, ((3, 0, 0, 2), (7, 1, 2, 5))
    )
    assert pooled.shape == (2, 6)
    assert np.allclose(pooled[0, :2], [0.5, 0.5])
    assert np.allclose(pooled[1, :2], [1.0 / 3.0, 2.0 / 3.0])

    pair_codes = np.asarray(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    pair_pooled = variable_graph_pair_readout(
        pair_codes, ((3, 0, 0, 2), (7, 1, 2, 3))
    )
    assert pair_pooled.shape == (2, 3)
    assert np.allclose(pair_pooled[0], [0.5, 0.5, 0.0])
    assert np.allclose(pair_pooled[1], [0.0, 0.0, 1.0])

    rows = np.arange(300, dtype=np.int64).reshape(100, 3)
    shuffled_a = _permute_rows(rows, 731421)
    shuffled_b = _permute_rows(rows, 731421)
    assert np.array_equal(shuffled_a, shuffled_b)
    assert not np.array_equal(shuffled_a, rows)
    assert sorted(map(tuple, shuffled_a.tolist())) == sorted(map(tuple, rows.tolist()))

    graphs = tuple(_graph(index, index % 2) for index in range(50))
    outer = FoldSplit(
        fold_index=2,
        train_indices=tuple(range(40)),
        test_indices=tuple(range(40, 50)),
    )
    strat_inner_a, strat_audit_a = make_inner_train_validation_split(
        graphs, outer, groups=None, grouped=False, seed=731413
    )
    strat_inner_b, strat_audit_b = make_inner_train_validation_split(
        graphs, outer, groups=None, grouped=False, seed=731413
    )
    assert strat_inner_a == strat_inner_b
    assert strat_audit_a == strat_audit_b
    assert len(strat_inner_a.train_indices) == 32
    assert len(strat_inner_a.test_indices) == 8
    assert strat_audit_a["validation_class_counts"] == {"0": 4, "1": 4}

    grouped_graphs = tuple(_graph(index, index % 2) for index in range(60))
    grouped_outer = FoldSplit(
        fold_index=0,
        train_indices=tuple(range(50)),
        test_indices=tuple(range(50, 60)),
    )
    groups = tuple(
        IsomorphismGroup(
            group_id=group_id,
            member_indices=(2 * group_id, 2 * group_id + 1),
            labels=(0, 1),
            signature=f"g{group_id}",
        )
        for group_id in range(30)
    )
    grouped_inner, grouped_audit = make_inner_train_validation_split(
        grouped_graphs, grouped_outer, groups=groups, grouped=True, seed=731411
    )
    assert grouped_audit["full_inner_partition_audit"]["total_group_leakage_count"] == 0
    validation = set(grouped_inner.test_indices)
    for group in groups[:25]:
        assert set(group.member_indices).issubset(validation) or set(group.member_indices).isdisjoint(validation)

    rng = np.random.default_rng(19)
    examples = tuple(_example(index, index % 2, rng) for index in range(50))
    result, dictionaries = run_imdb_downstream_fold(
        graphs,
        examples,
        outer,
        strat_inner_a,
        n_atoms=4,
        sparsity=2,
        minimum_sparsity=1,
        n_iterations=2,
        graph_shuffle_seed=731423,
        label_shuffle_seed=731433,
    )
    assert dictionaries["init"].shape == (21, 4)
    assert dictionaries["final"].shape == (21, 4)
    expected = {
        "stats", "init", "final", "stats_plus_init", "stats_plus_final",
        "stats_plus_raw_walk", "stats_plus_pca12", "stats_plus_medoid_bag",
        "stats_plus_fixed_gaussian", "stats_plus_shuffled_final",
        "label_shuffle_stats_plus_final",
    }
    assert set(result["evaluations"]) == expected
    assert result["config"]["graph_code_dimension"] == 12
    for evaluation in result["evaluations"].values():
        assert 0.0 <= evaluation["test_balanced_accuracy"] <= 1.0
    assert np.isclose(
        result["attribution"]["update_gain"],
        result["evaluations"]["stats_plus_final"]["test_balanced_accuracy"]
        - result["evaluations"]["stats_plus_init"]["test_balanced_accuracy"],
    )

    pair_result, _pair_dictionaries = run_imdb_downstream_fold(
        graphs,
        examples,
        outer,
        strat_inner_a,
        n_atoms=4,
        sparsity=2,
        minimum_sparsity=1,
        n_iterations=2,
        graph_shuffle_seed=731423,
        label_shuffle_seed=731433,
        include_pair_readout=True,
    )
    assert pair_result["config"]["pair_code_dimension"] == 6
    assert pair_result["evaluations"]["stats_plus_final_pair"]["raw_dimension"] == 12 + 12 + 6
    assert "pair_update_gain" in pair_result["attribution"]
    assert "stats_plus_shuffled_final_pair" in pair_result["evaluations"]
    assert "label_shuffle_stats_plus_final_pair" in pair_result["evaluations"]

    print("imdb_walk_downstream self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
