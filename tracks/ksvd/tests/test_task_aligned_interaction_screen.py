from __future__ import annotations

import numpy as np

from tracks.ksvd.experiments.luyin16.task_aligned_interaction_screen import (
    _clip_logit,
    assemble_task_views,
    nested_scaffold_splits,
    rank_interactions,
    rank_stable_interactions,
)


def _blocks(rows: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    dimensions = {
        "node_role": 4,
        "edge_role": 3,
        "node_attribute": 5,
        "edge_attribute": 2,
        "context": 2,
        "node_raw": 20,
        "edge_raw": 6,
        "node_binding": 20,
        "edge_binding": 6,
    }
    result = {
        name: rng.normal(size=(rows, dimension)).astype(np.float32)
        for name, dimension in dimensions.items()
    }
    for name in ("node_raw", "edge_raw", "node_binding", "edge_binding"):
        result[f"{name}_shuffled_0"] = rng.normal(
            size=result[name].shape
        ).astype(np.float32)
    return result


def test_task_views_keep_marginals_outside_interaction_blocks() -> None:
    xgb, interactions = assemble_task_views(_blocks())
    assert xgb["t_a"].shape == (5, 16)
    assert interactions["raw"].shape == (5, 26)
    assert interactions["centered"].shape == (5, 26)
    assert xgb["f_raw_factorized"].shape == (5, 42)
    assert xgb["f_centered"].shape == (5, 42)
    assert not np.array_equal(interactions["raw"], interactions["raw_shuffled"])


def test_nested_scaffold_splits_exclude_outer_group_and_cover_once() -> None:
    indices = np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64)
    mapping = {10: 1, 11: 1, 12: 1, 13: 2, 14: 2, 15: 2, 16: 0}
    labels = np.asarray([0, 1, 0, 0, 1, 0], dtype=np.int64)
    splits, groups = nested_scaffold_splits(
        indices, mapping, held_out_group=0, labels=labels
    )
    assert len(splits) == 2
    coverage = np.zeros(indices.size, dtype=np.int64)
    for train, valid in splits:
        assert not np.intersect1d(train, valid).size
        assert set(groups[train]) != set(groups[valid])
        coverage[valid] += 1
    np.testing.assert_array_equal(coverage, np.ones(indices.size, dtype=np.int64))


def test_stable_interaction_ranking_rejects_sign_flip() -> None:
    base = np.linspace(-1.0, 1.0, 40)
    residual = np.concatenate([base, base])
    matrix = np.stack(
        [
            residual,
            np.concatenate([base, -base]),
            np.concatenate([np.zeros(35), np.ones(5), np.zeros(35), np.ones(5)]),
        ],
        axis=1,
    )
    groups = np.asarray([0] * 40 + [1] * 40, dtype=np.int64)
    ranked, scores, metadata = rank_stable_interactions(
        matrix,
        residual,
        groups,
        minimum_support=20,
        support_tolerance=1.0e-8,
    )
    assert ranked[0] == 0
    assert 1 not in ranked
    assert 2 not in ranked
    assert scores[0] > 0.99
    assert metadata["n_sign_stable"] == 1


def test_single_group_ranking_respects_support() -> None:
    residual = np.linspace(-1.0, 1.0, 20)
    matrix = np.stack(
        [residual, np.asarray([1.0] + [0.0] * 19)], axis=1
    )
    ranked, scores = rank_interactions(
        matrix,
        residual,
        minimum_support=5,
        support_tolerance=1.0e-8,
    )
    np.testing.assert_array_equal(ranked, np.asarray([0]))
    assert scores[0] > 0.99


def test_clip_logit_is_finite_at_probability_boundaries() -> None:
    values = _clip_logit(np.asarray([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(values))
    assert values[0] < 0.0 < values[-1]
    assert abs(values[1]) < 1.0e-12
