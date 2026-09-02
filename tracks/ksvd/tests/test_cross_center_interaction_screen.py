from __future__ import annotations

import numpy as np

from tracks.ksvd.experiments.luyin16.cross_center_interaction_screen import (
    _cross_covar,
    _conditional_interaction_blocks,
    _distribution_mean_std,
    _fold_indices_excluding,
    _load_excluded_indices,
)


def test_cross_covar_detects_same_centre_alignment() -> None:
    topology = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    attributes = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    shuffled = attributes[[1, 0]]
    aligned = _cross_covar(topology, attributes).reshape(2, 2)
    broken = _cross_covar(topology, shuffled).reshape(2, 2)
    assert np.linalg.norm(aligned) > 0.0
    np.testing.assert_allclose(broken, -aligned, atol=1e-7)


def test_distribution_readout_keeps_mean_and_dispersion() -> None:
    rows = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    result = _distribution_mean_std(rows)
    np.testing.assert_allclose(result, [1.0, 2.0, 1.0, 1.0], atol=1e-7)


def test_cross_covar_single_centre_is_zero() -> None:
    topology = np.asarray([[1.0, 0.0]], dtype=np.float32)
    attributes = np.asarray([[0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(_cross_covar(topology, attributes), 0.0)


def test_excluded_confirmation_selection_is_disjoint() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    archive = {
        "original_indices": np.arange(10, dtype=np.int64),
        "fold_0_train_indices": np.arange(0, 6, dtype=np.int64),
        "fold_0_valid_indices": np.arange(6, 10, dtype=np.int64),
    }
    screen = {
        "seed": 17,
        "max_train_graphs_per_fold": 4,
        "max_valid_graphs_per_fold": 2,
    }
    train, valid = _fold_indices_excluding(
        archive, 0, labels, screen, np.asarray([0, 6, 9], dtype=np.int64)
    )
    assert set(train).isdisjoint({0, 6, 9})
    assert set(valid).isdisjoint({0, 6, 9})
    assert train.size == 4
    assert valid.size == 2


def test_exclusion_manifest_reads_unique_dataset_indices(tmp_path) -> None:
    path = tmp_path / "manifest.npz"
    np.savez(path, dataset_indices=np.asarray([4, 2, 4, 1], dtype=np.int64))
    indices, metadata = _load_excluded_indices(path)
    np.testing.assert_array_equal(indices, [1, 2, 4])
    assert metadata["enabled"] is True
    assert metadata["n_indices"] == 3


def test_exclusion_manifest_unions_multiple_sources(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez(first, dataset_indices=np.asarray([1, 3], dtype=np.int64))
    np.savez(second, dataset_indices=np.asarray([3, 5], dtype=np.int64))
    indices, metadata = _load_excluded_indices([first, second])
    np.testing.assert_array_equal(indices, [1, 3, 5])
    assert metadata["path"] is None
    assert len(metadata["sources"]) == 2


def test_conditional_interaction_blocks_keep_true_and_null_coordinates() -> None:
    cross = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
    )
    binding = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]], dtype=np.float32
    )
    cross_null = [cross[[1, 0, 3, 2]]]
    binding_null = [binding[[1, 0, 3, 2]]]
    blocks, metadata = _conditional_interaction_blocks(
        cross, binding, 3, cross_null, binding_null, rank=2
    )
    assert metadata["bilinear_projection"]["n_components"] == 2
    assert blocks["norm_gate"].shape == (4, 2)
    assert blocks["bilinear_gate"].shape == (4, 2)
    assert not np.allclose(
        blocks["norm_gate"], blocks["norm_gate_cross_true_binding_shuffled_0"]
    )
    assert not np.allclose(
        blocks["bilinear_gate"], blocks["bilinear_gate_both_shuffled_0"]
    )
