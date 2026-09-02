from __future__ import annotations

import numpy as np

from tracks.ksvd.experiments.luyin16.invariant_joint_reconstruction_screen import (
    _balanced_rows,
    _mean_std_readout,
    _shared_l2_rows,
    _shuffle_patch_attributes,
)


REPRESENTATION = {
    "node_role_bins": 2,
    "edge_role_bins": 2,
}


def _rows() -> np.ndarray:
    # Width is 2 node roles + 2 edge roles + 40 atom + 13 bond coordinates.
    rows = np.zeros((3, 57), dtype=np.float32)
    rows[:, 0:2] = [[1, 0], [0, 1], [0.5, 0.5]]
    rows[:, 2:4] = [[1, 0], [0, 1], [0.5, 0.5]]
    rows[0, 4:11] = 1.0
    rows[1, 11:18] = 1.0
    rows[2, 18:25] = 1.0
    rows[0, 44:47] = 1.0
    rows[1, 47:50] = 1.0
    rows[2, 50:53] = 1.0
    return rows


def test_balanced_and_shared_l2_rows_have_expected_mass_and_norm() -> None:
    rows = _rows()
    balanced = _balanced_rows(rows, REPRESENTATION)
    np.testing.assert_allclose(balanced[:, 4:44].sum(axis=1), 1.0)
    np.testing.assert_allclose(balanced[:, 44:].sum(axis=1), 1.0)
    normalized = _shared_l2_rows(rows, REPRESENTATION)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_attribute_shuffle_preserves_modality_multisets() -> None:
    rows = _rows()
    shuffled = _shuffle_patch_attributes(rows, REPRESENTATION, seed=7)
    structure_width = 4
    np.testing.assert_allclose(
        np.sort(rows[:, :structure_width], axis=0),
        np.sort(shuffled[:, :structure_width], axis=0),
    )
    np.testing.assert_allclose(
        np.sort(rows[:, structure_width:], axis=0),
        np.sort(shuffled[:, structure_width:], axis=0),
    )
    np.testing.assert_allclose(
        _mean_std_readout(rows[:, :structure_width]),
        _mean_std_readout(shuffled[:, :structure_width]),
    )
    np.testing.assert_allclose(
        _mean_std_readout(rows[:, structure_width:]),
        _mean_std_readout(shuffled[:, structure_width:]),
    )
