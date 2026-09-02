from __future__ import annotations

import numpy as np

from .run_luyin14_rich_readout import _rich_code_readouts


def test_rich_readout_shape_and_reconstruction() -> None:
    dictionary = np.eye(3, dtype=np.float64)
    codes = np.asarray(
        [[1.0, 0.0, 0.2, 0.0], [0.0, -2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    patches = dictionary @ codes
    readout = _rich_code_readouts(codes, patches, dictionary)
    assert readout["rich_no_recon"].shape == (30,)
    assert readout["rich"].shape == (38,)
    assert np.allclose(readout["rich"][-8:-2], 0.0)
    assert np.allclose(readout["rich"][-2:], [4.0, np.log1p(4.0)])


def test_rich_readout_detects_distribution_beyond_coarse_mean() -> None:
    dictionary = np.eye(2, dtype=np.float64)
    left = np.asarray([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    right = np.asarray([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]])
    left_readout = _rich_code_readouts(left, dictionary @ left, dictionary)
    right_readout = _rich_code_readouts(right, dictionary @ right, dictionary)
    assert not np.allclose(
        left_readout["rich_no_recon"], right_readout["rich_no_recon"]
    )
