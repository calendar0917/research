from __future__ import annotations

import unittest

import numpy as np

from .run_molhiv_ssl_context_ksvd_gate import (
    _code_features,
    _graph_features,
    _random_dictionary,
)


class MolhivSslContextKsvdGateTests(unittest.TestCase):
    def test_feature_shapes(self) -> None:
        values = np.arange(40, dtype=np.float64).reshape(5, 8)
        self.assertEqual(_graph_features(values).shape, (24,))
        self.assertEqual(_code_features(values).shape, (32,))

    def test_random_dictionary_columns_are_normalized(self) -> None:
        values = np.random.default_rng(3).normal(size=(20, 8))
        dictionary = _random_dictionary(values, atoms=6, seed=5)
        self.assertEqual(dictionary.shape, (8, 6))
        np.testing.assert_allclose(np.linalg.norm(dictionary, axis=0), 1.0)


if __name__ == "__main__":
    unittest.main()
