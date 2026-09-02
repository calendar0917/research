"""Self-tests for R1-A IMDB WALK basis characterization."""
from __future__ import annotations

import numpy as np

from .imdb_walk_basis_characterization import (
    matched_dictionary_similarity,
    nearest_real_patch_proximity,
    top_activating_patch_summary,
)


def main() -> int:
    dictionary = np.eye(3, dtype=np.float64)
    patches = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    proximity = nearest_real_patch_proximity(
        dictionary, patches, np.asarray([2, 3, 4], dtype=np.int64)
    )
    assert np.allclose(proximity["nearest_absolute_cosine"][:2], [1.0, 1.0])
    assert proximity["nearest_patch_edge_counts"][:2] == [2, 3]

    codes = np.asarray(
        [[3.0, 2.0, 0.0, 0.0], [0.0, 1.0, 4.0, 2.0]], dtype=np.float64
    )
    walk = np.asarray(
        [[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=np.float64
    )
    tops = top_activating_patch_summary(
        codes,
        walk,
        walk.copy(),
        np.asarray([1, 2, 2, 1], dtype=np.int64),
        ((0, 0, 0, 2), (1, 1, 2, 4)),
        top_k=2,
    )
    assert len(tops) == 2
    assert tops[0]["graph_indices"] == [0, 0]
    assert tops[0]["unique_walk_vector_count"] == 2
    assert tops[1]["graph_indices"] == [1, 1]
    assert np.isclose(tops[1]["mean_absolute_coefficient"], 3.0)

    right = dictionary[:, [1, 2, 0]] * np.asarray([-1.0, 1.0, -1.0])[None, :]
    similarity = matched_dictionary_similarity(dictionary, right)
    assert np.isclose(similarity["mean_absolute_matched_atom_cosine"], 1.0)
    assert np.isclose(similarity["minimum_absolute_matched_atom_cosine"], 1.0)

    print("imdb_walk_basis_characterization self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
