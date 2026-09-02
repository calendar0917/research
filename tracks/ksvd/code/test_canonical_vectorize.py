"""Self-tests for exact non-WL canonical patch vectors."""
from __future__ import annotations

import numpy as np

from .graph import from_edges
from .sampling_route import PatchCollection, vectorize_collection
from .vectorize import canonical_adjacency_features, wl_patch_features


def _relabel(g, permutation: np.ndarray):
    return from_edges(g.n, [(int(permutation[u]), int(permutation[v])) for u, v in g.edges()])


def main() -> None:
    # Triangular prism and K3,3 are connected 3-regular graphs on six nodes.
    # The current degree-initialized 1-WL histogram cannot distinguish them.
    prism = from_edges(6, [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)])
    k33 = from_edges(6, [(u,v) for u in range(3) for v in range(3,6)])
    wl_prism = np.asarray(wl_patch_features(prism, set(prism.nodes), 8))
    wl_k33 = np.asarray(wl_patch_features(k33, set(k33.nodes), 8))
    assert np.array_equal(wl_prism, wl_k33)
    can_prism = np.asarray(canonical_adjacency_features(prism, set(prism.nodes), 8))
    can_k33 = np.asarray(canonical_adjacency_features(k33, set(k33.nodes), 8))
    assert not np.array_equal(can_prism, can_k33)

    rng = np.random.default_rng(4)
    for rooted in (False, True):
        base_root = 0 if rooted else None
        base = np.asarray(canonical_adjacency_features(prism, set(prism.nodes), 8, base_root))
        for _ in range(20):
            perm = rng.permutation(prism.n)
            gp = _relabel(prism, perm)
            root = int(perm[base_root]) if rooted else None
            cur = np.asarray(canonical_adjacency_features(gp, set(gp.nodes), 8, root))
            assert np.array_equal(base, cur)

    pc = PatchCollection("manual", [set(prism.nodes)], [0], [[]])
    Y = vectorize_collection(prism, pc, 8, "rooted_canonical")
    assert Y.shape == (8 * 7 // 2 + 16, 1)
    assert np.isclose(np.linalg.norm(Y[:, 0]), 1.0)
    print("canonical_vectorize self-tests: PASS")


if __name__ == "__main__":
    main()
