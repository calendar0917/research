"""Utilities for a two-level sparse dictionary representation.

The first dictionary encodes the *content* of individual patches.  The
second is fitted to explicit, observed relations between encoded patches.  It
does not ask a patch code to guess which other patch it should be connected
to: the patch graph supplied to the classifier already contains that fact.
"""

from __future__ import annotations

import numpy as np


def relation_event_vectors(
    codes: np.ndarray,
    pair_index: np.ndarray,
    relation_attributes: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Turn unordered patch pairs into fixed-width relation-event vectors.

    For a pair of sparse content codes, its sum, absolute difference, and
    coordinate-wise product are invariant to swapping the two endpoints.
    ``relation_attributes`` holds observed graph facts such as overlap and
    cross-patch chemical bonds.  The result has one column per relation event.
    """
    Z = np.asarray(codes, dtype=np.float64)
    pairs = np.asarray(pair_index, dtype=np.int64)
    attrs = np.asarray(relation_attributes, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("codes must have shape (n_content_atoms, n_patches)")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_index must have shape (n_pairs, 2)")
    if attrs.ndim != 2 or attrs.shape[0] != pairs.shape[0]:
        raise ValueError("relation_attributes must align with pair_index")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= Z.shape[1]):
        raise ValueError("pair_index is outside the patch-code columns")
    if not len(pairs):
        return np.zeros((3 * Z.shape[0] + attrs.shape[1], 0), dtype=np.float64)
    left = Z[:, pairs[:, 0]].T
    right = Z[:, pairs[:, 1]].T
    events = np.concatenate(
        [left + right, np.abs(left - right), left * right, attrs], axis=1
    ).T
    if normalize:
        events /= np.maximum(np.linalg.norm(events, axis=0, keepdims=True), 1e-12)
    return events


def relation_readout(codes: np.ndarray) -> np.ndarray:
    """A fixed graph-level readout for patch or relation code columns.

    The last value records how many objects were pooled.  This makes an empty
    relation set distinguishable from a nonempty set whose code average is
    zero, without exposing an arbitrary patch ordering.
    """
    X = np.asarray(codes, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("codes must have shape (n_features, n_objects)")
    width, n = X.shape
    if not n:
        return np.concatenate([np.zeros(4 * width, dtype=np.float64), [0.0]])
    return np.concatenate(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.max(axis=1),
            X.min(axis=1),
            [np.log1p(float(n))],
        ]
    )


def shuffled_relation_events(
    n_patches: int,
    pair_index: np.ndarray,
    relation_attributes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Destroy patch-relation binding while preserving event count and types.

    The shuffled control picks the same number of unordered patch pairs from
    the current graph and permutes the relation attributes across them.  It
    therefore keeps the patch count, event count, and multiset of relation
    attributes but breaks *which* patches are related.
    """
    pairs = np.asarray(pair_index, dtype=np.int64)
    attrs = np.asarray(relation_attributes, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_index must have shape (n_pairs, 2)")
    if attrs.ndim != 2 or attrs.shape[0] != pairs.shape[0]:
        raise ValueError("relation_attributes must align with pair_index")
    candidates = np.array(
        [(i, j) for i in range(n_patches) for j in range(i + 1, n_patches)],
        dtype=np.int64,
    )
    if len(pairs) > len(candidates):
        raise ValueError("more relation events than unordered patch pairs")
    if not len(pairs):
        return pairs.copy(), attrs.copy()
    chosen = rng.choice(len(candidates), size=len(pairs), replace=False)
    return candidates[chosen], attrs[rng.permutation(len(attrs))]


def shuffled_patch_code_binding(
    codes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Break the binding of patch content codes to relation endpoints.

    Patch-relation graphs can be dense.  Simply resampling their edges then
    leaves many original endpoints untouched.  A derangement of code columns
    is a stronger matched control: it preserves each graph's patch-code bag,
    every relation edge, and every relation attribute, but no relation event
    sees the code of its true endpoint.
    """
    Z = np.asarray(codes, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("codes must have shape (n_features, n_patches)")
    n = Z.shape[1]
    if n < 2:
        return Z.copy()
    identity = np.arange(n)
    permutation = identity.copy()
    # A random derangement avoids the usual one fixed point of a random
    # permutation.  The deterministic cyclic fallback is also a derangement.
    for _ in range(32):
        candidate = rng.permutation(n)
        if not np.any(candidate == identity):
            permutation = candidate
            break
    else:
        permutation = np.roll(identity, int(rng.integers(1, n)))
    return Z[:, permutation]
