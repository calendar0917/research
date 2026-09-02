from __future__ import annotations

import numpy as np

from .local_chemical_change import decompose_molecule, one_branch_replacement, replacement_vector


def test_single_same_site_substitution_becomes_one_signed_change() -> None:
    # The phenyl core and its attachment position are shared; ethyl is
    # replaced by methoxy.  The returned vector must retain a direction.
    positive = decompose_molecule("COc1ccccc1")
    negative = decompose_molecule("CCc1ccccc1")
    assert positive is not None and negative is not None
    replacement = one_branch_replacement(positive, negative)
    assert replacement is not None
    vector = replacement_vector(replacement)
    assert vector.shape == (512,)
    assert np.any(vector > 0) and np.any(vector < 0)


def test_multiple_simultaneous_substitutions_are_rejected() -> None:
    # One molecule changes two branches relative to the other.  Calling this
    # a local replacement would introduce an avoidable confound.
    left = decompose_molecule("COc1cc(Cl)ccc1")
    right = decompose_molecule("CCc1cc(F)ccc1")
    assert left is not None and right is not None
    assert one_branch_replacement(left, right) is None


def test_different_cores_are_rejected() -> None:
    left = decompose_molecule("COc1ccccc1")
    right = decompose_molecule("COc1ccncc1")
    assert left is not None and right is not None
    assert one_branch_replacement(left, right) is None
