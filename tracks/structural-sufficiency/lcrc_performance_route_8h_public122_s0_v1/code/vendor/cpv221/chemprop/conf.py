"""Shim for ``chemprop.conf``.

The pinned release computes these from
``SimpleMoleculeMolGraphFeaturizer().shape``; the values are 72 (atom, V2
multi-hot) and 14 (bond), asserted by ``lcpr_dmpnn.featurizer_shapes``.
"""
DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = 72, 14
DEFAULT_HIDDEN_DIM = 300
