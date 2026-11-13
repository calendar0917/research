"""Vendored subset of chemprop v2.2.1 (pinned source, minimal dependency shim).

Only the message-passing / aggregation / FFN modules needed to instantiate the
round's D-MPNN reference are vendored.  ``nn/message_passing/base.py``,
``nn/message_passing/mixins.py``, ``nn/agg.py``, ``nn/ffn.py`` and
``nn/utils.py`` and ``exceptions.py`` are BYTE COPIES of the pinned release
(tag ``v2.2.1``, ``chemprop-2.2.1`` wheel); see ``VENDOR.json`` for hashes.

The modules below (``conf.py``, ``data/``, ``nn/transforms.py``,
``nn/hparams.py``, ``nn/proto.py``, ``utils/``) plus ``lightning/`` are shims that
provide exactly the names those files import, so the pinned computation runs
without the lightning/chemprop CLI stack.  They add no numerical behaviour:
``BatchMolGraph`` is a plain container, the transforms are ``nn.Identity`` when
no transform is supplied, and the hyper-parameter mixin stores nothing.
"""
