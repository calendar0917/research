"""Shim for ``chemprop.nn.hparams.HasHParams``."""
class HasHParams:
    def __init__(self, *args, **kwargs):
        self.hparams = {}
