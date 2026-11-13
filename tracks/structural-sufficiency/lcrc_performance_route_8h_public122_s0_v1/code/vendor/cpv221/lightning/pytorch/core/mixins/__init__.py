"""Shim for ``lightning.pytorch.core.mixins.HyperparametersMixin``.

The pinned message-passing module only *stores* hparams; nothing in the forward
pass reads them.
"""


class HyperparametersMixin:
    def save_hyperparameters(self, *args, **kwargs):
        return None
