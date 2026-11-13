"""Shim for ``chemprop.nn.message_passing.proto.MessagePassing``.

The pinned subclass writes into ``self.hparams``; the real class inherits that
dict from ``HasHParams``.  Nothing in the forward pass reads it.
"""
from torch import nn


class MessagePassing(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hparams = {}

    def forward(self, bmg, V_d=None):  # pragma: no cover - abstract in the pinned release
        raise NotImplementedError
