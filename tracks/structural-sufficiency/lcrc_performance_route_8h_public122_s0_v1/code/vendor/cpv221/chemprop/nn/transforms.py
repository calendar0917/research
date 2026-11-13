"""Shim: graph/scale transforms are unused (``graph_transform=None``)."""
import torch.nn as nn


class GraphTransform(nn.Identity):
    pass


class ScaleTransform(nn.Identity):
    pass
