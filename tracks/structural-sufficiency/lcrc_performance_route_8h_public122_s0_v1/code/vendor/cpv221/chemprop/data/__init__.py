"""Shim: ``BatchMolGraph`` container with the pinned attribute semantics."""
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    V: Tensor
    E: Tensor
    edge_index: Tensor
    rev_edge_index: Tensor
    batch: Tensor

    __size: int = field(init=False)

    def __post_init__(self):
        self.__size = self.batch[-1].item() + 1

    def __len__(self) -> int:
        return self.__size

    def to(self, device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
        return self

    @property
    def n_molecules(self) -> int:
        return int(self.__size)
