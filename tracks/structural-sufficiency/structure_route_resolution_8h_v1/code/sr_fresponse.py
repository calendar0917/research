"""Free small-subgraph response ``psi`` and its permutation average.

A rooted support of size k <= 4 is written into four fixed atom slots (root in
slot 0) and six fixed undirected bond slots ``01,02,03,12,13,23``.  Every
permutation of the non-root atoms is materialised, the same shared
``support_mlp`` is applied to each arrangement, and the response is the mean.

* present atom: 16-dim projected full-9-field embedding, then a 4-bit present mask
* present bond: 8-dim projected full-3-field embedding, then a 6-bit present mask
* absent slots: exactly zero (the projection bias is multiplied by the mask)
* input 122 = 4*16 + 6*8 + 4 + 6;  output 32 (no final activation)

Nothing in this module knows about energies, Mobius or labels.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import sr_struct as ST

PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
PAIR_INDEX = {pair: i for i, pair in enumerate(PAIRS)}


def _node_perms(k: int) -> list[tuple[int, int, int, int]]:
    """Target slot s holds canonical node ``perm[s]`` (-1 beyond k)."""
    if k == 1:
        return [(0, -1, -1, -1)]
    out = []
    for p in itertools.permutations(range(1, k)):
        slots = (0,) + p
        out.append(tuple(list(slots) + [-1] * (4 - k)))
    return out


def _edge_maps(k: int) -> tuple[list[tuple[int, ...]], list[tuple[float, ...]]]:
    """For each permutation, the canonical bond slot and validity of the 6 pairs."""
    src: list[tuple[int, ...]] = []
    ok: list[tuple[float, ...]] = []
    for perm in _node_perms(k):
        row_src = []
        row_ok = []
        for a, b in PAIRS:
            ia, ib = perm[a], perm[b]
            if ia < 0 or ib < 0:
                row_src.append(0)
                row_ok.append(0.0)
            else:
                row_src.append(PAIR_INDEX[(min(ia, ib), max(ia, ib))])
                row_ok.append(1.0)
        src.append(tuple(row_src))
        ok.append(tuple(row_ok))
    return src, ok


PERM_TABLE = {k: torch.as_tensor(_node_perms(k), dtype=torch.long) for k in (1, 2, 3, 4)}
EDGE_SRC = {k: torch.as_tensor(_edge_maps(k)[0], dtype=torch.long) for k in (1, 2, 3, 4)}
EDGE_OK = {k: torch.as_tensor(_edge_maps(k)[1], dtype=torch.float32) for k in (1, 2, 3, 4)}


class FreeResponse(nn.Module):
    def __init__(self, atom_emb: int = 16, bond_emb: int = 12,
                 atom_columns: int = 9, bond_columns: int = 3,
                 support_hidden: int = 96, out_dim: int = 32,
                 bond_out: int = 8) -> None:
        super().__init__()
        self.atom_dim = int(atom_emb) * int(atom_columns)
        self.bond_dim = int(bond_emb) * int(bond_columns)
        self.atom_out = int(atom_emb)
        self.bond_out = int(bond_out)
        self.atom_projection = nn.Linear(self.atom_dim, self.atom_out)
        self.bond_projection = nn.Linear(self.bond_dim, self.bond_out)
        in_dim = 4 * self.atom_out + 6 * self.bond_out + 4 + 6
        self.support_mlp = nn.Sequential(
            nn.Linear(in_dim, int(support_hidden)), nn.SiLU(),
            nn.Linear(int(support_hidden), int(out_dim)),
        )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    # -- deterministic init (independent RNG stream) ------------------------
    def init_parameters(self, seed: int, salt: int = 97) -> dict[str, str]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + int(salt))
        import math

        with torch.no_grad():
            for module in (self.atom_projection, self.bond_projection, self.support_mlp):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        bound = 1.0 / math.sqrt(layer.in_features)
                        layer.weight.uniform_(-bound, bound, generator=gen)
                        if layer.bias is not None:
                            layer.bias.uniform_(-bound, bound, generator=gen)
        return self.hashes()

    def hashes(self) -> dict[str, str]:
        import sr_common as C

        return C.module_hashes({"atom_projection": self.atom_projection,
                                "bond_projection": self.bond_projection,
                                "support_mlp": self.support_mlp})

    def ledger(self) -> dict[str, int]:
        def n(m: nn.Module) -> int:
            return int(sum(p.numel() for p in m.parameters()))

        out = {"atom_projection": n(self.atom_projection),
               "bond_projection": n(self.bond_projection),
               "support_mlp": n(self.support_mlp)}
        out["total"] = int(sum(out.values()))
        return out

    # -- type-level psi -----------------------------------------------------
    def _psi_group(self, sel: torch.Tensor, k: int, reg: dict[str, Any],
                   atom_emb: Sequence[nn.Embedding], bond_emb: Sequence[nn.Embedding]
                   ) -> torch.Tensor:
        device = sel.device
        af = reg["atom_fields"][sel]                     # [n,4,9]
        am = reg["atom_mask"][sel]                       # [n,4]
        bf = reg["bond_fields"][sel]                     # [n,6,3]
        em = reg["edge_mask"][sel]                       # [n,6]
        idx = PERM_TABLE[k].to(device)                   # [P,4]
        esrc = EDGE_SRC[k].to(device)                    # [P,6]
        eok = EDGE_OK[k].to(device)                      # [P,6]
        n = int(sel.shape[0])
        p = int(idx.shape[0])

        node_vals = af[:, idx]                           # [n,P,4,9]
        node_mask = am[:, idx]                           # [n,P,4]
        slot_ok = torch.as_tensor([s < k for s in range(4)], device=device,
                                  dtype=node_mask.dtype)
        node_mask = node_mask * slot_ok
        parts = [emb(node_vals[..., i]) for i, emb in enumerate(atom_emb)]
        node_in = torch.cat(parts, dim=-1)               # [n,P,4,144]
        node_vec = torch.nn.functional.silu(self.atom_projection(node_in))
        node_vec = node_vec * node_mask.unsqueeze(-1)

        bond_vals = bf[:, esrc]                          # [n,P,6,3]
        bond_mask = em[:, esrc] * eok                    # [n,P,6]
        bparts = [emb(bond_vals[..., i]) for i, emb in enumerate(bond_emb)]
        bond_in = torch.cat(bparts, dim=-1)              # [n,P,6,36]
        bond_vec = torch.nn.functional.silu(self.bond_projection(bond_in))
        bond_vec = bond_vec * bond_mask.unsqueeze(-1)

        flat = torch.cat([node_vec.reshape(n, p, -1),
                          bond_vec.reshape(n, p, -1),
                          node_mask, bond_mask], dim=-1)
        if int(flat.shape[-1]) != self.in_dim:
            raise ValueError(f"free response width {flat.shape[-1]} != {self.in_dim}")
        out = self.support_mlp(flat)                     # [n,P,32]
        return out.mean(dim=1)

    def psi(self, type_ids: torch.Tensor, reg: dict[str, Any],
            atom_emb: Sequence[nn.Embedding], bond_emb: Sequence[nn.Embedding]
            ) -> torch.Tensor:
        """Response for every id in ``type_ids`` (may repeat); returns [len,32]."""
        device = type_ids.device
        uniq, inv = torch.unique(type_ids, return_inverse=True)
        out = torch.zeros((int(uniq.shape[0]), self.out_dim), device=device,
                          dtype=self.support_mlp[0].weight.dtype)
        kinds = reg["k"][uniq]
        for k in (1, 2, 3, 4):
            mask = kinds == k
            if not bool(mask.any()):
                continue
            sel = uniq[mask]
            out.index_copy_(0, torch.nonzero(mask, as_tuple=False).view(-1),
                            self._psi_group(sel, k, reg, atom_emb, bond_emb))
        return out[inv]


def registry_tensors(reg: ST.TypeRegistry, device: Any) -> dict[str, Any]:
    return reg.torch_tensors(device)