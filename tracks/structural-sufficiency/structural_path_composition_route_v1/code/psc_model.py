"""PSC: shared path encoder + additive (SUM) or pooled (POOL) readout.

Only the *position* of the shared nonlinear scalar head changes between the
two readouts; parameters, shapes, init and per-path encoder are identical.

    SUM : F(G) = beta + sum_l c_l sum_p 0.5 [h(E(x_p)) + h(E(rev_p))]
    POOL: z(G) =      sum_l c_l sum_p 0.5 [ E(x_p)  +  E(rev_p) ]; F = beta + h(z)

``beta`` is the only intercept; ``h`` has no biases so SUM cannot turn a head
bias into a path-count term and POOL cannot turn it into a repeated intercept.
No BatchNorm, no graph-size normalization, no GNN / attention / fragment ids.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

READOUTS = ("SUM", "POOL")
INPUTS = ("ORDER", "BAG")
TARGETS = ("MOLECULE", "PAIR", "DIAG")


class PathEncoder(nn.Module):
    """Shared categorical path encoder ``input -> 256 -> 128 -> 64``."""

    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int], max_len: int,
                 width: int = 256, mid: int = 128, out: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), 32) for d in atom_dims])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(d), 16) for d in bond_dims])
        in_dim = ((int(max_len) + 1) * 32 + int(max_len) * 16
                  + (2 * int(max_len) + 1) + (int(max_len) + 1))
        self.in_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(width)), nn.SiLU(), nn.Dropout(float(dropout)),
            nn.Linear(int(width), int(mid)), nn.SiLU(), nn.Dropout(float(dropout)),
            nn.Linear(int(mid), int(out)), nn.SiLU(),
        )
        self.out_dim = int(out)

    def build_input(self, atom_vals: torch.Tensor, bond_vals: torch.Tensor,
                    plen: torch.Tensor, reverse: bool) -> torch.Tensor:
        """Embedding-sum per slot + masks + length one-hot (padding stays zero)."""
        n = atom_vals.shape[0]
        L = self.max_len
        slot = torch.arange(L + 1, device=atom_vals.device)
        if reverse:
            a_idx = torch.clamp(plen[:, None] - slot[None, :], min=0)
            b_idx = torch.clamp(plen[:, None] - 1 - torch.arange(L, device=atom_vals.device)[None, :], min=0)
        else:
            a_idx = slot[None, :].expand(n, L + 1)
            b_idx = torch.arange(L, device=atom_vals.device)[None, :].expand(n, L)
        atoms = torch.gather(atom_vals, 1, a_idx[:, :, None].expand(n, L + 1, atom_vals.shape[2]))
        bonds = torch.gather(bond_vals, 1, b_idx[:, :, None].expand(n, L, bond_vals.shape[2]))
        a = sum(emb(atoms[:, :, i]) for i, emb in enumerate(self.atom_emb)) / math.sqrt(len(self.atom_emb))
        b = sum(emb(bonds[:, :, i]) for i, emb in enumerate(self.bond_emb)) / math.sqrt(len(self.bond_emb))
        amask = (slot[None, :] <= plen[:, None]).to(a.dtype)
        bmask = (torch.arange(L, device=atom_vals.device)[None, :] < plen[:, None]).to(b.dtype)
        a = a * amask[:, :, None]
        b = b * bmask[:, :, None]
        onehot = F.one_hot(plen, L + 1).to(a.dtype)
        return torch.cat([a.reshape(n, -1), b.reshape(n, -1), amask, bmask, onehot], dim=-1)

    def forward(self, atom_vals: torch.Tensor, bond_vals: torch.Tensor,
                plen: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        return self.net(self.build_input(atom_vals, bond_vals, plen, reverse))


class ScalarHead(nn.Module):
    """Shared scalar head; no biases by construction."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(dim), int(dim), bias=False), nn.SiLU(),
                                 nn.Linear(int(dim), 1, bias=False))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class PSCModel(nn.Module):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int], max_len: int,
                 readout: str, width: int = 256, mid: int = 128, out: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout}")
        self.readout = readout
        self.max_len = int(max_len)
        self.encoder = PathEncoder(atom_dims, bond_dims, max_len, width, mid, out, dropout)
        self.head = ScalarHead(int(out))
        self.beta = nn.Parameter(torch.zeros(()))

    # -- init ---------------------------------------------------------------
    def init_parameters(self, seed: int, head_last_std: float = 1e-3) -> dict[str, Any]:
        """Identical init for every arm of a given seed (explicit function)."""
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 17)
        with torch.no_grad():
            for emb in list(self.encoder.atom_emb) + list(self.encoder.bond_emb):
                emb.weight.normal_(0.0, 0.1, generator=gen)
            for module in self.encoder.net.modules():
                if isinstance(module, nn.Linear):
                    bound = 1.0 / math.sqrt(module.in_features)
                    module.weight.uniform_(-bound, bound, generator=gen)
                    if module.bias is not None:
                        module.bias.uniform_(-bound, bound, generator=gen)
            for module in self.head.net.modules():
                if isinstance(module, nn.Linear):
                    if module.out_features == 1:
                        module.weight.normal_(0.0, float(head_last_std), generator=gen)
                    else:
                        bound = 1.0 / math.sqrt(module.in_features)
                        module.weight.uniform_(-bound, bound, generator=gen)
            self.beta.zero_()
        return self.shared_state()

    def shared_state(self) -> dict[str, Any]:
        return {
            "atom_emb_sha256": C_hash([e.weight.detach().cpu().numpy()
                                       for e in self.encoder.atom_emb]),
            "bond_emb_sha256": C_hash([e.weight.detach().cpu().numpy()
                                       for e in self.encoder.bond_emb]),
            "encoder_sha256": C_hash([p.detach().cpu().numpy()
                                      for p in self.encoder.parameters()]),
            "head_sha256": C_hash([p.detach().cpu().numpy()
                                   for p in self.head.parameters()]),
            "beta": float(self.beta.detach().cpu()),
        }

    # -- forward ------------------------------------------------------------
    def forward_batch(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        atom = batch["atom"]
        bond = batch["bond"]
        plen = batch["plen"]
        n_mol = int(batch["n_molecules"])
        device = atom.device
        e_f = self.encoder(atom, bond, plen, reverse=False)
        if batch["is_bag"]:
            # BAG input is invariant to reversal by construction, so the second
            # direction is the same function value (not an approximation).
            e_r = e_f
        else:
            e_r = self.encoder(atom, bond, plen, reverse=True)
        inv = batch["inv"]
        owner = batch["owner"]
        weight = batch["weight"]
        if self.readout == "SUM":
            s = 0.5 * (self.head(e_f) + self.head(e_r))
            contrib = (s[inv] * weight)
            yhat = self.beta + torch.zeros(n_mol, device=device).index_add_(0, owner, contrib)
            z_mol = None
            local = s.detach()
        else:
            zp = 0.5 * (e_f + e_r)
            z_mol = torch.zeros(n_mol, zp.shape[1], device=device).index_add_(
                0, owner, zp[inv] * weight[:, None])
            yhat = self.beta + self.head(z_mol)
            local = None
        return {"yhat_std": yhat, "local": local, "z_mol": z_mol,
                "n_paths_encoded": int(atom.shape[0])}


def C_hash(arrays: Sequence[np.ndarray]) -> str:
    import hashlib

    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.shape).encode())
        h.update(arr.dtype.str.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def build_model(atom_dims: Sequence[int], bond_dims: Sequence[int], max_len: int,
                readout: str, seed: int, cfg: Mapping[str, Any] | None = None
                ) -> tuple[PSCModel, dict[str, Any]]:
    dims = dict((cfg or {}).get("dims", {}))
    model = PSCModel(atom_dims, bond_dims, max_len, readout,
                     width=int(dims.get("width", 256)), mid=int(dims.get("mid", 128)),
                     out=int(dims.get("out", 64)), dropout=float(dims.get("dropout", 0.1)))
    hashes = model.init_parameters(int(seed), float(dims.get("head_last_std", 1e-3)))
    return model, hashes


def parameter_ledger(model: nn.Module) -> dict[str, int]:
    groups: dict[str, int] = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + int(param.numel())
    groups["total"] = sum(int(p.numel()) for p in model.parameters())
    return groups


def path_scores(model: PSCModel, cache: Any, path_ids: Sequence[int] | np.ndarray,
                bag: bool = False, batch_paths: int = 65536) -> np.ndarray:
    """``s_theta(t)`` for canonical path ids, eval mode, no labels."""
    model.eval()
    ids = np.asarray(path_ids, np.int64)
    out: list[np.ndarray] = []
    src_atom = cache.tensor("atom_bag" if bag else "atom")
    src_bond = cache.tensor("bond_bag" if bag else "bond")
    with torch.no_grad():
        for start in range(0, ids.size, int(batch_paths)):
            part = torch.as_tensor(ids[start:start + int(batch_paths)]).to(cache.device)
            plen = cache.tensor("plen")[part]
            e_f = model.encoder(src_atom[part], src_bond[part], plen, reverse=False)
            if bag:
                e_r = e_f
            else:
                e_r = model.encoder(src_atom[part], src_bond[part], plen, reverse=True)
            s = 0.5 * (model.head(e_f) + model.head(e_r))
            out.append(s.detach().double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def path_embeddings(model: PSCModel, cache: Any, path_ids: np.ndarray,
                    bag: bool = False, batch_paths: int = 65536) -> np.ndarray:
    """Averaged double-direction path embeddings for diagnostic ledgers."""
    model.eval()
    ids = np.asarray(path_ids, np.int64)
    out: list[np.ndarray] = []
    src_atom = cache.tensor("atom_bag" if bag else "atom")
    src_bond = cache.tensor("bond_bag" if bag else "bond")
    with torch.no_grad():
        for start in range(0, ids.size, int(batch_paths)):
            part = torch.as_tensor(ids[start:start + int(batch_paths)]).to(cache.device)
            plen = cache.tensor("plen")[part]
            e_f = model.encoder(src_atom[part], src_bond[part], plen, reverse=False)
            e_r = e_f if bag else model.encoder(src_atom[part], src_bond[part], plen,
                                               reverse=True)
            out.append((0.5 * (e_f + e_r)).detach().double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.encoder.out_dim))