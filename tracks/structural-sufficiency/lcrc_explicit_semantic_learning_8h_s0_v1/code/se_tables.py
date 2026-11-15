"""Codec, semantic table generator (F) and the paired free-lookup tables.

Everything here is *exact* with respect to the frozen ``lipo_v1`` codec:

* ``p = path_cat``:    ``mid = p // 25``, ``b1 = (p % 25) // 5``, ``b2 = p % 5``
* ``e = nbr/end code``:``atom = e // 5``, ``bond = e % 5``

``path_end63`` is a historical field name; the live schema has 80 end classes
(16 atom types x 5 bond types), not 63.  The generator never uses the ZINC
189/63 vocabularies.

The generator F is *one* shared pair of composition tables plus *one* small MLP
per free bank.  Six free tables of the C1 local encoder are replaced by
``materialise(...)``: the generator's current output passed through *fixed*
per-column affine buffers that were computed once at ``theta_0`` and never
re-estimated.  The gate bank is not standardised and is ``1 + net``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import se_common as C


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------


def decode_path(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``path_cat`` -> (middle atom type, bond1, bond2)."""
    mid = torch.div(codes, C.N_BOND_TYPES * C.N_BOND_TYPES, rounding_mode="floor")
    rem = codes - mid * (C.N_BOND_TYPES * C.N_BOND_TYPES)
    b1 = torch.div(rem, C.N_BOND_TYPES, rounding_mode="floor")
    b2 = rem - b1 * C.N_BOND_TYPES
    return mid, b1, b2


def decode_binary(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``nbr``/``end`` code -> (atom type, bond type)."""
    atom = torch.div(codes, C.N_BOND_TYPES, rounding_mode="floor")
    bond = codes - atom * C.N_BOND_TYPES
    return atom, bond


def encode_path(mid: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return (
        mid.astype(np.int64) * (C.N_BOND_TYPES * C.N_BOND_TYPES)
        + b1.astype(np.int64) * C.N_BOND_TYPES
        + b2.astype(np.int64)
    )


def encode_binary(atom: np.ndarray, bond: np.ndarray) -> np.ndarray:
    return atom.astype(np.int64) * C.N_BOND_TYPES + bond.astype(np.int64)


def codec_roundtrip() -> dict[str, Any]:
    """Full-grid encode/decode round trip on both banks."""
    paths = np.arange(C.N_PATH_CODES, dtype=np.int64)
    mid = paths // (C.N_BOND_TYPES * C.N_BOND_TYPES)
    rem = paths % (C.N_BOND_TYPES * C.N_BOND_TYPES)
    b1 = rem // C.N_BOND_TYPES
    b2 = rem % C.N_BOND_TYPES
    ok_path = np.array_equal(encode_path(mid, b1, b2), paths)
    binary = np.arange(C.N_NBR_CODES, dtype=np.int64)
    atom = binary // C.N_BOND_TYPES
    bond = binary % C.N_BOND_TYPES
    ok_bin = np.array_equal(encode_binary(atom, bond), binary)
    return {
        "path_roundtrip": bool(ok_path),
        "binary_roundtrip": bool(ok_bin),
        "path_ranges": {
            "mid": [int(mid.min()), int(mid.max())],
            "b1": [int(b1.min()), int(b1.max())],
            "b2": [int(b2.min()), int(b2.max())],
        },
        "atom_range": [int(atom.min()), int(atom.max())],
        "bond_range": [int(bond.min()), int(bond.max())],
        "n_path_codes": C.N_PATH_CODES,
        "n_binary_codes": C.N_NBR_CODES,
    }


# ---------------------------------------------------------------------------
# permutation (composite-code bijection, not a per-atom rename)
# ---------------------------------------------------------------------------


def build_permutations(seed: int = C.PERM_SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    pi = rng.permutation(C.N_PATH_CODES).astype(np.int64)     # 400
    sigma = rng.permutation(C.N_NBR_CODES).astype(np.int64)   # 80
    return {"path": pi, "binary": sigma}


def perm_for_bank(bank: str) -> str:
    return "path" if bank in ("path_emb", "path_u", "path_gate") else "binary"


# ---------------------------------------------------------------------------
# generator F
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Linear -> SiLU -> Linear (no output activation)."""

    def __init__(self, dims: list[int]):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
        return x


class SemanticGenerator(nn.Module):
    """Shared atom/bond composition tables + six bank MLPs.

    * ``A``: Embedding(16, 16) (atom type -> 16-d)
    * ``B``: Embedding(5, 8)  (bond type -> 8-d)
    * path banks:   MLP(32 -> 197 -> 32)
    * binary banks: MLP(24 -> 40 -> 32)
    * gate:         1 + MLP(32 -> 23 -> 1) with the final Linear zeroed
    """

    def __init__(self):
        super().__init__()
        self.atom = nn.Embedding(C.N_ATOM_TYPES, C.ATOM_EMB_DIM)
        self.bond = nn.Embedding(C.N_BOND_TYPES, C.BOND_EMB_DIM)
        self.path_emb = _MLP([C.RANK, C.PATH_MLP_HIDDEN, C.RANK])
        self.path_u = _MLP([C.RANK, C.PATH_MLP_HIDDEN, C.RANK])
        self.nbr_emb = _MLP([C.ATOM_EMB_DIM + C.BOND_EMB_DIM, C.BINARY_MLP_HIDDEN, C.RANK])
        self.end_emb = _MLP([C.ATOM_EMB_DIM + C.BOND_EMB_DIM, C.BINARY_MLP_HIDDEN, C.RANK])
        self.path_v = _MLP([C.ATOM_EMB_DIM + C.BOND_EMB_DIM, C.BINARY_MLP_HIDDEN, C.RANK])
        self.path_gate = _MLP([C.RANK, C.GATE_MLP_HIDDEN, 1])

    # -- code -> composition vector ---------------------------------------
    def path_input(self, codes: torch.Tensor) -> torch.Tensor:
        mid, b1, b2 = decode_path(codes)
        return torch.cat([self.atom(mid), self.bond(b1), self.bond(b2)], dim=1)

    def binary_input(self, codes: torch.Tensor) -> torch.Tensor:
        atom, bond = decode_binary(codes)
        return torch.cat([self.atom(atom), self.bond(bond)], dim=1)

    def bank_raw(self, bank: str, codes: torch.Tensor) -> torch.Tensor:
        if bank in ("path_emb", "path_u"):
            return getattr(self, bank)(self.path_input(codes))
        if bank in ("nbr_emb", "end_emb", "path_v"):
            return getattr(self, bank)(self.binary_input(codes))
        if bank == "path_gate":
            return getattr(self, bank)(self.path_input(codes))
        raise KeyError(bank)

    def init_module(self, generator: torch.Generator) -> None:
        """Independent CPU-Generator init: embeddings N(0,1), Linear U(+-1/sqrt(in))."""
        with torch.no_grad():
            self.atom.weight.copy_(torch.randn(self.atom.weight.shape, generator=generator))
            self.bond.weight.copy_(torch.randn(self.bond.weight.shape, generator=generator))
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    bound = 1.0 / float(np.sqrt(module.in_features))
                    module.weight.copy_(
                        torch.empty_like(module.weight).uniform_(-bound, bound, generator=generator)
                    )
                    module.bias.copy_(
                        torch.empty_like(module.bias).uniform_(-bound, bound, generator=generator)
                    )
            # gate final layer zeroed -> gate == 1 at init
            self.path_gate.lins[-1].weight.zero_()
            self.path_gate.lins[-1].bias.zero_()

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


# ---------------------------------------------------------------------------
# affine buffers + materialisation
# ---------------------------------------------------------------------------


def _column_stats(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = table.mean(axis=0)
    sd = table.std(axis=0)
    return mu, sd


def build_buffers(
    generator: SemanticGenerator,
    e0_tables: Mapping[str, np.ndarray],
    permutations: Mapping[str, np.ndarray] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-bank fixed buffers from the theta_0 raw output on the (true) code grid.

    For both the semantic and permuted generators the buffers are computed from
    the *unpermuted* raw grid at ``theta_0``; the composite bijection is a
    bijection on the code grid, so the full-vocabulary multiset of a bank's
    values is preserved between FS and FP.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for bank in C.VECTOR_TABLES:
            n_codes, dim, _kind = C.TABLES[bank]
            codes = torch.arange(n_codes, dtype=torch.long)
            raw = generator.bank_raw(bank, codes).detach().cpu().numpy().astype(np.float64)
            mu_f0, sd_f0 = _column_stats(raw)
            sd_f0 = np.maximum(sd_f0, C.STD_FLOOR)
            mu_e0, sd_e0 = _column_stats(np.asarray(e0_tables[bank], dtype=np.float64))
            out[bank] = {
                "mu_f0": mu_f0,
                "sd_f0": sd_f0,
                "mu_e0": mu_e0,
                "sd_e0": sd_e0,
            }
    return out


def materialise_bank(
    generator: SemanticGenerator,
    bank: str,
    buffers: Mapping[str, Mapping[str, np.ndarray]],
    perm: np.ndarray | None = None,
    device: Any = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Full ``(n_codes, dim)`` table for one bank at the generator's current theta."""
    n_codes, dim, kind = C.TABLES[bank]
    codes = torch.arange(n_codes, dtype=torch.long, device=device)
    if perm is not None:
        codes = torch.as_tensor(np.asarray(perm, dtype=np.int64), device=device)[codes]
    raw = generator.bank_raw(bank, codes)
    if kind == "path_gate":
        return 1.0 + raw
    buf = buffers[bank]
    mu_f0 = torch.as_tensor(buf["mu_f0"], dtype=dtype, device=device)
    sd_f0 = torch.as_tensor(buf["sd_f0"], dtype=dtype, device=device)
    mu_e0 = torch.as_tensor(buf["mu_e0"], dtype=dtype, device=device)
    sd_e0 = torch.as_tensor(buf["sd_e0"], dtype=dtype, device=device)
    return (raw - mu_f0) / sd_f0 * sd_e0 + mu_e0


def materialise_all(
    generator: SemanticGenerator,
    buffers: Mapping[str, Mapping[str, np.ndarray]],
    permutations: Mapping[str, np.ndarray] | None = None,
    *,
    permute: bool,
    device: Any = None,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for bank in C.TABLES:
        perm = None
        if permute and permutations is not None:
            perm = permutations[perm_for_bank(bank)]
        out[bank] = materialise_bank(generator, bank, buffers, perm=perm, device=device)
    return out


# ---------------------------------------------------------------------------
# code table module used inside the local encoder
# ---------------------------------------------------------------------------


class CodeTable(nn.Module):
    """One bank with an ``nn.Embedding``-compatible ``forward(codes)``.

    ``mode == "free"``       -> a plain trainable embedding.
    ``mode == "generator"``  -> generated from the shared ``SemanticGenerator``.
    ``perm`` is applied to the *input composite code* before decoding.
    """

    def __init__(
        self,
        bank: str,
        mode: str,
        *,
        generator: SemanticGenerator | None = None,
        buffers: Mapping[str, Mapping[str, np.ndarray]] | None = None,
        perm: np.ndarray | None = None,
        init_table: torch.Tensor | None = None,
    ):
        super().__init__()
        self.bank = bank
        n_codes, dim, _kind = C.TABLES[bank]
        self.n_codes = n_codes
        self.dim = dim
        self.mode = mode
        self.perm_np = None if perm is None else np.asarray(perm, dtype=np.int64)
        if mode == "free":
            self.table = nn.Embedding(n_codes, dim)
            if init_table is not None:
                with torch.no_grad():
                    self.table.weight.copy_(init_table)
        elif mode == "generator":
            if generator is None or buffers is None:
                raise ValueError("generator mode needs a generator and buffers")
            # not registered as submodules again
            object.__setattr__(self, "_gen", generator)
            object.__setattr__(self, "_sem_buffers", buffers)
        else:
            raise ValueError(mode)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if self.mode == "free":
            return self.table(codes)
        gen: SemanticGenerator = object.__getattribute__(self, "_gen")
        buffers = object.__getattribute__(self, "_sem_buffers")
        if self.perm_np is not None:
            perm = torch.as_tensor(self.perm_np, device=codes.device)[codes]
        else:
            perm = codes
        raw = gen.bank_raw(self.bank, perm)
        if C.TABLES[self.bank][2] == "path_gate":
            return 1.0 + raw
        buf = buffers[self.bank]
        mu_f0 = torch.as_tensor(buf["mu_f0"], dtype=raw.dtype, device=raw.device)
        sd_f0 = torch.as_tensor(buf["sd_f0"], dtype=raw.dtype, device=raw.device)
        mu_e0 = torch.as_tensor(buf["mu_e0"], dtype=raw.dtype, device=raw.device)
        sd_e0 = torch.as_tensor(buf["sd_e0"], dtype=raw.dtype, device=raw.device)
        return (raw - mu_f0) / sd_f0 * sd_e0 + mu_e0


__all__ = [name for name in dir() if not name.startswith("_")]
