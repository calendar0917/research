"""I-CRATE-v0 — Incidence-Structured White-Box Dictionary Transformer (model).

Scientific object only (no training / run-control plumbing), so the algebraic
properties can be tested in isolation.  This implements exactly
``tracks/ksvd/notes/icrate_v0_preregistration.md``:

* persistent atom + bond tokens on the true incidence graph (one token per
  undirected chemical bond; the two PyG directed entries are deduplicated);
* ``L = 4`` structural layers, each with its own ``U^l`` (Incidence-MSSA
  subspaces), ``D_a^l`` (analysis dictionary) and ``D_s^l`` (synthesis
  dictionary); no cross-layer sharing;
* incidence-constrained CRATE-style compression followed by an overcomplete
  nonnegative sparse code (2-step ISTA) and a residual synthesis update;
* a permutation-invariant, size-sensitive graph seed, one-way whole-graph
  Cross-MSSA refinement, and a global sparse code ``alpha_G`` (4-step ISTA)
  over ``D_G``;
* a ``96 -> 64 -> 1`` head on ``alpha_G`` only; outer objective ``MAE`` only.

The same helpers operate on either a single graph (no batch dimension) or a
padded batch (leading batch dimension), so the reference and batched forwards
share the algebra exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# frozen schema / dimensions (pre-registration §3)
# ---------------------------------------------------------------------------

D_V = 21  # atom-type categories {0..20}
D_E = 3  # bond-type categories {1,2,3}
BOND_OFFSET = 1  # bond types are stored as {1,2,3}

D_MODEL = 48
H_HEADS = 4
HEAD_DIM = D_MODEL // H_HEADS  # 12
L_LAYERS = 4
OVERCOMPLETE = 2
M_DICT = D_MODEL * OVERCOMPLETE  # 96

LAMBDA_TOK = 0.10
LAMBDA_G = 0.10
R_TOK = 2
R_G = 4

EPS = 1.0e-8
NEG_INF = -1.0e9


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass
class Molecule:
    """One molecule as atom tokens + undirected bond tokens."""

    atom: torch.Tensor  # [n] long, atom categories
    src: torch.Tensor  # [m] long, endpoint 0
    dst: torch.Tensor  # [m] long, endpoint 1
    bond: torch.Tensor  # [m] long, bond categories (already re-indexed {1,2,3}->{0,1,2})
    y: float

    @property
    def n(self) -> int:
        return int(self.atom.numel())

    @property
    def m(self) -> int:
        return int(self.bond.numel())

    @property
    def n_tokens(self) -> int:
        return self.n + self.m


def molecule_raw(
    atom_types: Sequence[int],
    bonds: Sequence[tuple[int, int, int]],
    y: float = 0.0,
) -> Molecule:
    """Build a :class:`Molecule` preserving the given bond order.

    ``bonds`` are ``(u, v, bond_type)`` with ``bond_type in {1,2,3}``.  Bond
    order is preserved (needed by permutation tests); use
    :func:`molecule_from_arrays` for a canonical sorted object.
    """
    atom = torch.as_tensor(list(atom_types), dtype=torch.long)
    n = int(atom.numel())
    if n == 0:
        raise ValueError("empty molecule")
    if int(atom.min()) < 0 or int(atom.max()) >= D_V:
        raise ValueError(f"atom type out of range: {atom.tolist()}")
    src: list[int] = []
    dst: list[int] = []
    types: list[int] = []
    for u, v, b in bonds:
        if u == v:
            raise ValueError("self-loop bond is not a ZINC chemical bond")
        lo, hi = (u, v) if u < v else (v, u)
        if not (0 <= lo < hi < n):
            raise ValueError(f"bond endpoint out of range: {(u, v)}")
        if int(b) < 1 or int(b) > D_E:
            raise ValueError(f"bond type out of range: {b}")
        src.append(int(lo))
        dst.append(int(hi))
        types.append(int(b) - BOND_OFFSET)
    return Molecule(
        atom=atom,
        src=torch.as_tensor(src, dtype=torch.long),
        dst=torch.as_tensor(dst, dtype=torch.long),
        bond=torch.as_tensor(types, dtype=torch.long),
        y=float(y),
    )


def molecule_from_arrays(
    atom_types: Sequence[int],
    bonds: Sequence[tuple[int, int, int]],
    y: float = 0.0,
) -> Molecule:
    """Canonical :class:`Molecule` with bonds sorted by ``(src, dst)``."""
    mol = molecule_raw(atom_types, bonds, y)
    if mol.m:
        order = torch.argsort(mol.src * (mol.n + 1) + mol.dst)
        mol = Molecule(
            atom=mol.atom,
            src=mol.src[order],
            dst=mol.dst[order],
            bond=mol.bond[order],
            y=mol.y,
        )
    return mol


def molecule_from_edges(
    atom_types: Sequence[int],
    edge_index: Any,
    edge_attr: Any,
    y: float = 0.0,
) -> Molecule:
    """Adapter from directed ``(edge_index, edge_attr)`` lists.

    Deduplicates the two directed entries of every undirected bond and checks
    that both ``edge_attr`` copies agree (raises, never guesses).
    """
    ei = torch.as_tensor(edge_index, dtype=torch.long)
    ea = torch.as_tensor(edge_attr, dtype=torch.long).reshape(-1)
    seen: dict[tuple[int, int], int] = {}
    for k in range(int(ei.shape[1])):
        u = int(ei[0, k])
        v = int(ei[1, k])
        key = (u, v) if u <= v else (v, u)
        val = int(ea[k])
        prev = seen.get(key)
        if prev is None:
            seen[key] = val
        elif prev != val:
            raise RuntimeError(
                f"inconsistent directed bond attributes for {key}: {prev} vs {val}"
            )
    bonds = sorted((u, v, b) for (u, v), b in seen.items())
    return molecule_from_arrays(atom_types, bonds, y)


def molecule_from_pyg(data: Any) -> Molecule:
    atom = data.x.reshape(-1).to(torch.long)
    y = float(data.y.reshape(-1)[0]) if hasattr(data, "y") else 0.0
    return molecule_from_edges(atom.tolist(), data.edge_index, data.edge_attr, y)


# ---------------------------------------------------------------------------
# incidence
# ---------------------------------------------------------------------------


def incidence_keep(mol: Molecule) -> torch.Tensor:
    """Boolean ``[N, N]`` keep matrix: atom<->incident-bond pairs only."""
    n = mol.n
    m = mol.m
    total = n + m
    keep = torch.zeros(total, total, dtype=torch.bool)
    for e in range(m):
        u = int(mol.src[e])
        v = int(mol.dst[e])
        et = n + e
        keep[u, et] = True
        keep[et, u] = True
        keep[v, et] = True
        keep[et, v] = True
    return keep


@dataclass
class TokenBatch:
    """Padded token batch.  Token order per graph is ``[atoms ; bonds]``."""

    atom_idx: torch.Tensor  # [B, Nmax] long (pad 0)
    bond_idx: torch.Tensor  # [B, Mmax] long (pad 0)
    keep: torch.Tensor  # [B, T, T] bool
    valid: torch.Tensor  # [B, T] bool
    n_atoms: torch.Tensor  # [B] long
    m_bonds: torch.Tensor  # [B] long
    n_tokens: torch.Tensor  # [B] long
    y: torch.Tensor  # [B] float32
    n_graphs: int

    @property
    def T(self) -> int:
        return int(self.keep.shape[1])

    def to(self, device: torch.device | str) -> "TokenBatch":
        device = torch.device(device)
        return TokenBatch(
            atom_idx=self.atom_idx.to(device),
            bond_idx=self.bond_idx.to(device),
            keep=self.keep.to(device),
            valid=self.valid.to(device),
            n_atoms=self.n_atoms.to(device),
            m_bonds=self.m_bonds.to(device),
            n_tokens=self.n_tokens.to(device),
            y=self.y.to(device),
            n_graphs=self.n_graphs,
        )


def collate(samples: Sequence[Molecule]) -> TokenBatch:
    if not samples:
        raise ValueError("cannot collate an empty sequence")
    b = len(samples)
    nmax = max(s.n for s in samples)
    mmax = max(s.m for s in samples)
    t = nmax + mmax

    atom_idx = torch.zeros(b, nmax, dtype=torch.long)
    bond_idx = torch.zeros(b, mmax, dtype=torch.long)
    keep = torch.zeros(b, t, t, dtype=torch.bool)
    valid = torch.zeros(b, t, dtype=torch.bool)
    n_atoms = torch.zeros(b, dtype=torch.long)
    m_bonds = torch.zeros(b, dtype=torch.long)
    n_tokens = torch.zeros(b, dtype=torch.long)
    ys = torch.zeros(b, dtype=torch.float32)

    for g, mol in enumerate(samples):
        n, m = mol.n, mol.m
        atom_idx[g, :n] = mol.atom
        bond_idx[g, :m] = mol.bond
        valid[g, :n] = True
        valid[g, nmax : nmax + m] = True
        n_atoms[g] = n
        m_bonds[g] = m
        n_tokens[g] = n + m
        ys[g] = mol.y
        # padded layout: atom slots [0, Nmax), bond slots [Nmax, Nmax + Mmax)
        for e in range(m):
            bt = nmax + e
            for a in (int(mol.src[e]), int(mol.dst[e])):
                keep[g, a, bt] = True
                keep[g, bt, a] = True
    return TokenBatch(
        atom_idx=atom_idx,
        bond_idx=bond_idx,
        keep=keep,
        valid=valid,
        n_atoms=n_atoms,
        m_bonds=m_bonds,
        n_tokens=n_tokens,
        y=ys,
        n_graphs=b,
    )


# ---------------------------------------------------------------------------
# primitive operators
# ---------------------------------------------------------------------------


def normalize_columns(dictionary: torch.Tensor) -> torch.Tensor:
    """Normalise each column of a ``[d, M]`` dictionary to unit L2 norm."""
    return dictionary / (dictionary.norm(dim=0, keepdim=True) + EPS)


def normalize_columns_heads(u: torch.Tensor) -> torch.Tensor:
    """Normalise the columns of every ``[d, p]`` head matrix in ``[H, d, p]``."""
    return u / (u.norm(dim=1, keepdim=True) + EPS)


def masked_softmax(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Row softmax with ``keep`` (bool) selecting legal keys.

    Rows with no legal key return all-zeros (never NaN).
    """
    filled = torch.where(keep, scores, torch.full_like(scores, NEG_INF))
    row_has = keep.any(dim=-1, keepdim=True)
    row_max = filled.max(dim=-1, keepdim=True).values
    row_max = torch.where(row_has, row_max, torch.zeros_like(row_max))
    exp = torch.exp(filled - row_max)
    exp = torch.where(keep, exp, torch.zeros_like(exp))
    denom = exp.sum(dim=-1, keepdim=True)
    out = exp / denom.clamp_min(1.0e-30)
    return torch.where(row_has, out, torch.zeros_like(out))


def _incidence_mssa(z_norm: torch.Tensor, keep: torch.Tensor, u_hat: torch.Tensor) -> torch.Tensor:
    """``Delta_MSSA = 1/H sum_h A_h Y_h U_h^T`` for ``z_norm = LN_mssa(Z)``.

    ``z_norm``: ``[..., T, d]``; ``keep``: ``[..., T, T]``; ``u_hat``: ``[H, d, p]``.
    """
    y = torch.einsum("...td,hdk->...htk", z_norm, u_hat)
    scores = torch.einsum("...htk,...hsk->...hts", y, y) / math.sqrt(HEAD_DIM)
    attn = masked_softmax(scores, keep.unsqueeze(-3))
    ay = torch.einsum("...hts,...hsk->...htk", attn, y)
    return torch.einsum("...htk,hdk->...td", ay, u_hat) / H_HEADS


def token_odl(x_norm: torch.Tensor, d_a_hat: torch.Tensor, r_steps: int, lam: float) -> torch.Tensor:
    """2-step ISTA nonnegative sparse code ``A`` in ``[..., T, M]`` for ``X = x_norm``.

    Solves ``min_{A>=0} 1/2 ||X - D_a A||_F^2 + lam ||A||_1`` (per token column)
    with safe step ``eta = 0.9/(||D_a||_2^2+eps)`` (spectral norm detached).
    """
    spec = torch.linalg.matrix_norm(d_a_hat, ord=2).detach()
    eta = 0.9 / (spec * spec + EPS)
    dt = d_a_hat.t()
    a = torch.zeros(*x_norm.shape[:-1], d_a_hat.shape[1], dtype=x_norm.dtype, device=x_norm.device)
    for _ in range(int(r_steps)):
        err = a @ dt - x_norm
        grad = err @ d_a_hat
        a = F.relu(a - eta * grad - eta * lam)
    return a


def global_ista(g: torch.Tensor, d_g_hat: torch.Tensor, r_steps: int, lam: float) -> torch.Tensor:
    """4-step ISTA nonnegative sparse code ``alpha`` in ``[..., M]`` for ``g``."""
    spec = torch.linalg.matrix_norm(d_g_hat, ord=2).detach()
    eta = 0.9 / (spec * spec + EPS)
    dt = d_g_hat.t()
    alpha = torch.zeros(*g.shape[:-1], d_g_hat.shape[1], dtype=g.dtype, device=g.device)
    for _ in range(int(r_steps)):
        err = alpha @ dt - g
        grad = err @ d_g_hat
        alpha = F.relu(alpha - eta * grad - eta * lam)
    return alpha


def _graph_seed(z: torch.Tensor, valid: torch.Tensor, n_tokens: torch.Tensor) -> torch.Tensor:
    """``g0 = N^{-1/2} sum_i Z_i`` over valid tokens (size-sensitive)."""
    mask = valid.to(z.dtype).unsqueeze(-1)
    total = (z * mask).sum(dim=-2)
    return total / torch.sqrt(n_tokens.to(z.dtype)).unsqueeze(-1)


def _cross_mssa(
    g0_norm: torch.Tensor,
    z_norm: torch.Tensor,
    valid: torch.Tensor,
    u_g_hat: torch.Tensor,
) -> torch.Tensor:
    """One-way whole-graph attention update ``delta_G`` in ``[..., d]``."""
    q = torch.einsum("...d,hdk->...hk", g0_norm, u_g_hat)
    keys = torch.einsum("...td,hdk->...htk", z_norm, u_g_hat)
    scores = torch.einsum("...hk,...htk->...ht", q, keys) / math.sqrt(HEAD_DIM)
    attn = masked_softmax(scores, valid.unsqueeze(-2))
    ak = torch.einsum("...ht,...htk->...hk", attn, keys)
    return torch.einsum("...hk,hdk->...d", ak, u_g_hat) / H_HEADS


# ---------------------------------------------------------------------------
# structural layer / model
# ---------------------------------------------------------------------------


class StructuralLayer(nn.Module):
    """One structural stage: Incidence-MSSA then ODL (independent per layer)."""

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.ln_mssa = nn.LayerNorm(D_MODEL)
        self.u = nn.Parameter(
            torch.randn(H_HEADS, D_MODEL, HEAD_DIM, generator=generator) * 0.1
        )
        self.ln_odl = nn.LayerNorm(D_MODEL)
        self.d_a = nn.Parameter(
            torch.randn(D_MODEL, M_DICT, generator=generator) / math.sqrt(D_MODEL)
        )
        self.d_s = nn.Parameter(
            torch.randn(D_MODEL, M_DICT, generator=generator) / math.sqrt(D_MODEL)
        )


@dataclass
class ForwardInfo:
    z_layers: list[torch.Tensor] = field(default_factory=list)  # Z^l inputs (l=0..L)
    z_half: list[torch.Tensor] = field(default_factory=list)  # Z^{l+1/2}
    delta_mssa: list[torch.Tensor] = field(default_factory=list)
    coeffs: list[torch.Tensor] = field(default_factory=list)  # A^l
    attn: list[torch.Tensor] = field(default_factory=list)  # optional head-mean attention
    g0: torch.Tensor | None = None
    delta_g: torch.Tensor | None = None
    g_half: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ICrateV0(nn.Module):
    """Incidence-structured white-box dictionary transformer."""

    def __init__(
        self,
        *,
        seed: int = 0,
        init_std: float = 1.0,
        no_incidence: bool = False,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        self.e_v = nn.Embedding(D_V, D_MODEL)
        self.e_e = nn.Embedding(D_E, D_MODEL)
        with torch.no_grad():
            self.e_v.weight.normal_(0.0, init_std, generator=generator)
            self.e_e.weight.normal_(0.0, init_std, generator=generator)
        self.layers = nn.ModuleList([StructuralLayer(generator) for _ in range(L_LAYERS)])
        self.ln_g = nn.LayerNorm(D_MODEL)
        self.ln_tokens = nn.LayerNorm(D_MODEL)
        self.u_g = nn.Parameter(torch.randn(H_HEADS, D_MODEL, HEAD_DIM, generator=generator) * 0.1)
        self.d_g = nn.Parameter(torch.randn(D_MODEL, M_DICT, generator=generator) / math.sqrt(D_MODEL))
        self.head = nn.Sequential(nn.Linear(M_DICT, 64), nn.SiLU(), nn.Linear(64, 1))
        self.no_incidence = bool(no_incidence)

    # -- embedding ---------------------------------------------------------
    def embed_batch(self, batch: TokenBatch) -> torch.Tensor:
        za = self.e_v(batch.atom_idx)
        zb = self.e_e(batch.bond_idx)
        z = torch.cat([za, zb], dim=1)
        return z * batch.valid.unsqueeze(-1).to(z.dtype)

    def embed_molecule(self, mol: Molecule) -> torch.Tensor:
        za = self.e_v(mol.atom)
        zb = self.e_e(mol.bond)
        return torch.cat([za, zb], dim=0)

    # -- core --------------------------------------------------------------
    def core(
        self,
        z: torch.Tensor,
        keep: torch.Tensor,
        valid: torch.Tensor,
        n_tokens: torch.Tensor,
        *,
        record_attn: bool = False,
    ) -> tuple[torch.Tensor, ForwardInfo]:
        info = ForwardInfo()
        info.z_layers.append(z)
        for layer in self.layers:
            u_hat = normalize_columns_heads(layer.u)
            z_norm = layer.ln_mssa(z)
            if record_attn:
                y = torch.einsum("...td,hdk->...htk", z_norm, u_hat)
                scores = torch.einsum("...htk,...hsk->...hts", y, y) / math.sqrt(HEAD_DIM)
                info.attn.append(masked_softmax(scores, keep.unsqueeze(-3)).mean(dim=-3))
            delta = _incidence_mssa(z_norm, keep, u_hat)
            if self.no_incidence:
                delta = torch.zeros_like(delta)
            z_half = z + delta
            d_a_hat = normalize_columns(layer.d_a)
            x = layer.ln_odl(z_half)
            a = token_odl(x, d_a_hat, R_TOK, LAMBDA_TOK)
            d_s_hat = normalize_columns(layer.d_s)
            r = a @ d_s_hat.t()
            z = z_half + r
            info.delta_mssa.append(delta)
            info.z_half.append(z_half)
            info.coeffs.append(a)
            info.z_layers.append(z)

        z_l = z
        g0 = _graph_seed(z_l, valid, n_tokens)
        u_g_hat = normalize_columns_heads(self.u_g)
        delta_g = _cross_mssa(self.ln_g(g0), self.ln_tokens(z_l), valid, u_g_hat)
        g_half = g0 + delta_g
        d_g_hat = normalize_columns(self.d_g)
        alpha = global_ista(g_half, d_g_hat, R_G, LAMBDA_G)
        pred = self.head(alpha).squeeze(-1)
        info.g0 = g0
        info.delta_g = delta_g
        info.g_half = g_half
        info.alpha = alpha
        return pred, info

    # -- forwards ----------------------------------------------------------
    def forward(self, batch: TokenBatch, *, record_attn: bool = False) -> tuple[torch.Tensor, ForwardInfo]:
        z = self.embed_batch(batch)
        return self.core(z, batch.keep, batch.valid, batch.n_tokens, record_attn=record_attn)

    def forward_reference(
        self, mol: Molecule, *, record_attn: bool = False
    ) -> tuple[torch.Tensor, ForwardInfo]:
        z = self.embed_molecule(mol)
        keep = incidence_keep(mol).to(z.device)
        valid = torch.ones(mol.n_tokens, dtype=torch.bool, device=z.device)
        n_tokens = torch.tensor(float(mol.n_tokens), dtype=torch.float32, device=z.device)
        return self.core(z, keep, valid, n_tokens, record_attn=record_attn)

    # -- audit helpers -----------------------------------------------------
    def parameter_breakdown(self) -> dict[str, int]:
        tokenizer = int(self.e_v.weight.numel() + self.e_e.weight.numel())
        per_layer = int(
            sum(p.numel() for p in self.layers[0].parameters())
        )
        structural = int(sum(p.numel() for p in self.layers.parameters()))
        global_mssa = int(self.u_g.numel() + self.ln_g.weight.numel() + self.ln_g.bias.numel() + self.ln_tokens.weight.numel() + self.ln_tokens.bias.numel())
        global_dict = int(self.d_g.numel())
        head = int(sum(p.numel() for p in self.head.parameters()))
        return {
            "tokenizer": tokenizer,
            "structural_per_layer": per_layer,
            "structural_total": structural,
            "global_mssa": global_mssa,
            "global_dictionary": global_dict,
            "head": head,
            "total": int(sum(p.numel() for p in self.parameters())),
        }

    def frozen_dictionaries(self) -> dict[str, torch.Tensor]:
        """Column-normalised dictionaries used for the audit (detached)."""
        out: dict[str, torch.Tensor] = {}
        for li, layer in enumerate(self.layers):
            out[f"layer{li}_D_a"] = normalize_columns(layer.d_a).detach()
            out[f"layer{li}_D_s"] = normalize_columns(layer.d_s).detach()
        out["global_D_G"] = normalize_columns(self.d_g).detach()
        return out
