"""RBO — rooted branch operator (v0).

Computation, in order:

1. ``z_(v->u)``: one vector per directed branch, built *only* from static local
   content — centre atom features, bond type, neighbour atom features, and the
   neighbour's remaining typed incidence histogram (bond type x atom type
   counts after deleting the back bond).  No radius-2 key, no neighbour hidden
   state, no recursion.
2. Shared low-rank projection ``x_(v->u) = SiLU(W2 SiLU(W1 z))`` with r = 32.
3. Symmetric within-root product statistics, counts preserved:
   ``p1 = sum_i x_i``, ``e2 = sum_{i<j} x_i*x_j``, ``e3 = sum_{i<j<l} x_i*x_j*x_l``
   (element-wise; computed from power sums, O(k) per root).
   A relation-weighted two-body term
   ``rel2 = sum_{i<j} g(rel_ij) * (x_i*x_j)``
   uses the auditable structural relation of the branch pair (extra bond between
   the two neighbours, shared neighbour, ring separation in G - {root}).
4. Aggregation over roots: sum, max (existence sensitive) and mean of squares,
   plus generic count features and the frozen global-33 context channel.
5. One compact nonlinear reader -> raw y.

No P3/DCC composer, no frozen supervised checkpoint; only the frozen train-only
masked-SSL atom/bond encoders are reused as generic token encoders.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_V2 = _SSUF / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ad2_models as A2  # noqa: E402

RANK = 32
HIST_DIM = 64
N_REL = 64


def collate(rows, cache: Mapping, prep: Mapping, device) -> dict:
    """Batch assembly: pure index arithmetic, one numpy pass per batch."""
    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    ap = np.asarray(prep["atom_ptr"], dtype=np.int64)[rows]
    na = np.asarray(prep["atom_ptr"], dtype=np.int64)[rows + 1] - ap
    atom_idx = np.concatenate([np.arange(s, s + n) for s, n in zip(ap, na)])
    atom_batch = np.repeat(np.arange(B), na).astype(np.int64)
    atom_off = np.zeros(B, dtype=np.int64)
    atom_off[1:] = np.cumsum(na)[:-1]

    brp = np.asarray(prep["br_ptr"], dtype=np.int64)
    bs = brp[rows]
    nb = brp[rows + 1] - bs
    br_idx = np.concatenate([np.arange(s, s + n) for s, n in zip(bs, nb)])
    br_batch = np.repeat(np.arange(B), nb).astype(np.int64)
    br_off = np.zeros(B, dtype=np.int64)
    br_off[1:] = np.cumsum(nb)[:-1]

    br_src_local = np.asarray(prep["br_src_local"], dtype=np.int64)[br_idx]
    br_dst_local = np.asarray(prep["br_dst_local"], dtype=np.int64)[br_idx]
    br_bond_local = np.asarray(prep["br_bond_local"], dtype=np.int64)[br_idx]
    src_slot = atom_off[br_batch] + br_src_local
    dst_slot = atom_off[br_batch] + br_dst_local

    bond_ptr = np.asarray(prep["bond_ptr"], dtype=np.int64)
    bpp = bond_ptr[rows]
    bpn = bond_ptr[rows + 1] - bpp
    bond_idx = np.concatenate([np.arange(s, s + n) for s, n in zip(bpp, bpn)])
    bond_off = np.zeros(B, dtype=np.int64)
    bond_off[1:] = np.cumsum(bpn)[:-1]
    # bond feature slot of each branch (branch i of bond k uses bond k)
    bond_slot = bond_off[br_batch] + br_bond_local

    prp = np.asarray(prep["pair_ptr"], dtype=np.int64)
    ps = prp[rows]
    npr = prp[rows + 1] - ps
    pair_idx = np.concatenate([np.arange(s, s + n) for s, n in zip(ps, npr)])
    pair_batch = np.repeat(np.arange(B), npr).astype(np.int64)
    pair_off = np.zeros(B, dtype=np.int64)
    pair_off[1:] = np.cumsum(npr)[:-1]
    pair_root = atom_off[pair_batch] + np.asarray(prep["pair_root_local"], np.int64)[pair_idx]
    pair_i = br_off[pair_batch] + np.asarray(prep["pair_i_local"], np.int64)[pair_idx]
    pair_j = br_off[pair_batch] + np.asarray(prep["pair_j_local"], np.int64)[pair_idx]
    pair_rel = np.asarray(prep["pair_rel"], np.int64)[pair_idx]

    def T(x, dtype=None):
        t = torch.from_numpy(np.ascontiguousarray(x))
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)

    return {
        "rows": T(rows),
        "n_graphs": B,
        "atom_feat": torch.from_numpy(np.asarray(cache["atom_feat"])[atom_idx].astype(np.float32)).to(device),
        "atom_batch": T(atom_batch),
        "src": T(src_slot),
        "dst": T(dst_slot),
        "bond_feat": torch.from_numpy(
            np.asarray(cache["bond_feat"])[bond_idx][bond_slot].astype(np.float32)
        ).to(device),
        "bond_slot": T(bond_slot),
        "hist": torch.from_numpy(np.asarray(prep["br_hist"])[br_idx].astype(np.float32)).to(device),
        "pair_root": T(pair_root),
        "pair_i": T(pair_i),
        "pair_j": T(pair_j),
        "pair_rel": T(pair_rel),
        "pair_batch": T(pair_batch),
        "global": torch.from_numpy(np.asarray(cache["global_feat"])[rows].astype(np.float32)).to(device),
        "n_atoms": na.astype(np.float32),
        "n_branches": nb.astype(np.float32),
        "n_atoms_total": int(len(atom_idx)),
    }


def symmetric_stats(x: torch.Tensor, src: torch.Tensor, n_atom: int):
    """Symmetric one/two/three-body products of the branches of each root.

    ``x`` is [n_branch, r], ``src`` maps every branch to its root atom.  Returns
    ``p1 = sum_i x_i``, ``e2 = sum_{i<j} x_i*x_j``, ``e3 = sum_{i<j<l} ...``
    (element-wise, counts preserved; e2/e3 from power sums, O(k)).
    """
    r = x.shape[1]
    p1 = x.new_zeros((n_atom, r)).index_add_(0, src, x)
    p2 = x.new_zeros((n_atom, r)).index_add_(0, src, x * x)
    p3 = x.new_zeros((n_atom, r)).index_add_(0, src, x * x * x)
    e2 = 0.5 * (p1 * p1 - p2)
    e3 = (p1 * p1 * p1 - 3.0 * p1 * p2 + 2.0 * p3) / 6.0
    return p1, e2, e3


def relation_two_body(x, pair_i, pair_j, pair_root, pair_rel, rel_emb, n_atom):
    """``sum_{i<j} g(rel_ij) * (x_i * x_j)`` for the pairs of each root."""
    g = rel_emb(pair_rel)
    return x.new_zeros((n_atom, x.shape[1])).index_add_(
        0, pair_root, g * (x[pair_i] * x[pair_j])
    )


class RBOModel(nn.Module):
    def __init__(self, norm: Mapping[str, object], *, rank: int = RANK, hidden: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.rank = int(rank)
        # generic atom/bond token encoders only (the junction encoder is not
        # used anywhere in RBO, so it is not instantiated: no dead parameters)
        d = 96
        self.atom_encoder = M1.MLP(D.ATOM_FEAT_DIM, 128, d, 1, dropout)
        self.bond_encoder = M1.MLP(3 * d + D.BOND_FEAT_DIM, 160, d, 1, dropout)
        self.hist_proj = nn.Linear(HIST_DIM, 32)
        self.branch = nn.Sequential(
            nn.Linear(3 * d + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, rank), nn.SiLU(),
        )
        self.rel_emb = nn.Embedding(N_REL, rank)
        nn.init.normal_(self.rel_emb.weight, std=0.02)
        self.root = nn.Linear(4 * rank + 5, hidden)
        # audit switches (not parameters): used by the mechanism probes only
        self.register_buffer("three_body_scale", torch.tensor(1.0))
        self.register_buffer("rel_scale", torch.tensor(1.0))
        # 3 pooled vectors + global33 + 11 count features
        reader_in = 3 * hidden + D.GLOBAL_DIM + 11
        self.reader = M1.MLP(reader_in, 256, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    def load_ssl(self, path: Path, device) -> dict:
        state = torch.load(path, map_location="cpu", weights_only=False)
        for name in ("atom_encoder", "bond_encoder"):
            getattr(self, name).load_state_dict(
                {k: v.to(device) for k, v in state[name].items()}
            )
        return {"source": str(path), "sha256": C.sha256_file(path),
                "encoders_loaded": ["atom_encoder", "bond_encoder"]}

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a = self.atom_encoder(batch["atom_feat"])
        src, dst = batch["src"], batch["dst"]
        au, av = a[src], a[dst]
        b = self.bond_encoder(
            torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        )
        z = torch.cat([au, av, b, F.silu(self.hist_proj(batch["hist"]))], dim=1)
        x = self.branch(z)  # [n_branch, r]  >= 0

        n_atom = int(batch["atom_feat"].shape[0])
        p1, e2, e3 = symmetric_stats(x, src, n_atom)
        e3 = e3 * self.three_body_scale
        rel2 = self.rel_scale * relation_two_body(
            x, batch["pair_i"], batch["pair_j"], batch["pair_root"],
            batch["pair_rel"], self.rel_emb, n_atom,
        )
        k = torch.bincount(src, minlength=n_atom).to(x.dtype)
        kfeat = torch.stack([
            k / 4.0, torch.log1p(k), (k >= 2).to(x.dtype),
            (k >= 3).to(x.dtype), (k >= 4).to(x.dtype),
        ], dim=1)

        h = F.silu(self.root(torch.cat([p1, e2, e3, rel2, kfeat], dim=1)))
        atom_batch = batch["atom_batch"]
        B = int(batch["n_graphs"])
        d = h.shape[1]
        sum_h = h.new_zeros((B, d)).index_add_(0, atom_batch, h)
        max_h = h.new_zeros((B, d)).scatter_reduce_(
            0, atom_batch.unsqueeze(1).expand(-1, d), h, reduce="amax", include_self=False
        )
        n_at = k.new_zeros(B).index_add_(0, atom_batch, torch.ones_like(k))
        sq_h = h.new_zeros((B, d)).index_add_(0, atom_batch, h * h) / n_at.unsqueeze(1)

        # generic count features (structure only, auditable)
        def mol_sum(vec, index):
            return x.new_zeros(B).index_add_(0, index, vec)

        rel = batch["pair_rel"]
        pb = batch["pair_batch"]
        extra = torch.stack([
            torch.log1p(x.new_tensor(batch["n_atoms"])),
            torch.log1p(x.new_tensor(batch["n_branches"])),
            torch.log1p(mol_sum((k >= 2).to(x.dtype), atom_batch)),
            torch.log1p(mol_sum((k >= 3).to(x.dtype), atom_batch)),
            torch.log1p(mol_sum((k >= 4).to(x.dtype), atom_batch)),
            torch.log1p(mol_sum((rel % 4 > 0).to(x.dtype), pb)),
            torch.log1p(mol_sum(((rel // 4) % 4 > 0).to(x.dtype), pb)),
            torch.log1p(mol_sum((rel // 16 == 0).to(x.dtype), pb)),
            torch.log1p(mol_sum((rel // 16 == 1).to(x.dtype), pb)),
            torch.log1p(mol_sum((rel // 16 == 2).to(x.dtype), pb)),
            torch.log1p(mol_sum((rel // 16 == 3).to(x.dtype), pb)),
        ], dim=1)

        gstd = (batch["global"] - self.global_mean) / self.global_scale
        return self.reader(torch.cat([sum_h, max_h, sq_h, gstd, extra], dim=1)).view(-1)


def build_model(norm, seed: int, device, ssl_path: Path | None = None):
    ssl_path = Path(C.P3_SSL_S0 if ssl_path is None else ssl_path)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    model = RBOModel(norm).to(device)
    info = {"ssl": model.load_ssl(ssl_path, device)}
    info["parameters"] = C.parameter_count(model)
    if info["parameters"] > 500_000:
        raise RuntimeError("parameter cap exceeded")
    info["requires_grad_true"] = sum(1 for p in model.parameters() if p.requires_grad)
    return model, info
