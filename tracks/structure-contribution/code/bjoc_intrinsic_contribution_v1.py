"""IJCM — Intrinsic Junction Contribution Model (round ``cscl-bjoc-intrinsic-
contribution-feasibility-v1``).

This module defines the ONE fixed interpretable candidate of the round.  It
keeps BJOC's raw chemical inputs and object encoders and replaces the opaque
composer/reader path with an exactly additive scalar contribution algebra:

    y_hat = b_global
          + sum_i alpha_i                      (atom contributions)
          + sum_e beta_e                       (physical-bond contributions)
          + sum_j nu_j                         (junction contributions)
          + sum_k eta_k(global_feature_k)      (one explicit scalar per global feature)

Every per-object score is a *shared* function split into a type baseline and a
context correction (Phase C):

    alpha_i = alpha_type[t(i)] + delta_alpha(context_i)
    beta_e  = beta_type[t(e)]  + delta_beta(context_e)
    nu_j    = nu_type[t(j)]    + delta_nu(context_j)

The prediction is the literal sum of these terms; there is no opaque
graph-level MLP, no interaction head and no residual bypass.  ``sum(all
exported terms) == prediction`` holds by construction up to float32 rounding.

Junction identity: every real junction ``(i, centre, k)`` with ``i < k``
appears exactly once; its two incident physical bonds are carried explicitly.
Endpoint exchange ``(i, centre, k) == (k, centre, i)``; every physical bond
enters ``beta`` exactly once; every atom enters ``alpha`` exactly once.  The
underlying *local evidence* of different branches may overlap (a junction
encoder sees its two bond embeddings) and this is disclosed, not hidden.

Type vocabulary (fit-only, deterministic UNK backoff)
-----------------------------------------------------
* atom type  = raw atom category x degree bucket (0..4)
* bond type  = bond order x ring-bond flag
* junction type = central atom category x unordered incident bond-type pair
  x coarse central role (degree bucket 2/3/4+ x ring-centre flag)

Types with fewer than ``min_count`` fit instances, and any type unseen in the
fit rows (OOV), map to a single deterministic UNK baseline per branch.  No
target ``y``, residual or prediction ever enters the type construction.

Gauge (fit-only, prediction-invariant)
--------------------------------------
For every branch and known type ``t``:

    m_t = mean_fit(delta_raw | type=t)
    type_canonical(t) = type_raw(t) + m_t
    delta_canonical   = delta_raw - m_t

so the per-type fit mean of the canonical context correction is zero and the
prediction and every per-instance total are unchanged.  ``m_t`` is computed
from fit rows only and stored with the checkpoint; it is never used to select
anything.

This module never reads official valid/test, never uses ``g/ell/s/c/k``
supervision and contains no GNN/attention/iterative message passing.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import structural_relation_objects_v1 as sro  # noqa: E402

# ---------------------------------------------------------------------------
# frozen type definitions
# ---------------------------------------------------------------------------

ATOM_TYPE_BASE = sro.ATOM_CATEGORIES          # 28
ATOM_DEGREE_BUCKETS = sro.ATOM_DEGREE_SLOTS   # 5 (degree 0..4, 4 means >=4)
BOND_TYPE_BASE = sro.BOND_CATEGORIES          # 4
JUNC_DEGREE_BUCKETS = 3                       # centre degree 2 / 3 / 4+
JUNC_ROLES = JUNC_DEGREE_BUCKETS * 2          # x ring-centre flag
UNK_INDEX = 0
DEFAULT_MIN_TYPE_COUNT = 20

#: raw junction feature layout offsets (from structural_relation_objects_v1)
_JUNC_RING_SIZE_OFF = 6
_JUNC_BOND_HIST_OFF = _JUNC_RING_SIZE_OFF + sro.RING_SIZE_SLOTS


def raw_atom_type_ids(obj: sro.MolObjects) -> np.ndarray:
    """Per-atom raw type id: atom category * 5 + degree bucket."""
    cats = np.argmax(obj.atom_feat[:, :ATOM_TYPE_BASE], axis=1).astype(np.int64)
    deg = np.rint(obj.atom_feat[:, ATOM_TYPE_BASE : ATOM_TYPE_BASE + ATOM_DEGREE_BUCKETS].sum(axis=1)).astype(np.int64)
    deg = np.where(
        obj.atom_feat[:, ATOM_TYPE_BASE + ATOM_DEGREE_BUCKETS - 1] > 0.5,
        ATOM_DEGREE_BUCKETS - 1,
        deg,
    )
    deg = np.clip(deg, 0, ATOM_DEGREE_BUCKETS - 1)
    return cats * ATOM_DEGREE_BUCKETS + deg


def raw_bond_type_ids(obj: sro.MolObjects) -> np.ndarray:
    """Per-physical-bond raw type id: bond order * 2 + ring-bond flag."""
    order = np.argmax(obj.bond_feat[:, :BOND_TYPE_BASE], axis=1).astype(np.int64)
    ring = (obj.bond_feat[:, BOND_TYPE_BASE] > 0.5).astype(np.int64)
    return order * 2 + ring


def raw_junc_type_ids(obj: sro.MolObjects) -> np.ndarray:
    """Per-junction raw type id: (centre atom category, unordered bond-type pair, role)."""
    n = obj.junc_index.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    centre = obj.junc_index[:, 1]
    cats = np.argmax(obj.atom_feat[:, :ATOM_TYPE_BASE], axis=1).astype(np.int64)[centre]
    t1 = obj.bond_type[obj.junc_bond[:, 0]].astype(np.int64)
    t2 = obj.bond_type[obj.junc_bond[:, 1]].astype(np.int64)
    lo = np.minimum(t1, t2)
    hi = np.maximum(t1, t2)
    pair = lo * BOND_TYPE_BASE + hi  # symmetric unordered pair index (10 values)
    deg_raw = np.rint(obj.junc_feat[:, 0]).astype(np.int64)
    deg_bucket = np.clip(deg_raw - 2, 0, JUNC_DEGREE_BUCKETS - 1)
    ring_centre = (obj.junc_feat[:, 5] > 0.5).astype(np.int64)
    role = deg_bucket * 2 + ring_centre
    return (cats * BOND_TYPE_BASE * BOND_TYPE_BASE + pair) * JUNC_ROLES + role


def atom_type_string(raw_id: int) -> str:
    cat, deg = divmod(int(raw_id), ATOM_DEGREE_BUCKETS)
    return f"atom:{cat}:deg{deg}{'+' if deg == ATOM_DEGREE_BUCKETS - 1 else ''}"


def bond_type_string(raw_id: int) -> str:
    order, ring = divmod(int(raw_id), 2)
    return f"bond:{order}:ring{ring}"


def junc_type_string(raw_id: int) -> str:
    rest, role = divmod(int(raw_id), JUNC_ROLES)
    cat, pair = divmod(rest, BOND_TYPE_BASE * BOND_TYPE_BASE)
    lo, hi = divmod(pair, BOND_TYPE_BASE)
    deg_bucket, ring = divmod(role, 2)
    deg = deg_bucket + 2
    return f"junc:{cat}:b{lo}-b{hi}:deg{deg}{'+' if deg_bucket == JUNC_DEGREE_BUCKETS - 1 else ''}:ring{ring}"


TYPE_STRINGERS = {"atom": atom_type_string, "bond": bond_type_string, "junction": junc_type_string}
RAW_TYPE_FN = {"atom": raw_atom_type_ids, "bond": raw_bond_type_ids, "junction": raw_junc_type_ids}


# ---------------------------------------------------------------------------
# fit-only type vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeVocab:
    """Deterministic fit-only vocabulary with a single UNK bucket per branch."""

    kind: str
    min_count: int
    raw_to_index: dict[int, int]      # known raw types -> 1..V
    raw_counts: dict[int, int]        # all observed fit raw types -> count
    n_fit_instances: int
    n_fit_rows: int

    @property
    def size(self) -> int:
        """Number of embedding rows: UNK (0) + known types."""
        return 1 + len(self.raw_to_index)

    @property
    def known_raw_ids(self) -> list[int]:
        return sorted(self.raw_to_index)

    def map_raw(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.int64)
        out = np.full(raw.shape, UNK_INDEX, dtype=np.int64)
        lut = np.full(int(raw.max()) + 1 if raw.size else 1, UNK_INDEX, dtype=np.int64)
        for r, i in self.raw_to_index.items():
            if 0 <= r < lut.shape[0]:
                lut[r] = i
        np.take(lut, raw, out=out)
        return out

    def is_oov_mask(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.int64)
        return ~np.isin(raw, np.asarray(self.known_raw_ids, dtype=np.int64)) if self.known_raw_ids else np.ones(raw.shape, bool)

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "min_count": int(self.min_count),
            "size": int(self.size),
            "n_known_types": len(self.raw_to_index),
            "n_fit_instances": int(self.n_fit_instances),
            "n_fit_rows": int(self.n_fit_rows),
            "raw_counts": {str(k): int(v) for k, v in sorted(self.raw_counts.items())},
            "type_strings": {
                str(k): TYPE_STRINGERS[self.kind](k) for k in sorted(self.raw_to_index)
            },
            "unk_rule": f"fit count < {self.min_count} or unseen in fit -> index 0 (UNK)",
        }

    @staticmethod
    def build(kind: str, objs: Sequence[sro.MolObjects], fit_rows: Sequence[int], min_count: int = DEFAULT_MIN_TYPE_COUNT) -> "TypeVocab":
        fn = RAW_TYPE_FN[kind]
        counts: dict[int, int] = {}
        n_inst = 0
        for r in fit_rows:
            raw = fn(objs[int(r)])
            n_inst += int(raw.size)
            if raw.size:
                vals, cnts = np.unique(raw, return_counts=True)
                for v, c in zip(vals.tolist(), cnts.tolist()):
                    counts[int(v)] = counts.get(int(v), 0) + int(c)
        keep = sorted(k for k, c in counts.items() if c >= int(min_count))
        raw_to_index = {k: i + 1 for i, k in enumerate(keep)}
        return TypeVocab(
            kind=kind,
            min_count=int(min_count),
            raw_to_index=raw_to_index,
            raw_counts=counts,
            n_fit_instances=int(n_inst),
            n_fit_rows=len(fit_rows),
        )


def type_id_arrays(objs: Sequence[sro.MolObjects], rows: Sequence[int]) -> dict[str, list[np.ndarray]]:
    return {kind: [RAW_TYPE_FN[kind](objs[int(r)]) for r in rows] for kind in ("atom", "bond", "junction")}


@dataclass
class TypeVocabs:
    atom: TypeVocab
    bond: TypeVocab
    junction: TypeVocab

    @property
    def sizes(self) -> dict[str, int]:
        return {"atom": self.atom.size, "bond": self.bond.size, "junction": self.junction.size}

    def as_payload(self) -> dict[str, Any]:
        return {
            "atom": self.atom.as_payload(),
            "bond": self.bond.as_payload(),
            "junction": self.junction.as_payload(),
            "min_type_count": DEFAULT_MIN_TYPE_COUNT,
        }

    @staticmethod
    def build(objs: Sequence[sro.MolObjects], fit_rows: Sequence[int], min_count: int = DEFAULT_MIN_TYPE_COUNT) -> "TypeVocabs":
        return TypeVocabs(
            atom=TypeVocab.build("atom", objs, fit_rows, min_count),
            bond=TypeVocab.build("bond", objs, fit_rows, min_count),
            junction=TypeVocab.build("junction", objs, fit_rows, min_count),
        )


# ---------------------------------------------------------------------------
# batch construction
# ---------------------------------------------------------------------------


def build_ijcm_batch(objs: Sequence[sro.MolObjects], rows: Sequence[int], vocabs: TypeVocabs) -> dict[str, torch.Tensor]:
    """One-shot batch: the base BJOC object tensors plus fit-vocabulary type indices."""
    base = sro.batch_objects([objs[int(r)] for r in rows])
    raw = type_id_arrays(objs, rows)
    for kind in ("atom", "bond", "junction"):
        arrs = raw[kind]
        raw_cat = np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.int64)
        base[f"{kind}_raw_type"] = torch.as_tensor(raw_cat, dtype=torch.long)
        base[f"{kind}_type"] = torch.as_tensor(vocabs.__getattribute__(kind).map_raw(raw_cat), dtype=torch.long)
    return base


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Same plain MLP as the frozen BJOC backbone (Linear -> SiLU, repeated)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_hidden: int = 1, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * n_hidden + [out_dim]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
                if self.dropout > 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


@dataclass(frozen=True)
class IJCMConfig:
    """Frozen widths.  Encoders are copied from ``BJOCConfig``; the contribution
    heads are sized so that the total parameter count is close to the published
    BJOC-Null backbone (see ``parameter_and_init_audit.json``)."""

    atom_hidden: int = 128
    atom_out: int = 96
    bond_hidden: int = 160
    bond_out: int = 96
    junc_hidden: int = 160
    junc_out: int = 96
    contrib_hidden: int = 704
    eta_hidden: int = 8
    dropout: float = 0.0

    @property
    def atom_in(self) -> int:
        return sro.ATOM_FEAT_DIM

    @property
    def bond_in(self) -> int:
        return 3 * self.atom_out + sro.BOND_FEAT_DIM

    @property
    def junc_in(self) -> int:
        return 4 * self.atom_out + 2 * self.bond_out + sro.JUNC_FEAT_DIM

    def inventory(self, vocab_sizes: Mapping[str, int]) -> dict[str, int]:
        def lin(i: int, o: int) -> int:
            return i * o + o

        atom_enc = lin(self.atom_in, self.atom_hidden) + lin(self.atom_hidden, self.atom_out)
        bond_enc = lin(self.bond_in, self.bond_hidden) + lin(self.bond_hidden, self.bond_out)
        junc_enc = lin(self.junc_in, self.junc_hidden) + lin(self.junc_hidden, self.junc_out)
        alpha = lin(self.atom_out, self.contrib_hidden) + lin(self.contrib_hidden, 1)
        beta = lin(self.bond_out, self.contrib_hidden) + lin(self.contrib_hidden, 1)
        nu = lin(self.junc_out, self.contrib_hidden) + lin(self.contrib_hidden, 1)
        eta = sro.GLOBAL_DIM * (lin(1, self.eta_hidden) + lin(self.eta_hidden, 1))
        type_baselines = sum(int(vocab_sizes[k]) for k in ("atom", "bond", "junction"))
        return {
            "atom_encoder": atom_enc,
            "bond_encoder": bond_enc,
            "junction_encoder": junc_enc,
            "alpha_head": alpha,
            "beta_head": beta,
            "nu_head": nu,
            "eta_heads": eta,
            "type_baselines": type_baselines,
            "global_bias": 1,
            "total": atom_enc + bond_enc + junc_enc + alpha + beta + nu + eta + type_baselines + 1,
        }


class IJCM(nn.Module):
    """Exactly additive object-contribution predictor (see module docstring)."""

    def __init__(
        self,
        config: IJCMConfig,
        vocab_sizes: Mapping[str, int],
        global_mean: np.ndarray,
        global_scale: np.ndarray,
    ):
        super().__init__()
        self.config = config
        # ---- encoders: identical construction to the frozen BJOC backbone ----
        self.atom_encoder = _MLP(config.atom_in, config.atom_hidden, config.atom_out, 1, config.dropout)
        self.bond_encoder = _MLP(config.bond_in, config.bond_hidden, config.bond_out, 1, config.dropout)
        self.junction_encoder = _MLP(config.junc_in, config.junc_hidden, config.junc_out, 1, config.dropout)
        # ---- scalar contribution heads (shared per object kind) --------------
        self.alpha_head = _MLP(config.atom_out, config.contrib_hidden, 1, 1, config.dropout)
        self.beta_head = _MLP(config.bond_out, config.contrib_hidden, 1, 1, config.dropout)
        self.nu_head = _MLP(config.junc_out, config.contrib_hidden, 1, 1, config.dropout)
        # ---- explicit per-global-feature scalar heads ------------------------
        self.eta_heads = nn.ModuleList(
            [_MLP(1, config.eta_hidden, 1, 1, 0.0) for _ in range(sro.GLOBAL_DIM)]
        )
        # ---- type baselines (one scalar per vocabulary row; row 0 = UNK) -----
        self.atom_type_baseline = nn.Embedding(int(vocab_sizes["atom"]), 1)
        self.bond_type_baseline = nn.Embedding(int(vocab_sizes["bond"]), 1)
        self.junc_type_baseline = nn.Embedding(int(vocab_sizes["junction"]), 1)
        for emb in (self.atom_type_baseline, self.bond_type_baseline, self.junc_type_baseline):
            nn.init.zeros_(emb.weight)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("global_mean", torch.as_tensor(global_mean, dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(global_scale, dtype=torch.float32))
        self.vocab_sizes = {k: int(v) for k, v in vocab_sizes.items()}

    # -- encoders (same one-shot gathers as BJOC) ---------------------------
    def encode_atoms(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.atom_encoder(batch["atom_feat"])

    def encode_bonds(self, a: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        feats = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        return self.bond_encoder(feats)

    def encode_junctions(self, a: torch.Tensor, b: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        idx = batch["junc_index"]
        jb = batch["junc_bond"]
        ai, aj, ak = a[idx[:, 0]], a[idx[:, 1]], a[idx[:, 2]]
        b1, b2 = b[jb[:, 0]], b[jb[:, 1]]
        feats = torch.cat(
            [
                ai + ak,
                (ai - ak).abs(),
                ai * ak,
                aj,
                b1 + b2,
                (b1 - b2).abs(),
                batch["junc_feat"],
            ],
            dim=1,
        )
        return self.junction_encoder(feats)

    # -- raw additive terms --------------------------------------------------
    def raw_terms(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)
        alpha_type = self.atom_type_baseline(batch["atom_type"]).view(-1)
        beta_type = self.bond_type_baseline(batch["bond_type"]).view(-1)
        nu_type = self.junc_type_baseline(batch["junction_type"]).view(-1)
        alpha_ctx = self.alpha_head(a).view(-1)
        beta_ctx = self.beta_head(b).view(-1)
        nu_ctx = self.nu_head(w).view(-1)
        n_graphs = int(batch["n_graphs"])
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        eta_per_feature = torch.cat(
            [head(g[:, k : k + 1]).view(n_graphs, 1) for k, head in enumerate(self.eta_heads)], dim=1
        )
        return {
            "a": a,
            "b": b,
            "w": w,
            "alpha_type_raw": alpha_type,
            "alpha_ctx_raw": alpha_ctx,
            "beta_type_raw": beta_type,
            "beta_ctx_raw": beta_ctx,
            "nu_type_raw": nu_type,
            "nu_ctx_raw": nu_ctx,
            "eta_per_feature": eta_per_feature,
            "bias": self.global_bias.view(()).expand(n_graphs),
        }

    def sum_terms(self, terms: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        n_graphs = int(batch["n_graphs"])
        dtype = terms["alpha_ctx_raw"].dtype

        def _scatter(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(n_graphs, device=values.device, dtype=dtype)
            return out.index_add(0, index, values.to(dtype))

        pred = terms["bias"].to(dtype) + _scatter(terms["alpha_type_raw"] + terms["alpha_ctx_raw"], batch["atom_batch"])
        pred = pred + _scatter(terms["beta_type_raw"] + terms["beta_ctx_raw"], batch["bond_batch"])
        pred = pred + _scatter(terms["nu_type_raw"] + terms["nu_ctx_raw"], batch["junc_batch"])
        pred = pred + terms["eta_per_feature"].to(dtype).sum(dim=1)
        return pred

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        terms = self.raw_terms(batch)
        return self.sum_terms(terms, batch)


# ---------------------------------------------------------------------------
# gauge normalization (fit-only, prediction preserving)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_fit_offsets(
    model: IJCM,
    objs: Sequence[sro.MolObjects],
    fit_rows: Sequence[int],
    vocabs: TypeVocabs,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Per-branch, per-type fit mean of the raw context correction (float64).

    Types with zero fit instances keep offset 0.0; the counts are reported by
    the caller.  The model is switched to eval and restored afterwards.
    """
    was_training = model.training
    model.eval()
    sums = {k: np.zeros(getattr(vocabs, k).size, dtype=np.float64) for k in ("atom", "bond", "junction")}
    counts = {k: np.zeros(getattr(vocabs, k).size, dtype=np.int64) for k in ("atom", "bond", "junction")}
    for start in range(0, len(fit_rows), batch_size):
        rows = list(fit_rows[start : start + batch_size])
        batch = build_ijcm_batch(objs, rows, vocabs)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        terms = model.raw_terms(batch)
        for kind, key in (("atom", "alpha_ctx_raw"), ("bond", "beta_ctx_raw"), ("junction", "nu_ctx_raw")):
            idx = batch[f"{kind}_type"].detach().cpu().numpy()
            vals = terms[key].detach().float().cpu().numpy().astype(np.float64)
            np.add.at(sums[kind], idx, vals)
            np.add.at(counts[kind], idx, 1)
    if was_training:
        model.train()
    offsets: dict[str, np.ndarray] = {}
    for kind in ("atom", "bond", "junction"):
        off = np.zeros_like(sums[kind])
        present = counts[kind] > 0
        off[present] = sums[kind][present] / counts[kind][present]
        offsets[kind] = off
    return offsets


def gauge_terms(
    terms: Mapping[str, torch.Tensor],
    offsets: Mapping[str, np.ndarray],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Canonical (gauged) terms in float64; per-instance totals are invariant.

    Returns flat numpy arrays for atoms / bonds / junctions and per-graph eta
    and bias.
    """
    out: dict[str, np.ndarray] = {}
    for kind, head in (("atom", "alpha"), ("bond", "beta"), ("junction", "nu")):
        off = np.asarray(offsets[kind], dtype=np.float64)
        t_raw = terms[f"{head}_type_raw"].detach().float().cpu().numpy().astype(np.float64)
        c_raw = terms[f"{head}_ctx_raw"].detach().float().cpu().numpy().astype(np.float64)
        idx = batch[f"{kind}_type"].detach().cpu().numpy()
        assert off.shape[0] == int(idx.max()) + 1 or int(idx.max()) < off.shape[0]
        m = off[idx]
        out[f"{head}_type_raw"] = t_raw
        out[f"{head}_ctx_raw"] = c_raw
        out[f"{head}_type_canonical"] = t_raw + m
        out[f"{head}_ctx_canonical"] = c_raw - m
        out[f"{head}_score_canonical"] = t_raw + c_raw
        out[f"{head}_type_index"] = idx
    out["eta_per_feature"] = terms["eta_per_feature"].detach().float().cpu().numpy().astype(np.float64)
    out["bias"] = float(terms["bias"][0].detach().float().cpu())
    return out


def verify_conservation(
    model: IJCM,
    objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    vocabs: TypeVocabs,
    offsets: Mapping[str, np.ndarray] | None = None,
    device: torch.device | str = "cpu",
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Check pred == bias + sum(alpha) + sum(beta) + sum(nu) + sum(eta) exactly."""
    model.eval()
    batch = build_ijcm_batch(objs, rows, vocabs)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        terms = model.raw_terms(batch)
        pred = model.sum_terms(terms, batch).float().cpu().numpy().astype(np.float64)
        if offsets is None:
            offsets = {k: np.zeros(getattr(vocabs, k).size) for k in ("atom", "bond", "junction")}
        can = gauge_terms(terms, offsets, batch)
    n_graphs = int(batch["n_graphs"])
    rec = np.full(n_graphs, float(can["bias"]), dtype=np.float64)
    for head, kind in (("alpha", "atom"), ("beta", "bond"), ("nu", "junc")):
        rec += np.bincount(
            batch[f"{kind}_batch"].cpu().numpy(),
            weights=can[f"{head}_score_canonical"],
            minlength=n_graphs,
        )
    rec += can["eta_per_feature"].sum(axis=1)
    # canonical per-instance totals must equal the raw totals
    inst_gap = 0.0
    for head in ("alpha", "beta", "nu"):
        gap = np.abs(
            (can[f"{head}_type_canonical"] + can[f"{head}_ctx_canonical"]) - can[f"{head}_score_canonical"]
        ).max()
        inst_gap = max(inst_gap, float(gap))
    return {
        "n_molecules": n_graphs,
        "max_abs_pred_recomposed_gap": float(np.abs(rec - pred).max()),
        "max_abs_instance_identity_gap": float(inst_gap),
        "atol": float(atol),
        "ok": bool(np.abs(rec - pred).max() <= atol and inst_gap <= atol),
    }


# ---------------------------------------------------------------------------
# builders / audits
# ---------------------------------------------------------------------------


def build_model(
    *,
    seed: int,
    vocab_sizes: Mapping[str, int],
    global_mean: np.ndarray,
    global_scale: np.ndarray,
    config: IJCMConfig | None = None,
    device: torch.device | str = "cpu",
) -> IJCM:
    torch.manual_seed(int(seed))
    cfg = config or IJCMConfig()
    model = IJCM(cfg, vocab_sizes, global_mean, global_scale)
    return model.to(device)


def parameter_audit(model: IJCM) -> dict[str, Any]:
    inv = model.config.inventory(model.vocab_sizes)
    actual = int(sum(p.numel() for p in model.parameters()))
    return {
        "inventory": inv,
        "actual_parameters": actual,
        "parameter_exact_vs_inventory": actual == inv["total"],
        "vocab_sizes": dict(model.vocab_sizes),
    }


def state_dict_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def expression_diff(bjoc_inventory: Mapping[str, int], ijcm_inventory: Mapping[str, int]) -> dict[str, Any]:
    """Human-readable account of what differs between B and S (no hidden bypass)."""
    return {
        "shared": [
            "raw atom/bond/junction object tensors (structural_relation_objects_v1)",
            "atom encoder 108->128->96",
            "bond encoder (au+av, |au-av|, au*av, raw bond) ->160->96",
            "junction encoder (ai+ak, |ai-ak|, i*k, aj, b1+b2, |b1-b2|, raw junction) ->160->96",
            "global topology vector (33 dims), fit-only standardisation",
            "raw-y L1 supervision, Adam recipe, last-5 parameter soup",
        ],
        "bjoc_only": [
            "static incidence grouping to per-bond moments W_b / V_b (null mode: graph means)",
            "bond composer 296->160->96",
            "opaque graph reader (z, a_sum, a_sq, g) -> 256 -> 256 -> 1",
        ],
        "ijcm_only": [
            "per-object scalar contribution heads (type baseline + context correction)",
            f"per-global-feature scalar eta heads ({len(sro.GLOBAL_FEAT_NAMES)} features)",
            "exact additive graph aggregation",
            "fit-only per-type gauge offsets (post-training, prediction-invariant)",
        ],
        "parameters": {
            "bjoc_total": int(bjoc_inventory["total"]),
            "ijcm_total": int(ijcm_inventory["total"]),
            "ratio": float(ijcm_inventory["total"] / bjoc_inventory["total"]),
            "note": (
                "the two totals are within a few percent; the encoders and inputs are shared, so the "
                "difference is dominated by the shared composer/reader vs the contribution heads. "
                "A residual capacity difference is disclosed and cannot be attributed to additivity alone."
            ),
        },
    }
