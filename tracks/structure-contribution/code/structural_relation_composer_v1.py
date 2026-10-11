"""BJOC — Bond-Junction Organized Composer (cscl-structural-relation-composer-v1).

One-shot, non-iterative structural predictor.  See
``notes/structural_relation_composer_v1.md`` for the full design and the
pre-registered control arms.

Compute DAG (strictly one-shot; no state is ever refreshed from neighbours)::

    raw graph
      -> atom features (fixed, label-blind)          [structural_relation_objects_v1]
      -> a_i = ATOM(a_i_raw)                         shared atom MLP
      -> b_uv = BOND(a_u, a_v, raw bond)             shared bond MLP (symmetric)
      -> w_ijk = JUNC(a_i, a_j, a_k, b_ij, b_jk)     shared junction MLP (symmetric)
      -> incidence grouping (static, no learning):
             W_b = sum_{j touches b} w_j ,  V_b = sum w_j^2 ,  n_b = #junctions(b)
         [Arm N (null): W_b = n_b * graph_mean(w), V_b = n_b * graph_mean(w^2)]
      -> c_b = PSI(b_b, W_b, V_b, log1p(n_b), raw bond)
      -> Z   = [ sum_b c_b , sum_i a_i , sum_i a_i^2 , global topology ]
      -> y_hat = READER(Z)

There is no message passing, attention, Transformer block or recurrent
update: ``a`` is used to build ``b``, ``b`` is used to build ``w``, and the
grouping step never feeds back into ``a``/``b``/``w``.

The two arms differ **only** through ``organization_mode``:

* ``organized`` — true junction-to-bond ownership (W_b/V_b of the junctions
  that actually touch bond b);
* ``null``      — the same junction multiset, the same per-bond junction
  counts, the same trainable parameters, but every bond receives the graph's
  *mean* junction content scaled by its own count.  Graph-level first and
  second moments of ``w`` are exactly conserved:
  ``sum_b W_b_null = sum_b W_b_real`` and ``sum_b V_b_null = sum_b V_b_real``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
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

ORGANIZED = "organized"
NULL = "null"
ORGANIZATION_MODES = (ORGANIZED, NULL)

FORBIDDEN_TOKENS = (
    "MessagePassing",
    "GCN",
    "GIN",
    "GAT",
    "SAGEConv",
    "Transformer",
    "MultiheadAttention",
    "nn.MultiheadAttention",
    "self_attention",
    "TransformerEncoder",
    "GRU",
    "LSTM",
    "RNN",
    "Sequential_state",
)


# ---------------------------------------------------------------------------
# config + model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BJOCConfig:
    """Frozen model widths (protocol §5).  Set before any formal GPU run."""

    atom_hidden: int = 128
    atom_out: int = 96
    bond_hidden: int = 160
    bond_out: int = 96
    junc_hidden: int = 160
    junc_out: int = 96
    compose_hidden: int = 160
    compose_out: int = 96
    reader_hidden: int = 256
    reader_layers: int = 2
    dropout: float = 0.0

    #: derived input widths
    @property
    def atom_in(self) -> int:
        return sro.ATOM_FEAT_DIM

    @property
    def bond_in(self) -> int:
        return 3 * self.atom_out + sro.BOND_FEAT_DIM

    @property
    def junc_in(self) -> int:
        return 4 * self.atom_out + 2 * self.bond_out + sro.JUNC_FEAT_DIM

    @property
    def compose_in(self) -> int:
        return self.bond_out + 2 * self.junc_out + 1 + sro.BOND_FEAT_DIM

    @property
    def reader_in(self) -> int:
        return self.compose_out + 2 * self.atom_out + sro.GLOBAL_DIM

    def inventory(self) -> dict[str, int]:
        def lin(i: int, o: int) -> int:
            return i * o + o

        atom = lin(self.atom_in, self.atom_hidden) + lin(self.atom_hidden, self.atom_out)
        bond = lin(self.bond_in, self.bond_hidden) + lin(self.bond_hidden, self.bond_out)
        junc = lin(self.junc_in, self.junc_hidden) + lin(self.junc_hidden, self.junc_out)
        compose = lin(self.compose_in, self.compose_hidden) + lin(self.compose_hidden, self.compose_out)
        reader = lin(self.reader_in, self.reader_hidden)
        for _ in range(self.reader_layers - 1):
            reader += lin(self.reader_hidden, self.reader_hidden)
        reader += lin(self.reader_hidden, 1)
        return {
            "atom_encoder": atom,
            "bond_encoder": bond,
            "junction_encoder": junc,
            "group_composer": compose,
            "reader": reader,
            "total": atom + bond + junc + compose + reader,
        }


class _MLP(nn.Module):
    """Plain shared MLP: Linear -> SiLU (repeated), optional dropout."""

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


class BJOC(nn.Module):
    """Bond-Junction Organized Composer (one-shot; see module docstring)."""

    def __init__(self, config: BJOCConfig, global_mean: np.ndarray, global_scale: np.ndarray, organization_mode: str = ORGANIZED):
        super().__init__()
        if organization_mode not in ORGANIZATION_MODES:
            raise ValueError(f"unknown organization_mode {organization_mode!r}")
        self.config = config
        self.organization_mode = organization_mode
        self.atom_encoder = _MLP(config.atom_in, config.atom_hidden, config.atom_out, 1, config.dropout)
        self.bond_encoder = _MLP(config.bond_in, config.bond_hidden, config.bond_out, 1, config.dropout)
        self.junction_encoder = _MLP(config.junc_in, config.junc_hidden, config.junc_out, 1, config.dropout)
        self.group_composer = _MLP(config.compose_in, config.compose_hidden, config.compose_out, 1, config.dropout)
        self.reader = _MLP(config.reader_in, config.reader_hidden, 1, config.reader_layers, config.dropout)
        self.register_buffer("global_mean", torch.as_tensor(global_mean, dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(global_scale, dtype=torch.float32))

    # -- one-shot object encoders -------------------------------------------
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

    # -- static incidence grouping (no learning) -----------------------------
    def group_to_bonds(
        self,
        w: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (W_b, V_b, n_b) with W/V of shape [n_bonds, junc_out]."""
        n_bonds = int(batch["bond_feat"].shape[0])
        jb = batch["junc_bond"]
        jb_flat = jb.reshape(-1)
        n_b = torch.zeros(n_bonds, device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb_flat, torch.ones_like(jb_flat, dtype=w.dtype))
        if mode == ORGANIZED:
            W = torch.zeros((n_bonds, w.shape[1]), device=w.device, dtype=w.dtype)
            V = torch.zeros_like(W)
            W.index_add_(0, jb[:, 0], w)
            W.index_add_(0, jb[:, 1], w)
            w2 = w * w
            V.index_add_(0, jb[:, 0], w2)
            V.index_add_(0, jb[:, 1], w2)
            return W, V, n_b
        if mode == NULL:
            n_graphs = int(batch["n_graphs"])
            junc_batch = batch["junc_batch"]
            w_sum = torch.zeros((n_graphs, w.shape[1]), device=w.device, dtype=w.dtype)
            w2_sum = torch.zeros_like(w_sum)
            w_sum.index_add_(0, junc_batch, w)
            w2_sum.index_add_(0, junc_batch, w * w)
            counts = torch.bincount(junc_batch, minlength=n_graphs).to(w.dtype).clamp(min=1.0).unsqueeze(1)
            mean_w = w_sum / counts
            mean_w2 = w2_sum / counts
            bond_batch = batch["bond_batch"]
            W = n_b.unsqueeze(1) * mean_w[bond_batch]
            V = n_b.unsqueeze(1) * mean_w2[bond_batch]
            return W, V, n_b
        raise ValueError(f"unknown grouping mode {mode!r}")

    # -- forward -------------------------------------------------------------
    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = self.organization_mode if organization_mode is None else organization_mode
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)
        W, V, n_b = self.group_to_bonds(w, batch, mode)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]],
            dim=1,
        )
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        return {
            "a": a,
            "b": b,
            "w": w,
            "W": W,
            "V": V,
            "n_b": n_b,
            "c": c,
            "z": z,
            "reader_in": reader_in,
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = self.forward_features(batch, organization_mode)
        pred = self.reader(feats["reader_in"]).view(-1)
        if return_features:
            return pred, feats
        return pred


# ---------------------------------------------------------------------------
# builders / audits
# ---------------------------------------------------------------------------


def build_bjoc(
    config: BJOCConfig,
    *,
    seed: int,
    organization_mode: str,
    global_mean: np.ndarray,
    global_scale: np.ndarray,
    device: torch.device | str = "cpu",
) -> BJOC:
    """Deterministic builder: identical seed + config -> identical parameters
    for both organization modes (mode never touches initialisation)."""
    torch.manual_seed(int(seed))
    model = BJOC(config, global_mean, global_scale, organization_mode)
    return model.to(device)


def state_dict_hash(model: nn.Module) -> str:
    import hashlib

    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def parameter_audit(config: BJOCConfig, model: BJOC) -> dict[str, Any]:
    expected = config.inventory()
    actual = int(sum(p.numel() for p in model.parameters()))
    return {
        "expected": expected,
        "actual_parameters": actual,
        "parameter_exact": actual == expected["total"],
    }


def forbidden_module_scan(model: nn.Module) -> list[str]:
    """Static scan for GNN/attention/recursion tokens (correctness gate 16)."""
    found = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        for token in FORBIDDEN_TOKENS:
            if token in cls:
                found.append(f"{name}:{cls}")
    src = Path(__file__).read_text(encoding="utf-8")
    # drop the token-list definition itself before scanning for usage; the
    # scan tokens below are assembled at runtime so this scanner does not
    # match its own literals
    start = src.find("FORBIDDEN_TOKENS = (")
    end = src.find(")\n", start)
    body = src[:start] + src[end + 2 :]
    usage_tokens = ["nn." + "MultiheadAttention", "Message" + "Passing(", "Transformer" + "Encoder(", "nn." + "Transformer"]
    for token in usage_tokens:
        if token in body:
            found.append(f"source:{token}")
    return found


# ---------------------------------------------------------------------------
# synthetic organisation experiments (Phase D; CPU only)
# ---------------------------------------------------------------------------


@dataclass
class SyntheticInstance:
    """A tiny raw molecular graph + its targets (fixed functions of structure)."""

    atom_types: list[int]
    edges: list[tuple[int, int, int]]  # (u, v, bond_type), undirected, u<v
    target_incidence: float
    target_marginal: float


@dataclass
class ObjectInstance:
    """A static bond/junction incidence system (no atoms needed).

    * ``bond_tokens[b]``  — feature token of bond ``b``
    * ``junc_tokens[j]``  — feature token of junction ``j``
    * ``incidence[j]``    — the two (distinct) bonds junction ``j`` touches
    """

    bond_tokens: list[int]
    junc_tokens: list[int]
    incidence: list[tuple[int, int]]
    target_incidence: float
    target_marginal: float


def _junction_token(center: int, left: int, right: int, t1: int, t2: int) -> int:
    """A small deterministic token for one molecular junction."""
    endpoint_pair = tuple(sorted((left, right)))
    return (
        1
        + 7 * center
        + 3 * endpoint_pair[0]
        + 5 * endpoint_pair[1]
        + 11 * (t1 + t2)
        + 13 * abs(t1 - t2)
    )


def synthetic_targets(obj: sro.MolObjects, atom_types: Sequence[int], bond_types: Sequence[int]) -> tuple[float, float]:
    """Targets that do / do not depend on junction-to-bond ownership.

    * ``target_marginal``  = sum over junctions of the junction token
      (depends only on the junction multiset).
    * ``target_incidence`` = sum over *double* bonds of the tokens of the
      junctions that actually touch that bond (depends on ownership).
    """
    marginal = 0.0
    for idx, (i, j, k) in enumerate(obj.junc_index):
        b1, b2 = obj.junc_bond[idx]
        marginal += _junction_token(atom_types[j], atom_types[i], atom_types[k], bond_types[b1], bond_types[b2])
    per_bond: list[list[float]] = [[] for _ in range(obj.bond_index.shape[0])]
    for idx, (j1, j2) in enumerate(obj.junc_bond):
        i, j, k = obj.junc_index[idx]
        token = float(
            _junction_token(atom_types[j], atom_types[i], atom_types[k], bond_types[j1], bond_types[j2])
        )
        per_bond[j1].append(token)
        per_bond[j2].append(token)
    incidence = 0.0
    for b, tokens in enumerate(per_bond):
        if bond_types[b] != 1:
            continue
        incidence += float((1.0 + sum(tokens)) ** 2)
    return float(incidence), float(marginal)


# -- object-level instances (the clean mechanism experiment) ----------------


def object_targets(
    bond_tokens: Sequence[int], junc_tokens: Sequence[int], incidence: Sequence[tuple[int, int]]
) -> tuple[float, float]:
    """Incidence-dependent and marginal-only targets for an object system.

    ``target_incidence`` is *nonlinear in the per-bond junction multiset*
    (squared sum of incident junction tokens, weighted by the bond's own
    token), so it changes under a marginal-preserving ownership switch.
    ``target_marginal`` is a linear function of the two token multisets.
    """
    per_bond: list[list[int]] = [[] for _ in bond_tokens]
    for j, (a, b) in enumerate(incidence):
        per_bond[a].append(int(junc_tokens[j]))
        per_bond[b].append(int(junc_tokens[j]))
    inc = 0.0
    for b, tokens in enumerate(per_bond):
        c0 = float(sum(1 for t in tokens if t == 0))
        c1 = float(sum(1 for t in tokens if t == 1))
        inc += (bond_tokens[b] + 1) * (c0 - c1) ** 2 + (c0 * c1 if bond_tokens[b] == 1 else 0.0)
    mar = float(sum(3 * (t + 1) ** 2 for t in junc_tokens)) + float(sum(t + 1 for t in bond_tokens))
    return float(inc), float(mar)


def random_object_instance(
    rng: np.random.Generator, n_bonds: int, n_junc: int, n_tokens: int = 3
) -> ObjectInstance:
    """Random bond/junction incidence system (each junction touches 2 bonds)."""
    bond_tokens = [int(rng.integers(0, n_tokens)) for _ in range(n_bonds)]
    junc_tokens = [int(rng.integers(0, n_tokens)) for _ in range(n_junc)]
    incidence = []
    for _ in range(n_junc):
        a = int(rng.integers(0, n_bonds))
        b = int(rng.integers(0, n_bonds - 1))
        if b >= a:
            b += 1
        incidence.append((a, b))
    inc, mar = object_targets(bond_tokens, junc_tokens, incidence)
    return ObjectInstance(bond_tokens, junc_tokens, incidence, inc, mar)


def counterexample_object_pair() -> tuple[ObjectInstance, ObjectInstance]:
    """Explicit marginal-preserving ownership switch (the numeric witness).

    Four bonds P1,P2 (token P=0), Q1,Q2 (token Q=1); four junctions
    X1,X2,Y1,Y2 (tokens X=0,X=0,Y=1,Y=1); every bond touches exactly two
    junctions.

    * A: X1={P1,Q1}, X2={P1,Q2}, Y1={P2,Q1}, Y2={P2,Q2}
         -> per-bond junction tokens: P1{XX} P2{YY} Q1{XY} Q2{XY}
    * B: X1={P1,Q1}, X2={P2,Q2}, Y1={P1,Q2}, Y2={P2,Q1}
         -> per-bond junction tokens: all {XY}

    Bond multiset {P,P,Q,Q}, junction multiset {X,X,Y,Y}, per-bond counts
    {2,2,2,2}: identical.  Ownership differs.
    """
    bonds = [0, 0, 1, 1]
    junc = [0, 0, 1, 1]
    a = ObjectInstance(bonds, junc, [(0, 2), (0, 3), (1, 2), (1, 3)], 0.0, 0.0)
    b = ObjectInstance(bonds, junc, [(0, 2), (1, 3), (0, 3), (1, 2)], 0.0, 0.0)
    a.target_incidence, a.target_marginal = object_targets(a.bond_tokens, a.junc_tokens, a.incidence)
    b.target_incidence, b.target_marginal = object_targets(b.bond_tokens, b.junc_tokens, b.incidence)
    return a, b


def per_bond_junction_tokens(inst: ObjectInstance) -> list:
    """Marginal-preserving ownership fingerprint: per-bond junction tokens."""
    out = [[] for _ in inst.bond_tokens]
    for j, (a, b) in enumerate(inst.incidence):
        out[a].append(inst.junc_tokens[j])
        out[b].append(inst.junc_tokens[j])
    return [tuple(sorted(x)) for x in out]


def object_marginal_signature(inst: ObjectInstance) -> tuple:
    """Everything preserved by the null arm: bond tokens, junction tokens,
    per-bond junction counts."""
    counts = [0] * len(inst.bond_tokens)
    for a, b in inst.incidence:
        counts[a] += 1
        counts[b] += 1
    return (tuple(sorted(inst.bond_tokens)), tuple(sorted(inst.junc_tokens)), tuple(sorted(counts)))


def batch_object_instances(instances: Sequence[ObjectInstance]):
    """Build the BJOC batch dict for a list of object instances."""
    bond_feats, bond_batch, bond_index = [], [], []
    junc_feats, junc_batch, junc_index, junc_bond = [], [], [], []
    b_off = 0
    for g, inst in enumerate(instances):
        n_b = len(inst.bond_tokens)
        n_j = len(inst.junc_tokens)
        bf = torch.zeros((n_b, sro.BOND_FEAT_DIM), dtype=torch.float32)
        for b, t in enumerate(inst.bond_tokens):
            bf[b, t % sro.BOND_CATEGORIES] = 1.0
        bond_feats.append(bf)
        bond_batch.append(torch.full((n_b,), g, dtype=torch.long))
        bond_index.append(torch.zeros((n_b, 2), dtype=torch.long))
        jf = torch.zeros((n_j, sro.JUNC_FEAT_DIM), dtype=torch.float32)
        for j, t in enumerate(inst.junc_tokens):
            jf[j, 6 + (t % sro.RING_SIZE_SLOTS)] = 1.0  # dedicated token slot
        junc_feats.append(jf)
        junc_batch.append(torch.full((n_j,), g, dtype=torch.long))
        junc_index.append(torch.zeros((n_j, 3), dtype=torch.long))
        junc_bond.append(torch.as_tensor(inst.incidence, dtype=torch.long) + b_off)
        b_off += n_b
    batch = {
        "n_graphs": len(instances),
        "atom_feat": torch.zeros((len(instances), sro.ATOM_FEAT_DIM), dtype=torch.float32),
        "atom_batch": torch.arange(len(instances), dtype=torch.long),
        "bond_feat": torch.cat(bond_feats, 0),
        "bond_batch": torch.cat(bond_batch, 0),
        "bond_index": torch.cat(bond_index, 0),
        "junc_feat": torch.cat(junc_feats, 0),
        "junc_batch": torch.cat(junc_batch, 0),
        "junc_index": torch.cat(junc_index, 0),
        "junc_bond": torch.cat(junc_bond, 0),
        "global_feat": torch.zeros((len(instances), sro.GLOBAL_DIM), dtype=torch.float32),
    }
    batch["target_incidence"] = torch.tensor([i.target_incidence for i in instances], dtype=torch.float32)
    batch["target_marginal"] = torch.tensor([i.target_marginal for i in instances], dtype=torch.float32)
    return batch


# -- molecular-level synthetic instances (realistic small graphs) -----------


def random_synthetic_instance(
    rng: np.random.Generator,
    n_atoms: int,
    n_atom_types: int = 3,
    n_bond_types: int = 2,
) -> SyntheticInstance:
    """Random connected graph (tree + extra edges) as a tiny 'molecule'."""
    atom_types = [int(rng.integers(0, n_atom_types)) for _ in range(n_atoms)]
    edges: list[tuple[int, int, int]] = []
    for v in range(1, n_atoms):
        u = int(rng.integers(0, v))
        edges.append((u, v, int(rng.integers(0, n_bond_types))))
    n_extra = int(rng.integers(0, max(n_atoms // 3, 1) + 1))
    existing = {(u, v) for u, v, _ in edges}
    for _ in range(n_extra):
        u, v = rng.integers(0, n_atoms, size=2)
        u, v = int(min(u, v)), int(max(u, v))
        if u == v or (u, v) in existing:
            continue
        existing.add((u, v))
        edges.append((u, v, int(rng.integers(0, n_bond_types))))
    edges.sort()
    return _instance_from_edges(atom_types, edges)


def _instance_from_edges(atom_types: list[int], edges: list[tuple[int, int, int]]) -> SyntheticInstance:
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in edges:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    edge_index = np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.asarray(ea, dtype=np.int64)
    obj = sro.build_mol_objects(-1, x, edge_index, edge_attr)
    bt_sorted = [bt for _, _, bt in edges]
    inc, mar = synthetic_targets(obj, atom_types, bt_sorted)
    return SyntheticInstance(atom_types=atom_types, edges=edges, target_incidence=inc, target_marginal=mar)


def synthetic_instance_to_batch(instances: Sequence[SyntheticInstance]):
    """Convert synthetic raw molecular graphs into the BJOC batch dict."""
    objs = []
    for inst in instances:
        n = len(inst.atom_types)
        x = np.asarray(inst.atom_types, dtype=np.int64).reshape(n, 1)
        ei, ea = [], []
        for u, v, bt in inst.edges:
            ei.extend([(u, v), (v, u)])
            ea.extend([bt, bt])
        edge_index = np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
        objs.append(sro.build_mol_objects(-1, x, edge_index, np.asarray(ea, dtype=np.int64)))
    batch = sro.batch_objects(objs)
    batch["target_incidence"] = torch.tensor([i.target_incidence for i in instances], dtype=torch.float32)
    batch["target_marginal"] = torch.tensor([i.target_marginal for i in instances], dtype=torch.float32)
    return batch


def reverse_instance(inst: SyntheticInstance) -> SyntheticInstance:
    """Rebuild the graph with reversed labels (marginal-preserving relabel)."""
    n = len(inst.atom_types)
    atoms = list(reversed([int(t) for t in inst.atom_types]))
    edges = sorted((min(n - 1 - v, n - 1 - u), max(n - 1 - v, n - 1 - u), bt) for u, v, bt in inst.edges)
    return _instance_from_edges(atoms, edges)


def marginal_signature(inst: SyntheticInstance) -> tuple:
    """Permutation-invariant signature of all bond/junction marginals."""
    types = list(inst.atom_types)
    bond_sig = sorted((types[u], types[v], int(bt)) for u, v, bt in inst.edges)
    nbrs: dict[int, list[tuple[int, int]]] = {a: [] for a in range(len(types))}
    for u, v, bt in inst.edges:
        nbrs[u].append((v, int(bt)))
        nbrs[v].append((u, int(bt)))
    junc_sig = []
    n_b = [0] * len(inst.edges)
    edge_id = {(u, v): k for k, (u, v, _) in enumerate(inst.edges)}
    for j in range(len(types)):
        ordered = sorted(nbrs[j])
        for a in range(len(ordered)):
            for b in range(a + 1, len(ordered)):
                i, t1 = ordered[a]
                k, t2 = ordered[b]
                junc_sig.append(tuple(sorted((types[i], types[k]))) + (types[j],) + tuple(sorted((t1, t2))))
                n_b[edge_id[(min(i, j), max(i, j))]] += 1
                n_b[edge_id[(min(j, k), max(j, k))]] += 1
    return (tuple(bond_sig), tuple(sorted(junc_sig)), tuple(sorted(n_b)))
