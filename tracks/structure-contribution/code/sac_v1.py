"""BJOC-SAC v1 — structurally constrained second-order interaction tensors.

Protocol ``cscl-bjoc-sac-v1`` (``results/sac_v1/PROTOCOL.md``).

The atom / bond / junction encoders, the masked SSL objective and the Null
junction composer of BJOC-P3 are reused unchanged.  What is new is the
**graph-level composition function** and the reader input:

    q_s(e) = tanh(W_s c_e)                       s = 1,2,3 ; c_e in R^96
    R1 = {e,f : dist_line(e,f) = 1}
    R2 = {e,f : dist_line(e,f) = 2}
    R3 = {e,f : same ring-bearing biconnected block, dist_line(e,f) >= 3}
    T_s = sum_{(e,f) in R_s} (q_s(e) * q_s(f)) / sqrt(max(1, |R_s|))
    reader_in = [P3_original_reader_input, T_1, T_2, T_3,
                 log1p(|R_1|), log1p(|R_2|), log1p(|R_3|)]

``T_s`` is a graph-level sum over an explicit, structure-fixed pair list.  The
per-bond embedding ``c_e`` is **never** refreshed: there is no
``h_e <- aggregate(neighbours)`` update, no attention and no iteration.  The
only backward-path coupling between bonds is the gradient of the graph-level
sum, which the forward computation never reads back into per-bond state.

Arms
----
``P3``    the frozen BJOC-Null model (413,505 parameters, no relation tensors).
``SACR``  the SAC trunk with the **real** R1/R2/R3 pair lists.
``SACU``  the identical trunk, parameters, initialisation and budget, but each
          relation is replaced by the content-attribution null

              T_s^U = [m_s / C(n,2)] * sum_{e<f} q_s(e) q_s(f) / sqrt(max(1,m_s))

          with the molecule's true ``m_s`` relation count.  ``SACU`` keeps the
          total relation weight ``m_s`` and the object/content multiset but
          destroys the bond-pair <-> structure correspondence.  It is *not* a
          row-sum/degree-matched null and *not* an equal-FLOP control.

Everything here is label-free except :func:`train_supervised_sac`, which reads
Fit ``y`` only.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402
import p3_bond_relation_bridge_v1 as pb  # noqa: E402  (audited line graph)

PROTOCOL_ID = "cscl-bjoc-sac-v1"
SPLIT_TAG = "S"
SPLIT_SALT = "cscl-bjoc-sac-v1"

SEEDS = (0, 1)
ARMS = ("P3", "SACR", "SACU")
RELATION_ARMS = ("SACR", "SACU")
RELATION_MODE_OF_ARM = {"SACR": "R", "SACU": "U"}

REL_NAMES = ("R1", "R2", "R3")
N_RELATIONS = len(REL_NAMES)
PROJECT_DIM = 24
PROJECT_INIT_OFFSET = 104729
READER_INIT_OFFSET = 130363
PROJECT_OUT = PROJECT_DIM * N_RELATIONS

BASE_PARAMETERS = 413_505
P3_READER_PARAMETERS = 148_481
EXPECTED_SAC_PARAMETERS = 439_689
PARAM_CAP = 500_000

ZONES = ("monitor", "locked")
PRIMARY_ZONE = "locked"
REPLICATION_ZONE = "monitor"

MEANINGFUL_DELTA_R_VS_P3 = 0.004
MEANINGFUL_DELTA_R_VS_U = 0.003
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
BOOTSTRAP_N = 5_000
BOOTSTRAP_SEED = 0
PARTICIPATION_MEAN_ABS = 0.0025
PARTICIPATION_MAX_ABS = 0.02

INF_DIST = np.int64(1 << 20)

CONTRASTS = (
    ("SACR_minus_P3", "SACR", "P3"),
    ("SACR_minus_SACU", "SACR", "SACU"),
    ("SACU_minus_P3", "SACU", "P3"),
)


class SACError(RuntimeError):
    """Hard protocol violation (raises instead of silently continuing)."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise SACError(message)


# ---------------------------------------------------------------------------
# relation algebra on the physical-bond line graph
# ---------------------------------------------------------------------------


def line_distances(adj: np.ndarray) -> np.ndarray:
    """Unweighted shortest-path matrix on the line graph (INF = unreachable)."""
    n = int(adj.shape[0])
    dist = np.full((n, n), INF_DIST, dtype=np.int64)
    for s in range(n):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            row = dist[s]
            for u in frontier:
                for v in np.nonzero(adj[u])[0]:
                    v = int(v)
                    if row[v] == INF_DIST:
                        row[v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


def bond_ring_blocks(obj: sro.MolObjects) -> tuple[np.ndarray, int, int]:
    """Per-bond ring-bearing biconnected block id (-1 if none).

    A block is *ring-bearing* iff it has >= 3 edges.  Tarjan blocks partition
    the edge set, so every bond gets at most one block id.  Bridges sit in
    their own 1-edge block and are therefore never ring-bearing.
    """
    bonds = [(int(u), int(v), int(t)) for (u, v), t in zip(np.asarray(obj.bond_index), np.asarray(obj.bond_type))]
    n_bonds = len(bonds)
    block_of = np.full(n_bonds, -1, dtype=np.int64)
    key = {(min(u, v), max(u, v)): k for k, (u, v, _) in enumerate(bonds)}
    blocks = sro._biconnected_blocks(int(obj.n_atoms), bonds) if bonds else []
    n_ring_blocks = 0
    n_bridges = 0
    for _nodes, edges in blocks:
        if len(edges) >= 3:
            for u, v, _t in edges:
                block_of[key[(min(u, v), max(u, v))]] = n_ring_blocks
            n_ring_blocks += 1
        else:
            n_bridges += 1
    return block_of, n_ring_blocks, n_bridges


def relation_pair_lists(obj: sro.MolObjects) -> dict[str, dict[str, np.ndarray]]:
    """Disjoint unordered physical-bond pair lists for R1 / R2 / R3."""
    adj = pb.line_adjacency(obj)
    n = int(adj.shape[0])
    block_of, _n_blocks, _n_bridges = bond_ring_blocks(obj)
    rows_all, cols_all = np.triu_indices(n, 1)
    if rows_all.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return {name: {"src": empty, "dst": empty.copy()} for name in REL_NAMES}
    dist = line_distances(adj)
    d = dist[rows_all, cols_all]
    same_ring_block = (block_of[rows_all] == block_of[cols_all]) & (block_of[rows_all] >= 0)
    masks = {
        "R1": d == 1,
        "R2": d == 2,
        "R3": (d >= 3) & same_ring_block,
    }
    total = masks["R1"].astype(np.int64) + masks["R2"].astype(np.int64) + masks["R3"].astype(np.int64)
    require(int(total.max()) <= 1 if total.size else True, "relation sets overlap")
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in REL_NAMES:
        keep = masks[name]
        out[name] = {
            "src": rows_all[keep].astype(np.int32),
            "dst": cols_all[keep].astype(np.int32),
        }
    return out


def complete_pair_list(n_bonds: int) -> dict[str, np.ndarray]:
    """All unordered physical-bond pairs ``e < f`` (the SACU support)."""
    rows, cols = np.triu_indices(int(n_bonds), 1)
    return {"src": rows.astype(np.int32), "dst": cols.astype(np.int32)}


def choose2(n: int) -> int:
    n = int(n)
    return n * (n - 1) // 2 if n >= 2 else 0


def relation_counts(obj: sro.MolObjects) -> np.ndarray:
    pairs = relation_pair_lists(obj)
    return np.asarray([len(pairs[name]["src"]) for name in REL_NAMES], dtype=np.int64)


@dataclass
class MolRelations:
    """Relation pair lists and counts for one molecule."""

    mol_index: int
    n_bonds: int
    m: np.ndarray                       # [3] |R_s|
    pairs: dict[str, dict[str, np.ndarray]]
    all_pairs: dict[str, np.ndarray]


def build_mol_relations(obj: sro.MolObjects) -> MolRelations:
    pairs = relation_pair_lists(obj)
    m = np.asarray([len(pairs[name]["src"]) for name in REL_NAMES], dtype=np.int64)
    n_bonds = int(np.asarray(obj.bond_index).reshape(-1, 2).shape[0])
    return MolRelations(
        mol_index=int(obj.mol_index),
        n_bonds=n_bonds,
        m=m,
        pairs=pairs,
        all_pairs=complete_pair_list(n_bonds),
    )


# ---------------------------------------------------------------------------
# zone-wide relation store + batching
# ---------------------------------------------------------------------------


class ZoneRelations:
    """Concatenated relation pair lists for a whole zone, with graph offsets."""

    def __init__(self, mols: Sequence[MolRelations]) -> None:
        n = len(mols)
        self.n_bonds = np.asarray([m.n_bonds for m in mols], dtype=np.int64)
        self.m = np.stack([m.m for m in mols], 0) if n else np.zeros((0, N_RELATIONS), dtype=np.int64)
        # per-relation segment pointers
        self.starts = np.zeros((n, N_RELATIONS), dtype=np.int64)
        self.ends = np.zeros((n, N_RELATIONS), dtype=np.int64)
        self.src: list[np.ndarray] = []
        self.dst: list[np.ndarray] = []
        for s, name in enumerate(REL_NAMES):
            cursor = 0
            src_parts: list[np.ndarray] = []
            dst_parts: list[np.ndarray] = []
            for i, mol in enumerate(mols):
                p = mol.pairs[name]
                self.starts[i, s] = cursor
                src_parts.append(np.asarray(p["src"], dtype=np.int32))
                dst_parts.append(np.asarray(p["dst"], dtype=np.int32))
                cursor += int(src_parts[-1].shape[0])
                self.ends[i, s] = cursor
            self.src.append(np.concatenate(src_parts) if src_parts else np.zeros(0, dtype=np.int32))
            self.dst.append(np.concatenate(dst_parts) if dst_parts else np.zeros(0, dtype=np.int32))
        # dense e<f support (SACU) and its per-molecule pointers
        self.astart = np.zeros(n, dtype=np.int64)
        self.aend = np.zeros(n, dtype=np.int64)
        a_src: list[np.ndarray] = []
        a_dst: list[np.ndarray] = []
        cursor = 0
        for i, mol in enumerate(mols):
            self.astart[i] = cursor
            a_src.append(np.asarray(mol.all_pairs["src"], dtype=np.int32))
            a_dst.append(np.asarray(mol.all_pairs["dst"], dtype=np.int32))
            cursor += int(a_src[-1].shape[0])
            self.aend[i] = cursor
        self.asrc = np.concatenate(a_src) if a_src else np.zeros(0, dtype=np.int32)
        self.adst = np.concatenate(a_dst) if a_dst else np.zeros(0, dtype=np.int32)

    # -- auditing ----------------------------------------------------------
    def __len__(self) -> int:
        return int(self.n_bonds.shape[0])

    def totals(self) -> dict[str, int]:
        return {name: int(self.src[s].shape[0]) for s, name in enumerate(REL_NAMES)}

    def n_complete(self) -> int:
        return int(self.asrc.shape[0])

    # -- serialisation -----------------------------------------------------
    def to_state(self) -> dict[str, Any]:
        return {
            "n_bonds": self.n_bonds,
            "m": self.m,
            "starts": self.starts,
            "ends": self.ends,
            "src": {name: self.src[s] for s, name in enumerate(REL_NAMES)},
            "dst": {name: self.dst[s] for s, name in enumerate(REL_NAMES)},
            "astart": self.astart,
            "aend": self.aend,
            "asrc": self.asrc,
            "adst": self.adst,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ZoneRelations":
        obj = cls.__new__(cls)
        obj.n_bonds = np.asarray(state["n_bonds"], dtype=np.int64)
        obj.m = np.asarray(state["m"], dtype=np.int64)
        obj.starts = np.asarray(state["starts"], dtype=np.int64)
        obj.ends = np.asarray(state["ends"], dtype=np.int64)
        obj.src = [np.asarray(state["src"][name], dtype=np.int32) for name in REL_NAMES]
        obj.dst = [np.asarray(state["dst"][name], dtype=np.int32) for name in REL_NAMES]
        obj.astart = np.asarray(state["astart"], dtype=np.int64)
        obj.aend = np.asarray(state["aend"], dtype=np.int64)
        obj.asrc = np.asarray(state["asrc"], dtype=np.int32)
        obj.adst = np.asarray(state["adst"], dtype=np.int32)
        return obj

    # -- batching ----------------------------------------------------------
    def batch(
        self,
        indices: Sequence[int],
        mode: str,
        device: torch.device | str,
        *,
        zero_tensors: bool = False,
    ) -> dict[str, Any]:
        """Batch tensors for ``mode`` in {"R", "U"}.

        ``mode="R"`` uses the real relation pair lists with unit weight;
        ``mode="U"`` uses the dense ``e < f`` support with the per-molecule
        weight ``m_s / C(n,2)``.  Both are then divided by
        ``sqrt(max(1, m_s))`` per graph, so the only difference between the two
        arms is the pair list and the per-pair weight -- the computation graph
        is the same.
        """
        require(mode in ("R", "U"), f"unknown relation mode {mode!r}")
        idx = np.asarray(list(indices), dtype=np.int64)
        n_graphs = int(idx.shape[0])
        dev = torch.device(device)
        src_all: list[torch.Tensor] = []
        dst_all: list[torch.Tensor] = []
        graph_all: list[torch.Tensor] = []
        weight_all: list[torch.Tensor] = []
        m_sel = self.m[idx] if n_graphs else np.zeros((0, N_RELATIONS), dtype=np.int64)
        n_sel = self.n_bonds[idx] if n_graphs else np.zeros(0, dtype=np.int64)
        choose = np.asarray([choose2(v) for v in n_sel], dtype=np.int64)
        for s in range(N_RELATIONS):
            if mode == "R":
                seg_start = self.starts[idx, s]
                seg_end = self.ends[idx, s]
                src_ref, dst_ref = self.src[s], self.dst[s]
            else:
                seg_start = self.astart[idx]
                seg_end = self.aend[idx]
                src_ref, dst_ref = self.asrc, self.adst
            lens = seg_end - seg_start
            total = int(lens.sum())
            if total == 0:
                src_all.append(torch.zeros(0, dtype=torch.long, device=dev))
                dst_all.append(torch.zeros(0, dtype=torch.long, device=dev))
                graph_all.append(torch.zeros(0, dtype=torch.long, device=dev))
                weight_all.append(torch.zeros(0, dtype=torch.float32, device=dev))
                continue
            rep = np.repeat(np.arange(n_graphs, dtype=np.int64), lens)
            first = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
            pos = seg_start[rep] + (np.arange(total, dtype=np.int64) - first[rep])
            bond_off = np.repeat(np.cumsum(n_sel) - n_sel, lens)
            src_t = torch.from_numpy(src_ref[pos].astype(np.int64) + bond_off).to(dev)
            dst_t = torch.from_numpy(dst_ref[pos].astype(np.int64) + bond_off).to(dev)
            graph_t = torch.from_numpy(rep).to(dev)
            if mode == "R":
                w = np.ones(total, dtype=np.float32)
            else:
                denom = np.maximum(choose[rep], 1)
                w = np.where(choose[rep] > 0, m_sel[rep, s].astype(np.float32) / denom, 0.0).astype(np.float32)
            weight_all.append(torch.from_numpy(w).to(dev))
            src_all.append(src_t)
            dst_all.append(dst_t)
            graph_all.append(graph_t)
        m_t = torch.from_numpy(m_sel.astype(np.float32)).to(dev)
        scale = 1.0 / torch.sqrt(torch.clamp(m_t, min=1.0))
        valid = (torch.from_numpy(n_sel.astype(np.float32)).to(dev).unsqueeze(1) >= 2.0) & (m_t > 0)
        scale = scale * valid.to(scale.dtype)
        return {
            "mode": mode,
            "n_graphs": n_graphs,
            "src": src_all,
            "dst": dst_all,
            "graph": graph_all,
            "weight": weight_all,
            "scale": scale,
            "counts": torch.log1p(m_t),
            "m": m_sel,
            "n_bonds": n_sel,
            "zero_tensors": bool(zero_tensors),
        }


def build_zone_relations(mols: Sequence[MolRelations]) -> ZoneRelations:
    return ZoneRelations(mols)


# ---------------------------------------------------------------------------
# relation audit (label-free)
# ---------------------------------------------------------------------------


def audit_molecule(obj: sro.MolObjects) -> dict[str, Any]:
    adj = pb.line_adjacency(obj)
    n = int(adj.shape[0])
    pairs = relation_pair_lists(obj)
    block_of, n_ring_blocks, n_bridges = bond_ring_blocks(obj)
    dist = line_distances(adj)
    upper = dist[np.triu_indices(n, 1)] if n >= 2 else np.zeros(0, dtype=np.int64)
    finite = upper[upper < INF_DIST]
    counts = {name: int(pairs[name]["src"].shape[0]) for name in REL_NAMES}
    total = sum(counts.values())
    n_pairs = choose2(n)
    # disjointness / support sanity
    seen = np.zeros(n_pairs, dtype=np.int64)
    rows_all, cols_all = np.triu_indices(n, 1)
    index = {tuple(k): i for i, k in enumerate(zip(rows_all.tolist(), cols_all.tolist()))}
    for name in REL_NAMES:
        for a, b in zip(pairs[name]["src"].tolist(), pairs[name]["dst"].tolist()):
            seen[index[(a, b)]] += 1
    ring_bond_mask = block_of >= 0
    return {
        "n_atoms": int(obj.n_atoms),
        "n_bonds": n,
        "n_ring_blocks": int(n_ring_blocks),
        "n_bridge_blocks": int(n_bridges),
        "n_ring_bonds": int(ring_bond_mask.sum()),
        "ring_bond_fraction": float(ring_bond_mask.mean()) if n else 0.0,
        "counts": counts,
        "n_pairs_total": int(n_pairs),
        "relation_coverage": float(total / n_pairs) if n_pairs else 0.0,
        "max_pair_multiplicity": int(seen.max()) if seen.size else 0,
        "disjoint": bool((int(seen.max()) if seen.size else 0) <= 1),
        "max_line_distance": int(finite.max()) if finite.size else 0,
        "mean_line_distance": float(finite.mean()) if finite.size else 0.0,
        "unreachable_pairs": int((upper >= INF_DIST).sum()) if n >= 2 else 0,
        "counts_zero": [int(v == 0) for v in counts.values()],
    }


def summarize_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n_molecules": 0}
    out: dict[str, Any] = {"n_molecules": n}
    for key in ("n_bonds", "n_ring_blocks", "n_bridge_blocks", "ring_bond_fraction", "relation_coverage",
                "max_pair_multiplicity", "max_line_distance", "mean_line_distance", "unreachable_pairs",
                "n_pairs_total"):
        vals = np.asarray([float(r[key]) for r in records], dtype=np.float64)
        out[key] = {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max())}
    for s, name in enumerate(REL_NAMES):
        vals = np.asarray([float(r["counts"][name]) for r in records], dtype=np.float64)
        out[f"count_{name}"] = {
            "mean": float(vals.mean()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "frac_zero": float((vals == 0).mean()),
            "frac_ge1": float((vals >= 1).mean()),
        }
    out["all_disjoint"] = bool(all(bool(r["disjoint"]) for r in records))
    out["n_molecules_no_relations"] = int(sum(1 for r in records if sum(r["counts"].values()) == 0))
    out["n_molecules_with_ring_block"] = int(sum(1 for r in records if r["n_ring_blocks"] > 0))
    out["n_molecules_single_bond"] = int(sum(1 for r in records if r["n_bonds"] <= 1))
    out["n_molecules_two_bonds"] = int(sum(1 for r in records if r["n_bonds"] == 2))
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def graph_tensors(
    q: Sequence[torch.Tensor],
    rel: Mapping[str, Any],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """``T_s`` graph tensors (one [G, PROJECT_DIM] per relation)."""
    n_graphs = int(rel["n_graphs"])
    out: list[torch.Tensor] = []
    zero = bool(rel.get("zero_tensors", False))
    for s in range(N_RELATIONS):
        accum = torch.zeros((n_graphs, PROJECT_DIM), device=q[s].device, dtype=q[s].dtype)
        src, dst, graph = rel["src"][s], rel["dst"][s], rel["graph"][s]
        if src.numel():
            msg = q[s][src] * q[s][dst] * rel["weight"][s].unsqueeze(1)
            accum = accum.index_add(0, graph, msg)
        accum = accum * rel["scale"][:, s].unsqueeze(1)
        if zero:
            accum = torch.zeros_like(accum)
        out.append(accum)
    counts = torch.log1p(torch.as_tensor(rel["m"], device=q[0].device, dtype=q[0].dtype))
    return out, counts


class SAC(nn.Module):
    """BJOC content trunk + structurally constrained second-order tensors."""

    def __init__(self, base: src.BJOC, seed: int) -> None:
        super().__init__()
        self.base = base
        self.seed = int(seed)
        cfg = base.config
        self.config = cfg
        self.project_dim = PROJECT_DIM
        self.project = nn.ModuleList([nn.Linear(int(cfg.compose_out), PROJECT_DIM) for _ in range(N_RELATIONS)])
        self.reader = src._MLP(int(cfg.reader_in) + PROJECT_OUT + N_RELATIONS, cfg.reader_hidden, 1, cfg.reader_layers, cfg.dropout)
        self._init_parameters()
        # the P3 reader is replaced, not kept: no dead/unused parameters
        self.base.reader = nn.Identity()

    # -- initialisation ----------------------------------------------------
    def _init_parameters(self) -> None:
        gen_p = torch.Generator(device="cpu").manual_seed(self.seed + PROJECT_INIT_OFFSET)
        gen_r = torch.Generator(device="cpu").manual_seed(self.seed + READER_INIT_OFFSET)
        with torch.no_grad():
            for lin in self.project:
                bound = 1.0 / math.sqrt(float(lin.in_features))
                lin.weight.copy_(torch.empty(lin.weight.shape).uniform_(-bound, bound, generator=gen_p))
                lin.bias.copy_(torch.empty(lin.bias.shape).uniform_(-bound, bound, generator=gen_p))
            for i, lin in enumerate(self.reader.lins):
                bound = 1.0 / math.sqrt(float(lin.in_features))
                lin.weight.copy_(torch.empty(lin.weight.shape).uniform_(-bound, bound, generator=gen_r))
                lin.bias.copy_(torch.empty(lin.bias.shape).uniform_(-bound, bound, generator=gen_r))
            # preserve the P3 content path: copy the original reader exactly
            old_first = self.base.reader.lins[0]
            new_first = self.reader.lins[0]
            old_dim = int(old_first.in_features)
            new_first.weight[:, :old_dim].copy_(old_first.weight.detach())
            new_first.bias.copy_(old_first.bias.detach())
            for i in range(1, len(self.base.reader.lins)):
                self.reader.lins[i].load_state_dict(self.base.reader.lins[i].state_dict())

    # -- parameter audit ---------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    @property
    def new_parameters(self) -> int:
        proj = int(sum(p.numel() for p in self.project.parameters()))
        reader = int(sum(p.numel() for p in self.reader.parameters()))
        return proj + reader - P3_READER_PARAMETERS

    # -- forward -----------------------------------------------------------
    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        rel: Mapping[str, Any],
    ) -> dict[str, Any]:
        feats = self.base.forward_features(batch)
        c = feats["c"]
        q = [torch.tanh(lin(c)) for lin in self.project]
        tensors, counts = graph_tensors(q, rel)
        reader_in = torch.cat([feats["reader_in"], *tensors, counts], dim=1)
        return {**feats, "q": q, "T": tensors, "sac_counts": counts, "reader_in": reader_in}

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        rel: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        require(rel is not None, "SAC requires relation tensors")
        feats = self.forward_features(batch, rel)
        return self.reader(feats["reader_in"]).view(-1)


def build_sac(seed: int, norm: Mapping[str, Any], device: torch.device | str) -> SAC:
    """Deterministic SAC init on top of the exact P3 init for ``seed``."""
    base = bp.build_bjoc_null(seed, norm, device)
    model = SAC(base, seed)
    model.to(device)
    require(model.n_parameters == EXPECTED_SAC_PARAMETERS, f"SAC parameter audit failed: {model.n_parameters}")
    require(model.n_parameters <= PARAM_CAP, "SAC exceeds the 500k parameter cap")
    return model


def build_p3(seed: int, norm: Mapping[str, Any], device: torch.device | str) -> src.BJOC:
    model = bp.build_bjoc_null(seed, norm, device)
    require(int(sum(p.numel() for p in model.parameters())) == BASE_PARAMETERS, "P3 parameter audit failed")
    return model


def build_arm(arm: str, seed: int, norm: Mapping[str, Any], device: torch.device | str) -> nn.Module:
    require(arm in ARMS, f"unknown arm {arm!r}")
    return build_p3(seed, norm, device) if arm == "P3" else build_sac(seed, norm, device)


def full_state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def shared_state_hash(model: nn.Module) -> str:
    """Hash of the shared BJOC content trunk (encoders + composer + buffers)."""
    base = model if isinstance(model, src.BJOC) else model.base
    parts = {f"atom:{k}": v for k, v in base.atom_encoder.state_dict().items()}
    parts |= {f"bond:{k}": v for k, v in base.bond_encoder.state_dict().items()}
    parts |= {f"junc:{k}": v for k, v in base.junction_encoder.state_dict().items()}
    parts |= {f"compose:{k}": v for k, v in base.group_composer.state_dict().items()}
    return bp.state_group_sha256(parts)


def relation_tensor_state(model: SAC, batch, rel, device) -> np.ndarray:
    """Label-free ``T_1..T_3`` for auditing / sensitivity checks."""
    model.eval()
    with torch.no_grad():
        feats = model.base.forward_features(batch)
        q = [torch.tanh(lin(feats["c"])) for lin in model.project]
        tensors, _counts = graph_tensors(q, rel)
    return np.concatenate([t.detach().cpu().numpy() for t in tensors], axis=1)


def model_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict_model(
    model: nn.Module,
    objs: Sequence[sro.MolObjects],
    relations: ZoneRelations | None,
    mode: str | None,
    device: torch.device | str,
    chunk: int = 256,
    *,
    zero_tensors: bool = False,
) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    out = []
    for i in range(0, len(objs), chunk):
        part = list(objs[i : i + chunk])
        batch = move_batch(sro.batch_objects(part), dev)
        if relations is None:
            pred = model(batch)
        else:
            rel = relations.batch(list(range(i, i + len(part))), mode, dev, zero_tensors=zero_tensors)
            pred = model(batch, rel)
        out.append(pred.detach().cpu().numpy())
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------
# training loop (bitwise-mirror of bp.train_supervised / pb.train_supervised_bridge)
# ---------------------------------------------------------------------------


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0].keys()}
    return members, soup


def _module_grad_norm(module: nn.Module) -> float:
    norms = [float(p.grad.norm()) for p in module.parameters() if p.grad is not None]
    return float(np.sqrt(np.sum(np.square(norms)))) if norms else 0.0


@torch.no_grad()
def _probe_targets(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    base = model if isinstance(model, src.BJOC) else model.base
    groups = {
        "atom_encoder": list(base.atom_encoder.parameters()),
        "bond_encoder": list(base.bond_encoder.parameters()),
        "junction_encoder": list(base.junction_encoder.parameters()),
        "group_composer": list(base.group_composer.parameters()),
    }
    if isinstance(model, SAC):
        groups["project"] = list(model.project.parameters())
        groups["reader"] = list(model.reader.parameters())
    else:
        groups["reader"] = list(model.reader.parameters())
    return groups


def gradient_probe(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    rel: Mapping[str, Any] | None,
    y: torch.Tensor,
) -> dict[str, Any]:
    """One supervised step's gradient norms per module (no parameter update)."""
    model.train()
    groups = _probe_targets(model)
    captured: dict[str, torch.Tensor] = {}
    first_linear = model.reader.lins[0]
    handle = first_linear.register_forward_pre_hook(lambda _m, inp: captured.setdefault("x", inp[0]))
    try:
        pred = model(batch) if rel is None else model(batch, rel)
        loss = F.l1_loss(pred.view(-1), y.view(-1))
        flat_params = [p for g in groups.values() for p in g]
        targets = [captured["x"], *flat_params]
        grads = torch.autograd.grad(loss, targets, allow_unused=True, retain_graph=False)
        out: dict[str, Any] = {"loss": float(loss)}
        g_in = grads[0]
        out["reader_input_grad_norm"] = float(g_in.norm()) if g_in is not None else 0.0
        cursor = 1
        for name, params in groups.items():
            total = 0.0
            for _p in params:
                g = grads[cursor]
                cursor += 1
                if g is not None:
                    total += float((g * g).sum())
            out[f"grad_norm_{name}"] = float(math.sqrt(total))
        if isinstance(model, SAC):
            reader_in = captured["x"]
            old = int(model.base.config.reader_in)
            if g_in is not None and reader_in.shape[1] == old + PROJECT_OUT + N_RELATIONS:
                for s, name in enumerate(REL_NAMES):
                    blk = g_in[:, old + s * PROJECT_DIM : old + (s + 1) * PROJECT_DIM]
                    out[f"grad_norm_T_{name}"] = float(blk.norm())
                out["grad_norm_counts_block"] = float(g_in[:, old + PROJECT_OUT :].norm())
                out["grad_norm_content_block"] = float(g_in[:, :old].norm())
            with torch.no_grad():
                feats_q = model.base.forward_features(batch)
                tensors, _ = graph_tensors(
                    [torch.tanh(lin(feats_q["c"])) for lin in model.project], rel  # type: ignore[arg-type]
                )
                out["T_abs_mean"] = [float(t.abs().mean()) for t in tensors]
        return out
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def train_supervised_sac(
    *,
    arm: str,
    seed: int,
    model: nn.Module,
    relations: ZoneRelations | None,
    train_objs: Sequence[sro.MolObjects],
    y_fit: np.ndarray,
    device: torch.device,
    recipe_key: str = "R3",
    epochs: int | None = None,
    log_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The frozen R3 recipe, identical for all three arms.

    Mechanically identical to ``bjoc_perfmech_long_v1.train_supervised`` /
    ``p3_bond_relation_bridge_v1.train_supervised_bridge``: same optimiser,
    schedule, shuffle generator, clipping, soup rule and progress logging.  A
    CPU gate verifies bitwise soup equality with the frozen trainer for P3.
    """
    require(recipe_key in bp.RECIPES, f"unknown recipe {recipe_key!r}")
    recipe = bp.RECIPES[recipe_key]
    mode = RELATION_MODE_OF_ARM.get(arm)
    require((arm == "P3") == (relations is None), "P3 takes no relation store and SAC arms require one")
    n = int(len(train_objs))
    require(n == int(len(y_fit)), "train objects and labels differ in length")
    steps_per_epoch = math.ceil(n / bp.BATCH_SIZE)
    total_updates = int(epochs or recipe["epochs"]) * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = None
    if recipe["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"]))
    elif recipe["schedule"] != "constant":
        raise SACError(f"unknown schedule {recipe['schedule']!r}")

    gen = torch.Generator().manual_seed(int(seed) + bp.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=bp.SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    started = time.perf_counter()
    total_epochs = int(epochs or recipe["epochs"])

    for epoch in range(1, total_epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, bp.BATCH_SIZE, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = move_batch(sro.batch_objects([train_objs[i] for i in indices]), device)
            rel = relations.batch(indices, mode, device) if relations is not None else None  # type: ignore[arg-type]
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch) if rel is None else model(batch, rel)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), bp.GRAD_CLIP))
            if not math.isfinite(gn):
                raise SACError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > bp.GRAD_CLIP:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                expected_lr = bp.schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
                lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        train_mae = total / max(n_mol, 1)
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == total_epochs:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            raise SACError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")

    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), total_epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_model(model, train_objs, relations, mode, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    info: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "relation_mode": mode,
        "seed": int(seed),
        "recipe_key": recipe_key,
        "recipe": dict(recipe),
        "epochs": total_epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {bp.SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_state": soup_state,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "schedule": {
            "type": recipe["schedule"],
            "per": "step",
            "lr0": float(recipe["lr"]),
            "lr_min": None if recipe["lr_min"] is None else float(recipe["lr_min"]),
            "t_max_updates": total_updates,
            "lr_first_step": curve[0]["lr_first"],
            "lr_last_step": curve[-1]["lr_last"],
            "cosine_formula_max_abs_deviation": lr_deviation_max,
        },
        "anomalies": {"grad_clip_events": int(clip_events), "grad_norm_max": float(grad_norm_max)},
        "curve": curve,
        "wall_seconds": float(wall),
        "molecules_seen": int(n) * total_epochs,
        "device": str(device),
        "batch_order": {"full_hash": order_hash.hexdigest(), "full_matches": True},
        "parameter_audit": {"actual_parameters": model_parameters(model)},
    }
    if isinstance(model, SAC):
        scale = None
        info["sac"] = {
            "new_parameters": model.new_parameters,
            "project_parameters": int(sum(p.numel() for p in model.project.parameters())),
            "reader_parameters": int(sum(p.numel() for p in model.reader.parameters())),
            "replaced_p3_reader_parameters": int(P3_READER_PARAMETERS),
        }
        del scale
    return info


# ---------------------------------------------------------------------------
# frozen decision vocabulary
# ---------------------------------------------------------------------------


def contrast_label(
    delta_seed_mean: float,
    per_seed_delta: Mapping[str, float],
    ci_lo: float,
    ci_hi: float,
    ordinary_delta: float,
    meaningful: float,
) -> str:
    values = [float(v) for v in per_seed_delta.values()]
    if (
        delta_seed_mean <= -meaningful
        and all(v < 0.0 for v in values)
        and ci_hi < 0.0
        and ordinary_delta <= 0.0
    ):
        return "GAIN"
    if delta_seed_mean >= meaningful and all(v > 0.0 for v in values) and ci_lo > 0.0:
        return "LOSS"
    return "NEUTRAL"


def sac_case(
    *,
    label_r_p3: str,
    label_r_u: str,
    label_u_p3: str,
    replicated_r_p3: bool,
    replicated_r_u: bool,
    replicated_u_p3: bool,
    top10_r_p3: float,
    participation: bool,
) -> dict[str, Any]:
    """Frozen A/B/C/E/D rule (PROTOCOL.md §8).  Ordered, mutually exclusive."""
    if (
        label_r_p3 == "GAIN"
        and replicated_r_p3
        and label_r_u == "GAIN"
        and replicated_r_u
        and top10_r_p3 <= 0.0
        and participation
    ):
        return {
            "case": "CASE_A_STRUCTURAL_RELATION_GAIN",
            "reasons": [
                "SACR beats P3 and SACU beyond the frozen practical scales, both replicated on Monitor",
                "top-10-removed not worse",
                "the trained SACR model depends on its relation tensors",
            ],
        }
    if label_r_u == "GAIN" and replicated_r_u and label_r_p3 != "GAIN":
        return {
            "case": "CASE_B_RELATION_USED_NO_HEADLINE_GAIN",
            "reasons": ["SACR beats SACU but not P3 beyond the frozen scale"],
        }
    if label_r_p3 == "GAIN" and replicated_r_p3 and label_u_p3 == "GAIN" and replicated_u_p3 and label_r_u != "GAIN":
        return {
            "case": "CASE_C_GENERIC_SECOND_ORDER",
            "reasons": ["both SACR and SACU beat P3, but SACR does not beat SACU: a generic second-order gain"],
        }
    if label_r_p3 == "GAIN":
        reasons = []
        if not replicated_r_p3:
            reasons.append("SACR-P3 gain did not replicate on Monitor")
        if label_r_u != "GAIN":
            reasons.append("SACR-SACU is not GAIN")
        if label_u_p3 != "GAIN":
            reasons.append("SACU-P3 is not GAIN")
        if top10_r_p3 > 0.0:
            reasons.append("top-10-removed worsened")
        if not participation:
            reasons.append("the trained model does not depend on the relation tensors")
        return {"case": "CASE_E_R_GAIN_NOT_ATTRIBUTED", "reasons": reasons or ["residual"]}
    return {
        "case": "CASE_D_NO_GAIN",
        "reasons": ["SACR does not improve P3 beyond the frozen practical scale"],
    }
