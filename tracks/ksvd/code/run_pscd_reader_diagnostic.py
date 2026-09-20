#!/usr/bin/env python
"""PSCD-R-v0 — Frozen-Dictionary Motif Readability Audit.

Question
--------
PSCD-v0 established that the port-structured compositional code ``G <-> (D, C_G)``
is lossless / permutation-invariant / compressive / reusable / port-clean, yet a
matched tiny reader read the composition graph at valid MAE 0.5429 against a raw
atom graph at 0.3113.  This round freezes the whole PSCD-v0 dictionary and tests
**why** the lossless composition code is hard for a tiny reader, distinguishing:

* **H1 (opaque-atom)** — an independent per-motif ID embedding breaks
  atom/bond-level parameter sharing.
* **H2 (port-bottleneck)** — the composition reader collapses all attachment
  ports of an occurrence into one state too early.

Only the reader's representation of the dictionary atom changes.  Nothing about
the dictionary / merge sequence / partitions / canonical motifs / ports /
composition graph is relearned.

Readers (all hidden width 32, outer depth 2, seed 0, identical optimizer / MAE
loss / batch regime / tiny final head):
* ``R0``    opaque-ID composition reader (motif-ID lookup), **batching-corrected**
* ``R0ship`` the PSCD-v0 as-shipped composition reader (historical reproduction)
* ``R1``    structured-static motif reader (shared intra-motif graph encoder)
* ``R2``    port-resolved structured reader (per-active-port dynamic state)
* ``R3``    matched raw atom reader

Pre-run defect note
-------------------
Inspection found that the PSCD-v0 as-shipped composition collate lays nodes out
interleaved per graph (``[occ_g0, conn_g0, occ_g1, ...]``) while the reader
forward assumes ``h[:M] = E_M(occ_ids)`` for the *whole batch* and
``h[M:] = E_B(conn_ids)``.  The two layouts disagree for any multi-graph batch,
so the shipped composition reader is **not batch-invariant** (verified: batch
predictions differ from single-graph predictions by >1.3 MAE).  ``R0ship``
reproduces that behaviour unchanged; ``R0`` keeps the same module and only fixes
the collate node order (all occ nodes first, then all connection nodes, edges
remapped).  ``bag`` and ``raw`` readers are batch-invariant.

Discipline
----------
* Only official ZINC ``train`` (10 000) / ``valid`` (1 000) are loaded; the
  official ``test`` split is never read, instantiated or referenced.
* The frozen PSCD-v0 artifact (dictionary + merge sequence + occurrence
  partitions + canonical motifs + ports + composition edges) is loaded and
  re-asserted for 100 % exact decode before any training.  If exactness changed,
  the run stops without training.
* No vocabulary / merge-rule / port-definition change; no task-aware BPE; no
  motif-size or vocabulary sweep; no attention / transformer / virtual node;
  no depth/width sweep; single seed.
* Screening budget is expressed in optimizer steps (3000), never epochs.

Reproduce
---------
    uv run python tracks/ksvd/code/run_pscd_reader_diagnostic.py --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import platform
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (  # noqa: E402
    MAX_MOTIF,
    MotifType,
    Vocabulary,
    canon_subgraph,
    load_mols_and_y,
    mol_iso_key,
    decode,
)

RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/pscd_reader_diagnostic"
FROZEN_ARTIFACT = REPO_ROOT / "tracks/ksvd/results/pscd_compositional/stageA.pkl"

# ---------------------------------------------------------------------------
# Frozen configuration (no sweep)
# ---------------------------------------------------------------------------
D_MODEL = 32
OUTER_LAYERS = 2
L_INTRA = 2
SEED = 0

BATCH = 128
LR = 1e-3
WD = 0.0
CLIP = 5.0

MAX_STEPS = 3000
EXTENDED_STEPS = 6000
EVAL_EVERY = 100
CHECKPOINTS = (500, 1000, 2000, 3000, 6000)

EARLY_START = 1000
EARLY_PATIENCE = 5
EARLY_MIN_DELTA = 0.003

HARD_FAIL_STEP = 2000
HARD_FAIL_VALID = 0.48
HARD_FAIL_TRAIN = 0.43

RECOVERY_GATE = 0.50
RECOVERY_GATE_RAW_MARGIN = 0.10
RECOVERY_GATE_TREND = 0.02


def _pscd():
    return importlib.import_module(
        "tracks.ksvd.code.run_pscd_compositional_dictionary_audit")


def _torch():
    import torch

    return torch


# ===========================================================================
# 0. Frozen artifact load (with ``__main__`` remap for the historical pickle)
# ===========================================================================
def load_frozen_artifact(path: Path):
    pscd = _pscd()

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "__main__":
                return getattr(pscd, name)
            return super().find_class(module, name)

    with path.open("rb") as fh:
        data = _Unpickler(fh).load()
    return data


# ===========================================================================
# 1. Shared structural dictionary encoder (R1 / R2)
# ===========================================================================
def build_motif_bank(vocab: Vocabulary, atom_index: dict[int, int],
                     bond_index: dict[int, int]):
    """Block-diagonal bank of the K canonical motif graphs.

    Returns plain python lists; the torch module registers them as buffers.
    """
    K = len(vocab)
    node_type: list[int] = []
    edge_a: list[int] = []
    edge_b: list[int] = []
    edge_type: list[int] = []
    node_motif: list[int] = []
    node_off = [0] * K
    motif_size = [0] * K
    for mid in range(K):
        mt = vocab.by_id[mid]
        off = len(node_type)
        node_off[mid] = off
        motif_size[mid] = mt.size
        node_type.extend(int(atom_index[int(c)]) for c in mt.canon_nt)
        node_motif.extend([int(mid)] * mt.size)
        for (a, b, t) in mt.canon_bonds:
            e = int(bond_index[int(t)])
            edge_a.extend([off + int(a), off + int(b)])
            edge_b.extend([off + int(b), off + int(a)])
            edge_type.extend([e, e])
    return {
        "node_type": node_type,
        "edge_index": [edge_a, edge_b],
        "edge_type": edge_type,
        "node_motif": node_motif,
        "node_off": node_off,
        "motif_size": motif_size,
    }


def make_structural_encoder(vocab: Vocabulary, atom_index: dict[int, int],
                            bond_index: dict[int, int], d: int = D_MODEL,
                            layers: int = L_INTRA):
    torch = _torch()
    nn = torch.nn
    bank = build_motif_bank(vocab, atom_index, bond_index)

    class StructuralDictEncoder(nn.Module):
        """Shared tiny graph encoder over all dictionary motifs.

        ``z_v^0 = E_V(x_v)``; per layer
        ``m_v = sum_{u in N(v)} psi_l(z_u, b_uv)`` and
        ``z_v' = SiLU(W_s z_v + W_m m_v + b)``.  No attention / norm / residual.
        Exposes ``sum_v z_v`` as the motif state and ``z_p`` as the port state.
        """

        def __init__(self):
            super().__init__()
            self.E_V = nn.Embedding(max(atom_index.values()) + 1, d)
            self.E_E = nn.Embedding(max(bond_index.values()) + 1, d)
            self.W_s = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.W_m = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.b = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(layers)])
            self.psi = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
            self.d = d
            self.layers = layers
            self.register_buffer("node_type", torch.as_tensor(bank["node_type"], dtype=torch.long))
            self.register_buffer("edge_index",
                                 torch.as_tensor(bank["edge_index"], dtype=torch.long))
            self.register_buffer("edge_type", torch.as_tensor(bank["edge_type"], dtype=torch.long))
            self.register_buffer("node_motif", torch.as_tensor(bank["node_motif"], dtype=torch.long))
            self.register_buffer("node_off", torch.as_tensor(bank["node_off"], dtype=torch.long))
            self.register_buffer("motif_size",
                                 torch.as_tensor(bank["motif_size"], dtype=torch.float32))
            self.K = len(bank["motif_size"])

        def forward(self):
            x = self.E_V(self.node_type)
            e = self.E_E(self.edge_type)
            src, dst = self.edge_index
            for li in range(self.layers):
                msg = self.psi[li](torch.cat([x[src], e], dim=-1))
                agg = torch.zeros_like(x)
                agg.index_add_(0, dst, msg)
                x = torch.nn.functional.silu(self.W_s[li](x) + self.W_m[li](agg) + self.b[li])
            h_dict = torch.zeros(self.K, self.d, device=x.device, dtype=x.dtype)
            h_dict.index_add_(0, self.node_motif, x)
            return h_dict, x

    return StructuralDictEncoder()


# ===========================================================================
# 2. Batch collation
# ===========================================================================
def collate_comp_fixed(items, vocab, bond_index, device):
    """Correctly-ordered composition batch: all occurrence nodes first, then all
    connection nodes, edges remapped accordingly.  Keeps the PSCD-v0 module
    (which assumes ``h[:M] = E_M(occ_ids)`` / ``h[M:] = E_B(conn_ids)``)."""
    torch = _torch()
    occ_ids: list[int] = []
    conn_ids: list[int] = []
    src: list[int] = []
    dst: list[int] = []
    port: list[int] = []
    pool: list[int] = []
    Ms = [len(sc.occ_motif_ids) for sc in items]
    Ecs = [len(sc.comp_edges) for sc in items]
    m_tot = int(sum(Ms))
    occ_base = []
    conn_base = []
    a = b = 0
    for m, ec in zip(Ms, Ecs):
        occ_base.append(a)
        conn_base.append(b)
        a += m
        b += ec
    for gi, sc in enumerate(items):
        occ_ids.extend(int(k) for k in sc.occ_motif_ids)
        pool.extend([gi] * Ms[gi])
        for idx, (i, pi, t, j, pj) in enumerate(sc.comp_edges):
            conn_ids.append(int(bond_index[int(t)]))
            c = m_tot + conn_base[gi] + idx
            for (s, dd, p) in ((occ_base[gi] + int(i), c, int(pi)),
                               (c, occ_base[gi] + int(i), int(pi)),
                               (occ_base[gi] + int(j), c, int(pj)),
                               (c, occ_base[gi] + int(j), int(pj))):
                src.append(s)
                dst.append(dd)
                port.append(p)
    return {
        "occ_ids": torch.as_tensor(occ_ids, dtype=torch.long, device=device),
        "conn_ids": torch.as_tensor(conn_ids, dtype=torch.long, device=device),
        "n_nodes": m_tot + len(conn_ids),
        "ports": torch.as_tensor(port, dtype=torch.long, device=device),
        "edge_index": torch.as_tensor([src, dst], dtype=torch.long, device=device),
        "pool_index": torch.as_tensor(pool, dtype=torch.long, device=device),
        "n_graphs": len(items),
    }


def collate_r1(items, vocab, bond_index, device):
    """Structured-static batch: same [all occ | all conn] node order as
    ``collate_comp_fixed`` but with per-directed-edge occurrence/slot tags so the
    reader can gather the dictionary port vectors ``r_{k,p}``."""
    torch = _torch()
    Ms = [len(sc.occ_motif_ids) for sc in items]
    m_tot = int(sum(Ms))
    occ_base = []
    a = 0
    for m in Ms:
        occ_base.append(a)
        a += m
    occ_ids: list[int] = []
    conn_ids: list[int] = []
    src: list[int] = []
    dst: list[int] = []
    edge_occ: list[int] = []       # global local occurrence-node index per directed edge
    edge_occ_id: list[int] = []    # motif id of that occurrence
    edge_slot: list[int] = []
    pool: list[int] = []
    conn_cursor = 0
    for gi, sc in enumerate(items):
        M = len(sc.occ_motif_ids)
        occ_ids.extend(int(k) for k in sc.occ_motif_ids)
        pool.extend([gi] * M)
        for idx, (i, pi, t, j, pj) in enumerate(sc.comp_edges):
            c = m_tot + conn_cursor + idx  # connection node global index
            conn_ids.append(int(bond_index[int(t)]))
            for (s, dd, ocal, slot) in ((occ_base[gi] + int(i), c, int(i), int(pi)),
                                        (c, occ_base[gi] + int(i), int(i), int(pi)),
                                        (occ_base[gi] + int(j), c, int(j), int(pj)),
                                        (c, occ_base[gi] + int(j), int(j), int(pj))):
                src.append(s)
                dst.append(dd)
                edge_occ.append(occ_base[gi] + ocal)
                edge_occ_id.append(int(sc.occ_motif_ids[ocal]))
                edge_slot.append(slot)
        conn_cursor += len(sc.comp_edges)
    return {
        "occ_ids": torch.as_tensor(occ_ids, dtype=torch.long, device=device),
        "conn_ids": torch.as_tensor(conn_ids, dtype=torch.long, device=device),
        "n_nodes": m_tot + len(conn_ids),
        "m_tot": m_tot,
        "edge_index": torch.as_tensor([src, dst], dtype=torch.long, device=device),
        "edge_occ": torch.as_tensor(edge_occ, dtype=torch.long, device=device),
        "edge_occ_id": torch.as_tensor(edge_occ_id, dtype=torch.long, device=device),
        "edge_slot": torch.as_tensor(edge_slot, dtype=torch.long, device=device),
        "pool_index": torch.as_tensor(pool, dtype=torch.long, device=device),
        "n_graphs": len(items),
    }


def collate_r2(items, vocab, bond_index, device):
    """Port-resolved batch: flat active-port states + directed external messages."""
    torch = _torch()
    occ_ids: list[int] = []
    port_occ: list[int] = []
    port_slot: list[int] = []
    msg_dst: list[int] = []
    msg_src: list[int] = []
    msg_bond: list[int] = []
    pool: list[int] = []
    off_occ = 0
    for gi, sc in enumerate(items):
        M = len(sc.occ_motif_ids)
        occ_ids.extend(int(k) for k in sc.occ_motif_ids)
        pool.extend([gi] * M)
        port_index: dict[tuple[int, int], int] = {}
        for (i, pi, t, j, pj) in sc.comp_edges:
            for (o, p) in ((int(i), int(pi)), (int(j), int(pj))):
                key = (o, p)
                if key not in port_index:
                    port_index[key] = len(port_occ)
                    port_occ.append(off_occ + o)
                    port_slot.append(p)
        for (i, pi, t, j, pj) in sc.comp_edges:
            b = int(bond_index[int(t)])
            # message into port (i,pi) from (j,pj)
            msg_dst.append(port_index[(int(i), int(pi))])
            msg_src.append(port_index[(int(j), int(pj))])
            msg_bond.append(b)
            # message into port (j,pj) from (i,pi)
            msg_dst.append(port_index[(int(j), int(pj))])
            msg_src.append(port_index[(int(i), int(pi))])
            msg_bond.append(b)
        off_occ += M
    return {
        "occ_ids": torch.as_tensor(occ_ids, dtype=torch.long, device=device),
        "port_occ": torch.as_tensor(port_occ, dtype=torch.long, device=device),
        "port_slot": torch.as_tensor(port_slot, dtype=torch.long, device=device),
        "msg_dst": torch.as_tensor(msg_dst, dtype=torch.long, device=device),
        "msg_src": torch.as_tensor(msg_src, dtype=torch.long, device=device),
        "msg_bond": torch.as_tensor(msg_bond, dtype=torch.long, device=device),
        "pool_index": torch.as_tensor(pool, dtype=torch.long, device=device),
        "n_graphs": len(items),
        "m_tot": off_occ,
    }


# ===========================================================================
# 3. Readers
# ===========================================================================
def make_r1_reader(encoder, n_bond: int, d: int = D_MODEL, layers: int = OUTER_LAYERS):
    torch = _torch()
    nn = torch.nn

    class R1Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = "r1"
            self.enc = encoder
            self.E_B = nn.Embedding(n_bond, d)
            self.W_self = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.W_msg = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.W_edge = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
            self.d = d
            self.layers = layers

        def forward(self, batch, port_mean: bool = False):
            h_dict, x = self.enc()
            m_tot = batch["m_tot"]
            h = torch.zeros(batch["n_nodes"], self.d, device=h_dict.device, dtype=h_dict.dtype)
            h[:m_tot] = h_dict[batch["occ_ids"]]
            h[m_tot:] = self.E_B(batch["conn_ids"])
            if port_mean:
                mean_port = h_dict / self.enc.motif_size.unsqueeze(-1)
                edge_feat = mean_port[batch["edge_occ_id"]]
            else:
                node_global = self.enc.node_off[batch["edge_occ_id"]] + batch["edge_slot"]
                edge_feat = x[node_global]
            src, dst = batch["edge_index"]
            for li in range(self.layers):
                msg = self.W_msg[li](h)[src] + self.W_edge[li](edge_feat)
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, msg)
                h = torch.nn.functional.silu(self.W_self[li](h) + agg)
            z = torch.zeros(batch["n_graphs"], self.d, device=h.device, dtype=h.dtype)
            z.index_add_(0, batch["pool_index"], h[:m_tot])
            return self.head(z).squeeze(-1)

    return R1Reader()


def make_r2_reader(encoder, n_bond: int, d: int = D_MODEL, layers: int = OUTER_LAYERS):
    torch = _torch()
    nn = torch.nn

    class R2Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = "r2"
            self.enc = encoder
            self.E_B = nn.Embedding(n_bond, d)
            self.W_P = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.W_R = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.W_S = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.W_M = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.W_H = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.W_A = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.psi = nn.ModuleList([nn.Linear(3 * d, d) for _ in range(layers)])
            self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
            self.d = d
            self.layers = layers

        def forward(self, batch, port_mean: bool = False):
            h_dict, x = self.enc()
            m_tot = batch["m_tot"]
            h = h_dict[batch["occ_ids"]]
            if batch["port_occ"].numel() == 0:
                z = torch.zeros(batch["n_graphs"], self.d, device=h.device, dtype=h.dtype)
                z.index_add_(0, batch["pool_index"], h)
                return self.head(z).squeeze(-1)
            if port_mean:
                r = h_dict[batch["occ_ids"][batch["port_occ"]]] \
                    / self.enc.motif_size[batch["occ_ids"][batch["port_occ"]]].unsqueeze(-1)
            else:
                node_global = self.enc.node_off[batch["occ_ids"][batch["port_occ"]]] + batch["port_slot"]
                r = x[node_global]
            p_tot = r.shape[0]
            eb_all = self.E_B(batch["msg_bond"])
            for li in range(self.layers):
                msg = self.psi[li](torch.cat([r[batch["msg_dst"]], r[batch["msg_src"]], eb_all], dim=-1))
                m = torch.zeros(p_tot, self.d, device=r.device, dtype=r.dtype)
                m.index_add_(0, batch["msg_dst"], msg)
                wpr = self.W_P[li](r)
                agg = torch.zeros(m_tot, self.d, device=r.device, dtype=r.dtype)
                agg.index_add_(0, batch["port_occ"], wpr)
                s = h + agg
                r = torch.nn.functional.silu(
                    self.W_R[li](r) + self.W_S[li](s[batch["port_occ"]]) + self.W_M[li](m))
                agg2 = torch.zeros(m_tot, self.d, device=r.device, dtype=r.dtype)
                agg2.index_add_(0, batch["port_occ"], r)
                h = torch.nn.functional.silu(self.W_H[li](h) + self.W_A[li](agg2))
            z = torch.zeros(batch["n_graphs"], self.d, device=h.device, dtype=h.dtype)
            z.index_add_(0, batch["pool_index"], h)
            return self.head(z).squeeze(-1)

    return R2Reader()


# ===========================================================================
# 4. Step-budget trainer
# ===========================================================================
class StepTrainer:
    def __init__(self, name: str, model, model_kind: str, build_batch,
                 tr_items, va_items, ytr, yva, device, log, structured: bool):
        torch = _torch()
        self.name = name
        self.model = model
        self.kind = model_kind
        self.build_batch = build_batch
        self.tr_items = tr_items
        self.va_items = va_items
        self.ytr = ytr
        self.yva = yva
        self.device = device
        self.log = log
        self.structured = structured
        self.opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        self.gen = torch.Generator().manual_seed(SEED + 4242)
        self.n = len(tr_items)
        self.order: list[int] = []
        self.pos = 0
        self.step = 0
        self.history: list[dict[str, Any]] = []
        self.best_valid = float("inf")
        self.best_step = 0
        self.best_valid_after = float("inf")
        self.stale = 0
        self.stopped = False
        self.stop_reason: str | None = None
        self.early_enabled = True
        self.t_start = time.time()
        self.peak_gpu_mb = 0.0
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    # -- helpers -----------------------------------------------------------
    def _forward_all(self, items):
        torch = _torch()
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(items), BATCH):
                batch = self.build_batch(items[start:start + BATCH])
                preds.append(self.model(batch))
        return torch.cat(preds)

    def _ref_metric(self, items, y):
        pred = self._forward_all(items)
        return float((pred - y).abs().mean())

    def evaluate(self) -> tuple[float, float]:
        tr = self._ref_metric(self.tr_items, self.ytr)
        va = self._ref_metric(self.va_items, self.yva)
        wall = time.time() - self.t_start
        rec = {"step": self.step, "train_mae": tr, "valid_mae": va, "wall_s": wall}
        if self.device.startswith("cuda"):
            torch = _torch()
            self.peak_gpu_mb = max(
                self.peak_gpu_mb, float(torch.cuda.max_memory_allocated() / 1e6))
            rec["peak_gpu_mb"] = self.peak_gpu_mb
        self.history.append(rec)
        if va < self.best_valid - 1e-12:
            self.best_valid = va
            self.best_step = self.step
        if self.early_enabled and self.step > EARLY_START:
            if va < self.best_valid_after - EARLY_MIN_DELTA:
                self.best_valid_after = va
                self.stale = 0
            else:
                self.stale += 1
                if self.stale >= EARLY_PATIENCE:
                    self.stopped = True
                    self.stop_reason = f"early_stop@best={self.best_valid:.4f}"
        if self.structured and self.step == HARD_FAIL_STEP:
            if va > HARD_FAIL_VALID and tr > HARD_FAIL_TRAIN:
                self.stopped = True
                self.stop_reason = (
                    f"hard_fail@2000 train={tr:.4f}> {HARD_FAIL_TRAIN} "
                    f"valid={va:.4f}> {HARD_FAIL_VALID}")
        return tr, va

    # -- training ----------------------------------------------------------
    def run_to(self, target: int):
        torch = _torch()
        self.model.train()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        self.t_start = time.time() - (self.history[-1]["wall_s"] if self.history else 0.0)
        while self.step < target and not self.stopped:
            if self.pos >= len(self.order):
                self.order = torch.randperm(self.n, generator=self.gen).tolist()
                self.pos = 0
            take = min(BATCH, self.n - self.pos)
            idx = self.order[self.pos:self.pos + take]
            self.pos += take
            batch = self.build_batch([self.tr_items[i] for i in idx])
            pred = self.model(batch)
            tgt = self.ytr[torch.as_tensor(idx, dtype=torch.long, device=self.device)]
            loss = torch.nn.functional.l1_loss(pred, tgt)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
            self.opt.step()
            self.step += 1
            if self.step % EVAL_EVERY == 0:
                tr, va = self.evaluate()
                if self.step in CHECKPOINTS or self.step % 500 == 0:
                    self.log(f"[{self.name}] step {self.step} train={tr:.4f} "
                             f"valid={va:.4f} best={self.best_valid:.4f} "
                             f"wall={time.time() - self.t_start:.0f}s")
                if self.stopped:
                    self.log(f"[{self.name}] stopped at step {self.step}: {self.stop_reason}")
                    break
        return self

    def cont_to(self, target: int):
        self.stopped = False
        self.stop_reason = None
        self.early_enabled = False
        self.stale = 0
        self.best_valid_after = float("inf")
        return self.run_to(target)

    # -- reporting ---------------------------------------------------------
    def final_metrics(self) -> dict[str, Any]:
        last = self.history[-1]
        return {
            "name": self.name,
            "kind": self.kind,
            "steps": self.step,
            "train_mae": last["train_mae"],
            "valid_mae": last["valid_mae"],
            "best_valid": self.best_valid,
            "best_step": self.best_step,
            "wall_s": last["wall_s"],
            "steps_per_sec": self.step / max(last["wall_s"], 1e-9),
            "peak_gpu_mb": self.peak_gpu_mb,
            "stop_reason": self.stop_reason,
            "early_stopped": bool(self.stop_reason and "early_stop" in str(self.stop_reason)),
            "hard_fail": bool(self.stop_reason and "hard_fail" in str(self.stop_reason)),
        }

    def at_step(self, step: int):
        for rec in self.history:
            if rec["step"] == step:
                return rec
        return None


# ===========================================================================
# 5. Parameter accounting
# ===========================================================================
def param_breakdown(model) -> dict[str, Any]:
    buckets = defaultdict(int)
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        total += n
        if name.startswith("enc."):
            key = "dictionary_encoder"
        elif name.startswith("head."):
            key = "head"
        elif name.startswith("E_M."):
            key = "per_motif_lookup"
        elif name.startswith("E_P."):
            key = "port_lookup"
        elif name.startswith("E_A."):
            key = "atom_lookup"
        elif name.startswith("E_B."):
            key = "bond_lookup"
        else:
            key = "outer_reader"
        buckets[key] += n
    out = dict(buckets)
    out["total_trainable"] = total
    out["dictionary_shared"] = buckets.get("dictionary_encoder", 0)
    out["per_motif_lookup_total"] = buckets.get("per_motif_lookup", 0)
    return out


# ===========================================================================
# 6. Mechanism audit + ablation
# ===========================================================================
def _canon_key_of_subset(mt: MotifType, subset: Sequence[int]) -> bytes:
    sub = sorted(int(s) for s in subset)
    bonds = [(int(a), int(b), int(t)) for (a, b, t) in mt.canon_bonds]
    key, _cn, _cb = canon_subgraph(sub, bonds, mt.canon_nt)
    return key


def shared_substructure_pairs(vocab: Vocabulary, mids: Sequence[int], max_size: int = 3):
    """Pairs of learned motifs sharing a connected induced subgraph up to size 4."""
    keysets: dict[int, set] = {}
    for mid in mids:
        mt = vocab.by_id[mid]
        ks = set()
        for s in range(2, min(mt.size, max_size + 1) + 1):
            for sub in combinations(range(mt.size), s):
                ks.add(_canon_key_of_subset(mt, sub))
        keysets[mid] = ks
    related = []
    for a, b in combinations(mids, 2):
        if keysets[a] & keysets[b]:
            related.append((a, b))
    return related


def mechanism_audit(encoder, vocab: Vocabulary, learned_mids: Sequence[int],
                    log=print) -> dict[str, Any]:
    torch = _torch()
    encoder.eval()
    with torch.no_grad():
        h_dict, x = encoder()
    h = h_dict.detach().cpu().numpy()
    xnp = x.detach().cpu().numpy()
    off = encoder.node_off.detach().cpu().numpy()

    # A. motif embedding diversity
    H = h[list(learned_mids)]
    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-9)
    cos = Hn @ Hn.T
    iu = np.triu_indices(len(H), k=1)
    cos_vals = cos[iu]
    dmat = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=-1)
    dist_vals = dmat[iu]
    s = np.linalg.svd(H - H.mean(0, keepdims=True), compute_uv=False)
    eff_rank = float((s ** 2).sum() ** 2 / max((s ** 4).sum(), 1e-12))

    # B. structural similarity sanity
    related = shared_substructure_pairs(vocab, learned_mids)
    rel_set = set()
    for a, b in related:
        rel_set.add((min(a, b), max(a, b)))
    idx = {mid: i for i, mid in enumerate(learned_mids)}
    rel_d, unrel_d = [], []
    for a, b in combinations(learned_mids, 2):
        d = float(dmat[idx[a], idx[b]])
        if (min(a, b), max(a, b)) in rel_set:
            rel_d.append(d)
        else:
            unrel_d.append(d)

    # C. port differentiation
    port_cos_mean = []
    for mid in learned_mids:
        mt = vocab.by_id[mid]
        if mt.size < 2:
            continue
        z = xnp[off[mid]:off[mid] + mt.size]
        zn = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
        pc = zn @ zn.T
        iu2 = np.triu_indices(mt.size, k=1)
        port_cos_mean.append(float(pc[iu2].mean()))
    port_cos_mean = np.asarray(port_cos_mean)

    return {
        "n_learned": len(learned_mids),
        "diversity": {
            "mean_pairwise_cosine": float(cos_vals.mean()),
            "min_pairwise_cosine": float(cos_vals.min()),
            "max_pairwise_cosine": float(cos_vals.max()),
            "frac_cosine_gt_0.99": float((cos_vals > 0.99).mean()),
            "mean_pairwise_euclidean": float(dist_vals.mean()),
            "min_pairwise_euclidean": float(dist_vals.min()),
            "effective_rank": eff_rank,
        },
        "structural_similarity": {
            "n_related_pairs": len(rel_d),
            "n_unrelated_pairs": len(unrel_d),
            "mean_euclidean_related": float(np.mean(rel_d)) if rel_d else None,
            "mean_euclidean_unrelated": float(np.mean(unrel_d)) if unrel_d else None,
        },
        "port_differentiation": {
            "n_motifs_ge2": int(port_cos_mean.size),
            "mean_port_pairwise_cosine": float(port_cos_mean.mean()) if port_cos_mean.size else None,
            "min_motif_mean_port_cosine": float(port_cos_mean.min()) if port_cos_mean.size else None,
            "max_motif_mean_port_cosine": float(port_cos_mean.max()) if port_cos_mean.size else None,
            "frac_motifs_port_cos_lt_0.9": float((port_cos_mean < 0.9).mean()) if port_cos_mean.size else None,
        },
    }


def eval_only_ablation(trainer: StepTrainer, log=print) -> dict[str, Any]:
    """Replace every dictionary port vector by the motif mean at inference."""
    torch = _torch()
    model = trainer.model
    model.eval()

    def run(port_mean: bool):
        tr_pred, va_pred = [], []
        with torch.no_grad():
            for start in range(0, len(trainer.tr_items), BATCH):
                b = trainer.build_batch(trainer.tr_items[start:start + BATCH])
                tr_pred.append(model(b, port_mean=port_mean))
            for start in range(0, len(trainer.va_items), BATCH):
                b = trainer.build_batch(trainer.va_items[start:start + BATCH])
                va_pred.append(model(b, port_mean=port_mean))
        tr = float((torch.cat(tr_pred) - trainer.ytr).abs().mean())
        va = float((torch.cat(va_pred) - trainer.yva).abs().mean())
        return tr, va

    tr0, va0 = run(False)
    tr1, va1 = run(True)
    return {
        "train_mae_intact": tr0,
        "valid_mae_intact": va0,
        "train_mae_port_mean": tr1,
        "valid_mae_port_mean": va1,
        "delta_valid": va1 - va0,
        "delta_train": tr1 - tr0,
    }


# ===========================================================================
# 7. Orchestration
# ===========================================================================
def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
                    encoding="utf-8")


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def recovery_ratio(m_opaque, m_structured, m_raw):
    denom = m_opaque - m_raw
    if abs(denom) < 1e-9:
        return None
    return (m_opaque - m_structured) / denom


def make_plots(out: Path, trainers: dict[str, StepTrainer], log=print):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        log(f"[plot] skipped: {exc!r}")
        return
    colors = {"R0": "#1f77b4", "R0ship": "#8c564b", "R1": "#2ca02c",
              "R2": "#d62728", "R3": "#7f7f7f"}
    for metric, fname, title in (("train_mae", "learning_curve_train.png", "train MAE"),
                                 ("valid_mae", "learning_curve_valid.png", "valid MAE")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, tr in trainers.items():
            steps = [r["step"] for r in tr.history]
            vals = [r[metric] for r in tr.history]
            ax.plot(steps, vals, label=name, color=colors.get(name, None), lw=1.8)
        for c in CHECKPOINTS:
            ax.axvline(c, color="#cccccc", lw=0.6, zorder=0)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(metric)
        ax.set_title(f"PSCD-R-v0 — {title} vs optimizer step")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)
    log("[plot] wrote learning curves")


def main(argv: Sequence[str] | None = None) -> int:
    global EVAL_EVERY, MAX_STEPS, CHECKPOINTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--artifact", type=Path, default=FROZEN_ARTIFACT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="smoke only")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--skip-exactness", action="store_true")
    parser.add_argument("--readers", type=str, default="R0ship,R0,R1,R2,R3",
                        help="comma-separated subset of R0ship,R0,R1,R2,R3")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="disable early termination (phase-2 extension only)")
    args = parser.parse_args(argv)

    EVAL_EVERY = args.eval_every
    MAX_STEPS = args.max_steps

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log("=== PSCD-R-v0 start ===")
    provenance = {
        "round": "PSCD-R-v0",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": args.device,
        "seed": SEED,
        "data_root": str(args.data_root),
        "artifact": str(args.artifact),
        "batch": BATCH, "lr": LR, "wd": WD, "clip": CLIP,
        "eval_every": EVAL_EVERY, "max_steps": MAX_STEPS,
    }
    write_json(out / "provenance.json", provenance)

    # --- load frozen artifact -------------------------------------------
    if not args.artifact.exists():
        raise SystemExit(f"frozen artifact missing: {args.artifact}")
    data = load_frozen_artifact(args.artifact)
    vocab: Vocabulary = data["vocab"]
    sequence = data["sequence"]
    train_codes = list(data["train_codes"])
    valid_codes = list(data["valid_codes"])
    log(f"[load] frozen dictionary |D|={len(vocab)} (64 learned), "
        f"rules={len(sequence)}, train_codes={len(train_codes)}, "
        f"valid_codes={len(valid_codes)}")

    # --- data ------------------------------------------------------------
    limit = args.limit
    train_mols, y_train = load_mols_and_y(args.data_root, "train", limit)
    valid_mols, y_valid = load_mols_and_y(
        args.data_root, "valid", None if limit is None else max(100, limit // 3))
    n_tr = min(len(train_codes), len(train_mols))
    n_va = min(len(valid_codes), len(valid_mols))
    train_codes, valid_codes = train_codes[:n_tr], valid_codes[:n_va]
    train_mols, y_train = train_mols[:n_tr], y_train[:n_tr]
    valid_mols, y_valid = valid_mols[:n_va], y_valid[:n_va]
    log(f"[load] ZINC train={len(train_mols)} valid={len(valid_mols)} "
        f"(official test never loaded)")

    # --- exactness gate --------------------------------------------------
    if not args.skip_exactness:
        log("[freeze] re-asserting Decode(D, C_G) == G on train + valid")
        recon = {"train": 0, "valid": 0}
        for split, mols, codes in (("train", train_mols, train_codes),
                                   ("valid", valid_mols, valid_codes)):
            for mol, code in zip(mols, codes):
                if mol_iso_key(decode(code, vocab)) == mol_iso_key(mol):
                    recon[split] += 1
        recon["train_frac"] = recon["train"] / max(len(train_mols), 1)
        recon["valid_frac"] = recon["valid"] / max(len(valid_mols), 1)
        log(f"[freeze] exact train={recon['train_frac']:.6f} valid={recon['valid_frac']:.6f}")
        if recon["train_frac"] != 1.0 or recon["valid_frac"] != 1.0:
            write_json(out / "SUMMARY.json", {"provenance": provenance,
                                              "reconstruction": recon, "aborted": True})
            raise SystemExit("frozen artifact exactness changed — aborting (no training)")
    else:
        recon = {"skipped": True}

    pscd = _pscd()
    torch = _torch()
    device = args.device
    torch.manual_seed(SEED)

    atom_cats = sorted({int(x) for mol in list(train_mols) + list(valid_mols)
                        for x in mol.node_types.tolist()}
                       | {int(c) for mt in vocab.by_id.values() for c in mt.canon_nt})
    bond_cats = sorted({int(x) for mol in list(train_mols) + list(valid_mols)
                        for x in mol.bond_types.tolist()}
                       | {int(t) for mt in vocab.by_id.values()
                          for (_a, _b, t) in mt.canon_bonds})
    atom_index = {c: i for i, c in enumerate(atom_cats)}
    bond_index = {c: i for i, c in enumerate(bond_cats)}
    n_motif = len(vocab)
    n_bond = len(bond_cats)

    ytr = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    yva = torch.as_tensor(y_valid, dtype=torch.float32, device=device)
    learned_mids = [mt.mid for mt in vocab.by_id.values() if not mt.singleton]

    # --- readers ---------------------------------------------------------
    def bb_r0ship(items):
        return pscd.collate(items, vocab, atom_index, bond_index, "comp", device)

    def bb_r0(items):
        return collate_comp_fixed(items, vocab, bond_index, device)

    def bb_r1(items):
        return collate_r1(items, vocab, bond_index, device)

    def bb_r2(items):
        return collate_r2(items, vocab, bond_index, device)

    def bb_r3(items):
        return pscd.collate(items, vocab, atom_index, bond_index, "raw", device)

    def new_encoder():
        return make_structural_encoder(vocab, atom_index, bond_index).to(device)

    torch.manual_seed(SEED)
    models = {
        "R0": (pscd.make_module("comp", n_motif, n_bond, MAX_MOTIF).to(device),
               "comp", bb_r0, train_codes, valid_codes, False),
        "R0ship": (pscd.make_module("comp", n_motif, n_bond, MAX_MOTIF).to(device),
                   "comp_ship", bb_r0ship, train_codes, valid_codes, False),
        "R1": (make_r1_reader(new_encoder(), n_bond).to(device),
               "r1", bb_r1, train_codes, valid_codes, True),
        "R2": (make_r2_reader(new_encoder(), n_bond).to(device),
               "r2", bb_r2, train_codes, valid_codes, True),
        "R3": (pscd.make_module("raw", n_motif, n_bond, MAX_MOTIF).to(device),
               "raw", bb_r3, train_mols, valid_mols, False),
    }

    trainers: dict[str, StepTrainer] = {}
    order = ["R0ship", "R0", "R1", "R2", "R3"]
    selected = [s for s in args.readers.split(",") if s]
    for name in order:
        if name not in selected:
            continue
        model, kind, bb, tr_items, va_items, structured = models[name]
        log(f"[train] {name} kind={kind} params={sum(p.numel() for p in model.parameters())}")
        tr = StepTrainer(name, model, kind, bb, tr_items, va_items, ytr, yva,
                         device, log, structured)
        if args.no_early_stop:
            tr.early_enabled = False
        tr.run_to(MAX_STEPS)
        trainers[name] = tr
        log(f"[train] {name} done: {json.dumps(tr.final_metrics())}")

    # --- recovery ratios -------------------------------------------------
    have_core = all(n in trainers for n in ("R0", "R1", "R2", "R3"))
    recovery: dict[str, Any] = {}
    gate_info: dict[str, Any] = {}
    extensions: dict[str, Any] = {}

    def build_recovery():
        common = sorted({r["step"] for r in trainers["R0"].history} &
                        {r["step"] for r in trainers["R3"].history})
        out = {}
        for step in common:
            r0 = trainers["R0"].at_step(step)
            r3 = trainers["R3"].at_step(step)
            if r0 is None or r3 is None:
                continue
            row = {"step": step}
            for sname in ("R1", "R2", "R0ship"):
                rec = trainers[sname].at_step(step)
                if rec is None:
                    continue
                row[sname] = {
                    "train_recover": recovery_ratio(r0["train_mae"], rec["train_mae"],
                                                    r3["train_mae"]),
                    "valid_recover": recovery_ratio(r0["valid_mae"], rec["valid_mae"],
                                                    r3["valid_mae"]),
                }
            out[str(step)] = row
        return out, common

    best_structured = None
    if have_core:
        recovery, common_ckpts = build_recovery()

        # --- extension gate ----------------------------------------------
        structured_names = ["R1", "R2"]
        best_structured = min(structured_names,
                              key=lambda s: trainers[s].final_metrics()["valid_mae"])
        bs = trainers[best_structured]
        raw = trainers["R3"]
        matched_step = max([s for s in common_ckpts if s <= bs.step], default=bs.step)
        gate_info = {"best_structured": best_structured, "matched_step": matched_step,
                     "criteria": {}}
        rec_last = bs.at_step(matched_step) or bs.history[-1]
        prev = [r for r in bs.history if r["step"] <= matched_step - 1000]
        trend_valid = (prev[-1]["valid_mae"] - rec_last["valid_mae"]) if prev else None
        trend_train = (prev[-1]["train_mae"] - rec_last["train_mae"]) if prev else None
        raw_last = raw.at_step(matched_step) or raw.history[-1]
        raw_prev = [r for r in raw.history if r["step"] <= raw_last["step"] - 1000]
        raw_trend_valid = (raw_prev[-1]["valid_mae"] - raw_last["valid_mae"]) \
            if raw_prev else None
        recov_valid = (recovery.get(str(matched_step), {}).get(best_structured, {})
                       .get("valid_recover"))
        e1 = recov_valid is not None and recov_valid >= RECOVERY_GATE
        e2 = rec_last["valid_mae"] <= raw_last["valid_mae"] + RECOVERY_GATE_RAW_MARGIN
        e3 = (trend_valid is not None and trend_valid >= RECOVERY_GATE_TREND
              and trend_train is not None and trend_train > 0)
        gate_info["criteria"] = {
            "recover_valid_at_matched_step": recov_valid,
            "e1_recover_ge_0.50": bool(e1),
            "e2_within_raw_plus_0.10": bool(e2),
            "e3_trend_ge_0.02": bool(e3),
            "trend_valid_last1000": trend_valid,
            "trend_train_last1000": trend_train,
            "raw_trend_valid_last1000": raw_trend_valid,
        }
        gate_info["pass"] = bool(e1 or e2 or e3)
        log(f"[gate] {json.dumps(gate_info['criteria'], default=float)} "
            f"pass={gate_info['pass']}")

        # --- extended budget ---------------------------------------------
        if gate_info["pass"] and bs.step < EXTENDED_STEPS and not args.no_early_stop:
            log(f"[extend] continuing {best_structured} to {EXTENDED_STEPS} steps")
            bs.cont_to(EXTENDED_STEPS)
            extensions[best_structured] = bs.final_metrics()
        if (not args.no_early_stop) and raw_trend_valid is not None \
                and raw_trend_valid >= RECOVERY_GATE_TREND and raw.step < EXTENDED_STEPS:
            log(f"[extend] raw control still improving ({raw_trend_valid:.4f}); "
                f"continuing to {EXTENDED_STEPS}")
            raw.cont_to(EXTENDED_STEPS)
            extensions["R3"] = raw.final_metrics()
        recovery, _ = build_recovery()

    # --- results tables --------------------------------------------------
    table = {name: trainers[name].final_metrics() for name in trainers}
    for name in trainers:
        table[name]["params"] = param_breakdown(models[name][0])

    # --- mechanism audit + ablation -------------------------------------
    mechanism: dict[str, Any] = {}
    ablation: dict[str, Any] = {}
    audit_reader: str | None = None
    structured_present = [n for n in ("R1", "R2") if n in trainers]
    if structured_present:
        audit_reader = min(structured_present,
                           key=lambda s: trainers[s].final_metrics()["valid_mae"])
        enc = models[audit_reader][0].enc
        mechanism = mechanism_audit(enc, vocab, learned_mids, log=log)
        ablation = eval_only_ablation(trainers[audit_reader], log=log)

    make_plots(out, trainers, log=log)
    for name, tr in trainers.items():
        write_json(out / f"history_{name}.json", tr.history)

    summary = {
        "provenance": provenance,
        "reconstruction": recon,
        "frozen_dictionary": {
            "n_types": len(vocab),
            "n_learned": len(learned_mids),
            "rules": len(sequence),
        },
        "readers": table,
        "recovery": recovery,
        "extension_gate": gate_info,
        "extensions": extensions,
        "mechanism_audit": {"reader": audit_reader, **mechanism},
        "port_mean_ablation": {"reader": audit_reader, **ablation},
        "checkpoints": list(CHECKPOINTS),
    }
    write_json(out / "SUMMARY.json", summary)
    log("=== PSCD-R-v0 done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
