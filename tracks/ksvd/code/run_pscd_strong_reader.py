#!/usr/bin/env python
"""PSCD-SC-v0 — Strong-Capacity Port-Operator Reader Audit.

With the frozen PSCD-v0 dictionary ``(D, C_G)`` held completely fixed, does a
**strong** port-operator reader (full canonical training budget, ``d=64``,
``L_outer=4``, alternating inter-motif / intra-motif operator) follow the raw
graph down into a ``0.1-0.2`` MAE band?  PSCD-R-v0 localised the historical
failure to the reader (batching defect + opaque/port-collapsed reader) at a tiny
budget; this round asks the converged-capacity question.

* **Arm A** (separate, reused builder): canonical strong raw ``zinc-b-full``.
* **Arm B** (this module): decode oracle -- same Arm-A weights on ``G`` vs
  ``Decode(C_G)``.
* **Arm C** (this module): strong compressed port-operator reader.

Discipline: only official ZINC train (10000) / valid (1000); official test never
loaded.  Frozen artifact re-decoded for 100 % exact reconstruction before any
training.  No task-aware search / sweep / Gumbel / learned boundaries.  Fixed
capacity ``d=64, L_outer=4, L_motif=2, L_intra=2`` (no hidden-size tuning).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (  # noqa: E402
    MAX_MOTIF,
    StructuredCode,
    Vocabulary,
    build_structured_code,
    decode,
    load_mols_and_y,
    mol_iso_key,
    mol_rank,
    tokenize_frozen,
)
from tracks.ksvd.code.run_aiom_representation_audit import permute_mol  # noqa: E402
from tracks.ksvd.code.run_pscd_reader_diagnostic import (  # noqa: E402
    load_frozen_artifact,
    make_structural_encoder,
)

RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/pscd_strong_reader"
FROZEN_ARTIFACT = REPO_ROOT / "tracks/ksvd/results/pscd_compositional/stageA.pkl"
BFULL_STATES = REPO_ROOT / "tracks/ksvd/results/shared_structural_patch_encoder/states"

# --- frozen capacity (no sweep) --------------------------------------------
D_MODEL = 64
OUTER_LAYERS = 4
L_MOTIF = 2
L_INTRA = 2
SEED = 0

# --- canonical B-Full OPTIMIZED_PROTOCOL (reused) --------------------------
BATCH = 128
LR = 1e-3
WD = 1e-5
CLIP = 5.0
MAX_EPOCHS = 240
PATIENCE = 40
TRAIN_SHUFFLE_SEED_OFFSET = 91011
TOP_K = 5


def _torch():
    import torch

    return torch


# ===========================================================================
# 0. Frozen artifact
# ===========================================================================
def load_frozen(path: Path) -> dict[str, Any]:
    data = load_frozen_artifact(path)
    return {
        "vocab": data["vocab"],
        "sequence": data["sequence"],
        "train_codes": list(data["train_codes"]),
        "valid_codes": list(data["valid_codes"]),
    }


def assert_exact_decode(vocab: Vocabulary, mols, codes) -> dict[str, Any]:
    exact = sum(1 for mol, code in zip(mols, codes)
                if mol_iso_key(decode(code, vocab)) == mol_iso_key(mol))
    return {"exact": exact, "n": len(mols), "frac": exact / max(len(mols), 1)}


def category_index(vocab: Vocabulary, mols) -> tuple[dict[int, int], dict[int, int]]:
    atom_cats = sorted({int(x) for mol in mols for x in mol.node_types.tolist()}
                       | {int(c) for mt in vocab.by_id.values() for c in mt.canon_nt})
    bond_cats = sorted({int(x) for mol in mols for x in mol.bond_types.tolist()}
                       | {int(t) for mt in vocab.by_id.values() for (_a, _b, t) in mt.canon_bonds})
    return ({c: i for i, c in enumerate(atom_cats)},
            {c: i for i, c in enumerate(bond_cats)})


# ===========================================================================
# 1. Frozen-code precompute (shared by forward, tests, compression)
# ===========================================================================
def motif_node_offsets(vocab: Vocabulary) -> list[int]:
    off = []
    acc = 0
    for k in range(len(vocab)):
        off.append(acc)
        acc += vocab.by_id[k].size
    return off


def build_tau_index(vocab: Vocabulary, node_off: Sequence[int]) -> dict[str, Any]:
    """Dictionary-level canonical-slot-pair table for the transfer descriptor.

    Row ``motif_off[k] + p*size_k + q`` -> ``(g_k, s_{k,p}, s_{k,q})``.
    """
    node_p: list[int] = []
    node_q: list[int] = []
    motif: list[int] = []
    motif_off = [0] * len(vocab)
    acc = 0
    for k in range(len(vocab)):
        motif_off[k] = acc
        size = vocab.by_id[k].size
        off = node_off[k]
        for p in range(size):
            for q in range(size):
                node_p.append(off + p)
                node_q.append(off + q)
                motif.append(k)
        acc += size * size
    return {
        "node_p": np.asarray(node_p, dtype=np.int64),
        "node_q": np.asarray(node_q, dtype=np.int64),
        "motif": np.asarray(motif, dtype=np.int64),
        "motif_off": motif_off,
        "n_rows": acc,
    }


def precompute_graph(sc: StructuredCode, vocab: Vocabulary, tau_index,
                     node_off: Sequence[int], bond_index: dict[int, int] | None = None
                     ) -> dict[str, Any]:
    occ_ids = np.asarray(sc.occ_motif_ids, dtype=np.int64)
    M = int(occ_ids.shape[0])
    port_index: dict[tuple[int, int], int] = {}
    port_occ: list[int] = []
    port_slot: list[int] = []
    port_node: list[int] = []
    for (i, pi, t, j, pj) in sc.comp_edges:
        for (o, p) in ((int(i), int(pi)), (int(j), int(pj))):
            if (o, p) not in port_index:
                port_index[(o, p)] = len(port_occ)
                port_occ.append(o)
                port_slot.append(p)
                port_node.append(node_off[int(occ_ids[o])] + p)
    ext_src: list[int] = []
    ext_dst: list[int] = []
    ext_bond: list[int] = []
    for (i, pi, t, j, pj) in sc.comp_edges:
        bt = int(bond_index[int(t)]) if bond_index is not None else int(t)
        ext_dst.append(port_index[(int(i), int(pi))])
        ext_src.append(port_index[(int(j), int(pj))])
        ext_bond.append(bt)
        ext_dst.append(port_index[(int(j), int(pj))])
        ext_src.append(port_index[(int(i), int(pi))])
        ext_bond.append(bt)
    occ_ports: dict[int, list[int]] = defaultdict(list)
    for pid, o in enumerate(port_occ):
        occ_ports[o].append(pid)
    intra_p: list[int] = []
    intra_q: list[int] = []
    intra_row: list[int] = []
    for o in range(M):
        pids = occ_ports.get(o, [])
        k = int(occ_ids[o])
        size = vocab.by_id[k].size
        base = tau_index["motif_off"][k]
        for pid_p in pids:
            for pid_q in pids:
                intra_p.append(pid_p)
                intra_q.append(pid_q)
                intra_row.append(base + port_slot[pid_p] * size + port_slot[pid_q])
    return {
        "occ_ids": occ_ids,
        "port_occ": np.asarray(port_occ, dtype=np.int64),
        "port_node": np.asarray(port_node, dtype=np.int64),
        "ext_src": np.asarray(ext_src, dtype=np.int64),
        "ext_dst": np.asarray(ext_dst, dtype=np.int64),
        "ext_bond": np.asarray(ext_bond, dtype=np.int64),
        "intra_p": np.asarray(intra_p, dtype=np.int64),
        "intra_q": np.asarray(intra_q, dtype=np.int64),
        "intra_row": np.asarray(intra_row, dtype=np.int64),
        "n_occ": M,
        "n_port": len(port_occ),
        "n_ext": len(ext_src),
        "n_intra": len(intra_p),
    }


def precompute_all(codes, vocab: Vocabulary, bond_index: dict[int, int] | None = None):
    node_off = motif_node_offsets(vocab)
    tau_index = build_tau_index(vocab, node_off)
    pre = [precompute_graph(sc, vocab, tau_index, node_off, bond_index) for sc in codes]
    return pre, tau_index, node_off


def collate_portop(items: Sequence[dict], device) -> dict[str, Any]:
    torch = _torch()
    occ, port_occ, port_node = [], [], []
    ext_src, ext_dst, ext_bond = [], [], []
    intra_p, intra_q, intra_row = [], [], []
    pool = []
    m_off = 0
    p_off = 0
    for gi, it in enumerate(items):
        occ.append(it["occ_ids"])
        pool.append(np.full(it["n_occ"], gi, dtype=np.int64))
        port_occ.append(it["port_occ"] + m_off)
        port_node.append(it["port_node"])
        if it["n_ext"]:
            ext_src.append(it["ext_src"] + p_off)
            ext_dst.append(it["ext_dst"] + p_off)
            ext_bond.append(it["ext_bond"])
        if it["n_intra"]:
            intra_p.append(it["intra_p"] + p_off)
            intra_q.append(it["intra_q"] + p_off)
            intra_row.append(it["intra_row"])
        m_off += it["n_occ"]
        p_off += it["n_port"]

    def cat(parts):
        if not parts:
            return torch.zeros(0, dtype=torch.long, device=device)
        return torch.as_tensor(np.concatenate(parts), dtype=torch.long, device=device)

    return {
        "occ_ids": cat(occ), "port_occ": cat(port_occ), "port_node": cat(port_node),
        "ext_src": cat(ext_src), "ext_dst": cat(ext_dst), "ext_bond": cat(ext_bond),
        "intra_p": cat(intra_p), "intra_q": cat(intra_q), "intra_row": cat(intra_row),
        "pool_index": cat(pool), "n_graphs": len(items),
    }


# ===========================================================================
# 2. Arm C — strong compressed port-operator reader
# ===========================================================================
def make_port_operator(encoder, n_bond: int, tau_index, d: int = D_MODEL,
                       layers: int = OUTER_LAYERS, intra_transfer: bool = True):
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    class StrongPortOperator(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = "pscd_port_operator"
            self.enc = encoder
            self.d = d
            self.layers = layers
            self.intra_transfer = bool(intra_transfer)
            self.E_B = nn.Embedding(n_bond, d)
            self.W_C = nn.Linear(d, d)   # shared state map (init + center self)
            self.W_S = nn.Linear(d, d)   # port self (shared)
            self.W_G = nn.Linear(d, d)   # center -> port (shared)
            self.W_P = nn.Linear(d, d)   # port -> center (shared)
            self.W_D = nn.Linear(d, d)   # motif descriptor -> center (shared)
            self.psi = nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d))
            self.phi_ext = nn.ModuleList(
                [nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d))
                 for _ in range(layers)])
            self.phi_int = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
                 for _ in range(layers)])
            self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
            self.register_buffer("tau_node_p", torch.as_tensor(tau_index["node_p"]))
            self.register_buffer("tau_node_q", torch.as_tensor(tau_index["node_q"]))
            self.register_buffer("tau_motif", torch.as_tensor(tau_index["motif"]))

        def forward(self, batch, intra_transfer: bool | None = None):
            use_intra = self.intra_transfer if intra_transfer is None else bool(intra_transfer)
            g, x = self.enc()  # enc returns (h_dict=K motif descriptors, x=per-node states)
            occ = batch["occ_ids"]
            M = int(occ.shape[0])
            P = int(batch["port_occ"].shape[0])
            c = g[occ]
            wc_c = self.W_C(c)
            tau = self.psi(torch.cat([x[self.tau_node_p], x[self.tau_node_q],
                                      g[self.tau_motif]], dim=-1))
            h = x[batch["port_node"]] + wc_c[batch["port_occ"]] if P > 0 \
                else x.new_zeros((0, self.d))
            for li in range(self.layers):
                if P > 0:
                    m_ext = self.phi_ext[li](torch.cat(
                        [h[batch["ext_src"]], h[batch["ext_dst"]],
                         self.E_B(batch["ext_bond"])], dim=-1))
                    tilde = h.clone()
                    tilde.index_add_(0, batch["ext_dst"], m_ext)
                else:
                    tilde = h
                if P > 0:
                    if use_intra and batch["intra_p"].numel() > 0:
                        m_int = self.phi_int[li](torch.cat(
                            [tilde[batch["intra_p"]], tau[batch["intra_row"]]], dim=-1))
                        agg = h.new_zeros((P, self.d))
                        agg.index_add_(0, batch["intra_q"], m_int)
                        h_new = F.silu(self.W_S(h) + agg + self.W_G(c)[batch["port_occ"]])
                    else:
                        # ablation: external message reaches the center through
                        # the port; no p -> q pairwise transfer.
                        h_new = F.silu(self.W_S(tilde) + self.W_G(c)[batch["port_occ"]])
                    agg_c = h.new_zeros((M, self.d))
                    agg_c.index_add_(0, batch["port_occ"], self.W_P(h_new))
                else:
                    h_new = h
                    agg_c = c.new_zeros((M, self.d))
                c = F.silu(wc_c + agg_c + self.W_D(g[occ]))
                wc_c = self.W_C(c)
                h = h_new
            z = c.new_zeros((batch["n_graphs"], self.d))
            z.index_add_(0, batch["pool_index"], c)
            return self.head(z).squeeze(-1)

    return StrongPortOperator()


def param_breakdown(model) -> dict[str, Any]:
    buckets: dict[str, int] = defaultdict(int)
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        total += n
        if name.startswith("enc."):
            key = "motif_structural_encoder"
        elif name.startswith("head."):
            key = "head"
        elif name.startswith("psi."):
            key = "intra_transfer_descriptor"
        elif name.startswith("phi_ext."):
            key = "inter_motif"
        elif name.startswith("phi_int."):
            key = "intra_motif"
        elif name.startswith("E_B."):
            key = "bond_lookup"
        else:
            key = "shared_state_maps"
        buckets[key] += n
    out = dict(buckets)
    out["total_trainable"] = total
    return out


# ===========================================================================
# 3. Compression / dynamic-compute metrics
# ===========================================================================
def compression_metrics(codes, pre, mols, vocab: Vocabulary) -> dict[str, Any]:
    n_raw = int(sum(m.n for m in mols))
    n_occ = int(sum(it["n_occ"] for it in pre))
    n_port = int(sum(it["n_port"] for it in pre))
    n_ext = int(sum(it["n_ext"] for it in pre))
    n_intra = int(sum(it["n_intra"] for it in pre))
    n_raw_edges = int(sum(2 * m.m for m in mols))
    return {
        "n_graphs": len(mols), "N_raw": n_raw, "N_pscd": n_occ + n_port,
        "n_occ": n_occ, "n_active_ports": n_port,
        "r_state": (n_occ + n_port) / max(n_raw, 1),
        "occurrence_atom_ratio": n_occ / max(n_raw, 1),
        "inter_motif_messages": n_ext,
        "intra_motif_port_pair_messages": n_intra,
        "raw_edge_messages": n_raw_edges,
        "inter_over_raw": n_ext / max(n_raw_edges, 1),
        "intra_over_raw": n_intra / max(n_raw_edges, 1),
        "pscd_messages_over_raw": (n_ext + n_intra) / max(n_raw_edges, 1),
    }


# ===========================================================================
# 4. Data-free tests
# ===========================================================================
def _synth_code(vocab, mids, edges, n):
    return StructuredCode(n, len(edges), list(mids), sorted(edges))


def data_free_tests(vocab: Vocabulary, mols_for_cats, device: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    atom_index, bond_index = category_index(vocab, mols_for_cats)
    n_bond = max(len(bond_index), 1)
    torch.manual_seed(SEED)
    enc = make_structural_encoder(vocab, atom_index, bond_index, d=D_MODEL, layers=L_MOTIF)
    _, tau_index, node_off = precompute_all([], vocab)
    model = make_port_operator(enc, n_bond=n_bond, tau_index=tau_index).to(device).eval()
    checks: dict[str, Any] = {}

    frozen = load_frozen(FROZEN_ARTIFACT)
    seq = frozen["sequence"]
    val_codes = frozen["valid_codes"][:24]
    pre = [precompute_graph(sc, vocab, tau_index, node_off, bond_index) for sc in val_codes]

    def fwd(items):
        with torch.no_grad():
            return model(collate_portop(items, device))

    # A. batch invariance
    batched = fwd(pre)
    singles = torch.stack([fwd([it])[0] for it in pre])
    d = float((batched - singles).abs().max())
    checks["A_batch_invariance"] = {"max_abs_diff": d, "pass": bool(d < 1e-5)}

    # B. molecule permutation invariance (relabel the decoded valid molecule and
    #    refreeze the canonical code) + C. motif automorphism invariance.
    perm_max = 0.0
    auto_max = 0.0
    auto_checked = 0
    for idx, sc in enumerate(val_codes):
        mol = decode(sc, vocab)
        rng = random.Random(1000 + idx)
        pa = list(range(mol.n))
        rng.shuffle(pa)
        pb = list(range(mol.m))
        rng.shuffle(pb)
        pm = permute_mol(mol, pa, pb)
        toks = tokenize_frozen(pm, vocab, seq, mol_rank(pm))
        code = build_structured_code(pm, toks, vocab, mol_rank(pm))["code"]
        pp = precompute_graph(code, vocab, tau_index, node_off, bond_index)
        diff = float((fwd([pp])[0] - fwd([pre[idx]])[0]).abs().max())
        perm_max = max(perm_max, diff)

    # C. motif automorphism: build an isomorphic pair of graphs that differ only
    #    by which same-orbit canonical slot carries the (identical) external bond.
    for k in range(len(vocab)):
        mt = vocab.by_id[k]
        if mt.size < 2:
            continue
        by_orbit: dict[int, list[int]] = defaultdict(list)
        for s, o in enumerate(mt.orbits):
            by_orbit[int(o)].append(s)
        pair = None
        for orbit in by_orbit.values():
            if len(orbit) >= 2:
                pair = (orbit[0], orbit[1])
                break
        if pair is None:
            continue
        p, q = pair
        # external singleton of the motif's boundary atom type; isolated atom 1
        atom_type = int(mt.canon_nt[0])
        singleton_mid = vocab.by_key.get(b"s" + atom_type.to_bytes(2, "little"))
        if singleton_mid is None:
            continue
        singleton_mid = singleton_mid.mid
        t = int(mt.canon_bonds[0][2]) if mt.canon_bonds else 1
        n = mt.size + 1
        g1 = _synth_code(vocab, [k, singleton_mid], [(0, p, t, 1, 0)], n)
        g2 = _synth_code(vocab, [k, singleton_mid], [(0, q, t, 1, 0)], n)
        pp1 = precompute_graph(g1, vocab, tau_index, node_off, bond_index)
        pp2 = precompute_graph(g2, vocab, tau_index, node_off, bond_index)
        auto_max = max(auto_max, float((fwd([pp1])[0] - fwd([pp2])[0]).abs().max()))
        auto_checked += 1
        if auto_checked >= 8:
            break
    checks["B_molecule_permutation_invariance"] = {"max_abs_diff": perm_max, "pass": bool(perm_max < 1e-5)}
    checks["C_motif_automorphism_invariance"] = {
        "n_pairs_checked": auto_checked, "max_abs_diff": auto_max,
        "pass": bool(auto_checked == 0 or auto_max < 1e-5)}

    # D. empty / single-port / multi-port synthetic motifs are finite
    singletons = [mid for mid, mt in vocab.by_id.items() if mt.singleton][:6]
    finite = True
    if len(singletons) >= 3:
        empty = _synth_code(vocab, singletons[:3], [], 3)
        one = _synth_code(vocab, singletons[:2], [(0, 0, 1, 1, 0)], 2)
        multi = _synth_code(vocab, singletons[:4],
                            [(0, 0, 1, 1, 0), (0, 0, 1, 2, 0), (0, 0, 1, 3, 0)], 4)
        for sc in (empty, one, multi):
            pp = precompute_graph(sc, vocab, tau_index, node_off, bond_index)
            out = fwd([pp])
            finite = finite and bool(torch.isfinite(out).all())
    checks["D_empty_single_multi_port_finite"] = {"pass": bool(finite)}

    # E. port routing sensitivity: same motif, different canonical port slot
    route_diff = 0.0
    for idx, sc in enumerate(val_codes):
        if len(sc.comp_edges) < 1:
            continue
        i, pi, t, j, pj = sc.comp_edges[0]
        size_i = vocab.by_id[sc.occ_motif_ids[i]].size
        if size_i < 2:
            continue
        ce = list(sc.comp_edges)
        ce[0] = (i, (pi + 1) % size_i, t, j, pj)
        alt = _synth_code(vocab, sc.occ_motif_ids, ce, sc.n)
        pp0 = precompute_graph(sc, vocab, tau_index, node_off, bond_index)
        pp1 = precompute_graph(alt, vocab, tau_index, node_off, bond_index)
        route_diff = float((fwd([pp0])[0] - fwd([pp1])[0]).abs().max())
        break
    checks["E_port_routing_sensitivity"] = {"max_abs_diff": route_diff,
                                            "pass": bool(route_diff > 1e-8)}

    passed = all(bool(v.get("pass")) for v in checks.values())
    return {"passed": passed, "checks": checks}


# ===========================================================================
# 5. Arm B — decode oracle (same Arm-A model + weights)
# ===========================================================================
def _mol_to_data(mol, y: float):
    torch = _torch()
    from torch_geometric.data import Data

    src: list[int] = []
    dst: list[int] = []
    bt: list[int] = []
    for i, (a, b) in enumerate(mol.bonds):
        t = int(mol.bond_types[i])
        src.extend([int(a), int(b)])
        dst.extend([int(b), int(a)])
        bt.extend([t, t])
    return Data(
        x=torch.as_tensor(mol.node_types, dtype=torch.long).view(-1, 1),
        edge_index=torch.as_tensor([src, dst], dtype=torch.long),
        edge_attr=torch.as_tensor(bt, dtype=torch.long).view(-1, 1),
        y=torch.tensor([float(y)], dtype=torch.float32), num_nodes=int(mol.n))


def arm_b_decode_oracle(n_train: int, n_valid: int, device: str, state_path: Path,
                        log=print) -> dict[str, Any]:
    torch = _torch()
    import importlib

    sbse = importlib.import_module(
        "tracks.ksvd.experiments.luyin16.zinc_shared_structural_patch_encoder")
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
    from tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency import (
        base_config as _v4_base_config,
    )

    frozen = load_frozen(FROZEN_ARTIFACT)
    vocab = frozen["vocab"]
    train_mols, ys_train = load_mols_and_y(REPO_ROOT / "data/ZINC", "train", n_train)
    valid_mols, ys_valid = load_mols_and_y(REPO_ROOT / "data/ZINC", "valid", n_valid)

    train_bundle, valid_bundle, _meta = sbse.extract_records()
    train_records = train_bundle["records"]
    valid_records = valid_bundle["records"]

    config = dict(_v4_base_config())
    config["model"]["device"] = "cpu"
    model = sbse.build_candidate(0)
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()

    def build_decoded(codes, ref_records, mols, ys):
        out_records, out_data = [], []
        cache: dict[bytes, bytes] = {}
        decoded = [(decode(code, vocab), float(y)) for code, y in zip(codes, ys)]
        for (dm, y), ref in zip(decoded, ref_records):
            out_data.append(_mol_to_data(dm, y))
        for dd, ref in zip(out_data, ref_records):
            out_records.append(zpp._graph_record(
                dd, ref.global_context, cache, patch_radius=sbse.PATCH_RADIUS,
                topology_features=ref.topology_features))
        return out_records, out_data

    t0 = time.time()
    dec_train_records, dec_train_data = build_decoded(
        frozen["train_codes"][:n_train], train_records[:n_train], train_mols, ys_train)
    dec_valid_records, dec_valid_data = build_decoded(
        frozen["valid_codes"][:n_valid], valid_records[:n_valid], valid_mols, ys_valid)
    log(f"[armb] decoded records built in {time.time() - t0:.1f}s")

    _, enc_dec_train, _ = zpp._phase_data(train_records, dec_train_records, config=dict(config))
    _, enc_dec_valid, _ = zpp._phase_data(train_records, dec_valid_records, config=dict(config))
    sbse._attach_struct_tensors(enc_dec_train, sbse._patch_graphs_from_dataset(dec_train_data))
    sbse._attach_struct_tensors(enc_dec_valid, sbse._patch_graphs_from_dataset(dec_valid_data))
    # reference graph encodings come straight from the canonical pipeline (with
    # struct tensors attached), so the comparison is apples-to-apples.
    enc_train_ref, enc_valid_ref, _audit = sbse.build_encoded_records()
    enc_ref_train = enc_train_ref[:n_train]
    enc_ref_valid = enc_valid_ref[:n_valid]

    def predict(encoded):
        loader = sbse._make_struct_loader(encoded, 64, False, 0)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds.append(model(batch).view(-1).cpu())
        return torch.cat(preds)

    dv = (predict(enc_ref_valid) - predict(enc_dec_valid)).abs()
    dt = (predict(enc_ref_train) - predict(enc_dec_train)).abs()
    payload = {
        "arm": "B_decode_oracle", "model": "zinc-b-full (same instance as Arm A)",
        "state_path": str(state_path), "device": device,
        "n_train": n_train, "n_valid": n_valid,
        "valid": {"max_abs_delta": float(dv.max()), "mean_abs_delta": float(dv.mean()),
                  "p95_abs_delta": float(dv.quantile(0.95)),
                  "frac_lt_1e-5": float((dv < 1e-5).float().mean())},
        "train": {"max_abs_delta": float(dt.max()), "mean_abs_delta": float(dt.mean()),
                  "p95_abs_delta": float(dt.quantile(0.95)),
                  "frac_lt_1e-5": float((dt < 1e-5).float().mean())},
        "pass_1e-5": bool(float(dv.max()) < 1e-5 and float(dt.max()) < 1e-5),
        "official_test_loaded": False,
    }
    return payload


# ===========================================================================
# 6. Training (canonical protocol), soup, evaluation
# ===========================================================================
def _evaluate_valid(model, pre, y, device, batch=BATCH) -> float:
    torch = _torch()
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for s in range(0, len(pre), batch):
            b = collate_portop(pre[s:s + batch], device)
            pred = model(b)
            tgt = y[s:s + batch]
            total += float((pred - tgt).abs().sum())
            seen += int(tgt.numel())
    return total / max(seen, 1)


def _soup(states):
    torch = _torch()
    return {k: torch.stack([s[k].float() for s in states], 0).mean(0).to(states[0][k].dtype)
            for k in states[0]}


def train_seed(vocab: Vocabulary, atom_index, bond_index, train_codes, valid_codes,
               y_train, y_valid, device: str = "cpu", tag: str = "sc", seed: int = SEED,
               intra_transfer: bool = True, max_epochs: int = MAX_EPOCHS,
               patience: int = PATIENCE, log=print) -> dict[str, Any]:
    torch = _torch()
    pre_train, tau_index, _ = precompute_all(train_codes, vocab, bond_index)
    pre_valid, _, _ = precompute_all(valid_codes, vocab, bond_index)
    ytr = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    yva = torch.as_tensor(y_valid, dtype=torch.float32, device=device)
    n_bond = max(len(bond_index), 1)
    torch.manual_seed(seed)
    enc = make_structural_encoder(vocab, atom_index, bond_index, d=D_MODEL, layers=L_MOTIF)
    model = make_port_operator(enc, n_bond=n_bond, tau_index=tau_index,
                               intra_transfer=intra_transfer).to(device)
    params = param_breakdown(model)
    log(f"[{tag}] params={json.dumps(params)}")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(seed + TRAIN_SHUFFLE_SEED_OFFSET)
    n = len(pre_train)
    steps_per_epoch = int(math.ceil(n / BATCH))
    best, best_epoch, best_state, stale = float("inf"), 1, None, 0
    top: list[tuple[float, int, dict]] = []
    curve: list[dict[str, Any]] = []
    t0 = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=gen).tolist()
        ep_loss, seen = 0.0, 0
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            b = collate_portop([pre_train[i] for i in idx], device)
            pred = model(b)
            tgt = ytr[torch.as_tensor(idx, dtype=torch.long, device=device)]
            loss = torch.nn.functional.l1_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            ep_loss += float(loss.detach()) * int(tgt.numel())
            seen += int(tgt.numel())
        train_mae = ep_loss / max(seen, 1)
        valid_mae = _evaluate_valid(model, pre_valid, yva, device)
        curve.append({"epoch": epoch, "train_mae": train_mae, "valid_mae": valid_mae,
                      "lr": float(opt.param_groups[0]["lr"]),
                      "optimizer_steps": epoch * steps_per_epoch})
        state = copy.deepcopy(model.state_dict())
        top.append((valid_mae, epoch, state))
        top.sort(key=lambda z: (z[0], z[1]))
        top = top[:TOP_K]
        if valid_mae < best:
            best, best_epoch, best_state, stale = valid_mae, epoch, state, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            log(f"[{tag}] epoch={epoch:03d} train={train_mae:.6f} valid={valid_mae:.6f} "
                f"best={best:.6f}@{best_epoch} wall={time.time() - t0:.0f}s")
        if stale >= patience:
            log(f"[{tag}] early_stop epoch={epoch} best={best_epoch}")
            break
    wall = time.time() - t0
    peak = float(torch.cuda.max_memory_allocated() / 1e6) if device.startswith("cuda") else 0.0
    if best_state is not None:
        model.load_state_dict(best_state)
    best_valid = _evaluate_valid(model, pre_valid, yva, device)
    soup_valid = None
    soup_state = None
    if len(top) >= TOP_K:
        soup_state = _soup([s for _, _, s in top])
        model.load_state_dict(soup_state)
        soup_valid = _evaluate_valid(model, pre_valid, yva, device)
    summary = {
        "tag": tag, "seed": seed, "intra_transfer": bool(intra_transfer), "params": params,
        "epochs_run": len(curve), "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": len(curve) * steps_per_epoch,
        "best_valid_mae": float(best), "best_epoch": int(best_epoch),
        "best_checkpoint_valid_mae": float(best_valid),
        "top5_soup_valid_mae": (None if soup_valid is None else float(soup_valid)),
        "train_mae_at_best": float(curve[best_epoch - 1]["train_mae"]),
        "wall_s": wall, "graphs_per_sec": n * len(curve) / max(wall, 1e-9),
        "peak_gpu_mb": peak, "official_test_loaded": False,
    }
    return {"summary": summary, "curve": curve, "top5": [(v, e) for v, e, _ in top],
            "best_state": best_state, "soup_state": soup_state}


# ===========================================================================
# 7. IO
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


def provenance(stage: str, device: str) -> dict[str, Any]:
    return {
        "round": "PSCD-SC-v0", "stage": stage,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "device": device, "seed": SEED,
        "config": {"d": D_MODEL, "L_outer": OUTER_LAYERS, "L_motif": L_MOTIF,
                   "L_intra": L_INTRA, "batch": BATCH, "lr": LR, "wd": WD, "clip": CLIP,
                   "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                   "train_shuffle_seed_offset": TRAIN_SHUFFLE_SEED_OFFSET},
    }


# ===========================================================================
# 8. CLI
# ===========================================================================
def _load_split(args, frozen):
    train_mols, y_train = load_mols_and_y(args.data_root, "train", args.limit)
    valid_mols, y_valid = load_mols_and_y(
        args.data_root, "valid", None if args.limit is None else max(100, args.limit // 3))
    n_tr = min(len(frozen["train_codes"]), len(train_mols))
    n_va = min(len(frozen["valid_codes"]), len(valid_mols))
    return (frozen["train_codes"][:n_tr], frozen["valid_codes"][:n_va],
            train_mols[:n_tr], valid_mols[:n_va], np.asarray(y_train[:n_tr]),
            np.asarray(y_valid[:n_va]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prechecks", "armb", "train", "metrics"])
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--artifact", type=Path, default=FROZEN_ARTIFACT)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--tag", type=str, default="sc")
    parser.add_argument("--intra-transfer", type=str, default="full", choices=["full", "none"])
    parser.add_argument("--n-train-armb", type=int, default=512)
    parser.add_argument("--n-valid-armb", type=int, default=512)
    parser.add_argument("--bfull-state", type=Path,
                        default=BFULL_STATES / "sc_bfull_s0_seed0_selection_state.pt")
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    if args.stage == "armb":
        log("=== PSCD-SC-v0 Arm B (decode oracle) ===")
        if not args.bfull_state.exists():
            raise SystemExit(f"Arm-A state missing: {args.bfull_state}")
        payload = arm_b_decode_oracle(args.n_train_armb, args.n_valid_armb,
                                      args.device, args.bfull_state, log=log)
        payload["provenance"] = provenance("armb", args.device)
        write_json(out / "arm_b_decode_oracle.json", payload)
        log(f"[armb] {json.dumps(payload, default=float)}")
        return 0

    frozen = load_frozen(args.artifact)
    vocab = frozen["vocab"]
    train_codes, valid_codes, train_mols, valid_mols, y_train, y_valid = _load_split(args, frozen)

    if args.stage == "prechecks":
        log("=== PSCD-SC-v0 prechecks ===")
        rec_tr = assert_exact_decode(vocab, train_mols, train_codes)
        rec_va = assert_exact_decode(vocab, valid_mols, valid_codes)
        log(f"[freeze] train {rec_tr['exact']}/{rec_tr['n']} valid {rec_va['exact']}/{rec_va['n']}")
        if rec_tr["frac"] != 1.0 or rec_va["frac"] != 1.0:
            raise SystemExit("frozen artifact exactness changed — aborting")
        tests = data_free_tests(vocab, train_mols + valid_mols, args.device)
        log(f"[tests] {json.dumps(tests, default=float)}")
        _ai, _bi = category_index(vocab, train_mols + valid_mols)
        pre_tr, _, _ = precompute_all(train_codes, vocab, _bi)
        pre_va, _, _ = precompute_all(valid_codes, vocab, _bi)
        payload = {
            "provenance": provenance("prechecks", args.device),
            "reconstruction": {"train": rec_tr, "valid": rec_va},
            "data_free_tests": tests,
            "compression": {
                "train": compression_metrics(train_codes, pre_tr, train_mols, vocab),
                "valid": compression_metrics(valid_codes, pre_va, valid_mols, vocab),
            },
        }
        write_json(out / "prechecks.json", payload)
        if not tests["passed"]:
            raise SystemExit("data-free tests failed — aborting")
        return 0

    if args.stage == "metrics":
        _ai, _bi = category_index(vocab, train_mols + valid_mols)
        pre_tr, _, _ = precompute_all(train_codes, vocab, _bi)
        pre_va, _, _ = precompute_all(valid_codes, vocab, _bi)
        payload = {
            "provenance": provenance("metrics", args.device),
            "train": compression_metrics(train_codes, pre_tr, train_mols, vocab),
            "valid": compression_metrics(valid_codes, pre_va, valid_mols, vocab),
        }
        write_json(out / "compression.json", payload)
        log(json.dumps(payload, default=float))
        return 0

    # --- train -----------------------------------------------------------
    log("=== PSCD-SC-v0 Arm C train ===")
    rec_tr = assert_exact_decode(vocab, train_mols, train_codes)
    rec_va = assert_exact_decode(vocab, valid_mols, valid_codes)
    log(f"[freeze] train {rec_tr['exact']}/{rec_tr['n']} valid {rec_va['exact']}/{rec_va['n']}")
    if rec_tr["frac"] != 1.0 or rec_va["frac"] != 1.0:
        raise SystemExit("frozen artifact exactness changed — aborting")
    tests = data_free_tests(vocab, train_mols + valid_mols, args.device)
    log(f"[tests] {json.dumps(tests, default=float)}")
    if not tests["passed"]:
        raise SystemExit("data-free tests failed — aborting")
    atom_index, bond_index = category_index(vocab, train_mols)
    result = train_seed(vocab, atom_index, bond_index, train_codes, valid_codes,
                        y_train, y_valid, device=args.device, tag=args.tag, seed=SEED,
                        intra_transfer=(args.intra_transfer == "full"),
                        max_epochs=args.max_epochs, patience=args.patience, log=log)
    write_json(out / f"history_{args.tag}.json", result["curve"])
    torch = _torch()
    if result["best_state"] is not None:
        torch.save(result["best_state"], out / f"{args.tag}_best_state.pt")
    if result["soup_state"] is not None:
        torch.save(result["soup_state"], out / f"{args.tag}_top5_soup.pt")
    write_json(out / f"summary_{args.tag}.json", {
        "provenance": provenance("train", args.device),
        "reconstruction": {"train": rec_tr, "valid": rec_va},
        "data_free_tests": tests, **result["summary"],
    })
    log(f"[train] done {json.dumps(result['summary'], default=float)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
