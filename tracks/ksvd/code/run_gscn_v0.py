#!/usr/bin/env python
"""GSCN-v0 -- Task-Driven Sparse Dictionary-Core Graph Network (ZINC feasibility).

Pre-registration: ``tracks/ksvd/notes/gscn_v0_preregistration.md`` (frozen before
any run).  Study ``zinc-context-gap``, deterministic A100, canonical PyG ZINC
``subset=True`` official **train 10000 / valid 1000**; **official test never
loaded**.

Core question
-------------
Can a graph model learn task-relevant structure directly from *raw* topology by
repeatedly coding relational states over shared sparse dictionaries, instead of
using handcrafted structural statistics?

Architecture (autonomous; no side channel, no raw bypass)::

    H^(l) --relational mixing--> Y^(l) --ISTA sparse dictionary code--> H^(l+1)

with ``H^(l+1) = LayerNorm(A^(l) Dbar^(l))`` and ``A^(l)`` the non-negative
``T=3`` ISTA code over the per-layer overcomplete dictionary ``D^(l) in R^{K x d}``
(``K=2d``).  Inputs are ONLY raw atom category / raw bond category / adjacency.

Arms
----
* ``bnull_raw`` -- raw baseline ``SiLU(W_self h + sum(W_msg h_u + W_edge e_uv))``.
* ``generic``   -- param-matched generic nonlinear control (mixing + MLP).
* ``nothresh``  -- full GSCN block with ``lambda = 0``.
* ``sparse``    -- full GSCN block with the label-free calibrated ``lambda``.

Run::

    python -m tracks.ksvd.code.run_gscn_v0 stage0
    python -m tracks.ksvd.code.run_gscn_v0 smoke
    python -m tracks.ksvd.code.run_gscn_v0 formal --arm sparse
    python -m tracks.ksvd.code.run_gscn_v0 audit --arm sparse
    python -m tracks.ksvd.code.run_gscn_v0 ablate --arm sparse
    python -m tracks.ksvd.code.run_gscn_v0 semantics --arm sparse
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

from tracks.ksvd.code.run_aiom_representation_audit import (  # noqa: E402
    Mol,
    mol_from_pyg,
    permute_mol,
    repo_loader,
)

RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/gscn_v0"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SOUP_DIR = RESULTS_DIR / "soup_states"

PROTOCOL_VERSION = "gscn_v0"

# --- frozen capacity (no sweep) --------------------------------------------
D_MODEL = 64
LAYERS = 4
T_ISTA = 3
K_MULT = 2          # K = 2*d
EPS = 1e-8
POWER_ITERS = 30
CALIB_GRAPHS = 512
LAMBDA_PCTL = 80.0

# --- canonical B-Null/B-Full OPTIMIZED_PROTOCOL (inherited) ----------------
SEED = 0
BATCH = 128
LR = 1.0e-3
WD = 1.0e-5
CLIP = 5.0
MAX_EPOCHS = 240
PATIENCE = 40
TRAIN_SHUFFLE_SEED_OFFSET = 91011
TORCH_THREADS = 4
TOP_K = 5

EVAL_LIMIT = None   # full valid 1000

# --- canonical handcrafted-statistics references (seed 0 fixed Top-5 soup) --
# From records/claims/claim-local-token-null-20260919.yaml (A100, official test
# never loaded).
M_FULL_REFERENCE = 0.119818          # canonical B-Full (84,495 params)
M_BNULL_CANONICAL_REFERENCE = 0.123028  # canonical B-Null (49,343 params)


def _torch():
    import torch

    return torch


def _configure_determinism() -> None:
    torch = _torch()
    torch.set_num_threads(int(TORCH_THREADS))
    torch.use_deterministic_algorithms(False)


def _seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=float) + "\n", encoding="utf-8")


def _git(*args: str):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def provenance(device: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    torch = _torch()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "seed": SEED,
        "d_model": D_MODEL,
        "layers": LAYERS,
        "t_ista": T_ISTA,
        "K": K_MULT * D_MODEL,
        "official_test_loaded": False,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        payload["gpu"] = torch.cuda.get_device_name(0)
        payload["cuda"] = torch.version.cuda
    if extra:
        payload.update(extra)
    return payload


# ===========================================================================
# 0. Data (raw atom/bond category + adjacency only)
# ===========================================================================
def load_mols_and_y(root: Path, split: str, limit: int | None = None):
    if split == "test":
        raise RuntimeError("GSCN-v0 never loads the official test split")
    load_zinc, data_to_graph = repo_loader()
    pyg_split = "val" if split == "valid" else split
    dataset = load_zinc(root, pyg_split)
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    mols: list[Mol] = []
    ys: list[float] = []
    for i in range(n):
        data = dataset[i]
        mols.append(mol_from_pyg(data, data_to_graph))
        ys.append(float(data.y.reshape(-1)[0]))
    return mols, np.asarray(ys, dtype=np.float64)


def category_index(mols: Sequence[Mol]) -> tuple[dict[int, int], dict[int, int]]:
    atom_cats = sorted({int(x) for mol in mols for x in mol.node_types.tolist()})
    bond_cats = sorted({int(x) for mol in mols for x in mol.bond_types.tolist()})
    return ({c: i for i, c in enumerate(atom_cats)},
            {c: i for i, c in enumerate(bond_cats)})


def build_raw_batch(mols: Sequence[Mol], atom_index: dict[int, int],
                    bond_index: dict[int, int], device: str) -> dict[str, Any]:
    torch = _torch()
    atom_all: list[int] = []
    bt_all: list[int] = []
    src: list[int] = []
    dst: list[int] = []
    pool: list[int] = []
    off = 0
    for gi, mol in enumerate(mols):
        atom_all.extend(int(atom_index[int(x)]) for x in mol.node_types.tolist())
        pool.extend([gi] * int(mol.n))
        for i, (a, b) in enumerate(mol.bonds):
            t = int(bond_index[int(mol.bond_types[i])])
            src.extend([off + a, off + b])
            dst.extend([off + b, off + a])
            bt_all.extend([t, t])
        off += int(mol.n)
    return {
        "atom_types": torch.as_tensor(atom_all, dtype=torch.long, device=device),
        "bond_types": torch.as_tensor(bt_all, dtype=torch.long, device=device),
        "edge_index": torch.as_tensor([src, dst], dtype=torch.long, device=device),
        "pool_index": torch.as_tensor(pool, dtype=torch.long, device=device),
        "n_graphs": len(mols),
    }


# ===========================================================================
# 1. Models
# ===========================================================================
def _make_head(nn, d: int):
    return nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))


class RawBaseline:
    """Raw-graph baseline: SiLU edge-aware message passing (repo raw primitive)."""

    @staticmethod
    def build(n_atom: int, n_bond: int, d: int = D_MODEL, layers: int = LAYERS):
        torch = _torch()
        nn = torch.nn

        class _Raw(nn.Module):
            def __init__(self):
                super().__init__()
                self.arm = "bnull_raw"
                self.d = d
                self.layers = layers
                self.E_A = nn.Embedding(n_atom, d)
                self.E_B = nn.Embedding(n_bond, d)
                self.W_self = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.W_msg = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.W_edge = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.head = _make_head(nn, d)

            def forward_layer_states(self, batch):
                torch = _torch()
                h = self.E_A(batch["atom_types"])
                ef = self.E_B(batch["bond_types"])
                src, dst = batch["edge_index"]
                states = [h]
                for li in range(self.layers):
                    msg = self.W_msg[li](h)[src] + self.W_edge[li](ef)
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst, msg)
                    h = torch.nn.functional.silu(self.W_self[li](h) + agg)
                    states.append(h)
                return states

            def forward(self, batch):
                torch = _torch()
                h = self.forward_layer_states(batch)[-1]
                z = torch.zeros(batch["n_graphs"], self.d, device=h.device, dtype=h.dtype)
                z.index_add_(0, batch["pool_index"], h)
                return self.head(z).squeeze(-1)

        return _Raw()


class GenericControl:
    """Param-matched generic nonlinear hidden update (mixing + MLP)."""

    @staticmethod
    def build(n_atom: int, n_bond: int, d: int = D_MODEL, layers: int = LAYERS,
              m: int | None = None):
        torch = _torch()
        nn = torch.nn
        if m is None:
            m = generic_width(d)

        class _Generic(nn.Module):
            def __init__(self):
                super().__init__()
                self.arm = "generic"
                self.d = d
                self.m = m
                self.layers = layers
                self.E_A = nn.Embedding(n_atom, d)
                self.E_B = nn.Embedding(n_bond, d)
                self.W_self = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.W_msg = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.W_edge = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.LN_mix = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
                self.W1 = nn.ModuleList([nn.Linear(d, m) for _ in range(layers)])
                self.W2 = nn.ModuleList([nn.Linear(m, d) for _ in range(layers)])
                self.LN_out = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
                self.head = _make_head(nn, d)

            def forward_layer_states(self, batch):
                torch = _torch()
                h = self.E_A(batch["atom_types"])
                ef = self.E_B(batch["bond_types"])
                src, dst = batch["edge_index"]
                states = [h]
                for li in range(self.layers):
                    msg = self.W_msg[li](h)[src] + self.W_edge[li](ef)
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst, msg)
                    y = self.LN_mix[li](self.W_self[li](h) + agg)
                    y = self.W2[li](torch.nn.functional.silu(self.W1[li](y)))
                    h = self.LN_out[li](y)
                    states.append(h)
                return states

            def forward(self, batch):
                torch = _torch()
                h = self.forward_layer_states(batch)[-1]
                z = torch.zeros(batch["n_graphs"], self.d, device=h.device, dtype=h.dtype)
                z.index_add_(0, batch["pool_index"], h)
                return self.head(z).squeeze(-1)

        return _Generic()


def generic_width(d: int = D_MODEL) -> int:
    """Best integer m with 5d^2+7d == 3d^2+7d + m(2d+1)  ->  m = 2d^2/(2d+1)."""
    target = 2.0 * d * d / (2.0 * d + 1.0)
    cand = {int(math.floor(target)), int(math.ceil(target))}
    best = min(cand, key=lambda x: abs(x * (2 * d + 1) - 2 * d * d))
    return int(best)


class GSCN:
    """Graph relational mixing -> unrolled ISTA sparse dictionary code."""

    @staticmethod
    def build(n_atom: int, n_bond: int, d: int = D_MODEL, layers: int = LAYERS,
              K: int | None = None, arm: str = "sparse"):
        torch = _torch()
        nn = torch.nn
        if K is None:
            K = K_MULT * d

        class _GSCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.arm = arm
                self.d = d
                self.K = K
                self.layers = layers
                self.E_A = nn.Embedding(n_atom, d)
                self.E_B = nn.Embedding(n_bond, d)
                self.W_self = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.W_msg = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.W_edge = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.LN_mix = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
                self.D = nn.ParameterList([nn.Parameter(torch.empty(K, d)) for _ in range(layers)])
                self.LN_out = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
                self.head = _make_head(nn, d)
                lam0 = 0.0 if arm == "nothresh" else 1.0
                self.register_buffer("lambdas", torch.full((layers,), float(lam0)))
                for li in range(layers):
                    nn.init.normal_(self.D[li], std=1.0 / math.sqrt(d))

            # -- sparse coding -------------------------------------------
            def _sigma2(self, DN):
                torch = _torch()
                with torch.no_grad():
                    d = DN.shape[1]
                    v = torch.ones(d, 1, device=DN.device, dtype=DN.dtype) / math.sqrt(d)
                    for _ in range(POWER_ITERS):
                        w = DN.t() @ (DN @ v)
                        v = w / w.norm().clamp_min(EPS)
                    sigma2 = (v.t() @ (DN.t() @ (DN @ v))).clamp_min(0.0).reshape(())
                return sigma2

            def _code(self, Y, layer, capture=None, code_transform=None, meta=None):
                torch = _torch()
                D = self.D[layer]
                DN = D / (D.norm(dim=1, keepdim=True) + EPS)
                C = Y @ DN.t()
                G = DN @ DN.t()
                eta = 0.9 / (self._sigma2(DN) + EPS)
                A = torch.zeros_like(C)
                lam = self.lambdas[layer]
                for _ in range(T_ISTA):
                    A = torch.relu(A + eta * (C - A @ G) - eta * lam)
                if capture is not None:
                    capture["Y"].append(Y)
                    capture["C"].append(C)
                    capture["A"].append(A)
                    capture["DN"].append(DN)
                if code_transform is not None:
                    A = code_transform(A, layer, meta)
                return A, DN

            def forward_layer_states(self, batch, capture=None, code_transform=None):
                torch = _torch()
                h = self.E_A(batch["atom_types"])
                ef = self.E_B(batch["bond_types"])
                src, dst = batch["edge_index"]
                states = [h]
                for li in range(self.layers):
                    msg = self.W_msg[li](h)[src] + self.W_edge[li](ef)
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, dst, msg)
                    y = self.LN_mix[li](self.W_self[li](h) + agg)
                    A, DN = self._code(y, li, capture=capture, code_transform=code_transform,
                                       meta=batch)
                    h = self.LN_out[li](A @ DN)
                    states.append(h)
                return states

            def forward(self, batch, capture=None, code_transform=None):
                torch = _torch()
                h = self.forward_layer_states(batch, capture=capture,
                                              code_transform=code_transform)[-1]
                z = torch.zeros(batch["n_graphs"], self.d, device=h.device, dtype=h.dtype)
                z.index_add_(0, batch["pool_index"], h)
                return self.head(z).squeeze(-1)

        return _GSCN()


def build_model(arm: str, n_atom: int, n_bond: int):
    if arm == "bnull_raw":
        return RawBaseline.build(n_atom, n_bond)
    if arm == "generic":
        return GenericControl.build(n_atom, n_bond)
    if arm in ("nothresh", "sparse"):
        return GSCN.build(n_atom, n_bond, arm=arm)
    raise ValueError(arm)


# ===========================================================================
# 2. Parameter accounting
# ===========================================================================
def param_breakdown(model) -> dict[str, int]:
    buckets: dict[str, int] = defaultdict(int)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        if name.startswith("E_A.") or name.startswith("E_B."):
            key = "input_embedding"
        elif name.startswith("D."):
            key = "dictionary"
        elif name.startswith("head."):
            key = "head"
        else:
            key = "mixing_or_hidden"
        buckets[key] += n
    out = dict(buckets)
    out["total_trainable"] = int(sum(buckets.values()))
    return out


# ===========================================================================
# 3. Lambda calibration (label-free, one-shot)
# ===========================================================================
def calibrate_lambdas(model, calib_mols, atom_index, bond_index, device,
                      log=print) -> list[float]:
    torch = _torch()
    assert model.arm in ("nothresh", "sparse")
    model.eval()
    per_layer: list[list[np.ndarray]] = [[] for _ in range(model.layers)]
    with torch.no_grad():
        for start in range(0, len(calib_mols), BATCH):
            batch = build_raw_batch(calib_mols[start:start + BATCH], atom_index,
                                    bond_index, device)
            cap: dict[str, list] = {"Y": [], "C": [], "A": [], "DN": []}
            model.forward_layer_states(batch, capture=cap)
            for li in range(model.layers):
                per_layer[li].append(cap["C"][li].detach().cpu().numpy().reshape(-1))
    vals: list[float] = []
    for li in range(model.layers):
        c = np.concatenate(per_layer[li])
        pos = c[c > 0]
        lam = float(np.percentile(pos, LAMBDA_PCTL)) if pos.size else 0.0
        vals.append(lam)
    with torch.no_grad():
        model.lambdas.copy_(torch.tensor(vals, dtype=model.lambdas.dtype,
                                         device=model.lambdas.device))
    log(f"[lambda] calibrated={vals}")
    return vals


# ===========================================================================
# 4. Stage 0 -- data-free implementation tests
# ===========================================================================
def _capture_states(model, batch):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        if model.arm in ("nothresh", "sparse"):
            return model.forward_layer_states(batch)
        return model.forward_layer_states(batch)


def stage0_tests(atom_index, bond_index, device, log=print) -> dict[str, Any]:
    torch = _torch()
    load_zinc, data_to_graph = repo_loader()
    dataset = load_zinc(REPO_ROOT / "data/ZINC", "train")
    mol = mol_from_pyg(dataset[0], data_to_graph)
    mol2 = mol_from_pyg(dataset[1], data_to_graph)
    n_atom = max(atom_index.values()) + 1
    n_bond = max(bond_index.values()) + 1
    results: dict[str, Any] = {}

    for arm in ("bnull_raw", "generic", "nothresh", "sparse"):
        _seed_everything(SEED)
        model = build_model(arm, n_atom, n_bond).to(device)
        if arm == "sparse":
            calibrate_lambdas(model, [mol, mol2], atom_index, bond_index, device,
                              log=lambda *_: None)
        # A. permutation equivariance
        perm = np.random.RandomState(0).permutation(mol.n)
        eb = np.random.RandomState(1).permutation(mol.m)
        molp = permute_mol(mol, perm, eb)
        b = build_raw_batch([mol], atom_index, bond_index, device)
        bp = build_raw_batch([molp], atom_index, bond_index, device)
        s_g = _capture_states(model, b)
        s_p = _capture_states(model, bp)
        max_eq = 0.0
        for hg, hp in zip(s_g, s_p):
            hg = hg.detach().cpu().numpy()
            hp = hp.detach().cpu().numpy()
            # hp[v] corresponds to original node perm[v]
            diff = hp[perm] - hg
            max_eq = max(max_eq, float(np.abs(diff).max()))
        pg = float(model(b)[0].detach().cpu())
        with torch.no_grad():
            pp = float(model(bp)[0].detach().cpu())
        # B. batching invariance
        b2 = build_raw_batch([mol, mol2], atom_index, bond_index, device)
        with torch.no_grad():
            out2 = model(b2).detach().cpu()
        batch_inv = float(abs(float(out2[0]) - pg))
        # C. dictionary gradient / D-E-F (GSCN only)
        dict_grad = None
        code_nonneg = None
        dims_ok = None
        bypass_delta = None
        if arm in ("nothresh", "sparse"):
            rb = [mol2, mol_from_pyg(dataset[2], data_to_graph),
                  mol_from_pyg(dataset[3], data_to_graph)]
            bb = build_raw_batch(rb, atom_index, bond_index, device)
            y = torch.as_tensor([0.0, 1.0, 0.5], device=device)
            model.train()
            cap: dict[str, list] = {"Y": [], "C": [], "A": [], "DN": []}
            pred = model(bb, capture=cap)
            loss = torch.nn.functional.l1_loss(pred, y)
            gnorm = []
            for li in range(model.layers):
                g = torch.autograd.grad(loss, model.D[li], retain_graph=(li < model.layers - 1))[0]
                gnorm.append(float(g.norm()))
            dict_grad = gnorm
            code_nonneg = bool(all(float(a.min()) >= -1e-7 for a in cap["A"]))
            dims_ok = bool(
                cap["Y"][0].shape[1] == model.d
                and cap["A"][0].shape[1] == model.K
                and cap["A"][0].shape[0] == cap["Y"][0].shape[0]
            )
            # F. no hidden bypass: perturb D, fixed Y, output must change
            model.eval()
            with torch.no_grad():
                cap0: dict[str, list] = {"Y": [], "C": [], "A": [], "DN": []}
                model(bb, capture=cap0)
                y0 = cap0["A"][0] @ cap0["DN"][0]
                saved = [p.detach().clone() for p in model.D]
                for li in range(model.layers):
                    model.D[li].data.add_(torch.randn_like(model.D[li]) * 0.1)
                cap1: dict[str, list] = {"Y": [], "C": [], "A": [], "DN": []}
                model(bb, capture=cap1)
                y1 = cap1["A"][0] @ cap1["DN"][0]
                bypass_delta = float((y1 - y0).abs().mean())
                for li in range(model.layers):
                    model.D[li].data.copy_(saved[li])
        passed = (max_eq < 1e-5) and (batch_inv < 1e-5) and (abs(pg - pp) < 1e-5)
        results[arm] = {
            "param_permutation_equivariance_max_err": max_eq,
            "graph_prediction_invariance_err": abs(pg - pp),
            "batching_invariance_err": batch_inv,
            "dictionary_grad_norms": dict_grad,
            "code_nonneg": code_nonneg,
            "dims_ok": dims_ok,
            "bypass_perturbation_delta": bypass_delta,
            "passed": bool(passed),
        }
        log(f"[stage0] {arm}: eq_err={max_eq:.2e} batch_err={batch_inv:.2e} "
            f"pred_inv={abs(pg - pp):.2e} passed={passed}")
    results["all_passed"] = all(v["passed"] for k, v in results.items()
                                if isinstance(v, dict))
    return results


# ===========================================================================
# 5. Trainer (canonical epoch protocol + Top-5 soup)
# ===========================================================================
class EpochTrainer:
    def __init__(self, model, train_mols, valid_mols, ytr, yva, atom_index,
                 bond_index, device, arm, log=print):
        torch = _torch()
        self.model = model
        self.train_mols = list(train_mols)
        self.valid_mols = list(valid_mols)
        self.ytr = torch.as_tensor(ytr, dtype=torch.float32, device=device)
        self.yva = torch.as_tensor(yva, dtype=torch.float32, device=device)
        self.atom_index = atom_index
        self.bond_index = bond_index
        self.device = device
        self.arm = arm
        self.log = log
        self.opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        self.gen = torch.Generator().manual_seed(SEED + TRAIN_SHUFFLE_SEED_OFFSET)
        self.history: list[dict[str, Any]] = []
        self.best_valid = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.top5: list[tuple[float, int, dict]] = []
        self.d_grad_history: list[dict[str, float]] = []
        self.t0 = time.time()
        self.peak_gpu_mb = 0.0
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def _batch(self, mols):
        return build_raw_batch(mols, self.atom_index, self.bond_index, self.device)

    def _forward_all(self, mols):
        torch = _torch()
        self.model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(mols), BATCH):
                preds.append(self.model(self._batch(mols[s:s + BATCH])))
        return torch.cat(preds)

    def evaluate(self) -> tuple[float, float]:
        tr = float((self._forward_all(self.train_mols) - self.ytr).abs().mean())
        va = float((self._forward_all(self.valid_mols) - self.yva).abs().mean())
        if self.device.startswith("cuda"):
            torch = _torch()
            self.peak_gpu_mb = max(self.peak_gpu_mb,
                                   float(torch.cuda.max_memory_allocated() / 1e6))
        return tr, va

    def train(self, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE):
        torch = _torch()
        n = len(self.train_mols)
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            order = torch.randperm(n, generator=self.gen).tolist()
            ep_loss = 0.0
            nb = 0
            for s in range(0, n, BATCH):
                idx = order[s:s + BATCH]
                batch = self._batch([self.train_mols[i] for i in idx])
                pred = self.model(batch)
                tgt = self.ytr[torch.as_tensor(idx, dtype=torch.long, device=self.device)]
                loss = torch.nn.functional.l1_loss(pred, tgt)
                self.opt.zero_grad()
                loss.backward()
                if hasattr(self.model, "D"):
                    rec = {}
                    with torch.no_grad():
                        for li in range(self.model.layers):
                            rec[f"D{li}"] = float(self.model.D[li].grad.norm()) \
                                if self.model.D[li].grad is not None else 0.0
                    self.d_grad_history.append(rec)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
                self.opt.step()
                ep_loss += float(loss.detach())
                nb += 1
            tr, va = self.evaluate()
            wall = time.time() - self.t0
            self.history.append({"epoch": epoch, "train_mae": tr, "valid_mae": va,
                                 "train_batch_l1": ep_loss / max(nb, 1), "wall_s": wall})
            if va < self.best_valid - 1e-12:
                self.best_valid = va
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in self.model.state_dict().items()})
            state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in self.model.state_dict().items()})
            self.top5.append((va, epoch, state))
            self.top5.sort(key=lambda x: x[0])
            self.top5 = self.top5[:TOP_K]
            if epoch % 10 == 0 or epoch <= 3:
                self.log(f"[{self.arm}] ep {epoch} train={tr:.4f} valid={va:.4f} "
                         f"best={self.best_valid:.4f}@{self.best_epoch} wall={wall:.0f}s")
            if epoch - self.best_epoch >= patience:
                self.log(f"[{self.arm}] early stop at epoch {epoch} "
                         f"(best {self.best_valid:.4f}@{self.best_epoch})")
                break
        soup = self._soup_state()
        soup_mae = self._eval_state(soup)
        return {
            "history": self.history,
            "best_valid": self.best_valid,
            "best_epoch": self.best_epoch,
            "soup_valid": soup_mae,
            "soup_members": [e for _, e, _ in self.top5],
            "wall_s": self.history[-1]["wall_s"],
            "epochs_run": self.history[-1]["epoch"],
            "peak_gpu_mb": self.peak_gpu_mb,
            "state_best": self.best_state,
            "state_soup": soup,
            "d_grad_history": self.d_grad_history,
        }

    def _soup_state(self):
        torch = _torch()
        if not self.top5:
            return None
        keys = self.top5[0][2].keys()
        out = {}
        for k in keys:
            stacked = torch.stack([self.top5[i][2][k].float() for i in range(len(self.top5))], 0)
            out[k] = stacked.mean(0).to(self.top5[0][2][k].dtype)
        return out

    def _eval_state(self, state):
        if state is None:
            return float("nan")
        backup = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
        self.model.load_state_dict(state)
        va = float((self._forward_all(self.valid_mols) - self.yva).abs().mean())
        self.model.load_state_dict(backup)
        return va


# ===========================================================================
# 6. Code / geometry audits
# ===========================================================================
def collect_codes(model, mols, atom_index, bond_index, device):
    torch = _torch()
    model.eval()
    per_layer_A: list[list[np.ndarray]] = [[] for _ in range(model.layers)]
    with torch.no_grad():
        for s in range(0, len(mols), BATCH):
            batch = build_raw_batch(mols[s:s + BATCH], atom_index, bond_index, device)
            cap: dict[str, list] = {"Y": [], "C": [], "A": [], "DN": []}
            model.forward_layer_states(batch, capture=cap)
            for li in range(model.layers):
                per_layer_A[li].append(cap["A"][li].detach().cpu().numpy())
    return per_layer_A


def code_stats(model, mols, atom_index, bond_index, device) -> dict[str, Any]:
    per_layer_A = collect_codes(model, mols, atom_index, bond_index, device)
    out: dict[str, Any] = {}
    for li, chunks in enumerate(per_layer_A):
        A = np.concatenate(chunks, 0)
        N, K = A.shape
        nz_per_node = (A > 0).sum(1)
        active = float(nz_per_node.sum()) / float(N * K)
        usage = (A > 0).sum(0)
        dead = int((usage == 0).sum())
        frac_atoms_used = float((usage > 0).mean())
        total_nz = float((A > 0).sum())
        max_atom_share = float(usage.max()) / max(total_nz, 1.0)
        out[f"layer{li}"] = {
            "active_fraction": active,
            "median_active_per_node": float(np.median(nz_per_node)),
            "p90_active_per_node": float(np.percentile(nz_per_node, 90)),
            "atoms_used": int((usage > 0).sum()),
            "atoms_total": int(K),
            "frac_atoms_used": frac_atoms_used,
            "dead_atoms": dead,
            "max_single_atom_share": max_atom_share,
            "n_nodes": int(N),
        }
    return out


def dict_geometry(model, D_init: list[np.ndarray] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for li in range(model.layers):
        D = model.D[li].detach().cpu().numpy()
        DN = D / (np.linalg.norm(D, axis=1, keepdims=True) + EPS)
        G = DN @ DN.T
        off = G - np.eye(G.shape[0])
        iu = np.triu_indices(G.shape[0], k=1)
        cos = off[iu]
        s = np.linalg.svd(DN, compute_uv=False)
        eff_rank = float((s ** 2).sum() ** 2 / max((s ** 4).sum(), 1e-12))
        rec = {
            "effective_rank": eff_rank,
            "coherence": float(np.abs(cos).max()) if cos.size else 0.0,
            "mean_abs_pairwise_cosine": float(np.abs(cos).mean()) if cos.size else 0.0,
            "mean_pairwise_cosine": float(cos.mean()) if cos.size else 0.0,
            "norm_min": float(np.linalg.norm(D, axis=1).min()),
            "norm_max": float(np.linalg.norm(D, axis=1).max()),
            "norm_mean": float(np.linalg.norm(D, axis=1).mean()),
        }
        if D_init is not None:
            rec["displacement_fro"] = float(np.linalg.norm(D - D_init[li]))
        out[f"layer{li}"] = rec
    return out


# ===========================================================================
# 7. Inference ablations
# ===========================================================================
def _valid_mae(model, valid_mols, yva, atom_index, bond_index, device,
               code_transform=None):
    torch = _torch()
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(valid_mols), BATCH):
            batch = build_raw_batch(valid_mols[s:s + BATCH], atom_index, bond_index, device)
            preds.append(model(batch, code_transform=code_transform))
    y = torch.as_tensor(yva, dtype=torch.float32, device=device)
    return float((torch.cat(preds) - y).abs().mean())


def run_ablations(model, valid_mols, yva, atom_index, bond_index, device,
                  log=print) -> dict[str, Any]:
    torch = _torch()
    base = _valid_mae(model, valid_mols, yva, atom_index, bond_index, device)
    res: dict[str, Any] = {"intact_valid_mae": base}

    # A. atom-row permutation (sanity; prediction invariant)
    perm = torch.randperm(model.K)
    D_backup = [p.detach().clone() for p in model.D]
    with torch.no_grad():
        for li in range(model.layers):
            model.D[li].data.copy_(D_backup[li][perm])
    res["atom_permutation_valid_mae"] = _valid_mae(model, valid_mols, yva, atom_index,
                                                   bond_index, device)
    with torch.no_grad():
        for li in range(model.layers):
            model.D[li].data.copy_(D_backup[li])

    # B. random-normalised dictionary destruction
    _seed_everything(SEED + 777)
    with torch.no_grad():
        for li in range(model.layers):
            r = torch.randn_like(model.D[li])
            model.D[li].data.copy_(r / (r.norm(dim=1, keepdim=True) + EPS))
    res["random_dictionary_valid_mae"] = _valid_mae(model, valid_mols, yva, atom_index,
                                                    bond_index, device)
    with torch.no_grad():
        for li in range(model.layers):
            model.D[li].data.copy_(D_backup[li])

    # C. within-batch code shuffle
    gen = torch.Generator().manual_seed(SEED + 1)

    def shuffle_fn(A, layer, meta):
        idx = torch.randperm(A.shape[0], generator=gen, device=A.device)
        return A[idx]

    res["code_shuffle_valid_mae"] = _valid_mae(model, valid_mols, yva, atom_index,
                                               bond_index, device,
                                               code_transform=shuffle_fn)

    # D. layer-wide mean-code replacement
    def mean_fn(A, layer, meta):
        return A.mean(0, keepdim=True).expand_as(A).contiguous()

    res["mean_code_valid_mae"] = _valid_mae(model, valid_mols, yva, atom_index,
                                            bond_index, device,
                                            code_transform=mean_fn)
    for k in ("atom_permutation", "random_dictionary", "code_shuffle", "mean_code"):
        res[f"delta_{k}"] = res[f"{k}_valid_mae"] - base
    log(f"[ablate] {json.dumps({k: v for k, v in res.items() if isinstance(v, float)})}")
    return res


# ===========================================================================
# 8. Post-hoc structural semantics (explanation only)
# ===========================================================================
def _wl_keys(mol: Mol, rounds: int) -> dict[int, int]:
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, (a, b) in enumerate(mol.bonds):
        t = int(mol.bond_types[i])
        adj[a].append((int(b), t))
        adj[b].append((int(a), t))
    color = {v: int(mol.node_types[v]) for v in range(mol.n)}
    for _ in range(rounds):
        new = {}
        for v in range(mol.n):
            sig = tuple(sorted((t, color[u]) for (u, t) in adj[v]))
            new[v] = hash((color[v], sig)) & 0xFFFFFFFF
        color = new
    return color


def structural_semantics(model, mols, atom_index, bond_index, device,
                         top_n: int = 100, log=print) -> dict[str, Any]:
    torch = _torch()
    model.eval()
    per_layer_A: list[list[np.ndarray]] = [[] for _ in range(model.layers)]
    for s in range(0, len(mols), BATCH):
        batch = build_raw_batch(mols[s:s + BATCH], atom_index, bond_index, device)
        cap: dict[str, list] = {"Y": [], "C": [], "A": [], "DN": []}
        with torch.no_grad():
            model.forward_layer_states(batch, capture=cap)
        for li in range(model.layers):
            per_layer_A[li].append(cap["A"][li].detach().cpu().numpy())
    # per-node global index -> (mol_idx, node_idx)
    node_map = []
    for gi, mol in enumerate(mols):
        for v in range(mol.n):
            node_map.append((gi, v))
    moles = mols
    out: dict[str, Any] = {}
    for li, chunks in enumerate(per_layer_A):
        A = np.concatenate(chunks, 0)
        N, K = A.shape
        keys_r1 = {}
        keys_r2 = {}
        layer_rec = {"atoms": []}
        for k in range(K):
            col = A[:, k]
            if not np.any(col > 0):
                continue
            order = np.argsort(-col)[:min(top_n, N)]
            r1, r2 = [], []
            for node in order:
                gi, v = node_map[int(node)]
                if gi not in keys_r1:
                    keys_r1[gi] = _wl_keys(moles[gi], 1)
                    keys_r2[gi] = _wl_keys(moles[gi], 2)
                r1.append(keys_r1[gi][v])
                r2.append(keys_r2[gi][v])
            rec = {"atom": int(k), "n_top": int(len(order))}
            for tag, vals in (("r1", r1), ("r2", r2)):
                uniq, counts = np.unique(np.asarray(vals), return_counts=True)
                frac = counts / counts.sum()
                entropy = float(-(frac * np.log(frac + 1e-12)).sum())
                rec[f"{tag}_top_fraction"] = float(frac.max())
                rec[f"{tag}_entropy"] = entropy
                rec[f"{tag}_n_unique"] = int(uniq.size)
            layer_rec["atoms"].append(rec)
        # random baseline over the same eval set
        rand_keys_r1 = []
        rand_keys_r2 = []
        for gi, v in node_map:
            if gi not in keys_r1:
                keys_r1[gi] = _wl_keys(moles[gi], 1)
                keys_r2[gi] = _wl_keys(moles[gi], 2)
        rand_keys_r1 = [keys_r1[gi][v] for (gi, v) in node_map]
        rand_keys_r2 = [keys_r2[gi][v] for (gi, v) in node_map]
        base_rec = {}
        for tag, vals in (("r1", rand_keys_r1), ("r2", rand_keys_r2)):
            uniq, counts = np.unique(np.asarray(vals), return_counts=True)
            frac = counts / counts.sum()
            base_rec[f"{tag}_random_top_fraction"] = float(frac.max())
        atom_fracs = [a.get("r2_top_fraction", 0.0) for a in layer_rec["atoms"]]
        layer_rec.update(base_rec)
        layer_rec["n_atoms_scored"] = len(atom_fracs)
        layer_rec["mean_r2_top_fraction"] = float(np.mean(atom_fracs)) if atom_fracs else 0.0
        out[f"layer{li}"] = layer_rec
    log(f"[semantics] layers={len(out)}")
    return out


# ===========================================================================
# 9. Orchestration
# ===========================================================================
def _load_splits(data_root: Path, limit: int | None):
    train_mols, y_train = load_mols_and_y(data_root, "train", limit)
    valid_mols, y_valid = load_mols_and_y(data_root, "valid", limit if limit else EVAL_LIMIT)
    return train_mols, valid_mols, y_train, y_valid


def _atom_bond_index(train_mols, valid_mols):
    atom_index, bond_index = category_index(list(train_mols) + list(valid_mols))
    return atom_index, bond_index


def run_stage0(args, log) -> int:
    train_mols, valid_mols, _, _ = _load_splits(args.data_root, args.limit)
    atom_index, bond_index = _atom_bond_index(train_mols, valid_mols)
    res = stage0_tests(atom_index, bond_index, args.device, log=log)
    write_json(RESULTS_DIR / "stage0_tests.json",
               {"provenance": provenance(args.device), "tests": res})
    log(f"[stage0] all_passed={res['all_passed']}")
    return 0 if res["all_passed"] else 1


def run_smoke(args, log) -> int:
    _seed_everything(SEED)
    train_mols, valid_mols, y_train, y_valid = _load_splits(args.data_root, args.limit)
    atom_index, bond_index = _atom_bond_index(train_mols, valid_mols)
    n_atom = max(atom_index.values()) + 1
    n_bond = max(bond_index.values()) + 1
    model = GSCN.build(n_atom, n_bond, arm="sparse").to(args.device)
    calib = train_mols[:CALIB_GRAPHS]
    lam = calibrate_lambdas(model, calib, atom_index, bond_index, args.device, log=log)
    trainer = EpochTrainer(model, train_mols, valid_mols, y_train, y_valid,
                           atom_index, bond_index, args.device, "sparse_smoke", log=log)
    out = trainer.train(max_epochs=args.smoke_epochs, patience=999)
    model.load_state_dict(out["state_best"])
    cs = code_stats(model, train_mols[:2000], atom_index, bond_index, args.device)
    active = [v["active_fraction"] for v in cs.values()]
    used = [v["frac_atoms_used"] for v in cs.values()]
    dom = [v["max_single_atom_share"] for v in cs.values()]
    geo = dict_geometry(model)
    eff = [v["effective_rank"] for v in geo.values()]
    pooled_used = (sum(v["atoms_used"] for v in cs.values())
                   / max(sum(v["atoms_total"] for v in cs.values()), 1))
    grads = out["d_grad_history"]
    nonzero_grad = bool(grads) and all(any(v > 0 for v in h.values()) for h in grads[:50])
    fin = bool(np.isfinite(out["best_valid"]) and np.isfinite(out["soup_valid"]))
    losedec = bool(out["history"][-1]["train_batch_l1"]
                   < out["history"][0]["train_batch_l1"])
    gate = {
        "code_activity_all_layers_in_range": bool(all(0.02 < a < 0.50 for a in active)),
        "pooled_atoms_used_ge_50pct": bool(pooled_used >= 0.50),
        "no_atom_domination": bool(all(d <= 0.50 for d in dom)),
        "effective_rank_gt_1p5": bool(all(e > 1.5 for e in eff)),
        "loss_decreased": losedec,
        "finite": fin,
        "nonzero_D_grad": nonzero_grad,
    }
    gate["passed"] = all(bool(v) for k, v in gate.items() if k != "passed")
    payload = {
        "provenance": provenance(args.device),
        "lambdas": lam,
        "smoke_epochs": args.smoke_epochs,
        "best_valid": out["best_valid"],
        "soup_valid": out["soup_valid"],
        "history": out["history"],
        "code_stats": cs,
        "dict_geometry": geo,
        "pooled_atoms_used": pooled_used,
        "gate": gate,
    }
    write_json(RESULTS_DIR / "smoke_gate.json", payload)
    log(f"[smoke] best_valid={out['best_valid']:.4f} active={active} used={used} "
        f"pooled_used={pooled_used:.3f} dom={dom} effrank={eff} "
        f"gate_passed={gate['passed']}")
    return 0 if gate["passed"] else 1


def run_formal(args, log) -> int:
    torch = _torch()
    _seed_everything(SEED)
    train_mols, valid_mols, y_train, y_valid = _load_splits(args.data_root, args.limit)
    atom_index, bond_index = _atom_bond_index(train_mols, valid_mols)
    n_atom = max(atom_index.values()) + 1
    n_bond = max(bond_index.values()) + 1
    arm = args.arm
    model = build_model(arm, n_atom, n_bond).to(args.device)
    params = param_breakdown(model)
    D_init = None
    lambdas = None
    if arm in ("nothresh", "sparse"):
        calib = train_mols[:CALIB_GRAPHS]
        lambdas = calibrate_lambdas(model, calib, atom_index, bond_index,
                                    args.device, log=log) if arm == "sparse" else [0.0] * LAYERS
        if arm == "nothresh":
            with torch.no_grad():
                model.lambdas.zero_()
        D_init = [p.detach().cpu().numpy().copy() for p in model.D]
    log(f"[formal:{arm}] params={params['total_trainable']}")
    trainer = EpochTrainer(model, train_mols, valid_mols, y_train, y_valid,
                           atom_index, bond_index, args.device, arm, log=log)
    out = trainer.train(max_epochs=args.max_epochs, patience=args.patience)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CURVE_DIR / f"history_{arm}.json", out["history"])
    torch.save(out["state_best"], STATE_DIR / f"{arm}_best.pt")
    torch.save(out["state_soup"], SOUP_DIR / f"{arm}_top5_soup.pt")

    extra: dict[str, Any] = {"arm": arm, "params": params}
    if lambdas is not None:
        extra["lambdas"] = lambdas
    summary = {
        "provenance": provenance(args.device, extra),
        "arm": arm,
        "params": params,
        "lambdas": lambdas,
        "best_valid": out["best_valid"],
        "best_epoch": out["best_epoch"],
        "soup_valid": out["soup_valid"],
        "soup_members": out["soup_members"],
        "epochs_run": out["epochs_run"],
        "wall_s": out["wall_s"],
        "epochs_per_sec": out["epochs_run"] / max(out["wall_s"], 1e-9),
        "peak_gpu_mb": out["peak_gpu_mb"],
    }
    if arm in ("nothresh", "sparse"):
        model.load_state_dict(out["state_best"])
        summary["code_stats_train"] = code_stats(model, train_mols[:2000], atom_index,
                                                 bond_index, args.device)
        summary["code_stats_valid"] = code_stats(model, valid_mols, atom_index,
                                                 bond_index, args.device)
        summary["dict_geometry"] = dict_geometry(model, D_init)
        grads = out["d_grad_history"]
        if grads:
            summary["task_grad_D"] = {
                k: {"first50_mean": float(np.mean([h[k] for h in grads[:50]])),
                    "last50_mean": float(np.mean([h[k] for h in grads[-50:]]))}
                for k in grads[0]
            }
    write_json(RESULTS_DIR / f"summary_{arm}.json", summary)
    log(f"[formal:{arm}] best_valid={out['best_valid']:.4f} soup={out['soup_valid']:.4f} "
        f"epoch={out['best_epoch']} wall={out['wall_s']:.0f}s")
    return 0


def _load_state(path: Path):
    torch = _torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_audit(args, log) -> int:
    train_mols, valid_mols, y_train, y_valid = _load_splits(args.data_root, args.limit)
    atom_index, bond_index = _atom_bond_index(train_mols, valid_mols)
    n_atom = max(atom_index.values()) + 1
    n_bond = max(bond_index.values()) + 1
    arm = args.arm
    model = build_model(arm, n_atom, n_bond).to(args.device)
    model.load_state_dict(_load_state(STATE_DIR / f"{arm}_best.pt"))
    payload = {
        "provenance": provenance(args.device, {"arm": arm}),
        "code_stats_train": code_stats(model, train_mols[:3000], atom_index,
                                       bond_index, args.device),
        "code_stats_valid": code_stats(model, valid_mols, atom_index, bond_index, args.device),
        "dict_geometry": dict_geometry(model),
        "valid_mae_best": _valid_mae(model, valid_mols, y_valid, atom_index,
                                     bond_index, args.device),
    }
    write_json(RESULTS_DIR / f"audit_{arm}.json", payload)
    log(f"[audit:{arm}] done")
    return 0


def run_ablate(args, log) -> int:
    train_mols, valid_mols, y_train, y_valid = _load_splits(args.data_root, args.limit)
    atom_index, bond_index = _atom_bond_index(train_mols, valid_mols)
    n_atom = max(atom_index.values()) + 1
    n_bond = max(bond_index.values()) + 1
    arm = args.arm
    model = build_model(arm, n_atom, n_bond).to(args.device)
    model.load_state_dict(_load_state(STATE_DIR / f"{arm}_best.pt"))
    res = run_ablations(model, valid_mols, y_valid, atom_index, bond_index,
                        args.device, log=log)
    write_json(RESULTS_DIR / f"ablation_{arm}.json",
               {"provenance": provenance(args.device, {"arm": arm}), **res})
    return 0


def run_semantics(args, log) -> int:
    train_mols, valid_mols, y_train, y_valid = _load_splits(args.data_root, args.limit)
    atom_index, bond_index = _atom_bond_index(train_mols, valid_mols)
    n_atom = max(atom_index.values()) + 1
    n_bond = max(bond_index.values()) + 1
    arm = args.arm
    model = build_model(arm, n_atom, n_bond).to(args.device)
    model.load_state_dict(_load_state(STATE_DIR / f"{arm}_best.pt"))
    res = structural_semantics(model, valid_mols[:500], atom_index, bond_index,
                               args.device, log=log)
    write_json(RESULTS_DIR / f"semantics_{arm}.json",
               {"provenance": provenance(args.device, {"arm": arm}), "layers": res})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GSCN-v0")
    parser.add_argument("stage", choices=["stage0", "smoke", "formal", "audit",
                                          "ablate", "semantics", "params"])
    parser.add_argument("--arm", default="sparse",
                        choices=["bnull_raw", "generic", "nothresh", "sparse"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--smoke-epochs", type=int, default=5)
    args = parser.parse_args(argv)

    _configure_determinism()

    def log(msg: str, *a):
        print(msg, *a, flush=True)

    args.device = args.device
    if args.stage == "params":
        train_mols, valid_mols, _, _ = _load_splits(args.data_root, args.limit)
        atom_index, bond_index = _atom_bond_index(train_mols, valid_mols)
        n_atom = max(atom_index.values()) + 1
        n_bond = max(bond_index.values()) + 1
        out = {"generic_width": generic_width(D_MODEL)}
        for arm in ("bnull_raw", "generic", "nothresh", "sparse"):
            m = build_model(arm, n_atom, n_bond)
            out[arm] = param_breakdown(m)
        write_json(RESULTS_DIR / "params.json", out)
        log(json.dumps(out, indent=2))
        return 0
    if args.stage == "stage0":
        return run_stage0(args, log)
    if args.stage == "smoke":
        return run_smoke(args, log)
    if args.stage == "formal":
        return run_formal(args, log)
    if args.stage == "audit":
        return run_audit(args, log)
    if args.stage == "ablate":
        return run_ablate(args, log)
    if args.stage == "semantics":
        return run_semantics(args, log)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
