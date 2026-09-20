#!/usr/bin/env python
"""PSD-v0 -- Pair-Structural representation + Function-Preserving Dictionary.

Pre-registration: ``tracks/ksvd/notes/zinc_pair_dict_v0_preregistration.md``
(frozen before any run).  Protocol id ``zinc-pair-struct-dict-v0``.  Study
``zinc-context-gap``.  Canonical PyG ZINC ``subset=True`` official **train
10000 / valid 1000**; the **official test is never loaded, instantiated or
referenced** (``test_access`` is enforced by the runner).

Stages
------
* ``stage0``   -- data-free correctness of the pair encoder (targeted checks).
* ``synth``    -- mechanism sanity (C6 vs 2*C3 2-FWL distinguishability).
* ``overfit``  -- 128-graph train-only overfit sanity (Q1 Stage 1).
* ``screen``   -- train-only structural screen: pair vs raw vs raw_wide (Q1 Stage 2).
* ``params``   -- parameter accounting for all arms.
* ``preserve`` -- function-preserving dictionary insertion check (Q2 gate).
* ``dictrun``  -- matched-control dictionary run from a frozen pair checkpoint (Q2).
* ``dictaudit``-- dictionary mechanism / ablation audit (Q2).

Run examples::

    python -m tracks.ksvd.code.run_zinc_pair_dict_v0 stage0
    python -m tracks.ksvd.code.run_zinc_pair_dict_v0 overfit --arm pair --device cuda
    python -m tracks.ksvd.code.run_zinc_pair_dict_v0 screen --arm pair --device cuda
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
from pathlib import Path
from typing import Any, Callable, Sequence

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
from tracks.ksvd.code.run_gscn_v0 import (  # noqa: E402
    RawBaseline,
    build_raw_batch,
    category_index,
    param_breakdown as _param_breakdown,
)

RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/zinc_pair_dict_v0"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SOUP_DIR = RESULTS_DIR / "soup_states"

PROTOCOL_VERSION = "zinc-pair-struct-dict-v0"

# --- frozen capacity (no sweep) --------------------------------------------
D_MODEL = 64
PAIR_LAYERS = 3
RAW_D = 64
RAW_LAYERS = 4
RAW_WIDE_D = 76
RAW_WIDE_LAYERS = 4
EPS = 1e-8
POWER_ITERS = 30

# --- canonical OPTIMIZED_PROTOCOL (inherited from GSCN-v0) -----------------
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

# --- frozen round constants -------------------------------------------------
OVERFIT_N = 128
OVERFIT_EPOCHS = 60
OVERFIT_BATCH = 64
SCREEN_DEV_N = 2048
SCREEN_MON_N = 512
SCREEN_EPOCHS = 60
SPLIT_SEED = 20260922

LAMBDA_ACTIVE_TARGET = 0.25      # fraction of non-zero coefficients
ISTA_T = 6

NULL_REL = 0                      # non-bond off-diagonal


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
        "pair_layers": PAIR_LAYERS,
        "raw_d": RAW_D,
        "raw_wide_d": RAW_WIDE_D,
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
        raise RuntimeError("PSD-v0 never loads the official test split")
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


def build_pair_batch(mols: Sequence[Mol], atom_index: dict[int, int],
                     bond_index: dict[int, int], device: str) -> dict[str, Any]:
    """Dense padded pair-state batch from raw atom/bond category + adjacency."""
    torch = _torch()
    b = len(mols)
    n = max(m.n for m in mols)
    self_rel = max(bond_index.values()) + 2 if bond_index else 1
    atom_ids = np.zeros((b, n), dtype=np.int64)
    pair_rel = np.zeros((b, n, n), dtype=np.int64)
    node_mask = np.zeros((b, n), dtype=bool)
    for gi, mol in enumerate(mols):
        k = int(mol.n)
        node_mask[gi, :k] = True
        for v in range(k):
            atom_ids[gi, v] = int(atom_index[int(mol.node_types[v])])
            pair_rel[gi, v, v] = self_rel
        for i, (a, bb) in enumerate(mol.bonds):
            t = int(bond_index[int(mol.bond_types[i])]) + 1
            pair_rel[gi, a, bb] = t
            pair_rel[gi, bb, a] = t
    return {
        "atom_ids": torch.as_tensor(atom_ids, dtype=torch.long, device=device),
        "pair_rel": torch.as_tensor(pair_rel, dtype=torch.long, device=device),
        "node_mask": torch.as_tensor(node_mask, dtype=torch.bool, device=device),
        "n_graphs": len(mols),
        "max_n": int(n),
    }


def _make_head(nn, d: int):
    return nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))


def _soup_state(top5):
    torch = _torch()
    if not top5:
        return None
    keys = top5[0]["state"].keys()
    out = {}
    for k in keys:
        stacked = torch.stack([top5[i]["state"][k].float() for i in range(len(top5))], 0)
        out[k] = stacked.mean(0).to(top5[0]["state"][k].dtype)
    return out


# ===========================================================================
# 1. Models
# ===========================================================================
class PairEncoder:
    """Pair-state relation-preserving structural encoder (raw input only)."""

    @staticmethod
    def build(n_atom: int, n_bond: int, d: int = D_MODEL, layers: int = PAIR_LAYERS):
        torch = _torch()
        nn = torch.nn
        f = torch.nn.functional
        n_rel = n_bond + 2
        self_rel = n_bond + 1

        class _Pair(nn.Module):
            def __init__(self):
                super().__init__()
                self.arm = "pair"
                self.d = d
                self.layers = layers
                self.n_rel = n_rel
                self.self_rel = self_rel
                self.E_A = nn.Embedding(n_atom, d)
                self.E_R = nn.Embedding(n_rel, d)
                self.W_init = nn.Linear(3 * d, d)
                self.phi = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.psi = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
                self.eta = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
                self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
                self.W_read = nn.Linear(2 * d, d)
                self.head = _make_head(nn, d)

            def forward_layer_states(self, batch):
                atom = batch["atom_ids"]
                rel = batch["pair_rel"]
                nmask = batch["node_mask"]
                b, n = atom.shape
                a = self.E_A(atom)
                r = self.E_R(rel)
                ai = a.unsqueeze(2).expand(b, n, n, d)
                aj = a.unsqueeze(1).expand(b, n, n, d)
                h = self.W_init(torch.cat([ai, aj, r], dim=-1))
                pmask = nmask.unsqueeze(2) & nmask.unsqueeze(1)
                h = h * pmask.unsqueeze(-1)
                states = [h]
                for li in range(layers):
                    phi = self.phi[li](h) * nmask[:, None, :, None]
                    psi = self.psi[li](h) * nmask[:, :, None, None]
                    phi2 = phi.permute(0, 3, 1, 2).reshape(b * d, n, n)
                    psi2 = psi.permute(0, 3, 1, 2).reshape(b * d, n, n)
                    m = torch.bmm(phi2, psi2).view(b, d, n, n).permute(0, 2, 3, 1)
                    upd = f.silu(self.eta[li](torch.cat([h, m], dim=-1)))
                    h = self.ln[li](h + upd)
                    h = h * pmask.unsqueeze(-1)
                    states.append(h)
                return states

            def encode(self, batch):
                h = self.forward_layer_states(batch)[-1]
                nmask = batch["node_mask"]
                pmask = nmask.unsqueeze(2) & nmask.unsqueeze(1)
                hsum = (h * pmask.unsqueeze(-1)).sum(dim=(1, 2))
                denom = pmask.sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(-1)
                mp = hsum / denom
                diag = torch.diagonal(h, dim1=1, dim2=2).transpose(1, 2)  # (B, N, d)
                dmask = nmask.unsqueeze(-1).to(h.dtype)
                dmean = (diag * dmask).sum(dim=1) / dmask.sum(dim=1).clamp_min(1.0)
                return f.silu(self.W_read(torch.cat([mp, dmean], dim=-1)))

            def forward(self, batch):
                return self.head(self.encode(batch)).squeeze(-1)

        return _Pair()


def build_raw(n_atom: int, n_bond: int, d: int = RAW_D, layers: int = RAW_LAYERS):
    return RawBaseline.build(n_atom, n_bond, d=d, layers=layers)


def build_model(arm: str, n_atom: int, n_bond: int):
    if arm == "pair":
        return PairEncoder.build(n_atom, n_bond)
    if arm == "raw":
        return build_raw(n_atom, n_bond)
    if arm == "raw_wide":
        return build_raw(n_atom, n_bond, d=RAW_WIDE_D, layers=RAW_WIDE_LAYERS)
    raise ValueError(arm)


def batch_fn_for(arm: str) -> Callable:
    if arm == "pair":
        return build_pair_batch
    return build_raw_batch


# ===========================================================================
# 2. Sparse dictionary / matched controls (Q2, function-preserving)
# ===========================================================================
def _sigma2(DN):
    torch = _torch()
    with torch.no_grad():
        d = DN.shape[1]
        v = torch.ones(d, 1, device=DN.device, dtype=DN.dtype) / math.sqrt(d)
        for _ in range(POWER_ITERS):
            w = DN.t() @ (DN @ v)
            v = w / w.norm().clamp_min(EPS)
        return (v.t() @ (DN.t() @ (DN @ v))).clamp_min(0.0).reshape(())


def _soft(x, t):
    torch = _torch()
    return torch.sign(x) * torch.relu(x.abs() - t)


class GraphCodeModel:
    """Warm-started pair encoder + head with a pluggable coding bottleneck.

    ``kind`` in {``dense``, ``learned_dict``, ``fixed_identity``, ``learned_nosp``,
    ``generic_topk``}.  At ``lambda=0`` / ``k=d`` / ``D=I`` the coding is the
    identity, so the model is function-preserving at insertion.
    """

    @staticmethod
    def build(encoder, head, d: int, kind: str, K: int | None = None, k_active: int | None = None):
        torch = _torch()
        nn = torch.nn
        K = d if K is None else K

        class _Coded(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.head = head
                self.kind = kind
                self.d = d
                self.K = K
                self.k_active = d if k_active is None else int(k_active)
                if kind in ("learned_dict", "learned_nosp"):
                    eye = torch.eye(K, d)
                    self.D = nn.Parameter(eye.clone())
                elif kind == "fixed_identity":
                    self.register_buffer("D", torch.eye(K, d))
                if kind == "generic_topk":
                    self.W = nn.Parameter(torch.eye(d))

            def encode_u(self, batch):
                return self.encoder.encode(batch)

            def code(self, u, lam):
                if self.kind in ("dense",):
                    return u
                if self.kind == "generic_topk":
                    c = u @ self.W
                    k = max(1, min(self.d, int(self.k_active)))
                    if k >= self.d:
                        ck = c
                    else:
                        kth = c.abs().kthvalue(self.d - k + 1, dim=-1, keepdim=True).values
                        ck = c * (c.abs() >= kth).to(c.dtype)
                    return ck @ self.W.t()
                D = self.D
                DN = D / (D.norm(dim=1, keepdim=True) + EPS)
                C = u @ DN.t()
                G = DN @ DN.t()
                eta = 0.9 / (_sigma2(DN) + EPS)
                A = C
                for _ in range(ISTA_T):
                    A = _soft(A + eta * (C - A @ G), eta * lam)
                return A @ DN

            def forward(self, batch, lam: float = 0.0):
                u = self.encode_u(batch)
                z = self.code(u, lam)
                return self.head(z).squeeze(-1)

            def sparse_stats(self, batch, lam: float = 0.0):
                torch = _torch()
                with torch.no_grad():
                    u = self.encode_u(batch)
                    if self.kind in ("learned_dict", "fixed_identity", "learned_nosp"):
                        D = self.D
                        DN = D / (D.norm(dim=1, keepdim=True) + EPS)
                        C = u @ DN.t()
                        G = DN @ DN.t()
                        eta = 0.9 / (_sigma2(DN) + EPS)
                        A = C
                        for _ in range(ISTA_T):
                            A = _soft(A + eta * (C - A @ G), eta * lam)
                    elif self.kind == "generic_topk":
                        c = u @ self.W
                        k = max(1, min(self.d, int(self.k_active)))
                        if k >= self.d:
                            A = c
                        else:
                            kth = c.abs().kthvalue(self.d - k + 1, dim=-1,
                                                   keepdim=True).values
                            A = c * (c.abs() >= kth).to(c.dtype)
                    else:
                        A = u
                    nz = float((A.abs() > 1e-9).float().mean())
                    return {"active_fraction": nz,
                            "mean_abs_code": float(A.abs().mean()),
                            "codes": A}

        return _Coded()


# ===========================================================================
# 3. Trainer (canonical epoch protocol + Top-5 soup on the monitored split)
# ===========================================================================
class EpochTrainer:
    def __init__(self, model, train_mols, monitor_mols, ytr, ymon, atom_index,
                 bond_index, device, arm, batch_fn, log=print, eval_monitor=True,
                 lam: float = 0.0):
        torch = _torch()
        self.model = model
        self.train_mols = list(train_mols)
        self.monitor_mols = list(monitor_mols) if monitor_mols else []
        self.eval_monitor = bool(eval_monitor and self.monitor_mols)
        self.ytr = torch.as_tensor(ytr, dtype=torch.float32, device=device)
        self.ymon = torch.as_tensor(ymon, dtype=torch.float32, device=device) \
            if ymon is not None else None
        self.atom_index = atom_index
        self.bond_index = bond_index
        self.device = device
        self.arm = arm
        self.batch_fn = batch_fn
        self.log = log
        self.lam = float(lam)
        self.opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        self.gen = torch.Generator().manual_seed(SEED + TRAIN_SHUFFLE_SEED_OFFSET)
        self.history: list[dict[str, Any]] = []
        self.best_score = float("inf")
        self.best_monitor = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.top5: list[dict] = []
        self.t0 = time.time()
        self.peak_gpu_mb = 0.0
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def _batch(self, mols):
        return self.batch_fn(mols, self.atom_index, self.bond_index, self.device)

    def _fwd(self, batch):
        if hasattr(self.model, "kind"):
            return self.model(batch, lam=self.lam)
        return self.model(batch)

    def _forward_all(self, mols, lam: float | None = None):
        torch = _torch()
        self.model.eval()
        prev = self.lam
        if lam is not None:
            self.lam = lam
        preds = []
        with torch.no_grad():
            for s in range(0, len(mols), BATCH):
                preds.append(self._fwd(self._batch(mols[s:s + BATCH])))
        self.lam = prev
        return torch.cat(preds)

    def evaluate(self):
        tr = float((self._forward_all(self.train_mols) - self.ytr).abs().mean())
        if self.eval_monitor:
            mo = float((self._forward_all(self.monitor_mols) - self.ymon).abs().mean())
        else:
            mo = float("nan")
        if self.device.startswith("cuda"):
            torch = _torch()
            self.peak_gpu_mb = max(self.peak_gpu_mb,
                                   float(torch.cuda.max_memory_allocated() / 1e6))
        return tr, mo

    def train(self, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
              lam_schedule: Callable[[int], float] | None = None,
              k_schedule: Callable[[int], int] | None = None):
        torch = _torch()
        n = len(self.train_mols)
        for epoch in range(1, max_epochs + 1):
            if lam_schedule is not None and hasattr(self.model, "kind"):
                self.lam = float(lam_schedule(epoch))
            if k_schedule is not None and getattr(self.model, "kind", None) == "generic_topk":
                self.model.k_active = int(k_schedule(epoch))
            self.model.train()
            order = torch.randperm(n, generator=self.gen).tolist()
            ep_loss = 0.0
            nb = 0
            for s in range(0, n, BATCH):
                idx = order[s:s + BATCH]
                batch = self._batch([self.train_mols[i] for i in idx])
                pred = self._fwd(batch)
                tgt = self.ytr[torch.as_tensor(idx, dtype=torch.long, device=self.device)]
                loss = torch.nn.functional.l1_loss(pred, tgt)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
                self.opt.step()
                ep_loss += float(loss.detach())
                nb += 1
            tr, mo = self.evaluate()
            wall = time.time() - self.t0
            self.history.append({"epoch": epoch, "train_mae": tr, "monitor_mae": mo,
                                 "train_batch_l1": ep_loss / max(nb, 1), "wall_s": wall,
                                 "lam": self.lam,
                                 "k_active": int(getattr(self.model, "k_active", 0))})
            score = mo if self.eval_monitor else tr
            if score < self.best_score - 1e-12:
                self.best_score = score
                if self.eval_monitor:
                    self.best_monitor = mo
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in self.model.state_dict().items()})
            state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in self.model.state_dict().items()})
            self.top5.append({"score": score, "epoch": epoch, "state": state})
            self.top5.sort(key=lambda x: x["score"])
            self.top5 = self.top5[:TOP_K]
            if epoch % 5 == 0 or epoch <= 2 or epoch == max_epochs:
                self.log(f"[{self.arm}] ep {epoch} train={tr:.4f} monitor={mo:.4f} "
                         f"best={self.best_score:.4f}@{self.best_epoch} wall={wall:.0f}s")
            if self.eval_monitor and epoch - self.best_epoch >= patience:
                self.log(f"[{self.arm}] early stop at epoch {epoch} "
                         f"(best {self.best_score:.4f}@{self.best_epoch})")
                break
        soup = _soup_state(self.top5)
        soup_mae = self._eval_state(soup)
        return {
            "history": self.history,
            "best_score": self.best_score,
            "best_monitor": self.best_monitor if self.eval_monitor else float("nan"),
            "best_epoch": self.best_epoch,
            "soup_mae": soup_mae,
            "eval_monitor": self.eval_monitor,
            "wall_s": self.history[-1]["wall_s"],
            "epochs_run": self.history[-1]["epoch"],
            "peak_gpu_mb": self.peak_gpu_mb,
            "state_best": self.best_state,
            "state_soup": soup,
        }

    def _eval_state(self, state):
        if state is None or not self.eval_monitor:
            return float("nan")
        backup = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
        self.model.load_state_dict(state)
        ma = float((self._forward_all(self.monitor_mols) - self.ymon).abs().mean())
        self.model.load_state_dict(backup)
        return ma


# ===========================================================================
# 4. Splits (train-only; official valid/test never used here)
# ===========================================================================
def train_only_split(n_total: int, n_dev: int, n_mon: int, seed: int = SPLIT_SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    dev = sorted(int(i) for i in perm[:n_dev])
    mon = sorted(int(i) for i in perm[n_dev:n_dev + n_mon])
    return dev, mon


# ===========================================================================
# 5. Stage 0 -- data-free correctness
# ===========================================================================
def _synth_mol(seed: int, n: int = 8) -> Mol:
    rng = np.random.RandomState(seed)
    node_types = rng.randint(0, 4, size=n).astype(np.int64)
    bonds = []
    for v in range(1, n):
        u = int(rng.randint(0, v))
        bonds.append((min(u, v), max(u, v)))
    bond_types = rng.randint(0, 2, size=len(bonds)).astype(np.int64)
    return Mol(n=n, bonds=bonds, bond_types=bond_types, node_types=node_types)


def stage0_tests(device: str, log=print) -> dict[str, Any]:
    torch = _torch()
    mols = [_synth_mol(s) for s in range(5)]
    ai, bi = category_index(mols)
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    results: dict[str, Any] = {}

    _seed_everything(SEED)
    model = PairEncoder.build(n_atom, n_bond).to(device)
    bfn = build_pair_batch

    # 1. permutation equivariance of layer states + prediction invariance
    mol = mols[0]
    perm = np.random.RandomState(3).permutation(mol.n)
    eb = np.random.RandomState(4).permutation(mol.m)
    molp = permute_mol(mol, perm, eb)
    b = bfn([mol], ai, bi, device)
    bp = bfn([molp], ai, bi, device)
    model.eval()
    with torch.no_grad():
        sg = model.forward_layer_states(b)
        sp = model.forward_layer_states(bp)
        pg = float(model(b)[0])
        pp = float(model(bp)[0])
    max_eq = 0.0
    for hg, hp in zip(sg, sp):
        # hp[v, w] corresponds to original (perm[v], perm[w])
        hg_np = hg.detach().cpu().numpy()[0]
        hp_np = hp.detach().cpu().numpy()[0]
        idx = np.ix_(perm, perm)
        max_eq = max(max_eq, float(np.abs(hp_np[idx] - hg_np).max()))
    pred_inv = abs(pg - pp)

    # 2. batching invariance
    b2 = bfn(mols[:4], ai, bi, device)
    with torch.no_grad():
        out2 = model(b2).detach().cpu().numpy()
    batch_inv = float(np.abs(out2[0] - pg).max())

    # 3. padding correctness (batch of mols[0] alone vs padded with bigger graphs)
    bpad = bfn([mol, mols[3], mols[4]], ai, bi, device)
    with torch.no_grad():
        outpad = model(bpad).detach().cpu().numpy()
    pad_inv = float(abs(outpad[0] - pg))

    # 4. graph-order invariance
    border = bfn([mols[1], mol, mols[2]], ai, bi, device)
    with torch.no_grad():
        outorder = model(border).detach().cpu().numpy()
    order_inv = float(abs(outorder[1] - pg))

    # 5. gradient reaches all structural tensors
    y = torch.as_tensor([0.0, 1.0, 0.5, -0.5, 2.0], device=device)
    bb = bfn(mols, ai, bi, device)
    model.train()
    pred = model(bb)
    loss = torch.nn.functional.l1_loss(pred, y)
    names = []
    for name, p in model.named_parameters():
        if "E_A" in name or "E_R" in name or "W_init" in name or ".phi" in name \
                or ".psi" in name or ".eta" in name or "W_read" in name or "head" in name:
            g = torch.autograd.grad(loss, p, retain_graph=True, allow_unused=True)[0]
            names.append((name, None if g is None else float(g.norm())))
    grad_ok = all(v is not None and v > 0 for _, v in names)

    # 6. structural: no positional/absolute embedding parameter
    param_names = [n for n, _ in model.named_parameters()]
    no_pos = not any(("pos" in n or "position" in n or "node_id" in n) for n in param_names)

    # 7. official test guard
    test_blocked = True
    try:
        load_mols_and_y(REPO_ROOT / "data/ZINC", "test", 1)
        test_blocked = False
    except Exception:
        test_blocked = True

    # 8. params
    params = _param_breakdown(model)["total_trainable"]

    passed = (max_eq < 1e-5 and pred_inv < 1e-5 and batch_inv < 1e-5
              and pad_inv < 1e-6 and order_inv < 1e-5 and grad_ok and no_pos
              and test_blocked)
    results["pair"] = {
        "layer_state_equivariance_max": max_eq,
        "graph_prediction_invariance": pred_inv,
        "batching_invariance": batch_inv,
        "padding_invariance": pad_inv,
        "graph_order_invariance": order_inv,
        "all_structural_gradients_nonzero": grad_ok,
        "no_absolute_position_parameter": no_pos,
        "official_test_blocked": test_blocked,
        "total_trainable": params,
        "passed": bool(passed),
        "grad_norms": {k: v for k, v in names},
    }
    results["all_passed"] = bool(passed)
    log(f"[stage0] eq={max_eq:.2e} pred_inv={pred_inv:.2e} batch_inv={batch_inv:.2e} "
        f"pad_inv={pad_inv:.2e} order_inv={order_inv:.2e} grad_ok={grad_ok} "
        f"params={params} passed={passed}")
    return results


# ===========================================================================
# 6. Synthetic mechanism sanity (mechanism-only)
# ===========================================================================
def _cycle_mol(n: int) -> Mol:
    bonds = [(i, (i + 1) % n) for i in range(n)]
    bonds = [(min(a, b), max(a, b)) for a, b in bonds]
    return Mol(n=n, bonds=bonds, bond_types=np.zeros(len(bonds), dtype=np.int64),
               node_types=np.zeros(n, dtype=np.int64))


def _two_triangles_mol() -> Mol:
    bonds = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    return Mol(n=6, bonds=bonds, bond_types=np.zeros(6, dtype=np.int64),
               node_types=np.zeros(6, dtype=np.int64))


def synth_mechanism(device: str, log=print) -> dict[str, Any]:
    torch = _torch()
    c6 = _cycle_mol(6)
    t2 = _two_triangles_mol()
    mols = [c6, t2]
    ai, bi = category_index(mols)
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    _seed_everything(SEED)

    # raw node-MPNN: node-state multisets (identical for 1-WL-equivalent graphs)
    raw = build_raw(n_atom, n_bond).to(device)
    raw.eval()
    rb = build_raw_batch(mols, ai, bi, device)
    with torch.no_grad():
        rs = raw.forward_layer_states(rb)[-1].detach().cpu().numpy()
    parts = {}
    off = 0
    for gi, m in enumerate(mols):
        parts[gi] = rs[off:off + m.n]
        off += m.n
    raw_gap = float(np.abs(np.sort(np.linalg.norm(parts[0], axis=1))
                           - np.sort(np.linalg.norm(parts[1], axis=1))).max())

    # pair encoder: pair-state multiset (permutation-invariant via sorted projections)
    pair = PairEncoder.build(n_atom, n_bond).to(device)
    pair.eval()
    pb = build_pair_batch(mols, ai, bi, device)
    with torch.no_grad():
        ps = pair.forward_layer_states(pb)[-1]
        ug = pair.encode(pb)
    nmask = pb["node_mask"].detach().cpu().numpy()
    states = []
    for gi in range(2):
        n = int(nmask[gi].sum())
        states.append(ps[gi][:n, :n, :].reshape(-1, ps.shape[-1]))
    gen = torch.Generator().manual_seed(1234)
    proj_gaps = []
    for _ in range(4):
        v = torch.randn(ps.shape[-1], generator=gen)
        v = v / v.norm()
        a = (states[0] @ v).sort().values
        b = (states[1] @ v).sort().values
        proj_gaps.append(float((a - b).abs().max()))
    pair_gap = float(np.mean(proj_gaps))
    ug_gap = float((ug[0] - ug[1]).norm())

    out = {
        "raw_node_state_multiset_gap": raw_gap,
        "pair_state_projection_gap": pair_gap,
        "graph_representation_gap": ug_gap,
        "mechanism_distinguishes_1wl_pair": bool(pair_gap > 1e-3 and ug_gap > 1e-3
                                                       and raw_gap < 1e-4),
    }
    log(f"[synth] raw_gap={raw_gap:.2e} pair_proj_gap={pair_gap:.4f} "
        f"u_G_gap={ug_gap:.4f} distinguishes={out['mechanism_distinguishes_1wl_pair']}")
    return out


# ===========================================================================
# 7. Orchestration helpers
# ===========================================================================
def _atom_bond_index(train_mols, valid_mols=None):
    mols = list(train_mols) + (list(valid_mols) if valid_mols else [])
    return category_index(mols)


def run_params(log) -> int:
    mols, _ = load_mols_and_y(REPO_ROOT / "data/ZINC", "train", None)
    ai, bi = _atom_bond_index(mols)
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    out: dict[str, Any] = {"n_atom": n_atom, "n_bond": n_bond}
    for arm in ("pair", "raw", "raw_wide"):
        m = build_model(arm, n_atom, n_bond)
        out[arm] = _param_breakdown(m)
    write_json(RESULTS_DIR / "params.json", out)
    log(json.dumps(out, indent=2))
    return 0


def run_stage0(args, log) -> int:
    res = stage0_tests(args.device, log=log)
    write_json(RESULTS_DIR / "stage0_tests.json",
               {"provenance": provenance(args.device), "tests": res})
    return 0 if res["all_passed"] else 1


def run_synth(args, log) -> int:
    res = synth_mechanism(args.device, log=log)
    write_json(RESULTS_DIR / "synth_mechanism.json",
               {"provenance": provenance(args.device), "tests": res})
    return 0 if res["mechanism_distinguishes_1wl_pair"] else 1


def run_overfit(args, log) -> int:
    torch = _torch()
    _seed_everything(SEED)
    train_mols, y_train = load_mols_and_y(REPO_ROOT / "data/ZINC", "train", OVERFIT_N)
    ai, bi = _atom_bond_index(train_mols)
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    arm = args.arm
    model = build_model(arm, n_atom, n_bond).to(args.device)
    trainer = EpochTrainer(model, train_mols, None, y_train, None, ai, bi,
                           args.device, f"{arm}_overfit", batch_fn_for(arm), log=log,
                           eval_monitor=False)
    out = trainer.train(max_epochs=args.max_epochs, patience=999)
    hist = out["history"]
    mae1 = hist[0]["train_mae"]
    best = out["best_score"]
    gate = {"best_train_mae_le_0.15": bool(best <= 0.15),
            "best_le_half_epoch1": bool(best <= 0.5 * mae1)}
    gate["passed"] = all(bool(v) for k, v in gate.items() if k != "passed")
    payload = {"provenance": provenance(args.device, {"arm": arm, "n": OVERFIT_N}),
               "arm": arm, "history": hist, "best_train_mae": best,
               "epoch1_train_mae": mae1, "params": _param_breakdown(model),
               "gate": gate, "wall_s": out["wall_s"], "peak_gpu_mb": out["peak_gpu_mb"]}
    write_json(RESULTS_DIR / f"overfit_{arm}.json", payload)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out["state_best"], STATE_DIR / f"overfit_{arm}_best.pt")
    log(f"[overfit:{arm}] best_train={best:.4f} epoch1={mae1:.4f} gate={gate['passed']}")
    return 0 if gate["passed"] else 1


def run_screen(args, log) -> int:
    torch = _torch()
    _seed_everything(SEED)
    train_mols, y_train = load_mols_and_y(REPO_ROOT / "data/ZINC", "train", None)
    dev_idx, mon_idx = train_only_split(len(train_mols), SCREEN_DEV_N, SCREEN_MON_N)
    dev = [train_mols[i] for i in dev_idx]
    mon = [train_mols[i] for i in mon_idx]
    ydev = np.asarray([y_train[i] for i in dev_idx])
    ymon = np.asarray([y_train[i] for i in mon_idx])
    ai, bi = _atom_bond_index(train_mols)
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    arm = args.arm
    model = build_model(arm, n_atom, n_bond).to(args.device)
    trainer = EpochTrainer(model, dev, mon, ydev, ymon, ai, bi, args.device,
                           f"{arm}_screen", batch_fn_for(arm), log=log,
                           eval_monitor=True)
    out = trainer.train(max_epochs=args.max_epochs, patience=args.patience)
    payload = {"provenance": provenance(args.device, {"arm": arm,
                                                      "n_dev": SCREEN_DEV_N,
                                                      "n_mon": SCREEN_MON_N,
                                                      "split_seed": SPLIT_SEED}),
               "arm": arm, "history": out["history"],
               "best_monitor": out["best_monitor"], "best_epoch": out["best_epoch"],
               "soup_mae": out["soup_mae"], "params": _param_breakdown(model),
               "epochs_run": out["epochs_run"], "wall_s": out["wall_s"],
               "peak_gpu_mb": out["peak_gpu_mb"],
               "train_monitor_gap": out["history"][-1]["train_mae"] - out["history"][-1]["monitor_mae"]}
    write_json(RESULTS_DIR / f"screen_{arm}.json", payload)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CURVE_DIR / f"history_screen_{arm}.json", out["history"])
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out["state_best"], STATE_DIR / f"screen_{arm}_best.pt")
    if out["state_soup"] is not None:
        SOUP_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(out["state_soup"], SOUP_DIR / f"screen_{arm}_top5_soup.pt")
    log(f"[screen:{arm}] best_monitor={out['best_monitor']:.4f} soup={out['soup_mae']:.4f} "
        f"epoch={out['best_epoch']} wall={out['wall_s']:.0f}s")
    return 0


def _load_state(path: Path):
    torch = _torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_encoder_head(arm: str, n_atom: int, n_bond: int, state_path: Path, device: str,
                       log=print):
    base = build_model(arm, n_atom, n_bond)
    base.load_state_dict(_load_state(state_path))
    if arm == "pair":
        encoder = base
        head = base.head
    else:
        encoder = None
        head = base.head
    return base, encoder, head


# ===========================================================================
# 8. Q2 -- function-preserving insertion
# ===========================================================================
def run_preserve(args, log) -> int:
    torch = _torch()
    _seed_everything(SEED)
    train_mols, _ = load_mols_and_y(REPO_ROOT / "data/ZINC", "train", None)
    dev_idx, mon_idx = train_only_split(len(train_mols), SCREEN_DEV_N, SCREEN_MON_N)
    mon = [train_mols[i] for i in mon_idx]
    ai, bi = _atom_bond_index(train_mols)
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    ckpt = STATE_DIR / "screen_pair_best.pt"
    if not ckpt.exists():
        ckpt = STATE_DIR / "overfit_pair_best.pt"
    enc, _, head = _load_encoder_head("pair", n_atom, n_bond, ckpt, args.device, log=log)
    enc.eval()
    bfn = build_pair_batch

    def pred_all(model):
        out = []
        with torch.no_grad():
            for s in range(0, len(mon), BATCH):
                out.append(model(bfn(mon[s:s + BATCH], ai, bi, args.device)))
        return torch.cat(out)

    base_pred = pred_all(enc)

    rows = []
    for kind in ("dense", "learned_dict", "fixed_identity", "generic_topk", "learned_nosp"):
        coded = GraphCodeModel.build(copy.deepcopy(enc), copy.deepcopy(head), D_MODEL,
                                     kind).to(args.device)
        coded.eval()
        p = pred_all(coded)
        delta = float((p - base_pred).abs().max())
        rows.append({"kind": kind, "max_abs_pred_delta_at_insertion": delta})
        log(f"[preserve] {kind}: max|dpred|={delta:.3e}")
    passed = all(r["max_abs_pred_delta_at_insertion"] < 1e-4 for r in rows)
    write_json(RESULTS_DIR / "insertion_preservation.json",
               {"provenance": provenance(args.device, {"ckpt": str(ckpt)}),
                "rows": rows, "passed": bool(passed)})
    return 0 if passed else 1


# ===========================================================================
# 9. Q2 -- matched-control dictionary run
# ===========================================================================
def _calibrate_lambda(encoder, mols, ai, bi, device, log=print):
    torch = _torch()
    encoder.eval()
    mags = []
    bfn = build_pair_batch
    with torch.no_grad():
        for s in range(0, len(mols), BATCH):
            u = encoder.encode(bfn(mols[s:s + BATCH], ai, bi, device))
            mags.append(u.abs().detach().cpu().numpy().reshape(-1))
    m = np.concatenate(mags)
    lam = float(np.quantile(m, 1.0 - LAMBDA_ACTIVE_TARGET))
    log(f"[lambda] calibrated={lam:.5f} (target_active={LAMBDA_ACTIVE_TARGET})")
    return lam


def run_dictrun(args, log) -> int:
    torch = _torch()
    _seed_everything(SEED)
    train_mols, y_train = load_mols_and_y(REPO_ROOT / "data/ZINC", "train", None)
    dev_idx, mon_idx = train_only_split(len(train_mols), SCREEN_DEV_N, SCREEN_MON_N)
    dev = [train_mols[i] for i in dev_idx]
    mon = [train_mols[i] for i in mon_idx]
    ydev = np.asarray([y_train[i] for i in dev_idx])
    ymon = np.asarray([y_train[i] for i in mon_idx])
    ai, bi = _atom_bond_index(train_mols)
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    kind = args.kind
    ckpt = STATE_DIR / "screen_pair_best.pt"
    enc, _, head = _load_encoder_head("pair", n_atom, n_bond, ckpt, args.device, log=log)
    enc = copy.deepcopy(enc)
    head = copy.deepcopy(head)
    for p in enc.parameters():
        p.requires_grad = True
    lam_target = _calibrate_lambda(enc, dev[:512], ai, bi, args.device, log=log) \
        if kind in ("learned_dict", "fixed_identity") else 0.0
    k_target = int(round(LAMBDA_ACTIVE_TARGET * D_MODEL))
    model = GraphCodeModel.build(enc, head, D_MODEL, kind, k_active=D_MODEL).to(args.device)
    epochs = args.max_epochs

    def lam_schedule(ep):
        if kind in ("learned_dict", "fixed_identity") and epochs > 1:
            return lam_target * min(1.0, ep / max(1.0, 0.5 * epochs))
        return 0.0

    def k_schedule(ep):
        if epochs <= 1:
            return k_target
        frac = min(1.0, ep / max(1.0, 0.5 * epochs))
        return int(round(D_MODEL - frac * (D_MODEL - k_target)))

    trainer = EpochTrainer(model, dev, mon, ydev, ymon, ai, bi, args.device,
                           f"dict_{kind}", build_pair_batch, log=log, eval_monitor=True)
    out = trainer.train(max_epochs=epochs, patience=args.patience,
                        lam_schedule=lam_schedule if kind in ("learned_dict", "fixed_identity") else None,
                        k_schedule=k_schedule if kind == "generic_topk" else None)
    model.load_state_dict(out["state_best"])
    sp = model.sparse_stats(build_pair_batch(mon, ai, bi, args.device), lam=lam_target)
    sp.pop("codes", None)
    payload = {"provenance": provenance(args.device, {"kind": kind, "ckpt": str(ckpt)}),
               "kind": kind, "lam_target": lam_target, "k_target": k_target,
               "history": out["history"], "best_monitor": out["best_monitor"],
               "best_epoch": out["best_epoch"], "soup_mae": out["soup_mae"],
               "params": _param_breakdown(model), "sparse_stats": sp,
               "epochs_run": out["epochs_run"], "wall_s": out["wall_s"],
               "peak_gpu_mb": out["peak_gpu_mb"]}
    write_json(RESULTS_DIR / f"dict_{kind}.json", payload)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CURVE_DIR / f"history_dict_{kind}.json", out["history"])
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out["state_best"], STATE_DIR / f"dict_{kind}_best.pt")
    log(f"[dict:{kind}] best_monitor={out['best_monitor']:.4f} best_ep={out['best_epoch']} "
        f"active={sp['active_fraction']:.3f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PSD-v0")
    parser.add_argument("stage", choices=["stage0", "synth", "overfit", "screen",
                                          "params", "preserve", "dictrun"])
    parser.add_argument("--arm", default="pair", choices=["pair", "raw", "raw_wide"])
    parser.add_argument("--kind", default="learned_dict",
                        choices=["dense", "learned_dict", "fixed_identity",
                                 "generic_topk", "learned_nosp"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args(argv)
    _configure_determinism()

    def log(msg: str, *a):
        print(msg, *a, flush=True)

    if args.stage == "params":
        return run_params(log)
    if args.stage == "stage0":
        return run_stage0(args, log)
    if args.stage == "synth":
        return run_synth(args, log)
    if args.stage == "overfit":
        if args.max_epochs is None:
            args.max_epochs = OVERFIT_EPOCHS
        return run_overfit(args, log)
    if args.stage == "screen":
        if args.max_epochs is None:
            args.max_epochs = SCREEN_EPOCHS
        return run_screen(args, log)
    if args.stage == "preserve":
        return run_preserve(args, log)
    if args.stage == "dictrun":
        if args.max_epochs is None:
            args.max_epochs = 40
        return run_dictrun(args, log)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
