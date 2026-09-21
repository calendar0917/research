#!/usr/bin/env python
"""TCCD-v0 — Task-Coupled Compositional Dictionary: shared library.

Pre-registration: ``tracks/ksvd/notes/tccd_v0_preregistration.md`` (frozen).
Study ``zinc-context-gap``; canonical PyG ZINC ``subset=True`` official
**train 10000 / valid 1000**; **official test is never loaded**.

This module contains only the frozen TCCD-v0 primitives:

* exact rooted colored-incidence patch slot order + fixed coordinate vector
  ``x_v in R^F`` (§1.2),
* label-free detached K-SVD / OMP sparse coding (§1.3, Gate 1),
* tied unrolled iterative hard thresholding sharing ``D`` (§1.3, Gates 2-3),
* one-shot relation contractions with **no learned message passing** (§1.4),
* the linear reader and the frozen task-coupled loss (§1.5 / §1.6).

The runner is ``tracks/ksvd/code/run_tccd_v0.py``; data-free correctness tests
are ``tracks/ksvd/tests/test_tccd_v0.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOCOL_VERSION = "tccd_v0"
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tccd_v0"
CACHE_DIR = RESULTS_DIR / "cache"

# --- frozen architecture constants (§1.2-1.6) -------------------------------
RADIUS = 2
K_DICT = 64
SPARSITY = 8
TAU = 1.0
IHT_STEPS = 10
IHT_POWER_ITERS = 30
EPS = 1e-8
SPLIT_SEED = 20260922
DICT_SEED = 20260922
N_INTERNAL_TRAIN = 8000
N_INTERNAL_DEV = 2000

# canonical protocol (inherited from GSCN-v0 / OPTIMIZED_PROTOCOL)
BATCH = 32
LR = 1.0e-3
WD = 1.0e-5
CLIP = 5.0
MAX_EPOCHS = 240
PATIENCE = 40
TOP_K_SOUP = 5
TORCH_THREADS = 4

CONSTRUCTION_FINGERPRINT = (
    f"tccd_v0;r={RADIUS};slot=(shell,canonrank);blocks=topo,bond,atom,shell,mask;"
    f"tau={TAU};rint_diag=removed"
)


# ===========================================================================
# provenance
# ===========================================================================
def git(*args: str):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def provenance(device: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "construction_fingerprint": CONSTRUCTION_FINGERPRINT,
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device": device,
        "official_test_loaded": False,
        "config": {
            "radius": RADIUS,
            "K": K_DICT,
            "sparsity": SPARSITY,
            "tau": TAU,
            "iht_steps": IHT_STEPS,
            "split_seed": SPLIT_SEED,
            "dict_seed": DICT_SEED,
            "n_internal_train": N_INTERNAL_TRAIN,
            "n_internal_dev": N_INTERNAL_DEV,
        },
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        if device.startswith("cuda") and torch.cuda.is_available():
            payload["gpu"] = torch.cuda.get_device_name(0)
            payload["cuda"] = torch.version.cuda
    except Exception:  # pragma: no cover
        pass
    if extra:
        payload.update(extra)
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=float)
        + "\n",
        encoding="utf-8",
    )


# ===========================================================================
# data loading (canonical repo loader, train/valid only)
# ===========================================================================
def repo_loader():
    from tracks.ksvd.code.run_aiom_representation_audit import (  # type: ignore
        mol_from_pyg,
        repo_loader as _rl,
    )

    return _rl()[0], _rl()[1], mol_from_pyg


def load_mols(root: Path, split: str, limit: int | None = None):
    """Load a canonical ZINC split. ``test`` is refused."""
    if split == "test":
        raise RuntimeError("TCCD-v0 never loads the official test split")
    load_zinc, data_to_graph, mol_from_pyg = repo_loader()
    pyg_split = "val" if split == "valid" else split
    dataset = load_zinc(root, pyg_split)
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    mols = []
    ys = []
    for i in range(n):
        data = dataset[i]
        mols.append(mol_from_pyg(data, data_to_graph))
        ys.append(float(data.y.reshape(-1)[0]))
    return mols, np.asarray(ys, dtype=np.float64)


def category_catalog(mols: Sequence[Any]) -> tuple[dict[int, int], dict[int, int]]:
    """Observed atom / bond categories (train-derived, label-free)."""
    atoms = sorted({int(x) for m in mols for x in m.node_types.tolist()})
    bonds = sorted({int(x) for m in mols for x in m.bond_types.tolist()})
    return ({c: i for i, c in enumerate(atoms)}, {c: i for i, c in enumerate(bonds)})


# ===========================================================================
# §1.2 exact rooted colored-incidence patch slot order + coordinate
# ===========================================================================
def _adjacency(mol) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(int(mol.n))]
    for a, b in mol.bonds:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def bfs_distances(adj: Sequence[Sequence[int]], root: int, radius: int) -> dict[int, int]:
    dist = {int(root): 0}
    queue = [int(root)]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        if dist[u] >= radius:
            continue
        for w in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                queue.append(w)
    return dist


def patch_slot_order(mol, root: int, adj=None, radius: int = RADIUS):
    """Exact canonical slot order of a rooted radius-``radius`` patch.

    Uses the repository's exact rooted **colored-incidence** canonicalization
    (``typed_patch_tokenizer.build_colored_incidence``): atom vertices are
    coloured ``("node", is_root, distance_from_root, atom_type)`` and bond
    (incidence) vertices ``("edge", bond_type)``; ``pynauty.canon_label``
    yields the canonical labelling.  The frozen slot order is root first, then
    shell-1 / shell-2 atoms by increasing canonical rank.

    Returns ``(slot_nodes, slot_shells, canonical_key)`` with ``slot_nodes``
    the original atom ids.  The construction is permutation-invariant: for
    automorphic (identical-colour) atoms the canonical labelled graph is the
    same, so the coordinate vector is unchanged.
    """
    from tracks.ksvd.code.graph import from_edges  # type: ignore
    from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (  # type: ignore
        _import_pynauty,
        _pynauty_graph,
        build_colored_incidence,
        corrected_canonical_key,
    )

    edges: list[tuple[int, int]] = []
    edge_types: dict[tuple[int, int], int] = {}
    for e, (a, b) in enumerate(mol.bonds):
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        edges.append((a, b))
        edge_types[(a, b)] = int(mol.bond_types[e])
    graph = from_edges(int(mol.n), edges)
    incidence = build_colored_incidence(
        graph, int(root), np.asarray(mol.node_types, dtype=np.int64), edge_types, int(radius)
    )
    pynauty = _import_pynauty()
    pg = _pynauty_graph(pynauty, incidence)
    label = [int(v) for v in pynauty.canon_label(pg)]
    atom_label = [v for v in label if v < incidence.n_nodes]
    if sorted(atom_label) != list(range(incidence.n_nodes)):
        raise RuntimeError("canon_label is not a permutation on patch atoms")
    rank = {v: i for i, v in enumerate(atom_label)}
    shells = {v: int(incidence.color_keys[v][2]) for v in range(incidence.n_nodes)}
    slots = sorted(range(incidence.n_nodes), key=lambda v: (shells[v], rank[v]))
    slot_nodes = [int(incidence.original_nodes[v]) for v in slots]
    slot_shells = [shells[v] for v in slots]
    key = corrected_canonical_key(incidence)
    return slot_nodes, slot_shells, key


@dataclass(frozen=True)
class PatchLayout:
    """Fixed-coordinate layout of `x_v in R^F` (frozen in the pre-registration)."""

    capacity: int
    n_atom: int
    n_bond: int

    @property
    def num_pairs(self) -> int:
        return self.capacity * (self.capacity - 1) // 2

    @property
    def feature_dim(self) -> int:
        return (
            1 * self.num_pairs
            + self.n_bond * self.num_pairs
            + self.capacity * self.n_atom
            + self.capacity * 3
            + self.capacity
        )

    @property
    def pair_i(self) -> np.ndarray:
        return np.triu_indices(self.capacity, k=1)[0]

    @property
    def pair_j(self) -> np.ndarray:
        return np.triu_indices(self.capacity, k=1)[1]

    def block_slices(self) -> dict[str, slice]:
        c = 0
        out: dict[str, slice] = {}
        out["topology"] = slice(c, c + self.num_pairs); c += self.num_pairs
        out["bond"] = slice(c, c + self.n_bond * self.num_pairs); c += self.n_bond * self.num_pairs
        out["atom"] = slice(c, c + self.capacity * self.n_atom); c += self.capacity * self.n_atom
        out["shell"] = slice(c, c + self.capacity * 3); c += self.capacity * 3
        out["mask"] = slice(c, c + self.capacity); c += self.capacity
        assert c == self.feature_dim
        return out

    def descriptor(self) -> dict[str, Any]:
        return {
            "capacity": int(self.capacity),
            "n_atom": int(self.n_atom),
            "n_bond": int(self.n_bond),
            "num_pairs": int(self.num_pairs),
            "feature_dim": int(self.feature_dim),
        }


def _pair_lookup(layout: PatchLayout) -> np.ndarray:
    lut = -np.ones((layout.capacity, layout.capacity), dtype=np.int64)
    lut[layout.pair_i, layout.pair_j] = np.arange(layout.num_pairs, dtype=np.int64)
    return lut


def build_mol_record(
    mol,
    layout: PatchLayout,
    atom_index: Mapping[int, int],
    bond_index: Mapping[int, int],
    radius: int = RADIUS,
) -> dict[str, Any]:
    """Per-molecule frozen record: patch coordinates + raw relation matrices.

    ``X``      [n_v, F]  patch coordinates
    ``B``      [n_v, N]  patch<->original-atom incidence
    ``Rb``     [n_bond, n_v, n_v]  root-level native bond attachment per category
    ``Rgeo``   [n_v, n_v]  exp(-d(root_i, root_j)/tau) exact shortest-path
    """
    n = int(mol.n)
    adj = _adjacency(mol)
    slices = layout.block_slices()
    lut = _pair_lookup(layout)
    cap = layout.capacity
    pi, pj = layout.pair_i, layout.pair_j
    F = layout.feature_dim
    X = np.zeros((n, F), dtype=np.float32)
    truncations = 0
    keys: list[bytes] = []
    root_cat: list[int] = []
    slot_cache: list[list[int]] = []
    for v in range(n):
        slot_nodes, slot_shells, key = patch_slot_order(mol, v, adj=adj, radius=radius)
        slot_cache.append(slot_nodes)
        keys.append(key)
        root_cat.append(int(atom_index[int(mol.node_types[v])]))
        if len(slot_nodes) > cap:
            truncations += 1
        row = X[v]
        topo = row[slices["topology"]]
        bondblk = row[slices["bond"]].reshape(layout.n_bond, layout.num_pairs)
        atomblk = row[slices["atom"]].reshape(cap, layout.n_atom)
        shellblk = row[slices["shell"]].reshape(cap, 3)
        for s, a in enumerate(slot_nodes[:cap]):
            atomblk[s, int(atom_index[int(mol.node_types[a])])] = 1.0
            shellblk[s, int(min(slot_shells[s], 2))] = 1.0
        row[slices["mask"]][: min(len(slot_nodes), cap)] = 1.0
        slot_pos = {a: s for s, a in enumerate(slot_nodes[:cap])}
        for e, (a, b) in enumerate(mol.bonds):
            a, b = int(a), int(b)
            if a in slot_pos and b in slot_pos:
                i, j = slot_pos[a], slot_pos[b]
                if i > j:
                    i, j = j, i
                pid = int(lut[i, j])
                topo[pid] = 1.0
                bondblk[int(bond_index[int(mol.bond_types[e])]), pid] = 1.0

    # patch <-> original atom incidence
    B = np.zeros((n, n), dtype=np.float32)
    for v in range(n):
        for a in slot_cache[v][:cap]:
            B[v, int(a)] = 1.0

    # native bond categories at root level
    Rb = np.zeros((layout.n_bond, n, n), dtype=np.float32)
    for e, (a, b) in enumerate(mol.bonds):
        i, j = int(a), int(b)
        Rb[int(bond_index[int(mol.bond_types[e])]), i, j] = 1.0
        Rb[int(bond_index[int(mol.bond_types[e])]), j, i] = 1.0

    # one fixed global relative-position operator
    Rgeo = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        d = bfs_distances(adj, i, radius=n)
        for j, dij in d.items():
            Rgeo[i, j] = math.exp(-float(dij) / float(TAU))
    # disconnected pairs are zero (ZINC molecules are connected)

    return {
        "X": X,
        "B": B,
        "Rb": Rb,
        "Rgeo": Rgeo,
        "keys": keys,
        "root_cat": np.asarray(root_cat, dtype=np.int64),
        "n": n,
        "truncations": int(truncations),
    }


def choose_capacity(mols: Sequence[Any], radius: int = RADIUS) -> int:
    """M = max radius-2 patch atom count over the given (train) molecules."""
    best = 1
    for mol in mols:
        adj = _adjacency(mol)
        for v in range(int(mol.n)):
            best = max(best, len(bfs_distances(adj, v, radius)))
    return int(best)


def build_records(
    mols: Sequence[Any],
    layout: PatchLayout,
    atom_index: Mapping[int, int],
    bond_index: Mapping[int, int],
    ys: Sequence[float] | None = None,
    log=print,
) -> list[dict[str, Any]]:
    records = []
    t0 = time.time()
    for i, mol in enumerate(mols):
        rec = build_mol_record(mol, layout, atom_index, bond_index)
        if ys is not None:
            rec["y"] = float(ys[i])
        rec["graph_index"] = i
        records.append(rec)
        if (i + 1) % 1000 == 0:
            log(f"  records {i + 1}/{len(mols)} ({time.time() - t0:.0f}s)")
    return records


def records_fingerprint(records: Sequence[dict[str, Any]], layout: PatchLayout) -> str:
    h = hashlib.sha256()
    h.update(CONSTRUCTION_FINGERPRINT.encode())
    h.update(json.dumps(layout.descriptor(), sort_keys=True).encode())
    h.update(str(len(records)).encode())
    for rec in records[:64]:
        h.update(np.asarray(rec["X"], dtype=np.float32).tobytes())
    return h.hexdigest()


def save_records(path: Path, records: Sequence[dict[str, Any]], meta: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"records": list(records), "meta": dict(meta)}, fh, protocol=4)


def load_records(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


# ===========================================================================
# §1.3 sparse coding: detached OMP / K-SVD  and tied IHT
# ===========================================================================
def omp_codes(D: np.ndarray, X: np.ndarray, s: int = SPARSITY, chunk: int = 16384) -> np.ndarray:
    """OMP codes for patches ``X [N, F]`` over ``D [F, K]`` -> ``C [N, K]``."""
    from sklearn.linear_model import orthogonal_mp  # type: ignore

    D = np.asarray(D, dtype=np.float64)
    out = np.zeros((X.shape[0], D.shape[1]), dtype=np.float32)
    for start in range(0, X.shape[0], chunk):
        block = np.asarray(X[start : start + chunk], dtype=np.float64)
        C = orthogonal_mp(D, block.T, n_nonzero_coefs=int(s), precompute=True)
        out[start : start + chunk] = np.asarray(C.T, dtype=np.float32)
    return out


def normalize_columns(D: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return D / norms


def random_normalized_dictionary(F: int, K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    D = rng.standard_normal((F, K)).astype(np.float64)
    return normalize_columns(D)


def ksvd_fit(
    X: np.ndarray,
    D0: np.ndarray,
    s: int = SPARSITY,
    epochs: int = 12,
    chunk: int = 16384,
    seed: int = DICT_SEED,
    log=print,
    progress: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mini-batch K-SVD (OMP coding + residual rank-1 atom update).

    ``X`` [N, F] are the label-free dictionary-fit patch coordinates.
    """
    from sklearn.linear_model import orthogonal_mp  # type: ignore

    D = normalize_columns(np.asarray(D0, dtype=np.float64).copy())
    F, K = D.shape
    N = X.shape[0]
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []
    for ep in range(int(epochs)):
        order = rng.permutation(N)
        ep_err = 0.0
        ep_cnt = 0
        for start in range(0, N, chunk):
            idx = order[start : start + chunk]
            Xc = np.asarray(X[idx], dtype=np.float64)
            C = orthogonal_mp(D, Xc.T, n_nonzero_coefs=int(s), precompute=True).T  # [n, K]
            resid = Xc - C @ D.T
            # atom updates
            for k in range(K):
                supp = np.flatnonzero(np.abs(C[:, k]) > 0)
                if supp.size == 0:
                    continue
                E = (resid[supp] + np.outer(C[supp, k], D[:, k])).T  # [F, |supp|]
                U, sv, Vt = np.linalg.svd(E, full_matrices=False)
                D[:, k] = U[:, 0]
                C[supp, k] = sv[0] * Vt[0]
            D = normalize_columns(D)
            err = float(np.sum(resid * resid))
            ep_err += err
            ep_cnt += Xc.shape[0]
        entry = {"epoch": ep + 1, "mean_sq_err": ep_err / max(ep_cnt, 1)}
        history.append(entry)
        if log:
            log(f"  ksvd epoch {ep + 1}/{epochs} fit_mse={entry['mean_sq_err']:.6f}")
    if progress is not None:
        progress["ksvd_history"] = history
    return D, {"history": history}


def relative_reconstruction_error(D: np.ndarray, X: np.ndarray, C: np.ndarray | None = None,
                                  s: int = SPARSITY) -> float:
    if C is None:
        C = omp_codes(D, X, s=s)
    X = np.asarray(X, dtype=np.float64)
    R = X - C.astype(np.float64) @ np.asarray(D, dtype=np.float64).T
    num = float(np.mean(np.sum(R * R, axis=1)))
    den = float(np.mean(np.sum(X * X, axis=1)))
    return num / (den + EPS)


# --- torch IHT (tied, no free encoder) --------------------------------------
def _torch():
    import torch

    return torch


def power_iter_sigma(D):
    """Spectral norm of ``D`` by power iteration with a **frozen deterministic**
    start vector (no RNG), so ``eta`` is a deterministic function of ``D``."""
    torch = _torch()
    with torch.no_grad():
        Dd = D.detach()
        k = Dd.shape[1]
        idx = torch.arange(k, device=Dd.device, dtype=Dd.dtype)
        x = torch.cos(2.0 * math.pi * idx / max(k, 1))
        x = x / (x.norm() + 1e-12)
        for _ in range(IHT_POWER_ITERS):
            y = Dd @ x
            x = Dd.t() @ y
            nrm = x.norm()
            if nrm < 1e-12:
                return torch.tensor(0.0, device=Dd.device)
            x = x / nrm
        sigma = (Dd @ x).norm()
    return sigma


def hard_threshold_rows(C, s: int):
    torch = _torch()
    if s >= C.shape[1]:
        return C
    _, idx = torch.topk(C.abs(), k=int(s), dim=1)
    mask = torch.zeros_like(C, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return C * mask


def iht_codes(D, X, s: int = SPARSITY, steps: int = IHT_STEPS, eta=None):
    """Tied unrolled IHT sharing ``D`` for encoding and reconstruction."""
    torch = _torch()
    if eta is None:
        sigma = power_iter_sigma(D)
        eta = 1.0 / (sigma * sigma + EPS)
    C = torch.zeros(X.shape[0], D.shape[1], device=X.device, dtype=X.dtype)
    for _ in range(int(steps)):
        C = C + eta * ((X - C @ D.t()) @ D)
        C = hard_threshold_rows(C, s)
    return C


def dense_codes(D, X):
    """Tied linear dense local representation ``z_v = Dᵀ x_v`` (matched width)."""
    return X @ D


# ===========================================================================
# §1.4 / §1.5 composition + reader
# ===========================================================================
def relation_matrices(rec: Mapping[str, Any]):
    """(R_int, [R_b...], R_geo) with the frozen diagonal conventions."""
    B = np.asarray(rec["B"], dtype=np.float32)
    Rint = B @ B.T
    np.fill_diagonal(Rint, 0.0)
    Rb = [np.asarray(r, dtype=np.float32) for r in rec["Rb"]]
    Rgeo = np.asarray(rec["Rgeo"], dtype=np.float32)
    return Rint, Rb, Rgeo


def sym_indices(K: int) -> tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(K)
    return iu[0], iu[1]


def compose_torch(C, Rint, Rb_list, Rgeo, iu0, iu1, shuffle_perm=None):
    """h_G for one graph. ``C`` [n, K]; returns [h_dim]."""
    torch = _torch()
    if shuffle_perm is not None:
        C = C.index_select(0, shuffle_perm)
    parts = [C.sum(dim=0)]
    for R in [Rint, *Rb_list, Rgeo]:
        M = C.t() @ R @ C
        parts.append(M[iu0, iu1])
    return torch.cat(parts, dim=0)


def h_dim(K: int = K_DICT, n_rel: int = 5) -> int:
    return K + n_rel * (K * (K + 1) // 2)


# ===========================================================================
# torch model
# ===========================================================================
class TCCDModel:
    """Constructs the frozen TCCD-v0 torch module."""

    @staticmethod
    def build(F: int, K: int, n_rel: int, D_init: np.ndarray, dense: bool = False,
              frozen_D: bool = False, readout_dim: int | None = None, seed: int = 0):
        torch = _torch()
        nn = torch.nn

        H = readout_dim if readout_dim is not None else h_dim(K, n_rel)

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                D = torch.as_tensor(np.asarray(D_init, dtype=np.float32).copy())
                self.D = nn.Parameter(D)
                self.head = nn.Linear(H, 1)
                self.dense = bool(dense)
                self.frozen_D = bool(frozen_D)
                self.K = int(K)
                self.F = int(F)

            def encode(self, X, shuffle_perm=None, s: int = SPARSITY):
                if self.dense:
                    return dense_codes(self.D, X)
                return iht_codes(self.D, X, s=s)

            def forward(self, batch, shuffle_perms=None):
                # batch: list of dicts with x_all + graph offsets + relations
                return self.forward_from_codes(batch, encode=True, shuffle_perms=shuffle_perms)

            def forward_from_codes(self, batch, encode=True, shuffle_perms=None):
                device = self.D.device
                iu0, iu1 = batch["iu0"], batch["iu1"]
                C_all = None
                if encode:
                    C_all = self.encode(batch["X_all"])
                preds = []
                for gi, graph in enumerate(batch["graphs"]):
                    if C_all is not None:
                        C = C_all[graph["slice"]]
                    else:
                        C = graph["C"]
                    perm = None
                    if shuffle_perms is not None:
                        perm = shuffle_perms[gi]
                    h = compose_torch(
                        C, graph["Rint"], graph["Rb"], graph["Rgeo"], iu0, iu1, shuffle_perm=perm
                    )
                    preds.append(self.head(h))
                return torch.stack(preds).reshape(-1)

            def renormalize_(self):
                if self.frozen_D:
                    return
                with _torch().no_grad():
                    norms = self.D.detach().norm(dim=0, keepdim=True)
                    norms = norms.clamp_min(1e-6)
                    self.D.mul_(1.0 / norms)

        return _M()


# ===========================================================================
# batched tensor assembly
# ===========================================================================
def make_batch(records: Sequence[Mapping[str, Any]], indices: Sequence[int], device: str):
    torch = _torch()
    iu0_np, iu1_np = sym_indices(K_DICT)
    iu0 = torch.as_tensor(iu0_np, dtype=torch.long, device=device)
    iu1 = torch.as_tensor(iu1_np, dtype=torch.long, device=device)
    Xs = []
    graphs = []
    ys = []
    off = 0
    for gi in indices:
        rec = records[gi]
        X = torch.as_tensor(np.asarray(rec["X"], dtype=np.float32), device=device)
        n = X.shape[0]
        Rint, Rb, Rgeo = relation_matrices(rec)
        graphs.append(
            {
                "slice": slice(off, off + n),
                "Rint": torch.as_tensor(Rint, device=device),
                "Rb": [torch.as_tensor(r, device=device) for r in Rb],
                "Rgeo": torch.as_tensor(Rgeo, device=device),
                "n": n,
            }
        )
        Xs.append(X)
        ys.append(float(rec.get("y", 0.0)))
        off += n
    return {
        "X_all": torch.cat(Xs, dim=0),
        "graphs": graphs,
        "iu0": iu0,
        "iu1": iu1,
        "y": torch.as_tensor(ys, dtype=torch.float32, device=device),
    }


def fixed_shuffle_perms(records: Sequence[Mapping[str, Any]], indices: Sequence[int],
                        seed: int, device: str):
    torch = _torch()
    perms = []
    for gi in indices:
        n = int(records[gi]["n"])
        rng = np.random.default_rng((int(seed), int(gi)))
        perm = rng.permutation(n)
        if n > 1 and np.array_equal(perm, np.arange(n)):
            perm = np.roll(perm, 1)
        perms.append(torch.as_tensor(perm, dtype=torch.long, device=device))
    return perms


def evaluate_mae(model, records, indices, device, *, shuffle=False, seed=0, batch=64) -> float:
    torch = _torch()
    model.eval()
    errs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            chunk = list(indices[start : start + batch])
            b = make_batch(records, chunk, device)
            perms = fixed_shuffle_perms(records, chunk, seed, device) if shuffle else None
            pred = model(b, shuffle_perms=perms)
            errs.append((pred - b["y"]).abs().cpu().numpy())
    return float(np.concatenate(errs).mean())


@dataclass
class TrainResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    train_history: list[dict[str, Any]] = field(default_factory=list)
    state_soup: Any = None
    lam_rec: float | None = None


def train_model(
    model,
    records_train,
    records_dev,
    train_indices,
    dev_indices,
    device: str,
    *,
    seed: int = 0,
    task_weight: bool = True,
    shuffle_train: bool = False,
    shuffle_seed: int = 0,
    s: int = SPARSITY,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch: int = BATCH,
    lr: float = LR,
    wd: float = WD,
    clip: float = CLIP,
    calibrate_rec: bool = False,
    log=print,
) -> TrainResult:
    torch = _torch()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    lam = None
    if calibrate_rec:
        # frozen calibration: detached initial task / reconstruction value ratio
        cal_idx = list(train_indices[: min(512, len(train_indices))])
        b = make_batch(records_train, cal_idx, device)
        with torch.no_grad():
            C = model.encode(b["X_all"])
            rec = ((b["X_all"] - C @ model.D.t()) ** 2).sum(dim=1) / (
                (b["X_all"] ** 2).sum(dim=1) + EPS
            )
            l_rec = float(rec.mean())
            pred = model(b)
            l_task = float((pred - b["y"]).abs().mean())
        lam = l_task / (l_rec + 1e-12)
        log(f"  calibrated lambda_rec={lam:.6f} (task={l_task:.6f} rec={l_rec:.6f})")

    best = math.inf
    best_epoch = -1
    best_state = None
    top: list[tuple[float, int, Any]] = []
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(train_indices))
        ep_loss = 0.0
        ep_n = 0
        for start in range(0, len(order), batch):
            sel = [train_indices[i] for i in order[start : start + batch]]
            b = make_batch(records_train, sel, device)
            perms = (
                fixed_shuffle_perms(records_train, sel, shuffle_seed, device)
                if shuffle_train
                else None
            )
            opt.zero_grad(set_to_none=True)
            pred = model(b, shuffle_perms=perms)
            task = (pred - b["y"]).abs().mean()
            loss = task
            if lam is not None:
                C = model.encode(b["X_all"], s=s)
                rec = ((b["X_all"] - C @ model.D.t()) ** 2).sum(dim=1) / (
                    (b["X_all"] ** 2).sum(dim=1) + EPS
                )
                loss = task + lam * rec.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip
            )
            opt.step()
            model.renormalize_()
            ep_loss += float(loss.detach()) * len(sel)
            ep_n += len(sel)
        dev_mae = evaluate_mae(
            model, records_dev, dev_indices, device, shuffle=shuffle_train,
            seed=shuffle_seed,
        )
        history.append({"epoch": epoch, "train_loss": ep_loss / max(ep_n, 1), "valid": dev_mae})
        log(f"  epoch={epoch:03d} train={ep_loss / max(ep_n, 1):.6f} valid={dev_mae:.6f} best@{best_epoch}")
        if dev_mae < best - 1e-9:
            best = dev_mae
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if len(top) < TOP_K_SOUP or dev_mae < max(t[0] for t in top):
            top.append((dev_mae, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}))
            top.sort(key=lambda t: t[0])
            top = top[:TOP_K_SOUP]
        if stale >= int(patience):
            log(f"  early stop at epoch {epoch}")
            break

    soup = None
    soup_mae = None
    members: list[int] = []
    if top:
        members = [e for _, e, _ in top]
        keys = top[0][2].keys()
        soup = {k: sum(st[k].float() for _, _, st in top) / len(top) for k in keys}
        soup_mae = _eval_state(model, soup, records_dev, dev_indices, device,
                               shuffle=shuffle_train, shuffle_seed=shuffle_seed)
    model.load_state_dict(best_state if best_state is not None else model.state_dict())
    return TrainResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members,
        train_history=history,
        state_soup=soup,
        lam_rec=lam,
    )


def _eval_state(model, state, records, indices, device, *, shuffle=False, shuffle_seed=0,
                batch=64) -> float:
    torch = _torch()
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    val = evaluate_mae(model, records, indices, device, shuffle=shuffle, seed=shuffle_seed,
                       batch=batch)
    model.load_state_dict(backup)
    return val


# ===========================================================================
# Gate 0 helpers
# ===========================================================================
def permute_mol_ids(mol, rng: np.random.Generator):
    """Random atom relabelling -> new Mol + the atom permutation used."""
    from tracks.ksvd.code.run_aiom_representation_audit import (  # type: ignore
        Mol,
        permute_mol,
    )

    perm_atom = rng.permutation(int(mol.n))
    perm_bond = rng.permutation(len(mol.bonds))
    return permute_mol(mol, perm_atom, perm_bond), perm_atom, perm_bond


def support_jaccard(a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> float:
    sa = set(np.flatnonzero(np.abs(a) > tol).tolist())
    sb = set(np.flatnonzero(np.abs(b) > tol).tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def coefficient_entropy(C: np.ndarray) -> np.ndarray:
    A = np.abs(np.asarray(C, dtype=np.float64))
    p = A / (A.sum(axis=1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=1)
