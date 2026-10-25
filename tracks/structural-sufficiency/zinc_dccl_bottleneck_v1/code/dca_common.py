"""Shared loaders and pre-fixed, target-free similarity machinery for the
DCC-L bottleneck audit (phase A).

Nothing here reads official test.  Valid labels are used only to compute errors;
train labels are used for neighbour-label statistics and the kNN reference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SUFF = ROUND.parent
REPO = SUFF.parents[1]
V1 = SUFF / "zinc_arch_discovery_v1"
PERF = SUFF / "zinc_performance_search_v1"
AUD = SUFF / "zinc_generalization_audit_v1"
ANALYSIS = ROUND / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

DCC_SEEDS = (0, 1, 2, 3)
K_NN = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_cache(split: str) -> dict:
    assert split in ("train", "valid"), "audit is train/valid only"
    return torch.load(V1 / "cache" / f"ad_{split}.pt", map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- graphs
def molecule_graphs(cache: dict):
    """Return a list of (atom_types: int array, bonds: list[(u, v, bond_type)])."""
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)  # [n_bonds, 2]
    n = len(ap) - 1
    out = []
    for i in range(n):
        a0, a1 = int(ap[i]), int(ap[i + 1])
        b0, b1 = int(bp[i]), int(bp[i + 1])
        atoms = at[a0:a1]
        bonds = []
        for e in range(b0, b1):
            # bond_index stores per-molecule local atom indices
            u = int(bi[e, 0])
            v = int(bi[e, 1])
            bonds.append((u, v, int(bt[e])))
        out.append((atoms, bonds))
    return out


def wl_features(graphs, h: int = 2):
    """Weisfeiler-Lehman node-label multiset per molecule, levels 0..h."""
    per_mol: list[list[str]] = []
    for atoms, bonds in graphs:
        nbrs: list[list[tuple[int, int]]] = [[] for _ in range(len(atoms))]
        for u, v, t in bonds:
            nbrs[u].append((t, v))
            nbrs[v].append((t, u))
        labels = [f"a{int(x)}" for x in atoms]
        levels = [list(labels)]
        for _ in range(h):
            new = []
            for i, lab in enumerate(labels):
                key = lab + "|" + ",".join(sorted(f"{t}:{labels[j]}" for t, j in nbrs[i]))
                new.append(key)
            labels = new
            levels.append(list(labels))
        feats: list[str] = []
        for lev, labs in enumerate(levels):
            feats.extend(f"{lev}:{x}" for x in labs)
        per_mol.append(feats)
    return per_mol


def build_vocab(per_mol):
    vocab = sorted({f for feats in per_mol for f in feats})
    return {f: i for i, f in enumerate(vocab)}


def vectorize(per_mol, idx, *, normalize: bool = True):
    rows, cols, vals = [], [], []
    for r, feats in enumerate(per_mol):
        for f, c in Counter(feats).items():
            j = idx.get(f)
            if j is None:  # subtree pattern absent from the reference split
                continue
            rows.append(r)
            cols.append(j)
            vals.append(float(c))
    mat = sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(len(per_mol), len(idx)),
    )
    if normalize:
        norms = np.sqrt(np.asarray(mat.multiply(mat).sum(1)).reshape(-1))
        norms[norms == 0] = 1.0
        mat = sparse.diags(1.0 / norms) @ mat
    return mat.tocsr()


def radius2_signature_sets(graphs):
    """Exact typed radius-2 rooted signatures (same construction as the audit)."""
    sets: list[set[str]] = []
    per_atom: list[list[str]] = []
    for atoms, bonds in graphs:
        nbrs: list[list[tuple[int, int]]] = [[] for _ in range(len(atoms))]
        for u, v, t in bonds:
            nbrs[u].append((t, v))
            nbrs[v].append((t, u))
        one = []
        for i in range(len(atoms)):
            one.append(
                f"r1({int(atoms[i])};" + ",".join(sorted(f"{t}-{int(atoms[j])}" for t, j in nbrs[i])) + ")"
            )
        two = []
        for i in range(len(atoms)):
            two.append(
                f"r2({one[i]};" + ",".join(sorted(f"{t}-{one[j]}" for t, j in nbrs[i])) + ")"
            )
        sets.append(set(two))
        per_atom.append(two)
    return sets, per_atom


def binary_tanimoto(query_sets, ref_sets):
    vocab = sorted({s for ss in list(query_sets) + list(ref_sets) for s in ss})
    idx = {s: i for i, s in enumerate(vocab)}

    def build(sets):
        rows, cols = [], []
        for r, ss in enumerate(sets):
            for s in ss:
                rows.append(r)
                cols.append(idx[s])
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sets), len(vocab)))

    q = build(query_sets)
    r = build(ref_sets)
    inter = np.asarray((q @ r.T).todense(), dtype=np.float64)
    qn = np.asarray(q.sum(1)).reshape(-1, 1)
    rn = np.asarray(r.sum(1)).reshape(-1)
    union = qn + rn[None, :] - inter
    union[union <= 0] = 1.0
    return inter / union


def neighbour_stats(sim: np.ndarray, y_ref: np.ndarray, k: int = K_NN) -> dict:
    """Top-k (by similarity) neighbour support statistics for every query row."""
    n = sim.shape[0]
    out = {key: np.zeros(n) for key in
           ("sim_top1", "sim_topk", "sim_topk_sum", "nn_ystd", "nn_ymad", "nn_yrange",
            "nn_pred", "nn_wpred", "nn_ymean")}
    order = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    for i in range(n):
        j = order[i]
        s = sim[i, j]
        ys = y_ref[j]
        out["sim_top1"][i] = s[0]
        out["sim_topk"][i] = s.mean()
        out["sim_topk_sum"][i] = s.sum()
        out["nn_ystd"][i] = ys.std()
        out["nn_ymad"][i] = np.abs(ys - np.median(ys)).mean()
        out["nn_yrange"][i] = ys.max() - ys.min()
        out["nn_ymean"][i] = ys.mean()
        out["nn_pred"][i] = ys.mean()
        w = s / s.sum() if s.sum() > 0 else np.full(len(s), 1.0 / len(s))
        out["nn_wpred"][i] = float((w * ys).sum())
    return out


# --------------------------------------------------------------------------- errors
def load_dcc(valid_cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, preds[k, n]) for the four frozen DCC-L checkpoints."""
    path = PERF / "analysis" / "per_molecule_valid.csv"
    rows = list(csv.DictReader(open(path)))
    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    pred = np.asarray(
        [[float(r[f"pred_B-R0-online-s{s}"]) for r in rows] for s in DCC_SEEDS],
        dtype=np.float64,
    )
    lab = np.asarray(valid_cache["labels"], dtype=np.float64)
    if not np.allclose(y, lab, atol=1e-6):
        raise RuntimeError("valid row alignment failed")
    return y, pred


def spearman(x, y):
    rx = rank(x)
    ry = rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    s = np.sort(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def partial_spearman(x, y, covars):
    """Spearman of x,y after rank-residualising both on rank covariates."""
    rx = rank(x)
    ry = rank(y)
    Z = np.column_stack([np.ones(len(rx))] + [rank(c) for c in covars])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex = rx - Z @ bx
    ey = ry - Z @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def boot_ci(x, y=None, n_boot=2000, seed=20261025, alpha=0.05):
    """Paired bootstrap CI of mean(x) (or mean(x - y)) over rows."""
    rng = np.random.default_rng(seed)
    v = np.asarray(x, dtype=np.float64) if y is None else np.asarray(x) - np.asarray(y)
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
