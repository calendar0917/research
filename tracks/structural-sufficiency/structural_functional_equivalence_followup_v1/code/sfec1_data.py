"""Canonical cell/context/pair tables, fits and paired statistics for SFEC v1.

The parent cell de-duplication and context masks are reused read-only.  The one
scientific correction implemented here is canonical pair orientation: every
pair row satisfies A < B and delta = y(B) - y(A).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import sfec1_common as C


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


# ---------------------------------------------------------------------------
# raw tables
# ---------------------------------------------------------------------------


def load_cells(prefix: Path) -> dict[str, np.ndarray]:
    z = np.load(Path(prefix).with_suffix(".npz"))
    return {k: z[k] for k in z.files}


def load_universe(prefix: Path) -> dict[str, np.ndarray]:
    z = np.load(Path(str(prefix) + ".universe.npz"))
    return {k: z[k] for k in z.files}


def load_C(prefix: Path, columns=None) -> dict[str, np.ndarray]:
    tab = pq.read_table(Path(prefix).with_suffix(".C.parquet"), columns=columns)
    return {name: tab[name].to_numpy(zero_copy_only=False) for name in tab.column_names}


def load_S(prefix: Path) -> dict[str, np.ndarray]:
    tab = pq.read_table(Path(prefix).with_suffix(".S.parquet"))
    return {name: tab[name].to_numpy(zero_copy_only=False) for name in tab.column_names}


def load_M(prefix: Path, columns=None) -> dict[str, np.ndarray]:
    tab = pq.read_table(Path(prefix).with_suffix(".M.parquet"), columns=columns)
    return {name: tab[name].to_numpy(zero_copy_only=False) for name in tab.column_names}


# ---------------------------------------------------------------------------
# context tables
# ---------------------------------------------------------------------------


@dataclass
class Contexts:
    C: np.ndarray
    off: np.ndarray
    k: np.ndarray
    S: np.ndarray
    y: np.ndarray
    n: np.ndarray
    sd: np.ndarray
    cell_key: np.ndarray
    cell_C: np.ndarray
    cell_S: np.ndarray
    cell_b: np.ndarray
    cell_mol_first: np.ndarray
    cell_mol_n: np.ndarray

    @property
    def n_ctx(self) -> int:
        return int(len(self.C))


def _empty_contexts() -> Contexts:
    return Contexts(
        np.zeros(0, np.int32), np.zeros(1, np.int64), np.zeros(0, np.int32),
        np.zeros(0, np.int32), np.zeros(0, np.float64), np.zeros(0, np.int32),
        np.zeros(0, np.float64), np.zeros(0, np.int64), np.zeros(0, np.int32),
        np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.int32),
        np.zeros(0, np.int32),
    )


def dedup_cells(
    cells: dict[str, np.ndarray],
    mask: np.ndarray,
    labels: np.ndarray,
    n_s: int,
) -> dict[str, np.ndarray]:
    """One row per (C,S,b), mean over distinct molecule identities."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {
            "key": np.zeros(0, np.int64), "C": np.zeros(0, np.int32),
            "S": np.zeros(0, np.int32), "b": np.zeros(0, np.int8),
            "y": np.zeros(0, np.float64), "n": np.zeros(0, np.int32),
            "sd": np.zeros(0, np.float64), "mol_first": np.zeros(0, np.int32),
        }
    Cid = cells["C_id"][idx].astype(np.int64)
    Sid = cells["S_id"][idx].astype(np.int64)
    B = cells["bond_type"][idx].astype(np.int64)
    mol = cells["mol_id"][idx].astype(np.int64)
    y = np.asarray(labels, dtype=np.float64)[cells["row_id"][idx]]
    key = (Cid * int(n_s) + Sid) * 8 + B
    # unique (cell, molecule identity), then one label per such pair
    order0 = np.lexsort((mol, key))
    key_sorted = key[order0]
    mol_sorted = mol[order0]
    y_sorted = y[order0]
    first = np.r_[True, (key_sorted[1:] != key_sorted[:-1]) | (mol_sorted[1:] != mol_sorted[:-1])]
    uk = key_sorted[first]
    um = mol_sorted[first]
    uy = y_sorted[first]
    order = np.argsort(uk, kind="stable")
    uk = uk[order]
    um = um[order]
    uy = uy[order]
    starts = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    ends = np.r_[starts[1:], len(uk)]
    cnt = ends - starts
    sums = np.add.reduceat(uy, starts)
    mean = sums / cnt
    sq = np.add.reduceat(uy * uy, starts)
    var = np.maximum(sq / cnt - mean * mean, 0.0)
    sd = np.sqrt(var)
    cell_key = uk[starts]
    cell_C = (cell_key // (int(n_s) * 8)).astype(np.int64)
    rem = cell_key % (int(n_s) * 8)
    cell_S = (rem // 8).astype(np.int64)
    cell_b = (rem % 8).astype(np.int64)
    return {
        "key": cell_key.astype(np.int64),
        "C": cell_C.astype(np.int32),
        "S": cell_S.astype(np.int32),
        "b": cell_b.astype(np.int8),
        "y": mean.astype(np.float64),
        "n": cnt.astype(np.int32),
        "sd": sd.astype(np.float64),
        "mol_first": um[starts].astype(np.int32),
    }


def build_contexts(
    cells: dict[str, np.ndarray],
    mask: np.ndarray,
    labels: np.ndarray,
    n_s: int,
    primary: np.ndarray,
    rank: np.ndarray,
) -> Contexts:
    de = dedup_cells(cells, mask, labels, n_s)
    keep = primary[de["S"]]
    de = {k: v[keep] for k, v in de.items()}
    if len(de["C"]) == 0:
        return _empty_contexts()
    order = np.argsort(de["C"], kind="stable")
    Cs = de["C"][order]
    starts = np.flatnonzero(np.r_[True, Cs[1:] != Cs[:-1]])
    ends = np.r_[starts[1:], len(Cs)]
    k = ends - starts
    off = np.r_[0, np.cumsum(k)].astype(np.int64)
    return Contexts(
        C=Cs[starts].astype(np.int32),
        off=off,
        k=k.astype(np.int32),
        S=de["S"][order].astype(np.int32),
        y=de["y"][order].astype(np.float64),
        n=de["n"][order].astype(np.int32),
        sd=de["sd"][order].astype(np.float64),
        cell_key=de["key"][order].astype(np.int64),
        cell_C=de["C"][order].astype(np.int32),
        cell_S=de["S"][order].astype(np.int32),
        cell_b=de["b"][order].astype(np.int8),
        cell_mol_first=de["mol_first"][order].astype(np.int32),
        cell_mol_n=de["n"][order].astype(np.int32),
    )


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------


def build_pairs(ctx: Contexts, rank: np.ndarray, *, canonical: bool) -> dict[str, np.ndarray]:
    """All unordered cell pairs inside contexts with k>=2.

    canonical=True enforces A<B and delta=y(B)-y(A).  canonical=False
    reproduces the parent stored-order orientation for the correction table.
    """
    rows_ctx: list[np.ndarray] = []
    rows_A: list[np.ndarray] = []
    rows_B: list[np.ndarray] = []
    rows_delta: list[np.ndarray] = []
    rows_pi: list[np.ndarray] = []
    rows_pj: list[np.ndarray] = []
    for i in range(ctx.n_ctx):
        a, b = int(ctx.off[i]), int(ctx.off[i + 1])
        if b - a < 2:
            continue
        r = rank[ctx.S[a:b]].astype(np.int64)
        y = ctx.y[a:b]
        ii, jj = np.triu_indices(len(r), 1)
        if canonical:
            left = r[ii]
            right = r[jj]
            rows_A.append(np.minimum(left, right).astype(np.int16))
            rows_B.append(np.maximum(left, right).astype(np.int16))
            rows_delta.append(
                np.where(left < right, y[jj] - y[ii], y[ii] - y[jj]).astype(np.float64)
            )
        else:
            rows_A.append(r[ii].astype(np.int16))
            rows_B.append(r[jj].astype(np.int16))
            rows_delta.append((y[jj] - y[ii]).astype(np.float64))
        rows_ctx.append(np.full(len(ii), i, dtype=np.int32))
        rows_pi.append((a + ii).astype(np.int64))
        rows_pj.append((a + jj).astype(np.int64))
    if not rows_ctx:
        z = np.zeros(0)
        return {
            "ctx": z.astype(np.int32), "A": z.astype(np.int16),
            "B": z.astype(np.int16), "delta": z.astype(np.float64),
            "pos_i": z.astype(np.int64), "pos_j": z.astype(np.int64),
        }
    return {
        "ctx": np.concatenate(rows_ctx),
        "A": np.concatenate(rows_A),
        "B": np.concatenate(rows_B),
        "delta": np.concatenate(rows_delta),
        "pos_i": np.concatenate(rows_pi),
        "pos_j": np.concatenate(rows_pj),
    }


def pair_key(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A.astype(np.int64) * 64 + B.astype(np.int64)


def pair_stats(pairs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-canonical-pair discovery summaries (one row per pair type)."""
    key = pair_key(pairs["A"], pairs["B"])
    d = pairs["delta"]
    order = np.argsort(key, kind="stable")
    key = key[order]
    A = pairs["A"][order].astype(np.int64)
    B = pairs["B"][order].astype(np.int64)
    d = d[order]
    if len(key) == 0:
        z = np.zeros(0)
        return {
            "pair_key": z.astype(np.int64), "A": z.astype(np.int16),
            "B": z.astype(np.int16), "n": z.astype(np.int32),
            "median": z.astype(np.float64), "mean": z.astype(np.float64),
            "mad": z.astype(np.float64), "p80_abs_dev": z.astype(np.float64),
            "sd": z.astype(np.float64), "sign_pos": z.astype(np.float64),
            "q10": z.astype(np.float64), "q90": z.astype(np.float64),
        }
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    n = (ends - starts).astype(np.int32)
    out = {
        "pair_key": key[starts].astype(np.int64),
        "A": A[starts].astype(np.int16),
        "B": B[starts].astype(np.int16),
        "n": n,
        "median": np.zeros(len(starts), np.float64),
        "mean": np.zeros(len(starts), np.float64),
        "mad": np.zeros(len(starts), np.float64),
        "p80_abs_dev": np.zeros(len(starts), np.float64),
        "sd": np.zeros(len(starts), np.float64),
        "sign_pos": np.zeros(len(starts), np.float64),
        "q10": np.zeros(len(starts), np.float64),
        "q90": np.zeros(len(starts), np.float64),
    }
    for i, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        x = d[a:b]
        med = float(np.median(x))
        out["median"][i] = med
        out["mean"][i] = float(np.mean(x))
        out["mad"][i] = float(np.median(np.abs(x - med)))
        out["p80_abs_dev"][i] = float(np.quantile(np.abs(x - med), 0.80))
        out["sd"][i] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        out["sign_pos"][i] = float(np.mean(x > 0))
        out["q10"][i] = float(np.quantile(x, 0.10))
        out["q90"][i] = float(np.quantile(x, 0.90))
    return out


def lookup_pair(stats: dict[str, np.ndarray], key: np.ndarray, field: str) -> np.ndarray:
    key = np.asarray(key, dtype=np.int64)
    pos = np.searchsorted(stats["pair_key"], key)
    out = np.full(len(key), np.nan, dtype=np.float64)
    ok = pos < len(stats["pair_key"])
    ok[ok] = stats["pair_key"][pos[ok]] == key[ok]
    out[ok] = stats[field][pos[ok]]
    return out


def lookup_pair_count(stats: dict[str, np.ndarray], key: np.ndarray) -> np.ndarray:
    key = np.asarray(key, dtype=np.int64)
    pos = np.searchsorted(stats["pair_key"], key)
    out = np.zeros(len(key), dtype=np.int64)
    ok = pos < len(stats["pair_key"])
    ok[ok] = stats["pair_key"][pos[ok]] == key[ok]
    out[ok] = stats["n"][pos[ok]]
    return out


def pair_count_matrix(pairs: dict[str, np.ndarray]) -> np.ndarray:
    mat = np.zeros((64, 64), dtype=np.int32)
    np.add.at(mat, (pairs["A"].astype(np.int64), pairs["B"].astype(np.int64)), 1)
    return mat + mat.T


# ---------------------------------------------------------------------------
# triple (A,B,q) discovery statistics
# ---------------------------------------------------------------------------


def triple_stats(
    pairs: dict[str, np.ndarray], q_ctx: np.ndarray, n_q: int
) -> dict[str, np.ndarray]:
    """Median/mean/P80 absolute deviation per (A,B,q) triple."""
    q = q_ctx[pairs["ctx"]].astype(np.int64)
    key = (pair_key(pairs["A"], pairs["B"]) * int(n_q) + q)
    d = pairs["delta"]
    order = np.argsort(key, kind="stable")
    key = key[order]
    d = d[order]
    if len(key) == 0:
        z = np.zeros(0)
        return {"key": z.astype(np.int64), "n": z.astype(np.int32),
                "median": z.astype(np.float64), "mean": z.astype(np.float64),
                "p80_abs_dev": z.astype(np.float64), "sd": z.astype(np.float64)}
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    n = (ends - starts).astype(np.int32)
    out = {
        "key": key[starts].astype(np.int64),
        "n": n,
        "median": np.zeros(len(starts), np.float64),
        "mean": np.zeros(len(starts), np.float64),
        "p80_abs_dev": np.zeros(len(starts), np.float64),
        "sd": np.zeros(len(starts), np.float64),
    }
    for i, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        x = d[a:b]
        med = float(np.median(x))
        out["median"][i] = med
        out["mean"][i] = float(np.mean(x))
        out["p80_abs_dev"][i] = float(np.quantile(np.abs(x - med), 0.80))
        out["sd"][i] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return out


def lookup_triple(
    stats: dict[str, np.ndarray],
    padj: np.ndarray,
    B: np.ndarray,
    q: np.ndarray,
    n_q: int,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    key = (pair_key(np.asarray(padj, np.int64), np.asarray(B, np.int64)) * int(n_q)
           + np.asarray(q, np.int64))
    pos = np.searchsorted(stats["key"], key)
    out = np.full(len(key), np.nan, dtype=np.float64)
    cnt = np.zeros(len(key), dtype=np.int64)
    ok = pos < len(stats["key"])
    ok[ok] = stats["key"][pos[ok]] == key[ok]
    out[ok] = stats[field][pos[ok]]
    cnt[ok] = stats["n"][pos[ok]]
    return out, cnt


# ---------------------------------------------------------------------------
# design / fits
# ---------------------------------------------------------------------------


@dataclass
class Design:
    """Discovery contexts with k>=2, in rank space."""

    ctx_idx: np.ndarray          # global context index
    off: np.ndarray
    O: np.ndarray                # rank of S per cell
    y: np.ndarray                # cell label
    q: np.ndarray                # q id per cell's context (repeated)
    k: np.ndarray
    cell_pos: np.ndarray         # global position in Contexts arrays

    @property
    def n_ctx(self) -> int:
        return int(len(self.ctx_idx))


def make_design(ctx: Contexts, rank: np.ndarray, q_ctx: np.ndarray) -> Design:
    sel = np.flatnonzero(ctx.k >= 2).astype(np.int64)
    off = [0]
    O_parts = []
    y_parts = []
    q_parts = []
    pos_parts = []
    k_parts = []
    for ci in sel.tolist():
        a, b = int(ctx.off[ci]), int(ctx.off[ci + 1])
        O_parts.append(rank[ctx.S[a:b]].astype(np.int64))
        y_parts.append(ctx.y[a:b].astype(np.float64))
        q_parts.append(np.full(b - a, int(q_ctx[ci]), dtype=np.int64))
        pos_parts.append(np.arange(a, b, dtype=np.int64))
        k_parts.append(b - a)
        off.append(off[-1] + (b - a))
    return Design(
        ctx_idx=sel,
        off=np.asarray(off, dtype=np.int64),
        O=np.concatenate(O_parts) if O_parts else np.zeros(0, np.int64),
        y=np.concatenate(y_parts) if y_parts else np.zeros(0, np.float64),
        q=np.concatenate(q_parts) if q_parts else np.zeros(0, np.int64),
        k=np.asarray(k_parts, dtype=np.int64),
        cell_pos=np.concatenate(pos_parts) if pos_parts else np.zeros(0, np.int64),
    )


def _filter_design(
    design: Design, keep: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filtered (ctx local index, O list, y list, q list, k, cell_pos)."""
    if keep is None:
        return (
            np.arange(design.n_ctx, dtype=np.int64), design.off, design.O,
            design.y, design.q, design.k, design.cell_pos,
        )
    out_off = [0]
    O_parts = []
    y_parts = []
    q_parts = []
    pos_parts = []
    k_parts = []
    ctx_parts = []
    for j in range(design.n_ctx):
        a, b = int(design.off[j]), int(design.off[j + 1])
        m = keep[design.cell_pos[a:b]]
        kk = int(m.sum())
        if kk < 2:
            continue
        O_parts.append(design.O[a:b][m])
        y_parts.append(design.y[a:b][m])
        q_parts.append(design.q[a:b][m])
        pos_parts.append(design.cell_pos[a:b][m])
        ctx_parts.append(j)
        k_parts.append(kk)
        out_off.append(out_off[-1] + kk)
    if not O_parts:
        z = np.zeros(0)
        return (np.zeros(0, np.int64), np.zeros(1, np.int64), z.astype(np.int64),
                z.astype(np.float64), z.astype(np.int64), z.astype(np.int64),
                z.astype(np.int64))
    return (
        np.asarray(ctx_parts, dtype=np.int64),
        np.asarray(out_off, dtype=np.int64),
        np.concatenate(O_parts),
        np.concatenate(y_parts),
        np.concatenate(q_parts),
        np.asarray(k_parts, dtype=np.int64),
        np.concatenate(pos_parts),
    )


def _components(L: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    comp = -np.ones(n, dtype=np.int16)
    adj = (np.abs(L) > 1e-12) & ~np.eye(n, dtype=bool)
    cid = 0
    for start in range(n):
        if comp[start] != -1 or not np.any(adj[start]):
            continue
        stack = [start]
        comp[start] = cid
        members = []
        while stack:
            v = stack.pop()
            members.append(v)
            for w in np.flatnonzero(adj[v]).tolist():
                if comp[w] == -1:
                    comp[w] = cid
                    stack.append(w)
        cid += 1
    return comp


def _solve_shared(L: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bvec = np.zeros(64, dtype=np.float64)
    comp = _components(L)
    adj = (np.abs(L) > 1e-12) & ~np.eye(64, dtype=bool)
    for cid in range(int(comp.max()) + 1 if np.any(comp >= 0) else 0):
        ix = np.flatnonzero(comp == cid)
        if len(ix) < 2:
            continue
        sub = L[np.ix_(ix, ix)]
        sol = np.linalg.lstsq(sub, rhs[ix], rcond=1e-12)[0]
        sol = sol - sol.mean()
        bvec[ix] = sol
    return bvec, comp


def fit_shared(design: Design, keep: np.ndarray | None = None) -> dict[str, Any]:
    _, off, O, y, _q, k, _pos = _filter_design(design, keep)
    L = np.zeros((64, 64), dtype=np.float64)
    rhs = np.zeros(64, dtype=np.float64)
    if len(k):
        # k == 2 exactly: vectorised
        sel2 = np.flatnonzero(k == 2)
        if len(sel2):
            i1 = O[off[sel2]]
            i2 = O[off[sel2] + 1]
            y1 = y[off[sel2]]
            y2 = y[off[sel2] + 1]
            r1 = y1 - 0.5 * (y1 + y2)
            r2 = y2 - 0.5 * (y1 + y2)
            np.add.at(L, (i1, i1), 0.25)
            np.add.at(L, (i2, i2), 0.25)
            np.add.at(L, (i1, i2), -0.25)
            np.add.at(L, (i2, i1), -0.25)
            np.add.at(rhs, i1, 0.5 * r1)
            np.add.at(rhs, i2, 0.5 * r2)
        # k >= 3: small residual loop
        for j in np.flatnonzero(k >= 3).tolist():
            a, b = int(off[j]), int(off[j + 1])
            OO = O[a:b]
            yy = y[a:b]
            kk = b - a
            yt = yy - yy.mean()
            L[np.ix_(OO, OO)] += (np.eye(kk) - 1.0 / kk) / kk
            rhs[OO] += yt / kk
    bvec, comp = _solve_shared(L, rhs)
    grad = L @ bvec - rhs
    return {
        "b": bvec,
        "comp": comp,
        "L": L,
        "rhs": rhs,
        "n_contexts_used": int(len(k)),
        "n_cells_used": int(len(O)),
        "grad_max_abs": float(np.max(np.abs(grad))) if len(grad) else 0.0,
        "residual_sum": float(np.sum(grad ** 2)),
    }


def _local_normal_equations(
    q: np.ndarray, O: np.ndarray, y: np.ndarray, bvec: np.ndarray,
    off: np.ndarray, k: np.ndarray, n_q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Lq = np.zeros((n_q, 64, 64), dtype=np.float64)
    rhs = np.zeros((n_q, 64), dtype=np.float64)
    n_used = np.zeros(n_q, dtype=np.int64)
    n_cells = np.zeros(n_q, dtype=np.int64)
    if not len(k):
        return Lq, rhs, n_used, n_cells
    q_ctx = q[off[:-1]]
    for qq in range(n_q):
        idx = np.flatnonzero(q_ctx == qq)
        if len(idx) == 0:
            continue
        n_used[qq] = len(idx)
        n_cells[qq] = int(k[idx].sum())
        sel2 = idx[k[idx] == 2]
        if len(sel2):
            i1 = O[off[sel2]]
            i2 = O[off[sel2] + 1]
            y1 = y[off[sel2]]
            y2 = y[off[sel2] + 1]
            mu = 0.5 * (y1 + y2)
            bt = 0.5 * (bvec[i1] + bvec[i2])
            r1 = (y1 - mu) - (bvec[i1] - bt)
            r2 = (y2 - mu) - (bvec[i2] - bt)
            np.add.at(Lq[qq], (i1, i1), 0.25)
            np.add.at(Lq[qq], (i2, i2), 0.25)
            np.add.at(Lq[qq], (i1, i2), -0.25)
            np.add.at(Lq[qq], (i2, i1), -0.25)
            np.add.at(rhs[qq], i1, 0.5 * r1)
            np.add.at(rhs[qq], i2, 0.5 * r2)
        for j in idx[k[idx] >= 3].tolist():
            a, b = int(off[j]), int(off[j + 1])
            OO = O[a:b]
            yy = y[a:b]
            kk = b - a
            yt = yy - yy.mean()
            bt = bvec[OO] - bvec[OO].mean()
            rr = yt - bt
            Lq[qq][np.ix_(OO, OO)] += (np.eye(kk) - 1.0 / kk) / kk
            rhs[qq][OO] += rr / kk
    return Lq, rhs, n_used, n_cells


def fit_local(
    design: Design,
    bvec: np.ndarray,
    *,
    n_q: int,
    lam: float,
    min_contexts: int,
    keep: np.ndarray | None = None,
    q_override: np.ndarray | None = None,
) -> dict[str, Any]:
    _, off, O, y, q, k, _pos = _filter_design(design, keep)
    if q_override is not None:
        # q_override is per full discovery context; design stores q per cell
        q = q_override[design.ctx_idx][_context_cell_first(design)]
    Lq, rhs, n_used, n_cells = _local_normal_equations(q, O, y, bvec, off, k, n_q)
    d = np.zeros((n_q, 64), dtype=np.float64)
    comp = -np.ones((n_q, 64), dtype=np.int16)
    supported = np.zeros(n_q, dtype=bool)
    for qq in range(n_q):
        if n_used[qq] < min_contexts:
            continue
        supported[qq] = True
        d[qq] = np.linalg.solve(Lq[qq] + lam * np.eye(64), rhs[qq])
        comp[qq] = _components(Lq[qq])
    return {
        "d": d,
        "comp": comp,
        "supported": supported,
        "n_contexts": n_used,
        "n_cells": n_cells,
        "L": Lq,
    }


def _context_cell_first(design: Design) -> np.ndarray:
    """A 0/1 mask selecting the first cell of each filtered design context."""
    return design.off[:-1]


# The helper above is not enough for q_override after filtering.  A dedicated
# filtered design builder keeps the context id per row instead.
def fit_local_ctx(
    design: Design,
    bvec: np.ndarray,
    *,
    n_q: int,
    lam: float,
    min_contexts: int,
    keep: np.ndarray | None = None,
    q_ctx_all: np.ndarray | None = None,
) -> dict[str, Any]:
    """LOCAL fit with the q of each filtered context taken from q_ctx_all."""
    ctx_sel, off, O, y, q, k, pos = _filter_design(design, keep)
    if q_ctx_all is not None:
        # one q per filtered context; expand to cells
        q_ctx_f = q_ctx_all[design.ctx_idx[ctx_sel]]
        q = np.repeat(q_ctx_f, k)
    Lq, rhs, n_used, n_cells = _local_normal_equations(q, O, y, bvec, off, k, n_q)
    d = np.zeros((n_q, 64), dtype=np.float64)
    comp = -np.ones((n_q, 64), dtype=np.int16)
    supported = np.zeros(n_q, dtype=bool)
    for qq in range(n_q):
        if n_used[qq] < min_contexts:
            continue
        supported[qq] = True
        d[qq] = np.linalg.solve(Lq[qq] + lam * np.eye(64), rhs[qq])
        comp[qq] = _components(Lq[qq])
    return {
        "d": d,
        "comp": comp,
        "supported": supported,
        "n_contexts": n_used,
        "n_cells": n_cells,
        "L": Lq,
    }


def predict_local(
    pair_ctx: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    q_ctx: np.ndarray,
    bvec: np.ndarray,
    fit: dict[str, Any],
) -> dict[str, np.ndarray]:
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    q = np.asarray(q_ctx, dtype=np.int64)[np.asarray(pair_ctx, dtype=np.int64)]
    shared = bvec[B] - bvec[A]
    ca = fit["comp"][q, A]
    cb = fit["comp"][q, B]
    enabled = fit["supported"][q] & (ca >= 0) & (ca == cb)
    corr = fit["d"][q, B] - fit["d"][q, A]
    pred = shared + np.where(enabled, corr, 0.0)
    return {"shared": shared, "pred": pred, "enabled": enabled, "correction": corr, "q": q}


def make_null_q(ctx: Contexts, q_ctx: np.ndarray, seed: int) -> np.ndarray:
    """Parent rule: within context row degree strata, permute q with seed 0."""
    return make_null_q_masked(ctx, q_ctx, ctx.k.astype(np.int64), seed)


def make_null_q_masked(
    ctx: Contexts, q_ctx: np.ndarray, k_masked: np.ndarray, seed: int
) -> np.ndarray:
    """Permute q within (context masked degree) strata with fixed seed."""
    rng = np.random.default_rng(int(seed))
    out = q_ctx.copy()
    k_ctx = np.asarray(k_masked, dtype=np.int64)
    for kval in np.unique(k_ctx).tolist():
        idx = np.flatnonzero(k_ctx == kval)
        if len(idx) > 1:
            out[idx] = rng.permutation(q_ctx[idx])
    return out


# ---------------------------------------------------------------------------
# metrics and bootstrap
# ---------------------------------------------------------------------------


def compact_ctx(row_ctx: np.ndarray) -> tuple[np.ndarray, int]:
    if len(row_ctx) == 0:
        return row_ctx.astype(np.int64), 0
    u, inv = np.unique(row_ctx.astype(np.int64), return_inverse=True)
    return inv.astype(np.int64), int(len(u))


def basic_metric(delta: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, np.float64) - np.asarray(delta, np.float64)
    ae = np.abs(err)
    n = len(ae)
    if n == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "sign_accuracy": float("nan")}
    m = np.abs(delta) > 1e-12
    sign = float(np.mean(np.sign(delta[m]) == np.sign(pred[m]))) if np.any(m) else float("nan")
    return {
        "n": int(n),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
        "sign_accuracy": sign,
    }


def context_equal_mae(row_ctx: np.ndarray, ae: np.ndarray, n_ctx: int) -> float:
    cnt = np.bincount(row_ctx, minlength=n_ctx).astype(np.float64)
    num = np.bincount(row_ctx, weights=ae, minlength=n_ctx)
    ok = cnt > 0
    return float(np.mean(num[ok] / cnt[ok])) if np.any(ok) else float("nan")


def pair_macro_mae(pair_idx: np.ndarray, ae: np.ndarray) -> float:
    if len(ae) == 0:
        return float("nan")
    sums = np.bincount(pair_idx, weights=ae)
    cnt = np.bincount(pair_idx)
    ok = cnt > 0
    return float(np.mean(sums[ok] / cnt[ok]))


def weighted_metric(
    delta: np.ndarray,
    pred: np.ndarray,
    row_ctx: np.ndarray,
    pair_idx: np.ndarray,
    n_ctx: int,
) -> dict[str, Any]:
    ae = np.abs(np.asarray(pred, np.float64) - np.asarray(delta, np.float64))
    return {
        "observation": basic_metric(delta, pred),
        "context_equal_mae": context_equal_mae(row_ctx, ae, n_ctx),
        "pair_type_equal_mae": pair_macro_mae(pair_idx, ae),
    }


def _bootstrap_counts(rng: np.random.Generator, n_ctx: int, reps: int) -> np.ndarray:
    return rng.integers(0, n_ctx, size=(reps, n_ctx))


def bootstrap_obs_context(
    delta: np.ndarray,
    preds: dict[str, np.ndarray],
    row_ctx: np.ndarray,
    n_ctx: int,
    *,
    context_equal: bool,
    reps: int,
    seed: int,
    comparisons: list[tuple[str, str]],
) -> dict[str, Any]:
    n = len(delta)
    if n == 0:
        return {"models": {}, "diffs": {}}
    cnt = np.bincount(row_ctx, minlength=n_ctx).astype(np.float64)
    if context_equal:
        w = 1.0 / np.maximum(cnt[row_ctx], 1.0)
        den = np.bincount(row_ctx, weights=w, minlength=n_ctx)
    else:
        w = None
        den = cnt
    nums = {}
    for name, pred in preds.items():
        ae = np.abs(np.asarray(pred, np.float64) - delta)
        nums[name] = np.bincount(row_ctx, weights=(ae if w is None else w * ae), minlength=n_ctx)
    rng = np.random.default_rng(seed)
    mc = _bootstrap_counts(rng, n_ctx, reps)
    all_vals = {name: np.empty(reps) for name in preds}
    for r in range(reps):
        counts = np.bincount(mc[r], minlength=n_ctx).astype(np.float64)
        dsum = max(float(counts @ den), 1e-12)
        for name in preds:
            all_vals[name][r] = float(counts @ nums[name]) / dsum
    point = {name: float(nums[name].sum() / max(den.sum(), 1e-12)) for name in preds}
    models = {
        name: {
            "point": point[name],
            "ci": [float(np.quantile(all_vals[name], 0.025)),
                   float(np.quantile(all_vals[name], 0.975))],
        }
        for name in preds
    }
    diffs = {}
    for a, b in comparisons:
        vals = all_vals[a] - all_vals[b]
        diffs[f"{a}_minus_{b}"] = {
            "point": point[a] - point[b],
            "ci": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        }
    return {"models": models, "diffs": diffs}


def bootstrap_pair_macro(
    delta: np.ndarray,
    preds: dict[str, np.ndarray],
    row_ctx: np.ndarray,
    pair_idx: np.ndarray,
    n_ctx: int,
    *,
    reps: int,
    seed: int,
    comparisons: list[tuple[str, str]],
) -> dict[str, Any]:
    n = len(delta)
    if n == 0:
        return {"models": {}, "diffs": {}}
    pair_idx = np.asarray(pair_idx, dtype=np.int64)
    n_pairs = int(pair_idx.max()) + 1 if len(pair_idx) else 0
    errs = {name: np.abs(np.asarray(pred, np.float64) - delta) for name, pred in preds.items()}
    point = {name: pair_macro_mae(pair_idx, errs[name]) for name in preds}
    rng = np.random.default_rng(seed)
    mc = _bootstrap_counts(rng, n_ctx, reps)
    vals = {name: np.empty(reps) for name in preds}
    row_counts = np.bincount(row_ctx, minlength=n_ctx).astype(np.float64)
    for r in range(reps):
        c = np.bincount(mc[r], minlength=n_ctx).astype(np.float64)
        w = c[row_ctx]
        pc = np.bincount(pair_idx, weights=w, minlength=n_pairs)
        ok = pc > 0
        for name in preds:
            ps = np.bincount(pair_idx, weights=w * errs[name], minlength=n_pairs)
            vals[name][r] = float(np.mean(ps[ok] / pc[ok]))
    models = {
        name: {
            "point": point[name],
            "ci": [float(np.quantile(vals[name], 0.025)),
                   float(np.quantile(vals[name], 0.975))],
        }
        for name in preds
    }
    diffs = {}
    for a, b in comparisons:
        v = vals[a] - vals[b]
        diffs[f"{a}_minus_{b}"] = {
            "point": point[a] - point[b],
            "ci": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
        }
    return {"models": models, "diffs": diffs}


# ---------------------------------------------------------------------------
# clean subset helpers
# ---------------------------------------------------------------------------


def molecule_clean_flags(
    cells: dict[str, np.ndarray], M: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """No explicitly encoded stereo, and no duplicate identity row."""
    keys = [str(x) for x in M["mol_key"].tolist()]
    stereo = np.asarray([("@" in k) or ("/" in k) or ("\\" in k) for k in keys], dtype=bool)
    n_mol = len(keys)
    mol = cells["mol_id"].astype(np.int64)
    row = cells["row_id"].astype(np.int64)
    order = np.lexsort((row, mol))
    ms = mol[order]
    rs = row[order]
    starts = np.flatnonzero(np.r_[True, ms[1:] != ms[:-1]])
    ends = np.r_[starts[1:], len(ms)]
    n_rows = np.zeros(n_mol, dtype=np.int32)
    for a, b in zip(starts.tolist(), ends.tolist()):
        n_rows[int(ms[a])] = len(np.unique(rs[a:b]))
    dup = n_rows > 1
    return {
        "stereo": stereo,
        "duplicate_identity": dup,
        "n_distinct_rows": n_rows,
        "clean": (~stereo) & (~dup),
    }


def pair_clean_mask(
    ctx: Contexts,
    pairs: dict[str, np.ndarray],
    mol_clean: np.ndarray,
) -> np.ndarray:
    first = ctx.cell_mol_first.astype(np.int64)
    cnt = ctx.cell_mol_n.astype(np.int64)
    cell_clean = (cnt == 1) & mol_clean[np.maximum(first, 0)]
    return cell_clean[pairs["pos_i"]] & cell_clean[pairs["pos_j"]]


# ---------------------------------------------------------------------------
# convenience bundle
# ---------------------------------------------------------------------------


@dataclass
class Bundle:
    prefix: Path
    cells: dict[str, np.ndarray]
    uni: dict[str, np.ndarray]
    S: dict[str, np.ndarray]
    C: dict[str, np.ndarray]
    M: dict[str, np.ndarray]
    labels: np.ndarray
    primary: np.ndarray
    rank: np.ndarray
    ctx_disc: Contexts
    ctx_eval: Contexts
    pairs_disc: dict[str, np.ndarray]
    pairs_eval: dict[str, np.ndarray]


def load_bundle(cfg: dict[str, Any] | None = None) -> Bundle:
    if cfg is None:
        cfg = C.load_config()
    prefix = (C.PARENT_DIR / Path(cfg["datasets"]["pcqm"]["train_prefix"])).resolve()
    cells = load_cells(prefix)
    uni = load_universe(prefix)
    S = load_S(prefix)
    Ctab = load_C(prefix, columns=["C_id", "q_id", "nC", "scaffold_id", "hash_mod", "hash64"])
    M = load_M(prefix, columns=["mol_id", "mol_key", "split_id", "row_id", "n_atoms", "scaffold_id"])
    labels = C.load_pcqm_labels_cached()
    primary = uni["primary"]
    rank = uni["rank"]
    n_s = len(S["S_id"])
    train = cells["split_id"] == 0
    disc_mask = uni["discovery_mask"] & train
    eval_mask = uni["primary_evaluation_cells"] & train
    ctx_disc = build_contexts(cells, disc_mask, labels, n_s, primary, rank)
    ctx_eval = build_contexts(cells, eval_mask, labels, n_s, primary, rank)
    pairs_disc = build_pairs(ctx_disc, rank, canonical=True)
    pairs_eval = build_pairs(ctx_eval, rank, canonical=True)
    return Bundle(
        prefix=prefix, cells=cells, uni=uni, S=S, C=Ctab, M=M, labels=labels,
        primary=primary, rank=rank, ctx_disc=ctx_disc, ctx_eval=ctx_eval,
        pairs_disc=pairs_disc, pairs_eval=pairs_eval,
    )


def rank_to_sid(rank: np.ndarray) -> np.ndarray:
    out = -np.ones(64, dtype=np.int64)
    for sid, r in enumerate(rank.tolist()):
        if 0 <= int(r) < 64:
            out[int(r)] = int(sid)
    return out


def rank_key(rank: np.ndarray, S: dict[str, np.ndarray]) -> dict[int, str]:
    out = {}
    for sid, r in enumerate(rank.tolist()):
        if int(r) >= 0:
            out[int(r)] = str(S["S_key"][sid])
    return out
