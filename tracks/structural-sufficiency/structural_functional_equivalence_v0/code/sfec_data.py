"""Cell de-duplication, context tables and pair tables for SFEC v0."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import sfec_common as C


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


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


@dataclass
class Contexts:
    C: np.ndarray          # context id per context
    off: np.ndarray        # offsets into S/y
    k: np.ndarray          # number of S in context
    S: np.ndarray          # concatenated primary S ids (or ranks)
    y: np.ndarray          # cell mean labels
    n: np.ndarray          # molecules per cell
    sd: np.ndarray         # label SD per cell
    cell_key: np.ndarray   # composite (C,S,b) key per cell
    cell_C: np.ndarray
    cell_S: np.ndarray
    cell_b: np.ndarray

    @property
    def n_ctx(self) -> int:
        return int(len(self.C))


def dedup_cells(
    cells: dict[str, np.ndarray],
    mask: np.ndarray,
    labels: np.ndarray,
    n_s: int,
) -> dict[str, np.ndarray]:
    """One row per (C,S,b) with mean over distinct molecules."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {
            "key": np.zeros(0, np.int64), "C": np.zeros(0, np.int32),
            "S": np.zeros(0, np.int32), "b": np.zeros(0, np.int8),
            "y": np.zeros(0, np.float64), "n": np.zeros(0, np.int32),
            "sd": np.zeros(0, np.float64),
        }
    Cid = cells["C_id"][idx].astype(np.int64)
    Sid = cells["S_id"][idx].astype(np.int64)
    B = cells["bond_type"][idx].astype(np.int64)
    mol = cells["mol_id"][idx].astype(np.int64)
    y = np.asarray(labels, dtype=np.float64)[cells["row_id"][idx]]
    key = (Cid * int(n_s) + Sid) * 8 + B
    # distinct molecule identity inside a cell first
    pair = np.unique(np.stack([key, mol], axis=1), axis=0)
    ukey = pair[:, 0]
    # labels for the unique (cell, molecule) pairs: take the first matching row
    # (same molecule key can occur in duplicate rows with different labels; the
    # first is recorded and the duplicate-label spread is checked separately).
    # Build a lookup from (key,mol) to label using the first occurrence.
    order0 = np.lexsort((mol, key))
    key_sorted = key[order0]
    mol_sorted = mol[order0]
    y_sorted = y[order0]
    first = np.r_[True, (key_sorted[1:] != key_sorted[:-1]) | (mol_sorted[1:] != mol_sorted[:-1])]
    uk = key_sorted[first]
    uy = y_sorted[first]
    # group unique (cell,molecule) rows by key
    order = np.argsort(uk, kind="stable")
    uk = uk[order]
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
    }


def build_contexts(
    cells: dict[str, np.ndarray],
    mask: np.ndarray,
    labels: np.ndarray,
    n_s: int,
    primary: np.ndarray,
    rank: np.ndarray,
) -> Contexts:
    # de-dup cells first
    de = dedup_cells(cells, mask, labels, n_s)
    keep = primary[de["S"]]
    de = {k: v[keep] for k, v in de.items()}
    if len(de["C"]) == 0:
        return Contexts(
            np.zeros(0, np.int32), np.zeros(1, np.int64), np.zeros(0, np.int32),
            np.zeros(0, np.int32), np.zeros(0, np.float64), np.zeros(0, np.int32),
            np.zeros(0, np.float64), np.zeros(0, np.int64), np.zeros(0, np.int32),
            np.zeros(0, np.int32), np.zeros(0, np.int8),
        )
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
        cell_key=de["key"].astype(np.int64),
        cell_C=de["C"].astype(np.int32),
        cell_S=de["S"].astype(np.int32),
        cell_b=de["b"].astype(np.int8),
    )


def build_pairs(ctx: Contexts, rank: np.ndarray) -> dict[str, np.ndarray]:
    """All A<B pair rows inside contexts with k>=2 (rank order is canonical)."""
    rows_ctx = []
    rows_A = []
    rows_B = []
    rows_delta = []
    for i in range(ctx.n_ctx):
        a, b = int(ctx.off[i]), int(ctx.off[i + 1])
        if b - a < 2:
            continue
        r = rank[ctx.S[a:b]].astype(np.int64)
        y = ctx.y[a:b]
        ii, jj = np.triu_indices(len(r), 1)
        rows_ctx.append(np.full(len(ii), i, dtype=np.int32))
        rows_A.append(r[ii].astype(np.int16))
        rows_B.append(r[jj].astype(np.int16))
        rows_delta.append((y[jj] - y[ii]).astype(np.float64))
    if not rows_ctx:
        return {
            "ctx": np.zeros(0, np.int32), "A": np.zeros(0, np.int16),
            "B": np.zeros(0, np.int16), "delta": np.zeros(0, np.float64),
        }
    return {
        "ctx": np.concatenate(rows_ctx),
        "A": np.concatenate(rows_A),
        "B": np.concatenate(rows_B),
        "delta": np.concatenate(rows_delta),
    }


def pair_matrix(pairs: dict[str, np.ndarray], kdim: int = 64) -> np.ndarray:
    mat = np.zeros((kdim, kdim), dtype=np.int32)
    np.add.at(mat, (pairs["A"].astype(np.int64), pairs["B"].astype(np.int64)), 1)
    return mat + mat.T


def pair_stats(
    pairs: dict[str, np.ndarray], kdim: int = 64, max_pairs_for_quantile: int | None = None
) -> dict[str, Any]:
    """Per-pair discovery summaries."""
    A = pairs["A"].astype(np.int64)
    B = pairs["B"].astype(np.int64)
    d = pairs["delta"]
    key = A * kdim + B
    order = np.argsort(key, kind="stable")
    key = key[order]
    A = A[order]
    B = B[order]
    d = d[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(key)] if len(key) else np.zeros(0, np.int64)
    out = {
        "pair_key": key[starts].astype(np.int64),
        "A": A[starts].astype(np.int16),
        "B": B[starts].astype(np.int16),
        "n": (ends - starts).astype(np.int32),
        "median": np.zeros(len(starts), np.float64),
        "mad": np.zeros(len(starts), np.float64),
        "q10": np.zeros(len(starts), np.float64),
        "q90": np.zeros(len(starts), np.float64),
        "p80_abs_dev": np.zeros(len(starts), np.float64),
        "sign_pos": np.zeros(len(starts), np.float64),
        "mean": np.zeros(len(starts), np.float64),
        "sd": np.zeros(len(starts), np.float64),
    }
    for i, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        x = d[a:b]
        med = float(np.median(x))
        out["median"][i] = med
        out["mad"][i] = float(np.median(np.abs(x - med)))
        out["q10"][i] = float(np.quantile(x, 0.10))
        out["q90"][i] = float(np.quantile(x, 0.90))
        out["p80_abs_dev"][i] = float(np.quantile(np.abs(x - med), 0.80))
        out["sign_pos"][i] = float(np.mean(x > 0))
        out["mean"][i] = float(np.mean(x))
        out["sd"][i] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return out


def pair_double_differences(
    pairs: dict[str, np.ndarray], rng: np.random.Generator, max_per_pair: int = 256
) -> dict[str, Any]:
    """|D| distribution over context pairs, sampled per structure pair."""
    A = pairs["A"].astype(np.int64)
    B = pairs["B"].astype(np.int64)
    d = pairs["delta"]
    key = A * 64 + B
    order = np.argsort(key, kind="stable")
    key = key[order]
    d = d[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(key)] if len(key) else np.zeros(0, np.int64)
    samples = []
    n_sampled_pairs = 0
    total_possible = 0
    for a, b in zip(starts.tolist(), ends.tolist()):
        n = b - a
        if n < 2:
            continue
        total_possible += n * (n - 1) // 2
        if n * (n - 1) // 2 <= max_per_pair:
            ii, jj = np.triu_indices(n, 1)
        else:
            n_sampled_pairs += 1
            ii = rng.integers(0, n, size=max_per_pair)
            jj = rng.integers(0, n, size=max_per_pair)
            same = ii == jj
            ii, jj = ii[~same], jj[~same]
        x = d[a:b]
        samples.append(np.abs(x[jj] - x[ii]))
    if not samples:
        return {"abs_D": np.zeros(0), "total_possible": 0, "sampled_pairs": 0,
                "n": 0, "median": None, "p90": None, "mean": None}
    arr = np.concatenate(samples)
    return {
        "abs_D": arr,
        "total_possible": int(total_possible),
        "sampled_pairs": int(n_sampled_pairs),
        "n": int(len(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "mean": float(np.mean(arr)),
    }


def context_bootstrap_sums(
    row_ctx: np.ndarray, abs_err: np.ndarray, n_ctx: int, context_equal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Per-context numerator and denominator for MAE bootstrap."""
    cnt = np.bincount(row_ctx, minlength=n_ctx).astype(np.float64)
    if context_equal:
        w = 1.0 / np.maximum(cnt[row_ctx], 1.0)
        num = np.bincount(row_ctx, weights=w * abs_err, minlength=n_ctx)
        den = np.bincount(row_ctx, weights=w, minlength=n_ctx)
    else:
        num = np.bincount(row_ctx, weights=abs_err, minlength=n_ctx)
        den = cnt
    return num, den


def bootstrap_mae(
    row_ctx: np.ndarray,
    abs_err: np.ndarray,
    n_ctx: int,
    *,
    replicates: int,
    seed: int,
    context_equal: bool = False,
) -> dict[str, Any]:
    num, den = context_bootstrap_sums(row_ctx, abs_err, n_ctx, context_equal)
    rng = np.random.default_rng(seed)
    out = np.empty(replicates, dtype=np.float64)
    n = n_ctx
    for r in range(replicates):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        out[r] = (counts @ num) / max(counts @ den, 1e-12)
    return {
        "point": float(num.sum() / max(den.sum(), 1e-12)),
        "lo": float(np.quantile(out, 0.025)),
        "hi": float(np.quantile(out, 0.975)),
        "mean": float(out.mean()),
    }


def bootstrap_diff(
    row_ctx: np.ndarray,
    err_a: np.ndarray,
    err_b: np.ndarray,
    n_ctx: int,
    *,
    replicates: int,
    seed: int,
    context_equal: bool = False,
) -> dict[str, Any]:
    """Bootstrap of MAE(a)-MAE(b); positive means b is better."""
    num_a, den_a = context_bootstrap_sums(row_ctx, err_a, n_ctx, context_equal)
    num_b, den_b = context_bootstrap_sums(row_ctx, err_b, n_ctx, context_equal)
    rng = np.random.default_rng(seed)
    out = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        counts = np.bincount(rng.integers(0, n_ctx, size=n_ctx), minlength=n_ctx).astype(np.float64)
        ma = (counts @ num_a) / max(counts @ den_a, 1e-12)
        mb = (counts @ num_b) / max(counts @ den_b, 1e-12)
        out[r] = ma - mb
    return {
        "diff": float(num_a.sum() / max(den_a.sum(), 1e-12) - num_b.sum() / max(den_b.sum(), 1e-12)),
        "lo": float(np.quantile(out, 0.025)),
        "hi": float(np.quantile(out, 0.975)),
        "replicates": out,
    }