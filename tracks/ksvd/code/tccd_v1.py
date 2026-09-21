#!/usr/bin/env python
"""TCCD-v1 — Frozen Composition -> End-to-End Dictionary: shared library.

Pre-registration: ``tracks/ksvd/notes/tccd_v1_preregistration.md`` (frozen).
Study ``zinc-context-gap``; canonical PyG ZINC ``subset=True`` official
**train 10000 / valid 1000**; **official test is never loaded**.

TCCD-v1 **reuses the frozen TCCD-v0 dictionary** ``D0`` (K=64, s=8) and the
cached canonical patch records.  It **never refits K-SVD**.  This module adds
only:

* a one-shot frozen ``OMP(x_v; D0, s=8)`` encoder (cached),
* whole-graph BAG / REL / REL-SHUFFLE composition features (cached),
* a lightweight shared linear reader (same protocol for every arm),
* gate-decision helpers with the frozen TCCD-v1 thresholds.

All patch coordinates, relations, layout, sparsity and reader semantics are
inherited unchanged from ``tccd_v0``.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_VERSION = "tccd_v1"

RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tccd_v1"
CACHE_DIR = RESULTS_DIR / "cache"
V0_RESULTS = REPO_ROOT / "tracks/ksvd/results/tccd_v0"
FROZEN_DICT_PATH = V0_RESULTS / "dictionary_gate1.pkl"

# --- frozen TCCD-v1 decision thresholds (pre-registration §6-§10) -----------
GATE_A_PASS = 0.015
GATE_A_FAIL = 0.005
GATE_A_AMBIG_MEAN_PASS = 0.010
GATE_B_PASS = 0.010
GATE_B_AMBIG_MEAN_PASS = 0.005
GATE_C_MARGIN = 0.005
GATE_D_STRONG = 0.005
GATE_D_COMPETITIVE = 0.015
CANONICAL_GPU1_BASELINE = 0.119818
PRIMARY_SEED = 0
PAIRED_SEED = 1

# reader protocol == the repository's canonical lightweight regression protocol
READER_MAX_EPOCHS = T.MAX_EPOCHS
READER_PATIENCE = T.PATIENCE
READER_BATCH = T.BATCH
READER_LR = T.LR
READER_WD = T.WD
READER_CLIP = T.CLIP
READER_SOUP = T.TOP_K_SOUP

N_REL = 5  # R_int, R_b1, R_b2, R_b3, R_geo


# ===========================================================================
# frozen dictionary provenance
# ===========================================================================
def load_frozen_dictionary(path: Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the TCCD-v0 learned ``D0``.  There is no code path that refits it."""
    p = Path(path) if path is not None else FROZEN_DICT_PATH
    if not p.exists():
        raise SystemExit(
            f"frozen TCCD-v0 dictionary not found at {p}; TCCD-v1 must not refit K-SVD "
            "— restore the artefact (results/tccd_v0/dictionary_gate1.pkl) instead."
        )
    with p.open("rb") as fh:
        obj = pickle.load(fh)
    D = np.asarray(obj["D"], dtype=np.float64)
    return D, obj


def dictionary_fingerprint(D: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(D, dtype=np.float64)).tobytes()).hexdigest()


V0_TRAIN_CACHE = V0_RESULTS / "cache" / "tccd_v0_train_r2_M14_A21_B3_n10000.pkl"


def frozen_layout() -> T.PatchLayout:
    """The exact TCCD-v0 layout (M=14, A=21, B=3, F=714) that ``D0`` lives in."""
    _, obj = load_frozen_dictionary()
    d = obj["layout"]
    return T.PatchLayout(
        capacity=int(d["capacity"]), n_atom=int(d["n_atom"]), n_bond=int(d["n_bond"])
    )


def load_train_records(path: Path | None = None):
    """Load the cached canonical TCCD-v0 official-train records (no recomputation)."""
    p = Path(path) if path is not None else V0_TRAIN_CACHE
    obj = T.load_records(p)
    if obj is None:
        raise SystemExit(
            f"canonical TCCD-v0 train records not found at {p}; these are a reused "
            "artefact and must not be rebuilt by refitting K-SVD."
        )
    return obj["records"], obj.get("meta", {})


# ===========================================================================
# one-shot frozen OMP encode (never updates D)
# ===========================================================================
CODES_PATH = CACHE_DIR / "codes_D0.pkl"


def encode_all_codes(
    records: Sequence[Mapping[str, Any]],
    D: np.ndarray,
    s: int = T.SPARSITY,
    chunk: int = 16384,
    log=print,
) -> list[np.ndarray]:
    """One-shot frozen ``OMP(x_v; D, s=8)`` for every recorded patch."""
    t0 = time.time()
    X = np.concatenate([np.asarray(r["X"], dtype=np.float32) for r in records], axis=0)
    log(f"[codes] OMP-encoding {X.shape[0]} patches (F={X.shape[1]}) with frozen D0, one shot")
    C = T.omp_codes(np.asarray(D, dtype=np.float64), X, s=s, chunk=chunk)
    out: list[np.ndarray] = []
    off = 0
    for r in records:
        n = int(r["n"])
        out.append(np.asarray(C[off : off + n], dtype=np.float32))
        off += n
    log(f"[codes] done in {time.time() - t0:.0f}s")
    return out


def load_or_build_codes(
    records: Sequence[Mapping[str, Any]],
    D: np.ndarray,
    s: int = T.SPARSITY,
    chunk: int = 16384,
    log=print,
) -> list[np.ndarray]:
    fp = dictionary_fingerprint(D)
    n_patches = int(sum(int(r["n"]) for r in records))
    if CODES_PATH.exists():
        with CODES_PATH.open("rb") as fh:
            obj = pickle.load(fh)
        if (
            obj.get("D_fp") == fp
            and obj.get("n_graphs") == len(records)
            and obj.get("n_patches") == n_patches
            and obj.get("s") == int(s)
        ):
            log(f"[codes] reused cached frozen codes ({CODES_PATH.name})")
            return obj["codes"]
    codes = encode_all_codes(records, D, s=s, chunk=chunk, log=log)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CODES_PATH.open("wb") as fh:
        pickle.dump(
            {
                "codes": codes,
                "D_fp": fp,
                "s": int(s),
                "n_graphs": len(records),
                "n_patches": n_patches,
                "protocol": PROTOCOL_VERSION,
            },
            fh,
            protocol=4,
        )
    log(f"[codes] cached -> {CODES_PATH}")
    return codes


# ===========================================================================
# whole-graph composition features (frozen, cached)
# ===========================================================================
def rel_operator_list(rec: Mapping[str, Any]) -> list[np.ndarray]:
    """Frozen 5-relation operator list: [R_int, R_b1..R_b3, R_geo]."""
    Rint, Rb, Rgeo = T.relation_matrices(rec)
    return [Rint, *Rb, Rgeo]


def compose_feature(
    C: np.ndarray,
    ops: Sequence[np.ndarray],
    iu0: np.ndarray,
    iu1: np.ndarray,
    perm: np.ndarray | None = None,
) -> np.ndarray:
    """``h_G`` for one graph from its code matrix; optional fixed row permutation.

    ``perm=None`` -> REL (and BAG is just the ``m_G`` prefix).
    ``perm`` given -> REL-SHUFFLE: same code multiset, same ``R_G``, rows permuted.
    """
    C = np.asarray(C, dtype=np.float32)
    if perm is not None:
        C = C[np.asarray(perm, dtype=np.int64)]
    parts = [C.sum(axis=0)]
    for R in ops:
        M = C.T @ np.asarray(R, dtype=np.float32) @ C
        parts.append(M[iu0, iu1])
    return np.concatenate(parts).astype(np.float32)


def shuffle_perm(n: int, seed: int, graph_index: int) -> np.ndarray:
    """Exact copy of ``tccd_v0.fixed_shuffle_perms`` RNG semantics (numpy side)."""
    rng = np.random.default_rng((int(seed), int(graph_index)))
    perm = rng.permutation(int(n))
    if n > 1 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return perm.astype(np.int64)


def bag_slice(K: int = T.K_DICT) -> slice:
    return slice(0, K)


def build_bag_rel(
    codes: Sequence[np.ndarray],
    records: Sequence[Mapping[str, Any]],
    log=print,
) -> tuple[np.ndarray, np.ndarray]:
    K = T.K_DICT
    iu0, iu1 = T.sym_indices(K)
    h_len = K + N_REL * len(iu0)
    bag = np.zeros((len(records), K), dtype=np.float32)
    rel = np.zeros((len(records), h_len), dtype=np.float32)
    for i, rec in enumerate(records):
        h = compose_feature(codes[i], rel_operator_list(rec), iu0, iu1)
        rel[i] = h
        bag[i] = h[:K]
    log(f"[features] BAG {bag.shape} REL {rel.shape}")
    return bag, rel


def build_shuffle(
    codes: Sequence[np.ndarray],
    records: Sequence[Mapping[str, Any]],
    seed: int,
    log=print,
) -> np.ndarray:
    K = T.K_DICT
    iu0, iu1 = T.sym_indices(K)
    h_len = K + N_REL * len(iu0)
    out = np.zeros((len(records), h_len), dtype=np.float32)
    for i, rec in enumerate(records):
        n = int(rec["n"])
        perm = shuffle_perm(n, seed, i)
        out[i] = compose_feature(codes[i], rel_operator_list(rec), iu0, iu1, perm=perm)
    log(f"[features] REL-SHUFFLE(seed={seed}) {out.shape}")
    return out


COMMON_PATH = CACHE_DIR / "features_bag_rel.pkl"
SHUFFLE_PATH = CACHE_DIR / "features_shuffle_seed{seed}.pkl"


def load_or_build_common(codes, records, log=print):
    n_patches = int(sum(int(r["n"]) for r in records))
    if COMMON_PATH.exists():
        with COMMON_PATH.open("rb") as fh:
            obj = pickle.load(fh)
        if obj.get("n_graphs") == len(records) and obj.get("n_patches") == n_patches:
            log(f"[features] reused cached BAG/REL ({COMMON_PATH.name})")
            return obj["bag"], obj["rel"]
    bag, rel = build_bag_rel(codes, records, log=log)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with COMMON_PATH.open("wb") as fh:
        pickle.dump({"bag": bag, "rel": rel, "n_graphs": len(records), "n_patches": n_patches}, fh, protocol=4)
    return bag, rel


def load_or_build_shuffle(codes, records, seed, log=print):
    p = Path(str(SHUFFLE_PATH).format(seed=int(seed)))
    n_patches = int(sum(int(r["n"]) for r in records))
    if p.exists():
        with p.open("rb") as fh:
            obj = pickle.load(fh)
        if obj.get("n_graphs") == len(records) and obj.get("n_patches") == n_patches:
            log(f"[features] reused cached REL-SHUFFLE ({p.name})")
            return obj["shuffle"]
    shuf = build_shuffle(codes, records, seed, log=log)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump({"shuffle": shuf, "seed": int(seed), "n_graphs": len(records), "n_patches": n_patches}, fh, protocol=4)
    return shuf


# ===========================================================================
# lightweight shared linear reader (identical protocol for every Gate A arm)
# ===========================================================================
# The pre-registered reader is a *linear regression* (brief §6 permits
# "linear regression / linear head").  We use ridge regression on standardized
# features with a single, data-independent regularization
# ``alpha = n_train`` applied identically to every arm, so that REL and
# REL-SHUFFLE are matched in feature dimension, parameter count and
# regularization.  Internal-dev is never used for any selection; the full
# alpha curve is reported only as a robustness diagnostic.
RIDGE_ALPHA_RULE = "n_train"
RIDGE_DIAG_GRID = (10.0, 100.0, 1000.0, 10000.0)


@dataclass
class ReaderResult:
    best_valid: float
    alpha: float
    grid_valid: dict[str, float]
    wall_s: float = 0.0
    peak_mem_mb: float | None = None


def _standardize(X_train: np.ndarray, X: np.ndarray):
    mu = np.asarray(X_train, dtype=np.float64).mean(axis=0, keepdims=True)
    sd = np.asarray(X_train, dtype=np.float64).std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (np.asarray(X, dtype=np.float64) - mu) / sd


def ridge_reader(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    *,
    alpha: float | None = None,
    grid: Sequence[float] = RIDGE_DIAG_GRID,
    log=print,
) -> ReaderResult:
    """Frozen shared linear reader: ridge regression on standardized features."""
    from sklearn.linear_model import Ridge  # type: ignore

    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_dev = np.asarray(y_dev, dtype=np.float64).reshape(-1)
    n_train = int(np.asarray(X_train).shape[0])
    a = float(n_train) if alpha is None else float(alpha)
    Xtr = _standardize(X_train, X_train)
    Xdv = _standardize(X_train, X_dev)
    t0 = time.time()
    model = Ridge(alpha=a).fit(Xtr, y_train)
    mae = float(np.abs(model.predict(Xdv) - y_dev).mean())
    grid_valid: dict[str, float] = {}
    for g in grid:
        gm = Ridge(alpha=float(g)).fit(Xtr, y_train)
        grid_valid[f"{float(g):g}"] = float(np.abs(gm.predict(Xdv) - y_dev).mean())
    wall = time.time() - t0
    log(f"  [reader] ridge alpha={a:g} n_train={n_train} dev_MAE={mae:.6f} "
        f"grid={ {k: round(v, 5) for k, v in grid_valid.items()} } ({wall:.1f}s)")
    return ReaderResult(best_valid=mae, alpha=a, grid_valid=grid_valid, wall_s=wall)


# ===========================================================================
# gate decision helpers (frozen thresholds)
# ===========================================================================
def gate_a_verdict(delta: float, seed: int, seed0_delta: float | None = None) -> str:
    if int(seed) == PRIMARY_SEED:
        if delta >= GATE_A_PASS:
            return "PASS"
        if delta < GATE_A_FAIL:
            return "FAIL"
        return "AMBIGUOUS_NEEDS_SEED1"
    if seed0_delta is None:
        raise ValueError("seed-1 verdict requires the seed-0 delta")
    mean = 0.5 * (float(seed0_delta) + float(delta))
    return "PASS" if mean >= GATE_A_AMBIG_MEAN_PASS else "FAIL"


def gate_b_verdict(delta: float, seed: int, seed0_delta: float | None = None) -> str:
    if int(seed) == PRIMARY_SEED:
        if delta >= GATE_B_PASS:
            return "PASS"
        if delta <= 0.0:
            return "FAIL"
        return "AMBIGUOUS_NEEDS_SEED1"
    if seed0_delta is None:
        raise ValueError("seed-1 verdict requires the seed-0 delta")
    mean = 0.5 * (float(seed0_delta) + float(delta))
    return "PASS" if mean >= GATE_B_AMBIG_MEAN_PASS else "FAIL"


def gate_d_band(delta_abs: float) -> str:
    if delta_abs <= 0.0:
        return "VERY_STRONG"
    if delta_abs <= GATE_D_STRONG:
        return "STRONG"
    if delta_abs <= GATE_D_COMPETITIVE:
        return "COMPETITIVE"
    return "NOT_VIABLE"


def verdict_to_exit(verdict: str) -> int:
    return 0 if verdict == "PASS" else 1
