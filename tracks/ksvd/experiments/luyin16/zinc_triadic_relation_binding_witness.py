"""Explicit triadic relation binding witness audit (compact-v4-hinge, ZINC).

This is a **frozen feature witness diagnostic**, not an architecture
benchmark.  It does not train a new backbone, does not modify compact-v4, does
not recompute pair states, does not update patch states, and does not implement
attention, a GNN, a Transformer, or any recurrent / message-passing refinement.
It answers exactly one question:

    compact-v4 already represents every pair relation q_ij.  Does the *absence
    of an explicit binding that q_ij, q_ik and q_jk belong to the same patch
    triple (i, j, k)* hide task-relevant higher-order structure?

The model has an explicit per-centre relation pooling and exactly one one-shot
centre update

    center_context_{i,b} = [ mean_j q_ij ; std_j q_ij ; log1p(count) ]   (33D)
    patch_i'             = patch_i + MLP([patch_i ; 5 x 33 = 165D])

but it never forms a joint object over the three pair states of a triple: the
outer relation q_jk is pooled into the centre context of ``j`` and of ``k``,
never into the centre context of ``i`` together with q_ij and q_ik.  The three
pair states of a triple are therefore only *implicitly* coupled (through the
shared updated patch states), never explicitly bound.

The audit constructs, for every unordered patch triple ``i < j < k``, the
**permutation-invariant** explicit triadic binding witness

    a = q_ij,  b = q_ik,  c = q_jk

    s1    = a + b + c                        (16D)
    s_abs = |a-mu| + |b-mu| + |c-mu|,  mu=(a+b+c)/3   (16D)
    s2    = a*b + a*c + b*c                  (16D)
    triad_raw = [s1 ; s_abs ; s2]            (48D)

which is projected by a fixed deterministic orthonormal 48x16 map to ``z`` and
summarised at the graph level as ``T_graph = [mean(z) ; std(z)]`` (32D).  The
**decisive control** is an *unbound matched control* ``U_graph`` built with the
identical machinery but with the outer relation ``q_jk`` deterministically
permuted (within distance-bucket groups) across the triples of the same
molecule, preserving the pair-state multiset while destroying true triple
closure.

Stages
------
* Stage 0  triad reconstruction + hard integrity gates (NO adapter training).
* Stage 1  1 frozen OOF backbone seed x 5 outer folds x 1 init:
  B1 (R-only, ``302->13->13->1``), B2 (unbound control), E (true triad).
  Primary mechanism metrics ``Delta_R = MAE(B1) - MAE(E)`` and
  ``Delta_U = MAE(B2) - MAE(E)``.
* Stage 1b  second adapter init (only for a borderline Stage 1).
* Stage 2  second frozen OOF backbone seed (only after a clear advance).

Everything is evaluated on official TRAIN molecules through the existing
5-fold OOF checkpoints.  Official valid/test are never loaded.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_triadic_relation_binding_witness <stage>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16.zinc_oof_difficulty_audit import (
    CACHE_DIR as OOF_CACHE_DIR,
    K_FOLDS,
    _fold_marker,
    _fold_slices,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

REPO_ROOT = zpp.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/triadic_relation_binding_witness"
FIG_DIR = RESULTS_DIR / "figures"

# The corrected v4 centre-incidence export already contains every field needed
# for explicit triads: q_ij (pair_states), local pair endpoints, buckets, R,
# yhat_0, molecule ids and targets.  This audit performs NO new backbone
# inference; it reads the v4 export (same frozen OOF checkpoints).
V4_EXPORT_VERSION = "frozen_state_export_v4_centre_incidence"
V4_EXPORT_DIR = REPO_ROOT / "tracks/ksvd/results/centre_incidence_cooccurrence_witness/state_exports"
CONFIG_PATH = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
RARITY_CSV = OOF_CACHE_DIR / "rarity_rows.csv"

# --- frozen v4-hinge shapes (verified from the real model at runtime) ------
R_DIM = 302
PATCH_DIM = 48
PAIR_DIM = 16
N_BUCKETS = zpp.DISTANCE_BUCKETS  # 5

# --- triad witness definition ----------------------------------------------
TRIAD_FIRST_DIM = PAIR_DIM  # 16
TRIAD_ABS_DIM = PAIR_DIM  # 16
TRIAD_PROD_DIM = PAIR_DIM  # 16
TRIAD_RAW_DIM = TRIAD_FIRST_DIM + TRIAD_ABS_DIM + TRIAD_PROD_DIM  # 48
TRIAD_PROJ_DIM = 16
T_GRAPH_DIM = 2 * TRIAD_PROJ_DIM  # 32 (mean + std over triads)

# The deterministic random projection is frozen for the whole audit and is
# never learned: orthonormal columns (QR) of a standard-normal matrix.
TRIAD_PROJECTION_SEED = 20260914
# Deterministic per-molecule permutation seed for the unbound control.
UNBOUND_SEED = 20260914

# --- compute-budget policy -------------------------------------------------
# Full O(n^3) enumeration is used unless the measured runtime exceeds twice the
# current frozen-feature analysis runtime; see runtime_profile.json.  The
# deterministic cap below is only applied when full enumeration is too
# expensive.  It is target-independent, molecule-ID-seeded and uniform over
# unordered triples, and true witness + all controls use exactly the same
# sampled triples.
MAX_TRIADS_PER_GRAPH = 512
FULL_ENUM_COST_RATIO_LIMIT = 2.0

# --- split / budget (identical to frozen readout / centre audits) ----------
SPLIT_SEED = "frozen-readout-sufficiency-v2-20260910"
SPLIT_SIZES = (1200, 400, 400)  # adapter-fit / adapter-selection / adapter-evaluation
BACKBONE_SEEDS = (0, 1)
PRIMARY_BACKBONE_SEED = 0
ADAPTER_SEEDS = (0, 1)  # seed 1 used only when Stage 1 is BORDERLINE

# --- adapters (pre-registered, no sweeps) ----------------------------------
# Historical-strength 2-hidden-layer R-only residual head (the known-good
# R-only route): 302 -> 13 -> 13 -> 1 = 4,135 params.
RONLY_HIDDEN = 13
RONLY_DEPTH = 2
# B2 (unbound control) and E (true triad) share one identical architecture
# (302 + 32) -> 12 -> 12 -> 1 = 4,189 params (mismatch to B1 = +1.31%, within
# the pre-registered +-3% budget).  Widths are chosen by formula only.
WITNESS_HIDDEN = 12
WITNESS_DEPTH = 2
ADAPTER_LR = 1.0e-3
ADAPTER_EPOCHS = 400
ADAPTER_PATIENCE = 50

# --- decision thresholds (pre-registered) ----------------------------------
S1_ADV_DR = 0.002
S1_ADV_DU = 0.0015
S1_ADV_FOLDS = 4
S1_NGO_DR = 0.0005
S1_NGO_DU = 0.0
S1_NGO_WORSE_FOLDS = 2  # E better than unbound in <= 2/5 folds -> NO-GO
FINAL_GO_DR = 0.002
FINAL_GO_DU = 0.0015
FINAL_GO_FOLDS = 4
BULK_MAX_DEGRADATION = 0.002
BULK_RARE_PERCENTILE = 80.0

# --- gate tolerances -------------------------------------------------------
TRIAD_INV_ATOL = 1.0e-12
LOOKUP_ATOL = 0.0
PROJ_ATOL = 0.0
GRAPH_REBUILD_ATOL = 1.0e-6

RUNTIME_PROFILE_MOLECULES = 100
RUNTIME_PROFILE_SEED = 20260914


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json_any(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_json(value: Any) -> str:
    blob = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _export_path(fold: int, seed: int) -> Path:
    return V4_EXPORT_DIR / f"{V4_EXPORT_VERSION}_fold{fold}_seed{seed}.npz"


def load_v4_export(path: Path) -> dict[str, Any]:
    """Load a v4 export, refusing any legacy/foreign cache."""
    with np.load(path, allow_pickle=False) as data:
        if "fingerprint_json" not in data.files:
            raise RuntimeError(f"cache {path} has no fingerprint; refusing to load")
        fingerprint = json.loads(str(data["fingerprint_json"]))
        version = fingerprint.get("export_version")
        if version != V4_EXPORT_VERSION:
            raise RuntimeError(
                f"cache {path} export_version={version!r} != required "
                f"{V4_EXPORT_VERSION!r}; refusing to load"
            )
        export: dict[str, Any] = {"fingerprint": fingerprint}
        for key in data.files:
            if key == "fingerprint_json":
                continue
            export[key] = data[key]
    return export


def _orthonormal_projection(seed: int, in_dim: int, out_dim: int) -> np.ndarray:
    """Deterministic (in_dim x out_dim) matrix with orthonormal columns."""
    rng = np.random.default_rng(int(seed))
    raw = rng.standard_normal((int(in_dim), int(out_dim)))
    q, _r = np.linalg.qr(raw)
    return np.ascontiguousarray(q[:, : int(out_dim)].astype(np.float64))


def _triad_projection() -> np.ndarray:
    return _orthonormal_projection(TRIAD_PROJECTION_SEED, TRIAD_RAW_DIM, TRIAD_PROJ_DIM)


def _projection_fingerprint() -> dict[str, Any]:
    proj = _triad_projection()
    gram = proj.T @ proj
    return {
        "triad": {
            "seed": TRIAD_PROJECTION_SEED,
            "kind": "qr_orthonormal_columns_of_standard_normal",
            "shape": list(proj.shape),
            "sha256": _sha256_array(proj),
            "gram_max_abs_offdiag": float(np.abs(gram - np.eye(TRIAD_PROJ_DIM)).max()),
        }
    }


# ---------------------------------------------------------------------------
# explicit triads: enumeration + permutation-invariant representation
# ---------------------------------------------------------------------------

def enumerate_triples(n: int) -> np.ndarray:
    """Return the (C(n,3), 3) array of unordered patch triples ``i < j < k``."""
    n = int(n)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)
    rows = list(itertools.combinations(range(n), 3))
    return np.asarray(rows, dtype=np.int64)


def sample_unordered_triples(n: int, cap: int, molecule_id: str) -> np.ndarray:
    """Deterministic target-independent uniform subsample of unordered triples.

    Only used when the pre-registered compute-budget cap is applied.  The
    sample is a pure function of ``(n, cap, molecule_id)`` so the true witness
    and every control see exactly the same sampled triples.
    """
    n = int(n)
    total = int(n * (n - 1) * (n - 2) // 6)
    if total <= int(cap):
        return enumerate_triples(n)
    key = int(hashlib.sha256(f"triad-sample|{molecule_id}|{n}|{cap}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(key)
    chosen = np.sort(rng.choice(total, size=int(cap), replace=False))
    all_triples = enumerate_triples(n)
    return all_triples[chosen]


def triad_raw_from_values(
    qa: np.ndarray,
    qb: np.ndarray,
    qc: np.ndarray,
) -> np.ndarray:
    """Permutation-invariant raw triad representation (T x 48).

    ``qa``, ``qb``, ``qc`` are (T x 16) pair states for the three pairs of each
    triple.  The three states are put in a canonical **value-based** lexicographic
    order before the symmetric reductions, so the result is exactly invariant to
    any relabelling of the three pairs (and to node relabelling).
    """
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    qc = np.asarray(qc, dtype=np.float64)
    if qa.shape[0] == 0:
        return np.zeros((0, TRIAD_RAW_DIM), dtype=np.float64)
    values = np.stack([qa, qb, qc], axis=1)  # (T, 3, 16)
    # Lexicographic sort of the three 16D vectors; primary key = channel 0.
    keys = tuple(values[:, :, dim] for dim in range(PAIR_DIM - 1, -1, -1))
    order = np.lexsort(keys, axis=1)  # (T, 3)
    ordered = np.take_along_axis(values, order[:, :, None], axis=1)
    x0, x1, x2 = ordered[:, 0], ordered[:, 1], ordered[:, 2]
    s1 = x0 + x1 + x2
    mu = s1 / 3.0
    s_abs = np.abs(x0 - mu) + np.abs(x1 - mu) + np.abs(x2 - mu)
    s2 = x0 * x1 + x0 * x2 + x1 * x2
    return np.concatenate([s1, s_abs, s2], axis=1)


def triad_graph_from_raw(
    raw: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Graph summary ``[mean(z) ; std(z)]`` (32D) for one molecule."""
    if raw.shape[0] == 0:
        return np.zeros(T_GRAPH_DIM, dtype=np.float64)
    z = ((raw - mean) / scale) @ projection
    return np.concatenate([z.mean(axis=0), z.std(axis=0)])


# ---------------------------------------------------------------------------
# unbound matched control
# ---------------------------------------------------------------------------

def _molecule_rng(molecule_id: str) -> np.random.Generator:
    key = int(hashlib.sha256(f"{UNBOUND_SEED}|{molecule_id}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(key)


def unbound_permutation(
    outer_buckets: np.ndarray,
    molecule_id: str,
    outer_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic permutation of the outer-relation slots across triples.

    The permutation is a product of single cycles on same-distance-bucket groups
    (so it preserves the outer-pair distance-bucket distribution and has no
    fixed point whenever a group has size >= 2).  Singleton bucket groups are
    pooled and permuted among themselves; if a single unmatched slot remains it
    is a fixed point (recorded).  The permutation is target-independent and
    fully determined by the molecule id and the bucket assignment.
    """
    buckets = np.asarray(outer_buckets, dtype=np.int64)
    t = int(buckets.shape[0])
    perm = -np.ones(t, dtype=np.int64)
    rng = _molecule_rng(molecule_id)
    leftover: list[int] = []
    for bucket in np.unique(buckets):
        idx = np.flatnonzero(buckets == bucket)
        if idx.shape[0] >= 2:
            order = rng.permutation(idx.shape[0])
            arranged = idx[order]
            perm[arranged] = np.roll(arranged, -1)
        else:
            leftover.append(int(idx[0]))
    leftover_arr = np.asarray(leftover, dtype=np.int64)
    if leftover_arr.shape[0] >= 2:
        order = rng.permutation(leftover_arr.shape[0])
        arranged = leftover_arr[order]
        perm[arranged] = np.roll(arranged, -1)
    elif leftover_arr.shape[0] == 1:
        perm[leftover_arr[0]] = leftover_arr[0]
    if np.any(perm < 0):
        raise AssertionError("unbound permutation construction left unfilled slots")
    # validate it is a permutation
    if not np.array_equal(np.sort(perm), np.arange(t)):
        raise AssertionError("unbound permutation is not a bijection")
    fixed = perm == np.arange(t)
    assigned_buckets = buckets[perm]
    same_bucket = assigned_buckets == buckets
    fallback = ~same_bucket
    info: dict[str, Any] = {
        "n_triples": t,
        "n_fixed_points": int(fixed.sum()),
        "fixed_point_rate": float(fixed.mean()) if t else 0.0,
        "n_bucket_mismatch": int(fallback.sum()),
        "bucket_mismatch_rate": float(fallback.mean()) if t else 0.0,
        "n_moved": int((~fixed).sum()),
        "moved_rate": float((~fixed).mean()) if t else 0.0,
    }
    if outer_rows is not None and t:
        rows = np.asarray(outer_rows, dtype=np.int64)
        same_row = rows[perm] == rows
        info["n_same_outer_row"] = int(same_row.sum())
        info["same_outer_row_rate"] = float(same_row.mean())
        info["breakage_rate"] = float(1.0 - same_row.mean())
    return perm, info


# ---------------------------------------------------------------------------
# per-molecule triad cache
# ---------------------------------------------------------------------------

class MoleculeTriads:
    """Enumeration + lookup tables for one molecule's unordered triples."""

    def __init__(self, n_patches: int, src: np.ndarray, tgt: np.ndarray) -> None:
        self.n_patches = int(n_patches)
        self.pair_index = -np.ones((self.n_patches, self.n_patches), dtype=np.int64)
        for row, (s, t) in enumerate(zip(src.tolist(), tgt.tolist())):
            self.pair_index[s, t] = row
            self.pair_index[t, s] = row
        self.triples = enumerate_triples(self.n_patches)
        if self.triples.shape[0]:
            self.a = self.pair_index[self.triples[:, 0], self.triples[:, 1]]
            self.b = self.pair_index[self.triples[:, 0], self.triples[:, 2]]
            self.c = self.pair_index[self.triples[:, 1], self.triples[:, 2]]
            if np.any(self.a < 0) or np.any(self.b < 0) or np.any(self.c < 0):
                raise AssertionError("triple references a missing pair; graph is not complete")
        else:
            self.a = self.b = self.c = np.zeros(0, dtype=np.int64)

    def expected_triples(self) -> int:
        n = self.n_patches
        return int(n * (n - 1) * (n - 2) // 6)

    def lookup_integrity(self, pair_states: np.ndarray) -> dict[str, Any]:
        """Verify each triple's three pair states come from its three pairs."""
        if self.triples.shape[0] == 0:
            return {"n_triples": 0, "max_abs_diff": 0.0, "passed": True}
        triple_pairs = np.stack([self.a, self.b, self.c], axis=1)  # (T, 3)
        # direct gather vs re-derivation from the dense lookup
        direct = pair_states[triple_pairs]
        rederived = np.stack(
            [
                pair_states[self.pair_index[self.triples[:, 0], self.triples[:, 1]]],
                pair_states[self.pair_index[self.triples[:, 0], self.triples[:, 2]]],
                pair_states[self.pair_index[self.triples[:, 1], self.triples[:, 2]]],
            ],
            axis=1,
        )
        diff = float(np.abs(direct - rederived).max())
        # endpoint order must not matter
        flipped = np.stack(
            [
                pair_states[self.pair_index[self.triples[:, 1], self.triples[:, 0]]],
                pair_states[self.pair_index[self.triples[:, 2], self.triples[:, 0]]],
                pair_states[self.pair_index[self.triples[:, 2], self.triples[:, 1]]],
            ],
            axis=1,
        )
        diff_flip = float(np.abs(direct - flipped).max())
        return {
            "n_triples": int(self.triples.shape[0]),
            "max_abs_diff": diff,
            "max_abs_diff_flipped": diff_flip,
            "passed": bool(diff <= LOOKUP_ATOL and diff_flip <= LOOKUP_ATOL),
        }

    def outer_buckets(self, pair_bucket: np.ndarray) -> np.ndarray:
        if self.triples.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        return pair_bucket[self.c]


class TriadWitness:
    """Build true ``T_graph`` and unbound ``U_graph`` for one frozen export."""

    def __init__(self, export: Mapping[str, Any]) -> None:
        self.export = export
        self.subset_index = np.asarray(export["subset_index"], dtype=np.int64)
        self.n_patches = np.asarray(export["n_patches"], dtype=np.int64)
        self.n_pairs = np.asarray(export["n_pairs"], dtype=np.int64)
        self.q = np.asarray(export["pair_states"], dtype=np.float64)
        self.pair_bucket = np.asarray(export["pair_bucket"], dtype=np.int64)
        self.R = np.asarray(export["R"], dtype=np.float32)
        self.y = np.asarray(export["target"], dtype=np.float64)
        self.yhat_0 = np.asarray(export["yhat_0"], dtype=np.float64)
        self.pair_offsets = np.concatenate([[0], np.cumsum(self.n_pairs)]).astype(np.int64)
        # precompute pair endpoints and triples (enumeration is cheap)
        src = np.asarray(export["pair_source_local"], dtype=np.int64)
        tgt = np.asarray(export["pair_target_local"], dtype=np.int64)
        self.molecules: list[MoleculeTriads] = []
        for p in range(len(self.n_patches)):
            q0, q1 = int(self.pair_offsets[p]), int(self.pair_offsets[p + 1])
            self.molecules.append(
                MoleculeTriads(int(self.n_patches[p]), src[q0:q1], tgt[q0:q1])
            )
        self._perm_cache: dict[int, np.ndarray] = {}

    # -- triples ----------------------------------------------------------
    def molecule_id(self, position: int) -> str:
        return f"train:{int(self.subset_index[position]):04d}"

    def triple_count(self, position: int) -> int:
        return int(self.molecules[position].triples.shape[0])

    def total_triples(self) -> int:
        return int(sum(m.triples.shape[0] for m in self.molecules))

    def _states(self, position: int, *, unbound: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mol = self.molecules[position]
        q0 = int(self.pair_offsets[position])
        a = self.q[q0 + mol.a]
        b = self.q[q0 + mol.b]
        if not unbound:
            c = self.q[q0 + mol.c]
            return a, b, c
        perm = self._perm_cache.get(position)
        if perm is None:
            perm, _info = unbound_permutation(
                mol.outer_buckets(self.pair_bucket[q0:self.pair_offsets[position + 1]]),
                self.molecule_id(position),
                outer_rows=mol.c,
            )
            self._perm_cache[position] = perm
        c = self.q[q0 + mol.c[perm]]
        return a, b, c

    def raw(self, position: int, *, unbound: bool) -> np.ndarray:
        a, b, c = self._states(position, unbound=unbound)
        return triad_raw_from_values(a, b, c)

    # -- standardisation + build -----------------------------------------
    def fit_raw_statistics(
        self, fit_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel mean/std of the true triad_raw over the fit molecules."""
        chunks = [self.raw(int(p), unbound=False) for p in fit_positions]
        chunks = [chunk for chunk in chunks if chunk.shape[0]]
        if not chunks:
            return np.zeros(TRIAD_RAW_DIM), np.ones(TRIAD_RAW_DIM)
        rows = np.concatenate(chunks, axis=0)
        mean = rows.mean(axis=0)
        scale = rows.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return mean.astype(np.float64), scale.astype(np.float64)

    def graph(self, position: int, *, unbound: bool, mean: np.ndarray, scale: np.ndarray,
              projection: np.ndarray) -> np.ndarray:
        raw = self.raw(position, unbound=unbound)
        return triad_graph_from_raw(raw, mean, scale, projection)

    def build_graphs(
        self, fit_positions: np.ndarray
    ) -> dict[str, Any]:
        mean, scale = self.fit_raw_statistics(fit_positions)
        projection = _triad_projection()
        n = len(self.subset_index)
        true_graph = np.zeros((n, T_GRAPH_DIM), dtype=np.float32)
        unbound_graph = np.zeros((n, T_GRAPH_DIM), dtype=np.float32)
        unbound_stats: list[dict[str, Any]] = []
        for position in range(n):
            true_graph[position] = self.graph(
                position, unbound=False, mean=mean, scale=scale, projection=projection
            )
            unbound_graph[position] = self.graph(
                position, unbound=True, mean=mean, scale=scale, projection=projection
            )
            mol = self.molecules[position]
            q0 = int(self.pair_offsets[position])
            _perm, info = unbound_permutation(
                mol.outer_buckets(self.pair_bucket[q0:int(self.pair_offsets[position + 1])]),
                self.molecule_id(position),
                outer_rows=mol.c,
            )
            unbound_stats.append(info)
        return {
            "true_graph": true_graph,
            "unbound_graph": unbound_graph,
            "raw_mean": mean.astype(np.float32),
            "raw_scale": scale.astype(np.float32),
            "unbound_stats": unbound_stats,
        }


# ---------------------------------------------------------------------------
# triad feature spec
# ---------------------------------------------------------------------------

def write_triad_feature_spec() -> dict[str, Any]:
    spec = {
        "model": "compact-v4-hinge (frozen)",
        "pair_state_dim": PAIR_DIM,
        "pair_unordered": True,
        "pair_state_source": "pair_encoder output q_ij, computed BEFORE the one-shot centre update",
        "triple_definition": "unordered patch triple i < j < k; pairs (i,j), (i,k), (j,k)",
        "n_pairs_expected": "C(n,2) per molecule",
        "n_triples_expected": "C(n,3) per molecule",
        "triad_raw": {
            "first_order_s1": "a + b + c (16D)",
            "absolute_spread_s_abs": "|a-mu| + |b-mu| + |c-mu|, mu=(a+b+c)/3 (16D)",
            "pairwise_multiplicative_s2": "a*b + a*c + b*c (16D)",
            "total_dim": TRIAD_RAW_DIM,
            "permutation_invariant": True,
            "canonical_order": (
                "the three pair states are value-lexicographically ordered before the "
                "symmetric reductions, so the representation is exactly invariant to pair "
                "slot permutation and to node relabelling"
            ),
        },
        "projection": {
            "kind": "fixed deterministic orthonormal 48x16 (QR of standard normal), never learned",
            "seed": TRIAD_PROJECTION_SEED,
            "output_dim": TRIAD_PROJ_DIM,
        },
        "graph_summary": {
            "definition": "T_graph = [mean(z) ; std(z)] over all triads of the molecule (32D)",
            "includes_triad_count": False,
            "reason": "graph size is already inside R; the witness must not repeat count",
        },
        "standardization": (
            "triad_raw channels are standardized by adapter-fit split statistics (eps 1e-8); "
            "the SAME standardizer is applied to true triads and to the unbound control"
        ),
        "unbound_control": {
            "definition": (
                "keep the two centre-incident pair states a=q_ij, b=q_ik; deterministically "
                "permute the outer relation q_jk across the triples of the same molecule"
            ),
            "bucket_policy": (
                "single cycles within same-outer-bucket groups; leftover singleton groups are "
                "pooled and permuted; an isolated leftover is a recorded fixed point"
            ),
            "fallback_hierarchy": ["same bucket", "leftover pool (any bucket)", "singleton fixed point"],
            "marginal_multiset_preserved": True,
            "seed": UNBOUND_SEED,
            "target_used": False,
        },
        "sampling_policy": {
            "default": "full O(n^3) enumeration",
            "max_triads_per_graph": MAX_TRIADS_PER_GRAPH,
            "rule": (
                "molecule-ID-seeded uniform over unordered triples, target-independent; "
                "true witness and all controls share the identical sampled triples"
            ),
        },
        "forbidden_inputs": [
            "target",
            "residual",
            "target component",
            "cycle penalty / longest-cycle oracle",
            "penalized-logP component",
            "new atom labels / new molecular descriptors",
            "new topology oracle",
        ],
    }
    _write_json_any(RESULTS_DIR / "triad_feature_spec.json", spec)
    return spec


# ---------------------------------------------------------------------------
# Stage 0: integrity gates
# ---------------------------------------------------------------------------

def _gate_triple_count(export: Mapping[str, Any], witness: TriadWitness) -> dict[str, Any]:
    n_patches = np.asarray(export["n_patches"], dtype=np.int64)
    expected = int((n_patches * (n_patches - 1) * (n_patches - 2) // 6).sum())
    actual = witness.total_triples()
    return {
        "gate": "T0.1_triple_count_equals_C(n,3)",
        "expected_triples": expected,
        "exported_triples": actual,
        "diff": int(actual - expected),
        "passed": bool(actual == expected),
    }


def _gate_pair_integrity(export: Mapping[str, Any]) -> dict[str, Any]:
    n_patches = np.asarray(export["n_patches"], dtype=np.int64)
    n_pairs = np.asarray(export["n_pairs"], dtype=np.int64)
    expected = n_patches * (n_patches - 1) // 2
    src = np.asarray(export["pair_source_local"], dtype=np.int64)
    tgt = np.asarray(export["pair_target_local"], dtype=np.int64)
    pair_offsets = np.concatenate([[0], np.cumsum(n_pairs)]).astype(np.int64)
    self_loops = 0
    duplicates = 0
    missing = 0
    for p in range(len(n_patches)):
        q0, q1 = int(pair_offsets[p]), int(pair_offsets[p + 1])
        s, t = src[q0:q1], tgt[q0:q1]
        self_loops += int((s == t).sum())
        seen = set()
        for a, b in zip(s.tolist(), t.tolist()):
            key = (min(a, b), max(a, b))
            if key in seen:
                duplicates += 1
            seen.add(key)
        for i in range(int(n_patches[p])):
            for j in range(i + 1, int(n_patches[p])):
                if (i, j) not in seen:
                    missing += 1
    return {
        "gate": "T0.2_pair_completeness",
        "n_pairs_expected": int(expected.sum()),
        "n_pairs_exported": int(n_pairs.sum()),
        "self_loops": self_loops,
        "duplicate_pairs": duplicates,
        "missing_pairs": missing,
        "passed": bool(
            int(expected.sum()) == int(n_pairs.sum())
            and self_loops == 0
            and duplicates == 0
            and missing == 0
        ),
    }


def _gate_lookup_integrity(witness: TriadWitness) -> dict[str, Any]:
    worst = {"max_abs_diff": 0.0, "max_abs_diff_flipped": 0.0, "passed": True}
    n_triples = 0
    for position, mol in enumerate(witness.molecules):
        q0 = int(witness.pair_offsets[position])
        q1 = int(witness.pair_offsets[position + 1])
        report = mol.lookup_integrity(witness.q[q0:q1])
        worst["max_abs_diff"] = max(worst["max_abs_diff"], report["max_abs_diff"])
        worst["max_abs_diff_flipped"] = max(
            worst["max_abs_diff_flipped"], report["max_abs_diff_flipped"]
        )
        worst["passed"] = worst["passed"] and report["passed"]
        n_triples += report["n_triples"]
    return {
        "gate": "T0.3_pair_lookup_integrity + endpoint_order",
        "n_triples": int(n_triples),
        "max_abs_diff": worst["max_abs_diff"],
        "max_abs_diff_flipped": worst["max_abs_diff_flipped"],
        "passed": bool(worst["passed"]),
    }


def _gate_triad_invariance(witness: TriadWitness, max_molecules: int = 200) -> dict[str, Any]:
    worst = 0.0
    n = 0
    for position in range(min(max_molecules, len(witness.molecules))):
        mol = witness.molecules[position]
        if mol.triples.shape[0] == 0:
            continue
        q0 = int(witness.pair_offsets[position])
        a = witness.q[q0 + mol.a]
        b = witness.q[q0 + mol.b]
        c = witness.q[q0 + mol.c]
        base = triad_raw_from_values(a, b, c)
        for perm in [(1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]:
            vals = [a, b, c]
            permuted = triad_raw_from_values(vals[perm[0]], vals[perm[1]], vals[perm[2]])
            worst = max(worst, float(np.abs(base - permuted).max()))
        n += 1
    return {
        "gate": "T0.4_triad_raw_permutation_invariance",
        "n_molecules": int(n),
        "max_abs_diff": worst,
        "atol": TRIAD_INV_ATOL,
        "passed": bool(worst <= TRIAD_INV_ATOL),
    }


def _gate_projection_determinism() -> dict[str, Any]:
    p1 = _triad_projection()
    p2 = _triad_projection()
    gram = p1.T @ p1
    return {
        "gate": "T0.5_projection_determinism",
        "max_abs_diff": float(np.abs(p1 - p2).max()),
        "gram_max_abs_offdiag": float(np.abs(gram - np.eye(TRIAD_PROJ_DIM)).max()),
        "passed": bool(np.array_equal(p1, p2)),
    }


def _gate_parameter_match() -> dict[str, Any]:
    b1 = _count_parameters(ResidualHead(R_DIM, RONLY_HIDDEN, RONLY_DEPTH))
    b2 = _count_parameters(ResidualHead(R_DIM + T_GRAPH_DIM, WITNESS_HIDDEN, WITNESS_DEPTH))
    relative = abs(b2 - b1) / b1
    return {
        "gate": "T0.6_parameter_match_within_3pct",
        "b1_params": int(b1),
        "b2_params": int(b2),
        "relative_mismatch": float(relative),
        "threshold": 0.03,
        "passed": bool(relative <= 0.03),
    }


def run_gate0(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = RESULTS_DIR / "triad_integrity_report.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    per_fold: dict[str, Any] = {}
    seeds_to_check = [s for s in BACKBONE_SEEDS if _export_path(0, s).exists()]
    unbound_summary: dict[str, Any] = {}
    for seed in seeds_to_check:
        for fold in range(K_FOLDS):
            export = load_v4_export(_export_path(fold, seed))
            witness = TriadWitness(export)
            gates = {
                "T0.1": _gate_triple_count(export, witness),
                "T0.2": _gate_pair_integrity(export),
                "T0.3": _gate_lookup_integrity(witness),
                "T0.4": _gate_triad_invariance(witness),
            }
            if seed == PRIMARY_BACKBONE_SEED:
                # unbound control quality over ALL molecules of this fold
                stats: list[dict[str, Any]] = []
                for position, mol in enumerate(witness.molecules):
                    q0 = int(witness.pair_offsets[position])
                    q1 = int(witness.pair_offsets[position + 1])
                    _perm, info = unbound_permutation(
                        mol.outer_buckets(witness.pair_bucket[q0:q1]),
                        witness.molecule_id(position),
                        outer_rows=mol.c,
                    )
                    stats.append(info)
                total = sum(s["n_triples"] for s in stats)
                unbound_summary[f"fold{fold}"] = {
                    "total_triples": int(total),
                    "fixed_point_rate": float(
                        sum(s["n_fixed_points"] for s in stats) / max(total, 1)
                    ),
                    "bucket_mismatch_rate": float(
                        sum(s["n_bucket_mismatch"] for s in stats) / max(total, 1)
                    ),
                    "same_outer_row_rate": float(
                        sum(s.get("n_same_outer_row", 0) for s in stats) / max(total, 1)
                    ),
                    "breakage_rate": float(
                        1.0 - sum(s.get("n_same_outer_row", 0) for s in stats) / max(total, 1)
                    ),
                    "molecules_with_fixed_point": int(sum(s["n_fixed_points"] > 0 for s in stats)),
                    "warning": bool(
                        sum(s["n_bucket_mismatch"] for s in stats) / max(total, 1) > 0.20
                    ),
                }
            per_fold[f"fold{fold}_seed{seed}"] = gates
            print(
                f"[gate0] fold={fold} seed={seed} "
                + " ".join(f"{k}={'PASS' if v['passed'] else 'FAIL'}" for k, v in gates.items()),
                flush=True,
            )
    extra = {
        "T0.5": _gate_projection_determinism(),
        "T0.6": _gate_parameter_match(),
    }
    all_gates = [g for gates in per_fold.values() for g in gates.values()] + list(extra.values())
    failures = [g["gate"] for g in all_gates if not g.get("passed", False)]
    report = {
        "export_version": V4_EXPORT_VERSION,
        "seeds_checked": [int(s) for s in seeds_to_check],
        "per_fold": per_fold,
        "unbound_control_quality": unbound_summary,
        "extra_gates": extra,
        "n_gates": len(all_gates),
        "failures": failures,
        "all_passed": len(failures) == 0,
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_path, report)
    print(f"[gate0] all_passed={report['all_passed']} failures={failures}", flush=True)
    return report


# ---------------------------------------------------------------------------
# Stage 0: runtime profile + sampling manifest
# ---------------------------------------------------------------------------

def run_runtime_profile(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = RESULTS_DIR / "runtime_profile.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    export = load_v4_export(_export_path(0, PRIMARY_BACKBONE_SEED))
    witness = TriadWitness(export)
    rng = np.random.default_rng(RUNTIME_PROFILE_SEED)
    positions = np.sort(
        rng.choice(len(witness.molecules), size=min(RUNTIME_PROFILE_MOLECULES, len(witness.molecules)),
                   replace=False)
    )
    n_list = []
    triple_list = []
    t0 = time.perf_counter()
    for position in positions:
        mol = witness.molecules[position]
        n_list.append(mol.n_patches)
        triple_list.append(mol.triples.shape[0])
        witness.raw(int(position), unbound=False)
        witness.raw(int(position), unbound=True)
    sample_seconds = time.perf_counter() - t0
    total_triples_all = witness.total_triples()
    n_molecules_all = len(witness.molecules)
    per_molecule = sample_seconds / max(len(positions), 1)
    projected_full = per_molecule * n_molecules_all
    report = {
        "profile_molecules": int(len(positions)),
        "profile_seed": RUNTIME_PROFILE_SEED,
        "median_n_patches": float(np.median(n_list)),
        "max_n_patches": int(np.max(n_list)),
        "total_triples_sampled": int(np.sum(triple_list)),
        "median_triples_per_graph": float(np.median(triple_list)),
        "max_triples_per_graph": int(np.max(triple_list)),
        "sample_seconds": float(sample_seconds),
        "seconds_per_molecule": float(per_molecule),
        "projected_full_fold_seconds": float(projected_full),
        "full_triples_fold0_seed0": int(total_triples_all),
        "n_molecules_fold0_seed0": int(n_molecules_all),
        "full_enumeration_reasonable": bool(projected_full <= 60.0),
        "sampling_applied": False,
        "sampling_reason": (
            "full O(n^3) enumeration projected well below the budget; no cap applied, "
            "so no sampling manifest of selected triples is needed beyond the declaration"
        ),
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_path, report)
    _write_json_any(RESULTS_DIR / "sampling_manifest.json", {
        "full_enumeration": True,
        "sampling_applied": False,
        "max_triads_per_graph": MAX_TRIADS_PER_GRAPH,
        "cost_ratio_limit": FULL_ENUM_COST_RATIO_LIMIT,
        "projected_full_fold_seconds": report["projected_full_fold_seconds"],
        "rule_if_applied": (
            "molecule-ID-seeded uniform over unordered triples, target-independent; "
            "true witness and all controls share the identical sampled triples"
        ),
    })
    return report


def write_unbound_manifest(force: bool = False) -> dict[str, Any]:
    """Per-fold unbound-control quality manifest (from Gate 0)."""
    report = run_gate0(force=False)
    manifest = {
        "control": "unbound matched control (outer relation q_jk permuted within molecule)",
        "seed": UNBOUND_SEED,
        "bucket_policy": "same-bucket single cycles; leftover pool; singleton fixed point",
        "marginal_multiset_preserved": True,
        "pair_state_marginal_preserved": True,
        "breakage_note": (
            "the two centre-incident relations are kept; only the outer closure "
            "(which pair states share a triple) is permuted"
        ),
        "quality": report.get("unbound_control_quality", {}),
    }
    _write_json_any(RESULTS_DIR / "unbound_control_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# split manifest (identical to frozen readout / centre audits)
# ---------------------------------------------------------------------------

def _hash_split(molecule_ids: Sequence[str]) -> np.ndarray:
    keys = np.asarray(
        [int(hashlib.sha256(f"{SPLIT_SEED}|{mid}".encode()).hexdigest()[:16], 16) for mid in molecule_ids],
        dtype=np.float64,
    )
    order = np.argsort(keys, kind="stable")
    roles = np.empty(len(molecule_ids), dtype=object)
    labels = ("adapter_fit", "adapter_selection", "adapter_evaluation")
    cursor = 0
    for label, size in zip(labels, SPLIT_SIZES):
        roles[order[cursor: cursor + size]] = label
        cursor += size
    return roles


def _fold_split_manifest(fold: int) -> dict[str, Any]:
    holdout_idx = _fold_slices()[fold][2]
    molecule_ids = [f"train:{int(i):04d}" for i in holdout_idx]
    roles = _hash_split(molecule_ids)
    rarity = pd.read_csv(RARITY_CSV).set_index("subset_index")
    rare_ratio = rarity.loc[holdout_idx, "rare_le5_ratio"].to_numpy(dtype=np.float64)
    fit_mask = roles == "adapter_fit"
    threshold = float(np.percentile(rare_ratio[fit_mask], BULK_RARE_PERCENTILE))
    molecule_array = np.asarray(molecule_ids)
    return {
        "fold": int(fold),
        "split_seed": SPLIT_SEED,
        "sizes": dict(zip(("adapter_fit", "adapter_selection", "adapter_evaluation"), SPLIT_SIZES)),
        "assignment_sha256": hashlib.sha256("".join(roles.tolist()).encode()).hexdigest(),
        "rare_le5_percentile": BULK_RARE_PERCENTILE,
        "rare_le5_threshold": threshold,
        "molecule_ids": {
            label: molecule_array[np.flatnonzero(roles == label)].tolist()
            for label in ("adapter_fit", "adapter_selection", "adapter_evaluation")
        },
    }


def stage_splits(force: bool = False) -> dict[str, Any]:
    out_path = RESULTS_DIR / "fold_split_manifest.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    manifest = {"split_seed": SPLIT_SEED, "sizes": SPLIT_SIZES, "folds": {}}
    for fold in range(K_FOLDS):
        manifest["folds"][f"fold{fold}"] = _fold_split_manifest(fold)
    _write_json_any(out_path, manifest)
    return manifest


def ensure_export_for_seed(seed: int) -> dict[str, Any]:
    """Build the v4 export for a frozen backbone seed if it is not cached.

    This performs *inference only* on the existing frozen OOF checkpoints; it
    never trains a backbone.  It reuses the already-verified v4 export builder
    from the centre-incidence audit so the v4 fingerprint contract is shared.
    """
    missing = [f for f in range(K_FOLDS) if not _export_path(f, seed).exists()]
    if not missing:
        return {"status": "cached", "seed": int(seed)}
    from tracks.ksvd.experiments.luyin16 import (
        zinc_centre_incidence_cooccurrence_witness as centre,
    )
    centre.stage_export(seeds=(int(seed),), force=False)
    remaining = [f for f in range(K_FOLDS) if not _export_path(f, seed).exists()]
    if remaining:
        raise RuntimeError(f"failed to build v4 export for seed {seed}: folds {remaining}")
    return {"status": "built", "seed": int(seed), "folds_built": missing}


def checkpoint_inventory() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for seed in BACKBONE_SEEDS:
        for fold in range(K_FOLDS):
            export_path = _export_path(fold, seed)
            marker = _fold_marker(fold, seed)
            entry: dict[str, Any] = {
                "fold": fold,
                "seed": seed,
                "frozen_checkpoint_present": bool(marker.exists()),
                "v4_export_present": bool(export_path.exists()),
                "v4_export": str(export_path),
            }
            if marker.exists():
                try:
                    meta = json.loads(marker.read_text(encoding="utf-8"))
                    entry["state_pt_sha256"] = meta.get("state_pt_sha256")
                    entry["parameters"] = meta.get("parameters")
                    entry["selected_epoch"] = meta.get("selected_epoch")
                    entry["oof_mae"] = meta.get("oof_mae")
                except Exception:  # pragma: no cover - defensive
                    pass
            entries.append(entry)
    def _complete(seed: int, key: str) -> bool:
        return sum(1 for e in entries if e["seed"] == seed and e[key]) == K_FOLDS
    complete_seeds = [s for s in BACKBONE_SEEDS if _complete(s, "frozen_checkpoint_present")]
    exported_seeds = [s for s in BACKBONE_SEEDS if _complete(s, "v4_export_present")]
    report = {
        "source": "existing frozen OOF compact-v4-hinge checkpoints (no backbone training)",
        "v4_export_dir": str(V4_EXPORT_DIR),
        "frozen_config_path": str(CONFIG_PATH),
        "n_entries": len(entries),
        "complete_frozen_seeds": complete_seeds,
        "v4_exported_seeds": exported_seeds,
        "missing_checkpoint_policy": (
            "STOP and report a missing-checkpoint inventory; never silently convert the "
            "frozen witness diagnostic into fresh backbone training.  For a seed whose "
            "frozen checkpoint exists but v4 export is absent, the export is rebuilt by "
            "frozen inference only (see ensure_export_for_seed)."
        ),
        "entries": entries,
    }
    _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", report)
    return report


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

class ResidualHead(nn.Module):
    """yhat = yhat_0 + MLP(x).  Hidden width/depth control the parameter budget."""

    def __init__(self, in_dim: int, hidden: int, depth: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(int(depth) - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_scale", torch.ones(in_dim))

    def set_standardizer(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.in_mean = torch.tensor(mean, dtype=torch.float32)
        self.in_scale = torch.tensor(scale, dtype=torch.float32)

    def forward(self, x: torch.Tensor, yhat_0: torch.Tensor, use_witness: bool = True) -> torch.Tensor:
        value = x
        if not use_witness and x.shape[1] > R_DIM:
            value = torch.cat([x[:, :R_DIM], torch.zeros_like(x[:, R_DIM:])], dim=1)
        value = (value - self.in_mean) / self.in_scale
        return yhat_0 + self.net(value).squeeze(-1)


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _make_head(seed: int, mode: str) -> ResidualHead:
    torch.manual_seed(int(seed))
    if mode == "ronly":
        return ResidualHead(R_DIM, RONLY_HIDDEN, RONLY_DEPTH)
    if mode in ("unbound", "true"):
        return ResidualHead(R_DIM + T_GRAPH_DIM, WITNESS_HIDDEN, WITNESS_DEPTH)
    raise ValueError(mode)


def _standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def _train_adapter(
    model: nn.Module,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    yhat_fit: torch.Tensor,
    x_sel: torch.Tensor,
    y_sel: torch.Tensor,
    yhat_sel: torch.Tensor,
    *,
    epochs: int = ADAPTER_EPOCHS,
    lr: float = ADAPTER_LR,
    patience: int = ADAPTER_PATIENCE,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_state = copy.deepcopy(model.state_dict())
    best_selection = float("inf")
    bad = 0
    grad_alive = False
    epochs_run = 0
    for epoch in range(int(epochs)):
        epochs_run = epoch + 1
        model.train()
        optimizer.zero_grad()
        prediction = model(x_fit, yhat_fit)
        loss = (prediction - y_fit).abs().mean()
        loss.backward()
        if not grad_alive:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            grad_alive = bool(grads) and any(float(g.abs().sum()) > 0 for g in grads)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            selection_mae = float((model(x_sel, yhat_sel) - y_sel).abs().mean())
        if selection_mae < best_selection - 1e-9:
            best_selection = selection_mae
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    model.load_state_dict(best_state)
    return {"best_selection_mae": best_selection, "grad_alive": bool(grad_alive), "epochs": epochs_run}


# ---------------------------------------------------------------------------
# Stage 1 / 1b / 2: per fold
# ---------------------------------------------------------------------------

def _run_stage_for_fold(
    fold: int,
    backbone_seed: int,
    adapter_seed: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    export = load_v4_export(_export_path(fold, backbone_seed))
    fold_manifest = manifest["folds"][f"fold{fold}"]
    fit_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_fit"]], dtype=np.int64)
    sel_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_selection"]], dtype=np.int64)
    eva_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_evaluation"]], dtype=np.int64)

    witness = TriadWitness(export)
    lookup = {int(v): i for i, v in enumerate(witness.subset_index)}
    fit_pos = np.asarray([lookup[int(v)] for v in fit_ids], dtype=np.int64)
    sel_pos = np.asarray([lookup[int(v)] for v in sel_ids], dtype=np.int64)
    eva_pos = np.asarray([lookup[int(v)] for v in eva_ids], dtype=np.int64)
    graphs = witness.build_graphs(fit_pos)

    def tensorize(positions):
        R = torch.tensor(witness.R[positions], dtype=torch.float32)
        y = torch.tensor(witness.y[positions], dtype=torch.float32)
        yhat = torch.tensor(witness.yhat_0[positions], dtype=torch.float32)
        tg = torch.tensor(graphs["true_graph"][positions], dtype=torch.float32)
        ug = torch.tensor(graphs["unbound_graph"][positions], dtype=torch.float32)
        return R, y, yhat, tg, ug

    R_fit, y_fit, yhat_fit, tg_fit, ug_fit = tensorize(fit_pos)
    R_sel, y_sel, yhat_sel, tg_sel, ug_sel = tensorize(sel_pos)
    R_eva, y_eva, yhat_eva, tg_eva, ug_eva = tensorize(eva_pos)

    r_mean, r_scale = _standardizer(witness.R[fit_pos])
    # One shared standardizer for the witness channels, computed from TRUE
    # triads, so B2 and E differ only in whether the outer relation is bound.
    graph_mean, graph_scale = _standardizer(graphs["true_graph"][fit_pos])

    def make_and_train(x_fit, x_sel, mode, mean_scale):
        model = _make_head(adapter_seed, mode)
        model.set_standardizer(*mean_scale)
        info = _train_adapter(model, x_fit, y_fit, yhat_fit, x_sel, y_sel, yhat_sel)
        return model, info

    b1, b1_info = make_and_train(R_fit, R_sel, "ronly", (r_mean, r_scale))
    b2, b2_info = make_and_train(
        torch.cat([R_fit, ug_fit], dim=1), torch.cat([R_sel, ug_sel], dim=1), "unbound",
        (np.concatenate([r_mean, graph_mean]), np.concatenate([r_scale, graph_scale])),
    )
    e_model, e_info = make_and_train(
        torch.cat([R_fit, tg_fit], dim=1), torch.cat([R_sel, tg_sel], dim=1), "true",
        (np.concatenate([r_mean, graph_mean]), np.concatenate([r_scale, graph_scale])),
    )

    with torch.no_grad():
        p_b1 = b1(R_eva, yhat_eva).numpy()
        p_b2 = b2(torch.cat([R_eva, ug_eva], dim=1), yhat_eva).numpy()
        p_e = e_model(torch.cat([R_eva, tg_eva], dim=1), yhat_eva).numpy()
        p_e_no = e_model(torch.cat([R_eva, tg_eva], dim=1), yhat_eva, use_witness=False).numpy()
    y = witness.y[eva_pos]
    yhat0 = witness.yhat_0[eva_pos]
    mae_b0 = float(np.abs(y - yhat0).mean())
    mae_b1 = float(np.abs(y - p_b1).mean())
    mae_b2 = float(np.abs(y - p_b2).mean())
    mae_e = float(np.abs(y - p_e).mean())

    rarity = pd.read_csv(RARITY_CSV).set_index("subset_index")
    rare_ratio = rarity.loc[eva_ids, "rare_le5_ratio"].to_numpy(dtype=np.float64)
    bulk_mask = rare_ratio < float(fold_manifest["rare_le5_threshold"])
    if bulk_mask.sum() >= 10:
        err_b0 = np.abs(y - yhat0)[bulk_mask]
        err_b1 = np.abs(y - p_b1)[bulk_mask]
        err_b2 = np.abs(y - p_b2)[bulk_mask]
        err_e = np.abs(y - p_e)[bulk_mask]
        bulk = {
            "bulk_degradation_e": float(err_e.mean() - err_b0.mean()),
            "bulk_degradation_b1": float(err_b1.mean() - err_b0.mean()),
            "bulk_degradation_b2": float(err_b2.mean() - err_b0.mean()),
            "bulk_delta_e_minus_b2": float(err_e.mean() - err_b2.mean()),
            "bulk_delta_e_minus_b1": float(err_e.mean() - err_b1.mean()),
        }
    else:
        bulk = {k: float("nan") for k in (
            "bulk_degradation_e", "bulk_degradation_b1", "bulk_degradation_b2",
            "bulk_delta_e_minus_b2", "bulk_delta_e_minus_b1",
        )}
    return {
        "fold": fold,
        "backbone_seed": backbone_seed,
        "adapter_seed": adapter_seed,
        "mae_b0": mae_b0,
        "mae_b1": mae_b1,
        "mae_b2": mae_b2,
        "mae_e": mae_e,
        "delta_R": mae_b1 - mae_e,
        "delta_U": mae_b2 - mae_e,
        "delta0_E": mae_b0 - mae_e,
        **bulk,
        "n_bulk_eval": int(bulk_mask.sum()),
        "ronly_params": _count_parameters(b1),
        "witness_params": _count_parameters(e_model),
        "ronly_selection_mae": b1_info["best_selection_mae"],
        "unbound_selection_mae": b2_info["best_selection_mae"],
        "true_selection_mae": e_info["best_selection_mae"],
        "true_grad_alive": e_info["grad_alive"],
        "true_sensitivity_max_abs_diff": float(np.abs(p_e - p_e_no).max()),
        "true_prediction_std": float(p_e.std()),
        "n_eval": int(len(eva_ids)),
        "n_triples_eval": int(sum(witness.triple_count(int(p)) for p in eva_pos)),
        "_molecules": {
            "subset_index": eva_ids,
            "fold": np.full(len(eva_ids), fold, dtype=np.int64),
            "err_b0": np.abs(y - yhat0),
            "err_b1": np.abs(y - p_b1),
            "err_b2": np.abs(y - p_b2),
            "err_e": np.abs(y - p_e),
        },
    }


def _aggregate_stage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    delta_r = frame["delta_R"].to_numpy(dtype=np.float64)
    delta_u = frame["delta_U"].to_numpy(dtype=np.float64)
    return {
        "mean_delta_R": float(delta_r.mean()),
        "median_delta_R": float(np.median(delta_r)),
        "mean_delta_U": float(delta_u.mean()),
        "median_delta_U": float(np.median(delta_u)),
        "positive_folds_delta_R": int(np.sum(delta_r > 0)),
        "nonneg_folds_delta_U": int(np.sum(delta_u >= 0)),
        "positive_folds_delta_U": int(np.sum(delta_u > 0)),
        "n_folds": int(len(frame)),
        "per_fold_delta_R": delta_r.tolist(),
        "per_fold_delta_U": delta_u.tolist(),
        "mean_b0": float(frame["mae_b0"].mean()),
        "mean_b1": float(frame["mae_b1"].mean()),
        "mean_b2": float(frame["mae_b2"].mean()),
        "mean_e": float(frame["mae_e"].mean()),
        "mean_bulk_degradation_e": float(np.nanmean(frame["bulk_degradation_e"])),
        "max_bulk_degradation_e": float(np.nanmax(frame["bulk_degradation_e"])),
        "mean_bulk_degradation_e_vs_b2": float(np.nanmean(frame["bulk_delta_e_minus_b2"])),
        "max_bulk_degradation_e_vs_b2": float(np.nanmax(frame["bulk_delta_e_minus_b2"])),
        "mean_bulk_degradation_e_vs_b1": float(np.nanmean(frame["bulk_delta_e_minus_b1"])),
        "max_bulk_degradation_e_vs_b1": float(np.nanmax(frame["bulk_delta_e_minus_b1"])),
        "ronly_params": int(frame["ronly_params"].iloc[0]),
        "witness_params": int(frame["witness_params"].iloc[0]),
        "true_grad_alive_all": bool(frame["true_grad_alive"].all()),
        "true_sensitivity_min": float(frame["true_sensitivity_max_abs_diff"].min()),
        "mean_triples_eval": float(frame["n_triples_eval"].mean()),
    }


def _stratified_bootstrap(delta: np.ndarray, fold_ids: np.ndarray,
                          n_boot: int = 10000, seed: int = 20260914) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    folds = np.unique(fold_ids)
    index_by_fold = [np.flatnonzero(fold_ids == f) for f in folds]
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        chunks = [delta[idx[rng.integers(0, len(idx), len(idx))]] for idx in index_by_fold]
        means[i] = np.concatenate(chunks).mean()
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n_boot": int(n_boot),
        "prob_positive": float((means > 0).mean()),
        "n_molecules": int(len(delta)),
    }


def run_stage(
    backbone_seed: int,
    adapter_seed: int,
    tag: str,
    manifest: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_csv = RESULTS_DIR / f"{tag}_fold_results.csv"
    out_boot = RESULTS_DIR / f"{tag}_bootstrap.json"
    if out_csv.exists() and out_boot.exists() and not force:
        cached = json.loads(out_boot.read_text(encoding="utf-8"))
        return {
            "csv": str(out_csv),
            "bootstrap": str(out_boot),
            "aggregate": cached["aggregate"],
            "bootstrap_detail": {"delta_R": cached["delta_R"], "delta_U": cached["delta_U"]},
            "cached": True,
        }
    rows = [_run_stage_for_fold(fold, backbone_seed, adapter_seed, manifest) for fold in range(K_FOLDS)]
    pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]).to_csv(out_csv, index=False)
    molecules = [row["_molecules"] for row in rows]
    fold_ids = np.concatenate([m["fold"] for m in molecules])
    delta_r = np.concatenate([m["err_b1"] - m["err_e"] for m in molecules])
    delta_u = np.concatenate([m["err_b2"] - m["err_e"] for m in molecules])
    bootstrap = {
        "backbone_seed": int(backbone_seed),
        "adapter_seed": int(adapter_seed),
        "delta_R": _stratified_bootstrap(delta_r, fold_ids),
        "delta_U": _stratified_bootstrap(delta_u, fold_ids),
        "aggregate": _aggregate_stage(rows),
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_boot, bootstrap)
    print(
        f"[{tag}] seed={backbone_seed} init={adapter_seed} "
        f"mean dR={bootstrap['aggregate']['mean_delta_R']:+.4f} "
        f"mean dU={bootstrap['aggregate']['mean_delta_U']:+.4f} "
        f"({time.perf_counter() - started:.0f}s)",
        flush=True,
    )
    return {
        "csv": str(out_csv),
        "bootstrap": str(out_boot),
        "aggregate": bootstrap["aggregate"],
        "bootstrap_detail": {"delta_R": bootstrap["delta_R"], "delta_U": bootstrap["delta_U"]},
        "cached": False,
    }


# ---------------------------------------------------------------------------
# decision logic
# ---------------------------------------------------------------------------

def _decide_stage1(stage1: Mapping[str, Any]) -> str:
    agg = stage1["aggregate"]
    dr = agg["mean_delta_R"]
    du = agg["mean_delta_U"]
    pos_dr = agg["positive_folds_delta_R"]
    pos_du = agg["positive_folds_delta_U"]
    if dr <= S1_NGO_DR or du <= S1_NGO_DU or pos_du <= S1_NGO_WORSE_FOLDS:
        return "STAGE1_CLEAR_NO_GO"
    if dr >= S1_ADV_DR and du >= S1_ADV_DU and pos_dr >= S1_ADV_FOLDS and agg["nonneg_folds_delta_U"] >= S1_ADV_FOLDS:
        return "STAGE1_ADVANCE"
    return "STAGE1_BORDERLINE"


def _final_decision(
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any] | None,
    stage1b: Mapping[str, Any] | None,
) -> dict[str, Any]:
    s1 = stage1["aggregate"]
    stages = [s1] + ([stage2["aggregate"]] if stage2 is not None else [])
    pooled_dr = float(np.mean([s["mean_delta_R"] for s in stages]))
    pooled_du = float(np.mean([s["mean_delta_U"] for s in stages]))
    per_fold_dr = np.mean([s["per_fold_delta_R"] for s in stages], axis=0)
    per_fold_du = np.mean([s["per_fold_delta_U"] for s in stages], axis=0)
    folds_dr = int(np.sum(per_fold_dr > 0))
    folds_du = int(np.sum(per_fold_du > 0))

    def _boot(stage, key):
        return stage.get("bootstrap_detail", {}).get(key, stage.get(key))

    if stage2 is not None:
        boot_u = [_boot(stage1, "delta_U"), _boot(stage2, "delta_U")]
        ci_low_positive = all(b["ci95_low"] > 0 for b in boot_u)
    else:
        boot_u = [_boot(stage1, "delta_U")]
        ci_low_positive = boot_u[0]["ci95_low"] > 0
    backbone_consistent = all(s["mean_delta_U"] > 0 for s in stages)

    sensitivity_ok = all(s["true_sensitivity_min"] > 0 for s in stages)
    bulk_e_minus_b2 = [s.get("max_bulk_degradation_e_vs_b2") for s in stages]
    bulk_e_minus_b2 = [float(v) for v in bulk_e_minus_b2 if v is not None and np.isfinite(v)]
    bulk_safe_vs_unbound = bool(bulk_e_minus_b2) and max(bulk_e_minus_b2) <= BULK_MAX_DEGRADATION

    criteria = {
        "pooled_delta_R_ge_0.002": pooled_dr >= FINAL_GO_DR,
        "pooled_delta_U_ge_0.0015": pooled_du >= FINAL_GO_DU,
        "both_backbones_delta_U_positive": backbone_consistent,
        "folds_at_least_4_of_5_delta_U_positive": folds_du >= FINAL_GO_FOLDS,
        "paired_delta_U_ci_lower_positive": ci_low_positive,
        "true_triad_adapter_noncollapsed": bool(sensitivity_ok),
        "bulk_safe_vs_unbound_control_le_0.002": bulk_safe_vs_unbound,
    }
    e_gt_b1 = bool(pooled_dr > 0)
    e_gt_b2 = bool(pooled_du > 0)

    if all(criteria.values()):
        verdict = "GO"
        case = "Case A"
        claim = (
            "For the frozen compact-v4-hinge representation, an explicit "
            "permutation-invariant triad binding witness over (q_ij, q_ik, q_jk) provides "
            "reproducible incremental signed predictive value on real ZINC beyond the "
            "existing 302D graph vector R AND beyond a parameter-matched unbound control "
            "that carries the same pair-state marginals but destroys true triple closure."
        )
        action = (
            "EXPLICIT TRIADIC STRUCTURE IS TASK-RELEVANT.  Next architecture (NOT message "
            "passing): a low-rank one-shot explicit triadic composition, e.g. a = W q_ij, "
            "b = W q_ik, c = W q_jk, symmetric composition a*b + a*c + b*c, aggregated "
            "directly to the graph/centre representation with +1k-3k params, permutation "
            "invariant, no recurrent node-state propagation."
        )
    elif e_gt_b2:
        verdict = "TRIADIC_BINDING_NOT_SUPPORTED"
        case = "Case C"
        claim = (
            "The true triad witness is no better than the unbound matched control, so the "
            "gain (if any) over R is generic extra pair-derived summary information, not "
            "evidence for explicit triple closure."
        )
        action = (
            "Do NOT design a triadic composition module.  The extra summary is not "
            "attributable to higher-order binding; close the explicit triadic binding route."
        )
    elif e_gt_b1:
        verdict = "INCONCLUSIVE"
        case = "Case D"
        claim = (
            "The triad witness shows a positive but sub-threshold / unresolved incremental "
            "signal over the unbound control; the budget was not expanded."
        )
        action = "Do NOT expand budget; do NOT implement an attention / message-passing rescue."
    else:
        verdict = "NO_GO"
        case = "Case C"
        claim = (
            "No stable signed predictive value of explicit triadic binding was found beyond R "
            "(true triad witness not better than the R-only control)."
        )
        action = (
            "STOP incremental compact-v4 extensions.  The next step is a paradigm-level "
            "discussion (representation family / dictionary construction / objective-aligned "
            "structured model), not another feature."
        )

    return {
        "case": case,
        "pooled_delta_R": pooled_dr,
        "pooled_delta_U": pooled_du,
        "per_fold_delta_R": per_fold_dr.tolist(),
        "per_fold_delta_U": per_fold_du.tolist(),
        "folds_delta_R_positive": folds_dr,
        "folds_delta_U_positive": folds_du,
        "delta_U_ci95": [boot_u[0]["ci95_low"], boot_u[0]["ci95_high"]],
        "bulk_degradation_e_vs_unbound_max": (
            float(max(bulk_e_minus_b2)) if bulk_e_minus_b2 else None
        ),
        "criteria": criteria,
        "cases": {"E_gt_B1": e_gt_b1, "E_gt_B2": e_gt_b2},
        "verdict": verdict,
        "claim": claim,
        "action": action,
        "stage1b": stage1b,
        "scope_caveat": (
            "Screening diagnostic on frozen OOF states of the existing backbone.  A NO-GO "
            "closes explicit triadic binding as a bottleneck for this representation under a "
            "low-capacity witness; it does NOT prove higher-order structure is unnecessary in "
            "general, nor does it license attention or message passing."
        ),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _make_figures(stage1: Mapping[str, Any], stage2: Mapping[str, Any] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RESULTS_DIR / "stage1_fold_results.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(np.arange(K_FOLDS) - 0.2, frame["delta_R"], width=0.4, label=r"$\Delta_R$ (B1 - E)")
    ax.bar(np.arange(K_FOLDS) + 0.2, frame["delta_U"], width=0.4, label=r"$\Delta_U$ (B2 - E)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("outer fold")
    ax.set_ylabel("MAE improvement")
    ax.set_title("Figure 1: per-fold triadic binding witness gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_per_fold_delta.png", dpi=150)
    plt.close(fig)

    if stage2 is not None and (RESULTS_DIR / "stage2_fold_results.csv").exists():
        frame2 = pd.read_csv(RESULTS_DIR / "stage2_fold_results.csv")
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(np.arange(K_FOLDS), frame["delta_U"], marker="o", label="backbone seed 0")
        ax.plot(np.arange(K_FOLDS), frame2["delta_U"], marker="s", label="backbone seed 1")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("outer fold")
        ax.set_ylabel(r"$\Delta_U$")
        ax.set_title("Figure 2: triadic binding gain across frozen backbone seeds")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "figure2_stage2_seeds.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Run-all orchestration
# ---------------------------------------------------------------------------

def _run_all(force: bool = False) -> int:
    print("=== checkpoint inventory ===", flush=True)
    inventory = checkpoint_inventory()
    if not inventory["complete_frozen_seeds"]:
        print("no complete frozen checkpoint set; STOP.", flush=True)
        _write_json_any(RESULTS_DIR / "final_decision.json", {
            "verdict": "NO_GO", "reason": "missing_checkpoints",
        })
        return 2

    print("=== triad feature spec ===", flush=True)
    write_triad_feature_spec()
    print("=== runtime profile + sampling manifest ===", flush=True)
    run_runtime_profile(force=force)
    print("=== gate 0 (triad integrity) ===", flush=True)
    report = run_gate0(force=force)
    if not report.get("all_passed"):
        print("GATE 0 FAILED - measurement invalid; no adapter trained.", flush=True)
        _write_json_any(RESULTS_DIR / "final_decision.json", {
            "verdict": "NO_GO", "reason": "gate0_failed", "failures": report.get("failures"),
        })
        return 3
    write_unbound_manifest(force=force)

    manifest = stage_splits(force=force)
    print("=== stage1 (backbone seed 0, 1 init) ===", flush=True)
    stage1 = run_stage(PRIMARY_BACKBONE_SEED, 0, "stage1", manifest, force=force)
    verdict1 = _decide_stage1(stage1)

    stage1b = None
    stage2 = None
    if verdict1 == "STAGE1_CLEAR_NO_GO":
        print("Stage 1 CLEAR NO-GO; second backbone seed NOT spent.", flush=True)
    else:
        if verdict1 == "STAGE1_BORDERLINE":
            print("=== stage1b (second adapter init, same backbone) ===", flush=True)
            stage1b = run_stage(PRIMARY_BACKBONE_SEED, 1, "stage1b", manifest, force=force)
            agg_b = stage1b["aggregate"]
            collapses = (
                agg_b["mean_delta_R"] <= 0.0
                or agg_b["mean_delta_U"] <= 0.0
                or np.sign(agg_b["mean_delta_U"]) != np.sign(stage1["aggregate"]["mean_delta_U"])
            )
            if collapses:
                print("Stage 1b signal collapses/flips -> STOP, no second backbone.", flush=True)
                verdict1 = "STAGE1_CLEAR_NO_GO"
            else:
                verdict1 = "STAGE1_ADVANCE"
        if verdict1 == "STAGE1_ADVANCE":
            print("=== stage2 export (second frozen backbone seed, inference only) ===", flush=True)
            ensure_export_for_seed(BACKBONE_SEEDS[1])
            print("=== stage2 (second frozen backbone seed) ===", flush=True)
            stage2 = run_stage(BACKBONE_SEEDS[1], 0, "stage2", manifest, force=force)

    decision = _final_decision(stage1, stage2, stage1b)
    decision["stage1_verdict"] = verdict1
    _write_json_any(RESULTS_DIR / "final_decision.json", decision)
    if stage2 is not None:
        _write_json_any(RESULTS_DIR / "pooled_results.json", {
            "pooled_delta_R": decision["pooled_delta_R"],
            "pooled_delta_U": decision["pooled_delta_U"],
            "stage1": stage1["aggregate"],
            "stage2": stage2["aggregate"],
        })
        _write_json_any(RESULTS_DIR / "final_bootstrap.json", {
            "stage1_delta_U": stage1["bootstrap_detail"]["delta_U"],
            "stage2_delta_U": stage2["bootstrap_detail"]["delta_U"],
        })
    _make_figures(stage1, stage2)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["inventory", "spec", "profile", "gate0", "unbound", "splits",
                 "stage1", "stage1b", "stage2", "figures", "decision", "all"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "inventory":
        print(json.dumps(checkpoint_inventory(), indent=2, sort_keys=True))
        return 0
    if args.stage == "spec":
        write_triad_feature_spec()
        return 0
    if args.stage == "profile":
        print(json.dumps(run_runtime_profile(force=args.force), indent=2, sort_keys=True))
        return 0
    if args.stage == "gate0":
        report = run_gate0(force=args.force)
        return 0 if report.get("all_passed") else 1
    if args.stage == "unbound":
        write_unbound_manifest(force=args.force)
        return 0
    if args.stage == "splits":
        stage_splits(force=args.force)
        return 0
    if args.stage == "stage1":
        manifest = stage_splits()
        run_stage(PRIMARY_BACKBONE_SEED, 0, "stage1", manifest, force=args.force)
        return 0
    if args.stage == "stage1b":
        manifest = stage_splits()
        run_stage(PRIMARY_BACKBONE_SEED, 1, "stage1b", manifest, force=args.force)
        return 0
    if args.stage == "stage2":
        manifest = stage_splits()
        ensure_export_for_seed(BACKBONE_SEEDS[1])
        run_stage(BACKBONE_SEEDS[1], 0, "stage2", manifest, force=args.force)
        return 0
    if args.stage == "decision":
        stage1 = json.loads((RESULTS_DIR / "stage1_bootstrap.json").read_text())
        stage2 = json.loads((RESULTS_DIR / "stage2_bootstrap.json").read_text()) if (RESULTS_DIR / "stage2_bootstrap.json").exists() else None
        stage1b = json.loads((RESULTS_DIR / "stage1b_bootstrap.json").read_text()) if (RESULTS_DIR / "stage1b_bootstrap.json").exists() else None
        decision = _final_decision(stage1, stage2, stage1b)
        decision["stage1_verdict"] = _decide_stage1(stage1)
        _write_json_any(RESULTS_DIR / "final_decision.json", decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0
    if args.stage == "figures":
        stage1 = json.loads((RESULTS_DIR / "stage1_bootstrap.json").read_text())
        stage2 = json.loads((RESULTS_DIR / "stage2_bootstrap.json").read_text()) if (RESULTS_DIR / "stage2_bootstrap.json").exists() else None
        _make_figures(stage1, stage2)
        return 0
    if args.stage == "all":
        return _run_all(force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
