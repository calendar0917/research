"""ZINC local-tuple dictionary: does seeing the *correspondence* between a
root's local structural descriptor, its own atom type, each neighbour atom
type and the bond that connects them generalise the chemical target ``g``
better than the same raw attributes combined by their marginals alone?

Round: ``zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1``.

Two paired arms, seed 0, exactly 240 epochs each:

* ``J`` (joint)       — per-root code ``e_J(v) = sum_{t,a} C_v[t,a] f_D(x(v,a,t)) / d_v``
* ``I`` (independent) — per-root code ``e_I(v) = sum_{t,a} C_ind_v[t,a] f_D(x(v,a,t)) / d_v``

with ``x(v,a,t) = [phi_v ; onehot(a_v) ; onehot(a) ; onehot(t)]`` (125-D),
``C_v[t,a] = sum_{u in N(v)} 1[bond(v,u)=t, atom(u)=a]`` the *real* incident
count and ``C_ind_v[t,a] = n_t n_a / d_v`` the marginal-product count.  The
two codes share the same encoder ``f_D`` (a 64-atom tied-IHT dictionary), the
same ``D_loc`` initialisation, the same ``W_loc`` zero initialisation and the
same body.  Only the per-root combination weights differ, so the comparison is
a clean joint-vs-independent correspondence test at fixed capacity.

Reference ``B`` is the already-completed ``M_g`` raw soup of
``zinc_chemistry_dictionary_vs_mlp_seed0_v1`` (same fold, same ``g`` target,
read-only).  The new model starts from the *untrained* M_g initial tensors
(verified item-for-item) plus the new local-tuple branch; it is never warm
started from a trained soup.

The only fold is the frozen 8000/2000 model-internal split of the 10000
official-train rows (``fit_only_targets.npz``).  Official valid/test are never
loaded.  Label constants are the frozen fit-only constants; this module never
refits them.

Usage::

    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 --build-tuple
    ... --operator-checks
    ... --phase-a
    ... --smoke --device cuda
    ... --arm J --device cuda --out <dir>
    ... --arm I --device cuda --out <dir>
    ... --analyze --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.code import tccd_v0
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_chemistry_dictionary_vs_mlp_seed0_v1 as zcdm
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw
from tracks.ksvd.experiments.luyin16 import zinc_zero_binding_baseline_seed0_v1 as zbn

PROTOCOL_VERSION = "zinc-local-tuple-dictionary-joint-vs-independent-seed0-v1"
RESULT_SLUG = "zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1"
RESULTS_DIR = zjd.TRACK_ROOT / "results" / RESULT_SLUG
REF_DIR = zcdm.RESULTS_DIR  # historical reference B = M_g raw soup
TUPLE_NPZ = RESULTS_DIR / "local_tuple_index.npz"
ZINC_TRAIN_PROCESSED = zjd.REPO_ROOT / "data/ZINC/subset/processed/train.pt"
ENV_TRAIN_CACHE = zjd.TRACK_ROOT / "results/e2e_dictenv_p1/cache/env_train.pt"

#: frozen recipe, copied from the paired g round; never selected on dev.
SEED = zw.SEED
EPOCHS = zw.EPOCHS
LR = zw.LR
WEIGHT_DECAY = zw.WEIGHT_DECAY
GRAD_CLIP = zw.GRAD_CLIP
BATCH_SIZE = zw.BATCH_SIZE
SOUP_EPOCHS = zw.SOUP_EPOCHS
TRAIN_SHUFFLE_OFFSET = zw.TRAIN_SHUFFLE_OFFSET
FOLD_SEED = zw.FOLD_SEED
N_FIT = zw.N_FIT
N_DEV = zw.N_DEV
BOOT_SEED = zw.BOOT_SEED
N_BOOT = zw.N_BOOT
DELTA = zw.DELTA
DELTA_SLACK = zw.DELTA_SLACK
REPLAY_TOL = zw.REPLAY_TOL
EXPECTED_BODY_PARAMETERS = zw.EXPECTED_BODY_PARAMETERS  # 184,667
BRIDGE_PARAMETERS = zw.BRIDGE_PARAMETERS                # 82,944
EXPECTED_TOTAL_PARAMETERS = 297_499                     # + 29,888 new parameters

#: new local-tuple operator (frozen before any dev score).
PHI_DIM = 65
ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4
TUPLE_DIM = PHI_DIM + ATOM_CATEGORIES + ATOM_CATEGORIES + BOND_CATEGORIES  # 125
K_TUPLE = 64
SPARSITY = 8
IHT_STEPS = 10
IHT_ETA_FACTOR = 1.05
W_LOC_OUT = 342
FRAME_SEED = 20261004
KAPPA_SEED = 20261004
KAPPA_SAMPLE_MAX = 8192
PHI_STD_FLOOR = 1.0e-3
TUPLE_LOCAL_PARAMETERS = TUPLE_DIM * K_TUPLE + W_LOC_OUT * K_TUPLE  # 29,888

ARMS = ("J", "I")
WEIGHT_KEY = {"J": "joint", "I": "independent"}
LOG_EPOCHS = (1, 40, 120, 240)
TARGET_NOT_READ_TOL = 1.0e-5
TOL_FLOAT64 = 1.0e-10
TOL_FLOAT32 = 1.0e-5

GROUP_NAMES = zw.GROUP_NAMES
jsonable = zcdm.jsonable
write_json = zcdm.write_json
file_sha256 = zcdm.file_sha256
tensor_hash = zcdm.tensor_hash
state_hash = zcdm.state_hash
seed_everything = zcdm.seed_everything
group_masks = zcdm.group_masks
build_schedule = zw.build_schedule
build_fold = zw.build_fold
_metric_table = zcdm._metric_table
_bootstrap_ci = zcdm._bootstrap_ci
_bootstrap_self_tests = zcdm._bootstrap_self_tests
_ci_inside = zcdm._ci_inside


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray, dtype: Any = np.float32) -> str:
    return _sha256_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes())


# ---------------------------------------------------------------------------
# 1. raw incidence: C_v[t, a] and the marginal-product C_ind_v[t, a]
# ---------------------------------------------------------------------------


def load_raw_train_graphs() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Raw official-train PyG graphs from ``processed/train.pt`` (train only).

    The file is the PyG ``(collated_data, slices, Data)`` tuple; the per-graph
    node/edge slices are reconstructed positionally.  Only the official-train
    file is opened; the valid/test processed files are never touched.
    """
    loaded = torch.load(ZINC_TRAIN_PROCESSED, map_location="cpu", weights_only=False)
    data, slices, _cls = loaded
    x = data["x"].reshape(-1).numpy().astype(np.int64, copy=False)
    edge_index = data["edge_index"].numpy().astype(np.int64, copy=False)
    edge_attr = data["edge_attr"].reshape(-1).numpy().astype(np.int64, copy=False)
    node_slices = slices["x"].numpy().astype(np.int64)
    edge_slices = slices["edge_index"].numpy().astype(np.int64)
    graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index in range(len(node_slices) - 1):
        lo, hi = int(node_slices[index]), int(node_slices[index + 1])
        elo, ehi = int(edge_slices[index]), int(edge_slices[index + 1])
        graphs.append(
            (
                x[lo:hi].copy(),
                edge_index[:, elo:ehi].copy(),
                edge_attr[elo:ehi].copy(),
            )
        )
    if len(graphs) != 10000:
        raise RuntimeError(f"raw train graph count {len(graphs)} != 10000")
    return graphs


def molecule_incidence(
    atom_types: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """``C_v[t, a]`` for one molecule from raw ``x``/``edge_index``/``edge_attr``.

    Directed symmetric edges are de-duplicated with the canonical ``(min, max)``
    key, first occurrence wins (the repository's audited graph convention).
    Every de-duplicated incident edge is counted exactly once.
    """
    n = int(atom_types.shape[0])
    canonical: dict[tuple[int, int], int] = {}
    n_directed = int(edge_index.shape[1])
    n_self = 0
    n_double = 0
    n_attr_mismatch = 0
    for column in range(n_directed):
        u = int(edge_index[0, column])
        v = int(edge_index[1, column])
        if u == v:
            n_self += 1
            continue
        key = (u, v) if u < v else (v, u)
        bond = int(edge_attr[column])
        if key in canonical:
            n_double += 1
            if canonical[key] != bond:
                n_attr_mismatch += 1
            continue
        canonical[key] = bond
    count = np.zeros((n, BOND_CATEGORIES, ATOM_CATEGORIES), dtype=np.int64)
    for (u, v), bond in canonical.items():
        count[u, bond, int(atom_types[v])] += 1
        count[v, bond, int(atom_types[u])] += 1
    stats = {
        "n_nodes": n,
        "n_directed_edges": n_directed,
        "n_canonical_edges": int(len(canonical)),
        "n_double_directed": n_double,
        "n_self_loops": n_self,
        "n_bond_attr_mismatch_on_double": n_attr_mismatch,
    }
    return count, stats


def root_weights(
    count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-molecule ``(d, n_t, n_a, w_joint, w_independent)`` as float64.

    ``d`` [n], ``n_t`` [n, 4], ``n_a`` [n, 28], ``w_joint = C / d`` and
    ``w_independent = n_t n_a / d^2`` (both [n, 4, 28]); ``d == 0`` rows are
    zero.  The marginals of ``w_independent * d`` equal those of ``C`` exactly.
    """
    count64 = count.astype(np.float64)
    d = count64.sum(axis=(1, 2))
    n_t = count64.sum(axis=2)
    n_a = count64.sum(axis=1)
    denom = d[:, None, None]
    safe = np.where(denom > 0.0, denom, 1.0)
    w_joint = np.where(denom > 0.0, count64 / safe, 0.0)
    w_ind = np.where(denom > 0.0, n_t[:, :, None] * n_a[:, None, :] / (safe * safe), 0.0)
    return d, n_t, n_a, w_joint, w_ind


def _support_union(count: np.ndarray, n_t: np.ndarray, n_a: np.ndarray) -> np.ndarray:
    realized = count > 0
    marginal = (n_t[:, :, None] > 0) & (n_a[:, None, :] > 0)
    return realized | marginal


# ---------------------------------------------------------------------------
# 2. build + load the frozen tuple index (label-free, fit-only scalers)
# ---------------------------------------------------------------------------


def init_d_loc(seed: int = FRAME_SEED) -> torch.Tensor:
    """Fixed Gaussian frame on an independent generator (never the body RNG)."""
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randn(TUPLE_DIM, K_TUPLE, generator=generator, dtype=torch.float32)


def _env_phi_atom() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = torch.load(ENV_TRAIN_CACHE, map_location="cpu", weights_only=False)
    phi = blob["phi"].numpy().astype(np.float32)
    atom = blob["atom"].numpy().astype(np.int64)
    node_sizes = blob["node_sizes"].numpy().astype(np.int64)
    return phi, atom, node_sizes


def build_tuple_index(log: Any = print) -> dict[str, Any]:
    """Full local (CPU) build of the per-root tuple index, saved as npz.

    All scaler constants and the kappa scale are fitted on fit rows only; the
    dev raw graphs are used only to build their (label-free) incidence rows.
    """
    started = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold = build_fold()
    fit_idx, dev_idx = fold["fit_idx"], fold["dev_idx"]
    phi, atom, node_sizes = _env_phi_atom()
    graphs = load_raw_train_graphs()

    if int(node_sizes.sum()) != int(phi.shape[0]) or int(node_sizes.sum()) != int(atom.shape[0]):
        raise RuntimeError("env cache / node-size mismatch")
    for index, (atom_types, edge_index, edge_attr) in enumerate(graphs):
        if int(atom_types.shape[0]) != int(node_sizes[index]):
            raise RuntimeError(f"raw graph {index} node count mismatch")
        if not np.array_equal(atom[int(node_sizes[:index].sum()) : int(node_sizes[: index + 1].sum())], atom_types):
            raise RuntimeError(f"raw graph {index} root atom order mismatch with env cache")
    root_base = np.concatenate([[0], np.cumsum(node_sizes)]).astype(np.int64)
    total_roots = int(node_sizes.sum())

    pair_t: list[np.ndarray] = []
    pair_a: list[np.ndarray] = []
    pair_j: list[np.ndarray] = []
    pair_i: list[np.ndarray] = []
    counts: list[int] = []
    edge_stats: list[dict[str, int]] = []
    fit_contrast_roots = 0
    fit_roots_total = 0
    max_marginal_delta = 0.0
    max_marginal_delta32 = 0.0
    root_d = np.zeros(total_roots, dtype=np.int64)
    n_contrast_roots = 0
    phi_dupe_groups: dict[bytes, list[int]] = {}
    phi_dupe_contrast: dict[bytes, list[int]] = {}

    for index in range(10000):
        atom_types, edge_index, edge_attr = graphs[index]
        count, stats = molecule_incidence(atom_types, edge_index, edge_attr)
        edge_stats.append(stats)
        d, n_t, n_a, w_joint, w_ind = root_weights(count)
        support = _support_union(count, n_t, n_a)
        base = int(root_base[index])
        root_d[base : base + len(d)] = d.astype(np.int64)
        position = int(np.searchsorted(fit_idx, index))
        is_fit = bool(position < fit_idx.size and int(fit_idx[position]) == index)
        for local in range(len(d)):
            root = base + local
            rows = np.nonzero(support[local].reshape(-1))[0]
            if rows.size == 0:
                counts.append(0)
                continue
            tt = (rows // ATOM_CATEGORIES).astype(np.int64)
            aa = (rows % ATOM_CATEGORIES).astype(np.int64)
            pair_t.append(tt)
            pair_a.append(aa)
            pair_j.append(w_joint[local].reshape(-1)[rows].astype(np.float32))
            pair_i.append(w_ind[local].reshape(-1)[rows].astype(np.float32))
            counts.append(int(rows.size))
            # float64 marginal identity: sum_a C_ind == n_t and sum_t C_ind == n_a
            ind = n_t[local][:, None] * n_a[local][None, :] / max(d[local], 1.0)
            delta = float(np.max(np.abs(ind.sum(axis=1) - n_t[local])))
            delta = max(delta, float(np.max(np.abs(ind.sum(axis=0) - n_a[local]))))
            max_marginal_delta = max(max_marginal_delta, delta)
            ind32 = (n_t[local][:, None].astype(np.float32) * n_a[local][None, :].astype(np.float32)) / np.float32(
                max(d[local], 1.0)
            )
            delta32 = float(np.max(np.abs(ind32.astype(np.float64).sum(axis=1) - n_t[local]))) if d[local] > 0 else 0.0
            delta32 = max(delta32, float(np.max(np.abs(ind32.astype(np.float64).sum(axis=0) - n_a[local]))) if d[local] > 0 else 0.0)
            max_marginal_delta32 = max(max_marginal_delta32, delta32)
            if np.any(count[local] != ind):
                n_contrast_roots += 1
                if is_fit:
                    fit_contrast_roots += 1
            if is_fit:
                fit_roots_total += 1
        if index % 1000 == 0:
            log(f"[tuple-build] {index}/10000 roots={sum(counts)} elapsed={time.perf_counter() - started:.1f}s")

    pair_t_arr = np.concatenate(pair_t).astype(np.int64)
    pair_a_arr = np.concatenate(pair_a).astype(np.int64)
    pair_j_arr = np.concatenate(pair_j).astype(np.float32)
    pair_i_arr = np.concatenate(pair_i).astype(np.float32)
    pair_ptr = np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))]).astype(np.int64)
    n_pairs = int(pair_ptr[-1])
    log(f"[tuple-build] pairs={n_pairs} roots={total_roots} in {time.perf_counter() - started:.1f}s")

    # ---- phi scaler: fit rows, real incident (realized C>0) tuples only ----
    fit_mol = np.zeros(10000, dtype=bool)
    fit_mol[fit_idx] = True
    fit_root_mask = np.zeros(total_roots, dtype=bool)
    for index in np.nonzero(fit_mol)[0]:
        fit_root_mask[root_base[index] : root_base[index + 1]] = True
    realized = pair_j_arr > 0.0
    realized = realized & fit_root_mask[np.repeat(np.arange(total_roots, dtype=np.int64), np.diff(pair_ptr))]
    # pair -> root id
    pair_root = np.repeat(np.arange(total_roots, dtype=np.int64), np.diff(pair_ptr))
    fit_realized_roots = pair_root[realized]
    if fit_realized_roots.size == 0:
        raise RuntimeError("no realized fit incident tuples")
    phi_sample = phi[fit_realized_roots].astype(np.float64)
    phi_mean = phi_sample.mean(axis=0)
    phi_std = phi_sample.std(axis=0)
    phi_std = np.maximum(phi_std, PHI_STD_FLOOR)
    z = (phi[fit_realized_roots].astype(np.float64) - phi_mean) / phi_std
    phi_scale = float(np.sqrt(np.mean(z * z)))
    if not np.isfinite(phi_scale) or phi_scale <= 1e-12:
        raise RuntimeError("phi L2-RMS scale is degenerate")

    # ---- one-shot label-free kappa calibration on fit roots ----
    fit_roots = np.nonzero(fit_root_mask)[0]
    rng = np.random.default_rng(int(KAPPA_SEED))
    take = min(int(KAPPA_SAMPLE_MAX), int(fit_roots.size))
    sample = np.sort(fit_roots[rng.permutation(int(fit_roots.size))[:take]])
    dloc = init_d_loc()
    kappa, kappa_stats = _kappa_for_roots(
        dloc, phi, atom, pair_ptr, pair_t_arr, pair_a_arr, pair_j_arr, pair_i_arr,
        phi_mean.astype(np.float32), phi_std.astype(np.float32), np.float32(phi_scale), sample,
    )
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise RuntimeError("MECHANISM_INIT_BLOCKED: kappa is invalid")

    np.savez_compressed(
        TUPLE_NPZ,
        root_base=root_base,
        node_sizes=node_sizes.astype(np.int64),
        pair_ptr=pair_ptr,
        pair_t=pair_t_arr,
        pair_a=pair_a_arr,
        pair_wJ=pair_j_arr,
        pair_wI=pair_i_arr,
        root_atom=atom.astype(np.int64),
        phi_mean=phi_mean.astype(np.float32),
        phi_std=phi_std.astype(np.float32),
        phi_scale=np.asarray(phi_scale, np.float32),
        kappa=np.asarray(kappa, np.float32),
        kappa_sample=sample.astype(np.int64),
    )
    npz_hash = file_sha256(TUPLE_NPZ)
    stats = {
        "protocol_version": PROTOCOL_VERSION,
        "n_molecules": 10000,
        "n_roots": total_roots,
        "n_pairs_union_support": n_pairs,
        "mean_pairs_per_root": float(n_pairs / total_roots),
        "root_d_zero": int((root_d == 0).sum()),
        "max_marginal_delta_float64": float(max_marginal_delta),
        "max_marginal_delta_float32": float(max_marginal_delta32),
        "marginal_tol_float64_pass": bool(max_marginal_delta <= TOL_FLOAT64),
        "marginal_tol_float32_pass": bool(max_marginal_delta32 <= TOL_FLOAT32),
        "contrast_roots": int(n_contrast_roots),
        "fit_roots": int(fit_roots_total),
        "fit_contrast_roots": int(fit_contrast_roots),
        "fit_contrast_fraction": float(fit_contrast_roots / max(fit_roots_total, 1)),
        "phi_scaler": {
            "fitted_on": "fit rows, realized C>0 incident tuples",
            "n_samples": int(fit_realized_roots.size),
            "std_floor": PHI_STD_FLOOR,
            "phi_scale_l2_rms": phi_scale,
            "phi_mean_sha256": _array_sha256(phi_mean.astype(np.float32)),
            "phi_std_sha256": _array_sha256(phi_std.astype(np.float32)),
        },
        "kappa": {
            "value": float(kappa),
            "definition": "1 / RMS(initial e_J, e_I) over up to 8192 uniformly sampled fit roots",
            "seed": int(KAPPA_SEED),
            "n_roots": int(sample.size),
            "sample_sha256": _array_sha256(sample, np.int64),
            "sample_head": sample[:8].tolist(),
            "raw_rms": kappa_stats["rms"],
            "e_J_rms": kappa_stats["e_J_rms"],
            "e_I_rms": kappa_stats["e_I_rms"],
        },
        "tuple_npz": str(TUPLE_NPZ),
        "tuple_npz_sha256": npz_hash,
        "edge_stats_aggregate": {
            key: int(sum(item[key] for item in edge_stats))
            for key in ("n_nodes", "n_directed_edges", "n_canonical_edges", "n_double_directed",
                        "n_self_loops", "n_bond_attr_mismatch_on_double")
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - started),
    }
    write_json(RESULTS_DIR / "tuple_index_meta.json", stats)
    log(f"[tuple-build] saved {TUPLE_NPZ} sha={npz_hash[:16]} kappa={float(kappa):.6f}")
    return stats


class TuplePayload:
    """Plain tensor container (not in the model state dict)."""

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.root_base = torch.as_tensor(arrays["root_base"], dtype=torch.long)
        self.node_sizes = torch.as_tensor(arrays["node_sizes"], dtype=torch.long)
        self.pair_ptr = torch.as_tensor(arrays["pair_ptr"], dtype=torch.long)
        self.pair_t = torch.as_tensor(arrays["pair_t"], dtype=torch.long)
        self.pair_a = torch.as_tensor(arrays["pair_a"], dtype=torch.long)
        self.pair_wJ = torch.as_tensor(arrays["pair_wJ"], dtype=torch.float32)
        self.pair_wI = torch.as_tensor(arrays["pair_wI"], dtype=torch.float32)
        self.root_atom = torch.as_tensor(arrays["root_atom"], dtype=torch.long)
        self.phi_mean = torch.as_tensor(arrays["phi_mean"], dtype=torch.float32)
        self.phi_std = torch.as_tensor(arrays["phi_std"], dtype=torch.float32)
        self.phi_scale = float(np.asarray(arrays["phi_scale"]).reshape(-1)[0])
        self.kappa = float(np.asarray(arrays["kappa"]).reshape(-1)[0])
        self.kappa_sample = torch.as_tensor(arrays["kappa_sample"], dtype=torch.long)

    def __getitem__(self, name: str) -> torch.Tensor:
        return getattr(self, name)


def load_tuple_payload() -> TuplePayload:
    if not TUPLE_NPZ.exists():
        raise FileNotFoundError(f"tuple index missing: {TUPLE_NPZ}")
    with np.load(TUPLE_NPZ, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    return TuplePayload(arrays)


def _device_buffer(self, payload: TuplePayload, name: str, device: torch.device) -> torch.Tensor:
    cache = getattr(self, "_device_cache", None)
    if cache is None:
        cache = {}
        self._device_cache = cache
    key = str(device)
    bucket = cache.setdefault(key, {})
    if name not in bucket:
        bucket[name] = payload[name].to(device)
    return bucket[name]


# ---------------------------------------------------------------------------
# 3. production operator functions (shared by the model and the checks)
# ---------------------------------------------------------------------------


def tuple_features(
    phi_raw: torch.Tensor,
    atom_v: torch.Tensor,
    atom_u: torch.Tensor,
    bond_t: torch.Tensor,
    phi_mean: torch.Tensor,
    phi_std: torch.Tensor,
    phi_scale: float,
) -> torch.Tensor:
    """``x(v,a,t) = [normalised phi_v ; onehot28(a_v) ; onehot28(a) ; onehot4(t)]``."""
    phi_n = (phi_raw - phi_mean) / phi_std / float(phi_scale)
    return torch.cat(
        [
            phi_n,
            F.one_hot(atom_v, ATOM_CATEGORIES).to(phi_raw.dtype),
            F.one_hot(atom_u, ATOM_CATEGORIES).to(phi_raw.dtype),
            F.one_hot(bond_t, BOND_CATEGORIES).to(phi_raw.dtype),
        ],
        dim=1,
    )


def tied_tuple_codes(d_loc_raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """10-step tied IHT with ``eta = 1 / (1.05 * lambda_max(Dbar^T Dbar))``."""
    dbar = F.normalize(d_loc_raw, dim=0, eps=1e-12)
    sigma = tccd_v0.power_iter_sigma(dbar)
    eta = (1.0 / (IHT_ETA_FACTOR * (sigma * sigma) + 1e-12)).detach()
    alpha = torch.zeros(x.shape[0], dbar.shape[1], device=x.device, dtype=x.dtype)
    for _ in range(int(IHT_STEPS)):
        alpha = alpha + eta * ((x - alpha @ dbar.t()) @ dbar)
        alpha = tccd_v0.hard_threshold_rows(alpha, int(SPARSITY))
    return alpha


def compose_root_codes(
    d_loc_raw: torch.Tensor,
    phi: torch.Tensor,
    atom: torch.Tensor,
    pair_ptr: torch.Tensor,
    pair_t: torch.Tensor,
    pair_a: torch.Tensor,
    pair_w: torch.Tensor,
    root_ids: torch.Tensor,
    phi_mean: torch.Tensor,
    phi_std: torch.Tensor,
    phi_scale: float,
) -> torch.Tensor:
    """``e(v) = sum_p w_p f_D(x_p)`` for the requested (global) roots.

    ``pair_ptr`` are global per-root offsets; ``root_ids`` must be any subset
    of global root ids.  This is exactly the operator used inside the model.
    """
    start = pair_ptr[root_ids]
    end = pair_ptr[root_ids + 1]
    counts = end - start
    total = int(counts.sum().item())
    if total == 0:
        return torch.zeros(root_ids.numel(), K_TUPLE, device=phi.device, dtype=phi.dtype)
    offsets = counts.cumsum(0) - counts
    flat = torch.repeat_interleave(start, counts) + (
        torch.arange(total, device=phi.device) - torch.repeat_interleave(offsets, counts)
    )
    tuple_row = torch.repeat_interleave(torch.arange(root_ids.numel(), device=phi.device), counts)
    t = pair_t[flat]
    a_u = pair_a[flat]
    w = pair_w[flat]
    x = tuple_features(phi[root_ids][tuple_row], atom[root_ids][tuple_row], a_u, t, phi_mean, phi_std, phi_scale)
    alpha = tied_tuple_codes(d_loc_raw, x)
    out = torch.zeros(root_ids.numel(), K_TUPLE, device=phi.device, dtype=alpha.dtype)
    out.index_add_(0, tuple_row, alpha * w.unsqueeze(1))
    return out


def _kappa_for_roots(
    dloc: torch.Tensor,
    phi: np.ndarray,
    atom: np.ndarray,
    pair_ptr: np.ndarray,
    pair_t: np.ndarray,
    pair_a: np.ndarray,
    pair_wJ: np.ndarray,
    pair_wI: np.ndarray,
    phi_mean: np.ndarray,
    phi_std: np.ndarray,
    phi_scale: np.float32,
    sample: np.ndarray,
) -> tuple[float, dict[str, float]]:
    device = torch.device("cpu")
    phi_t = torch.as_tensor(phi, dtype=torch.float32)
    atom_t = torch.as_tensor(atom, dtype=torch.long)
    ptr_t = torch.as_tensor(pair_ptr, dtype=torch.long)
    t_t = torch.as_tensor(pair_t, dtype=torch.long)
    a_t = torch.as_tensor(pair_a, dtype=torch.long)
    j_t = torch.as_tensor(pair_wJ, dtype=torch.float32)
    i_t = torch.as_tensor(pair_wI, dtype=torch.float32)
    mean_t = torch.as_tensor(phi_mean, dtype=torch.float32)
    std_t = torch.as_tensor(phi_std, dtype=torch.float32)
    root_ids = torch.as_tensor(sample, dtype=torch.long)
    with torch.no_grad():
        e_j = compose_root_codes(dloc, phi_t, atom_t, ptr_t, t_t, a_t, j_t, root_ids, mean_t, std_t, float(phi_scale))
        e_i = compose_root_codes(dloc, phi_t, atom_t, ptr_t, t_t, a_t, i_t, root_ids, mean_t, std_t, float(phi_scale))
        rms = float(torch.sqrt((e_j.pow(2).mean() + e_i.pow(2).mean()) / 2.0))
        kappa = 1.0 / rms if rms > 0 else float("nan")
    return kappa, {
        "rms": rms,
        "e_J_rms": float(torch.sqrt(e_j.pow(2).mean())),
        "e_I_rms": float(torch.sqrt(e_i.pow(2).mean())),
    }


# ---------------------------------------------------------------------------
# 4. the model: M_g skeleton + shared local tuple dictionary + W_loc
# ---------------------------------------------------------------------------


class LocalTupleEncoder(nn.Module):
    """Shared local dictionary over 125-D tuples + zero-init W_loc into fusion."""

    def __init__(self, payload: TuplePayload, weight_key: str) -> None:
        super().__init__()
        if weight_key not in ("joint", "independent"):
            raise ValueError(weight_key)
        self.weight_key = weight_key
        self.kappa = float(payload.kappa)
        self.D_loc_raw = nn.Parameter(init_d_loc().clone())
        self.W_loc = nn.Parameter(torch.zeros(W_LOC_OUT, K_TUPLE, dtype=torch.float32))
        self._payload = payload
        self._device_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.ablate = False
        self.last_stats: dict[str, Any] = {}

    def _buf(self, name: str, device: torch.device) -> torch.Tensor:
        return _device_buffer(self, self._payload, name, device)

    def tuple_features(self, phi_raw: torch.Tensor, atom_v: torch.Tensor, atom_u: torch.Tensor, bond_t: torch.Tensor) -> torch.Tensor:
        return tuple_features(
            phi_raw, atom_v, atom_u, bond_t,
            self._buf("phi_mean", phi_raw.device), self._buf("phi_std", phi_raw.device), self._payload.phi_scale,
        )

    def root_codes(self, data: Any) -> torch.Tensor:
        phi = data.dict_phi
        atom = data.dict_atom.to(torch.long)
        device = phi.device
        n_rows = int(phi.shape[0])
        mol = data.local_mol_id.to(torch.long).reshape(-1)
        n_graphs = int(mol.numel())
        batch = data.batch.to(torch.long)
        graph_counts = torch.bincount(batch, minlength=n_graphs)
        ptr = torch.cat([graph_counts.new_zeros(1), graph_counts.cumsum(0)])
        rows = torch.arange(n_rows, device=device)
        local_pos = rows - ptr[batch]
        root_base = self._buf("root_base", device)
        global_root = root_base[mol][batch] + local_pos
        pair_ptr = self._buf("pair_ptr", device)
        start = pair_ptr[global_root]
        end = pair_ptr[global_root + 1]
        counts = end - start
        total = int(counts.sum().item())
        if total == 0:
            return torch.zeros(n_rows, K_TUPLE, device=device, dtype=phi.dtype)
        offsets = counts.cumsum(0) - counts
        flat = torch.repeat_interleave(start, counts) + (
            torch.arange(total, device=device) - torch.repeat_interleave(offsets, counts)
        )
        tuple_row = torch.repeat_interleave(rows, counts)
        t = self._buf("pair_t", device)[flat]
        a_u = self._buf("pair_a", device)[flat]
        w = self._buf("pair_wJ" if self.weight_key == "joint" else "pair_wI", device)[flat]
        x = self.tuple_features(phi[tuple_row], atom[tuple_row], a_u, t)
        alpha = tied_tuple_codes(self.D_loc_raw, x)
        out = torch.zeros(n_rows, K_TUPLE, device=device, dtype=alpha.dtype)
        out.index_add_(0, tuple_row, alpha * w.unsqueeze(1))
        with torch.no_grad():
            nonzero = alpha != 0
            used_atoms = nonzero.any(dim=0)
            self.last_stats = {
                "n_roots": n_rows,
                "n_molecules": n_graphs,
                "n_tuples": total,
                "alpha_nonzero_fraction": float(nonzero.float().mean()),
                "alpha_per_tuple_nonzero": float(nonzero.float().sum(1).mean()),
                "atoms_used": int(used_atoms.sum()),
                "atoms_used_fraction": float(used_atoms.float().mean()),
                "atoms_used_mask": used_atoms.detach().cpu(),
                "root_code_nonzero_fraction": float((out != 0).float().mean()),
                "root_code_dead_dims": int((out.std(dim=0) < 1e-8).sum()),
                "root_code_rms": float(out.pow(2).mean().sqrt()),
            }
        if self.ablate:
            return torch.zeros_like(out)
        return out

    def forward(self, data: Any) -> torch.Tensor:
        codes = self.root_codes(data)
        scaled = codes * self.kappa
        with torch.no_grad():
            contribution = F.linear(scaled, self.W_loc)
            self.last_stats = dict(self.last_stats)
            self.last_stats["e_rms_scaled"] = float(scaled.pow(2).mean().sqrt())
            self.last_stats["contribution_rms"] = float(contribution.pow(2).mean().sqrt())
            self.last_stats["W_loc_norm"] = float(self.W_loc.norm())
            self.last_stats["D_loc_norm"] = float(self.D_loc_raw.norm())
        return scaled


class LocalTupleFull(zbn.DeployFull):
    """``DeployFull`` with ``preact = fusion0(Sem110) + W_loc @ (kappa * e_arm)``."""

    def __init__(self, payload: TuplePayload, weight_key: str) -> None:
        super().__init__()
        bridge = self.local_dictionary_bridge
        dictionary = bridge.D_L.detach().clone()
        values = bridge.V_L.detach().clone()
        # exact matched-MLP bridge of the reference skeleton (M_g family)
        self.local_dictionary_bridge = zw.MLPBridge(dictionary, values)
        self.local_tuple = LocalTupleEncoder(payload, weight_key)
        self.weight_key = weight_key
        if int(self.fusion[0].in_features) != 110 or int(self.fusion[0].out_features) != W_LOC_OUT:
            raise RuntimeError(
                f"unexpected fusion layout {int(self.fusion[0].in_features)} -> {int(self.fusion[0].out_features)}"
            )

    def environments_masked(self, coord: torch.Tensor, data: Any, mask: Any, fill: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        interface = sem.SEM108Model.semantic_interface(self, coord, data, mask, fill)
        preact = self.fusion[0](interface)
        if not self.local_tuple.ablate:
            preact = preact + F.linear(self.local_tuple(data), self.local_tuple.W_loc)
        hidden = self.fusion[1](preact)
        return self.local_dictionary_bridge(self.fusion[2](hidden))


def local_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for name, p in model.named_parameters() if name.startswith("local_tuple.")))


def bridge_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for name, p in model.named_parameters() if name.startswith("local_dictionary_bridge.")))


def base_body_parameter_count(model: nn.Module) -> int:
    return int(sum(
        p.numel() for name, p in model.named_parameters()
        if not name.startswith("local_tuple.") and not name.startswith("local_dictionary_bridge.")
    ))


def build_arm(arm: str, payload: TuplePayload | None = None) -> nn.Module:
    """Fresh J/I model (M_g untrained init + shared D_loc + zero W_loc)."""
    if arm not in ARMS:
        raise ValueError(arm)
    if payload is None:
        payload = load_tuple_payload()
    torch.manual_seed(int(SEED))
    model = LocalTupleFull(payload, WEIGHT_KEY[arm])
    audit = {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "base_body_parameters": base_body_parameter_count(model),
        "bridge_parameters": bridge_parameter_count(model),
        "local_tuple_parameters": local_parameter_count(model),
        "D_loc_parameters": int(model.local_tuple.D_loc_raw.numel()),
        "W_loc_parameters": int(model.local_tuple.W_loc.numel()),
    }
    if audit != {
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "base_body_parameters": EXPECTED_BODY_PARAMETERS,
        "bridge_parameters": BRIDGE_PARAMETERS,
        "local_tuple_parameters": TUPLE_LOCAL_PARAMETERS,
        "D_loc_parameters": TUPLE_DIM * K_TUPLE,
        "W_loc_parameters": W_LOC_OUT * K_TUPLE,
    }:
        raise RuntimeError(f"parameter audit failed: {audit}")
    return model


def build_reference_m() -> nn.Module:
    """The reference skeleton used by B (M_g family), fresh untrained init."""
    return zw.build_arm("M")


# ---------------------------------------------------------------------------
# 5. data preparation for the new arms
# ---------------------------------------------------------------------------


def load_prepared() -> tuple[dict[str, Any], dict[str, Any], list[Any], list[Any], dict[str, np.ndarray]]:
    fold, prep_meta, fit_data, dev_data, tgt = zcdm.load_prepared()
    for position, index in enumerate(fold["fit_idx"].tolist()):
        fit_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    for position, index in enumerate(fold["dev_idx"].tolist()):
        dev_data[position].local_mol_id = torch.tensor([int(index)], dtype=torch.long)
    return fold, prep_meta, fit_data, dev_data, tgt


def verify_env_cache(payload: TuplePayload) -> dict[str, Any]:
    """The committed tuple index must match the env cache loaded on this host."""
    phi, atom, node_sizes = _env_phi_atom()
    checks = {
        "node_sizes_match": bool(np.array_equal(node_sizes, payload.node_sizes.numpy())),
        "root_atom_match": bool(np.array_equal(atom, payload.root_atom.numpy())),
        "phi_sha256": _array_sha256(phi),
        "atom_sha256": _array_sha256(atom, np.int64),
    }
    return checks


# ---------------------------------------------------------------------------
# 6. training one arm
# ---------------------------------------------------------------------------


def evaluate_state(model: nn.Module, data_list: Sequence[Any], target_local: np.ndarray, device: torch.device) -> np.ndarray:
    return zw.evaluate_state(model, data_list, target_local, device)


def train_arm(
    arm: str,
    *,
    device: torch.device,
    out_dir: Path,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_tuple_payload()
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    schedule, schedule_hash = build_schedule(len(fit_data), int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)

    seed_everything(SEED)
    model = build_arm(arm, payload).to(device)
    audit = {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "base_body_parameters": base_body_parameter_count(model),
        "bridge_parameters": bridge_parameter_count(model),
        "local_tuple_parameters": local_parameter_count(model),
    }
    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    init_dloc = model.local_tuple.D_loc_raw.detach().cpu().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probe_log: list[dict[str, Any]] = []
    steps_done = 0
    stream = hashlib.sha256()
    started = time.perf_counter()
    stopped_reason = "completed"

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, n_mol, n_steps, gnorm_sum, clip_hits = 0.0, 0, 0, 0.0, 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = schedule[epoch - 1][start : start + BATCH_SIZE]
            index_list = [int(i) for i in indices.tolist()]
            stream.update(np.asarray(index_list, np.int64).tobytes())
            batch = zftd.make_batch(fit_data, index_list, target_fit, device)
            prediction = model(batch, mask=cm.C6_MASK)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                entry = {"epoch": int(epoch), "step_in_epoch": 1, "clip_total_norm": total_norm}
                entry["W_loc_grad_norm"] = (
                    float(model.local_tuple.W_loc.grad.norm()) if model.local_tuple.W_loc.grad is not None else None
                )
                entry["D_loc_grad_norm"] = (
                    float(model.local_tuple.D_loc_raw.grad.norm())
                    if model.local_tuple.D_loc_raw.grad is not None else None
                )
                entry["W_loc_norm"] = float(model.local_tuple.W_loc.detach().norm())
                entry["D_loc_norm"] = float(model.local_tuple.D_loc_raw.detach().norm())
                entry["health"] = {k: v for k, v in model.local_tuple.last_stats.items()}
                probe_log.append(entry)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "clip_fraction": float(clip_hits / max(n_steps, 1)),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if epoch in soup_epochs:
            soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[{arm}:local] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"gnorm={curve[-1]['grad_norm']:.3g} clip={curve[-1]['clip_fraction']:.3f} "
                f"{curve[-1]['seconds']:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    if not soup and max_steps is not None:
        soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    dloc_drift = float((last_state["local_tuple.D_loc_raw"] - init_dloc).norm() / init_dloc.norm())
    wloc_norm = float(last_state["local_tuple.W_loc"].norm())

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "weight_key": WEIGHT_KEY[arm],
        "supervision_target": "g = y - c (frozen fit-only decomposition)",
        "seed": SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "steps_expected": int(epochs) * math.ceil(len(fit_data) / BATCH_SIZE),
        "stopped_reason": stopped_reason,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
        "soup_epochs": members if max_steps is None else [],
        "n_fit": int(len(fit_data)), "n_dev": int(len(dev_data)),
        "fold": {"seed": FOLD_SEED, "fit_idx_sha256": fold["fit_idx_sha256"], "dev_idx_sha256": fold["dev_idx_sha256"]},
        "target": {
            "source": str(zcdm.TARGETS_NPZ), "sha256": file_sha256(zcdm.TARGETS_NPZ),
            "definition": "g = y - c; constants fitted on 8000 fit rows only",
            "target_sha256": hashlib.sha256(np.ascontiguousarray(g, np.float64).tobytes()).hexdigest(),
        },
        "tuple_npz": {"path": str(TUPLE_NPZ), "sha256": file_sha256(TUPLE_NPZ)},
        "parameter_audit": audit,
        "init_state_hash": state_hash(init_state),
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "init_D_loc_sha256": _array_sha256(init_dloc.numpy()),
        "soup_D_loc_sha256": _array_sha256(soup_state["local_tuple.D_loc_raw"].numpy()),
        "soup_W_loc_sha256": _array_sha256(soup_state["local_tuple.W_loc"].numpy()),
        "soup_W_loc_norm": float(soup_state["local_tuple.W_loc"].norm()),
        "last_W_loc_norm": wloc_norm,
        "D_loc_relative_drift_last": dloc_drift,
        "schedule_sha256": schedule_hash,
        "data_stream_sha256": stream.hexdigest(),
        "curve": curve,
        "probe_log": probe_log,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if max_steps is not None:
        result["smoke_last_state_hash"] = state_hash(last_state)
        return result

    torch.save(init_state, out_dir / f"{arm}_init_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")

    raw_predictions: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = build_arm(arm, payload)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        raw_predictions[f"{state_name}_fit"] = evaluate_state(replay, fit_data, g_fit, device)
        raw_predictions[f"{state_name}_dev"] = evaluate_state(replay, dev_data, g_dev, device)
    np.savez_compressed(
        out_dir / f"{arm}_raw_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_predictions.items()},
    )

    b_soup = float(np.median(g_fit - raw_predictions["raw_soup_fit"]))
    b_init = float(np.median(g_fit - raw_predictions["init_fit"]))
    b_last = float(np.median(g_fit - raw_predictions["last_fit"]))
    result["calibration_b"] = {"init": b_init, "last": b_last, "raw_soup": b_soup}

    replay = build_arm(arm, payload)
    replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in soup_state.items()}, strict=True)
    batch_data = fit_data[:128]
    gpu_pred = evaluate_state(replay.to(device), batch_data, g_fit[:128], device)
    cpu_pred = evaluate_state(replay.to(torch.device("cpu")), batch_data, g_fit[:128], torch.device("cpu"))
    max_diff = float(np.max(np.abs(gpu_pred - cpu_pred))) if device.type != "cpu" else 0.0
    result["replay_max_abs_diff"] = max_diff
    if device.type != "cpu" and max_diff > REPLAY_TOL:
        raise RuntimeError(f"CPU/GPU replay maxdiff {max_diff} > {REPLAY_TOL}")
    write_json(out_dir / f"{arm}_meta.json", result)
    write_json(out_dir / f"{arm}_curve.json", curve)
    log(f"[{arm}:local] done steps={steps_done} wall={result['wall_clock_s']:.1f}s b={b_soup:.6f} replay={max_diff:.2e}")
    return result


# ---------------------------------------------------------------------------
# 7. operator checks (local, raw-graph side; frozen before any dev score)
# ---------------------------------------------------------------------------


def _production_operator_single(
    dloc: torch.Tensor,
    phi_rows: np.ndarray,
    atom_rows: np.ndarray,
    t: np.ndarray,
    a: np.ndarray,
    w_j: np.ndarray,
    w_i: np.ndarray,
    phi_mean: np.ndarray,
    phi_std: np.ndarray,
    phi_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Production path on one synthetic molecule (no batching)."""
    phi_t = torch.as_tensor(phi_rows, dtype=torch.float32).reshape(1, -1)
    n_tuples = int(np.asarray(t).size)
    atom_v = torch.as_tensor(np.asarray(atom_rows).reshape(-1), dtype=torch.long)
    if atom_v.numel() == 1:
        atom_v = atom_v.repeat(n_tuples)
    x = tuple_features(
        phi_t.repeat(n_tuples, 1),
        atom_v,
        torch.as_tensor(a, dtype=torch.long),
        torch.as_tensor(t, dtype=torch.long),
        torch.as_tensor(phi_mean, dtype=torch.float32),
        torch.as_tensor(phi_std, dtype=torch.float32),
        float(phi_scale),
    )
    with torch.no_grad():
        alpha = tied_tuple_codes(dloc, x)
    w_j_t = torch.as_tensor(w_j, dtype=torch.float32)
    w_i_t = torch.as_tensor(w_i, dtype=torch.float32)
    e_j = (alpha * w_j_t.unsqueeze(1)).sum(0).numpy()
    e_i = (alpha * w_i_t.unsqueeze(1)).sum(0).numpy()
    return e_j, e_i


def _env_cache_cross_check(graphs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], sample_indices: Sequence[int]) -> dict[str, Any]:
    """Independent per-root incidence from the frozen env cache's bond occurrences."""
    blob = torch.load(ENV_TRAIN_CACHE, map_location="cpu", weights_only=False)
    node_sizes = blob["node_sizes"].numpy().astype(np.int64)
    bond_sizes = blob["bond_sizes"].numpy().astype(np.int64)
    bond_root = blob["bond_root"].numpy().astype(np.int64)
    bond_type = blob["bond_type"].numpy().astype(np.int64)
    bond_u = blob["bond_u"].numpy().astype(np.int64)
    bond_v = blob["bond_v"].numpy().astype(np.int64)
    atom = blob["atom"].numpy().astype(np.int64)
    node_off = np.concatenate([[0], np.cumsum(node_sizes)])
    bond_off = np.concatenate([[0], np.cumsum(bond_sizes)])
    mismatches = 0
    roots_checked = 0
    for index in sample_indices:
        atom_types, edge_index, edge_attr = graphs[int(index)]
        count, _ = molecule_incidence(atom_types, edge_index, edge_attr)
        n = int(node_sizes[index])
        env_count = np.zeros((n, BOND_CATEGORIES, ATOM_CATEGORIES), dtype=np.int64)
        for row in range(int(bond_off[index]), int(bond_off[index + 1])):
            root = int(bond_root[row])
            u, v = int(bond_u[row]), int(bond_v[row])
            bond = int(bond_type[row])
            if u == root and v != root:
                env_count[root, bond, int(atom[int(node_off[index]) + v])] += 1
            elif v == root and u != root:
                env_count[root, bond, int(atom[int(node_off[index]) + u])] += 1
        if not np.array_equal(count, env_count):
            mismatches += 1
        roots_checked += n
    return {
        "molecules_checked": int(len(sample_indices)),
        "roots_checked": int(roots_checked),
        "molecule_level_mismatches": int(mismatches),
        "pass": bool(mismatches == 0),
        "source": "env_train.pt bond_root/bond_u/bond_v/bond_type/atom (independent cache)",
    }


def operator_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    payload = load_tuple_payload()
    fold = build_fold()
    phi, atom, node_sizes = _env_phi_atom()
    graphs = load_raw_train_graphs()
    phi_mean, phi_std, phi_scale = (
        payload.phi_mean.numpy(), payload.phi_std.numpy(), payload.phi_scale,
    )
    dloc = init_d_loc()
    checks: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION}

    # -- 1/2. incidence reference: independent scipy adjacency rebuild ---------
    import scipy.sparse as sp

    m_bad_degree = 0
    m_bad_bond_marginal = 0
    max_marginal_delta64 = 0.0
    max_marginal_delta32 = 0.0
    n_roots_checked = 0
    for index in range(0, 10000, 7):  # deterministic stride; full build did all 10000
        atom_types, edge_index, edge_attr = graphs[index]
        count, _stats = molecule_incidence(atom_types, edge_index, edge_attr)
        n = int(atom_types.shape[0])
        rows = np.concatenate([edge_index[0], edge_index[1]])
        cols = np.concatenate([edge_index[1], edge_index[0]])
        keep = rows != cols
        adj = sp.coo_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n)).tocsr()
        adj.data[:] = 1.0
        degree = np.asarray(adj.sum(axis=1)).reshape(-1)
        d, n_t, n_a, _wj, _wi = root_weights(count)
        if not np.array_equal(d.astype(np.int64), degree.astype(np.int64)):
            m_bad_degree += 1
        for t in range(BOND_CATEGORIES):
            ref = np.zeros(n, dtype=np.int64)
            cols_idx = np.nonzero(edge_attr == t)[0]
            for column in cols_idx:
                u, v = int(edge_index[0, column]), int(edge_index[1, column])
                if u == v:
                    continue
                ref[u] += 1
                ref[v] += 1
            if not np.array_equal(n_t[:, t].astype(np.int64), ref):
                # the raw directed count double-counts symmetric edges; rebuild canonical
                ref = np.zeros(n, dtype=np.int64)
                seen: set[tuple[int, int]] = set()
                for column in range(edge_index.shape[1]):
                    u, v = int(edge_index[0, column]), int(edge_index[1, column])
                    if u == v:
                        continue
                    key = (u, v) if u < v else (v, u)
                    if key in seen:
                        continue
                    seen.add(key)
                    if int(edge_attr[column]) == t:
                        ref[u] += 1
                        ref[v] += 1
                if not np.array_equal(n_t[:, t].astype(np.int64), ref):
                    m_bad_bond_marginal += 1
        for local in range(n):
            dd = d[local]
            if dd <= 0:
                continue
            ind = n_t[local][:, None] * n_a[local][None, :] / dd
            max_marginal_delta64 = max(
                max_marginal_delta64,
                float(np.max(np.abs(ind.sum(axis=1) - n_t[local]))),
                float(np.max(np.abs(ind.sum(axis=0) - n_a[local]))),
            )
            ind32 = (n_t[local][:, None].astype(np.float32) * n_a[local][None, :].astype(np.float32)) / np.float32(dd)
            max_marginal_delta32 = max(
                max_marginal_delta32,
                float(np.max(np.abs(ind32.astype(np.float64).sum(axis=1) - n_t[local]))),
                float(np.max(np.abs(ind32.astype(np.float64).sum(axis=0) - n_a[local]))),
            )
        n_roots_checked += n
    checks["adjacency_reference"] = {
        "molecules_checked": len(range(0, 10000, 7)),
        "roots_checked": n_roots_checked,
        "degree_mismatch_molecules": int(m_bad_degree),
        "bond_marginal_mismatch_molecules_independent": int(m_bad_bond_marginal),
        "max_marginal_delta_float64": float(max_marginal_delta64),
        "max_marginal_delta_float32": float(max_marginal_delta32),
        "float64_tol": TOL_FLOAT64,
        "float32_tol": TOL_FLOAT32,
        "pass": bool(m_bad_degree == 0 and m_bad_bond_marginal == 0
                     and max_marginal_delta64 <= TOL_FLOAT64 and max_marginal_delta32 <= TOL_FLOAT32),
    }

    checks["env_cache_cross_check"] = _env_cache_cross_check(graphs, list(range(0, 10000, 271)))

    # -- 3. batch / offset / shuffle equivalence on the production model ------
    _f, _p, fit_data, dev_data, _t = load_prepared()
    model_j = build_arm("J", payload)
    model_i = build_arm("I", payload)
    model_j.eval()
    model_i.eval()
    device = torch.device("cpu")
    chosen = [0, 1, 2, 413, 2001]
    target = torch.zeros(len(fit_data), dtype=torch.float32)

    def _root_codes(model: nn.Module, mol_list: Sequence[Any]) -> torch.Tensor:
        batch = zftd.make_batch(list(mol_list), list(range(len(mol_list))), target[: len(mol_list)], device)
        with torch.no_grad():
            return model.local_tuple.root_codes(batch)

    single_codes = []
    for index in chosen:
        item = fit_data[index]
        code = _root_codes(model_j, [item])
        single_codes.append(code)
    batched = _root_codes(model_j, [fit_data[index] for index in chosen])
    single_stacked = torch.cat(single_codes, dim=0)
    checks["single_vs_batch"] = {
        "molecules": chosen,
        "max_abs_diff": float((batched - single_stacked).abs().max()),
        "tol": TOL_FLOAT32,
        "pass": bool(float((batched - single_stacked).abs().max()) <= TOL_FLOAT32),
    }

    shuffled = [4, 0, 3, 1, 2]
    order_positions = [chosen[position] for position in shuffled]
    reordered_codes = _root_codes(model_j, [fit_data[position] for position in order_positions])
    row_counts = [int(fit_data[position].dict_phi.shape[0]) for position in order_positions]
    row_ptr = np.concatenate([[0], np.cumsum(row_counts)])
    pos_of = {position: index for index, position in enumerate(order_positions)}
    restore_rows = np.concatenate([
        np.arange(int(row_ptr[pos_of[chosen[original]]]), int(row_ptr[pos_of[chosen[original]] + 1]))
        for original in range(len(chosen))
    ])
    restored = reordered_codes[torch.as_tensor(restore_rows, dtype=torch.long)]
    checks["shuffled_order_restore"] = {
        "order": shuffled,
        "max_abs_diff_after_stable_restore": float((restored - batched).abs().max()),
        "tol": TOL_FLOAT32,
        "pass": bool(float((restored - batched).abs().max()) <= TOL_FLOAT32),
    }

    # minimal two-graph concatenation endpoint offset
    two = _root_codes(model_j, [fit_data[0], fit_data[1]])
    two_single = torch.cat([single_codes[0], single_codes[1]], dim=0)
    checks["two_graph_offset"] = {
        "max_abs_diff": float((two - two_single).abs().max()),
        "tol": TOL_FLOAT32,
        "pass": bool(float((two - two_single).abs().max()) <= TOL_FLOAT32),
    }
    model_i.local_tuple.weight_key = "joint"
    same_key_codes_i = _root_codes(model_i, [fit_data[index] for index in chosen])
    model_i.local_tuple.weight_key = "independent"
    checks["J_I_same_codes"] = {
        "D_loc_equal": bool(torch.equal(model_j.local_tuple.D_loc_raw, model_i.local_tuple.D_loc_raw)),
        "max_abs_diff_same_weights": float((same_key_codes_i - batched).abs().max()),
        "tol": 0.0,
        "pass": bool(
            torch.equal(model_j.local_tuple.D_loc_raw, model_i.local_tuple.D_loc_raw)
            and torch.equal(same_key_codes_i, batched)
        ),
    }

    # -- 4. synthetic witness: two stars, same marginals, swapped pairing ----
    star_phi = np.linspace(0.1, 0.9, PHI_DIM).astype(np.float32)
    atom_star = np.asarray([0], dtype=np.int64)  # centre atom type; tuples carry (t, a_u)
    # graph 1: A--single, B--double ; graph 2: A--double, B--single
    t1 = np.asarray([1, 2], np.int64); a1 = np.asarray([1, 2], np.int64)
    t2 = np.asarray([2, 1], np.int64); a2 = np.asarray([1, 2], np.int64)
    # operator-level counts
    c1 = np.zeros((BOND_CATEGORIES, ATOM_CATEGORIES)); c1[1, 1] = 1; c1[2, 2] = 1
    c2 = np.zeros((BOND_CATEGORIES, ATOM_CATEGORIES)); c2[2, 1] = 1; c2[1, 2] = 1
    d1, nt1, na1, _wj, _wi = root_weights(c1[None, :, :].astype(np.int64))
    d2, nt2, na2, _wj, _wi = root_weights(c2[None, :, :].astype(np.int64))
    ind1 = nt1[0][:, None] * na1[0][None, :] / d1[0]
    ind2 = nt2[0][:, None] * na2[0][None, :] / d2[0]
    # exact I support: full product support (t,a) with positive marginals (4 terms)
    t_union = np.asarray([1, 1, 2, 2], np.int64)
    a_union = np.asarray([1, 2, 1, 2], np.int64)
    wi_union = np.asarray([0.25, 0.25, 0.25, 0.25], np.float32)
    wj1_union = np.asarray([0.5, 0.0, 0.0, 0.5], np.float32)
    wj2_union = np.asarray([0.0, 0.5, 0.5, 0.0], np.float32)
    e_j1, e_i1 = _production_operator_single(dloc, star_phi, atom_star, t_union, a_union, wj1_union, wi_union, phi_mean, phi_std, phi_scale)
    e_j2, e_i2 = _production_operator_single(dloc, star_phi, atom_star, t_union, a_union, wj2_union, wi_union, phi_mean, phi_std, phi_scale)
    checks["synthetic_witness"] = {
        "construction": "star G1 {A--single, B--double}; G2 {A--double, B--single}",
        "raw_pairs_J1": [[int(t), int(a)] for t, a in zip(t1, a1)],
        "raw_pairs_J2": [[int(t), int(a)] for t, a in zip(t2, a2)],
        "C_differ": bool(np.any(c1 != c2)),
        "C_ind_identical": bool(np.allclose(ind1, ind2, atol=TOL_FLOAT64)),
        "marginals_identical": bool(
            np.allclose(nt1, nt2, atol=TOL_FLOAT64) and np.allclose(na1, na2, atol=TOL_FLOAT64)
        ),
        "e_J_differ": bool(np.max(np.abs(e_j1 - e_j2)) > 1e-6),
        "e_I_identical": bool(np.allclose(e_i1, e_i2, atol=TOL_FLOAT32)),
        "max_abs_e_I_diff": float(np.max(np.abs(e_i1 - e_i2))),
        "max_abs_e_J_diff": float(np.max(np.abs(e_j1 - e_j2))),
    }

    # -- 5. fit contrast statistics -----------------------------------------
    counts = np.diff(payload.pair_ptr.numpy())
    pair_root = np.repeat(np.arange(payload.pair_ptr.numel() - 1, dtype=np.int64), counts)
    pair_wj = payload.pair_wJ.numpy()
    pair_wi = payload.pair_wI.numpy()
    realized = pair_wj > 0
    differing = realized & (np.abs(pair_wj - pair_wi) > 0)
    fit_mask = np.zeros(payload.pair_ptr.numel() - 1, dtype=bool)
    for index in fold["fit_idx"].tolist():
        fit_mask[payload.root_base[index] : payload.root_base[index + 1]] = True
    fit_pair = fit_mask[pair_root]
    diff_norm = np.abs(pair_wj - pair_wi)
    checks["fit_contrast"] = {
        "fit_roots": int(fit_mask.sum()),
        "fit_pairs": int(fit_pair.sum()),
        "fit_pairs_with_wJ_ne_wI": int((fit_pair & (diff_norm > 0)).sum()),
        "fit_pairs_realized": int((fit_pair & realized).sum()),
        "fit_fraction_pairs_contrast": float((fit_pair & (diff_norm > 0)).sum() / max(int(fit_pair.sum()), 1)),
        "diff_norm_p50": float(np.percentile(diff_norm[fit_pair], 50)) if int(fit_pair.sum()) else None,
        "diff_norm_p95": float(np.percentile(diff_norm[fit_pair], 95)) if int(fit_pair.sum()) else None,
        "no_operator_contrast": bool(not np.any(fit_pair & (diff_norm > 0))),
    }
    contrast_roots = np.zeros(payload.pair_ptr.numel() - 1, dtype=np.int64)
    np.add.at(contrast_roots, pair_root[diff_norm > 0], 1)
    checks["fit_contrast"]["fit_roots_with_any_contrast"] = int((fit_mask & (contrast_roots > 0)).sum())
    checks["fit_contrast"]["fit_roots_with_any_contrast_fraction"] = float(
        (fit_mask & (contrast_roots > 0)).sum() / max(int(fit_mask.sum()), 1)
    )

    # -- 6. same phi, different C witness ------------------------------------
    # exact-duplicate raw phi rows among fit roots, plus a different incident C
    fit_roots = np.nonzero(fit_mask)[0]
    phi_groups: dict[bytes, list[int]] = {}
    for root in fit_roots.tolist():
        phi_groups.setdefault(phi[root].tobytes(), []).append(int(root))
    dup_groups = [group for group in phi_groups.values() if len(group) > 1]
    mol_cache: dict[int, np.ndarray] = {}

    def _c_row(root: int) -> np.ndarray:
        root_base_np = payload.root_base.numpy()
        molecule = int(np.searchsorted(root_base_np, root, side="right") - 1)
        if molecule not in mol_cache:
            mol_cache[molecule] = molecule_incidence(*graphs[molecule])[0]
        return mol_cache[molecule][root - int(root_base_np[molecule])]

    witness = None
    dup_groups_with_contrast = 0
    for group in dup_groups:
        c_rows = [_c_row(root) for root in group[:8]]
        contrast_pair = None
        for i in range(len(c_rows)):
            for j in range(i + 1, len(c_rows)):
                if not np.array_equal(c_rows[i], c_rows[j]):
                    contrast_pair = (i, j)
                    break
            if contrast_pair is not None:
                break
        if contrast_pair is not None:
            dup_groups_with_contrast += 1
            if witness is None:
                i, j = contrast_pair
                witness = {
                    "root_a": int(group[i]), "root_b": int(group[j]),
                    "phi_sha256": _array_sha256(phi[group[i]]),
                    "phi_max_abs_diff": float(np.max(np.abs(phi[group[i]] - phi[group[j]]))),
                    "C_a_sum": int(c_rows[i].sum()), "C_b_sum": int(c_rows[j].sum()),
                    "C_a_sha256": _array_sha256(c_rows[i], np.int64),
                    "C_b_sha256": _array_sha256(c_rows[j], np.int64),
                }
    checks["same_phi_different_C_witness"] = {
        "found": witness is not None,
        "witness": witness,
        "duplicate_phi_groups": int(len(dup_groups)),
        "duplicate_phi_groups_with_C_contrast": int(dup_groups_with_contrast),
        "note": "exact raw phi65 duplicates among fit roots; no matching threshold was tuned",
    }

    checks["all_ok"] = bool(
        checks["adjacency_reference"]["pass"]
        and checks["env_cache_cross_check"]["pass"]
        and checks["single_vs_batch"]["pass"]
        and checks["shuffled_order_restore"]["pass"]
        and checks["two_graph_offset"]["pass"]
        and checks["J_I_same_codes"]["pass"]
        and checks["synthetic_witness"]["C_differ"]
        and checks["synthetic_witness"]["C_ind_identical"]
        and checks["synthetic_witness"]["e_J_differ"]
        and checks["synthetic_witness"]["e_I_identical"]
    )
    write_json(out_dir / "operator_checks.json", checks)
    log(f"[operator-checks] all_ok={checks['all_ok']}")
    return checks


# ---------------------------------------------------------------------------
# 8. phase A: anchors, init identity, manifest
# ---------------------------------------------------------------------------


def phase_a(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    payload = load_tuple_payload()
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    k = np.asarray(tgt["k"], np.int64)
    k_dev = k[fold["dev_idx"]]

    # -- historical anchor B (M_g raw soup), recomputed once ---------------
    with np.load(REF_DIR / "M_raw_predictions.npz") as z:
        b_pred = {key: np.asarray(z[key], np.float64) for key in z.files}
    b_meta = json.loads((REF_DIR / "M_meta.json").read_text())
    b_bias = float(b_meta["calibration_b"]["raw_soup"])
    b_bias_local = float(np.median(g_fit - b_pred["raw_soup_fit"]))
    published = json.loads((REF_DIR / "analysis.json").read_text())
    anchor = {
        "arm": "B = M_g raw soup (historical, read-only)",
        "bias_from_meta": b_bias,
        "bias_recomputed": b_bias_local,
        "bias_max_abs_diff": abs(b_bias - b_bias_local),
        "fit_overall_cal_mae": float(np.mean(np.abs(g_fit - (b_pred["raw_soup_fit"] + b_bias)))),
        "dev_overall_cal_mae": float(np.mean(np.abs(g_dev - (b_pred["raw_soup_dev"] + b_bias)))),
        "dev_G0_cal_mae": float(np.mean(np.abs((g_dev - (b_pred["raw_soup_dev"] + b_bias))[k_dev == 0]))),
        "published_fit_overall_cal_mae": 0.029280,
        "published_dev_overall_cal_mae": 0.103717,
        "published_dev_G0_cal_mae": 0.101875,
        "published_analysis_arm": published["arms"]["dev"]["M"]["cal"],
        "fold_hashes_match": bool(
            fold["fit_idx_sha256"] == "7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9"
            and fold["dev_idx_sha256"] == "a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff"
        ),
        "schedule_hash": b_meta["schedule_sha256"],
        "steps": int(b_meta["steps_done"]),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    anchor["reproduction_max_abs_delta"] = max(
        abs(anchor["bias_max_abs_diff"]),
        abs(anchor["fit_overall_cal_mae"] - anchor["published_fit_overall_cal_mae"]),
        abs(anchor["dev_overall_cal_mae"] - anchor["published_dev_overall_cal_mae"]),
        abs(anchor["dev_G0_cal_mae"] - anchor["published_dev_G0_cal_mae"]),
    )
    write_json(out_dir / "historical_anchor_checks.json", anchor)

    # -- init identity: J / I / fresh-M / old M_init ------------------------
    model_j = build_arm("J", payload)
    model_i = build_arm("I", payload)
    model_ref = build_reference_m()
    # exact identity comparisons must exclude dropout (the training encoders carry dropout;
    # eval mode removes only the stochastic masks, matching the deployed prediction path)
    model_j.eval(); model_i.eval(); model_ref.eval()
    fixed_target = torch.zeros(128, dtype=torch.float32)
    batch = zftd.make_batch(fit_data[:128], list(range(128)), fixed_target, torch.device("cpu"))
    with torch.no_grad():
        out_j = model_j(batch, mask=cm.C6_MASK).view(-1).numpy()
        out_i = model_i(batch, mask=cm.C6_MASK).view(-1).numpy()
        out_ref = model_ref(batch, mask=cm.C6_MASK).view(-1).numpy()
    old_init = torch.load(REF_DIR / "M_init_state.pt", map_location="cpu", weights_only=False)
    new_sd = {key: value.detach().clone() for key, value in model_j.state_dict().items()}
    old_keys = sorted(old_init)
    shared_mismatch = [key for key in old_keys if key in new_sd and not torch.equal(old_init[key], new_sd[key])]
    missing_old_keys = [key for key in old_keys if key not in new_sd]
    new_only = [key for key in new_sd if key not in old_init]
    identity = {
        "fixed_batch": "first 128 fit rows",
        "max_abs_diff_J_vs_I": float(np.max(np.abs(out_j - out_i))),
        "max_abs_diff_J_vs_fresh_M": float(np.max(np.abs(out_j - out_ref))),
        "max_abs_diff_I_vs_fresh_M": float(np.max(np.abs(out_i - out_ref))),
        "old_M_init_keys": len(old_keys),
        "old_keys_mismatch": shared_mismatch,
        "old_keys_missing_in_new": missing_old_keys,
        "new_only_keys": new_only,
        "old_M_init_shared_exact": bool(not shared_mismatch and not missing_old_keys),
        "J_D_loc_sha256": _array_sha256(model_j.local_tuple.D_loc_raw.detach().numpy()),
        "I_D_loc_sha256": _array_sha256(model_i.local_tuple.D_loc_raw.detach().numpy()),
        "J_W_loc_all_zero": bool(float(model_j.local_tuple.W_loc.abs().sum()) == 0.0),
        "I_W_loc_all_zero": bool(float(model_i.local_tuple.W_loc.abs().sum()) == 0.0),
        "parameter_audit": {
            "total": int(sum(p.numel() for p in model_j.parameters())),
            "base_body": base_body_parameter_count(model_j),
            "bridge": bridge_parameter_count(model_j),
            "local_tuple": local_parameter_count(model_j),
        },
        "tuple_kappa": float(payload.kappa),
        "tuple_npz_sha256": file_sha256(TUPLE_NPZ),
    }
    write_json(out_dir / "init_identity.json", identity)

    # -- training environment consistency -----------------------------------
    env_checks = verify_env_cache(payload)
    meta = json.loads((RESULTS_DIR / "tuple_index_meta.json").read_text()) if (RESULTS_DIR / "tuple_index_meta.json").exists() else {}
    env_checks["kappa"] = float(payload.kappa)
    env_checks["pair_count"] = int(payload.pair_ptr[-1])
    env_checks["phi_scaler_fit_only"] = True
    write_json(out_dir / "tuple_environment_checks.json", env_checks)

    # -- input manifest ------------------------------------------------------
    sources = {
        "encoded_train": zjd.TRACK_ROOT / "results/zinc_static_dictionary_pair/cache/encoded_train.pt",
        "env_train": ENV_TRAIN_CACHE,
        "fold_objects": zw.PREP_BLOB,
        "raw_train_processed": ZINC_TRAIN_PROCESSED,
        "targets_npz": zcdm.TARGETS_NPZ,
        "tuple_npz": TUPLE_NPZ,
        "ref_M_raw_predictions": REF_DIR / "M_raw_predictions.npz",
        "ref_M_meta": REF_DIR / "M_meta.json",
        "ref_M_init_state": REF_DIR / "M_init_state.pt",
        "ref_analysis": REF_DIR / "analysis.json",
        "audit_formula_verification": zcdm.AUDIT_DIR / "formula_verification_per_molecule.csv",
        "audit_train_label": zcdm.AUDIT_DIR / "train_cycle_audit_label.csv",
    }
    manifest = {
        "files": {
            name: {
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() else None,
                "bytes": int(path.stat().st_size) if path.exists() else None,
            }
            for name, path in sources.items()
        },
        "missing": [name for name, path in sources.items() if not path.exists()],
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "raw_graph_file": "data/ZINC/subset/processed/train.pt (official train only)",
    }
    write_json(out_dir / "input_manifest.json", manifest)
    out = {
        "historical_anchor": anchor,
        "init_identity": identity,
        "env_checks": env_checks,
        "manifest_missing": manifest["missing"],
        "seconds": float(time.perf_counter() - t0),
    }
    write_json(out_dir / "phase_a.json", out)
    log(f"[phase-a] done in {out['seconds']:.1f}s")
    return out


# ---------------------------------------------------------------------------
# 9. analysis: three-arm gains, gates, tables
# ---------------------------------------------------------------------------


def _paired_bootstrap_gains(
    err_b: np.ndarray,
    err_j: np.ndarray,
    err_i: np.ndarray,
    *,
    seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """1000 paired bootstrap draws with shared resample indices."""
    abs_b, abs_j, abs_i = np.abs(err_b), np.abs(err_j), np.abs(err_i)
    if not (abs_b.size == abs_j.size == abs_i.size):
        raise RuntimeError("paired arrays must share length")

    def points(idx: np.ndarray | None = None) -> dict[str, float]:
        def mean(a: np.ndarray) -> float:
            return float(a.mean() if idx is None else a[idx].mean())
        return {
            "G_BJ": mean(abs_b) - mean(abs_j),
            "G_BI": mean(abs_b) - mean(abs_i),
            "G_IJ": mean(abs_i) - mean(abs_j),
        }

    rng = np.random.default_rng(int(seed))
    n = abs_b.size
    samples = {key: np.empty(int(n_boot), np.float64) for key in ("G_BJ", "G_BI", "G_IJ")}
    for draw in range(int(n_boot)):
        idx = rng.choice(n, size=n, replace=True)
        values = points(idx)
        for key, value in values.items():
            samples[key][draw] = value
    point = points(None)
    out: dict[str, Any] = {"n": int(n), "seed": int(seed), "n_boot": int(n_boot), "shared_indices": True}
    for key in ("G_BJ", "G_BI", "G_IJ"):
        out[key] = {
            "point": point[key],
            "ci95": [float(np.percentile(samples[key], 2.5)), float(np.percentile(samples[key], 97.5))],
        }
    out["identity_checks"] = {
        "point_identity_max_abs": abs(point["G_BJ"] - point["G_BI"] - point["G_IJ"]),
        "bootstrap_identity_max_abs": float(
            np.max(np.abs(samples["G_BJ"] - samples["G_BI"] - samples["G_IJ"]))
        ),
        "bootstrap_identity_within_tol": bool(
            float(np.max(np.abs(samples["G_BJ"] - samples["G_BI"] - samples["G_IJ"]))) < 1e-9
        ),
    }
    return out


def analyze(*, out_dir: Path, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    y = np.asarray(tgt["y"], np.float64)
    c = np.asarray(tgt["c"], np.float64)
    g = np.asarray(tgt["g"], np.float64)
    k = np.asarray(tgt["k"], np.int64)
    fit_idx, dev_idx = fold["fit_idx"], fold["dev_idx"]
    y_fit, y_dev = y[fit_idx], y[dev_idx]
    c_fit, c_dev = c[fit_idx], c[dev_idx]
    g_fit, g_dev = g[fit_idx], g[dev_idx]
    k_fit, k_dev = k[fit_idx], k[dev_idx]

    new_meta = {arm: json.loads((out_dir / f"{arm}_meta.json").read_text()) for arm in ARMS}
    new_pred: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        with np.load(out_dir / f"{arm}_raw_predictions.npz") as z:
            new_pred[arm] = {key: np.asarray(z[key], np.float64) for key in z.files}
    with np.load(REF_DIR / "M_raw_predictions.npz") as z:
        b_pred = {key: np.asarray(z[key], np.float64) for key in z.files}
    b_meta = json.loads((REF_DIR / "M_meta.json").read_text())
    b_bias = float(np.median(g_fit - b_pred["raw_soup_fit"]))

    contract_errors: list[str] = []
    for arm in ARMS:
        meta = new_meta[arm]
        if meta["parameter_audit"] != {
            "total_parameters": EXPECTED_TOTAL_PARAMETERS,
            "base_body_parameters": EXPECTED_BODY_PARAMETERS,
            "bridge_parameters": BRIDGE_PARAMETERS,
            "local_tuple_parameters": TUPLE_LOCAL_PARAMETERS,
        }:
            contract_errors.append(f"{arm}: parameter audit mismatch")
        if int(meta["steps_done"]) != 15120 or int(meta["steps_expected"]) != 15120:
            contract_errors.append(f"{arm}: step count {meta['steps_done']}")
        if meta["stopped_reason"] != "completed":
            contract_errors.append(f"{arm}: stopped_reason={meta['stopped_reason']}")
        if meta["official_valid_loaded"] or meta["official_test_loaded"]:
            contract_errors.append(f"{arm}: official split loaded")
        if abs(float(meta["replay_max_abs_diff"])) > REPLAY_TOL:
            contract_errors.append(f"{arm}: replay maxdiff {meta['replay_max_abs_diff']}")
        if new_meta[arm]["tuple_npz"]["sha256"] != file_sha256(TUPLE_NPZ):
            contract_errors.append(f"{arm}: tuple npz hash mismatch")
        for part in ("fit", "dev"):
            arr = new_pred[arm][f"raw_soup_{part}"]
            if arr.ndim != 1 or arr.shape[0] != (N_FIT if part == "fit" else N_DEV):
                contract_errors.append(f"{arm}: {part} prediction shape {arr.shape}")
        if meta["schedule_sha256"] != "7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65":
            contract_errors.append(f"{arm}: schedule hash mismatch")
    if new_meta["J"]["schedule_sha256"] != new_meta["I"]["schedule_sha256"]:
        contract_errors.append("J/I schedule hash mismatch")
    if new_meta["J"]["data_stream_sha256"] != new_meta["I"]["data_stream_sha256"]:
        contract_errors.append("J/I data stream hash mismatch")
    if new_meta["J"]["init_D_loc_sha256"] != new_meta["I"]["init_D_loc_sha256"]:
        contract_errors.append("J/I initial D_loc hash mismatch")

    np.savez_compressed(out_dir / "fold_indices.npz", fit_idx=fit_idx, dev_idx=dev_idx)
    np.savez_compressed(
        out_dir / "targets.npz",
        y=y, c=c, g=g, k=k, gid=np.asarray(tgt["gid"], np.int64), fit_idx=fit_idx, dev_idx=dev_idx,
    )

    b_bias_arm = {"B": b_bias}
    biases = {**b_bias_arm, **{arm: float(new_meta[arm]["calibration_b"]["raw_soup"]) for arm in ARMS}}
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "fold": {
            "fit_idx_sha256": fold["fit_idx_sha256"], "dev_idx_sha256": fold["dev_idx_sha256"],
            "n_fit": fold["n_fit"], "n_dev": fold["n_dev"],
            "fit_k_counts": {name: int(mask.sum()) for name, mask in group_masks(k_fit).items()},
            "dev_k_counts": {name: int(mask.sum()) for name, mask in group_masks(k_dev).items()},
        },
        "calibration_b": biases,
        "contract_errors": contract_errors,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "target_definition": "g = y - c; fit-only constants frozen in fit_only_targets.npz",
    }

    for part, target, kk in (("fit", g_fit, k_fit), ("dev", g_dev, k_dev)):
        report.setdefault("arms", {}).setdefault(part, {})
        for arm, pred in (("B", b_pred), ("J", new_pred["J"]), ("I", new_pred["I"])):
            raw = pred[f"raw_soup_{part}"]
            cal = raw + biases[arm]
            report["arms"][part][arm] = {
                "raw": _metric_table(target - raw, kk),
                "cal": _metric_table(target - cal, kk),
                "b": biases[arm],
            }

    g0 = k_dev == 0
    errs = {
        arm: {
            "raw": g_dev - (b_pred[f"raw_soup_dev"] if arm == "B" else new_pred[arm]["raw_soup_dev"]),
        }
        for arm in ("B", "J", "I")
    }
    for arm in ("B", "J", "I"):
        base = b_pred["raw_soup_dev"] if arm == "B" else new_pred[arm]["raw_soup_dev"]
        errs[arm]["cal"] = g_dev - (base + biases[arm])
    report["gains"] = {
        "dev_G0_cal": _paired_bootstrap_gains(errs["B"]["cal"][g0], errs["J"]["cal"][g0], errs["I"]["cal"][g0]),
        "dev_G0_raw": _paired_bootstrap_gains(errs["B"]["raw"][g0], errs["J"]["raw"][g0], errs["I"]["raw"][g0]),
        "dev_overall_cal": _paired_bootstrap_gains(errs["B"]["cal"], errs["J"]["cal"], errs["I"]["cal"]),
        "dev_overall_raw": _paired_bootstrap_gains(errs["B"]["raw"], errs["J"]["raw"], errs["I"]["raw"]),
    }
    for part, target in (("fit_G0_cal", g_fit[k_fit == 0]), ("fit_overall_cal", g_fit)):
        prefix = "fit_G0" if part == "fit_G0_cal" else "fit_overall"
        if prefix == "fit_G0":
            mask = k_fit == 0
            eb = target - (b_pred["raw_soup_fit"][mask] + biases["B"])
            ej = target - (new_pred["J"]["raw_soup_fit"][mask] + biases["J"])
            ei = target - (new_pred["I"]["raw_soup_fit"][mask] + biases["I"])
        else:
            eb = target - (b_pred["raw_soup_fit"] + biases["B"])
            ej = target - (new_pred["J"]["raw_soup_fit"] + biases["J"])
            ei = target - (new_pred["I"]["raw_soup_fit"] + biases["I"])
        report["gains"][f"{prefix}_cal"] = _paired_bootstrap_gains(eb, ej, ei)

    # contributions (mutually exclusive groups; sums equal overall)
    contribution_rows = []
    for part, kk, target in (("fit", k_fit, g_fit), ("dev", k_dev, g_dev)):
        for arm in ("B", "J", "I"):
            base = b_pred[f"raw_soup_{part}"] if arm == "B" else new_pred[arm][f"raw_soup_{part}"]
            err = target - (base + biases[arm])
            entry = _metric_table(err, kk)
            report.setdefault("contributions", {}).setdefault(part, {})[arm] = entry
            for name in GROUP_NAMES:
                info = entry[name]
                contribution_rows.append(
                    f"{part},{arm},{name},{info['n']},{info['mae']},{info['contribution']}"
                )
    report["contribution_sums"] = {
        part: {
            arm: float(sum(report["contributions"][part][arm][name]["contribution"] for name in GROUP_NAMES))
            for arm in ("B", "J", "I")
        }
        for part in ("fit", "dev")
    }

    # gates
    def _gate_a(arm: str) -> dict[str, Any]:
        g0_cal = report["gains"]["dev_G0_cal"][f"G_B{arm}"]
        g0_raw = report["gains"]["dev_G0_raw"][f"G_B{arm}"]
        ov_cal = report["gains"]["dev_overall_cal"][f"G_B{arm}"]
        ov_raw = report["gains"]["dev_overall_raw"][f"G_B{arm}"]
        passed = bool(
            g0_cal["point"] >= DELTA and g0_cal["ci95"][0] > 0
            and ov_cal["point"] >= DELTA and ov_cal["ci95"][0] > 0
            and g0_raw["point"] > 0 and ov_raw["point"] > 0
        )
        return {
            "G0_cal": g0_cal, "G0_raw": g0_raw, "overall_cal": ov_cal, "overall_raw": ov_raw,
            "pass": passed,
            "requires": "G0 cal>=0.003 & CI_low>0; overall cal>=0.003 & CI_low>0; G0/overall raw>0",
        }

    gain_ij_g0_cal = report["gains"]["dev_G0_cal"]["G_IJ"]
    gain_ij_g0_raw = report["gains"]["dev_G0_raw"]["G_IJ"]
    gain_ij_ov_cal = report["gains"]["dev_overall_cal"]["G_IJ"]
    raw_opposite = bool(
        (np.sign(gain_ij_g0_raw["point"]) == -np.sign(gain_ij_g0_cal["point"]) and abs(gain_ij_g0_raw["point"]) >= DELTA)
        or (np.sign(report["gains"]["dev_overall_raw"]["G_IJ"]["point"]) == -np.sign(gain_ij_ov_cal["point"])
            and abs(report["gains"]["dev_overall_raw"]["G_IJ"]["point"]) >= DELTA)
    )
    joint_support = bool(
        gain_ij_g0_cal["point"] >= DELTA and gain_ij_g0_cal["ci95"][0] > 0
        and gain_ij_g0_raw["point"] > 0 and gain_ij_ov_cal["point"] >= -DELTA_SLACK
    )
    indep_support = bool(
        gain_ij_g0_cal["point"] <= -DELTA and gain_ij_g0_cal["ci95"][1] < 0
        and gain_ij_g0_raw["point"] < 0 and gain_ij_ov_cal["point"] <= DELTA_SLACK
    )
    local_equivalence = bool(
        _ci_inside(gain_ij_g0_cal["ci95"], -DELTA, DELTA)
        and _ci_inside(gain_ij_ov_cal["ci95"], -DELTA, DELTA)
        and not raw_opposite
    )
    if contract_errors:
        category = "INVALID"
    elif joint_support:
        category = "JOINT_SUPPORT"
    elif indep_support:
        category = "INDEPENDENT_SUPPORT"
    elif local_equivalence:
        category = "LOCAL_EQUIVALENCE"
    else:
        category = "INCONCLUSIVE"
    report["gate"] = {
        "category": category,
        "gate_A_J": _gate_a("J"),
        "gate_A_I": _gate_a("I"),
        "gate_B": {
            "G0_cal": gain_ij_g0_cal, "G0_raw": gain_ij_g0_raw, "overall_cal": gain_ij_ov_cal,
            "overall_raw": report["gains"]["dev_overall_raw"]["G_IJ"],
            "definitions_checked": {
                "JOINT_SUPPORT": joint_support, "INDEPENDENT_SUPPORT": indep_support,
                "LOCAL_EQUIVALENCE": local_equivalence,
            },
            "raw_opposite_signal": raw_opposite,
            "rules": {
                "JOINT_SUPPORT": "G0 cal >= +0.003 & CI_low>0; G0 raw>0; overall cal >= -0.001",
                "INDEPENDENT_SUPPORT": "G0 cal <= -0.003 & CI_high<0; G0 raw<0; overall cal <= +0.001",
                "LOCAL_EQUIVALENCE": "G0 and overall cal CIs inside [-0.003,+0.003]; no raw>=0.003 opposite",
            },
        },
        "note": "performance gate A is exploratory against the historical B; gate B is the paired J/I test",
        "official_valid_loaded": False,
    }

    # sensitivity: drop the single dev row with the largest combined |err| of J/I/B
    combined = np.abs(errs["J"]["cal"]) + np.abs(errs["I"]["cal"]) + np.abs(errs["B"]["cal"])
    drop = int(np.argmax(combined))
    keep = np.ones(k_dev.size, dtype=bool)
    keep[drop] = False
    report["sensitivity"] = {
        "drop_rule": "max(|err_B|+|err_J|+|err_I|) on g-cal dev",
        "dropped_row_stable_id": int(dev_idx[drop]),
        "dropped_combined_error": float(combined[drop]),
        "overall_cal": {
            key: float(np.abs(errs[key]["cal"][keep]).mean()) for key in ("B", "J", "I")
        },
        "G0_cal": {
            key: float(np.abs(errs[key]["cal"][keep & g0]).mean()) for key in ("B", "J", "I")
        },
    }

    # bootstrap self-tests on the J/B pair (same invariants as the prior round)
    report["bootstrap_self_tests"] = _bootstrap_self_tests(errs["J"]["cal"], errs["B"]["cal"])

    write_json(out_dir / "analysis.json", report)
    write_json(out_dir / "gains.json", report["gains"])
    write_json(out_dir / "gate.json", report["gate"])
    write_json(out_dir / "bootstrap.json", {
        "gains": report["gains"], "self_tests": report["bootstrap_self_tests"],
        "sensitivity": report["sensitivity"],
    })

    header = "part,arm,group,n,mae,contribution"
    (out_dir / "group_table.csv").write_text(
        header + "\n" + "\n".join(contribution_rows) + "\n", encoding="utf-8"
    )
    main_rows = ["part,arm,raw_mae,cal_mae,raw_G0,cal_G0,raw_k-1,cal_k-1,raw_k-2,cal_k-2,raw_k<=-3,cal_k<=-3,raw_k<=-2,cal_k<=-2"]
    for part in ("fit", "dev"):
        for arm in ("B", "J", "I"):
            entry = report["arms"][part][arm]
            values: list[Any] = [part, arm, entry["raw"]["mae"], entry["cal"]["mae"]]
            for name in ("k=0", "k=-1", "k=-2", "k<=-3", "k<=-2"):
                values += [
                    entry["raw"][name]["mae"] if entry["raw"][name]["mae"] is not None else "",
                    entry["cal"][name]["mae"] if entry["cal"][name]["mae"] is not None else "",
                ]
            main_rows.append(",".join(str(v) for v in values))
    (out_dir / "main_table.csv").write_text("\n".join(main_rows) + "\n", encoding="utf-8")

    paired_rows = [
        "stable_id,k,y,g,c,err_B_cal,err_J_cal,err_I_cal,err_B_raw,err_J_raw,err_I_raw,"
        "abs_err_B_cal,abs_err_J_cal,abs_err_I_cal,gain_BJ_cal,gain_BI_cal,gain_IJ_cal"
    ]
    for position, stable in enumerate(dev_idx.tolist()):
        eb_cal, ej_cal, ei_cal = errs["B"]["cal"][position], errs["J"]["cal"][position], errs["I"]["cal"][position]
        eb_raw, ej_raw, ei_raw = errs["B"]["raw"][position], errs["J"]["raw"][position], errs["I"]["raw"][position]
        paired_rows.append(
            f"train:{int(stable):04d},{int(k_dev[position])},{y_dev[position]:.10f},{g_dev[position]:.10f},"
            f"{c_dev[position]:.10f},{eb_cal:.10f},{ej_cal:.10f},{ei_cal:.10f},"
            f"{eb_raw:.10f},{ej_raw:.10f},{ei_raw:.10f},"
            f"{abs(eb_cal):.10f},{abs(ej_cal):.10f},{abs(ei_cal):.10f},"
            f"{abs(eb_cal)-abs(ej_cal):.10f},{abs(eb_cal)-abs(ei_cal):.10f},{abs(ei_cal)-abs(ej_cal):.10f}"
        )
    (out_dir / "paired_gains.csv").write_text("\n".join(paired_rows) + "\n", encoding="utf-8")

    for part, part_idx in (("fit", fit_idx), ("dev", dev_idx)):
        target_g = g_fit if part == "fit" else g_dev
        target_y = y_fit if part == "fit" else y_dev
        kk = k_fit if part == "fit" else k_dev
        cc = c_fit if part == "fit" else c_dev
        lines = [f"stable_id,k,y,g,c,B_raw,B_cal,J_raw,J_cal,I_raw,I_cal"]
        b_raw = b_pred[f"raw_soup_{part}"]
        for position, stable in enumerate(part_idx.tolist()):
            lines.append(
                f"train:{int(stable):04d},{int(kk[position])},{target_y[position]:.10f},{target_g[position]:.10f},{cc[position]:.10f},"
                f"{b_raw[position]:.10f},{b_raw[position]+biases['B']:.10f},"
                f"{new_pred['J'][f'raw_soup_{part}'][position]:.10f},{new_pred['J'][f'raw_soup_{part}'][position]+biases['J']:.10f},"
                f"{new_pred['I'][f'raw_soup_{part}'][position]:.10f},{new_pred['I'][f'raw_soup_{part}'][position]+biases['I']:.10f}"
            )
        (out_dir / f"per_graph_{part}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(out_dir / "paired_gains.json", {
        "endpoint": "new-dev",
        "biases": biases,
        "gains": report["gains"],
        "gate": report["gate"],
    })
    log(f"[analysis] gate={category} B_J_G0cal={gain_ij_g0_cal['point']:.6f} "
        f"G_BJ={report['gains']['dev_G0_cal']['G_BJ']['point']:.6f} CI={report['gains']['dev_G0_cal']['G_BJ']['ci95']}")
    return report


# ---------------------------------------------------------------------------
# 10. mechanism: health + three frozen forward interventions
# ---------------------------------------------------------------------------


def _root_health(model: nn.Module, data_list: Sequence[Any], device: torch.device) -> dict[str, Any]:
    model.eval()
    n_rows = len(data_list)
    atom_used = torch.zeros(K_TUPLE, dtype=torch.bool)
    n_tuples = 0
    root_codes = []
    target = torch.zeros(n_rows, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, n_rows, 256):
            indices = list(range(start, min(start + 256, n_rows)))
            batch = zftd.make_batch(data_list, indices, target[: len(indices)], device)
            codes = model.local_tuple.root_codes(batch)
            stats = model.local_tuple.last_stats
            n_tuples += int(stats["n_tuples"])
            atom_used |= stats["atoms_used_mask"].to(torch.bool)
            root_codes.append(codes.cpu())
    stacked = torch.cat(root_codes, dim=0)
    std = stacked.std(dim=0)
    return {
        "n_roots": int(stacked.shape[0]),
        "n_tuples": int(n_tuples),
        "root_code_rms": float(stacked.pow(2).mean().sqrt()),
        "root_code_mean_abs": float(stacked.abs().mean()),
        "root_code_dead_dims_std_lt_1e-8": int((std < 1e-8).sum()),
        "root_code_nonzero_fraction": float((stacked != 0).float().mean()),
        "root_code_std_mean": float(std.mean()),
        "atoms_used_full": int(atom_used.sum()),
        "atoms_used_full_fraction": float(atom_used.float().mean()),
    }


def _tuple_health(model: nn.Module, data_list: Sequence[Any], device: torch.device) -> dict[str, Any]:
    """Per-tuple alpha health over a fixed subset (max 1024 molecules)."""
    model.eval()
    subset = list(data_list[:1024])
    target = torch.zeros(len(subset), dtype=torch.float32)
    nonzero_fraction = 0.0
    per_tuple = 0.0
    used = torch.zeros(K_TUPLE, dtype=torch.bool)
    n_tuples = 0
    with torch.no_grad():
        for start in range(0, len(subset), 128):
            indices = list(range(start, min(start + 128, len(subset))))
            batch = zftd.make_batch(subset, indices, target[: len(indices)], device)
            model.local_tuple.root_codes(batch)
            stats = model.local_tuple.last_stats
            n_tuples += int(stats["n_tuples"])
            nonzero_fraction += float(stats["alpha_nonzero_fraction"]) * int(stats["n_tuples"])
            per_tuple += float(stats["alpha_per_tuple_nonzero"]) * int(stats["n_tuples"])
            used |= stats["atoms_used_mask"].to(torch.bool)
    return {
        "molecules": len(subset),
        "n_tuples": int(n_tuples),
        "alpha_nonzero_fraction": float(nonzero_fraction / max(n_tuples, 1)),
        "alpha_per_tuple_nonzero": float(per_tuple / max(n_tuples, 1)),
        "atoms_used": int(used.sum()),
        "atoms_used_fraction": float(used.float().mean()),
        "expected_sparsity": SPARSITY,
        "sparsity_matches": bool(abs(per_tuple / max(n_tuples, 1) - SPARSITY) < 1e-3),
    }


def _predict_with(model: nn.Module, data_list: Sequence[Any], target_local: np.ndarray, device: torch.device) -> np.ndarray:
    return evaluate_state(model, data_list, target_local, device)


def collect_mechanism(*, out_dir: Path, device: torch.device, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    payload = load_tuple_payload()
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    k_dev = np.asarray(tgt["k"], np.int64)[fold["dev_idx"]]
    g0 = k_dev == 0
    new_meta = {arm: json.loads((out_dir / f"{arm}_meta.json").read_text()) for arm in ARMS}
    report: dict[str, Any] = {"arms": {}, "interventions": {}, "official_valid_loaded": False}

    for arm in ARMS:
        meta = new_meta[arm]
        init_state = torch.load(out_dir / f"{arm}_init_state.pt", map_location="cpu", weights_only=False)
        soup_state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False)
        last_state = torch.load(out_dir / f"{arm}_last_state.pt", map_location="cpu", weights_only=False)
        dloc0 = init_state["local_tuple.D_loc_raw"]
        entry = {
            "curve_summary": {
                "train_mae_epoch1": meta["curve"][0]["train_task_mae"],
                "train_mae_epoch120": meta["curve"][119]["train_task_mae"],
                "train_mae_epoch240": meta["curve"][-1]["train_task_mae"],
                "grad_norm_epoch1": meta["curve"][0]["grad_norm"],
                "clip_fraction_mean": float(np.mean([e["clip_fraction"] for e in meta["curve"]])),
                "seconds_total": float(sum(e["seconds"] for e in meta["curve"])),
            },
            "probe_log": meta["probe_log"],
            "D_loc_drift_last": float((last_state["local_tuple.D_loc_raw"] - dloc0).norm() / dloc0.norm()),
            "D_loc_drift_soup": float((soup_state["local_tuple.D_loc_raw"] - dloc0).norm() / dloc0.norm()),
            "W_loc_norm_last": float(last_state["local_tuple.W_loc"].norm()),
            "W_loc_norm_soup": float(soup_state["local_tuple.W_loc"].norm()),
            "D_loc_sha256_soup": _array_sha256(soup_state["local_tuple.D_loc_raw"].numpy()),
            "W_loc_sha256_soup": _array_sha256(soup_state["local_tuple.W_loc"].numpy()),
            "replay_max_abs_diff": float(meta["replay_max_abs_diff"]),
            "wall_clock_s": float(meta["wall_clock_s"]),
        }
        for state_name, state in (("init", init_state), ("raw_soup", soup_state)):
            model = build_arm(arm, payload)
            model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
            model.eval()
            # fixed batch probe for contribution / code activity with isolated RNG
            seed_everything(SEED)
            batch = zftd.make_batch(fit_data[:128], list(range(128)), torch.zeros(128), device)
            with torch.no_grad():
                base = model(batch, mask=cm.C6_MASK)
                model.local_tuple.ablate = True
                ablated = model(batch, mask=cm.C6_MASK)
                model.local_tuple.ablate = False
                stats = dict(model.local_tuple.last_stats)
            entry[f"{state_name}_probe"] = {
                "prediction_shift_when_loc_zeroed_max": float((ablated - base).abs().max()),
                "prediction_shift_when_loc_zeroed_mean": float((ablated - base).abs().mean()),
                "contribution_rms": stats.get("contribution_rms", 0.0),
                "e_rms_scaled": stats.get("e_rms_scaled", 0.0),
                "alpha_nonzero_fraction": stats.get("alpha_nonzero_fraction", 0.0),
                "alpha_per_tuple_nonzero": stats.get("alpha_per_tuple_nonzero", 0.0),
                "atoms_used": stats.get("atoms_used", None),
                "root_code_nonzero_fraction": stats.get("root_code_nonzero_fraction", 0.0),
                "root_code_dead_dims": stats.get("root_code_dead_dims", None),
                "W_loc_norm": stats.get("W_loc_norm", 0.0),
                "D_loc_norm": stats.get("D_loc_norm", 0.0),
                "rng_isolated": True,
            }
            if state_name == "raw_soup":
                entry["fit_health"] = _root_health(model, fit_data, device)
                entry["dev_health"] = _root_health(model, dev_data, device)
                entry["tuple_health_fit"] = _tuple_health(model, fit_data, device)
        report["arms"][arm] = entry

    # -- three frozen forward interventions on the soup ----------------------
    for arm in ARMS:
        other = "I" if arm == "J" else "J"
        soup_state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False)
        model = build_arm(arm, payload)
        model.load_state_dict({k: v.to(device) for k, v in soup_state.items()}, strict=True)
        model = model.to(device)
        model.eval()
        pred_orig = {"fit": _predict_with(model, fit_data, g_fit, device),
                     "dev": _predict_with(model, dev_data, g_dev, device)}
        model.local_tuple.ablate = True
        pred_zero = {"fit": _predict_with(model, fit_data, g_fit, device),
                     "dev": _predict_with(model, dev_data, g_dev, device)}
        model.local_tuple.ablate = False
        model.local_tuple.weight_key = WEIGHT_KEY[other]
        model.weight_key = WEIGHT_KEY[other]
        pred_switch = {"fit": _predict_with(model, fit_data, g_fit, device),
                       "dev": _predict_with(model, dev_data, g_dev, device)}
        bias = float(new_meta[arm]["calibration_b"]["raw_soup"])
        entry: dict[str, Any] = {"arm": arm, "switch_to": other}
        for part, target in (("fit", g_fit), ("dev", g_dev)):
            kk = np.asarray(tgt["k"], np.int64)[fold["fit_idx"]] if part == "fit" else k_dev
            g0_part = kk == 0
            orig = pred_orig[part]
            for label, pred in (("loc_zeroed", pred_zero[part]), ("operator_switched", pred_switch[part])):
                delta = pred - orig
                mae_orig = float(np.mean(np.abs(target - (orig + bias))))
                mae_new = float(np.mean(np.abs(target - (pred + bias))))
                entry[f"{part}_{label}"] = {
                    "prediction_delta_mean_abs": float(np.mean(np.abs(delta))),
                    "prediction_delta_p95_abs": float(np.percentile(np.abs(delta), 95)),
                    "prediction_delta_max_abs": float(np.max(np.abs(delta))),
                    "signed_mean": float(delta.mean()),
                    "mae_orig": mae_orig,
                    "mae_new": mae_new,
                    "mae_change": float(mae_new - mae_orig),
                    "G0_mae_orig": float(np.mean(np.abs(target[g0_part] - (orig[g0_part] + bias)))),
                    "G0_mae_new": float(np.mean(np.abs(target[g0_part] - (pred[g0_part] + bias)))),
                }
        report["interventions"][arm] = entry

    # J and I trained weights are compared at the code level too
    j_state = torch.load(out_dir / "J_raw_soup_state.pt", map_location="cpu", weights_only=False)
    i_state = torch.load(out_dir / "I_raw_soup_state.pt", map_location="cpu", weights_only=False)
    report["trained_weight_compare"] = {
        "D_loc_max_abs_diff": float((j_state["local_tuple.D_loc_raw"] - i_state["local_tuple.D_loc_raw"]).abs().max()),
        "D_loc_relative_diff": float(
            (j_state["local_tuple.D_loc_raw"] - i_state["local_tuple.D_loc_raw"]).norm()
            / j_state["local_tuple.D_loc_raw"].norm()
        ),
        "W_loc_max_abs_diff": float((j_state["local_tuple.W_loc"] - i_state["local_tuple.W_loc"]).abs().max()),
        "weight_keys": {"J": "joint", "I": "independent"},
        "note": "arms are trained independently from the same init; D_loc differs because the task gradient differs",
    }
    write_json(out_dir / "mechanism_health.json", report)
    return report


# ---------------------------------------------------------------------------
# 11. replay, figures, budget, manifest
# ---------------------------------------------------------------------------


def collect_replay(*, out_dir: Path, device: torch.device, log: Any = print) -> dict[str, Any]:
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    new_meta = {arm: json.loads((out_dir / f"{arm}_meta.json").read_text()) for arm in ARMS}
    checks: dict[str, Any] = {"arms": {}, "contract_errors": []}
    for arm in ARMS:
        with np.load(out_dir / f"{arm}_raw_predictions.npz") as z:
            pred = {key: np.asarray(z[key], np.float64) for key in z.files}
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False)
        model = build_arm(arm)
        model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
        model = model.to(device)
        replay_fit = _predict_with(model, fit_data[:128], g_fit[:128], device)
        replay_dev = _predict_with(model, dev_data[:128], g_dev[:128], device)
        entry = {
            "reloaded_soup_fit_replay_max_abs_diff": float(np.max(np.abs(replay_fit - pred["raw_soup_fit"][:128]))),
            "reloaded_soup_dev_replay_max_abs_diff": float(np.max(np.abs(replay_dev - pred["raw_soup_dev"][:128]))),
            "train_replay_max_abs_diff": float(new_meta[arm]["replay_max_abs_diff"]),
            "steps_done": int(new_meta[arm]["steps_done"]),
            "steps_expected": int(new_meta[arm]["steps_expected"]),
            "stopped_reason": new_meta[arm]["stopped_reason"],
        }
        entry["backbone_ok"] = bool(
            entry["reloaded_soup_fit_replay_max_abs_diff"] <= REPLAY_TOL
            and entry["reloaded_soup_dev_replay_max_abs_diff"] <= REPLAY_TOL
        )
        if not entry["backbone_ok"]:
            checks["contract_errors"].append(f"{arm}: replay maxdiff")
        checks["arms"][arm] = entry
    checks["schedule_sha256"] = {arm: new_meta[arm]["schedule_sha256"] for arm in ARMS}
    checks["data_stream_sha256"] = {arm: new_meta[arm]["data_stream_sha256"] for arm in ARMS}
    checks["schedule_match_between_arms"] = bool(
        new_meta["J"]["schedule_sha256"] == new_meta["I"]["schedule_sha256"]
    )
    checks["data_stream_match_between_arms"] = bool(
        new_meta["J"]["data_stream_sha256"] == new_meta["I"]["data_stream_sha256"]
    )
    checks["schedule_matches_frozen"] = bool(
        new_meta["J"]["schedule_sha256"] == "7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65"
    )
    checks["fold_match_expected"] = bool(
        fold["fit_idx_sha256"] == "7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9"
        and fold["dev_idx_sha256"] == "a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff"
    )
    checks["all_ok"] = bool(
        not checks["contract_errors"] and checks["schedule_match_between_arms"]
        and checks["data_stream_match_between_arms"] and checks["schedule_matches_frozen"]
        and checks["fold_match_expected"] and all(e["backbone_ok"] for e in checks["arms"].values())
    )
    write_json(out_dir / "replay_checks.json", checks)
    return checks


def make_figures(*, out_dir: Path, log: Any = print) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((out_dir / "analysis.json").read_text())
    mechanism = json.loads((out_dir / "mechanism_health.json").read_text())
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    arms = ["B(M_g)", "J", "I"]
    values = [report["arms"]["dev"][arm]["cal"]["k=0"]["mae"] for arm in ("B", "J", "I")]
    bars = ax.bar(arms, values, color=["#888888", "#1f77b4", "#d62728"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.5f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("dev G0 calibrated g-MAE")
    ax.set_title("Dev G0 (k=0) calibrated g-MAE")
    fig.tight_layout()
    fig.savefig(figures / "g0_three_arms.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm, color in (("J", "#1f77b4"), ("I", "#d62728")):
        curve = mechanism["arms"][arm]["curve_summary"]
        # per-epoch curves live in the meta JSON
        meta = json.loads((out_dir / f"{arm}_meta.json").read_text())
        ax.plot([e["epoch"] for e in meta["curve"]], [e["train_task_mae"] for e in meta["curve"]], color=color, label=f"{arm} train MAE")
        del curve
    ax.set_xlabel("epoch")
    ax.set_ylabel("train L1(g)")
    ax.set_yscale("log")
    ax.set_title("J/I training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, arm in zip(axes, ("J", "I")):
        probes = mechanism["arms"][arm]["probe_log"]
        epochs = [p["epoch"] for p in probes]
        ax.plot(epochs, [p["W_loc_grad_norm"] for p in probes], marker="o", label="W_loc grad norm")
        ax.plot(epochs, [p["D_loc_grad_norm"] for p in probes], marker="s", label="D_loc grad norm")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(f"{arm}: local-tuple task gradients")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "dictionary_task_gradients.png", dpi=160)
    plt.close(fig)
    log(f"[figures] wrote {figures}")


def make_budget(*, out_dir: Path, round_start: str, round_end: str) -> dict[str, Any]:
    meta = {arm: json.loads((out_dir / f"{arm}_meta.json").read_text()) for arm in ARMS}
    gpu_seconds = sum(float(meta[arm]["wall_clock_s"]) for arm in ARMS)
    budget = {
        "round_start_cst": round_start,
        "round_end_cst": round_end,
        "formal_arms": 2,
        "seed": SEED,
        "epochs": EPOCHS,
        "max_parallel_gpus": 2,
        "gpu_hours_training_only": gpu_seconds / 3600.0,
        "per_arm_seconds": {arm: float(meta[arm]["wall_clock_s"]) for arm in ARMS},
        "steps_per_arm": {arm: int(meta[arm]["steps_done"]) for arm in ARMS},
        "gpu_hours_limit": 1.2,
        "cpu_threads_cap": 8,
        "note": "budget is an upper bound; smoke/analysis seconds are recorded in EXECUTION.md",
    }
    write_json(out_dir / "budget.json", budget)
    return budget


def make_manifest(*, out_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(out_dir))
        if relative == "manifest.json":
            continue
        files[relative] = {"sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
    sources = {
        "runner": Path(__file__),
        "reference_runner": Path(zcdm.__file__),
        "protocol_md": out_dir / "PROTOCOL.md",
        "method_contract_md": out_dir / "METHOD_CONTRACT.md",
        "evidence_scope_md": out_dir / "EVIDENCE_SCOPE.md",
        "errata_md": out_dir / "ERRATA.md",
        "report_md": out_dir / "REPORT.md",
        "decision_md": out_dir / "DECISION.md",
        "execution_md": out_dir / "EXECUTION.md",
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "result_files": files,
        "source_files": {
            name: {"sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
            for name, path in sources.items() if path.exists()
        },
        "missing_source_files": [name for name, path in sources.items() if not path.exists()],
    }
    write_json(out_dir / "manifest.json", payload)
    return payload


# ---------------------------------------------------------------------------
# 12. smoke
# ---------------------------------------------------------------------------


def run_smoke(*, device: torch.device, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_tuple_payload()
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    y = np.asarray(tgt["y"], np.float64)
    g_fit = g[fold["fit_idx"]]
    y_fit = y[fold["fit_idx"]]
    checks: dict[str, Any] = {}

    checks["tuple_contract"] = {
        "npz_sha256": file_sha256(TUPLE_NPZ),
        "pair_count": int(payload.pair_ptr[-1]),
        "kappa": float(payload.kappa),
        "phi_scale": float(payload.phi_scale),
        "fold_hashes_expected": bool(
            fold["fit_idx_sha256"] == "7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9"
            and fold["dev_idx_sha256"] == "a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    model_j = build_arm("J", payload).to(device)
    model_i = build_arm("I", payload).to(device)
    model_j.eval(); model_i.eval()
    target_fit_all = torch.as_tensor(g_fit, dtype=torch.float32)
    target_y_all = torch.as_tensor(y_fit, dtype=torch.float32)
    batch_g = zftd.make_batch(fit_data, list(range(32)), target_fit_all, device)
    batch_y = zftd.make_batch(fit_data, list(range(32)), target_y_all, device)

    with torch.no_grad():
        pred_j = model_j(batch_g, mask=cm.C6_MASK)
        pred_i = model_i(batch_g, mask=cm.C6_MASK)
    checks["init_identity_J_I_max_abs"] = float((pred_j - pred_i).abs().max())
    checks["init_identity_mode"] = "eval (dropout disabled)"
    checks["init_identity_ok"] = bool(checks["init_identity_J_I_max_abs"] <= 1e-6)

    # dropout stream identity: with the same seed the two arms consume the same masks
    model_j.train(); model_i.train()
    seed_everything(SEED)
    model_j(batch_g, mask=cm.C6_MASK)
    rng_j = torch.get_rng_state().clone()
    seed_everything(SEED)
    model_i(batch_g, mask=cm.C6_MASK)
    rng_i = torch.get_rng_state().clone()
    model_j.eval(); model_i.eval()
    checks["dropout_stream_identical"] = bool(torch.equal(rng_j, rng_i))

    # target is never a model input
    with torch.no_grad():
        pred_y = model_j(batch_y, mask=cm.C6_MASK)
        pred_perm = model_j(
            zftd.make_batch(fit_data, list(range(32)), target_fit_all.flip(0).clone(), device),
            mask=cm.C6_MASK,
        )
    checks["target_not_read_maxdiff"] = float((pred_j - pred_y).abs().max())
    checks["label_permutation_maxdiff"] = float((pred_j - pred_perm).abs().max())
    checks["target_not_read_ok"] = bool(
        checks["target_not_read_maxdiff"] <= TARGET_NOT_READ_TOL
        and checks["label_permutation_maxdiff"] <= TARGET_NOT_READ_TOL
    )

    # loss semantics
    loss = F.l1_loss(pred_j.view(-1), batch_g.y.view(-1))
    checks["loss_is_l1_g"] = float(loss)

    # consumer + gradients (states discarded)
    model2 = build_arm("J", payload).to(device)
    model2.load_state_dict({k: v.to(device) for k, v in model_j.state_dict().items()}, strict=True)
    model2.train()
    batch = zftd.make_batch(fit_data, list(range(32)), target_fit_all, device)
    pred = model2(batch, mask=cm.C6_MASK)
    loss = F.l1_loss(pred.view(-1), batch.y.view(-1))
    model2.zero_grad()
    loss.backward()
    grad_w_first = float(model2.local_tuple.W_loc.grad.norm()) if model2.local_tuple.W_loc.grad is not None else 0.0
    grad_d_first = float(model2.local_tuple.D_loc_raw.grad.norm()) if model2.local_tuple.D_loc_raw.grad is not None else 0.0
    # first-step D_loc gradient is zero by construction (W_loc = 0); W_loc must be nonzero
    checks["w_loc_grad_first_step"] = grad_w_first
    checks["d_loc_grad_first_step"] = grad_d_first
    checks["w_loc_grad_nonzero"] = bool(grad_w_first > 0.0)

    # path establishment: after a non-zero W_loc the D_loc gradient exists
    model3 = build_arm("J", payload).to(device)
    model3.train()
    generator = torch.Generator(device=device).manual_seed(123)
    with torch.no_grad():
        model3.local_tuple.W_loc.normal_(0.0, 1e-3, generator=generator)
    pred3 = model3(batch, mask=cm.C6_MASK)
    loss3 = F.l1_loss(pred3.view(-1), batch.y.view(-1))
    model3.zero_grad()
    loss3.backward()
    grad_d_path = float(model3.local_tuple.D_loc_raw.grad.norm()) if model3.local_tuple.D_loc_raw.grad is not None else 0.0
    checks["d_loc_grad_after_w_loc_path"] = grad_d_path
    checks["d_loc_path_established"] = bool(grad_d_path > 0.0)
    with torch.no_grad():
        pred3_base = model3(batch, mask=cm.C6_MASK)
        model3.local_tuple.ablate = True
        pred3_ablate = model3(batch, mask=cm.C6_MASK)
        model3.local_tuple.ablate = False
    checks["ablate_shift_with_nonzero_w_loc"] = float((pred3_ablate - pred3_base).abs().max())
    checks["ablate_changes_function"] = bool(checks["ablate_shift_with_nonzero_w_loc"] > 0.0)

    # endpoint offsets: single vs batch
    groups = list(range(8))
    batch_grp = zftd.make_batch(fit_data, groups, target_fit_all, device)
    with torch.no_grad():
        grouped = model_j(batch_grp, mask=cm.C6_MASK).cpu().numpy()
        single = np.array([
            float(model_j(zftd.make_batch(fit_data, [i], target_fit_all, device), mask=cm.C6_MASK).item())
            for i in groups
        ])
    checks["endpoint_offset_maxdiff"] = float(np.max(np.abs(grouped - single)))
    checks["endpoint_offset_ok"] = bool(checks["endpoint_offset_maxdiff"] < 1e-5)

    for arm in ARMS:
        res = train_arm(arm, device=device, out_dir=out_dir / "smoke", epochs=1, max_steps=3, log=log)
        checks[f"tiny_train_{arm}"] = {
            "steps": res["steps_done"], "loss_curve": [e["train_task_mae"] for e in res["curve"]],
            "schedule_sha256": res["schedule_sha256"],
        }
    checks["tiny_train_schedule_match"] = bool(
        checks["tiny_train_J"]["schedule_sha256"] == checks["tiny_train_I"]["schedule_sha256"]
    )
    checks["all_ok"] = bool(
        checks["init_identity_ok"]
        and checks["dropout_stream_identical"]
        and checks["target_not_read_ok"]
        and checks["w_loc_grad_nonzero"]
        and checks["d_loc_path_established"]
        and checks["ablate_changes_function"]
        and checks["endpoint_offset_ok"]
        and checks["tiny_train_schedule_match"]
    )
    write_json(out_dir / "smoke_checks.json", checks)
    if not checks["all_ok"]:
        raise RuntimeError(f"smoke checks failed: {checks}")
    log(f"[smoke] all_ok={checks['all_ok']} replay_max={checks.get('replay_max_abs_diff')}")
    return checks


# ---------------------------------------------------------------------------
# 13. entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZINC local tuple dictionary: J vs I")
    parser.add_argument("--build-tuple", action="store_true")
    parser.add_argument("--operator-checks", action="store_true")
    parser.add_argument("--phase-a", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--round-start", default="2026-10-04 12:54:00 CST")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    device = zw.resolve_device(args.device)
    if args.build_tuple:
        print(json.dumps(jsonable(build_tuple_index()), indent=2)[:2000])
        return 0
    if args.operator_checks:
        print(json.dumps(jsonable(operator_checks(out_dir=out_dir)), indent=2)[:2000])
        return 0
    if args.phase_a:
        print(json.dumps(jsonable(phase_a(out_dir=out_dir)), indent=2)[:2000])
        return 0
    if args.smoke:
        run_smoke(device=device, out_dir=out_dir)
        return 0
    if args.arm:
        train_arm(args.arm, device=device, out_dir=out_dir, epochs=int(args.epochs), max_steps=args.max_steps)
        return 0
    if args.analyze:
        analyze(out_dir=out_dir, device=device)
        return 0
    if args.mechanism:
        collect_mechanism(out_dir=out_dir, device=device)
        return 0
    if args.replay:
        collect_replay(out_dir=out_dir, device=device)
        return 0
    if args.figures:
        make_figures(out_dir=out_dir)
        return 0
    if args.manifest:
        make_manifest(out_dir=out_dir)
        return 0
    if args.budget:
        import datetime

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
        make_budget(out_dir=out_dir, round_start=args.round_start, round_end=now)
        return 0
    parser.error("choose --build-tuple, --operator-checks, --phase-a, --smoke, --arm, --analyze, --mechanism, --replay, --figures, --manifest or --budget")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
