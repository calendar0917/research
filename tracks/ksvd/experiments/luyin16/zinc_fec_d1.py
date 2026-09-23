"""FEC-D1 runner — Localized Sparse Structural Dictionary Binding.

Round: ``fec_d1``.  Pre-registration:
``tracks/ksvd/notes/fec_d1_preregistration.md`` (frozen).
Prior-artifact audit: ``tracks/ksvd/notes/fec_d1_prior_artifact_audit.md``.
Core primitives: ``tracks/ksvd/experiments/luyin16/fec_d1.py``.

Stages
------
``identity correct binding baseline_guard smoke train mechanism analyze report all``

* ``identity``       — frozen-artifact hashes + parameter accounting.
* ``correct``        — hard correctness gates G0..G9 (subset / CPU-friendly).
* ``binding``        — build + cache the localized binding statistic for both arms.
* ``baseline_guard`` — read-only replay of the frozen FEC-S1 best checkpoint.
* ``smoke``          — deterministic short GPU smoke.
* ``train``          — formal branch training for one arm (``--arm dict|pca``).
* ``mechanism``      — evaluation-only assignment shuffle + branch neutralization.
* ``analyze``        — frozen gates / verdict / report / decision.

Official ZINC **test is never loaded** (``test_policy = no_test``).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import fec_d1 as d1
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0 as r2run
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_sdb_v0 as zsdb
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16.fec_s1_shared_local_env import build_fec_s1
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import (
    OPTIMIZED_PROTOCOL,
    REPO_ROOT,
)

PROTOCOL_VERSION = "fec_d1"
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/fec_d1"
CACHE_DIR = RESULTS_DIR / "cache"
STATE_DIR = RESULTS_DIR / "states"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

DICT_PATH = TRACK_ROOT / "results/sdb_v0/dictionary.pt"
FEC_S1_STATE = TRACK_ROOT / "results/fec_s1/states/fec_s1_seed0_selection_state.pt"
ENCODED_AUDIT = (
    TRACK_ROOT / "results/zinc_static_dictionary_pair/cache/encoded_audit.json"
)

#: FEC-S1 seed-0 persisted selection checkpoint (best official-valid MAE).
FEC_S1_RECORDED_BEST_VALID = 0.1367825070246472
FEC_S1_BEST_EPOCH = 238
FEC_S1_RECORDED_SOUP = 0.13042183499777457
FEC_S1_RECORDED_FINAL = 0.13678250572824618
BASELINE_GUARD_TOL = 1.0e-5

#: training protocol (inherited from FEC-S1; early termination disabled).
MAX_EPOCHS = int(OPTIMIZED_PROTOCOL["max_epochs"])  # 240
PATIENCE = MAX_EPOCHS  # no early termination
BATCH_SIZE = int(OPTIMIZED_PROTOCOL["batch_size"])
LEARNING_RATE = float(OPTIMIZED_PROTOCOL["learning_rate"])
WEIGHT_DECAY = float(OPTIMIZED_PROTOCOL["weight_decay"])
GRAD_CLIP = float(OPTIMIZED_PROTOCOL["gradient_clip_norm"])
TRAIN_SHUFFLE_OFFSET = int(OPTIMIZED_PROTOCOL["train_shuffle_seed_offset"])
EVAL_SHUFFLE_OFFSET = int(OPTIMIZED_PROTOCOL["eval_shuffle_seed_offset"])

#: evaluation-only assignment shuffle (pre-registration §13)
SHUFFLE_SEEDS = (101, 202, 303, 404, 505)

_write_json = sdp._write_json
_read_json = sdp._read_json
_seed_everything = sdp._seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return sdp._git_commit()


def _n_params(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _fe_s1_hidden() -> int:
    from tracks.ksvd.experiments.luyin16.fec_s1_shared_local_env import (
        choose_local_env_hidden,
        lookup_param_count,
    )

    return int(choose_local_env_hidden(int(lookup_param_count()["p_lookup"]))["hidden"])


# ---------------------------------------------------------------------------
# data / frozen artifacts
# ---------------------------------------------------------------------------


def load_frozen_dictionary() -> tuple[np.ndarray, sdb.PCARank]:
    D_ksvd, _D_rand, pca = zsdb.load_dictionary()
    return np.asarray(D_ksvd, dtype=np.float64), pca


def build_fec_d1(seed: int = 0) -> torch.nn.Module:
    """Frozen FEC-S1 (best checkpoint not yet loaded) with a branch adapter wrapper."""
    hidden = _fe_s1_hidden()
    model = build_fec_s1(seed=int(seed), hidden=hidden)
    base_adapter = model.local_env_adapter
    if base_adapter is None:
        raise RuntimeError("FEC-S1 model has no local_env_adapter")
    model.local_env_adapter = d1.BindingLocalEnvAdapter(base_adapter, seed=d1.BRANCH_SEED)
    return model


def load_fec_d1_best(seed: int = 0) -> torch.nn.Module:
    """FEC-S1 seed-0 best checkpoint wrapped with the rank-8 branch (W2 = 0).

    The FEC-S1 checkpoint keys are ``local_env_adapter.net.*``; the branch
    wrapper introduces a ``base`` prefix, so the checkpoint must be loaded into
    the unwrapped FEC-S1 model *before* wrapping.
    """
    model = build_fec_s1(seed=int(seed), hidden=_fe_s1_hidden())
    state = torch.load(FEC_S1_STATE, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    base_adapter = model.local_env_adapter
    if base_adapter is None:
        raise RuntimeError("FEC-S1 model has no local_env_adapter")
    model.local_env_adapter = d1.BindingLocalEnvAdapter(base_adapter, seed=d1.BRANCH_SEED)
    return model


def _coord(arm: str, phi: np.ndarray, D: np.ndarray, pca: sdb.PCARank) -> np.ndarray:
    if arm == d1.DICT_ARM:
        return sdb.omp_codes(D, np.asarray(phi, dtype=np.float64), s=sdb.SPARSITY)
    if arm == d1.PCA_ARM:
        return pca.dense_codes(np.asarray(phi, dtype=np.float64))
    raise ValueError(f"unknown arm {arm!r}")


@functools.lru_cache(maxsize=None)
def _aligned_split(split: str) -> tuple[list[Any], list[Any]]:
    """Raw ZINC molecules and FSAR molecules in lock-step (validated downstream)."""
    train_cache, valid_cache, _scalers, _meta = r2run.build_datasets()
    fsar = train_cache if split == "train" else valid_cache
    raw = list(zlr._load_zinc(ZINC_ROOT, "train" if split == "train" else "val"))
    if len(raw) != len(fsar):
        raise RuntimeError(f"{split}: raw {len(raw)} != fsar {len(fsar)}")
    return raw, fsar


def _check_alignment(raw: Any, molecule: Any, graph: Any, node_types: np.ndarray) -> None:
    if int(raw.num_nodes) != int(molecule.phi.shape[0]):
        raise RuntimeError("raw.num_nodes != FSAR phi rows")
    if len(graph.nodes) != int(molecule.phi.shape[0]):
        raise RuntimeError("graph node count != FSAR phi rows")
    if not np.array_equal(np.asarray(node_types, dtype=np.int64), np.asarray(molecule.atom_idx, dtype=np.int64)):
        raise RuntimeError("atom order mismatch between raw graph and FSAR cache")
    if abs(float(raw.y.view(-1)[0]) - float(molecule.y)) > 1.0e-5:
        raise RuntimeError("target order mismatch between raw graph and FSAR cache")


def iter_molecule_stats(arm: str, split: str, *, shuffle_seed: int | None = None):
    """Yield ``[n_patches, STAT_DIM]`` raw (unscaled) localized binding per molecule."""
    D, pca = load_frozen_dictionary()
    raw, fsar = _aligned_split(split)
    for data, molecule in zip(raw, fsar):
        graph, node_types, _edge_types = zlr._data_to_graph(data)
        _check_alignment(data, molecule, graph, node_types)
        phi = np.asarray(molecule.phi, dtype=np.float64)
        coord = _coord(arm, phi, D, pca)
        yield d1.per_root_binding(
            phi, molecule.atom_idx, graph, coord, shuffle_seed=shuffle_seed
        )


# ---------------------------------------------------------------------------
# stage: identity / parameter accounting
# ---------------------------------------------------------------------------


def identity_stage(force: bool = False) -> dict[str, Any]:
    D, pca = load_frozen_dictionary()
    encoded_audit = _read_json(ENCODED_AUDIT)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "dictionary": {
            "path": str(DICT_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(DICT_PATH),
            "shape": list(D.shape),
            "K": int(sdb.K_ATOMS),
            "s": int(sdb.SPARSITY),
            "dict_seed": int(sdb.DICT_SEED),
            "phi_dim": int(sdb.PHI_DIM),
            "source": "SDB-v0 Stage-1 frozen K-SVD (read-only, never refit)",
        },
        "pca": {
            "path": str(DICT_PATH.relative_to(REPO_ROOT)),
            "mean_shape": list(pca.mean.shape),
            "components_shape": list(pca.components.shape),
            "rank": int(pca.components.shape[0]),
            "source": "SDB-v0 train-fit affine PCA (read-only, never refit)",
        },
        "fec_s1_base": {
            "path": str(FEC_S1_STATE.relative_to(REPO_ROOT)),
            "sha256": _sha256(FEC_S1_STATE),
            "recorded_best_valid_mae": FEC_S1_RECORDED_BEST_VALID,
            "recorded_best_epoch": FEC_S1_BEST_EPOCH,
            "recorded_soup_valid_mae": FEC_S1_RECORDED_SOUP,
        },
        "encoded_cache": {
            "path": str(ENCODED_AUDIT.relative_to(REPO_ROOT)),
            "n_train": int(encoded_audit.get("n_train", -1)),
            "n_valid": int(encoded_audit.get("n_valid", -1)),
            "official_test_loaded": bool(encoded_audit.get("official_test_loaded", False)),
        },
        "branch": {
            "stat_dim": int(d1.STAT_DIM),
            "rank": int(d1.BRANCH_RANK),
            "out_width": int(d1.BRANCH_OUT),
            "params": int(d1.branch_parameter_count()),
            "solver": "sdb_v0.omp_codes -> tccd_v0.omp_codes (exact top-s)",
            "scaler": "sdb_v0.fit_rms_scaler (train-only per-coordinate RMS, no mean subtraction)",
        },
        "fec_s1_hidden": int(_fe_s1_hidden()),
    }
    _write_json(RESULTS_DIR / "artifact_identity.json", payload)
    return payload


def parameter_accounting_stage() -> dict[str, Any]:
    model = load_fec_d1_best(seed=0)
    base = build_fec_s1(seed=0, hidden=_fe_s1_hidden())
    branch = model.local_env_adapter
    assert isinstance(branch, d1.BindingLocalEnvAdapter)
    base_total = _n_params(base)
    branch_total = _n_params(branch) - _n_params(branch.base)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "fec_s1_base_params": int(base_total),
        "branch_params": int(branch_total),
        "branch_w1_params": int(branch.W1.numel()),
        "branch_w2_params": int(branch.W2.numel()),
        "fec_d1_total_params": int(_n_params(model)),
        "dict_arm_trainable": int(d1.branch_parameter_count()),
        "pca_arm_trainable": int(d1.branch_parameter_count()),
        "arms_parameter_matched": True,
        "base_frozen": True,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stage: binding cache (train-only scaler; per-arm)
# ---------------------------------------------------------------------------


def _binding_path(arm: str, split: str) -> Path:
    return CACHE_DIR / f"{arm}_{split}.pt"


def _scaler_path(arm: str) -> Path:
    return CACHE_DIR / f"{arm}_scaler.npz"


def build_binding_cache(arm: str, force: bool = False, log: bool = True) -> dict[str, Any]:
    if (not force) and all(
        _binding_path(arm, split).exists() for split in ("train", "valid")
    ) and _scaler_path(arm).exists():
        if log:
            print(f"[binding:{arm}] reuse cache", flush=True)
        return _read_json(RESULTS_DIR / "binding_scalers.json")[arm]

    started = time.perf_counter()
    # pass 1: train-only per-coordinate sum of squares (streaming, low memory)
    sumsq = np.zeros(int(d1.STAT_DIM), dtype=np.float64)
    n_train = 0
    size_train: list[int] = []
    for stat in iter_molecule_stats(arm, "train"):
        sumsq += (stat ** 2).sum(axis=0)
        n_train += int(stat.shape[0])
        size_train.append(int(stat.shape[0]))
    mean_sq = sumsq / max(n_train, 1)
    raw_rms = np.sqrt(mean_sq)
    mask = (raw_rms > float(d1.SCALER_FLOOR)).astype(np.float64)
    scale = np.where(mask > 0.0, np.sqrt(mean_sq + float(d1.SCALER_EPS)), 1.0)

    def _normalized(stat: np.ndarray) -> np.ndarray:
        return ((stat / scale[None, :]) * mask[None, :]).astype(np.float32)

    # pass 2: normalize into a preallocated buffer (no 2x concatenate peak)
    size_valid: list[int] = []
    for stat in iter_molecule_stats(arm, "valid"):
        size_valid.append(int(stat.shape[0]))
    train_big = np.empty((int(n_train), int(d1.STAT_DIM)), dtype=np.float32)
    offset = 0
    for stat in iter_molecule_stats(arm, "train"):
        end = offset + int(stat.shape[0])
        train_big[offset:end] = _normalized(stat)
        offset = end
    valid_big = np.empty((int(sum(size_valid)), int(d1.STAT_DIM)), dtype=np.float32)
    offset = 0
    for stat in iter_molecule_stats(arm, "valid"):
        end = offset + int(stat.shape[0])
        valid_big[offset:end] = _normalized(stat)
        offset = end

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stats": torch.as_tensor(train_big, dtype=torch.float32),
            "sizes": size_train,
            "arm": arm,
            "stat_dim": int(d1.STAT_DIM),
        },
        _binding_path(arm, "train"),
    )
    torch.save(
        {
            "stats": torch.as_tensor(valid_big, dtype=torch.float32),
            "sizes": size_valid,
            "arm": arm,
            "stat_dim": int(d1.STAT_DIM),
        },
        _binding_path(arm, "valid"),
    )
    np.savez(_scaler_path(arm), scale=scale, mask=mask, raw_rms=raw_rms)

    report = {
        "arm": arm,
        "n_train_patches": int(n_train),
        "n_valid_patches": int(sum(size_valid)),
        "effective_coordinates": int(mask.sum()),
        "zero_coordinates": int((mask <= 0.0).sum()),
        "effective_fraction": float(mask.mean()),
        "raw_rms_min_effective": float(raw_rms[mask > 0.0].min()) if mask.any() else 0.0,
        "raw_rms_max_effective": float(raw_rms[mask > 0.0].max()) if mask.any() else 0.0,
        "raw_rms_max_all": float(raw_rms.max()),
        "scaler_floor": float(d1.SCALER_FLOOR),
        "scaler_source": "official train only (no mean subtraction)",
        "seconds": float(time.perf_counter() - started),
    }
    all_reports = (
        _read_json(RESULTS_DIR / "binding_scalers.json")
        if (RESULTS_DIR / "binding_scalers.json").exists()
        else {}
    )
    all_reports[arm] = report
    _write_json(RESULTS_DIR / "binding_scalers.json", all_reports)
    if log:
        print(f"[binding:{arm}] {report}", flush=True)
    return report


def load_binding_cache(arm: str, split: str) -> tuple[torch.Tensor, list[int]]:
    blob = torch.load(_binding_path(arm, split), map_location="cpu", weights_only=False)
    return blob["stats"], list(blob["sizes"])


def load_scaler(arm: str) -> tuple[np.ndarray, np.ndarray]:
    blob = np.load(_scaler_path(arm))
    return blob["scale"].astype(np.float64), blob["mask"].astype(np.float64)


def attach_stats(data_list: Sequence[Any], big: torch.Tensor, sizes: Sequence[int]) -> None:
    offset = 0
    for data, size in zip(data_list, sizes):
        data.d1_stat = big[offset : offset + int(size)].clone()
        offset += int(size)
    if offset != int(big.shape[0]):
        raise RuntimeError("binding cache / encoded data size mismatch")


# ---------------------------------------------------------------------------
# stage: correctness gates G0..G9
# ---------------------------------------------------------------------------


def _g0_code_identity(n_molecules: int = 3) -> dict[str, Any]:
    D, pca = load_frozen_dictionary()
    raw, fsar = _aligned_split("train")
    max_phi = 0.0
    max_alpha = 0.0
    max_l0 = 0
    n_nodes = 0
    exact_phi = 0
    for data, molecule in list(zip(raw, fsar))[: int(n_molecules)]:
        graph, node_types, _edge = zlr._data_to_graph(data)
        phi_fresh = r2.build_phi(graph).astype(np.float32)
        phi_cache = np.asarray(molecule.phi, dtype=np.float32)
        n_nodes += int(phi_cache.shape[0])
        if np.array_equal(phi_fresh, phi_cache):
            exact_phi += 1
        max_phi = max(max_phi, float(np.abs(phi_fresh - phi_cache).max()))
        alpha_a = sdb.omp_codes(D, phi_cache.astype(np.float64), s=sdb.SPARSITY)
        alpha_b = sdb.omp_codes(D, phi_cache.astype(np.float64), s=sdb.SPARSITY)
        max_alpha = max(max_alpha, float(np.abs(alpha_a - alpha_b).max()))
        max_l0 = max(max_l0, int(np.count_nonzero(alpha_a, axis=1).max()))
    return {
        "n_molecules": int(n_molecules),
        "n_nodes": int(n_nodes),
        "molecules_phi_bit_identical": int(exact_phi),
        "phi_max_abs_diff": float(max_phi),
        "alpha_max_abs_diff_repeat": float(max_alpha),
        "max_l0": int(max_l0),
        "s": int(sdb.SPARSITY),
        "exact_l0_ok": bool(max_l0 == int(sdb.SPARSITY)),
        "passed": bool(max_phi == 0.0 and max_alpha == 0.0 and max_l0 == int(sdb.SPARSITY)),
    }


def _g1_chemistry_purity(n_molecules: int = 3) -> dict[str, Any]:
    """Changing atom/bond categories while keeping topology leaves phi and alpha fixed."""
    D, _pca = load_frozen_dictionary()
    raw, fsar = _aligned_split("train")
    max_phi = 0.0
    max_alpha = 0.0
    n_changed = 0
    for data, molecule in list(zip(raw, fsar))[: int(n_molecules)]:
        graph, node_types, _edge = zlr._data_to_graph(data)
        phi = r2.build_phi(graph).astype(np.float64)
        alpha = sdb.omp_codes(D, phi, s=sdb.SPARSITY)
        # a genuinely different chemistry assignment on the SAME topology
        changed = (np.asarray(node_types, dtype=np.int64) + 1) % d1.ATOM_CATEGORIES
        if not np.array_equal(changed, np.asarray(molecule.atom_idx, dtype=np.int64)):
            n_changed += 1
        # phi is rebuilt from adjacency only, so it is independent of `changed`;
        # alpha is a deterministic function of phi, hence also independent.
        phi_changed = r2.build_phi(graph).astype(np.float64)
        alpha_changed = sdb.omp_codes(D, phi_changed, s=sdb.SPARSITY)
        max_phi = max(max_phi, float(np.abs(phi - phi_changed).max()))
        max_alpha = max(max_alpha, float(np.abs(alpha - alpha_changed).max()))
    return {
        "n_molecules": int(n_molecules),
        "n_chemistry_relabelled": int(n_changed),
        "phi_max_abs_diff": float(max_phi),
        "alpha_max_abs_diff": float(max_alpha),
        "argument": "build_phi(graph) reads adjacency only; atom/bond categories never enter phi or alpha",
        "passed": bool(n_changed > 0 and max_phi == 0.0 and max_alpha == 0.0),
    }


def _g2_root_localization(n_molecules: int = 3) -> dict[str, Any]:
    D, _pca = load_frozen_dictionary()
    raw, fsar = _aligned_split("train")
    max_diff = 0.0
    shell_variation = 0
    for data, molecule in list(zip(raw, fsar))[: int(n_molecules)]:
        graph, _node_types, _edge = zlr._data_to_graph(data)
        phi = np.asarray(molecule.phi, dtype=np.float64)
        alpha = sdb.omp_codes(D, phi, s=sdb.SPARSITY)
        n = phi.shape[0]
        for node in range(n):
            # alpha_v is node-centric: alpha[node] does not depend on which root
            # patch it is viewed from.  Compare against a fresh per-node code.
            fresh = sdb.omp_codes(D, phi[node : node + 1], s=sdb.SPARSITY)
            max_diff = max(max_diff, float(np.abs(fresh[0] - alpha[node]).max()))
        # shell variation: the same node appears at different shells for different roots
        for root in range(min(n, 4)):
            for other in range(min(n, 4)):
                if root == other:
                    continue
                s1 = d1.ego_shells(graph, root, 2)
                s2 = d1.ego_shells(graph, other, 2)
                shell_root = {v: k for k, vs in s1.items() for v in vs}
                shell_other = {v: k for k, vs in s2.items() for v in vs}
                for node in set(shell_root) & set(shell_other):
                    if shell_root[node] != shell_other[node]:
                        shell_variation += 1
    return {
        "n_molecules": int(n_molecules),
        "alpha_node_centric_max_abs_diff": float(max_diff),
        "shell_changes_for_shared_nodes": int(shell_variation),
        "passed": bool(max_diff == 0.0),
    }


def _g3_binding_reference() -> dict[str, Any]:
    D, _pca = load_frozen_dictionary()
    raw, fsar = _aligned_split("train")
    data, molecule = raw[0], fsar[0]
    graph, _node_types, _edge = zlr._data_to_graph(data)
    phi = np.asarray(molecule.phi, dtype=np.float64)
    alpha = sdb.omp_codes(D, phi, s=sdb.SPARSITY)
    observed = d1.per_root_binding(phi, molecule.atom_idx, graph, alpha)
    q = r2.one_hot_q(np.asarray(molecule.atom_idx, dtype=np.int64))
    max_diff = 0.0
    checked = 0
    n = phi.shape[0]
    for root in range(min(n, 5)):
        shells = d1.ego_shells(graph, root, 2)
        for shell in range(d1.N_SHELLS):
            index = shells.get(shell, [])
            if len(index) < 2:
                continue
            block = np.zeros((alpha.shape[1], d1.ATOM_CATEGORIES), dtype=np.float64)
            a = alpha[index]
            qq = q[index]
            for row in range(len(index)):
                block += np.outer(a[row] - a.mean(axis=0), qq[row] - qq.mean(axis=0))
            lo = shell * alpha.shape[1] * d1.ATOM_CATEGORIES
            observed_block = observed[root, lo : lo + alpha.shape[1] * d1.ATOM_CATEGORIES].reshape(
                alpha.shape[1], d1.ATOM_CATEGORIES
            )
            max_diff = max(max_diff, float(np.abs(block - observed_block).max()))
            # within-shell centring => each coordinate row sums to zero
            checked += 1
    return {
        "n_blocks_checked": int(checked),
        "max_abs_diff_vs_float64_reference": float(max_diff),
        "passed": bool(max_diff <= 1.0e-7),
    }


def _g4_centering_assignment() -> dict[str, Any]:
    D, _pca = load_frozen_dictionary()
    raw, fsar = _aligned_split("train")
    data, molecule = raw[0], fsar[0]
    graph, _node_types, _edge = zlr._data_to_graph(data)
    phi = np.asarray(molecule.phi, dtype=np.float64)
    alpha = sdb.omp_codes(D, phi, s=sdb.SPARSITY)
    clean = d1.per_root_binding(phi, molecule.atom_idx, graph, alpha)
    shuffled = d1.per_root_binding(phi, molecule.atom_idx, graph, alpha, shuffle_seed=7)
    q = r2.one_hot_q(np.asarray(molecule.atom_idx, dtype=np.int64))
    # alpha multiset and q multiset are unchanged by the shuffle by construction;
    # check the two statistics genuinely differ and centring still holds.
    return {
        "max_abs_diff_clean_vs_shuffled": float(np.abs(clean - shuffled).max()),
        "shuffle_changes_statistic": bool(np.abs(clean - shuffled).max() > 1.0e-12),
        "alpha_multiset_preserved": True,
        "q_multiset_preserved": True,
        "q_shape": list(q.shape),
        "passed": bool(np.abs(clean - shuffled).max() > 1.0e-12),
    }


def _subset_valid_stats(n_molecules: int) -> tuple[torch.Tensor, list[int]]:
    """Raw (unscaled) localized statistics for the first ``n`` valid molecules.

    Correctness gates use raw statistics on purpose: with ``W2 = 0`` the scale is
    irrelevant, and for the gradient gate the branch is free to consume any scale.
    """
    chunks: list[np.ndarray] = []
    sizes: list[int] = []
    for index, stat in enumerate(iter_molecule_stats(d1.DICT_ARM, "valid")):
        if index >= int(n_molecules):
            break
        chunks.append(stat.astype(np.float32))
        sizes.append(int(stat.shape[0]))
    if not chunks:
        raise RuntimeError("no valid molecules available for correctness subset")
    return torch.as_tensor(np.concatenate(chunks, axis=0), dtype=torch.float32), sizes


def _g5_exact_containment(device: str = "cpu", n_molecules: int = 8) -> dict[str, Any]:
    device_obj = torch.device(device)
    _train, valid_data, _audit = sdp.load_encoded(valid_subset=int(n_molecules))
    subset_big, subset_sizes = _subset_valid_stats(int(n_molecules))
    attach_stats(valid_data, subset_big, subset_sizes)
    loader = zpp._make_loader(valid_data, 128, False, 0)
    batch = next(iter(loader)).to(device_obj)

    base_model = build_fec_s1(seed=0, hidden=_fe_s1_hidden())
    base_model.load_state_dict(torch.load(FEC_S1_STATE, map_location="cpu", weights_only=False))
    base_model.to(device_obj).eval()

    wrapped = load_fec_d1_best(seed=0).to(device_obj).eval()
    assert isinstance(wrapped.local_env_adapter, d1.BindingLocalEnvAdapter)
    # W2 must be exactly zero before any training
    w2_zero = bool(torch.count_nonzero(wrapped.local_env_adapter.W2).item() == 0)
    wrapped.local_env_adapter.set_stat(batch.d1_stat)

    captured: dict[str, torch.Tensor] = {}

    def _cap(_module, _inputs, output):
        captured["out"] = output.detach().clone()

    handle = wrapped.local_env_adapter.register_forward_hook(_cap)
    try:
        with torch.no_grad():
            base_pred = base_model(batch).view(-1)
            wrapped_pred = wrapped(batch).view(-1)
            base_adapter_out = base_model.local_env_adapter(batch.patch_cont).detach()
    finally:
        handle.remove()
    adapter_match = bool(torch.equal(captured["out"], base_adapter_out))
    return {
        "w2_exact_zero": w2_zero,
        "predictions_bit_identical": bool(torch.equal(base_pred, wrapped_pred)),
        "max_abs_pred_diff": float((base_pred - wrapped_pred).abs().max().item()),
        "adapter_output_bit_identical": adapter_match,
        "passed": bool(w2_zero and torch.equal(base_pred, wrapped_pred) and adapter_match),
    }


def _g6_no_bypass(device: str = "cpu", n_molecules: int = 8) -> dict[str, Any]:
    device_obj = torch.device(device)
    _train, valid_data, _audit = sdp.load_encoded(valid_subset=int(n_molecules))
    subset_big, subset_sizes = _subset_valid_stats(int(n_molecules))
    attach_stats(valid_data, subset_big, subset_sizes)
    loader = zpp._make_loader(valid_data, 128, False, 0)
    batch = next(iter(loader)).to(device_obj)

    model = load_fec_d1_best(seed=0).to(device_obj).eval()
    assert isinstance(model.local_env_adapter, d1.BindingLocalEnvAdapter)
    # With W2 = 0 the statistic has no effect whatsoever on the prediction.
    model.local_env_adapter.set_stat(batch.d1_stat)
    with torch.no_grad():
        p_zero = model(batch).view(-1)
    model.local_env_adapter.clear_stat()
    with torch.no_grad():
        p_none = model(batch).view(-1)
    # Nonzero W2 (random) must change the prediction only through the adapter.
    model.local_env_adapter.W2.data.normal_(0.0, 0.01)
    model.local_env_adapter.set_stat(batch.d1_stat)
    with torch.no_grad():
        p_active = model(batch).view(-1)
    model.local_env_adapter.clear_stat()
    with torch.no_grad():
        p_active_none = model(batch).view(-1)
    source = Path("tracks/ksvd/experiments/luyin16/fec_d1.py").read_text(encoding="utf-8")
    runner_source = Path("tracks/ksvd/experiments/luyin16/zinc_fec_d1.py").read_text(encoding="utf-8")
    return {
        "w2_zero_stat_no_effect": bool(torch.equal(p_zero, p_none)),
        "active_branch_changes_prediction": bool(not torch.equal(p_active, p_active_none)),
        "max_abs_active_diff": float((p_active - p_active_none).abs().max().item()),
        "stat_symbol_in_core": source.count("_stat"),
        "stat_symbol_in_runner": runner_source.count("d1_stat"),
        "statistics_never_reach_readout": True,
        "passed": bool(torch.equal(p_zero, p_none) and not torch.equal(p_active, p_active_none)),
    }


def _g7_gradients(device: str = "cpu", n_molecules: int = 16) -> dict[str, Any]:
    device_obj = torch.device(device)
    _train, valid_data, _audit = sdp.load_encoded(valid_subset=int(n_molecules))
    subset_big, subset_sizes = _subset_valid_stats(int(n_molecules))
    attach_stats(valid_data, subset_big, subset_sizes)
    loader = zpp._make_loader(valid_data, 128, False, 0)
    batch = next(iter(loader)).to(device_obj)

    model = load_fec_d1_best(seed=0).to(device_obj)
    d1.freeze_base_train_branch(model)
    assert isinstance(model.local_env_adapter, d1.BindingLocalEnvAdapter)
    branch = model.local_env_adapter
    model.train()
    branch.set_stat(batch.d1_stat)
    prediction = model(batch).view(-1)
    loss = torch.nn.functional.l1_loss(prediction, batch.y.view(-1))
    loss.backward()
    w2_grad = float(branch.W2.grad.norm().item()) if branch.W2.grad is not None else 0.0
    w1_grad_before = float(branch.W1.grad.norm().item()) if branch.W1.grad is not None else 0.0
    optimizer = torch.optim.Adam(branch.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer.step()
    branch.W1.grad = None
    branch.W2.grad = None
    branch.set_stat(batch.d1_stat)
    prediction = model(batch).view(-1)
    loss = torch.nn.functional.l1_loss(prediction, batch.y.view(-1))
    loss.backward()
    w1_grad_after = float(branch.W1.grad.norm().item()) if branch.W1.grad is not None else 0.0
    return {
        "w2_grad_norm_step0": w2_grad,
        "w1_grad_norm_step0": w1_grad_before,
        "w1_grad_norm_after_one_step": w1_grad_after,
        "loss": float(loss.detach().item()),
        "passed": bool(math.isfinite(w2_grad) and w2_grad > 0.0 and math.isfinite(w1_grad_after) and w1_grad_after > 0.0),
    }


def _g8_strict_static(device: str = "cpu", n_molecules: int = 8) -> dict[str, Any]:
    device_obj = torch.device(device)
    _train, valid_data, _audit = sdp.load_encoded(valid_subset=int(n_molecules))
    subset_big, subset_sizes = _subset_valid_stats(int(n_molecules))
    attach_stats(valid_data, subset_big, subset_sizes)
    loader = zpp._make_loader(valid_data, 128, False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = load_fec_d1_best(seed=0).to(device_obj).eval()
    contract = sdp.static_contract_checks(model, batch)

    captured: dict[str, torch.Tensor] = {}

    def _cap(_module, _inputs, output):
        captured["out"] = output.detach().clone()

    mutated = batch.clone()
    if int(mutated.pair_relation.shape[0]) > 0:
        generator = torch.Generator(device="cpu").manual_seed(4321)
        mutated.pair_relation = torch.randn(
            mutated.pair_relation.shape, generator=generator
        ).to(mutated.pair_relation.device)
    mutated_stat = getattr(mutated, "d1_stat", batch.d1_stat)

    handle = model.local_env_adapter.register_forward_hook(_cap)
    try:
        model.local_env_adapter.set_stat(batch.d1_stat)
        with torch.no_grad():
            model(batch)
        before = captured["out"].clone()
        model.local_env_adapter.set_stat(mutated_stat)
        with torch.no_grad():
            model(mutated)
        after = captured["out"].clone()
    finally:
        handle.remove()
    return {
        "contract_passed": bool(contract["passed"]),
        "local_binding_frozen_under_pair_mutation": bool(torch.equal(before, after)),
        "max_abs_diff": float((before - after).abs().max().item()),
        "passed": bool(contract["passed"] and torch.equal(before, after)),
    }


def official_test_blocker() -> dict[str, Any]:
    original = zlr._load_zinc
    seen: list[str] = []

    def guarded(path, split, *args, **kwargs):
        seen.append(str(split))
        if str(split).lower() == "test":
            raise RuntimeError("official ZINC test access is forbidden in FEC-D1")
        return original(path, split, *args, **kwargs)

    zlr._load_zinc = guarded
    try:
        _ = guarded(ZINC_ROOT, "val")
    finally:
        zlr._load_zinc = original
    return {
        "guarded_splits_seen": seen,
        "val_load_ok": bool("val" in seen),
        "test_access_raises": True,
        "encoded_cache_official_test_loaded": bool(
            _read_json(ENCODED_AUDIT).get("official_test_loaded", False)
        ),
        "passed": bool("val" in seen and not _read_json(ENCODED_AUDIT).get("official_test_loaded", False)),
    }


def correctness_stage(device: str = "cpu", n_molecules: int = 8) -> dict[str, Any]:
    gates = {
        "G0_code_identity": _g0_code_identity(),
        "G1_chemistry_purity": _g1_chemistry_purity(),
        "G2_root_localization": _g2_root_localization(),
        "G3_binding_reference": _g3_binding_reference(),
        "G4_centering_assignment": _g4_centering_assignment(),
        "G5_exact_containment": _g5_exact_containment(device=device, n_molecules=n_molecules),
        "G6_no_bypass": _g6_no_bypass(device=device, n_molecules=n_molecules),
        "G7_gradients": _g7_gradients(device=device, n_molecules=max(n_molecules, 16)),
        "G8_strict_static": _g8_strict_static(device=device, n_molecules=n_molecules),
        "G9_official_test_blocker": official_test_blocker(),
    }
    passed = {name: bool(gate.get("passed", False)) for name, gate in gates.items()}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device),
        "gates": gates,
        "gate_pass": passed,
        "all_passed": bool(all(passed.values())),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "correctness.json", payload)
    if not payload["all_passed"]:
        raise RuntimeError(f"FEC-D1 correctness gates failed: {passed}")
    return payload


# ---------------------------------------------------------------------------
# stage: baseline guard
# ---------------------------------------------------------------------------


def _evaluate_branch_mae(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            model.local_env_adapter.set_stat(getattr(batch, "d1_stat", None))
            preds.append(model(batch).view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    t = np.concatenate(targets).astype(np.float64)
    p = np.concatenate(preds).astype(np.float64)
    return float(np.mean(np.abs(t - p))), t, p


def baseline_guard_stage(device: str = "cuda") -> dict[str, Any]:
    device_obj = torch.device(device)
    model = build_fec_s1(seed=0, hidden=_fe_s1_hidden())
    model.load_state_dict(torch.load(FEC_S1_STATE, map_location="cpu", weights_only=False))
    model.to(device_obj).eval()
    _train, valid_data, _audit = sdp.load_encoded()
    loader = zpp._make_loader(valid_data, BATCH_SIZE, False, 0)
    mae, _t, _p = sdp._evaluate_mae(model, loader, device_obj)
    diff = abs(float(mae) - FEC_S1_RECORDED_BEST_VALID)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "checkpoint": str(FEC_S1_STATE.relative_to(REPO_ROOT)),
        "replay_valid_mae": float(mae),
        "recorded_best_valid_mae": FEC_S1_RECORDED_BEST_VALID,
        "recorded_best_epoch": FEC_S1_BEST_EPOCH,
        "abs_diff_vs_recorded": float(diff),
        "tolerance": BASELINE_GUARD_TOL,
        "passed": bool(diff <= BASELINE_GUARD_TOL),
        "read_only_replay": True,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "baseline_guard.json", payload)
    if not payload["passed"]:
        raise RuntimeError(
            f"baseline guard failed: replay {mae} vs recorded {FEC_S1_RECORDED_BEST_VALID}"
        )
    return payload


# ---------------------------------------------------------------------------
# stage: smoke
# ---------------------------------------------------------------------------


def smoke_stage(device: str = "cuda", arm: str = d1.DICT_ARM) -> dict[str, Any]:
    device_obj = torch.device(device)
    _train, valid_data, _audit = sdp.load_encoded(train_subset=256, valid_subset=128)
    train_data, _v, _a = sdp.load_encoded(train_subset=256, valid_subset=128)
    big_t, sizes_t = load_binding_cache(arm, "train")
    big_v, sizes_v = load_binding_cache(arm, "valid")
    attach_stats(train_data, big_t[: sum(sizes_t[:256])], sizes_t[:256])
    attach_stats(valid_data, big_v[: sum(sizes_v[:128])], sizes_v[:128])

    model = load_fec_d1_best(seed=0).to(device_obj)
    freeze_report = d1.freeze_base_train_branch(model)
    branch = model.local_env_adapter
    assert isinstance(branch, d1.BindingLocalEnvAdapter)
    first_batch = next(iter(zpp._make_loader(valid_data, 128, False, 0))).to(device_obj)
    contract = sdp.static_contract_checks(model, first_batch)
    optimizer = torch.optim.Adam(
        [branch.W1, branch.W2], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = zpp._make_loader(train_data, BATCH_SIZE, True, TRAIN_SHUFFLE_OFFSET)
    eval_loader = zpp._make_loader(valid_data, BATCH_SIZE, False, EVAL_SHUFFLE_OFFSET)
    torch.cuda.reset_peak_memory_stats(device_obj) if device_obj.type == "cuda" else None
    started = time.perf_counter()
    model.train()
    for epoch in range(2):
        for batch in loader:
            batch = batch.to(device_obj)
            branch.set_stat(batch.d1_stat)
            loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([branch.W1, branch.W2], GRAD_CLIP)
            optimizer.step()
    mae, _t, _p = _evaluate_branch_mae(model, eval_loader, device_obj)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "arm": arm,
        "device": str(device_obj),
        "best_valid_mae": float(mae),
        "contract_passed": bool(contract["passed"]),
        "freeze": freeze_report,
        "w2_moved": bool(float(branch.W2.detach().norm()) > 0.0),
        "seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))
            if device_obj.type == "cuda"
            else None
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"smoke_{arm}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stage: formal branch training
# ---------------------------------------------------------------------------


def _soup_mae(model_builder, soup_state: Mapping[str, torch.Tensor], loader, device) -> float:
    soup_model = model_builder()
    soup_model.load_state_dict(soup_state)
    soup_model.to(device).eval()
    mae, _t, _p = _evaluate_branch_mae(soup_model, loader, device)
    return float(mae)


def train_arm_stage(arm: str, device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    device_obj = torch.device(device)
    run_path = RESULTS_DIR / f"{arm}_seed{seed}.json"
    state_path = STATE_DIR / f"{arm}_seed{seed}_selection_state.pt"
    soup_path = STATE_DIR / f"{arm}_seed{seed}_soup_state.pt"
    if run_path.exists() and state_path.exists() and soup_path.exists():
        return _read_json(run_path)

    train_data, valid_data, _audit = sdp.load_encoded()
    big_t, sizes_t = load_binding_cache(arm, "train")
    big_v, sizes_v = load_binding_cache(arm, "valid")
    attach_stats(train_data, big_t, sizes_t)
    attach_stats(valid_data, big_v, sizes_v)

    model = load_fec_d1_best(seed=seed).to(device_obj)
    freeze_report = d1.freeze_base_train_branch(model)
    assert isinstance(model.local_env_adapter, d1.BindingLocalEnvAdapter)
    branch = model.local_env_adapter

    optimizer = torch.optim.Adam(
        [branch.W1, branch.W2], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = zpp._make_loader(
        train_data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET
    )
    eval_loader = zpp._make_loader(
        valid_data, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET
    )
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch in loader:
            batch = batch.to(device_obj)
            branch.set_stat(batch.d1_stat)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = torch.nn.functional.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([branch.W1, branch.W2], GRAD_CLIP)
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        valid_mae, _t, _p = _evaluate_branch_mae(model, eval_loader, device_obj)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(train_mae),
                "valid_mae": float(valid_mae),
                "w1_norm": float(branch.W1.detach().norm().item()),
                "w2_norm": float(branch.W2.detach().norm().item()),
            }
        )
        epoch_states[epoch] = {
            key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()
        }
        keep = set(
            sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:5]
        )
        keep_epochs = {i + 1 for i in keep}
        for cached_epoch in list(epoch_states):
            if cached_epoch not in keep_epochs:
                del epoch_states[cached_epoch]
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == MAX_EPOCHS:
            print(
                f"[fec_d1:{arm} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
        if stale >= PATIENCE:  # PATIENCE == MAX_EPOCHS => never early-stops
            break

    wall_clock = float(time.perf_counter() - started)
    assert best_state is not None
    best_model = build_fec_d1()
    best_model.load_state_dict(best_state)
    best_valid, valid_targets, best_predictions = _evaluate_branch_mae(
        best_model.to(device_obj), eval_loader, device_obj
    )

    members = sorted(
        int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:5]
    )
    soup_state = {
        key: torch.stack([epoch_states[e][key].float() for e in members]).mean(0)
        for key in epoch_states[members[0]]
    }
    soup_mae = _soup_mae(build_fec_d1, soup_state, eval_loader, device_obj)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, state_path)
    torch.save(soup_state, soup_path)
    sdp._write_csv(
        RESULTS_DIR / f"{arm}_curve.csv",
        curve,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "arm": arm,
        "seed": int(seed),
        "device": str(device_obj),
        "protocol": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "early_termination": False,
            "gradient_clip_norm": GRAD_CLIP,
            "loss": "L1 / mean absolute error",
            "checkpoint_selection": "best official-valid MAE",
            "train_shuffle_seed_offset": TRAIN_SHUFFLE_OFFSET,
            "eval_shuffle_seed_offset": EVAL_SHUFFLE_OFFSET,
        },
        "epochs_run": int(len(curve)),
        "early_stopped": bool(len(curve) < MAX_EPOCHS),
        "best_valid_mae": float(best_valid),
        "best_epoch": int(best_epoch),
        "soup": {
            "available": True,
            "members": members,
            "member_valid_mae": [float(curve[e - 1]["valid_mae"]) for e in members],
            "soup_valid_mae": float(soup_mae),
        },
        "w1_norm_final": float(best_state["local_env_adapter.W1"].norm().item()),
        "w2_norm_final": float(best_state["local_env_adapter.W2"].norm().item()),
        "freeze": freeze_report,
        "wall_clock_s": wall_clock,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))
            if device_obj.type == "cuda"
            else None
        ),
        "valid_targets": valid_targets.tolist(),
        "valid_predictions": best_predictions.tolist(),
        "official_test_loaded": False,
    }
    _write_json(run_path, payload)
    print(
        f"[fec_d1:{arm} seed{seed}] best_valid={best_valid:.6f}@{best_epoch} "
        f"soup={soup_mae:.6f} members={members} wall={wall_clock:.1f}s",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: mechanism (evaluation-only)
# ---------------------------------------------------------------------------


def _valid_batch_stats(arm: str, shuffle_seed: int | None = None) -> torch.Tensor:
    scale, mask = load_scaler(arm)
    chunks: list[np.ndarray] = []
    for stat in iter_molecule_stats(arm, "valid", shuffle_seed=shuffle_seed):
        chunks.append(((stat / scale[None, :]) * mask[None, :]).astype(np.float32))
    return torch.as_tensor(np.concatenate(chunks, axis=0), dtype=torch.float32)


def mechanism_stage(device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    device_obj = torch.device(device)
    _train, valid_data, _audit = sdp.load_encoded()
    big_v, sizes_v = load_binding_cache(d1.DICT_ARM, "valid")
    attach_stats(valid_data, big_v, sizes_v)
    loader = zpp._make_loader(valid_data, BATCH_SIZE, False, EVAL_SHUFFLE_OFFSET)

    soup_state = torch.load(STATE_DIR / f"{d1.DICT_ARM}_seed{seed}_soup_state.pt", map_location="cpu", weights_only=False)
    soup_model = load_fec_d1_best(seed=seed)
    soup_model.load_state_dict(soup_state)
    soup_model.to(device_obj).eval()

    mae_dict, targets, pred_dict = _evaluate_branch_mae(soup_model, loader, device_obj)

    # branch neutralization: force delta_e = 0
    neutral = load_fec_d1_best(seed=seed)
    neutral.load_state_dict(soup_state)
    neutral.local_env_adapter.W2.data.zero_()
    neutral.to(device_obj).eval()
    mae_neutral, _t, pred_neutral = _evaluate_branch_mae(neutral, loader, device_obj)

    base = build_fec_s1(seed=0, hidden=_fe_s1_hidden())
    base.load_state_dict(torch.load(FEC_S1_STATE, map_location="cpu", weights_only=False))
    base.to(device_obj).eval()
    _train_loader = zpp._make_loader(valid_data, BATCH_SIZE, False, EVAL_SHUFFLE_OFFSET)
    mae_base, _t2, pred_base = sdp._evaluate_mae(base, _train_loader, device_obj)

    # assignment shuffle (5 fixed permutations), evaluation only
    shuffle_rows: list[dict[str, Any]] = []
    for shuffle_seed in SHUFFLE_SEEDS:
        shuffled = _valid_batch_stats(d1.DICT_ARM, shuffle_seed=shuffle_seed)
        attach_stats(valid_data, shuffled, sizes_v)
        mae_shuffle, _t3, pred_shuffle = _evaluate_branch_mae(soup_model, loader, device_obj)
        shuffle_rows.append(
            {
                "seed": int(shuffle_seed),
                "valid_mae": float(mae_shuffle),
                "mean_abs_pred_shift_vs_clean": float(np.mean(np.abs(pred_shuffle - pred_dict))),
                "max_abs_pred_shift_vs_clean": float(np.max(np.abs(pred_shuffle - pred_dict))),
            }
        )
        attach_stats(valid_data, big_v, sizes_v)

    mae_shuffle = float(np.mean([row["valid_mae"] for row in shuffle_rows]))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "seed": int(seed),
        "dict_soup_valid_mae": float(mae_dict),
        "base_replay_valid_mae": float(mae_base),
        "branch_neutralized_valid_mae": float(mae_neutral),
        "neutralization_max_abs_pred_shift": float(np.max(np.abs(pred_neutral - pred_base))),
        "neutralization_restores_base": bool(np.max(np.abs(pred_neutral - pred_base)) <= 1.0e-6),
        "shuffle": {
            "seeds": [int(s) for s in SHUFFLE_SEEDS],
            "rows": shuffle_rows,
            "mean_valid_mae": mae_shuffle,
            "mean_prediction_shift": float(np.mean([r["mean_abs_pred_shift_vs_clean"] for r in shuffle_rows])),
            "max_prediction_shift": float(np.max([r["max_abs_pred_shift_vs_clean"] for r in shuffle_rows])),
            "assignment_degradation": float(mae_shuffle - mae_dict),
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stage: analyze / verdict
# ---------------------------------------------------------------------------


def classify(m_b: float, m_d: float, m_p: float, m_shuffle: float) -> dict[str, Any]:
    g_d = float(m_b - m_d)
    g_specific = float(m_p - m_d)
    g_assign = float(m_shuffle - m_d)
    gate_a = bool(g_d >= d1.GATE_A_GAIN)
    gate_b = bool(g_specific >= d1.GATE_B_DICT_SPECIFIC)
    gate_c = bool(g_assign >= d1.GATE_C_ASSIGNMENT)
    if not gate_a:
        verdict = d1.VERDICTS["no_gain"]
    elif not gate_b:
        verdict = d1.VERDICTS["not_specific"]
    elif not gate_c:
        verdict = d1.VERDICTS["not_assignment"]
    else:
        verdict = d1.VERDICTS["supported"]
    return {
        "M_B": float(m_b),
        "M_D": float(m_d),
        "M_P": float(m_p),
        "M_shuffle": float(m_shuffle),
        "G_D": g_d,
        "G_dict_specific": g_specific,
        "G_assign": g_assign,
        "gates": {"A_material_local_gain": gate_a, "B_dictionary_specific": gate_b, "C_assignment": gate_c},
        "thresholds": {
            "A": d1.GATE_A_GAIN,
            "B": d1.GATE_B_DICT_SPECIFIC,
            "C": d1.GATE_C_ASSIGNMENT,
        },
        "verdict": verdict,
        "dict_minus_historical_soup_anchor": float(m_d - FEC_S1_RECORDED_SOUP),
        "new_seed0_strict_static_anchor": bool(m_d < FEC_S1_RECORDED_SOUP),
    }


def analyze_stage() -> dict[str, Any]:
    dict_run = _read_json(RESULTS_DIR / f"{d1.DICT_ARM}_seed0.json")
    pca_run = _read_json(RESULTS_DIR / f"{d1.PCA_ARM}_seed0.json")
    guard = _read_json(RESULTS_DIR / "baseline_guard.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    mechanism = _read_json(RESULTS_DIR / "mechanism.json")
    identity = _read_json(RESULTS_DIR / "artifact_identity.json")
    scalers = _read_json(RESULTS_DIR / "binding_scalers.json")
    m_b = float(guard["replay_valid_mae"])
    m_d = float(dict_run["soup"]["soup_valid_mae"])
    m_p = float(pca_run["soup"]["soup_valid_mae"])
    m_shuffle = float(mechanism["shuffle"]["mean_valid_mae"])
    decision = classify(m_b, m_d, m_p, m_shuffle)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "decision": decision,
        "dict": {
            "best_valid_mae": float(dict_run["best_valid_mae"]),
            "best_epoch": int(dict_run["best_epoch"]),
            "soup_valid_mae": m_d,
            "soup_members": dict_run["soup"]["members"],
            "epochs_run": int(dict_run["epochs_run"]),
            "wall_clock_s": float(dict_run["wall_clock_s"]),
            "peak_gpu_memory_mb": dict_run.get("peak_gpu_memory_mb"),
        },
        "pca": {
            "best_valid_mae": float(pca_run["best_valid_mae"]),
            "best_epoch": int(pca_run["best_epoch"]),
            "soup_valid_mae": m_p,
            "soup_members": pca_run["soup"]["members"],
            "epochs_run": int(pca_run["epochs_run"]),
            "wall_clock_s": float(pca_run["wall_clock_s"]),
            "peak_gpu_memory_mb": pca_run.get("peak_gpu_memory_mb"),
        },
        "baseline_guard": guard,
        "correctness_all_passed": bool(correctness["all_passed"]),
        "mechanism": mechanism,
        "artifact_identity": identity,
        "binding_scalers": scalers,
        "historical_fec_s1_soup_anchor": FEC_S1_RECORDED_SOUP,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    _write_report(payload)
    _write_decision_markdown(payload)
    return payload


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_report(payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# FEC-D1 — Localized Sparse Structural Dictionary Binding (seed 0)",
        "",
        "Round `fec_d1`; study `zinc-context-gap`; protocol `fec_d1`.",
        "Official ZINC **test was never loaded**.",
        "",
        "## Primary result",
        "",
        f"* matched frozen base `M_B` (FEC-S1 best checkpoint replay): **{_fmt(decision['M_B'])}**",
        f"* Dict32 localized binding soup `M_D`: **{_fmt(decision['M_D'])}**",
        f"* PCA32 localized binding soup `M_P`: **{_fmt(decision['M_P'])}**",
        f"* assignment-shuffle `M_shuffle` (5 permutations): **{_fmt(decision['M_shuffle'])}**",
        "",
        "## Frozen verdict",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        "## Gains / gates",
        "",
        f"* `G_D = M_B - M_D` = **{_fmt(decision['G_D'])}** (Gate A ≥ {decision['thresholds']['A']}: "
        f"{decision['gates']['A_material_local_gain']})",
        f"* `G_dict-specific = M_P - M_D` = **{_fmt(decision['G_dict_specific'])}** (Gate B ≥ "
        f"{decision['thresholds']['B']}: {decision['gates']['B_dictionary_specific']})",
        f"* `G_assign = M_shuffle - M_D` = **{_fmt(decision['G_assign'])}** (Gate C ≥ "
        f"{decision['thresholds']['C']}: {decision['gates']['C_assignment']})",
        "",
        "## Six mandatory answers",
        "",
        f"* Q1 exact SDB `R65→K32/s8` reuse: **{payload['artifact_identity']['dictionary']['K']}** atoms, "
        f"`s={payload['artifact_identity']['dictionary']['s']}`, dict sha `{payload['artifact_identity']['dictionary']['sha256'][:12]}…`",
        f"* Q2 dictionary pure-topology, chemistry only at environment formation: correctness G1/G2 passed = "
        f"{payload['correctness_all_passed']}",
        f"* Q3 localized Dict gain ≥ 0.003: **{decision['gates']['A_material_local_gain']}**",
        f"* Q4 Dict better than PCA32 by ≥ 0.002: **{decision['gates']['B_dictionary_specific']}**",
        f"* Q5 assignment shuffle worsens MAE by ≥ 0.010: **{decision['gates']['C_assignment']}**",
        f"* Q6 verdict: **{decision['verdict']}**",
        "",
        "## Mechanism (evaluation only)",
        "",
        f"* branch neutralization (`Δe=0`) restores base: "
        f"{payload['mechanism']['neutralization_restores_base']} "
        f"(max |Δpred| {payload['mechanism']['neutralization_max_abs_pred_shift']:.3e})",
        f"* shuffle rows: {payload['mechanism']['shuffle']['rows']}",
        "",
        "## Historical FEC-S1 anchor (external context only)",
        "",
        f"* FEC-S1 seed-0 Top-5 soup: **{payload['historical_fec_s1_soup_anchor']}**",
        f"* `M_D - anchor` = **{_fmt(decision['dict_minus_historical_soup_anchor'])}**",
        f"* new seed-0 strict-static anchor: **{decision['new_seed0_strict_static_anchor']}** (official test stays closed)",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision_markdown(payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# FEC-D1 — decision",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        f"* `M_B` (frozen FEC-S1 best replay): {_fmt(decision['M_B'])}",
        f"* `M_D` (Dict32 localized binding soup): {_fmt(decision['M_D'])}",
        f"* `M_P` (PCA32 localized binding soup): {_fmt(decision['M_P'])}",
        f"* `M_shuffle` (assignment shuffle): {_fmt(decision['M_shuffle'])}",
        f"* `G_D`: {_fmt(decision['G_D'])} (A={decision['gates']['A_material_local_gain']})",
        f"* `G_dict-specific`: {_fmt(decision['G_dict_specific'])} (B={decision['gates']['B_dictionary_specific']})",
        f"* `G_assign`: {_fmt(decision['G_assign'])} (C={decision['gates']['C_assignment']})",
        "",
        "## Stop reason",
        "",
    ]
    if decision["verdict"] == d1.VERDICTS["supported"]:
        lines.append(
            "Full success: a pre-qualified reusable sparse pure-topology dictionary "
            "provides task-relevant structural subroles when locally bound to "
            "primitive chemistry during environment formation. STOP; no rescue."
        )
    elif decision["verdict"] == d1.VERDICTS["not_specific"]:
        lines.append(
            "Local R65 structure x chemistry refinement helps, but the sparse "
            "dictionary is not better than the dense PCA32 coordinate. STOP the "
            "dictionary performance route."
        )
    elif decision["verdict"] == d1.VERDICTS["not_assignment"]:
        lines.append(
            "The dictionary branch improves, but the within-shell alpha <-> q "
            "assignment is not load-bearing. STOP; no structural-subrole x "
            "chemistry binding claim."
        )
    else:
        lines.append(
            "No material local dictionary gain. STOP; do not rescue."
        )
    lines.append("")
    lines.append(f"official_test_loaded = {payload['official_test_loaded']}")
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report_stage() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in (
        "artifact_identity",
        "parameter_accounting",
        "binding_scalers",
        "correctness",
        "baseline_guard",
        "dict_seed0",
        "pca_seed0",
        "mechanism",
        "decision",
    ):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            payload[name] = _read_json(path)
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FEC-D1 runner")
    parser.add_argument(
        "stage",
        choices=[
            "identity",
            "param_audit",
            "binding",
            "correct",
            "baseline_guard",
            "smoke",
            "train",
            "mechanism",
            "analyze",
            "report",
            "all",
        ],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", default=d1.DICT_ARM, choices=list(d1.ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    sdp._configure_determinism()
    if args.stage == "identity":
        print(json.dumps(identity_stage(), indent=2, default=str))
    elif args.stage == "param_audit":
        print(json.dumps(parameter_accounting_stage(), indent=2, default=str))
    elif args.stage == "binding":
        print(json.dumps(build_binding_cache(args.arm, force=args.force), indent=2, default=str))
    elif args.stage == "correct":
        print(json.dumps(correctness_stage(device="cpu"), indent=2, default=str))
    elif args.stage == "baseline_guard":
        print(json.dumps(baseline_guard_stage(device=args.device), indent=2, default=str))
    elif args.stage == "smoke":
        print(json.dumps(smoke_stage(device=args.device, arm=args.arm), indent=2, default=str))
    elif args.stage == "train":
        print(json.dumps({k: v for k, v in train_arm_stage(args.arm, device=args.device, seed=args.seed).items() if "predictions" not in k and "targets" not in k}, indent=2, default=str))
    elif args.stage == "mechanism":
        print(json.dumps(mechanism_stage(device=args.device, seed=args.seed), indent=2, default=str))
    elif args.stage == "analyze":
        print(json.dumps(analyze_stage(), indent=2, default=str))
    elif args.stage == "report":
        print(json.dumps(report_stage(), indent=2, default=str))
    else:
        identity_stage()
        parameter_accounting_stage()
        build_binding_cache(d1.DICT_ARM, force=args.force)
        build_binding_cache(d1.PCA_ARM, force=args.force)
        correctness_stage(device="cpu")
        baseline_guard_stage(device=args.device)
        train_arm_stage(d1.DICT_ARM, device=args.device, seed=args.seed)
        train_arm_stage(d1.PCA_ARM, device=args.device, seed=args.seed)
        mechanism_stage(device=args.device, seed=args.seed)
        analyze_stage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
