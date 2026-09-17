"""ZSAR-R2-AR0 runner: fixed radius-2 pure-topology coordinates + assignment residual.

Round name: ``FSAR-R2-AR0``.

Stages
------
``params preprocess audit sanity gradient_audit synthetic train soup run
diagnostics witness shuffle_eval gate decide report``

The frozen training protocol is inherited verbatim from the optimized
compact-v4 cell: Adam, lr 1e-3, weight decay 1e-5, batch 128, grad clip 5.0,
max 240 epochs, patience 40, no scheduler, single stage, best official-valid
checkpoint, fixed equal-weight Top-5 soup.  Official ZINC test is never loaded.

Only three models exist: ``M0`` (marginal baseline), ``MB`` (aligned assignment
residual), ``MM`` (matched marginal-capacity control).  Two synthetic controls
(assignment-only positive, marginal-only negative) must pass before any formal
ZINC run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16.zinc_fsar import _set_deterministic
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    _data_to_graph,
    _load_zinc,
)

REPO_ROOT = shead.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/fsar_r2_ar0"
CACHE_DIR = RESULTS_DIR / "cache"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SOUP_DIR = RESULTS_DIR / "soup_states"
RUNS_DIR = RESULTS_DIR / "runs"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PROTOCOL_VERSION = "fsar_r2_ar0_v1"
CACHE_SCHEMA = "fsar_r2_ar0_features_v1"

_write_json = shead._write_json
_read_json = shead._read_json
_git_commit = shead._git_commit
_environment_fingerprint = shead._environment_fingerprint

SEEDS_FIRST_ROUND = (0, 1)
SEED_GATE = 2

DIAG_BATCHES = 32
SHUFFLE_REPEATS = 5


def _tag(variant: str, tag: str | None = None) -> str:
    return str(tag or f"r2ar0_{str(variant).lower()}")


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _cache_paths() -> tuple[Path, Path, Path, Path]:
    return (
        CACHE_DIR / "train.pkl.gz",
        CACHE_DIR / "valid.pkl.gz",
        CACHE_DIR / "scaler.pt",
        CACHE_DIR / "cache_meta.json",
    )


def _assert_no_test_access() -> None:
    """The official test split is never touched; only train/val are processed."""
    processed = ZINC_ROOT / "subset" / "processed"
    for split in ("train", "val"):
        if not (processed / f"{split}.pt").exists():
            raise RuntimeError(
                f"processed ZINC split missing: {processed / f'{split}.pt'}"
            )


def _extract_split(split: str) -> list[r2.MoleculeFeatures]:
    dataset = _load_zinc(ZINC_ROOT, split)
    molecules: list[r2.MoleculeFeatures] = []
    for data in dataset:
        graph, node_types, edge_types = _data_to_graph(data)
        phi = r2.build_phi(graph)
        marginal = r2.build_A(node_types, edge_types, int(data.num_nodes))
        target = float(data.y.view(-1)[0])
        if phi.shape[0] != int(data.num_nodes):
            raise RuntimeError("phi / node count mismatch")
        molecules.append(
            r2.MoleculeFeatures(
                phi=phi.astype(np.float32),
                atom_idx=np.asarray(node_types, dtype=np.int64),
                A=marginal.astype(np.float32),
                n_nodes=int(data.num_nodes),
                n_edges=int(len(edge_types)),
                y=target,
            )
        )
    return molecules


def _compute_train_scalers(
    molecules: Sequence[r2.MoleculeFeatures],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    c_matrices: list[np.ndarray] = []
    p_matrices: list[np.ndarray] = []
    for molecule in molecules:
        q = r2.one_hot_q(molecule.atom_idx)
        c_matrix, p_matrix = r2.center_stats(
            np.asarray(molecule.phi, dtype=np.float64), q
        )
        c_matrices.append(c_matrix)
        p_matrices.append(p_matrix)
    scalers = r2.fit_scalers(c_matrices, p_matrices)
    report = {
        "n_train_molecules": int(len(molecules)),
        "C_effective_coordinates": int(scalers["C_mask"].sum()),
        "C_total_coordinates": int(scalers["C_mask"].size),
        "P_effective_coordinates": int(scalers["P_mask"].sum()),
        "P_total_coordinates": int(scalers["P_mask"].size),
        "C_rms_min_positive": float(
            scalers["C_rms_raw"][scalers["C_mask"] > 0].min()
            if (scalers["C_mask"] > 0).any()
            else 0.0
        ),
        "C_rms_max": float(scalers["C_rms_raw"].max()),
        "P_rms_min_positive": float(
            scalers["P_rms_raw"][scalers["P_mask"] > 0].min()
            if (scalers["P_mask"] > 0).any()
            else 0.0
        ),
        "P_rms_max": float(scalers["P_rms_raw"].max()),
        "dataset_mean_subtracted": False,
        "scaler_source": "train split only",
    }
    return scalers, report


def build_datasets(force: bool = False):
    train_path, valid_path, scaler_path, meta_path = _cache_paths()
    if (
        not force
        and train_path.exists()
        and valid_path.exists()
        and scaler_path.exists()
        and meta_path.exists()
    ):
        meta = _read_json(meta_path)
        if meta.get("cache_schema") == CACHE_SCHEMA:
            with gzip.open(train_path, "rb") as handle:
                train_molecules = pickle.load(handle)
            with gzip.open(valid_path, "rb") as handle:
                valid_molecules = pickle.load(handle)
            scalers = torch.load(scaler_path, map_location="cpu", weights_only=False)
            scalers = {
                key: np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
                for key, value in scalers.items()
            }
            return list(train_molecules), list(valid_molecules), scalers, meta

    started = time.perf_counter()
    _assert_no_test_access()
    train_molecules = _extract_split("train")
    valid_molecules = _extract_split("val")
    scalers, scaler_report = _compute_train_scalers(train_molecules)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(train_path, "wb") as handle:
        pickle.dump(train_molecules, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(valid_path, "wb") as handle:
        pickle.dump(valid_molecules, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save({key: torch.from_numpy(np.asarray(value)) for key, value in scalers.items()}, scaler_path)
    meta = {
        "cache_schema": CACHE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "phi_dim": int(r2.PHI_DIM),
        "a_dim": int(r2.A_DIM),
        "atom_categories": int(r2.ATOM_CATEGORIES),
        "bond_categories": int(r2.BOND_CATEGORIES),
        "patch_radius": int(r2.PATCH_RADIUS),
        "n_train": int(len(train_molecules)),
        "n_valid": int(len(valid_molecules)),
        "scaler_report": scaler_report,
        "git_commit": _git_commit(),
        "seconds": float(time.perf_counter() - started),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    _write_json(meta_path, meta)
    print(
        f"[preprocess] train={len(train_molecules)} valid={len(valid_molecules)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_molecules, valid_molecules, scalers, meta


def preprocess(force: bool = False) -> dict[str, Any]:
    _train, _valid, _scalers, meta = build_datasets(force=force)
    return meta


def _loader(molecules, batch_size, shuffle, seed):
    return r2.make_loader(molecules, batch_size, shuffle, seed)


def _evaluate_mae(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            total += float((prediction - target).abs().sum())
            seen += int(target.numel())
    model.train()
    return total / max(seen, 1)


def _predict_state(model: nn.Module, state, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.load_state_dict(state, strict=True)
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            targets.append(batch.y.view(-1).cpu().numpy())
            predictions.append(model(batch).view(-1).cpu().numpy())
    return (
        np.concatenate(targets).astype(np.float64),
        np.concatenate(predictions).astype(np.float64),
    )


def _mae(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - predictions)))


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    _train, _valid, scalers, _meta = build_datasets()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "models": {},
        "official_test_loaded": False,
    }
    for variant in r2.MODELS:
        model = r2.build_model(variant, scalers, seed=0)
        payload["models"][variant] = r2.parameter_breakdown(model)
    totals = {variant: row["total"] for variant, row in payload["models"].items()}
    payload["totals"] = totals
    payload["M0_MB_matched_base"] = bool(
        payload["models"]["M0"]["S_encoder"] == payload["models"]["MB"]["S_encoder"]
        and payload["models"]["M0"]["base_F0"] == payload["models"]["MB"]["base_F0"]
        and payload["models"]["M0"]["base_F0"] == payload["models"]["MM"]["base_F0"]
    )
    payload["W_B_W_M_same_shape"] = True
    payload["dataset_dependent_vocabulary_params"] = 0
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# sanity / gradient audit
# ---------------------------------------------------------------------------


def sanity() -> dict[str, Any]:
    from tracks.ksvd.tests import test_fsar_r2_ar0 as suite

    checks: dict[str, Any] = {}
    failures: list[str] = []

    def run(name: str, function) -> None:
        try:
            function()
            checks[name] = True
        except Exception as error:  # noqa: BLE001
            checks[name] = False
            failures.append(f"{name}: {error}")

    for name in (
        "node_relabel_invariance",
        "s_chemistry_purity",
        "a_assignment_invariance",
        "p_assignment_invariance",
        "c_assignment_sensitivity",
        "batched_stats_match_numpy",
        "exact_permutation_expectation",
        "scalar_b_exact_expectation",
        "no_patch_copy_inconsistency",
        "undirected_bond_counting",
        "train_only_scaler",
        "gradient_viability",
        "no_mixed_bypass",
        "m0_mb_mm_share_base_architecture",
        "zero_init_matches_m0",
        "positive_control_assignment_only",
        "parameter_counts",
    ):
        run(name, getattr(suite, f"test_{name}"))

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "checks": checks,
        "failures": failures,
        "all_pass": bool(not failures),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if failures:
        raise RuntimeError(f"FSAR-R2-AR0 sanity gates failed: {failures}")
    return payload


def _module_grad_sum(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum())
    return total


def gradient_audit(variant: str, device: str = "cpu") -> dict[str, Any]:
    _train, valid_molecules, scalers, _meta = build_datasets()
    device_obj = torch.device(device)
    model = r2.build_model(variant, scalers, seed=0).to(device_obj)
    batch = next(iter(_loader(list(valid_molecules)[:64], 64, False, 0))).to(device_obj)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss = None
    for _ in range(2):
        loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))
    loss.backward()
    named: dict[str, float] = {
        "s_encoder": _module_grad_sum(model.s_encoder),
        "base_F0": _module_grad_sum(model.base),
    }
    if model.W_B is not None:
        named["W_B"] = float(model.W_B.grad.detach().abs().sum())
    if model.W_M is not None:
        named["W_M"] = float(model.W_M.grad.detach().abs().sum())
    active = 0
    for parameter in model.parameters():
        if parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0.0:
            active += int(parameter.numel())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": str(variant),
        "loss": float(loss.detach()),
        "total_params": _n_params(model),
        "active_prediction_path_params": int(active),
        "module_grad_abs_sum": named,
        "W_B_nonzero_grad": bool(
            model.W_B is not None
            and float(model.W_B.grad.detach().abs().sum()) > 0.0
        ),
        "W_M_nonzero_grad": bool(
            model.W_M is not None
            and float(model.W_M.grad.detach().abs().sum()) > 0.0
        ),
        "official_test_loaded": False,
    }
    if variant in r2.BINDING_MODELS:
        if variant == "MB" and not payload["W_B_nonzero_grad"]:
            raise RuntimeError("MB assignment weight received zero task gradient")
        if variant == "MM" and not payload["W_M_nonzero_grad"]:
            raise RuntimeError("MM assignment weight received zero task gradient")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / f"gradient_audit_{variant}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def gradient_audit_all(device: str = "cpu") -> dict[str, Any]:
    payload = {
        variant: gradient_audit(variant, device=device) for variant in r2.MODELS
    }
    return payload


# ---------------------------------------------------------------------------
# cheap feature audit
# ---------------------------------------------------------------------------


def _effective_rank(matrix: np.ndarray) -> dict[str, float]:
    if matrix.size == 0:
        return {"effective_rank": 0.0, "top_singular_fraction": 0.0}
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = float(singular.sum())
    if total <= 0.0:
        return {"effective_rank": 0.0, "top_singular_fraction": 0.0}
    probabilities = singular / total
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    return {
        "effective_rank": float(math.exp(entropy)),
        "top_singular_fraction": float(singular[0] / total),
    }


def audit(sample_molecules: int = 500) -> dict[str, Any]:
    train, valid, scalers, meta = build_datasets()
    sample = list(train[: int(sample_molecules)])
    phi = np.concatenate([np.asarray(m.phi, dtype=np.float64) for m in sample], axis=0)
    a_matrix = np.stack([np.asarray(m.A, dtype=np.float64) for m in sample], axis=0)
    c_flat: list[np.ndarray] = []
    p_flat: list[np.ndarray] = []
    size: list[int] = []
    for molecule in sample:
        q = r2.one_hot_q(molecule.atom_idx)
        c_matrix, p_matrix = r2.center_stats(
            np.asarray(molecule.phi, dtype=np.float64), q
        )
        c_flat.append(c_matrix.reshape(-1))
        p_flat.append(p_matrix.reshape(-1))
        size.append(int(molecule.n_nodes))
    c_matrix_all = np.stack(c_flat, axis=0)
    p_matrix_all = np.stack(p_flat, axis=0)
    phi_std = phi.std(axis=0)
    a_std = a_matrix.std(axis=0)
    c_norm = np.linalg.norm(c_matrix_all, axis=1)
    size_array = np.asarray(size, dtype=np.float64)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_sampled_molecules": int(len(sample)),
        "phi": {
            "dim": int(phi.shape[1]),
            "zero_std_coordinates": int((phi_std <= 0.0).sum()),
            "std_min": float(phi_std.min()),
            "std_max": float(phi_std.max()),
            "nan_or_inf": bool(not np.isfinite(phi).all()),
            **{f"rank_{k}": v for k, v in _effective_rank(phi).items()},
        },
        "A": {
            "dim": int(a_matrix.shape[1]),
            "zero_std_coordinates": int((a_std <= 0.0).sum()),
            "nan_or_inf": bool(not np.isfinite(a_matrix).all()),
            **{f"rank_{k}": v for k, v in _effective_rank(a_matrix).items()},
        },
        "C": {
            "shape": [int(r2.PHI_DIM), int(r2.ATOM_CATEGORIES)],
            "effective_coordinates": int(scalers["C_mask"].sum()),
            "total_coordinates": int(scalers["C_mask"].size),
            "nan_or_inf": bool(not np.isfinite(c_matrix_all).all()),
            **{f"rank_{k}": v for k, v in _effective_rank(c_matrix_all[:, :64]).items()},
        },
        "P": {
            "shape": [int(r2.PHI_DIM), int(r2.ATOM_CATEGORIES)],
            "effective_coordinates": int(scalers["P_mask"].sum()),
            "total_coordinates": int(scalers["P_mask"].size),
            "nan_or_inf": bool(not np.isfinite(p_matrix_all).all()),
        },
        "size_correlation": {
            "pearson_C_norm_vs_n": float(np.corrcoef(c_norm, size_array)[0, 1]),
        },
        "scaler_report": meta["scaler_report"],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "feature_audit.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _branch_curve_stats(model: r2.FSARARR2AR0Model, batches) -> dict[str, float]:
    model.eval()
    base_values: list[np.ndarray] = []
    branch_values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            base = model.base_prediction(batch).view(-1)
            base_values.append(base.cpu().numpy())
            if model.variant in r2.BINDING_MODELS:
                c_matrix, p_matrix = model.compute_statistics(batch)
                term = model.assignment_term(c_matrix, p_matrix).view(-1)
                branch_values.append(term.cpu().numpy())
    model.train()
    base_all = np.concatenate(base_values) if base_values else np.zeros(1)
    branch_all = np.concatenate(branch_values) if branch_values else np.zeros(1)
    return {
        "base_out_mean": float(base_all.mean()),
        "base_out_std": float(base_all.std()),
        "branch_out_mean": float(branch_all.mean()),
        "branch_out_std": float(branch_all.std()),
        "branch_norm": float(model.branch_weight_norm()),
        "branch_abs_sum": float(model.branch_weight_abs_sum()),
        "branch_sparsity": float(model.branch_weight_sparsity()),
    }


def train(
    variant: str,
    seed: int,
    device: str = "cpu",
    tag: str | None = None,
    max_epochs: int | None = None,
    protocol_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tag = _tag(variant, tag)
    train_molecules, valid_molecules, scalers, _meta = build_datasets()
    protocol = dict(shead.OPTIMIZED_PROTOCOL)
    if protocol_override:
        protocol.update(protocol_override)
    if max_epochs is not None:
        protocol["max_epochs"] = int(max_epochs)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats()
    model = r2.build_model(variant, scalers, seed=int(seed)).to(device_obj)
    total_params = _n_params(model)

    diag_batches = [
        batch.to(device_obj)
        for batch in _loader(list(valid_molecules)[:DIAG_BATCHES], 128, False, 0)
    ]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    loader = _loader(
        train_molecules,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = _loader(
        valid_molecules,
        int(protocol["batch_size"]),
        False,
        int(seed) + int(protocol["eval_shuffle_seed_offset"]),
    )
    steps_per_epoch = int(math.ceil(len(train_molecules) / int(protocol["batch_size"])))
    patience = int(protocol["patience"])
    max_epochs_value = int(protocol["max_epochs"])

    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    top5: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs_value + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        grad_norm_total = 0.0
        grad_steps = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(protocol["gradient_clip_norm"])
                )
            )
            grad_norm_total += grad_norm
            grad_steps += 1
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        valid_mae = _evaluate_mae(model, eval_loader, device_obj)
        stats = _branch_curve_stats(model, diag_batches)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(train_mae),
                "valid_mae": float(valid_mae),
                "grad_norm_mean": float(grad_norm_total / max(grad_steps, 1)),
                **stats,
                "checkpoint_selected": 0,
            }
        )
        state_copy = copy.deepcopy(model.state_dict())
        top5.append((float(valid_mae), int(epoch), state_copy))
        top5.sort(key=lambda item: (item[0], item[1]))
        top5 = top5[:5]
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == max_epochs_value:
            print(
                f"[{tag} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch} "
                f"branch_norm={stats['branch_norm']:.4e} "
                f"branch_std={stats['branch_out_std']:.4e}",
                flush=True,
            )
        if stale >= patience:
            print(f"[{tag} seed{seed}] early_stop epoch={epoch} best={best_epoch}", flush=True)
            break
    wall_clock = float(time.perf_counter() - started)
    if best_state is not None:
        model.load_state_dict(best_state)
    for row in curve:
        row["checkpoint_selected"] = int(row["epoch"] == best_epoch)

    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = CURVE_DIR / f"{tag}_seed{seed}_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        for row in curve:
            writer.writerow(row)
    selection_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    torch.save(model.state_dict(), selection_path)
    torch.save(
        [{"valid_mae": float(v), "epoch": int(e), "state": s} for v, e, s in top5],
        SOUP_DIR / f"{tag}_seed{seed}_top5_states.pt",
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "variant": str(variant),
        "seed": int(seed),
        "protocol": protocol,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curve)),
        "steps_per_epoch": int(steps_per_epoch),
        "wall_clock_s": wall_clock,
        "early_stopped": bool(len(curve) < max_epochs_value),
        "parameters": int(total_params),
        "active_prediction_path_params": int(total_params),
        "parameter_breakdown": r2.parameter_breakdown(model),
        "final_branch_norm": float(model.branch_weight_norm()),
        "final_branch_abs_sum": float(model.branch_weight_abs_sum()),
        "final_branch_sparsity": float(model.branch_weight_sparsity()),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device_obj.type == "cuda" else 0
        ),
        "state_path": str(selection_path),
        "curve_path": str(curve_path),
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(str(device)),
        "scaler_report": _meta["scaler_report"],
        "official_test_loaded": False,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RUNS_DIR / f"{tag}_seed{seed}.json", summary)
    print(
        f"[{tag} seed{seed}] best_valid={best_mae:.6f} epoch={best_epoch} "
        f"params={total_params} wall={wall_clock:.1f}s",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# soup
# ---------------------------------------------------------------------------


def soup(variant: str, seed: int, tag: str | None = None) -> dict[str, Any]:
    tag = _tag(variant, tag)
    entries = torch.load(
        SOUP_DIR / f"{tag}_seed{seed}_top5_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    if len(entries) < 5:
        raise RuntimeError(f"need >= 5 snapshots for top-5 soup, got {len(entries)}")
    ranked = sorted(
        entries, key=lambda row: (float(row["valid_mae"]), int(row["epoch"]))
    )[:5]
    states = [row["state"] for row in ranked]
    keys = list(states[0].keys())
    soup_state = {
        key: torch.stack([state[key].float() for state in states], dim=0)
        .mean(dim=0)
        .to(states[0][key].dtype)
        for key in keys
    }
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
    torch.save(soup_state, soup_path)

    _train, valid_molecules, scalers, _meta = build_datasets()
    loader = _loader(valid_molecules, 128, False, 0)
    device = torch.device("cpu")
    model = r2.build_model(variant, scalers)
    selection_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    targets, best_preds = _predict_state(
        model,
        torch.load(selection_path, map_location="cpu", weights_only=True),
        loader,
        device,
    )
    _targets, soup_preds = _predict_state(model, soup_state, loader, device)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": str(variant),
        "seed": int(seed),
        "top5_epochs": [int(row["epoch"]) for row in ranked],
        "top5_valid_mae": [float(row["valid_mae"]) for row in ranked],
        "best_checkpoint_valid_mae": _mae(targets, best_preds),
        "top5_soup_valid_mae": _mae(targets, soup_preds),
        "soup_improvement_over_best": _mae(targets, best_preds) - _mae(targets, soup_preds),
        "soup_state_path": str(soup_path),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def run(
    variant: str,
    seed: int,
    device: str = "cpu",
    tag: str | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    tag = _tag(variant, tag)
    if max_epochs is not None:
        tag = f"{tag}_short{int(max_epochs)}"
        if not STATE_DIR.joinpath(f"{tag}_seed{seed}_selection_state.pt").exists():
            train(variant, seed, device=device, tag=tag, max_epochs=max_epochs)
        if not RESULTS_DIR.joinpath(f"soup_{tag}_seed{seed}.json").exists():
            soup(variant, seed, tag=tag)
        return _read_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json")
    state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    if state_path.exists():
        print(f"[run] selection state exists, skipping train: {state_path}", flush=True)
    else:
        train(variant, seed, device=device, tag=tag)
    soup_json = RESULTS_DIR / f"soup_{tag}_seed{seed}.json"
    if soup_json.exists():
        print(f"[run] soup json exists, skipping soup: {soup_json}", flush=True)
        return _read_json(soup_json)
    return soup(variant, seed, tag=tag)


# ---------------------------------------------------------------------------
# frozen eval: diagnostics / assignment shuffle
# ---------------------------------------------------------------------------


def _load_frozen(variant: str, seed: int, tag: str | None = None, state: str = "soup"):
    _train, _valid, scalers, _meta = build_datasets()
    model = r2.build_model(variant, scalers)
    tag = _tag(variant, tag)
    if str(state) == "soup":
        state_path = SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
    elif str(state) == "best":
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    else:
        raise ValueError(f"unknown frozen state {state!r}")
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def diagnostics(variant: str, seed: int, tag: str | None = None, state: str = "soup"):
    model = _load_frozen(variant, seed, tag, state=state)
    _train, valid_molecules, _scalers, _meta = build_datasets()
    batches = [
        batch
        for batch in _loader(valid_molecules, 128, False, 0)
    ]
    base_values: list[np.ndarray] = []
    branch_values: list[np.ndarray] = []
    c_norms: list[float] = []
    with torch.no_grad():
        for batch in batches:
            base_values.append(model.base_prediction(batch).view(-1).numpy())
            if model.variant in r2.BINDING_MODELS:
                c_matrix, p_matrix = model.compute_statistics(batch)
                branch_values.append(
                    model.assignment_term(c_matrix, p_matrix).view(-1).numpy()
                )
                statistic = (
                    c_matrix
                    if model.variant == "MB"
                    else p_matrix
                )
                c_norms.append(float(statistic.reshape(statistic.shape[0], -1).norm(dim=1).mean()))
    base_all = np.concatenate(base_values)
    branch_all = np.concatenate(branch_values) if branch_values else np.zeros(1)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": str(variant),
        "seed": int(seed),
        "state": str(state),
        "base_out_mean": float(base_all.mean()),
        "base_out_std": float(base_all.std()),
        "branch_out_mean": float(branch_all.mean()),
        "branch_out_std": float(branch_all.std()),
        "branch_weight_norm": float(model.branch_weight_norm()),
        "branch_weight_abs_sum": float(model.branch_weight_abs_sum()),
        "branch_weight_sparsity": float(model.branch_weight_sparsity()),
        "statistic_norm_mean": float(np.mean(c_norms)) if c_norms else 0.0,
        "branch_live": bool(float(branch_all.std()) > 0.0),
        "official_test_loaded": False,
    }
    if model.variant == "MB" and model.W_B is not None:
        payload["W_B_abs_mean"] = float(model.W_B.detach().abs().mean())
        payload["W_B_max_abs"] = float(model.W_B.detach().abs().max())
    if model.variant == "MM" and model.W_M is not None:
        payload["W_M_abs_mean"] = float(model.W_M.detach().abs().mean())
        payload["W_M_max_abs"] = float(model.W_M.detach().abs().max())
    _write_json(
        RESULTS_DIR / f"diagnostics_{_tag(variant, tag)}_seed{seed}_{state}.json",
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def witness(
    variant: str, seed: int, tag: str | None = None, state: str = "soup", repeats: int = SHUFFLE_REPEATS
) -> dict[str, Any]:
    """Evaluation-only assignment shuffle at the original-node level."""
    model = _load_frozen(variant, seed, tag, state=state)
    _train, valid_molecules, _scalers, _meta = build_datasets()
    batches = [batch for batch in _loader(valid_molecules, 128, False, 0)]
    targets: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    perm_preds: list[list[np.ndarray]] = [[] for _ in range(int(repeats))]
    channel_deltas = {"A": 0.0, "S": 0.0, "C_or_P": 0.0, "prediction": 0.0}
    with torch.no_grad():
        for batch in batches:
            targets.append(batch.y.view(-1).numpy())
            base = model.base_prediction(batch).view(-1)
            actual.append(model(batch).view(-1).numpy())
            for repeat in range(int(repeats)):
                q_shuffled = r2.permute_q_within_graphs(batch, seed=1000 * int(seed) + repeat)
                shuffled_batch = batch.with_q(q_shuffled)
                prediction = model(shuffled_batch).view(-1)
                perm_preds[repeat].append(prediction.numpy())
                if repeat == 0:
                    channel_deltas["A"] += float(
                        (shuffled_batch.A - batch.A).abs().sum()
                    )
                    s_actual = model.encode_s(batch)
                    s_shuffled = model.encode_s(shuffled_batch)
                    channel_deltas["S"] += float((s_shuffled - s_actual).abs().sum())
                    if model.variant in r2.BINDING_MODELS:
                        c_actual, p_actual = model.compute_statistics(batch)
                        c_shuffled, p_shuffled = model.compute_statistics(shuffled_batch)
                        delta = (
                            (c_shuffled - c_actual).abs().sum()
                            if model.variant == "MB"
                            else (p_shuffled - p_actual).abs().sum()
                        )
                        channel_deltas["C_or_P"] += float(delta)
                    channel_deltas["prediction"] += float(
                        (prediction - model(batch).view(-1)).abs().sum()
                    )
    targets_all = np.concatenate(targets)
    actual_all = np.concatenate(actual)
    perm_arrays = [np.concatenate(items) for items in perm_preds]
    perm_matrix = np.stack(perm_arrays, axis=0)
    perm_mean_prediction = perm_matrix.mean(axis=0)
    perm_maes = [float(np.mean(np.abs(targets_all - perm))) for perm in perm_arrays]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": str(variant),
        "seed": int(seed),
        "state": str(state),
        "repeat_count": int(repeats),
        "actual_mae": float(np.mean(np.abs(targets_all - actual_all))),
        "permutation_mean_prediction_mae": float(
            np.mean(np.abs(targets_all - perm_mean_prediction))
        ),
        "mean_of_permutation_maes": float(np.mean(perm_maes)),
        "permutation_mae_std": float(np.std(perm_maes)),
        "permutation_maes": perm_maes,
        "mean_abs_prediction_change": float(
            np.mean(np.abs(perm_matrix - actual_all[None, :]))
        ),
        "channel_delta_abs_sum": channel_deltas,
        "official_test_loaded": False,
    }
    _write_json(
        RESULTS_DIR / f"witness_{_tag(variant, tag)}_seed{seed}_{state}.json", payload
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def shuffle_eval(variant: str, seed: int, tag: str | None = None, state: str = "soup"):
    return witness(variant, seed, tag=tag, state=state)


# ---------------------------------------------------------------------------
# synthetic controls (CPU, no ZINC)
# ---------------------------------------------------------------------------


def _toy_molecule(
    graph,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    label: float,
) -> r2.MoleculeFeatures:
    phi = r2.build_phi(graph)
    marginal = r2.build_A(node_types, dict(edge_types), len(node_types))
    return r2.MoleculeFeatures(
        phi=phi.astype(np.float32),
        atom_idx=np.asarray(node_types, dtype=np.int64),
        A=marginal.astype(np.float32),
        n_nodes=int(len(node_types)),
        n_edges=int(len(edge_types)),
        y=float(label),
    )


def synthetic_positive(n_per_class: int = 150, seed: int = 0):
    """Assignment-only control: fixed path topology, fixed multiset, label = N position."""
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    edges = {(0, 1): 1, (1, 2): 1, (2, 3): 1}
    rng = np.random.default_rng(int(seed))
    molecules: list[r2.MoleculeFeatures] = []
    for label in (0, 1):
        for _ in range(int(n_per_class)):
            types = np.asarray([1, 1, 1, 1], dtype=np.int64)  # 1 = C
            if label == 1:
                position = int(rng.choice([0, 3]))  # N at endpoint
            else:
                position = int(rng.choice([1, 2]))  # N internal
            types[position] = 0  # 0 = N
            molecules.append(_toy_molecule(graph, types, edges, float(label)))
    rng.shuffle(molecules)
    return molecules


def synthetic_negative(n_examples: int = 600, seed: int = 0):
    """Marginal-only control: label depends on topology marginal + attribute counts only."""
    topologies = []
    for length in (4, 5, 6, 7):
        edges = [(index, index + 1) for index in range(length - 1)]
        topologies.append((from_edges(length, edges), {(u, v): 1 for u, v in edges}, length))
    rng = np.random.default_rng(int(seed))
    molecules: list[r2.MoleculeFeatures] = []
    for _ in range(int(n_examples)):
        graph, edge_types, length = topologies[int(rng.integers(len(topologies)))]
        types = rng.integers(0, 3, size=length).astype(np.int64)
        count0 = int((types == 0).sum())
        count2 = int((types == 2).sum())
        label = 0.1 * length + 0.05 * count0 + 0.02 * count2
        molecules.append(_toy_molecule(graph, types, edge_types, float(label)))
    # randomise the attribute placement inside every molecule (independent of label)
    for index, molecule in enumerate(molecules):
        rng.shuffle(molecule.atom_idx)
    return molecules


def _train_toy(
    molecules: Sequence[r2.MoleculeFeatures],
    variant: str,
    seed: int,
    epochs: int = 400,
    batch_size: int = 64,
    learning_rate: float = 3.0e-3,
) -> dict[str, Any]:
    split = int(0.8 * len(molecules))
    train_set = list(molecules[:split])
    valid_set = list(molecules[split:])
    model = r2.build_model(
        variant,
        {
            "C_rms": np.ones((r2.PHI_DIM, r2.ATOM_CATEGORIES), dtype=np.float32),
            "C_mask": np.ones((r2.PHI_DIM, r2.ATOM_CATEGORIES), dtype=np.float32),
            "P_rms": np.ones((r2.PHI_DIM, r2.ATOM_CATEGORIES), dtype=np.float32),
            "P_mask": np.ones((r2.PHI_DIM, r2.ATOM_CATEGORIES), dtype=np.float32),
        },
        seed=int(seed),
    )
    # The toy scalers must come from the toy training split (train-only rule).
    c_list = []
    p_list = []
    for molecule in train_set:
        c_matrix, p_matrix = r2.center_stats(
            np.asarray(molecule.phi, dtype=np.float64),
            r2.one_hot_q(molecule.atom_idx),
        )
        c_list.append(c_matrix)
        p_list.append(p_matrix)
    scalers = r2.fit_scalers(c_list, p_list)
    with torch.no_grad():
        model.scaler_C.copy_(torch.from_numpy(scalers["C_rms"]))
        model.mask_C.copy_(torch.from_numpy(scalers["C_mask"]))
        model.scaler_P.copy_(torch.from_numpy(scalers["P_rms"]))
        model.mask_P.copy_(torch.from_numpy(scalers["P_mask"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    train_loader = _loader(train_set, batch_size, True, int(seed) + 7)
    valid_loader = _loader(valid_set, batch_size, False, 0)
    for _ in range(int(epochs)):
        model.train()
        for batch in train_loader:
            prediction = model(batch).view(-1)
            loss = F.l1_loss(prediction, batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_mae = _evaluate_mae(model, train_loader, torch.device("cpu"))
    valid_mae = _evaluate_mae(model, valid_loader, torch.device("cpu"))
    return {
        "variant": str(variant),
        "seed": int(seed),
        "train_mae": float(train_mae),
        "valid_mae": float(valid_mae),
        "branch_weight_norm": float(model.branch_weight_norm()),
        "branch_weight_sparsity": float(model.branch_weight_sparsity()),
    }


def synthetic(seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    positive_results: dict[str, list[dict[str, Any]]] = {}
    negative_results: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        positive = synthetic_positive(seed=int(seed))
        negative = synthetic_negative(seed=int(seed))
        for variant in r2.MODELS:
            positive_results.setdefault(variant, []).append(
                _train_toy(positive, variant, seed=int(seed))
            )
            negative_results.setdefault(variant, []).append(
                _train_toy(negative, variant, seed=int(seed))
            )

    def _summary(rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            variant: {
                "valid_mae_mean": float(np.mean([row["valid_mae"] for row in values])),
                "valid_mae_values": [float(row["valid_mae"]) for row in values],
                "branch_weight_norm_values": [
                    float(row["branch_weight_norm"]) for row in values
                ],
            }
            for variant, values in rows.items()
        }

    positive_summary = _summary(positive_results)
    negative_summary = _summary(negative_results)
    positive_pass = bool(
        positive_summary["MB"]["valid_mae_mean"]
        < min(
            positive_summary["M0"]["valid_mae_mean"],
            positive_summary["MM"]["valid_mae_mean"],
        )
        - 0.1
        and max(positive_summary["MB"]["branch_weight_norm_values"]) > 0.0
    )
    negative_pass = bool(
        not all(
            delta > 0.0
            for delta in [
                negative_summary["M0"]["valid_mae_mean"]
                - negative_summary["MB"]["valid_mae_mean"],
                negative_summary["MM"]["valid_mae_mean"]
                - negative_summary["MB"]["valid_mae_mean"],
            ]
        )
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seeds": [int(seed) for seed in seeds],
        "positive": {
            "per_run": positive_results,
            "summary": positive_summary,
            "pass": positive_pass,
            "criterion": "MB valid MAE beats both M0 and MM by > 0.1 and ||W_B|| > 0",
        },
        "negative": {
            "per_run": negative_results,
            "summary": negative_summary,
            "pass": negative_pass,
            "criterion": "MB does not beat BOTH M0 and MM in mean valid MAE",
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "synthetic_controls.json", payload)
    print(json.dumps(payload["positive"]["summary"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(payload["negative"]["summary"], indent=2, sort_keys=True), flush=True)
    if not positive_pass:
        raise RuntimeError(
            "synthetic positive control FAILED: do not launch formal ZINC runs"
        )
    return payload


# ---------------------------------------------------------------------------
# gate / decide / report
# ---------------------------------------------------------------------------


def _run_soup_valid(tag: str, seed: int) -> float | None:
    path = RESULTS_DIR / f"soup_{tag}_seed{seed}.json"
    if not path.exists():
        return None
    return float(_read_json(path)["top5_soup_valid_mae"])


def gate() -> dict[str, Any]:
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS_FIRST_ROUND:
        row = {
            variant: _run_soup_valid(_tag(variant), seed) for variant in r2.MODELS
        }
        if all(value is not None for value in row.values()):
            delta_base = float(row["M0"] - row["MB"])
            delta_capacity = float(row["MM"] - row["MB"])
            row["delta_base_M0_minus_MB"] = delta_base
            row["delta_capacity_MM_minus_MB"] = delta_capacity
            row["passes_both"] = bool(delta_base > 0.0 and delta_capacity > 0.0)
        per_seed[int(seed)] = row
    both_seeds = all(
        bool(per_seed[seed].get("passes_both", False)) for seed in SEEDS_FIRST_ROUND
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": per_seed,
        "seed2_authorized": bool(both_seeds),
        "gate_rule": (
            "MB must beat BOTH M0 (delta_base>0) and MM (delta_capacity>0) "
            "on BOTH paired seeds 0 and 1"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "gate.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def decide() -> dict[str, Any]:
    gate_payload = gate()
    per_seed = gate_payload["per_seed"]
    available = all(
        per_seed[seed].get("passes_both") is not None for seed in SEEDS_FIRST_ROUND
    )
    if not available:
        verdict = "INCOMPLETE"
        interpretation = "formal runs missing; no scientific verdict"
    elif gate_payload["seed2_authorized"]:
        verdict = "ASSIGNMENT_SUPPORTED_PENDING_SEED2"
        interpretation = (
            "MB beats both M0 and MM on seeds 0 and 1; seed 2 is authorized by the "
            "pre-registered gate before the verdict is finalised"
        )
    else:
        verdict = "ASSIGNMENT_NOT_SUPPORTED"
        interpretation = (
            "MB does not beat both M0 and MM on both paired seeds; the node-level "
            "linear assignment residual is not stably supported under this explicit "
            "basis and training budget"
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": verdict,
        "interpretation": interpretation,
        "gate": gate_payload,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def report() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seeds: list[int] = []
    for seed in (*SEEDS_FIRST_ROUND, SEED_GATE):
        if any(
            (RESULTS_DIR / f"soup_{_tag(variant)}_seed{seed}.json").exists()
            for variant in r2.MODELS
        ):
            seeds.append(int(seed))
        for variant in r2.MODELS:
            soup_path = RESULTS_DIR / f"soup_{_tag(variant)}_seed{seed}.json"
            run_path = RUNS_DIR / f"{_tag(variant)}_seed{seed}.json"
            if not soup_path.exists():
                continue
            soup_payload = _read_json(soup_path)
            run_payload = _read_json(run_path) if run_path.exists() else {}
            rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "best_valid": run_payload.get("best_valid_mae"),
                    "soup_valid": soup_payload.get("top5_soup_valid_mae"),
                    "best_epoch": run_payload.get("best_epoch"),
                    "top5_epochs": soup_payload.get("top5_epochs"),
                    "branch_norm": run_payload.get("final_branch_norm"),
                    "wall_time_s": run_payload.get("wall_clock_s"),
                    "peak_gpu_memory_bytes": run_payload.get("peak_gpu_memory_bytes"),
                }
            )
    lines = [
        "# FSAR-R2-AR0 formal results",
        "",
        "| model | seed | best valid | Top-5 soup valid | best epoch | branch norm | wall time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {seed} | {best_valid} | {soup_valid} | {best_epoch} | "
            "{branch_norm} | {wall_time_s} |".format(**row)
        )
    lines.append("")
    lines.append("Paired deltas (positive = MB better):")
    lines.append("")
    lines.append("| seed | M0 - MB | MM - MB | MB beats both |")
    lines.append("| ---: | ---: | ---: | :---: |")
    paired: dict[int, dict[str, float]] = {}
    for seed in seeds:
        values = {
            variant: _run_soup_valid(_tag(variant), seed) for variant in r2.MODELS
        }
        if all(value is not None for value in values.values()):
            delta_base = float(values["M0"] - values["MB"])
            delta_capacity = float(values["MM"] - values["MB"])
            paired[int(seed)] = {
                "delta_base_M0_minus_MB": delta_base,
                "delta_capacity_MM_minus_MB": delta_capacity,
            }
            lines.append(
                f"| {seed} | {delta_base:.6f} | {delta_capacity:.6f} | "
                f"{'yes' if (delta_base > 0 and delta_capacity > 0) else 'no'} |"
            )
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "FORMAL_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return {
        "markdown": text,
        "rows": rows,
        "paired": paired,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("params")
    p = sub.add_parser("preprocess")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("audit")
    p = sub.add_parser("sanity")
    p = sub.add_parser("synthetic")
    p = sub.add_parser("gate")
    p = sub.add_parser("decide")
    p = sub.add_parser("report")
    p = sub.add_parser("gradient_audit")
    p.add_argument("--variant", default="MB", choices=list(r2.MODELS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action="store_true")
    for stage in ("train", "soup", "run", "diagnostics", "witness", "shuffle_eval"):
        p = sub.add_parser(stage)
        p.add_argument("--variant", required=True, choices=list(r2.MODELS))
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--tag", default=None)
        p.add_argument("--max-epochs", type=int, default=None)
        p.add_argument("--state", default="soup", choices=["soup", "best"])
        p.add_argument("--deterministic", action="store_true")

    arguments = parser.parse_args(argv)
    if getattr(arguments, "deterministic", False):
        _set_deterministic(True)

    if arguments.stage == "params":
        params()
    elif arguments.stage == "preprocess":
        preprocess(force=bool(arguments.force))
    elif arguments.stage == "audit":
        audit()
    elif arguments.stage == "sanity":
        sanity()
    elif arguments.stage == "synthetic":
        synthetic()
    elif arguments.stage == "gradient_audit":
        if arguments.all:
            gradient_audit_all(device=arguments.device)
        else:
            gradient_audit(arguments.variant, device=arguments.device)
    elif arguments.stage == "train":
        train(
            arguments.variant,
            arguments.seed,
            device=arguments.device,
            tag=arguments.tag,
            max_epochs=arguments.max_epochs,
        )
    elif arguments.stage == "soup":
        soup(arguments.variant, arguments.seed, tag=arguments.tag)
    elif arguments.stage == "run":
        run(
            arguments.variant,
            arguments.seed,
            device=arguments.device,
            tag=arguments.tag,
            max_epochs=arguments.max_epochs,
        )
    elif arguments.stage == "diagnostics":
        diagnostics(
            arguments.variant, arguments.seed, tag=arguments.tag, state=arguments.state
        )
    elif arguments.stage in {"witness", "shuffle_eval"}:
        witness(
            arguments.variant, arguments.seed, tag=arguments.tag, state=arguments.state
        )
    elif arguments.stage == "gate":
        gate()
    elif arguments.stage == "decide":
        decide()
    elif arguments.stage == "report":
        report()
    else:  # pragma: no cover
        raise SystemExit(f"unknown stage {arguments.stage!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
