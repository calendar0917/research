"""FSAR-R2-AR0-EDGE runner: edge structural-role <-> bond-type assignment residual.

Round name: ``FSAR-R2-AR0-EDGE``.

Stages
------
``params preprocess audit sanity gradient_audit synthetic train soup run
diagnostics witness gate decide analyze report``

Three models, all sharing the frozen base ``F0`` and the AR0 **node**
assignment term ``B_V``:

* ``BV``    ``yhat = F0 + <W_B, C~_V>``                     (node baseline)
* ``BVE``   ``yhat = F0 + <W_B, C~_V> + <W_E, C~_E>``       (real edge assignment)
* ``BVEM``  ``yhat = F0 + <W_B, C~_V> + <W_ME, P~_E>``      (matched control)

The training protocol is inherited verbatim from AR0.  Official ZINC test is
never loaded.  The synthetic positive control must pass before any formal ZINC
run; the negative control must NOT show a repeatable BVE advantage.
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
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0_edge as edge
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0 as ar0
from tracks.ksvd.experiments.luyin16.zinc_fsar import _set_deterministic
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _data_to_graph, _load_zinc

REPO_ROOT = ar0.REPO_ROOT
TRACK_ROOT = ar0.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/fsar_r2_ar0_edge"
CACHE_DIR = RESULTS_DIR / "cache"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SOUP_DIR = RESULTS_DIR / "soup_states"
RUNS_DIR = RESULTS_DIR / "runs"

ZINC_ROOT = ar0.ZINC_ROOT
PROTOCOL_VERSION = "fsar_r2_ar0_edge_v1"
CACHE_SCHEMA = "fsar_r2_ar0_edge_features_v1"

_write_json = ar0._write_json
_read_json = ar0._read_json
_git_commit = ar0._git_commit
_environment_fingerprint = ar0._environment_fingerprint

SEEDS_FIRST_ROUND = (0, 1)
SEED_GATE = 2

DIAG_BATCHES = 32
SHUFFLE_REPEATS = 5


def _tag(variant: str, tag: str | None = None) -> str:
    return str(tag or f"r2ar0e_{str(variant).lower()}")


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


def _extract_edge_split(split: str) -> list[edge.EdgeMoleculeFeatures]:
    dataset = _load_zinc(ZINC_ROOT, split)
    molecules: list[edge.EdgeMoleculeFeatures] = []
    for data in dataset:
        graph, node_types, edge_types = _data_to_graph(data)
        phi = r2.build_phi(graph)
        edges = sorted((int(u), int(v)) for u, v in graph.edges())
        edge_u = np.asarray([u for u, _v in edges], dtype=np.int64)
        edge_v = np.asarray([v for _u, v in edges], dtype=np.int64)
        bond_type = np.asarray([int(edge_types[(u, v)]) for u, v in edges], dtype=np.int64)
        n_nodes = int(data.num_nodes)
        marginal = r2.build_A(node_types, edge_types, n_nodes)
        if phi.shape[0] != n_nodes:
            raise RuntimeError("phi / node count mismatch")
        if edge_u.shape[0] != len(edge_types):
            raise RuntimeError("undirected edge count mismatch")
        molecules.append(
            edge.EdgeMoleculeFeatures(
                phi=phi.astype(np.float32),
                atom_idx=np.asarray(node_types, dtype=np.int64),
                edge_u=edge_u,
                edge_v=edge_v,
                bond_type=bond_type,
                A=marginal.astype(np.float32),
                n_nodes=n_nodes,
                n_edges=int(len(edges)),
                y=float(data.y.view(-1)[0]),
            )
        )
    return molecules


def _compute_scalers(
    molecules: Sequence[edge.EdgeMoleculeFeatures],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    c_matrices: list[np.ndarray] = []
    p_matrices: list[np.ndarray] = []
    ce_matrices: list[np.ndarray] = []
    pe_matrices: list[np.ndarray] = []
    for molecule in molecules:
        c_matrix, p_matrix = r2.center_stats(
            np.asarray(molecule.phi, dtype=np.float64),
            r2.one_hot_q(molecule.atom_idx),
        )
        c_matrices.append(c_matrix)
        p_matrices.append(p_matrix)
        ps = edge.build_edge_roles(
            np.asarray(molecule.phi, dtype=np.float64),
            list(zip(molecule.edge_u.tolist(), molecule.edge_v.tolist())),
        )
        r = edge.one_hot_bond(molecule.bond_type)
        c_edge, p_edge = edge.edge_center_stats(ps, r)
        ce_matrices.append(c_edge)
        pe_matrices.append(p_edge)
    scalers = r2.fit_scalers(c_matrices, p_matrices)
    scalers.update(edge.fit_edge_scalers(ce_matrices, pe_matrices))
    report = {
        "n_train_molecules": int(len(molecules)),
        "C_effective_coordinates": int(scalers["C_mask"].sum()),
        "C_total_coordinates": int(scalers["C_mask"].size),
        "CE_effective_coordinates": int(scalers["CE_mask"].sum()),
        "CE_total_coordinates": int(scalers["CE_mask"].size),
        "PE_effective_coordinates": int(scalers["PE_mask"].sum()),
        "PE_total_coordinates": int(scalers["PE_mask"].size),
        "CE_rms_max": float(scalers["CE_rms_raw"].max()),
        "PE_rms_max": float(scalers["PE_rms_raw"].max()),
        "dataset_mean_subtracted": False,
        "scaler_source": "train split only",
    }
    return scalers, report


def build_edge_datasets(force: bool = False):
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
    ar0._assert_no_test_access()
    train_molecules = _extract_edge_split("train")
    valid_molecules = _extract_edge_split("val")
    scalers, scaler_report = _compute_scalers(train_molecules)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(train_path, "wb") as handle:
        pickle.dump(train_molecules, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(valid_path, "wb") as handle:
        pickle.dump(valid_molecules, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(
        {key: torch.from_numpy(np.asarray(value)) for key, value in scalers.items()},
        scaler_path,
    )
    meta = {
        "cache_schema": CACHE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "phi_dim": int(edge.PHI_DIM),
        "edge_role_dim": int(edge.EDGE_ROLE_DIM),
        "bond_categories": int(edge.BOND_CATEGORIES),
        "atom_categories": int(edge.ATOM_CATEGORIES),
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
        f"[preprocess-edge] train={len(train_molecules)} valid={len(valid_molecules)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_molecules, valid_molecules, scalers, meta


def preprocess(force: bool = False) -> dict[str, Any]:
    _train, _valid, _scalers, meta = build_edge_datasets(force=force)
    return meta


def _loader(molecules, batch_size, shuffle, seed):
    return edge.make_edge_loader(molecules, batch_size, shuffle, seed)


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
    _train, _valid, scalers, _meta = build_edge_datasets()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "models": {},
        "official_test_loaded": False,
    }
    for variant in edge.MODELS:
        model = edge.build_edge_model(variant, scalers, seed=0)
        payload["models"][variant] = edge.parameter_breakdown_edge(model)
    totals = {variant: row["total"] for variant, row in payload["models"].items()}
    payload["totals"] = totals
    payload["shared_init_identity"] = bool(
        payload["models"]["BV"]["S_encoder"] == payload["models"]["BVE"]["S_encoder"]
        and payload["models"]["BV"]["base_F0"] == payload["models"]["BVE"]["base_F0"]
        and payload["models"]["BV"]["node_binding"] == payload["models"]["BVE"]["node_binding"]
        and payload["models"]["BVE"]["node_binding"] == payload["models"]["BVEM"]["node_binding"]
    )
    payload["W_E_W_ME_same_shape"] = True
    payload["dataset_dependent_vocabulary_params"] = 0
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# sanity / gradient audit
# ---------------------------------------------------------------------------


def sanity() -> dict[str, Any]:
    from tracks.ksvd.tests import test_fsar_r2_ar0_edge as suite

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
        "edge_endpoint_swap_invariance",
        "edge_role_chemistry_purity",
        "node_relabel_invariance",
        "bond_permutation_channel_invariance",
        "bond_permutation_changes_ce",
        "exact_permutation_expectation_edge",
        "scalar_edge_exact_expectation",
        "undirected_bond_canonicalization",
        "train_only_edge_scaler",
        "batched_edge_stats_match_numpy",
        "edge_weight_gradient_viability",
        "no_edge_bypass",
        "bv_bve_bvem_share_init",
        "zero_init_prediction_identity",
        "parameter_counts_edge",
        "positive_control_edge_assignment",
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
        raise RuntimeError(f"FSAR-R2-AR0-EDGE sanity gates failed: {failures}")
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
    _train, valid_molecules, scalers, _meta = build_edge_datasets()
    device_obj = torch.device(device)
    model = edge.build_edge_model(variant, scalers, seed=0).to(device_obj)
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
        "W_B": float(model.W_B.grad.detach().abs().sum()),
    }
    if model.W_E is not None:
        named["W_E"] = float(model.W_E.grad.detach().abs().sum())
    if model.W_ME is not None:
        named["W_ME"] = float(model.W_ME.grad.detach().abs().sum())
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
        "W_B_nonzero_grad": bool(float(model.W_B.grad.detach().abs().sum()) > 0.0),
        "W_E_nonzero_grad": bool(
            model.W_E is not None and float(model.W_E.grad.detach().abs().sum()) > 0.0
        ),
        "W_ME_nonzero_grad": bool(
            model.W_ME is not None and float(model.W_ME.grad.detach().abs().sum()) > 0.0
        ),
        "official_test_loaded": False,
    }
    if variant == "BVE" and not payload["W_E_nonzero_grad"]:
        raise RuntimeError("BVE edge weight received zero task gradient")
    if variant == "BVEM" and not payload["W_ME_nonzero_grad"]:
        raise RuntimeError("BVEM matched edge weight received zero task gradient")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / f"gradient_audit_{variant}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def gradient_audit_all(device: str = "cpu") -> dict[str, Any]:
    return {variant: gradient_audit(variant, device=device) for variant in edge.MODELS}


# ---------------------------------------------------------------------------
# cheap feature audit
# ---------------------------------------------------------------------------


def audit(sample_molecules: int = 500) -> dict[str, Any]:
    train, _valid, scalers, meta = build_edge_datasets()
    sample = list(train[: int(sample_molecules)])
    psi_rows: list[np.ndarray] = []
    ce_flat: list[np.ndarray] = []
    pe_flat: list[np.ndarray] = []
    n_edges: list[int] = []
    for molecule in sample:
        ps = edge.build_edge_roles(
            np.asarray(molecule.phi, dtype=np.float64),
            list(zip(molecule.edge_u.tolist(), molecule.edge_v.tolist())),
        )
        r = edge.one_hot_bond(molecule.bond_type)
        c_edge, p_edge = edge.edge_center_stats(ps, r)
        psi_rows.append(ps)
        ce_flat.append(c_edge.reshape(-1))
        pe_flat.append(p_edge.reshape(-1))
        n_edges.append(int(molecule.n_edges))
    psi_all = np.concatenate(psi_rows, axis=0) if psi_rows else np.zeros((0, edge.EDGE_ROLE_DIM))
    ce_all = np.stack(ce_flat, axis=0)
    pe_all = np.stack(pe_flat, axis=0)
    psi_std = psi_all.std(axis=0) if psi_all.size else np.zeros(edge.EDGE_ROLE_DIM)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_sampled_molecules": int(len(sample)),
        "psi": {
            "dim": int(edge.EDGE_ROLE_DIM),
            "zero_std_coordinates": int((psi_std <= 0.0).sum()),
            "nan_or_inf": bool(not np.isfinite(psi_all).all()),
            **{f"rank_{k}": v for k, v in ar0._effective_rank(psi_all).items()},
        },
        "C_E": {
            "shape": [int(edge.EDGE_ROLE_DIM), int(edge.BOND_CATEGORIES)],
            "effective_coordinates": int(scalers["CE_mask"].sum()),
            "total_coordinates": int(scalers["CE_mask"].size),
            "nan_or_inf": bool(not np.isfinite(ce_all).all()),
            **{f"rank_{k}": v for k, v in ar0._effective_rank(ce_all).items()},
        },
        "P_E": {
            "shape": [int(edge.EDGE_ROLE_DIM), int(edge.BOND_CATEGORIES)],
            "effective_coordinates": int(scalers["PE_mask"].sum()),
            "total_coordinates": int(scalers["PE_mask"].size),
            "nan_or_inf": bool(not np.isfinite(pe_all).all()),
        },
        "mean_n_edges": float(np.mean(n_edges)) if n_edges else 0.0,
        "scaler_report": meta["scaler_report"],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "feature_audit.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _branch_curve_stats(model: edge.FSARR2AR0EdgeModel, batches) -> dict[str, float]:
    model.eval()
    base_values: list[np.ndarray] = []
    node_values: list[np.ndarray] = []
    edge_values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            base_values.append(model.base_prediction(batch).view(-1).cpu().numpy())
            c_node, _p_node = model.compute_node_statistics(batch)
            node_values.append(model.node_assignment_term(c_node).view(-1).cpu().numpy())
            if model.variant in edge.EDGE_MODELS:
                c_edge, p_edge = model.compute_edge_statistics(batch)
                edge_values.append(
                    model.edge_assignment_term(c_edge, p_edge).view(-1).cpu().numpy()
                )
    model.train()
    base_all = np.concatenate(base_values) if base_values else np.zeros(1)
    node_all = np.concatenate(node_values) if node_values else np.zeros(1)
    edge_all = np.concatenate(edge_values) if edge_values else np.zeros(1)
    weight = model.W_E if model.W_E is not None else model.W_ME
    return {
        "base_out_mean": float(base_all.mean()),
        "base_out_std": float(base_all.std()),
        "node_branch_out_mean": float(node_all.mean()),
        "node_branch_out_std": float(node_all.std()),
        "edge_branch_out_mean": float(edge_all.mean()),
        "edge_branch_out_std": float(edge_all.std()),
        "branch_norm": float(model.branch_weight_norm()),
        "node_branch_norm": float(model.node_weight_norm()),
        "edge_branch_norm": float(model.edge_weight_norm()),
        "edge_weight_abs_sum": float(weight.detach().abs().sum()) if weight is not None else 0.0,
        "edge_weight_zero_count": int((weight.detach().abs() <= 1.0e-8).sum()) if weight is not None else 0,
        "edge_weight_numel": int(weight.numel()) if weight is not None else 0,
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
    train_molecules, valid_molecules, scalers, _meta = build_edge_datasets()
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
    model = edge.build_edge_model(variant, scalers, seed=int(seed)).to(device_obj)
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
                f"edge_norm={stats['edge_branch_norm']:.4e} "
                f"edge_std={stats['edge_branch_out_std']:.4e}",
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
        "parameter_breakdown": edge.parameter_breakdown_edge(model),
        "final_branch_norm": float(model.branch_weight_norm()),
        "final_node_branch_norm": float(model.node_weight_norm()),
        "final_edge_branch_norm": float(model.edge_weight_norm()),
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
        SOUP_DIR / f"{tag}_seed{seed}_top5_states.pt", map_location="cpu", weights_only=False
    )
    if len(entries) < 5:
        raise RuntimeError(f"need >= 5 snapshots for top-5 soup, got {len(entries)}")
    ranked = sorted(entries, key=lambda row: (float(row["valid_mae"]), int(row["epoch"])))[:5]
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

    _train, valid_molecules, scalers, _meta = build_edge_datasets()
    loader = _loader(valid_molecules, 128, False, 0)
    device = torch.device("cpu")
    model = edge.build_edge_model(variant, scalers)
    selection_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    targets, best_preds = _predict_state(
        model, torch.load(selection_path, map_location="cpu", weights_only=True), loader, device
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
# frozen eval: diagnostics / bond-type permutation
# ---------------------------------------------------------------------------


def _load_frozen(variant: str, seed: int, tag: str | None = None, state: str = "soup"):
    _train, _valid, scalers, _meta = build_edge_datasets()
    model = edge.build_edge_model(variant, scalers)
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
    _train, valid_molecules, _scalers, _meta = build_edge_datasets()
    batches = list(_loader(valid_molecules, 128, False, 0))
    base_values: list[np.ndarray] = []
    node_values: list[np.ndarray] = []
    edge_values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            base_values.append(model.base_prediction(batch).view(-1).numpy())
            c_node, _p = model.compute_node_statistics(batch)
            node_values.append(model.node_assignment_term(c_node).view(-1).numpy())
            if model.variant in edge.EDGE_MODELS:
                c_edge, p_edge = model.compute_edge_statistics(batch)
                edge_values.append(model.edge_assignment_term(c_edge, p_edge).view(-1).numpy())
    base_all = np.concatenate(base_values)
    node_all = np.concatenate(node_values)
    edge_all = np.concatenate(edge_values) if edge_values else np.zeros(1)
    weight = model.W_E if model.W_E is not None else model.W_ME
    # task gradient norm of the edge parameter on one real valid batch
    edge_grad_norm = 0.0
    if weight is not None:
        model.train()
        batch = next(iter(_loader(list(valid_molecules)[:128], 128, False, 0)))
        loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))
        model.zero_grad()
        loss.backward()
        edge_grad_norm = float(weight.grad.detach().norm()) if weight.grad is not None else 0.0
        model.zero_grad()
        model.eval()
    mask = model.mask_CE if model.variant == "BVE" else model.mask_PE
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": str(variant),
        "seed": int(seed),
        "state": str(state),
        "base_out_mean": float(base_all.mean()),
        "base_out_std": float(base_all.std()),
        "node_branch_out_mean": float(node_all.mean()),
        "node_branch_out_std": float(node_all.std()),
        "node_branch_norm": float(model.node_weight_norm()),
        "edge_branch_out_mean": float(edge_all.mean()),
        "edge_branch_out_std": float(edge_all.std()),
        "edge_branch_norm": float(model.edge_weight_norm()),
        "edge_weight_zero_count": int((weight.detach().abs() <= 1.0e-8).sum()) if weight is not None else 0,
        "edge_weight_numel": int(weight.numel()) if weight is not None else 0,
        "edge_weight_sparsity": (
            float((weight.detach().abs() <= 1.0e-8).to(torch.float32).mean()) if weight is not None else 0.0
        ),
        "edge_effective_nonmasked_coordinates": int(mask.sum()) if weight is not None else 0,
        "edge_task_gradient_norm": edge_grad_norm,
        "branch_live": bool(float(edge_all.std()) > 0.0),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"diagnostics_{_tag(variant, tag)}_seed{seed}_{state}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def witness(
    variant: str, seed: int, tag: str | None = None, state: str = "soup", repeats: int = SHUFFLE_REPEATS
) -> dict[str, Any]:
    """Evaluation-only bond-type permutation across the undirected edges."""
    model = _load_frozen(variant, seed, tag, state=state)
    _train, valid_molecules, _scalers, _meta = build_edge_datasets()
    batches = list(_loader(valid_molecules, 128, False, 0))
    targets: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    perm_preds: list[list[np.ndarray]] = [[] for _ in range(int(repeats))]
    channel_deltas = {"S": 0.0, "A": 0.0, "node_C": 0.0, "P_E": 0.0, "C_E": 0.0, "prediction": 0.0}
    with torch.no_grad():
        for batch in batches:
            targets.append(batch.y.view(-1).numpy())
            actual.append(model(batch).view(-1).numpy())
            c_node_actual, _p_node = model.compute_node_statistics(batch)
            c_edge_actual, p_edge_actual = model.compute_edge_statistics(batch)
            for repeat in range(int(repeats)):
                r_permuted = edge.permute_bond_types_within_graphs(
                    batch, seed=1000 * int(seed) + repeat
                )
                shuffled_batch = batch.with_r(r_permuted)
                prediction = model(shuffled_batch).view(-1)
                perm_preds[repeat].append(prediction.numpy())
                if repeat == 0:
                    s_actual = model.encode_s(batch)
                    s_shuffled = model.encode_s(shuffled_batch)
                    channel_deltas["S"] += float((s_shuffled - s_actual).abs().sum())
                    channel_deltas["A"] += float((shuffled_batch.A - batch.A).abs().sum())
                    c_node_shuffled, _p = model.compute_node_statistics(shuffled_batch)
                    c_edge_shuffled, p_edge_shuffled = model.compute_edge_statistics(shuffled_batch)
                    channel_deltas["node_C"] += float((c_node_shuffled - c_node_actual).abs().sum())
                    channel_deltas["P_E"] += float((p_edge_shuffled - p_edge_actual).abs().sum())
                    channel_deltas["C_E"] += float((c_edge_shuffled - c_edge_actual).abs().sum())
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
        "permutation_mean_prediction_mae": float(np.mean(np.abs(targets_all - perm_mean_prediction))),
        "mean_of_permutation_maes": float(np.mean(perm_maes)),
        "permutation_mae_std": float(np.std(perm_maes)),
        "permutation_maes": perm_maes,
        "mean_abs_prediction_change": float(np.mean(np.abs(perm_matrix - actual_all[None, :]))),
        "channel_delta_abs_sum": channel_deltas,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"witness_{_tag(variant, tag)}_seed{seed}_{state}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# synthetic controls (CPU, no ZINC)
# ---------------------------------------------------------------------------


def _toy_molecule(graph, node_types, edges, bond_type, label) -> edge.EdgeMoleculeFeatures:
    phi = r2.build_phi(graph)
    edge_list = sorted((int(u), int(v)) for u, v in graph.edges())
    edge_u = np.asarray([u for u, _v in edge_list], dtype=np.int64)
    edge_v = np.asarray([v for _u, v in edge_list], dtype=np.int64)
    bond_type = np.asarray(bond_type, dtype=np.int64)
    marginal = r2.build_A(node_types, dict(zip(edge_list, bond_type.tolist())), len(node_types))
    return edge.EdgeMoleculeFeatures(
        phi=phi.astype(np.float32),
        atom_idx=np.asarray(node_types, dtype=np.int64),
        edge_u=edge_u,
        edge_v=edge_v,
        bond_type=bond_type,
        A=marginal.astype(np.float32),
        n_nodes=int(len(node_types)),
        n_edges=int(len(edge_list)),
        y=float(label),
    )


def synthetic_edge_positive(n_examples: int = 600, seed: int = 0):
    """Assignment-only control: fixed tree, all-C atoms, one DOUBLE bond."""

    graph = from_edges(5, [(0, 1), (1, 2), (2, 3), (2, 4)])
    edges = sorted((int(u), int(v)) for u, v in graph.edges())
    target_edge = (1, 2)
    rng = np.random.default_rng(int(seed))
    molecules: list[edge.EdgeMoleculeFeatures] = []
    for _ in range(int(n_examples)):
        position = int(rng.integers(len(edges)))
        bond_type = np.full(len(edges), 1, dtype=np.int64)
        bond_type[position] = 2
        label = 1.0 if edges[position] == target_edge else 0.0
        molecules.append(
            _toy_molecule(
                graph,
                np.zeros(5, dtype=np.int64),
                edges,
                bond_type,
                label,
            )
        )
    rng.shuffle(molecules)
    return molecules


def synthetic_edge_negative(n_examples: int = 600, seed: int = 0):
    """Marginal-only control: label depends only on topology / atom / bond counts."""

    topologies = []
    for length in (4, 5, 6):
        edges = [(index, index + 1) for index in range(length - 1)]
        topologies.append((from_edges(length, edges), length))
    rng = np.random.default_rng(int(seed))
    molecules: list[edge.EdgeMoleculeFeatures] = []
    for _ in range(int(n_examples)):
        graph, length = topologies[int(rng.integers(len(topologies)))]
        edges = sorted((int(u), int(v)) for u, v in graph.edges())
        node_types = rng.integers(0, 3, size=length).astype(np.int64)
        bond_type = rng.integers(1, 4, size=len(edges)).astype(np.int64)
        count0 = int((node_types == 0).sum())
        label = 0.1 * length + 0.05 * count0 + 0.02 * int((bond_type == 2).sum())
        molecule = _toy_molecule(graph, node_types, edges, bond_type, float(label))
        # Randomise the assignment inside every molecule (label-independent).
        rng.shuffle(molecule.atom_idx)
        order = rng.permutation(molecule.bond_type.shape[0])
        molecule.bond_type = molecule.bond_type[order]
        molecules.append(molecule)
    return molecules


def _dummy_edge_scalers(sample) -> dict[str, np.ndarray]:
    c_node: list[np.ndarray] = []
    p_node: list[np.ndarray] = []
    c_edge: list[np.ndarray] = []
    p_edge: list[np.ndarray] = []
    for molecule in sample:
        c, p = r2.center_stats(
            np.asarray(molecule.phi, dtype=np.float64), r2.one_hot_q(molecule.atom_idx)
        )
        c_node.append(c)
        p_node.append(p)
        ps = edge.build_edge_roles(
            np.asarray(molecule.phi, dtype=np.float64),
            list(zip(molecule.edge_u.tolist(), molecule.edge_v.tolist())),
        )
        ce, pe = edge.edge_center_stats(ps, edge.one_hot_bond(molecule.bond_type))
        c_edge.append(ce)
        p_edge.append(pe)
    scalers = r2.fit_scalers(c_node, p_node)
    scalers.update(edge.fit_edge_scalers(c_edge, p_edge))
    return scalers


def _train_edge_toy(
    molecules: Sequence[edge.EdgeMoleculeFeatures],
    variant: str,
    seed: int,
    epochs: int = 400,
    batch_size: int = 64,
    learning_rate: float = 3.0e-3,
) -> dict[str, Any]:
    split = int(0.8 * len(molecules))
    train_set = list(molecules[:split])
    valid_set = list(molecules[split:])
    scalers = _dummy_edge_scalers(train_set)
    model = edge.build_edge_model(variant, scalers, seed=int(seed))
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
    weight = model.W_E if model.W_E is not None else model.W_ME
    return {
        "variant": str(variant),
        "seed": int(seed),
        "train_mae": float(train_mae),
        "valid_mae": float(valid_mae),
        "branch_weight_norm": float(model.branch_weight_norm()),
        "edge_weight_norm": float(model.edge_weight_norm()),
        "edge_weight_nonzero": int((weight.detach().abs() > 1.0e-12).sum()) if weight is not None else 0,
    }


def synthetic(seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    positive_results: dict[str, list[dict[str, Any]]] = {}
    negative_results: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        positive = synthetic_edge_positive(seed=int(seed))
        negative = synthetic_edge_negative(seed=int(seed))
        for variant in edge.MODELS:
            positive_results.setdefault(variant, []).append(
                _train_edge_toy(positive, variant, seed=int(seed))
            )
            negative_results.setdefault(variant, []).append(
                _train_edge_toy(negative, variant, seed=int(seed))
            )

    def _summary(rows):
        return {
            variant: {
                "valid_mae_mean": float(np.mean([row["valid_mae"] for row in values])),
                "valid_mae_values": [float(row["valid_mae"]) for row in values],
                "edge_weight_norm_values": [float(row["edge_weight_norm"]) for row in values],
            }
            for variant, values in rows.items()
        }

    positive_summary = _summary(positive_results)
    negative_summary = _summary(negative_results)
    # Positive: BVE beats both BV and BVEM; BVEM must NOT gain the real assignment.
    positive_pass = bool(
        positive_summary["BVE"]["valid_mae_mean"]
        < min(positive_summary["BV"]["valid_mae_mean"], positive_summary["BVEM"]["valid_mae_mean"]) - 0.1
        and positive_summary["BVEM"]["valid_mae_mean"] > 0.2
        and max(positive_summary["BVE"]["edge_weight_norm_values"]) > 0.0
    )
    # Negative: BVE must not beat BOTH BV and BVEM in mean valid MAE.
    negative_pass = bool(
        not all(
            delta > 0.0
            for delta in [
                negative_summary["BV"]["valid_mae_mean"] - negative_summary["BVE"]["valid_mae_mean"],
                negative_summary["BVEM"]["valid_mae_mean"] - negative_summary["BVE"]["valid_mae_mean"],
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
            "criterion": "BVE beats both BV and BVEM by > 0.1, BVEM stays high, ||W_E|| > 0",
        },
        "negative": {
            "per_run": negative_results,
            "summary": negative_summary,
            "pass": negative_pass,
            "criterion": "BVE does not beat BOTH BV and BVEM in mean valid MAE",
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "synthetic_controls.json", payload)
    print(json.dumps(payload["positive"]["summary"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(payload["negative"]["summary"], indent=2, sort_keys=True), flush=True)
    if not positive_pass:
        raise RuntimeError("synthetic edge positive control FAILED: do not launch formal ZINC runs")
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
        row = {variant: _run_soup_valid(_tag(variant), seed) for variant in edge.MODELS}
        if all(value is not None for value in row.values()):
            delta_edge = float(row["BV"] - row["BVE"])
            delta_control = float(row["BVEM"] - row["BVE"])
            row["delta_edge_BV_minus_BVE"] = delta_edge
            row["delta_edge_control_BVEM_minus_BVE"] = delta_control
            row["passes_both"] = bool(delta_edge > 0.0 and delta_control > 0.0)
        per_seed[int(seed)] = row
    both_seeds = all(bool(per_seed[seed].get("passes_both", False)) for seed in SEEDS_FIRST_ROUND)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": per_seed,
        "seed2_authorized": bool(both_seeds),
        "gate_rule": (
            "BVE must beat BOTH BV (delta_edge>0) and BVEM (delta_control>0) "
            "on BOTH paired seeds 0 and 1"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "gate.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def decide() -> dict[str, Any]:
    gate_payload = gate()
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in (*SEEDS_FIRST_ROUND, SEED_GATE):
        values = {variant: _run_soup_valid(_tag(variant), seed) for variant in edge.MODELS}
        if not all(value is not None for value in values.values()):
            continue
        delta_edge = float(values["BV"] - values["BVE"])
        delta_control = float(values["BVEM"] - values["BVE"])
        per_seed[int(seed)] = {
            **values,
            "delta_edge_BV_minus_BVE": delta_edge,
            "delta_edge_control_BVEM_minus_BVE": delta_control,
            "passes_both": bool(delta_edge > 0.0 and delta_control > 0.0),
        }
    complete_seeds = sorted(per_seed)
    all_pass = bool(complete_seeds) and all(per_seed[seed]["passes_both"] for seed in complete_seeds)
    if len(complete_seeds) == 0:
        verdict = "INCOMPLETE"
        interpretation = "formal runs missing; no scientific verdict"
    elif len(complete_seeds) >= 2 and all_pass:
        verdict = "EDGE_ASSIGNMENT_SUPPORTED"
        interpretation = (
            "BVE beats both BV and BVEM on every paired seed run "
            f"{complete_seeds}: under the fixed radius-2 explicit coordinates, the real "
            "edge structural-role <-> bond-type assignment provides a stable predictive "
            "increment beyond the node assignment and a matched edge marginal control"
        )
    else:
        verdict = "EDGE_ASSIGNMENT_NOT_SUPPORTED_OR_UNSTABLE"
        interpretation = (
            "BVE does not beat both BV and BVEM on all paired seeds; the edge-level linear "
            "assignment residual is not stably supported under this explicit basis / budget"
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": verdict,
        "interpretation": interpretation,
        "complete_seeds": [int(seed) for seed in complete_seeds],
        "all_complete_seeds_pass": bool(all_pass),
        "per_seed": per_seed,
        "gate": gate_payload,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# local paired analysis on pulled results
# ---------------------------------------------------------------------------


def _soup_predictions(variant: str, seed: int, loader, scalers):
    state_path = SOUP_DIR / f"{_tag(variant)}_seed{seed}_top5_soup.pt"
    if not state_path.exists():
        return None
    model = edge.build_edge_model(variant, scalers)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    return _predict_state(model, state, loader, torch.device("cpu"))


def analyze(repeats: int = 5000, bootstrap_seed: int = 20260918) -> dict[str, Any]:
    _train, valid_molecules, scalers, _meta = build_edge_datasets()
    loader = _loader(valid_molecules, 128, False, 0)
    rng = np.random.default_rng(int(bootstrap_seed))
    targets: np.ndarray | None = None
    per_seed: dict[str, Any] = {}
    for seed in (*SEEDS_FIRST_ROUND, SEED_GATE):
        predictions = {variant: _soup_predictions(variant, seed, loader, scalers) for variant in edge.MODELS}
        if any(value is None for value in predictions.values()):
            continue
        targets = predictions["BV"][0]
        matrix = {variant: predictions[variant][1] for variant in edge.MODELS}
        n = int(targets.shape[0])
        index = rng.integers(0, n, size=(int(repeats), n))
        bv_mae = np.abs(matrix["BV"][index] - targets[index]).mean(axis=1)
        bve_mae = np.abs(matrix["BVE"][index] - targets[index]).mean(axis=1)
        bvem_mae = np.abs(matrix["BVEM"][index] - targets[index]).mean(axis=1)
        delta_edge = bv_mae - bve_mae
        delta_control = bvem_mae - bve_mae

        def _ci(values):
            return {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "prob_gt_zero": float((values > 0.0).mean()),
            }

        per_seed[str(int(seed))] = {
            "point_mae": {variant: float(np.mean(np.abs(matrix[variant] - targets))) for variant in edge.MODELS},
            "delta_edge_BV_minus_BVE": _ci(delta_edge),
            "delta_edge_control_BVEM_minus_BVE": _ci(delta_control),
            "passes_both": bool(delta_edge.mean() > 0.0 and delta_control.mean() > 0.0),
        }
    complete = sorted(per_seed, key=int)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "bootstrap_repeats": int(repeats),
        "bootstrap_seed": int(bootstrap_seed),
        "n_valid_molecules": int(targets.shape[0]) if targets is not None else 0,
        "per_seed": per_seed,
        "complete_seeds": complete,
        "all_complete_seeds_pass": bool(complete) and all(per_seed[seed]["passes_both"] for seed in complete),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "analysis_paired_bootstrap.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def report() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seeds: list[int] = []
    for seed in (*SEEDS_FIRST_ROUND, SEED_GATE):
        if any((RESULTS_DIR / f"soup_{_tag(variant)}_seed{seed}.json").exists() for variant in edge.MODELS):
            seeds.append(int(seed))
        for variant in edge.MODELS:
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
                    "node_branch_norm": run_payload.get("final_node_branch_norm"),
                    "edge_branch_norm": run_payload.get("final_edge_branch_norm"),
                    "wall_time_s": run_payload.get("wall_clock_s"),
                    "peak_gpu_memory_bytes": run_payload.get("peak_gpu_memory_bytes"),
                }
            )
    lines = [
        "# FSAR-R2-AR0-EDGE formal results",
        "",
        "| model | seed | best valid | Top-5 soup valid | best epoch | node B norm | edge branch norm | wall time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {seed} | {best_valid} | {soup_valid} | {best_epoch} | "
            "{node_branch_norm} | {edge_branch_norm} | {wall_time_s} |".format(**row)
        )
    lines.append("")
    lines.append("Paired deltas (positive = BVE better):")
    lines.append("")
    lines.append("| seed | BV - BVE | BVEM - BVE | BVE beats both |")
    lines.append("| ---: | ---: | ---: | :---: |")
    paired: dict[int, dict[str, float]] = {}
    for seed in seeds:
        values = {variant: _run_soup_valid(_tag(variant), seed) for variant in edge.MODELS}
        if all(value is not None for value in values.values()):
            delta_edge = float(values["BV"] - values["BVE"])
            delta_control = float(values["BVEM"] - values["BVE"])
            paired[int(seed)] = {"delta_edge": delta_edge, "delta_control": delta_control}
            lines.append(
                f"| {seed} | {delta_edge:.6f} | {delta_control:.6f} | "
                f"{'yes' if (delta_edge > 0 and delta_control > 0) else 'no'} |"
            )
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "FORMAL_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return {"markdown": text, "rows": rows, "paired": paired, "official_test_loaded": False}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("params")
    p = sub.add_parser("preprocess")
    p.add_argument("--force", action="store_true")
    sub.add_parser("audit")
    sub.add_parser("sanity")
    sub.add_parser("synthetic")
    sub.add_parser("gate")
    sub.add_parser("decide")
    sub.add_parser("report")
    p = sub.add_parser("analyze")
    p.add_argument("--repeats", type=int, default=5000)
    p = sub.add_parser("gradient_audit")
    p.add_argument("--variant", default="BVE", choices=list(edge.MODELS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action="store_true")
    for stage in ("train", "soup", "run", "diagnostics", "witness", "shuffle_eval"):
        p = sub.add_parser(stage)
        p.add_argument("--variant", required=True, choices=list(edge.MODELS))
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
        train(arguments.variant, arguments.seed, device=arguments.device, tag=arguments.tag, max_epochs=arguments.max_epochs)
    elif arguments.stage == "soup":
        soup(arguments.variant, arguments.seed, tag=arguments.tag)
    elif arguments.stage == "run":
        run(arguments.variant, arguments.seed, device=arguments.device, tag=arguments.tag, max_epochs=arguments.max_epochs)
    elif arguments.stage == "diagnostics":
        diagnostics(arguments.variant, arguments.seed, tag=arguments.tag, state=arguments.state)
    elif arguments.stage in {"witness", "shuffle_eval"}:
        witness(arguments.variant, arguments.seed, tag=arguments.tag, state=arguments.state)
    elif arguments.stage == "gate":
        gate()
    elif arguments.stage == "decide":
        decide()
    elif arguments.stage == "report":
        report()
    elif arguments.stage == "analyze":
        analyze(repeats=int(arguments.repeats))
    else:  # pragma: no cover
        raise SystemExit(f"unknown stage {arguments.stage!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
