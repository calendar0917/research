"""FSAR-v2 ZINC runner: exact attribute marginals + explicit structural basis.

Stages
------
``schema_audit collision_audit params preprocess sanity smoke run train soup
diagnostics witness interventions gate decide report``

The frozen optimized compact-v4 protocol is inherited verbatim (Adam, lr 1e-3,
weight decay 1e-5, batch 128, grad clip 5.0, max 240 epochs, patience 40, no
scheduler, single stage, checkpoint = best official-valid MAE, fixed equal-weight
Top-5 soup, deterministic algorithms when requested).

Only the ZINC regression L1 / MAE loss is optimized.  Channel zeroing and the
fixed-centre attribute shuffle are **evaluation-only**.  Official ZINC test is
never loaded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import fsar
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_variance_diagnosis as vd
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_fsar as v1_runner
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_shared_structural_patch_encoder as sspe
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/fsar_v2"
CACHE_DIR = RESULTS_DIR / "cache"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SOUP_DIR = RESULTS_DIR / "soup_states"
RUNS_DIR = RESULTS_DIR / "runs"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PROTOCOL_VERSION = "fsar_v2"
CACHE_SCHEMA = "fsar_v2"

# --- frozen FSAR-v1 references (never retrained / used as input) ------------
FSAR_V1_A = 0.157917
FSAR_V1_SA = 0.151617
FSAR_V1_SAB = 0.130913
FSAR_V1_SAM = 0.134182
FSAR_V1_ALIGNED = FSAR_V1_SAM - FSAR_V1_SAB  # 0.003269

REFERENCE_BBAG_SOUP_PER_SEED = {0: 0.12738177864899625, 1: 0.12022926583880325}
REFERENCE_BFULL_SOUP_PER_SEED = {0: 0.11981802638241788, 1: 0.11812638340244302}
REFERENCE_A2_SOUP_PER_SEED = {0: 0.12169362585240742, 1: 0.1221344729354023}

# --- pre-registered FSAR-v2 gates -------------------------------------------
A_STRONG = 0.135
A_PARTIAL = 0.145
A_IMPROVEMENT_FLOOR = 0.010
ALIGNED_GATE = 0.001
EXPLICIT_PERFORMANCE_REFERENCE = 0.132  # SAB_exact <= 0.132 opens the gate
EXPLICIT_PARITY_BAND = 0.002
EXPLICIT_MODERATE_BAND = 0.005
MECHANISM_FLOOR = 1.0e-4

DIAG_BATCHES = 128
WITNESS_BATCHES = 64
SHUFFLE_REPEATS = 3
INTERVENTION_REPEATS = 3
COLLISION_AUDIT_MOLECULES = 200

_write_json = sspe._write_json
_read_json = sspe._read_json
_git_commit = sspe._git_commit
_environment_fingerprint = sspe._environment_fingerprint
_set_deterministic = v1_runner._set_deterministic
_shuffle_attributes = v1_runner._shuffle_attributes


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_model(mode: str, seed: int = 0) -> v2.PatchPathFSARV2Model:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    return v2.PatchPathFSARV2Model(mode=str(mode))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "fsar_v2_train.pkl.gz",
        CACHE_DIR / "fsar_v2_valid.pkl.gz",
        CACHE_DIR / "fsar_v2_cache_meta.json",
    )


def build_datasets(force: bool = False) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train_path, valid_path, meta_path = _cache_paths()
    if not force and train_path.exists() and valid_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema") == CACHE_SCHEMA:
            with gzip.open(train_path, "rb") as handle:
                train_data = pickle.load(handle)
            with gzip.open(valid_path, "rb") as handle:
                valid_data = pickle.load(handle)
            return list(train_data), list(valid_data), meta

    started = time.perf_counter()
    train_bundle, valid_bundle, base_meta = sspe.extract_records()
    _assert_processed_ready("train")
    _assert_processed_ready("val")
    train_ds = zpp._load_zinc(ZINC_ROOT, "train")
    valid_ds = zpp._load_zinc(ZINC_ROOT, "val")
    topo_train = ztopo.matrices_for_split("train", train_ds, fsar.TOPOLOGY_MODE)[0]
    topo_valid = ztopo.matrices_for_split("valid", valid_ds, fsar.TOPOLOGY_MODE)[0]
    if len(topo_train) != len(train_bundle["graphs"]):
        raise RuntimeError("train topology / patch graph count mismatch")
    if len(topo_valid) != len(valid_bundle["graphs"]):
        raise RuntimeError("valid topology / patch graph count mismatch")

    topology_mean = topo_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    topology_scale = topo_train.std(axis=0, dtype=np.float64).astype(np.float32)
    topology_scale[~np.isfinite(topology_scale) | (topology_scale < 1.0e-6)] = 1.0

    train_data, _ = v2.build_fsar_v2_dataset(
        train_ds,
        train_bundle["graphs"],
        topo_train,
        topology_mean=topology_mean,
        topology_scale=topology_scale,
    )
    valid_data, _ = v2.build_fsar_v2_dataset(
        valid_ds,
        valid_bundle["graphs"],
        topo_valid,
        topology_mean=topology_mean,
        topology_scale=topology_scale,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(train_path, "wb") as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(valid_path, "wb") as handle:
        pickle.dump(valid_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "cache_schema": CACHE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "n_train": int(len(train_data)),
        "n_valid": int(len(valid_data)),
        "a_raw_dim": int(v2.A_RAW_DIM),
        "node_basis_dim": int(v2.NODE_BASIS_DIM),
        "edge_basis_dim": int(v2.EDGE_BASIS_DIM),
        "relation_width": int(fsar.RELATION_WIDTH),
        "topology_width": int(fsar.TOPOLOGY_WIDTH),
        "patch_radius": int(fsar.PATCH_RADIUS),
        "mean_patches_per_molecule": float(
            np.mean([int(row.num_nodes) for row in train_data])
        ),
        "base_meta": {
            "train": base_meta.get("train"),
            "valid": base_meta.get("valid"),
        },
        "seconds": float(time.perf_counter() - started),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    _write_json(meta_path, meta)
    print(
        f"[preprocess] train={len(train_data)} valid={len(valid_data)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_data, valid_data, meta


def preprocess(force: bool = False) -> dict[str, Any]:
    _train, _valid, meta = build_datasets(force=force)
    return meta


# ---------------------------------------------------------------------------
# schema / collision audits (brief sections 3-6)
# ---------------------------------------------------------------------------


def _assert_processed_ready(split: str) -> None:
    """Refuse to trigger PyG ``process()`` (which reads every split incl. test)."""
    processed = ZINC_ROOT / "subset" / "processed"
    path = processed / f"{split}.pt"
    if not path.exists():
        raise RuntimeError(
            f"processed ZINC split missing: {path}; refuse to run process() "
            "because it would read the official test split"
        )


def schema_audit() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for split, name in (("train", "train"), ("val", "valid")):
        _assert_processed_ready(split)
        dataset = zpp._load_zinc(ZINC_ROOT, split)
        atoms: list[torch.Tensor] = []
        bonds: list[torch.Tensor] = []
        for data in dataset:
            atoms.append(data.x.detach().cpu().long().reshape(-1))
            bonds.append(data.edge_attr.detach().cpu().long().reshape(-1))
        atom_values = torch.cat(atoms)
        bond_values = torch.cat(bonds)
        rows[name] = {
            "n_molecules": int(len(dataset)),
            "x_shape_per_molecule": list(dataset[0].x.shape),
            "edge_attr_shape_per_molecule": list(dataset[0].edge_attr.shape),
            "x_is_single_categorical_field": bool(dataset[0].x.dim() == 2 and dataset[0].x.shape[1] == 1),
            "atom_categories_observed": sorted(int(v) for v in torch.unique(atom_values)),
            "bond_categories_observed": sorted(int(v) for v in torch.unique(bond_values)),
            "atom_category_min": int(atom_values.min()),
            "atom_category_max": int(atom_values.max()),
            "bond_category_min": int(bond_values.min()),
            "bond_category_max": int(bond_values.max()),
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "encoder_schema": {
            "atom_categories": int(v2.ATOM_CATEGORIES),
            "bond_categories": int(v2.BOND_CATEGORIES),
            "source": "inherited FSAR-v1 / patch-path schema (zpp.ATOM_CATEGORIES, zpp.BOND_CATEGORIES)",
        },
        "splits": rows,
        "atom_schema_covers_observed": bool(
            rows["train"]["atom_category_max"] < v2.ATOM_CATEGORIES
            and rows["valid"]["atom_category_max"] < v2.ATOM_CATEGORIES
        ),
        "bond_schema_covers_observed": bool(
            rows["train"]["bond_category_max"] < v2.BOND_CATEGORIES
            and rows["valid"]["bond_category_max"] < v2.BOND_CATEGORIES
        ),
        "single_categorical_field": bool(
            rows["train"]["x_is_single_categorical_field"]
            and rows["valid"]["x_is_single_categorical_field"]
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "schema_audit.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def _molecule_marginal_rows(data: Any) -> tuple[torch.Tensor, list[Any]]:
    n_patches = int(data.num_nodes)
    raw = v2.raw_marginal_vector(
        data.struct_atom,
        data.struct_root,
        data.struct_patch,
        data.struct_bond,
        data.struct_edge_patch,
        n_patches,
    )
    keys = v2.marginal_key(
        data.struct_atom.numpy(),
        data.struct_root.numpy(),
        data.struct_patch.numpy(),
        data.struct_bond.numpy(),
        data.struct_edge_patch.numpy(),
        n_patches,
    )
    return raw, keys


def collision_audit() -> dict[str, Any]:
    """Verify the raw exact marginal is collision-free for the defined marginal."""
    train_data, valid_data, _meta = build_datasets()
    rng = np.random.default_rng(20260917)
    pool = list(train_data) + list(valid_data)
    indices = rng.choice(len(pool), size=min(COLLISION_AUDIT_MOLECULES, len(pool)), replace=False)
    key_to_raw: dict[Any, tuple[float, ...]] = {}
    collisions = 0
    n_frames = 0
    same_key_raw_mismatch = 0
    for index in indices:
        raw, keys = _molecule_marginal_rows(pool[int(index)])
        for row, key in zip(raw.tolist(), keys):
            n_frames += 1
            fingerprint = tuple(float(value) for value in row)
            if key in key_to_raw:
                if key_to_raw[key] != fingerprint:
                    same_key_raw_mismatch += 1
            else:
                key_to_raw[key] = fingerprint
    # distinct marginal keys must map to distinct raw vectors
    raw_values = list(key_to_raw.values())
    collisions = len(raw_values) - len(set(raw_values))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_molecules_sampled": int(len(indices)),
        "n_frames": int(n_frames),
        "n_unique_marginal_keys": int(len(key_to_raw)),
        "n_unique_raw_vectors": int(len(set(raw_values))),
        "raw_collisions": int(collisions),
        "same_key_raw_mismatch": int(same_key_raw_mismatch),
        "lossless_for_defined_marginal": bool(
            collisions == 0 and same_key_raw_mismatch == 0
        ),
        "raw_dim": int(v2.A_RAW_DIM),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "collision_audit.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# parameter accounting / sanity
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "modes": {},
    }
    for mode in v2.ALL_MODES:
        payload["modes"][mode] = v2.parameter_breakdown(build_model(mode, seed=0))
    totals = {mode: row["total"] for mode, row in payload["modes"].items()}
    payload["nested_totals"] = totals
    payload["nested_ordering"] = bool(
        totals["A"] < totals["SA"] < totals["SAB"]
    )
    payload["budget_ok"] = bool(max(totals.values()) <= 200_000)
    payload["capacity_control_gap_implicit"] = int(totals["SAM"] - totals["SAB"])
    payload["capacity_control_gap_explicit"] = int(totals["SAME"] - totals["SABE"])
    payload["dataset_dependent_vocabulary_params"] = 0
    payload["references"] = {
        "FSAR_v1_A": FSAR_V1_A,
        "FSAR_v1_SA": FSAR_V1_SA,
        "FSAR_v1_SAB": FSAR_V1_SAB,
        "FSAR_v1_SAM": FSAR_V1_SAM,
        "B-Bag_per_seed": REFERENCE_BBAG_SOUP_PER_SEED,
        "B-Full_per_seed": REFERENCE_BFULL_SOUP_PER_SEED,
        "A2_per_seed": REFERENCE_A2_SOUP_PER_SEED,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def sanity() -> dict[str, Any]:
    from tracks.ksvd.tests import test_fsar_v2 as suite

    checks: dict[str, Any] = {}
    failures: list[str] = []

    def run(name: str, function) -> None:
        try:
            function()
            checks[name] = True
        except Exception as error:  # noqa: BLE001
            checks[name] = False
            failures.append(f"{name}: {error}")

    run(
        "exact_marginal_assignment_invariance_and_collision_freedom",
        suite.test_exact_marginal_assignment_invariance_and_collision_freedom,
    )
    run(
        "a_exact_reconstructs_full_attribute_multiset",
        suite.test_a_exact_reconstructs_full_attribute_multiset,
    )
    run("a_exact_is_blind_to_topology_fields", suite.test_a_exact_is_blind_to_topology_fields)
    run("explicit_basis_chemistry_purity", suite.test_explicit_basis_chemistry_purity)
    run("explicit_basis_relabel_equivariance", suite.test_explicit_basis_relabel_equivariance)
    run("explicit_path_has_no_message_passing", suite.test_explicit_path_has_no_message_passing)
    run("assignment_moves_b_only", suite.test_assignment_moves_b_only)
    run("no_mixed_bypass_v2", suite.test_no_mixed_bypass_v2)
    run("node_relabel_invariance_v2", suite.test_node_relabel_invariance_v2)
    run("batch_invariance_v2", suite.test_batch_invariance_v2)
    run("gradient_viability_v2", suite.test_gradient_viability_v2)
    run("parameter_accounting_v2", suite.test_parameter_accounting_v2)

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
        raise RuntimeError(f"FSAR-v2 sanity gates failed: {failures}")
    return payload


# ---------------------------------------------------------------------------
# GPU smoke
# ---------------------------------------------------------------------------


def smoke(device: str = "cpu", steps: int = 5, mode: str = "SAB") -> dict[str, Any]:
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _train, valid_data, _meta = build_datasets()
    model = build_model(mode, seed=0).to(device_obj)
    batch = next(iter(_loader(list(valid_data)[:64], 64, False, 0))).to(device_obj)
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    losses: list[float] = []
    finite = True
    model.train()
    for _ in range(int(steps)):
        prediction = model(batch).view(-1)
        loss = F.l1_loss(prediction, batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        finite = finite and bool(torch.isfinite(loss).all())
    model.eval()
    with torch.no_grad():
        channels = model.encoder.forward_channels(batch)
        prediction = model(batch)
    cross_mol = {"A": float(channels["attributes"].std(dim=0, unbiased=False).mean())}
    if "structure" in channels:
        cross_mol["S"] = float(channels["structure"].std(dim=0, unbiased=False).mean())
    if "binding" in channels:
        cross_mol["B"] = float(channels["binding"].std(dim=0, unbiased=False).mean())
    grads = {}
    model.train()
    loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    grads = {
        "A": float(model.encoder.a_exact_mlp[0].weight.grad.abs().sum()),
        "relation": float(model.relation_encoder[0].weight.grad.abs().sum()),
        "head": float(model.head[-1].weight.grad.abs().sum()),
    }
    if model.encoder.include_structure:
        grads["S"] = float(model.encoder.structure_pool[0].weight.grad.abs().sum())
    if model.encoder.include_binding:
        grads["B"] = float(
            model.encoder.node_role_projection.weight.grad.abs().sum()
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "structure_kind": str(model.encoder.structure_kind),
        "device": str(device),
        "steps": int(steps),
        "losses": losses,
        "loss_decreasing": bool(losses[-1] <= losses[0]),
        "loss_finite": bool(finite),
        "output_not_constant": bool(prediction.std() > 1e-8),
        "output_std": float(prediction.std()),
        "cross_molecule_channel_std": cross_mol,
        "channels_alive": bool(all(value > 1e-8 for value in cross_mol.values())),
        "grad_abs_sum": grads,
        "grads_alive": bool(all(value > 0.0 for value in grads.values())),
        "parameters": _n_params(model),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device_obj.type == "cuda"
            else 0
        ),
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(str(device)),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"smoke_{mode}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# loaders / evaluation
# ---------------------------------------------------------------------------


def _loader(graphs: Sequence[Any], batch_size: int, shuffle: bool, seed: int):
    return v2.make_fsar_v2_loader(graphs, batch_size, shuffle, seed)


def _evaluate_mae(model: nn.Module, loader, device: torch.device) -> float:
    return sspe._evaluate_mae(model, loader, device)


def _selection_loader(valid_data: Sequence[Any]):
    return _loader(valid_data, 128, False, 0)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(
    mode: str,
    seed: int,
    device: str = "cpu",
    tag: str | None = None,
    protocol_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tag = str(tag or f"fsar_v2_{mode.lower()}")
    train_data, valid_data, _meta = build_datasets()
    protocol = dict(shead.OPTIMIZED_PROTOCOL)
    if protocol_override:
        protocol.update(protocol_override)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats()
    model = build_model(mode, seed).to(device_obj)
    total_params = _n_params(model)

    diag_batches = [
        batch.to(device_obj)
        for batch in _loader(list(valid_data)[:DIAG_BATCHES], 128, False, 0)
    ]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    loader = _loader(
        train_data,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = _loader(
        valid_data,
        int(protocol["batch_size"]),
        False,
        int(seed) + int(protocol["eval_shuffle_seed_offset"]),
    )
    steps_per_epoch = int(math.ceil(len(train_data) / int(protocol["batch_size"])))
    patience = int(protocol["patience"])
    max_epochs = int(protocol["max_epochs"])

    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    top5: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(protocol["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae = _evaluate_mae(model, eval_loader, device_obj)
        stats = _channel_curve_stats(model, diag_batches)
        grads = _encoder_grad_norms(model)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(train_mae),
                "valid_mae": float(valid_mae),
                **stats,
                **grads,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps": int(epoch * steps_per_epoch),
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
        if epoch == 1 or epoch % 20 == 0 or epoch == max_epochs:
            print(
                f"[{tag} seed{seed} mode={mode}] epoch={epoch:03d} "
                f"train={train_mae:.6f} valid={valid_mae:.6f} "
                f"best={best_mae:.6f}@{best_epoch} "
                f"stdA={stats.get('A_cross_mol_std', 0.0):.3e} "
                f"stdS={stats.get('S_cross_mol_std', 0.0):.3e} "
                f"stdB={stats.get('B_cross_mol_std', 0.0):.3e}",
                flush=True,
            )
        if stale >= patience:
            print(
                f"[{tag} seed{seed}] early_stop epoch={epoch} best={best_epoch}",
                flush=True,
            )
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
        "mode": str(mode),
        "structure_kind": str(model.encoder.structure_kind),
        "seed": int(seed),
        "protocol": protocol,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(losses)),
        "steps_per_epoch": int(steps_per_epoch),
        "wall_clock_s": wall_clock,
        "epoch_time_s": float(wall_clock) / max(len(losses), 1),
        "early_stopped": bool(len(losses) < max_epochs),
        "parameters": int(total_params),
        "parameter_breakdown": v2.parameter_breakdown(model),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device_obj.type == "cuda"
            else 0
        ),
        "state_path": str(selection_path),
        "curve_path": str(curve_path),
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(str(device)),
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


def _channel_curve_stats(
    model: v2.PatchPathFSARV2Model, batches: Sequence[Any]
) -> dict[str, float]:
    encoder = model.encoder
    values: dict[str, list[torch.Tensor]] = {"A": [], "S": [], "B": []}
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        for batch in batches:
            channels = encoder.forward_channels(batch)
            values["A"].append(channels["attributes"].detach())
            if "structure" in channels:
                values["S"].append(channels["structure"].detach())
            if "binding" in channels:
                values["B"].append(channels["binding"].detach())
    encoder.train(was_training)
    stats: dict[str, float] = {}
    for name, blocks in values.items():
        if not blocks:
            stats[f"{name}_norm_mean"] = 0.0
            stats[f"{name}_norm_std"] = 0.0
            stats[f"{name}_cross_mol_std"] = 0.0
            continue
        matrix = torch.cat(blocks, dim=0)
        norms = matrix.norm(dim=1)
        stats[f"{name}_norm_mean"] = float(norms.mean())
        stats[f"{name}_norm_std"] = float(norms.std(unbiased=False))
        stats[f"{name}_cross_mol_std"] = float(
            matrix.std(dim=0, unbiased=False).mean()
        )
    return stats


def _encoder_grad_norms(model: v2.PatchPathFSARV2Model) -> dict[str, float]:
    def _norm(parameters) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        return float(math.sqrt(total))

    encoder = model.encoder
    s_params = []
    if encoder.structure_pool is not None:
        s_params.extend(encoder.structure_pool.parameters())
    if encoder.message is not None:
        s_params.extend(encoder.message.parameters())
    if encoder.update is not None:
        s_params.extend(encoder.update.parameters())
    return {
        "grad_A": _norm(
            list(encoder.atom_embedding.parameters())
            + list(encoder.atom_mlp.parameters())
            + list(encoder.bond_mlp.parameters())
            + list(encoder.a_exact_mlp.parameters())
        ),
        "grad_S": _norm(s_params),
        "grad_B": _norm(
            [
                parameter
                for module in (
                    encoder.node_role_projection,
                    encoder.node_attribute_projection,
                    encoder.binding_fuse,
                )
                if module is not None
                for parameter in module.parameters()
            ]
        ),
        "grad_relation": _norm(model.relation_encoder.parameters()),
        "grad_head": _norm(model.head.parameters()),
    }


# ---------------------------------------------------------------------------
# soup
# ---------------------------------------------------------------------------


def soup(mode: str, seed: int, tag: str | None = None) -> dict[str, Any]:
    tag = str(tag or f"fsar_v2_{mode.lower()}")
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

    _train, valid_data, _ = build_datasets()
    loader = _selection_loader(valid_data)
    model = build_model(mode, seed)
    selection_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    targets, best_preds = vd._predict_state(
        model,
        torch.load(selection_path, map_location="cpu", weights_only=True),
        loader,
    )
    _t, soup_preds = vd._predict_state(model, soup_state, loader)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "structure_kind": str(model.encoder.structure_kind),
        "seed": int(seed),
        "top5_epochs": [int(row["epoch"]) for row in ranked],
        "top5_valid_mae": [float(row["valid_mae"]) for row in ranked],
        "best_checkpoint_valid_mae": vd._mae(targets, best_preds),
        "top5_soup_valid_mae": vd._mae(targets, soup_preds),
        "soup_improvement_over_best": vd._mae(targets, best_preds)
        - vd._mae(targets, soup_preds),
        "soup_state_path": str(soup_path),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def run(mode: str, seed: int, device: str = "cpu", tag: str | None = None) -> dict[str, Any]:
    """Train then soup one mode/seed (resumable: skips completed stages)."""
    tag = str(tag or f"fsar_v2_{mode.lower()}")
    state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    if state_path.exists():
        print(f"[run] selection state exists, skipping train: {state_path}", flush=True)
    else:
        train(mode, seed, device=device, tag=tag)
    soup_path = SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
    soup_json = RESULTS_DIR / f"soup_{tag}_seed{seed}.json"
    if soup_json.exists():
        print(f"[run] soup json exists, skipping soup: {soup_json}", flush=True)
        return _read_json(soup_json)
    return soup(mode, seed, tag=tag)


# ---------------------------------------------------------------------------
# diagnostics / witness / interventions (frozen checkpoint, eval only)
# ---------------------------------------------------------------------------


def _load_frozen(mode: str, seed: int, tag: str | None = None, state: str = "best"):
    tag = str(tag or f"fsar_v2_{mode.lower()}")
    model = build_model(mode, seed)
    if str(state) == "soup":
        state_path = SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
    elif str(state) == "best":
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    else:
        raise ValueError(f"unknown frozen state {state!r}; expected 'best' or 'soup'")
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _channel_matrix(model: v2.PatchPathFSARV2Model, batches, channel: str) -> np.ndarray:
    blocks = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            channels = model.encoder.forward_channels(batch)
            if channel in channels:
                blocks.append(channels[channel].cpu().numpy())
    model.train(was_training)
    if not blocks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(blocks, axis=0)


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


def _basis_block_diagnostics(model: v2.PatchPathFSARV2Model) -> dict[str, Any]:
    """Raw / projected explicit-basis diagnostics (brief section 32)."""
    encoder = model.encoder
    if encoder.structure_kind != "explicit":
        return {}
    weight = encoder.structure_pool[0].weight.detach()
    blocks = {
        "root": (0, v2.NODE_BASIS_DIM),
        "node_mean": (v2.NODE_BASIS_DIM, 2 * v2.NODE_BASIS_DIM),
        "node_std": (2 * v2.NODE_BASIS_DIM, 3 * v2.NODE_BASIS_DIM),
        "edge_mean": (3 * v2.NODE_BASIS_DIM, 3 * v2.NODE_BASIS_DIM + v2.EDGE_BASIS_DIM),
        "edge_std": (
            3 * v2.NODE_BASIS_DIM + v2.EDGE_BASIS_DIM,
            3 * v2.NODE_BASIS_DIM + 2 * v2.EDGE_BASIS_DIM,
        ),
    }
    projected = {
        name: float(weight[:, start:stop].norm()) for name, (start, stop) in blocks.items()
    }
    return {
        "structure_pool_input_dim": int(weight.shape[1]),
        "projection_norms": projected,
        "projection_norm_total": float(weight.norm()),
    }


def diagnostics(
    mode: str = "SAB", seed: int = 0, tag: str | None = None, state: str = "best"
) -> dict[str, Any]:
    model = _load_frozen(mode, seed, tag, state=state)
    _train, valid_data, _ = build_datasets()
    batches = list(_loader(list(valid_data)[:DIAG_BATCHES], 128, False, 0))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "structure_kind": str(model.encoder.structure_kind),
        "seed": int(seed),
        "channels": {},
        "node_init_contribution": {},
        "explicit_basis": _basis_block_diagnostics(model),
        "official_test_loaded": False,
    }
    for name, key in (("A", "attributes"), ("S", "structure"), ("B", "binding")):
        matrix = _channel_matrix(model, batches, key)
        stats = _effective_rank(matrix)
        payload["channels"][name] = {
            "dim": int(matrix.shape[1]) if matrix.size else 0,
            "norm_mean": float(np.linalg.norm(matrix, axis=1).mean()) if matrix.size else 0.0,
            "cross_mol_std": float(matrix.std(axis=0).mean()) if matrix.size else 0.0,
            **stats,
        }
    if model.encoder.structure_kind == "explicit":
        node_basis = _channel_matrix(model, batches, "node_basis")
        edge_basis = _channel_matrix(model, batches, "edge_basis")
        payload["raw_basis_distribution"] = {
            "node_mean": node_basis.mean(axis=0).tolist() if node_basis.size else [],
            "node_std": node_basis.std(axis=0).tolist() if node_basis.size else [],
            "node_nonzero_fraction": (
                float((node_basis != 0).mean()) if node_basis.size else 0.0
            ),
            "edge_mean": edge_basis.mean(axis=0).tolist() if edge_basis.size else [],
            "edge_std": edge_basis.std(axis=0).tolist() if edge_basis.size else [],
            "edge_nonzero_fraction": (
                float((edge_basis != 0).mean()) if edge_basis.size else 0.0
            ),
        }
    with torch.no_grad():
        batch = batches[0]
        init = model.node_init
        z = model.encoder(batch)
        offsets = {"A": (0, v2.A_DIM)}
        if model.encoder.include_structure:
            offsets["S"] = (v2.A_DIM, v2.A_DIM + v2.S_DIM)
        if model.encoder.include_binding:
            offsets["B"] = (
                v2.A_DIM + v2.S_DIM,
                v2.A_DIM + v2.S_DIM + v2.B_DIM,
            )
        for name, (start, stop) in offsets.items():
            block = z[:, start:stop]
            weight = init[0].weight[:, start:stop]
            payload["node_init_contribution"][name] = float(
                (block @ weight.t()).norm(dim=1).mean()
            )
    model.train()
    loss = F.l1_loss(model(batches[0]).view(-1), batches[0].y.view(-1))
    model.zero_grad()
    loss.backward()
    payload["grad_norms"] = _encoder_grad_norms(model)
    model.zero_grad()
    _write_json(
        RESULTS_DIR
        / (
            f"diagnostics_{tag or f'fsar_v2_{mode.lower()}'}_seed{seed}"
            + ("_soup" if str(state) == "soup" else "")
            + ".json"
        ),
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def witness(
    mode: str = "SAB", seed: int = 0, tag: str | None = None, state: str = "best"
) -> dict[str, Any]:
    model = _load_frozen(mode, seed, tag, state=state)
    _train, valid_data, _ = build_datasets()
    batches = list(_loader(list(valid_data)[:WITNESS_BATCHES], 128, False, 0))
    channel_deltas: dict[str, list[float]] = {"A": [], "S": [], "B": []}
    prediction_deltas: list[float] = []
    with torch.no_grad():
        for batch in batches:
            original = model.encoder.forward_channels(batch)
            for repeat in range(SHUFFLE_REPEATS):
                shuffled_atom, shuffled_bond = _shuffle_attributes(batch, repeat)
                view = batch.clone()
                view.struct_atom = shuffled_atom
                view.struct_bond = shuffled_bond
                other = model.encoder.forward_channels(view)
                channel_deltas["A"].append(
                    float((original["attributes"] - other["attributes"]).abs().mean())
                )
                if "structure" in original:
                    channel_deltas["S"].append(
                        float((original["structure"] - other["structure"]).abs().mean())
                    )
                if "binding" in original:
                    channel_deltas["B"].append(
                        float((original["binding"] - other["binding"]).abs().mean())
                    )
                prediction_deltas.append(float((model(batch) - model(view)).abs().mean()))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "structure_kind": str(model.encoder.structure_kind),
        "seed": int(seed),
        "channel_mean_abs_delta": {
            key: float(np.mean(values)) if values else 0.0
            for key, values in channel_deltas.items()
        },
        "prediction_mean_abs_delta": float(np.mean(prediction_deltas)),
        "A_stable": bool(np.mean(channel_deltas["A"]) <= MECHANISM_FLOOR),
        "S_stable": bool(
            (not channel_deltas["S"])
            or np.mean(channel_deltas["S"]) <= MECHANISM_FLOOR
        ),
        "B_responds": bool(
            bool(channel_deltas["B"]) and np.mean(channel_deltas["B"]) > MECHANISM_FLOOR
        ),
        "official_test_loaded": False,
    }
    _write_json(
        RESULTS_DIR
        / (
            f"witness_{tag or f'fsar_v2_{mode.lower()}'}_seed{seed}"
            + ("_soup" if str(state) == "soup" else "")
            + ".json"
        ),
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def interventions(
    mode: str = "SAB", seed: int = 0, tag: str | None = None, state: str = "best"
) -> dict[str, Any]:
    model = _load_frozen(mode, seed, tag, state=state)
    _train, valid_data, _ = build_datasets()
    loader = _selection_loader(valid_data)
    rows: dict[str, float] = {}
    with torch.no_grad():
        model.encoder.set_intervention(None)
        rows["true"] = _evaluate_mae(model, loader, torch.device("cpu"))
        for key in ("A", "S", "B"):
            if key == "S" and not model.encoder.include_structure:
                continue
            if key == "B" and not model.encoder.include_binding:
                continue
            model.encoder.set_intervention([key])
            rows[f"no_{key}"] = _evaluate_mae(model, loader, torch.device("cpu"))
        model.encoder.set_intervention(None)
        shuffled_maes = []
        for repeat in range(INTERVENTION_REPEATS):
            total = 0.0
            seen = 0
            for batch in loader:
                shuffled_atom, shuffled_bond = _shuffle_attributes(batch, repeat)
                view = batch.clone()
                view.struct_atom = shuffled_atom
                view.struct_bond = shuffled_bond
                prediction = model(view).view(-1)
                target = view.y.view(-1)
                total += float((prediction - target).abs().sum())
                seen += int(target.numel())
            shuffled_maes.append(float(total / max(seen, 1)))
        rows["shuffle_attributes"] = float(np.mean(shuffled_maes))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "structure_kind": str(model.encoder.structure_kind),
        "seed": int(seed),
        "valid_mae": rows,
        "delta_no_S": (
            None if "no_S" not in rows else rows["no_S"] - rows["true"]
        ),
        "delta_no_A": (
            None if "no_A" not in rows else rows["no_A"] - rows["true"]
        ),
        "delta_no_B": (
            None if "no_B" not in rows else rows["no_B"] - rows["true"]
        ),
        "delta_shuffle_attributes": rows["shuffle_attributes"] - rows["true"],
        "official_test_loaded": False,
    }
    _write_json(
        RESULTS_DIR
        / (
            f"interventions_{tag or f'fsar_v2_{mode.lower()}'}_seed{seed}"
            + ("_soup" if str(state) == "soup" else "")
            + ".json"
        ),
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# gate / decision
# ---------------------------------------------------------------------------


def _soup_valid(mode: str, seed: int) -> float | None:
    tag = f"fsar_v2_{mode.lower()}"
    path = RESULTS_DIR / f"soup_{tag}_seed{seed}.json"
    if not path.exists():
        return None
    return float(_read_json(path)["top5_soup_valid_mae"])


def gate() -> dict[str, Any]:
    a = _soup_valid("A", 0)
    sab = _soup_valid("SAB", 0)
    sam = _soup_valid("SAM", 0)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "soups": {"A": a, "SAB": sab, "SAM": sam},
        "a_exact": a,
        "delta_A_vs_fsar_v1": (None if a is None else a - FSAR_V1_A),
        "aligned_residual_implicit": (None if (sam is None or sab is None) else sam - sab),
        "official_test_loaded": False,
    }
    condition_a = a is not None and a <= A_PARTIAL
    condition_b = (sab is not None and sab <= EXPLICIT_PERFORMANCE_REFERENCE) or (
        payload["aligned_residual_implicit"] is not None
        and payload["aligned_residual_implicit"] >= ALIGNED_GATE
    )
    payload["condition_A_exact_le_0.145"] = bool(condition_a)
    payload["condition_SAB_or_aligned"] = bool(condition_b)
    payload["explicit_stage_open"] = bool(condition_a and condition_b)
    _write_json(RESULTS_DIR / "explicit_gate.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def decide() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "soups_seed0": {},
        "deltas_seed0": {},
        "status": "INCOMPLETE",
        "case": None,
        "official_test_loaded": False,
    }
    for mode in v2.ALL_MODES:
        payload["soups_seed0"][mode] = _soup_valid(mode, 0)
    a = payload["soups_seed0"]["A"]
    sa = payload["soups_seed0"]["SA"]
    sab = payload["soups_seed0"]["SAB"]
    sam = payload["soups_seed0"]["SAM"]
    sae = payload["soups_seed0"]["SAE"]
    sabe = payload["soups_seed0"]["SABE"]
    same = payload["soups_seed0"]["SAME"]

    if a is not None:
        payload["delta_A_vs_fsar_v1"] = float(a - FSAR_V1_A)
        payload["A_improvement"] = float(FSAR_V1_A - a)
        if a <= A_STRONG:
            payload["a_class"] = "strong_success"
        elif a <= A_PARTIAL and (FSAR_V1_A - a) >= A_IMPROVEMENT_FLOOR:
            payload["a_class"] = "partial_success"
        else:
            payload["a_class"] = "failure"
    if sa is not None and sab is not None:
        payload["deltas_seed0"]["SA_to_SAB"] = sa - sab
    if sab is not None and sam is not None:
        payload["deltas_seed0"]["SAM_minus_SAB"] = sam - sab
        payload["aligned_increment_implicit"] = float(sam - sab)
        payload["capacity_confound_excluded_implicit"] = bool(
            (sam - sab) >= ALIGNED_GATE
        )
    if sae is not None and sabe is not None:
        payload["deltas_seed0"]["SAE_to_SABE"] = sae - sabe
    if sabe is not None and same is not None:
        payload["deltas_seed0"]["SAME_minus_SABE"] = same - sabe
        payload["aligned_increment_explicit"] = float(same - sabe)
        payload["capacity_confound_excluded_explicit"] = bool(
            (same - sabe) >= ALIGNED_GATE
        )
    if sabe is not None and sab is not None:
        payload["explicit_vs_implicit_gap"] = float(sabe - sab)
        if sabe <= sab + EXPLICIT_PARITY_BAND:
            payload["explicit_class"] = "parity"
        elif sabe <= sab + EXPLICIT_MODERATE_BAND:
            payload["explicit_class"] = "moderate_cost"
        else:
            payload["explicit_class"] = "failure"

    gate_payload = gate()
    payload["explicit_stage_open"] = bool(gate_payload.get("explicit_stage_open"))

    # mechanism evidence for the implicit aligned organ
    def _maybe_read(path: Path) -> Any:
        return _read_json(path) if path.exists() else None

    witness = _maybe_read(RESULTS_DIR / "witness_fsar_v2_sab_seed0.json")
    interventions = _maybe_read(RESULTS_DIR / "interventions_fsar_v2_sab_seed0.json")
    mechanism = {
        "witness_available": witness is not None,
        "interventions_available": interventions is not None,
    }
    if witness is not None:
        mechanism["A_stable"] = bool(witness.get("A_stable"))
        mechanism["S_stable"] = bool(witness.get("S_stable"))
        mechanism["B_responds"] = bool(witness.get("B_responds"))
    if interventions is not None:
        mechanism["delta_no_B"] = interventions.get("delta_no_B")
        mechanism["delta_shuffle_attributes"] = interventions.get(
            "delta_shuffle_attributes"
        )
    mechanism_supported = bool(
        witness is not None
        and interventions is not None
        and mechanism.get("A_stable")
        and mechanism.get("S_stable")
        and mechanism.get("B_responds")
        and float(mechanism.get("delta_no_B") or 0.0) > 0.0
        and float(mechanism.get("delta_shuffle_attributes") or 0.0) > 0.0
    )
    payload["mechanism"] = mechanism
    payload["mechanism_supported"] = mechanism_supported

    payload["seed1_authorized_implicit"] = bool(
        a is not None
        and a <= A_PARTIAL
        and sab is not None
        and sab <= EXPLICIT_PERFORMANCE_REFERENCE
        and sam is not None
        and (sam - sab) >= ALIGNED_GATE
        and mechanism_supported
    )

    if payload.get("a_class") == "failure":
        payload["status"] = "STOP_A_FAILED"
        payload["case"] = "Case_D_A_exact_still_poor"
    elif payload.get("a_class") in {"strong_success", "partial_success"}:
        payload["status"] = "PHASE1_COMPLETE"
        if sabe is None:
            payload["case"] = "Case_A_or_B_phase1_implicit"
        else:
            payload["case"] = (
                f"phase1_{payload.get('a_class')}__explicit_{payload.get('explicit_class')}"
            )
    _write_json(RESULTS_DIR / "decision.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / f"{name}.json"
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "schema_audit": _maybe("schema_audit"),
        "collision_audit": _maybe("collision_audit"),
        "parameter_accounting": _maybe("parameter_accounting"),
        "sanity": _maybe("sanity"),
        "smoke_sab": _maybe("smoke_SAB"),
        "explicit_gate": _maybe("explicit_gate"),
        "decision": _maybe("decision"),
        "soups_seed0": {mode: _soup_valid(mode, 0) for mode in v2.ALL_MODES},
        "soups_seed1": {mode: _soup_valid(mode, 1) for mode in v2.ALL_MODES},
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FSAR-v2 ZINC runner")
    parser.add_argument(
        "stage",
        choices=[
            "schema_audit",
            "collision_audit",
            "params",
            "preprocess",
            "sanity",
            "smoke",
            "run",
            "train",
            "soup",
            "diagnostics",
            "witness",
            "interventions",
            "gate",
            "decide",
            "report",
        ],
    )
    parser.add_argument("--mode", default="SAB", choices=list(v2.ALL_MODES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--state", default="best", choices=["best", "soup"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    arguments = parser.parse_args(argv)
    _set_deterministic(bool(arguments.deterministic))

    if arguments.stage == "schema_audit":
        schema_audit()
    elif arguments.stage == "collision_audit":
        collision_audit()
    elif arguments.stage == "params":
        params()
    elif arguments.stage == "preprocess":
        preprocess(force=bool(arguments.force))
    elif arguments.stage == "sanity":
        sanity()
    elif arguments.stage == "smoke":
        smoke(device=arguments.device, steps=int(arguments.steps), mode=arguments.mode)
    elif arguments.stage == "run":
        run(arguments.mode, int(arguments.seed), device=arguments.device, tag=arguments.tag)
    elif arguments.stage == "train":
        override = {}
        if arguments.epochs is not None:
            override["max_epochs"] = int(arguments.epochs)
        if arguments.patience is not None:
            override["patience"] = int(arguments.patience)
        train(
            arguments.mode,
            int(arguments.seed),
            device=arguments.device,
            tag=arguments.tag,
            protocol_override=override or None,
        )
    elif arguments.stage == "soup":
        soup(arguments.mode, int(arguments.seed), tag=arguments.tag)
    elif arguments.stage == "diagnostics":
        diagnostics(
            mode=arguments.mode,
            seed=int(arguments.seed),
            tag=arguments.tag,
            state=arguments.state,
        )
    elif arguments.stage == "witness":
        witness(
            mode=arguments.mode,
            seed=int(arguments.seed),
            tag=arguments.tag,
            state=arguments.state,
        )
    elif arguments.stage == "interventions":
        interventions(
            mode=arguments.mode,
            seed=int(arguments.seed),
            tag=arguments.tag,
            state=arguments.state,
        )
    elif arguments.stage == "gate":
        gate()
    elif arguments.stage == "decide":
        decide()
    elif arguments.stage == "report":
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
