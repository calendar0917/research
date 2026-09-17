"""FSAR route-recheck ZINC runner: clean binding null + explicit local S.

Stages
------
``params preprocess sanity smoke gradient_audit train soup diagnostics witness
interventions explicit_basis_diagnostics gate decide report``

Frozen protocol inherited verbatim from the optimized compact-v4 cell: Adam,
lr 1e-3, weight decay 1e-5, batch 128, grad clip 5.0, max 240 epochs,
patience 40, no scheduler, single stage, checkpoint = best official-valid MAE,
fixed equal-weight Top-5 soup, deterministic algorithms when requested.

Wave 1 (historical seed1 replication) is run with the *unmodified* FSAR-v1
runner ``zinc_fsar.py`` (modes ``SAB`` / ``SAM``, seed 1).  Waves 2-5 use this
runner.  Official ZINC test is never loaded.
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

from tracks.ksvd.experiments.luyin16 import fsar
from tracks.ksvd.experiments.luyin16 import fsar_route_recheck as rr
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_variance_diagnosis as vd
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_fsar as v1_runner
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_shared_structural_patch_encoder as sspe
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/fsar_route_recheck"
FSAR_V1_RESULTS = TRACK_ROOT / "results/fsar"
CACHE_DIR = RESULTS_DIR / "cache"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SOUP_DIR = RESULTS_DIR / "soup_states"
RUNS_DIR = RESULTS_DIR / "runs"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PROTOCOL_VERSION = "fsar_route_recheck"
CACHE_SCHEMA = "fsar_route_recheck"

# --- frozen FSAR-v1 references (never retrained / used as input) ------------
FSAR_V1_A = 0.157917
FSAR_V1_SA = 0.151617
FSAR_V1_SAB = 0.130913
FSAR_V1_SAM = 0.134182

REFERENCE_BBAG_SOUP_PER_SEED = {0: 0.12738177864899625, 1: 0.12022926583880325}
REFERENCE_BFULL_SOUP_PER_SEED = {0: 0.11981802638241788, 1: 0.11812638340244302}
REFERENCE_A2_SOUP_PER_SEED = {0: 0.12169362585240742, 1: 0.1221344729354023}

# --- pre-registered gates / bands -------------------------------------------
BINDING_MEAN_GATE = 0.001
EXPLICIT_PARITY_BAND = 0.002
EXPLICIT_MODERATE_BAND = 0.005
EXPLICIT_WORTH_CONTINUING_BAND = 0.005
MECHANISM_FLOOR = 1.0e-4

#: modes actually launched by this runner (Wave 1 uses the FSAR-v1 runner)
ROUTE_NEW_MODES = ("SABI", "SAE", "SABE", "SABEI")

DIAG_BATCHES = 128
WITNESS_BATCHES = 64
SHUFFLE_REPEATS = 3
INTERVENTION_REPEATS = 3

_write_json = sspe._write_json
_read_json = sspe._read_json
_git_commit = sspe._git_commit
_environment_fingerprint = sspe._environment_fingerprint
_set_deterministic = v1_runner._set_deterministic
_shuffle_attributes = v1_runner._shuffle_attributes


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _tag(mode: str, tag: str | None = None) -> str:
    return str(tag or f"route_{mode.lower()}")


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_model(mode: str, seed: int = 0) -> rr.PatchPathFSARRouteModel:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    return rr.PatchPathFSARRouteModel(mode=str(mode))


# ---------------------------------------------------------------------------
# data (FSAR-v1 base fields + explicit basis; one shared cache)
# ---------------------------------------------------------------------------


def _cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "route_train.pkl.gz",
        CACHE_DIR / "route_valid.pkl.gz",
        CACHE_DIR / "route_cache_meta.json",
    )


def _assert_processed_ready(split: str) -> None:
    processed = ZINC_ROOT / "subset" / "processed"
    path = processed / f"{split}.pt"
    if not path.exists():
        raise RuntimeError(
            f"processed ZINC split missing: {path}; refuse to run process() "
            "because it would read the official test split"
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
        "node_basis_dim": int(rr.NODE_BASIS_DIM),
        "edge_basis_dim": int(rr.EDGE_BASIS_DIM),
        "relation_width": int(fsar.RELATION_WIDTH),
        "topology_width": int(fsar.TOPOLOGY_WIDTH),
        "patch_radius": int(fsar.PATCH_RADIUS),
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


def _loader(graphs: Sequence[Any], batch_size: int, shuffle: bool, seed: int):
    return rr.make_fsar_route_loader(graphs, batch_size, shuffle, seed)


def _selection_loader(valid_data: Sequence[Any]):
    return _loader(valid_data, 128, False, 0)


def _evaluate_mae(model: nn.Module, loader, device: torch.device) -> float:
    return sspe._evaluate_mae(model, loader, device)


# ---------------------------------------------------------------------------
# parameter accounting / sanity / gradient audit
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "modes": {},
        "official_test_loaded": False,
    }
    for mode in rr.ALL_MODES:
        payload["modes"][mode] = rr.parameter_breakdown(build_model(mode, seed=0))
    totals = {mode: row["total"] for mode, row in payload["modes"].items()}
    payload["totals"] = totals
    payload["aligned_null_total_equal_implicit"] = bool(
        totals["SAB"] == totals["SABI"]
    )
    payload["aligned_null_total_equal_explicit"] = bool(
        totals["SABE"] == totals["SABEI"]
    )
    payload["b_block_total_equal_implicit"] = bool(
        payload["modes"]["SAB"]["B_params"]
        == payload["modes"]["SABI"]["B_params"]
    )
    payload["b_block_total_equal_explicit"] = bool(
        payload["modes"]["SABE"]["B_params"]
        == payload["modes"]["SABEI"]["B_params"]
    )
    payload["budget_ok"] = bool(max(totals.values()) <= 200_000)
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
    from tracks.ksvd.tests import test_fsar_route_recheck as suite

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
        "latent_align_matches_fsar_v1",
        "latent_aligned_null_parameter_identity",
        "b_indep_same_marginal_invariance",
        "b_align_assignment_sensitivity",
        "chemistry_coverage_shared",
        "structure_coverage_shared",
        "relabel_invariance",
        "explicit_basis_chemistry_purity",
        "explicit_basis_relabel_equivariance",
        "explicit_path_has_no_message_passing",
        "explicit_aligned_null_parameter_identity",
        "no_mixed_bypass",
        "gradient_viability_all_modes",
        "b_indep_module_gradients_alive",
        "parameter_accounting",
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
        raise RuntimeError(f"FSAR route-recheck sanity gates failed: {failures}")
    return payload


def gradient_audit(mode: str, device: str = "cpu") -> dict[str, Any]:
    """Blocking pre-training gradient audit for one mode."""
    device_obj = torch.device(device)
    model = build_model(mode, seed=0).to(device_obj)
    _train, valid_data, _ = build_datasets()
    batch = next(iter(_loader(list(valid_data)[:64], 64, False, 0))).to(device_obj)
    model.train()
    # The final ``center_update`` layer is zero-initialised (FSAR-v1 residual
    # trick), so the pair/centre stack only receives a task gradient from the
    # second step on.  Run two real optimizer steps before auditing.
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
    payload = rr.active_path_from_grads(model)
    payload["loss"] = float(loss.detach())
    payload["optimizer_steps_before_audit"] = 2
    encoder = model.encoder
    named_grads: dict[str, float] = {}

    def _grad(module: nn.Module | None, name: str) -> None:
        if module is None:
            return
        total = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().abs().sum())
        named_grads[name] = total

    _grad(encoder.atom_embedding, "atom_embedding")
    _grad(encoder.atom_mlp, "atom_mlp")
    _grad(encoder.bond_embedding, "bond_embedding")
    _grad(encoder.bond_mlp, "bond_mlp")
    _grad(encoder.attribute_fuse, "attribute_fuse")
    _grad(encoder.structure_pool, "structure_pool")
    _grad(getattr(encoder, "message", None), "message")
    _grad(getattr(encoder, "update", None), "update")
    _grad(encoder.node_role_projection, "node_role_projection")
    _grad(encoder.node_attribute_projection, "node_attribute_projection")
    _grad(encoder.edge_role_mlp, "edge_role_mlp")
    _grad(encoder.edge_attribute_mlp, "edge_attribute_mlp")
    _grad(encoder.edge_role_projection, "edge_role_projection")
    _grad(encoder.edge_attribute_projection, "edge_attribute_projection")
    _grad(encoder.binding_fuse, "binding_fuse")
    _grad(model.capacity_mlp, "capacity_mlp")
    _grad(model.node_init, "node_init")
    _grad(model.relation_encoder, "relation_encoder")
    _grad(model.pair_encoder, "pair_encoder")
    _grad(model.center_update, "center_update")
    _grad(model.head, "head")

    required = [
        "atom_embedding",
        "atom_mlp",
        "bond_embedding",
        "bond_mlp",
        "node_init",
        "relation_encoder",
        "pair_encoder",
        "center_update",
        "head",
    ]
    if rr.MODE_LOCAL_S[mode] is not None:
        required.append("structure_pool")
    if rr.MODE_BINDING[mode] is not None:
        required.extend(
            [
                "node_role_projection",
                "node_attribute_projection",
                "edge_role_mlp",
                "edge_attribute_mlp",
                "edge_role_projection",
                "edge_attribute_projection",
                "binding_fuse",
            ]
        )
    if rr.MODE_CAPACITY.get(mode, False):
        required.append("capacity_mlp")
    missing = [
        name
        for name in required
        if not (float(named_grads.get(name, 0.0)) > 0.0)
    ]
    b_modules = [
        module
        for module in (
            encoder.node_role_projection,
            encoder.node_attribute_projection,
            encoder.edge_role_mlp,
            encoder.edge_attribute_mlp,
            encoder.edge_role_projection,
            encoder.edge_attribute_projection,
            encoder.binding_fuse,
        )
        if module is not None
    ]
    b_modules_active = bool(
        all(
            any(
                parameter.grad is not None
                and bool((parameter.grad != 0).any())
                for parameter in module.parameters()
            )
            for module in b_modules
        )
    )
    payload.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "mode": str(mode),
            "local_s_kind": rr.MODE_LOCAL_S[mode],
            "binding_kind": rr.MODE_BINDING[mode],
            "device": str(device),
            "module_grad_abs_sum": named_grads,
            "required_modules": required,
            "missing_grad_modules": missing,
            "all_required_grads_alive": bool(not missing),
            "all_b_modules_active": b_modules_active,
            "gradient_audit_pass": bool(
                not missing and (b_modules_active or not b_modules)
            ),
            "official_test_loaded": False,
        }
    )
    model.zero_grad()
    _write_json(RESULTS_DIR / f"gradient_audit_{mode}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def gradient_audit_all(device: str = "cpu") -> dict[str, Any]:
    rows = {}
    for mode in rr.ALL_MODES:
        rows[mode] = gradient_audit(mode, device=device)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "modes": rows,
        "all_pass": bool(
            all(row["gradient_audit_pass"] for row in rows.values())
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "gradient_audit.json", payload)
    return payload


def smoke(device: str = "cpu", steps: int = 5, mode: str = "SABI") -> dict[str, Any]:
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
    cross_mol = {
        "A": float(channels["attributes"].std(dim=0, unbiased=False).mean())
    }
    if "structure" in channels:
        cross_mol["S"] = float(channels["structure"].std(dim=0, unbiased=False).mean())
    if "binding" in channels:
        cross_mol["B"] = float(channels["binding"].std(dim=0, unbiased=False).mean())
    audit = rr.active_path_accounting(model, batch)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "local_s_kind": rr.MODE_LOCAL_S[mode],
        "binding_kind": rr.MODE_BINDING[mode],
        "device": str(device),
        "steps": int(steps),
        "losses": losses,
        "loss_decreasing": bool(losses[-1] <= losses[0]),
        "loss_finite": bool(finite),
        "output_not_constant": bool(prediction.std() > 1e-8),
        "cross_molecule_channel_std": cross_mol,
        "channels_alive": bool(all(value > 1e-8 for value in cross_mol.values())),
        "active_prediction_path_params": audit["active_prediction_path_params"],
        "all_modules_active": audit["all_modules_active"],
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
# training
# ---------------------------------------------------------------------------


def _channel_curve_stats(
    model: rr.PatchPathFSARRouteModel, batches: Sequence[Any]
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


def train(
    mode: str,
    seed: int,
    device: str = "cpu",
    tag: str | None = None,
    protocol_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tag = _tag(mode, tag)
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
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(train_mae),
                "valid_mae": float(valid_mae),
                **stats,
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
        "local_s_kind": rr.MODE_LOCAL_S[mode],
        "binding_kind": rr.MODE_BINDING[mode],
        "seed": int(seed),
        "protocol": protocol,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(losses)),
        "steps_per_epoch": int(steps_per_epoch),
        "wall_clock_s": wall_clock,
        "early_stopped": bool(len(losses) < max_epochs),
        "parameters": int(total_params),
        "parameter_breakdown": rr.parameter_breakdown(model),
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


# ---------------------------------------------------------------------------
# soup
# ---------------------------------------------------------------------------


def soup(mode: str, seed: int, tag: str | None = None) -> dict[str, Any]:
    tag = _tag(mode, tag)
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
        "local_s_kind": rr.MODE_LOCAL_S[mode],
        "binding_kind": rr.MODE_BINDING[mode],
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
    tag = _tag(mode, tag)
    state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    if state_path.exists():
        print(f"[run] selection state exists, skipping train: {state_path}", flush=True)
    else:
        train(mode, seed, device=device, tag=tag)
    soup_json = RESULTS_DIR / f"soup_{tag}_seed{seed}.json"
    if soup_json.exists():
        print(f"[run] soup json exists, skipping soup: {soup_json}", flush=True)
        return _read_json(soup_json)
    return soup(mode, seed, tag=tag)


# ---------------------------------------------------------------------------
# frozen eval: diagnostics / witness / interventions
# ---------------------------------------------------------------------------


def _load_frozen(mode: str, seed: int, tag: str | None = None, state: str = "best"):
    tag = _tag(mode, tag)
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


def _channel_matrix(model, batches, channel: str) -> np.ndarray:
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


def diagnostics(
    mode: str = "SABI", seed: int = 0, tag: str | None = None, state: str = "best"
) -> dict[str, Any]:
    model = _load_frozen(mode, seed, tag, state=state)
    _train, valid_data, _ = build_datasets()
    batches = list(_loader(list(valid_data)[:DIAG_BATCHES], 128, False, 0))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "local_s_kind": rr.MODE_LOCAL_S[mode],
        "binding_kind": rr.MODE_BINDING[mode],
        "seed": int(seed),
        "channels": {},
        "node_init_contribution": {},
        "official_test_loaded": False,
    }
    for name, key in (("A", "attributes"), ("S", "structure"), ("B", "binding")):
        matrix = _channel_matrix(model, batches, key)
        stats = _effective_rank(matrix)
        payload["channels"][name] = {
            "dim": int(matrix.shape[1]) if matrix.size else 0,
            "norm_mean": (
                float(np.linalg.norm(matrix, axis=1).mean()) if matrix.size else 0.0
            ),
            "cross_mol_std": (
                float(matrix.std(axis=0).mean()) if matrix.size else 0.0
            ),
            **stats,
        }
    with torch.no_grad():
        batch = batches[0]
        init = model.node_init
        z = model.encoder(batch)
        offsets = {"A": (0, rr.A_DIM)}
        if rr.MODE_LOCAL_S[mode] is not None:
            offsets["S"] = (rr.A_DIM, rr.A_DIM + rr.S_DIM)
        if rr.MODE_BINDING[mode] is not None:
            offsets["B"] = (
                rr.A_DIM + rr.S_DIM,
                rr.A_DIM + rr.S_DIM + rr.B_DIM,
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
    payload["active_path"] = rr.active_path_accounting(model, batches[0])
    model.zero_grad()
    _write_json(
        RESULTS_DIR
        / (
            f"diagnostics_{_tag(mode, tag)}_seed{seed}"
            + ("_soup" if str(state) == "soup" else "")
            + ".json"
        ),
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def _branch_weight_norms(encoder: nn.Module) -> dict[str, float | None]:
    """Per-sub-branch |weight| sums for the B channel (None if absent)."""

    def total(module: nn.Module | None) -> float | None:
        if module is None:
            return None
        return float(sum(p.detach().abs().sum() for p in module.parameters()))

    return {
        name: total(getattr(encoder, name, None))
        for name in (
            "structure_pool",
            "message",
            "node_role_projection",
            "node_attribute_projection",
            "edge_role_mlp",
            "edge_attribute_mlp",
            "edge_role_projection",
            "edge_attribute_projection",
            "binding_fuse",
        )
    }


def checkpoint_audit(
    mode: str = "SABI",
    seed: int = 0,
    tag: str | None = None,
    state: str = "soup",
) -> dict[str, Any]:
    """Post-hoc liveness audit of the trained sub-branches (eval-only).

    Reports the |weight| sum of every B-channel sub-branch so that
    ``Adam + L2`` primitive annihilation is visible without loading tensors
    back on the analysis machine.  Route states are preferred; the frozen
    FSAR-v1 state is used as a fallback so the v1 references can be audited
    with the same code path.
    """
    tag = _tag(mode, tag)
    if str(state) == "soup":
        candidates = [
            SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt",
            FSAR_V1_RESULTS / "soup_states" / f"fsar_{str(mode).lower()}_seed{seed}_top5_soup.pt",
        ]
    elif str(state) == "best":
        candidates = [
            STATE_DIR / f"{tag}_seed{seed}_selection_state.pt",
            FSAR_V1_RESULTS / "states" / f"fsar_{str(mode).lower()}_seed{seed}_selection_state.pt",
        ]
    else:
        raise ValueError(f"unknown state {state!r}; expected 'best' or 'soup'")
    state_path = next((path for path in candidates if path.exists()), None)
    if state_path is None:
        raise FileNotFoundError(
            f"no checkpoint for mode={mode} seed={seed} state={state}; tried "
            + ", ".join(str(path) for path in candidates)
        )
    model = build_model(mode, seed)
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    fresh = build_model(mode, seed)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "local_s_kind": rr.MODE_LOCAL_S[mode],
        "binding_kind": rr.MODE_BINDING[mode],
        "seed": int(seed),
        "state": str(state),
        "state_path": str(state_path.relative_to(TRACK_ROOT.parent)),
        "source": (
            "route_harness" if str(state_path).startswith(str(RESULTS_DIR)) else "fsar_v1"
        ),
        "trained_branch_weight_abs_sum": _branch_weight_norms(model.encoder),
        "fresh_branch_weight_abs_sum": _branch_weight_norms(fresh.encoder),
        "capacity_mlp_weight_abs_sum": (
            None
            if model.capacity_mlp is None
            else float(
                sum(p.detach().abs().sum() for p in model.capacity_mlp.parameters())
            )
        ),
        "annihilated_branches": [],
        "official_test_loaded": False,
    }
    trained = payload["trained_branch_weight_abs_sum"]
    fresh_norms = payload["fresh_branch_weight_abs_sum"]
    for name, value in trained.items():
        if value is None:
            continue
        reference = fresh_norms.get(name)
        if reference is None or reference <= 0.0:
            continue
        if value <= 1.0e-6 * reference:
            payload["annihilated_branches"].append(name)
    _write_json(
        RESULTS_DIR
        / (
            f"checkpoint_audit_{_tag(mode, tag)}_seed{seed}"
            + ("_soup" if str(state) == "soup" else "_best")
            + ".json"
        ),
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def witness(
    mode: str = "SABI", seed: int = 0, tag: str | None = None, state: str = "best"
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
                prediction_deltas.append(
                    float((model(batch) - model(view)).abs().mean())
                )
    mean_b = float(np.mean(channel_deltas["B"])) if channel_deltas["B"] else 0.0
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "local_s_kind": rr.MODE_LOCAL_S[mode],
        "binding_kind": rr.MODE_BINDING[mode],
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
        "B_responds": bool(bool(channel_deltas["B"]) and mean_b > MECHANISM_FLOOR),
        "B_assignment_invariant": bool(mean_b <= MECHANISM_FLOOR),
        "official_test_loaded": False,
    }
    _write_json(
        RESULTS_DIR
        / (
            f"witness_{_tag(mode, tag)}_seed{seed}"
            + ("_soup" if str(state) == "soup" else "")
            + ".json"
        ),
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def interventions(
    mode: str = "SABI", seed: int = 0, tag: str | None = None, state: str = "best"
) -> dict[str, Any]:
    """Frozen dependency / reliance diagnostics (NOT causal MAE estimates)."""
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
        "local_s_kind": rr.MODE_LOCAL_S[mode],
        "binding_kind": rr.MODE_BINDING[mode],
        "seed": int(seed),
        "valid_mae": rows,
        "delta_no_S": (None if "no_S" not in rows else rows["no_S"] - rows["true"]),
        "delta_no_A": (None if "no_A" not in rows else rows["no_A"] - rows["true"]),
        "delta_no_B": (None if "no_B" not in rows else rows["no_B"] - rows["true"]),
        "delta_shuffle_attributes": rows["shuffle_attributes"] - rows["true"],
        "interpretation": (
            "dependency / reliance diagnostics, not additive causal estimates "
            "of MAE contribution"
        ),
        "official_test_loaded": False,
    }
    _write_json(
        RESULTS_DIR
        / (
            f"interventions_{_tag(mode, tag)}_seed{seed}"
            + ("_soup" if str(state) == "soup" else "")
            + ".json"
        ),
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# structural explicitness diagnostics (descriptive attribution only)
# ---------------------------------------------------------------------------


def explicit_basis_diagnostics(
    mode: str = "SABE", seed: int = 0, tag: str | None = None, state: str = "best"
) -> dict[str, Any]:
    if rr.MODE_LOCAL_S[mode] != "explicit":
        raise ValueError(f"mode {mode!r} has no explicit local structure")
    model = _load_frozen(mode, seed, tag, state=state)
    _train, valid_data, _ = build_datasets()
    batches = list(_loader(list(valid_data)[:DIAG_BATCHES], 128, False, 0))
    node_basis = _channel_matrix(model, batches, "node_basis")
    edge_basis = _channel_matrix(model, batches, "edge_basis")

    node_groups = {
        "root_indicator": [0],
        "shell_one_hot": [1, 2, 3],
        "induced_degree": [4],
        "neighbour_by_shell": [5, 6, 7],
        "rooted_walk_1": [8],
        "rooted_walk_2": [9],
        "rooted_walk_3": [10],
    }
    edge_groups = {
        "shell_pair_one_hot": [0, 1, 2, 3, 4, 5],
        "degree_sum": [6],
        "degree_abs_diff": [7],
        "common_neighbours": [8],
        "shell_sum": [9, 10, 11],
        "shell_abs_diff": [12, 13, 14],
    }

    def _coordinate_stats(matrix: np.ndarray, groups: Mapping[str, Any]) -> dict[str, Any]:
        if matrix.size == 0:
            return {}
        result: dict[str, Any] = {
            "mean": matrix.mean(axis=0).tolist(),
            "std": matrix.std(axis=0).tolist(),
            "min": matrix.min(axis=0).tolist(),
            "max": matrix.max(axis=0).tolist(),
            "nonzero_frequency": (matrix != 0).mean(axis=0).tolist(),
            "groups": {
                name: {
                    "indices": list(indices),
                    "group_mean": float(matrix[:, indices].mean()),
                    "group_std": float(matrix[:, indices].std()),
                    "group_nonzero_frequency": float((matrix[:, indices] != 0).mean()),
                }
                for name, indices in groups.items()
            },
        }
        if matrix.shape[1] > 1:
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            scale = centered.std(axis=0)
            scale[scale < 1e-9] = 1.0
            correlation = (centered / scale).T @ (centered / scale) / max(
                matrix.shape[0] - 1, 1
            )
            result["correlation_matrix"] = correlation.tolist()
        return result

    encoder = model.encoder
    projection_column_norms: dict[str, Any] = {}
    structure_weight = encoder.structure_pool[0].weight.detach().numpy()
    blocks = {
        "root": (0, rr.NODE_BASIS_DIM),
        "node_mean": (rr.NODE_BASIS_DIM, 2 * rr.NODE_BASIS_DIM),
        "node_std": (2 * rr.NODE_BASIS_DIM, 3 * rr.NODE_BASIS_DIM),
        "edge_mean": (
            3 * rr.NODE_BASIS_DIM,
            3 * rr.NODE_BASIS_DIM + rr.EDGE_BASIS_DIM,
        ),
        "edge_std": (
            3 * rr.NODE_BASIS_DIM + rr.EDGE_BASIS_DIM,
            3 * rr.NODE_BASIS_DIM + 2 * rr.EDGE_BASIS_DIM,
        ),
    }
    projection_column_norms["structure_pool_first_layer"] = {
        "per_coordinate": np.linalg.norm(structure_weight, axis=0).tolist(),
        "per_block": {
            name: float(np.linalg.norm(structure_weight[:, start:stop]))
            for name, (start, stop) in blocks.items()
        },
    }
    if encoder.node_role_projection is not None:
        node_role_weight = encoder.node_role_projection.weight.detach().numpy()
        projection_column_norms["node_role_projection"] = {
            "per_coordinate": np.linalg.norm(node_role_weight, axis=0).tolist(),
            "per_group": {
                name: float(np.linalg.norm(node_role_weight[:, indices]))
                for name, indices in node_groups.items()
            },
        }
    if encoder.edge_role_mlp is not None:
        edge_role_weight = encoder.edge_role_mlp[0].weight.detach().numpy()
        projection_column_norms["edge_role_mlp_first_layer"] = {
            "per_coordinate": np.linalg.norm(edge_role_weight, axis=0).tolist(),
            "per_group": {
                name: float(np.linalg.norm(edge_role_weight[:, indices]))
                for name, indices in edge_groups.items()
            },
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
        "seed": int(seed),
        "state": str(state),
        "node_basis_dim": int(rr.NODE_BASIS_DIM),
        "edge_basis_dim": int(rr.EDGE_BASIS_DIM),
        "node_basis": _coordinate_stats(node_basis, node_groups),
        "edge_basis": _coordinate_stats(edge_basis, edge_groups),
        "learned_first_projection_column_norms": projection_column_norms,
        "note": (
            "descriptive coordinate-usage attribution only; not a causal claim"
        ),
        "official_test_loaded": False,
    }
    _write_json(
        RESULTS_DIR
        / (
            f"explicit_basis_diagnostics_{_tag(mode, tag)}_seed{seed}"
            + ("_soup" if str(state) == "soup" else "")
            + ".json"
        ),
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# gate / decision / report
# ---------------------------------------------------------------------------


def _soup_valid(mode: str, seed: int) -> float | None:
    path = RESULTS_DIR / f"soup_{_tag(mode)}_seed{seed}.json"
    if not path.exists():
        return None
    return float(_read_json(path)["top5_soup_valid_mae"])


def _v1_soup_valid(mode: str, seed: int) -> float | None:
    path = FSAR_V1_RESULTS / f"soup_fsar_{mode.lower()}_seed{seed}.json"
    if not path.exists():
        return None
    return float(_read_json(path)["top5_soup_valid_mae"])


def gate() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "total_params": {},
        "b_params": {},
        "official_test_loaded": False,
    }
    for mode in rr.ALL_MODES:
        breakdown = rr.parameter_breakdown(build_model(mode, seed=0))
        payload["total_params"][mode] = breakdown["total"]
        payload["b_params"][mode] = breakdown["B_params"]
    payload["implicit_total_equal"] = bool(
        payload["total_params"]["SAB"] == payload["total_params"]["SABI"]
    )
    payload["explicit_total_equal"] = bool(
        payload["total_params"]["SABE"] == payload["total_params"]["SABEI"]
    )
    payload["implicit_B_equal"] = bool(
        payload["b_params"]["SAB"] == payload["b_params"]["SABI"]
    )
    payload["explicit_B_equal"] = bool(
        payload["b_params"]["SABE"] == payload["b_params"]["SABEI"]
    )
    audit_path = RESULTS_DIR / "gradient_audit.json"
    audit = _read_json(audit_path) if audit_path.exists() else None
    payload["gradient_audit_available"] = audit is not None
    if audit is not None:
        payload["gradient_audit_all_pass"] = bool(audit.get("all_pass"))
        payload["zero_grad_on_nulls"] = {
            mode: audit["modes"][mode]["active_path"][
                "zero_grad_parameter_elements"
            ]
            for mode in audit["modes"]
        }
        payload["active_prediction_path_params"] = {
            mode: audit["modes"][mode]["active_path"][
                "active_prediction_path_params"
            ]
            for mode in audit["modes"]
        }
    payload["matched_control_valid"] = bool(
        payload["implicit_total_equal"]
        and payload["explicit_total_equal"]
        and payload["implicit_B_equal"]
        and payload["explicit_B_equal"]
        and (audit is None or bool(audit.get("all_pass")))
    )
    _write_json(RESULTS_DIR / "gate.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def decide() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "soups": {},
        "official_test_loaded": False,
    }
    for mode in rr.ALL_MODES:
        payload["soups"][mode] = {str(seed): _soup_valid(mode, seed) for seed in (0, 1)}
    payload["v1_legacy_soups"] = {
        mode: {str(seed): _v1_soup_valid(mode, seed) for seed in (0, 1)}
        for mode in ("SAB", "SAM")
    }
    # explicit seed0 reference may be the FSAR-v1 number for SAB
    latent_sab = {
        0: payload["soups"]["SAB"]["0"]
        if payload["soups"]["SAB"]["0"] is not None
        else FSAR_V1_SAB,
        1: payload["soups"]["SAB"]["1"]
        if payload["soups"]["SAB"]["1"] is not None
        else payload["v1_legacy_soups"]["SAB"]["1"],
    }
    payload["latent_sab_reference"] = latent_sab
    # latent SA reference: FSAR-v1 seed0 is the frozen number; seed1 was not run.
    latent_sa = {0: FSAR_V1_SA, 1: _v1_soup_valid("SA", 1)}
    payload["latent_sa_reference"] = latent_sa

    # --- A. historical replication ------------------------------------------
    hist = {}
    for seed in (0, 1):
        sab = latent_sab[seed]
        sam = payload["v1_legacy_soups"]["SAM"][str(seed)]
        hist[str(seed)] = {
            "sab": sab,
            "sam": sam,
            "sam_minus_sab": (None if (sab is None or sam is None) else sam - sab),
        }
    payload["historical_replication"] = hist
    payload["historical_replication_supported"] = bool(
        all(
            row["sam_minus_sab"] is not None and row["sam_minus_sab"] > 0.0
            for row in hist.values()
        )
    )

    # --- B. clean binding evidence -----------------------------------------
    latent_delta = {}
    for seed in (0, 1):
        sab = latent_sab[seed]
        sabi = payload["soups"]["SABI"][str(seed)]
        latent_delta[str(seed)] = None if (sab is None or sabi is None) else sabi - sab
    payload["delta_B_latent"] = latent_delta
    known = [v for v in latent_delta.values() if v is not None]
    payload["delta_B_latent_mean"] = float(np.mean(known)) if known else None
    payload["clean_binding_supported"] = bool(
        len(known) == 2
        and all(v > 0.0 for v in known)
        and float(np.mean(known)) >= BINDING_MEAN_GATE
    )

    # --- C. structural explicitness ----------------------------------------
    explicit = {}
    for seed in (0, 1):
        sab = latent_sab[seed]
        sae = payload["soups"]["SAE"][str(seed)]
        sabe = payload["soups"]["SABE"][str(seed)]
        gap = None if (sab is None or sabe is None) else sabe - sab
        if gap is None:
            band = None
        elif gap <= EXPLICIT_PARITY_BAND:
            band = "near_parity"
        elif gap <= EXPLICIT_MODERATE_BAND:
            band = "moderate_cost"
        else:
            band = "large_cost"
        explicit[str(seed)] = {
            "sab": sab,
            "sae": sae,
            "sabe": sabe,
            "sae_minus_sa": (
                None
                if (sae is None or latent_sa[seed] is None)
                else sae - latent_sa[seed]
            ),
            "sabe_minus_sab": gap,
            "band": band,
        }
    payload["explicit_vs_latent"] = explicit
    payload["explicit_seed0_class"] = explicit["0"]["band"]
    known_gap = [
        row["sabe_minus_sab"]
        for row in explicit.values()
        if row["sabe_minus_sab"] is not None
    ]
    payload["sabe_minus_sab_mean"] = (
        float(np.mean(known_gap)) if known_gap else None
    )
    payload["sae_minus_sa_seed0"] = explicit["0"]["sae_minus_sa"]

    # --- D. explicit binding ------------------------------------------------
    explicit_delta = {}
    for seed in (0, 1):
        sabe = payload["soups"]["SABE"][str(seed)]
        sabei = payload["soups"]["SABEI"][str(seed)]
        explicit_delta[str(seed)] = (
            None if (sabe is None or sabei is None) else sabei - sabe
        )
    payload["delta_B_explicit"] = explicit_delta
    known_explicit = [v for v in explicit_delta.values() if v is not None]
    payload["explicit_binding_supported"] = bool(
        len(known_explicit) >= 1 and all(v > 0.0 for v in known_explicit)
    )

    # --- conditional waves ---------------------------------------------------
    sabe0 = payload["soups"]["SABE"]["0"]
    sae0 = payload["soups"]["SAE"]["0"]
    sab0 = latent_sab[0]
    payload["wave4_condition"] = bool(
        sabe0 is not None
        and (
            (sab0 is not None and sabe0 <= sab0 + EXPLICIT_WORTH_CONTINUING_BAND)
            or (sae0 is not None and sabe0 < sae0 - 0.001)
        )
    )
    sabe_seed0 = payload["soups"]["SABE"]["0"]
    sabei_seed0 = payload["soups"]["SABEI"]["0"]
    payload["wave5_condition"] = bool(
        sabe_seed0 is not None
        and sabei_seed0 is not None
        and sabe_seed0 < sabei_seed0 - 0.001
    )

    payload["references"] = {
        "B-Bag_per_seed": REFERENCE_BBAG_SOUP_PER_SEED,
        "B-Full_per_seed": REFERENCE_BFULL_SOUP_PER_SEED,
        "A2_per_seed": REFERENCE_A2_SOUP_PER_SEED,
        "FSAR_v1_A": FSAR_V1_A,
        "FSAR_v1_SA": FSAR_V1_SA,
        "FSAR_v1_SAB": FSAR_V1_SAB,
        "FSAR_v1_SAM": FSAR_V1_SAM,
    }
    # --- four independent answers (never merged into one verdict) -----------
    payload["answer_A_historical_replication"] = {
        "question": "does FSAR-v1 SAB < SAM replicate at seed1?",
        "sam_minus_sab_per_seed": {
            seed: hist[str(seed)]["sam_minus_sab"] for seed in (0, 1)
        },
        "replicated": bool(payload["historical_replication_supported"]),
    }
    payload["answer_B_clean_binding"] = {
        "question": (
            "SAB < SABI (aligned vs operator-matched assignment-independent "
            "null) on paired seeds?"
        ),
        "delta_B_latent_per_seed": latent_delta,
        "delta_B_latent_mean": payload["delta_B_latent_mean"],
        "both_seeds_positive": bool(
            len(known) == 2 and all(v > 0.0 for v in known)
        ),
        "mean_ge_0.001": bool(
            payload["delta_B_latent_mean"] is not None
            and payload["delta_B_latent_mean"] >= BINDING_MEAN_GATE
        ),
        "supported": bool(payload["clean_binding_supported"]),
    }
    payload["answer_C_structural_explicitness"] = {
        "question": "what does explicit local S cost vs latent local S?",
        "sabe_minus_sab_per_seed": {
            seed: explicit[str(seed)]["sabe_minus_sab"] for seed in (0, 1)
        },
        "sabe_minus_sab_mean": payload["sabe_minus_sab_mean"],
        "sae_minus_sa_seed0": payload["sae_minus_sa_seed0"],
        "seed0_band": explicit["0"]["band"],
        "seed1_band": explicit["1"]["band"],
    }
    payload["answer_D_explicit_binding"] = {
        "question": "SABE < SABEI (explicit aligned vs explicit null)?",
        "delta_B_explicit_per_seed": explicit_delta,
        "seed0_ran": bool(explicit_delta["0"] is not None or sabe_seed0 is not None),
        "supported": bool(payload["explicit_binding_supported"]),
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / f"{name}.json"
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "parameter_accounting": _maybe("parameter_accounting"),
        "sanity": _maybe("sanity"),
        "gradient_audit": _maybe("gradient_audit"),
        "gate": _maybe("gate"),
        "decision": _maybe("decision"),
        "soups_seed0": {mode: _soup_valid(mode, 0) for mode in rr.ALL_MODES},
        "soups_seed1": {mode: _soup_valid(mode, 1) for mode in rr.ALL_MODES},
        "v1_legacy_soups": {
            mode: {str(seed): _v1_soup_valid(mode, seed) for seed in (0, 1)}
            for mode in ("SAB", "SAM")
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FSAR route-recheck ZINC runner")
    parser.add_argument(
        "stage",
        choices=[
            "params",
            "preprocess",
            "sanity",
            "smoke",
            "gradient_audit",
            "train",
            "soup",
            "run",
            "diagnostics",
            "checkpoint_audit",
            "witness",
            "interventions",
            "explicit_basis_diagnostics",
            "gate",
            "decide",
            "report",
        ],
    )
    parser.add_argument("--mode", default="SABI", choices=list(rr.ALL_MODES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--state", default="best", choices=["best", "soup"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--all-modes", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    arguments = parser.parse_args(argv)
    _set_deterministic(bool(arguments.deterministic))

    if arguments.stage == "params":
        params()
    elif arguments.stage == "preprocess":
        preprocess(force=bool(arguments.force))
    elif arguments.stage == "sanity":
        sanity()
    elif arguments.stage == "smoke":
        smoke(device=arguments.device, steps=int(arguments.steps), mode=arguments.mode)
    elif arguments.stage == "gradient_audit":
        if arguments.all_modes:
            gradient_audit_all(device=arguments.device)
        else:
            gradient_audit(arguments.mode, device=arguments.device)
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
    elif arguments.stage == "run":
        run(
            arguments.mode,
            int(arguments.seed),
            device=arguments.device,
            tag=arguments.tag,
        )
    elif arguments.stage == "diagnostics":
        diagnostics(
            mode=arguments.mode,
            seed=int(arguments.seed),
            tag=arguments.tag,
            state=arguments.state,
        )
    elif arguments.stage == "checkpoint_audit":
        checkpoint_audit(
            mode=arguments.mode,
            seed=int(arguments.seed),
            tag=arguments.tag,
            state=arguments.state if arguments.state in {"best", "soup"} else "soup",
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
    elif arguments.stage == "explicit_basis_diagnostics":
        explicit_basis_diagnostics(
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
