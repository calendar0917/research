"""FSAR ZINC runner: Factorized Structure--Attribute Relational Network.

Stages
------
``params preprocess sanity smoke train soup diagnostics witness interventions
decide report``

Protocol: the frozen optimized compact-v4 protocol is inherited verbatim
(Adam, lr 1e-3, weight decay 1e-5, batch 128, grad clip 5.0, max 240 epochs,
patience 40, no scheduler, single stage, checkpoint = best official-valid MAE,
fixed equal-weight Top-5 soup, deterministic algorithms when requested).

Only the ZINC regression L1 / MAE loss is optimized.  Attribute shuffle and
channel zeroing are **evaluation-only** (never differentiated through).

Official ZINC test is **never** loaded.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import gzip
import json
import math
import os
import pickle
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import fsar
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_variance_diagnosis as vd
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_shared_structural_patch_encoder as sspe
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/fsar"
CACHE_DIR = RESULTS_DIR / "cache"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SOUP_DIR = RESULTS_DIR / "soup_states"
RUNS_DIR = RESULTS_DIR / "runs"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PROTOCOL_VERSION = "fsar_v1"
CACHE_SCHEMA = "fsar_v1"

# --- matched references (frozen; never retrained here) ----------------------
REFERENCE_BBAG_SOUP_PER_SEED = {0: 0.12738177864899625, 1: 0.12022926583880325}
REFERENCE_BFULL_SOUP_PER_SEED = {0: 0.11981802638241788, 1: 0.11812638340244302}
REFERENCE_A2_SOUP_PER_SEED = {0: 0.12169362585240742, 1: 0.1221344729354023}

# --- pre-registered engineering guards (brief sections 31/34) ---------------
A_SEED0_GUARD = 0.135          # A-init must reach at least this collapsed/null band
B_GAIN_GATE = 0.001            # SA - SAB must reach this for the B claim
SEED0_REFERENCE_TOLERANCE = 0.002
MECHANISM_FLOOR = 1.0e-4

SEEDS = (0, 1)
DIAG_BATCHES = 128
WITNESS_BATCHES = 64
SHUFFLE_REPEATS = 3
INTERVENTION_REPEATS = 3

_write_json = sspe._write_json
_read_json = sspe._read_json
_git_commit = sspe._git_commit
_environment_fingerprint = sspe._environment_fingerprint


def _set_deterministic(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_model(mode: str, seed: int = 0) -> fsar.PatchPathFSARModel:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = fsar.PatchPathFSARModel(mode=str(mode))
    return model


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "fsar_train.pkl.gz",
        CACHE_DIR / "fsar_valid.pkl.gz",
        CACHE_DIR / "fsar_cache_meta.json",
    )


def build_datasets(force: bool = False) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Build (and cache) the FSAR train / valid ``Data`` lists."""
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

    train_data, _ = fsar.build_fsar_dataset(
        train_ds,
        train_bundle["graphs"],
        topo_train,
        topology_mean=topology_mean,
        topology_scale=topology_scale,
    )
    valid_data, _ = fsar.build_fsar_dataset(
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
# loaders / evaluation
# ---------------------------------------------------------------------------


def _loader(graphs: Sequence[Any], batch_size: int, shuffle: bool, seed: int):
    return fsar.make_fsar_loader(graphs, batch_size, shuffle, seed)


def _evaluate_mae(model: nn.Module, loader, device: torch.device) -> float:
    return sspe._evaluate_mae(model, loader, device)


def _selection_loader(valid_data: Sequence[Any]):
    return _loader(valid_data, 128, False, 0)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "modes": {},
    }
    for mode in fsar.MODES:
        model = build_model(mode, seed=0)
        payload["modes"][mode] = fsar.parameter_breakdown(model)
    totals = {mode: row["total"] for mode, row in payload["modes"].items()}
    payload["nested_totals"] = totals
    payload["nested_ordering"] = bool(totals["A"] < totals["SA"] < totals["SAB"])
    payload["budget_ok"] = bool(max(totals.values()) <= 200_000)
    payload["dataset_dependent_vocabulary_params"] = 0
    payload["references"] = {
        "B-Bag_per_seed": REFERENCE_BBAG_SOUP_PER_SEED,
        "B-Full_per_seed": REFERENCE_BFULL_SOUP_PER_SEED,
        "A2_per_seed": REFERENCE_A2_SOUP_PER_SEED,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    PRINT_PARAMS(payload)
    return payload


def PRINT_PARAMS(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


# ---------------------------------------------------------------------------
# synthetic correctness gate (architecture level)
# ---------------------------------------------------------------------------


def sanity() -> dict[str, Any]:
    from tracks.ksvd.tests import test_fsar as suite

    checks: dict[str, Any] = {}
    failures: list[str] = []

    def run(name: str, function) -> None:
        try:
            function()
            checks[name] = True
        except Exception as error:  # noqa: BLE001 - report the gate failure
            checks[name] = False
            failures.append(f"{name}: {error}")

    run("no_mixed_bypass", suite.test_no_mixed_bypass_attributes_and_forward_audit)
    run("s_chemistry_invariance", suite.test_s_chemistry_invariance_exact)
    run("a_assignment_invariance", suite.test_a_context_assignment_invariance)
    run("b_assignment_sensitivity", suite.test_b_assignment_sensitivity)
    run("relation_purity", suite.test_relation_purity_topology_only)
    run("node_relabel_invariance", suite.test_node_relabel_invariance)
    run("batch_invariance", suite.test_batch_invariance)
    run("gradient_viability", suite.test_gradient_viability)
    run("parameter_accounting", suite.test_parameter_accounting_and_nested_modes)

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
        raise RuntimeError(f"FSAR sanity gates failed: {failures}")
    return payload


# ---------------------------------------------------------------------------
# GPU smoke
# ---------------------------------------------------------------------------


def smoke(device: str = "cpu", steps: int = 5, mode: str = "SAB") -> dict[str, Any]:
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_data, valid_data, _meta = build_datasets()
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
        "A": float(channels["attributes"].std(dim=0, unbiased=False).mean()),
        "S": float(channels["structure"].std(dim=0, unbiased=False).mean()),
        "B": float(channels["binding"].std(dim=0, unbiased=False).mean()),
    }
    grads = {}
    model.train()
    loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    grads = {
        "A": float(model.encoder.atom_embedding.weight.grad.abs().sum()),
        "S": float(model.encoder.root_embedding.weight.grad.abs().sum()),
        "B": float(model.encoder.node_role_projection.weight.grad.abs().sum()),
        "relation": float(model.relation_encoder[0].weight.grad.abs().sum()),
        "head": float(model.head[-1].weight.grad.abs().sum()),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
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
# training
# ---------------------------------------------------------------------------


def train(
    mode: str,
    seed: int,
    device: str = "cpu",
    tag: str | None = None,
    protocol_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tag = str(tag or f"fsar_{mode.lower()}")
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
        "parameter_breakdown": fsar.parameter_breakdown(model),
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
    model: fsar.PatchPathFSARModel, batches: Sequence[Any]
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


def _encoder_grad_norms(model: fsar.PatchPathFSARModel) -> dict[str, float]:
    def _norm(parameters) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        return float(math.sqrt(total))

    encoder = model.encoder
    s_params = []
    if encoder.topology_base is not None:
        s_params.append(encoder.topology_base)
    if encoder.message is not None:
        s_params.extend(encoder.message.parameters())
    if encoder.update is not None:
        s_params.extend(encoder.update.parameters())
    if encoder.structure_pool is not None:
        s_params.extend(encoder.structure_pool.parameters())
    return {
        "grad_A": _norm(
            list(encoder.atom_embedding.parameters())
            + list(encoder.atom_mlp.parameters())
            + list(encoder.bond_mlp.parameters())
            + list(encoder.attribute_fuse.parameters())
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
    tag = str(tag or f"fsar_{mode.lower()}")
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


# ---------------------------------------------------------------------------
# diagnostics / witness / interventions (frozen checkpoint, eval only)
# ---------------------------------------------------------------------------


def _load_frozen(mode: str, seed: int, tag: str | None = None):
    tag = str(tag or f"fsar_{mode.lower()}")
    model = build_model(mode, seed)
    state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _channel_matrix(model: fsar.PatchPathFSARModel, batches, channel: str) -> np.ndarray:
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


def diagnostics(mode: str = "SAB", seed: int = 0, tag: str | None = None) -> dict[str, Any]:
    model = _load_frozen(mode, seed, tag)
    _train, valid_data, _ = build_datasets()
    batches = list(_loader(list(valid_data)[:DIAG_BATCHES], 128, False, 0))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
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
            "norm_mean": float(np.linalg.norm(matrix, axis=1).mean()) if matrix.size else 0.0,
            "cross_mol_std": float(matrix.std(axis=0).mean()) if matrix.size else 0.0,
            **stats,
        }
    with torch.no_grad():
        batch = batches[0]
        channels = model.encoder.forward_channels(batch)
        init = model.node_init
        z = model.encoder(batch)
        # contribution of each channel to the node-init input
        offsets = {"A": (0, fsar.A_DIM)}
        if model.encoder.include_structure:
            offsets["S"] = (fsar.A_DIM, fsar.A_DIM + fsar.S_DIM)
        if model.encoder.include_binding:
            offsets["B"] = (
                fsar.A_DIM + fsar.S_DIM,
                fsar.A_DIM + fsar.S_DIM + fsar.B_DIM,
            )
        for name, (start, stop) in offsets.items():
            block = z[:, start:stop]
            weight = init[0].weight[:, start:stop]
            payload["node_init_contribution"][name] = float(
                (block @ weight.t()).norm(dim=1).mean()
            )
    # gradient norms on one batch
    model.train()
    loss = F.l1_loss(
        model(batches[0]).view(-1), batches[0].y.view(-1)
    )
    model.zero_grad()
    loss.backward()
    payload["grad_norms"] = _encoder_grad_norms(model)
    model.zero_grad()
    _write_json(RESULTS_DIR / f"diagnostics_{tag or f'fsar_{mode.lower()}'}_seed{seed}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def _shuffle_attributes(batch, seed: int):
    from tracks.ksvd.experiments.luyin16.zinc_factorized_binding_encoder import (
        shuffle_attribute_assignment,
    )

    return shuffle_attribute_assignment(batch, seed)


def witness(mode: str = "SAB", seed: int = 0, tag: str | None = None) -> dict[str, Any]:
    model = _load_frozen(mode, seed, tag)
    _train, valid_data, _ = build_datasets()
    batches = list(_loader(list(valid_data)[:WITNESS_BATCHES], 128, False, 0))
    channel_deltas = {"A": [], "S": [], "B": []}
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
                    float(
                        (model(batch) - model(view)).abs().mean()
                    )
                )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": str(mode),
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
        RESULTS_DIR / f"witness_{tag or f'fsar_{mode.lower()}'}_seed{seed}.json",
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def interventions(mode: str = "SAB", seed: int = 0, tag: str | None = None) -> dict[str, Any]:
    model = _load_frozen(mode, seed, tag)
    _train, valid_data, _ = build_datasets()
    loader = _selection_loader(valid_data)
    rows: dict[str, float] = {}
    with torch.no_grad():
        model.encoder.set_intervention(None)
        rows["true"] = _evaluate_mae(model, loader, torch.device("cpu"))
        for key in ("S", "A", "A_self", "A_ctx", "B"):
            if key == "S" and not model.encoder.include_structure:
                continue
            if key == "B" and not model.encoder.include_binding:
                continue
            model.encoder.set_intervention([key])
            rows[f"no_{key}"] = _evaluate_mae(model, loader, torch.device("cpu"))
        model.encoder.set_intervention(None)
        # shuffled-B: replace B by the B of an attribute-shuffled view, then
        # keep the rest of the forward identical.  Implemented by evaluating on
        # shuffled-attribute molecules (topology untouched).
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
        "seed": int(seed),
        "valid_mae": rows,
        "delta_no_S": rows.get("no_S", 0.0) - rows["true"],
        "delta_no_A": rows.get("no_A", 0.0) - rows["true"],
        "delta_A_self": rows.get("no_A_self", 0.0) - rows["true"],
        "delta_A_ctx": rows.get("no_A_ctx", 0.0) - rows["true"],
        "delta_no_B": rows.get("no_B", 0.0) - rows["true"],
        "delta_shuffle_attributes": rows["shuffle_attributes"] - rows["true"],
        "official_test_loaded": False,
    }
    _write_json(
        RESULTS_DIR / f"interventions_{tag or f'fsar_{mode.lower()}'}_seed{seed}.json",
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def _soup_valid(mode: str, seed: int) -> float | None:
    tag = f"fsar_{mode.lower()}"
    path = RESULTS_DIR / f"soup_{tag}_seed{seed}.json"
    if not path.exists():
        return None
    return float(_read_json(path)["top5_soup_valid_mae"])


def decide() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "soups": {},
        "deltas": {},
        "status": "INCOMPLETE",
        "case": None,
        "seed1_authorized": False,
        "official_test_loaded": False,
    }
    for mode in fsar.MODES:
        value = _soup_valid(mode, 0)
        payload["soups"][mode] = value
    a, sa, sab = (payload["soups"][m] for m in fsar.MODES)
    if a is None:
        payload["status"] = "A_NOT_RUN"
        payload["a_guard_pass"] = None
    else:
        payload["a_guard_pass"] = bool(a <= A_SEED0_GUARD)
        if not payload["a_guard_pass"]:
            payload["status"] = "STOP_A_GUARD"
    if a is not None and sa is not None:
        payload["deltas"]["A_to_SA"] = a - sa
    if sa is not None and sab is not None:
        payload["deltas"]["SA_to_SAB"] = sa - sab
    if a is not None and sa is not None and sab is not None:
        payload["status"] = "COMPLETE_SEED0"
        b_gain = payload["deltas"]["SA_to_SAB"]
        payload["case"] = (
            "SAB_adds_increment" if b_gain >= B_GAIN_GATE else "SAB_no_increment"
        )
        payload["seed1_authorized"] = bool(b_gain >= B_GAIN_GATE)
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
        "smoke_sab": _maybe("smoke_SAB"),
        "decision": _maybe("decision"),
        "soups_seed0": {mode: _soup_valid(mode, 0) for mode in fsar.MODES},
        "soups_seed1": {mode: _soup_valid(mode, 1) for mode in fsar.MODES},
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FSAR ZINC runner")
    parser.add_argument(
        "stage",
        choices=[
            "params",
            "preprocess",
            "sanity",
            "smoke",
            "train",
            "soup",
            "diagnostics",
            "witness",
            "interventions",
            "decide",
            "report",
        ],
    )
    parser.add_argument("--mode", default="SAB", choices=list(fsar.MODES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
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
        diagnostics(mode=arguments.mode, seed=int(arguments.seed), tag=arguments.tag)
    elif arguments.stage == "witness":
        witness(mode=arguments.mode, seed=int(arguments.seed), tag=arguments.tag)
    elif arguments.stage == "interventions":
        interventions(mode=arguments.mode, seed=int(arguments.seed), tag=arguments.tag)
    elif arguments.stage == "decide":
        decide()
    elif arguments.stage == "report":
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
