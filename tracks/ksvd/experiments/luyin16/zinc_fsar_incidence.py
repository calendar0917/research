"""FSAR-C1 runner: parameter-matched incidence processor (ZINC cell A protocol).

Round name: ``FSAR-C1``.  Branch ``exp/fsar-incidence-processor-zinc``.

Stages
------
``params preprocess audit sanity gradient_audit smoke synthetic train soup run
diagnostics witness gate decide report``

Two variants share identical modules and an identical parameter count:

* ``C1`` -- persistent node/edge S-A-B objects + real shared recurrent incidence
  processor (node <-> incident edge).
* ``C0`` -- the parameter-exact incidence-free bag control (global aggregation
  only), used only as a mechanism control, never as a promotion candidate.

The training protocol is inherited verbatim from
``zinc_compact_v4_smallhead_e2e.OPTIMIZED_PROTOCOL`` with the single documented
deviation that binding / LayerNorm / residual-scale parameters are placed in a
``weight_decay = 0`` group (pre-registration §4).  Official ZINC test is never
loaded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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
from tracks.ksvd.experiments.luyin16 import fsar_incidence as inc
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0 as ar0
from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0_edge as edge_runner
from tracks.ksvd.experiments.luyin16.zinc_fsar import _set_deterministic

REPO_ROOT = ar0.REPO_ROOT
TRACK_ROOT = ar0.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/fsar_incidence"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SOUP_DIR = RESULTS_DIR / "soup_states"
RUNS_DIR = RESULTS_DIR / "runs"

ZINC_ROOT = ar0.ZINC_ROOT
PROTOCOL_VERSION = "fsar_incidence_v1"
CACHE_SCHEMA = edge_runner.CACHE_SCHEMA

_write_json = ar0._write_json
_read_json = ar0._read_json
_git_commit = ar0._git_commit
_environment_fingerprint = ar0._environment_fingerprint

DIAG_BATCHES = 32
GRAD_DIAG_EVERY = 20
SHUFFLE_REPEATS = 5

#: frozen strong references (never used as model input)
REFERENCES = {
    "B_Full_soup_seed0": 0.11981802638241788,
    "B_Full_soup_seed1": 0.118126,
    "B_Bag_soup_seed0": 0.12738177864899625,
    "B_Bag_soup_seed1": 0.120229,
    "cell_A_soup_2seed_mean": 0.126368,
    "ar0_MB_assign_soup_seed0": 0.4692853513918235,
    "ar0e_BVE_assign_soup_seed0": 0.44595771902793785,
}
REFERENCE_PARAMS = {"B_Full": 84495, "B_Bag": 84511, "cell_A": 85763, "AR0_edge_BVE": 25317}

#: pre-registered seed0 gate on C1 valid Top-5 soup
GATE_STOP_ABOVE = 0.20
GATE_INSPECT_ABOVE = 0.14
GATE_STRONG_AT_OR_BELOW = 0.12


def _tag(variant: str, tag: str | None = None) -> str:
    return str(tag or f"inc_{str(variant).lower()}")


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# data (reuses the validated AR0-EDGE feature cache; test split never touched)
# ---------------------------------------------------------------------------


def build_datasets():
    return edge_runner.build_edge_datasets()


def preprocess(force: bool = False) -> dict[str, Any]:
    return edge_runner.preprocess(force=force)


def _loader(molecules, batch_size, shuffle, seed):
    return inc.make_incidence_loader(molecules, batch_size, shuffle, seed)


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
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "models": {},
        "references": REFERENCES,
        "reference_params": REFERENCE_PARAMS,
        "official_test_loaded": False,
    }
    for variant in inc.MODELS:
        model = inc.build_incidence_model(variant, seed=0)
        payload["models"][variant] = inc.parameter_breakdown_incidence(model)
    totals = {variant: row["total"] for variant, row in payload["models"].items()}
    payload["totals"] = totals
    payload["C0_C1_parameter_exact"] = bool(totals["C0"] == totals["C1"])
    payload["target_band"] = [85000, 95000]
    payload["in_target_band"] = bool(85000 <= totals["C1"] <= 95000)
    payload["dataset_dependent_vocabulary_params"] = 0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# sanity / gradient audit
# ---------------------------------------------------------------------------


def sanity() -> dict[str, Any]:
    from tracks.ksvd.tests import test_fsar_incidence as suite

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
        "bond_permutation_changes_c1",
        "incidence_used_by_c1_not_c0",
        "atom_permutation_changes_c1",
        "batch_invariance",
        "incidence_control_parameter_match",
        "all_parameters_receive_gradient",
        "one_step_updates_binding_branches",
        "parameter_counts_incidence",
        "no_forbidden_feature_access",
        "determinism",
        "empty_edges_safe",
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
        raise RuntimeError(f"FSAR-C1 sanity gates failed: {failures}")
    return payload


def _module_grad_sum(modules: Sequence[nn.Module]) -> float:
    total = 0.0
    for module in modules:
        parameters = (
            [module] if isinstance(module, nn.Parameter) else list(module.parameters())
        )
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().norm() ** 2)
    return float(total**0.5)


_GRAD_GROUPS: dict[str, tuple[str, ...]] = {
    "node_initializer": ("node_s", "node_a"),
    "edge_initializer": ("edge_s", "edge_a"),
    "node_binding": ("node_us", "node_ua", "node_b"),
    "edge_binding": ("edge_us", "edge_ua", "edge_b"),
    "edge_update": ("edge_in", "edge_out", "edge_scale"),
    "node_update": (
        "node_msg_in",
        "node_msg_out",
        "node_update_in",
        "node_update_out",
        "node_scale",
    ),
    "readout": ("readout",),
}


def _named_grad_norms(model: inc.FSARIncidenceModel) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, names in _GRAD_GROUPS.items():
        out[group] = _module_grad_sum([getattr(model, name) for name in names])
    return out


def _named_weight_norms(model: inc.FSARIncidenceModel) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, names in _GRAD_GROUPS.items():
        total = 0.0
        for name in names:
            module = getattr(model, name)
            if isinstance(module, nn.Parameter):
                total += float(module.detach().norm() ** 2)
            else:
                for parameter in module.parameters():
                    total += float(parameter.detach().norm() ** 2)
        out[group] = float(total**0.5)
    return out


def gradient_audit(variant: str, device: str = "cpu") -> dict[str, Any]:
    _train, valid_molecules, _scalers, _meta = build_datasets()
    device_obj = torch.device(device)
    model = inc.build_incidence_model(variant, seed=0).to(device_obj)
    batch = next(iter(_loader(list(valid_molecules)[:64], 64, False, 0))).to(device_obj)
    model.train()
    optimizer = torch.optim.Adam(
        inc.optimizer_parameter_groups(model, 1.0e-5), lr=1.0e-3
    )
    loss = None
    for _ in range(2):
        loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))
    loss.backward()
    grads = _named_grad_norms(model)
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
        "module_grad_norms": grads,
        "all_groups_nonzero": bool(all(value > 0.0 for value in grads.values())),
        "official_test_loaded": False,
    }
    if not payload["all_groups_nonzero"]:
        raise RuntimeError(f"{variant} has a zero-task-gradient group: {grads}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / f"gradient_audit_{variant}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def gradient_audit_all(device: str = "cpu") -> dict[str, Any]:
    return {variant: gradient_audit(variant, device=device) for variant in inc.MODELS}


def smoke(device: str = "cpu", steps: int = 6) -> dict[str, Any]:
    """Short forward/backward smoke on real (valid) batches; no performance claim."""
    _train, valid_molecules, _scalers, _meta = build_datasets()
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "device": str(device),
        "steps": int(steps),
        "variants": {},
        "official_test_loaded": False,
    }
    loader = _loader(list(valid_molecules)[:512], 128, False, 0)
    for variant in inc.MODELS:
        model = inc.build_incidence_model(variant, seed=0).to(device_obj)
        optimizer = torch.optim.Adam(
            inc.optimizer_parameter_groups(model, 1.0e-5), lr=1.0e-3
        )
        losses: list[float] = []
        predictions: list[float] = []
        step = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction = model(batch).view(-1)
            loss = F.l1_loss(prediction, batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            predictions.append(float(prediction.detach().std()))
            step += 1
            if step >= int(steps):
                break
        grads = _named_grad_norms(model)
        payload["variants"][variant] = {
            "losses": losses,
            "prediction_std": predictions,
            "finite": bool(all(math.isfinite(value) for value in losses)),
            "output_not_constant": bool(max(predictions) > 0.0),
            "module_grad_norms": grads,
            "all_groups_nonzero": bool(all(value > 0.0 for value in grads.values())),
            "parameters": _n_params(model),
        }
    if device_obj.type == "cuda":
        payload["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    _write_json(RESULTS_DIR / "smoke.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _curve_stats(model: inc.FSARIncidenceModel, batches) -> dict[str, float]:
    model.eval()
    node_binding: list[float] = []
    edge_binding: list[float] = []
    node_object: list[float] = []
    edge_object: list[float] = []
    edge_residual: list[float] = []
    node_residual: list[float] = []
    with torch.no_grad():
        for batch in batches:
            _prediction, h, g, binding, stats = model.forward_with_states(batch)
            node_binding.append(float(binding["node_binding"].std()))
            edge_binding.append(float(binding["edge_binding"].std()))
            node_object.append(float(h.std()))
            edge_object.append(float(g.std()))
            edge_residual.append(float(stats["edge_residual_std"]))
            node_residual.append(float(stats["node_residual_std"]))
    weights = _named_weight_norms(model)
    edge_scale, node_scale = model.residual_scales()
    model.train()
    return {
        "node_binding_std": float(np.mean(node_binding)),
        "edge_binding_std": float(np.mean(edge_binding)),
        "node_object_std": float(np.mean(node_object)),
        "edge_object_std": float(np.mean(edge_object)),
        "edge_residual_std": float(np.mean(edge_residual)),
        "node_residual_std": float(np.mean(node_residual)),
        "node_binding_norm": float(weights["node_binding"]),
        "edge_binding_norm": float(weights["edge_binding"]),
        "edge_scale": float(edge_scale),
        "node_scale": float(node_scale),
    }


def _grad_stats(model: inc.FSARIncidenceModel, batch) -> dict[str, float]:
    training = model.training
    model.train()
    prediction = model(batch).view(-1)
    loss = F.l1_loss(prediction, batch.y.view(-1))
    model.zero_grad()
    loss.backward()
    grads = _named_grad_norms(model)
    model.zero_grad()
    if not training:
        model.eval()
    return {f"grad_{key}": value for key, value in grads.items()}


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


def train(
    variant: str,
    seed: int,
    device: str = "cpu",
    tag: str | None = None,
    max_epochs: int | None = None,
    protocol_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tag = _tag(variant, tag)
    train_molecules, valid_molecules, _scalers, _meta = build_datasets()
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
    model = inc.build_incidence_model(variant, seed=int(seed)).to(device_obj)
    total_params = _n_params(model)

    diag_batches = [
        batch.to(device_obj)
        for batch in _loader(list(valid_molecules)[:DIAG_BATCHES], 128, False, 0)
    ]
    grad_batch = next(iter(_loader(valid_molecules, 128, False, 0))).to(device_obj)
    optimizer = torch.optim.Adam(
        inc.optimizer_parameter_groups(model, float(protocol["weight_decay"])),
        lr=float(protocol["learning_rate"]),
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
        stats = _curve_stats(model, diag_batches)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_mae": float(train_mae),
            "valid_mae": float(valid_mae),
            "grad_norm_mean": float(grad_norm_total / max(grad_steps, 1)),
            **stats,
        }
        if epoch == 1 or epoch % GRAD_DIAG_EVERY == 0 or epoch == max_epochs_value:
            row.update(_grad_stats(model, grad_batch))
        curve.append({**row, "checkpoint_selected": 0})
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
                f"node_bind_std={stats['node_binding_std']:.4e} "
                f"edge_bind_std={stats['edge_binding_std']:.4e} "
                f"edge_res_std={stats['edge_residual_std']:.4e} "
                f"node_res_std={stats['node_residual_std']:.4e}",
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
    fieldnames = list(curve[0].keys())
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
        "optimizer": "Adam with parameter groups "
        "(binding/LayerNorm/residual-scale wd=0; rest wd=%.1e)" % float(protocol["weight_decay"]),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curve)),
        "steps_per_epoch": int(steps_per_epoch),
        "wall_clock_s": wall_clock,
        "early_stopped": bool(len(curve) < max_epochs_value),
        "parameters": int(total_params),
        "parameter_breakdown": inc.parameter_breakdown_incidence(model),
        "final_binding_norm": float(model.binding_weight_norm()),
        "final_node_binding_norm": float(model.node_binding_norm()),
        "final_edge_binding_norm": float(model.edge_binding_norm()),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device_obj.type == "cuda" else 0
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

    _train, valid_molecules, _scalers, _meta = build_datasets()
    loader = _loader(valid_molecules, 128, False, 0)
    device = torch.device("cpu")
    model = inc.build_incidence_model(variant)
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
        "references": REFERENCES,
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
# frozen eval: diagnostics / attribute permutation witness
# ---------------------------------------------------------------------------


def _load_frozen(variant: str, seed: int, tag: str | None = None, state: str = "soup"):
    model = inc.build_incidence_model(variant)
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
    batches = list(_loader(valid_molecules, 128, False, 0))
    node_rows: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    node_binding: list[float] = []
    edge_binding: list[float] = []
    edge_residual: list[float] = []
    node_residual: list[float] = []
    with torch.no_grad():
        for batch in batches:
            _prediction, h, g, binding, stats = model.forward_with_states(batch)
            node_rows.append(h.numpy())
            edge_rows.append(g.numpy())
            node_binding.append(float(binding["node_binding"].std()))
            edge_binding.append(float(binding["edge_binding"].std()))
            edge_residual.append(float(stats["edge_residual_std"]))
            node_residual.append(float(stats["node_residual_std"]))
    node_matrix = np.concatenate(node_rows, axis=0)
    edge_matrix = np.concatenate(edge_rows, axis=0)
    # gradient norms on one real valid batch, in training mode
    model.train()
    batch = next(iter(_loader(valid_molecules, 128, False, 0)))
    loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))
    model.zero_grad()
    loss.backward()
    grads = _named_grad_norms(model)
    model.zero_grad()
    model.eval()
    weights = _named_weight_norms(model)
    edge_scale, node_scale = model.residual_scales()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": str(variant),
        "seed": int(seed),
        "state": str(state),
        "node_binding_std": float(np.mean(node_binding)),
        "edge_binding_std": float(np.mean(edge_binding)),
        "node_binding_weight_norm": float(model.node_binding_norm()),
        "edge_binding_weight_norm": float(model.edge_binding_norm()),
        "binding_weight_norm": float(model.binding_weight_norm()),
        "edge_residual_std": float(np.mean(edge_residual)),
        "node_residual_std": float(np.mean(node_residual)),
        "edge_scale": float(edge_scale),
        "node_scale": float(node_scale),
        "gradient_norms": grads,
        "weight_norms": weights,
        "node_representation": {
            "mean_per_dim_std": float(np.mean(node_matrix.std(axis=0))),
            "max_per_dim_std": float(np.max(node_matrix.std(axis=0))),
            "variance_sum": float(node_matrix.var(axis=0).sum()),
            **_effective_rank(node_matrix),
        },
        "edge_representation": {
            "mean_per_dim_std": float(np.mean(edge_matrix.std(axis=0))),
            "max_per_dim_std": float(np.max(edge_matrix.std(axis=0))),
            "variance_sum": float(edge_matrix.var(axis=0).sum()),
            **_effective_rank(edge_matrix),
        },
        "branch_live": {
            "node_binding": bool(float(np.mean(node_binding)) > 1.0e-12),
            "edge_binding": bool(float(np.mean(edge_binding)) > 1.0e-12),
            "edge_update": bool(float(np.mean(edge_residual)) > 1.0e-12),
            "node_update": bool(float(np.mean(node_residual)) > 1.0e-12),
        },
        "all_branches_nonzero_gradient": bool(
            all(
                grads[key] > 0.0
                for key in (
                    "node_initializer",
                    "edge_initializer",
                    "node_binding",
                    "edge_binding",
                    "edge_update",
                    "node_update",
                    "readout",
                )
            )
        ),
        "official_test_loaded": False,
    }
    _write_json(
        RESULTS_DIR / f"diagnostics_{_tag(variant, tag)}_seed{seed}_{state}.json", payload
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def witness(
    variant: str,
    seed: int,
    tag: str | None = None,
    state: str = "soup",
    repeats: int = SHUFFLE_REPEATS,
) -> dict[str, Any]:
    """Evaluation-only atom and bond-type permutations (topology fixed)."""
    model = _load_frozen(variant, seed, tag, state=state)
    _train, valid_molecules, _scalers, _meta = build_datasets()
    batches = list(_loader(valid_molecules, 128, False, 0))
    targets: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    atom_perm_preds: list[list[np.ndarray]] = [[] for _ in range(int(repeats))]
    bond_perm_preds: list[list[np.ndarray]] = [[] for _ in range(int(repeats))]
    channel_deltas = {"atom_permutation": 0.0, "bond_permutation": 0.0}
    with torch.no_grad():
        for batch in batches:
            targets.append(batch.y.view(-1).numpy())
            actual.append(model(batch).view(-1).numpy())
            for repeat in range(int(repeats)):
                q_permuted = inc.permute_q_within_graphs(batch, seed=1000 * int(seed) + repeat)
                atom_batch = batch.with_q(q_permuted)
                atom_perm_preds[repeat].append(model(atom_batch).view(-1).numpy())
                r_permuted = inc.permute_bond_types_within_graphs(
                    batch, seed=1000 * int(seed) + repeat
                )
                bond_batch = batch.with_r(r_permuted)
                bond_perm_preds[repeat].append(model(bond_batch).view(-1).numpy())
                if repeat == 0:
                    channel_deltas["atom_permutation"] += float(
                        (model(atom_batch).view(-1) - model(batch).view(-1)).abs().sum()
                    )
                    channel_deltas["bond_permutation"] += float(
                        (model(bond_batch).view(-1) - model(batch).view(-1)).abs().sum()
                    )
    targets_all = np.concatenate(targets)
    actual_all = np.concatenate(actual)

    def _summary(perm_preds: list[list[np.ndarray]]) -> dict[str, Any]:
        arrays = [np.concatenate(items) for items in perm_preds]
        matrix = np.stack(arrays, axis=0)
        maes = [float(np.mean(np.abs(targets_all - values))) for values in arrays]
        return {
            "mean_of_permutation_maes": float(np.mean(maes)),
            "permutation_mae_std": float(np.std(maes)),
            "permutation_maes": maes,
            "mean_abs_prediction_change": float(
                np.mean(np.abs(matrix - actual_all[None, :]))
            ),
        }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": str(variant),
        "seed": int(seed),
        "state": str(state),
        "repeat_count": int(repeats),
        "actual_mae": float(np.mean(np.abs(targets_all - actual_all))),
        "atom_permutation": _summary(atom_perm_preds),
        "bond_permutation": _summary(bond_perm_preds),
        "channel_delta_abs_sum": channel_deltas,
        "bond_permutation_exactly_invariant": bool(
            float(channel_deltas["bond_permutation"]) == 0.0
        ),
        "official_test_loaded": False,
    }
    _write_json(
        RESULTS_DIR / f"witness_{_tag(variant, tag)}_seed{seed}_{state}.json", payload
    )
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


def synthetic_incidence_positive(n_examples: int = 600, seed: int = 0):
    """Incidence-only control: is the DOUBLE bond incident to the N atom?

    Fixed 4-leaf star (all leaves are structurally equivalent, all edges are
    structurally equivalent).  Exactly one leaf carries the N atom and exactly
    one edge carries the DOUBLE bond; the label is 1 iff they are the same
    leaf/edge.

    The node multiset (one N leaf, three C leaves, one C centre) and the edge
    multiset (one DOUBLE, three SINGLE) are **identical** for label 0 and
    label 1, so any model whose readout depends only on per-object S-A-B
    features plus permutation-invariant global pooling (C0) must predict a
    constant.  Only a processor that routes the bond type to its endpoint node
    (C1) can decide the label.
    """

    graph = from_edges(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    edges = sorted((int(u), int(v)) for u, v in graph.edges())
    rng = np.random.default_rng(int(seed))
    molecules: list[edge.EdgeMoleculeFeatures] = []
    for _ in range(int(n_examples)):
        atom_leaf = int(rng.integers(len(edges)))
        if bool(rng.integers(2)):
            double_leaf = atom_leaf
            label = 1.0
        else:
            choices = [index for index in range(len(edges)) if index != atom_leaf]
            double_leaf = int(rng.choice(choices))
            label = 0.0
        node_types = np.ones(5, dtype=np.int64)  # 1 = C
        node_types[1 + atom_leaf] = 0  # 0 = N at a leaf
        bond_type = np.full(len(edges), 1, dtype=np.int64)
        bond_type[double_leaf] = 2
        molecules.append(_toy_molecule(graph, node_types, edges, bond_type, label))
    rng.shuffle(molecules)
    return molecules


def synthetic_incidence_negative(n_examples: int = 600, seed: int = 0):
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
        rng.shuffle(molecule.atom_idx)
        order = rng.permutation(molecule.bond_type.shape[0])
        molecule.bond_type = molecule.bond_type[order]
        molecules.append(molecule)
    return molecules


def _train_toy(
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
    model = inc.build_incidence_model(variant, seed=int(seed))
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
        "binding_weight_norm": float(model.binding_weight_norm()),
    }


def synthetic(seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    positive_results: dict[str, list[dict[str, Any]]] = {}
    negative_results: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        positive = synthetic_incidence_positive(seed=int(seed))
        negative = synthetic_incidence_negative(seed=int(seed))
        for variant in inc.MODELS:
            positive_results.setdefault(variant, []).append(
                _train_toy(positive, variant, seed=int(seed), epochs=250)
            )
            negative_results.setdefault(variant, []).append(
                _train_toy(negative, variant, seed=int(seed), epochs=250)
            )

    def _summary(rows):
        return {
            variant: {
                "valid_mae_mean": float(np.mean([row["valid_mae"] for row in values])),
                "valid_mae_values": [float(row["valid_mae"]) for row in values],
            }
            for variant, values in rows.items()
        }

    positive_summary = _summary(positive_results)
    negative_summary = _summary(negative_results)
    positive_pass = bool(
        positive_summary["C1"]["valid_mae_mean"] + 0.05 < positive_summary["C0"]["valid_mae_mean"]
    )
    negative_pass = bool(
        abs(
            negative_summary["C1"]["valid_mae_mean"] - negative_summary["C0"]["valid_mae_mean"]
        )
        < 0.05
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seeds": [int(seed) for seed in seeds],
        "positive": {
            "per_run": positive_results,
            "summary": positive_summary,
            "pass": positive_pass,
            "criterion": "C1 valid MAE beats C0 by > 0.05 on an edge-assignment-only task",
        },
        "negative": {
            "per_run": negative_results,
            "summary": negative_summary,
            "pass": negative_pass,
            "criterion": "C1 and C0 are within 0.05 on a marginal-only task",
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "synthetic_controls.json", payload)
    print(json.dumps(payload["positive"]["summary"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(payload["negative"]["summary"], indent=2, sort_keys=True), flush=True)
    if not positive_pass:
        raise RuntimeError("synthetic incidence positive control FAILED: do not launch ZINC")
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
    """Pre-registered seed0 gate for C1 (see preregistration §6)."""
    c1 = _run_soup_valid(_tag("C1"), 0)
    if c1 is None:
        call = "INCOMPLETE"
    elif c1 > GATE_STOP_ABOVE:
        call = "STOP"
    elif c1 > GATE_INSPECT_ABOVE:
        call = "INSPECT"
    elif c1 <= GATE_STRONG_AT_OR_BELOW:
        call = "STRONG"
    else:
        call = "SEED1_WORTHY"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "C1_seed0_soup_valid": c1,
        "references": REFERENCES,
        "delta_vs_B_Full_seed0": (None if c1 is None else c1 - REFERENCES["B_Full_soup_seed0"]),
        "delta_vs_B_Bag_seed0": (None if c1 is None else c1 - REFERENCES["B_Bag_soup_seed0"]),
        "delta_vs_AR0_edge_BVE_seed0": (
            None if c1 is None else c1 - REFERENCES["ar0e_BVE_assign_soup_seed0"]
        ),
        "thresholds": {
            "STOP_above": GATE_STOP_ABOVE,
            "INSPECT_above": GATE_INSPECT_ABOVE,
            "STRONG_at_or_below": GATE_STRONG_AT_OR_BELOW,
        },
        "call": call,
        "c0_authorized": bool(call not in {"STOP", "INCOMPLETE"}),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "gate.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def decide() -> dict[str, Any]:
    gate_payload = gate()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": gate_payload["call"],
        "C1_seed0_soup_valid": gate_payload["C1_seed0_soup_valid"],
        "C0_seed0_soup_valid": _run_soup_valid(_tag("C0"), 0),
        "references": REFERENCES,
        "gate": gate_payload,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def report() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in inc.MODELS:
        for seed in (0, 1):
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
                    "params": run_payload.get("parameters"),
                    "wall_time_s": run_payload.get("wall_clock_s"),
                    "peak_gpu_memory_bytes": run_payload.get("peak_gpu_memory_bytes"),
                }
            )
    lines = [
        "# FSAR-C1 incidence processor — results",
        "",
        "| variant | seed | best valid | Top-5 soup valid | best ep | params | wall (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {seed} | {best_valid} | {soup_valid} | {best_epoch} | "
            "{params} | {wall_time_s} |".format(**row)
        )
    lines += [
        "",
        "Frozen strong references (valid Top-5 soup):",
        "",
        f"* B-Full seed0 {REFERENCES['B_Full_soup_seed0']:.6f} / seed1 {REFERENCES['B_Full_soup_seed1']:.6f}",
        f"* B-Bag seed0 {REFERENCES['B_Bag_soup_seed0']:.6f} / seed1 {REFERENCES['B_Bag_soup_seed1']:.6f}",
        f"* cell A 2-seed soup mean {REFERENCES['cell_A_soup_2seed_mean']:.6f}",
        f"* AR0-edge BVE seed0 {REFERENCES['ar0e_BVE_assign_soup_seed0']:.6f}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "FORMAL_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return {"markdown": text, "rows": rows, "official_test_loaded": False}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("params")
    p = sub.add_parser("preprocess")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("sanity")
    p = sub.add_parser("synthetic")
    p = sub.add_parser("gate")
    p = sub.add_parser("decide")
    p = sub.add_parser("report")
    p = sub.add_parser("gradient_audit")
    p.add_argument("--variant", default="C1", choices=list(inc.MODELS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--deterministic", action="store_true")
    p = sub.add_parser("smoke")
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--deterministic", action="store_true")
    for stage in ("train", "soup", "run", "diagnostics", "witness"):
        p = sub.add_parser(stage)
        p.add_argument("--variant", required=True, choices=list(inc.MODELS))
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
    elif arguments.stage == "sanity":
        sanity()
    elif arguments.stage == "synthetic":
        synthetic()
    elif arguments.stage == "gradient_audit":
        if arguments.all:
            gradient_audit_all(device=arguments.device)
        else:
            gradient_audit(arguments.variant, device=arguments.device)
    elif arguments.stage == "smoke":
        smoke(device=arguments.device, steps=int(arguments.steps))
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
        diagnostics(arguments.variant, arguments.seed, tag=arguments.tag, state=arguments.state)
    elif arguments.stage == "witness":
        witness(arguments.variant, arguments.seed, tag=arguments.tag, state=arguments.state)
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
