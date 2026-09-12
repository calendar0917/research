"""Canonical Late-Readout Adaptation Confirmation (compact-v4-hinge, ZINC).

This experiment tests a **training-dynamics / representation-readout
optimisation-coupling** hypothesis, not a representation-capacity question.

After canonical compact-v4 joint training finishes (frozen selected
checkpoint), is there a leak-free way to improve official-valid MAE by

* freezing every parameter that produces the final graph representation
  ``R`` and
* continuing to train only the *existing* graph head,

and is that improvement larger than the improvement obtained by spending the
**same** extra optimisation budget on a full-model continuation?

Primary matrix (same starting checkpoint, same train data, same optimizer
reset, same minibatch order, same step count):

* ``B0`` -- stop: the canonical selected checkpoint, no further training;
* ``C``  -- full continuation: backbone + graph head all trainable;
* ``E``  -- head-only adaptation: all R-producing modules frozen, only the
  original graph head (Linear + LayerNorm affine) trainable.

The decisive mechanism comparison is ``Delta_F = MAE(C) - MAE(E)`` (positive
=> freezing the representation helps), never ``B0`` vs ``E`` alone.

Leakage safety: official test is never loaded; official valid is never used
to update parameters, pick an epoch, tune a learning rate, early-stop, or
select a parameter set.  The adaptation horizon ``K*`` is preregistered from
the *already completed* OOF late-refit audit (median best-selection epoch of
the L-warm OOF runs), before any canonical-valid number is seen.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_canonical_late_readout_adaptation <stage>

Stages: ``inventory lock prepare seed0 seed1 summary figures all``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Batch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
    V4_RUN_IDS,
    _build_v4_model,
    _extract_v4_records,
    load_run_result,
    run_path,
)

# ---------------------------------------------------------------------------
# frozen locations / protocol constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/canonical_late_readout_adaptation"
CACHE_DIR = RESULTS_DIR / "cache"
FIG_DIR = RESULTS_DIR / "figures"

CANONICAL_CONFIG_PATH = (
    REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
)
OOF_FOLD_RESULTS = (
    REPO_ROOT
    / "tracks/ksvd/results/graph_head_refit_capacity_decomposition/fold_results.csv"
)
OOF_FINAL_DECISION = (
    REPO_ROOT
    / "tracks/ksvd/results/graph_head_refit_capacity_decomposition/final_decision.json"
)

PROTOCOL_VERSION = "canonical_late_readout_adaptation_v1"

# backbone seeds of the canonical compact-v4-hinge multi-seed confirmation
BACKBONE_SEEDS = (0, 1)
PRIMARY_SEED = 0
R_DIM = 302

# --- adaptation optimiser / schedule (pre-registered) -----------------------
ADAPT_LR = 1.0e-3
ADAPT_WEIGHT_DECAY = 0.0
ADAPT_BATCH_SIZE = 512
BATCH_ORDER_SEED = 0
STANDARDIZE_EPS = 1.0e-6

# --- adaptation horizon -----------------------------------------------------
# K* is read from the completed OOF late-refit audit, NOT from canonical valid.
K_STAR_COLUMN = "best_epoch_lwarm"
K_STAR_RULE = "round_half_up(median(best_epoch_lwarm over 2 seeds x 5 folds))"
K_STAR_FALLBACK = 800  # task section 12: historical protocol horizon if unrecoverable

# --- equivalence tolerances -------------------------------------------------
REPARAM_ATOL = 1.0e-4  # float32 deployment of the exact affine reparameterisation
STEP0_ATOL = 1.0e-4
FROZEN_R_ATOL = 1.0e-9

# --- probe subset for representation-drift diagnostics ----------------------
DRIFT_PROBE_N = 2000
TRACE_EPOCH_DIVISORS = (1, 10, 25, 50, 75, 100)  # -> int(K* * d / 100)

# --- pre-registered decision thresholds (task sections 29-38) ---------------
THRESHOLD_ADAPT_GAIN = 0.003
THRESHOLD_FREEZE_GAIN = 0.0015
THRESHOLD_CLEAR_NO_GO = 0.0015
THRESHOLD_EXTRA_TRAINING_BAND = 0.001
THRESHOLD_FULL_CONT_BETTER = 0.0015
THRESHOLD_FINAL_NO_GO_MEAN = 0.002

# ---------------------------------------------------------------------------
# small io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    head = REPO_ROOT / ".git" / "HEAD"
    if head.exists():
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = REPO_ROOT / ".git" / text.split(" ", 1)[1].strip()
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()
        return text
    return "unknown"


# ---------------------------------------------------------------------------
# checkpoint inventory / canonical run resolution
# ---------------------------------------------------------------------------


def canonical_run(seed: int) -> dict[str, Any]:
    """Resolve the canonical compact-v4-hinge selected checkpoint for a seed."""
    run_id = V4_RUN_IDS[int(seed)]
    run_dir = run_path(run_id)
    state_path = run_dir / "artifacts" / "legacy_full_result_selection_state.pt"
    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    result = load_run_result(run_id)
    return {
        "seed": int(seed),
        "run_id": run_id,
        "run_dir": run_dir,
        "state_path": state_path,
        "config": config,
        "result": result,
    }


def checkpoint_inventory(
    seeds: Sequence[int] = BACKBONE_SEEDS, write: bool = True
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for seed in seeds:
        info = canonical_run(seed)
        state_path = info["state_path"]
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        config = info["config"]
        result = info["result"]
        evaluation = result["evaluation"]["valid"]
        entries.append(
            {
                "seed": int(seed),
                "run_id": info["run_id"],
                "state_path": str(state_path),
                "state_sha256": _sha256_file(state_path),
                "state_keys": len(state),
                "parameters": int(sum(p.numel() for p in state.values())),
                "config_hash": info["result"].get("config_hash"),
                "protocol_id": config.get("protocol_id"),
                "typed_tokenizer_version": config.get("representation", {}).get(
                    "typed_tokenizer_version"
                ),
                "topology_mode": config.get("model", {}).get("topology_mode"),
                "graph_head_hidden_0": config.get("model", {}).get("graph_head_hidden_0"),
                "graph_head_hidden_1": config.get("model", {}).get("graph_head_hidden_1"),
                "dropout": config.get("model", {}).get("dropout"),
                "selected_epoch": int(evaluation["selected_epoch"]),
                "canonical_valid_mae": float(evaluation["mae"]),
                "canonical_valid_best_mae": float(evaluation["best_mae"]),
                "official_test_loaded": False,
            }
        )
    inventory = {
        "protocol_version": PROTOCOL_VERSION,
        "canonical_config": str(CANONICAL_CONFIG_PATH),
        "canonical_config_sha256": _sha256_file(CANONICAL_CONFIG_PATH),
        "entries": entries,
        "complete_seeds": [e["seed"] for e in entries],
        "note": (
            "Selected validation checkpoints reused from the canonical "
            "compact-v4-hinge multi-seed confirmation; no baseline retrained. "
            "official test is never loaded by this experiment."
        ),
    }
    if write:
        _write_json(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    return inventory


# ---------------------------------------------------------------------------
# adaptation-budget preregistration
# ---------------------------------------------------------------------------


def _read_oof_best_epochs() -> list[int]:
    import csv

    rows = list(csv.DictReader(open(OOF_FOLD_RESULTS, encoding="utf-8")))
    return [int(row[K_STAR_COLUMN]) for row in rows]


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def compute_k_star() -> dict[str, Any]:
    """Derive K* from the completed OOF L-warm late-refit audit.

    Per task sections 11-12: read the per-fold best-selection epoch of the
    L-warm OOF runs (2 backbone seeds x 5 folds); K* is the deterministic
    integer median.  If the per-fold epochs are not recoverable, fall back to
    the already-used 800-epoch protocol horizon.  Official valid is never used.
    """
    if not OOF_FOLD_RESULTS.exists():
        return {
            "source": "fallback",
            "k_star": K_STAR_FALLBACK,
            "reason": "OOF fold_results.csv not present; use 800-epoch protocol horizon",
            "best_epochs": None,
        }
    best_epochs = _read_oof_best_epochs()
    if not best_epochs:
        return {
            "source": "fallback",
            "k_star": K_STAR_FALLBACK,
            "reason": "OOF best epochs empty; use 800-epoch protocol horizon",
            "best_epochs": [],
        }
    median = float(statistics.median(best_epochs))
    return {
        "source": "oof_late_refit_best_selection_epoch",
        "k_star_column": K_STAR_COLUMN,
        "best_epochs": sorted(int(value) for value in best_epochs),
        "median": median,
        "rule": K_STAR_RULE,
        "k_star": _round_half_up(median),
        "fallback": K_STAR_FALLBACK,
    }


def protocol_lock(write: bool = True) -> dict[str, Any]:
    """Write adaptation_protocol_lock.json BEFORE any canonical-valid evaluation."""
    k_star = compute_k_star()
    oof_decision = _read_json(OOF_FINAL_DECISION)
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_audit": {
            "git_commit": _git_commit(),
            "protocol_version": PROTOCOL_VERSION,
            "canonical_config": str(CANONICAL_CONFIG_PATH),
            "canonical_config_sha256": _sha256_file(CANONICAL_CONFIG_PATH),
        },
        "oof_result_fingerprint": {
            "fold_results_csv": str(OOF_FOLD_RESULTS),
            "fold_results_sha256": _sha256_file(OOF_FOLD_RESULTS),
            "final_decision_json": str(OOF_FINAL_DECISION),
            "final_decision_sha256": _sha256_file(OOF_FINAL_DECISION),
            "oof_verdict": oof_decision.get("verdict"),
            "oof_pooled_mean_delta_w": oof_decision.get("pooled_mean_delta_w"),
        },
        "adaptation_budget": k_star,
        "optimizer": {
            "name": "Adam",
            "learning_rate": ADAPT_LR,
            "weight_decay": ADAPT_WEIGHT_DECAY,
            "fresh_state_for_C_and_E": True,
            "resume_old_optimizer_state": False,
        },
        "batch_size": ADAPT_BATCH_SIZE,
        "data_order_seed": BATCH_ORDER_SEED,
        "torch_global_seed": BATCH_ORDER_SEED,
        "head_dropout_active_during_adaptation": True,
        "loss": "L1 / mean absolute error",
        "standardization": {
            "rule": "z = (R - mu) / max(sigma, 1e-6)",
            "fit_scope": "official train R at the starting canonical checkpoint",
            "statistics_frozen_during_adaptation": True,
            "equal_for_C_and_E": True,
            "degenerate_handling": "max(sigma, eps); reparameterisation stays exact",
        },
        "first_layer_reparameterization": {
            "weight": "W' = W * diag(scale)",
            "bias": "b' = b + W @ mu",
            "purpose": "step 0 function equivalence with B0",
        },
        "trainable_parameter_set": {
            "B0": "none (no training)",
            "C": "all canonical model parameters",
            "E": "original graph head only (Linear + LayerNorm affine)",
        },
        "adaptation_uses_official_train_only": True,
        "official_valid_used_during_adaptation": False,
        "official_test_loaded": False,
        "decision_thresholds": {
            "strong_advance_seed0": {
                "delta_A_min": THRESHOLD_ADAPT_GAIN,
                "delta_F_min": THRESHOLD_FREEZE_GAIN,
            },
            "clear_no_go_seed0": {"delta_A_max": THRESHOLD_CLEAR_NO_GO},
            "adaptation_only_advance": {
                "delta_A_min": THRESHOLD_ADAPT_GAIN,
                "delta_F_range": [0.0, THRESHOLD_FREEZE_GAIN],
            },
            "extra_training_band": {"abs_delta_F_max": THRESHOLD_EXTRA_TRAINING_BAND},
            "full_continuation_better": {"C_lt_E_by": THRESHOLD_FULL_CONT_BETTER},
            "final_go": {
                "mean_delta_A_min": THRESHOLD_ADAPT_GAIN,
                "mean_delta_F_min": THRESHOLD_FREEZE_GAIN,
                "per_seed_delta_F_min": 0.0,
            },
            "final_no_go_mean_delta_A_max": THRESHOLD_FINAL_NO_GO_MEAN,
        },
        "seeds_authorized": list(BACKBONE_SEEDS),
        "primary_seed": PRIMARY_SEED,
    }
    if write:
        _write_json(RESULTS_DIR / "adaptation_protocol_lock.json", lock)
    return lock


# ---------------------------------------------------------------------------
# data + model preparation
# ---------------------------------------------------------------------------


def build_data(config: Mapping[str, Any]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train_records, valid_records = _extract_v4_records()
    train_data, valid_data, audit = zpp._phase_data(
        train_records, valid_records, config=config
    )
    return train_data, valid_data, audit


def load_canonical_model(
    config: Mapping[str, Any], audit: Mapping[str, Any], run_dir: Path
) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    model = _build_v4_model(config, audit)
    state = torch.load(
        run_dir / "artifacts" / "legacy_full_result_selection_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    return model, state


def _batch_graphs(graphs: Sequence[Any], indices: Sequence[int]) -> Any:
    return Batch.from_data_list([graphs[int(i)] for i in indices])


def encode_representation(
    model: nn.Module, graphs: Sequence[Any], batch_size: int = 512
) -> np.ndarray:
    """Frozen ``R`` for a list of encoded graphs (eval mode, no grad)."""
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(graphs), int(batch_size)):
            batch = _batch_graphs(graphs, range(start, min(start + batch_size, len(graphs))))
            chunks.append(model.encode(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def predict_model(model: nn.Module, graphs: Sequence[Any], batch_size: int = 512) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(graphs), int(batch_size)):
            batch = _batch_graphs(graphs, range(start, min(start + batch_size, len(graphs))))
            chunks.append(model(batch).view(-1).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def targets_of(graphs: Sequence[Any]) -> np.ndarray:
    return np.asarray([float(graph.y.view(-1)[0]) for graph in graphs], dtype=np.float64)


# ---------------------------------------------------------------------------
# fit-only standardisation + exact first-layer reparameterisation
# ---------------------------------------------------------------------------


def fit_standardizer(R_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(R_train, dtype=np.float64).mean(axis=0)
    std = np.asarray(R_train, dtype=np.float64).std(axis=0)
    scale = np.maximum(std, STANDARDIZE_EPS)
    return mean, scale


def reparameterize_head(
    head: nn.Sequential, mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact affine reparameterisation ``W' = W diag(scale)``, ``b' = b + W mean``.

    With ``z = (R - mean) / scale`` this gives ``W' z + b' = W R + b`` exactly
    (for ``scale = max(std, eps)``), so the reparameterised head is the *same
    function* as the canonical head at optimisation step 0.
    """
    weight = head[0].weight.detach().cpu().numpy().astype(np.float64)
    bias = head[0].bias.detach().cpu().numpy().astype(np.float64)
    mean64 = np.asarray(mean, dtype=np.float64)
    scale64 = np.asarray(scale, dtype=np.float64)
    new_weight = weight * scale64[None, :]
    new_bias = bias + weight @ mean64
    with torch.no_grad():
        head[0].weight.copy_(torch.tensor(new_weight, dtype=head[0].weight.dtype))
        head[0].bias.copy_(torch.tensor(new_bias, dtype=head[0].bias.dtype))
    return new_weight, new_bias


def reparameterized_head_reference(
    weight: np.ndarray, bias: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    return (
        w * np.asarray(scale, dtype=np.float64)[None, :],
        b + w @ np.asarray(mean, dtype=np.float64),
    )


class StandardizedReadout(nn.Module):
    """Wraps the canonical model with a *fixed* affine standardiser.

    ``forward`` returns ``head((R - mean) / scale)`` where ``R`` is the frozen
    (or trainable, for C) representation from ``backbone.encode``.  Only the
    fixed buffers are added; the graph head itself is unchanged.
    """

    def __init__(self, backbone: nn.Module, mean: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer(
            "standardizer_mean", torch.tensor(mean, dtype=torch.float32)
        )
        self.register_buffer(
            "standardizer_scale", torch.tensor(scale, dtype=torch.float32)
        )

    def forward(self, data: Any) -> torch.Tensor:
        representation = self.backbone.encode(data)
        standardized = (
            representation - self.standardizer_mean
        ) / self.standardizer_scale
        return self.backbone.head(standardized).view(-1)


HEAD_PREFIX = "backbone.head."


def head_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if name.startswith(HEAD_PREFIX)]


def set_head_only_trainable(model: nn.Module) -> list[str]:
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        allowed = name.startswith(HEAD_PREFIX)
        parameter.requires_grad_(allowed)
        if allowed:
            trainable.append(name)
    return trainable


def _fresh_wrapper(
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
    run_dir: Path,
    mean: np.ndarray,
    scale: np.ndarray,
) -> StandardizedReadout:
    model, _state = load_canonical_model(config, audit, run_dir)
    reparameterize_head(model.head, mean, scale)
    return StandardizedReadout(model, mean, scale)


# ---------------------------------------------------------------------------
# adaptation loop (official train only; no valid loader parameter)
# ---------------------------------------------------------------------------


def _trace_epochs(k_star: int) -> list[int]:
    epochs = sorted(
        {
            max(1, min(int(k_star), int(round(k_star * divisor / 100.0))))
            for divisor in TRACE_EPOCH_DIVISORS
        }
    )
    return epochs


def run_adaptation(
    model: StandardizedReadout,
    train_graphs: Sequence[Any],
    *,
    epochs: int,
    mode: str,
    batch_size: int = ADAPT_BATCH_SIZE,
    lr: float = ADAPT_LR,
    weight_decay: float = ADAPT_WEIGHT_DECAY,
    order_seed: int = BATCH_ORDER_SEED,
    trace_epochs: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Deterministic mini-batch L1 training on official train only.

    ``mode`` is ``"C"`` (backbone + head trainable, full model in train mode)
    or ``"E"`` (R frozen: backbone eval, only the head trainable and dropout
    active in the head).  This function deliberately has **no** valid loader
    argument: official valid cannot influence adaptation.
    """
    if mode not in ("C", "E"):
        raise ValueError(f"unknown adaptation mode {mode!r}")
    trainable_names = (
        set_head_only_trainable(model) if mode == "E" else
        [name for name, _ in model.named_parameters()]
    )
    for parameter in model.parameters():
        if mode == "C":
            parameter.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(lr), weight_decay=float(weight_decay))

    if trace_epochs is None:
        trace_epochs = _trace_epochs(int(epochs))
    trace_set = {int(epoch) for epoch in trace_epochs}

    n = len(train_graphs)
    # Reproducibility: the explicit generator fixes the mini-batch order and
    # the global seed fixes the head dropout masks, so a serial run is
    # bit-reproducible (see notes/reproducibility_cpu_determinism.md).
    torch.manual_seed(int(order_seed))
    generator = torch.Generator().manual_seed(int(order_seed))
    curve: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(n, generator=generator).tolist()
        if mode == "E":
            # R is frozen: backbone runs in eval mode (no dropout); the head
            # keeps its canonical training-mode dropout.
            model.eval()
            model.backbone.head.train()
        else:
            model.train()
        total_loss = 0.0
        seen = 0
        first_batch_indices: list[int] | None = None
        for start in range(0, n, int(batch_size)):
            indices = permutation[start : start + int(batch_size)]
            if first_batch_indices is None:
                first_batch_indices = list(indices)
            batch = _batch_graphs(train_graphs, indices)
            target = batch.y.view(-1)
            prediction = model(batch)
            loss = (prediction - target).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        row: dict[str, Any] = {
            "variant": mode,
            "epoch": int(epoch),
            "train_loss": total_loss / max(seen, 1),
            "first_batch_indices": first_batch_indices,
        }
        if epoch in trace_set:
            model.eval()
            with torch.no_grad():
                prediction = predict_model(model, train_graphs, batch_size)
            row["train_mae_eval"] = float(
                mean_absolute_error(targets_of(train_graphs), prediction)
            )
        curve.append(row)

    model.eval()
    return {
        "mode": mode,
        "epochs": int(epochs),
        "trainable_parameters": sorted(trainable_names),
        "trainable_parameter_count": int(len(trainable_names)),
        "trainable_numel": int(sum(p.numel() for p in parameters)),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "curve": curve,
    }


def evaluate_valid(
    model: StandardizedReadout, valid_graphs: Sequence[Any], batch_size: int = ADAPT_BATCH_SIZE
) -> float:
    """Single post-adaptation official-valid evaluation (never used to select)."""
    prediction = predict_model(model, valid_graphs, batch_size)
    return float(mean_absolute_error(targets_of(valid_graphs), prediction))


# ---------------------------------------------------------------------------
# per-seed runner
# ---------------------------------------------------------------------------


def _tensor_fingerprint(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return {"sha256": digest.hexdigest()}


def _head_movement(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    names = [name for name in before if name.startswith(HEAD_PREFIX)]
    start = torch.cat([before[name].detach().cpu().reshape(-1) for name in names])
    end = torch.cat([after[name].detach().cpu().reshape(-1) for name in names])
    delta = (end - start).norm().item()
    start_norm = start.norm().item()
    return {
        "head_param_l2_delta": float(delta),
        "head_param_l2_start": float(start_norm),
        "head_param_l2_relative": float(delta / max(start_norm, 1e-12)),
    }


def _prediction_movement(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    delta = np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)
    return {
        "std_prediction_change": float(delta.std()),
        "mean_abs_prediction_change": float(np.abs(delta).mean()),
        "mean_prediction_change": float(delta.mean()),
    }


def _representation_drift(
    probe_start: np.ndarray, probe_after: np.ndarray
) -> dict[str, float]:
    delta = np.asarray(probe_after, dtype=np.float64) - np.asarray(probe_start, dtype=np.float64)
    l2 = np.linalg.norm(delta, axis=1)
    start_norm = np.linalg.norm(probe_start, axis=1)
    cosine = np.sum(probe_start * probe_after, axis=1) / np.maximum(
        start_norm * np.linalg.norm(probe_after, axis=1), 1e-12
    )
    return {
        "mean_l2_drift": float(l2.mean()),
        "normalized_l2_drift": float(l2.mean() / max(start_norm.mean(), 1e-12)),
        "mean_cosine_similarity": float(cosine.mean()),
        "max_abs_drift": float(np.abs(delta).max()),
    }


def prepare_seed(seed: int, force: bool = False) -> dict[str, Any]:
    """Build data, R, standardisation, step-0 equivalence, trainable audit."""
    info = canonical_run(seed)
    config = info["config"]
    run_dir = info["run_dir"]
    train_data, valid_data, audit = build_data(config)
    y_valid = targets_of(valid_data)

    cache_path = CACHE_DIR / f"seed{seed}_prepare.npz"
    base_model, _state = load_canonical_model(config, audit, run_dir)
    # raw canonical head parameters, captured before reparameterisation
    original_weight = base_model.head[0].weight.detach().cpu().numpy().astype(np.float64)
    original_bias = base_model.head[0].bias.detach().cpu().numpy().astype(np.float64)

    if cache_path.exists() and not force:
        payload = np.load(cache_path, allow_pickle=True)
        R_train = payload["R_train"]
        R_valid = payload["R_valid"]
        mean = payload["mean"]
        scale = payload["scale"]
        b0_valid = float(payload["b0_valid"])
        b0_valid_pred = payload["b0_valid_pred"]
        config_hash = str(payload["config_hash"])
        state_hash = str(payload["state_hash"])
    else:
        R_train = encode_representation(base_model, train_data)
        R_valid = encode_representation(base_model, valid_data)
        mean, scale = fit_standardizer(R_train)
        b0_valid_pred = predict_model(base_model, valid_data)
        b0_valid = float(mean_absolute_error(y_valid, b0_valid_pred))
        config_hash = _sha256_file(run_dir / "config.resolved.yaml")
        state_hash = _sha256_file(info["state_path"])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            R_train=R_train.astype(np.float32),
            R_valid=R_valid.astype(np.float32),
            mean=mean,
            scale=scale,
            b0_valid=b0_valid,
            b0_valid_pred=b0_valid_pred.astype(np.float64),
            config_hash=config_hash,
            state_hash=state_hash,
        )

    # B0 must reproduce the canonical run's own valid MAE bit-for-bit.
    canonical_valid = float(info["result"]["evaluation"]["valid"]["mae"])
    canonical_best = float(info["result"]["evaluation"]["valid"]["best_mae"])

    raw_train = predict_model(base_model, train_data)
    raw_valid = predict_model(base_model, valid_data)

    # exact first-layer reparameterisation + step-0 equivalence (real path)
    reparameterize_head(base_model.head, mean, scale)
    wrapper = StandardizedReadout(base_model, mean, scale)
    step0_valid = predict_model(wrapper, valid_data)
    step0_train = predict_model(wrapper, train_data)

    # float64 algebraic identity on a fixed sample of the official train R
    sample = np.asarray(R_train[:256], dtype=np.float64)
    z_sample = (sample - mean) / scale
    new_weight, new_bias = reparameterized_head_reference(
        original_weight, original_bias, mean, scale
    )
    reference_original = sample @ original_weight.T + original_bias
    reference_transformed = z_sample @ new_weight.T + new_bias
    equivalence = {
        "valid_max_abs_diff_original_vs_reparameterized": float(
            np.abs(step0_valid - raw_valid).max()
        ),
        "train_max_abs_diff_original_vs_reparameterized": float(
            np.abs(step0_train - raw_train).max()
        ),
        "atol": STEP0_ATOL,
        "passed": bool(
            np.abs(step0_valid - raw_valid).max() <= STEP0_ATOL
            and np.abs(step0_train - raw_train).max() <= STEP0_ATOL
        ),
        "float64_identity_max_abs_diff": float(
            np.abs(reference_transformed - reference_original).max()
        ),
        "n_identity_samples": int(sample.shape[0]),
    }

    standardization_payload = {
        "seed": int(seed),
        "eps": STANDARDIZE_EPS,
        "fit_scope": "official train R at the canonical selected checkpoint (10000 molecules)",
        "rule": "z = (R - mu) / max(sigma, 1e-6)",
        "n_train": int(len(train_data)),
        "n_valid": int(len(valid_data)),
        "R_dim": int(R_train.shape[1]),
        "mean_abs_mean": float(np.abs(mean).mean()),
        "mean_abs_std": float(np.abs(scale).mean()),
        "n_degenerate": int(np.sum(scale <= STANDARDIZE_EPS)),
        "degenerate_indices": [int(i) for i in np.flatnonzero(scale <= STANDARDIZE_EPS)],
        "statistics_frozen": True,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }

    prepare = {
        "seed": int(seed),
        "run_id": info["run_id"],
        "config_hash": config_hash,
        "state_sha256": state_hash,
        "b0_valid_mae": float(b0_valid),
        "b0_valid_mae_recomputed": float(mean_absolute_error(y_valid, raw_valid)),
        "canonical_valid_mae": canonical_valid,
        "canonical_valid_best_mae": canonical_best,
        "b0_matches_canonical_valid": bool(abs(b0_valid - canonical_valid) <= 1e-9),
        "step0_equivalence": equivalence,
        "standardization": standardization_payload,
        "n_train": int(len(train_data)),
        "n_valid": int(len(valid_data)),
    }
    return prepare


def run_seed(seed: int, force: bool = False) -> dict[str, Any]:
    """Run B0 / C / E for one canonical seed under the frozen protocol."""
    lock = _read_json(RESULTS_DIR / "adaptation_protocol_lock.json")
    k_star = int(lock["adaptation_budget"]["k_star"])
    info = canonical_run(seed)
    config = info["config"]
    run_dir = info["run_dir"]
    train_data, valid_data, audit = build_data(config)

    payload = np.load(CACHE_DIR / f"seed{seed}_prepare.npz", allow_pickle=True)
    R_train = np.asarray(payload["R_train"], dtype=np.float64)
    mean = np.asarray(payload["mean"], dtype=np.float64)
    scale = np.asarray(payload["scale"], dtype=np.float64)
    y_train = targets_of(train_data)
    y_valid = targets_of(valid_data)
    base_model, _state = load_canonical_model(config, audit, run_dir)
    b0_valid_pred = predict_model(base_model, valid_data)
    b0_valid = float(mean_absolute_error(y_valid, b0_valid_pred))
    b0_train = float(mean_absolute_error(y_train, predict_model(base_model, train_data)))
    del base_model

    probe_index = np.arange(min(DRIFT_PROBE_N, len(train_data)), dtype=np.int64)
    R_probe_start = R_train[probe_index]
    probe_graphs = [train_data[int(i)] for i in probe_index]

    branches: dict[str, Any] = {}
    for mode in ("C", "E"):
        started = time.perf_counter()
        wrapper = _fresh_wrapper(config, audit, run_dir, mean, scale)
        start_state = {
            name: parameter.detach().clone()
            for name, parameter in wrapper.named_parameters()
        }
        start_valid_pred = predict_model(wrapper, valid_data)
        start_train_probe_pred = predict_model(wrapper, probe_graphs)
        start_train_mae = float(
            mean_absolute_error(y_train, predict_model(wrapper, train_data))
        )
        start_R_probe = encode_representation(wrapper.backbone, probe_graphs)

        result = run_adaptation(
            wrapper,
            train_data,
            epochs=k_star,
            mode=mode,
            trace_epochs=_trace_epochs(k_star),
        )

        end_state = {
            name: parameter.detach().clone()
            for name, parameter in wrapper.named_parameters()
        }
        final_valid_pred = predict_model(wrapper, valid_data)
        final_train_pred = predict_model(wrapper, train_data)
        final_train_mae = float(mean_absolute_error(y_train, final_train_pred))
        final_valid_mae = float(mean_absolute_error(y_valid, final_valid_pred))
        end_R_probe = encode_representation(wrapper.backbone, probe_graphs)

        finished = time.perf_counter()
        diagnostics = {
            "seed": int(seed),
            "variant": mode,
            "k_star": int(k_star),
            "trainable_parameters": result["trainable_parameters"],
            "trainable_parameter_count": result["trainable_parameter_count"],
            "total_parameter_count": result["total_parameter_count"],
            "optimizer_parameter_names": result["trainable_parameters"],
            "optimizer_only_head": bool(
                all(name.startswith(HEAD_PREFIX) for name in result["trainable_parameters"])
            )
            and mode == "E",
            "starting_train_mae": start_train_mae,
            "final_train_mae": final_train_mae,
            "b0_train_mae": b0_train,
            "valid_mae": final_valid_mae,
            "b0_valid_mae": b0_valid,
            "head_parameter_movement": _head_movement(start_state, end_state),
            "prediction_movement_train": _prediction_movement(
                start_train_probe_pred, predict_model(wrapper, probe_graphs)
            ),
            "prediction_movement_valid": _prediction_movement(
                start_valid_pred, final_valid_pred
            ),
            "representation_drift": _representation_drift(start_R_probe, end_R_probe),
            "representation_frozen_check": {
                "max_abs_representation_change": float(
                    np.abs(end_R_probe - R_probe_start).max()
                ),
                "max_abs_change_after_vs_start_checkpoint": float(
                    np.abs(start_R_probe - R_probe_start).max()
                ),
                "equal": bool(
                    np.allclose(
                        end_R_probe, start_R_probe, atol=FROZEN_R_ATOL, rtol=0.0
                    )
                ),
            },
            "batch_order_first_epoch": result["curve"][0]["first_batch_indices"][:8],
            "epochs_run": int(result["epochs"]),
            "seconds": float(finished - started),
        }
        branches[mode] = {
            "diagnostics": diagnostics,
            "curve": result["curve"],
            "valid_predictions": final_valid_pred.tolist(),
        }

        # persist the adapted state for reproducibility
        torch.save(end_state, CACHE_DIR / f"seed{seed}_{mode}_adapted_state.pt")
        if mode == "E":
            frozen_ok = diagnostics["representation_frozen_check"]["equal"]
            if not frozen_ok:
                raise AssertionError(
                    f"seed {seed} E: frozen representation changed during adaptation"
                )
        print(
            f"[seed{seed} {mode}] valid_mae={final_valid_mae:.6f} "
            f"train_mae={final_train_mae:.6f} ({finished - started:.0f}s)",
            flush=True,
        )

    results = {
        "seed": int(seed),
        "k_star": int(k_star),
        "b0_valid_mae": b0_valid,
        "c_valid_mae": branches["C"]["diagnostics"]["valid_mae"],
        "e_valid_mae": branches["E"]["diagnostics"]["valid_mae"],
        "delta_A_B0_minus_E": float(b0_valid - branches["E"]["diagnostics"]["valid_mae"]),
        "delta_C_B0_minus_C": float(b0_valid - branches["C"]["diagnostics"]["valid_mae"]),
        "delta_F_C_minus_E": float(
            branches["C"]["diagnostics"]["valid_mae"]
            - branches["E"]["diagnostics"]["valid_mae"]
        ),
        "branches": branches,
    }
    _write_json(RESULTS_DIR / f"seed{seed}_results.json", results)

    import csv

    curve_path = RESULTS_DIR / f"seed{seed}_training_curves.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "variant", "epoch", "train_loss", "train_mae_eval"])
        for mode in ("C", "E"):
            for row in branches[mode]["curve"]:
                writer.writerow(
                    [
                        int(seed),
                        mode,
                        row["epoch"],
                        row["train_loss"],
                        row.get("train_mae_eval", ""),
                    ]
                )
    _write_json(
        RESULTS_DIR / f"seed{seed}_representation_drift.json",
        {
            "seed": int(seed),
            "probe_n": int(len(probe_index)),
            "C": branches["C"]["diagnostics"]["representation_drift"],
            "E": branches["E"]["diagnostics"]["representation_drift"],
            "E_representation_frozen": branches["E"]["diagnostics"][
                "representation_frozen_check"
            ]["equal"],
            "C_representation_frozen": False,
        },
    )
    return results


# ---------------------------------------------------------------------------
# decision logic
# ---------------------------------------------------------------------------


def seed0_decision(results: Mapping[str, Any]) -> dict[str, Any]:
    delta_a = float(results["delta_A_B0_minus_E"])
    delta_f = float(results["delta_F_C_minus_E"])
    delta_c = float(results["delta_C_B0_minus_C"])
    if delta_f < -THRESHOLD_FULL_CONT_BETTER:
        status = "FULL_CONTINUATION_BETTER"
        run_seed1 = False
    elif delta_a < THRESHOLD_CLEAR_NO_GO:
        status = "CLEAR_NO_GO"
        run_seed1 = False
    elif delta_a >= THRESHOLD_ADAPT_GAIN and delta_f >= THRESHOLD_FREEZE_GAIN:
        status = "STRONG_ADVANCE"
        run_seed1 = True
    elif delta_a >= THRESHOLD_ADAPT_GAIN and 0.0 < delta_f < THRESHOLD_FREEZE_GAIN:
        status = "ADAPTATION_SIGNAL_MECHANISM_WEAK"
        run_seed1 = True
    elif (
        delta_a >= THRESHOLD_ADAPT_GAIN
        and abs(delta_f) <= THRESHOLD_EXTRA_TRAINING_BAND
        and abs(delta_c - delta_a) <= THRESHOLD_EXTRA_TRAINING_BAND
    ):
        status = "EXTRA_OPTIMIZATION_SIGNAL"
        run_seed1 = True
    else:
        status = "BORDERLINE"
        run_seed1 = True
    return {
        "seed": int(results["seed"]),
        "delta_A": delta_a,
        "delta_C": delta_c,
        "delta_F": delta_f,
        "status": status,
        "run_seed1": bool(run_seed1),
    }


def final_decision(
    results_by_seed: Mapping[int, Mapping[str, Any]], write: bool = True
) -> dict[str, Any]:
    seeds = sorted(int(seed) for seed in results_by_seed)
    delta_a = {seed: float(results_by_seed[seed]["delta_A_B0_minus_E"]) for seed in seeds}
    delta_f = {seed: float(results_by_seed[seed]["delta_F_C_minus_E"]) for seed in seeds}
    delta_c = {seed: float(results_by_seed[seed]["delta_C_B0_minus_C"]) for seed in seeds}
    mean_a = float(np.mean(list(delta_a.values())))
    mean_f = float(np.mean(list(delta_f.values())))
    all_a_positive = all(value > 0 for value in delta_a.values())
    all_f_nonneg = all(value >= 0 for value in delta_f.values())

    if len(seeds) >= 2 and all_a_positive and mean_a >= THRESHOLD_ADAPT_GAIN and mean_f >= THRESHOLD_FREEZE_GAIN and all_f_nonneg:
        verdict = "GO_CANONICAL_LATE_READOUT_ADAPTATION_CONFIRMED"
    elif len(seeds) >= 2 and any(value < -THRESHOLD_FULL_CONT_BETTER for value in delta_f.values()):
        verdict = "FULL_CONTINUATION_BETTER"
    elif len(seeds) >= 2 and mean_a >= THRESHOLD_ADAPT_GAIN and mean_f < THRESHOLD_FREEZE_GAIN:
        verdict = "CANONICAL_EXTRA_OPTIMIZATION_ADAPTATION_SIGNAL"
    elif len(seeds) >= 2 and mean_a < THRESHOLD_FINAL_NO_GO_MEAN:
        verdict = "CANONICAL_LATE_READOUT_ADAPTATION_NOT_SUPPORTED"
    elif len(seeds) >= 2 and not all_f_nonneg and not all_a_positive:
        verdict = "INCONCLUSIVE"
    elif len(seeds) == 1:
        seed = seeds[0]
        single = seed0_decision({"seed": seed, **dict(results_by_seed[seed])})
        if single["status"] == "FULL_CONTINUATION_BETTER":
            verdict = "FULL_CONTINUATION_BETTER"
        elif single["status"] == "CLEAR_NO_GO":
            verdict = "CANONICAL_LATE_READOUT_ADAPTATION_NOT_SUPPORTED"
        else:
            verdict = "SEED0_ONLY_" + single["status"]
    else:
        verdict = "INCONCLUSIVE"

    decision = {
        "verdict": verdict,
        "seeds_run": seeds,
        "per_seed": {
            str(seed): {
                "delta_A_B0_minus_E": delta_a[seed],
                "delta_C_B0_minus_C": delta_c[seed],
                "delta_F_C_minus_E": delta_f[seed],
                "b0_valid_mae": float(results_by_seed[seed]["b0_valid_mae"]),
                "c_valid_mae": float(results_by_seed[seed]["c_valid_mae"]),
                "e_valid_mae": float(results_by_seed[seed]["e_valid_mae"]),
            }
            for seed in seeds
        },
        "mean_delta_A": mean_a,
        "mean_delta_C": float(np.mean(list(delta_c.values()))),
        "mean_delta_F": mean_f,
        "all_seed_delta_A_positive": bool(all_a_positive),
        "all_seed_delta_F_nonnegative": bool(all_f_nonneg),
        "decision_case": _decision_case(verdict),
        "official_test_accessed": False,
    }
    if write:
        _write_json(RESULTS_DIR / "final_decision.json", decision)
    return decision


def _decision_case(verdict: str) -> str:
    mapping = {
        "GO_CANONICAL_LATE_READOUT_ADAPTATION_CONFIRMED": "Case A",
        "CANONICAL_EXTRA_OPTIMIZATION_ADAPTATION_SIGNAL": "Case B",
        "FULL_CONTINUATION_BETTER": "Case C",
        "CANONICAL_LATE_READOUT_ADAPTATION_NOT_SUPPORTED": "Case D",
        "INCONCLUSIVE": "Case E",
    }
    return mapping.get(verdict, "Case E")


def write_method_candidate_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "method": "Two-Phase Compact-v4 Training",
        "phase_I": "canonical joint representation learning (unchanged compact-v4-hinge)",
        "phase_II": "freeze representation; late MAE readout adaptation of the existing head",
        "evidence": "canonical late-readout adaptation (this experiment)",
        "decision_verdict": decision.get("verdict"),
        "mean_delta_A": decision.get("mean_delta_A"),
        "mean_delta_F": decision.get("mean_delta_F"),
        "next_steps": [
            "additional valid seeds as confirmation (not ritual)",
            "freeze method, then frozen official test",
        ],
        "not_claimed": [
            "head capacity was insufficient",
            "a new regression architecture is stronger",
            "representation drift is the sole causal mechanism",
            "the effect is specific to MAE without an MSE control",
        ],
    }
    _write_json(RESULTS_DIR / "method_candidate_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# pooled outputs
# ---------------------------------------------------------------------------


def write_pooled(seeds: Sequence[int]) -> dict[str, Any]:
    import csv

    rows: list[dict[str, Any]] = []
    results_by_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = RESULTS_DIR / f"seed{seed}_results.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        results_by_seed[int(seed)] = payload
        rows.append(
            {
                "seed": int(seed),
                "B0_stop": payload["b0_valid_mae"],
                "C_full_continue": payload["c_valid_mae"],
                "E_head_only": payload["e_valid_mae"],
                "delta_A_B0_minus_E": payload["delta_A_B0_minus_E"],
                "delta_C_B0_minus_C": payload["delta_C_B0_minus_C"],
                "delta_F_C_minus_E": payload["delta_F_C_minus_E"],
            }
        )
    if not rows:
        raise RuntimeError("no seed results available for pooling")
    pooled_path = RESULTS_DIR / "pooled_seed_results.csv"
    with open(pooled_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    decision = final_decision(results_by_seed)
    if decision["verdict"] == "GO_CANONICAL_LATE_READOUT_ADAPTATION_CONFIRMED":
        write_method_candidate_summary(decision)
    return decision


# ---------------------------------------------------------------------------
# figures (max 3)
# ---------------------------------------------------------------------------


def make_figures(seeds: Sequence[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results_by_seed = {
        seed: _read_json(RESULTS_DIR / f"seed{seed}_results.json")
        for seed in seeds
        if (RESULTS_DIR / f"seed{seed}_results.json").exists()
    }
    if not results_by_seed:
        return

    # Figure 1: official-valid MAE for B0 / C / E per seed
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    xs = np.arange(len(results_by_seed))
    for offset, (key, label) in enumerate(
        [("b0_valid_mae", "B0 stop"), ("c_valid_mae", "C full continue"), ("e_valid_mae", "E head-only")]
    ):
        values = [results_by_seed[seed][key] for seed in sorted(results_by_seed)]
        ax.bar(xs + (offset - 1) * width, values, width, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {seed}" for seed in sorted(results_by_seed)])
    ax.set_ylabel("official-valid MAE")
    ax.set_title("B0 / C / E official-valid MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_valid_mae_b0_c_e.png", dpi=150)
    plt.close(fig)

    # Figure 2: train-only MAE vs adaptation epoch for C and E
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed in sorted(results_by_seed):
        for mode in ("C", "E"):
            curve = results_by_seed[seed]["branches"][mode]["curve"]
            epochs = [row["epoch"] for row in curve]
            losses = [row["train_loss"] for row in curve]
            ax.plot(epochs, losses, label=f"seed {seed} {mode}")
    ax.set_xlabel("adaptation epoch")
    ax.set_ylabel("train-only L1 loss")
    ax.set_title("train-only adaptation curve (no valid selection)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_train_curve_c_e.png", dpi=150)
    plt.close(fig)

    # Figure 3: C representation drift (only if meaningful)
    drift = {
        seed: results_by_seed[seed]["branches"]["C"]["diagnostics"]["representation_drift"]
        for seed in results_by_seed
    }
    if any(value["mean_l2_drift"] > 1e-9 for value in drift.values()):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            [f"seed {seed}" for seed in sorted(drift)],
            [drift[seed]["normalized_l2_drift"] for seed in sorted(drift)],
        )
        ax.set_ylabel("normalized L2 representation drift (C)")
        ax.set_title("C representation drift on fixed train probe")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_representation_drift.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# stage orchestration
# ---------------------------------------------------------------------------


def stage_all(force: bool = False) -> int:
    checkpoint_inventory()
    lock = protocol_lock()
    k_star = int(lock["adaptation_budget"]["k_star"])
    print(f"[lock] K* = {k_star}", flush=True)

    prepare = prepare_seed(PRIMARY_SEED, force=force)
    trainable_audit = _trainable_audit(PRIMARY_SEED, prepare)
    _write_json(RESULTS_DIR / "standardization_stats.json", prepare["standardization"])
    _write_json(RESULTS_DIR / "starting_equivalence.json", {
        "seed": PRIMARY_SEED,
        "b0_matches_canonical_valid": prepare["b0_matches_canonical_valid"],
        "step0_equivalence": prepare["step0_equivalence"],
    })
    print(f"[prepare] B0 valid MAE = {prepare['b0_valid_mae']!r}", flush=True)

    seed0 = run_seed(PRIMARY_SEED, force=force)
    seed0["seed0_status"] = seed0_decision(seed0)["status"]
    _write_json(RESULTS_DIR / "seed0_results.json", seed0)
    decision0 = seed0_decision(seed0)
    print(f"[seed0] {decision0}", flush=True)

    seeds_run = [PRIMARY_SEED]
    if decision0["run_seed1"]:
        run_seed(1, force=force)
        seeds_run.append(1)
    else:
        print("[seed1] skipped (seed0 clear negative / full-continuation better)", flush=True)

    write_pooled(seeds_run)
    try:
        make_figures(seeds_run)
    except Exception as error:  # figures are descriptive only
        print(f"[figures] skipped: {error}", flush=True)
    _write_json(RESULTS_DIR / "trainable_parameter_audit.json", trainable_audit)
    print(f"[done] seeds run: {seeds_run}", flush=True)
    return 0


def _trainable_audit(seed: int, prepare: Mapping[str, Any]) -> dict[str, Any]:
    info = canonical_run(seed)
    config, run_dir = info["config"], info["run_dir"]
    train_data, valid_data, audit = build_data(config)
    payload = np.load(CACHE_DIR / f"seed{seed}_prepare.npz", allow_pickle=True)
    mean = np.asarray(payload["mean"], dtype=np.float64)
    scale = np.asarray(payload["scale"], dtype=np.float64)
    wrapper = _fresh_wrapper(config, audit, run_dir, mean, scale)
    total_names = [name for name, _ in wrapper.named_parameters()]
    head_names = head_parameter_names(wrapper)
    all_names = [name for name, _ in wrapper.named_parameters()]
    return {
        "seed": int(seed),
        "C": {
            "trainable": all_names,
            "trainable_count": len(all_names),
        },
        "E": {
            "trainable": head_names,
            "trainable_count": len(head_names),
            "frozen_count": len(total_names) - len(head_names),
            "optimizer_parameter_list_only_head": len(set(head_names)) == len(head_names),
        },
        "total_parameter_count": len(total_names),
    }


def stage_prepare(seed: int, force: bool = False) -> dict[str, Any]:
    checkpoint_inventory()
    protocol_lock()
    prepare = prepare_seed(int(seed), force=force)
    _write_json(RESULTS_DIR / "standardization_stats.json", prepare["standardization"])
    _write_json(
        RESULTS_DIR / "starting_equivalence.json",
        {
            "seed": int(seed),
            "b0_matches_canonical_valid": prepare["b0_matches_canonical_valid"],
            "step0_equivalence": prepare["step0_equivalence"],
        },
    )
    _write_json(RESULTS_DIR / "trainable_parameter_audit.json", _trainable_audit(int(seed), prepare))
    return prepare


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["inventory", "lock", "prepare", "seed0", "seed1", "summary", "figures", "all"],
        nargs="?",
        default="all",
    )
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(4)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "inventory":
        checkpoint_inventory()
        return 0
    if args.stage == "lock":
        protocol_lock()
        return 0
    if args.stage == "prepare":
        stage_prepare(int(args.seed), force=args.force)
        return 0
    if args.stage == "seed0":
        run_seed(PRIMARY_SEED, force=args.force)
        return 0
    if args.stage == "seed1":
        run_seed(1, force=args.force)
        return 0
    if args.stage == "summary":
        write_pooled(BACKBONE_SEEDS)
        return 0
    if args.stage == "figures":
        make_figures(BACKBONE_SEEDS)
        return 0
    return stage_all(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
