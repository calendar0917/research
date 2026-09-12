"""Compact-v4 Small-Head End-to-End Compression Confirmation (ZINC).

This is **not** a new representation, parameter-reallocation, new-recipient,
architecture-expansion, P1/P2/cell rescue, HPO, head-family search or test
benchmark.  It is a single from-scratch compression-confirmation experiment.

Single hypothesis
-----------------
The optimized compact-v4 graph representation ``R`` (302D) does not require the
current 21,633-parameter graph head for successful *end-to-end* learning; a
fixed ~4,135-parameter small raw head can remove ~17,498 parameters while
remaining validation-non-inferior.

The Parameter Allocation & Representation Leverage audit established the donor
on *already-trained frozen backbones* (frozen ``R`` -> refit small head):
seed0 0.146420 -> 0.143208, seed1 0.149332 -> 0.147754 (H-STRONG).  A frozen
readout fit is **not** an end-to-end compression proof.  This module tests the
actual from-scratch joint optimization:

    random init -> small-head model -> full joint end-to-end optimization.

Everything except the graph head is frozen to optimized compact-v4-hinge:

* tokenizer / historical aliased rooted-topology representation;
* token embedding / parent embedding / patch encoder / pair projection /
  pair encoder / pair construction / distance features;
* centre mean/std/log-count pooling / centre update;
* final unary moments / final pair moments / global descriptors;
* topology branch; ``R = 302D``;
* loss / optimizer / training protocol.

The only change is ``head``: the original
``Linear(302,64) -> LayerNorm -> ReLU -> Dropout -> Linear(64,32) -> ReLU ->
Linear(32,1)`` (21,633 params) is replaced by the pre-audited fixed raw small
head ``302 -> 13 -> 13 -> 1`` (4,135 params, raw ``R``, direct ``yhat=f(R)``,
no standardisation, no LayerNorm, no dropout, no residual over ``yhat_0``).
This exact module is the historical ``Sraw`` / ``Hsmall`` reader
(``GenericReader``), reused verbatim -- there is no head-width search.

Protocol (inherited optimized compact-v4, not re-tuned)
-------------------------------------------------------
Adam, lr 1e-3, weight decay 1e-5, batch 128, max epochs 240, patience 40,
scheduler none, L1, best official-valid checkpoint, single stage.

Budget rules
------------
* Reuse the existing optimized compact-v4 baseline (never retrain 99,613).
* One full-training run (seed0) first.
* Buy seed1 only if seed0 loses no more than +0.002 valid MAE.
* Never run seed2/seed3.
* The released 17,498 parameters are **not** reallocated to anything else.
* Official **test is never loaded**, even after a two-seed success.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e <stage>

Stages: ``architecture_lock baseline param_audit init_match_seed0 integrity
compute_audit stage0 stage1_seed0 stage1_decision bulk_seed0 init_match_seed1
stage2_seed1 stage2_decision bulk_seed1 two_seed drift answers final figures
all``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import platform
import random
import resource
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency import (
    V4_EXPECTED_VALID,
    base_config as _sufficiency_base_config,
    build_encoded,
    load_train_valid_records,
)
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import V4_RUN_IDS

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_smallhead_e2e"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
FIGURE_DIR = RESULTS_DIR / "figures"
RUNS_DIR = RESULTS_DIR / "runs"

CANONICAL_V4_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
SUFFICIENCY_DIR = TRACK_ROOT / "results/compact_v4_training_sufficiency"
V4_SEED0_STATE = SUFFICIENCY_DIR / "states/Pstar_A2_long_seed0_selection_state.pt"
V4_SEED1_STATE = SUFFICIENCY_DIR / "states/Pstar_A2_long_seed1_selection_state.pt"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2
PROTOCOL_VERSION = "compact_v4_smallhead_e2e_v1"

# --- locked architecture constants -----------------------------------------
R_DIM = 302
SMALL_HEAD_HIDDEN = (13, 13)  # audited Sraw / Hsmall definition
SMALL_HEAD_SEED = 0  # historical small-head seed convention (head_seed 0)
SMALL_HEAD_ACTIVATION = "ReLU"
SMALL_HEAD_INPUT = "raw R (no standardisation)"
EXPECTED_BASELINE_TOTAL = 99613
EXPECTED_BASELINE_HEAD = 21633
EXPECTED_SMALL_HEAD = 4135
EXPECTED_SMALL_TOTAL = EXPECTED_BASELINE_TOTAL - EXPECTED_BASELINE_HEAD + EXPECTED_SMALL_HEAD
EXPECTED_REDUCTION = EXPECTED_BASELINE_HEAD - EXPECTED_SMALL_HEAD

# --- frozen optimized training protocol (inherited, not re-tuned) ----------
OPTIMIZED_PROTOCOL: dict[str, Any] = {
    "optimizer": "Adam",
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "batch_size": 128,
    "max_epochs": 240,
    "patience": 40,
    "scheduler": "none",
    "gradient_clip_norm": 5.0,
    "loss": "L1 / mean absolute error",
    "checkpoint_selection": "best official-valid MAE",
    "single_stage": True,
    "train_shuffle_seed_offset": 91011,
    "eval_shuffle_seed_offset": 91012,
}
TORCH_THREADS = 4

# --- reused optimized-v4 references (do NOT retrain) -----------------------
V4_SEED0_VALID = 0.14642022556537995
V4_SEED1_VALID = 0.1493322635096847
V4_SEED0_EPOCH = 169
V4_SEED1_EPOCH = 104
V4_PARAMS = 99613

# --- pre-registered non-inferiority thresholds -----------------------------
NI_GATE = 0.002          # per-seed D_s = S_s - B_s <= +0.002
NI_MEAN_GATE = 0.0015    # mean(D_0, D_1) <= +0.0015
MILD_UPPER = 0.0035      # +0.002 < D_0 <= +0.0035 => sub-threshold signal
BULK_CAUTION = 0.003     # bulk degradation caution threshold (secondary)


# ---------------------------------------------------------------------------
# io / determinism helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
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


def _environment_fingerprint() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "device": "cpu",
    }


def _configure_determinism() -> None:
    torch.set_num_threads(int(TORCH_THREADS))
    torch.use_deterministic_algorithms(False)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


# ---------------------------------------------------------------------------
# model: compact-v4 with the frozen small raw head
# ---------------------------------------------------------------------------


class PatchPathSmallHeadModel(zpp.PatchPathModel):
    """compact-v4 backbone with the pre-audited fixed small raw head.

    ``super().__init__`` constructs every shared representation-forming module
    exactly as canonical compact-v4 does; the single graph head is then
    replaced by the historical ``Sraw`` / ``Hsmall`` reader
    (``GenericReader(302, (13, 13))``): raw ``R`` -> Linear(302,13) -> ReLU ->
    Linear(13,13) -> ReLU -> Linear(13,1).  No LayerNorm, no dropout, no
    standardisation, no residual over ``yhat_0``.
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        small_head_hidden: Sequence[int] = SMALL_HEAD_HIDDEN,
        **kwargs: Any,
    ) -> None:
        super().__init__(typed_vocabulary_size, parent_vocabulary_size, **kwargs)
        if int(self.unified_graph_width) != R_DIM:
            raise ValueError(
                f"small head requires a {R_DIM}D pre-head representation; "
                f"got {self.unified_graph_width}"
            )
        self.small_head_hidden = tuple(int(width) for width in small_head_hidden)
        self.head = GenericReader(int(self.unified_graph_width), self.small_head_hidden)


def load_config(path: Path = CANONICAL_V4_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def base_config() -> dict[str, Any]:
    config = load_config()
    config["test_policy"] = "no_test"
    config["parameter_audit"] = False
    config["model"]["device"] = "cpu"
    config["output"] = {
        "json": str(RESULTS_DIR / "scratch.json"),
        "markdown": str(RESULTS_DIR / "scratch.md"),
    }
    return config


def _base_kwargs() -> dict[str, Any]:
    model_config = base_config()["model"]
    return dict(
        patch_hidden=int(model_config.get("patch_hidden", 48)),
        pair_hidden=int(model_config.get("pair_hidden", 16)),
        token_width=int(model_config.get("token_width", 16)),
        dropout=float(model_config.get("dropout", 0.05)),
        embedding_mode=str(model_config.get("embedding_mode", "hybrid")),
        embedding_rank=int(model_config.get("embedding_rank", 4)),
        hybrid_full_typed_tokens=(
            None
            if model_config.get("hybrid_full_typed_tokens") is None
            else int(model_config["hybrid_full_typed_tokens"])
        ),
        hybrid_full_parent_tokens=(
            None
            if model_config.get("hybrid_full_parent_tokens") is None
            else int(model_config["hybrid_full_parent_tokens"])
        ),
        center_context=bool(model_config.get("center_context", True)),
        center_context_hidden=(
            None
            if model_config.get("center_context_hidden") is None
            else int(model_config["center_context_hidden"])
        ),
        graph_head_hidden_0=(
            None
            if model_config.get("graph_head_hidden_0") is None
            else int(model_config["graph_head_hidden_0"])
        ),
        graph_head_hidden_1=(
            None
            if model_config.get("graph_head_hidden_1") is None
            else int(model_config["graph_head_hidden_1"])
        ),
        topology_mode=str(model_config.get("topology_mode", "hinge")),
        topology_input_width=int(ztopo.raw_width("hinge")),
        topology_hidden_dim=int(model_config.get("topology_hidden_dim", 16)),
        topology_out_dim=int(model_config.get("topology_out_dim", 8)),
    )


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def build_baseline(seed: int = 0) -> zpp.PatchPathModel:
    """Canonical optimized compact-v4-hinge (99,613 params) under ``seed``."""
    _seed_everything(seed)
    return zpp.PatchPathModel(6785, 32, **_base_kwargs())


def build_smallhead(
    seed: int,
    *,
    typed_vocabulary_size: int = 6785,
    parent_vocabulary_size: int = 32,
    head_seed: int = SMALL_HEAD_SEED,
) -> PatchPathSmallHeadModel:
    """From-scratch small-head model with exact baseline shared tensors.

    Construction-order discipline (no RNG confound):

    1. deterministically instantiate the canonical baseline under ``seed`` and
       snapshot every shared tensor;
    2. instantiate the small-head model under the same ``seed``;
    3. copy every shared initial tensor back from the baseline snapshot
       (exact copy, so shared initialization is bit-identical to baseline);
    4. initialize the small head independently under the historical small-head
       seed convention (``head_seed = 0``).

    This is a normal from-scratch initialization -- never a trained-checkpoint
    warm start.
    """
    baseline = build_baseline(seed)
    baseline_state = {k: v.detach().clone() for k, v in baseline.state_dict().items()}

    _seed_everything(seed)
    model = PatchPathSmallHeadModel(
        int(typed_vocabulary_size), int(parent_vocabulary_size), **_base_kwargs()
    )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in baseline_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)

    # historical small-head initialization convention: fresh head_seed stream.
    torch.manual_seed(int(head_seed))
    model.head = GenericReader(int(model.unified_graph_width), SMALL_HEAD_HIDDEN)
    return model


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def build_encoded_records():
    train_records, valid_records = load_train_valid_records()
    config = _sufficiency_base_config()
    config["test_policy"] = "no_test"
    config["model"]["device"] = "cpu"
    return build_encoded(train_records, valid_records, config)


# ---------------------------------------------------------------------------
# training / evaluation (faithful mirror of zpp._train_phase)
# ---------------------------------------------------------------------------


def _evaluate_mae(
    model: nn.Module, loader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    if not targets:
        return float("nan"), np.zeros(0), np.zeros(0)
    t = np.concatenate(targets).astype(np.float64)
    p = np.concatenate(preds).astype(np.float64)
    return float(np.mean(np.abs(t - p))), t, p


def _module_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return math.sqrt(total)


def _module_weight_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        total += float(parameter.detach().pow(2).sum())
    return math.sqrt(total)


def train_model(
    *,
    build_fn: Callable[[int], nn.Module],
    train_data: Sequence[Data],
    valid_data: Sequence[Data],
    seed: int,
    tag: str,
    save_state: bool = True,
    real_batch_identity: bool = True,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model = build_fn(seed).to(device)

    identity: dict[str, Any] = {}
    if real_batch_identity:
        batch = next(
            iter(
                zpp._make_loader(
                    list(valid_data)[:128],
                    int(OPTIMIZED_PROTOCOL["batch_size"]),
                    False,
                    0,
                )
            )
        ).to(device)
        baseline = build_baseline(seed).to(device)
        baseline.eval()
        model.eval()
        with torch.no_grad():
            r_base = baseline.encode(batch)
            r_small = model.encode(batch)
        identity = {
            "split": "first 128 official-valid molecules",
            "R_dim": int(r_base.shape[1]),
            "max_abs_diff": float((r_base - r_small).abs().max()),
            "bit_identical": bool(float((r_base - r_small).abs().max()) == 0.0),
            "official_test_loaded": False,
        }
        model.train()

    total_params = _n_params(model)
    head_params = _n_params(model.head)
    if int(total_params) != EXPECTED_SMALL_TOTAL:
        raise RuntimeError(
            f"small-head total params {total_params} != expected "
            f"{EXPECTED_SMALL_TOTAL}"
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(OPTIMIZED_PROTOCOL["learning_rate"]),
        weight_decay=float(OPTIMIZED_PROTOCOL["weight_decay"]),
    )
    loader = zpp._make_loader(
        train_data,
        int(OPTIMIZED_PROTOCOL["batch_size"]),
        True,
        seed + int(OPTIMIZED_PROTOCOL["train_shuffle_seed_offset"]),
    )
    eval_loader = zpp._make_loader(
        valid_data,
        int(OPTIMIZED_PROTOCOL["batch_size"]),
        False,
        seed + int(OPTIMIZED_PROTOCOL["eval_shuffle_seed_offset"]),
    )
    steps_per_epoch = int(
        math.ceil(len(train_data) / int(OPTIMIZED_PROTOCOL["batch_size"]))
    )
    patience = int(OPTIMIZED_PROTOCOL["patience"])
    max_epochs = int(OPTIMIZED_PROTOCOL["max_epochs"])
    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    head_params_list = list(model.head.parameters())
    head_ids = {id(p) for p in head_params_list}
    backbone_params_list = [p for p in model.parameters() if id(p) not in head_ids]
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        head_grad_sum = 0.0
        backbone_grad_sum = 0.0
        grad_steps = 0
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            head_grad_sum += _module_grad_norm(head_params_list)
            backbone_grad_sum += _module_grad_norm(backbone_params_list)
            grad_steps += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(OPTIMIZED_PROTOCOL["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae, _, _ = _evaluate_mae(model, eval_loader, device)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_mae": float(train_mae),
            "valid_mae": float(valid_mae),
            "lr": float(OPTIMIZED_PROTOCOL["learning_rate"]),
            "optimizer_steps": int(epoch * steps_per_epoch),
            "checkpoint_selected": 0,
            "head_grad_norm": float(head_grad_sum / max(grad_steps, 1)),
            "backbone_grad_norm": float(backbone_grad_sum / max(grad_steps, 1)),
            "head_weight_norm": float(_module_weight_norm(head_params_list)),
        }
        curve.append(row)
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == max_epochs:
            print(
                f"[{tag} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch} "
                f"head_g={row['head_grad_norm']:.3e} "
                f"back_g={row['backbone_grad_norm']:.3e}",
                flush=True,
            )
        if stale >= patience:
            print(
                f"[{tag} seed{seed}] early_stop epoch={epoch} best={best_epoch}",
                flush=True,
            )
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    for row in curve:
        row["checkpoint_selected"] = int(row["epoch"] == best_epoch)
    wall_clock = float(time.perf_counter() - started)

    curve_path = CURVE_DIR / f"{tag}_seed{seed}_curve.csv"
    _write_csv(
        curve_path,
        curve,
        (
            "epoch",
            "train_mae",
            "valid_mae",
            "lr",
            "optimizer_steps",
            "checkpoint_selected",
            "head_grad_norm",
            "backbone_grad_norm",
            "head_weight_norm",
        ),
    )
    state_path = None
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        torch.save(model.state_dict(), state_path)

    final_valid, valid_targets, valid_predictions = _evaluate_mae(
        model, eval_loader, device
    )
    horizon_warning = bool(best_epoch >= max_epochs - max(1, max_epochs // 10))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "seed": int(seed),
        "protocol": dict(OPTIMIZED_PROTOCOL),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(final_valid),
        "train_loss_at_best": float(losses[best_epoch - 1]),
        "epochs_run": int(len(losses)),
        "steps_per_epoch": int(steps_per_epoch),
        "optimizer_steps": int(len(losses) * steps_per_epoch),
        "wall_clock_s": wall_clock,
        "early_stopped": bool(len(losses) < max_epochs),
        "horizon_boundary_warning": horizon_warning,
        "parameters": int(total_params),
        "head_parameters": int(head_params),
        "state_path": None if state_path is None else str(state_path),
        "curve_path": str(curve_path),
        "pre_head_R_identity": identity,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(),
        "official_test_loaded": False,
        "valid_predictions": valid_predictions.tolist(),
        "valid_targets": valid_targets.tolist(),
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
# Stage 0: locks, audits, gates
# ---------------------------------------------------------------------------


def architecture_lock() -> dict[str, Any]:
    model = build_smallhead(0)
    baseline = build_baseline(0)
    payload = {
        "name": "compact-v4-smallhead end-to-end compression confirmation",
        "protocol_version": PROTOCOL_VERSION,
        "single_principle_changed": "graph head only",
        "baseline_architecture_fingerprint": {
            "params": int(_n_params(baseline)),
            "head": "302 -> 64 -> 32 -> 1 (Linear/LayerNorm/ReLU/Dropout)",
            "head_params": int(_n_params(baseline.head)),
            "R_dim": int(baseline.unified_graph_width),
        },
        "small_head_exact_layers": [
            f"Linear({R_DIM},13)",
            SMALL_HEAD_ACTIVATION,
            "Linear(13,13)",
            SMALL_HEAD_ACTIVATION,
            "Linear(13,1)",
        ],
        "small_head_definition": (
            "historical audited Sraw / Hsmall reader "
            "(zinc_graph_head_function_family.GenericReader)"
        ),
        "small_head_hidden": list(SMALL_HEAD_HIDDEN),
        "small_head_input": SMALL_HEAD_INPUT,
        "small_head_activation": SMALL_HEAD_ACTIVATION,
        "normalization": "none",
        "dropout": "none",
        "residual_over_yhat0": "none",
        "R_dimension": int(model.unified_graph_width),
        "small_head_params": int(_n_params(model.head)),
        "total_params": int(_n_params(model)),
        "training_protocol": dict(OPTIMIZED_PROTOCOL),
        "seed_policy": {
            "stage1": [0],
            "stage2_conditional": [1],
            "forbidden": [2, 3],
            "small_head_init_seed": int(SMALL_HEAD_SEED),
            "shared_init_source": "canonical optimized compact-v4 baseline init",
        },
        "non_inferiority_threshold": float(NI_GATE),
        "replication_threshold": {
            "per_seed": float(NI_GATE),
            "mean": float(NI_MEAN_GATE),
        },
        "official_test_lock": "never loaded",
        "forbidden": [
            "any head-width / activation / LayerNorm / dropout / residual search",
            "warm start from a trained checkpoint",
            "knowledge distillation",
            "reallocating the released 17,498 parameters to any module",
            "changing tokenizer/radius/patch encoder/pair path/centre update/topology/global",
            "scheduler / longer horizon / head-specific LR / gradient-clip change",
            "seed2 / seed3",
            "official test access",
        ],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "architecture_lock.json", payload)
    return payload


def baseline_inventory() -> dict[str, Any]:
    payload = {
        "protocol": dict(OPTIMIZED_PROTOCOL),
        "optimized_v4_seed0_valid": V4_SEED0_VALID,
        "optimized_v4_seed1_valid": V4_SEED1_VALID,
        "optimized_v4_seed0_epoch": V4_SEED0_EPOCH,
        "optimized_v4_seed1_epoch": V4_SEED1_EPOCH,
        "optimized_v4_params": V4_PARAMS,
        "optimized_v4_seed0_state": str(V4_SEED0_STATE),
        "optimized_v4_seed0_state_sha256": (
            _sha256_file(V4_SEED0_STATE) if V4_SEED0_STATE.exists() else None
        ),
        "optimized_v4_seed1_state": str(V4_SEED1_STATE),
        "optimized_v4_seed1_state_sha256": (
            _sha256_file(V4_SEED1_STATE) if V4_SEED1_STATE.exists() else None
        ),
        "v4_run_id_seed0": V4_RUN_IDS[0],
        "canonical_config": str(CANONICAL_V4_CONFIG),
        "canonical_config_sha256": _sha256_file(CANONICAL_V4_CONFIG),
        "baseline_retrained": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "baseline_inventory.json", payload)
    return payload


def baseline_anchor() -> dict[str, Any]:
    """Verify the canonical data pipeline reproduces the reused baseline.

    Loads the existing optimized seed0 checkpoint into a freshly constructed
    baseline and evaluates official valid; the MAE must equal 0.146420 exactly.
    No baseline training is performed.
    """
    _train_data, valid_data, audit = build_encoded_records()
    model = zpp.PatchPathModel(
        int(audit["typed_vocabulary_size_with_oov"]),
        int(audit["parent_vocabulary_size_with_oov"]),
        **_base_kwargs(),
    )
    model.load_state_dict(torch.load(V4_SEED0_STATE, map_location="cpu"))
    model.eval()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    mae, _, _ = _evaluate_mae(model, loader, torch.device("cpu"))
    payload = {
        "baseline_seed0_valid_mae": float(mae),
        "expected": float(V4_SEED0_VALID),
        "matches_reference": bool(abs(float(mae) - float(V4_SEED0_VALID)) == 0.0),
        "typed_vocabulary_size_with_oov": int(audit["typed_vocabulary_size_with_oov"]),
        "parent_vocabulary_size_with_oov": int(audit["parent_vocabulary_size_with_oov"]),
        "baseline_params": int(_n_params(model)),
        "baseline_retrained": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "baseline_anchor.json", payload)
    return payload


def parameter_audit() -> dict[str, Any]:
    baseline = build_baseline(0)
    model = build_smallhead(0)
    baseline_total = _n_params(baseline)
    baseline_head = _n_params(baseline.head)
    small_total = _n_params(model)
    small_head = _n_params(model.head)
    absolute_reduction = int(baseline_head - small_head)
    payload = {
        "baseline_total": int(baseline_total),
        "baseline_head": int(baseline_head),
        "smallhead_total": int(small_total),
        "smallhead_head": int(small_head),
        "absolute_reduction": int(absolute_reduction),
        "relative_total_reduction": float(absolute_reduction / baseline_total),
        "relative_head_reduction": float(absolute_reduction / baseline_head),
        "expected_baseline_total": int(EXPECTED_BASELINE_TOTAL),
        "expected_baseline_head": int(EXPECTED_BASELINE_HEAD),
        "expected_smallhead_head": int(EXPECTED_SMALL_HEAD),
        "expected_smallhead_total": int(EXPECTED_SMALL_TOTAL),
        "expected_absolute_reduction": int(EXPECTED_REDUCTION),
        "mechanical_account_consistent": bool(
            small_total == baseline_total - baseline_head + small_head
            and small_total == EXPECTED_SMALL_TOTAL
            and absolute_reduction == EXPECTED_REDUCTION
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


def _init_match_for(seed: int, out_name: str) -> dict[str, Any]:
    baseline = build_baseline(seed)
    baseline_state = {k: v.detach().clone() for k, v in baseline.state_dict().items()}
    model = build_smallhead(seed)

    state = model.state_dict()
    shared_keys = [
        key
        for key in baseline_state
        if key in state and state[key].shape == baseline_state[key].shape
    ]
    max_abs = 0.0
    mismatch: dict[str, float] = {}
    for key in shared_keys:
        diff = float((state[key] - baseline_state[key]).abs().max())
        max_abs = max(max_abs, diff)
        if diff != 0.0:
            mismatch[key] = diff
    exact = len(mismatch) == 0
    copied = False
    if not exact:
        with torch.no_grad():
            for key in shared_keys:
                state[key].copy_(baseline_state[key])
        copied = True
    baseline_shared = {k: baseline_state[k] for k in shared_keys}
    small_shared = {k: state[k].detach().clone() for k in shared_keys}
    payload = {
        "policy": (
            "from-scratch initialization matching (NOT a trained-checkpoint "
            "warm start)"
        ),
        "seed": int(seed),
        "baseline_total_params": int(_n_params(baseline)),
        "smallhead_total_params": int(_n_params(model)),
        "smallhead_head_params": int(_n_params(model.head)),
        "n_shared_tensors": int(len(shared_keys)),
        "shared_names": sorted(shared_keys),
        "max_abs_diff": float(max_abs),
        "exact_equal_to_baseline_init": bool(exact),
        "hash_match": bool(
            _state_hash(baseline_shared) == _state_hash(small_shared)
        ),
        "explicit_copy_applied": bool(copied),
        "mismatch": mismatch,
        "shared_state_sha256": _state_hash(small_shared),
        "baseline_shared_state_sha256": _state_hash(baseline_shared),
        "small_head_init_seed": int(SMALL_HEAD_SEED),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / out_name, payload)
    return payload


def initialization_match_seed0() -> dict[str, Any]:
    return _init_match_for(0, "initialization_match_seed0.json")


def initialization_match_seed1() -> dict[str, Any]:
    return _init_match_for(1, "initialization_match_seed1.json")


def _non_head_structure(model: nn.Module) -> dict[str, tuple]:
    return {
        name: (tuple(parameter.shape), str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if not name.startswith("head.")
    }


def integrity_gates(report: bool = True) -> dict[str, Any]:
    results: dict[str, bool] = {}
    details: dict[str, Any] = {}

    baseline = build_baseline(0)
    model = build_smallhead(0)
    model.eval()
    baseline.eval()

    # G0.1 baseline total = 99,613
    results["G0.1_baseline_total_99613"] = (
        int(_n_params(baseline)) == EXPECTED_BASELINE_TOTAL
    )
    details["G0.1_baseline_total"] = int(_n_params(baseline))

    # G0.2 baseline head = 21,633
    results["G0.2_baseline_head_21633"] = (
        int(_n_params(baseline.head)) == EXPECTED_BASELINE_HEAD
    )
    details["G0.2_baseline_head"] = int(_n_params(baseline.head))

    # G0.3 small head = 4,135
    results["G0.3_small_head_4135"] = (
        int(_n_params(model.head)) == EXPECTED_SMALL_HEAD
    )
    details["G0.3_small_head"] = int(_n_params(model.head))

    # G0.4 candidate total = 82,115 and mechanically consistent
    results["G0.4_smallhead_total_82115"] = bool(
        int(_n_params(model)) == EXPECTED_SMALL_TOTAL
        and int(_n_params(model))
        == int(_n_params(baseline))
        - int(_n_params(baseline.head))
        + int(_n_params(model.head))
    )
    details["G0.4_smallhead_total"] = int(_n_params(model))

    # G0.5 R still 302D
    results["G0.5_R_302"] = bool(
        int(model.unified_graph_width) == R_DIM
        and int(baseline.unified_graph_width) == R_DIM
    )
    details["G0.5_R_dim"] = int(model.unified_graph_width)

    # G0.6 all non-head trainable modules structurally identical
    base_struct = _non_head_structure(baseline)
    small_struct = _non_head_structure(model)
    results["G0.6_non_head_identical"] = bool(base_struct == small_struct)
    details["G0.6_non_head_tensors"] = int(len(base_struct))
    details["G0.6_non_head_names_match"] = bool(
        sorted(base_struct.keys()) == sorted(small_struct.keys())
    )

    # G0.7 shared initial tensors seed0 exact match
    init0 = initialization_match_seed0()
    results["G0.7_shared_init_exact"] = bool(
        init0["exact_equal_to_baseline_init"]
        and init0["hash_match"]
        and init0["max_abs_diff"] == 0.0
    )
    details["G0.7_shared_tensors"] = int(init0["n_shared_tensors"])

    # G0.8 pre-head R exact match on a real batch
    _train_data, valid_data, _audit = build_encoded_records()
    batch = next(
        iter(
            zpp._make_loader(
                list(valid_data)[:128], int(OPTIMIZED_PROTOCOL["batch_size"]), False, 0
            )
        )
    )
    with torch.no_grad():
        r_base = baseline.encode(batch)
        r_small = model.encode(batch)
    r_diff = float((r_base - r_small).abs().max())
    results["G0.8_pre_head_R_exact"] = bool(r_diff == 0.0)
    details["G0.8_R_max_abs_diff"] = r_diff
    details["G0.8_R_dim"] = int(r_base.shape[1])

    # G0.9 optimizer / loss / training protocol identical
    canonical = {
        "optimizer": "Adam",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-5,
        "batch_size": 128,
        "max_epochs": 240,
        "patience": 40,
        "scheduler": "none",
        "loss": "L1 / mean absolute error",
        "checkpoint_selection": "best official-valid MAE",
        "single_stage": True,
    }
    results["G0.9_protocol_identical"] = bool(
        all(OPTIMIZED_PROTOCOL[key] == value for key, value in canonical.items())
        and float(OPTIMIZED_PROTOCOL["gradient_clip_norm"]) == 5.0
    )
    details["G0.9_protocol"] = dict(OPTIMIZED_PROTOCOL)

    # G0.10 official test inaccessible / never loaded
    config = base_config()
    source = inspect.getsource(inspect.getmodule(integrity_gates))
    # split so this gate line does not match its own search pattern
    test_load_pattern = "_load_zinc(ZINC_ROOT, " + '"test")'
    test_load_hits = source.count(test_load_pattern)
    results["G0.10_official_test_blocked"] = bool(
        config.get("test_policy") == "no_test" and test_load_hits == 0
    )
    details["G0.10_test_policy"] = config.get("test_policy")

    details["results"] = results
    details["passed"] = all(results.values())
    details["official_test_loaded"] = False
    if report:
        _write_json(RESULTS_DIR / "integrity_gates.json", details)
    return details


def compute_audit() -> dict[str, Any]:
    _train, valid_data, _audit = build_encoded_records()
    baseline = build_baseline(0)
    baseline.eval()
    model = build_smallhead(0)
    model.eval()
    batch = next(
        iter(
            zpp._make_loader(
                list(valid_data)[:128], int(OPTIMIZED_PROTOCOL["batch_size"]), False, 0
            )
        )
    )
    n_pairs = int(batch.pair_index.shape[1])
    n_centres = int(batch.num_nodes)
    n_graphs = int(batch.num_graphs)
    with torch.no_grad():
        for _ in range(2):
            baseline(batch)
            model(batch)
        t0 = time.perf_counter()
        for _ in range(5):
            baseline(batch)
        base_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
        t0 = time.perf_counter()
        for _ in range(5):
            model(batch)
        small_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload = {
        "batch_graphs": int(n_graphs),
        "batch_centres": int(n_centres),
        "batch_pairs": int(n_pairs),
        "forward_ms_baseline": float(base_ms),
        "forward_ms_smallhead": float(small_ms),
        "forward_overhead_ms": float(small_ms - base_ms),
        "forward_change_ratio": float(small_ms / max(base_ms, 1e-9) - 1.0),
        "peak_rss_mb": float(peak_rss_mb),
        "baseline_total_params": int(EXPECTED_BASELINE_TOTAL),
        "smallhead_total_params": int(EXPECTED_SMALL_TOTAL),
        "actual_parameter_reduction": int(EXPECTED_REDUCTION),
        "note": (
            "the backbone dominates compute; a 17.6% parameter reduction is "
            "NOT assumed to give a proportional speedup"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "compute_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage1_smallhead_seed0() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records()
    summary = train_model(
        build_fn=lambda s: build_smallhead(
            s,
            typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
            parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        ),
        train_data=train_data,
        valid_data=valid_data,
        seed=0,
        tag="smallhead",
    )
    summary["baseline_v4_valid"] = float(V4_SEED0_VALID)
    summary["delta_D0"] = float(summary["best_valid_mae"] - V4_SEED0_VALID)
    summary["non_inferiority_gate"] = float(NI_GATE)
    summary["passes_non_inferiority"] = bool(summary["delta_D0"] <= NI_GATE)
    _write_json(RESULTS_DIR / "stage1_smallhead_seed0.json", summary)
    return summary


def stage2_smallhead_seed1() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records()
    initialization_match_seed1()
    summary = train_model(
        build_fn=lambda s: build_smallhead(
            s,
            typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
            parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        ),
        train_data=train_data,
        valid_data=valid_data,
        seed=1,
        tag="smallhead",
    )
    summary["baseline_v4_valid"] = float(V4_SEED1_VALID)
    summary["delta_D1"] = float(summary["best_valid_mae"] - V4_SEED1_VALID)
    summary["non_inferiority_gate"] = float(NI_GATE)
    summary["passes_non_inferiority"] = bool(summary["delta_D1"] <= NI_GATE)
    _write_json(RESULTS_DIR / "stage2_smallhead_seed1.json", summary)
    return summary


def _optimization_ambiguity(summary: Mapping[str, Any]) -> dict[str, Any]:
    curve_path = Path(str(summary["curve_path"]))
    rows = list(csv.DictReader(curve_path.open(encoding="utf-8")))
    valid = [float(row["valid_mae"]) for row in rows]
    head_grads = [float(row["head_grad_norm"]) for row in rows]
    backbone_grads = [float(row["backbone_grad_norm"]) for row in rows]
    max_epochs = int(OPTIMIZED_PROTOCOL["max_epochs"])
    best_epoch = int(summary["best_epoch"])
    boundary = best_epoch >= max_epochs - max(1, max_epochs // 10)
    nan = bool(
        any(not math.isfinite(value) for value in valid + head_grads + backbone_grads)
    )
    tail = valid[best_epoch - 1 :] if valid else []
    monotone_tail = bool(len(tail) >= 5 and tail[-1] < min(tail[: max(1, len(tail) // 2)]))
    return {
        "nan_or_nonfinite": nan,
        "best_epoch": best_epoch,
        "max_epochs": max_epochs,
        "horizon_boundary_pinned": bool(boundary),
        "head_grad_alive": bool(max(head_grads) > 0.0 and min(head_grads) >= 0.0),
        "backbone_grad_alive": bool(max(backbone_grads) > 0.0),
        "valid_still_improving_at_horizon": bool(boundary and monotone_tail),
        "ambiguous": bool(nan or (boundary and monotone_tail)),
    }


def stage1_decision() -> dict[str, Any]:
    summary = _read_json(RESULTS_DIR / "stage1_smallhead_seed0.json")
    d0 = float(summary["delta_D0"])
    ambiguity = _optimization_ambiguity(summary)
    if ambiguity["ambiguous"]:
        verdict = "OPTIMIZATION_OR_IMPLEMENTATION_AMBIGUOUS"
        buy_seed1 = False
    elif d0 <= NI_GATE:
        verdict = "SEED0_NON_INFERIOR"
        buy_seed1 = True
    elif d0 <= MILD_UPPER:
        verdict = "COMPRESSION_SIGNAL_SUB_THRESHOLD_STOP"
        buy_seed1 = False
    else:
        verdict = "END_TO_END_SMALL_HEAD_COMPRESSION_NO_GO"
        buy_seed1 = False
    payload = {
        "S0": float(summary["best_valid_mae"]),
        "B0": float(V4_SEED0_VALID),
        "D0": d0,
        "non_inferiority_gate": float(NI_GATE),
        "mild_upper": float(MILD_UPPER),
        "best_epoch": int(summary["best_epoch"]),
        "epochs_run": int(summary["epochs_run"]),
        "optimization_ambiguity": ambiguity,
        "verdict": verdict,
        "buy_seed1": bool(buy_seed1),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage1_decision.json", payload)
    return payload


def stage2_decision() -> dict[str, Any]:
    s1 = _read_json(RESULTS_DIR / "stage2_smallhead_seed1.json")
    d1 = float(s1["delta_D1"])
    s0 = _read_json(RESULTS_DIR / "stage1_smallhead_seed0.json")
    d0 = float(s0["delta_D0"])
    mean_d = float((d0 + d1) / 2.0)
    ambiguity = _optimization_ambiguity(s1)
    per_seed_pass = bool(d0 <= NI_GATE and d1 <= NI_GATE)
    mean_pass = bool(mean_d <= NI_MEAN_GATE)
    if ambiguity["ambiguous"]:
        verdict = "COMPRESSION_RESULT_AMBIGUOUS"
    elif per_seed_pass and mean_pass:
        if d0 < 0 and d1 < 0:
            verdict = (
                "REPLICATED_NON_INFERIOR_COMPRESSION_WITH_FAVORABLE_POINT_ESTIMATES"
            )
        else:
            verdict = "END_TO_END_SMALL_HEAD_COMPRESSION_REPLICATED_NON_INFERIOR"
    else:
        verdict = "COMPRESSION_REPLICATION_INCONCLUSIVE"
    payload = {
        "D0": d0,
        "D1": d1,
        "mean_D": mean_d,
        "per_seed_gate": float(NI_GATE),
        "mean_gate": float(NI_MEAN_GATE),
        "per_seed_pass": per_seed_pass,
        "mean_pass": mean_pass,
        "S1": float(s1["best_valid_mae"]),
        "B1": float(V4_SEED1_VALID),
        "best_epoch": int(s1["best_epoch"]),
        "optimization_ambiguity": ambiguity,
        "verdict": verdict,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage2_replication_decision.json", payload)
    return payload


def two_seed_summary() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    refs = {0: V4_SEED0_VALID, 1: V4_SEED1_VALID}
    paths = {
        0: RESULTS_DIR / "stage1_smallhead_seed0.json",
        1: RESULTS_DIR / "stage2_smallhead_seed1.json",
    }
    for seed in (0, 1):
        if not paths[seed].exists():
            continue
        summary = _read_json(paths[seed])
        rows.append(
            {
                "seed": int(seed),
                "baseline_valid": float(refs[seed]),
                "smallhead_valid": float(summary["best_valid_mae"]),
                "D": float(summary["best_valid_mae"] - refs[seed]),
                "best_epoch": int(summary["best_epoch"]),
                "epochs_run": int(summary["epochs_run"]),
                "params": int(summary["parameters"]),
            }
        )
    _write_csv(
        RESULTS_DIR / "two_seed_summary.csv",
        rows,
        ("seed", "baseline_valid", "smallhead_valid", "D", "best_epoch", "epochs_run", "params"),
    )
    return {"rows": rows}


def _v4_predictions(valid_data: Sequence[Data], seed: int) -> np.ndarray:
    state_path = V4_SEED0_STATE if seed == 0 else V4_SEED1_STATE
    baseline = build_baseline(seed)
    baseline.load_state_dict(torch.load(state_path, map_location="cpu"))
    baseline.eval()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(baseline(batch).view(-1).cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def common_input_bulk(seed: int) -> dict[str, Any]:
    stage_path = (
        RESULTS_DIR / "stage1_smallhead_seed0.json"
        if seed == 0
        else RESULTS_DIR / "stage2_smallhead_seed1.json"
    )
    summary = _read_json(stage_path)
    train_records, valid_records = load_train_valid_records()
    _train_data, valid_data, _audit = build_encoded_records()
    from collections import Counter

    train_freq: Counter[bytes] = Counter()
    for record in train_records:
        for patch in record.patches:
            train_freq[patch.typed_certificate] += 1

    def rare_le5_ratio(records) -> np.ndarray:
        out = []
        for record in records:
            freqs = [
                int(train_freq.get(patch.typed_certificate, 0))
                for patch in record.patches
            ]
            out.append(float(np.mean(np.asarray(freqs) <= 5)) if freqs else 0.0)
        return np.asarray(out, dtype=np.float64)

    train_rare = rare_le5_ratio(train_records)
    valid_rare = rare_le5_ratio(valid_records)
    targets = np.asarray(summary["valid_targets"], dtype=np.float64)
    cand_pred = np.asarray(summary["valid_predictions"], dtype=np.float64)
    v4_pred = _v4_predictions(valid_data, seed)
    threshold = float(np.percentile(train_rare, 80.0))
    bulk = valid_rare < threshold
    v4_mae = float(np.mean(np.abs(targets - v4_pred)))
    cand_mae = float(np.mean(np.abs(targets - cand_pred)))
    bulk_v4 = float(np.mean(np.abs(targets[bulk] - v4_pred[bulk])))
    bulk_cand = float(np.mean(np.abs(targets[bulk] - cand_pred[bulk])))
    payload = {
        "seed": int(seed),
        "bulk_definition": (
            "target-independent: valid molecules whose train-derived "
            "rare_le5_ratio is below the 80th percentile of the train split"
        ),
        "rare_le5_threshold": threshold,
        "n_bulk_valid": int(bulk.sum()),
        "n_valid": int(len(valid_rare)),
        "v4_valid_mae": v4_mae,
        "smallhead_valid_mae": cand_mae,
        "valid_mae_diff_smallhead_minus_v4": float(cand_mae - v4_mae),
        "bulk_v4_mae": bulk_v4,
        "bulk_smallhead_mae": bulk_cand,
        "bulk_mae_diff_smallhead_minus_v4": float(bulk_cand - bulk_v4),
        "bulk_caution_threshold": float(BULK_CAUTION),
        "bulk_caution": bool((bulk_cand - bulk_v4) > BULK_CAUTION),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"common_input_bulk_seed{seed}.json", payload)
    return payload


def representation_drift() -> dict[str, Any]:
    """Descriptive only: R drift between baseline and small-head seed0 ckpts."""
    _train_data, valid_data, _audit = build_encoded_records()
    baseline = build_baseline(0)
    baseline.load_state_dict(torch.load(V4_SEED0_STATE, map_location="cpu"))
    baseline.eval()
    small_path = RESULTS_DIR / "states/smallhead_seed0_selection_state.pt"
    if not small_path.exists():
        return {"error": "seed0 small-head checkpoint missing", "official_test_loaded": False}
    small = build_smallhead(0)
    small.load_state_dict(torch.load(small_path, map_location="cpu"))
    small.eval()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    r_base: list[np.ndarray] = []
    r_small: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            r_base.append(baseline.encode(batch).cpu().numpy())
            r_small.append(small.encode(batch).cpu().numpy())
    a = np.concatenate(r_base).astype(np.float64)
    b = np.concatenate(r_small).astype(np.float64)
    diff = b - a
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    cosine = np.sum(a * b, axis=1) / np.clip(norm_a * norm_b, 1e-12, None)
    blocks = {
        "unary": (0, 97),
        "pair": (97, 262),
        "global": (262, 294),
        "topology": (294, 302),
    }
    block_norms = {
        name: {
            "baseline_mean_norm": float(np.linalg.norm(a[:, lo:hi], axis=1).mean()),
            "smallhead_mean_norm": float(np.linalg.norm(b[:, lo:hi], axis=1).mean()),
        }
        for name, (lo, hi) in blocks.items()
    }
    payload = {
        "descriptive_only": True,
        "not_a_gate": True,
        "split": "official-valid",
        "n": int(a.shape[0]),
        "normalized_L2_drift_mean": float(
            (np.linalg.norm(diff, axis=1) / np.clip(norm_a, 1e-12, None)).mean()
        ),
        "normalized_L2_drift_max": float(
            (np.linalg.norm(diff, axis=1) / np.clip(norm_a, 1e-12, None)).max()
        ),
        "mean_cosine_similarity": float(cosine.mean()),
        "block_norms": block_norms,
        "note": (
            "the head participates in joint training, so the small head may "
            "change the learned backbone representation; this is not a gate"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "representation_drift.json", payload)
    return payload


# ---------------------------------------------------------------------------
# answers / final decision / figures
# ---------------------------------------------------------------------------


def answers_q1_q18() -> dict[str, Any]:
    audit = _read_json(RESULTS_DIR / "parameter_audit.json")
    stage1 = _read_json(RESULTS_DIR / "stage1_smallhead_seed0.json")
    decision0 = _read_json(RESULTS_DIR / "stage1_decision.json")
    compute = (
        _read_json(RESULTS_DIR / "compute_audit.json")
        if (RESULTS_DIR / "compute_audit.json").exists()
        else {}
    )
    ans: dict[str, Any] = {
        "Q1": {
            "baseline_total": int(audit["baseline_total"]),
            "baseline_head": int(audit["baseline_head"]),
        },
        "Q2": (
            "302 -> 13 -> 13 -> 1, raw R, direct yhat=f(R), ReLU, no LayerNorm, "
            "no dropout, no residual"
        ),
        "Q3": int(audit["smallhead_head"]),
        "Q4": int(audit["smallhead_total"]),
        "Q5": {
            "absolute": int(audit["absolute_reduction"]),
            "relative_total": float(audit["relative_total_reduction"]),
        },
        "Q6": float(audit["relative_head_reduction"]),
        "Q7": {
            "exact_match": bool(
                _read_json(RESULTS_DIR / "initialization_match_seed0.json")[
                    "exact_equal_to_baseline_init"
                ]
            ),
            "max_abs_diff": float(
                _read_json(RESULTS_DIR / "initialization_match_seed0.json")[
                    "max_abs_diff"
                ]
            ),
        },
        "Q8": {
            "best_valid": float(stage1["best_valid_mae"]),
            "best_epoch": int(stage1["best_epoch"]),
        },
        "Q9": float(decision0["D0"]),
        "Q10": bool(stage1["passes_non_inferiority"]),
        "Q11": decision0["optimization_ambiguity"],
        "Q12": {
            "forward_ms_baseline": compute.get("forward_ms_baseline"),
            "forward_ms_smallhead": compute.get("forward_ms_smallhead"),
            "forward_change_ratio": compute.get("forward_change_ratio"),
            "params_baseline": int(audit["baseline_total"]),
            "params_smallhead": int(audit["smallhead_total"]),
            "wall_clock_s_seed0": float(stage1["wall_clock_s"]),
            "optimizer_steps_seed0": int(stage1["optimizer_steps"]),
        },
        "Q13": bool(decision0["buy_seed1"]),
        "Q18": decision0["verdict"],
    }
    stage2_path = RESULTS_DIR / "stage2_smallhead_seed1.json"
    if stage2_path.exists():
        stage2 = _read_json(stage2_path)
        decision1 = _read_json(RESULTS_DIR / "stage2_replication_decision.json")
        ans["Q14"] = float(stage2["best_valid_mae"])
        ans["Q15"] = float(decision1["D1"])
        ans["Q16"] = float(decision1["mean_D"])
        ans["Q17"] = bool(decision1["per_seed_pass"] and decision1["mean_pass"])
        ans["Q18"] = decision1["verdict"]
    else:
        ans["Q14"] = "not run (seed1 not purchased)"
        ans["Q15"] = None
        ans["Q16"] = None
        ans["Q17"] = False
    _write_json(RESULTS_DIR / "answers_q1_q18.json", ans)
    return ans


def final_decision() -> dict[str, Any]:
    decision0 = _read_json(RESULTS_DIR / "stage1_decision.json")
    stage2_path = RESULTS_DIR / "stage2_replication_decision.json"
    if stage2_path.exists():
        decision1 = _read_json(stage2_path)
        verdict = decision1["verdict"]
        payload = {
            "final_verdict": verdict,
            "stage1": decision0,
            "stage2": decision1,
            "official_test_loaded": False,
            "next_step": (
                "if replicated non-inferior, the ~82.1K model becomes a new "
                "compact baseline candidate; released parameters remain unused; "
                "official test still locked"
            ),
        }
    else:
        payload = {
            "final_verdict": decision0["verdict"],
            "stage1": decision0,
            "stage2": None,
            "official_test_loaded": False,
            "next_step": (
                "seed1 not purchased; keep the 99,613 optimized compact-v4 as "
                "the baseline; official test still locked"
            ),
        }
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    return payload


def make_figures() -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return {"figures": [], "error": repr(exc)}
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    # Figure 1: parameter allocation baseline vs small-head
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = ["compact-v4\n(99,613)", "compact-v4-smallhead\n(82,115)"]
    backbones = [EXPECTED_BASELINE_TOTAL - EXPECTED_BASELINE_HEAD, EXPECTED_BASELINE_TOTAL - EXPECTED_BASELINE_HEAD]
    heads = [EXPECTED_BASELINE_HEAD, EXPECTED_SMALL_HEAD]
    ax.bar(labels, backbones, color="#4c72b0", label="backbone (unchanged)")
    ax.bar(labels, heads, bottom=backbones, color="#dd8452", label="graph head")
    for index, (backbone, head) in enumerate(zip(backbones, heads)):
        ax.text(index, backbone / 2, f"{backbone:,}", ha="center", va="center", fontsize=8)
        ax.text(index, backbone + head / 2, f"{head:,}", ha="center", va="center", fontsize=8)
    ax.set_ylabel("parameters")
    ax.set_title("Parameter allocation: 21,633 -> 4,135 head (-17,498)", fontsize=10)
    ax.legend(fontsize=8)
    path = FIGURE_DIR / "figure1_parameter_allocation.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))

    # Figure 2: seed0 learning curves
    curve_path = CURVE_DIR / "smallhead_seed0_curve.csv"
    if curve_path.exists():
        rows = list(csv.DictReader(curve_path.open(encoding="utf-8")))
        epochs = [int(row["epoch"]) for row in rows]
        train = [float(row["train_mae"]) for row in rows]
        valid = [float(row["valid_mae"]) for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, train, label="train MAE", color="#4c72b0")
        ax.plot(epochs, valid, label="valid MAE", color="#dd8452")
        ax.axhline(V4_SEED0_VALID, ls="--", color="grey", label="baseline seed0 valid")
        ax.set_xlabel("epoch")
        ax.set_ylabel("L1 MAE")
        ax.set_title("compact-v4-smallhead seed0 learning curves", fontsize=10)
        ax.legend(fontsize=8)
        path = FIGURE_DIR / "figure2_seed0_curves.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

    # Figure 3: only if seed1 ran
    s0 = RESULTS_DIR / "stage1_smallhead_seed0.json"
    s1 = RESULTS_DIR / "stage2_smallhead_seed1.json"
    if s0.exists() and s1.exists():
        s0d = _read_json(s0)
        s1d = _read_json(s1)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        x = np.arange(2)
        width = 0.35
        baseline_vals = [V4_SEED0_VALID, V4_SEED1_VALID]
        small_vals = [float(s0d["best_valid_mae"]), float(s1d["best_valid_mae"])]
        ax.bar(x - width / 2, baseline_vals, width, label="baseline", color="#4c72b0")
        ax.bar(x + width / 2, small_vals, width, label="small-head", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(["seed0", "seed1"])
        ax.set_ylabel("official-valid MAE")
        ax.set_title("seed0 / seed1: baseline vs small-head", fontsize=10)
        ax.legend(fontsize=8)
        path = FIGURE_DIR / "figure3_valid_mae.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

    return {"figures": created}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_stage0() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    architecture_lock()
    baseline_inventory()
    baseline_anchor()
    parameter_audit()
    gate = integrity_gates()
    if not gate["passed"]:
        raise RuntimeError(f"integrity gates failed: {gate['results']}")
    compute_audit()
    make_figures()
    print("Stage 0 complete.", flush=True)
    return gate


def run_all() -> None:
    run_stage0()
    stage1_smallhead_seed0()
    stage1_decision_out = stage1_decision()
    common_input_bulk(0)
    if stage1_decision_out["buy_seed1"]:
        stage2_smallhead_seed1()
        stage2_decision()
        common_input_bulk(1)
        two_seed_summary()
        representation_drift()
    answers_q1_q18()
    final_decision()
    make_figures()
    print("Experiment complete.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "architecture_lock",
            "baseline",
            "baseline_anchor",
            "param_audit",
            "init_match_seed0",
            "init_match_seed1",
            "integrity",
            "compute_audit",
            "stage0",
            "stage1_seed0",
            "stage1_decision",
            "bulk_seed0",
            "stage2_seed1",
            "stage2_decision",
            "bulk_seed1",
            "two_seed",
            "drift",
            "answers",
            "final",
            "figures",
            "all",
        ],
    )
    args = parser.parse_args(argv)
    stage = args.stage
    _configure_determinism()
    if stage == "architecture_lock":
        print(json.dumps(architecture_lock(), indent=2))
    elif stage == "baseline":
        print(json.dumps(baseline_inventory(), indent=2))
    elif stage == "baseline_anchor":
        print(json.dumps(baseline_anchor(), indent=2))
    elif stage == "param_audit":
        print(json.dumps(parameter_audit(), indent=2))
    elif stage == "init_match_seed0":
        print(json.dumps(initialization_match_seed0(), indent=2))
    elif stage == "init_match_seed1":
        print(json.dumps(initialization_match_seed1(), indent=2))
    elif stage == "integrity":
        print(json.dumps(integrity_gates(), indent=2))
    elif stage == "compute_audit":
        print(json.dumps(compute_audit(), indent=2))
    elif stage == "stage0":
        run_stage0()
    elif stage == "stage1_seed0":
        print(stage1_smallhead_seed0()["delta_D0"])
    elif stage == "stage1_decision":
        print(json.dumps(stage1_decision(), indent=2))
    elif stage == "bulk_seed0":
        print(json.dumps(common_input_bulk(0), indent=2))
    elif stage == "stage2_seed1":
        print(stage2_smallhead_seed1()["delta_D1"])
    elif stage == "stage2_decision":
        print(json.dumps(stage2_decision(), indent=2))
    elif stage == "bulk_seed1":
        print(json.dumps(common_input_bulk(1), indent=2))
    elif stage == "two_seed":
        print(json.dumps(two_seed_summary(), indent=2))
    elif stage == "drift":
        print(json.dumps(representation_drift(), indent=2))
    elif stage == "answers":
        print(json.dumps(answers_q1_q18(), indent=2))
    elif stage == "final":
        print(json.dumps(final_decision(), indent=2))
    elif stage == "figures":
        print(json.dumps(make_figures(), indent=2))
    else:
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
