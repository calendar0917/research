"""ZINC local 16-D patch-token channel necessity (frozen knockout + B-Null).

Question
--------
The strong ZINC mixed backbone (cell-A geometry: radius-2 rooted patches,
``patch_cont`` shell descriptor, typed parent token, chemistry pair relation,
T=2 weight-tied recurrent pair--centre, global encoder, topology hinge and a
fixed small raw head) feeds the downstream patch encoder a **local 16-D patch
token** ``e_patch in R^16``.  Historically that token has several
implementations, all of which are *molecule-/patch-dependent*:

* ``typed_lookup`` -- a learned row per exact rooted certificate (cell A / A0);
* ``shared_bag`` -- a shared connectivity-free bag encoder (B-Bag);
* ``shared_structural`` -- a shared radius-2 structural message-passing encoder
  (B-Full).

The three references **share one downstream backbone** and differ *only* in the
local-token generator.  This experiment asks the channel-necessity question:

    Can the molecule-dependent local 16-D channel be removed entirely while the
    strong backbone stays strong?

It deliberately does **not** re-invest the released parameter budget: the
parameter drop is the experimental variable, not a confound.

Stage A (eval-only frozen knockout)
-----------------------------------
For each frozen reference soup (A0, B-Bag, B-Full) the downstream backbone is
kept frozen and the local token is forced to ``0`` (A1) or to a train-side
constant (A2), optionally permuted within each molecule (A3).  This is a cheap
screening gate with distribution shift; it is never a causal claim.

Stage B (B-Null retraining)
---------------------------
``patch_representation="null"`` keeps the 16-D slot but feeds an exact zero
vector for every patch and instantiates **no** local-token generator (its
parameters are genuinely absent).  Only B-Null seed0 is trained.

Stage C (Constant-16 retraining)
--------------------------------
``patch_representation="constant"`` feeds one trainable graph-wide 16-vector
(16 params) to every patch of every molecule.  Only run if B-Null seed0 has
hope.

Official ZINC test is **never** loaded.  All comparisons use the official valid
split only.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_pair_centre as rec
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity_decomposition as cd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_shared_bag_patch_encoder as sbpe
from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_structural_patch_encoder as sspe,
)

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/local_token_null"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"

PROTOCOL_VERSION = "local_token_null_v1"

# --- frozen cell-A geometry (identical for A0 / B-Bag / B-Full / Null) -------
CELL = "A"
H_DIM = 64
Q_DIM = 16
PATCH_ENCODER_HIDDEN = 64
GLOBAL_ENCODER_HIDDEN = 32
TYPED_VOCABULARY_SIZE = 6785
PARENT_VOCABULARY_SIZE = 32
TOKEN_WIDTH = 16

# --- matched reference parameter counts -------------------------------------
DOWNSTREAM_BACKBONE_PARAMS = 49343  # shared by A0 / B-Bag / B-Full / Null
A0_TOTAL_PARAMS = 85763  # typed lookup 36,420
BAG_TOTAL_PARAMS = 84511  # shared bag encoder 35,168
FULL_TOTAL_PARAMS = 84495  # shared structural encoder 35,152
NULL_TOTAL_PARAMS = 49343  # no local-token generator at all
CONSTANT_EXTRA_PARAMS = TOKEN_WIDTH  # one graph-wide trainable 16-vector
LOCAL_TOKEN_PARAMS = {
    "A0": A0_TOTAL_PARAMS - DOWNSTREAM_BACKBONE_PARAMS,  # 36,420
    "B-Bag": BAG_TOTAL_PARAMS - DOWNSTREAM_BACKBONE_PARAMS,  # 35,168
    "B-Full": FULL_TOTAL_PARAMS - DOWNSTREAM_BACKBONE_PARAMS,  # 35,152
}

# --- pre-registered matched references (valid soup, deterministic A100) ------
REFERENCE_SOUP = {
    "A0": {0: 0.12470435495121637, 1: 0.12803170100128045},
    "B-Bag": {0: 0.12738177864899625, 1: 0.12022926583880325},
    "B-Full": {0: 0.11981802638241788, 1: 0.11812638340244302},
}
REFERENCE_RAW = {
    "A0": {0: 0.12970963285310427, 1: 0.13141501806577435},
    "B-Bag": {0: 0.12957351383467902, 1: 0.12821901564946164},
    "B-Full": {0: 0.12532435323769459, 1: 0.12439350808685412},
}

# --- pre-registered experiment-economy gates (NOT significance thresholds) ---
STAGE_A_CATASTROPHIC_DELTA = 0.13  # ~0.12 -> > 0.25 is catastrophic
STAGE_A_MODERATE_DELTA = 0.06  # 0.14-0.18 band: worth retraining Null
STAGE_A_SMALL_DELTA = 0.02  # < 0.02 strongly supports B-Null

B_NULL_STRONG = 0.13
B_NULL_MILD = 0.15
B_NULL_SUBSTANTIAL = 0.20

SEEDS = (0, 1)
DETERMINISTIC = False

REFERENCE_NAMES = ("A0", "B-Bag", "B-Full")


def _set_deterministic(enabled: bool) -> None:
    if enabled:
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# io helpers
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


def _environment_fingerprint(device: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# model builders
# ---------------------------------------------------------------------------


def build_reference(name: str, seed: int = 0) -> nn.Module:
    """Build a frozen strong reference (A0 / B-Bag / B-Full)."""
    name = str(name)
    sspe._set_deterministic(DETERMINISTIC)
    if name == "A0":
        return cd.build_cell(CELL, int(seed))
    if name == "B-Bag":
        return sbpe.build_candidate(int(seed))
    if name == "B-Full":
        return sspe.build_candidate(int(seed))
    raise ValueError(f"unknown reference {name!r}; expected {REFERENCE_NAMES}")


def _build_representation(
    patch_representation: str, seed: int, **representation_kwargs: Any
) -> nn.Module:
    """Cell-A backbone with an explicit local-token representation.

    Construction mirrors ``cd.build_cell('A', seed)``: the canonical baseline is
    instantiated first, every shape-matching shared tensor is copied bit-exactly,
    and the fixed small head is re-drawn from the historical ``head_seed=0``
    stream.  Only ``patch_representation`` (and its width arguments) changes.
    """
    baseline = shead.build_baseline(int(seed))
    baseline_state = {
        key: value.detach().clone() for key, value in baseline.state_dict().items()
    }
    shead._seed_everything(int(seed))
    kwargs = shead._base_kwargs()
    kwargs["patch_hidden"] = H_DIM
    kwargs["patch_encoder_hidden"] = PATCH_ENCODER_HIDDEN
    kwargs["global_encoder_hidden"] = GLOBAL_ENCODER_HIDDEN
    kwargs.pop("pair_hidden", None)
    with cap._construction_guards(H_DIM, Q_DIM):
        model = rec.PatchPathRecurrentPairCentreModel(
            TYPED_VOCABULARY_SIZE,
            PARENT_VOCABULARY_SIZE,
            pair_hidden=Q_DIM,
            recurrence_rounds=rec.RECURRENCE_ROUNDS,
            recurrence_enabled=True,
            recurrence_mode="refresh",
            patch_representation=str(patch_representation),
            **representation_kwargs,
            **kwargs,
        )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in baseline_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)
    torch.manual_seed(int(shead.SMALL_HEAD_SEED))
    model.head = shead.GenericReader(
        int(model.unified_graph_width), shead.SMALL_HEAD_HIDDEN
    )
    return model


def build_null(seed: int = 0) -> nn.Module:
    """Strong backbone with an exact zero local 16-D slot and no generator."""
    return _build_representation("null", int(seed))


def build_constant(seed: int = 0) -> nn.Module:
    """Strong backbone with one trainable graph-wide 16-D local token."""
    return _build_representation("constant", int(seed))


BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    "A0": lambda seed: build_reference("A0", seed),
    "B-Bag": lambda seed: build_reference("B-Bag", seed),
    "B-Full": lambda seed: build_reference("B-Full", seed),
    "Null": build_null,
    "Constant": build_constant,
}

EXPECTED_TOTAL = {
    "A0": A0_TOTAL_PARAMS,
    "B-Bag": BAG_TOTAL_PARAMS,
    "B-Full": FULL_TOTAL_PARAMS,
    "Null": NULL_TOTAL_PARAMS,
    "Constant": NULL_TOTAL_PARAMS + CONSTANT_EXTRA_PARAMS,
}


# ---------------------------------------------------------------------------
# data / loaders
# ---------------------------------------------------------------------------


def reference_records(name: str) -> tuple[list[Data], list[Data], dict[str, Any]]:
    name = str(name)
    if name == "A0":
        train_data, valid_data, audit = rec.load_encoded()
    elif name == "B-Bag":
        train_data, valid_data, audit = sbpe.build_encoded_records()
    elif name == "B-Full":
        train_data, valid_data, audit = sspe.build_encoded_records()
    else:
        raise ValueError(f"unknown reference {name!r}")
    return list(train_data), list(valid_data), dict(audit)


def reference_loader(name: str, data: Sequence[Data]):
    name = str(name)
    if name == "A0":
        return zpp._make_loader(list(data), 128, False, 0)
    if name == "B-Bag":
        return sbpe._make_bag_loader(list(data), 128, False, 0)
    if name == "B-Full":
        return sspe._make_struct_loader(list(data), 128, False, 0)
    raise ValueError(f"unknown reference {name!r}")


def reference_soup_path(name: str, seed: int) -> Path:
    if name == "A0":
        return cd.SOUP_DIR / f"cell_A_seed{int(seed)}_top5_soup.pt"
    if name == "B-Bag":
        return sbpe.SOUP_DIR / f"sbpe_seed{int(seed)}_top5_soup.pt"
    if name == "B-Full":
        return sspe.SOUP_DIR / f"sspe_seed{int(seed)}_top5_soup.pt"
    raise ValueError(f"unknown reference {name!r}")


def reference_soup_json(name: str, seed: int) -> Path:
    if name == "A0":
        return cd.RESULTS_DIR / f"soup_cell_A_seed{int(seed)}.json"
    if name == "B-Bag":
        return sbpe.RESULTS_DIR / f"soup_sbpe_seed{int(seed)}.json"
    if name == "B-Full":
        return sspe.RESULTS_DIR / f"soup_sspe_seed{int(seed)}.json"
    raise ValueError(f"unknown reference {name!r}")


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate(
    model: nn.Module, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
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


def train_side_mean_token(
    model: nn.Module, train_loader, device: torch.device, max_batches: int | None = None
) -> torch.Tensor:
    """Global mean of the frozen generator output over the train split only.

    The intervention constant must never be fitted on the valid targets.  This
    averages ``e_patch`` over every patch of the official-*train* molecules with
    the frozen generator weights (no targets involved).
    """
    model.set_local_token_intervention(None)
    model.eval()
    total: torch.Tensor | None = None
    count = 0
    with torch.no_grad():
        for index, batch in enumerate(train_loader):
            if max_batches is not None and index >= int(max_batches):
                break
            batch = batch.to(device)
            value = model._patch_token_value(batch).detach().double()
            row = value.sum(dim=0)
            total = row if total is None else total + row
            count += int(value.shape[0])
    if total is None:
        raise RuntimeError("no train batches available for the mean token")
    return (total / max(count, 1)).to(dtype=torch.float32)


def _reference_state(name: str, seed: int) -> dict[str, torch.Tensor]:
    path = reference_soup_path(name, int(seed))
    if not path.exists():
        raise FileNotFoundError(
            f"frozen soup for {name} seed{seed} not found at {path}; "
            "sync the reference checkpoint before running Stage A"
        )
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def parameter_accounting() -> dict[str, Any]:
    rows: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "downstream_backbone_params": DOWNSTREAM_BACKBONE_PARAMS,
        "total_reference": {},
        "removed_local_token_generator": {},
        "total_candidate": {},
        "candidate_minus_downstream": {},
        "official_test_loaded": False,
    }
    for name in ("A0", "B-Bag", "B-Full", "Null", "Constant"):
        model = BUILDERS[name](0)
        total = _n_params(model)
        rows["total_reference" if name in REFERENCE_NAMES else "total_candidate"][
            name
        ] = int(total)
        rows["candidate_minus_downstream"][name] = int(
            total - DOWNSTREAM_BACKBONE_PARAMS
        )
        if name in REFERENCE_NAMES:
            rows["removed_local_token_generator"][name] = int(
                total - DOWNSTREAM_BACKBONE_PARAMS
            )
        if total != EXPECTED_TOTAL[name]:
            raise RuntimeError(
                f"{name} total {total} != expected {EXPECTED_TOTAL[name]}"
            )
    rows["null_parameter_delta_vs"] = {
        name: int(NULL_TOTAL_PARAMS - EXPECTED_TOTAL[name])
        for name in REFERENCE_NAMES
    }
    rows["constant_extra_params"] = CONSTANT_EXTRA_PARAMS
    return rows


# ---------------------------------------------------------------------------
# Stage A -- frozen knockout on the existing strong checkpoints
# ---------------------------------------------------------------------------


def stage_a_reference(
    name: str, seed: int, device: str = "cpu", train_batches: int | None = None
) -> dict[str, Any]:
    device_obj = torch.device(device)
    train_data, valid_data, _audit = reference_records(name)
    train_loader = reference_loader(name, train_data)
    valid_loader = reference_loader(name, valid_data)
    model = build_reference(name, int(seed)).to(device_obj)
    state = _reference_state(name, int(seed))
    model.load_state_dict(state, strict=True)

    original_targets, original_preds = _evaluate(model, valid_loader, device_obj)
    original_mae = _mae(original_targets, original_preds)

    constant = train_side_mean_token(
        model, train_loader, device_obj, max_batches=train_batches
    )
    recorded = (
        _read_json(reference_soup_json(name, int(seed)))
        if reference_soup_json(name, int(seed)).exists()
        else {}
    )

    interventions: dict[str, Any] = {}
    for mode in ("zero", "constant", "permute"):
        kwargs: dict[str, Any] = {}
        if mode == "constant":
            kwargs["constant"] = constant
        if mode == "permute":
            kwargs["permute_seed"] = 20260919 + int(seed)
        model.set_local_token_intervention(mode, **kwargs)
        targets, preds = _evaluate(model, valid_loader, device_obj)
        interventions[mode] = {
            "valid_mae": _mae(targets, preds),
            "delta_vs_original": _mae(targets, preds) - original_mae,
            "prediction_sha256": _predictions_sha256(preds),
        }
    model.clear_local_token_intervention()

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reference": str(name),
        "seed": int(seed),
        "parameters": int(_n_params(model)),
        "local_token_generator_params": int(
            LOCAL_TOKEN_PARAMS.get(str(name), 0)
        ),
        "original_valid_mae": float(original_mae),
        "recorded_soup_valid_mae": (
            None if not recorded else float(recorded["top5_soup_valid_mae"])
        ),
        "recomputed_minus_recorded": (
            None
            if not recorded
            else float(original_mae - float(recorded["top5_soup_valid_mae"]))
        ),
        "recomputed_equals_recorded": (
            None
            if not recorded
            else bool(abs(original_mae - float(recorded["top5_soup_valid_mae"])) < 1e-6)
        ),
        "train_side_mean_token": [float(v) for v in constant.tolist()],
        "train_side_mean_token_norm": float(constant.norm()),
        "interventions": interventions,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"stage_a_{str(name).lower()}_seed{int(seed)}.json", payload)
    return payload


def _predictions_sha256(predictions: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(
        np.asarray(predictions, dtype=np.float64).tobytes()
    ).hexdigest()


def stage_a(
    seeds: Sequence[int] = (0,), device: str = "cpu", train_batches: int | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "A_frozen_knockout",
        "seeds": [int(s) for s in seeds],
        "references": {},
        "official_test_loaded": False,
    }
    for name in REFERENCE_NAMES:
        payload["references"][name] = {}
        for seed in seeds:
            payload["references"][name][str(int(seed))] = stage_a_reference(
                name, int(seed), device=device, train_batches=train_batches
            )
    _write_json(RESULTS_DIR / "stage_a_frozen_knockout.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage A interpretation gate (experiment-economy only)
# ---------------------------------------------------------------------------


def stage_a_gate() -> dict[str, Any]:
    path = RESULTS_DIR / "stage_a_frozen_knockout.json"
    payload = _read_json(path)
    per_reference: dict[str, Any] = {}
    for name in REFERENCE_NAMES:
        rows = payload["references"].get(name, {})
        if not rows:
            continue
        seed_rows = list(rows.values())
        original = float(np.mean([row["original_valid_mae"] for row in seed_rows]))
        zero = float(
            np.mean([row["interventions"]["zero"]["valid_mae"] for row in seed_rows])
        )
        constant = float(
            np.mean(
                [row["interventions"]["constant"]["valid_mae"] for row in seed_rows]
            )
        )
        zero_delta = zero - original
        constant_delta = constant - original
        worst = max(zero_delta, constant_delta)
        if worst > STAGE_A_CATASTROPHIC_DELTA:
            verdict = "CATASTROPHIC_STOP"
        elif worst >= STAGE_A_MODERATE_DELTA:
            verdict = "MODERATE_RETRAIN_NULL"
        elif worst < STAGE_A_SMALL_DELTA:
            verdict = "SMALL_RETRAIN_NULL"
        else:
            verdict = "SMALL_MEDIUM_RETRAIN_NULL"
        per_reference[name] = {
            "original_valid_mae": original,
            "zero_valid_mae": zero,
            "constant_valid_mae": constant,
            "zero_delta": zero_delta,
            "constant_delta": constant_delta,
            "worst_delta": worst,
            "verdict": verdict,
        }
    verdicts = [row["verdict"] for row in per_reference.values()]
    if not verdicts:
        overall = "NO_STAGE_A_RESULTS"
    elif all(v == "CATASTROPHIC_STOP" for v in verdicts):
        overall = "CATASTROPHIC_STOP"
    elif any(v == "CATASTROPHIC_STOP" for v in verdicts):
        overall = "MIXED_INVESTIGATE"
    else:
        overall = "RETRAIN_NULL_AUTHORIZED"
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "per_reference": per_reference,
        "overall": overall,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage_a_gate.json", result)
    return result


# ---------------------------------------------------------------------------
# Stage B/C -- training the Null / Constant backbone
# ---------------------------------------------------------------------------


def tag_for(representation: str) -> str:
    return f"lt_{str(representation).lower()}"


def expected_total_for(representation: str) -> int:
    rep = str(representation)
    if rep == "null":
        return NULL_TOTAL_PARAMS
    if rep == "constant":
        return NULL_TOTAL_PARAMS + CONSTANT_EXTRA_PARAMS
    raise ValueError(f"unknown representation {rep!r}")


def _train_representation(
    representation: str, seed: int, device: str = "cpu"
) -> dict[str, Any]:
    rep = str(representation)
    builder = build_null if rep == "null" else build_constant
    tag = tag_for(rep)
    train_data, valid_data, _ = rec.load_encoded()
    original = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = (
        CURVE_DIR,
        STATE_DIR,
        RUNS_DIR,
    )
    snapshot_dir = SNAPSHOT_DIR / f"{tag}_seed{int(seed)}"
    started = time.perf_counter()
    try:
        summary = shead.train_model(
            build_fn=lambda s: builder(int(s)),
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=tag,
            save_state=True,
            real_batch_identity=False,
            expected_total=expected_total_for(rep),
            snapshot_dir=snapshot_dir,
            device=device,
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original
    summary = dict(summary)
    summary["representation"] = rep
    summary["epoch_time_s"] = float(summary["wall_clock_s"]) / max(
        int(summary["epochs_run"]), 1
    )
    summary["peak_gpu_memory_mb"] = (
        float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if torch.cuda.is_available()
        else 0.0
    )
    summary["outer_wall_clock_s"] = float(time.perf_counter() - started)
    summary["official_test_loaded"] = False
    _write_json(RUNS_DIR / f"{tag}_seed{int(seed)}.json", summary)
    soup(rep, int(seed))
    return summary


def train_null(seed: int, device: str = "cpu") -> dict[str, Any]:
    return _train_representation("null", int(seed), device=device)


def train_constant(seed: int, device: str = "cpu") -> dict[str, Any]:
    return _train_representation("constant", int(seed), device=device)


def _valid_loader() -> Any:
    _train_data, valid_data, _ = rec.load_encoded()
    return zpp._make_loader(list(valid_data), 128, False, 0)


def soup(representation: str, seed: int) -> dict[str, Any]:
    rep = str(representation)
    tag = tag_for(rep)
    summary = _read_json(RUNS_DIR / f"{tag}_seed{int(seed)}.json")
    top = vd._top5_epochs(summary)
    soup_state = vd.build_soup_state(top)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = SOUP_DIR / f"{tag}_seed{int(seed)}_top5_soup.pt"
    torch.save(soup_state, soup_path)

    loader = _valid_loader()
    model = (build_null if rep == "null" else build_constant)(int(seed))
    selection_path = STATE_DIR / f"{tag}_seed{int(seed)}_selection_state.pt"
    targets, best_preds = vd._predict_state(
        model, torch.load(selection_path, map_location="cpu", weights_only=True), loader
    )
    _t2, soup_preds = vd._predict_state(model, soup_state, loader)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "representation": rep,
        "tag": tag,
        "seed": int(seed),
        "top5_epochs": [int(row["epoch"]) for row in top],
        "top5_valid_mae": [float(row["valid_mae"]) for row in top],
        "best_epoch": int(summary["best_epoch"]),
        "best_checkpoint_valid_mae": _mae(targets, best_preds),
        "top5_soup_valid_mae": _mae(targets, soup_preds),
        "train_loss_at_best": float(summary.get("train_loss_at_best", float("nan"))),
        "epochs_run": int(summary.get("epochs_run", 0)),
        "wall_clock_s": float(summary.get("wall_clock_s", float("nan"))),
        "peak_gpu_memory_mb": float(summary.get("peak_gpu_memory_mb", 0.0)),
        "parameters": int(_n_params(model)),
        "official_test_loaded": False,
        "valid_targets": targets.tolist(),
        "best_predictions": best_preds.tolist(),
        "soup_predictions": soup_preds.tolist(),
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{int(seed)}.json", payload)
    return payload


def train_queue(representations: Sequence[str], seeds: Sequence[int], device: str) -> None:
    for representation in representations:
        for seed in seeds:
            tag = tag_for(representation)
            if (RUNS_DIR / f"{tag}_seed{int(seed)}.json").exists() and (
                SOUP_DIR / f"{tag}_seed{int(seed)}_top5_soup.pt"
            ).exists():
                print(f"skip existing {representation} seed{seed}", flush=True)
                continue
            print(
                f"=== train {representation} seed{seed} device={device} ===",
                flush=True,
            )
            _train_representation(str(representation), int(seed), device=device)


# ---------------------------------------------------------------------------
# mechanism / diagnostic requirements
# ---------------------------------------------------------------------------


def _module_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    names = (
        "patch_encoder",
        "parent_embedding",
        "pair_projection",
        "relation_encoder",
        "distance_gate",
        "pair_encoder",
        "center_update",
        "global_encoder",
        "topology_encoder",
        "head",
    )
    groups: dict[str, list[nn.Parameter]] = {}
    for name in names:
        module = getattr(model, name, None)
        groups[name] = [] if module is None else list(module.parameters())
    return groups


def _gradient_audit(model: nn.Module, batch: Data, device: torch.device) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    batch = batch.to(device)
    prediction = model(batch).view(-1)
    target = batch.y.view(-1)
    loss = (prediction - target).abs().mean()
    loss.backward()
    groups = _module_groups(model)
    result: dict[str, Any] = {"loss": float(loss.detach()), "groups": {}}
    for name, parameters in groups.items():
        norm = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                norm += float(parameter.grad.detach().norm() ** 2)
        result["groups"][name] = {
            "params": int(sum(p.numel() for p in parameters)),
            "grad_norm": float(norm**0.5),
        }
    result["all_groups_nonzero"] = bool(
        all(row["grad_norm"] > 0.0 for row in result["groups"].values() if row["params"] > 0)
    )
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


def sanity() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    accounting = parameter_accounting()
    checks["parameter_accounting"] = accounting

    null = build_null(0).eval()
    constant = build_constant(0).eval()
    checks["null_has_no_typed_embedding"] = bool(null.typed_embedding is None)
    checks["null_has_no_structural_encoder"] = bool(null.structural_encoder is None)
    checks["null_total_params"] = _n_params(null)
    checks["constant_total_params"] = _n_params(constant)
    checks["constant_is_trainable_16_vector"] = bool(
        constant.local_token_constant is not None
        and tuple(constant.local_token_constant.shape) == (TOKEN_WIDTH,)
        and constant.local_token_constant.requires_grad
    )

    # downstream input width unchanged
    typed = build_reference("A0", 0).eval()
    checks["patch_encoder_input_width_null_equals_A0"] = bool(
        int(null.patch_encoder.layers[0].in_features)
        == int(typed.patch_encoder.layers[0].in_features)
    )
    checks["downstream_backbone_params_null"] = int(
        _n_params(null) - _n_params_typed_like_local(null)
    )

    from tracks.ksvd.experiments.luyin16 import (
        zinc_compact_v4_identity_capacity_control as ic,
    )

    batch = ic._synthetic_batch()
    with torch.no_grad():
        null_pred = null(batch)
        typed.set_local_token_intervention("zero")
        typed_zero_pred = typed(batch)
        typed.clear_local_token_intervention()
        typed.set_local_token_intervention(
            "constant", constant=torch.arange(TOKEN_WIDTH, dtype=torch.float32)
        )
        typed_const_pred = typed(batch)
        typed.clear_local_token_intervention()
        constant.local_token_constant.data.copy_(
            torch.arange(TOKEN_WIDTH, dtype=torch.float32)
        )
        constant_pred = constant(batch)
    checks["zero_intervention_equals_null"] = float(
        (null_pred - typed_zero_pred).abs().max()
    )
    checks["constant_intervention_equals_constant_model"] = float(
        (typed_const_pred - constant_pred).abs().max()
    )
    checks["null_forward_finite"] = bool(torch.isfinite(null_pred).all())

    gradient_audit = _gradient_audit(build_null(0), batch, torch.device("cpu"))
    checks["null_gradient_audit"] = gradient_audit
    checks["official_test_loaded"] = False
    checks["all_pass"] = bool(
        checks["null_has_no_typed_embedding"]
        and checks["null_has_no_structural_encoder"]
        and checks["null_total_params"] == NULL_TOTAL_PARAMS
        and checks["constant_total_params"] == NULL_TOTAL_PARAMS + CONSTANT_EXTRA_PARAMS
        and checks["constant_is_trainable_16_vector"]
        and checks["patch_encoder_input_width_null_equals_A0"]
        and checks["zero_intervention_equals_null"] == 0.0
        and checks["constant_intervention_equals_constant_model"] == 0.0
        and gradient_audit["all_groups_nonzero"]
    )
    _write_json(RESULTS_DIR / "sanity.json", checks)
    return checks


def _n_params_typed_like_local(model: nn.Module) -> int:
    """Parameters of the (absent) local-token generator on a null model."""
    total = 0
    if getattr(model, "typed_embedding", None) is not None:
        total += _n_params(model.typed_embedding)
    if getattr(model, "structural_encoder", None) is not None:
        total += _n_params(model.structural_encoder)
    if getattr(model, "local_token_constant", None) is not None:
        total += int(model.local_token_constant.numel())
    return int(total)


# ---------------------------------------------------------------------------
# decision / report
# ---------------------------------------------------------------------------


def decide() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "references": {name: REFERENCE_SOUP[name] for name in REFERENCE_NAMES},
        "candidates": {},
        "gate": None,
        "official_test_loaded": False,
    }
    null_soup = None
    for rep, tag in (("null", "lt_null"), ("constant", "lt_constant")):
        path = RESULTS_DIR / f"soup_{tag}_seed0.json"
        if not path.exists():
            payload["candidates"][rep] = {"available": False}
            continue
        soup = _read_json(path)
        payload["candidates"][rep] = {
            "available": True,
            "seed0_soup_valid_mae": float(soup["top5_soup_valid_mae"]),
            "seed0_raw_best_valid_mae": float(soup["best_checkpoint_valid_mae"]),
            "best_epoch": int(soup["best_epoch"]),
            "parameters": int(soup["parameters"]),
        }
        if rep == "null":
            null_soup = float(soup["top5_soup_valid_mae"])
    if null_soup is None:
        payload["gate"] = {
            "verdict": "B_NULL_SEED0_NOT_RUN",
            "authorize_constant": False,
        }
    else:
        if null_soup <= B_NULL_STRONG:
            verdict = "STRONG_LOCAL_TOKEN_CHANNEL_NOT_REQUIRED"
            authorize = True
        elif null_soup <= B_NULL_MILD:
            verdict = "CONTRIBUTES_BUT_PARAMETER_DISPROPORTIONATE"
            authorize = True
        elif null_soup <= B_NULL_SUBSTANTIAL:
            verdict = "SUBSTANTIAL_CONTRIBUTION_ANALYZE"
            authorize = False
        else:
            verdict = "STOP_LOCAL_TOKEN_FUNCTIONALLY_IMPORTANT"
            authorize = False
        payload["gate"] = {
            "b_null_seed0_soup": null_soup,
            "verdict": verdict,
            "authorize_constant": authorize,
            "thresholds": {
                "strong": B_NULL_STRONG,
                "mild": B_NULL_MILD,
                "substantial": B_NULL_SUBSTANTIAL,
            },
            "note": "experiment-economy gate only, not a statistical claim",
        }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def report() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "files": {},
        "official_test_loaded": False,
    }
    for name in (
        "parameter_accounting.json",
        "sanity.json",
        "stage_a_frozen_knockout.json",
        "stage_a_gate.json",
        "decision.json",
    ):
        path = RESULTS_DIR / name
        if path.exists():
            payload["files"][name] = _read_json(path)
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "params",
            "sanity",
            "stage_a",
            "stage_a_gate",
            "train",
            "train_queue",
            "soup",
            "decide",
            "report",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--representation", type=str, default="null")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--references", type=str, default="A0,B-Bag,B-Full")
    parser.add_argument("--train-batches", type=int, default=-1)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    global DETERMINISTIC
    DETERMINISTIC = bool(args.deterministic)
    torch.set_num_threads(4)
    _set_deterministic(DETERMINISTIC)
    train_batches = None if int(args.train_batches) < 0 else int(args.train_batches)

    if args.stage == "params":
        print(json.dumps(parameter_accounting(), indent=2, default=str), flush=True)
    if args.stage == "sanity":
        result = sanity()
        print(json.dumps(result, indent=2, default=str), flush=True)
    if args.stage == "stage_a":
        global REFERENCE_NAMES
        REFERENCE_NAMES = tuple(
            name for name in args.references.split(",") if name != ""
        )
        payload = stage_a(
            [int(s) for s in args.seeds.split(",") if s != ""],
            device=args.device,
            train_batches=train_batches,
        )
        print(json.dumps(payload, indent=2, default=str), flush=True)
    if args.stage == "stage_a_gate":
        print(json.dumps(stage_a_gate(), indent=2, default=str), flush=True)
    if args.stage == "train":
        print(
            json.dumps(
                _train_representation(
                    args.representation, int(args.seed), device=args.device
                ),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "train_queue":
        train_queue(
            [r for r in args.representation.split(",") if r != ""],
            [int(s) for s in args.seeds.split(",") if s != ""],
            args.device,
        )
    if args.stage == "soup":
        print(
            json.dumps(
                soup(args.representation, int(args.seed)), indent=2, default=str
            ),
            flush=True,
        )
    if args.stage == "decide":
        print(json.dumps(decide(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
