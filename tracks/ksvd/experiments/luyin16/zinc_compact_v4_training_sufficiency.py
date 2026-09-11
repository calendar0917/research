"""Compact-v4 Training Sufficiency & Protocol Calibration (ZINC, compact-v4-hinge).

This is **not** a new-architecture experiment.  It asks a single
optimization-level question:

    Is the historical canonical 60-epoch training protocol substantially
    under-training compact-v4-hinge, or is the observed +0.00909
    continuation gain an optimisation-regime (batch / weight-decay / fresh
    optimizer) effect rather than a training-horizon effect?

Nothing about the model, representation, loss or target is changed.  The only
variables searched are the training protocol:

* horizon / early-stopping patience  (Stage 1A)
* batch size / weight decay          (Stage 1B, conditional)
* learning rate                      (Stage 2, conditional)

official **test is never loaded** during any search stage.  Only a frozen
protocol lock authorises a single one-shot test evaluation at the very end.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency <stage>

Stages: ``stage0 lock stage1a stage1b stage2 candidate seed1 seed2 seed3
freeze decision drift compute figures test all``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Batch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
    V4_EXPECTED_EPOCH,
    V4_EXPECTED_VALID,
    V4_RUN_IDS,
    _build_v4_model,
    _extract_v4_records,
    load_run_result,
)

# ---------------------------------------------------------------------------
# frozen locations / protocol constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
PROTOCOL_ID = "luyin16-zinc-hierarchical-patch-relation-context-compact-v4-topology-hinge"
CANONICAL_CONFIG_PATH = TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_training_sufficiency"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
CACHE_DIR = RESULTS_DIR / "cache"
FIG_DIR = RESULTS_DIR / "figures"

PROTOCOL_VERSION = "compact_v4_training_sufficiency_v1"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

PRIMARY_SEED = 0
ARCH_PARAM_COUNT = 99613  # canonical selection-phase total trainable params

# historical canonical results (multiseed confirmation, train-only fits)
HISTORICAL = {
    0: {
        "run_id": V4_RUN_IDS[0],
        "valid_best_mae": V4_EXPECTED_VALID[0],
        "best_epoch": V4_EXPECTED_EPOCH[0],
        "test_selection_checkpoint_mae": 0.13390139845572413,
    },
    1: {
        "run_id": V4_RUN_IDS[1],
        "valid_best_mae": V4_EXPECTED_VALID[1],
        "best_epoch": V4_EXPECTED_EPOCH[1],
        "test_selection_checkpoint_mae": 0.13200611347239463,
    },
}
HISTORICAL_TEST = {  # selection-checkpoint test MAE of the historical protocol
    0: 0.13390139845572413,
    1: 0.13200611347239463,
    2: 0.1446138486834243,
    3: 0.13701914092712103,
}
FOUR_SEED_HISTORICAL_VALID = {
    int(seed): float(value) for seed, value in V4_EXPECTED_VALID.items()
}

# continuation anchor (NOT a benchmark candidate; an optimisation-headroom probe)
CONTINUATION_ANCHOR = 0.1609755826992332

# ---------------------------------------------------------------------------
# pre-registered decision thresholds
# ---------------------------------------------------------------------------
THRESHOLD_MEANINGFUL = 0.003      # meaningful upgrade over historical
THRESHOLD_H1_UNDERTRAIN = 0.004   # A0 - A2 gain for clear undertraining
THRESHOLD_H2_ANCHOR = 0.002       # A2 within this of the continuation anchor
THRESHOLD_H3_STALL = 0.002        # A2 - A0 below this == still stalls
THRESHOLD_REPLICATION = 0.003     # per-seed seed0 gain gate
THRESHOLD_SMALL_IMPROVEMENT = 0.003

TRAINING_KEYS = ("batch_size", "weight_decay", "learning_rate", "dropout")

# Stage specs ---------------------------------------------------------------
STAGE1A = [
    ("A0_historical", 60, 12, 128, 1.0e-5, 1.0e-3),
    ("A1_moderate", 120, 24, 128, 1.0e-5, 1.0e-3),
    ("A2_long", 240, 40, 128, 1.0e-5, 1.0e-3),
]
STAGE1B = [
    ("B0_b128_wd1e-5", 240, 40, 128, 1.0e-5, 1.0e-3),
    ("B1_b512_wd1e-5", 240, 40, 512, 1.0e-5, 1.0e-3),
    ("B2_b128_wd0", 240, 40, 128, 0.0, 1.0e-3),
    ("B3_b512_wd0", 240, 40, 512, 0.0, 1.0e-3),
]


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def base_config() -> dict[str, Any]:
    config = yaml.safe_load(CANONICAL_CONFIG_PATH.read_text(encoding="utf-8"))
    config["test_policy"] = "no_test"  # search never loads official test
    config["output"] = {
        "json": str(RESULTS_DIR / "cache" / "legacy_scratch.json"),
        "markdown": str(RESULTS_DIR / "cache" / "legacy_scratch.md"),
    }
    return config


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------


def load_train_valid_records() -> tuple[list[Any], list[Any]]:
    """Cached compact-v4-hinge train+valid GraphRecords (no test)."""
    return _extract_v4_records()


def extract_test_records() -> list[Any]:
    """Extract (and cache) official test GraphRecords.  ONLY called by `test`."""
    test_path = CACHE_DIR / "v4_records_test.pkl.gz"
    if test_path.exists():
        with gzip.open(test_path, "rb") as handle:
            return list(pickle.load(handle))
    test_ds = _load_zinc(ZINC_ROOT, "test")
    topo_test = ztopo.matrices_for_split("test", test_ds, "hinge")[0]
    certificate_cache: dict[bytes, bytes] = {}
    test_records, _ = zpp._extract_split(
        test_ds,
        "test",
        certificate_cache,
        topology_mode="hinge",
        topology_matrix=topo_test,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(test_path, "wb") as handle:
        pickle.dump(list(test_records), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return test_records


def build_encoded(
    train_records: Sequence[Any],
    other_records: Sequence[Any],
    config: Mapping[str, Any],
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    return zpp._phase_data(train_records, other_records, config=config)


# ---------------------------------------------------------------------------
# one protocol run
# ---------------------------------------------------------------------------


def _steps_per_epoch(n_graphs: int, batch_size: int) -> int:
    return int(math.ceil(int(n_graphs) / int(batch_size)))


def train_one(
    name: str,
    seed: int,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    weight_decay: float,
    learning_rate: float,
    encoded: tuple[list[Any], list[Any], dict[str, Any]],
    config: Mapping[str, Any],
    save_state: bool = False,
) -> dict[str, Any]:
    """Train one protocol from scratch (selection phase) with full curves."""
    train_data, valid_data, audit = encoded
    cfg = copy.deepcopy(dict(config))
    cfg["model"].update(
        {
            "epochs": int(epochs),
            "patience": int(patience),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
            "learning_rate": float(learning_rate),
        }
    )
    cfg["seed"] = int(seed)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_threads", 4)))
    started = time.perf_counter()
    phase = zpp._train_phase(
        train_data,
        valid_data,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        config=cfg,
        seed=int(seed),
        select_best=True,
        epochs=int(epochs),
        shell_width=int(audit["shell_width"]),
        context_width=0,
        direct_token_readout=False,
        structural_context_vocabulary_size=int(audit["structural_context_vocabulary_size"]),
        topology_input_width=int(audit["topology"]["input_width"]),
        topology_hidden_dim=int(cfg["model"].get("topology_hidden_dim", 16)),
        topology_out_dim=int(cfg["model"].get("topology_out_dim", 8)),
        topology_shuffle_test=bool(cfg["model"].get("topology_shuffle_test", False)),
    )
    wall_clock = float(time.perf_counter() - started)
    trace = list(phase["trace"])
    losses = list(phase["losses"])
    epochs_run = int(phase["epochs_run"])
    best_epoch = int(phase["selected_epoch"])
    best_valid = float(phase["best_mae"])
    steps_per_epoch = _steps_per_epoch(len(train_data), int(batch_size))
    optimizer_steps = int(epochs_run) * steps_per_epoch

    curve_rows: list[dict[str, Any]] = []
    for index in range(epochs_run):
        epoch = index + 1
        curve_rows.append(
            {
                "protocol": name,
                "seed": int(seed),
                "epoch": epoch,
                "train_loss": float(losses[index]),
                "valid_mae": float(trace[index]["mae"]),
                "lr": float(learning_rate),
                "optimizer_step": epoch * steps_per_epoch,
                "checkpoint_selected": int(epoch == best_epoch),
            }
        )
    _write_csv(
        CURVE_DIR / f"{name}_seed{seed}.csv",
        curve_rows,
        (
            "protocol",
            "seed",
            "epoch",
            "train_loss",
            "valid_mae",
            "lr",
            "optimizer_step",
            "checkpoint_selected",
        ),
    )

    state_path = None
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"{name}_seed{seed}_selection_state.pt"
        torch.save(phase["model"].state_dict(), state_path)

    summary = {
        "protocol": name,
        "seed": int(seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "learning_rate": float(learning_rate),
        "scheduler": "none",
        "best_valid_mae": best_valid,
        "best_epoch": best_epoch,
        "train_loss_at_best": float(losses[best_epoch - 1]),
        "final_train_loss": float(losses[-1]),
        "final_epoch_valid_mae": float(trace[-1]["mae"]),
        "epochs_run": epochs_run,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": optimizer_steps,
        "wall_clock_s": wall_clock,
        "early_stopped": bool(epochs_run < int(epochs)),
        "parameters": int(phase["parameters"]),
        "typed_vocabulary_size_with_oov": int(audit["typed_vocabulary_size_with_oov"]),
        "parent_vocabulary_size_with_oov": int(audit["parent_vocabulary_size_with_oov"]),
        "state_path": None if state_path is None else str(state_path),
        "valid_shuffled_topology_mae": phase.get("diagnostics", {}).get(
            "valid_shuffled_topology_mae"
        ),
    }
    print(
        f"[{name} seed{seed}] best_valid={best_valid:.6f} best_epoch={best_epoch} "
        f"run={epochs_run} steps={optimizer_steps} wall={wall_clock:.1f}s "
        f"early_stop={summary['early_stopped']}",
        flush=True,
    )
    return summary


def _encoded_for(config: Mapping[str, Any]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train_records, valid_records = load_train_valid_records()
    return build_encoded(train_records, valid_records, config)


# ---------------------------------------------------------------------------
# Stage 0 -- historical protocol reconstruction
# ---------------------------------------------------------------------------


def stage0() -> dict[str, Any]:
    config = base_config()
    result = load_run_result(V4_RUN_IDS[PRIMARY_SEED])
    train_trace = result["evaluation"]["valid"]["trace"]
    selected_epoch = int(result["evaluation"]["valid"]["selected_epoch"])
    steps_per_epoch = _steps_per_epoch(10000, int(config["model"]["batch_size"]))
    rows = []
    for entry in train_trace:
        epoch = int(entry["epoch"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": "",  # not stored in the historical result
                "valid_mae": float(entry["mae"]),
                "lr": float(config["model"]["learning_rate"]),
                "optimizer_step": epoch * steps_per_epoch,
                "checkpoint_selected": int(epoch == selected_epoch),
            }
        )
    _write_csv(
        RESULTS_DIR / "historical_learning_curve.csv",
        rows,
        (
            "epoch",
            "train_loss",
            "valid_mae",
            "lr",
            "optimizer_step",
            "checkpoint_selected",
        ),
    )
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": PROTOCOL_ID,
        "config_path": str(CANONICAL_CONFIG_PATH),
        "config_sha256": _sha256_file(CANONICAL_CONFIG_PATH),
        "optimizer": "Adam",
        "learning_rate": float(config["model"]["learning_rate"]),
        "weight_decay": float(config["model"]["weight_decay"]),
        "batch_size": int(config["model"]["batch_size"]),
        "max_epochs": int(config["model"]["epochs"]),
        "patience": int(config["model"]["patience"]),
        "scheduler": "none",
        "gradient_clip_norm": 5.0,
        "checkpoint_selection_metric": "valid_mae (best, lower is better)",
        "loss": "L1 / mean absolute error",
        "seed": PRIMARY_SEED,
        "parameter_count": ARCH_PARAM_COUNT,
        "representation": {
            "typed_tokenizer_version": config.get("representation", {}).get(
                "typed_tokenizer_version", "default_historical"
            ),
            "patch_radius": 2,
            "topology_mode": config["model"]["topology_mode"],
            "topology_input_width": 25,
        },
        "data": {
            "root": "data/ZINC",
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {"train": 10000, "val": 1000, "test": 1000},
            "valid_vocab_scope": "official train only",
        },
        "historical_runs": {
            str(seed): {
                "run_id": info["run_id"],
                "valid_best_mae": info["valid_best_mae"],
                "best_epoch": info["best_epoch"],
                "test_selection_checkpoint_mae": info["test_selection_checkpoint_mae"],
            }
            for seed, info in HISTORICAL.items()
        },
        "historical_seed0_trace": [
            {"epoch": int(entry["epoch"]), "valid_mae": float(entry["mae"])}
            for entry in train_trace
        ],
        "note": (
            "Reconstructed from the committed canonical config and the "
            "20260909-194445-182c7021 run result. The historical result stores "
            "valid MAE per epoch but NOT train MAE; Stage 1A A0 re-runs the "
            "exact protocol to recover the full curve and verify reproduction."
        ),
    }
    _write_json(RESULTS_DIR / "historical_protocol.json", protocol)
    print(f"stage0: wrote historical_protocol.json (seed0 valid={HISTORICAL[0]['valid_best_mae']:.6f})")
    return protocol


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------


def preregistration(write: bool = True) -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "canonical_config": str(CANONICAL_CONFIG_PATH),
        "canonical_config_sha256": _sha256_file(CANONICAL_CONFIG_PATH),
        "question": (
            "Is the historical 60-epoch canonical protocol substantially "
            "under-training compact-v4-hinge?"
        ),
        "search_variables": [
            "training horizon",
            "early-stopping patience",
            "batch size",
            "weight decay",
            "learning rate",
            "(conditional) one fixed scheduler family",
        ],
        "forbidden_variables": [
            "model width",
            "head width",
            "dropout",
            "activation",
            "topology features",
            "loss",
            "tokenizer",
            "optimizer family",
            "patch radius",
            "embedding dimension",
        ],
        "stage1a": [
            {
                "protocol": name,
                "max_epochs": epochs,
                "patience": patience,
                "batch_size": bs,
                "weight_decay": wd,
                "learning_rate": lr,
            }
            for name, epochs, patience, bs, wd, lr in STAGE1A
        ],
        "stage1b_conditional": [
            {
                "protocol": name,
                "max_epochs": epochs,
                "patience": patience,
                "batch_size": bs,
                "weight_decay": wd,
                "learning_rate": lr,
            }
            for name, epochs, patience, bs, wd, lr in STAGE1B
        ],
        "stage2_conditional": {"learning_rates": [3.0e-4, 1.0e-3], "extra_if_unstable": 2.0e-3},
        "thresholds": {
            "meaningful_upgrade_valid": THRESHOLD_MEANINGFUL,
            "clear_undertraining_A0_minus_A2": THRESHOLD_H1_UNDERTRAIN,
            "horizon_reaches_anchor_within": THRESHOLD_H2_ANCHOR,
            "still_stalls_below": THRESHOLD_H3_STALL,
            "seed0_replication_gain": THRESHOLD_REPLICATION,
            "small_improvement_only_below": THRESHOLD_SMALL_IMPROVEMENT,
        },
        "continuation_anchor": CONTINUATION_ANCHOR,
        "historical_seed0_valid": HISTORICAL[0]["valid_best_mae"],
        "historical_seed1_valid": HISTORICAL[1]["valid_best_mae"],
        "test_policy": "official test unavailable until final_training_protocol_lock.json exists",
        "seeds": {"search": [0], "replication": [1]},
    }
    if write:
        _write_json(RESULTS_DIR / "search_preregistration.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage 1A -- horizon ladder
# ---------------------------------------------------------------------------


def _matching_run(
    spec: Mapping[str, Any], seed: int, exclude: str | None = None
) -> dict[str, Any] | None:
    """Return an already-completed run summary with an identical spec/seed."""
    for path in sorted((RESULTS_DIR / "runs").glob("*.json")):
        if exclude is not None and path.name == exclude:
            continue
        row = _read_json(path)
        if int(row.get("seed", -1)) != int(seed):
            continue
        if (
            int(row["epochs"]) == int(spec["epochs"])
            and int(row["patience"]) == int(spec["patience"])
            and int(row["batch_size"]) == int(spec["batch_size"])
            and float(row["weight_decay"]) == float(spec["weight_decay"])
            and float(row["learning_rate"]) == float(spec["learning_rate"])
        ):
            return row
    return None


def _load_or_train(name, seed, spec, encoded, config, save_state=False):
    summary_path = RESULTS_DIR / "runs" / f"{name}_seed{seed}.json"
    if summary_path.exists():
        summary = _read_json(summary_path)
        if (
            int(summary["epochs"]) == int(spec["epochs"])
            and int(summary["patience"]) == int(spec["patience"])
            and int(summary["batch_size"]) == int(spec["batch_size"])
            and float(summary["learning_rate"]) == float(spec["learning_rate"])
            and float(summary["weight_decay"]) == float(spec["weight_decay"])
        ):
            print(f"[{name} seed{seed}] reused cached summary", flush=True)
            return summary
    match = _matching_run(spec, seed, exclude=summary_path.name)
    if match is not None and not save_state:
        print(
            f"[{name} seed{seed}] reusing identical run {match['protocol']}",
            flush=True,
        )
        source_curve = CURVE_DIR / f"{match['protocol']}_seed{seed}.csv"
        target_curve = CURVE_DIR / f"{name}_seed{seed}.csv"
        if source_curve.exists() and not target_curve.exists():
            target_curve.write_bytes(source_curve.read_bytes())
        summary = dict(match)
        summary["protocol"] = name
        summary["reused_from"] = match["protocol"]
        _write_json(summary_path, summary)
        return summary
    summary = train_one(
        name,
        seed,
        epochs=spec["epochs"],
        patience=spec["patience"],
        batch_size=spec["batch_size"],
        weight_decay=spec["weight_decay"],
        learning_rate=spec["learning_rate"],
        encoded=encoded,
        config=config,
        save_state=save_state,
    )
    _write_json(summary_path, summary)
    return summary


def _spec_dict(epochs, patience, bs, wd, lr):
    return {
        "epochs": epochs,
        "patience": patience,
        "batch_size": bs,
        "weight_decay": wd,
        "learning_rate": lr,
    }


def stage1a(seed: int = PRIMARY_SEED) -> dict[str, Any]:
    config = base_config()
    encoded = _encoded_for(config)
    rows = []
    summaries = {}
    for name, epochs, patience, bs, wd, lr in STAGE1A:
        summary = _load_or_train(
            name, seed, _spec_dict(epochs, patience, bs, wd, lr), encoded, config
        )
        summaries[name] = summary
        rows.append(summary)
    fields = (
        "protocol",
        "max_epochs",
        "patience",
        "batch_size",
        "weight_decay",
        "learning_rate",
        "best_valid_mae",
        "best_epoch",
        "train_loss_at_best",
        "final_train_loss",
        "final_epoch_valid_mae",
        "epochs_run",
        "optimizer_steps",
        "wall_clock_s",
        "early_stopped",
        "parameters",
    )
    table = [
        {
            "protocol": row["protocol"],
            "max_epochs": row["epochs"],
            "patience": row["patience"],
            "batch_size": row["batch_size"],
            "weight_decay": row["weight_decay"],
            "learning_rate": row["learning_rate"],
            "best_valid_mae": row["best_valid_mae"],
            "best_epoch": row["best_epoch"],
            "train_loss_at_best": row["train_loss_at_best"],
            "final_train_loss": row["final_train_loss"],
            "final_epoch_valid_mae": row["final_epoch_valid_mae"],
            "epochs_run": row["epochs_run"],
            "optimizer_steps": row["optimizer_steps"],
            "wall_clock_s": row["wall_clock_s"],
            "early_stopped": row["early_stopped"],
            "parameters": row["parameters"],
        }
        for row in rows
    ]
    _write_csv(RESULTS_DIR / "stage1a_horizon_results.csv", table, fields)
    a0 = summaries["A0_historical"]["best_valid_mae"]
    a1 = summaries["A1_moderate"]["best_valid_mae"]
    a2 = summaries["A2_long"]["best_valid_mae"]
    interpretation = {
        "A0_valid": a0,
        "A1_valid": a1,
        "A2_valid": a2,
        "A0_minus_A2": a0 - a2,
        "A0_best_epoch": summaries["A0_historical"]["best_epoch"],
        "A2_best_epoch": summaries["A2_long"]["best_epoch"],
        "historical_seed0_reproduced": abs(a0 - HISTORICAL[0]["valid_best_mae"]) < 1e-9,
        "cases": {
            "H1_clear_undertraining": bool(
                (a0 - a2) >= THRESHOLD_H1_UNDERTRAIN
                and summaries["A2_long"]["best_epoch"] > 60
            ),
            "H2_horizon_reaches_anchor": bool(abs(a2 - CONTINUATION_ANCHOR) <= THRESHOLD_H2_ANCHOR),
            "H3_longer_still_stalls": bool(
                (a0 - a2) < THRESHOLD_H3_STALL
                and (a2 - CONTINUATION_ANCHOR) > THRESHOLD_H2_ANCHOR
            ),
        },
        "continuation_anchor": CONTINUATION_ANCHOR,
    }
    _write_json(RESULTS_DIR / "stage1a_interpretation.json", interpretation)
    print("stage1a interpretation:", json.dumps(interpretation["cases"]))
    return {"summaries": summaries, "interpretation": interpretation}


# ---------------------------------------------------------------------------
# Stage 1B -- batch / weight-decay 2x2 decomposition
# ---------------------------------------------------------------------------


def stage1b(seed: int = PRIMARY_SEED) -> dict[str, Any]:
    config = base_config()
    encoded = _encoded_for(config)
    summaries = {}
    for name, epochs, patience, bs, wd, lr in STAGE1B:
        summaries[name] = _load_or_train(
            name, seed, _spec_dict(epochs, patience, bs, wd, lr), encoded, config
        )
    fields = (
        "protocol",
        "batch_size",
        "weight_decay",
        "learning_rate",
        "best_valid_mae",
        "best_epoch",
        "train_loss_at_best",
        "epochs_run",
        "optimizer_steps",
        "wall_clock_s",
        "early_stopped",
        "parameters",
    )
    table = [
        {
            "protocol": row["protocol"],
            "batch_size": row["batch_size"],
            "weight_decay": row["weight_decay"],
            "learning_rate": row["learning_rate"],
            "best_valid_mae": row["best_valid_mae"],
            "best_epoch": row["best_epoch"],
            "train_loss_at_best": row["train_loss_at_best"],
            "epochs_run": row["epochs_run"],
            "optimizer_steps": row["optimizer_steps"],
            "wall_clock_s": row["wall_clock_s"],
            "early_stopped": row["early_stopped"],
            "parameters": row["parameters"],
        }
        for row in summaries.values()
    ]
    _write_csv(RESULTS_DIR / "stage1b_regime_results.csv", table, fields)
    deltas = {
        "B0_to_B1_batch_only": summaries["B0_b128_wd1e-5"]["best_valid_mae"]
        - summaries["B1_b512_wd1e-5"]["best_valid_mae"],
        "B0_to_B2_wd_only": summaries["B0_b128_wd1e-5"]["best_valid_mae"]
        - summaries["B2_b128_wd0"]["best_valid_mae"],
        "B0_to_B3_both": summaries["B0_b128_wd1e-5"]["best_valid_mae"]
        - summaries["B3_b512_wd0"]["best_valid_mae"],
    }
    payload = {"deltas": deltas, "summaries": summaries}
    _write_json(RESULTS_DIR / "stage1b_interpretation.json", payload)
    print("stage1b deltas:", json.dumps({k: round(v, 6) for k, v in deltas.items()}))
    return payload


# ---------------------------------------------------------------------------
# Stage 2 -- learning-rate calibration on the winning regime
# ---------------------------------------------------------------------------


def _winning_regime() -> dict[str, Any]:
    """Read the Stage 1B table and pick the best (lowest) batch/wd cell."""
    rows = _read_csv(RESULTS_DIR / "stage1b_regime_results.csv")
    if not rows:
        raise RuntimeError("stage1b_regime_results.csv missing; run stage1b first")
    best = min(rows, key=lambda row: float(row["best_valid_mae"]))
    return {
        "batch_size": int(best["batch_size"]),
        "weight_decay": float(best["weight_decay"]),
        "max_epochs": 240,
        "patience": 40,
        "source_protocol": best["protocol"],
    }


def stage2(seed: int = PRIMARY_SEED) -> dict[str, Any]:
    regime = _winning_regime()
    config = base_config()
    encoded = _encoded_for(config)
    learning_rates = [3.0e-4, 1.0e-3]
    summaries = {}
    for lr in learning_rates:
        name = f"S_lr{lr:.0e}".replace("e-0", "e-")
        spec = _spec_dict(regime["max_epochs"], regime["patience"], regime["batch_size"], regime["weight_decay"], lr)
        summaries[name] = _load_or_train(name, seed, spec, encoded, config)
    fields = (
        "protocol",
        "batch_size",
        "weight_decay",
        "learning_rate",
        "best_valid_mae",
        "best_epoch",
        "train_loss_at_best",
        "epochs_run",
        "optimizer_steps",
        "wall_clock_s",
        "early_stopped",
    )
    table = [
        {
            "protocol": row["protocol"],
            "batch_size": row["batch_size"],
            "weight_decay": row["weight_decay"],
            "learning_rate": row["learning_rate"],
            "best_valid_mae": row["best_valid_mae"],
            "best_epoch": row["best_epoch"],
            "train_loss_at_best": row["train_loss_at_best"],
            "epochs_run": row["epochs_run"],
            "optimizer_steps": row["optimizer_steps"],
            "wall_clock_s": row["wall_clock_s"],
            "early_stopped": row["early_stopped"],
        }
        for row in summaries.values()
    ]
    _write_csv(RESULTS_DIR / "stage2_lr_results.csv", table, fields)
    _write_json(RESULTS_DIR / "stage2_interpretation.json", {"regime": regime, "summaries": summaries})
    return {"regime": regime, "summaries": summaries}


# ---------------------------------------------------------------------------
# candidate selection + protocol lock
# ---------------------------------------------------------------------------


def _candidate_spec() -> dict[str, Any]:
    """Choose P*: best completed protocol among horizon/regime/LR stages.

    Selection rule (pre-registered): lowest best-valid MAE, tie-broken toward
    the earlier / simpler stage (longer-single-stage preferred over
    two-stage), and a parameter/compute budget sanity check.
    """
    candidates: list[dict[str, Any]] = []
    for path in sorted((RESULTS_DIR / "runs").glob("*.json")):
        summary = _read_json(path)
        candidates.append(summary)
    if not candidates:
        raise RuntimeError("no completed protocol runs found under results/.../runs/")
    best = min(
        candidates,
        key=lambda row: (
            float(row["best_valid_mae"]),
            int(row["optimizer_steps"]),
        ),
    )
    return best


def candidate() -> dict[str, Any]:
    best = _candidate_spec()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_protocol": best["protocol"],
        "seed": int(best["seed"]),
        "max_epochs": int(best["epochs"]),
        "patience": int(best["patience"]),
        "batch_size": int(best["batch_size"]),
        "weight_decay": float(best["weight_decay"]),
        "learning_rate": float(best["learning_rate"]),
        "scheduler": "none",
        "best_valid_mae": float(best["best_valid_mae"]),
        "best_epoch": int(best["best_epoch"]),
        "optimizer_steps": int(best["optimizer_steps"]),
        "early_stopped": bool(best["early_stopped"]),
        "parameters": int(best["parameters"]),
        "historical_seed0_valid": HISTORICAL[0]["valid_best_mae"],
        "seed0_gain_over_historical": HISTORICAL[0]["valid_best_mae"]
        - float(best["best_valid_mae"]),
        "selection_rule": (
            "lowest best-valid MAE, tie-break toward fewer optimizer steps; "
            "prefer a single-stage from-scratch protocol unless a two-stage "
            "phase switch is shown to add reproducible value beyond longer "
            "canonical training"
        ),
    }
    _write_json(RESULTS_DIR / "candidate_protocol.json", payload)
    # candidate seed0 curve alias
    curve_path = CURVE_DIR / f"{best['protocol']}_seed{best['seed']}.csv"
    if curve_path.exists():
        import shutil

        shutil.copyfile(curve_path, RESULTS_DIR / "candidate_seed0_curve.csv")
    return payload


# ---------------------------------------------------------------------------
# Stage 3 -- seed replication
# ---------------------------------------------------------------------------


def _run_candidate_seed(seed: int, save_state: bool = True) -> dict[str, Any]:
    cand = _read_json(RESULTS_DIR / "candidate_protocol.json")
    config = base_config()
    encoded = _encoded_for(config)
    name = f"Pstar_{cand['selected_protocol']}"
    spec = _spec_dict(
        cand["max_epochs"],
        cand["patience"],
        cand["batch_size"],
        cand["weight_decay"],
        cand["learning_rate"],
    )
    return _load_or_train(name, seed, spec, encoded, config, save_state=save_state)


def seed1() -> dict[str, Any]:
    summary = _run_candidate_seed(1)
    cand = _read_json(RESULTS_DIR / "candidate_protocol.json")
    old = HISTORICAL[1]["valid_best_mae"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_protocol": cand,
        "seed1_result": summary,
        "historical_seed1_valid": old,
        "seed1_gain": old - float(summary["best_valid_mae"]),
        "seed0_gain": cand["seed0_gain_over_historical"],
        "paired_mean_gain": (
            cand["seed0_gain_over_historical"] + (old - float(summary["best_valid_mae"]))
        )
        / 2.0,
        "seed0_gate_pass": cand["seed0_gain_over_historical"] >= THRESHOLD_REPLICATION,
        "seed1_gate_pass": (old - float(summary["best_valid_mae"])) > 0,
    }
    payload["replication_pass"] = bool(
        payload["seed0_gate_pass"]
        and payload["seed1_gate_pass"]
        and payload["paired_mean_gain"] >= THRESHOLD_REPLICATION
    )
    _write_json(RESULTS_DIR / "seed1_replication.json", payload)
    print("seed1 replication:", json.dumps({k: payload[k] for k in ("seed1_gain", "seed0_gain", "paired_mean_gain", "replication_pass")}))
    return payload


def seed_extra(seed: int) -> dict[str, Any]:
    summary = _run_candidate_seed(seed, save_state=True)
    old = FOUR_SEED_HISTORICAL_VALID.get(int(seed))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "result": summary,
        "historical_valid": old,
        "gain": None if old is None else old - float(summary["best_valid_mae"]),
    }
    _write_json(RESULTS_DIR / f"seed{seed}_results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# protocol lock / decision
# ---------------------------------------------------------------------------


def freeze(test_authorized: bool = True) -> dict[str, Any]:
    cand = _read_json(RESULTS_DIR / "candidate_protocol.json")
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": _git_commit(),
        "canonical_config": str(CANONICAL_CONFIG_PATH),
        "canonical_config_sha256": _sha256_file(CANONICAL_CONFIG_PATH),
        "protocol_id": PROTOCOL_ID,
        "exact_config": {
            "optimizer": "Adam",
            "max_epochs": int(cand["max_epochs"]),
            "patience": int(cand["patience"]),
            "batch_size": int(cand["batch_size"]),
            "learning_rate": float(cand["learning_rate"]),
            "weight_decay": float(cand["weight_decay"]),
            "scheduler": str(cand["scheduler"]),
            "scheduler_params": None,
            "gradient_clip_norm": 5.0,
            "loss": "L1 / mean absolute error",
            "dropout": 0.05,
        },
        "seed_policy": {
            "search": [0],
            "replication": [1],
            "test_seeds": sorted(
                int(part[4:])
                for path in STATE_DIR.glob("Pstar_*_selection_state.pt")
                for part in path.stem.split("_")
                if part.startswith("seed") and part[4:].isdigit()
            ),
        },
        "checkpoint_selection_rule": "lowest official-valid MAE; frozen checkpoint -> official test once",
        "tokenizer_fingerprint": zpp.typed_tokenizer_fingerprint(
            zpp.resolve_typed_tokenizer_version(None), 2
        ),
        "topology_implementation_fingerprint": {
            "feature_version": ztopo.FEATURE_VERSION,
            "mode": "hinge",
            "input_width": 25,
        },
        "model_parameter_count": ARCH_PARAM_COUNT,
        "selection_checkpoint_rule": "official train only training; official valid for epoch selection",
        "official_test_authorized": bool(test_authorized),
        "note": "Do NOT modify this protocol after test results are seen.",
    }
    _write_json(RESULTS_DIR / "final_training_protocol_lock.json", lock)
    print("freeze: wrote final_training_protocol_lock.json")
    return lock


def valid_summary() -> dict[str, Any]:
    """Write the paired historical-vs-optimized validation summary (all seeds)."""
    cand = _read_json(RESULTS_DIR / "candidate_protocol.json")
    rows = []
    for seed in sorted(FOUR_SEED_HISTORICAL_VALID):
        summary_path = (
            RESULTS_DIR
            / "runs"
            / f"Pstar_{cand['selected_protocol']}_seed{int(seed)}.json"
        )
        if summary_path.exists():
            optimized = float(_read_json(summary_path)["best_valid_mae"])
        elif int(seed) == PRIMARY_SEED:
            optimized = float(cand["best_valid_mae"])
        else:
            continue
        historical = float(FOUR_SEED_HISTORICAL_VALID[int(seed)])
        rows.append(
            {
                "seed": int(seed),
                "historical_valid": historical,
                "optimized_valid": optimized,
                "gain": historical - optimized,
            }
        )
    _write_csv(
        RESULTS_DIR / "valid_summary.csv",
        rows,
        ("seed", "historical_valid", "optimized_valid", "gain"),
    )
    gains = [row["gain"] for row in rows]
    payload = {
        "seeds": [row["seed"] for row in rows],
        "historical_valid_mean": float(np.mean([row["historical_valid"] for row in rows])),
        "historical_valid_std": float(np.std([row["historical_valid"] for row in rows])),
        "optimized_valid_mean": float(np.mean([row["optimized_valid"] for row in rows])),
        "optimized_valid_std": float(np.std([row["optimized_valid"] for row in rows])),
        "paired_mean_gain": float(np.mean(gains)),
        "all_positive": bool(all(gain > 0 for gain in gains)),
    }
    _write_json(RESULTS_DIR / "valid_summary.json", payload)
    return payload


def decision() -> dict[str, Any]:
    cand = _read_json(RESULTS_DIR / "candidate_protocol.json")
    seed1_path = RESULTS_DIR / "seed1_replication.json"
    seed1 = _read_json(seed1_path) if seed1_path.exists() else None
    stage1a_path = RESULTS_DIR / "stage1a_interpretation.json"
    stage1a = _read_json(stage1a_path) if stage1a_path.exists() else {}
    stage1b_path = RESULTS_DIR / "stage1b_interpretation.json"
    stage1b = _read_json(stage1b_path) if stage1b_path.exists() else {}

    seed0_gain = float(cand["seed0_gain_over_historical"])
    seed1_gain = None if seed1 is None else float(seed1["seed1_gain"])
    anchor = CONTINUATION_ANCHOR

    cases = {
        "H1_clear_undertraining": bool(stage1a.get("cases", {}).get("H1_clear_undertraining")),
        "H2_horizon_reaches_anchor": bool(stage1a.get("cases", {}).get("H2_horizon_reaches_anchor")),
        "H3_longer_still_stalls": bool(stage1a.get("cases", {}).get("H3_longer_still_stalls")),
    }

    if seed0_gain < THRESHOLD_SMALL_IMPROVEMENT and (seed1_gain is None or seed1_gain < THRESHOLD_SMALL_IMPROVEMENT):
        verdict = "SMALL_TRAINING_IMPROVEMENT_ONLY"
    elif seed1 is None:
        verdict = "INCONCLUSIVE_PENDING_SEED1"
    elif seed0_gain >= THRESHOLD_REPLICATION and (seed1_gain is not None and seed1_gain > 0) and bool(seed1.get("replication_pass")):
        if cases["H1_clear_undertraining"]:
            verdict = "HISTORICAL_UNDERTRAINING_CONFIRMED"
        elif cases["H3_longer_still_stalls"]:
            verdict = "OPTIMIZATION_REGIME_SWITCH_GO"
        else:
            verdict = "OPTIMIZED_PROTOCOL_GO"
    elif seed0_gain >= THRESHOLD_REPLICATION and (seed1_gain is not None and seed1_gain <= 0):
        verdict = "INCONCLUSIVE_SEED_SPECIFIC"
    else:
        verdict = "NO_GO"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate": cand,
        "stage1a_cases": cases,
        "stage1b_deltas": stage1b.get("deltas"),
        "seed0_gain": seed0_gain,
        "seed1_gain": seed1_gain,
        "paired_mean_gain": None if seed1 is None else float(seed1["paired_mean_gain"]),
        "valid_summary": valid_summary(),
        "continuation_anchor": anchor,
        "candidate_vs_anchor": float(cand["best_valid_mae"]) - anchor,
        "verdict": verdict,
        "official_test_authorized": verdict
        in ("HISTORICAL_UNDERTRAINING_CONFIRMED", "OPTIMIZATION_REGIME_SWITCH_GO", "OPTIMIZED_PROTOCOL_GO"),
        "decision_cases_reference": {
            "A": "HISTORICAL_UNDERTRAINING_CONFIRMED",
            "B": "OPTIMIZATION_REGIME_SWITCH_GO",
            "C": "SMALL_TRAINING_IMPROVEMENT_ONLY",
            "D": "INCONCLUSIVE_SEED_SPECIFIC / INCONCLUSIVE_PENDING_SEED1",
            "E": "NO_GO",
        },
    }
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    print("decision verdict:", verdict)
    return payload


# ---------------------------------------------------------------------------
# representation-drift descriptive audit
# ---------------------------------------------------------------------------


def drift(probe_n: int = 1000) -> dict[str, Any]:
    config = base_config()
    encoded = _encoded_for(config)
    train_data, valid_data, audit = encoded
    probe = list(valid_data[: int(probe_n)])

    def _encode_state(state_path: Path) -> np.ndarray:
        model = _build_v4_model(config, audit)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(probe), 128):
                batch = Batch.from_data_list(probe[start : start + 128])
                chunks.append(model.encode(batch).cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float64)

    from tracks.ksvd.experiments.luyin16.zinc_canonical_late_readout_adaptation import (
        canonical_run as _canonical_run,
    )

    hist = _canonical_run(PRIMARY_SEED)
    old_state = hist["state_path"]
    cand_state = STATE_DIR / f"Pstar_{_read_json(RESULTS_DIR / 'candidate_protocol.json')['selected_protocol']}_seed{PRIMARY_SEED}_selection_state.pt"
    if not cand_state.exists():
        raise RuntimeError(f"candidate state missing: {cand_state}")
    r_old = _encode_state(old_state)
    r_new = _encode_state(cand_state)
    diff = r_new - r_old
    norm_old = float(np.mean(np.linalg.norm(r_old, axis=1)))
    norm_diff = float(np.mean(np.linalg.norm(diff, axis=1)))
    cosine = float(np.mean(np.sum(r_old * r_new, axis=1) / (np.linalg.norm(r_old, axis=1) * np.linalg.norm(r_new, axis=1) + 1e-12)))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "probe_n": int(len(probe)),
        "historical_state": str(old_state),
        "candidate_state": str(cand_state),
        "mean_R_norm_historical": norm_old,
        "mean_R_drift_norm": norm_diff,
        "normalized_L2_drift": norm_diff / max(norm_old, 1e-12),
        "mean_cosine_similarity": cosine,
        "note": "descriptive only; not part of any GO gate",
    }
    _write_json(RESULTS_DIR / "representation_drift.json", payload)
    return payload


# ---------------------------------------------------------------------------
# compute accounting
# ---------------------------------------------------------------------------


def compute_accounting() -> dict[str, Any]:
    rows = []
    total_steps = 0
    total_wall = 0.0
    for path in sorted((RESULTS_DIR / "runs").glob("*.json")):
        row = _read_json(path)
        total_steps += int(row["optimizer_steps"])
        total_wall += float(row["wall_clock_s"])
        rows.append(
            {
                "protocol": row["protocol"],
                "seed": row["seed"],
                "epochs_run": row["epochs_run"],
                "optimizer_steps": row["optimizer_steps"],
                "wall_clock_s": row["wall_clock_s"],
                "parameters": row["parameters"],
            }
        )
    fields = ("protocol", "seed", "epochs_run", "optimizer_steps", "wall_clock_s", "parameters")
    _write_csv(RESULTS_DIR / "compute_accounting.csv", rows, fields)
    payload = {
        "runs": len(rows),
        "total_optimizer_steps": total_steps,
        "total_wall_clock_s": total_wall,
        "parameter_count": ARCH_PARAM_COUNT,
    }
    _write_json(RESULTS_DIR / "compute_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# official test (authorized only after lock)
# ---------------------------------------------------------------------------


def test() -> dict[str, Any]:
    lock_path = RESULTS_DIR / "final_training_protocol_lock.json"
    if not lock_path.exists():
        raise RuntimeError("refusing official test: final_training_protocol_lock.json missing")
    lock = _read_json(lock_path)
    if not lock.get("official_test_authorized", False):
        raise RuntimeError("refusing official test: protocol lock does not authorise test")
    decision_path = RESULTS_DIR / "final_decision.json"
    if not decision_path.exists():
        raise RuntimeError("refusing official test: final_decision.json missing")
    verdict = _read_json(decision_path)["verdict"]
    if verdict not in (
        "HISTORICAL_UNDERTRAINING_CONFIRMED",
        "OPTIMIZATION_REGIME_SWITCH_GO",
        "OPTIMIZED_PROTOCOL_GO",
    ):
        raise RuntimeError(f"refusing official test: verdict={verdict}")

    config = base_config()
    train_records, valid_records = load_train_valid_records()
    test_records = extract_test_records()
    # encode test with train-only vocab/standardisers (identical to selection)
    _train_data, test_data, audit = build_encoded(train_records, test_records, config)
    cand = _read_json(RESULTS_DIR / "candidate_protocol.json")
    cand_run = _read_json(
        RESULTS_DIR / "runs" / f"Pstar_{cand['selected_protocol']}_seed{PRIMARY_SEED}.json"
    )
    if int(audit["typed_vocabulary_size_with_oov"]) != int(
        cand_run["typed_vocabulary_size_with_oov"]
    ):
        raise RuntimeError(
            "train-only vocabulary mismatch between selection and test encoding: "
            f"{int(audit['typed_vocabulary_size_with_oov'])} vs "
            f"{int(cand_run['typed_vocabulary_size_with_oov'])}"
        )
    test_targets = np.asarray(
        [float(graph.y.view(-1)[0]) for graph in test_data], dtype=np.float64
    )

    test_seeds = lock["seed_policy"]["test_seeds"]
    rows = []
    for seed in test_seeds:
        state_path = STATE_DIR / f"Pstar_{cand['selected_protocol']}_seed{int(seed)}_selection_state.pt"
        if not state_path.exists():
            continue
        model = _build_v4_model(config, audit)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(test_data), 128):
                batch = Batch.from_data_list(test_data[start : start + 128])
                preds.append(model(batch).view(-1).cpu().numpy())
        predictions = np.concatenate(preds).astype(np.float64)
        test_mae = float(mean_absolute_error(test_targets, predictions))
        valid_mae = None
        hist_valid = HISTORICAL.get(int(seed), {}).get("valid_best_mae")
        # candidate valid from runs
        for run in (RESULTS_DIR / "runs").glob(f"Pstar_*_seed{seed}.json"):
            valid_mae = float(_read_json(run)["best_valid_mae"])
        rows.append(
            {
                "seed": int(seed),
                "valid": valid_mae,
                "test": test_mae,
                "historical_valid": hist_valid,
                "historical_test": HISTORICAL_TEST.get(int(seed)),
            }
        )
    _write_csv(RESULTS_DIR / "test_results.csv", rows, ("seed", "valid", "test", "historical_valid", "historical_test"))
    valid_values = [row["valid"] for row in rows if row["valid"] is not None]
    test_values = [row["test"] for row in rows if row["test"] is not None]
    historical_test_values = [row["historical_test"] for row in rows if row["historical_test"] is not None]
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "seeds": [row["seed"] for row in rows],
        "optimized_valid_mean": float(np.mean(valid_values)) if valid_values else None,
        "optimized_valid_std": float(np.std(valid_values)) if len(valid_values) > 1 else 0.0,
        "optimized_test_mean": float(np.mean(test_values)) if test_values else None,
        "optimized_test_std": float(np.std(test_values)) if len(test_values) > 1 else 0.0,
        "historical_test_mean": float(np.mean(historical_test_values)) if historical_test_values else None,
        "test_gain_vs_historical": (
            float(np.mean(historical_test_values) - np.mean(test_values))
            if test_values and historical_test_values
            else None
        ),
        "official_test_accessed": True,
    }
    _write_json(RESULTS_DIR / "test_summary.json", summary)
    print("test summary:", json.dumps(summary))
    return summary


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def figures() -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []

    # Figure 1: historical / 120 / 240 valid MAE vs epoch (seed 0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, label in (
        ("A0_historical", "A0 historical (60/12)"),
        ("A1_moderate", "A1 moderate (120/24)"),
        ("A2_long", "A2 long (240/40)"),
    ):
        rows = _read_csv(CURVE_DIR / f"{name}_seed0.csv")
        if not rows:
            continue
        ax.plot([int(r["epoch"]) for r in rows], [float(r["valid_mae"]) for r in rows], label=label)
    ax.axhline(CONTINUATION_ANCHOR, color="k", ls="--", lw=1, label="continuation anchor 0.160976")
    ax.set_xlabel("epoch")
    ax.set_ylabel("official valid MAE")
    ax.set_title("Stage 1A horizon ladder (seed 0)")
    ax.set_ylim(0.15, 0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path1 = FIG_DIR / "fig1_horizon_ladder.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    outputs.append(str(path1))

    # Figure 2: candidate train / valid vs epoch
    cand = _read_json(RESULTS_DIR / "candidate_protocol.json")
    rows = _read_csv(CURVE_DIR / f"Pstar_{cand['selected_protocol']}_seed0.csv")
    if rows:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        epochs = [int(r["epoch"]) for r in rows]
        ax1.plot(epochs, [float(r["train_loss"]) for r in rows], color="C0", label="train L1")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train L1", color="C0")
        ax2 = ax1.twinx()
        ax2.plot(epochs, [float(r["valid_mae"]) for r in rows], color="C1", label="valid MAE")
        ax2.axhline(CONTINUATION_ANCHOR, color="k", ls="--", lw=1)
        ax2.set_ylabel("valid MAE", color="C1")
        ax1.set_title(f"Candidate {cand['selected_protocol']} (seed 0)")
        fig.tight_layout()
        path2 = FIG_DIR / "fig2_candidate_curve.png"
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        outputs.append(str(path2))

    # Figure 3: seed0 best valid MAE across protocols
    names, values = [], []
    for path in sorted((RESULTS_DIR / "runs").glob("*_seed0.json")):
        row = _read_json(path)
        names.append(row["protocol"])
        values.append(float(row["best_valid_mae"]))
    if names:
        order = np.argsort(values)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh([names[i] for i in order], [values[i] for i in order], color="C2")
        ax.axvline(HISTORICAL[0]["valid_best_mae"], color="k", ls="--", lw=1, label="historical 0.170066")
        ax.axvline(CONTINUATION_ANCHOR, color="r", ls=":", lw=1, label="continuation 0.160976")
        ax.set_xlabel("best official valid MAE (seed 0)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path3 = FIG_DIR / "fig3_protocol_comparison.png"
        fig.savefig(path3, dpi=150)
        plt.close(fig)
        outputs.append(str(path3))

    # Figure 4: multi-seed / test paired comparison (only if test exists)
    test_csv = RESULTS_DIR / "test_results.csv"
    rows = _read_csv(test_csv)
    if rows:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        seeds = [int(r["seed"]) for r in rows]
        x = np.arange(len(seeds))
        width = 0.35
        ax.bar(x - width / 2, [float(r["valid"]) for r in rows], width, label="optimized valid")
        ax.bar(x + width / 2, [float(r["test"]) for r in rows], width, label="optimized test")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_ylabel("MAE")
        ax.legend(fontsize=8)
        ax.set_title("Optimized vs historical (valid/test)")
        fig.tight_layout()
        path4 = FIG_DIR / "fig4_multiseed_test.png"
        fig.savefig(path4, dpi=150)
        plt.close(fig)
        outputs.append(str(path4))
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "stage0",
            "lock",
            "stage1a",
            "stage1b",
            "stage2",
            "candidate",
            "seed1",
            "seed2",
            "seed3",
            "freeze",
            "decision",
            "drift",
            "compute",
            "figures",
            "test",
            "all",
        ],
    )
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--allow-test", action="store_true", help="authorise test in freeze lock")
    args = parser.parse_args(argv)

    stage = args.stage
    if stage == "stage0":
        stage0()
    elif stage == "lock":
        preregistration()
    elif stage == "stage1a":
        stage1a(seed=args.seed)
    elif stage == "stage1b":
        stage1b(seed=args.seed)
    elif stage == "stage2":
        stage2(seed=args.seed)
    elif stage == "candidate":
        candidate()
    elif stage == "seed1":
        seed1()
    elif stage in ("seed2", "seed3"):
        seed_extra(int(stage[-1]))
    elif stage == "freeze":
        freeze(test_authorized=bool(args.allow_test))
    elif stage == "decision":
        decision()
    elif stage == "drift":
        drift()
    elif stage == "compute":
        compute_accounting()
    elif stage == "figures":
        figures()
    elif stage == "test":
        test()
    elif stage == "all":
        preregistration()
        stage0()
        stage1a(seed=PRIMARY_SEED)
        stage1b(seed=PRIMARY_SEED)
        stage2(seed=PRIMARY_SEED)
        candidate()
        seed1()
        compute_accounting()
        decision()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
