"""Optimized-Baseline Critical Revalidation Audit (ZINC).

This stage is **not** a new-architecture search.  It re-checks exactly two
historical architecture conclusions against the *new, stronger* optimized
compact-v4-hinge baseline (the 240-epoch single-stage protocol frozen by
``zinc_compact_v4_training_sufficiency``), because the historical conclusions
were obtained under the under-trained 60-epoch protocol.

* **Question A** -- does the global topology hinge still provide independent
  architecture value after fair optimization?
  ``optimized compact-v2  vs  optimized compact-v4-hinge``.
* **Question B** -- does the corrected typed tokenizer remain worse after
  fair optimization?
  ``optimized historical-token v4  vs  optimized corrected-token v4``.

Discipline (see the task prompt and the track notes):

* exactly one new full-training run per branch by default
  (``optimized-v2 seed0``, ``optimized-corrected-v4 seed0``);
* seed1 is purchased only when the seed0 decision gate creates genuine
  uncertainty; seed2/seed3 are forbidden here;
* the frozen optimized protocol is *inherited* unchanged
  (Adam lr 1e-3, wd 1e-5, batch 128, max_epochs 240, patience 40,
  no scheduler, L1, best official-valid checkpoint, single-stage);
* only the explicitly tested variable moves (Part A: topology channel;
  Part B: tokenizer version);
* **official test is never loaded** in this stage;
* runs execute serially (one training process at a time), deterministic CPU
  settings (``torch.set_num_threads(4)``, fixed global torch RNG).

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_optimized_baseline_critical_revalidation <stage>

Stages: ``protocol_lock inventory part_a seed0_a_decision part_b
seed0_b_decision seed1_a seed1_b final compute all``.
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
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    TYPED_TOKENIZER_V1_HISTORICAL,
    TYPED_TOKENIZER_V2_CORRECTED,
    resolve_typed_tokenizer_version,
    typed_tokenizer_fingerprint,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
    V4_EXPECTED_EPOCH,
    V4_EXPECTED_VALID,
    V4_RUN_IDS,
)

# ---------------------------------------------------------------------------
# frozen locations / protocol constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/optimized_baseline_critical_revalidation"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
CACHE_DIR = RESULTS_DIR / "cache"
SCRATCH_DIR = RESULTS_DIR / "scratch"

V2_CONFIG_PATH = TRACK_ROOT / "configs/luyin16/zinc_optimized_v2_historical.yaml"
CORR_CONFIG_PATH = TRACK_ROOT / "configs/luyin16/zinc_optimized_corrected_v4.yaml"
HISTORICAL_V2_CONFIG_PATH = (
    TRACK_ROOT
    / "configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v2_budget_reallocation.yaml"
)
HISTORICAL_V4_CONFIG_PATH = (
    TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
)
HISTORICAL_CORR_CONFIG_PATH = (
    TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge_typed_v2.yaml"
)
CANONICAL_SUFFICIENCY_DIR = TRACK_ROOT / "results/compact_v4_training_sufficiency"

PROTOCOL_VERSION = "optimized_baseline_critical_revalidation_v1"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

PATCH_RADIUS = 2

# --- frozen optimized training protocol (inherited from compact-v4 sufficiency)
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
}

TORCH_THREADS = 4
ZINC_SPLIT_SIZES = {"train": 10000, "val": 1000, "test": 1000}

# historical 60-epoch results (for the record only; not the comparison baseline)
HISTORICAL_60EPOCH = {
    "v2_historical": {
        "valid": 0.18415821571176638,
        "params": 98549,
        "best_epoch": 56,
        "record": "20260907-193612-46c1a12f",
    },
    "v4_historical": {
        "valid": 0.17006561887910357,
        "params": 99613,
        "best_epoch": 53,
        "record": V4_RUN_IDS[0],
    },
    "v4_corrected": {
        "valid": 0.17815828679403056,
        "params": 107201,
        "best_epoch": 44,
        "record": "20260910-111954-64845bd9",
    },
}

# --- reused optimized compact-v4-hinge reference (do NOT retrain) ----------
OPTIMIZED_V4_SEED0_VALID = 0.14642022556537995
OPTIMIZED_V4_SEED1_VALID = 0.1493322635096847
OPTIMIZED_V4_SEED0_EPOCH = 169
OPTIMIZED_V4_SEED1_EPOCH = 104
OPTIMIZED_V4_PARAMS = 99613
SUFFICIENCY_COMMIT = "586542e1e03b8c4517a20b1ac24520f4e7a3ffed"

# ---------------------------------------------------------------------------
# pre-registered decision thresholds (Part A / Part B)
# ---------------------------------------------------------------------------

A_STRONG = 0.006
A_THREATENED = -0.002
A_GO_MEAN = 0.003
A_DOWNGRADE_BAND = 0.002

B_CLEAR_NOGO = 0.004  # corrected worse by at least this
B_REOPEN_BETTER = 0.003  # corrected better by at least this
B_REOPEN_CLOSE = 0.002  # |delta| <= this -> reopen
B_WEAK_NEG_LOW = 0.002
B_WEAK_NEG_HIGH = 0.004

CACHE_SCHEMA_VERSION = "optimized_revalidation_records_v1"

# variant tags -> (topology_mode, tokenizer_version)
VARIANTS = {
    "v2_hist": {
        "label": "optimized-compact-v2-historical-token",
        "topology_mode": "none",
        "tokenizer_version": TYPED_TOKENIZER_V1_HISTORICAL,
    },
    "v4_corr": {
        "label": "optimized-compact-v4-hinge-corrected-token",
        "topology_mode": "hinge",
        "tokenizer_version": TYPED_TOKENIZER_V2_CORRECTED,
    },
}


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


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
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


def _environment_fingerprint() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "configured_torch_threads": TORCH_THREADS,
        "device": "cpu",
    }


def _configure_determinism() -> None:
    """Serial, deterministic CPU policy (see reproducibility_cpu_determinism.md)."""
    torch.set_num_threads(int(TORCH_THREADS))
    torch.use_deterministic_algorithms(False)  # matches canonical legacy runs


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def v2_config() -> dict[str, Any]:
    config = load_config(V2_CONFIG_PATH)
    config["test_policy"] = "no_test"
    config["output"] = {
        "json": str(SCRATCH_DIR / "optimized_v2_historical.json"),
        "markdown": str(SCRATCH_DIR / "optimized_v2_historical.md"),
    }
    return config


def corrected_v4_config() -> dict[str, Any]:
    config = load_config(CORR_CONFIG_PATH)
    config["test_policy"] = "no_test"
    config["output"] = {
        "json": str(SCRATCH_DIR / "optimized_corrected_v4.json"),
        "markdown": str(SCRATCH_DIR / "optimized_corrected_v4.md"),
    }
    return config


def config_for_variant(tag: str) -> dict[str, Any]:
    if tag == "v2_hist":
        return v2_config()
    if tag == "v4_corr":
        return corrected_v4_config()
    raise KeyError(tag)


# ---------------------------------------------------------------------------
# record extraction / caching (tokenizer-version pinned)
# ---------------------------------------------------------------------------


def _cache_paths(tag: str) -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / f"{tag}_records_train.pkl.gz",
        CACHE_DIR / f"{tag}_records_valid.pkl.gz",
        CACHE_DIR / f"{tag}_metadata.json",
    )


def extract_records(tag: str, force: bool = False) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Extract (once) official train+valid GraphRecords for a variant.

    The tokenizer version is pinned per ``VARIANTS[tag]``; a cache written with
    a different tokenizer version is refused.
    """
    spec = VARIANTS[tag]
    topology_mode = str(spec["topology_mode"])
    tokenizer_version = resolve_typed_tokenizer_version(spec["tokenizer_version"])
    train_path, valid_path, meta_path = _cache_paths(tag)
    if not force and train_path.exists() and valid_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            raise RuntimeError(f"{tag}: stale cache schema; regenerate")
        if meta.get("typed_tokenizer_version") != tokenizer_version:
            raise RuntimeError(
                f"{tag}: cache tokenizer mismatch "
                f"{meta.get('typed_tokenizer_version')} != {tokenizer_version}"
            )
        if str(meta.get("topology_mode")) != topology_mode:
            raise RuntimeError(f"{tag}: cache topology-mode mismatch")
        if meta.get("split_sizes") != ZINC_SPLIT_SIZES:
            raise RuntimeError(f"{tag}: cache split-size mismatch")
        with gzip.open(train_path, "rb") as handle:
            train_records = list(pickle.load(handle))
        with gzip.open(valid_path, "rb") as handle:
            valid_records = list(pickle.load(handle))
        return train_records, valid_records, meta

    started = time.perf_counter()
    train_ds = _load_zinc(ZINC_ROOT, "train")
    valid_ds = _load_zinc(ZINC_ROOT, "val")
    if topology_mode == "none":
        topo_train = None
        topo_valid = None
    else:
        topo_train = ztopo.matrices_for_split("train", train_ds, topology_mode)[0]
        topo_valid = ztopo.matrices_for_split("valid", valid_ds, topology_mode)[0]
    certificate_cache: dict[bytes, bytes] = {}
    train_records, train_meta = zpp._extract_split(
        train_ds,
        "train",
        certificate_cache,
        topology_mode=topology_mode,
        topology_matrix=topo_train,
        tokenizer_version=tokenizer_version,
    )
    valid_records, valid_meta = zpp._extract_split(
        valid_ds,
        "valid",
        certificate_cache,
        topology_mode=topology_mode,
        topology_matrix=topo_valid,
        tokenizer_version=tokenizer_version,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path, records in ((train_path, train_records), (valid_path, valid_records)):
        with gzip.open(path, "wb") as handle:
            pickle.dump(list(records), handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "tag": tag,
        "label": spec["label"],
        "topology_mode": topology_mode,
        "typed_tokenizer_version": tokenizer_version,
        "typed_tokenizer_fingerprint": typed_tokenizer_fingerprint(
            tokenizer_version, PATCH_RADIUS
        ),
        "patch_radius": PATCH_RADIUS,
        "split_sizes": ZINC_SPLIT_SIZES,
        "mean_centres_train": train_meta["mean_centres"],
        "mean_pairs_train": train_meta["mean_pairs"],
        "seconds": float(time.perf_counter() - started),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(meta_path, meta)
    print(
        f"[cache:{tag}] tokenizer={tokenizer_version} topo={topology_mode} "
        f"train={len(train_records)} valid={len(valid_records)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_records, valid_records, meta


def build_encoded(tag: str) -> tuple[list[Any], list[Any], dict[str, Any], dict[str, Any]]:
    config = config_for_variant(tag)
    train_records, valid_records, meta = extract_records(tag)
    encoded = zpp._phase_data(train_records, valid_records, config=config)
    return encoded, config, meta


# ---------------------------------------------------------------------------
# one optimized training run
# ---------------------------------------------------------------------------


def _steps_per_epoch(n_graphs: int, batch_size: int) -> int:
    return int(math.ceil(int(n_graphs) / int(batch_size)))


def train_variant(
    tag: str,
    seed: int,
    *,
    encoded: tuple[list[Any], list[Any], dict[str, Any]],
    config: Mapping[str, Any],
    save_state: bool = True,
) -> dict[str, Any]:
    """Train one variant from scratch with the frozen optimized protocol."""
    train_data, valid_data, audit = encoded
    cfg = copy.deepcopy(dict(config))
    cfg["model"].update(
        {
            "epochs": int(OPTIMIZED_PROTOCOL["max_epochs"]),
            "patience": int(OPTIMIZED_PROTOCOL["patience"]),
            "batch_size": int(OPTIMIZED_PROTOCOL["batch_size"]),
            "weight_decay": float(OPTIMIZED_PROTOCOL["weight_decay"]),
            "learning_rate": float(OPTIMIZED_PROTOCOL["learning_rate"]),
        }
    )
    cfg["seed"] = int(seed)
    _configure_determinism()
    started = time.perf_counter()
    phase = zpp._train_phase(
        train_data,
        valid_data,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        config=cfg,
        seed=int(seed),
        select_best=True,
        epochs=int(OPTIMIZED_PROTOCOL["max_epochs"]),
        shell_width=int(audit["shell_width"]),
        context_width=0,
        direct_token_readout=False,
        structural_context_vocabulary_size=int(audit["structural_context_vocabulary_size"]),
        topology_input_width=int(audit["topology"]["input_width"]),
        topology_hidden_dim=int(cfg["model"].get("topology_hidden_dim", 16)),
        topology_out_dim=int(cfg["model"].get("topology_out_dim", 8)),
        topology_shuffle_test=False,  # shuffle diagnostic not needed here (extra compute)
    )
    wall_clock = float(time.perf_counter() - started)
    trace = list(phase["trace"])
    losses = list(phase["losses"])
    epochs_run = int(phase["epochs_run"])
    best_epoch = int(phase["selected_epoch"])
    best_valid = float(phase["best_mae"])
    spec = VARIANTS[tag]
    steps_per_epoch = _steps_per_epoch(len(train_data), int(OPTIMIZED_PROTOCOL["batch_size"]))
    optimizer_steps = int(epochs_run) * steps_per_epoch

    curve_rows: list[dict[str, Any]] = []
    for index in range(epochs_run):
        epoch = index + 1
        curve_rows.append(
            {
                "epoch": epoch,
                "train_mae": float(losses[index]),
                "valid_mae": float(trace[index]["mae"]),
                "optimizer_step": epoch * steps_per_epoch,
                "checkpoint_selected": int(epoch == best_epoch),
            }
        )
    curve_path = CURVE_DIR / f"{tag}_seed{seed}_curve.csv"
    _write_csv(
        curve_path,
        curve_rows,
        ("epoch", "train_mae", "valid_mae", "optimizer_step", "checkpoint_selected"),
    )

    state_path = None
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        torch.save(phase["model"].state_dict(), state_path)

    horizon_boundary_warning = bool(best_epoch >= int(OPTIMIZED_PROTOCOL["max_epochs"]) - 1)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": tag,
        "label": spec["label"],
        "seed": int(seed),
        "topology_mode": spec["topology_mode"],
        "typed_tokenizer_version": spec["tokenizer_version"],
        "typed_tokenizer_fingerprint": typed_tokenizer_fingerprint(
            spec["tokenizer_version"], PATCH_RADIUS
        ),
        "protocol": dict(OPTIMIZED_PROTOCOL),
        "best_valid_mae": best_valid,
        "best_epoch": best_epoch,
        "train_loss_at_best": float(losses[best_epoch - 1]),
        "final_train_loss": float(losses[-1]),
        "final_epoch_valid_mae": float(trace[-1]["mae"]),
        "epochs_run": epochs_run,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": optimizer_steps,
        "wall_clock_s": wall_clock,
        "early_stopped": bool(epochs_run < int(OPTIMIZED_PROTOCOL["max_epochs"])),
        "horizon_boundary_warning": horizon_boundary_warning,
        "parameters": int(phase["parameters"]),
        "typed_vocabulary_size_with_oov": int(audit["typed_vocabulary_size_with_oov"]),
        "parent_vocabulary_size_with_oov": int(audit["parent_vocabulary_size_with_oov"]),
        "state_path": None if state_path is None else str(state_path),
        "curve_path": str(curve_path),
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(),
        "official_test_loaded": False,
    }
    run_path = RESULTS_DIR / "runs" / f"{tag}_seed{seed}.json"
    _write_json(run_path, summary)
    print(
        f"[{tag} seed{seed}] best_valid={best_valid:.6f} best_epoch={best_epoch} "
        f"run={epochs_run} steps={optimizer_steps} params={summary['parameters']} "
        f"wall={wall_clock:.1f}s horizon_warn={horizon_boundary_warning}",
        flush=True,
    )
    return summary


def _load_or_train(tag: str, seed: int) -> dict[str, Any]:
    run_path = RESULTS_DIR / "runs" / f"{tag}_seed{seed}.json"
    if run_path.exists():
        summary = _read_json(run_path)
        if (
            int(summary["protocol"]["max_epochs"]) == int(OPTIMIZED_PROTOCOL["max_epochs"])
            and int(summary["protocol"]["patience"]) == int(OPTIMIZED_PROTOCOL["patience"])
            and float(summary["protocol"]["learning_rate"])
            == float(OPTIMIZED_PROTOCOL["learning_rate"])
        ):
            print(f"[{tag} seed{seed}] reused cached run summary", flush=True)
            return summary
    encoded, config, _meta = build_encoded(tag)
    return train_variant(tag, seed, encoded=encoded, config=config)


# ---------------------------------------------------------------------------
# Stage 0 -- protocol integrity / lock + checkpoint inventory
# ---------------------------------------------------------------------------


def protocol_lock() -> dict[str, Any]:
    v2cfg = v2_config()
    corrcfg = corrected_v4_config()
    hist_v2 = load_config(HISTORICAL_V2_CONFIG_PATH)
    hist_v4 = load_config(HISTORICAL_V4_CONFIG_PATH)
    hist_corr = load_config(HISTORICAL_CORR_CONFIG_PATH)
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "inherited_optimized_protocol": dict(OPTIMIZED_PROTOCOL),
        "inherited_source": {
            "note": "optimized compact-v4-hinge training sufficiency protocol",
            "results_dir": str(CANONICAL_SUFFICIENCY_DIR.relative_to(REPO_ROOT)),
            "selected_protocol": "A2_long",
            "commit": SUFFICIENCY_COMMIT,
            "config_sha256": _sha256_file(
                CANONICAL_SUFFICIENCY_DIR / "final_training_protocol_lock.json"
            ),
        },
        "compare_against": (
            "optimized compact-v4-hinge (historical tokenizer), NOT the 60-epoch "
            "historical baseline"
        ),
        "official_test_policy": {
            "loaded": False,
            "note": "Stage A/B screening forbids official test; only train/valid used.",
        },
        "seed_policy": {
            "default_new_runs": 1,
            "replication_triggered_only": True,
            "forbidden_seeds": [2, 3],
        },
        "part_a": {
            "question": "does the global topology hinge survive fair optimization?",
            "new_variant": "optimized-compact-v2-historical-token seed0",
            "config": str(V2_CONFIG_PATH.relative_to(REPO_ROOT)),
            "config_sha256": _sha256_file(V2_CONFIG_PATH),
            "historical_base_config": str(HISTORICAL_V2_CONFIG_PATH.relative_to(REPO_ROOT)),
            "historical_base_config_sha256": _sha256_file(HISTORICAL_V2_CONFIG_PATH),
            "architecture_delta_vs_historical_base": "training protocol + explicit tokenizer pin only",
            "pinned_tokenizer_version": TYPED_TOKENIZER_V1_HISTORICAL,
            "tokenizer_fingerprint": typed_tokenizer_fingerprint(
                TYPED_TOKENIZER_V1_HISTORICAL, PATCH_RADIUS
            ),
            "reference": "optimized compact-v4-hinge seed0 (reused, not retrained)",
        },
        "part_b": {
            "question": "does the corrected typed tokenizer remain worse?",
            "new_variant": "optimized-corrected-token compact-v4-hinge seed0",
            "config": str(CORR_CONFIG_PATH.relative_to(REPO_ROOT)),
            "config_sha256": _sha256_file(CORR_CONFIG_PATH),
            "historical_base_config": str(HISTORICAL_CORR_CONFIG_PATH.relative_to(REPO_ROOT)),
            "historical_base_config_sha256": _sha256_file(HISTORICAL_CORR_CONFIG_PATH),
            "architecture_delta_vs_historical_base": "training protocol only (tokenizer already pinned)",
            "pinned_tokenizer_version": TYPED_TOKENIZER_V2_CORRECTED,
            "tokenizer_fingerprint": typed_tokenizer_fingerprint(
                TYPED_TOKENIZER_V2_CORRECTED, PATCH_RADIUS
            ),
            "reference": "optimized historical-token compact-v4-hinge seed0 (same as Part A reference)",
        },
        "thresholds": {
            "A_strong": A_STRONG,
            "A_threatened": A_THREATENED,
            "A_go_mean": A_GO_MEAN,
            "A_downgrade_band": A_DOWNGRADE_BAND,
            "B_clear_nogo": B_CLEAR_NOGO,
            "B_reopen_better": B_REOPEN_BETTER,
            "B_reopen_close": B_REOPEN_CLOSE,
            "B_weak_neg_low": B_WEAK_NEG_LOW,
            "B_weak_neg_high": B_WEAK_NEG_HIGH,
        },
        "not_rerun": [
            "radius-3",
            "local ring conditioning",
            "v5 quantile",
            "v6 factorization",
            "endpoint association",
            "centre covariance",
            "triadic binding",
            "FM",
            "head-family experiments",
            "277k large model",
        ],
        "determinism": {
            "serial_execution": True,
            "torch_threads": TORCH_THREADS,
            "fixed_global_rng": True,
            "environment": _environment_fingerprint(),
        },
        "sanitized_configs": {
            "part_a_model": v2cfg["model"],
            "part_a_representation": v2cfg["representation"],
            "part_b_model": corrcfg["model"],
            "part_b_representation": corrcfg["representation"],
            "historical_v2_model": hist_v2["model"],
            "historical_v2_representation": hist_v2["representation"],
            "historical_v4_model": hist_v4["model"],
            "historical_v4_representation": hist_v4["representation"],
            "historical_corr_model": hist_corr["model"],
            "historical_corr_representation": hist_corr["representation"],
        },
    }
    _write_json(RESULTS_DIR / "protocol_lock.json", lock)
    print("protocol_lock: wrote protocol_lock.json")
    return lock


def checkpoint_inventory() -> dict[str, Any]:
    """Inventory the reused optimized-v4 reference checkpoints (verified)."""
    cand = _read_json(CANONICAL_SUFFICIENCY_DIR / "candidate_protocol.json")
    lock = _read_json(CANONICAL_SUFFICIENCY_DIR / "final_training_protocol_lock.json")
    entries: dict[str, Any] = {}
    for seed, expected_valid, expected_epoch in (
        (0, OPTIMIZED_V4_SEED0_VALID, OPTIMIZED_V4_SEED0_EPOCH),
        (1, OPTIMIZED_V4_SEED1_VALID, OPTIMIZED_V4_SEED1_EPOCH),
    ):
        state_path = (
            CANONICAL_SUFFICIENCY_DIR
            / "states"
            / f"Pstar_{cand['selected_protocol']}_seed{seed}_selection_state.pt"
        )
        run_path = (
            CANONICAL_SUFFICIENCY_DIR
            / "runs"
            / f"Pstar_{cand['selected_protocol']}_seed{seed}.json"
        )
        run = _read_json(run_path)
        ok = (
            state_path.exists()
            and abs(float(run["best_valid_mae"]) - float(expected_valid)) < 1e-9
            and int(run["best_epoch"]) == int(expected_epoch)
            and int(run["parameters"]) == OPTIMIZED_V4_PARAMS
        )
        entries[f"optimized_v4_seed{seed}"] = {
            "role": "optimized historical-token compact-v4-hinge reference",
            "state_path": str(state_path.relative_to(REPO_ROOT)),
            "state_exists": bool(state_path.exists()),
            "run_record": str(run_path.relative_to(REPO_ROOT)),
            "best_valid_mae": float(run["best_valid_mae"]),
            "best_epoch": int(run["best_epoch"]),
            "parameters": int(run["parameters"]),
            "typed_tokenizer_version": TYPED_TOKENIZER_V1_HISTORICAL,
            "typed_tokenizer_fingerprint": typed_tokenizer_fingerprint(
                TYPED_TOKENIZER_V1_HISTORICAL, PATCH_RADIUS
            ),
            "commit": lock.get("code_commit"),
            "expected_valid": float(expected_valid),
            "verified": bool(ok),
        }
    # historical (60-epoch) records kept only as provenance
    inv = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "optimized_v4_reference": entries,
        "optimized_v4_reference_config_sha256": lock.get("canonical_config_sha256"),
        "historical_60epoch_provenance": {
            "v2_historical": HISTORICAL_60EPOCH["v2_historical"],
            "v4_historical": {
                **HISTORICAL_60EPOCH["v4_historical"],
                "run_id": V4_RUN_IDS[0],
                "expected_valid": V4_EXPECTED_VALID[0],
                "expected_epoch": V4_EXPECTED_EPOCH[0],
            },
            "v4_corrected": HISTORICAL_60EPOCH["v4_corrected"],
        },
        "new_checkpoints": {},
        "note": "optimized-v4 references are reused, never retrained in this stage",
    }
    # attach any new checkpoints already produced
    for tag in VARIANTS:
        for seed in (0, 1):
            run_path = RESULTS_DIR / "runs" / f"{tag}_seed{seed}.json"
            if run_path.exists():
                run = _read_json(run_path)
                inv["new_checkpoints"][f"{tag}_seed{seed}"] = {
                    "state_path": run.get("state_path"),
                    "best_valid_mae": run["best_valid_mae"],
                    "best_epoch": run["best_epoch"],
                    "parameters": run["parameters"],
                    "typed_tokenizer_version": run["typed_tokenizer_version"],
                    "typed_tokenizer_fingerprint": run["typed_tokenizer_fingerprint"],
                    "topology_mode": run["topology_mode"],
                    "git_commit": run["git_commit"],
                }
    _write_json(RESULTS_DIR / "checkpoint_inventory.json", inv)
    print(
        "checkpoint_inventory: optimized-v4 seed0 verified="
        f"{entries['optimized_v4_seed0']['verified']}"
    )
    return inv


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------


def part_a_v2_seed0() -> dict[str, Any]:
    run = _load_or_train("v2_hist", 0)
    ref = _read_json(
        CANONICAL_SUFFICIENCY_DIR / "runs/Pstar_A2_long_seed0.json"
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question_a": "does the global topology hinge survive fair optimization?",
        "v2": run,
        "v4_reference": {
            "variant": "optimized-compact-v4-hinge-historical-token seed0",
            "best_valid_mae": float(ref["best_valid_mae"]),
            "best_epoch": int(ref["best_epoch"]),
            "parameters": int(ref["parameters"]),
            "state_path": str(
                (
                    CANONICAL_SUFFICIENCY_DIR
                    / "states"
                    / "Pstar_A2_long_seed0_selection_state.pt"
                ).relative_to(REPO_ROOT)
            ),
            "run_record": str(
                (CANONICAL_SUFFICIENCY_DIR / "runs/Pstar_A2_long_seed0.json").relative_to(
                    REPO_ROOT
                )
            ),
        },
        "delta_topo_v2_minus_v4": float(run["best_valid_mae"])
        - float(ref["best_valid_mae"]),
        "definition": "delta_topo = MAE_valid(v2) - MAE_valid(v4); positive => v4 topology better",
        "historical_60epoch_reference": {
            "v2": HISTORICAL_60EPOCH["v2_historical"],
            "v4": HISTORICAL_60EPOCH["v4_historical"],
            "historical_delta_topo": HISTORICAL_60EPOCH["v2_historical"]["valid"]
            - HISTORICAL_60EPOCH["v4_historical"]["valid"],
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "part_a_v2_seed0.json", payload)
    print(
        f"part_a seed0: v2={run['best_valid_mae']:.6f} "
        f"v4_ref={float(ref['best_valid_mae']):.6f} "
        f"delta_topo={payload['delta_topo_v2_minus_v4']:+.6f}"
    )
    return payload


def part_a_decision() -> dict[str, Any]:
    payload = _read_json(RESULTS_DIR / "part_a_v2_seed0.json")
    delta = float(payload["delta_topo_v2_minus_v4"])
    if delta >= A_STRONG:
        verdict = "TOPO_STRONG_SURVIVES_SEED0"
        need_seed1 = False
    elif delta <= A_THREATENED:
        verdict = "TOPO_THREATENED"
        need_seed1 = True
    else:
        verdict = "TOPO_BORDERLINE"
        need_seed1 = True
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "part": "A",
        "metric": "delta_topo = MAE_valid(v2) - MAE_valid(v4)",
        "observed_delta_topo": delta,
        "thresholds": {"strong>=": A_STRONG, "threatened<=": A_THREATENED},
        "verdict": verdict,
        "need_seed1": bool(need_seed1),
        "interpretation": {
            "TOPO_STRONG_SURVIVES_SEED0": (
                "topology advantage remains far above noise-scale under fair "
                "optimization; stop at seed0, keep topology as core architecture"
            ),
            "TOPO_BORDERLINE": (
                "topology claim is in the uncertain band; buy optimized-v2 seed1"
            ),
            "TOPO_THREATENED": (
                "optimized-v2 is clearly better; buy seed1 because this threatens "
                "the topology main line"
            ),
        }[verdict],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "part_a_decision.json", decision)
    print(f"part_a_decision: {verdict} (need_seed1={need_seed1})")
    return decision


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------


def part_b_corrected_seed0() -> dict[str, Any]:
    run = _load_or_train("v4_corr", 0)
    ref = _read_json(CANONICAL_SUFFICIENCY_DIR / "runs/Pstar_A2_long_seed0.json")
    hist_valid = float(ref["best_valid_mae"])
    delta_correct = hist_valid - float(run["best_valid_mae"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question_b": "does the corrected typed tokenizer remain worse after fair optimization?",
        "corrected": run,
        "historical_token_reference": {
            "variant": "optimized-compact-v4-hinge-historical-token seed0",
            "best_valid_mae": hist_valid,
            "best_epoch": int(ref["best_epoch"]),
            "parameters": int(ref["parameters"]),
            "run_record": str(
                (CANONICAL_SUFFICIENCY_DIR / "runs/Pstar_A2_long_seed0.json").relative_to(
                    REPO_ROOT
                )
            ),
        },
        "delta_correct_hist_minus_corrected": float(delta_correct),
        "corrected_minus_historical": float(run["best_valid_mae"]) - hist_valid,
        "definition": (
            "delta_correct = MAE_valid(historical) - MAE_valid(corrected); "
            "positive => corrected better; corrected_minus_historical positive => historical better"
        ),
        "historical_60epoch_reference": {
            "v4_historical": HISTORICAL_60EPOCH["v4_historical"],
            "v4_corrected": HISTORICAL_60EPOCH["v4_corrected"],
            "historical_degradation": HISTORICAL_60EPOCH["v4_corrected"]["valid"]
            - HISTORICAL_60EPOCH["v4_historical"]["valid"],
        },
        "params_historical": int(ref["parameters"]),
        "params_corrected": int(run["parameters"]),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "part_b_corrected_seed0.json", payload)
    print(
        f"part_b seed0: corrected={run['best_valid_mae']:.6f} "
        f"historical_ref={hist_valid:.6f} delta_correct={delta_correct:+.6f}"
    )
    return payload


def part_b_decision() -> dict[str, Any]:
    payload = _read_json(RESULTS_DIR / "part_b_corrected_seed0.json")
    degradation = float(payload["corrected_minus_historical"])
    delta_correct = float(payload["delta_correct_hist_minus_corrected"])
    if degradation >= B_CLEAR_NOGO:
        verdict = "CORRECTED_CLEAR_NO_GO"
        need_seed1 = False
    elif delta_correct >= B_REOPEN_BETTER or abs(delta_correct) <= B_REOPEN_CLOSE:
        verdict = "CORRECTED_REOPEN"
        need_seed1 = True
    elif B_WEAK_NEG_LOW < degradation < B_WEAK_NEG_HIGH:
        verdict = "CORRECTED_WEAK_NEGATIVE_TRADEOFF"
        need_seed1 = False
    else:
        verdict = "CORRECTED_BORDERLINE_INDETERMINATE"
        need_seed1 = False
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "part": "B",
        "metric": "delta_correct = MAE_valid(historical) - MAE_valid(corrected)",
        "observed_delta_correct": delta_correct,
        "observed_corrected_minus_historical": degradation,
        "thresholds": {
            "clear_nogo_degradation>=": B_CLEAR_NOGO,
            "reopen_delta_correct>=": B_REOPEN_BETTER,
            "reopen_abs_delta<=": B_REOPEN_CLOSE,
            "weak_neg_band": [B_WEAK_NEG_LOW, B_WEAK_NEG_HIGH],
        },
        "verdict": verdict,
        "need_seed1": bool(need_seed1),
        "interpretation": {
            "CORRECTED_CLEAR_NO_GO": (
                "longer training does not rescue the corrected exact-token "
                "representation; stop at seed0"
            ),
            "CORRECTED_REOPEN": (
                "corrected is clearly better or essentially tied; buy seed1 because "
                "this changes the tokenizer methodology choice"
            ),
            "CORRECTED_WEAK_NEGATIVE_TRADEOFF": (
                "performance/correctness tradeoff remains; stop at seed0"
            ),
            "CORRECTED_BORDERLINE_INDETERMINATE": (
                "between the pre-registered bands; default stop at seed0"
            ),
        }[verdict],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "part_b_decision.json", decision)
    print(f"part_b_decision: {verdict} (need_seed1={need_seed1})")
    return decision


# ---------------------------------------------------------------------------
# conditional seed1 replication
# ---------------------------------------------------------------------------


def seed1_a() -> dict[str, Any]:
    decision = _read_json(RESULTS_DIR / "part_a_decision.json")
    if not decision.get("need_seed1"):
        raise RuntimeError(
            f"seed1_a not authorised: part A verdict={decision['verdict']} does not require replication"
        )
    run = _load_or_train("v2_hist", 1)
    ref = _read_json(CANONICAL_SUFFICIENCY_DIR / "runs/Pstar_A2_long_seed1.json")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": 1,
        "v2": run,
        "v4_reference": {
            "best_valid_mae": float(ref["best_valid_mae"]),
            "best_epoch": int(ref["best_epoch"]),
            "parameters": int(ref["parameters"]),
        },
        "delta_topo_v2_minus_v4": float(run["best_valid_mae"]) - float(ref["best_valid_mae"]),
    }
    _write_json(RESULTS_DIR / "part_a_v2_seed1.json", payload)
    return payload


def seed1_b() -> dict[str, Any]:
    decision = _read_json(RESULTS_DIR / "part_b_decision.json")
    if not decision.get("need_seed1"):
        raise RuntimeError(
            f"seed1_b not authorised: part B verdict={decision['verdict']} does not require replication"
        )
    run = _load_or_train("v4_corr", 1)
    ref = _read_json(CANONICAL_SUFFICIENCY_DIR / "runs/Pstar_A2_long_seed1.json")
    hist_valid = float(ref["best_valid_mae"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": 1,
        "corrected": run,
        "historical_token_reference": {
            "best_valid_mae": hist_valid,
            "best_epoch": int(ref["best_epoch"]),
            "parameters": int(ref["parameters"]),
        },
        "delta_correct_hist_minus_corrected": hist_valid - float(run["best_valid_mae"]),
    }
    _write_json(RESULTS_DIR / "part_b_corrected_seed1.json", payload)
    return payload


# ---------------------------------------------------------------------------
# final decision / summaries
# ---------------------------------------------------------------------------


def _two_seed_summary(part: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in (0, 1):
        if part == "A":
            p0 = RESULTS_DIR / "part_a_v2_seed0.json"
            p1 = RESULTS_DIR / "part_a_v2_seed1.json"
            path = p0 if seed == 0 else p1
            if not path.exists():
                continue
            payload = _read_json(path)
            v2 = float(payload["v2"]["best_valid_mae"])
            v4 = float(payload["v4_reference"]["best_valid_mae"])
            rows.append({"seed": seed, "v2_valid": v2, "v4_valid": v4, "delta_topo": v2 - v4})
        else:
            p0 = RESULTS_DIR / "part_b_corrected_seed0.json"
            p1 = RESULTS_DIR / "part_b_corrected_seed1.json"
            path = p0 if seed == 0 else p1
            if not path.exists():
                continue
            payload = _read_json(path)
            corr = float(payload["corrected"]["best_valid_mae"])
            hist = float(payload["historical_token_reference"]["best_valid_mae"])
            rows.append(
                {
                    "seed": seed,
                    "historical_valid": hist,
                    "corrected_valid": corr,
                    "delta_correct": hist - corr,
                }
            )
    if part == "A":
        fields = ("seed", "v2_valid", "v4_valid", "delta_topo")
        out = RESULTS_DIR / "part_a_two_seed_summary.csv"
    else:
        fields = ("seed", "historical_valid", "corrected_valid", "delta_correct")
        out = RESULTS_DIR / "part_b_two_seed_summary.csv"
    _write_csv(out, rows, fields)
    metric = "delta_topo" if part == "A" else "delta_correct"
    values = [row[metric] for row in rows]
    return {
        "rows": rows,
        "n_seeds": len(rows),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if len(values) > 1 else 0.0,
        "all_positive": bool(all(v > 0 for v in values)) if values else False,
        "csv": str(out.relative_to(REPO_ROOT)),
    }


def final_decision() -> dict[str, Any]:
    a_decision = _read_json(RESULTS_DIR / "part_a_decision.json")
    b_decision = _read_json(RESULTS_DIR / "part_b_decision.json")
    a0 = _read_json(RESULTS_DIR / "part_a_v2_seed0.json")
    b0 = _read_json(RESULTS_DIR / "part_b_corrected_seed0.json")
    delta_topo0 = float(a0["delta_topo_v2_minus_v4"])
    delta_corr0 = float(b0["delta_correct_hist_minus_corrected"])

    a_two = None
    if (RESULTS_DIR / "part_a_v2_seed1.json").exists():
        a_two = _two_seed_summary("A")
    b_two = None
    if (RESULTS_DIR / "part_b_corrected_seed1.json").exists():
        b_two = _two_seed_summary("B")

    # --- Part A final verdict
    if a_two is None:
        if a_decision["verdict"] == "TOPO_STRONG_SURVIVES_SEED0":
            a_final = "OPTIMIZED_TOPOLOGY_STRONGLY_SURVIVES_SEED0"
        else:
            a_final = "OPTIMIZED_TOPOLOGY_PENDING_SEED1"
    else:
        mean = float(a_two["mean"])
        if a_two["all_positive"] and mean >= A_GO_MEAN:
            a_final = "OPTIMIZED_TOPOLOGY_GO"
        elif abs(mean) < A_DOWNGRADE_BAND:
            a_final = "HISTORICAL_TOPOLOGY_PERFORMANCE_CLAIM_DOWNGRADED"
        elif mean <= 0:
            a_final = "TOPOLOGY_PERFORMANCE_GO_DOES_NOT_SURVIVE_OPTIMIZED_BASELINE"
        else:
            a_final = "OPTIMIZED_TOPOLOGY_MIXED"

    # --- Part B final verdict
    if b_two is None:
        if b_decision["verdict"] == "CORRECTED_CLEAR_NO_GO":
            b_final = "CORRECTED_CLEAR_NO_GO_SEED0"
        elif b_decision["verdict"] == "CORRECTED_REOPEN":
            b_final = "CORRECTED_REOPEN_PENDING_SEED1"
        elif b_decision["verdict"] == "CORRECTED_WEAK_NEGATIVE_TRADEOFF":
            b_final = "PERFORMANCE_CORRECTNESS_TRADEOFF_REMAINS"
        else:
            b_final = "CORRECTED_BORDERLINE_STOP_SEED0"
    else:
        mean = float(b_two["mean"])
        if mean >= 0:
            b_final = "CORRECTED_GE_HISTORICAL_PREFER_CORRECTED"
        elif mean < 0:
            b_final = "CORRECTED_STILL_WORSE_KEEP_HISTORICAL_TOKEN"
        else:
            b_final = "CORRECTED_MIXED"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "new_full_training_runs": int(
            sum(
                1
                for tag in VARIANTS
                for seed in (0, 1)
                if (RESULTS_DIR / "runs" / f"{tag}_seed{seed}.json").exists()
            )
        ),
        "part_a": {
            "delta_topo_seed0": delta_topo0,
            "seed0_verdict": a_decision["verdict"],
            "two_seed_summary": a_two,
            "final_verdict": a_final,
        },
        "part_b": {
            "delta_correct_seed0": delta_corr0,
            "seed0_verdict": b_decision["verdict"],
            "two_seed_summary": b_two,
            "final_verdict": b_final,
        },
        "correctness_fact_preserved": (
            "the historical tokenizer is NOT a complete typed canonical invariant "
            "key; if the performance model keeps using it, the paper must call it a "
            "historical aliased rooted-topology token, never an exact typed patch token"
        ),
        "not_rerun": [
            "radius-3",
            "local ring conditioning",
            "v5 quantile",
            "v6 factorization",
            "endpoint association",
            "centre covariance",
            "triadic binding",
            "FM",
            "head-family experiments",
            "277k large model",
        ],
    }
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    print(
        f"final_decision: A={a_final} B={b_final} "
        f"new_runs={payload['new_full_training_runs']}"
    )
    return payload


# ---------------------------------------------------------------------------
# compute accounting
# ---------------------------------------------------------------------------


def compute_accounting() -> dict[str, Any]:
    rows = []
    total_steps = 0
    total_wall = 0.0
    for tag in VARIANTS:
        for seed in (0, 1):
            path = RESULTS_DIR / "runs" / f"{tag}_seed{seed}.json"
            if not path.exists():
                continue
            run = _read_json(path)
            total_steps += int(run["optimizer_steps"])
            total_wall += float(run["wall_clock_s"])
            rows.append(
                {
                    "variant": tag,
                    "seed": int(seed),
                    "topology_mode": run["topology_mode"],
                    "typed_tokenizer_version": run["typed_tokenizer_version"],
                    "epochs_run": int(run["epochs_run"]),
                    "optimizer_steps": int(run["optimizer_steps"]),
                    "wall_clock_s": round(float(run["wall_clock_s"]), 3),
                    "parameters": int(run["parameters"]),
                    "torch_threads": TORCH_THREADS,
                    "peak_memory_mb": "",
                }
            )
    fields = (
        "variant",
        "seed",
        "topology_mode",
        "typed_tokenizer_version",
        "epochs_run",
        "optimizer_steps",
        "wall_clock_s",
        "parameters",
        "torch_threads",
        "peak_memory_mb",
    )
    _write_csv(RESULTS_DIR / "compute_accounting.csv", rows, fields)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "new_runs": len(rows),
        "total_optimizer_steps": total_steps,
        "total_wall_clock_s": total_wall,
        "torch_threads": TORCH_THREADS,
        "environment": _environment_fingerprint(),
        "note": "peak memory not re-profiled (no extra runs); leave blank rather than re-run",
    }
    _write_json(RESULTS_DIR / "compute_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "protocol_lock",
            "inventory",
            "part_a",
            "seed0_a_decision",
            "part_b",
            "seed0_b_decision",
            "seed1_a",
            "seed1_b",
            "final",
            "compute",
            "all",
        ],
    )
    args = parser.parse_args(argv)
    stage = args.stage
    if stage == "protocol_lock":
        protocol_lock()
    elif stage == "inventory":
        checkpoint_inventory()
    elif stage == "part_a":
        part_a_v2_seed0()
    elif stage == "seed0_a_decision":
        part_a_decision()
    elif stage == "part_b":
        part_b_corrected_seed0()
    elif stage == "seed0_b_decision":
        part_b_decision()
    elif stage == "seed1_a":
        seed1_a()
    elif stage == "seed1_b":
        seed1_b()
    elif stage == "final":
        final_decision()
    elif stage == "compute":
        compute_accounting()
    elif stage == "all":
        protocol_lock()
        checkpoint_inventory()
        part_a_v2_seed0()
        part_a_decision()
        part_b_corrected_seed0()
        part_b_decision()
        compute_accounting()
        final_decision()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
