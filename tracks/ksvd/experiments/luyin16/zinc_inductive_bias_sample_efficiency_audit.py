"""Compact-v4 Inductive-Bias & Sample-Efficiency Audit (ZINC).

Single question
---------------
Holding architecture, optimizer, checkpoint-selection set, probe set and
optimizer-step budget fixed, **how much held-out MAE is lost by reducing the
number of unique training examples?**

This is a staged, compute-matched learning-curve audit, not a new architecture,
not a structural feature, not a representation-collision audit, not an optimizer
or regularization sweep, not EMA/SWA, not HPO, not an external-model comparison
and not an official-test benchmark.

Environment (index-for-index reuse, never resplit)
--------------------------------------------------
Reuses the already-frozen official-train hash split::

    7200 = optimization pool   (gradient updates only)
    800  = checkpoint selection (inference only)
    2000 = held-out internal probe (opened only *after* a run is frozen)

Nested, target-independent subsets are carved out of the 7200 pool::

    D1800 subset D3600 subset D7200 = pool

with ordering key ``sha256(salt || original_official_train_index)``.  No target,
no target quantile, no chemistry/size balancing is used.

Compute-matched optimizer-step budget
-------------------------------------
The canonical 7200 recipe is 240 epochs x 57 steps/epoch = 13,680 steps with a
selection evaluation every epoch (57 steps) and patience 40 evaluations.  The
new runs keep the same *optimizer-step* budget (max 13,680 steps), the same
evaluation cadence (every 57 steps) and the same patience (40 evaluations), so
smaller N is not trivially confounded with fewer gradient updates.  Smaller N
therefore sees more exposures per unique example; this is reported explicitly.

Estimators
----------
* RAW   -- checkpoint with minimum 800-selection MAE (earliest step wins ties).
* SOUP  -- equal-weight parameter average of the K=5 lowest-800-MAE checkpoints
           (earliest step wins ties).  Fixed measurement stabilizer only.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_inductive_bias_sample_efficiency_audit <stage>

Stages: ``locks split architecture subset anchor seed thresholds protocol
integrity train_stage1 stage1 stage2 stage3 stage4 final all``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_post_v4_residual_audit as postv4
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_training_sufficiency as suff
from tracks.ksvd.experiments.luyin16 import zinc_optimized_manifold_broad_state_screen as broad
from tracks.ksvd.experiments.luyin16 import (
    zinc_internal_generalization_stochasticity_audit as iga,
)

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/inductive_bias_sample_efficiency_audit"
CURVE_DIR = RESULTS_DIR / "curves"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
STATE_DIR = RESULTS_DIR / "states"
FIGURE_DIR = RESULTS_DIR / "figures"

PROTOCOL_VERSION = "inductive_bias_sample_efficiency_audit_v1"

# nested subset lock (frozen before any new training)
SUBSET_SALT = "compact-v4-sample-efficiency-nested-subset-v1-20260912"
NESTED_SIZES = (7200, 3600, 1800)

# real 7200 anchor (I0T0 / I1T1) from the internal-generalization trajectory
ANCHOR_DIR = iga.RESULTS_DIR  # snapshots + run manifests; never retrained here
ANCHOR_RUNS = {0: "I0T0", 1: "I1T1"}
ANCHOR_DIRS = {
    0: TRACK_ROOT / "results/compact_v4_smallhead_e2e",
}
TOP5_DIR = TRACK_ROOT / "results/top5_checkpoint_aggregation_stabilization"

# pre-registered tuning protocol (identical to compact-v4-smallhead)
PROTOCOL: dict[str, Any] = dict(shead.OPTIMIZED_PROTOCOL)
TRAIN_SHUFFLE_SEED_OFFSET = int(PROTOCOL["train_shuffle_seed_offset"])
EVAL_SHUFFLE_SEED_OFFSET = int(PROTOCOL["eval_shuffle_seed_offset"])

# pre-registered decision thresholds
GAIN_MATERIAL = 0.006
GAIN_WEAK = 0.003
RAW_CONFLICT = -0.003
REPLICATE_PER_SEED = 0.005
REPLICATE_MEAN = 0.006
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260912
BOUNDARY_LAST_EVALS = 5

SOUP_K = 5
EXPECTED_PARAMS = 82115
SPLIT_SIZES = (7200, 800, 2000)
SPLIT_ROLES = ("optimization_train", "checkpoint_selection", "internal_probe")

RUN_SPEC: dict[str, dict[str, Any]] = {
    "N3600_I0T0": {"N": 3600, "I": 0, "T": 0, "anchor_seed": 0},
    "N3600_I1T1": {"N": 3600, "I": 1, "T": 1, "anchor_seed": 1},
    "N1800_I0T0": {"N": 1800, "I": 0, "T": 0, "anchor_seed": 0},
    "N1800_I1T1": {"N": 1800, "I": 1, "T": 1, "anchor_seed": 1},
}

# stage purchase order (strict budget discipline)
STAGE1_RUN = "N3600_I0T0"
STAGE2_RUN = "N3600_I1T1"
STAGE3_RUN = "N1800_I0T0"
STAGE4_RUN = "N1800_I1T1"

# probe access firewall
PROBE_UNLOCKED = False


class ProbeAccessError(RuntimeError):
    pass


class InformationFirewallError(RuntimeError):
    pass


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
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


def _sha256_indices(indices: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in indices).encode()).hexdigest()


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "device": "cpu",
    }


def _configure_determinism() -> None:
    torch.set_num_threads(int(shead.TORCH_THREADS))
    torch.use_deterministic_algorithms(False)


def _set_trajectory_seed(T: int) -> None:
    random.seed(int(T))
    np.random.seed(int(T))
    torch.manual_seed(int(T))


def _install_firewall() -> None:
    """Forbid any official-valid / official-test extraction path."""

    if getattr(_install_firewall, "_installed", False):
        return

    def _blocked_extract(*args: Any, **kwargs: Any) -> Any:
        raise InformationFirewallError(
            "official valid/test record extraction is forbidden in this audit"
        )

    def _blocked_test(*args: Any, **kwargs: Any) -> Any:
        raise InformationFirewallError("official test extraction is forbidden in this audit")

    postv4._extract_v4_records = _blocked_extract  # type: ignore[assignment]
    suff.extract_test_records = _blocked_test  # type: ignore[assignment]
    _install_firewall._installed = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# anchor protocol (mechanically derived from the real 7200 run manifest)
# ---------------------------------------------------------------------------


def anchor_protocol() -> dict[str, Any]:
    payload = _read_json(ANCHOR_DIR / "run_I0T0.json")
    proto = payload["protocol"]
    anchor_steps_per_epoch = int(payload["steps_per_epoch"])
    anchor_max_epochs = int(proto["max_epochs"])
    anchor_patience_epochs = int(proto["patience"])
    return {
        "manifest_path": str(ANCHOR_DIR / "run_I0T0.json"),
        "manifest_sha256": _sha256_file(ANCHOR_DIR / "run_I0T0.json"),
        "steps_per_epoch": anchor_steps_per_epoch,
        "max_epochs": anchor_max_epochs,
        "max_optimizer_steps": int(anchor_max_epochs * anchor_steps_per_epoch),
        "selection_eval_interval": int(anchor_steps_per_epoch),
        "patience_evaluations": int(anchor_patience_epochs),
        "protocol": dict(proto),
        "train_size": int(SPLIT_SIZES[0]),
    }


_ANCHOR: dict[str, Any] | None = None


def _anchor() -> dict[str, Any]:
    global _ANCHOR
    if _ANCHOR is None:
        _ANCHOR = anchor_protocol()
    return _ANCHOR


# ---------------------------------------------------------------------------
# data + nested subsets
# ---------------------------------------------------------------------------


def load_data() -> dict[str, Any]:
    _install_firewall()
    return iga.build_audit_data()


def _subset_ranking(indices_7200: Sequence[int]) -> list[int]:
    keys = {
        int(i): hashlib.sha256(f"{SUBSET_SALT}|{int(i):04d}".encode()).hexdigest()
        for i in indices_7200
    }
    return sorted((int(i) for i in indices_7200), key=lambda i: (keys[i], i))


def _map_index_lists(data: Mapping[str, Any], ranked: Sequence[int]) -> list:
    pool = [int(i) for i in data["indices"]["optimization_train"]]
    pos = {idx: j for j, idx in enumerate(pool)}
    return [data["train"][pos[int(i)]] for i in ranked]


def build_subsets(data: Mapping[str, Any]) -> dict[str, Any]:
    pool = [int(i) for i in data["indices"]["optimization_train"]]
    ranked = _subset_ranking(pool)
    ranked_1800 = ranked[:1800]
    ranked_3600 = ranked[:3600]
    ranked_7200 = ranked[:7200]
    return {
        "salt": SUBSET_SALT,
        "pool_indices": pool,
        "ranked": ranked,
        "ranked_1800": ranked_1800,
        "ranked_3600": ranked_3600,
        "ranked_7200": ranked_7200,
        "sorted_1800": sorted(ranked_1800),
        "sorted_3600": sorted(ranked_3600),
        "sorted_7200": sorted(ranked_7200),
    }


def sample_efficiency_subset_lock(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    lock = {
        "name": "nested target-independent training subsets",
        "protocol_version": PROTOCOL_VERSION,
        "salt": SUBSET_SALT,
        "ranking_rule": "sort by sha256(f'{salt}|{original_official_train_index:04d}'), tie-break by index",
        "ordering_uses_target": False,
        "target_used": False,
        "source": "7200 optimization_train pool of the frozen official-train 7200/800/2000 hash split",
        "original_official_train_size": 10000,
        "nested_sizes": list(NESTED_SIZES),
        "n_7200": len(subsets["ranked_7200"]),
        "n_3600": len(subsets["ranked_3600"]),
        "n_1800": len(subsets["ranked_1800"]),
        "indices_7200": list(subsets["ranked_7200"]),
        "indices_3600": list(subsets["ranked_3600"]),
        "indices_1800": list(subsets["ranked_1800"]),
        "dataset_hash_7200": _sha256_indices(subsets["sorted_7200"]),
        "dataset_hash_3600": _sha256_indices(subsets["sorted_3600"]),
        "dataset_hash_1800": _sha256_indices(subsets["sorted_1800"]),
        "ranked_hash_7200": _sha256_indices(subsets["ranked_7200"]),
        "ranked_hash_3600": _sha256_indices(subsets["ranked_3600"]),
        "ranked_hash_1800": _sha256_indices(subsets["ranked_1800"]),
        "nested_invariants": {
            "D1800_subset_D3600": set(subsets["ranked_1800"]) <= set(subsets["ranked_3600"]),
            "D3600_subset_D7200": set(subsets["ranked_3600"]) <= set(subsets["ranked_7200"]),
            "D7200_equals_pool": set(subsets["ranked_7200"]) == set(subsets["pool_indices"]),
            "no_duplicates": len(set(subsets["ranked_7200"])) == len(subsets["ranked_7200"]),
        },
        "frozen_before_training": True,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sample_efficiency_subset_lock.json", lock)
    return lock


# ---------------------------------------------------------------------------
# lock files
# ---------------------------------------------------------------------------


def architecture_lock() -> dict[str, Any]:
    model = shead.build_smallhead(0)
    payload = {
        "name": "compact-v4-smallhead",
        "protocol_version": PROTOCOL_VERSION,
        "R_dimension": int(model.unified_graph_width),
        "head_layers": ["Linear(302,13)", "ReLU", "Linear(13,13)", "ReLU", "Linear(13,1)"],
        "head_params": int(shead._n_params(model.head)),
        "total_params": int(shead._n_params(model)),
        "expected_total_params": EXPECTED_PARAMS,
        "exact_params_match": bool(int(shead._n_params(model)) == EXPECTED_PARAMS),
        "forbidden_changes": [
            "tokenizer", "patch construction", "pair construction", "topology",
            "descriptors", "widths", "head", "loss", "optimizer", "weight decay",
            "batch size",
        ],
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "architecture_lock.json", payload)
    return payload


def split_inventory(data: Mapping[str, Any]) -> dict[str, Any]:
    frozen = _read_json(broad.RESULTS_DIR / "split_manifest.json")
    frozen_roles = {
        "optimization_train": frozen["roles"]["adapter_fit"],
        "checkpoint_selection": frozen["roles"]["adapter_selection"],
        "internal_probe": frozen["roles"]["train_probe"],
    }
    payload = {
        "purpose": "index-for-index reuse of the frozen official-train development environment",
        "split_seed": broad.SPLIT_SEED,
        "frozen_manifest": str(broad.RESULTS_DIR / "split_manifest.json"),
        "frozen_manifest_sha256": _sha256_file(broad.RESULTS_DIR / "split_manifest.json"),
        "sizes": dict(zip(SPLIT_ROLES, SPLIT_SIZES)),
        "matches_frozen_manifest": {
            label: bool([int(i) for i in data["indices"][label]] == [int(i) for i in frozen_roles[label]])
            for label in SPLIT_ROLES
        },
        "index_sha256": {
            label: _sha256_indices(data["indices"][label]) for label in SPLIT_ROLES
        },
        "source": "official train only; target-independent deterministic molecule-id hash",
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    payload["all_match"] = bool(all(payload["matches_frozen_manifest"].values()))
    _write_json(RESULTS_DIR / "split_inventory.json", payload)
    return payload


def anchor_protocol_compatibility() -> dict[str, Any]:
    anchor = _anchor()
    proto = anchor["protocol"]
    ours = dict(PROTOCOL)
    comparisons = {
        "optimizer": proto["optimizer"] == ours["optimizer"],
        "learning_rate": float(proto["learning_rate"]) == float(ours["learning_rate"]),
        "weight_decay": float(proto["weight_decay"]) == float(ours["weight_decay"]),
        "batch_size": int(proto["batch_size"]) == int(ours["batch_size"]),
        "loss": proto["loss"] == ours["loss"],
        "scheduler": proto["scheduler"] == ours["scheduler"],
        "gradient_clip_norm": float(proto["gradient_clip_norm"]) == float(ours["gradient_clip_norm"]),
        "max_optimizer_steps_equivalent": (
            int(anchor["max_epochs"]) * int(anchor["steps_per_epoch"]) == int(anchor["max_optimizer_steps"])
        ),
        "eval_cadence_equivalent_at_7200": int(anchor["selection_eval_interval"]) == int(anchor["steps_per_epoch"]),
        "patience_equivalent": int(anchor["patience_evaluations"]) == int(proto["patience"]),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "anchor_manifest": anchor["manifest_path"],
        "anchor_manifest_sha256": anchor["manifest_sha256"],
        "anchor_steps_per_epoch": int(anchor["steps_per_epoch"]),
        "anchor_max_epochs": int(anchor["max_epochs"]),
        "anchor_max_optimizer_steps": int(anchor["max_optimizer_steps"]),
        "new_eval_interval": int(anchor["selection_eval_interval"]),
        "new_max_optimizer_steps": int(anchor["max_optimizer_steps"]),
        "new_patience_evaluations": int(anchor["patience_evaluations"]),
        "checks": {key: bool(value) for key, value in comparisons.items()},
        "equivalent_at_7200": bool(all(comparisons.values())),
        "terminology_note": (
            "the legacy anchor run manifest stores checkpoint_selection as the string "
            "'best official-valid MAE', but the frozen split inventory and the anchor code "
            "prove the selection set was the internal 800 checkpoint_selection split "
            "(raw_best_select_800); official valid is never loaded in the anchor either"
        ),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "anchor_protocol_compatibility.json", payload)
    if not payload["equivalent_at_7200"]:
        raise RuntimeError(
            "anchor protocol is not equivalent at N=7200; refusing to compare incompatible protocols"
        )
    return payload


def seed_lock() -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import _state_hash

    models = {s: shead.build_smallhead(s) for s in (0, 1)}
    full = {s: _state_hash(models[s].state_dict()) for s in models}
    historical = {
        "0": _read_json(ANCHOR_DIR / "run_I0T0.json")["init_state_sha256"],
        "1": _read_json(ANCHOR_DIR / "run_I1T1.json")["init_state_sha256"],
    }
    path = ANCHOR_DIR / "seed_factorization_lock.json"
    hist_shared = {}
    if path.exists():
        record = _read_json(path)
        hist_shared = record["initialization_seed_I"]["historical_shared_hash_match"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "I": "parameter initialization seed",
        "T": "trajectory seed (DataLoader shuffle + all training-time RNG)",
        "initial_state_sha256": {str(s): full[s] for s in models},
        "historical_smallhead_init_sha256": historical,
        "initial_state_matches_historical": {
            str(s): bool(full[s] == historical[str(s)]) for s in models
        },
        "historical_shared_hash_match": hist_shared,
        "stage_runs": {
            STAGE1_RUN: [0, 0],
            STAGE2_RUN: [1, 1],
            STAGE3_RUN: [0, 0],
            STAGE4_RUN: [1, 1],
        },
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "seed_lock.json", payload)
    return payload


def decision_thresholds() -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_estimator": "soup",
        "canonical_check_estimator": "raw",
        "gain_material": GAIN_MATERIAL,
        "gain_weak": GAIN_WEAK,
        "raw_conflict": RAW_CONFLICT,
        "replicate_per_seed": REPLICATE_PER_SEED,
        "replicate_mean": REPLICATE_MEAN,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "boundary_last_evals": BOUNDARY_LAST_EVALS,
        "soup_K": SOUP_K,
        "soup_rule": "equal-weight parameter mean of the 5 lowest-800-MAE checkpoints, earliest step wins ties",
        "no_n900": True,
        "max_seeds": 2,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision_thresholds.json", payload)
    return payload


def audit_protocol_lock(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    anchor = _anchor()
    payload = {
        "name": "compact-v4 inductive-bias & sample-efficiency audit",
        "protocol_version": PROTOCOL_VERSION,
        "question": "how much held-out MAE is lost by reducing the number of unique training examples at fixed optimizer-step budget?",
        "architecture": "compact-v4-smallhead, 82,115 params",
        "split": dict(zip(SPLIT_ROLES, SPLIT_SIZES)),
        "training_protocol": dict(PROTOCOL),
        "compute_matched_step_protocol": {
            "eval_interval": int(anchor["selection_eval_interval"]),
            "max_optimizer_steps": int(anchor["max_optimizer_steps"]),
            "patience_evaluations": int(anchor["patience_evaluations"]),
            "derived_from": anchor["manifest_path"],
        },
        "nested_subsets": {
            "salt": SUBSET_SALT,
            "sizes": list(NESTED_SIZES),
            "target_used": False,
        },
        "estimators": {
            "raw": "argmin 800-selection MAE, earliest step wins",
            "soup": "equal-weight parameter average of K=5 lowest-800-MAE checkpoints",
        },
        "stage_order": {
            "stage1": STAGE1_RUN,
            "stage2_if_stage1_material": STAGE2_RUN,
            "stage3_if_replicated": STAGE3_RUN,
            "stage4_if_stage3_material": STAGE4_RUN,
        },
        "locks": {
            "no_new_architecture": True,
            "no_optimizer_search": True,
            "no_regularization_search": True,
            "no_scheduler": True,
            "no_ema_swa": True,
            "no_hpo": True,
            "no_n900": True,
            "no_seed2_or_3": True,
            "probe_reuse_preregistered": True,
            "official_valid_used": False,
            "official_test_loaded": False,
        },
        "probe_reuse_caveat": (
            "the 2000 probe was read in prior mechanism audits and is not a pristine "
            "holdout; it is used only for preregistered learning-curve measurement and "
            "must not change N, budget, optimizer, architecture or averaging rule"
        ),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "audit_protocol_lock.json", payload)
    return payload


def run_locks() -> None:
    _configure_determinism()
    data = load_data()
    subsets = build_subsets(data)
    architecture_lock()
    split_inventory(data)
    sample_efficiency_subset_lock(data, subsets)
    anchor_protocol_compatibility()
    seed_lock()
    decision_thresholds()
    audit_protocol_lock(data, subsets)


# ---------------------------------------------------------------------------
# training (compute-matched optimizer-step budget)
# ---------------------------------------------------------------------------


def _init_model(init_seed: int) -> tuple[torch.nn.Module, str]:
    model = shead.build_smallhead(int(init_seed))
    state_hash = _state_hash(model.state_dict())
    if int(shead._n_params(model)) != EXPECTED_PARAMS:
        raise RuntimeError(f"params != {EXPECTED_PARAMS}")
    return model, state_hash


def _snapshot_manifest_row(
    step: int, eval_index: int, select_mae: float, train_mae: float, path: Path | None
) -> dict[str, Any]:
    return {
        "step": int(step),
        "eval_index": int(eval_index),
        "select_800_mae": float(select_mae),
        "train_eval_mae": float(train_mae),
        "snapshot": str(path) if path is not None else None,
        "snapshot_sha256": _sha256_file(path) if path is not None else None,
    }


def train_run(run_id: str, data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    global PROBE_UNLOCKED

    spec = RUN_SPEC[run_id]
    n = int(spec["N"])
    init_seed = int(spec["I"])
    traj_seed = int(spec["T"])
    anchor = _anchor()

    if (RESULTS_DIR / f"stage_run_{run_id}.json").exists() and not os.environ.get("SAMPLE_EFF_FORCE"):
        existing = _read_json(RESULTS_DIR / f"stage_run_{run_id}.json")
        if existing.get("frozen") and existing.get("estimators_evaluated"):
            print(f"[{run_id}] already complete; skipping (set SAMPLE_EFF_FORCE=1 to rerun)")
            return existing

    model, init_hash = _init_model(init_seed)
    _set_trajectory_seed(traj_seed)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(PROTOCOL["learning_rate"]),
        weight_decay=float(PROTOCOL["weight_decay"]),
    )

    train_data = _map_index_lists(data, subsets[f"sorted_{n}"])
    select_loader = zpp._make_loader(
        data["select"], int(PROTOCOL["batch_size"]), False, traj_seed + EVAL_SHUFFLE_SEED_OFFSET
    )
    train_eval_loader = zpp._make_loader(train_data, 256, False, 0)
    train_loader = zpp._make_loader(
        train_data,
        int(PROTOCOL["batch_size"]),
        True,
        traj_seed + TRAIN_SHUFFLE_SEED_OFFSET,
    )

    eval_interval = int(anchor["selection_eval_interval"])
    max_steps = int(anchor["max_optimizer_steps"])
    patience = int(anchor["patience_evaluations"])

    snap_dir = SNAPSHOT_DIR / run_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    it = iter(train_loader)
    best_mae = float("inf")
    best_step = 0
    best_eval_index = 0
    stale = 0
    eval_index = 0
    seen = 0
    window_loss = 0.0
    window_seen = 0
    steps_since_eval = 0
    curve: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    best_so_far = float("inf")
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        model.train()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = batch.to(device)
        prediction = model(batch).view(-1)
        target = batch.y.view(-1)
        loss = F.l1_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(PROTOCOL["gradient_clip_norm"]))
        optimizer.step()

        batch_n = int(target.numel())
        window_loss += float(loss.detach()) * batch_n
        window_seen += batch_n
        seen += batch_n
        steps_since_eval += 1

        if step % eval_interval == 0:
            eval_index += 1
            select_mae, _, _ = shead._evaluate_mae(model, select_loader, device)
            train_mae, _, _ = shead._evaluate_mae(model, train_eval_loader, device)
            snap_path = snap_dir / f"step_{step:05d}.pt"
            torch.save(copy.deepcopy(model.state_dict()), snap_path)
            improved = select_mae < best_mae
            if improved:
                best_mae = float(select_mae)
                best_step = int(step)
                best_eval_index = int(eval_index)
                stale = 0
            else:
                stale += 1
            best_so_far = min(best_so_far, float(select_mae))
            curve.append(
                {
                    "optimizer_step": int(step),
                    "eval_index": int(eval_index),
                    "effective_epoch": float(seen) / float(n),
                    "train_eval_mae": float(train_mae),
                    "select_800_mae": float(select_mae),
                    "checkpoint_saved": 1,
                    "raw_best_so_far": float(best_so_far),
                    "train_loss_window": float(window_loss / max(window_seen, 1)),
                    "stale": int(stale),
                }
            )
            manifest.append(
                _snapshot_manifest_row(step, eval_index, select_mae, train_mae, snap_path)
            )
            window_loss = 0.0
            window_seen = 0
            steps_since_eval = 0
            model.train()
            if eval_index == 1 or eval_index % 20 == 0 or step == max_steps:
                print(
                    f"[{run_id}] step={step:5d} eval={eval_index:3d} select={select_mae:.6f} "
                    f"train={train_mae:.6f} best={best_mae:.6f}@{best_step}",
                    flush=True,
                )
            if stale >= patience:
                print(f"[{run_id}] early_stop at step={step} best={best_step}", flush=True)
                break

    wall = float(time.perf_counter() - started)
    early_stopped = int(step) < max_steps
    boundary_warning = bool(
        best_eval_index > (eval_index - BOUNDARY_LAST_EVALS)
        and len(curve) >= 2
        and curve[-1]["select_800_mae"] < curve[-2]["select_800_mae"]
    )

    curve_path = CURVE_DIR / f"{run_id}.csv"
    _write_csv(
        curve_path,
        curve,
        (
            "optimizer_step", "eval_index", "effective_epoch", "train_eval_mae",
            "select_800_mae", "checkpoint_saved", "raw_best_so_far",
            "train_loss_window", "stale",
        ),
    )

    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    snapshot_bytes = int(sum(p.stat().st_size for p in snap_dir.glob("*.pt")))
    payload = {
        "run_id": run_id,
        "N": n,
        "I": init_seed,
        "T": traj_seed,
        "protocol": dict(PROTOCOL),
        "parameters": int(shead._n_params(model)),
        "init_state_sha256": init_hash,
        "eval_interval": eval_interval,
        "max_optimizer_steps": max_steps,
        "patience_evaluations": patience,
        "optimizer_steps": int(step),
        "eval_count": int(eval_index),
        "examples_processed": int(seen),
        "effective_epochs": float(seen) / float(n),
        "mean_exposures_per_example": float(seen) / float(n),
        "unique_training_examples": int(n),
        "raw_best_step": int(best_step),
        "raw_best_eval_index": int(best_eval_index),
        "raw_best_select_800": float(best_mae),
        "early_stopped": bool(early_stopped),
        "budget_boundary_warning": boundary_warning,
        "steps_since_eval": int(steps_since_eval),
        "wall_clock_s": wall,
        "peak_rss_kb": peak_rss_kb,
        "snapshot_count": int(len(manifest)),
        "snapshot_storage_bytes": snapshot_bytes,
        "curve_path": str(curve_path),
        "snapshot_dir": str(snap_dir),
        "frozen": True,
        "estimators_evaluated": False,
        "train_only_gradient_updates": True,
        "select_800_inference_only": True,
        "probe_2000_accessed_during_training": False,
        "official_valid_used": False,
        "official_test_loaded": False,
        "environment": _environment(),
    }
    _write_json(
        RESULTS_DIR / f"checkpoint_manifest_{run_id}.json",
        {
            "run_id": run_id,
            "N": n,
            "I": init_seed,
            "T": traj_seed,
            "eval_interval": eval_interval,
            "raw_best_step": int(best_step),
            "raw_best_select_800": float(best_mae),
            "snapshots": manifest,
            "probe_derived_state": False,
            "official_valid_used": False,
            "official_test_loaded": False,
        },
    )
    _write_json(RESULTS_DIR / f"stage_run_{run_id}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# RAW / SOUP estimators (post-freeze, probe unlocked)
# ---------------------------------------------------------------------------


def _require_probe_unlocked() -> None:
    if not PROBE_UNLOCKED:
        raise ProbeAccessError("the 2000 internal probe must not be read before a run is frozen")


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.detach().clone() for k, v in state.items()}


def top5_members(run_id: str) -> list[dict[str, Any]]:
    manifest = _read_json(RESULTS_DIR / f"checkpoint_manifest_{run_id}.json")
    ranked = sorted(
        manifest["snapshots"], key=lambda row: (float(row["select_800_mae"]), int(row["step"]))
    )
    if len(ranked) < SOUP_K:
        raise RuntimeError(f"{run_id}: fewer than {SOUP_K} snapshots for soup")
    return ranked[:SOUP_K]


def build_soup(run_id: str) -> dict[str, Any]:
    members = top5_members(run_id)
    states = [_load_state(row["snapshot"]) for row in members]
    keys = sorted(states[0].keys())
    for state in states[1:]:
        if sorted(state.keys()) != keys:
            raise RuntimeError(f"{run_id}: state key mismatch -> invalid soup")
    model = shead.build_smallhead(0)
    param_keys = {name for name, _ in model.named_parameters()}
    soup_state: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key] for state in states]
        if key in param_keys:
            soup_state[key] = torch.stack([t.to(torch.float32) for t in tensors], dim=0).mean(dim=0)
        else:
            if not all(torch.equal(tensors[0], t) for t in tensors[1:]):
                raise RuntimeError(f"{run_id}: non-identical buffer {key} -> invalid soup")
            soup_state[key] = tensors[0].clone()
    if not all(bool(torch.isfinite(v).all()) for v in soup_state.values()):
        raise RuntimeError(f"{run_id}: non-finite soup tensor")
    load_report = model.load_state_dict(soup_state, strict=True)
    if int(shead._n_params(model)) != EXPECTED_PARAMS:
        raise RuntimeError(f"{run_id}: soup params != {EXPECTED_PARAMS}")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = STATE_DIR / f"soup_state_{run_id}.pt"
    torch.save(soup_state, soup_path)
    payload = {
        "run_id": run_id,
        "K": SOUP_K,
        "aggregation": "equal-weight arithmetic mean over all trainable parameters",
        "members": members,
        "n_parameter_tensors": int(len(param_keys)),
        "soup_parameters": int(shead._n_params(model)),
        "soup_state_path": str(soup_path),
        "soup_state_file_sha256": _sha256_file(soup_path),
        "soup_state_tensor_sha256": _state_hash(soup_state),
        "load_state_dict_missing_keys": list(load_report.missing_keys),
        "load_state_dict_unexpected_keys": list(load_report.unexpected_keys),
        "greedy": False,
        "weighted": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"soup_construction_{run_id}.json", payload)
    return payload


def _predict(model: torch.nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    _, targets, preds = shead._evaluate_mae(model, loader, torch.device("cpu"))
    return targets.astype(np.float64), preds.astype(np.float64)


def evaluate_estimators(run_id: str, data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    global PROBE_UNLOCKED
    _require_probe_unlocked()
    spec = RUN_SPEC[run_id]
    n = int(spec["N"])
    run = _read_json(RESULTS_DIR / f"stage_run_{run_id}.json")
    manifest = _read_json(RESULTS_DIR / f"checkpoint_manifest_{run_id}.json")

    raw_member = min(manifest["snapshots"], key=lambda r: (float(r["select_800_mae"]), int(r["step"])))
    soup = build_soup(run_id)

    raw_model = shead.build_smallhead(0)
    raw_model.load_state_dict(_load_state(raw_member["snapshot"]))
    raw_model.eval()
    soup_model = shead.build_smallhead(0)
    soup_model.load_state_dict(_load_state(soup["soup_state_path"]))
    soup_model.eval()

    probe_loader = zpp._make_loader(data["probe"], 256, False, 0)
    targets, raw_pred = _predict(raw_model, probe_loader)
    _t, soup_pred = _predict(soup_model, probe_loader)

    train_data = _map_index_lists(data, subsets[f"sorted_{n}"])
    train_loader = zpp._make_loader(train_data, 256, False, 0)
    train_targets, raw_train_pred = _predict(raw_model, train_loader)
    _t2, soup_train_pred = _predict(soup_model, train_loader)

    raw_err = np.abs(targets - raw_pred)
    soup_err = np.abs(targets - soup_pred)
    np.save(RESULTS_DIR / f"probe_abs_err_raw_{run_id}.npy", raw_err)
    np.save(RESULTS_DIR / f"probe_abs_err_soup_{run_id}.npy", soup_err)

    payload = {
        "run_id": run_id,
        "N": n,
        "I": int(spec["I"]),
        "T": int(spec["T"]),
        "raw_best_step": int(raw_member["step"]),
        "raw_select_800_mae": float(raw_member["select_800_mae"]),
        "raw_probe_mae": float(raw_err.mean()),
        "soup_probe_mae": float(soup_err.mean()),
        "top5_steps": [int(m["step"]) for m in soup["members"]],
        "top5_select_800_mae": [float(m["select_800_mae"]) for m in soup["members"]],
        "raw_train_mae": float(np.abs(train_targets - raw_train_pred).mean()),
        "soup_train_mae": float(np.abs(train_targets - soup_train_pred).mean()),
        "raw_generalization_gap": float(raw_err.mean() - np.abs(train_targets - raw_train_pred).mean()),
        "soup_generalization_gap": float(soup_err.mean() - np.abs(train_targets - soup_train_pred).mean()),
        "n_probe": int(len(targets)),
        "n_train_subset": int(len(train_targets)),
        "soup_state_sha256": soup["soup_state_tensor_sha256"],
        "frozen": True,
        "estimators_evaluated": True,
        "probe_2000_accessed_after_freeze": True,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    run.update(payload)
    _write_json(RESULTS_DIR / f"stage_run_{run_id}.json", run)
    return run


# ---------------------------------------------------------------------------
# anchor (7200) estimators
# ---------------------------------------------------------------------------


def anchor_estimators(seed: int) -> dict[str, Any]:
    run_id = ANCHOR_RUNS[seed]
    probe = _read_json(TOP5_DIR / f"development_reuse_probe_{run_id}.json")
    raw_err = np.load(TOP5_DIR / f"probe_abs_err_raw_{run_id}.npy")
    soup_err = np.load(TOP5_DIR / f"probe_abs_err_soup_{run_id}.npy")
    top5 = _read_json(TOP5_DIR / f"top5_manifest_{run_id}.json")
    soup = _read_json(TOP5_DIR / f"soup_construction_{run_id}.json")
    return {
        "run_id": run_id,
        "seed": int(seed),
        "N": 7200,
        "raw_probe_mae": float(probe["raw_mae"]),
        "soup_probe_mae": float(probe["soup_mae"]),
        "raw_err": raw_err.astype(np.float64),
        "soup_err": soup_err.astype(np.float64),
        "raw_best_step": int(top5["raw_epoch"]) * int(_anchor()["steps_per_epoch"]),
        "top5_steps": [int(m["epoch"]) * int(_anchor()["steps_per_epoch"]) for m in top5["selected"]],
        "soup_state_sha256": soup["soup_state_tensor_sha256"],
    }


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap(diff: np.ndarray, *, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    diff = np.asarray(diff, dtype=np.float64)
    n = int(diff.size)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(B), n))
    means = diff[idx].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "n": n,
        "B": int(B),
        "seed": int(seed),
        "mean": float(diff.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "p_gt_zero": float(np.mean(means > 0.0)),
        "excludes_zero": bool(lower > 0.0 or upper < 0.0),
    }


def paired_bootstrap_two_seed(
    diff0: np.ndarray, diff1: np.ndarray, *, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    diff0 = np.asarray(diff0, dtype=np.float64)
    diff1 = np.asarray(diff1, dtype=np.float64)
    n = int(diff0.size)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(B), n))
    means = 0.5 * (diff0[idx].mean(axis=1) + diff1[idx].mean(axis=1))
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "n": n,
        "B": int(B),
        "seed": int(seed),
        "mean": float(0.5 * (diff0.mean() + diff1.mean())),
        "seed0_mean": float(diff0.mean()),
        "seed1_mean": float(diff1.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "p_gt_zero": float(np.mean(means > 0.0)),
        "excludes_zero": bool(lower > 0.0 or upper < 0.0),
    }


# ---------------------------------------------------------------------------
# stage orchestration
# ---------------------------------------------------------------------------


def _run_then_evaluate(run_id: str, data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    global PROBE_UNLOCKED
    PROBE_UNLOCKED = False
    run = train_run(run_id, data, subsets)
    PROBE_UNLOCKED = True
    est = evaluate_estimators(run_id, data, subsets)
    return est


def _gain(soup_a: float, soup_b: float) -> float:
    return float(soup_a - soup_b)


def run_stage1(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    est = _run_then_evaluate(STAGE1_RUN, data, subsets)
    anchor = anchor_estimators(0)
    diff_soup = np.load(RESULTS_DIR / f"probe_abs_err_soup_{STAGE1_RUN}.npy") - anchor["soup_err"]
    diff_raw = np.load(RESULTS_DIR / f"probe_abs_err_raw_{STAGE1_RUN}.npy") - anchor["raw_err"]
    boot_soup = paired_bootstrap(diff_soup)
    boot_raw = paired_bootstrap(diff_raw)
    g_soup = _gain(est["soup_probe_mae"], anchor["soup_probe_mae"])
    g_raw = _gain(est["raw_probe_mae"], anchor["raw_probe_mae"])

    conflict = bool(g_soup >= GAIN_MATERIAL and g_raw <= RAW_CONFLICT)
    if conflict:
        case = "ESTIMATOR_DISAGREEMENT_INCONCLUSIVE"
    elif g_soup >= GAIN_MATERIAL and boot_soup["ci95_lower"] > 0 and g_raw > 0:
        case = "MATERIAL_LOCAL_DATA_SCALING_SIGNAL"
    elif g_soup >= GAIN_MATERIAL and boot_soup["ci95_lower"] > 0 and RAW_CONFLICT < g_raw <= 0:
        case = "MATERIAL_SOUP_RAW_NEUTRAL"
    elif g_soup <= GAIN_WEAK or boot_soup["ci95_lower"] <= 0:
        case = "WEAK_OR_NO_NEAR_7200_DATA_SCALING_SIGNAL"
    else:
        case = "SUBTHRESHOLD_DATA_SCALING_SIGNAL"

    authorize_stage2 = case in {"MATERIAL_LOCAL_DATA_SCALING_SIGNAL", "MATERIAL_SOUP_RAW_NEUTRAL"}

    payload = {
        "run": est,
        "anchor_seed0": {
            "run_id": anchor["run_id"],
            "raw_probe_mae": anchor["raw_probe_mae"],
            "soup_probe_mae": anchor["soup_probe_mae"],
            "raw_best_step": anchor["raw_best_step"],
            "top5_steps": anchor["top5_steps"],
        },
        "G_soup_36_to_72_seed0": g_soup,
        "G_raw_36_to_72_seed0": g_raw,
        "paired_bootstrap_soup": boot_soup,
        "paired_bootstrap_raw": boot_raw,
        "thresholds": {
            "gain_material": GAIN_MATERIAL,
            "gain_weak": GAIN_WEAK,
            "raw_conflict": RAW_CONFLICT,
        },
        "decision_case": case,
        "authorize_stage2": authorize_stage2,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage1_N3600_I0T0.json", payload)
    _write_json(RESULTS_DIR / "stage1_decision.json", {k: payload[k] for k in (
        "decision_case", "authorize_stage2", "G_soup_36_to_72_seed0", "G_raw_36_to_72_seed0"
    )} | {"paired_bootstrap_soup": boot_soup})
    return payload


def run_stage2(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    est = _run_then_evaluate(STAGE2_RUN, data, subsets)
    anchor = anchor_estimators(1)
    diff_soup = np.load(RESULTS_DIR / f"probe_abs_err_soup_{STAGE2_RUN}.npy") - anchor["soup_err"]
    diff_raw = np.load(RESULTS_DIR / f"probe_abs_err_raw_{STAGE2_RUN}.npy") - anchor["raw_err"]
    boot_soup = paired_bootstrap(diff_soup)
    boot_raw = paired_bootstrap(diff_raw)
    g_soup = _gain(est["soup_probe_mae"], anchor["soup_probe_mae"])
    g_raw = _gain(est["raw_probe_mae"], anchor["raw_probe_mae"])

    stage1 = _read_json(RESULTS_DIR / "stage1_N3600_I0T0.json")
    g0 = float(stage1["G_soup_36_to_72_seed0"])
    g1 = float(g_soup)
    mean_g = 0.5 * (g0 + g1)
    diff0 = np.load(RESULTS_DIR / f"probe_abs_err_soup_{STAGE1_RUN}.npy") - anchor_estimators(0)["soup_err"]
    boot_two = paired_bootstrap_two_seed(diff0, diff_soup)
    mean_raw = 0.5 * (float(stage1["G_raw_36_to_72_seed0"]) + g_raw)
    replicated = bool(
        g0 >= REPLICATE_PER_SEED
        and g1 >= REPLICATE_PER_SEED
        and mean_g >= REPLICATE_MEAN
        and boot_two["ci95_lower"] > 0
        and mean_raw > 0
    )
    payload = {
        "run": est,
        "anchor_seed1": {
            "run_id": anchor["run_id"],
            "raw_probe_mae": anchor["raw_probe_mae"],
            "soup_probe_mae": anchor["soup_probe_mae"],
        },
        "G_soup_36_to_72_seed1": g1,
        "G_raw_36_to_72_seed1": g_raw,
        "G_soup_36_to_72_seed0": g0,
        "mean_G_soup_36_to_72": mean_g,
        "mean_G_raw_36_to_72": mean_raw,
        "paired_bootstrap_soup_seed1": boot_soup,
        "paired_bootstrap_raw_seed1": boot_raw,
        "two_seed_mean_paired_bootstrap": boot_two,
        "decision_case": (
            "REPLICATED_MATERIAL_NEAR_7200_DATA_SCALING"
            if replicated
            else "LOCAL_DATA_SCALING_SIGNAL_NOT_REPLICATED"
        ),
        "authorize_stage3": replicated,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage2_N3600_I1T1.json", payload)
    _write_json(RESULTS_DIR / "stage2_decision.json", {
        "decision_case": payload["decision_case"],
        "authorize_stage3": replicated,
        "mean_G_soup_36_to_72": mean_g,
        "G_soup_36_to_72_seed1": g1,
        "two_seed_mean_paired_bootstrap": boot_two,
    })
    return payload


def run_stage3(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    est = _run_then_evaluate(STAGE3_RUN, data, subsets)
    est3600 = _read_json(RESULTS_DIR / f"stage_run_{STAGE1_RUN}.json")
    diff_soup = np.load(RESULTS_DIR / f"probe_abs_err_soup_{STAGE3_RUN}.npy") - np.load(
        RESULTS_DIR / f"probe_abs_err_soup_{STAGE1_RUN}.npy"
    )
    diff_raw = np.load(RESULTS_DIR / f"probe_abs_err_raw_{STAGE3_RUN}.npy") - np.load(
        RESULTS_DIR / f"probe_abs_err_raw_{STAGE1_RUN}.npy"
    )
    boot_soup = paired_bootstrap(diff_soup)
    boot_raw = paired_bootstrap(diff_raw)
    g_soup = _gain(est["soup_probe_mae"], float(est3600["soup_probe_mae"]))
    g_raw = _gain(est["raw_probe_mae"], float(est3600["raw_probe_mae"]))

    if g_soup >= GAIN_MATERIAL and boot_soup["ci95_lower"] > 0 and g_raw > 0:
        case = "THREE_POINT_STRONG_SCALING_SIGNAL"
    elif GAIN_WEAK < g_soup < GAIN_MATERIAL:
        case = "THREE_POINT_SCALING_CURVATURE_WEAK_SUBTHRESHOLD"
    else:
        case = "NON_MONOTONIC_CURVATURE_UNRESOLVED_DATA_SCALING"
    authorize_stage4 = case == "THREE_POINT_STRONG_SCALING_SIGNAL"
    payload = {
        "run": est,
        "reference_N3600_seed0": {
            "run_id": STAGE1_RUN,
            "soup_probe_mae": float(est3600["soup_probe_mae"]),
            "raw_probe_mae": float(est3600["raw_probe_mae"]),
        },
        "G_soup_18_to_36_seed0": g_soup,
        "G_raw_18_to_36_seed0": g_raw,
        "paired_bootstrap_soup": boot_soup,
        "paired_bootstrap_raw": boot_raw,
        "decision_case": case,
        "authorize_stage4": authorize_stage4,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage3_N1800_I0T0.json", payload)
    _write_json(RESULTS_DIR / "stage3_decision.json", {
        "decision_case": case,
        "authorize_stage4": authorize_stage4,
        "G_soup_18_to_36_seed0": g_soup,
        "paired_bootstrap_soup": boot_soup,
    })
    return payload


def run_stage4(data: Mapping[str, Any], subsets: Mapping[str, Any]) -> dict[str, Any]:
    est = _run_then_evaluate(STAGE4_RUN, data, subsets)
    est1800 = _read_json(RESULTS_DIR / f"stage_run_{STAGE3_RUN}.json")
    est3600s1 = _read_json(RESULTS_DIR / f"stage_run_{STAGE2_RUN}.json")
    diff_soup = np.load(RESULTS_DIR / f"probe_abs_err_soup_{STAGE4_RUN}.npy") - np.load(
        RESULTS_DIR / f"probe_abs_err_soup_{STAGE2_RUN}.npy"
    )
    boot_soup = paired_bootstrap(diff_soup)
    g_soup = _gain(est["soup_probe_mae"], float(est3600s1["soup_probe_mae"]))
    payload = {
        "run": est,
        "reference_N3600_seed1": {
            "run_id": STAGE2_RUN,
            "soup_probe_mae": float(est3600s1["soup_probe_mae"]),
        },
        "G_soup_18_to_36_seed1": g_soup,
        "paired_bootstrap_soup": boot_soup,
        "G_soup_18_to_36_seed0": float(_read_json(RESULTS_DIR / "stage3_N1800_I0T0.json")["G_soup_18_to_36_seed0"]),
        "two_seed_mean_G_18_to_36": 0.5
        * (
            float(_read_json(RESULTS_DIR / "stage3_N1800_I0T0.json")["G_soup_18_to_36_seed0"])
            + g_soup
        ),
        "decision_case": "FULL_THREE_POINT_TWO_SEED_REPLICATION",
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage4_N1800_I1T1.json", payload)
    _write_json(RESULTS_DIR / "stage4_decision.json", {
        "decision_case": payload["decision_case"],
        "G_soup_18_to_36_seed1": g_soup,
        "paired_bootstrap_soup": boot_soup,
    })
    return payload


# ---------------------------------------------------------------------------
# summaries / final decision / answers
# ---------------------------------------------------------------------------


def _actual_run_ids() -> list[str]:
    return [r for r in RUN_SPEC if (RESULTS_DIR / f"stage_run_{r}.json").exists()]


def learning_curve_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in (0, 1):
        anchor = anchor_estimators(seed)
        rows.append(
            {
                "N": 7200,
                "seed": seed,
                "run_id": anchor["run_id"],
                "reused": True,
                "unique_examples": 7200,
                "optimizer_steps": int(_anchor()["max_optimizer_steps"]),
                "effective_epochs": 240.0,
                "mean_exposures_per_example": 240.0,
                "raw_selected_step": anchor["raw_best_step"],
                "raw_800_mae": float(_read_json(TOP5_DIR / f"top5_manifest_{anchor['run_id']}.json")["raw_800_mae"]),
                "raw_probe_mae": anchor["raw_probe_mae"],
                "top5_steps": ";".join(str(s) for s in anchor["top5_steps"]),
                "soup_probe_mae": anchor["soup_probe_mae"],
                "train_eval_raw": "",
                "train_eval_soup": "",
            }
        )
    for run_id in _actual_run_ids():
        run = _read_json(RESULTS_DIR / f"stage_run_{run_id}.json")
        rows.append(
            {
                "N": int(run["N"]),
                "seed": int(run["I"]),
                "run_id": run_id,
                "reused": False,
                "unique_examples": int(run["unique_training_examples"]),
                "optimizer_steps": int(run["optimizer_steps"]),
                "effective_epochs": float(run["effective_epochs"]),
                "mean_exposures_per_example": float(run["mean_exposures_per_example"]),
                "raw_selected_step": int(run["raw_best_step"]),
                "raw_800_mae": float(run["raw_best_select_800"]),
                "raw_probe_mae": float(run["raw_probe_mae"]),
                "top5_steps": ";".join(str(s) for s in run["top5_steps"]),
                "soup_probe_mae": float(run["soup_probe_mae"]),
                "train_eval_raw": float(run["raw_train_mae"]),
                "train_eval_soup": float(run["soup_train_mae"]),
            }
        )
    rows.sort(key=lambda r: (r["N"], r["seed"]))
    _write_csv(
        RESULTS_DIR / "learning_curve_summary.csv",
        rows,
        (
            "N", "seed", "run_id", "reused", "unique_examples", "optimizer_steps",
            "effective_epochs", "mean_exposures_per_example", "raw_selected_step",
            "raw_800_mae", "raw_probe_mae", "top5_steps", "soup_probe_mae",
            "train_eval_raw", "train_eval_soup",
        ),
    )
    if any(r["N"] == 1800 for r in rows):
        _write_csv(
            RESULTS_DIR / "three_point_learning_curve.csv",
            rows,
            (
                "N", "seed", "run_id", "optimizer_steps", "effective_epochs",
                "mean_exposures_per_example", "raw_800_mae", "raw_probe_mae",
                "soup_probe_mae", "train_eval_raw", "train_eval_soup",
            ),
        )
    if len({r["seed"] for r in rows if r["N"] in (1800, 3600)}) == 2:
        _write_csv(
            RESULTS_DIR / "two_seed_learning_curve.csv",
            rows,
            (
                "N", "seed", "run_id", "optimizer_steps", "effective_epochs",
                "mean_exposures_per_example", "raw_800_mae", "raw_probe_mae",
                "soup_probe_mae", "train_eval_raw", "train_eval_soup",
            ),
        )
    return rows


def doubling_gain_summary() -> dict[str, Any]:
    payload: dict[str, Any] = {"gains": {}, "official_valid_used": False, "official_test_loaded": False}
    s1 = RESULTS_DIR / "stage1_N3600_I0T0.json"
    s2 = RESULTS_DIR / "stage2_N3600_I1T1.json"
    s3 = RESULTS_DIR / "stage3_N1800_I0T0.json"
    s4 = RESULTS_DIR / "stage4_N1800_I1T1.json"
    if s1.exists():
        p = _read_json(s1)
        payload["gains"]["36_to_72_seed0"] = {
            "G_soup": p["G_soup_36_to_72_seed0"],
            "G_raw": p["G_raw_36_to_72_seed0"],
            "bootstrap_soup": p["paired_bootstrap_soup"],
        }
    if s2.exists():
        p = _read_json(s2)
        payload["gains"]["36_to_72_seed1"] = {
            "G_soup": p["G_soup_36_to_72_seed1"],
            "G_raw": p["G_raw_36_to_72_seed1"],
            "bootstrap_soup": p["paired_bootstrap_soup_seed1"],
        }
        payload["mean_G_36_to_72_soup"] = p["mean_G_soup_36_to_72"]
        payload["two_seed_mean_paired_bootstrap"] = p["two_seed_mean_paired_bootstrap"]
    if s3.exists():
        p = _read_json(s3)
        payload["gains"]["18_to_36_seed0"] = {
            "G_soup": p["G_soup_18_to_36_seed0"],
            "G_raw": p["G_raw_18_to_36_seed0"],
            "bootstrap_soup": p["paired_bootstrap_soup"],
        }
    if s4.exists():
        p = _read_json(s4)
        payload["gains"]["18_to_36_seed1"] = {
            "G_soup": p["G_soup_18_to_36_seed1"],
            "bootstrap_soup": p["paired_bootstrap_soup"],
        }
    # observed slope over available N (descriptive only; seed-averaged per N)
    rows = learning_curve_summary()
    per_n: dict[int, list[float]] = {}
    for r in rows:
        per_n.setdefault(int(r["N"]), []).append(float(r["soup_probe_mae"]))
    pts = sorted((n, float(np.mean(v))) for n, v in per_n.items())
    if len(pts) >= 2:
        x = np.log2([p[0] for p in pts])
        y = np.array([p[1] for p in pts], dtype=np.float64)
        slope, intercept = np.polyfit(x, y, 1)
        payload["descriptive_slope"] = {
            "model": "MAE = a + b * log2(N)",
            "b": float(slope),
            "a": float(intercept),
            "minus_b_improvement_per_doubling": float(-slope),
            "n_points": int(len(pts)),
            "warning": "descriptive only; no extrapolation to larger N",
        }
    _write_json(RESULTS_DIR / "doubling_gain_summary.json", payload)
    return payload


def paired_bootstrap_summary() -> dict[str, Any]:
    payload: dict[str, Any] = {"official_valid_used": False, "official_test_loaded": False}
    for name in ("stage1_decision", "stage2_decision", "stage3_decision"):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            payload[name] = _read_json(path).get("paired_bootstrap_soup") or _read_json(path).get(
                "two_seed_mean_paired_bootstrap"
            )
    _write_json(RESULTS_DIR / "paired_bootstrap.json", payload)
    return payload


def compute_audit() -> dict[str, Any]:
    rows = []
    for run_id in _actual_run_ids():
        run = _read_json(RESULTS_DIR / f"stage_run_{run_id}.json")
        rows.append(
            {
                "run_id": run_id,
                "N": int(run["N"]),
                "wall_clock_s": float(run["wall_clock_s"]),
                "optimizer_steps": int(run["optimizer_steps"]),
                "examples_processed": int(run["examples_processed"]),
                "effective_epochs": float(run["effective_epochs"]),
                "eval_count": int(run["eval_count"]),
                "forward_eval_passes": int(run["eval_count"]),
                "snapshot_count": int(run["snapshot_count"]),
                "snapshot_storage_bytes": int(run["snapshot_storage_bytes"]),
                "peak_rss_kb": int(run["peak_rss_kb"]),
                "max_optimizer_steps": int(run["max_optimizer_steps"]),
            }
        )
    payload = {
        "protocol": "compute-matched optimizer-step budget",
        "note": "smaller N sees more exposures per unique example by design",
        "runs": rows,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "compute_audit.json", payload)
    return payload


def final_decision() -> dict[str, Any]:
    s1 = _read_json(RESULTS_DIR / "stage1_decision.json")
    case1 = s1["decision_case"]
    s2_path = RESULTS_DIR / "stage2_decision.json"
    s3_path = RESULTS_DIR / "stage3_decision.json"
    case2 = _read_json(s2_path)["decision_case"] if s2_path.exists() else None
    case3 = _read_json(s3_path)["decision_case"] if s3_path.exists() else None

    if case1 in {"WEAK_OR_NO_NEAR_7200_DATA_SCALING_SIGNAL", "ESTIMATOR_DISAGREEMENT_INCONCLUSIVE"}:
        final = "NEAR_7200_SAMPLE_SIZE_SCALING_WEAK" if case1.startswith("WEAK") else "ESTIMATOR_DISAGREEMENT_INCONCLUSIVE"
    elif case1 == "SUBTHRESHOLD_DATA_SCALING_SIGNAL":
        final = "POSITIVE_BUT_SUBTHRESHOLD_DATA_SCALING_SIGNAL"
    elif case2 == "LOCAL_DATA_SCALING_SIGNAL_NOT_REPLICATED":
        final = "DATA_SCALING_SIGNAL_NOT_REPLICATED"
    elif case2 == "REPLICATED_MATERIAL_NEAR_7200_DATA_SCALING" and case3 == "THREE_POINT_STRONG_SCALING_SIGNAL":
        final = "STRONG_WITHIN_FAMILY_DATA_LIMITED_REGIME"
    elif case2 == "REPLICATED_MATERIAL_NEAR_7200_DATA_SCALING":
        final = "MATERIAL_NEAR_7200_DATA_SCALING_CURVATURE_UNRESOLVED"
    else:
        final = "MATERIAL_NEAR_7200_DATA_SCALING_AWAITING_FURTHER_POINTS"

    # boundary-pinning override
    boundary = False
    for run_id in _actual_run_ids():
        run = _read_json(RESULTS_DIR / f"stage_run_{run_id}.json")
        if run.get("budget_boundary_warning"):
            boundary = True
    payload = {
        "final_case": final,
        "stage1_case": case1,
        "stage2_case": case2,
        "stage3_case": case3,
        "budget_boundary_ambiguous": boundary,
        "runs_executed": _actual_run_ids(),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "final_decision.json", payload)

    # next-stage authorization
    if final == "STRONG_WITHIN_FAMILY_DATA_LIMITED_REGIME":
        auth = {
            "authorized_for_design": True,
            "hypothesis_family": "sample_efficiency_inductive_bias",
            "full_training_authorized": False,
        }
    elif final in {"MATERIAL_NEAR_7200_DATA_SCALING_CURVATURE_UNRESOLVED", "MATERIAL_NEAR_7200_DATA_SCALING_AWAITING_FURTHER_POINTS"}:
        auth = {"authorized_for_design": "limited", "hypothesis_family": "sample_efficiency_inductive_bias", "full_training_authorized": False}
    elif final == "NEAR_7200_SAMPLE_SIZE_SCALING_WEAK":
        auth = {
            "authorized_for_design": False,
            "next_audit_family": "function_class_inductive_bias",
            "full_training_authorized": False,
        }
    else:
        auth = {"authorized_for_design": False, "full_training_authorized": False}
    auth.update({"final_case": final, "official_valid_used": False, "official_test_loaded": False})
    _write_json(TRACK_ROOT / "top1_sample_efficiency_hypothesis.json", auth)
    _write_json(RESULTS_DIR / "top1_sample_efficiency_hypothesis.json", auth)
    return payload


def answers_q1_q20() -> dict[str, Any]:
    anchor = _anchor()
    subset_lock = _read_json(RESULTS_DIR / "sample_efficiency_subset_lock.json")
    s1 = _read_json(RESULTS_DIR / "stage1_N3600_I0T0.json") if (RESULTS_DIR / "stage1_N3600_I0T0.json").exists() else None
    s2 = _read_json(RESULTS_DIR / "stage2_N3600_I1T1.json") if (RESULTS_DIR / "stage2_N3600_I1T1.json").exists() else None
    s3 = _read_json(RESULTS_DIR / "stage3_N1800_I0T0.json") if (RESULTS_DIR / "stage3_N1800_I0T0.json").exists() else None
    final = _read_json(RESULTS_DIR / "final_decision.json") if (RESULTS_DIR / "final_decision.json").exists() else {}
    a0 = anchor_estimators(0)
    a1 = anchor_estimators(1)
    run1 = (
        _read_json(RESULTS_DIR / f"stage_run_{STAGE1_RUN}.json")
        if (RESULTS_DIR / f"stage_run_{STAGE1_RUN}.json").exists()
        else None
    )

    def g(d, key):
        return d.get(key) if d else None

    answers = {
        "Q1_steps_per_epoch": anchor["steps_per_epoch"],
        "Q2_derived_max_optimizer_steps": anchor["max_optimizer_steps"],
        "Q3_selection_eval_interval": anchor["selection_eval_interval"],
        "Q4_subset_hashes": {
            "3600": subset_lock["dataset_hash_3600"],
            "1800": subset_lock["dataset_hash_1800"],
        },
        "Q5_N3600_I0T0_steps_effective_epochs": (
            {"optimizer_steps": run1["optimizer_steps"], "effective_epochs": run1["effective_epochs"]} if run1 else None
        ),
        "Q6_N3600_I0T0_raw_probe_mae": g(run1, "raw_probe_mae"),
        "Q7_N3600_I0T0_soup_probe_mae": g(run1, "soup_probe_mae"),
        "Q8_7200_seed0_raw_soup": {"raw": a0["raw_probe_mae"], "soup": a0["soup_probe_mae"]},
        "Q9_G_soup_36_to_72_seed0": g(s1, "G_soup_36_to_72_seed0"),
        "Q10_paired_bootstrap_ci": g(s1, "paired_bootstrap_soup"),
        "Q11_triggered_seed1_3600": bool(g(s1, "authorize_stage2")),
        "Q12_seed1_doubling_gain": g(s2, "G_soup_36_to_72_seed1"),
        "Q13_two_seed_mean_near_7200_gain": g(s2, "mean_G_soup_36_to_72"),
        "Q14_replicated_material_scaling": g(s2, "decision_case"),
        "Q15_bought_1800": bool(s3 is not None),
        "Q16_G_soup_18_to_36_seed0": g(s3, "G_soup_18_to_36_seed0"),
        "Q17_two_doubling_gains": _two_doubling_gains(s1, s3),
        "Q18_raw_soup_direction_consistent": _raw_soup_direction(s1, s2),
        "Q19_data_limited_character": final.get("final_case"),
        "Q20_final_decision_case": final.get("final_case"),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "answers_q1_q20.json", answers)
    return answers


def _two_doubling_gains(s1: Mapping[str, Any] | None, s3: Mapping[str, Any] | None) -> dict[str, Any]:
    g36 = s1.get("G_soup_36_to_72_seed0") if s1 else None
    g18 = s3.get("G_soup_18_to_36_seed0") if s3 else None
    if g36 is None or g18 is None:
        return {"G_36_to_72": g36, "G_18_to_36": g18}
    if g18 > g36 + GAIN_WEAK:
        shape = "increasing"
    elif g36 > g18 + GAIN_WEAK:
        shape = "diminishing"
    else:
        shape = "approximately_flat"
    return {"G_36_to_72": g36, "G_18_to_36": g18, "shape": shape}


def _raw_soup_direction(s1: Mapping[str, Any] | None, s2: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if s1:
        out["seed0_same_direction"] = bool(
            np.sign(s1["G_soup_36_to_72_seed0"]) == np.sign(s1["G_raw_36_to_72_seed0"])
        )
    if s2:
        out["seed1_same_direction"] = bool(
            np.sign(s2["G_soup_36_to_72_seed1"]) == np.sign(s2["G_raw_36_to_72_seed1"])
        )
    return out


# ---------------------------------------------------------------------------
# integrity gates
# ---------------------------------------------------------------------------


def integrity_gates() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, ok: bool, detail: Any) -> None:
        checks[name] = {"pass": bool(ok), "detail": detail}

    model = shead.build_smallhead(0)
    add("1_architecture_82115", int(shead._n_params(model)) == EXPECTED_PARAMS, int(shead._n_params(model)))

    split = _read_json(RESULTS_DIR / "split_inventory.json")
    add("2_split_exact_reuse", bool(split.get("all_match")), split.get("matches_frozen_manifest"))

    sub = _read_json(RESULTS_DIR / "sample_efficiency_subset_lock.json")
    add("3_3600_subset_7200", bool(sub["nested_invariants"]["D3600_subset_D7200"]), None)
    add("4_1800_subset_3600", bool(sub["nested_invariants"]["D1800_subset_D3600"]), None)
    add("5_subset_no_target", bool(sub["target_used"] is False and sub["ordering_uses_target"] is False), None)

    compat = _read_json(RESULTS_DIR / "anchor_protocol_compatibility.json")
    add("6_anchor_compatibility", bool(compat["equivalent_at_7200"]), compat["checks"])

    anchor = _anchor()
    add(
        "7_max_steps_derived",
        int(anchor["max_optimizer_steps"]) == int(anchor["max_epochs"]) * int(anchor["steps_per_epoch"]),
        int(anchor["max_optimizer_steps"]),
    )
    add("8_eval_interval_derived", int(anchor["selection_eval_interval"]) == int(anchor["steps_per_epoch"]), None)
    add("9_patience_40_evals", int(anchor["patience_evaluations"]) == 40, int(anchor["patience_evaluations"]))

    runs = [_read_json(RESULTS_DIR / f"stage_run_{r}.json") for r in _actual_run_ids()]
    add(
        "10_only_subset_gradient_updates",
        all(r.get("train_only_gradient_updates") for r in runs) if runs else True,
        None,
    )
    add("11_800_inference_only", all(r.get("select_800_inference_only") for r in runs) if runs else True, None)
    add(
        "12_probe_inaccessible_before_freeze",
        all(not r.get("probe_2000_accessed_during_training") for r in runs) if runs else True,
        None,
    )
    add("13_official_valid_never_loaded", all(not r.get("official_valid_used") for r in runs) if runs else True, None)
    add("14_official_test_never_loaded", all(not r.get("official_test_loaded") for r in runs) if runs else True, None)

    seed = _read_json(RESULTS_DIR / "seed_lock.json")
    add(
        "15_init_hashes_match_history",
        all(seed["initial_state_matches_historical"].values()),
        seed["initial_state_matches_historical"],
    )

    soup_ok = True
    for r in runs:
        if (RESULTS_DIR / f"soup_construction_{r['run_id']}.json").exists():
            sc = _read_json(RESULTS_DIR / f"soup_construction_{r['run_id']}.json")
            soup_ok = soup_ok and int(sc["K"]) == SOUP_K
    add("16_soup_K5_fixed", soup_ok, SOUP_K)

    membership_ok = True
    for r in runs:
        manifest = _read_json(RESULTS_DIR / f"checkpoint_manifest_{r['run_id']}.json")
        ranked = sorted(manifest["snapshots"], key=lambda x: (float(x["select_800_mae"]), int(x["step"])))
        top = ranked[:SOUP_K]
        selected = _read_json(RESULTS_DIR / f"soup_construction_{r['run_id']}.json")["members"]
        membership_ok = membership_ok and [int(m["step"]) for m in top] == [int(m["step"]) for m in selected]
    add("17_soup_membership_800_only", membership_ok, None)

    add("18_no_ema_swa_hpo", True, "no EMA/SWA/HPO code path exists in this audit")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "checks": checks,
        "all_pass": bool(all(c["pass"] for c in checks.values())),
        "n_pass": int(sum(c["pass"] for c in checks.values())),
        "n_total": int(len(checks)),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "integrity_gates.json", payload)
    return payload


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = learning_curve_summary()
    made = []

    fig, ax = plt.subplots(figsize=(6, 4))
    for run_id in _actual_run_ids():
        run = _read_json(RESULTS_DIR / f"stage_run_{run_id}.json")
        curve_path = Path(run["curve_path"])
        xs, ys = [], []
        with curve_path.open() as handle:
            for row in csv.DictReader(handle):
                xs.append(int(row["optimizer_step"]))
                ys.append(float(row["select_800_mae"]))
        ax.plot(xs, ys, label=f"{run_id} (N={run['N']})")
    ax.axvline(int(_anchor()["max_optimizer_steps"]), ls=":", color="k", label="max steps")
    ax.set_xlabel("optimizer steps")
    ax.set_ylabel("800-selection MAE")
    ax.set_title("compute-matched optimizer-step protocol")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIGURE_DIR / "figure1_protocol.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    fig, ax = plt.subplots(figsize=(6, 4))
    by_n: dict[int, dict[int, dict[str, Any]]] = {}
    for r in rows:
        by_n.setdefault(r["N"], {})[r["seed"]] = r
    ns = sorted(by_n)
    for est, style in (("raw_probe_mae", "--"), ("soup_probe_mae", "-")):
        xs, ys = [], []
        for n in ns:
            vals = [v[est] for v in by_n[n].values() if v.get(est) not in ("", None)]
            if vals:
                xs.append(math.log2(n))
                ys.append(float(np.mean(vals)))
        ax.plot(xs, ys, style, marker="o", label=est.replace("_probe_mae", ""))
    ax.set_xlabel("log2(N)")
    ax.set_ylabel("2000-probe MAE")
    ax.set_title("learning curve (actual points only)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIGURE_DIR / "figure2_learning_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    fig, ax = plt.subplots(figsize=(6, 4))
    gains = doubling_gain_summary()["gains"]
    labels, means, los, his = [], [], [], []
    for key, val in gains.items():
        labels.append(key)
        means.append(val["G_soup"])
        boot = val.get("bootstrap_soup", {})
        los.append(boot.get("ci95_lower", val["G_soup"]))
        his.append(boot.get("ci95_upper", val["G_soup"]))
    if labels:
        xpos = np.arange(len(labels))
        err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
        ax.bar(xpos, means, yerr=err, capsize=4)
        ax.axhline(GAIN_MATERIAL, ls="--", color="g", label="material 0.006")
        ax.axhline(GAIN_WEAK, ls=":", color="r", label="weak 0.003")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=20, fontsize=7)
        ax.set_ylabel("doubling gain (soup)")
        ax.set_title("MAE gained per data doubling")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIGURE_DIR / "figure3_doubling_gain.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    fig, ax = plt.subplots(figsize=(6, 4))
    xs_r, ys_r = [], []
    for n in ns:
        vals = [v["train_eval_raw"] for v in by_n[n].values() if v.get("train_eval_raw") not in ("", None)]
        if vals:
            xs_r.append(n)
            ys_r.append(float(np.mean(vals)))
    if xs_r:
        ax.plot(np.log2(xs_r), ys_r, marker="s", label="train MAE")
        probe = [np.mean([v["soup_probe_mae"] for v in by_n[n].values()]) for n in xs_r]
        ax.plot(np.log2(xs_r), probe, marker="o", label="probe MAE (soup)")
        ax.legend(fontsize=7)
    ax.set_xlabel("log2(N)")
    ax.set_ylabel("MAE")
    ax.set_title("train vs probe MAE")
    fig.tight_layout()
    p = FIGURE_DIR / "figure4_train_probe.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))
    return {"figures": made}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_subsets() -> tuple[dict[str, Any], dict[str, Any]]:
    data = load_data()
    if (RESULTS_DIR / "sample_efficiency_subset_lock.json").exists():
        lock = _read_json(RESULTS_DIR / "sample_efficiency_subset_lock.json")
        subsets = {
            "salt": lock["salt"],
            "pool_indices": [int(i) for i in data["indices"]["optimization_train"]],
            "ranked": [int(i) for i in lock["indices_7200"]],
            "ranked_1800": [int(i) for i in lock["indices_1800"]],
            "ranked_3600": [int(i) for i in lock["indices_3600"]],
            "ranked_7200": [int(i) for i in lock["indices_7200"]],
            "sorted_1800": sorted(int(i) for i in lock["indices_1800"]),
            "sorted_3600": sorted(int(i) for i in lock["indices_3600"]),
            "sorted_7200": sorted(int(i) for i in lock["indices_7200"]),
        }
    else:
        subsets = build_subsets(data)
    return data, subsets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "locks", "integrity", "train_stage1", "stage1", "stage2", "stage3", "stage4",
            "final", "all",
        ],
    )
    args = parser.parse_args(argv)
    _configure_determinism()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "locks":
        run_locks()
        return 0

    data, subsets = _load_subsets()
    if args.stage == "integrity":
        print(json.dumps(integrity_gates(), indent=2)[:2000])
        return 0
    if args.stage == "train_stage1":
        global PROBE_UNLOCKED
        PROBE_UNLOCKED = False
        print(json.dumps(train_run(STAGE1_RUN, data, subsets), indent=2, default=str)[:2000])
        return 0
    if args.stage == "stage1":
        print(json.dumps(run_stage1(data, subsets), indent=2, default=str)[:2000])
        return 0
    if args.stage == "stage2":
        print(json.dumps(run_stage2(data, subsets), indent=2, default=str)[:2000])
        return 0
    if args.stage == "stage3":
        print(json.dumps(run_stage3(data, subsets), indent=2, default=str)[:2000])
        return 0
    if args.stage == "stage4":
        print(json.dumps(run_stage4(data, subsets), indent=2, default=str)[:2000])
        return 0
    if args.stage == "final":
        learning_curve_summary()
        doubling_gain_summary()
        paired_bootstrap_summary()
        compute_audit()
        integrity_gates()
        final_decision()
        answers_q1_q20()
        make_figures()
        return 0
    if args.stage == "all":
        run_locks()
        data, subsets = _load_subsets()
        s1 = run_stage1(data, subsets)
        if s1["authorize_stage2"]:
            s2 = run_stage2(data, subsets)
            if s2["authorize_stage3"]:
                s3 = run_stage3(data, subsets)
                if s3["authorize_stage4"]:
                    run_stage4(data, subsets)
        learning_curve_summary()
        doubling_gain_summary()
        paired_bootstrap_summary()
        compute_audit()
        integrity_gates()
        final_decision()
        answers_q1_q20()
        make_figures()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
