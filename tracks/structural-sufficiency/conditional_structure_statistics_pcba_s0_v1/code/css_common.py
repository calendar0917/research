"""Frozen definitions for the conditional structure-statistics round.

Round / study / protocol: ``css-pcba-s0-v1``

The round trains *one from-scratch prediction subject* (EARLY_TASK ``E``,
LATE_TASK ``L`` and, only if purchased, EARLY_SHARED ``S``) on the official
``ogbg-molpcba`` split.  Nothing here is a residual of, a distillation of, or a
fusion with any historical LCRC model: only the *data adapter*, the *structure
index*, the *frozen normalisation* and the *training recipe* are inherited,
read-only.

Read-only inheritance (never modified, never re-defaulted):

* ``lcrc_pcba_structural_gain_sprint_v1/code/sg_common.py`` -- 40-epoch
  from-scratch recipe, ``scheduled_lr(phase=3)``, the frozen 40-epoch order plan,
  the official OGB evaluator, all frozen data identities,
* ``lcrc_pcba_scale_pilot_v1`` -- lipo_v1 PCBA feature cache / official split /
  Train-only normalisation,
* ``lcrc_pcba_cpc_v1/code/cpc_paths.py`` -- the audited rooted two-bond path
  (S2) structure cache and its batch collation,
* ``lcrc_pcba_pipeline_perf_v1/code/pf_pipeline.py`` -- the already-gated
  ``PinnedStager``,
* ``lcrc_pcba_continuation_recipe_v1/code/cc_data.py`` -- only the *source and
  alignment* of the official Test labels (``raw/graph-label.csv.gz``); its
  one-shot freeze gate is explicitly NOT inherited (this round is
  ``test_policy=always`` / DEVELOPMENT_TEST by user authorisation).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# paths / inherited code (read-only)
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

SG_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1"
PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PERF = REPO / "tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1"
LR_PROBE = REPO / "tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1"
CPC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_cpc_v1"
CC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1"
BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"

for _p in (
    str(SG_SPRINT / "code"),  # sg_common -> pilot/perf/lr_probe paths -> bp/prep
    str(CPC_ROUND / "code"),  # cpc_paths / cpc_common
    str(CC_ROUND / "code"),
    str(BS_ROUND / "code"),
    str(CODE_DIR),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sg_common as SC  # noqa: E402  (frozen recipe / schedule / identities)
import pc_metrics as MET  # noqa: E402
import cpc_paths as P  # noqa: E402  (S2 structure cache)
from pf_pipeline import PinnedStager, PageableTransfer  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
CONFIGS = ROUND / "configs"

ROUND_ID = "css-pcba-s0-v1"
STUDY_ID = ROUND_ID
PROTOCOL_ID = ROUND_ID

SCHEMA_RUN = "css-pcba-s0-v1-run-v1"
SCHEMA_CHECKPOINT = "css-pcba-s0-v1-checkpoint-v1"
SCHEMA_PREFLIGHT = "css-pcba-s0-v1-preflight-v1"
SCHEMA_CANONICAL_INIT = "css-pcba-s0-v1-canonical-init-v1"

DATASET = SC.DATASET
N_TASKS = SC.N_TASKS
N_MOLECULES = SC.N_MOLECULES
N_TRAIN = SC.N_TRAIN
N_VALID = SC.N_VALID
N_TEST = SC.N_TEST
BATCH_SIZE = SC.BATCH_SIZE
TOTAL_EPOCHS = SC.TOTAL_EPOCHS
STEPS_PER_EPOCH = SC.STEPS_PER_EPOCH
TOTAL_STEPS_40 = SC.TOTAL_STEPS_40
SEED = 0

#: Dedicated dropout RNG (module-level contract in PROTOCOL.md section 5).
DROPOUT_GENERATOR_SEED = 230_000 + SEED
#: Canonical initial state: ``torch.manual_seed(CANONICAL_INIT_SEED)`` before the
#: single construction of ``E``; all arms then *load* that state.
CANONICAL_INIT_SEED = 0

TASK_CHUNK_DEFAULT = 16

#: Frozen input widths (lipo_v1 PCBA cache).
FROZEN_INPUTS = {
    "atom_feat_dim": 51,
    "bond_feat_dim": 13,
    "atom_scalar_dim": 9,
    "global_dim": 42,
}

#: Frozen provenance identities of the inherited assets.
SOURCE = dict(SC.SOURCE)
FEATURE_SPEC_HASH = SOURCE["feature_spec_hash"]
NORM_FINGERPRINT = SOURCE["norm_fingerprint"]
SPLIT_PARTITION_SHA256 = SOURCE["split_partition_sha256"]
CACHE_SHA256 = SOURCE["cache_sha256"]
LABEL_ARTIFACT_SHA256 = SOURCE["label_artifact_sha256"]

#: Frozen seed-0 40-epoch order plan (``sg_data.make_plan40``).
PLAN = {
    "global_plan_sha256": "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
    "consumed40_prefix_sha256": "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58",
    "order_seed": 0,
    "batch_size": BATCH_SIZE,
    "steps_per_epoch": STEPS_PER_EPOCH,
}

RECIPE = dict(SC.SCRATCH_RECIPE)
scheduled_lr = SC.scheduled_lr
lr_audit = SC.lr_audit

# ---------------------------------------------------------------------------
# the three arms
# ---------------------------------------------------------------------------

EXPECTED_PARAMETERS = {"E": 831_088, "L": 831_088, "S": 829_040}

PARAMETER_BREAKDOWN = {
    "atom": 62_656,
    "bond": 250_432,
    "s1_event": 129_312,
    "s2_event": 270_944,
    "keys": 3_104,
    "values": 6_208,
    "root_static": 25_216,
    "root_stat": 4_096,
    "layer_norm": 128,
    "graph_mlp": 63_840,
    "task_output": 12_416,
    "query": 2_736,
}

ARMS: dict[str, dict[str, Any]] = {
    "E": {
        "name": "EARLY_TASK",
        "conditioning": "per_occurrence_then_aggregate",
        "task_query": True,
        "assigned_pool": "res-gpu0",
        "role": "primary_arm",
    },
    "L": {
        "name": "LATE_TASK",
        "conditioning": "aggregate_then_condition",
        "task_query": True,
        "assigned_pool": "res-gpu1",
        "role": "primary_arm",
    },
    "S": {
        "name": "EARLY_SHARED",
        "conditioning": "per_occurrence_then_aggregate",
        "task_query": False,
        "assigned_pool": None,
        "role": "conditional_arm_purchased_by_gate",
    },
}
ARM_ORDER = ("E", "L", "S")

#: Fixed evaluator / analysis constants.
SUPPORT_MIN_POSITIVE = 20
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 20261201
MECHANISM_VALID_ROWS = 4_096

#: Pre-registered S purchase rule (PROTOCOL.md section 8).
S_GATE = {
    "w2_best_val_reference": 0.265273,
    "d40_early_min": 0.002,
    "e_best_high_min": 0.268273,
    "e_best_floor": 0.262273,
    "cost_multiplier": 1.2,
    "gpu_soft_hours": 12.0,
    "gpu_hard_hours": 16.0,
}

#: Historical performance anchors (read-only, never retrained here).
HISTORICAL = {
    "W2": {
        "parameters": 994_324,
        "best_valid_ap": 0.26527331610028143,
        "best_state_test_ap": 0.253617,
        "epoch40_valid_ap": 0.261867,
        "epoch40_test_ap": 0.250986,
        "source": (
            "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/runs/W2/run.json "
            "+ lcrc_pcba_dynamic_v1/analysis/report_tables.md row 'W2 historical'"
        ),
    },
    "W2_DYNAMIC": {
        "parameters": 1_128_468,
        "best_valid_ap": 0.2750671855066329,
        "best_state_test_ap": 0.262226,
        "epoch40_valid_ap": 0.268791,
        "epoch40_test_ap": 0.260136,
        "source": (
            "tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/runs/W2_DYNAMIC/run.json "
            "+ analysis/report_tables.md row 'W2-DYNAMIC'"
        ),
    },
    "LCRC_CPC": {
        "parameters": 1_043_333,
        "best_valid_ap": 0.2659443284254145,
        "best_epoch": 36,
        "source": "tracks/structural-sufficiency/lcrc_pcba_cpc_v1/runs/LCRC_CPC/run.json",
    },
    "LCRC_RC_REAL": {
        "parameters": 1_051_022,
        "best_valid_ap": 0.2677177024413342,
        "best_epoch": 33,
        "source": "tracks/structural-sufficiency/lcrc_pcba_rc_v1/runs/REAL/run.json",
    },
}

TEST_ROLE = "DEVELOPMENT_TEST"

BUDGET = {
    "wall_soft_hours": 6.0,
    "wall_hard_hours": 10.0,
    "gpu_soft_hours": 12.0,
    "gpu_hard_hours": 16.0,
    "preflight_gpu_hours": 0.25,
}

#: Official Test label source (the *only* place this round reads Test targets).
OGB_GRAPH_LABEL = REPO / "data/ogb/ogbg_molpcba/raw/graph-label.csv.gz"

# ---------------------------------------------------------------------------
# IO helpers (thin wrappers over the inherited, already-audited helpers)
# ---------------------------------------------------------------------------

utcnow = SC.utcnow
write_json = SC.write_json
read_json = SC.read_json
sha256_file = SC.sha256_file
sha256_bytes = SC.sha256_bytes
sha256_int64 = SC.sha256_int64
sha256_batches = SC.sha256_batches
track_rel = SC.track_rel
log = SC.log
environment = SC.environment
seed_everything = SC.seed_everything
capture_rng_state = SC.capture_rng_state
rng_state_hashes = SC.rng_state_hashes
git_commit = SC.git_commit
_json_default = SC._json_default


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)
    return sha256_file(p)


def sha256_array(array: np.ndarray, dtype=np.float32) -> str:
    return sha256_bytes(np.ascontiguousarray(array, dtype=dtype).tobytes())


def state_dict_cpu(state: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    return {k: (v.detach().to("cpu").clone() if torch.is_tensor(v) else v) for k, v in state.items()}


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def state_fingerprint(state: Mapping[str, Any]) -> str:
    import torch

    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        h.update(name.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def git_state() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(REPO), capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover
            return None

    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def host_facts() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "git": git_state(),
    }
    if torch.cuda.is_available():
        out["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    out["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return out


def gpu_peak(device: Any) -> dict[str, int]:
    import torch

    if getattr(device, "type", str(device)) != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def recipe_for_arm(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise KeyError(arm)
    spec = ARMS[arm]
    recipe = dict(RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "arm": arm,
            "arm_name": spec["name"],
            "conditioning": spec["conditioning"],
            "task_query": bool(spec["task_query"]),
            "assigned_pool": spec["assigned_pool"],
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_cadence": 1,
            "eval_splits": ["train", "valid", "test"],
            "test_policy": "always (DEVELOPMENT_TEST, user-authorised every-epoch exposure)",
            "parameters": int(EXPECTED_PARAMETERS[arm]),
            "canonical_init_seed": int(CANONICAL_INIT_SEED),
            "dropout_generator_seed": int(DROPOUT_GENERATOR_SEED),
            "task_chunk": int(TASK_CHUNK_DEFAULT),
            "lr_schedule": lr_audit()["phase3"],
        }
    )
    return recipe


# ---------------------------------------------------------------------------
# resource ledger (JSONL -> JSON summary)
# ---------------------------------------------------------------------------


class Ledger:
    """Append-only stage ledger for one ``research run``."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, stage: str, **fields: Any) -> dict[str, Any]:
        entry = {"stage": str(stage), "at_utc": utcnow(), **fields}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=_json_default) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def summary(self) -> dict[str, Any]:
        entries = self.entries()
        gpu_seconds = 0.0
        for e in entries:
            try:
                gpu_seconds += float(e.get("gpu_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        return {
            "n_entries": len(entries),
            "gpu_seconds_total": gpu_seconds,
            "gpu_hours_total": gpu_seconds / 3600.0,
            "entries": entries,
        }


class stage_timer:
    """Context manager that records wall (+ GPU) seconds of one stage."""

    def __init__(self, ledger: Ledger, name: str, *, device: str = "cpu", purpose: str = ""):
        self.ledger = ledger
        self.name = name
        self.device = device
        self.purpose = purpose

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        seconds = time.perf_counter() - self.t0
        gpu_seconds = seconds if str(self.device).startswith("cuda") else 0.0
        self.ledger.record(
            self.name,
            purpose=self.purpose,
            device=str(self.device),
            seconds=float(seconds),
            gpu_seconds=float(gpu_seconds),
            ok=bool(exc_type is None),
            error=None if exc_type is None else f"{exc_type.__name__}: {exc}",
        )
        return False


__all__ = [name for name in dir() if not name.startswith("_")]
