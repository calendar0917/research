"""Frozen definitions for DSCR v0 (direct structural computation and recurrence).

Round: ``direct-structural-computation-recurrence-v0``

The round compares five parameter-matched structural computations on top of the
frozen historical LCRC-W2 base graph:

* ``DYNAMIC``  -- the historical two-layer real-bond message passing that reads
  the *updated* neighbour state ``h1`` (reused verbatim, never redesigned);
* ``SELF``     -- same module/incoming-bond conditioning, but the source state is
  the target node's own state: no other node's hidden state is ever read;
* ``FIXED1``   -- layer 2 reads only the *initial* neighbour states ``h0``;
* ``REAL2``    -- direct two-hop rooted-branch computation from ``h0``;
* ``BAG2``     -- REAL2 with the real outer-branch/branch-context pairing removed
  (multiset cross-product control).

Frozen, read-only inheritance (never modified, never re-defaulted):

* ``lcrc_pcba_backbone_scaling_v1/code/bs_model.py`` / ``bs_common.py`` -- the
  W2 model (994,324 parameters), the official seed-0 constructor, the frozen
  from-scratch recipe (AdamW, 1-epoch linear warmup 3e-5 -> 3e-4, per-step
  cosine to 1e-5 over 109,520 updates, dropout 0.10, clip 5.0), the data
  identities and the official evaluator;
* ``lcrc_pcba_dynamic_v1/code/dy_model.py`` / ``dy_common.py`` -- the historical
  ``W2DynamicModel``, ``MessageFFN`` wiring, the private-generator
  new-module initialisation rule, the parameter audit and ``usage_metrics``;
* ``lcrc_pcba_structural_gain_sprint_v1/code/sg_model.py`` -- ``MessageFFN``;
* ``lcrc_pcba_structural_gain_sprint_v1/code/sg_data.py`` -- task loading, the
  frozen 40-epoch order plans, split evaluation helpers;
* ``lcrc_pcba_continuation_recipe_v1/code/cc_data.py`` -- the raw OGB label file
  used to score the (already exposed) official Test split.

Test policy of this round: ``DEVELOPMENT_TEST / EXPOSED_TEST_ENGINEERING_COMPARISON_SET``.
Every epoch scores the full official Test split, but no Test number is ever an
input to checkpoint selection, horizon choice or any configuration decision.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
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

SS = REPO / "tracks/structural-sufficiency"
BS_ROUND = SS / "lcrc_pcba_backbone_scaling_v1"
SG_ROUND = SS / "lcrc_pcba_structural_gain_sprint_v1"
DY_ROUND = SS / "lcrc_pcba_dynamic_v1"
CC_ROUND = SS / "lcrc_pcba_continuation_recipe_v1"
RC_ROUND = SS / "lcrc_pcba_rc_v1"
S3_ROUND = SS / "lcrc_pcba_s3_v1"
CPC_ROUND = SS / "lcrc_pcba_cpc_v1"
BOC_ROUND = SS / "lcrc_pcba_boc_v1"

for _p in (
    str(BS_ROUND / "code"),
    str(SG_ROUND / "code"),
    str(DY_ROUND / "code"),
    str(CC_ROUND / "code"),
    str(CODE_DIR),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bs_common as BC  # noqa: E402  (W2 widths, recipe, data identity, IO helpers)
import bs_model as BM  # noqa: E402  (frozen W2 model / constructor)
import sg_data as SD  # noqa: E402  (task, order plan, evaluation helpers)
import dy_common as DC  # noqa: E402  (DYNAMIC definitions, parameter audit helpers)
import dy_model as DM  # noqa: E402  (W2DynamicModel + private init)
import pc_metrics as MET  # noqa: E402  (masked BCE + official OGB AP)
import pc_model as PC  # noqa: E402  (collate / predict_logits)

ANALYSIS = ROUND / "analysis"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "runs"
RESULTS = ROUND / "results"
TESTS = ROUND / "tests"

ROUND_ID = "direct-structural-computation-recurrence-v0"
PROTOCOL_ID = "dscr-v0"
SCHEMA_RUN = "dscr-run-v1"
SCHEMA_CHECKPOINT = "dscr-checkpoint-v1"
SCHEMA_CHECKS = "dscr-checks-v1"
SCHEMA_PREFLIGHT = "dscr-preflight-v1"
SCHEMA_MECHANISM = "dscr-mechanism-v1"
SCHEMA_SCHEDULE = "dscr-schedule-v1"
SCHEMA_SUMMARY = "dscr-summary-v1"

TEST_ROLE = "DEVELOPMENT_TEST / EXPOSED_TEST_ENGINEERING_COMPARISON_SET"

# ---------------------------------------------------------------------------
# data / recipe (inherited, never re-derived)
# ---------------------------------------------------------------------------

DATASET = BC.DATASET
N_TASKS = BC.N_TASKS
N_MOLECULES = BC.N_MOLECULES
N_TRAIN = BC.N_TRAIN
N_VALID = BC.N_VALID
N_TEST = BC.N_TEST
BATCH_SIZE = BC.BATCH_SIZE
STEPS_PER_EPOCH = BC.STEPS_PER_EPOCH
TOTAL_EPOCHS = BC.TOTAL_EPOCHS
TOTAL_STEPS_40 = BC.TOTAL_STEPS_40

RECIPE = dict(BC.RECIPE)
SOURCE = dict(BC.SOURCE)
scheduled_lr = BC.scheduled_lr
set_lrs = BM.set_lrs
state_fingerprint = BC.state_fingerprint
utcnow = BC.utcnow
write_json = BC.write_json
read_json = BC.read_json
sha256_file = BC.sha256_file
sha256_bytes = BC.sha256_bytes
log = BC.log
environment = BC.environment
seed_everything = BC.seed_everything
capture_rng_state = BC.capture_rng_state
rng_state_hashes = BC.rng_state_hashes
git_commit = BC.git_commit
git_state = BC.git_state
track_rel = BC.track_rel
host_facts = BC.host_facts
gpu_occupancy = BC.gpu_occupancy
state_cpu = BM.state_cpu
optimizer_state_cpu = BM.optimizer_state_cpu
restore_rng_state = BM.restore_rng_state
permute_atoms = BM.permute_atoms
is_new_parameter = DC.is_new_parameter
new_parameter_names = DC.is_new_parameter
state_subset = DC.state_subset
trunk_subset = DC.trunk_subset
parameter_audit = DM.parameter_audit
initialize_dynamic_modules = DM.initialize_dynamic_modules
compare_states = DM.compare_states

#: Horizons allowed by the protocol (all of them use the *prefix* of the frozen
#: 40-epoch cosine schedule; the schedule is never recompressed).
HORIZONS = (40, 30, 25)
DEFAULT_HORIZON = 25
SEEDS = (0, 1)

#: The frozen W2 seed-0 40-epoch order plan (shared by every seed-0 arm).
FROZEN_PLAN_SEED0 = {
    "global_plan_sha256": "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
    "consumed40_prefix_sha256": "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58",
    "order_seed": 0,
}
#: Seed-1 plan is generated by the same rule and frozen in ``configs/plan_seed1.json``
#: by the preflight gate before any seed-1 formal run.
SEED1_PLAN_FILE = CONFIGS / "plan_seed1.json"

#: Frozen parameter budget of every new arm (identical for all five).
EXPECTED_PARAMETERS = dict(DC.EXPECTED_PARAMETERS)
PARAMS_TOTAL = 1_128_468
PARAMS_W2 = 994_324

#: Two-hop branch context strength (fixed by the protocol; never searched).
GAMMA = 0.5

#: Historical seed-0 references (read-only; never retrained here).
W2_SEED0 = dict(DC.W2_SEED0)
RC_REAL = dict(DC.RC_REAL)
BOC_TEST_ROWS_NPY_SHA256 = DC.BOC_TEST_ROWS_NPY_SHA256

ARMS: dict[str, dict[str, Any]] = {
    "W2": {
        "role": "original non-MP performance baseline",
        "params": PARAMS_W2,
        "new_modules": False,
        "access": "no structural computation after h0",
        "class": "strict_direct_local",
    },
    "SELF": {
        "role": "pointwise depth / capacity control",
        "params": PARAMS_TOTAL,
        "new_modules": True,
        "access": "no other node's hidden state is read",
        "class": "strict_direct_local",
    },
    "FIXED1": {
        "role": "fixed-source single-hop structural computation",
        "params": PARAMS_TOTAL,
        "new_modules": True,
        "access": "h0 of the real 1-hop neighbours (never h1[j])",
        "class": "fixed_source_one_hop",
    },
    "REAL2": {
        "role": "direct two-hop rooted branch computation",
        "params": PARAMS_TOTAL,
        "new_modules": True,
        "access": "h0 of the real 2-hop rooted branch (never h1[j])",
        "class": "fixed_source_two_hop",
    },
    "BAG2": {
        "role": "branch-correspondence control (multiset cross product)",
        "params": PARAMS_TOTAL,
        "new_modules": True,
        "access": "h0 of the real 2-hop rooted branch (never h1[j]), pairing removed",
        "class": "fixed_source_two_hop_unbound",
    },
    "DYNAMIC": {
        "role": "iterative neighbour-state boundary reference",
        "params": PARAMS_TOTAL,
        "new_modules": True,
        "access": "h1 of the real 1-hop neighbours in layer 2",
        "class": "iterative_hidden_state",
    },
}
ARM_ORDER = ("W2", "SELF", "FIXED1", "REAL2", "BAG2", "DYNAMIC")
DEEP_ARMS = ("SELF", "FIXED1", "REAL2", "BAG2", "DYNAMIC")
CANDIDATES = ("SELF", "FIXED1", "REAL2", "BAG2")
#: Fixed mapping C* -> K* (mechanism control used for the second seed).
CONTROL_OF = {"SELF": "FIXED1", "FIXED1": "SELF", "REAL2": "BAG2", "BAG2": "REAL2"}
#: Engineering tie-break order when Valid AP values are within 0.001 AP.
CANDIDATE_PREFERENCE = ("SELF", "FIXED1", "REAL2", "BAG2")
TIE_BREAK_AP = 0.001

SUPPORT_MIN = 20

# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

EVAL_BATCH = {"train": 1024, "valid": 512, "test": 512}
#: Batch for the fixed small diagnostic batch used by the mechanism metrics.
DIAG_VALID_ROWS = 32


def _array_sha(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def evaluate_split(
    model: Any,
    task: Any,
    device: Any,
    split: str,
    *,
    y: np.ndarray | None = None,
    batch_size: int | None = None,
    per_task: bool = True,
    keep_logits: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    """Full-split eval-mode scoring (official OGB macro AP is the primary metric)."""
    if split == "train":
        rows = task.train_rows
        y = np.asarray(task.labels[rows]) if y is None else y
    elif split == "valid":
        rows = task.valid_rows
        y = np.asarray(task.labels[rows]) if y is None else y
    elif split == "test":
        rows = task.test_rows
        if y is None:
            raise RuntimeError("Test scoring requires the officially exposed label matrix")
    else:  # pragma: no cover - defensive
        raise KeyError(split)
    bs = int(EVAL_BATCH[split] if batch_size is None else batch_size)
    t0 = time.perf_counter()
    model.eval()
    logits = PC.predict_logits(model, task.cache, rows, device, batch_size=bs)
    infer_seconds = float(time.perf_counter() - t0)
    probs = PC.sigmoid(logits)
    t1 = time.perf_counter()
    if per_task:
        report = MET.validation_report(y, probs)
        ap = float(report["ap"])
        per_task_ap = report["per_task_ap"]
        observed = report["observed"]
        positive = report["positive"]
        negative = report["negative"]
        n_scored = int(report["n_tasks_scored"])
    else:
        official = MET.official_ap(y, probs)
        ap = float(official["value"])
        per_task_ap = None
        observed = positive = negative = None
        n_scored = -1
    bce = float(MET.masked_bce_numpy(logits, y))
    metric_seconds = float(time.perf_counter() - t1)
    out: dict[str, Any] = {
        "split": split,
        "ap": ap,
        "bce": bce,
        "ap_source": "OGB official Evaluator(ogbg-molpcba)",
        "n_tasks_scored": n_scored,
        "evaluated_rows": int(len(rows)),
        "infer_seconds": infer_seconds,
        "metric_seconds": metric_seconds,
        "seconds": float(time.perf_counter() - t0),
        "logits_sha256": _array_sha(logits),
        "probs_sha256": _array_sha(probs),
        "reason": reason,
    }
    if per_task_ap is not None:
        out.update(
            {
                "per_task_ap": per_task_ap,
                "observed": observed,
                "positive": positive,
                "negative": negative,
                "n_tasks_with_observed_labels": int(sum(1 for x in observed if x > 0)),
            }
        )
    if keep_logits:
        out["logits"] = logits
    return out


# ---------------------------------------------------------------------------
# official Test labels (exposed development set; alignment-gated)
# ---------------------------------------------------------------------------

OGB_GRAPH_LABEL = CC_ROUND / ".." / ".." / ".." / "data/ogb/ogbg_molpcba/raw/graph-label.csv.gz"
OGB_GRAPH_LABEL = (REPO / "data/ogb/ogbg_molpcba/raw/graph-label.csv.gz").resolve()


def load_test_labels(task: Any, *, cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Read the official Test targets once, after proving Train/Valid alignment.

    The frozen Train/Valid label artifact is reproduced *exactly* from the raw
    OGB file before a single Test row is exposed, and the official Test row ids
    are checked against the BOC-round identity.  The result is cached as a
    ``.npy`` file so that eight training runs do not re-parse the raw CSV.
    """
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    rows_sha = BC.SC.sha256_int64(test_rows.tolist())
    if rows_sha != BOC_TEST_ROWS_NPY_SHA256:
        raise RuntimeError(f"official Test row ids {rows_sha} != frozen {BOC_TEST_ROWS_NPY_SHA256}")
    raw_sha = sha256_file(OGB_GRAPH_LABEL)
    cache_path = None
    meta_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"test_labels_{raw_sha[:16]}_{rows_sha[:16]}.npy"
        meta_path = Path(cache_dir) / "test_labels_manifest.json"
    if cache_path is not None and cache_path.is_file():
        y = np.load(cache_path)
        if y.shape != (N_TEST, N_TASKS):
            raise RuntimeError(f"cached Test label matrix {y.shape}")
    else:
        with gzip.open(OGB_GRAPH_LABEL, "rt", newline="") as fh:
            raw = [[float("nan") if v == "" else float(v) for v in row] for row in csv.reader(fh)]
        labels = np.asarray(raw, dtype=np.float32)
        if labels.shape != (N_MOLECULES, N_TASKS):
            raise RuntimeError(f"official label matrix {labels.shape}")
        train = np.asarray(task.train_rows, dtype=np.int64)
        valid = np.asarray(task.valid_rows, dtype=np.int64)

        def _same(a: np.ndarray, b: np.ndarray) -> bool:
            return bool(np.array_equal(np.nan_to_num(a, nan=-9.0), np.nan_to_num(b, nan=-9.0)))

        if not _same(labels[train], task.labels[train]):
            raise RuntimeError("raw OGB Train labels do not match the frozen label artifact")
        if not _same(labels[valid], task.labels[valid]):
            raise RuntimeError("raw OGB Valid labels do not match the frozen label artifact")
        y = np.ascontiguousarray(labels[test_rows])
        finite = np.isfinite(y)
        if not np.isin(y[finite], (0.0, 1.0)).all():
            raise RuntimeError("official Test labels are not binary 0/1")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_name(cache_path.stem + f".{os.getpid()}.tmp.npy")
            np.save(tmp, y)
            os.replace(tmp, cache_path)
    finite = np.isfinite(y)
    out = {
        "labels": y,
        "path": str(OGB_GRAPH_LABEL),
        "raw_sha256": raw_sha,
        "test_rows_sha256": rows_sha,
        "shape": list(y.shape),
        "observed_entries": int(finite.sum()),
        "positive_entries": int((y[finite] == 1.0).sum()),
        "negative_entries": int((y[finite] == 0.0).sum()),
        "train_valid_alignment_verified": True,
        "cache_path": None if cache_path is None else track_rel(cache_path),
        "test_role": TEST_ROLE,
    }
    if meta_path is not None:
        write_json(
            meta_path,
            {
                "schema": "dscr-test-label-read-v1",
                "test_role": TEST_ROLE,
                "raw_label_file": out["path"],
                "raw_label_sha256": raw_sha,
                "test_rows_sha256": rows_sha,
                "observed_entries": out["observed_entries"],
                "positive_entries": out["positive_entries"],
                "negative_entries": out["negative_entries"],
                "read_utc": utcnow(),
                "note": (
                    "Test is an open engineering comparison set in this round; every epoch is "
                    "scored, and no Test number enters checkpoint/horizon/configuration choice"
                ),
            },
        )
    return out


# ---------------------------------------------------------------------------
# order plans
# ---------------------------------------------------------------------------


def plan_for_seed(task: Any, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The frozen 40-epoch order plan of a seed group, with its identity."""
    plan = SD.make_plan40(task, seed=int(seed))
    summary = SD.plan_summary(plan, task)
    got = plan[0]["global_plan_sha256"]
    if int(seed) == 0:
        if got != FROZEN_PLAN_SEED0["global_plan_sha256"]:
            raise RuntimeError(f"order plan {got} != frozen W2 seed-0 plan")
    else:
        if not SEED1_PLAN_FILE.is_file():
            raise RuntimeError(f"seed-{seed} plan has not been frozen ({SEED1_PLAN_FILE})")
        frozen = read_json(SEED1_PLAN_FILE)
        if got != frozen["global_plan_sha256"]:
            raise RuntimeError(f"order plan {got} != frozen seed-{seed} plan")
    return plan, {
        "order_seed": int(seed),
        "global_plan_sha256": got,
        "plan_summary": summary["summary"],
        "plan_isolation": summary["isolation"],
    }


def freeze_seed1_plan(task: Any) -> dict[str, Any]:
    """Generate and freeze the seed-1 order plan (preflight gate)."""
    plan = SD.make_plan40(task, seed=1)
    summary = SD.plan_summary(plan, task)
    payload = {
        "schema": "dscr-plan-freeze-v1",
        "order_seed": 1,
        "batch_size": int(BATCH_SIZE),
        "steps_per_epoch": int(STEPS_PER_EPOCH),
        "epochs": int(len(plan)),
        "global_plan_sha256": plan[0]["global_plan_sha256"],
        "consumed40_prefix_sha256": sha256_bytes(
            "|".join(p["row_order_sha256"] for p in plan).encode()
        ),
        "per_epoch_row_order_sha256": [p["row_order_sha256"] for p in plan],
        "plan_summary": summary["summary"],
        "plan_isolation": summary["isolation"],
        "generation_rule": (
            "lr_data.make_order_plan(train_rows, batch_size=128, seed=1, epochs=40): one CPU "
            "torch.Generator seeded with the order seed, one randperm per epoch"
        ),
        "frozen_utc": utcnow(),
    }
    write_json(SEED1_PLAN_FILE, payload)
    return payload


def plan_meta(plan: Sequence[Mapping[str, Any]], consumed: int, order_seed: int) -> dict[str, Any]:
    prefix = [p["row_order_sha256"] for p in plan[: int(consumed)]]
    return {
        "global_plan_sha256": plan[0]["global_plan_sha256"],
        "order_seed": int(order_seed),
        "batch_size": int(BATCH_SIZE),
        "epochs": int(len(plan)),
        "consumed_epochs": int(consumed),
        "prefix_sha256": sha256_bytes("|".join(prefix).encode()),
    }


# ---------------------------------------------------------------------------
# reference coordinates (historical, never retrained here)
# ---------------------------------------------------------------------------

HISTORICAL = {
    "W2": {
        "parameters": 994_324,
        "best_epoch": 24,
        "best_valid_ap": 0.26527331610028143,
        "test_ap_best": 0.2536164787414065,
        "valid_ap_at_40": 0.26186691813350904,
        "run_dir": BS_ROUND / "runs/W2",
        "seed": 0,
        "protocol": "lcrc-pcba-backbone-scaling-v1 (same scratch recipe, same plan, seed 0)",
    },
    "DYNAMIC": {
        "parameters": 1_128_468,
        "best_epoch": 25,
        "best_valid_ap": 0.275067,
        "test_ap_best": 0.262226,
        "valid_ap_at_40": 0.268791,
        "run_dir": DY_ROUND / "runs/W2_DYNAMIC",
        "seed": 0,
        "protocol": "lcrc-pcba-dynamic-v1 (same scratch recipe, same plan, seed 0)",
    },
}
REFERENCE_OTHER = {
    "W3": {"parameters": 2_120_844, "best_valid_ap": 0.263670, "test_ap_best": 0.251822},
    "S3": {"parameters": 1_018_532, "best_valid_ap": 0.271489, "test_ap_best": 0.250633},
    "RC_REAL": {"parameters": 1_051_022, "best_valid_ap": 0.267718, "test_ap_best": 0.255941},
    "CPC": {"parameters": 1_043_333, "best_valid_ap": 0.265944, "test_ap_best": 0.262356},
}


def recipe(arm: str, seed: int, horizon: int) -> dict[str, Any]:
    if arm not in ARMS:
        raise KeyError(arm)
    out = dict(RECIPE)
    out.update(
        {
            "round": ROUND_ID,
            "arm": arm,
            "arm_role": ARMS[arm]["role"],
            "access_class": ARMS[arm]["class"],
            "seed": int(seed),
            "horizon": int(horizon),
            "prefix_of_frozen_schedule": True,
            "widths": dict(BC.config_for_scale(2)),
            "gamma": float(GAMMA),
            "parameters": int(ARMS[arm]["params"]),
            "eval_cadence": "every epoch: full Train, full Valid, full official Test",
            "selection": "highest full official Valid macro AP within the common horizon, ties -> earlier epoch",
            "test_role": TEST_ROLE,
            "test_used_for_selection": False,
        }
    )
    return out


def split_analysis_dir() -> Path:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    return ANALYSIS


def run_dir(arm: str, seed: int) -> Path:
    return RUNS / f"{arm}_seed{int(seed)}"


__all__ = [
    "ANALYSIS",
    "ARM_ORDER",
    "ARMS",
    "BATCH_SIZE",
    "CANDIDATES",
    "CANDIDATE_PREFERENCE",
    "CONFIGS",
    "CONTROL_OF",
    "DATASET",
    "DEEP_ARMS",
    "DIAG_VALID_ROWS",
    "EVAL_BATCH",
    "EXPECTED_PARAMETERS",
    "FROZEN_PLAN_SEED0",
    "GAMMA",
    "HISTORICAL",
    "HORIZONS",
    "N_MOLECULES",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "OGB_GRAPH_LABEL",
    "PARAMS_TOTAL",
    "PARAMS_W2",
    "PROTOCOL_ID",
    "RECIPE",
    "REFERENCE_OTHER",
    "RESULTS",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCHEMA_CHECKS",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_MECHANISM",
    "SCHEMA_PREFLIGHT",
    "SCHEMA_RUN",
    "SCHEMA_SCHEDULE",
    "SCHEMA_SUMMARY",
    "SEEDS",
    "SEED1_PLAN_FILE",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "SUPPORT_MIN",
    "TEST_ROLE",
    "TESTS",
    "TIE_BREAK_AP",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "W2_SEED0",
    "capture_rng_state",
    "compare_states",
    "environment",
    "evaluate_split",
    "freeze_seed1_plan",
    "git_commit",
    "git_state",
    "gpu_occupancy",
    "host_facts",
    "initialize_dynamic_modules",
    "is_new_parameter",
    "load_test_labels",
    "log",
    "parameter_audit",
    "permute_atoms",
    "plan_for_seed",
    "plan_meta",
    "read_json",
    "recipe",
    "restore_rng_state",
    "rng_state_hashes",
    "run_dir",
    "scheduled_lr",
    "seed_everything",
    "set_lrs",
    "sha256_bytes",
    "sha256_file",
    "state_cpu",
    "state_fingerprint",
    "state_subset",
    "track_rel",
    "trunk_subset",
    "utcnow",
    "write_json",
]
