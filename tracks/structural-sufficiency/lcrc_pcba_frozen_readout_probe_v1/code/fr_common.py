"""Frozen definitions for the LCRC PCBA frozen-readout probe.

Round: ``lcrc-pcba-frozen-readout-probe-v1``

This round performs **one** frozen feature extraction and **one** short paired
comparison of two tiny residual heads.  The LCRC-C1 encoder is never trained,
never unfrozen and never re-initialised; its weights are loaded from the frozen
``lcrc_pcba_lr_probe_v1`` L-low-lr epoch-9 best checkpoint and every parameter
is placed behind ``requires_grad_(False)`` before the first forward pass.

Definitions frozen here (before any GPU run):

* the source checkpoint and its SHA-256,
* ``AP0`` (historical Validation AP) and ``TRAIN_AP0`` (the frozen epoch-9
  Train macro AP already reported by ``lcrc_pcba_loss_screen_v1``),
* the residual-head recipe (shared by both arms, only the architecture differs),
* the evaluation epochs ``0, 2, 4, 8`` and the epoch-8 primary comparison,
* the decision thresholds and the resource budget.

The official Test targets are unreachable: the inherited label artifact has
every Test row set to NaN, ``lr_data.load_task`` refuses a finite Test label,
and nothing in this round indexes ``task.split['test']``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# paths / inherited code
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PILOT_CODE = PILOT / "code"
PERF = REPO / "tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1"
PERF_CODE = PERF / "code"
LR_PROBE = REPO / "tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1"
LR_PROBE_CODE = LR_PROBE / "code"
LOSS_SCREEN = REPO / "tracks/structural-sufficiency/lcrc_pcba_loss_screen_v1"
LOSS_SCREEN_CODE = LOSS_SCREEN / "code"
for _p in (str(PILOT_CODE), str(PERF_CODE), str(LR_PROBE_CODE), str(LOSS_SCREEN_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lr_common as LR  # noqa: E402  (also puts bp/prep/tuq/mb2 code on sys.path)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
FEATURE_CACHE = RUNS / "feature_cache"
HEADS_RUNS = RUNS / "heads"
PREFLIGHT_RUN = RUNS / "preflight"

ROUND_ID = "lcrc-pcba-frozen-readout-probe-v1"
PROTOCOL_ID = ROUND_ID

DATASET = "ogbg-molpcba"
N_TASKS = 128
N_MOLECULES = 437_929
N_TRAIN = 350_343
N_VALID = 43_793
N_TEST = 43_793
SEED = 0

#: Frozen source checkpoint.  Never copied, never rewritten, never re-trained.
SOURCE: dict[str, Any] = {
    "round": "lcrc-pcba-lr-probe-v1",
    "group": "L",
    "group_name": "L_low_lr",
    "checkpoint": LR_PROBE / "runs" / "L_low_lr" / "checkpoint_best.pt",
    "checkpoint_sha256": "b158f1669f6d0ccbf5fc05f510050c69ef2fba9f6258cbe59d4d3d9a938eb6c5",
    "best_epoch": 9,
    "best_valid_ap": 0.2374902968831333,
    "feature_spec_hash": "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee",
    "norm_fingerprint": "6ce50f989fc0789d9573f7f787a44a49a282531ac5a2db09c6600c67dc3293fb",
    "split_partition_sha256": "c872bd6311796826db9365c5c739c4cee94c3baa78fbd18006616feefc67bf4f",
    "label_artifact_sha256": "5c3b5fbe642d80113522832fe01090097b860010868c59f8696bc247d5027d01",
    "cache_sha256": "caad3e90dd76d1f1483d2a01501eba23413072a7b5a2fc4b62780c315aa59035",
    "n_parameters": 245_184,
}

#: Frozen reference values.  ``AP0`` is the decision anchor fixed by the round
#: instruction; ``TRAIN_AP0`` is the epoch-9 Train macro AP measured by
#: ``lcrc_pcba_loss_screen_v1`` (mean of the 128 per-task AP values).
AP0 = 0.2374902968831333
TRAIN_AP0 = 0.38833559949014146
VALID_AP_TOL = 1e-4
TRAIN_AP_TOL = 1e-4
N_SCORABLE_TASKS_EXPECTED = 127

#: ``rho_in`` column layout of the frozen model: [g | mean_h | mean_h2 | log1p(n)].
#: Widths are asserted against ``tu_model.rho_layout(spec, width)`` and against
#: ``model.rho.lins[0].weight.shape[1]`` at run time; nothing is hard-coded into
#: the model itself.
RHO_LAYOUT: dict[str, int] = {"g": 42, "mean_h": 64, "mean_h2": 64, "log1p_n": 1}
RHO_IN_DIM_EXPECTED = sum(RHO_LAYOUT.values())  # 171

# ---------------------------------------------------------------------------
# residual heads (the only trainable objects in this round)
# ---------------------------------------------------------------------------

BATCH_SIZE = 1024
EPOCHS = 8
EVAL_EPOCHS: tuple[int, ...] = (0, 2, 4, 8)
STEPS_PER_EPOCH = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE  # 343
MAIN_EPOCH = 8

ARMS: dict[str, dict[str, Any]] = {
    "L": {
        "name": "L_linear_residual",
        "role": "linear_residual_head",
        "architecture": "Linear(d, 128)",
        "assigned_pool": "res-gpu1",
        "trainable_parameters": 171 * N_TASKS + N_TASKS,
    },
    "N": {
        "name": "N_nonlinear_residual",
        "role": "nonlinear_residual_head",
        "architecture": "Linear(d, 128) -> SiLU -> Linear(128, 128)",
        "assigned_pool": "res-gpu1",
        "trainable_parameters": 171 * N_TASKS + N_TASKS + N_TASKS * N_TASKS + N_TASKS,
    },
}
ARM_ORDER = ("L", "N")

HEAD_RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "learning_rate": 3.0e-4,
    "weight_decay": 1.0e-3,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "loss": "masked BCEWithLogitsLoss, every observed label entry equally weighted",
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "eval_epochs": list(EVAL_EPOCHS),
    "scheduler": "none",
    "early_stopping": False,
    "precision": "fp32",
    "seed": SEED,
    "row_order": "torch.Generator().manual_seed(0) + torch.randperm(n_train) per epoch; identical for both arms",
    "initialization": {
        "L": "Linear(d,128) weight=0, bias=0",
        "N": "first Linear = PyTorch default nn.Linear init under torch.manual_seed(0); "
        "final Linear weight=0, bias=0",
    },
    "base_model_gradients": "forbidden; the frozen encoder is not part of any optimizer",
    "selection": "fixed epoch 8 is the primary comparison; best Validation AP is auxiliary only",
    "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
}

#: One-shot extraction settings.  Only Train/Validation rows are encoded.
EXTRACT: dict[str, Any] = {
    "batch_size": 512,
    "standardizer": "per-column mean/std of Train rho_in only, population std (ddof=0)",
    "zero_variance_policy": "columns with std <= 1e-12 get scale 1.0 (no division blow-up)",
    "valid_rows_used_for_fit": 0,
    "inference_mode": True,
    "model_eval": True,
}

#: Resource guards.  Total budget is for the whole round (smoke + extraction +
#: both heads).  It is not topped up if a phase overruns.
BUDGET: dict[str, Any] = {
    "gpu_hours_total": 0.4,
    "preflight_soft_seconds": 300.0,
    "extract_soft_seconds": 600.0,
    "heads_soft_seconds": 900.0,
    "outer_kill_seconds": 1800.0,
}

#: Decision rule, fixed before the first formal run.
DECISION: dict[str, Any] = {
    "ap0": AP0,
    "improvement_floor": 0.005,
    "task_majority": 64,
    "n_scorable_tasks_expected": N_SCORABLE_TASKS_EXPECTED,
    "nonlinear_extra_floor": 0.003,
    "main_epoch": MAIN_EPOCH,
    "main_metric": "AP_L/AP_N at fixed epoch 8 (never the best-Validation checkpoint)",
    "rules": {
        "READOUT_SIGNAL": (
            "both arms complete; at least one head has "
            "D = AP_head(epoch8) - AP0 >= +0.005 AND >= 64 of 127 scorable tasks improved"
        ),
        "NONLINEAR_EXTRA_SIGNAL": (
            "READOUT_SIGNAL holds AND D_NL = AP_N(epoch8) - AP_L(epoch8) >= +0.003"
        ),
        "WEAK_OR_MIXED": (
            "both arms complete; max(D_L, D_N) > 0 but READOUT_SIGNAL does not hold"
        ),
        "NO_FAST_READOUT_SIGNAL": (
            "both arms complete; D_L <= 0 and D_N <= 0 (no directional readout gain)"
        ),
        "INCOMPLETE": (
            "a correctness/checkpoint/data-isolation gate failed, an arm did not "
            "complete the fixed 8 epochs and all four evaluation epochs, a formal "
            "result is missing, or the GPU budget was exhausted"
        ),
    },
    "interpretation_guardrails": [
        "A failed readout probe does not show that the S1/S2 representation is invalid.",
        "A residual-head gain does not identify a specific mechanism defect in the frozen model.",
        "Single-seed differences are resource-decision evidence, not statistical significance.",
        "Validation labels have already been used for model selection in earlier rounds; "
        "they are not an independent final confirmation set.",
    ],
    "statistical_claim": "none; these are resource-decision labels only",
}

RESULT_LABELS = (
    "READOUT_SIGNAL",
    "WEAK_OR_MIXED",
    "NO_FAST_READOUT_SIGNAL",
    "INCOMPLETE",
)

# ---------------------------------------------------------------------------
# IO / hashing helpers (thin aliases over the inherited, already-audited helpers)
# ---------------------------------------------------------------------------

write_json = LR.write_json
read_json = LR.read_json
sha256_file = LR.sha256_file
sha256_bytes = LR.sha256_bytes
sha256_int64 = LR.sha256_int64
sha256_batches = LR.sha256_batches
track_rel = LR.track_rel
log = LR.log
utcnow = LR.utcnow
environment = LR.environment
seed_everything = LR.seed_everything
git_commit = LR.git_commit
git_dirty = LR.git_dirty
json_default = LR._json_default


def sha256_array(arr: np.ndarray) -> str:
    """SHA-256 of dtype + shape + contiguous bytes (content fingerprint)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def gpu_occupancy() -> dict[str, Any]:
    """Visible free memory and co-tenant processes (execution fact, not a knob)."""
    import torch

    out: dict[str, Any] = {"captured_utc": utcnow()}
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            out["visible_device_mem_get_info"] = {
                "free_bytes": int(free),
                "total_bytes": int(total),
                "used_bytes": int(total - free),
            }
        except Exception as exc:  # pragma: no cover - diagnostic only
            out["visible_device_mem_get_info_error"] = f"{type(exc).__name__}: {exc}"
    import subprocess

    try:
        out["nvidia_smi_compute_apps"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        out["nvidia_smi_compute_apps"] = f"unavailable: {type(exc).__name__}: {exc}"
    return out


def gpu_peak(device: Any) -> dict[str, int]:
    import torch

    if getattr(device, "type", str(device)) != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


# ---------------------------------------------------------------------------
# inherited-constant gates
# ---------------------------------------------------------------------------


def assert_inherited_constants() -> dict[str, Any]:
    """Fail hard if the frozen model/recipe/checkpoint identity drifted."""
    inherited = LR.assert_inherited_constants()
    import ls_common as LS

    mismatches: dict[str, Any] = {}
    for key, expected in SOURCE.items():
        found = LS.SOURCE.get(key)
        if found != expected:
            mismatches[key] = {"expected": str(expected), "found": str(found)}
    if mismatches:
        raise RuntimeError(f"frozen source constants drifted from the loss screen: {mismatches}")
    if int(LS.N_TRAIN) != N_TRAIN or int(LS.N_VALID) != N_VALID or int(LS.N_TEST) != N_TEST:
        raise RuntimeError("split size constants drifted")
    expected_arms = {"L": 171 * 128 + 128, "N": 171 * 128 + 128 + 128 * 128 + 128}
    for arm, n_params in expected_arms.items():
        if int(ARMS[arm]["trainable_parameters"]) != n_params:
            raise RuntimeError(f"arm {arm} trainable parameter count drifted")
    return {
        "inherited": inherited,
        "source_identity_matches_loss_screen": True,
        "source": {k: (track_rel(v) if isinstance(v, Path) else v) for k, v in SOURCE.items()},
        "ap0": AP0,
        "train_ap0": TRAIN_AP0,
        "rho_in_dim_expected": RHO_IN_DIM_EXPECTED,
        "arms": {k: dict(v) for k, v in ARMS.items()},
    }


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------


def decide_label(
    *,
    status_l: str,
    status_n: str,
    gates_ok: bool,
    d_l: float | None,
    d_n: float | None,
    d_nl: float | None,
    improved_l: int | None,
    improved_n: int | None,
    n_scorable: int | None,
) -> dict[str, Any]:
    """Apply the pre-registered resource-decision rule verbatim."""
    floor = float(DECISION["improvement_floor"])
    majority = int(DECISION["task_majority"])
    n_expected = int(DECISION["n_scorable_tasks_expected"])
    complete = (
        gates_ok
        and status_l == "COMPLETED"
        and status_n == "COMPLETED"
        and d_l is not None
        and d_n is not None
        and d_nl is not None
        and improved_l is not None
        and improved_n is not None
    )
    reasons: list[str] = []
    nonlinear_extra = False
    passes = {"L": False, "N": False}
    if not complete:
        label = "INCOMPLETE"
        if not gates_ok:
            reasons.append("at least one correctness/resource gate failed or is missing")
        if status_l != "COMPLETED" or status_n != "COMPLETED":
            reasons.append(f"arm status L={status_l} N={status_n} (both must be COMPLETED)")
        if d_l is None or d_n is None or d_nl is None:
            reasons.append("one or more primary epoch-8 AP values are missing")
    else:
        if n_scorable is not None and int(n_scorable) != n_expected:
            reasons.append(
                f"scorable task count {n_scorable} != expected {n_expected}; majority rule "
                "is applied to the observed scorable set"
            )
        for arm, d, improved in (("L", float(d_l), int(improved_l)), ("N", float(d_n), int(improved_n))):
            magnitude = d >= floor
            breadth = improved >= majority
            passes[arm] = bool(magnitude and breadth)
            reasons.append(
                f"{arm}: D={d:+.6f} (floor {floor:+.3f} -> {magnitude}), "
                f"improved {improved}/127 (majority {majority} -> {breadth})"
            )
        if any(passes.values()):
            label = "READOUT_SIGNAL"
            nonlinear_extra = bool(float(d_nl) >= float(DECISION["nonlinear_extra_floor"]))
            reasons.append(f"D_NL={float(d_nl):+.6f} (extra-signal floor {DECISION['nonlinear_extra_floor']:+.3f})")
        elif max(float(d_l), float(d_n)) > 0.0:
            label = "WEAK_OR_MIXED"
            reasons.append("directional improvement exists but READOUT_SIGNAL does not hold")
        else:
            label = "NO_FAST_READOUT_SIGNAL"
            reasons.append("neither head improved on the frozen checkpoint at epoch 8")
    return {
        "label": label,
        "nonlinear_extra_signal": bool(nonlinear_extra),
        "arm_passes_readout_floor": passes,
        "D_L": None if d_l is None else float(d_l),
        "D_N": None if d_n is None else float(d_n),
        "D_NL": None if d_nl is None else float(d_nl),
        "improved_tasks_L": None if improved_l is None else int(improved_l),
        "improved_tasks_N": None if improved_n is None else int(improved_n),
        "n_scorable_tasks": None if n_scorable is None else int(n_scorable),
        "thresholds": {
            "improvement_floor": floor,
            "task_majority": majority,
            "nonlinear_extra_floor": float(DECISION["nonlinear_extra_floor"]),
            "ap0": float(AP0),
        },
        "reasons": reasons,
        "rules": DECISION["rules"],
        "statistical_claim": DECISION["statistical_claim"],
    }


__all__ = [
    "ANALYSIS",
    "AP0",
    "ARM_ORDER",
    "ARMS",
    "BATCH_SIZE",
    "BUDGET",
    "CODE_DIR",
    "DATASET",
    "DECISION",
    "EPOCHS",
    "EVAL_EPOCHS",
    "EXTRACT",
    "FEATURE_CACHE",
    "HEAD_RECIPE",
    "HEADS_RUNS",
    "LOSS_SCREEN",
    "LOSS_SCREEN_CODE",
    "LR_PROBE",
    "LR_PROBE_CODE",
    "MAIN_EPOCH",
    "N_MOLECULES",
    "N_SCORABLE_TASKS_EXPECTED",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PERF",
    "PERF_CODE",
    "PILOT",
    "PILOT_CODE",
    "PREFLIGHT_RUN",
    "PROTOCOL_ID",
    "REPO",
    "RESULT_LABELS",
    "RHO_IN_DIM_EXPECTED",
    "RHO_LAYOUT",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SEED",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TRAIN_AP0",
    "TRAIN_AP_TOL",
    "TRACK",
    "VALID_AP_TOL",
    "assert_inherited_constants",
    "decide_label",
    "environment",
    "git_commit",
    "gpu_occupancy",
    "gpu_peak",
    "json_default",
    "log",
    "read_json",
    "seed_everything",
    "sha256_array",
    "sha256_batches",
    "sha256_bytes",
    "sha256_file",
    "sha256_int64",
    "track_rel",
    "utcnow",
    "write_json",
]
