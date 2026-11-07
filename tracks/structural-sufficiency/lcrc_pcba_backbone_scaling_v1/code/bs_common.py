"""Frozen definitions for the LCRC PCBA backbone scaling round v1.

Round: ``lcrc-pcba-backbone-scaling-v1``

This round is a pure capacity scan of the historical LCRC-A computation
(single-shot S1/S2 local statistics + one atom encoding + one bond encoding +
per-atom task contribution ``f`` + global correction ``rho`` + the four-head
attention pool readout).  Only the *widths* change; no new module, depth,
residual block, objective, dropout, LR or data change is introduced.

Frozen, read-only inheritance (never modified, never re-defaulted):

* ``lcrc_pcba_structural_gain_sprint_v1/code/sg_common.py`` -- the from-scratch
  recipe (AdamW, 1-epoch linear warmup 3e-5 -> 3e-4, per-step cosine to 1e-5
  over 109,520 updates, dropout 0.10, clip 5.0) and :func:`scheduled_lr`,
* ``sg_data.py`` -- feature cache / official split / labels / order plan /
  official OGB evaluator,
* ``lcrc_tu_quick_screen_v1/code/tu_model.py`` -- ``TULCRC`` (variant ``C1``)
  and its MLP / local-content definitions,
* ``lcrc_pcba_no_mp_sprint_v1/code/nmp_model.py`` -- ``AttentionPoolReadout``
  (the historical LCRC-A readout) and its private-generator initialisation,
* ``bp_model.C1Model`` -- the C1 construction the historical A used.

Official Test targets are unreachable: the label artifact has every Test row
set to NaN and every code path in this round refuses an index into it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# paths / inherited code (read-only)
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

SG_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1"
NMP_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1"
EP_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_explicit_performance_sprint_v1"
TU_SCREEN = REPO / "tracks/structural-sufficiency/lcrc_tu_quick_screen_v1"

for _p in (str(SG_SPRINT / "code"), str(NMP_SPRINT / "code"), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sg_common as SC  # noqa: E402  (recipe, schedule, frozen identities)
import sg_data as SD  # noqa: E402  (data, order plan, official evaluator)
import pc_metrics as MET  # noqa: E402
import pc_model as PC  # noqa: E402
import tu_model as TU  # noqa: E402
import nmp_model as NM  # noqa: E402
import nmp_common as NC  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-backbone-scaling-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-backbone-scaling-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-backbone-scaling-checkpoint-v1"

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

SOURCE = SC.SOURCE  # frozen data / spec / normalisation identities
RECIPE = SC.SCRATCH_RECIPE
scheduled_lr = SC.scheduled_lr

# ---------------------------------------------------------------------------
# the historical reference (never retrained in this round)
# ---------------------------------------------------------------------------

A0_RUN_DIR = NMP_SPRINT / "runs" / "phaseC_seed0_A"
A0_CHECKPOINT = A0_RUN_DIR / "checkpoint_epoch40.pt"
A0_RUN_JSON = A0_RUN_DIR / "run.json"
A0_TRAIN_EVAL = A0_RUN_DIR / "train_eval_epoch40.json"
A0_CURVE = A0_RUN_DIR / "train_curve.csv"

A0 = {
    "round": "lcrc-pcba-no-mp-sprint-v1",
    "phase": "C",
    "arm": "A",
    "arm_name": "A_attention_pooling",
    "seed": 0,
    "epochs": 40,
    "parameters": 287_644,
    "new_parameters": 42_460,
    "base_parameters": 245_184,
    "ap40": 0.24635609609821457,
    "train_ap40": 0.4196570175055804,
    "train_bce40": 0.030857778624664715,
    "valid_bce40": 0.04577261283603607,
    "run_json_sha256": "120604326256b95190e741f0d58062c113d2537f33d6902a4025e1c0508105c3",
    "checkpoint_sha256": "f110935a6055f3c087f9d0b40580506a512cdeb09f6af67a56bde787cc93a437",
    "train_curve_sha256": "fde90374cb38f7fbb73e62b78ed200d41c5f4fbc2f3e4ea6d5a3ad84870c30b2",
    "base_init_fingerprint": "1b1f1ad35ab33f4107bc600166fce838d5e16ff5def2348743c73c0211dcf7c3",
    "global_plan_sha256": "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
    "consumed40_prefix_sha256": "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58",
    "commit": "b1b917dc",
}

AP0 = float(A0["ap40"])
TRAIN_AP0 = float(A0["train_ap40"])

# ---------------------------------------------------------------------------
# the two scaled models
# ---------------------------------------------------------------------------

#: Parameter : scale-1 value.  Every scaled width is ``scale * value`` except
#: ``n_heads`` (frozen at 4) and ``n_out`` (frozen at the 128 PCBA tasks).
REFERENCE_WIDTHS: dict[str, int] = {
    "atom_hidden": 128,
    "atom_out": 96,
    "bond_hidden": 160,
    "bond_out": 96,
    "stat_rank": 32,
    "local_hidden": 192,
    "width": 64,
    "rho_hidden": 64,
    "scorer_hidden": 32,
    "pool_hidden": 88,
}
FROZEN_WIDTHS = {"n_heads": 4, "n_out": N_TASKS}

#: Exact ``sum(p.numel())`` targets.  All biases, LayerNorms and embedding
#: tables are included; buffers are not parameters.
EXPECTED_PARAMETERS: dict[int, int] = {1: 287_644, 2: 994_324, 3: 2_120_844}
EXPECTED_BASE_PARAMETERS: dict[int, int] = {1: 245_184, 2: 848_096, 3: 1_809_408}
EXPECTED_READOUT_PARAMETERS: dict[int, int] = {1: 42_460, 2: 146_228, 3: 311_436}

#: The human-readable width table of the protocol (scale 1 is ``A0``).
SCALES = (1, 2, 3)

MODELS: dict[str, dict[str, Any]] = {
    "W2": {
        "scale": 2,
        "name": "LCRC-A-Wide-2x",
        "role": "capacity_scan_2x",
        "parameters": EXPECTED_PARAMETERS[2],
        "assigned_pool": "res-gpu0",
    },
    "W3": {
        "scale": 3,
        "name": "LCRC-A-Wide-3x",
        "role": "capacity_scan_3x",
        "parameters": EXPECTED_PARAMETERS[3],
        "assigned_pool": "res-gpu1",
    },
}
MODEL_ORDER = ("W2", "W3")


def config_for_scale(scale: int) -> dict[str, Any]:
    """The exact width dict passed to every constructor (no implicit default)."""
    s = int(scale)
    if s not in SCALES:
        raise KeyError(f"scale {scale!r} is not in {SCALES}")
    cfg: dict[str, Any] = {k: int(v) * s for k, v in REFERENCE_WIDTHS.items()}
    cfg.update(dict(FROZEN_WIDTHS))
    return cfg


def width_table() -> dict[str, dict[int, int]]:
    return {k: {s: int(v) * s for s in SCALES} for k, v in REFERENCE_WIDTHS.items()}


#: Input vocabulary / feature widths: identical in every arm, never scaled.
FROZEN_INPUTS = {
    "atom_feat_dim": 51,
    "bond_feat_dim": 13,
    "atom_scalar_dim": 9,
    "global_dim": 42,
    "n_nbr_codes": 80,
    "n_path_codes": 400,
}

# ---------------------------------------------------------------------------
# recipe / evaluation protocol
# ---------------------------------------------------------------------------

SEED = 0
MODEL_SEED = 0
INIT_SEED_RULE = (
    "torch.manual_seed(seed) -> build the scale's FULL TULCRC -> remap to C1 "
    "(drop the 3*atom_out b_mean columns) -> load -> attention readout from the "
    "private generator seed*1000003+17 with readout.fc2 zeroed (historical A rule)"
)
#: Fraction-of-epoch budget that decides the Validation cadence (frozen rule).
EVAL_RULE = {
    "threshold_fraction_of_projected_epoch": 0.15,
    "cadence_if_all_cheap": 1,
    "cadence_if_any_expensive": 2,
    "cadence_applies_to_both_models": True,
    "primary_epoch": TOTAL_EPOCHS,
    "optional_diagnostic_epoch0": True,
}

BUDGET = {"gpu_target_hours": 8.0, "gpu_hard_hours": 12.0, "max_concurrent_gpus": 2}

#: Resource-decision thresholds (identical to the round brief; resource
#: allocation only, not a significance test).
DECISION_RULES = {
    "promising": {
        "delta_ap40_min": 0.005,
        "mean_delta_support_ge20_min": 0.003,
        "improved_support_ge20_min": 60,
        "support_ge20_total": 107,
        "label": "PROMISING_SCALE_SIGNAL",
    },
    "weak": {"delta_ap40_min": 0.002, "delta_ap40_max": 0.005, "label": "WEAK_SCALE_SIGNAL"},
    "none": {"delta_ap40_min": -0.002, "delta_ap40_max": 0.002, "label": "NO_CLEAR_SCALE_GAIN"},
    "regression": {"delta_ap40_max": -0.002, "label": "SCALE_REGRESSION"},
    "support_strata": {"ge20": 20, "ge5_lt20": 5},
}

#: Fixed probe sizes.
PROBE_VALID_BATCH = 256
PROBE_TRAIN_STEPS = 30
PROBE_WARMUP_STEPS = 5

# ---------------------------------------------------------------------------
# IO helpers (thin wrappers over the inherited, already-audited helpers)
# ---------------------------------------------------------------------------

utcnow = SC.utcnow
write_json = SC.write_json
read_json = SC.read_json
sha256_file = SC.sha256_file
sha256_bytes = SC.sha256_bytes
log = SC.log
environment = SC.environment
seed_everything = SC.seed_everything
capture_rng_state = SC.capture_rng_state
rng_state_hashes = SC.rng_state_hashes
git_commit = SC.git_commit
track_rel = SC.track_rel
_json_default = SC._json_default


def state_fingerprint(state: Mapping[str, Any]) -> str:
    return TU.state_fingerprint(state)


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def recipe_for_model(model_key: str) -> dict[str, Any]:
    if model_key not in MODELS:
        raise KeyError(model_key)
    m = MODELS[model_key]
    cfg = config_for_scale(int(m["scale"]))
    recipe = dict(RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "model": model_key,
            "model_name": m["name"],
            "role": m["role"],
            "scale": int(m["scale"]),
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_cadence": int(EVAL_CADENCE["every"]),
            "eval_epochs": list(calc_eval_epochs(int(EVAL_CADENCE["every"]))),
            "widths": cfg,
            "parameters": int(EXPECTED_PARAMETERS[int(m["scale"])]),
            "lr_schedule": lr_audit(),
            "init_rule": INIT_SEED_RULE,
            "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
        }
    )
    return recipe


def lr_audit() -> dict[str, Any]:
    steps = STEPS_PER_EPOCH
    return {
        "steps_per_epoch": steps,
        "total_steps_40_epochs": TOTAL_STEPS_40,
        "step0": scheduled_lr(0, phase=3, kind="all"),
        "warmup_last": scheduled_lr(steps - 1, phase=3, kind="all"),
        "post_warmup_first": scheduled_lr(steps, phase=3, kind="all"),
        "step10e": scheduled_lr(10 * steps, phase=3, kind="all"),
        "step20e": scheduled_lr(20 * steps, phase=3, kind="all"),
        "step40e": scheduled_lr(TOTAL_STEPS_40, phase=3, kind="all"),
        "formula": (
            "steps 0..2737 linear 3e-5->3e-4; then "
            "1e-5+(3e-4-1e-5)*0.5*(1+cos(pi*(s-2738)/(109520-2738)))"
        ),
    }


#: Filled from ``analysis/evaluation_cost_probe.json`` before the first formal
#: step (see PROTOCOL.md section 6).  The *rule* below is frozen; only the
#: resolved cadence is a measurement.
EVAL_CADENCE: dict[str, Any] = {"every": 1, "resolved": False, "source": "protocol_default"}


def set_eval_cadence(every: int, *, source: str) -> dict[str, Any]:
    every = int(every)
    if every not in (1, 2):
        raise KeyError(f"cadence {every} is not the 1-or-2 rule")
    EVAL_CADENCE.update({"every": every, "resolved": True, "source": str(source)})
    return dict(EVAL_CADENCE)


def calc_eval_epochs(every: int) -> list[int]:
    return [e for e in range(1, TOTAL_EPOCHS + 1) if e % int(every) == 0]


# ---------------------------------------------------------------------------
# GPU occupancy / environment facts
# ---------------------------------------------------------------------------


def gpu_occupancy() -> dict[str, Any]:
    """Visible free memory, co-tenant processes and utilisation (facts only)."""
    out: dict[str, Any] = {"captured_utc": utcnow(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch

        out["torch"] = torch.__version__
        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["device_count"] = int(torch.cuda.device_count())
            out["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            free, total = torch.cuda.mem_get_info()
            out["visible_device_mem_get_info"] = {
                "free_bytes": int(free),
                "total_bytes": int(total),
                "used_bytes": int(total - free),
            }
    except Exception as exc:  # pragma: no cover - diagnostic only
        out["torch_error"] = f"{type(exc).__name__}: {exc}"
    for key, args in (
        (
            "nvidia_smi_gpus",
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,"
                "utilization.gpu,utilization.memory,temperature.gpu,driver_version",
                "--format=csv,noheader",
            ],
        ),
        (
            "nvidia_smi_compute_apps",
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        ),
    ):
        try:
            out[key] = subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=20).strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            out[key] = f"unavailable: {type(exc).__name__}: {exc}"
    return out


def gpu_peak(device: Any) -> dict[str, int]:
    import torch

    if getattr(device, "type", str(device)) != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


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
        "dirty_files": [x for x in (_git("status", "--porcelain") or "").splitlines() if x],
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
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "git": git_state(),
    }
    if torch.cuda.is_available():
        out["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        out["cuda_capability"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    return out


__all__ = [
    "A0",
    "A0_CHECKPOINT",
    "A0_CURVE",
    "A0_RUN_DIR",
    "A0_RUN_JSON",
    "A0_TRAIN_EVAL",
    "ANALYSIS",
    "AP0",
    "BATCH_SIZE",
    "BUDGET",
    "CODE_DIR",
    "DATASET",
    "DECISION_RULES",
    "EP_SPRINT",
    "EVAL_CADENCE",
    "EVAL_RULE",
    "EXPECTED_BASE_PARAMETERS",
    "EXPECTED_PARAMETERS",
    "EXPECTED_READOUT_PARAMETERS",
    "FROZEN_INPUTS",
    "FROZEN_WIDTHS",
    "MODELS",
    "MODEL_ORDER",
    "MODEL_SEED",
    "MET",
    "NC",
    "NM",
    "NMP_SPRINT",
    "N_MOLECULES",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PC",
    "PROBE_TRAIN_STEPS",
    "PROBE_VALID_BATCH",
    "PROBE_WARMUP_STEPS",
    "PROTOCOL_ID",
    "RECIPE",
    "REFERENCE_WIDTHS",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SC",
    "SCALES",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_RUN",
    "SD",
    "SEED",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TRACK",
    "TRAIN_AP0",
    "TU",
    "TU_SCREEN",
    "calc_eval_epochs",
    "capture_rng_state",
    "config_for_scale",
    "environment",
    "git_commit",
    "git_state",
    "gpu_occupancy",
    "gpu_peak",
    "host_facts",
    "log",
    "lr_audit",
    "parameter_count",
    "read_json",
    "recipe_for_model",
    "rng_state_hashes",
    "scheduled_lr",
    "seed_everything",
    "set_eval_cadence",
    "sha256_bytes",
    "sha256_file",
    "state_fingerprint",
    "track_rel",
    "utcnow",
    "width_table",
    "write_json",
]
