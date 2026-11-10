"""Frozen definitions for the PCQM4Mv2 transfer pilot round v1.

Round: ``lcrc-pcqm4mv2-transfer-pilot-v1``

The research question, the model, the data split, the training recipe and the
stopping rules were fixed by the execution instruction *before* any PCQM4Mv2
SMILES was parsed or any model step was taken.  This module is the single place
where the pre-registered quantities live; nothing here is chosen after seeing a
score.

Read-only inheritance (imported, never modified, never re-defaulted):

* ``lcrc_pcba_backbone_scaling_v1/code/bs_common.py`` -- the scaled-trunk
  width table (``config_for_scale``), the frozen initialisation rule and the
  frozen recipe values,
* ``lcrc_pcba_backbone_scaling_v1/code/bs_model.py`` -- ``BSModel`` (the
  one-shot LCRC-A graph) and the FULL -> C1 state remap / readout init,
* ``lcrc_tu_quick_screen_v1/code/tu_model.py`` -- ``TULCRC`` and the one-shot
  S1/S2 ``TULocalContent`` local encoder,
* ``lcrc_pcba_no_mp_sprint_v1/code/nmp_model.py`` -- ``AttentionPoolReadout``
  (the historical four-head readout),
* ``lcrc_cross_task_preparation_v1/code/{feature_spec,molecular_graph,lcrc_data}.py``
  -- the ``lipo_v1`` structural schema, the canonical molecular graph and the
  cache / collate / Fit-only-normalisation layer,
* ``lcrc_molnet_benchmark_pilot_v2/code/mb2_rdkit.py`` -- the frozen
  SMILES -> raw-attribute adapter,
* ``lcrc_pcba_scale_pilot_v1/code/pc_data.py`` -- the sharded build / mmap
  merge / cache verification design (re-implemented in ``pq_data`` with a
  dataset label of this round, and checked numerically against it in
  ``tests/test_pq_cache.py``),
* ``lcrc_pcba_pipeline_perf_v1/code/pf_pipeline.py`` -- ``PinnedStager``.

Migration changes (the complete list, per the instruction):

1. the model output is **1** regression value instead of 128 class logits,
2. the loss is L1 on the standardised gap instead of masked BCE on logits,
3. PCQM4Mv2 SMILES / labels / official split replace the PCBA graph dataset,
4. training, evaluation, cache and resource plumbing are adapted.

No module, width, depth, residual block, objective term, normalisation policy
or training hyper-parameter is added, removed or retuned.
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-pcqm4mv2-transfer-pilot-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcqm4mv2-transfer-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcqm4mv2-transfer-checkpoint-v1"

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

#: The frozen trunk recipe rests on the PCBA rounds' code.  Importing
#: ``pc_common`` puts the preparation / TU / molnet code directories on
#: ``sys.path`` (it is the already-audited entry point of that chain); the
#: backbone-scaling code is added explicitly below.
PC_SCALE = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PC_SCALE_CODE = PC_SCALE / "code"
BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
BS_CODE = BS_ROUND / "code"
NMP_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1"

for _p in (str(PC_SCALE_CODE), str(BS_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pc_common as PC  # noqa: E402  (frozen IO helpers + sys.path for prep code)
import bs_common as BS  # noqa: E402  (width table, init rule, recipe values)

ANALYSIS = ROUND / "analysis"
CACHE = ROUND / "cache"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
CONFIGS = ROUND / "configs"

# ---------------------------------------------------------------------------
# dataset identity (verified, not assumed -- see analysis/pcqm_data_audit.json)
# ---------------------------------------------------------------------------

DATASET = "pcqm4m-v2"
TASK = "HOMO-LUMO gap regression (eV)"
N_OUT = 1

#: ``PROJECT_DATA_DIR`` (AGENTS.md) with the repository ``data/`` as fallback.
DATA = Path(os.environ.get("PROJECT_DATA_DIR") or (REPO / "data"))
OGB_ROOT = DATA / "ogb"
PCQM_ROOT = OGB_ROOT / DATASET

#: The official split keys that the public dataset object exposes.  Only
#: ``train`` and ``valid`` carry labels; ``test-dev`` / ``test-challenge``
#: labels are hidden and are never indexed by this round.
SPLIT_KEYS = ("train", "valid", "test-dev", "test-challenge")
LABELLED_SPLITS = ("train", "valid")
HIDDEN_SPLITS = ("test-dev", "test-challenge")

# ---------------------------------------------------------------------------
# fixed subsets, seeds and monitor rows
# ---------------------------------------------------------------------------

#: Sampling RNG -- one full random permutation of the official Train row ids.
SAMPLE_SEED = 42
#: Model initialisation and batch-order RNG (separate from the sampling RNG).
MODEL_SEED = 0
ORDER_SEED = 0
#: Train-monitor selection RNG (again independent).
MONITOR_SEED = 43
N_MONITOR = 20_000

#: Formal runs.  ``rows`` is the number of cache rows used (the cache itself
#: stores the 1M subset, so the 300K run is the first 300,000 of those rows).
RUNS_TABLE: dict[str, dict[str, Any]] = {
    "PCQM_W2_300K": {
        "train_size": 300_000,
        "epochs": 40,
        "seed": MODEL_SEED,
        "cache": "pcqm_train1m",
        "cache_rows": 300_000,
    },
    "PCQM_W2_1M": {
        "train_size": 1_000_000,
        "epochs": 40,
        "seed": MODEL_SEED,
        "cache": "pcqm_train1m",
        "cache_rows": 1_000_000,
    },
}
RUN_ORDER = ("PCQM_W2_300K", "PCQM_W2_1M")

# ---------------------------------------------------------------------------
# model (W2 widths, one regression output)
# ---------------------------------------------------------------------------

MODEL_KEY = "W2"
MODEL_NAME = "LCRC-W2"
WIDTH_AUDIT_SCALE = 2
#: ``bs_common.config_for_scale(2)`` with the *only* migrated field changed.
MODEL_CFG: dict[str, Any] = {**BS.config_for_scale(WIDTH_AUDIT_SCALE), "n_out": int(N_OUT)}
FROZEN_INPUTS = dict(BS.FROZEN_INPUTS)
INIT_SEED_RULE = BS.INIT_SEED_RULE
REFERENCE_WIDTHS = dict(BS.REFERENCE_WIDTHS)

# ---------------------------------------------------------------------------
# recipe (fixed by the instruction; values inherited where they coincide)
# ---------------------------------------------------------------------------

BATCH_SIZE = int(BS.BATCH_SIZE)
TOTAL_EPOCHS = 40
RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "weight_decay": 1.0e-3,
    "learning_rate": 3.0e-4,
    "warmup_epochs": 1.0,
    "warmup_start_lr": 3.0e-5,
    "end_lr": 1.0e-5,
    "scheduler": "1-epoch linear warmup then per-step cosine over the 40-epoch horizon",
    "dropout": 0.10,
    "batch_size": BATCH_SIZE,
    "gradient_clip": 5.0,
    "precision": "fp32",
    "epochs": TOTAL_EPOCHS,
    "early_stopping": False,
    "loss": "L1 (mean absolute error) on the Train-standardised gap z",
    "target": "z = (gap_eV - mean_gap_300k) / std_gap_300k (ddof=0, fixed Train 300K)",
    "batch_order": "fresh deterministic permutation of the run's rows at every epoch (numpy default_rng(ORDER_SEED))",
    "selection_metric": "official PCQM4Mv2Evaluator MAE on the full official Validation split, eV",
    "test_policy": "official test-dev / test-challenge labels are hidden and are never indexed or scored",
}

#: Documented PCQM4Mv2 reference numbers (OGB-LSC), used only for *positioning*
#: in the report.  They are NOT this round's measurements and were not used to
#: choose anything.
PUBLIC_REFERENCE = {
    "source": "OGB-LSC PCQM4Mv2 paper / leaderboard (full 3.78M-molecule training set)",
    "GCN": {"valid_mae": 0.1960, "test_dev_mae": 0.2021},
    "GIN": {"valid_mae": 0.1536, "test_dev_mae": 0.1678},
    "GIN-virtual": {"valid_mae": 0.1396, "test_dev_mae": 0.1487},
    "note": (
        "public numbers train on the FULL official Train split (~337.9M/337.9万 = 3,378,606 "
        "molecules) for 100+ epochs; this round trains on 0.3M / 1.0M molecules for 40 epochs, "
        "so the comparison only locates the model, it is not a like-for-like benchmark"
    ),
}


def steps_per_epoch(n_rows: int, batch_size: int = BATCH_SIZE) -> int:
    return int(math.ceil(int(n_rows) / int(batch_size)))


def total_steps(n_rows: int, epochs: int = TOTAL_EPOCHS, batch_size: int = BATCH_SIZE) -> int:
    return int(epochs) * steps_per_epoch(n_rows, batch_size)


def scheduled_lr(global_step: int, *, steps_epoch: int, steps_total: int) -> float:
    """LR applied *before* optimizer step ``global_step`` (0-based).

    Identical shape to the frozen from-scratch recipe of the PCBA rounds
    (``sg_common.scheduled_lr`` phase 3), with the horizon of *this* run:
    one epoch of linear warmup ``3e-5 -> 3e-4`` followed by a per-step cosine
    from ``3e-4`` to ``1e-5`` over the remaining steps.
    """
    step = max(0, int(global_step))
    steps_epoch = int(steps_epoch)
    steps_total = int(steps_total)
    hi = float(RECIPE["learning_rate"])
    lo = float(RECIPE["end_lr"])
    start = float(RECIPE["warmup_start_lr"])
    if step < steps_epoch:
        return float(start + (hi - start) * (step + 1) / steps_epoch)
    frac = min(1.0, (step - steps_epoch) / float(max(1, steps_total - steps_epoch)))
    return float(lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * frac)))


def lr_audit(n_rows: int) -> dict[str, Any]:
    s = steps_per_epoch(n_rows)
    t = total_steps(n_rows)
    return {
        "rows": int(n_rows),
        "steps_per_epoch": s,
        "total_steps_40_epochs": t,
        "batch_size": BATCH_SIZE,
        "lr_step0": scheduled_lr(0, steps_epoch=s, steps_total=t),
        "lr_warmup_last": scheduled_lr(s - 1, steps_epoch=s, steps_total=t),
        "lr_post_warmup_first": scheduled_lr(s, steps_epoch=s, steps_total=t),
        "lr_step10e": scheduled_lr(10 * s, steps_epoch=s, steps_total=t),
        "lr_step20e": scheduled_lr(20 * s, steps_epoch=s, steps_total=t),
        "lr_step30e": scheduled_lr(30 * s, steps_epoch=s, steps_total=t),
        "lr_last_step_used": scheduled_lr(t - 1, steps_epoch=s, steps_total=t),
        "lr_at_horizon": scheduled_lr(t, steps_epoch=s, steps_total=t),
        "formula": (
            f"steps 0..{s - 1}: linear 3e-5->3e-4; steps {s}..{t}: "
            f"1e-5+(3e-4-1e-5)*0.5*(1+cos(pi*(step-{s})/({t}-{s})))"
        ),
    }


def recipe_for_run(run: str) -> dict[str, Any]:
    if run not in RUNS_TABLE:
        raise KeyError(run)
    spec = RUNS_TABLE[run]
    n = int(spec["train_size"])
    return {
        **{k: v for k, v in RECIPE.items() if not isinstance(v, tuple)},
        "adam_betas": list(RECIPE["adam_betas"]),
        "round": ROUND_ID,
        "run": run,
        "model": MODEL_KEY,
        "model_name": MODEL_NAME,
        "train_size": n,
        "cache_rows": int(spec["cache_rows"]),
        "seed": int(spec["seed"]),
        "epochs": int(spec["epochs"]),
        "widths": dict(MODEL_CFG),
        "lr_schedule": lr_audit(n),
        "init_rule": INIT_SEED_RULE,
        "feature_schema": "lipo_v1",
        "n_outputs": int(N_OUT),
    }


# ---------------------------------------------------------------------------
# parameter accounting (independent of torch: shapes from the width table)
# ---------------------------------------------------------------------------


def expected_parameter_count(cfg: Mapping[str, Any], *, atom_feat_dim: int, bond_feat_dim: int,
                            n_scalars: int, global_dim: int, n_nbr_codes: int,
                            n_path_codes: int) -> dict[str, Any]:
    """Analytic parameter count of the migrated W2 model, by module.

    Recomputed from the width table rather than read from the live model, so
    ``pq_model`` can assert that the built module has exactly these numbers.
    """
    a_out = int(cfg["atom_out"])
    width = int(cfg["width"])
    rank = int(cfg["stat_rank"])
    n_out = int(cfg["n_out"])

    def mlp(dims: Sequence[int]) -> int:
        return int(sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1)))

    local_in = 3 * a_out + 4 * rank + 8 + int(n_scalars)  # a, b_sum, 4 stat blocks, closure, s2 content, scalars
    parts = {
        "atom_encoder": mlp([atom_feat_dim, int(cfg["atom_hidden"]), a_out]),
        "bond_encoder": mlp([3 * a_out + bond_feat_dim, int(cfg["bond_hidden"]), int(cfg["bond_out"])]),
        "local_mlp": mlp([local_in, int(cfg["local_hidden"]), width]),
        "local_norm": 2 * width,
        "local_embeddings": (
            int(n_nbr_codes) * rank  # nbr_emb
            + int(n_path_codes) * rank  # path_emb
            + int(n_nbr_codes) * rank  # end_emb
            + int(n_path_codes) * rank  # path_u
            + int(n_nbr_codes) * rank  # path_v
            + rank * rank + rank  # joint_proj
            + 2 * 8  # close_emb
            + int(n_path_codes)  # path_gate
        ),
        "f": mlp([width, width, n_out]),
        "rho": mlp([int(global_dim) + 2 * width + 1, int(cfg["rho_hidden"]), n_out]),
        "readout": (
            int(cfg["n_heads"]) * mlp([width, int(cfg["scorer_hidden"]), 1])
            + mlp([int(cfg["n_heads"]) * width, int(cfg["pool_hidden"]), n_out])
        ),
    }
    trunk = int(sum(v for k, v in parts.items() if k != "readout"))
    return {
        "by_module": parts,
        "local_input_dim": int(local_in),
        "rho_input_dim": int(global_dim) + 2 * width + 1,
        "base_parameters": trunk,
        "readout_parameters": int(parts["readout"]),
        "total_parameters": int(trunk + parts["readout"]),
        "historical_pcba_w2_parameters": int(BS.EXPECTED_PARAMETERS[2]),
        "delta_vs_historical": int(trunk + parts["readout"]) - int(BS.EXPECTED_PARAMETERS[2]),
        "explanation": (
            "the only parameter difference to the historical W2 is the output layer: 128 class "
            "logits -> 1 regression value in f, rho and the readout head"
        ),
    }


# ---------------------------------------------------------------------------
# IO helpers (inherited, already audited)
# ---------------------------------------------------------------------------

utcnow = PC.utcnow
write_json = PC.write_json
read_json = PC.read_json
sha256_file = PC.sha256_file
sha256_tree = PC.sha256_tree
sha256_bytes = PC.sha256_bytes
index_hash = PC.index_hash
_json_default = PC._json_default
log = PC.log
git_commit = PC.git_commit
git_dirty = PC.git_dirty
environment = PC.environment
seed_everything = PC.seed_everything


def track_rel(path: Path | str) -> str:
    """Path relative to the repository when possible, else absolute."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def capture_rng_state() -> dict[str, Any]:
    import random

    import torch

    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [s.detach().cpu() for s in torch.cuda.get_rng_state_all()]
    return state


def rng_state_hashes(state: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    def _h(value: Any) -> str:
        if isinstance(value, torch.Tensor):
            return sha256_bytes(value.detach().cpu().numpy().tobytes())
        if isinstance(value, np.ndarray):
            return sha256_bytes(value.tobytes())
        return sha256_bytes(repr(value).encode("utf-8"))

    cuda = state.get("torch_cuda")
    return {
        "torch_cpu_sha256": _h(state["torch_cpu"]),
        "numpy_sha256": _h(np.asarray(state["numpy"][1])),
        "python_sha256": _h(state["python_random"]),
        "torch_cuda_sha256": [_h(s) for s in (cuda or [])],
        "combined_sha256": sha256_bytes(
            b"".join(
                [
                    _h(state["torch_cpu"]).encode(),
                    _h(np.asarray(state["numpy"][1])).encode(),
                    _h(state["python_random"]).encode(),
                ]
                + [s.encode() for s in (_h(x) for x in (cuda or []))]
            )
        ),
    }


def gpu_occupancy() -> dict[str, Any]:
    """Visible free memory, co-tenant processes and utilisation (facts only)."""
    out: dict[str, Any] = {
        "captured_utc": utcnow(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
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
        "git_commit": git_commit(),
    }
    if torch.cuda.is_available():
        out["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return out


def peak_rss_mb() -> float:
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


__all__ = [
    "ANALYSIS",
    "BATCH_SIZE",
    "BS",
    "BS_CODE",
    "BS_ROUND",
    "CACHE",
    "CODE_DIR",
    "CONFIGS",
    "DATA",
    "DATASET",
    "FROZEN_INPUTS",
    "HIDDEN_SPLITS",
    "INIT_SEED_RULE",
    "LABELLED_SPLITS",
    "MODEL_CFG",
    "MODEL_KEY",
    "MODEL_NAME",
    "MODEL_SEED",
    "MONITOR_SEED",
    "NMP_SPRINT",
    "N_MONITOR",
    "N_OUT",
    "OGB_ROOT",
    "ORDER_SEED",
    "PC",
    "PCQM_ROOT",
    "PC_SCALE",
    "PC_SCALE_CODE",
    "PROTOCOL_ID",
    "PUBLIC_REFERENCE",
    "RECIPE",
    "REFERENCE_WIDTHS",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "RUNS_TABLE",
    "RUN_ORDER",
    "SAMPLE_SEED",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_RUN",
    "SPLIT_KEYS",
    "TASK",
    "TESTS",
    "TOTAL_EPOCHS",
    "TRACK",
    "WIDTH_AUDIT_SCALE",
    "capture_rng_state",
    "environment",
    "expected_parameter_count",
    "git_commit",
    "git_dirty",
    "gpu_occupancy",
    "host_facts",
    "index_hash",
    "log",
    "lr_audit",
    "peak_rss_mb",
    "read_json",
    "recipe_for_run",
    "rng_state_hashes",
    "scheduled_lr",
    "seed_everything",
    "sha256_bytes",
    "sha256_file",
    "sha256_tree",
    "steps_per_epoch",
    "total_steps",
    "track_rel",
    "utcnow",
    "write_json",
]
