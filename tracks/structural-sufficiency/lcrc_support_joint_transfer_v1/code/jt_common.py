"""Shared constants, paths and IO for ``lcrc-support-joint-transfer-v1``.

This round only *uses* four historical implementations (all read-only):

* ``lcrc_readout_context_pilot_small_v1`` -- the official-split
  ``ogbg-mollipo`` C1 A arm (data cache, row ids, target normaliser, canonical
  C1 initial state, 360-epoch recipe) and its frozen A checkpoint;
* ``lcrc_support_joint_function_v0`` -- the JOINT / ADDITIVE S2 joint-channel
  parameterisations, the Train-only Omega construction, the marginal projector
  and the unseen-pair fallback;
* ``lcrc_pcba_backbone_scaling_v1`` -- the W2 backbone and its historical
  seed-0 run (reused, never retrained);
* ``direct_structural_computation_recurrence_v0`` -- the FIXED1 two-stage
  computation and the historical seed-0 25-epoch result (reused).

Nothing in this file chooses a scientific quantity from a score.  Every frozen
constant is also written into ``configs/experiment.json`` by
:func:`config_payload`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-support-joint-transfer-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO = TRACK_DIR.parents[1]

SS = REPO / "tracks/structural-sufficiency"
READOUT_ROUND = SS / "lcrc_readout_context_pilot_small_v1"
READOUT_CODE = READOUT_ROUND / "code"
SJ_V0 = SS / "lcrc_support_joint_function_v0"
SJ_CODE = SJ_V0 / "code"
BS_ROUND = SS / "lcrc_pcba_backbone_scaling_v1"
BS_CODE = BS_ROUND / "code"
DSCR = SS / "direct_structural_computation_recurrence_v0"
DSCR_CODE = DSCR / "code"
SG_ROUND = SS / "lcrc_pcba_structural_gain_sprint_v1"
SG_CODE = SG_ROUND / "code"
LR_ROUND = SS / "lcrc_pcba_lr_probe_v1"
LR_CODE = LR_ROUND / "code"
CC_ROUND = SS / "lcrc_pcba_continuation_recipe_v1"
CC_CODE = CC_ROUND / "code"
PILOT_ROUND = SS / "lcrc_pcba_scale_pilot_v1"
PC_CACHE = PILOT_ROUND / "cache/pcba_v1"
PCBA_SPEC_PATH = PILOT_ROUND / "analysis/pcba_spec.json"
PCBA_SPLIT_PATH = PILOT_ROUND / "analysis/pcba_split.json"

ANALYSIS = ROUND_DIR / "analysis"
CONFIGS = ROUND_DIR / "configs"
RUNS = ROUND_DIR / "runs"
RESULTS = ROUND_DIR / "results"
TESTS = ROUND_DIR / "tests"

SPLITS = ("train", "valid", "test")
SEED = 0
N_SEEDS = (SEED,)

# ---------------------------------------------------------------------------
# the four new arms (exactly four training units; no further models)
# ---------------------------------------------------------------------------

ARM_LIPO_JOINT = "C1_JOINT"
ARM_LIPO_ADD = "C1_ADD"
ARM_W2_JOINT = "W2_JOINT"
ARM_FIXED1_JOINT = "FIXED1_JOINT"
NEW_ARMS = (ARM_LIPO_JOINT, ARM_LIPO_ADD, ARM_W2_JOINT, ARM_FIXED1_JOINT)
TRAIN_QUEUE = (ARM_LIPO_JOINT, ARM_LIPO_ADD, ARM_FIXED1_JOINT, ARM_W2_JOINT)
LIPO_ARMS = (ARM_LIPO_JOINT, ARM_LIPO_ADD)
PCBA_ARMS = (ARM_W2_JOINT, ARM_FIXED1_JOINT)

# ---------------------------------------------------------------------------
# historical reused baselines (never retrained, never modified)
# ---------------------------------------------------------------------------

LIPO_A_DIR = READOUT_ROUND / "results/ogbg-mollipo/A"
LIPO_A_CHECKPOINT = LIPO_A_DIR / "checkpoint_best_val.pt"
LIPO_A_STATUS = LIPO_A_DIR / "status.json"
LIPO_A_TRAJECTORY = LIPO_A_DIR / "trajectory.json"
LIPO_A_CURVE = LIPO_A_DIR / "curve.csv"

W2_DIR = BS_ROUND / "runs/W2"
W2_CHECKPOINT = W2_DIR / "checkpoint_best.pt"
W2_RUN = W2_DIR / "run.json"
W2_CURVE = W2_DIR / "train_curve.csv"

FIXED1_DIR = DSCR / "runs/FIXED1_seed0"
FIXED1_CHECKPOINT = FIXED1_DIR / "checkpoint_best.pt"
FIXED1_EVAL = FIXED1_DIR / "best_checkpoint_eval.json"
FIXED1_RECORD = DSCR / "results/runs/FIXED1_seed0/run.json"
FIXED1_CURVE = DSCR / "results/runs/FIXED1_seed0/train_curve.csv"

LIPO_A_EXPECTED = {
    "parameters": 228_674,
    "best_epoch": 275,
    "valid_rmse_raw": 0.7086243164299489,
    "test_rmse_raw": 0.7253920405078428,
    "batch_order_sha256": "1a3a440fded933ec18dc02b54de25317987a4b7c5581bbbc941fa24a428b1618",
    "split_sizes": {"train": 3360, "valid": 420, "test": 420},
}
W2_EXPECTED = {
    "parameters": 994_324,
    "base_parameters": 848_096,
    "readout_parameters": 146_228,
    "best_epoch": 24,
    "best_valid_ap": 0.26527331610028143,
    "test_ap_best": 0.2536164787414065,
    "total_epochs_trained": 40,
    "horizon": 25,
    "global_plan_sha256": "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
    "prefix25_sha256": "8c857251a58b4534f287338052fa0be259cf01f341da81d55716a3c2a6b29b40",
}
FIXED1_EXPECTED = {
    "parameters": 1_128_468,
    "best_epoch": 25,
    "best_valid_ap": 0.27584552136653284,
    "test_ap_best": 0.25700143219823374,
    "horizon": 25,
    "global_plan_sha256": "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
    "prefix25_sha256": "8c857251a58b4534f287338052fa0be259cf01f341da81d55716a3c2a6b29b40",
}
DYNAMIC_REFERENCE = {"valid_ap": 0.275067, "test_ap": 0.262226, "parameters": 1_128_468}
LIPO_GINE_REFERENCE = {"valid_rmse_raw": 0.6091, "test_rmse_raw": 0.6456}

#: Frozen parameter targets of the new support-indexed joint modules.
LIPO_WIDTH = 64
LIPO_STAT_RANK = 32
LIPO_FULL_C1_PARAMS = 228_674
PCBA_WIDTH = 128
PCBA_STAT_RANK = 64
PCBA_N_TASKS = 128
PCBA_W2_PARAMS = 994_324
PCBA_FIXED1_PARAMS = 1_128_468

#: Recipe / budget (frozen before any formal training).
RECIPE_COMMON = {
    "optimizer": "AdamW",
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "weight_decay": 1e-3,
    "batch_size": 128,
    "gradient_clip": 5.0,
    "dropout": 0.10,
    "seed": 0,
    "amp": False,
    "ema": False,
    "swa": False,
    "soup": False,
    "early_stopping": False,
    "hyperparameter_search": False,
}
LIPO_RECIPE = {
    **RECIPE_COMMON,
    "epochs": 360,
    "peak_lr": 3e-4,
    "min_lr": 1e-5,
    "scheduler": "per_step_cosine over 360 * ceil(3360/128) steps, no warmup",
    "eval_batch_size": 256,
    "eval_every_epoch": "full Train + Valid + Test",
    "selection": "minimum Valid RMSE; strict improvement; ties keep the earlier epoch",
    "shuffle": "np.random.default_rng(0).permutation over Train split-local rows, one per epoch",
}
PCBA_RECIPE = {
    **RECIPE_COMMON,
    "epochs": 40,
    "horizon": 25,
    "peak_lr": 3e-4,
    "warmup_start_lr": 3e-5,
    "min_lr": 1e-5,
    "warmup_epochs": 1,
    "scheduler": "1-epoch linear warmup then per-step cosine over the historical 40-epoch schedule; horizon 25 is a prefix, never recompressed",
    "eval_batch": {"train": 1024, "valid": 512, "test": 512},
    "eval_every_epoch": "full Train + Valid + Test",
    "selection": "highest Valid macro AP within the 25-epoch prefix; strict improvement; ties keep the earlier epoch",
    "test_role": "DEVELOPMENT_TEST / EXPOSED_TEST_ENGINEERING_COMPARISON_SET",
}
BUDGET = {
    "gpu_soft_cap_hours": 3.0,
    "gpu_hard_cap_hours": 4.0,
    "wall_clock_hard_cap_hours": 3.5,
    "max_new_training_units": 4,
    "preferred_gpu": 1,
    "priority": "Lipo first; PCBA independently scheduled; FIXED1_JOINT before W2_JOINT",
}

#: Statistical / diagnostic constants frozen in advance.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261126
MOLECULE_BOOTSTRAP_REPLICATES = 200
SUPPORT_MIN = 20
EXTREME_K = (1, 5, 10)
TOLERANCES = {
    "forward_abs": 1e-5,
    "param_bitwise": 0.0,
    "projection_abs": 1e-8,
    "invariance_abs": 1e-5,
    "replay_abs": 1e-5,
    "replay_ap_abs": 5e-4,
    "gradient_nonzero_abs": 0.0,
}

#: Test-role / reporting labels.
LIPO_TEST_ROLE = "DEVELOPMENT_TEST"
PCBA_TEST_ROLE = "DEVELOPMENT_TEST / EXPOSED_TEST_ENGINEERING_COMPARISON_SET"

#: Reference effect sizes (exploratory, not significance thresholds).
REFERENCE_DELTA = {"lipo_rmse_relative": 0.03, "pcba_ap_abs": 0.005}
TEST_AP_TARGETS = (0.28, 0.29)


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def sha256_state(state: Mapping[str, Any], keys: Sequence[str] | None = None) -> str:
    digest = hashlib.sha256()
    selected = sorted(state.keys()) if keys is None else list(keys)
    for key in selected:
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    if columns is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        columns = ordered
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return target


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def atomic_save_npz(path: Path | str, arrays: Mapping[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}.npz")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **{str(k): np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, target)
    return target


def load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as handle:
        return {k: np.asarray(handle[k]) for k in handle.files}


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _cmd(args: Sequence[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return None


def git_commit() -> str | None:
    return _cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"])


def git_state() -> dict[str, Any]:
    status = _cmd(["git", "-C", str(REPO), "status", "--porcelain"])
    return {"commit": git_commit(), "dirty": bool(status) if status is not None else None}


def environment() -> dict[str, Any]:
    import torch

    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "git_commit": git_commit(),
    }
    if torch.cuda.is_available():
        out["cuda_device_name"] = torch.cuda.get_device_name(0)
        out["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return out


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# frozen configuration payload
# ---------------------------------------------------------------------------


def config_payload() -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "protocol_id": PROTOCOL_ID,
        "frozen_utc": "2026-11-25T00:00:00Z",
        "seed": SEED,
        "new_arms": list(NEW_ARMS),
        "training_queue": list(TRAIN_QUEUE),
        "lipo": {
            "dataset": "ogbg-mollipo",
            "split": "official OGB scaffold split (train/valid/test = 3360/420/420)",
            "split_source": "data/ogb/ogbg_mollipo/split/scaffold/{train,valid,test}.csv.gz",
            "reused_a_arm": track_rel(LIPO_A_DIR),
            "a_expected": LIPO_A_EXPECTED,
            "gine_reference": LIPO_GINE_REFERENCE,
            "recipe": LIPO_RECIPE,
            "width": LIPO_WIDTH,
            "stat_rank": LIPO_STAT_RANK,
            "full_c1_parameters": LIPO_FULL_C1_PARAMS,
            "test_role": LIPO_TEST_ROLE,
        },
        "pcba": {
            "dataset": "ogbg-molpcba",
            "split": "official OGB scaffold split (train/valid/test = 350343/43793/43793)",
            "split_source": "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1/analysis/pcba_split.json (generated by OGB GraphPropPredDataset.get_idx_split())",
            "reused_w2": track_rel(W2_DIR),
            "reused_fixed1": track_rel(FIXED1_DIR),
            "w2_expected": W2_EXPECTED,
            "fixed1_expected": FIXED1_EXPECTED,
            "dynamic_reference": DYNAMIC_REFERENCE,
            "recipe": PCBA_RECIPE,
            "width": PCBA_WIDTH,
            "stat_rank": PCBA_STAT_RANK,
            "n_tasks": PCBA_N_TASKS,
            "w2_parameters": PCBA_W2_PARAMS,
            "fixed1_parameters": PCBA_FIXED1_PARAMS,
            "test_role": PCBA_TEST_ROLE,
        },
        "support": {
            "grouping": "p = path_cat, e = path_end63 (production S2 integer codes)",
            "omega": "distinct (p,e) pairs that occur in the official Train split only",
            "weights": "type-equal: every Omega element has weight 1",
            "design": "bipartite incidence matrix X with one row per Omega element and two unit columns (p, e)",
            "svd": "float64 reduced SVD of X; keep s > 1e-10 * s_max; P = U U^T",
            "joint_arm": "J(p,e) = J_table[index(p,e)]",
            "additive_arm": "J_table = U Theta (orthogonal marginal projection only)",
            "fallback": {
                "same_component": "least-squares marginal fit Theta_hat[p] + Theta_hat[e]",
                "otherwise": "type-equal mean over Omega rows of the current J_table",
            },
            "unseen_policy": "no new learnable parameter",
        },
        "interpretation": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "molecule_bootstrap_replicates": MOLECULE_BOOTSTRAP_REPLICATES,
            "support_min": SUPPORT_MIN,
            "extreme_k": list(EXTREME_K),
            "test_ap_targets": list(TEST_AP_TARGETS),
            "reference_delta": REFERENCE_DELTA,
        },
        "tolerances": TOLERANCES,
        "budget": BUDGET,
        "reference_values": {
            "lipo": {
                "C1": {"valid_rmse_raw": LIPO_A_EXPECTED["valid_rmse_raw"], "test_rmse_raw": LIPO_A_EXPECTED["test_rmse_raw"]},
                "GINE": LIPO_GINE_REFERENCE,
            },
            "pcba": {
                "W2_ORIGINAL": {"valid_ap": W2_EXPECTED["best_valid_ap"], "test_ap": W2_EXPECTED["test_ap_best"]},
                "FIXED1_ORIGINAL": {"valid_ap": FIXED1_EXPECTED["best_valid_ap"], "test_ap": FIXED1_EXPECTED["test_ap_best"]},
                "DYNAMIC": DYNAMIC_REFERENCE,
            },
        },
    }


__all__ = [name for name in dir() if not name.startswith("_")]
