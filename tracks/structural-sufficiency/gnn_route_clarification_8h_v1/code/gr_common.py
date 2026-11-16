"""Identity, frozen design constants, paths, hashing and IO for
``gnn-route-clarification-8h-v1``.

Nothing scientific is decided here: every scientific constant is copied from the
task brief (``GNN_route_8h_Agent_prompt_v1``) and hashed into
``analysis/protocol_freeze.json`` by :mod:`gr_prepare`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "gnn-route-clarification-8h-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID
DESIGN_BASIS_COMMIT = "7de32adf4c8941b7e5e0389738fe82c0f18586b2"
MAINLINE_BASIS_COMMIT = "9b3dba098915b0cdd800767f61b523681ee9e25d"

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
LEDGER_DIR = ANALYSIS_DIR / "ledgers"

#: Read-only reuse roots (never written by this round).
RC_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1"
RC_CODE = RC_DIR / "code"
RC_RESULTS = RC_DIR / "results"
SQL_DIR = REPO_ROOT / "tracks/structural-query-learning"
SQL_CODE = SQL_DIR / "code"
SQL_DATA = SQL_DIR / "data"
PREP_CODE = REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code"
DISSECT_CODE = REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code"
EDIT_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1"
EDIT_ANALYSIS = EDIT_DIR / "analysis"

DATASETS: tuple[str, ...] = ("ogbg-molesol", "ogbg-mollipo")
SPLITS: tuple[str, ...] = ("train", "valid", "test")
EXPECTED_SPLIT_SIZES = {
    "ogbg-molesol": {"train": 902, "valid": 113, "test": 113},
    "ogbg-mollipo": {"train": 3360, "valid": 420, "test": 420},
}
EXPECTED_TOTAL_ROWS = {"ogbg-molesol": 1128, "ogbg-mollipo": 4200}
EXPECTED_CACHE_SHA256 = {
    "ogbg_molesol_sq-pilot-cache-v1.npz": "3b7a8e16ca9c7ab240add84fbe26050dc81e22359bcc36f1c8a8111b77aec24a",
    "ogbg_mollipo_sq-pilot-cache-v1.npz": "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c",
    "ogbg_molesol_sq-pilot-lipo-raw-v1.npz": "a3ab9f7e22798a8f6fdc7f0913b1136b7cf59d4ffc7168faecc7f8b3094710b1",
    "ogbg_mollipo_sq-pilot-lipo-raw-v1.npz": "3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3",
}

TRAIN_SEED = 0
BOOTSTRAP_SEED = 1703
INIT_SEED_CANONICAL = 0          # C1 canonical state and the G0 canonical state
INIT_SEED_GX = 0                 # isolated stream for the Gx input projections + head
INIT_SEED_HEAD = 0               # isolated stream for every new additive/dense readout head

EPOCHS = 360
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
DROPOUT = 0.10
GRAD_CLIP = 5.0
MIN_LR = 1e-5
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
#: A raw global column is active when its Train span exceeds this (frozen).
ACTIVE_SPAN_EPS = 1e-8

#: The two frozen recipe packs (task brief 4.1).  The whole pack is compared;
#: no single component is attributed.
RECIPES: dict[str, dict[str, Any]] = {
    "R3": {
        "name": "R3",
        "peak_lr": 3.0e-4,
        "weight_decay": 1.0e-3,
        "warmup_epochs": 0,
        "scheduler": "per_step_cosine_no_warmup",
    },
    "Rf": {
        "name": "Rf",
        "peak_lr": 1.0e-3,
        "weight_decay": 1.0e-5,
        "warmup_epochs": 5,
        "scheduler": "per_step_linear_warmup_then_cosine",
    },
}
RECIPE_ORDER: tuple[str, ...] = ("R3", "Rf")

ARCHS: tuple[str, ...] = ("L", "G0", "Gx")     # stage 1
ARCH_LABEL = {"L": "LCRC-C1", "G0": "GINE-original", "Gx": "GINE-bridge"}

#: Parameter budgets frozen by the brief.
LCRC_C1_PARAMS = 228674
G0_READOUT_PARAMS = 33153
GX_READOUT_PARAMS = 38529
ADDITIVE_HIDDEN = 128
UNIVARIATE_WIDTH = 16          # 48 parameters per single-variable function

#: Stage-2/3 readout family A (end-to-end) and D (frozen representation).
STAGE1 = "stage1"
STAGE2 = "stage2_A"
STAGE3 = "stage3_D"
STAGE_ORACLE = "oracle"
STAGE_HIV = "molhiv"

ORACLE_PROPERTIES: tuple[str, ...] = ("WC_LOGP", "MOLWT")
ORACLE_MODELS: tuple[str, ...] = ("L", "T", "A")

#: Decision thresholds fixed by the brief (never re-tuned after seeing results).
A_GATE_RATIO = 1.05
A_POINT_RATIO = 1.05
BRIDGE_R_GEO = 0.99
BRIDGE_R_MAX = 1.10
D_FIDELITY_SIGMA = 0.02
D_REAL_RATIO = 1.05
ORACLE_ACCURACY_SIGMA = 0.10
ORACLE_COMPARABILITY_SIGMA = 0.02
EXPLANATION_IMPROVEMENT = 0.10
IG_TOL_REL = 1e-4
IG_TOL_ABS = 1e-3
LOW_SUPPORT_COMPONENTS = 30
MAX_EXPLAIN_NONTRAIN = 256
MAX_EXPLAIN_TRAIN = 64
MMP_MAIN_MODELS: tuple[str, ...] = ("L", "T", "A", "D")

SPLIT_FILE = {"train": "train_idx", "valid": "valid_idx", "test": "test_idx"}


# ---------------------------------------------------------------------------
# hashing / atomic IO
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
    import torch

    digest = hashlib.sha256()
    names = sorted(state) if keys is None else sorted(keys)
    for name in names:
        value = state[name]
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path | str, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_npz(path: Path | str, payload: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(path: Path | str, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# metrics / seeds (same semantics as the previous round)
# ---------------------------------------------------------------------------


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(d)))


def set_global_seeds(seed: int) -> dict[str, Any]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    return {
        "seed": int(seed),
        "numpy": int(np.random.get_state()[1][0]),
        "python": random.getstate()[1][0],
        "torch": int(torch.initial_seed()),
    }


def rng_state_hash() -> str:
    import torch

    digest = hashlib.sha256()
    digest.update(np.random.get_state()[1].tobytes())
    digest.update(str(random.getstate()[1]).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        digest.update(torch.cuda.get_rng_state_all()[0].cpu().numpy().tobytes())
    return digest.hexdigest()


def training_rng_payload() -> dict[str, Any]:
    import torch

    return {
        "np": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_payload(payload: Mapping[str, Any]) -> None:
    import torch

    np.random.set_state(payload["np"])
    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch"])
    if payload.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda"])


# ---------------------------------------------------------------------------
# frozen recipes
# ---------------------------------------------------------------------------


def steps_per_epoch(n_train: int) -> int:
    return int(np.ceil(int(n_train) / BATCH_SIZE))


def total_steps(n_train: int, epochs: int = EPOCHS) -> int:
    return int(epochs) * steps_per_epoch(n_train)


def lr_at(recipe: str, step: int, n_train: int, epochs: int = EPOCHS) -> float:
    """Exact frozen schedule: ``step`` is the 0-based optimizer step."""
    cfg = RECIPES[recipe]
    peak = float(cfg["peak_lr"])
    spe = steps_per_epoch(n_train)
    total = int(epochs) * spe
    warm = int(cfg["warmup_epochs"]) * spe
    if warm > 0 and step < warm:
        return peak * (0.1 + 0.9 * float(step) / float(warm))
    start = step - warm if warm > 0 else step
    span = max(1, total - warm)
    progress = min(1.0, max(0.0, float(start) / float(span)))
    return MIN_LR + 0.5 * (peak - MIN_LR) * (1.0 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# period / environment bookkeeping
# ---------------------------------------------------------------------------


def environment() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": torch.version.cuda,
        "hostname": platform.node(),
        "cwd": os.getcwd(),
    }
    try:
        import torch_geometric

        out["torch_geometric"] = torch_geometric.__version__
    except Exception as exc:  # pragma: no cover
        out["torch_geometric"] = f"unavailable: {exc}"
    try:
        import rdkit

        out["rdkit"] = rdkit.__version__
    except Exception as exc:  # pragma: no cover
        out["rdkit"] = f"unavailable: {exc}"
    if torch.cuda.is_available():
        out["devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "uuid": str(torch.cuda.get_device_properties(i).uuid),
                "total_memory": int(torch.cuda.get_device_properties(i).total_memory),
            }
            for i in range(torch.cuda.device_count())
        ]
        out["current_device"] = int(torch.cuda.current_device())
        # physical identity must be read from the environment mapping, not the pool index
        out["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    out["git"] = git_state()
    return out


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=20)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def code_hashes() -> dict[str, str]:
    out = {}
    for path in sorted(CODE_DIR.glob("gr_*.py")):
        out[f"code/{path.name}"] = sha256_file(path)
    for name in ("cr_data.py", "cr_models.py", "cr_train.py", "cr_common.py"):
        path = RC_CODE / name
        if path.is_file():
            out[f"reused/lcrc_readout_context_pilot_small_v1/{name}"] = sha256_file(path)
    for name in ("sq_data.py", "sq_lcrc.py", "sq_gine.py", "sq_common.py"):
        path = SQL_CODE / name
        if path.is_file():
            out[f"reused/structural-query-learning/{name}"] = sha256_file(path)
    for name in ("feature_spec.py", "adapters.py", "molecular_graph.py", "lcrc_data.py"):
        path = PREP_CODE / name
        if path.is_file():
            out[f"reused/lcrc_cross_task_preparation_v1/{name}"] = sha256_file(path)
    for name in ("dissect_common.py", "dissect_model.py"):
        path = DISSECT_CODE / name
        if path.is_file():
            out[f"reused/lcrc_lipo_model_dissection_v1/{name}"] = sha256_file(path)
    return out


def round_dir() -> Path:
    return ROUND_DIR


def results_dir() -> Path:
    return RESULTS_DIR


def task_dir(stage: str, dataset: str, model: str, recipe: str) -> Path:
    base = RESULTS_DIR / stage / dataset / model
    return base / recipe if recipe else base


# ---------------------------------------------------------------------------
# task catalog
# ---------------------------------------------------------------------------


def stage1_tasks() -> list[dict[str, Any]]:
    out = []
    for dataset in DATASETS:
        for arch in ARCHS:
            for recipe in RECIPE_ORDER:
                out.append(
                    {
                        "key": f"{STAGE1}:{dataset}:{arch}:{recipe}",
                        "stage": STAGE1,
                        "dataset": dataset,
                        "model": arch,
                        "recipe": recipe,
                        "epochs": EPOCHS,
                        "kind": "train",
                    }
                )
    return out


def stage2_tasks() -> list[dict[str, Any]]:
    return [
        {
            "key": f"{STAGE2}:{dataset}:A:{recipe}",
            "stage": STAGE2,
            "dataset": dataset,
            "model": "A",
            "recipe": recipe,
            "epochs": EPOCHS,
            "kind": "train",
            "depends": "stage1",
        }
        for dataset in DATASETS
        for recipe in RECIPE_ORDER
    ]


def stage3_tasks() -> list[dict[str, Any]]:
    return [
        {
            "key": f"{STAGE3}:{dataset}:D",
            "stage": STAGE3,
            "dataset": dataset,
            "model": "D",
            "recipe": "D600",
            "epochs": 600,
            "kind": "frozen_head",
            "depends": "stage1",
        }
        for dataset in DATASETS
    ]


def oracle_tasks() -> list[dict[str, Any]]:
    return [
        {
            "key": f"{STAGE_ORACLE}:{prop}:{model}",
            "stage": STAGE_ORACLE,
            "dataset": "ogbg-mollipo",
            "model": model,
            "recipe": "selected",
            "epochs": EPOCHS,
            "kind": "train",
            "property": prop,
            "depends": "stage1",
        }
        for prop in ORACLE_PROPERTIES
        for model in ORACLE_MODELS
    ]


def hiv_tasks() -> list[dict[str, Any]]:
    return [
        {
            "key": f"{STAGE_HIV}:ogbg-molhiv:{model}",
            "stage": STAGE_HIV,
            "dataset": "ogbg-molhiv",
            "model": model,
            "recipe": "selected",
            "epochs": 120,
            "kind": "binary",
            "depends": "stage1",
        }
        for model in ("L", "T", "A")
    ]


def all_tasks() -> list[dict[str, Any]]:
    return stage1_tasks() + stage2_tasks() + stage3_tasks() + oracle_tasks()


def task_status_path(task: Mapping[str, Any]) -> Path:
    return task_dir(task["stage"], task["dataset"], task["model"], task["recipe"]) / "status.json"


def task_complete(task: Mapping[str, Any]) -> bool:
    path = task_status_path(task)
    if not path.is_file():
        return False
    try:
        status = read_json(path)
    except Exception:
        return False
    return status.get("state") == "completed"
