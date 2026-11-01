"""Shared paths, frozen constants and provenance helpers for ``lcrc_esol_quickstart_v1``.

This round asks one question: does the *original* LCRC-A structural
representation (explicit S1 neighbour statistics + explicit S2 two-hop typed
paths + local contribution readout + global ``rho``) learn ESOL / Delaney
water solubility reasonably, and what does a single, pre-justified change buy?

Everything structural is **reused read-only** from the frozen preparation round
``lcrc_cross_task_preparation_v1``:

* ``feature_spec.lipo_v1_spec`` — the atom/bond/global/S1/S2 schema;
* ``adapters.lipo_row_to_graph`` — raw-graph row -> canonical molecular graph;
* ``lcrc_data`` — S1/S2 assembly, Fit-only normalisation, collation;
* ``lcrc_model.LCRC`` — the unchanged LCRC-A computational graph;
* ``cpu_gates.reference_enumeration`` — the independent S1/S2 reference.

Nothing in a historical directory is written.  The ESOL dataset is new: it is
downloaded once from the official DeepChem location, hashed, converted to a
label-free raw-graph artifact with the *same* RDKit encoding used for the
Lipophilicity round, and frozen.

RDKit is only needed for (a) the one-off raw-graph/scaffold/physchem
preparation and (b) the split audit.  Both run locally on CPU.  Training never
imports RDKit, because the GPU hosts do not install it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

# --- read-only reused code roots --------------------------------------------
PREP_ROUND = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_ROUND / "code"
LIPOPHILICITY_CODE = REPO / "tracks/structure-contribution/code"

for _path in (str(CODE_DIR), str(PREP_CODE), str(LIPOPHILICITY_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import lcrc_common as PREP_COMMON  # noqa: E402  (paths/hashing only)

# ---------------------------------------------------------------------------
# round identity
# ---------------------------------------------------------------------------

ROUND_ID = "lcrc-esol-quickstart-v1"
PROTOCOL_ID = ROUND_ID

#: Bemis--Murcko scaffold split seed, fixed by the task book.
SPLIT_SEED = 20261103
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)

# ---------------------------------------------------------------------------
# ESOL source
# ---------------------------------------------------------------------------

N_MOLECULES = 1128
SOURCE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
LOADER_URL = "https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/delaney_datasets.py"

#: SHA256 of the downloaded file, pinned before any modelling.
CSV_SHA256 = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"

CSV_COLUMNS = (
    "Compound ID",
    "ESOL predicted log solubility in mols per litre",
    "Minimum Degree",
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Polar Surface Area",
    "measured log solubility in mols per litre",
    "smiles",
)

ID_COLUMN = "Compound ID"
TARGET_COLUMN = "measured log solubility in mols per litre"
SMILES_COLUMN = "smiles"

#: Never used as an input, a target, or a selection signal.
FORBIDDEN_COLUMNS = (
    "ESOL predicted log solubility in mols per litre",
)
#: Pre-computed physchem columns of the CSV.  Only ever read by the *audit*
#: that cross-checks our RDKit recomputation; never used as a model input.
CSV_DESCRIPTOR_COLUMNS = (
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rotatable Bonds",
    "Polar Surface Area",
)

PHYSCHEM_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
)

#: The six RDKit descriptors, with the exact RDKit call each one uses.
PHYSCHEM_DEFINITION = {
    "MolWt": "rdkit.Chem.Descriptors.MolWt",
    "MolLogP": "rdkit.Chem.Crippen.MolLogP",
    "TPSA": "rdkit.Chem.Descriptors.TPSA",
    "HBD": "rdkit.Chem.Lipinski.NumHDonors",
    "HBA": "rdkit.Chem.Lipinski.NumHAcceptors",
    "RotatableBonds": "rdkit.Chem.Lipinski.NumRotatableBonds",
}

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

ANALYSIS = EXP_DIR / "analysis"
SOURCE_DIR = ANALYSIS / "source"
SPLIT_DIR = ANALYSIS / "split"
P0_DIR = ANALYSIS / "p0"
RUNS = ANALYSIS / "runs"
EVAL = ANALYSIS / "eval"
SMOKE = ANALYSIS / "smoke"
CACHE = EXP_DIR / "cache"
CONFIGS = EXP_DIR / "configs"

CSV_PATH = SOURCE_DIR / "delaney-processed.csv"
RAW_GRAPH = SOURCE_DIR / "esol_raw_v1.json.gz"
PHYSCHEM_FILE = SOURCE_DIR / "esol_physchem_v1.json"
SCAFFOLD_FILE = SOURCE_DIR / "esol_scaffolds_v1.json"
SOURCE_MANIFEST = SOURCE_DIR / "source_manifest.json"

SPLIT_MANIFEST = SPLIT_DIR / "split_esol_scaffold_seed_20261103.json"
SPLIT_AUDIT = SPLIT_DIR / "split_audit.json"
P0_SUMMARY = P0_DIR / "p0_summary.json"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
MANIFEST = ANALYSIS / "manifest.json"
MANIFEST_FROZEN = ANALYSIS / "MANIFEST_FROZEN"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"
CONFIG_PATH = CONFIGS / "esol.yaml"
P2_HYPOTHESIS = EXP_DIR / "P2_HYPOTHESIS.md"

# ---------------------------------------------------------------------------
# models / arms
# ---------------------------------------------------------------------------

SEEDS = (0, 1, 2)

MODEL_DISPLAY = {
    "base": "ESOL-BASE",
    "drop20": "ESOL-DROP20",
    "physchem6": "ESOL-PHYSCHEM6",
}
MODEL_DROPOUT = {"base": 0.10, "drop20": 0.20, "physchem6": 0.10}

#: P1 is fixed by the task book; P2 is decided from the P1 evidence.
P1_MODELS = ("base",)
P2_CANDIDATES = ("drop20", "physchem6")


def arm_key(model: str, seed: int) -> str:
    return f"{model}_s{int(seed)}"


P1_ARMS = tuple(arm_key(m, s) for m in P1_MODELS for s in SEEDS)


def arm_parts(arm: str) -> tuple[str, int]:
    if "_s" not in arm:
        raise ValueError(f"bad arm name {arm!r}")
    model, seed = arm.rsplit("_s", 1)
    if model not in MODEL_DISPLAY:
        raise ValueError(f"unknown model {model!r}")
    return model, int(seed)


# ---------------------------------------------------------------------------
# frozen training recipe (identical to the validated R3 / best-epoch recipe
# of lcrc_lipo_best_epoch_distance_fix_v1, checkpoint rule included)
# ---------------------------------------------------------------------------

TRAIN_RECIPE = {
    "optimizer": "AdamW",
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1.0e-8,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "scheduler": "per_step_cosine",
    "weight_decay": 1.0e-3,
    "batch_size": 128,
    "epochs": 360,
    "gradient_clip": 5.0,
    "dropout": 0.10,
    "dropout_position": "final_atom_level_local_h_before_readout",
    "loss": "mse_on_train_standardised_target",
    "checkpoint": "single_best_validation_rmse_epoch",
    "early_stopping": False,
    "ssl": False,
}
SHUFFLE_OFFSET = 917_000
GPU_HARD_CAP_HOURS = 1.5

# ---------------------------------------------------------------------------
# generic helpers (identical conventions to the frozen rounds)
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(encoded)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return path


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def log(message: str) -> None:
    print(message, flush=True)


def utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def track_rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def rows_sha256(rows: Sequence[int] | np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(rows, dtype="<i8")).tobytes())


def vector_sha256_float64(values: Sequence[float] | np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())


def state_dict_fingerprint(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True
        )
        return bool(out.strip())
    except Exception:
        return True


def code_hashes() -> dict[str, str]:
    """This round's code plus the read-only reused modules it imports."""
    out = {f"esol/{path.name}": sha256_file(path) for path in sorted(CODE_DIR.glob("*.py"))}
    for name in (
        "feature_spec.py",
        "adapters.py",
        "lcrc_data.py",
        "lcrc_model.py",
        "molecular_graph.py",
        "lcrc_common.py",
        "cpu_gates.py",
    ):
        out[f"reused/prep/{name}"] = sha256_file(PREP_CODE / name)
    out["reused/lipo_adapter/bjoc_lipophilicity_adapter_v1.py"] = sha256_file(
        LIPOPHILICITY_CODE / "bjoc_lipophilicity_adapter_v1.py"
    )
    return out


def env_info() -> dict[str, Any]:
    import torch

    info = {
        "round_id": ROUND_ID,
        "timestamp": utcnow(),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        import networkx

        info["networkx"] = networkx.__version__
    except Exception:
        info["networkx"] = None
    try:
        import rdkit

        info["rdkit"] = rdkit.__version__
    except Exception:
        info["rdkit"] = None
    return info


def config_sha256() -> str:
    return sha256_file(CONFIG_PATH)


def load_config() -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if str(cfg.get("protocol_id")) != PROTOCOL_ID:
        raise ValueError("esol config protocol_id drift")
    if int(cfg.get("split_seed", -1)) != SPLIT_SEED:
        raise ValueError("esol config split_seed drift")
    training = dict(cfg["training"])
    for key, expected in TRAIN_RECIPE.items():
        if key not in training:
            raise ValueError(f"config missing training.{key}")
        if training[key] != expected:
            raise ValueError(f"config training.{key} drift: {training[key]!r} != {expected!r}")
    if float(cfg["budget"]["gpu_hard_cap_hours"]) != GPU_HARD_CAP_HOURS:
        raise ValueError("config GPU cap drift")
    return cfg


def run_dir(arm: str) -> Path:
    return RUNS / str(arm)


def spec():
    """The frozen ``lipo_v1`` FeatureSpec (unchanged structural schema)."""
    import feature_spec as PREP_SPEC

    return PREP_SPEC.lipo_v1_spec()


__all__ = [name for name in dir() if not name.startswith("_")]


# ---------------------------------------------------------------------------
# metrics in the original logS units
# ---------------------------------------------------------------------------


def _metric_pair(pred, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(y, dtype=np.float64).reshape(-1)
    if p.shape != t.shape:
        raise ValueError(f"prediction shape {p.shape} != target shape {t.shape}")
    if p.size == 0:
        raise ValueError("empty metric input")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("non-finite prediction or target")
    return p, t


def all_metrics(pred, y) -> dict[str, float]:
    p, t = _metric_pair(pred, y)
    err = p - t
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "n": int(t.size),
    }


# ---------------------------------------------------------------------------
# freeze discipline
# ---------------------------------------------------------------------------


def require_freeze() -> dict[str, Any]:
    if not PROTOCOL_FREEZE.exists():
        raise PermissionError("ESOL protocol freeze missing; run `run_esol.py freeze` first")
    frozen = read_json(PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("ESOL protocol freeze has the wrong protocol id")
    if frozen.get("config_sha256") != config_sha256():
        raise PermissionError("ESOL config changed after the freeze")
    if frozen.get("code_hashes") != code_hashes():
        raise PermissionError("ESOL code changed after the freeze")
    if frozen.get("split_manifest_sha256") != sha256_file(SPLIT_MANIFEST):
        raise PermissionError("ESOL split manifest changed after the freeze")
    if frozen.get("split_audit_sha256") != sha256_file(SPLIT_AUDIT):
        raise PermissionError("ESOL split audit changed after the freeze")
    return frozen


def freeze_protocol() -> dict[str, Any]:
    if not SPLIT_MANIFEST.exists() or not SPLIT_AUDIT.exists():
        raise PermissionError("ESOL split manifest/audit missing; run `split` first")
    audit = read_json(SPLIT_AUDIT)
    if not audit.get("passed"):
        raise PermissionError("ESOL split audit failed; refusing to freeze")
    cfg = load_config()
    payload = {
        "protocol_id": PROTOCOL_ID,
        "round_id": ROUND_ID,
        "created_utc": utcnow(),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "status": "FROZEN_BEFORE_ESOL_TRAINING",
        "scope": (
            "ESOL / Delaney (1,128 molecules), one fixed label-blind Bemis-Murcko scaffold "
            "partition (seed 20261103), P1 baseline + at most one P2 iteration"
        ),
        "split_seed": SPLIT_SEED,
        "split_fractions": list(SPLIT_FRACTIONS),
        "split_manifest": track_rel(SPLIT_MANIFEST),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "split_audit": track_rel(SPLIT_AUDIT),
        "split_audit_sha256": sha256_file(SPLIT_AUDIT),
        "config": track_rel(CONFIG_PATH),
        "config_sha256": config_sha256(),
        "code_hashes": code_hashes(),
        "training_recipe": dict(TRAIN_RECIPE),
        "model_dropout": dict(MODEL_DROPOUT),
        "models": sorted(MODEL_DISPLAY),
        "seeds": list(SEEDS),
        "p1_arms": list(P1_ARMS),
        "checkpoint_rule": "single_best_validation_rmse_epoch",
        "test_evaluation_policy": (
            "freeze every checkpoint hash before reading any Test label; Test labels are read "
            "exactly once, after the manifest freeze, and never used for selection"
        ),
        "planned_run_counts": {"p1": 3, "p2": "0 or 3", "hard_max": 6},
        "gpu_hard_cap_hours": GPU_HARD_CAP_HOURS,
        "test_labels_read_at_freeze_time": False,
    }
    payload["freeze_sha256"] = sha256_json({k: v for k, v in payload.items() if k != "freeze_sha256"})
    write_json(PROTOCOL_FREEZE, payload)
    return payload


def arms_with_runs() -> list[str]:
    if not RUNS.exists():
        return []
    out = []
    for path in sorted(RUNS.iterdir()):
        if (path / "train.json").exists():
            model, _seed = arm_parts(path.name)
            out.append(path.name)
    return out
