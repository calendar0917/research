"""Frozen constants, paths, hashes and the pre-registered decision rules.

Round: ``lcrc-lipo-route-clarification-v1`` (v2, 2026-11-12, Asia/Shanghai).

This module is the single source of truth for every scientific constant of the
round.  It is written *before* any formal run and must not be edited while
formal training is in flight; analysis-only additions live in
``lr_analysis.py``.

The round is consumed read-only from:

* ``lcrc_local_edit_error_diagnosis_v1`` -- the frozen 509 real edit pairs, the
  S0 component split, the label-access discipline and the edit table;
* ``lcrc_lipo_model_dissection_v1`` -- the frozen C1 (``NO_BOND_MEAN``) model,
  its FULL->C1 remap and the structure pipeline;
* ``lcrc_failure_mechanism_p0_v1`` -- label-free radius-2 signatures / Morgan
  fingerprints / support features.

No function here writes into an upstream directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROUND_ID = "lcrc-lipo-route-clarification-v1"
PROTOCOL_ID = ROUND_ID
VERSION = "v2"
DESIGN_ANCHOR_COMMIT = "ecb28cf1789c4b7369ff1041ef6d2defe11865a7"
TASK_ID = "lcrc-lipo-route-clarification-v1"

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

SS = REPO / "tracks/structural-sufficiency"
EDIT_ROUND = SS / "lcrc_local_edit_error_diagnosis_v1"
DISSECT = SS / "lcrc_lipo_model_dissection_v1"
P0 = SS / "lcrc_failure_mechanism_p0_v1"
PILOT = SS / "lcrc_molnet_benchmark_pilot_v1"

for _p in (
    str(CODE_DIR),
    str(EDIT_ROUND / "code"),
    str(DISSECT / "code"),
    str(P0 / "code"),
    str(SS / "lcrc_cross_task_preparation_v1/code"),
    str(PILOT / "code"),
):
    if _p not in __import__("sys").path:
        __import__("sys").path.insert(0, _p)

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"

# ---------------------------------------------------------------------------
# data identity (inherited frozen bytes; asserted against the files at use)
# ---------------------------------------------------------------------------

DATA_REPEAT = 122
LIPOP_CSV_SHA256 = "34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758"
SPLIT_FILE_SHA256 = "bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d"
TRAIN_ROWS_SHA256 = "4e8d42a95c0b705f43f932b3bbb265277c76a7aa65684b0e5e6ca028580aaf96"
RAW_ARTIFACT_SHA256 = "1127607de45e341c89ffc67db85714117831ec84b0845c75aaa11005489c51be"
TRAIN_ROWS = 3360
VALID_ROWS = 420
TEST_ROWS = 420
FEATURE_SPEC_HASH = "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee"

#: Frozen edit-pair table of the upstream round (read-only).
PAIRS_CSV = EDIT_ROUND / "analysis/pairs.csv"
PAIRS_CSV_SHA256 = "98db43ec1f6f2d712d4ae59ed6a832a0860c54fec230a85ad5fcb303561ae356"
S0_SPLIT_JSON = EDIT_ROUND / "analysis/component_split.json"
S0_SPLIT_JSON_SHA256 = ""

AUDIT_SCOPE = "WITHIN_DATASET_PROSPECTIVE_AUDIT"

# ---------------------------------------------------------------------------
# label-access purposes (enforced in code, not by convention)
# ---------------------------------------------------------------------------

#: ``purpose -> partitions whose labels that purpose may read``
PURPOSE_SCOPE = {
    "train": ("fit",),
    "select": ("fit", "select"),
    "diagnose": ("fit", "select", "diagnose"),
    "audit_score": ("fit", "select", "diagnose", "audit"),
}

# ---------------------------------------------------------------------------
# the five component splits
# ---------------------------------------------------------------------------

SPLITS = ("s0", "s1", "s2", "sg")
PARTITIONS = ("fit", "select", "diagnose", "audit")
FRACTIONS = {"fit": 0.60, "select": 0.10, "diagnose": 0.15, "audit": 0.15}
SPLIT_SALTS = {
    "s1": "lipo-route-v1-split-20261112-1",
    "s2": "lipo-route-v1-split-20261112-2",
}
SG_SALT_PREFIX = "lipo-route-v1-SG|"
SG_FRACTIONS = {"fit": 0.60, "select": 0.10, "diagnose": 0.15, "audit": 0.15}
SG_MIN_ROWS = {"fit": 1680, "select": 200, "diagnose": 300, "audit": 300}
CONTEXT_SALT_PREFIX = "lipo-route-v1-context"

#: Editorial (v2) minimums for an edit-confirmation partition.
MIN_EDIT_PAIRS = 40
MIN_EDIT_COMPONENTS = 15
MIN_FIT_COMPONENTS = 30

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

MODELS = ("B", "S", "O", "D", "F", "G")
LOSSES = ("A", "R", "W", "N")
SEEDS = (0, 1, 2)
MODEL_NAMES = {
    "B": "C1 base (no residual blocks)",
    "S": "static capacity (c_t = h_t)",
    "O": "fixed context (c_t = T^t h0)",
    "D": "dynamic composition (c_t = T h_t)",
    "F": "Morgan-2048 + g42 MLP",
    "G": "GINE-style graph reference",
}
EXPECTED_PARAMS = {"B": 228674, "S": 278722, "O": 278722, "D": 278722, "F": 207009, "G": 251781}
#: branches that may be deployment candidates (D/G are references only)
DEPLOY_ELIGIBLE = ("B", "S", "O")

#: The five models used in the S0 loss-purchase screen.
PURCHASE_MODELS = ("B", "S", "O", "D", "F")

# ---------------------------------------------------------------------------
# recipe (frozen R3, inherited verbatim)
# ---------------------------------------------------------------------------

RECIPE = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-3,
    "batch_size": 128,
    "dropout": 0.10,
    "gradient_clip": 5.0,
    "scheduler": "per_step_cosine",
    "shuffle_offset": 917_000,
    "epochs": 360,
    "pair_batch": 16,
    "soup": False,
    "ema": False,
    "ssl": False,
    "early_stopping": False,
    "hyperparameter_search": False,
}
SHUFFLE_OFFSET = RECIPE["shuffle_offset"]
FORMAL_UPDATES = 5760
EPOCHS = RECIPE["epochs"]
BATCH = RECIPE["batch_size"]
PAIR_BATCH = RECIPE["pair_batch"]

#: SG / learning-curve variants keep ``updates`` fixed at 5760.
SG_UPDATES = 5760
SG_EVAL_PERIOD = 16
LC_UPDATES = 5760
LC_LABELS = {"half": 1008, "three_quarter": 1512}
LC_STEPS = {"half": 8, "three_quarter": 12}
LC_EPOCHS = {"half": 720, "three_quarter": 480}
FT_UPDATES = 9720

# ---------------------------------------------------------------------------
# doses (one-shot calibration, never re-tuned)
# ---------------------------------------------------------------------------

DOSE_TARGET_FRACTION = 0.2
DOSE_CLIP = (0.001, 10.0)
DOSE_BATCHES = 32
DOSE_PROBE_STEPS = (1, 100, 1000)

# ---------------------------------------------------------------------------
# statistical / decision constants (pre-registered resource rules)
# ---------------------------------------------------------------------------

BOOTSTRAP_SEED = 20261112
BOOTSTRAP_REPLICATES = 2000

PURCHASE_MIN_GM = 0.010           # RMSE(A) - RMSE(R) averaged over seeds
PURCHASE_EDIT_FLOOR = 0.020       # de = max(0.020, 0.05 * mean PairMAE(A))
PURCHASE_EDIT_FRACTION = 0.05
PURCHASE_MIN_SEEDS = 2            # at least 2/3 seeds positive on both metrics

MATERIAL_MOL = 0.010
MATERIAL_EDIT_FLOOR = 0.020
MATERIAL_EDIT_FRACTION = 0.05
CONTROL_INCREMENT_MOL = 0.005
CONTROL_INCREMENT_EDIT = 0.010
PAIR_NONINFERIOR = -0.010
REPLICATION_RULE = "4 of 6 split-seed units positive AND both split means positive AND bootstrap LB > 0"

DEPLOY_MIN_RMSE_GAIN = 0.010
DEPLOY_MIN_POSITIVE_UNITS = 4
DEPLOY_EDIT_SLACK = 0.010
DEPLOY_TIE_TOL = 0.002
CONTEXT_MIN_COMPARISONS = 20

#: Budget (wall clock hours from task acceptance, GPU A100-hours).
WALL_TARGET_HOURS = 7.5
WALL_HARD_HOURS = 8.0
GPU_SOFT_CAP_HOURS = 14.0
GPU_HARD_CAP_HOURS = 16.0


# ---------------------------------------------------------------------------
# hashing / IO helpers (same conventions as the upstream rounds)
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def row_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def stable_id(rows: Iterable[int]) -> str:
    """Stable component id: sorted global row ids joined with commas."""
    return ",".join(str(int(r)) for r in sorted(int(x) for x in rows))


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def env_info() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def rdkit_version() -> str:
    try:
        from rdkit import rdBase

        return str(rdBase.rdkitVersion)
    except Exception:
        return "unavailable"


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for name, rel in (
        ("edit_common", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/code/edit_common.py"),
        ("edit_data", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/code/edit_data.py"),
        ("dissect_common", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py"),
        ("dissect_model", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py"),
        ("dissect_data", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_data.py"),
        ("dissect_source", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_source.py"),
        ("p0_structure", "tracks/structural-sufficiency/lcrc_failure_mechanism_p0_v1/code/p0_structure.py"),
        ("feature_spec", "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/feature_spec.py"),
        ("lcrc_data", "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_data.py"),
        ("pairs_csv", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/analysis/pairs.csv"),
        ("s0_split", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/analysis/component_split.json"),
    ):
        p = REPO / rel
        out[f"upstream:{name}"] = sha256_file(p) if p.exists() else ""
    for p in sorted((TRACK / "configs").glob("*")):
        if p.is_file():
            out[f"configs/{p.name}"] = sha256_file(p)
    return out


# ---------------------------------------------------------------------------
# run identity
# ---------------------------------------------------------------------------


def run_id(split: str, model: str, loss: str, seed: int, stage: str | None = None, tag: str | None = None) -> str:
    """Canonical unique run id: ``<split>[_<stage>]_<model>_<loss>_s<seed>``."""
    parts = [split]
    if stage:
        parts.append(stage)
    parts += [model, loss, f"s{int(seed)}"]
    if tag:
        parts.append(tag)
    return "_".join(parts)


def run_dir(rid: str) -> Path:
    return RUNS / rid


__all__ = [n for n in dir() if not n.startswith("_")]
