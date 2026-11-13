"""Frozen constants, paths, provenance and small IO helpers.

Round: ``lcrc-lipo-s3-locality-public122-s0-v1``.

The module only *imports* the upstream rounds read-only:

* ``lcrc_lipo_model_dissection_v1`` - public122 loader, frozen ``lipo_v1``
  structure semantics, ``DissectLCRC`` / ``build_full_state`` / ``remap_state``
  and the frozen metric / norm helpers.
* ``lcrc_cross_task_preparation_v1`` - schema, ``build_molecule``, ``collate``,
  Train-only normalisation.
* ``lcrc_molnet_benchmark_pilot_v1`` - committed label-free raw attributes.
* ``lcrc_local_edit_error_diagnosis_v1`` - labelled graph / isomorphism helpers.

Nothing upstream is monkey-patched or written to.  Every new artifact lives
under this round's directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-lipo-s3-locality-public122-s0-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

DISSECT = REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
EDIT = REPO / "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1"
EDIT_CODE = EDIT / "code"
PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
PILOT = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1"
PILOT_CODE = PILOT / "code"

#: the round that trained the ordinary C1 baseline reused here (read-only)
BASELINE_ROUND = REPO / "tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1"

for _p in (str(CODE_DIR), str(DISSECT_CODE), str(EDIT_CODE), str(PREP_CODE), str(PILOT_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dissect_common as DC  # noqa: E402
import dissect_data as DD  # noqa: E402
import dissect_model as DM  # noqa: E402
import dissect_source as DS  # noqa: E402
import edit_common as EC  # noqa: E402
import lcrc_common as PREP_COMMON  # noqa: E402
import lcrc_data as PREP_DATA  # noqa: E402

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"

# ---------------------------------------------------------------------------
# frozen data identity
# ---------------------------------------------------------------------------

DATA_REPEAT = 122
SPLITS = ("train", "valid", "test")
SPLIT_SIZES = (3360, 420, 420)
N_MOLECULES = 4200
LABEL_COLUMN = "exp"

LIPOP_CSV_SHA256 = DC.LIPOP_CSV_SHA256
SPLIT_FILE = "Lipop.random122.ind.pkl"
SPLIT_FILE_SHA256 = DC.SPLIT_FILE_SHA256[122]
SPLIT_ROWS_SHA256 = tuple(DC.SPLIT_ROWS_SHA256[122])
RAW_ARTIFACT = EC.RAW_ARTIFACT
RAW_ARTIFACT_SHA256 = EC.RAW_ARTIFACT_SHA256

FEATURE_SPEC_HASH = DC.FEATURE_SPEC_HASH
ATOM_FEAT_DIM = DC.ATOM_FEAT_DIM  # 51
BOND_FEAT_DIM = DC.BOND_FEAT_DIM  # 13
ATOM_SCALAR_DIM = DC.ATOM_SCALAR_DIM  # 9
GLOBAL_DIM = DC.GLOBAL_DIM  # 42
N_NBR_CODES = DC.N_NBR_CODES  # 80
N_PATH_CODES = DC.N_PATH_CODES  # 400
N_BOND_TYPES = DC.N_BOND_TYPES  # 5
N_ATOM_TYPES = 16

# ---------------------------------------------------------------------------
# frozen baseline (ordinary C1) identity
# ---------------------------------------------------------------------------

BASELINE_ARM_DIR = BASELINE_ROUND / "runs" / "A_s0"
BASELINE_BEST = BASELINE_ARM_DIR / "best.pt"
BASELINE_PER_MOLECULE = BASELINE_ROUND / "analysis" / "per_molecule.csv"
BASELINE_TRAIN_JSON = BASELINE_ARM_DIR / "train.json"
BASELINE_RESULTS = BASELINE_ROUND / "analysis" / "results.csv"

VARIANT = "C1"
VARIANT_NAME = DC.VARIANT_NAMES["C1"]  # NO_BOND_MEAN
BASE_PARAMS = 228674
CORRECTOR_PARAMS = 4704
EXPECTED_PARAMS = BASE_PARAMS + CORRECTOR_PARAMS  # 233378

BASELINE_EXPECTED = {
    "parameters": BASE_PARAMS,
    "best_epoch": 244,
    "valid_rmse_raw": 0.7025358558198603,
    "test_rmse_raw": 0.5761674771952362,
    "init_state_hash_prefix": "aac1e061",
}

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

ARMS = ("LOCAL3", "DUP2", "NULL3")
STAGE1_ARMS = ("LOCAL3", "DUP2")
ARM_DESCRIPTIONS = {
    "LOCAL3": "h' = h + psi(h, s3_v)  -- per-root exact distance-3 atom-type counts",
    "DUP2": "h' = h + psi(h, s2_v)   -- per-root S2 path-end-type counts (existing-information readout)",
    "NULL3": "h' = h + mean_{u in C(v)} psi(h, s3_u) -- same-molecule, same-atom-type mean field",
}
MAIN_COMPARISONS = (("LOCAL3", "A"), ("LOCAL3", "DUP2"))
NULL_COMPARISON = ("LOCAL3", "NULL3")

S3_DIM = N_ATOM_TYPES  # 16
CORRECTOR_HIDDEN = 32
CORRECTOR_INIT_SEED = 0

# ---------------------------------------------------------------------------
# frozen recipe (R3, unchanged)
# ---------------------------------------------------------------------------

RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "weight_decay": 1.0e-3,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "batch_size": 128,
    "dropout": 0.10,
    "gradient_clip": 5.0,
    "epochs": 360,
    "global_batch_size": 3360,
    "steps_per_epoch": 27,
    "total_optimizer_steps": 9720,
    "scheduler": "per_step_cosine",
    "shuffle_offset": 917_000,
    "early_stopping": False,
    "soup": False,
    "ema": False,
    "swa": False,
    "amp": False,
}

TRAIN_SEED = 0
SHUFFLE_SEED = int(RECIPE["shuffle_offset"])
BOOTSTRAP_SEED = 20261114
BOOTSTRAP_REPLICATES = 1000
DIAGNOSTIC_STEPS = (1, 28, 271, 2701)
EVAL_BATCH_SIZE = 256
REPRO_TOL = 1e-5
ALIAS_PRED_TOL = 2e-5

GPU_SOFT_CAP_HOURS = 1.0
GPU_HARD_CAP_HOURS = 2.0

# ---------------------------------------------------------------------------
# P0 alias candidates (0-based ChemBench row ids, frozen at design time)
# ---------------------------------------------------------------------------

ALIAS_PAIRS: tuple[tuple[int, int], ...] = (
    (294, 1665),
    (748, 913),
    (963, 4186),
    (1467, 2674),
    (2102, 3823),
)
#: design-time expectation: U3 separates these four, leaves 1467/2674 identical
ALIAS_EXPECTED_SEPARATED = ((294, 1665), (748, 913), (963, 4186), (2102, 3823))
ALIAS_EXPECTED_UNSEPARATED = (1467, 2674)
#: design-time A raw predictions (independent replication target, not a result)
ALIAS_EXPECTED_PRED = {
    294: 1.696551646,
    1665: 1.696551574,
    748: 2.821447260,
    913: 2.821446972,
    963: 0.371185149,
    4186: 0.371185581,
    1467: 3.597154216,
    2674: 3.597153784,
    2102: 1.047859854,
    3823: 1.047859781,
}


# ---------------------------------------------------------------------------
# hashing / IO
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


def array_sha256(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr, dtype=np.int64)
    return sha256_bytes(a.tobytes())


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
    return array_sha256(np.asarray(list(rows), dtype=np.int64))


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def run_name(arm: str, seed: int = TRAIN_SEED) -> str:
    return f"{arm}_s{int(seed)}"


def run_dir(arm: str, seed: int = TRAIN_SEED) -> Path:
    return RUNS / run_name(arm, seed)


def cache_path(split: str) -> Path:
    return CACHE / f"Lipophilicity_r{DATA_REPEAT}_{split}.pt"


def feature_path() -> Path:
    return CACHE / "s3_features.npz"


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


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
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


UPSTREAM_FILES = (
    ("upstream:dissect_common", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py"),
    ("upstream:dissect_model", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py"),
    ("upstream:dissect_data", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_data.py"),
    ("upstream:dissect_source", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_source.py"),
    ("upstream:edit_common", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/code/edit_common.py"),
    ("upstream:edit_graph", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/code/edit_graph.py"),
    ("upstream:lcrc_data", "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_data.py"),
    ("upstream:molecular_graph", "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/molecular_graph.py"),
    ("upstream:feature_spec", "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/feature_spec.py"),
    ("upstream:adapters", "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/adapters.py"),
    ("upstream:mb_rdkit", "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_rdkit.py"),
    ("upstream:mb_raw_artifact", "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_raw_artifact.py"),
    ("upstream:mb_data", "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_data.py"),
)


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for p in sorted(TESTS.glob("*.py")):
        out[f"tests/{p.name}"] = sha256_file(p)
    for p in sorted(CONFIGS.glob("*.yaml")):
        out[f"configs/{p.name}"] = sha256_file(p)
    for p in (TRACK / "PROTOCOL.md",):
        if p.exists():
            out[p.name] = sha256_file(p)
    for name, rel in UPSTREAM_FILES:
        out[name] = sha256_file(REPO / rel)
    return out


# ---------------------------------------------------------------------------
# source verification
# ---------------------------------------------------------------------------


class DataMismatch(RuntimeError):
    pass


def _resolve_by_hash(candidates: Sequence[Path], expected: str, label: str) -> Path:
    seen: set[Path] = set()
    for p in candidates:
        p = Path(p)
        if p in seen:
            continue
        seen.add(p)
        if p.exists() and sha256_file(p) == expected:
            return p
    present = [str(p) for p in seen if p.exists()]
    raise DataMismatch(f"DATA_OR_SPLIT_MISMATCH: no {label} with sha256 {expected}; checked {present}")


def resolve_source_files() -> dict[str, Path]:
    csv_candidates = [DC.LIPOP_CSV]
    csv_candidates += sorted(REPO.glob("tracks/structural-sufficiency/*/analysis/source/chembench/Lipop/Lipop.csv.gz"))
    split_candidates = [DC.CHEMBENCH_LIPOP / SPLIT_FILE]
    split_candidates += sorted(
        REPO.glob(f"tracks/structural-sufficiency/*/analysis/source/chembench/Lipop/{SPLIT_FILE}")
    )
    raw_candidates = [RAW_ARTIFACT]
    raw_candidates += sorted(REPO.glob("tracks/structural-sufficiency/*/analysis/source/raw_attributes_v1.json.gz"))
    return {
        "csv": _resolve_by_hash(csv_candidates, LIPOP_CSV_SHA256, "Lipop.csv.gz"),
        "split": _resolve_by_hash(split_candidates, SPLIT_FILE_SHA256, "Lipop.random122.ind.pkl"),
        "raw": _resolve_by_hash(raw_candidates, RAW_ARTIFACT_SHA256, "raw_attributes_v1.json.gz"),
    }


def verify_source() -> dict[str, Any]:
    paths = resolve_source_files()
    src = DS.load_lipop(DATA_REPEAT, labels="none")
    rows = src["rows"]
    if tuple(len(rows[s]) for s in SPLITS) != SPLIT_SIZES:
        raise DataMismatch(f"DATA_OR_SPLIT_MISMATCH: split sizes {tuple(len(rows[s]) for s in SPLITS)} != {SPLIT_SIZES}")
    for split, expected in zip(SPLITS, SPLIT_ROWS_SHA256):
        got = row_sha256(rows[split])
        if got != expected:
            raise DataMismatch(f"DATA_OR_SPLIT_MISMATCH: {split} row-index sha256 {got} != {expected}")
    if src["csv_sha256"] != LIPOP_CSV_SHA256:
        raise DataMismatch("DATA_OR_SPLIT_MISMATCH: CSV hash")
    if src["split_file_sha256"] != SPLIT_FILE_SHA256:
        raise DataMismatch("DATA_OR_SPLIT_MISMATCH: split-file hash")
    from rdkit import rdBase  # local-only dependency

    return {
        "paths": {k: track_rel(v) for k, v in paths.items()},
        "csv_sha256": LIPOP_CSV_SHA256,
        "split_file": SPLIT_FILE,
        "split_file_sha256": SPLIT_FILE_SHA256,
        "split_rows_sha256": list(SPLIT_ROWS_SHA256),
        "split_sizes": list(SPLIT_SIZES),
        "n_molecules": N_MOLECULES,
        "source_commit": DC.CHEMBENCH_COMMIT,
        "label_column": LABEL_COLUMN,
        "raw_artifact_sha256": RAW_ARTIFACT_SHA256,
        "rdkit": str(rdBase.rdkitVersion),
    }


def source_with_labels(scope: str = "all") -> dict[str, Any]:
    if scope not in ("none", "train_valid", "all"):
        raise ValueError(scope)
    return DS.load_lipop(DATA_REPEAT, labels=scope)


__all__ = [name for name in dir() if not name.startswith("_")]
