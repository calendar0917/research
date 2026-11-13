"""Frozen constants, paths, provenance and small IO helpers.

Round: ``lcrc-lipo-edit-supervision-public122-s0-v1``.

This module only *imports* the upstream rounds.  It does not monkey-patch them,
does not write into their directories, and does not inherit their output paths.
The upstream files are consumed read-only:

* ``lcrc_lipo_model_dissection_v1`` - public122 loader, frozen lipo_v1
  structure cache semantics, C1 model and FULL -> C1 state remap.
* ``lcrc_local_edit_error_diagnosis_v1`` - labelled graph, single-bond cut
  enumeration, exact core isomorphism, component construction.

Every new artifact lives under this round's directory.
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

ROUND_ID = "lcrc-lipo-edit-supervision-public122-s0-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

DISSECT = REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
EDIT = REPO / "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1"
EDIT_CODE = EDIT / "code"

for _p in (str(CODE_DIR), str(DISSECT_CODE), str(EDIT_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ``dissect_common`` adds the frozen prep/pilot code dirs and carries the
# authoritative byte hashes of the public data.
import dissect_common as DC  # noqa: E402
import dissect_data as DD  # noqa: E402
import dissect_model as DM  # noqa: E402
import dissect_source as DS  # noqa: E402
import edit_common as EC  # noqa: E402
import edit_graph as EG  # noqa: E402
import edit_pairs as EP  # noqa: E402
import edit_split as ES  # noqa: E402

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"

# ---------------------------------------------------------------------------
# frozen data
# ---------------------------------------------------------------------------

DATA_REPEAT = 122
MAIN_REPEAT = 122
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
ATOM_FEAT_DIM = DC.ATOM_FEAT_DIM
BOND_FEAT_DIM = DC.BOND_FEAT_DIM
ATOM_SCALAR_DIM = DC.ATOM_SCALAR_DIM
GLOBAL_DIM = DC.GLOBAL_DIM
N_NBR_CODES = DC.N_NBR_CODES
N_PATH_CODES = DC.N_PATH_CODES
N_BOND_TYPES = DC.N_BOND_TYPES

# ---------------------------------------------------------------------------
# frozen model / arms
# ---------------------------------------------------------------------------

VARIANT = "C1"
VARIANT_NAME = DM.C.VARIANT_NAMES["C1"]  # NO_BOND_MEAN
EXPECTED_PARAMS = 228674
ARMS = ("A", "R", "W", "N")
ARM_DESCRIPTIONS = {
    "A": "ordinary molecule MSE only",
    "R": "ordinary molecule MSE + real-edit residual coupling",
    "W": "ordinary molecule MSE + endpoint reweighting (same endpoint weight as R)",
    "N": "ordinary molecule MSE + token-random-pair residual coupling",
}
MAIN_COMPARISON = ("R", "A")
MECHANISM_COMPARISONS = (("R", "W"), ("R", "N"))

# ---------------------------------------------------------------------------
# frozen training recipe (inherited R3, unchanged except the auxiliary term)
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
SCHEDULE_SEED = 20261113
BOOTSTRAP_SEED = 20261114
BOOTSTRAP_REPLICATES = 1000

AUX_PAIRS = 32
LAMBDA = 1.0
DIAGNOSTIC_STEPS = (1, 28, 271, 2701)
EVAL_BATCH_SIZE = 256
REPRO_TOL = 1e-5

GPU_SOFT_CAP_HOURS = 2.0
GPU_HARD_CAP_HOURS = 4.0

# ---------------------------------------------------------------------------
# P0 support gates (resource rule for this round only)
# ---------------------------------------------------------------------------

MIN_P_TRAIN_PAIRS = 100
MIN_P_TRAIN_COMPONENTS = 30
MIN_E_T_PAIRS = 40
MIN_E_T_COMPONENTS = 15
LOW_SUPPORT_PAIRS = 40
LOW_SUPPORT_COMPONENTS = 15

PAIR_RULES = {
    "eligibility": (
        "single connected component; no explicit H atoms; canonical SMILES without "
        "@ / backslash stereo tokens; every atom is a heavy atom"
    ),
    "cut": "non-ring single bonds only; RDKit FragmentOnBonds with one dummy (valence preserved)",
    "variable_fragment_heavy_atoms": (1, 4),
    "core_min_heavy_atoms": 10,
    "core_min_fraction_of_parent": 0.75,
    "atom_label": "element, is_aromatic, formal_charge, total_H, is_attachment",
    "bond_label": "SINGLE/DOUBLE/TRIPLE/AROMATIC/OTHER",
    "candidate_generation": "Weisfeiler-Lehman refinement hash (bucketing only)",
    "identity_check": "explicit labelled-graph isomorphism with attachment-point marker",
    "definition_choice": (
        "one unordered row-id pair -> maximum core_heavy, then core_smiles, then "
        "variable-fragment strings; final deterministic tie-break by cut indices"
    ),
    "direction": "smaller canonical variable-fragment string is A, larger is B",
    "forbidden_pair": "identical complete canonical molecule, or identical variable fragment",
    "pair_id": "p<min(row_a,row_b):05d>_<max(row_a,row_b):05d>",
    "training_set": "P_train = real pairs with both endpoints in the published Train 3360",
    "evaluation_set": "E_T = real pairs with at least one Test endpoint; E_HH = Valid-Test union Test-Test",
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


def rdkit_version() -> str:
    from rdkit import rdBase

    return str(rdBase.rdkitVersion)


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
    for rel in DC.REUSED_MODULES:
        out[rel] = sha256_file(REPO / rel)
    for name, rel in (
        ("upstream:dissect_common", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py"),
        ("upstream:dissect_model", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py"),
        ("upstream:dissect_data", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_data.py"),
        ("upstream:dissect_source", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_source.py"),
        ("upstream:edit_graph", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/code/edit_graph.py"),
        ("upstream:edit_pairs", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/code/edit_pairs.py"),
        ("upstream:edit_split", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/code/edit_split.py"),
        ("upstream:edit_common", "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/code/edit_common.py"),
    ):
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
    raise DataMismatch(
        f"DATA_OR_SPLIT_MISMATCH: no {label} with sha256 {expected}; checked {present}"
    )


def resolve_source_files() -> dict[str, Path]:
    """Find the frozen public bytes by content hash (path migration tolerated)."""
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
        "rdkit": rdkit_version(),
    }


def source_with_labels(scope: str = "all") -> dict[str, Any]:
    """Load published split bytes and targets under an explicit scope."""
    if scope not in ("none", "train_valid", "all"):
        raise ValueError(scope)
    return DS.load_lipop(DATA_REPEAT, labels=scope)


__all__ = [
    "ROUND_ID",
    "PROTOCOL_ID",
    "CODE_DIR",
    "TRACK",
    "REPO",
    "DISSECT",
    "DISSECT_CODE",
    "EDIT",
    "EDIT_CODE",
    "DC",
    "DD",
    "DM",
    "DS",
    "EC",
    "EG",
    "EP",
    "ES",
    "ANALYSIS",
    "CONFIGS",
    "RUNS",
    "CACHE",
    "TESTS",
    "DATA_REPEAT",
    "MAIN_REPEAT",
    "SPLITS",
    "SPLIT_SIZES",
    "N_MOLECULES",
    "LABEL_COLUMN",
    "LIPOP_CSV_SHA256",
    "SPLIT_FILE",
    "SPLIT_FILE_SHA256",
    "SPLIT_ROWS_SHA256",
    "RAW_ARTIFACT",
    "RAW_ARTIFACT_SHA256",
    "FEATURE_SPEC_HASH",
    "ATOM_FEAT_DIM",
    "BOND_FEAT_DIM",
    "ATOM_SCALAR_DIM",
    "GLOBAL_DIM",
    "N_NBR_CODES",
    "N_PATH_CODES",
    "N_BOND_TYPES",
    "VARIANT",
    "VARIANT_NAME",
    "EXPECTED_PARAMS",
    "ARMS",
    "ARM_DESCRIPTIONS",
    "MAIN_COMPARISON",
    "MECHANISM_COMPARISONS",
    "RECIPE",
    "TRAIN_SEED",
    "SHUFFLE_SEED",
    "SCHEDULE_SEED",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_REPLICATES",
    "AUX_PAIRS",
    "LAMBDA",
    "DIAGNOSTIC_STEPS",
    "EVAL_BATCH_SIZE",
    "REPRO_TOL",
    "GPU_SOFT_CAP_HOURS",
    "GPU_HARD_CAP_HOURS",
    "MIN_P_TRAIN_PAIRS",
    "MIN_P_TRAIN_COMPONENTS",
    "MIN_E_T_PAIRS",
    "MIN_E_T_COMPONENTS",
    "LOW_SUPPORT_PAIRS",
    "LOW_SUPPORT_COMPONENTS",
    "PAIR_RULES",
    "utcnow",
    "sha256_file",
    "sha256_bytes",
    "sha256_text",
    "array_sha256",
    "stable_json",
    "write_json",
    "read_json",
    "track_rel",
    "row_sha256",
    "log",
    "run_name",
    "run_dir",
    "rdkit_version",
    "git_commit",
    "git_dirty",
    "env_info",
    "code_hashes",
    "DataMismatch",
    "resolve_source_files",
    "verify_source",
    "source_with_labels",
]
