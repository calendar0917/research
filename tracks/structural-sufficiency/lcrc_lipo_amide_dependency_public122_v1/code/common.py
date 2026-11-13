"""Frozen inputs, provenance and model plumbing for the amide-dependency round.

Round / protocol: ``lcrc-lipo-amide-dependency-public122-v1``.

The scientific question is *prediction dependence* of the frozen LCRC-C1
Lipophilicity model on one connected structural element (the amide
``c(=o)-n`` link).  This module owns nothing scientific; it only pins the
already frozen data / split / checkpoint / feature-spec / normalisation and
provides IO helpers.  No optimiser, no training, no new model, no new seed.

Upstream modules consumed read-only (hashed into every record):

* ``lcrc_cross_task_preparation_v1`` - ``lipo_v1`` FeatureSpec, molecular graph
  primitives, cache builder/validator, collate.
* ``lcrc_molnet_benchmark_pilot_v1`` - label-free raw attribute artifact and
  the ``smiles -> raw row`` vocabulary (local RDKit only).
* ``lcrc_lipo_model_dissection_v1`` - public122 loader, C1 model, frozen
  reference checkpoint of the public122 FULL seed0 model.
* ``lcrc_lipo_edit_supervision_public122_s0_v1`` - the round that produced the
  frozen ``A_s0`` checkpoint (ordinary MSE, no auxiliary term).
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

# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

ROUND_ID = "lcrc-lipo-amide-dependency-public122-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

SS = REPO / "tracks/structural-sufficiency"
DISSECT = SS / "lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
PREP = SS / "lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
PILOT = SS / "lcrc_molnet_benchmark_pilot_v1"
PILOT_CODE = PILOT / "code"
EDITS = SS / "lcrc_lipo_edit_supervision_public122_s0_v1"

for _p in (str(CODE_DIR), str(DISSECT_CODE), str(PREP_CODE), str(PILOT_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dissect_common as DC  # noqa: E402  (frozen paths/hashes/reference)
import dissect_data as DD  # noqa: E402  (frozen spec + cache semantics)
import dissect_model as DM  # noqa: E402  (frozen C1 model)
import dissect_source as DS  # noqa: E402  (public122 split + targets)
import mb_raw_artifact as RAW  # noqa: E402  (label-free raw attribute artifact)
import mb_rdkit as RD  # noqa: E402  (frozen raw row -> lipo_v1 graph; RDKit only inside functions)

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"
CASES = ANALYSIS / "cases"

REUSED_MODULES = tuple(DC.REUSED_MODULES) + (
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_data.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_source.py",
    "tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1/code/common.py",
    "tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1/code/data.py",
    "tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1/code/train.py",
)

# ---------------------------------------------------------------------------
# frozen data (public122), copied verbatim from the executed protocols
# ---------------------------------------------------------------------------

DATA_REPEAT = 122
SPLITS = ("train", "valid", "test")
SPLIT_SIZES = (3360, 420, 420)
N_MOLECULES = 4200
LABEL_COLUMN = "exp"

SOURCE_COMMIT = "0b10d4c1ac72b6929503694a5f60446824736391"
LIPOP_CSV_SHA256 = "34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758"
SPLIT_FILE = "Lipop.random122.ind.pkl"
SPLIT_FILE_SHA256 = "bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d"
RAW_ARTIFACT_SHA256 = "1127607de45e341c89ffc67db85714117831ec84b0845c75aaa11005489c51be"
SPLIT_ROWS_SHA256 = {
    "train": "4e8d42a95c0b705f43f932b3bbb265277c76a7aa65684b0e5e6ca028580aaf96",
    "valid": "31e3f28f31da69249f70922126f19795920bc1e32cc05d5e29e27507bd5aa365",
    "test": "a4d92ad939bf268c00ff69afe77aa5d58c19a6f809b6a09f6dadc51cd648ec93",
}

# ---------------------------------------------------------------------------
# frozen model: LCRC-C1 = NO_BOND_MEAN, ordinary MSE arm A, seed 0
# ---------------------------------------------------------------------------

VARIANT = "C1"
VARIANT_NAME = "NO_BOND_MEAN"
EXPECTED_PARAMS = 228674
CHECKPOINT_REL = "tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1/runs/A_s0/best.pt"
CHECKPOINT = REPO / CHECKPOINT_REL
CHECKPOINT_SHA256 = "bdd670e8df5ee03ba28f2f4898e7fd137991a9fd23e9add91c08d8b6e0a1d6a7"
STATE_HASH = "7d32221e0c3f482a16e2e170b7df1d4753ff4d7783cc2b5a3386b183a7941e36"
BEST_EPOCH = 244

#: Frozen reference metrics (recomputed from ``predictions_best.npz``).
REFERENCE = {
    "valid_rmse": 0.7025358944353126,
    "test_rmse": 0.5761674771952362,
    "test_mae": 0.4280564144318241,
}
REPRO_TOL = 1e-5

# ---------------------------------------------------------------------------
# frozen structure caches produced by the same label-free pipeline
# ---------------------------------------------------------------------------

CACHE_DIR = EDITS / "cache"
CACHE_MANIFEST = CACHE_DIR / "manifest.json"
REFERENCE_PREDICTIONS = EDITS / "runs/A_s0/predictions_best.npz"

# ---------------------------------------------------------------------------
# analysis constants (frozen before any view is predicted)
# ---------------------------------------------------------------------------

BOOTSTRAP_SEED = 20261114
BOOTSTRAP_REPLICATES = 1000
INFLUENCE_FRACTION = 0.05

POPULATIONS = ("ALL", "MATCHED", "BALANCED", "R")
MAIN_POPULATION = "MATCHED"
SCOPES = ("FULL", "G_FIXED")
VIEW_KINDS = ("O", "T", "K", "R")

#: Instance-level structural rule constants (protocol section 3).
MF_C_CLASS = 2  # non-aromatic C
MF_AROMATIC_C_CLASS = 3
MF_N_CLASS = 7  # non-aromatic N
MF_O_CLASS = 9
MF_NEUTRAL_CHARGE = 2  # charge bucket index of formal charge 0
MF_SINGLE_BOND = 0  # raw edge_class of a single bond
MF_DOUBLE_BOND = 1  # raw edge_class of a double bond
MATCHED_SIDE_TOL = 2  # BALANCED subset: |n-side - acyl-side| <= 2


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


def rdkit_version() -> str:
    try:
        from rdkit import rdBase

        return str(rdBase.rdkitVersion)
    except Exception:  # pragma: no cover
        return "unavailable"


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
        "rdkit": rdkit_version(),
    }


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for p in sorted(TESTS.glob("*.py")):
        out[f"tests/{p.name}"] = sha256_file(p)
    for p in sorted(CONFIGS.glob("*.yaml")):
        out[f"configs/{p.name}"] = sha256_file(p)
    for name in ("PROTOCOL.md", "REPORT.md"):
        p = TRACK / name
        if p.exists():
            out[name] = sha256_file(p)
    for rel in REUSED_MODULES:
        p = REPO / rel
        out[rel] = sha256_file(p) if p.exists() else "missing"
    return out


def config_snapshot() -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "protocol_id": PROTOCOL_ID,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "generated_utc": utcnow(),
        "environment": env_info(),
        "data": {
            "source_commit": SOURCE_COMMIT,
            "csv_sha256": LIPOP_CSV_SHA256,
            "split_file": SPLIT_FILE,
            "split_file_sha256": SPLIT_FILE_SHA256,
            "split_rows_sha256": dict(SPLIT_ROWS_SHA256),
            "split_sizes": dict(zip(SPLITS, SPLIT_SIZES)),
            "raw_artifact_sha256": RAW_ARTIFACT_SHA256,
            "label_column": LABEL_COLUMN,
            "repeat": DATA_REPEAT,
        },
        "model": {
            "variant": VARIANT,
            "variant_name": VARIANT_NAME,
            "parameter_count": EXPECTED_PARAMS,
            "checkpoint": CHECKPOINT_REL,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "state_hash": STATE_HASH,
            "best_epoch": BEST_EPOCH,
            "training": "none (frozen checkpoint, zero new training)",
        },
        "feature_spec": {
            "hash": DC.FEATURE_SPEC_HASH,
            "atom_feat_dim": DC.ATOM_FEAT_DIM,
            "bond_feat_dim": DC.BOND_FEAT_DIM,
            "atom_scalar_dim": DC.ATOM_SCALAR_DIM,
            "global_dim": DC.GLOBAL_DIM,
            "nbr_codes": DC.N_NBR_CODES,
            "path_codes": DC.N_PATH_CODES,
            "bond_types": DC.N_BOND_TYPES,
        },
        "analysis": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "influence_fraction": INFLUENCE_FRACTION,
            "repro_tol": REPRO_TOL,
            "populations": list(POPULATIONS),
            "scopes": list(SCOPES),
        },
    }


# ---------------------------------------------------------------------------
# source verification
# ---------------------------------------------------------------------------


class DataMismatch(RuntimeError):
    """Raised when a frozen byte hash does not match."""


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
    """Find the frozen public bytes by content hash (path migration tolerated)."""
    csv_candidates = [DC.LIPOP_CSV]
    csv_candidates += sorted(SS.glob("*/analysis/source/chembench/Lipop/Lipop.csv.gz"))
    split_candidates = [DC.CHEMBENCH_LIPOP / SPLIT_FILE]
    split_candidates += sorted(SS.glob(f"*/analysis/source/chembench/Lipop/{SPLIT_FILE}"))
    raw_candidates = [DC.DATA_DIR / "raw_attributes_v1.json.gz"]
    raw_candidates += sorted(SS.glob("*/analysis/source/raw_attributes_v1.json.gz"))
    return {
        "csv": _resolve_by_hash(csv_candidates, LIPOP_CSV_SHA256, "Lipop.csv.gz"),
        "split": _resolve_by_hash(split_candidates, SPLIT_FILE_SHA256, SPLIT_FILE),
        "raw": _resolve_by_hash(raw_candidates, RAW_ARTIFACT_SHA256, "raw_attributes_v1.json.gz"),
    }


def load_source(scope: str = "all") -> dict[str, Any]:
    """Published split bytes + optional targets, all hashes re-verified."""
    if scope not in ("none", "train_valid", "all"):
        raise ValueError(scope)
    src = DS.load_lipop(DATA_REPEAT, labels=scope)
    for split, expected in zip(SPLITS, (SPLIT_ROWS_SHA256[s] for s in SPLITS)):
        got = row_sha256(src["rows"][split])
        if got != expected:
            raise DataMismatch(f"DATA_OR_SPLIT_MISMATCH: {split} row-index sha256 {got} != {expected}")
        if len(src["rows"][split]) != dict(zip(SPLITS, SPLIT_SIZES))[split]:
            raise DataMismatch(f"DATA_OR_SPLIT_MISMATCH: {split} size")
    if src["csv_sha256"] != LIPOP_CSV_SHA256:
        raise DataMismatch("DATA_OR_SPLIT_MISMATCH: CSV hash")
    if src["split_file_sha256"] != SPLIT_FILE_SHA256:
        raise DataMismatch("DATA_OR_SPLIT_MISMATCH: split-file hash")
    return src


def load_raw_rows() -> dict[str, Any]:
    """Label-free raw attribute artifact (both directions per physical bond)."""
    payload = RAW.load(verify=True)
    rows = payload["rows"]["Lipophilicity"]
    if len(rows) != N_MOLECULES:
        raise DataMismatch(f"raw artifact has {len(rows)} Lipophilicity rows != {N_MOLECULES}")
    return payload


def split_of_row(src: Mapping[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for split in SPLITS:
        for row in src["rows"][split]:
            out[int(row)] = split
    if len(out) != N_MOLECULES:
        raise ValueError(f"split union {len(out)} != {N_MOLECULES}")
    return out


# ---------------------------------------------------------------------------
# frozen model
# ---------------------------------------------------------------------------


def load_checkpoint() -> dict[str, Any]:
    import torch

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"MISSING_BASELINE: {track_rel(CHECKPOINT)} not found")
    got = sha256_file(CHECKPOINT)
    if got != CHECKPOINT_SHA256:
        raise DataMismatch(f"MISSING_BASELINE: checkpoint sha256 {got} != {CHECKPOINT_SHA256}")
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ck["state_dict"]
    fp = DM.state_fingerprint(state)
    if fp != STATE_HASH:
        raise DataMismatch(f"MISSING_BASELINE: state hash {fp} != {STATE_HASH}")
    if str(ck.get("variant")) != VARIANT:
        raise DataMismatch(f"checkpoint variant {ck.get('variant')!r} != {VARIANT!r}")
    if str(ck.get("feature_spec_hash")) != DC.FEATURE_SPEC_HASH:
        raise DataMismatch("checkpoint feature spec hash differs from frozen lipo_v1")
    if int(ck.get("parameters", -1)) != EXPECTED_PARAMS:
        raise DataMismatch(f"checkpoint parameter count {ck.get('parameters')} != {EXPECTED_PARAMS}")
    if int(ck.get("best_epoch", -1)) != BEST_EPOCH:
        raise DataMismatch(f"checkpoint best epoch {ck.get('best_epoch')} != {BEST_EPOCH}")
    return ck


def frozen_norm(ck: Mapping[str, Any]) -> dict[str, np.ndarray]:
    state = ck["state_dict"]
    return {
        key: np.asarray(state[key].detach().cpu().numpy(), dtype=np.float32)
        for key in ("global_mean", "global_scale", "scalar_mean", "scalar_scale")
    }


def target_norm(ck: Mapping[str, Any]) -> dict[str, float]:
    t = dict(ck["target_norm"])
    mean, scale = float(t["mean"]), float(t["scale"])
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("degenerate frozen target normaliser")
    return {
        "mean": mean,
        "scale": scale,
        "n": int(t["n"]),
        "source": str(t["source"]),
        "formula": str(t["formula"]),
        "valid_or_test_used": bool(t.get("valid_or_test_used", False)),
    }


def build_frozen_model(ck: Mapping[str, Any], device="cpu"):
    """C1 model with the checkpoint state; parameters are never modified."""
    spec = DD.frozen_spec()
    model = DM.load_checkpoint_model(spec, frozen_norm(ck), VARIANT, ck["state_dict"], device)
    n = DM.parameter_count(model)
    if n != EXPECTED_PARAMS:
        raise ValueError(f"model parameter count {n} != frozen {EXPECTED_PARAMS}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, spec


__all__ = [
    "ROUND_ID",
    "PROTOCOL_ID",
    "CODE_DIR",
    "TRACK",
    "REPO",
    "SS",
    "DISSECT",
    "PREP",
    "PILOT",
    "EDITS",
    "ANALYSIS",
    "CONFIGS",
    "CACHE",
    "TESTS",
    "CASES",
    "REUSED_MODULES",
    "DATA_REPEAT",
    "SPLITS",
    "SPLIT_SIZES",
    "N_MOLECULES",
    "LABEL_COLUMN",
    "SOURCE_COMMIT",
    "LIPOP_CSV_SHA256",
    "SPLIT_FILE",
    "SPLIT_FILE_SHA256",
    "RAW_ARTIFACT_SHA256",
    "SPLIT_ROWS_SHA256",
    "VARIANT",
    "VARIANT_NAME",
    "EXPECTED_PARAMS",
    "CHECKPOINT_REL",
    "CHECKPOINT",
    "CHECKPOINT_SHA256",
    "STATE_HASH",
    "BEST_EPOCH",
    "REFERENCE",
    "REPRO_TOL",
    "CACHE_DIR",
    "CACHE_MANIFEST",
    "REFERENCE_PREDICTIONS",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_REPLICATES",
    "INFLUENCE_FRACTION",
    "POPULATIONS",
    "MAIN_POPULATION",
    "SCOPES",
    "VIEW_KINDS",
    "MF_C_CLASS",
    "MF_AROMATIC_C_CLASS",
    "MF_N_CLASS",
    "MF_O_CLASS",
    "MF_NEUTRAL_CHARGE",
    "MF_SINGLE_BOND",
    "MF_DOUBLE_BOND",
    "MATCHED_SIDE_TOL",
    "DC",
    "DD",
    "DM",
    "DS",
    "RAW",
    "RD",
    "utcnow",
    "sha256_file",
    "sha256_bytes",
    "stable_json",
    "write_json",
    "read_json",
    "track_rel",
    "row_sha256",
    "log",
    "git_commit",
    "git_dirty",
    "rdkit_version",
    "env_info",
    "code_hashes",
    "config_snapshot",
    "DataMismatch",
    "resolve_source_files",
    "load_source",
    "load_raw_rows",
    "split_of_row",
    "load_checkpoint",
    "frozen_norm",
    "target_norm",
    "build_frozen_model",
]
