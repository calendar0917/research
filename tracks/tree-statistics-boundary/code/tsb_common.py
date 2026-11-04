"""Shared constants, paths, hashing and metrics for ``tsb-lipo-feature-ladder-v1``.

This module holds the *frozen* protocol constants.  Every number that a run
depends on lives here once, is copied verbatim into
``configs/tsb-lipo-feature-ladder-v1.yaml``, and is asserted equal by the
correctness gates.  Editing a constant is a protocol change, not a bug fix.

Design rules encoded here (see ``PROTOCOL.md``):

* label-free features: no function in this track receives ``y`` while building
  a feature vector;
* deterministic tokens: the typed-WL token is a canonical *string*; Python's
  built-in ``hash()`` is never used for an identifier;
* train-only vocabularies: a vocabulary is built from the official Train rows
  of one published split and then frozen.
"""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# <repo>/tracks/tree-statistics-boundary/code/tsb_common.py
REPO = Path(__file__).resolve().parents[3]
TRACK = REPO / "tracks/tree-statistics-boundary"
CODE = TRACK / "code"
CONFIGS = TRACK / "configs"
CACHE = TRACK / "cache"
ANALYSIS = TRACK / "analysis"

PROTOCOL_ID = "tsb-lipo-feature-ladder-v1"
TRACK_ID = "tree-statistics-boundary"

DATASET = "Lipophilicity"
LABEL_UNITS = "logD (log10 octanol/water distribution coefficient, experimental)"

# ---------------------------------------------------------------------------
# data provenance (all hashes are of files already in this repository)
# ---------------------------------------------------------------------------

CHEMBENCH_DIR = (
    REPO
    / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench/Lipop"
)
CHEMBENCH_CSV = CHEMBENCH_DIR / "Lipop.csv.gz"
CHEMBENCH_CSV_SHA256 = "34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758"

#: Published random repeats.  ``suffix`` is the ChemBench file suffix, not
#: ChemBench's own (glob-order dependent) repeat index.
SPLIT_SUFFIXES: tuple[int, ...] = (122, 123, 124)
SPLIT_FILES: dict[int, str] = {s: f"Lipop.random{s}.ind.pkl" for s in SPLIT_SUFFIXES}
SPLIT_SHA256: dict[int, str] = {
    122: "bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d",
    123: "afaed4c6d95b64ac64e8fb760b6de516802008965befe1601b3664303119a0e8",
    124: "2b48a41ccc39b622bbb74f18331ca9d9513190eec2de29144ce2bc7d47ef7ef3",
}
#: SHA-256 of the int64 Train/Validation/Test row index arrays.  These are the
#: values recorded by ``lcrc_molnet_benchmark_pilot_v1/analysis/split_manifest_Lipophilicity.json``
#: and by the Lipophilicity repeats of ``lcrc_taskaware_representation_pilot_v1``.
SPLIT_ROWS_SHA256: dict[int, dict[str, str]] = {
    122: {
        "train": "4e8d42a95c0b705f43f932b3bbb265277c76a7aa65684b0e5e6ca028580aaf96",
        "valid": "31e3f28f31da69249f70922126f19795920bc1e32cc05d5e29e27507bd5aa365",
        "test": "a4d92ad939bf268c00ff69afe77aa5d58c19a6f809b6a09f6dadc51cd648ec93",
    },
    123: {
        "train": "a3af8ce779db41a288e4e82a7d7ce3d07a556ecf8f2e54ef60f43ad480626db4",
        "valid": "a79e192e922d95bbfa9ff4be1564405d80a0d5ca4c1421fdb5abef5565170982",
        "test": "6c335c844cb1a53273fb784de389d7df25281df86a057bbe22882f130aa06df2",
    },
    124: {
        "train": "dacdee79f1e34d258c55e3bdc6b7b87ecf9bd63039516b64df9f2b2a68cbd887",
        "valid": "e2100b1b236d236f7a3542b057f9026ff77f7bb920aeb662f64f924c3991ef94",
        "test": "c4a5b71d8245ab426cffe475cd8862e11b16106d303b3ddb5dc409611fd0427d",
    },
}
SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}

# ---------------------------------------------------------------------------
# frozen design (protocol content)
# ---------------------------------------------------------------------------

SEED = 20261104
SAMPLE_SIZES: tuple[int, ...] = (1000, 2000, 3360)
FULL_SIZE = 3360

ARMS: tuple[str, ...] = ("G0", "WL2_BIN", "WL2_COUNT", "WL3_COUNT", "MORGAN_COUNT")

WL_ROUNDS = {"WL2": (0, 1, 2), "WL3": (0, 1, 2, 3)}
WL_VOCAB_SIZE = 512
WL_OOV_COLUMN = True  # per round: 512 named tokens + 1 OOV bucket

#: Round-0 atom attributes (all are integer/categorical properties of the
#: RDKit atom, none of them is a learned or label-derived value).
WL_ATOM_ATTRIBUTES: tuple[str, ...] = (
    "atomic_number",
    "formal_charge",
    "aromaticity",
    "total_h",
    "hybridization",
    "chirality",
)
#: Relation attributes of an edge in the typed-WL refinement.
WL_BOND_ATTRIBUTES: tuple[str, ...] = (
    "bond_type",
    "aromaticity",
    "conjugation",
    "stereo",
)

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
MORGAN_USE_COUNTS = True

XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 1,
    "random_state": 0,
    "n_jobs": 4,
}

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261104

CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("count_vs_presence", "WL2_COUNT", "WL2_BIN"),
    ("radius_3_vs_2", "WL3_COUNT", "WL2_COUNT"),
    ("local_identity_vs_G0", "WL2_COUNT", "G0"),
    ("morgan_vs_typed_wl", "MORGAN_COUNT", "WL2_COUNT"),
)

#: External, read-only reference (NOT part of this protocol): the repository's
#: ECFP4 binary + 217 RDKit descriptors + XGBoost baseline.  Kept out of every
#: arm and every contrast on purpose.
EXTERNAL_REFERENCE = {
    "label": "EXTERNAL_READONLY_ECFP4_BINARY_PLUS_217_RDKIT_DESCRIPTORS_XGB",
    "sources": {
        122: "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/baseline/Lipophilicity.json",
        123: "tracks/structural-sufficiency/lcrc_taskaware_representation_pilot_v1/analysis/baseline_repeat/Lipophilicity_random123.json",
        124: "tracks/structural-sufficiency/lcrc_taskaware_representation_pilot_v1/analysis/baseline_repeat/Lipophilicity_random124.json",
    },
    "note": "different input block and different preprocessing; never used for selection in this round",
}

#: Test exposure: Lipophilicity Test labels were already read by earlier rounds
#: of this repository (Sprint 01 of the structural-sufficiency track).  This
#: round therefore never calls its Test numbers a first blind read.
TEST_EXPOSURE_LABEL = "EXPOSED_PUBLIC_TEST"


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def row_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(Path(path).parent)
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def track_rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - git is not required for unit tests
        return "unknown"


def git_dirty_tracked() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return False


def peak_rss_mb() -> float:
    """Process high-water RSS in MiB (Linux ``ru_maxrss`` is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# metrics (original label scale, lower RMSE is better)
# ---------------------------------------------------------------------------


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError(f"prediction/label shape mismatch: {pred.shape} vs {y.shape}")
    if not np.all(np.isfinite(pred)):
        raise ValueError("non-finite predictions")
    if not np.all(np.isfinite(y)):
        raise ValueError("non-finite labels")
    residual = pred - y
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": float(y.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(residual)),
    }


def paired_bootstrap_delta(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired molecule bootstrap for ``RMSE(a) - RMSE(b)`` on one split.

    This quantifies *molecule-sampling* uncertainty on a fixed pair of fitted
    models.  It is **not** an estimate of training-seed randomness: every model
    in this round uses one frozen seed.
    """
    pred_a = np.asarray(pred_a, dtype=np.float64).reshape(-1)
    pred_b = np.asarray(pred_b, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    n = y.size
    deltas = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        idx = rng.integers(0, n, size=n)
        deltas[i] = np.sqrt(np.mean((pred_a[idx] - y[idx]) ** 2)) - np.sqrt(
            np.mean((pred_b[idx] - y[idx]) ** 2)
        )
    point = float(np.sqrt(np.mean((pred_a - y) ** 2)) - np.sqrt(np.mean((pred_b - y) ** 2)))
    return {
        "delta_rmse": point,
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "replicates": int(replicates),
        "seed": int(seed),
        "interpretation": "molecule-sampling interval for one frozen seed; NOT training-seed variance",
    }


def mean_std(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def assert_mapping(actual: Mapping[str, Any], expected: Mapping[str, Any], where: str) -> None:
    if dict(actual) != dict(expected):
        raise ValueError(f"{where}: frozen value drifted\n  actual={actual}\n  expected={expected}")
