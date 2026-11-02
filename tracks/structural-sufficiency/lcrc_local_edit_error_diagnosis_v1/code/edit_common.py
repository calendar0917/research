"""Shared constants, hashes, paths and frozen scientific definitions.

Round: ``lcrc-local-edit-error-diagnosis-v1``.

The round asks a *different* question from the dissection round: not "which
branch of the frozen LCRC-A model matters", but "given two real molecules that
differ by one small fragment substitution, can the explicit statistics of the
model trace the *local edit response error*

    d_pred  = pred(B) - pred(A)
    d_y     = y(B)    - y(A)
    e_delta = d_pred  - d_y

through which local statistics and readout channels it flows, and does that
trace predict ``e_delta`` on *new core structures*?

The upstream dissection round is consumed strictly read-only: its model
re-implementation, R3 recipe, structure pipeline and hashes are reused, its
output paths and run-matrix guards are not.  Every new artifact of this round
lives under this directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROUND_ID = "lcrc-local-edit-error-diagnosis-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

#: Frozen dissection round (read-only upstream implementation + recipe).
DISSECT = REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"

for _p in (str(CODE_DIR), str(DISSECT_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ``dissect_common`` inserts the frozen prep/pilot code dirs into ``sys.path``
# and carries the authoritative byte hashes of the public data.
import dissect_common as DC  # noqa: E402

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"
FREEZE = ANALYSIS / "freeze"

# ---------------------------------------------------------------------------
# data (inherited frozen bytes; asserted against the files before use)
# ---------------------------------------------------------------------------

DATA_REPEAT = 122
LIPOP_CSV_SHA256 = DC.LIPOP_CSV_SHA256
SPLIT_FILE = "Lipop.random122.ind.pkl"
SPLIT_FILE_SHA256 = DC.SPLIT_FILE_SHA256[122]
TRAIN_ROWS_SHA256 = DC.SPLIT_ROWS_SHA256[122][0]
TRAIN_ROWS = 3360
RAW_ARTIFACT = DC.DATA_DIR / "raw_attributes_v1.json.gz"
RAW_ARTIFACT_SHA256 = "1127607de45e341c89ffc67db85714117831ec84b0845c75aaa11005489c51be"
SPLIT_FREEZE_DIR = REPO / "tracks/structural-sufficiency/lcrc_taskaware_representation_pilot_v1/analysis"

#: Structure-pipeline identity inherited from the dissection round.
FEATURE_SPEC_HASH = DC.FEATURE_SPEC_HASH

#: Label scope of this round's internal evaluation.
AUDIT_SCOPE = "WITHIN_DATASET_PROSPECTIVE_AUDIT"

# ---------------------------------------------------------------------------
# model units: exactly six runs, fixed by the protocol
# ---------------------------------------------------------------------------

MODEL_MAIN = "C1"  # NO_BOND_MEAN, 228,674 parameters - primary conclusion
MODEL_REF = "A0"  # FULL, 247,106 parameters - reference only
MODELS = (MODEL_MAIN, MODEL_REF)
SEEDS = (0, 1, 2)
EXPECTED_PARAMS = {"C1": 228674, "A0": 247106}

RUN_UNITS = tuple(
    (model, seed) for model, seed in zip((MODEL_MAIN, MODEL_REF, MODEL_REF, MODEL_MAIN, MODEL_MAIN, MODEL_REF), (0, 0, 1, 1, 2, 2))
)  # frozen wave plan, see PROTOCOL.md

#: Frozen waves: (gpu0, gpu1) per wave.  C1 is *not* pinned to one device.
WAVES = (
    (("C1", 0), ("A0", 0)),
    (("A0", 1), ("C1", 1)),
    (("C1", 2), ("A0", 2)),
)

RECIPE = dict(DC.RECIPE)  # R3, untouched
EPOCHS = int(RECIPE["epochs"])

#: Budget (A100 GPU-hours over all GPU work of this round, smoke included).
GPU_HARD_CAP_HOURS = 4.0
GPU_EXPECTED_LOW = 0.5
GPU_EXPECTED_HIGH = 1.5

# ---------------------------------------------------------------------------
# P0: pair construction + partition (frozen before any label is read)
# ---------------------------------------------------------------------------

PAIR_RULES = {
    "cut": "non-ring single bonds only",
    "variable_fragment_heavy_atoms": (1, 4),
    "core_min_heavy_atoms": 10,
    "core_min_fraction_of_parent": 0.75,
    "require_single_connected_component": True,
    "exclude_stereo_canonical_smiles_tokens": ["@", "/", "\\"],
    "atom_label": "element, is_aromatic, formal_charge, total_H (capped at the cut with a dummy)",
    "bond_label": "SINGLE/DOUBLE/TRIPLE/AROMATIC/OTHER",
    "candidate_generation": "Weisfeiler-Lehman refinement hash on the labelled core",
    "core_verification": "explicit labelled-graph isomorphism with attachment-point marker",
    "definition_choice": "maximum common core, then canonical core string, then fragment strings",
    "direction": "smaller canonical variable-fragment string is A, larger is B",
}

PARTITIONS = ("fit", "select", "diagnose", "audit")
SPLIT_FRACTIONS = {"fit": 0.60, "select": 0.10, "diagnose": 0.15, "audit": 0.15}
SPLIT_SALT = 20261102
SPLIT_RULE = (
    "unit = connected component of the bipartite molecule<->matched-core graph "
    "(shared molecules, identical canonical molecules and shared matched cores are merged); "
    "units that contain at least one pair are placed first, then pair-free units fill the "
    "remaining quota; each unit goes to the partition with the largest relative shortfall "
    "(target_fraction*n_rows - filled)/target_fraction, ties in the order fit>select>diagnose>audit; "
    "ordering is (unit size desc, sha256(salt|unit id) asc)"
)

MIN_DIAGNOSE_PAIRS = 40
MIN_DIAGNOSE_COMPONENTS = 15
MIN_DIAGNOSE_MOLECULES = 60
MIN_AUDIT_PAIRS = 40
MIN_AUDIT_COMPONENTS = 15
MIN_AUDIT_MOLECULES = 60
MIN_NON_SHARED_PAIRS = 40

# ---------------------------------------------------------------------------
# P1/P2/P3 thresholds and probe constants (frozen before any label is read)
# ---------------------------------------------------------------------------

#: FP32 identity tolerance for ``d_pred_raw`` vs the three components (raw logD).
IDENTITY_ABS_TOL = 1e-5
IDENTITY_REL_TOL = 1e-5

#: Ridge penalty shared by every scaled continuous coefficient and class term.
PROBE_L2 = 5.0
#: All probes of one (model, seed) share the same transformation-class prior.
PROBE_METHODS = ("NATIVE", "FAMILY", "GLOBAL", "TRACE", "LOCAL_ONLY")
PROBE_BASELINES = ("NATIVE", "FAMILY", "GLOBAL")
PROBE_MAIN = "TRACE"

BOOTSTRAP_SEED = 20261102
BOOTSTRAP_REPLICATES = 2000

#: Pre-registered *resource* decision thresholds (not universal effect sizes).
PROMISING_MIN_GAIN = 0.03  # logD
PROMISING_MIN_POSITIVE_SEEDS = 2

DESCRIPTIVE_MAX_CLASSES = 3
DESCRIPTIVE_MIN_COMPONENTS = 3

#: Coordinate-gauge demonstration constant (raw logD per atom / per molecule).
GAUGE_K = 0.5


# ---------------------------------------------------------------------------
# hashing / IO (same conventions as the upstream rounds)
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
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def rdkit_version() -> str:
    from rdkit import rdBase

    return str(rdBase.rdkitVersion)


def run_name(model: str, seed: int) -> str:
    return f"edit_test_s{int(seed)}_{model}"


def run_dir(model: str, seed: int) -> Path:
    return RUNS / run_name(model, seed)


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for name, rel in (
        ("dissect_common", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py"),
        ("dissect_model", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py"),
        ("dissect_data", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_data.py"),
        ("dissect_source", "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_source.py"),
    ):
        out[f"upstream:{name}"] = sha256_file(REPO / rel)
    for rel in DC.REUSED_MODULES:
        out[rel] = sha256_file(REPO / rel)
    return out


__all__ = [
    "ROUND_ID",
    "CODE_DIR",
    "TRACK",
    "REPO",
    "DISSECT",
    "DISSECT_CODE",
    "ANALYSIS",
    "CONFIGS",
    "RUNS",
    "CACHE",
    "TESTS",
    "FREEZE",
    "DATA_REPEAT",
    "LIPOP_CSV_SHA256",
    "SPLIT_FILE",
    "SPLIT_FILE_SHA256",
    "TRAIN_ROWS_SHA256",
    "TRAIN_ROWS",
    "RAW_ARTIFACT",
    "RAW_ARTIFACT_SHA256",
    "SPLIT_FREEZE_DIR",
    "FEATURE_SPEC_HASH",
    "AUDIT_SCOPE",
    "MODEL_MAIN",
    "MODEL_REF",
    "MODELS",
    "SEEDS",
    "EXPECTED_PARAMS",
    "RUN_UNITS",
    "WAVES",
    "RECIPE",
    "EPOCHS",
    "GPU_HARD_CAP_HOURS",
    "PAIR_RULES",
    "PARTITIONS",
    "SPLIT_FRACTIONS",
    "SPLIT_SALT",
    "SPLIT_RULE",
    "MIN_DIAGNOSE_PAIRS",
    "MIN_DIAGNOSE_COMPONENTS",
    "MIN_DIAGNOSE_MOLECULES",
    "MIN_AUDIT_PAIRS",
    "MIN_AUDIT_COMPONENTS",
    "MIN_AUDIT_MOLECULES",
    "MIN_NON_SHARED_PAIRS",
    "IDENTITY_ABS_TOL",
    "IDENTITY_REL_TOL",
    "PROBE_L2",
    "PROBE_METHODS",
    "PROBE_BASELINES",
    "PROBE_MAIN",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_REPLICATES",
    "PROMISING_MIN_GAIN",
    "PROMISING_MIN_POSITIVE_SEEDS",
    "DESCRIPTIVE_MAX_CLASSES",
    "DESCRIPTIVE_MIN_COMPONENTS",
    "GAUGE_K",
    "utcnow",
    "sha256_file",
    "sha256_bytes",
    "sha256_text",
    "write_json",
    "read_json",
    "stable_json",
    "track_rel",
    "row_sha256",
    "log",
    "git_commit",
    "git_dirty",
    "env_info",
    "rdkit_version",
    "run_name",
    "run_dir",
    "code_hashes",
]
