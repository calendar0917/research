"""SSRL v0 shared paths, frozen constants and IO helpers.

Round: ``tracks/structural-sufficiency/structural_state_response_learning_v0``

The round is a small *continued-training* experiment on the frozen LCRC-W3
1M checkpoint.  All parent artefacts are read-only:

* ``lcrc_pcqm4mv2_efficiency_width_v1`` -- the W3 checkpoint and model code,
* ``lcrc_pcqm4mv2_transfer_pilot_v1`` -- the 1M / Valid caches and recipe,
* ``structural_functional_equivalence_v0`` -- the Train 1M and Valid cell
  tables (canonical replacement / exact-context assets),
* ``structural_functional_equivalence_followup_v1`` -- frozen labels and
  clean-cell definition,
* ``structural_effect_diagnosis_v0`` / ``structural_state_action_v0`` -- the
  frozen official-Valid evaluation directory and state replay.

No official Test label is indexed anywhere in this round.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

SS = TRACK_DIR
SFEC0 = SS / "structural_functional_equivalence_v0"
SFEC0_CODE = SFEC0 / "code"
SFEC0_DATA = SFEC0 / "data"
SFEC1 = SS / "structural_functional_equivalence_followup_v1"
SFEC1_CODE = SFEC1 / "code"
SFEC1_DATA = SFEC1 / "data"
SED = SS / "structural_effect_diagnosis_v0"
SED_CODE = SED / "code"
SSTA = SS / "structural_state_action_v0"
SSTA_CODE = SSTA / "code"
LCW = SS / "lcrc_pcqm4mv2_efficiency_width_v1"
LCW_CODE = LCW / "code"
LCW_RUN = LCW / "runs" / "PCQM_W3_1M"
LCW_CHECKPOINT = LCW_RUN / "checkpoint_best_valid.pt"
LCT = SS / "lcrc_pcqm4mv2_transfer_pilot_v1"
LCT_CODE = LCT / "code"
LCT_CACHE = LCT / "cache"
LCT_ANALYSIS = LCT / "analysis"
TRAIN_CACHE = LCT_CACHE / "pcqm_train1m"
VALID_CACHE = LCT_CACHE / "pcqm_valid"
TRAIN_GIDS = TRAIN_CACHE / "gids.npy"
VALID_GIDS = VALID_CACHE / "gids.npy"
LABELS_TRAINVALID = LCT_CACHE / "pcqm_labels_trainvalid.npy"
SUBSET_ROWS = LCT_CACHE / "pcqm_subset_rows.npy"

CONFIG_PATH = ROUND_DIR / "configs" / "ssrl_v0.json"
DATA_DIR = ROUND_DIR / "data"
ANALYSIS_DIR = ROUND_DIR / "analysis"
RUNS_DIR = ROUND_DIR / "runs"
for _d in (DATA_DIR, ANALYSIS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

for _p in (str(LCW_CODE), str(LCT_CODE), str(SFEC0_CODE), str(SFEC1_CODE),
           str(SED_CODE), str(SSTA_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# SplitMix64 constants, identical to the frozen SSTA rectangle sampler.
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)
C1 = np.uint64(0x9E3779B97F4A7C15)
C2 = np.uint64(0xC2B2AE3D27D4EB4F)
C3 = np.uint64(0x165667B19E3779F9)
U64 = np.uint64

#: Direction accuracy is only scored on |true| >= this value (eV), the same
#: pre-declared threshold used by Structural State and Action v0.
DIRECTION_THRESHOLD_EV = 0.10
#: bootstrap repetitions for every response interval
BOOTSTRAP_REPS = 1_000


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_sanitize(payload), indent=2, ensure_ascii=False,
                   allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.10g}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return ""
    return value


def write_csv(path: Path | str, rows: Iterable[Mapping[str, Any]],
              columns: list[str] | None = None) -> Path:
    import csv

    path = Path(path)
    rows = list(rows)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})
    tmp.replace(path)
    return path


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def hash64(*parts: Any, salt: str = "ssrl-v0") -> int:
    payload = "|".join([salt] + [str(p) for p in parts]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def mix64(pair_key: np.ndarray, ctx1: np.ndarray, ctx2: np.ndarray) -> np.ndarray:
    """Fixed label-free SplitMix64 hash of ``(pair, context, context)``."""
    z = (np.asarray(pair_key, np.uint64) * C1) ^ (np.asarray(ctx1, np.uint64) * C2) ^ (
        np.asarray(ctx2, np.uint64) * C3
    )
    z ^= z >> U64(30)
    z *= M1
    z ^= z >> U64(27)
    z *= M2
    z ^= z >> U64(31)
    return z


def mix64_plain(rows: np.ndarray, salt: int) -> np.ndarray:
    """Fixed label-free hash of a 1M-Plan cache row."""
    with np.errstate(over="ignore"):
        z = (np.asarray(rows, np.uint64) * C1) ^ (U64(int(salt)) * C2)
    z ^= z >> U64(30)
    z *= M1
    z ^= z >> U64(27)
    z *= M2
    z ^= z >> U64(31)
    return z


def to_z(y_eV: np.ndarray, label_norm: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(y_eV, np.float64) - float(label_norm["mean_eV"])) / float(
        label_norm["std_eV"]
    )


def from_z(z: np.ndarray, label_norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(z, np.float64) * float(label_norm["std_eV"]) + float(
        label_norm["mean_eV"]
    )


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "UNKNOWN"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout
        return bool(out.strip())
    except Exception:
        return True


def set_torch_threads(n: int = 8) -> int:
    import torch

    torch.set_num_threads(int(min(n, os.cpu_count() or 1)))
    return int(torch.get_num_threads())


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def load_label_norm() -> dict[str, Any]:
    return read_json(ROUND_DIR / "configs" / "pcqm_label_norm.json")


def load_feature_norm() -> dict[str, np.ndarray]:
    raw = read_json(ROUND_DIR / "configs" / "pcqm_feature_norm.json")
    return {
        "global_mean": np.asarray(raw["global_mean_list"], np.float32),
        "global_scale": np.asarray(raw["global_scale_list"], np.float32),
        "scalar_mean": np.asarray(raw["scalar_mean_list"], np.float32),
        "scalar_scale": np.asarray(raw["scalar_scale_list"], np.float32),
    }


def load_spec():
    import feature_spec as FS

    return FS.load_spec(ROUND_DIR / "configs" / "pcqm_spec.json")


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


__all__ = [
    "ANALYSIS_DIR",
    "BOOTSTRAP_REPS",
    "CODE_DIR",
    "CONFIG_PATH",
    "C1",
    "C2",
    "C3",
    "DATA_DIR",
    "DIRECTION_THRESHOLD_EV",
    "LABELS_TRAINVALID",
    "LCW",
    "LCW_CHECKPOINT",
    "LCW_CODE",
    "LCW_RUN",
    "LCT",
    "LCT_ANALYSIS",
    "LCT_CACHE",
    "LCT_CODE",
    "M1",
    "M2",
    "REPO_ROOT",
    "ROUND_DIR",
    "RUNS_DIR",
    "SED",
    "SED_CODE",
    "SFEC0",
    "SFEC0_CODE",
    "SFEC0_DATA",
    "SFEC1",
    "SFEC1_CODE",
    "SFEC1_DATA",
    "SS",
    "SSTA",
    "SSTA_CODE",
    "SUBSET_ROWS",
    "TRAIN_CACHE",
    "TRAIN_GIDS",
    "TRACK_DIR",
    "U64",
    "VALID_CACHE",
    "VALID_GIDS",
    "cpu_seconds",
    "from_z",
    "git_commit",
    "git_dirty",
    "hash64",
    "load_feature_norm",
    "load_label_norm",
    "load_spec",
    "mix64",
    "mix64_plain",
    "now_utc",
    "peak_rss_gb",
    "read_json",
    "require",
    "set_torch_threads",
    "sha256_arrays",
    "sha256_bytes",
    "sha256_file",
    "to_z",
    "track_rel",
    "write_csv",
    "write_json",
]
