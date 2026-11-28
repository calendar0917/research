"""Shared paths, hashing and IO helpers for the MolHIV multigranular round (v0).

Round: ``tracks/structural-sufficiency/molhiv_multigranular_structure_sharing_v0``
Protocol id: ``molhiv-multigranular-structure-sharing-v0``

Only label-free structure caches live under ``analysis/cache`` (git-ignored by the
repo-wide ``**/data/**`` rule for ``data/``); JSON/CSV/PNG deliverables live in
``analysis/`` and are committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "molhiv-multigranular-structure-sharing-v0"
SEED = 0

CODE = Path(__file__).resolve().parent
ROUND = CODE.parent
ANALYSIS = ROUND / "analysis"
CACHE = ROUND / "data"  # label-free caches; git-ignored by the repo-wide **/data/** rule
FIGURES = ANALYSIS / "figures"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "runs"

REPO = ROUND.parents[2]
DATA_DIR = REPO / "data"
OGB_MOLHIV = DATA_DIR / "ogb" / "ogbg_molhiv"

#: historical artefacts, read-only references (never modified)
HIST_BASELINE_DIR = (
    REPO / "tracks" / "structural-sufficiency" / "lcrc_molnet_benchmark_pilot_v2" / "analysis" / "baseline"
)
HIST_BASELINE_MANIFEST = (
    REPO / "tracks" / "structural-sufficiency" / "lcrc_molnet_benchmark_pilot_v2" / "analysis"
    / "baseline_manifest.json"
)
HIST_BASELINE_FEATURES = (
    REPO / "tracks" / "structural-sufficiency" / "lcrc_molnet_benchmark_pilot_v2" / "cache"
    / "baseline_features" / "ogbg-molhiv.npz"
)

#: OGB ogbg-molhiv raw column semantics (documented in PROTOCOL.md §3)
ATOM_FEATURE_NAMES = (
    "atomic_num",
    "chirality",
    "degree",
    "formal_charge",
    "num_h",
    "num_radical_e",
    "hybridization",
    "aromatic",
    "in_ring",
)
BOND_FEATURE_NAMES = ("bond_type", "stereo", "conjugated")
#: COARSE keeps these raw OGB columns only (atom), and ``bond_type`` for bonds
COARSE_ATOM_COLUMNS = (0, 3, 7)  # atomic_num, formal_charge, aromatic
COARSE_BOND_COLUMNS = (0,)  # bond_type

FAMILIES = ("size-2", "size-3", "size-4", "cycle-5", "cycle-6")
FAMILY_SIZE = {"size-2": 2, "size-3": 3, "size-4": 4, "cycle-5": 5, "cycle-6": 6}

#: frozen vocabulary filters (PROTOCOL.md §4.3)
COARSE_MIN_DOC_FREQ = 20
COARSE_MIN_SCAFFOLDS = 5
COARSE_MAX_PER_FAMILY = 256
FULL_MIN_DOC_FREQ = 5
FULL_MIN_SCAFFOLDS = 3
FULL_MAX_PER_FAMILY = 1024


def ensure_dirs() -> None:
    for p in (ANALYSIS, CACHE, FIGURES, CONFIGS, RUNS):
        p.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def write_json(path: Path | str, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    path.write_text(text + "\n")
    return sha256_file(path)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def git_state() -> dict[str, Any]:
    """Commit + dirty flag of the local checkout, without touching history."""
    import subprocess

    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked": bool(status),
        "status_short": status.splitlines()[:20],
    }


def environment() -> dict[str, Any]:
    import platform

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for name in ("scipy", "sklearn", "pandas", "torch", "xgboost", "rdkit"):
        try:
            mod = __import__(name)
        except Exception:  # pragma: no cover - optional stack
            env[name] = None
            continue
        env[name] = getattr(mod, "__version__", "unknown")
    env["cpu_count"] = os.cpu_count()
    return env


class Timer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.t0 = time.perf_counter()

    def __enter__(self) -> "Timer":
        print(f"[{self.label}] start", flush=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.perf_counter() - self.t0
        print(f"[{self.label}] done in {self.seconds:.1f}s", flush=True)


def add_code_path() -> None:
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
