"""Shared paths, config and IO helpers for SFEC v1 (follow-up round).

The parent round is read-only.  Only parent caches are loaded; no file under
``structural_functional_equivalence_v0`` is modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
PARENT_DIR = ROUND_DIR.parent / "structural_functional_equivalence_v0"
PARENT_DATA = PARENT_DIR / "data"
PARENT_ANALYSIS = PARENT_DIR / "analysis"
CONFIG_PATH = ROUND_DIR / "configs" / "sfec_v1.json"
DATA_DIR = ROUND_DIR / "data"
ANALYSIS_DIR = ROUND_DIR / "analysis"
for _d in (DATA_DIR, ANALYSIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PCQM_ROOT = REPO_ROOT / "data" / "ogb" / "pcqm4m-v2"
LABEL_CACHE = DATA_DIR / "pcqm_labels.npy"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            _sanitize_json(payload), indent=2, ensure_ascii=False,
            allow_nan=False, default=_json_default,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path | str, rows: Iterable[Mapping[str, Any]], columns=None) -> Path:
    import csv

    path = Path(path)
    rows = list(rows)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})
    tmp.replace(path)
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.10g}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return ""
    return value


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


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "pid": os.getpid(),
        "blas_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
    }


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_pcqm_labels_cached() -> np.ndarray:
    """Raw HOMO-LUMO gap by official CSV row index.

    The gzip file is parsed once and cached under this round's ``data/``.
    Only the label column is retained.
    """
    if LABEL_CACHE.exists():
        labels = np.load(LABEL_CACHE)
        if labels.ndim == 1 and len(labels) == 3_746_620:
            return labels
    import gzip

    path = PCQM_ROOT / "raw" / "data.csv.gz"
    labels = np.full(3_746_620, np.nan, dtype=np.float64)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        assert header == ["idx", "smiles", "homolumogap"], header
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 3:
                continue
            idx = int(parts[0])
            if parts[2].strip():
                labels[idx] = float(parts[2])
    LABEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(LABEL_CACHE, labels)
    return labels


def find_parent_prefix(name: str) -> Path:
    path = PARENT_DATA / name
    if not path.with_suffix(".npz").exists():
        raise FileNotFoundError(path)
    return path
