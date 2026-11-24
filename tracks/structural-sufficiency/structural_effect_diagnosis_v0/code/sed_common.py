"""Shared paths, IO and paired-bootstrap helpers for the diagnosis round.

This round only re-analyses existing frozen artefacts.  It never trains a model,
never edits a parent checkpoint and never regenerates a molecule directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

ANALYSIS_DIR = ROUND_DIR / "analysis"

SES_DIR = TRACK_DIR / "structural_evidence_selection_pilot_v0"
SES_RESULTS = SES_DIR / "results"
SES_ANALYSIS = SES_DIR / "analysis"

SFEC_DIR = TRACK_DIR / "structural_functional_equivalence_followup_v1"
SFEC_CODE = SFEC_DIR / "code"
SFEC_DATA = SFEC_DIR / "data"
SFEC_ANALYSIS = SFEC_DIR / "analysis"

LCW_DIR = TRACK_DIR / "lcrc_pcqm4mv2_efficiency_width_v1"
LCW_CODE = LCW_DIR / "code"
LCW_RUNS = LCW_DIR / "runs"
LCW_RUN = LCW_RUNS / "PCQM_W3_1M"
LCW_CHECKPOINT = LCW_RUN / "checkpoint_best_valid.pt"

LC_OLD_DIR = TRACK_DIR / "lcrc_pcqm4mv2_transfer_pilot_v1"
LC_OLD_CACHE = LC_OLD_DIR / "cache"
LC_OLD_ANALYSIS = LC_OLD_DIR / "analysis"


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


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def paired_bootstrap(se_a: np.ndarray, se_b: np.ndarray, *, n_boot: int = 1000,
                     seed: int = 0) -> dict[str, Any]:
    """Molecule/observation-level paired bootstrap of the RMSE difference.

    The two arrays must be aligned per molecule/observation.  The returned
    interval is for ``RMSE(a) - RMSE(b)``; the point estimate uses the same
    arrays without resampling.
    """
    se_a = np.asarray(se_a, np.float64)
    se_b = np.asarray(se_b, np.float64)
    if se_a.shape != se_b.shape:
        raise ValueError("paired bootstrap needs equal-length arrays")
    rng = np.random.default_rng(int(seed))
    n = se_a.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    rmse_a = np.sqrt(se_a[idx].mean(axis=1))
    rmse_b = np.sqrt(se_b[idx].mean(axis=1))
    delta = rmse_a - rmse_b
    return {
        "n": int(n),
        "n_boot": int(n_boot),
        "rmse_a": float(np.sqrt(se_a.mean())),
        "rmse_b": float(np.sqrt(se_b.mean())),
        "delta_a_minus_b": float(np.sqrt(se_a.mean()) - np.sqrt(se_b.mean())),
        "delta_bootstrap_mean": float(delta.mean()),
        "delta_ci95": [float(np.percentile(delta, 2.5)),
                       float(np.percentile(delta, 97.5))],
        "delta_bootstrap_se": float(delta.std(ddof=0)),
    }


def rmse(se: np.ndarray) -> float:
    return float(np.sqrt(np.asarray(se, np.float64).mean()))


def mae_from_ae(ae: np.ndarray) -> float:
    return float(np.asarray(ae, np.float64).mean())


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
