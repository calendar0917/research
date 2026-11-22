"""Shared plumbing for the ACRA-v1 pilot.

Only label-free, path-independent helpers live here.  The parent round
``task_response_composition_s0_v1`` is imported **read-only** for the raw OGB
cache loader, the rooted 1-4 frame machinery and the exact coloured-graph key;
none of its model output definitions (per-cut whole-molecule prediction) are
used.  Local model definitions are in :mod:`acra_model`.

Everything that fits a parameter (target scaler, atom/bond category support,
model selection) is computed from the official **Train** rows only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
PARENT_ROUND = ROUND_DIR.parent / "task_response_composition_s0_v1"
PARENT_CODE = PARENT_ROUND / "code"

if str(PARENT_CODE) not in sys.path:
    sys.path.insert(0, str(PARENT_CODE))

import trc_common as PC        # noqa: E402  path-independent plumbing
import trc_data as PD          # noqa: E402  raw cache, frames, exact keys
import trc_model as PM         # noqa: E402  deterministic per-module init only

CONFIG_PATH = ROUND_DIR / "configs" / "acra_v1.json"

SPLITS = PC.SPLITS
ATOM_COLUMNS = PC.ATOM_COLUMNS
BOND_COLUMNS = PC.BOND_COLUMNS
MAX_OBJECT_ATOMS = PC.MAX_OBJECT_ATOMS
FRAME_PAIRS = PC.FRAME_PAIRS

# path-independent helpers, re-exported verbatim
rmse = PC.rmse
mae = PC.mae
jsonable = PC.jsonable
read_json = PC.read_json
write_json = PC.write_json
write_csv = PC.write_csv
sha256_file = PC.sha256_file
target_scaler = PC.target_scaler
official_feature_dims = PC.official_feature_dims
set_deterministic = PC.set_deterministic
environment = PC.environment
git_commit = PC.git_commit

deterministic_init = PM.deterministic_init
state_hash = PM.state_hash
module_hashes = PM.module_hashes


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def data_dir() -> Path:
    return ROUND_DIR / "data"


def analysis_dir() -> Path:
    return ROUND_DIR / "analysis"


def results_root() -> Path:
    override = os.environ.get("ACRA_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return ROUND_DIR / "results"


def run_dir(run_id: str) -> Path:
    raw = Path(run_id)
    if raw.is_absolute():
        path = raw
    elif len(raw.parts) > 1:
        path = REPO_ROOT / raw
    else:
        path = results_root() / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_csv(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                rec[key] = value
                continue
            try:
                rec[key] = float(value)
            except (TypeError, ValueError):
                rec[key] = value
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def code_provenance() -> dict[str, Any]:
    files = ("acra_common.py", "acra_data.py", "acra_cache.py", "acra_model.py",
             "acra_train.py")
    out = {name: sha256_file(CODE_DIR / name) for name in files
           if (CODE_DIR / name).is_file()}
    return {
        "commit": git_commit(),
        "code": out,
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_id": load_config()["protocol_id"],
    }


def parent_provenance() -> dict[str, str]:
    files = ("trc_common.py", "trc_data.py", "trc_model.py")
    return {name: sha256_file(PARENT_CODE / name) for name in files}


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def quantiles(values: np.ndarray, points: Sequence[float] = (5, 25, 50, 75, 95)) -> dict[str, float]:
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return {"n": 0}
    out = {f"p{int(p)}": float(np.percentile(v, p)) for p in points}
    out["mean"] = float(np.mean(v))
    out["std"] = float(np.std(v))
    out["min"] = float(np.min(v))
    out["max"] = float(np.max(v))
    out["n"] = int(v.size)
    return out


def paired_bootstrap(pred_control: np.ndarray, pred_candidate: np.ndarray,
                     labels: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """RMSE(control) - RMSE(candidate) with molecules resampled as a unit."""
    a = np.asarray(pred_control, np.float64)
    b = np.asarray(pred_candidate, np.float64)
    y = np.asarray(labels, np.float64)
    n = y.size
    if n == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    delta = rmse(a, y) - rmse(b, y)
    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        stats[i] = rmse(a[idx], y[idx]) - rmse(b[idx], y[idx])
    return {
        "delta_rmse": float(delta),
        "delta_mae": float(mae(a, y) - mae(b, y)),
        "lo": float(np.percentile(stats, 2.5)),
        "hi": float(np.percentile(stats, 97.5)),
        "n": int(n),
        "reps": int(n_boot),
        "seed": int(seed),
        "positive_means_candidate_better": bool(delta > 0),
        "train_seed_uncertainty_included": False,
    }


def cluster_bootstrap_metric(pred: np.ndarray, truth: np.ndarray, clusters: np.ndarray,
                             metric: str = "rmse", n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """**Absolute** error level with a cluster bootstrap over the given units."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(truth, np.float64)
    c = np.asarray(clusters)
    if t.size == 0:
        return {"n": 0, "n_clusters": 0}
    uniq = np.unique(c)
    rng = np.random.default_rng(int(seed))
    func = rmse if metric == "rmse" else mae
    stats = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([np.nonzero(c == uniq[j])[0] for j in pick])
        stats[i] = func(p[idx], t[idx])
    point = func(p, t)
    return {
        "metric": metric,
        "value": float(point),
        "lo": float(np.percentile(stats, 2.5)),
        "hi": float(np.percentile(stats, 97.5)),
        "n": int(t.size),
        "n_clusters": int(uniq.size),
        "reps": int(n_boot),
        "seed": int(seed),
        "train_seed_uncertainty_included": False,
    }


def cluster_bootstrap_delta(pred_a: np.ndarray, pred_b: np.ndarray, truth: np.ndarray,
                            clusters: np.ndarray, metric: str = "mae",
                            n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """``metric(a) - metric(b)`` resampled by cluster (positive => b better)."""
    a = np.asarray(pred_a, np.float64)
    b = np.asarray(pred_b, np.float64)
    t = np.asarray(truth, np.float64)
    c = np.asarray(clusters)
    if t.size == 0:
        return {"n": 0, "n_clusters": 0}
    uniq = np.unique(c)
    func = rmse if metric == "rmse" else mae
    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([np.nonzero(c == uniq[j])[0] for j in pick])
        stats[i] = func(a[idx], t[idx]) - func(b[idx], t[idx])
    return {
        "metric": metric,
        "delta": float(func(a, t) - func(b, t)),
        "lo": float(np.percentile(stats, 2.5)),
        "hi": float(np.percentile(stats, 97.5)),
        "n": int(t.size),
        "n_clusters": int(uniq.size),
        "reps": int(n_boot),
        "train_seed_uncertainty_included": False,
    }


def paired_delta_stats(true_delta: np.ndarray, pred_delta: np.ndarray,
                       clusters: np.ndarray | None = None,
                       n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Delta MAE/RMSE/direction accuracy with an optional cluster bootstrap."""
    t = np.asarray(true_delta, np.float64)
    p = np.asarray(pred_delta, np.float64)
    if t.size == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(t.size),
        "delta_mae": float(np.mean(np.abs(p - t))),
        "delta_rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "direction_accuracy": float(np.mean(np.sign(p) == np.sign(t))),
        "pearson_pred_true": pearson(p, t),
        "mean_pred_delta": float(np.mean(p)),
        "mean_true_delta": float(np.mean(t)),
    }
    if clusters is not None:
        c = np.asarray(clusters)
        uniq = np.unique(c)
        rng = np.random.default_rng(int(seed))
        mae_b, rmse_b, dir_b = [], [], []
        for _ in range(int(n_boot)):
            pick = rng.integers(0, uniq.size, uniq.size)
            idx = np.concatenate([np.nonzero(c == uniq[j])[0] for j in pick])
            mae_b.append(np.mean(np.abs(p[idx] - t[idx])))
            rmse_b.append(np.sqrt(np.mean((p[idx] - t[idx]) ** 2)))
            dir_b.append(np.mean(np.sign(p[idx]) == np.sign(t[idx])))
        out["delta_mae_lo"] = float(np.percentile(mae_b, 2.5))
        out["delta_mae_hi"] = float(np.percentile(mae_b, 97.5))
        out["delta_rmse_lo"] = float(np.percentile(rmse_b, 2.5))
        out["delta_rmse_hi"] = float(np.percentile(rmse_b, 97.5))
        out["direction_accuracy_lo"] = float(np.percentile(dir_b, 2.5))
        out["direction_accuracy_hi"] = float(np.percentile(dir_b, 97.5))
        out["n_clusters"] = int(uniq.size)
        out["bootstrap"] = {"unit": "cluster", "reps": int(n_boot), "seed": int(seed)}
    return out