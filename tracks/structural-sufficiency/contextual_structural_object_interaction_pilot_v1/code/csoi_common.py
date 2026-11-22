"""Shared plumbing for contextual_structural_object_interaction_pilot_v1.

Path resolution lives here because the parent round's helpers resolve *inside*
the parent round directory.  Everything that is path independent (metrics,
json/csv IO, seeding, environment/provenance, official feature dims, the
deterministic per-module init, the pair directory, the structural-object cache
builder) is imported **read-only** from
``tracks/structural-sufficiency/task_response_composition_s0_v1/code``.

The parent round is never modified: this module only puts its ``code/``
directory on ``sys.path`` so the exact same object definitions, cut directory
and pair store are reused instead of re-implemented.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
PARENT_ROUND = ROUND_DIR.parent / "task_response_composition_s0_v1"
PARENT_CODE = PARENT_ROUND / "code"

if str(PARENT_CODE) not in sys.path:
    sys.path.insert(0, str(PARENT_CODE))

import trc_common as PC        # noqa: E402  parent plumbing (path independent parts reused)
import trc_data as PD          # noqa: E402  rooted objects / frames / exact keys
import trc_model as PM         # noqa: E402  deterministic init (shared init rule)
import trc_pairs as PPAIR      # noqa: E402  train-only PairStore
import trc_prepare as PP       # noqa: E402  cache builder (objects, cuts, directory)
import trc_train as PTR        # noqa: E402  lr schedule / horizon rule

CONFIG_PATH = ROUND_DIR / "configs" / "csoi_v1.json"
RUN_MANIFEST_PATH = ROUND_DIR / "run_manifest.json"

#: parent module files that this round reuses, for provenance
REUSED_PARENT_MODULES = ("trc_common.py", "trc_data.py", "trc_prepare.py",
                         "trc_pairs.py", "trc_model.py", "trc_train.py")

SPLITS = PC.SPLITS
N_BUCKETS = PC.N_BUCKETS
BUCKET_LABELS = PC.BUCKET_LABELS
FRAME_PAIRS = PC.FRAME_PAIRS
MAX_OBJECT_ATOMS = PC.MAX_OBJECT_ATOMS
ATOM_COLUMNS = PC.ATOM_COLUMNS
BOND_COLUMNS = PC.BOND_COLUMNS
NULL_BUCKET = N_BUCKETS - 1
CELL_NAMES = PP.CELL_NAMES

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
module_hashes = PM.module_hashes
state_hash = PM.state_hash
apply_atom_bond_support = PM.apply_atom_bond_support
category_support = PM.category_support
lr_at = PTR.lr_at
choose_horizon = PTR.choose_horizon


def load_config() -> dict[str, Any]:
    import json

    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def data_dir() -> Path:
    return ROUND_DIR / "data"


def analysis_dir() -> Path:
    return ROUND_DIR / "analysis"


def results_root() -> Path:
    override = os.environ.get("CSOI_RESULTS_DIR", "").strip()
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


def parent_module_provenance() -> dict[str, str]:
    return {name: sha256_file(PARENT_CODE / name) for name in REUSED_PARENT_MODULES}


def check_signature() -> dict[str, Any]:
    """Identity of the code that produced a number."""
    return {
        "commit": git_commit(),
        "parent_round": str(PARENT_ROUND.relative_to(REPO_ROOT)),
        "parent_modules": parent_module_provenance(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_id": load_config()["protocol_id"],
    }


# ---------------------------------------------------------------------------
# numeric helpers used by more than one entry point
# ---------------------------------------------------------------------------


def quantiles(values: np.ndarray, points: tuple[float, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    v = np.asarray(values, np.float64)
    out = {f"p{int(p)}": float(np.percentile(v, p)) for p in points}
    out["mean"] = float(np.mean(v))
    out["max"] = float(np.max(v)) if v.size else 0.0
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def paired_bootstrap(pred_control: np.ndarray, pred_candidate: np.ndarray,
                     labels: np.ndarray, n_boot: int, seed: int = 0) -> dict[str, float]:
    """RMSE(control) - RMSE(candidate) on the same resampled Valid molecules.

    Positive delta means the candidate is better.  This interval covers the
    evaluation-molecule sampling only; it does **not** include training-seed
    uncertainty (single seed 0 in this round).
    """
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
        "delta": float(delta),
        "lo": float(np.percentile(stats, 2.5)),
        "hi": float(np.percentile(stats, 97.5)),
        "n": int(n),
        "reps": int(n_boot),
        "seed": int(seed),
        "train_seed_uncertainty_included": False,
    }
