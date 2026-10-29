"""Shared plumbing for ZPR-v1 (ZINC performance-debug + recipe pilot).

The round does **not** add an architecture.  It reuses the frozen LCRC-A body
byte-for-byte (``zinc_local_relation_composition_v1/code/lc_*.py``, read-only)
and changes exactly one training detail in one arm: ``L1 -> SmoothL1(beta=0.05)``.
A passive parameter EMA is maintained in both arms as a secondary checkpoint
policy.

This module contains paths, frozen identities, the frozen valid-only group
masks and small metric helpers.  No model/data/training logic lives here.

Boundaries
----------
* official **train** labels + features: used (identical to LCRC-v1).
* official **valid** labels + features: used for checkpoint selection + analysis.
* official **test**: never constructed by the trainer; one descriptive read only
  after every checkpoint is frozen (``zpr_test.py``, ``TEST_AS_DEV_DESCRIPTIVE``).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "zinc-perf-recipe-pilot-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]
ANALYSIS = EXP_DIR / "analysis"
RUNS = EXP_DIR / "runs"
CONFIGS = EXP_DIR / "configs"

#: Read-only reuse of the frozen LCRC-A track (code, cache, norm, checkpoints).
LCRC_TRACK = REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1"
_LCRC_CODE = LCRC_TRACK / "code"
if str(_LCRC_CODE) not in sys.path:
    sys.path.insert(0, str(_LCRC_CODE))

import lc_common as L  # noqa: E402
import lc_data as D  # noqa: E402,F401
import lc_models as M  # noqa: E402,F401
import lc_train as T  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Frozen identities (verified by sha256 / fingerprint before any use)
# ---------------------------------------------------------------------------

FROZEN = {
    "train_cache_file_sha256": "8777494f7745ff5a9a8e3d2913cae8102ad84d267e54fb65235d7eff4c3a6cb2",
    "train_cache_content_sha256": "ded004e8a0c03af31413b0f8d0e8223d80c6ba424a2cbd34db42e16c9b85459b",
    "valid_cache_file_sha256": "aae2caa50d322a25ce75cbd8160e61a831b1488d43295e872db31d0bc7a6dbfe",
    "valid_cache_content_sha256": "f28a0cceed34444c39523e1d301a02350a67087c15b3de725850e45976212aef9",
    "ssl_s0_sha256": "006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97",
    "norm_fingerprint": "8c3e88137f18ca6841758b5779b57d7362d9dd57c2c29db2067f822cafd1c89f",
    "feature_version": "lcrc-v1-atom108-s1joint-s2typedpath",
    "lcrc_a_s0_initial_state_sha256": "11c698ae669dbee1bca93ccc13048163e725b29659d54fe6498fa6440a0cfb1e",
    "lcrc_a_s0_best_sha256": "06340bcd9e45af03e887e706031b6ea14c719cac2b68ba716ce3d686bb21fdad",
    "params": 237519,
    "n_train": 10000,
    "n_valid": 1000,
    "total_updates": 47400,
    "steps_per_epoch": 79,
    "ema_start_step": 11850,
}

#: Historical anchor (read-only; never retrained in this round).
HIST = {
    "lcrc_a_s0_valid": 0.07658163924043765,
    "lcrc_a_s0_epoch": 476,
    "lcrc_a_s0_test": 0.066782,
    "dcc_l_s0_test": 0.066267,
    "dcc_l_s0_valid": 0.078537,
    "blend_s0_test": 0.058831,
    "blend_s0_valid": 0.070932,
    "blend_4seed_test": 0.057300,
    "blend_4seed_valid": 0.067534,
}

LCRC_A_S0_RUN = LCRC_TRACK / "runs/lcrcA_s0"
LCRC_A_S0_CURVE = LCRC_A_S0_RUN / "curve.csv"

ROW172 = 172
#: ``zinc_generalization_audit_v1`` fixed worst-10 (P3B_s0 top-10 AE, frozen
#: before other models were scored).  Reused verbatim; *not* re-selected here.
FIXED_WORST10_JSON = (
    REPO / "tracks/structural-sufficiency/zinc_generalization_audit_v1/analysis/fixed_worst10.json"
)
ANCHOR_CSV = (
    REPO / "tracks/structural-sufficiency/zinc_performance_search_v1/analysis/per_molecule_valid.csv"
)

#: EMA policy (frozen in PROTOCOL.md §2.2).
EMA_DECAY = 0.999
EMA_START_FRAC = 0.25
SMOOTH_L1_BETA = 0.05

ARMS = {
    "G0": {"loss": "l1", "label": "L1 control"},
    "G1": {"loss": "smooth_l1", "label": "SmoothL1(beta=0.05) candidate"},
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def track_rel(path) -> str:
    return L.track_rel(path)


def read_json(path):
    return L.read_json(path)


def write_json(path, obj) -> None:
    L.write_json(path, obj)


def now() -> str:
    return L.now()


def ensure_dir(path):
    return L.ensure_dir(path)


def sha256_file(path) -> str:
    return L.C.sha256_file(path)


def log(msg: str) -> None:
    L.log(msg)


# ---------------------------------------------------------------------------
# frozen valid-only group masks
# ---------------------------------------------------------------------------

def read_anchor() -> dict:
    """Official valid-1000 row order (0..999) with the frozen grouping columns."""
    with open(ANCHOR_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["row_id"]))
    row_id = np.asarray([int(r["row_id"]) for r in rows], dtype=np.int64)
    if not np.array_equal(row_id, np.arange(len(rows))):
        raise RuntimeError("anchor rows are not 0..999 in order")
    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    unseen = np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows], dtype=np.float64) > 0
    n_rings = np.asarray([int(float(r["n_rings"])) for r in rows], dtype=np.int64)
    return {"row_id": row_id, "y": y, "unseen": unseen, "n_rings": n_rings}


def fixed_worst10_rows() -> np.ndarray:
    payload = read_json(FIXED_WORST10_JSON)
    rows = np.zeros(FROZEN["n_valid"], dtype=bool)
    for r in payload["row_ids"]:
        rows[int(r)] = True
    if int(rows.sum()) != 10:
        raise RuntimeError(f"fixed worst10 has {int(rows.sum())} rows, expected 10")
    return rows


def group_masks(anchor: Mapping | None = None) -> dict[str, np.ndarray]:
    """The frozen valid grouping used by every table in this round."""
    a = read_anchor() if anchor is None else anchor
    y = a["y"]
    idx = np.arange(len(y))
    worst = fixed_worst10_rows()
    ordinary = y > -3.0
    return {
        "full": np.ones(len(y), dtype=bool),
        "ordinary": ordinary,
        "r2_unseen": a["unseen"],
        "r2_unseen_ordinary": a["unseen"] & ordinary,
        "seen_ordinary": (~a["unseen"]) & ordinary,
        "minus_row172": idx != ROW172,
        "minus_fixed_worst10": ~worst,
        "tail": y <= -3.0,
        "row172": idx == ROW172,
        "fixed_worst10": worst,
    }


def mae(pred: Sequence[float], y: Sequence[float], mask=None) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(y, dtype=np.float64).reshape(-1)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {t.shape}")
    if mask is None:
        return float(np.mean(np.abs(p - t)))
    m = np.asarray(mask, dtype=bool)
    return float(np.mean(np.abs(p[m] - t[m])))


def window_diagnostics(values: Iterable[float], *, w20: int = 20, t100: int = 100) -> dict:
    """W20 / T100, identical to the frozen LCRC ``_window_diagnostics``."""
    vals = [float(v) for v in values]
    if not vals:
        return {"W20": None, "T100": None}
    w = None
    if len(vals) >= w20:
        w = min(float(np.median(vals[i:i + w20])) for i in range(len(vals) - w20 + 1))
    return {"W20": w, "T100": float(np.median(vals[-min(t100, len(vals)):]))}


def group_table(pred, y, masks: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {name: mae(pred, y, mask) for name, mask in masks.items()}


__all__ = [
    "ROUND_ID", "CODE_DIR", "EXP_DIR", "REPO", "ANALYSIS", "RUNS", "CONFIGS",
    "LCRC_TRACK", "LCRC_A_S0_RUN", "LCRC_A_S0_CURVE",
    "L", "D", "M", "T", "FROZEN", "HIST", "ROW172", "ANCHOR_CSV",
    "EMA_DECAY", "EMA_START_FRAC", "SMOOTH_L1_BETA", "ARMS",
    "track_rel", "read_json", "write_json", "now", "ensure_dir", "sha256_file", "log",
    "read_anchor", "fixed_worst10_rows", "group_masks", "mae", "window_diagnostics",
    "group_table",
]
