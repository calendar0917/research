"""Shared plumbing for the structural conditional moments (SCM) round.

Frozen before any GPU run:
  * BANDS   -- the two line-graph distance bands of the structural relation
               (``t=1``: distance 1-2, ``t=2``: distance 3-8; self excluded).
  * RANK    -- the shared low-rank Q/K/V width (8).
  * SCALES  -- one label-free global scale per band, the mean band size over the
               Fit10k (official train 10k) raw graphs.  S and A are both
               divided by the same constants in both arms; a per-query ``n_t``
               normalisation would erase the relative count and is refuted by
               construction here.

Nothing in this module reads labels, and nothing here reads the official test
split.  The only data paths are the frozen P3 object caches (train / valid).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_SUFF = _TRACK.parent
_REPO = _SUFF.parents[1]
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_V2 = _SUFF / "zinc_recipe_trunk_v2" / "code"
_DCC = _SUFF / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TRACK_ID = "zinc-structural-conditional-moments-v1"
CODE = _HERE
CACHE = _TRACK / "cache"
RUNS = _TRACK / "runs"
ANALYSIS = _TRACK / "analysis"
CONFIGS = _TRACK / "configs"

#: (lo, hi) inclusive line-graph distance bands.  Frozen by the round brief and
#: never re-tuned after any MAE is observed.
BANDS: tuple[tuple[int, int], ...] = ((1, 2), (3, 8))
#: Shared low-rank projection width for Q/K/V.
RANK = 8
#: Independent RNG offset for the new module (the backbone keeps the historical
#: DCC-L seed stream untouched).
SCM_SEED_OFFSET = 50_731
#: Number of *new* parameter tensors (Q/K/V/out) and total deployment budget.
NEW_TENSORS = 4
PARAM_CAP = 500_000
DCC_L_PARAMS = 494_193


def frozen_config() -> dict:
    """Read the committed frozen constants (bands, rank, per-band scales)."""
    path = CONFIGS / "scm_v1.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing frozen config {path}; the per-band scales must be frozen "
            "from Fit10k structure before any training run"
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if tuple(tuple(b) for b in cfg["bands"]) != BANDS:
        raise RuntimeError(f"config bands {cfg['bands']} != code bands {BANDS}")
    if int(cfg["rank"]) != RANK:
        raise RuntimeError(f"config rank {cfg['rank']} != code rank {RANK}")
    return cfg


def band_scales() -> tuple[float, float]:
    cfg = frozen_config()
    s = [float(v) for v in cfg["scales"]]
    if len(s) != len(BANDS) or not all(v > 0 for v in s):
        raise RuntimeError(f"invalid frozen scales {s}")
    return s[0], s[1]


def cache_path(name: str) -> Path:
    return CACHE / name


def relation_path(split: str) -> Path:
    return CACHE / f"lgdist_{split}.npz"


def json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "detach") and hasattr(obj, "tolist"):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)
