"""SSTA v0: shared paths, pre-declared constants and IO helpers.

Round: ``tracks/structural-sufficiency/structural_state_action_v0``

Everything this round reads is frozen and read-only:

* ``structural_effect_diagnosis_v0/code/pcqm_response.py`` -- the real cell
  aggregation, support filter, data identity checks and the W3 inference path,
* ``structural_functional_equivalence_followup_v1`` -- canonical pair
  orientation, frozen PAIR_Q / LOCAL estimates, context masks,
* ``lcrc_pcqm4mv2_efficiency_width_v1`` -- the W3 1M checkpoint,
* ``lcrc_pcqm4mv2_transfer_pilot_v1/cache/pcqm_valid`` -- the Valid cache.

No model is trained, fine-tuned or distilled.  No weight is modified.  No new
molecule label is generated.  Official Test labels are never indexed.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

ANALYSIS_DIR = ROUND_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SED_DIR = TRACK_DIR / "structural_effect_diagnosis_v0"
SED_CODE = SED_DIR / "code"
SFEC_DIR = TRACK_DIR / "structural_functional_equivalence_followup_v1"
SFEC_CODE = SFEC_DIR / "code"
SFEC_DATA = SFEC_DIR / "data"
SFEC_ANALYSIS = SFEC_DIR / "analysis"
LCW_DIR = TRACK_DIR / "lcrc_pcqm4mv2_efficiency_width_v1"
LCW_CODE = LCW_DIR / "code"
LCW_CHECKPOINT = LCW_DIR / "runs" / "PCQM_W3_1M" / "checkpoint_best_valid.pt"
LC_OLD_DIR = TRACK_DIR / "lcrc_pcqm4mv2_transfer_pilot_v1"
LC_OLD_CACHE = LC_OLD_DIR / "cache"
LC_OLD_ANALYSIS = LC_OLD_DIR / "analysis"

for _p in (str(SED_CODE), str(SFEC_CODE), str(LCW_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ``sed_common`` already resolves every frozen parent path and the JSON / CSV
# writers used by the parent rounds; it is imported, never re-implemented.
import sed_common as C  # noqa: E402

write_json = C.write_json
write_csv = C.write_csv
read_json = C.read_json
peak_rss_gb = C.peak_rss_gb
sha256_file = C.sha256_file

# ---------------------------------------------------------------------------
# pre-declared analysis constants (fixed before any result is inspected)
# ---------------------------------------------------------------------------

#: ROUND / protocol identity
ROUND_ID = "structural-state-action-v0"
SCHEMA = "ssta-v0"

#: at most this many context pairs (= "rectangles") per canonical pair type
MAX_RECTANGLES_PER_PAIR = 64
#: at most this many unique molecules enter the record / state analysis
MAX_STATE_MOLECULES = 12_000
#: at most this many molecules enter the Section 4 reconstruction check
MAX_REPLAY_MOLECULES = 1_024
#: molecules used for the "same record -> same h/phi" grouping check
MAX_GROUP_MOLECULES = 2_000
#: molecules used for the atom-renumbering audit
RENUMBER_MOLECULES = 32
#: direction accuracy is only scored on |D_true| >= this value (eV)
DIRECTION_THRESHOLD_EV = 0.10
#: independent hash salt of every sampling rule in this round
HASH_SALT = "ssta-v0"
#: Section 6 support rule, declared before looking at any Section 6 number.
#: ``INSUFFICIENT_EXACT_ACTION_SUPPORT`` is reported if either bound fails.
ACTION_SUPPORT_MIN_RECTANGLES = 50
ACTION_SUPPORT_MIN_PAIR_TYPES = 10
#: bootstrap repetitions for every reported interval
BOOTSTRAP_REPS = 1_000

#: numerical tolerances of the equivalence checks (never used to loosen a
#: *structural* equality; only to absorb float32/float64 reassociation)
TOL_STATE_RECONSTRUCTION_EV = 1e-5
TOL_STATE_RECONSTRUCTION_REL = 1e-6
TOL_RECORD_DELTA_U_REL = 1e-5
TOL_RECORD_DELTA_U_ABS = 1e-4
TOL_SAME_RECORD_H = 1e-5
TOL_ATTENTION_SHIFT = 1e-9


def hash64(*parts: Any) -> int:
    """Deterministic, label-free 64-bit hash of a tuple of identifiers."""
    payload = "|".join([HASH_SALT] + [str(p) for p in parts]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def hash_order(*parts: Any) -> np.uint64:
    return np.uint64(hash64(*parts))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def cpu_seconds() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def torch_threads() -> int:
    import torch

    return int(torch.get_num_threads())


def set_torch_threads() -> int:
    import torch

    n = min(8, os.cpu_count() or 1)
    torch.set_num_threads(int(n))
    return int(torch.get_num_threads())


__all__ = [
    "ACTION_SUPPORT_MIN_PAIR_TYPES",
    "ACTION_SUPPORT_MIN_RECTANGLES",
    "ANALYSIS_DIR",
    "BOOTSTRAP_REPS",
    "CODE_DIR",
    "DIRECTION_THRESHOLD_EV",
    "HASH_SALT",
    "LCW_CHECKPOINT",
    "LCW_CODE",
    "LC_OLD_ANALYSIS",
    "LC_OLD_CACHE",
    "MAX_GROUP_MOLECULES",
    "MAX_RECTANGLES_PER_PAIR",
    "MAX_REPLAY_MOLECULES",
    "MAX_STATE_MOLECULES",
    "RENUMBER_MOLECULES",
    "REPO_ROOT",
    "ROUND_DIR",
    "ROUND_ID",
    "SCHEMA",
    "TOL_ATTENTION_SHIFT",
    "TOL_RECORD_DELTA_U_ABS",
    "TOL_RECORD_DELTA_U_REL",
    "TOL_SAME_RECORD_H",
    "TOL_STATE_RECONSTRUCTION_EV",
    "TOL_STATE_RECONSTRUCTION_REL",
    "cpu_seconds",
    "hash64",
    "hash_order",
    "json_safe",
    "peak_rss_gb",
    "read_json",
    "set_torch_threads",
    "sha256_file",
    "torch_threads",
    "write_csv",
    "write_json",
]
