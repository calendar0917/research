"""Shared plumbing for ``zinc_adaptive_arch_search_v1``.

Everything structural (caches, collation, relation caches, evaluation loop,
frozen recipe) is imported unchanged from the frozen BOSI rounds:

* ``bjm_common`` — paths to the frozen official train/valid caches, the
  label-free relation caches, the train-only global-33 standardisation, the
  train-only masked-SSL encoders, ``predict`` / ``mae`` / ``curve_metrics`` /
  ``radius2_support``.
* ``bosi_data`` — the deterministic half-edge relation construction and
  ``collate``.
* ``bosi_models`` — the frozen BOSI v0 modules.

Only round-local paths, the frozen arm table and the frozen reference numbers
of this round live here.  The official test cache is never referenced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SUFF = ROUND.parent
REPO = SUFF.parents[1]

BJM = SUFF / "zinc_bosi_joint_moment_v1" / "code"
BOSI_V0 = SUFF / "zinc_bosi_architecture_v0" / "code"
V1 = SUFF / "zinc_arch_discovery_v1" / "code"
for _p in (str(HERE), str(BJM), str(BOSI_V0), str(V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import bjm_common as BC  # noqa: E402

ROUND_ID = "zinc-adaptive-arch-search-v1"
CONFIGS = ROUND / "configs"
CONFIG = CONFIGS / "aas_v1.json"
RUNS = ROUND / "runs"
RUNS_SMOKE = ROUND / "runs_smoke"
# Remote jobs write their JSON into a gitignored directory so the compute
# checkout never becomes dirty (a dirty remote tree blocks the next deploy).
ANALYSIS = Path(os.environ.get("AAS_ANALYSIS_DIR", str(RUNS / "_analysis")))

#: exactly the frozen BOSI / BJM official caches and recipe (no re-derivation).
DATA_TRAIN = BC.DATA_TRAIN
DATA_VALID = BC.DATA_VALID
NORM_JSON = BC.NORM_JSON
RECIPE = dict(BC.RECIPE)
PARAM_CAP = 500_000

#: BOSI v0 constants reused verbatim.
RANK = 32
WIDTH = 96
N_ATOM_FEAT = 108
BOND_FEAT = 7
GLOBAL_DIM = 33
N_REL = 4
SIDE_IN_BASE = 617
PAIR_DIM = 32

#: New modules of this round: ``pair_phi`` (128 -> 64 -> 32) + the 32 extra
#: input columns of ``side.lins.0`` (128 x 32).
PAIR_PHI_PARAMS = 128 * 64 + 64 + 64 * 32 + 32          # 10,336
SIDE_WIDEN_PARAMS = 128 * PAIR_DIM                       # 4,096
BOSI_B_PARAMS = 473_889
EXPECTED_PARAMETERS = {
    "A1": BOSI_B_PARAMS + PAIR_PHI_PARAMS + SIDE_WIDEN_PARAMS,
    "A0": BOSI_B_PARAMS + PAIR_PHI_PARAMS + SIDE_WIDEN_PARAMS,
}
ARMS = ("A1", "A0")

#: Frozen reference numbers of *this* repository, quoted from their own
#: reports.  Only ``B_s0`` below is a same-protocol, same-seed, same-recipe
#: matched baseline with a frozen per-molecule prediction file in-repo.
REFERENCE = {
    "B_s0": {
        "valid_best": 0.08240547278517624,
        "W20": 0.08320282487751683,
        "T100": 0.08324073242762825,
        "params": 473_889,
        "source": "tracks/structural-sufficiency/zinc_bosi_joint_moment_v1/runs/B_s0/train.json",
    },
    "DCC_L_s0": {"valid_best": 0.078537, "params": 494_193,
                 "source": "zinc_performance_search_v1 (B-R0 seed 0, historical)"},
    "DCC_L_mean4": {"valid_best": 0.075990, "params": 494_193,
                    "source": "zinc_performance_search_v1 (B-R0 4 seeds)"},
    "P3B_s0": {"valid_best": 0.076224, "params": 413_505,
               "source": "zinc_recipe_trunk_v2 / ad2 baseline"},
}


def run_dir(tag: str, *, smoke: bool = False) -> Path:
    return (RUNS_SMOKE if smoke else RUNS) / tag


def load_rel(split: str) -> dict:
    return BC.load_rel(split)


def load_cache(split: str) -> dict:
    return BC.load_cache(split)


def load_norm() -> dict:
    return BC.load_norm()


def predict(model, cache, rel, rows, *, device, batch_size=256):
    return BC.predict(model, cache, rel, rows, device=device, batch_size=batch_size)


mae = BC.mae
curve_metrics = BC.curve_metrics
radius2_support = BC.radius2_support
molecules_from_cache = BC.molecules_from_cache
write_json = BC.write_json
read_json = BC.read_json
log = BC.log
now = BC.now
sha256_file = BC.sha256_file
sha256_array = BC.sha256_array
sha256_bytes = BC.sha256_bytes
rows_hash = BC.rows_hash
state_hash = BC.state_hash
rel_or_abs = BC.rel_or_abs
