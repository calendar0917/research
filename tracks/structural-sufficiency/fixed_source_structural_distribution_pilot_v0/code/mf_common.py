"""Shared constants / paths for the MEANFIELD distribution pilot.

The round only adds one computation on top of the *frozen* DSCR FIXED1 arm.
Everything else (data, W2 trunk, S1/S2 local content, both FIXED1 blocks,
LayerNorm, ``bond_proj``, residuals, FFN, ``f``, ``rho``, attention readout,
initialisation and the training recipe) is imported read-only from

    tracks/structural-sufficiency/direct_structural_computation_recurrence_v0/code
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]
SS = REPO / "tracks/structural-sufficiency"
FIXED1_ROUND = SS / "direct_structural_computation_recurrence_v0"

for _p in (str(FIXED1_ROUND / "code"), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ds_common as D  # noqa: E402  (frozen DSCR definitions)
import ds_model as DM  # noqa: E402  (frozen FIXED1 model)

# ---------------------------------------------------------------------------
# round identity
# ---------------------------------------------------------------------------

ROUND_ID = "fixed-source-structural-distribution-pilot-v0"
PROTOCOL_ID = "fsdp-v0"
SCHEMA_RUN = "fsdp-run-v1"
SCHEMA_CHECKPOINT = "fsdp-checkpoint-v1"
SCHEMA_CHECKS = "fsdp-allocation-checks-v1"
SCHEMA_MANIFEST = "fsdp-checkpoint-manifest-v1"
SCHEMA_SUMMARY = "fsdp-summary-v1"

ARM = "MEANFIELD"
REFERENCE_ARM = "FIXED1"
SEED = 0
HORIZON = 12
TRAIN_EVAL_EPOCHS = (4, 8, 12)
BATCH_SIZE = 128
STEPS_PER_EPOCH = int(D.STEPS_PER_EPOCH)
PARAMS_TOTAL = 1_128_468

#: Textual statement of the information-access difference, carried into every
#: run record so that no reader can mistake MEANFIELD for an information-
#: equivalent variant of FIXED1.
ACCESS_NOTE = (
    "MEANFIELD keeps every real directed edge, every per-centre in-edge count and every "
    "per-centre bond-type count, but replaces each edge's message content by the mean of "
    "its (graph_id, raw_bond_feature_signature) group. This introduces molecule-wide "
    "information sharing inside a bond-type group, so it is NOT an information-access-"
    "equivalent variant of FIXED1."
)

# ---------------------------------------------------------------------------
# directories
# ---------------------------------------------------------------------------

ANALYSIS = ROUND / "analysis"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "results/runs"
RESULTS = ROUND / "results"
TESTS = ROUND / "tests"
RUN_DIR = RUNS / f"{ARM}_seed{SEED}"

#: frozen historical reference curves (read-only)
REF_RUNS = FIXED1_ROUND / "results/runs"
REF_RUN_DIR = REF_RUNS / f"{REFERENCE_ARM}_seed{SEED}"
W2_RUN_DIR = SS / "lcrc_pcba_backbone_scaling_v1/runs/W2"
SELF_RUN_DIR = REF_RUNS / "SELF_seed0"

#: historical FIXED1 12-epoch prefix, re-verified from ``REF_RUN_DIR/train_curve.csv``
HISTORICAL_FIXED1_PREFIX = {
    "best_valid_ap": 0.2699915873887504,
    "best_epoch": 12,
    "endpoint_valid_ap": 0.2699915873887504,
    "last3_valid_ap_mean": 0.26325568958752027,
    "train_ap_epoch12": 0.4479189031400282,
    "valid_bce_epoch12": 0.045382457113853594,
}


def ensure_dirs() -> None:
    for path in (ANALYSIS, CONFIGS, RUNS, RESULTS, TESTS):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [name for name in dir() if not name.startswith("_")]
