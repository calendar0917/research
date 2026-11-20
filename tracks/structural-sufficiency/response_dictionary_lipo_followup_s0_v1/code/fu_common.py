"""Shared identity, paths and configuration for the Lipo follow-up round.

The parent round ``response_dictionary_small_s0_v1`` is imported **read-only**:
its frozen config, fragment caches, canonical initial state, model, training
recipe and dictionary code are reused unchanged.  This module only adds paths
for the new work directory and a protocol hash that covers the parent config
plus the follow-up decisions.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
S0_DIR = ROUND_DIR.parent / "response_dictionary_small_s0_v1"
S0_CODE = S0_DIR / "code"
S0_RESULTS = S0_DIR / "results"

if str(S0_CODE) not in sys.path:
    sys.path.insert(0, str(S0_CODE))

import rd_common as RC  # noqa: E402  (parent round, read-only)
import rd_dictionary as RD  # noqa: E402
import rd_fragment as RF  # noqa: E402
import rd_model as RM  # noqa: E402
import rd_train as RT  # noqa: E402

CONFIG_PATH = ROUND_DIR / "configs" / "fu_s0.json"
S0_CONFIG_PATH = S0_DIR / "configs" / "rd_small_s0.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_parent_config() -> dict[str, Any]:
    """The frozen S0 recipe (epochs, batch, lr, model sizes)."""
    return json.loads(S0_CONFIG_PATH.read_text(encoding="utf-8"))


def results_root() -> Path:
    override = os.environ.get("FU_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return ROUND_DIR / "results"


def run_dir(task: str, arm: str) -> Path:
    return results_root() / f"{task}_{arm}"


def protocol_payload() -> dict[str, Any]:
    return {"parent_config": load_parent_config(), "followup": load_config()}


def protocol_hash() -> str:
    return RC.SQ.sha256_json(protocol_payload())


#: Frozen identity of the parent artefacts this round reads.
EXPECTED_INIT_STATE_HASH = "ccc8171d4cdc47daadd226c382947a0e3e62f6c9266dcad6f952ba98a989072c"
EXPECTED_TRAIN_ORDER_SHA256_PREFIX = "361af4df"
EXPECTED_N_PARAMETERS = 198262
EXPECTED_FRAGMENT_CACHE = dict(RC.EXPECTED_FRAGMENT_CACHE_SHA256)
