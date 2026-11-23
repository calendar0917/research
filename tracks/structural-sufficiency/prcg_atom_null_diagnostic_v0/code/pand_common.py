"""Shared plumbing for PRCG-ATOM-NULL.

This directory is a *diagnostic* on top of the frozen PRCG-v0 artefacts.  It
imports the PRCG-v0 code read-only; nothing in the parent round is modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
STRUCT = REPO_ROOT / "tracks" / "structural-sufficiency"
PARENT_DIR = STRUCT / "predictive_residual_composition_grammar_pilot_v0"
PARENT_CODE = PARENT_DIR / "code"
PARENT_CONFIG_PATH = PARENT_DIR / "configs" / "prcg.json"
PARENT_CONFIG_SHA256 = "d4afc405ec3723c27ccae956b689352c06d9aa358278077775b25e3e873c88ce"

if str(PARENT_CODE) not in sys.path:
    sys.path.insert(0, str(PARENT_CODE))

import prcg_common as C          # noqa: E402
import prcg_data as D            # noqa: E402
import prcg_discover as PD       # noqa: E402
import prcg_grammar as GD        # noqa: E402
import prcg_graph as G           # noqa: E402
import prcg_model as M           # noqa: E402
import prcg_checks as CK         # noqa: E402
import prcg_tensor as T          # noqa: E402

CONFIG_PATH = ROUND_DIR / "configs" / "atom_only.json"
RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"
TESTS_DIR = ROUND_DIR / "tests"
PROTOCOL_ID = "prcg-atom-null-diagnostic-v0"
ARMS_PARENT = ("FREQ_GRAMMAR", "PRED_GRAMMAR", "PRED_PORT_MEAN")
ARM_ATOM = "ATOM_ONLY"
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261202


def sha256_file(path: Path | str) -> str:
    return C.sha256_file(Path(path))


def parent_config() -> dict[str, Any]:
    sha = sha256_file(PARENT_CONFIG_PATH)
    if sha != PARENT_CONFIG_SHA256:
        raise ValueError(f"parent config sha mismatch: {sha}")
    return C.read_json(PARENT_CONFIG_PATH)


def load_config() -> dict[str, Any]:
    cfg = C.read_json(CONFIG_PATH)
    parent = parent_config()
    if cfg["parent_config_sha256"] != PARENT_CONFIG_SHA256:
        raise ValueError("new config does not name the frozen parent config")
    if cfg["protocol_id"] != PROTOCOL_ID:
        raise ValueError("unexpected protocol id")
    return cfg


def new_code_hashes() -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(CODE_DIR.glob("pand_*.py"))}


def git_commit() -> str:
    return C.git_commit()


def write_json(path: Path | str, payload: Any) -> Path:
    return C.write_json(path, payload)


def read_json(path: Path | str) -> Any:
    return C.read_json(path)


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    return C.write_csv(path, rows)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return C.rmse(pred, true)


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return C.mae(pred, true)


def bootstrap_indices(n: int, replicates: int = BOOTSTRAP,
                      seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """The same common-resample recipe used by PRCG-v0 analysis."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n), size=(int(replicates), int(n)))


def empty_grammar() -> G.Grammar:
    """The only new grammar arm: no merge rules at all.

    Every atom remains a single-atom word; every real bond is an inter-word
    bond.  The returned object is a fresh ``Grammar`` so callers cannot mutate
    a shared instance.
    """
    return G.Grammar()


def parent_results(arm: str, seed: int = 0) -> dict[str, Any]:
    path = PARENT_DIR / "results" / f"{arm}_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return C.read_json(path)