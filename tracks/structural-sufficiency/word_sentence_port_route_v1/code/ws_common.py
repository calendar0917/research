"""Identity, paths, configuration and numeric helpers for the word/sentence route.

Nothing scientific is chosen here: only IO, hashing and frozen identity checks.
The frozen OGB raw-graph caches of ``tracks/structural-query-learning`` are
re-used read-only through the same verification entry point as the previous
response-dictionary round.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
CONFIG_PATH = ROUND_DIR / "configs" / "ws_v1.json"

RD_CODE = REPO_ROOT / "tracks" / "structural-sufficiency" / "response_dictionary_small_s0_v1" / "code"
SQ_CODE = REPO_ROOT / "tracks" / "structural-query-learning" / "code"
for _p in (RD_CODE, SQ_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rd_common as RC  # noqa: E402
import sq_common as SQ  # noqa: E402

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
#: Word-type atom label columns (official OGB schema names, verified at load).
WORD_ATOM_COLUMNS: tuple[int, ...] = (0, 3, 7)
#: Word-type internal bond label columns.
WORD_BOND_COLUMNS: tuple[int, ...] = (0,)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def official_dims() -> dict[str, list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != ATOM_COLUMNS or len(bond) != BOND_COLUMNS:
        raise ValueError("unexpected official OGB feature definition")
    return {"atom": atom, "bond": bond}


def results_root() -> Path:
    override = os.environ.get("WS_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return ROUND_DIR / "results"


def data_dir() -> Path:
    return ROUND_DIR / "data"


def analysis_dir() -> Path:
    return ROUND_DIR / "analysis"


def sha256_file(path: Path | str) -> str:
    return SQ.sha256_file(path)


def sha256_array(array: np.ndarray) -> str:
    return SQ.sha256_array(np.ascontiguousarray(array))


def sha256_json(payload: Any) -> str:
    return SQ.sha256_json(payload)


def write_json(path: Path | str, payload: Any) -> Path:
    return SQ.write_json(path, payload)


def read_json(path: Path | str) -> Any:
    return SQ.read_json(path)


def write_csv(path: Path | str, rows, columns=None) -> Path:
    return SQ.write_csv(path, rows, columns)


def rmse(pred, target) -> float:
    return SQ.rmse(np.asarray(pred), np.asarray(target))


def mae(pred, target) -> float:
    return SQ.mae(np.asarray(pred), np.asarray(target))


def dataset_cache(dataset: str) -> dict[str, Any]:
    return RC.dataset_cache(dataset)


def target_scaler(payload: Mapping[str, Any]) -> dict[str, float]:
    return RC.target_scaler(payload)


def environment_snapshot() -> dict[str, Any]:
    return SQ.environment_snapshot()


def git_commit() -> str | None:
    return SQ.git_commit()


def jsonable(value: Any) -> Any:
    return SQ.jsonable(value)


def check_word_columns(atom_names, bond_names) -> dict[str, Any]:
    """Verify the word-type columns by name against the official schema."""
    atom_expected = ("atomic_num", "formal_charge", "is_aromatic")
    bond_expected = ("bond_type",)
    got_atoms = tuple(atom_names[i] for i in WORD_ATOM_COLUMNS)
    got_bonds = tuple(bond_names[i] for i in WORD_BOND_COLUMNS)
    if got_atoms != atom_expected:
        raise ValueError(f"word atom columns mismatch: {got_atoms} != {atom_expected}")
    if got_bonds != bond_expected:
        raise ValueError(f"word bond columns mismatch: {got_bonds} != {bond_expected}")
    return {"atom_columns": list(WORD_ATOM_COLUMNS), "atom_names": list(got_atoms),
            "bond_columns": list(WORD_BOND_COLUMNS), "bond_names": list(got_bonds)}
