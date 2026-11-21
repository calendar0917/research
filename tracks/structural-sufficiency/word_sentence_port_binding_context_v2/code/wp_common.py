"""Identity, paths, frozen-config access and numeric helpers (port-binding v2).

Only IO, hashing and frozen identity live here.  The word/structure artefacts of
``word_sentence_port_route_v1`` are re-used **read-only** through ``ws_words`` so
that word types, merge history, exact canonicalisation and the ``R(D;p,q)``
primitives are literally the same code, not a re-implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
CONFIG_PATH = ROUND_DIR / "configs" / "wp_v2.json"

REFERENCE_COMMIT = "60285ae8"
OLD_ROUND = REPO_ROOT / "tracks" / "structural-sufficiency" / "word_sentence_port_route_v1"
OLD_CODE = OLD_ROUND / "code"
RD_CODE = REPO_ROOT / "tracks" / "structural-sufficiency" / "response_dictionary_small_s0_v1" / "code"
SQ_CODE = REPO_ROOT / "tracks" / "structural-query-learning" / "code"
for _p in (RD_CODE, SQ_CODE, OLD_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rd_common as RC  # noqa: E402
import sq_common as SQ  # noqa: E402
import ws_common as WC  # noqa: E402  (old round: dataset caches, config plumbing)
import ws_words as W  # noqa: E402  (old round: templates, canonicalisation, R descriptor)

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
R_DIM = W.R_DIM
PAIR_FIELD_DIM = W.PAIR_FIELD_DIM
OLD_CACHE_VERSION = "ws-word-route-v1"
CACHE_VERSION = "portbind-v2"


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
    import os

    override = os.environ.get("WP_RESULTS_DIR", "").strip()
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


def old_freeze_path(dataset: str) -> Path:
    return OLD_ROUND / "data" / f"{dataset}_words_{OLD_CACHE_VERSION}_freeze.json"


def old_word_path(dataset: str) -> Path:
    return OLD_ROUND / "data" / f"{dataset}_words_{OLD_CACHE_VERSION}.npz"


def old_record_path(dataset: str) -> Path:
    return OLD_ROUND / "analysis" / f"structure_freeze_{dataset.replace('ogbg-', '')}.json"


def derived_path(dataset: str) -> Path:
    return data_dir() / f"{dataset}_portbind_{CACHE_VERSION}.npz"


def derived_record_path(dataset: str) -> Path:
    return analysis_dir() / f"structure_freeze_{dataset.replace('ogbg-', '')}.json"


def content_sha256(payload: Mapping[str, Any]) -> str:
    """Machine-independent identity over every ndarray value (sorted by key)."""
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(k for k in payload if isinstance(payload[k], np.ndarray)):
        array = np.ascontiguousarray(payload[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
