"""Identity, paths, frozen-config access and numeric helpers (word response operator s0).

Only IO, hashing and frozen identity live here.  The frozen Train-only word
templates / merge history of ``word_sentence_port_route_v1`` are re-used
**read-only** through ``ws_words`` so that word types, canonicalisation and the
partition are literally the old code, not a re-implementation.  The raw
label-free OGB cache is the committed one of ``structural-query-learning``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
CONFIG_PATH = ROUND_DIR / "configs" / "wo_s0.json"

OLD_ROUND = REPO_ROOT / "tracks" / "structural-sufficiency" / "word_sentence_port_route_v1"
OLD_CODE = OLD_ROUND / "code"
OLD_FREEZE = OLD_ROUND / "analysis" / "structure_freeze_mollipo.json"
PAIRS_CSV = (REPO_ROOT / "tracks" / "structural-sufficiency"
             / "response_dictionary_task_transfer_s0_v1" / "analysis" / "pairs.csv")
RD_CODE = REPO_ROOT / "tracks" / "structural-sufficiency" / "response_dictionary_small_s0_v1" / "code"
SQ_CODE = REPO_ROOT / "tracks" / "structural-query-learning" / "code"
for _p in (RD_CODE, SQ_CODE, OLD_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rd_common as RC  # noqa: E402
import sq_common as SQ  # noqa: E402
import ws_common as WC  # noqa: E402
import ws_words as W  # noqa: E402

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
WORD_ATOM_COLUMNS = tuple(WC.WORD_ATOM_COLUMNS)
WORD_BOND_COLUMNS = tuple(WC.WORD_BOND_COLUMNS)
MAX_WORD_ATOMS = 6
REFERENCE_COMMIT = "09455737"
SPLITS = ("train", "valid", "test")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def official_dims() -> dict[str, list[int]]:
    diffs = WC.official_dims()
    return diffs


def data_dir() -> Path:
    path = ROUND_DIR / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def analysis_dir() -> Path:
    path = ROUND_DIR / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_root() -> Path:
    override = os.environ.get("WO_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return ROUND_DIR / "results"


def run_dir(task: str, arm: str) -> Path:
    return results_root() / f"{task}_{arm}"


def sha256_file(path: Path | str) -> str:
    return SQ.sha256_file(path)


def sha256_json(payload: Any) -> str:
    return SQ.sha256_json(payload)


def sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQ.write_json(path, payload)


def read_json(path: Path | str) -> Any:
    return SQ.read_json(path)


def write_csv(path: Path | str, rows, columns=None) -> Path:
    return SQ.write_csv(path, rows, columns)


def rmse(pred, target) -> float:
    return SQ.rmse(np.asarray(pred), np.asarray(target))


def mae(pred, target) -> float:
    return SQ.mae(np.asarray(pred), np.asarray(target))


def dataset_cache(dataset: str = "ogbg-mollipo") -> dict[str, Any]:
    return RC.dataset_cache(dataset)


def jsonable(value: Any) -> Any:
    return SQ.jsonable(value)


def set_determinism() -> None:
    """FP32 without TF32; deterministic cuDNN; frozen seed."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit() -> str | None:
    return SQ.git_commit()


def environment_snapshot() -> dict[str, Any]:
    return SQ.environment_snapshot()


def old_freeze() -> dict[str, Any]:
    return read_json(OLD_FREEZE)


def load_templates(freeze: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(freeze["template_table"]["templates"])


def content_sha256(payload: Mapping[str, np.ndarray]) -> str:
    """Order-independent content hash of a packed npz payload."""
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(payload):
        arr = np.ascontiguousarray(payload[key])
        digest.update(key.encode())
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def load_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def save_npz_atomic(path: Path | str, payload: Mapping[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **{k: v for k, v in payload.items()})
    os.replace(tmp, path)
    return path
