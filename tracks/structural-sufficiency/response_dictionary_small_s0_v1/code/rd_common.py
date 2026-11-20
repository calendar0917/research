"""Shared identity, paths, configuration and numeric helpers.

The round reuses the frozen OGB raw-graph cache of
``tracks/structural-query-learning``; this module only wires it to the new
work directory.  No scientific quantity is chosen here.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
CONFIG_PATH = ROUND_DIR / "configs" / "rd_small_s0.json"
DATA_DIR = ROUND_DIR / "data"

SQ_CODE = REPO_ROOT / "tracks" / "structural-query-learning" / "code"
if str(SQ_CODE) not in sys.path:
    sys.path.insert(0, str(SQ_CODE))

import sq_common as SQ  # noqa: E402
import sq_data as SD  # noqa: E402


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def protocol_hash(config: Mapping[str, Any] | None = None) -> str:
    payload = dict(config or load_config())
    return SQ.sha256_json(payload)


def results_root() -> Path:
    override = os.environ.get("RD_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return ROUND_DIR / "results"


def run_dir(task: str, arm: str) -> Path:
    return results_root() / f"{task}_{arm}"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def write_json(path: Path | str, payload: Any) -> Path:
    return SQ.write_json(path, payload)


def read_json(path: Path | str) -> Any:
    return SQ.read_json(path)


def write_csv(path: Path | str, rows, columns=None) -> Path:
    return SQ.write_csv(path, rows, columns)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return SQ.rmse(pred, target)


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return SQ.mae(pred, target)


#: Frozen SHA-256 of the committed fragment caches (built by ``rd_prepare.py``).
EXPECTED_FRAGMENT_CACHE_SHA256 = {
    "ogbg-molesol": "fac644d0d7b73f8bc06239853c97fe65cd3010ab02ea539f9a6bdf4f4852a62d",
    "ogbg-mollipo": "ba6e9a41990fc344e5962519caf02e616c9c67d299884843c8b67a5bbab0a1d8",
}


def fragment_cache(dataset: str) -> dict[str, Any]:
    import rd_fragment as RF

    return RF.load_dataset_cache(dataset, verify_sha=EXPECTED_FRAGMENT_CACHE_SHA256[dataset])


def dataset_cache(dataset: str) -> dict[str, Any]:
    """Load the frozen official OGB cache, refusing a changed byte identity."""
    filename = f"{dataset.replace('-', '_')}_{SD.CACHE_VERSION}.npz"
    expected = SQ.EXPECTED_CACHE_SHA256.get(filename)
    if expected is None:
        raise KeyError(f"no frozen SHA-256 for {filename}")
    payload = SD.load_cache(dataset, verify_sha=expected)
    payload["expected_cache_sha256"] = expected
    return payload


def target_scaler(payload: Mapping[str, Any]) -> dict[str, float]:
    train = np.asarray(payload["train_idx"], dtype=np.int64)
    return SQ.train_target_scaler(np.asarray(payload["labels"])[train])


def git_commit() -> str | None:
    return SQ.git_commit()


def environment_snapshot() -> dict[str, Any]:
    return SQ.environment_snapshot()


def source_hashes(paths) -> dict[str, str]:
    return {Path(p).name: sha256_file(p) for p in paths if Path(p).is_file()}


def jsonable(payload: Any) -> Any:
    return SQ.jsonable(payload)


def git_blob_sha256(commit: str, relpath: str) -> str:
    """SHA-256 of a file exactly as committed, for executed-code provenance."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{relpath}"],
        check=True, capture_output=True,
    ).stdout
    return hashlib.sha256(out).hexdigest()
