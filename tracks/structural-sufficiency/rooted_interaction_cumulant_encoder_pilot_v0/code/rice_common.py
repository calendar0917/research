"""Shared plumbing for RICE-v0 (rooted interaction cumulant encoder pilot).

Only paths, hashing, IO, metrics, seeding and the frozen-parent import live here.
Every scientific choice is frozen in ``configs/rice.json`` and ``PROTOCOL.md``
before any formal training run.

The parent round ``graph_word_incidence_composition_pilot_v0`` is **read only**:
its label-safe data layer is imported byte-for-byte (sha256 pinned below) and no
hidden feature, embedding, checkpoint or statistic of any parent model is
reused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
STRUCT = REPO_ROOT / "tracks" / "structural-sufficiency"
GWIC_DIR = STRUCT / "graph_word_incidence_composition_pilot_v0"

# frozen parent sources reused for label-safe raw loading, split validation and
# label isolation only.  No model / feature / checkpoint module is imported.
GWIC_SOURCE_PINS = {
    "gwic_common.py": "6d081769940ed00682c95c8a7c8be382f1592763db269c3067fc14bc3f1a97cb",
    "gwic_data.py": "f02863be6e9f37a593e234c4cb7c5e1d9a760088be0c903aad36619a46aa0cef",
}

CONFIG_PATH = ROUND_DIR / "configs" / "rice.json"
RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
CHANNELS = 16
STATES = 2
KAPPA_DIM = CHANNELS * STATES      # 32 flattened (channel, root-state) channels
MAX_WORD_SIZE = 4
MAX_WORD_BONDS = 6


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def assert_frozen_parent() -> dict[str, str]:
    """Verify the reused parent data modules are byte-identical to their pins."""
    found: dict[str, str] = {}
    for name, expected in GWIC_SOURCE_PINS.items():
        path = GWIC_DIR / "code" / name
        sha = sha256_file(path)
        if sha != expected:
            raise RuntimeError(f"frozen parent source changed: {name} {sha} != {expected}")
        found[name] = sha
    return found


def import_parent_data() -> Any:
    """Import the pinned, label-safe parent data layer (structure loading only)."""
    assert_frozen_parent()
    path = str(GWIC_DIR / "code")
    if path not in sys.path:
        sys.path.insert(0, path)
    import gwic_data  # noqa: PLC0415  (deliberate late import after the pin check)

    return gwic_data


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if (np.isnan(value) or np.isinf(value)) else value
    return value


def sha256_json(payload: Any) -> str:
    return sha256_bytes(json.dumps(jsonable(payload), sort_keys=True).encode("utf-8"))


def array_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.shape).encode())
        h.update(arr.dtype.str.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def module_hashes(modules: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, module in modules.items():
        digest = hashlib.sha256()
        if hasattr(module, "state_dict"):
            items = sorted(module.state_dict().items())
        else:
            items = [("", module)]
        for pname, param in items:
            arr = param.detach().cpu().numpy()
            digest.update(pname.encode())
            digest.update(str(arr.shape).encode())
            digest.update(arr.dtype.str.encode())
            digest.update(np.ascontiguousarray(arr).tobytes())
        out[name] = digest.hexdigest()
    return out


def code_sha(paths: Sequence[str | Path]) -> str:
    h = hashlib.sha256()
    for name in sorted(str(p) for p in paths):
        path = Path(name)
        if not path.is_absolute():
            path = CODE_DIR / path
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(payload), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return target


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    keys = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})
    return target


def load_config() -> dict[str, Any]:
    return read_json(CONFIG_PATH)


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    return float(math.sqrt(np.mean((pred - true) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(true, np.float64))))


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    import subprocess

    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                      text=True)
        return bool(out.strip())
    except Exception:
        return True