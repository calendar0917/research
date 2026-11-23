"""Shared plumbing for WOBR-v0 (word-owned bilinear response pilot).

Only paths, hashing, IO, metrics, seeding and provenance live here.  Every
scientific choice is frozen in ``configs/wobr.json`` and ``PROTOCOL.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
STRUCT = REPO_ROOT / "tracks" / "structural-sufficiency"
GWIC_DIR = STRUCT / "graph_word_incidence_composition_pilot_v0"

RAW_CACHE = REPO_ROOT / "tracks" / "structural-query-learning" / "data" / "ogbg_mollipo_sq-pilot-cache-v1.npz"
RAW_CACHE_SHA256 = "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c"

SCAFFOLD_SPLIT = (STRUCT / "acra_relation_generalization_pilot_v1" / "data"
                  / "split_train_sub_dev.json")
SCAFFOLD_SPLIT_SHA256 = "60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e"

CONFIG_PATH = ROUND_DIR / "configs" / "wobr.json"
RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
EXPECTED_SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}

# Frozen GWIC sources that were copied and then modified for this round.
# These are provenance pins, not runtime imports: no GWIC checkpoint, model or
# learned statistic is loaded.
GWIC_SOURCE_PINS = {
    "gwic_structure.py": "97e2e35b272e36043e03a476f58fb6a2edeb3beefc60b15e967c1096c65ecb56",
    "gwic_tensor.py": "f969c1fd47255b3b4b897f540f501671def42ec45ec9515751ca7288204bedd0",
    "gwic_model.py": "66926facafc97935905ba32ae4771f2a15585a34e0749f26cdb2fb8dff187e73",
    "gwic_data.py": "f02863be6e9f37a593e234c4cb7c5e1d9a760088be0c903aad36619a46aa0cef",
    "gwic_common.py": "6d081769940ed00682c95c8a7c8be382f1592763db269c3067fc14bc3f1a97cb",
}

# Label-free structure members passed into the structure/collate path.  The
# labels and the official split row ids are loaded separately in ``wobr_data``
# and never enter a model batch.
RAW_STRUCT_KEYS = (
    "dataset", "cache_version", "node_ptr", "bond_ptr", "atom_feat", "bond_feat",
    "bond_u", "bond_v", "degree", "min_ring", "in_cycle",
    "graph_n_components", "graph_cycle_rank", "graph_n_bonds", "graph_n_atoms",
    "smiles", "row_ids",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def state_fingerprint(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        digest.update(str(name).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.dtype.str.encode())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


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


def assert_gwic_sources() -> dict[str, str]:
    """Verify the copied parent sources are still byte-identical to their pins."""
    found: dict[str, str] = {}
    for name, expected in GWIC_SOURCE_PINS.items():
        path = GWIC_DIR / "code" / name
        sha = sha256_file(path)
        if sha != expected:
            raise RuntimeError(f"frozen GWIC source changed: {name} {sha} != {expected}")
        found[name] = sha
    return found