"""Shared plumbing for ``structure_route_resolution_8h_v1``.

The round is read-only with respect to every source round it reuses:

* ``rooted_interaction_cumulant_encoder_pilot_v0`` (RICE-v0): data layer,
  structure support, exact energy / root-conditioned Mobius, head dimensions and
  the training recipe for the energy arms.
* ``r2_sentence_replication_v1`` (R2): split loader, GWIC SENTENCE_BASE model,
  training recipe and the frozen D0/D1 identities.
* ``response_dictionary_task_transfer_s0_v1`` / ``response_dictionary_small_s0_v1``:
  pure bridge-cut enumeration, exact fragment identity and reassembly checks for
  the real-edit catalogue.
* ``task_response_composition_s0_v1``: only the orientation/dedupe convention for
  real molecule pairs.

Every pinned source is re-hashed at runtime.  No parent file is written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
STRUCT = REPO_ROOT / "tracks" / "structural-sufficiency"

RICE_DIR = STRUCT / "rooted_interaction_cumulant_encoder_pilot_v0"
RICE_CODE = RICE_DIR / "code"
R2_DIR = STRUCT / "r2_sentence_replication_v1"
R2_CODE = R2_DIR / "code"
TT_DIR = STRUCT / "response_dictionary_task_transfer_s0_v1"
TT_CODE = TT_DIR / "code"
RD_DIR = STRUCT / "response_dictionary_small_s0_v1"
RD_CODE = RD_DIR / "code"

for _p in (RICE_CODE, R2_CODE, TT_CODE, RD_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONFIG_PATH = ROUND_DIR / "configs" / "sprint.json"
RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"

# ---------------------------------------------------------------------------
# frozen source pins (sha256, verified on import)
# ---------------------------------------------------------------------------
SRC_PINS = {
    "rice_common.py": (RICE_CODE / "rice_common.py",
                       "4b1f9341690e69dfc74c1166e6eca82f53367c614d9dc5c61c8ac99db368f111"),
    "rice_data.py": (RICE_CODE / "rice_data.py",
                     "fa2a4d306e19f92ca6f976281a4898e1c925c3fa8078b714729ec39a4a4af73c"),
    "rice_structure.py": (RICE_CODE / "rice_structure.py",
                          "cb044e37e3c2bfd90881fca196b3e40fa7e05429bc389747da3957a56269629b"),
    "rice_energy.py": (RICE_CODE / "rice_energy.py",
                       "336242633e06fced2a0c275c9de8247a2f13bd3a7c3e15df6074fc47088180d5"),
    "rice_tensor.py": (RICE_CODE / "rice_tensor.py",
                       "de7df5ec67ba352f683544764ee53840cf8712573f1c9bc0cf04c9b49a87e697"),
    "rice_model.py": (RICE_CODE / "rice_model.py",
                      "4610d5736cc6fec185ebed39685212e4ffe0983e259ba5d9fb56c156bbc701be"),
    "rice_train.py": (RICE_CODE / "rice_train.py",
                      "7d24144a2767b3047756240f2454c83bf52a49fce3300b23cf162077b9196509"),
    "r2_common.py": (R2_CODE / "r2_common.py",
                     "6d0e0a5f7fc3d901ff89e2e24d04656e6fdf3cf9ae5a25b5ad1928561b9100c5"),
    "r2_data.py": (R2_CODE / "r2_data.py",
                   "a45806a2dcb76a0081a4198ccfcab5b1c6c82b0277c22e581523b9d5e2bfbb7a"),
    "r2_train.py": (R2_CODE / "r2_train.py",
                    "56be8f8ea60bd5a0ad7d5fcc3c456c247e8d8d07c4a862c759fce3b1b4336157"),
    "tt_catalogue.py": (TT_CODE / "tt_catalogue.py",
                        "7e0f0ed2fe4645dc9186225ab7544ae3cebdb8b2f207d6362c3a976be9e0d4b3"),
    "rd_fragment.py": (RD_CODE / "rd_fragment.py",
                       "38bf5f7918ec4e81799d18a05fd0fc03d940c7145d2a6600c0ffe155cbcb8f59"),
}

D0_IDENTITY = "60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e"
D1_IDENTITY = "c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(json.dumps(jsonable(payload), sort_keys=True).encode("utf-8"))


def assert_pins() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, (path, expected) in SRC_PINS.items():
        found = sha256_file(path)
        if found != expected:
            raise RuntimeError(f"pinned source changed: {name} {found} != {expected}")
        out[name] = found
    return out


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
    if isinstance(value, set):
        return sorted(jsonable(v) for v in value)
    return value


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(jsonable(payload), indent=1, sort_keys=True,
                                 ensure_ascii=False) + "\n", encoding="utf-8")
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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k)) for k in keys})
    return target


def append_jsonl(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")
    return target


def read_jsonl(path: Path | str) -> list[Any]:
    target = Path(path)
    if not target.is_file():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()]


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
        items = sorted(module.state_dict().items()) if hasattr(module, "state_dict") \
            else [("", module)]
        for pname, param in items:
            arr = param.detach().cpu().numpy()
            digest.update(pname.encode())
            digest.update(str(arr.shape).encode())
            digest.update(arr.dtype.str.encode())
            digest.update(np.ascontiguousarray(arr).tobytes())
        out[name] = digest.hexdigest()
    return out


def code_sha(files: Sequence[str | Path]) -> str:
    h = hashlib.sha256()
    for name in sorted(str(p) for p in files):
        path = Path(name)
        if not path.is_absolute():
            path = CODE_DIR / path
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    return float(np.mean(np.abs(np.asarray(pred, np.float64)
                                - np.asarray(true, np.float64))))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)