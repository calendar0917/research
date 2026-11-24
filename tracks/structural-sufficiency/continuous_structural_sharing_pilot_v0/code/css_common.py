"""Shared plumbing for CSS-v0 (continuous structural sharing pilot).

Only paths, frozen identities, hashing, IO, metrics and seeding live here.
Scientific choices are frozen in ``configs/css.json`` and ``PROTOCOL.md``.

Reused read-only sources (byte-pinned):

* ``gwic_model.py``  — the frozen R2/SENTENCE_BASE predictor (use_words=False);
* ``ses_data.py``    — official Train/Valid/Test adapter (3360/420/420);
* ``ses_structure.py``/``ses_tensor.py`` — complete-graph struct + collation for
  exactly the batch keys consumed by ``GwicModel(use_words=False)``.

Nothing in those files is modified by this round.
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
import torch

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
STRUCT = REPO_ROOT / "tracks" / "structural-sufficiency"

GWIC_ROUND = STRUCT / "graph_word_incidence_composition_pilot_v0"
GWIC_CODE = GWIC_ROUND / "code"
GWIC_CONFIG = GWIC_ROUND / "configs" / "gwic.json"
SES_ROUND = STRUCT / "structural_evidence_selection_pilot_v0"
SES_CODE = SES_ROUND / "code"

for _path in (str(GWIC_CODE), str(SES_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gwic_model as GM  # noqa: E402  (frozen predictor)
import ses_data as SD  # noqa: E402  (official split adapter)
import ses_structure as SS  # noqa: E402
import ses_tensor as ST  # noqa: E402

CONFIG_PATH = ROUND_DIR / "configs" / "css.json"
RESULTS_DIR = ROUND_DIR / "results"
PREP_DIR = RESULTS_DIR / "prep"
ANALYSIS_DIR = ROUND_DIR / "analysis"

RAW_CACHE = REPO_ROOT / "tracks" / "structural-query-learning" / "data" / "ogbg_mollipo_sq-pilot-cache-v1.npz"
RAW_CACHE_SHA256 = "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c"
LABEL_VAULT = REPO_ROOT / "data" / "lipophilicity" / "labels.npy"
LABEL_VAULT_SHA256 = "af22781a3fc970510ffafcb3388391b64d02701e1ead15724c7d6850e8265de3"

GWIC_MODEL_SHA256 = "66926facafc97935905ba32ae4771f2a15585a34e0749f26cdb2fb8dff187e73"
GWIC_COMMON_SHA256 = "6d081769940ed00682c95c8a7c8be382f1592763db269c3067fc14bc3f1a97cb"
SES_DATA_SHA256 = "1a64ceb3aa5c218035a227eeac03c3e1b3dbe79bb7cb13dece2fc163ae70d498"
SES_STRUCTURE_SHA256 = "9029deb22519f5efcfe8ec6bd61cfc06648a381131b474ac07f1f7358f8ef1a3"
SES_TENSOR_SHA256 = "6db35d431edfeedbc98c3ec502c1faa1f6e41e08d5fa566972ffb016b75814d7"

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
N_ATOM_TYPES = 119
N_BOND_TYPES = 5
FROZEN_PREDICTOR_PARAMS = 416177

ARM_TO_BRANCH = {"DIRECT": "z", "SHARE": "q"}
ARMS = ("DIRECT", "SHARE")
SIZES = (100, 50)


class FrozenSourceMismatch(RuntimeError):
    pass


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


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]],
              columns: Sequence[str] | None = None) -> None:
    import csv

    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k)) for k in columns})


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, np.float64) - np.asarray(y_true, np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, np.float64)
                                - np.asarray(y_true, np.float64))))


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def code_sha(files: Sequence[str]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        path = CODE_DIR / name
        h.update(name.encode())
        h.update(sha256_file(path).encode())
    return h.hexdigest()


def load_config() -> dict[str, Any]:
    return dict(read_json(CONFIG_PATH))


def assert_frozen_sources() -> dict[str, str]:
    pins = {
        GWIC_CODE / "gwic_model.py": GWIC_MODEL_SHA256,
        GWIC_CODE / "gwic_common.py": GWIC_COMMON_SHA256,
        SES_CODE / "ses_data.py": SES_DATA_SHA256,
        SES_CODE / "ses_structure.py": SES_STRUCTURE_SHA256,
        SES_CODE / "ses_tensor.py": SES_TENSOR_SHA256,
        RAW_CACHE: RAW_CACHE_SHA256,
    }
    got = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in pins}
    for path, expected in pins.items():
        name = str(path.relative_to(REPO_ROOT))
        if got[name] != expected:
            raise FrozenSourceMismatch(f"{name}: {got[name]} != {expected}")
    vault = {"available": LABEL_VAULT.is_file(), "sha256": None}
    if vault["available"]:
        vault["sha256"] = sha256_file(LABEL_VAULT)
        if vault["sha256"] != LABEL_VAULT_SHA256:
            raise FrozenSourceMismatch("label vault sha mismatch")
    return {"sources": got, "label_vault": vault}


def model_norm(bundle: Any) -> dict[str, np.ndarray]:
    return {
        "atom_scalar_mean": np.asarray(bundle.atom_scalar_mean, np.float64),
        "atom_scalar_scale": np.asarray(bundle.atom_scalar_scale, np.float64),
        "graph_mean": np.asarray(bundle.graph_mean, np.float64),
        "graph_scale": np.asarray(bundle.graph_scale, np.float64),
    }


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float,
          minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(peak) * (step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return float(minimum) + 0.5 * (float(peak) - float(minimum)) * (
        1.0 + math.cos(math.pi * progress))
