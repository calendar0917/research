"""Identity, paths, frozen constants, hashing and statistics for the
LCRC explanation task-relevance sprint.

No scientific value is decided in this module beyond the constants frozen in
``PROTOCOL.md``.  The module is import-safe on CPU and does not load any model.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-explanation-task-relevance-sprint-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

ANALYSIS_DIR = ROUND_DIR / "analysis"
RESULTS_DIR = ROUND_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
MEASURED_DIR = RESULTS_DIR / "measured"
ORACLE_DIR = RESULTS_DIR / "oracle"
PREPARED_DIR = RESULTS_DIR / "prepared"

#: Read-only source rounds.  They are imported, never written.
ROUTE_DIR = TRACK_DIR / "gnn_route_clarification_8h_v1"
ROUTE_CODE = ROUTE_DIR / "code"
PILOT_DIR = TRACK_DIR / "lcrc_auto_subgraph_faithfulness_pilot_v1"
PILOT_CODE = PILOT_DIR / "code"
RC_DIR = TRACK_DIR / "lcrc_readout_context_pilot_small_v1"
RC_CODE = RC_DIR / "code"
SQL_DIR = REPO_ROOT / "tracks/structural-query-learning"
SQL_CODE = SQL_DIR / "code"
PREP_DIR = TRACK_DIR / "lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_DIR / "code"
DISSECT_DIR = TRACK_DIR / "lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT_DIR / "code"

DATASET = "ogbg-mollipo"
SPLITS: tuple[str, ...] = ("train", "valid", "test")
EXPECTED_SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}

#: Frozen seeds / search constants (inherited, not re-tuned).
SAMPLE_SEED = 20261117
CONTROL_SEED = 20261117
BOOTSTRAP_SEED = 20261117
BOOTSTRAP_REPLICATES = 2000

SEARCH_MIN_K = 2
SEARCH_MAX_K = 6
BEAM_WIDTH = 12
TOP_K_SUBGRAPHS = 3
JACCARD_MAX_OVERLAP = 0.5

N_RANDOM = 16
MIN_RANDOM = 8
MAX_GROWTH_ATTEMPTS = 512
S2_MATCH_REL_TOL = 0.25

FD_EPS = 1e-3
FD_MOLECULES = 8
FD_BONDS_PER_MOLECULE = 4
RENUMBER_MOLECULES = 8

#: Decision thresholds frozen in PROTOCOL.md §10.
PREDICTION_RMSE_MAX = 0.80
PREDICTION_DIRECTION_MIN = 0.75
PAIR_ACCURATE_ABS = 0.25
ORACLE_PAIR_ACCURATE_ABS = 0.10
ORACLE_ENRICHMENT_MIN = 0.02
LOW_SUPPORT_COMPONENTS = 30

#: Raw-unit tolerance from the inherited protocol.
def trace_tolerance(value: float | np.ndarray) -> np.ndarray | float:
    return 1e-5 + 1e-6 * np.maximum(1.0, np.abs(value))

#: WC_LOGP contribution closure tolerance (stored float64 precision).
def oracle_tolerance(value: float | np.ndarray) -> np.ndarray | float:
    return 1e-8 + 1e-8 * np.maximum(1.0, np.abs(value))

#: Frozen asset names in this round.
PAIR_CATALOG = ANALYSIS_DIR / "pair_catalog.jsonl.gz"
ROW_ATOM_MAPPING = ANALYSIS_DIR / "row_atom_mapping.jsonl.gz"
ORACLE_NPZ = ANALYSIS_DIR / "ogbg-mollipo.npz"
ORACLE_SUMMARY = ANALYSIS_DIR / "oracle_contrib_summary.json"
SOURCE_PAIRS_SUMMARY = ANALYSIS_DIR / "source_pairs_summary.json"
PAIR_ENDPOINTS = ANALYSIS_DIR / "pair_endpoints.jsonl.gz"

CHECKPOINTS = {
    "L_lipophilicity": CHECKPOINT_DIR / "L_lipophilicity_Rf_best.pt",
    "G0_lipophilicity": CHECKPOINT_DIR / "G0_lipophilicity_Rf_best.pt",
    "L_WC_LOGP": CHECKPOINT_DIR / "L_WC_LOGP_best.pt",
    "T_WC_LOGP": CHECKPOINT_DIR / "T_WC_LOGP_best.pt",
}

SOURCE_CHECKPOINT_PATHS = {
    "L_lipophilicity": "results/stage1/ogbg-mollipo/L/Rf/checkpoint_best_val.pt",
    "G0_lipophilicity": "results/stage1/ogbg-mollipo/G0/Rf/checkpoint_best_val.pt",
    "L_WC_LOGP": "results/oracle/ogbg-mollipo/L/WC_LOGP/selected/checkpoint_best_val.pt",
    "T_WC_LOGP": "results/oracle/ogbg-mollipo/T/WC_LOGP/selected/checkpoint_best_val.pt",
}

#: Frozen source hashes copied from the source round's identity manifests.
EXPECTED_CACHE_SHA256 = {
    "ogbg_mollipo_sq-pilot-cache-v1.npz": "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c",
    "ogbg_mollipo_sq-pilot-lipo-raw-v1.npz": "3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3",
}
EXPECTED_ASSET_SHA256 = {
    "pair_catalog.jsonl.gz": "12376174e7d95a20a25c3f3ed5cca510850db1c71807b30cbd826005c21e11ba",
    "row_atom_mapping.jsonl.gz": "01e861f379dfdbdd83aaf4caee236eceb3bebd7e7d4568e191d55aa96043f64b",
    "ogbg-mollipo.npz": "5690c0e3c7ba0ed51fbe30541e4a83a49abf174c6440aaebfb46a3c5cbc48257",
}
EXPECTED_CHECKPOINT_SHA256 = {
    "L_lipophilicity": "9ef542e591963cf8ed012e95036d0f6f95e9a3c3d142a1838f28581bbbed10bc",
    "G0_lipophilicity": "04282be531bdc5883e31684e79720d3b42f5c998b5442921b56e61aecfb559ca",
    "L_WC_LOGP": "4522a6361909b1b7d1aee317bdbead0798e07eee0dd3143a396ade64ea788a67",
    "T_WC_LOGP": "5e812d7e45813bbb9100e3b3b98ac9f0d320573d7370abbade8d5fa8417fc165",
}

#: Source-round selected epochs / Val RMSE (for identity cross-checks only).
EXPECTED_CHECKPOINT_META = {
    "L_lipophilicity": {"stage": "stage1", "kind": "L", "recipe": "Rf", "epoch": 124, "valid_rmse_raw": 0.7068848410118609},
    "G0_lipophilicity": {"stage": "stage1", "kind": "G0", "recipe": "Rf", "epoch": 183, "valid_rmse_raw": 0.616077290013576},
    "L_WC_LOGP": {"stage": "oracle", "kind": "L", "property": "WC_LOGP", "epoch": 333, "valid_rmse_raw": 0.11440873206013732},
    "T_WC_LOGP": {"stage": "oracle", "kind": "T", "property": "WC_LOGP", "epoch": 355, "valid_rmse_raw": None},
}

#: Model/stratum names used throughout.
MODELS = ("LCRC", "GINE")
ORACLE_MODELS = ("LCRC_WC", "GINE_WC")
STRATA = ("train_train", "at_least_one_nontrain", "both_nontrain", "all")
STRATUM_LABEL = {
    "train_train": "Train–Train",
    "at_least_one_nontrain": "at least one endpoint non-Train",
    "both_nontrain": "both endpoints non-Train (H)",
    "all": "all evaluable pairs",
}

#: Plot / report colours.
COLORS = {
    "candidate": (0.12, 0.35, 0.85),
    "edit": (0.90, 0.45, 0.05),
    "oracle": (0.10, 0.60, 0.25),
    "combined_candidate": (0.12, 0.35, 0.85),
    "combined_edit": (0.90, 0.45, 0.05),
}

# ---------------------------------------------------------------------------
# import path discipline
# ---------------------------------------------------------------------------

_PATHS_READY = False


def install_source_paths() -> None:
    """Put the frozen read-only source code on sys.path (idempotent)."""
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (str(SQL_CODE), str(PREP_CODE), str(DISSECT_CODE), str(RC_CODE), str(ROUTE_CODE), str(PILOT_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


def source_modules() -> dict[str, Any]:
    install_source_paths()
    import gr_common  # noqa: F401
    import gr_data
    import gr_models
    import as_common
    import as_controls
    import as_data
    import as_gate
    import as_search
    import as_worker
    import cr_common
    import cr_data
    import cr_models

    return {
        "gr_common": gr_common,
        "gr_data": gr_data,
        "gr_models": gr_models,
        "as_common": as_common,
        "as_controls": as_controls,
        "as_data": as_data,
        "as_gate": as_gate,
        "as_search": as_search,
        "as_worker": as_worker,
        "cr_common": cr_common,
        "cr_data": cr_data,
        "cr_models": cr_models,
    }


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


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


def sha256_state(state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu() if hasattr(value, "detach") else torch.as_tensor(value)
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_jsonl_gz(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with gzip.GzipFile(tmp, "wb", mtime=0) as handle:
        for row in rows:
            handle.write((json.dumps(jsonable(row), sort_keys=True) + "\n").encode("utf-8"))
            n += 1
    tmp.replace(path)
    return {"path": str(path.relative_to(ROUND_DIR)), "sha256": sha256_file(path), "n": n}


def read_jsonl_gz(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    tmp.replace(path)


def append_jsonl(path: Path | str, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(pred - target)))


# ---------------------------------------------------------------------------
# environment / git
# ---------------------------------------------------------------------------


def _cmd(args: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_commit() -> str | None:
    return _cmd(["git", "rev-parse", "HEAD"])


def git_branch() -> str | None:
    return _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def git_dirty_tracked() -> bool | None:
    out = _cmd(["git", "status", "--porcelain", "--untracked-files=no"])
    return None if out is None else bool(out)


def environment() -> dict[str, Any]:
    import torch

    try:
        import rdkit

        rdkit_version = getattr(rdkit, "__version__", None)
    except Exception:
        rdkit_version = None
    try:
        import torch_geometric

        pyg_version = getattr(torch_geometric, "__version__", None)
    except Exception:
        pyg_version = None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "rdkit": rdkit_version,
        "torch_geometric": pyg_version,
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty_tracked": git_dirty_tracked(),
        "cwd": str(Path.cwd()),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def set_seeds(seed: int = SAMPLE_SEED) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# component bootstrap
# ---------------------------------------------------------------------------


def component_bootstrap(
    values: Mapping[str, np.ndarray],
    components: np.ndarray,
    stat_fn: Callable[[Mapping[str, np.ndarray]], Mapping[str, float]],
    *,
    mask: np.ndarray | None = None,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Shared-endpoint component bootstrap.

    ``values`` maps metric name -> array aligned with pair rows.  ``components``
    is the component id per pair.  Components (not pairs) are resampled with
    replacement; every pair of a selected component is included, so a component
    selected twice contributes its pairs twice, as in a cluster bootstrap.
    """
    components = np.asarray(components, dtype=np.int64)
    if mask is None:
        mask = np.ones(components.size, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    idx_all = np.flatnonzero(mask)
    comps = np.unique(components[idx_all])
    point = stat_fn({key: np.asarray(val, dtype=np.float64)[idx_all] for key, val in values.items()})
    out: dict[str, Any] = {
        "n_pairs": int(idx_all.size),
        "n_components": int(comps.size),
        "point": {k: float(v) for k, v in point.items()},
        "n_boot": int(n_boot),
        "seed": int(seed),
        "ci95": {},
        "boot": {},
    }
    if comps.size == 0 or n_boot <= 0:
        out["ci95"] = {k: [None, None] for k in point}
        out["boot"] = {k: [] for k in point}
        return out
    comp_to_idx = {int(c): np.flatnonzero(components == c) for c in comps.tolist()}
    rng = np.random.default_rng(int(seed))
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(int(n_boot)):
        chosen = rng.choice(comps, size=comps.size, replace=True)
        if chosen.size == 0:
            continue
        idx = np.concatenate([comp_to_idx[int(c)] for c in chosen.tolist()])
        stats = stat_fn({key: np.asarray(val, dtype=np.float64)[idx] for key, val in values.items()})
        for key, value in stats.items():
            draws[key].append(float(value))
    for key, vals in draws.items():
        arr = np.asarray(vals, dtype=np.float64)
        out["boot"][key] = arr.tolist()
        out["ci95"][key] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if arr.size else [None, None]
    return out


def mean_stat(mapping: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.mean(np.asarray(value, dtype=np.float64))) for key, value in mapping.items()}


def support_label(n_components: int, n_pairs: int) -> str:
    if int(n_pairs) == 0:
        return "NO_SUPPORT"
    if int(n_components) < LOW_SUPPORT_COMPONENTS:
        return "LOW_SUPPORT"
    return "OK"


def relpath(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
