"""Shared plumbing for BBPA-v0 (bridge-block port algebra pilot).

Nothing scientific lives here: paths, IO, hashing, metrics, seeding and
label-safe identities.  Every scientific choice is frozen in
``configs/bbpa.json`` and ``PROTOCOL.md`` before formal training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
STRUCT = REPO_ROOT / "tracks" / "structural-sufficiency"

#: frozen parent artefacts (read-only, label-safe members only)
RAW_CACHE = REPO_ROOT / "tracks" / "structural-query-learning" / "data" / "ogbg_mollipo_sq-pilot-cache-v1.npz"
RAW_CACHE_SHA256 = "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c"
SPLIT_F0 = STRUCT / "acra_relation_generalization_pilot_v1" / "data" / "split_train_sub_dev.json"
SPLIT_F0_SHA256_EXPECTED = "60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e"
TRAIN_LABELS = STRUCT / "structural_path_composition_route_v1" / "data" / "labels_official_train.npz"
TRAIN_LABELS_ROW_SHA256_EXPECTED = "a44d29deba1fd36e"  # prefix recorded by PSC

CONFIG_PATH = ROUND_DIR / "configs" / "bbpa.json"
RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"

#: OGB raw schema
ATOM_COLUMNS = 9
BOND_COLUMNS = 3

#: label-free raw members read from the frozen structure cache
RAW_STRUCT_KEYS = (
    "dataset", "cache_version", "node_ptr", "bond_ptr", "atom_feat", "bond_feat",
    "bond_u", "bond_v", "degree", "min_ring", "in_cycle",
    "graph_n_components", "graph_cycle_rank", "graph_n_bonds", "graph_n_atoms",
    "smiles", "row_ids", "train_idx", "valid_idx",
)


# ---------------------------------------------------------------------------
# IO and identity
# ---------------------------------------------------------------------------


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
    """Content hash of every parameter in each named module (eval-independent)."""
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


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]],
              columns: Sequence[str] | None = None) -> Path:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    if columns is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        columns = ordered
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return target


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def rmse(pred: Sequence[float], target: Sequence[float]) -> float:
    p = np.asarray(pred, np.float64)
    t = np.asarray(target, np.float64)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(pred: Sequence[float], target: Sequence[float]) -> float:
    p = np.asarray(pred, np.float64)
    t = np.asarray(target, np.float64)
    return float(np.mean(np.abs(p - t)))


def paired_bootstrap(pred_a: Sequence[float], pred_b: Sequence[float],
                     target: Sequence[float], n_boot: int = 5000,
                     seed: int = 0) -> dict[str, Any]:
    """Descriptive paired bootstrap for RMSE(a)-RMSE(b) and MAE(a)-MAE(b).

    Positive RMSE delta means model *a* is worse than model *b*.
    """
    a = np.asarray(pred_a, np.float64)
    b = np.asarray(pred_b, np.float64)
    y = np.asarray(target, np.float64)
    n = int(y.size)
    rng = np.random.default_rng(int(seed))
    rmse_d = np.empty(int(n_boot), np.float64)
    mae_d = np.empty(int(n_boot), np.float64)
    ea = (a - y) ** 2
    eb = (b - y) ** 2
    aa = np.abs(a - y)
    ab = np.abs(b - y)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        rmse_d[i] = math.sqrt(ea[idx].mean()) - math.sqrt(eb[idx].mean())
        mae_d[i] = aa[idx].mean() - ab[idx].mean()
    out = {
        "n_molecules": n,
        "n_bootstrap": int(n_boot),
        "rmse_delta_point": float(rmse(a, y) - rmse(b, y)),
        "rmse_delta_ci95": [float(np.quantile(rmse_d, 0.025)), float(np.quantile(rmse_d, 0.975))],
        "mae_delta_point": float(mae(a, y) - mae(b, y)),
        "mae_delta_ci95": [float(np.quantile(mae_d, 0.025)), float(np.quantile(mae_d, 0.975))],
        "rmse_delta_boot_mean": float(rmse_d.mean()),
        "rmse_delta_boot_sd": float(rmse_d.std(ddof=1)),
        "mae_delta_boot_mean": float(mae_d.mean()),
        "mae_delta_boot_sd": float(mae_d.std(ddof=1)),
        "positive_means_a_worse_than_b": True,
    }
    return out


def effective_rank(features: np.ndarray) -> dict[str, float]:
    """Participation ratio and entropy rank of a centered feature matrix."""
    x = np.asarray(features, np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return {"n": int(x.shape[0]) if x.ndim == 2 else 0, "participation_ratio": 0.0,
                "entropy_rank": 0.0, "top1_share": 0.0}
    x = x - x.mean(axis=0, keepdims=True)
    cov = x.T @ x / max(1, x.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if total <= 0:
        return {"n": int(x.shape[0]), "participation_ratio": 0.0,
                "entropy_rank": 0.0, "top1_share": 0.0}
    p = eig / total
    p = p[p > 0]
    pr = float((eig.sum() ** 2) / np.sum(eig ** 2))
    ent = float(np.exp(-np.sum(p * np.log(p))))
    return {"n": int(x.shape[0]), "participation_ratio": pr, "entropy_rank": ent,
            "top1_share": float(p.max())}


def _unused_guard() -> None:  # pragma: no cover
    if os.environ.get("BBPA_STRICT") == "1":
        raise RuntimeError("strict guard")