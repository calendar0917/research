"""Shared constants, hashing, metric and bootstrap helpers.

CPU-only.  Nothing here imports torch, CUDA, OGB or any training/scoring entry
point.  All predictions are the frozen ``test_predictions.npz`` files saved by
the previous round.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[4]
ROUND = REPO / "tracks/structural-sufficiency/lcrc_posthoc_diagnosis_v1"
PTB = REPO / "tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1"
EDIT = REPO / "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1"
MOLNET_V2 = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2"
OGB_SOURCE = MOLNET_V2 / "analysis/source/ogb/ogbg_molhiv"
TUD_NCI1 = REPO / "data/TUD/NCI1/NCI1"
LOCAL = ROUND / "_local/predictions"
ANALYSIS = ROUND / "analysis"
CODE = ROUND / "code"

BASELINE_COMMIT = "6cf9ca4eb3ddd45701c73baea66e80642034b756"
ROUND_ID = "lcrc-posthoc-diagnosis-v1"
RESULT_LABEL = "POST_HOC_EXPOSED_EVALUATION"

BOOTSTRAP_SEED = 20261103
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_MAX_ATTEMPTS = 10000

MOLECULE_SEEDS = (0, 1, 2)
TU_SEEDS = (0, 1)
MODELS = ("A0", "C1")

DATASETS: dict[str, dict[str, Any]] = {
    "ESOL": {"task": "regression", "primary": "rmse", "direction": "lower"},
    "FreeSolv": {"task": "regression", "primary": "rmse", "direction": "lower"},
    "BACE": {"task": "binary", "primary": "roc_auc", "direction": "higher"},
    "BBBP": {"task": "binary", "primary": "roc_auc", "direction": "higher"},
    "Tox21": {"task": "multitask", "primary": "macro_roc_auc", "direction": "higher"},
    "ogbg-molhiv": {"task": "binary", "primary": "roc_auc", "direction": "higher"},
}
MOLECULE_ORDER = ("ESOL", "FreeSolv", "BACE", "BBBP", "Tox21", "ogbg-molhiv")
TU_ORDER = ("NCI1", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI")


# ---------------------------------------------------------------------------
# IO / hashing
# ---------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def file_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": rel(path),
        "role": role,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def environment() -> dict[str, Any]:
    import rdkit

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "rdkit": rdkit.__version__,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }


# ---------------------------------------------------------------------------
# prediction files
# ---------------------------------------------------------------------------


def mol_run_dir(dataset: str, model: str, seed: int) -> Path:
    return LOCAL / f"mol_{dataset}_s{seed}_{model}"


def tu_run_dir(dataset: str, fold: int, seed: int, model: str) -> Path:
    return LOCAL / f"tu_{dataset}_f{fold}_s{seed}_{model}"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_mol(dataset: str, model: str, seed: int) -> dict[str, np.ndarray]:
    return load_npz(mol_run_dir(dataset, model, seed) / "test_predictions.npz")


def load_mol_eval(dataset: str, model: str, seed: int) -> dict[str, Any]:
    return read_json(mol_run_dir(dataset, model, seed) / "test_eval.json")


def load_mol_run(dataset: str, model: str, seed: int) -> dict[str, Any]:
    return read_json(mol_run_dir(dataset, model, seed) / "run.json")


def target_norm(dataset: str) -> dict[str, float]:
    """Train-only target normaliser saved by the previous round."""
    payload = load_mol_run(dataset, "A0", 0)["data"]["target_norm"]
    return {"mean": float(payload["mean"]), "scale": float(payload["scale"])}


def denormalise(raw: np.ndarray, dataset: str) -> np.ndarray:
    meta = DATASETS[dataset]
    arr = np.asarray(raw, dtype=np.float64)
    if meta["task"] != "regression":
        return arr
    tn = target_norm(dataset)
    return arr * tn["scale"] + tn["mean"]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def row_scores(pred: np.ndarray) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(f"expected [N] or [N,1], got {arr.shape}")


def binary_auc(score: np.ndarray, y: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    ss = np.asarray(score, dtype=np.float64).reshape(-1)
    return float(roc_auc_score(yy, ss))


def binary_pr_auc(score: np.ndarray, y: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    ss = np.asarray(score, dtype=np.float64).reshape(-1)
    return float(average_precision_score(yy, ss))


def regression_rmse(pred_raw: np.ndarray, y: np.ndarray, dataset: str) -> float:
    p = denormalise(pred_raw, dataset).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((p - yy) ** 2)))


def regression_mae(pred_raw: np.ndarray, y: np.ndarray, dataset: str) -> float:
    p = denormalise(pred_raw, dataset).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(p - yy)))


def multitask_auc(pred: np.ndarray, y: np.ndarray, task_mask: Sequence[int]) -> float:
    p = np.asarray(pred, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    vals = []
    for j in task_mask:
        obs = np.isfinite(yy[:, j])
        yj = yy[obs, j]
        if np.unique(yj).size != 2:
            raise ValueError(f"task {j}: resample does not contain both classes")
        vals.append(float(roc_auc_score(yj, p[obs, j])))
    if not vals:
        raise ValueError("no definable tasks")
    return float(np.mean(vals))


def original_task_mask(y: np.ndarray) -> list[int]:
    yy = np.asarray(y, dtype=np.float64)
    out: list[int] = []
    for j in range(yy.shape[1]):
        obs = np.isfinite(yy[:, j])
        if obs.sum() and np.unique(yy[obs, j]).size == 2:
            out.append(j)
    return out


def primary_metric(
    dataset: str,
    pred: np.ndarray,
    y: np.ndarray,
    mask: Sequence[int] | None = None,
    *,
    denormalised: bool = False,
) -> float:
    meta = DATASETS[dataset]
    if meta["task"] == "regression":
        p = np.asarray(pred, dtype=np.float64) if denormalised else denormalise(pred, dataset)
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        return float(np.sqrt(np.mean((p.reshape(-1) - yy) ** 2)))
    if meta["task"] == "binary":
        return binary_auc(pred, y)
    if mask is None:
        mask = original_task_mask(y)
    return multitask_auc(pred, y, mask)


def seed_primary(dataset: str, model: str, seed: int, *, mask: Sequence[int] | None = None) -> float:
    z = load_mol(dataset, model, seed)
    return primary_metric(dataset, z["pred"], z["y_true"], mask)


# ---------------------------------------------------------------------------
# bootstrap indices
# ---------------------------------------------------------------------------


def bootstrap_rows(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n, dtype=np.int64)


def bootstrap_stratified(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yy = np.asarray(y).reshape(-1)
    pos = np.flatnonzero(yy == 1)
    neg = np.flatnonzero(yy == 0)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("stratified bootstrap needs both classes")
    take_pos = rng.choice(pos, size=pos.size, replace=True)
    take_neg = rng.choice(neg, size=neg.size, replace=True)
    return np.concatenate([take_pos, take_neg])


def bootstrap_components(components: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    uniq = np.asarray(sorted(set(components)), dtype=object)
    return rng.choice(uniq, size=uniq.size, replace=True)


# ---------------------------------------------------------------------------
# component-equal weighting
# ---------------------------------------------------------------------------


def component_equal_mae(errors: np.ndarray, components: Sequence[str]) -> float:
    """Each component has total weight 1; pairs inside it split that weight."""
    err = np.abs(np.asarray(errors, dtype=np.float64).reshape(-1))
    comp = np.asarray([str(c) for c in components], dtype=object)
    if err.size != comp.size:
        raise ValueError("error/component length mismatch")
    total = 0.0
    n_components = 0
    for c in sorted(set(comp.tolist())):
        vals = err[comp == c]
        total += float(vals.mean())
        n_components += 1
    if n_components == 0:
        raise ValueError("no components")
    return total / n_components


def component_equal_rmse(errors: np.ndarray, components: Sequence[str]) -> float:
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    comp = np.asarray([str(c) for c in components], dtype=object)
    total = 0.0
    n_components = 0
    for c in sorted(set(comp.tolist())):
        vals = err[comp == c]
        total += float(np.sqrt(np.mean(vals * vals)))
        n_components += 1
    return total / n_components


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def quantile_ci(values: Sequence[float], lo: float = 2.5, hi: float = 97.5) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"lo": None, "hi": None, "n_finite": 0}
    return {
        "lo": float(np.percentile(arr, lo)),
        "hi": float(np.percentile(arr, hi)),
        "n_finite": int(arr.size),
    }


def mean_sd(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def sign(x: float, tol: float = 0.0) -> int:
    if x > tol:
        return 1
    if x < -tol:
        return -1
    return 0


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(columns) if columns is not None else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})
