"""Shared identity, paths, IO, metrics and statistics for
``structural-response-route-clarification-8h-v0``.

No scientific quantity is defined here; every frozen definition lives in
``PROTOCOL.md`` and ``configs/experiment.json``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "structural-response-route-clarification-8h-v0"
CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

CONFIG_PATH = ROUND_DIR / "configs" / "experiment.json"
ANALYSIS_DIR = ROUND_DIR / "analysis"
RESULTS_DIR = ROUND_DIR / "results"
PRED_DIR = ROUND_DIR / "predictions"
CURVE_DIR = ROUND_DIR / "curves"
CKPT_DIR = ROUND_DIR / "checkpoints"

HIST_SS = REPO_ROOT / "tracks" / "structural-sufficiency"
SQ_DIR = REPO_ROOT / "tracks" / "structural-query-learning"

#: Read-only reuse roots.
FGCR_DIR = HIST_SS / "full_graph_cavity_response_pilot_v0"
FGCR_CODE = FGCR_DIR / "code"
FGCR_CONFIG = FGCR_DIR / "configs" / "fgcr.json"
FGCR_CKPT = FGCR_DIR / "results" / "seed0" / "best.pt"
FGCR_CKPT_SHA = "?"  # filled by replay script at run time
AUDIT_DIR = HIST_SS / "fgcr_joint_response_increment_audit_v0"
RD_SMALL_DIR = HIST_SS / "response_dictionary_small_s0_v1"
RD_SMALL_CODE = RD_SMALL_DIR / "code"
RD_FOLLOW_DIR = HIST_SS / "response_dictionary_lipo_followup_s0_v1"
RD_FOLLOW_CODE = RD_FOLLOW_DIR / "code"
TT_DIR = HIST_SS / "response_dictionary_task_transfer_s0_v1"
GR_DIR = HIST_SS / "gnn_route_clarification_8h_v1"
GR_CODE = GR_DIR / "code"
D0_DIR = HIST_SS / "acra_relation_generalization_pilot_v1"

DATA_DIR = SQ_DIR / "data"
FRAG_DIR = RD_SMALL_DIR / "data"

EXPECTED_SHA = {
    "lipo_cache": "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c",
    "lipo_frag_cache": "ba6e9a41990fc344e5962519caf02e616c9c67d299884843c8b67a5bbab0a1d8",
    "esol_frag_cache": "fac644d0d7b73f8bc06239853c97fe65cd3010ab02ea539f9a6bdf4f4852a62d",
    "d0_split": "675289f6f997008d44fdabb4ab65350459fe5bccc5e1e5703e31e31fd794e845",
    "catalogue_pairs": "64186ac5397e7cfca635f7aa620a5fe3fd60df65e750a2bbae0ccfb14ec14bcc",
}

DATASET_CACHE = {
    "ogbg-mollipo": DATA_DIR / "ogbg_mollipo_sq-pilot-cache-v1.npz",
    "ogbg-molesol": DATA_DIR / "ogbg_molesol_sq-pilot-cache-v1.npz",
}
DATASET_FRAG = {
    "ogbg-mollipo": FRAG_DIR / "ogbg_mollipo_rd_frag_v1.npz",
    "ogbg-molesol": FRAG_DIR / "ogbg_molesol_rd_frag_v1.npz",
}
DATASET_SPLIT_SIZES = {
    "ogbg-mollipo": {"train": 3360, "valid": 420, "test": 420},
    "ogbg-molesol": {"train": 902, "valid": 113, "test": 113},
}

CATALOGUE = TT_DIR / "analysis" / "structure" / "catalogue_pairs.csv"
D0_SPLIT = D0_DIR / "data" / "split_train_sub_dev.json"

LAMBDA = 0.5
PAIR_SEED = 23017
BOOT_SEED = 1703
N_BOOT = 2000
TRAIN_SEED = 0

ATOM_COLUMNS = 9
BOND_COLUMNS = 3

#: Frozen recipes (copied from the parent rounds; see PROTOCOL.md section 4/6).
FGCR_RECIPE = dict(
    optimizer="AdamW", peak_lr=3e-4, min_lr=1e-5, weight_decay=1e-4,
    batch_molecules=64, eval_batch_molecules=256, warmup_epochs=5,
    grad_clip=5.0, dropout=0.1, epochs=200, schedule="fgcr_cosine",
)
RESP_RECIPE = dict(
    optimizer="AdamW", peak_lr=1e-3, min_lr=1e-5, weight_decay=1e-5,
    betas=(0.9, 0.999), eps=1e-8, batch_molecules=128,
    eval_batch_molecules=256, warmup_epochs=5, grad_clip=5.0,
    dropout=0.10, epochs=360, schedule="rd_linear_warmup_then_cosine",
)
GINE_RECIPE = dict(
    optimizer="AdamW", peak_lr=1e-3, min_lr=1e-5, weight_decay=1e-5,
    betas=(0.9, 0.999), eps=1e-8, batch_molecules=128,
    eval_batch_molecules=256, warmup_epochs=5, grad_clip=5.0,
    dropout=0.10, epochs=360, schedule="gr_linear_warmup_then_cosine",
)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                             capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# JSON / CSV / config
# ---------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    return read_json(CONFIG_PATH)


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]],
              columns: Sequence[str] | None = None) -> Path:
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


# ---------------------------------------------------------------------------
# seeding / metrics / environment
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(d)))


def target_scaler(labels: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    return {"mean": float(y.mean()), "std": float(y.std(ddof=0)), "n": int(y.size),
            "min": float(y.min()), "max": float(y.max())}


def environment_snapshot() -> dict[str, Any]:
    import torch

    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "commit": git_commit(),
        "dirty": git_dirty(),
    }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def paired_component_bootstrap(unit_a: np.ndarray, unit_b: np.ndarray,
                               component: np.ndarray, *, n_boot: int = N_BOOT,
                               seed: int = BOOT_SEED, alpha: float = 0.05
                               ) -> dict[str, Any]:
    """Paired bootstrap over **components** of a per-pair statistic.

    ``unit_a``/``unit_b`` are per-pair point estimates of the same functional
    (lower is better); ``component`` gives each pair's component id.  Each
    resample draws components with replacement, then averages pair values inside
    each drawn component and weights components equally.

    Returns the CI of ``mean(b) - mean(a)`` and of ``rmse(b) - rmse(a)``.
    """
    unit_a = np.asarray(unit_a, dtype=np.float64)
    unit_b = np.asarray(unit_b, dtype=np.float64)
    component = np.asarray(component, dtype=np.int64)
    uniq = np.unique(component)
    n_comp = int(uniq.size)
    if n_comp < 2:
        return {"n_components": n_comp, "n_boot": 0, "note": "fewer than two components"}
    index = {int(c): i for i, c in enumerate(uniq.tolist())}
    comp_of_pair = np.asarray([index[int(c)] for c in component.tolist()], dtype=np.int64)
    sum_a = np.bincount(comp_of_pair, weights=unit_a, minlength=n_comp)
    sum_b = np.bincount(comp_of_pair, weights=unit_b, minlength=n_comp)
    cnt = np.bincount(comp_of_pair, minlength=n_comp).astype(np.float64)

    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n_comp, size=(int(n_boot), n_comp))
    w = np.where(cnt[draws] > 0, 1.0 / np.maximum(cnt[draws], 1.0), 0.0)
    mean_a = (sum_a[draws] * w).sum(axis=1) / n_comp
    mean_b = (sum_b[draws] * w).sum(axis=1) / n_comp
    rmse_a = np.sqrt((sum_a[draws] * w).sum(axis=1) / n_comp)
    rmse_b = np.sqrt((sum_b[draws] * w).sum(axis=1) / n_comp)
    d_mean = mean_b - mean_a
    d_rmse = rmse_b - rmse_a
    lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
    return {
        "n_components": n_comp,
        "n_pairs": int(unit_a.size),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "delta_mean": float(d_mean.mean()),
        "delta_mean_ci_low": float(np.quantile(d_mean, lo)),
        "delta_mean_ci_high": float(np.quantile(d_mean, hi)),
        "delta_rmse": float(d_rmse.mean()),
        "delta_rmse_ci_low": float(np.quantile(d_rmse, lo)),
        "delta_rmse_ci_high": float(np.quantile(d_rmse, hi)),
        "frac_resamples_b_better_mean": float((d_mean < 0).mean()),
    }


def component_equal(values: np.ndarray, component: np.ndarray) -> tuple[float, int]:
    """Equal-weight mean over components of the within-component pair mean."""
    values = np.asarray(values, dtype=np.float64)
    component = np.asarray(component, dtype=np.int64)
    if values.size == 0:
        return float("nan"), 0
    uniq, inv = np.unique(component, return_inverse=True)
    if uniq.size == 0:
        return float("nan"), 0
    sums = np.bincount(inv, weights=values, minlength=uniq.size)
    cnt = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    per = sums / np.maximum(cnt, 1.0)
    return float(per.mean()), int(uniq.size)


def drop_worst_component(per_pair_a: np.ndarray, per_pair_b: np.ndarray,
                         component: np.ndarray) -> dict[str, Any]:
    """Drop the single component with the largest *average* error of both models."""
    per_pair_a = np.asarray(per_pair_a, dtype=np.float64)
    per_pair_b = np.asarray(per_pair_b, dtype=np.float64)
    component = np.asarray(component, dtype=np.int64)
    uniq, inv = np.unique(component, return_inverse=True)
    cnt = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    mean_a = np.bincount(inv, weights=per_pair_a, minlength=uniq.size) / np.maximum(cnt, 1.0)
    mean_b = np.bincount(inv, weights=per_pair_b, minlength=uniq.size) / np.maximum(cnt, 1.0)
    key = 0.5 * (mean_a + mean_b)
    worst = int(np.argmax(key))
    keep = inv != worst
    n_kept = int(np.unique(inv[keep]).size)
    ca, _ = component_equal(per_pair_a[keep], component[keep])
    cb, _ = component_equal(per_pair_b[keep], component[keep])
    return {
        "dropped_component": int(uniq[worst]),
        "n_components_kept": n_kept,
        "ce_a_kept": ca,
        "ce_b_kept": cb,
        "ce_delta_kept": cb - ca,
    }


# ---------------------------------------------------------------------------
# resource ledger
# ---------------------------------------------------------------------------


def ledger_path() -> Path:
    return ROUND_DIR / "resource_ledger.json"


def ledger_append(entry: Mapping[str, Any]) -> None:
    path = ledger_path()
    payload = read_json(path) if path.is_file() else {"round_id": ROUND_ID, "entries": []}
    payload.setdefault("entries", []).append(dict(jsonable(entry)))
    total = 0.0
    for item in payload["entries"]:
        total += float(item.get("gpu_hours", 0.0) or 0.0)
    payload["gpu_hours_total"] = total
    write_json(path, payload)
