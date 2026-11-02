"""Shared, read-only access to the completed ``lcrc_lipo_model_dissection_v1`` round.

The follow-up adds no training and never writes into the original round.  All
paths below point at the completed round's artefacts; this module only reads
them and creates files under the follow-up directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

FOLLOWUP = Path(__file__).resolve().parents[1]
TRACK = FOLLOWUP.parent
REPO = TRACK.parents[1]
ORIG = TRACK / "lcrc_lipo_model_dissection_v1"
ORIG_CODE = ORIG / "code"
ORIG_ANALYSIS = ORIG / "analysis"
ORIG_RUNS = ORIG / "runs"
ANALYSIS = FOLLOWUP / "analysis"
CODE = FOLLOWUP / "code"
TABLES = ANALYSIS / "corrected_tables"
ERRORS = ANALYSIS / "error_concentration"
TRAINED_JC = ANALYSIS / "trained_jc"

# Import the frozen round code exactly as it was executed.  The original files
# stay untouched; only their public functions are reused.
for _p in (str(ORIG_CODE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dissect_common as C  # noqa: E402
import dissect_data as D  # noqa: E402
import dissect_model as M  # noqa: E402
import dissect_source as S  # noqa: E402

MAIN_RUNS = ORIG_ANALYSIS / "tables" / "main_runs.csv"
MAIN_REPEAT = int(C.MAIN_REPEAT)
SEEDS = tuple(int(s) for s in C.SEEDS)
REPEATS = tuple(int(r) for r in C.REPEATS)
N_VALID = 420

ERROR_VARIANTS = ("A0", "A3", "A5", "C1", "B5")
INTERVENTION_VARIANTS = ("A0", "C1")


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_tensors(tensors: Mapping[str, Any]) -> str:
    import torch

    h = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous().numpy()
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value, dtype=np.float32).tobytes())
    return h.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})
    return p


def _json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_status_short() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def env_info() -> dict[str, Any]:
    import rdkit
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "rdkit": rdkit.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": "cpu",
        "torch_num_threads": torch.get_num_threads(),
    }


# ---------------------------------------------------------------------------
# per-run records and predictions
# ---------------------------------------------------------------------------


def run_dir(repeat: int, variant: str, seed: int) -> Path:
    return ORIG_RUNS / C.run_name(repeat, variant, seed)


def load_train_record(repeat: int, variant: str, seed: int) -> dict[str, Any]:
    path = run_dir(repeat, variant, seed) / "train.json"
    rec = read_json(path)
    if rec.get("status") != "completed":
        raise ValueError(f"{rel(path)} is not a completed run")
    if (int(rec["repeat"]), str(rec["variant"]), int(rec["seed"])) != (int(repeat), variant, int(seed)):
        raise ValueError(f"{rel(path)} identity mismatch")
    return rec


def load_valid_predictions(repeat: int, variant: str, seed: int) -> dict[str, np.ndarray]:
    path = run_dir(repeat, variant, seed) / "valid_predictions.npz"
    with np.load(path) as z:
        keys = set(z.files)
        required = {"rows", "raw_standardised", "raw", "y_true"}
        if keys != required:
            raise ValueError(f"{rel(path)} keys {sorted(keys)} != {sorted(required)}")
        return {key: np.asarray(z[key]) for key in z.files}


def load_main_rows() -> list[dict[str, Any]]:
    """Read the already-reported scalar table.  Never reads Test predictions."""
    rows: list[dict[str, Any]] = []
    with MAIN_RUNS.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = dict(raw)
            row["repeat"] = int(raw["repeat"])
            row["seed"] = int(raw["seed"])
            for key in (
                "params_nominal",
                "params_trainable",
                "params_expected",
                "optimizer_steps",
                "epochs_run",
                "best_epoch",
                "valid_n",
            ):
                row[key] = int(float(raw[key]))
            for key in (
                "train_rmse",
                "train_mae",
                "train_r2",
                "valid_rmse",
                "valid_mae",
                "valid_r2",
                "valid_bias",
                "test_rmse",
                "test_mae",
                "test_r2",
                "gpu_hours",
            ):
                row[key] = None if raw.get(key) in (None, "", "None") else float(raw[key])
            rows.append(row)
    return rows


def check_main_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities = [(int(r["repeat"]), str(r["variant"]), int(r["seed"])) for r in rows]
    duplicates = sorted({x for x in identities if identities.count(x) > 1})
    if duplicates:
        raise ValueError(f"duplicate (repeat, variant, seed): {duplicates}")
    expected = {(r, v, s) for r in REPEATS for v in (C.VARIANT_ORDER if r == MAIN_REPEAT else ("A0", "C1", "B5")) for s in SEEDS}
    found = set(identities)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if missing:
        raise ValueError(f"missing run identities: {missing}")
    if unexpected:
        raise ValueError(f"unexpected run identities: {unexpected}")
    per_repeat: dict[str, int] = {}
    for repeat, _, _ in identities:
        per_repeat[str(repeat)] = per_repeat.get(str(repeat), 0) + 1
    return {
        "n_rows": len(rows),
        "n_unique_identities": len(set(identities)),
        "per_repeat": per_repeat,
        "duplicates": duplicates,
        "missing": missing,
        "unexpected": unexpected,
        "n_valid_is_420": all(int(r["valid_n"]) == N_VALID for r in rows),
        "variants_by_repeat": {
            str(r): sorted({str(x["variant"]) for x in rows if int(x["repeat"]) == r}) for r in REPEATS
        },
    }


def valid_reference() -> dict[str, Any]:
    view = D.build_view(MAIN_REPEAT)
    rows = np.asarray(view.rows["valid"], dtype=np.int64)
    if rows.size != N_VALID:
        raise ValueError(f"Valid has {rows.size} rows, expected {N_VALID}")
    if not np.array_equal(rows, np.asarray(view.valid_cache["rows"], dtype=np.int64)):
        raise ValueError("Valid cache rows differ from the published split")
    return {
        "view": view,
        "rows": rows,
        "y": np.asarray(view.y_valid, dtype=np.float64),
        "smiles": [str(s) for s in view.valid_cache["smiles"]],
    }


def align_predictions(payload: Mapping[str, np.ndarray], ref: Mapping[str, Any]) -> dict[str, Any]:
    """Align by published Valid row_id (never by array position or SMILES)."""
    rows = np.asarray(payload["rows"], dtype=np.int64)
    ref_rows = np.asarray(ref["rows"], dtype=np.int64)
    if rows.size != ref_rows.size or set(rows.tolist()) != set(ref_rows.tolist()):
        return {"ok": False, "reason": "row_id set differs", "n_pred": int(rows.size), "n_ref": int(ref_rows.size)}
    if len(set(rows.tolist())) != rows.size:
        return {"ok": False, "reason": "duplicate row_id in predictions"}
    pos = {int(r): i for i, r in enumerate(rows.tolist())}
    order = np.asarray([pos[int(r)] for r in ref_rows.tolist()], dtype=np.int64)
    raw = np.asarray(payload["raw"], dtype=np.float64)[order]
    y_true = np.asarray(payload["y_true"], dtype=np.float64)[order]
    y_ref = np.asarray(ref["y"], dtype=np.float64)
    y_diff = float(np.max(np.abs(y_true - y_ref)))
    if y_diff > 0.0:
        return {"ok": False, "reason": f"y_true differs (max abs {y_diff})", "y_max_abs_diff": y_diff}
    return {
        "ok": True,
        "raw": raw,
        "y": y_ref,
        "rows": ref_rows,
        "order": order,
        "y_max_abs_diff": y_diff,
        "raw_standardised": np.asarray(payload["raw_standardised"], dtype=np.float64)[order],
    }


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {
        "n": int(err.size),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "sse": float(np.sum(err * err)),
    }


# ---------------------------------------------------------------------------
# input manifest helpers
# ---------------------------------------------------------------------------


def hash_entry(path: Path | str, role: str, *, required: bool = True, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p = Path(path)
    item: dict[str, Any] = {"role": role, "path": rel(p), "exists": p.exists()}
    if p.exists():
        item["size_bytes"] = p.stat().st_size
        item["sha256"] = sha256_file(p)
    elif required:
        item["missing"] = True
    if extra:
        item.update(dict(extra))
    return item


__all__ = [
    "FOLLOWUP",
    "TRACK",
    "REPO",
    "ORIG",
    "ORIG_CODE",
    "ORIG_ANALYSIS",
    "ORIG_RUNS",
    "ANALYSIS",
    "CODE",
    "TABLES",
    "ERRORS",
    "TRAINED_JC",
    "C",
    "D",
    "M",
    "S",
    "MAIN_RUNS",
    "MAIN_REPEAT",
    "SEEDS",
    "REPEATS",
    "N_VALID",
    "ERROR_VARIANTS",
    "INTERVENTION_VARIANTS",
    "sha256_file",
    "sha256_array",
    "sha256_tensors",
    "read_json",
    "write_json",
    "write_csv",
    "utcnow",
    "git_commit",
    "git_status_short",
    "rel",
    "env_info",
    "run_dir",
    "load_train_record",
    "load_valid_predictions",
    "load_main_rows",
    "check_main_rows",
    "valid_reference",
    "align_predictions",
    "metrics",
    "hash_entry",
]
