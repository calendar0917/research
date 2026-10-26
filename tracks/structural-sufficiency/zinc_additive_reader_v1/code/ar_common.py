"""Shared plumbing for the additive-reader follow-up.

The round reuses the frozen Fit8k / Monitor / Locked rows of
``bjoc_predictive_bottleneck_v1`` and the historical DCC-L run artifacts.  No
official valid or official test cache is ever opened here.

Only the following row roles exist:

* Fit8k  -> supervised training and fit-only standardisation,
* Monitor -> per-epoch evaluation and best-checkpoint selection,
* Locked -> one frozen read after all checkpoints and hashes are frozen.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SUFF = ROUND.parent
REPO = SUFF.parents[1]

V1 = SUFF / "zinc_arch_discovery_v1" / "code"
DCC = SUFF / "zinc_dual_channel_composer_v1" / "code"
BJOC = REPO / "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1"
CONFIG = ROUND / "configs" / "ar_v1.json"
RUNS = ROUND / "runs"
RUNS_SMOKE = ROUND / "runs_smoke"
ANALYSIS = ROUND / "analysis"

TRAIN_CACHE = SUFF / "zinc_arch_discovery_v1" / "cache" / "ad_train.pt"
DCCL_RUNS = SUFF / "zinc_dccl_bottleneck_v1" / "runs"

ROUND_ID = "zinc-additive-reader-v1"
OFFICIAL_TEST_LOADED = False
OFFICIAL_VALID_LOADED = False


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray, dtype="<i8") -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a), dtype=dtype).tobytes()).hexdigest()


def state_hash(model: torch.nn.Module, prefix: tuple[str, ...] | None = None) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if prefix is not None and not key.startswith(prefix):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False, default=_json_default)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config() -> dict:
    return read_json(CONFIG)


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def split_rows() -> dict:
    """Load the frozen split and verify its declared row hashes and roles."""
    s = read_json(BJOC / "split_a.json")
    f4 = read_json(BJOC / "fit4k_subset.json")
    cfg = load_config()
    rows = {
        "fit8k": np.asarray(s["fit_rows"], dtype=np.int64),
        "fit4k": np.asarray(f4["rows"], dtype=np.int64),
        "monitor": np.asarray(s["monitor_rows"], dtype=np.int64),
        "locked": np.asarray(s["locked_rows"], dtype=np.int64),
    }
    recorded = {
        "fit8k": s["fit8k_sha256_rows"],
        "monitor": s["monitor_sha256_rows"],
        "locked": s["locked_sha256_rows"],
    }
    for name, digest in recorded.items():
        got = sha256_array(rows[name], "<i8")
        if got != digest:
            raise RuntimeError(f"{name} row hash mismatch: {got} != {digest}")
        expected = cfg["split"]["expected_sha256_rows"][name]
        if got != expected:
            raise RuntimeError(f"{name} row hash differs from config: {got} != {expected}")
    expected_n = cfg["split"]
    if len(rows["fit8k"]) != expected_n["fit_rows"]:
        raise RuntimeError("fit8k row count mismatch")
    if len(rows["monitor"]) != expected_n["monitor_rows"]:
        raise RuntimeError("monitor row count mismatch")
    if len(rows["locked"]) != expected_n["locked_rows"]:
        raise RuntimeError("locked row count mismatch")
    if sorted(np.concatenate([rows["fit8k"], rows["monitor"], rows["locked"]]).tolist()) != list(range(10000)):
        raise RuntimeError("Fit8k / Monitor / Locked are not a disjoint partition of the 10k train cache")
    if not set(rows["fit4k"].tolist()) <= set(rows["fit8k"].tolist()):
        raise RuntimeError("Fit4k is not nested in Fit8k")
    return {
        **rows,
        "hashes": recorded,
        "partition_ok": True,
        "roles": cfg["split"]["role"],
        "locked_labels_read": bool(s.get("locked_labels_read", False)),
    }


# ---------------------------------------------------------------------------
# cache and standardisation
# ---------------------------------------------------------------------------
def load_train_cache() -> dict:
    """Load the official-train cache; refuse anything else by construction."""
    if TRAIN_CACHE.name != "ad_train.pt":
        raise RuntimeError("train cache path must be ad_train.pt")
    if not TRAIN_CACHE.exists():
        raise FileNotFoundError(f"missing train cache {TRAIN_CACHE}")
    return torch.load(TRAIN_CACHE, map_location="cpu", weights_only=False)


def fit_only_norm(train: Mapping[str, Any], rows: np.ndarray) -> dict:
    """Standardisation from the declared Fit rows only (copied from dcb_train)."""
    g = np.asarray(train["global_feat"], dtype=np.float64)[rows]
    gmean = g.mean(0).astype(np.float32)
    gscale = np.maximum(g.std(0).astype(np.float32), 1e-6)

    ap = np.asarray(train["atom_ptr"])
    atom_rows = np.concatenate([np.arange(ap[i], ap[i + 1]) for i in rows])
    rs = np.asarray(train["root_scalar"], dtype=np.float64)[atom_rows]
    rmean = rs.mean(0).astype(np.float32)
    rscale = np.maximum(rs.std(0).astype(np.float32), 1e-6)

    bp = np.asarray(train["block_ptr"])
    block_rows = np.concatenate([np.arange(bp[i], bp[i + 1]) for i in rows])
    bs = np.asarray(train["block_scalar"], dtype=np.float64)[block_rows]
    bmean = bs.mean(0).astype(np.float32)
    bscale = np.maximum(bs.std(0).astype(np.float32), 1e-6)

    return {
        "source": f"computed_from_fit_rows_n={len(rows)}",
        "global_mean": gmean, "global_scale": gscale,
        "root_scalar_mean": rmean, "root_scalar_scale": rscale,
        "block_scalar_mean": bmean, "block_scalar_scale": bscale,
        "fit_split": "bjoc-internal-fit",
        "valid_or_test_used": False,
        "n_fit_rows": int(len(rows)),
    }


def norm_arrays(norm: Mapping[str, Any]) -> dict:
    return {k: np.asarray(v) for k, v in norm.items() if isinstance(v, np.ndarray)}


def norm_hash(norm: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for k, v in norm_arrays(norm).items():
        h.update(k.encode())
        h.update(np.ascontiguousarray(v, dtype=np.float32).tobytes())
    return h.hexdigest()


def cosine_with_warmup_lr(step: int, total_steps: int, warmup_steps: int,
                          base_lr: float, min_lr: float) -> float:
    import math

    step = int(step)
    if warmup_steps > 0 and step <= warmup_steps:
        return float(base_lr) * float(step) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return float(base_lr)
    frac = float(step - warmup_steps) / float(total_steps - warmup_steps)
    frac = min(max(frac, 0.0), 1.0)
    return float(min_lr) + 0.5 * (float(base_lr) - float(min_lr)) * (1.0 + math.cos(math.pi * frac))


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def atomic_torch_save(obj: Any, path: Path | str) -> None:
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
