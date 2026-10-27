"""Shared plumbing for SOC-v1 (splits, norms, builders, metrics, provenance)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
ROUND = _HERE.parent
SUFF = ROUND.parent
REPO = SUFF.parents[1]
_V1 = SUFF / "zinc_arch_discovery_v1" / "code"
_DCC = SUFF / "zinc_dual_channel_composer_v1" / "code"
_DCB = SUFF / "zinc_dccl_bottleneck_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DCC), str(_DCB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dcb_train as DB  # noqa: E402
import dcc_runner as DR  # noqa: E402
import soc_models as SM  # noqa: E402

ROUND_ID = "zinc-statistical-operator-closure-v1"
CFG_PATH = ROUND / "configs" / "soc_v1.json"
ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
BJOC = REPO / "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1"
DCCL = SUFF / "zinc_dual_channel_composer_v1"

RECIPE = dict(DB.RECIPE)
TOTAL_UPDATES = int(DB.TOTAL_UPDATES)  # 47,400
OFFICIAL_VALID_ANCHOR = 0.075990  # DCC-L 4-seed mean, historical
OFFICIAL_TEST_ANCHOR = 0.069400  # DCC-L 4-seed mean, historical
FIT8K_MONITOR_ANCHOR = {0: 0.123151, 1: 0.127772}  # frozen L fit8k checkpoints
FIT8K_MONITOR_MEAN = float(np.mean(list(FIT8K_MONITOR_ANCHOR.values())))


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_rows(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default) + "\n",
                    encoding="utf-8")


def json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def load_config() -> dict:
    return json.loads(CFG_PATH.read_text())


def split_rows() -> dict:
    rows = DB.split_rows()
    rows["all"] = np.arange(10000, dtype=np.int64)
    return rows


def fit_norm(train_cache: Mapping[str, Any], rows: np.ndarray) -> dict:
    return DB.fit_only_norm(train_cache, rows)


def make_builder(arm: str, cfg: Mapping[str, Any]):
    """Return a deterministic builder for SOC arms and the matched control."""
    K = int(cfg.get("K", 3))
    readout = "mean" if arm == "T" else "final"
    rank = int(cfg.get("rank", 24))
    hidden = int(cfg.get("hidden", 48))
    ctrl_hidden = int(cfg.get("control_reader_hidden", 274))

    def builder(candidate: str, norm, seed: int, device):
        if arm in ("R", "T"):
            return SM.build_model(candidate, norm, seed, device, K=K,
                                  readout=readout, rank=rank, hidden=hidden)
        if arm == "CTRL":
            model, info = SM.build_control_state(norm, seed, device,
                                                 reader_hidden=ctrl_hidden)
            info = dict(info)
            info["candidate"] = candidate
            return model, info
        raise ValueError(f"unknown arm {arm!r}")

    return builder


def dcc_anchor_builder(candidate: str, norm, seed: int, device):
    """Frozen DCC-L builder, used only to reproduce anchor predictions."""
    model, info = SM.DM.build_initial_state(norm, seed)
    model.to(device)
    info = dict(info)
    info["candidate"] = candidate
    return model, info


def evaluate(model, cache, rows, *, device, batch_size: int = 256):
    return T1.evaluate(model, cache, np.asarray(rows, dtype=np.int64), candidate="P3N",
                       device=device, batch_size=batch_size,
                       collate_fn=DR.collate_p3n)


def monitor_report(pred: np.ndarray, y: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ae = np.abs(pred - y)
    return {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "bias": float(np.mean(pred - y)),
        "p50_ae": float(np.quantile(ae, 0.50)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "p99_ae": float(np.quantile(ae, 0.99)),
    }


def gpu_hours(seconds: float) -> float:
    return float(seconds) / 3600.0


def environment() -> dict:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
