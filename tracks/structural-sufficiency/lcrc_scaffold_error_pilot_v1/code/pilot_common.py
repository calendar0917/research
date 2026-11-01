"""Shared constants and provenance helpers for ``lcrc_scaffold_error_pilot_v1``.

This round reuses the frozen training engine, model definitions and data cache
of ``lcrc_lipo_best_epoch_distance_fix_v1`` **read-only**.  The new files in
this directory only add:

* a new Bemis--Murcko scaffold partition with a fixed split seed;
* per-molecule error attribution on the *old* scaffold checkpoints;
* six new training runs (BASE / G2L-ALL x seeds 0,1,2) on the new partition.

No historical directory is written.  The old training engine is imported and
its round identity is overridden in memory so that new checkpoints carry the
new protocol id.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

ROUND_ID = "lcrc-scaffold-error-pilot-v1"
PROTOCOL_ID = ROUND_ID
SPLIT_SEED = 20261102
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)

OLD_ROUND = "lcrc_lipo_best_epoch_distance_fix_v1"
OLD_EXP = REPO / "tracks/structural-sufficiency" / OLD_ROUND
OLD_CODE = OLD_EXP / "code"

for _path in (str(CODE_DIR), str(OLD_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import be_common as OLD  # noqa: E402  (read-only reused engine/common)

ANALYSIS = EXP_DIR / "analysis"
P0_DIR = ANALYSIS / "p0"
P0_FIGURES = P0_DIR / "figures"
SPLIT_DIR = ANALYSIS / "split"
SPLIT_MANIFEST = SPLIT_DIR / "split_pilot_v1.json"
SPLIT_AUDIT = SPLIT_DIR / "split_audit.json"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
RUNS = ANALYSIS / "runs"
EVAL = ANALYSIS / "eval"
MANIFEST = ANALYSIS / "manifest.json"
MANIFEST_FROZEN = ANALYSIS / "MANIFEST_FROZEN"
RR_STATUS = ANALYSIS / "rr_status"
SMOKE = ANALYSIS / "smoke"

CONFIG_PATH = EXP_DIR / "configs/pilot.yaml"
P0_FINDINGS = EXP_DIR / "P0_FINDINGS.md"
REPORT = EXP_DIR / "REPORT.md"

CSV_PATH = OLD.CSV_PATH
CSV_SHA256 = OLD.CSV_SHA256
N_MOLECULES = OLD.N_MOLECULES
MODEL_DISPLAY = {"base": "BASE", "roleblind": "ROLEBLIND", "g2l_all": "G2L-ALL"}
PILOT_MODELS = ("base", "g2l_all")
PILOT_SEEDS = (0, 1, 2)
PILOT_ARMS = tuple(f"pilot_{family}_s{seed}" for family in PILOT_MODELS for seed in PILOT_SEEDS)

# Old scaffold stage: all nine checkpoints are read-only inputs for P0.
OLD_SCAFFOLD_MODELS = ("base", "roleblind", "g2l_all")
OLD_SCAFFOLD_ARMS = tuple(f"scaffold_{family}_s{seed}" for family in OLD_SCAFFOLD_MODELS for seed in PILOT_SEEDS)


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(encoded)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log(message: str) -> None:
    print(message, flush=True)


def utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def track_rel(path: Path | str) -> str:
    return str(Path(path).resolve().relative_to(REPO))


def rows_sha256(rows) -> str:
    return OLD.rows_sha256_int64([int(x) for x in rows])


def vector_sha256_float64(values) -> str:
    return OLD.vector_sha256_float64(np.asarray(values, dtype=np.float64))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True
        )
        return bool(out.strip())
    except Exception:
        return True


def code_hashes() -> dict[str, str]:
    """New pilot code plus the read-only training engine used to run it."""
    out = {
        f"pilot/{path.name}": sha256_file(path)
        for path in sorted(CODE_DIR.glob("*.py"))
    }
    for path in sorted(OLD_CODE.glob("*.py")):
        out[f"reused/{OLD_ROUND}/{path.name}"] = sha256_file(path)
    return out


def patch_reused_engine() -> None:
    """Make the reused training engine stamp this round's identity.

    Only provenance fields are changed in memory.  Recipe, model definitions,
    data cache and split-independent behaviour are untouched.
    """
    OLD.PROTOCOL_ID = PROTOCOL_ID
    OLD.ROUND_ID = ROUND_ID
    OLD.CONFIG_PATH = CONFIG_PATH
    OLD.code_file_hashes = code_hashes


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if str(cfg.get("protocol_id")) != PROTOCOL_ID:
        raise ValueError("pilot config protocol_id drift")
    if int(cfg.get("split_seed", -1)) != SPLIT_SEED:
        raise ValueError("pilot config split_seed drift")
    training = dict(cfg["training"])
    for key, expected in OLD.TRAIN_RECIPE.items():
        if key not in training:
            raise ValueError(f"pilot config missing training.{key}")
        if training[key] != expected:
            raise ValueError(f"pilot config training.{key} drift: {training[key]!r} != {expected!r}")
    return cfg


def config_sha256() -> str:
    return sha256_file(CONFIG_PATH)


def env_info() -> dict[str, Any]:
    import torch

    return {
        "round_id": ROUND_ID,
        "timestamp": utcnow(),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "numpy": np.__version__,
    }


def require_freeze() -> dict[str, Any]:
    if not PROTOCOL_FREEZE.exists():
        raise PermissionError("pilot protocol freeze missing; run `run_pilot.py freeze` first")
    frozen = read_json(PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("pilot protocol freeze has the wrong protocol id")
    if frozen.get("config_sha256") != config_sha256():
        raise PermissionError("pilot config changed after the freeze")
    if frozen.get("code_hashes") != code_hashes():
        raise PermissionError("pilot code changed after the freeze")
    if frozen.get("split_manifest_sha256") != sha256_file(SPLIT_MANIFEST):
        raise PermissionError("pilot split manifest changed after the freeze")
    if frozen.get("split_audit_sha256") != sha256_file(SPLIT_AUDIT):
        raise PermissionError("pilot split audit changed after the freeze")
    return frozen


def require_split() -> dict[str, Any]:
    if not SPLIT_MANIFEST.exists():
        raise PermissionError("pilot split manifest missing; run `run_pilot.py split` first")
    if not SPLIT_AUDIT.exists():
        raise PermissionError("pilot split audit missing; run `run_pilot.py audit-split` first")
    audit = read_json(SPLIT_AUDIT)
    if not audit.get("passed"):
        raise PermissionError(f"pilot split audit failed: {audit}")
    return read_json(SPLIT_MANIFEST)


def freeze_protocol() -> dict[str, Any]:
    require_split()
    cfg = load_config()
    payload = {
        "protocol_id": PROTOCOL_ID,
        "round_id": ROUND_ID,
        "created_utc": utcnow(),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "status": "FROZEN_BEFORE_PILOT_TRAINING",
        "scope": "one new Bemis-Murcko scaffold partition of the 4,200 Lipophilicity molecules",
        "split_seed": SPLIT_SEED,
        "split_fractions": list(SPLIT_FRACTIONS),
        "split_manifest": track_rel(SPLIT_MANIFEST),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "split_audit": track_rel(SPLIT_AUDIT),
        "split_audit_sha256": sha256_file(SPLIT_AUDIT),
        "config": track_rel(CONFIG_PATH),
        "config_sha256": config_sha256(),
        "code_hashes": code_hashes(),
        "training_recipe": dict(cfg["training"]),
        "models": list(PILOT_MODELS),
        "seeds": list(PILOT_SEEDS),
        "arms": list(PILOT_ARMS),
        "checkpoint_rule": "single_best_inner_validation_rmse_epoch",
        "outer_evaluation_policy": (
            "freeze all six checkpoint hashes before reading any outer-evaluation label; "
            "outer labels are used exactly once, after the manifest freeze"
        ),
        "outer_labels_read_at_freeze_time": False,
        "gpu_hard_cap_hours": 2.0,
    }
    payload["freeze_sha256"] = sha256_json({k: v for k, v in payload.items() if k != "freeze_sha256"})
    write_json(PROTOCOL_FREEZE, payload)
    return payload


__all__ = [name for name in dir() if not name.startswith("_")]
