"""Frozen constants + read-only upstream bridge for the readout closeout round.

Round: ``lcrc-readout-diagnosis-closeout-public122-s0-v1``

This is a *development / post-diagnosis follow-up*, not an independent
confirmation.  It recovers the originally planned frozen-representation source
(the LIP_C1_R1_F100 E1 R_SEL state at update 8165) and re-runs the three readout
probes on the correct source, adding direct branch-response measurements.

Everything scientific (source selection, probe specs, optimiser, grid, rules) is
written in ``PROTOCOL.md`` + ``configs/closeout.yaml`` before any new probe
prediction.  The upstream round
``lcrc_bottleneck_diagnosis_public122_s0_v1`` is **read-only**; we import its
``lb_common`` / ``lb_data`` / ``lb_train`` / ``lb_selection`` modules and hash
them into every record.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-readout-diagnosis-closeout-public122-s0-v1"
PROTOCOL_ID = ROUND_ID
STATUS_TAG = "DEVELOPMENT / POST_DIAGNOSIS_FOLLOWUP"

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

STRUCT_ROOT = REPO / "tracks/structural-sufficiency"
UPSTREAM_TRACK = STRUCT_ROOT / "lcrc_bottleneck_diagnosis_public122_s0_v1"
UPSTREAM_CODE = UPSTREAM_TRACK / "code"

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"
SYNTHESIS = ANALYSIS / "synthesis"

if str(UPSTREAM_CODE) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_CODE))

import lb_common as C  # noqa: E402  upstream frozen constants + hashing helpers
import lb_data as D  # noqa: E402
import lb_train as T  # noqa: E402
import lb_selection as SEL  # noqa: E402
import lcpr_common as PC  # noqa: E402
import lcpr_model as PM  # noqa: E402

#: upstream norm/cache files already materialised by the previous round; reused
#: verbatim (never refitted).
UPSTREAM_CACHE = UPSTREAM_TRACK / "cache"

DATASET = "Lipophilicity"
TASK = "regression"
PRIMARY_DATASET = DATASET

#: the recovered frozen source: E1 scope R_SEL state of LIP_C1_R1_F100
SOURCE_ARM = "LIP_C1_R1_F100"
SOURCE_CKPT_REL = f"tracks/structural-sufficiency/{UPSTREAM_TRACK.name}/runs/{SOURCE_ARM}/best_R_SEL.pt"
SOURCE_EXPECTED_UPDATE = 8165
U0_SNAPSHOT_REL = f"tracks/structural-sufficiency/{UPSTREAM_TRACK.name}/runs/{SOURCE_ARM}/u0_snapshot.pt"

#: reference numbers recorded in the upstream report/manifest for update 8165
REFERENCE = {
    "valid_rmse": 0.7226158853735515,
    "test_rmse": 0.5920667914129428,
    "grid_update": 8165,
    "tolerance_max_abs": 1e-5,
}

#: probe science (unchanged from the upstream protocol §10)
PROBE_ARMS = ("P_WARM", "P_POST", "P_PRE")
PROBE_TRAIN_UPDATES = 1800
PROBE_LR = 3.0e-4
PROBE_WEIGHT_DECAY = 1.0e-3
PROBE_BATCH = 128
PROBE_PSI_SEED = 1101
PROBE_NEW_PARAMS = 14705
GRID_POINTS = 50
WINDOW = 5
SELECTION_RULES = ("R_SEL", "A_SEL", "W_SEL")
TRAIN_SEED = 0

#: diagnostics
DIAG_UPDATES = (1, 2, 10, 100, 900, 1800)
DIAG_N_MOLECULES = 32

#: budgets (this round)
WALL_TARGET_HOURS = 1.0
WALL_HARD_HOURS = 1.5
GPU_SOFT_CAP_HOURS = 0.10
GPU_HARD_CAP_HOURS = 0.25

#: purchase threshold (same research-investment ruler as upstream, not significance)
PURCHASE_RELATIVE_GAIN = 0.01
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261114

SPLITS = ("train", "valid", "test")
EVAL_BATCH = 256  # smaller than upstream to keep local memory modest


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def state_hash(state: Mapping[str, Any]) -> str:
    return PC.DM.state_fingerprint({k: v.detach().cpu() for k, v in state.items()})


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover
        return ""


CODE_PATHS = ("code", "configs", "PROTOCOL.md")


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS],
                                      cwd=TRACK, text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def env_info() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def upstream_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(UPSTREAM_CODE.glob("*.py")):
        out[track_rel(p)] = sha256_file(p)
    for p in sorted(PC.CODE_DIR.glob("*.py")):
        out[track_rel(p)] = sha256_file(p)
    for p in sorted(C.DISSECT_CODE.glob("*.py")):
        out[track_rel(p)] = sha256_file(p)
    for p in sorted(C.MOLNET_CODE.glob("*.py")):
        out[track_rel(p)] = sha256_file(p)
    return out


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.rglob("*.py")):
        out[track_rel(p)] = sha256_file(p)
    for p in sorted(TESTS.glob("*.py")):
        out[track_rel(p)] = sha256_file(p)
    for p in sorted(CONFIGS.glob("*.yaml")):
        out[track_rel(p)] = sha256_file(p)
    if (TRACK / "PROTOCOL.md").exists():
        out["PROTOCOL.md"] = sha256_file(TRACK / "PROTOCOL.md")
    return out


# ---------------------------------------------------------------------------
# source / norms (reuse upstream artefacts verbatim, never refit)
# ---------------------------------------------------------------------------


def source_checkpoint() -> Path:
    return REPO / SOURCE_CKPT_REL


def structure_norm(dataset: str = DATASET) -> dict[str, np.ndarray]:
    """Read the upstream frozen structure norm verbatim; never refit."""
    path = UPSTREAM_CACHE / f"struct_norm_{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen structure norm {path}; refusing to refit")
    z = np.load(path, allow_pickle=False)
    return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}


def target_norm(dataset: str = DATASET) -> dict[str, Any]:
    """Read the upstream shared regression target norm verbatim; never refit."""
    path = UPSTREAM_CACHE / f"target_norm_{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen target norm {path}; refusing to refit")
    z = np.load(path, allow_pickle=False)
    return {"mean": float(z["mean"]), "scale": float(z["scale"]), "n": int(z["n"]),
            "source": str(z["source"])}


def norm_hashes(dataset: str = DATASET) -> dict[str, str]:
    out = {}
    for name in (f"struct_norm_{dataset}.npz", f"target_norm_{dataset}.npz"):
        p = UPSTREAM_CACHE / name
        out[name] = sha256_file(p) if p.exists() else "MISSING"
    return out


def fit_positions(dataset: str = DATASET) -> dict[str, Any]:
    """The full official-Train nested subset ("100"); label-blind, reused verbatim."""
    return D.nested_subset_positions(dataset, "100")


def targets(dataset: str = DATASET):
    return D.targets(dataset, include_test=True)


def build_base_model(ckpt_state: Mapping[str, Any], spec, snorm):
    model = PM.build_model(spec, snorm, "c1", variant="C1")
    model.load_state_dict({k: v for k, v in ckpt_state.items()})
    return model


def denorm(pred_std: np.ndarray, tnorm: Mapping[str, Any]) -> np.ndarray:
    """Undo the standardisation exactly once."""
    return np.asarray(pred_std, np.float64) * float(tnorm["scale"]) + float(tnorm["mean"])


def metrics(pred_std: np.ndarray, y_raw: np.ndarray, tnorm: Mapping[str, Any] | None) -> dict[str, float]:
    return T.split_metrics(TASK, np.asarray(pred_std, np.float64), np.asarray(y_raw, np.float64), tnorm)


# ---------------------------------------------------------------------------
# identity-keyed clean-h cache (NOT the old dataset-only ``hcache_{dataset}``)
# ---------------------------------------------------------------------------


def feature_spec_hash(spec) -> str:
    return str(spec.hash())


def hcache_identity(*, ckpt_state_hash: str, spec_hash: str, norms: Mapping[str, str],
                    rows_hashes: Mapping[str, str]) -> dict[str, Any]:
    ident = {
        "dataset": DATASET,
        "source_state_hash": ckpt_state_hash,
        "feature_spec_hash": spec_hash,
        "struct_norm_hash": norms[f"struct_norm_{DATASET}.npz"],
        "target_norm_hash": norms[f"target_norm_{DATASET}.npz"],
        "split_row_hashes": dict(rows_hashes),
        "hcache_schema": "lcrc-readout-closeout-hcache-v1",
    }
    ident["identity_sha256"] = sha256_text(stable_json(ident))
    return ident


def hcache_path(identity: Mapping[str, Any]) -> Path:
    return CACHE / f"hclean_{DATASET}_{identity['identity_sha256'][:16]}.npz"


__all__ = [n for n in dir() if not n.startswith("_")]
