"""FSF follow-up v1 common helpers.

Read-only inheritance of the FSR v0 (RSSC) implementation.  Nothing in the v0
round directory is modified: this module only imports its code and reads its
artefacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

# Never create __pycache__ inside the historical round.
sys.dont_write_bytecode = True

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
REPO = ROUND.parents[2]
SS = REPO / "tracks/structural-sufficiency"

SRC_ROUND = SS / "fixed_source_role_separation_pilot_v0"
SRC_CODE = SRC_ROUND / "code"
SRC_RUNS = SRC_ROUND / "runs" / "RSSC_seed0"
SRC_ANALYSIS = SRC_ROUND / "analysis"

for _p in (str(SRC_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import frs_common as FC  # noqa: E402  (v0 round constants + historical FIXED1 paths)
import frs_train as FT  # noqa: E402  (v0 training primitives; unchanged computation)
import frs_model as FM  # noqa: E402
import frs_mechanism as FSM  # noqa: E402  (only the light D/W + scale helpers)
import ds_common as C  # noqa: E402
import ds_model as DSM  # noqa: E402
import pc_metrics as MET  # noqa: E402

ROUND_ID = "fixed-source-role-separation-followup-v1"
PROTOCOL_ID = "fsf-v1"
SCHEMA_RUN = "fsf-run-v1"
SCHEMA_CHECKS = "fsf-resume-checks-v1"
SCHEMA_TIMING = "fsf-timing-v1"
SCHEMA_MANIFEST = "fsf-checkpoint-manifest-v1"

SEED = 0
SOURCE_ARM = "RSSC"
RESUME_EPOCH = 12
TARGET_HORIZON = 25
TRAIN_EVAL_EPOCHS = (16, 20, 25)
BATCH_SIZE = int(C.BATCH_SIZE)
STEPS_PER_EPOCH = int(C.STEPS_PER_EPOCH)
TOTAL_STEPS_40 = int(C.TOTAL_STEPS_40)
RESUME_GLOBAL_STEP = RESUME_EPOCH * STEPS_PER_EPOCH

RUN_DIR = ROUND / "runs" / "RSSC_resume_seed0"
ANALYSIS = ROUND / "analysis"
RESULTS = ROUND / "results"

SOURCE_LAST = SRC_RUNS / "checkpoint_last.pt"
SOURCE_BEST = SRC_RUNS / "checkpoint_best.pt"
SOURCE_RUN_JSON = SRC_RUNS / "run.json"
SOURCE_CURVE = SRC_RUNS / "train_curve.csv"
SOURCE_PER_TASK = SRC_RUNS / "per_task_valid.csv"
SOURCE_MECHANISM = SRC_RUNS / "mechanism.json"
SOURCE_USAGE = SRC_RUNS / "structural_usage.json"
SOURCE_REPORTS = SRC_RUNS / "best_epoch_reports.json"
SOURCE_MANIFEST = SRC_ROUND / "results" / "manifest.json"
SOURCE_CHECKS = SRC_ANALYSIS / "checks.json"

HIST_RUN_DIR = FC.HIST["curve"].parent
HIST_RUN_JSON = FC.HIST["run_json"]
HIST_CURVE = FC.HIST["curve"]
HIST_PER_TASK = FC.HIST["per_task"]
HIST_ORDER = HIST_RUN_DIR / "training_order.json"

#: The files that define data/model/loss/training computation.  Their git blobs
#: must be identical to the blob at the checkpoint's commit.
COMPUTE_FILES = (
    "tracks/structural-sufficiency/fixed_source_role_separation_pilot_v0/code/frs_model.py",
    "tracks/structural-sufficiency/fixed_source_role_separation_pilot_v0/code/frs_train.py",
    "tracks/structural-sufficiency/direct_structural_computation_recurrence_v0/code/ds_model.py",
    "tracks/structural-sufficiency/direct_structural_computation_recurrence_v0/code/ds_common.py",
    "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1/code/pc_model.py",
    "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1/code/pc_metrics.py",
    "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/code/sg_data.py",
    "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/code/bs_model.py",
    "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/code/bs_common.py",
)


def sha256_file(path: Path) -> str:
    return FC.sha256_file(Path(path))


def read_json(path: Path) -> Any:
    return FC.read_json(Path(path))


def write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    FC.write_json(Path(path), payload)


def utcnow() -> str:
    return C.utcnow()


def log(msg: str) -> None:
    C.log(msg)


def track_rel(path: Path) -> str:
    return C.track_rel(Path(path))


def load_source_manifest() -> dict[str, Any]:
    return read_json(SOURCE_MANIFEST)


def source_checkpoint_record(kind: str) -> dict[str, Any]:
    manifest = load_source_manifest()
    return manifest["arms"]["RSSC"]["checkpoints"][kind]


def verify_source_checkpoint(path: Path, kind: str) -> dict[str, Any]:
    rec = source_checkpoint_record(kind)
    got = sha256_file(path)
    if got != rec["sha256"]:
        raise RuntimeError(f"source checkpoint {kind} sha256 {got} != manifest {rec['sha256']}")
    if int(rec["bytes"]) != int(Path(path).stat().st_size):
        raise RuntimeError(f"source checkpoint {kind} size mismatch")
    return {"path": track_rel(Path(path)), "sha256": got, "manifest_sha256": rec["sha256"],
            "bytes": int(rec["bytes"]), "epoch": int(rec["epoch"]),
            "global_step": int(rec["global_step"]), "valid_ap": rec.get("valid_ap"),
            "parameters": rec.get("parameters"), "test_scored": rec.get("test_scored")}


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def state_fingerprint(state: Mapping[str, Any]) -> str:
    return C.state_fingerprint(dict(state))


def restore_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore Python/NumPy/torch CPU and visible-CUDA generators exactly."""
    import random

    import numpy as np

    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = list(state.get("torch_cuda") or [])
    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if cuda_states and visible != len(cuda_states):
        raise RuntimeError(f"checkpoint has {len(cuda_states)} CUDA RNG states but {visible} devices are visible")
    for i, s in enumerate(cuda_states):
        torch.cuda.set_rng_state(s, device=i)
    return C.rng_state_hashes(C.capture_rng_state())


def optimizer_state_equal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Exact comparison of two serialised AdamW state dicts (CPU tensors)."""
    ls, rs = left["state"], right["state"]
    if set(ls) != set(rs):
        return {"equal": False, "reason": "state keys differ"}
    max_diff = 0.0
    for pid in ls:
        if set(ls[pid]) != set(rs[pid]):
            return {"equal": False, "reason": f"tensor keys differ for param {pid}"}
        for key in ls[pid]:
            a, b = ls[pid][key], rs[pid][key]
            if torch.is_tensor(a) or torch.is_tensor(b):
                if not (torch.is_tensor(a) and torch.is_tensor(b)):
                    return {"equal": False, "reason": f"{pid}.{key} tensor vs non-tensor"}
                aa = a.detach().cpu()
                bb = b.detach().cpu()
                if tuple(aa.shape) != tuple(bb.shape) or aa.dtype != bb.dtype:
                    return {"equal": False, "reason": f"{pid}.{key} shape/dtype differs"}
                if aa.numel():
                    max_diff = max(max_diff, float((aa.float() - bb.float()).abs().max().item()))
                if not torch.equal(aa, bb):
                    return {"equal": False, "reason": f"{pid}.{key} not bitwise equal",
                            "max_abs_diff": max_diff}
            elif a != b:
                return {"equal": False, "reason": f"{pid}.{key} differs"}
    return {"equal": True, "max_abs_diff": max_diff, "n_state_tensors": len(ls)}


def git_blob_sha256(commit: str, rel: str) -> str:
    blob = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{commit}:{rel}"], stderr=subprocess.PIPE
    )
    return hashlib.sha256(blob).hexdigest()


def compute_files_unchanged(commit: str) -> dict[str, Any]:
    rows = {}
    for rel in COMPUTE_FILES:
        path = REPO / rel
        if not path.is_file():
            rows[rel] = {"present": False}
            continue
        now = sha256_file(path)
        old = git_blob_sha256(commit, rel)
        rows[rel] = {"present": True, "current_sha256": now, "checkpoint_commit_sha256": old,
                     "equal": bool(now == old)}
    return {"checkpoint_commit": commit, "all_equal": all(v.get("equal") for v in rows.values()),
            "files": rows}


def source_curve(horizon: int | None = None) -> list[dict[str, Any]]:
    import csv

    with open(SOURCE_CURVE, newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    out = []
    for r in rows:
        if horizon is not None and int(r["epoch"]) > int(horizon):
            continue
        out.append({"source": "RSSC_v0_epochs_1_12", **r})
    return out


def source_per_task(epoch: int) -> dict[int, dict[str, Any]]:
    import csv

    out: dict[int, dict[str, Any]] = {}
    with open(SOURCE_PER_TASK, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "valid" or int(r["epoch"]) != int(epoch):
                continue
            out[int(r["task"])] = {
                "ap": (None if r["ap"] == "" else float(r["ap"])),
                "observed": int(r["observed"]),
                "positive": int(r["positive"]),
                "negative": int(r["negative"]),
            }
    return out


def historical_plan_detail() -> dict[str, Any]:
    return read_json(HIST_ORDER)["plan_summary"]["epochs_detail"]


def historical_lr(epoch: int) -> float:
    import csv

    with open(HIST_CURVE, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["epoch"]) == int(epoch):
                return float(r["learning_rate"])
    raise KeyError(epoch)


def copy_source_best_to_run_dir() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    dest = RUN_DIR / "checkpoint_best.pt"
    src = SOURCE_BEST
    if not dest.is_file():
        shutil.copy2(src, dest)
    src_sha = sha256_file(src)
    dst_sha = sha256_file(dest)
    if src_sha != dst_sha:
        raise RuntimeError(f"copied best checkpoint sha {dst_sha} != source {src_sha}")
    return {"path": track_rel(dest), "sha256": dst_sha, "source_path": track_rel(src),
            "source_sha256": src_sha, "epoch": RESUME_EPOCH,
            "note": "byte-identical copy of the v0 epoch-12 best checkpoint"}
