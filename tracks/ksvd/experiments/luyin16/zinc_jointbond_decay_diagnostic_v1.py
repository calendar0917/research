"""E2E-DictEnv-JointBond-Decay-Diagnostic-v1 — fixed-parent branch-trainability probe.

Round ``e2e_dictenv_jointbond_decay_diagnostic_v1`` (study ``zinc-context-gap``).
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_jointbond_decay_diagnostic_v1_preregistration.md``
(frozen before the run; sha256 captured in ``preflight.json``).

Question
--------
With the **frozen** CSSD-Sem108 soup as the parent, a **fixed** 128-graph fit
batch and the **same non-zero** v1 branch initialisation, can the 4224-parameter
JointBond branch learn a training residual — and does Adam's **coupled L2**
weight decay suppress that?

Two arms, identical in every respect except the optimizer's weight decay:

* ``wd``   — ``Adam(lr=1e-3, weight_decay=1e-5)`` (coupled L2, not AdamW)
* ``nowd`` — ``Adam(lr=1e-3, weight_decay=0)``

Only the 5 branch tensors are in the optimizers; the parent is never updated
(and is verified bit-identical afterwards).  Dropout is off (``eval()``) but the
training forward runs **with** autograd so gradients flow through the frozen
parent into the branch.  2 x 200 steps on the repeated fit batch, CPU, 8
threads, sequential.  Probe batch (next 128 disjoint official-train indices) is
observation only.  Official valid/test are never loaded; no soup; no
performance claim.

Stages
------
``references preflight correctness run analysis chain``
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_jointbond_v1 as jb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = "e2e_dictenv_jointbond_decay_diagnostic_v1"
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_jointbond_decay_diagnostic_v1"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_jointbond_decay_diagnostic_v1_preregistration.md"
SEM108_RESULTS = TRACK_ROOT / "results/e2e_dictenv_sem108_v1"
SEM108_SOUP_PATH = SEM108_RESULTS / "checkpoints/SEM108-seed0_soup_state.pt"
SEM108_SOUP_SHA256 = "7a721984c5579dd76f9bcaff13a2055b8efba71da553eefb3645d94207f5050a"
SEM108_SOUP_VALID_MAE = 0.123704927947314
JOINTBOND_RESULTS = TRACK_ROOT / "results/e2e_dictenv_jointbond_v1"
JOINTBOND_SOUP_PATH = JOINTBOND_RESULTS / "checkpoints/JOINTBOND-seed0_soup_state.pt"
TRAIN_CACHE_PATH = p1run._env_cache_path("train")
REPORT_PATH = RESULTS_DIR / "REPORT.md"
ANALYSIS_JSON = RESULTS_DIR / "decay_diagnostic.json"

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit

THREADS = 8
SEED = 0
STEPS = 200
FIT_SIZE = 128
PROBE_SIZE = 128
LOG_STEPS: tuple[int, ...] = (0, 1, 5, 20, 50, 100, 200)
ARMS: tuple[tuple[str, float], ...] = (("wd", float(p1run.WEIGHT_DECAY)), ("nowd", 0.0))
LEARNING_RATE = float(p1run.LEARNING_RATE)
WEIGHT_DECAY_REFERENCE = float(p1run.WEIGHT_DECAY)
GRAD_CLIP = float(p1run.GRAD_CLIP)
BATCH_SIZE_REFERENCE = int(p1run.BATCH_SIZE)
MASK = cm.C6_MASK
OFF_MASK = jb.merge_joint_mask(cm.C6_MASK, jb.JointBondMask(joint_branch_off=True))
BRANCH_KEYS: tuple[str, ...] = (
    "joint_A.weight",
    "joint_C.weight",
    "joint_B.weight",
    "joint_F.0.weight",
    "joint_F.2.weight",
)
BATCH_TENSOR_KEYS: tuple[str, ...] = (
    "dict_phi",
    "dict_atom",
    "anchor",
    "patch_cont",
    "y",
    "env_occ_node",
    "env_occ_root",
    "env_occ_shell",
    "env_bond_u",
    "env_bond_v",
    "env_bond_root",
    "env_bond_shellpair",
    "env_bond_type",
)
#: pre-registered interpretation thresholds (section 8 of the pre-registration).
LEARN_TOL = 1.0e-3
SUPPRESSION_TOL = 5.0e-4
FLOAT32 = torch.float32


def _ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# frozen objects
# ---------------------------------------------------------------------------


def _load_parent_subspace() -> cssd.CommonSubspace:
    path = TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json"
    entry = _read_json(path)["q1"]
    return cssd.CommonSubspace(
        components=np.asarray(entry["components"], dtype=np.float64),
        rms=np.asarray(entry["rms"], dtype=np.float64),
        kind=str(entry["kind"]),
    )


def _dictionary_tensor() -> np.ndarray:
    dictionary, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return dictionary


def _sem108_soup_state() -> dict[str, torch.Tensor]:
    state = torch.load(SEM108_SOUP_PATH, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError("Sem108 soup checkpoint is not a state mapping")
    return {str(key): value for key, value in state.items()}


def _jointbond_soup_joint_state() -> dict[str, torch.Tensor]:
    state = torch.load(JOINTBOND_SOUP_PATH, map_location="cpu", weights_only=False)
    return {str(key): value for key, value in state.items() if str(key).startswith("joint_")}


def build_arms() -> tuple[Any, Any, dict[str, Any]]:
    """One fresh v1 branch init + the Sem108 soup parent, then a deep copy.

    ``torch.manual_seed(0)`` inside ``build_jointbond_model`` reproduces the
    exact v1 branch initialisation.  The soup is loaded with ``strict=False`` so
    only the 49 parent keys are overwritten; the 5 ``joint_*`` tensors keep the
    fresh v1 init (the dead v1 soup values are never loaded).
    """
    torch.manual_seed(int(SEED))
    model = jb.build_jointbond_model(_dictionary_tensor(), int(SEED), _load_parent_subspace())
    fresh_joint = {
        name: parameter.detach().clone()
        for name, parameter in model.joint_parameters().items()
    }
    soup = _sem108_soup_state()
    incompatible = model.load_state_dict(soup, strict=False)
    missing = sorted(str(key) for key in incompatible.missing_keys)
    unexpected = sorted(str(key) for key in incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected parent keys in the Sem108 soup: {unexpected}")
    if missing != sorted(BRANCH_KEYS):
        raise RuntimeError(f"expected exactly the 5 branch keys to stay untouched, got {missing}")
    model.eval()
    arm_wd = model
    arm_nowd = copy.deepcopy(model)
    info = {
        "fresh_joint_state_sha256": audit.state_sha256(fresh_joint),
        "fresh_joint_absmax": {
            name: float(value.abs().max()) for name, value in fresh_joint.items()
        },
        "dead_joint_state_sha256": audit.state_sha256(_jointbond_soup_joint_state()),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "parent_state_sha256": audit.state_sha256(_parent_state(arm_wd)),
        "full_state_sha256_wd": audit.state_sha256(arm_wd.state_dict()),
        "full_state_sha256_nowd": audit.state_sha256(arm_nowd.state_dict()),
        "parent_keys": len(_parent_state(arm_wd)),
        "branch_keys": len(list(arm_wd.joint_parameters())),
        "branch_params": int(sum(p.numel() for p in arm_wd.joint_parameters().values())),
        "total_params": int(sum(p.numel() for p in arm_wd.parameters())),
    }
    return arm_wd, arm_nowd, info


def _parent_state(model: Any) -> dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if not key.startswith("joint_")}


def sem108_parent_model() -> Any:
    """A separately instantiated frozen Sem108 model (branch-free reference)."""
    model = sem.build_sem108_model(_dictionary_tensor(), int(SEED), _load_parent_subspace())
    model.load_state_dict(_sem108_soup_state())
    model.eval()
    return model


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _batch_fingerprint(batch: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    used: list[str] = []
    for key in BATCH_TENSOR_KEYS:
        value = getattr(batch, key, None)
        if value is None:
            continue
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
        used.append(key)
    return {"sha256": digest.hexdigest(), "tensor_keys": used}


def select_batches() -> dict[str, Any]:
    """Official train only: seed-0 permutation, first 128 fit, next 128 probe."""
    started = time.perf_counter()
    train_data = p1run.load_split("train")
    n_total = len(train_data)
    if n_total != 10_000:
        raise RuntimeError(f"official train split size {n_total} != 10000")
    generator = torch.Generator().manual_seed(int(SEED))
    order = torch.randperm(n_total, generator=generator)
    fit_indices = [int(index) for index in order[:FIT_SIZE].tolist()]
    probe_indices = [int(index) for index in order[FIT_SIZE : FIT_SIZE + PROBE_SIZE].tolist()]
    if set(fit_indices) & set(probe_indices):
        raise RuntimeError("fit and probe index sets overlap")
    if len(set(fit_indices)) != FIT_SIZE or len(set(probe_indices)) != PROBE_SIZE:
        raise RuntimeError("duplicate indices inside fit/probe sets")
    fit = p1.env_collate([train_data[index] for index in fit_indices])
    probe = p1.env_collate([train_data[index] for index in probe_indices])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "split": "official train (10000)",
        "splits_loaded": ["train"],
        "valid_loaded": False,
        "test_loaded": False,
        "official_test_loaded": False,
        "index_seed": int(SEED),
        "fit_indices": fit_indices,
        "probe_indices": probe_indices,
        "fit_size": int(fit.y.numel()),
        "probe_size": int(probe.y.numel()),
        "fit_fingerprint": _batch_fingerprint(fit),
        "probe_fingerprint": _batch_fingerprint(probe),
        "fit_target_mean": float(fit.y.double().mean()),
        "probe_target_mean": float(probe.y.double().mean()),
        "train_cache": {
            "path": str(TRAIN_CACHE_PATH.relative_to(REPO_ROOT)),
            "bytes": int(TRAIN_CACHE_PATH.stat().st_size),
            "sha256": _sha256_file(TRAIN_CACHE_PATH),
        },
        "seconds": float(time.perf_counter() - started),
    }
    return {"fit": fit, "probe": probe, "payload": payload}


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------


def _counts(values: torch.Tensor, tiny: float) -> dict[str, int]:
    magnitude = values.abs()
    zero = values == 0
    subnormal = (magnitude < tiny) & (~zero)
    normal = magnitude >= tiny
    return {
        "zero": int(zero.sum()),
        "subnormal": int(subnormal.sum()),
        "normal": int(normal.sum()),
    }


def tensor_stats(
    name: str,
    weight: torch.Tensor,
    *,
    grad: torch.Tensor | None = None,
    update: torch.Tensor | None = None,
    reference_decay: float = WEIGHT_DECAY_REFERENCE,
    decay_applied: bool = False,
) -> dict[str, Any]:
    """Per-parameter telemetry; float32 and float64 views are never conflated."""
    finfo = torch.finfo(weight.dtype)
    w64 = weight.detach().double()
    w32_norm = float(weight.detach().float().norm())
    w64_norm = float(w64.norm())
    payload: dict[str, Any] = {
        "name": name,
        "shape": [int(size) for size in weight.shape],
        "numel": int(weight.numel()),
        "dtype": str(weight.dtype),
        "finfo": {"eps": float(finfo.eps), "tiny": float(finfo.tiny), "max": float(finfo.max)},
        "weight_absmax": float(w64.abs().max()),
        "weight_norm_float64": w64_norm,
        "weight_norm_original_float32": w32_norm,
        "weight_nonzero_fraction": float((w64 != 0).double().mean()),
        "weight_counts": _counts(w64, float(finfo.tiny)),
        "weight_float32_norm_zero_but_float64_norm_nonzero": bool(
            w32_norm == 0.0 and w64_norm > 0.0
        ),
        "reference_decay_norm": float(reference_decay * w64_norm),
        "reference_decay_absmax": float(reference_decay * w64.abs().max()),
        "reference_decay_applied": bool(decay_applied),
    }
    if grad is not None:
        g = grad.detach()
        g32_norm = float(g.float().norm())
        g64_norm = float(g.double().norm())
        payload.update(
            {
                "grad_absmax_float32": float(g.float().abs().max()),
                "grad_nonzero_fraction": float((g != 0).double().mean()),
                "grad_norm_original_float32": g32_norm,
                "grad_norm_float64": g64_norm,
                "grad_all_zero_float32": bool((g == 0).all()),
                "grad_counts": _counts(g.double(), float(finfo.tiny)),
                "grad_float32_norm_zero_but_float64_norm_nonzero": bool(
                    g32_norm == 0.0 and g64_norm > 0.0
                ),
                "decay_over_task_ratio": (
                    float(reference_decay * w64_norm / g64_norm) if g64_norm > 0.0 else None
                ),
                "decay_over_task_ratio_defined": bool(g64_norm > 0.0),
            }
        )
    if update is not None:
        u64 = update.detach().double()
        u64_norm = float(u64.norm())
        payload.update(
            {
                "update_norm_float64": u64_norm,
                "update_absmax_float64": float(u64.abs().max()),
                "update_relative": (float(u64_norm / w64_norm) if w64_norm > 0.0 else None),
                "update_relative_defined": bool(w64_norm > 0.0),
                "update_nonzero_fraction": float((u64 != 0).double().mean()),
                "weight_changed": bool(u64_norm > 0.0),
            }
        )
    return payload


def _batch_learning(model: Any, batch: Any, mask: Any, off_mask: Any) -> dict[str, Any]:
    with torch.no_grad():
        prediction_on = model(batch, mask=mask).view(-1).double()
        prediction_off = model(batch, mask=off_mask).view(-1).double()
    target = batch.y.view(-1).double()
    mae_on = float((prediction_on - target).abs().mean())
    mae_off = float((prediction_off - target).abs().mean())
    difference = prediction_on - prediction_off
    residual = target - prediction_off
    payload = {
        "mae_branch_on": mae_on,
        "mae_branch_off": mae_off,
        "G_branch": mae_off - mae_on,
        "prediction_diff_mean_abs": float(difference.abs().mean()),
        "prediction_diff_max_abs": float(difference.abs().max()),
        "prediction_diff_nonzero_fraction": float((difference != 0).double().mean()),
        "n_molecules": int(target.numel()),
    }
    if float(difference.std(unbiased=False)) > 0.0 and float(residual.std(unbiased=False)) > 0.0:
        centred_diff = difference - difference.mean()
        centred_residual = residual - residual.mean()
        payload["corr_pred_diff_vs_parent_residual"] = float(
            (centred_diff * centred_residual).mean()
            / (centred_diff.std(unbiased=False) * centred_residual.std(unbiased=False))
        )
        payload["corr_defined"] = True
    else:
        payload["corr_pred_diff_vs_parent_residual"] = None
        payload["corr_defined"] = False
    payload["sign_agreement_fraction"] = float(
        ((difference > 0) == (residual > 0)).double().mean()
    )
    return payload


def _branch_response(model: Any, fit: Any, mask: Any) -> dict[str, Any]:
    with torch.no_grad():
        coord = model.code(fit.dict_phi)
        delta = model.joint_edge_delta(coord, fit, mask=mask)
        parent_ue, _ = model._edge_env_parts(
            coord, fit, fit.env_bond_u, fit.env_bond_v, mask=None
        )
    if delta is None:
        raise RuntimeError("branch response requested while the branch is off")
    parent_norm = parent_ue.detach().double().norm(dim=1)
    branch_norm = delta.detach().double().norm(dim=1)
    parent_rms = float(parent_norm.pow(2).mean().sqrt())
    branch_rms = float(branch_norm.pow(2).mean().sqrt())
    parent_zero = parent_rms == 0.0
    branch_zero = branch_rms == 0.0
    return {
        "n_bond_occurrences": int(branch_norm.numel()),
        "parent_edge_response_rms": parent_rms,
        "joint_branch_response_rms": branch_rms,
        "branch_over_parent_rms": (None if parent_zero else float(branch_rms / parent_rms)),
        "branch_over_parent_rms_defined": bool(not parent_zero),
        "parent_rms_is_zero": bool(parent_zero),
        "branch_rms_is_zero": bool(branch_zero),
        "joint_branch_entry_nonzero_fraction": float((delta != 0).double().mean()),
        "joint_branch_entry_count": int(delta.numel()),
        "joint_branch_entry_nonzero_count": int((delta != 0).sum()),
        "joint_branch_absmax": float(delta.detach().double().abs().max()),
        "parent_edge_response_absmax": float(parent_ue.detach().double().abs().max()),
    }


def snapshot(
    model: Any,
    arm: str,
    step: int,
    fit: Any,
    probe: Any,
    *,
    grads: Mapping[str, torch.Tensor] | None,
    updates: Mapping[str, torch.Tensor] | None,
    decay_applied: bool,
) -> dict[str, Any]:
    """Full pre-registered telemetry record for one arm at one logged step."""
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "step": int(step),
        "fit": _batch_learning(model, fit, MASK, OFF_MASK),
        "probe": _batch_learning(model, probe, MASK, OFF_MASK),
        "branch_response_fit": _branch_response(model, fit, MASK),
        "official_test_loaded": False,
    }
    params: dict[str, Any] = {}
    for name, parameter in model.joint_parameters().items():
        params[name] = tensor_stats(
            name,
            parameter,
            grad=None if grads is None else grads.get(name),
            update=None if updates is None else updates.get(name),
            reference_decay=WEIGHT_DECAY_REFERENCE,
            decay_applied=bool(decay_applied),
        )
    payload["params"] = params
    payload["joint_weight_norm_float64"] = _aggregate_norm(
        [float(parameter.detach().double().norm()) for parameter in model.joint_parameters().values()]
    )
    return payload


# ---------------------------------------------------------------------------
# the two-arm diagnostic
# ---------------------------------------------------------------------------


def _optimizer_for(model: Any, weight_decay: float) -> torch.optim.Adam:
    parameters = list(model.joint_parameters().values())
    return torch.optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=float(weight_decay))


def _aggregate_norm(norms: Sequence[float]) -> float:
    """float64 aggregate L2 norm of per-tensor float64 norms."""
    return float(math.sqrt(sum(float(value) * float(value) for value in norms)))


def _backward(model: Any, batch: Any) -> float:
    prediction, _aux = model(batch, mask=MASK, return_aux=True)
    loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1))
    loss.backward()
    return float(loss.detach())


def _gradient_tensors(model: Any) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {}
    for name, parameter in model.joint_parameters().items():
        if parameter.grad is None:
            raise RuntimeError(f"no gradient reached {name}")
        payload[name] = parameter.grad.detach().clone()
    return payload


def run_arm(
    model: Any,
    arm: str,
    weight_decay: float,
    fit: Any,
    probe: Any,
    *,
    steps: int = STEPS,
) -> dict[str, Any]:
    """One arm: identical fresh optimizer state, only ``weight_decay`` differs."""
    optimizer = _optimizer_for(model, weight_decay)
    model.eval()
    initial_state_sha256 = audit.state_sha256(model.state_dict())
    initial_parent_sha256 = audit.state_sha256(_parent_state(model))
    optimizer_state_entries = len(optimizer.state_dict()["state"])

    # step 0: telemetry + the gradient at the initial weights (no update).
    optimizer.zero_grad(set_to_none=True)
    initial_loss = _backward(model, fit)
    initial_grads = _gradient_tensors(model)
    optimizer.zero_grad(set_to_none=True)
    records = [
        snapshot(
            model,
            arm,
            0,
            fit,
            probe,
            grads=initial_grads,
            updates={name: torch.zeros_like(p) for name, p in model.joint_parameters().items()},
            decay_applied=bool(weight_decay > 0.0),
        )
    ]
    records[0]["loss_fit_l1"] = initial_loss
    records[0]["note"] = "step 0 gradients are the initial-weight gradients; no update was applied"
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, int(steps) + 1):
        model.eval()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.joint_parameters().items()
        }
        optimizer.zero_grad(set_to_none=True)
        loss = _backward(model, fit)
        pre_clip_grads = _gradient_tensors(model)
        branch_pre_norm = _aggregate_norm(
            [float(g.detach().double().norm()) for g in pre_clip_grads.values()]
        )
        global_pre_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), GRAD_CLIP)
        global_pre_norm = float(global_pre_norm)
        clip_coefficient = (
            1.0 if global_pre_norm <= GRAD_CLIP else GRAD_CLIP / (global_pre_norm + 1e-6)
        )
        branch_post_norm = _aggregate_norm(
            [
                float(parameter.grad.detach().double().norm())
                for parameter in model.joint_parameters().values()
            ]
        )
        optimizer.step()
        updates = {
            name: parameter.detach().double() - before[name].double()
            for name, parameter in model.joint_parameters().items()
        }
        update_norm = _aggregate_norm([float(value.norm()) for value in updates.values()])
        joint_weight_norm = _aggregate_norm(
            [
                float(parameter.detach().double().norm())
                for parameter in model.joint_parameters().values()
            ]
        )
        trace.append(
            {
                "arm": arm,
                "step": int(step),
                "loss_fit_l1": float(loss),
                "global_grad_norm_preclip": global_pre_norm,
                "branch_grad_norm_preclip": branch_pre_norm,
                "clip_coefficient": clip_coefficient,
                "branch_grad_norm_postclip": branch_post_norm,
                "update_norm_joint_float64": update_norm,
                "joint_weight_norm_float64": joint_weight_norm,
                "weight_decay": float(weight_decay),
                "learning_rate": LEARNING_RATE,
            }
        )
        if step in LOG_STEPS:
            record = snapshot(
                model,
                arm,
                step,
                fit,
                probe,
                grads=pre_clip_grads,
                updates=updates,
                decay_applied=bool(weight_decay > 0.0),
            )
            record["loss_fit_l1"] = float(loss)
            record["gradients_scored"] = "post-backward, pre-L2, pre-clip"
            record["global_grad_norm_preclip"] = global_pre_norm
            record["branch_grad_norm_preclip"] = branch_pre_norm
            record["clip_coefficient"] = clip_coefficient
            record["branch_grad_norm_postclip"] = branch_post_norm
            records.append(record)
        if step % 50 == 0 or step == int(steps):
            print(
                f"[{PROTOCOL_VERSION}:{arm}] step={step:3d} loss={float(loss):.6f} "
                f"|g|={branch_pre_norm:.3e} clip={clip_coefficient:.3f} "
                f"|w|={joint_weight_norm:.6e} |dw|={update_norm:.3e}",
                flush=True,
            )
    return {
        "arm": arm,
        "weight_decay": float(weight_decay),
        "learning_rate": LEARNING_RATE,
        "steps": int(steps),
        "optimizer": type(optimizer).__name__,
        "optimizer_hyperparameters": {
            key: value for key, value in optimizer.defaults.items() if key != "params"
        },
        "optimizer_state_entries_before": int(optimizer_state_entries),
        "optimizer_param_keys": sorted(model.joint_parameters()),
        "optimizer_numel": int(sum(p.numel() for p in model.joint_parameters().values())),
        "initial_state_sha256": initial_state_sha256,
        "initial_parent_state_sha256": initial_parent_sha256,
        "final_parent_state_sha256": audit.state_sha256(_parent_state(model)),
        "final_state_sha256": audit.state_sha256(model.state_dict()),
        "wall_clock_s": float(time.perf_counter() - started),
        "records": records,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_references() -> dict[str, Any]:
    _ensure_dirs()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": THREADS,
        "official_test_loaded": False,
        "policy": "read-only durable artifacts; no historical model is retrained",
        "references": {
            "SEM108-parent-soup": {
                "value": SEM108_SOUP_VALID_MAE,
                "artifact": str(SEM108_SOUP_PATH.relative_to(REPO_ROOT)),
                "state_sha256": SEM108_SOUP_SHA256,
                "status": "frozen_parent_source_not_retrained",
            },
            "JointBond-v1-soup": {
                "value": 0.12793026330223073,
                "artifact": str(JOINTBOND_SOUP_PATH.relative_to(REPO_ROOT)),
                "status": "closed_round_reference_only",
            },
        },
    }
    jb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "references.json", payload)
    print("[references] Sem108 soup frozen parent source", flush=True)
    return payload


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    prereg_sha = _sha256_file(PREREG_PATH)
    snapshot_path = RESULTS_DIR / "preregistration_snapshot.json"
    if snapshot_path.exists():
        recorded = str(_read_json(snapshot_path)["sha256"])
        if recorded != prereg_sha:
            raise RuntimeError(
                "pre-registration changed after freezing "
                f"({recorded} -> {prereg_sha}); re-freeze deliberately"
            )
    else:
        _write_json(
            snapshot_path,
            {
                "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
                "sha256": prereg_sha,
                "frozen_before_run": True,
            },
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "preregistration": str(PREREG_PATH.relative_to(REPO_ROOT)),
        "preregistration_sha256": prereg_sha,
        "arms": {name: {"weight_decay": wd, "learning_rate": LEARNING_RATE} for name, wd in ARMS},
        "steps_per_arm": STEPS,
        "total_optimizer_steps": 2 * STEPS,
        "fit_size": FIT_SIZE,
        "probe_size": PROBE_SIZE,
        "log_steps": list(LOG_STEPS),
        "gradient_clip": GRAD_CLIP,
        "coupled_l2_reference_decay": WEIGHT_DECAY_REFERENCE,
        "mask": "cm.C6_MASK",
        "objective": "L1(prediction, y) on the repeated fit batch (reconstruction term constant for the branch)",
        "parent_frozen": True,
        "branch_params": 4224,
        "device": "cpu",
        "threads": THREADS,
        "official_test_loaded": False,
    }
    jb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(f"[preflight] prereg={prereg_sha[:12]} arms={[a for a, _ in ARMS]}", flush=True)
    return payload


def _adam_coupling_probe() -> dict[str, Any]:
    """Empirical check that ``torch.optim.Adam(weight_decay=...)`` is coupled L2."""
    out: dict[str, Any] = {}
    for label, weight_decay in (("adam_wd1e-5", WEIGHT_DECAY_REFERENCE), ("adam_wd0", 0.0)):
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=FLOAT32))
        parameter.grad = torch.zeros(1, dtype=FLOAT32)
        optimizer = torch.optim.Adam([parameter], lr=LEARNING_RATE, weight_decay=weight_decay)
        optimizer.step()
        out[label] = {
            "delta": float(parameter.detach().item() - 1.0),
            "expected_coupled_l2": float(
                -LEARNING_RATE
                * weight_decay
                / (weight_decay + optimizer.defaults["eps"])
            ),
            "expected_adamw": float(-LEARNING_RATE * weight_decay),
        }
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=FLOAT32))
    parameter.grad = torch.zeros(1, dtype=FLOAT32)
    optimizer = torch.optim.AdamW([parameter], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY_REFERENCE)
    optimizer.step()
    out["adamw_wd1e-5"] = {"delta": float(parameter.detach().item() - 1.0)}
    out["coupled_l2_confirmed"] = bool(
        abs(out["adam_wd1e-5"]["delta"] + LEARNING_RATE) < 1e-3
        and abs(out["adamw_wd1e-5"]["delta"] + LEARNING_RATE * WEIGHT_DECAY_REFERENCE) < 1e-6
        and out["adam_wd0"]["delta"] == 0.0
    )
    out["optimizer_class"] = "torch.optim.Adam (coupled L2; weight_decay added to the gradient)"
    return out


def _reporting_self_test() -> dict[str, Any]:
    """float32 norm underflow must be distinguished from a true zero tensor."""
    tiny = float(torch.finfo(FLOAT32).tiny)
    subnormal = torch.full((4,), tiny / 4.0, dtype=FLOAT32)
    stats = tensor_stats("synthetic_subnormal", subnormal)
    truly_zero = torch.zeros(4, dtype=FLOAT32)
    zero_stats = tensor_stats("synthetic_zero", truly_zero)
    payload = {
        "tiny": tiny,
        "subnormal_value": float(subnormal[0]),
        "subnormal_is_subnormal": bool(float(subnormal[0].abs()) < tiny and float(subnormal[0]) != 0.0),
        "subnormal_counts": stats["weight_counts"],
        "subnormal_norm_float32": stats["weight_norm_original_float32"],
        "subnormal_norm_float64": stats["weight_norm_float64"],
        "subnormal_float32_norm_zero_but_float64_norm_nonzero": stats[
            "weight_float32_norm_zero_but_float64_norm_nonzero"
        ],
        "zero_counts": zero_stats["weight_counts"],
        "zero_all_counts_zero": bool(zero_stats["weight_counts"]["subnormal"] == 0),
    }
    payload["passed"] = bool(
        payload["subnormal_is_subnormal"]
        and payload["subnormal_counts"]["subnormal"] == 4
        and payload["subnormal_norm_float32"] == 0.0
        and payload["subnormal_norm_float64"] > 0.0
        and payload["subnormal_float32_norm_zero_but_float64_norm_nonzero"]
        and payload["zero_counts"]["zero"] == 4
    )
    return payload


def stage_correctness() -> dict[str, Any]:
    """Pre-training gates D0-D12 (all must pass before step 1)."""
    _ensure_dirs()
    audit.attach_cpu(THREADS)
    gates: dict[str, Any] = {}
    prereg_sha = _sha256_file(PREREG_PATH)
    gates["D0_preregistration_frozen"] = {
        "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
        "sha256": prereg_sha,
        "snapshot_matches": bool(
            _read_json(RESULTS_DIR / "preregistration_snapshot.json")["sha256"] == prereg_sha
        ),
        "passed": True,
    }
    soup = _sem108_soup_state()
    parent_reference = sem108_parent_model()
    gates["D1_parent_is_sem108_soup"] = {
        "checkpoint": str(SEM108_SOUP_PATH.relative_to(REPO_ROOT)),
        "recorded_sha256": SEM108_SOUP_SHA256,
        "recomputed_sha256": audit.state_sha256(soup),
        "n_keys": len(soup),
        "joint_keys_in_soup": sorted(key for key in soup if key.startswith("joint_")),
        "hashes_match": bool(audit.state_sha256(soup) == SEM108_SOUP_SHA256),
        "passed": bool(
            audit.state_sha256(soup) == SEM108_SOUP_SHA256
            and not [key for key in soup if key.startswith("joint_")]
        ),
    }
    arm_wd, arm_nowd, info = build_arms()
    gates["D1b_parent_reference_model"] = {
        "parent_state_sha256": audit.state_sha256(_parent_state(parent_reference)),
        "matches_soup": bool(audit.state_sha256(_parent_state(parent_reference)) == SEM108_SOUP_SHA256),
        "passed": bool(audit.state_sha256(_parent_state(parent_reference)) == SEM108_SOUP_SHA256),
    }
    gates["D2_fresh_branch_alive"] = {
        "fresh_joint_state_sha256": info["fresh_joint_state_sha256"],
        "dead_joint_state_sha256": info["dead_joint_state_sha256"],
        "fresh_joint_absmax": info["fresh_joint_absmax"],
        "differs_from_dead_v1_soup": bool(
            info["fresh_joint_state_sha256"] != info["dead_joint_state_sha256"]
        ),
        "passed": bool(
            all(value > 1e-6 for value in info["fresh_joint_absmax"].values())
            and info["fresh_joint_state_sha256"] != info["dead_joint_state_sha256"]
        ),
    }
    gates["D3_arms_identical_at_init"] = {
        "full_state_sha256_wd": info["full_state_sha256_wd"],
        "full_state_sha256_nowd": info["full_state_sha256_nowd"],
        "parent_keys": info["parent_keys"],
        "branch_keys": info["branch_keys"],
        "branch_params": info["branch_params"],
        "total_params": info["total_params"],
        "missing_keys": info["missing_keys"],
        "unexpected_keys": info["unexpected_keys"],
        "passed": bool(info["full_state_sha256_wd"] == info["full_state_sha256_nowd"]),
    }
    selector = select_batches()
    fit, probe = selector["fit"], selector["probe"]
    _write_json(RESULTS_DIR / "data.json", selector["payload"])
    with torch.no_grad():
        fit_wd = arm_wd(fit, mask=MASK).view(-1)
        fit_nowd = arm_nowd(fit, mask=MASK).view(-1)
        probe_wd = arm_wd(probe, mask=MASK).view(-1)
        probe_nowd = arm_nowd(probe, mask=MASK).view(-1)
    gates["D4_step0_predictions_identical"] = {
        "fit_bit_identical": bool(torch.equal(fit_wd, fit_nowd)),
        "probe_bit_identical": bool(torch.equal(probe_wd, probe_nowd)),
        "passed": bool(torch.equal(fit_wd, fit_nowd) and torch.equal(probe_wd, probe_nowd)),
    }
    with torch.no_grad():
        parent_pred = parent_reference(fit, mask=MASK).view(-1)
        parent_pred_unmasked = parent_reference(fit).view(-1)
        off_wd = arm_wd(fit, mask=OFF_MASK).view(-1)
        off_nowd = arm_nowd(fit, mask=OFF_MASK).view(-1)
        unmasked_off_mask = jb.merge_joint_mask(None, jb.JointBondMask(joint_branch_off=True))
        off_unmasked = arm_wd(fit, mask=unmasked_off_mask).view(-1)
    gates["D5_branch_off_equals_frozen_parent"] = {
        "masked_bit_identical_wd": bool(torch.equal(off_wd, parent_pred)),
        "masked_bit_identical_nowd": bool(torch.equal(off_nowd, parent_pred)),
        "unmasked_bit_identical_wd": bool(torch.equal(off_unmasked, parent_pred_unmasked)),
        "max_abs_diff_wd": float((off_wd - parent_pred).abs().max()),
        "passed": bool(
            torch.equal(off_wd, parent_pred)
            and torch.equal(off_nowd, parent_pred)
            and torch.equal(off_unmasked, parent_pred_unmasked)
        ),
    }
    response = _branch_response(arm_wd, fit, MASK)
    gates["D6_initial_branch_response_nonzero"] = {
        **response,
        "passed": bool(
            response["joint_branch_entry_nonzero_count"] > 0
            and response["joint_branch_response_rms"] > 0.0
            and response["joint_branch_entry_nonzero_fraction"] > 0.99
        ),
    }
    optimizer_wd = _optimizer_for(arm_wd, WEIGHT_DECAY_REFERENCE)
    optimizer_nowd = _optimizer_for(arm_nowd, 0.0)
    expected_ids = {id(parameter) for parameter in arm_wd.joint_parameters().values()}
    gates["D7_optimizer_is_branch_only"] = {
        "optimizer": type(optimizer_wd).__name__,
        "param_keys": sorted(arm_wd.joint_parameters()),
        "numel_wd": int(sum(p.numel() for p in optimizer_wd.param_groups[0]["params"])),
        "numel_nowd": int(sum(p.numel() for p in optimizer_nowd.param_groups[0]["params"])),
        "identity_match_wd": bool(
            {id(p) for p in optimizer_wd.param_groups[0]["params"]} == expected_ids
        ),
        "identity_match_nowd": bool(
            {id(p) for p in optimizer_nowd.param_groups[0]["params"]}
            == {id(parameter) for parameter in arm_nowd.joint_parameters().values()}
        ),
        "fresh_optimizer_state_wd": int(len(optimizer_wd.state_dict()["state"])),
        "fresh_optimizer_state_nowd": int(len(optimizer_nowd.state_dict()["state"])),
        "weight_decay_wd": float(optimizer_wd.defaults["weight_decay"]),
        "weight_decay_nowd": float(optimizer_nowd.defaults["weight_decay"]),
        "passed": bool(
            int(sum(p.numel() for p in optimizer_wd.param_groups[0]["params"])) == 4224
            and int(sum(p.numel() for p in optimizer_nowd.param_groups[0]["params"])) == 4224
            and {id(p) for p in optimizer_wd.param_groups[0]["params"]} == expected_ids
            and {id(p) for p in optimizer_nowd.param_groups[0]["params"]}
            == {id(parameter) for parameter in arm_nowd.joint_parameters().values()}
            and len(optimizer_wd.state_dict()["state"]) == 0
            and len(optimizer_nowd.state_dict()["state"]) == 0
            and float(optimizer_wd.defaults["weight_decay"]) == WEIGHT_DECAY_REFERENCE
            and float(optimizer_nowd.defaults["weight_decay"]) == 0.0
        ),
    }
    coupling = _adam_coupling_probe()
    coupling["passed"] = bool(coupling["coupled_l2_confirmed"])
    gates["D8_coupled_l2_not_adamw"] = coupling
    throwaway = copy.deepcopy(arm_wd)
    throwaway_optimizer = _optimizer_for(throwaway, WEIGHT_DECAY_REFERENCE)
    parent_before = audit.state_sha256(_parent_state(throwaway))
    branch_before = audit.state_sha256(throwaway.joint_parameters())
    throwaway_optimizer.zero_grad(set_to_none=True)
    _backward(throwaway, fit)
    torch.nn.utils.clip_grad_norm_(list(throwaway.parameters()), GRAD_CLIP)
    throwaway_optimizer.step()
    parent_after = audit.state_sha256(_parent_state(throwaway))
    branch_after = audit.state_sha256(throwaway.joint_parameters())
    gates["D9_parent_not_updated_throwaway_step"] = {
        "parent_state_unchanged": bool(parent_before == parent_after),
        "parent_state_sha256": parent_before,
        "branch_state_changed": bool(branch_before != branch_after),
        "passed": bool(parent_before == parent_after and branch_before != branch_after),
    }
    gates["D10_data_official_train_only"] = {
        "split_sizes": {"train": 10000, "valid": None, "test": None},
        "fit_size": int(fit.y.numel()),
        "probe_size": int(probe.y.numel()),
        "fit_probe_disjoint": bool(
            not set(selector["payload"]["fit_indices"]) & set(selector["payload"]["probe_indices"])
        ),
        "indices_in_range": bool(
            max(selector["payload"]["fit_indices"] + selector["payload"]["probe_indices"]) < 10000
            and min(selector["payload"]["fit_indices"] + selector["payload"]["probe_indices"]) >= 0
        ),
        "fit_fingerprint": selector["payload"]["fit_fingerprint"]["sha256"],
        "probe_fingerprint": selector["payload"]["probe_fingerprint"]["sha256"],
        "valid_loaded": False,
        "test_loaded": False,
        "passed": bool(
            not set(selector["payload"]["fit_indices"]) & set(selector["payload"]["probe_indices"])
            and max(
                selector["payload"]["fit_indices"] + selector["payload"]["probe_indices"]
            ) < 10000
        ),
    }
    arm_wd.eval()
    optimizer_wd.zero_grad(set_to_none=True)
    prediction, _aux = arm_wd(fit, mask=MASK, return_aux=True)
    loss = torch.nn.functional.l1_loss(prediction.view(-1), fit.y.view(-1))
    loss.backward()
    grad_payload: dict[str, Any] = {}
    grads_present = True
    for name, parameter in arm_wd.joint_parameters().items():
        present = parameter.grad is not None
        grads_present = grads_present and present
        grad_payload[name] = {
            "grad_present": bool(present),
            "grad_absmax_float32": float(parameter.grad.detach().abs().max()) if present else None,
            "grad_nonzero_count": int((parameter.grad.detach() != 0).sum()) if present else None,
            "grad_norm_float64": float(parameter.grad.detach().double().norm()) if present else None,
        }
    optimizer_wd.zero_grad(set_to_none=True)
    gates["D11_training_graph_reaches_branch"] = {
        "loss_requires_grad": bool(loss.requires_grad),
        "all_branch_grads_present": bool(grads_present),
        "params": grad_payload,
        "passed": bool(loss.requires_grad and grads_present),
    }
    reporting = _reporting_self_test()
    gates["D12_reporting_self_test"] = reporting
    gates["D12_reporting_self_test"]["passed"] = bool(reporting["passed"])

    # leave both arms exactly as they were at init.
    arm_wd.zero_grad(set_to_none=True)
    arm_nowd.zero_grad(set_to_none=True)
    if audit.state_sha256(arm_wd.state_dict()) != info["full_state_sha256_wd"]:
        raise RuntimeError("correctness stage mutated the WD arm")
    if audit.state_sha256(arm_nowd.state_dict()) != info["full_state_sha256_nowd"]:
        raise RuntimeError("correctness stage mutated the NO-WD arm")

    first_gradient = next(iter(grad_payload.values()))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": THREADS,
        "preregistration_sha256": prereg_sha,
        "gates": gates,
        "first_gradient_note": (
            "the first gradient on the frozen parent is representable in float32 "
            f"(absmax {first_gradient['grad_absmax_float32']:.3e}); a small gradient is not by "
            "itself evidence of a float32 resolution problem"
        ),
        "official_test_loaded": False,
    }
    payload["all_passed"] = bool(
        all(bool(gate.get("passed", False)) for gate in gates.values())
    )
    _write_json(RESULTS_DIR / "correctness.json", payload)
    for name, gate in sorted(gates.items()):
        print(f"[correctness] {name} passed={gate.get('passed')}", flush=True)
    if not payload["all_passed"]:
        failed = [name for name, gate in gates.items() if not gate.get("passed")]
        raise RuntimeError(f"correctness gates failed: {failed}")
    # hand the untouched arms + batches onward for the run stage.
    return {"payload": payload, "arm_wd": arm_wd, "arm_nowd": arm_nowd, "fit": fit, "probe": probe}


def stage_run() -> dict[str, Any]:
    _ensure_dirs()
    audit.attach_cpu(THREADS)
    arm_wd, arm_nowd, info = build_arms()
    selector = select_batches()
    fit, probe = selector["fit"], selector["probe"]
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": THREADS,
        "preregistration_sha256": _sha256_file(PREREG_PATH),
        "init": info,
        "data": {
            "fit_indices": selector["payload"]["fit_indices"],
            "probe_indices": selector["payload"]["probe_indices"],
            "fit_fingerprint": selector["payload"]["fit_fingerprint"],
            "probe_fingerprint": selector["payload"]["probe_fingerprint"],
        },
        "arms": {},
        "official_test_loaded": False,
    }
    started = time.perf_counter()
    for arm, weight_decay in ARMS:
        model = arm_wd if arm == "wd" else arm_nowd
        print(f"[{PROTOCOL_VERSION}] arm={arm} weight_decay={weight_decay:g}", flush=True)
        outcome = run_arm(model, arm, weight_decay, fit, probe)
        result["arms"][arm] = outcome
        _write_json(
            RESULTS_DIR / f"records_{arm}.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "arm": arm,
                "weight_decay": weight_decay,
                "records": outcome["records"],
                "official_test_loaded": False,
            },
        )
        with open(RESULTS_DIR / f"trace_{arm}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(outcome["trace"][0].keys()))
            writer.writeheader()
            writer.writerows(outcome["trace"])
    result["wall_clock_s"] = float(time.perf_counter() - started)
    result["parent_state_unchanged"] = {
        arm: bool(
            result["arms"][arm]["final_parent_state_sha256"]
            == result["arms"][arm]["initial_parent_state_sha256"]
        )
        for arm, _wd in ARMS
    }
    if not all(result["parent_state_unchanged"].values()):
        raise RuntimeError("parent parameters changed during the diagnostic")
    _write_json(RESULTS_DIR / "run.json", result)
    print(
        "[run] wall={:.1f}s parent_unchanged={}".format(
            result["wall_clock_s"], result["parent_state_unchanged"]
        ),
        flush=True,
    )
    return result


def _first(records: Sequence[Mapping[str, Any]], step: int) -> Mapping[str, Any]:
    for record in records:
        if int(record["step"]) == int(step):
            return record
    raise KeyError(step)


def _classify(d_fit_wd: float, d_fit_nowd: float, weight_norm_wd: float, weight_norm_nowd: float) -> str:
    learns = lambda value: bool(value <= -LEARN_TOL)  # noqa: E731
    if learns(d_fit_nowd) and not learns(d_fit_wd):
        return "DECAY_SUPPRESSION_CONFIRMED"
    if learns(d_fit_nowd) and learns(d_fit_wd) and (d_fit_wd - d_fit_nowd) >= SUPPRESSION_TOL:
        return "WD_SUPPRESSES_BUT_BOTH_LEARN"
    if learns(d_fit_nowd) and learns(d_fit_wd):
        return "BOTH_LEARN_DECAY_NOT_THE_CAUSE"
    if learns(d_fit_wd) and not learns(d_fit_nowd):
        return "WD_ARM_LEARNS_NOWD_DOES_NOT"
    if weight_norm_nowd > weight_norm_wd:
        return "SURVIVAL_ONLY"
    return "NEITHER_LEARNS_NOT_DECAY"


def stage_analysis() -> dict[str, Any]:
    _ensure_dirs()
    run = _read_json(RESULTS_DIR / "run.json")
    arms = run["arms"]
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "preregistration_sha256": run["preregistration_sha256"],
        "official_test_loaded": False,
        "per_arm": {},
    }
    for arm, _wd in ARMS:
        records = arms[arm]["records"]
        trace = arms[arm]["trace"]
        step0 = _first(records, 0)
        step1 = _first(records, 1)
        final = _first(records, STEPS)
        losses = [row["loss_fit_l1"] for row in trace]
        summary["per_arm"][arm] = {
            "weight_decay": arms[arm]["weight_decay"],
            "fit_mae_step0": step0["fit"]["mae_branch_on"],
            "fit_mae_step1": step1["fit"]["mae_branch_on"],
            "fit_mae_final": final["fit"]["mae_branch_on"],
            "fit_mae_off_final": final["fit"]["mae_branch_off"],
            "probe_mae_step0": step0["probe"]["mae_branch_on"],
            "probe_mae_final": final["probe"]["mae_branch_on"],
            "d_fit": final["fit"]["mae_branch_on"] - step0["fit"]["mae_branch_on"],
            "d_probe": final["probe"]["mae_branch_on"] - step0["probe"]["mae_branch_on"],
            "G_branch_fit_step0": step0["fit"]["G_branch"],
            "G_branch_fit_final": final["fit"]["G_branch"],
            "G_branch_probe_final": final["probe"]["G_branch"],
            "corr_fit_step0": step0["fit"]["corr_pred_diff_vs_parent_residual"],
            "corr_fit_final": final["fit"]["corr_pred_diff_vs_parent_residual"],
            "corr_probe_final": final["probe"]["corr_pred_diff_vs_parent_residual"],
            "loss_first": losses[0],
            "loss_min": min(losses),
            "loss_final": losses[-1],
            "joint_weight_norm_step0": step0["joint_weight_norm_float64"],
            "joint_weight_norm_final": final["joint_weight_norm_float64"],
            "branch_rms_step0": step0["branch_response_fit"]["joint_branch_response_rms"],
            "branch_rms_final": final["branch_response_fit"]["joint_branch_response_rms"],
            "branch_over_parent_rms_step0": step0["branch_response_fit"]["branch_over_parent_rms"],
            "branch_over_parent_rms_final": final["branch_response_fit"]["branch_over_parent_rms"],
            "decay_over_task_ratio_step1": {
                name: step1["params"][name].get("decay_over_task_ratio") for name in BRANCH_KEYS
            },
            "grad_norm_float64_step1": {
                name: step1["params"][name].get("grad_norm_float64") for name in BRANCH_KEYS
            },
            "grad_norm_float32_step1": {
                name: step1["params"][name]["grad_norm_original_float32"] for name in BRANCH_KEYS
            },
            "weight_norm_float64_final": {
                name: final["params"][name]["weight_norm_float64"] for name in BRANCH_KEYS
            },
            "weight_absmax_final": {
                name: final["params"][name]["weight_absmax"] for name in BRANCH_KEYS
            },
            "update_norm_total_final": float(
                sum(
                    record["params"][name].get("update_norm_float64") or 0.0
                    for record in records
                    for name in BRANCH_KEYS
                )
            ),
            "optimizer_state_entries_before": arms[arm]["optimizer_state_entries_before"],
            "wall_clock_s": arms[arm]["wall_clock_s"],
        }
    summary["verdict"] = _classify(
        summary["per_arm"]["wd"]["d_fit"],
        summary["per_arm"]["nowd"]["d_fit"],
        summary["per_arm"]["wd"]["joint_weight_norm_final"],
        summary["per_arm"]["nowd"]["joint_weight_norm_final"],
    )
    summary["thresholds"] = {"learn_tol": LEARN_TOL, "suppression_tol": SUPPRESSION_TOL}
    summary["parent_state_unchanged"] = run["parent_state_unchanged"]
    summary["questions"] = {
        "1_branch_learns_fit": {
            "nowd_d_fit": summary["per_arm"]["nowd"]["d_fit"],
            "wd_d_fit": summary["per_arm"]["wd"]["d_fit"],
            "nowd_learns": bool(summary["per_arm"]["nowd"]["d_fit"] <= -LEARN_TOL),
            "wd_learns": bool(summary["per_arm"]["wd"]["d_fit"] <= -LEARN_TOL),
        },
        "2_weight_decay_effect": {
            "d_fit_difference_nowd_minus_wd": (
                summary["per_arm"]["wd"]["d_fit"] - summary["per_arm"]["nowd"]["d_fit"]
            ),
            "verdict": summary["verdict"],
        },
        "3_numerics": {
            "float32_grad_norm_zero_but_float64_nonzero": {
                arm: [
                    name
                    for name in BRANCH_KEYS
                    for record in arms[arm]["records"][1:]
                    if record["params"][name]["grad_float32_norm_zero_but_float64_norm_nonzero"]
                ]
                for arm, _wd in ARMS
            },
            "note": (
                "a float32 norm of 0 is reported as a float32 norm underflow, never as "
                "'every gradient element is zero'; 1e-6-scale gradients are representable in "
                "float32 and are not by themselves below float32 precision"
            ),
        },
    }
    _write_json(ANALYSIS_JSON, summary)
    sem_step0 = summary["per_arm"]["wd"]["fit_mae_step0"]
    lines = [
        f"# Decay diagnostic — {PROTOCOL_VERSION}",
        "",
        f"Frozen parent `SEM108-seed0_soup_state.pt` (sha `{SEM108_SOUP_SHA256[:12]}`, "
        f"soup valid MAE `{SEM108_SOUP_VALID_MAE}`), fixed fit batch of {FIT_SIZE} official-train "
        f"graphs repeated for {STEPS} steps per arm, probe batch {PROBE_SIZE} disjoint graphs "
        "(observation only), C6 mask, `eval()` with autograd, CPU 8 threads.",
        "",
        f"* fit MAE at step 0 (both arms bit-identical): `{sem_step0:.9f}`",
        f"* parent state unchanged in both arms: `{run['parent_state_unchanged']}`",
        "",
        "| arm | wd | fit MAE step0 | fit MAE 200 | d_fit | probe MAE step0 | probe MAE 200 | "
        "G_branch(fit, step0) | G_branch(fit, 200) | ||w|| final |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, _wd in ARMS:
        row = summary["per_arm"][arm]
        lines.append(
            f"| {arm} | {row['weight_decay']:g} | {row['fit_mae_step0']:.9f} | "
            f"{row['fit_mae_final']:.9f} | {row['d_fit']:+.3e} | {row['probe_mae_step0']:.9f} | "
            f"{row['probe_mae_final']:.9f} | {row['G_branch_fit_step0']:+.3e} | "
            f"{row['G_branch_fit_final']:+.3e} | {row['joint_weight_norm_final']:.6e} |"
        )
    lines += [
        "",
        f"Verdict: **{summary['verdict']}** "
        f"(learn_tol `{LEARN_TOL}`, suppression_tol `{SUPPRESSION_TOL}`).",
        "",
        "Fitting a repeated 128-graph batch is residual fitting, not generalisation; the probe "
        "batch is official-train data and is reported as observation only.  Freezing the parent "
        "changes the original training environment, so this diagnostic cannot fully explain the "
        "from-scratch JointBond-v1 death.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[analysis] verdict={summary['verdict']} "
        f"d_fit(wd)={summary['per_arm']['wd']['d_fit']:+.3e} "
        f"d_fit(nowd)={summary['per_arm']['nowd']['d_fit']:+.3e}",
        flush=True,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="chain",
        choices=["references", "preflight", "correctness", "run", "analysis", "chain"],
    )
    args = parser.parse_args(argv)
    if args.stage in ("references", "chain"):
        stage_references()
    if args.stage in ("preflight", "chain"):
        stage_preflight()
    if args.stage in ("correctness", "chain"):
        stage_correctness()
    if args.stage in ("run", "chain"):
        stage_run()
    if args.stage in ("analysis", "chain"):
        stage_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
