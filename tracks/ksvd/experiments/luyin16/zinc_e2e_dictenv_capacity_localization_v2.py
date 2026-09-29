"""Runner for ``e2e_dictenv_capacity_localization_v2``.

Stages
------
``preflight  init-audit  calibration  screen-arm  wave1  wave2  summarize
  full-run  mechanism  report  chain``

The round repairs the v1 warm-adaptation screen instead of re-designing the
candidates:

* Phase A — one 20-epoch M0 calibration run at ``Adam(lr = 1e-4)``; a frozen
  stability gate decides whether the screen is authorised at all;
* Phase B — four 40-epoch warm-start arms from the same CAP-BASE soup
  (``M0``/``F``/``R``/``G``) with differential LR (base ``1e-4``, new capacity
  ``1e-3``);
* Phase C — at most ONE from-scratch 320-epoch seed-0 winner run.

CPU only; the official ZINC test split is never loaded.  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_capacity_localization_v2_preregistration.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_capacity_localization_v1 as cl
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_capacity_localization_v2 as cv2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = cv2.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_capacity_localization_v2"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_capacity_localization_v2_preregistration.md"

CALIBRATION_DIR = RESULTS_DIR / "calibration"
CALIBRATION_M0_DIR = CALIBRATION_DIR / "m0"
SCREENING_DIR = RESULTS_DIR / "screening"
FULL_DIR = RESULTS_DIR / "full"
FULL_WINNER_DIR = FULL_DIR / "winner"
FULL_CHECKPOINT_DIR = FULL_WINNER_DIR / "checkpoints"
MECHANISM_DIR = RESULTS_DIR / "mechanism"

CSSD_DIR = TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1"
CSSD_SOUP_PATH = CSSD_DIR / "training/checkpoints/CSSD-Q1-seed0_soup_state.pt"
CSSD_SUBSPACE_PATH = CSSD_DIR / "common_subspace.json"
CSSD_CURVE_PATH = CSSD_DIR / "training/curve.csv"

PROBES = ("P1_phi_row_shuffle", "P2_node_correspondence", "P3_edge_correspondence", "P4_relation")
PROBE_SEEDS = (4242, 5150, 6262)

#: v1 added-parameter counts: cross-round architecture identity check.
V1_ADDED_PARAMS = {"F": 29568, "R": 34400, "G": 30336}
V1_TOTAL_PARAMS = {"M0": 97727, "F": 127295, "R": 132127, "G": 128063}
V1_STEP0_MEAN_SHIFT = {"F": 0.01003, "R": 0.01741, "G": 0.00113}

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit

ARM_DIRS = {
    "M0": SCREENING_DIR / "m0",
    "F": SCREENING_DIR / "fusion",
    "R": SCREENING_DIR / "relation",
    "G": SCREENING_DIR / "readout",
}
ARM_TAGS = {"M0": "M0", "F": "FUSION", "R": "RELATION", "G": "READOUT"}

THREADS = 4
SCREEN_CONCURRENCY = 3
FULL_THREADS = 4


# ---------------------------------------------------------------------------
# small io helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([row.get(key, "") for key in header])


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ensure_dirs() -> None:
    for path in (
        RESULTS_DIR,
        CALIBRATION_DIR,
        CALIBRATION_M0_DIR,
        SCREENING_DIR,
        FULL_DIR,
        FULL_WINNER_DIR,
        FULL_CHECKPOINT_DIR,
        MECHANISM_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
    for path in ARM_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def _load_subspace() -> cssd.CommonSubspace:
    payload = _read_json(CSSD_SUBSPACE_PATH)["q1"]
    return cssd.CommonSubspace(
        components=np.asarray(payload["components"], dtype=np.float64),
        rms=np.asarray(payload["rms"], dtype=np.float64),
        kind=str(payload["kind"]),
    )


def _load_soup() -> dict[str, torch.Tensor]:
    return torch.load(CSSD_SOUP_PATH, map_location="cpu", weights_only=False)


def _payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    cv2.official_test_blocker(payload)
    return dict(payload)


def _calibration_decision() -> dict[str, Any] | None:
    path = CALIBRATION_DIR / "calibration_decision.json"
    if not path.exists():
        return None
    return _read_json(path)


def _calibration_passed() -> bool:
    decision = _calibration_decision()
    return bool(decision and decision.get("overall_pass"))


# ---------------------------------------------------------------------------
# stage: parameter budget + architecture fingerprints
# ---------------------------------------------------------------------------


def parameter_budget_v2(subspace: cssd.CommonSubspace, dictionary: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for kind in cv2.KINDS:
        model = cv2.build_v2_model(dictionary, 0, subspace, kind)
        partition = cv2.parameter_partition(model)
        parameters = dict(model.named_parameters())
        rows[kind] = {
            "kind": kind,
            "total_params": int(sum(parameter.numel() for parameter in model.parameters())),
            "capacity_params": int(
                sum(parameters[name].numel() for name in partition["new"])
            ),
            "capacity_parameter_names": list(partition["new"]),
            "base_params": int(sum(parameters[name].numel() for name in partition["base"])),
        }
    base_total = int(rows["M0"]["total_params"])
    for kind in cv2.KINDS:
        rows[kind]["added_params_vs_cap_base"] = int(rows[kind]["total_params"]) - base_total
        rows[kind]["relative_increase"] = float(rows[kind]["total_params"]) / float(base_total) - 1.0
        rows[kind]["within_preferred"] = bool(
            cv2.BUDGET_PREFERRED[0] <= rows[kind]["total_params"] <= cv2.BUDGET_PREFERRED[1]
        )
        rows[kind]["within_hard_ceiling"] = bool(rows[kind]["total_params"] <= cv2.BUDGET_HARD_CEILING)
        if kind in V1_ADDED_PARAMS:
            rows[kind]["v1_added_params"] = int(V1_ADDED_PARAMS[kind])
            rows[kind]["matches_v1_added_params"] = bool(
                int(rows[kind]["added_params_vs_cap_base"]) == int(V1_ADDED_PARAMS[kind])
            )
            rows[kind]["v1_total_params"] = int(V1_TOTAL_PARAMS[kind])
            rows[kind]["matches_v1_total_params"] = bool(
                int(rows[kind]["total_params"]) == int(V1_TOTAL_PARAMS[kind])
            )
    candidate_added = [max(int(rows[kind]["added_params_vs_cap_base"]), 1) for kind in ("F", "R", "G")]
    ratio = float(max(candidate_added)) / float(min(candidate_added))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "cap_base": cv2.CAP_BASE,
        "cap_base_params": base_total,
        "budget_preferred": list(cv2.BUDGET_PREFERRED),
        "budget_hard_ceiling": int(cv2.BUDGET_HARD_CEILING),
        "budget_ratio_max": float(cv2.BUDGET_RATIO_MAX),
        "rows": rows,
        "candidate_added_params": {
            kind: int(rows[kind]["added_params_vs_cap_base"]) for kind in ("F", "R", "G")
        },
        "added_params_ratio": ratio,
        "added_params_ratio_ok": bool(ratio <= cv2.BUDGET_RATIO_MAX),
        "all_candidates_match_v1": bool(
            all(bool(rows[kind]["matches_v1_added_params"]) for kind in ("F", "R", "G"))
        ),
        "official_test_loaded": False,
    }


def architecture_fingerprints(subspace: cssd.CommonSubspace, dictionary: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for kind in cv2.KINDS:
        model = cv2.build_v2_model(dictionary, 0, subspace, kind)
        rows[kind] = cv2.architecture_fingerprint(kind, model)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    cv2.cpu_only_guard(torch.device("cpu"))
    if not PREREG_PATH.exists():
        raise FileNotFoundError(f"preregistration missing: {PREREG_PATH}")
    for path in (CSSD_SOUP_PATH, CSSD_SUBSPACE_PATH, CSSD_CURVE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"frozen CAP-BASE artifact missing: {path}")
    subspace = _load_subspace()
    if int(subspace.q) != 1:
        raise RuntimeError(f"CAP-BASE subspace is q{int(subspace.q)}, expected q1")
    soup = _load_soup()
    if "D" not in soup or "U" not in soup:
        raise RuntimeError("CAP-BASE soup checkpoint is missing dictionary/subspace buffers")
    dictionary = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0]

    # re-verify the frozen CAP-BASE soup MAE on the official-valid split
    model = cv2.build_v2_model(dictionary, 0, subspace, "M0")
    model.load_state_dict(soup)
    valid = p1run.load_split("valid")
    if not valid:
        raise RuntimeError("official-valid data not available")
    loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    device = audit.attach_cpu(int(THREADS))
    soup_mae = cv2.evaluate_mae(model, loader, device, cssd.CSSD_MASK)

    budget = parameter_budget_v2(subspace, dictionary)
    fingerprints = architecture_fingerprints(subspace, dictionary)
    prereg = {
        "protocol_version": PROTOCOL_VERSION,
        "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(PREREG_PATH),
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": int(THREADS),
        "official_test_loaded": False,
        "preregistration": prereg,
        "cap_base": {
            "name": cv2.CAP_BASE,
            "soup_checkpoint": str(CSSD_SOUP_PATH.relative_to(REPO_ROOT)),
            "soup_checkpoint_sha256": _sha256_file(CSSD_SOUP_PATH),
            "frozen_soup_mae": cv2.CAP_BASE_SOUP_MAE,
            "recomputed_soup_mae": float(soup_mae),
            "mae_reproduced": bool(abs(float(soup_mae) - cv2.CAP_BASE_SOUP_MAE) < 5e-7),
            "subspace_q": int(subspace.q),
            "subspace_path": str(CSSD_SUBSPACE_PATH.relative_to(REPO_ROOT)),
            "subspace_sha256": _sha256_file(CSSD_SUBSPACE_PATH),
        },
        "historical_reference": {
            "final_clean_sparse_seed0_soup_mae": cv2.FINAL_CLEAN_SPARSE_SOUP_MAE,
            "note": "historical performance reference only; never a matched control",
        },
        "data": {
            "train_cache": str(p1run._env_cache_path("train").relative_to(REPO_ROOT)),
            "valid_cache": str(p1run._env_cache_path("valid").relative_to(REPO_ROOT)),
        },
        "warm_adaptation": {
            "calibration": {
                "epochs": cv2.CALIBRATION_EPOCHS,
                "lr": cv2.CALIBRATION_LR,
                "soup_window": list(cv2.CALIB_SOUP_WINDOW),
                "soup_k": cv2.CALIB_SOUP_K,
                "last_k": cv2.CALIB_LAST_K,
                "late_window": list(cv2.CALIB_LATE_WINDOW),
                "gates": {
                    "C1_soup_max": cv2.CALIB_SOUP_MAX,
                    "C2_late_mean_max": cv2.CALIB_LATE_MEAN_MAX,
                    "C3_epoch20_max": cv2.CALIB_EPOCH20_MAX,
                    "C3_late_slope_max": cv2.CALIB_LATE_SLOPE_MAX,
                },
            },
            "screen": {
                "epochs": cv2.SCREEN_EPOCHS,
                "base_lr": cv2.BASE_LR,
                "new_lr": cv2.NEW_LR,
                "weight_decay": cv2.WEIGHT_DECAY,
                "grad_clip": cv2.GRAD_CLIP,
                "soup_window": list(cv2.SCREEN_SOUP_WINDOW),
                "soup_k": cv2.SCREEN_SOUP_K,
                "last_k": cv2.SCREEN_LAST10,
                "gates": {
                    "S1_delta_vs_M0": cv2.S1_DELTA_VS_M0,
                    "S2_abs_max": cv2.S2_ABS_MAX,
                    "S3_delta_last10_vs_M0": cv2.S3_DELTA_LAST10_VS_M0,
                    "S4_branch_usage": "finite > 0 gradient and update on the new capacity group",
                    "strong_screen_max": cv2.STRONG_SCREEN_MAX,
                },
                "tie_tolerance": cv2.TIE_TOLERANCE,
                "tie_order": list(cv2.TIE_ORDER),
            },
            "init_contract": {
                "hard_mean_shift_max": cv2.INIT_SHIFT_HARD_MAX,
                "preferred_mean_shift_max": cv2.INIT_SHIFT_PREFERRED_MAX,
                "fusion_residual_scale": cv2.FUSION_RESIDUAL_SCALE,
                "relation_residual_scale": cv2.RELATION_RESIDUAL_SCALE,
                "readout_summary_scale": cv2.READOUT_SUMMARY_SCALE,
                "v1_step0_mean_shift_reference": dict(V1_STEP0_MEAN_SHIFT),
            },
        },
        "full_run": {"epochs": cv2.FULL_EPOCHS, "seed": cv2.FULL_SEED, "lr": cv2.FULL_LR},
        "candidates": {
            kind: cl.CAPACITY_SPECS[kind].as_dict() for kind in cv2.KINDS
        },
        "parameter_budget": budget,
        "architecture_fingerprints": fingerprints,
    }
    _write_json(RESULTS_DIR / "preregistration_snapshot.json", prereg)
    _write_json(RESULTS_DIR / "parameter_budget.json", budget)
    _write_json(RESULTS_DIR / "architecture_fingerprints.json", fingerprints)
    _write_json(RESULTS_DIR / "preflight.json", _payload(payload))
    if not payload["cap_base"]["mae_reproduced"]:
        raise RuntimeError("CAP-BASE soup MAE was not reproduced")
    if not budget["all_candidates_match_v1"]:
        raise RuntimeError("v2 candidate parameter counts do not match v1")
    print(
        f"[preflight] prereg={prereg['sha256'][:12]} soup={soup_mae:.6f} "
        f"added={budget['candidate_added_params']} ratio={budget['added_params_ratio']:.4f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: initialisation audit (frozen record, no training)
# ---------------------------------------------------------------------------


def stage_init_audit() -> dict[str, Any]:
    _ensure_dirs()
    cv2.cpu_only_guard(torch.device("cpu"))
    subspace = _load_subspace()
    soup = _load_soup()
    dictionary = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0]
    train = p1run.load_split("train", subset=64)
    valid = p1run.load_split("valid")
    if not train or not valid:
        raise RuntimeError("data not available")
    device = audit.attach_cpu(int(THREADS))
    eval_loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    grad_batch = next(iter(p1.make_env_loader(train, p2run.BATCH_SIZE, False, 0)))

    base = cv2.build_v2_model(dictionary, 0, subspace, "M0")
    base.load_state_dict(soup)
    base_prediction = cv2.predictions_for(base, eval_loader, device, cssd.CSSD_MASK)
    targets = np.concatenate([batch.y.numpy() for batch in eval_loader]).reshape(-1)
    base_mae = float(np.mean(np.abs(base_prediction - targets)))

    rows: dict[str, Any] = {}
    for kind in cv2.KINDS:
        model = cv2.build_v2_model(dictionary, 0, subspace, kind)
        report = cv2.load_capacity_warm_state(model, soup)
        mae = cv2.evaluate_mae(model, eval_loader, device, cssd.CSSD_MASK)
        shift = cv2.prediction_shift(model, eval_loader, device, base_prediction)
        candidate_prediction = cv2.predictions_for(model, eval_loader, device, cssd.CSSD_MASK)
        delta = np.abs(candidate_prediction - base_prediction)
        if kind == "M0":
            norms: dict[str, float] = {}
        else:
            norms = cv2.new_module_gradient_norms(model, grad_batch, device)
        finite = all(np.isfinite(value) for value in norms.values())
        positive = all(value > 0.0 for value in norms.values())
        # residual-zero identity: zeroing the residual projection must restore
        # the CAP-BASE prediction exactly (the augmentation is additive).
        model.zero_residual()
        zero_prediction = cv2.predictions_for(model, eval_loader, device, cssd.CSSD_MASK)
        identity_delta = float(np.abs(zero_prediction - base_prediction).max()) if kind != "M0" else 0.0
        rows[kind] = {
            "kind": kind,
            "params": int(sum(parameter.numel() for parameter in model.parameters())),
            "added_params": int(model.capacity_parameter_count()),
            "shared_state_keys": int(report["shared_keys"]),
            "shared_state_bit_identical": bool(report["bit_identical"]),
            "step0_valid_mae": float(mae),
            "step0_mae_delta_vs_base": float(mae) - base_mae,
            "step0_mean_abs_prediction_delta": float(shift["mean_abs_prediction_delta"]),
            "step0_median_abs_prediction_delta": float(np.median(delta)),
            "step0_max_abs_prediction_delta": float(shift["max_abs_prediction_delta"]),
            "residual_zero_max_abs_prediction_delta": identity_delta,
            "residual_zero_identity": bool(identity_delta <= 1e-6),
            "new_module_gradient_norms": norms,
            "new_module_gradients_finite": finite,
            "new_module_gradients_positive": positive,
            "init_mean_shift_hard_ok": bool(
                float(shift["mean_abs_prediction_delta"]) <= cv2.INIT_SHIFT_HARD_MAX
            ),
            "init_mean_shift_preferred_ok": bool(
                float(shift["mean_abs_prediction_delta"]) <= cv2.INIT_SHIFT_PREFERRED_MAX
            ),
            "audit_pass": bool(
                report["bit_identical"]
                and float(shift["mean_abs_prediction_delta"]) <= cv2.INIT_SHIFT_HARD_MAX
                and finite
                and positive
                and (kind == "M0" or identity_delta <= 1e-6)
            ),
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "base_valid_mae": base_mae,
        "rows": rows,
        "all_candidates_pass": bool(all(rows[kind]["audit_pass"] for kind in cv2.KINDS)),
        "init_contract": {
            "hard_mean_shift_max": cv2.INIT_SHIFT_HARD_MAX,
            "preferred_mean_shift_max": cv2.INIT_SHIFT_PREFERRED_MAX,
            "fusion_residual_scale": cv2.FUSION_RESIDUAL_SCALE,
            "relation_residual_scale": cv2.RELATION_RESIDUAL_SCALE,
            "readout_summary_scale": cv2.READOUT_SUMMARY_SCALE,
        },
    }
    _write_json(RESULTS_DIR / "init_audit.json", _payload(payload))
    print(
        f"[init-audit] base={base_mae:.6f} "
        + " ".join(
            f"{kind}:{rows[kind]['step0_valid_mae']:.6f}/shift={rows[kind]['step0_mean_abs_prediction_delta']:.5f}"
            for kind in cv2.KINDS
        ),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: Phase A — M0 calibration
# ---------------------------------------------------------------------------


def stage_calibration(threads: int = THREADS) -> dict[str, Any]:
    _ensure_dirs()
    result_path = CALIBRATION_M0_DIR / "result.json"
    if result_path.exists():
        print("[calibration] cache hit", flush=True)
        return _read_json(result_path)
    cv2.cpu_only_guard(torch.device("cpu"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    subspace = _load_subspace()
    soup = _load_soup()
    dictionary = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0]
    train = p1run.load_split("train")
    valid = p1run.load_split("valid")
    if not train or not valid:
        raise RuntimeError("data not available")

    p2run._seed_everything(cv2.FULL_SEED)
    model = cv2.build_v2_model(dictionary, cv2.FULL_SEED, subspace, "M0")
    warm_report = cv2.load_capacity_warm_state(model, soup)
    device = audit.attach_cpu(int(threads))
    eval_loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    step0_mae = cv2.evaluate_mae(model, eval_loader, device, cssd.CSSD_MASK)

    payload = cv2.train_warm(
        tag="CALIB-M0",
        model=model,
        dictionary=dictionary,
        subspace=subspace,
        epochs=cv2.CALIBRATION_EPOCHS,
        threads=int(threads),
        train_data=train,
        valid_data=valid,
        seed=cv2.FULL_SEED,
        base_lr=cv2.CALIBRATION_LR,
        new_lr=cv2.CALIBRATION_LR,
        soup_window=cv2.CALIB_SOUP_WINDOW,
        soup_k=cv2.CALIB_SOUP_K,
        last_k=cv2.CALIB_LAST_K,
        log=True,
        return_soup_state=True,
    )
    summary = cv2.calibration_summary(payload)
    payload.update(
        {
            "git_commit": _git_commit(),
            "arm": "M0",
            "phase": "calibration",
            "warm_start": str(CSSD_SOUP_PATH.relative_to(REPO_ROOT)),
            "warm_state_shared_keys": int(warm_report["shared_keys"]),
            "warm_state_bit_identical": bool(warm_report["bit_identical"]),
            "step0_valid_mae": float(step0_mae),
            "step0_mae_delta_vs_start": float(step0_mae) - float(cv2.CAP_BASE_SOUP_MAE),
            "params": int(sum(parameter.numel() for parameter in model.parameters())),
            "added_params": 0,
            "lr": float(cv2.CALIBRATION_LR),
        }
    )
    curve = payload.pop("curve")
    _write_csv(
        CALIBRATION_M0_DIR / "curve.csv",
        curve,
        [
            "epoch",
            "train_mae",
            "train_rec",
            "train_rec_term",
            "train_total_loss",
            "valid_mae",
            "d_norm",
            "grad_norm_base",
            "grad_norm_new",
            "grad_norm_total",
            "update_norm_base",
            "update_norm_new",
            "update_norm_total",
            "seconds",
        ],
    )
    soup_state = payload.pop("soup_state")
    _write_json(
        CALIBRATION_M0_DIR / "soup.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "kind": "M0",
            "members": payload["soup_members"],
            "member_valid_mae": payload["soup_member_valid_mae"],
            "soup_valid_mae": payload["soup_valid_mae"],
        },
    )
    torch.save(soup_state, CALIBRATION_M0_DIR / "soup_state.pt")
    _write_json(result_path, _payload(payload))
    _write_json(CALIBRATION_DIR / "calibration_summary.json", _payload(summary))
    decision = {
        **_payload(summary),
        "git_commit": _git_commit(),
        "result": str(result_path.relative_to(REPO_ROOT)),
        "curve": str((CALIBRATION_M0_DIR / "curve.csv").relative_to(REPO_ROOT)),
        "soup": str((CALIBRATION_M0_DIR / "soup.json").relative_to(REPO_ROOT)),
        "step0_valid_mae": float(step0_mae),
        "warm_state_bit_identical": bool(warm_report["bit_identical"]),
    }
    _write_json(CALIBRATION_DIR / "calibration_decision.json", _payload(decision))
    if not summary["overall_pass"]:
        _write_json(
            SCREENING_DIR / "NOT_RUN.json",
            _payload(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "git_commit": _git_commit(),
                    "official_test_loaded": False,
                    "run": False,
                    "reason": "WARM_ADAPTATION_PROTOCOL_UNSTABLE",
                    "calibration_decision": str(
                        (CALIBRATION_DIR / "calibration_decision.json").relative_to(REPO_ROOT)
                    ),
                }
            ),
        )
        _write_json(
            FULL_DIR / "NOT_RUN.json",
            _payload(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "git_commit": _git_commit(),
                    "official_test_loaded": False,
                    "run": False,
                    "reason": "WARM_ADAPTATION_PROTOCOL_UNSTABLE",
                }
            ),
        )
        _write_json(
            MECHANISM_DIR / "probes.json",
            _payload(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "official_test_loaded": False,
                    "run": False,
                    "reason": "WARM_ADAPTATION_PROTOCOL_UNSTABLE",
                }
            ),
        )
    print(
        f"[calibration] soup={summary['calibration_soup']:.6f} last5={summary['calibration_last5']:.6f} "
        f"epoch20={summary['epoch20']:.6f} slope={summary['late_slope']:+.2e} "
        f"C1={summary['C1_pass']} C2={summary['C2_pass']} C3={summary['C3_pass']} "
        f"-> {summary['verdict']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: Phase B — one screening arm (subprocess entry)
# ---------------------------------------------------------------------------


def stage_screen_arm(kind: str, threads: int = THREADS) -> dict[str, Any]:
    kind = str(kind).upper()
    if kind not in cv2.KINDS:
        raise ValueError(f"unknown arm {kind!r}")
    if not _calibration_passed():
        raise RuntimeError(
            "calibration gate did not pass (WARM_ADAPTATION_PROTOCOL_UNSTABLE); "
            "candidate arms are not authorised"
        )
    out_dir = ARM_DIRS[kind]
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"[screen-{kind}] cache hit", flush=True)
        return _read_json(result_path)

    cv2.cpu_only_guard(torch.device("cpu"))
    cv2.official_test_blocker({"official_test_loaded": False})
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    subspace = _load_subspace()
    soup = _load_soup()
    dictionary = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0]
    train = p1run.load_split("train")
    valid = p1run.load_split("valid")
    if not train or not valid:
        raise RuntimeError("data not available")

    p2run._seed_everything(cv2.FULL_SEED)
    model = cv2.build_v2_model(dictionary, cv2.FULL_SEED, subspace, kind)
    warm_report = cv2.load_capacity_warm_state(model, soup)

    device = audit.attach_cpu(int(threads))
    eval_loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    base = cv2.build_v2_model(dictionary, cv2.FULL_SEED, subspace, "M0")
    base.load_state_dict(soup)
    base_prediction = cv2.predictions_for(base, eval_loader, device, cssd.CSSD_MASK)
    targets = np.concatenate([batch.y.numpy() for batch in eval_loader]).reshape(-1)
    base_mae = float(np.mean(np.abs(base_prediction - targets)))
    step0_mae = cv2.evaluate_mae(model, eval_loader, device, cssd.CSSD_MASK)
    candidate_prediction = cv2.predictions_for(model, eval_loader, device, cssd.CSSD_MASK)
    shift_delta = np.abs(candidate_prediction - base_prediction)
    grad_batch = next(iter(p1.make_env_loader(train, p2run.BATCH_SIZE, False, 0)))
    grad_norms = cv2.new_module_gradient_norms(model, grad_batch, device)

    payload = cv2.train_warm(
        tag=f"SCREEN-{ARM_TAGS[kind]}",
        model=model,
        dictionary=dictionary,
        subspace=subspace,
        epochs=cv2.SCREEN_EPOCHS,
        threads=int(threads),
        train_data=train,
        valid_data=valid,
        seed=cv2.FULL_SEED,
        base_lr=cv2.BASE_LR,
        new_lr=cv2.NEW_LR,
        soup_window=cv2.SCREEN_SOUP_WINDOW,
        soup_k=cv2.SCREEN_SOUP_K,
        last_k=cv2.SCREEN_LAST10,
        log=True,
        return_soup_state=True,
    )
    payload.update(
        {
            "git_commit": _git_commit(),
            "arm": kind,
            "phase": "screen",
            "warm_start": str(CSSD_SOUP_PATH.relative_to(REPO_ROOT)),
            "warm_state_shared_keys": int(warm_report["shared_keys"]),
            "warm_state_bit_identical": bool(warm_report["bit_identical"]),
            "step0_base_valid_mae": base_mae,
            "step0_valid_mae": float(step0_mae),
            "step0_mae_delta_vs_base": float(step0_mae) - base_mae,
            "step0_mean_abs_prediction_delta": float(shift_delta.mean()),
            "step0_median_abs_prediction_delta": float(np.median(shift_delta)),
            "step0_max_abs_prediction_delta": float(shift_delta.max()),
            "new_module_gradient_norms": grad_norms,
            "new_module_gradient_norm_max": float(max(grad_norms.values())) if grad_norms else 0.0,
            "params": int(sum(parameter.numel() for parameter in model.parameters())),
            "added_params": int(model.capacity_parameter_count()),
        }
    )
    curve = payload.pop("curve")
    _write_csv(
        out_dir / "curve.csv",
        curve,
        [
            "epoch",
            "train_mae",
            "train_rec",
            "train_rec_term",
            "train_total_loss",
            "valid_mae",
            "d_norm",
            "grad_norm_base",
            "grad_norm_new",
            "grad_norm_total",
            "update_norm_base",
            "update_norm_new",
            "update_norm_total",
            "seconds",
        ],
    )
    soup_state = payload.pop("soup_state")
    _write_json(
        out_dir / "soup.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "kind": kind,
            "members": payload["soup_members"],
            "member_valid_mae": payload["soup_member_valid_mae"],
            "soup_valid_mae": payload["soup_valid_mae"],
        },
    )
    torch.save(soup_state, out_dir / "soup_state.pt")
    _write_json(result_path, _payload(payload))
    print(
        f"[screen-{kind}] soup={payload['soup_valid_mae']:.6f} "
        f"last10={payload['last_mean_valid_mae']:.6f} best={payload['best_valid_mae']:.6f} "
        f"branch_grad={payload['branch_grad_max']:.3e} wall={payload['wall_clock_s']:.0f}s",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# launch helpers
# ---------------------------------------------------------------------------


def _spawn(
    args: Sequence[str], threads: int, env_extra: Mapping[str, str] | None = None
) -> subprocess.Popen:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(int(threads))
    env["MKL_NUM_THREADS"] = str(int(threads))
    env["CUDA_VISIBLE_DEVICES"] = ""
    if env_extra:
        env.update(env_extra)
    command = [
        sys.executable,
        "-m",
        "tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_capacity_localization_v2",
        *args,
    ]
    print(f"[spawn] {' '.join(args)} (threads={threads})", flush=True)
    return subprocess.Popen(command, cwd=str(REPO_ROOT), env=env)


def launch_wave(kinds: Sequence[str], threads: int, concurrency: int) -> None:
    queue = [str(kind) for kind in kinds]
    running: list[tuple[str, subprocess.Popen]] = []
    concurrency = max(1, min(int(concurrency), len(queue)))
    while queue or running:
        while queue and len(running) < concurrency:
            kind = queue.pop(0)
            if (ARM_DIRS[kind] / "result.json").exists():
                print(f"[wave] skip {kind} (cached)", flush=True)
                continue
            running.append(
                (kind, _spawn(["screen-arm", "--kind", kind, "--threads", str(int(threads))], threads))
            )
        time.sleep(2.0)
        for item in list(running):
            kind, process = item
            if process.poll() is not None:
                running.remove(item)
                if process.returncode != 0:
                    raise RuntimeError(f"screening arm {kind} failed with code {process.returncode}")


# ---------------------------------------------------------------------------
# stage: summarize + frozen decision
# ---------------------------------------------------------------------------


def _arm_row(kind: str) -> dict[str, Any]:
    return _read_json(ARM_DIRS[kind] / "result.json")


def _calibration_vs_screen_m0() -> dict[str, Any]:
    """Cross-check that the M0 calibration run and the M0 screen arm agree."""
    calibration_path = CALIBRATION_M0_DIR / "curve.csv"
    screen_path = ARM_DIRS["M0"] / "curve.csv"
    if not (calibration_path.exists() and screen_path.exists()):
        return {"available": False}
    calibration_rows = {int(row["epoch"]): row for row in _read_csv_rows(calibration_path)}
    screen_rows = {int(row["epoch"]): row for row in _read_csv_rows(screen_path)}
    shared = sorted(set(calibration_rows) & set(screen_rows))
    valid_deltas = [
        abs(float(calibration_rows[epoch]["valid_mae"]) - float(screen_rows[epoch]["valid_mae"]))
        for epoch in shared
    ]
    train_deltas = [
        abs(float(calibration_rows[epoch]["train_mae"]) - float(screen_rows[epoch]["train_mae"]))
        for epoch in shared
    ]
    return {
        "available": True,
        "epochs": len(shared),
        "max_abs_valid_mae_delta": float(max(valid_deltas)) if valid_deltas else None,
        "max_abs_train_mae_delta": float(max(train_deltas)) if train_deltas else None,
        "bit_reproduced": bool(valid_deltas and max(valid_deltas) == 0.0 and max(train_deltas) == 0.0),
    }


def stage_summarize() -> dict[str, Any]:
    _ensure_dirs()
    calibration = _calibration_decision()
    if not calibration:
        raise RuntimeError("calibration decision missing; run the calibration stage first")
    if not calibration.get("overall_pass"):
        payload = _read_json(SCREENING_DIR / "NOT_RUN.json")
        print("[summarize] calibration failed -> screen NOT_RUN", flush=True)
        return payload

    rows = {kind: _arm_row(kind) for kind in cv2.KINDS}
    deltas = cv2.screening_deltas(rows)
    gates = {kind: cv2.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    selection = cv2.select_winner(rows, gates)
    budget = _read_json(RESULTS_DIR / "parameter_budget.json")

    table: list[dict[str, Any]] = []
    for kind in ("M0", "F", "R", "G"):
        row = rows[kind]
        entry: dict[str, Any] = {
            "candidate": kind,
            "params": row["params"],
            "added_params": row["added_params"],
            "base_lr": row["optimizer_groups"]["base_lr"],
            "new_lr": row["optimizer_groups"]["new_lr"],
            "step0_mae": row["step0_valid_mae"],
            "step0_mean_shift": row["step0_mean_abs_prediction_delta"],
            "step0_max_shift": row["step0_max_abs_prediction_delta"],
            "best_valid": row["best_valid_mae"],
            "best_epoch": row["best_epoch"],
            "screen_soup": row["soup_valid_mae"],
            "soup_members": row["soup_members"],
            "last10": row["last_mean_valid_mae"],
            "epoch40": row["last_valid_mae"],
            "branch_grad": row["branch_grad_max"],
            "branch_update": row["branch_update_max"],
            "seconds_per_epoch": row["seconds_per_epoch"],
            "wall_clock_s": row["wall_clock_s"],
        }
        if kind == "M0":
            entry.update(
                {
                    "delta_soup_vs_M0": 0.0,
                    "delta_vs_start": float(row["soup_valid_mae"]) - float(cv2.CAP_BASE_SOUP_MAE),
                    "delta_last10_vs_M0": 0.0,
                    "S1": "control",
                    "S2": "control",
                    "S3": "control",
                    "S4": "control",
                    "gate_pass": "control",
                    "verdict": "CONTROL",
                }
            )
        else:
            gate = gates[kind]
            entry.update(
                {
                    "delta_soup_vs_M0": deltas[kind]["delta_soup_vs_M0"],
                    "delta_vs_start": deltas[kind]["delta_soup_vs_start"],
                    "delta_last10_vs_M0": deltas[kind]["delta_last10_vs_M0"],
                    "S1": gate["S1_matched_control"],
                    "S2": gate["S2_absolute"],
                    "S3": gate["S3_late_window"],
                    "S4": gate["S4_branch_usage"],
                    "gate_pass": bool(gate["passed"]),
                    "verdict": gate["verdict"],
                }
            )
        table.append(entry)
    header = [
        "candidate",
        "params",
        "added_params",
        "base_lr",
        "new_lr",
        "step0_mae",
        "step0_mean_shift",
        "step0_max_shift",
        "best_valid",
        "best_epoch",
        "screen_soup",
        "soup_members",
        "last10",
        "epoch40",
        "delta_soup_vs_M0",
        "delta_vs_start",
        "delta_last10_vs_M0",
        "branch_grad",
        "branch_update",
        "S1",
        "S2",
        "S3",
        "S4",
        "gate_pass",
        "verdict",
        "seconds_per_epoch",
        "wall_clock_s",
    ]
    _write_csv(SCREENING_DIR / "screening_summary.csv", table, header)
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "calibration": calibration,
        "calibration_vs_screen_m0": _calibration_vs_screen_m0(),
        "control": "M0",
        "control_soup": rows["M0"]["soup_valid_mae"],
        "control_last10": rows["M0"]["last_mean_valid_mae"],
        "start_mae": float(cv2.CAP_BASE_SOUP_MAE),
        "rows": table,
        "deltas": deltas,
        "gates": gates,
        "selection": selection,
        "parameter_budget": {
            "cap_base_params": budget["cap_base_params"],
            "candidate_added_params": budget["candidate_added_params"],
            "added_params_ratio": budget["added_params_ratio"],
        },
    }
    _write_json(SCREENING_DIR / "decision.json", _payload(decision))
    print(
        f"[summarize] M0={rows['M0']['soup_valid_mae']:.6f} "
        + " ".join(
            f"{kind}:{deltas[kind]['delta_soup_vs_M0']:+.6f}/{gates[kind]['verdict']}"
            for kind in ("F", "R", "G")
        )
        + f" -> winner={selection['winner']}",
        flush=True,
    )
    return decision


# ---------------------------------------------------------------------------
# stage: Phase C — the single full winner run
# ---------------------------------------------------------------------------


def _cssd_reference_curve() -> list[float]:
    rows = _read_csv_rows(CSSD_CURVE_PATH)
    rows.sort(key=lambda row: int(row["epoch"]))
    return [float(row["valid_mae"]) for row in rows]


def _full_early_stop_callback(reference: Sequence[float], capacity_names: Sequence[str]):
    state = {"worse_streak": 0, "window": []}

    def callback(epoch: int, model: Any, optimizer: Any, curve: list[dict[str, Any]]):
        row = curve[-1]
        valid = float(row["valid_mae"])
        payload: dict[str, Any] = {"epoch": int(epoch), "valid_mae": valid}
        if not np.isfinite(valid) or not np.isfinite(float(row["train_mae"])):
            payload["reason"] = "non_finite"
            return False, payload
        if valid > float(cssd.CATASTROPHIC_MAE):
            payload["reason"] = "divergence"
            return False, payload
        if not bool(torch.isfinite(model.D).all()):
            payload["reason"] = "dictionary_non_finite"
            return False, payload
        if int(epoch) % 20 == 0:
            parameters = dict(model.named_parameters())
            norms = {
                name: float(parameters[name].grad.detach().norm())
                for name in capacity_names
                if parameters[name].grad is not None
            }
            payload["capacity_grad_norms"] = norms
            if norms and all(value == 0.0 for value in norms.values()):
                payload["reason"] = "capacity_gradient_zero"
                return False, payload
        if int(epoch) <= len(reference):
            delta = valid - float(reference[int(epoch) - 1])
            payload["delta_vs_cssd_curve"] = delta
            if delta > 0.03:
                state["worse_streak"] += 1
            else:
                state["worse_streak"] = 0
            state["window"].append(valid)
            if len(state["window"]) > 20:
                state["window"].pop(0)
            improving = True
            if len(state["window"]) == 20:
                last10 = min(state["window"][-10:])
                previous10 = min(state["window"][:10])
                improving = last10 < previous10
            payload["worse_streak"] = state["worse_streak"]
            payload["improving_trend"] = improving
            if state["worse_streak"] >= 40 and not improving:
                payload["reason"] = "matched_cssd_curve_hopeless"
                return False, payload
        return True, payload

    return callback


def stage_full(threads: int = FULL_THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    final_path = FULL_WINNER_DIR / "final.json"
    if final_path.exists() and not force:
        print("[full] cache hit", flush=True)
        return _read_json(final_path)
    decision = _read_json(SCREENING_DIR / "decision.json")
    winner = decision.get("selection", {}).get("winner")
    if winner is None:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "run": False,
            "reason": "LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED",
            "selection": decision.get("selection"),
        }
        _write_json(FULL_DIR / "NOT_RUN.json", _payload(payload))
        print("[full] no winner -> NOT_RUN", flush=True)
        return payload

    cv2.cpu_only_guard(torch.device("cpu"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    subspace = _load_subspace()
    train = p1run.load_split("train")
    valid = p1run.load_split("valid")
    if not train or not valid:
        raise RuntimeError("data not available")

    def factory(dictionary: np.ndarray, seed: int, subspace_: cssd.CommonSubspace) -> cl.CapacityModel:
        return cv2.build_v2_model(dictionary, seed, subspace_, winner)

    reference = _cssd_reference_curve()
    probe = cv2.build_v2_model(
        p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0], 0, subspace, winner
    )
    capacity_names = probe.capacity_parameter_names()
    callback = _full_early_stop_callback(reference, capacity_names)
    tag = f"CAP-{winner}-seed{cv2.FULL_SEED}"
    started = time.perf_counter()
    payload = cssd.train_cssd(
        tag=tag,
        epochs=cv2.FULL_EPOCHS,
        threads=int(threads),
        out_dir=FULL_CHECKPOINT_DIR,
        subspace=subspace,
        train_data=train,
        valid_data=valid,
        seed=cv2.FULL_SEED,
        callback=callback,
        model_factory=factory,
        save_states=True,
        log=True,
    )
    payload.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "winner": winner,
            "seed": cv2.FULL_SEED,
            "from_scratch": True,
            "selection": decision.get("selection"),
            "params": int(sum(parameter.numel() for parameter in probe.parameters())),
            "cap_base_params": int(_read_json(RESULTS_DIR / "parameter_budget.json")["cap_base_params"]),
            "round_seconds": float(time.perf_counter() - started),
        }
    )
    payload["interpretation"] = cv2.full_interpretation(float(payload["soup"]["soup_valid_mae"]))
    curve = payload.pop("curve")
    _write_csv(
        FULL_WINNER_DIR / "curve.csv",
        curve,
        ["epoch", "train_mae", "train_rec", "train_rec_term", "valid_mae", "d_norm"],
    )
    _write_json(
        FULL_WINNER_DIR / "soup.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "winner": winner,
            "members": payload["soup"]["members"],
            "member_valid_mae": payload["soup"]["member_valid_mae"],
            "soup_valid_mae": payload["soup"]["soup_valid_mae"],
            "interpretation": payload["interpretation"],
        },
    )
    _write_json(final_path, _payload(payload))
    print(
        f"[full] winner={winner} completed={payload['completed']} epochs={payload['epochs_run']} "
        f"soup={payload['soup']['soup_valid_mae']:.6f} band={payload['interpretation']['band']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: mechanism audit (only when the full run exists)
# ---------------------------------------------------------------------------


def _predictions_with_phi_shuffle(
    model: cl.CapacityModel, loader: Any, device: torch.device, mask: audit.AuditMask, seed: int
) -> np.ndarray:
    generator = torch.Generator().manual_seed(int(seed))
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            permutation = torch.randperm(int(batch.dict_phi.shape[0]), generator=generator)
            batch.dict_phi = batch.dict_phi[permutation]
            predictions.append(model(batch, mask=mask).view(-1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.zeros((0,))


def _probe_delta(
    base_mae: float,
    probe_mae: float,
    base_prediction: np.ndarray | None,
    probe_prediction: np.ndarray | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"mae": float(probe_mae), "delta_mae": float(probe_mae) - float(base_mae)}
    if base_prediction is not None and probe_prediction is not None:
        shift = np.abs(probe_prediction - base_prediction)
        payload["mean_abs_prediction_delta"] = float(shift.mean())
        payload["max_abs_prediction_delta"] = float(shift.max())
    return payload


def stage_mechanism(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out_path = MECHANISM_DIR / "probes.json"
    if out_path.exists() and not force:
        print("[mechanism] cache hit", flush=True)
        return _read_json(out_path)
    final_path = FULL_WINNER_DIR / "final.json"
    final = _read_json(final_path) if final_path.exists() else None
    if final is None or not final.get("winner"):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "run": False,
            "reason": "full winner run not available",
        }
        _write_json(out_path, _payload(payload))
        return payload

    winner = str(final["winner"])
    subspace = _load_subspace()
    state = torch.load(
        FULL_CHECKPOINT_DIR / f"CAP-{winner}-seed{cv2.FULL_SEED}_soup_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = cv2.build_v2_model(
        p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0], cv2.FULL_SEED, subspace, winner
    )
    model.load_state_dict(state)
    valid = p1run.load_split("valid")
    if not valid:
        raise RuntimeError("valid data not available")
    train_diag = p1run.load_split("train", subset=p2run.BATCH_SIZE)
    if not train_diag:
        raise RuntimeError("train data not available")
    grad_batch = next(iter(p1.make_env_loader(train_diag, p2run.BATCH_SIZE, False, 0)))
    device = audit.attach_cpu(int(threads))
    mask = cssd.CSSD_MASK
    loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    base_prediction = cv2.predictions_for(model, loader, device, mask)
    targets = np.concatenate([batch.y.numpy() for batch in loader]).reshape(-1)
    base_mae = float(np.mean(np.abs(base_prediction - targets)))

    rows: list[dict[str, Any]] = []
    # P1 — dictionary coordinate row shuffle (distribution preserving)
    for seed in PROBE_SEEDS:
        prediction = _predictions_with_phi_shuffle(model, loader, device, mask, seed)
        rows.append(
            {
                "probe": "P1_phi_row_shuffle",
                "probe_group": "all",
                "seed": int(seed),
                **_probe_delta(
                    base_mae, float(np.mean(np.abs(prediction - targets))), base_prediction, prediction
                ),
            }
        )
    # P2 / P3 — assignment-preserving node / edge correspondence shuffles
    for name, kind, field in (
        ("P2_node_correspondence", "node", "node"),
        ("P3_edge_correspondence", "edge", "edge"),
    ):
        probe_mask = cm.merge_masks(mask, audit.AuditMask(**{f"use_{field}_shuffle": True}))
        for seed in PROBE_SEEDS:
            audit.prepare_shuffles(valid, kind, (int(seed), int(seed) + 1))
            try:
                result = audit.evaluate_mask(model, loader, device, probe_mask)
            finally:
                audit.clear_shuffles(valid)
            rows.append(
                {
                    "probe": name,
                    "probe_group": kind,
                    "seed": int(seed),
                    **_probe_delta(
                        base_mae, float(result["mae"]), base_prediction, np.asarray(result["predictions"])
                    ),
                }
            )
    # P4 — distribution-preserving relation row shuffle
    for group in ("all", "distance", "overlap", "boundary"):
        for seed in PROBE_SEEDS:
            restore = audit.permute_pair_rows(valid, group, seed)
            try:
                result = audit.evaluate_mask(model, loader, device, mask)
            finally:
                restore()
            rows.append(
                {
                    "probe": "P4_relation",
                    "probe_group": group,
                    "seed": int(seed),
                    **_probe_delta(
                        base_mae, float(result["mae"]), base_prediction, np.asarray(result["predictions"])
                    ),
                }
            )
    # P5 — winner branch disable
    model.capacity_off = True
    disabled_prediction = cv2.predictions_for(model, loader, device, mask)
    model.capacity_off = False
    rows.append(
        {
            "probe": "P5_branch_disable",
            "probe_group": winner,
            "seed": 0,
            **_probe_delta(
                base_mae,
                float(np.mean(np.abs(disabled_prediction - targets))),
                base_prediction,
                disabled_prediction,
            ),
        }
    )
    _write_csv(
        MECHANISM_DIR / "probes.csv",
        rows,
        [
            "probe",
            "probe_group",
            "seed",
            "mae",
            "delta_mae",
            "mean_abs_prediction_delta",
            "max_abs_prediction_delta",
        ],
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "winner": winner,
        "base_valid_mae": base_mae,
        "rows": rows,
        "candidate_specific": _candidate_specific_diagnostics(model, winner, grad_batch, device, loader),
    }
    _write_json(out_path, _payload(payload))
    _write_json(MECHANISM_DIR / "candidate_specific.json", payload["candidate_specific"])
    print(f"[mechanism] winner={winner} base={base_mae:.6f} probes={len(rows)}", flush=True)
    return payload


def _candidate_specific_diagnostics(
    model: cl.CapacityModel,
    winner: str,
    grad_batch: Any,
    device: torch.device,
    loader: Any,
) -> dict[str, Any]:
    """Winner-specific frozen diagnostics (no loss term, no rescue)."""
    model.eval()
    batch = next(iter(loader)).to(device)
    payload: dict[str, Any] = {"winner": winner}
    if winner == "F":
        with torch.no_grad():
            coord = model.code(batch.dict_phi)
            heads = model.head_products(coord, batch)
            for name, value in heads.items():
                norms = value.norm(dim=2)  # [H, occurrences]
                payload[f"{name}_head_mean_norm"] = norms.mean(dim=1).tolist()
                payload[f"{name}_head_std_norm"] = norms.std(dim=1).tolist()
                flat = value.permute(1, 0, 2).reshape(value.shape[1], -1)
                cosine = torch.nn.functional.cosine_similarity(
                    flat[:, None, :], flat[None, :, :], dim=2
                )
                payload[f"{name}_head_cosine"] = cosine.tolist()
        train_batch = grad_batch.to(device)
        parameters = dict(model.named_parameters())
        model.train(False)
        prediction, aux = model(train_batch, mask=cssd.CSSD_MASK, return_aux=True)
        loss = torch.nn.functional.l1_loss(prediction.view(-1), train_batch.y.view(-1)) + float(
            cm.H1_LAMBDA
        ) * model.reconstruction_loss(aux["phi"], aux["coord"])
        model.zero_grad(set_to_none=True)
        loss.backward()
        per_head: dict[str, float] = {}
        for head in range(cl.FUSION_HEADS):
            total = 0.0
            for name in (
                f"F_NS.{head}.weight",
                f"F_NC.{head}.weight",
                f"F_ES.{head}.weight",
                f"F_EC.{head}.weight",
            ):
                grad = parameters[name].grad
                if grad is not None:
                    total += float(grad.detach().norm() ** 2)
            per_head[f"head{head}"] = float(np.sqrt(total))
        payload["per_head_gradient_norm"] = per_head
        # residual branch disable delta on the frozen checkpoint
        model.capacity_off = True
        disabled = cv2.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
        model.capacity_off = False
        model.zero_grad(set_to_none=True)
        model.train(True)
        payload["disable_residual_delta_mae"] = float(disabled) - float(
            cv2.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
        )
    elif winner == "R":
        captured: dict[str, torch.Tensor] = {}

        def hook(module, inputs, output):  # noqa: ANN001
            captured["input"] = inputs[0].detach()

        handle = model.pair_encoder.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(batch, mask=cssd.CSSD_MASK)
        finally:
            handle.remove()
        with torch.no_grad():
            contributions = model.pair_block_contributions(
                captured["input"], captured["input"][:, -int(p2.PAIR_HIDDEN) :]
            )
        for index, entry in enumerate(contributions):
            payload[f"block{index}_contribution_norm"] = float(entry["contribution"].norm(dim=1).mean())
            payload[f"block{index}_modulation_mean_abs"] = float(entry["modulation"].abs().mean())
            payload[f"block{index}_shift_mean_abs"] = float(entry["shift"].abs().mean())
        for index, block in enumerate(model.pair_blocks):
            with torch.no_grad():
                block.fc2.weight.zero_()
                block.fc2.bias.zero_()
                disabled = cv2.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
            payload[f"block{index}_disable_delta_mae"] = float(disabled) - float(
                cv2.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
            )
    elif winner == "G":
        with torch.no_grad():
            _, aux = model(batch, mask=cssd.CSSD_MASK, return_aux=True)
            pair_batch = batch.batch[batch.pair_index[0]]
            h_env, h_pair = model.summary_values(
                aux["E"], batch.batch, aux["pair_value"], pair_batch, int(batch.global_context.shape[0])
            )
            payload["summary_env_norm_mean"] = float(h_env.norm(dim=1).mean())
            payload["summary_pair_norm_mean"] = float(h_pair.norm(dim=1).mean())
            activation_env = torch.sigmoid(model.summary_gate_env(aux["E"]))
            activation_pair = torch.sigmoid(model.summary_gate_pair(aux["pair_value"]))
            payload["gate_env_mean"] = float(activation_env.mean())
            payload["gate_env_std"] = float(activation_env.std())
            payload["gate_pair_mean"] = float(activation_pair.mean())
            payload["gate_pair_std"] = float(activation_pair.std())
            for name, activation in (("env", activation_env), ("pair", activation_pair)):
                mean_activation = activation.mean(dim=0)
                probabilities = mean_activation / mean_activation.sum().clamp_min(1e-12)
                entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
                payload[f"gate_{name}_entropy"] = entropy
        for kind in ("env", "pair"):
            model.disabled_summaries = {kind}
            disabled = cv2.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
            model.disabled_summaries = set()
            payload[f"disable_{kind}_summary_delta_mae"] = float(disabled) - float(
                cv2.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
            )
        payload["summary_proj_norm"] = float(model.reader.summary_proj.weight.detach().norm())
    return payload


# ---------------------------------------------------------------------------
# stage: report tables
# ---------------------------------------------------------------------------


def stage_report() -> dict[str, Any]:
    _ensure_dirs()
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "preregistration": _read_json(RESULTS_DIR / "preregistration_snapshot.json"),
        "parameter_budget": _read_json(RESULTS_DIR / "parameter_budget.json"),
        "architecture_fingerprints": _read_json(RESULTS_DIR / "architecture_fingerprints.json"),
    }
    if (RESULTS_DIR / "init_audit.json").exists():
        summary["init_audit"] = _read_json(RESULTS_DIR / "init_audit.json")
    if (CALIBRATION_DIR / "calibration_decision.json").exists():
        summary["calibration"] = _read_json(CALIBRATION_DIR / "calibration_decision.json")
    if (SCREENING_DIR / "decision.json").exists():
        summary["screening"] = _read_json(SCREENING_DIR / "decision.json")
    elif (SCREENING_DIR / "NOT_RUN.json").exists():
        summary["screening"] = {"run": False, "reason": _read_json(SCREENING_DIR / "NOT_RUN.json")["reason"]}
    if (FULL_WINNER_DIR / "final.json").exists():
        final = _read_json(FULL_WINNER_DIR / "final.json")
        summary["full"] = {
            "winner": final.get("winner"),
            "params": final.get("params"),
            "cap_base_params": final.get("cap_base_params"),
            "best_valid_mae": final.get("best_valid_mae"),
            "best_epoch": final.get("best_epoch"),
            "soup": final.get("soup"),
            "completed": final.get("completed"),
            "epochs_run": final.get("epochs_run"),
            "wall_clock_s": final.get("wall_clock_s"),
            "interpretation": final.get("interpretation"),
        }
    elif (FULL_DIR / "NOT_RUN.json").exists():
        summary["full"] = {"run": False, "reason": _read_json(FULL_DIR / "NOT_RUN.json")["reason"]}
    if (MECHANISM_DIR / "probes.json").exists():
        summary["mechanism"] = _read_json(MECHANISM_DIR / "probes.json")
    _write_json(RESULTS_DIR / "summary.json", _payload(summary))
    _write_analysis_tables(summary)
    print("[report] analysis_tables.md and summary.json written", flush=True)
    return summary


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def _write_analysis_tables(summary: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Analysis tables — `e2e_dictenv_capacity_localization_v2`")
    lines.append("")
    prereg = summary["preregistration"]
    lines.append(
        f"Preregistration sha256 `{prereg['sha256'][:12]}…`; commit `{summary['git_commit'][:12]}`; "
        f"CAP-BASE = {cv2.CAP_BASE}; official test never loaded."
    )
    lines.append("")
    lines.append("## 1. Parameter budget and architecture fingerprints")
    lines.append("")
    budget = summary["parameter_budget"]
    fingerprints = summary["architecture_fingerprints"]["rows"]
    lines.append(
        f"CAP-BASE {budget['cap_base_params']} params; added-parameter ratio "
        f"{budget['added_params_ratio']:.4f} (max {cv2.BUDGET_RATIO_MAX}); v2 counts match v1: "
        f"{budget['all_candidates_match_v1']}."
    )
    lines.append("")
    lines.append("| candidate | added | total | matches v1 | shape fingerprint |")
    lines.append("|---|---|---|---|---|")
    for kind in ("M0", "F", "R", "G"):
        row = budget["rows"][kind]
        matches = row.get("matches_v1_total_params", "-")
        lines.append(
            f"| {kind} | {row['added_params_vs_cap_base']} | {row['total_params']} | {matches} | "
            f"`{fingerprints[kind]['shape_fingerprint_sha256'][:12]}…` |"
        )
    lines.append("")
    lines.append("## 2. Initialization audit (step 0, before any training)")
    lines.append("")
    init_audit = summary.get("init_audit")
    if init_audit:
        lines.append(
            "| candidate | step0 MAE | Δ vs CAP-BASE | mean shift | median shift | max shift | "
            "residual-zero identity | grads > 0 | hard ≤ 0.002 | preferred ≤ 0.001 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for kind in ("M0", "F", "R", "G"):
            row = init_audit["rows"][kind]
            lines.append(
                f"| {kind} | {_fmt(row['step0_valid_mae'])} | {_fmt(row['step0_mae_delta_vs_base'])} | "
                f"{_fmt(row['step0_mean_abs_prediction_delta'], 5)} | "
                f"{_fmt(row['step0_median_abs_prediction_delta'], 5)} | "
                f"{_fmt(row['step0_max_abs_prediction_delta'], 5)} | "
                f"{row['residual_zero_identity']} | {row['new_module_gradients_positive']} | "
                f"{row['init_mean_shift_hard_ok']} | {row['init_mean_shift_preferred_ok']} |"
            )
    else:
        lines.append("Not available.")
    lines.append("")
    lines.append("## 3. Phase A — M0 calibration (20 epochs, fresh Adam lr 1e-4)")
    lines.append("")
    calibration = summary.get("calibration")
    if calibration:
        lines.append("| metric | value | threshold | pass |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| soup (1–20) | {_fmt(calibration['calibration_soup'])} | ≤ {calibration['thresholds']['soup_max']} | "
            f"{calibration['C1_pass']} |"
        )
        lines.append(
            f"| last-5 mean (16–20) | {_fmt(calibration['calibration_last5'])} | "
            f"≤ {calibration['thresholds']['late_mean_max']} | {calibration['C2_pass']} |"
        )
        lines.append(
            f"| epoch 20 | {_fmt(calibration['epoch20'])} | ≤ {calibration['thresholds']['epoch20_max']} | "
            f"{calibration['C3_pass']} |"
        )
        lines.append(
            f"| late slope (16–20) | {calibration['late_slope']:+.3e} | "
            f"≤ {calibration['thresholds']['late_slope_max']:+.1e} | {calibration['C3_pass']} |"
        )
        lines.append("")
        lines.append(
            f"best {_fmt(calibration['calibration_best'])} (Δ {_fmt(calibration['delta_best'])}), "
            f"soup Δ {_fmt(calibration['delta_soup'])}, last-5 Δ {_fmt(calibration['delta_last5'])}; "
            f"verdict **{calibration['verdict']}**."
        )
    else:
        lines.append("Not available.")
    lines.append("")
    lines.append("## 4. Phase B — repaired differential-LR screen (40 epochs, base 1e-4 / new 1e-3)")
    lines.append("")
    screening = summary.get("screening")
    if screening and screening.get("rows"):
        lines.append(
            "| candidate | params | step0 MAE | best valid | soup (21–40) | Δsoup vs M0 | Δ vs start | "
            "last-10 | Δlast10 vs M0 | branch grad | S1 | S2 | S3 | S4 | verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in screening["rows"]:
            lines.append(
                f"| {row['candidate']} | {row['params']} | {_fmt(row['step0_mae'])} | "
                f"{_fmt(row['best_valid'])} | {_fmt(row['screen_soup'])} | "
                f"{_fmt(row['delta_soup_vs_M0'])} | {_fmt(row['delta_vs_start'])} | {_fmt(row['last10'])} | "
                f"{_fmt(row['delta_last10_vs_M0'])} | {_fmt(row['branch_grad'], 3)} | {row['S1']} | "
                f"{row['S2']} | {row['S3']} | {row['S4']} | {row['verdict']} |"
            )
        lines.append("")
        selection = screening["selection"]
        lines.append(
            f"Winner: **{selection['winner']}** ({selection['reason']}); passing: {selection['passing']}; "
            f"frozen tie order F > R > G."
        )
    else:
        lines.append(f"Not run: {screening.get('reason') if screening else 'no decision artifact'}.")
    lines.append("")
    lines.append("## 5. Phase C — full winner run (only if bought)")
    lines.append("")
    full = summary.get("full")
    if full and full.get("winner"):
        lines.append(
            f"Winner **{full['winner']}**, params {full['params']} (CAP-BASE {full['cap_base_params']}), "
            f"best valid {_fmt(full['best_valid_mae'])} @ {full['best_epoch']}, "
            f"soup {_fmt(full['soup']['soup_valid_mae'])} (members {full['soup']['members']}), "
            f"epochs {full['epochs_run']}, wall {full['wall_clock_s']:.0f}s."
        )
        interpretation = full["interpretation"]
        lines.append("")
        lines.append(
            f"Band **{interpretation['band']}**; Δ vs CAP-BASE {_fmt(interpretation['delta_vs_cap_base'])}, "
            f"Δ vs FINAL-CLEAN sparse {_fmt(interpretation['delta_vs_final_clean_sparse'])}."
        )
    else:
        lines.append(f"Not run: {full.get('reason') if full else 'no decision artifact'}.")
    lines.append("")
    mechanism = summary.get("mechanism")
    lines.append("## 6. Mechanism audit (frozen winner checkpoint)")
    lines.append("")
    if mechanism and mechanism.get("base_valid_mae") is not None:
        lines.append("| probe | group | seed | MAE | Δ MAE | mean shift | max shift |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in mechanism["rows"]:
            lines.append(
                f"| {row['probe']} | {row['probe_group']} | {row['seed']} | {_fmt(row['mae'])} | "
                f"{_fmt(row['delta_mae'])} | {_fmt(row.get('mean_abs_prediction_delta'), 5)} | "
                f"{_fmt(row.get('max_abs_prediction_delta'), 5)} |"
            )
        lines.append("")
        lines.append("Candidate-specific diagnostics: `mechanism/candidate_specific.json`.")
    else:
        lines.append("Not run (no full winner).")
    lines.append("")
    (RESULTS_DIR / "analysis_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# chain / CLI
# ---------------------------------------------------------------------------


def chain(threads: int = THREADS) -> None:
    _ensure_dirs()
    stage_preflight()
    stage_init_audit()
    stage_calibration(threads=threads)
    if not _calibration_passed():
        print("[chain] calibration failed -> stop before Phase B", flush=True)
        stage_report()
        return
    launch_wave(("M0", "F", "R"), threads, SCREEN_CONCURRENCY)
    launch_wave(("G",), threads, 1)
    stage_summarize()
    stage_full(threads=FULL_THREADS)
    stage_mechanism(threads=threads)
    stage_report()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "init-audit",
        "calibration",
        "wave1",
        "wave2",
        "summarize",
        "full-run",
        "mechanism",
        "report",
    ):
        stage = sub.add_parser(name)
        stage.add_argument("--threads", type=int, default=THREADS)
        if name in ("full-run", "mechanism", "report"):
            stage.add_argument("--force", action="store_true")
    screen = sub.add_parser("screen-arm")
    screen.add_argument("--kind", required=True)
    screen.add_argument("--threads", type=int, default=THREADS)
    chain_parser = sub.add_parser("chain")
    chain_parser.add_argument("--threads", type=int, default=THREADS)

    args = parser.parse_args(argv)
    if args.command == "preflight":
        stage_preflight()
    elif args.command == "init-audit":
        stage_init_audit()
    elif args.command == "calibration":
        stage_calibration(args.threads)
    elif args.command == "screen-arm":
        stage_screen_arm(args.kind, args.threads)
    elif args.command == "wave1":
        launch_wave(("M0", "F", "R"), args.threads, SCREEN_CONCURRENCY)
    elif args.command == "wave2":
        launch_wave(("G",), args.threads, 1)
    elif args.command == "summarize":
        stage_summarize()
    elif args.command == "full-run":
        stage_full(threads=args.threads, force=bool(args.force))
    elif args.command == "mechanism":
        stage_mechanism(threads=args.threads, force=bool(args.force))
    elif args.command == "report":
        stage_report()
    elif args.command == "chain":
        chain(threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
