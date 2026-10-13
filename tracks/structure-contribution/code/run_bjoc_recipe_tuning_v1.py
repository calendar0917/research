"""cscl-bjoc-recipe-tuning-v1 driver.

Protocol: ``configs/bjoc_recipe_tuning_v1.yaml`` (frozen before any GPU
trajectory).  Track: ``tracks/structure-contribution``.

Single scientific question
--------------------------
Holding the validated masked-pretrained BJOC-Null P completely fixed except
for the *supervision-phase* recipe, do the five pre-declared recipes

    R0  240 ep, constant 1e-3,          wd 1e-5   (historical P, replay only)
    R1  240 ep, cosine 1e-3 -> 1e-5,    wd 1e-5
    R2  360 ep, constant 1e-3,          wd 1e-5
    R3  360 ep, cosine 1e-3 -> 1e-5,    wd 1e-5
    R4  240 ep, constant 1e-3,          wd 1e-4

change official-valid raw-y MAE relative to R0 on the same seed and split?

Discipline
----------
* No model / architecture / objective / data / label / pretraining change.
* R0 is never retrained: its frozen checkpoints and frozen valid predictions
  are replayed and hash-verified.
* Candidate training re-uses exactly the P supervision-phase initialisation:
  scratch BJOC init (hash-matched), frozen full-10k SSL encoder state
  (hash-matched), same per-seed batch order (prefix hash-matched), same
  raw-y L1, gradient clip, batch size and last-5 soup window.
* R1-R4 differ from R0 only in the declared epochs / schedule / weight decay.
* Predictions are written label-blind and hashed before the single
  official-valid label read.
* Official test is never loaded in any form.

Stages
------
``verify-inputs``       re-hash every reused input against the frozen config
``replay-r0``           CPU replay of the frozen P checkpoints vs frozen P preds
``freeze``              protocol freeze (code/config/input hashes)
``prep``                verify-inputs + replay-r0 + freeze
``smoke``               short GPU correctness run (never a result)
``train``               one formal candidate trajectory
``predict``             label-blind official-valid forward for a frozen soup
``freeze-predictions``  screening / confirmation prediction hash freeze
``score-screening``     the one official-valid label read (seed 0 candidates)
``screen``              pre-registered seed-0 gate + unique-candidate selection
``freeze-confirmation`` hash the purchased seed-1 prediction (+ P1 reference)
``score-confirmation``  two-seed paired metrics for the purchased candidate
``decision``            frozen final decision
``analysis``            descriptive error / fit structure (post-read)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-recipe-tuning-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_recipe_tuning_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1"
F10K = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
TERMINAL = REPO_ROOT / "tracks/structure-contribution/results/bjoc_pretraining_mechanism_terminal_test_v1"

# ---- official data (no internal split; test never referenced) --------------
N_TRAIN = 10000
N_VALID = 1000
SEEDS = (0, 1)

# ---- frozen recipe constants ------------------------------------------------
BATCH_SIZE = 128
REF_EPOCHS = 240
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
GRAD_CLIP = 5.0
STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)  # 79
REF_UPDATES = REF_EPOCHS * STEPS_PER_EPOCH  # 18960

# ---- frozen analysis constants ---------------------------------------------
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
TOP_K_REMOVED = (1, 5, 10)
TAIL_THRESHOLD = -3.0
EXTREME_VALID_ROW = 172
CURVE_LOG_EVERY = 20

# ---- pre-registered screening / confirmation thresholds --------------------
SCREEN_GAIN_THRESHOLD = -0.003
SCREEN_EXTREME_EXCLUDED_THRESHOLD = -0.002
CONFIRM_GAIN_THRESHOLD = -0.003
R0_REPLAY_TOLERANCE = 1.0e-5

# ---- candidate registry (must equal the frozen config) ----------------------
CANDIDATES: dict[str, dict[str, Any]] = {
    "R0": {"epochs": 240, "schedule": "constant", "lr": 1.0e-3, "lr_min": None, "weight_decay": 1.0e-5, "reuse": "historical_P_checkpoint"},
    "R1": {"epochs": 240, "schedule": "cosine", "lr": 1.0e-3, "lr_min": 1.0e-5, "weight_decay": 1.0e-5, "reuse": None},
    "R2": {"epochs": 360, "schedule": "constant", "lr": 1.0e-3, "lr_min": None, "weight_decay": 1.0e-5, "reuse": None},
    "R3": {"epochs": 360, "schedule": "cosine", "lr": 1.0e-3, "lr_min": 1.0e-5, "weight_decay": 1.0e-5, "reuse": None},
    "R4": {"epochs": 240, "schedule": "constant", "lr": 1.0e-3, "lr_min": None, "weight_decay": 1.0e-4, "reuse": None},
}
SCREENING_CANDIDATES = ("R1", "R2", "R3", "R4")

# ---- artifacts --------------------------------------------------------------
INPUT_AUDIT = OUT / "input_audit.json"
R0_REPLAY_JSON = OUT / "r0_replay.json"
FREEZE_JSON = OUT / "protocol_freeze.json"
PRED_DIR = OUT / "predictions"
PRED_ACCESS = OUT / "prediction_access_log.json"
PRED_FREEZE = OUT / "prediction_freeze.json"
CONFIRM_FREEZE = OUT / "confirmation_prediction_freeze.json"
VALID_LOCK = OUT / "valid_read.lock"
VALID_Y_COPY = OUT / "valid_y_read_once.npy"
SCREENING_METRICS = OUT / "screening_metrics.json"
SCREENING_JSON = OUT / "screening.json"
CONFIRMATION_METRICS = OUT / "confirmation_metrics.json"
ANALYSIS_JSON = OUT / "analysis.json"
SCIENTIFIC_DECISION = OUT / "scientific_decision.json"
SMOKE_DIR = OUT / "smoke"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# io / hashing helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(f10k._j(payload), indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return f10k.file_sha256(path)


def git_commit() -> str:
    return f10k.git_commit()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def candidates_fingerprint() -> str:
    return hashlib.sha256(json.dumps(CANDIDATES, sort_keys=True).encode()).hexdigest()


def _code_paths() -> dict[str, Path]:
    return {
        "driver": _CODE_DIR / "run_bjoc_recipe_tuning_v1.py",
        "config": CONFIG_PATH,
        "base_driver": _CODE_DIR / "run_bjoc_full10k_pretraining_transfer_v1.py",
        "base_module": _CODE_DIR / "bjoc_masked_pretraining_v1.py",
        "objects": _CODE_DIR / "structural_relation_objects_v1.py",
        "composer": _CODE_DIR / "structural_relation_composer_v1.py",
        "features": _CODE_DIR / "cscl_features.py",
    }


def code_hashes() -> dict[str, str]:
    return {k: file_sha256(p) for k, p in _code_paths().items()}


def recipe_of(key: str) -> dict[str, Any]:
    if key not in CANDIDATES:
        raise SystemExit(f"unknown recipe {key!r}; candidates are {sorted(CANDIDATES)}")
    return CANDIDATES[key]


def _check_code_matches_config(cfg: Mapping[str, Any]) -> None:
    """Refuse to run when the code candidate registry and the frozen config drift."""
    cfg_candidates = cfg["candidates"]
    for key, spec in CANDIDATES.items():
        if key not in cfg_candidates:
            raise SystemExit(f"config is missing candidate {key}")
        for field_name in ("epochs", "schedule", "lr", "lr_min", "weight_decay"):
            if cfg_candidates[key][field_name] != spec[field_name]:
                raise SystemExit(
                    f"candidate {key} field {field_name}: config {cfg_candidates[key][field_name]!r} != code {spec[field_name]!r}"
                )
    if set(cfg_candidates) != set(CANDIDATES):
        raise SystemExit(f"config candidate set {sorted(cfg_candidates)} != code {sorted(CANDIDATES)}")
    if cfg["recipe"]["steps_per_epoch"] != STEPS_PER_EPOCH:
        raise SystemExit("config steps_per_epoch mismatch")
    if cfg["recipe"]["soup_last"] != SOUP_LAST:
        raise SystemExit("config soup_last mismatch")
    if cfg["recipe"]["train_shuffle_offset"] != TRAIN_SHUFFLE_OFFSET:
        raise SystemExit("config train_shuffle_offset mismatch")
    if cfg["frozen"]["ref_epochs"] != REF_EPOCHS:
        raise SystemExit("config ref_epochs mismatch")
    if cfg["frozen"]["expected_parameters"] != 413505:
        raise SystemExit("config expected_parameters mismatch")


# ---------------------------------------------------------------------------
# input audit + protocol freeze
# ---------------------------------------------------------------------------


def _encoder_state_sha256(enc_state: Mapping[str, Mapping[str, torch.Tensor]]) -> str:
    combined: dict[str, torch.Tensor] = {}
    for prefix, group in (("atom", "atom_encoder"), ("bond", "bond_encoder"), ("junc", "junction_encoder")):
        for k, v in enc_state[group].items():
            combined[f"{prefix}:{k}"] = v
    return f10k.state_group_sha256(combined)


def audit_inputs() -> dict[str, Any]:
    """Re-hash and re-derive every reused frozen input (no valid/test labels)."""
    cfg = load_config()
    frozen = cfg["frozen"]
    checks: dict[str, Any] = {}

    def file_check(name: str, path: Path, expected: str) -> dict[str, Any]:
        exists = Path(path).exists()
        actual = file_sha256(path) if exists else None
        checks[name] = {
            "path": str(Path(path).relative_to(REPO_ROOT)),
            "exists": bool(exists),
            "sha256": actual,
            "expected": expected,
            "ok": bool(exists and actual == expected),
        }
        return checks[name]

    manifest = read_json(F10K / "objects_manifest.json")
    file_check("objects_train10k", F10K / "objects_train10k.pt", frozen["objects_train10k_sha256"])
    file_check("objects_valid", F10K / "objects_valid.pt", frozen["objects_valid_sha256"])
    file_check("prep_train10k", F10K / "prep_train10k.json", frozen["prep_train10k_sha256"])
    checks["objects_manifest_recorded"] = {
        "objects_train10k_sha256": manifest["objects_train10k_sha256"],
        "objects_valid_sha256": manifest["objects_valid_sha256"],
        "prep_train10k_sha256": manifest["prep_train10k_sha256"],
        "matches_frozen": bool(
            manifest["objects_train10k_sha256"] == frozen["objects_train10k_sha256"]
            and manifest["objects_valid_sha256"] == frozen["objects_valid_sha256"]
            and manifest["prep_train10k_sha256"] == frozen["prep_train10k_sha256"]
        ),
    }
    checks["objects_manifest_recorded"]["ok"] = checks["objects_manifest_recorded"]["matches_frozen"]

    audit = read_json(TERMINAL / "model_reuse_audit.json")
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        enc_path = F10K / f"pretrain_encoder_s{seed}.pt"
        exp_enc = frozen[f"pretrain_encoder_s{seed}_sha256"]
        enc_state = torch.load(enc_path, map_location="cpu", weights_only=False)
        state_hash = _encoder_state_sha256(enc_state)
        pre_rec = read_json(F10K / f"pretraining_seed{seed}.json")
        enc_ok = bool(
            file_sha256(enc_path) == exp_enc
            and state_hash == pre_rec["encoder_final_sha256"]
            and state_hash == frozen["p_encoder_used_sha256"][str(seed)]
        )
        checks[f"pretrain_encoder_s{seed}"] = {
            "path": str(enc_path.relative_to(REPO_ROOT)),
            "exists": bool(enc_path.exists()),
            "file_sha256": file_sha256(enc_path),
            "expected_file_sha256": exp_enc,
            "encoder_state_sha256": state_hash,
            "recorded_training_sha256": pre_rec["encoder_final_sha256"],
            "ok": enc_ok,
        }
        soup_check = file_check(f"soup_P_seed{seed}", F10K / f"soup_P_seed{seed}.pt", frozen[f"soup_P_seed{seed}_sha256"])
        soup_check["terminal_recorded_sha256"] = audit["checkpoint_records"][f"P{seed}"]["checkpoint_sha256"]
        soup_check["terminal_matches"] = bool(soup_check["terminal_recorded_sha256"] == frozen[f"soup_P_seed{seed}_sha256"])
        soup_check["ok"] = bool(soup_check["ok"] and soup_check["terminal_matches"])
        # reference-only: the P prediction npz files are git-ignored and may be
        # absent from the compute host; replay verifies the regenerated ARRAY
        # against a frozen hash instead (see replay_r0)
        pred_path = F10K / "predictions" / f"valid_P{seed}.npz"
        pred_expected = frozen[f"valid_P_pred_s{seed}_sha256"]
        checks[f"valid_P{seed}_preds"] = {
            "path": str(pred_path.relative_to(REPO_ROOT)),
            "exists": bool(pred_path.exists()),
            "sha256": file_sha256(pred_path) if pred_path.exists() else None,
            "expected": pred_expected,
            "expected_array_sha256": frozen[f"valid_P_pred_s{seed}_array_sha256"],
            "unavailable_on_host": bool(not pred_path.exists()),
            "ok": bool(pred_path.exists() and file_sha256(pred_path) == pred_expected) if pred_path.exists() else None,
        }

        # independently rebuild the P supervision-phase initialisation
        norm = read_json(F10K / "prep_train10k.json")
        model = f10k._make_downstream_model(seed, norm, "cpu")
        init_full = src.state_dict_hash(model)
        init_group = mp.group_state_hash(model)
        init_encoder = f10k._encoder_group_hash(model)
        mp.load_encoder_state(model, enc_state, "cpu")
        encoder_used = f10k._encoder_group_hash(model)
        p_rec = read_json(F10K / f"train_P_seed{seed}.json")
        order_hash = f10k.batch_order_hash(seed, N_TRAIN, BATCH_SIZE, REF_EPOCHS)
        per_seed[str(seed)] = {
            "init_full_sha256": init_full,
            "init_group_state_sha256": init_group,
            "init_encoder_sha256": init_encoder,
            "encoder_used_sha256": encoder_used,
            "batch_order_hash": order_hash,
            "matches_frozen_config": bool(
                init_full == frozen["p_init_full_sha256"][str(seed)]
                and init_group == frozen["p_init_group_state_sha256"][str(seed)]
                and init_encoder == frozen["p_init_encoder_sha256"][str(seed)]
                and encoder_used == frozen["p_encoder_used_sha256"][str(seed)]
                and order_hash == frozen["p_batch_order_hash"][str(seed)]
            ),
            "matches_P_round_json": bool(
                init_full == p_rec["init_full_sha256"]
                and init_group == p_rec["init_group_state_sha256"]
                and init_encoder == p_rec["init_encoder_sha256"]
                and encoder_used == p_rec["encoder_used_sha256"]
                and order_hash == p_rec["batch_order_hash"]
            ),
            "parameters": src.parameter_audit(src.BJOCConfig(), model)["actual_parameters"],
            "forbidden_modules": src.forbidden_module_scan(model),
        }
    checks["p_supervision_init_replay"] = {"per_seed": per_seed, "ok": all(v["matches_frozen_config"] and v["matches_P_round_json"] for v in per_seed.values())}

    blocking_keys = [
        "objects_train10k",
        "objects_valid",
        "prep_train10k",
        "pretrain_encoder_s0",
        "pretrain_encoder_s1",
        "soup_P_seed0",
        "soup_P_seed1",
        "objects_manifest_recorded",
        "p_supervision_init_replay",
    ]
    reference_keys = ["valid_P0_preds", "valid_P1_preds"]
    checks["all_training_inputs_ok"] = all(checks[k]["ok"] is True for k in blocking_keys)
    checks["evaluation_references_ok"] = all(checks[k]["ok"] is True for k in reference_keys)
    checks["evaluation_references_present"] = all(not checks[k]["unavailable_on_host"] for k in reference_keys)
    checks["all_ok"] = bool(checks["all_training_inputs_ok"])
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "checks": checks,
        "all_ok": checks["all_ok"],
        "all_training_inputs_ok": checks["all_training_inputs_ok"],
        "evaluation_references_ok": checks["evaluation_references_ok"],
        "evaluation_references_present": checks["evaluation_references_present"],
        "official_valid_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(INPUT_AUDIT, payload)
    log(
        f"[verify-inputs] training_ok={checks['all_training_inputs_ok']} "
        f"references_present={checks['evaluation_references_present']} references_ok={checks['evaluation_references_ok']}"
    )
    if not checks["all_training_inputs_ok"]:
        bad = [k for k in blocking_keys if checks[k]["ok"] is not True]
        raise SystemExit(f"input audit failed: {bad}")
    return payload


def replay_r0() -> dict[str, Any]:
    """CPU replay of the frozen P soups against the frozen P valid predictions.

    The primary check is a device/serialisation-independent hash of the
    regenerated float32 prediction array against the frozen config constant.
    When the git-ignored npz reference is present locally, the max|dpred| and
    the npz file hash are checked as well.
    """
    cfg = load_config()
    frozen = cfg["frozen"]
    norm = read_json(F10K / "prep_train10k.json")
    valid_objs = f10k.load_objects(F10K / "objects_valid.pt")
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        soup = torch.load(F10K / f"soup_P_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = f10k._make_downstream_model(seed, norm, "cpu")
        model.load_state_dict(soup)
        pred = f10k.predict_bjoc(model, valid_objs, "cpu")
        pred_f32 = np.ascontiguousarray(pred.astype(np.float32))
        array_hash = f10k.array_sha256(pred_f32)
        expected_array_hash = frozen[f"valid_P_pred_s{seed}_array_sha256"]
        audit = src.parameter_audit(src.BJOCConfig(), model)
        ref_path = F10K / "predictions" / f"valid_P{seed}.npz"
        npz_entry: dict[str, Any] = {"available": bool(ref_path.exists())}
        npz_ok = True
        if ref_path.exists():
            ref = np.asarray(np.load(ref_path)["pred"], dtype=np.float32)
            npz_entry.update(
                {
                    "max_abs_pred_diff_vs_frozen": float(np.abs(pred_f32 - ref).max()),
                    "mean_abs_pred_diff_vs_frozen": float(np.abs(pred_f32 - ref).mean()),
                    "array_sha256": f10k.array_sha256(ref),
                    "file_sha256": file_sha256(ref_path),
                    "expected_file_sha256": frozen[f"valid_P_pred_s{seed}_sha256"],
                }
            )
            npz_ok = bool(
                npz_entry["max_abs_pred_diff_vs_frozen"] <= R0_REPLAY_TOLERANCE
                and npz_entry["file_sha256"] == npz_entry["expected_file_sha256"]
            )
        per_seed[str(seed)] = {
            "array_sha256": array_hash,
            "expected_array_sha256": expected_array_hash,
            "array_hash_matches": bool(array_hash == expected_array_hash),
            "npz_reference": npz_entry,
            "predictions_finite": bool(np.isfinite(pred).all()),
            "parameter_audit": audit,
            "forbidden_modules": src.forbidden_module_scan(model),
            "replay_ok": bool(
                array_hash == expected_array_hash
                and npz_ok
                and np.isfinite(pred).all()
                and audit["parameter_exact"]
                and audit["actual_parameters"] == 413505
                and not src.forbidden_module_scan(model)
            ),
        }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "device": "cpu",
        "tolerance": R0_REPLAY_TOLERANCE,
        "per_seed": per_seed,
        "all_ok": all(v["replay_ok"] for v in per_seed.values()),
        "official_valid_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(R0_REPLAY_JSON, payload)
    log(
        f"[replay-r0] array-hash match {[v['array_hash_matches'] for v in per_seed.values()]} "
        f"npz diff {[v['npz_reference'].get('max_abs_pred_diff_vs_frozen') for v in per_seed.values()]} "
        f"all_ok={payload['all_ok']}"
    )
    if not payload["all_ok"]:
        raise SystemExit("R0 replay failed")
    return payload


def write_protocol_freeze(input_audit: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    input_files = {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in (
            F10K / "objects_train10k.pt",
            F10K / "objects_valid.pt",
            F10K / "prep_train10k.json",
            F10K / "pretrain_encoder_s0.pt",
            F10K / "pretrain_encoder_s1.pt",
            F10K / "soup_P_seed0.pt",
            F10K / "soup_P_seed1.pt",
        )
    }
    for seed in SEEDS:
        pred_path = F10K / "predictions" / f"valid_P{seed}.npz"
        if pred_path.exists():
            input_files[str(pred_path.relative_to(REPO_ROOT))] = file_sha256(pred_path)
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "code_hashes": code_hashes(),
        "candidates_sha256": candidates_fingerprint(),
        "input_files": input_files,
        "input_audit_sha256": file_sha256(INPUT_AUDIT),
        "r0_replay_sha256": file_sha256(R0_REPLAY_JSON),
        "input_audit_ok": bool(input_audit["all_ok"]),
        "r0_replay_ok": bool(replay["all_ok"]),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    if not (payload["input_audit_ok"] and payload["r0_replay_ok"]):
        raise SystemExit("refusing to freeze: input audit or R0 replay failed")
    write_json(FREEZE_JSON, payload)
    log(f"[freeze] code {payload['code_hashes']['driver'][:12]} config {payload['config_sha256'][:12]}")
    return payload


def verify_freeze() -> dict[str, Any]:
    if not FREEZE_JSON.exists():
        raise SystemExit("protocol freeze missing; run prep first")
    payload = read_json(FREEZE_JSON)
    if payload["protocol"] != PROTOCOL_ID:
        raise SystemExit("protocol freeze mismatch")
    for key, path in _code_paths().items():
        actual = file_sha256(path)
        if actual != payload["code_hashes"][key]:
            raise SystemExit(f"code hash mismatch for {key}: {actual} != {payload['code_hashes'][key]}")
    if file_sha256(CONFIG_PATH) != payload["config_sha256"]:
        raise SystemExit("config changed after freeze")
    if candidates_fingerprint() != payload["candidates_sha256"]:
        raise SystemExit("candidate registry changed after freeze")
    for rel, expected in payload["input_files"].items():
        actual = file_sha256(REPO_ROOT / rel)
        if actual != expected:
            raise SystemExit(f"frozen input changed: {rel}")
    if not (payload["input_audit_ok"] and payload["r0_replay_ok"]):
        raise SystemExit("freeze recorded a failed precondition")
    return payload


def cmd_verify_inputs(args: argparse.Namespace) -> int:
    audit_inputs()
    return 0


def cmd_replay_r0(args: argparse.Namespace) -> int:
    replay_r0()
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    if not INPUT_AUDIT.exists() or not R0_REPLAY_JSON.exists():
        raise SystemExit("run verify-inputs and replay-r0 first")
    write_protocol_freeze(read_json(INPUT_AUDIT), read_json(R0_REPLAY_JSON))
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    """Always re-run the CPU gates so a blocked artifact cannot be silently reused."""
    _check_code_matches_config(load_config())
    audit = audit_inputs()
    replay = replay_r0()
    write_protocol_freeze(audit, replay)
    return 0


# ---------------------------------------------------------------------------
# schedule helpers
# ---------------------------------------------------------------------------


def cosine_lr(step: int, total: int, lr0: float, lr_min: float) -> float:
    """LR used by the optimizer before its (step+1)-th update.

    Matches ``torch.optim.lr_scheduler.CosineAnnealingLR(T_max=total,
    eta_min=lr_min)`` stepped once after every optimizer step (verified in
    tests and in-loop).
    """
    if total <= 0:
        return float(lr0)
    return float(lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * float(step) / float(total))))


def schedule_lr(recipe: Mapping[str, Any], step: int, total: int) -> float:
    if recipe["schedule"] == "constant":
        return float(recipe["lr"])
    if recipe["schedule"] == "cosine":
        return cosine_lr(step, total, float(recipe["lr"]), float(recipe["lr_min"]))
    raise SystemExit(f"unknown schedule {recipe['schedule']!r}")


# ---------------------------------------------------------------------------
# model construction (matched P supervision-phase initialisation replay)
# ---------------------------------------------------------------------------


def _build_matched_model(seed: int, device: torch.device | str) -> tuple[src.BJOC, dict[str, Any]]:
    """Rebuild the exact P supervision-phase initialisation before training.

    Order is identical to ``f10k._train_downstream('P', seed)``:
    ``_make_downstream_model`` (seed_everything + scratch BJOC init) and then
    the frozen SSL encoder state.  All hashes are compared to the recorded P
    round values and to the frozen config.
    """
    cfg = load_config()
    frozen = cfg["frozen"]
    norm = read_json(F10K / "prep_train10k.json")
    model = f10k._make_downstream_model(seed, norm, device)
    init_full = src.state_dict_hash(model)
    init_group = mp.group_state_hash(model)
    init_encoder = f10k._encoder_group_hash(model)
    enc_path = F10K / f"pretrain_encoder_s{seed}.pt"
    mp.load_encoder_state(model, torch.load(enc_path, map_location="cpu", weights_only=False), device)
    encoder_used = f10k._encoder_group_hash(model)
    p_rec = read_json(F10K / f"train_P_seed{seed}.json")
    pre_rec = read_json(F10K / f"pretraining_seed{seed}.json")
    info = {
        "seed": int(seed),
        "init_full_sha256": init_full,
        "init_group_state_sha256": init_group,
        "init_encoder_sha256": init_encoder,
        "encoder_used_sha256": encoder_used,
        "pretrain_encoder_file_sha256": file_sha256(enc_path),
        "scratch_equals_recorded_P_init": bool(init_full == p_rec["init_full_sha256"] == frozen["p_init_full_sha256"][str(seed)]),
        "composer_reader_equals_recorded_P_init": bool(init_group == p_rec["init_group_state_sha256"] == frozen["p_init_group_state_sha256"][str(seed)]),
        "scratch_encoder_equals_recorded_P_init": bool(init_encoder == p_rec["init_encoder_sha256"] == frozen["p_init_encoder_sha256"][str(seed)]),
        "encoder_equals_recorded_P": bool(encoder_used == p_rec["encoder_used_sha256"] == frozen["p_encoder_used_sha256"][str(seed)]),
        "encoder_equals_pretraining_final": bool(encoder_used == pre_rec["encoder_final_sha256"]),
    }
    audit = src.parameter_audit(src.BJOCConfig(), model)
    info["parameter_audit"] = audit
    if not audit["parameter_exact"] or audit["actual_parameters"] != 413505:
        raise SystemExit(f"parameter audit failed: {audit}")
    for key in (
        "scratch_equals_recorded_P_init",
        "composer_reader_equals_recorded_P_init",
        "scratch_encoder_equals_recorded_P_init",
        "encoder_equals_recorded_P",
        "encoder_equals_pretraining_final",
    ):
        if not info[key]:
            raise SystemExit(f"initialisation gate failed: {key}")
    return model, info


# ---------------------------------------------------------------------------
# formal / smoke training
# ---------------------------------------------------------------------------


def train_recipe(
    recipe_key: str,
    seed: int,
    device: torch.device,
    *,
    smoke: bool = False,
    smoke_epochs: int = 2,
    log_fn: Callable[[str], None] = log,
) -> dict[str, Any]:
    verify_freeze()
    cfg = load_config()
    recipe = recipe_of(recipe_key)
    if recipe_key == "R0":
        raise SystemExit("R0 is replay-only; never retrain the historical baseline")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")
    epochs = int(smoke_epochs) if smoke else int(recipe["epochs"])
    if smoke and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    out_dir = SMOKE_DIR if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    total_updates = epochs * STEPS_PER_EPOCH

    train_objs = f10k.load_objects(F10K / "objects_train10k.pt")
    y_fit = f10k._load_y_rows(list(range(N_TRAIN)))
    model, init_info = _build_matched_model(seed, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = None
    if recipe["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"]))
    elif recipe["schedule"] != "constant":
        raise SystemExit(f"unknown schedule {recipe['schedule']!r}")
    optimizer_audit = {
        "optimizer": "Adam",
        "defaults": {k: float(v) if isinstance(v, (int, float)) else v for k, v in optimizer.defaults.items()},
        "n_parameter_groups": len(optimizer.param_groups),
        "optimizer_parameters": int(sum(p.numel() for group in optimizer.param_groups for p in group["params"])),
        "scheduler": None if scheduler is None else type(scheduler).__name__,
        "t_max": None if scheduler is None else total_updates,
        "eta_min": None if scheduler is None else float(recipe["lr_min"]),
    }

    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    prefix_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    nonfinite_loss = False
    nonfinite_grad = False
    total_mol = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in f10k._shuffle_batches(N_TRAIN, BATCH_SIZE, gen):
            batch_arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(batch_arr.tobytes())
            if epoch <= REF_EPOCHS:
                prefix_hash.update(batch_arr.tobytes())
            batch = f10k._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not math.isfinite(gn):
                nonfinite_grad = True
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {recipe_key} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > GRAD_CLIP:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                expected = schedule_lr(recipe, steps + (epoch - 1) * STEPS_PER_EPOCH, total_updates)
                lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        train_mae = total / max(n_mol, 1)
        total_mol += n_mol
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % CURVE_LOG_EVERY == 0 or epoch == 1 or epoch == epochs:
            log_fn(f"[{recipe_key} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            nonfinite_loss = True
            raise RuntimeError(f"non-finite loss at epoch {epoch} {recipe_key} seed {seed}")
    wall = time.perf_counter() - started

    members, soup = f10k._last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = f10k.predict_bjoc(model, train_objs, device)
    soup_path = out_dir / f"soup_{recipe_key}_seed{seed}{suffix}.pt"
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, soup_path)

    full_order_expected = f10k.batch_order_hash(seed, N_TRAIN, BATCH_SIZE, epochs)
    prefix_epochs = min(epochs, REF_EPOCHS)
    prefix_expected = f10k.batch_order_hash(seed, N_TRAIN, BATCH_SIZE, prefix_epochs)
    p_rec = read_json(F10K / f"train_P_seed{seed}.json")
    order_matches_full = order_hash.hexdigest() == full_order_expected
    prefix_matches_p = prefix_hash.hexdigest() == prefix_expected
    prefix_matches_recorded_p = bool(epochs >= REF_EPOCHS and prefix_hash.hexdigest() == p_rec["batch_order_hash"])
    if not order_matches_full:
        raise SystemExit("batch order does not match the independently recomputed order")
    if not (prefix_matches_p and (smoke or prefix_matches_recorded_p)):
        raise SystemExit("batch order prefix does not match the frozen P batch order")

    lr_final_scheduled = float(optimizer.param_groups[0]["lr"]) if scheduler is not None else float(recipe["lr"])
    anomalies = {
        "nonfinite_loss": nonfinite_loss,
        "nonfinite_grad": nonfinite_grad,
        "grad_clip_events": int(clip_events),
        "grad_norm_max": grad_norm_max,
        "lr_schedule_deviation_max": lr_deviation_max,
    }
    payload = {
        "protocol": PROTOCOL_ID,
        "arm": recipe_key,
        "seed": int(seed),
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "definition": cfg["candidates"][recipe_key]["label"],
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": STEPS_PER_EPOCH,
        "soup_rule": f"parameter mean of the last {SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_path": str(soup_path.relative_to(REPO_ROOT)),
        "soup_sha256": file_sha256(soup_path),
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "schedule": {
            "type": recipe["schedule"],
            "per": "step",
            "warmup": "none",
            "lr0": float(recipe["lr"]),
            "lr_min": None if recipe["lr_min"] is None else float(recipe["lr_min"]),
            "t_max_updates": total_updates,
            "lr_final_scheduled": lr_final_scheduled,
            "lr_first_step": curve[0]["lr_first"],
            "lr_last_step": curve[-1]["lr_last"],
            "cosine_formula_max_abs_deviation": lr_deviation_max,
        },
        "hyper": {
            "optimizer": "Adam",
            "lr": float(recipe["lr"]),
            "weight_decay": float(recipe["weight_decay"]),
            "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "loss": "raw-y L1 only",
            "dropout": 0.0,
            "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
            "parameter_groups": {"all": 413505},
        },
        "optimizer_audit": optimizer_audit,
        "init": init_info,
        "parameter_audit": init_info["parameter_audit"],
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "full_hash_independent_recomputation": full_order_expected,
            "full_matches": order_matches_full,
            "prefix_epochs": prefix_epochs,
            "prefix_hash": prefix_hash.hexdigest(),
            "prefix_independent_recomputation": prefix_expected,
            "prefix_matches_independent": prefix_matches_p,
            "prefix_matches_frozen_P": prefix_matches_recorded_p,
        },
        "anomalies": anomalies,
        "lr_curve": curve,
        "wall_seconds": wall,
        "molecules_seen": int(total_mol),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_used": "official train 10000 raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / f"train_{recipe_key}_seed{seed}{suffix}.json", payload)
    log_fn(f"[{recipe_key} s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {wall:.0f}s soup {payload['soup_sha256'][:12]}")
    return payload


def cmd_train(args: argparse.Namespace) -> int:
    train_recipe(args.recipe, int(args.seed), torch.device(args.device), smoke=False)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    train_recipe(args.recipe, int(args.seed), torch.device(args.device), smoke=True, smoke_epochs=int(args.epochs))
    return 0


# ---------------------------------------------------------------------------
# predictions + freeze
# ---------------------------------------------------------------------------


def _checkpoint_path(recipe_key: str, seed: int) -> Path:
    if recipe_key == "R0":
        return F10K / f"soup_P_seed{seed}.pt"
    return OUT / f"soup_{recipe_key}_seed{seed}.pt"


def model_key(recipe_key: str, seed: int) -> str:
    return f"{recipe_key}{seed}"


def prediction_path(key: str) -> Path:
    if key.startswith("P"):
        return F10K / "predictions" / f"valid_{key}.npz"
    if key.startswith("R0"):
        return F10K / "predictions" / f"valid_P{key[-1]}.npz"
    return PRED_DIR / f"valid_{key}.npz"


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze()
    device = torch.device(args.device)
    norm = read_json(F10K / "prep_train10k.json")
    valid_objs = f10k.load_objects(F10K / "objects_valid.pt")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    freeze_existing = read_json(PRED_FREEZE)["prediction_files"] if PRED_FREEZE.exists() else {}
    access = read_json(PRED_ACCESS) if PRED_ACCESS.exists() else {"protocol": PROTOCOL_ID, "entries": {}}
    for seed in [int(s) for s in args.seeds]:
        for recipe_key in args.recipes:
            if recipe_key == "R0":
                raise SystemExit("R0 is replay-only: its frozen reference predictions are never regenerated")
            key = model_key(recipe_key, seed)
            soup_path = _checkpoint_path(recipe_key, seed)
            if not soup_path.exists():
                raise SystemExit(f"missing checkpoint {soup_path}")
            model = f10k._make_downstream_model(seed, norm, device)
            model.load_state_dict(torch.load(soup_path, map_location=device, weights_only=False))
            pred = f10k.predict_bjoc(model, valid_objs, device)
            if not np.isfinite(pred).all():
                raise RuntimeError("non-finite predictions")
            path = prediction_path(key)
            if key in freeze_existing:
                raise SystemExit(f"{key} is already frozen; refusing to overwrite a frozen prediction")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, row=np.arange(N_VALID, dtype=np.int64), pred=pred.astype(np.float32))
            access["entries"][key] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(path),
                "checkpoint_path": str(soup_path.relative_to(REPO_ROOT)),
                "checkpoint_sha256": file_sha256(soup_path),
                "device": str(device),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            log(f"[predict] {key} written ({access['entries'][key]['sha256'][:12]})")
    access["protocol"] = PROTOCOL_ID
    access["git_commit"] = git_commit()
    access["valid_labels_read"] = False
    access["official_test_loaded"] = False
    write_json(PRED_ACCESS, access)
    return 0


def _freeze_file(files: Mapping[str, Path], out_path: Path, *, stage: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for key, path in files.items():
        if not path.exists():
            raise SystemExit(f"missing prediction file {path}")
        with np.load(path) as z:
            if not np.isfinite(z["pred"]).all():
                raise SystemExit(f"non-finite prediction in {path}")
        entries[key] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
    combined = hashlib.sha256()
    for key in sorted(entries):
        combined.update(key.encode())
        combined.update(entries[key]["sha256"].encode())
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "stage": stage,
        "git_commit": git_commit(),
        "n_prediction_files": len(entries),
        "prediction_files": entries,
        "combined_prediction_sha256": combined.hexdigest(),
        "valid_labels_read": bool(VALID_LOCK.exists()),
        "frozen_before_valid_read": stage == "screening",
        "official_test_loaded": False,
    }
    if extra:
        payload.update(extra)
    write_json(out_path, payload)
    log(f"[freeze-predictions:{stage}] {len(entries)} files combined {payload['combined_prediction_sha256'][:16]}")
    return payload


def cmd_freeze_predictions(args: argparse.Namespace) -> int:
    verify_freeze()
    cfg = load_config()
    frozen = cfg["frozen"]
    if args.stage == "screening":
        if VALID_LOCK.exists():
            raise SystemExit("valid labels were already read; refusing to re-freeze screening predictions")
        files = {"R0": prediction_path(model_key("R0", 0))}
        for key in SCREENING_CANDIDATES:
            files[model_key(key, 0)] = prediction_path(model_key(key, 0))
        _freeze_file(files, PRED_FREEZE, stage="screening")
        payload = read_json(PRED_FREEZE)
        if payload["prediction_files"]["R0"]["sha256"] != frozen["valid_P_pred_s0_sha256"]:
            raise SystemExit("R0 reference prediction hash does not match the frozen P record")
        return 0
    if args.stage == "confirmation":
        selected = read_json(SCREENING_JSON)["selected_recipe"]
        if selected is None:
            raise SystemExit("no candidate was selected; nothing to confirm")
        _freeze_file(
            {
                model_key(selected, 0): prediction_path(model_key(selected, 0)),
                model_key(selected, 1): prediction_path(model_key(selected, 1)),
                "P1": prediction_path("P1"),
            },
            CONFIRM_FREEZE,
            stage="confirmation",
            extra={"selected_recipe": selected, "conditional_on_screening": True},
        )
        payload = read_json(CONFIRM_FREEZE)
        if payload["prediction_files"]["P1"]["sha256"] != frozen["valid_P_pred_s1_sha256"]:
            raise SystemExit("P1 reference prediction hash does not match the frozen P record")
        if payload["prediction_files"][model_key(selected, 0)]["sha256"] != read_json(PRED_FREEZE)["prediction_files"][model_key(selected, 0)]["sha256"]:
            raise SystemExit("selected-candidate seed-0 prediction changed between screening and confirmation")
        return 0
    raise SystemExit(f"unknown freeze stage {args.stage!r}")


# ---------------------------------------------------------------------------
# the ONE official-valid label read + score
# ---------------------------------------------------------------------------


def _load_pred(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return z["pred"].astype(np.float64)


def _contrast(preds_new: Sequence[np.ndarray], preds_ref: Sequence[np.ndarray], y: np.ndarray, tag: str, seeds: Sequence[int]) -> dict[str, Any]:
    e_new = np.mean([np.abs(p - y) for p in preds_new], axis=0)
    e_ref = np.mean([np.abs(p - y) for p in preds_ref], axis=0)
    delta = e_new - e_ref
    out = {
        "tag": tag,
        "delta_seed_mean": float(delta.mean()),
        "seed_mean_mae_new": float(np.mean([np.abs(p - y).mean() for p in preds_new])),
        "seed_mean_mae_ref": float(np.mean([np.abs(p - y).mean() for p in preds_ref])),
        "per_seed_delta": {
            f"seed{s}": float(np.abs(preds_new[i] - y).mean() - np.abs(preds_ref[i] - y).mean()) for i, s in enumerate(seeds)
        },
        "per_seed_mae_new": {f"seed{s}": float(np.abs(preds_new[i] - y).mean()) for i, s in enumerate(seeds)},
        "per_seed_mae_ref": {f"seed{s}": float(np.abs(preds_ref[i] - y).mean()) for i, s in enumerate(seeds)},
        "seed_mean_rmse_new": float(np.mean([np.sqrt(np.mean((p - y) ** 2)) for p in preds_new])),
        "seed_mean_rmse_ref": float(np.mean([np.sqrt(np.mean((p - y) ** 2)) for p in preds_ref])),
        "bootstrap": f10k.paired_bootstrap(delta),
        "ordinary_y_gt_-3": {"n": int((y > TAIL_THRESHOLD).sum()), "delta": float(delta[y > TAIL_THRESHOLD].mean())},
        "tail_y_le_-3": {"n": int((y <= TAIL_THRESHOLD).sum()), "delta": float(delta[y <= TAIL_THRESHOLD].mean())},
        "better_fraction": float((delta < 0).mean()),
        "better_fraction_ordinary": float((delta[y > TAIL_THRESHOLD] < 0).mean()),
    }
    for k in TOP_K_REMOVED:
        out.update(f10k.top_k_removed(delta, e_new, e_ref, k))
    extreme = EXTREME_VALID_ROW
    out["extreme_row"] = {
        "row": f"valid:{extreme:04d}",
        "y": float(y[extreme]),
        "delta_row": float(delta[extreme]),
        "delta_without_row": float(np.delete(delta, extreme).mean()),
        "share_of_total_delta": float(delta[extreme] / N_VALID / delta.mean()) if delta.mean() != 0 else None,
    }
    return out


def _read_valid_labels_once() -> np.ndarray:
    if VALID_LOCK.exists():
        raise SystemExit("official-valid labels already read once; refusing a second read")
    val_ds = f10k.load_valid_dataset()
    y = np.asarray([float(val_ds[i].y.reshape(-1)[0]) for i in range(N_VALID)], dtype=np.float64)
    np.save(VALID_Y_COPY, y)
    enc = torch.load(f10k.ENCODED_VALID, map_location="cpu", weights_only=False)
    y_enc = np.asarray([float(enc[i].y.reshape(-1)[0]) for i in range(N_VALID)], dtype=np.float64)
    del enc
    y_crosscheck = float(np.abs(y - y_enc).max())
    if y_crosscheck != 0.0:
        raise SystemExit("raw official-valid y and the encoded-valid cache disagree")
    VALID_LOCK.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_ID,
                "git_commit": git_commit(),
                "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_valid": N_VALID,
                "y_sha256": f10k.array_sha256(y),
                "y_raw_vs_encoded_max_abs": y_crosscheck,
                "prediction_freeze_sha256": file_sha256(PRED_FREEZE),
                "extreme_row": f"valid:{EXTREME_VALID_ROW:04d}",
                "extreme_row_y": float(y[EXTREME_VALID_ROW]),
                "official_test_loaded": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return y


def cmd_score_screening(args: argparse.Namespace) -> int:
    verify_freeze()
    freeze = read_json(PRED_FREEZE)
    for key, entry in freeze["prediction_files"].items():
        if file_sha256(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"frozen prediction hash mismatch: {key}")
    if not freeze["frozen_before_valid_read"]:
        raise SystemExit("screening prediction freeze is not marked pre-label")
    if VALID_LOCK.exists():
        # the single label read already happened; re-scoring the frozen files is allowed
        y = np.load(VALID_Y_COPY)
        if f10k.array_sha256(y) != read_json(VALID_LOCK)["y_sha256"]:
            raise SystemExit("cached valid y hash mismatch")
    else:
        y = _read_valid_labels_once()
    preds = {key: _load_pred(REPO_ROOT / entry["path"]) for key, entry in freeze["prediction_files"].items()}

    models = {}
    for key, pred in preds.items():
        models[key] = f10k.row_metrics(pred, y)
    contrasts = {}
    for key in SCREENING_CANDIDATES:
        ckey = model_key(key, 0)
        contrasts[f"{key}_minus_R0"] = _contrast([preds[ckey]], [preds["R0"]], y, f"{ckey}-R00", seeds=[0])
    fit_mae = {}
    for key in SCREENING_CANDIDATES:
        rec = read_json(OUT / f"train_{key}_seed0.json")
        fit_mae[key] = rec["soup_fit_mae"]
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_valid": N_VALID,
        "models": models,
        "pairwise": contrasts,
        "fit_mae": fit_mae,
        "y_raw_vs_encoded_max_abs": 0.0,
        "single_read": True,
        "official_test_loaded": False,
    }
    write_json(SCREENING_METRICS, payload)
    for key in SCREENING_CANDIDATES:
        c = contrasts[f"{key}_minus_R0"]
        log(f"[score-screening] {key}: valid MAE {c['seed_mean_mae_new']:.6f} delta {c['delta_seed_mean']:+.6f} ordinary {c['ordinary_y_gt_-3']['delta']:+.6f}")
    return 0


def _finite_and_audited(recipe_key: str, seed: int) -> tuple[bool, list[str]]:
    path = OUT / f"train_{recipe_key}_seed{seed}.json"
    if not path.exists():
        return False, [f"missing train record {path.name}"]
    rec = read_json(path)
    reasons: list[str] = []
    anomalies = rec["anomalies"]
    if anomalies["nonfinite_loss"] or anomalies["nonfinite_grad"]:
        reasons.append("non-finite loss/gradient")
    if not rec["batch_order"]["full_matches"] or not rec["batch_order"]["prefix_matches_frozen_P"]:
        reasons.append("batch order audit failed")
    init = rec["init"]
    for gate in (
        "scratch_equals_recorded_P_init",
        "composer_reader_equals_recorded_P_init",
        "scratch_encoder_equals_recorded_P_init",
        "encoder_equals_recorded_P",
        "encoder_equals_pretraining_final",
    ):
        if not init[gate]:
            reasons.append(f"init gate failed: {gate}")
    if not rec["parameter_audit"]["parameter_exact"] or rec["parameter_audit"]["actual_parameters"] != 413505:
        reasons.append("parameter audit failed")
    return (not reasons), reasons


def screen_gates(contrast: Mapping[str, Any], audit_ok: bool) -> dict[str, bool]:
    """Pre-registered seed-0 screening gate (screening.gate in the config)."""
    return {
        "seed0_mae_delta_le_-0.003": bool(contrast["delta_seed_mean"] <= SCREEN_GAIN_THRESHOLD),
        "ordinary_not_worse": bool(contrast["ordinary_y_gt_-3"]["delta"] <= 0.0),
        "extreme_row_excluded_delta_le_-0.002": bool(contrast["extreme_row"]["delta_without_row"] <= SCREEN_EXTREME_EXCLUDED_THRESHOLD),
        "numerics_and_audits_ok": bool(audit_ok),
    }


def confirmation_conditions(contrast: Mapping[str, Any], audit_ok: bool) -> dict[str, bool]:
    """Pre-registered two-seed confirmation gate (confirmation.final_gate)."""
    return {
        "both_seed_deltas_negative": all(v < 0 for v in contrast["per_seed_delta"].values()),
        "seed_mean_delta_le_-0.003": bool(contrast["delta_seed_mean"] <= CONFIRM_GAIN_THRESHOLD),
        "ordinary_seed_mean_delta_le_0": bool(contrast["ordinary_y_gt_-3"]["delta"] <= 0.0),
        "top10_removed_delta_negative": bool(contrast["delta_top10_removed"] < 0),
        "extreme_row_excluded_delta_negative": bool(contrast["extreme_row"]["delta_without_row"] < 0),
        "numerics_and_audits_ok": bool(audit_ok),
    }


def classify_confirmation(conditions: Mapping[str, bool], contrast: Mapping[str, Any]) -> str:
    """Map the frozen gate + CI to the decision vocabulary."""
    if all(conditions.values()):
        return "RECIPE_GAIN_CANDIDATE" if contrast["bootstrap"]["ci_hi"] < 0 else "WEAK_OR_INCONCLUSIVE"
    if contrast["delta_seed_mean"] < 0:
        return "WEAK_OR_INCONCLUSIVE"
    return "NO_ROBUST_RECIPE_GAIN"


def cmd_screen(args: argparse.Namespace) -> int:
    metrics = read_json(SCREENING_METRICS)
    per_candidate: dict[str, Any] = {}
    passed: list[str] = []
    for key in SCREENING_CANDIDATES:
        c = metrics["pairwise"][f"{key}_minus_R0"]
        audit_ok, audit_reasons = _finite_and_audited(key, 0)
        gates = screen_gates(c, audit_ok)
        ok = all(gates.values())
        if ok:
            passed.append(key)
        per_candidate[key] = {
            "gates": gates,
            "passed": ok,
            "audit_reasons": audit_reasons,
            "seed0_valid_mae": c["seed_mean_mae_new"],
            "seed0_delta": c["delta_seed_mean"],
            "ordinary_delta": c["ordinary_y_gt_-3"]["delta"],
            "tail_delta": c["tail_y_le_-3"]["delta"],
            "rmse_new": c["seed_mean_rmse_new"],
            "rmse_ref": c["seed_mean_rmse_ref"],
            "delta_top10_removed": c["delta_top10_removed"],
            "extreme_row": c["extreme_row"],
            "bootstrap": c["bootstrap"],
            "updates": CANDIDATES[key]["epochs"] * STEPS_PER_EPOCH,
        }
    if passed:
        ordered = sorted(passed, key=lambda k: (per_candidate[k]["seed0_valid_mae"], per_candidate[k]["updates"]))
        selected = ordered[0]
        selection_rule = "lowest seed-0 valid MAE among passers; tie -> fewer updates"
    else:
        selected = None
        selection_rule = "none passed; no purchase"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "thresholds": {
            "gain": SCREEN_GAIN_THRESHOLD,
            "extreme_excluded": SCREEN_EXTREME_EXCLUDED_THRESHOLD,
        },
        "per_candidate": per_candidate,
        "passed_candidates": passed,
        "selected_recipe": selected,
        "selection_rule": selection_rule,
        "official_test_loaded": False,
    }
    write_json(SCREENING_JSON, payload)
    log(f"[screen] passed={passed} selected={selected}")
    return 0


def cmd_score_confirmation(args: argparse.Namespace) -> int:
    if not VALID_LOCK.exists():
        raise SystemExit("run score-screening first (the single label read)")
    freeze = read_json(CONFIRM_FREEZE)
    for key, entry in freeze["prediction_files"].items():
        if file_sha256(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"confirmation prediction hash mismatch: {key}")
    selected = freeze["selected_recipe"]
    y = np.load(VALID_Y_COPY)
    if f10k.array_sha256(y) != read_json(VALID_LOCK)["y_sha256"]:
        raise SystemExit("cached valid y hash mismatch")
    preds = {key: _load_pred(REPO_ROOT / entry["path"]) for key, entry in freeze["prediction_files"].items()}
    p0 = _load_pred(F10K / "predictions" / "valid_P0.npz")
    keys_new = [model_key(selected, 0), model_key(selected, 1)]
    keys_ref = ["P0", "P1"]
    preds["P0"] = p0
    contrast = _contrast([preds[k] for k in keys_new], [preds[k] for k in keys_ref], y, f"{selected}-P", seeds=[0, 1])
    contrast["seed0_vs_screening_reference_R0"] = _contrast(
        [preds[model_key(selected, 0)]], [p0], y, f"{selected}0-P0", seeds=[0]
    )
    models = {k: f10k.row_metrics(preds[k], y) for k in preds}
    fit_mae = {}
    for seed in SEEDS:
        rec = read_json(OUT / f"train_{selected}_seed{seed}.json")
        fit_mae[f"seed{seed}"] = rec["soup_fit_mae"]
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "selected_recipe": selected,
        "contrast": contrast,
        "models": models,
        "fit_mae": fit_mae,
        "label_read_was_single_and_earlier": True,
        "conditional_on_screening": True,
        "official_test_loaded": False,
    }
    write_json(CONFIRMATION_METRICS, payload)
    log(f"[score-confirmation] {selected}: seed deltas {contrast['per_seed_delta']} seed-mean {contrast['delta_seed_mean']:+.6f}")
    return 0


def cmd_decision(args: argparse.Namespace) -> int:
    screening = read_json(SCREENING_JSON)
    selected = screening["selected_recipe"]
    details: dict[str, Any] = {
        "screening_thresholds": screening["thresholds"],
        "candidates": {k: {"passed": v["passed"], "seed0_delta": v["seed0_delta"], "seed0_valid_mae": v["seed0_valid_mae"]} for k, v in screening["per_candidate"].items()},
    }
    flags: list[str] = []
    if selected is None:
        branch = "NO_ROBUST_RECIPE_GAIN"
        details["reason"] = "no candidate passed the pre-registered seed-0 screening gate; no confirmation purchased"
    else:
        conf = read_json(CONFIRMATION_METRICS)["contrast"]
        audit_ok, audit_reasons = _finite_and_audited(selected, 1)
        conditions = confirmation_conditions(conf, audit_ok)
        branch = classify_confirmation(conditions, conf)
        if branch == "WEAK_OR_INCONCLUSIVE":
            if all(conditions.values()):
                flags.append("BOOTSTRAP_CI_CROSSES_ZERO")
            else:
                flags.append("SEED_MEAN_NEGATIVE_BUT_GATE_INCOMPLETE")
        elif branch == "NO_ROBUST_RECIPE_GAIN" and not all(conditions.values()):
            flags.append("CONFIRMATION_GATE_FAILED")
        tail_worse = conf["tail_y_le_-3"]["delta"] > 0
        rmse_worse = conf["seed_mean_rmse_new"] > conf["seed_mean_rmse_ref"]
        if tail_worse:
            flags.append("TAIL_WORSE")
        if rmse_worse:
            flags.append("RMSE_WORSE")
        details.update(
            {
                "selected_recipe": selected,
                "confirmation": conf,
                "conditions": conditions,
                "audit_reasons": audit_reasons,
            }
        )
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "branch": branch,
        "flags": sorted(set(flags)),
        "details": details,
        "purchased_seed1": selected is not None,
        "answers": {
            "q1_r1_vs_r0_240ep_cosine": screening["per_candidate"]["R1"]["seed0_delta"],
            "q2_r2_vs_r0_more_updates": screening["per_candidate"]["R2"]["seed0_delta"],
            "q3_r3_vs_r2_cosine_at_360": screening["per_candidate"]["R3"]["seed0_delta"],
            "q4_r4_vs_r0_more_weight_decay": screening["per_candidate"]["R4"]["seed0_delta"],
            "official_test_loaded": False,
        },
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    write_json(SCIENTIFIC_DECISION, payload)
    log(f"[decision] {branch} flags={payload['flags']}")
    return 0


# ---------------------------------------------------------------------------
# descriptive analysis (post-read only)
# ---------------------------------------------------------------------------


def cmd_analysis(args: argparse.Namespace) -> int:
    if not VALID_LOCK.exists():
        raise SystemExit("run score-screening first")
    y = np.load(VALID_Y_COPY)
    assert f10k.array_sha256(y) == read_json(VALID_LOCK)["y_sha256"]
    valid_objs = f10k.load_objects(F10K / "objects_valid.pt")
    screening = read_json(SCREENING_JSON)
    selected = screening["selected_recipe"]

    freeze = read_json(PRED_FREEZE)
    preds = {key: _load_pred(REPO_ROOT / entry["path"]) for key, entry in freeze["prediction_files"].items()}
    if CONFIRM_FREEZE.exists():
        cfreeze = read_json(CONFIRM_FREEZE)
        for key, entry in cfreeze["prediction_files"].items():
            preds[key] = _load_pred(REPO_ROOT / entry["path"])

    # structure strata (label-free features only)
    junc = np.asarray([o.junc_index.shape[0] for o in valid_objs])
    n_atoms = np.asarray([o.n_atoms for o in valid_objs])
    diameter = np.asarray([float(o.global_feat[sro.GLOBAL_FEAT_NAMES.index("diameter")]) for o in valid_objs])
    masks = {
        "ordinary_y_gt_-3": y > TAIL_THRESHOLD,
        "tail_y_le_-3": y <= TAIL_THRESHOLD,
        "junction_rich": junc > np.median(junc),
        "atom_rich": n_atoms > np.median(n_atoms),
        "large_diameter": diameter > np.median(diameter),
    }
    per_recipe: dict[str, Any] = {}
    model_keys = [f"R{k}0" for k in range(1, 5)]
    for mk in model_keys:
        if mk not in preds:
            continue
        new_pred = preds[mk]
        ref_pred = preds["R0"] if "R0" in preds else preds["P0"]
        delta = np.abs(new_pred - y) - np.abs(ref_pred - y)
        rec = read_json(OUT / f"train_{mk[0:2]}_seed0.json")
        entry: dict[str, Any] = {
            "valid_mae": float(np.abs(new_pred - y).mean()),
            "fit_mae": rec["soup_fit_mae"],
            "fit_minus_valid": rec["soup_fit_mae"] - float(np.abs(new_pred - y).mean()),
            "delta_vs_R0": float(delta.mean()),
            "strata": {name: {"n": int(m.sum()), "delta": float(delta[m].mean())} for name, m in masks.items()},
        }
        per_recipe[mk] = entry
    if selected is not None and CONFIRMATION_METRICS.exists():
        conf = read_json(CONFIRMATION_METRICS)
        c = conf["contrast"]
        per_recipe[f"{selected}(confirm)"] = {
            "seed0_delta": c["per_seed_delta"]["seed0"],
            "seed1_delta": c["per_seed_delta"]["seed1"],
            "seed_mean_delta": c["delta_seed_mean"],
            "seed_mean_rmse_new": c["seed_mean_rmse_new"],
            "seed_mean_rmse_ref": c["seed_mean_rmse_ref"],
            "ordinary_delta": c["ordinary_y_gt_-3"]["delta"],
            "tail_delta": c["tail_y_le_-3"]["delta"],
            "delta_top10_removed": c["delta_top10_removed"],
            "extreme_row": c["extreme_row"],
            "bootstrap": c["bootstrap"],
            "fit_mae": conf["fit_mae"],
        }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "per_recipe": per_recipe,
        "r0_reference": {"seed0_mae": float(np.abs(preds["R0"] - y).mean()) if "R0" in preds else None},
        "selected_recipe": selected,
        "official_test_loaded": False,
    }
    write_json(ANALYSIS_JSON, payload)
    log(f"[analysis] wrote {ANALYSIS_JSON}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-inputs").set_defaults(func=cmd_verify_inputs)
    sub.add_parser("replay-r0").set_defaults(func=cmd_replay_r0)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    sub.add_parser("prep").set_defaults(func=cmd_prep)

    p = sub.add_parser("smoke")
    p.add_argument("--recipe", choices=[k for k in CANDIDATES if k != "R0"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("train")
    p.add_argument("--recipe", choices=[k for k in CANDIDATES if k != "R0"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--recipes", nargs="+", choices=list(CANDIDATES), required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=["0"])
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("freeze-predictions")
    p.add_argument("--stage", choices=["screening", "confirmation"], required=True)
    p.set_defaults(func=cmd_freeze_predictions)

    sub.add_parser("score-screening").set_defaults(func=cmd_score_screening)
    sub.add_parser("screen").set_defaults(func=cmd_screen)
    sub.add_parser("score-confirmation").set_defaults(func=cmd_score_confirmation)
    sub.add_parser("decision").set_defaults(func=cmd_decision)
    sub.add_parser("analysis").set_defaults(func=cmd_analysis)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
