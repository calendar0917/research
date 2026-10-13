"""``cscl-bjoc-r3-frozen-terminal-test-v1`` driver.

Authorized terminal evaluation of the frozen full-10k P3 checkpoints
(masked-SSL + R3 360-epoch cosine, BJOC-Null, 413,505 params) on the official
ZINC-subset test split, with the historical full-10k P0 checkpoints as the only
primary control.

Scope discipline
----------------
* Inference, auditing, scoring and reporting only.  No training, no tuning, no
  model selection, no test-driven follow-up.
* The official test instrument was already exposed in this project (KSVD track
  2026-10-03 and ``cscl-bjoc-pretraining-mechanism-terminal-test-v1``).  This is
  an *exposed-test terminal evaluation*, not a first blind test.
* All predictions are written and hashed before the single test-label read of
  this round.  The previous round's frozen files (freeze JSON, lock, y copy,
  metrics) are opened read-only and are never overwritten.
* The previous round's scoring state is not re-run: this driver refuses to touch
  its outputs and writes everything under ``results/bjoc_r3_frozen_terminal_test_v1``.

Reused verified implementation (imported from the previous terminal test):
``LabelGuard``, ``build_test_objects``, ``predict_test_objs``, ``objects_hash``,
``objects_signature``, ``load_test_dataset``, ``_test_structure_fingerprint``,
``_label_storage``, ``_clear_slice_cache``; the model/replay and metric helpers
come from ``run_bjoc_full10k_pretraining_transfer_v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import run_bjoc_pretraining_mechanism_terminal_test_v1 as t0  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-r3-frozen-terminal-test-v1"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_r3_frozen_terminal_test_v1"
REC = REPO_ROOT / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1"
F10K = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
T0 = REPO_ROOT / "tracks/structure-contribution/results/bjoc_pretraining_mechanism_terminal_test_v1"

N_TEST = 1000
KEYS = ("P00", "P01", "P30", "P31")
SEED_OF = {"P00": 0, "P01": 1, "P30": 0, "P31": 1}
ARM_OF = {"P00": "P0", "P01": "P0", "P30": "P3", "P31": "P3"}

SOUPS = {
    "P00": F10K / "soup_P_seed0.pt",
    "P01": F10K / "soup_P_seed1.pt",
    "P30": REC / "soup_R3_seed0.pt",
    "P31": REC / "soup_R3_seed1.pt",
}

#: recorded soup hashes and their provenance files (read-only)
RECORDED_SOUP_SHA = {
    "P00": ("73118e9be466c62a252eacb83d9c86d1cb4bf0ce01478db2f278b7c27a5fab2a", F10K / "prediction_freeze.json", ("checkpoints", "soup_P_seed0")),
    "P01": ("bf640cd8c11a9f6e0d11ceccd318185e6067ef2ba5a950b4774d5185a9ab053c", F10K / "prediction_freeze.json", ("checkpoints", "soup_P_seed1")),
    "P30": (None, REC / "train_R3_seed0.json", ("soup_sha256",)),
    "P31": (None, REC / "train_R3_seed1.json", ("soup_sha256",)),
}

HIST_VALID_MAE = {"P00": 0.084786, "P01": 0.082313, "P30": 0.079101, "P31": 0.080586}
HIST_VALID_NPZ = {
    "P00": F10K / "predictions" / "valid_P0.npz",
    "P01": F10K / "predictions" / "valid_P1.npz",
    "P30": REC / "predictions" / "valid_R30.npz",
    "P31": REC / "predictions" / "valid_R31.npz",
}
HIST_TEST_MAE = {"P00": 0.074825, "P01": 0.073088, "P0_seed_mean": 0.073957}
#: previous round's frozen test prediction files (read-only control copies)
T0_TEST_PRED = {"P00": T0 / "predictions" / "test_P0.npz", "P01": T0 / "predictions" / "test_P1.npz"}

TAIL_THRESHOLD = -3.0
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
REPLAY_NPZ_ATOL = 5.0e-5
REPLAY_MAE_ATOL = 1.0e-5
TARGET_LO, TARGET_HI = 0.06, 0.07
ENSEMBLE_NOTE = "seed-mean is the arithmetic mean of the two single-model metrics; predictions are never averaged"

write_json = f10k.write_json
read_json = f10k.read_json
file_sha256 = f10k.file_sha256
git_commit = f10k.git_commit
array_sha256 = f10k.array_sha256
rows_sha256 = f10k.rows_sha256
paired_bootstrap = f10k.paired_bootstrap
row_metrics = f10k.row_metrics


def log(msg: str) -> None:
    print(msg, flush=True)


def _code_hashes() -> dict[str, str]:
    return {
        "driver": file_sha256(Path(__file__).resolve()),
        "terminal_test_driver": file_sha256(_CODE_DIR / "run_bjoc_pretraining_mechanism_terminal_test_v1.py"),
        "previous_round_driver": file_sha256(_CODE_DIR / "run_bjoc_full10k_pretraining_transfer_v1.py"),
        "model": file_sha256(_CODE_DIR / "structural_relation_composer_v1.py"),
        "objects": file_sha256(_CODE_DIR / "structural_relation_objects_v1.py"),
        "masked_module": file_sha256(_CODE_DIR / "bjoc_masked_pretraining_v1.py"),
    }


def _environment() -> dict[str, Any]:
    return {
        "device": "cpu",
        "torch": torch.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_used": False,
    }


def _norm() -> dict[str, Any]:
    return read_json(F10K / "prep_train10k.json")


def load_model(key: str, device: torch.device | str = "cpu") -> src.BJOC:
    model = f10k._make_downstream_model(SEED_OF[key], _norm(), device)
    model.load_state_dict(torch.load(SOUPS[key], map_location=device, weights_only=False))
    model.eval()
    return model


def _recorded_hash(key: str) -> tuple[str, str]:
    """Return (recorded_sha256, source_description) for one soup."""
    _unused, path, where = RECORDED_SOUP_SHA[key]
    doc = read_json(path)
    node: Any = doc
    for part in where:
        node = node[part]
    return str(node), f"{path.relative_to(REPO_ROOT)}::{'.'.join(where)}"


# ---------------------------------------------------------------------------
# Step 1 -- audit (no test label access)
# ---------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = t0.load_test_dataset()
    fingerprint = t0._test_structure_fingerprint(ds)
    hist_freeze = read_json(T0 / "terminal_test_freeze.json")
    hist_fp = hist_freeze["structure_fingerprint"]
    fingerprint_checks = {
        "n_molecules": fingerprint["n_molecules"] == hist_fp["n_molecules"] == N_TEST,
        "processed_pt_sha256": fingerprint["processed_pt_sha256"] == hist_fp["processed_pt_sha256"],
        "structure_sha256": fingerprint["structure_sha256"] == hist_fp["structure_sha256"],
        "rows_sha256": fingerprint["rows_sha256"] == hist_fp["rows_sha256"],
    }

    valid_objs = f10k.load_objects(F10K / "objects_valid.pt")
    y_valid = np.asarray(np.load(F10K / "valid_y_read_once.npy"), dtype=np.float64)
    model_records: dict[str, Any] = {}
    valid_preds: dict[str, np.ndarray] = {}
    for key in KEYS:
        soup = SOUPS[key]
        sha = file_sha256(soup)
        recorded, recorded_src = _recorded_hash(key)
        model = load_model(key)
        audit = src.parameter_audit(src.BJOCConfig(), model)
        forbidden = src.forbidden_module_scan(model)
        head_keys = [k for k in model.state_dict() if k.startswith(("atom_head", "bond_head", "pretext"))]
        pred = t0.predict_objs(model, valid_objs, "cpu")
        ref = np.asarray(np.load(HIST_VALID_NPZ[key])["pred"], dtype=np.float64)
        metrics = row_metrics(pred, y_valid)
        valid_preds[key] = pred
        model_records[key] = {
            "arm": ARM_OF[key],
            "seed": SEED_OF[key],
            "checkpoint": str(soup.relative_to(REPO_ROOT)),
            "checkpoint_exists": soup.exists(),
            "checkpoint_sha256": sha,
            "recorded_sha256": recorded,
            "recorded_sha256_source": recorded_src,
            "checkpoint_sha256_matches_record": sha == recorded,
            "parameters": int(audit["actual_parameters"]),
            "parameter_exact": bool(audit["parameter_exact"]),
            "parameters_413505": int(audit["actual_parameters"]) == 413505,
            "forbidden_modules": forbidden,
            "pretext_head_keys_in_state": head_keys,
            "valid_mae": metrics["mae"],
            "valid_rmse": metrics["rmse"],
            "historical_valid_mae": HIST_VALID_MAE[key],
            "valid_mae_diff_vs_historical": float(metrics["mae"] - HIST_VALID_MAE[key]),
            "valid_replay_max_abs_pred_diff_vs_frozen_npz": float(np.abs(pred - ref).max()),
        }
        log(
            f"[audit] {key} params={audit['actual_parameters']} valid MAE {metrics['mae']:.6f} "
            f"(historical {HIST_VALID_MAE[key]:.6f}) sha_ok={sha == recorded} "
            f"replay_maxdiff={model_records[key]['valid_replay_max_abs_pred_diff_vs_frozen_npz']:.2e}"
        )

    valid_seed_mean = {
        arm: float(np.mean([model_records[k]["valid_mae"] for k in KEYS if ARM_OF[k] == arm])) for arm in ("P0", "P3")
    }
    normaliser = {
        "file": str((F10K / "prep_train10k.json").relative_to(REPO_ROOT)),
        "sha256": file_sha256(F10K / "prep_train10k.json"),
        "fit_only": bool(_norm().get("fit_only")),
        "n_fit": int(_norm().get("n_fit", -1)),
        "official_valid_statistics_used": bool(_norm().get("official_valid_statistics_used")),
        "official_test_statistics_used": bool(_norm().get("official_test_statistics_used")),
    }
    conditions = {
        "all_checkpoints_exist": all(r["checkpoint_exists"] for r in model_records.values()),
        "all_checkpoint_hashes_match_records": all(r["checkpoint_sha256_matches_record"] for r in model_records.values()),
        "all_413505_parameters": all(r["parameters_413505"] and r["parameter_exact"] for r in model_records.values()),
        "no_forbidden_modules": all(not r["forbidden_modules"] for r in model_records.values()),
        "no_pretext_heads_in_state": all(not r["pretext_head_keys_in_state"] for r in model_records.values()),
        "valid_replay_exact_enough": all(r["valid_replay_max_abs_pred_diff_vs_frozen_npz"] <= REPLAY_NPZ_ATOL for r in model_records.values()),
        "valid_mae_matches_historical": all(abs(r["valid_mae_diff_vs_historical"]) <= REPLAY_MAE_ATOL for r in model_records.values()),
        "test_fingerprint_matches_history": all(fingerprint_checks.values()),
        "normaliser_train_only": normaliser["fit_only"] and normaliser["n_fit"] == 10000 and not normaliser["official_valid_statistics_used"] and not normaliser["official_test_statistics_used"],
    }
    decision = "READY_FOR_TEST_PREDICTION" if all(conditions.values()) else "AUDIT_BLOCKED"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "audit of the frozen P3 target checkpoints and the frozen P0 control before any test prediction",
        "models": model_records,
        "valid_seed_mean": valid_seed_mean,
        "test_structure_fingerprint": fingerprint,
        "test_structure_fingerprint_checks_vs_history": fingerprint_checks,
        "normaliser": normaliser,
        "environment": _environment(),
        "code_hashes": _code_hashes(),
        "conditions": conditions,
        "decision": decision,
        "official_test_structures_loaded": True,
        "official_test_labels_read": False,
        "no_training": True,
    }
    write_json(OUT / "checkpoint_audit.json", payload)
    np.savez_compressed(OUT / "valid_preds_audit.npz", **{k: v.astype(np.float32) for k, v in valid_preds.items()})
    log(f"[audit] P0 valid seed-mean {valid_seed_mean['P0']:.6f} P3 {valid_seed_mean['P3']:.6f} -> {decision}")
    if decision != "READY_FOR_TEST_PREDICTION":
        raise SystemExit("audit failed; test prediction is blocked")
    return 0


# ---------------------------------------------------------------------------
# Step 2 -- pre-registration (written before any test label read)
# ---------------------------------------------------------------------------


def cmd_test_protocol(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "test_protocol.json").exists():
        raise SystemExit("test protocol already written; refusing to rewrite the preregistration")
    audit = read_json(OUT / "checkpoint_audit.json")
    if audit["decision"] != "READY_FOR_TEST_PREDICTION":
        raise SystemExit("run and pass the audit first")
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authorization": "user-authorized terminal evaluation of the frozen P3 model on the official test split of this round",
        "target_models": {
            key: {
                "checkpoint": str(SOUPS[key].relative_to(REPO_ROOT)),
                "checkpoint_sha256": file_sha256(SOUPS[key]),
                "role": "target",
                "description": "full-10k masked-SSL encoder + R3 360-epoch cosine last-5 soup, BJOC-Null 413,505 params",
            }
            for key in ("P30", "P31")
        },
        "control_models": {
            key: {
                "checkpoint": str(SOUPS[key].relative_to(REPO_ROOT)),
                "checkpoint_sha256": file_sha256(SOUPS[key]),
                "role": "historical primary control",
                "description": "full-10k masked-SSL encoder + R0 240-epoch constant last-5 soup, identical architecture/init/order",
            }
            for key in ("P00", "P01")
        },
        "input": "official PyG ZINC subset=True split=test processed split (1000 molecules); labels guarded until the prediction freeze",
        "preprocessing": {
            "file": "results/bjoc_full10k_pretraining_transfer_v1/prep_train10k.json",
            "sha256": file_sha256(F10K / "prep_train10k.json"),
            "fit": "official train 10000 only; no valid/test statistics",
        },
        "primary_metric": "raw-y single-model MAE on the official test split; P3 seed-mean = arithmetic mean of the P3 seed-0 and seed-1 single-model MAEs",
        "primary_contrast": "paired per-molecule delta d_i = 0.5*[(|e_P3s0|-|e_P0s0|) + (|e_P3s1|-|e_P0s1|)]; 95% bootstrap CI, 5000 resamples, seed 0, unit molecule",
        "secondary_metrics": ["rmse", "median_abs", "p90_abs", "ordinary y>-3", "tail y<=-3", "top1/top5/top10 removed", "better fraction", "per-seed MAE", "P3-B0 context"],
        "p0_replay_expectation": {
            "P00_mae": HIST_TEST_MAE["P00"],
            "P01_mae": HIST_TEST_MAE["P01"],
            "P0_seed_mean_mae": HIST_TEST_MAE["P0_seed_mean"],
            "source": "results/bjoc_pretraining_mechanism_terminal_test_v1/official_test_metrics.json",
            "tolerance_abs": REPLAY_MAE_ATOL,
            "npz_max_abs_tolerance": REPLAY_NPZ_ATOL,
            "action_if_mismatch": "STOP: report the blocker and do not claim the P3 test number is comparable",
        },
        "target_band": {"band": [TARGET_LO, TARGET_HI], "judged_on": "P3 seed-mean single-model official-test MAE"},
        "decision_vocabulary": [
            "P3_TEST_IMPROVES",
            "P3_TEST_WORSE",
            "P3_TEST_INCONCLUSIVE_OR_MIXED",
        ],
        "decision_rule": {
            "P3_TEST_IMPROVES": "both per-seed deltas < 0 and paired bootstrap CI95 upper < 0",
            "P3_TEST_WORSE": "both per-seed deltas > 0 and paired bootstrap CI95 lower > 0",
            "P3_TEST_INCONCLUSIVE_OR_MIXED": "otherwise",
        },
        "exposure_disclosure": "official test was previously read in this project (KSVD track 2026-10-03) and by cscl-bjoc-pretraining-mechanism-terminal-test-v1; this is an exposed-test terminal evaluation, not a first blind test",
        "no_training": True,
        "no_tuning": True,
        "no_model_selection": True,
        "no_test_driven_followup": True,
        "single_test_read_this_round": True,
        "official_test_labels_read": False,
    }
    write_json(OUT / "test_protocol.json", payload)
    log(f"[test-protocol] frozen at {payload['created_utc']}")
    return 0


# ---------------------------------------------------------------------------
# Step 3 -- label-blind prediction freeze
# ---------------------------------------------------------------------------


def cmd_test_predict(args: argparse.Namespace) -> int:
    if (OUT / "test_read.lock").exists():
        raise SystemExit("this round's test labels were already read; refusing to regenerate predictions")
    if not (OUT / "test_protocol.json").exists():
        raise SystemExit("run test-protocol first")
    audit = read_json(OUT / "checkpoint_audit.json")
    if audit["decision"] != "READY_FOR_TEST_PREDICTION":
        raise SystemExit("audit did not pass")
    if (OUT / "test_prediction_freeze.json").exists():
        raise SystemExit("test predictions already frozen")

    ds = t0.load_test_dataset()
    fingerprint = t0._test_structure_fingerprint(ds)
    hist = read_json(T0 / "terminal_test_freeze.json")
    assert fingerprint["structure_sha256"] == hist["structure_fingerprint"]["structure_sha256"]
    assert fingerprint["processed_pt_sha256"] == hist["structure_fingerprint"]["processed_pt_sha256"]

    guarded = [t0.LabelGuard(ds[i]) for i in range(N_TEST)]
    objs = t0.build_test_objects(guarded, N_TEST)
    sig = t0.objects_signature(objs)
    obj_hash, per_mol = t0.objects_hash(objs)
    objects_match_history = sig == hist["objects"]["signature"] and obj_hash == hist["objects"]["combined_sha256"]
    log(f"[test-predict] objects {sig} hash {obj_hash[:16]} matches_history={objects_match_history}")
    if not objects_match_history:
        raise SystemExit("test object construction does not match the historical terminal test objects")

    # --- label-storage mutation check (values are overwritten, never read) ---
    store = t0._label_storage(ds)
    mutation: dict[str, Any] = {"available": store is not None}
    if store is not None:
        orig = store.y.detach().clone()
        t0._clear_slice_cache(ds)
        h_before, _ = t0.objects_hash(t0.build_test_objects([t0.LabelGuard(ds[i]) for i in range(16)], 16))
        store.y = torch.zeros_like(store.y)
        t0._clear_slice_cache(ds)
        h_zero, _ = t0.objects_hash(t0.build_test_objects([t0.LabelGuard(ds[i]) for i in range(16)], 16))
        store.y = torch.full_like(orig, 999.0)
        t0._clear_slice_cache(ds)
        h_999, _ = t0.objects_hash(t0.build_test_objects([t0.LabelGuard(ds[i]) for i in range(16)], 16))
        store.y = orig
        t0._clear_slice_cache(ds)
        mutation.update(
            {
                "objects_hash_before": h_before,
                "objects_hash_zero_labels": h_zero,
                "objects_hash_999_labels": h_999,
                "objects_unchanged": h_before == h_zero == h_999,
            }
        )

    d0 = ds[0]
    base = objs[0]
    perm = list(reversed(range(base.n_atoms)))
    inv = {old: new for new, old in enumerate(perm)}
    x_p = np.asarray(d0.x).copy().reshape(-1)
    x_new = np.zeros_like(x_p)
    for old in range(base.n_atoms):
        x_new[inv[old]] = x_p[old]
    edge_rows = [(inv[int(d0.edge_index[0, c])], inv[int(d0.edge_index[1, c])]) for c in range(np.asarray(d0.edge_index).shape[1])]
    o_perm = sro.build_mol_objects(0, x_new, np.asarray(edge_rows, dtype=np.int64).T, np.asarray(d0.edge_attr))

    files: dict[str, Any] = {}
    invariance: dict[str, Any] = {}
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    for key in KEYS:
        model = load_model(key)
        pred = t0.predict_test_objs(model, objs, "cpu")
        if not np.isfinite(pred).all():
            raise RuntimeError(f"non-finite test predictions for {key}")
        pred_base = t0.predict_test_objs(model, [base], "cpu")[0]
        pred_perm = t0.predict_test_objs(model, [o_perm], "cpu")[0]
        pred_batch8 = t0.predict_test_objs(model, objs[:8], "cpu")[0]
        pred16 = t0.predict_test_objs(model, objs[:16], "cpu")
        model2 = load_model(key)
        pred_replay = t0.predict_test_objs(model2, objs[:16], "cpu")
        invariance[key] = {
            "relabelling_abs_diff": float(abs(pred_base - pred_perm)),
            "batch_composition_abs_diff": float(abs(pred_base - pred_batch8)),
            "checkpoint_reload_max_abs_diff": float(np.abs(pred16 - pred_replay).max()),
        }
        path = OUT / "predictions" / f"test_{key}.npz"
        meta = {
            "protocol": PROTOCOL_ID,
            "model": key,
            "arm": ARM_OF[key],
            "seed": SEED_OF[key],
            "row": list(range(N_TEST)),
            "object_hash": obj_hash,
            "checkpoint_sha256": file_sha256(SOUPS[key]),
            "label_free": True,
        }
        np.savez_compressed(path, row=np.arange(N_TEST, dtype=np.int64), pred=pred.astype(np.float32), meta=json.dumps(meta))
        files[key] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
        log(f"[test-predict] {key} pred[0]={pred[0]:+.6f} relabel={invariance[key]['relabelling_abs_diff']:.2e}")

    combined = hashlib.sha256()
    for key in sorted(files):
        combined.update(key.encode())
        combined.update(files[key]["sha256"].encode())
    checks = {
        "structures_match_history": objects_match_history,
        "all_predictions_finite": True,
        "relabelling_invariance": all(v["relabelling_abs_diff"] < 1e-6 for v in invariance.values()),
        "batch_composition_invariance": all(v["batch_composition_abs_diff"] < 1e-6 for v in invariance.values()),
        "checkpoint_reload_exact": all(v["checkpoint_reload_max_abs_diff"] == 0.0 for v in invariance.values()),
        "label_storage_independent": bool(mutation.get("objects_unchanged", True)),
        "row_order": True,
    }
    freeze = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_before_test_label_read": True,
        "test_labels_read": False,
        "models": {
            key: {
                "checkpoint": str(SOUPS[key].relative_to(REPO_ROOT)),
                "checkpoint_sha256": file_sha256(SOUPS[key]),
                "arm": ARM_OF[key],
                "seed": SEED_OF[key],
                "parameters": 413505,
            }
            for key in KEYS
        },
        "prediction_files": files,
        "combined_prediction_sha256": combined.hexdigest(),
        "objects": {"signature": sig, "combined_sha256": obj_hash, "per_molecule_sha256": per_mol, "matches_history": objects_match_history},
        "row_order": {"rows": list(range(N_TEST)), "rows_sha256": rows_sha256(range(N_TEST))},
        "structure_fingerprint": fingerprint,
        "label_independence": {"label_guard_active": True, "label_storage_mutation": mutation},
        "invariance": invariance,
        "checks": checks,
        "environment": _environment(),
        "code_hashes": _code_hashes(),
        "official_test_structures_loaded": True,
        "official_test_labels_read": False,
    }
    write_json(OUT / "test_prediction_freeze.json", freeze)
    if not all(checks.values()):
        raise SystemExit(f"freeze checks failed: {checks}")
    log(f"[test-predict] frozen combined {combined.hexdigest()[:16]}")
    return 0


# ---------------------------------------------------------------------------
# Step 4 -- single test-label read, replay gate and terminal metrics
# ---------------------------------------------------------------------------


def _subgroup_delta(delta: np.ndarray, mask: np.ndarray) -> float | None:
    return float(delta[mask].mean()) if int(mask.sum()) else None


def cmd_test_score(args: argparse.Namespace) -> int:
    if (OUT / "test_read.lock").exists():
        raise SystemExit("this round's test labels were already read; refusing a second read")
    freeze = read_json(OUT / "test_prediction_freeze.json")
    for key, entry in freeze["prediction_files"].items():
        if file_sha256(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"frozen test prediction changed: {key}")
    for key, entry in freeze["models"].items():
        if file_sha256(REPO_ROOT / entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint changed after freeze: {key}")

    started = time.perf_counter()
    ds = t0.load_test_dataset()
    y = np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(N_TEST)], dtype=np.float64)
    np.save(OUT / "test_y_read_once.npy", y)
    y_sha = array_sha256(y)

    hist_lock = read_json(T0 / "test_read.lock")
    hist_y = np.asarray(np.load(T0 / "test_y_read_once.npy"), dtype=np.float64)
    y_cross_hist_max_abs = float(np.abs(y - hist_y).max())
    ksvd_path = REPO_ROOT / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1/heldout_test_predictions.npz"
    y_cross_ksvd_max_abs = None
    if ksvd_path.exists():
        y_cross_ksvd_max_abs = float(np.abs(y - np.asarray(np.load(ksvd_path, allow_pickle=True)["y"], dtype=np.float64)).max())
    lock = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_test": N_TEST,
        "rows_sha256": rows_sha256(range(N_TEST)),
        "y_sha256": y_sha,
        "y_sha256_matches_previous_round_lock": y_sha == hist_lock["y_sha256"],
        "y_max_abs_diff_vs_previous_round_copy": y_cross_hist_max_abs,
        "y_max_abs_diff_vs_ksvd_copy": y_cross_ksvd_max_abs,
        "source": "PyG ZINC subset=True test processed split (data/ZINC/subset/processed/test.pt)",
        "single_read_this_round": True,
        "not_first_ever_read": True,
        "previous_round_lock_protocol": hist_lock.get("protocol"),
        "prediction_freeze_sha256": file_sha256(OUT / "test_prediction_freeze.json"),
        "ensembles_used": False,
        "official_test_loaded": True,
    }
    write_json(OUT / "test_read.lock", lock)
    if y_cross_hist_max_abs != 0.0 or not lock["y_sha256_matches_previous_round_lock"]:
        raise SystemExit("the fresh test y disagrees with the previous round's stored copy; stopping")

    preds: dict[str, np.ndarray] = {}
    for key in KEYS:
        with np.load(REPO_ROOT / freeze["prediction_files"][key]["path"]) as z:
            assert np.array_equal(z["row"], np.arange(N_TEST)), f"{key}: row order broken"
            preds[key] = np.asarray(z["pred"], dtype=np.float64)
        assert preds[key].shape == y.shape
    e = {k: np.abs(preds[k] - y) for k in KEYS}
    metrics = {k: row_metrics(preds[k], y) for k in KEYS}

    # ---- P0 replay gate before any P3 claim -------------------------------
    replay: dict[str, Any] = {}
    replay_ok = True
    for key in ("P00", "P01"):
        hist_mae = HIST_TEST_MAE[key]
        mae = metrics[key]["mae"]
        diff = float(mae - hist_mae)
        hist_pred = np.asarray(np.load(T0_TEST_PRED[key])["pred"], dtype=np.float64)
        npz_diff = float(np.abs(preds[key] - hist_pred).max())
        replay[key] = {
            "expected_mae": hist_mae,
            "actual_mae": mae,
            "mae_diff": diff,
            "npz_max_abs_diff_vs_previous_round": npz_diff,
            "ok": abs(diff) <= REPLAY_MAE_ATOL and npz_diff <= REPLAY_NPZ_ATOL,
        }
        replay_ok = replay_ok and replay[key]["ok"]
        log(f"[test-score] replay {key} MAE {mae:.6f} expected {hist_mae:.6f} diff {diff:+.2e} npz_maxdiff {npz_diff:.2e}")
    replay["P0_seed_mean"] = {
        "expected_mae": HIST_TEST_MAE["P0_seed_mean"],
        "actual_mae": float(np.mean([metrics["P00"]["mae"], metrics["P01"]["mae"]])),
        "ok": abs(float(np.mean([metrics["P00"]["mae"], metrics["P01"]["mae"]])) - HIST_TEST_MAE["P0_seed_mean"]) <= REPLAY_MAE_ATOL,
    }
    replay_ok = replay_ok and replay["P0_seed_mean"]["ok"]
    if not replay_ok:
        blocked = {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "blocked": True,
            "reason": "P0 control replay does not match the historical terminal-test MAE; P3 numbers are not comparable",
            "p0_replay": replay,
            "test_read_lock_sha256": file_sha256(OUT / "test_read.lock"),
            "official_test_loaded": True,
        }
        write_json(OUT / "official_test_metrics.json", blocked)
        log("[test-score] BLOCKED: P0 replay mismatch")
        return 2

    # ---- P3 terminal metrics and the P3-P0 contrast -----------------------
    seed_mean_metrics = {
        arm: {name: float(np.mean([metrics[k][name] for k in KEYS if ARM_OF[k] == arm])) for name in metrics["P00"]}
        for arm in ("P0", "P3")
    }
    seed_mean_metrics["P0"]["per_seed_mae"] = [metrics["P00"]["mae"], metrics["P01"]["mae"]]
    seed_mean_metrics["P3"]["per_seed_mae"] = [metrics["P30"]["mae"], metrics["P31"]["mae"]]
    d = ((e["P30"] - e["P00"]) + (e["P31"] - e["P01"])) / 2.0
    boot = paired_bootstrap(d, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    per_seed_delta = {"seed0": float(e["P30"].mean() - e["P00"].mean()), "seed1": float(e["P31"].mean() - e["P01"].mean())}
    tail = y <= TAIL_THRESHOLD
    ordinary = ~tail
    d_b0 = ((e["P30"] - np.abs(np.load(T0 / "predictions/test_B0.npz")["pred"].astype(np.float64) - y)) + (e["P31"] - np.abs(np.load(T0 / "predictions/test_B1.npz")["pred"].astype(np.float64) - y))) / 2.0
    subgroups = {
        "ordinary_y_gt_-3": {
            "n": int(ordinary.sum()),
            "mae_P3_seed_mean": float(np.mean([e[f"P3{s}"][ordinary].mean() for s in (0, 1)])),
            "mae_P0_seed_mean": float(np.mean([e[f"P0{s}"][ordinary].mean() for s in (0, 1)])),
            "delta": _subgroup_delta(d, ordinary),
        },
        "tail_y_le_-3": {
            "n": int(tail.sum()),
            "mae_P3_seed_mean": float(np.mean([e[f"P3{s}"][tail].mean() for s in (0, 1)])),
            "mae_P0_seed_mean": float(np.mean([e[f"P0{s}"][tail].mean() for s in (0, 1)])),
            "delta": _subgroup_delta(d, tail),
            "mean_signed_P3": float(np.mean([(preds[f"P3{s}"] - y)[tail].mean() for s in (0, 1)])),
            "mean_signed_P0": float(np.mean([(preds[f"P0{s}"] - y)[tail].mean() for s in (0, 1)])),
        },
        "top_k_removed": {
            f"top{k}": f10k.top_k_removed(d, np.maximum(e["P30"], e["P31"]), np.maximum(e["P00"], e["P01"]), k) for k in (1, 5, 10)
        },
    }
    both_negative = per_seed_delta["seed0"] < 0 and per_seed_delta["seed1"] < 0
    both_positive = per_seed_delta["seed0"] > 0 and per_seed_delta["seed1"] > 0
    if both_negative and boot["ci_hi"] < 0:
        decision = "P3_TEST_IMPROVES"
    elif both_positive and boot["ci_lo"] > 0:
        decision = "P3_TEST_WORSE"
    else:
        decision = "P3_TEST_INCONCLUSIVE_OR_MIXED"
    p3_mean = seed_mean_metrics["P3"]["mae"]
    in_band = TARGET_LO <= p3_mean <= TARGET_HI
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_molecules": N_TEST,
        "per_model": metrics,
        "seed_mean": seed_mean_metrics,
        "p3_minus_p0": {
            "per_seed": per_seed_delta,
            "delta_seed_mean": float(d.mean()),
            "ci95": [boot["ci_lo"], boot["ci_hi"]],
            "better_fraction": float((d < 0).mean()),
            "bootstrap": boot,
            "both_seeds_same_direction": bool(both_negative or both_positive),
            "both_seeds_improve": bool(both_negative),
            "definition": "d_i = 0.5*[(|e_P3s0|-|e_P0s0|) + (|e_P3s1|-|e_P0s1|)]",
        },
        "p3_minus_b0_context": {
            "delta_seed_mean": float(d_b0.mean()),
            "ci95": [paired_bootstrap(d_b0, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)["ci_lo"], paired_bootstrap(d_b0, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)["ci_hi"]],
            "note": "context only; B0/B1 predictions are the previous round's frozen test files",
        },
        "subgroups": subgroups,
        "p0_replay": replay,
        "target_band": {
            "band": [TARGET_LO, TARGET_HI],
            "p3_seed_mean_mae": p3_mean,
            "p0_seed_mean_mae": seed_mean_metrics["P0"]["mae"],
            "reached": bool(in_band),
            "statement": (
                f"P3 seed-mean official-test MAE {p3_mean:.6f} is inside the 0.06-0.07 band"
                if in_band
                else f"P3 seed-mean official-test MAE {p3_mean:.6f} is {p3_mean - TARGET_HI:+.6f} relative to the 0.07 upper edge"
            ),
        },
        "decision": decision,
        "ensemble_used": False,
        "single_read": True,
        "exposed_test": True,
        "exposure_disclosure": "official test was previously read in this project; this is an exposed-test terminal evaluation, not a first blind test",
        "official_test_loaded": True,
        "no_training_after_test_read": True,
        "no_test_driven_followup": True,
        "seconds": time.perf_counter() - started,
    }
    write_json(OUT / "official_test_metrics.json", payload)
    log(
        f"[test-score] P0 {seed_mean_metrics['P0']['mae']:.6f} P3 {p3_mean:.6f} "
        f"delta {d.mean():+.6f} CI [{boot['ci_lo']:+.6f},{boot['ci_hi']:+.6f}] -> {decision}"
    )
    return 0


# ---------------------------------------------------------------------------
# Step 5 -- report
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    m = read_json(OUT / "official_test_metrics.json")
    if m.get("blocked"):
        raise SystemExit("scoring is blocked; report not generated")
    pm, sm, dp3 = m["per_model"], m["seed_mean"], m["p3_minus_p0"]
    sub = m["subgroups"]
    p3, p0 = sm["P3"], sm["P0"]
    rows = "\n".join(
        f"| {key} | {ARM_OF[key]} | {SEED_OF[key]} | {pm[key]['mae']:.6f} | {pm[key]['rmse']:.6f} | {pm[key]['median_abs']:.6f} | {pm[key]['p90_abs']:.6f} |"
        for key in KEYS
    )
    band = m["target_band"]
    md = f"""# P3 frozen exposed-test terminal evaluation — REPORT

Protocol `{PROTOCOL_ID}` · commit `{git_commit()[:12]}` · driver
`code/run_bjoc_r3_frozen_terminal_test_v1.py` (`sha256 {_code_hashes()['driver'][:12]}…`) ·
audit `checkpoint_audit.json` · freeze `test_prediction_freeze.json` ·
preregistration `test_protocol.json`.

**Scope.** Authorized terminal evaluation of the frozen full-10k **P3** model
(masked-SSL encoder + R3 360-epoch cosine last-5 soup, BJOC-Null, 413,505
params) on the official `ZINC(subset=True, split="test")` 1,000-molecule split,
against the historical full-10k **P0** control. Inference/audit/scoring only; no
training, tuning or model selection. All predictions were frozen and hashed
before the single test-label read of this round. The previous round's frozen
artifacts were opened read-only and were not modified.

**Exposure disclosure.** The official test instrument was already read in this
project (KSVD track 2026-10-03, then `cscl-bjoc-pretraining-mechanism-terminal-test-v1`).
This is an **exposed-test terminal evaluation**, not a first blind test. It may
not be used to select models, seeds, recipes or post-processing, and no
test-driven follow-up was run.

---

## A. One-line conclusion

**P3 official-test MAE (seed-mean) = {p3['mae']:.6f}** (seed 0 {pm['P30']['mae']:.6f},
seed 1 {pm['P31']['mae']:.6f}); P0 replay confirmed at {p0['mae']:.6f}
({pm['P00']['mae']:.6f} / {pm['P01']['mae']:.6f}). P3−P0 = **{dp3['delta_seed_mean']:+.6f}**
(CI95 [{dp3['ci95'][0]:+.6f}, {dp3['ci95'][1]:+.6f}], per-seed
{dp3['per_seed']['seed0']:+.6f} / {dp3['per_seed']['seed1']:+.6f}),
decision `{m['decision']}`. Target band 0.06–0.07: **{'REACHED' if band['reached'] else 'NOT REACHED'}** — {band['statement']}.

---

## B. Replay gate (P0 control)

| model | expected MAE (previous round) | replayed MAE | diff | npz max abs diff | ok |
|---|---:|---:|---:|---:|---|
| P0 seed 0 | {m['p0_replay']['P00']['expected_mae']:.6f} | {m['p0_replay']['P00']['actual_mae']:.6f} | {m['p0_replay']['P00']['mae_diff']:+.2e} | {m['p0_replay']['P00']['npz_max_abs_diff_vs_previous_round']:.2e} | {m['p0_replay']['P00']['ok']} |
| P0 seed 1 | {m['p0_replay']['P01']['expected_mae']:.6f} | {m['p0_replay']['P01']['actual_mae']:.6f} | {m['p0_replay']['P01']['mae_diff']:+.2e} | {m['p0_replay']['P01']['npz_max_abs_diff_vs_previous_round']:.2e} | {m['p0_replay']['P01']['ok']} |
| P0 seed-mean | {m['p0_replay']['P0_seed_mean']['expected_mae']:.6f} | {m['p0_replay']['P0_seed_mean']['actual_mae']:.6f} | — | — | {m['p0_replay']['P0_seed_mean']['ok']} |

Checkpoint hashes, 413,505-parameter audits, forbidden-module scans, train-only
normaliser and the historical test structure fingerprint all pass
(`checkpoint_audit.json`).

## C. Official-test metrics

| key | arm | seed | MAE | RMSE | median abs | p90 abs |
|---|---|---:|---:|---:|---:|---:|
{rows}

| seed-mean | MAE | RMSE | median | p90 |
|---|---:|---:|---:|---:|
| P0 | {p0['mae']:.6f} | {p0['rmse']:.6f} | {p0['median_abs']:.6f} | {p0['p90_abs']:.6f} |
| P3 | {p3['mae']:.6f} | {p3['rmse']:.6f} | {p3['median_abs']:.6f} | {p3['p90_abs']:.6f} |

Primarily requested contrast (paired per molecule, 5000 bootstrap resamples):

| statistic | value |
|---|---:|
| P3−P0 seed-mean Δ MAE | {dp3['delta_seed_mean']:+.6f} |
| 95% CI | [{dp3['ci95'][0]:+.6f}, {dp3['ci95'][1]:+.6f}] |
| per-seed Δ (seed 0 / seed 1) | {dp3['per_seed']['seed0']:+.6f} / {dp3['per_seed']['seed1']:+.6f} |
| both seeds improve | {dp3['both_seeds_improve']} |
| better fraction | {dp3['better_fraction']:.4f} |
| ordinary (y > −3, n={sub['ordinary_y_gt_-3']['n']}) Δ | {sub['ordinary_y_gt_-3']['delta']:+.6f} |
| tail (y ≤ −3, n={sub['tail_y_le_-3']['n']}) Δ | {sub['tail_y_le_-3']['delta']:+.6f} |
| top-10 removed Δ | {sub['top_k_removed']['top10']['delta_top10_removed']:+.6f} |
| context: P3−B0 seed-mean Δ | {m['p3_minus_b0_context']['delta_seed_mean']:+.6f} |

Decision: **`{m['decision']}`**. Target band judgement (P3 seed-mean):
**{'reached' if band['reached'] else 'not reached'}** ({band['statement']}).

## D. What this does and does not show

* Shows: the exact exposed-test MAE of the frozen P3 checkpoints and the exact
  same-round paired difference to the frozen P0 control, with the previous
  round's P0 numbers reproduced to <{REPLAY_MAE_ATOL:g} MAE.
* Does not show: a first blind test result; a statement about arbitrary new
  molecule distributions; any justification for changing the model, seeds,
  recipe, checkpoint choice or error post-processing after seeing the test.
* P3 was chosen before this evaluation by the recipe round on the exposed
  official-valid split; the test numbers here are descriptive terminal
  measurements, not a new selection.
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    log("[report] REPORT.md written")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=PROTOCOL_ID)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    sub.add_parser("test-protocol").set_defaults(func=cmd_test_protocol)
    sub.add_parser("test-predict").set_defaults(func=cmd_test_predict)
    sub.add_parser("test-score").set_defaults(func=cmd_test_score)
    sub.add_parser("report").set_defaults(func=cmd_report)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
