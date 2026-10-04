"""ZINC chemistry task: after removing the ring target, does the *current task
dictionary* bridge ``D_g`` generalise the remaining chemical target ``g`` better
than a matched MLP bridge ``M_g``?

Round: ``zinc_chemistry_dictionary_vs_mlp_seed0_v1``.

This is a **single-variable** follow-up of
``zinc_task_dictionary_and_cycle_witness_seed0_v1`` (execution revision
``77c40dda9d5b730a64ea653b63b053a11460993d``).  The fold, input standardizers,
initial states, bridge implementations, optimizer, batch schedule and soup
protocol are frozen from that round.  The only scientific change is the
supervision target ``y -> g`` with ``g = y - c`` and ``c`` the label-derived
cycle component.

Target provenance: ``c`` must not carry a normalisation constant fitted on the
current dev rows.  The old round reused ``target_decomposition.npz`` whose
``mu_cycle`` / ``sigma_cycle`` had been fitted (Nelder-Mead snap fit) on the
whole audit table, i.e. on labels that include the current dev rows.  This round
therefore rebuilds the decomposition with constants fitted **only on the 8000
fit rows** and freezes that definition into ``fit_only_targets.npz`` (committed
next to this module).  The remote training run only reads that frozen npz; it
never re-fits on labels and never opens official-valid/test.

Usage (local Phase A, then remote arms, then local analysis)::

    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_chemistry_dictionary_vs_mlp_seed0_v1 --phase-a
    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_chemistry_dictionary_vs_mlp_seed0_v1 --smoke
    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_chemistry_dictionary_vs_mlp_seed0_v1 --arm D --device cuda --out <dir>
    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_chemistry_dictionary_vs_mlp_seed0_v1 --arm M --device cuda --out <dir>
    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_chemistry_dictionary_vs_mlp_seed0_v1 --analyze --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw

PROTOCOL_VERSION = "zinc-chemistry-dictionary-vs-mlp-seed0-v1"
RESULT_SLUG = "zinc_chemistry_dictionary_vs_mlp_seed0_v1"
RESULTS_DIR = zjd.TRACK_ROOT / "results" / RESULT_SLUG
TARGETS_NPZ = RESULTS_DIR / "fit_only_targets.npz"
PREV_RESULTS = zw.RESULTS_DIR
AUDIT_DIR = zjd.CYCLE

#: frozen recipe — copied from the paired y round, never selected on dev.
SEED = zw.SEED
EPOCHS = zw.EPOCHS
LR = zw.LR
WEIGHT_DECAY = zw.WEIGHT_DECAY
GRAD_CLIP = zw.GRAD_CLIP
BATCH_SIZE = zw.BATCH_SIZE
SOUP_EPOCHS = zw.SOUP_EPOCHS
TRAIN_SHUFFLE_OFFSET = zw.TRAIN_SHUFFLE_OFFSET
FOLD_SEED = zw.FOLD_SEED
N_FIT = zw.N_FIT
N_DEV = zw.N_DEV
BRIDGE_DIM = zw.BRIDGE_DIM
BRIDGE_ATOMS = zw.BRIDGE_ATOMS
BRIDGE_PARAMETERS = zw.BRIDGE_PARAMETERS
EXPECTED_BODY_PARAMETERS = zw.EXPECTED_BODY_PARAMETERS
EXPECTED_TOTAL_PARAMETERS = zw.EXPECTED_TOTAL_PARAMETERS
BOOT_SEED = zw.BOOT_SEED
N_BOOT = zw.N_BOOT
DELTA = zw.DELTA
G0_TOL = zw.G0_TOL
DELTA_SLACK = zw.DELTA_SLACK
REPLAY_TOL = zw.REPLAY_TOL
ARMS = zw.ARMS

jsonable = zw.jsonable
write_json = zw.write_json
file_sha256 = zw.file_sha256
tensor_hash = zw.tensor_hash
state_hash = zw.state_hash
seed_everything = zw.seed_everything
group_masks = zw.group_masks
build_schedule = zw.build_schedule
build_arm = zw.build_arm
build_pair_hashes = zw.build_pair_hashes
build_fold = zw.build_fold
evaluate_state = zw.evaluate_state
MLPBridge = zw.MLPBridge


# ---------------------------------------------------------------------------
# Phase A — fit-only target decomposition (label side, no model input)
# ---------------------------------------------------------------------------


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Ordinary least squares with intercept, numpy only."""
    a = np.concatenate([np.asarray(x, np.float64), np.ones((x.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(a, np.asarray(y, np.float64), rcond=None)
    return coef[:-1], float(coef[-1])


def _snap_loss(params: np.ndarray, eff_norm: np.ndarray) -> float:
    s_c, mu_c = float(params[0]), float(params[1])
    eff = eff_norm * s_c + mu_c
    snap = np.clip(np.round(eff), -20, 0)
    return float(np.mean((eff - snap) ** 2))


def _refit_constants(
    rows: np.ndarray,
    in_fit: np.ndarray,
    *,
    mu_logp: float,
) -> dict[str, float]:
    """Refit ``sL, sS, muS`` (cycle-free bulk) and ``sC, muC`` (snap fit).

    ``rows`` are structured-dtype audit rows for the 10000 train molecules with
    fields ``logP``, ``SA``, ``y_stored``, ``cycle_score_gvae``, ``in_fit``.
    Only rows with ``in_fit == True`` are used for both fits.
    """
    z0 = rows[(rows["cycle_score_gvae"] == 0) & in_fit]
    if int(z0.shape[0]) < 100:
        raise RuntimeError("not enough cycle-free fit rows for the constant refit")
    coef, intercept = _linear_fit(np.stack([z0["logP"], z0["SA"]], axis=1), z0["y_stored"])
    s_l = 1.0 / float(coef[0])
    s_s = 1.0 / float(coef[1])
    mu_s = -s_s * (intercept + mu_logp / s_l)

    fit_rows = rows[in_fit]
    eff_norm = fit_rows["y_stored"] - (
        (fit_rows["logP"] - mu_logp) / s_l + (fit_rows["SA"] - mu_s) / s_s
    )
    from scipy.optimize import minimize

    best = minimize(_snap_loss, np.array([0.288, -0.049]), args=(eff_norm,), method="Nelder-Mead")
    return {
        "sigma_logP": s_l,
        "sigma_SA": s_s,
        "mu_SA": mu_s,
        "sigma_cycle": float(best.x[0]),
        "mu_cycle": float(best.x[1]),
        "fit_rows_cycle_free": int(z0.shape[0]),
        "fit_rows_snap": int(fit_rows.shape[0]),
    }


def _apply_constants(rows: np.ndarray, constants: Mapping[str, float], mu_logp: float) -> dict[str, np.ndarray]:
    s_l = float(constants["sigma_logP"])
    s_s = float(constants["sigma_SA"])
    mu_s = float(constants["mu_SA"])
    s_c = float(constants["sigma_cycle"])
    mu_c = float(constants["mu_cycle"])
    eff_norm = rows["y_stored"] - ((rows["logP"] - mu_logp) / s_l + (rows["SA"] - mu_s) / s_s)
    eff = eff_norm * s_c + mu_c
    k = np.clip(np.round(eff), -20, 0).astype(np.int64)
    c = (k.astype(np.float64) - mu_c) / s_c
    return {"effective_cycle": eff, "k": k, "c": c}


def build_fit_only_targets() -> dict[str, Any]:
    """Rebuild ``y, c, g, k`` with constants fitted on the new 8000 fit rows."""
    import pandas as pd

    audit_csv = AUDIT_DIR / "formula_verification_per_molecule.csv"
    label_csv = AUDIT_DIR / "train_cycle_audit_label.csv"
    refine_json = AUDIT_DIR / "stage_refine.json"
    handoff_path = zftd.HANDOFF_TRAIN
    df = pd.read_csv(audit_csv)
    df["molecule_id"] = df["molecule_id"].astype(str)
    train = df[df["split"] == "train"].copy()
    train = train.sort_values("subset_index").reset_index(drop=True)
    if not np.array_equal(train["subset_index"].to_numpy(np.int64), np.arange(10000)):
        raise RuntimeError("audit train rows are not positional")
    if not bool(train["matched"].all()):
        raise RuntimeError("some train audit rows are unmatched")

    handoff = np.load(handoff_path, allow_pickle=True)
    ids = np.asarray(handoff["ids"]).astype(str)
    y = np.asarray(handoff["y"], np.float64)
    gid = np.asarray(handoff["canonical_group_id"], np.int64)
    if len(ids) != 10000 or not np.array_equal(np.sort(np.unique(gid)), np.unique(gid)):
        raise RuntimeError("handoff shape/uniqueness check failed")

    fold = build_fold()
    in_fit = np.zeros(10000, dtype=bool)
    in_fit[fold["fit_idx"]] = True

    rows = np.empty(10000, dtype=[("logP", np.float64), ("SA", np.float64), ("y_stored", np.float64),
                                  ("cycle_score_gvae", np.float64)])
    rows["logP"] = train["logP"].to_numpy(np.float64)
    rows["SA"] = train["SA"].to_numpy(np.float64)
    rows["y_stored"] = train["y_stored"].to_numpy(np.float64)
    rows["cycle_score_gvae"] = train["cycle_score_gvae"].to_numpy(np.float64)

    refine = json.loads(refine_json.read_text())
    old_constants = dict(refine["constants_fitted"])
    mu_logp = 2.4570953396190123  # COMMUNITY_CONSTANTS["logP_mean"], fixed label-generation constant
    new_constants = _refit_constants(rows, in_fit, mu_logp=mu_logp)
    labels = _apply_constants(rows, new_constants, mu_logp)
    k = labels["k"]
    c = labels["c"]
    g = y - c

    # cross-check against the audit's stored snapped labels and y_stored.
    label_tbl = pd.read_csv(label_csv)
    if not np.array_equal(label_tbl["subset_index"].to_numpy(np.int64), np.arange(10000)):
        raise RuntimeError("label table is not positional")
    stored_k = label_tbl["label_effective_cycle_snapped"].to_numpy(np.float64)

    old = zftd.load_target_decomposition()
    old_k = np.asarray(old["k"], np.int64)
    old_c = np.asarray(old["c"], np.float64)
    old_g = np.asarray(old["g"], np.float64)
    old_y = np.asarray(old["y"], np.float64)

    checks = {
        "n_rows": 10000,
        "handoff_ids_positional": bool(list(ids) == [str(i) for i in range(10000)]),
        "y_handoff_equals_audit_y_stored_max_abs": float(np.max(np.abs(y - rows["y_stored"]))),
        "k_equals_audit_stored_snapped_diff_rows": int(np.sum(k.astype(np.float64) != stored_k)),
        "g_equals_y_minus_c_max_abs": float(np.max(np.abs(g - (y - c)))),
        "k_fit_counts": {name: int(mask.sum()) for name, mask in group_masks(k[fold["fit_idx"]]).items()},
        "k_dev_counts": {name: int(mask.sum()) for name, mask in group_masks(k[fold["dev_idx"]]).items()},
        "c_g0_spread_max_abs": float(np.max(np.abs(c[k == 0] - np.median(c[k == 0])))) if int((k == 0).sum()) else None,
        "c_g0_value": float(np.median(c[k == 0])) if int((k == 0).sum()) else None,
        "old_vs_new": {
            "k_diff_rows": int(np.sum(old_k != k)),
            "c_max_abs_diff": float(np.max(np.abs(old_c - c))),
            "g_max_abs_diff": float(np.max(np.abs(old_g - g))),
            "y_max_abs_diff": float(np.max(np.abs(old_y - y))),
            "old_constants": old_constants,
            "new_constants": new_constants,
        },
        "constants_changes": {
            key: float(new_constants[key]) - float(old_constants[key])
            for key in ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")
        },
        "uses_dev_labels_in_fit": False,
        "fit_rows_only": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if {key: checks["k_fit_counts"][key] for key in ("k=0", "k=-1", "k=-2", "k<=-3")} != {
        "k=0": 7713, "k=-1": 252, "k=-2": 30, "k<=-3": 5,
    }:
        raise RuntimeError(f"fit k counts changed: {checks['k_fit_counts']}")
    if {key: checks["k_dev_counts"][key] for key in ("k=0", "k=-1", "k=-2", "k<=-3")} != {
        "k=0": 1915, "k=-1": 73, "k=-2": 10, "k<=-3": 2,
    }:
        raise RuntimeError(f"dev k counts changed: {checks['k_dev_counts']}")

    payload = {"y": y, "c": c, "g": g, "k": k, "gid": gid}
    return {"arrays": payload, "checks": checks, "constants": new_constants, "fold": fold}


def save_targets_npz(targets: Mapping[str, Any]) -> str:
    a = targets["arrays"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TARGETS_NPZ,
        y=np.asarray(a["y"], np.float64),
        c=np.asarray(a["c"], np.float64),
        g=np.asarray(a["g"], np.float64),
        k=np.asarray(a["k"], np.int64),
        gid=np.asarray(a["gid"], np.int64),
        constants=np.asarray([targets["constants"][key] for key in
                              ("sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle")], np.float64),
        constant_names=np.asarray(["sigma_logP", "sigma_SA", "mu_SA", "sigma_cycle", "mu_cycle"]),
    )
    return file_sha256(TARGETS_NPZ)


def load_frozen_targets() -> dict[str, np.ndarray]:
    with np.load(TARGETS_NPZ, allow_pickle=False) as z:
        return {key: z[key] for key in ("y", "c", "g", "k", "gid")}


def load_prepared() -> tuple[dict[str, Any], dict[str, Any], list[Any], list[Any], dict[str, np.ndarray]]:
    fold, prep_meta, fit_data, dev_data, _old_decomp, _targets = zw.load_and_prepare()
    tgt = load_frozen_targets()
    if int(tgt["g"].shape[0]) != 10000:
        raise RuntimeError("frozen target npz length mismatch")
    return fold, prep_meta, fit_data, dev_data, tgt


# ---------------------------------------------------------------------------
# Phase A — historical anchors, init identity, manifests
# ---------------------------------------------------------------------------


def _reload_prev_predictions(arm: str) -> dict[str, np.ndarray]:
    with np.load(PREV_RESULTS / f"{arm}_raw_predictions.npz") as z:
        return {key: np.asarray(z[key], np.float64) for key in z.files}


def phase_a() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    targets = build_fit_only_targets()
    target_npz_hash = save_targets_npz(targets)
    fold = targets["fold"]
    checks = targets["checks"]
    checks["target_npz"] = str(TARGETS_NPZ)
    checks["target_npz_sha256"] = target_npz_hash
    write_json(RESULTS_DIR / "target_provenance.json", checks)

    # -- historical anchor: recompute the previous y round's G0 numbers ------
    prev_meta = {arm: json.loads((PREV_RESULTS / f"{arm}_meta.json").read_text()) for arm in ARMS}
    prev_pred = {arm: _reload_prev_predictions(arm) for arm in ARMS}
    y = targets["arrays"]["y"]
    k = targets["arrays"]["k"]
    y_fit, y_dev = y[fold["fit_idx"]], y[fold["dev_idx"]]
    k_fit, k_dev = k[fold["fit_idx"]], k[fold["dev_idx"]]
    published = json.loads((PREV_RESULTS / "analysis.json").read_text())
    anchor: dict[str, Any] = {"published_gate": published["gate"]["category"], "arms": {}}
    for arm in ARMS:
        b = float(prev_meta[arm]["calibration_b"]["raw_soup"])
        b_local = float(np.median(y_fit - prev_pred[arm]["raw_soup_fit"]))
        raw_soup_dev = prev_pred[arm]["raw_soup_dev"]
        cal_dev = raw_soup_dev + b
        g0 = k_dev == 0
        anchor["arms"][arm] = {
            "b_from_meta": b,
            "b_recomputed": b_local,
            "b_max_abs_diff": abs(b - b_local),
            "dev_G0_raw_mae": float(np.mean(np.abs(y_dev[g0] - raw_soup_dev[g0]))),
            "dev_G0_cal_mae": float(np.mean(np.abs(y_dev[g0] - cal_dev[g0]))),
            "dev_overall_raw_mae": float(np.mean(np.abs(y_dev - raw_soup_dev))),
            "dev_overall_cal_mae": float(np.mean(np.abs(y_dev - cal_dev))),
            "published_dev_G0_cal_mae": float(published["arms"]["dev"][arm]["cal"]["k=0"]["mae"]),
            "published_dev_overall_cal_mae": float(published["arms"]["dev"][arm]["cal"]["mae"]),
            "steps_done": int(prev_meta[arm]["steps_done"]),
            "steps_expected": int(prev_meta[arm]["steps_expected"]),
            "schedule_sha256": prev_meta[arm]["schedule_sha256"],
            "data_stream_sha256": prev_meta[arm]["data_stream_sha256"],
            "replay_max_abs_diff": float(prev_meta[arm]["replay_max_abs_diff"]),
        }
        anchor["arms"][arm]["G0_cal_recompute_delta"] = (
            anchor["arms"][arm]["dev_G0_cal_mae"] - anchor["arms"][arm]["published_dev_G0_cal_mae"]
        )
        anchor["arms"][arm]["overall_cal_recompute_delta"] = (
            anchor["arms"][arm]["dev_overall_cal_mae"] - anchor["arms"][arm]["published_dev_overall_cal_mae"]
        )
    anchor["published_G0_cal"] = {
        "D": float(published["arms"]["dev"]["D"]["cal"]["k=0"]["mae"]),
        "M": float(published["arms"]["dev"]["M"]["cal"]["k=0"]["mae"]),
        "G_y_cal_gain": float(published["gains"]["dev_G0_cal"]["point"]),
        "G_y_cal_ci": published["gains"]["dev_G0_cal"]["ci95"],
        "G_y_raw_gain": float(published["gains"]["dev_G0_raw"]["point"]),
    }
    anchor["recompute_max_abs_delta"] = max(
        abs(entry[key]) for entry in anchor["arms"].values()
        for key in ("G0_cal_recompute_delta", "overall_cal_recompute_delta", "b_max_abs_diff")
    )
    anchor["fold_hashes_match"] = bool(
        fold["fit_idx_sha256"] == "7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9"
        and fold["dev_idx_sha256"] == "a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff"
    )
    anchor["schedule_hash_match_expected"] = bool(
        prev_meta["D"]["schedule_sha256"] == "7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65"
        and prev_meta["D"]["schedule_sha256"] == prev_meta["M"]["schedule_sha256"]
    )
    write_json(RESULTS_DIR / "historical_anchor_checks.json", anchor)

    # -- init identity: fresh build vs saved (untrained) init states ---------
    init_identity: dict[str, Any] = {"arms": {}, "fresh_source": "zw.build_arm(arm)"}
    for arm in ARMS:
        old_init = torch.load(PREV_RESULTS / f"{arm}_init_state.pt", map_location="cpu", weights_only=False)
        old_last = torch.load(PREV_RESULTS / f"{arm}_last_state.pt", map_location="cpu", weights_only=False)
        old_soup = torch.load(PREV_RESULTS / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False)
        fresh = build_arm(arm)
        fresh_sd = {key: value.detach().clone() for key, value in fresh.state_dict().items()}
        mismatch = [key for key in old_init if not torch.equal(old_init[key], fresh_sd[key])]
        init_identity["arms"][arm] = {
            "n_tensors": len(old_init),
            "fresh_vs_saved_init_mismatch_keys": mismatch,
            "saved_init_hash": state_hash(old_init),
            "fresh_init_hash": state_hash(fresh_sd),
            "init_differs_from_last_tensors": int(sum(
                not torch.equal(old_init[key], old_last[key]) for key in old_init
            )),
            "init_differs_from_soup_tensors": int(sum(
                not torch.equal(old_init[key], old_soup[key]) for key in old_init
            )),
        }
    pair = build_pair_hashes()
    init_identity["shared_body"] = {
        "shared_tensor_count": pair["shared_tensor_count"],
        "shared_state_hash": pair["shared_state_hash"],
        "published_shared_init_sha256": json.loads((PREV_RESULTS / "analysis.json").read_text())["init_pair_check"]["shared_init_sha256"],
        "body_parameters": pair["body_parameters"],
        "bridge_parameters": pair["bridge_parameters"],
        "total_parameters": pair["total_parameters"],
    }
    # initial D and M functions differ on a fixed fit batch (expected, not claimed away).
    _fold, _prep, fit_data, _dev, _tgt = load_prepared()
    g_fit = _tgt["g"][_fold["fit_idx"]]
    fixed_target = torch.as_tensor(g_fit[:128], dtype=torch.float32)
    batch = zftd.make_batch(fit_data, list(range(128)), fixed_target, torch.device("cpu"))
    with torch.no_grad():
        out_d = build_arm("D")(batch, mask=cm.C6_MASK).view(-1).numpy()
        out_m = build_arm("M")(batch, mask=cm.C6_MASK).view(-1).numpy()
    init_identity["initial_function_gap"] = {
        "fixed_batch": "first 128 fit rows",
        "max_abs_diff_D_vs_M": float(np.max(np.abs(out_d - out_m))),
        "rms_diff_D_vs_M": float(np.sqrt(np.mean((out_d - out_m) ** 2))),
        "note": "initial functions are not identical; no function matching is claimed",
    }
    write_json(RESULTS_DIR / "init_identity.json", init_identity)

    # -- input manifest ------------------------------------------------------
    sources = {
        "encoded_train": zjd.TRACK_ROOT / "results/zinc_static_dictionary_pair/cache/encoded_train.pt",
        "env_train": zjd.TRACK_ROOT / "results/e2e_dictenv_p1/cache/env_train.pt",
        "fold_objects": zw.PREP_BLOB,
        "audit_formula_verification": AUDIT_DIR / "formula_verification_per_molecule.csv",
        "audit_train_label": AUDIT_DIR / "train_cycle_audit_label.csv",
        "audit_stage_refine": AUDIT_DIR / "stage_refine.json",
        "handoff_train": zftd.HANDOFF_TRAIN,
        "prev_D_init_state": PREV_RESULTS / "D_init_state.pt",
        "prev_M_init_state": PREV_RESULTS / "M_init_state.pt",
        "prev_D_raw_predictions": PREV_RESULTS / "D_raw_predictions.npz",
        "prev_M_raw_predictions": PREV_RESULTS / "M_raw_predictions.npz",
        "prev_D_meta": PREV_RESULTS / "D_meta.json",
        "prev_M_meta": PREV_RESULTS / "M_meta.json",
        "prev_analysis": PREV_RESULTS / "analysis.json",
        "target_npz": TARGETS_NPZ,
    }
    manifest = {
        "files": {
            name: {
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() else None,
                "bytes": int(path.stat().st_size) if path.exists() else None,
            }
            for name, path in sources.items()
        },
        "missing": [name for name, path in sources.items() if not path.exists()],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(RESULTS_DIR / "input_manifest.json", manifest)

    protocol = json.loads((RESULTS_DIR / "protocol.json").read_text()) if (RESULTS_DIR / "protocol.json").exists() else {}
    out = {
        "target_provenance": checks,
        "historical_anchor": anchor,
        "init_identity": init_identity,
        "input_manifest_missing": manifest["missing"],
        "seconds": float(time.perf_counter() - t0),
    }
    return out


# ---------------------------------------------------------------------------
# training one g arm
# ---------------------------------------------------------------------------


def train_arm_g(
    arm: str,
    *,
    device: torch.device,
    out_dir: Path,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    schedule, schedule_hash = build_schedule(len(fit_data), int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)

    model = build_arm(arm).to(device)
    audit = {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "body_parameters": zw.body_parameter_count(model),
        "bridge_parameters": zw.bridge_parameter_count(model),
    }
    seed_everything(SEED)
    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    steps_done = 0
    stream = hashlib.sha256()
    started = time.perf_counter()
    stopped_reason = "completed"
    bridge_grad_log: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, n_mol, n_steps, gnorm_sum, clip_hits = 0.0, 0, 0, 0.0, 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = schedule[epoch - 1][start : start + BATCH_SIZE]
            index_list = [int(i) for i in indices.tolist()]
            stream.update(np.asarray(index_list, np.int64).tobytes())
            batch = zftd.make_batch(fit_data, index_list, target_fit, device)
            prediction = model(batch, mask=cm.C6_MASK)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if epoch in (1, 120, 240):
                entry = {"epoch": int(epoch)}
                for name, parameter in model.named_parameters():
                    if name.startswith("local_dictionary_bridge.") and parameter.grad is not None:
                        entry[name] = float(parameter.grad.norm())
                bridge_grad_log.append(entry)
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "clip_fraction": float(clip_hits / max(n_steps, 1)),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if epoch in soup_epochs:
            soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[{arm}:g] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"gnorm={curve[-1]['grad_norm']:.3g} clip={curve[-1]['clip_fraction']:.3f} "
                f"{curve[-1]['seconds']:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    if not soup and max_steps is not None:
        soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    health: dict[str, Any] = {"fit": {}, "dev": {}}
    for split_name, data_list in (("fit", fit_data[:512]), ("dev", dev_data[:512])):
        batch = zftd.make_batch(data_list, list(range(min(128, len(data_list)))), torch.zeros(len(data_list)), device)
        health[split_name] = zw._bridge_aux(model, batch)
    if arm == "D":
        bridge = model.local_dictionary_bridge
        health["param_change"] = {
            "D_L_rel": float((bridge.D_L.detach().cpu() - init_state["local_dictionary_bridge.D_L"]).norm() / init_state["local_dictionary_bridge.D_L"].norm()),
            "V_L_rel": float((bridge.V_L.detach().cpu() - init_state["local_dictionary_bridge.V_L"]).norm() / init_state["local_dictionary_bridge.V_L"].norm()),
        }
    else:
        bridge = model.local_dictionary_bridge
        health["param_change"] = {
            "fc1_rel": float((bridge.fc1.weight.detach().cpu() - init_state["local_dictionary_bridge.fc1.weight"]).norm() / init_state["local_dictionary_bridge.fc1.weight"].norm()),
            "fc2_rel": float((bridge.fc2.weight.detach().cpu() - init_state["local_dictionary_bridge.fc2.weight"]).norm() / init_state["local_dictionary_bridge.fc2.weight"].norm()),
        }

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "supervision_target": "g = y - c (fit-only decomposition)",
        "seed": SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "steps_expected": int(epochs) * math.ceil(len(fit_data) / BATCH_SIZE),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "soup_epochs": members if max_steps is None else [],
        "n_fit": int(len(fit_data)),
        "n_dev": int(len(dev_data)),
        "fold": {
            "seed": FOLD_SEED,
            "fit_idx_sha256": fold["fit_idx_sha256"],
            "dev_idx_sha256": fold["dev_idx_sha256"],
        },
        "target": {
            "source": str(TARGETS_NPZ),
            "sha256": file_sha256(TARGETS_NPZ),
            "definition": "g = y - c; c=(k-mu_cycle)/sigma_cycle; constants fitted on 8000 fit rows only",
            "target_sha256": hashlib.sha256(np.ascontiguousarray(g, np.float64).tobytes()).hexdigest(),
        },
        "prep": {key: value for key, value in prep_meta.items() if not isinstance(value, np.ndarray)},
        "parameter_audit": audit,
        "init_state_hash": state_hash(init_state),
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "schedule_sha256": schedule_hash,
        "data_stream_sha256": stream.hexdigest(),
        "curve": curve,
        "bridge_grad_log": bridge_grad_log,
        "health": health,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if max_steps is not None:
        result["smoke_last_state_hash"] = state_hash(last_state)
        return result

    torch.save(init_state, out_dir / f"{arm}_init_state.pt")
    torch.save(last_state, out_dir / f"{arm}_last_state.pt")
    torch.save(soup_state, out_dir / f"{arm}_raw_soup_state.pt")

    raw_predictions: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = build_arm(arm)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        raw_predictions[f"{state_name}_fit"] = evaluate_state(replay, fit_data, g_fit, device)
        raw_predictions[f"{state_name}_dev"] = evaluate_state(replay, dev_data, g_dev, device)
    np.savez_compressed(
        out_dir / f"{arm}_raw_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_predictions.items()},
    )

    b_soup = float(np.median(g_fit - raw_predictions["raw_soup_fit"]))
    b_init = float(np.median(g_fit - raw_predictions["init_fit"]))
    b_last = float(np.median(g_fit - raw_predictions["last_fit"]))
    result["calibration_b"] = {"init": b_init, "last": b_last, "raw_soup": b_soup}

    # fresh-init replay on a fixed batch: CPU vs GPU device equivalence.
    replay = build_arm(arm)
    replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in soup_state.items()}, strict=True)
    batch_data = fit_data[:128]
    gpu_pred = evaluate_state(replay.to(device), batch_data, g_fit[:128], device)
    cpu_pred = evaluate_state(replay.to(torch.device("cpu")), batch_data, g_fit[:128], torch.device("cpu"))
    max_diff = float(np.max(np.abs(gpu_pred - cpu_pred))) if device.type != "cpu" else 0.0
    result["replay_max_abs_diff"] = max_diff
    if device.type != "cpu" and max_diff > REPLAY_TOL:
        raise RuntimeError(f"CPU/GPU replay maxdiff {max_diff} > {REPLAY_TOL}")
    write_json(out_dir / f"{arm}_meta.json", result)
    write_json(out_dir / f"{arm}_curve.json", curve)
    log(f"[{arm}:g] done steps={steps_done} wall={result['wall_clock_s']:.1f}s b={b_soup:.6f} replay={max_diff:.2e}")
    return result


# ---------------------------------------------------------------------------
# smoke checks (engineering only; states discarded)
# ---------------------------------------------------------------------------


def run_smoke(*, device: torch.device, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    y = np.asarray(tgt["y"], np.float64)
    g_fit = g[fold["fit_idx"]]
    checks: dict[str, Any] = {}

    checks["target_contract"] = {
        "g_equals_y_minus_c_max_abs": float(np.max(np.abs(g - (y - np.asarray(tgt["c"], np.float64))))),
        "g_sha256": hashlib.sha256(np.ascontiguousarray(g, np.float64).tobytes()).hexdigest(),
        "npz_sha256": file_sha256(TARGETS_NPZ),
        "fold_hashes_expected": bool(
            fold["fit_idx_sha256"] == "7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9"
            and fold["dev_idx_sha256"] == "a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    pair = build_pair_hashes()
    checks["shared_body"] = {
        "shared_tensor_count": pair["shared_tensor_count"],
        "shared_state_hash": pair["shared_state_hash"],
        "body_parameters": pair["body_parameters"],
        "bridge_parameters": pair["bridge_parameters"],
        "total_parameters": pair["total_parameters"],
    }
    checks["parameter_contract"] = bool(
        pair["body_parameters"] == EXPECTED_BODY_PARAMETERS
        and pair["bridge_parameters"] == BRIDGE_PARAMETERS
        and pair["total_parameters"] == EXPECTED_TOTAL_PARAMETERS
    )

    model_d = build_arm("D").to(device)
    model_m = build_arm("M").to(device)
    model_d.eval()
    model_m.eval()
    g_target = torch.as_tensor(g_fit[:512], dtype=torch.float32)
    y_target = torch.as_tensor(y[fold["fit_idx"]][:512], dtype=torch.float32)
    batch_g = zftd.make_batch(fit_data, list(range(32)), g_target, device)
    batch_y = zftd.make_batch(fit_data, list(range(32)), y_target, device)

    captured: list[torch.Tensor] = []
    handle = model_d.local_dictionary_bridge.register_forward_hook(lambda m, i, o: captured.append(o.detach().clone()))
    with torch.no_grad():
        _, aux_d = model_d(batch_g, mask=cm.C6_MASK, return_aux=True)
    handle.remove()
    checks["dictionary_consumer_bridge_output_maxdiff"] = float((captured[0] - aux_d["E"]).abs().max())
    checks["dictionary_consumer_ok"] = checks["dictionary_consumer_bridge_output_maxdiff"] < 1e-6

    captured_m: list[torch.Tensor] = []
    handle = model_m.local_dictionary_bridge.register_forward_hook(lambda m, i, o: captured_m.append(o.detach().clone()))
    with torch.no_grad():
        _, aux_m = model_m(batch_g, mask=cm.C6_MASK, return_aux=True)
    handle.remove()
    checks["mlp_consumer_bridge_output_maxdiff"] = float((captured_m[0] - aux_m["E"]).abs().max())
    checks["mlp_consumer_ok"] = checks["mlp_consumer_bridge_output_maxdiff"] < 1e-6

    # target is never a model input: swapping g for y cannot change predictions.
    with torch.no_grad():
        pred_dg = model_d(batch_g, mask=cm.C6_MASK)
        pred_dy = model_d(batch_y, mask=cm.C6_MASK)
        pred_mg = model_m(batch_g, mask=cm.C6_MASK)
        pred_my = model_m(batch_y, mask=cm.C6_MASK)
    checks["target_not_read_d_maxdiff"] = float((pred_dg - pred_dy).abs().max())
    checks["target_not_read_m_maxdiff"] = float((pred_mg - pred_my).abs().max())
    checks["target_not_read_ok"] = bool(
        checks["target_not_read_d_maxdiff"] == 0.0 and checks["target_not_read_m_maxdiff"] == 0.0
    )

    # the training loss really is L1 against g.
    loss_d = F.l1_loss(pred_dg.view(-1), batch_g.y.view(-1))
    loss_d_wrong = F.l1_loss(pred_dg.view(-1), batch_y.y.view(-1))
    checks["loss_is_l1_g"] = float(loss_d)
    checks["loss_with_y_target"] = float(loss_d_wrong)
    checks["loss_target_semantics_ok"] = bool(abs(float(loss_d) - float((pred_dg.view(-1) - batch_g.y.view(-1)).abs().mean())) < 1e-9
                                              and abs(float(loss_d) - float(loss_d_wrong)) > 1e-6)

    # batched endpoint offsets.
    groups = list(range(8))
    batch_grp = zftd.make_batch(fit_data, groups, g_target, device)
    with torch.no_grad():
        grouped_d = model_d(batch_grp, mask=cm.C6_MASK).cpu().numpy()
        single_d = np.array([
            float(model_d(zftd.make_batch(fit_data, [i], g_target, device), mask=cm.C6_MASK).item()) for i in groups
        ])
        grouped_m = model_m(batch_grp, mask=cm.C6_MASK).cpu().numpy()
        single_m = np.array([
            float(model_m(zftd.make_batch(fit_data, [i], g_target, device), mask=cm.C6_MASK).item()) for i in groups
        ])
    checks["endpoint_offset_d_maxdiff"] = float(np.max(np.abs(grouped_d - single_d)))
    checks["endpoint_offset_m_maxdiff"] = float(np.max(np.abs(grouped_m - single_m)))
    checks["endpoint_offset_ok"] = bool(
        checks["endpoint_offset_d_maxdiff"] < 1e-6 and checks["endpoint_offset_m_maxdiff"] < 1e-6
    )

    for arm, model in (("D", model_d), ("M", model_m)):
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        prediction = model(batch_g, mask=cm.C6_MASK)
        loss = F.l1_loss(prediction.view(-1), batch_g.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        changed = [key for key, value in model.state_dict().items() if not torch.equal(value, before[key])]
        checks[f"{arm}_optimizer_step_changed_tensors"] = len(changed)
        checks[f"{arm}_loss_value"] = float(loss.detach())
    checks["optimizer_ok"] = bool(
        checks["D_optimizer_step_changed_tensors"] > 0 and checks["M_optimizer_step_changed_tensors"] > 0
    )

    for arm, model in (("D", model_d), ("M", model_m)):
        bridge = model.local_dictionary_bridge
        old = {key: value.detach().clone() for key, value in bridge.state_dict().items()}
        with torch.no_grad():
            original = model(batch_g, mask=cm.C6_MASK)
            for value in bridge.parameters():
                value.add_(0.01)
            perturbed = model(batch_g, mask=cm.C6_MASK)
        checks[f"{arm}_bridge_perturbation_maxdiff"] = float((perturbed - original).abs().max())
        bridge.load_state_dict(old)
    checks["consumers_depend_on_bridge"] = bool(
        checks["D_bridge_perturbation_maxdiff"] > 0.0 and checks["M_bridge_perturbation_maxdiff"] > 0.0
    )
    checks["no_structural_aux_loss"] = True
    checks["fold"] = {
        "fit_idx_sha256": fold["fit_idx_sha256"],
        "dev_idx_sha256": fold["dev_idx_sha256"],
        "n_fit": fold["n_fit"],
        "n_dev": fold["n_dev"],
    }
    checks["prep_meta"] = {key: value for key, value in prep_meta.items() if not isinstance(value, np.ndarray)}

    t0 = time.perf_counter()
    res_d = train_arm_g("D", device=device, out_dir=out_dir / "smoke", epochs=1, max_steps=3, log=log)
    res_m = train_arm_g("M", device=device, out_dir=out_dir / "smoke", epochs=1, max_steps=3, log=log)
    checks["tiny_train"] = {
        "D_steps": res_d["steps_done"],
        "M_steps": res_m["steps_done"],
        "D_loss_curve": [entry["train_task_mae"] for entry in res_d["curve"]],
        "M_loss_curve": [entry["train_task_mae"] for entry in res_m["curve"]],
        "schedule_match": bool(res_d["schedule_sha256"] == res_m["schedule_sha256"]),
        "target_kind": res_d["supervision_target"],
        "target_sha256_D": res_d["target"]["target_sha256"],
        "target_sha256_M": res_m["target"]["target_sha256"],
        "seconds": float(time.perf_counter() - t0),
    }
    checks["all_ok"] = bool(
        checks["parameter_contract"]
        and checks["target_contract"]["g_equals_y_minus_c_max_abs"] < 1e-12
        and checks["dictionary_consumer_ok"]
        and checks["mlp_consumer_ok"]
        and checks["target_not_read_ok"]
        and checks["loss_target_semantics_ok"]
        and checks["endpoint_offset_ok"]
        and checks["optimizer_ok"]
        and checks["consumers_depend_on_bridge"]
        and checks["tiny_train"]["schedule_match"]
    )
    write_json(out_dir / "smoke_checks.json", checks)
    if not checks["all_ok"]:
        raise RuntimeError("smoke checks failed")
    log(f"[smoke] all_ok={checks['all_ok']} in {time.perf_counter() - t0:.1f}s")
    return checks


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------


def _metric_table(error: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(k.shape[0]), "mae": float(np.mean(np.abs(error))) if error.size else None}
    total = int(k.shape[0])
    for name, mask in group_masks(k).items():
        n = int(mask.sum())
        out[name] = {
            "n": n,
            "mae": float(np.mean(np.abs(error[mask]))) if n else None,
            "contribution": float(np.abs(error[mask]).sum() / total) if total else None,
        }
    return out


def _bootstrap_ci(values: np.ndarray, *, seed: int, n_boot: int) -> dict[str, Any]:
    point = float(values.mean())
    rng = np.random.default_rng(int(seed))
    n = values.size
    draws = np.empty(int(n_boot), np.float64)
    for b in range(int(n_boot)):
        idx = rng.choice(n, size=n, replace=True)
        draws[b] = values[idx].mean()
    return {
        "point": point,
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def _four_arm_bootstrap(
    err_dy: np.ndarray,
    err_my: np.ndarray,
    err_dg: np.ndarray,
    err_mg: np.ndarray,
    *,
    seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Pair bootstrap with shared resample indices for G_y, G_g, B_D, B_M, I."""
    e_dy = np.abs(np.asarray(err_dy, np.float64))
    e_my = np.abs(np.asarray(err_my, np.float64))
    e_dg = np.abs(np.asarray(err_dg, np.float64))
    e_mg = np.abs(np.asarray(err_mg, np.float64))
    n = e_dy.size
    if not (e_my.size == e_dg.size == e_mg.size == n):
        raise RuntimeError("four-arm bootstrap arrays must share length")

    def points(idx: np.ndarray | None = None) -> dict[str, float]:
        def mean(a: np.ndarray) -> float:
            return float(a.mean()) if idx is None else float(a[idx].mean())
        gy = mean(e_my) - mean(e_dy)
        gg = mean(e_mg) - mean(e_dg)
        bd = mean(e_dy) - mean(e_dg)
        bm = mean(e_my) - mean(e_mg)
        return {"G_y": gy, "G_g": gg, "B_D": bd, "B_M": bm, "I": bd - bm}

    rng = np.random.default_rng(int(seed))
    samples = {key: np.empty(int(n_boot), np.float64) for key in ("G_y", "G_g", "B_D", "B_M", "I")}
    for b in range(int(n_boot)):
        idx = rng.choice(n, size=n, replace=True)
        pts = points(idx)
        for key, value in pts.items():
            samples[key][b] = value
    out: dict[str, Any] = {"n": int(n), "seed": int(seed), "n_boot": int(n_boot), "shared_indices": True}
    point = points(None)
    for key in ("G_y", "G_g", "B_D", "B_M", "I"):
        out[key] = {
            "point": point[key],
            "ci95": [float(np.percentile(samples[key], 2.5)), float(np.percentile(samples[key], 97.5))],
        }
    out["identity_checks"] = {
        "I_equals_G_g_minus_G_y_point": abs(point["I"] - (point["G_g"] - point["G_y"])) < 1e-12,
        "I_equals_G_g_minus_G_y_bootstrap_max_abs": float(np.max(np.abs(samples["I"] - (samples["G_g"] - samples["G_y"])))),
        "I_equals_B_D_minus_B_M_bootstrap_max_abs": float(np.max(np.abs(samples["I"] - (samples["B_D"] - samples["B_M"])))),
    }
    return out


def _bootstrap_self_tests(err_d: np.ndarray, err_m: np.ndarray) -> dict[str, Any]:
    same = _bootstrap_ci(np.abs(np.asarray(err_d)) - np.abs(np.asarray(err_d)), seed=BOOT_SEED, n_boot=N_BOOT)
    swapped = _bootstrap_ci(np.abs(np.asarray(err_d)) - np.abs(np.asarray(err_m)), seed=BOOT_SEED, n_boot=N_BOOT)
    forward = _bootstrap_ci(np.abs(np.asarray(err_m)) - np.abs(np.asarray(err_d)), seed=BOOT_SEED, n_boot=N_BOOT)
    shift = _bootstrap_ci((np.abs(err_d) + 0.001) - np.abs(err_m), seed=BOOT_SEED, n_boot=N_BOOT)
    return {
        "same_predictions": same,
        "swapped": swapped,
        "forward_sanity": forward,
        "constant_shift_one_sided": shift,
        "checks": {
            "same_predictions_zero": bool(same["point"] == 0.0 and same["ci95"] == [0.0, 0.0]),
            "swapped_mirror_point": bool(abs(swapped["point"] + forward["point"]) < 1e-12),
            "swapped_mirror_ci": bool(
                abs(swapped["ci95"][0] + forward["ci95"][1]) < 1e-9
                and abs(swapped["ci95"][1] + forward["ci95"][0]) < 1e-9
            ),
            "constant_shift_bounded": bool(shift["point"] <= forward["point"] + 0.001 + 1e-9),
        },
    }


def _ci_inside(ci: Sequence[float], low: float, high: float) -> bool:
    return bool(ci[0] >= low and ci[1] <= high)


# ---------------------------------------------------------------------------
# analysis (Phase C)
# ---------------------------------------------------------------------------


def analyze(*, out_dir: Path, device: torch.device, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    y = np.asarray(tgt["y"], np.float64)
    c = np.asarray(tgt["c"], np.float64)
    g = np.asarray(tgt["g"], np.float64)
    k = np.asarray(tgt["k"], np.int64)
    fit_idx, dev_idx = fold["fit_idx"], fold["dev_idx"]
    y_fit, y_dev = y[fit_idx], y[dev_idx]
    g_fit, g_dev = g[fit_idx], g[dev_idx]
    c_fit, c_dev = c[fit_idx], c[dev_idx]
    k_fit, k_dev = k[fit_idx], k[dev_idx]

    new_meta = {arm: json.loads((out_dir / f"{arm}_meta.json").read_text()) for arm in ARMS}
    new_pred = {}
    for arm in ARMS:
        with np.load(out_dir / f"{arm}_raw_predictions.npz") as z:
            new_pred[arm] = {key: np.asarray(z[key], np.float64) for key in z.files}
    prev_meta = {arm: json.loads((PREV_RESULTS / f"{arm}_meta.json").read_text()) for arm in ARMS}
    prev_pred = {arm: _reload_prev_predictions(arm) for arm in ARMS}
    published = json.loads((PREV_RESULTS / "analysis.json").read_text())

    contract_errors: list[str] = []
    for arm in ARMS:
        meta = new_meta[arm]
        if meta["parameter_audit"] != {
            "total_parameters": EXPECTED_TOTAL_PARAMETERS,
            "body_parameters": EXPECTED_BODY_PARAMETERS,
            "bridge_parameters": BRIDGE_PARAMETERS,
        }:
            contract_errors.append(f"{arm}: parameter audit mismatch")
        if meta["steps_expected"] != 15120 or meta["steps_done"] != 15120:
            contract_errors.append(f"{arm}: step count {meta['steps_done']} != 15120")
        if meta["stopped_reason"] != "completed":
            contract_errors.append(f"{arm}: stopped_reason={meta['stopped_reason']}")
        if meta["official_valid_loaded"] or meta["official_test_loaded"]:
            contract_errors.append(f"{arm}: official split loaded")
        if abs(meta["replay_max_abs_diff"]) > REPLAY_TOL:
            contract_errors.append(f"{arm}: replay maxdiff {meta['replay_max_abs_diff']} > {REPLAY_TOL}")
        for part in ("fit", "dev"):
            arr = new_pred[arm][f"raw_soup_{part}"]
            if arr.ndim != 1 or arr.shape[0] != (N_FIT if part == "fit" else N_DEV):
                contract_errors.append(f"{arm}: {part} prediction shape {arr.shape}")
    if new_meta["D"]["schedule_sha256"] != new_meta["M"]["schedule_sha256"]:
        contract_errors.append("schedule hash mismatch")
    schedule_local, schedule_local_hash = build_schedule(len(fit_data), EPOCHS, SEED + TRAIN_SHUFFLE_OFFSET)
    if schedule_local_hash != new_meta["D"]["schedule_sha256"]:
        contract_errors.append("schedule regeneration mismatch")
    if schedule_local_hash != "7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65":
        contract_errors.append("schedule hash does not match the frozen y round")
    if new_meta["D"]["data_stream_sha256"] != new_meta["M"]["data_stream_sha256"]:
        contract_errors.append("data stream hash mismatch")
    if new_meta["D"]["fold"]["fit_idx_sha256"] != new_meta["M"]["fold"]["fit_idx_sha256"]:
        contract_errors.append("fold mismatch")
    if new_meta["D"]["target"]["target_sha256"] != new_meta["M"]["target"]["target_sha256"]:
        contract_errors.append("target mismatch between arms")
    np.savez_compressed(out_dir / "fold_indices.npz", fit_idx=fit_idx, dev_idx=dev_idx)
    np.savez_compressed(
        out_dir / "new_fit_prep.npz",
        **{key: value for key, value in prep_meta.items() if isinstance(value, np.ndarray)},
    )
    np.savez_compressed(
        out_dir / "targets.npz",
        y=y, c=c, g=g, k=k, fit_idx=fit_idx, dev_idx=dev_idx, gid=np.asarray(tgt["gid"], np.int64),
    )

    b_new = {arm: float(new_meta[arm]["calibration_b"]["raw_soup"]) for arm in ARMS}
    b_prev = {arm: float(prev_meta[arm]["calibration_b"]["raw_soup"]) for arm in ARMS}

    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "fold": {
            "fit_idx_sha256": fold["fit_idx_sha256"],
            "dev_idx_sha256": fold["dev_idx_sha256"],
            "n_fit": fold["n_fit"],
            "n_dev": fold["n_dev"],
            "fit_k_counts": {name: int(mask.sum()) for name, mask in group_masks(k_fit).items()},
            "dev_k_counts": {name: int(mask.sum()) for name, mask in group_masks(k_dev).items()},
        },
        "c_g0_value": float(np.median(c_dev[k_dev == 0])),
        "c_g0_spread_max_abs": float(np.max(np.abs(c_dev[k_dev == 0] - np.median(c_dev[k_dev == 0])))),
        "calibration_b": {"new": b_new, "prev_y": b_prev},
        "contract_errors": contract_errors,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "targets": {
            "definition": "g = y - c; c=(k-mu_cycle)/sigma_cycle; constants fitted on fit rows only",
            "npz_sha256": file_sha256(out_dir / "targets.npz"),
        },
    }

    # -- per-state metrics ----------------------------------------------------
    for state in ("init", "last", "raw_soup"):
        for part, _target, kk in (("fit", g_fit, k_fit), ("dev", g_dev, k_dev)):
            for arm in ARMS:
                raw = new_pred[arm][f"{state}_{part}"]
                cal = raw + b_new[arm]
                report.setdefault("state_metrics", {}).setdefault(state, {}).setdefault(part, {})[arm] = {
                    "raw": _metric_table(_target - raw, kk),
                    "cal": _metric_table(_target - cal, kk),
                }
    for part, target, kk, part_idx in (("fit", g_fit, k_fit, fit_idx), ("dev", g_dev, k_dev, dev_idx)):
        for arm in ARMS:
            raw = new_pred[arm][f"raw_soup_{part}"]
            cal = raw + b_new[arm]
            report.setdefault("arms", {}).setdefault(part, {})[arm] = {
                "label": "g",
                "raw": _metric_table(target - raw, kk),
                "cal": _metric_table(target - cal, kk),
                "b": b_new[arm],
            }
        for arm in ARMS:
            raw = prev_pred[arm][f"raw_soup_{part}"]
            cal = raw + b_prev[arm]
            report.setdefault("arms_y_reference", {}).setdefault(part, {})[arm] = {
                "label": "y",
                "raw": _metric_table((y_fit if part == "fit" else y_dev) - raw, kk),
                "cal": _metric_table((y_fit if part == "fit" else y_dev) - cal, kk),
                "b": b_prev[arm],
            }

    # -- G0 / overall gains on the new g arms --------------------------------
    g0 = k_dev == 0
    err_dg_raw = g_dev - new_pred["D"]["raw_soup_dev"]
    err_mg_raw = g_dev - new_pred["M"]["raw_soup_dev"]
    err_dg_cal = g_dev - (new_pred["D"]["raw_soup_dev"] + b_new["D"])
    err_mg_cal = g_dev - (new_pred["M"]["raw_soup_dev"] + b_new["M"])
    report["gains"] = {
        "dev_overall_cal": _bootstrap_ci(np.abs(err_mg_cal) - np.abs(err_dg_cal), seed=BOOT_SEED, n_boot=N_BOOT),
        "dev_overall_raw": _bootstrap_ci(np.abs(err_mg_raw) - np.abs(err_dg_raw), seed=BOOT_SEED, n_boot=N_BOOT),
        "dev_G0_cal": _bootstrap_ci((np.abs(err_mg_cal) - np.abs(err_dg_cal))[g0], seed=BOOT_SEED, n_boot=N_BOOT),
        "dev_G0_raw": _bootstrap_ci((np.abs(err_mg_raw) - np.abs(err_dg_raw))[g0], seed=BOOT_SEED, n_boot=N_BOOT),
        "fit_overall_cal": _bootstrap_ci(
            np.abs(g_fit - (new_pred["M"]["raw_soup_fit"] + b_new["M"])) - np.abs(g_fit - (new_pred["D"]["raw_soup_fit"] + b_new["D"])),
            seed=BOOT_SEED, n_boot=N_BOOT,
        ),
        "fit_G0_cal": _bootstrap_ci(
            (np.abs(g_fit - (new_pred["M"]["raw_soup_fit"] + b_new["M"])) - np.abs(g_fit - (new_pred["D"]["raw_soup_fit"] + b_new["D"])))[k_fit == 0],
            seed=BOOT_SEED, n_boot=N_BOOT,
        ),
    }

    # -- four-arm G0 comparison and interaction ------------------------------
    err_dy_cal = y_dev - (prev_pred["D"]["raw_soup_dev"] + b_prev["D"])
    err_my_cal = y_dev - (prev_pred["M"]["raw_soup_dev"] + b_prev["M"])
    err_dy_raw = y_dev - prev_pred["D"]["raw_soup_dev"]
    err_my_raw = y_dev - prev_pred["M"]["raw_soup_dev"]
    four_cal = _four_arm_bootstrap(err_dy_cal[g0], err_my_cal[g0], err_dg_cal[g0], err_mg_cal[g0])
    four_raw = _four_arm_bootstrap(err_dy_raw[g0], err_my_raw[g0], err_dg_raw[g0], err_mg_raw[g0])
    interaction = {
        "endpoint": "new-dev k=0 (G0) rows; y=g+c0 with c0 constant on G0",
        "c0": float(np.median(c_dev[g0])),
        "cal": four_cal,
        "raw": four_raw,
        "published_G_y_cal": published["gains"]["dev_G0_cal"],
        "G_y_cal_reproduced": {
            "point": four_cal["G_y"]["point"],
            "published_point": float(published["gains"]["dev_G0_cal"]["point"]),
            "delta": four_cal["G_y"]["point"] - float(published["gains"]["dev_G0_cal"]["point"]),
        },
    }
    report["interaction"] = interaction

    # -- contributions per group ---------------------------------------------
    for arm in ARMS:
        err = g_dev - (new_pred[arm]["raw_soup_dev"] + b_new[arm])
        report.setdefault("contributions", {})[arm] = _metric_table(err, k_dev)
        report["contributions"][arm + "_raw"] = _metric_table(g_dev - new_pred[arm]["raw_soup_dev"], k_dev)
    for arm in ARMS:
        err = y_dev - (prev_pred[arm]["raw_soup_dev"] + b_prev[arm])
        report["contributions"][arm + "_y_reference"] = _metric_table(err, k_dev)

    # -- sensitivity: drop the single worst combined new-dev row -------------
    combined = np.abs(err_dg_cal) + np.abs(err_mg_cal)
    drop = int(np.argmax(combined))
    keep = np.ones(k_dev.size, dtype=bool)
    keep[drop] = False
    report["sensitivity"] = {
        "drop_rule": "max(|err_D_g_cal| + |err_M_g_cal|) on new dev",
        "dropped_row_stable_id": int(dev_idx[drop]),
        "dropped_combined_error": float(combined[drop]),
        "overall_cal_gain_without_row": float(np.abs(err_mg_cal[keep]).mean() - np.abs(err_dg_cal[keep]).mean()),
        "overall_raw_gain_without_row": float(np.abs(err_mg_raw[keep]).mean() - np.abs(err_dg_raw[keep]).mean()),
        "G0_cal_gain_without_row": float(
            np.abs(err_mg_cal[keep & g0]).mean() - np.abs(err_dg_cal[keep & g0]).mean()
        ),
    }

    # -- bootstrap self-tests -------------------------------------------------
    report["bootstrap_self_tests"] = _bootstrap_self_tests(err_dg_cal, err_mg_cal)

    # -- gate -----------------------------------------------------------------
    gg_cal = report["gains"]["dev_G0_cal"]
    gg_raw = report["gains"]["dev_G0_raw"]
    ov_cal = report["gains"]["dev_overall_cal"]
    ov_raw = report["gains"]["dev_overall_raw"]
    d_support = bool(
        gg_cal["point"] >= DELTA and gg_cal["ci95"][0] > 0 and gg_raw["point"] > 0 and ov_cal["point"] >= -DELTA_SLACK
    )
    m_support = bool(
        gg_cal["point"] <= -DELTA and gg_cal["ci95"][1] < 0 and gg_raw["point"] < 0 and ov_cal["point"] <= DELTA_SLACK
    )
    raw_opposite = bool(
        (np.sign(gg_raw["point"]) == -np.sign(gg_cal["point"]) and abs(gg_raw["point"]) >= DELTA)
        or (np.sign(ov_raw["point"]) == -np.sign(ov_cal["point"]) and abs(ov_raw["point"]) >= DELTA)
    )
    local_equivalence = bool(
        _ci_inside(gg_cal["ci95"], -DELTA, DELTA)
        and _ci_inside(ov_cal["ci95"], -DELTA, DELTA)
        and not raw_opposite
    )
    tradeoff = bool(
        (gg_cal["point"] >= DELTA and gg_cal["ci95"][0] > 0 and gg_raw["point"] > 0 and ov_cal["point"] < -DELTA_SLACK)
        or (gg_cal["point"] <= -DELTA and gg_cal["ci95"][1] < 0 and gg_raw["point"] < 0 and ov_cal["point"] > DELTA_SLACK)
    )
    if contract_errors:
        category = "INVALID"
    elif d_support:
        category = "D_CHEM_SUPPORT"
    elif m_support:
        category = "M_CHEM_SUPPORT"
    elif local_equivalence:
        category = "LOCAL_EQUIVALENCE"
    elif tradeoff:
        category = "TRADEOFF"
    else:
        category = "INCONCLUSIVE"
    report["gate"] = {
        "category": category,
        "definitions_checked": {
            "D_CHEM_SUPPORT": d_support,
            "M_CHEM_SUPPORT": m_support,
            "LOCAL_EQUIVALENCE": local_equivalence,
            "TRADEOFF": tradeoff,
        },
        "rules": {
            "D_CHEM_SUPPORT": "G_g cal>=+0.003 & CI_low>0; G0 raw gain>0; overall g-cal gain>=-0.001",
            "M_CHEM_SUPPORT": "G_g cal<=-0.003 & CI_high<0; G0 raw gain<0; overall g-cal gain<=+0.001",
            "LOCAL_EQUIVALENCE": "G0 and overall g-cal CIs inside [-0.003,+0.003]; no raw>=0.003 opposite signal",
            "TRADEOFF": "G0 cal>=0.003, CI separated, raw same direction, winner overall g-cal worsens >0.001",
        },
        "raw_opposite_signal": raw_opposite,
        "primary_metrics": {
            "G_g_cal": gg_cal,
            "G_g_raw": gg_raw,
            "overall_g_cal": ov_cal,
            "overall_g_raw": ov_raw,
        },
    }

    # -- relief classification (section 7B) -----------------------------------
    cal = four_cal
    raw = four_raw
    bd_ok = bool(
        cal["B_D"]["point"] >= DELTA and cal["B_D"]["ci95"][0] > 0
        and cal["B_M"]["point"] >= DELTA and cal["B_M"]["ci95"][0] > 0
        and raw["B_D"]["point"] > 0 and raw["B_M"]["point"] > 0
    )
    d_relief = bool(cal["I"]["point"] >= DELTA and cal["I"]["ci95"][0] > 0 and raw["I"]["point"] > 0)
    m_relief = bool(cal["I"]["point"] <= -DELTA and cal["I"]["ci95"][1] < 0 and raw["I"]["point"] < 0)
    if bd_ok:
        relief = "BULK_RELIEF_SUPPORTED"
    elif d_relief:
        relief = "D_SPECIFIC_RELIEF"
    elif m_relief:
        relief = "M_SPECIFIC_RELIEF"
    else:
        relief = "RELIEF_UNCONFIRMED"
    report["relief"] = {
        "category": relief,
        "descriptive": True,
        "note": "old y arms and new g arms share the same code path, schedule and dropout RNG stream; differences are target-only within each bridge family",
        "B_D_cal": cal["B_D"],
        "B_M_cal": cal["B_M"],
        "I_cal": cal["I"],
        "B_D_raw": raw["B_D"],
        "B_M_raw": raw["B_M"],
        "I_raw": raw["I"],
        "checks": {"BULK_RELIEF_SUPPORTED": bd_ok, "D_SPECIFIC_RELIEF": d_relief, "M_SPECIFIC_RELIEF": m_relief},
    }

    # -- tables ---------------------------------------------------------------
    rows = ["endpoint,label,arm,raw_mae,cal_mae,raw_G0,cal_G0,raw_k-1,cal_k-1,raw_k-2,cal_k-2,raw_k<=-3,cal_k<=-3,raw_k<=-2,cal_k<=-2"]
    for part in ("fit", "dev"):
        for arm in ARMS:
            entry = report["arms"][part][arm]
            values = [part, "g", arm, entry["raw"]["mae"], entry["cal"]["mae"]]
            for name in ("k=0", "k=-1", "k=-2", "k<=-3", "k<=-2"):
                values += [entry["raw"][name]["mae"] if entry["raw"][name]["mae"] is not None else "",
                           entry["cal"][name]["mae"] if entry["cal"][name]["mae"] is not None else ""]
            rows.append(",".join(str(v) for v in values))
        for arm in ARMS:
            entry = report["arms_y_reference"][part][arm]
            values = [part, "y(ref)", arm, entry["raw"]["mae"], entry["cal"]["mae"]]
            for name in ("k=0", "k=-1", "k=-2", "k<=-3", "k<=-2"):
                values += [entry["raw"][name]["mae"] if entry["raw"][name]["mae"] is not None else "",
                           entry["cal"][name]["mae"] if entry["cal"][name]["mae"] is not None else ""]
            rows.append(",".join(str(v) for v in values))
    (out_dir / "main_table.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    group_rows = ["arm,label,part,group,n,raw_mae,cal_mae,raw_contribution,cal_contribution"]
    for part in ("fit", "dev"):
        for arm in ARMS:
            for label, entry in (("g", report["arms"][part][arm]), ("y_ref", report["arms_y_reference"][part][arm])):
                for name in ("k=0", "k=-1", "k=-2", "k<=-3", "k<=-2"):
                    group_rows.append(
                        f"{arm},{label},{part},{name},{entry['raw'][name]['n']},"
                        f"{entry['raw'][name]['mae']},{entry['cal'][name]['mae']},"
                        f"{entry['raw'][name]['contribution']},{entry['cal'][name]['contribution']}"
                    )
    (out_dir / "group_table.csv").write_text("\n".join(group_rows) + "\n", encoding="utf-8")

    paired_rows = [
        "stable_id,k,y,g,c,err_D_g_raw,err_M_g_raw,err_D_g_cal,err_M_g_cal,"
        "err_D_y_raw,err_M_y_raw,err_D_y_cal,err_M_y_cal,abs_err_D_y_cal,abs_err_M_y_cal,abs_err_D_g_cal,abs_err_M_g_cal"
    ]
    for pos, stable in enumerate(dev_idx.tolist()):
        paired_rows.append(
            f"train:{int(stable):04d},{int(k_dev[pos])},{y_dev[pos]:.10f},{g_dev[pos]:.10f},{c_dev[pos]:.10f},"
            f"{err_dg_raw[pos]:.10f},{err_mg_raw[pos]:.10f},{err_dg_cal[pos]:.10f},{err_mg_cal[pos]:.10f},"
            f"{err_dy_raw[pos]:.10f},{err_my_raw[pos]:.10f},{err_dy_cal[pos]:.10f},{err_my_cal[pos]:.10f},"
            f"{abs(err_dy_cal[pos]):.10f},{abs(err_my_cal[pos]):.10f},{abs(err_dg_cal[pos]):.10f},{abs(err_mg_cal[pos]):.10f}"
        )
    (out_dir / "paired_g0_gains.csv").write_text("\n".join(paired_rows) + "\n", encoding="utf-8")

    for part, part_idx in (("fit", fit_idx), ("dev", dev_idx)):
        target_g = g_fit if part == "fit" else g_dev
        target_y = y_fit if part == "fit" else y_dev
        kk = k_fit if part == "fit" else k_dev
        cc = c_fit if part == "fit" else c_dev
        lines = ["stable_id,k,y,g,c,D_g_raw,D_g_cal,M_g_raw,M_g_cal,D_y_raw,D_y_cal,M_y_raw,M_y_cal"]
        d_g_raw = new_pred["D"][f"raw_soup_{part}"]
        m_g_raw = new_pred["M"][f"raw_soup_{part}"]
        d_y_raw = prev_pred["D"][f"raw_soup_{part}"]
        m_y_raw = prev_pred["M"][f"raw_soup_{part}"]
        for position, stable in enumerate(part_idx.tolist()):
            lines.append(
                f"train:{int(stable):04d},{int(kk[position])},{target_y[position]:.10f},{target_g[position]:.10f},{cc[position]:.10f},"
                f"{d_g_raw[position]:.10f},{d_g_raw[position] + b_new['D']:.10f},"
                f"{m_g_raw[position]:.10f},{m_g_raw[position] + b_new['M']:.10f},"
                f"{d_y_raw[position]:.10f},{d_y_raw[position] + b_prev['D']:.10f},"
                f"{m_y_raw[position]:.10f},{m_y_raw[position] + b_prev['M']:.10f}"
            )
        (out_dir / f"per_graph_{part}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(out_dir / "analysis.json", report)
    write_json(out_dir / "gate.json", report["gate"])
    write_json(out_dir / "gains.json", report["gains"])
    write_json(out_dir / "bootstrap.json", {
        "gains": report["gains"],
        "interaction": interaction,
        "self_tests": report["bootstrap_self_tests"],
    })
    write_json(out_dir / "interaction.json", interaction)
    write_json(out_dir / "sensitivity.json", report["sensitivity"])
    write_json(out_dir / "paired_g0_gains.json", {
        "endpoint": "new-dev k=0",
        "c0": interaction["c0"],
        "cal": four_cal,
        "raw": four_raw,
    })
    log(f"[analysis] category={category} relief={relief} G_g_cal={gg_cal['point']:.6f} CI={gg_cal['ci95']}")
    return report


# ---------------------------------------------------------------------------
# mechanism / replay / budget / figures
# ---------------------------------------------------------------------------


def _bridge_output_stats(model: nn.Module, interfaces: list[torch.Tensor]) -> dict[str, Any]:
    out = []
    with torch.no_grad():
        for interface in interfaces:
            module = model.local_dictionary_bridge
            if isinstance(module, MLPBridge):
                scale = torch.sqrt(interface.pow(2).mean(dim=1, keepdim=True) + float(lb.BRIDGE_EPS))
                hidden = F.silu(module.fc1(interface / scale))
                out.append(scale * module.fc2(hidden))
            else:
                value, _aux = module(interface, return_aux=True)
                out.append(value)
    stacked = torch.cat(out, dim=0)
    std = stacked.std(dim=0)
    return {
        "n_rows": int(stacked.shape[0]),
        "output_rms": float(stacked.pow(2).mean().sqrt()),
        "output_abs_mean": float(stacked.abs().mean()),
        "dead_dims_std_lt_1e-8": int((std < 1e-8).sum()),
        "dim": int(stacked.shape[1]),
    }


def collect_mechanism(*, out_dir: Path, device: torch.device, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    new_meta = {arm: json.loads((out_dir / f"{arm}_meta.json").read_text()) for arm in ARMS}
    report: dict[str, Any] = {"arms": {}}
    fixed_idx = list(range(128))
    fixed_target = torch.as_tensor(g_fit[:128], dtype=torch.float32)

    for arm in ARMS:
        entry: dict[str, Any] = {
            "curve_summary": {
                "train_mae_epoch1": new_meta[arm]["curve"][0]["train_task_mae"],
                "train_mae_epoch120": new_meta[arm]["curve"][119]["train_task_mae"],
                "train_mae_epoch240": new_meta[arm]["curve"][-1]["train_task_mae"],
                "grad_norm_epoch1": new_meta[arm]["curve"][0]["grad_norm"],
                "clip_fraction_mean": float(np.mean([e["clip_fraction"] for e in new_meta[arm]["curve"]])),
                "seconds_total": float(sum(e["seconds"] for e in new_meta[arm]["curve"])),
            },
            "bridge_grad_log": new_meta[arm]["bridge_grad_log"],
            "health": new_meta[arm]["health"],
            "param_change": new_meta[arm]["health"]["param_change"],
            "wall_clock_s": new_meta[arm]["wall_clock_s"],
            "replay_max_abs_diff": new_meta[arm]["replay_max_abs_diff"],
        }
        # perturb / fixed-batch probes at init and soup, with isolated RNG.
        for state_name in ("init", "raw_soup"):
            state = torch.load(out_dir / f"{arm}_{state_name}_state.pt", map_location="cpu", weights_only=False)
            seed_everything(SEED)
            model = build_arm(arm).to(device)
            model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
            model.eval()
            batch = zftd.make_batch(fit_data, fixed_idx, fixed_target, device)
            with torch.no_grad():
                coord = model.code(batch.dict_phi)
                interface = model.fusion(sem.SEM108Model.semantic_interface(model, coord, batch, cm.C6_MASK, None))
                base = model(batch, mask=cm.C6_MASK)
                module = model.local_dictionary_bridge
                if isinstance(module, MLPBridge):
                    scale = torch.sqrt(interface.pow(2).mean(dim=1, keepdim=True) + float(lb.BRIDGE_EPS))
                    hidden = F.silu(module.fc1(interface / scale))
                    bridge_out = scale * module.fc2(hidden)
                    extra = {"hidden_rms": float(hidden.pow(2).mean().sqrt()),
                             "hidden_dead_dims": int((hidden.std(dim=0) < 1e-8).sum())}
                else:
                    bridge_out, aux = module(interface, return_aux=True)
                    extra = {"code_nonzero_fraction": float((aux["alpha"] != 0).float().mean()),
                             "code_per_row_nonzero": float((aux["alpha"] != 0).float().sum(dim=1).mean()),
                             "normalized_input_rms": float(aux["normalized"].pow(2).mean().sqrt())}
                saved = {k: v.detach().clone() for k, v in module.state_dict().items()}
                for value in module.parameters():
                    value.add_(1e-3)
                with torch.no_grad():
                    perturbed = model(batch, mask=cm.C6_MASK)
                module.load_state_dict(saved)
            entry[f"{state_name}_probe"] = {
                "interface_rms": float(interface.pow(2).mean().sqrt()),
                "bridge_output_rms": float(bridge_out.pow(2).mean().sqrt()),
                "prediction_shift_for_1e-3_bridge_perturbation": float((perturbed - base).abs().max()),
                "rng_isolated": True,
                **extra,
            }
        # full fit/dev representation activity from the soup state.
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False)
        model = build_arm(arm).to(device)
        model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
        model.eval()
        for part, data_list in (("fit", fit_data), ("dev", dev_data)):
            interfaces: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, len(data_list), 256):
                    batch = zftd.make_batch(
                        data_list, list(range(start, min(start + 256, len(data_list)))),
                        torch.zeros(len(data_list)), device,
                    )
                    coord = model.code(batch.dict_phi)
                    interfaces.append(model.fusion(sem.SEM108Model.semantic_interface(model, coord, batch, cm.C6_MASK, None)).detach())
            entry[f"{part}_representation"] = _bridge_output_stats(model, interfaces)
        report["arms"][arm] = entry

    # relative training cost: measure forward+backward seconds over 8 fixed batches.
    batch = zftd.make_batch(fit_data, fixed_idx, fixed_target, device)
    for arm in ARMS:
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False)
        model = build_arm(arm).to(device)
        model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
        model.train()
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(8):
            prediction = model(batch, mask=cm.C6_MASK)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            model.zero_grad()
            loss.backward()
        torch.cuda.synchronize() if device.type == "cuda" else None
        report["arms"][arm]["seconds_8_train_steps"] = float(time.perf_counter() - t0)
    if "D" in report["arms"] and "M" in report["arms"]:
        report["cost_ratio_D_over_M_train_steps"] = (
            report["arms"]["D"]["seconds_8_train_steps"] / max(report["arms"]["M"]["seconds_8_train_steps"], 1e-12)
        )
        report["note"] = "parameter counts are matched; FLOP/time are not."
    report["official_valid_loaded"] = False
    report["official_test_loaded"] = False
    write_json(out_dir / "mechanism_health.json", report)
    return report


def collect_replay(*, out_dir: Path, device: torch.device, log: Any = print) -> dict[str, Any]:
    fold, prep_meta, fit_data, dev_data, tgt = load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    new_meta = {arm: json.loads((out_dir / f"{arm}_meta.json").read_text()) for arm in ARMS}
    checks: dict[str, Any] = {"arms": {}, "contract_errors": []}
    for arm in ARMS:
        with np.load(out_dir / f"{arm}_raw_predictions.npz") as z:
            pred = {key: np.asarray(z[key], np.float64) for key in z.files}
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False)
        model = build_arm(arm).to(device)
        model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
        replay = evaluate_state(model, fit_data[:128], g_fit[:128], device)
        max_diff = float(np.max(np.abs(replay - pred["raw_soup_fit"][:128])))
        entry = {
            "reloaded_soup_replay_max_abs_diff": max_diff,
            "train_replay_max_abs_diff": float(new_meta[arm]["replay_max_abs_diff"]),
            "fit_prediction_shape": list(pred["raw_soup_fit"].shape),
            "dev_prediction_shape": list(pred["raw_soup_dev"].shape),
            "fit_pred_1d": bool(pred["raw_soup_fit"].ndim == 1 and pred["raw_soup_fit"].shape[0] == 8000),
            "dev_pred_1d": bool(pred["raw_soup_dev"].ndim == 1 and pred["raw_soup_dev"].shape[0] == 2000),
            "steps_done": int(new_meta[arm]["steps_done"]),
            "steps_expected": int(new_meta[arm]["steps_expected"]),
            "stopped_reason": new_meta[arm]["stopped_reason"],
            "backbone_ok": max_diff <= REPLAY_TOL,
        }
        if not entry["backbone_ok"]:
            checks["contract_errors"].append(f"{arm}: soup replay maxdiff {max_diff}")
        checks["arms"][arm] = entry
    checks["schedule_sha256"] = {arm: new_meta[arm]["schedule_sha256"] for arm in ARMS}
    checks["data_stream_sha256"] = {arm: new_meta[arm]["data_stream_sha256"] for arm in ARMS}
    checks["schedule_match_between_arms"] = bool(
        new_meta["D"]["schedule_sha256"] == new_meta["M"]["schedule_sha256"]
    )
    checks["data_stream_match_between_arms"] = bool(
        new_meta["D"]["data_stream_sha256"] == new_meta["M"]["data_stream_sha256"]
    )
    checks["fold_match_expected"] = bool(
        fold["fit_idx_sha256"] == "7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9"
        and fold["dev_idx_sha256"] == "a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff"
    )
    checks["all_ok"] = bool(
        not checks["contract_errors"] and checks["schedule_match_between_arms"]
        and checks["data_stream_match_between_arms"] and checks["fold_match_expected"]
        and all(entry["backbone_ok"] for entry in checks["arms"].values())
    )
    write_json(out_dir / "replay_checks.json", checks)
    return checks


def make_figures(*, out_dir: Path, log: Any = print) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((out_dir / "analysis.json").read_text())
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    arms = ["D_y", "M_y", "D_g", "M_g"]
    values = [
        report["arms_y_reference"]["dev"]["D"]["cal"]["k=0"]["mae"],
        report["arms_y_reference"]["dev"]["M"]["cal"]["k=0"]["mae"],
        report["arms"]["dev"]["D"]["cal"]["k=0"]["mae"],
        report["arms"]["dev"]["M"]["cal"]["k=0"]["mae"],
    ]
    colors = ["#888888", "#888888", "#1f77b4", "#d62728"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(arms, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.5f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("G0 dev calibrated MAE")
    ax.set_title("G0 (k=0): y-unit MAE, old y arms vs new g arms")
    ax.text(0.5, 0.02, "old arms: y target; new arms: g target; G0 y=g+c0", transform=ax.transAxes,
            ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "g0_four_arms.png", dpi=160)
    plt.close(fig)

    parts = ["fit", "dev"]
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(parts))
    for offset, (arm, color) in enumerate(((("D", "#1f77b4"), ("M", "#d62728")))):
        vals = [report["arms"][part][arm]["cal"]["mae"] for part in parts]
        g0vals = [report["arms"][part][arm]["cal"]["k=0"]["mae"] for part in parts]
        bars = ax.bar(x + (offset - 0.5) * width, vals, width=width * 0.9, color=color, label=f"{arm}_g overall")
        ax.bar(x + (offset - 0.5) * width, g0vals, width=width * 0.45, color=color, alpha=0.45, label=f"{arm}_g G0")
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["fit (8000)", "dev (2000)"])
    ax.set_ylabel("calibrated MAE vs g")
    ax.set_title("New g arms: fit/dev calibrated MAE (overall and G0)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "fit_dev_gap.png", dpi=160)
    plt.close(fig)
    log(f"[figures] wrote {figures}")


def make_budget(*, out_dir: Path, round_start: str, round_end: str) -> dict[str, Any]:
    new_meta = {}
    for arm in ARMS:
        path = out_dir / f"{arm}_meta.json"
        new_meta[arm] = json.loads(path.read_text()) if path.exists() else None
    gpu_seconds = sum(float(new_meta[arm]["wall_clock_s"]) for arm in ARMS if new_meta[arm])
    budget = {
        "round_start_cst": round_start,
        "round_end_cst": round_end,
        "formal_arms": 2,
        "seed": SEED,
        "epochs": EPOCHS,
        "max_parallel_gpus": 2,
        "gpu_hours_used_reported": gpu_seconds / 3600.0,
        "gpu_hours_limit": 1.0,
        "per_arm": {arm: (float(new_meta[arm]["wall_clock_s"]) if new_meta[arm] else None) for arm in ARMS},
        "steps_per_arm": {arm: (int(new_meta[arm]["steps_done"]) if new_meta[arm] else None) for arm in ARMS},
        "cpu_threads_cap": 8,
        "smoke_note": "smoke + replay + analysis are included in the GPU-hour ledger where they used a GPU",
        "note": "budget is an upper bound; the round stops computing before the wall-clock limit",
    }
    write_json(out_dir / "budget.json", budget)
    return budget


def make_manifest(*, out_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(out_dir))
        if relative in ("manifest.json",):
            continue
        files[relative] = {"sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
    sources = {
        "runner": Path(__file__),
        "prev_runner": Path(zw.__file__),
        "protocol_md": out_dir / "PROTOCOL.md",
        "method_contract_md": out_dir / "METHOD_CONTRACT.md",
        "evidence_scope_md": out_dir / "EVIDENCE_SCOPE.md",
        "report_md": out_dir / "REPORT.md",
        "decision_md": out_dir / "DECISION.md",
        "execution_md": out_dir / "EXECUTION.md",
    }
    source_files = {
        name: {"sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
        for name, path in sources.items() if path.exists()
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "result_files": files,
        "source_files": source_files,
        "missing_source_files": [name for name, path in sources.items() if not path.exists()],
    }
    write_json(out_dir / "manifest.json", payload)
    return payload


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZINC chemistry task: dictionary vs MLP on g")
    parser.add_argument("--phase-a", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--round-start", default="2026-10-04 11:38:05 CST")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    device = zw.resolve_device(args.device)
    if args.phase_a:
        print(json.dumps(jsonable(phase_a()), indent=2)[:2000])
        return 0
    if args.smoke:
        run_smoke(device=device, out_dir=out_dir)
        return 0
    if args.arm:
        train_arm_g(args.arm, device=device, out_dir=out_dir, epochs=int(args.epochs), max_steps=args.max_steps)
        return 0
    if args.analyze:
        analyze(out_dir=out_dir, device=device)
        return 0
    if args.mechanism:
        collect_mechanism(out_dir=out_dir, device=device)
        return 0
    if args.replay:
        collect_replay(out_dir=out_dir, device=device)
        return 0
    if args.figures:
        make_figures(out_dir=out_dir)
        return 0
    if args.manifest:
        make_manifest(out_dir=out_dir)
        return 0
    if args.budget:
        import datetime

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
        make_budget(out_dir=out_dir, round_start=args.round_start, round_end=now)
        return 0
    parser.error("choose --phase-a, --smoke, --arm, --analyze, --mechanism, --replay, --figures, --manifest or --budget")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
