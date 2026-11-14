"""Directed corrections for the 8 h round: ERRATA evidence, no new science.

Everything in this module is *additive*: the old checkpoints, predictions,
curves, ``train.json`` records and derived tables are read only.  Corrected
derived quantities are written to new files (``*_corrected.json``,
``support_v2_*.npz``, ``corrected_manifest.json``) so that the original record
stays byte-identical and every difference is auditable.

Five corrections (see ``ERRATA.md``):

E1  ``tail_view`` selected the *smallest* errors while callers used them as the
    drop set -> ``tail_corrected.json`` (plus the fixed function in
    ``lcpr_analysis``).
E2  ``build_support`` counted *roots*, not *molecules* -> ``support_v2_*``
    (``lcpr_data.build_support_v2``), legacy S runs relabelled
    ``root-support legacy``.
E3  ``_grad_diagnostics`` normed all parameters (private heads included) and did
    not weight the aux term by the 0.1 in the actual loss ->
    ``grad_checkpoint_diagnostic.json`` (shared-trunk, weighted; explicitly a
    checkpoint diagnostic, not the training-time gradient).
E4  ``Adam(lr=peak_lr)`` plus a ``LambdaLR`` returning ``desired/init`` tripled
    every lr -> ``dmpnn_schedule_audit.json`` (toy trace) and the two
    ``*_FIX`` runs.
E5  ``run_table`` was written before the ``_curve_best_epoch`` fix and read
    ``provenance.commit`` instead of ``provenance.git_commit`` ->
    ``corrected_manifest.json`` (per-field provenance; the raw records are
    correct, the derived table was stale).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import lcpr_common as C

#: the ten run identities the correction prompt names explicitly
IDENTITY_RUNS = (
    "C1_R0", "C1_R1", "C1_R3", "C1_CHEM8", "C1_CONTRIB", "K_COMBO",
    "A0_R0", "A0_CHEM8_CONTRIB_R0", "DMPNN_LR3E4", "DMPNN_LR1E3",
)
#: runs added by ERRATA E4 (only present once the corrected trainings finished)
IDENTITY_RUNS_FIX = ("DMPNN_LR3E4_FIX", "DMPNN_LR1E3_FIX")

#: tail comparisons: (control, arm) on Lipophilicity
TAIL_PAIRS = (
    ("C1_R1", "K_COMBO"),
    ("C1_R1", "C1_CHEM8"),
    ("C1_R1", "C1_CONTRIB"),
    ("C1_R1", "C1_R3"),
    ("K_COMBO", "C1_R3"),
    ("K_COMBO", "C1_CHEM8"),
)

#: shared trunk of the C1-style model: the modules every arm trains jointly.
#: ``f``/``rho`` are the main-task private head and ``q_aux``/``rho_aux`` the
#: aux private head, so none of them is "shared".
SHARED_PREFIXES = ("atom_encoder.", "bond_encoder.", "local.")
PRIVATE_HEADS = ("f.", "rho.", "q_aux.", "rho_aux.", "q.", "delta.", "W_msg.", "W_self.", "W_out.")

CURVE_CHECKPOINTS = (30, 60, 90, 120, 180, 240, 360)
CURVE_WINDOWS = ((60, 120), (120, 240), (240, 360))


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _rmse(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def _mae(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _curve_rows(run_dir: Path) -> list[dict[str, str]]:
    import csv as _csv

    path = Path(run_dir) / "curve.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(_csv.DictReader(fh))


def _curve_best_epoch(rows: Sequence[Mapping[str, str]], task: str) -> int | None:
    """Best Valid epoch from the curve; RMSE minimised, AUC maximised."""
    key = "valid_rmse" if task == "regression" else "valid_roc_auc"
    sign = 1.0 if task == "regression" else -1.0
    best, best_epoch = math.inf, None
    for row in rows:
        try:
            score = sign * float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if score < best:
            best, best_epoch = score, int(float(row["epoch"]))
    return best_epoch


def _predictions(run_dir: Path, name: str = "predictions_best.npz") -> dict[str, np.ndarray]:
    z = np.load(Path(run_dir) / name, allow_pickle=False)
    return {k: np.asarray(z[k]) for k in z.files}


def _decoded(run_dir: Path, dataset: str, name: str = "predictions_best.npz") -> dict[str, Any]:
    """Predictions decoded to published units exactly once (Train-only norm)."""
    import lcpr_data as D

    raw = _predictions(run_dir, name)
    norm = D.targets(dataset, include_test=False)["target_norm"]
    out: dict[str, Any] = {"norm": norm, "raw_keys": sorted(raw)}
    for split in C.SPLITS:
        out[f"{split}_y"] = np.asarray(raw[f"{split}_y"], float)
        out[f"{split}_pred"] = (np.asarray(raw[f"{split}_pred_std"], float) * float(norm["scale"])
                               + float(norm["mean"]))
        out[f"{split}_rows"] = (np.asarray(raw[f"{split}_rows"]).astype(np.int64)
                                if f"{split}_rows" in raw else None)
    return out


# --------------------------------------------------------------------------- #
# E1 + E5 a: corrected tails
# --------------------------------------------------------------------------- #


def tail_comparison(control: Mapping[str, np.ndarray], arm: Mapping[str, np.ndarray],
                    split: str, fractions: Sequence[float] = C.TAIL_FRACTIONS) -> dict[str, Any]:
    """One (control, arm) tail comparison on one split.

    The row set is defined by the **control's** absolute error: the ``k`` rows
    with the largest error (ties by ascending global row id).  Every model in the
    comparison is scored on exactly that kept set; the main result is always the
    full sample.
    """
    import lcpr_analysis as A

    y = np.asarray(control[f"{split}_y"], float)
    p0 = np.asarray(control[f"{split}_pred"], float)
    p1 = np.asarray(arm[f"{split}_pred"], float)
    rows = control.get(f"{split}_rows")
    sets = A.tail_sets(y, p0, rows, fractions)
    e0, e1 = p0 - y, p1 - y
    ae0, ae1 = np.abs(e0), np.abs(e1)
    d_ae = ae1 - ae0
    rmse0, rmse1 = _rmse(p0, y), _rmse(p1, y)
    rec: dict[str, Any] = {
        "split": split,
        "n": int(y.size),
        "full": {
            "control_rmse": rmse0, "arm_rmse": rmse1,
            "control_mae": _mae(p0, y), "arm_mae": _mae(p1, y),
            "gain": float((rmse0 - rmse1) / rmse0),
            "mae_gain": float((_mae(p0, y) - _mae(p1, y)) / _mae(p0, y)),
            "median_delta_abs_error": float(np.median(d_ae)),
            "mean_delta_abs_error": float(np.mean(d_ae)),
            "n_rows_improved": int(np.sum(d_ae < 0)),
            "n_rows_worse": int(np.sum(d_ae > 0)),
            "n_rows_tied": int(np.sum(d_ae == 0)),
            "delta_sse_sum": float(np.sum(e1 ** 2 - e0 ** 2)),
            "control_sse": float(np.sum(e0 ** 2)),
        },
        "tail_rule": sets["order"],
        "top_row": {"position": sets["top_row_position"],
                    "global_row_id": sets["top_global_row_id"],
                    "abs_error": sets["top_row_abs_error"]},
        "by_fraction": {},
    }
    for frac in fractions:
        block = sets[f"exclude_top_{frac:g}"]
        keep = np.setdiff1d(np.arange(y.size), np.asarray(block["positions"], np.int64))
        b_rmse = _rmse(p0[keep], y[keep])
        a_rmse = _rmse(p1[keep], y[keep])
        rec["by_fraction"][f"exclude_top_{frac:g}"] = {
            "k": block["k"],
            "n_kept": int(keep.size),
            "control_rmse_same_rows": b_rmse,
            "arm_rmse_same_rows": a_rmse,
            "gain": float((b_rmse - a_rmse) / b_rmse),
            "gain_full": rec["full"]["gain"],
            "control_sse_share_removed": block["sse_share_of_base"],
            "control_rmse_full": rmse0,
            "control_rmse_change_from_drop": float(b_rmse - rmse0),
            "drop_row_global_ids": block["global_row_ids"],
            "drop_row_positions": block["positions"],
            "drop_rows_n_improved": int(np.sum(d_ae[np.asarray(block["positions"])] < 0)),
        }
        #: dropping the largest residuals can never inflate the control's RMSE
        if b_rmse > rmse0 + 1e-12:
            raise AssertionError(
                f"control RMSE increased after dropping its largest errors ({b_rmse} > {rmse0})")
    #: gain by control-error stratum: does the improvement sit in one stratum?
    q = np.argsort(np.argsort(ae0)) / max(y.size - 1, 1)
    strata: dict[str, Any] = {}
    for i in range(5):
        sel = (q >= i / 5) & (q < (i + 1) / 5)
        if not sel.any():
            continue
        c_rmse = _rmse(p0[sel], y[sel])
        a_rmse = _rmse(p1[sel], y[sel])
        strata[f"quintile_{i + 1}"] = {
            "n": int(sel.sum()),
            "control_rmse": c_rmse,
            "arm_rmse": a_rmse,
            #: a stratum of perfect predictions has no relative gain
            "gain": (float((c_rmse - a_rmse) / c_rmse) if c_rmse > 1e-12 else None),
            "median_delta_abs_error": float(np.median(d_ae[sel])),
            "frac_improved": float(np.mean(d_ae[sel] < 0)),
            "delta_sse_sum": float(np.sum(e1[sel] ** 2 - e0[sel] ** 2)),
        }
    rec["gain_by_control_error_quintile"] = strata
    #: per-row description of the largest wins / losses (global ids, not positions)
    order = A.tail_rank(y, p0, rows)
    d_ae_ordered = d_ae[order]
    rec["cumulative_delta_sse"] = np.cumsum((e1 ** 2 - e0 ** 2)[order]).tolist()
    order_ = np.argsort(d_ae_ordered, kind="stable")
    worst = list(order_[-5:][::-1])
    best = list(order_[:5])
    rec["rows"] = {
        "note": "positions are the index inside the split array; global_row_id is the published row",
        "largest_improvements": [
            {"position": int(order[i]),
             "global_row_id": int(rows[order[i]]) if rows is not None else int(order[i]),
             "abs_error_control": float(ae0[order[i]]), "abs_error_arm": float(ae1[order[i]]),
             "delta_abs_error": float(d_ae[order[i]])} for i in best],
        "largest_regressions": [
            {"position": int(order[i]),
             "global_row_id": int(rows[order[i]]) if rows is not None else int(order[i]),
             "abs_error_control": float(ae0[order[i]]), "abs_error_arm": float(ae1[order[i]]),
             "delta_abs_error": float(d_ae[order[i]])} for i in worst],
    }
    return rec


def run_tail() -> dict[str, Any]:
    """E1: recompute every affected tail statement with the corrected drop set."""
    out: dict[str, Any] = {
        "created_utc": C.utcnow(),
        "errata": "E1",
        "rule": ("drop set = the k rows with the largest *control* absolute error, "
                 "ties broken by ascending published global row id; "
                 "k = max(1, round(frac*n)); every model scored on the same kept rows; "
                 "headline numbers always use the full sample"),
        "fractions": list(C.TAIL_FRACTIONS),
        "pairs": {},
        "legacy_artifact": {
            "path": "analysis/tail_summary.json",
            "status": "SUPERSEDED (kept unmodified)",
            "defect": ("np.argsort(ae)[:k] returns the k *smallest* errors while the keys are "
                       "named exclude_top_*; the reported 4.94 % 'after dropping the largest 5 %' "
                       "is therefore a computation on the best-fitting rows"),
        },
    }
    for control, arm in TAIL_PAIRS:
        d0 = C.run_dir(control, "Lipophilicity")
        d1 = C.run_dir(arm, "Lipophilicity")
        if not (d0 / "predictions_best.npz").exists() or not (d1 / "predictions_best.npz").exists():
            out["pairs"][f"{arm}_vs_{control}"] = {"status": "missing_predictions"}
            continue
        c = _decoded(d0, "Lipophilicity")
        a = _decoded(d1, "Lipophilicity")
        key = f"{arm}_vs_{control}"
        out["pairs"][key] = {s: tail_comparison(c, a, s) for s in C.SPLITS}
    C.write_json(C.ANALYSIS / "tail_corrected.json", out)
    return out


# --------------------------------------------------------------------------- #
# E5: prediction / checkpoint identity manifest
# --------------------------------------------------------------------------- #


def _run_identity(candidate: str, dataset: str = "Lipophilicity") -> dict[str, Any]:
    import lcpr_analysis as A
    import lcpr_data as D
    import torch

    run_dir = C.run_dir(candidate, dataset)
    tj_path = run_dir / "train.json"
    rec: dict[str, Any] = {"candidate": candidate, "dataset": dataset,
                           "run_dir": C.track_rel(run_dir), "fields": {}, "verdict": {}}
    if not tj_path.exists():
        rec["verdict"] = {"identity": "MISSING_RUN"}
        return rec
    payload = json.loads(tj_path.read_text())
    task = payload["task"]
    recipe = payload.get("recipe") or {}
    prov = payload.get("provenance") or {}
    curve = _curve_rows(run_dir)
    rec["fields"]["train_json"] = {
        "status": payload.get("status"),
        "best_epoch_key_present": "best_epoch" in payload,
        "best_epoch": payload.get("best_epoch"),
        "epochs_run_key_present": "epochs_run" in payload,
        "epochs_run": payload.get("epochs_run"),
        "recipe_epochs": recipe.get("epochs"),
        "recipe_epochs_run": recipe.get("epochs_run"),
        "steps_per_epoch": recipe.get("steps_per_epoch"),
        "optimizer_steps": recipe.get("optimizer_steps"),
        "seed": payload.get("seed", C.TRAIN_SEED),
        "parameters": (payload.get("model") or {}).get("parameters"),
        "source": "runs/<run>/train.json (written by the training process)",
    }
    rec["fields"]["curve"] = {
        "exists": bool(curve),
        "rows": len(curve),
        "first_epoch": int(float(curve[0]["epoch"])) if curve else None,
        "last_epoch": int(float(curve[-1]["epoch"])) if curve else None,
        "valid_argbest_epoch_recomputed": _curve_best_epoch(curve, task) if curve else None,
        "columns": sorted(curve[0]) if curve else [],
        "source": "runs/<run>/curve.csv (per-epoch, written by the training process)",
    }
    rec["fields"]["provenance"] = {
        "git_commit": prov.get("git_commit"),
        "git_dirty_tracked": prov.get("git_dirty_tracked"),
        "provenance_commit_key_present": "commit" in prov,
        "note": ("run_table.csv printed provenance['commit'], which does not exist (the key is "
                 "'git_commit'), hence commit=None in the stale table"),
        "code_hash_lcpr_train": (prov.get("code_hashes") or {}).get("code/lcpr_train.py"),
        "code_hash_current_lcpr_train": C.sha256_file(C.CODE_DIR / "lcpr_train.py"),
        "code_hash_lcpr_dmpnn": (prov.get("code_hashes") or {}).get("code/lcpr_dmpnn.py"),
        "code_hash_current_lcpr_dmpnn": C.sha256_file(C.CODE_DIR / "lcpr_dmpnn.py"),
    }
    # ---- predictions vs the published split and vs the recorded metrics -----
    pred_path = run_dir / "predictions_best.npz"
    tgt = D.targets(dataset, include_test=True)
    if pred_path.exists():
        dec = _decoded(run_dir, dataset, "predictions_best.npz")
        checks: dict[str, Any] = {"keys": dec["raw_keys"]}
        ok_rows = ok_y = True
        for split in C.SPLITS:
            pub_rows = np.asarray(tgt["rows"][split], np.int64)
            pub_y = np.asarray(tgt["y"][split], float)
            rows = dec[f"{split}_rows"]
            if rows is None:
                ok_rows = False
                checks[f"{split}_rows"] = "missing from npz"
                continue
            checks[f"{split}_rows_match_published"] = bool(rows.size == pub_rows.size
                                                          and np.array_equal(rows, pub_rows))
            ok_rows &= bool(checks[f"{split}_rows_match_published"])
            checks[f"{split}_rows_sha256"] = _sha(rows)
            checks[f"{split}_y_matches_published_labels"] = bool(
                np.allclose(dec[f"{split}_y"], pub_y, rtol=0, atol=1e-9))
            ok_y &= bool(checks[f"{split}_y_matches_published_labels"])
            checks[f"{split}_y_sha256"] = _sha(dec[f"{split}_y"])
        recomputed = {}
        for split in C.SPLITS:
            p, y = dec[f"{split}_pred"], dec[f"{split}_y"]
            recomputed[split] = {"rmse": _rmse(p, y), "mae": _mae(p, y), "n": int(y.size)}
        recorded = (payload.get("metrics") or {}).get("best") or {}
        deltas = {s: abs(recomputed[s]["rmse"] - float(recorded[s]["rmse"]))
                  for s in C.SPLITS if s in recorded}
        checks["recomputed_from_predictions"] = recomputed
        checks["recorded_train_json"] = {s: recorded.get(s, {}).get("rmse") for s in C.SPLITS}
        checks["abs_diff_vs_train_json"] = deltas
        checks["matches_train_json"] = bool(all(v <= 1e-6 for v in deltas.values()))
        checks["normalisation"] = dec["norm"]
        checks["units"] = D.dataset_meta(dataset)["units"]
        #: the same numbers must appear on the best-Valid row of the curve
        best_ep = _curve_best_epoch(curve, task) if curve else None
        if curve and best_ep is not None:
            row = next((r for r in curve if int(float(r["epoch"])) == best_ep), None)
            if row is not None:
                checks["curve_best_epoch"] = best_ep
                checks["curve_value_at_best_epoch"] = {
                    s: float(row.get(f"{s}_rmse", "nan")) for s in C.SPLITS}
                checks["abs_diff_vs_curve_best_epoch"] = {
                    s: abs(recomputed[s]["rmse"] - float(row.get(f"{s}_rmse", "nan")))
                    for s in C.SPLITS}
                checks["matches_curve_best_epoch"] = bool(
                    all(v <= 1e-6 for v in checks["abs_diff_vs_curve_best_epoch"].values()))
        last_path = run_dir / "predictions_last.npz"
        if last_path.exists() and curve:
            dl = _decoded(run_dir, dataset, "predictions_last.npz")
            checks["last_epoch_predictions"] = {
                s: {"rmse": _rmse(dl[f"{s}_pred"], dl[f"{s}_y"]),
                    "curve_last_rmse": float(curve[-1].get(f"{s}_rmse", "nan")),
                    "abs_diff": abs(_rmse(dl[f"{s}_pred"], dl[f"{s}_y"])
                                    - float(curve[-1].get(f"{s}_rmse", "nan")))}
                for s in C.SPLITS}
            checks["last_epoch_matches_curve"] = bool(all(
                v["abs_diff"] <= 1e-6 for v in checks["last_epoch_predictions"].values()))
        checks["rows_ok"] = ok_rows
        checks["labels_ok"] = ok_y
        rec["fields"]["predictions"] = checks
        rec["verdict"]["prediction_identity"] = (
            "VERIFIED" if (ok_rows and ok_y and checks["matches_train_json"]
                           and checks.get("matches_curve_best_epoch", False))
            else "UNRESOLVED_IDENTITY")
    else:
        rec["fields"]["predictions"] = {"exists": False}
        rec["verdict"]["prediction_identity"] = "MISSING_PREDICTIONS"

    # ---- best.pt: retained? reload reproduces the saved predictions? -------
    ckpt = run_dir / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        info = {
            "exists": True,
            "bytes": int(ckpt.stat().st_size),
            "sha256": C.sha256_file(ckpt),
            "best_epoch_in_checkpoint": int(state.get("best_epoch", -1)),
            "best_metric_value": float(state.get("best_metric_value", float("nan"))),
        }
        if C.CANDIDATES.get(candidate, {}).get("kind") == "dmpnn":
            #: D-MPNN replay would need the directed edge / feature cache, so instead
            #: check what the checkpoint *itself* claims against the curve and
            #: train.json. This is explicitly weaker than a forward replay.
            sd = state.get("state_dict") or {}
            n_params = int(sum(int(v.numel()) for v in sd.values())) if sd else -1
            ck_best = int(state.get("best_epoch", -1))
            ck_val = float(state.get("best_metric_value", float("nan")))
            curve_argbest = None
            curve_val_at_best = None
            if curve:
                row = min(curve, key=lambda r: float(r.get("valid_rmse", "inf")))
                curve_argbest = int(row["epoch"])
                curve_val_at_best = float(row["valid_rmse"])
            tj_params = (rec.get("fields", {}).get("train_json") or {})
            info["metadata_audit"] = {
                "checkpoint_best_epoch": ck_best,
                "curve_argbest_epoch": curve_argbest,
                "checkpoint_best_metric_value": ck_val,
                "curve_valid_rmse_at_argbest": curve_val_at_best,
                "best_epoch_matches_curve": bool(ck_best == curve_argbest),
                "best_metric_matches_curve": bool(
                    curve_val_at_best is not None and abs(ck_val - curve_val_at_best) <= 1e-6),
                "state_dict_tensors": len(sd),
                "state_dict_parameters": n_params,
                "declared_parameters": state.get("parameters"),
                "train_json_parameters": tj_params.get("parameters"),
            }
            ok = bool(info["metadata_audit"]["best_epoch_matches_curve"]
                      and info["metadata_audit"]["best_metric_matches_curve"])
            info["replay"] = "not_implemented_for_dmpnn_here"
            rec["verdict"]["weight_identity"] = "VERIFIED_METADATA" if ok else "UNRESOLVED_IDENTITY"
        else:
            model = A._load_trained(C.CANDIDATES[candidate]["kind"], candidate, dataset)
            if model is not None:
                import lcpr_train as T

                caches = {s: D.load_cache(dataset, s, tgt["rows"][s]) for s in C.SPLITS}
                import lcpr_model as LM

                kind = C.CANDIDATES[candidate]["kind"]
                path_counts = None
                if kind in LM.SHRINK_KINDS:
                    #: the S arms contract the joint block with the frozen support
                    #: table, which lives in cache/support_*.npz, not in best.pt
                    sup = D.load_support(dataset)
                    path_counts = {s: sup["path_count"][s] for s in C.SPLITS}
                preds = T.evaluate_all(model, caches, torch.device("cpu"), path_counts)
                #: ``evaluate_all`` returns z-scored predictions; decode once with the
                #: Train-only normaliser before any comparison with raw labels
                tnorm = tgt["target_norm"]
                rep = {s: {"rmse": _rmse(preds[s] * float(tnorm["scale"]) + float(tnorm["mean"]),
                                         np.asarray(tgt["y"][s], float))} for s in C.SPLITS}
                if pred_path.exists():
                    dec = _decoded(run_dir, dataset, "predictions_best.npz")
                    rep_max = max(float(np.max(np.abs(
                        preds[s] * float(tnorm["scale"]) + float(tnorm["mean"])
                        - dec[f"{s}_pred"]))) for s in C.SPLITS)
                else:
                    rep_max = float("nan")
                info["replay"] = {
                    "device": "cpu",
                    "metrics": rep,
                    "max_abs_pred_diff_vs_predictions_best": rep_max,
                    "consistent": bool(rep_max <= 1e-5),
                }
                rec["verdict"]["weight_identity"] = (
                    "VERIFIED" if rep_max <= 1e-5 else "UNRESOLVED_IDENTITY")
        rec["fields"]["checkpoint"] = info
    else:
        rec["fields"]["checkpoint"] = {
            "exists": False,
            "note": ("runs/ is git-ignored and only three checkpoints were retained on this "
                     "host (A_LOGP, C1_R1, S_SELECT_K1); everything else was deleted after the "
                     "round, so weight identity cannot be re-verified - only prediction "
                     "identity (curve + train.json + npz) can"),
        }
        rec["verdict"]["weight_identity"] = "NOT_RETAINED"
    return rec


def run_identity() -> dict[str, Any]:
    """E5 (+ step 2 of the prompt): per-field identity for the named runs."""
    runs = list(IDENTITY_RUNS) + [c for c in IDENTITY_RUNS_FIX
                                  if C.run_dir(c, "Lipophilicity").exists()]
    entries = {c: _run_identity(c) for c in runs}
    #: every Lipophilicity arm, so the manifest also covers the mechanism panels
    extra = {}
    for path in sorted(C.RUNS.glob("*__Lipophilicity/train.json")):
        cand = path.parent.name.split("__")[0]
        if cand in entries:
            continue
        extra[cand] = _run_identity(cand)
    out = {
        "created_utc": C.utcnow(),
        "errata": "E5",
        "scope": ("the ten identities named by the correction request plus every other "
                  "Lipophilicity arm (mechanism panels depend on their weights)"),
        "checked_against": {
            "published_rows": "lcrc_molnet_benchmark_pilot_v2 published split (via lcpr_data.targets)",
            "published_labels": "same",
            "normalisation": "Train-split mean/std, decoded exactly once",
            "metrics": "train.json metrics.best, curve.csv row at the best-Valid epoch",
        },
        "runs": entries,
        "other_lipophilicity_arms": extra,
        "stale_derived_table": {
            "artifact": "analysis/run_table.csv / .json",
            "created_utc": json.loads((C.ANALYSIS / "run_table.json").read_text()).get("created_utc")
            if (C.ANALYSIS / "run_table.json").exists() else None,
            "superseded_by": "analysis/run_table_corrected.csv / .json",
            "defects": [
                "written before commit d6e8d1f4 fixed _curve_best_epoch (it maximised valid_rmse, "
                "so best_epoch=1 for most runs even though train.json metrics are at the true "
                "best-Valid epoch)",
                "commit column read provenance['commit']; the record stores 'git_commit'",
                "epochs column is recipe['epochs'], not epochs_run (LONG 720, ogbg-molhiv 160)",
            ],
        },
    }
    C.write_json(C.ANALYSIS / "corrected_manifest.json", out)
    return out


def write_corrected_run_table() -> dict[str, Any]:
    """A corrected machine table built *from the raw records*, never from REPORT.md."""
    import csv as _csv

    rows = []
    for path in sorted(C.RUNS.glob("*/train.json")) + sorted((C.RUNS / "smoke").glob("*/train.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "completed" or payload.get("smoke"):
            continue
        run_dir = path.parent
        recipe = payload.get("recipe") or {}
        prov = payload.get("provenance") or {}
        curve = _curve_rows(run_dir)
        best_epoch = payload.get("best_epoch")
        if best_epoch is None:
            best_epoch = _curve_best_epoch(curve, payload["task"])
        rec = {
            "run": run_dir.name,
            "candidate": payload.get("candidate"),
            "dataset": payload.get("dataset"),
            "task": payload.get("task"),
            "recipe": recipe.get("name") or recipe.get("recipe") or (
                C.CANDIDATES.get(payload.get("candidate"), {}).get("recipe")),
            "optimizer_base_lr": recipe.get("optimizer_base_lr"),
            "peak_lr_declared": recipe.get("peak_lr"),
            "weight_decay": recipe.get("weight_decay"),
            "parameters": (payload.get("model") or {}).get("parameters"),
            "epochs_run": len(curve) if curve else recipe.get("epochs_run"),
            "epochs_declared": recipe.get("epochs"),
            "steps_per_epoch": recipe.get("steps_per_epoch"),
            "optimizer_updates": (int(recipe["steps_per_epoch"]) * len(curve))
            if (recipe.get("steps_per_epoch") and curve) else recipe.get("optimizer_steps"),
            "best_epoch": best_epoch,
            "best_epoch_source": ("train.json" if payload.get("best_epoch") is not None
                                  else "curve.csv argmin(valid_rmse)/argmax(valid_roc_auc)"),
            "git_commit": prov.get("git_commit"),
            "git_dirty_tracked": prov.get("git_dirty_tracked"),
            "seed": payload.get("seed", C.TRAIN_SEED),
        "parameters": (payload.get("model") or {}).get("parameters"),
            "checkpoint_retained": (run_dir / "best.pt").exists(),
            "predictions_retained": (run_dir / "predictions_best.npz").exists(),
        }
        for split in C.SPLITS:
            metrics = ((payload.get("metrics") or {}).get("best") or {}).get(split) or {}
            for key in ("rmse", "mae", "r2", "roc_auc", "bce", "n"):
                rec[f"{split}_{key}"] = metrics.get(key)
        rows.append(rec)
    C.write_json(C.ANALYSIS / "run_table_corrected.json",
                 {"created_utc": C.utcnow(), "source": "runs/*/train.json + curve.csv (raw records)",
                  "note": "replaces the stale analysis/run_table.{json,csv}; old files untouched",
                  "runs": rows})
    cols = ["run", "candidate", "dataset", "task", "recipe", "optimizer_base_lr", "peak_lr_declared",
            "weight_decay", "parameters", "epochs_run", "epochs_declared", "steps_per_epoch",
            "optimizer_updates", "best_epoch", "best_epoch_source", "git_commit",
            "git_dirty_tracked", "seed", "checkpoint_retained", "predictions_retained"]
    metric_cols = [f"{s}_{k}" for s in C.SPLITS for k in ("rmse", "mae", "r2", "roc_auc", "bce", "n")
                   if any(r.get(f"{s}_{k}") is not None for r in rows)]
    lines = [",".join(cols + metric_cols)]
    for r in rows:
        vals = ["" if r.get(c) is None else str(r.get(c)) for c in cols]
        for mc in metric_cols:
            v = r.get(mc)
            vals.append("" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v)))
        lines.append(",".join(vals))
    (C.ANALYSIS / "run_table_corrected.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"runs": len(rows)}


# --------------------------------------------------------------------------- #
# E2: molecule-level support (definition fix, no retraining)
# --------------------------------------------------------------------------- #


def run_support(datasets: Sequence[str] = ("Lipophilicity", "ESOL", "BACE")) -> dict[str, Any]:
    import lcpr_data as D

    out: dict[str, Any] = {
        "created_utc": C.utcnow(),
        "errata": "E2",
        "status": ("the four S arms were trained with the legacy ROOT support; they are "
                   "relabelled root-support legacy experiments and are NOT a test of the "
                   "registered molecule-support hypothesis. No S arm is retrained here."),
        "definition": "c(key) = # distinct Train molecules containing key >= once",
        "datasets": {},
    }
    for ds in datasets:
        if not (D.support_path(ds).exists() and (C.MOLNET / "cache").exists()
                and (C.UPSTREAM_CACHE / f"{ds}_train.pt").exists()):
            out["datasets"][ds] = {"status": "cache_missing"}
            continue
        v2 = D.build_support_v2(ds)
        z_old = np.load(D.support_path(ds), allow_pickle=False)
        z_new = np.load(D.support_v2_path(ds), allow_pickle=False)
        old_k = np.asarray(z_old["keys"]).reshape(-1, 2)
        old_c = np.asarray(z_old["counts"]).astype(np.float64)
        new_k = np.asarray(z_new["keys"]).reshape(-1, 2)
        new_c = np.asarray(z_new["counts"]).astype(np.float64)
        n_codes = int(C.N_PATH_CODES) * int(C.N_NBR_CODES)
        old_flat = old_k[:, 0] * n_codes + old_k[:, 1]
        new_flat = new_k[:, 0] * n_codes + new_k[:, 1]
        pos = np.clip(np.searchsorted(old_flat, new_flat), 0, max(len(old_c) - 1, 0))
        ok = old_flat[pos] == new_flat
        ratio = np.full(new_c.size, np.nan)
        ratio[ok] = new_c[ok] / np.maximum(old_c[pos[ok]], 1.0)
        alpha = {}
        for split in C.SPLITS:
            c = np.asarray(z_new[f"path_count_{split}"], dtype=np.float64)
            a = c / (c + float(z_new["kappa"]))
            alpha[split] = {
                "n_path_tokens": int(c.size),
                "mean_alpha": float(np.mean(a)) if c.size else None,
                "median_alpha": float(np.median(a)) if c.size else None,
                "p05_alpha": float(np.percentile(a, 5)) if c.size else None,
                "p95_alpha": float(np.percentile(a, 95)) if c.size else None,
                "frac_count_le_1": float(np.mean(c <= 1)) if c.size else None,
                "frac_count_zero_unknown_key": float(np.mean(c == 0)) if c.size else None,
            }
        out["datasets"][ds] = {
            "v2": v2,
            "comparison_with_legacy_root_support": {
                "n_keys_legacy": int(old_c.size),
                "n_keys_v2": int(new_c.size),
                "keys_only_in_v2": int(new_c.size - int(ok.sum())),
                "legacy_count_max": float(old_c.max()) if old_c.size else None,
                "v2_count_max": float(new_c.max()) if new_c.size else None,
                "legacy_count_exceeds_n_train_molecules":
                    bool(old_c.size and float(old_c.max()) > v2["n_train_molecules"]),
                "kappa_legacy": float(z_old["kappa"]),
                "kappa_v2": float(z_new["kappa"]),
                "kappa_ratio": float(z_new["kappa"]) / float(z_old["kappa"]),
                "a_bar_k1_legacy": float(z_old["a_bar_k1"]),
                "a_bar_k1_v2": float(z_new["a_bar_k1"]),
                "a_bar_k4_legacy": float(z_old["a_bar_k4"]),
                "a_bar_k4_v2": float(z_new["a_bar_k4"]),
                "per_key_ratio_c_mol_over_c_root": {
                    "median": float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else None,
                    "p05": float(np.nanpercentile(ratio, 5)) if np.isfinite(ratio).any() else None,
                    "p95": float(np.nanpercentile(ratio, 95)) if np.isfinite(ratio).any() else None,
                    "frac_equal": float(np.nanmean(ratio == 1.0)) if np.isfinite(ratio).any() else None,
                },
            },
            "alpha_by_split_v2": alpha,
            "legacy_cache_untouched": C.track_rel(D.support_path(ds)),
            "v2_cache": C.track_rel(D.support_v2_path(ds)),
        }
    C.write_json(C.ANALYSIS / "support_v2_summary.json", out)
    return out


# --------------------------------------------------------------------------- #
# E3: shared-trunk, loss-weighted gradient diagnostic on retained checkpoints
# --------------------------------------------------------------------------- #


def _split_params(model) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    shared, private = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (shared if any(name.startswith(pre) for pre in SHARED_PREFIXES) else private).append((name, p))
    return shared, private


def _gnorm(grads: Sequence[Any]) -> float:
    total = 0.0
    for g in grads:
        if g is not None:
            total += float((g.detach().double() ** 2).sum())
    return math.sqrt(total)


def _dot(a: Sequence[Any], b: Sequence[Any]) -> float:
    return sum(float((x.detach().double() * y.detach().double()).sum())
               for x, y in zip(a, b) if x is not None and y is not None)


def run_grad() -> dict[str, Any]:
    """E3: recompute *what can be recomputed* and withdraw the rest.

    Reported as ``CHECKPOINT_DIAGNOSTIC``: these are gradients of the retained
    best-Validation weights in eval mode on fixed Train minibatches, **not** the
    gradients the original training saw at steps 1/100/1000.
    """
    import torch
    import torch.nn.functional as F

    import lcpr_analysis as A
    import lcpr_data as D
    import lcpr_train as T

    aux_weight = 0.1
    n_batches, batch_size = 4, 128
    out: dict[str, Any] = {
        "created_utc": C.utcnow(),
        "errata": "E3",
        "label": "CHECKPOINT_DIAGNOSTIC",
        "scope": {
            "shared_trunk": list(SHARED_PREFIXES),
            "excluded_private_heads": list(PRIVATE_HEADS),
            "sub_weight": aux_weight,
        },
        "defect": ("the recorded _grad_diagnostics normed *all* trainable parameters "
                   "(private main f/rho and aux q_aux/rho_aux heads included) and used the "
                   "unweighted aux loss, so 'aux/main = 9.64 / 20.4' is neither the shared-trunk "
                   "ratio nor the ratio that entered the update"),
        "recoverable_from_saved_scalars": {},
        "arms": {},
    }
    #: what the saved global scalars become under the correct 0.1 weighting alone
    for cand in ("A_LOGP", "A_MW", "A_SHUFFLE"):
        diag_path = C.run_dir(cand, "Lipophilicity") / "diagnostics.json"
        if not diag_path.exists():
            continue
        steps = (json.loads(diag_path.read_text()).get("grad_steps") or [])
        out["recoverable_from_saved_scalars"][cand] = {
            "note": ("all-parameter scope, both losses as recorded; the only thing that can "
                     "be fixed without the weights is the missing 0.1 weight"),
            "steps": [{"step": s["step"], "epoch": s["epoch"],
                       "aux_over_main_ratio_as_recorded": s["aux_over_main_ratio"],
                       "aux_over_main_ratio_weighted_all_params":
                           s.get("aux_over_main_ratio") * aux_weight if s.get("aux_over_main_ratio") else None,
                       "cosine_as_recorded": s.get("grad_cosine"),
                       "grad_main_l2": s.get("grad_main_l2"),
                       "grad_aux_l2": s.get("grad_aux_l2")} for s in steps],
            "checkpoint_retained": (C.run_dir(cand, "Lipophilicity") / "best.pt").exists(),
        }

    dataset = "Lipophilicity"
    spec = D.frozen_spec()
    cache = D.load_cache(dataset, "train")
    tgt = D.targets(dataset, include_test=True)
    tnorm = tgt["target_norm"]
    aux = D.load_aux_targets(dataset)
    n_train = int(cache["n_molecules"])
    train_global_rows = np.asarray(tgt["rows"]["train"], dtype=np.int64)
    #: fixed Train rows by hash order (no RNG, no labels), cut into 4 x 128 minibatches
    h = A._row_hash([str(s) for s in cache["smiles"]])
    order = np.argsort(h, kind="stable")[: n_batches * batch_size]
    batches = [order[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]

    for cand, aux_key in (("A_LOGP", "wc_logp"), ("A_MW", "molwt"), ("A_SHUFFLE", "wc_logp_shuffled")):
        ckpt = C.run_dir(cand, dataset) / "best.pt"
        entry: dict[str, Any] = {"checkpoint": C.track_rel(ckpt), "checkpoint_exists": ckpt.exists()}
        if not ckpt.exists():
            entry["status"] = "NOT_AVAILABLE_NO_CHECKPOINT"
            entry["consequence"] = ("the shared-trunk ratio at the retained weights cannot be "
                                    "recomputed for this arm; only the 0.1 weighting of the "
                                    "historical all-parameter scalars is recoverable")
            out["arms"][cand] = entry
            continue
        model = A._load_trained(C.CANDIDATES[cand]["kind"], cand, dataset)
        if model is None:
            entry["status"] = "CHECKPOINT_RELOAD_FAILED"
            out["arms"][cand] = entry
            continue
        model.eval()
        shared, private = _split_params(model)
        norm = aux[f"{'wc_logp' if aux_key == 'wc_logp_shuffled' else aux_key}_norm"]
        raw_aux = np.asarray(aux[aux_key], dtype=np.float64)[np.asarray(tgt["rows"]["train"], np.int64)]
        per_batch = []
        for bi, rows in enumerate(batches):
            rows = np.asarray(rows, np.int64)
            batch = D.to_device(D.collate(cache, list(rows)), torch.device("cpu"))
            outb = model(batch, return_aux=True)
            z = torch.as_tensor((tgt["y"]["train"][rows] - float(tnorm["mean"])) / float(tnorm["scale"]),
                                dtype=torch.float32)
            loss_main = F.mse_loss(outb["y"], z)
            za = torch.as_tensor((raw_aux[rows] - float(norm["mean"])) / float(norm["scale"]),
                                 dtype=torch.float32)
            loss_aux = F.mse_loss(outb["aux_pred"], za)
            loss_total = loss_main + aux_weight * loss_aux
            gm_s = torch.autograd.grad(loss_main, [p for _, p in shared], retain_graph=True,
                                       allow_unused=True)
            ga_s = torch.autograd.grad(loss_aux, [p for _, p in shared], retain_graph=True,
                                       allow_unused=True)
            gm_p = torch.autograd.grad(loss_main, [p for _, p in private], retain_graph=True,
                                       allow_unused=True)
            ga_p = torch.autograd.grad(loss_aux, [p for _, p in private], retain_graph=True,
                                       allow_unused=True)
            g_tot = torch.autograd.grad(loss_total, [p for _, p in shared + private],
                                        retain_graph=False, allow_unused=True)
            gm_s_n, ga_s_n = _gnorm(gm_s), _gnorm(ga_s)
            gm_all = _gnorm(list(gm_s) + list(gm_p))
            ga_all = _gnorm(list(ga_s) + list(ga_p))
            tot_n = _gnorm(g_tot)
            wa_s = [None if g is None else g * aux_weight for g in ga_s]
            wa_s_n = _gnorm(wa_s)
            cos_s = _dot(gm_s, wa_s) / (gm_s_n * wa_s_n) if gm_s_n > 0 and wa_s_n > 0 else None
            per_batch.append({
                "batch": bi,
                "n_rows": int(rows.size),
                "global_row_ids_head": [int(x) for x in train_global_rows[rows[:3]]],
                "loss_main": float(loss_main.detach()),
                "loss_aux": float(loss_aux.detach()),
                "shared_trunk": {
                    "grad_main_l2": gm_s_n,
                    "grad_aux_l2_unweighted": ga_s_n,
                    "grad_aux_l2_weighted": wa_s_n,
                    "aux_over_main_ratio_weighted": (wa_s_n / gm_s_n) if gm_s_n > 0 else None,
                    "cosine_main_vs_weighted_aux": cos_s,
                },
                "all_parameters": {
                    "grad_main_l2": gm_all,
                    "grad_aux_l2_unweighted": ga_all,
                    "aux_over_main_ratio_weighted": (aux_weight * ga_all / gm_all) if gm_all > 0 else None,
                    "grad_total_l2_of_weighted_loss": tot_n,
                    "clip_norm": 5.0,
                    "clip_coefficient_norm_over_clip": tot_n / 5.0,
                    "would_be_clipped": bool(tot_n > 5.0),
                },
            })
        def _mean(path: Sequence[str]) -> float:
            vals = []
            for b in per_batch:
                v = b
                for k in path:
                    v = v[k]
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
            return float(np.mean(vals)) if vals else float("nan")
        entry.update({
            "status": "COMPUTED",
            "mode": "eval (dropout off), no optimiser step, weights = best-Validation checkpoint",
            "n_shared_params": len(shared),
            "n_private_params": len(private),
            "batches": per_batch,
            "mean": {
                "shared_aux_over_main_ratio_weighted": _mean(("shared_trunk", "aux_over_main_ratio_weighted")),
                "shared_cosine": _mean(("shared_trunk", "cosine_main_vs_weighted_aux")),
                "all_param_aux_over_main_ratio_weighted":
                    _mean(("all_parameters", "aux_over_main_ratio_weighted")),
                "all_param_clip_coefficient": _mean(("all_parameters", "clip_coefficient_norm_over_clip")),
            },
            "withdraw": ("the historical claim that the optimisation burden is a *proven* cause of "
                         "the A-family failure is downgraded to an untested explanation: at the "
                         "retained weights the aux term is a small, weakly aligned contribution"),
        })
        out["arms"][cand] = entry
    out["arms_not_available"] = [c for c, e in out["arms"].items() if e.get("checkpoint_exists") is False]
    C.write_json(C.ANALYSIS / "grad_checkpoint_diagnostic.json", out)
    return out


# --------------------------------------------------------------------------- #
# E4: D-MPNN schedule audit (toy trace, no data)
# --------------------------------------------------------------------------- #


def toy_schedule_trace(base_lr: float, peak_lr: float, init_lr: float, min_lr: float,
                       steps_per_epoch: int, warmup_epochs: int, epochs: int) -> dict[str, Any]:
    """Run the real ``LambdaLR`` over a one-parameter toy ``Adam`` for every step."""
    import torch

    import lcpr_dmpnn as DMN

    total = int(steps_per_epoch) * int(epochs)
    warmup = int(round(int(warmup_epochs) * int(steps_per_epoch)))
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=float(base_lr), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: DMN._noam_lambda(s, warmup, total - warmup,
                                                  float(init_lr), float(peak_lr), float(min_lr)))
    lrs: list[float] = []
    lrs_after: list[float] = []
    for step in range(total):
        lrs.append(float(opt.param_groups[0]["lr"]))     # lr used by THIS update
        opt.zero_grad(set_to_none=True)
        (param * 1.0).sum().backward()                   # toy gradient, no data
        opt.step()
        sched.step()                                     # after optimizer.step (same order as train)
        lrs_after.append(float(opt.param_groups[0]["lr"]))
    after = float(opt.param_groups[0]["lr"])
    return {
        "optimizer_base_lr": float(base_lr),
        "scheduler_initial_lr": float(opt.param_groups[0]["initial_lr"]),
        "steps_per_epoch": int(steps_per_epoch),
        "warmup_steps": warmup,
        "total_steps": total,
        "lr_at_first_update": lrs[0],
        "lr_at_step_2": lrs[1] if len(lrs) > 1 else None,
        "lr_at_warmup_end_update": lrs[warmup - 1],
        "lr_after_warmup_end": lrs_after[warmup - 1],
        "lr_at_last_update": lrs[-1],
        "lr_after_scheduler_done": after,
        "lr_max_over_run": max(lrs),
        "lr_min_over_run": min(lrs),
        #: ``curve.csv:lr_end`` is read *after* ``scheduler.step()``, so epoch e
        #: records the lr that update e*steps_per_epoch + 1 will actually use
        "lr_end_of_epoch_1_as_curve_records_it": lrs_after[steps_per_epoch - 1],
        "lr_end_of_epoch_2_as_curve_records_it": lrs_after[2 * steps_per_epoch - 1],
        "endpoints_as_traced": {
            #: the three protocol endpoints: the lr of the first update, the
            #: maximum realised lr, and the lr the scheduler ends on (the
            #: curve's ``lr_end`` of the final epoch records exactly this)
            "init": lrs[0],
            "peak": max(lrs),
            "final": after,
        },
    }


def run_schedule() -> dict[str, Any]:
    """E4: was the *actual run* on the wrong lr, or only the current code?"""
    out: dict[str, Any] = {
        "created_utc": C.utcnow(),
        "errata": "E4",
        "defect": ("Adam(base lr = peak_lr) combined with LambdaLR(lambda = desired/init_lr) makes "
                   "the realised lr peak_lr * desired/init_lr, i.e. 3x the protocol curve for D1 "
                   "and 10x for D2"),
        "protocol_expectation": {"D1": {"init": 1e-4, "peak": 3e-4, "final": 1e-4},
                                 "D2": {"init": 1e-4, "peak": 1e-3, "final": 1e-4}},
        "toy_traces": {},
        "observed_runs": {},
        "code_revision_check": {},
    }
    for tag, peak in (("D1", 3.0e-4), ("D2", 1.0e-3)):
        out["toy_traces"][f"{tag}_legacy_base_peak"] = toy_schedule_trace(
            base_lr=peak, peak_lr=peak, init_lr=1e-4, min_lr=1e-4,
            steps_per_epoch=53, warmup_epochs=2, epochs=100)
        out["toy_traces"][f"{tag}_fixed_base_init"] = toy_schedule_trace(
            base_lr=1e-4, peak_lr=peak, init_lr=1e-4, min_lr=1e-4,
            steps_per_epoch=53, warmup_epochs=2, epochs=100)
    for cand, recipe in (("DMPNN_LR3E4", "D1"), ("DMPNN_LR1E3", "D2")):
        run_dir = C.run_dir(cand, "Lipophilicity")
        payload_path = run_dir / "train.json"
        if not payload_path.exists():
            out["observed_runs"][cand] = {"status": "missing"}
            continue
        payload = json.loads(payload_path.read_text())
        curve = _curve_rows(run_dir)
        lrs = [float(r["lr_end"]) for r in curve] if curve else []
        prov = payload.get("provenance") or {}
        hashes = prov.get("code_hashes") or {}
        observed = {
            "recipe_declared": {k: (payload.get("recipe") or {}).get(k)
                                for k in ("peak_lr", "init_lr", "min_lr", "warmup_epochs", "epochs",
                                          "steps_per_epoch")},
            "curve_lr_end_epoch1": lrs[0] if lrs else None,
            "curve_lr_end_epoch2": lrs[1] if len(lrs) > 1 else None,
            "curve_lr_end_last": lrs[-1] if lrs else None,
            "curve_lr_max": max(lrs) if lrs else None,
            "curve_rows": len(curve),
            "git_commit": prov.get("git_commit"),
            "git_dirty_tracked": prov.get("git_dirty_tracked"),
            "lcpr_dmpnn_sha256_at_run": hashes.get("code/lcpr_dmpnn.py"),
            "lcpr_dmpnn_sha256_current": C.sha256_file(C.CODE_DIR / "lcpr_dmpnn.py"),
        }
        #: the executed code must be the code that contains the defect
        observed["run_used_defective_code"] = bool(
            observed["lcpr_dmpnn_sha256_at_run"]
            and observed["lcpr_dmpnn_sha256_at_run"] == C.DMPNN_LEGACY_SHA256)
        toy_legacy = out["toy_traces"][f"{recipe}_legacy_base_peak"]
        toy_fixed = out["toy_traces"][f"{recipe}_fixed_base_init"]
        observed["consistent_with_legacy_trace"] = bool(
            lrs and abs(lrs[0] - toy_legacy["lr_end_of_epoch_1_as_curve_records_it"]) <= 1e-9
            and abs(lrs[-1] - toy_legacy["lr_after_scheduler_done"]) <= 1e-9
            and abs(lrs[1] - toy_legacy["lr_end_of_epoch_2_as_curve_records_it"]) <= 1e-9)
        observed["consistent_with_fixed_trace"] = bool(
            lrs and abs(lrs[0] - toy_fixed["lr_end_of_epoch_1_as_curve_records_it"]) <= 1e-9)
        observed["toy_legacy_epoch1_endpoint"] = toy_legacy["lr_end_of_epoch_1_as_curve_records_it"]
        observed["toy_fixed_epoch1_endpoint"] = toy_fixed["lr_end_of_epoch_1_as_curve_records_it"]
        observed["verdict"] = ("DMPNN_LEGACY_LR_MISMATCH" if observed["consistent_with_legacy_trace"]
                               else "SCHEDULE_MATCHES_PROTOCOL")
        out["observed_runs"][cand] = observed
    out["code_revision_check"] = {
        "lcpr_dmpnn_sha256_that_contained_the_defect": C.DMPNN_LEGACY_SHA256,
        "lcpr_dmpnn_sha256_after_fix": C.sha256_file(C.CODE_DIR / "lcpr_dmpnn.py"),
        "note": ("both historical runs recorded the defect-bearing hash, so the wrong lr was the "
                 "executed lr; the forward-parity check does not test the optimiser schedule and "
                 "could never have caught it"),
    }
    for cand in IDENTITY_RUNS_FIX:
        run_dir = C.run_dir(cand, "Lipophilicity")
        if not run_dir.exists():
            out["observed_runs"][cand] = {"status": "not_run"}
            continue
        payload = json.loads((run_dir / "train.json").read_text())
        curve = _curve_rows(run_dir)
        lrs = [float(r["lr_end"]) for r in curve] if curve else []
        out["observed_runs"][cand] = {
            "status": "completed",
            "recipe_declared": {k: (payload.get("recipe") or {}).get(k)
                                for k in ("peak_lr", "init_lr", "min_lr", "warmup_epochs", "epochs",
                                          "steps_per_epoch", "optimizer_base_lr")},
            "schedule_endpoints": (payload.get("recipe") or {}).get("schedule_endpoints"),
            "curve_lr_end_epoch1": lrs[0] if lrs else None,
            "curve_lr_end_epoch2": lrs[1] if len(lrs) > 1 else None,
            "curve_lr_end_last": lrs[-1] if lrs else None,
            "curve_lr_max": max(lrs) if lrs else None,
            "git_commit": (payload.get("provenance") or {}).get("git_commit"),
            "best_epoch": payload.get("best_epoch"),
            "metrics_best": (payload.get("metrics") or {}).get("best"),
        }
    C.write_json(C.ANALYSIS / "dmpnn_schedule_audit.json", out)
    return out


# --------------------------------------------------------------------------- #
# 7: learning curves from the existing runs (no new training)
# --------------------------------------------------------------------------- #


CURVE_RUNS = ("C1_R1", "C1_R3", "C1_CHEM8", "C1_CONTRIB", "K_COMBO")


def run_curves() -> dict[str, Any]:
    out: dict[str, Any] = {
        "created_utc": C.utcnow(),
        "scope": ("existing complete epoch-level curves only; no new checkpoint, no weight "
                  "averaging, no Test-based epoch selection"),
        "epochs": list(CURVE_CHECKPOINTS),
        "windows": [list(w) for w in CURVE_WINDOWS],
        "models": {},
        "selection_positions": {},
    }
    for cand in CURVE_RUNS:
        curve = _curve_rows(C.run_dir(cand, "Lipophilicity"))
        if not curve:
            out["models"][cand] = {"status": "curve_missing"}
            continue
        by_epoch = {int(float(r["epoch"])): r for r in curve}
        samples = {}
        for ep in CURVE_CHECKPOINTS:
            row = by_epoch.get(int(ep))
            samples[str(ep)] = None if row is None else {
                k: float(row[f"{k}_rmse"]) for k in C.SPLITS if f"{k}_rmse" in row}
        windows = {}
        for lo, hi in CURVE_WINDOWS:
            sel = [by_epoch[e] for e in range(int(lo), int(hi) + 1) if e in by_epoch]
            windows[f"{lo}-{hi}"] = None if not sel else {
                f"{s}_median": float(np.median([float(r[f"{s}_rmse"]) for r in sel]))
                for s in C.SPLITS if f"{s}_rmse" in sel[0]}
        best_valid = _curve_best_epoch(curve, "regression")
        best_test_ep = min((int(float(r["epoch"])) for r in curve), key=lambda e: float(by_epoch[e]["test_rmse"]))
        out["models"][cand] = {
            "curve_rows": len(curve),
            "samples": samples,
            "window_medians": windows,
            "best_valid_epoch": best_valid,
            "best_valid_valid_rmse": float(by_epoch[best_valid]["valid_rmse"]),
            "best_valid_test_rmse": float(by_epoch[best_valid]["test_rmse"]),
            "best_valid_train_rmse": float(by_epoch[best_valid]["train_rmse"]),
            "best_test_epoch_not_used_for_selection": best_test_ep,
            "best_test_test_rmse": float(by_epoch[best_test_ep]["test_rmse"]),
            "last_epoch": {"train": float(curve[-1]["train_rmse"]),
                           "valid": float(curve[-1]["valid_rmse"]),
                           "test": float(curve[-1]["test_rmse"])},
        }
    base = out["models"].get("C1_R1", {})
    positions = {}
    base_curve = {int(float(r["epoch"])): r for r in _curve_rows(C.run_dir("C1_R1", "Lipophilicity"))}
    for cand, rec in out["models"].items():
        if rec.get("status") or not base.get("best_valid_epoch"):
            continue
        curve = {int(float(r["epoch"])): r for r in _curve_rows(C.run_dir(cand, "Lipophilicity"))}
        at_base = curve.get(int(base["best_valid_epoch"]))
        positions[cand] = {
            "best_valid_epoch": rec["best_valid_epoch"],
            "base_best_valid_epoch": base["best_valid_epoch"],
            "delta_epoch": rec["best_valid_epoch"] - int(base["best_valid_epoch"]),
            "test_rmse_at_its_own_best_valid": rec["best_valid_test_rmse"],
            "test_rmse_at_base_best_valid_epoch": (
                None if at_base is None else float(at_base["test_rmse"])),
        }
    out["selection_positions"] = positions
    C.write_json(C.ANALYSIS / "learning_curves.json", out)
    return out


# --------------------------------------------------------------------------- #
# paired intervals for the head-to-head comparisons the report needed
# --------------------------------------------------------------------------- #


PAIR_HEADS = (("C1_CHEM8", "K_COMBO"), ("C1_R3", "K_COMBO"), ("C1_R1", "K_COMBO"),
              ("C1_R1", "DMPNN_LR1E3_FIX"), ("C1_R1", "DMPNN_LR3E4_FIX"),
              ("C1_R1", "DMPNN_LR3E4"), ("C1_R1", "DMPNN_LR1E3"))


def run_pairs() -> dict[str, Any]:
    """Paired scaffold-cluster bootstrap for the *same-recipe* K increments.

    ``K_COMBO`` vs ``C1_CHEM8`` is the only architecture-only increment (both run
    the R1 recipe); the vs-``C1_R3`` and vs-``C1_R1`` pairs change the recipe too
    and are therefore performance coordinates, not pure architecture effects.
    """
    import lcpr_analysis as A

    out: dict[str, Any] = {
        "created_utc": C.utcnow(),
        "method": ("paired scaffold-cluster bootstrap, 2000 replicates, seed 20261113 "
                   "(identical to analysis/paired_comparisons.json); delta = RMSE(control) - "
                   "RMSE(arm), positive = the arm is better"),
        "dataset": "Lipophilicity",
        "pairs": {},
        "full_metrics": {},
    }
    cands = sorted({c for pair in PAIR_HEADS for c in pair})
    for cand in cands:
        if not (C.run_dir(cand, "Lipophilicity") / "predictions_best.npz").exists():
            continue
        d = _decoded(C.run_dir(cand, "Lipophilicity"), "Lipophilicity")
        out["full_metrics"][cand] = {
            s: {"rmse": _rmse(d[f"{s}_pred"], d[f"{s}_y"]),
                "mae": _mae(d[f"{s}_pred"], d[f"{s}_y"]), "n": int(d[f"{s}_y"].size)}
            for s in C.SPLITS}
    for control, arm in PAIR_HEADS:
        entry: dict[str, Any] = {}
        cdir = C.run_dir(control, "Lipophilicity")
        adir = C.run_dir(arm, "Lipophilicity")
        if not ((cdir / "predictions_best.npz").exists() and (adir / "predictions_best.npz").exists()):
            out["pairs"][f"{arm}_vs_{control}"] = {"status": "missing_predictions"}
            continue
        c = _decoded(cdir, "Lipophilicity")
        a = _decoded(adir, "Lipophilicity")
        for split in ("valid", "test"):
            y, index = A._cluster_index("Lipophilicity", split, "regression")
            entry[split] = A.paired_bootstrap(
                control=np.asarray(c[f"{split}_pred"], float),
                candidate=np.asarray(a[f"{split}_pred"], float),
                kind="rmse", index=index, y=y, replicates=C.BOOTSTRAP_REPLICATES,
                seed=C.BOOTSTRAP_SEED)
            entry[split]["gain_rel"] = float(
                (entry[split]["delta"]) / _rmse(c[f"{split}_pred"], c[f"{split}_y"]))
        entry["same_recipe"] = (C.CANDIDATES.get(control, {}).get("recipe")
                                == C.CANDIDATES.get(arm, {}).get("recipe"))
        entry["parameters"] = {control: (json.loads((cdir / "train.json").read_text())
                                         .get("model", {}).get("parameters")),
                               arm: (json.loads((adir / "train.json").read_text())
                                     .get("model", {}).get("parameters"))}
        out["pairs"][f"{arm}_vs_{control}"] = entry
    C.write_json(C.ANALYSIS / "paired_corrected.json", out)
    return out


# --------------------------------------------------------------------------- #
# figure: cumulative ΔSSE along the corrected tail order
# --------------------------------------------------------------------------- #


def figure_tail(tail: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = "K_COMBO_vs_C1_R1"
    pair = tail.get("pairs", {}).get(key)
    if not pair:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=170)
    for ax, split in zip(axes, ("valid", "test"), strict=True):
        rec = pair[split]
        n = rec["n"]
        #: per-row ΔSE in the corrected order needs the row arrays, so re-derive from
        #: the recorded quintile/tail aggregates is not enough -> use the saved full
        #: comparison only for the cut markers, and plot the cumulative ΔSSE computed
        #: by ``run_tail`` (stored under rows/order-independent aggregates)
        if "cumulative_delta_sse" not in rec:
            ax.axis("off")
            continue
        cum = np.asarray(rec["cumulative_delta_sse"], float)
        ax.plot(np.arange(1, n + 1), cum, lw=1.4, c="#2b6cb0")
        ax.axhline(0, color="#999", lw=0.8, ls=":")
        for frac, colour in ((0.01, "#dd6b20"), (0.05, "#c53030")):
            k = rec["by_fraction"][f"exclude_top_{frac:g}"]["k"]
            ax.axvline(k, color=colour, lw=0.9, ls="--")
            ax.annotate(f"drop top {frac:g}% (k={k})", (k, cum[-1]), fontsize=6.2, color=colour,
                        xytext=(4, 0), textcoords="offset points")
        ax.set_xlabel("rows, ordered by base |error| descending")
        ax.set_ylabel("cumulative ΔSSE (arm − base)")
        ax.set_title(f"{key} — {split}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run(stages: Sequence[str] = ("identity", "tail", "pairs", "support", "grad", "schedule",
                                 "curves")) -> dict[str, Any]:
    done: dict[str, Any] = {"created_utc": C.utcnow(), "stages": {}}
    if "identity" in stages:
        C.log("[correct] identity manifest")
        done["stages"]["identity"] = {"runs": len(run_identity()["runs"])}
        done["stages"]["run_table"] = write_corrected_run_table()
    if "tail" in stages:
        C.log("[correct] tails (largest-|error| drop sets)")
        tail = run_tail()
        done["stages"]["tail"] = {"pairs": len(tail["pairs"])}
        fig = C.ANALYSIS / "figures" / "tail_corrected_cumulative_dsse.png"
        figure_tail(tail, fig)
    if "pairs" in stages:
        C.log("[correct] head-to-head paired intervals")
        pr = run_pairs()
        done["stages"]["pairs"] = sorted(pr["pairs"])
    if "support" in stages:
        C.log("[correct] molecule-level support v2")
        sup = run_support()
        done["stages"]["support"] = {ds: (v.get("v2") or {}).get("kappa")
                                     for ds, v in sup["datasets"].items()}
    if "grad" in stages:
        C.log("[correct] checkpoint gradient diagnostic")
        g = run_grad()
        done["stages"]["grad"] = {c: e.get("status") for c, e in g["arms"].items()}
    if "schedule" in stages:
        C.log("[correct] D-MPNN schedule audit")
        s = run_schedule()
        done["stages"]["schedule"] = {c: e.get("verdict") or e.get("status")
                                     for c, e in s["observed_runs"].items()}
    if "curves" in stages:
        C.log("[correct] learning curves")
        c = run_curves()
        done["stages"]["curves"] = sorted(c["models"])
    C.write_json(C.ANALYSIS / "correction_run.json", done)
    return done


__all__ = [n for n in dir() if not n.startswith("_")]
