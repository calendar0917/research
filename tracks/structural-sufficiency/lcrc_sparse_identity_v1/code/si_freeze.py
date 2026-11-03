"""Freeze every checkpoint and every protocol artefact BEFORE any Test access.

Writes ``analysis/checkpoint_manifest.json`` (the frozen object), copies the
Validation predictions into ``analysis/predictions/`` and completes gates 13/14
in ``analysis/correctness_gates.json``.

The manifest records, for all 12 neural units and the SPARSE_ONLY model:

* checkpoint path + SHA-256, parameter count, initialisation hash, best epoch,
  best Validation metric, Validation metrics, curve hash;
* the feature/vocabulary hashes the unit was trained with;
* the branch norms and the branch contribution distribution (wide arms);
* GPU hours and peak memory.

It also freezes the code hashes (training + model + reused modules), the
vocabulary/feature matrix hashes per split, the FOLD64 assignment hash and the
SHA-256 of the Test scoring script.  ``si_score.py`` refuses to run if any of
these changed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import si_common as C
import si_features as FEAT
import si_gates as GATES

import ptb_data as PD


def _sha_or_none(path: Path) -> str | None:  # pragma: no cover - tiny helper
    return C.sha256_file(path) if path.exists() else None


def collect_units(spec, feats: Mapping[str, Any], fold: Mapping[str, Any], vocab: Mapping[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm, seed in C.units():
        run_dir = C.run_dir(arm, seed)
        run_json = run_dir / "run.json"
        sel_json = run_dir / "selection.json"
        ckpt = run_dir / "checkpoint.pt"
        if not (run_json.exists() and sel_json.exists() and ckpt.exists()):
            missing.append(C.run_name(arm, seed))
            continue
        record = C.read_json(run_json)
        selection = C.read_json(sel_json)
        sha = C.sha256_file(ckpt)
        if sha != selection["checkpoint_sha256"] or sha != record["checkpoint"]["sha256"]:
            raise RuntimeError(f"{C.run_name(arm, seed)}: checkpoint hash differs from the run record")
        if bool(selection.get("test_evaluated")):
            raise RuntimeError(f"{C.run_name(arm, seed)}: Test was already evaluated before the freeze")
        if int(record["epochs_run"]) != C.EPOCHS or bool(record["smoke"]):
            raise RuntimeError(f"{C.run_name(arm, seed)}: not a formal 100-epoch run")
        if bool(record["data"]["test_labels_read"]) or bool(record["data"]["test_cache_built"]):
            raise RuntimeError(f"{C.run_name(arm, seed)}: run record claims Test access")
        expected_params = C.PC.EXPECTED_PARAMS_SINGLE[C.MODEL_A0] + int(C.BRANCH_DIM[arm])
        if int(record["parameters"]) != expected_params:
            raise RuntimeError(f"{C.run_name(arm, seed)}: parameter count {record['parameters']} != {expected_params}")
        meta = record.get("feature_meta", {})
        if arm != C.MODEL_A0 and str(meta.get("vocabulary_sha256")) != vocab["vocabulary_sha256"]:
            raise RuntimeError(f"{C.run_name(arm, seed)}: trained with a different vocabulary hash")
        if arm == C.MODEL_FOLD and str(meta.get("fold_assignment_sha256")) != fold["assignment_sha256"]:
            raise RuntimeError(f"{C.run_name(arm, seed)}: trained with a different FOLD64 assignment")
        reload_check = selection.get("reload_check", {})
        if not bool(reload_check.get("pass")):
            raise RuntimeError(f"{C.run_name(arm, seed)}: checkpoint reload check failed")
        pred_src = run_dir / "valid_predictions.npz"
        pred_dst = C.PREDICTIONS / f"{arm}_s{int(seed)}_valid.npz"
        pred_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pred_src, pred_dst)
        with np.load(pred_dst) as handle:
            pred_hash = C.array_hash_f32(handle["pred"])
            rows_hash = C.index_sha256(np.asarray(handle["rows"], dtype=np.int64).tolist())
        units.append(
            {
                "unit": C.run_name(arm, seed),
                "arm": arm,
                "seed": int(seed),
                "branch_dim": int(C.BRANCH_DIM[arm]),
                "parameters": int(record["parameters"]),
                "epochs_run": int(record["epochs_run"]),
                "best_epoch": int(record["checkpoint"]["best_epoch"]),
                "best_metric_name": str(selection["best_metric_name"]),
                "best_valid_value": float(selection["best_metric_value"]),
                "valid_metrics": record["validation"]["metrics"],
                "train_metrics": record["train"]["metrics"],
                "gap": record["gap"],
                "branch": record["branch"],
                "checkpoint_path": C.track_rel(ckpt),
                "checkpoint_sha256": sha,
                "best_state_hash": record["checkpoint"]["best_state_hash"],
                "init_state_hash": record["model"]["init_state_hash"],
                "curve_sha256": selection["curve_sha256"],
                "reload_check": reload_check,
                "selection_rule": selection["selection_rule"],
                "valid_predictions_path": C.track_rel(pred_dst),
                "valid_predictions_sha256": pred_hash,
                "valid_rows_sha256": rows_hash,
                "gpu_hours": float(record["timing"]["gpu_hours_estimate"]),
                "gpu_peak_allocated_bytes": int(record["timing"]["gpu_peak_allocated_bytes"]),
                "gpu_peak_reserved_bytes": int(record["timing"]["gpu_peak_reserved_bytes"]),
                "mean_epoch_seconds": float(record["timing"]["mean_epoch_seconds"]),
                "git_commit": record["git_commit"],
                "env": record["env"],
                "feature_meta": meta,
            }
        )
    if missing:
        raise RuntimeError(f"cannot freeze: {len(missing)} unit(s) missing: {missing}")
    if len(units) != len(C.NEURAL) * len(C.SEEDS):
        raise RuntimeError(f"expected 12 units, found {len(units)}")
    return units


def collect_sparse(vocab: Mapping[str, Any], feats: Mapping[str, Any]) -> dict[str, Any]:
    path = C.ANALYSIS / "sparse_only.json"
    if not path.exists():
        raise RuntimeError("SPARSE_ONLY has not been fitted")
    record = C.read_json(path)
    if bool(record.get("test_evaluated")):
        raise RuntimeError("SPARSE_ONLY claims Test access before the freeze")
    if str(record["vocabulary_sha256"]) != vocab["vocabulary_sha256"]:
        raise RuntimeError("SPARSE_ONLY used a different vocabulary hash")
    model_path = C.REPO / record["model_path"]
    with np.load(model_path) as handle:
        coef = np.asarray(handle["coef"], dtype=np.float64)
    probe = C.array_hash_f32(coef.astype(np.float32))
    if probe != record["coef_sha256"]:
        raise RuntimeError("SPARSE_ONLY coefficient hash does not match the fitted record")
    pred_path = C.REPO / record["valid_predictions_path"]
    with np.load(pred_path) as handle:
        pred_hash = C.array_hash_f32(handle["pred"])
    return {
        "arm": C.MODEL_SPARSE,
        "seed": C.SPARSE_RANDOM_STATE,
        "estimator": record["estimator"],
        "hyperparameters": record["hyperparameters"],
        "model_path": record["model_path"],
        "model_sha256": C.sha256_file(model_path),
        "coef_sha256": record["coef_sha256"],
        "n_iter": int(record["n_iter"]),
        "coef_l1": float(record["coef_l1"]),
        "coef_l2": float(record["coef_l2"]),
        "coef_n_nonzero": int(record["coef_n_nonzero"]),
        "intercept": float(record["intercept"]),
        "valid_metrics": record["validation"]["metrics"],
        "train_metrics": record["train"]["metrics"],
        "gap": record["gap"],
        "valid_predictions_path": record["valid_predictions_path"],
        "valid_predictions_sha256": pred_hash,
        "feature_sha256": record["feature_sha256"],
        "vocabulary_sha256": vocab["vocabulary_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze all checkpoints before Test scoring")
    ap.parse_args(argv)
    out_path = C.ANALYSIS / "checkpoint_manifest.json"
    if out_path.exists():
        raise SystemExit(f"{C.track_rel(out_path)} exists; the manifest is frozen exactly once")

    view = PD.molecule_view(C.DATASET)
    spec = view.spec
    if spec.hash() != C.FROZEN_SPEC_HASH:
        raise RuntimeError("FeatureSpec hash changed")
    vocab = C.read_json(C.ANALYSIS / "feature_vocabulary.json")
    if vocab.get("status") != "OK":
        raise RuntimeError("vocabulary is not feasible/frozen")
    feats: dict[str, Any] = {}
    for split, cache in (("train", view.train_cache), ("valid", view.valid_cache)):
        feats[split] = FEAT.encode_split(cache, spec, vocab)
    if feats["train"]["stats"]["vector_sha256"] != C.read_json(
        C.ANALYSIS / "feature_checks.json"
    )["splits"]["train"]["vector_sha256"]:
        raise RuntimeError("Train feature matrix changed since the gate run")
    fold = FEAT.fold_buckets(vocab)
    units = collect_units(spec, feats, fold, vocab)
    sparse = collect_sparse(vocab, feats)

    code = C.code_hashes()
    null_stats = C.read_json(C.ANALYSIS / "null_checks.json")
    reload_max = max(float(u["reload_check"]["max_abs_valid_logit_diff"]) for u in units)
    gate13 = {
        "pass": bool(all(bool(u["reload_check"]["pass"]) for u in units)),
        "details": {
            "units_checked": len(units),
            "max_abs_validation_logit_diff": reload_max,
            "tolerance": 1e-5,
            "per_unit": {u["unit"]: float(u["reload_check"]["max_abs_valid_logit_diff"]) for u in units},
        },
    }
    ledger_path = C.ANALYSIS / "test_access_ledger.json"
    ledger_events = int(len(C.read_json(ledger_path).get("events", []))) if ledger_path.exists() else 0
    test_artifacts = sorted(
        str(p) for p in C.RUNS.rglob("test_*") if p.is_file()
    )
    gate14_details = {
        "test_access_ledger_events_before_freeze": ledger_events,
        "test_artifacts_present_before_freeze": test_artifacts,
        "run_records_claim_test_access": any(
            bool(u.get("feature_meta", {}).get("test_labels_read")) for u in units
        ),
        "static_ast_scan": C.read_json(C.ANALYSIS / "correctness_gates.json")["gate_14_static"]["details"],
    }
    gate14 = {
        "pass": bool(ledger_events == 0 and not test_artifacts),
        "details": gate14_details,
    }
    gates = GATES.finalize_gates({13: gate13, 14: gate14})

    manifest: dict[str, Any] = {
        "schema": "lcrc-sparse-identity-checkpoint-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "dataset": C.DATASET,
        "task": C.TASK,
        "n_units_expected": len(C.NEURAL) * len(C.SEEDS),
        "n_units_present": len(units),
        "frozen": True,
        "test_rows_sha256": C.index_sha256(view.rows["test"]),
        "test_n": int(len(view.rows["test"])),
        "test_rows_loaded_from_split_file_only": True,
        "frozen_before_test_access": True,
        "test_exposure_label": C.RESULT_LABEL,
        "arms": list(C.NEURAL),
        "seeds": list(C.SEEDS),
        "epochs": int(C.EPOCHS),
        "units": units,
        "sparse_only": sparse,
        "code_hashes": code,
        "protocol": {
            "k": int(C.K),
            "n_buckets": int(C.N_BUCKETS),
            "per_bucket": int(C.PER_BUCKET),
            "fold_salt": C.FOLD_SALT,
            "null_seed": dict(C.NULL_SEED),
            "null_swap_multiplier": int(C.NULL_SWAP_MULTIPLIER),
            "bootstrap_seed": int(C.BOOTSTRAP_SEED),
            "bootstrap_replicates": int(C.BOOTSTRAP_REPLICATES),
            "decision_thresholds": {
                "true_minus_a0": C.DECISION_MIN_TRUE_MINUS_A0,
                "true_minus_null": C.DECISION_MIN_TRUE_MINUS_NULL,
                "min_seeds_beating_a0": int(C.DECISION_MIN_SEEDS_BEATING_A0),
            },
            "gpu_soft_cap_hours": float(C.GPU_SOFT_CAP_HOURS),
            "gpu_hard_cap_hours": float(C.GPU_HARD_CAP_HOURS),
        },
        "features": {
            "vocabulary_path": "analysis/feature_vocabulary.json",
            "vocabulary_sha256": vocab["vocabulary_sha256"],
            "feature_spec_hash": spec.hash(),
            "fold_assignment_sha256": fold["assignment_sha256"],
            "splits": {k: feats[k]["stats"] for k in ("train", "valid")},
            "train_norm_fingerprint": C.PREP_DATA.norm_fingerprint(view.train_norm),
            "train_norm_source": view.train_norm["source"],
            "train_norm_valid_or_test_used": view.train_norm["valid_or_test_used"],
        },
        "null_checks_sha256": C.sha256_file(C.ANALYSIS / "null_checks.json"),
        "null_checks_summary": {
            "per_split": {
                k: {
                    "n_attempts": v["n_attempts"],
                    "n_accepted": v["n_accepted"],
                    "changed_fraction_of_nonzero": v["changed_fraction_of_nonzero"],
                    "jaccard_nonzero": v["jaccard_nonzero"],
                }
                for k, v in null_stats["gates"][0]["details"]["per_split"].items()
            }
        },
        "test_scoring": {
            "script": "code/si_score.py",
            "script_sha256": C.sha256_file(C.CODE_DIR / "si_score.py") if (C.CODE_DIR / "si_score.py").exists() else None,
            "protocol": "single one-shot pass over all 12 frozen checkpoints + SPARSE_ONLY; official OGB Evaluator "
            "ROC-AUC cross-checked with sklearn; AP reported; no re-selection",
            "test_randomisation_seed": int(C.NULL_SEED["test"]),
            "test_randomisation_built_inside_the_pass": True,
        },
        "gates": {
            "correctness_gates_sha256": C.sha256_file(C.ANALYSIS / "correctness_gates.json"),
            "status": gates["status"],
        },
        "training_gpu_hours": float(sum(u["gpu_hours"] for u in units)),
        "result_label": C.RESULT_LABEL,
    }
    manifest["manifest_sha256"] = C.sha256_text(C.stable_json(manifest))
    C.write_json(out_path, manifest)
    C.log(
        f"frozen {len(units)} neural units + SPARSE_ONLY; manifest sha256={manifest['manifest_sha256'][:16]} "
        f"gates={gates['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
