"""Post-hoc recomputation of the descriptive CONTEXT diagnostics.

The formal runs at ``b7f520e`` used a diagnostic helper that stored the signed
``f(h_new) - f(h_original)`` values under fields named ``*_abs_*``.  The model,
the training recipe, the checkpoints, the predictions and the decision rule are
unaffected; only the reported magnitudes were wrong (they could be negative).

This script recomputes those descriptive statistics from the frozen checkpoint
on CPU, verifies the checkpoint hash recorded by the training run, and writes
``analysis/context_diagnostics_recomputed.json``.  It then replaces only the
``used`` block of ``analysis/context_usage.json`` and records the provenance.
"""

from __future__ import annotations

import numpy as np
import torch

import x1_common as C
import x1_data as XD
import x1_models as M

import r42_metrics as MET


def main() -> dict:
    cfg = C.load_config()
    split, labels = XD.load_frozen_split_and_labels()
    views = XD.build_lcrc_views(split, labels)
    run_dir = C.RUNS / C.ARM_RUN_KEY["context"]
    payload = C.read_json(run_dir / "train.json")
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    checkpoint_sha = C.sha256_file(run_dir / "checkpoint.pt")
    if checkpoint_sha != payload["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed since the training run")
    if checkpoint["arm"] != "context":
        raise RuntimeError("unexpected checkpoint arm")

    model, _base, _meta = M.build_context(views["spec"], views["train_norm"], int(cfg["seed"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if M.parameter_count(model) != payload["model_audit"]["parameters"]:
        raise RuntimeError("parameter count changed")

    valid_diag = M.context_diagnostics(model, views["valid_cache"], "cpu")
    train_diag = M.context_diagnostics(model, views["train_cache"], "cpu")

    valid = np.load(run_dir / "valid_predictions.npz")
    y_valid = np.asarray(valid["y_true"], dtype=np.float64)
    full_pred = np.asarray(valid["pred"], dtype=np.float64)
    disabled_pred = XD.denormalise(
        M.predict_with_context_disabled(model, views["valid_cache"], "cpu"),
        views["target_norm"],
    )
    ablation = {
        "rmse": MET.rmse(disabled_pred, y_valid),
        "mae": MET.mae(disabled_pred, y_valid),
        "r2": MET.r2(disabled_pred, y_valid),
        "rmse_change_vs_full_context": MET.rmse(disabled_pred, y_valid) - MET.rmse(full_pred, y_valid),
        "mean_abs_prediction_change_vs_full_context": float(np.mean(np.abs(disabled_pred - full_pred))),
        "max_abs_prediction_change_vs_full_context": float(np.max(np.abs(disabled_pred - full_pred))),
        "note": "full CONTEXT vs the same trained CONTEXT with W_out zeroed in eval (its own base path)",
    }

    original = (payload.get("diagnostics") or {}).get("context_usage") or {}
    original_valid = original.get("valid420") or {}
    artifact = {
        "created_utc": C.utcnow(),
        "status": "POST_HOC_REPORTING_FIX",
        "training_commit": payload.get("git_commit"),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha256_matches_run_record": True,
        "predictions_unchanged": True,
        "model_definition_unchanged": True,
        "decision_unchanged": True,
        "why": (
            "the training-time diagnostic stored signed f(h_new)-f(h_original) under "
            "*_abs_* names; the corrected helper takes np.abs() for the hetero-atom "
            "fields. Recomputing this descriptive block cannot change the trained "
            "weights, the predictions, the metrics or the pre-registered decision."
        ),
        "device_for_recompute": "cpu (checkpoint trained on cuda:0; float32 equivalence)",
        "original_signed_values": {
            key: original_valid.get(key)
            for key in (
                "f_delta_abs_mean_over_hetero_atoms",
                "f_delta_abs_max_over_hetero_atoms",
            )
        },
        "valid420": valid_diag,
        "train3360": train_diag,
        "w_out_zero_ablation_valid": ablation,
    }
    out = C.ANALYSIS / "context_diagnostics_recomputed.json"
    C.write_json(out, artifact)

    usage_path = C.CONTEXT_USAGE
    if usage_path.exists():
        usage = C.read_json(usage_path)
        usage["used"]["valid420"] = valid_diag
        usage["used"]["train3360"] = train_diag
        usage["used"]["w_out_zero_ablation_valid"] = ablation
        usage["provenance"] = {
            "recomputed_after_training": True,
            "artifact": C.track_rel(out),
            "training_commit": payload.get("git_commit"),
            "checkpoint_sha256": checkpoint_sha,
            "note": "descriptive diagnostics only; predictions, metrics and decision were not touched",
        }
        C.write_json(usage_path, usage)
    C.log(f"[posthoc] recomputed descriptive diagnostics from {C.track_rel(run_dir / 'checkpoint.pt')}")
    return artifact


if __name__ == "__main__":
    main()
