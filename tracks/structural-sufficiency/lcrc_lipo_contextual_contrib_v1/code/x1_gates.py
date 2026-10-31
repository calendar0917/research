"""CPU correctness gates for the contextual-contribution round.

Twelve required gates are evaluated on CPU before any GPU work, plus two
supplementary mechanical checks (dropout execution and role coverage / parameter
counts).  Every gate writes a machine-readable record; any failure aborts the
round before CUDA is used.

The runtime ``open`` audit installed here is the evidence for the Test-isolation
gate: while data, models, two-batch training smoke and validation prediction all
run, no Test label / Test prediction / Test metric artifact may be opened.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import x1_common as C
import x1_data as XD
import x1_models as M

import lcrc_data  # noqa: E402
import r42_metrics as MET  # noqa: E402


GATE_SPECS: tuple[tuple[str, str], ...] = (
    ("data_split_integrity", "random42 Train3360/Validation420 rows, sizes, order and hashes match the frozen file"),
    ("test_isolation", "no Test label, Test prediction, Test metric or per-molecule Test artifact is opened"),
    ("base_matches_original_lcrc", "BASE equals the original lcrc_model.LCRC at the same seed and input"),
    ("shared_initialisation", "CONTEXT shared parameters/buffers are bit-identical to BASE"),
    ("zero_out_equivalence", "W_out forced to zero makes CONTEXT equal same-weight BASE (tol 1e-5)"),
    ("element_role_masks", "hetero / carbon_halogen masks match spec.atom_type_metadata"),
    ("renumber_and_reorder", "atom renumbering and batch molecule reordering do not change predictions"),
    ("no_context_mixing", "per-molecule context never mixes across molecules in a batch"),
    ("new_layer_gradients", "all four new Linear layers get finite non-zero gradients after two steps"),
    ("rho_uses_original_h", "rho reads only the original local h, never h_new"),
    ("cpu_two_batch_smoke", "two-batch CPU forward/backward/optimizer step succeeds for both arms"),
    ("denormalisation_and_metrics", "per-molecule denormalisation and RMSE/MAE/R2 are correct"),
)


def _gate(checks: list[dict[str, Any]], gate_id: str, name: str, fn: Callable[[], tuple[bool, dict[str, Any]]]) -> None:
    try:
        passed, details = fn()
    except Exception as exc:  # noqa: BLE001 - gate failure must be recorded, not hidden
        passed, details = False, {"error": f"{type(exc).__name__}: {exc}"}
    checks.append({"id": gate_id, "name": name, "passed": bool(passed), "details": details})
    C.log(f"[gate] {gate_id}: {'PASS' if passed else 'FAIL'}")


# ---------------------------------------------------------------------------
# individual gates
# ---------------------------------------------------------------------------


def gate_split_integrity(split: Mapping[str, Any], labels, views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    audit = views["split_audit"]
    ok = (
        audit["sha256"] == C.V1_SPLIT_SHA256
        and audit["n_train"] == C.SPLIT_SIZES["train"]
        and audit["n_valid"] == C.SPLIT_SIZES["valid"]
        and audit["train_rows_sha256"] == C.SPLIT_SHA256["train"]
        and audit["valid_rows_sha256"] == C.SPLIT_SHA256["valid"]
        and audit["first_five"] == C.SPLIT_FIRST_ROWS
        and audit["test_row_values_used"] is False
        and C.sha256_file(C.V1_LABELS_FILE) == C.V1_LABELS_SHA256
        and labels.audit["test_labels_included"] is False
        and labels.audit["n_train"] == C.SPLIT_SIZES["train"]
        and labels.audit["n_valid"] == C.SPLIT_SIZES["valid"]
        and int(views["train_cache"]["n_molecules"]) == C.SPLIT_SIZES["train"]
        and int(views["valid_cache"]["n_molecules"]) == C.SPLIT_SIZES["valid"]
    )
    return ok, {
        "split": audit,
        "label_audit": dict(labels.audit),
        "target_norm": dict(views["target_norm"]),
        "train_norm_fingerprint": views["train_norm_fingerprint"],
        "lcrc_cache": {
            "path": C.track_rel(views["built"]["path"]),
            "file_sha256": views["built"]["file_sha256"],
            "content_fingerprint": views["built"]["content_fingerprint"],
            "source": views["built"]["source"],
            "labels_present": False,
        },
        "n_train_cache": int(views["train_cache"]["n_molecules"]),
        "n_valid_cache": int(views["valid_cache"]["n_molecules"]),
    }


def gate_test_isolation(opened: Sequence[str], labels) -> tuple[bool, dict[str, Any]]:
    forbidden = XD.forbidden_reads(opened)
    try:
        labels.get("test")
        raises = False
    except Exception:
        raises = True
    # Static scan: no implementation module may reference a Test artifact.
    tokens = tuple(t for t in XD.FORBIDDEN_READ_TOKENS if t != "/predictions/")
    offending: dict[str, list[str]] = {}
    for path in sorted(C.CODE_DIR.glob("*.py")):
        if path.name in ("x1_data.py", "x1_gates.py"):
            continue  # these two files define and exercise the guard itself
        text = path.read_text(encoding="utf-8")
        hits = [token for token in tokens if token in text]
        if hits:
            offending[path.name] = hits
    ok = not forbidden and raises and not offending
    return ok, {
        "forbidden_tokens": list(XD.FORBIDDEN_READ_TOKENS),
        "n_files_opened_during_gates": len(opened),
        "forbidden_opens": forbidden,
        "test_label_accessor_raises": raises,
        "static_scan_offending_files": offending,
        "test_labels_read": False,
        "test_predictions_read": False,
        "test_metrics_read": False,
        "note": "the frozen split JSON is read as a whole; its Test row values are never used and no Test artifact is opened",
    }


def build_data_audit(split: Mapping[str, Any], labels, views: Mapping[str, Any]) -> dict[str, Any]:
    spec = views["spec"]
    return {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "PASS",
        "git_commit": C.git_commit(),
        "environment": C.env_info(),
        "split": dict(views["split_audit"]),
        "labels": {
            **dict(labels.audit),
            "valid_vector_sha256": C.VALID_VECTOR_SHA256,
            "train_vector_sha256": C.TARGET_VECTOR_SHA256,
            "scope": "Train3360 + Validation420 only",
        },
        "target_norm": dict(views["target_norm"]),
        "train_norm_fingerprint": views["train_norm_fingerprint"],
        "lcrc_cache": {
            "path": C.track_rel(views["built"]["path"]),
            "file_sha256": views["built"]["file_sha256"],
            "content_fingerprint": views["built"]["content_fingerprint"],
            "source": views["built"]["source"],
            "v1_content_fingerprint": views["built"]["v1_content_fingerprint"],
            "labels_present": False,
        },
        "feature_spec": {
            "name": spec.name,
            "hash": spec.hash(),
            "atom_feat_dim": int(spec.atom_feat_dim),
            "n_atom_types": int(spec.n_atom_types),
        },
        "role_coverage": {
            "train3360": XD.role_coverage(views["train_cache"], spec),
            "valid420": XD.role_coverage(views["valid_cache"], spec),
        },
        "test_row_values_used": False,
        "test_labels_read": False,
        "test_predictions_read": False,
        "test_metrics_read": False,
    }


def gate_denormalisation_and_metrics(labels, views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    rng = np.random.default_rng(20260205)
    y = np.asarray(labels.y_valid, dtype=np.float64)
    pred_std = rng.normal(0.0, 0.6, size=y.size)
    norm = views["target_norm"]
    pred_raw = XD.denormalise(pred_std, norm)
    expected_raw = pred_std * float(norm["scale"]) + float(norm["mean"])
    manual = {
        "rmse": float(np.sqrt(np.mean((pred_raw - y) ** 2))),
        "mae": float(np.mean(np.abs(pred_raw - y))),
        "r2": float(1.0 - np.sum((y - pred_raw) ** 2) / np.sum((y - np.mean(y)) ** 2)),
    }
    module = MET.all_metrics(pred_raw, y)
    rmse_std = float(np.sqrt(np.mean((pred_std - (y - float(norm["mean"])) / float(norm["scale"])) ** 2)))
    ok = (
        np.array_equal(pred_raw, expected_raw)
        and max(abs(manual[k] - module[k]) for k in manual) <= 1.0e-12
        and abs(rmse_std * float(norm["scale"]) - module["rmse"]) <= 1.0e-12
        and abs(module["r2"] - manual["r2"]) <= 1.0e-12
    )
    return ok, {
        "n": int(y.size),
        "denormalisation_exact": bool(np.array_equal(pred_raw, expected_raw)),
        "manual": manual,
        "module": module,
        "max_abs_module_vs_manual": max(abs(manual[k] - module[k]) for k in manual),
        "rmse_scale_identity_error": abs(rmse_std * float(norm["scale"]) - module["rmse"]),
        "target_norm": dict(norm),
        "tolerance": 1.0e-12,
    }


def gate_cpu_two_batch_smoke(spec, views: Mapping[str, Any], labels) -> tuple[bool, dict[str, Any]]:
    """Two real batches through forward/backward/AdamW for both arms."""
    train_cache = views["train_cache"]
    target_norm = views["target_norm"]
    y_std = XD.target_tensor(labels.y_train, target_norm)
    batches = [list(range(0, 32)), list(range(32, 64))]
    results: dict[str, Any] = {}
    ok = True
    for arm in C.ARMS:
        model, meta = M.build_arm(arm, spec, views["train_norm"], C.TRAIN_SEED, "cpu")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(C.TRAIN_RECIPE["peak_lr"]),
            weight_decay=float(C.TRAIN_RECIPE["weight_decay"]),
        )
        before = {name: param.detach().clone() for name, param in model.named_parameters()}
        losses: list[float] = []
        grad_norms: list[float] = []
        model.train()
        for rows in batches:
            batch = M.to_device(lcrc_data.collate(train_cache, rows), "cpu")
            target = y_std[rows]
            pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.TRAIN_RECIPE["gradient_clip"])))
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(gnorm)
            ok = ok and math.isfinite(losses[-1]) and math.isfinite(gnorm)
        updates = {
            name: float((param.detach() - before[name]).abs().max())
            for name, param in model.named_parameters()
        }
        model.eval()
        with torch.no_grad():
            finite_pred = model(M.to_device(lcrc_data.collate(train_cache, batches[0]), "cpu"))
        results[arm] = {
            "parameters": int(meta["parameters"]),
            "losses_standardised": losses,
            "grad_norms": grad_norms,
            "all_parameters_finite": bool(all(np.isfinite(v) for v in losses + grad_norms)),
            "max_abs_parameter_update": float(max(updates.values())),
            "all_parameter_tensors_updated": bool(all(v > 0.0 for v in updates.values())),
            "prediction_finite": bool(torch.isfinite(finite_pred).all()),
            "prediction_shape": list(finite_pred.shape),
        }
        ok = ok and results[arm]["all_parameter_tensors_updated"] and results[arm]["prediction_finite"]
    return ok, results


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_gates() -> dict[str, Any]:
    cfg = C.load_config()
    checks: list[dict[str, Any]] = []
    rows = list(range(8))

    with XD.read_audit() as opened:
        split, labels = XD.load_frozen_split_and_labels()
        views = XD.build_lcrc_views(split, labels)
        spec = views["spec"]
        train_cache = views["train_cache"]
        valid_cache = views["valid_cache"]

        _gate(checks, *GATE_SPECS[0], lambda: gate_split_integrity(split, labels, views))
        _gate(checks, *GATE_SPECS[2], lambda: (lambda r: (r["passed"], r))(
            M.base_matches_original_lcrc(spec, views["train_norm"], C.TRAIN_SEED, train_cache, rows)
        ))
        base, base_info = M.build_base(spec, views["train_norm"], C.TRAIN_SEED, "cpu")
        context, base_ref, context_info = M.build_context(spec, views["train_norm"], C.TRAIN_SEED, "cpu")

        _gate(
            checks,
            *GATE_SPECS[3],
            lambda: (
                context_info["shared_init"]["all_shared_bitwise_identical"],
                {
                    **context_info["shared_init"],
                    "n_shared_entries_base": len(base.state_dict()),
                    "n_shared_entries_context": len(context.state_dict()),
                    "parameters_base": base_info["parameters"],
                    "parameters_context": context_info["parameters"],
                    "parameters_new": context_info["parameters_new"],
                    "context_is_base_plus_new_only": set(context.state_dict()) - set(base.state_dict())
                    == set(context_info["shared_init"]["model_only_keys"]),
                },
            ),
        )
        _gate(checks, *GATE_SPECS[4], lambda: (lambda r: (r["passed"], r))(M.zero_out_equivalence_audit(base_ref, context, train_cache, rows)))
        _gate(checks, *GATE_SPECS[5], lambda: (lambda r: (r["passed"], r))(M.role_mask_audit(spec, context, train_cache, rows)))
        _gate(
            checks,
            *GATE_SPECS[6],
            lambda: (
                lambda renum, reorder: (
                    bool(renum["passed"] and reorder["passed"]),
                    {"renumber": renum, "molecule_reorder": reorder},
                )
            )(
                M.renumber_invariance_audit(context, train_cache, rows, seed=C.TRAIN_SEED),
                M.molecule_reorder_audit(context, train_cache, rows),
            ),
        )
        _gate(checks, *GATE_SPECS[7], lambda: (lambda r: (r["passed"], r))(M.context_no_mixing_audit(context, train_cache, rows)))
        def _gradient_gate():
            probe_context, _probe_base, _probe_meta = M.build_context(
                spec, views["train_norm"], C.TRAIN_SEED, "cpu"
            )
            result = M.gradient_probe(probe_context, train_cache, list(range(32)), steps=2)
            return result["passed"], result

        _gate(checks, *GATE_SPECS[8], _gradient_gate)
        _gate(checks, *GATE_SPECS[9], lambda: (lambda r: (r["passed"], r))(M.rho_input_audit(context, train_cache, rows)))
        _gate(checks, *GATE_SPECS[10], lambda: gate_cpu_two_batch_smoke(spec, views, labels))
        _gate(checks, *GATE_SPECS[11], lambda: gate_denormalisation_and_metrics(labels, views))

        # Validation predictions under the audit (proves the analysis path is clean too).
        valid_pred_base = M.predict(base, valid_cache, "cpu")
        valid_pred_context = M.predict(context, valid_cache, "cpu")
        validation_prediction_finite = bool(np.isfinite(valid_pred_base).all() and np.isfinite(valid_pred_context).all())

        _gate(checks, *GATE_SPECS[1], lambda: gate_test_isolation(list(opened), labels))

        supplementary = {
            "dropout_execution_base": M.dropout_activity_audit(base, train_cache, rows),
            "dropout_execution_context": M.dropout_activity_audit(context, train_cache, rows),
            "dropout_p": float(C.TRAIN_RECIPE["dropout"]),
            "parameter_counts": {
                "base": base_info["parameters"],
                "context": context_info["parameters"],
                "new": context_info["parameters_new"],
                "references": {
                    "base": C.LCRC_BASE_PARAMS_REFERENCE,
                    "new": C.CONTEXT_NEW_PARAMS_REFERENCE,
                    "context": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
                },
                "counts_match_reference": (
                    base_info["parameters"] == C.LCRC_BASE_PARAMS_REFERENCE
                    and context_info["parameters"] == C.CONTEXT_TOTAL_PARAMS_REFERENCE
                    and context_info["parameters_new"] == C.CONTEXT_NEW_PARAMS_REFERENCE
                ),
            },
            "role_coverage_train3360": XD.role_coverage(train_cache, spec),
            "role_coverage_valid420": XD.role_coverage(valid_cache, spec),
            "context_initial_magnitude_valid420": M.context_diagnostics(context, valid_cache, "cpu"),
            "validation_prediction_finite": validation_prediction_finite,
            "validation_prediction_preview": {
                "base_first5": valid_pred_base[:5].tolist(),
                "context_first5": valid_pred_context[:5].tolist(),
                "max_abs_base_minus_context": float(np.abs(valid_pred_base - valid_pred_context).max()),
            },
        }
        supplementary["passed"] = bool(
            supplementary["dropout_execution_base"]["passed"]
            and supplementary["dropout_execution_context"]["passed"]
            and supplementary["parameter_counts"]["counts_match_reference"]
            and validation_prediction_finite
        )

    order = {spec_id: index for index, (spec_id, _) in enumerate(GATE_SPECS)}
    checks.sort(key=lambda row: order.get(row["id"], len(order)))
    n_passed = int(sum(1 for c in checks if c["passed"]))
    ok = bool(checks and n_passed == len(GATE_SPECS) and supplementary["passed"])
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.config_hash(cfg),
        "code_hashes": C.code_file_hashes(),
        "environment": C.env_info(),
        "device": "cpu",
        "n_gates": len(GATE_SPECS),
        "n_passed": n_passed,
        "ok": ok,
        "status": "PASS" if ok else "FAIL",
        "checks": checks,
        "supplementary": supplementary,
        "test_labels_read": False,
        "test_predictions_read": False,
        "test_metrics_read": False,
    }
    C.write_json(C.CPU_GATES, payload)
    return payload


__all__ = ["GATE_SPECS", "run_gates", "build_data_audit", "gate_denormalisation_and_metrics"]
