"""CPU correctness gates for the context mechanism check v2.

All gates run on CPU before any formal GPU training.  The required gates are the
ones named in the task: frozen split/hash integrity, shared parameter
initialisation, parameter counts, atom renumbering invariance, cross-molecule
aggregation isolation, correct ROLEBLIND all-atom aggregation, non-zero
gradients, Test inference not updating weights, and ROLEBLIND prediction
invariance to batch composition and graph order.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import m2_common as C
import m2_data as MD
import m2_models as M

import lcrc_data  # noqa: E402
import r42_metrics as MET  # noqa: E402
import x1_models as X1M  # noqa: E402


GATE_SPECS: tuple[tuple[str, str], ...] = (
    ("data_split_hashes", "Train3360 / Validation420 / Test420 rows, order, sizes and label hashes match the frozen artifacts"),
    ("shared_init_seed1", "BASE_s1 and CONTEXT_s1 share bit-identical seed-1 public parameters, batch order and optimisation settings"),
    ("roleblind_init_equals_context_s0", "ROLEBLIND_s0 has exactly CONTEXT_s0's state entries, shapes, parameter names and initial values"),
    ("roleblind_all_atom_aggregation", "ROLEBLIND uses m_het = m_ch = exact per-molecule all-atom mean of h"),
    ("roleblind_batch_invariance", "ROLEBLIND predictions do not depend on batch composition or graph order"),
    ("atom_renumbering_invariance", "atom renumbering does not change BASE / CONTEXT / ROLEBLIND predictions"),
    ("cross_molecule_isolation", "per-molecule context never mixes across molecules in a batch"),
    ("nonzero_gradients", "all four new ROLEBLIND and CONTEXT layers receive finite non-zero gradients and update"),
    ("test_inference_freezes_weights", "Test inference changes no weight and creates no gradient; predictions stay row-aligned"),
    ("denormalisation_and_metrics", "per-molecule denormalisation and RMSE/MAE/R2 recompute correctly"),
    ("cpu_two_batch_smoke", "two-batch CPU forward/backward/AdamW step succeeds for BASE_s1, CONTEXT_s1 and ROLEBLIND_s0"),
)


def _gate(checks: list[dict[str, Any]], gate_id: str, name: str, fn: Callable[[], tuple[bool, dict[str, Any]]]) -> None:
    try:
        passed, details = fn()
    except Exception as exc:  # noqa: BLE001 - a gate failure must be recorded, not hidden
        passed, details = False, {"error": f"{type(exc).__name__}: {exc}"}
    checks.append({"id": gate_id, "name": name, "passed": bool(passed), "details": details})
    C.log(f"[gate] {gate_id}: {'PASS' if passed else 'FAIL'}")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_data(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    split = views["split"]
    labels = views["labels"]
    test = views["test"]
    rows = {name: [int(x) for x in split["rows"][name]] for name in ("train", "valid", "test")}
    hashes = {name: C.rows_sha256_int64(values) for name, values in rows.items()}
    details = {
        "split_file": C.track_rel(C.V1_SPLIT_FILE),
        "split_file_sha256": C.V1_SPLIT_SHA256,
        "row_hashes": hashes,
        "row_hashes_expected": dict(C.SPLIT_SHA256),
        "first_five": {name: values[:5] for name, values in rows.items()},
        "sizes": {name: len(values) for name, values in rows.items()},
        "disjoint": (
            not (set(rows["train"]) & set(rows["valid"]))
            and not (set(rows["train"]) & set(rows["test"]))
            and not (set(rows["valid"]) & set(rows["test"]))
        ),
        "union_is_4200": set(rows["train"]) | set(rows["valid"]) | set(rows["test"]) == set(range(C.N_MOLECULES)),
        "labels": dict(labels["audit"]),
        "test_labels": dict(test["audit"]),
        "cache": {
            "path": C.track_rel(views["cache_built"]["path"]),
            "file_sha256": views["cache_built"]["file_sha256"],
            "content_fingerprint": views["cache_built"]["content_fingerprint"],
            "labels_present": False,
        },
        "train_norm_fingerprint": views["train_norm_fingerprint"],
        "target_norm": views["target_norm"],
        "n_train_cache": int(views["train_cache"]["n_molecules"]),
        "n_valid_cache": int(views["valid_cache"]["n_molecules"]),
        "n_test_cache": int(views["test_cache"]["n_molecules"]),
        "cache_row_identity_checked_in_build_views": True,
    }
    ok = bool(
        hashes == C.SPLIT_SHA256
        and details["first_five"] == C.SPLIT_FIRST_ROWS
        and details["sizes"] == C.SPLIT_SIZES
        and details["disjoint"]
        and details["union_is_4200"]
        and labels["audit"]["train_vector_sha256"] == C.TARGET_VECTOR_SHA256
        and labels["audit"]["valid_vector_sha256"] == C.VALID_VECTOR_SHA256
        and test["audit"]["label_vector_sha256"] == C.TEST_VECTOR_SHA256
        and test["audit"]["rows_sha256"] == C.SPLIT_SHA256["test"]
        and views["cache_built"]["content_fingerprint"] == C.LCRC_CACHE_CONTENT_FINGERPRINT
        and views["train_norm_fingerprint"] == C.LCRC_TRAIN_NORM_FINGERPRINT
        and views["target_norm"]["mean"] == C.TARGET_MEAN
        and views["target_norm"]["scale"] == C.TARGET_SCALE
    )
    return ok, details


def _batch_order(seed: int, n_train: int, batch_size: int, epochs: int = 1) -> tuple[list[list[int]], str]:
    generator = torch.Generator().manual_seed(int(seed) + C.SHUFFLE_OFFSET)
    batches: list[list[int]] = []
    for _ in range(epochs):
        perm = torch.randperm(int(n_train), generator=generator).tolist()
        batches.extend([perm[i : i + int(batch_size)] for i in range(0, len(perm), int(batch_size))])
    import hashlib

    h = hashlib.sha256()
    for batch in batches:
        h.update(np.ascontiguousarray(np.asarray(batch, dtype=np.int64)).tobytes())
    return batches, h.hexdigest()


def gate_shared_init_seed1(views: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec = views["spec"]
    norm = views["train_norm"]
    base, base_meta = M.build_base(spec, norm, 1, "cpu")
    context, base_ref, context_meta = M.build_context(spec, norm, 1, "cpu")
    shared = M.shared_init_audit(base, context)
    model_only = sorted(set(context.state_dict()) - set(base.state_dict()))
    context_only_params = sorted(
        set(dict(context.named_parameters())) - set(dict(base.named_parameters()))
    )
    n_train = int(views["train_cache"]["n_molecules"])
    batches_a, hash_a = _batch_order(1, n_train, int(cfg["training"]["batch_size"]), epochs=2)
    batches_b, hash_b = _batch_order(1, n_train, int(cfg["training"]["batch_size"]), epochs=2)
    details = {
        "shared_init": shared,
        "model_only_state_entries": model_only,
        "model_only_parameters": context_only_params,
        "model_only_entries_are_all_context_layers": all(key.startswith("context.") for key in model_only),
        "parameters_base": base_meta["parameters"],
        "parameters_context": context_meta["parameters"],
        "parameters_new": context_meta["parameters_new"],
        "parameter_counts_expected": {
            "base": C.LCRC_BASE_PARAMS_REFERENCE,
            "context": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
            "new": C.CONTEXT_NEW_PARAMS_REFERENCE,
        },
        "batch_order_sha256_first_two_epochs_arm_a": hash_a,
        "batch_order_sha256_first_two_epochs_arm_b": hash_b,
        "batch_order_identical": hash_a == hash_b,
        "shuffle_generator_seed": 1 + C.SHUFFLE_OFFSET,
        "dropout_seed_after_init": 1,
        "optimizer_and_schedule_identical_by_config": {
            "optimizer": cfg["training"]["optimizer"],
            "adam_betas": cfg["training"]["adam_betas"],
            "adam_eps": cfg["training"]["adam_eps"],
            "peak_lr": cfg["training"]["peak_lr"],
            "min_lr": cfg["training"]["min_lr"],
            "scheduler": cfg["training"]["scheduler"],
            "weight_decay": cfg["training"]["weight_decay"],
            "gradient_clip": cfg["training"]["gradient_clip"],
            "batch_size": cfg["training"]["batch_size"],
            "epochs": cfg["training"]["epochs"],
        },
        "n_batches_first_two_epochs": len(batches_a),
    }
    ok = bool(
        shared["all_shared_bitwise_identical"]
        and not shared["mismatched"]
        and details["model_only_entries_are_all_context_layers"]
        and set(model_only) == set(context_only_params)
        and base_meta["parameters"] == C.LCRC_BASE_PARAMS_REFERENCE
        and context_meta["parameters"] == C.CONTEXT_TOTAL_PARAMS_REFERENCE
        and context_meta["parameters_new"] == C.CONTEXT_NEW_PARAMS_REFERENCE
        and details["batch_order_identical"]
    )
    return ok, details


def gate_roleblind_init(views: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec = views["spec"]
    norm = views["train_norm"]
    base0, base0_meta = M.build_base(spec, norm, 0, "cpu")
    context0, base0_ref, context0_meta = M.build_context(spec, norm, 0, "cpu")
    roleblind0, context0_ref, roleblind0_meta = M.build_roleblind(spec, norm, 0, "cpu")
    parity = M.init_parity(context0, roleblind0)
    hist_context = C.read_json(C.HISTORICAL_CHECKPOINTS["context_s0"]["train_json"])
    hist_base = C.read_json(C.HISTORICAL_CHECKPOINTS["base_s0"]["train_json"])
    fresh_context_hash = C.state_dict_fingerprint(context0.state_dict())
    fresh_base_hash = C.state_dict_fingerprint(base0.state_dict())
    details = {
        "init_parity_context_s0_vs_roleblind_s0": parity,
        "parameters_context": context0_meta["parameters"],
        "parameters_roleblind": roleblind0_meta["parameters"],
        "parameters_new_roleblind": roleblind0_meta["parameters_new"],
        "parameter_counts_expected": {
            "context": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
            "roleblind": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
            "new": C.CONTEXT_NEW_PARAMS_REFERENCE,
        },
        "fresh_context_s0_init_hash": fresh_context_hash,
        "historical_context_s0_init_hash": hist_context["model_audit"]["init_state_hash"],
        "fresh_context_s0_init_matches_historical": fresh_context_hash == hist_context["model_audit"]["init_state_hash"],
        "fresh_base_s0_init_hash": fresh_base_hash,
        "historical_base_s0_init_hash": hist_base["model_audit"]["init_state_hash"],
        "fresh_base_s0_init_matches_historical": fresh_base_hash == hist_base["model_audit"]["init_state_hash"],
        "roleblind_meta_init_parity": roleblind0_meta["init_parity_with_context"],
    }
    ok = bool(
        parity["all_entries_bitwise_identical"]
        and parity["all_parameters_bitwise_identical"]
        and parity["state_dict_keys_identical"]
        and parity["parameter_names_identical"]
        and parity["parameter_counts_equal"]
        and context0_meta["parameters"] == C.CONTEXT_TOTAL_PARAMS_REFERENCE
        and roleblind0_meta["parameters"] == C.CONTEXT_TOTAL_PARAMS_REFERENCE
        and details["fresh_context_s0_init_matches_historical"]
        and details["fresh_base_s0_init_matches_historical"]
    )
    return ok, details


def gate_roleblind_aggregation(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec = views["spec"]
    norm = views["train_norm"]
    roleblind0, _context, _meta = M.build_roleblind(spec, norm, 0, "cpu")
    valid_audit = M.roleblind_aggregation_audit(roleblind0, views["valid_cache"], list(range(16)), "cpu")
    train_audit = M.roleblind_aggregation_audit(roleblind0, views["train_cache"], list(range(8)), "cpu")
    details = {"valid420_first16": valid_audit, "train3360_first8": train_audit}
    return bool(valid_audit["passed"] and train_audit["passed"]), details


def gate_roleblind_batch_invariance(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec = views["spec"]
    norm = views["train_norm"]
    roleblind0, _context, _meta = M.build_roleblind(spec, norm, 0, "cpu")
    audit = M.roleblind_batch_invariance_audit(
        roleblind0, views["valid_cache"], list(range(32)), "cpu", subset_size=32, tolerance=1.0e-6
    )
    return bool(audit["passed"]), audit


def gate_renumbering(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec = views["spec"]
    norm = views["train_norm"]
    base1, _ = M.build_base(spec, norm, 1, "cpu")
    context1, _base, _meta = M.build_context(spec, norm, 1, "cpu")
    roleblind0, _context, _meta = M.build_roleblind(spec, norm, 0, "cpu")
    rows = list(range(8))
    audits = {
        "base_s1": X1M.renumber_invariance_audit(base1, views["train_cache"], rows, seed=1),
        "context_s1": X1M.renumber_invariance_audit(context1, views["train_cache"], rows, seed=1),
        "roleblind_s0": X1M.renumber_invariance_audit(roleblind0, views["train_cache"], rows, seed=0),
    }
    return bool(all(a["passed"] for a in audits.values())), audits


def gate_cross_molecule(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec = views["spec"]
    norm = views["train_norm"]
    context1, _base, _meta = M.build_context(spec, norm, 1, "cpu")
    roleblind0, _context, _meta = M.build_roleblind(spec, norm, 0, "cpu")
    rows = list(range(8))
    audits = {
        "context_s1": X1M.context_no_mixing_audit(context1, views["train_cache"], rows),
        "roleblind_s0": X1M.context_no_mixing_audit(roleblind0, views["train_cache"], rows),
    }
    return bool(all(a["passed"] for a in audits.values())), audits


def _two_step_probe(model, cache, rows, lr: float) -> dict[str, Any]:
    result = X1M.gradient_probe(model, cache, rows, steps=2, lr=lr, device="cpu")
    return result


def gate_gradients(views: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec = views["spec"]
    norm = views["train_norm"]
    context1, _base, _cmeta = M.build_context(spec, norm, 1, "cpu")
    roleblind0, _context, _rmeta = M.build_roleblind(spec, norm, 0, "cpu")
    rows = list(range(32))
    lr = float(cfg["training"]["peak_lr"])
    audits = {
        "context_s1": _two_step_probe(context1, views["train_cache"], rows, lr),
        "roleblind_s0": _two_step_probe(roleblind0, views["train_cache"], rows, lr),
    }
    details = {}
    for name, audit in audits.items():
        details[name] = {
            "new_parameter_names": audit["new_parameter_names"],
            "all_new_layers_finite_gradient": audit["all_new_layers_finite_gradient"],
            "all_new_tensors_nonzero_gradient_at_last_probe_step": audit["all_new_tensors_nonzero_gradient_at_last_probe_step"],
            "all_new_tensors_updated": audit["all_new_tensors_updated"],
            "parameter_max_abs_update": audit["parameter_max_abs_update"],
            "per_tensor_min_gradient_norm_across_steps": {
                key: min(step["grads"][key]["norm"] for step in audit["history"])
                for key in audit["new_parameter_names"]
            },
            "passed": audit["passed"],
        }
    return bool(all(a["passed"] for a in audits.values())), details


def gate_test_inference(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec = views["spec"]
    norm = views["train_norm"]
    test_cache = views["test_cache"]
    audits: dict[str, Any] = {}
    for key, spec_entry in C.HISTORICAL_CHECKPOINTS.items():
        model, _meta = M.build_family(spec_entry["family"], spec, norm, int(spec_entry["seed"]), "cpu")
        payload = M.load_checkpoint(
            model,
            spec_entry["path"],
            expected_sha256=spec_entry["sha256"],
            expected_family=spec_entry["family"],
            expected_seed=int(spec_entry["seed"]),
        )
        audit = M.freeze_weights_audit(model, test_cache, "cpu")
        audit["checkpoint_sha256"] = C.sha256_file(spec_entry["path"])
        audit["soup_state_hash"] = payload.get("soup_state_hash")
        audits[key] = audit

    # Fresh models for the new families (weights are arbitrary but frozen).
    fresh: dict[str, Any] = {}
    for family, seed in (("base", 1), ("context", 1), ("roleblind", 0)):
        model, _meta = M.build_family(family, spec, norm, seed, "cpu")
        fresh[f"{family}_s{seed}"] = M.freeze_weights_audit(model, test_cache, "cpu")
    audits.update(fresh)

    # Row alignment: full Test420 prediction versus three single-molecule calls.
    base0, _meta = M.build_family("base", spec, norm, 0, "cpu")
    M.load_checkpoint(
        base0,
        C.HISTORICAL_CHECKPOINTS["base_s0"]["path"],
        expected_sha256=C.HISTORICAL_CHECKPOINTS["base_s0"]["sha256"],
        expected_family="base",
        expected_seed=0,
    )
    full = M.predict(base0, test_cache, "cpu")
    rows = [int(r) for r in views["split"]["rows"]["test"]]
    alignment = {}
    for index in (0, 137, 419):
        single_cache = MD._select(views["cache_built"]["cache"], [rows[index]], "alignment_single")
        single_pred = M.predict(base0, single_cache, "cpu")[0]
        alignment[str(index)] = {
            "row_id": rows[index],
            "full_prediction": float(full[index]),
            "single_prediction": float(single_pred),
            "abs_diff": float(abs(full[index] - single_pred)),
        }
    alignment_ok = bool(
        all(v["abs_diff"] <= 1.0e-6 for v in alignment.values())
        and len(full) == C.SPLIT_SIZES["test"]
        and np.isfinite(full).all()
    )
    details = {
        "historical_checkpoints": audits,
        "fresh_models": fresh,
        "test_row_alignment": alignment,
        "test_row_alignment_ok": alignment_ok,
        "test_n": int(full.size),
        "note": "inference used torch.no_grad and model.eval(); state entries were compared bitwise before/after",
    }
    ok = bool(
        all(a["passed"] for a in audits.values())
        and all(a["passed"] for a in fresh.values())
        and alignment_ok
    )
    return ok, details


def gate_denormalisation(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    rng = np.random.default_rng(20261101)
    y = np.asarray(views["test"]["y_test"], dtype=np.float64)
    pred_std = rng.normal(0.0, 0.6, size=y.size)
    norm = views["target_norm"]
    pred_raw = MD.denormalise(pred_std, norm)
    expected_raw = pred_std * float(norm["scale"]) + float(norm["mean"])
    manual = {
        "rmse": float(np.sqrt(np.mean((pred_raw - y) ** 2))),
        "mae": float(np.mean(np.abs(pred_raw - y))),
        "r2": float(1.0 - np.sum((y - pred_raw) ** 2) / np.sum((y - np.mean(y)) ** 2)),
    }
    module = MET.all_metrics(pred_raw, y)
    rmse_std = float(np.sqrt(np.mean((pred_std - (y - float(norm["mean"])) / float(norm["scale"])) ** 2)))
    max_diff = max(abs(manual[k] - module[k]) for k in manual)
    ok = bool(
        np.array_equal(pred_raw, expected_raw)
        and max_diff <= 1.0e-12
        and abs(rmse_std * float(norm["scale"]) - module["rmse"]) <= 1.0e-12
    )
    return ok, {
        "n": int(y.size),
        "denormalisation_exact": bool(np.array_equal(pred_raw, expected_raw)),
        "manual": manual,
        "module": module,
        "max_abs_module_vs_manual": float(max_diff),
        "rmse_scale_identity_error": float(abs(rmse_std * float(norm["scale"]) - module["rmse"])),
        "target_norm": dict(norm),
        "tolerance": 1.0e-12,
    }


def gate_two_batch_smoke(views: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    train_cache = views["train_cache"]
    target_norm = views["target_norm"]
    y_std = MD.target_tensor(views["labels"]["y_train"], target_norm)
    batches = [list(range(0, 32)), list(range(32, 64))]
    results: dict[str, Any] = {}
    ok = True
    for arm_key in C.ARM_SPECS:
        arm = C.ARM_SPECS[arm_key]
        model, meta = M.build_family(arm["family"], views["spec"], views["train_norm"], int(arm["seed"]), "cpu")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg["training"]["peak_lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        before = {name: param.detach().clone() for name, param in model.named_parameters()}
        losses: list[float] = []
        grad_norms: list[float] = []
        model.train()
        for rows in batches:
            batch = M.to_device(lcrc_data.collate(train_cache, rows), "cpu")
            target = y_std[rows]
            pred = model(batch)
            loss = F.mse_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip"])))
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
        results[arm_key] = {
            "family": arm["family"],
            "seed": int(arm["seed"]),
            "parameters": int(meta["parameters"]),
            "losses_standardised": losses,
            "grad_norms": grad_norms,
            "all_finite": bool(all(np.isfinite(v) for v in losses + grad_norms)),
            "all_parameter_tensors_updated": bool(all(v > 0.0 for v in updates.values())),
            "max_abs_parameter_update": float(max(updates.values())),
            "prediction_finite": bool(torch.isfinite(finite_pred).all()),
        }
        ok = ok and results[arm_key]["all_parameter_tensors_updated"] and results[arm_key]["prediction_finite"]
    return ok, results


# ---------------------------------------------------------------------------
# supplementary checks
# ---------------------------------------------------------------------------


def supplementary(views: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    spec = views["spec"]
    norm = views["train_norm"]
    base1, base1_meta = M.build_base(spec, norm, 1, "cpu")
    context0, base0, context0_meta = M.build_context(spec, norm, 0, "cpu")
    roleblind0, context0_ref, roleblind0_meta = M.build_roleblind(spec, norm, 0, "cpu")
    rows8 = list(range(8))
    extra = {
        "dropout_execution_base_s1": X1M.dropout_activity_audit(base1, views["train_cache"], rows8),
        "dropout_execution_context_s1": X1M.dropout_activity_audit(
            M.build_context(spec, norm, 1, "cpu")[0], views["train_cache"], rows8
        ),
        "dropout_execution_roleblind_s0": X1M.dropout_activity_audit(roleblind0, views["train_cache"], rows8),
        "parameter_counts": {
            "base": base1_meta["parameters"],
            "context": context0_meta["parameters"],
            "roleblind": roleblind0_meta["parameters"],
            "new_context": context0_meta["parameters_new"],
            "new_roleblind": roleblind0_meta["parameters_new"],
            "references": {
                "base": C.LCRC_BASE_PARAMS_REFERENCE,
                "context": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
                "roleblind": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
                "new": C.CONTEXT_NEW_PARAMS_REFERENCE,
            },
        },
        "zero_out_equivalence_roleblind": X1M.zero_out_equivalence_audit(base0, roleblind0, views["train_cache"], rows8),
        "roleblind_readout_parity": M.roleblind_readout_parity_audit(
            context0, roleblind0, views["train_cache"], rows8, "cpu"
        ),
        "roleblind_usage_valid420": M.roleblind_usage(roleblind0, views["valid_cache"], "cpu"),
        "training_code_test_access_scan": _scan_training_code(),
        "historical_checkpoint_metadata": {},
        "config_hash": C.config_hash(dict(cfg)),
    }
    # Historical checkpoint metadata (read only).
    for key, entry in C.HISTORICAL_CHECKPOINTS.items():
        payload = torch.load(entry["path"], map_location="cpu", weights_only=False)
        extra["historical_checkpoint_metadata"][key] = {
            "sha256": C.sha256_file(entry["path"]),
            "arm": payload.get("arm"),
            "family": payload.get("family"),
            "seed": payload.get("seed"),
            "soup_members": payload.get("soup_members"),
            "soup_state_hash": payload.get("soup_state_hash"),
            "init_state_hash": payload.get("init_state_hash"),
        }
    # Role coverage via the frozen role tables.
    extra["role_coverage"] = {
        "train3360": _role_counts(views["train_cache"], spec),
        "valid420": _role_counts(views["valid_cache"], spec),
        "test420": _role_counts(views["test_cache"], spec),
    }
    extra["passed"] = bool(
        extra["dropout_execution_base_s1"]["passed"]
        and extra["dropout_execution_context_s1"]["passed"]
        and extra["dropout_execution_roleblind_s0"]["passed"]
        and extra["zero_out_equivalence_roleblind"]["passed"]
        and extra["roleblind_readout_parity"]["passed"]
        and extra["roleblind_usage_valid420"]["m_het_input_equals_m_ch_input_for_every_batch"]
        and extra["training_code_test_access_scan"]["passed"]
    )
    return extra


def _role_counts(cache: Mapping[str, Any], spec) -> dict[str, Any]:
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    import x1_data as XD  # historical role tables only

    tables = XD.role_tables(spec)
    het = np.isin(atom_type, np.asarray(tables["hetero_type_ids"], dtype=np.int64))
    ch = np.isin(atom_type, np.asarray(tables["carbon_halogen_type_ids"], dtype=np.int64))
    counts = {"both": 0, "hetero_only": 0, "carbon_halogen_only": 0, "neither": 0}
    for m in range(int(cache["n_molecules"])):
        lo, hi = int(atom_ptr[m]), int(atom_ptr[m + 1])
        h = bool(het[lo:hi].any())
        c = bool(ch[lo:hi].any())
        key = "both" if h and c else "hetero_only" if h else "carbon_halogen_only" if c else "neither"
        counts[key] += 1
    return {
        "n_molecules": int(cache["n_molecules"]),
        "n_atoms": int(atom_type.size),
        "n_hetero_atoms": int(het.sum()),
        "n_carbon_halogen_atoms": int(ch.sum()),
        "molecule_counts": counts,
        "role_tables": tables,
    }


def _scan_training_code() -> dict[str, Any]:
    """Training code must not reference Test labels/cache."""
    offenders: dict[str, list[str]] = {}
    for name in ("m2_train.py", "m2_models.py"):
        path = C.CODE_DIR / name
        text = path.read_text(encoding="utf-8")
        hits = [
            token
            for token in ("load_test_labels", "test_cache", "y_test", "TEST_VECTOR_SHA256")
            if token in text
        ]
        if hits:
            offenders[name] = hits
    return {
        "files_scanned": ["m2_train.py", "m2_models.py"],
        "forbidden_tokens": ["load_test_labels", "test_cache", "y_test", "TEST_VECTOR_SHA256"],
        "offending_files": offenders,
        "passed": not offenders,
        "note": "Test evaluation lives only in m2_evaluate.py; training receives train/valid views only",
    }


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_gates() -> dict[str, Any]:
    cfg = C.load_config()
    checks: list[dict[str, Any]] = []
    views = MD.build_views(include_test=True)

    _gate(checks, *GATE_SPECS[0], lambda: gate_data(views))
    _gate(checks, *GATE_SPECS[1], lambda: gate_shared_init_seed1(views, cfg))
    _gate(checks, *GATE_SPECS[2], lambda: gate_roleblind_init(views, cfg))
    _gate(checks, *GATE_SPECS[3], lambda: gate_roleblind_aggregation(views))
    _gate(checks, *GATE_SPECS[4], lambda: gate_roleblind_batch_invariance(views))
    _gate(checks, *GATE_SPECS[5], lambda: gate_renumbering(views))
    _gate(checks, *GATE_SPECS[6], lambda: gate_cross_molecule(views))
    _gate(checks, *GATE_SPECS[7], lambda: gate_gradients(views, cfg))
    _gate(checks, *GATE_SPECS[8], lambda: gate_test_inference(views))
    _gate(checks, *GATE_SPECS[9], lambda: gate_denormalisation(views))
    _gate(checks, *GATE_SPECS[10], lambda: gate_two_batch_smoke(views, cfg))
    supp = supplementary(views, cfg)

    order = {spec_id: index for index, (spec_id, _) in enumerate(GATE_SPECS)}
    checks.sort(key=lambda row: order.get(row["id"], len(order)))
    n_passed = int(sum(1 for c in checks if c["passed"]))
    ok = bool(checks and n_passed == len(GATE_SPECS) and supp["passed"])
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
        "supplementary": supp,
        "test_access": "EXPOSED_TEST_EXPLORATORY",
        "test_labels_read": True,
        "test_predictions_read": False,
        "test_metrics_read": False,
    }
    C.write_json(C.CPU_GATES, payload)
    return payload


__all__ = ["GATE_SPECS", "run_gates"]
