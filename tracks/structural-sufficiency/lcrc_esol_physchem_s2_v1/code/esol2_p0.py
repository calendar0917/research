"""P0 gates for ``lcrc_esol_physchem_s2_v1`` (CPU, no GPU, no training).

Only the checks the task book asks for are run here — the old round's full
25-gate audit is **not** repeated.  Four blocks:

A. frozen inputs (split / cache / descripts / BASE checkpoints) are byte-identical
   to the previous round and the BASE replay reproduces the recorded Validation
   RMSE;
B. P0.1 — descriptor row alignment, verified for **every** Train/Validation
   molecule by recomputing the six RDKit descriptors from the cache SMILES;
C. P0.2 — exactly one standardisation, with the Train-only scaler, and the six
   descriptors reaching ``rho`` only;
D. P0.3 — BASE-identical initialisation of the widened rho layer, non-zero
   gradients on the six new columns, an actual update after one optimiser step,
   and identical batch order for the same seed;
E. P2 — capacity-matched E: effective parameter count, shared init, typed-S2
   disconnection (restricted batch + invariance + zero dead slot), and a
   positive control that BASE does respond to the same typed-S2 perturbation.

Requires RDKit (local CPU only); the GPU hosts never run this file.
"""

from __future__ import annotations

import csv
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import esol2_common as C
import esol2_data as D
import esol2_models as M
import esol_common as V1C
import esol_raw as V1R


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _examples(values: np.ndarray, n: int = 6) -> list[Any]:
    return [float(x) for x in np.asarray(values, dtype=np.float64).reshape(-1)[:n]]


# ---------------------------------------------------------------------------
# B. row alignment (P0.1)
# ---------------------------------------------------------------------------


def _alignment_block(views: Mapping[str, Any], spec) -> dict[str, Any]:
    physchem_all = np.asarray(views["physchem"], dtype=np.float64)
    split_rows = {name: [int(x) for x in views["split"]["rows"][name]] for name in ("train", "valid")}
    rows_written = 0
    mismatched: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    per_split: dict[str, Any] = {}
    with open(C.P0_ROW_ALIGNMENT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "cache_position",
                "csv_row",
                "molecule_id",
                "smiles",
                *C.PHYSCHEM_NAMES,
                "recomputed_max_abs_diff",
                "artifact_row_max_abs_diff",
                "match",
            ]
        )
        for name, cache_key in (("train", "train_cache"), ("valid", "valid_cache")):
            cache = views[cache_key]
            matrix = np.asarray(views[f"physchem_{name}"], dtype=np.float64)
            local_max = 0.0
            local_rows = 0
            for pos, row in enumerate(split_rows[name]):
                smiles = str(cache["smiles"][pos])
                molecule_id = str(cache["molecule_ids"][pos])
                if int(molecule_id) != int(row):
                    mismatched.append({"split": name, "position": pos, "molecule_id": molecule_id, "csv_row": row})
                recomputed = np.asarray(V1R.compute_physchem(smiles), dtype=np.float64)
                aligned_diff = float(np.max(np.abs(recomputed - matrix[pos])))
                artifact_diff = float(np.max(np.abs(matrix[pos] - physchem_all[row])))
                ok = bool(aligned_diff <= 1.0e-9 and artifact_diff == 0.0)
                if not ok:
                    mismatched.append(
                        {
                            "split": name,
                            "position": pos,
                            "csv_row": row,
                            "aligned_diff": aligned_diff,
                            "artifact_diff": artifact_diff,
                        }
                    )
                local_max = max(local_max, aligned_diff, artifact_diff)
                local_rows += 1
                writer.writerow(
                    [
                        name,
                        pos,
                        row,
                        molecule_id,
                        smiles,
                        *[f"{value:.10g}" for value in recomputed],
                        f"{aligned_diff:.3e}",
                        f"{artifact_diff:.3e}",
                        ok,
                    ]
                )
                rows_written += 1
            max_abs_diff = max(max_abs_diff, local_max)
            per_split[name] = {
                "n_rows_checked": local_rows,
                "max_abs_diff": local_max,
                "matrix_sha256": C.sha256_bytes(np.ascontiguousarray(matrix, dtype="<f8").tobytes()),
            }

    # duplicate SMILES: identical strings must map to identical descriptor rows
    smiles_all = [str(x) for x in views["split"]["smiles"]]
    groups: dict[str, list[int]] = {}
    for row, smiles in enumerate(smiles_all):
        groups.setdefault(smiles, []).append(row)
    dup_groups = {smiles: rows for smiles, rows in groups.items() if len(rows) > 1}
    dup_bad = [
        {"smiles": smiles, "rows": rows}
        for smiles, rows in dup_groups.items()
        if not np.allclose(physchem_all[rows], physchem_all[rows[0]], atol=0.0, rtol=0.0)
    ]
    split_of = {}
    for name, rows in split_rows.items():
        for row in rows:
            split_of[row] = name

    # the v1 failure mode must now be impossible: a cache-local matrix is required
    guard_raised = False
    try:
        M.collate(views["train_cache"], [0, 1], physchem=physchem_all)
    except ValueError:
        guard_raised = True

    return {
        "rows_written": rows_written,
        "per_split": per_split,
        "mismatched": mismatched[:10],
        "n_mismatched": len(mismatched),
        "max_abs_diff_all_rows": max_abs_diff,
        "duplicate_smiles_groups": len(dup_groups),
        "duplicate_smiles_descriptor_mismatches": dup_bad,
        "duplicate_smiles_splits": {
            smiles: sorted({split_of[row] for row in rows}) for smiles, rows in dup_groups.items()
        },
        "collate_guard_rejects_full_matrix": guard_raised,
        "csv": C.track_rel(C.P0_ROW_ALIGNMENT),
        "v1_bug_quantification": views["v1_alignment_bug"],
    }


# ---------------------------------------------------------------------------
# C. single standardisation + rho-only (P0.2)
# ---------------------------------------------------------------------------


def _standardisation_block(views: Mapping[str, Any], model, meta) -> dict[str, Any]:
    cache = views["train_cache"]
    matrix = np.asarray(views["physchem_train"], dtype=np.float64)
    rows = list(range(8))
    batch = M.collate(cache, rows, physchem=matrix)
    raw_seen = batch["physchem"].detach().numpy().astype(np.float64)
    model.eval()
    with torch.no_grad():
        _y, aux = model(batch, return_aux=True)
    seen = aux["rho_input"][:, -C.N_PHYSCHEM :].detach().numpy().astype(np.float64)
    mean = np.asarray(views["physchem_norm"]["mean"], dtype=np.float64)
    scale = np.asarray(views["physchem_norm"]["scale"], dtype=np.float64)
    expected = matrix[rows]
    single = (expected - mean) / scale
    double = (single - mean) / scale
    single_diff = float(np.max(np.abs(single - seen)))
    double_diff = float(np.max(np.abs(double - seen)))
    # the batch carries float32, so the honest comparison is float32 equality
    raw_diff = float(
        np.max(np.abs(raw_seen.astype(np.float32).astype(np.float64) - expected.astype(np.float32).astype(np.float64)))
    )
    raw_float32_cast_error = float(np.max(np.abs(expected - expected.astype(np.float32).astype(np.float64))))

    # Descriptors may only move rho.  At initialisation the six new rho columns
    # are exactly zero by construction, so the readout must first be made live.
    w = model.rho.lins[0].weight
    n_old = int(w.shape[1]) - C.N_PHYSCHEM
    with torch.no_grad():
        generator = torch.Generator().manual_seed(20261103)
        w[:, n_old:] = 0.05 * torch.randn((w.shape[0], C.N_PHYSCHEM), generator=generator)
    with torch.no_grad():
        y_live, aux_live = model(batch, return_aux=True)
        y_live2, aux_live2 = model(batch, return_aux=True)
    live_weights_norm = float(w[:, n_old:].norm())

    delta = np.zeros_like(matrix[: len(rows)])
    delta[:, 0] = 25.0  # +25 g/mol MolWt
    delta[:, 2] = 10.0  # +10 A^2 TPSA
    batch2 = M.collate(cache, rows, physchem=matrix)
    batch2["physchem"] = batch2["physchem"] + torch.as_tensor(delta, dtype=torch.float32)
    with torch.no_grad():
        y2, aux2 = model(batch2, return_aux=True)
    unchanged = {
        key: bool(torch.equal(aux_live[key], aux2[key]))
        for key in ("h", "contrib", "contrib_atom", "mean_h", "mean_h2")
    }
    rho_change = float((aux2["rho"] - aux_live["rho"]).abs().max())
    y_change = float(((y2 - y_live)).abs().max())
    determinism = float((y_live2 - y_live).abs().max())

    return {
        "raw_descriptor_max_abs_diff_vs_artifact": raw_diff,
        "raw_float32_cast_error": raw_float32_cast_error,
        "single_standardisation_max_abs_diff": single_diff,
        "double_standardisation_max_abs_diff": double_diff,
        "single_vs_double_max_abs_diff": float(np.max(np.abs(single - double))),
        "single_vs_double_mean_abs_diff": float(np.mean(np.abs(single - double))),
        "first_row_raw": _examples(expected[0], 6),
        "first_row_single_standardised": _examples(single[0], 6),
        "first_row_model_seen": _examples(seen[0], 6),
        "first_row_double_standardised": _examples(double[0], 6),
        "live_readout_weight_norm": live_weights_norm,
        "repeat_forward_max_abs_diff": determinism,
        "unchanged_when_descriptors_perturbed": unchanged,
        "rho_abs_change_when_descriptors_perturbed": rho_change,
        "prediction_abs_change_when_descriptors_perturbed": y_change,
        "norm_mean": _examples(mean, 6),
        "norm_scale": _examples(scale, 6),
        "norm_fit_on": views["physchem_norm"]["fit_on"],
        "norm_valid_or_test_used": bool(views["physchem_norm"]["valid_or_test_used"]),
        "norm_constant_indices": views["physchem_norm"]["constant_indices"],
        "physchem_input": meta["physchem_input"],
        "single_standardisation_passed": bool(raw_diff == 0.0 and single_diff <= 1e-5),
        "single_standardisation_is_discriminating": bool(double_diff > 1e-3),
        "rho_only_passed": bool(
            live_weights_norm > 0.0
            and determinism == 0.0
            and all(unchanged.values())
            and rho_change > 1e-4
            and y_change > 1e-3
        ),
    }


# ---------------------------------------------------------------------------
# D. initialisation + gradients (P0.3)
# ---------------------------------------------------------------------------


def _init_block(views: Mapping[str, Any], spec, seed: int = 0) -> dict[str, Any]:
    base, base_meta = M.build_base(spec, views["train_norm"], seed, "cpu", 0.10)
    model, meta = M.build_physchem6(
        spec, views["train_norm"], seed, "cpu", 0.10, physchem_norm=views["physchem_norm"]
    )
    valid_cache = views["valid_cache"]
    physchem_valid = np.asarray(views["physchem_valid"], dtype=np.float64)
    base.eval()
    model.eval()
    pred_base = M.predict(base, valid_cache, "cpu")
    pred_phys = M.predict(model, valid_cache, "cpu", physchem=physchem_valid)
    max_pred_diff = float(np.max(np.abs(pred_base - pred_phys)))
    max_pred_abs = float(np.max(np.abs(pred_base)))

    # gradients on the six new columns, and an actual update after one step
    model.train()
    rows = list(range(16))
    batch = M.to_device(M.collate(views["train_cache"], rows, physchem=views["physchem_train"]), "cpu")
    target = torch.full((len(rows),), -0.3, dtype=torch.float32)
    w = model.rho.lins[0].weight
    before = w.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(batch).view(-1), target)
    loss.backward()
    grad = w.grad.detach()
    n_old = before.shape[1] - C.N_PHYSCHEM
    new_grad_norm = float(grad[:, n_old:].norm())
    per_column_grad_norm = [float(grad[:, n_old + i].norm()) for i in range(C.N_PHYSCHEM)]
    grads_finite = bool(torch.isfinite(grad).all())
    optimizer.step()
    after = w.detach()
    new_delta = (after[:, n_old:] - before[:, n_old:]).abs()
    old_delta = (after[:, :n_old] - before[:, :n_old]).abs()

    return {
        "seed": int(seed),
        "parameters": int(meta["parameters"]),
        "parameters_base": int(meta["parameters_base"]),
        "parameters_new": int(meta["parameters_new"]),
        "parameters_new_expected": int(C.N_PHYSCHEM * 64),
        "shared_init": {k: v for k, v in meta["shared_init"].items() if k not in ("shared_keys_excluding_rho_first_layer",)},
        "init_max_abs_prediction_diff_vs_base": max_pred_diff,
        "init_max_abs_prediction_base": max_pred_abs,
        "init_predictions_n": int(valid_cache["n_molecules"]),
        "loss_finite": bool(np.isfinite(float(loss.detach()))),
        "new_column_grad_norm": new_grad_norm,
        "new_column_grad_norm_per_column": per_column_grad_norm,
        "all_grads_finite": grads_finite,
        "old_columns_changed_after_step": float(old_delta.max()),
        "new_columns_changed_after_step": float(new_delta.max()),
        "new_columns_changed_after_step_mean": float(new_delta.mean()),
        "dropout_p": float(meta["dropout"]),
    }


def _batch_order_block() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in C.SEEDS:
        replayed = C.batch_order_sha256(C.N_TRAIN, int(C.TRAIN_RECIPE["epochs"]), int(C.TRAIN_RECIPE["batch_size"]), seed)
        out[f"s{seed}"] = {
            "replayed_sha256": replayed,
            "v1_base_sha256": C.V1_BASE_BATCH_ORDER_SHA256[seed],
            "matches": bool(replayed == C.V1_BASE_BATCH_ORDER_SHA256[seed]),
        }
    return out


# ---------------------------------------------------------------------------
# E. capacity-matched S2-blind control (P2)
# ---------------------------------------------------------------------------


def _e_block(views: Mapping[str, Any], spec, seed: int = 0) -> dict[str, Any]:
    # The positive control uses the *trained* frozen BASE checkpoint (read-only):
    # it shows that the same perturbation moves the model whose input path we
    # are actually asking about, not just a random initialisation.
    base, base_meta = M.build_base(spec, views["train_norm"], seed, "cpu", 0.10)
    base_pin = C.v1_base_checkpoint_pins()[seed]
    M.load_checkpoint(
        base,
        base_pin["checkpoint"],
        expected_sha256=base_pin["checkpoint_sha256"],
        expected_family="base",
        expected_seed=seed,
    )
    model, meta = M.build_s2blind_e(spec, views["train_norm"], seed, "cpu", 0.10)
    capacity = meta["capacity"]
    valid_cache = views["valid_cache"]
    rows = list(range(int(valid_cache["n_molecules"])))
    batch = M.to_device(M.collate(valid_cache, rows), "cpu")

    # 1. restricted batch: typed-S2 access raises, forward must still succeed
    model.eval()
    with torch.no_grad():
        pred_full = model(batch).detach()
        pred_restricted = model(M.restricted_batch(batch)).detach()
    restricted_ok = bool(torch.isfinite(pred_restricted).all())
    restricted_identical = bool(torch.equal(pred_full, pred_restricted))

    # 2. changing typed-S2 content must not change E at all
    e_invariance: dict[str, Any] = {}
    for mode in ("code_shift", "closure_flip", "path_end_shift"):
        perturbed = M.to_device(M.perturb_typed_s2(batch, mode=mode), "cpu")
        with torch.no_grad():
            pred_perturbed = model(perturbed).detach()
        e_invariance[mode] = {
            "max_abs_change": float((pred_perturbed - pred_full).abs().max()),
            "bitwise_identical": bool(torch.equal(pred_perturbed, pred_full)),
        }

    # 3. the 200-wide S2 slot into the LocalMLP is exactly zero
    captured: dict[str, torch.Tensor] = {}
    handle = model.local.mlp.lins[0].register_forward_pre_hook(
        lambda module, inputs: captured.__setitem__("x", inputs[0].detach())  # noqa: ARG005
    )
    with torch.no_grad():
        model(batch)
    handle.remove()
    x = captured["x"]
    lo = int(capacity["dead_column_range"][0])
    hi = int(capacity["dead_column_range"][1])
    slot_max_abs = float(x[:, lo:hi].abs().max())

    # 4. dead columns receive exactly zero gradient; replacement does not
    model.train()
    train_rows = list(range(16))
    train_batch = M.to_device(M.collate(views["train_cache"], train_rows), "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(train_batch).view(-1), torch.full((len(train_rows),), -0.3))
    loss.backward()
    dead_grad_max = float(model.local.mlp.lins[0].weight.grad[:, lo:hi].abs().max())
    replacement_grads = {
        name: float(param.grad.norm()) if param.grad is not None else 0.0
        for name, param in model.local.replacement.named_parameters()
    }
    optimizer.step()

    # 5. positive control: the same perturbation *does* move BASE
    base.eval()
    base_sensitivity: dict[str, Any] = {}
    with torch.no_grad():
        base_pred = base(batch).detach()
        for mode in ("code_shift", "closure_flip", "path_end_shift"):
            perturbed = M.to_device(M.perturb_typed_s2(batch, mode=mode), "cpu")
            base_pert = base(perturbed).detach()
            change = (base_pert - base_pred).abs()
            base_sensitivity[mode] = {
                "max_abs_change": float(change.max()),
                "mean_abs_change": float(change.mean()),
                "n_molecules_changed_gt_1e-3": int((change > 1.0e-3).sum()),
                "n_molecules": int(change.numel()),
            }

    return {
        "seed": int(seed),
        "base_positive_control_arm": base_pin["arm"],
        "base_positive_control_checkpoint_sha256": base_pin["checkpoint_sha256"],
        "capacity": capacity,
        "shared_with_base": bool(meta["shared_with_base"]),
        "reference_base_matches_our_builder": bool(meta["reference_base_matches_our_builder"]),
        "restricted_batch": {
            "forward_finite": restricted_ok,
            "identical_to_full_batch": restricted_identical,
        },
        "invariance_to_typed_s2": e_invariance,
        "s2_slot_max_abs_activation": slot_max_abs,
        "dead_column_grad_max_abs": dead_grad_max,
        "replacement_parameter_grad_norms": replacement_grads,
        "base_sensitivity": base_sensitivity,
        "loss_finite": bool(np.isfinite(float(loss.detach()))),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(*, base_replay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    readonly = C.verify_readonly_inputs()
    views = D.build_views()
    spec = views["spec"]
    split_payload = V1C.read_json(C.SPLIT_MANIFEST)
    v1_data = V1C.read_json(C.V1_RUNS / "base_s0" / "train.json")["data"]

    alignment = _alignment_block(views, spec)

    physchem_model, physchem_meta = M.build_physchem6(
        spec, views["train_norm"], 0, "cpu", 0.10, physchem_norm=views["physchem_norm"]
    )
    standardisation = _standardisation_block(views, physchem_model, physchem_meta)
    init = _init_block(views, spec, seed=0)
    order = _batch_order_block()
    e_block = _e_block(views, spec, seed=0)

    split_matches = {
        "sizes": split_payload.get("sizes") == {"train": C.N_TRAIN, "valid": C.N_VALID, "test": C.N_TEST},
        "train_rows_sha256": C.rows_sha256(views["split"]["rows"]["train"]) == v1_data["train_rows_sha256"],
        "valid_rows_sha256": C.rows_sha256(views["split"]["rows"]["valid"]) == v1_data["valid_rows_sha256"],
        "test_rows_sha256_declared_only": True,
        "split_manifest_sha256": C.sha256_file(C.SPLIT_MANIFEST) == C.SPLIT_MANIFEST_SHA256,
        "cache_content_fingerprint": views["cache_built"]["content_fingerprint"]
        == v1_data["lcrc_cache_content_fingerprint"]
        == C.LCRC_CACHE_CONTENT_FINGERPRINT,
        "train_norm_fingerprint": views["train_norm_fingerprint"] == C.V1_TRAIN_NORM_FINGERPRINT,
        "target_norm_fit_on_train_only": views["target_norm"]["fit_on"] == "ESOL Train split only",
    }

    checks: dict[str, bool] = {}
    checks["readonly_inputs_pinned"] = bool(readonly["all_matched"])
    checks["split_identical_to_v1"] = all(bool(v) for v in split_matches.values())
    checks["base_replay_matches_v1"] = bool(base_replay is None or base_replay.get("passed") is True)
    checks["descriptor_rows_aligned_all_molecules"] = bool(
        alignment["n_mismatched"] == 0 and alignment["max_abs_diff_all_rows"] <= 1e-9
    )
    checks["descriptor_collate_guard_active"] = bool(alignment["collate_guard_rejects_full_matrix"])
    checks["duplicate_smiles_descriptors_consistent"] = not alignment["duplicate_smiles_descriptor_mismatches"]
    checks["single_standardisation_only"] = bool(standardisation["single_standardisation_passed"])
    checks["single_standardisation_test_is_discriminating"] = bool(
        standardisation["single_standardisation_is_discriminating"]
    )
    checks["descriptors_enter_rho_only"] = bool(standardisation["rho_only_passed"])
    checks["physchem_scaler_train_only"] = bool(
        standardisation["norm_fit_on"] == "ESOL Train split only" and standardisation["norm_valid_or_test_used"] is False
    )
    checks["physchem_shared_init_bitwise"] = bool(init["shared_init"]["all_shared_bitwise_identical"])
    checks["physchem_rho_first_layer_geometry"] = bool(
        init["shared_init"]["rho_old_columns_bitwise_identical"]
        and init["shared_init"]["rho_new_columns_zero"]
        and init["shared_init"]["rho_bias_bitwise_identical"]
    )
    checks["physchem_parameters_new_is_6x64"] = bool(init["parameters_new"] == init["parameters_new_expected"] == 384)
    checks["physchem_init_predictions_equal_base"] = bool(init["init_max_abs_prediction_diff_vs_base"] <= 1e-6)
    checks["physchem_new_columns_receive_gradient"] = bool(
        init["new_column_grad_norm"] > 0.0
        and all(value > 0.0 for value in init["new_column_grad_norm_per_column"])
        and init["all_grads_finite"]
    )
    checks["physchem_new_columns_update_after_step"] = bool(init["new_columns_changed_after_step"] > 0.0)
    checks["physchem_dropout_is_010"] = bool(float(init["dropout_p"]) == 0.10)
    checks["batch_order_paired_with_v1_base"] = all(bool(v["matches"]) for v in order.values())
    checks["e_capacity_matched_to_base"] = bool(e_block["capacity"]["passed"])
    checks["e_shared_init_matches_base"] = bool(
        e_block["shared_with_base"] and e_block["reference_base_matches_our_builder"]
    )
    checks["e_restricted_batch_forward"] = bool(
        e_block["restricted_batch"]["forward_finite"] and e_block["restricted_batch"]["identical_to_full_batch"]
    )
    checks["e_invariant_to_typed_s2"] = bool(
        all(v["bitwise_identical"] for v in e_block["invariance_to_typed_s2"].values())
    )
    checks["e_s2_slot_carries_no_information"] = bool(e_block["s2_slot_max_abs_activation"] == 0.0)
    checks["e_dead_columns_zero_gradient"] = bool(e_block["dead_column_grad_max_abs"] == 0.0)
    checks["e_replacement_receives_gradient"] = bool(
        all(value > 0.0 for value in e_block["replacement_parameter_grad_norms"].values())
    )
    checks["base_responds_to_typed_s2"] = bool(
        all(v["n_molecules_changed_gt_1e-3"] > 0 for v in e_block["base_sensitivity"].values())
        and all(v["max_abs_change"] > 1e-3 for v in e_block["base_sensitivity"].values())
    )

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": int(sum(1 for value in checks.values() if not value)),
        "failed_checks": [key for key, value in checks.items() if not value],
        "readonly_inputs": readonly,
        "split_and_norm": split_matches,
        "row_alignment": alignment,
        "single_standardisation": standardisation,
        "physchem_init_and_gradients": init,
        "batch_order": order,
        "capacity_matched_e": e_block,
        "base_replay": dict(base_replay) if base_replay is not None else None,
        "environment": C.env_info(),
        "scope_note": (
            "P0 only: the old round's full data/vocabulary audit is not repeated. "
            "No training, no GPU, no Test cache, no Test labels."
        ),
    }
    payload = _strip_nonjson(payload)
    C.write_json(C.P0_GATES, payload)
    return payload


def _strip_nonjson(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _strip_nonjson(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_nonjson(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


__all__ = ["run"]
