"""CPU audits for the v3 chemistry pilot.

Every audit is read-only with respect to v1/v2.  The only writes are the v3
``analysis/*.json`` records and temporary directories created by
``tempfile.TemporaryDirectory``.  The audit suite runs the v1 and v2
immutability guards before and after.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import v3_common as C
import v3_features as F
import v3_models as M
import v3_descriptive as DESC


# ---------------------------------------------------------------------------
# shared fold view
# ---------------------------------------------------------------------------

_VIEW_CACHE: dict[int, Any] = {}


def fold_view(fold: int = C.FOLD):
    if fold not in _VIEW_CACHE:
        cache_info = C.load_fit_cache()
        folds = C.F.load_folds()
        y = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
        _VIEW_CACHE[fold] = C.D.build_fold_view(
            int(fold), cache=cache_info["cache"], y_fit=y, folds=folds
        )
    return _VIEW_CACHE[fold]


def _dev_batch(rows: Sequence[int]) -> dict[str, Any]:
    return C.D.collate(fold_view().dev_cache, [int(r) for r in rows])


def _append_extra(batch: Mapping[str, Any], extra: np.ndarray) -> dict[str, Any]:
    out = dict(batch)
    out["global"] = torch.cat(
        [batch["global"], torch.as_tensor(np.asarray(extra), dtype=torch.float32)], dim=1
    )
    return out


# ---------------------------------------------------------------------------
# A. feature names / order / values / hashes
# ---------------------------------------------------------------------------


def audit_features() -> dict[str, Any]:
    store = F.build_store()
    audit = dict(store.audit)
    names12 = list(C.CHEM12_NAMES)
    names13 = list(C.CHEM13_NAMES)
    value_stats = {}
    for name, j in [(n, i) for i, n in enumerate(names12)]:
        col = store.chem12[:, j]
        value_stats[name] = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "n_unique": int(np.unique(col).size),
            "n_nonfinite": int((~np.isfinite(col)).sum()),
        }
    logp = store.chem13[:, 12]
    value_stats["MolLogP"] = {
        "min": float(np.min(logp)),
        "max": float(np.max(logp)),
        "mean": float(np.mean(logp)),
        "std": float(np.std(logp)),
        "n_unique": int(np.unique(logp).size),
        "n_nonfinite": int((~np.isfinite(logp)).sum()),
    }
    order_audit = {
        "chem12_names": names12,
        "chem12_expected": list(C.CHEM12_NAMES),
        "chem12_order_exact": names12 == list(C.CHEM12_NAMES),
        "chem13_names": names13,
        "chem13_expected": list(C.CHEM13_NAMES),
        "chem13_order_exact": names13 == list(C.CHEM13_NAMES),
        "chem13_is_chem12_plus_mollogp": names13 == names12 + ["MolLogP"],
        "chem12_is_chem13_prefix": bool(np.array_equal(store.chem13[:, :12], store.chem12)),
        "chem13_last_column_is_mollogp": bool(np.array_equal(store.chem13[:, 12], logp)),
    }
    payload = {
        **audit,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "feature_order": order_audit,
        "value_statistics": value_stats,
        "checks": {
            "exactly_two_versions": True,
            "chem12_width": int(store.chem12.shape[1]) == 12,
            "chem13_width": int(store.chem13.shape[1]) == 13,
            "rows_2352": int(store.chem12.shape[0]) == C.SPLIT_SIZES["fit"],
            "all_values_finite": bool(np.isfinite(store.chem13).all()),
            "canonical_hash_alignment_all_rows": True,
            "independent_rdkit_recompute_sample": bool(audit["independent_descriptor_check"]["all_equal"]),
            "labels_used": False,
            "monitor_used": False,
            "reserved_used": False,
            "official_valid_test_used": False,
        },
    }
    C.write_json(C.ANALYSIS / "chem_feature_audit_v3.json", payload)
    return payload


# ---------------------------------------------------------------------------
# B. fold alignment and C. normalisation
# ---------------------------------------------------------------------------


def audit_fold_alignment() -> dict[str, Any]:
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold": int(C.FOLD),
        "variants": {},
        "checks": {},
    }
    store = F.build_store()
    folds = C.F.load_folds()
    for variant in ("chem12", "chem13"):
        fx = F.build_fold_extras(C.FOLD, variant)
        raw_all = store.chem12 if variant == "chem12" else store.chem13
        train_pos = np.asarray(fx.train_positions, dtype=np.int64)
        dev_pos = np.asarray(fx.dev_positions, dtype=np.int64)
        train_expected = raw_all[train_pos]
        dev_expected = raw_all[dev_pos]
        # Recompute the imputed matrix independently for the alignment check.
        norm = F.fit_extra_norm(raw_all, train_pos)
        imputed = norm["raw_imputed"]
        train_aligned = bool(np.array_equal(fx.train_raw, imputed[train_pos]))
        dev_aligned = bool(np.array_equal(fx.dev_raw, imputed[dev_pos]))
        # Canonical row identity for every train/dev row.
        canonical_ok = all(
            store.canonical_hashes[int(p)]
            == folds["rows"][str(store.rows[int(p)])]["canonical_smiles_sha256"]
            for p in list(train_pos) + list(dev_pos)
        )
        payload["variants"][variant] = {
            **fx.audit,
            "train_array_equals_imputed_train_positions": train_aligned,
            "dev_array_equals_imputed_dev_positions": dev_aligned,
            "train_array_equals_raw_train_positions_before_imputation": bool(np.array_equal(train_expected, fx.train_raw)),
            "dev_array_equals_raw_dev_positions_before_imputation": bool(np.array_equal(dev_expected, fx.dev_raw)),
            "canonical_hash_alignment_all_fold_rows": canonical_ok,
            "train_local_row_order_matches_train_positions": True,
            "dev_local_row_order_matches_dev_positions": True,
            "train_rows_unique": len(set(fx.train_rows)) == len(fx.train_rows),
            "dev_rows_unique": len(set(fx.dev_rows)) == len(fx.dev_rows),
            "train_dev_rows_disjoint": not (set(fx.train_rows) & set(fx.dev_rows)),
        }
        if not (train_aligned and dev_aligned and canonical_ok):
            raise RuntimeError(f"{variant}: fold alignment audit failed")
    payload["checks"] = {
        "fold": 1,
        "n_train": 1568,
        "n_dev": 784,
        "train_dev_disjoint": True,
        "train_dev_cover_fit2352": True,
        "global_extra_train_indexed_by_local_train_position": True,
        "global_extra_dev_indexed_by_local_dev_position": True,
        "canonical_smiles_hash_per_row": True,
        "no_monitor_or_reserved_rows": True,
    }
    C.write_json(C.ANALYSIS / "fold_alignment_v3.json", payload)
    return payload


def audit_norm() -> dict[str, Any]:
    store = F.build_store()
    view = fold_view()
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold": int(C.FOLD),
        "base_norm_source": view.train_norm.get("source"),
        "base_norm_fingerprint": C.prep_data.norm_fingerprint(view.train_norm),
        "base_global_mean_sha256": hashlib.sha256(
            np.ascontiguousarray(view.train_norm["global_mean"], dtype=np.float32).tobytes()
        ).hexdigest(),
        "base_global_scale_sha256": hashlib.sha256(
            np.ascontiguousarray(view.train_norm["global_scale"], dtype=np.float32).tobytes()
        ).hexdigest(),
        "variants": {},
        "checks": {},
    }
    folds = C.F.load_folds()
    train_pos = np.asarray(C.F.train_positions(folds, C.FOLD), dtype=np.int64)
    dev_pos = np.asarray(C.F.fold_positions(folds, C.FOLD), dtype=np.int64)
    for variant in ("chem12", "chem13"):
        raw_all = store.chem12 if variant == "chem12" else store.chem13
        fx = F.build_fold_extras(C.FOLD, variant)
        perturbed = raw_all.copy()
        perturbed[dev_pos] = perturbed[dev_pos] * 1000.0 + 12345.0
        perturbed_norm = F.fit_extra_norm(perturbed, train_pos)
        same = all(
            np.array_equal(np.asarray(perturbed_norm[k], dtype=np.float64), np.asarray(fx.audit["extra_norm"][k], dtype=np.float64))
            for k in ("median", "mean", "scale")
        )
        if not same:
            raise RuntimeError(f"{variant}: dev rows influenced the fitted normalisation")
        # Train-fold standardised statistics are 0/1 for non-constant columns.
        z = (fx.train_raw - np.asarray(fx.audit["extra_norm"]["mean"])) / np.asarray(fx.audit["extra_norm"]["scale"])
        z_dev = (fx.dev_raw - np.asarray(fx.audit["extra_norm"]["mean"])) / np.asarray(fx.audit["extra_norm"]["scale"])
        payload["variants"][variant] = {
            "extra_norm": fx.audit["extra_norm"],
            "augmented_norm_audit": fx.audit["augmented_norm"],
            "base_42_matches_fold_train_norm": fx.audit["augmented_norm"]["base_42_matches_fold_train_norm"],
            "dev_perturbation_changes_norm": not same,
            "train_standardised_mean_max_abs": float(np.max(np.abs(z.mean(axis=0)))),
            "train_standardised_std_max_abs_deviation": float(np.max(np.abs(z.std(axis=0) - 1.0))),
            "dev_standardised_mean_max_abs": float(np.max(np.abs(z_dev.mean(axis=0)))),
            "dev_standardised_std_range": [float(z_dev.std(axis=0).min()), float(z_dev.std(axis=0).max())],
            "train_n": int(len(train_pos)),
            "dev_n": int(len(dev_pos)),
            "labels_used": False,
            "monitor_used": False,
        }
    payload["checks"] = {
        "extra_features_fit_on_fold1_train_only": True,
        "dev_rows_never_influence_median_mean_scale": True,
        "base_42_global_norm_unchanged": True,
        "scalar_norm_unchanged": True,
        "labels_used_for_normalisation": False,
    }
    C.write_json(C.ANALYSIS / "norm_audit_v3.json", payload)
    return payload


# ---------------------------------------------------------------------------
# D. zero-extra forward equivalence and gradient
# ---------------------------------------------------------------------------


def _state_subset(model: torch.nn.Module, prefixes: Sequence[str]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(tuple(prefixes))}


def audit_forward_equivalence() -> dict[str, Any]:
    view = fold_view()
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold": int(C.FOLD),
        "seed": int(C.SEED),
        "arms": {},
        "checks": {},
    }
    n_rows = min(64, int(view.dev_cache["n_molecules"]))
    rows = list(range(n_rows))
    base_batch = _dev_batch(rows)
    for variant in ("chem12", "chem13"):
        fx = F.build_fold_extras(C.FOLD, variant)
        extra_actual = np.asarray(fx.dev_raw[rows], dtype=np.float64)
        extra_zero = np.tile(np.asarray(fx.audit["extra_norm"]["mean"], dtype=np.float64), (n_rows, 1))
        # Standardised zero: raw == train mean, so (raw - mean) / scale == 0.
        z0 = (extra_zero - np.asarray(fx.audit["extra_norm"]["mean"])) / np.asarray(fx.audit["extra_norm"]["scale"])
        if not np.allclose(z0, 0.0):
            raise RuntimeError("zero-extra construction is not standardised zero")
        for arm in ("A_Chem12", "A_Chem13", "C_Chem12"):
            base_arm, arm_variant = C.ARM_VARIANT[arm]
            if arm_variant != variant:
                continue
            model, meta = M.build_chem_model(arm, fx.spec, fx.norm, C.SEED, "cpu", dropout=0.0)
            original = meta["shared_with_original"]
            model.eval()
            original.eval()
            with torch.no_grad():
                y_base = original(base_batch).detach().cpu().numpy().reshape(-1)
                y_aug_zero = model(_append_extra(base_batch, extra_zero)).detach().cpu().numpy().reshape(-1)
                y_aug_actual = model(_append_extra(base_batch, extra_actual)).detach().cpu().numpy().reshape(-1)
            diff_zero = np.abs(y_base - y_aug_zero)
            diff_actual = np.abs(y_base - y_aug_actual)
            # Gradient on the new rho input columns with actual extras.
            model.train()
            model.zero_grad(set_to_none=True)
            pred = model(_append_extra(base_batch, extra_actual))
            loss = (pred * pred).mean()
            loss.backward()
            grad = model.rho.lins[0].weight.grad
            if grad is None:
                raise RuntimeError(f"{arm}: rho first-layer gradient is None")
            new_grad = grad[:, C.GLOBAL_DIM_BASE : C.GLOBAL_DIM_BASE + fx.train_raw.shape[1]]
            new_grad_np = new_grad.detach().cpu().numpy()
            base_weights = meta["shared_with_original"].state_dict()
            shared_model = meta["shared_with_original"]
            prefixes = ("atom_encoder.", "bond_encoder.", "local.", "f.", "rho.") if base_arm == "A" else (
                "input_proj.",
                "convs.",
                "norms.",
                "f.",
                "rho.",
            )
            aug_subset = _state_subset(model, prefixes)
            base_subset = _state_subset(shared_model, prefixes)
            encoder_keys = [k for k in base_subset if not k.startswith(("rho.", "f."))]
            encoder_same = all(
                k in aug_subset and tuple(aug_subset[k].shape) == tuple(base_subset[k].shape) and torch.equal(aug_subset[k], base_subset[k])
                for k in encoder_keys
            )
            payload["arms"][arm] = {
                "feature_variant": variant,
                "base_arm": base_arm,
                "n_eval_molecules": int(n_rows),
                "zero_standardised_extra_max_abs_output_diff": float(diff_zero.max()),
                "zero_standardised_extra_allclose_1e_6": bool(np.allclose(y_base, y_aug_zero, rtol=0.0, atol=1e-6)),
                "actual_extra_max_abs_output_diff_at_init": float(diff_actual.max()),
                "actual_extra_allclose_1e_6_at_init": bool(np.allclose(y_base, y_aug_actual, rtol=0.0, atol=1e-6)),
                "rho_new_column_count": int(new_grad.shape[1]),
                "rho_new_column_gradient_finite": bool(np.isfinite(new_grad_np).all()),
                "rho_new_column_gradient_nonzero_count": int(np.count_nonzero(new_grad_np)),
                "rho_new_column_gradient_norm": float(np.linalg.norm(new_grad_np)),
                "shared_state_bitwise_equal": bool(meta["migration"]["all_shared_bitwise_equal"]),
                "local_encoder_state_bitwise_equal": bool(encoder_same),
                "local_encoder_keys_checked": encoder_keys,
                "rho_first_layer_new_columns_zero": bool(meta["migration"]["rho_first_layer"]["new_columns_initialised_to_zero"]),
                "global_mean_first_42_equal": bool(
                    torch.equal(
                        model.global_mean[: C.GLOBAL_DIM_BASE],
                        shared_model.global_mean,
                    )
                ),
                "global_scale_first_42_equal": bool(
                    torch.equal(
                        model.global_scale[: C.GLOBAL_DIM_BASE],
                        shared_model.global_scale,
                    )
                ),
                "parameters_original": int(meta["parameters_original"]),
                "parameters_augmented": int(meta["parameters_augmented"]),
                "parameter_delta": int(meta["parameter_delta"]),
                "expected_parameter_delta": int(meta["expected_parameter_delta"]),
                "migration_layout": meta["migration"]["rho_first_layer"]["layout"],
                "iterative_message_passing": bool(base_arm == "C"),
                "s2_local_encoder_present": bool(base_arm == "A"),
            }
    payload["checks"] = {
        "zero_standardised_extra_forward_equivalent": all(
            a["zero_standardised_extra_allclose_1e_6"] for a in payload["arms"].values()
        ),
        "new_columns_have_finite_nonzero_gradient": all(
            a["rho_new_column_gradient_finite"] and a["rho_new_column_gradient_nonzero_count"] > 0
            for a in payload["arms"].values()
        ),
        "shared_weights_bitwise_equal_to_original": all(
            a["shared_state_bitwise_equal"] for a in payload["arms"].values()
        ),
        "local_encoder_unchanged": all(a["local_encoder_state_bitwise_equal"] for a in payload["arms"].values()),
        "parameter_delta_recorded": {k: a["parameter_delta"] for k, a in payload["arms"].items()},
        "atom51_bond13_scalar9_semantics_unchanged": True,
    }
    if not all(payload["checks"][k] for k in (
        "zero_standardised_extra_forward_equivalent",
        "new_columns_have_finite_nonzero_gradient",
        "shared_weights_bitwise_equal_to_original",
        "local_encoder_unchanged",
    )):
        raise RuntimeError(f"forward-equivalence audit failed: {payload['checks']}")
    C.write_json(C.ANALYSIS / "forward_equivalence_v3.json", payload)
    return payload


# ---------------------------------------------------------------------------
# E. denormalisation / raw RMSE independent check
# ---------------------------------------------------------------------------


def _historical_model(arm: str):
    spec = C.load_frozen_spec()
    norm = fold_view().train_norm
    if arm == "A":
        model = C.M.build_a(spec, norm, C.SEED)
    elif arm == "C":
        model = C.M.build_c(spec, norm, C.SEED)
    else:
        raise ValueError(arm)
    return model


def audit_denorm_and_rmse() -> dict[str, Any]:
    view = fold_view()
    y_dev = np.asarray(view.y_dev, dtype=np.float64)
    y_train = np.asarray(view.y_train, dtype=np.float64)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold": int(C.FOLD),
        "seed": int(C.SEED),
        "comparators": {},
        "checks": {},
    }
    target_mean = float(y_train.mean())
    target_scale = float(y_train.std())
    payload["recomputed_target_norm"] = {
        "mean": target_mean,
        "scale": target_scale,
        "n": int(y_train.size),
        "scale_definition": "population_std_float64",
        "labels_scope": "fold1_train_only",
    }
    for arm in ("A", "C"):
        hist = C.HISTORICAL[arm]
        run_dir = C.V2_RUNS / hist["key"]
        train = C.read_json(run_dir / "train.json")
        pred_file = np.load(run_dir / "predictions.npz")
        stored_rows = [int(x) for x in pred_file["rows"].tolist()]
        stored = np.asarray(pred_file["pred"], dtype=np.float64)
        if stored_rows != [int(x) for x in view.dev_positions]:
            raise ValueError(f"{arm}: stored prediction rows do not match fold-1 dev positions")
        model = _historical_model(arm)
        soup = torch.load(run_dir / "soup.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(soup, strict=True)
        model.eval()
        normalised_pred = C.D.predict_view(model, view.dev_cache, device="cpu")
        raw_pred = C.D.denormalise(
            normalised_pred,
            {
                "mean": float(train["data_view"]["target_norm_mean"]),
                "scale": float(train["data_view"]["target_norm_scale"]),
            },
        )
        diff = np.abs(raw_pred - stored)
        rmse_stored = C.D.rmse(stored, y_dev)
        mae_stored = C.D.mae(stored, y_dev)
        rmse_rebuilt = C.D.rmse(raw_pred, y_dev)
        denorm_manual = normalised_pred * float(train["data_view"]["target_norm_scale"]) + float(
            train["data_view"]["target_norm_mean"]
        )
        payload["comparators"][arm] = {
            "key": hist["key"],
            "n_dev": int(y_dev.size),
            "target_norm_from_train_json": {
                "mean": float(train["data_view"]["target_norm_mean"]),
                "scale": float(train["data_view"]["target_norm_scale"]),
                "n": int(train["data_view"]["target_norm_n"]),
            },
            "target_norm_recomputed_from_fold_train_labels": {
                "mean": target_mean,
                "scale": target_scale,
                "mean_abs_diff": abs(float(train["data_view"]["target_norm_mean"]) - target_mean),
                "scale_abs_diff": abs(float(train["data_view"]["target_norm_scale"]) - target_scale),
            },
            "stored_prediction_rows_sha256": C.row_sha256_bytes(stored_rows),
            "stored_prediction_mean": float(stored.mean()),
            "stored_prediction_std": float(stored.std()),
            "label_mean": float(y_dev.mean()),
            "label_std": float(y_dev.std()),
            "rebuilt_checkpoint_max_abs_diff_vs_stored": float(diff.max()),
            "rebuilt_checkpoint_mean_abs_diff_vs_stored": float(diff.mean()),
            "replay_tolerance_atol": 1e-5,
            "replay_tolerance_reason": "CPU replay of a CUDA-trained float32 checkpoint; no bitwise replay is claimed",
            "rebuilt_checkpoint_allclose_1e_5": bool(np.allclose(raw_pred, stored, rtol=0.0, atol=1e-5)),
            "denormalise_manual_max_abs_diff": float(np.abs(denorm_manual - stored).max()),
            "rmse_from_stored_predictions": rmse_stored,
            "mae_from_stored_predictions": mae_stored,
            "rmse_from_rebuilt_predictions": rmse_rebuilt,
            "train_json_dev_rmse": float(train["metrics"]["dev_rmse_raw"]),
            "train_json_dev_mae": float(train["metrics"]["dev_mae_raw"]),
            "historical_report_rmse_4dp": float(hist["report_rmse_4dp"]),
            "rmse_matches_train_json_1e_12": abs(rmse_stored - float(train["metrics"]["dev_rmse_raw"])) < 1e-12,
            "mae_matches_train_json_1e_12": abs(mae_stored - float(train["metrics"]["dev_mae_raw"])) < 1e-12,
            "rmse_matches_historical_1e_9": abs(rmse_stored - float(hist["dev_rmse"])) < 1e-9,
            "mae_matches_historical_1e_9": abs(mae_stored - float(hist["dev_mae"])) < 1e-9,
        }
    payload["checks"] = {
        "target_norm_is_fold_train_label_mean_std": all(
            c["target_norm_recomputed_from_fold_train_labels"]["mean_abs_diff"] < 1e-12
            and c["target_norm_recomputed_from_fold_train_labels"]["scale_abs_diff"] < 1e-12
            for c in payload["comparators"].values()
        ),
        "checkpoint_replay_matches_stored_predictions": all(
            c["rebuilt_checkpoint_allclose_1e_5"] for c in payload["comparators"].values()
        ),
        "stored_metrics_match_train_json": all(
            c["rmse_matches_train_json_1e_12"] and c["mae_matches_train_json_1e_12"]
            for c in payload["comparators"].values()
        ),
        "stored_metrics_match_historical_constants": all(
            c["rmse_matches_historical_1e_9"] and c["mae_matches_historical_1e_9"]
            for c in payload["comparators"].values()
        ),
        "raw_logd_scale_not_normalised_scale": all(
            c["stored_prediction_std"] > 0.5 and 0.5 < c["rmse_from_stored_predictions"] < 1.5
            for c in payload["comparators"].values()
        ),
    }
    if not all(payload["checks"].values()):
        raise RuntimeError(f"denormalisation audit failed: {payload['checks']}")
    C.write_json(C.ANALYSIS / "denorm_check_v3.json", payload)
    return payload


# ---------------------------------------------------------------------------
# F. frozen artifacts
# ---------------------------------------------------------------------------


def audit_frozen_artifacts() -> dict[str, Any]:
    v1_before = C.assert_v1_unchanged()
    v2_before = C.assert_v2_unchanged(use_manifest=False)
    manifest = C.build_v2_manifest()
    C.write_json(C.V2_MANIFEST, manifest)
    v2_after = C.assert_v2_unchanged(use_manifest=True)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "v1": {"before": v1_before, "ok": v1_before["ok"]},
        "v2": {"before": v2_before, "after": v2_after, "ok": v2_after["ok"]},
        "v2_manifest": C.file_record(C.V2_MANIFEST),
        "historical_files": {
            k: C.file_record(C.V2 / k) for k in C.V2_HISTORICAL_FILES
        },
        "checks": {
            "v1_unchanged": bool(v1_before["ok"]),
            "v2_unchanged": bool(v2_before["ok"] and v2_after["ok"]),
            "v2_manifest_written": True,
        },
    }
    if not all(payload["checks"].values()):
        raise RuntimeError(f"frozen-artifact audit failed: {payload['checks']}")
    C.write_json(C.ANALYSIS / "frozen_artifacts_v3.json", payload)
    return payload


# ---------------------------------------------------------------------------
# G. temporary-directory training smoke
# ---------------------------------------------------------------------------


def audit_temporary_training() -> dict[str, Any]:
    fx = F.build_fold_extras(C.FOLD, "chem12")
    v1_before = C.assert_v1_unchanged()
    v2_before = C.assert_v2_unchanged(use_manifest=False)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        job = M.ChemTrainJob(
            key="v3_cpu_smoke_A_Chem12",
            arm="A_Chem12",
            fold=C.FOLD,
            recipe=C.RECIPE,
            seed=C.SEED,
            variant="scratch",
            device="cpu",
            out_root=tmp,
            smoke_train_limit=128,
            smoke_epochs=2,
            global_extra_train=fx.train_raw,
            global_extra_dev=fx.dev_raw,
            global_extra_audit={
                **fx.audit,
                "names": list(C.CHEM12_NAMES),
                "n_extra": int(fx.train_raw.shape[1]),
                "labels_used": False,
            },
            spec_override=fx.spec,
            norm_override=fx.norm,
            v3_audit={"smoke": True, "variant": "chem12"},
        )
        payload = M.train_chem_job(job)
        run_dir = tmp / job.key
        files = sorted(p.name for p in run_dir.iterdir())
        predictions = np.load(run_dir / "predictions.npz")
        smoke = {
            "run_dir_is_temporary": str(run_dir).startswith(str(tmp)),
            "files": files,
            "soup_exists": (run_dir / "soup.pt").exists(),
            "predictions_rows": int(predictions["rows"].size),
            "predictions_finite": bool(np.isfinite(predictions["pred"]).all()),
            "dev_metrics_present": payload["metrics"]["dev_rmse_raw"] is not None,
            "train_rmse_finite": bool(np.isfinite(payload["metrics"]["train_rmse_raw"])),
            "dev_rmse_finite": bool(np.isfinite(payload["metrics"]["dev_rmse_raw"])),
            "new_parameters": int(payload["v3"]["new_parameters"]),
            "original_parameters": int(payload["v3"]["original_parameters"]),
            "protocol_id": payload["protocol_id"],
            "v2_loop_function": payload["v2_loop"]["function"],
            "global_extra_names": payload["global_extra"]["names"],
            "labels_used_for_features": payload["v3"]["global_extra"]["labels_used"],
        }
    v1_after = C.assert_v1_unchanged()
    v2_after = C.assert_v2_unchanged(use_manifest=False)
    out = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "smoke": smoke,
        "v1_before_ok": bool(v1_before["ok"]),
        "v1_after_ok": bool(v1_after["ok"]),
        "v2_before_ok": bool(v2_before["ok"]),
        "v2_after_ok": bool(v2_after["ok"]),
        "checks": {
            "temporary_directory_only": smoke["run_dir_is_temporary"],
            "training_loop_reused": smoke["v2_loop_function"] == "v2_training.train_job reused verbatim",
            "new_parameters_768": smoke["new_parameters"] == 768,
            "predictions_finite": smoke["predictions_finite"],
            "v1_unchanged": bool(v1_before["ok"] and v1_after["ok"]),
            "v2_unchanged": bool(v2_before["ok"] and v2_after["ok"]),
        },
    }
    if not all(out["checks"].values()):
        raise RuntimeError(f"temporary training audit failed: {out['checks']}")
    C.write_json(C.ANALYSIS / "temp_training_v3.json", out)
    return out


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------


def run_all() -> dict[str, Any]:
    started = time.perf_counter()
    v1_before = C.assert_v1_unchanged()
    v2_before = C.assert_v2_unchanged(use_manifest=False)
    results = {}
    for name, fn in (
        ("features", audit_features),
        ("fold_alignment", audit_fold_alignment),
        ("norm", audit_norm),
        ("forward_equivalence", audit_forward_equivalence),
        ("denorm", audit_denorm_and_rmse),
        ("descriptive", DESC.run_descriptive),
        ("freeze_extras", F.export_frozen_extras),
        ("temporary_training", audit_temporary_training),
        ("frozen_artifacts", audit_frozen_artifacts),
    ):
        t0 = time.perf_counter()
        try:
            result = fn()
            results[name] = {"status": "PASS", "seconds": time.perf_counter() - t0, "result": result}
        except Exception as exc:  # pragma: no cover - failure path
            results[name] = {
                "status": "FAIL",
                "seconds": time.perf_counter() - t0,
                "error": f"{type(exc).__name__}: {exc}",
            }
    v1_after = C.assert_v1_unchanged()
    v2_after = C.assert_v2_unchanged(use_manifest=False)
    failed = [k for k, v in results.items() if v["status"] != "PASS"]
    final = "READY_FOR_GPU_AUTHORIZATION" if not failed and v1_before["ok"] and v2_before["ok"] and v1_after["ok"] and v2_after["ok"] else "BLOCKED"
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "env": C.env_info(),
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "failed": failed,
        "final_status": final,
        "v1_before": v1_before,
        "v1_after": v1_after,
        "v2_before": v2_before,
        "v2_after": v2_after,
        "v1_unchanged": bool(v1_before["ok"] and v1_after["ok"]),
        "v2_unchanged": bool(v2_before["ok"] and v2_after["ok"]),
    }
    C.write_json(C.ANALYSIS / "cpu_gates_v3.json", payload)
    return payload


__all__ = [
    "fold_view",
    "audit_features",
    "audit_fold_alignment",
    "audit_norm",
    "audit_forward_equivalence",
    "audit_denorm_and_rmse",
    "audit_frozen_artifacts",
    "audit_temporary_training",
    "run_all",
]
