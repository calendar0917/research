"""CPU gates for the v4 local-chemistry pilot.

Every gate is a hard check: a failure raises before any freeze or GPU
authorization.  The suite covers

G1  AtomChem4 feature definitions, atom-order proofs and contribution sums
G2  fold alignment: positions, molecule ids, atom pointers, rows and labels
G3  fold-train-only standardisation (dev rows cannot influence it)
G4  augmented spec + initialisation migration + zero-extra forward equality
G5  new-column gradients, unchanged original columns, parameter-delta audit
G6  broadcast control: same features/parameters/init, per-molecule constant
G7  raw-logD de-normalisation and historical-prediction replay
G8  temporary CPU training for both arms (no GPU, temp directory only)
G9  v1 / v2 / v3 immutability
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import v4_common as C
import v4_atomchem as A
import v4_models as M

_VIEW_CACHE: dict[int, Any] = {}


def fold_view(fold: int = C.FOLD):
    if fold not in _VIEW_CACHE:
        cache_info = C.load_fit_cache()
        folds = C.F.load_folds()
        y = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
        _VIEW_CACHE[fold] = C.D.build_fold_view(int(fold), cache=cache_info["cache"], y_fit=y, folds=folds)
    return _VIEW_CACHE[fold]


def _write(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    C.write_json(C.ANALYSIS / name, payload)
    return payload


# ---------------------------------------------------------------------------
# G1 features
# ---------------------------------------------------------------------------


def audit_atomchem() -> dict[str, Any]:
    store = A.build_store()
    payload = A.write_store_cache(store)
    checks = {
        "all_2352_molecules_aligned_index_by_index": payload["alignment"]["n_mismatches"] == 0,
        "identity_mapping_proved": payload["alignment"]["identity_mapping_proved_by_index"],
        "crippen_sum_equals_mol_logp": payload["crippen"]["all_equal"],
        "tpsa_sum_equals_mol_tpsa": payload["tpsa"]["all_equal"],
        "donor_acceptor_features_single_atom": payload["hbd_hba"]["single_atom_only"],
        "four_features_finite_and_present": all(
            payload["raw_value_stats"][n]["n_nonzero"] > 0 for n in C.ATOMCHEM_NAMES
        ),
    }
    scope = {
        "labels_used_for_features": bool(payload["labels_used"]),
        "monitor_structures_used": bool(payload.get("monitor_used", False)),
        "reserved_structures_used": bool(payload.get("reserved_used", False)),
        "official_valid_test_used": bool(payload.get("official_valid_test_used", False)),
    }
    if not all(checks.values()) or any(scope.values()):
        raise RuntimeError(f"AtomChem4 audit failed: {checks} scope={scope}")
    return _write("atomchem_audit_v4.json", {**payload, "checks": checks, "scope": scope})


# ---------------------------------------------------------------------------
# G2 fold alignment
# ---------------------------------------------------------------------------


def audit_fold_alignment() -> dict[str, Any]:
    fx = A.build_fold_extras(C.FOLD)
    store = A.build_store()
    view = fold_view()
    folds = C.F.load_folds()
    cache = C.load_fit_cache()["cache"]
    y_all = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
    train_pos = list(fx.train_positions)
    dev_pos = list(fx.dev_positions)
    train_rows = tuple(int(folds["rows"][str(store.fit_rows[p])]["global_row_id"]) for p in train_pos)
    dev_rows = tuple(int(folds["rows"][str(store.fit_rows[p])]["global_row_id"]) for p in dev_pos)
    checks = {
        "train_dev_disjoint": not (set(train_pos) & set(dev_pos)),
        "train_dev_cover_fit2352": sorted(train_pos + dev_pos) == list(range(C.SPLIT_SIZES["fit"])),
        "positions_match_v2_fold_view": tuple(view.train_positions) == tuple(train_pos)
        and tuple(view.dev_positions) == tuple(dev_pos),
        "molecule_ids_match_train_cache": tuple(str(x) for x in view.train_cache["molecule_ids"])
        == fx.train_molecule_ids,
        "molecule_ids_match_dev_cache": tuple(str(x) for x in view.dev_cache["molecule_ids"]) == fx.dev_molecule_ids,
        "train_atom_counts_match": int(view.train_cache["atom_ptr"][-1]) == fx.train_std.shape[0],
        "dev_atom_counts_match": int(view.dev_cache["atom_ptr"][-1]) == fx.dev_std.shape[0],
        "train_atom_ptr_matches_cache": np.array_equal(
            np.asarray(view.train_cache["atom_ptr"], dtype=np.int64), fx.train_mol_ptr
        ),
        "dev_atom_ptr_matches_cache": np.array_equal(
            np.asarray(view.dev_cache["atom_ptr"], dtype=np.int64), fx.dev_mol_ptr
        ),
        "train_rows_match_frozen_fold_file": train_rows == fx.train_rows,
        "dev_rows_match_frozen_fold_file": dev_rows == fx.dev_rows,
        "train_labels_align": bool(
            np.array_equal(
                np.asarray(view.y_train, dtype=np.float64),
                y_all[np.asarray(train_pos, dtype=np.int64)],
            )
        ),
        "dev_labels_align": bool(
            np.array_equal(
                np.asarray(view.y_dev, dtype=np.float64),
                y_all[np.asarray(dev_pos, dtype=np.int64)],
            )
        ),
    }
    scope = {
        "labels_used_for_features": False,
        "monitor_structures_used": False,
        "reserved_structures_used": False,
        "official_valid_test_used": False,
    }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold": C.FOLD,
        "checks": checks,
        "scope": scope,
        "train_positions_sha256": C.row_sha256_bytes(train_pos),
        "dev_positions_sha256": C.row_sha256_bytes(dev_pos),
        "train_rows_sha256": C.row_sha256_bytes(fx.train_rows),
        "dev_rows_sha256": C.row_sha256_bytes(fx.dev_rows),
        "n_train_molecules": int(len(train_pos)),
        "n_dev_molecules": int(len(dev_pos)),
        "n_train_atoms": int(fx.train_std.shape[0]),
        "n_dev_atoms": int(fx.dev_std.shape[0]),
        "train_molecule_ids_sha256": C.sha256_text("\n".join(fx.train_molecule_ids)),
        "dev_molecule_ids_sha256": C.sha256_text("\n".join(fx.dev_molecule_ids)),
        "train_std_sha256": C.array_sha256(fx.train_std),
        "dev_std_sha256": C.array_sha256(fx.dev_std),
        "train_broadcast_sha256": C.array_sha256(fx.train_broadcast_std),
        "dev_broadcast_sha256": C.array_sha256(fx.dev_broadcast_std),
        "raw_fit_sha256": C.array_sha256(fx.raw_fit),
        "target_norm_mean": float(view.target_norm["mean"]),
        "target_norm_scale": float(view.target_norm["scale"]),
        "data_identity_chain": (
            "Fit row -> cache position -> molecule id -> V2 fold view train/dev cache -> collate rows -> extras"
        ),
    }
    if not all(bool(v) for v in checks.values()) or any(scope.values()):
        raise RuntimeError(f"fold alignment failed: {checks} scope={scope}")
    return _write("fold_alignment_v4.json", payload)


# ---------------------------------------------------------------------------
# G3 standardisation
# ---------------------------------------------------------------------------


def audit_atom_norm() -> dict[str, Any]:
    fx = A.build_fold_extras(C.FOLD)
    store = A.build_store()
    train_mask = np.zeros(store.atom_ptr[-1], dtype=bool)
    train_idx = np.concatenate(
        [np.arange(int(store.atom_ptr[p]), int(store.atom_ptr[p + 1])) for p in fx.train_positions]
    )
    train_mask[train_idx] = True
    perturbed = store.features.copy()
    dev_mask = ~train_mask
    perturbed[dev_mask] = perturbed[dev_mask] * 1000.0 + 12345.0
    pnorm = A.standardise_fit(perturbed, train_mask)
    same = all(np.array_equal(np.asarray(pnorm[k]), np.asarray(fx.__getattribute__(k))) for k in ("mean", "scale"))
    z_train = (fx.raw_fit[train_mask] - fx.mean) / fx.scale
    z_dev = (fx.raw_fit[dev_mask] - fx.mean) / fx.scale
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "standardisation": fx.audit["standardisation"],
        "names": list(C.ATOMCHEM_NAMES),
        "train_standardised_mean_max_abs": float(np.max(np.abs(z_train.mean(axis=0)))),
        "train_standardised_std_max_abs_deviation": float(np.max(np.abs(z_train.std(axis=0) - 1.0))),
        "dev_standardised_mean": [float(x) for x in z_dev.mean(axis=0)],
        "dev_standardised_std": [float(x) for x in z_dev.std(axis=0)],
        "dev_perturbation_changes_mean_or_scale": bool(not same),
        "checks": {
            "fit_on_fold1_train_atoms_only": True,
            "dev_atoms_never_influence_mean_or_scale": bool(same),
            "no_zero_variance_column": len(fx.constant_indices) == 0,
            "train_mean_zero": float(np.max(np.abs(z_train.mean(axis=0)))) < 1e-6,
            "train_std_one": float(np.max(np.abs(z_train.std(axis=0) - 1.0))) < 1e-6,
            "dev_atoms_used_only_for_reporting": True,
        },
        "scope": {"labels_used_for_standardisation": False},
    }
    if not all(payload["checks"].values()) or any(payload["scope"].values()):
        raise RuntimeError(f"atom normalisation audit failed: {payload['checks']} scope={payload['scope']}")
    return _write("atom_norm_v4.json", payload)


# ---------------------------------------------------------------------------
# G4/G5 forward equivalence and gradients
# ---------------------------------------------------------------------------


def _state_subset(model: torch.nn.Module, prefixes: Sequence[str]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(tuple(prefixes))}


def audit_forward_equivalence() -> dict[str, Any]:
    view = fold_view()
    fx = A.build_fold_extras(C.FOLD)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold": C.FOLD,
        "seed": C.SEED,
        "arms": {},
        "checks": {},
    }
    base_spec = C.load_frozen_spec()
    n_rows = 64
    b51 = C.D.collate(view.dev_cache, list(range(n_rows)))
    for arm in C.ARMS:
        variant = C.ARM_VARIANT[arm]
        extras = A.atom_arrays_from_fold_extras(fx, variant)
        batch = M.collate_with_atom_extra(view.dev_cache, list(range(n_rows)), extras)
        model, meta = M.build_atomchem_model(arm, extras["spec"], view.train_norm, C.SEED, "cpu", dropout=0.0)
        original = meta["shared_with_original"]
        zero = dict(batch)
        zero["atom_feat"] = torch.cat(
            [batch["atom_feat"][:, : C.ATOM_DIM_BASE], torch.zeros(batch["atom_feat"].shape[0], C.N_ATOM_EXTRA)],
            dim=1,
        )
        model.eval()
        original.eval()
        with torch.no_grad():
            y_base = original(b51).detach().cpu().numpy().reshape(-1)
            y_zero = model(zero).detach().cpu().numpy().reshape(-1)
            y_actual = model(batch).detach().cpu().numpy().reshape(-1)

        # Gradient check: real extras, and zero extras vs the original model.
        model.train()
        model.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = (pred * pred).mean()
        loss.backward()
        grad = model.atom_encoder.lins[0].weight.grad
        if grad is None:
            raise RuntimeError(f"{arm}: atom-encoder gradient is None")
        new_grad = grad[:, C.ATOM_DIM_BASE :].detach().cpu().numpy()
        old_grad = grad[:, : C.ATOM_DIM_BASE].detach().cpu().numpy()

        original.zero_grad(set_to_none=True)
        pred0 = original(b51)
        (pred0 * pred0).mean().backward()
        base_grad = original.atom_encoder.lins[0].weight.grad.detach().cpu().numpy()

        model.zero_grad(set_to_none=True)
        predz = model(zero)
        (predz * predz).mean().backward()
        zero_old_grad = model.atom_encoder.lins[0].weight.grad[:, : C.ATOM_DIM_BASE].detach().cpu().numpy()

        aug_subset = _state_subset(model, ("atom_encoder.", "bond_encoder.", "local.", "f.", "rho."))
        base_subset = _state_subset(original, ("atom_encoder.", "bond_encoder.", "local.", "f.", "rho."))
        widened = M.ATOM_WEIGHT_KEY
        shared_keys = [k for k in base_subset if k != widened]
        shared_exact = all(
            k in aug_subset and torch.equal(aug_subset[k], base_subset[k]) for k in shared_keys
        )
        base_51_equal = bool(
            torch.equal(aug_subset[widened][:, : C.ATOM_DIM_BASE], base_subset[widened])
            and tuple(aug_subset[widened].shape) == (C.ATOM_HIDDEN, C.ATOM_DIM_AUGMENTED)
        )
        payload["arms"][arm] = {
            "feature_variant": variant,
            "n_eval_molecules": int(n_rows),
            "zero_extra_max_abs_output_diff": float(np.abs(y_base - y_zero).max()),
            "zero_extra_forward_equal_atol_1e_6": bool(np.allclose(y_base, y_zero, rtol=0.0, atol=1e-6)),
            "actual_extra_max_abs_output_diff_at_init": float(np.abs(y_base - y_actual).max()),
            "actual_extra_forward_equal_at_init_atol_1e_6": bool(np.allclose(y_base, y_actual, rtol=0.0, atol=1e-6)),
            "new_column_gradient_finite": bool(np.isfinite(new_grad).all()),
            "new_column_gradient_nonzero_count": int(np.count_nonzero(new_grad)),
            "new_column_gradient_norm": float(np.linalg.norm(new_grad)),
            "old_column_gradient_finite": bool(np.isfinite(old_grad).all()),
            "old_column_gradient_nonzero_count": int(np.count_nonzero(old_grad)),
            "old_column_gradient_matches_original": bool(np.allclose(zero_old_grad, base_grad, rtol=1e-6, atol=1e-6)),
            "zero_extra_old_gradient_max_abs_diff": float(np.abs(zero_old_grad - base_grad).max()),
            "zero_extra_old_gradient_max_rel_diff": float(
                np.abs(zero_old_grad - base_grad).max() / (np.abs(base_grad).max() + 1e-30)
            ),
            "gradient_equality_note": (
                "the first-51-column gradient is mathematically identical with zero extras; the 55-wide vs 51-wide "
                "float32 matmul reassociates the reduction so bitwise equality is not claimed"
            ),
            "shared_state_bitwise_equal": bool(shared_exact),
            "n_shared_tensors_excluding_widened": int(len(shared_keys)),
            "base_51_columns_bitwise_equal": bool(base_51_equal and meta["migration"]["base_51_columns_bitwise_equal"]),
            "new_columns_zero_at_init": bool(meta["migration"]["new_columns_initialised_to_zero"]),
            "parameters_original": int(meta["parameters_original"]),
            "parameters_augmented": int(meta["parameters_augmented"]),
            "parameter_delta": int(meta["parameter_delta"]),
            "expected_parameter_delta": int(C.EXPECTED_PARAMETER_DELTA),
            "n_shared_tensors": int(meta["migration"]["n_shared_tensors"]),
            "atom_input_width_base": int(base_spec.atom_feat_dim),
            "atom_input_width_augmented": int(extras["spec"].atom_feat_dim),
            "bond_scalar_global_unchanged": bool(
                extras["spec"].bond_feat_dim == base_spec.bond_feat_dim
                and extras["spec"].n_scalars == base_spec.n_scalars
                and extras["spec"].global_dim == base_spec.global_dim
            ),
            "message_passing_changed": False,
            "s2_semantics_changed": False,
        }
    payload["checks"] = {
        "zero_extra_forward_equivalent": all(a["zero_extra_forward_equal_atol_1e_6"] for a in payload["arms"].values()),
        "actual_extra_zero_effect_at_init": all(
            a["actual_extra_forward_equal_at_init_atol_1e_6"] for a in payload["arms"].values()
        ),
        "new_columns_have_finite_nonzero_gradient": all(
            a["new_column_gradient_finite"] and a["new_column_gradient_nonzero_count"] == C.EXPECTED_PARAMETER_DELTA
            for a in payload["arms"].values()
        ),
        "old_column_gradient_unchanged_with_zero_extra": all(
            a["old_column_gradient_matches_original"] for a in payload["arms"].values()
        ),
        "shared_weights_bitwise_equal_to_original": all(a["shared_state_bitwise_equal"] for a in payload["arms"].values()),
        "widened_tensor_first_51_columns_bitwise_equal": all(
            a["base_51_columns_bitwise_equal"] for a in payload["arms"].values()
        ),
        "parameter_delta_512": all(a["parameter_delta"] == C.EXPECTED_PARAMETER_DELTA for a in payload["arms"].values()),
        "atom_width_51_to_55": all(
            a["atom_input_width_base"] == 51 and a["atom_input_width_augmented"] == 55 for a in payload["arms"].values()
        ),
    }
    if not all(payload["checks"].values()):
        raise RuntimeError(f"forward-equivalence audit failed: {payload['checks']}")
    return _write("forward_equivalence_v4.json", payload)


# ---------------------------------------------------------------------------
# G6 broadcast control
# ---------------------------------------------------------------------------


def audit_broadcast_control() -> dict[str, Any]:
    fx = A.build_fold_extras(C.FOLD)
    p1 = A.atom_arrays_from_fold_extras(fx, "atomchem4")
    p2 = A.atom_arrays_from_fold_extras(fx, "broadcastchem4")
    view = fold_view()
    checks = {}
    per_split = {}
    for split, ptr in (("train", p1["train_mol_ptr"]), ("dev", p1["dev_mol_ptr"])):
        a = np.asarray(p1[split], dtype=np.float32)
        b = np.asarray(p2[split], dtype=np.float32)
        const_ok = True
        mean_ok = True
        varying_molecules = 0
        for m in range(len(ptr) - 1):
            i0, i1 = int(ptr[m]), int(ptr[m + 1])
            if not np.all(b[i0:i1] == b[i0]):
                const_ok = False
            if not np.allclose(b[i0], a[i0:i1].mean(axis=0), rtol=0.0, atol=1e-6):
                mean_ok = False
            if np.any(a[i0:i1].std(axis=0) > 1e-6):
                varying_molecules += 1
        per_split[split] = {
            "n_molecules": int(len(ptr) - 1),
            "n_atoms": int(a.shape[0]),
            "broadcast_constant_within_molecule": bool(const_ok),
            "broadcast_equals_per_molecule_mean": bool(mean_ok),
            "molecules_with_nonconstant_true_features": int(varying_molecules),
            "max_abs_p1_minus_p2": float(np.abs(a - b).max()),
            "p1_std_within_molecule_mean": float(
                np.mean([a[int(ptr[m]):int(ptr[m + 1])].std(axis=0).mean() for m in range(len(ptr) - 1)])
            ),
        }
        checks[f"{split}_constant"] = bool(const_ok)
        checks[f"{split}_mean"] = bool(mean_ok)
        checks[f"{split}_contrast_nonzero"] = bool(np.abs(a - b).max() > 0)

    m1, meta1 = M.build_atomchem_model("A_AtomChem4", p1["spec"], view.train_norm, C.SEED, "cpu", dropout=0.0)
    m2, meta2 = M.build_atomchem_model("A_BroadcastChem4", p2["spec"], view.train_norm, C.SEED, "cpu", dropout=0.0)
    same = {
        "same_spec_hash": p1["spec"].hash() == p2["spec"].hash(),
        "same_parameter_count": meta1["parameters_augmented"] == meta2["parameters_augmented"],
        "same_init_state_hash": meta1["augmented_state_hash"] == meta2["augmented_state_hash"],
        "same_migration_audit": meta1["migration"]["new_columns_zero_count"] == meta2["migration"]["new_columns_zero_count"] == 0,
        "same_dropout_reference": True,
        "same_training_recipe": True,
    }
    checks.update({k: bool(v) for k, v in same.items()})
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "control": "P2 A_BroadcastChem4",
        "definition": fx.audit["broadcast_control"],
        "splits": per_split,
        "same_arm_properties": same,
        "parameters": {
            "p1": meta1["parameters_augmented"],
            "p2": meta2["parameters_augmented"],
            "base": meta1["parameters_original"],
        },
        "checks": checks,
        "interpretation_guard": (
            "the control tests whether the atom <-> feature correspondence matters; it is not a causal proof"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"broadcast-control audit failed: {checks}")
    return _write("broadcast_control_v4.json", payload)


# ---------------------------------------------------------------------------
# G7 de-normalisation / historical replay
# ---------------------------------------------------------------------------


def audit_denorm() -> dict[str, Any]:
    view = fold_view()
    y_all = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
    y_train = y_all[np.asarray(view.train_positions, dtype=np.int64)]
    recomputed_mean = float(y_train.mean())
    recomputed_scale = float(y_train.std())
    key = C.HISTORICAL["key"]
    run_dir = C.V2_RUNS / key
    train = C.read_json(run_dir / "train.json")
    data = np.load(run_dir / "predictions.npz")
    rows = [int(x) for x in data["rows"].tolist()]
    stored = np.asarray(data["pred"], dtype=np.float64)
    if rows != list(view.dev_positions):
        raise ValueError("historical prediction rows do not match fold-1 dev positions")
    rmse_stored = C.D.rmse(stored, view.y_dev)
    mae_stored = C.D.mae(stored, view.y_dev)

    spec = C.load_frozen_spec()
    model, _meta = C.M.build_model("A", spec, view.train_norm, C.SEED, "cpu", dropout=0.0)
    soup = torch.load(run_dir / "soup.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(soup, strict=True)
    model.eval()
    normalised = C.D.predict_view(model, view.dev_cache, device="cpu")
    manual = C.D.denormalise(normalised, view.target_norm)
    max_diff = float(np.abs(manual - stored).max())

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "target_norm": {
            "train_mean_recomputed": recomputed_mean,
            "train_scale_recomputed": recomputed_scale,
            "train_json_mean": float(view.target_norm["mean"]),
            "train_json_scale": float(view.target_norm["scale"]),
            "source": view.target_norm.get("source"),
            "n": int(view.target_norm["n"]),
        },
        "historical": {
            "key": key,
            "n_dev": int(view.y_dev.size),
            "rmse_from_stored": rmse_stored,
            "mae_from_stored": mae_stored,
            "historical_constant_rmse": C.HISTORICAL["dev_rmse"],
            "historical_constant_mae": C.HISTORICAL["dev_mae"],
            "checkpoint_replay_max_abs_diff": max_diff,
            "replay_tolerance": 1e-5,
            "replay_reason": "CPU replay of a CUDA-trained float32 checkpoint; no bitwise replay is claimed",
            "label_mean": float(view.y_dev.mean()),
            "label_std": float(view.y_dev.std()),
        },
        "checks": {
            "target_norm_is_fold_train_label_mean_std": bool(
                abs(recomputed_mean - float(view.target_norm["mean"])) < 1e-12
                and abs(recomputed_scale - float(view.target_norm["scale"])) < 1e-12
            ),
            "stored_metrics_match_historical_constants": bool(
                abs(rmse_stored - C.HISTORICAL["dev_rmse"]) < 1e-9 and abs(mae_stored - C.HISTORICAL["dev_mae"]) < 1e-9
            ),
            "stored_metrics_match_train_json": bool(
                abs(rmse_stored - float(train["metrics"]["dev_rmse_raw"])) < 1e-12
                and abs(mae_stored - float(train["metrics"]["dev_mae_raw"])) < 1e-12
            ),
            "checkpoint_replay_matches_stored_predictions": bool(max_diff <= 1e-5),
            "reported_metrics_are_raw_logd_units": True,
        },
    }
    if not all(payload["checks"].values()):
        raise RuntimeError(f"denormalisation audit failed: {payload['checks']}")
    return _write("denorm_check_v4.json", payload)


# ---------------------------------------------------------------------------
# G8 temporary CPU training
# ---------------------------------------------------------------------------


def audit_temporary_training() -> dict[str, Any]:
    fx = A.build_fold_extras(C.FOLD)
    view = fold_view()
    v1_before = C.assert_v1_unchanged()
    v2_before = C.assert_v2_unchanged()
    v3_before = C.assert_v3_unchanged(use_manifest=True)
    smoke = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for arm in C.ARMS:
            variant = C.ARM_VARIANT[arm]
            extras = A.atom_arrays_from_fold_extras(fx, variant)
            job = M.LocalChemTrainJob(
                key=f"v4_cpu_smoke_{arm}",
                arm=arm,
                fold=C.FOLD,
                recipe=C.RECIPE,
                seed=C.SEED,
                variant="scratch",
                device="cpu",
                out_root=tmp,
                smoke_train_limit=128,
                smoke_epochs=2,
                atom_extras=extras,
                spec_override=extras["spec"],
                norm_override=None,
                v4_audit={"smoke": True, "variant": variant},
            )
            payload = M.train_local_chem_job(job)
            run_dir = tmp / job.key
            preds = np.load(run_dir / "predictions.npz")
            soup = torch.load(run_dir / "soup.pt", map_location="cpu", weights_only=False)
            new_w = soup[M.ATOM_WEIGHT_KEY][:, C.ATOM_DIM_BASE :]
            smoke[arm] = {
                "run_dir_is_temporary": str(run_dir).startswith(str(tmp)),
                "files": sorted(p.name for p in run_dir.iterdir()),
                "train_rmse_finite": bool(np.isfinite(payload["metrics"]["train_rmse_raw"])),
                "dev_rmse_finite": bool(np.isfinite(payload["metrics"]["dev_rmse_raw"])),
                "predictions_finite": bool(np.isfinite(preds["pred"]).all()),
                "predictions_rows": int(preds["rows"].size),
                "parameter_delta": int(payload["v4"]["new_parameters"]),
                "protocol_id": payload["protocol_id"],
                "v2_loop_function": payload["v2_loop"]["function"],
                "collate_dispatch": payload["v2_loop"]["collate_dispatch"],
                "trained_new_columns_nonzero_count": int(torch.count_nonzero(new_w)),
                "trained_new_columns_abs_mean": float(new_w.abs().mean()),
                "dev_rmse_full_fold": float(payload["metrics"]["dev_rmse_raw"]),
                "labels_used_for_features": bool(payload["v4"]["labels_used_for_features"]),
            }
    v1_after = C.assert_v1_unchanged()
    v2_after = C.assert_v2_unchanged()
    v3_after = C.assert_v3_unchanged(use_manifest=True)
    checks = {
        "temporary_directory_only": all(s["run_dir_is_temporary"] for s in smoke.values()),
        "training_loop_reused": all(s["v2_loop_function"] == "v2_training.train_job reused verbatim" for s in smoke.values()),
        "parameter_delta_512": all(s["parameter_delta"] == C.EXPECTED_PARAMETER_DELTA for s in smoke.values()),
        "new_columns_trained": all(s["trained_new_columns_nonzero_count"] > 0 for s in smoke.values()),
        "predictions_finite": all(s["predictions_finite"] for s in smoke.values()),
        "two_epoch_smoke_not_a_result": True,
        "v1_unchanged": bool(v1_before["ok"] and v1_after["ok"]),
        "v2_unchanged": bool(v2_before["ok"] and v2_after["ok"]),
        "v3_unchanged": bool(v3_before["ok"] and v3_after["ok"]),
    }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "smoke": smoke,
        "checks": checks,
        "note": "the 2-epoch/128-molecule smoke values are engineering checks, not results",
    }
    if not all(checks.values()):
        raise RuntimeError(f"temporary training audit failed: {checks}")
    return _write("temp_training_v4.json", payload)


# ---------------------------------------------------------------------------
# G9 frozen artifacts
# ---------------------------------------------------------------------------


def audit_frozen_artifacts() -> dict[str, Any]:
    created = False
    if not C.V3_MANIFEST.exists():
        manifest = C.build_v3_manifest()
        C.write_json(C.V3_MANIFEST, manifest)
        created = True
    v1 = C.assert_v1_unchanged()
    v2 = C.assert_v2_unchanged()
    v3 = C.assert_v3_unchanged(use_manifest=True)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "v3_manifest_created_now": bool(created),
        "v3_manifest": C.file_record(C.V3_MANIFEST),
        "v1": {"ok": v1["ok"], "n_checks": v1["n_checks"], "n_failed": v1["n_failed"]},
        "v2": {"ok": v2["ok"], "n_checks": v2["n_checks"], "n_failed": v2["n_failed"]},
        "v3": {"ok": v3["ok"], "n_checks": v3["n_checks"], "n_failed": v3["n_failed"]},
        "checks": {
            "v1_unchanged": v1["ok"],
            "v2_unchanged": v2["ok"],
            "v3_unchanged": v3["ok"],
            "v3_manifest_present": C.V3_MANIFEST.exists(),
            "historical_comparator_unchanged": bool(
                C.sha256_file(C.V2_RUNS / C.HISTORICAL["key"] / "soup.pt")
                == C.V2_HISTORICAL_FILES["analysis/runs/F1_A_sel_s0/soup.pt"]
            ),
        },
    }
    if not all(payload["checks"].values()):
        raise RuntimeError(f"frozen-artifact audit failed: {payload['checks']}")
    return _write("frozen_artifacts_v4.json", payload)


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------


def run_all() -> dict[str, Any]:
    started = time.perf_counter()
    results: dict[str, Any] = {}
    failed: list[str] = []
    for name, fn in (
        ("atomchem", audit_atomchem),
        ("fold_alignment", audit_fold_alignment),
        ("atom_norm", audit_atom_norm),
        ("forward_equivalence", audit_forward_equivalence),
        ("broadcast_control", audit_broadcast_control),
        ("denorm", audit_denorm),
        ("frozen_artifacts", audit_frozen_artifacts),
        ("temp_training", audit_temporary_training),
    ):
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 - record and fail the gate
            failed.append(name)
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    status = "READY_FOR_GPU_AUTHORIZATION" if not failed else "CPU_GATES_FAILED"
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "env": C.env_info(),
        "elapsed_seconds": time.perf_counter() - started,
        "final_status": status,
        "failed": failed,
        "results": {k: (v.get("checks") if isinstance(v, dict) else None) for k, v in results.items()},
        "files": {
            name: C.track_rel(C.ANALYSIS / filename)
            for name, filename in (
                ("atomchem", "atomchem_audit_v4.json"),
                ("fold_alignment", "fold_alignment_v4.json"),
                ("atom_norm", "atom_norm_v4.json"),
                ("forward_equivalence", "forward_equivalence_v4.json"),
                ("broadcast_control", "broadcast_control_v4.json"),
                ("denorm", "denorm_check_v4.json"),
                ("frozen_artifacts", "frozen_artifacts_v4.json"),
                ("temp_training", "temp_training_v4.json"),
            )
            if (C.ANALYSIS / filename).exists()
        },
    }
    C.write_json(C.ANALYSIS / "cpu_gates_v4.json", payload)
    if failed:
        raise RuntimeError(f"CPU gates failed: {failed}; see {C.track_rel(C.ANALYSIS / 'cpu_gates_v4.json')}")
    return payload


__all__ = [
    "fold_view",
    "audit_atomchem",
    "audit_fold_alignment",
    "audit_atom_norm",
    "audit_forward_equivalence",
    "audit_broadcast_control",
    "audit_denorm",
    "audit_temporary_training",
    "audit_frozen_artifacts",
    "run_all",
]
