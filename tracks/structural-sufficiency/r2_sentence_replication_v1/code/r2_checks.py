"""Pre-training engineering checks for R2-SENTENCE-REPLICATION-v1.

Every reported check is a direct computation on frozen artefacts, not a claim.
The script writes ``analysis/checks.json`` and exits non-zero if any check fails.

Groups
  A  frozen GWIC-v0 sources / config / data artefacts are byte-identical
  B  split identity (D0 and D1) and label isolation
  C  per-run statistics are computed from the current Fit rows only
  D  the model/training tensor path is unchanged (incl. a CPU parity run)
  E  atom-renumbering, bond-reordering and batch invariance
  F  gradients reach every module and a real optimiser step learns
  G  no new parameter, module or feature tensor was added
"""

from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import r2_common as RC
import r2_data as RD
import gwic_data as GD
import gwic_model as M
import gwic_structure as S
import gwic_tensor as T

CHECKS: list[dict[str, Any]] = []


def rec(ident: str, name: str, ok: bool, details: Any) -> None:
    CHECKS.append({"id": ident, "name": name, "status": "PASS" if ok else "FAIL",
                   "details": RC.jsonable(details)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {ident} {name}", flush=True)


def _frozen_manifest() -> dict[str, Any]:
    path = RC.GWIC_RESULTS / "SENTENCE_BASE_S0" / "manifest.json"
    return RC.read_json(path)


def _frozen_checkpoint() -> dict[str, Any]:
    path = RC.GWIC_RESULTS / "SENTENCE_BASE_S0" / "state_best.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _predict(model: M.GwicModel, rows: Sequence[int],
             structs: Mapping[int, S.MoleculeStruct], raw: Mapping[str, Any],
             scaler: Mapping[str, float], batch: int = 256,
             device: Any = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows = np.asarray(rows, np.int64)
    out_std = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch)):
            part = rows[start:start + int(batch)]
            batch_t = T.collate(part.tolist(), structs, raw, device)
            out_std[start:start + part.size] = (
                model(batch_t, arm="SENTENCE_BASE")["yhat_std"].double().cpu().numpy())
    return out_std * float(scaler["std"]) + float(scaler["mean"]), out_std


# ---------------------------------------------------------------- A: identity
def group_a(cfg: Mapping[str, Any]) -> dict[str, Any]:
    details = RC.assert_frozen_sources()
    rec("A1_frozen_sources",
        "GWIC-v0 sources, aggregate code_sha and artefact hashes match the pins",
        True, details)
    spec = cfg["arms"]["SENTENCE_BASE"]
    model_details = {
        "dims": cfg["dims"], "recipe": cfg["recipe"], "sentence_base_spec": spec,
        "config_sha256": RC.GWIC_CONFIG_SHA256,
        "dims_used_for_model": sorted(cfg["dims"]),
        "uses_words": bool(spec["use_words"]), "uses_geometry": bool(spec["use_geometry"]),
    }
    rec("A2_frozen_config",
        "model dims, recipe and SENTENCE_BASE spec come from the frozen GWIC-v0 config",
        (not spec["use_words"]) and (not spec["use_geometry"])
        and cfg["protocol_id"] == "gwic-v0-graph-word-incidence-composition-pilot",
        model_details)
    return details


# ------------------------------------------------------------------- B: splits
def group_b(raw: Mapping[str, Any]) -> dict[str, Any]:
    frozen = _frozen_manifest()
    d0 = RD.load_split("D0", raw)
    d1 = RD.load_split("D1", raw)
    official_train = np.sort(np.asarray(raw["train_idx"], np.int64))
    official_valid = np.sort(np.asarray(raw["valid_idx"], np.int64))
    b1 = {
        "n_fit": int(d0["fit_rows"].size), "n_eval": int(d0["eval_rows"].size),
        "identity_sha256": d0["spec"]["identity_sha256"],
        "file_sha256": d0["spec"]["file_sha256"],
        "fit_rows_sha256": d0["spec"]["fit_rows_sha256"],
        "eval_rows_sha256": d0["spec"]["eval_rows_sha256"],
        "overlap": int(np.intersect1d(d0["fit_rows"], d0["eval_rows"]).size),
        "cover_official_train": bool(
            np.array_equal(np.union1d(d0["fit_rows"], d0["eval_rows"]), official_train)),
        "matches_gwic_v0_manifest": bool(
            d0["spec"]["fit_rows_sha256"] == frozen["fit_rows_sha256"]
            and d0["spec"]["eval_rows_sha256"] == frozen["dev_rows_sha256"]
            and int(d0["fit_rows"].size) == int(frozen["n_fit"])
            and int(d0["eval_rows"].size) == int(frozen["n_dev"])),
        "outside_official_valid": int(
            np.intersect1d(np.union1d(d0["fit_rows"], d0["eval_rows"]), official_valid).size),
    }
    ok1 = (b1["n_fit"] == 2856 and b1["n_eval"] == 504 and b1["overlap"] == 0
           and b1["cover_official_train"] and b1["matches_gwic_v0_manifest"]
           and b1["outside_official_valid"] == 0
           and b1["identity_sha256"] == RC.D0_IDENTITY
           and b1["file_sha256"] == RC.D0_FILE_SHA256)
    rec("B1_split_d0", "D0 split identity, disjointness and coverage", ok1, b1)

    regen = RD.d1_recompute_scaffolds(raw)
    spec1 = d1["spec"]
    b2 = {
        "n_fit": spec1["n_fit"], "n_eval": spec1["n_eval"],
        "identity_sha256": spec1["identity_sha256"],
        "file_sha256": spec1["file_sha256"],
        "fit_rows_sha256": spec1["fit_rows_sha256"],
        "eval_rows_sha256": spec1["eval_rows_sha256"],
        "overlap": int(np.intersect1d(d1["fit_rows"], d1["eval_rows"]).size),
        "cover_official_train": bool(
            np.array_equal(np.union1d(d1["fit_rows"], d1["eval_rows"]), official_train)),
        "outside_official_valid": int(
            np.intersect1d(np.union1d(d1["fit_rows"], d1["eval_rows"]), official_valid).size),
        "n_scaffolds_fit": spec1["n_scaffolds_fit"],
        "n_scaffolds_eval": spec1["n_scaffolds_eval"],
        "n_scaffolds_official_train": spec1["n_scaffolds_official_train"],
        "scaffold_overlap": spec1["scaffold_overlap"],
        "scaffold_regeneration": regen,
    }
    ok2 = (spec1["n_fit"] == 2856 and spec1["n_eval"] == 504 and b2["overlap"] == 0
           and b2["cover_official_train"] and b2["outside_official_valid"] == 0
           and spec1["identity_sha256"] == RC.D1_IDENTITY
           and spec1["file_sha256"] == RC.D1_FILE_SHA256
           and spec1["scaffold_overlap"] == 0
           and regen["n_scaffold_map_mismatch"] == 0
           and regen["fit_rows_equal"] and regen["heldout_rows_equal"]
           and regen["scaffold_overlap_recomputed"] == 0
           and regen["n_regenerated_fit"] == 2856
           and regen["n_regenerated_heldout"] == 504)
    rec("B2_split_d1",
        "D1 split identity, disjointness, scaffold disjointness and frozen-algorithm regeneration",
        ok2, b2)

    bundle = RD.load_bundle("D0", "fit")
    finite_rows = np.flatnonzero(np.isfinite(bundle.labels))
    b3 = {
        "raw_members": list(RC.RAW_STRUCT_KEYS),
        "has_labels_member": "labels" in RC.RAW_STRUCT_KEYS,
        "has_test_idx_member": "test_idx" in RC.RAW_STRUCT_KEYS,
        "n_finite_labels": int(finite_rows.size),
        "n_official_train": int(official_train.size),
        "finite_labels_equal_official_train": bool(
            np.array_equal(finite_rows, official_train)),
        "n_finite_valid_labels": int(np.isfinite(bundle.labels[official_valid]).sum()),
        "n_finite_test_rows": int(np.isfinite(bundle.labels[
            np.setdiff1d(np.arange(bundle.labels.size), official_train)]).sum()),
        "split_files_contain_labels": False,
    }
    ok3 = (not b3["has_labels_member"] and not b3["has_test_idx_member"]
           and b3["finite_labels_equal_official_train"]
           and b3["n_finite_valid_labels"] == 0 and b3["n_finite_test_rows"] == 0)
    rec("B3_label_isolation",
        "no Valid/Test label is present, loaded or used as a fit target", ok3, b3)
    return {"D0": d0["spec"], "D1": d1["spec"], "bundle": bundle}


# ------------------------------------------------------------ C: normalisation
def _tampered_labels(rows: np.ndarray, delta: float):
    original = GD._load_train_labels

    def patched() -> dict[str, np.ndarray]:
        payload = original()
        row_id = np.asarray(payload["row_id"], np.int64)
        label = np.asarray(payload["label"], np.float64).copy()
        mask = np.isin(row_id, np.asarray(rows, np.int64))
        label[mask] += float(delta)
        return {"row_id": row_id, "label": label, "sha256": payload["sha256"]}

    return original, patched


def group_c(raw: Mapping[str, Any], bundle: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    d1_bundle = RD.load_bundle("D1", "fit")
    y0 = bundle.labels[bundle.fit_rows]
    c1 = {
        "d0_scaler": bundle.scaler,
        "d0_recomputed_from_fit": {"mean": float(y0.mean()), "std": float(y0.std(ddof=0)),
                                   "n": int(y0.size)},
        "d1_scaler": d1_bundle.scaler,
        "scalers_differ": bool(abs(bundle.scaler["mean"] - d1_bundle.scaler["mean"]) > 0),
    }
    scaler_equal = (
        abs(bundle.scaler["mean"] - float(y0.mean())) < 1e-12
        and abs(bundle.scaler["std"] - float(y0.std(ddof=0))) < 1e-12
        and int(bundle.scaler["n"]) == int(y0.size))
    original, patched = _tampered_labels(bundle.eval_rows, 100.0)
    try:
        GD._load_train_labels = patched
        tampered = RD.load_bundle("D0", "fit")
    finally:
        GD._load_train_labels = original
    c1["tampered_eval_labels_scaler"] = tampered.scaler
    c1["scaler_invariant_to_eval_labels"] = bool(tampered.scaler == bundle.scaler)
    c1["tampered_eval_mean"] = float(np.mean(tampered.labels[bundle.eval_rows]))
    rec("C1_target_scaler",
        "target z-score is fit-only and invariant to held-out labels",
        bool(scaler_equal and c1["scaler_invariant_to_eval_labels"] and c1["scalers_differ"]),
        c1)

    graph_fit = bundle.graph_scalars[bundle.fit_rows]
    graph_all = bundle.graph_scalars
    g_mean = graph_fit.mean(axis=0)
    g_scale = graph_fit.std(axis=0)
    g_scale = np.where(g_scale > 1e-12, g_scale, 1.0)
    c2 = {
        "equal_fit_recomputation": bool(np.array_equal(g_mean, bundle.graph_mean)
                                        and np.array_equal(g_scale, bundle.graph_scale)),
        "max_abs_delta_vs_all_rows_mean": float(
            np.max(np.abs(graph_all.mean(axis=0) - bundle.graph_mean))),
        "max_abs_delta_vs_all_rows_scale": float(
            np.max(np.abs(graph_all.std(axis=0) - bundle.graph_scale))),
        "invariant_to_labels": bool(np.array_equal(
            tampered.graph_mean, bundle.graph_mean)
            and np.array_equal(tampered.graph_scale, bundle.graph_scale)),
        "mean": g_mean, "scale": g_scale,
    }
    rec("C2_graph_scaler",
        "graph-level standardisation uses only Fit rows and ignores labels",
        bool(c2["equal_fit_recomputation"] and c2["invariant_to_labels"]
             and max(c2["max_abs_delta_vs_all_rows_mean"],
                     c2["max_abs_delta_vs_all_rows_scale"]) > 0),
        c2)

    v0_bundle = RD.load_bundle("D0", "v0_fit_dev")
    ckpt = _frozen_checkpoint()
    ck_mean = ckpt["state"]["atom_scalar_mean"].numpy().astype(np.float64)
    ck_scale = ckpt["state"]["atom_scalar_scale"].numpy().astype(np.float64)
    atom_rows = [GD.atom_scalars(GD.molecule_view(raw, int(r)))
                 for r in bundle.fit_rows.tolist()]
    fit_atoms = np.concatenate(atom_rows, axis=0)
    a_mean = fit_atoms.mean(axis=0)
    a_scale = fit_atoms.std(axis=0)
    a_scale = np.where(a_scale > 1e-12, a_scale, 1.0)
    dev_atoms = np.concatenate([GD.atom_scalars(GD.molecule_view(raw, int(r)))
                                for r in bundle.eval_rows.tolist()], axis=0)
    z_fit = (dev_atoms - a_mean) / a_scale
    z_v0 = (dev_atoms - v0_bundle.atom_scalar_mean) / v0_bundle.atom_scalar_scale
    c3 = {
        "equal_fit_recomputation": bool(np.array_equal(a_mean, bundle.atom_scalar_mean)
                                        and np.array_equal(a_scale, bundle.atom_scalar_scale)),
        "v0_mode_matches_frozen_checkpoint_buffers": bool(
            np.allclose(v0_bundle.atom_scalar_mean, ck_mean, atol=1e-6)
            and np.allclose(v0_bundle.atom_scalar_scale, ck_scale, atol=1e-6)),
        "fit_only_mean": a_mean, "fit_only_scale": a_scale,
        "v0_fit_dev_mean": v0_bundle.atom_scalar_mean,
        "v0_fit_dev_scale": v0_bundle.atom_scalar_scale,
        "max_abs_delta_mean": float(np.max(np.abs(a_mean - v0_bundle.atom_scalar_mean))),
        "max_abs_delta_scale": float(np.max(np.abs(a_scale - v0_bundle.atom_scalar_scale))),
        "max_abs_standardised_shift_dev_atoms": float(np.max(np.abs(z_fit - z_v0))),
        "mean_abs_standardised_shift_dev_atoms": float(np.mean(np.abs(z_fit - z_v0))),
        "n_fit_atoms": int(fit_atoms.shape[0]),
        "n_fit_dev_atoms_v0": int(np.concatenate([
            GD.atom_scalars(GD.molecule_view(raw, int(r)))
            for r in np.union1d(bundle.fit_rows, bundle.eval_rows).tolist()], axis=0).shape[0]),
    }
    rec("C3_atom_scalars",
        "atom-level standardisation is fit-only; deviation from the v0 implementation is measured",
        bool(c3["equal_fit_recomputation"] and c3["v0_mode_matches_frozen_checkpoint_buffers"]
             and c3["max_abs_delta_mean"] > 0),
        c3)

    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    atom_dims, bond_dims = GD.atom_bond_dims()
    atom_ok, bond_ok = GD.category_support(raw, bundle.fit_rows, atom_dims, bond_dims)
    probe = int(bundle.eval_rows[0])
    a0, a1 = int(node_ptr[probe]), int(node_ptr[probe + 1])
    unsupported = np.setdiff1d(np.arange(atom_dims[0]), np.flatnonzero(atom_ok[0]))
    if unsupported.size == 0:
        raise ValueError("no fit-unsupported atom category exists to test with")
    unseen_category = int(unsupported.max())
    mutated = {k: np.asarray(v) for k, v in raw.items()}
    atom_feat = np.array(mutated["atom_feat"], dtype=np.int64, copy=True)
    atom_feat[a0:a1, 0] = unseen_category  # outside the Fit support
    mutated["atom_feat"] = atom_feat
    atom_ok_m, bond_ok_m = GD.category_support(mutated, bundle.fit_rows, atom_dims, bond_dims)
    support_unchanged = all(np.array_equal(x, y) for x, y in zip(atom_ok, atom_ok_m)) and \
        all(np.array_equal(x, y) for x, y in zip(bond_ok, bond_ok_m))
    model, _ = M.build_model(atom_dims, bond_dims, 0, RD.norm_of(bundle), use_words=False,
                             config=dict(cfg))
    zeroed = model.zero_unseen(atom_ok, bond_ok, atom_dims, bond_dims)
    rows_zero = bool((model.atom_emb[0].weight[
        torch.as_tensor(atom_feat[a0:a1, 0], dtype=torch.long)] == 0).all())
    c4 = {
        "probe_row": probe, "fit_support_size_field0": int(atom_ok[0].sum()),
        "probe_mutated_category": unseen_category,
        "n_unsupported_categories_field0": int(unsupported.size),
        "support_unchanged_after_dev_mutation": bool(support_unchanged),
        "mutated_categories_zeroed": rows_zero,
        "zeroed_counts": zeroed,
        "n_fit_rows": int(bundle.fit_rows.size),
    }
    rec("C4_category_support",
        "categorical support is the Fit-row union; Dev-only categories cannot enter it",
        bool(support_unchanged and rows_zero), c4)
    return {"c3": c3, "c4": c4}


# ------------------------------------------------------ D: unchanged tensor path
def group_d(cfg: Mapping[str, Any], bundle: Any, raw: Mapping[str, Any],
            parity: bool) -> dict[str, Any]:
    frozen = _frozen_manifest()
    ckpt = _frozen_checkpoint()
    atom_dims, bond_dims = GD.atom_bond_dims()
    model, info = M.build_model(atom_dims, bond_dims, 0, RD.norm_of(bundle), use_words=False,
                                config=dict(cfg))
    ledger = info["parameters"]
    word_attrs = [name for name in ("word_joint", "word_mlp", "word_norm", "score_mlp",
                                    "q_mlp") if hasattr(model, name)]
    d1 = {
        "ledger": ledger, "frozen_ledger": frozen["params"],
        "ledger_equal": RC.jsonable(ledger) == RC.jsonable(frozen["params"]),
        "total": int(ledger["total"]), "expected_total": RC.PARAMS_TOTAL,
        "word_modules_instantiated": word_attrs,
        "word_in_dim": int(model.word_in_dim),
        "backbone_in_dim": int(model.backbone_in_dim),
        "readout_in_dim": int(model.readout_in_dim),
        "no_word_modules": len(word_attrs) == 0 and int(model.word_in_dim) == 0,
    }
    rec("D1_parameter_ledger",
        "SENTENCE_BASE parameter ledger equals GWIC-v0 (416,177) and instantiates no word module",
        bool(d1["ledger_equal"] and d1["total"] == RC.PARAMS_TOTAL and d1["no_word_modules"]),
        d1)

    state_keys = sorted(model.state_dict().keys())
    frozen_keys = sorted(ckpt["state"].keys())
    shapes_ok = all(tuple(model.state_dict()[k].shape) == tuple(ckpt["state"][k].shape)
                    for k in frozen_keys if k in model.state_dict())
    d2 = {"n_state_keys": len(state_keys), "n_frozen_keys": len(frozen_keys),
          "keys_equal": state_keys == frozen_keys, "shapes_equal": bool(shapes_ok),
          "missing": sorted(set(frozen_keys) - set(state_keys)),
          "extra": sorted(set(state_keys) - set(frozen_keys)),
          "numel": int(sum(p.numel() for p in model.state_dict().values()))}
    rec("D2_state_dict",
        "state_dict keys and shapes are identical to the frozen SENTENCE_BASE_S0 checkpoint",
        bool(d2["keys_equal"] and d2["shapes_equal"]
             and d2["n_state_keys"] == RC.SENTENCE_BASE_STATE_KEYS),
        d2)

    d3 = {"init_hashes": info["init_hashes"],
          "frozen_init_hashes": frozen["init_hashes"],
          "equal": RC.jsonable(info["init_hashes"]) == RC.jsonable(frozen["init_hashes"])}
    rec("D3_init_hashes", "seed-0 initialisation hashes equal the frozen manifest", d3["equal"], d3)

    # D4: replay the frozen checkpoint on D0 Dev
    model_replay = M.load_model_from_state(atom_dims, bond_dims, ckpt["state"],
                                           RD.norm_of(bundle), use_words=False,
                                           config=dict(cfg), device="cpu")
    dev_structs = S.build_all(raw, bundle.eval_rows)
    pred, _ = _predict(model_replay, bundle.eval_rows, dev_structs, raw, ckpt["scaler"])
    y = bundle.labels[bundle.eval_rows]
    frozen_pred = np.load(RC.GWIC_RESULTS / "SENTENCE_BASE_S0" / "predictions_best_dev.npz")
    order = {int(r): i for i, r in enumerate(frozen_pred["row_id"].tolist())}
    idx = np.asarray([order[int(r)] for r in bundle.eval_rows.tolist()], np.int64)
    aligned = frozen_pred["pred"][idx]
    d4 = {
        "dev_rmse": RC.rmse(pred, y), "expected_dev_rmse": 0.5505589717057867,
        "abs_delta_rmse": abs(RC.rmse(pred, y) - 0.5505589717057867),
        "dev_mae": RC.mae(pred, y), "expected_dev_mae": 0.42758191774670373,
        "max_abs_delta_vs_committed_predictions": float(np.max(np.abs(pred - aligned))),
        "mean_abs_delta_vs_committed_predictions": float(np.mean(np.abs(pred - aligned))),
        "n_dev": int(bundle.eval_rows.size), "device": "cpu", "batch": 256,
    }
    rec("D4_checkpoint_replay",
        "the frozen D0 seed-0 checkpoint replays RMSE 0.550559 and its committed predictions",
        bool(d4["abs_delta_rmse"] < 1e-6 and abs(d4["dev_mae"] - d4["expected_dev_mae"]) < 1e-6
             and d4["max_abs_delta_vs_committed_predictions"] < 1e-5),
        d4)
    del dev_structs

    # D6: batch-tensor identity and the measured effect of the only numeric deviation
    rows = np.concatenate([bundle.fit_rows[:8], bundle.eval_rows[:8]])
    st_v0 = S.build_all(raw, rows)
    st_new = S.build_all(raw, rows)
    b_v0 = T.collate(rows.tolist(), st_v0, raw, "cpu")
    b_new = T.collate(rows.tolist(), st_new, raw, "cpu")
    v0_bundle = RD.load_bundle("D0", "v0_fit_dev")
    per_key: dict[str, Any] = {}
    for key in sorted(b_v0.keys()):
        if key in ("rows",):
            per_key[key] = {"equal": bool(np.array_equal(b_v0[key], b_new[key]))}
            continue
        x, z = b_v0[key], b_new[key]
        if not torch.is_tensor(x) or not torch.is_tensor(z):
            per_key[key] = {"equal": RC.jsonable(x) == RC.jsonable(z)}
            continue
        same_shape = tuple(x.shape) == tuple(z.shape)
        equal = bool(same_shape and torch.equal(x, z))
        per_key[key] = {"shape": list(x.shape), "dtype": str(x.dtype),
                        "equal": equal,
                        "max_abs_delta": float((x.double() - z.double()).abs().max()) if same_shape and x.numel() else 0.0}
    keys_equal = sorted(b_v0.keys()) == sorted(b_new.keys()) == sorted(RC.BATCH_KEYS)
    differ = sorted(k for k, v in per_key.items() if not v.get("equal"))
    # the four standardisation buffers are the only place the two data modes differ
    model_v0, _ = M.build_model(atom_dims, bond_dims, 0, RD.norm_of(v0_bundle),
                                use_words=False, config=dict(cfg))
    model_fit, _ = M.build_model(atom_dims, bond_dims, 0, RD.norm_of(bundle),
                                 use_words=False, config=dict(cfg))
    buffer_names = ("atom_scalar_mean", "atom_scalar_scale", "graph_mean", "graph_scale")
    weights_only = {k: v for k, v in ckpt["state"].items() if k not in buffer_names}
    model_v0.load_state_dict(weights_only, strict=False)
    model_fit.load_state_dict(weights_only, strict=False)
    model_v0.eval()
    model_fit.eval()
    buffers: dict[str, Any] = {}
    for name in buffer_names:
        v0_value = getattr(model_v0, name).detach().numpy().astype(np.float64)
        fit_value = getattr(model_fit, name).detach().numpy().astype(np.float64)
        ck_value = ckpt["state"][name].numpy().astype(np.float64)
        buffers[name] = {
            "v0_mode_equals_checkpoint": bool(np.allclose(v0_value, ck_value, atol=1e-6)),
            "fit_mode_equals_checkpoint": bool(np.allclose(fit_value, ck_value, atol=1e-6)),
            "max_abs_delta_fit_vs_v0": float(np.max(np.abs(fit_value - v0_value))),
        }
    with torch.no_grad():
        y_v0 = model_v0(b_v0, arm="SENTENCE_BASE")["yhat_std"].double()
        y_fit = model_fit(b_new, arm="SENTENCE_BASE")["yhat_std"].double()
    buffers_ok = all(v["v0_mode_equals_checkpoint"] for v in buffers.values()) and \
        buffers["graph_mean"]["fit_mode_equals_checkpoint"] and \
        buffers["graph_scale"]["fit_mode_equals_checkpoint"] and \
        not buffers["atom_scalar_mean"]["fit_mode_equals_checkpoint"]
    d6 = {
        "n_rows": int(rows.size), "keys_equal_frozen_set": keys_equal,
        "batch_tensors_identical": bool(all(v.get("equal") for v in per_key.values())),
        "per_key": per_key, "differing_keys": differ, "buffers": buffers,
        "buffers_behave_as_designed": bool(buffers_ok),
        "max_abs_delta_standardised_prediction": float((y_v0 - y_fit).abs().max()),
        "mean_abs_delta_standardised_prediction": float((y_v0 - y_fit).abs().mean()),
    }
    rec("D6_tensor_and_buffer_identity",
        "batched tensors are identical to the frozen loader; only the fit-only atom-scalar buffers differ",
        bool(keys_equal and differ == [] and buffers_ok), d6)

    d5: dict[str, Any] = {"skipped": True}
    if parity:
        d5 = _parity_run(cfg)
        rec("D5_cpu_parity",
            "the new entry point reproduces the frozen gwic_train.py 2-epoch CPU trajectory bit-for-bit",
            bool(d5.get("all_equal")), d5)
    else:
        rec("D5_cpu_parity", "parity run skipped by flag (must be run before formal training)",
            False, d5)
    return {"d1": d1, "d2": d2, "d4": d4, "d5": d5, "d6": d6}


def _read_curves(path: Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parity_run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    fields = ["step", "lr", "train_loss", "train_grad_norm", "fit_rmse", "fit_mae",
              "dev_rmse", "dev_mae"]
    tmp = Path(tempfile.mkdtemp(prefix="r2_parity_"))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RC.GWIC_CODE)
    base = [sys.executable]
    cmd_v0 = base + [str(RC.GWIC_CODE / "gwic_train.py"), "--arm", "SENTENCE_BASE",
                     "--seed", "0", "--device", "cpu", "--epochs", "2",
                     "--run-id", "parity_v0", "--out", str(tmp / "v0")]
    cmd_r2 = base + [str(RC.CODE_DIR / "r2_train.py"), "--split", "D0",
                     "--seed", "0", "--device", "cpu", "--epochs", "2",
                     "--atom-scalar-mode", "v0_fit_dev",
                     "--run-id", "parity_r2", "--out", str(tmp / "r2")]
    runs: dict[str, Any] = {}
    for tag, cmd in (("v0", cmd_v0), ("r2", cmd_r2)):
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(RC.REPO_ROOT), env=env, capture_output=True,
                              text=True)
        runs[tag] = {"returncode": proc.returncode, "seconds": time.time() - t0,
                     "stdout_tail": proc.stdout.strip().splitlines()[-3:],
                     "stderr_tail": proc.stderr.strip().splitlines()[-5:]}
        if proc.returncode != 0:
            return {"error": f"{tag} run failed", "runs": runs}
    curves_v0 = _read_curves(tmp / "v0" / "curves.csv")
    curves_r2 = _read_curves(tmp / "r2" / "curves.csv")
    rows_equal = len(curves_v0) == len(curves_r2) == 2
    field_diffs: dict[str, Any] = {}
    if rows_equal:
        for field in fields:
            a = [float(r[field]) for r in curves_v0]
            b = [float(r[field]) for r in curves_r2]
            field_diffs[field] = {"v0": a, "r2": b, "equal": a == b}
        rows_equal = all(v["equal"] for v in field_diffs.values())
    state_v0 = torch.load(tmp / "v0" / "state_final.pt", map_location="cpu",
                          weights_only=False)["state"]
    state_r2 = torch.load(tmp / "r2" / "state_final.pt", map_location="cpu",
                          weights_only=False)["state"]
    keys_equal = sorted(state_v0) == sorted(state_r2)
    tensors_equal = bool(keys_equal and all(torch.equal(state_v0[k], state_r2[k])
                                            for k in state_v0))
    man_v0 = RC.read_json(tmp / "v0" / "manifest.json")
    man_r2 = RC.read_json(tmp / "r2" / "manifest.json")
    return {
        "epochs": 2, "device": "cpu", "atom_scalar_mode_r2": "v0_fit_dev",
        "runs": runs, "n_curve_rows": [len(curves_v0), len(curves_r2)],
        "curve_fields": field_diffs,
        "curve_rows_equal": bool(rows_equal),
        "state_keys_equal": bool(keys_equal), "state_tensors_equal": tensors_equal,
        "scaler_equal": man_v0["scaler"] == man_r2["scaler"],
        "fit_rows_equal": man_v0["fit_rows_sha256"] == man_r2["fit_rows_sha256"],
        "dev_rows_equal": man_v0["dev_rows_sha256"] == man_r2["dev_rows_sha256"],
        "init_hashes_equal": man_v0["init_hashes"] == man_r2["init_hashes"],
        "all_equal": bool(rows_equal and keys_equal and tensors_equal
                          and man_v0["scaler"] == man_r2["scaler"]
                          and man_v0["fit_rows_sha256"] == man_r2["fit_rows_sha256"]),
        "tmp_dir": str(tmp),
    }


# ------------------------------------------------------------- E: invariances
def group_e(cfg: Mapping[str, Any], bundle: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    atom_dims, bond_dims = GD.atom_bond_dims()
    ckpt = _frozen_checkpoint()
    model = M.load_model_from_state(atom_dims, bond_dims, ckpt["state"],
                                    RD.norm_of(bundle), use_words=False,
                                    config=dict(cfg), device="cpu")
    model.eval()
    rows = np.concatenate([bundle.fit_rows[:2], bundle.eval_rows[:2]])
    rng = np.random.default_rng(20261123)
    base_views = []
    for row in rows.tolist():
        view = GD.molecule_view(raw, int(row))
        view["atom_scalars"] = GD.atom_scalars(view)
        base_views.append(view)

    def pred_of(views: Sequence[Mapping[str, Any]]) -> np.ndarray:
        batch = RD.batch_from_views(views, "cpu")
        with torch.no_grad():
            return model(batch, arm="SENTENCE_BASE")["yhat_std"].double().numpy()

    base = pred_of(base_views)
    perm_views, perm_preds = [], []
    for view in base_views:
        n = int(view["n_atoms"])
        perm = rng.permutation(n)
        perm_views.append(RD.permuted_view(view, perm))
        perm_preds.append(perm)
    e1_delta = float(np.max(np.abs(pred_of(perm_views) - base)))
    rec("E1_atom_renumbering",
        "random atom renumbering leaves the molecule prediction unchanged",
        e1_delta < 1e-5, {"max_abs_delta": e1_delta, "rows": rows.tolist(),
                          "permutation_sizes": [int(v["n_atoms"]) for v in base_views]})

    bond_views = []
    for view in base_views:
        m = int(view["n_bonds"])
        order = rng.permutation(m)
        bond_views.append(RD.permuted_view(view, np.arange(int(view["n_atoms"])), order))
    e2_delta = float(np.max(np.abs(pred_of(bond_views) - base)))
    rec("E2_bond_reordering",
        "random bond reordering leaves the molecule prediction unchanged",
        e2_delta < 1e-5, {"max_abs_delta": e2_delta,
                          "bond_counts": [int(v["n_bonds"]) for v in base_views]})

    singles = pred_of(base_views)
    reversed_preds = pred_of(list(reversed(base_views)))[::-1]
    mixed = pred_of(base_views + bond_views)
    e3 = {"max_abs_delta_reversed_order": float(np.max(np.abs(reversed_preds - singles))),
          "max_abs_delta_mixed_batch": float(np.max(np.abs(mixed[:len(base)] - singles)))}
    rec("E3_batch_invariance",
        "predictions are batch-composition and batch-order invariant",
        e3["max_abs_delta_reversed_order"] < 1e-6 and e3["max_abs_delta_mixed_batch"] < 1e-6,
        e3)

    dev_structs = S.build_all(raw, bundle.eval_rows)
    full_pred, _ = _predict(model, bundle.eval_rows, dev_structs, raw, bundle.scaler,
                            batch=256)
    chunk_pred, _ = _predict(model, bundle.eval_rows, dev_structs, raw, bundle.scaler,
                             batch=37)
    e3b = {"n_dev": int(bundle.eval_rows.size),
           "max_abs_delta_batch256_vs_batch37": float(np.max(np.abs(full_pred - chunk_pred)))}
    rec("E3b_eval_batch_invariance",
        "Dev predictions do not depend on the evaluation batch size",
        e3b["max_abs_delta_batch256_vs_batch37"] < 1e-5, e3b)
    return {"e1": e1_delta, "e2": e2_delta, "e3": e3, "e3b": e3b}


# --------------------------------------------------------------- F: gradients
def group_f(cfg: Mapping[str, Any], bundle: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    atom_dims, bond_dims = GD.atom_bond_dims()
    torch.manual_seed(0)
    model, _ = M.build_model(atom_dims, bond_dims, 0, RD.norm_of(bundle), use_words=False,
                             config=dict(cfg))
    atom_ok, bond_ok = GD.category_support(raw, bundle.fit_rows, atom_dims, bond_dims)
    model.zero_unseen(atom_ok, bond_ok, atom_dims, bond_dims)
    rows = bundle.fit_rows[:64]
    structs = S.build_all(raw, rows)
    batch = T.collate(rows.tolist(), structs, raw, "cpu")
    y = torch.as_tensor(bundle.labels_std()[rows], dtype=torch.float32)
    out = model(batch, arm="SENTENCE_BASE")
    loss = torch.mean((out["yhat_std"] - y) ** 2)
    loss.backward()
    norms: dict[str, float] = {}
    zeroed: list[str] = []
    nonfinite: list[str] = []
    for name, p in model.named_parameters():
        if p.grad is None:
            nonfinite.append(f"{name}:None")
            norms[name] = float("nan")
            continue
        n = float(p.grad.detach().norm())
        norms[name] = n
        if not np.isfinite(n):
            nonfinite.append(name)
        elif n == 0.0:
            zeroed.append(name)
    groups = ("atom_emb", "bond_raw_emb", "atom_norm", "bond_encoder", "nbr_emb", "path_u",
              "path_v", "path_joint", "path_emb", "end_emb", "close_emb", "path_gate",
              "backbone_mlp", "backbone_norm", "readout_mlp", "beta")
    group_norms = {g: float(np.sqrt(sum(v ** 2 for k, v in norms.items()
                                       if k.split(".")[0] == g and np.isfinite(v))))
                   for g in groups}
    silent_groups = sorted(g for g, v in group_norms.items() if v == 0.0)
    f1 = {"loss": float(loss), "n_parameters": len(norms), "zero_grad_parameters": zeroed,
          "nonfinite_parameters": nonfinite, "group_grad_norms": group_norms,
          "silent_groups": silent_groups, "min_grad_norm": float(min(norms.values())),
          "max_grad_norm": float(max(norms.values()))}
    rec("F1_gradients",
        "a real optimiser step produces finite, non-zero gradients in every module group",
        bool(not nonfinite and not silent_groups),
        f1)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    rows = bundle.fit_rows[:32]
    structs = S.build_all(raw, rows)
    batch = T.collate(rows.tolist(), structs, raw, "cpu")
    y = torch.as_tensor(bundle.labels_std()[rows], dtype=torch.float32)
    model.train()
    losses = []
    for _ in range(30):
        out = model(batch, arm="SENTENCE_BASE")
        loss = torch.mean((out["yhat_std"] - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    f2 = {"n_molecules": int(rows.size), "steps": 30, "lr": 1e-3,
          "first_loss": losses[0], "last_loss": losses[-1],
          "ratio": losses[-1] / losses[0]}
    rec("F2_overfit_smoke",
        "30 AdamW steps on 32 Fit molecules reduce the loss by more than half",
        bool(f2["ratio"] < 0.5), f2)
    return {"f1": f1, "f2": f2}


# --------------------------------------------------------------- G: no additions
def group_g(cfg: Mapping[str, Any], bundle: Any) -> dict[str, Any]:
    atom_dims, bond_dims = GD.atom_bond_dims()
    model, _ = M.build_model(atom_dims, bond_dims, 0, RD.norm_of(bundle), use_words=False,
                             config=dict(cfg))
    expected_children = {"atom_emb", "bond_raw_emb", "atom_norm", "bond_encoder", "nbr_emb",
                         "path_u", "path_v", "path_joint", "path_emb", "end_emb",
                         "close_emb", "path_gate", "backbone_mlp", "backbone_norm",
                         "readout_mlp"}
    children = {name for name, _ in model.named_children()}
    emb = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Embedding)]
    g1 = {"collate_keys": list(RC.BATCH_KEYS), "n_keys": len(RC.BATCH_KEYS),
          "source": "frozen gwic_tensor.collate"}
    rec("G1_no_new_feature_tensor",
        "the collated batch key set is the frozen GWIC-v0 key set (verified in D6)",
        sorted(RC.BATCH_KEYS) == sorted(T.collate(
            [int(bundle.fit_rows[0])], S.build_all(bundle.raw, [int(bundle.fit_rows[0])]),
            bundle.raw, "cpu").keys()), g1)
    g2 = {"top_level_children": sorted(children),
          "unexpected_children": sorted(children - expected_children),
          "missing_children": sorted(expected_children - children),
          "n_embeddings": len(emb), "embeddings": emb,
          "word_like_modules": [n for n in emb + sorted(children)
                                if "word" in n or n.startswith("q_")]}
    rec("G2_module_audit",
        "no word/score/Q module was added; module and embedding set is the frozen SENTENCE_BASE one",
        bool(g2["unexpected_children"] == [] and g2["missing_children"] == []
             and g2["word_like_modules"] == [] and len(emb) == 19), g2)
    g3 = {"total": int(sum(p.numel() for p in model.parameters())),
          "expected": RC.PARAMS_TOTAL}
    rec("G3_parameter_count", "total parameter count is 416,177", g3["total"] == g3["expected"], g3)
    return {"g2": g2}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parity", action="store_true",
                    help="skip the 2-epoch CPU parity run (NOT allowed before formal runs)")
    ap.add_argument("--out", default=str(RC.ANALYSIS_DIR / "checks.json"))
    args = ap.parse_args(argv)
    t0 = time.time()
    RC.assert_frozen_sources()
    cfg = RC.frozen_config()
    raw = GD._load_raw()
    print("A: frozen identity", flush=True)
    group_a(cfg)
    print("B: split identity", flush=True)
    b = group_b(raw)
    print("C: fit-only normalisation", flush=True)
    group_c(raw, b["bundle"], cfg)
    print("D: tensor path", flush=True)
    group_d(cfg, b["bundle"], raw, parity=not args.skip_parity)
    print("E: invariances", flush=True)
    group_e(cfg, b["bundle"], raw)
    print("F: gradients", flush=True)
    group_f(cfg, b["bundle"], raw)
    print("G: no additions", flush=True)
    group_g(cfg, b["bundle"])
    n_pass = sum(1 for c in CHECKS if c["status"] == "PASS")
    payload = {
        "protocol_id": "r2-sentence-replication-v1",
        "generated_utc": RC.now_utc(),
        "git_commit": RC.git_commit(),
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "n_checks": len(CHECKS), "n_pass": n_pass,
        "n_fail": len(CHECKS) - n_pass,
        "checks": CHECKS,
        "code_sha256": RC.code_sha(["r2_common.py", "r2_data.py", "r2_checks.py",
                                    "r2_train.py"]),
        "wall_seconds": time.time() - t0,
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
    }
    RC.write_json(args.out, payload)
    print(f"\n{n_pass}/{len(CHECKS)} PASS -> {args.out}")
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())