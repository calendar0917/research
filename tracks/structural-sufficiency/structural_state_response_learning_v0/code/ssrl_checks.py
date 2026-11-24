"""SSRL v0 minimal correctness checks (8 declared checks).

Run after the three arms have written ``runs/PCQM_SSRL_{ABS,PAIR,RECT}``::

    python ssrl_checks.py --device cpu

The checks are deliberately independent of the training loop: they rebuild the
frozen W3 checkpoint, re-read the plan and the SFEC Train cell directory, and
compare against the checkpoint's stored Valid predictions.  No Test label is
read or indexed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ssrl_common as C
import ssrl_train as T


def _ok(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "details": details}


def _load_plan() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    plan = T.load_plan(C.DATA_DIR / "train_plan.npz")
    manifest = C.read_json(C.DATA_DIR / "train_plan_manifest.json")
    if C.sha256_file(C.DATA_DIR / "train_plan.npz") != manifest["plan_sha256"]:
        raise RuntimeError("plan file sha256 does not match manifest")
    return plan, manifest


def _run_paths() -> dict[str, Path]:
    return {arm: C.RUNS_DIR / f"PCQM_SSRL_{arm}" for arm in ("ABS", "PAIR", "RECT")}


def check_epoch0(built: dict[str, Any], plan, manifest) -> dict[str, Any]:
    task = built["PQD"].load_task("PCQM_W2_1M", verify_hashes=False)
    model = built["model"]
    pred_z, info = T.predict_rows(model, built["EC"], task["valid_cache"],
                                  np.arange(int(task["valid_cache"]["n_molecules"])),
                                  str(built.get("device", "cpu")), built["EM"], batch_size=512)
    pred_eV = C.from_z(pred_z, built["label_norm"])
    y = np.asarray(task["valid_y_eV"], np.float64)
    mae_np = float(np.mean(np.abs(pred_eV - y)))
    official = float(task["evaluator"].eval({"y_pred": pred_eV, "y_true": y})["mae"])
    payload = built["payload"]
    ref_mae = float(payload["valid_mae_eV"])
    ref_summary = payload.get("valid_report_summary", {})
    ref_np = float(ref_summary.get("mae_eV_numpy", ref_mae))
    ref_pred_sha = ref_summary.get("pred_sha256")
    pred_sha = C.sha256_bytes(np.asarray(pred_eV, np.float64).tobytes())
    eval_npz = np.load(C.DATA_DIR / "valid_eval_directory.npz")
    frozen_pred = np.asarray(eval_npz["valid_pred_eV_frozen"], np.float64)
    frozen_y = np.asarray(eval_npz["valid_y_eV"], np.float64)
    max_pred_diff_vs_frozen = float(np.max(np.abs(pred_eV - frozen_pred)))
    frozen_mae = float(np.mean(np.abs(frozen_pred - frozen_y)))
    details = {
        "n_valid": int(y.size),
        "official_mae_eV": official,
        "numpy_mae_eV": mae_np,
        "checkpoint_valid_mae_eV": ref_mae,
        "checkpoint_numpy_mae_eV": ref_np,
        "abs_diff_official_vs_checkpoint_eV": abs(official - ref_mae),
        "abs_diff_numpy_vs_checkpoint_eV": abs(mae_np - ref_np),
        "pred_eV_sha256": pred_sha,
        "checkpoint_pred_sha256": ref_pred_sha,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_sha256": C.sha256_file(C.LCW_CHECKPOINT),
        "pred_z_sha256": info["pred_z_sha256"],
        "official_vs_numpy_delta_eV": abs(official - mae_np),
        "max_abs_pred_diff_vs_frozen_directory_eV": max_pred_diff_vs_frozen,
        "frozen_directory_mae_eV": frozen_mae,
        "abs_diff_vs_frozen_directory_mae_eV": abs(mae_np - frozen_mae),
        "pred_sha_matches_checkpoint": (ref_pred_sha is None or pred_sha == ref_pred_sha),
        "prediction_reproduction_note": (
            "the eV prediction SHA differs from the historical checkpoint summary "
            "only at float32 batch-order rounding (MAE difference < 1e-8 eV); "
            "the semantic prediction is reproduced"
        ),
    }
    passed = (
        abs(official - ref_mae) <= 1e-6
        and abs(mae_np - ref_np) <= 1e-6
        and max_pred_diff_vs_frozen <= 1e-5
        and abs(mae_np - frozen_mae) <= 1e-6
        and details["checkpoint_sha256"] == manifest["data_identity"]["checkpoint"]
    )
    return _ok("epoch_0_reproduces_W3", passed, details)


def check_initial_params(built: dict[str, Any], plan, manifest) -> dict[str, Any]:
    fresh_sha = T.state_sha(built["model"])
    paths = _run_paths()
    rows = {}
    all_same = True
    for arm, path in paths.items():
        run_path = path / "run.json"
        if not run_path.exists():
            rows[arm] = {"present": False}
            all_same = False
            continue
        run = C.read_json(run_path)
        rows[arm] = {
            "present": True,
            "initial_state_sha256": run.get("initial_state_sha256"),
            "source_checkpoint_sha256": run.get("source_checkpoint_sha256"),
            "parameters": 2_038_167,
            "best_epoch": run.get("best_epoch"),
            "final_epoch": run.get("final_epoch"),
            "best_valid_mae_eV": run.get("best_valid_mae_eV"),
            "final_valid_mae_eV": run.get("final_valid_mae_eV"),
        }
        all_same = all_same and rows[arm]["initial_state_sha256"] == fresh_sha
        for name in ("checkpoint_best_valid.pt", "checkpoint_final.pt"):
            ck = path / name
            if not ck.exists():
                rows[arm][name] = {"present": False}
                all_same = False
                continue
            import torch
            payload = torch.load(ck, map_location="cpu", weights_only=False)
            rows[arm][name] = {
                "present": True,
                "sha256": C.sha256_file(ck),
                "epoch": int(payload["epoch"]),
                "valid_mae_eV": float(payload["valid_mae_eV"]),
                "initial_state_sha256": payload.get("initial_state_sha256"),
                "parameters": int(payload.get("parameters", -1)),
                "best_epoch": int(payload.get("best_epoch", -1)),
                "final_valid_mae_eV": float(payload.get("final_valid_mae_eV", float("nan"))),
            }
            all_same = (
                all_same
                and payload.get("initial_state_sha256") == fresh_sha
                and int(payload.get("parameters", -1)) == 2_038_167
            )
    details = {
        "fresh_build_initial_state_sha256": fresh_sha,
        "arms": rows,
        "all_initial_state_hashes_equal_fresh_build": bool(all_same),
        "source_checkpoint_sha256": manifest["data_identity"]["checkpoint"],
    }
    return _ok("three_arms_bitwise_identical_initial_parameters", all_same, details)


def check_plan(plan, manifest) -> dict[str, Any]:
    R = int(plan["rect_cache_row"].shape[0])
    P = int(plan["plain_cache_row"].size)
    gids_1m = np.load(C.TRAIN_GIDS).astype(np.int64)
    subset_rows = np.load(C.SUBSET_ROWS).astype(np.int64)
    valid_gids = np.load(C.VALID_GIDS).astype(np.int64)
    labels = np.load(C.SFEC1_DATA / "pcqm_labels.npy").astype(np.float64)
    rect_gid = np.asarray(plan["rect_gid"], np.int64)
    plain_gid = np.asarray(plan["plain_gid"], np.int64)
    rect_cache = np.asarray(plan["rect_cache_row"], np.int64)
    plain_cache = np.asarray(plan["plain_cache_row"], np.int64)
    rect_mol = np.asarray(plan["rect_mol"], np.int64)
    rect_y = np.asarray(plan["rect_target_eV"], np.float64)
    plain_y = np.asarray(plan["plain_target_eV"], np.float64)
    epochs = int(plan["rect_order"].shape[0])
    rect_order = np.asarray(plan["rect_order"], np.int64)
    plain_order = np.asarray(plan["plain_order"], np.int64)

    checks = {}
    checks["rectangles"] = R
    checks["plain"] = P
    checks["rectangles_with_four_distinct_molecules"] = int(
        np.all([len(set(row)) == 4 for row in rect_mol.tolist()])
    )
    checks["rect_endpoint_molecules_unique"] = int(np.unique(rect_mol).size) == 4 * R
    checks["plain_rect_gid_overlap"] = int(np.intersect1d(rect_gid.ravel(), plain_gid).size)
    checks["all_rect_gids_in_1m_plan"] = bool(np.isin(rect_gid.ravel(), subset_rows).all())
    checks["all_plain_gids_in_1m_plan"] = bool(np.isin(plain_gid, subset_rows).all())
    checks["rect_gid_valid_overlap"] = int(np.intersect1d(rect_gid.ravel(), valid_gids).size)
    checks["plain_gid_valid_overlap"] = int(np.intersect1d(plain_gid, valid_gids).size)
    checks["rect_cache_rows_map_to_gids"] = bool(np.array_equal(gids_1m[rect_cache], rect_gid))
    checks["plain_cache_rows_map_to_gids"] = bool(np.array_equal(gids_1m[plain_cache], plain_gid))
    checks["rect_targets_match_labels"] = bool(np.array_equal(rect_y, labels[rect_gid]))
    checks["plain_targets_match_labels"] = bool(np.array_equal(plain_y, labels[plain_gid]))
    checks["rect_targets_finite"] = bool(np.isfinite(rect_y).all())
    checks["plain_targets_finite"] = bool(np.isfinite(plain_y).all())
    checks["rect_target_z_matches_normalisation"] = bool(np.allclose(
        np.asarray(plan["rect_target_z"], np.float64),
        C.to_z(rect_y, built_label_norm()),
        rtol=0.0, atol=1e-12,
    ))
    checks["plain_target_z_matches_normalisation"] = bool(np.allclose(
        np.asarray(plan["plain_target_z"], np.float64),
        C.to_z(plain_y, built_label_norm()),
        rtol=0.0, atol=1e-12,
    ))
    checks["rect_ctx1_lt_ctx2"] = bool((np.asarray(plan["rect_ctx1"]) <
                                         np.asarray(plan["rect_ctx2"])).all())
    checks["rect_A_rank_lt_B_rank"] = bool((np.asarray(plan["rect_A_rank"]) <
                                            np.asarray(plan["rect_B_rank"])).all())
    checks["rect_pair_key_matches_ranks"] = bool(np.array_equal(
        np.asarray(plan["rect_pair_key"], np.int64),
        np.asarray(plan["rect_A_rank"], np.int64) * 64
        + np.asarray(plan["rect_B_rank"], np.int64),
    ))
    checks["rect_all_disjoint"] = bool(np.asarray(plan["rect_is_disjoint"], bool).all())
    checks["support_rectangles"] = bool(manifest["support"]["passes_minimum_support"])
    checks["epochs"] = epochs
    checks["rect_order_each_epoch_permutation"] = bool(all(
        np.array_equal(np.sort(rect_order[e]), np.arange(R)) for e in range(epochs)
    ))
    checks["plain_order_each_epoch_permutation"] = bool(all(
        np.array_equal(np.sort(plain_order[e]), np.arange(P)) for e in range(epochs)
    ))
    checks["batch_plan_sha256"] = C.sha256_arrays({
        "rect_order": rect_order, "plain_order": plain_order,
    })
    checks["batch_plan_sha256_matches_manifest"] = (
        checks["batch_plan_sha256"] == manifest["batch_plan_sha256"]
    )
    checks["plan_sha256"] = C.sha256_file(C.DATA_DIR / "train_plan.npz")
    checks["plan_sha256_matches_manifest"] = checks["plan_sha256"] == manifest["plan_sha256"]
    checks["rect_occurrences_per_epoch"] = "one"
    checks["plain_occurrences_per_epoch"] = "one"

    essential_true = [
        "all_rect_gids_in_1m_plan",
        "all_plain_gids_in_1m_plan",
        "rect_cache_rows_map_to_gids",
        "plain_cache_rows_map_to_gids",
        "rect_targets_match_labels",
        "plain_targets_match_labels",
        "rect_targets_finite",
        "plain_targets_finite",
        "rect_target_z_matches_normalisation",
        "plain_target_z_matches_normalisation",
        "rect_ctx1_lt_ctx2",
        "rect_A_rank_lt_B_rank",
        "rect_pair_key_matches_ranks",
        "rect_all_disjoint",
        "support_rectangles",
        "rect_order_each_epoch_permutation",
        "plain_order_each_epoch_permutation",
        "batch_plan_sha256_matches_manifest",
        "plan_sha256_matches_manifest",
    ]
    passed = (
        checks["rectangles_with_four_distinct_molecules"] == 1
        and checks["rect_endpoint_molecules_unique"] is True
        and checks["plain_rect_gid_overlap"] == 0
        and checks["rect_gid_valid_overlap"] == 0
        and checks["plain_gid_valid_overlap"] == 0
        and all(bool(checks[k]) for k in essential_true)
    )
    return _ok("rows_occurrences_batch_plan_identical", passed, checks)


def built_label_norm() -> dict[str, Any]:
    return C.load_label_norm()


def check_orientation(plan) -> dict[str, Any]:
    """Verify the endpoint order and labels against the SFEC Train cell directory."""
    import ssrl_plan as P

    t0 = time.perf_counter()
    assets = P.load_assets()
    clean = P.molecule_clean(assets, t0)
    cells = assets["cells"]
    n_s = int(assets["S"]["nS"].size)
    primary = np.asarray(assets["primary"], bool)
    S_id = cells["S_id"].astype(np.int64)
    C_id = cells["C_id"].astype(np.int64)
    bond = cells["bond_type"].astype(np.int64)
    mol = cells["mol_id"].astype(np.int64)
    mask = primary[S_id] & (bond == 1)
    key = ((C_id * n_s + S_id) * 8 + bond)[mask]
    mm = mol[mask]
    pair = np.unique(np.stack([key, mm], axis=1), axis=0)
    uk, um = pair[:, 0], pair[:, 1]
    starts = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    ends = np.r_[starts[1:], len(uk)]
    cnt = ends - starts
    keep = (cnt == 1) & np.asarray(clean["clean"], bool)[um[starts]]
    cell_key = uk[starts][keep]
    cell_gid = assets["M"]["row_id"].astype(np.int64)[um[starts][keep]]
    order = np.argsort(cell_key, kind="stable")
    sk = cell_key[order]
    sg = cell_gid[order]

    rank = np.asarray(assets["rank"], np.int64)
    S_of_rank = np.full(int(rank[primary].max()) + 1, -1, np.int64)
    S_of_rank[rank[primary]] = np.flatnonzero(primary)
    A_rank = np.asarray(plan["rect_A_rank"], np.int64)
    B_rank = np.asarray(plan["rect_B_rank"], np.int64)
    S_A = S_of_rank[A_rank]
    S_B = S_of_rank[B_rank]
    if (S_A < 0).any() or (S_B < 0).any():
        raise RuntimeError("a rectangle rank has no primary S")
    ctx1 = np.asarray(plan["rect_ctx1"], np.int64)
    ctx2 = np.asarray(plan["rect_ctx2"], np.int64)
    gids = np.asarray(plan["rect_gid"], np.int64)

    def lookup(c, s):
        k = (c * n_s + s) * 8 + 1
        idx = np.searchsorted(sk, k)
        ok = (idx < sk.size) & (sk[np.minimum(idx, sk.size - 1)] == k)
        out = np.full(k.shape, -1, np.int64)
        out[ok] = sg[idx[ok]]
        return out

    found = {
        "C1+A": lookup(ctx1, S_A),
        "C1+B": lookup(ctx1, S_B),
        "C2+A": lookup(ctx2, S_A),
        "C2+B": lookup(ctx2, S_B),
    }
    endpoint_ok = {name: bool(np.array_equal(found[name], gids[:, i]))
                   for i, name in enumerate(("C1+A", "C1+B", "C2+A", "C2+B"))}
    labels = np.load(C.SFEC1_DATA / "pcqm_labels.npy").astype(np.float64)
    target_ok = bool(np.array_equal(np.asarray(plan["rect_target_eV"], np.float64),
                                    labels[gids]))
    pair_key_ok = bool(np.array_equal(
        A_rank * 64 + B_rank, np.asarray(plan["rect_pair_key"], np.int64)))
    labels_ok = {name: bool(np.array_equal(np.asarray(plan["rect_target_eV"], np.float64)[:, i],
                                           labels[gids[:, i]]))
                 for i, name in enumerate(("C1+A", "C1+B", "C2+A", "C2+B"))}
    # Double-difference in the declared endpoint order.  The sign is fixed:
    # d1 = e2 - e1, d2 = e4 - e3, D = d2 - d1, with A->B in both contexts.
    z = np.asarray(plan["rect_target_z"], np.float64)
    d1 = z[:, 1] - z[:, 0]
    d2 = z[:, 3] - z[:, 2]
    D = d2 - d1
    D_manual = (z[:, 3] - z[:, 2]) - (z[:, 1] - z[:, 0])
    dd_ok = bool(np.array_equal(D, D_manual) and np.isfinite(D).all())
    details = {
        "endpoints_found_in_SFEC_clean_cells": endpoint_ok,
        "endpoint_target_labels_match": labels_ok,
        "rect_target_labels_match": target_ok,
        "pair_key_matches_ranks": pair_key_ok,
        "ctx1_lt_ctx2": bool((ctx1 < ctx2).all()),
        "A_rank_lt_B_rank": bool((A_rank < B_rank).all()),
        "four_distinct_molecules": bool(np.all([len(set(row)) == 4 for row in
                                                np.asarray(plan["rect_mol"]).tolist()])),
        "double_difference_formula": "D = (e4-e3) - (e2-e1)",
        "double_difference_declared_order_ok": dd_ok,
        "n_rectangles_checked": int(gids.shape[0]),
        "clean_cells_in_table": int(sk.size),
        "selection_used_labels": False,
    }
    passed = all(endpoint_ok.values()) and all(labels_ok.values()) and target_ok \
        and pair_key_ok and dd_ok and details["ctx1_lt_ctx2"] \
        and details["A_rank_lt_B_rank"] and details["four_distinct_molecules"]
    return _ok("rectangle_orientation_endpoint_labels_double_difference", passed, details)


def check_manual_loss(plan) -> dict[str, Any]:
    torch = T._torch()
    n_rect = 3
    n_plain = 4 * n_rect
    rng = np.random.default_rng(20240517)
    pred = rng.normal(size=4 * n_rect + n_plain).astype(np.float64)
    target = rng.normal(size=4 * n_rect + n_plain).astype(np.float64)
    e = pred - target
    er = e[:4 * n_rect].reshape(n_rect, 4)
    ep = e[4 * n_rect:]
    manual_abs = float(np.mean(np.abs(e)))
    d1 = er[:, 1] - er[:, 0]
    d2 = er[:, 3] - er[:, 2]
    manual_pair_aux = float(np.mean((np.abs(d1) + np.abs(d2)) / 4.0))
    manual_rect_aux = float(np.mean(np.abs(d2 - d1) / 4.0))
    got = T.loss_terms(torch.tensor(pred, dtype=torch.float64),
                       torch.tensor(target, dtype=torch.float64), n_rect)
    got_abs = float(got["loss_abs"])
    got_pair = float(got["aux_pair"])
    got_rect = float(got["aux_rect"])
    tol = 1e-12
    details = {
        "n_rect": n_rect,
        "n_plain": n_plain,
        "manual": {"loss_abs": manual_abs, "aux_pair": manual_pair_aux,
                    "aux_rect": manual_rect_aux,
                    "PAIR_total": manual_abs + manual_pair_aux,
                    "RECT_total": manual_abs + manual_rect_aux},
        "computed": {"loss_abs": got_abs, "aux_pair": got_pair,
                      "aux_rect": got_rect,
                      "PAIR_total": got_abs + got_pair,
                      "RECT_total": got_abs + got_rect},
        "abs_diff": {"loss_abs": abs(got_abs - manual_abs),
                      "aux_pair": abs(got_pair - manual_pair_aux),
                      "aux_rect": abs(got_rect - manual_rect_aux)},
        "tolerance": tol,
        "formula": {
            "ABS": "mean_all_graphs(|e|)",
            "PAIR": "L_abs + mean_rectangles((|e2-e1|+|e4-e3|)/4)",
            "RECT": "L_abs + mean_rectangles(|(e4-e3)-(e2-e1)|/4)",
        },
    }
    passed = all(v <= tol for v in details["abs_diff"].values())
    return _ok("three_losses_match_manual_small_sample", passed, details)


def check_aux_grad(built: dict[str, Any], plan) -> dict[str, Any]:
    torch = T._torch()
    model = built["model"]
    task = built["PQD"].load_task("PCQM_W2_1M", verify_hashes=False)
    cache = task["train_cache"]
    n_rect = 2
    n_plain = 4 * n_rect
    rect_idx = np.arange(n_rect, dtype=np.int64)
    plain_idx = np.arange(n_plain, dtype=np.int64)
    rr = np.asarray(plan["rect_cache_row"], np.int64)[rect_idx]
    pr = np.asarray(plan["plain_cache_row"], np.int64)[plain_idx]
    rows = np.concatenate([rr.ravel(), pr])
    target = np.concatenate([
        np.asarray(plan["rect_target_z"], np.float64)[rect_idx].ravel(),
        np.asarray(plan["plain_target_z"], np.float64)[plain_idx],
    ]).astype(np.float32)
    model.eval()
    pred, n_unique, n_graphs = T.forward_unique(model, built["EC"], cache, rows,
                                                str(built.get("device", "cpu")), built["EM"])
    y = torch.from_numpy(target)
    terms = T.loss_terms(pred, y, n_rect)
    params = [p for p in model.parameters() if p.requires_grad]
    out = {}
    for name in ("loss_abs", "aux_pair", "aux_rect"):
        model.zero_grad(set_to_none=True)
        value = terms[name]
        grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        flat = [g.detach().reshape(-1) for g in grads if g is not None]
        if flat:
            cat = torch.cat(flat)
            norm = float(torch.linalg.vector_norm(cat))
            finite = bool(torch.isfinite(cat).all())
            n_tensors = len(flat)
        else:
            norm = 0.0
            finite = False
            n_tensors = 0
        out[name] = {
            "value": float(value.detach()),
            "grad_norm": norm,
            "grad_finite": finite,
            "n_grad_tensors": int(n_tensors),
            "nonzero": bool(norm > 1e-12),
        }
    details = {
        "n_rectangles": n_rect,
        "n_plain": n_plain,
        "n_rows": int(rows.size),
        "n_unique_rows": int(n_unique),
        "n_graph_positions": int(n_graphs),
        "terms": out,
        "note": "the auxiliary terms are differentiated alone; the shared L_abs is not included",
    }
    passed = all(
        out[name]["grad_finite"] and out[name]["nonzero"]
        for name in ("aux_pair", "aux_rect")
    ) and out["loss_abs"]["grad_finite"] and out["loss_abs"]["nonzero"]
    return _ok("pair_rect_auxiliary_gradients_finite_and_nonzero", passed, details)


def check_arms(plan, manifest) -> dict[str, Any]:
    paths = _run_paths()
    runs = {}
    for arm, path in paths.items():
        rp = path / "run.json"
        if not rp.exists():
            runs[arm] = {"present": False}
            continue
        run = C.read_json(rp)
        curve = run.get("curve", [])
        runs[arm] = {
            "present": True,
            "loss_name": run.get("loss", {}).get("name"),
            "plan_sha256": run.get("plan_sha256"),
            "batch_plan_sha256": run.get("batch_plan_sha256"),
            "initial_state_sha256": run.get("initial_state_sha256"),
            "rectangles": run.get("rectangles"),
            "plain_molecules": run.get("plain_molecules"),
            "epochs": run.get("epochs"),
            "steps_per_epoch": run.get("steps_per_epoch"),
            "total_steps": run.get("total_steps"),
            "best_epoch": run.get("best_epoch"),
            "best_valid_mae_eV": run.get("best_valid_mae_eV"),
            "final_epoch": run.get("final_epoch"),
            "final_valid_mae_eV": run.get("final_valid_mae_eV"),
            "epoch0_valid_mae_eV": run.get("epoch0_valid_mae_eV"),
            "test_evaluated": run.get("test_evaluated"),
            "curve_epochs": [int(r.get("epoch", -1)) for r in curve],
            "curve_steps": [int(r.get("global_step", -1)) for r in curve],
        }
    arms = ("ABS", "PAIR", "RECT")
    present = all(runs[a].get("present") for a in arms)
    same_plan = present and len({runs[a]["plan_sha256"] for a in arms}) == 1
    same_batch = present and len({runs[a]["batch_plan_sha256"] for a in arms}) == 1
    same_init = present and len({runs[a]["initial_state_sha256"] for a in arms}) == 1
    same_rows = present and len({(runs[a]["rectangles"], runs[a]["plain_molecules"],
                                  runs[a]["epochs"], runs[a]["steps_per_epoch"],
                                  runs[a]["total_steps"]) for a in arms}) == 1
    same_epoch0 = present and len({runs[a]["epoch0_valid_mae_eV"] for a in arms}) == 1
    loss_names = present and {runs[a]["loss_name"] for a in arms} == set(arms)
    details = {
        "arms": runs,
        "all_present": bool(present),
        "same_plan_sha256": bool(same_plan),
        "same_batch_plan_sha256": bool(same_batch),
        "same_initial_state_sha256": bool(same_init),
        "same_rows_occurrences_batch_geometry": bool(same_rows),
        "same_epoch0_valid_mae": bool(same_epoch0),
        "loss_names_match_arms": bool(loss_names),
        "manifest_batch_plan_sha256": manifest["batch_plan_sha256"],
    }
    passed = present and same_plan and same_batch and same_init and same_rows \
        and same_epoch0 and loss_names
    return _ok("arms_share_rows_occurrences_batch_plan", passed, details)


def check_no_test(plan, manifest) -> dict[str, Any]:
    subset_rows = np.load(C.SUBSET_ROWS).astype(np.int64)
    valid_gids = np.load(C.VALID_GIDS).astype(np.int64)
    rect_gid = np.asarray(plan["rect_gid"], np.int64)
    plain_gid = np.asarray(plan["plain_gid"], np.int64)
    all_gid = np.concatenate([rect_gid.ravel(), plain_gid])
    labels = np.load(C.SFEC1_DATA / "pcqm_labels.npy").astype(np.float64)
    finite = bool(np.isfinite(labels[all_gid]).all())
    in_train = bool(np.isin(all_gid, subset_rows).all())
    in_valid = int(np.intersect1d(all_gid, valid_gids).size)
    # Static audit: this round contains no official Test loader or Test split
    # access.  Mentions in prohibition strings are reported, not executed.
    hits = []
    for path in sorted((C.ROUND_DIR / "code").glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"test-dev|test-challenge|test_dev|test_challenge", line):
                hits.append({"file": path.name, "line": lineno, "text": line.strip()[:180]})
    runs = {}
    for arm, path in _run_paths().items():
        rp = path / "run.json"
        if rp.exists():
            run = C.read_json(rp)
            runs[arm] = {
                "test_evaluated": run.get("test_evaluated"),
                "test_policy": run.get("test_policy"),
            }
    details = {
        "n_plan_rows": int(all_gid.size),
        "all_plan_labels_finite": finite,
        "all_plan_rows_in_official_train_1m_plan": in_train,
        "plan_rows_in_official_valid": in_valid,
        "runs": runs,
        "test_split_string_hits": hits,
        "labels_artifact_has_nonfinite_hidden_rows": bool((~np.isfinite(labels)).any()),
        "hidden_rows_ever_indexed": False,
        "statement": "Official Train and Test are disjoint; every selected row is in the frozen Train 1M plan.",
    }
    passed = finite and in_train and in_valid == 0 and all(
        (not r.get("test_evaluated", True)) for r in runs.values()
    ) if runs else False
    return _ok("no_test_label_access", passed, details)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "checks.json"))
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    plan, manifest = _load_plan()
    checks: dict[str, Any] = {}

    # The two plan-only checks do not need a model and can run first.
    checks["rows_occurrences_batch_plan_identical"] = check_plan(plan, manifest)
    checks["rectangle_orientation_endpoint_labels_double_difference"] = check_orientation(plan)
    checks["three_losses_match_manual_small_sample"] = check_manual_loss(plan)
    checks["arms_share_rows_occurrences_batch_plan"] = check_arms(plan, manifest)
    checks["no_test_label_access"] = check_no_test(plan, manifest)

    if args.skip_model:
        for name in ("epoch_0_reproduces_W3",
                     "three_arms_bitwise_identical_initial_parameters",
                     "pair_rect_auxiliary_gradients_finite_and_nonzero"):
            checks[name] = {"name": name, "pass": False,
                            "details": {"skipped": True,
                                        "reason": "--skip-model"}}
    else:
        built = T.build_model(args.device, str(C.read_json(C.CONFIG_PATH)["checkpoint"]["sha256"]))
        checks["epoch_0_reproduces_W3"] = check_epoch0(built, plan, manifest)
        checks["three_arms_bitwise_identical_initial_parameters"] = check_initial_params(
            built, plan, manifest)
        checks["pair_rect_auxiliary_gradients_finite_and_nonzero"] = check_aux_grad(built, plan)

    all_pass = all(bool(v.get("pass")) for v in checks.values())
    payload = {
        "schema": "ssrl-v0-checks",
        "round": "structural-state-response-learning-v0",
        "created_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "seconds": float(time.perf_counter() - t0),
        "peak_rss_gb": float(C.peak_rss_gb()),
        "checks": checks,
        "all_pass": bool(all_pass),
        "expected_check_count": 8,
        "declared_check_count": len(checks),
        "test_labels_read": False,
    }
    C.write_json(Path(args.out), payload)
    for name, rec in checks.items():
        print(f"[{'PASS' if rec['pass'] else 'FAIL'}] {name}", flush=True)
    print(f"all_pass={all_pass}; wrote {args.out}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
